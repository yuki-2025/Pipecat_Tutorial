"""
Step 11 — Observer 系统（非侵入式监控）
=========================================
Observer 让你监控 pipeline 里所有 frame 的流动，不需要插入处理器，不影响数据流。

Observer vs FrameProcessor 的根本区别：
    FrameProcessor = IN pipeline，frame 必须经过，可以修改/过滤
    BaseObserver   = OUTSIDE pipeline，被动旁观所有 frame，不影响数据

你会学到：
    1. 内建 Observer：LLMLogObserver, TranscriptionLogObserver, MetricsLogObserver
    2. BaseObserver 的三个 hook：on_push_frame, on_process_frame, on_pipeline_started
    3. FramePushed dataclass：source, destination, frame, direction, timestamp
    4. MetricsFrame 里有哪些数据：TTFB, processing time, token usage, TTS chars
    5. 写自定义 Observer：实时收集 session 统计并在结束时汇总打印
    6. 观察 BotStartedSpeakingFrame / BotStoppedSpeakingFrame 来测量 bot 说话时长

Pipeline 结构和 step2 完全相同，只在 PipelineTask 里加 observers：
    transport.input() → stt → user_aggregator → llm → tts → transport.output() → assistant_aggregator

关键：enable_metrics=True 和 enable_usage_metrics=True 必须开启，
      否则 MetricsFrame 不会发出，MetricsLogObserver 收不到数据。

安装依赖：（和 step2 一样）
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]" python-dotenv loguru

所需 API key：DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)

# ── Observer imports ──────────────────────────────────────────────────────
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()

# 把 loguru 设为 DEBUG，才能看到内建 observer 的输出
logger.remove(0)
logger.add(sys.stderr, level="DEBUG", format="<level>{level}</level> | {message}")


# ═══════════════════════════════════════════════════════════════════════════
# 自定义 Observer：Session 统计收集器
#
# 这个 observer 示范了：
# - 如何在 on_push_frame 里筛选特定 frame 类型
# - 如何追踪有状态的数据（bot 说话开始时间、累计指标）
# - 如何在 on_pipeline_started 里做初始化
# - 如何在 session 结束后汇总打印报告
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionStats:
    """一次完整对话 session 的统计数据"""
    start_time: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    turns: int = 0                      # 用户说话的轮次
    transcriptions: list[str] = field(default_factory=list)  # 所有转录文字

    total_prompt_tokens: int = 0        # LLM 总 prompt token
    total_completion_tokens: int = 0    # LLM 总 completion token
    total_tts_chars: int = 0            # TTS 消耗总字符数

    ttfb_values: list[float] = field(default_factory=list)  # 所有 TTFB（秒）
    processing_times: dict = field(default_factory=lambda: defaultdict(list))

    bot_speaking_durations: list[float] = field(default_factory=list)
    _bot_speaking_start: float | None = None  # bot 开始说话的时间戳

    def record_bot_started(self, timestamp_ns: int):
        self._bot_speaking_start = timestamp_ns / 1e9

    def record_bot_stopped(self, timestamp_ns: int):
        if self._bot_speaking_start is not None:
            duration = timestamp_ns / 1e9 - self._bot_speaking_start
            self.bot_speaking_durations.append(duration)
            self._bot_speaking_start = None

    def summary(self) -> str:
        elapsed = asyncio.get_event_loop().time() - self.start_time
        avg_ttfb = sum(self.ttfb_values) / len(self.ttfb_values) if self.ttfb_values else 0
        avg_bot_speaking = (
            sum(self.bot_speaking_durations) / len(self.bot_speaking_durations)
            if self.bot_speaking_durations else 0
        )

        lines = [
            "",
            "═" * 55,
            " Session Summary",
            "═" * 55,
            f" Duration        : {elapsed:.1f}s",
            f" Conversation turns: {self.turns}",
            "",
            " STT Transcriptions:",
        ]
        for i, t in enumerate(self.transcriptions, 1):
            lines.append(f"   {i}. {t!r}")

        lines += [
            "",
            " LLM Usage:",
            f"   Prompt tokens   : {self.total_prompt_tokens}",
            f"   Completion tokens: {self.total_completion_tokens}",
            f"   Total tokens    : {self.total_prompt_tokens + self.total_completion_tokens}",
            "",
            " TTS Usage:",
            f"   Total characters: {self.total_tts_chars}",
            "",
            " Latency:",
            f"   Avg TTFB        : {avg_ttfb * 1000:.0f}ms  (across {len(self.ttfb_values)} measurements)",
            f"   Avg bot speaking: {avg_bot_speaking:.1f}s per turn",
            "═" * 55,
        ]
        return "\n".join(lines)


class SessionStatsObserver(BaseObserver):
    """
    收集整个对话 session 的统计数据。

    on_push_frame 会被 pipeline 里每一次 frame 传递都调用到。
    data.source    = 推送这个 frame 的 processor
    data.destination = 接收这个 frame 的 processor
    data.frame     = frame 本身
    data.direction = DOWNSTREAM 或 UPSTREAM
    data.timestamp = 纳秒时间戳（pipeline 内部时钟）
    """

    def __init__(self):
        super().__init__()
        self._stats = SessionStats()

    async def on_pipeline_started(self):
        """Pipeline 完全启动后调用（StartFrame 经过所有 processor 后）"""
        print("\n[Observer] Pipeline started — session tracking begins\n")

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        ts = data.timestamp

        # ── STT 转录 ─────────────────────────────────────────────────────
        if isinstance(frame, TranscriptionFrame):
            self._stats.turns += 1
            self._stats.transcriptions.append(frame.text)

        # ── Bot 说话时长 ──────────────────────────────────────────────────
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._stats.record_bot_started(ts)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._stats.record_bot_stopped(ts)

        # ── Metrics Frame：性能指标 ───────────────────────────────────────
        # MetricsFrame 只在 enable_metrics=True 时才会出现
        # 需要先 import MetricsFrame 才能 isinstance 检查
        # 用 frame.__class__.__name__ 避免 import 噪音
        elif frame.__class__.__name__ == "MetricsFrame":
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    # TTFB = Time To First Byte，以秒为单位
                    self._stats.ttfb_values.append(d.value)

                elif isinstance(d, LLMUsageMetricsData):
                    self._stats.total_prompt_tokens += d.value.prompt_tokens
                    self._stats.total_completion_tokens += d.value.completion_tokens

                elif isinstance(d, TTSUsageMetricsData):
                    self._stats.total_tts_chars += d.value

                elif isinstance(d, ProcessingMetricsData):
                    service_name = str(data.source).split("#")[0]
                    self._stats.processing_times[service_name].append(d.value)

    def print_summary(self):
        print(self._stats.summary())


# ═══════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=1,
        )
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM"),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=(
                "You are a helpful assistant. Keep responses short and conversational."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # Pipeline 和 step2 完全一样，没有任何改动
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    # ── 实例化自定义 observer ─────────────────────────────────────────────
    stats_observer = SessionStatsObserver()

    # ── 所有 observer 都在这里注册，不插入 pipeline ──────────────────────
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,          # 必须开启 → MetricsFrame 才会发出
            enable_usage_metrics=True,    # 必须开启 → LLMUsageMetricsData / TTSUsageMetricsData
        ),
        observers=[
            # ── 内建 observers ─────────────────────────────────────────
            TranscriptionLogObserver(),   # 💬 打印每次 STT 转录结果
            LLMLogObserver(),             # 🧠 打印 LLM 生成的 token（流式）
            MetricsLogObserver(           # 📊 打印性能指标
                include_metrics={
                    TTFBMetricsData,       # 只看 TTFB
                    LLMUsageMetricsData,   # 和 token 用量
                }
            ),
            # ── 自定义 observer ────────────────────────────────────────
            stats_observer,               # 📈 收集 session 统计
        ],
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user briefly. Let them know they can talk to you.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Observer Demo")
    print(" 你会在终端看到：")
    print("   💬 STT 转录（TranscriptionLogObserver）")
    print("   🧠 LLM 生成 token（LLMLogObserver）")
    print("   📊 TTFB 和 token 用量（MetricsLogObserver）")
    print(" Ctrl+C 结束后会打印 Session Summary")
    print("=" * 55)

    try:
        await runner.run(task)
    finally:
        # Ctrl+C 或 pipeline 结束后打印汇总
        stats_observer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
