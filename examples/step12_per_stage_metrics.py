"""
Step 12 — 每个阶段的 Duration 和 Processing Time
=================================================
专注于用 Observer 测量每一个 pipeline 阶段的性能指标。

你会学到：
    1. MetricsData.processor — 每条指标都标注了来自哪个 service（Deepgram/OpenAI/ElevenLabs）
    2. 三种核心延迟指标的定义：
       - TTFB（Time To First Byte）：从请求发出到第一个输出的时间
       - Processing Time：整个 service 从开始到完成的总时间
       - Text Aggregation：从第一个 LLM token 到第一个完整句子的等待时间（TTS 的"等第一句"延迟）
    3. 用 BotStartedSpeakingFrame 时间戳计算端到端（E2E）延迟
    4. 每一轮对话结束后，打印该轮的完整 stage breakdown 表格

Pipeline 各阶段的指标含义：

  [STT - Deepgram]
    TTFB          = 音频进来 → 第一个转录字出来（网络 + 模型首字时间）
    ProcessingTime = 整段语音转录完毕的总时间

  [LLM - OpenAI]
    TTFB          = context 发送 → 第一个 token 出来（网络 + 模型首 token 时间）
    ProcessingTime = 整个 LLM 回复生成完毕的总时间（通常 = TTFB + 生成时间）

  [TTS - ElevenLabs]
    TTFB          = 第一句文字进来 → 第一段音频出来（合成首音节时间）
    ProcessingTime = 合成引擎的内部处理时间（通常很短，因为是流式的）
    TextAggregation = 第一个 LLM token 出来 → 凑够第一个完整句子（句子积累等待时间）

  E2E Latency    = 用户停止说话 → Bot 开始播音（这是用户实际感受到的延迟）
                   ≈ STT 收尾 + LLM TTFB + TTS TTFB + TextAggregation

所需 API key：DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")  # 只看我们自己打印的内容


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构：一轮对话的指标快照
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TurnMetrics:
    turn_number: int
    transcription: str = ""

    # 各 stage 的 TTFB（秒）
    stt_ttfb: float | None = None
    llm_ttfb: float | None = None
    tts_ttfb: float | None = None

    # 各 stage 的 processing time（秒）
    stt_processing: float | None = None
    llm_processing: float | None = None
    tts_processing: float | None = None

    # TTS 特有：从第一个 LLM token 到第一个完整句子的等待时间
    tts_text_aggregation: float | None = None

    # LLM token 用量
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0

    # TTS 字符数
    tts_chars: int = 0

    # E2E 延迟（用户停止说话 → bot 开始播音）
    user_stopped_ts: float | None = None    # 纳秒时间戳
    bot_started_ts: float | None = None     # 纳秒时间戳

    @property
    def e2e_latency_ms(self) -> float | None:
        if self.user_stopped_ts and self.bot_started_ts:
            return (self.bot_started_ts - self.user_stopped_ts) / 1e6  # ns → ms
        return None

    def print_table(self):
        def ms(v):
            return f"{v * 1000:6.0f}ms" if v is not None else "     —  "

        e2e = f"{self.e2e_latency_ms:.0f}ms" if self.e2e_latency_ms else "—"

        print(f"\n{'═' * 65}")
        print(f" Turn {self.turn_number}: \"{self.transcription}\"")
        print(f"{'─' * 65}")
        print(f" {'Stage':<18} {'TTFB':>10}  {'Processing':>12}  Notes")
        print(f"{'─' * 65}")

        # STT row
        print(
            f" {'STT (Deepgram)':<18} {ms(self.stt_ttfb):>10}  {ms(self.stt_processing):>12}"
        )

        # LLM row
        token_note = ""
        if self.llm_prompt_tokens:
            token_note = f"  p:{self.llm_prompt_tokens} c:{self.llm_completion_tokens}"
        print(
            f" {'LLM (OpenAI)':<18} {ms(self.llm_ttfb):>10}  {ms(self.llm_processing):>12}{token_note}"
        )

        # TTS row
        char_note = f"  {self.tts_chars}chars" if self.tts_chars else ""
        print(
            f" {'TTS (ElevenLabs)':<18} {ms(self.tts_ttfb):>10}  {ms(self.tts_processing):>12}{char_note}"
        )

        # Text aggregation（TTS 等第一句的时间）
        if self.tts_text_aggregation is not None:
            print(
                f" {'  └ text aggregation':<18} {'':>10}  {ms(self.tts_text_aggregation):>12}"
                f"  (wait for 1st sentence)"
            )

        print(f"{'─' * 65}")
        print(f" E2E Latency (user stop → bot speak): {e2e}")
        print(f"{'═' * 65}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-Stage Metrics Observer
# ═══════════════════════════════════════════════════════════════════════════

class PerStageMetricsObserver(BaseObserver):
    """
    每一轮对话结束时，打印该轮的 stage breakdown 表格。

    工作原理：
    1. 监听 MetricsFrame → 提取每个 service 的 TTFB / Processing 指标
       MetricsData.processor 字段标识来源（如 "OpenAILLMService#0"）
    2. 监听 UserStoppedSpeakingFrame → 记录用户停止说话的时间戳
    3. 监听 BotStartedSpeakingFrame  → 记录 bot 开始播音的时间戳，计算 E2E
    4. 监听 BotStoppedSpeakingFrame  → 一轮对话结束，打印表格，重置当前轮数据
    5. 监听 TranscriptionFrame       → 记录转录文字（用于表格标题）
    """

    def __init__(self):
        super().__init__()
        self._turn = 0
        self._current: TurnMetrics | None = None
        self._session_turns: list[TurnMetrics] = []

    def _ensure_current_turn(self):
        if self._current is None:
            self._turn += 1
            self._current = TurnMetrics(turn_number=self._turn)

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        ts = data.timestamp  # 纳秒

        # ── E2E 时间戳 ────────────────────────────────────────────────────
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._ensure_current_turn()
            self._current.user_stopped_ts = ts

        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._current and self._current.bot_started_ts is None:
                self._current.bot_started_ts = ts

        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Bot 说完话 = 这一轮结束，打印表格
            if self._current:
                self._session_turns.append(self._current)
                self._current.print_table()
                self._current = None

        # ── 转录文字 ──────────────────────────────────────────────────────
        elif isinstance(frame, TranscriptionFrame):
            self._ensure_current_turn()
            self._current.transcription = frame.text

        # ── MetricsFrame：从这里拿所有性能数据 ───────────────────────────
        elif isinstance(frame, MetricsFrame):
            self._ensure_current_turn()
            for d in frame.data:
                p = d.processor.lower()  # 转小写，方便 contains 判断

                # --- TTFB ---
                if isinstance(d, TTFBMetricsData):
                    if "deepgram" in p or "stt" in p:
                        self._current.stt_ttfb = d.value
                    elif "openai" in p or "llm" in p or "gpt" in p:
                        self._current.llm_ttfb = d.value
                    elif "elevenlabs" in p or "tts" in p or "cartesia" in p:
                        self._current.tts_ttfb = d.value

                # --- Processing Time ---
                elif isinstance(d, ProcessingMetricsData):
                    if "deepgram" in p or "stt" in p:
                        self._current.stt_processing = d.value
                    elif "openai" in p or "llm" in p or "gpt" in p:
                        self._current.llm_processing = d.value
                    elif "elevenlabs" in p or "tts" in p or "cartesia" in p:
                        self._current.tts_processing = d.value

                # --- Text Aggregation（TTS 特有）---
                elif isinstance(d, TextAggregationMetricsData):
                    self._current.tts_text_aggregation = d.value

                # --- Token Usage ---
                elif isinstance(d, LLMUsageMetricsData):
                    self._current.llm_prompt_tokens = d.value.prompt_tokens
                    self._current.llm_completion_tokens = d.value.completion_tokens

                # --- TTS chars ---
                elif isinstance(d, TTSUsageMetricsData):
                    self._current.tts_chars += d.value

    def print_session_summary(self):
        if not self._session_turns:
            return

        total_turns = len(self._session_turns)
        stt_ttfbs = [t.stt_ttfb for t in self._session_turns if t.stt_ttfb]
        llm_ttfbs = [t.llm_ttfb for t in self._session_turns if t.llm_ttfb]
        llm_procs = [t.llm_processing for t in self._session_turns if t.llm_processing]
        tts_ttfbs = [t.tts_ttfb for t in self._session_turns if t.tts_ttfb]
        e2es = [t.e2e_latency_ms for t in self._session_turns if t.e2e_latency_ms]
        total_tokens = sum(
            t.llm_prompt_tokens + t.llm_completion_tokens for t in self._session_turns
        )
        total_chars = sum(t.tts_chars for t in self._session_turns)

        def avg_ms(lst):
            return f"{sum(lst) / len(lst) * 1000:.0f}ms" if lst else "—"

        print(f"\n{'═' * 65}")
        print(f" Session Summary  ({total_turns} turns)")
        print(f"{'─' * 65}")
        print(f" {'Stage':<25} {'Avg TTFB':>10}  {'Avg Processing':>14}")
        print(f"{'─' * 65}")
        print(f" {'STT (Deepgram)':<25} {avg_ms(stt_ttfbs):>10}  {'—':>14}")
        print(f" {'LLM (OpenAI)':<25} {avg_ms(llm_ttfbs):>10}  {avg_ms(llm_procs):>14}")
        print(f" {'TTS (ElevenLabs)':<25} {avg_ms(tts_ttfbs):>10}  {'—':>14}")
        print(f"{'─' * 65}")
        print(f" Avg E2E latency : {avg_ms(e2es)}")
        print(f" Total LLM tokens: {total_tokens}")
        print(f" Total TTS chars : {total_chars}")
        print(f"{'═' * 65}\n")


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
            system_instruction="You are a helpful assistant. Keep responses short.",
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

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    metrics_observer = PerStageMetricsObserver()

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,          # 开启 → TTFBMetricsData, ProcessingMetricsData
            enable_usage_metrics=True,    # 开启 → LLMUsageMetricsData, TTSUsageMetricsData
        ),
        observers=[metrics_observer],
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user briefly.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 65)
    print(" Per-Stage Metrics Demo")
    print(" 每轮对话结束后会打印该轮的 stage breakdown 表格")
    print(" Ctrl+C 结束后会打印 session 平均值")
    print("=" * 65)

    try:
        await runner.run(task)
    finally:
        metrics_observer.print_session_summary()


if __name__ == "__main__":
    asyncio.run(main())
