"""
Step 13 — 完整可观测性（超出 performance 的全部指标）
=====================================================
Pipecat 提供 6 个层次的可观测性，这个 example 展示所有层次。

层次总览：
    层 1  性能指标           MetricsFrame → TTFBMetricsData / ProcessingMetricsData（step12）
    层 2  E2E 延迟 breakdown  UserBotLatencyObserver（内建，比 step12 更完整）
    层 3  对话轮次事件        TurnTrackingObserver（轮次开始/结束/被打断/持续时间）
    层 4  Pipeline 启动时序   StartupTimingObserver（每个 processor 的初始化时间）
    层 5  OpenTelemetry 追踪  enable_tracing=True（需要 Jaeger/Langfuse，本例只说明）
    层 6  自定义对话分析      ConversationAnalyticsObserver（手写，跟踪打断/说话时长等）

关键认识：
    UserBotLatencyObserver 已经内建了 step12 手写的大部分功能，
    而且还有 on_latency_breakdown 提供按时间排序的详细事件时间线。
    建议生产环境直接用内建 observer，不用手写。

安装：（和 step2 一样，不需要新 extras）
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"

所需 API key：DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

# ── 内建 observers ─────────────────────────────────────────────────────────
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.startup_timing_observer import StartupTimingObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver

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
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# 层 6：自定义对话分析 Observer
#
# 跟踪的是"用户行为"和"对话质量"，而不是技术性能：
# - 打断次数（InterruptionFrame）
# - 用户说话时长分布（UserStartedSpeakingFrame → UserStoppedSpeakingFrame）
# - Bot 说话时长分布（BotStartedSpeakingFrame → BotStoppedSpeakingFrame）
# - 每轮是否被打断
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationAnalytics:
    total_turns: int = 0
    interrupted_turns: int = 0
    e2e_latencies: list[float] = field(default_factory=list)
    user_speaking_durations: list[float] = field(default_factory=list)
    bot_speaking_durations: list[float] = field(default_factory=list)

    # 内部状态
    _user_speaking_start: float | None = None
    _bot_speaking_start: float | None = None
    _user_stopped_ts: float | None = None
    _bot_started_ts: float | None = None
    _interrupted_this_turn: bool = False

    def avg(self, lst) -> str:
        return f"{sum(lst)/len(lst):.2f}s" if lst else "—"

    def summary(self) -> str:
        pct = (
            f"{self.interrupted_turns / self.total_turns * 100:.0f}%"
            if self.total_turns else "—"
        )
        avg_e2e = (
            f"{sum(self.e2e_latencies)/len(self.e2e_latencies)*1000:.0f}ms"
            if self.e2e_latencies else "—"
        )
        return (
            f"\n{'═'*55}\n"
            f" Conversation Quality Report\n"
            f"{'─'*55}\n"
            f" Total turns         : {self.total_turns}\n"
            f" Interrupted turns   : {self.interrupted_turns} ({pct})\n"
            f" Avg E2E latency     : {avg_e2e}\n"
            f" Avg user speaking   : {self.avg(self.user_speaking_durations)}\n"
            f" Avg bot speaking    : {self.avg(self.bot_speaking_durations)}\n"
            f"{'═'*55}"
        )


class ConversationAnalyticsObserver(BaseObserver):
    """
    跟踪对话行为指标，不关心技术性能细节。

    可以用来回答：
    - 用户喜欢打断 bot 吗？（interruption rate）
    - 用户平均说多长时间？（user speaking duration）
    - Bot 平均说多长时间？（bot speaking duration）
    - E2E 响应延迟是多少？（user stop → bot start）
    """

    def __init__(self):
        super().__init__()
        self._data = ConversationAnalytics()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        ts_secs = data.timestamp / 1e9

        # 用户开始说话
        if isinstance(frame, UserStartedSpeakingFrame):
            self._data._user_speaking_start = ts_secs
            self._data._user_stopped_ts = None  # reset

        # 用户停止说话
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._data._user_stopped_ts = ts_secs
            if self._data._user_speaking_start:
                duration = ts_secs - self._data._user_speaking_start
                self._data.user_speaking_durations.append(duration)
                self._data._user_speaking_start = None

        # Bot 开始说话
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._data._bot_speaking_start = ts_secs
            self._data._interrupted_this_turn = False
            # 计算 E2E 延迟
            if self._data._user_stopped_ts:
                e2e = ts_secs - self._data._user_stopped_ts
                self._data.e2e_latencies.append(e2e)

        # Bot 停止说话 = 这一轮结束
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._data._bot_speaking_start:
                duration = ts_secs - self._data._bot_speaking_start
                self._data.bot_speaking_durations.append(duration)
                self._data._bot_speaking_start = None
            self._data.total_turns += 1
            if self._data._interrupted_this_turn:
                self._data.interrupted_turns += 1
            self._data._interrupted_this_turn = False

        # 打断事件（用户说话打断 bot）
        elif isinstance(frame, InterruptionFrame):
            if self._data._bot_speaking_start is not None:
                self._data._interrupted_this_turn = True

    def print_summary(self):
        print(self._data.summary())


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

    # ── 层 4：StartupTimingObserver ────────────────────────────────────────
    startup_observer = StartupTimingObserver()

    @startup_observer.event_handler("on_startup_timing_report")
    async def on_startup(observer, report):
        print(f"\n[Startup] Pipeline ready in {report.total_duration_secs:.2f}s")
        for t in report.processor_timings:
            print(f"  {t.processor_name:<35} {t.duration_secs:.3f}s")

    # ── 层 2：UserBotLatencyObserver ───────────────────────────────────────
    # 内建 observer，比 step12 手写更完整：
    # - on_latency_measured : 简单 E2E 秒数
    # - on_latency_breakdown: 按时间排序的详细事件时间线
    # - on_first_bot_speech_latency: 客户端连接 → bot 第一次开口
    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_speech(observer, latency):
        print(f"\n[First Speech] {latency:.3f}s after pipeline start")

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):
        print(f"\n[E2E Latency] {latency * 1000:.0f}ms")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_breakdown(observer, breakdown):
        # breakdown.chronological_events() 返回按时间排序的事件字符串列表
        # 这是 Pipecat 帮你整理好的：user turn → STT TTFB → LLM TTFB → TTS TTFB → text agg
        print("[Latency Breakdown]")
        for event in breakdown.chronological_events():
            print(f"  {event}")
        if breakdown.user_turn_secs:
            print(f"  user speaking duration: {breakdown.user_turn_secs:.2f}s")
        if breakdown.function_calls:
            for fc in breakdown.function_calls:
                print(f"  function call: {fc}")

    # ── 层 3：TurnTrackingObserver ─────────────────────────────────────────
    # 跟踪轮次节奏，独立于性能指标
    # turn_end_timeout_secs：bot 停说话后等多久才算这轮结束
    turn_observer = TurnTrackingObserver(turn_end_timeout_secs=2.0)

    @turn_observer.event_handler("on_turn_started")
    async def on_turn_started(observer, turn_count):
        print(f"\n[Turn {turn_count}] Started")

    @turn_observer.event_handler("on_turn_ended")
    async def on_turn_ended(observer, turn_count, duration, was_interrupted):
        status = "INTERRUPTED ⚡" if was_interrupted else "completed"
        print(f"[Turn {turn_count}] Ended — {duration:.1f}s, {status}")

    # ── 层 6：自定义对话分析 ───────────────────────────────────────────────
    analytics_observer = ConversationAnalyticsObserver()

    # ── 挂载所有 observers ─────────────────────────────────────────────────
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,       # UserBotLatencyObserver 的 breakdown 需要这个
            enable_usage_metrics=True,
        ),
        observers=[
            startup_observer,          # 层 4：启动时序
            latency_observer,          # 层 2：E2E breakdown
            turn_observer,             # 层 3：轮次事件
            analytics_observer,        # 层 6：对话质量分析
        ],
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user briefly.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Full Observability Demo")
    print(" 层 2 E2E breakdown — 层 3 轮次事件 — 层 4 启动时序")
    print(" 层 6 对话质量分析（Ctrl+C 后打印）")
    print()
    print(" 层 5 OpenTelemetry（未在此展示）：")
    print("   uv add 'pipecat-ai[tracing]'")
    print("   PipelineTask(..., enable_tracing=True, enable_turn_tracking=True)")
    print("   → 集成 Jaeger / Langfuse / Opik")
    print("=" * 55)

    try:
        await runner.run(task)
    finally:
        analytics_observer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
