"""
Step 16 — LLMSwitcher（运行时切换 LLM + 故障转移）
====================================================
在 pipeline 运行时无缝切换 LLM，不需要重启。

用途：
    - 成本优化：默认用便宜模型（gpt-4o-mini），复杂问题切换到贵的（gpt-4o）
    - 故障转移：某个 LLM API 报错，自动切换到备用
    - 多语言：切换到更擅长某语言的模型
    - A/B 测试：对比不同模型的表现

你会学到：
    1. LLMSwitcher — 替代单个 LLM service，在 pipeline 里管理多个 LLM
    2. ServiceSwitcherStrategyManual — 手动触发切换
    3. ServiceSwitcherStrategyFailover — 自动故障转移（某 LLM 报错 → 切下一个）
    4. ManuallySwitchServiceFrame — 发这个 frame 触发手动切换
    5. 关键前提：所有参与切换的 LLM 必须共享同一个 LLMContext

在这个例子里：
    - 默认用 gpt-4o-mini（快速便宜）
    - 说 "switch to smart" 切换到 gpt-4o（精确但贵）
    - 说 "switch to fast" 切回 gpt-4o-mini
    注：如果你有 Anthropic 或 Google key，把第二个 LLM 换成那个更有意义

安装：（和 step2 一样）
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"

所需 API key：DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    ManuallySwitchServiceFrame,  # ← 这个 frame 触发手动 LLM 切换
    TranscriptionFrame,
    TTSSpeakFrame,
)

# ── LLMSwitcher imports ───────────────────────────────────────────────────
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.service_switcher import (
    ServiceSwitcherStrategyFailover,  # 自动故障转移
    ServiceSwitcherStrategyManual,    # 手动触发切换
)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ── Switch Command Detector ───────────────────────────────────────────────
class SwitchCommandDetector(FrameProcessor):
    """监听用户指令，触发 LLM 切换"""

    def __init__(self, llm_switcher: LLMSwitcher, llm_mini, llm_full, tts):
        super().__init__()
        self._switcher = llm_switcher
        self._llm_mini = llm_mini
        self._llm_full = llm_full
        self._tts = tts
        self._current = "mini"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.lower()

            if ("switch to smart" in text or "smart mode" in text) and self._current != "full":
                self._current = "full"
                # ManuallySwitchServiceFrame：告诉 LLMSwitcher 切换到指定的 LLM
                await self.push_frame(
                    ManuallySwitchServiceFrame(service=self._llm_full)
                )
                await self._tts.queue_frame(
                    TTSSpeakFrame("Switched to smart mode. Using GPT-4o now.")
                )
                return

            elif ("switch to fast" in text or "fast mode" in text) and self._current != "mini":
                self._current = "mini"
                await self.push_frame(
                    ManuallySwitchServiceFrame(service=self._llm_mini)
                )
                await self._tts.queue_frame(
                    TTSSpeakFrame("Switched to fast mode. Using GPT-4o-mini now.")
                )
                return

        await self.push_frame(frame, direction)


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

    # ── 两个 LLM 实例 ─────────────────────────────────────────────────────
    # 关键：必须用 universal LLMContext（不是 OpenAILLMContext）
    # 这样两个 LLM 才能共享同一个对话历史
    system_instruction = (
        "You are a helpful assistant. "
        "Keep responses short and conversational. "
        "Mention which model you are when asked."
    )

    llm_mini = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=system_instruction + " You are the fast/mini model.",
        ),
    )
    llm_full = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o",
            system_instruction=system_instruction + " You are the smart/full model.",
        ),
    )

    # ── LLMSwitcher：管理多个 LLM ─────────────────────────────────────────
    # ServiceSwitcherStrategyManual：等待 ManuallySwitchServiceFrame 触发切换
    # ServiceSwitcherStrategyFailover：某 LLM 报非致命错误时自动切下一个
    llm_switcher = LLMSwitcher(
        llms=[llm_mini, llm_full],           # 第一个是默认 active
        strategy_type=ServiceSwitcherStrategyManual,
    )
    # 如果要自动故障转移，换成：
    # llm_switcher = LLMSwitcher(llms=[llm_mini, llm_full], strategy_type=ServiceSwitcherStrategyFailover)

    # 注册工具时用 switcher（会同时注册到所有 LLM）
    # llm_switcher.register_function("my_tool", my_tool_handler)

    context = LLMContext()  # universal LLMContext（不是 OpenAILLMContext）
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    switch_detector = SwitchCommandDetector(llm_switcher, llm_mini, llm_full, tts)

    pipeline = Pipeline([
        transport.input(),
        stt,
        switch_detector,         # ← 监听 "switch to smart/fast"
        user_aggregator,
        llm_switcher,            # ← 替代单个 llm，内部管理哪个 LLM active
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    context.add_message({
        "role": "developer",
        "content": "Greet the user. Tell them you're currently gpt-4o-mini (fast mode). They can say 'switch to smart' to use gpt-4o.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" LLMSwitcher Demo")
    print(" Default: gpt-4o-mini (fast, cheap)")
    print(" Say 'switch to smart' → gpt-4o (smart, expensive)")
    print(" Say 'switch to fast'  → back to gpt-4o-mini")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
