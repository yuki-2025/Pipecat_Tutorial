"""
Step 6 — 动态 Context 注入（Persona 切换 + 用户记忆）
======================================================
一个"真实感"更强的 agent：知道你是谁、可以中途换性格、可以注入外部信息。

你会学到：
    1. 用户 Profile 注入 ── 启动时把用户信息加入 context
    2. Persona 切换 ── 说 "switch to formal" 让 agent 变成正式语气
    3. 动态 context 注入 ── 运行时插入信息（如搜索结果、数据库查询）
    4. LLMMessagesAppendFrame ── 不打断对话地往 context 加消息
    5. LLMMessagesUpdateFrame ── 完全替换 context（核弹级重置）
    6. 用 FrameProcessor 监听关键词，触发 context 操作

运行方式：
    uv run python examples/step6_context_injection.py

可以说的话：
    - "switch to formal"  → agent 变成正式语气
    - "switch to casual"  → agent 变回轻松语气
    - "who am I"          → agent 会用注入的 profile 回答
    - "what time is it"   → 演示注入实时数据

所需 API key：DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
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

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ── 模拟用户 Profile（实际项目里从数据库读）──────────────────────────────────
USER_PROFILE = {
    "name": "Alex",
    "language": "English",
    "preferences": "prefers concise answers, interested in technology",
    "subscription": "Pro user since 2024",
}

# ── Persona 定义 ──────────────────────────────────────────────────────────
PERSONAS = {
    "casual": (
        "You are a friendly, casual assistant. Use relaxed language, "
        "contractions, and a warm tone. Keep responses short."
    ),
    "formal": (
        "You are a professional, formal assistant. Use proper grammar, "
        "avoid contractions, and maintain a respectful tone. Keep responses concise."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# CommandDetector：监听关键词，触发 context 操作
#
# 这是 step4 学到的 FrameProcessor 模式的实际应用：
# 拦截 TranscriptionFrame，检查关键词，注入对应的 context 变化
# ═══════════════════════════════════════════════════════════════════════════
class CommandDetector(FrameProcessor):

    def __init__(self, context: LLMContext, task_ref: list, tts_ref: list):
        super().__init__()
        self._context = context
        self._task_ref = task_ref
        self._tts_ref = tts_ref
        self._current_persona = "casual"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.lower().strip()
            task = self._task_ref[0]

            # ── 命令 1：切换 persona ──────────────────────────────────────
            if "switch to formal" in text or "be more formal" in text:
                await self._switch_persona("formal", task)
                return  # 吞掉这个 frame，不进入 LLM

            if "switch to casual" in text or "be more casual" in text:
                await self._switch_persona("casual", task)
                return

            # ── 命令 2：注入实时数据（演示：当前时间）────────────────────
            if "what time" in text or "what's the time" in text:
                now = datetime.now().strftime("%I:%M %p on %A, %B %d")
                # LLMMessagesAppendFrame：不打断对话，直接往 context 里追加信息
                # 这和 context.add_message() 的区别是：
                # AppendFrame 会通过 pipeline 处理，确保时序正确
                inject_frame = LLMMessagesAppendFrame(messages=[{
                    "role": "developer",
                    "content": f"[Real-time data injected] Current time: {now}. "
                               f"Answer the user's time question using this.",
                }])
                await self.push_frame(inject_frame, direction)
                # 这次不 return，让原来的 TranscriptionFrame 也继续走
                # LLM 会先看到注入的 developer 消息，再看到用户问题

        await self.push_frame(frame, direction)

    async def _switch_persona(self, persona_name: str, task):
        """切换 persona：更新 system instruction + 触发 LLM 确认"""
        if persona_name == self._current_persona:
            return

        self._current_persona = persona_name
        new_instruction = PERSONAS[persona_name]

        # LLMMessagesUpdateFrame：完全替换整个 context
        # 用新的 system instruction 重建 context，但保留用户 profile
        new_messages = [
            {
                "role": "developer",
                "content": (
                    f"{new_instruction}\n\n"
                    f"User profile: {USER_PROFILE}\n\n"
                    f"You just switched to {persona_name} mode. "
                    f"Briefly acknowledge this switch."
                ),
            }
        ]

        update_frame = LLMMessagesUpdateFrame(messages=new_messages)
        await task.queue_frames([update_frame, LLMRunFrame()])

        print(f"\n[System] 🎭 Switched to {persona_name} persona")


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM"),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
    )

    # ── Context 初始化：注入用户 Profile ──────────────────────────────────
    # 这模拟了真实 agent 的常见模式：
    # 用户登录后，从数据库读 profile，注入到 context 的 system message
    context = LLMContext()
    context.add_message({
        "role": "developer",
        "content": (
            f"{PERSONAS['casual']}\n\n"
            f"You know the following about this user:\n"
            f"- Name: {USER_PROFILE['name']}\n"
            f"- Language: {USER_PROFILE['language']}\n"
            f"- Preferences: {USER_PROFILE['preferences']}\n"
            f"- Account: {USER_PROFILE['subscription']}\n\n"
            f"Use this information to personalize your responses. "
            f"You can say the user's name occasionally."
        ),
    })

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    task_ref = [None]
    tts_ref = [tts]
    command_detector = CommandDetector(context, task_ref, tts_ref)

    pipeline = Pipeline([
        transport.input(),
        stt,
        command_detector,        # ← 在 user_aggregator 之前拦截命令
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=False),
    )
    task_ref[0] = task

    # ── 启动：让 agent 先打招呼，并且用上 profile 里的名字 ──────────────
    context.add_message({
        "role": "developer",
        "content": "Greet the user by name. Keep it to one sentence.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Context Injection Demo")
    print(" Try saying:")
    print("   'who am I'          → agent uses your profile")
    print("   'switch to formal'  → change persona")
    print("   'switch to casual'  → change back")
    print("   'what time is it'   → real-time data injection")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
