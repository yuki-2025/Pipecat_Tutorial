"""
Step 17 — Context Summarization（长对话记忆压缩）
==================================================
对话越长，LLM context 越大，token 越贵，速度越慢，最终超出 context window 报错。
Pipecat 内建自动压缩：超过阈值时，让 LLM 把旧消息压缩成摘要，保留近期内容。

效果：
    消息 1-100：→ 摘要（"用户聊了天气、问了餐厅推荐，喜欢意大利菜…"）
    消息 95-100：→ 保留（最近几条保持完整）
    消息 101+：继续正常对话，LLM 看到：摘要 + 近期消息

你会学到：
    1. enable_auto_context_summarization — 一个参数开启自动压缩
    2. LLMAutoContextSummarizationConfig — 配置触发条件（token 数、消息数）
    3. LLMContextSummaryConfig — 配置摘要的目标大小和保留消息数
    4. on_summary_applied — 压缩发生时的事件 hook
    5. 手动触发：LLMSummarizeContextFrame（按需压缩，不等自动触发）

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
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# ── Context Summarization imports ──────────────────────────────────────────
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


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
                "You are a helpful assistant for long conversations. "
                "When you notice context is getting summarized, acknowledge it naturally. "
                "Keep responses short."
            ),
        ),
    )

    context = LLMContext()

    # ── Context Summarization 配置 ─────────────────────────────────────────
    # 触发条件（满足任一即触发）：
    #   max_context_tokens = 1000   → context 估算超过 1000 token（~4000 字符）时触发
    #   max_unsummarized_messages=5 → 新增超过 5 条消息时触发（测试用，很容易触发）
    #
    # 实际生产建议值：
    #   max_context_tokens = 8000 (默认), max_unsummarized_messages = 20 (默认)
    summarization_config = LLMAutoContextSummarizationConfig(
        max_context_tokens=1000,         # 设小一点，容易触发，便于测试
        max_unsummarized_messages=5,     # 每 5 条消息压缩一次（测试用）
        summary_config=LLMContextSummaryConfig(
            target_context_tokens=500,   # 压缩后目标 token 数
            min_messages_after_summary=2, # 保留最近 2 条消息不压缩
        ),
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
        assistant_params=LLMAssistantAggregatorParams(
            # ── 核心：开启自动压缩 ──────────────────────────────────────
            enable_auto_context_summarization=True,
            auto_context_summarization_config=summarization_config,
        ),
    )

    # ── 监听压缩事件 ──────────────────────────────────────────────────────
    @assistant_aggregator.event_handler("on_summary_applied")
    async def on_summary_applied(aggregator, summary: str, messages_before: int, messages_after: int):
        print(f"\n[Context Summarized]")
        print(f"  Before: {messages_before} messages")
        print(f"  After : {messages_after} messages")
        print(f"  Summary preview: {summary[:100]}...")

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    # 也可以手动触发压缩（不等自动触发）：
    # from pipecat.frames.frames import LLMSummarizeContextFrame
    # await task.queue_frames([LLMSummarizeContextFrame()])

    context.add_message({
        "role": "developer",
        "content": (
            "Start a long conversation. Ask the user about their day, "
            "their favorite things, and have a natural chat. "
            "The context will be summarized after a few exchanges."
        ),
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Context Summarization Demo")
    print(" 每 5 条消息（或超 1000 token）自动压缩 context")
    print(" 压缩时会打印 [Context Summarized]")
    print(" 尽量多聊几轮触发压缩")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
