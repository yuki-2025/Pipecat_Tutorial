"""
Step 15 — Speech-to-Speech：Gemini Live（LocalAudio 版）
=========================================================
和 step14 相同的 Speech-to-Speech 概念，但用 Google Gemini Live。

OpenAI Realtime vs Gemini Live 对比：
    OpenAI Realtime：  稳定，成熟，声音选择多，语义 turn 检测强
    Gemini Live：      集成 Google 搜索工具，支持视频（multimodal），
                       可以用 Google VAD 参数调整

你会学到：
    1. GeminiLiveLLMService — Gemini 的 S2S 实现
    2. LiveVADParams / GeminiVADParams — Gemini 的 VAD 配置
    3. 两个 S2S 服务的参数差异
    4. 为什么 S2S pipeline 结构基本一样（universal LLMContext 的好处）

安装：
    uv add "pipecat-ai[local,google,silero]"

所需 API key：GOOGLE_API_KEY（需要 Gemini API 访问权限）
申请：https://aistudio.google.com/apikey（免费）
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

# ── Gemini Live imports ───────────────────────────────────────────────────
from pipecat.services.google.gemini_multimodal_live.gemini import (
    GeminiLiveLLMService,
    GeminiLiveParams,
    InputParams,
)

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

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

    # ── GeminiLiveLLMService ──────────────────────────────────────────────
    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        system_instruction=(
            "You are a helpful voice assistant. "
            "Keep responses short and conversational. "
            "No markdown or bullet points."
        ),
        params=GeminiLiveParams(
            model="gemini-2.0-flash-live-001",  # 最新的 Gemini Live 模型
            voice_name="Puck",                   # Puck/Charon/Kore/Fenrir/Aoede
            input=InputParams(
                # Gemini 的 VAD 参数（类似 OpenAI 的 turn_detection）
                # 通过 GOOGLE_API_KEY 就能访问，无需额外配置
            ),
        ),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # ── Pipeline 结构和 step14 完全一样 ──────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        llm,                           # GeminiLive 做 STT + LLM + TTS
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    context.add_message({
        "role": "user",
        "content": "Please greet the user and introduce yourself briefly.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Speech-to-Speech: Gemini Live (LocalAudio)")
    print(" 一个模型做 STT + LLM + TTS")
    print(" 建议戴耳机 (echo 问题)")
    print(" Ctrl+C 结束")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
