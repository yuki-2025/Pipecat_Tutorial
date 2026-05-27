"""
Step 18 — Multimodal（图片/图像进入 LLM Context）
==================================================
让 voice agent 能"看"图片：把图片加入 LLM context，用语音询问关于图片的问题。

你会学到：
    1. LLMContext.create_image_url_message() — 把图片 URL 加入 context
    2. LLMContext.create_image_message()     — 把本地图片（bytes）加入 context
    3. LLMMessagesAppendFrame + run_llm=True  — 注入图片后立刻触发 LLM
    4. 用 FrameProcessor 监听关键词，动态注入图片到对话
    5. 哪些 LLM 支持 multimodal（GPT-4o, Claude 3, Gemini）

用途：
    - 视觉问答（"这张图里有什么？"）
    - 文档理解（"帮我解释这张图表"）
    - 实时视觉助手（截图 → 问问题）
    - 医疗/工业图像分析

安装：（和 step2 一样，gpt-4o-mini 支持图片）
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
    LLMMessagesAppendFrame,
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
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")

# 测试用图片 URLs（公开可访问）
SAMPLE_IMAGES = {
    "pipecat": "https://raw.githubusercontent.com/pipecat-ai/pipecat/main/docs/assets/logo.png",
    "chart":   "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg",
    "diagram": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Culinary_fruits_front_view.jpg/320px-Culinary_fruits_front_view.jpg",
}


class ImageInjector(FrameProcessor):
    """
    监听用户说 "show image" 或 "load image"，把图片注入 LLM context。

    核心模式：
        1. LLMContext.create_image_url_message() 创建包含图片的 message
        2. LLMMessagesAppendFrame(messages=[image_msg], run_llm=True)
           → 把图片加入 context，并立刻触发 LLM 生成描述
    """

    def __init__(self, context: LLMContext, tts, task_ref: list):
        super().__init__()
        self._context = context
        self._tts = tts
        self._task_ref = task_ref
        self._image_loaded = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.lower()

            # 用户说 "show image" 或 "describe image" 时加载图片
            if ("show image" in text or "load image" in text or "describe image" in text):
                await self._load_image("chart", text)
                return

            # 用户说 "show pipecat logo" 时加载 pipecat logo
            elif "pipecat" in text and ("logo" in text or "image" in text):
                await self._load_image("pipecat", text)
                return

            elif "fruit" in text or "食物" in text:
                await self._load_image("diagram", text)
                return

        await self.push_frame(frame, direction)

    async def _load_image(self, image_key: str, user_text: str):
        url = SAMPLE_IMAGES.get(image_key, SAMPLE_IMAGES["chart"])

        # ── 核心：把图片 URL 加入 LLM context ────────────────────────────
        # create_image_url_message() 创建 multimodal message（图片 + 文字）
        image_message = LLMContext.create_image_url_message(
            url=url,
            text=f"The user said: '{user_text}'. Describe what you see in this image in 2-3 sentences.",
        )

        task = self._task_ref[0]
        if task:
            # LLMMessagesAppendFrame 把图片消息加入 context
            # run_llm=True → 加入后立刻触发 LLM 生成回复
            await task.queue_frames([
                LLMMessagesAppendFrame(
                    messages=[image_message],
                    run_llm=True,
                )
            ])

        self._image_loaded = True
        print(f"\n[Multimodal] Loaded image: {url}")


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

    # 使用支持图片的模型（gpt-4o-mini 也支持 vision）
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",  # vision 支持：gpt-4o, gpt-4o-mini, gpt-4-vision
            system_instruction=(
                "You are a helpful voice assistant that can see and describe images. "
                "When shown an image, describe what you see naturally in conversational language. "
                "Keep responses short."
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

    task_ref = [None]
    image_injector = ImageInjector(context, tts, task_ref)

    pipeline = Pipeline([
        transport.input(),
        stt,
        image_injector,          # ← 监听用户指令，注入图片
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())
    task_ref[0] = task

    context.add_message({
        "role": "developer",
        "content": "Greet the user. Tell them they can say 'show image' to load an image for you to describe.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Multimodal Demo (Voice + Vision)")
    print(" Say 'show image'      → load a nature photo")
    print(" Say 'pipecat logo'    → load Pipecat logo")
    print(" Say 'fruit image'     → load a fruit photo")
    print(" Then ask questions about the image!")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
