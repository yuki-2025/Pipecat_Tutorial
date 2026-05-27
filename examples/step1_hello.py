"""
Step 1 — 最简单的 Pipecat 示例
================================
只需要 ElevenLabs API key。
运行后，电脑喇叭会说一句话，然后程序结束。

目的：理解三个最核心的概念
    1. Pipeline  - 处理器的链条
    2. Frame     - 数据在链条里的"容器"（TTSSpeakFrame, EndFrame）
    3. Transport - 音频怎么出去（LocalAudioTransport = 本地喇叭）

安装依赖：
    pip install "pipecat-ai[local,elevenlabs]" python-dotenv loguru

配置：
    复制 .env.example 为 .env，填入 ELEVENLABS_API_KEY
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

# 把 loguru 日志输出到 stderr，级别 INFO（不会太吵）
logger.remove(0)
logger.add(sys.stderr, level="INFO")


async def main():
    # ── 1. Transport ──────────────────────────────────────────────────────
    # Transport 负责音频怎么进来（input）、怎么出去（output）
    # LocalAudioTransport = 用电脑的麦克风和喇叭
    # 这里只开 audio_out（输出），因为我们只需要说话，不需要听
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_out_enabled=True)
    )

    # ── 2. TTS Service ────────────────────────────────────────────────────
    # TTS = Text-to-Speech，负责把文字变成语音音频
    # 它接收 TTSSpeakFrame（文字），输出 AudioRawFrame（音频）
    # tts = CartesiaTTSService(                                                                                                                 
    #   api_key=os.environ["CARTESIA_API_KEY"],                                                                                               
    #   settings=CartesiaTTSService.Settings(                                                                                                 
    #   voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady 声音                                                        
    # ),      
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        # voice_id 可以在 ElevenLabs 网站的 Voice Library 找到
        # 这是内建的 "Rachel" 声音（免费账号可用）
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM") 
    )

    # ── 3. Pipeline ───────────────────────────────────────────────────────
    # Pipeline 把处理器按顺序串起来
    # 数据流向：tts → transport.output()
    #
    # 当 TTSSpeakFrame("Hello!") 进入 pipeline：
    #   tts 处理 → 生成 AudioRawFrame
    #   transport.output() 接收 → 播放出来
    pipeline = Pipeline([tts, transport.output()])

    # ── 4. PipelineTask ───────────────────────────────────────────────────
    # PipelineTask 把 pipeline 包装成一个可以运行的异步任务
    task = PipelineTask(pipeline)

    # ── 5. 发送 Frame ─────────────────────────────────────────────────────
    async def say_something():
        # 等 1 秒让 pipeline 完全初始化
        await asyncio.sleep(1)

        # queue_frames 把 frame 放入 pipeline 的处理队列
        # TTSSpeakFrame：让 TTS 读出这段文字
        # EndFrame：告诉 pipeline "任务完成，可以结束了"
        await task.queue_frames([
            TTSSpeakFrame("Hello! I am your Pipecat voice agent. This is Step 1."),
            EndFrame(),
        ])

    # ── 6. PipelineRunner ─────────────────────────────────────────────────
    # PipelineRunner 运行 task，处理 asyncio 事件循环
    # Windows 上 handle_sigint 要设为 False（不然报错）
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    # asyncio.gather 同时运行 runner 和 say_something
    # runner.run(task) 会一直跑，直到收到 EndFrame 才结束
    await asyncio.gather(runner.run(task), say_something())


if __name__ == "__main__":
    asyncio.run(main())
