"""
Step 5 — Web Transport（浏览器可访问）
======================================
从"本地麦克风"升级成"浏览器连接"。
运行后，在 http://localhost:7860/client 用浏览器打开就可以对话。

你会学到：
    1. Pipecat Runner 系统 ── 统一处理 transport 选择和服务器启动
    2. transport_params 字典 ── 支持多种 transport，命令行切换
    3. 事件处理 ── on_client_connected / on_client_disconnected
    4. 两种 web transport 的区别：
       - webrtc (SmallWebRTC) ── 无需额外 key，P2P 直连
       - daily              ── 需要 DAILY_API_KEY，多方通话支持更好

运行方式：
    # 方式 1：WebRTC（无需 Daily key）
    uv run python examples/step5_web_transport.py --transport webrtc

    # 方式 2：Daily（需要 DAILY_API_KEY）
    uv run python examples/step5_web_transport.py --transport daily

    然后打开浏览器 → http://localhost:7860/client

所需 API key：DEEPGRAM + OPENAI + ELEVENLABS
可选 API key：DAILY_API_KEY（只在 --transport daily 时需要）

申请 Daily 免费账号：https://dashboard.daily.co/u/signup
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# Runner 工具：负责解析 --transport 参数，创建对应的 transport
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

# FastAPI WebSocket（--transport twilio 时用，也可以直接 WebSocket 连接）
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Daily 只在实际选择时才 import（daily-python 不支持 Windows，延迟 import 避免报错）
def _daily_params():
    from pipecat.transports.daily.transport import DailyParams
    return DailyParams(audio_in_enabled=True, audio_out_enabled=True)

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# transport_params：一个字典，key = transport 名字，value = 对应的 Params
#
# 这是 Pipecat 推荐的模式，让同一份代码支持多种 transport。
# 实际用哪个由命令行参数 --transport 决定。
# ═══════════════════════════════════════════════════════════════════════════
transport_params = {
    # SmallWebRTC：轻量 P2P WebRTC，内建于 pipecat，不需要额外服务
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    # Daily WebRTC：需要 DAILY_API_KEY，功能更完整（录制、多方通话等）
    # Windows 不支持 daily-python，需要在 Linux/macOS 或 WSL2 下运行
    "daily": _daily_params,
    # Twilio / WebSocket：电话接入用
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Bot 逻辑，transport 已经由 runner 决定好了"""
    logger.info("Bot starting...")

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
                "You are a friendly voice assistant accessible via web browser. "
                "Keep responses short and conversational."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # Web 用户通常用耳机，可以开启中断
            enable_metrics=True,
        ),
        # idle_timeout_secs：多久没活动就自动结束（从 runner_args 读取）
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # ── 事件处理 ────────────────────────────────────────────────────────────
    # on_client_connected：有浏览器/客户端连进来时触发
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # 客户端连进来后，让 agent 先开口打招呼
        context.add_message({
            "role": "developer",
            "content": "A user just connected via web browser. Greet them warmly and ask how you can help.",
        })
        await task.queue_frames([LLMRunFrame()])

    # on_client_disconnected：客户端离开时触发
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        # 取消 pipeline，释放资源
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


# ── bot() 函数是 Pipecat Cloud 的入口点 ────────────────────────────────────
# Pipecat Cloud 部署时会调用这个函数
# 本地运行时，main() 也会调用它
async def bot(runner_args: RunnerArguments):
    """Pipecat Cloud 兼容的入口点"""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


# ── 本地运行 ────────────────────────────────────────────────────────────────
# pipecat.runner.run.main() 会：
#   1. 解析命令行参数（--transport, --port 等）
#   2. 启动 FastAPI 服务器（默认 port 7860）
#   3. 在 http://localhost:7860/client 提供内建的浏览器客户端
#   4. 等待客户端连接后调用 bot(runner_args)
if __name__ == "__main__":
    from pipecat.runner.run import main
    main()

#   ┌──────────────────────────────────┬──────────────────────────────────────────────────────┐
#   │               方案               │                         说明                         │
#   ├──────────────────────────────────┼──────────────────────────────────────────────────────┤
#   │ WSL2（推荐）                     │ Windows 里的 Linux 环境，装 pipecat-ai[daily] 没问题 │
#   ├──────────────────────────────────┼──────────────────────────────────────────────────────┤
#   │ Docker                           │ 跑 Linux 容器                                        │
#   ├──────────────────────────────────┼──────────────────────────────────────────────────────┤
#   │ 部署到 Pipecat Cloud / Linux VPS │ 生产环境自然就是 Linux                               │
#   └──────────────────────────────────┴──────────────────────────────────────────────────────┘
#   uv run python .\examples\step5_web_transport.py --transport webrtc