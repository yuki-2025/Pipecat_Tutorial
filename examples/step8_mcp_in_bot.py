"""
Step 8 — MCPClient（在 bot 里接 MCP 工具）
==========================================
让 bot 的 LLM 直接调用 MCP server 提供的工具，不需要手动写 FunctionSchema。

你会学到：
    1. MCPClient                — 连接到 MCP server
    2. mcp.register_tools(llm) — 自动发现工具 + 注册给 LLM（一行搞定 step3 手写的那些）
    3. 三种连接方式：
       - StdioServerParameters  → 本地进程（本例使用）
       - SseServerParameters    → 远程 SSE
       - StreamableHttpParameters → 远程 HTTP（如 GitHub Copilot MCP）
    4. tools_filter             — 只暴露部分工具给 LLM
    5. tools_output_filters     — 截断/过滤工具返回值（防止 context 爆炸）

对比 step3：
    step3 = 手动写 FunctionSchema + register_function + 处理函数
    step8 = MCPClient 自动发现 + 一行注册，LLM 直接调用 MCP server 的工具

这个例子用 mcp-server-time（Pipecat 官方推荐的入门 MCP server）：
    - 提供 get_current_time / list_timezones 工具
    - 用 uvx 运行，不需要手动安装，不需要 Node.js
    - 试着问 bot: "What time is it?" 或 "What time is it in Tokyo?"

安装：
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero,mcp]"
    （mcp-server-time 由 uvx 自动处理）
"""

import asyncio
import os
import shutil
import sys

from dotenv import load_dotenv
from loguru import logger

from mcp import StdioServerParameters

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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService

# ── MCPClient：连接 MCP server 的核心类 ───────────────────────────────────
from pipecat.services.mcp_service import MCPClient

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
                "You are a helpful voice assistant with access to time tools. "
                "Keep responses short and conversational. "
                "When asked about time or timezones, use your tools."
            ),
        ),
    )

    # ── MCPClient：用 async with 管理连接生命周期 ─────────────────────────
    # StdioServerParameters = 用本地子进程运行 MCP server
    # uvx 是 uv 的工具运行器，类似 npx，自动安装并运行 Python 包
    async with MCPClient(
        server_params=StdioServerParameters(
            command=shutil.which("uvx"),          # 找到 uvx 的路径
            args=["mcp-server-time"],              # 运行 mcp-server-time
        ),
        # tools_filter：只暴露这两个工具给 LLM，忽略其他
        tools_filter=["get_current_time", "convert_time"],
        # tools_output_filters：截断过长的返回值
        tools_output_filters={
            "get_current_time": lambda r: str(r)[:200],
        },
    ) as mcp:
        # register_tools 做了三件事：
        # 1. 连接 MCP server，列出所有可用工具
        # 2. 把工具 schema 转换成 Pipecat 的 FunctionSchema 格式
        # 3. 调用 llm.register_function() 绑定每个工具的 handler
        # 返回值是 ToolsSchema，传给 LLMContext 让 LLM 知道有哪些工具
        tools = await mcp.register_tools(llm)

        # LLMContext 需要传入 tools，LLM 才能在回复里调用它们
        context = LLMContext(tools=tools)
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

        task = PipelineTask(pipeline, params=PipelineParams())

        context.add_message({
            "role": "developer",
            "content": "Greet the user and let them know they can ask about the current time.",
        })
        await task.queue_frames([LLMRunFrame()])

        runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

        print("=" * 55)
        print(" MCP Integration Demo")
        print(" MCP server: mcp-server-time (via uvx)")
        print(" Try asking:")
        print("   'What time is it?'")
        print("   'What time is it in Tokyo?'")
        print("=" * 55)

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
