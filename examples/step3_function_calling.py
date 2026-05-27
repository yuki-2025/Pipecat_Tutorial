"""
Step 3 — Function Calling（工具调用）
======================================
在 step2 的基础上，给 agent 加上调用外部工具的能力。
运行后可以问 agent "What's the weather in Tokyo?" 或 "Any restaurant suggestions in Seattle?"

你会学到：
    1. FunctionSchema — 怎么定义一个工具的参数格式
    2. ToolsSchema     — 把多个工具打包给 LLM
    3. llm.register_function() — 注册工具的实际处理函数
    4. FunctionCallParams — 函数被调用时的参数和回调
    5. 事件 on_function_calls_started — 工具被调用时的 hook

安装依赖：
    pip install "pipecat-ai[local,deepgram,openai,cartesia,silero]" python-dotenv loguru
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数（模拟 API 调用）
# 真实场景里这里会调用天气 API、数据库等
# ═══════════════════════════════════════════════════════════════════════════

async def get_current_weather(params: FunctionCallParams):
    location = params.arguments.get("location", "Unknown")
    format_ = params.arguments.get("format", "fahrenheit")
    logger.info(f"Tool called: get_current_weather({location}, {format_})")
    # 模拟 API 结果
    await params.result_callback({
        "location": location,
        "conditions": "sunny",
        "temperature": "75" if format_ == "fahrenheit" else "24",
        "unit": format_,
    })


async def get_restaurant_recommendation(params: FunctionCallParams):
    location = params.arguments.get("location", "Unknown")
    logger.info(f"Tool called: get_restaurant_recommendation({location})")
    # 模拟推荐
    await params.result_callback({
        "name": "The Golden Spoon",
        "cuisine": "Italian",
        "rating": 4.8,
        "location": location,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 工具的 schema（告诉 LLM 工具叫什么、有哪些参数）
# ═══════════════════════════════════════════════════════════════════════════

weather_tool = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather in a city",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. Tokyo, Japan",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit. Infer from the user's location.",
        },
    },
    required=["location", "format"],
)

restaurant_tool = FunctionSchema(
    name="get_restaurant_recommendation",
    description="Get a restaurant recommendation for a given city",
    properties={
        "location": {
            "type": "string",
            "description": "The city, e.g. Seattle, WA",
        },
    },
    required=["location"],
)

# 把所有工具打包
tools = ToolsSchema(standard_tools=[weather_tool, restaurant_tool])


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM") 
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=(
                "You are a helpful assistant. You can check the weather and "
                "recommend restaurants. Keep responses short and conversational."
            ),
        ),
    )

    # ── 注册工具函数 ──────────────────────────────────────────────────────
    # 把 LLM 的 function call 和实际的 Python 函数绑定
    llm.register_function("get_current_weather", get_current_weather)
    llm.register_function("get_restaurant_recommendation", get_restaurant_recommendation)

    # ── 事件：工具被调用时，先说一句话让用户知道在处理 ──────────────────
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that for you."))

    # Context 里传入 tools，LLM 就知道有哪些工具可以用
    context = LLMContext(tools=tools)
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
        params=PipelineParams(enable_metrics=True),
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user and let them know you can check weather and recommend restaurants.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("\n🎤 Voice agent with tools is running!")
    print("   Try asking: 'What's the weather in Tokyo?' or 'Restaurant suggestions in Seattle?'")
    print("   Press Ctrl+C to stop.\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
