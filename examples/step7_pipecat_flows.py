"""
Step 7 — Pipecat Flows（结构化对话状态机）
==========================================
适合有明确流程的场景：订餐、预约、问卷、客服引导。
每个 Node（节点）只给 LLM 一个任务 + 一组工具，避免 prompt 太大导致幻觉。

你会学到：
    1. NodeConfig  — 定义一个对话节点（角色 / 任务 / 函数）
    2. FlowManager — 管理节点之间的转换和全局状态
    3. FlowsFunctionSchema — 节点的函数定义，带自动 handler 绑定
    4. Edge Function — 返回 (result, next_node) 触发节点转换
    5. Node Function — 返回 (result, None)   留在当前节点
    6. flow_manager.state — 跨节点共享数据
    7. post_actions — 节点完成后执行动作（end_conversation 等）

对比 step3 的 Function Calling：
    step3 = 自由对话 + 工具调用（LLM 自己决定走向）
    step7 = 结构化流程 + 明确路径（你决定走向，LLM 只做当前节点的事）

安装：
    uv add pipecat-ai-flows
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"

这个例子：三节点咖啡订单机器人
    greeting   → 询问用户名
    take_order → 询问要什么
    confirm    → 确认并结束对话
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# ── Pipecat Flows 的核心 imports ──────────────────────────────────────────
from pipecat_flows import (
    FlowArgs,       # 函数被调用时的参数 dict
    FlowManager,    # 管理节点转换和状态
    FlowsFunctionSchema,  # 定义节点可用的函数
    NodeConfig,     # 定义一个对话节点
)

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# 节点定义
# 每个节点 = 一个 dict 或 NodeConfig 对象，包含：
#   role_message  : LLM 的系统指令（persona，只需在第一个节点设置，后续沿用）
#   task_messages : 这个节点要做什么（developer 角色）
#   functions     : 这个节点可以调用哪些函数
#   post_actions  : 节点完成后自动执行（如 end_conversation）
# ═══════════════════════════════════════════════════════════════════════════

def create_greeting_node() -> NodeConfig:
    """第一个节点：问用户名字"""

    # FlowsFunctionSchema = 这个节点的函数定义 + handler 绑定
    # 当 LLM 调用 record_name 时，Pipecat Flows 自动调用 handle_record_name
    record_name_func = FlowsFunctionSchema(
        name="record_name",
        description="Record the customer's name after they provide it.",
        properties={
            "name": {"type": "string", "description": "The customer's name"},
        },
        required=["name"],
        handler=handle_record_name,  # Edge function：返回 (result, next_node)
    )

    return NodeConfig(
        name="greeting",
        role_message=(
            "You are a friendly barista at a coffee shop. "
            "Be warm and brief. Responses will be spoken aloud — no markdown."
        ),
        task_messages=[{
            "role": "developer",
            "content": "Greet the customer warmly and ask for their name.",
        }],
        functions=[record_name_func],
    )


async def handle_record_name(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Edge function：记录名字，转换到下一个节点"""
    name = args["name"]
    # flow_manager.state 是跨节点共享的字典
    flow_manager.state["customer_name"] = name
    return f"Name recorded: {name}", create_take_order_node()


def create_take_order_node() -> NodeConfig:
    """第二个节点：接受点单"""

    record_order_func = FlowsFunctionSchema(
        name="record_order",
        description="Record what the customer wants to order.",
        properties={
            "item": {"type": "string", "description": "The coffee/drink item ordered"},
            "size": {
                "type": "string",
                "enum": ["small", "medium", "large"],
                "description": "The size of the drink",
            },
        },
        required=["item", "size"],
        handler=handle_record_order,
    )

    return NodeConfig(
        name="take_order",
        task_messages=[{
            "role": "developer",
            "content": (
                "Ask the customer what they'd like to order. "
                "We have coffee, tea, and hot chocolate in small, medium, and large."
            ),
        }],
        functions=[record_order_func],
    )


async def handle_record_order(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Edge function：记录订单，转换到确认节点"""
    flow_manager.state["order_item"] = args["item"]
    flow_manager.state["order_size"] = args["size"]
    return f"Order recorded: {args['size']} {args['item']}", create_confirm_node()


def create_confirm_node() -> NodeConfig:
    """第三个节点：确认订单并结束"""

    # 从 state 里读名字和订单（这里用 lambda 延迟获取，因为 state 在运行时才有值）
    return NodeConfig(
        name="confirm",
        task_messages=[{
            "role": "developer",
            "content": (
                "Confirm the order details using the customer's name and what they ordered "
                "(available in the conversation history). Tell them the order will be ready shortly. "
                "Be warm and brief."
            ),
        }],
        # post_actions：节点完成后自动执行，end_conversation 会关闭 pipeline
        post_actions=[{"type": "end_conversation"}],
    )


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
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
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

    task = PipelineTask(pipeline, params=PipelineParams())

    # FlowManager：连接 task / llm / aggregator / transport
    # 它接管对话的节点转换逻辑
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=(user_aggregator, assistant_aggregator),
        transport=transport,
    )

    async def start_flow():
        await asyncio.sleep(1)
        # initialize() 设置第一个节点，触发 LLM 开口
        await flow_manager.initialize(create_greeting_node())

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 50)
    print(" Pipecat Flows Demo — Coffee Shop Bot")
    print(" Flow: greeting → take_order → confirm → end")
    print("=" * 50)

    await asyncio.gather(runner.run(task), start_flow())


if __name__ == "__main__":
    asyncio.run(main())
