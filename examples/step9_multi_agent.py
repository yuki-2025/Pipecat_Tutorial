"""
Step 9 — Multi-Agent 架构（Pipecat Subagents）
===============================================
当一个 pipeline 不够用时，用多个 agent 协作。
每个 agent 有自己的 LLM + pipeline，通过 AgentBus 通信。

你会学到：
    1. AgentRunner      — 管理所有 agent 的生命周期
    2. AgentBus         — agent 间通信的消息总线
    3. BusBridgeProcessor — 主 pipeline 里的"路由器"，把 frame 分发给 active agent
    4. LLMAgent         — 有 LLM pipeline 的 agent 基类
    5. @tool decorator  — 在 LLMAgent 里注册工具（比 register_function 更简洁）
    6. handoff_to()     — 把控制权转交给另一个 agent（无缝切换）
    7. @agent_ready     — 等指定 agent 启动完成后执行

架构图：
    AgentRunner
      └── MainAgent（拥有 transport + BusBridgeProcessor）
            ├── GreeterAgent（问好 + 路由到 SupportAgent）
            └── SupportAgent（回答问题 + 可以结束对话）

    MainAgent 的 Pipeline：
      mic → STT → user_agg → [BusBridgeProcessor] → TTS → speaker → assistant_agg
                                     ↑↓  （通过 Bus）
                               GreeterAgent / SupportAgent（各自有 LLM）

对比 step2（单 agent）：
    step2 = 一个 pipeline 里所有东西串在一起
    step9 = main agent 路由音频，多个 LLM agent 并行待命，active agent 处理对话

安装：
    uv add pipecat-ai-subagents
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# ── Subagents 的核心 imports ──────────────────────────────────────────────
from pipecat_subagents.agents import (
    BaseAgent,
    LLMAgent,               # 有 LLM pipeline 的 agent
    LLMAgentActivationArgs, # 激活 agent 时传入的参数
    agent_ready,            # 等某个 agent 准备好后执行的 decorator
    tool,                   # 在 LLMAgent 里注册工具（替代 register_function）
)
from pipecat_subagents.bus import AgentBus, BusBridgeProcessor  # 消息总线和路由器
from pipecat_subagents.runner import AgentRunner                 # 管理所有 agent
from pipecat_subagents.types import AgentReadyData               # @agent_ready 回调收到的数据

load_dotenv()
logger.remove(0)
# logger.add(sys.stderr, level="WARNING")
logger.add(sys.stderr, level="INFO")

# ═══════════════════════════════════════════════════════════════════════════
# LLM Agent 基类：两个 LLM agent 都继承这个
# 共享工具：transfer_to_agent（切换 agent）和 end_conversation（结束对话）
# ═══════════════════════════════════════════════════════════════════════════
class BaseVoiceAgent(LLMAgent):

    def __init__(self, name: str, *, bus: AgentBus, system_instruction: str):
        # bridged=() 表示这个 agent 从 bus 接收 frame（不直接拥有 transport）
        super().__init__(name, bus=bus, bridged=())
        self._system_instruction = system_instruction

    def build_llm(self) -> LLMService:
        """LLMAgent 要求实现：返回这个 agent 用的 LLM"""
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                model="gpt-4o-mini",
                system_instruction=self._system_instruction,
            ),
        )

    # ── @tool：比 register_function 更简洁的工具注册方式 ─────────────────
    # cancel_on_interruption=False：即使用户打断，也要等工具执行完（确保 handoff 完成）
    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(
        self, params: FunctionCallParams, agent: str, reason: str
    ):
        """Transfer the conversation to another agent.

        Args:
            agent (str): Target agent name ('greeter' or 'support').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"[{self.name}] handoff to '{agent}': {reason}")
        await self.handoff_to(
            agent,
            activation_args=LLMAgentActivationArgs(
                messages=[{"role": "user", "content": reason}],
            ),
            result_callback=params.result_callback,
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"[{self.name}] ending conversation: {reason}")
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": reason}],
                run_llm=True,
            )
        )
        await self.end(reason=reason, result_callback=params.result_callback)


# ═══════════════════════════════════════════════════════════════════════════
# GreeterAgent：欢迎用户，了解需求后转给 SupportAgent
# ═══════════════════════════════════════════════════════════════════════════
class GreeterAgent(BaseVoiceAgent):

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(
            name,
            bus=bus,
            system_instruction=(
                "You are a friendly greeter. Welcome the user briefly and ask how you can help. "
                "If they have a product question or need support, transfer them to the support agent. "
                "Keep responses very short."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SupportAgent：处理具体问题，可以转回 greeter 或结束对话
# ═══════════════════════════════════════════════════════════════════════════
class SupportAgent(BaseVoiceAgent):

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(
            name,
            bus=bus,
            system_instruction=(
                "You are a helpful support agent for a fictional coffee shop app. "
                "Answer questions about orders, menu, and hours. "
                "If the user just wants to chat, transfer them back to the greeter. "
                "End the conversation when the user says goodbye. "
                "Keep responses short."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# MainAgent：拥有 transport，用 BusBridgeProcessor 替代 LLM
# 负责音频 I/O，把 frame 通过 bus 路由给 active agent
# ═══════════════════════════════════════════════════════════════════════════
class MainAgent(BaseAgent):

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(name, bus=bus, active=True)  # active=True 表示这个 agent 一开始就启动
        self._transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                input_device_index=1,
            )
        )

    async def build_pipeline(self) -> Pipeline:
        """BaseAgent 可选实现：定义这个 agent 的 pipeline"""
        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM"),
        )

        context = LLMContext()
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                user_mute_strategies=[AlwaysUserMuteStrategy()],
            ),
        )

        # BusBridgeProcessor：替代 LLM 的位置
        # 它把来自 STT 的 frame 发到 bus，active agent 处理后把回复发回来
        bridge = BusBridgeProcessor(bus=self.bus, agent_name=self.name)

        pipeline = Pipeline([
            self._transport.input(),
            stt,
            user_aggregator,
            bridge,             # ← 核心：路由 frame 给 active LLM agent
            tts,
            self._transport.output(),
            assistant_aggregator,
        ])

        self._task = PipelineTask(pipeline, params=PipelineParams())
        return pipeline

    # name= 是关键字参数（agent_ready 签名是 *, name: str）
    # handler 会被调用为 handler(data)，所以必须接收 data: AgentReadyData
    @agent_ready(name="greeter")
    async def on_greeter_ready(self, data: AgentReadyData):
        """等 GreeterAgent 准备好后，激活它并让它先开口打招呼"""
        logger.info("Greeter agent ready, activating...")
        # 关键：必须传 args + messages，greeter 的 LLM 才会运行、主动问好。
        # 源码 LLMAgent.on_activated 里只有 `if activation.messages:` 时才会
        # queue LLMMessagesAppendFrame(run_llm=True)。不传 messages 的话
        # greeter 虽然被激活，但 LLM 永远不跑 → bot 一直静默等你先说话。
        await self.activate_agent(
            "greeter",
            args=LLMAgentActivationArgs(
                messages=[{
                    "role": "user",
                    "content": "Greet the user warmly and ask how you can help.",
                }],
                run_llm=True,
            ),
        )


async def main():
    # AgentRunner：管理所有 agent 的生命周期
    # 默认用 AsyncQueueBus（in-process，不需要 Redis）
    runner = AgentRunner(handle_sigint=False if sys.platform == "win32" else True)

    # 建立所有 agent（共享 runner.bus）
    main_agent = MainAgent("main", bus=runner.bus)
    greeter = GreeterAgent("greeter", bus=runner.bus)
    support = SupportAgent("support", bus=runner.bus)

    # 把 main 加入 runner。run() 之前加的 root agent 会被 runner 暂存，
    # 在 run() 启动时统一拉起 —— 这条没问题。
    await runner.add_agent(main_agent)

    # ⚠️ 关键修复：子 agent 不能在 run() 之前加。
    # main_agent.add_agent(child) 会往 bus 发一条 BusAddAgentMessage，
    # 但 runner 是在 run() 里才 subscribe + start bus 的。run() 之前发的消息
    # 因为「没有订阅者」被 AgentBus.on_message_received 直接丢弃（for 空循环），
    # 于是 runner 永远不知道 greeter/support 的存在 → 它们的 pipeline 从不启动
    # → 永不 ready → on_greeter_ready 不触发 → bot 全程静默、连 log 都没有。
    #
    # 解决：等 main 的 pipeline 起来（on_ready 触发，此时 bus 已经在跑、runner
    # 已订阅）之后再 add_agent，消息才送得到 runner。
    @main_agent.event_handler("on_ready")
    async def _add_children(agent):
        logger.info("Main agent ready, adding child agents...")
        await agent.add_agent(greeter)
        await agent.add_agent(support)

    print("=" * 55)
    print(" Multi-Agent Demo")
    print(" Agents: MainAgent → GreeterAgent ↔ SupportAgent")
    print(" Greeter will welcome you, then transfer to Support")
    print(" Say 'goodbye' to end the conversation")
    print("=" * 55)

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
