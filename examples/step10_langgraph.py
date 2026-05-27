"""
Step 10 — LangGraph 集成（兼容，不是不兼容）
=============================================
LangGraph 和 Pipecat 是完全兼容的。
集成方式：写一个自定义 FrameProcessor 作为桥接，替换掉 pipeline 里的 LLM service。

你会学到：
    1. 为什么 Pipecat + LangGraph 是兼容的
    2. LangGraphProcessor — 桥接 FrameProcessor 的写法
    3. 关键帧协议：LLMContextFrame → LLMFullResponseStartFrame → LLMTextFrame × N → LLMFullResponseEndFrame
    4. Pipecat messages（OpenAI dict）和 LangGraph messages（LangChain Message 对象）之间的转换
    5. LangGraph graph 如何管理自己的对话 state

为什么要用 LangGraph 而不是直接用 Pipecat 内建 LLM service？
    - 你已经有一个 LangGraph workflow（有复杂的 conditional edges、tools、memory）
    - 想给它加上实时语音界面，而不重写整个 agent 逻辑
    - 需要 LangGraph 的 checkpointing、human-in-the-loop 等特性

局限：
    - 对话历史由 LangGraph MessagesState 管理，不用 Pipecat LLMContext
    - 如果需要 Pipecat 的 context summarization 或 function calling，需要额外适配
    - 不支持 streaming interruption（LangGraph 调用是原子的）

安装依赖：
    uv add langgraph langchain-openai langchain-core
    （不需要新的 Pipecat extras）

所需 API key：DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from typing import Annotated

from dotenv import load_dotenv
from loguru import logger

# ── LangGraph imports ──────────────────────────────────────────────────────
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ── Pipecat imports ────────────────────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# 1. LangGraph Graph 定义
#
# 这里是一个最简单的 chatbot graph：
#   用户输入 → chatbot 节点（调用 LLM）→ 结束
#
# 真实场景里这里可以是任何复杂的 LangGraph workflow：
#   - 有条件边（conditional edges）
#   - 多个节点（retrieval、tools、routing）
#   - 人工审核（human-in-the-loop）
#   - 持久化记忆（checkpointing）
# ═══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    # add_messages 是 LangGraph 内建的 reducer，自动把新消息追加到列表
    messages: Annotated[list, add_messages]


def build_langgraph() -> "CompiledStateGraph":
    """构建 LangGraph graph"""
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.7,
    )

    def chatbot_node(state: GraphState):
        """调用 LLM，返回回复"""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(GraphState)
    builder.add_node("chatbot", chatbot_node)
    builder.set_entry_point("chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
# 2. LangGraphProcessor — 核心桥接类
#
# 这个类遵循 Pipecat 官方的框架集成模式（和内建的 LangchainProcessor 相同）：
#
# 输入：LLMContextFrame（Pipecat 的对话历史）
# 输出：LLMFullResponseStartFrame → LLMTextFrame × N → LLMFullResponseEndFrame
#
# 关键帧要求（官方 Discord staff 确认）：
#   - 必须用 LLMTextFrame，不能用 TextFrame
#     LLMTextFrame 的 includes_inter_frame_spaces=True，TTS aggregator 需要这个
#   - 必须推送 LLMFullResponseEndFrame，否则 TTS 不会 flush 最后一句话
#   - LLMFullResponseStartFrame 让 transport 知道 bot 开始说话（影响 VAD 和中断逻辑）
# ═══════════════════════════════════════════════════════════════════════════

class LangGraphProcessor(FrameProcessor):
    """把 LangGraph graph 嵌入 Pipecat pipeline 的桥接处理器"""

    def __init__(self, graph):
        super().__init__()
        self._graph = graph
        # LangGraph 自己管理对话历史（不用 Pipecat 的 LLMContext）
        self._lg_messages: list = [
            SystemMessage(content=(
                "You are a helpful voice assistant. "
                "Keep responses short and conversational. "
                "No markdown, no bullet points."
            ))
        ]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            # LLMContextFrame 里的 messages 是 Pipecat 的格式（OpenAI dict list）
            # 我们只取最新一条用户消息，交给 LangGraph 处理
            pipecat_messages = frame.context.get_messages()
            last_message = pipecat_messages[-1] if pipecat_messages else None

            if not last_message or not isinstance(last_message, dict):
                await self.push_frame(frame, direction)
                return

            role = last_message.get("role", "")
            content = last_message.get("content", "")

            # 只处理用户消息
            if role not in ("user", "human") or not content.strip():
                await self.push_frame(frame, direction)
                return

            logger.info(f"LangGraphProcessor: user said: {content!r}")

            # 把用户消息加入 LangGraph 的消息历史
            self._lg_messages.append(HumanMessage(content=content.strip()))

            # ── 推送帧序列：告诉 Pipecat bot 开始回复 ──────────────────────
            await self.push_frame(LLMFullResponseStartFrame())

            try:
                # 调用 LangGraph graph，获取完整回复
                # 注意：这里用 ainvoke（非流式），因为 LangGraph streaming 需要额外配置
                result = await self._graph.ainvoke({"messages": self._lg_messages})

                # 从 result 里取出 AI 的回复
                response_messages = result.get("messages", [])
                ai_message = None
                for msg in reversed(response_messages):
                    if isinstance(msg, AIMessage):
                        ai_message = msg
                        break

                if ai_message and ai_message.content:
                    response_text = ai_message.content
                    logger.info(f"LangGraphProcessor: LangGraph replied: {response_text!r}")

                    # 把 AI 回复加入 LangGraph 的历史，下轮对话用
                    self._lg_messages.append(ai_message)

                    # 把回复分成小块推送（模拟 streaming）
                    # 真实场景里可以用 graph.astream() 实现真正的 token streaming
                    words = response_text.split()
                    chunk_size = 5  # 每次推 5 个词
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        # 在 chunk 末尾加空格，确保 TTS aggregator 正确拼接句子
                        await self.push_frame(LLMTextFrame(chunk + " "))

            except Exception as e:
                logger.error(f"LangGraphProcessor error: {e}")
            finally:
                # ── 必须推送 EndFrame，否则 TTS 不会 flush 最后一句 ──────────
                await self.push_frame(LLMFullResponseEndFrame())

        else:
            # 所有其他 frame 直接透传
            await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 主程序
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    # 构建 LangGraph graph
    graph = build_langgraph()
    logger.info("LangGraph graph compiled successfully")

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

    # ── LangGraphProcessor 替代 OpenAILLMService ──────────────────────────
    langgraph_processor = LangGraphProcessor(graph)

    # Pipecat 的 context 在这里只用来传递用户消息给 LangGraphProcessor
    # 对话历史由 LangGraph 的 MessagesState 自己管理
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # Pipeline：用 LangGraphProcessor 替换了 OpenAILLMService
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        langgraph_processor,   # ← LangGraph 在这里
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    # 启动时让 agent 先说话（直接加消息到 LangGraph 历史，然后触发）
    langgraph_processor._lg_messages.append(
        HumanMessage(content="Please greet the user briefly.")
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" LangGraph + Pipecat Voice Agent")
    print(" LangGraph graph: simple chatbot (可换成任何 graph)")
    print(" Try: 'What is LangGraph?' or 'Tell me a joke'")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
