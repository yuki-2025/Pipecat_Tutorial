"""
Step 4 — 自定义 FrameProcessor
================================
在 Pipecat 里，STT / LLM / TTS / VAD 全都是 FrameProcessor。
这一步学会自己写 FrameProcessor，插入 pipeline 任意位置。

你会学到：
    1. FrameProcessor 的基本结构和生命周期
    2. process_frame(frame, direction) ── 所有 frame 都经过这里
    3. push_frame(frame) ── 把 frame 传给下一个处理器
    4. "吞掉" frame ── 不 push_frame 就等于过滤掉
    5. 三个实际例子：
       - TranscriptionPrinter   把用户说的话打印到终端
       - ConversationLogger     把对话存成 JSON 文件
       - ConversationResetter   说 "reset" 就清空对话历史

运行方式：
    uv run python examples/step4_custom_processor.py

所需 API key（和 step2 一样）：DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
# ── 核心 import：写自定义 processor 需要这两个 ──
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# Processor 1：TranscriptionPrinter
#
# 插入位置：stt → [TranscriptionPrinter] → user_aggregator
# 功能：每次 STT 有转录结果，就打印到终端
#
# 关键概念：
#   - 继承 FrameProcessor
#   - 重写 process_frame，检查 frame 类型
#   - 最后一定要 push_frame，不然 frame 不会往下走
# ═══════════════════════════════════════════════════════════════════════════
class TranscriptionPrinter(FrameProcessor):

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # 1. 先调 super()，让基类处理系统级 frame（如 StartFrame、EndFrame）
        await super().process_frame(frame, direction)

        # 2. 检查 frame 类型，TranscriptionFrame = STT 的最终转录结果
        if isinstance(frame, TranscriptionFrame):
            print(f"\n👤 YOU: {frame.text}")

        # 3. 把 frame 传给下一个处理器
        #    如果不 push_frame，frame 在这里消失（= 被过滤掉）
        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# Processor 2：ConversationLogger
#
# 插入位置：llm → [ConversationLogger] → tts
# 功能：把 LLM 的回复累积成完整句子，打印并存到 JSON 文件
#
# 关键概念：
#   - LLM 输出是"流式"的：一句话 = 很多个 TextFrame 或 LLMTextFrame
#   - 用 buffer 把碎片拼成完整句子
#   - FrameDirection.DOWNSTREAM = frame 从上游往下流（正常方向）
# ═══════════════════════════════════════════════════════════════════════════
class ConversationLogger(FrameProcessor):

    def __init__(self, log_file="conversation_log.json"):
        super().__init__()
        self._log_file = log_file
        self._log = []
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # LLM 输出的文字 frame：TextFrame 或 LLMTextFrame
        # direction == DOWNSTREAM 确保只处理从上游来的 frame（避免重复）
        if isinstance(frame, (TextFrame, LLMTextFrame)) and direction == FrameDirection.DOWNSTREAM:
            self._buffer += frame.text

            # 遇到句子结束符，就认为一句话说完了
            if self._buffer.strip() and frame.text.endswith((".", "!", "?", "\n")):
                sentence = self._buffer.strip()
                print(f"🤖 BOT: {sentence}")
                self._log.append({
                    "timestamp": datetime.now().isoformat(),
                    "role": "assistant",
                    "text": sentence,
                })
                self._buffer = ""
                self._save_log()

        await self.push_frame(frame, direction)

    def _save_log(self):
        with open(self._log_file, "w", encoding="utf-8") as f:
            json.dump(self._log, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Processor 3：ConversationResetter
#
# 插入位置：stt → TranscriptionPrinter → [ConversationResetter] → user_aggregator
# 功能：用户说 "reset" 就清空对话历史，重新开始
#
# 关键概念：
#   - 可以在 FrameProcessor 里直接操作外部对象（context）
#   - 不 push_frame = 吞掉这个 frame（不让 "reset" 进入 LLM）
#   - 直接改 context.messages 再发 LLMRunFrame 触发新开场白
# ═══════════════════════════════════════════════════════════════════════════
class ConversationResetter(FrameProcessor):

    def __init__(self, context: LLMContext, task_ref: list):
        super().__init__()
        self._context = context
        # task_ref 是一个列表 [task]，用来间接引用 task（避免循环引用）
        self._task_ref = task_ref

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            if "reset" in frame.text.lower():
                # 清空对话历史，只保留系统指令
                self._context.messages = []
                self._context.add_message({
                    "role": "developer",
                    "content": "The conversation was just reset. Greet the user again briefly.",
                })
                print("\n[System] 🔄 Conversation reset!")

                # 触发 LLM 立刻执行（说新的开场白）
                task = self._task_ref[0]
                if task:
                    await task.queue_frames([LLMRunFrame()])

                # 吞掉这个 TranscriptionFrame，不让 "reset" 进 LLM context
                return

        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
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
                "You are a helpful assistant. Keep responses short. "
                "If the user asks you to reset, a separate system will handle it."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # task_ref 用列表包装，让 ConversationResetter 能在初始化后引用 task
    task_ref = [None]

    # 实例化自定义处理器
    transcription_printer = TranscriptionPrinter()
    resetter = ConversationResetter(context, task_ref)
    conversation_logger = ConversationLogger("conversation_log.json")

    # Pipeline 数据流：
    #   mic → stt → TranscriptionPrinter → ConversationResetter → user_agg
    #       → llm → ConversationLogger → tts → speaker → assistant_agg
    pipeline = Pipeline([
        transport.input(),
        stt,
        transcription_printer,   # ← Processor 1: 打印转录
        resetter,                # ← Processor 3: 检测 "reset" 命令
        user_aggregator,
        llm,
        conversation_logger,     # ← Processor 2: 记录 bot 回复
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=False),
    )
    task_ref[0] = task  # 让 ConversationResetter 能访问 task

    context.add_message({
        "role": "developer",
        "content": "Greet the user. Let them know they can say 'reset' to restart the conversation.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Custom Processor Demo")
    print(" 👤 YOU: (your speech appears here)")
    print(" 🤖 BOT: (bot responses appear here)")
    print(" Say 'reset' to clear conversation history")
    print(" Conversation saved to: conversation_log.json")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
