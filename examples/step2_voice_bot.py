"""
Step 2 — 完整本地语音 Agent
================================
需要：Deepgram + OpenAI + Cartesia API keys（三个都要）。
运行后，直接对着麦克风说话就可以和 agent 对话。
Ctrl+C 结束程序。

你会学到：
    1. 完整的 STT → LLM → TTS 管道
    2. VAD（Voice Activity Detection）—— 判断你什么时候说完
    3. LLMContext & Aggregators —— 怎么管理对话历史
    4. LocalAudioTransport 的双向音频
    5. LLMRunFrame —— 主动触发 LLM

安装依赖：
    pip install "pipecat-ai[local,deepgram,openai,cartesia,silero]" python-dotenv loguru

配置：
    复制 .env.example 为 .env，填入三个 API key
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# VAD：检测用户说话的开始和结束
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Frames：数据的"容器"
from pipecat.frames.frames import LLMRunFrame

# Pipeline：处理器的链条
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Context：存对话历史
from pipecat.processors.aggregators.llm_context import LLMContext

# Aggregators：积累转录/回复到 context
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# AlwaysUserMuteStrategy：Pipecat 内建策略，bot 说话时静音用户输入
# 官方文档：https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-mute-strategies
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# AI Services
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

# 本地音频 Transport
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


#   ┌───────────────────────────┬──────────────────────────────────────────────┬───────────────────────────────┐
#   │           场景            │                     方案                     │             效果              │
#   ├───────────────────────────┼──────────────────────────────────────────────┼───────────────────────────────┤
#   │ LocalAudio + 喇叭（现在） │ BotSpeakingUserMuteStrategy                  │ 无回声，但 bot 说话时无法打断 │
#   ├───────────────────────────┼──────────────────────────────────────────────┼───────────────────────────────┤
#   │ LocalAudio + 耳机         │ 去掉 mute strategy，allow_interruptions=True │ 无回声 + 可以打断             │
#   ├───────────────────────────┼──────────────────────────────────────────────┼───────────────────────────────┤
#   │ Web transport（step5）    │ Daily / WebRTC，浏览器内建 AEC               │ 无回声 + 可以打断             │
#   └───────────────────────────┴──────────────────────────────────────────────┴───────────────────────────────┘


async def main():
    # ═══════════════════════════════════════════════════════════════════════
    # 1. TRANSPORT
    # 用电脑的麦克风（input）和喇叭（output）
    # ═══════════════════════════════════════════════════════════════════════
    # 列出设备编号：
    #   uv run python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
    # 把你的实体麦克风编号填入 input_device_index（None = 用系统默认）
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # device 1 = Microphone Array on SoundWire（实体麦克风）
            # 避免用 device 0 (Sound Mapper) ── 它可能映射到 loopback 设备
            # 避免用 device 12/16 (Input SoundWire Speaker) ── 那是录系统声音的 loopback
            input_device_index=1,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. SERVICES（三个 AI 服务）
    # ═══════════════════════════════════════════════════════════════════════

    # STT：把你的语音变成文字
    # 输入：AudioRawFrame  →  输出：TranscriptionFrame（转录文字）
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # TTS：把 LLM 的文字回复变成语音
    # 输入：TextFrame  →  输出：AudioRawFrame（语音音频）
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM") 
    )

    # LLM：处理对话，生成回复
    # 输入：LLMContextFrame  →  输出：TextFrame（回复文字）
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",  # 便宜快速，学习用
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Keep your responses short and conversational (1-3 sentences). "
                "Do not use bullet points, markdown, or emojis."
            ),
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CONTEXT & AGGREGATORS
    # ═══════════════════════════════════════════════════════════════════════

    # LLMContext：存对话历史（就是 messages list，和 OpenAI API 格式一样）
    context = LLMContext()

    # LLMContextAggregatorPair 返回两个处理器：
    #
    # user_aggregator：
    #   - 放在 STT 后面
    #   - 监听 TranscriptionFrame（转录）
    #   - 用 SileroVADAnalyzer 判断你说完了没
    #   - 说完后把完整句子加入 context，发 LLMContextFrame 触发 LLM
    #
    # assistant_aggregator：
    #   - 放在 transport.output() 后面
    #   - 收集 LLM 说的所有 TextFrame，等说完了加入 context
    #   - 这样下一轮 LLM 就知道自己上次说了什么
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # 防回声：bot 说话时静音用户输入，bot 停说后 0.5s 恢复
            # 想支持真正的 barge-in → 换成 Daily/WebRTC transport（浏览器内建 AEC）
            # bot 说话时静音麦克风，防止回声触发 VAD 打断自己
            # 副作用：用户在 bot 说话时也无法 barge-in
            # 真正支持 barge-in → 用 Web transport（Daily/WebRTC，浏览器内建 AEC）
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PIPELINE（核心数据流）
    # ═══════════════════════════════════════════════════════════════════════
    #
    # 完整的数据流向：
    #
    #  [麦克风] ──AudioRawFrame──► transport.input()
    #                                     │
    #                              AudioRawFrame
    #                                     │
    #                                    stt  ◄── Deepgram 实时转录
    #                                     │
    #                           TranscriptionFrame
    #                                     │
    #                            user_aggregator  ◄── 等你说完 + 积累 context
    #                                     │
    #                           LLMContextFrame（完整对话历史）
    #                                     │
    #                                    llm  ◄── OpenAI 处理
    #                                     │
    #                               TextFrame（回复文字，流式）
    #                                     │
    #                                    tts  ◄── Cartesia 合成
    #                                     │
    #                             AudioRawFrame（语音）
    #                                     │
    #                           transport.output()
    #                                     │
    #                               [喇叭播放] ──► assistant_aggregator
    #                                                  │
    #                                         记录到 context，下轮 LLM 用
    pipeline = Pipeline([
        transport.input(),       # 从麦克风接收音频
        stt,                     # 语音 → 文字
        user_aggregator,         # 积累用户说的话
        llm,                     # 生成回复
        tts,                     # 文字 → 语音
        transport.output(),      # 播放到喇叭
        assistant_aggregator,    # 记录 agent 说的话
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # 5. PIPELINE TASK
    # ═══════════════════════════════════════════════════════════════════════
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. 启动时让 AGENT 先开口
    # ═══════════════════════════════════════════════════════════════════════
    # 给 context 加一条系统消息，告诉 LLM 先介绍自己
    context.add_message({
        "role": "developer",
        "content": "Please greet the user and briefly introduce yourself. Keep it under 2 sentences."
    })
    # LLMRunFrame：立刻触发 LLM 处理 context（不等用户先说话）
    await task.queue_frames([LLMRunFrame()])

    # ═══════════════════════════════════════════════════════════════════════
    # 7. RUN
    # ═══════════════════════════════════════════════════════════════════════
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("\n🎤 Voice agent is running! Speak into your microphone.")
    print("   Press Ctrl+C to stop.\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
