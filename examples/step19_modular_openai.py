"""
Step 19 — Modular Pipeline：全部用 OpenAI（STT + LLM + TTS）
==============================================================
和 step2 一样是"三段式"语音 Agent，但三个服务全部换成 OpenAI，
只需要 **一个** OPENAI_API_KEY 就能跑完整条 pipeline。

两种用 OpenAI 做语音 Agent 的方式：

    Modular（本例，step19）        Speech-to-Speech（step14）
    ────────────────────────       ──────────────────────────
    mic → OpenAI STT               mic → OpenAI Realtime → speaker
        → OpenAI LLM                          ↑
        → OpenAI TTS               一个模型同时做 STT+LLM+TTS
        → speaker                  延迟低（~300ms），但贵
    三个独立模型，可单独替换/调试    黑盒，难插入自定义逻辑
    延迟略高（~800ms），便宜        需要 Realtime API 访问权限

为什么叫 "modular"（模块化）：
    STT / LLM / TTS 是三个独立的处理器，可以各自替换。
    想换 STT？把 OpenAISTTService 换成 DeepgramSTTService 即可（step2 就是混搭）。
    想换 TTS？换成 ElevenLabsTTSService / CartesiaTTSService 即可。
    本例展示的是"全 OpenAI"这一种组合 —— 单一供应商、单一 key、单一账单。

你会学到：
    1. OpenAISTTService —— OpenAI 的语音转文字（gpt-4o-transcribe）
    2. OpenAITTSService —— OpenAI 的文字转语音（gpt-4o-mini-tts）
    3. 三个服务共用一个 OPENAI_API_KEY
    4. OpenAI STT 是 "segmented"（按句转录）：靠 VAD 切句子，
       aggregator 上的 VAD 会把 speaking 事件向上游广播给 STT

安装依赖：
    uv add "pipecat-ai[local,openai,silero]" python-dotenv loguru
    （不需要 deepgram / elevenlabs / cartesia）

所需 API key：只要 OPENAI_API_KEY 一个

配置：
    复制 .env.example 为 .env，填入 OPENAI_API_KEY
    Ctrl+C 结束程序
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# VAD：检测用户说话的开始和结束（OpenAI segmented STT 靠它切句子）
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

# AlwaysUserMuteStrategy：bot 说话时静音用户输入，防止扬声器回声打断自己
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# ── 三个 AI 服务，全部来自 OpenAI ──────────────────────────────────────────
from pipecat.services.openai.stt import OpenAISTTService   # 语音 → 文字
from pipecat.services.openai.llm import OpenAILLMService   # 文字 → 回复
from pipecat.services.openai.tts import OpenAITTSService   # 文字 → 语音

# 语言枚举（给 STT 指定输入语言）
from pipecat.transcriptions.language import Language

# 本地音频 Transport（电脑麦克风 + 喇叭）
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    # ═══════════════════════════════════════════════════════════════════════
    # 1. TRANSPORT —— 本地麦克风（input）+ 喇叭（output）
    # ═══════════════════════════════════════════════════════════════════════
    # 列出设备编号：
    #   uv run python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
    # 把实体麦克风编号填入 input_device_index（None = 系统默认）
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=1,  # 实体麦克风，避免 loopback 设备（device 0 / Sound Mapper）
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. SERVICES —— 三个全是 OpenAI，共用同一个 OPENAI_API_KEY
    # ═══════════════════════════════════════════════════════════════════════
    api_key = os.environ["OPENAI_API_KEY"]

    # STT：把语音变成文字
    # 输入：AudioRawFrame  →  输出：TranscriptionFrame
    # OpenAI STT 是 REST "segmented" 模式：VAD 判断你说完一句后，
    # 把这一段音频整体发给 gpt-4o-transcribe 转录（不是逐字流式）。
    stt = OpenAISTTService(
        api_key=api_key,
        settings=OpenAISTTService.Settings(
            model="gpt-4o-transcribe",  # 也可用 "whisper-1" / "gpt-4o-mini-transcribe"
            language=Language.EN,
        ),
    )

    # LLM：处理对话，生成回复
    # 输入：LLMContextFrame  →  输出：TextFrame（流式）
    llm = OpenAILLMService(
        api_key=api_key,
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",  # 便宜快速，学习用
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Keep your responses short and conversational (1-3 sentences). "
                "Do not use bullet points, markdown, or emojis."
            ),
        ),
    )

    # TTS：把 LLM 的文字回复变成语音
    # 输入：TextFrame  →  输出：TTSAudioRawFrame（24kHz PCM）
    # 可选 voice：alloy / ash / ballad / cedar / coral / echo / fable /
    #            marin / nova / onyx / sage / shimmer / verse
    tts = OpenAITTSService(
        api_key=api_key,
        settings=OpenAITTSService.Settings(
            model="gpt-4o-mini-tts",
            voice="alloy",
            # instructions：gpt-4o-mini-tts 支持"演技指导"，控制语气/情绪
            instructions="Speak in a warm, friendly and natural tone.",
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CONTEXT & AGGREGATORS
    # ═══════════════════════════════════════════════════════════════════════
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # VAD 放在 aggregator 上：它检测到"说话开始/结束"后，会把
            # VADUserStarted/StoppedSpeakingFrame 向【上游】广播。
            # 上游的 OpenAI segmented STT 收到这些帧，才知道何时把缓存的
            # 音频整段发去转录。所以这里的 VAD 同时驱动了 STT 切句 + 触发 LLM。
            vad_analyzer=SileroVADAnalyzer(),
            # 防回声：bot 说话时静音麦克风（用扬声器时必备；戴耳机可去掉以支持打断）
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PIPELINE —— 和 step2 结构完全一样，只是三个服务都换成了 OpenAI
    # ═══════════════════════════════════════════════════════════════════════
    #  [麦克风] → transport.input() → stt → user_aggregator → llm → tts
    #           → transport.output() → [喇叭] → assistant_aggregator
    pipeline = Pipeline([
        transport.input(),       # 从麦克风接收音频
        stt,                     # OpenAI: 语音 → 文字
        user_aggregator,         # 积累用户说的话（VAD 在这里）
        llm,                     # OpenAI: 生成回复
        tts,                     # OpenAI: 文字 → 语音
        transport.output(),      # 播放到喇叭
        assistant_aggregator,    # 记录 agent 说的话
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # 5. PIPELINE TASK
    # ═══════════════════════════════════════════════════════════════════════
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. 启动时让 AGENT 先开口
    # ═══════════════════════════════════════════════════════════════════════
    context.add_message({
        "role": "developer",
        "content": "Please greet the user and briefly introduce yourself. Keep it under 2 sentences.",
    })
    await task.queue_frames([LLMRunFrame()])

    # ═══════════════════════════════════════════════════════════════════════
    # 7. RUN
    # ═══════════════════════════════════════════════════════════════════════
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("\n🎤 Modular OpenAI voice agent is running! Speak into your microphone.")
    print("   STT + LLM + TTS 全部由 OpenAI 提供，只用一个 API key。")
    print("   Press Ctrl+C to stop.\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
