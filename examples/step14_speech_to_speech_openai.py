"""
Step 14 — Speech-to-Speech：OpenAI Realtime（LocalAudio 版）
=============================================================
传统 pipeline：mic → Deepgram STT → OpenAI LLM → ElevenLabs TTS → speaker
                    ↑                                              ↑
              独立 STT 服务                               独立 TTS 服务
              E2E 延迟 ~800ms

Speech-to-Speech：mic → OpenAI Realtime → speaker
                              ↑
                   一个服务同时做 STT + LLM + TTS
                   E2E 延迟 ~300ms

你会学到：
    1. OpenAIRealtimeLLMService — 一个服务取代三个（STT + LLM + TTS）
    2. SessionProperties — 配置 Realtime session（语音、VAD、转录等）
    3. SemanticTurnDetection — 比 Silero VAD 更智能的语义 turn 检测
    4. InputAudioNoiseReduction — 内建降噪（far_field = 扬声器场景）
    5. Pipeline 结构变化：不需要 Deepgram / ElevenLabs 了
    6. universal LLMContext + LLMContextAggregatorPair（和传统 pipeline 一样）

Pipeline 对比：
    传统：transport.input() → stt → user_agg → llm → tts → transport.output() → asst_agg
    S2S ：transport.input() → user_agg → [OpenAIRealtime] → transport.output() → asst_agg
                                              (STT+LLM+TTS 都在里面)

你的 Twilio 版本（生产级，带 WebSocket server）：
    C:\\Users\\Yuki.Leong\\github\\twilio

安装：
    uv add "pipecat-ai[local,openai,silero]"
    （不需要 deepgram 或 elevenlabs）

所需 API key：OPENAI_API_KEY（需要 Realtime API 访问权限：gpt-4o-realtime-preview）
注意：建议戴耳机，或开启 InputAudioNoiseReduction 减少回声
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

# ── OpenAI Realtime imports ───────────────────────────────────────────────
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

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

    # ── SessionProperties：配置 Realtime session ──────────────────────────
    # 这里集中配置所有行为：语音选择、VAD 策略、降噪、转录模型等
    session_properties = SessionProperties(
        instructions=(
            "You are a helpful voice assistant. "
            "Keep responses short and conversational. "
            "No markdown or bullet points."
        ),
        output_modalities=["audio"],  # 输出纯音频（不需要 TTS）
        audio=AudioConfiguration(
            input=AudioInput(
                # 内建转录：把用户说的话转成文字（可选，便于调试）
                transcription=InputAudioTranscription(
                    model="gpt-4o-transcribe"
                ),
                # 降噪：far_field 适合扬声器场景（mic 离扬声器较远）
                noise_reduction=InputAudioNoiseReduction(type="far_field"),
                # 语义 turn 检测：比 Silero VAD 更智能，理解语义边界
                # eagerness='low' = 等用户说完整句子再回复（减少误打断）
                turn_detection=SemanticTurnDetection(
                    eagerness="low",
                    interrupt_response=True,  # 允许用户打断 bot
                ),
            ),
            output=AudioOutput(
                voice="alloy",  # shimmer / echo / onyx / nova / fable / alloy
            ),
        ),
    )

    # ── OpenAI Realtime LLM Service ───────────────────────────────────────
    # 注意：这一个服务替代了传统 pipeline 里的 STT + LLM + TTS 三个服务
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-4o-realtime-preview",
            session_properties=session_properties,
        ),
    )

    # ── Context 和 Aggregators ────────────────────────────────────────────
    # 和传统 pipeline 完全相同！universal LLMContext 的优势之一
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # ── Pipeline ──────────────────────────────────────────────────────────
    # 注意没有 STT 和 TTS！OpenAI Realtime 内部处理了所有音频
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),    # 处理用户 context（不再需要 VAD 参数）
        llm,                          # ← 一个服务做三件事
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    context.add_message({
        "role": "user",
        "content": "Please greet the user and introduce yourself briefly.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Speech-to-Speech: OpenAI Realtime (LocalAudio)")
    print(" 一个模型做 STT + LLM + TTS，延迟 ~300ms")
    print(" 建议戴耳机 (echo 问题)")
    print(" Ctrl+C 结束")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
