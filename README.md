# Pipecat Learning Guide

> **Language:** English | [中文版](README_CN.md)

## Table of Contents
1. [What is Pipecat](#1-what-is-pipecat)
2. [Core Architecture Concepts](#2-core-architecture-concepts)
3. [Learning Roadmap](#3-learning-roadmap-19-examples)
4. [Which Libraries to Install](#4-which-libraries-to-install)
5. [What is pipecat-ai-cli](#5-what-is-pipecat-ai-cli)
6. [What is MCP](#6-what-is-mcp)
7. [Important Gotchas & Official Corrections](#7-important-gotchas--official-corrections)

---

## 1. What is Pipecat

Pipecat is an open-source Python framework designed specifically for building **real-time voice and multimodal AI agents**.  
Initiated by Daily.co, the core idea is to connect the chain — audio input → STT → LLM → TTS → audio output — using standardized "pipelines".

Supports 100+ AI services (OpenAI, Anthropic, Deepgram, ElevenLabs, etc.).

**Pipecat Ecosystem:**

| Component | Description | Should You Learn It |
|---|---|---|
| **Pipecat Framework** | Core Python framework | ✅ Primary |
| **Pipecat Flows** | Structured conversation state machine | ✅ step7 |
| **Client SDKs** | JS/React/iOS/Android clients | Advanced |
| **Pipecat Subagents** | Multi-agent collaboration | Advanced |
| **Pipecat Cloud** | Managed deployment | Skip for now |

---

## 2. Core Architecture Concepts

### Frame — Data Container
All information is wrapped in Frames and flows through the Pipeline.

```
AudioRawFrame       → raw audio
TranscriptionFrame  → STT transcription result
TextFrame           → text content
LLMRunFrame         → immediately triggers the LLM
TTSSpeakFrame       → tells TTS to speak this text
EndFrame            → ends the pipeline
BotStartedSpeakingFrame / BotStoppedSpeakingFrame  → bot speaking state (UPSTREAM)
LLMMessagesAppendFrame  → appends a message to context (without interrupting conversation)
LLMMessagesUpdateFrame  → fully replaces context (used for persona switching)
```

### Pipeline — Processing Chain
```python
pipeline = Pipeline([
    transport.input(),   # microphone audio
    stt,                 # AudioRawFrame → TranscriptionFrame
    user_aggregator,     # accumulates transcription, waits for silence, triggers LLM
    llm,                 # LLMContextFrame → TextFrame
    tts,                 # TextFrame → AudioRawFrame
    transport.output(),  # plays audio
    assistant_aggregator # records response into context
])
```

### Transport — Audio Gateway
| Transport | Use Case | Account Required | Barge-in Support |
|---|---|---|---|
| `LocalAudioTransport` | Local mic/speaker | Not required | ⚠️ Needs headphones or AlwaysUserMuteStrategy |
| `SmallWebRTC` (`--transport webrtc`) | Local dev / self-hosted, P2P | Not required | ✅ Browser built-in AEC |
| `DailyTransport` (`--transport daily`) | Production, global users | Daily account required | ✅ Daily managed AEC, more robust |
| `FastAPIWebsocket` | Phone (Twilio) / server-to-server | Depends on provider | ⚠️ No AEC |

**SmallWebRTC vs Daily** (from official docs):
- SmallWebRTC = P2P direct connection, no account needed, preferred for self-hosting
- Daily = 75 global PoPs, network relay, preferred for production and global users
- step5 supports both: `--transport webrtc` uses SmallWebRTC

### Services — AI Services
- **STT**: Deepgram, OpenAI Whisper, AssemblyAI, etc.
- **LLM**: OpenAI, Anthropic Claude, Google Gemini, etc.
- **TTS**: ElevenLabs, Cartesia, OpenAI TTS, etc.

### Context & Aggregators — Conversation Memory
```python
context = LLMContext()          # stores messages list
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_mute_strategies=[AlwaysUserMuteStrategy()],  # prevents echo
    ),
)
```

### FrameProcessor — Custom Processor
```python
class MyProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # check frame type and handle logic
        if isinstance(frame, TranscriptionFrame):
            print(frame.text)
        await self.push_frame(frame, direction)  # pass to next processor
        # if push_frame is not called → frame is dropped (filtered out)
```

---

## 3. Learning Roadmap (19 Examples)

### ✅ Level 1 — Get the Basics Running
| File | What You Learn | Keys Required |
|---|---|---|
| `step1_hello.py` | Pipeline, TTSSpeakFrame, EndFrame, Transport | ElevenLabs |
| `step2_voice_bot.py` | STT+LLM+TTS, VAD, Context, Aggregators, AlwaysUserMuteStrategy | Deepgram + OpenAI + ElevenLabs |
| `step3_function_calling.py` | FunctionSchema, ToolsSchema, register_function, event handler | Same as above |

### ✅ Level 2 — Customization & Intermediate
| File | What You Learn | Keys Required |
|---|---|---|
| `step4_custom_processor.py` | FrameProcessor, intercept/modify/filter frames, session reset | Same as above |
| `step5_web_transport.py` | Pipecat Runner, transport_params, browser access, event handling | Same as above (Daily optional) |
| `step6_context_injection.py` | LLMMessagesAppendFrame, LLMMessagesUpdateFrame, persona switching | Same as above |

### 📖 Level 3 — Structured Flows & Tool Integration
| File | What You Learn | Keys Required |
|---|---|---|
| `step7_pipecat_flows.py` | NodeConfig, FlowManager, Edge function, flow_manager.state | Same as above |
| `step8_mcp_in_bot.py` | MCPClient, register_tools, StdioServerParameters, tools_filter | Same as above |
| `step9_multi_agent.py` | AgentRunner, BusBridgeProcessor, LLMAgent, handoff_to, @tool | Same as above |
| `step10_langgraph.py` | LangGraphProcessor, LLMContextFrame, LLMTextFrame frame protocol | Same as above |

### 🔭 Level 4 — Observability
| File | What You Learn | Keys Required |
|---|---|---|
| `step11_observers.py` | BaseObserver, LLMLogObserver, TranscriptionLogObserver, MetricsLogObserver | Same as above |
| `step12_per_stage_metrics.py` | TTFBMetricsData, ProcessingMetricsData, per-stage timing table | Same as above |
| `step13_full_observability.py` | UserBotLatencyObserver, TurnTrackingObserver, StartupTimingObserver | Same as above |

### 🚀 Level 5 — Advanced Features
| File | What You Learn | Keys Required |
|---|---|---|
| `step14_speech_to_speech_openai.py` | OpenAIRealtimeLLMService, SessionProperties, SemanticTurnDetection | OpenAI (Realtime) |
| `step15_speech_to_speech_gemini.py` | GeminiLiveLLMService, GeminiLiveParams | Google |
| `step16_llm_switcher.py` | LLMSwitcher, ManuallySwitchServiceFrame, ServiceSwitcherStrategyFailover | OpenAI |
| `step17_context_summarization.py` | enable_auto_context_summarization, LLMAutoContextSummarizationConfig | Same as above |
| `step18_multimodal.py` | LLMContext.create_image_url_message, images into context | Same as above |
| `step19_modular_openai.py` | OpenAISTTService + OpenAITTSService, all-OpenAI modular pipeline, single API key (contrast with step14 speech-to-speech) | **OpenAI only** |

> **Modular vs Speech-to-Speech**: step19 builds the same 3-stage STT→LLM→TTS pipeline as step2, but uses OpenAI for *all three* services — so it needs only one `OPENAI_API_KEY`. step14 does it with a single Realtime model (lower latency, higher cost). step19 = modular & swappable & cheap; step14 = unified & fast.

> **Twilio + OpenAI Realtime (production phone version)**: `C:\Users\Yuki.Leong\github\twilio`
> Includes WebSocket server + Twilio webhook + full phone integration

### Run Order
```bash
# Level 1-2 (basics)
python examples/step1_hello.py
python examples/step2_voice_bot.py
python examples/step3_function_calling.py
python examples/step4_custom_processor.py
uv run python examples/step5_web_transport.py --transport webrtc
python examples/step6_context_injection.py

# Level 3 (advanced features)
python examples/step7_pipecat_flows.py          # uv add pipecat-ai-flows
python examples/step8_mcp_in_bot.py             # uv add "pipecat-ai[mcp]"
python examples/step9_multi_agent.py            # uv add pipecat-ai-subagents
python examples/step10_langgraph.py             # uv add langgraph langchain-openai

# Level 4 (observability)
python examples/step11_observers.py
python examples/step12_per_stage_metrics.py
python examples/step13_full_observability.py

# Level 5 (advanced features)
python examples/step14_speech_to_speech_openai.py   # requires Realtime API access
python examples/step15_speech_to_speech_gemini.py   # GOOGLE_API_KEY + uv add "pipecat-ai[google]"
python examples/step16_llm_switcher.py
python examples/step17_context_summarization.py
python examples/step18_multimodal.py
python examples/step19_modular_openai.py        # all OpenAI STT+LLM+TTS, only needs OPENAI_API_KEY
```

---

## 4. Which Libraries to Install

### Base Installation (Steps 1-6)
```bash
uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]" python-dotenv loguru
```

### step7 requires additional install
```bash
uv add pipecat-ai-flows
```

### step8 requires additional install
```bash
uv add "pipecat-ai[mcp]"
# mcp-server-time is handled automatically by uvx, no manual install needed
```

### step9 requires additional install
```bash
uv add pipecat-ai-subagents
```

### step10 requires additional install
```bash
uv add langgraph langchain-openai langchain-core
```

### step15 requires additional install
```bash
uv add "pipecat-ai[google]"
# GOOGLE_API_KEY free signup: https://aistudio.google.com/apikey
```

### step14 notes
Requires OpenAI Realtime API access (`gpt-4o-realtime-preview`).  
Realtime API is more expensive than regular OpenAI API — billed per audio minute.  
Your production Twilio version: `C:\Users\Yuki.Leong\github\twilio`

### step19 notes (all-OpenAI modular pipeline)
```bash
# No extra services needed — only the OpenAI extra and a local mic/speaker
uv add "pipecat-ai[local,openai,silero]"
```
Uses `OpenAISTTService` (gpt-4o-transcribe) + `OpenAILLMService` (gpt-4o-mini) + `OpenAITTSService` (gpt-4o-mini-tts).  
Only `OPENAI_API_KEY` is required — no Deepgram / ElevenLabs / Cartesia keys.  
OpenAI STT here is **segmented** (REST): it transcribes each utterance after VAD detects you stopped talking, so it's slightly higher-latency than streaming Deepgram. For lower latency, swap in `OpenAIRealtimeSTTService`.

### step5 with Daily transport
```bash
uv add "pipecat-ai[daily]"
# and add to .env: DAILY_API_KEY=...
```

### Extras Reference
| Extra | Description |
|---|---|
| `local` | PyAudio, local mic/speaker |
| `deepgram` | Deepgram STT |
| `openai` | OpenAI LLM / TTS |
| `elevenlabs` | ElevenLabs TTS |
| `cartesia` | Cartesia TTS |
| `silero` | Silero VAD (local voice activity detection) |
| `daily` | Daily WebRTC transport |
| `mcp` | MCP client (connect to MCP servers) |
| `anthropic` | Anthropic Claude LLM |

---

## 5. What is pipecat-ai-cli

**`pipecat-ai-cli`** is a standalone command-line tool — it is a separate package from `pipecat-ai` (the framework).

```bash
uv tool install pipecat-ai-cli
```

| Command | Purpose |
|---|---|
| `pipecat init <project-name>` | Generate project template (bot.py, .env, etc.) |
| `pipecat cloud auth login` | Log in to Pipecat Cloud |
| `pipecat cloud deploy` | Deploy to Pipecat Cloud |

**Not needed during learning** — come back to this when you're ready to deploy.

---

## 6. What is MCP

### MCP in Claude Code (tool assistance)
```bash
claude mcp add --transport http pipecat-docs https://daily-docs.mcp.kapa.ai
```
Adds Pipecat documentation lookup capability to Claude Code (AI assistant). Already configured — takes effect after restarting the session.

### MCP in a Pipecat Bot (step8)

MCP = **Model Context Protocol**, an open standard that lets LLMs call external tools.

```
Common MCP servers:
  mcp-server-time       → time/timezone queries
  mcp-server-filesystem → filesystem access
  mcp-server-fetch      → web scraping
  GitHub MCP            → GitHub operations
  Brave Search MCP      → web search
```

Connect in Pipecat via `MCPClient`:

```python
async with MCPClient(server_params=StdioServerParameters(...)) as mcp:
    tools = await mcp.register_tools(llm)    # auto-discovers and registers tools
    context = LLMContext(tools=tools)        # pass to context so LLM can call them
```

Compared to the manual `FunctionSchema` approach in step3, `MCPClient` advantages:
- No need to write schemas by hand
- Register all tools in one line
- Can connect to any MCP-compatible service

---

## 7. Important Gotchas & Official Corrections

### `allow_interruptions` is deprecated
From an official contributor:
> *"allow_interruptions is deprecated and generally not well suited for voice AI applications."*

**Do not use** `PipelineParams(allow_interruptions=False/True)`.  
**Instead use** `user_mute_strategies` to control user input:

```python
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

user_params=LLMUserAggregatorParams(
    vad_analyzer=SileroVADAnalyzer(),
    user_mute_strategies=[AlwaysUserMuteStrategy()],
)
```

### LocalAudio + Speaker Echo Problem
- **Root cause**: the laptop microphone physically picks up speaker audio
- **Best solution**: use headphones (no echo + supports barge-in)
- **Code solution**: `AlwaysUserMuteStrategy` (prevents echo interruptions, but user also can't interrupt while bot is speaking)
- **Best experience**: use Web transport (Daily/WebRTC) — browser built-in AEC, supports barge-in simultaneously

### LocalAudio Device Selection (Windows)
If `device 0 (Sound Mapper)` maps to a loopback device, the bot will hear itself speaking:
```python
transport = LocalAudioTransport(
    LocalAudioTransportParams(
        input_device_index=1,  # physical mic, avoid loopback
        ...
    )
)
# List devices:
# python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

### AlwaysUserMuteStrategy Built-in Strategies
```python
from pipecat.turns.user_mute import (
    AlwaysUserMuteStrategy,              # bot speaking → mute user
    FirstSpeechUserMuteStrategy,         # only mute on first bot speech
    MuteUntilFirstBotCompleteUserMuteStrategy,  # stay muted until first bot finishes
    FunctionCallUserMuteStrategy,        # mute during tool calls
)
```

---

## Reference Resources

| Resource | URL |
|---|---|
| Official Docs | https://docs.pipecat.ai |
| GitHub | https://github.com/pipecat-ai/pipecat |
| API Reference | https://reference-server.pipecat.ai |
| Pipecat Flows | https://github.com/pipecat-ai/pipecat-flows |
| Official Examples | https://github.com/pipecat-ai/pipecat/tree/main/examples |
| Discord | https://discord.gg/pipecat |
| Visual Flows Editor | https://flows.pipecat.ai |
