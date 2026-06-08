# Pipecat 学习指南

> **语言：** 中文 | [English Version](README.md)

## 目录
1. [Pipecat 是什么](#1-pipecat-是什么)
2. [核心架构概念](#2-核心架构概念)
3. [学习路线图](#3-学习路线图-19-个-example)
4. [安装什么 Library](#4-安装什么-library)
5. [pipecat-ai-cli 是什么](#5-pipecat-ai-cli-是什么)
6. [MCP 是什么](#6-mcp-是什么)
7. [重要的坑和官方纠正](#7-重要的坑和官方纠正)

---

## 1. Pipecat 是什么

Pipecat 是一个开源 Python 框架，专门用来构建**实时语音和多模态 AI agent**。  
由 Daily.co 发起，核心思路：把 音频输入 → STT → LLM → TTS → 音频输出 这条链路，用标准化的"管道"串起来。

支持 100+ AI 服务 (OpenAI、Anthropic、Deepgram、ElevenLabs 等)。

**Pipecat 生态系统：**

| 组件 | 说明 | 是否需要学 |
|---|---|---|
| **Pipecat Framework** | 核心 Python 框架 | ✅ 主要 |
| **Pipecat Flows** | 结构化对话状态机 | ✅ step7 |
| **Client SDKs** | JS/React/iOS/Android 客户端 | 进阶 |
| **Pipecat Subagents** | 多 agent 协作 | 进阶 |
| **Pipecat Cloud** | 托管部署 | 跳过 |

---

## 2. 核心架构概念

### Frame — 数据容器
所有信息都被包装成 Frame 在 Pipeline 里流动。

```
AudioRawFrame       → 原始音频
TranscriptionFrame  → STT 转录结果
TextFrame           → 文字内容
LLMRunFrame         → 立刻触发 LLM
TTSSpeakFrame       → 让 TTS 读这段话
EndFrame            → 结束 pipeline
BotStartedSpeakingFrame / BotStoppedSpeakingFrame  → bot 说话状态（UPSTREAM）
LLMMessagesAppendFrame  → 追加消息到 context（不打断对话）
LLMMessagesUpdateFrame  → 完全替换 context（换 persona 用）
```

### Pipeline — 处理链
```python
pipeline = Pipeline([
    transport.input(),   # 麦克风音频
    stt,                 # AudioRawFrame → TranscriptionFrame
    user_aggregator,     # 积累转录，等说完，触发 LLM
    llm,                 # LLMContextFrame → TextFrame
    tts,                 # TextFrame → AudioRawFrame
    transport.output(),  # 播放语音
    assistant_aggregator # 记录回复到 context
])
```

### Transport — 音频入口
| Transport | 场景 | 需要账号 | 是否支持 barge-in |
|---|---|---|---|
| `LocalAudioTransport` | 本地麦克风/喇叭 | 不需要 | ⚠️ 需耳机或 AlwaysUserMuteStrategy |
| `SmallWebRTC` (`--transport webrtc`) | 本地开发 / 自部署，P2P | 不需要 | ✅ 浏览器内建 AEC |
| `DailyTransport` (`--transport daily`) | 生产环境，全球用户 | 需要 Daily 账号 | ✅ Daily 托管 AEC，更强 |
| `FastAPIWebsocket` | 电话 (Twilio) / 服务器间 | 看服务商 | ⚠️ 无 AEC |

**SmallWebRTC vs Daily**（来自官方文档）：
- SmallWebRTC = P2P 直连，无需任何账号，自部署首选
- Daily = 全球 75 个 PoP，网络中转，生产和全球用户首选
- step5 同时支持两种，`--transport webrtc` 用 SmallWebRTC

### Services — AI 服务
- **STT**: Deepgram, OpenAI Whisper, AssemblyAI 等
- **LLM**: OpenAI, Anthropic Claude, Google Gemini 等
- **TTS**: ElevenLabs, Cartesia, OpenAI TTS 等

### Context & Aggregators — 对话记忆
```python
context = LLMContext()          # 存 messages list
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_mute_strategies=[AlwaysUserMuteStrategy()],  # 防回声
    ),
)
```

### FrameProcessor — 自定义处理器
```python
class MyProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # 检查 frame 类型，处理逻辑
        if isinstance(frame, TranscriptionFrame):
            print(frame.text)
        await self.push_frame(frame, direction)  # 传给下一个处理器
        # 如果不 push_frame → frame 消失（过滤）
```

---

## 3. 学习路线图（19 个 example）

### ✅ Level 1 — 基础跑通
| 文件 | 学到什么 | 需要的 Key |
|---|---|---|
| `step1_hello.py` | Pipeline, TTSSpeakFrame, EndFrame, Transport | ElevenLabs |
| `step2_voice_bot.py` | STT+LLM+TTS, VAD, Context, Aggregators, AlwaysUserMuteStrategy | Deepgram + OpenAI + ElevenLabs |
| `step3_function_calling.py` | FunctionSchema, ToolsSchema, register_function, event handler | 同上 |

### ✅ Level 2 — 定制和进阶
| 文件 | 学到什么 | 需要的 Key |
|---|---|---|
| `step4_custom_processor.py` | FrameProcessor, 拦截/修改/过滤 frame, 会话重置 | 同上 |
| `step5_web_transport.py` | Pipecat Runner, transport_params, 浏览器访问, 事件处理 | 同上（可选 Daily） |
| `step6_context_injection.py` | LLMMessagesAppendFrame, LLMMessagesUpdateFrame, persona 切换 | 同上 |

### 📖 Level 3 — 结构化 & 工具集成
| 文件 | 学到什么 | 需要的 Key |
|---|---|---|
| `step7_pipecat_flows.py` | NodeConfig, FlowManager, Edge function, flow_manager.state | 同上 |
| `step8_mcp_in_bot.py` | MCPClient, register_tools, StdioServerParameters, tools_filter | 同上 |
| `step9_multi_agent.py` | AgentRunner, BusBridgeProcessor, LLMAgent, handoff_to, @tool | 同上 |
| `step10_langgraph.py` | LangGraphProcessor, LLMContextFrame, LLMTextFrame 帧协议 | 同上 |

### 🔭 Level 4 — 可观测性
| 文件 | 学到什么 | 需要的 Key |
|---|---|---|
| `step11_observers.py` | BaseObserver, LLMLogObserver, TranscriptionLogObserver, MetricsLogObserver | 同上 |
| `step12_per_stage_metrics.py` | TTFBMetricsData, ProcessingMetricsData, per-stage 时序表格 | 同上 |
| `step13_full_observability.py` | UserBotLatencyObserver, TurnTrackingObserver, StartupTimingObserver | 同上 |

### 🚀 Level 5 — 高级特性
| 文件 | 学到什么 | 需要的 Key |
|---|---|---|
| `step14_speech_to_speech_openai.py` | OpenAIRealtimeLLMService, SessionProperties, SemanticTurnDetection | OpenAI (Realtime) |
| `step15_speech_to_speech_gemini.py` | GeminiLiveLLMService, GeminiLiveParams | Google |
| `step16_llm_switcher.py` | LLMSwitcher, ManuallySwitchServiceFrame, ServiceSwitcherStrategyFailover | OpenAI |
| `step17_context_summarization.py` | enable_auto_context_summarization, LLMAutoContextSummarizationConfig | 同上 |
| `step18_multimodal.py` | LLMContext.create_image_url_message, 图片进 context | 同上 |
| `step19_modular_openai.py` | OpenAISTTService + OpenAITTSService，全 OpenAI 模块化 pipeline，只用一个 API key（对比 step14 speech-to-speech） | **只要 OpenAI** |

> **模块化 vs Speech-to-Speech**：step19 和 step2 一样是 STT→LLM→TTS 三段式 pipeline，但三个服务全用 OpenAI —— 所以只需要一个 `OPENAI_API_KEY`。step14 则用单个 Realtime 模型搞定（延迟更低，成本更高）。step19 = 模块化、可替换、便宜；step14 = 一体化、快。

> **Twilio + OpenAI Realtime（生产级电话版）**：`C:\Users\Yuki.Leong\github\twilio`
> 包含 WebSocket server + Twilio webhook + 完整电话接入

### 运行顺序
```bash
# Level 1-2（基础）
python examples/step1_hello.py
python examples/step2_voice_bot.py
python examples/step3_function_calling.py
python examples/step4_custom_processor.py
uv run python examples/step5_web_transport.py --transport webrtc
python examples/step6_context_injection.py

# Level 3（进阶功能）
python examples/step7_pipecat_flows.py          # uv add pipecat-ai-flows
python examples/step8_mcp_in_bot.py             # uv add "pipecat-ai[mcp]"
python examples/step9_multi_agent.py            # uv add pipecat-ai-subagents
python examples/step10_langgraph.py             # uv add langgraph langchain-openai

# Level 4（可观测性）
python examples/step11_observers.py
python examples/step12_per_stage_metrics.py
python examples/step13_full_observability.py

# Level 5（高级特性）
python examples/step14_speech_to_speech_openai.py   # 需要 Realtime API 访问权限
python examples/step15_speech_to_speech_gemini.py   # GOOGLE_API_KEY + uv add "pipecat-ai[google]"
python examples/step16_llm_switcher.py
python examples/step17_context_summarization.py
python examples/step18_multimodal.py
python examples/step19_modular_openai.py        # 全 OpenAI STT+LLM+TTS，只需 OPENAI_API_KEY
```

---

## 4. 安装什么 Library

### 基础安装（步骤 1-6）
```bash
uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]" python-dotenv loguru
```

### step7 需要额外装
```bash
uv add pipecat-ai-flows
```

### step8 需要额外装
```bash
uv add "pipecat-ai[mcp]"
# mcp-server-time 由 uvx 自动处理，不需要手动安装
```

### step9 需要额外装
```bash
uv add pipecat-ai-subagents
```

### step10 需要额外装
```bash
uv add langgraph langchain-openai langchain-core
```

### step15 需要额外装
```bash
uv add "pipecat-ai[google]"
# GOOGLE_API_KEY 免费申请：https://aistudio.google.com/apikey
```

### step14 注意事项
需要 OpenAI Realtime API 访问权限（`gpt-4o-realtime-preview`）。
Realtime API 比普通 OpenAI API 贵，按音频分钟计费。
你的生产级 Twilio 版本：`C:\Users\Yuki.Leong\github\twilio`

### step19 注意事项（全 OpenAI 模块化 pipeline）
```bash
# 不需要额外服务 —— 只要 openai extra + 本地麦克风/喇叭
uv add "pipecat-ai[local,openai,silero]"
```
用 `OpenAISTTService`（gpt-4o-transcribe）+ `OpenAILLMService`（gpt-4o-mini）+ `OpenAITTSService`（gpt-4o-mini-tts）。  
只需要 `OPENAI_API_KEY`，不需要 Deepgram / ElevenLabs / Cartesia 的 key。  
这里的 OpenAI STT 是 **segmented**（REST）模式：VAD 判断你说完一句后整段转录，所以比 Deepgram 流式略高延迟。想要更低延迟可换成 `OpenAIRealtimeSTTService`。

### step5 如果用 Daily transport
```bash
uv add "pipecat-ai[daily]"
# 并在 .env 加: DAILY_API_KEY=...
```

### 各 extra 说明
| Extra | 说明 |
|---|---|
| `local` | PyAudio，本地麦克风/喇叭 |
| `deepgram` | Deepgram STT |
| `openai` | OpenAI LLM / TTS |
| `elevenlabs` | ElevenLabs TTS |
| `cartesia` | Cartesia TTS |
| `silero` | Silero VAD（本地语音检测） |
| `daily` | Daily WebRTC transport |
| `mcp` | MCP client（连接 MCP server） |
| `anthropic` | Anthropic Claude LLM |

---

## 5. pipecat-ai-cli 是什么

**`pipecat-ai-cli`** 是独立的命令行工具，和 `pipecat-ai`（框架）是两个不同的包。

```bash
uv tool install pipecat-ai-cli
```

| 命令 | 作用 |
|---|---|
| `pipecat init <项目名>` | 生成项目模板（bot.py, .env 等）|
| `pipecat cloud auth login` | 登录 Pipecat Cloud |
| `pipecat cloud deploy` | 部署到 Pipecat Cloud |

**学习时不需要用**，等需要部署时再了解。

---

## 6. MCP 是什么

### MCP 在 Claude Code 里（工具辅助）
```bash
claude mcp add --transport http pipecat-docs https://daily-docs.mcp.kapa.ai
```
给 Claude Code（AI 助手）加 Pipecat 文档查询能力，已配置好，重启 session 后生效。

### MCP 在 Pipecat bot 里（step8）

MCP = **Model Context Protocol**，让 LLM 能调用外部工具的开放标准。

```
常见 MCP server：
  mcp-server-time       → 时间/时区查询
  mcp-server-filesystem → 文件系统
  mcp-server-fetch      → 网页抓取
  GitHub MCP            → GitHub 操作
  Brave Search MCP      → 网页搜索
```

Pipecat 里通过 `MCPClient` 连接：

```python
async with MCPClient(server_params=StdioServerParameters(...)) as mcp:
    tools = await mcp.register_tools(llm)    # 自动发现并注册
    context = LLMContext(tools=tools)        # 传给 context，LLM 就能调用
```

相比 step3 的手动 `FunctionSchema`，MCPClient 的优势：
- 不用手写 schema
- 一行注册所有工具
- 可连接任何 MCP-compatible 服务

---

## 7. 重要的坑和官方纠正

### `allow_interruptions` 已 deprecated
官方 contributor 原话：
> *"allow_interruptions is deprecated and generally not well suited for voice AI applications."*

**不要用** `PipelineParams(allow_interruptions=False/True)`。  
**应该用** `user_mute_strategies` 控制用户输入：

```python
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

user_params=LLMUserAggregatorParams(
    vad_analyzer=SileroVADAnalyzer(),
    user_mute_strategies=[AlwaysUserMuteStrategy()],
)
```

### LocalAudio + 喇叭的 echo 问题
- **根本原因**：笔记本麦克风物理上会收到喇叭的声音
- **最佳方案**：用耳机（无回声 + 支持 barge-in）
- **代码方案**：`AlwaysUserMuteStrategy`（防止回声打断，但 bot 说话时用户也无法打断）
- **最佳体验**：用 Web transport（Daily/WebRTC），浏览器内建 AEC，同时支持 barge-in

### LocalAudio 设备选择（Windows）
如果 `device 0 (Sound Mapper)` 映射到 loopback 设备，会导致 bot 听到自己说话：
```python
transport = LocalAudioTransport(
    LocalAudioTransportParams(
        input_device_index=1,  # 实体麦克风，避免 loopback
        ...
    )
)
# 查看设备列表：
# python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

### AlwaysUserMuteStrategy 内建策略清单
```python
from pipecat.turns.user_mute import (
    AlwaysUserMuteStrategy,              # bot 说话时 → 静音用户
    FirstSpeechUserMuteStrategy,         # 只在第一次 bot 说话时静音
    MuteUntilFirstBotCompleteUserMuteStrategy,  # 直到第一次 bot 说完才开放
    FunctionCallUserMuteStrategy,        # 工具调用期间静音
)
```

---

## 参考资源

| 资源 | 地址 |
|---|---|
| 官方文档 | https://docs.pipecat.ai |
| GitHub | https://github.com/pipecat-ai/pipecat |
| API Reference | https://reference-server.pipecat.ai |
| Pipecat Flows | https://github.com/pipecat-ai/pipecat-flows |
| 官方 examples | https://github.com/pipecat-ai/pipecat/tree/main/examples |
| Discord | https://discord.gg/pipecat |
| 可视化 Flows 编辑器 | https://flows.pipecat.ai |
