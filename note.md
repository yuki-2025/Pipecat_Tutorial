# Pipecat Tutorial Key Points

---

## step1 — 最简 Pipeline（TTS only）
三个核心概念：**Pipeline**（链条）/ **Frame**（数据容器）/ **Transport**（音频出口）。
```python
pipeline = Pipeline([tts, transport.output()])
await task.queue_frames([TTSSpeakFrame("Hello!"), EndFrame()])
```
说一句话，EndFrame 结束。无 STT，无 LLM。

---

## step2 — 完整 STT → LLM → TTS
声明三个 service，pipeline 串起来：
```python
pipeline = Pipeline([
    transport.input(),   # 麦克风
    stt,                 # 语音 → 文字
    user_aggregator,     # 积累 + VAD 触发 LLM
    llm,
    tts,
    transport.output(),  # 喇叭
    assistant_aggregator,
])
```
`LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))` 管对话历史和 VAD。

---

## step3 — Function Calling（工具调用）
1. 定义 `FunctionSchema` → 2. `register_function` 绑定 Python 函数 → 3. `LLMContext(tools=tools)` 告诉 LLM 有什么工具：
```python
llm.register_function("get_weather", get_weather)

@llm.event_handler("on_function_calls_started")
async def on_function_calls_started(service, function_calls):
    await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

context = LLMContext(tools=tools)
```

---

## step4 — 自定义 FrameProcessor
继承 `FrameProcessor`，在 `process_frame` 里拦截/修改/过滤 frame：
```python
class MyProcessor(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            print(frame.text)
        await self.push_frame(frame, direction)  # 不 push = 吞掉
```
插入 pipeline 任意位置。

---

## step5 — Web Transport（浏览器访问）
用 `transport_params` 字典 + `create_transport` 支持多种 transport，命令行 `--transport webrtc/daily` 切换：
```python
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily": _daily_params,
}
async def bot(runner_args):
    transport = await create_transport(runner_args, transport_params)

@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    await task.queue_frames([LLMRunFrame()])
```
`python step5.py --transport webrtc` → 开浏览器 `http://localhost:7860/client`。

---

## step6 — 动态 Context 注入
两种注入方式：
- `LLMMessagesAppendFrame` — 追加消息，不中断对话
- `LLMMessagesUpdateFrame` — 完全替换 context（persona 切换）

```python
# 追加（注入外部数据/用户 profile）
await task.queue_frames([LLMMessagesAppendFrame(messages=[{"role": "user", "content": info}])])

# 替换（persona 切换）
await task.queue_frames([LLMMessagesUpdateFrame(messages=new_persona_messages)])
```

---

## step7 — Pipecat Flows（结构化状态机）
每个节点（`NodeConfig`）只做一件事 + 一组工具。Edge function 返回 `(result, next_node)` 触发跳转，Node function 返回 `(result, None)` 留在当前节点：
```python
NodeConfig(
    name="greeting",
    task_messages=[{"role": "developer", "content": "Ask for name."}],
    functions=[record_name_func],   # Edge function → 跳到下一节点
)

flow_manager.state["customer_name"] = name  # 跨节点共享数据
```
`FlowManager` 管跳转，`post_actions=[{"type": "end_conversation"}]` 结束对话。

---

## step8 — MCPClient（连 MCP 工具）
一行注册所有工具，取代 step3 手写 schema：
```python
async with MCPClient(
    server_params=StdioServerParameters(command=uvx, args=["mcp-server-time"]),
    tools_filter=["get_current_time"],
) as mcp:
    tools = await mcp.register_tools(llm)   # 自动发现 + 注册
    context = LLMContext(tools=tools)
```

---

## step9 — Multi-Agent（多 Agent 协作）
主 pipeline 里放 `BusBridgeProcessor`，多个 `LLMAgent` 通过 `AgentBus` 通信，`handoff_to()` 切换 active agent：
```python
# @tool 替代 register_function，更简洁
@tool(cancel_on_interruption=False)
async def transfer_to_agent(self, params, agent: str, reason: str):
    await self.handoff_to(agent, activation_args=...)

# 主 pipeline
Pipeline([transport.input(), stt, user_agg, BusBridgeProcessor(bus), tts, transport.output(), asst_agg])
```

---

## step10 — LangGraph 集成
写桥接 `FrameProcessor`，遵循帧协议替换 LLM service 位置：
```python
class LangGraphProcessor(FrameProcessor):
    async def process_frame(self, frame, direction):
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(LLMFullResponseStartFrame())
            async for chunk in graph.astream({"messages": msgs}):
                await self.push_frame(LLMTextFrame(chunk))   # 必须用 LLMTextFrame，不是 TextFrame
            await self.push_frame(LLMFullResponseEndFrame())  # 必须有，否则 TTS 不 flush
```
LangGraph 自己管 state/history，不用 Pipecat `LLMContext`。

---

## step11 — Observer（非侵入监控）
Observer 在 pipeline 外旁观，不影响数据流。传给 `PipelineTask`：
```python
task = PipelineTask(pipeline, observers=[
    LLMLogObserver(),
    TranscriptionLogObserver(),
    MetricsLogObserver(),
])
# 自定义 Observer
class MyObserver(BaseObserver):
    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, MetricsFrame): ...
```
需要 `enable_metrics=True, enable_usage_metrics=True`。

---

## step12 — Per-Stage Metrics（各阶段延迟）
每条 `MetricsFrame` 标注了来自哪个 service，三种延迟：
- **TTFB** — 请求发出 → 第一个输出
- **ProcessingTime** — 整个 service 完成的总时间
- **TextAggregation** — LLM 第一个 token → 凑够第一句完整句子（TTS 等待）

E2E = STT 收尾 + LLM TTFB + TextAggregation + TTS TTFB

---

## step13 — 完整可观测性
内建 observer 替代手写：
```python
UserBotLatencyObserver()     # E2E 延迟 + 详细 breakdown
TurnTrackingObserver()       # 轮次开始/结束/被打断/持续时间
StartupTimingObserver()      # 每个 processor 的初始化时间
```

---

## step14 — Speech-to-Speech（OpenAI Realtime）
一个 `OpenAIRealtimeLLMService` 取代 STT + LLM + TTS 三个 service：
```python
llm = OpenAIRealtimeLLMService(
    api_key=...,
    session_properties=SessionProperties(
        voice="alloy",
        turn_detection=SemanticTurnDetection(),
        input_audio_noise_reduction=InputAudioNoiseReduction(type="far_field"),
    ),
)
# Pipeline 不需要 stt 和 tts
Pipeline([transport.input(), user_agg, llm, transport.output(), asst_agg])
```

---

## step15 — Speech-to-Speech（Gemini Live）
同 step14 概念，换成 Google：
```python
llm = GeminiLiveLLMService(
    api_key=os.environ["GOOGLE_API_KEY"],
    params=GeminiLiveParams(audio_out_enabled=True, ...),
)
```
优势：集成 Google 搜索、支持视频（multimodal）。需要 `GOOGLE_API_KEY`（免费申请）。

---

## step16 — LLMSwitcher（运行时切换 LLM）
`LLMSwitcher` 替代单个 llm 放进 pipeline，管多个 LLM：
```python
llm_switcher = LLMSwitcher(
    llms=[llm_mini, llm_full],           # 第一个是默认
    strategy_type=ServiceSwitcherStrategyManual,   # 或 ServiceSwitcherStrategyFailover（自动故障转移）
)
# 触发切换
await task.queue_frames([ManuallySwitchServiceFrame()])
```
所有 LLM 必须共享同一个 `LLMContext`。

---

## step17 — Context Summarization（长对话压缩）
在 `LLMAssistantAggregatorParams` 开启，超过阈值自动压缩旧消息为摘要：
```python
assistant_params=LLMAssistantAggregatorParams(
    enable_auto_context_summarization=True,
    auto_context_summarization_config=LLMAutoContextSummarizationConfig(
        max_context_tokens=8000,
        summary_config=LLMContextSummaryConfig(min_messages_after_summary=5),
    ),
)
@assistant_aggregator.event_handler("on_summary_applied")
async def on_summary_applied(aggregator, summary, messages_before, messages_after): ...
```

---

## step18 — Multimodal（图片进 LLM）
```python
# URL 方式
msg = LLMContext.create_image_url_message(url="https://...", text="What's in this image?")
# 本地图片
msg = LLMContext.create_image_message(image=image_bytes, text="Describe this.")

# 注入并立刻触发 LLM
await task.queue_frames([LLMMessagesAppendFrame(messages=[msg], run_llm=True)])
```

---

## step19 — Modular OpenAI（全 OpenAI 单 key）
三个服务全用 OpenAI，只需一个 `OPENAI_API_KEY`：
```python
stt = OpenAISTTService(api_key=api_key, settings=OpenAISTTService.Settings(model="gpt-4o-transcribe"))
llm = OpenAILLMService(api_key=api_key, settings=OpenAILLMService.Settings(model="gpt-4o-mini"))
tts = OpenAITTSService(api_key=api_key, settings=OpenAITTSService.Settings(voice="alloy"))
```
Pipeline 结构和 step2 完全一样。STT 是 segmented（REST），VAD 触发整段转录。
对比 step14：三段式可单独替换/调试，延迟略高但便宜。
