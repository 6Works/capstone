# 🎼 Orchestrator & Dependency Injection Factory (`orchestrator/`)

The **Orchestrator Layer** coordinates the end-to-end execution flow of the 6 microservices while strictly enforcing the **Dependency Inversion Principle (DIP)**:

> *High-level modules should not depend on low-level modules. Both should depend on abstractions.*

---

## 🏛️ Architecture & Components

```
┌────────────────────────────────────────────────────────┐
│                   PipelineFactory                      │
│        (Resolves and instantiates providers)          │
└───────────────────────────┬────────────────────────────┘
                            │ instantiates & injects
                            ▼
┌────────────────────────────────────────────────────────┐
│               DecoupledVoicePipeline                   │
│                                                        │
│  self.audio_input : IAudioInputService                 │
│  self.asr         : IASRService                        │
│  self.nlp         : INLPService                        │
│  self.dialogue    : IDialogueManagerService            │
│  self.tts         : ITTSService                        │
│  self.audio_output: IAudioOutputService                │
└────────────────────────────────────────────────────────┘
```

The `DecoupledVoicePipeline` class does **not** import any concrete speech recognition, NLP, or TTS library. It only knows about the abstract interfaces:
- `IAudioInputService`
- `IASRService`
- `INLPService`
- `IDialogueManagerService`
- `ITTSService`
- `IAudioOutputService`

---

## 🏭 Assembling Pipelines via `PipelineFactory`

The `PipelineFactory` ([orchestrator/factory.py](file:///c:/Users/shive/projects/capstonedemo/orchestrator/factory.py)) acts as the composition root, wiring together any requested provider combination:

```python
from orchestrator.factory import PipelineFactory

# 1. Assembling for Local Offline Testing
local_pipeline = PipelineFactory.create_pipeline(
    asr_provider="mock",
    nlp_provider="rule_based",
    tts_provider="pyttsx3",
    output_provider="speaker"
)

# 2. Assembling for Cloud/Neural Mode (Whisper + ElevenLabs)
neural_pipeline = PipelineFactory.create_pipeline(
    asr_provider="whisper",
    nlp_provider="llm",
    tts_provider="elevenlabs",
    output_provider="web"
)
```

---

## 💻 Standalone Execution Example

```python
from orchestrator.factory import PipelineFactory

# Create pipeline
pipeline = PipelineFactory.create_pipeline(
    asr_provider="google",
    tts_provider="pyttsx3",
    nlp_provider="rule_based"
)

# Run a simulated turn
telemetry = pipeline.run_turn_from_text("What time is it?", play_audio=False)

stages = telemetry["stages"]
print("ASR Provider:", stages["stage2_asr"]["provider"])
print("NLP Intent:", stages["stage3_nlp"]["intent"])
print("Dialogue Reply:", stages["stage4_dialogue_management"]["response_text"])
print("TTS Audio File:", stages["stage5_tts"]["file_path"])
print("Total Turn Latency:", telemetry["total_latency_sec"], "seconds")
```

---

## 🔄 Dynamic Runtime Reconfiguration

Because the pipeline is decoupled, the gateway server ([server.py](file:///c:/Users/shive/projects/capstonedemo/server.py)) exposes the `/api/pipeline/configure` endpoint:
- Clients or the web UI can switch providers on the fly (e.g., from `google` to `whisper`, or `pyttsx3` to `elevenlabs`) without restarting the server or dropping connections.
