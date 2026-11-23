# 6Works — Conversational Voice Agent & Technical Interview Platform

Final-year CSE capstone: a 6-person team building toward a startup.

- Workflow and rules: see [CONTRIBUTING.md](CONTRIBUTING.md)
- Ownership and rhythm: see [docs/TEAM.md](docs/TEAM.md)

---

## 🎙️ Conversational Voice Agent — Decoupled Microservices Pipeline (SOLID)

An end-to-end conversational speech AI system refactored into **independent, loosely-coupled microservices** following the **SOLID principles**:

```
[Audio Input] 🔁 [Speech Recognition (ASR)] 🔁 [Natural Language Processing (NLP)] 🔁 [Dialogue Management] 🔁 [Text-to-Speech (TTS)] 🔁 [Audio Output]
```

Every stage is completely disintegrated from the others, communicates via strict **Data Transfer Object (DTO) contracts**, and can be upgraded or swapped independently without modifying any other service or pipeline code.

---

## 🏛️ SOLID Architecture & Microservices

```
                        ┌───────────────────────────────┐
                        │      Pipeline Factory         │
                        │    (Dependency Injection)     │
                        └───────────────┬───────────────┘
                                        │ injects
                                        ▼
                        ┌───────────────────────────────┐
                        │   DecoupledVoicePipeline      │
                        │   (Depends ONLY on ABCs)      │
                        └───────────────┬───────────────┘
                                        │
      ┌───────────────┬─────────────────┼─────────────────┬───────────────┬───────────────┐
      ▼               ▼                 ▼                 ▼               ▼               ▼
┌───────────┐   ┌───────────┐     ┌───────────┐     ┌───────────┐   ┌───────────┐   ┌───────────┐
│  Stage 1  │   │  Stage 2  │     │  Stage 3  │     │  Stage 4  │   │  Stage 5  │   │  Stage 6  │
│Audio Input│   │    ASR    │     │    NLP    │     │ Dialogue  │   │    TTS    │   │Output Sink│
└─────┬─────┘   └─────┬─────┘     └─────┬─────┘     └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
      │               │                 │                 │               │               │
  Produces:       Produces:         Produces:         Produces:       Produces:       Produces:
AudioPayload      ASRResult         NLPResult     DialogueTurnResult  TTSResult     PlaybackResult
```

### 1. Single Responsibility Principle (SRP)
- **Audio Input** (`IAudioInputService`): Only captures and buffers acoustic audio signals.
- **ASR** (`IASRService`): Only transcribes acoustic signals to text transcripts.
- **NLP** (`INLPService`): Only extracts intent, entities, sentiment, and defines tool invocations.
- **Dialogue Management** (`IDialogueManagerService`): Only tracks state, memory context, and determines conversational policy.
- **TTS** (`ITTSService`): Only synthesizes textual responses into natural acoustic speech.
- **Audio Output** (`IAudioOutputService`): Only routes and delivers audio playback.

### 2. Open / Closed Principle (OCP)
New models and providers (e.g., `WhisperTransformerASR`, `ElevenLabsTTSService`, `LLMNLPService`) extend abstract base classes without modifying existing working code.

### 3. Liskov Substitution Principle (LSP)
Any ASR provider (`GoogleWebSpeechASR`, `WhisperTransformerASR`, `MockASR`) can replace another seamlessly without altering pipeline correctness.

### 4. Interface Segregation Principle (ISP)
Each service contract defines only the minimal required method signature (`capture`, `transcribe`, `parse`, `process_turn`, `synthesize`, `play`).

### 5. Dependency Inversion Principle (DIP)
The `DecoupledVoicePipeline` orchestrator depends exclusively on abstract contracts (`IAudioInputService`, `IASRService`, etc.), never on concrete third-party packages or libraries.

---

## 📁 Project Structure & Microservice Guides

```
capstonedemo/
├── services/
│   ├── README.md                 # 📖 Microservices Overview & Architecture
│   ├── contracts.py              # DTOs (AudioPayload, ASRResult, etc.) & Service ABCs
│   ├── audio_input/
│   │   ├── README.md             # 📖 Stage 1: Audio Input Service Guide
│   │   ├── __init__.py
│   │   └── audio_services.py     # MicrophoneAudioInput, FileAudioInput, SyntheticAudioInput
│   ├── asr/
│   │   ├── README.md             # 📖 Stage 2: ASR Service & Whisper Guide
│   │   ├── __init__.py
│   │   └── asr_services.py       # GoogleWebSpeechASR, WhisperTransformerASR, MockASR
│   ├── nlp/
│   │   ├── README.md             # 📖 Stage 3: NLP & Tool Calling Guide
│   │   ├── __init__.py
│   │   └── nlp_services.py       # RuleBasedNLPService, LLMNLPService
│   ├── dialogue/
│   │   ├── README.md             # 📖 Stage 4: Dialogue & Memory Store Guide
│   │   ├── __init__.py
│   │   └── dialogue_services.py  # ModularDialogueManager, ConversationMemoryStore
│   ├── tts/
│   │   ├── README.md             # 📖 Stage 5: TTS & ElevenLabs Pauses Guide
│   │   ├── __init__.py
│   │   └── tts_services.py       # Pyttsx3TTSService, GTTSService, ElevenLabsTTSService
│   └── audio_output/
│       ├── README.md             # 📖 Stage 6: Audio Output Sink Guide
│       ├── __init__.py
│       └── output_services.py    # SystemSpeakerOutput, WebStreamAudioOutput
│
├── orchestrator/
│   ├── README.md                 # 📖 Orchestrator & Dependency Injection Guide
│   ├── __init__.py
│   ├── pipeline.py               # Decoupled orchestrator (pure DIP)
│   └── factory.py                # Service factory & Dependency Injection
│
├── web/
│   └── index.html                # Visual interactive dashboard with dynamic provider selector
├── server.py                     # Microservices gateway with hot-swappable provider config
├── main.py                       # CLI runner with provider flags
├── test_pipeline.py              # 12/12 unit and integration tests
├── requirements.txt              # Isolated dependencies
├── run.bat                       # One-click Windows runner (targets Conda env)
├── run.sh                        # One-click Linux / macOS runner
└── README.md                     # Documentation
```

---

## 🔮 Upgrading Individual Modules

Because each stage is decoupled behind a contract, you can upgrade them one-by-one:

### 1. Upgrading ASR to Whisper Transformers
Edit [services/asr/asr_services.py](file:///c:/Users/shive/projects/capstonedemo/services/asr/asr_services.py):
```python
class WhisperTransformerASR(IASRService):
    def __init__(self, model_size="base"):
        import whisper  # or from transformers import WhisperForConditionalGeneration
        self.model = whisper.load_model(model_size)

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        result = self.model.transcribe(payload.file_path)
        return ASRResult(transcript=result["text"], provider="whisper")
```

### 2. Upgrading Dialogue Management (Vector Memory & Tools)
Edit [services/dialogue/dialogue_services.py](file:///c:/Users/shive/projects/capstonedemo/services/dialogue/dialogue_services.py):
The `ConversationMemoryStore` is ready for plugging in ChromaDB / Pinecone vector stores, and `ModularDialogueManager` supports tool execution hooks.

### 3. Upgrading TTS to ElevenLabs (Natural Human Pauses)
Edit [services/tts/tts_services.py](file:///c:/Users/shive/projects/capstonedemo/services/tts/tts_services.py):
`ElevenLabsTTSService` is already implemented with:
- Natural human breath pauses (`add_human_pauses`)
- Voice ID configuration
- Direct API integration with automatic offline fallback if offline

---

## 🚀 Running the Project

### 1. Isolated Conda Environment
```bash
conda activate capstone_voice
# Or use the environment's python:
# %USERPROFILE%\.conda\envs\capstone_voice\python.exe
```

### 2. Interactive Web Dashboard
```bash
python server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.
- **Microservices Bar**: Hot-swap ASR (`Google`, `Whisper`, `Mock`), NLP (`Rule-Based`, `LLM`), and TTS (`pyttsx3`, `ElevenLabs`, `gTTS`) on the fly without restarting the server!
- **Visual Execution**: Live stage-by-stage lighting showing real-time latency and the exact provider handling each step.
- **Mic Voice Recording**: Automatically converted to 16-bit PCM WAV in the browser and displays recognized transcript directly beneath the recording banner.

### 3. Command Line with Provider Flags
```bash
# Run with Google ASR + pyttsx3 TTS
python main.py --mode demo --asr google --tts pyttsx3

# Run with Whisper ASR + ElevenLabs Human-like TTS
python main.py --mode demo --asr whisper --tts elevenlabs

# Run interactive microphone loop
python main.py --mode mic --asr google --tts pyttsx3
```

### 4. Running Tests
```bash
pytest test_pipeline.py -v
# 12 passed in 0.72s
```
