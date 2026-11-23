# 🧩 Microservices Directory (`services/`)

This directory houses the **6 decoupled microservices** powering the conversational voice agent. Each stage is treated as an isolated domain service following **SOLID design principles**, communicating solely via strict Data Transfer Objects (DTOs) and abstract contracts.

---

## 🏛️ Inter-Service Contracts (`services/contracts.py`)

All microservices adhere to explicit contracts defined in [services/contracts.py](file:///c:/Users/shive/projects/capstonedemo/services/contracts.py):

| Service | Interface Contract | Primary Input | Primary Output DTO |
| :--- | :--- | :--- | :--- |
| **Stage 1: Audio Input** | `IAudioInputService` | Capture parameters (`timeout`, `file_path`) | `AudioPayload` |
| **Stage 2: ASR** | `IASRService` | `AudioPayload` | `ASRResult` |
| **Stage 3: NLP** | `INLPService` | Raw or normalized text string | `NLPResult` |
| **Stage 4: Dialogue** | `IDialogueManagerService` | `NLPResult` | `DialogueTurnResult` |
| **Stage 5: TTS** | `ITTSService` | Generated response text string | `TTSResult` |
| **Stage 6: Audio Output** | `IAudioOutputService` | Path to audio file or raw buffer | `PlaybackResult` |

---

## 📂 Subdirectories & Individual Documentation

- [services/audio_input/README.md](file:///c:/Users/shive/projects/capstonedemo/services/audio_input/README.md): Microphone, WAV file ingestion, and synthetic signal generation.
- [services/asr/README.md](file:///c:/Users/shive/projects/capstonedemo/services/asr/README.md): Speech-to-text recognition (Google Web Speech, Whisper Transformer, Mock).
- [services/nlp/README.md](file:///c:/Users/shive/projects/capstonedemo/services/nlp/README.md): Natural language understanding, intent classification, entity parsing, and tool definitions.
- [services/dialogue/README.md](file:///c:/Users/shive/projects/capstonedemo/services/dialogue/README.md): Conversation memory store, finite state machine, and dialogue policy.
- [services/tts/README.md](file:///c:/Users/shive/projects/capstonedemo/services/tts/README.md): Text-to-speech synthesis (pyttsx3 offline, ElevenLabs with human cadence & pauses, gTTS).
- [services/audio_output/README.md](file:///c:/Users/shive/projects/capstonedemo/services/audio_output/README.md): Audio delivery (hardware speaker playback and browser HTML5 audio sinks).

---

## 🛠️ Testing Services Independently

Because each service conforms to an abstract base class, any service can be instantiated and executed in isolation:

```python
from services.contracts import AudioPayload
from services.asr import GoogleWebSpeechASR

# Test ASR completely standalone
asr_service = GoogleWebSpeechASR(language="en-US")
payload = AudioPayload(source_type="file_upload", file_path="test.wav", format="wav")
result = asr_service.transcribe(payload)

print("Transcript:", result.transcript)
print("Latency:", result.latency_sec)
```
