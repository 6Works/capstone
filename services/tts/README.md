# 🔊 Stage 5: Text-to-Speech (TTS) Microservice

The **TTS Microservice** transforms textual replies produced by Dialogue Management into synthesized acoustic speech audio files (`.wav` or `.mp3`).

---

## 📋 Contract Specification

**Abstract Interface**: `ITTSService` ([services/contracts.py](file:///c:/Users/shive/projects/capstonedemo/services/contracts.py))

```python
class ITTSService(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        """Synthesizes text into an audio file, returning a TTSResult."""
        pass
```

### Output Data Transfer Object (`TTSResult`)

```python
class TTSResult(BaseModel):
    file_path: str           # Path to generated audio file on disk
    audio_format: str        # "wav" | "mp3"
    latency_sec: float       # Generation time in seconds
    provider: str            # Engine name ("pyttsx3_sapi5", "elevenlabs_v1", "gtts")
    speech_rate: int         # Word rate per minute
    status: str              # "success" | "skipped" | "error"
    error: Optional[str]     # Descriptive error if failed
```

---

## 🛠️ Available Providers

| Provider Class | Technology | Key Characteristics |
| :--- | :--- | :--- |
| `Pyttsx3TTSService` | Windows SAPI5 offline | Zero internet dependency, instant local synthesis (100-300ms). |
| `ElevenLabsTTSService` | ElevenLabs API + Cadence Pauses | **Human-like realistic voice**, cadence pauses at punctuation, custom voice IDs. |
| `GTTSService` | Google Text-to-Speech | Cloud MP3 generation, clean pronunciation. |

---

## 🎭 Human Cadence & Natural Pauses

A key feature of the `ElevenLabsTTSService` is its `add_human_pauses()` method:
- Traditional TTS often sounds robotic because words are spoken in a continuous, unvarying stream.
- `add_human_pauses()` injects micro-pauses at punctuation boundaries (commas, periods, ellipses) to simulate human breathing intervals and natural rhythm:
  ```python
  service = ElevenLabsTTSService()
  paced_text = service.add_human_pauses("Hello, I am ready. Let us begin.")
  # Transforms to: "Hello, ... I am ready. ... Let us begin."
  ```

---

## 💻 Standalone Usage Example

```python
from services.tts import Pyttsx3TTSService, ElevenLabsTTSService

# 1. Offline SAPI5
tts_offline = Pyttsx3TTSService(speech_rate=175)
res1 = tts_offline.synthesize("Hello! This is offline speech synthesis.")
print(f"Generated: {res1.file_path} in {res1.latency_sec}s using {res1.provider}")

# 2. ElevenLabs with Human Cadence
tts_human = ElevenLabsTTSService(api_key="your_api_key_here") # Optional: falls back gracefully if key is None
res2 = tts_human.synthesize("Welcome to our decoupled voice architecture.")
print(f"Generated: {res2.file_path} in {res2.latency_sec}s using {res2.provider}")
```

---

## 🔮 Upgrading to Voice Clones & Neural Models

- **ElevenLabs Voice Cloning**: Set `voice_id="<your_custom_voice_id>"` in `ElevenLabsTTSService` to speak with cloned voices.
- **Edge-TTS / Bark / Coqui**: Create a new class implementing `ITTSService` (e.g. `EdgeTTSService`) and register it in `orchestrator/factory.py`. No other part of the codebase needs modification.
