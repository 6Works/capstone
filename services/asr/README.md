# 🧠 Stage 2: Automated Speech Recognition (ASR) Microservice

The **ASR Microservice** converts acoustic audio waveforms into clean textual transcripts. It knows nothing about dialogue or text synthesis; its sole responsibility is acoustic-to-text inference.

---

## 📋 Contract Specification

**Abstract Interface**: `IASRService` ([services/contracts.py](file:///c:/Users/shive/projects/capstonedemo/services/contracts.py))

```python
class IASRService(ABC):
    @abstractmethod
    def transcribe(self, payload: AudioPayload) -> ASRResult:
        """Converts an AudioPayload into an ASRResult."""
        pass
```

### Output Data Transfer Object (`ASRResult`)

```python
class ASRResult(BaseModel):
    transcript: str          # Transcribed text string
    confidence: float        # Metric between 0.0 and 1.0
    language: str            # Detected or configured language (e.g., "en-US")
    latency_sec: float       # Processing duration in seconds
    provider: str            # Name of engine ("google_web_speech", "whisper_transformer", etc.)
    status: str              # "success" | "unintelligible" | "error"
    error: Optional[str]     # Descriptive error details if failed
```

---

## 🛠️ Available Providers

| Provider Class | Underlying Engine | Key Feature |
| :--- | :--- | :--- |
| `GoogleWebSpeechASR` | Google Speech API | Fast, zero setup, free cloud recognition. |
| `WhisperTransformerASR` | OpenAI Whisper / HuggingFace | Offline neural transformer model, handles accents and noise. |
| `MockASR` | Deterministic stub | Instant, zero-latency unit testing. |

---

## 💻 Standalone Usage Example

```python
from services.contracts import AudioPayload
from services.asr import GoogleWebSpeechASR, WhisperTransformerASR

# 1. Initialize provider
asr = GoogleWebSpeechASR(language="en-US")

# 2. Transcribe payload
payload = AudioPayload(source_type="file", file_path="sample.wav", format="wav")
result = asr.transcribe(payload)

if result.status == "success":
    print(f"Recognized: \"{result.transcript}\" (in {result.latency_sec}s)")
else:
    print(f"ASR Error: {result.error}")
```

---

## 🔮 Upgrading to Whisper Transformers

To upgrade this microservice to local neural inference using Hugging Face Transformers or `faster-whisper`:

1. Install the model package inside the Conda environment:
   ```bash
   pip install openai-whisper
   # OR for 4x faster GPU/CPU CTranslate2 inference:
   pip install faster-whisper
   ```

2. Implement or customize `WhisperTransformerASR` in [services/asr/asr_services.py](file:///c:/Users/shive/projects/capstonedemo/services/asr/asr_services.py):
   ```python
   from faster_whisper import WhisperModel

   class FasterWhisperASR(IASRService):
       def __init__(self, model_size="small", device="cpu"):
           self.model = WhisperModel(model_size, device=device, compute_type="int8")

       def transcribe(self, payload: AudioPayload) -> ASRResult:
           segments, info = self.model.transcribe(payload.file_path, beam_size=5)
           text = " ".join([s.text for s in segments]).strip()
           return ASRResult(
               transcript=text,
               confidence=info.avg_logprob,
               language=info.language,
               provider="faster_whisper"
           )
   ```
3. Register it in `orchestrator/factory.py`. **Zero changes are required in any other stage!**
