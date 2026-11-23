# 🎙️ Stage 1: Audio Input Microservice

The **Audio Input Microservice** is responsible exclusively for capturing acoustic voice signals and packaging them into a standardized `AudioPayload` DTO.

---

## 📋 Contract Specification

**Abstract Interface**: `IAudioInputService` ([services/contracts.py](file:///c:/Users/shive/projects/capstonedemo/services/contracts.py))

```python
class IAudioInputService(ABC):
    @abstractmethod
    def capture(self, **kwargs) -> AudioPayload:
        """Captures acoustic signal and returns a standardized AudioPayload."""
        pass
```

### Output Data Transfer Object (`AudioPayload`)

```python
class AudioPayload(BaseModel):
    source_type: str         # "microphone" | "file_upload" | "synthetic" | "stream"
    format: str              # "wav" | "mp3" | "flac"
    sample_rate: int         # Typically 16000 Hz
    channels: int            # 1 (Mono)
    duration_sec: float      # Duration in seconds
    file_path: Optional[str] # Path to temporary/stored audio file
    raw_bytes: Optional[bytes] # Optional raw PCM byte buffer
    metadata: Dict[str, Any]
```

---

## 🛠️ Available Providers

| Provider Class | Description | Use Case |
| :--- | :--- | :--- |
| `MicrophoneAudioInput` | Captures directly from default hardware microphone using PyAudio & SpeechRecognition. | Live local terminal/desktop voice loop. |
| `FileAudioInput` | Ingests a pre-recorded `.wav` file from disk or user HTTP upload. | Web UI uploads, REST API endpoints. |
| `SyntheticAudioInput` | Generates a 440 Hz sinusoidal PCM WAV tone. | Automated CI/CD pipelines & test suites. |

---

## 💻 Standalone Usage Example

```python
from services.audio_input import MicrophoneAudioInput, FileAudioInput

# Capture from local microphone
mic = MicrophoneAudioInput(sample_rate=16000)
payload = mic.capture(timeout=5, phrase_time_limit=8)
print(f"Captured {payload.duration_sec}s from {payload.source_type} at {payload.file_path}")

# Ingest an audio file
file_input = FileAudioInput()
file_payload = file_input.capture(file_path="sample.wav")
print(f"Loaded file format: {file_payload.format}, duration: {file_payload.duration_sec}s")
```

---

## 🔮 Future Upgrade Path

- **Voice Activity Detection (VAD)**: Integrate `silero-vad` or `webrtcvad` inside a `VADAudioInput` provider to eliminate leading/trailing silence automatically.
- **WebRTC Streaming**: Add a `WebRTCStreamingInput` provider for continuous real-time duplex audio frames.
- **Multi-channel Beamforming**: Implement directional array input for noisy environments.
