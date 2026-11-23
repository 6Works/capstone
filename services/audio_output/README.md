# 🔈 Stage 6: Audio Output Microservice

The **Audio Output Microservice** is responsible for delivering synthesized speech audio to the user's ears via hardware speakers, sound drivers, or browser streaming endpoints.

---

## 📋 Contract Specification

**Abstract Interface**: `IAudioOutputService` ([services/contracts.py](file:///c:/Users/shive/projects/capstonedemo/services/contracts.py))

```python
class IAudioOutputService(ABC):
    @abstractmethod
    def play(self, audio_file_path: str, block: bool = True) -> PlaybackResult:
        """Plays the synthesized audio file through the designated output sink."""
        pass
```

### Output Data Transfer Object (`PlaybackResult`)

```python
class PlaybackResult(BaseModel):
    status: str              # "completed" | "skipped" | "error"
    playback_time_sec: float # Elapsed duration of playback
    device: str              # Output device description (e.g. "default_speaker")
    provider: str            # Output sink name ("system_speaker", "web_stream_audio_sink")
    error: Optional[str]     # Error details if playback failed
```

---

## 🛠️ Available Providers

| Provider Class | Output Sink | Use Case |
| :--- | :--- | :--- |
| `SystemSpeakerOutput` | Physical PC speakers via `winsound` / `sounddevice` | Terminal CLI runs, desktop standalone mode. |
| `WebStreamAudioOutput` | Browser HTML5 `<audio>` element | Web applications, browser UI sessions. |

---

## 💻 Standalone Usage Example

```python
from services.audio_output import SystemSpeakerOutput, WebStreamAudioOutput

# Play through PC speakers
speaker = SystemSpeakerOutput()
res = speaker.play("storage/temp_audio/sample.wav", block=True)
print(f"Playback status: {res.status} on {res.device} ({res.playback_time_sec}s)")

# Delegate to web browser stream
web_sink = WebStreamAudioOutput()
web_res = web_sink.play("storage/temp_audio/sample.wav")
print(f"Web Sink ready: {web_res.provider}")
```

---

## 🔮 Future Upgrade Path

- **Duplex Audio Streaming**: Stream audio chunks in real-time over WebSockets as they are being synthesized by TTS rather than waiting for the complete file to finish.
- **Audio Output Selection**: Route audio to specific physical output device indices (USB headsets, external amplifiers) using `sounddevice.OutputStream`.
