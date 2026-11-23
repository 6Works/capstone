"""
Stage 6: Audio Output Microservices.
Handles delivering synthesized speech to hardware speakers or web streams.
Implements IAudioOutputService following SOLID.
"""

import os
import sys
import time
from typing import Optional
from ..contracts import IAudioOutputService, PlaybackResult


class SystemSpeakerOutput(IAudioOutputService):
    """Plays audio through system default hardware speakers."""

    def __init__(self):
        self.is_windows = sys.platform.startswith("win")

    def play(self, audio_file_path: str, block: bool = True) -> PlaybackResult:
        start_time = time.time()
        if not audio_file_path or not os.path.exists(audio_file_path):
            return PlaybackResult(
                status="error",
                error=f"File not found: {audio_file_path}",
                provider="system_speaker"
            )

        try:
            if self.is_windows and audio_file_path.lower().endswith(".wav"):
                import winsound
                flags = winsound.SND_FILENAME
                if not block:
                    flags |= winsound.SND_ASYNC
                winsound.PlaySound(audio_file_path, flags)
            else:
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    data, fs = sf.read(audio_file_path)
                    sd.play(data, fs)
                    if block:
                        sd.wait()
                except Exception:
                    if self.is_windows:
                        os.system(f'start "" /min "{audio_file_path}"')

            elapsed = round(time.time() - start_time, 3)
            return PlaybackResult(
                status="completed",
                playback_time_sec=elapsed,
                device="default_speaker",
                provider="system_speaker"
            )
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return PlaybackResult(
                status="error",
                error=str(e),
                playback_time_sec=elapsed,
                provider="system_speaker"
            )


class WebStreamAudioOutput(IAudioOutputService):
    """Delegates playback to client browser via web audio endpoints."""

    def play(self, audio_file_path: str, block: bool = False) -> PlaybackResult:
        exists = bool(audio_file_path and os.path.exists(audio_file_path))
        return PlaybackResult(
            status="completed" if exists else "error",
            playback_time_sec=0.01,
            device="browser_audio_element",
            provider="web_stream_audio_sink",
            error=None if exists else f"Audio file not found: {audio_file_path}"
        )
