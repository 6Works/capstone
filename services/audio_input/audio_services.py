"""
Stage 1: Audio Input Services.
Captures acoustic signals from physical microphone, audio files, or buffers.
"""

import os
import wave
import time
import tempfile
import speech_recognition as sr
from typing import Optional
from ..contracts import IAudioInputService, AudioPayload


class MicrophoneAudioInput(IAudioInputService):
    """Captures speech directly from the default hardware microphone."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def capture(self, timeout: int = 5, phrase_time_limit: int = 8, **kwargs) -> AudioPayload:
        with sr.Microphone(sample_rate=self.sample_rate) as source:
            print("[Stage 1: Audio Input] Listening to microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = self.recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )

        raw = audio_data.get_raw_data()
        duration = round(len(raw) / (audio_data.sample_rate * audio_data.sample_width), 2)
        
        # Save to temporary WAV for downstream processing
        temp_wav = os.path.join(tempfile.gettempdir(), f"mic_{int(time.time()*1000)}.wav")
        with open(temp_wav, "wb") as f:
            f.write(audio_data.get_wav_data())

        return AudioPayload(
            source_type="microphone",
            format="wav",
            sample_rate=audio_data.sample_rate,
            channels=1,
            duration_sec=duration,
            file_path=temp_wav,
            raw_bytes=raw,
            metadata={"device": "default_mic"}
        )


class FileAudioInput(IAudioInputService):
    """Loads an audio recording from an on-disk file."""

    def capture(self, file_path: str = "", **kwargs) -> AudioPayload:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        format_ext = os.path.splitext(file_path)[1].replace(".", "").lower() or "wav"

        # Read basic WAV metadata if WAV
        sample_rate = 16000
        duration = 1.0
        if format_ext == "wav":
            try:
                with wave.open(file_path, "rb") as wf:
                    sample_rate = wf.getframerate()
                    frames = wf.getnframes()
                    duration = round(frames / float(sample_rate), 2)
            except Exception:
                pass

        return AudioPayload(
            source_type="file_upload",
            format=format_ext,
            sample_rate=sample_rate,
            channels=1,
            duration_sec=duration,
            file_path=file_path,
            metadata={"size_bytes": file_size}
        )


class SyntheticAudioInput(IAudioInputService):
    """Generates a synthetic audio tone for automated testing without microphone."""

    def capture(self, duration_sec: float = 1.0, **kwargs) -> AudioPayload:
        import struct
        import math

        temp_wav = os.path.join(tempfile.gettempdir(), f"synthetic_{int(time.time()*1000)}.wav")
        sample_rate = 16000
        num_samples = int(sample_rate * duration_sec)

        with wave.open(temp_wav, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for i in range(num_samples):
                val = int(32767.0 * 0.05 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
                wav_file.writeframesraw(struct.pack("<h", val))

        return AudioPayload(
            source_type="synthetic",
            format="wav",
            sample_rate=sample_rate,
            channels=1,
            duration_sec=duration_sec,
            file_path=temp_wav
        )
