"""
Stage 2: Speech Recognition (ASR) Microservices.
Converts acoustic audio waveform into textual transcript.
Implements IASRService following Open/Closed Principle.
"""

import os
import time
import speech_recognition as sr
from typing import Optional
from ..contracts import IASRService, AudioPayload, ASRResult


class GoogleWebSpeechASR(IASRService):
    """ASR implementation using Google Web Speech API."""

    def __init__(self, language: str = "en-US"):
        self.language = language
        self.recognizer = sr.Recognizer()

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        start_time = time.time()
        
        if not payload.file_path or not os.path.exists(payload.file_path):
            return ASRResult(
                transcript="",
                provider="google_web_speech",
                status="error",
                error="Audio file path does not exist",
                latency_sec=0.0
            )

        try:
            with sr.AudioFile(payload.file_path) as source:
                audio_data = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio_data, language=self.language)
            elapsed = round(time.time() - start_time, 3)
            return ASRResult(
                transcript=text,
                confidence=0.92,
                language=self.language,
                latency_sec=elapsed,
                provider="google_web_speech",
                status="success"
            )

        except sr.UnknownValueError:
            elapsed = round(time.time() - start_time, 3)
            return ASRResult(
                transcript="",
                confidence=0.0,
                language=self.language,
                latency_sec=elapsed,
                provider="google_web_speech",
                status="unintelligible",
                error="Audio unintelligible or silent"
            )

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return ASRResult(
                transcript="",
                confidence=0.0,
                language=self.language,
                latency_sec=elapsed,
                provider="google_web_speech",
                status="error",
                error=str(e)
            )


class WhisperTransformerASR(IASRService):
    """
    Modular Whisper Transformer ASR Service.
    Engineered for plugging in HuggingFace Transformers, OpenAI Whisper, or faster-whisper.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", language: str = "en"):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self._is_initialized = False

    def _lazy_init(self):
        """Lazy load Whisper model weights only when invoked."""
        if not self._is_initialized:
            try:
                # Pluggable: check if faster_whisper or whisper is installed
                import whisper
                self.model = whisper.load_model(self.model_size, device=self.device)
                self._is_initialized = True
            except ImportError:
                # Graceful fallback: ready-to-load stub
                self.model = None
                self._is_initialized = True

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        start_time = time.time()
        self._lazy_init()

        if self.model is not None and payload.file_path:
            # Full local Whisper Transformer inference
            result = self.model.transcribe(payload.file_path, language=self.language)
            elapsed = round(time.time() - start_time, 3)
            return ASRResult(
                transcript=result.get("text", "").strip(),
                confidence=0.95,
                language=self.language,
                latency_sec=elapsed,
                provider=f"whisper_transformer_{self.model_size}",
                status="success"
            )

        # Fallback to Google ASR if whisper weights are not downloaded
        print("[Whisper ASR] Falling back to standard ASR provider...")
        fallback = GoogleWebSpeechASR(language="en-US")
        res = fallback.transcribe(payload)
        res.provider = f"whisper_transformer_{self.model_size} (fallback)"
        return res


class MockASR(IASRService):
    """Deterministic mock ASR for testing without network or audio devices."""

    def __init__(self, fixed_transcript: str = "Hello Mira, what is the time?"):
        self.fixed_transcript = fixed_transcript

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        return ASRResult(
            transcript=self.fixed_transcript,
            confidence=1.0,
            language="en-US",
            latency_sec=0.001,
            provider="mock_asr",
            status="success"
        )
