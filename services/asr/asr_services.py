"""
Stage 2: Speech Recognition (ASR) Microservices.
Converts acoustic audio waveform into textual transcript.
Implements IASRService following Open/Closed Principle.
"""

import os
import time
import math
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


# Module-level model cache to avoid re-loading weights on each turn
_WHISPER_CACHE = {}


class WhisperTransformerASR(IASRService):
    """
    High-Performance Offline Whisper Transformer ASR Service powered by faster-whisper (CTranslate2).
    Supports INT8 CPU quantization, Silero Voice Activity Detection (VAD), and ultra-low latency inference.
    Adheres strictly to SOLID: Open for extension with any CTranslate2/Whisper model size, closed for modification.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        vad_filter: bool = True,
        beam_size: int = 5
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.model = None
        self._is_initialized = False

    def _lazy_init(self):
        """Lazy load and cache Whisper model weights into memory."""
        cache_key = f"{self.model_size}_{self.device}_{self.compute_type}"
        global _WHISPER_CACHE

        if cache_key in _WHISPER_CACHE:
            self.model = _WHISPER_CACHE[cache_key]
            self._is_initialized = True
            return

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            _WHISPER_CACHE[cache_key] = model
            self.model = model
            self._is_initialized = True
        except Exception as e:
            # Pluggable fallback to openai-whisper if present
            try:
                import whisper
                model = whisper.load_model(self.model_size.split(".")[0], device=self.device)
                _WHISPER_CACHE[cache_key] = model
                self.model = model
                self._is_initialized = True
            except Exception:
                self.model = None
                self._is_initialized = True

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        start_time = time.time()
        self._lazy_init()

        if not payload.file_path or not os.path.exists(payload.file_path):
            return ASRResult(
                transcript="",
                confidence=0.0,
                language=self.language,
                latency_sec=0.0,
                provider=f"whisper_{self.model_size}",
                status="error",
                error="Audio file path does not exist"
            )

        if self.model is not None:
            try:
                import faster_whisper
                if isinstance(self.model, faster_whisper.WhisperModel):
                    # faster-whisper CTranslate2 inference with optional VAD
                    lang_param = None if self.model_size.endswith(".en") else self.language
                    segments, info = self.model.transcribe(
                        payload.file_path,
                        language=lang_param,
                        beam_size=self.beam_size,
                        vad_filter=self.vad_filter
                    )
                    
                    text_parts = []
                    probs = []
                    segments_list = []
                    for seg in segments:
                        txt = seg.text.strip()
                        if txt:
                            text_parts.append(txt)
                            if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                                probs.append(math.exp(seg.avg_logprob))
                            segments_list.append({
                                "start": round(seg.start, 2),
                                "end": round(seg.end, 2),
                                "text": txt
                            })

                    full_transcript = " ".join(text_parts).strip()
                    avg_confidence = round(sum(probs) / len(probs), 2) if probs else 0.95
                    elapsed = round(time.time() - start_time, 3)

                    meta = {
                        "segments": segments_list,
                        "beam_size": self.beam_size,
                        "vad_filter": self.vad_filter,
                        "language_probability": round(float(getattr(info, "language_probability", 1.0)), 3)
                    }

                    return ASRResult(
                        transcript=full_transcript,
                        confidence=avg_confidence,
                        language=info.language or self.language,
                        latency_sec=elapsed,
                        provider=f"whisper_{self.model_size}_{self.compute_type}",
                        status="success" if full_transcript else "unintelligible",
                        metadata=meta
                    )
                else:
                    # Fallback to standard OpenAI Whisper
                    result = self.model.transcribe(payload.file_path, language=self.language)
                    elapsed = round(time.time() - start_time, 3)
                    return ASRResult(
                        transcript=result.get("text", "").strip(),
                        confidence=0.92,
                        language=self.language,
                        latency_sec=elapsed,
                        provider=f"whisper_{self.model_size}",
                        status="success"
                    )
            except Exception as e:
                print(f"[Whisper ASR] Transcription error: {e}. Falling back...")

        # Graceful fallback to Google Web Speech ASR
        fallback = GoogleWebSpeechASR(language="en-US")
        res = fallback.transcribe(payload)
        res.provider = f"whisper_{self.model_size} (fallback)"
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
