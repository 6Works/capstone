"""
Stage 5: Text-to-Speech (TTS) Microservices.
Synthesizes generated text response into acoustic speech audio.
Implements ITTSService following Open/Closed Principle.
"""

import os
import time
import wave
import tempfile
from typing import Optional, Dict, Any
from ..contracts import ITTSService, TTSResult
from .text_normalizer import TextNormalizer, G2PPhonemizer
from .voice_cloner import voice_cloner, VoiceCloner, SpeakerProfile


class Pyttsx3TTSService(ITTSService):
    """Offline Windows SAPI5 TTS engine (zero internet dependency, zero latency)."""

    def __init__(self, speech_rate: int = 175, volume: float = 0.95):
        self.speech_rate = speech_rate
        self.volume = volume

    def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        start_time = time.time()
        if not text or not text.strip():
            return TTSResult(
                file_path="",
                provider="pyttsx3",
                status="skipped",
                latency_sec=0.0
            )

        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.wav")

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.setProperty("volume", self.volume)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            del engine

            elapsed = round(time.time() - start_time, 3)
            return TTSResult(
                file_path=output_path,
                audio_format="wav",
                latency_sec=elapsed,
                provider="pyttsx3_sapi5",
                speech_rate=self.speech_rate,
                status="success"
            )
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return TTSResult(
                file_path="",
                provider="pyttsx3",
                status="error",
                error=str(e),
                latency_sec=elapsed
            )


class GTTSService(ITTSService):
    """Google Text-To-Speech cloud engine producing natural MP3."""

    def __init__(self, lang: str = "en", slow: bool = False):
        self.lang = lang
        self.slow = slow

    def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        start_time = time.time()
        if not text or not text.strip():
            return TTSResult(file_path="", provider="gtts", status="skipped")

        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.mp3")

        try:
            from gtts import gTTS
            tts_obj = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts_obj.save(output_path)
            elapsed = round(time.time() - start_time, 3)
            return TTSResult(
                file_path=output_path,
                audio_format="mp3",
                latency_sec=elapsed,
                provider="gtts",
                status="success"
            )
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return TTSResult(file_path="", provider="gtts", status="error", error=str(e), latency_sec=elapsed)


class ElevenLabsTTSService(ITTSService):
    """
    Realistic Human-like TTS Service (ElevenLabs Architecture).
    Injects realistic speech pauses, breath marks, and custom voice models.
    """

    def __init__(self, api_key: Optional[str] = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.fallback = Pyttsx3TTSService()

    def add_human_pauses(self, text: str) -> str:
        """Injects micro-pauses after punctuation to emulate human breathing & cadence."""
        enhanced = text
        enhanced = enhanced.replace(", ", ", ... ")
        enhanced = enhanced.replace(". ", ". ... ")
        return enhanced

    def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        start_time = time.time()
        paced_text = self.add_human_pauses(text)

        if self.api_key:
            try:
                import requests
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
                headers = {
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                body = {
                    "text": paced_text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.6, "similarity_boost": 0.8}
                }
                resp = requests.post(url, json=body, headers=headers, timeout=10)
                if resp.status_code == 200:
                    if not output_path:
                        output_path = os.path.join(tempfile.gettempdir(), f"eleven_{int(time.time()*1000)}.mp3")
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    elapsed = round(time.time() - start_time, 3)
                    return TTSResult(
                        file_path=output_path,
                        audio_format="mp3",
                        latency_sec=elapsed,
                        provider="elevenlabs_v1",
                        status="success"
                    )
            except Exception as e:
                print(f"[ElevenLabs TTS] Cloud call failed: {e}. Falling back to local offline engine...")

        res = self.fallback.synthesize(paced_text, output_path=output_path)
        res.provider = "elevenlabs_human_cadence (local)"
        return res


class NeuralVoiceCloningTTSService(ITTSService):
    """
    Neural Text-to-Speech & Zero-Shot Voice Cloning Service.
    Implements end-to-end neural acoustic synthesis & vocoding based on
    'The Sound of AI' Monster TTS & Voice Cloning Course.

    Pipeline:
    1. Text Normalization (NSW expansion: numbers, currency, dates, acronyms)
    2. Conversational Prosody & Cadence Micro-Pause Injection (Breathing acoustics)
    3. Neural VITS ONNX Acoustic & Waveform Synthesis
    4. Target Speaker Conditioning & Acoustic Timbre Equalization
    """

    _cached_voice = None
    _cached_model_path = None

    def __init__(
        self,
        model_path: str = "models/piper/en_US-lessac-low.onnx",
        speaker_profile_id: str = "mira_neural",
        speech_rate: int = 175,
        with_human_pauses: bool = True,
        with_normalization: bool = True
    ):
        self.model_path = model_path
        self.speaker_profile_id = speaker_profile_id
        self.speech_rate = speech_rate
        self.with_human_pauses = with_human_pauses
        self.with_normalization = with_normalization
        self.fallback = Pyttsx3TTSService(speech_rate=speech_rate)

    @classmethod
    def get_or_load_voice(cls, model_path: str):
        """Lazy-loads and caches the neural ONNX model; auto-downloads if missing."""
        if cls._cached_voice is None or cls._cached_model_path != model_path:
            from piper.voice import PiperVoice
            if not os.path.exists(model_path):
                try:
                    import pathlib
                    from piper.download_voices import download_voice
                    out_dir = pathlib.Path(model_path).parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    voice_name = pathlib.Path(model_path).stem
                    download_voice(voice_name, out_dir)
                except Exception as dl_err:
                    print(f"[Neural TTS] Auto-download failed: {dl_err}")

            if os.path.exists(model_path):
                cls._cached_voice = PiperVoice.load(model_path)
                cls._cached_model_path = model_path
            else:
                return None
        return cls._cached_voice

    def add_human_pauses(self, text: str) -> str:
        """Injects prosodic breathing micro-pauses at grammatical boundaries."""
        enhanced = text
        enhanced = enhanced.replace(", ", ", ... ")
        enhanced = enhanced.replace("; ", "; ... ")
        enhanced = enhanced.replace(". ", ". ... ")
        return enhanced

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        speaker_profile_id: Optional[str] = None
    ) -> TTSResult:
        start_time = time.time()
        if not text or not text.strip():
            return TTSResult(
                file_path="",
                provider="neural_vits",
                status="skipped",
                latency_sec=0.0
            )

        active_profile_id = speaker_profile_id or self.speaker_profile_id
        profile = voice_cloner.get_profile(active_profile_id)
        if not profile:
            profile = voice_cloner.get_profile("mira_neural")

        # 1. Text Normalization
        processed_text = text
        norm_info = {}
        if self.with_normalization:
            norm_res = TextNormalizer.normalize(text)
            processed_text = norm_res["normalized_text"]
            norm_info = norm_res

        # 2. Conversational Prosody Pauses
        if self.with_human_pauses:
            processed_text = self.add_human_pauses(processed_text)

        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"neural_tts_{int(time.time()*1000)}.wav")

        # 3. Neural VITS Synthesis
        try:
            voice = self.get_or_load_voice(self.model_path)
            if voice is None:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            from piper.config import SynthesisConfig
            syn_cfg = SynthesisConfig(
                length_scale=profile.length_scale if profile else 1.0,
                noise_scale=profile.noise_scale if profile else 0.667,
                noise_w_scale=profile.noise_w_scale if profile else 0.8,
            )

            chunks = list(voice.synthesize(processed_text, syn_config=syn_cfg))
            if not chunks:
                raise RuntimeError("No audio chunks produced by neural model.")

            raw_chunk = chunks[0]
            temp_raw_path = output_path + ".raw.wav"

            with wave.open(temp_raw_path, "wb") as wf:
                wf.setnchannels(raw_chunk.sample_channels)
                wf.setsampwidth(raw_chunk.sample_width)
                wf.setframerate(raw_chunk.sample_rate)
                wf.writeframes(raw_chunk.audio_int16_bytes)

            # 4. Acoustic Timbre Conditioning for Cloned Speaker
            if profile:
                VoiceCloner.apply_acoustic_timbre_conditioning(temp_raw_path, profile, output_path)
                try:
                    os.remove(temp_raw_path)
                except Exception:
                    pass
            else:
                if os.path.exists(temp_raw_path):
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(temp_raw_path, output_path)

            elapsed = round(time.time() - start_time, 3)
            return TTSResult(
                file_path=output_path,
                audio_format="wav",
                latency_sec=elapsed,
                provider=f"neural_vits_{active_profile_id}",
                speech_rate=self.speech_rate,
                status="success",
                metadata={
                    "normalized_text": processed_text,
                    "speaker_profile": profile.to_dict() if profile else {},
                    "phonemes": raw_chunk.phonemes if hasattr(raw_chunk, "phonemes") else [],
                    "f0_hz": profile.fundamental_freq_f0 if profile else 175.0,
                    "spectral_centroid": profile.spectral_centroid if profile else 2000.0,
                    "gender": profile.gender_estimate if profile else "neutral"
                }
            )

        except Exception as e:
            # Fallback seamlessly to Pyttsx3
            print(f"[Neural TTS] Synthesis fallback due to: {e}")
            res = self.fallback.synthesize(processed_text, output_path=output_path)
            res.provider = f"pyttsx3_fallback ({active_profile_id})"
            res.metadata = {"fallback_reason": str(e), "normalized_text": processed_text}
            return res
