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


class EdgeTTSService(ITTSService):
    """
    Microsoft Azure Neural Speech Service via edge-tts (100% Free, Studio Quality).
    Features ultra-natural phrasing, breathing pauses, and multilingual support
    including native Indian English accents and proper name pronunciation.
    """

    DEFAULT_VOICE = "en-US-GuyNeural"

    VOICE_CATALOG = {
        "prabhat_in_male": {
            "id": "en-IN-PrabhatNeural",
            "name": "Prabhat (Indian English Male - Fluent & Natural)",
            "gender": "male"
        },
        "neerja_in_female": {
            "id": "en-IN-NeerjaNeural",
            "name": "Neerja (Indian English Female - Expressive)",
            "gender": "female"
        },
        "guy_us_male": {
            "id": "en-US-GuyNeural",
            "name": "Guy (US Male - Conversational & Clear)",
            "gender": "male"
        },
        "jenny_us_female": {
            "id": "en-US-JennyNeural",
            "name": "Jenny (US Female - Warm & Articulate)",
            "gender": "female"
        },
        "aria_us_female": {
            "id": "en-US-AriaNeural",
            "name": "Aria (US Female - Professional Executive)",
            "gender": "female"
        },
        "ryan_gb_male": {
            "id": "en-GB-RyanNeural",
            "name": "Ryan (UK Male - Authoritative Interviewer)",
            "gender": "male"
        }
    }

    def __init__(self, voice: str = "en-US-GuyNeural", speech_rate: str = "+0%"):
        self.voice = voice
        self.speech_rate = speech_rate
        self.fallback = Pyttsx3TTSService()

    def synthesize(self, text: str, output_path: Optional[str] = None, voice: Optional[str] = None) -> TTSResult:
        start_time = time.time()
        if not text or not text.strip():
            return TTSResult(
                file_path="",
                provider="edge_tts",
                status="skipped",
                latency_sec=0.0
            )

        target_voice = voice or self.voice
        # If user passed a catalog key, resolve to full Azure voice identifier
        if target_voice in self.VOICE_CATALOG:
            target_voice = self.VOICE_CATALOG[target_voice]["id"]

        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"edge_{int(time.time()*1000)}.mp3")

        try:
            import asyncio
            import edge_tts

            async def _synthesize_edge():
                comm = edge_tts.Communicate(text, target_voice, rate=self.speech_rate)
                await comm.save(output_path)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(lambda: asyncio.run(_synthesize_edge())).result()
                else:
                    loop.run_until_complete(_synthesize_edge())
            except RuntimeError:
                asyncio.run(_synthesize_edge())

            elapsed = round(time.time() - start_time, 3)
            return TTSResult(
                file_path=output_path,
                audio_format="mp3",
                latency_sec=elapsed,
                provider=f"edge_tts_{target_voice}",
                status="success",
                metadata={"voice": target_voice, "speech_rate": self.speech_rate}
            )
        except Exception as e:
            print(f"[EdgeTTSService] Synthesis failed: {e}. Falling back to pyttsx3...")
            res = self.fallback.synthesize(text, output_path=output_path)
            res.provider = f"pyttsx3_fallback (edge_tts: {e})"
            return res


class ElevenLabsTTSService(ITTSService):
    """
    Realistic Human-like TTS & Voice Cloning Service (ElevenLabs Architecture).
    Supports instant zero-shot voice cloning from audio files and synthesis
    using cloned or premade voices with emotional prosody.
    """

    def __init__(self, api_key: Optional[str] = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.fallback = Pyttsx3TTSService()

    @classmethod
    def clone_voice(cls, api_key: str, name: str, reference_audio_path: str, description: str = "Cloned Voice via Capstone") -> Optional[str]:
        """
        Uploads a reference audio clip to ElevenLabs /v1/voices/add API.
        Returns the created voice_id for immediate zero-shot synthesis.
        """
        if not api_key:
            return None
        import requests
        try:
            url = "https://api.elevenlabs.io/v1/voices/add"
            headers = {"xi-api-key": api_key}
            with open(reference_audio_path, "rb") as f:
                files = {"files": (os.path.basename(reference_audio_path), f, "audio/wav")}
                data = {"name": name, "description": description}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=25)
                if resp.status_code == 200:
                    voice_id = resp.json().get("voice_id")
                    print(f"[ElevenLabs] Successfully created cloned voice '{name}' (ID: {voice_id})")
                    return voice_id
                else:
                    print(f"[ElevenLabs] Voice clone failed ({resp.status_code}): {resp.text}")
                    return None
        except Exception as e:
            print(f"[ElevenLabs] Voice cloning error: {e}")
            return None

    @classmethod
    def delete_voice(cls, api_key: str, voice_id: str) -> bool:
        """Deletes a cloned voice from ElevenLabs."""
        if not api_key or not voice_id:
            return False
        import requests
        try:
            url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
            resp = requests.delete(url, headers={"xi-api-key": api_key}, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def add_human_pauses(self, text: str) -> str:
        """Injects micro-pauses after punctuation to emulate human breathing & cadence."""
        enhanced = text
        enhanced = enhanced.replace(", ", ", ... ")
        enhanced = enhanced.replace(". ", ". ... ")
        return enhanced

    def synthesize(self, text: str, output_path: Optional[str] = None, voice_id: Optional[str] = None) -> TTSResult:
        start_time = time.time()
        paced_text = self.add_human_pauses(text)
        active_voice_id = voice_id or self.voice_id

        if self.api_key:
            try:
                import requests
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{active_voice_id}"
                headers = {
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                body = {
                    "text": paced_text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.55, "similarity_boost": 0.82}
                }
                resp = requests.post(url, json=body, headers=headers, timeout=15)
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
                        provider=f"elevenlabs_{active_voice_id}",
                        status="success"
                    )
                else:
                    print(f"[ElevenLabs TTS] HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"[ElevenLabs TTS] Cloud call failed: {e}. Falling back to local offline engine...")

        res = self.fallback.synthesize(paced_text, output_path=output_path)
        res.provider = "elevenlabs_fallback (local pyttsx3)"
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

        # Select base neural acoustic model matching speaker gender
        gender = profile.gender_estimate if profile else "neutral"
        male_model_path = "models/piper/en_US-ryan-low.onnx"
        if gender == "male" and os.path.exists(male_model_path):
            active_model = male_model_path
        else:
            active_model = self.model_path

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
            voice = self.get_or_load_voice(active_model)
            if voice is None:
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

            first_chunk = chunks[0]
            temp_raw_path = output_path + ".raw.wav"

            with wave.open(temp_raw_path, "wb") as wf:
                wf.setnchannels(first_chunk.sample_channels)
                wf.setsampwidth(first_chunk.sample_width)
                wf.setframerate(first_chunk.sample_rate)
                for chunk in chunks:
                    wf.writeframes(chunk.audio_int16_bytes)

            all_phonemes = []
            for chunk in chunks:
                if hasattr(chunk, "phonemes") and chunk.phonemes:
                    all_phonemes.extend(chunk.phonemes)

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
                    "phonemes": all_phonemes if all_phonemes else (first_chunk.phonemes if hasattr(first_chunk, "phonemes") else []),
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
