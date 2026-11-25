"""
Neural Voice Cloning & Speaker Acoustic Profiling Service.
Based on "Demystifying TTS and Voice Cloning" from the Neural Speech Course.

Implements Zero-Shot Voice Cloning via Speaker Embedding Extraction:
1. Respiration & Phonation acoustics (Fundamental Frequency F0 tracking, Harmonics, Formants).
2. Spectral envelope & Timbre extraction (Spectral Centroid, Flatness, Energy dynamics).
3. 128-Dimensional Acoustic Latent Embedding (d-vector / x-vector representation).
4. Acoustic Conditioning & DSP Formant Equalization for target cloned speaker.
"""

import os
import math
import json
import time
import wave
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class SpeakerProfile:
    """Acoustic identity profile representing a unique speaker's vocal tract and timbre."""
    profile_id: str
    name: str
    description: str
    gender_estimate: str
    fundamental_freq_f0: float          # Hz (Average vocal fold vibration rate)
    pitch_variability_std: float        # Hz (Dynamic pitch range)
    spectral_centroid: float            # Hz (Timbre brightness / vocal tract length)
    spectral_flatness: float            # Tonality (0.0 pure tone -> 1.0 noise)
    pitch_shift_semitones: float        # Relative semitone offset for vocoder
    length_scale: float                 # Cadence duration scaling (0.8 fast -> 1.2 slow)
    noise_scale: float                  # Vocal fold jitter / breathiness (0.4 - 0.9)
    noise_w_scale: float                # Phoneme duration stochasticity (0.6 - 1.0)
    embedding_vector: List[float]       # Acoustic or neural speaker embedding (512-D CAM++ or 128-D baseline)
    ref_audio_path: Optional[str] = None # Path to original recorded reference sample
    eleven_voice_id: Optional[str] = None # Cloned voice ID on ElevenLabs if synced
    encoder_backend: str = "random_proj_baseline" # "wespeaker_campp_lm" or "random_proj_baseline"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Summarize embedding vector for lightweight serialization
        d["embedding_dim"] = len(self.embedding_vector)
        d["embedding_preview"] = [round(x, 4) for x in self.embedding_vector[:8]]
        d["has_reference_audio"] = bool(self.ref_audio_path and os.path.exists(self.ref_audio_path))
        d["has_elevenlabs_clone"] = bool(self.eleven_voice_id)
        return d


class VoiceCloner:
    """
    Speaker Profiler and Neural Voice Cloning Manager.
    Extracts acoustic embeddings from 3-10 second reference audio
    and transforms neural synthesis into the cloned vocal identity.
    """

    PRESET_IDS = {"mira_neural", "alex_interviewer", "sarah_executive", "david_architect"}
    BASE_F0 = 175.0  # Base reference pitch of default Lessac model (Hz)
    DEFAULT_SPEAKER_ENCODER_PATH = "models/speaker/wespeaker_en_voxceleb_CAM++_LM.onnx"
    _cached_speaker_extractor = None
    _cached_speaker_model_path = None

    def __init__(self, profiles_dir: str = "storage/cloned_voices"):
        self.profiles_dir = profiles_dir
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.profiles: Dict[str, SpeakerProfile] = {}
        self._init_preset_profiles()
        self._load_saved_profiles()

    def _init_preset_profiles(self):
        """Initializes built-in archetypal speaker profiles based on vocal tract physics."""
        self.profiles["mira_neural"] = SpeakerProfile(
            profile_id="mira_neural",
            name="Mira (Warm Conversational)",
            description="Warm, empathetic AI conversationalist with natural micro-pauses and clear articulation.",
            gender_estimate="female",
            fundamental_freq_f0=218.0,
            pitch_variability_std=26.5,
            spectral_centroid=2480.0,
            spectral_flatness=0.045,
            pitch_shift_semitones=2.8,
            length_scale=0.98,
            noise_scale=0.667,
            noise_w_scale=0.8,
            embedding_vector=self._generate_synthetic_embedding(seed=101, f0=218.0, centroid=2480.0)
        )

        self.profiles["alex_interviewer"] = SpeakerProfile(
            profile_id="alex_interviewer",
            name="Alex (Technical Interviewer)",
            description="Authoritative, analytical interviewer voice with measured cadence and deep baritone warmth.",
            gender_estimate="male",
            fundamental_freq_f0=124.0,
            pitch_variability_std=14.2,
            spectral_centroid=1620.0,
            spectral_flatness=0.038,
            pitch_shift_semitones=-4.5,
            length_scale=1.06,
            noise_scale=0.55,
            noise_w_scale=0.72,
            embedding_vector=self._generate_synthetic_embedding(seed=102, f0=124.0, centroid=1620.0)
        )

        self.profiles["sarah_executive"] = SpeakerProfile(
            profile_id="sarah_executive",
            name="Sarah (Executive Recruiter)",
            description="Dynamic, articulate, high-energy executive presence with crisp upper-mid clarity.",
            gender_estimate="female",
            fundamental_freq_f0=238.0,
            pitch_variability_std=32.0,
            spectral_centroid=2850.0,
            spectral_flatness=0.052,
            pitch_shift_semitones=4.2,
            length_scale=0.92,
            noise_scale=0.72,
            noise_w_scale=0.85,
            embedding_vector=self._generate_synthetic_embedding(seed=103, f0=238.0, centroid=2850.0)
        )

        self.profiles["david_architect"] = SpeakerProfile(
            profile_id="david_architect",
            name="David (Systems Architect)",
            description="Deep, resonant bass timbre with deliberate pacing and grounded cadence.",
            gender_estimate="male",
            fundamental_freq_f0=98.0,
            pitch_variability_std=11.5,
            spectral_centroid=1380.0,
            spectral_flatness=0.032,
            pitch_shift_semitones=-6.8,
            length_scale=1.14,
            noise_scale=0.48,
            noise_w_scale=0.68,
            embedding_vector=self._generate_synthetic_embedding(seed=104, f0=98.0, centroid=1380.0)
        )

    def _generate_synthetic_embedding(self, seed: int, f0: float, centroid: float) -> List[float]:
        """Generates a deterministic 128-D acoustic speaker vector normalized to unit sphere."""
        rng = np.random.RandomState(seed)
        vec = rng.randn(128)
        # Condition initial latent dimensions on acoustic biology
        vec[0] = (f0 - 150.0) / 100.0
        vec[1] = (centroid - 2000.0) / 1000.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def _load_saved_profiles(self):
        """Loads custom user-cloned voices from persistent storage."""
        if not os.path.exists(self.profiles_dir):
            return
        for fname in os.listdir(self.profiles_dir):
            if fname.endswith(".json"):
                path = os.path.join(self.profiles_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        gender = data.get("gender_estimate", "neutral")
                        f0 = data["fundamental_freq_f0"]
                        # Auto-calibrate semitones relative to gender baseline (145Hz male / 218Hz female)
                        base_ref_f0 = 145.0 if (gender == "male" or f0 < 155.0) else 218.0
                        calibrated_shift = round(float(12.0 * math.log2(max(50.0, f0) / base_ref_f0)), 2)
                        calibrated_shift = max(-4.0, min(4.0, calibrated_shift))

                        ref_wav = data.get("ref_audio_path")
                        if not ref_wav:
                            candidate_wav = os.path.join(self.profiles_dir, f"{data['profile_id']}_ref.wav")
                            if os.path.exists(candidate_wav):
                                ref_wav = candidate_wav

                        prof = SpeakerProfile(
                            profile_id=data["profile_id"],
                            name=data["name"],
                            description=data.get("description", "Custom Cloned Voice"),
                            gender_estimate=gender,
                            fundamental_freq_f0=f0,
                            pitch_variability_std=data.get("pitch_variability_std", 20.0),
                            spectral_centroid=data["spectral_centroid"],
                            spectral_flatness=data.get("spectral_flatness", 0.04),
                            pitch_shift_semitones=calibrated_shift,
                            length_scale=data.get("length_scale", 1.0),
                            noise_scale=data.get("noise_scale", 0.667),
                            noise_w_scale=data.get("noise_w_scale", 0.8),
                            embedding_vector=data["embedding_vector"],
                            ref_audio_path=ref_wav,
                            eleven_voice_id=data.get("eleven_voice_id"),
                            encoder_backend=data.get("encoder_backend", "random_proj_baseline"),
                            created_at=data.get("created_at", time.time())
                        )
                        self.profiles[prof.profile_id] = prof
                except Exception as e:
                    print(f"[VoiceCloner] Error loading profile {fname}: {e}")

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Returns all available preset and user-cloned speaker profiles with preset flags."""
        result = []
        for p in self.profiles.values():
            d = p.to_dict()
            d["is_preset"] = p.profile_id in self.PRESET_IDS
            result.append(d)
        return result

    def get_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        return self.profiles.get(profile_id)

    def delete_profile(self, profile_id: str) -> bool:
        """
        Deletes a cloned or morphed profile from memory and disk storage.
        Built-in archetypal presets (mira, alex, sarah, david) are protected.
        """
        if profile_id in self.PRESET_IDS:
            raise ValueError(f"Cannot delete built-in archetypal preset '{profile_id}'")
        if profile_id not in self.profiles:
            return False

        prof = self.profiles[profile_id]

        # Remove from in-memory registry
        del self.profiles[profile_id]

        # Remove persistent JSON and reference audio file from storage
        path = os.path.join(self.profiles_dir, f"{profile_id}.json")
        ref_path = os.path.join(self.profiles_dir, f"{profile_id}_ref.wav")
        for p in (path, ref_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"[VoiceCloner] Error removing file {p}: {e}")

        # If ElevenLabs clone existed, clean it up if API key is in environment
        if prof and prof.eleven_voice_id and os.getenv("ELEVENLABS_API_KEY"):
            try:
                from .tts_services import ElevenLabsTTSService
                ElevenLabsTTSService.delete_voice(os.getenv("ELEVENLABS_API_KEY"), prof.eleven_voice_id)
            except Exception as el_err:
                print(f"[VoiceCloner] Notice: could not remove remote ElevenLabs voice: {el_err}")

        return True

    def rename_profile(self, profile_id: str, new_name: str) -> Optional[SpeakerProfile]:
        """Renames an existing profile and persists the update to disk."""
        if not new_name or not new_name.strip():
            raise ValueError("New voice name cannot be empty")
        new_name = new_name.strip()

        prof = self.get_profile(profile_id)
        if not prof:
            return None

        prof.name = new_name
        self.profiles[profile_id] = prof

        # Persist update to disk if it's a saved custom profile
        if profile_id not in self.PRESET_IDS:
            path = os.path.join(self.profiles_dir, f"{profile_id}.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(prof.to_dict(), f, indent=2)
            except Exception as e:
                print(f"[VoiceCloner] Error updating profile file {path}: {e}")

        return prof

    @classmethod
    def load_audio_signal(cls, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Loads any audio file (WAV, WebM, MP3, OGG, Opus, AAC, FLAC) into normalized mono float32 array.
        Seamlessly falls back to PyAV for non-standard or compressed formats (e.g. browser MediaRecorder).
        """
        try:
            with wave.open(audio_path, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

            if sample_width == 2:
                audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            if sample_rate != target_sr:
                target_len = int(len(audio) * target_sr / sample_rate)
                audio = signal.resample(audio, target_len)
                sample_rate = target_sr

            return audio, sample_rate
        except Exception:
            pass

        # Fallback to PyAV for WebM (Opus), MP3, OGG, AAC
        try:
            import av
            container = av.open(audio_path)
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            chunks = []
            for frame in container.decode(audio=0):
                for r in resampler.resample(frame):
                    chunks.append(r.to_ndarray()[0])

            if chunks:
                audio = np.concatenate(chunks, axis=0)
            else:
                audio = np.zeros(target_sr, dtype=np.float32)
            return audio, target_sr
        except Exception as e:
            raise ValueError(f"Unable to decode audio from {audio_path}: {e}")

    @classmethod
    def extract_acoustic_profile_from_wav(
        cls,
        wav_path: str,
        profile_name: str = "Cloned Voice",
        profile_id: Optional[str] = None,
        save_ref_dir: Optional[str] = None
    ) -> SpeakerProfile:
        """
        Analyzes a reference audio recording using acoustic DSP and psychoacoustics.
        Extracts fundamental frequency F0, pitch contour, spectral centroid,
        and generates a 128-D acoustic speaker embedding vector.
        Persists reference audio if save_ref_dir is supplied.
        Supports WAV, WebM, MP3, OGG, and other audio formats.
        """
        audio, sample_rate = cls.load_audio_signal(wav_path, target_sr=16000)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Ensure minimum duration for reliable extraction (pad or truncate)
        if len(audio) < sample_rate * 0.5:
            audio = np.tile(audio, int(np.ceil((sample_rate * 0.5) / max(len(audio), 1))))

        # 1. Pitch (F0) Extraction via Short-Time Autocorrelation
        frame_size = int(0.04 * sample_rate)  # 40ms frame
        hop_size = int(0.01 * sample_rate)    # 10ms hop
        min_lag = int(sample_rate / 400.0)    # 400 Hz upper pitch limit
        max_lag = int(sample_rate / 65.0)     # 65 Hz lower pitch limit

        f0_estimates = []
        for start in range(0, len(audio) - frame_size, hop_size * 2):
            frame = audio[start:start + frame_size]
            energy = np.sum(frame ** 2)
            if energy < 0.001:  # Skip unvoiced / silence frames
                continue
            windowed = frame * np.hanning(len(frame))
            autocorr = signal.correlate(windowed, windowed, mode="full")
            autocorr = autocorr[len(windowed) - 1:]

            # Search for peak within biological vocal range
            search_region = autocorr[min_lag:max_lag]
            if len(search_region) > 0 and np.max(search_region) > 0.3 * autocorr[0]:
                peak_idx = np.argmax(search_region) + min_lag
                f0 = sample_rate / peak_idx
                if 65.0 <= f0 <= 400.0:
                    f0_estimates.append(f0)

        if f0_estimates:
            f0_mean = float(np.median(f0_estimates))
            f0_std = float(np.std(f0_estimates))
        else:
            f0_mean = 160.0
            f0_std = 18.0

        # 2. Spectral Analysis (Centroid, Flatness)
        freqs, psd = signal.welch(audio, fs=sample_rate, nperseg=min(len(audio), 2048))
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            spectral_centroid = float(np.sum(freqs * psd) / psd_sum)
            # Spectral flatness = geometric mean / arithmetic mean
            geom_mean = np.exp(np.mean(np.log(psd + 1e-12)))
            arith_mean = np.mean(psd) + 1e-12
            spectral_flatness = float(min(1.0, geom_mean / arith_mean))
        else:
            spectral_centroid = 2000.0
            spectral_flatness = 0.04

        # 3. Classify gender estimate & semitone pitch shift relative to baseline model
        gender_est = "female" if f0_mean > 175.0 else ("male" if f0_mean < 155.0 else "neutral")
        base_ref_f0 = 145.0 if gender_est == "male" else 218.0
        semitones = float(12.0 * math.log2(max(50.0, f0_mean) / base_ref_f0))

        # Clamp semitones to safe natural limits (-3.0 to +3.0 semitones)
        semitones = max(-3.0, min(3.0, semitones))

        # 4. Cadence & Stochastic Variability
        length_scale = float(1.0 - (semitones * 0.015))  # Higher pitch speakers tend to speak slightly faster
        length_scale = max(0.90, min(1.15, length_scale))
        noise_scale = float(max(0.50, min(0.80, 0.667 + spectral_flatness * 1.5)))
        noise_w_scale = float(max(0.70, min(0.90, 0.8 + (f0_std / 120.0))))

        # 5. Extract Speaker Embedding (WeSpeaker CAM++ LM neural encoder if available, else 128-D baseline)
        embedding, encoder_backend = cls.extract_speaker_embedding(
            audio,
            sample_rate=sample_rate,
            use_neural=True,
            freqs=freqs,
            psd=psd,
            f0_mean=f0_mean
        )

        final_pid = profile_id or f"clone_{int(time.time()*1000)}"

        # Persist reference audio if target directory requested
        ref_path = None
        if save_ref_dir:
            os.makedirs(save_ref_dir, exist_ok=True)
            ref_path = os.path.join(save_ref_dir, f"{final_pid}_ref.wav")
            try:
                # Save as clean 16kHz mono WAV
                out_pcm = (audio * 32767.0).astype(np.int16)
                with wave.open(ref_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(out_pcm.tobytes())
            except Exception as ref_err:
                print(f"[VoiceCloner] Warning: Failed to persist reference audio: {ref_err}")
                ref_path = None

        return SpeakerProfile(
            profile_id=final_pid,
            name=profile_name,
            description=f"Cloned voice from audio ({gender_est}, F0: {f0_mean:.1f}Hz, Centroid: {spectral_centroid:.0f}Hz)",
            gender_estimate=gender_est,
            fundamental_freq_f0=round(f0_mean, 1),
            pitch_variability_std=round(f0_std, 1),
            spectral_centroid=round(spectral_centroid, 1),
            spectral_flatness=round(spectral_flatness, 4),
            pitch_shift_semitones=round(semitones, 2),
            length_scale=round(length_scale, 2),
            noise_scale=round(noise_scale, 3),
            noise_w_scale=round(noise_w_scale, 3),
            embedding_vector=embedding,
            ref_audio_path=ref_path,
            encoder_backend=encoder_backend
        )

    @classmethod
    def _get_speaker_extractor(cls, model_path: Optional[str] = None):
        """Lazy-loads and caches the sherpa-onnx WeSpeaker CAM++ speaker embedding extractor."""
        target_path = model_path or cls.DEFAULT_SPEAKER_ENCODER_PATH
        if not os.path.exists(target_path):
            return None
        if cls._cached_speaker_extractor is not None and cls._cached_speaker_model_path == target_path:
            return cls._cached_speaker_extractor
        try:
            import sherpa_onnx
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=target_path, num_threads=2, debug=False, provider="cpu"
            )
            if not cfg.validate():
                return None
            cls._cached_speaker_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            cls._cached_speaker_model_path = target_path
            return cls._cached_speaker_extractor
        except Exception as e:
            print(f"[VoiceCloner] Neural speaker extractor unavailable ({e}); using baseline.")
            return None

    @classmethod
    def extract_speaker_embedding(
        cls,
        audio: np.ndarray,
        sample_rate: int = 16000,
        use_neural: bool = True,
        freqs: Optional[np.ndarray] = None,
        psd: Optional[np.ndarray] = None,
        f0_mean: float = 160.0
    ) -> Tuple[List[float], str]:
        """Extracts unit-normalized speaker embedding using WeSpeaker CAM++ LM ONNX or 128-D Mel baseline."""
        audio_1d = np.asarray(audio, dtype=np.float32).reshape(-1)
        if use_neural:
            extractor = cls._get_speaker_extractor()
            if extractor is not None and audio_1d.size > 0:
                stream = extractor.create_stream()
                stream.accept_waveform(int(sample_rate), audio_1d)
                stream.input_finished()
                if extractor.is_ready(stream):
                    raw_emb = np.asarray(extractor.compute(stream), dtype=np.float32)
                    norm = float(np.linalg.norm(raw_emb))
                    if norm > 1e-12:
                        raw_emb = raw_emb / norm
                    return raw_emb.tolist(), "wespeaker_campp_lm"

        if freqs is None or psd is None:
            freqs, psd = signal.welch(audio_1d, fs=sample_rate, nperseg=min(max(len(audio_1d), 8), 2048))
        mel_filters = cls._compute_mel_energies(freqs, psd, n_mels=64)
        rng = np.random.RandomState(int(f0_mean * 100) % 99999)
        projection_matrix = rng.randn(64, 128)
        embedding_raw = np.dot(mel_filters, projection_matrix)
        norm = float(np.linalg.norm(embedding_raw))
        embedding = (embedding_raw / norm).tolist() if norm > 0 else [0.0] * 128
        return embedding, "random_proj_baseline"

    @classmethod
    def _compute_mel_energies(cls, freqs: np.ndarray, psd: np.ndarray, n_mels: int = 64) -> np.ndarray:
        """Decomposes power spectrum into perceptual Mel-scale energy bins."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(0)
        max_mel = hz_to_mel(max(freqs[-1], 4000))
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        mel_energies = np.zeros(n_mels)
        for m in range(1, n_mels + 1):
            left = hz_points[m - 1]
            center = hz_points[m]
            right = hz_points[m + 1]

            # Linear interpolation triangular filter
            filter_weights = np.maximum(0.0, np.minimum((freqs - left) / (center - left + 1e-6),
                                                        (right - freqs) / (right - center + 1e-6)))
            mel_energies[m - 1] = np.sum(psd * filter_weights)

        # Log mel energy
        return np.log(mel_energies + 1e-8)

    def save_cloned_profile(self, profile: SpeakerProfile) -> str:
        """Saves a cloned profile to disk and registers it in memory."""
        self.profiles[profile.profile_id] = profile
        out_path = os.path.join(self.profiles_dir, f"{profile.profile_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2)
        return out_path

    def morph_profiles(
        self,
        profile_id_a: str,
        profile_id_b: str,
        alpha: float = 0.5,
        custom_name: Optional[str] = None
    ) -> SpeakerProfile:
        """
        Interpolates in latent acoustic space between two speaker profiles.
        Based on Module 5 (Speaker Embeddings & Latent Space Interpolation).
        alpha = 0.0 -> 100% Profile A; alpha = 1.0 -> 100% Profile B.
        """
        prof_a = self.get_profile(profile_id_a) or self.profiles["mira_neural"]
        prof_b = self.get_profile(profile_id_b) or self.profiles["alex_interviewer"]
        alpha = max(0.0, min(1.0, float(alpha)))

        # 1. Linear interpolation of scalar acoustic parameters
        f0 = (1.0 - alpha) * prof_a.fundamental_freq_f0 + alpha * prof_b.fundamental_freq_f0
        f0_std = (1.0 - alpha) * prof_a.pitch_variability_std + alpha * prof_b.pitch_variability_std
        centroid = (1.0 - alpha) * prof_a.spectral_centroid + alpha * prof_b.spectral_centroid
        flatness = (1.0 - alpha) * prof_a.spectral_flatness + alpha * prof_b.spectral_flatness
        shift = (1.0 - alpha) * prof_a.pitch_shift_semitones + alpha * prof_b.pitch_shift_semitones
        length = (1.0 - alpha) * prof_a.length_scale + alpha * prof_b.length_scale
        noise = (1.0 - alpha) * prof_a.noise_scale + alpha * prof_b.noise_scale
        noise_w = (1.0 - alpha) * prof_a.noise_w_scale + alpha * prof_b.noise_w_scale

        # 2. Spherical normalized interpolation of embedding vectors (dimension-safe across 128-D & 512-D)
        va = np.array(prof_a.embedding_vector, dtype=np.float32)
        vb = np.array(prof_b.embedding_vector, dtype=np.float32)
        if va.size != vb.size:
            max_dim = max(va.size, vb.size)
            va = np.pad(va, (0, max_dim - va.size))
            vb = np.pad(vb, (0, max_dim - vb.size))
        v_blend = (1.0 - alpha) * va + alpha * vb
        norm = np.linalg.norm(v_blend)
        if norm > 0:
            v_blend = v_blend / norm

        encoder_backend = (
            prof_a.encoder_backend
            if prof_a.encoder_backend == prof_b.encoder_backend
            else "hybrid_morphed"
        )

        gender = "female" if f0 > 180.0 else ("male" if f0 < 145.0 else "neutral")
        pct_b = int(alpha * 100)
        pct_a = 100 - pct_b
        name = custom_name or f"Morph: {prof_a.name.split()[0]} ({pct_a}%) + {prof_b.name.split()[0]} ({pct_b}%)"
        morph_id = f"morph_{prof_a.profile_id[:4]}_{prof_b.profile_id[:4]}_{int(alpha*100)}"

        morphed_profile = SpeakerProfile(
            profile_id=morph_id,
            name=name,
            description=f"Acoustic latent morph between {prof_a.name} and {prof_b.name} (alpha={alpha:.2f})",
            gender_estimate=gender,
            fundamental_freq_f0=round(float(f0), 1),
            pitch_variability_std=round(float(f0_std), 1),
            spectral_centroid=round(float(centroid), 1),
            spectral_flatness=round(float(flatness), 4),
            pitch_shift_semitones=round(float(shift), 2),
            length_scale=round(float(length), 2),
            noise_scale=round(float(noise), 3),
            noise_w_scale=round(float(noise_w), 3),
            embedding_vector=[float(x) for x in v_blend],
            encoder_backend=encoder_backend
        )

        self.save_cloned_profile(morphed_profile)
        return morphed_profile

    @classmethod
    def _pitch_shift_wsola(cls, audio: np.ndarray, orig_sr: int, semitones: float) -> np.ndarray:
        """
        Shifts fundamental pitch by `semitones` while preserving time duration
        using Waveform Similarity Overlap-Add (WSOLA).
        """
        if abs(semitones) < 0.2:
            return audio
        ratio = 2.0 ** (semitones / 12.0)
        target_len = int(len(audio) / ratio)
        if target_len < 512:
            return audio
        resampled = signal.resample(audio, target_len)

        # Stretch or compress time back to original length
        s = float(len(audio)) / float(len(resampled))
        hop_out = 256
        window_size = 1024
        win = np.hanning(window_size).astype(np.float32)

        out_len = len(audio)
        output = np.zeros(out_len + window_size * 2, dtype=np.float32)
        weights = np.zeros(out_len + window_size * 2, dtype=np.float32)

        in_pos = 0.0
        out_pos = 0
        search_range = 64

        while int(in_pos) + window_size + search_range < len(resampled) and out_pos + window_size < len(output):
            nominal_in = int(in_pos)
            best_offset = 0
            if out_pos > 0:
                ref = output[out_pos : out_pos + search_range]
                cands = resampled[nominal_in : nominal_in + search_range * 2]
                if len(ref) == search_range and len(cands) >= search_range:
                    corr = np.correlate(cands, ref, mode='valid')
                    if len(corr) > 0:
                        best_offset = int(np.argmax(corr))

            actual_in = nominal_in + best_offset
            if actual_in + window_size <= len(resampled):
                output[out_pos : out_pos + window_size] += resampled[actual_in : actual_in + window_size] * win
                weights[out_pos : out_pos + window_size] += win

            in_pos += hop_out / s
            out_pos += hop_out

        mask = weights > 1e-4
        output[mask] /= weights[mask]
        return output[:out_len]

    @classmethod
    def apply_acoustic_timbre_conditioning(cls, wav_path: str, profile: SpeakerProfile, output_path: str) -> str:
        """
        Applies gentle vocal tract conditioning to match target speaker timbre:
        - For built-in presets (mira, alex, sarah, david), preserves pure neural audio untouched.
        - For custom clones, uses high-quality WSOLA overlap-add to preserve natural cadence and phoneme timing.
        - Subtle presence / warmth equalization without artificial distortion.
        """
        # Built-in presets are calibrated neural models; pass through untouched
        if profile.profile_id in cls.PRESET_IDS:
            if wav_path != output_path:
                import shutil
                shutil.copyfile(wav_path, output_path)
            return output_path

        audio, orig_sr = cls.load_audio_signal(wav_path, target_sr=16000)
        channels = 1

        target_f0 = profile.fundamental_freq_f0
        base_ref_f0 = 145.0 if (profile.gender_estimate == "male" or target_f0 < 155.0) else 218.0
        semitones = float(12.0 * math.log2(max(50.0, target_f0) / base_ref_f0))
        semitones = max(-2.5, min(2.5, semitones))

        # Use WSOLA overlap-add (maintains duration and phoneme speed)
        if abs(semitones) > 0.4:
            audio = cls._pitch_shift_wsola(audio, orig_sr, semitones)

        # Equalization filter based on spectral centroid and gender
        nyquist = orig_sr / 2.0
        if profile.gender_estimate == "male" or profile.fundamental_freq_f0 < 155.0:
            # Low-frequency chest resonance warmth (160 Hz)
            boost_freq = min(160.0, nyquist - 50)
            b, a = signal.iirpeak(boost_freq, Q=2.0, fs=orig_sr)
            audio = audio + 0.15 * signal.lfilter(b, a, audio)
        elif profile.gender_estimate == "female" or profile.fundamental_freq_f0 > 185.0:
            # Upper-mid vocal presence (3.2 kHz)
            air_freq = min(3200.0, nyquist - 50)
            b, a = signal.iirpeak(air_freq, Q=2.0, fs=orig_sr)
            audio = audio + 0.15 * signal.lfilter(b, a, audio)

        # Normalize and prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)

        # Convert back to 16-bit PCM WAV
        out_pcm = (audio * 32767.0).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(orig_sr)
            wf.writeframes(out_pcm.tobytes())

        return output_path


# Global singleton instance for voice cloning
voice_cloner = VoiceCloner()
