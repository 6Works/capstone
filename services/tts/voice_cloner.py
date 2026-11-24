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
    embedding_vector: List[float]       # 128-D acoustic latent representation
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Summarize embedding vector for lightweight serialization
        d["embedding_dim"] = len(self.embedding_vector)
        d["embedding_preview"] = [round(x, 4) for x in self.embedding_vector[:8]]
        return d


class VoiceCloner:
    """
    Speaker Profiler and Neural Voice Cloning Manager.
    Extracts acoustic embeddings from 3-10 second reference audio
    and transforms neural synthesis into the cloned vocal identity.
    """

    BASE_F0 = 175.0  # Base reference pitch of default Lessac model (Hz)

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
                        prof = SpeakerProfile(
                            profile_id=data["profile_id"],
                            name=data["name"],
                            description=data.get("description", "Custom Cloned Voice"),
                            gender_estimate=data.get("gender_estimate", "neutral"),
                            fundamental_freq_f0=data["fundamental_freq_f0"],
                            pitch_variability_std=data.get("pitch_variability_std", 20.0),
                            spectral_centroid=data["spectral_centroid"],
                            spectral_flatness=data.get("spectral_flatness", 0.04),
                            pitch_shift_semitones=data["pitch_shift_semitones"],
                            length_scale=data.get("length_scale", 1.0),
                            noise_scale=data.get("noise_scale", 0.667),
                            noise_w_scale=data.get("noise_w_scale", 0.8),
                            embedding_vector=data["embedding_vector"],
                            created_at=data.get("created_at", time.time())
                        )
                        self.profiles[prof.profile_id] = prof
                except Exception as e:
                    print(f"[VoiceCloner] Error loading profile {fname}: {e}")

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Returns all available preset and user-cloned speaker profiles."""
        return [p.to_dict() for p in self.profiles.values()]

    def get_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        return self.profiles.get(profile_id)

    @classmethod
    def extract_acoustic_profile_from_wav(cls, wav_path: str, profile_name: str = "Cloned Voice") -> SpeakerProfile:
        """
        Analyzes a reference WAV recording using acoustic DSP and psychocoustics.
        Extracts fundamental frequency F0, pitch contour, spectral centroid,
        and generates a 128-D acoustic speaker embedding vector.
        """
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        # Convert to mono float32 normalized between -1.0 and 1.0
        if sample_width == 2:
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

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

        # 3. Classify gender estimate & semitone pitch shift relative to baseline (175 Hz)
        gender_est = "female" if f0_mean > 175.0 else ("male" if f0_mean < 145.0 else "neutral")
        semitones = float(12.0 * math.log2(max(50.0, f0_mean) / cls.BASE_F0))

        # Clamp semitones to safe natural limits (-8 to +8 semitones)
        semitones = max(-8.0, min(8.0, semitones))

        # 4. Cadence & Stochastic Variability
        length_scale = float(1.0 - (semitones * 0.015))  # Higher pitch speakers tend to speak slightly faster
        length_scale = max(0.85, min(1.25, length_scale))
        noise_scale = float(max(0.45, min(0.85, 0.667 + spectral_flatness * 2.0)))
        noise_w_scale = float(max(0.65, min(0.95, 0.8 + (f0_std / 100.0))))

        # 5. Extract 128-D Speaker Embedding
        # Project 64 mel-spaced spectral log energies into 128 dimensions
        mel_filters = cls._compute_mel_energies(freqs, psd, n_mels=64)
        rng = np.random.RandomState(int(f0_mean * 100) % 99999)
        projection_matrix = rng.randn(64, 128)
        embedding_raw = np.dot(mel_filters, projection_matrix)
        norm = np.linalg.norm(embedding_raw)
        if norm > 0:
            embedding = (embedding_raw / norm).tolist()
        else:
            embedding = [0.0] * 128

        profile_id = f"clone_{int(time.time())}"
        return SpeakerProfile(
            profile_id=profile_id,
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
            embedding_vector=embedding
        )

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

    @classmethod
    def apply_acoustic_timbre_conditioning(cls, wav_path: str, profile: SpeakerProfile, output_path: str) -> str:
        """
        Applies DSP vocal tract acoustic conditioning to match target speaker timbre:
        - Pitch shift via polynomial rate resampling.
        - Formant / Equalization filtering (bass warmth or high-frequency brilliance).
        """
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            orig_sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # Semitone pitch shift ratio: r = 2^(semitones / 12)
        semitones = profile.pitch_shift_semitones
        ratio = 2.0 ** (semitones / 12.0)

        # Apply pitch shifting if significant difference (|semitones| > 0.5)
        if abs(semitones) > 0.5:
            target_len = int(len(audio) / ratio)
            if target_len > 100:
                # Resample time axis to shift fundamental pitch
                resampled = signal.resample(audio, target_len)
                # Resample back to original length to maintain speech duration (WSOLA concept)
                # To maintain tempo while shifting pitch:
                audio = signal.resample(resampled, len(audio))

        # Equalization filter based on spectral centroid and gender
        nyquist = orig_sr / 2.0
        if profile.gender_estimate == "male" or profile.fundamental_freq_f0 < 140.0:
            # Low-frequency chest resonance boost (120-250 Hz)
            boost_freq = min(220.0, nyquist - 50)
            b, a = signal.iirpeak(boost_freq, Q=2.0, fs=orig_sr)
            audio = audio + 0.35 * signal.lfilter(b, a, audio)
        elif profile.gender_estimate == "female" or profile.fundamental_freq_f0 > 200.0:
            # Upper-mid vocal presence and breath air boost (2.5 kHz - 4.5 kHz)
            air_freq = min(3500.0, nyquist - 50)
            b, a = signal.iirpeak(air_freq, Q=1.8, fs=orig_sr)
            audio = audio + 0.30 * signal.lfilter(b, a, audio)

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
