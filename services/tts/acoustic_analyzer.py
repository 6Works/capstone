"""
Acoustic Analysis & Perceptual Spectrogram Generator.
Based on Module 1 (Human Speech Physics & Psychoacoustics) from
'The Monster Text-to-Speech & Voice Cloning Course' by Valerio Velardo.

Implements:
1. Short-Time Fourier Transform (STFT) with Hanning windowing.
2. 80-Band Mel-scale filterbank energy mapping (80 Hz to 8000 Hz).
3. Frame-by-frame Fundamental Frequency F0(t) pitch contour extraction.
4. RMS Energy and Spectral Centroid dynamics.
"""

import math
import wave
import numpy as np
import scipy.signal as signal
from typing import Dict, Any, List, Optional, Tuple


class AcousticAnalyzer:
    """
    Analyzes acoustic speech signals into perceptual representations
    matching the human auditory system (cochlear critical bands)
    and modern acoustic models (e.g., FastSpeech, Tacotron 2, VITS).
    """

    @classmethod
    def hz_to_mel(cls, hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @classmethod
    def mel_to_hz(cls, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def create_mel_filterbank(cls, sr: int, n_fft: int = 1024, n_mels: int = 80, f_min: float = 80.0, f_max: float = 7600.0) -> np.ndarray:
        """Constructs an 80-band triangular Mel-scale filterbank matrix."""
        f_max = min(f_max, sr / 2.0)
        mel_min = cls.hz_to_mel(f_min)
        mel_max = cls.hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = [cls.mel_to_hz(m) for m in mel_points]

        bin_points = np.floor((n_fft + 1) * np.array(hz_points) / sr).astype(int)
        filterbank = np.zeros((n_mels, int(n_fft // 2 + 1)))

        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                filterbank[m - 1, k] = (k - f_m_minus) / max(1, f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                filterbank[m - 1, k] = (f_m_plus - k) / max(1, f_m_plus - f_m)

        return filterbank

    @classmethod
    def analyze_wav(cls, wav_path: str, max_frames: int = 160) -> Dict[str, Any]:
        """
        Extracts 80-band Mel-Spectrogram and Pitch F0(t) from an audio file.
        Supports WAV, WebM, MP3, OGG, and other formats.
        Returns visualizable telemetry for real-time frontend canvas rendering.
        """
        from .voice_cloner import VoiceCloner
        audio, sr = VoiceCloner.load_audio_signal(wav_path, target_sr=16000)

        # Remove DC offset
        audio = audio - np.mean(audio)
        duration_sec = len(audio) / sr

        # Short-Time Fourier Transform parameters
        n_fft = 1024
        hop_length = int(sr * 0.0125)  # 12.5ms hop
        win_length = int(sr * 0.030)   # 30ms window
        window = np.hanning(win_length)

        # STFT computation
        frames = []
        f0_contour = []
        energy_contour = []

        min_lag = int(sr / 400.0)  # 400 Hz
        max_lag = int(sr / 65.0)   # 65 Hz

        for start in range(0, len(audio) - win_length, hop_length):
            segment = audio[start:start + win_length]
            rms = float(np.sqrt(np.mean(segment ** 2)))
            energy_contour.append(round(rms, 4))

            # Autocorrelation for F0 pitch contour
            windowed = segment * window
            autocorr = signal.correlate(windowed, windowed, mode="full")[len(windowed) - 1:]
            
            f0_val = 0.0
            if rms > 0.01:
                search_region = autocorr[min_lag:max_lag]
                if len(search_region) > 0 and np.max(search_region) > 0.35 * autocorr[0]:
                    peak = np.argmax(search_region) + min_lag
                    f0_candidate = sr / peak
                    if 65.0 <= f0_candidate <= 400.0:
                        f0_val = round(float(f0_candidate), 1)
            f0_contour.append(f0_val)

            # Pad or truncate windowed frame to n_fft
            padded = np.zeros(n_fft)
            padded[:win_length] = windowed
            fft_mag = np.abs(np.fft.rfft(padded))
            frames.append(fft_mag ** 2)

        if not frames:
            return {"status": "empty", "duration_sec": 0.0}

        power_spec = np.array(frames).T  # Shape: [n_fft//2 + 1, n_frames]
        mel_filters = cls.create_mel_filterbank(sr, n_fft=n_fft, n_mels=80)
        mel_spec = np.dot(mel_filters, power_spec)  # Shape: [80, n_frames]

        # Log-dynamic range compression (dB)
        log_mel = 10.0 * np.log10(np.maximum(1e-5, mel_spec))
        # Normalize between 0.0 and 1.0 for rendering
        min_db = -50.0
        max_db = float(np.max(log_mel))
        norm_mel = np.clip((log_mel - min_db) / max(1.0, (max_db - min_db)), 0.0, 1.0)

        # Downsample frames along time axis if exceeds max_frames for lightweight JSON transfer
        n_frames = norm_mel.shape[1]
        if n_frames > max_frames:
            step = n_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            norm_mel = norm_mel[:, indices]
            f0_contour = [f0_contour[i] for i in indices]
            energy_contour = [energy_contour[i] for i in indices]

        voiced_f0 = [f for f in f0_contour if f > 0]
        mean_f0 = float(np.mean(voiced_f0)) if voiced_f0 else 0.0

        # Round mel values to 2 decimals
        mel_grid = [[round(float(val), 2) for val in row] for row in norm_mel]

        return {
            "status": "success",
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sr,
            "num_mel_bands": 80,
            "num_time_frames": len(f0_contour),
            "mel_spectrogram": mel_grid,
            "f0_contour": f0_contour,
            "energy_contour": energy_contour,
            "mean_f0_hz": round(mean_f0, 1),
            "max_f0_hz": round(float(np.max(voiced_f0)), 1) if voiced_f0 else 0.0,
            "min_f0_hz": round(float(np.min(voiced_f0)), 1) if voiced_f0 else 0.0
        }
