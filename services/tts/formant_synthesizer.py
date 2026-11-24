"""
Historical Formant Synthesizer (Klatt Source-Filter Model).
Based on Module 3 (Traditional Speech Synthesis) from
'The Monster Text-to-Speech & Voice Cloning Course' by Valerio Velardo.

Simulates rule-based formant acoustic synthesis (1980s DECtalk / Klatt paradigm):
- Periodic glottal pulse train excitation (Rosenberg pulse)
- White noise friction generator for unvoiced phonemes
- Cascaded digital biquad resonators for formant frequencies F1, F2, F3
"""

import os
import wave
import tempfile
import numpy as np
import scipy.signal as signal
from typing import Dict, Any, List, Optional, Tuple


class FormantSynthesizer:
    """
    Simulates early acoustic speech synthesis using cascaded resonators
    filtering glottal pulse vibrations. Demonstrates the source-filter model.
    """

    # Vowel Formant Targets (F1, F2, F3 in Hz, Bandwidths B1, B2, B3 in Hz)
    VOWEL_FORMANTS = {
        "i": {"f1": 280, "f2": 2250, "f3": 3000, "b1": 50, "b2": 100, "b3": 120, "name": "beet (/i/)"},
        "e": {"f1": 530, "f2": 1840, "f3": 2480, "b1": 60, "b2": 90,  "b3": 110, "name": "bet (/ɛ/)"},
        "a": {"f1": 730, "f2": 1090, "f3": 2440, "b1": 80, "b2": 90,  "b3": 130, "name": "father (/ɑː/)"},
        "o": {"f1": 570, "f2": 840,  "f3": 2410, "b1": 70, "b2": 80,  "b3": 100, "name": "boat (/oʊ/)"},
        "u": {"f1": 300, "f2": 870,  "f3": 2240, "b1": 50, "b2": 70,  "b3": 110, "name": "boot (/uː/)"},
    }

    @classmethod
    def rosenberg_glottal_pulse(cls, sr: int, f0: float, duration: float) -> np.ndarray:
        """Generates glottal volume velocity pulse train (Rosenberg pulse excitation)."""
        n_samples = int(sr * duration)
        period = int(sr / f0)
        pulse = np.zeros(n_samples, dtype=np.float32)

        n1 = int(0.4 * period)  # Opening phase
        n2 = int(0.16 * period) # Closing phase

        for p_start in range(0, n_samples - period, period):
            for t in range(period):
                idx = p_start + t
                if idx >= n_samples:
                    break
                if t < n1:
                    pulse[idx] = 0.5 * (1.0 - np.cos(np.pi * t / max(1, n1)))
                elif t < n1 + n2:
                    pulse[idx] = np.cos(0.5 * np.pi * (t - n1) / max(1, n2))
                else:
                    pulse[idx] = 0.0

        return pulse

    @classmethod
    def apply_digital_resonator(cls, x: np.ndarray, sr: int, freq: float, bw: float) -> np.ndarray:
        """Applies a 2nd-order digital IIR resonator filter for a formant resonance."""
        r = np.exp(-np.pi * bw / sr)
        theta = 2.0 * np.pi * freq / sr
        a1 = -2.0 * r * np.cos(theta)
        a2 = r * r
        gain = 1.0 + a1 + a2
        b = np.array([gain, 0.0, 0.0])
        a = np.array([1.0, a1, a2])
        return signal.lfilter(b, a, x)

    @classmethod
    def synthesize_vowel(cls, vowel_key: str = "a", f0: float = 130.0, duration: float = 0.8, sr: int = 16000) -> str:
        """
        Synthesizes a vowel using the Klatt cascade formant model.
        Returns path to generated WAV file.
        """
        vowel = cls.VOWEL_FORMANTS.get(vowel_key.lower(), cls.VOWEL_FORMANTS["a"])

        # 1. Glottal source excitation
        glottal = cls.rosenberg_glottal_pulse(sr, f0, duration)

        # 2. Add subtle aspiration breath noise (-30 dB)
        noise = np.random.randn(len(glottal)) * 0.015
        source = glottal + noise

        # 3. Cascaded resonators for F1, F2, F3, and F4 (lip radiation)
        y1 = cls.apply_digital_resonator(source, sr, vowel["f1"], vowel["b1"])
        y2 = cls.apply_digital_resonator(y1, sr, vowel["f2"], vowel["b2"])
        y3 = cls.apply_digital_resonator(y2, sr, vowel["f3"], vowel["b3"])
        y4 = cls.apply_digital_resonator(y3, sr, 3500.0, 200.0)

        # 4. Lip radiation high-pass filter (differentiation: 1 - 0.95 z^-1)
        audio = signal.lfilter([1.0, -0.95], [1.0], y4)

        # Envelope shaping (smooth attack & release)
        attack_len = int(sr * 0.04)
        release_len = int(sr * 0.06)
        env = np.ones(len(audio))
        env[:attack_len] = np.linspace(0, 1, attack_len)
        env[-release_len:] = np.linspace(1, 0, release_len)
        audio = audio * env

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.85 / peak)

        out_pcm = (audio * 32767.0).astype(np.int16)
        out_path = os.path.join(tempfile.gettempdir(), f"formant_{vowel_key}_{int(f0)}_{int(np.random.randint(9999))}.wav")
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(out_pcm.tobytes())

        return out_path

    @classmethod
    def list_vowels(cls) -> List[Dict[str, Any]]:
        """Returns metadata for all supported formant vowel configurations."""
        return [{"key": k, **v} for k, v in cls.VOWEL_FORMANTS.items()]
