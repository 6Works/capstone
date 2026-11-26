"""
Upgraded Speaker Verification & Cross-Channel / Dual-Encoder Audit:
Compares 3 encoders cleanly across identical splits:
  1. RandomProj-128D (Welch Mel PSD random projection baseline)
  2. WeSpeaker CAM++ LM (512-D ONNX, CC-BY-4.0 / Apache-2.0 sherpa-onnx)
  3. WeSpeaker ECAPA512_LM (512-D ECAPA-TDNN ONNX, CC-BY-4.0)

Evaluates:
  - librispeech_60 (clean & 20 dB AWGN)
  - True Cross-Channel Degradation (enroll/A clean vs B passed through 300-3400 Hz telephony bandpass + 18 dB AWGN + -6 dB gain)
  - librispeech_other_60 (challenging noisy/accented unseen speakers)
Reports encoder-comparable EER, ROC-AUC, and dimensionless Fisher Discriminability d'
plus bounded paired bootstrap tests (B=2000, p >= 0.001 floor).
"""

import os
import sys
import json
import time
import itertools
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.signal as signal

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.tts.voice_cloner import VoiceCloner
from ml.datasets import load_manifest, add_awgn
from ml.metrics import (
    compute_speaker_similarity,
    compute_verification_stats,
    paired_bootstrap_test,
)


def _apply_channel_shift(audio: np.ndarray, sr: int = 16000, seed: int = 101) -> np.ndarray:
    """Applies telephone-like bandpass (300-3400 Hz) + 18 dB AWGN + -6 dB gain attenuation."""
    sos = signal.butter(4, [300.0, 3400.0], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, np.asarray(audio, dtype=np.float32)).astype(np.float32)
    filtered = add_awgn(filtered * 0.5, snr_db=18.0, seed=seed)
    return filtered


def _extract_all(clips: List[Dict[str, Any]], signals: List[Tuple[np.ndarray, int]], encoder_cfg: Dict[str, Any]):
    use_neural = encoder_cfg["use_neural"]
    model_path = encoder_cfg.get("model_path")

    t_init = time.perf_counter()
    if use_neural:
        VoiceCloner._get_speaker_extractor(model_path=model_path)
    cold_ms = round((time.perf_counter() - t_init) * 1000.0, 2)

    embs = []
    latencies_ms = []
    backend_name = ""
    for audio, sr in signals:
        t0 = time.perf_counter()
        emb, backend_name = VoiceCloner.extract_speaker_embedding(
            audio, sample_rate=sr, use_neural=use_neural, model_path=model_path
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        embs.append(emb)

    return embs, backend_name, cold_ms, round(float(np.mean(latencies_ms)), 2)


def _eval_within_channel(clips: List[Dict[str, Any]], signals: List[Tuple[np.ndarray, int]], encoder_cfg: Dict[str, Any]) -> Dict[str, Any]:
    embs, backend_name, cold_ms, mean_lat = _extract_all(clips, signals, encoder_cfg)
    same_scores, diff_scores, pair_margins = [], [], []
    for i, j in itertools.combinations(range(len(clips)), 2):
        sim = compute_speaker_similarity(embs[i], embs[j])
        is_same = int(clips[i]["speaker_id"] == clips[j]["speaker_id"])
        pair_margins.append(sim if is_same else -sim)
        if is_same:
            same_scores.append(sim)
        else:
            diff_scores.append(sim)

    stats = compute_verification_stats(same_scores, diff_scores)
    stats.update({
        "backend": backend_name,
        "embedding_dim": len(embs[0]) if embs else 0,
        "cold_init_ms": cold_ms,
        "mean_latency_ms": mean_lat,
        "pair_margins": [round(float(x), 4) for x in pair_margins],
    })
    return stats


def _eval_cross_channel(
    clips: List[Dict[str, Any]],
    clean_signals: List[Tuple[np.ndarray, int]],
    shifted_signals: List[Tuple[np.ndarray, int]],
    encoder_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Pairs Clean enrollment/utterance i strictly against Channel-Shifted utterance j (i != j)."""
    embs_clean, backend_name, cold_ms, mean_lat = _extract_all(clips, clean_signals, encoder_cfg)
    embs_shift, _, _, _ = _extract_all(clips, shifted_signals, encoder_cfg)

    same_scores, diff_scores, pair_margins = [], [], []
    for i in range(len(clips)):
        for j in range(len(clips)):
            if clips[i]["id"] == clips[j]["id"]:
                continue  # Exclude identical utterance
            sim = compute_speaker_similarity(embs_clean[i], embs_shift[j])
            is_same = int(clips[i]["speaker_id"] == clips[j]["speaker_id"])
            pair_margins.append(sim if is_same else -sim)
            if is_same:
                same_scores.append(sim)
            else:
                diff_scores.append(sim)

    stats = compute_verification_stats(same_scores, diff_scores)
    stats.update({
        "backend": backend_name,
        "embedding_dim": len(embs_clean[0]) if embs_clean else 0,
        "cold_init_ms": cold_ms,
        "mean_latency_ms": mean_lat,
        "pair_margins": [round(float(x), 4) for x in pair_margins],
    })
    return stats


def main():
    ls_clean = load_manifest("librispeech_60")["items"]  # 12 speakers x 5 utts = 60 clips
    ls_clean_sigs = [VoiceCloner.load_audio_signal(os.path.join(ROOT_DIR, c["audio_path"]), target_sr=16000) for c in ls_clean]
    ls_awgn_sigs = [(add_awgn(a, 20.0, seed=42 + idx), sr) for idx, (a, sr) in enumerate(ls_clean_sigs)]
    ls_xchan_sigs = [(_apply_channel_shift(a, sr=sr, seed=100 + idx), sr) for idx, (a, sr) in enumerate(ls_clean_sigs)]

    encoders = [
        ("baseline_random_proj", {"use_neural": False}),
        ("neural_wespeaker_campp", {"use_neural": True, "model_path": VoiceCloner.DEFAULT_SPEAKER_ENCODER_PATH}),
    ]
    if os.path.exists(VoiceCloner.ECAPA_SPEAKER_ENCODER_PATH):
        encoders.append(("neural_wespeaker_ecapa512", {"use_neural": True, "model_path": VoiceCloner.ECAPA_SPEAKER_ENCODER_PATH}))

    results: Dict[str, Any] = {
        "experiment": "speaker_encoder_dual_audit_v2",
        "num_clips_clean": len(ls_clean),
        "num_speakers": len({c["speaker_id"] for c in ls_clean}),
        "conditions": {},
    }

    for cond_name, mode, sig_a, sig_b in [
        ("clean_librispeech_60", "within", ls_clean_sigs, None),
        ("awgn_20db_librispeech_60", "within", ls_awgn_sigs, None),
        ("cross_channel_telephony_eq", "cross", ls_clean_sigs, ls_xchan_sigs),
    ]:
        cond_out: Dict[str, Any] = {}
        margins_by_enc: Dict[str, List[float]] = {}
        for enc_id, enc_cfg in encoders:
            if mode == "within":
                res = _eval_within_channel(ls_clean, sig_a, enc_cfg)
            else:
                res = _eval_cross_channel(ls_clean, sig_a, sig_b, enc_cfg)
            margins_by_enc[enc_id] = res.pop("pair_margins")
            cond_out[enc_id] = res

        base_m = margins_by_enc["baseline_random_proj"]
        for neural_id in [k for k in margins_by_enc if k != "baseline_random_proj"]:
            cond_out[f"bootstrap_{neural_id}_vs_random_proj"] = paired_bootstrap_test(
                base_m, margins_by_enc[neural_id], n_resamples=2000, seed=42, higher_is_better=True
            )
        results["conditions"][cond_name] = cond_out

    # Evaluate on real noisy LibriSpeech test-other if present
    other_path = os.path.join(ROOT_DIR, "ml", "datasets", "manifests", "librispeech_other_60.json")
    if os.path.exists(other_path):
        ls_other = load_manifest("librispeech_other_60")["items"]
        other_sigs = [VoiceCloner.load_audio_signal(os.path.join(ROOT_DIR, c["audio_path"]), target_sr=16000) for c in ls_other]
        other_xchan = [(_apply_channel_shift(a, sr=sr, seed=300 + idx), sr) for idx, (a, sr) in enumerate(other_sigs)]
        for cond_name, mode, sig_a, sig_b in [
            ("clean_librispeech_other_60", "within", other_sigs, None),
            ("cross_channel_other_60", "cross", other_sigs, other_xchan),
        ]:
            cond_out = {}
            margins_by_enc = {}
            for enc_id, enc_cfg in encoders:
                res = _eval_within_channel(ls_other, sig_a, enc_cfg) if mode == "within" else _eval_cross_channel(ls_other, sig_a, sig_b, enc_cfg)
                margins_by_enc[enc_id] = res.pop("pair_margins")
                cond_out[enc_id] = res
            base_m = margins_by_enc["baseline_random_proj"]
            for neural_id in [k for k in margins_by_enc if k != "baseline_random_proj"]:
                cond_out[f"bootstrap_{neural_id}_vs_random_proj"] = paired_bootstrap_test(
                    base_m, margins_by_enc[neural_id], n_resamples=2000, seed=42, higher_is_better=True
                )
            results["conditions"][cond_name] = cond_out

    out_file = os.path.join(ROOT_DIR, "ml", "results", "speaker_encoder_eval.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved upgraded speaker encoder evaluation -> {out_file}")
    for cond, data in results["conditions"].items():
        print(f"\n[{cond}]")
        for k, v in data.items():
            if not k.startswith("bootstrap_"):
                print(f"  {k:28s}: AUC={v['roc_auc']:.4f} EER={v['eer']:.4f} d'={v['d_prime']:.3f} (same={v['same_mean']} diff={v['diff_mean']})")


if __name__ == "__main__":
    main()
