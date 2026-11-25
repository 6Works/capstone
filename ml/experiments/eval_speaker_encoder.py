"""
Session 2 Experiment: Speaker Verification & Embedding Quality Benchmark.
Compares baseline RandomProj-128D vs. WeSpeaker CAM++ LM (512-D ONNX)
on real unseen multi-speaker LibriSpeech test-clean utterances (clean & 20 dB AWGN).
"""

import os
import sys
import json
import time
import itertools
import numpy as np
import requests

# Ensure project root is importable when executed directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.tts.voice_cloner import VoiceCloner
from ml.metrics import compute_speaker_similarity, paired_bootstrap_test


def _fetch_librispeech_probe(cache_dir: str, num_speakers: int = 4, utts_per_speaker: int = 3):
    """Downloads a tiny cached multi-speaker FLAC subset from openslr/librispeech_asr test-clean."""
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= num_speakers * utts_per_speaker and all(os.path.exists(d["path"]) for d in data):
            return data

    by_speaker = {}
    offset = 0
    while len([s for s, items in by_speaker.items() if len(items) >= utts_per_speaker]) < num_speakers and offset < 500:
        url = f"https://datasets-server.huggingface.co/rows?dataset=openslr/librispeech_asr&config=clean&split=test&offset={offset}&length=100"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            spk = str(row["speaker_id"])
            bucket = by_speaker.setdefault(spk, [])
            if len(bucket) < utts_per_speaker:
                audio_url = row["audio"][0]["src"]
                local_path = os.path.join(cache_dir, f"spk_{spk}_{row['id']}.flac")
                if not os.path.exists(local_path):
                    ar = requests.get(audio_url, timeout=30)
                    ar.raise_for_status()
                    with open(local_path, "wb") as af:
                        af.write(ar.content)
                bucket.append({"speaker_id": spk, "utt_id": str(row["id"]), "path": local_path, "text": row["text"]})
        offset += 100

    selected = []
    for spk, items in by_speaker.items():
        if len(items) >= utts_per_speaker:
            selected.extend(items[:utts_per_speaker])
            if len(selected) >= num_speakers * utts_per_speaker:
                break

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    return selected


def _add_awgn(audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Adds deterministic Additive White Gaussian Noise calibrated to target SNR (dB)."""
    rng = np.random.RandomState(seed)
    sig_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def _evaluate_backend(clips, signals, use_neural: bool):
    """Computes verification cosine distributions, ROC-AUC, EER, and warm latency."""
    from sklearn.metrics import roc_auc_score, roc_curve

    # Measure cold initialization separately from steady-state utterance inference
    t_init = time.perf_counter()
    if use_neural:
        VoiceCloner._get_speaker_extractor()
    cold_init_ms = round((time.perf_counter() - t_init) * 1000.0, 2)

    embs = []
    latencies_ms = []
    backend_name = ""
    for audio, sr in signals:
        t0 = time.perf_counter()
        emb, backend_name = VoiceCloner.extract_speaker_embedding(audio, sample_rate=sr, use_neural=use_neural)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        embs.append(emb)

    y_true = []
    y_scores = []
    same_scores = []
    diff_scores = []
    lowest_same_pair = None

    for i, j in itertools.combinations(range(len(clips)), 2):
        sim = compute_speaker_similarity(embs[i], embs[j])
        is_same = int(clips[i]["speaker_id"] == clips[j]["speaker_id"])
        y_true.append(is_same)
        y_scores.append(sim)
        if is_same:
            same_scores.append(sim)
            if lowest_same_pair is None or sim < lowest_same_pair["cos_sim"]:
                lowest_same_pair = {
                    "spk": clips[i]["speaker_id"],
                    "utt_a": clips[i]["utt_id"],
                    "utt_b": clips[j]["utt_id"],
                    "cos_sim": sim,
                }
        else:
            diff_scores.append(sim)

    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    return {
        "backend": backend_name,
        "embedding_dim": len(embs[0]) if embs else 0,
        "same_speaker_mean": round(float(np.mean(same_scores)), 4),
        "same_speaker_std": round(float(np.std(same_scores, ddof=1)), 4),
        "diff_speaker_mean": round(float(np.mean(diff_scores)), 4),
        "diff_speaker_std": round(float(np.std(diff_scores, ddof=1)), 4),
        "separation_gap": round(float(np.mean(same_scores) - np.mean(diff_scores)), 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "cold_init_ms": cold_init_ms,
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "lowest_same_pair": lowest_same_pair,
        "num_same_pairs": len(same_scores),
        "num_diff_pairs": len(diff_scores),
        "pair_margins": [round(float(s if yt == 1 else -s), 4) for yt, s in zip(y_true, y_scores)],
    }


def main():
    cfg_path = os.path.join(ROOT_DIR, "ml", "configs", "speaker_encoder_campp.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cache_dir = os.path.join(ROOT_DIR, "storage", "benchmarks", "librispeech_speaker_probe")
    clips = _fetch_librispeech_probe(
        cache_dir=cache_dir,
        num_speakers=cfg["num_speakers"],
        utts_per_speaker=cfg["utts_per_speaker"],
    )

    clean_signals = [VoiceCloner.load_audio_signal(c["path"], target_sr=16000) for c in clips]
    noisy_signals = [(_add_awgn(a, cfg["noise_snr_db"], seed=cfg["seed"]), sr) for a, sr in clean_signals]

    results = {"config": cfg, "conditions": {}}
    for cond_name, sigs in [("clean", clean_signals), (f"awgn_{int(cfg['noise_snr_db'])}db", noisy_signals)]:
        base_res = _evaluate_backend(clips, sigs, use_neural=False)
        neur_res = _evaluate_backend(clips, sigs, use_neural=True)
        sig_test = paired_bootstrap_test(
            baseline_scores=base_res.pop("pair_margins"),
            candidate_scores=neur_res.pop("pair_margins"),
            n_resamples=1000,
            seed=cfg["seed"],
            higher_is_better=True,
        )
        results["conditions"][cond_name] = {
            "baseline_random_proj": base_res,
            "neural_wespeaker_campp": neur_res,
            "paired_bootstrap_discrimination": sig_test,
        }

    out_dir = os.path.join(ROOT_DIR, "ml", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "speaker_encoder_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved speaker encoder evaluation -> {out_file}")
    for cond, row in results["conditions"].items():
        b, n, p = row["baseline_random_proj"], row["neural_wespeaker_campp"], row["paired_bootstrap_discrimination"]
        print(f"\n[{cond}]")
        print(f"  Baseline (128-D): same={b['same_speaker_mean']} diff={b['diff_speaker_mean']} gap={b['separation_gap']} AUC={b['roc_auc']} EER={b['eer']} lat={b['mean_latency_ms']}ms")
        print(f"  CAM++ LM (512-D): same={n['same_speaker_mean']} diff={n['diff_speaker_mean']} gap={n['separation_gap']} AUC={n['roc_auc']} EER={n['eer']} lat={n['mean_latency_ms']}ms")
        print(f"  Bootstrap sig   : improvement={p['observed_improvement']} 95%CI=[{p['ci_95_low']}, {p['ci_95_high']}] p={p['p_value']}")


if __name__ == "__main__":
    main()
