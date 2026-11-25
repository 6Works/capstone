"""
Session 7 Multi-Modal Explainability Suite (Layers A & C).
Generates 3 publication-grade 300 DPI figures under ml/results/figures/:
  1. fig1_asr_word_confidence_heatmap.png  (Layer A word probability drop & contextual biasing recovery across SNRs)
  2. fig2_campp_spectral_saliency.png      (Layer C 8-band spectral occlusion sensitivity of WeSpeaker CAM++ cosine similarity)
  3. fig3_speaker_embedding_pca_tsne.png   (Layer C 2D embedding comparison: RandomProj-128D vs WeSpeaker CAM++-512D + Pocket TTS clones)
"""

import os
import sys
import json
import tempfile
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.asr import WhisperTransformerASR, build_dynamic_bias_prompt
from services.contracts import AudioPayload
from services.tts import VoiceCloner, PocketTTSCloningService, voice_cloner
from ml.datasets import load_manifest, add_awgn
from ml.metrics import compute_speaker_similarity
from ml.metrics.eval_metrics import _normalize_text
from ml.experiments.run_asr_ablation import INTERVIEW_ROLE_TERMS, CANDIDATE_ROSTER_NAMES


FIG_DIR = os.path.join(ROOT_DIR, "ml", "results", "figures")


def plot_asr_word_confidence_heatmap() -> str:
    """
    Plots per-word Whisper decoding probability across decoding variants & AWGN SNRs
    for a representative proper-noun utterance ('Priya benchmarked CTranslate2 quantization on Kubernetes nodes.').
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    items = load_manifest("interview_stress_v1")["items"]
    target_item = next(x for x in items if x["id"] == "pn_05")
    ref_words = target_item["text"].rstrip(".").split()

    wav_16k, sr = VoiceCloner.load_audio_signal(os.path.join(ROOT_DIR, target_item["audio_path"]), target_sr=16000)
    wav_20db = add_awgn(wav_16k, 20.0, seed=42)
    wav_10db = add_awgn(wav_16k, 10.0, seed=142)

    lex = INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES
    dyn_prompt = build_dynamic_bias_prompt(INTERVIEW_ROLE_TERMS, CANDIDATE_ROSTER_NAMES)

    tiny_base = WhisperTransformerASR(model_size="tiny.en", compute_type="int8", vad_filter=False, initial_prompt="", word_timestamps=True)
    tiny_imp = WhisperTransformerASR(
        model_size="tiny.en", compute_type="int8", vad_filter=True,
        initial_prompt=dyn_prompt, word_timestamps=True, repair_lexicon=lex,
        condition_on_previous_text=False, repetition_penalty=1.1, no_repeat_ngram_size=4,
        hallucination_silence_threshold=0.4,
    )
    small_imp = WhisperTransformerASR(
        model_size="small.en", compute_type="int8", vad_filter=True,
        initial_prompt=dyn_prompt, hotwords=" ".join(lex), word_timestamps=True, repair_lexicon=lex,
    )

    conditions = [
        ("tiny baseline (Clean)", tiny_base, wav_16k),
        ("tiny baseline (20 dB)", tiny_base, wav_20db),
        ("tiny baseline (10 dB)", tiny_base, wav_10db),
        ("tiny + Layer A (10 dB)", tiny_imp, wav_10db),
        ("small + Layer A (10 dB)", small_imp, wav_10db),
    ]

    matrix = np.zeros((len(conditions), len(ref_words)), dtype=np.float32)
    cell_labels: List[List[str]] = []

    for r_idx, (label, asr_svc, sig) in enumerate(conditions):
        res = asr_svc.transcribe(
            AudioPayload(
                source_type="stream_buffer", sample_rate=16000,
                raw_bytes=sig.astype(np.float32).tobytes(),
                metadata={"audio_dtype": "float32"},
            )
        )
        words_meta = res.metadata.get("words") or []
        hyp_norm_tokens = _normalize_text(res.transcript).split()
        row_labs: List[str] = []
        for c_idx, rw in enumerate(ref_words):
            rw_norm = _normalize_text(rw)
            # Align approximate position in words_meta and check exact match vs substitution
            matched_prob = 0.0
            exact_hit = rw_norm in hyp_norm_tokens
            window = words_meta[max(0, c_idx - 2) : min(len(words_meta), c_idx + 4)]
            best_w = None
            for w in window:
                wn = _normalize_text(w["word"])
                if wn == rw_norm or rw_norm[:3] in wn:
                    best_w = w
                    break
            if best_w is not None:
                matched_prob = float(best_w["probability"])
            elif c_idx < len(words_meta):
                matched_prob = float(words_meta[c_idx]["probability"]) * (0.5 if not exact_hit else 1.0)
            matrix[r_idx, c_idx] = min(1.0, max(0.0, matched_prob))
            tag = "OK" if exact_hit else "ERR"
            row_labs.append(f"{matrix[r_idx, c_idx]:.2f}\n[{tag}]")
        cell_labels.append(row_labs)

    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(ref_words)))
    ax.set_xticklabels(ref_words, fontsize=9.5, fontweight="bold", rotation=18, ha="right")
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels([c[0] for c in conditions], fontsize=9.5)
    for r in range(len(conditions)):
        for c in range(len(ref_words)):
            val = matrix[r, c]
            txt_color = "white" if val < 0.35 else "black"
            ax.text(c, r, cell_labels[r][c], ha="center", va="center", color=txt_color, fontsize=8.0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Token Decoding Probability", fontsize=9)
    ax.set_title("Layer A Explainability: Word-Level Confidence & Entity Recovery Across AWGN Stress", fontsize=11, pad=10)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_asr_word_confidence_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _bandstop_filter(wav: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Applies zero-phase Butterworth band-stop masking over [low_hz, high_hz] to measure spectral saliency."""
    nyq = 0.5 * sr
    hi = min(high_hz, nyq - 50.0)
    lo = max(20.0, low_hz)
    if lo >= hi:
        return wav.copy()
    if lo <= 25.0:
        sos = signal.butter(4, hi / nyq, btype="highpass", output="sos")
    else:
        sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandstop", output="sos")
    return signal.sosfiltfilt(sos, wav).astype(np.float32)


def plot_campp_spectral_saliency() -> Tuple[str, Dict[str, Any]]:
    """
    Computes 8-band spectral occlusion saliency on WeSpeaker CAM++ cosine similarity
    across all 12 unseen LibriSpeech enroll speakers, and renders a 300-DPI saliency chart.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    bands = [
        ("0-300 Hz\n(F0 Pitch)", 20.0, 300.0),
        ("300-600 Hz\n(F1 Low)", 300.0, 600.0),
        ("600-1.0 kHz\n(F1 High)", 600.0, 1000.0),
        ("1.0-1.8 kHz\n(F2 Vowel)", 1000.0, 1800.0),
        ("1.8-3.0 kHz\n(F3 Formant)", 1800.0, 3000.0),
        ("3.0-4.5 kHz\n(Speaker Presence)", 3000.0, 4500.0),
        ("4.5-6.5 kHz\n(Consonants)", 4500.0, 6500.0),
        ("6.5-7.9 kHz\n(Air / Fricative)", 6500.0, 7900.0),
    ]
    enroll_items = [x for x in load_manifest("librispeech_60")["items"] if x["split"] == "enroll"]
    drops_by_band = np.zeros((len(enroll_items), len(bands)), dtype=np.float32)

    for s_idx, it in enumerate(enroll_items):
        wav, sr = VoiceCloner.load_audio_signal(os.path.join(ROOT_DIR, it["audio_path"]), target_sr=16000)
        emb_clean, _ = VoiceCloner.extract_speaker_embedding(wav, sample_rate=sr, use_neural=True)
        for b_idx, (_, lo, hi) in enumerate(bands):
            masked = _bandstop_filter(wav, sr, lo, hi)
            emb_masked, _ = VoiceCloner.extract_speaker_embedding(masked, sample_rate=sr, use_neural=True)
            drops_by_band[s_idx, b_idx] = 1.0 - compute_speaker_similarity(emb_clean, emb_masked)

    means = np.mean(drops_by_band, axis=0)
    stds = np.std(drops_by_band, axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(10.0, 4.4), dpi=300)
    x = np.arange(len(bands))
    colors = ["#c0392b" if m >= np.percentile(means, 70) else "#2980b9" for m in means]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#1f2937", linewidth=0.7, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bands], fontsize=8.5)
    ax.set_ylabel("Mean Cosine Similarity Drop (1 - cos)", fontsize=9.5)
    ax.set_title("Layer C Explainability: WeSpeaker CAM++ 8-Band Spectral Occlusion Saliency (12 Speakers)", fontsize=10.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig2_campp_spectral_saliency.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    summary = {
        bands[i][0].replace("\n", " "): {"mean_drop": round(float(means[i]), 4), "std": round(float(stds[i]), 4)}
        for i in range(len(bands))
    }
    return out_path, summary


def _cosine_separation_gap(embeddings: np.ndarray, labels: List[int]) -> Tuple[float, float, float]:
    """Computes mean same-speaker cosine, diff-speaker cosine, and separation gap."""
    embs = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    normed = embs / norms
    sim_mat = normed @ normed.T
    same_sims, diff_sims = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            (same_sims if labels[i] == labels[j] else diff_sims).append(float(sim_mat[i, j]))
    s_mean = float(np.mean(same_sims))
    d_mean = float(np.mean(diff_sims))
    return round(s_mean, 4), round(d_mean, 4), round(s_mean - d_mean, 4)


def plot_speaker_embedding_projection() -> Tuple[str, Dict[str, Any]]:
    """
    Side-by-side 2D t-SNE projection across 60 clean + 60 channel-EQ-tilted LibriSpeech clips
    (12 unseen speakers, 120 points) + Pocket TTS clones (*), reporting cosine separation gap.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    items = load_manifest("librispeech_60")["items"]
    speakers = sorted({x["speaker_id"] for x in items})
    spk_to_idx = {s: i for i, s in enumerate(speakers)}

    emb_base: List[List[float]] = []
    emb_cam: List[List[float]] = []
    labels: List[int] = []
    is_eq: List[bool] = []

    for idx_it, it in enumerate(items):
        wav, sr = VoiceCloner.load_audio_signal(os.path.join(ROOT_DIR, it["audio_path"]), target_sr=16000)
        alpha = 0.45 if (idx_it % 2 == 0) else -0.35
        wav_eq = signal.lfilter([1.0, -alpha], [1.0], wav).astype(np.float32) if alpha > 0 else signal.lfilter([1.0], [1.0, alpha], wav).astype(np.float32)
        wav_eq = wav_eq / (np.max(np.abs(wav_eq)) + 1e-6) * (np.max(np.abs(wav)) + 1e-6)
        for flag_eq, sig in ((False, wav), (True, wav_eq)):
            eb, _ = VoiceCloner.extract_speaker_embedding(sig, sample_rate=sr, use_neural=False)
            ec, _ = VoiceCloner.extract_speaker_embedding(sig, sample_rate=sr, use_neural=True)
            emb_base.append(eb)
            emb_cam.append(ec)
            labels.append(spk_to_idx[it["speaker_id"]])
            is_eq.append(flag_eq)

    clone_embs: List[List[float]] = []
    clone_labels: List[int] = []
    pocket = PocketTTSCloningService(num_steps=5)
    tmp_dir = tempfile.mkdtemp(prefix="tsne_clones_")
    try:
        enroll_map = {x["speaker_id"]: os.path.join(ROOT_DIR, x["audio_path"]) for x in items if x["split"] == "enroll"}
        for spk in speakers[:6]:
            prof = VoiceCloner.extract_acoustic_profile_from_wav(enroll_map[spk], profile_id=f"tsne_{spk}", save_ref_dir=None)
            prof.ref_audio_path = enroll_map[spk]
            voice_cloner.profiles[prof.profile_id] = prof
            out_w = os.path.join(tmp_dir, f"clone_{spk}.wav")
            pocket.synthesize("Testing neural speaker identity alignment.", output_path=out_w, speaker_profile_id=prof.profile_id)
            if os.path.exists(out_w):
                cw, sr = VoiceCloner.load_audio_signal(out_w, target_sr=16000)
                ce, _ = VoiceCloner.extract_speaker_embedding(cw, sample_rate=sr, use_neural=True)
                clone_embs.append(ce)
                clone_labels.append(spk_to_idx[spk])
            voice_cloner.profiles.pop(prof.profile_id, None)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    X_base = np.asarray(emb_base, dtype=np.float32)
    X_cam = np.asarray(emb_cam, dtype=np.float32)
    X_cam_all = np.vstack([X_cam, np.asarray(clone_embs, dtype=np.float32)]) if clone_embs else X_cam

    Z_base = TSNE(n_components=2, perplexity=12, random_state=42, metric="cosine").fit_transform(X_base)
    Z_cam_all = TSNE(n_components=2, perplexity=12, random_state=42, metric="cosine").fit_transform(X_cam_all)
    Z_cam = Z_cam_all[:len(X_cam)]
    Z_clones = Z_cam_all[len(X_cam):]

    b_same, b_diff, b_gap = _cosine_separation_gap(X_base, labels)
    c_same, c_diff, c_gap = _cosine_separation_gap(X_cam, labels)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(speakers)))
    eq_arr = np.asarray(is_eq)
    lbl_arr = np.asarray(labels)

    for idx, spk in enumerate(speakers):
        m_clean = (lbl_arr == idx) & (~eq_arr)
        m_eq = (lbl_arr == idx) & eq_arr
        axes[0].scatter(Z_base[m_clean, 0], Z_base[m_clean, 1], color=cmap[idx], s=38, alpha=0.9, marker="o", edgecolors="k", linewidths=0.3)
        axes[0].scatter(Z_base[m_eq, 0], Z_base[m_eq, 1], color=cmap[idx], s=38, alpha=0.75, marker="^", edgecolors="k", linewidths=0.3)
        axes[1].scatter(Z_cam[m_clean, 0], Z_cam[m_clean, 1], color=cmap[idx], s=42, alpha=0.9, marker="o", edgecolors="k", linewidths=0.3, label=f"Spk {spk}" if idx < 6 else None)
        axes[1].scatter(Z_cam[m_eq, 0], Z_cam[m_eq, 1], color=cmap[idx], s=42, alpha=0.8, marker="^", edgecolors="k", linewidths=0.3)

    for j, c_lbl in enumerate(clone_labels):
        axes[1].scatter(
            Z_clones[j, 0], Z_clones[j, 1],
            marker="*", s=200, color=cmap[c_lbl], edgecolors="black", linewidths=1.0,
            label="Pocket TTS Clone (*)" if j == 0 else None,
        )

    axes[0].set_title(f"Baseline RandomProj-128D t-SNE (Clean o / EQ-Tilt ^)\nSame={b_same} Diff={b_diff} | Gap={b_gap}", fontsize=9.8)
    axes[1].set_title(f"WeSpeaker CAM++-512D + Pocket TTS (*) (Clean o / EQ-Tilt ^)\nSame={c_same} Diff={c_diff} | Gap={c_gap}", fontsize=9.8)
    for ax in axes:
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=7.2, loc="best", frameon=True)
    plt.suptitle("Layer C Explainability: Channel-EQ Invariance & Speaker Clustering Across 12 Unseen Speakers (120 Clips)", fontsize=10.8)
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "fig3_speaker_embedding_pca_tsne.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path, {
        "random_proj_128d": {"same_cos": b_same, "diff_cos": b_diff, "separation_gap": b_gap},
        "wespeaker_campp_512d": {"same_cos": c_same, "diff_cos": c_diff, "separation_gap": c_gap},
    }


def generate_all_figures() -> Dict[str, Any]:
    p1 = plot_asr_word_confidence_heatmap()
    print("Generated:", p1, flush=True)
    p2, sal_summary = plot_campp_spectral_saliency()
    print("Generated:", p2, flush=True)
    p3, sil_summary = plot_speaker_embedding_projection()
    print("Generated:", p3, flush=True)

    summary = {
        "figures": {
            "asr_word_confidence_heatmap": os.path.relpath(p1, ROOT_DIR).replace("\\", "/"),
            "campp_spectral_saliency": os.path.relpath(p2, ROOT_DIR).replace("\\", "/"),
            "speaker_embedding_pca": os.path.relpath(p3, ROOT_DIR).replace("\\", "/"),
        },
        "campp_spectral_saliency": sal_summary,
        "speaker_cluster_silhouette": sil_summary,
    }
    out_json = os.path.join(ROOT_DIR, "ml", "results", "explainability_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved explainability summary ->", out_json, flush=True)
    return summary


if __name__ == "__main__":
    generate_all_figures()
