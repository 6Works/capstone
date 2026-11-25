"""
Core Research Evaluation Primitives (ML Lab — Decoupled from Production API).
Provides ASR, SLU, Speaker Similarity, Audio Intelligibility, Cost Pareto,
and Paired Bootstrap significance helpers for reproducible ablation studies.
"""

from typing import Dict, Any, List, Optional, Sequence
import re
import numpy as np


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation except apostrophes/hyphens for fair text comparison."""
    cleaned = re.sub(r"[^\w\s'-]", " ", (text or "").lower())
    return " ".join(cleaned.split())


def compute_asr_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    target_terms: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Computes corpus WER, CER via jiwer, plus optional target proper-noun recall."""
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty sequences of equal length.")

    import jiwer

    norm_refs = [_normalize_text(r) or "<empty>" for r in references]
    norm_hyps = [_normalize_text(h) for h in hypotheses]

    wer_val = float(jiwer.wer(norm_refs, norm_hyps))
    cer_val = float(jiwer.cer(norm_refs, norm_hyps))

    out: Dict[str, Any] = {
        "wer": round(wer_val, 4),
        "cer": round(cer_val, 4),
        "num_samples": len(references),
    }

    if target_terms is not None:
        if len(target_terms) != len(references):
            raise ValueError("target_terms length must match references length.")
        total_terms = 0
        matched_terms = 0
        for hyp_norm, terms in zip(norm_hyps, target_terms):
            hyp_tokens = set(hyp_norm.split())
            for term in terms:
                t_norm = _normalize_text(term)
                if not t_norm:
                    continue
                total_terms += 1
                if t_norm in hyp_norm or t_norm in hyp_tokens:
                    matched_terms += 1
        out["target_term_recall"] = round(matched_terms / total_terms, 4) if total_terms else 1.0
        out["target_terms_total"] = total_terms
        out["target_terms_matched"] = matched_terms

    return out


def compute_slu_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    schema_valid_flags: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Computes classification Accuracy, Macro-F1, Weighted-F1, and JSON schema compliance."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty sequences of equal length.")

    from sklearn.metrics import accuracy_score, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    out: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "num_samples": len(y_true),
    }

    if schema_valid_flags is not None:
        if len(schema_valid_flags) != len(y_true):
            raise ValueError("schema_valid_flags length must match y_true length.")
        out["schema_valid_rate"] = round(sum(1 for x in schema_valid_flags if x) / len(schema_valid_flags), 4)

    return out


def compute_speaker_similarity(emb_a: Sequence[float], emb_b: Sequence[float]) -> float:
    """Computes cosine similarity in [-1.0, 1.0] between two speaker embedding vectors."""
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"Embedding shape mismatch: {a.shape} vs {b.shape}")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return round(float(np.dot(a, b) / denom), 4)


def compute_audio_intelligibility(clean_wav_path: str, degraded_wav_path: str, target_sr: int = 16000) -> float:
    """Computes Short-Time Objective Intelligibility (STOI) in [0, 1] for two WAV files."""
    import soundfile as sf
    import scipy.signal as signal
    from pystoi import stoi

    clean, sr_c = sf.read(clean_wav_path, dtype="float32")
    deg, sr_d = sf.read(degraded_wav_path, dtype="float32")

    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if deg.ndim > 1:
        deg = deg.mean(axis=1)

    if sr_c != target_sr:
        clean = signal.resample(clean, int(len(clean) * target_sr / sr_c))
    if sr_d != target_sr:
        deg = signal.resample(deg, int(len(deg) * target_sr / sr_d))

    min_len = min(len(clean), len(deg))
    if min_len < int(0.3 * target_sr):
        raise ValueError("Audio too short for reliable STOI evaluation (<0.3s).")

    score = float(stoi(clean[:min_len], deg[:min_len], target_sr, extended=False))
    return round(score, 4)


def paired_bootstrap_test(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """
    Runs a seeded paired bootstrap significance test over per-sample metric arrays.
    Reports mean improvement delta, 95% confidence interval, and two-sided p-value.
    """
    base = np.asarray(baseline_scores, dtype=np.float64)
    cand = np.asarray(candidate_scores, dtype=np.float64)
    if base.shape != cand.shape or base.size < 2:
        raise ValueError("Paired score arrays must have identical length >= 2.")

    diffs = (cand - base) if higher_is_better else (base - cand)
    observed_delta = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    n = diffs.size
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    # Two-sided p-value: fraction of bootstrap resamples crossing or opposing 0
    if observed_delta >= 0:
        p_one_sided = float(np.mean(boot_means <= 0.0))
    else:
        p_one_sided = float(np.mean(boot_means >= 0.0))
    p_value = min(1.0, 2.0 * p_one_sided)

    return {
        "observed_improvement": round(observed_delta, 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "p_value": round(p_value, 4),
    }


def summarize_seeds(values: Sequence[float]) -> Dict[str, float]:
    """Returns mean, sample std (ddof=1), min, max across >=1 random seeds."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty.")
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(std_val, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "n_seeds": int(arr.size),
    }


def estimate_cost_per_1k_turns(
    avg_latency_sec: float,
    compute_hourly_usd: float = 0.05,
    cloud_api_usd_per_1k_turns: float = 0.0,
) -> float:
    """Estimates marginal dollar cost per 1,000 voice pipeline turns for Pareto plots."""
    compute_cost = (max(0.0, avg_latency_sec) * 1000.0 / 3600.0) * compute_hourly_usd
    return round(compute_cost + cloud_api_usd_per_1k_turns, 4)
