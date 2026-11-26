"""
Core Research Evaluation Primitives (ML Lab — Decoupled from Production API).
Provides ASR (WER, CER, Non-Entity WER, Over-Correction), SLU (Accuracy, Macro-F1, Schema Validity),
Speaker Similarity (EER, ROC-AUC, Fisher d', Cloning Specificity Margin), Audio Intelligibility (STOI),
Cost Pareto, TOST Non-Inferiority, and Bounded Paired Bootstrap significance helpers.
"""

from typing import Dict, Any, List, Optional, Sequence
import re
import numpy as np


_SPOKEN_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation except apostrophes/hyphens for fair text comparison."""
    cleaned = re.sub(r"[^\w\s'-]", " ", (text or "").lower())
    return " ".join(cleaned.split())


def normalize_text_itn(text: str) -> str:
    """Applies lightweight ITN-equivalent normalization (% -> percent, numeric hyphens, cardinal digits)."""
    s = (text or "").lower().replace("%", " percent ")
    s = re.sub(r"(?<=\d)-(?=\d)", " ", s)
    s = re.sub(r"[^\w\s'-]", " ", s)
    out = []
    for tok in s.split():
        if tok in _SPOKEN_NUMBERS:
            out.append(_SPOKEN_NUMBERS[tok])
        elif tok == "alright":
            out.extend(["all", "right"])
        else:
            out.append(tok)
    return " ".join(out)


def compute_itn_wer(references: Sequence[str], hypotheses: Sequence[str]) -> Dict[str, float]:
    """Computes ITN-normalized WER and CER alongside verbatim metrics to isolate formatting artifacts."""
    import jiwer
    n_refs = [normalize_text_itn(r) or "<empty>" for r in references]
    n_hyps = [normalize_text_itn(h) for h in hypotheses]
    return {
        "itn_wer": round(float(jiwer.wer(n_refs, n_hyps)), 4),
        "itn_cer": round(float(jiwer.cer(n_refs, n_hyps)), 4),
    }


def compute_asr_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    target_terms: Optional[Sequence[Sequence[str]]] = None,
    raw_hypotheses: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Computes corpus WER & CER via jiwer, optional target proper-noun recall,
    non-entity WER (fluff/fluent word accuracy), and repair precision / over-correction rate.
    """
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
        non_ent_refs: List[str] = []
        non_ent_hyps: List[str] = []

        for ref_n, hyp_n, terms in zip(norm_refs, norm_hyps, target_terms):
            hyp_tokens = set(hyp_n.split())
            ent_tok_set = set()
            for term in terms:
                t_norm = _normalize_text(term)
                if not t_norm:
                    continue
                total_terms += 1
                if t_norm in hyp_n or t_norm in hyp_tokens:
                    matched_terms += 1
                ent_tok_set.update(t_norm.split())

            rem_r = [w for w in ref_n.split() if w not in ent_tok_set]
            rem_h = [w for w in hyp_n.split() if w not in ent_tok_set]
            if rem_r:
                non_ent_refs.append(" ".join(rem_r))
                non_ent_hyps.append(" ".join(rem_h) if rem_h else "")

        out["target_term_recall"] = round(matched_terms / total_terms, 4) if total_terms else 1.0
        out["target_terms_total"] = total_terms
        out["target_terms_matched"] = matched_terms
        if non_ent_refs:
            out["non_entity_wer"] = round(float(jiwer.wer(non_ent_refs, non_ent_hyps)), 4)

    if raw_hypotheses is not None:
        if len(raw_hypotheses) != len(references):
            raise ValueError("raw_hypotheses length must match references length.")
        tokens_modified = 0
        true_fixes = 0
        false_repairs = 0
        for ref_n, raw_h, rep_h in zip(norm_refs, raw_hypotheses, norm_hyps):
            ref_toks = set(ref_n.split())
            raw_toks = (_normalize_text(raw_h)).split()
            rep_toks = rep_h.split()
            # Track multi-set difference conservatively when lengths align or via word set diffs
            if len(raw_toks) == len(rep_toks):
                for rw, pw in zip(raw_toks, rep_toks):
                    if rw != pw:
                        tokens_modified += 1
                        if pw in ref_toks and rw not in ref_toks:
                            true_fixes += 1
                        elif rw in ref_toks and pw not in ref_toks:
                            false_repairs += 1
            else:
                raw_set, rep_set = set(raw_toks), set(rep_toks)
                added = rep_set - raw_set
                removed = raw_set - rep_set
                tokens_modified += max(len(added), len(removed))
                true_fixes += len(added & ref_toks)
                false_repairs += len(removed & ref_toks)

        out["tokens_modified"] = tokens_modified
        out["true_repairs"] = true_fixes
        out["false_repairs"] = false_repairs
        out["repair_precision"] = round(true_fixes / tokens_modified, 4) if tokens_modified else 1.0
        out["overcorrection_rate"] = round(false_repairs / max(sum(len(_normalize_text(h).split()) for h in raw_hypotheses), 1), 4)

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


def compute_verification_stats(same_scores: Sequence[float], diff_scores: Sequence[float]) -> Dict[str, float]:
    """
    Computes encoder-comparable speaker verification statistics:
    ROC-AUC, EER, Fisher Discriminability d', and raw cosine mean/std distributions.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    s_arr = np.asarray(same_scores, dtype=np.float64)
    d_arr = np.asarray(diff_scores, dtype=np.float64)
    y_true = np.concatenate([np.ones(len(s_arr)), np.zeros(len(d_arr))])
    y_scores = np.concatenate([s_arr, d_arr])

    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    s_mean = float(np.mean(s_arr))
    d_mean = float(np.mean(d_arr))
    s_var = float(np.var(s_arr, ddof=1)) if len(s_arr) > 1 else 0.0
    d_var = float(np.var(d_arr, ddof=1)) if len(d_arr) > 1 else 0.0
    pooled_std = float(np.sqrt(0.5 * (s_var + d_var) + 1e-12))
    d_prime = (s_mean - d_mean) / pooled_std

    return {
        "same_mean": round(s_mean, 4),
        "same_std": round(float(np.sqrt(s_var)), 4),
        "diff_mean": round(d_mean, 4),
        "diff_std": round(float(np.sqrt(d_var)), 4),
        "cosine_gap": round(s_mean - d_mean, 4),
        "d_prime": round(d_prime, 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "num_same": int(len(s_arr)),
        "num_diff": int(len(d_arr)),
    }


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
    n_resamples: int = 2000,
    seed: int = 42,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Runs a seeded paired bootstrap significance test over per-unit metric arrays.
    Uses the Phipson & Smyth (2010) bounded estimator p = (count + 1) / (B + 1)
    so p_value is strictly bounded > 0 (e.g. <= 0.001 at B=2000 when count=0).
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
    if observed_delta >= 0:
        count = int(np.sum(boot_means <= 0.0))
    else:
        count = int(np.sum(boot_means >= 0.0))

    p_one_sided = (count + 1.0) / (float(n_resamples) + 1.0)
    p_value = min(1.0, 2.0 * p_one_sided)

    return {
        "observed_improvement": round(observed_delta, 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
        "p_value": round(p_value, 4),
        "p_below_001": bool(p_value < 0.001),
        "n_resamples": int(n_resamples),
        "n_units": int(n),
    }


def tost_non_inferiority(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    margin: float = -0.05,
    n_resamples: int = 2000,
    seed: int = 42,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Bootstrap Two One-Sided Tests (TOST) non-inferiority & equivalence check against
    margin Delta (default -0.05 on [0, 1] scale). Candidate is declared non-inferior
    only if the 90% CI lower bound strictly exceeds `margin` (p_non_inferiority < 0.05).
    """
    base = np.asarray(baseline_scores, dtype=np.float64)
    cand = np.asarray(candidate_scores, dtype=np.float64)
    if base.shape != cand.shape or base.size < 2:
        raise ValueError("Paired score arrays must have identical length >= 2.")

    diffs = (cand - base) if higher_is_better else (base - cand)
    rng = np.random.RandomState(seed)
    n = diffs.size
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])

    ci_90_low, ci_90_high = np.percentile(boot_means, [5.0, 95.0])
    ci_95_low, ci_95_high = np.percentile(boot_means, [2.5, 97.5])
    abs_margin = abs(float(margin))
    low_bound = -abs_margin
    high_bound = abs_margin

    p_lower = (int(np.sum(boot_means <= low_bound)) + 1.0) / (float(n_resamples) + 1.0)
    p_upper = (int(np.sum(boot_means >= high_bound)) + 1.0) / (float(n_resamples) + 1.0)
    p_tost = max(p_lower, p_upper)

    return {
        "observed_delta": round(float(np.mean(diffs)), 4),
        "margin": round(low_bound, 4),
        "ci_90_low": round(float(ci_90_low), 4),
        "ci_90_high": round(float(ci_90_high), 4),
        "ci_95_low": round(float(ci_95_low), 4),
        "ci_95_high": round(float(ci_95_high), 4),
        "p_non_inferiority": round(p_lower, 4),
        "p_tost_equivalence": round(p_tost, 4),
        "non_inferior": bool(ci_90_low > low_bound and p_lower < 0.05),
        "equivalent": bool(ci_90_low > low_bound and ci_90_high < high_bound and p_tost < 0.05),
        "n_units": int(n),
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
    """
    Estimates marginal dollar cost per 1,000 voice pipeline turns:
      C_1k = ((L_sec * 1000) / 3600) * R_hr + C_api
    where R_hr = compute_hourly_usd ($0.05/hr for <=4 GB CPU/edge VM, $0.15/hr for >4 GB GPU/8B offload).
    """
    compute_cost = (max(0.0, avg_latency_sec) * 1000.0 / 3600.0) * compute_hourly_usd
    return round(compute_cost + cloud_api_usd_per_1k_turns, 4)
