"""
Session 6 Experiment (Upgraded): Layer B (Stage 3/4 SLU, Cross-Layer Conditioning & Interview Rubric).
Addresses all methodological & licensing points:
  - Evaluates Apache-2.0 Qwen3-4B & DistilBERT-LoRA, MIT Phi-4-mini, Llama 3.2 3B (calibrated prompt),
    Qwen 2.5 Coder 1.5B, legacy Qwen 2.5 3B (qwen-research documented), and 8B Llama 3.1 ceiling.
  - Compares against fair Ollama GBNF JSON-Schema Constrained Decoding baselines (not just raw unguided text).
  - Tunes OOS probability threshold strictly on VAL via Ollama token logprobs.
  - Tests Cross-Layer ASR Word-Confidence Conditioning (parse_with_asr_context) & G2P pre-repair.
  - Reports TOST Non-Inferiority (Delta = -0.05) + Bounded Paired Bootstrap (N=2000, Phipson-Smyth).
  - Evaluates FULL MINDS-14 (498 real human clips) + FULL CLINC-OOS (5,500 items) on deterministic/neural models
    plus stratified N=56 MINDS-14 & N=150 CLINC-OOS across generative SLMs (--skip-full optional).
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.llm.gateway import llm_gateway
from services.nlp import RuleBasedNLPService, LLMNLPService, TrainedDistilBertSLUService, HybridConfidenceSLUService
from services.roles.role_service import RoleService
from services.context import context_manager
from ml.datasets import load_manifest
from ml.metrics import (
    compute_slu_metrics,
    paired_bootstrap_test,
    tost_non_inferiority,
    summarize_seeds,
    estimate_cost_per_1k_turns,
)
from ml.experiments.run_asr_ablation import (
    INTERVIEW_ROLE_TERMS,
    CANDIDATE_ROSTER_NAMES,
    MINDS14_BANKING_TERMS,
)


CONFIG_NAMES = [
    "slu_rule_based",
    "slu_distilbert_lora",
    "slu_hybrid_cascade",
    "slu_qwen15b_raw",
    "slu_qwen15b_schema",
    "slu_qwen15b_repair",
    "slu_llama3b_raw",
    "slu_llama3b_schema",
    "slu_llama3b_calibrated_repair",
    "slu_qwen3_4b_schema",
    "slu_qwen3_4b_repair",
    "slu_phi4_mini_schema",
    "slu_phi4_mini_repair",
    "slu_qwen3b_repair",
    "slu_llama8b_ref",
]

INTERVIEW_INTENTS = [
    "greeting",
    "identity_query",
    "project_query",
    "calculation_query",
    "time_query",
    "answer",
    "farewell",
]

TIER_TO_ORDINAL = {"low": 0.0, "mid": 1.0, "high": 2.0}

CLINC_15_CLASSES = sorted([
    "oos",
    "change_language",
    "definition",
    "find_phone",
    "fun_fact",
    "improve_credit_score",
    "insurance_change",
    "meaning_of_life",
    "payday",
    "pto_request",
    "replacement_card_duration",
    "timer",
    "transfer",
    "translate",
    "travel_alert",
])


def _ollama_model_info(model_name: str) -> Dict[str, float]:
    """Reads disk weight (GB) and live VRAM allocation (GB) from local Ollama daemon."""
    disk_gb, vram_gb = 0.0, 0.0
    try:
        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=3).json()
        for m in tags.get("models", []):
            if m.get("name") == model_name or m.get("name", "").split(":")[0] == model_name.split(":")[0]:
                disk_gb = round(float(m.get("size", 0)) / (1024 ** 3), 2)
                break
        ps = requests.get("http://127.0.0.1:11434/api/ps", timeout=3).json()
        for m in ps.get("models", []):
            if m.get("name") == model_name or m.get("name", "").split(":")[0] == model_name.split(":")[0]:
                vram_gb = round(float(m.get("size_vram", 0)) / (1024 ** 3), 2)
                break
    except Exception:
        pass
    return {"disk_gb": disk_gb, "vram_gb": vram_gb}


def _load_benchmarks() -> Dict[str, Any]:
    """Loads stratified + full benchmarks alongside Session 5 real ASR hypotheses & word confidences."""
    m14_full_test = [x for x in load_manifest("minds14_en_full")["items"] if x["split"] == "test"]
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for it in m14_full_test:
        by_intent.setdefault(it["intent_label"], []).append(it)
    m14_stratified: List[Dict[str, Any]] = []
    for label in sorted(by_intent.keys()):
        m14_stratified.extend(by_intent[label][:4])  # 4 x 14 = 56 balanced clips

    # Load Session 5 ASR transcripts & optional cached Whisper word confidences
    asr_path = os.path.join(ROOT_DIR, "ml", "results", "asr_ablation.json")
    full_tr_path = os.path.join(ROOT_DIR, "ml", "results", "minds14_full_asr_transcripts.json")
    tiny_cache_path = os.path.join(ROOT_DIR, "storage", "benchmarks", "minds14_full_tiny_asr_cache.json")

    asr_maps: Dict[str, Dict[str, str]] = {"tiny": {}, "small": {}, "parakeet": {}}
    iv_maps: Dict[str, Dict[str, str]] = {"tiny": {}, "small": {}, "parakeet": {}}
    if os.path.exists(asr_path):
        asr_data = json.load(open(asr_path, "r", encoding="utf-8"))
        for key, target_k in [("asr_tiny_baseline", "tiny"), ("asr_small_improved", "small"), ("asr_parakeet_tdt", "parakeet")]:
            cfg_e = asr_data["configs"].get(key) or {}
            for r in (cfg_e.get("minds14_clean") or {}).get("transcripts") or []:
                asr_maps[target_k][r["id"]] = r["hypothesis"]
            for r in (cfg_e.get("interview_stress_clean") or {}).get("transcripts") or []:
                iv_maps[target_k][r["id"]] = r["hypothesis"]

    if os.path.exists(full_tr_path):
        tr_store = json.load(open(full_tr_path, "r", encoding="utf-8"))
        for key, target_k in [("asr_tiny_baseline", "tiny"), ("asr_small_improved", "small"), ("asr_parakeet_tdt", "parakeet")]:
            for r in tr_store.get(key) or []:
                asr_maps[target_k][r["id"]] = r["hypothesis"]

    tiny_words_cache = json.load(open(tiny_cache_path, "r", encoding="utf-8")) if os.path.exists(tiny_cache_path) else {}

    for it in m14_full_test:
        it["asr_tiny_baseline"] = asr_maps["tiny"].get(it["id"], tiny_words_cache.get(it["id"], {}).get("hyp", it["text"]))
        it["asr_small_improved"] = asr_maps["small"].get(it["id"], it["text"])
        it["asr_parakeet_tdt"] = asr_maps["parakeet"].get(it["id"], it["text"])
        it["tiny_words_meta"] = tiny_words_cache.get(it["id"], {}).get("words", [])
    for it in m14_stratified:
        it["asr_tiny_baseline"] = asr_maps["tiny"].get(it["id"], tiny_words_cache.get(it["id"], {}).get("hyp", it["text"]))
        it["asr_small_improved"] = asr_maps["small"].get(it["id"], it["text"])
        it["asr_parakeet_tdt"] = asr_maps["parakeet"].get(it["id"], it["text"])
        it["tiny_words_meta"] = tiny_words_cache.get(it["id"], {}).get("words", [])

    # CLINC-OOS: 15-class stratified val (45) & test (150) PLUS full 151-class test (5,500)
    clinc_all = load_manifest("clinc_oos_full")["items"]
    clinc_val_15, clinc_test_15, clinc_full_test = [], [], []
    by_cl_val: Dict[str, List[Dict[str, Any]]] = {}
    by_cl_test: Dict[str, List[Dict[str, Any]]] = {}
    for it in clinc_all:
        if it["split"] == "val" and it["intent_label"] in CLINC_15_CLASSES:
            by_cl_val.setdefault(it["intent_label"], []).append(it)
        elif it["split"] == "test":
            clinc_full_test.append(it)
            if it["intent_label"] in CLINC_15_CLASSES:
                by_cl_test.setdefault(it["intent_label"], []).append(it)

    for lbl in CLINC_15_CLASSES:
        lim_v = 10 if lbl == "oos" else 3
        lim_t = 30 if lbl == "oos" else 9  # 30 + 14*9 = 156 items (if definition has >=9)
        clinc_val_15.extend(by_cl_val.get(lbl, [])[:lim_v])
        clinc_test_15.extend(by_cl_test.get(lbl, [])[:lim_t])

    # Interview QA (30 test turns)
    iv_qa = [
        x for x in load_manifest("interview_stress_v1")["items"]
        if x["split"] == "test" and x.get("subset") == "interview_qa"
    ]
    for it in iv_qa:
        it["asr_tiny_baseline"] = iv_maps["tiny"].get(it["id"], it["text"])
        it["asr_small_improved"] = iv_maps["small"].get(it["id"], it["text"])
        it["asr_parakeet_tdt"] = iv_maps["parakeet"].get(it["id"], it["text"])
    iv_answers = [x for x in iv_qa if x.get("intent_label") == "answer"]

    return {
        "minds14_stratified": m14_stratified,
        "minds14_full_test": m14_full_test,
        "minds14_labels": sorted(by_intent.keys()),
        "clinc_val_15": clinc_val_15,
        "clinc_test_15": clinc_test_15,
        "clinc_15_labels": CLINC_15_CLASSES,
        "clinc_full_test": clinc_full_test,
        "clinc_full_labels": sorted({x["intent_label"] for x in clinc_full_test}),
        "interview_qa": iv_qa,
        "interview_answers": iv_answers,
    }


def _calibrate_slm_oos_threshold(
    cfg: Dict[str, Any],
    val_items: Sequence[Dict[str, Any]],
    clinc_labels: Sequence[str],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Tunes logprob seq_prob abstention threshold strictly on CLINC VAL slice."""
    if not cfg.get("use_logprobs") or cfg["provider"] != "ollama":
        return None, {}
    svc = LLMNLPService(
        model_name=cfg["model_name"],
        allowed_intents=clinc_labels,
        enable_schema_repair=bool(cfg.get("enable_schema_repair", True)),
        enable_repair_retry=bool(cfg.get("enable_repair_retry", True)),
        use_json_mode=bool(cfg.get("use_json_mode", True)),
        use_schema_mode=bool(cfg.get("use_schema_mode", True)),
        use_logprobs=True,
        calibrated_prompt=bool(cfg.get("calibrated_prompt", True)),
        temperature=0.0,
        seed=42,
    )
    y_true = [it["intent_label"] for it in val_items]
    raw_preds, seq_probs = [], []
    for it in val_items:
        res = svc.parse(it["text"])
        raw_preds.append((res.intent or "").strip().lower().replace(" ", "_"))
        sp = (res.metadata or {}).get("seq_prob")
        seq_probs.append(float(sp) if sp is not None else 1.0)

    best_tau = 0.0
    best_f1 = float(f1_score(y_true, raw_preds, average="macro", zero_division=0))
    for tau in [0.0, 0.35, 0.45, 0.55, 0.65, 0.75]:
        preds = ["oos" if (p < tau and tau > 0.0) else rp for rp, p in zip(raw_preds, seq_probs)]
        sc = float(f1_score(y_true, preds, average="macro", zero_division=0))
        if sc > best_f1:
            best_f1 = sc
            best_tau = tau
    return (best_tau if best_tau > 0.0 else None), {
        "best_oos_threshold": best_tau,
        "val_macro_f1": round(best_f1, 4),
        "val_samples": len(val_items),
    }


def _make_nlp_service(
    cfg: Dict[str, Any],
    allowed_intents: Optional[Sequence[str]],
    repair_lexicon: Optional[Sequence[str]],
    seed: int,
    oos_tau: Optional[float] = None,
):
    prov = cfg["provider"]
    if prov == "rule_based":
        return RuleBasedNLPService(repair_lexicon=repair_lexicon if cfg.get("asr_pre_repair") else None)
    if prov == "distilbert_lora":
        return TrainedDistilBertSLUService(
            model_dir=cfg["model_name"],
            allowed_intents=allowed_intents,
            repair_lexicon=repair_lexicon if cfg.get("asr_pre_repair") else None,
            oos_prob_threshold=oos_tau,
            use_conf_tags=True,
        )
    if prov == "hybrid_cascade":
        fast_svc = TrainedDistilBertSLUService(
            model_dir=cfg["model_name"],
            allowed_intents=allowed_intents,
            repair_lexicon=repair_lexicon if cfg.get("asr_pre_repair") else None,
            oos_prob_threshold=oos_tau,
            use_conf_tags=True,
        )
        slow_svc = LLMNLPService(
            model_name=cfg.get("slow_model_name", "qwen3:4b"),
            allowed_intents=allowed_intents,
            repair_lexicon=repair_lexicon if cfg.get("asr_pre_repair") else None,
            enable_schema_repair=bool(cfg.get("enable_schema_repair", True)),
            enable_repair_retry=bool(cfg.get("enable_repair_retry", True)),
            use_json_mode=bool(cfg.get("use_json_mode", True)),
            use_schema_mode=bool(cfg.get("use_schema_mode", True)),
            use_logprobs=bool(cfg.get("use_logprobs", True)),
            oos_prob_threshold=oos_tau,
            calibrated_prompt=bool(cfg.get("calibrated_prompt", True)),
            temperature=float(cfg.get("temperature", 0.1)),
            seed=seed,
        )
        return HybridConfidenceSLUService(
            fast_service=fast_svc,
            slow_service=slow_svc,
            confidence_threshold=float(cfg.get("confidence_threshold", 0.85)),
            margin_threshold=float(cfg.get("margin_threshold", 0.25)),
            escalate_on_oos=bool(cfg.get("escalate_on_oos", True)),
        )
    return LLMNLPService(
        model_name=cfg["model_name"],
        allowed_intents=allowed_intents,
        repair_lexicon=repair_lexicon if cfg.get("asr_pre_repair") else None,
        enable_schema_repair=bool(cfg.get("enable_schema_repair", True)),
        enable_repair_retry=bool(cfg.get("enable_repair_retry", True)),
        use_json_mode=bool(cfg.get("use_json_mode", True)),
        use_schema_mode=bool(cfg.get("use_schema_mode", False)),
        use_logprobs=bool(cfg.get("use_logprobs", False)),
        oos_prob_threshold=oos_tau,
        calibrated_prompt=bool(cfg.get("calibrated_prompt", False)),
        temperature=float(cfg.get("temperature", 0.1)),
        seed=seed,
    )


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _eval_slu_pass(
    nlp_svc: Any,
    items: Sequence[Dict[str, Any]],
    text_key: str,
    label_set: Sequence[str],
    use_words_meta: bool = False,
    ckpt_path: Optional[str] = None,
    resume: bool = True,
    progress_tag: str = "",
) -> Dict[str, Any]:
    """Runs SLU pass with batch-level progress and atomic item-level checkpointing/resumability."""
    cached: Dict[str, Dict[str, Any]] = {}
    if resume and ckpt_path and os.path.exists(ckpt_path):
        try:
            cached = json.load(open(ckpt_path, "r", encoding="utf-8"))
        except Exception:
            cached = {}

    y_true, y_pred = [], []
    raw_valid_flags, schema_valid_flags = [], []
    latencies_ms, correct_flags = [], []
    repaired_count, retried_count, asr_fix_count, cascade_count = 0, 0, 0, 0
    valid_labels = {l.lower(): l for l in label_set}
    total = len(items)
    t_pass = time.time()

    for idx, it in enumerate(items, 1):
        uid = f"{it['id']}_{idx}"
        if uid in cached:
            rec = cached[uid]
        else:
            txt = it[text_key]
            wmeta = it.get("tiny_words_meta") if use_words_meta else None
            t0 = time.perf_counter()
            if wmeta and hasattr(nlp_svc, "parse_with_asr_context"):
                res = nlp_svc.parse_with_asr_context(txt, words_meta=wmeta)
            elif wmeta and isinstance(nlp_svc, TrainedDistilBertSLUService):
                res = nlp_svc.parse(txt, words_meta=wmeta)
            else:
                res = nlp_svc.parse(txt)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            gold = it["intent_label"]
            raw_intent = (res.intent or "").strip().lower().replace(" ", "_")
            pred = valid_labels.get(raw_intent, raw_intent or "__invalid__")
            if "rule_based_fallback" in (res.provider or "").lower() and raw_intent not in valid_labels:
                pred = "__fallback__"

            meta = res.metadata or {}
            rec = {
                "gold": gold,
                "pred": pred,
                "lat_ms": round(lat_ms, 2),
                "raw_valid": bool(meta.get("raw_valid", True)),
                "schema_valid": bool(meta.get("schema_valid", "fallback" not in (res.provider or "").lower())),
                "repaired": bool(meta.get("repaired")),
                "retried": bool(meta.get("retried")),
                "asr_fixes": len(meta.get("asr_repairs") or []),
                "cascaded": bool(meta.get("cascaded_to_slow", False)),
            }
            cached[uid] = rec

        y_true.append(rec["gold"])
        y_pred.append(rec["pred"])
        correct_flags.append(1.0 if rec["pred"] == rec["gold"] else 0.0)
        latencies_ms.append(float(rec["lat_ms"]))
        raw_valid_flags.append(bool(rec["raw_valid"]))
        schema_valid_flags.append(bool(rec["schema_valid"]))
        if rec["repaired"]:
            repaired_count += 1
        if rec["retried"]:
            retried_count += 1
        asr_fix_count += int(rec["asr_fixes"])
        if rec.get("cascaded"):
            cascade_count += 1

        if ckpt_path and (idx % 25 == 0 or idx == total):
            _atomic_json_dump(cached, ckpt_path)
        if total >= 50 and (idx % 25 == 0 or idx == total):
            tag_lbl = progress_tag or text_key
            print(f"    [{tag_lbl}] batch {idx}/{total} ({time.time() - t_pass:.1f}s)", flush=True)

    metrics = compute_slu_metrics(y_true, y_pred, schema_valid_flags=schema_valid_flags)
    metrics["raw_valid_rate"] = round(sum(1 for x in raw_valid_flags if x) / len(raw_valid_flags), 4)
    metrics["avg_latency_ms"] = round(float(np.mean(latencies_ms)), 1)
    metrics["p95_latency_ms"] = round(float(np.percentile(latencies_ms, 95)), 1)
    metrics["repaired_turns"] = repaired_count
    metrics["retried_turns"] = retried_count
    metrics["asr_pre_repairs"] = asr_fix_count
    metrics["cascade_rate"] = round(cascade_count / len(items), 4)
    metrics["per_item_correct"] = correct_flags
    if "oos" in label_set:
        metrics["oos_f1"] = round(float(f1_score([y == "oos" for y in y_true], [p == "oos" for p in y_pred], zero_division=0)), 4)
    return metrics


def _eval_rubric_pass(
    cfg: Dict[str, Any],
    answer_items: Sequence[Dict[str, Any]],
    seed: int,
    ckpt_path: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Evaluates RoleService rubric alignment with item-level checkpointing across interview QA answers."""
    role = RoleService()
    cached: Dict[str, Dict[str, Any]] = {}
    if resume and ckpt_path and os.path.exists(ckpt_path):
        try:
            cached = json.load(open(ckpt_path, "r", encoding="utf-8"))
        except Exception:
            cached = {}

    scores, tier_ordinals, mid_targets = [], [], []
    raw_flags, schema_flags, latencies_ms = [], [], []
    target_model = cfg.get("slow_model_name", cfg["model_name"])
    use_cal_rubric = bool(cfg.get("calibrated_rubric", True))

    for idx, it in enumerate(answer_items, 1):
        uid = f"{it['id']}_{idx}"
        if uid in cached:
            rec = cached[uid]
        else:
            context_manager.sessions.clear()
            role.current_interview_question_idx = 2
            t0 = time.perf_counter()
            if cfg["provider"] == "rule_based":
                prev_fb = llm_gateway.active_provider_id
                llm_gateway.active_provider_id = "rule_based"
                try:
                    out = role.process_interviewer_turn(it["text"], mode="solo", calibrated_rubric=use_cal_rubric)
                finally:
                    llm_gateway.active_provider_id = prev_fb
            else:
                out = role.process_interviewer_turn(
                    it["text"],
                    mode="solo",
                    model_name=target_model,
                    use_json_mode=bool(cfg.get("use_json_mode", True)),
                    use_schema_mode=bool(cfg.get("use_schema_mode", False)),
                    enable_schema_repair=bool(cfg.get("enable_schema_repair", True)),
                    enable_repair_retry=bool(cfg.get("enable_repair_retry", True)),
                    seed=seed,
                    temperature=float(cfg.get("temperature", 0.1)),
                    calibrated_rubric=use_cal_rubric,
                )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            diag = out.get("schema_diagnostics") or {}
            rec = {
                "score": float(out["evaluation"]["overall_score"]),
                "raw_valid": bool(diag.get("raw_valid", True)),
                "schema_valid": bool(diag.get("schema_valid", True)),
                "lat_ms": round(lat_ms, 1),
            }
            cached[uid] = rec
            if ckpt_path:
                _atomic_json_dump(cached, ckpt_path)

        scores.append(float(rec["score"]))
        latencies_ms.append(float(rec["lat_ms"]))
        tier_ordinals.append(TIER_TO_ORDINAL[it["rubric_tier"]])
        lo, hi = it["expected_score_range"]
        mid_targets.append((lo + hi) / 2.0)
        raw_flags.append(bool(rec["raw_valid"]))
        schema_flags.append(bool(rec["schema_valid"]))

    rho_tier, p_tier = spearmanr(scores, tier_ordinals)
    mae_mid = float(np.mean(np.abs(np.asarray(scores) - np.asarray(mid_targets))))
    return {
        "scores": scores,
        "spearman_vs_tier": round(float(rho_tier) if np.isfinite(rho_tier) else 0.0, 4),
        "spearman_p_value": round(float(p_tier) if np.isfinite(p_tier) else 1.0, 4),
        "mae_vs_expected_midpoint": round(mae_mid, 3),
        "raw_valid_rate": round(sum(1 for x in raw_flags if x) / len(raw_flags), 4),
        "schema_valid_rate": round(sum(1 for x in schema_flags if x) / len(schema_flags), 4),
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 1),
    }


def _aggregate_seed_runs(runs: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    return {k: summarize_seeds([r[k] for r in runs if k in r]) for k in keys if any(k in r for r in runs)}


def run_ablation(
    only_configs: Optional[Sequence[str]] = None,
    run_full_on_top: bool = True,
    resume: bool = True,
    rubric_only: bool = False,
) -> Dict[str, Any]:
    bench = _load_benchmarks()
    out_path = os.path.join(ROOT_DIR, "ml", "results", "slu_ablation.json")
    ckpt_dir = os.path.join(ROOT_DIR, "ml", "results", "checkpoints", "slu")
    os.makedirs(ckpt_dir, exist_ok=True)
    existing: Dict[str, Any] = {}
    if os.path.exists(out_path) and (only_configs or resume):
        try:
            loaded = json.load(open(out_path, "r", encoding="utf-8"))
            if loaded.get("experiment") == "layer_b_slu_rubric_ablation_v2" or only_configs:
                existing = loaded
        except Exception:
            existing = {}

    results: Dict[str, Any] = {
        "experiment": "layer_b_slu_rubric_ablation_v2",
        "configs": dict(existing.get("configs", {})),
        "failed_configs": dict(existing.get("failed_configs", {})) if resume else {},
        "paired_bootstrap": {},
        "tost_non_inferiority": {},
        "sample_counts": {
            "minds14_stratified": len(bench["minds14_stratified"]),
            "minds14_full_test": len(bench["minds14_full_test"]),
            "clinc_test_15": len(bench["clinc_test_15"]),
            "clinc_full_test": len(bench["clinc_full_test"]),
            "interview_qa_test": len(bench["interview_qa"]),
            "interview_rubric_answers": len(bench["interview_answers"]),
        },
    }

    llm_gateway.configure(provider_id="ollama", fallback_enabled=False)
    targets = list(only_configs) if only_configs else list(CONFIG_NAMES)
    per_item_correct_seed42: Dict[str, Dict[str, List[float]]] = {}
    full_eval_slms = {"slu_qwen3_4b_repair", "slu_phi4_mini_repair", "slu_qwen15b_repair"}

    for cname in targets:
        if resume and not only_configs and cname in results["configs"]:
            print(f"\n=== [{cname}] Resumed from completed v2 checkpoint ===", flush=True)
            continue

        cfg_path = os.path.join(ROOT_DIR, "ml", "configs", f"{cname}.json")
        try:
            cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
            seeds = cfg.get("seeds", [42, 1337, 2026])
            if cfg["provider"] in ("rule_based", "distilbert_lora"):
                seeds = [42]

            if cfg["provider"] == "distilbert_lora" and not os.path.exists(os.path.join(ROOT_DIR, cfg["model_name"], "label_map.json")):
                raise FileNotFoundError(f"Trained LoRA adapter not present yet at {cfg['model_name']} (awaiting Kaggle download)")

            if rubric_only:
                rub_runs = []
                for s in seeds:
                    rb_ckpt = os.path.join(ckpt_dir, f"{cname}_rubric_cal_s{s}.json")
                    rr = _eval_rubric_pass(cfg, bench["interview_answers"], s, ckpt_path=rb_ckpt, resume=resume)
                    rub_runs.append(rr)
                    print(f"  [RUBRIC CAL s={s}] rho={rr['spearman_vs_tier']} MAE={rr['mae_vs_expected_midpoint']} scores={rr['scores']}", flush=True)
                if cname in results["configs"]:
                    results["configs"][cname]["interview_rubric"] = {
                        "summary": _aggregate_seed_runs(rub_runs, ["spearman_vs_tier", "mae_vs_expected_midpoint", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                        "seed_42_scores": rub_runs[0]["scores"],
                    }
                    _atomic_json_dump(results, out_path)
                continue

            print(f"\n=== Running {cname} ({cfg['provider']}:{cfg['model_name']}, schema={cfg.get('use_schema_mode', False)}, lp={cfg.get('use_logprobs', False)}) ===", flush=True)
            if cfg["provider"] == "ollama":
                _make_nlp_service(cfg, bench["minds14_labels"], None, 42).parse("check my balance")
            model_info = _ollama_model_info(cfg.get("slow_model_name", cfg["model_name"])) if cfg["provider"] in ("ollama", "hybrid_cascade") else {"disk_gb": 0.25 if cfg["provider"] == "distilbert_lora" else 0.0, "vram_gb": 0.0}

            oos_tau, val_cal_info = _calibrate_slm_oos_threshold(cfg, bench["clinc_val_15"], bench["clinc_15_labels"])
            if val_cal_info:
                print(f"  [VAL OOS calibration] tau={oos_tau} -> val_F1={val_cal_info['val_macro_f1']}", flush=True)

            m14_gold_runs, clinc_runs, iv_slu_runs, rubric_runs = [], [], [], []
            for s in seeds:
                m14_r = _eval_slu_pass(
                    _make_nlp_service(cfg, bench["minds14_labels"], None, s),
                    bench["minds14_stratified"], "text", bench["minds14_labels"],
                    ckpt_path=os.path.join(ckpt_dir, f"{cname}_m14_s{s}.json"), resume=resume, progress_tag=f"{cname}:m14_s{s}",
                )
                cl_r = _eval_slu_pass(
                    _make_nlp_service(cfg, bench["clinc_15_labels"], None, s, oos_tau=oos_tau),
                    bench["clinc_test_15"], "text", bench["clinc_15_labels"],
                    ckpt_path=os.path.join(ckpt_dir, f"{cname}_clinc_s{s}.json"), resume=resume, progress_tag=f"{cname}:clinc_s{s}",
                )
                m14_gold_runs.append(m14_r)
                clinc_runs.append(cl_r)

                if cfg["provider"] != "distilbert_lora":
                    iv_r = _eval_slu_pass(
                        _make_nlp_service(cfg, INTERVIEW_INTENTS, None, s),
                        bench["interview_qa"], "text", INTERVIEW_INTENTS,
                        ckpt_path=os.path.join(ckpt_dir, f"{cname}_iv_s{s}.json"), resume=resume, progress_tag=f"{cname}:iv_s{s}",
                    )
                    rub_r = _eval_rubric_pass(
                        cfg, bench["interview_answers"], s,
                        ckpt_path=os.path.join(ckpt_dir, f"{cname}_rubric_cal_s{s}.json"), resume=resume,
                    )
                    iv_slu_runs.append(iv_r)
                    rubric_runs.append(rub_r)
                    print(
                        f"  seed={s}: MINDS14({len(bench['minds14_stratified'])}) F1={m14_r['macro_f1']} rawV={m14_r['raw_valid_rate']} schV={m14_r['schema_valid_rate']} | "
                        f"CLINC({len(bench['clinc_test_15'])}) F1={cl_r['macro_f1']} oosF1={cl_r.get('oos_f1')} | "
                        f"IV_F1={iv_r['macro_f1']} | Rubric rho={rub_r['spearman_vs_tier']}",
                        flush=True,
                    )
                else:
                    print(f"  seed={s}: MINDS14({len(bench['minds14_stratified'])}) F1={m14_r['macro_f1']} | CLINC({len(bench['clinc_test_15'])}) F1={cl_r['macro_f1']} oosF1={cl_r.get('oos_f1')}", flush=True)

            m14_tiny_asr = _eval_slu_pass(_make_nlp_service(cfg, bench["minds14_labels"], MINDS14_BANKING_TERMS, 42), bench["minds14_stratified"], "asr_tiny_baseline", bench["minds14_labels"], ckpt_path=os.path.join(ckpt_dir, f"{cname}_m14_tiny.json"), resume=resume, progress_tag=f"{cname}:m14_tiny")
            m14_tiny_conf = _eval_slu_pass(_make_nlp_service(cfg, bench["minds14_labels"], MINDS14_BANKING_TERMS, 42), bench["minds14_stratified"], "asr_tiny_baseline", bench["minds14_labels"], use_words_meta=True, ckpt_path=os.path.join(ckpt_dir, f"{cname}_m14_tiny_conf.json"), resume=resume, progress_tag=f"{cname}:m14_tiny_conf")
            m14_small_asr = _eval_slu_pass(_make_nlp_service(cfg, bench["minds14_labels"], MINDS14_BANKING_TERMS, 42), bench["minds14_stratified"], "asr_small_improved", bench["minds14_labels"], ckpt_path=os.path.join(ckpt_dir, f"{cname}_m14_small.json"), resume=resume, progress_tag=f"{cname}:m14_small")
            m14_parakeet_asr = _eval_slu_pass(_make_nlp_service(cfg, bench["minds14_labels"], MINDS14_BANKING_TERMS, 42), bench["minds14_stratified"], "asr_parakeet_tdt", bench["minds14_labels"], ckpt_path=os.path.join(ckpt_dir, f"{cname}_m14_para.json"), resume=resume, progress_tag=f"{cname}:m14_para")

            iv_tiny_asr = _eval_slu_pass(_make_nlp_service(cfg, INTERVIEW_INTENTS, INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES, 42), bench["interview_qa"], "asr_tiny_baseline", INTERVIEW_INTENTS) if cfg["provider"] != "distilbert_lora" else None
            iv_small_asr = _eval_slu_pass(_make_nlp_service(cfg, INTERVIEW_INTENTS, INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES, 42), bench["interview_qa"], "asr_small_improved", INTERVIEW_INTENTS) if cfg["provider"] != "distilbert_lora" else None

            m14_full_clean, m14_full_tiny, clinc_full_res = None, None, None
            if cfg["provider"] in ("rule_based", "distilbert_lora", "hybrid_cascade") or (run_full_on_top and cname in full_eval_slms):
                print(f"  [FULL 498 MINDS-14 sweep] running {cname}...", flush=True)
                m14_full_clean = _eval_slu_pass(
                    _make_nlp_service(cfg, bench["minds14_labels"], None, 42),
                    bench["minds14_full_test"], "text", bench["minds14_labels"],
                    ckpt_path=os.path.join(ckpt_dir, f"{cname}_m14_full498_gold.json"), resume=resume, progress_tag=f"{cname}:m14_498_gold",
                )
                m14_full_tiny = _eval_slu_pass(
                    _make_nlp_service(cfg, bench["minds14_labels"], MINDS14_BANKING_TERMS, 42),
                    bench["minds14_full_test"], "asr_tiny_baseline", bench["minds14_labels"], use_words_meta=True,
                    ckpt_path=os.path.join(ckpt_dir, f"{cname}_m14_full498_tiny.json"), resume=resume, progress_tag=f"{cname}:m14_498_tiny",
                )
                print(f"    MINDS-14 FULL (498): gold F1={m14_full_clean['macro_f1']} | tiny_asr+conf F1={m14_full_tiny['macro_f1']}", flush=True)
            if cfg["provider"] in ("rule_based", "distilbert_lora"):
                clinc_full_res = _eval_slu_pass(
                    _make_nlp_service(cfg, bench["clinc_full_labels"], None, 42),
                    bench["clinc_full_test"], "text", bench["clinc_full_labels"],
                    ckpt_path=os.path.join(ckpt_dir, f"{cname}_clinc_full5500.json"), resume=resume, progress_tag=f"{cname}:clinc_5500",
                )
                print(f"    CLINC FULL (5,500): F1={clinc_full_res['macro_f1']} OOS_F1={clinc_full_res.get('oos_f1')}", flush=True)

            avg_lat_s = m14_gold_runs[0]["avg_latency_ms"] / 1000.0
            hourly_rate = 0.15 if "8b" in cname else (0.05 if cfg["provider"] in ("ollama", "hybrid_cascade") else 0.01)

            def _strip(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
                if d is None:
                    return None
                return {k: v for k, v in d.items() if k not in ("per_item_correct", "y_pred")}

            entry: Dict[str, Any] = {
                "config": cfg,
                "model_disk_gb": model_info["disk_gb"],
                "model_vram_gb": model_info["vram_gb"],
                "cost_per_1k_turns_usd": estimate_cost_per_1k_turns(avg_lat_s, compute_hourly_usd=hourly_rate),
                "val_oos_calibration": val_cal_info,
                "minds14_gold": {
                    "summary": _aggregate_seed_runs(m14_gold_runs, ["accuracy", "macro_f1", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                    "seed_42": _strip(m14_gold_runs[0]),
                    "per_item_correct_s42": m14_gold_runs[0]["per_item_correct"],
                },
                "minds14_asr_tiny_baseline": _strip(m14_tiny_asr),
                "minds14_asr_tiny_with_conf": _strip(m14_tiny_conf),
                "minds14_asr_small_improved": _strip(m14_small_asr),
                "minds14_asr_parakeet_tdt": _strip(m14_parakeet_asr),
                "minds14_full_498_gold": _strip(m14_full_clean),
                "minds14_full_498_tiny_asr": _strip(m14_full_tiny),
                "clinc_oos": {
                    "summary": _aggregate_seed_runs(clinc_runs, ["accuracy", "macro_f1", "oos_f1", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                    "seed_42": _strip(clinc_runs[0]),
                    "per_item_correct_s42": clinc_runs[0]["per_item_correct"],
                },
                "clinc_full_5500": _strip(clinc_full_res),
                "interview_slu_gold": {
                    "summary": _aggregate_seed_runs(iv_slu_runs, ["accuracy", "macro_f1", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                    "seed_42": _strip(iv_slu_runs[0]),
                } if iv_slu_runs else None,
                "interview_slu_asr_tiny_baseline": _strip(iv_tiny_asr),
                "interview_slu_asr_small_improved": _strip(iv_small_asr),
                "interview_rubric": {
                    "summary": _aggregate_seed_runs(rubric_runs, ["spearman_vs_tier", "mae_vs_expected_midpoint", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                    "seed_42_scores": rubric_runs[0]["scores"],
                } if rubric_runs else None,
            }
            results["configs"][cname] = entry
            results["failed_configs"].pop(cname, None)
            per_item_correct_seed42[cname] = {
                "minds14": m14_gold_runs[0]["per_item_correct"],
                "clinc": clinc_runs[0]["per_item_correct"],
            }
            _atomic_json_dump(results, out_path)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  [ERROR in {cname}] {err_msg} (continuing cleanly to next config!)", flush=True)
            results["failed_configs"][cname] = err_msg[:300]
            _atomic_json_dump(results, out_path)

    llm_gateway.configure(provider_id="ollama", fallback_enabled=True)

    cfgs = results["configs"]
    if "slu_llama8b_ref" in cfgs and cfgs["slu_llama8b_ref"].get("interview_rubric"):
        ref_scores = np.asarray(cfgs["slu_llama8b_ref"]["interview_rubric"]["seed_42_scores"], dtype=np.float64)
        for cname, entry in cfgs.items():
            if entry.get("interview_rubric"):
                cand = np.asarray(entry["interview_rubric"]["seed_42_scores"], dtype=np.float64)
                rho_8b, _ = spearmanr(cand, ref_scores)
                entry["interview_rubric"]["spearman_vs_8b_ref"] = round(float(rho_8b) if np.isfinite(rho_8b) else 0.0, 4)
                entry["interview_rubric"]["mae_vs_8b_ref"] = round(float(np.mean(np.abs(cand - ref_scores))), 3)

    for cname, entry in cfgs.items():
        if cname not in per_item_correct_seed42:
            per_item_correct_seed42[cname] = {
                "minds14": entry["minds14_gold"].get("per_item_correct_s42", []),
                "clinc": entry["clinc_oos"].get("per_item_correct_s42", []),
            }

    # Bounded Paired Bootstrap (N=2000) & TOST Non-Inferiority (margin=-0.05)
    pairs = [
        ("qwen15b_schema_vs_raw_minds14", "slu_qwen15b_raw", "slu_qwen15b_schema", "minds14"),
        ("qwen15b_repair_vs_schema_minds14", "slu_qwen15b_schema", "slu_qwen15b_repair", "minds14"),
        ("qwen3_4b_repair_vs_schema_clinc", "slu_qwen3_4b_schema", "slu_qwen3_4b_repair", "clinc"),
        ("phi4_mini_repair_vs_schema_clinc", "slu_phi4_mini_schema", "slu_phi4_mini_repair", "clinc"),
        ("qwen3_4b_vs_8b_ref_minds14", "slu_llama8b_ref", "slu_qwen3_4b_repair", "minds14"),
        ("phi4_mini_vs_8b_ref_minds14", "slu_llama8b_ref", "slu_phi4_mini_repair", "minds14"),
        ("distilbert_lora_vs_8b_ref_minds14", "slu_llama8b_ref", "slu_distilbert_lora", "minds14"),
        ("qwen3_4b_vs_rule_based_minds14", "slu_rule_based", "slu_qwen3_4b_repair", "minds14"),
        ("distilbert_lora_vs_rule_based_minds14", "slu_rule_based", "slu_distilbert_lora", "minds14"),
        ("hybrid_vs_qwen3_minds14", "slu_qwen3_4b_repair", "slu_hybrid_cascade", "minds14"),
        ("hybrid_vs_qwen3_clinc", "slu_qwen3_4b_repair", "slu_hybrid_cascade", "clinc"),
        ("hybrid_vs_8b_ref_minds14", "slu_llama8b_ref", "slu_hybrid_cascade", "minds14"),
    ]
    for label, bk, ck, ds_k in pairs:
        ba = per_item_correct_seed42.get(bk, {}).get(ds_k)
        ca = per_item_correct_seed42.get(ck, {}).get(ds_k)
        if ba and ca and len(ba) == len(ca):
            results["paired_bootstrap"][label] = paired_bootstrap_test(
                baseline_scores=ba, candidate_scores=ca, n_resamples=2000, seed=42, higher_is_better=True
            )
            if "8b_ref" in label:
                results["tost_non_inferiority"][label] = tost_non_inferiority(
                    baseline_scores=ba, candidate_scores=ca, margin=-0.05, n_resamples=2000, seed=42, higher_is_better=True
                )

    _atomic_json_dump(results, out_path)
    print(f"\nSaved complete Layer B v2 ablation results -> {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Run specific SLU configs and merge into slu_ablation.json")
    parser.add_argument("--skip-full-slms", action="store_true", help="Skip 498-item MINDS-14 sweeps on generative SLMs")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and re-run from scratch")
    parser.add_argument("--rubric-only", action="store_true", help="Re-evaluate calibrated RoleService rubric and update existing configs")
    args = parser.parse_args()
    run_ablation(
        only_configs=args.only,
        run_full_on_top=not args.skip_full_slms,
        resume=not args.no_resume,
        rubric_only=args.rubric_only,
    )
