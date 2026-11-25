"""
Session 6 Experiment: Layer B (Stage 3/4 SLU & Interview Rubric) Quantized SLM Recovery & Cost Pareto.
Evaluates 7 reproducible configurations across 3 seeds ([42, 1337, 2026]):
  1. slu_rule_based    (Deterministic regex & keyword baseline)
  2. slu_qwen15b_raw   (Qwen 2.5 Coder 1.5B raw JSON without repair)
  3. slu_qwen15b_repair (Qwen 2.5 Coder 1.5B + JSON mode + shared repair + 1-shot retry + ASR pre-repair)
  4. slu_llama3b_raw   (Llama 3.2 3B raw JSON without repair)
  5. slu_llama3b_repair (Llama 3.2 3B + JSON mode + shared repair + 1-shot retry + ASR pre-repair)
  6. slu_qwen3b_repair (Qwen 2.5 3B + JSON mode + shared repair + 1-shot retry + ASR pre-repair)
  7. slu_llama8b_ref   (Llama 3.1 8B reference ceiling)

Evaluates across:
  - MINDS-14 test slice (28 items = 2/intent x 14 classes): 3-seed gold + real ASR transcripts (asr_tiny_baseline, asr_small_improved)
  - CLINC-OOS test slice (35 items = 7 oos + 2 x 14 ID classes): 3-seed Macro-F1 & OOS F1
  - Interview Stress QA test (30 items, 7 intents) + Rubric Evaluation (10 balanced answer items x 3 seeds via RoleService)

Saves incremental checkpoints per config and full stats/Pareto/bootstrap p-values to ml/results/slu_ablation.json.
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
from services.nlp import RuleBasedNLPService, LLMNLPService
from services.roles.role_service import RoleService
from services.context import context_manager
from ml.datasets import load_manifest
from ml.metrics import (
    compute_slu_metrics,
    paired_bootstrap_test,
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
    "slu_qwen15b_raw",
    "slu_llama3b_raw",
    "slu_qwen15b_repair",
    "slu_llama3b_repair",
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


def _ollama_model_info(model_name: str) -> Dict[str, float]:
    """Reads disk weight (GB) and live VRAM allocation (GB) from local Ollama daemon."""
    disk_gb = 0.0
    vram_gb = 0.0
    try:
        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=3).json()
        for m in tags.get("models", []):
            if m.get("name") == model_name:
                disk_gb = round(float(m.get("size", 0)) / (1024 ** 3), 2)
        ps = requests.get("http://127.0.0.1:11434/api/ps", timeout=3).json()
        for m in ps.get("models", []):
            if m.get("name") == model_name:
                vram_gb = round(float(m.get("size_vram", 0)) / (1024 ** 3), 2)
    except Exception:
        pass
    return {"disk_gb": disk_gb, "vram_gb": vram_gb}


def _load_benchmarks() -> Dict[str, Any]:
    """Selects deterministic, class-balanced test subsets and aligns Session 5 ASR transcripts."""
    rng = np.random.RandomState(42)

    # 1) MINDS-14: 2 test items per each of 14 classes = 28 items
    m14_all = [x for x in load_manifest("minds14_en_140")["items"] if x["split"] == "test"]
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for it in m14_all:
        by_intent.setdefault(it["intent_label"], []).append(it)
    m14_subset: List[Dict[str, Any]] = []
    for label in sorted(by_intent.keys()):
        m14_subset.extend(by_intent[label][:2])

    # Load Session 5 real ASR transcripts keyed by utterance ID
    asr_path = os.path.join(ROOT_DIR, "ml", "results", "asr_ablation.json")
    asr_tiny_m14: Dict[str, str] = {}
    asr_small_m14: Dict[str, str] = {}
    asr_tiny_iv: Dict[str, str] = {}
    asr_small_iv: Dict[str, str] = {}
    if os.path.exists(asr_path):
        asr_data = json.load(open(asr_path, "r", encoding="utf-8"))
        for r in asr_data["configs"]["asr_tiny_baseline"]["minds14_clean"]["transcripts"]:
            asr_tiny_m14[r["id"]] = r["hypothesis"]
        for r in asr_data["configs"]["asr_small_improved"]["minds14_clean"]["transcripts"]:
            asr_small_m14[r["id"]] = r["hypothesis"]
        for r in asr_data["configs"]["asr_tiny_baseline"]["interview_stress_clean"]["transcripts"]:
            asr_tiny_iv[r["id"]] = r["hypothesis"]
        for r in asr_data["configs"]["asr_small_improved"]["interview_stress_clean"]["transcripts"]:
            asr_small_iv[r["id"]] = r["hypothesis"]

    for it in m14_subset:
        it["asr_tiny_baseline"] = asr_tiny_m14.get(it["id"], it["text"])
        it["asr_small_improved"] = asr_small_m14.get(it["id"], it["text"])

    # 2) CLINC-OOS: all 7 oos + 2 per 14 in-domain classes = 35 items
    clinc_test = [x for x in load_manifest("clinc_oos_200")["items"] if x["split"] == "test"]
    by_clinc: Dict[str, List[Dict[str, Any]]] = {}
    for it in clinc_test:
        by_clinc.setdefault(it["intent_label"], []).append(it)
    clinc_subset: List[Dict[str, Any]] = list(by_clinc.get("oos", [])[:7])
    for label in sorted(by_clinc.keys()):
        if label != "oos":
            clinc_subset.extend(by_clinc[label][:2])
    clinc_labels = sorted(by_clinc.keys())

    # 3) Interview Stress QA (30 test items) & 10 balanced 'answer' turns for rubric evaluation
    iv_qa = [
        x for x in load_manifest("interview_stress_v1")["items"]
        if x["split"] == "test" and x.get("subset") == "interview_qa"
    ]
    for it in iv_qa:
        it["asr_tiny_baseline"] = asr_tiny_iv.get(it["id"], it["text"])
        it["asr_small_improved"] = asr_small_iv.get(it["id"], it["text"])
    iv_answers = [x for x in iv_qa if x.get("intent_label") == "answer"]

    return {
        "minds14": m14_subset,
        "minds14_labels": sorted(by_intent.keys()),
        "clinc": clinc_subset,
        "clinc_labels": clinc_labels,
        "interview_qa": iv_qa,
        "interview_answers": iv_answers,
    }


def _make_nlp_service(cfg: Dict[str, Any], allowed_intents: Optional[Sequence[str]], repair_lexicon: Optional[Sequence[str]], seed: int):
    if cfg["provider"] == "rule_based":
        return RuleBasedNLPService(repair_lexicon=repair_lexicon if cfg.get("asr_pre_repair") else None)
    return LLMNLPService(
        model_name=cfg["model_name"],
        allowed_intents=allowed_intents,
        repair_lexicon=repair_lexicon if cfg.get("asr_pre_repair") else None,
        enable_schema_repair=bool(cfg.get("enable_schema_repair", True)),
        enable_repair_retry=bool(cfg.get("enable_repair_retry", True)),
        use_json_mode=bool(cfg.get("use_json_mode", True)),
        temperature=float(cfg.get("temperature", 0.1)),
        seed=seed,
    )


def _eval_slu_pass(
    nlp_svc: Any,
    items: Sequence[Dict[str, Any]],
    text_key: str,
    label_set: Sequence[str],
) -> Dict[str, Any]:
    """Runs one deterministic pass of SLU parsing and returns macro_f1, validity, latencies, per-item correctness."""
    y_true: List[str] = []
    y_pred: List[str] = []
    raw_valid_flags: List[bool] = []
    schema_valid_flags: List[bool] = []
    latencies_ms: List[float] = []
    correct_flags: List[float] = []
    repaired_count = 0
    retried_count = 0
    asr_fix_count = 0
    valid_labels = {l.lower(): l for l in label_set}

    for it in items:
        txt = it[text_key]
        t0 = time.perf_counter()
        res = nlp_svc.parse(txt)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        gold = it["intent_label"]
        raw_intent = (res.intent or "").strip().lower().replace(" ", "_")
        pred = valid_labels.get(raw_intent, raw_intent or "__invalid__")
        if cfg_fallback := ("fallback" in (res.provider or "").lower()):
            if it.get("subset") != "rule_based_expected":
                pred = "__fallback__" if raw_intent not in valid_labels or "rule_based_fallback" in (res.provider or "").lower() else pred

        y_true.append(gold)
        y_pred.append(pred)
        correct_flags.append(1.0 if pred == gold else 0.0)

        meta = res.metadata or {}
        raw_v = bool(meta.get("raw_valid", True))
        sch_v = bool(meta.get("schema_valid", "fallback" not in (res.provider or "").lower()))
        raw_valid_flags.append(raw_v)
        schema_valid_flags.append(sch_v)
        if meta.get("repaired"):
            repaired_count += 1
        if meta.get("retried"):
            retried_count += 1
        asr_fix_count += len(meta.get("asr_repairs") or [])

    metrics = compute_slu_metrics(y_true, y_pred, schema_valid_flags=schema_valid_flags)
    metrics["raw_valid_rate"] = round(sum(1 for x in raw_valid_flags if x) / len(raw_valid_flags), 4)
    metrics["avg_latency_ms"] = round(float(np.mean(latencies_ms)), 1)
    metrics["p95_latency_ms"] = round(float(np.percentile(latencies_ms, 95)), 1)
    metrics["repaired_turns"] = repaired_count
    metrics["retried_turns"] = retried_count
    metrics["asr_pre_repairs"] = asr_fix_count
    metrics["per_item_correct"] = correct_flags
    metrics["y_pred"] = y_pred

    if "oos" in label_set:
        oos_f1 = float(f1_score([y == "oos" for y in y_true], [p == "oos" for p in y_pred], zero_division=0))
        metrics["oos_f1"] = round(oos_f1, 4)
    return metrics


def _eval_rubric_pass(cfg: Dict[str, Any], answer_items: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """Evaluates RoleService.process_interviewer_turn alignment on the 10 stratified test interview answers."""
    role = RoleService()
    scores: List[float] = []
    tier_ordinals: List[float] = []
    mid_targets: List[float] = []
    raw_flags: List[bool] = []
    schema_flags: List[bool] = []
    latencies_ms: List[float] = []

    for it in answer_items:
        context_manager.sessions.clear()
        role.current_interview_question_idx = 2  # Technical architecture stage
        t0 = time.perf_counter()
        if cfg["provider"] == "rule_based":
            # Force heuristic fallback without contacting Ollama
            prev_fb = llm_gateway.active_provider_id
            llm_gateway.active_provider_id = "rule_based"
            try:
                out = role.process_interviewer_turn(it["text"], mode="solo")
            finally:
                llm_gateway.active_provider_id = prev_fb
        else:
            out = role.process_interviewer_turn(
                it["text"],
                mode="solo",
                model_name=cfg["model_name"],
                use_json_mode=bool(cfg.get("use_json_mode", True)),
                enable_schema_repair=bool(cfg.get("enable_schema_repair", True)),
                enable_repair_retry=bool(cfg.get("enable_repair_retry", True)),
                seed=seed,
                temperature=float(cfg.get("temperature", 0.1)),
            )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        sc = float(out["evaluation"]["overall_score"])
        scores.append(sc)
        tier_ordinals.append(TIER_TO_ORDINAL[it["rubric_tier"]])
        lo, hi = it["expected_score_range"]
        mid_targets.append((lo + hi) / 2.0)
        diag = out.get("schema_diagnostics") or {}
        raw_flags.append(bool(diag.get("raw_valid", True)))
        schema_flags.append(bool(diag.get("schema_valid", True)))

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
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            out[k] = summarize_seeds(vals)
    return out


def run_ablation(only_configs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    bench = _load_benchmarks()
    out_path = os.path.join(ROOT_DIR, "ml", "results", "slu_ablation.json")
    existing: Dict[str, Any] = {}
    if only_configs and os.path.exists(out_path):
        existing = json.load(open(out_path, "r", encoding="utf-8"))

    results: Dict[str, Any] = {
        "experiment": "layer_b_slu_rubric_ablation_v1",
        "configs": dict(existing.get("configs", {})),
        "paired_bootstrap": {},
        "sample_counts": {
            "minds14_test_stratified": len(bench["minds14"]),
            "clinc_oos_test_stratified": len(bench["clinc"]),
            "interview_qa_test": len(bench["interview_qa"]),
            "interview_rubric_answers": len(bench["interview_answers"]),
        },
    }

    # Ensure gateway points to local Ollama cleanly without silent fallback masking failures
    llm_gateway.configure(provider_id="ollama", fallback_enabled=False)

    targets = list(only_configs) if only_configs else list(CONFIG_NAMES)
    per_item_correct_seed42: Dict[str, Dict[str, List[float]]] = {}

    for cname in targets:
        cfg_path = os.path.join(ROOT_DIR, "ml", "configs", f"{cname}.json")
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        seeds = cfg.get("seeds", [42, 1337, 2026])
        if cfg["provider"] == "rule_based":
            seeds = [42]

        print(f"\n=== Running {cname} (model={cfg['model_name']}, json={cfg['use_json_mode']}, repair={cfg['enable_schema_repair']}, pre_repair={cfg['asr_pre_repair']}) ===", flush=True)

        # Warm model once before timed sweeps
        if cfg["provider"] == "ollama":
            _make_nlp_service(cfg, bench["minds14_labels"], None, 42).parse("check my balance")
        model_info = _ollama_model_info(cfg["model_name"]) if cfg["provider"] == "ollama" else {"disk_gb": 0.0, "vram_gb": 0.0}

        m14_gold_runs = []
        clinc_runs = []
        iv_slu_runs = []
        rubric_runs = []

        for s in seeds:
            m14_r = _eval_slu_pass(
                _make_nlp_service(cfg, bench["minds14_labels"], None, s),
                bench["minds14"], "text", bench["minds14_labels"]
            )
            cl_r = _eval_slu_pass(
                _make_nlp_service(cfg, bench["clinc_labels"], None, s),
                bench["clinc"], "text", bench["clinc_labels"]
            )
            iv_r = _eval_slu_pass(
                _make_nlp_service(cfg, INTERVIEW_INTENTS, None, s),
                bench["interview_qa"], "text", INTERVIEW_INTENTS
            )
            rub_r = _eval_rubric_pass(cfg, bench["interview_answers"], s)
            m14_gold_runs.append(m14_r)
            clinc_runs.append(cl_r)
            iv_slu_runs.append(iv_r)
            rubric_runs.append(rub_r)
            print(
                f"  seed={s}: MINDS14 F1={m14_r['macro_f1']} rawV={m14_r['raw_valid_rate']} schV={m14_r['schema_valid_rate']} | "
                f"CLINC F1={cl_r['macro_f1']} oosF1={cl_r.get('oos_f1')} | "
                f"IV_F1={iv_r['macro_f1']} | Rubric rho={rub_r['spearman_vs_tier']} rawV={rub_r['raw_valid_rate']} schV={rub_r['schema_valid_rate']}",
                flush=True,
            )

        # Cross-layer ASR transcript cascade passes on MINDS-14 & Interview (seed=42)
        m14_tiny_asr = _eval_slu_pass(
            _make_nlp_service(cfg, bench["minds14_labels"], MINDS14_BANKING_TERMS, 42),
            bench["minds14"], "asr_tiny_baseline", bench["minds14_labels"]
        )
        m14_small_asr = _eval_slu_pass(
            _make_nlp_service(cfg, bench["minds14_labels"], MINDS14_BANKING_TERMS, 42),
            bench["minds14"], "asr_small_improved", bench["minds14_labels"]
        )
        iv_tiny_asr = _eval_slu_pass(
            _make_nlp_service(cfg, INTERVIEW_INTENTS, INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES, 42),
            bench["interview_qa"], "asr_tiny_baseline", INTERVIEW_INTENTS
        )
        iv_small_asr = _eval_slu_pass(
            _make_nlp_service(cfg, INTERVIEW_INTENTS, INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES, 42),
            bench["interview_qa"], "asr_small_improved", INTERVIEW_INTENTS
        )
        print(
            f"  [ASR Cascade s=42] MINDS14(tiny_asr)={m14_tiny_asr['macro_f1']} preFix={m14_tiny_asr['asr_pre_repairs']} -> "
            f"(small_imp)={m14_small_asr['macro_f1']} | IV(tiny_asr)={iv_tiny_asr['macro_f1']} -> (small_imp)={iv_small_asr['macro_f1']}",
            flush=True,
        )

        # Additional causal ablation: pre-repair OFF vs ON on noisy tiny ASR for repaired SLMs
        m14_tiny_no_prerepair = None
        if cfg.get("asr_pre_repair"):
            m14_tiny_no_prerepair = _eval_slu_pass(
                _make_nlp_service({**cfg, "asr_pre_repair": False}, bench["minds14_labels"], None, 42),
                bench["minds14"], "asr_tiny_baseline", bench["minds14_labels"]
            )

        avg_lat_s = m14_gold_runs[0]["avg_latency_ms"] / 1000.0
        hourly_rate = 0.15 if "8b" in cname else (0.05 if cfg["provider"] == "ollama" else 0.01)

        def _strip(d: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in d.items() if k not in ("per_item_correct", "y_pred")}

        entry = {
            "config": cfg,
            "model_disk_gb": model_info["disk_gb"],
            "model_vram_gb": model_info["vram_gb"],
            "cost_per_1k_turns_usd": estimate_cost_per_1k_turns(avg_lat_s, compute_hourly_usd=hourly_rate),
            "minds14_gold": {
                "summary": _aggregate_seed_runs(m14_gold_runs, ["accuracy", "macro_f1", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                "seed_42": _strip(m14_gold_runs[0]),
                "per_item_correct_s42": m14_gold_runs[0]["per_item_correct"],
            },
            "minds14_asr_tiny_baseline": _strip(m14_tiny_asr),
            "minds14_asr_tiny_no_prerepair": _strip(m14_tiny_no_prerepair) if m14_tiny_no_prerepair else None,
            "minds14_asr_small_improved": _strip(m14_small_asr),
            "clinc_oos": {
                "summary": _aggregate_seed_runs(clinc_runs, ["accuracy", "macro_f1", "oos_f1", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                "seed_42": _strip(clinc_runs[0]),
                "per_item_correct_s42": clinc_runs[0]["per_item_correct"],
            },
            "interview_slu_gold": {
                "summary": _aggregate_seed_runs(iv_slu_runs, ["accuracy", "macro_f1", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                "seed_42": _strip(iv_slu_runs[0]),
            },
            "interview_slu_asr_tiny_baseline": _strip(iv_tiny_asr),
            "interview_slu_asr_small_improved": _strip(iv_small_asr),
            "interview_rubric": {
                "summary": _aggregate_seed_runs(rubric_runs, ["spearman_vs_tier", "mae_vs_expected_midpoint", "raw_valid_rate", "schema_valid_rate", "avg_latency_ms"]),
                "seed_42_scores": rubric_runs[0]["scores"],
            },
        }
        results["configs"][cname] = entry
        per_item_correct_seed42[cname] = {
            "minds14": m14_gold_runs[0]["per_item_correct"],
            "clinc": clinc_runs[0]["per_item_correct"],
        }

        # Incremental checkpoint save so partial multi-model sweeps are crash-safe
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Restore default gateway configuration when done
    llm_gateway.configure(provider_id="ollama", fallback_enabled=True)

    # Recompute rubric agreement vs 8B reference ceiling if slu_llama8b_ref is present
    cfgs = results["configs"]
    if "slu_llama8b_ref" in cfgs:
        ref_scores = np.asarray(cfgs["slu_llama8b_ref"]["interview_rubric"]["seed_42_scores"], dtype=np.float64)
        for cname, entry in cfgs.items():
            cand = np.asarray(entry["interview_rubric"]["seed_42_scores"], dtype=np.float64)
            rho_8b, _ = spearmanr(cand, ref_scores)
            entry["interview_rubric"]["spearman_vs_8b_ref"] = round(float(rho_8b) if np.isfinite(rho_8b) else 0.0, 4)
            entry["interview_rubric"]["mae_vs_8b_ref"] = round(float(np.mean(np.abs(cand - ref_scores))), 3)

    # Populate per_item cache for untouched configs from saved JSON
    for cname, entry in cfgs.items():
        if cname not in per_item_correct_seed42:
            per_item_correct_seed42[cname] = {
                "minds14": entry["minds14_gold"].get("per_item_correct_s42", []),
                "clinc": entry["clinc_oos"].get("per_item_correct_s42", []),
            }

    pairs = [
        ("qwen15b_repair_vs_raw_minds14_acc", "slu_qwen15b_raw", "slu_qwen15b_repair", "minds14"),
        ("qwen15b_repair_vs_raw_clinc_acc", "slu_qwen15b_raw", "slu_qwen15b_repair", "clinc"),
        ("llama3b_repair_vs_raw_minds14_acc", "slu_llama3b_raw", "slu_llama3b_repair", "minds14"),
        ("qwen3b_repair_vs_8b_ref_minds14_acc", "slu_llama8b_ref", "slu_qwen3b_repair", "minds14"),
        ("qwen3b_repair_vs_8b_ref_clinc_acc", "slu_llama8b_ref", "slu_qwen3b_repair", "clinc"),
        ("qwen3b_repair_vs_rule_based_minds14_acc", "slu_rule_based", "slu_qwen3b_repair", "minds14"),
    ]
    for label, bk, ck, ds_k in pairs:
        ba = per_item_correct_seed42.get(bk, {}).get(ds_k)
        ca = per_item_correct_seed42.get(ck, {}).get(ds_k)
        if ba and ca and len(ba) == len(ca):
            results["paired_bootstrap"][label] = paired_bootstrap_test(
                baseline_scores=ba, candidate_scores=ca, n_resamples=1000, seed=42, higher_is_better=True
            )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete Layer B ablation results -> {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Run specific SLU configs and merge into slu_ablation.json")
    args = parser.parse_args()
    run_ablation(only_configs=args.only)
