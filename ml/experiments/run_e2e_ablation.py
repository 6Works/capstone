"""
Session 8 Experiment (Upgraded): End-to-End 6-Stage Ablation Matrix & Accuracy-vs-Cost Frontier.
Executes real Audio -> Stage 2 ASR -> Stage 3 SLU -> Stage 4 Dialogue -> Stage 5 TTS -> Verification
across 9 rows (Baseline, +A, +B, +C, Full A+B+C, Full-A, Full-B, Full-C, Edge A_tiny+B+C)
over 3 random seeds ([42, 1337, 2026]) across 36 multi-domain spoken turns (108 turn evals/row).

Upgrades per review:
  - Per-Stage Pass Rates & Continuous Graded Composite Quality Q elevated alongside Conjunctive Strict E2E.
  - Turn-level 3-seed averaged Paired Bootstrap (N=36 unique turns, B=2000 Phipson-Smyth, never turn x seed!)
    plus TOST non-inferiority tests (margin = -0.05).
  - Supports G2P phoneme repair, calibrated Ollama GBNF JSON-Schema SLMs (Qwen3-4B / Phi-4-mini / Llama-3.2),
    Parakeet-TDT, cross-layer ASR word-confidence context, and dual CAM++/ECAPA speaker verification + TTFA.
  - Empirically verified mathematical Pareto frontier (strictly non-dominated configs only).
"""

import os
import sys
import json
import time
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.asr import WhisperTransformerASR, ParakeetTDTASR
from services.asr.asr_services import _WHISPER_CACHE
from services.nlp import LLMNLPService, TrainedDistilBertSLUService, HybridConfidenceSLUService
from services.dialogue import ModularDialogueManager
from services.tts import NeuralVoiceCloningTTSService, PocketTTSCloningService, VoiceCloner, voice_cloner
from services.llm.gateway import llm_gateway
from services.contracts import AudioPayload
from ml.datasets import load_manifest
from ml.metrics import (
    compute_asr_metrics,
    compute_speaker_similarity,
    paired_bootstrap_test,
    tost_non_inferiority,
    summarize_seeds,
    estimate_cost_per_1k_turns,
)
from ml.experiments.run_asr_ablation import (
    _build_domain_overrides,
    _sample_wer,
    _sample_term_recall,
    INTERVIEW_ROLE_TERMS,
    CANDIDATE_ROSTER_NAMES,
    MINDS14_BANKING_TERMS,
)


INTERVIEW_INTENTS = [
    "greeting",
    "identity_query",
    "project_query",
    "calculation_query",
    "time_query",
    "answer",
    "farewell",
]


def _curate_e2e_turns() -> Tuple[List[Dict[str, Any]], List[str]]:
    """Selects 36 balanced multi-domain spoken test turns (8 proper_nouns + 14 QA + 14 MINDS-14 across 3 accents)."""
    iv_test = [x for x in load_manifest("interview_stress_v1")["items"] if x["split"] == "test"]
    pn_items = [x for x in iv_test if x.get("subset") == "proper_nouns"]
    qa_items = [x for x in iv_test if x.get("subset") == "interview_qa"]

    chosen_pn = [dict(x) for x in pn_items[:8]]
    for it in chosen_pn:
        it["domain"] = "interview"
        it["e2e_intent"] = "answer"
        it["allowed_intents"] = INTERVIEW_INTENTS

    by_qa_int: Dict[str, List[Dict[str, Any]]] = {}
    for it in qa_items:
        by_qa_int.setdefault(it["intent_label"], []).append(it)
    chosen_qa: List[Dict[str, Any]] = []
    for k in ("greeting", "identity_query", "project_query", "calculation_query", "time_query", "answer", "farewell"):
        for x in by_qa_int.get(k, [])[:2]:
            chosen_qa.append(dict(x))
    for it in qa_items:
        if len(chosen_qa) >= 14:
            break
        if all(it["id"] != x["id"] for x in chosen_qa):
            chosen_qa.append(dict(it))
    for it in chosen_qa:
        it["domain"] = "interview"
        it["e2e_intent"] = it["intent_label"]
        it["allowed_intents"] = INTERVIEW_INTENTS

    m14_test = [x for x in load_manifest("minds14_en_full")["items"] if x["split"] == "test"]
    by_m14: Dict[str, List[Dict[str, Any]]] = {}
    for it in m14_test:
        by_m14.setdefault(it["intent_label"], []).append(it)
    m14_labels = sorted(by_m14.keys())
    chosen_m14: List[Dict[str, Any]] = []
    for lbl in m14_labels[:14]:
        it = dict(by_m14[lbl][0])
        it["domain"] = "minds14"
        it["e2e_intent"] = it["intent_label"]
        it["allowed_intents"] = m14_labels
        chosen_m14.append(it)

    all_turns = chosen_pn + chosen_qa + chosen_m14
    return all_turns, m14_labels


def _prepare_target_speakers() -> List[Dict[str, Any]]:
    """Loads 12 unseen LibriSpeech enroll/val speaker pairs cleanly in memory with CAM++ + ECAPA embeddings."""
    items = load_manifest("librispeech_60")["items"]
    by_spk: Dict[str, Dict[str, Any]] = {}
    for it in items:
        by_spk.setdefault(it["speaker_id"], {})[it["split"]] = it

    refs = []
    for spk in sorted(by_spk.keys()):
        enr_abs = os.path.join(ROOT_DIR, by_spk[spk]["enroll"]["audio_path"])
        val_abs = os.path.join(ROOT_DIR, by_spk[spk]["val"]["audio_path"])
        prof = VoiceCloner.extract_acoustic_profile_from_wav(
            enr_abs, profile_name=f"e2e_{spk}", profile_id=f"e2e_{spk}", save_ref_dir=None
        )
        prof.ref_audio_path = enr_abs
        val_16k, sr = VoiceCloner.load_audio_signal(val_abs, target_sr=16000)
        val_cam, _ = VoiceCloner.extract_speaker_embedding(val_16k, sr, use_neural=True, model_path=VoiceCloner.CAMPP_SPEAKER_ENCODER_PATH)
        val_eca, _ = VoiceCloner.extract_speaker_embedding(val_16k, sr, use_neural=True, model_path=VoiceCloner.ECAPA_SPEAKER_ENCODER_PATH)
        refs.append({
            "speaker_id": spk,
            "profile": prof,
            "val_emb": list(val_cam),
            "val_ecapa": list(val_eca),
        })
    return refs


def _run_asr_stage(asr_cfg_name: str, turns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cfg = json.load(open(os.path.join(ROOT_DIR, "ml", "configs", f"{asr_cfg_name}.json"), "r", encoding="utf-8"))
    _WHISPER_CACHE.clear()
    if cfg.get("provider") == "parakeet_tdt":
        asr = ParakeetTDTASR(decoding_method="greedy_search", phonetic_mode=cfg.get("phonetic_mode", "g2p"))
        asr._get_recognizer()
    else:
        asr = WhisperTransformerASR(
            model_size=cfg["model_size"],
            device=cfg["device"],
            compute_type=cfg["compute_type"],
            vad_filter=bool(cfg["vad_filter"]),
            beam_size=int(cfg["beam_size"]),
            initial_prompt="" if cfg["prompt_mode"] == "none" else None,
            word_timestamps=bool(cfg.get("word_timestamps", False)),
            repair_confidence_threshold=float(cfg.get("repair_confidence_threshold", 0.85)),
            phonetic_mode=cfg.get("phonetic_mode", "g2p"),
            condition_on_previous_text=bool(cfg.get("condition_on_previous_text", True)),
            repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(cfg.get("no_repeat_ngram_size", 0)),
            hallucination_silence_threshold=cfg.get("hallucination_silence_threshold"),
        )
        asr._lazy_init()

    iv_meta = _build_domain_overrides(cfg, "interview")
    m14_meta = _build_domain_overrides(cfg, "minds14")

    outputs: Dict[str, Dict[str, Any]] = {}
    for it in turns:
        meta = iv_meta if it["domain"] == "interview" else m14_meta
        abs_wav = os.path.join(ROOT_DIR, it["audio_path"])
        t0 = time.perf_counter()
        res = asr.transcribe(AudioPayload(source_type="file_upload", file_path=abs_wav, metadata=dict(meta)))
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        hyp = res.transcript or ""
        wer = _sample_wer(it["text"], hyp)
        t_terms = it.get("target_terms")
        r_ent = _sample_term_recall(hyp, t_terms) if t_terms else None
        asr_ok = (r_ent >= 0.8) if r_ent is not None else (wer <= 0.25)

        outputs[it["id"]] = {
            "transcript": hyp,
            "words_meta": res.metadata.get("words") or [],
            "wer": wer,
            "r_entity": r_ent,
            "asr_ok": bool(asr_ok),
            "asr_latency_ms": lat_ms,
        }
    return outputs


def _concise_reply(nlp_res: Any, dlg_mgr: ModularDialogueManager, domain: str) -> str:
    """Produces natural >=55 char spoken reply (~3-5s) so acoustic prosody isn't starved by ultra-short fragments."""
    dlg = dlg_mgr.process_turn(nlp_res)
    txt = dlg.response_text.strip()
    if domain == "minds14" and dlg.dialogue_action == "general_reply" and nlp_res.intent != "general_query":
        txt = f"I can assist you right away with your banking {nlp_res.intent.replace('_', ' ')} request."
    if len(txt) < 50:
        txt = f"{txt} Please let me know what else you would like to configure."
    if "." in txt[45:145]:
        txt = txt[: txt.index(".", 45) + 1]
    elif len(txt) > 140:
        txt = txt[:136].rstrip() + "."
    return txt


def _plot_pareto_and_breakdown(matrix_rows: List[Dict[str, Any]], out_fig: str) -> str:
    """Renders 2-panel Cost vs Composite Quality Q (true non-dominated frontier) + Per-Layer Gate Breakdown."""
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=300)

    # Panel A: Marginal Cost ($/1k turns) vs Primary Graded Composite Quality Q
    xs = [r["cost_per_1k_turns_usd"] for r in matrix_rows]
    ys = [r["e2e_composite_score"]["mean"] for r in matrix_rows]
    yerrs = [r["e2e_composite_score"]["std"] for r in matrix_rows]
    strict_ys = [r["e2e_task_success_pct"]["mean"] for r in matrix_rows]
    labels = [r["id"] for r in matrix_rows]

    colors = []
    for lb in labels:
        if lb == "Full (A+B+C)":
            colors.append("#27ae60")
        elif "Edge" in lb:
            colors.append("#2980b9")
        elif lb == "Baseline":
            colors.append("#c0392b")
        else:
            colors.append("#7f8c8d")

    for idx, (x, y, ye, sy, lb, col) in enumerate(zip(xs, ys, yerrs, strict_ys, labels, colors)):
        axes[0].errorbar(x, y, yerr=ye, fmt="o", markersize=8.0, color=col, ecolor=col, capsize=3.5, elinewidth=1.2)
        dy = 0.022 if idx % 2 == 0 else -0.042
        dx = 0.0015 if x < 0.07 else -0.012
        axes[0].annotate(
            f"{lb}\nQ={y:.3f} (E2E {sy:.0f}%)",
            (x, y),
            xytext=(x + dx, y + dy),
            fontsize=7.3,
            fontweight="bold" if col != "#7f8c8d" else "normal",
            arrowprops={"arrowstyle": "-", "color": "#aaaaaa", "lw": 0.6},
        )

    # Strictly mathematically non-dominated Pareto Frontier on (min Cost, max Composite Q)
    sorted_pts = sorted(zip(xs, ys, labels), key=lambda p: (p[0], -p[1]))
    pareto_x, pareto_y, pareto_labels = [], [], []
    best_y = -1.0
    for px, py, plb in sorted_pts:
        if py > best_y + 1e-6:
            best_y = py
            pareto_x.append(px)
            pareto_y.append(py)
            pareto_labels.append(plb)
    axes[0].plot(pareto_x, pareto_y, "--", color="#16a085", linewidth=1.8, label=f"Empirical Pareto ({', '.join(pareto_labels)})")
    axes[0].set_title("Composite Quality Q vs. Compute Cost ($ / 1k Turns)", fontsize=10.2)
    axes[0].set_xlabel("Estimated Compute Cost ($ / 1,000 Turns)", fontsize=9.2)
    axes[0].set_ylabel("Graded Composite Quality Q [0–1]", fontsize=9.2)
    axes[0].set_ylim(0.32, 0.95)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=7.6)

    # Panel B: Per-Stage Gate Pass Rates + Conjunctive Strict E2E across all 9 rows
    key_order = ["Baseline", "+A", "+B", "+C", "Full - A", "Full - B", "Full - C", "Edge (A_tiny+B+C)", "Full (A+B+C)"]
    row_map = {r["id"]: r for r in matrix_rows}
    ordered_rows = [row_map[k] for k in key_order if k in row_map]
    x_pos = np.arange(len(ordered_rows))
    w = 0.21
    a_pass = [r["stage_a_pass_pct"] for r in ordered_rows]
    b_pass = [r["stage_b_pass_pct"] for r in ordered_rows]
    c_pass = [r["stage_c_pass_pct"] for r in ordered_rows]
    e2e_pass = [r["e2e_task_success_pct"]["mean"] for r in ordered_rows]

    axes[1].bar(x_pos - 1.5 * w, a_pass, width=w, label="Stage A (ASR)", color="#3498db")
    axes[1].bar(x_pos - 0.5 * w, b_pass, width=w, label="Stage B (SLU+JSON)", color="#9b59b6")
    axes[1].bar(x_pos + 0.5 * w, c_pass, width=w, label="Stage C (TTS+CAM++)", color="#e67e22")
    axes[1].bar(x_pos + 1.5 * w, e2e_pass, width=w, label="Conjunctive E2E", color="#27ae60", edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([r["id"] for r in ordered_rows], rotation=26, ha="right", fontsize=8.0)
    axes[1].set_ylabel("Gate Pass Rate (%)", fontsize=9.2)
    axes[1].set_ylim(0, 114)
    axes[1].set_title("Per-Layer Gate Pass Rates & Conjunctive E2E Task Success", fontsize=10.2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=7.8, loc="upper left", ncol=2)

    plt.suptitle("End-to-End 6-Stage Ablation Matrix, Composite Q Frontier & Per-Stage Breakdown", fontsize=11.0)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    return out_fig


def _atomic_json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_e2e_matrix(resume: bool = True) -> Dict[str, Any]:
    cfg_matrix = json.load(open(os.path.join(ROOT_DIR, "ml", "configs", "e2e_ablation_matrix.json"), "r", encoding="utf-8"))
    seeds = cfg_matrix["seeds"]
    turns, m14_labels = _curate_e2e_turns()
    speakers = _prepare_target_speakers()
    for sp in speakers:
        voice_cloner.profiles[sp["profile"].profile_id] = sp["profile"]

    llm_gateway.configure(provider_id="ollama", fallback_enabled=False)

    asr_stats = json.load(open(os.path.join(ROOT_DIR, "ml", "results", "asr_ablation.json"), "r", encoding="utf-8"))
    slu_stats = json.load(open(os.path.join(ROOT_DIR, "ml", "results", "slu_ablation.json"), "r", encoding="utf-8"))
    tts_stats = json.load(open(os.path.join(ROOT_DIR, "ml", "results", "tts_ablation.json"), "r", encoding="utf-8"))

    ckpt_dir = os.path.join(ROOT_DIR, "ml", "results", "checkpoints", "e2e")
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="e2e_matrix_")
    try:
        asr_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        unique_asrs = list(dict.fromkeys(r["asr"] for r in cfg_matrix["rows"]))
        for acfg in unique_asrs:
            a_ckpt = os.path.join(ckpt_dir, f"asr_{acfg}.json")
            if resume and os.path.exists(a_ckpt):
                loaded_a = json.load(open(a_ckpt, "r", encoding="utf-8"))
                if len(loaded_a) >= len(turns):
                    asr_cache[acfg] = loaded_a
                    print(f"[Stage 2 ASR] Resumed {acfg} ({len(loaded_a)} turns) from checkpoint", flush=True)
                    continue
            print(f"\n[Stage 2 ASR] Running {acfg} on {len(turns)} spoken turns...", flush=True)
            asr_cache[acfg] = _run_asr_stage(acfg, turns)
            _atomic_json_dump(asr_cache[acfg], a_ckpt)

        slu_cache: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]] = {}
        unique_slu_pairs = list(dict.fromkeys((r["asr"], r["slu"]) for r in cfg_matrix["rows"]))
        dlg_mgr = ModularDialogueManager(bot_name="Mira")

        for acfg, scfg_name in unique_slu_pairs:
            scfg = json.load(open(os.path.join(ROOT_DIR, "ml", "configs", f"{scfg_name}.json"), "r", encoding="utf-8"))
            all_cached = all(
                resume
                and os.path.exists(os.path.join(ckpt_dir, f"slu_{acfg}__{scfg_name}__s{sd}.json"))
                for sd in seeds
            )
            if not all_cached:
                print(f"\n[Stage 3/4 SLU] Running ({acfg} -> {scfg_name}) across seeds {seeds}...", flush=True)
                if scfg["provider"] == "ollama":
                    LLMNLPService(
                        model_name=scfg["model_name"],
                        use_json_mode=bool(scfg["use_json_mode"]),
                        use_schema_mode=bool(scfg.get("use_schema_mode", False)),
                        temperature=0.0,
                    ).parse("warmup")

            for seed in seeds:
                s_ckpt = os.path.join(ckpt_dir, f"slu_{acfg}__{scfg_name}__s{seed}.json")
                if resume and os.path.exists(s_ckpt):
                    loaded_s = json.load(open(s_ckpt, "r", encoding="utf-8"))
                    if len(loaded_s) >= len(turns):
                        slu_cache[(acfg, scfg_name, seed)] = loaded_s
                        print(f"  [SLU ckpt] Resumed ({acfg} -> {scfg_name}, s={seed})", flush=True)
                        continue
                pair_out: Dict[str, Dict[str, Any]] = {}
                for idx_t, it in enumerate(turns, 1):
                    ar = asr_cache[acfg][it["id"]]
                    transcript = ar["transcript"]
                    domain = it["domain"]
                    lex = (INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES) if domain == "interview" else MINDS14_BANKING_TERMS
                    if scfg["provider"] == "distilbert_lora" and domain == "minds14":
                        nlp_svc = TrainedDistilBertSLUService(
                            model_dir=scfg["model_name"],
                            allowed_intents=it["allowed_intents"],
                            repair_lexicon=lex if scfg.get("asr_pre_repair") else None,
                        )
                        t0 = time.perf_counter()
                        res = nlp_svc.parse(transcript, words_meta=ar.get("words_meta"))
                    elif scfg["provider"] == "hybrid_cascade":
                        fast_s = TrainedDistilBertSLUService(
                            model_dir=scfg["model_name"],
                            allowed_intents=it["allowed_intents"],
                            repair_lexicon=lex if scfg.get("asr_pre_repair") else None,
                        )
                        slow_s = LLMNLPService(
                            model_name=scfg.get("slow_model_name", "qwen3:4b"),
                            allowed_intents=it["allowed_intents"],
                            repair_lexicon=lex if scfg.get("asr_pre_repair") else None,
                            enable_schema_repair=bool(scfg.get("enable_schema_repair", True)),
                            enable_repair_retry=bool(scfg.get("enable_repair_retry", True)),
                            use_json_mode=bool(scfg.get("use_json_mode", True)),
                            use_schema_mode=bool(scfg.get("use_schema_mode", True)),
                            use_logprobs=bool(scfg.get("use_logprobs", False)),
                            calibrated_prompt=bool(scfg.get("calibrated_prompt", True)),
                            temperature=float(scfg.get("temperature", 0.1)),
                            seed=seed,
                        )
                        nlp_svc = HybridConfidenceSLUService(
                            fast_service=fast_s,
                            slow_service=slow_s,
                            confidence_threshold=float(scfg.get("confidence_threshold", 0.85)),
                            margin_threshold=float(scfg.get("margin_threshold", 0.25)),
                            escalate_on_oos=bool(scfg.get("escalate_on_oos", True)),
                        )
                        t0 = time.perf_counter()
                        res = nlp_svc.parse_with_asr_context(transcript, words_meta=ar.get("words_meta"))
                    else:
                        model_fallback = scfg["model_name"] if scfg["provider"] == "ollama" else "qwen3:4b"
                        nlp_svc = LLMNLPService(
                            model_name=model_fallback,
                            allowed_intents=it["allowed_intents"],
                            repair_lexicon=lex if scfg.get("asr_pre_repair") else None,
                            enable_schema_repair=bool(scfg.get("enable_schema_repair", True)),
                            enable_repair_retry=bool(scfg.get("enable_repair_retry", True)),
                            use_json_mode=bool(scfg.get("use_json_mode", True)),
                            use_schema_mode=bool(scfg.get("use_schema_mode", False)),
                            use_logprobs=bool(scfg.get("use_logprobs", False)),
                            calibrated_prompt=bool(scfg.get("calibrated_prompt", False)),
                            temperature=float(scfg.get("temperature", 0.1)),
                            seed=seed,
                        )
                        t0 = time.perf_counter()
                        wmeta = ar.get("words_meta") if scfg.get("calibrated_prompt") else None
                        res = nlp_svc.parse_with_asr_context(transcript, words_meta=wmeta) if wmeta else nlp_svc.parse(transcript)
                    slu_ms = round((time.perf_counter() - t0) * 1000.0, 1)

                    meta = res.metadata or {}
                    sch_ok = bool(meta.get("schema_valid", False))
                    pred_intent = (res.intent or "").strip().lower().replace(" ", "_")
                    if "rule_based_fallback" in (res.provider or "").lower():
                        pred_intent = "__fallback__"
                    slu_ok = sch_ok and (pred_intent == it["e2e_intent"])

                    reply_txt = _concise_reply(res, dlg_mgr, domain)
                    pair_out[it["id"]] = {
                        "predicted_intent": pred_intent,
                        "schema_valid": sch_ok,
                        "slu_ok": bool(slu_ok),
                        "slu_latency_ms": slu_ms,
                        "reply_text": reply_txt,
                        "asr_pre_repairs": meta.get("asr_repairs") or [],
                    }
                    if idx_t % 18 == 0 or idx_t == len(turns):
                        print(f"    [SLU {acfg}->{scfg_name} s={seed}] {idx_t}/{len(turns)}", flush=True)
                slu_cache[(acfg, scfg_name, seed)] = pair_out
                _atomic_json_dump(pair_out, s_ckpt)

        tts_cache_path = os.path.join(ckpt_dir, "tts_cache.json")
        tts_cache_str: Dict[str, Dict[str, Any]] = json.load(open(tts_cache_path, "r", encoding="utf-8")) if (resume and os.path.exists(tts_cache_path)) else {}
        tts_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        unique_tts_jobs = set()
        for row in cfg_matrix["rows"]:
            tcfg = row["tts"]
            for seed in seeds:
                slu_recs = slu_cache[(row["asr"], row["slu"], seed)]
                for idx, it in enumerate(turns):
                    spk = speakers[idx % len(speakers)]
                    reply = slu_recs[it["id"]]["reply_text"]
                    unique_tts_jobs.add((tcfg, spk["profile"].profile_id, reply))

        pending_tts = []
        for job in sorted(unique_tts_jobs):
            jk = f"{job[0]}|||{job[1]}|||{job[2]}"
            if jk in tts_cache_str:
                tts_cache[job] = tts_cache_str[jk]
            else:
                pending_tts.append(job)

        print(f"\n[Stage 5 TTS] {len(pending_tts)}/{len(unique_tts_jobs)} pending TTS+DualEncoder/WER jobs (cached={len(tts_cache)})...", flush=True)
        if pending_tts:
            asr_judge = WhisperTransformerASR(model_size="tiny.en", compute_type="int8", vad_filter=False, initial_prompt="")
            asr_judge._lazy_init()

            tts_engines = {
                "tts_piper_dual_wsola": NeuralVoiceCloningTTSService(with_conditioning=True),
                "tts_pocket_clone": PocketTTSCloningService(num_steps=5, speed=1.0),
            }
            tts_engines["tts_pocket_clone"].get_or_load_tts()

            spk_by_pid = {sp["profile"].profile_id: sp for sp in speakers}
            for job_idx, (tcfg, pid, reply) in enumerate(pending_tts, 1):
                svc = tts_engines[tcfg]
                wav_p = os.path.join(tmp_dir, f"tts_{job_idx}.wav")
                t0 = time.perf_counter()
                res = svc.synthesize(reply, output_path=wav_p, speaker_profile_id=pid)
                tts_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                ttfa_ms = float((res.metadata or {}).get("ttfa_ms") or tts_ms)

                cam_sim, ecapa_sim, rt_wer, tts_ok = 0.0, 0.0, 1.0, False
                if res.status == "success" and os.path.exists(wav_p):
                    wav_16k, sr = VoiceCloner.load_audio_signal(wav_p, target_sr=16000)
                    emb_cam, _ = VoiceCloner.extract_speaker_embedding(wav_16k, sample_rate=sr, use_neural=True, model_path=VoiceCloner.CAMPP_SPEAKER_ENCODER_PATH)
                    emb_eca, _ = VoiceCloner.extract_speaker_embedding(wav_16k, sample_rate=sr, use_neural=True, model_path=VoiceCloner.ECAPA_SPEAKER_ENCODER_PATH)
                    cam_sim = compute_speaker_similarity(emb_cam, spk_by_pid[pid]["val_emb"])
                    ecapa_sim = compute_speaker_similarity(emb_eca, spk_by_pid[pid]["val_ecapa"])
                    dur_s = max(0.1, len(wav_16k) / float(sr))
                    rt_res = asr_judge.transcribe(
                        AudioPayload(
                            source_type="stream_buffer", sample_rate=16000, duration_sec=dur_s,
                            raw_bytes=wav_16k.astype(np.float32).tobytes(), metadata={"audio_dtype": "float32"}
                        )
                    )
                    rt_wer = _sample_wer(reply, rt_res.transcript or "")
                    tts_ok = (cam_sim >= 0.50) and (rt_wer <= 0.35)
                    os.remove(wav_p)

                rec = {
                    "campp_sim": round(cam_sim, 4),
                    "ecapa_sim": round(ecapa_sim, 4),
                    "roundtrip_wer": round(rt_wer, 4),
                    "tts_ok": bool(tts_ok),
                    "tts_latency_ms": tts_ms,
                    "ttfa_ms": round(ttfa_ms, 1),
                }
                tts_cache[(tcfg, pid, reply)] = rec
                tts_cache_str[f"{tcfg}|||{pid}|||{reply}"] = rec
                if job_idx % 15 == 0 or job_idx == len(pending_tts):
                    _atomic_json_dump(tts_cache_str, tts_cache_path)
                    print(f"  [TTS batch] {job_idx}/{len(pending_tts)} synthesized & checkpointed", flush=True)

        matrix_rows: List[Dict[str, Any]] = []
        turn_avg_e2e_vectors: Dict[str, List[float]] = {}
        turn_avg_comp_vectors: Dict[str, List[float]] = {}
        concrete_records_by_row: Dict[str, List[Dict[str, Any]]] = {}

        for row in cfg_matrix["rows"]:
            rid = row["id"]
            acfg, scfg_name, tcfg = row["asr"], row["slu"], row["tts"]
            asr_out = asr_cache[acfg]

            seed_succ, seed_comp = [], []
            all_lats, all_ttfa = [], []
            a_passes, b_passes, c_passes = [], [], []
            turn_e2e_matrix: List[List[float]] = [[] for _ in turns]
            turn_comp_matrix: List[List[float]] = [[] for _ in turns]
            s42_recs: List[Dict[str, Any]] = []

            for seed in seeds:
                slu_out = slu_cache[(acfg, scfg_name, seed)]
                turn_succ, turn_comp = [], []

                for idx, it in enumerate(turns):
                    spk = speakers[idx % len(speakers)]
                    pid = spk["profile"].profile_id
                    ar = asr_out[it["id"]]
                    sr_rec = slu_out[it["id"]]
                    tr = tts_cache[(tcfg, pid, sr_rec["reply_text"])]

                    conj_ok = 1.0 if (ar["asr_ok"] and sr_rec["slu_ok"] and tr["tts_ok"]) else 0.0
                    comp_score = (
                        0.35 * max(0.0, 1.0 - min(ar["wer"], 1.0))
                        + 0.35 * (1.0 if sr_rec["slu_ok"] else 0.0)
                        + 0.30 * max(0.0, tr["campp_sim"])
                    )
                    e2e_lat_ms = ar["asr_latency_ms"] + sr_rec["slu_latency_ms"] + tr["tts_latency_ms"]
                    e2e_ttfa_ms = ar["asr_latency_ms"] + sr_rec["slu_latency_ms"] + tr["ttfa_ms"]

                    turn_succ.append(conj_ok)
                    turn_comp.append(round(comp_score, 4))
                    turn_e2e_matrix[idx].append(conj_ok)
                    turn_comp_matrix[idx].append(comp_score)
                    all_lats.append(e2e_lat_ms)
                    all_ttfa.append(e2e_ttfa_ms)
                    a_passes.append(1.0 if ar["asr_ok"] else 0.0)
                    b_passes.append(1.0 if sr_rec["slu_ok"] else 0.0)
                    c_passes.append(1.0 if tr["tts_ok"] else 0.0)

                    if seed == seeds[0]:
                        s42_recs.append({
                            "id": it["id"],
                            "domain": it["domain"],
                            "reference_text": it["text"],
                            "gold_intent": it["e2e_intent"],
                            "asr_transcript": ar["transcript"],
                            "asr_wer": ar["wer"],
                            "r_entity": ar["r_entity"],
                            "predicted_intent": sr_rec["predicted_intent"],
                            "schema_valid": sr_rec["schema_valid"],
                            "reply_text": sr_rec["reply_text"],
                            "tts_campp_sim": tr["campp_sim"],
                            "tts_ecapa_sim": tr["ecapa_sim"],
                            "tts_roundtrip_wer": tr["roundtrip_wer"],
                            "e2e_success": bool(conj_ok),
                            "composite_score": round(comp_score, 4),
                        })

                seed_succ.append(round(100.0 * float(np.mean(turn_succ)), 2))
                seed_comp.append(round(float(np.mean(turn_comp)), 4))
                if seed == seeds[0]:
                    concrete_records_by_row[rid] = s42_recs

            # Aggregate per turn across 3 seeds strictly to avoid N x seed pseudoreplication!
            turn_avg_e2e_vectors[rid] = [float(np.mean(v)) for v in turn_e2e_matrix]
            turn_avg_comp_vectors[rid] = [float(np.mean(v)) for v in turn_comp_matrix]

            asr_ram_mb = float(asr_stats["configs"][acfg]["peak_rss_mb"])
            tts_ram_mb = float(tts_stats["configs"][tcfg]["peak_rss_mb"])
            slu_vram_mb = round(float(slu_stats["configs"][scfg_name]["model_vram_gb"]) * 1024.0, 1)
            est_peak_mem_mb = round(max(asr_ram_mb, tts_ram_mb) + 0.5 * min(asr_ram_mb, tts_ram_mb) + slu_vram_mb, 1)

            avg_e2e_ms = round(float(np.mean(all_lats)), 1)
            p95_e2e_ms = round(float(np.percentile(all_lats, 95)), 1)
            avg_ttfa_ms = round(float(np.mean(all_ttfa)), 1)
            cost_1k = estimate_cost_per_1k_turns(avg_e2e_ms / 1000.0, compute_hourly_usd=0.05)

            row_summary = {
                "id": rid,
                "asr": acfg,
                "slu": scfg_name,
                "tts": tcfg,
                "e2e_composite_score": summarize_seeds(seed_comp),
                "stage_a_pass_pct": round(100.0 * float(np.mean(a_passes)), 2),
                "stage_b_pass_pct": round(100.0 * float(np.mean(b_passes)), 2),
                "stage_c_pass_pct": round(100.0 * float(np.mean(c_passes)), 2),
                "e2e_task_success_pct": summarize_seeds(seed_succ),
                "e2e_latency_ms": avg_e2e_ms,
                "e2e_ttfa_ms": avg_ttfa_ms,
                "p95_latency_ms": p95_e2e_ms,
                "est_peak_memory_mb": est_peak_mem_mb,
                "cost_per_1k_turns_usd": cost_1k,
            }
            matrix_rows.append(row_summary)
            print(
                f"  [{rid:18s}] Q={row_summary['e2e_composite_score']['mean']:.3f} | "
                f"A={row_summary['stage_a_pass_pct']:5.1f} B={row_summary['stage_b_pass_pct']:5.1f} C={row_summary['stage_c_pass_pct']:5.1f} | "
                f"Strict={row_summary['e2e_task_success_pct']['mean']:5.1f}% | Lat={avg_e2e_ms}ms TTFA={avg_ttfa_ms}ms | Cost=${cost_1k}",
                flush=True,
            )

        # Turn-level (N=len(turns)) paired bootstrap (B=2000) & TOST non-inferiority (margin=-0.05)
        p_tests = {}
        tost_tests = {}
        for name, base_id, cand_id in [
            ("full_vs_baseline_e2e_success", "Baseline", "Full (A+B+C)"),
            ("full_vs_full_minus_a_e2e_success", "Full - A", "Full (A+B+C)"),
            ("full_vs_full_minus_b_e2e_success", "Full - B", "Full (A+B+C)"),
            ("full_vs_full_minus_c_e2e_success", "Full - C", "Full (A+B+C)"),
            ("edge_vs_baseline_e2e_success", "Baseline", "Edge (A_tiny+B+C)"),
            ("full_vs_edge_e2e_success", "Edge (A_tiny+B+C)", "Full (A+B+C)"),
        ]:
            p_tests[name] = paired_bootstrap_test(
                turn_avg_e2e_vectors[base_id], turn_avg_e2e_vectors[cand_id], n_resamples=2000, seed=42, higher_is_better=True
            )
        for name, base_id, cand_id in [
            ("full_vs_baseline_composite", "Baseline", "Full (A+B+C)"),
            ("full_vs_full_minus_a_composite", "Full - A", "Full (A+B+C)"),
            ("edge_vs_baseline_composite", "Baseline", "Edge (A_tiny+B+C)"),
            ("edge_vs_full_composite", "Full (A+B+C)", "Edge (A_tiny+B+C)"),
        ]:
            p_tests[name] = paired_bootstrap_test(
                turn_avg_comp_vectors[base_id], turn_avg_comp_vectors[cand_id], n_resamples=2000, seed=42, higher_is_better=True
            )
            if "edge_vs_full" in name or "full_vs_full_minus_a" in name:
                tost_tests[name] = tost_non_inferiority(
                    turn_avg_comp_vectors[base_id], turn_avg_comp_vectors[cand_id], margin=-0.05, n_resamples=2000, seed=42, higher_is_better=True
                )

        fig4_path = _plot_pareto_and_breakdown(
            matrix_rows,
            os.path.join(ROOT_DIR, "ml", "results", "figures", "fig4_e2e_pareto_front.png"),
        )
        print(f"Generated Pareto plot -> {fig4_path}", flush=True)

        full_failures = [r for r in concrete_records_by_row["Full (A+B+C)"] if not r["e2e_success"]]
        base_failures = [r for r in concrete_records_by_row["Baseline"] if not r["e2e_success"]][:5]

        payload = {
            "experiment": "session8_e2e_ablation_matrix_v2",
            "num_turns": len(turns),
            "seeds": seeds,
            "bootstrap_unit": f"turn_level_mean (N={len(turns)}, B=2000)",
            "pareto_figure": os.path.relpath(fig4_path, ROOT_DIR).replace("\\", "/"),
            "matrix": matrix_rows,
            "paired_bootstrap": p_tests,
            "tost_non_inferiority": tost_tests,
            "full_residual_failures": full_failures,
            "baseline_sample_failures": base_failures,
        }
        out_path = os.path.join(ROOT_DIR, "ml", "results", "e2e_ablation_matrix.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved E2E v2 Ablation Matrix -> {out_path}", flush=True)
        return payload
    finally:
        llm_gateway.configure(provider_id="ollama", fallback_enabled=True)
        for sp in speakers:
            voice_cloner.profiles.pop(sp["profile"].profile_id, None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    run_e2e_matrix(resume=not a.no_resume)
