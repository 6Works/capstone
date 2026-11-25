"""
Session 8 Experiment: End-to-End 6-Stage Ablation Matrix & Accuracy-vs-Cost Pareto Front.
Executes real Audio -> Stage 2 ASR -> Stage 3 SLU -> Stage 4 Dialogue -> Stage 5 TTS -> Verification
across 9 rows (Baseline, +A, +B, +C, Full A+B+C, Full-A, Full-B, Full-C, Edge A_tiny+B+C)
over 3 random seeds ([42, 1337, 2026]) on 24 multi-domain spoken turns per seed (72 turn evals/row).

Saves full metrics, per-stage gate pass rates, paired bootstrap p-values, and concrete E2E failure cases
to ml/results/e2e_ablation_matrix.json and renders ml/results/figures/fig4_e2e_pareto_front.png.
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

from services.asr import WhisperTransformerASR
from services.asr.asr_services import _WHISPER_CACHE
from services.nlp import LLMNLPService
from services.dialogue import ModularDialogueManager
from services.tts import NeuralVoiceCloningTTSService, PocketTTSCloningService, VoiceCloner, voice_cloner
from services.llm.gateway import llm_gateway
from services.contracts import AudioPayload
from ml.datasets import load_manifest
from ml.metrics import (
    compute_asr_metrics,
    compute_speaker_similarity,
    paired_bootstrap_test,
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
    """Selects 24 balanced multi-domain spoken test turns (4 proper_nouns + 10 QA + 10 MINDS-14)."""
    iv_test = [x for x in load_manifest("interview_stress_v1")["items"] if x["split"] == "test"]
    pn_items = [x for x in iv_test if x.get("subset") == "proper_nouns"]
    qa_items = [x for x in iv_test if x.get("subset") == "interview_qa"]

    # 4 proper_nouns spans distinct tech/South-Asian entities; true intent for engineering statements is 'answer'
    chosen_pn = [pn_items[0], pn_items[3], pn_items[5], pn_items[13]]  # pn_05, pn_08, pn_10, pn_18
    for it in chosen_pn:
        it["domain"] = "interview"
        it["e2e_intent"] = "answer"
        it["allowed_intents"] = INTERVIEW_INTENTS

    # 10 interview_qa spans all 7 interview SLU intents
    by_qa_int: Dict[str, List[Dict[str, Any]]] = {}
    for it in qa_items:
        by_qa_int.setdefault(it["intent_label"], []).append(it)
    chosen_qa: List[Dict[str, Any]] = []
    for k in ("greeting", "identity_query", "project_query", "calculation_query", "time_query", "farewell"):
        if by_qa_int.get(k):
            chosen_qa.append(by_qa_int[k][0])
    # Fill to 10 with diverse project/calc/answer turns
    for it in qa_items:
        if len(chosen_qa) >= 10:
            break
        if it not in chosen_qa:
            chosen_qa.append(it)
    for it in chosen_qa:
        it["domain"] = "interview"
        it["e2e_intent"] = it["intent_label"]
        it["allowed_intents"] = INTERVIEW_INTENTS

    # 10 MINDS-14 banking test turns across 10 distinct classes & 3 accents (en-US/GB/AU)
    m14_test = [x for x in load_manifest("minds14_en_140")["items"] if x["split"] == "test"]
    by_m14: Dict[str, List[Dict[str, Any]]] = {}
    for it in m14_test:
        by_m14.setdefault(it["intent_label"], []).append(it)
    m14_labels = sorted(by_m14.keys())
    chosen_m14: List[Dict[str, Any]] = []
    for lbl in m14_labels[:10]:
        it = dict(by_m14[lbl][0])
        it["domain"] = "minds14"
        it["e2e_intent"] = it["intent_label"]
        it["allowed_intents"] = m14_labels
        chosen_m14.append(it)

    all_turns = chosen_pn + chosen_qa + chosen_m14
    return all_turns, m14_labels


def _prepare_target_speakers() -> List[Dict[str, Any]]:
    """Loads 12 unseen LibriSpeech enroll/val speaker pairs cleanly in memory."""
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
        val_prof = VoiceCloner.extract_acoustic_profile_from_wav(val_abs, profile_name=f"val_{spk}", save_ref_dir=None)
        refs.append({
            "speaker_id": spk,
            "profile": prof,
            "val_emb": list(val_prof.embedding_vector),
        })
    return refs


def _run_asr_stage(asr_cfg_name: str, turns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cfg = json.load(open(os.path.join(ROOT_DIR, "ml", "configs", f"{asr_cfg_name}.json"), "r", encoding="utf-8"))
    _WHISPER_CACHE.clear()
    asr = WhisperTransformerASR(
        model_size=cfg["model_size"],
        device=cfg["device"],
        compute_type=cfg["compute_type"],
        vad_filter=bool(cfg["vad_filter"]),
        beam_size=int(cfg["beam_size"]),
        initial_prompt="" if cfg["prompt_mode"] == "none" else None,
        word_timestamps=bool(cfg.get("word_timestamps", False)),
        repair_confidence_threshold=float(cfg.get("repair_confidence_threshold", 0.85)),
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
            "wer": wer,
            "r_entity": r_ent,
            "asr_ok": bool(asr_ok),
            "asr_latency_ms": lat_ms,
        }
    return outputs


def _concise_reply(nlp_res: Any, dlg_mgr: ModularDialogueManager, domain: str) -> str:
    """Produces concise spoken response (<=125 chars) via ModularDialogueManager.process_turn."""
    dlg = dlg_mgr.process_turn(nlp_res)
    txt = dlg.response_text.strip()
    if domain == "minds14" and dlg.dialogue_action == "general_reply" and nlp_res.intent != "general_query":
        txt = f"I can assist you with your {nlp_res.intent.replace('_', ' ')} request."
    # Keep first sentence cleanly to measure steady-state turn synthesis
    if "." in txt[:130]:
        txt = txt[: txt.index(".") + 1]
    elif len(txt) > 125:
        txt = txt[:122].rstrip() + "."
    return txt


def _plot_pareto_and_breakdown(matrix_rows: List[Dict[str, Any]], out_fig: str) -> str:
    """Renders 2-panel E2E Pareto Frontier (Task Success vs Cost/1k turns) + Component Contribution Bar Chart."""
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), dpi=300)

    # Panel A: Cost ($/1k turns) vs Strict E2E Task Success (%)
    xs = [r["cost_per_1k_turns_usd"] for r in matrix_rows]
    ys = [r["e2e_task_success_pct"]["mean"] for r in matrix_rows]
    yerrs = [r["e2e_task_success_pct"]["std"] for r in matrix_rows]
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

    offsets = {
        "Baseline": (-0.002, 5.5),
        "+B": (0.002, 4.5),
        "+C": (0.0025, 3.0),
        "Full - A": (-0.006, 6.0),
        "Edge (A_tiny+B+C)": (0.003, -8.5),
        "+A": (-0.004, 5.0),
        "Full - C": (-0.011, 14.0),
        "Full - B": (-0.002, 13.0),
        "Full (A+B+C)": (-0.014, 6.0),
    }
    for x, y, ye, lb, col in zip(xs, ys, yerrs, labels, colors):
        axes[0].errorbar(x, y, yerr=ye, fmt="o", markersize=8.5, color=col, ecolor=col, capsize=3.5, elinewidth=1.2)
        dx, dy = offsets.get(lb, (0.002, 3.0))
        axes[0].annotate(
            f"{lb}\n({y:.1f}%)",
            (x, y),
            xytext=(x + dx, y + dy),
            fontsize=7.6,
            fontweight="bold" if col != "#7f8c8d" else "normal",
            arrowprops={"arrowstyle": "-", "color": "#aaaaaa", "lw": 0.6} if abs(dy) > 4.5 else None,
        )

    # Identify and plot Pareto Frontier (non-dominated ascending cost -> higher success)
    sorted_pts = sorted(zip(xs, ys, labels), key=lambda p: (p[0], -p[1]))
    pareto_x, pareto_y = [], []
    best_y = -1.0
    for px, py, _ in sorted_pts:
        if py > best_y:
            best_y = py
            pareto_x.append(px)
            pareto_y.append(py)
    axes[0].plot(pareto_x, pareto_y, "--", color="#16a085", linewidth=1.8, label="Pareto Frontier")
    axes[0].set_title("E2E Task Success (%) vs. Marginal Cost ($ / 1k Turns)", fontsize=10.5)
    axes[0].set_xlabel("Estimated Compute Cost ($ / 1,000 Turns)", fontsize=9.2)
    axes[0].set_ylabel("Strict E2E Task Success (%)", fontsize=9.2)
    axes[0].set_xlim(0.012, 0.104)
    axes[0].set_ylim(-6, 98)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel B: Per-Stage Gate Pass Rates across key ablation rows
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
    axes[1].set_xticklabels([r["id"] for r in ordered_rows], rotation=26, ha="right", fontsize=8.2)
    axes[1].set_ylabel("Gate Pass Rate (%)", fontsize=9.2)
    axes[1].set_ylim(0, 112)
    axes[1].set_title("Per-Layer Gate Pass Rates & Conjunctive E2E Task Success", fontsize=10.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=7.8, loc="upper left", ncol=2)

    plt.suptitle("Session 8 End-to-End 6-Stage Ablation Matrix & Efficiency-Accuracy Pareto Analysis (3 Seeds)", fontsize=11.3)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)
    return out_fig


def run_e2e_matrix() -> Dict[str, Any]:
    cfg_matrix = json.load(open(os.path.join(ROOT_DIR, "ml", "configs", "e2e_ablation_matrix.json"), "r", encoding="utf-8"))
    seeds = cfg_matrix["seeds"]
    turns, m14_labels = _curate_e2e_turns()
    speakers = _prepare_target_speakers()
    for sp in speakers:
        voice_cloner.profiles[sp["profile"].profile_id] = sp["profile"]

    llm_gateway.configure(provider_id="ollama", fallback_enabled=False)

    # Read subsystem peak RSS telemetry already verified in Sessions 5-7 to compute exact E2E memory bounds
    asr_stats = json.load(open(os.path.join(ROOT_DIR, "ml", "results", "asr_ablation.json"), "r", encoding="utf-8"))
    slu_stats = json.load(open(os.path.join(ROOT_DIR, "ml", "results", "slu_ablation.json"), "r", encoding="utf-8"))
    tts_stats = json.load(open(os.path.join(ROOT_DIR, "ml", "results", "tts_ablation.json"), "r", encoding="utf-8"))

    tmp_dir = tempfile.mkdtemp(prefix="e2e_matrix_")
    try:
        # Step 1: Run Stage 2 ASR across unique ASR configurations
        asr_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        unique_asrs = list(dict.fromkeys(r["asr"] for r in cfg_matrix["rows"]))
        for acfg in unique_asrs:
            print(f"\n[Stage 2 ASR] Running {acfg} on {len(turns)} spoken turns...", flush=True)
            asr_cache[acfg] = _run_asr_stage(acfg, turns)

        # Step 2: Run Stage 3/4 SLU + Dialogue across unique (asr, slu, seed) keys
        slu_cache: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]] = {}
        unique_slu_pairs = list(dict.fromkeys((r["asr"], r["slu"]) for r in cfg_matrix["rows"]))
        dlg_mgr = ModularDialogueManager(bot_name="Mira")

        for acfg, scfg_name in unique_slu_pairs:
            scfg = json.load(open(os.path.join(ROOT_DIR, "ml", "configs", f"{scfg_name}.json"), "r", encoding="utf-8"))
            print(f"\n[Stage 3/4 SLU] Running ({acfg} -> {scfg_name}) across seeds {seeds}...", flush=True)
            # Warm Ollama model once
            LLMNLPService(model_name=scfg["model_name"], use_json_mode=bool(scfg["use_json_mode"]), temperature=0.0).parse("warmup")

            for seed in seeds:
                pair_out: Dict[str, Dict[str, Any]] = {}
                for it in turns:
                    transcript = asr_cache[acfg][it["id"]]["transcript"]
                    domain = it["domain"]
                    lex = (INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES) if domain == "interview" else MINDS14_BANKING_TERMS
                    nlp_svc = LLMNLPService(
                        model_name=scfg["model_name"],
                        allowed_intents=it["allowed_intents"],
                        repair_lexicon=lex if scfg.get("asr_pre_repair") else None,
                        enable_schema_repair=bool(scfg["enable_schema_repair"]),
                        enable_repair_retry=bool(scfg["enable_repair_retry"]),
                        use_json_mode=bool(scfg["use_json_mode"]),
                        temperature=float(scfg["temperature"]),
                        seed=seed,
                    )
                    t0 = time.perf_counter()
                    res = nlp_svc.parse(transcript)
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
                slu_cache[(acfg, scfg_name, seed)] = pair_out

        # Step 3: Run Stage 5 TTS + Verification cleanly (deduplicated by (tts_cfg, speaker_id, reply_text))
        asr_judge = WhisperTransformerASR(model_size="tiny.en", compute_type="int8", vad_filter=False, initial_prompt="")
        asr_judge._lazy_init()

        tts_engines = {
            "tts_piper_dual_wsola": NeuralVoiceCloningTTSService(with_conditioning=True),
            "tts_pocket_clone": PocketTTSCloningService(num_steps=5, speed=1.0),
        }
        # Warm Pocket TTS & Piper
        tts_engines["tts_pocket_clone"].get_or_load_tts()

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

        print(f"\n[Stage 5 TTS] Synthesizing {len(unique_tts_jobs)} unique (engine, speaker, reply) waveforms + CAM++/WER...", flush=True)
        spk_by_pid = {sp["profile"].profile_id: sp for sp in speakers}
        for job_idx, (tcfg, pid, reply) in enumerate(sorted(unique_tts_jobs)):
            svc = tts_engines[tcfg]
            wav_p = os.path.join(tmp_dir, f"tts_{job_idx}.wav")
            t0 = time.perf_counter()
            res = svc.synthesize(reply, output_path=wav_p, speaker_profile_id=pid)
            tts_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            cam_sim = 0.0
            rt_wer = 1.0
            tts_ok = False
            if res.status == "success" and os.path.exists(wav_p):
                wav_16k, sr = VoiceCloner.load_audio_signal(wav_p, target_sr=16000)
                emb, _ = VoiceCloner.extract_speaker_embedding(wav_16k, sample_rate=sr, use_neural=True)
                cam_sim = compute_speaker_similarity(emb, spk_by_pid[pid]["val_emb"])
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

            tts_cache[(tcfg, pid, reply)] = {
                "campp_sim": round(cam_sim, 4),
                "roundtrip_wer": round(rt_wer, 4),
                "tts_ok": bool(tts_ok),
                "tts_latency_ms": tts_ms,
            }

        # Step 4: Assemble all 9 rows cleanly across 3 seeds
        matrix_rows: List[Dict[str, Any]] = []
        seed42_e2e_vectors: Dict[str, List[float]] = {}
        seed42_comp_vectors: Dict[str, List[float]] = {}
        concrete_records_by_row: Dict[str, List[Dict[str, Any]]] = {}

        for row in cfg_matrix["rows"]:
            rid = row["id"]
            acfg, scfg_name, tcfg = row["asr"], row["slu"], row["tts"]
            asr_out = asr_cache[acfg]

            seed_succ: List[float] = []
            seed_comp: List[float] = []
            all_lats: List[float] = []
            a_passes, b_passes, c_passes = [], [], []
            s42_recs: List[Dict[str, Any]] = []

            for seed in seeds:
                slu_out = slu_cache[(acfg, scfg_name, seed)]
                turn_succ: List[float] = []
                turn_comp: List[float] = []

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

                    turn_succ.append(conj_ok)
                    turn_comp.append(round(comp_score, 4))
                    all_lats.append(e2e_lat_ms)
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
                            "tts_roundtrip_wer": tr["roundtrip_wer"],
                            "e2e_success": bool(conj_ok),
                            "composite_score": round(comp_score, 4),
                        })

                seed_succ.append(round(100.0 * float(np.mean(turn_succ)), 2))
                seed_comp.append(round(float(np.mean(turn_comp)), 4))
                if seed == seeds[0]:
                    seed42_e2e_vectors[rid] = turn_succ
                    seed42_comp_vectors[rid] = turn_comp
                    concrete_records_by_row[rid] = s42_recs

            # Compute host RAM + VRAM footprint cleanly from subsystem measurements
            asr_ram_mb = float(asr_stats["configs"][acfg]["peak_rss_mb"])
            tts_ram_mb = float(tts_stats["configs"][tcfg]["peak_rss_mb"])
            slu_vram_mb = round(float(slu_stats["configs"][scfg_name]["model_vram_gb"]) * 1024.0, 1)
            est_peak_mem_mb = round(max(asr_ram_mb, tts_ram_mb) + 0.5 * min(asr_ram_mb, tts_ram_mb) + slu_vram_mb, 1)

            avg_e2e_ms = round(float(np.mean(all_lats)), 1)
            p95_e2e_ms = round(float(np.percentile(all_lats, 95)), 1)
            cost_1k = estimate_cost_per_1k_turns(avg_e2e_ms / 1000.0, compute_hourly_usd=0.05)

            row_summary = {
                "id": rid,
                "asr": acfg,
                "slu": scfg_name,
                "tts": tcfg,
                "e2e_task_success_pct": summarize_seeds(seed_succ),
                "e2e_composite_score": summarize_seeds(seed_comp),
                "stage_a_pass_pct": round(100.0 * float(np.mean(a_passes)), 2),
                "stage_b_pass_pct": round(100.0 * float(np.mean(b_passes)), 2),
                "stage_c_pass_pct": round(100.0 * float(np.mean(c_passes)), 2),
                "e2e_latency_ms": avg_e2e_ms,
                "p95_latency_ms": p95_e2e_ms,
                "est_peak_memory_mb": est_peak_mem_mb,
                "cost_per_1k_turns_usd": cost_1k,
            }
            matrix_rows.append(row_summary)
            print(
                f"  [{rid:18s}] E2E={row_summary['e2e_task_success_pct']['mean']:5.1f}+/-{row_summary['e2e_task_success_pct']['std']:4.1f}% "
                f"(A={row_summary['stage_a_pass_pct']:5.1f} B={row_summary['stage_b_pass_pct']:5.1f} C={row_summary['stage_c_pass_pct']:5.1f}) "
                f"| Comp={row_summary['e2e_composite_score']['mean']:.3f} | Lat={avg_e2e_ms}ms | Cost=${cost_1k}",
                flush=True,
            )

        # Paired bootstrap tests on strict E2E task success & composite score across Seed 42 items (N=24)
        p_tests = {}
        for name, base_id, cand_id in [
            ("full_vs_baseline_e2e_success", "Baseline", "Full (A+B+C)"),
            ("full_vs_full_minus_a_e2e_success", "Full - A", "Full (A+B+C)"),
            ("full_vs_full_minus_b_e2e_success", "Full - B", "Full (A+B+C)"),
            ("full_vs_full_minus_c_e2e_success", "Full - C", "Full (A+B+C)"),
            ("edge_vs_baseline_e2e_success", "Baseline", "Edge (A_tiny+B+C)"),
            ("full_vs_edge_e2e_success", "Edge (A_tiny+B+C)", "Full (A+B+C)"),
        ]:
            p_tests[name] = paired_bootstrap_test(
                seed42_e2e_vectors[base_id], seed42_e2e_vectors[cand_id], n_resamples=1000, seed=42, higher_is_better=True
            )
        for name, base_id, cand_id in [
            ("full_vs_baseline_composite", "Baseline", "Full (A+B+C)"),
            ("edge_vs_baseline_composite", "Baseline", "Edge (A_tiny+B+C)"),
        ]:
            p_tests[name] = paired_bootstrap_test(
                seed42_comp_vectors[base_id], seed42_comp_vectors[cand_id], n_resamples=1000, seed=42, higher_is_better=True
            )

        fig4_path = _plot_pareto_and_breakdown(
            matrix_rows,
            os.path.join(ROOT_DIR, "ml", "results", "figures", "fig4_e2e_pareto_front.png"),
        )
        print(f"Generated Pareto plot -> {fig4_path}", flush=True)

        # Collect residual failures in Full (A+B+C) & key baseline failures for Error Analysis
        full_failures = [r for r in concrete_records_by_row["Full (A+B+C)"] if not r["e2e_success"]]
        base_failures = [r for r in concrete_records_by_row["Baseline"] if not r["e2e_success"]][:5]

        payload = {
            "experiment": "session8_e2e_ablation_matrix_v1",
            "num_turns": len(turns),
            "seeds": seeds,
            "pareto_figure": os.path.relpath(fig4_path, ROOT_DIR).replace("\\", "/"),
            "matrix": matrix_rows,
            "paired_bootstrap": p_tests,
            "full_residual_failures": full_failures,
            "baseline_sample_failures": base_failures,
        }
        out_path = os.path.join(ROOT_DIR, "ml", "results", "e2e_ablation_matrix.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved E2E Ablation Matrix -> {out_path}", flush=True)
        return payload
    finally:
        llm_gateway.configure(provider_id="ollama", fallback_enabled=True)
        for sp in speakers:
            voice_cloner.profiles.pop(sp["profile"].profile_id, None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_e2e_matrix()
