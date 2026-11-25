"""
Session 5 Experiment: Layer A (Stage 2 ASR) Ablation & Robustness Evaluation.
Compares 4 reproducible Whisper configurations:
  1. asr_tiny_baseline  (tiny.en, vad=False, no prompt, no hotwords, no repair)
  2. asr_small_baseline (small.en, vad=True, static DEFAULT_PROMPT)
  3. asr_tiny_improved  (tiny.en + vad + capacity-gated prompt + loop guards + phonetic repair)
  4. asr_small_improved (small.en + vad + dynamic prompt + hotwords + phonetic repair)

Evaluates across:
  - interview_stress_v1 test (50 clean + 20 proper_nouns @ 20 dB AWGN + 20 @ 10 dB AWGN)
  - minds14_en_140 test (70 accented clips across en-US / en-GB / en-AU)
  - librispeech_60 test (36 clean + 12 unseen-speaker @ 20 dB + 12 @ 10 dB AWGN)

Saves metrics, paired bootstrap p-values, hardware efficiency, negative-result logs,
error cases, and real ASR transcripts (consumed downstream by Session 6 SLU)
to ml/results/asr_ablation.json.
"""

import os
import sys
import json
import time
import glob
import argparse
import subprocess
import ctypes
from ctypes import wintypes
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import soundfile as sf
import jiwer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.asr import WhisperTransformerASR, build_dynamic_bias_prompt
from services.asr.asr_services import _WHISPER_CACHE
from services.contracts import AudioPayload
from ml.datasets import load_manifest, add_awgn
from ml.metrics import compute_asr_metrics, paired_bootstrap_test, estimate_cost_per_1k_turns
from ml.metrics.eval_metrics import _normalize_text


# Fixed session-level Role Blueprint + Candidate Roster (non-oracle; shared identically across utterances)
INTERVIEW_ROLE_TERMS = [
    "FastAPI", "Redis", "PyTorch", "LoRA", "PostgreSQL", "CUDA", "ONNX", "gRPC", "Kafka",
    "CTranslate2", "Kubernetes", "OAuth", "TensorRT", "OpenVINO", "Silero", "Whisper",
    "SAPI5", "Llama", "ChromaDB", "asyncio", "WeSpeaker", "Docker", "pytest", "Nginx",
    "Prometheus", "Kyutai", "GraphQL", "Qwen", "SciPy", "WebSocket", "Python", "LibriSpeech",
    "SOLID", "Mira",
]
CANDIDATE_ROSTER_NAMES = [
    "Sachin", "Aradhya", "Nandu", "Bablu", "Rahul", "Priya", "Vikram", "Ananya",
    "Karthik", "Rohan", "Sneha", "Deepa", "Aditya", "Siddharth", "Krishnendu", "Meera",
    "Shreya", "Varun", "Gaurav", "Tanvi", "Harish", "Manish", "Pooja", "Sanjay",
    "Nikhil", "Swati",
]
MINDS14_BANKING_TERMS = [
    "abroad", "address", "error", "ATM", "limit", "balance", "business", "loan",
    "card", "cash", "deposit", "direct", "debit", "freeze", "payment", "joint",
    "account", "transactions", "bill",
]

CONFIG_NAMES = [
    "asr_tiny_baseline",
    "asr_small_baseline",
    "asr_tiny_improved",
    "asr_small_improved",
]


def _get_memory_mb() -> Tuple[float, float]:
    """Returns (current_rss_mb, peak_rss_mb) using 64-bit safe OS-native APIs."""
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return (
                round(pmc.WorkingSetSize / (1024.0 * 1024.0), 2),
                round(pmc.PeakWorkingSetSize / (1024.0 * 1024.0), 2),
            )
    else:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        return round(rss, 2), round(rss, 2)
    return 0.0, 0.0


def _probe_model_rss_mb(model_size: str) -> Tuple[float, float, float]:
    """Measures isolated cold_init_ms, current_rss_mb, and peak_rss_mb in a fresh Python process."""
    code = (
        "import json, time; "
        "from ml.experiments.run_asr_ablation import _get_memory_mb; "
        "from services.asr import WhisperTransformerASR; "
        "from services.contracts import AudioPayload; "
        "t0 = time.perf_counter(); "
        f"asr = WhisperTransformerASR(model_size='{model_size}', compute_type='int8'); "
        "asr._lazy_init(); "
        "cold = round((time.perf_counter() - t0) * 1000.0, 1); "
        "asr.transcribe(AudioPayload(source_type='file_upload', file_path='storage/benchmarks/interview_stress_v1/pn_00.wav')); "
        "rss, peak = _get_memory_mb(); "
        "print(json.dumps({'cold': cold, 'rss': rss, 'peak': peak}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return float(data["cold"]), float(data["rss"]), float(data["peak"])


def _hf_model_disk_mb(model_size: str) -> float:
    """Computes total CTranslate2 model weight footprint on disk from HuggingFace cache."""
    hub_dir = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub"))
    matches = glob.glob(os.path.join(hub_dir, f"*faster-whisper-{model_size}*"))
    total_bytes = 0
    for root in matches:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if os.path.isfile(fp):
                    total_bytes += os.path.getsize(fp)
    return round(total_bytes / (1024.0 * 1024.0), 2)


def _load_audio_16k_mono(path: str) -> Tuple[np.ndarray, float]:
    """Loads WAV/FLAC as contiguous float32 mono 16 kHz numpy array plus duration_sec."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal as signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
    else:
        audio = np.asarray(audio, dtype=np.float32)
    return np.ascontiguousarray(audio), round(len(audio) / 16000.0, 3)


def _sample_wer(ref: str, hyp: str) -> float:
    r = _normalize_text(ref) or "<empty>"
    h = _normalize_text(hyp)
    return round(float(jiwer.wer([r], [h])), 4)


def _sample_term_recall(hyp: str, terms: Sequence[str]) -> Optional[float]:
    if not terms:
        return None
    h = _normalize_text(hyp)
    h_toks = set(h.split())
    valid = [_normalize_text(t) for t in terms if _normalize_text(t)]
    if not valid:
        return None
    hits = sum(1 for t in valid if t in h or t in h_toks)
    return round(hits / len(valid), 4)


def _build_domain_overrides(cfg: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Builds per-domain prompt, hotwords, and lexicon without exposing test ground truth."""
    mode = cfg.get("prompt_mode", "none")
    use_hw = bool(cfg.get("hotwords_enabled", False))
    use_repair = bool(cfg.get("phonetic_repair_enabled", False))
    is_tiny = cfg.get("model_size", "").startswith("tiny")

    if domain == "interview":
        lex = INTERVIEW_ROLE_TERMS + CANDIDATE_ROSTER_NAMES
        if mode == "none":
            prompt = ""
        elif mode == "static":
            prompt = WhisperTransformerASR.DEFAULT_PROMPT
        else:
            prompt = build_dynamic_bias_prompt(INTERVIEW_ROLE_TERMS, CANDIDATE_ROSTER_NAMES)
        return {
            "initial_prompt": prompt,
            "hotwords": " ".join(lex) if use_hw else None,
            "repair_lexicon": lex if use_repair else None,
            "word_timestamps": bool(cfg.get("word_timestamps", False)),
        }

    if domain == "minds14":
        # Capacity gating: omit descriptive sentence prompt on <100M tiny.en to prevent prompt bleeding
        if mode == "none" or (is_tiny and mode == "dynamic"):
            prompt = ""
        elif mode == "static":
            prompt = WhisperTransformerASR.DEFAULT_PROMPT
        else:
            prompt = (
                "Banking customer query about abroad travel, address, app error, ATM limit, "
                "balance, business loan, card issues, cash deposit, direct debit, freeze card, "
                "high value payment, joint account, latest transactions, and pay bill."
            )
        return {
            "initial_prompt": prompt,
            "hotwords": " ".join(MINDS14_BANKING_TERMS) if use_hw else None,
            "repair_lexicon": MINDS14_BANKING_TERMS if use_repair else None,
            "word_timestamps": bool(cfg.get("word_timestamps", False)),
        }

    # Open-domain LibriSpeech: zero domain lexicon injection (checks false-positive safety)
    return {
        "initial_prompt": "" if (mode == "none" or is_tiny) else (
            WhisperTransformerASR.DEFAULT_PROMPT if mode == "static"
            else "Natural English conversational and literary speech."
        ),
        "hotwords": None,
        "repair_lexicon": None,
        "word_timestamps": bool(cfg.get("word_timestamps", False)),
    }


def _evaluate_split(
    asr: WhisperTransformerASR,
    items: List[Dict[str, Any]],
    domain_meta: Dict[str, Any],
    snr_db: Optional[float] = None,
    seed: int = 42,
    collect_transcripts: bool = False,
) -> Dict[str, Any]:
    """Runs ASR over an item list (clean WAV paths or in-memory AWGN arrays) and aggregates metrics."""
    refs: List[str] = []
    hyps: List[str] = []
    target_terms_list: Optional[List[List[str]]] = [] if any("target_terms" in it for it in items) else None
    latencies_ms: List[float] = []
    audio_durations: List[float] = []
    per_item_wer: List[float] = []
    per_item_entity_recall: List[float] = []
    transcripts: List[Dict[str, Any]] = []
    total_repairs = 0

    for idx, item in enumerate(items):
        abs_path = os.path.join(ROOT_DIR, item["audio_path"])
        if snr_db is None:
            dur = float(item.get("duration_sec") or 0.0)
            payload = AudioPayload(source_type="file_upload", file_path=abs_path, duration_sec=dur, metadata=dict(domain_meta))
        else:
            wav, dur = _load_audio_16k_mono(abs_path)
            noisy = add_awgn(wav, snr_db=snr_db, seed=seed + idx)
            meta = dict(domain_meta)
            meta["audio_dtype"] = "float32"
            payload = AudioPayload(
                source_type="stream_buffer",
                sample_rate=16000,
                duration_sec=dur,
                raw_bytes=noisy.tobytes(),
                metadata=meta,
            )

        t0 = time.perf_counter()
        res = asr.transcribe(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hyp = res.transcript or ""
        ref = item["text"]
        refs.append(ref)
        hyps.append(hyp)
        latencies_ms.append(elapsed_ms)
        audio_durations.append(dur if dur > 0 else 1.0)
        per_item_wer.append(_sample_wer(ref, hyp))

        if target_terms_list is not None:
            tlist = item.get("target_terms", [])
            target_terms_list.append(tlist)
            tr = _sample_term_recall(hyp, tlist)
            if tr is not None:
                per_item_entity_recall.append(tr)

        repairs = res.metadata.get("phonetic_repairs") or []
        total_repairs += len(repairs)

        if collect_transcripts:
            transcripts.append({
                "id": item["id"],
                "subset": item.get("subset"),
                "accent": item.get("accent"),
                "intent_label": item.get("intent_label"),
                "rubric_tier": item.get("rubric_tier"),
                "expected_score_range": item.get("expected_score_range"),
                "reference": ref,
                "hypothesis": hyp,
                "raw_hypothesis": res.metadata.get("raw_transcript", hyp),
                "repairs": repairs,
                "wer": per_item_wer[-1],
                "latency_ms": round(elapsed_ms, 1),
            })

    summary = compute_asr_metrics(refs, hyps, target_terms=target_terms_list if target_terms_list and any(target_terms_list) else None)
    total_audio_s = sum(audio_durations)
    total_infer_s = sum(latencies_ms) / 1000.0
    summary.update({
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "rtf": round(total_infer_s / total_audio_s, 4) if total_audio_s > 0 else 0.0,
        "total_phonetic_repairs": total_repairs,
        "cost_per_1k_turns_usd": estimate_cost_per_1k_turns(avg_latency_sec=total_infer_s / len(items)),
        "per_item_wer": per_item_wer,
        "per_item_entity_recall": per_item_entity_recall,
    })
    if collect_transcripts:
        summary["transcripts"] = transcripts
    return summary


def _item_wer_list_from_records(records: Sequence[Dict[str, Any]], targets_map: Optional[Dict[str, List[str]]] = None):
    wers = [float(r["wer"]) for r in records]
    ents = []
    if targets_map is not None:
        for r in records:
            terms = targets_map.get(r["id"])
            if terms:
                ents.append(_sample_term_recall(r["hypothesis"], terms) or 0.0)
    return wers, ents


def run_ablation(only_configs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    interview_all = load_manifest("interview_stress_v1")["items"]
    interview_test = [x for x in interview_all if x["split"] == "test"]
    pn_test = [x for x in interview_test if x.get("subset") == "proper_nouns"]
    targets_map = {x["id"]: x["target_terms"] for x in pn_test}

    minds_test = [x for x in load_manifest("minds14_en_140")["items"] if x["split"] == "test"]
    libri_test = [x for x in load_manifest("librispeech_60")["items"] if x["split"] == "test"]
    libri_noisy_12: List[Dict[str, Any]] = []
    seen_spk = set()
    for x in libri_test:
        if x["speaker_id"] not in seen_spk:
            seen_spk.add(x["speaker_id"])
            libri_noisy_12.append(x)

    out_path = os.path.join(ROOT_DIR, "ml", "results", "asr_ablation.json")
    existing: Dict[str, Any] = {}
    if only_configs and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    results: Dict[str, Any] = {
        "experiment": "layer_a_asr_ablation_v2",
        "configs": dict(existing.get("configs", {})) if existing else {},
        "paired_bootstrap": {},
        "negative_results": dict(existing.get("negative_results", {})),
        "biggest_wins_tiny": [],
        "residual_errors_small_improved": [],
    }

    # Preserve unguarded hotword negative result on tiny.en before overwriting with capacity-gated v2
    prev_tiny = (existing.get("configs") or {}).get("asr_tiny_improved")
    if prev_tiny and prev_tiny.get("config", {}).get("hotwords_enabled") is True:
        results["negative_results"]["tiny_unguarded_hotwords_v1"] = {
            "finding": "39M tiny.en suffers prompt/hotword echo & end-of-speech hallucination loops when fed dense 60-term hotwords without repetition guards.",
            "interview_clean_wer": prev_tiny["interview_stress_clean"]["wer"],
            "proper_nouns_wer": prev_tiny["interview_stress_clean"]["proper_nouns_wer"],
            "proper_nouns_r_entity": prev_tiny["interview_stress_clean"]["proper_nouns_r_entity"],
            "interview_qa_wer": prev_tiny["interview_stress_clean"]["interview_qa_wer"],
            "interview_rtf": prev_tiny["interview_stress_clean"]["rtf"],
            "minds14_clean_wer": prev_tiny["minds14_clean"]["wer"],
        }

    targets_to_run = list(only_configs) if only_configs else list(CONFIG_NAMES)
    per_item_cache: Dict[str, Dict[str, List[float]]] = {}

    # Reconstruct cached per-item arrays for untouched configs so bootstrap tests cover all 4 configs
    if only_configs:
        for cname in CONFIG_NAMES:
            if cname in targets_to_run or cname not in results["configs"]:
                continue
            entry = results["configs"][cname]
            iv_recs = entry["interview_stress_clean"]["transcripts"]
            m14_recs = entry["minds14_clean"]["transcripts"]
            iv_w, iv_ent = _item_wer_list_from_records(iv_recs, targets_map)
            m14_w, _ = _item_wer_list_from_records(m14_recs)
            # Re-evaluate lightweight librispeech clean/pn_10db arrays on CPU only if missing
            per_item_cache[cname] = {
                "iv_wer": iv_w,
                "iv_r_entity": iv_ent,
                "minds14_wer": m14_w,
            }

    for cfg_name in targets_to_run:
        cfg_path = os.path.join(ROOT_DIR, "ml", "configs", f"{cfg_name}.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        print(f"\n=== Running {cfg_name} ({cfg['model_size']}, vad={cfg['vad_filter']}, prompt={cfg['prompt_mode']}, hw={cfg['hotwords_enabled']}, rep={cfg.get('repetition_penalty', 1.0)}, ngram={cfg.get('no_repeat_ngram_size', 0)}) ===", flush=True)
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
        cold_init_ms, rss_mb, peak_rss_mb = _probe_model_rss_mb(cfg["model_size"])
        asr._lazy_init()

        iv_meta = _build_domain_overrides(cfg, "interview")
        m14_meta = _build_domain_overrides(cfg, "minds14")
        lib_meta = _build_domain_overrides(cfg, "librispeech")

        iv_clean = _evaluate_split(asr, interview_test, iv_meta, snr_db=None, collect_transcripts=True)
        pn_recs = [t for t in iv_clean["transcripts"] if t["subset"] == "proper_nouns"]
        qa_recs = [t for t in iv_clean["transcripts"] if t["subset"] == "interview_qa"]
        pn_clean_summary = compute_asr_metrics(
            [r["reference"] for r in pn_recs],
            [r["hypothesis"] for r in pn_recs],
            target_terms=[x["target_terms"] for x in pn_test],
        )
        qa_clean_summary = compute_asr_metrics([r["reference"] for r in qa_recs], [r["hypothesis"] for r in qa_recs])
        print(f"  [interview clean 50] WER={iv_clean['wer']} | pn_WER={pn_clean_summary['wer']} R_entity={pn_clean_summary.get('target_term_recall')} | qa_WER={qa_clean_summary['wer']} | RTF={iv_clean['rtf']}", flush=True)

        pn_20db = _evaluate_split(asr, pn_test, iv_meta, snr_db=20.0, seed=cfg["seed"])
        pn_10db = _evaluate_split(asr, pn_test, iv_meta, snr_db=10.0, seed=cfg["seed"] + 100)
        print(f"  [proper_nouns noise 20] 20dB WER={pn_20db['wer']} R_ent={pn_20db.get('target_term_recall')} | 10dB WER={pn_10db['wer']} R_ent={pn_10db.get('target_term_recall')}", flush=True)

        m14_clean = _evaluate_split(asr, minds_test, m14_meta, snr_db=None, collect_transcripts=True)
        accent_wer = {}
        for acc in ("en-US", "en-GB", "en-AU"):
            sub = [r for r in m14_clean["transcripts"] if r["accent"] == acc]
            if sub:
                accent_wer[acc] = compute_asr_metrics([r["reference"] for r in sub], [r["hypothesis"] for r in sub])["wer"]
        print(f"  [minds14 clean 70] WER={m14_clean['wer']} CER={m14_clean['cer']} accents={accent_wer} RTF={m14_clean['rtf']}", flush=True)

        lib_clean = _evaluate_split(asr, libri_test, lib_meta, snr_db=None)
        lib_20db = _evaluate_split(asr, libri_noisy_12, lib_meta, snr_db=20.0, seed=cfg["seed"])
        lib_10db = _evaluate_split(asr, libri_noisy_12, lib_meta, snr_db=10.0, seed=cfg["seed"] + 200)
        print(f"  [librispeech test] clean(36) WER={lib_clean['wer']} | 20dB(12) WER={lib_20db['wer']} | 10dB(12) WER={lib_10db['wer']} | RSS={rss_mb} MB Peak={peak_rss_mb} MB", flush=True)

        per_item_cache[cfg_name] = {
            "iv_wer": iv_clean["per_item_wer"],
            "iv_r_entity": iv_clean["per_item_entity_recall"],
            "minds14_wer": m14_clean["per_item_wer"],
            "pn10db_wer": pn_10db["per_item_wer"],
            "libri_wer": lib_clean["per_item_wer"],
        }

        results["configs"][cfg_name] = {
            "config": cfg,
            "cold_init_ms": cold_init_ms,
            "model_disk_mb": _hf_model_disk_mb(cfg["model_size"]),
            "current_rss_mb": rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "interview_stress_clean": {
                **{k: v for k, v in iv_clean.items() if k not in ("per_item_wer", "per_item_entity_recall")},
                "proper_nouns_wer": pn_clean_summary["wer"],
                "proper_nouns_cer": pn_clean_summary["cer"],
                "proper_nouns_r_entity": pn_clean_summary.get("target_term_recall"),
                "interview_qa_wer": qa_clean_summary["wer"],
            },
            "proper_nouns_20db": {k: v for k, v in pn_20db.items() if k not in ("per_item_wer", "per_item_entity_recall")},
            "proper_nouns_10db": {k: v for k, v in pn_10db.items() if k not in ("per_item_wer", "per_item_entity_recall")},
            "minds14_clean": {
                **{k: v for k, v in m14_clean.items() if k != "per_item_wer"},
                "accent_wer": accent_wer,
            },
            "librispeech_clean": {k: v for k, v in lib_clean.items() if k != "per_item_wer"},
            "librispeech_20db": {k: v for k, v in lib_20db.items() if k != "per_item_wer"},
            "librispeech_10db": {k: v for k, v in lib_10db.items() if k != "per_item_wer"},
        }

    # Ensure isolated cold_init & real RSS numbers populate every config cleanly
    rss_by_size: Dict[str, Tuple[float, float, float]] = {}
    for cfg_name, entry in results["configs"].items():
        if entry.get("peak_rss_mb", 0.0) <= 0.0:
            msize = entry["config"]["model_size"]
            if msize not in rss_by_size:
                rss_by_size[msize] = _probe_model_rss_mb(msize)
            c_ms, r_mb, p_mb = rss_by_size[msize]
            entry["cold_init_ms"] = c_ms
            entry["current_rss_mb"] = r_mb
            entry["peak_rss_mb"] = p_mb

    # Paired bootstrap significance tests (N=1000, seed=42)
    comparisons = [
        ("tiny_improved_vs_tiny_baseline_interview_wer", "asr_tiny_baseline", "asr_tiny_improved", "iv_wer", False),
        ("tiny_improved_vs_tiny_baseline_interview_r_entity", "asr_tiny_baseline", "asr_tiny_improved", "iv_r_entity", True),
        ("tiny_improved_vs_small_baseline_interview_wer", "asr_small_baseline", "asr_tiny_improved", "iv_wer", False),
        ("tiny_improved_vs_small_baseline_interview_r_entity", "asr_small_baseline", "asr_tiny_improved", "iv_r_entity", True),
        ("small_improved_vs_small_baseline_interview_wer", "asr_small_baseline", "asr_small_improved", "iv_wer", False),
        ("small_improved_vs_small_baseline_interview_r_entity", "asr_small_baseline", "asr_small_improved", "iv_r_entity", True),
        ("tiny_improved_vs_tiny_baseline_minds14_wer", "asr_tiny_baseline", "asr_tiny_improved", "minds14_wer", False),
        ("small_improved_vs_small_baseline_minds14_wer", "asr_small_baseline", "asr_small_improved", "minds14_wer", False),
    ]
    for label, base_k, cand_k, metric_key, hib in comparisons:
        b_arr = per_item_cache.get(base_k, {}).get(metric_key)
        c_arr = per_item_cache.get(cand_k, {}).get(metric_key)
        if b_arr and c_arr and len(b_arr) == len(c_arr):
            results["paired_bootstrap"][label] = paired_bootstrap_test(
                baseline_scores=b_arr, candidate_scores=c_arr, n_resamples=1000, seed=42, higher_is_better=hib
            )

    # Extract concrete wins and residual errors cleanly
    cfgs = results["configs"]
    if all(k in cfgs for k in ("asr_tiny_baseline", "asr_tiny_improved", "asr_small_improved")):
        t_base_iv = cfgs["asr_tiny_baseline"]["interview_stress_clean"]["transcripts"]
        t_imp_iv = cfgs["asr_tiny_improved"]["interview_stress_clean"]["transcripts"]
        s_imp_iv = cfgs["asr_small_improved"]["interview_stress_clean"]["transcripts"]
        deltas = []
        for rb, ri, rs in zip(t_base_iv, t_imp_iv, s_imp_iv):
            deltas.append({
                "id": rb["id"],
                "subset": rb["subset"],
                "reference": rb["reference"],
                "tiny_baseline_hyp": rb["hypothesis"],
                "tiny_improved_raw": ri["raw_hypothesis"],
                "tiny_improved_hyp": ri["hypothesis"],
                "tiny_improved_repairs": ri["repairs"],
                "small_improved_hyp": rs["hypothesis"],
                "tiny_wer_delta": round(rb["wer"] - ri["wer"], 4),
                "small_improved_wer": rs["wer"],
            })
        results["biggest_wins_tiny"] = sorted(deltas, key=lambda d: d["tiny_wer_delta"], reverse=True)[:5]
        results["residual_errors_small_improved"] = sorted(deltas, key=lambda d: d["small_improved_wer"], reverse=True)[:5]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete Layer A ablation results -> {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Run only specific config names and merge into asr_ablation.json")
    args = parser.parse_args()
    run_ablation(only_configs=args.only)
