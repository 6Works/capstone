"""
Session 5 Experiment (Upgraded): Layer A (Stage 2 ASR) Ablation, Generalization & Real-Speech Benchmark.
Compares 6 reproducible CPU INT8 ASR configurations across Whisper & 2026 SOTA FastConformer-TDT:
  1. asr_tiny_baseline  (Whisper tiny.en 39M, vad=False, no prompt, no hotwords, no repair)
  2. asr_small_baseline (Whisper small.en 244M, vad=True, static DEFAULT_PROMPT)
  3. asr_tiny_improved  (Whisper tiny.en 39M + vad + capacity-gated prompt + loop guards + G2P repair)
  4. asr_small_soundex  (Whisper small.en 244M + vad + dynamic prompt + hotwords + legacy Soundex repair)
  5. asr_small_improved (Whisper small.en 244M + vad + dynamic prompt + hotwords + G2P ARPAbet repair)
  6. asr_parakeet_tdt   (NVIDIA parakeet-tdt-0.6b-v3 INT8 CC-BY-4.0 + G2P repair)

Evaluates across:
  - interview_stress_v1 test (50 clean + 20 proper_nouns @ 20 dB & 10 dB AWGN)
  - Closed-Roster vs Zero-Shot Unseen Proper-Noun Generalization (50% proper nouns strictly withheld from prompt/hotwords/lexicon)
  - Non-Entity WER, Repair Precision, and False-Repair Over-Correction Rate
  - minds14_en_140 (70 clips) AND full minds14_en_full test (498 real human clips across en-US/GB/AU)
  - librispeech_60 test-clean (36 clean + 12 @ 20/10 dB) AND librispeech_other_60 test-other (36 clean + 12 @ 10 dB)
  - Bounded Paired Bootstrap tests (N=2000, Phipson-Smyth, never p=0.0) & same-size controls.
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

from services.asr import WhisperTransformerASR, ParakeetTDTASR, build_dynamic_bias_prompt, phonetic_repair
from services.asr.asr_services import _WHISPER_CACHE
from services.contracts import AudioPayload
from ml.datasets import load_manifest, add_awgn
from ml.metrics import compute_asr_metrics, compute_itn_wer, paired_bootstrap_test, estimate_cost_per_1k_turns
from ml.metrics.eval_metrics import _normalize_text


# Fixed session-level Role Blueprint + Candidate Roster (closed-roster test condition)
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

# Deterministically withheld 50% of proper-noun target entities for zero-shot open-vocabulary testing
_HELD_OUT_LOWER = {
    "aradhya", "chromadb", "ctranslate2", "docker", "fastapi", "graphql", "kafka",
    "krishnendu", "kyutai", "lora", "nginx", "openvino", "pooja", "postgresql",
    "prometheus", "pytest", "qwen", "rohan", "sachin", "sid", "siddharth",
    "tensorflow", "tensorrt", "wespeaker", "websocket",
}
SEEN_ROLE_TERMS = [t for t in INTERVIEW_ROLE_TERMS if t.lower() not in _HELD_OUT_LOWER]
SEEN_CANDIDATE_NAMES = [t for t in CANDIDATE_ROSTER_NAMES if t.lower() not in _HELD_OUT_LOWER]

CONFIG_NAMES = [
    "asr_tiny_baseline",
    "asr_small_baseline",
    "asr_tiny_improved",
    "asr_small_soundex",
    "asr_small_improved",
    "asr_parakeet_tdt",
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


def _probe_model_rss_mb(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Measures isolated cold_init_ms, current_rss_mb, and peak_rss_mb in a fresh Python process."""
    prov = cfg.get("provider", "whisper")
    if prov == "parakeet_tdt":
        init_stmt = "asr = ParakeetTDTASR(); asr._get_recognizer(); "
    else:
        init_stmt = f"asr = WhisperTransformerASR(model_size='{cfg['model_size']}', compute_type='int8'); asr._lazy_init(); "

    code = (
        "import json, time; "
        "from ml.experiments.run_asr_ablation import _get_memory_mb; "
        "from services.asr import WhisperTransformerASR, ParakeetTDTASR; "
        "from services.contracts import AudioPayload; "
        "t0 = time.perf_counter(); "
        + init_stmt
        + "cold = round((time.perf_counter() - t0) * 1000.0, 1); "
        "asr.transcribe(AudioPayload(source_type='file_upload', file_path='storage/benchmarks/interview_stress_v1/pn_00.wav')); "
        "rss, peak = _get_memory_mb(); "
        "print(json.dumps({'cold': cold, 'rss': rss, 'peak': peak}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return float(data["cold"]), float(data["rss"]), float(data["peak"])


def _model_disk_mb(cfg: Dict[str, Any]) -> float:
    """Computes total model weight footprint on disk in MB."""
    if cfg.get("provider") == "parakeet_tdt":
        pdir = os.path.join(ROOT_DIR, ParakeetTDTASR.DEFAULT_MODEL_DIR)
        total = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(pdir)
            for fn in fns
        )
        return round(total / (1024.0 * 1024.0), 2)

    hub_dir = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub"))
    matches = glob.glob(os.path.join(hub_dir, f"*faster-whisper-{cfg['model_size']}*"))
    total_bytes = 0
    for root in matches:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if os.path.isfile(fp):
                    total_bytes += os.path.getsize(fp)
    return round(total_bytes / (1024.0 * 1024.0), 2)


def _load_audio_16k_mono(path: str) -> Tuple[np.ndarray, float]:
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


def _build_domain_overrides(cfg: Dict[str, Any], domain: str, holdout_mode: bool = False) -> Dict[str, Any]:
    """Builds per-domain prompt, hotwords, and lexicon without exposing test ground truth."""
    mode = cfg.get("prompt_mode", "none")
    use_hw = bool(cfg.get("hotwords_enabled", False))
    use_repair = bool(cfg.get("phonetic_repair_enabled", False))
    ph_mode = cfg.get("phonetic_mode", "g2p")
    is_tiny = cfg.get("model_size", "").startswith("tiny")

    if domain == "interview":
        role_terms = SEEN_ROLE_TERMS if holdout_mode else INTERVIEW_ROLE_TERMS
        cand_names = SEEN_CANDIDATE_NAMES if holdout_mode else CANDIDATE_ROSTER_NAMES
        lex = role_terms + cand_names
        if mode == "none":
            prompt = ""
        elif mode == "static":
            prompt = WhisperTransformerASR.DEFAULT_PROMPT
        else:
            prompt = build_dynamic_bias_prompt(role_terms, cand_names)
        return {
            "initial_prompt": prompt,
            "hotwords": " ".join(lex) if use_hw else None,
            "repair_lexicon": lex if use_repair else None,
            "phonetic_mode": ph_mode,
            "word_timestamps": bool(cfg.get("word_timestamps", False)),
        }

    if domain == "minds14":
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
            "phonetic_mode": ph_mode,
            "word_timestamps": bool(cfg.get("word_timestamps", False)),
        }

    return {
        "initial_prompt": "" if (mode == "none" or is_tiny) else (
            WhisperTransformerASR.DEFAULT_PROMPT if mode == "static"
            else "Natural English conversational and literary speech."
        ),
        "hotwords": None,
        "repair_lexicon": None,
        "phonetic_mode": ph_mode,
        "word_timestamps": bool(cfg.get("word_timestamps", False)),
    }


def _evaluate_split(
    asr: Any,
    items: List[Dict[str, Any]],
    domain_meta: Dict[str, Any],
    snr_db: Optional[float] = None,
    seed: int = 42,
    collect_transcripts: bool = False,
    raw_cache_out: Optional[Dict[str, Tuple[str, List[Dict[str, Any]], float, float]]] = None,
    raw_cache_in: Optional[Dict[str, Tuple[str, List[Dict[str, Any]], float, float]]] = None,
) -> Dict[str, Any]:
    """Runs ASR over an item list (or deterministic raw-decode replay for Soundex vs G2P isolation)."""
    refs: List[str] = []
    hyps: List[str] = []
    raw_hyps: List[str] = []
    target_terms_list: Optional[List[List[str]]] = [] if any("target_terms" in it for it in items) else None
    latencies_ms: List[float] = []
    audio_durations: List[float] = []
    per_item_wer: List[float] = []
    per_item_entity_recall: List[float] = []
    transcripts: List[Dict[str, Any]] = []
    total_repairs = 0

    lex = domain_meta.get("repair_lexicon")
    ph_mode = domain_meta.get("phonetic_mode", "g2p")

    for idx, item in enumerate(items):
        item_id = item["id"]
        if raw_cache_in is not None and item_id in raw_cache_in:
            raw_hyp, words_meta, base_ms, dur = raw_cache_in[item_id]
            t0 = time.perf_counter()
            if lex and raw_hyp:
                hyp, repairs = phonetic_repair(
                    raw_hyp,
                    lexicon=lex,
                    words_meta=words_meta,
                    confidence_threshold=float(getattr(asr, "repair_confidence_threshold", 0.85)),
                    phonetic_mode=ph_mode,
                )
            else:
                hyp, repairs = raw_hyp, []
            elapsed_ms = base_ms + (time.perf_counter() - t0) * 1000.0
        else:
            abs_path = os.path.join(ROOT_DIR, item["audio_path"])
            if snr_db is None:
                dur = float(item.get("duration_sec") or 0.0)
                if dur <= 0.0:
                    _, dur = _load_audio_16k_mono(abs_path)
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
            raw_hyp = res.metadata.get("raw_transcript", hyp)
            words_meta = res.metadata.get("words", [])
            repairs = res.metadata.get("phonetic_repairs") or []
            if raw_cache_out is not None:
                raw_cache_out[item_id] = (raw_hyp, words_meta, elapsed_ms, dur if dur > 0 else 1.0)

        ref = item["text"]
        refs.append(ref)
        hyps.append(hyp)
        raw_hyps.append(raw_hyp)
        latencies_ms.append(elapsed_ms)
        audio_durations.append(dur if dur > 0 else 1.0)
        per_item_wer.append(_sample_wer(ref, hyp))

        if target_terms_list is not None:
            tlist = item.get("target_terms", [])
            target_terms_list.append(tlist)
            tr = _sample_term_recall(hyp, tlist)
            if tr is not None:
                per_item_entity_recall.append(tr)

        total_repairs += len(repairs)
        if collect_transcripts:
            transcripts.append({
                "id": item_id,
                "subset": item.get("subset"),
                "accent": item.get("accent"),
                "intent_label": item.get("intent_label"),
                "rubric_tier": item.get("rubric_tier"),
                "expected_score_range": item.get("expected_score_range"),
                "reference": ref,
                "hypothesis": hyp,
                "raw_hypothesis": raw_hyp,
                "repairs": repairs,
                "wer": per_item_wer[-1],
                "latency_ms": round(elapsed_ms, 1),
            })

    has_repairs = bool(lex)
    summary = compute_asr_metrics(
        refs,
        hyps,
        target_terms=target_terms_list if target_terms_list and any(target_terms_list) else None,
        raw_hypotheses=raw_hyps if has_repairs else None,
    )
    summary.update(compute_itn_wer(refs, hyps))
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


def _compute_heldout_pn_breakdown(pn_test: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Splits proper_nouns recall cleanly into closed-roster (seen) vs zero-shot held-out (unseen) terms."""
    seen_hits, seen_tot = 0, 0
    unseen_hits, unseen_tot = 0, 0
    rec_by_id = {r["id"]: r for r in records}
    for it in pn_test:
        r = rec_by_id.get(it["id"])
        if not r:
            continue
        hyp_n = _normalize_text(r["hypothesis"])
        hyp_toks = set(hyp_n.split())
        for term in it.get("target_terms", []):
            tn = _normalize_text(term)
            if not tn:
                continue
            hit = 1 if (tn in hyp_n or tn in hyp_toks) else 0
            if tn in _HELD_OUT_LOWER or term.lower() in _HELD_OUT_LOWER:
                unseen_tot += 1
                unseen_hits += hit
            else:
                seen_tot += 1
                seen_hits += hit
    return {
        "closed_roster_recall": round(seen_hits / seen_tot, 4) if seen_tot else None,
        "closed_roster_total": seen_tot,
        "unseen_zero_shot_recall": round(unseen_hits / unseen_tot, 4) if unseen_tot else None,
        "unseen_zero_shot_total": unseen_tot,
    }


def run_ablation(only_configs: Optional[Sequence[str]] = None, skip_full_m14: bool = False) -> Dict[str, Any]:
    interview_all = load_manifest("interview_stress_v1")["items"]
    interview_test = [x for x in interview_all if x["split"] == "test"]
    pn_test = [x for x in interview_test if x.get("subset") == "proper_nouns"]
    targets_map = {x["id"]: x["target_terms"] for x in pn_test}

    minds_140_test = [x for x in load_manifest("minds14_en_140")["items"] if x["split"] == "test"]
    minds_full_test = [x for x in load_manifest("minds14_en_full")["items"] if x["split"] == "test"]

    libri_test = [x for x in load_manifest("librispeech_60")["items"] if x["split"] == "test"]
    libri_noisy_12: List[Dict[str, Any]] = []
    seen_spk = set()
    for x in libri_test:
        if x["speaker_id"] not in seen_spk:
            seen_spk.add(x["speaker_id"])
            libri_noisy_12.append(x)

    other_test = [x for x in load_manifest("librispeech_other_60")["items"] if x["split"] == "test"]
    other_noisy_12: List[Dict[str, Any]] = []
    seen_o = set()
    for x in other_test:
        if x["speaker_id"] not in seen_o:
            seen_o.add(x["speaker_id"])
            other_noisy_12.append(x)

    out_path = os.path.join(ROOT_DIR, "ml", "results", "asr_ablation.json")
    existing: Dict[str, Any] = {}
    if only_configs and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    results: Dict[str, Any] = {
        "experiment": "layer_a_asr_ablation_v3",
        "configs": dict(existing.get("configs", {})) if existing else {},
        "paired_bootstrap": dict(existing.get("paired_bootstrap", {})) if only_configs else {},
        "negative_results": dict(existing.get("negative_results", {})),
        "biggest_wins_tiny": [],
        "residual_errors_small_improved": [],
    }
    if not results["negative_results"]:
        results["negative_results"]["tiny_unguarded_hotwords_v1"] = {
            "finding": "39M tiny.en suffers prompt/hotword echo & end-of-speech hallucination loops when fed dense 60-term hotwords without repetition guards.",
            "interview_clean_wer": 0.4508,
            "proper_nouns_wer": 0.7204,
            "proper_nouns_r_entity": 0.371,
            "interview_qa_wer": 0.2587,
            "interview_rtf": 0.2896,
            "minds14_clean_wer": 0.1419,
        }

    # Populate shared small.en raw decode cache so small_soundex & small_improved share identical hypotheses cleanly
    small_raw_caches: Dict[str, Dict[str, Tuple[str, List[Dict[str, Any]], float, float]]] = {
        "iv_clean": {},
        "iv_holdout": {},
        "pn_20db": {},
        "pn_10db": {},
        "m14_140": {},
        "m14_full": {},
    }

    targets_to_run = list(only_configs) if only_configs else list(CONFIG_NAMES)
    per_item_cache: Dict[str, Dict[str, List[float]]] = {}

    for cname, entry in results["configs"].items():
        if cname not in targets_to_run:
            iv_recs = (entry.get("interview_stress_clean") or {}).get("transcripts") or []
            m14_recs = (entry.get("minds14_clean") or {}).get("transcripts") or []
            if iv_recs:
                per_item_cache[cname] = {
                    "iv_wer": [float(r["wer"]) for r in iv_recs],
                    "iv_r_entity": [_sample_term_recall(r["hypothesis"], targets_map[r["id"]]) or 0.0 for r in iv_recs if r["id"] in targets_map],
                    "minds14_wer": [float(r["wer"]) for r in m14_recs] if m14_recs else [],
                }

    for cfg_name in targets_to_run:
        cfg_path = os.path.join(ROOT_DIR, "ml", "configs", f"{cfg_name}.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        print(f"\n=== Running {cfg_name} ({cfg.get('provider', 'whisper')}:{cfg['model_size']}, ph={cfg.get('phonetic_mode', 'none')}) ===", flush=True)
        _WHISPER_CACHE.clear()
        if cfg.get("provider") == "parakeet_tdt":
            asr = ParakeetTDTASR(
                decoding_method="greedy_search",
                phonetic_mode=cfg.get("phonetic_mode", "g2p"),
            )
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

        cold_init_ms, rss_mb, peak_rss_mb = _probe_model_rss_mb(cfg)

        iv_meta = _build_domain_overrides(cfg, "interview", holdout_mode=False)
        iv_hold_meta = _build_domain_overrides(cfg, "interview", holdout_mode=True)
        m14_meta = _build_domain_overrides(cfg, "minds14")
        lib_meta = _build_domain_overrides(cfg, "librispeech")

        use_small_cache = cfg_name in ("asr_small_soundex", "asr_small_improved")
        c_in = (lambda k: small_raw_caches[k] if use_small_cache and small_raw_caches[k] else None)
        c_out = (lambda k: small_raw_caches[k] if use_small_cache and not small_raw_caches[k] else None)

        iv_clean = _evaluate_split(asr, interview_test, iv_meta, snr_db=None, collect_transcripts=True, raw_cache_in=c_in("iv_clean"), raw_cache_out=c_out("iv_clean"))
        pn_recs = [t for t in iv_clean["transcripts"] if t["subset"] == "proper_nouns"]
        qa_recs = [t for t in iv_clean["transcripts"] if t["subset"] == "interview_qa"]
        pn_clean_summary = compute_asr_metrics(
            [r["reference"] for r in pn_recs],
            [r["hypothesis"] for r in pn_recs],
            target_terms=[x["target_terms"] for x in pn_test],
            raw_hypotheses=[r["raw_hypothesis"] for r in pn_recs] if iv_meta.get("repair_lexicon") else None,
        )
        qa_clean_summary = compute_asr_metrics([r["reference"] for r in qa_recs], [r["hypothesis"] for r in qa_recs])

        # Open-vocabulary zero-shot proper-noun test (50% proper nouns completely withheld from prompt/hotwords/lexicon)
        iv_hold = _evaluate_split(asr, pn_test, iv_hold_meta, snr_db=None, collect_transcripts=True, raw_cache_in=c_in("iv_holdout"), raw_cache_out=c_out("iv_holdout"))
        holdout_breakdown = _compute_heldout_pn_breakdown(pn_test, iv_hold["transcripts"])
        print(f"  [interview 50] WER={iv_clean['wer']} | pn_WER={pn_clean_summary['wer']} R_ent={pn_clean_summary.get('target_term_recall')} NonEnt_WER={pn_clean_summary.get('non_entity_wer')} OverCorr={pn_clean_summary.get('overcorrection_rate')} | ZeroShot_R={holdout_breakdown['unseen_zero_shot_recall']} | RTF={iv_clean['rtf']}", flush=True)

        pn_20db = _evaluate_split(asr, pn_test, iv_meta, snr_db=20.0, seed=cfg["seed"], raw_cache_in=c_in("pn_20db"), raw_cache_out=c_out("pn_20db"))
        pn_10db = _evaluate_split(asr, pn_test, iv_meta, snr_db=10.0, seed=cfg["seed"] + 100, raw_cache_in=c_in("pn_10db"), raw_cache_out=c_out("pn_10db"))

        m14_clean = _evaluate_split(asr, minds_140_test, m14_meta, snr_db=None, collect_transcripts=True, raw_cache_in=c_in("m14_140"), raw_cache_out=c_out("m14_140"))
        accent_wer = {}
        for acc in ("en-US", "en-GB", "en-AU"):
            sub = [r for r in m14_clean["transcripts"] if r["accent"] == acc]
            if sub:
                accent_wer[acc] = compute_asr_metrics([r["reference"] for r in sub], [r["hypothesis"] for r in sub])["wer"]

        m14_full_res = None
        if not skip_full_m14:
            t_m = time.perf_counter()
            m14_full = _evaluate_split(asr, minds_full_test, m14_meta, snr_db=None, collect_transcripts=True, raw_cache_in=c_in("m14_full"), raw_cache_out=c_out("m14_full"))
            acc_full = {}
            for acc in ("en-US", "en-GB", "en-AU"):
                sub = [r for r in m14_full["transcripts"] if r["accent"] == acc]
                if sub:
                    acc_full[acc] = compute_asr_metrics([r["reference"] for r in sub], [r["hypothesis"] for r in sub])["wer"]
            m14_full_res = {
                **{k: v for k, v in m14_full.items() if k not in ("per_item_wer", "transcripts")},
                "accent_wer": acc_full,
                "wall_sec": round(time.perf_counter() - t_m, 1),
            }
            m14_tr_path = os.path.join(ROOT_DIR, "ml", "results", "minds14_full_asr_transcripts.json")
            os.makedirs(os.path.dirname(m14_tr_path), exist_ok=True)
            tr_store = json.load(open(m14_tr_path, "r", encoding="utf-8")) if os.path.exists(m14_tr_path) else {}
            tr_store[cfg_name] = m14_full["transcripts"]
            with open(m14_tr_path, "w", encoding="utf-8") as tf:
                json.dump(tr_store, tf)
            print(f"  [minds14 FULL {len(minds_full_test)}] WER={m14_full['wer']} CER={m14_full['cer']} OverCorr={m14_full.get('overcorrection_rate')} Repairs={m14_full['total_phonetic_repairs']} accents={acc_full}", flush=True)

        lib_clean = _evaluate_split(asr, libri_test, lib_meta, snr_db=None)
        lib_20db = _evaluate_split(asr, libri_noisy_12, lib_meta, snr_db=20.0, seed=cfg["seed"])
        lib_10db = _evaluate_split(asr, libri_noisy_12, lib_meta, snr_db=10.0, seed=cfg["seed"] + 200)

        other_clean = _evaluate_split(asr, other_test, lib_meta, snr_db=None)
        other_10db = _evaluate_split(asr, other_noisy_12, lib_meta, snr_db=10.0, seed=cfg["seed"] + 300)
        print(f"  [libri clean/other] clean={lib_clean['wer']} other={other_clean['wer']} other_10dB={other_10db['wer']} | Peak={peak_rss_mb} MB", flush=True)

        per_item_cache[cfg_name] = {
            "iv_wer": iv_clean["per_item_wer"],
            "iv_r_entity": iv_clean["per_item_entity_recall"],
            "minds14_wer": m14_clean["per_item_wer"],
            "minds14_full_wer": m14_full["per_item_wer"] if m14_full_res else [],
            "libri_other_wer": other_clean["per_item_wer"],
        }

        results["configs"][cfg_name] = {
            "config": cfg,
            "cold_init_ms": cold_init_ms,
            "model_disk_mb": _model_disk_mb(cfg),
            "current_rss_mb": rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "interview_stress_clean": {
                **{k: v for k, v in iv_clean.items() if k not in ("per_item_wer", "per_item_entity_recall")},
                "proper_nouns_wer": pn_clean_summary["wer"],
                "proper_nouns_cer": pn_clean_summary["cer"],
                "proper_nouns_r_entity": pn_clean_summary.get("target_term_recall"),
                "proper_nouns_non_entity_wer": pn_clean_summary.get("non_entity_wer"),
                "proper_nouns_repair_precision": pn_clean_summary.get("repair_precision"),
                "proper_nouns_overcorrection_rate": pn_clean_summary.get("overcorrection_rate"),
                "zero_shot_holdout": holdout_breakdown,
                "interview_qa_wer": qa_clean_summary["wer"],
            },
            "proper_nouns_20db": {k: v for k, v in pn_20db.items() if k not in ("per_item_wer", "per_item_entity_recall")},
            "proper_nouns_10db": {k: v for k, v in pn_10db.items() if k not in ("per_item_wer", "per_item_entity_recall")},
            "minds14_clean": {
                **{k: v for k, v in m14_clean.items() if k != "per_item_wer"},
                "accent_wer": accent_wer,
            },
            "minds14_full_test": m14_full_res,
            "librispeech_clean": {k: v for k, v in lib_clean.items() if k != "per_item_wer"},
            "librispeech_20db": {k: v for k, v in lib_20db.items() if k != "per_item_wer"},
            "librispeech_10db": {k: v for k, v in lib_10db.items() if k != "per_item_wer"},
            "librispeech_other_clean": {k: v for k, v in other_clean.items() if k != "per_item_wer"},
            "librispeech_other_10db": {k: v for k, v in other_10db.items() if k != "per_item_wer"},
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Bounded Paired Bootstrap tests (N=2000, seed=42, Phipson-Smyth)
    comparisons = [
        ("tiny_improved_vs_tiny_baseline_interview_wer", "asr_tiny_baseline", "asr_tiny_improved", "iv_wer", False),
        ("tiny_improved_vs_tiny_baseline_interview_r_entity", "asr_tiny_baseline", "asr_tiny_improved", "iv_r_entity", True),
        ("small_improved_vs_small_baseline_interview_wer", "asr_small_baseline", "asr_small_improved", "iv_wer", False),
        ("small_improved_vs_small_baseline_interview_r_entity", "asr_small_baseline", "asr_small_improved", "iv_r_entity", True),
        ("small_g2p_vs_small_soundex_minds14_full_wer", "asr_small_soundex", "asr_small_improved", "minds14_full_wer", False),
        ("small_g2p_vs_small_soundex_interview_wer", "asr_small_soundex", "asr_small_improved", "iv_wer", False),
        ("small_improved_vs_small_baseline_minds14_full_wer", "asr_small_baseline", "asr_small_improved", "minds14_full_wer", False),
        ("parakeet_vs_small_improved_minds14_full_wer", "asr_small_improved", "asr_parakeet_tdt", "minds14_full_wer", False),
        ("parakeet_vs_small_improved_libri_other_wer", "asr_small_improved", "asr_parakeet_tdt", "libri_other_wer", False),
        ("tiny_improved_vs_tiny_baseline_minds14_wer", "asr_tiny_baseline", "asr_tiny_improved", "minds14_wer", False),
        ("small_improved_vs_small_baseline_minds14_wer", "asr_small_baseline", "asr_small_improved", "minds14_wer", False),
    ]
    for label, base_k, cand_k, metric_key, hib in comparisons:
        b_arr = per_item_cache.get(base_k, {}).get(metric_key)
        c_arr = per_item_cache.get(cand_k, {}).get(metric_key)
        if b_arr and c_arr and len(b_arr) == len(c_arr):
            results["paired_bootstrap"][label] = paired_bootstrap_test(
                baseline_scores=b_arr, candidate_scores=c_arr, n_resamples=2000, seed=42, higher_is_better=hib
            )

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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete Layer A v3 ablation results -> {out_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Run specific config names and merge into asr_ablation.json")
    parser.add_argument("--skip-full-m14", action="store_true", help="Skip 498-item MINDS-14 full test sweep for quick checks")
    args = parser.parse_args()
    run_ablation(only_configs=args.only, skip_full_m14=args.skip_full_m14)
