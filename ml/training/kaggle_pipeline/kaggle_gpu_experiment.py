"""
Restructured Independent, Fault-Isolated, Resumable GPU Experiment Suite (Kaggle / CUDA / CPU).

Design Guarantees:
  1. Independent Execution: Each experiment (`asr_cache`, `distilbert_lora`, `whisper_turbo`,
     `svarah`, `qwen3_4b_qlora`) runs standalone via `--only ...` or sequentially via registry.
  2. Fault Isolation: Any failed experiment logs traceback to `experiment_status.json`, releases
     CUDA memory (`torch.cuda.empty_cache()` + `gc.collect()`), and never interrupts subsequent runs.
  3. Batch-Level Progress: Every ASR extraction, PyTorch epoch/batch, and LLM batched inference loop
     flushes structured batch progress + elapsed time + ETA.
  4. Fine-Grained Checkpoints: Optuna trials, per-seed adapters/metrics, epoch adapters, and partial
     ASR/LLM inference chunks persist atomically to disk cleanly via temp-file replace.
  5. Resumability: Completed sub-steps, Optuna trials, seeds, epochs, and cached items skip
     automatically unless `--no-resume` is specified.
"""

import os
import sys
import re
import gc
import json
import time
import random
import shutil
import argparse
import tempfile
import traceback
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

DEFAULT_OUT_DIR = "/kaggle/working/results" if os.path.exists("/kaggle/working") else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "gpu_suite")
)


def atomic_json_dump(obj: Any, path: str) -> None:
    """Atomically writes JSON checkpoint via tmp file + os.replace so partial writes never corrupt state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def free_memory() -> None:
    """Releases Python & CUDA allocations between isolated experiments."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def install_deps() -> None:
    pkgs = [
        "transformers>=4.49.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "accelerate>=1.3.0",
        "optuna>=4.0.0",
        "openai-whisper",
        "jiwer",
        "g2p_en",
        "rapidfuzz",
        "datasets",
        "soundfile",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def locate_data_root() -> str:
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "manifests" in dirs or "minds14_en_full.json" in files:
                print(f"[Data] Found Kaggle dataset root at: {root}", flush=True)
                return root
    local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
    if os.path.exists(os.path.join(local_root, "manifests")):
        print(f"[Data] Using local dataset root: {local_root}", flush=True)
        return local_root
    raise FileNotFoundError("Dataset root containing manifests/ not found under /kaggle/input/ or ml/datasets/.")


def load_json_manifest(data_root: str, name: str) -> List[Dict[str, Any]]:
    for p in [
        os.path.join(data_root, "manifests", f"{name}.json"),
        os.path.join(data_root, f"{name}.json"),
    ]:
        if os.path.exists(p):
            return json.load(open(p, "r", encoding="utf-8"))["items"]
    raise FileNotFoundError(f"Manifest {name}.json not found in {data_root}")


def resolve_audio_path(data_root: str, rel_path: str) -> str:
    clean = rel_path.replace("\\", "/")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join(data_root, clean),
        os.path.join(data_root, clean.replace("storage/benchmarks/", "")),
        os.path.join(data_root, "benchmarks", clean.replace("storage/benchmarks/", "")),
        os.path.join("/tmp/hf_cache", clean.replace("storage/benchmarks/", "")),
        os.path.join(repo_root, clean),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def ensure_minds14_audio_hydrated(data_root: str, m14_items: List[Dict[str, Any]]) -> None:
    if resolve_audio_path(data_root, m14_items[0]["audio_path"]):
        return
    import io, requests
    import pyarrow.parquet as pq
    import soundfile as sf
    import scipy.signal as sig

    print("[Audio] Hydrating PolyAI/minds14 into /tmp/hf_cache/minds14_en_full...", flush=True)
    needed = {it["id"]: it["audio_path"].replace("\\", "/").replace("storage/benchmarks/", "") for it in m14_items}
    base_api = "https://huggingface.co/api/datasets/PolyAI/minds14/parquet/{cfg}"
    saved = 0
    for acc in ("en-US", "en-GB", "en-AU"):
        pq_map = requests.get(base_api.format(cfg=acc), timeout=30).json()
        for part_idx, u in enumerate(pq_map.get("train", [])):
            r = requests.get(u, timeout=120)
            r.raise_for_status()
            tbl = pq.read_table(io.BytesIO(r.content))
            for r_i in range(tbl.num_rows):
                cls_id = int(tbl["intent_class"][r_i].as_py())
                uid = f"{acc}_{part_idx}_{r_i}_{cls_id}"
                matched_rel = needed.get(uid)
                if matched_rel:
                    out_p = os.path.join("/tmp/hf_cache", matched_rel)
                    os.makedirs(os.path.dirname(out_p), exist_ok=True)
                    a = tbl["audio"][r_i].as_py()
                    raw_b = a.get("bytes") if isinstance(a, dict) else None
                    if raw_b:
                        arr, sr = sf.read(io.BytesIO(raw_b), dtype="float32")
                        if arr.ndim > 1:
                            arr = arr.mean(axis=1)
                        if sr != 16000:
                            arr = sig.resample(arr, int(len(arr) * 16000 / sr)).astype("float32")
                        sf.write(out_p, arr, 16000)
                        saved += 1
        print(f"  [Audio] {acc} ready ({saved}/{len(m14_items)} files)", flush=True)


def get_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "kaggle-svarah", "KAGGLE_SVARAH"):
        tok = os.environ.get(key)
        if tok:
            return tok
    try:
        from kaggle_secrets import UserSecretsClient
        uc = UserSecretsClient()
        for key in ("HF_TOKEN", "kaggle-svarah", "KAGGLE_SVARAH"):
            try:
                val = uc.get_secret(key)
                if val:
                    return val
            except Exception:
                pass
    except Exception:
        pass
    hf_cache_tok = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "token"))
    if os.path.exists(hf_cache_tok):
        return open(hf_cache_tok, "r", encoding="utf-8").read().strip()
    return None


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s'-]", " ", (text or "").lower()).split())


# ==============================================================================
# Experiment 1: MINDS-14 Whisper tiny.en ASR Cache (Checkpointed Every Batch)
# ==============================================================================
def run_asr_cache(ctx: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = ctx["out_dir"]
    data_root = ctx["data_root"]
    m14_items = ctx["m14"]
    resume = ctx["resume"]
    cache_path = os.path.join(out_dir, "minds14_full_tiny_asr_cache.json")

    for pre in [
        cache_path,
        os.path.join(data_root, "minds14_full_tiny_asr_cache.json"),
        os.path.join(data_root, "cached_asr", "minds14_full_tiny_asr_cache.json"),
    ]:
        if resume and os.path.exists(pre):
            data = json.load(open(pre, "r", encoding="utf-8"))
            if len(data) >= len(m14_items):
                print(f"[asr_cache] Resumed complete cache ({len(data)} items) from {pre}", flush=True)
                if pre != cache_path:
                    atomic_json_dump(data, cache_path)
                return {"items": len(data), "cache_path": cache_path, "resumed": True}

    cache: Dict[str, Any] = json.load(open(cache_path, "r", encoding="utf-8")) if (resume and os.path.exists(cache_path)) else {}
    pending = [it for it in m14_items if it["id"] not in cache]
    if not pending:
        return {"items": len(cache), "cache_path": cache_path, "resumed": True}

    ensure_minds14_audio_hydrated(data_root, m14_items)
    import torch
    import whisper

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[asr_cache] Processing {len(pending)}/{len(m14_items)} pending clips on {dev} (already cached={len(cache)})", flush=True)
    wm = whisper.load_model("tiny.en", device=dev)
    t0 = time.time()
    batch_sz = 100
    for start in range(0, len(pending), batch_sz):
        chunk = pending[start : start + batch_sz]
        for it in chunk:
            wav = resolve_audio_path(data_root, it["audio_path"])
            if not wav:
                cache[it["id"]] = {"hyp": it["text"], "words": []}
                continue
            res = wm.transcribe(wav, language="en", beam_size=5, word_timestamps=True, fp16=(dev == "cuda"))
            words = [
                {"word": (w.get("word") or "").strip(), "probability": round(float(w.get("probability", 1.0)), 3)}
                for seg in (res.get("segments") or [])
                for w in (seg.get("words") or [])
            ]
            cache[it["id"]] = {"hyp": (res.get("text") or it["text"]).strip(), "words": words}
        atomic_json_dump(cache, cache_path)
        done = min(start + batch_sz, len(pending))
        print(f"  [asr_cache batch] {done}/{len(pending)} new clips saved ({time.time()-t0:.1f}s)", flush=True)

    return {"items": len(cache), "cache_path": cache_path, "resumed": False}


def _load_asr_cache(ctx: Dict[str, Any]) -> Dict[str, Any]:
    p = os.path.join(ctx["out_dir"], "minds14_full_tiny_asr_cache.json")
    if os.path.exists(p):
        return json.load(open(p, "r", encoding="utf-8"))
    info = run_asr_cache(ctx)
    return json.load(open(info["cache_path"], "r", encoding="utf-8"))


def _patch_peft_torchao() -> None:
    """Bypasses PEFT torchao version check when Kaggle ships stale torchao 0.10.0."""
    try:
        import peft.import_utils as _piu
        import peft.tuners.lora.torchao as _ptao

        _piu.is_torchao_available = lambda *a, **k: False
        _ptao.is_torchao_available = lambda *a, **k: False
        _ptao.dispatch_torchao = lambda *a, **k: None
    except Exception:
        pass


def _asr_batch_predict(pipe: Any, paths: List[str], batch_size: int = 16, **kwargs: Any) -> List[Dict[str, Any]]:
    """Batched ASR pipeline inference with per-item fallback for >30s clips."""
    try:
        return pipe(paths, batch_size=batch_size, **kwargs)
    except RuntimeError as exc:
        print(f"  [asr fallback] Batch collation hit '{exc}'; retrying single-clip mode.", flush=True)
        out = []
        for p in paths:
            out.append(pipe(p, **kwargs))
        return out


# ==============================================================================
# Experiment 2: DistilBERT-LoRA + Optuna + 3 Seeds + Linear Probe (Resumable)
# ==============================================================================
def run_distilbert_lora(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType
    _patch_peft_torchao()
    import optuna
    from sklearn.metrics import accuracy_score, f1_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = ctx["out_dir"]
    resume = ctx["resume"]
    final_path = os.path.join(out_dir, "trained_slu_history.json")
    adapter_dir = os.path.join(out_dir, "slu_lora_adapter")
    ckpt_dir = os.path.join(out_dir, "checkpoints", "distilbert_lora")
    os.makedirs(ckpt_dir, exist_ok=True)

    if resume and os.path.exists(final_path) and os.path.exists(os.path.join(adapter_dir, "label_map.json")):
        print(f"[distilbert_lora] Found completed output at {final_path}; skipping via resume.", flush=True)
        return json.load(open(final_path, "r", encoding="utf-8"))["summary_3seeds"]

    m14, clinc = ctx["m14"], ctx["clinc"]
    asr_cache = _load_asr_cache(ctx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = "distilbert-base-uncased"

    def fmt_uncertain(text: str, words: List[Dict[str, Any]], use_tags: bool = True) -> str:
        c = " ".join((text or "").strip().split())
        if not use_tags or not words:
            return c
        lows = [f"{w['word']}({w['probability']:.2f})" for w in words if w.get("word") and float(w.get("probability", 1.0)) < 0.85]
        return f"{c} [UNCERTAIN: {', '.join(lows[:8])}]" if lows else c

    def inject_syn_noise(text: str, rng: random.Random):
        out_w, meta = [], []
        for w in text.split():
            if len(w) > 4 and rng.random() < 0.12:
                i = rng.randint(1, len(w) - 2)
                nw = w[:i] + w[i + 1 :]
                out_w.append(nw)
                meta.append({"word": nw, "probability": round(rng.uniform(0.35, 0.78), 2)})
            else:
                out_w.append(w)
                meta.append({"word": w, "probability": 0.96})
        return " ".join(out_w), meta

    m14_labels = sorted({x["intent_label"] for x in m14})
    clinc_labels = sorted({x["intent_label"] for x in clinc})
    all_labels = sorted(set(m14_labels) | set(clinc_labels))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}

    rng = random.Random(42)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    tm_tag_x, tm_plain_x, tm_y = [], [], []
    tc_x, tc_y = [], []

    for it in m14:
        tr = asr_cache.get(it["id"], {"hyp": it["text"], "words": []})
        lbl = label2id[it["intent_label"]]
        fx = fmt_uncertain(tr["hyp"], tr.get("words", []), True)
        if it["split"] == "dev":
            tr_x.extend([fx, it["text"]])
            tr_y.extend([lbl, lbl])
        elif it["split"] == "val":
            va_x.append(fx)
            va_y.append(lbl)
        elif it["split"] == "test":
            tm_tag_x.append(fx)
            tm_plain_x.append(tr["hyp"])
            tm_y.append(lbl)

    for it in clinc:
        lbl = label2id[it["intent_label"]]
        if it["split"] == "train":
            ns, mw = inject_syn_noise(it["text"], rng)
            tr_x.extend([fmt_uncertain(ns, mw, True), it["text"]])
            tr_y.extend([lbl, lbl])
        elif it["split"] == "val":
            va_x.append(it["text"])
            va_y.append(lbl)
        elif it["split"] == "test":
            tc_x.append(it["text"])
            tc_y.append(lbl)

    tok = AutoTokenizer.from_pretrained(base_name)

    class DS(Dataset):
        def __init__(self, texts, ys):
            self.enc = tok(texts, truncation=True, padding=True, max_length=72, return_tensors="pt")
            self.y = torch.tensor(ys, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            d = {k: v[idx] for k, v in self.enc.items()}
            d["labels"] = self.y[idx]
            return d

    tr_loader = DataLoader(DS(tr_x, tr_y), batch_size=64, shuffle=True)
    va_loader = DataLoader(DS(va_x, va_y), batch_size=128, shuffle=False)
    n_tr_batches = len(tr_loader)

    def train_one(tag: str, lora_r: int, lr: float, epochs: int, seed: int, use_lora: bool = True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=len(all_labels))
        if use_lora:
            mod = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.1, target_modules=["q_lin", "v_lin"]))
        else:
            for p in base.distilbert.parameters():
                p.requires_grad = False
            mod = base
        mod.to(device)
        opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
        hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
        best_f1, best_sd = -1.0, None
        for ep in range(epochs):
            mod.train()
            tloss = 0.0
            for b_idx, b in enumerate(tr_loader, 1):
                opt.zero_grad()
                out = mod(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), labels=b["labels"].to(device))
                out.loss.backward()
                opt.step()
                tloss += float(out.loss.item()) * len(b["labels"])
                if b_idx % max(1, n_tr_batches // 4) == 0 or b_idx == n_tr_batches:
                    print(f"    [{tag} ep {ep+1}/{epochs}] batch {b_idx}/{n_tr_batches}", flush=True)
            mod.eval()
            vloss, ps, ts = 0.0, [], []
            with torch.no_grad():
                for b in va_loader:
                    out = mod(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), labels=b["labels"].to(device))
                    vloss += float(out.loss.item()) * len(b["labels"])
                    ps.extend(out.logits.argmax(-1).cpu().tolist())
                    ts.extend(b["labels"].tolist())
            vf1 = float(f1_score(ts, ps, average="macro", zero_division=0))
            hist["train_loss"].append(round(tloss / len(tr_y), 4))
            hist["val_loss"].append(round(vloss / len(va_y), 4))
            hist["val_acc"].append(round(float(accuracy_score(ts, ps)), 4))
            hist["val_macro_f1"].append(round(vf1, 4))
            if vf1 > best_f1:
                best_f1 = vf1
                best_sd = {k: v.cpu().clone() for k, v in mod.state_dict().items()}
        if best_sd:
            mod.load_state_dict(best_sd)
        trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return best_f1, {"trainable_params": trainable, "history": hist}, mod

    # Optuna with persistent JSON checkpoint resume
    trials_ckpt = os.path.join(ckpt_dir, "optuna_trials.json")
    trials_log: List[Dict[str, Any]] = json.load(open(trials_ckpt, "r", encoding="utf-8")) if (resume and os.path.exists(trials_ckpt)) else []
    done_trial_ids = {t["trial"] for t in trials_log}

    def obj(trial):
        for prev in trials_log:
            if prev["trial"] == trial.number:
                return prev["val_macro_f1"]
        r = trial.suggest_categorical("lora_r", [8, 16, 32])
        lr = trial.suggest_float("lr", 1e-4, 4e-4, log=True)
        ep = trial.suggest_categorical("epochs", [3, 5])
        f1_v, info_v, _ = train_one(f"Optuna-{trial.number}", r, lr, ep, seed=42)
        trials_log.append({
            "trial": trial.number, "r": r, "lr": round(lr, 6), "epochs": ep,
            "val_macro_f1": round(f1_v, 4), "history": info_v["history"],
        })
        atomic_json_dump(trials_log, trials_ckpt)
        print(f"  [DistilBERT Optuna {trial.number}] r={r} lr={lr:.5f} ep={ep} -> val_F1={f1_v:.4f}", flush=True)
        return f1_v

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    # Enqueue previously evaluated params deterministically if resuming
    for prev in sorted(trials_log, key=lambda x: x["trial"]):
        study.enqueue_trial({"lora_r": prev["r"], "lr": prev["lr"], "epochs": prev["epochs"]})
    study.optimize(obj, n_trials=6)
    best = max(trials_log, key=lambda x: x["val_macro_f1"])
    best_params = {"lora_r": best["r"], "lr": best["lr"], "epochs": best["epochs"]}

    def eval_split(mod, texts, ys, allowed_sub, tau=0.0, tag="eval"):
        loader = DataLoader(DS(texts, ys), batch_size=128, shuffle=False)
        n_ev = len(loader)
        allowed_ids = [label2id[l] for l in allowed_sub]
        mask = np.full(len(all_labels), -1e9, dtype=np.float32)
        mask[allowed_ids] = 0.0
        mask_t = torch.tensor(mask, device=device)
        oos_id = label2id.get("oos")
        mod.eval()
        preds, trues = [], [id2label[y] for y in ys]
        t0 = time.perf_counter()
        with torch.no_grad():
            for b_i, b in enumerate(loader, 1):
                logits = mod(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits + mask_t
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    pi = int(np.argmax(row))
                    if tau > 0.0 and oos_id in allowed_ids and row[pi] < tau:
                        pi = oos_id
                    preds.append(id2label[pi])
                if b_i % max(1, n_ev // 3) == 0 or b_i == n_ev:
                    print(f"    [{tag} eval] batch {b_i}/{n_ev}", flush=True)
        lat_ms = round((time.perf_counter() - t0) * 1000.0 / len(ys), 3)
        out = {
            "accuracy": round(float(accuracy_score(trues, preds)), 4),
            "macro_f1": round(float(f1_score(trues, preds, average="macro", zero_division=0)), 4),
            "gpu_latency_ms": lat_ms,
            "per_item_correct": [1.0 if y == p else 0.0 for y, p in zip(trues, preds)],
        }
        if "oos" in allowed_sub:
            out["oos_f1"] = round(float(f1_score([t == "oos" for t in trues], [p == "oos" for p in preds], zero_division=0)), 4)
        return out

    def cal_tau(mod):
        oos_id = label2id["oos"]
        mod.eval()
        ps = []
        with torch.no_grad():
            for b in va_loader:
                ps.append(torch.softmax(mod(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits, dim=-1).cpu().numpy())
        P = np.vstack(ps)
        rp, mp = P.argmax(1), P.max(1)
        best_t, best_sc = 0.0, float(f1_score(va_y, rp, average="macro", zero_division=0))
        for t in [0.0, 0.35, 0.45, 0.55, 0.65, 0.75]:
            sc = float(f1_score(va_y, np.where(mp < t, oos_id, rp), average="macro", zero_division=0))
            if sc > best_sc:
                best_sc, best_t = sc, t
        return best_t

    seed_runs = []
    for s in [42, 1337, 2026]:
        s_ckpt = os.path.join(ckpt_dir, f"seed_{s}.json")
        s_adp = os.path.join(ckpt_dir, f"adapter_seed_{s}")
        if resume and os.path.exists(s_ckpt) and (s != 42 or os.path.exists(os.path.join(s_adp, "label_map.json"))):
            s_entry = json.load(open(s_ckpt, "r", encoding="utf-8"))
            print(f"  [DistilBERT Seed {s}] Resumed from checkpoint.", flush=True)
        else:
            f1_v, info_v, mod = train_one(f"Seed-{s}", best_params["lora_r"], best_params["lr"], best_params["epochs"], seed=s)
            tau = cal_tau(mod)
            rm_tag = eval_split(mod, tm_tag_x, tm_y, m14_labels, 0.0, tag=f"S{s}-M14Tag")
            rm_pln = eval_split(mod, tm_plain_x, tm_y, m14_labels, 0.0, tag=f"S{s}-M14Plain")
            rcl = eval_split(mod, tc_x, tc_y, clinc_labels, tau, tag=f"S{s}-CLINC5500")
            s_entry = {"seed": s, "info": info_v, "tau": tau, "minds14_conf": rm_tag, "minds14_plain": rm_pln, "clinc_5500": rcl}
            if s == 42:
                os.makedirs(s_adp, exist_ok=True)
                mod.save_pretrained(s_adp)
                tok.save_pretrained(s_adp)
                atomic_json_dump({"label2id": label2id, "oos_threshold": tau}, os.path.join(s_adp, "label_map.json"))
            atomic_json_dump(s_entry, s_ckpt)
            print(f"  [DistilBERT Seed {s}] M14(498) F1={rm_tag['macro_f1']} (noConf={rm_pln['macro_f1']}) | CLINC(5500) F1={rcl['macro_f1']} OOS_F1={rcl['oos_f1']}", flush=True)
        seed_runs.append(s_entry)

    # Save Seed-42 adapter cleanly to final location
    shutil.copytree(os.path.join(ckpt_dir, "adapter_seed_42"), adapter_dir, dirs_exist_ok=True)

    # Linear probe checkpointed independently
    probe_ckpt = os.path.join(ckpt_dir, "linear_probe.json")
    if resume and os.path.exists(probe_ckpt):
        probe_res = json.load(open(probe_ckpt, "r", encoding="utf-8"))
        print("  [Linear Probe] Resumed from checkpoint.", flush=True)
    else:
        _, pinfo, pmod = train_one("LinearProbe", 8, best_params["lr"], best_params["epochs"], seed=42, use_lora=False)
        pm14 = eval_split(pmod, tm_tag_x, tm_y, m14_labels, 0.0, tag="Probe-M14")
        pcl = eval_split(pmod, tc_x, tc_y, clinc_labels, seed_runs[0]["tau"], tag="Probe-CLINC")
        probe_res = {"trainable_params": pinfo["trainable_params"], "minds14": pm14, "clinc_5500": pcl}
        atomic_json_dump(probe_res, probe_ckpt)

    res = {
        "experiment": "trained_cross_layer_lora_slu_kaggle_v2",
        "base_model": base_name,
        "license": "Apache-2.0",
        "optuna_trials": trials_log,
        "best_params": best_params,
        "seed_runs": seed_runs,
        "summary_3seeds": {
            "minds14_conf_f1_mean": round(float(np.mean([r["minds14_conf"]["macro_f1"] for r in seed_runs])), 4),
            "minds14_conf_f1_std": round(float(np.std([r["minds14_conf"]["macro_f1"] for r in seed_runs], ddof=1)), 4),
            "minds14_plain_f1_mean": round(float(np.mean([r["minds14_plain"]["macro_f1"] for r in seed_runs])), 4),
            "clinc_f1_mean": round(float(np.mean([r["clinc_5500"]["macro_f1"] for r in seed_runs])), 4),
            "clinc_f1_std": round(float(np.std([r["clinc_5500"]["macro_f1"] for r in seed_runs], ddof=1)), 4),
            "clinc_oos_f1_mean": round(float(np.mean([r["clinc_5500"]["oos_f1"] for r in seed_runs])), 4),
            "clinc_oos_f1_std": round(float(np.std([r["clinc_5500"]["oos_f1"] for r in seed_runs], ddof=1)), 4),
        },
        "linear_probe_baseline": probe_res,
    }
    atomic_json_dump(res, final_path)
    return res["summary_3seeds"]


# ==============================================================================
# Experiment 3: Whisper-large-v3-turbo (MIT) Batched GPU ASR (Resumable)
# ==============================================================================
def run_whisper_turbo(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import jiwer
    from transformers import pipeline

    out_dir = ctx["out_dir"]
    data_root = ctx["data_root"]
    resume = ctx["resume"]
    final_path = os.path.join(out_dir, "kaggle_asr_turbo.json")
    ckpt_path = os.path.join(out_dir, "checkpoints", "whisper_turbo_preds.json")

    if resume and os.path.exists(final_path):
        print(f"[whisper_turbo] Resumed complete output from {final_path}", flush=True)
        return json.load(open(final_path, "r", encoding="utf-8"))["splits"]

    ensure_minds14_audio_hydrated(data_root, ctx["m14"])
    ckpt: Dict[str, Any] = json.load(open(ckpt_path, "r", encoding="utf-8")) if (resume and os.path.exists(ckpt_path)) else {"m14": {}, "iv": {}}

    m14_test = [(it, resolve_audio_path(data_root, it["audio_path"])) for it in ctx["m14"] if it["split"] == "test"]
    m14_pending = [(it, w) for it, w in m14_test if w and it["id"] not in ckpt["m14"]]
    iv_test = [(it, resolve_audio_path(data_root, it["audio_path"])) for it in ctx["iv"] if it["split"] == "test"]
    iv_pending = [(it, w) for it, w in iv_test if w and it["id"] not in ckpt["iv"]]

    if m14_pending or iv_pending:
        dev = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        asr_turbo = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=dtype,
            device=dev,
            chunk_length_s=30,
        )
        bs = 16
        for tag, pending_list, store_key in [("m14_498", m14_pending, "m14"), ("iv_50", iv_pending, "iv")]:
            for s_idx in range(0, len(pending_list), bs * 4):
                sub = pending_list[s_idx : s_idx + bs * 4]
                outs = _asr_batch_predict(
                    asr_turbo,
                    [w for _, w in sub],
                    batch_size=bs,
                    generate_kwargs={"language": "english"},
                )
                for (it, _), pr in zip(sub, outs):
                    ckpt[store_key][it["id"]] = {"ref": it["text"], "hyp": pr["text"], "accent": it.get("accent", "en")}
                atomic_json_dump(ckpt, ckpt_path)
                print(f"  [whisper_turbo {tag}] processed {min(s_idx + len(sub), len(pending_list))}/{len(pending_list)}", flush=True)

    refs, hyps, accs = [], [], {}
    for _, rec in ckpt["m14"].items():
        refs.append(_norm(rec["ref"]) or "<empty>")
        hyps.append(_norm(rec["hyp"]))
        accs.setdefault(rec.get("accent", "en"), []).append((refs[-1], hyps[-1]))

    splits: Dict[str, Any] = {
        "minds14_full_498": {
            "wer": round(float(jiwer.wer(refs, hyps)), 4) if refs else None,
            "cer": round(float(jiwer.cer(refs, hyps)), 4) if refs else None,
            "num_samples": len(refs),
            "accent_wer": {k: round(float(jiwer.wer([x[0] for x in v], [x[1] for x in v])), 4) for k, v in accs.items()},
        }
    }
    if ckpt["iv"]:
        ir = [_norm(v["ref"]) or "<empty>" for v in ckpt["iv"].values()]
        ih = [_norm(v["hyp"]) for v in ckpt["iv"].values()]
        splits["interview_stress_50"] = {"wer": round(float(jiwer.wer(ir, ih)), 4), "num_samples": len(ir)}

    res = {"model": "openai/whisper-large-v3-turbo", "license": "MIT", "splits": splits}
    atomic_json_dump(res, final_path)
    return splits


# ==============================================================================
# Experiment 4: Gated Real en-IN ai4bharat/Svarah (Completely Isolated)
# ==============================================================================
def run_svarah(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import jiwer
    import soundfile as sf
    from datasets import load_dataset
    from transformers import pipeline

    out_dir = ctx["out_dir"]
    resume = ctx["resume"]
    final_path = os.path.join(out_dir, "svarah_en_in_eval.json")
    ckpt_path = os.path.join(out_dir, "checkpoints", "svarah_ckpt.json")

    if resume and os.path.exists(final_path):
        print(f"[svarah] Resumed complete output from {final_path}", flush=True)
        return json.load(open(final_path, "r", encoding="utf-8"))

    hf_tok = get_hf_token()
    if not hf_tok:
        raise RuntimeError("No HF_TOKEN or kaggle-svarah secret found; isolated skip without touching other experiments.")

    dev = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr_turbo = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=dev,
        chunk_length_s=30,
    )
    asr_small = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",
        torch_dtype=dtype,
        device=dev,
        chunk_length_s=30,
    )

    done_rows: List[Dict[str, str]] = json.load(open(ckpt_path, "r", encoding="utf-8")) if (resume and os.path.exists(ckpt_path)) else []
    start_idx = len(done_rows)
    target_n = 1000
    if start_idx < target_n:
        ds = load_dataset("ai4bharat/Svarah", split="test", token=hf_tok, streaming=True)
        batch_buf, ref_buf = [], []
        with tempfile.TemporaryDirectory() as td:
            for idx, row in enumerate(ds):
                if idx < start_idx:
                    continue
                if idx >= target_n:
                    break
                wp = os.path.join(td, f"sv_{idx}.wav")
                aud = row.get("audio") or row.get("audio_filepath")
                if isinstance(aud, dict):
                    sf.write(wp, aud["array"], aud["sampling_rate"])
                elif hasattr(aud, "get_all_samples"):
                    samples = aud.get_all_samples()
                    arr = samples.data.mean(dim=0).cpu().numpy() if samples.data.ndim > 1 else samples.data.cpu().numpy()
                    sf.write(wp, arr, int(samples.sample_rate))
                else:
                    raise RuntimeError(f"Unsupported Svarah audio object type: {type(aud)}")
                batch_buf.append(wp)
                ref_buf.append(str(row.get("text", "")))
                if len(batch_buf) == 64 or idx + 1 == target_n:
                    pt = _asr_batch_predict(asr_turbo, batch_buf, batch_size=16, generate_kwargs={"language": "english"})
                    ps = _asr_batch_predict(asr_small, batch_buf, batch_size=16)
                    for rf, t_o, s_o in zip(ref_buf, pt, ps):
                        done_rows.append({"ref": rf, "turbo": t_o["text"], "small": s_o["text"]})
                    atomic_json_dump(done_rows, ckpt_path)
                    print(f"  [svarah batch] {len(done_rows)}/{target_n} clips evaluated", flush=True)
                    batch_buf.clear()
                    ref_buf.clear()

    refs = [_norm(r["ref"]) or "<empty>" for r in done_rows]
    res = {
        "dataset": "ai4bharat/Svarah",
        "license": "CC-BY-4.0",
        "num_samples": len(refs),
        "whisper_small_en_wer": round(float(jiwer.wer(refs, [_norm(r["small"]) for r in done_rows])), 4),
        "whisper_large_v3_turbo_wer": round(float(jiwer.wer(refs, [_norm(r["turbo"]) for r in done_rows])), 4),
    }
    atomic_json_dump(res, final_path)
    return res


# ==============================================================================
# Experiment 5: Apache-2.0 Qwen/Qwen3-4B NF4 QLoRA (Epoch & Batch Resumable)
# ==============================================================================
def run_qwen3_4b_qlora(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from sklearn.metrics import accuracy_score, f1_score

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for 4-bit Qwen3-4B NF4 QLoRA.")

    out_dir = ctx["out_dir"]
    resume = ctx["resume"]
    final_path = os.path.join(out_dir, "qwen3_4b_qlora_results.json")
    adapter_dir = os.path.join(out_dir, "qwen3_4b_slu_lora")
    ckpt_dir = os.path.join(out_dir, "checkpoints", "qwen3_4b")
    os.makedirs(ckpt_dir, exist_ok=True)

    if resume and os.path.exists(final_path):
        print(f"[qwen3_4b_qlora] Resumed complete output from {final_path}", flush=True)
        res = json.load(open(final_path, "r", encoding="utf-8"))
        return {"zero_shot_m14": res["zero_shot_minds14_498"]["macro_f1"], "qlora_m14": res["qlora_minds14_498_noisy"]["macro_f1"]}

    m14, clinc = ctx["m14"], ctx["clinc"]
    asr_cache = _load_asr_cache(ctx)
    model_id = "Qwen/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, torch_dtype=torch.float16, device_map="auto")

    m14_labels = sorted({x["intent_label"] for x in m14})
    clinc_15 = sorted(["oos", "change_language", "definition", "find_phone", "fun_fact", "improve_credit_score", "insurance_change", "meaning_of_life", "payday", "pto_request", "replacement_card_duration", "timer", "transfer", "translate", "travel_alert"])
    sys_msg = 'Classify the user utterance into one allowed intent. Return ONLY compact JSON: {"intent": "<label>", "confidence": <0-1>}. No prose.'

    def build_prompt(text: str, domain: str, opts: List[str], words: Optional[List[Dict[str, Any]]] = None) -> str:
        lows = [w["word"] for w in (words or []) if float(w.get("probability", 1.0)) < 0.85]
        unc = f" [UNCERTAIN: {', '.join(lows[:5])}]" if lows else ""
        user_msg = f"Domain: {domain}\nAllowed: {', '.join(opts)}\nUtterance: {text}{unc}"
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def infer_batched(mod_eval, items, domain, opts, tag: str, is_m14: bool = True, batch_size: int = 16) -> Dict[str, Any]:
        c_file = os.path.join(ckpt_dir, f"infer_{tag}.json")
        cached = json.load(open(c_file, "r", encoding="utf-8")) if (resume and os.path.exists(c_file)) else {}
        mod_eval.eval()
        if hasattr(mod_eval, "config"):
            mod_eval.config.use_cache = True
        tok.padding_side = "left"
        pending = [it for it in items if it["id"] not in cached]
        t_ev = time.time()
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompts = []
            for it in batch:
                tr = asr_cache.get(it["id"], {"hyp": it["text"], "words": []}) if is_m14 else {"hyp": it["text"], "words": []}
                prompts.append(build_prompt(tr["hyp"], domain, opts, tr.get("words") if is_m14 else None))
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=224).to("cuda")
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                gens = mod_eval.generate(
                    **enc,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            for it, gen in zip(batch, gens):
                out_txt = tok.decode(gen[in_len:], skip_special_tokens=True).strip()
                pred, ok = "__invalid__", False
                try:
                    m = re.search(r"\{.*\}", out_txt, re.S)
                    obj = json.loads(m.group(0) if m else out_txt)
                    cand = str(obj.get("intent", "")).strip().lower().replace(" ", "_")
                    if cand in opts:
                        pred, ok = cand, True
                except Exception:
                    pass
                cached[it["id"]] = {"gold": it["intent_label"], "pred": pred, "schema_ok": ok}
            atomic_json_dump(cached, c_file)
            print(f"    [{tag}] inferred {min(start + len(batch), len(pending))}/{len(pending)} ({time.time()-t_ev:.1f}s)", flush=True)

        y_true = [cached[it["id"]]["gold"] for it in items]
        y_pred = [cached[it["id"]]["pred"] for it in items]
        sch_ok = [1.0 if cached[it["id"]]["schema_ok"] else 0.0 for it in items]
        out = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "schema_valid_rate": round(float(sum(sch_ok) / len(sch_ok)), 4),
            "num_samples": len(y_true),
            "per_item_correct": [1.0 if y == p else 0.0 for y, p in zip(y_true, y_pred)],
        }
        if "oos" in opts:
            out["oos_f1"] = round(float(f1_score([y == "oos" for y in y_true], [p == "oos" for p in y_pred], zero_division=0)), 4)
        return out

    m14_test = [x for x in m14 if x["split"] == "test"]
    m14_val = [x for x in m14 if x["split"] == "val"]
    by_cl = {}
    for x in clinc:
        if x["split"] == "test" and x["intent_label"] in clinc_15:
            by_cl.setdefault(x["intent_label"], []).append(x)
    cl_test = [it for lbl in clinc_15 for it in by_cl.get(lbl, [])[: (30 if lbl == "oos" else 9)]]

    zs_m14 = infer_batched(base, m14_test, "banking", m14_labels, tag="zs_m14_498", is_m14=True)
    zs_cl = infer_batched(base, cl_test, "assistant", clinc_15, tag="zs_clinc_15", is_m14=False)
    print(f"  [Qwen3-4B Zero-Shot] M14={zs_m14['macro_f1']} SchV={zs_m14['schema_valid_rate']} | CLINC={zs_cl['macro_f1']} OOS={zs_cl['oos_f1']}", flush=True)

    # Resumable QLoRA epoch checkpoints
    ep_state_path = os.path.join(ckpt_dir, "train_state.json")
    t_state = json.load(open(ep_state_path, "r", encoding="utf-8")) if (resume and os.path.exists(ep_state_path)) else {"completed_epochs": 0, "losses": [], "val_f1s": []}
    last_ep = int(t_state["completed_epochs"])
    total_epochs = 2

    tok.padding_side = "right"
    train_samples = []
    for it in m14:
        if it["split"] == "dev":
            tr = asr_cache.get(it["id"], {"hyp": it["text"], "words": []})
            train_samples.append((build_prompt(tr["hyp"], "banking", m14_labels, tr.get("words")), json.dumps({"intent": it["intent_label"], "confidence": 0.95})))
            train_samples.append((build_prompt(it["text"], "banking", m14_labels), json.dumps({"intent": it["intent_label"], "confidence": 0.95})))
    rng = random.Random(42)
    cl_tr_by_lbl = {}
    for it in clinc:
        if it["split"] == "train":
            cl_tr_by_lbl.setdefault(it["intent_label"], []).append(it)
    for lbl, lst in sorted(cl_tr_by_lbl.items()):
        rng.shuffle(lst)
        opts = clinc_15 if lbl in clinc_15 else sorted(set(clinc_15) | {lbl})
        for it in lst[: (30 if lbl == "oos" else 6)]:
            train_samples.append((build_prompt(it["text"], "assistant", opts), json.dumps({"intent": lbl, "confidence": 0.95})))
    rng.shuffle(train_samples)

    class CausalSLUDS(Dataset):
        def __init__(self, pairs, max_len: int = 240):
            self.rows = []
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            for pr, tgt in pairs:
                p_ids = tok(pr, add_special_tokens=False)["input_ids"]
                t_ids = tok(tgt + tok.eos_token, add_special_tokens=False)["input_ids"]
                # Keep full target intact; truncate prompt from left if unusually long
                if len(p_ids) + len(t_ids) > max_len:
                    keep_p = max(16, max_len - len(t_ids))
                    p_ids = p_ids[:keep_p]
                raw_ids = (p_ids + t_ids)[:max_len]
                lbl_raw = ([-100] * len(p_ids) + t_ids)[:max_len]
                pad_n = max_len - len(raw_ids)
                ids = torch.tensor(raw_ids + [pad_id] * pad_n, dtype=torch.long)
                attn = torch.tensor([1] * len(raw_ids) + [0] * pad_n, dtype=torch.long)
                labels = torch.tensor(lbl_raw + [-100] * pad_n, dtype=torch.long)
                self.rows.append((ids, attn, labels))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    ep_ckpt_dir = os.path.join(ckpt_dir, f"adapter_ep_{last_ep}")
    base = prepare_model_for_kbit_training(base)
    if resume and last_ep > 0 and os.path.exists(ep_ckpt_dir):
        mod = PeftModel.from_pretrained(base, ep_ckpt_dir, is_trainable=True)
        print(f"  [Qwen3-4B QLoRA] Resumed adapter weights cleanly from Epoch {last_ep}", flush=True)
    else:
        mod = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)

    if last_ep < total_epochs:
        loader = DataLoader(CausalSLUDS(train_samples), batch_size=8, shuffle=True)
        n_batches = len(loader)
        opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
        for ep in range(last_ep, total_epochs):
            mod.train()
            if hasattr(mod, "config"):
                mod.config.use_cache = False
            tot_l = 0.0
            for b_idx, (b_ids, b_att, b_lbl) in enumerate(loader, 1):
                opt.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = mod(input_ids=b_ids.cuda(), attention_mask=b_att.cuda(), labels=b_lbl.cuda())
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in mod.parameters() if p.requires_grad], 1.0)
                opt.step()
                tot_l += float(out.loss.item()) * len(b_ids)
                if b_idx % max(1, n_batches // 5) == 0 or b_idx == n_batches:
                    print(f"    [Qwen3-4B ep {ep+1}/{total_epochs}] batch {b_idx}/{n_batches} | avg_loss={tot_l/(b_idx*8):.4f}", flush=True)
            ep_l = round(tot_l / len(train_samples), 4)
            t_state["losses"].append(ep_l)
            val_sub = m14_val[:112]
            val_ep = infer_batched(mod, val_sub, "banking", m14_labels, tag=f"val_ep_{ep+1}", is_m14=True) if val_sub else {"macro_f1": 0.0}
            t_state.setdefault("val_f1s", []).append(val_ep["macro_f1"])
            t_state["completed_epochs"] = ep + 1
            ep_save = os.path.join(ckpt_dir, f"adapter_ep_{ep+1}")
            os.makedirs(ep_save, exist_ok=True)
            mod.save_pretrained(ep_save)
            atomic_json_dump(t_state, ep_state_path)
            print(f"  [Qwen3-4B QLoRA Epoch {ep+1}/{total_epochs}] checkpointed (loss={ep_l} val_F1={val_ep['macro_f1']})", flush=True)

    os.makedirs(adapter_dir, exist_ok=True)
    mod.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    m14_res = infer_batched(mod, m14_test, "banking", m14_labels, tag="qlora_m14_498", is_m14=True)
    cl_res = infer_batched(mod, cl_test, "assistant", clinc_15, tag="qlora_clinc_15", is_m14=False)
    print(f"  [Qwen3-4B QLoRA TEST] M14={m14_res['macro_f1']} SchV={m14_res['schema_valid_rate']} | CLINC={cl_res['macro_f1']} OOS={cl_res['oos_f1']}", flush=True)

    out_q = {
        "model": model_id,
        "license": "Apache-2.0",
        "trainable_params": trainable,
        "train_samples": len(train_samples),
        "epochs": total_epochs,
        "train_losses": t_state["losses"],
        "val_f1s": t_state.get("val_f1s", []),
        "zero_shot_minds14_498": zs_m14,
        "zero_shot_clinc_15": zs_cl,
        "qlora_minds14_498_noisy": m14_res,
        "qlora_clinc_15_test": cl_res,
    }
    atomic_json_dump(out_q, final_path)
    return {"zero_shot_m14": zs_m14["macro_f1"], "qlora_m14": m14_res["macro_f1"], "qlora_clinc": cl_res["macro_f1"]}


# ==============================================================================
# Fault-Isolated Experiment Orchestrator
# ==============================================================================
@dataclass
class ExperimentSpec:
    name: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    output_file: str


EXPERIMENTS = [
    ExperimentSpec("asr_cache", run_asr_cache, "minds14_full_tiny_asr_cache.json"),
    ExperimentSpec("distilbert_lora", run_distilbert_lora, "trained_slu_history.json"),
    ExperimentSpec("whisper_turbo", run_whisper_turbo, "kaggle_asr_turbo.json"),
    ExperimentSpec("svarah", run_svarah, "svarah_en_in_eval.json"),
    ExperimentSpec("qwen3_4b_qlora", run_qwen3_4b_qlora, "qwen3_4b_qlora_results.json"),
]


def main():
    parser = argparse.ArgumentParser(description="Fault-isolated, checkpointed & resumable GPU experiment suite.")
    parser.add_argument("--only", nargs="*", default=None, help=f"Run subset from: {[e.name for e in EXPERIMENTS]}")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and re-run from scratch")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output/checkpoint root directory")
    parser.add_argument("--skip-pip", action="store_true", help="Skip runtime pip install (when running locally)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.skip_pip and os.path.exists("/kaggle"):
        install_deps()

    data_root = locate_data_root()
    ctx = {
        "out_dir": args.out_dir,
        "data_root": data_root,
        "resume": not args.no_resume,
        "m14": load_json_manifest(data_root, "minds14_en_full"),
        "clinc": load_json_manifest(data_root, "clinc_oos_full"),
        "iv": load_json_manifest(data_root, "interview_stress_v1"),
    }

    status_path = os.path.join(args.out_dir, "experiment_status.json")
    status_log: Dict[str, Any] = json.load(open(status_path, "r", encoding="utf-8")) if (ctx["resume"] and os.path.exists(status_path)) else {}

    selected = [e for e in EXPERIMENTS if (args.only is None or e.name in args.only)]
    for spec in selected:
        print(f"\n==================== [{spec.name}] ====================", flush=True)
        t0 = time.time()
        try:
            summary = spec.fn(ctx)
            status_log[spec.name] = {
                "status": "completed",
                "wall_sec": round(time.time() - t0, 1),
                "output_file": spec.output_file,
                "summary": summary,
            }
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR in {spec.name}] {err_msg} (continuing cleanly to next experiment!)", flush=True)
            traceback.print_exc()
            status_log[spec.name] = {
                "status": "failed",
                "wall_sec": round(time.time() - t0, 1),
                "error": err_msg[:300],
            }
        finally:
            free_memory()
            atomic_json_dump(status_log, status_path)

    print("\n=== Experiment Summary ===", flush=True)
    print(json.dumps(status_log, indent=2), flush=True)


if __name__ == "__main__":
    main()
