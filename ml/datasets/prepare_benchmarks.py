"""
Session 4 Benchmark Preparation & Leakage-Free Split Builder.
Curates 4 compact, license-clean benchmark manifests (<150 MB total audio):
  1. librispeech_60.json   (openslr/librispeech_asr test.clean, 12 speakers x 5 utts)
  2. minds14_en_140.json   (PolyAI/minds14 en-US/GB/AU, 14 intents, 35/15/50 split)
  3. clinc_oos_200.json    (clinc/clinc_oos small, dev/val/test text intents + OOS)
  4. interview_stress_v1.json (65 domain/South-Asian proper-noun & QA audio turns via EdgeTTS)
"""

import os
import sys
import json
import time
import wave
import hashlib
import random
from typing import Dict, Any, List, Optional
import numpy as np
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MANIFEST_DIR = os.path.join(ROOT_DIR, "ml", "datasets", "manifests")
AUDIO_CACHE_DIR = os.path.join(ROOT_DIR, "storage", "benchmarks")


def text_sha256(text: str) -> str:
    """Normalized SHA-256 hash used to enforce zero-overlap split deduplication."""
    norm = " ".join((text or "").lower().strip().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def add_awgn(audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Adds deterministic Additive White Gaussian Noise calibrated to target SNR (dB)."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    rng = np.random.RandomState(seed)
    sig_power = float(np.mean(wav ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return np.clip(wav + noise, -1.0, 1.0)


def load_manifest(name: str) -> Dict[str, Any]:
    """Loads a manifest JSON from ml/datasets/manifests/."""
    path = os.path.join(MANIFEST_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _download_file(url: str, dest_path: str) -> str:
    """Downloads URL to dest_path if not already cached on disk."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, timeout=45) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
    return dest_path


def _hf_rows(dataset: str, config: str, split: str, offset: int = 0, length: int = 100) -> List[Dict[str, Any]]:
    url = f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
    for attempt in range(5):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            time.sleep(0.35)
            return [r["row"] for r in resp.json().get("rows", [])]
        wait_s = int(resp.headers.get("Retry-After", 4 * (attempt + 1))) if resp.status_code == 429 else 2 * (attempt + 1)
        time.sleep(min(wait_s, 20))
    resp.raise_for_status()
    return []


def _cached_manifest_if_complete(name: str, check_audio: bool = True) -> Optional[Dict[str, Any]]:
    """Returns existing manifest immediately if all referenced audio files already exist locally."""
    path = os.path.join(MANIFEST_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    try:
        data = load_manifest(name)
        items = data.get("items", [])
        if not items:
            return None
        if check_audio and not all(os.path.exists(os.path.join(ROOT_DIR, it["audio_path"])) for it in items if "audio_path" in it):
            return None
        return data
    except Exception:
        return None


def _audio_duration(path: str) -> float:
    import soundfile as sf
    return round(float(sf.info(path).duration), 2)


def _save_manifest(name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    out = os.path.join(MANIFEST_DIR, f"{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def build_librispeech_60(seed: int = 42, num_speakers: int = 12, utts_per_spk: int = 5) -> Dict[str, Any]:
    """Curates 60 LibriSpeech test-clean utterances across 12 unseen speakers (5 each: enroll/val/3*test)."""
    cached = _cached_manifest_if_complete("librispeech_60", check_audio=True)
    if cached is not None:
        return cached
    cache_dir = os.path.join(AUDIO_CACHE_DIR, "librispeech_60")
    existing_probe = os.path.join(AUDIO_CACHE_DIR, "librispeech_speaker_probe")
    os.makedirs(cache_dir, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    seen_hashes = set()
    offset = 0
    while len([s for s, v in buckets.items() if len(v) >= utts_per_spk]) < num_speakers and offset < 900:
        rows = _hf_rows("openslr/librispeech_asr", "clean", "test", offset=offset, length=100)
        if not rows:
            break
        for row in rows:
            spk = str(row["speaker_id"])
            h = text_sha256(row["text"])
            if h in seen_hashes:
                continue
            bucket = buckets.setdefault(spk, [])
            if len(bucket) < utts_per_spk:
                seen_hashes.add(h)
                fname = f"spk_{spk}_{row['id']}.flac"
                local_path = os.path.join(cache_dir, fname)
                # Reuse Session 2 cached FLAC cleanly if present before hitting network
                probe_path = os.path.join(existing_probe, fname)
                if not os.path.exists(local_path) and os.path.exists(probe_path):
                    import shutil
                    shutil.copyfile(probe_path, local_path)
                else:
                    _download_file(row["audio"][0]["src"], local_path)

                split_role = "enroll" if len(bucket) == 0 else ("val" if len(bucket) == 1 else "test")
                bucket.append({
                    "id": str(row["id"]),
                    "speaker_id": spk,
                    "split": split_role,
                    "audio_path": os.path.relpath(local_path, ROOT_DIR).replace("\\", "/"),
                    "text": row["text"],
                    "text_hash": h,
                    "duration_sec": _audio_duration(local_path),
                })
        offset += 100

    items = []
    for spk in sorted(buckets.keys()):
        if len(buckets[spk]) >= utts_per_spk:
            items.extend(buckets[spk][:utts_per_spk])
            if len(items) >= num_speakers * utts_per_spk:
                break

    payload = {
        "name": "librispeech_60",
        "source": "openslr/librispeech_asr (clean/test)",
        "license": "CC-BY-4.0",
        "seed": seed,
        "total_items": len(items),
        "num_speakers": len({x["speaker_id"] for x in items}),
        "split_counts": {s: sum(1 for x in items if x["split"] == s) for s in ["enroll", "val", "test"]},
        "items": items,
    }
    _save_manifest("librispeech_60", payload)
    return payload


def build_minds14_140(seed: int = 42, target_per_intent: int = 10) -> Dict[str, Any]:
    """Curates 140 deduplicated MINDS-14 spoken intent clips balanced across 14 classes & 3 English accents."""
    cached = _cached_manifest_if_complete("minds14_en_140", check_audio=True)
    if cached is not None and cached.get("total_items", 0) >= 14 * target_per_intent:
        return cached
    cache_dir = os.path.join(AUDIO_CACHE_DIR, "minds14_en_140")
    os.makedirs(cache_dir, exist_ok=True)
    info = requests.get("https://datasets-server.huggingface.co/info?dataset=PolyAI/minds14&config=en-US", timeout=15).json()
    label_names = info["dataset_info"]["features"]["intent_class"]["names"]

    rng = random.Random(seed)
    pools: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(label_names))}
    seen_hashes = set()

    for accent in ["en-US", "en-GB", "en-AU"]:
        for off in [0, 100, 200, 300, 400, 500]:
            batch = _hf_rows("PolyAI/minds14", accent, "train", offset=off, length=100)
            for idx_in_batch, row in enumerate(batch):
                cls_id = int(row["intent_class"])
                h = text_sha256(row["transcription"])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                pools[cls_id].append({
                    "uid": f"{accent}_{off + idx_in_batch}_{cls_id}",
                    "accent": accent,
                    "intent_id": cls_id,
                    "intent_label": label_names[cls_id],
                    "transcription": row["transcription"],
                    "text_hash": h,
                    "audio_url": row["audio"][0]["src"],
                })
            if len(batch) < 100:
                break

    # Select balanced set (~10 per intent) mixing accents, then assign stratified 30/20/50 splits
    selected = []
    for cls_id in sorted(pools.keys()):
        candidates = pools[cls_id]
        rng.shuffle(candidates)
        chosen = candidates[:target_per_intent]
        # Per 10 items: 3 dev (30%), 2 val (20%), 5 test (50%)
        n_c = len(chosen)
        n_dev = max(1, int(n_c * 0.35))
        n_val = max(1, round(n_c * 0.15))
        for k, row in enumerate(chosen):
            split = "dev" if k < n_dev else ("val" if k < n_dev + n_val else "test")
            local_wav = os.path.join(cache_dir, f"{row['uid']}.wav")
            _download_file(row["audio_url"], local_wav)
            selected.append({
                "id": row["uid"],
                "accent": row["accent"],
                "intent_label": row["intent_label"],
                "split": split,
                "audio_path": os.path.relpath(local_wav, ROOT_DIR).replace("\\", "/"),
                "text": row["transcription"],
                "text_hash": row["text_hash"],
                "duration_sec": _audio_duration(local_wav),
            })

    payload = {
        "name": "minds14_en_140",
        "source": "PolyAI/minds14 (en-US, en-GB, en-AU)",
        "license": "CC-BY-4.0",
        "seed": seed,
        "num_intents": len(label_names),
        "intent_labels": label_names,
        "total_items": len(selected),
        "split_counts": {s: sum(1 for x in selected if x["split"] == s) for s in ["dev", "val", "test"]},
        "items": selected,
    }
    _save_manifest("minds14_en_140", payload)
    return payload


def build_clinc_oos_200(seed: int = 42, target_intents: int = 16) -> Dict[str, Any]:
    """Curates a 200-sample disjoint validation/test slice (15 frequent intents + oos) from clinc/clinc_oos."""
    cached = _cached_manifest_if_complete("clinc_oos_200", check_audio=False)
    if cached is not None:
        return cached
    info = requests.get("https://datasets-server.huggingface.co/info?dataset=clinc/clinc_oos&config=small", timeout=15).json()
    names = info["dataset_info"]["features"]["intent"]["names"]
    oos_idx = names.index("oos")

    # Fetch representative pages across validation (incl. offset=3000 for 'oos') and test (incl. offset=4500 for 'oos')
    val_rows = []
    for off in [0, 100, 200, 3000]:
        val_rows.extend(_hf_rows("clinc/clinc_oos", "small", "validation", offset=off, length=100))
    test_rows = []
    for off in [0, 100, 200, 300, 4500]:
        test_rows.extend(_hf_rows("clinc/clinc_oos", "small", "test", offset=off, length=100))

    # Pick common intents represented in both splits plus 'oos'
    val_by_id: Dict[int, List[Dict[str, Any]]] = {}
    test_by_id: Dict[int, List[Dict[str, Any]]] = {}
    for r in val_rows:
        val_by_id.setdefault(int(r["intent"]), []).append(r)
    for r in test_rows:
        test_by_id.setdefault(int(r["intent"]), []).append(r)

    shared_ids = [i for i in val_by_id if i != oos_idx and len(val_by_id[i]) >= 5 and len(test_by_id.get(i, [])) >= 5]
    rng = random.Random(seed)
    chosen_ids = ([oos_idx] if (oos_idx in val_by_id and oos_idx in test_by_id) else []) + sorted(
        rng.sample(shared_ids, min(target_intents - 1, len(shared_ids)))
    )

    items = []
    seen_hashes = set()
    # Select ~6-7 items per class for val (~100 total) and test (~100 total)
    for cls_id in chosen_ids:
        label = names[cls_id]
        v_list = list(val_by_id[cls_id])
        t_list = list(test_by_id[cls_id])
        rng.shuffle(v_list)
        rng.shuffle(t_list)

        v_added = 0
        for row in v_list:
            h = text_sha256(row["text"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            items.append({"id": f"val_{cls_id}_{v_added}", "split": "val", "intent_label": label, "text": row["text"], "text_hash": h})
            v_added += 1
            if len([x for x in items if x["split"] == "val"]) >= 100 or v_added >= 7:
                break

        t_added = 0
        for row in t_list:
            h = text_sha256(row["text"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            items.append({"id": f"test_{cls_id}_{t_added}", "split": "test", "intent_label": label, "text": row["text"], "text_hash": h})
            t_added += 1
            if len([x for x in items if x["split"] == "test"]) >= 100 or t_added >= 7:
                break

    chosen_labels = [names[i] for i in chosen_ids]
    payload = {
        "name": "clinc_oos_200",
        "source": "clinc/clinc_oos (small: validation/test)",
        "license": "CC-BY-3.0",
        "seed": seed,
        "num_intents": len(chosen_labels),
        "intent_labels": chosen_labels,
        "total_items": len(items),
        "split_counts": {s: sum(1 for x in items if x["split"] == s) for s in ["val", "test"]},
        "items": items,
    }
    _save_manifest("clinc_oos_200", payload)
    return payload


def _render_edgetts_wav(text: str, voice: str, out_wav: str) -> float:
    """Renders text using external Azure EdgeTTS and converts to 16 kHz mono WAV."""
    from services.tts import EdgeTTSService
    from services.tts.voice_cloner import VoiceCloner

    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 500:
        return _audio_duration(out_wav)

    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    tmp_mp3 = out_wav + ".tmp.mp3"
    svc = EdgeTTSService(voice=voice)
    res = svc.synthesize(text, output_path=tmp_mp3)
    if res.status != "success" or not os.path.exists(tmp_mp3):
        raise RuntimeError(f"EdgeTTS failed rendering benchmark audio: {res.error}")

    audio, sr = VoiceCloner.load_audio_signal(tmp_mp3, target_sr=16000)
    try:
        os.remove(tmp_mp3)
    except Exception:
        pass

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    return round(len(audio) / float(sr), 2)


def build_interview_stress_v1(seed: int = 42) -> Dict[str, Any]:
    """
    Creates the 65-item Interview Stress Suite rendered by held-out EdgeTTS voices
    (25 South-Asian & tech proper-noun utterances + 40 SLU/Rubric QA turns).
    """
    cache_dir = os.path.join(AUDIO_CACHE_DIR, "interview_stress_v1")
    os.makedirs(cache_dir, exist_ok=True)
    voices = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "en-US-GuyNeural", "en-US-JennyNeural"]

    proper_noun_specs = [
        ("Sachin built our FastAPI gateway with Redis caching.", ["Sachin", "FastAPI", "Redis"]),
        ("Aradhya fine tuned PyTorch transformers using LoRA adapters.", ["Aradhya", "PyTorch", "LoRA"]),
        ("Nandu migrated the PostgreSQL database to sharded partitions.", ["Nandu", "PostgreSQL"]),
        ("Bablu optimized CUDA kernels for ONNX Runtime inference.", ["Bablu", "CUDA", "ONNX"]),
        ("Rahul deployed the gRPC microservice with Kafka streaming.", ["Rahul", "gRPC", "Kafka"]),
        ("Priya benchmarked CTranslate2 quantization on Kubernetes nodes.", ["Priya", "CTranslate2", "Kubernetes"]),
        ("Vikram integrated OAuth tokens across the backend cluster.", ["Vikram", "OAuth"]),
        ("Ananya evaluated TensorRT latency against OpenVINO baselines.", ["Ananya", "TensorRT", "OpenVINO"]),
        ("Karthik paired Silero VAD with Whisper beam search.", ["Karthik", "Silero", "Whisper"]),
        ("Rohan implemented SAPI5 fallback for Windows TTS output.", ["Rohan", "SAPI5"]),
        ("Sneha audited the Llama embeddings inside ChromaDB.", ["Sneha", "Llama", "ChromaDB"]),
        ("Deepa fixed race conditions in asyncio event loops.", ["Deepa", "asyncio"]),
        ("Aditya verified WeSpeaker CAM embeddings at twenty decibels.", ["Aditya", "WeSpeaker"]),
        ("Siddharth led the system architecture review with Krishnendu.", ["Siddharth", "Krishnendu"]),
        ("Meera profiled Docker memory limits under concurrent load.", ["Meera", "Docker"]),
        ("Shreya wrote unit tests in pytest for FastAPI endpoints.", ["Shreya", "pytest", "FastAPI"]),
        ("Varun reduced p99 latency in the Nginx reverse proxy.", ["Varun", "Nginx"]),
        ("Gaurav configured Prometheus alerts on GPU utilization.", ["Gaurav", "Prometheus"]),
        ("Tanvi deployed Kyutai Pocket TTS models on CPU.", ["Tanvi", "Kyutai"]),
        ("Harish validated GraphQL schemas before production release.", ["Harish", "GraphQL"]),
        ("Manish compared Qwen and Llama three quantization curves.", ["Manish", "Qwen", "Llama"]),
        ("Pooja tuned Mel spectrogram filterbanks with SciPy.", ["Pooja", "SciPy"]),
        ("Sanjay debugged WebSocket audio buffering in Chrome.", ["Sanjay", "WebSocket"]),
        ("Nikhil refactored dependency injection in Python three point ten.", ["Nikhil", "Python"]),
        ("Swati inspected LibriSpeech transcription error rates.", ["Swati", "LibriSpeech"]),
    ]

    qa_specs = [
        ("Hello Mira, good morning, can we start the interview?", "greeting", "mid", [5, 7]),
        ("Hi everyone, great to meet you for today's system check.", "greeting", "mid", [5, 7]),
        ("Good afternoon, Sachin is ready to begin.", "greeting", "mid", [5, 7]),
        ("Hey there, let us start with our voice test.", "greeting", "mid", [5, 7]),
        ("Good evening, I am joining from Bengaluru for the interview.", "greeting", "mid", [5, 7]),
        ("Who are you and how does this assistant work?", "identity_query", "mid", [5, 7]),
        ("Introduce yourself and explain your role in this call.", "identity_query", "mid", [5, 7]),
        ("What is your name and what framework powers you?", "identity_query", "mid", [5, 7]),
        ("Are you an automated interviewer or a co-pilot?", "identity_query", "mid", [5, 7]),
        ("Tell me who built this assistant and its purpose.", "identity_query", "mid", [5, 7]),
        ("Can you explain the six stages of your SOLID pipeline?", "project_query", "high", [7, 9]),
        ("How is the capstone architecture decoupled across services?", "project_query", "high", [7, 9]),
        ("Walk me through the model contracts in your pipeline.", "project_query", "high", [7, 9]),
        ("Explain how the ASR and TTS modules communicate cleanly.", "project_query", "high", [7, 9]),
        ("What is the SOLID design behind your voice agent?", "project_query", "high", [7, 9]),
        ("Calculate 48 multiplied by 12 for throughput estimation.", "calculation_query", "mid", [5, 7]),
        ("What is 250 plus 175 milliseconds total latency?", "calculation_query", "mid", [5, 7]),
        ("How much is 1024 divided by 16?", "calculation_query", "mid", [5, 7]),
        ("Calculate 95 minus 28 percent error reduction.", "calculation_query", "mid", [5, 7]),
        ("What is 15 times 60 requests per minute?", "calculation_query", "mid", [5, 7]),
        ("What time is it right now on the server?", "time_query", "mid", [4, 7]),
        ("Can you tell me the current time before we wrap up?", "time_query", "mid", [4, 7]),
        ("What is the current time in our interview schedule?", "time_query", "mid", [4, 7]),
        ("I used Redis read replicas and consistent hashing shards to cut latency under forty milliseconds with idempotent retries.", "answer", "high", [8, 10]),
        ("To survive fifty times read traffic, I combined Kafka queues, Redis TTL invalidation, circuit breakers, and strict Prometheus observability.", "answer", "high", [8, 10]),
        ("We implemented the saga orchestration pattern across distributed microservices to keep database commits ACID-consistent under failover.", "answer", "high", [8, 10]),
        ("During the outage, I isolated the database connection pool leak, rolled back the faulty release, and added chaos tests.", "answer", "high", [8, 10]),
        ("For neural voice cloning, we separate the speaker encoder from the acoustic flow model to maintain low latency.", "answer", "high", [8, 10]),
        ("We added a database index and some caching to make queries faster when traffic increased.", "answer", "mid", [5, 7]),
        ("I think we can put a load balancer in front of the backend servers and restart failed containers.", "answer", "mid", [5, 7]),
        ("We normally store user records in SQL and send asynchronous tasks to a background queue.", "answer", "mid", [5, 7]),
        ("My team updated the API endpoint validation so bad requests fail early before touching disk.", "answer", "mid", [5, 7]),
        ("Maybe just add more RAM to the database server and check logs later.", "answer", "low", [2, 5]),
        ("I am not really sure how distributed consistency works, we never tested failures.", "answer", "low", [1, 4]),
        ("Usually I just copy code until tests pass without tracking latency metrics.", "answer", "low", [1, 4]),
        ("Yes, caching sounds fine to use somewhere in the project.", "answer", "low", [2, 5]),
        ("Thank you for the interview, goodbye for now!", "farewell", "mid", [5, 7]),
        ("That concludes my answers, see you later and bye.", "farewell", "mid", [5, 7]),
        ("Alright, let us exit the session now, goodbye.", "farewell", "mid", [5, 7]),
        ("Thanks Mira, quit the interview and have a good day.", "farewell", "mid", [5, 7]),
    ]

    items = []
    for idx, (text, terms) in enumerate(proper_noun_specs):
        split = "val" if idx < 5 else "test"
        voice = voices[idx % len(voices)]
        wav_path = os.path.join(cache_dir, f"pn_{idx:02d}.wav")
        dur = _render_edgetts_wav(text, voice, wav_path)
        items.append({
            "id": f"pn_{idx:02d}",
            "subset": "proper_nouns",
            "split": split,
            "voice": voice,
            "audio_path": os.path.relpath(wav_path, ROOT_DIR).replace("\\", "/"),
            "text": text,
            "text_hash": text_sha256(text),
            "target_terms": terms,
            "duration_sec": dur,
        })

    for idx, (text, intent, tier, rng_scores) in enumerate(qa_specs):
        split = "val" if idx % 4 == 0 else "test"  # 10 val, 30 test across all intents/tiers
        voice = voices[(idx + 1) % len(voices)]
        wav_path = os.path.join(cache_dir, f"qa_{idx:02d}.wav")
        dur = _render_edgetts_wav(text, voice, wav_path)
        items.append({
            "id": f"qa_{idx:02d}",
            "subset": "interview_qa",
            "split": split,
            "voice": voice,
            "audio_path": os.path.relpath(wav_path, ROOT_DIR).replace("\\", "/"),
            "text": text,
            "text_hash": text_sha256(text),
            "intent_label": intent,
            "rubric_tier": tier,
            "expected_score_range": rng_scores,
            "duration_sec": dur,
        })

    payload = {
        "name": "interview_stress_v1",
        "source": "Synthetic domain corpus rendered via held-out Azure EdgeTTS (en-IN / en-US)",
        "license": "MIT",
        "seed": seed,
        "total_items": len(items),
        "subset_counts": {
            "proper_nouns": len(proper_noun_specs),
            "interview_qa": len(qa_specs),
        },
        "split_counts": {s: sum(1 for x in items if x["split"] == s) for s in ["val", "test"]},
        "items": items,
    }
    _save_manifest("interview_stress_v1", payload)
    return payload


def build_all_benchmarks(seed: int = 42) -> Dict[str, Any]:
    """Builds all 4 Session 4 manifests and prints storage & split summary."""
    ls = build_librispeech_60(seed=seed)
    m14 = build_minds14_140(seed=seed)
    cl = build_clinc_oos_200(seed=seed)
    isv = build_interview_stress_v1(seed=seed)

    total_bytes = 0
    for root, _, files in os.walk(AUDIO_CACHE_DIR):
        for fn in files:
            total_bytes += os.path.getsize(os.path.join(root, fn))

    return {
        "total_disk_mb": round(total_bytes / (1024 * 1024), 2),
        "manifests": {
            m["name"]: {"total": m["total_items"], "splits": m["split_counts"]}
            for m in (ls, m14, cl, isv)
        },
    }


if __name__ == "__main__":
    summary = build_all_benchmarks(seed=42)
    print(json.dumps(summary, indent=2))
