"""
Session 7 Experiment (Upgraded): Layer C (Stage 5 TTS) Voice Cloning & Synthesis Ablation.
Evaluates 5 TTS architectures across all 12 unseen LibriSpeech speakers x 3 held-out interview utterances (36 paired runs):
  1. tts_pyttsx3           (Offline Windows SAPI5 robotic floor baseline)
  2. tts_piper_raw         (Unconditioned Piper VITS lessac-low, zero speaker conditioning)
  3. tts_piper_dual_wsola  (Dual-gender Piper VITS + F0 semitone WSOLA + spectral EQ)
  4. tts_pocket_clone      (Layer C zero-shot flow-matching Kyutai Pocket TTS INT8, 24 kHz)
  5. tts_edgetts_ref       (Microsoft Azure Neural EdgeTTS gender-matched cloud baseline)

Upgrades per review:
  - Dual independent neural speaker encoders: WeSpeaker CAM++ LM (512-D) AND WeSpeaker ECAPA-TDNN-512 LM (192-D).
  - Cloning Specificity Margin: matched target ref similarity minus mean similarity to 11 mismatched speakers.
  - Encoder Verification Stats (ROC-AUC, EER, Fisher d') across target vs mismatched speaker pairs.
  - Streaming Time-to-First-Audio (TTFA ms) telemetry for Pocket TTS INT8 plus short-reference warning count.
  - Bounded Paired Bootstrap (N=2000, Phipson-Smyth, never p=0.0) at both clip level (N=36) and speaker level (N=12).
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Tuple
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.asr import WhisperTransformerASR
from services.contracts import AudioPayload
from services.tts import (
    Pyttsx3TTSService,
    EdgeTTSService,
    NeuralVoiceCloningTTSService,
    PocketTTSCloningService,
    VoiceCloner,
    voice_cloner,
)
from ml.datasets import load_manifest
from ml.metrics import (
    compute_asr_metrics,
    compute_speaker_similarity,
    compute_audio_intelligibility,
    compute_verification_stats,
    paired_bootstrap_test,
    summarize_seeds,
    estimate_cost_per_1k_turns,
)


CONFIG_NAMES = [
    "tts_pyttsx3",
    "tts_piper_raw",
    "tts_piper_dual_wsola",
    "tts_pocket_clone",
    "tts_edgetts_ref",
]

PROMPT_SPLITS = [
    ("seed_42", "We designed a fault tolerant distributed cache using Redis and Kafka queues."),
    ("seed_1337", "Our offline voice pipeline achieves low latency while preserving speaker identity."),
    ("seed_2026", "Let us benchmark the speech synthesis quality across unseen technical speakers."),
]


def _probe_tts_rss_mb(cfg_name: str) -> float:
    """Measures isolated cold + synthesis Peak Working Set RSS (MB) in a fresh Python process."""
    code = (
        "import json, os, tempfile; "
        "from ml.experiments.run_asr_ablation import _get_memory_mb; "
        "from services.tts import Pyttsx3TTSService, EdgeTTSService, NeuralVoiceCloningTTSService, PocketTTSCloningService; "
        "wav = os.path.join(tempfile.gettempdir(), f'rss_probe_{os.getpid()}.wav'); "
        "txt = 'Testing voice pipeline synthesis memory footprint.'; "
    )
    if cfg_name == "tts_pyttsx3":
        code += "svc = Pyttsx3TTSService(); "
    elif cfg_name == "tts_piper_raw":
        code += "svc = NeuralVoiceCloningTTSService(with_conditioning=False); "
    elif cfg_name == "tts_piper_dual_wsola":
        code += "svc = NeuralVoiceCloningTTSService(with_conditioning=True); "
    elif cfg_name == "tts_pocket_clone":
        code += "svc = PocketTTSCloningService(speaker_profile_id='alex_interviewer'); "
    else:
        code += "svc = EdgeTTSService(); wav = wav.replace('.wav', '.mp3'); "

    code += (
        "svc.synthesize(txt, output_path=wav); "
        "rss, peak = _get_memory_mb(); "
        "print(json.dumps({'peak': peak})); "
        "os.path.exists(wav) and os.remove(wav)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT_DIR, capture_output=True, text=True, check=True)
    return float(json.loads(proc.stdout.strip().splitlines()[-1])["peak"])


def _extract_dual_embeddings(wav_path: str) -> Tuple[float, List[float], List[float]]:
    """Extracts median F0 (Hz), CAM++-512D embedding, and ECAPA512-192D embedding cleanly."""
    prof = VoiceCloner.extract_acoustic_profile_from_wav(wav_path, profile_name="probe", save_ref_dir=None)
    wav_16k, sr = VoiceCloner.load_audio_signal(wav_path, target_sr=16000)
    ecapa_emb, _ = VoiceCloner.extract_speaker_embedding(
        wav_16k, sample_rate=sr, use_neural=True, model_path=VoiceCloner.ECAPA_SPEAKER_ENCODER_PATH
    )
    return float(prof.fundamental_freq_f0), list(prof.embedding_vector), list(ecapa_emb)


def _prepare_speaker_refs() -> List[Dict[str, Any]]:
    """Loads 12 unseen LibriSpeech speakers (enroll WAV for conditioning, held-out val WAV for dual-encoder eval)."""
    items = load_manifest("librispeech_60")["items"]
    by_spk: Dict[str, Dict[str, Any]] = {}
    for it in items:
        by_spk.setdefault(it["speaker_id"], {})[it["split"]] = it

    refs: List[Dict[str, Any]] = []
    for spk in sorted(by_spk.keys()):
        enr = by_spk[spk]["enroll"]
        val = by_spk[spk]["val"]
        enr_abs = os.path.join(ROOT_DIR, enr["audio_path"])
        val_abs = os.path.join(ROOT_DIR, val["audio_path"])
        prof = VoiceCloner.extract_acoustic_profile_from_wav(
            enr_abs,
            profile_name=f"bench_{spk}",
            profile_id=f"bench_{spk}",
            save_ref_dir=None,
        )
        prof.ref_audio_path = enr_abs
        wav_e, sr_e = VoiceCloner.load_audio_signal(enr_abs, target_sr=16000)
        enr_dur = round(len(wav_e) / float(sr_e), 2)
        val_f0, val_cam, val_ecapa = _extract_dual_embeddings(val_abs)
        refs.append({
            "speaker_id": spk,
            "profile": prof,
            "enroll_path": enr_abs,
            "enroll_dur_sec": enr_dur,
            "val_path": val_abs,
            "val_f0": val_f0,
            "val_emb": val_cam,
            "val_ecapa": val_ecapa,
        })
    return refs


def run_ablation() -> Dict[str, Any]:
    speakers = _prepare_speaker_refs()
    asr_judge = WhisperTransformerASR(model_size="tiny.en", compute_type="int8", vad_filter=False, initial_prompt="")
    asr_judge._lazy_init()

    for s in speakers:
        voice_cloner.profiles[s["profile"].profile_id] = s["profile"]

    results: Dict[str, Any] = {
        "experiment": "layer_c_tts_cloning_ablation_v2",
        "num_speakers": len(speakers),
        "num_prompt_sets": len(PROMPT_SPLITS),
        "total_generations_per_engine": len(speakers) * len(PROMPT_SPLITS),
        "configs": {},
        "wsola_dsp_stoi": {},
        "paired_bootstrap": {},
    }

    per_engine_campp: Dict[str, List[float]] = {}
    per_engine_campp_margin: Dict[str, List[float]] = {}
    per_engine_ecapa: Dict[str, List[float]] = {}
    per_engine_ecapa_margin: Dict[str, List[float]] = {}
    per_engine_spk_margin: Dict[str, List[float]] = {}
    per_engine_f0_mae: Dict[str, List[float]] = {}
    per_engine_wer: Dict[str, List[float]] = {}
    piper_raw_wavs: Dict[str, str] = {}

    tmp_dir = tempfile.mkdtemp(prefix="tts_bench_")
    try:
        for cfg_name in CONFIG_NAMES:
            cfg = json.load(open(os.path.join(ROOT_DIR, "ml", "configs", f"{cfg_name}.json"), "r", encoding="utf-8"))
            print(f"\n=== Benchmarking {cfg_name} ({cfg['provider']}) ===", flush=True)
            peak_rss_mb = _probe_tts_rss_mb(cfg_name)

            if cfg["provider"] == "pyttsx3":
                svc = Pyttsx3TTSService(speech_rate=int(cfg.get("speech_rate", 175)))
            elif cfg["provider"] == "piper_raw":
                svc = NeuralVoiceCloningTTSService(with_conditioning=False)
            elif cfg["provider"] == "piper_dual_wsola":
                svc = NeuralVoiceCloningTTSService(with_conditioning=True)
            elif cfg["provider"] == "pocket_tts":
                svc = PocketTTSCloningService(num_steps=int(cfg.get("num_steps", 5)), speed=float(cfg.get("speed", 1.0)))
                svc.get_or_load_tts()
            else:
                svc = EdgeTTSService()

            if cfg["provider"] in ("piper_raw", "piper_dual_wsola", "pocket_tts"):
                warm_p = os.path.join(tmp_dir, f"warm_{cfg_name}.wav")
                svc.synthesize("Warmup synthesis.", output_path=warm_p, speaker_profile_id=speakers[0]["profile"].profile_id)
                if os.path.exists(warm_p):
                    os.remove(warm_p)

            all_campp, all_campp_margin, all_campp_diff_pool = [], [], []
            all_ecapa, all_ecapa_margin, all_ecapa_diff_pool = [], [], []
            all_enroll_campp, all_f0_mae, all_wer = [], [], []
            all_lat_ms, all_ttfa_ms, all_durs, wsola_stois = [], [], [], []
            short_ref_warnings = 0
            per_spk_margins: Dict[str, List[float]] = {sp["speaker_id"]: [] for sp in speakers}

            split_means: Dict[str, Dict[str, float]] = {}
            for split_name, prompt_txt in PROMPT_SPLITS:
                s_campp, s_ecapa, s_f0, s_wer = [], [], [], []
                for spk_info in speakers:
                    spk_id = spk_info["speaker_id"]
                    pid = spk_info["profile"].profile_id
                    ext = "mp3" if cfg["provider"] == "edge_tts" else "wav"
                    out_file = os.path.join(tmp_dir, f"{cfg_name}_{split_name}_{spk_id}.{ext}")

                    t0 = time.perf_counter()
                    if cfg["provider"] == "pyttsx3":
                        res = svc.synthesize(prompt_txt, output_path=out_file)
                    elif cfg["provider"] == "edge_tts":
                        gender = spk_info["profile"].gender_estimate
                        voice = cfg["voices"]["male"] if gender == "male" else cfg["voices"]["female"]
                        res = svc.synthesize(prompt_txt, output_path=out_file, voice=voice)
                    else:
                        res = svc.synthesize(prompt_txt, output_path=out_file, speaker_profile_id=pid)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if res.status != "success" or not os.path.exists(out_file):
                        raise RuntimeError(f"{cfg_name} failed on {spk_id}: {res.error}")

                    rmeta = res.metadata or {}
                    if rmeta.get("ttfa_ms") is not None:
                        all_ttfa_ms.append(float(rmeta["ttfa_ms"]))
                    else:
                        all_ttfa_ms.append(round(elapsed_ms, 1))
                    if rmeta.get("short_reference_warning"):
                        short_ref_warnings += 1

                    wav_16k, sr = VoiceCloner.load_audio_signal(out_file, target_sr=16000)
                    dur_s = max(0.1, len(wav_16k) / float(sr))

                    if cfg["provider"] == "piper_raw":
                        piper_raw_wavs[f"{split_name}_{spk_id}"] = out_file
                    elif cfg["provider"] == "piper_dual_wsola":
                        raw_mate = piper_raw_wavs.get(f"{split_name}_{spk_id}")
                        if raw_mate and spk_info["profile"].gender_estimate != "male":
                            wsola_stois.append(compute_audio_intelligibility(raw_mate, out_file))

                    syn_f0, syn_cam, syn_ecapa = _extract_dual_embeddings(out_file)
                    campp_same = compute_speaker_similarity(syn_cam, spk_info["val_emb"])
                    ecapa_same = compute_speaker_similarity(syn_ecapa, spk_info["val_ecapa"])
                    campp_enr = compute_speaker_similarity(syn_cam, spk_info["profile"].embedding_vector)

                    cam_other = [
                        compute_speaker_similarity(syn_cam, other["val_emb"])
                        for other in speakers
                        if other["speaker_id"] != spk_id
                    ]
                    eca_other = [
                        compute_speaker_similarity(syn_ecapa, other["val_ecapa"])
                        for other in speakers
                        if other["speaker_id"] != spk_id
                    ]
                    cam_margin = round(campp_same - float(np.mean(cam_other)), 4)
                    eca_margin = round(ecapa_same - float(np.mean(eca_other)), 4)
                    all_campp_diff_pool.extend(cam_other)
                    all_ecapa_diff_pool.extend(eca_other)
                    per_spk_margins[spk_id].append(cam_margin)

                    f0_err = abs(syn_f0 - spk_info["val_f0"])
                    asr_res = asr_judge.transcribe(
                        AudioPayload(
                            source_type="stream_buffer",
                            sample_rate=16000,
                            duration_sec=dur_s,
                            raw_bytes=wav_16k.astype(np.float32).tobytes(),
                            metadata={"audio_dtype": "float32"},
                        )
                    )
                    wer_val = compute_asr_metrics([prompt_txt], [asr_res.transcript or ""])["wer"]

                    s_campp.append(campp_same)
                    s_ecapa.append(ecapa_same)
                    s_f0.append(f0_err)
                    s_wer.append(wer_val)

                    all_campp.append(campp_same)
                    all_campp_margin.append(cam_margin)
                    all_ecapa.append(ecapa_same)
                    all_ecapa_margin.append(eca_margin)
                    all_enroll_campp.append(campp_enr)
                    all_f0_mae.append(f0_err)
                    all_wer.append(wer_val)
                    all_lat_ms.append(elapsed_ms)
                    all_durs.append(dur_s)

                split_means[split_name] = {
                    "campp_sim_mean": round(float(np.mean(s_campp)), 4),
                    "ecapa_sim_mean": round(float(np.mean(s_ecapa)), 4),
                    "f0_mae_hz": round(float(np.mean(s_f0)), 2),
                    "roundtrip_wer": round(float(np.mean(s_wer)), 4),
                }

            rtf = round((sum(all_lat_ms) / 1000.0) / sum(all_durs), 4)
            avg_lat_ms = round(float(np.mean(all_lat_ms)), 1)
            cam_ver = compute_verification_stats(all_campp, all_campp_diff_pool)
            eca_ver = compute_verification_stats(all_ecapa, all_ecapa_diff_pool)

            results["configs"][cfg_name] = {
                "config": cfg,
                "peak_rss_mb": peak_rss_mb,
                "avg_latency_ms": avg_lat_ms,
                "p95_latency_ms": round(float(np.percentile(all_lat_ms, 95)), 1),
                "ttfa_ms": summarize_seeds(all_ttfa_ms),
                "ttfa_p95_ms": round(float(np.percentile(all_ttfa_ms, 95)), 1),
                "short_reference_warnings": short_ref_warnings,
                "cpu_rtf": rtf,
                "cost_per_1k_turns_usd": estimate_cost_per_1k_turns(avg_lat_ms / 1000.0, cloud_api_usd_per_1k_turns=0.0),
                "campp_sim_heldout_val": summarize_seeds(all_campp),
                "campp_specificity_margin": summarize_seeds(all_campp_margin),
                "campp_verification": cam_ver,
                "ecapa_sim_heldout_val": summarize_seeds(all_ecapa),
                "ecapa_specificity_margin": summarize_seeds(all_ecapa_margin),
                "ecapa_verification": eca_ver,
                "campp_sim_enroll": summarize_seeds(all_enroll_campp),
                "f0_mae_hz": summarize_seeds(all_f0_mae),
                "roundtrip_wer": summarize_seeds(all_wer),
                "prompt_split_means": split_means,
            }
            if wsola_stois:
                results["wsola_dsp_stoi"]["piper_dual_wsola_vs_raw_female"] = summarize_seeds(wsola_stois)

            per_engine_campp[cfg_name] = all_campp
            per_engine_campp_margin[cfg_name] = all_campp_margin
            per_engine_ecapa[cfg_name] = all_ecapa
            per_engine_ecapa_margin[cfg_name] = all_ecapa_margin
            per_engine_spk_margin[cfg_name] = [float(np.mean(per_spk_margins[sp["speaker_id"]])) for sp in speakers]
            per_engine_f0_mae[cfg_name] = all_f0_mae
            per_engine_wer[cfg_name] = all_wer

            print(
                f"  CAM++={results['configs'][cfg_name]['campp_sim_heldout_val']['mean']} (margin={results['configs'][cfg_name]['campp_specificity_margin']['mean']:+.4f}, AUC={cam_ver['roc_auc']}, EER={cam_ver['eer']}, d'={cam_ver['d_prime']}) | "
                f"ECAPA={results['configs'][cfg_name]['ecapa_sim_heldout_val']['mean']} (margin={results['configs'][cfg_name]['ecapa_specificity_margin']['mean']:+.4f}, AUC={eca_ver['roc_auc']}) | "
                f"TTFA={results['configs'][cfg_name]['ttfa_ms']['mean']}ms | WER={results['configs'][cfg_name]['roundtrip_wer']['mean']}",
                flush=True,
            )

        # Bounded Paired Bootstrap tests (N=2000, Phipson-Smyth) across N=36 clips AND N=12 speakers
        for label, base_k, cand_k, data_map, hib in [
            ("pocket_vs_piper_wsola_campp", "tts_piper_dual_wsola", "tts_pocket_clone", per_engine_campp, True),
            ("pocket_vs_piper_wsola_campp_margin", "tts_piper_dual_wsola", "tts_pocket_clone", per_engine_campp_margin, True),
            ("pocket_vs_edgetts_campp_margin", "tts_edgetts_ref", "tts_pocket_clone", per_engine_campp_margin, True),
            ("pocket_vs_edgetts_ecapa_margin", "tts_edgetts_ref", "tts_pocket_clone", per_engine_ecapa_margin, True),
            ("pocket_vs_piper_wsola_ecapa_margin", "tts_piper_dual_wsola", "tts_pocket_clone", per_engine_ecapa_margin, True),
            ("pocket_vs_piper_wsola_speaker_margin_n12", "tts_piper_dual_wsola", "tts_pocket_clone", per_engine_spk_margin, True),
            ("pocket_vs_edgetts_speaker_margin_n12", "tts_edgetts_ref", "tts_pocket_clone", per_engine_spk_margin, True),
            ("pocket_vs_piper_raw_campp", "tts_piper_raw", "tts_pocket_clone", per_engine_campp, True),
            ("pocket_vs_pyttsx3_campp", "tts_pyttsx3", "tts_pocket_clone", per_engine_campp, True),
            ("pocket_vs_edgetts_campp", "tts_edgetts_ref", "tts_pocket_clone", per_engine_campp, True),
            ("pocket_vs_piper_wsola_f0_mae", "tts_piper_dual_wsola", "tts_pocket_clone", per_engine_f0_mae, False),
            ("pocket_vs_pyttsx3_wer", "tts_pyttsx3", "tts_pocket_clone", per_engine_wer, False),
        ]:
            results["paired_bootstrap"][label] = paired_bootstrap_test(
                data_map[base_k], data_map[cand_k], n_resamples=2000, seed=42, higher_is_better=hib
            )

        out_path = os.path.join(ROOT_DIR, "ml", "results", "tts_ablation.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved complete Layer C v2 ablation results -> {out_path}", flush=True)
        return results
    finally:
        for s in speakers:
            voice_cloner.profiles.pop(s["profile"].profile_id, None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_ablation()
