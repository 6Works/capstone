# Model Cards — Decoupled 6-Stage Voice AI Interview Assistant (`Pre-Upgrade Snapshot`)

Structured model cards for all ML/neural components evaluated or deployed prior to the `v2.0` architecture overhaul, grounded strictly in observed telemetry under `ml/results/*.json`.

---

## 1. Layer A — `Whisper INT8`, `Parakeet-TDT-0.6B-v3`, `Whisper-large-v3-turbo` & G2P Phonetic Repair

- **Models & Licenses:**
  - `Systran/faster-whisper-small.en` (`244M`, `463.58 MB` disk, `718.46 MB` Peak RSS, `MIT`, CPU `INT8`).
  - `Systran/faster-whisper-tiny.en` (`39M`, `74.47 MB` disk, `266.72 MB` Peak RSS, `MIT`, CPU `INT8`).
  - `nvidia/parakeet-tdt-0.6b-v3` (`600M`, `640.24 MB` disk, `843.89 MB` Peak RSS, `CC-BY-4.0`, `sherpa-onnx` CPU `INT8`). Note: exported ONNX bundle excludes `bpe.vocab`; quiet inputs (`peak < 0.05`) are gain-normalized in `services/asr/asr_services.py`.
  - `openai/whisper-large-v3-turbo` (`809M`, `MIT`, Kaggle T4 FP16).
- **Layer A Modifications (`services/asr/asr_services.py`):**
  - `small_improved`: Silero VAD + dynamic prompt + CTranslate2 trie `hotwords` + word-confidence gated **`g2p_en` ARPAbet (`Apache-2.0`) + Rapidfuzz** `phonetic_repair()`.
  - `tiny_improved`: Capacity-gated conditioning (`hotwords_enabled=False`, `condition_on_previous_text=False`, `repetition_penalty=1.1`, `no_repeat_ngram_size=4`, `hallucination_silence_threshold=0.4`).
  - *Post-audit hotfix (`2026-11-26`)*: Added `_is_morphological_inflection()`, expanded `_COMMON_PROSE_GUARD`, added `_FUNCTION_BIGRAM_GUARD` (`"a", "the", ...`), and enforced strict matching when `words_meta=None`.
- **Observed Pre-Fix Benchmarks:**
  - `asr_small_improved`: Clean IV WER **`0.0435`** (`p<0.001` vs `0.0849`), `R_ent` **`0.9783`** (`20 dB: 0.9783`, `10 dB: 0.9348`, closed `0.9643`, unseen `0.6111`), `LibriSpeech-clean` `0.0306` / `other` `0.0958`, `CPU RTF = 0.8958`; pre-fix `MINDS-14 (498)` `0.2500` (`36` repairs).
  - `asr_tiny_improved`: `R_ent` **`0.9130`** (`10 dB: 0.8043`), `CPU RTF = 0.2801`, `2.7x` lower RSS than `small.en`.
  - `asr_parakeet_tdt`: **`LibriSpeech-clean = 0.0221`**, **`LibriSpeech-other = 0.0605`** (`-36.8%` vs `small.en`, `p<0.001`), `IV = 0.0642`, **`CPU RTF = 0.1564`**; pre-fix `MINDS-14 (498) = 0.3018` (`155` un-guarded repairs — fixed in code).
  - `whisper-large-v3-turbo` (`T4`): **`ai4bharat/Svarah` (`N=1,000 en-IN`) = `0.0888`** vs `small.en` `0.1521` (`-41.6%`, `p<0.001`); `MINDS-14 (498) = 0.2517` (reflects noisy ground-truth refs); `IV_50 = 0.0787`.

---

## 2. Layer B — `Qwen3-4B`, `DistilBERT-LoRA`, `Qwen3-4B QLoRA`, `Phi-4-mini` & Hybrid Cascade

- **Permissive Production Models:**
  - `qwen3:4b` (`Q4_K_M`, `2.33 GB` disk, `3.33 GB` VRAM, **`Apache-2.0`**) + Kaggle NF4 **QLoRA adapter** (`11,796,480` trainable params, `models/qwen3_4b_slu_lora/`).
  - `DistilBERT-LoRA` (`distilbert-base-uncased` + PEFT LoRA `r=16, lr=2.67e-4, 5 epochs`, `1,010,851` trainable params, **`Apache-2.0`**, `models/slu_lora_adapter/`, `~0.25 GB` disk).
  - `HybridConfidenceSLUService`: Tier-2 `DistilBERT-LoRA` (`conf >= 0.85, margin >= 0.25, non-OOS`) cascading to Tier-3 `Qwen3-4B`.
  - `phi4-mini` (`3.8B`, `2.32 GB` disk, `3.06 GB` VRAM, **`MIT`**).
  - Reference / Restricted baselines: `llama3.2:3b` / `llama3.1:8b` (`Meta Community License`; `8B` requires `43% CPU` spill locally); legacy `qwen2.5:3b` (`qwen-research`, deprecated for commercial use).
- **Observed Benchmarks:**
  - **`slu_qwen3_4b_repair` (3 seeds Ollama):** `MINDS-14 (56)` **`0.9940 ± 0.0105`**, `CLINC (156)` **`0.8378 ± 0.0048` (`OOS = 0.7755`)**, `IV QA` **`0.8986`**, Calibrated 3-Tier Rubric **`rho = 0.8459 ± 0.0388` (`MAE = 1.4167`)**, `MINDS-14 (498)` **`0.9315` Gold / `0.9167` Tiny-ASR**, `100%` schema compliance, cost **`$0.0160/1k`**. Proven TOST non-inferior to `Llama-3.1-8B` ($p_{\text{NI}} = 0.0035$) at `9.4x` lower cost.
  - **`DistilBERT-LoRA`:** Local CPU **`0.9520` Gold (`28.8 ms`) / `0.9441` Tiny-ASR (`58.0 ms`)** on `MINDS-14 (498)`, **`0.9038` (`OOS = 0.7092`, `31.5 ms`)** on `CLINC (5,500)`, **`$0.0011/1k`**. Kaggle 3-seed T4: `0.9423 ± 0.0059` vs Linear Probe `0.7471` (`MINDS-14`), `0.9112 ± 0.0073` (`OOS = 0.7523`) vs `0.6657` (`CLINC-5500`).
  - **`Qwen3-4B QLoRA` (Kaggle T4):** Lifts `MINDS-14 (498)` noisy F1 `0.8364 ->` **`0.9484`** (`1.0` schema) and `CLINC (156)` `0.8431 ->` **`0.9366` (`OOS = 0.8814`)**.

---

## 3. Layer C — `Kyutai Pocket TTS INT8` & `Piper VITS` Fallback

- **Models & Licenses:** `Kyutai Pocket TTS INT8` (`~189 MB` ONNX, `238.87 MB` Peak RSS, `24 kHz`, 5 flow-matching steps; `CC-BY-4.0` weights + `MIT`/`Apache-2.0` runtime) and fallback `Piper VITS` (`en_US-lessac-low` / `en_US-ryan-low` `MIT`; `piper1-gpl` is `GPL-3.0-or-later`).
- **Observed Benchmarks (`36` pairs across `12` unseen LibriSpeech speakers):**
  - `CAM++` held-out **`0.5898`** (specificity margin **`+0.1402`**, `AUC = 0.7371`, `d' = 0.9224`, `EER = 0.3106`), `ECAPA512` **`0.8031`** (margin **`+0.1030`**), `RT-WER = 0.1084`, streaming **`ttfa_ms = 1913.7 ms`**, `avg_latency_ms = 2382.1 ms`, `CPU RTF = 0.5336`, `$0.0331/1k`. Outperforms `Piper+WSOLA` (`CAM++ = 0.4180`, `p < 0.001`) and `EdgeTTS` specificity (`+0.0937 CAM++`, `p < 0.001`).

---

## 4. Dual Biometric Encoders (`WeSpeaker CAM++ LM` & `ECAPA512 LM`)

- **Models:** `wespeaker_en_voxceleb_CAM++_LM.onnx` (`512-D`, `27.93 MB`) & `voxceleb_ECAPA512_LM.onnx` (`192-D`), `CC-BY-4.0` weights / `Apache-2.0` `sherpa-onnx`.
- **Observed Benchmarks:** Clean `other_60` `CAM++ d' = 2.6863` (`gap = 0.3744, AUC = 0.9619`) and `ECAPA512 EER = 0.0835` (`AUC = 0.9665`). Under `300–3400 Hz` telephony EQ, both collapse toward `EER 0.3333–0.4461`. `VoiceCloner` now emits `narrowband_warning=True` when Welch 95% spectral rolloff `<= 3600 Hz`.
- **Misuse Guardrails:** Used purely for evaluation, voice-profile telemetry, and quality alerts—never security authentication or unauthorized impersonation.
