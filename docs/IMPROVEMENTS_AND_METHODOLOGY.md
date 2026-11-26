# System Improvements, Upgraded Architecture & Verification Methodology

This document records every architectural, algorithmic, licensing, dataset, and statistical upgrade implemented across the 6-stage decoupled voice agent pipeline (`services/` and `ml/`), and establishes the exact verified baseline prior to the upcoming `v2.0` streaming architecture overhaul (see [`docs/ARCHITECTURE_UPGRADE_BASELINE.md`](ARCHITECTURE_UPGRADE_BASELINE.md)).

---

## 1. Licensing Corrections & Commercial Hygiene (`Verified 2026-11-26`)

| Component | Previous Claim | Verified License / Term | Action Taken |
| :--- | :--- | :--- | :--- |
| `Qwen2.5-3B-Instruct` | Claimed `Apache-2.0` | **`qwen-research`** (non-commercial research license on HuggingFace) | Replaced in commercial stack & benchmarks by **`Qwen3-4B` (`Apache-2.0`)**, **`Phi-4-mini-instruct` (`MIT`)**, and **`distilbert-base-uncased` LoRA (`Apache-2.0`)** |
| `piper-tts` (`OHF-Voice/piper1-gpl`) | Claimed `MIT, Cleared` | **`GPL-3.0-or-later`** (`rhasspy/piper` archived Oct 2025); `lessac` Blizzard 2013 data is non-commercial | Restricted Piper to optional GPL fallback; kept permissive `en_US-ryan-low` (`MIT` voice card) noted accurately alongside `Pocket TTS` |
| `kyutai/pocket-tts` (`arXiv:2509.06926`) | Labeled generically `CC-BY-4.0` | **`MIT` code / `Apache-2.0` `sherpa-onnx` export + `CC-BY-4.0` weights** (non-impersonation terms) | Documented code vs weight split and acceptable-use cloning constraints cleanly |
| `WeSpeaker` (`CAM++_LM`, `ECAPA512_LM`) | Mixed `Apache-2.0 / CC-BY-4.0` | **`CC-BY-4.0`** pretrained VoxCeleb weights + `Apache-2.0` `sherpa-onnx` runtime | Corrected attribution; integrated second independent encoder (`ECAPA512_LM`, 192-D ONNX) |
| `EdgeTTS` / `interview_stress_v1` | Labeled `MIT` | **Microsoft consumer Read-Aloud endpoint** (research proxy only) | Re-labeled `Research-Only Proxy`; added real human **`ai4bharat/Svarah` (`CC-BY-4.0`, `N=1,000`)**, full `minds14_en_full` (`1,662`), and `clinc_oos_full` (`16,196`) |
| `g2p_en 2.1.0` | New addition | **`Apache-2.0`** (avoids GPL `cmudict` PyPI package) | Installed cleanly inside `capstone_voice` for ARPAbet phoneme edit distance |
| `nvidia/parakeet-tdt-0.6b-v3` | New addition | **`CC-BY-4.0`** (`600M` FastConformer-TDT ONNX INT8) | Integrated via `ParakeetTDTASR(IASRService)` in `services/asr/asr_services.py` |

---

## 2. Core Algorithmic & Architectural Improvements Implemented

### 2.1 Layer A (Stage 2 ASR): G2P Phoneme Repair, Guard Hardening, Parakeet TDT & Real `en-IN` Svarah
1. **G2P ARPAbet Phoneme Edit Distance & Over-Correction Guards (`services/asr/asr_services.py:phonetic_repair`):**
   - Upgraded legacy Soundex to **`g2p_en` ARPAbet phoneme sequences** scored with articulatory confusable substitution costs (`0.40` vs `1.0` for `(F/P)`, `(B/P)`, `(IY/EY)`, `(S/Z)`, `(T/D)`, etc.).
   - Fixed a critical pre-upgrade over-correction flaw exposed by `ParakeetTDTASR` on `MINDS-14 Full` (`155` un-guarded false repairs such as `"cards"->"card"`, `"debits"->"debit"`, `"fees"->"freeze"`, `"amount"->"account"`, `"a payment"->"payment"` when `words_meta=None`): added `_is_morphological_inflection()`, `_FUNCTION_BIGRAM_GUARD`, expanded `_COMMON_PROSE_GUARD`, enforced `require_strict=True` whenever word confidences are absent, and added quiet-input gain normalization (`< 0.05` peak -> `0.25`) on `ParakeetTDTASR`.
   - Verified `asr_small_improved` interview proper-noun recall **`0.9783`** (`20 dB: 0.9783, 10 dB: 0.9348`, `OverCorr=0.0`, `p < 0.001` vs `small_baseline`), while un-guarded `ParakeetTDTASR` hit **`0.0221` LibriSpeech-clean** and **`0.0605` LibriSpeech-other** (`-36.8%` vs `small.en` `0.0958`, `p < 0.001`) at **`0.1564` CPU RTF**.
2. **Real Human `en-IN` Anchor (`ai4bharat/Svarah`, `N = 1,000`) & `Whisper-large-v3-turbo` (`Kaggle T4`):**
   - Evaluated `openai/whisper-large-v3-turbo` (`MIT`) on T4 GPU: **`MINDS-14 (498) = 0.2517`** (`CER=0.2013`), proving that the `~25%` `MINDS-14` WER plateau reflects noisy crowd/Google-ASR labels in `PolyAI/minds14`, whereas on clean human `en-IN` speech in **`ai4bharat/Svarah` (`N = 1,000`)**, `whisper-large-v3-turbo` slashed Corpus WER from **`0.1521`** (`small.en`, `CER = 0.1354`) down to **`0.0888`** (`CER = 0.0413`; Non-digit `N=885`: `0.1357 -> 0.0828`; Digit `N=115`: `0.2528 -> 0.1261`; paired bootstrap `+0.533` per-clip, `p < 0.001`).
3. **ITN Dual-Track & Generalization Controls (`ml/metrics/eval_metrics.py`):**
   - Added `normalize_text_itn()` and `compute_itn_wer()` (`% -> percent`, numeric hyphen separation, cardinal word normalization), alongside `non_entity_wer`, `repair_precision`, `overcorrection_rate`, and the 50% held-out zero-shot proper-noun split.

### 2.2 Layer B (Stage 3/4 SLU): Schema Conformance, Tuned LoRA/QLoRA, Hybrid Cascade & Calibrated Rubric
1. **GBNF JSON-Schema, Prompt Calibration & Logprob OOS Calibration (`services/llm/`, `services/nlp/`):**
   - Added Ollama native GBNF JSON-Schema constrained decoding (`payload["format"] = response_schema`), disabled `<think>` overhead on `qwen3`, and tuned validation-only `oos_prob_threshold` from sequence logprobs.
   - `slu_qwen3_4b_repair` (`Apache-2.0`, `Q4_K_M`, `$0.0160/1k`) reached **`0.9940`** on `MINDS-14 (56)`, **`0.9315` Gold / `0.9167` Tiny-ASR** on `MINDS-14 (498)`, **`0.8378`** (`OOS=0.7755`) on `CLINC (156)`, and **`0.8986`** on `IV QA`, proving **statistically non-inferior to `Llama-3.1-8B`** under TOST (`Delta=-0.05`, `p_NI = 0.0035`) at **9.4x lower cost**.
2. **Trained Neural Adapters (`Kaggle T4` & CPU ONNX/PEFT):**
   - **`DistilBERT-LoRA` (`Apache-2.0`, `models/slu_lora_adapter/`, Optuna `r=16, lr=2.67e-4, 5 epochs`):** `0.9423 ± 0.0059` (T4 3-seed noisy+conf vs `0.9375` plain and **`0.7471` Linear Probe**), `0.9112 ± 0.0073` (`OOS=0.7523`) on `CLINC (5,500)` at **`2.2 ms` GPU**; locally **`0.9520` Gold / `0.9441` Tiny-ASR** on `MINDS-14 (498)` and **`0.9038` (`OOS=0.7092`)** on `CLINC (5,500)` at **`28.8–58.0 ms` CPU (`$0.0011/1k`)**.
   - **`Qwen3-4B QLoRA` (4-bit NF4, `11.8M` params, `models/qwen3_4b_slu_lora/`):** Improved `MINDS-14 (498)` noisy F1 from `0.8364` (zero-shot) to **`0.9484`** (`1.0` Schema Validity) and `CLINC (156)` from `0.8431` (`OOS=0.8065`) to **`0.9366` (`OOS=0.8814`)**.
3. **Hybrid Confidence Cascade (`HybridConfidenceSLUService`):**
   - Implemented 2-tier routing (`DistilBERT-LoRA` exit when `confidence >= 0.85`, `margin >= 0.25`, non-OOS; otherwise escalate to `Qwen3-4B` / unsupported taxonomies) and wired into `PipelineFactory` and `server.py` (`/api/pipeline/providers`).
4. **3-Tier Behavioral Rubric Calibration (`services/roles/role_service.py`):**
   - Diagnosed central-tendency clustering (`{3,4,6,7}`) collapsing `Qwen3-4B` Spearman correlation to `rho = 0.5735` (`MAE = 1.9167`). Adding domain-agnostic 3-tier behavioral anchors (`1-4 / 5-7 / 8-10`) raised 3-seed `Qwen3-4B` rubric alignment to **`rho = 0.8459 ± 0.0388`, `MAE = 1.4167`** (`s42 p = 0.0019`).

### 2.3 Layer C (Stage 5 TTS & Speaker Verification): Dual Encoders, Specificity Margin, TTFA & Narrowband Check
1. **Dual Encoders (`CAM++` + `ECAPA512`):** Audited clean (`60`), `20 dB`, `LibriSpeech-other (60)`, and `300–3400 Hz` telephony across disjoint clips. On `librispeech_other_60`, `CAM++` achieved `d' = 2.6863` (`gap=0.3744, EER=0.0919, AUC=0.9619`) and `ECAPA512` achieved **`EER=0.0835, AUC=0.9665`**. Under `300–3400 Hz` telephony, both degraded to `EER 0.3333–0.4461` (`clean 0.4296 / 0.4419`).
2. **TTS Cloning & TTFA (`ml/results/tts_ablation.json`):** `Pocket TTS INT8` reached `CAM++ = 0.5898` (`+0.1402` specificity margin, `d'=0.9224, AUC=0.7371`) and `ECAPA = 0.8031` (`+0.1030` margin), beating `Piper+WSOLA` (`CAM++ = 0.4180`, `p < 0.001`) and `EdgeTTS` specificity (`+0.0937 CAM++`, `+0.0838 ECAPA`, `p < 0.001`), with streaming **`ttfa_ms = 1913.7 ms`** vs `avg_latency_ms = 2382.1 ms`.
3. **Runtime Narrowband Warning (`services/tts/voice_cloner.py`):** Computed Welch 95% spectral rolloff during profile extraction to set `narrowband_warning=True` whenever `spectral_rolloff_95_hz <= 3600 Hz`.

---

## 3. Statistical & Experimental Rigor Upgrades (`ml/metrics/eval_metrics.py`)
1. **Bounded Phipson-Smyth Bootstrap (`paired_bootstrap_test`, $B=2000$):** Bounds minimum two-sided $p \ge 0.0005$ via $(c+1)/(B+1)$ (reported cleanly as $p < 0.001$).
2. **Turn-Level Aggregation (`run_e2e_ablation.py`) & Stage Checkpoints:** Aggregates across $N$ unique turns (never inflating $N \times \text{seeds}$) with atomic ASR/SLU/TTS stage checkpointing.
3. **TOST Non-Inferiority (`tost_non_inferiority`, $\Delta = -0.05$):** Rigorously verifies non-inferiority when the 90% CI lower bound strictly exceeds $-0.05$.

---

## 4. Pre-Upgrade Execution Snapshot (`2026-11-26`)
- **Completed & Downloaded:** All `5/5` Kaggle GPU experiments (`DistilBERT-LoRA` 3-seed + LinearProbe, `Qwen3-4B` Zero-Shot vs QLoRA, `whisper-large-v3-turbo`, `Svarah N=1,000 en-IN`); local Dual-Encoder Audit v2; Layer C TTS v2 (`36` pairs x `5` engines + `TTFA`); Layer B SLU (`11` complete v2 configs + calibrated 3-seed `Qwen3-4B` Rubric + 2-seed `Llama-3.1-8B` checkpoints with TOST); `fig1–fig4`.
- **Hardened & Unit-Verified (Ready for `v2.0` Architecture Upgrade):** `phonetic_repair` morphological/prose/determiner guards + `Parakeet` gain scaling, `RoleService` 3-tier rubric (`0.8459`), `HybridConfidenceSLUService`, `VoiceCloner` `<3.6 kHz` `narrowband_warning`, and `compute_itn_wer`. Full `v2` E2E matrix and `Hybrid` sweep are cleanly queued post-upgrade per [`docs/ARCHITECTURE_UPGRADE_BASELINE.md`](ARCHITECTURE_UPGRADE_BASELINE.md).
