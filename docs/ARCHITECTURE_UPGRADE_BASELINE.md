# Pre-Upgrade Architecture Dossier & Empirical Baseline (`v1.x -> v2.0 Transition`)

**Snapshot Date:** `2026-11-26`  
**Purpose:** Complete technical inventory, empirical benchmark record, root-cause failure catalog, hotfix ledger, and target `v2.0` architectural blueprint recorded immediately prior to executing a major architectural overhaul across the 6-stage conversational voice pipeline (`6Works`).

> Every metric, symbol name, route, and file path below is directly verified from repository source files and `ml/results/*.json`. Where a targeted code fix was implemented and unit-tested (`py_compile` / `pytest`) without yet completing a full end-to-end sweep (e.g., Layer A `Parakeet` on full `MINDS-14`, `e2e_ablation_matrix_v2`), the pre-fix measurement and post-fix status are explicitly distinguished.

---

## 1. Current System Topology & Module Inventory (`v1.x`)

### 1.1 Pipeline Pattern & Contracts ([`services/contracts.py`](../services/contracts.py))
The current runtime implements a synchronous, single-turn linear chain orchestrated by [`DecoupledVoicePipeline`](../orchestrator/pipeline.py) and assembled via Dependency Injection in [`PipelineFactory`](../orchestrator/factory.py):

$$\text{AudioInput } (f_1) \;\longrightarrow\; \text{ASR } (f_2) \;\longrightarrow\; \text{NLP/SLU } (f_3) \;\longrightarrow\; \text{Dialogue/Role } (f_4) \;\longrightarrow\; \text{TTS } (f_5) \;\longrightarrow\; \text{AudioOutput } (f_6)$$

| Stage | Abstract Base Class | Primary Input DTO | Output DTO | Concrete Providers Currently Registered |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `IAudioInputService` | capture kwargs | `AudioPayload` (`source_type`, `sample_rate`, `channels`, `duration_sec`, `file_path`, `raw_bytes`, `metadata`) | `MicrophoneAudioInput`, `FileAudioInput`, `SyntheticAudioInput` |
| **2** | `IASRService` | `AudioPayload` | `ASRResult` (`transcript`, `confidence`, `language`, `latency_sec`, `provider`, `status`, `error`, `metadata`) | `WhisperTransformerASR` (`small.en` / `tiny.en` INT8), `ParakeetTDTASR` (`0.6b-v3` INT8), `GoogleWebSpeechASR`, `MockASR` |
| **3** | `INLPService` | `text: str` | `NLPResult` (`cleaned_text`, `intent`, `confidence`, `entities`, `sentiment`, `tool_calls`, `latency_sec`, `provider`, `metadata`) | `RuleBasedNLPService`, `LLMNLPService`, `TrainedDistilBertSLUService`, `HybridConfidenceSLUService` |
| **4** | `IDialogueManagerService` | `NLPResult` | `DialogueTurnResult` (`response_text`, `dialogue_action`, `state`, `turn`, `memory_context`, `latency_sec`, `provider`) | `ModularDialogueManager`, `LLMDialogueManager`, plus [`RoleService`](../services/roles/role_service.py), [`InterviewContextManager`](../services/context/context_manager.py), [`BlueprintService`](../services/blueprints/blueprint_service.py) |
| **5** | `ITTSService` | `text: str, output_path` | `TTSResult` (`file_path`, `audio_format`, `latency_sec`, `provider`, `speech_rate`, `sample_rate`, `status`, `metadata`, `error`) | `PocketTTSCloningService` (default), `NeuralVoiceCloningTTSService` (`Piper VITS` + WSOLA/EQ), `EdgeTTSService`, `ElevenLabsTTSService`, `GTTSService`, `Pyttsx3TTSService` |
| **6** | `IAudioOutputService` | `audio_file_path: str` | `PlaybackResult` (`status`, `playback_time_sec`, `device`, `provider`, `error`) | `WebStreamAudioOutput`, `SystemSpeakerOutput` |

### 1.2 LLM Gateway & Structured Output Layer
- [`services/llm/gateway.py`](../services/llm/gateway.py) (`LLMGateway`): routes generation requests across `OllamaLLMProvider` (`http://127.0.0.1:11434`, GBNF JSON-schema `format` enforcement, native `logprobs` parsing), `OpenAILLMProvider`, `GeminiLLMProvider`, `LocalOpenAICompatibleProvider`, and deterministic `RuleBasedLLMProvider`.
- [`services/llm/json_utils.py`](../services/llm/json_utils.py): shared `extract_and_repair_json` and `repair_and_parse_with_retry` invoked identically by Stage 3 (`LLMNLPService`) and Stage 4 (`RoleService`).

### 1.3 Web Application Surface ([`server.py`](../server.py) & `web/pages/`)
- **14 Frontend Pages (`web/pages/`):** `architecture.html`, `asr.html`, `audio_input.html`, `audio_output.html`, `dialogue.html`, `interview_call.html`, `interview_profiling.html`, `interview_workspace.html`, `interviewer_copilot.html`, `interviewer_mode.html`, `interviewer_solo.html`, `nlp.html`, `roles.html`, `tts.html`.
- **REST & Page Route Surface (`server.py`):**
  - **Pipeline & Health:** `/api/health`, `/api/pipeline/providers`, `/api/pipeline/configure`, `/api/pipeline/status`, `/api/pipeline/process-text`, `/api/pipeline/process-audio`, `/api/pipeline/audio/{filename}`, `/api/pipeline/reset`.
  - **Stage Diagnostics & Benchmarks:** `/api/test/stage1-audio-input`, `/api/test/stage2-asr`, `/api/test/stage3-nlp`, `/api/test/stage4-dialogue`, `/api/test/stage5-tts`, `/api/asr/transcribe-file`, `/api/asr/benchmark`.
  - **Voice Cloning & DSP Lab:** `/api/tts/engines`, `/api/tts/voices`, `/api/tts/voices/{profile_id}/reference`, `PATCH/DELETE /api/tts/voices/{profile_id}`, `/api/tts/clone-voice`, `/api/tts/morph-voices`, `/api/tts/spectrogram`, `/api/tts/formant-synth`, `/api/tts/normalize-text`, `/api/tts/homographs`, `/api/tts/formant-vowels`, `/api/tts/benchmark`.
  - **Interview Engine, Context, Blueprints & LLM:** `/api/llm/{status,configure,test}`, `/api/roles`, `/api/roles/active`, `/api/roles/configure`, `/api/roles/interviewer/{turn,reset}`, `/api/interviewer/run-code`, `/api/blueprints*`, `/api/context/{active,reset,update}`.
  - *(Critical Upgrade Finding)*: Zero `WebSocket` / SSE full-duplex streaming audio endpoints currently exist in `server.py`; every turn executes sequentially via HTTP POST request-response.

### 1.4 Compute Environment Baseline
- **Local Workstation:** `Intel Core i5-9300H` (4C/8T), `16 GB RAM`, `NVIDIA GTX 1650 (4 GB VRAM)` (Ollama runs <=`~3.5 GB` SLMs at 100% VRAM; CTranslate2/ONNX run exclusively on CPU INT8/FP32). Dedicated Conda binary: `C:\Users\shive\.conda\envs\capstone_voice\python.exe`.
- **Cloud Training & Large Evaluation Node:** Kaggle `NvidiaTeslaT4` (`16 GB VRAM`) running `ml/training/kaggle_pipeline/kaggle_gpu_experiment.py` (v6 & v7 completed and pulled locally into `ml/results/` and `models/`).

---

## 2. Models, Adapters & Licensing Posture

| Component / Artifact | Weights / Architecture | Footprint | Verified License | Commercial Status |
| :--- | :--- | :---: | :--- | :--- |
| **Whisper `small.en` / `tiny.en`** | `Systran/faster-whisper-{small,tiny}.en` (INT8 CTranslate2) | `463.6 MB` / `74.5 MB` | `MIT` | **Cleared** |
| **Whisper `large-v3-turbo`** | `openai/whisper-large-v3-turbo` (`809M`, evaluated on T4) | `~1.5 GB` FP16 | `MIT` | **Cleared** |
| **Parakeet-TDT-0.6B-v3** | `nvidia/parakeet-tdt-0.6b-v3` (`sherpa-onnx` INT8) | `640.2 MB` (`843.9 MB` RSS) | `CC-BY-4.0` | **Cleared** |
| **`Qwen3-4B` (Primary Local SLM)** | `qwen3:4b` `Q4_K_M` (`Ollama`) + Kaggle **QLoRA** (`models/qwen3_4b_slu_lora/`) | `2.33 GB` disk (`3.33 GB` VRAM) | **`Apache-2.0`** | **Cleared** |
| **`DistilBERT-LoRA` (Edge Neural SLU)** | `distilbert-base-uncased` + LoRA `r=16` (`models/slu_lora_adapter/`) | `~0.25 GB` (`1,010,851` trainable) | **`Apache-2.0`** | **Cleared** |
| **`Phi-4-mini`** | Microsoft `phi4-mini` (`3.8B` `Q4_K_M`) | `2.32 GB` disk (`3.06 GB` VRAM) | **`MIT`** | **Cleared** |
| **`Qwen 2.5 Coder 1.5B`** | `qwen2.5-coder:1.5b` (`Q4_K_M`) | `0.92 GB` disk (`1.31 GB` VRAM) | `Apache-2.0` | **Cleared** (raw schema fails) |
| **`Qwen 2.5 3B`** *(Legacy v1 Only)* | `qwen2.5:3b` (`Q4_K_M`) | `1.80 GB` disk | **`qwen-research`** | **Deprecated in Prod** (non-commercial research license) |
| **`Llama 3.2 3B` / `Llama 3.1 8B`** | Meta Llama Q4_K_M | `1.88 GB` / `4.58 GB` | Meta Community License | Upper Reference Only (`700M MAU` restriction; 8B spills 43% CPU) |
| **`Kyutai Pocket TTS INT8`** | `sherpa-onnx` flow-matching (`24 kHz`, 5 steps) | `~189 MB` ONNX (`238.9 MB` RSS) | `CC-BY-4.0` weights / `MIT` / `Apache-2.0` runtime | **Cleared** (non-impersonation terms apply) |
| **`Piper VITS` Fallback** | `en_US-lessac-low` (`non-comm` Blizzard 2013) / `ryan-low` (`MIT`) + `piper1-gpl` | `~60 MB` each (`240.5 MB` RSS) | **`GPL-3.0-or-later`** (`piper1-gpl`) | Optional GPL fallback only |
| **`WeSpeaker CAM++` & `ECAPA512`** | `wespeaker_en_voxceleb_CAM++_LM.onnx` (`512-D`), `voxceleb_ECAPA512_LM.onnx` (`192-D`) | `27.9 MB` + `ECAPA` ONNX | **`CC-BY-4.0`** weights / `Apache-2.0` runtime | **Cleared** |
| **`g2p_en 2.1.0`** | ARPAbet phoneme predictor (replaced PyPI GPL `cmudict`) | Lightweight CPU | **`Apache-2.0`** | **Cleared** |

---

## 3. Verified Experimental Baseline Snapshot (`ml/results/*.json`)

### 3.1 Dual-Encoder Biometric Verification Audit ([`ml/results/speaker_encoder_eval.json`](../ml/results/speaker_encoder_eval.json))
Evaluated on `12` unseen speakers x `5` clips each (`120` same pairs / `1,650` diff pairs clean & 20 dB; `240` same / `3,300` diff cross-channel):

| Acoustic Condition | `RandomProj-128D` (Gap / $d'$ / AUC / EER) | `WeSpeaker CAM++ 512-D` (Gap / $d'$ / AUC / EER) | `WeSpeaker ECAPA512 192-D` (Gap / $d'$ / AUC / EER) |
| :--- | :---: | :---: | :---: |
| **`clean_librispeech_60`** | `0.0059` / `1.960` / `0.9555` / `0.1183` | **`0.2664`** / **`2.037`** / `0.9211` / `0.1492` | `0.1753` / `1.521` / `0.8726` / `0.1913` |
| **`awgn_20db_librispeech_60`** | `0.0051` / `1.973` / `0.9615` / `0.1165` | **`0.1454`** / **`1.432`** / **`0.8636`** / `0.2162` | `0.1374` / `1.404` / `0.8633` / **`0.2093`** |
| **`clean_librispeech_other_60`** | `0.0078` / `2.274` / `0.9745` / `0.0841` | **`0.3744`** / **`2.686`** / `0.9619` / `0.0919` | `0.3121` / `2.534` / **`0.9665`** / **`0.0835`** |
| **`cross_channel_telephony_eq` (300–3400 Hz + 18 dB)** | `0.0019` / `0.526` / `0.6379` / `0.4028` | `0.0618` / `0.452` / `0.6163` / **`0.4296`** | `0.0367` / `0.331` / `0.5854` / `0.4419` |
| **`cross_channel_other_60` (300–3400 Hz + 18 dB)** | `0.0028` / `0.935` / `0.7528` / `0.3244` | `0.0612` / `0.394` / `0.6069` / `0.4461` | `0.0956` / `0.727` / `0.6979` / **`0.3333`** |

- **Explainability ([`ml/results/explainability_summary.json`](../ml/results/explainability_summary.json)):** 8-band spectral occlusion across 12 speakers proves `CAM++` identity peaks at **`1.8–3.0 kHz` F3 (`0.7687` cosine drop)** and **`1.0–1.8 kHz` F2 (`0.6515`)**, over 2x stronger than `0–300 Hz` F0 (`0.3382`). On 120 EQ-tilted clips (`fig3`), `RandomProj-128D` collapses into a `[0.9916, 0.9976]` cone (`gap = 0.0060`) while `CAM++` holds `gap = 0.2781`.

### 3.2 Layer A ASR Benchmarks ([`asr_ablation.json`](../ml/results/asr_ablation.json), [`kaggle_asr_turbo.json`](../ml/results/kaggle_asr_turbo.json), [`svarah_en_in_eval.json`](../ml/results/svarah_en_in_eval.json))
*(Pre-hotfix CPU INT8 ablation + Kaggle T4 GPU benchmarks; note `phonetic_repair()` inflection/prose/determiner guards have since been hardened in code.)*

| Model / Config | IV Clean WER ($R_{\text{ent}}$) | Closed / Unseen $R_{\text{ent}}$ | 20 / 10 dB $R_{\text{ent}}$ | `MINDS-14 (498)` WER (Repairs / OverC) | `LibriSpeech` (Clean / 10 dB / Other) | `Svarah (1,000 en-IN)` WER | CPU RTF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `asr_tiny_baseline` | `0.1387` (`0.4783`) | `0.429 / 0.556` | `0.500 / 0.391` | `0.2904` (`0 / —`) | `0.0548 / 0.1464 / 0.1311` | — | `0.1122` |
| `asr_tiny_improved` | `0.1573` (**`0.9130`**) | `0.929 / 0.556` | `0.913 / 0.804` | `0.2870` (`84 / 0.0034`) | `0.0643 / 0.1402 / 0.1223` | — | `0.2801` |
| `asr_small_baseline` | `0.0849` (`0.7391`) | `0.786 / 0.667` | `0.674 / 0.630` | `0.2508` (`0 / —`) | `0.0348 / 0.0530 / 0.0858` | **`0.1521`** (`0.1357` non-dig / `0.2528` dig) | `0.6548` |
| `asr_small_soundex` | **`0.0435`** (`0.9565`) | `0.964 / 0.611` | `0.978 / 0.913` | `0.2486` (`15 / 0.0013`) | `0.0306 / 0.0436 / 0.0958` | — | `0.8946` |
| **`asr_small_improved` (G2P)** | **`0.0435`** (**`0.9783`**) | **`0.964 / 0.611`** | **`0.978 / 0.935`** | `0.2500` (`36 / 0.0025` *pre-fix*) | `0.0306 / 0.0436 / 0.0958` | — | `0.8958` |
| `asr_parakeet_tdt` (*pre-fix*) | `0.0642` (`0.8478`) | `0.857 / 0.389` | `0.848 / 0.804` | `0.3018` (`155 / 0.0105` *pre-fix*) | **`0.0221`** / `0.0467` / **`0.0605`** | — | **`0.1564`** |
| `whisper-large-v3-turbo` (T4) | `0.0787` (`50` clips) | — | — | `0.2517` (`CER=0.2013`) | — | **`0.0888`** (`0.0828` non-dig / `0.1261` dig) | GPU T4 |

- **Svarah Significance:** On `ai4bharat/Svarah` (`N=1,000`), `whisper-large-v3-turbo` cuts Corpus WER by **41.6% relative** (`0.1521 -> 0.0888`, CER `0.1354 -> 0.0413`; per-clip bootstrap `+0.533`, 95% CI `[0.0626, 1.1469]`, **`p < 0.001`**).

### 3.3 Layer B SLU & Rubric Calibration ([`slu_ablation.json`](../ml/results/slu_ablation.json), [`trained_slu_history.json`](../ml/results/trained_slu_history.json), [`qwen3_4b_qlora_results.json`](../ml/results/qwen3_4b_qlora_results.json))

| Config / Model | `MINDS-14 (56)` F1 (SchV) | `CLINC (156)` F1 (`OOS`) | `IV QA` F1 | Rubric $\rho$ (`MAE`) | Full `M14 (498)` Gold / Tiny-ASR | Full `CLINC (5500)` (`OOS`) | Lat (ms) | Cost / 1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `slu_rule_based` | `0.0000` (`1.0`) | `0.0000` (`0.0`) | `0.6905` | `0.5437` (`2.15`) | `0.0 / 0.0` | `0.0039` (`0.0`) | `0.1` | `$0.0000` |
| `slu_qwen15b_raw` | `0.0415` (`0.030`) | `0.0873` (`0.0`) | `0.6905` | `0.7511` (`2.02`) | — | — | `511.6` | `$0.0070` |
| `slu_qwen15b_schema` / `_repair` | `0.7289` (`1.0`) | `0.7171` / `0.6838` | `0.7404` / `0.6505` | `0.6568` (`2.22`) | — | — | `1159 / 1198` | `$0.0167` |
| `slu_llama3b_schema` / `cal_rep` | `0.8991` / `0.8937` | `0.7080` / `0.7120` (`0.658`) | `0.4837` | `0.3941` / `0.4062` | — | — | `1122 / 1040` | `$0.0148` |
| `slu_phi4_mini_repair` (`MIT`) | `0.9246` (`1.0`) | `0.7335` (`0.5159`) | `0.7888` | `0.3799` (`2.08`, uncal) | `0.8818 / 0.8662` | — | `1527.5` | `$0.0221` |
| `slu_qwen3_4b_schema` (uncal rubric) | `0.9873` (`1.0`) | `0.8378` (`0.7755`) | `0.8986` | `0.5735` (`1.92`, uncal) | — | — | `1106.8` | `$0.0141` |
| **`slu_qwen3_4b_repair` (Apache-2.0)** | **`0.9940`** (`1.0`) | **`0.8378`** (`0.7755`) | **`0.8986`** | **`0.8459 ± 0.0388`** (`1.42`, calibrated) | **`0.9315 / 0.9167`** | — | `1190.8` | **`$0.0160`** |
| `slu_llama8b_ref` *(n=2 seeds)* | `0.9262` (`1.0`) | **`0.9158`** (`0.7636`) | `0.6698` | `0.4188` *(task-6459 uncal)* | — | — | `4768.4` (`m14`) | `$0.1510` |
| **`slu_distilbert_lora` (Local CPU)** | `0.9456` (`1.0`) | **`0.9184`** (`0.7937`) | — | — | **`0.9520`** (`28.8ms`) / **`0.9441`** (`58.0ms`) | **`0.9038`** (`0.7092`, `31.5ms`) | `28.8–58.0` | **`$0.0011`** |
| **DistilBERT-LoRA (Kaggle 3-seed T4)** | — | — | — | — | **`0.9423 ± 0.0059`** (`2.2ms` GPU) | **`0.9112 ± 0.0073`** (`0.7523 ± 0.0374`) | `0.9–2.2` GPU | — |
| **Linear Probe Baseline (T4)** | — | — | — | — | `0.7471` (`0.7490` acc) | `0.6657` (`0.4724`) | `0.9–2.1` GPU | — |
| **Qwen3-4B Zero-Shot vs QLoRA (T4)** | — | `0.8431` -> **`0.9366`** (`0.8814`) | — | — | `0.8364` -> **`0.9484`** (`1.0` schV) | — | T4 FP16/NF4 | — |

- **TOST Non-Inferiority:** `slu_qwen3_4b_repair` vs `slu_llama8b_ref` on `MINDS-14` establishes **statistical non-inferiority** ($\Delta=-0.05$, observed `+0.0536`, 90% CI `[-0.0009, 0.1250]`, **$p_{\text{NI}} = 0.0035$**) at **9.4x lower cost**.

### 3.4 Layer C TTS ([`tts_ablation.json`](../ml/results/tts_ablation.json)) & Frozen E2E v1 Matrix ([`e2e_ablation_matrix.json`](../ml/results/e2e_ablation_matrix.json))
- **Layer C (12 speakers x 3 prompts = 36 pairs):**
  - `tts_pocket_clone`: `CAM++ = 0.5898` (specificity margin **`+0.1402`**, `d'=0.9224`, `AUC=0.7371`, `EER=0.3106`), `ECAPA512 = 0.8031` (**`+0.1030`** margin, `d'=0.8454`), `RT-WER = 0.1084`, `F0 MAE = 31.32 Hz`, **`ttfa_ms = 1913.7 ms`**, `avg_latency_ms = 2382.1 ms`, `CPU RTF = 0.5336`, `Peak RSS = 238.87 MB`, `short_ref_warnings = 12`, cost **`$0.0331/1k`**. Beats `tts_piper_dual_wsola` (`CAM++ = 0.4180`, `+0.1718` boost, **$p < 0.001$**) and beats non-cloning `tts_edgetts_ref` on `CAM++` specificity margin (`+0.0937`, **$p < 0.001$**) and `ECAPA` margin (`+0.0838`, **$p < 0.001$**).
- **Frozen `e2e_ablation_matrix_v1` (24 turns x 3 seeds = 72 evals/row, pre-v2 SLU/TTS/guard upgrade):**
  - `Baseline` (`tiny_base / qwen15b_raw / WSOLA`): Strict **`0.0%`**, `Q = 0.4192`, Lat `1495.4 ms`, Cost `$0.0208`.
  - `+A` / `+C` alone: **`0.0%`** Strict (`Q = 0.4551 / 0.4610`). `+B` alone: `8.33%` (`Q = 0.6771`).
  - **`Full (A+B+C)`**: **`41.67%`** Strict ($p < 0.001$ vs Baseline), **`Q = 0.7552 ± 0.0096`**, Lat `6868.4 ms`, Mem `3121.8 MB`, Cost `$0.0954/1k`.
  - **`Edge (A_tiny+B+C)`**: **`37.50%`** Strict ($p = 0.656$ vs Full), **`Q = 0.6864`**, Lat `3591.7 ms`, Mem `2670.4 MB`, Cost **`$0.0499/1k`**.
  - *(Note: [`ml/configs/e2e_ablation_matrix.json`](../ml/configs/e2e_ablation_matrix.json) is already updated to `e2e_ablation_matrix_v2` using `slu_qwen3_4b_repair` & `36` turns/seed, ready to run after the architecture upgrade.)*

---

## 4. Diagnosed Bottlenecks, Code Hotfixes Already Applied, & Residual Gaps

| # | Diagnosed Problem (Evidence) | Code Hotfix Applied Today (`2026-11-26`) | Verification Status | Remaining Work in v2.0 Massive Upgrade |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Un-gated `phonetic_repair()` Over-Correction:** `ParakeetTDTASR` (`words_meta=None`) hit `155` false `MINDS-14` repairs (`0.3018` WER despite `0.0605` on `LibriSpeech-other`), stripping `"a payment"->"payment"`, `"cards"->"card"`, `"amount"->"account"`, `"fees"->"freeze"`. Also no `bpe.vocab` shipped in `parakeet-tdt-0.6b-v3-int8` and quiet WAV inputs risk frontend attenuation. | Hardened [`services/asr/asr_services.py`](../services/asr/asr_services.py): added `_is_morphological_inflection`, expanded `_COMMON_PROSE_GUARD`, added `_FUNCTION_BIGRAM_GUARD` (`"a", "the", ...`), enforced strict gating when `words_meta=None`, added `<0.05` quiet-audio gain scaling on `ParakeetTDTASR`. | **Unit-tested & passed** (`t2`: `0` false repairs; `pytest` `5/5` passed). **Full re-sweep pending.** | Replace static file-based `ASRResult` with chunked **Streaming VAD + Partial Hypothesis Bus**; add Double-Metaphone/contextual neural rescoring for `en-IN` onset stops (`Pooja`). |
| **2** | **Rubric Central-Tendency Compression:** Zero-shot prompts without anchors compressed `interview_answers` into `{3, 4, 6, 7}`, yielding `rho = 0.38–0.57` (`MAE = 1.92–2.22`) across `Phi-4-mini`, `Llama`, and uncalibrated `Qwen3-4B`. | Added domain-agnostic 3-tier behavioral rubric anchors & `[1, 10]` clamping to [`RoleService.process_interviewer_turn`](../services/roles/role_service.py) (`calibrated_rubric=True`). | **Verified across 3 seeds on `Qwen3-4B`:** `rho` jumped `0.5735 ->` **`0.8459 ± 0.0388`**, `MAE` dropped `1.92 ->` **`1.4167`**. | Generalize rubric evaluation per active [`RoleBlueprint`](../services/blueprints/blueprint_contracts.py) criterion (`RubricCriterion` weighted sub-scores) rather than a single global prompt string. |
| **3** | **SLM Latency (`1.15s`) vs. Edge Classifier Speed (`30ms`):** Pure generative LLMs add >1 s/turn even for routine closed intents where `DistilBERT-LoRA` scores `0.94–0.95` F1 in `28–58 ms` CPU. | Implemented [`HybridConfidenceSLUService`](../services/nlp/nlp_services.py) (`>=0.85` conf & `>=0.25` margin fast-path -> `Qwen3-4B` slow-path) and wired `hybrid` / `distilbert_lora` in [`PipelineFactory`](../orchestrator/factory.py) & [`server.py`](../server.py). | **Compiled & wired** into `run_slu_ablation.py` & `run_e2e_ablation.py`; **full sweep deferred to v2.0 upgrade.** | Calibrate DistilBERT softmax via temperature scaling (`OOS` boundary), and route fused SLU + Dialogue state in one streaming SLM turn when escalating. |
| **4** | **Synchronous HTTP Turn Blocking (`3.6–6.9s` E2E):** `DecoupledVoicePipeline` blocks ASR -> NLP -> Dialogue -> full-WAV TTS sequentially over REST POST. | Instrumented Pocket TTS callback `ttfa_ms` (`1913.7 ms` vs `2382.1 ms` full) and atomic stage checkpoints in `run_e2e_ablation.py`. | Live server & unit tests pass cleanly on HTTP baseline. | **Core v2.0 Target:** Event-driven **WebSocket full-duplex audio + barge-in** -> streaming Whisper/Parakeet -> token-streamed LLM -> sentence-chunked Pocket TTS (`TTFA < 1.0 s`). |
| **5** | **Cross-Channel Narrowband (`<3.5 kHz`) ASV Blind Spot:** `CAM++` & `ECAPA512` EER degrades from `0.08–0.19` clean to `0.33–0.45` on `300–3400 Hz` telephony EQ with zero caller warning. | Added Welch 95% spectral rolloff computation (`spectral_rolloff_95_hz <= 3600 Hz` -> `narrowband_warning=True`) in [`VoiceCloner.extract_acoustic_profile_from_wav`](../services/tts/voice_cloner.py). | **Verified** on wideband (`4500 Hz -> warn=False`) vs 3.4 kHz telephony LPF (`226.6 Hz -> warn=True`). | Integrate neural Speech Bandwidth Extension (BWE) or AS-Norm score calibration for telephony inputs. |
| **6** | **Noisy `MINDS-14` References & ITN WER Inflation:** `MINDS-14` ref strings contain ASR noise (~25% WER even on `whisper-large-v3-turbo`), and `%`/hyphen digits inflate synthetic WER (`"95-28%"`). | Validated clean `ai4bharat/Svarah` (`N=1,000`, `turbo=0.0888`) via Kaggle v7; added [`compute_itn_wer`](../ml/metrics/eval_metrics.py) + hooked into [`run_asr_ablation.py`](../ml/experiments/run_asr_ablation.py). | **Verified** on `eval_metrics` (`"95-28%"` vs `"95 minus 28 percent"` drops `0.57 -> 0.14` delta cleanly). | Adopt `Svarah` + ITN-aware dual WER reporting as primary ASR standard across all v2.0 benchmarks. |

---

## 5. Architectural Target for the Upcoming Massive Upgrade (`v2.0` Blueprint)

When refactoring from `v1.x` -> `v2.0`, preserve the `services/` Clean Contracts discipline and `ml/` Zero-Leakage Statistical Rigor while transforming these structural axes:

1. **Full-Duplex Streaming Orchestrator (`server.py` + `orchestrator/`):** Replace unary blocking HTTP turn execution with an `asyncio` WebSocket session bus (`AudioFrame -> VAD/Endpoint -> Incremental ASR -> Streaming SLU/Dialogue -> Sentence-Chunked TTS -> AudioChunk`) supporting interruption/barge-in.
2. **Unified SLU + Dialogue Router:** Merge closed-intent fast routing (`DistilBERT-LoRA` `~30 ms` or `Qwen3-4B QLoRA` adapter `0.9484 / 0.9366`) with blueprint-aware [`RoleService`](../services/roles/role_service.py) multi-criterion rubric scoring (`rho >= 0.85`), eliminating redundant LLM roundtrips on interview turns.
3. **ASR Decode Stack Unification:** Pair fast CPU `Parakeet-TDT-0.6B-v3` (`RTF=0.156`) and `Whisper` (`small.en` / `large-v3-turbo`) behind the hardened morphological/prose `phonetic_repair()` guard, reporting both verbatim WER and ITN WER.
4. **Adaptive Voice Cloning & Quality Gate:** Combine `Pocket TTS INT8` (`24 kHz`), `>=5 s` enrollment + `>=3.0 s` synthesis floor (eliminating `<0.50 CAM++` fragments), and automatic `narrowband_warning` gating before speaker enrollment/similarity checks.
5. **Re-Benchmark & Manuscript Alignment:** Execute the complete `e2e_ablation_matrix_v2` (36 turns x 3 seeds, dual `CAM++/ECAPA512`, `TTFA`, `Hybrid` frontier) after the upgrade and synchronize the frozen v1 IEEE papers ([`CAPSTONE_REPORT_IEEE.md`](CAPSTONE_REPORT_IEEE.md), [`RESEARCH_PAPER_IEEE.md`](RESEARCH_PAPER_IEEE.md)) and `.docx` builds once `v2.0` lands.
