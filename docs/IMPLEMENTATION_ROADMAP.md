# Research-First Implementation Specification & 10-Session Plan

**Project**: 6Works — Edge-First Conversational Voice Agent & Technical Interviewer Pipeline  
**Target Environment**: Windows CPU / 4 GB GTX 1650 (`capstone_voice` Python 3.10.22, Ollama local runtime, ONNX Runtime / CTranslate2)  
**Verified Baseline Check**: `45 passed` in `test_pipeline.py` (Commits `01934c6`, `c2089c5`).  
**Constraint Policy**: Strict SOLID decoupling between production `services/` and research `ml/`; permissive commercial licensing only (`MIT`, `Apache-2.0`, `CC-BY-4.0`); zero speculative boilerplate; every improvement row backed by a config file in `ml/configs/` and logged results in `ml/results/`.

---

## 1. Phase 0: Problem Lock, Pareto Objective & Mathematical Definitions

We frame the 6-stage conversational voice pipeline as a constrained **Accuracy-vs-Cost Pareto optimization problem** on low-cost edge hardware (`<= 4 GB` VRAM / 4-core CPU, `$0` cloud token cost):

$$\max_{\theta_A, \theta_B, \theta_C} \mathcal{U}(\text{WER}, R_{\text{entity}}, F_1, S_{\text{cos}}, \text{STOI}) \quad \text{s.t.} \quad L_{\text{e2e}} \le 2.5\text{ s}, \;\; C_{1\text{k}} \le \$0.05$$

where individual layer metrics (implemented in `ml/metrics/eval_metrics.py`) are formally defined as:
1. **Stage 2 ASR**: Word Error Rate $\text{WER} = \frac{S + D + I}{N}$, Character Error Rate $\text{CER}$, Target Entity Recall $R_{\text{entity}} = \frac{|\mathcal{E}_{\text{pred}} \cap \mathcal{E}_{\text{gold}}|}{|\mathcal{E}_{\text{gold}}|}$, Real-Time Factor $\text{RTF} = \frac{T_{\text{proc}}}{T_{\text{audio}}}$, and word log-probability calibration (`word_timestamps=True`).
2. **Stage 3/4 SLU & Rubric**: Multi-class Intent $\text{Macro-}F_1$, strict JSON Schema Compliance $S_{\text{valid}} \in [0, 1]$, Rubric Score MAE / Spearman $\rho$ against an `8B` teacher (`llama3.1:latest`), and per-turn token latency.
3. **Stage 5 TTS & Cloning**: Neural Speaker Verification Cosine Similarity $S_{\text{cos}}(e_{\text{ref}}, e_{\text{synth}}) = \frac{e_{\text{ref}}^\top e_{\text{synth}}}{\|e_{\text{ref}}\|_2 \|e_{\text{synth}}\|_2}$ using an independent neural encoder (`WeSpeaker CAM++`), Roundtrip Whisper $\text{WER}_{\text{TTS}}$, Short-Time Objective Intelligibility $\text{STOI} \in [0, 1]$, Fundamental Frequency MAE $| \bar{F}_{0,\text{ref}} - \bar{F}_{0,\text{synth}} |$, and CPU $\text{RTF}$.
4. **Statistical Validity**: Minimum 3 random seeds `[42, 1337, 2026]` for stochastic runs (`mean ± std`), evaluated for significance over baselines via **Paired Bootstrap Resampling** (`n_resamples=1000`, two-sided $p < 0.05$).

---

## 2. Phase 1: Literature Positioning & Exact Novelty Gaps

Verification against closest peer-reviewed baselines ensures our layer upgrades contribute distinct engineering/research value rather than duplicating prior work:

| Layer | Closest Prior Literature (Verified) | What Prior Work Does | Our Exact Gap & Edge Contribution |
| --- | --- | --- | --- |
| **Layer A: ASR** | Sun et al. (*Interspeech 2023*, TCPGen+Whisper), Li et al. (*LREC-COLING 2024*, CB-Whisper KWS), Yang et al. (*ICASSP 2024*, PromptASR), Nguyen & Thach (*Interspeech 2024*) | Trains auxiliary CNN keyword spotters, TCPGen pointer networks, or cross-attention text encoders | **Zero-training dual-path edge biasing**: dynamic context extraction from `BlueprintService`/`InterviewContextManager` driving Whisper `initial_prompt` + CTranslate2 `hotwords` logit biasing + `rapidfuzz` metaphone post-correction on INT8 CPU models (`tiny.en`/`small.en`) |
| **Layer B: SLU** | Everson et al. (*IEEE ICASSP 2024*, WCN+LLM ICL), Ma et al. (*Interspeech 2023*, N-best T5), Chen et al. (*NeurIPS 2023*, Hyporadise) | Consumes dense ASR confusion networks/N-best lattices using large cloud or T5 seq2seq correctors | **Quantized 1.5B/3B Q4_K_M recovery**: replaces naive `json.loads` drop-off in `LLMNLPService` / `RoleService` with ASR phonetic pre-repair + deterministic schema-repair retry + 2-shot rubric anchoring to match `8B` F1 at `<40%` CPU latency |
| **Layer C: Cloning** | Wan et al. (*IEEE ICASSP 2018*, GE2E), Jia et al. (*NeurIPS 2018*), Kim et al. (*ICML 2021*, VITS), Wang et al. (*arXiv:2303.00332*, CAM++), Kyutai (*Pocket TTS 2026*) | Evaluates heavy GPU autoregressive/flow codecs (`XTTS`, `VALL-E`, `ZipVoice NC`) or isolated TTS modules | **Commercially permissive ONNX CPU voice cloning**: replaces our baseline's seeded random projection + Piper WSOLA DSP shortcut with `WeSpeaker CAM++` (27.9 MB) neural embeddings + `Kyutai Pocket TTS INT8` (93.8 MB, `CC-BY-4.0`) evaluated cleanly inside a 6-stage turn budget |

---

## 3. Codebase Defects Identified (Pre-Implementation Audit)

Exact defects observed directly in current source files that Sessions 2–8 will eliminate:
1. **`services/tts/voice_cloner.py` (L379–L388)**: `extract_acoustic_profile_from_wav` computes embeddings using a Gaussian random matrix seeded by `int(f0_mean * 100)`, yielding non-discriminative speaker representations.
2. **`services/tts/tts_services.py` (L155–L331)**: `NeuralVoiceCloningTTSService` loads single-speaker Piper ONNX (`lessac-low` / `ryan-low`) and ignores `profile.ref_audio_path` and `profile.embedding_vector`.
3. **`services/asr/asr_services.py` (L86–L178)**: `WhisperTransformerASR` hardcodes a static 35-name `DEFAULT_PROMPT`, never exposes `hotwords` or `word_timestamps=True` (already supported by installed `faster_whisper.WhisperModel.transcribe`), and performs no dynamic blueprint/context injection or phonetic repair.
4. **`services/nlp/nlp_services.py` (L120–L139) & `services/roles/role_service.py` (L260–L288)**: Both parse raw JSON directly with `json.loads()` and immediately collapse to regex heuristics (`RuleBasedNLPService` / `tech_keywords`) if a compact `1.5B`/`3B` SLM emits minor syntax noise under ASR-corrupted transcripts.

---

## 4. Benchmark Datasets & Leakage Prevention Protocol

All remote manifests are fetched in **Session 4** via lightweight streaming slicing (HuggingFace `datasets-server.huggingface.co`, `<= 150 MB` total disk footprint) and hashed under `ml/datasets/`:

| Dataset | Exact Source & Verified Config | Size & Role | Split & Anti-Leakage Rule | License |
| --- | --- | --- | --- | --- |
| **LibriSpeech (`test-clean`)** | `openslr/librispeech_asr` (`config=clean`, `split=test`) — Panayotov et al. (*ICASSP 2015*) | 60 FLAC utterances across >=10 distinct `speaker_id`s (min 5 utt/speaker) | **ASR Clean/AWGN (20 dB, 10 dB)** + **Unseen Multi-Speaker Verification & Zero-Shot Cloning** (1 enrollment clip vs. held-out clips; zero self-test). State pretraining contamination caveat for Whisper. | CC-BY-4.0 |
| **MINDS-14 (`en-US/GB/AU`)** | `PolyAI/minds14` (`config=en-US/GB/AU`, `split=train`) — Gerz et al. (*Interspeech 2021*) | 140 WAV audio-to-intent samples across 14 classes (stratified across accents) | SHA-256 deduplicated by `transcription`; **35% dev calibration / 15% val / 50% blind held-out test**; zero tuning on test. Tests true **Audio -> ASR -> SLU** degradation. | CC-BY-4.0 |
| **CLINC-OOS (`small`)** | `clinc/clinc_oos` (`config=small`, `validation`/`test`) | 200-sample balanced 20-intent + `oos` slice for SLM JSON & schema compliance | Standard HF `validation` (tuning) / `test` (reporting only). | CC-BY-3.0 |
| **Domain Interview Stress** | `ml/datasets/interview_stress_v1.jsonl` (audio rendered via held-out `EdgeTTS en-IN-PrabhatNeural` & `en-US-JennyNeural`) | 40 interview QA turns + 25 South-Asian/tech proper-noun utterances (`Sachin`, `Aradhya`, `FastAPI`, `Redis`, `PyTorch`) | Audio generated strictly by external Azure `EdgeTTS` engine so neither `Piper` nor `Pocket TTS` shares generator characteristics during evaluation. | MIT (in-repo) |

---

## 5. License-Verified Open Model & API Matrix

| Role | Artifact / ID | Size | Code / Weights License | Exact Verified API / File Pattern |
| --- | --- | --- | --- | --- |
| **Speaker Encoder** | `wespeaker_en_voxceleb_CAM++_LM.onnx` | 27.9 MB | Apache-2.0 / CC-BY-4.0 | `sherpa_onnx.SpeakerEmbeddingExtractor(SpeakerEmbeddingExtractorConfig(model=..., num_threads=2, provider="cpu"))` |
| **Zero-Shot TTS** | `sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2` | 93.8 MB | MIT / CC-BY-4.0 *(user ref only)* | `OfflineTts(OfflineTtsConfig(model=OfflineTtsModelConfig(pocket=OfflineTtsPocketModelConfig(...))))` with `GenerationConfig(reference_audio=..., reference_sample_rate=...)` *(Note verified spelling: `reference_sample_rate`)* |
| **Excluded Model** | `zipvoice-distill-int8` | 104.1 MB | Apache-2.0 / **CC-BY-NC** | Excluded due to non-commercial weight restriction |
| **ASR** | `tiny.en`, `base.en`, `small.en` (CTranslate2 INT8) | 39M / 74M / 244M | MIT | `WhisperModel.transcribe(..., initial_prompt=..., hotwords=..., word_timestamps=True, vad_filter=...)` |
| **SLM Suite** | `qwen2.5-coder:1.5b`, `llama3.2:3b`, `qwen2.5:3b`, `llama3.1:latest` | 1.5B–8.0B (Q4_K_M) | Apache-2.0 / Llama 3 | Local Ollama `/api/chat` (`127.0.0.1:11434`) |

---

## 6. Session-by-Session Concrete Execution Specification

### Session 1 — Foundation (`Done`, Commits `01934c6`, `c2089c5`)
- **Completed**: Installed `sherpa-onnx==2.0.3`, `jiwer==4.0.0`, `soundfile==0.14.0`, `pystoi==0.4.1`, `scikit-learn==1.6.1`; built `ml/metrics/eval_metrics.py` & `ml/configs/baseline_pipeline.json`; verified `45 passed` tests.

### Session 2 — Neural Speaker Encoder (`WeSpeaker CAM++ LM` ONNX) — `Done`
- **Status**: Verified `46 passed` in `test_pipeline.py`; WeSpeaker CAM++ LM (`27.93 MB`, `512-D`) active in `VoiceCloner`; benchmark saved in `ml/results/speaker_encoder_eval.json`.

### Session 3 — True Zero-Shot Neural Voice Cloning (`Kyutai Pocket TTS INT8`) — `Next`
- **Files**: `services/tts/tts_services.py`, `services/tts/__init__.py`, `orchestrator/factory.py`, `server.py`, `web/pages/tts.html`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Download & extract `sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2` (93.8 MB) into gitignored `models/pocket_tts/`; auto-discover `.onnx` & `.json` files via deterministic globbing for `OfflineTtsPocketModelConfig`.
  2. Implement `PocketTTSCloningService(ITTSService)` (~90 lines max): loads cached `sherpa_onnx.OfflineTts`, loads target `profile.ref_audio_path` via `VoiceCloner.load_audio_signal`, builds `GenerationConfig(reference_audio=..., reference_sample_rate=16000)`, synthesizes zero-shot cloned speech, normalizes peak PCM safely to WAV, and falls back gracefully to `NeuralVoiceCloningTTSService` if weights are absent.
  3. Wire `"pocket_tts"` into `PipelineFactory`, `GET /api/tts/engines`, `POST /api/test/stage5-tts`, and the engine dropdown in `web/pages/tts.html`.
- **Exit Gate**: Synthesize real cloned WAV from a reference clip via API & unit test (`duration > 0.5s`, valid WAV headers), zero regression across suite.

### Session 4 — Reproducible Benchmark Datasets & Dataset Card
- **Files**: `ml/datasets/prepare_benchmarks.py`, `ml/cards/dataset_card.md`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Write clean fetcher caching compact slices (`<=150 MB` total in gitignored `storage/benchmarks/` + versioned manifest metadata):
     - `librispeech_60.json` (60 FLACs, >=10 speakers) + deterministic AWGN generator helper (`add_awgn(audio, snr_db, seed)` for `20 dB` & `10 dB`, never baked redundantly to disk).
     - `minds14_en_140.json` (stratified 35/15/50 splits across `en-US`, `en-GB`, `en-AU`, deduplicated by transcript hash).
     - `clinc_oos_200.json` (`small` val/test split).
     - `interview_stress_v1.jsonl` (40 QA turns + 25 proper nouns, rendered via held-out `EdgeTTS` MP3->WAV so no generator overlap exists with Piper or Pocket TTS).
  2. Write `ml/cards/dataset_card.md` covering collection, splits, leakage checks, noise protocol, and ethical/bias notes.
- **Exit Gate**: Manifest verification test confirms non-empty files, exact split disjointness (`set(val) & set(test) == empty`), and schema integrity.

### Session 5 — Layer A (Stage 2 ASR) Baseline + Contextual Logit Biasing & Phonetic Repair
- **Files**: `services/asr/asr_services.py`, `ml/configs/asr_*.json`, `ml/experiments/run_asr_ablation.py`, `ml/results/asr_ablation.json`.
- **Implementation Spec**:
  1. Upgrade `WhisperTransformerASR` minimally without bloat:
     - Expose `hotwords: Optional[str] = None` (passed to `model.transcribe(..., hotwords=...)`) and `word_timestamps: bool = False` (returning word `{word, start, end, probability}` in `ASRResult.metadata["words"]`).
     - Add clean `build_dynamic_bias_prompt(role_terms, candidate_names)` helper + optional lightweight `phonetic_repair(transcript, lexicon)` post-processor using `rapidfuzz.fuzz.ratio` / normalized Soundex/edit distance on domain terms when word confidence `< threshold`.
  2. Run 4 reproducible configs across clean, `20 dB`, `10 dB`, `MINDS-14` accents, and `Interview Stress`:
     - `asr_tiny_baseline.json` (`tiny.en`, `vad=false`, no prompt, no hotwords)
     - `asr_small_baseline.json` (`small.en`, `vad=true`, static prompt)
     - `asr_tiny_improved.json` (`tiny.en` + dynamic `initial_prompt` + `hotwords` + phonetic repair)
     - `asr_small_improved.json` (`small.en` + dynamic `initial_prompt` + `hotwords` + phonetic repair)
  3. Record WER, CER, $R_{\text{entity}}$, RTF, Latency ms, Peak RSS, and **Paired Bootstrap $p$-values** + failure examples.

### Session 6 — Layer B (Stage 3/4 SLU & Rubric) Quantized SLM Recovery & Cost Pareto
- **Files**: `services/nlp/nlp_services.py`, `services/roles/role_service.py`, `ml/configs/slu_*.json`, `ml/experiments/run_slu_ablation.py`, `ml/results/slu_ablation.json`.
- **Implementation Spec**:
  1. Extract ONE shared tiny helper `_extract_and_repair_json(raw_text, expected_keys)` (preventing duplication between `LLMNLPService` and `RoleService`) that strips markdown fences, fixes trailing commas / unquoted keys, and runs at most 1 constrained schema-repair retry if invalid.
  2. Add optional **Phonetic ASR Error Pre-Repair** hook before LLM/Rule-Based parsing for corrupted transcripts fed from `asr_tiny_baseline` / `asr_small_improved`.
  3. Evaluate 5 configurations across 3 seeds (`[42, 1337, 2026]`) on `MINDS-14` (gold + real ASR transcripts!), `CLINC-OOS`, and `Interview Stress` rubric alignment (vs `llama3.1:8b` judge):
     - `RuleBasedNLPService`, `qwen2.5-coder:1.5b` (raw vs. +repair), `llama3.2:3b` (raw vs. +repair), `qwen2.5:3b` (+repair), and `llama3.1:latest` (8B ceiling).
  4. Report `Macro-F1 (mean ± std)`, Schema Compliance %, Latency ms, Memory GB, Cost / 1k turns, Paired Bootstrap significance vs. `8B` ceiling!

### Session 7 — Layer C (Stage 5 TTS) Cloning Ablation & Multi-Modal Explainability
- **Files**: `ml/configs/tts_*.json`, `ml/experiments/run_tts_ablation.py`, `ml/experiments/explainability.py`, `ml/results/tts_ablation.json`, `ml/results/figures/`.
- **Implementation Spec**:
  1. Benchmark across unseen LibriSpeech test speakers (`>=10` ref clips × held-out texts) over 3 seeds:
     - `Pyttsx3` (SAPI5)
     - `Piper VITS` (raw neutral vs. dual-gender + WSOLA)
     - `Pocket TTS INT8` (zero-shot conditioned on reference WAV)
     - `EdgeTTS` (cloud upper reference)
  2. Log `CAM++ Cosine Similarity`, Whisper-Roundtrip `WER`, `STOI`, `F0 MAE`, and `CPU RTF`.
  3. Generate **Explainability Visualizations** (`ml/results/figures/*.png`):
     - **Layer A**: Word-level Whisper confidence & deletion/substitution error heatmap over clean vs. noisy SNR utterances (`word_timestamps=True`).
     - **Layer C**: Spectral Mel-band Occlusion Saliency (sensitivity of CAM++ cosine similarity to masking 8 frequency bands) + 2D PCA/t-SNE projection comparing `RandomProj-128D` vs `WeSpeaker CAM++` clusters across LibriSpeech speakers!

### Session 8 — End-to-End 6-Stage Ablation Matrix, Pareto Curve & Error Analysis
- **Files**: `ml/configs/e2e_ablation_matrix.json`, `ml/experiments/run_e2e_ablation.py`, `ml/results/e2e_ablation_matrix.json`, `ml/results/error_analysis.md`.
- **Implementation Spec**:
  1. Execute the full 8-row **Ablation Matrix** (from `Research-First Capstone Plan` Section 5) across 3 seeds `[42, 1337, 2026]` on end-to-end spoken Audio -> ASR -> SLU -> TTS turns:
     - `Baseline`, `+A`, `+B`, `+C`, `Full (A+B+C)`, `Full - A`, `Full - B`, `Full - C`.
  2. Compute Composite End-to-End Task Success (`mean ± std`), E2E Latency (ms), Peak Memory (MB), Cost / 1k turns, and Paired Bootstrap significance tests ($p$-values).
  3. Plot **Accuracy-vs-Cost Pareto Front** figure and compile **structured Failure & Error Analysis** (showing actual transcript, predicted JSON, TTS failure audio metadata, and root-cause diagnosis, including negative/failed experiment runs).

### Session 9 — Production Hardening, CI/CD, Model Cards & Live Soak Validation
- **Files**: `.github/workflows/ci.yml`, `Dockerfile` *(minimal multi-stage or slim Python 3.10 CPU)*, `ml/cards/model_cards.md`, `orchestrator/factory.py`.
- **Implementation Spec**:
  1. Wire winning defaults cleanly into `PipelineFactory` and `server.py` (configurable via env/runtime without breaking existing callers).
  2. Add lightweight GitHub Actions CI running `pytest test_pipeline.py` on PRs (skipping/mocking external network or missing weights gracefully).
  3. Finalize `ml/cards/model_cards.md` (intended use, training provenance, license, contamination risk, hardware telemetry).
  4. Verify live FastAPI endpoints under real microphone/file requests and capture UI screenshots for Ch 6.

### Session 10 — Publication-Grade Capstone Report & IEEE Research Paper
- **Files**: `docs/CAPSTONE_REPORT_IEEE.md`, `docs/RESEARCH_PAPER_IEEE.md`.
- **Implementation Spec**:
  1. Populate all 10 chapters + front/back matter using **only** directly verified results, plots, model/dataset cards, verified IEEE/ICML/NeurIPS/Interspeech citations, and student/faculty metadata (**Shivendra Pratap**, Section **K23DT**, Batch **2027**, Roll No **33**, Subject **Deep Learning**, Prof. **Anjar Hussain Lone**, plus `[MOCK]` placeholders only for unverified institutional headers/dates/survey placeholders).
  2. Produce the tight **8-page IEEE Research Paper** distilled directly from Phases 0–5 with exact tables, significance tests, and Pareto plot.
