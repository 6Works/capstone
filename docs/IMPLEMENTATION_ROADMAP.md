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

### Session 3 — True Zero-Shot Neural Voice Cloning (`Kyutai Pocket TTS INT8`) — `Done`
- **Status**: Verified `47 passed` in `test_pipeline.py` (Commits pending/Session 3); Kyutai Pocket TTS INT8 (`24 kHz`, CPU RTF `0.482`, `bria.wav` CAM++ CosSim `0.7444`, roundtrip WER `0.0`) integrated in `PocketTTSCloningService`, `PipelineFactory`, `/api/test/stage5-tts`, and `/tts` studio UI.

### Session 4 — Reproducible Benchmark Datasets & Dataset Card — `Done`
- **Files**: `ml/datasets/__init__.py`, `ml/datasets/prepare_benchmarks.py`, `ml/datasets/manifests/{librispeech_60,minds14_en_140,clinc_oos_200,interview_stress_v1}.json`, `ml/cards/dataset_card.md`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Write clean fetcher caching compact slices (`<=150 MB` total in gitignored `storage/benchmarks/` + versioned manifest metadata):
     - `librispeech_60.json` (60 FLACs, 12 speakers, `enroll:12 / val:12 / test:36`) + deterministic AWGN generator helper (`add_awgn(audio, snr_db, seed)` for `20 dB` & `10 dB`, never baked redundantly to disk).
     - `minds14_en_140.json` (140 WAVs, 14 intents x 10 items, stratified `dev:42 / val:28 / test:70` across `en-US`, `en-GB`, `en-AU`, deduplicated by `text_sha256`).
     - `clinc_oos_200.json` (200 text samples, 15 classes incl. `oos`, `val:100 / test:100`).
     - `interview_stress_v1.json` (40 QA turns across 7 intents & 3 rubric tiers + 25 South-Asian/tech proper-noun turns, `val:15 / test:50`, rendered via held-out `EdgeTTS` MP3->WAV so no generator overlap exists with Piper or Pocket TTS).
  2. Write `ml/cards/dataset_card.md` covering collection, splits, leakage checks (`0` cross-split `text_hash` collisions), noise protocol, and ethical/bias notes.
- **Status**: Verified `49 passed in 50.65s` in `test_pipeline.py`; total local audio cache `37.70 MB` across 265 files (`0` cross-split collisions across all 4 manifests).

### Session 5 — Layer A (Stage 2 ASR) Baseline + Contextual Logit Biasing & Phonetic Repair — `Done`
- **Files**: `services/asr/__init__.py`, `services/asr/asr_services.py`, `ml/configs/asr_*.json`, `ml/experiments/run_asr_ablation.py`, `ml/results/asr_ablation.json`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Upgraded `WhisperTransformerASR` minimally without bloat: exposed `hotwords`, `word_timestamps` (`ASRResult.metadata["words"]`), `condition_on_previous_text`, `repetition_penalty`, `no_repeat_ngram_size`, `hallucination_silence_threshold`, plus `build_dynamic_bias_prompt` and confidence-gated `phonetic_repair` (Soundex + Rapidfuzz + bigram join).
  2. Evaluated 4 configs across `interview_stress_v1` (50 clean + 20 `20dB` + 20 `10dB`), `minds14_en_140` (70 test), and `librispeech_60` (36 clean + 12 `20dB` + 12 `10dB`).
  3. Discovered and resolved 39M `tiny.en` prompt/hotword echo hallucination (`negative_results.tiny_unguarded_hotwords_v1`: WER `0.3292`, RTF `1.0014`) via capacity-gated hotwords (`hw=false` on tiny) + decoding guards + `hallucination_silence_threshold=0.4`.
- **Status**: Verified `51 passed in 53.05s` in `test_pipeline.py`; `asr_tiny_improved` (`74.5 MB`, Peak `266.9 MB`, RTF `0.266`) beat `asr_small_baseline` (`463.6 MB`, Peak `718.4 MB`, RTF `0.608`) on domain entity recall (`R_entity=0.913` vs `0.7391`, `p=0.018`); `asr_small_improved` reached `WER=0.0435` (`-48.8%` vs baseline, `p<0.001`), `R_entity=0.9565` clean / `0.9783` @ `20 dB` / `0.913` @ `10 dB`, and `0.0306` clean LibriSpeech WER.

### Session 6 — Layer B (Stage 3/4 SLU & Rubric) Quantized SLM Recovery & Cost Pareto — `Done`
- **Files**: `services/contracts.py`, `services/llm/__init__.py`, `services/llm/json_utils.py`, `services/llm/providers/ollama_provider.py`, `services/nlp/__init__.py`, `services/nlp/nlp_services.py`, `services/roles/role_service.py`, `ml/configs/slu_*.json`, `ml/experiments/run_slu_ablation.py`, `ml/results/slu_ablation.json`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Extracted shared `extract_and_repair_json` and `repair_and_parse_with_retry` in `services/llm/json_utils.py`, eliminating duplicated JSON parsing between `LLMNLPService` and `RoleService`.
  2. Added `preprocess_asr_transcript` ASR pre-repair hook and evaluated 7 reproducible configs across 3 seeds (`[42, 1337, 2026]`) on `MINDS-14` (`28` balanced test items across gold + `asr_tiny_baseline` + `asr_small_improved`), `CLINC-OOS` (`35` items incl. `7 oos`), `Interview QA` (`30` items), and `Interview Rubric` (`10` stratified test answers).
- **Status**: Verified `53 passed in 70.37s` in `test_pipeline.py`; `slu_qwen15b_repair` lifted MINDS-14 F1 `0.0 -> 0.8294 ± 0.0014` (`p=0.0`) & schema validity `0.0 -> 1.0`; `slu_llama3b_repair` lifted Interview Rubric schema validity `0.50 -> 1.0` & MINDS-14 F1 to `0.9262 ± 0.0`; `slu_qwen3b_repair` (`1.80 GB`, `2.23 GB VRAM`, `$0.0172 / 1k turns`) matched `llama3.1:8b` (`$0.1698 / 1k turns`) on MINDS-14 (`0.8667` vs `0.8619`, `p=0.798`) while beating 8B on Interview SLU (`0.7606` vs `0.6939`), Rubric Spearman rho (`0.6897` vs `0.4188`), and noisy `asr_tiny_baseline` (`0.9619` vs `0.9238` without pre-repair).

### Session 7 — Layer C (Stage 5 TTS) Cloning Ablation & Multi-Modal Explainability — `Done`
- **Files**: `services/tts/tts_services.py`, `ml/configs/tts_*.json`, `ml/experiments/run_tts_ablation.py`, `ml/experiments/explainability.py`, `ml/results/tts_ablation.json`, `ml/results/explainability_summary.json`, `ml/results/figures/*.png`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Benchmarked 5 TTS configs across all 12 unseen LibriSpeech test speakers x 3 held-out utterances (`36` pairs/engine) on held-out `val` target clips (`Pyttsx3`, unconditioned `Piper raw`, `Piper dual+WSOLA`, `Pocket TTS INT8` 5 steps, and `EdgeTTS` cloud reference).
  2. Rendered 3 multi-modal 300-DPI explainability visuals in `ml/results/figures/`: Whisper word-confidence heatmap across SNRs (`fig1`), WeSpeaker CAM++ 8-band spectral occlusion saliency (`fig2`, confirming F2/F3 formants `1.0–4.5 kHz` dominate cosine drop `0.620–0.769` vs `0.338` for `0–300 Hz` F0), and `RandomProj-128D` vs `CAM++-512D` + `Pocket TTS` t-SNE clustering (`fig3`).
- **Status**: Verified `55 passed in 59.85s` in `test_pipeline.py`; `Pocket TTS` lifted held-out `CAM++` speaker similarity to `0.5651` (`0.5896` vs enroll) from `0.4159` on `Piper dual+WSOLA` (`+0.1493`, `p=0.0`; WSOLA DSP degraded waveform `STOI=0.2843`) while running offline at `0.4384` CPU RTF and `239.96 MB` Peak RSS.

### Session 8 — End-to-End 6-Stage Ablation Matrix, Pareto Curve & Error Analysis — `Done`
- **Files**: `ml/configs/e2e_ablation_matrix.json`, `ml/experiments/run_e2e_ablation.py`, `ml/results/e2e_ablation_matrix.json`, `ml/results/error_analysis.md`, `ml/results/figures/fig4_e2e_pareto_front.png`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Executed all 8 core ablation rows + `Edge (A_tiny+B+C)` (`9` rows total) across 3 seeds `[42, 1337, 2026]` on 24 balanced multi-domain spoken turns/seed (`4` proper nouns, `10` interview QA, `10` MINDS-14 across 12 unseen LibriSpeech speakers).
  2. Computed Conjunctive E2E Task Success, Continuous Composite Score, E2E Latency/Cost Pareto Frontier (`fig4`), Paired Bootstrap p-values, and full Cross-Layer Error & Negative Result Analysis (`ml/results/error_analysis.md`).
- **Status**: Verified `Full (A+B+C)` lifted Conjunctive E2E Task Success from `0.0% -> 41.67%` (`p=0.0`, Composite `0.4192 -> 0.7552`, `p=0.0`); dropping Layer B (`Full-B`) collapsed E2E to `0.0%` (`p=0.0`), dropping Layer C (`Full-C`) collapsed E2E to `8.33%` (`p=0.002`), while `Edge (A_tiny+B+C)` reached `37.5%` E2E (`p=0.0` vs baseline) at `$0.0499 / 1k` turns (`47.7%` lower cost than Full).

### Session 9 — Production Hardening, CI/CD, Model Cards & Live Soak Validation — `Done`
- **Files**: `.github/workflows/ci.yml`, `Dockerfile`, `.dockerignore`, `ml/cards/model_cards.md`, `orchestrator/factory.py`, `server.py`, `test_pipeline.py`.
- **Implementation Spec**:
  1. Unified ASR wiring via `PipelineFactory.build_asr_service` (Layer A biasing/hotwords/phonetic repair for `small.en`, capacity-gated anti-loop guards for `tiny.en`) across `orchestrator/factory.py` and `server.py`, plus `PipelineFactory.create_preset("full" | "edge")` and `server.py` default `pocket_tts` (`CAPSTONE_ASR_PROVIDER` / `CAPSTONE_TTS_PROVIDER`).
  2. Added lightweight offline-safe GitHub Actions CI (`.github/workflows/ci.yml`), slim Python 3.10 CPU container (`Dockerfile` + `.dockerignore`), and structured ML Model Cards (`ml/cards/model_cards.md`).
- **Status**: Verified `py_compile` + `58 passed in 59.23s` in `test_pipeline.py` (incl. `TestSession9ProductionHardening` preset wiring & FastAPI `TestClient` soak across `/api/health`, `/api/tts/engines`, `/api/tts/voices`, `/api/llm/status`, `/`, `/tts`, `/api/asr/transcribe-file`); live Uvicorn port 8000 booted cleanly (`chrome_devtools` browser check blocked only on user Chrome remote debugging `DevToolsActivePort` not open).

### Session 10 — Publication-Grade Capstone Report & IEEE Research Paper — `Next`
- **Files**: `docs/CAPSTONE_REPORT_IEEE.md`, `docs/RESEARCH_PAPER_IEEE.md`.
- **Implementation Spec**:
  1. Populate all 10 chapters + front/back matter using **only** directly verified results, plots, model/dataset cards, verified IEEE/ICML/NeurIPS/Interspeech citations, and student/faculty metadata (**Shivendra Pratap**, Section **K23DT**, Batch **2027**, Roll No **33**, Subject **Deep Learning**, Prof. **Anjar Hussain Lone**, plus `[MOCK]` placeholders only for unverified institutional headers/dates/survey placeholders).
  2. Produce the tight **8-page IEEE Research Paper** distilled directly from Phases 0–5 with exact tables, significance tests, and Pareto plot.
