# Research-First Multi-Session Implementation Roadmap

**Project**: 6Works — Low-Cost Edge-First Conversational Voice & AI Interviewer Pipeline  
**Mode**: Research-First Per-Layer Engineering (Precedes IEEE & Capstone Report Generation)  
**Target Environment**: CPU / 4 GB VRAM Edge Workstation (`Python 3.10`, Conda `capstone_voice`, Ollama local SLMs, ONNX Runtime)

---

## 1. Phase 0: Problem Lock, Pareto Framing & Target Metrics

- **Core Research Claim (Draft)**: Strategic upgrades to three high-headroom layers—*(A) Context-Conditioned Quantized ASR, (B) Phonetic-Repaired Small SLM Structuring, and (C) ONNX Neural Speaker Conditioning & Zero-Shot Synthesis*—match or outperform heavier 8B+ / cloud baselines on accuracy and speaker fidelity while operating strictly within local edge CPU/4 GB VRAM latency and zero marginal API cost.
- **Optimization Axes (Pareto Front)**:
  - **Stage 2 ASR**: WER ($\downarrow$), Proper-Noun Recall ($\uparrow$), Real-Time Factor RTF ($\downarrow$), Latency ms ($\downarrow$)
  - **Stage 3/4 SLU & Evaluation**: Macro-F1 ($\uparrow$), JSON Schema Compliance % ($\uparrow$), Rubric Correlation ($\uparrow$), Latency ms / Peak RAM ($\downarrow$)
  - **Stage 5 TTS & Cloning**: Neural Speaker Cosine Similarity ($\uparrow$), Roundtrip ASR WER ($\downarrow$), STOI Intelligibility ($\uparrow$), CPU RTF ($\downarrow$)
  - **End-to-End**: Turn-to-Turn Latency ($< 1500\text{ ms}$ CPU target), Estimated Cost / 1k Turns (USD)

---

## 2. Commercial License & Compute Audit

Per Section 7 of [Research-First Capstone Plan Per-Layer Improvement and Paper Workflow.md](file:///c:/Users/shive/projects/capstonedemo/docs/Research-First%20Capstone%20Plan%20Per-Layer%20Improvement%20and%20Paper%20Workflow.md), startup viability requires permissive open weights (MIT / Apache-2.0 / CC-BY-4.0) and excluding Non-Commercial (`CC-BY-NC`) assets:

| Component | Candidate Model | Runtime / Size | Code License | Weights License | Decision |
| --- | --- | --- | --- | --- | --- |
| **ASR** | Whisper `tiny.en` / `small.en` (CTranslate2 INT8) | CPU ONNX/CT2 (~39M / 244M) | MIT | MIT | **Approved** |
| **Speaker Encoder** | WeSpeaker CAM++ LM (`wespeaker_en_voxceleb_CAM++_LM.onnx`) | ONNX CPU (~27.9 MB) | Apache-2.0 | CC-BY-4.0 (VoxCeleb) | **Approved** |
| **Zero-Shot TTS** | Kyutai Pocket TTS INT8 (`sherpa-onnx-pocket-tts-int8`) | ONNX CPU (~93.8 MB) | MIT | CC-BY-4.0 (user ref mode only) | **Approved** *(avoid bundled non-commercial voice caches)* |
| **Zero-Shot TTS Alt** | k2-fsa ZipVoice INT8 (`zipvoice-distill-int8`) | ONNX CPU (~104.1 MB) | Apache-2.0 | **CC-BY-NC** | **Excluded** *(non-commercial weights)* |
| **Multi-Speaker TTS** | Kokoro-82M (`kokoro-en-v0_19`) / Piper `lessac`/`ryan` | ONNX CPU (~305 MB / ~63 MB) | Apache-2.0 / MIT | Apache-2.0 / MIT | **Approved** |
| **SLU / Dialogue LLM** | `qwen2.5:3b`, `qwen2.5-coder:1.5b`, `llama3.2:3b`, `llama3.1:8b` | Ollama Q4_K_M | Apache-2.0 / Llama 3.2 | Permissive Community | **Approved** |

---

## 3. Focused Per-Layer Hypotheses (3 High-Headroom Layers)

| Layer | Existing Baseline | Proposed Change | Hypothesis (Why It Helps) | Evaluation Protocol |
| --- | --- | --- | --- | --- |
| **Layer A: Stage 2 ASR** | `Whisper tiny.en` / `small.en` static prompt | Dynamic Blueprint/Resume Lexical Prior + Adaptive Silero VAD + Post-ASR Phonetic Repair | Autoregressive decoders hallucinate out-of-vocabulary proper nouns (`Sachin`, `Aradhya`) and technical acronyms; contextual `initial_prompt` + phonetic matching boosts entity recall without larger weights | Clean & AWGN noisy (20 dB, 10 dB SNR) subsets; WER, Name Recall, RTF |
| **Layer B: Stage 3/4 SLU** | Single-shot small SLM (`1.5B`/`3B`) & Regex Rule-Based | ASR-Noise Phonetic Repair + Constrained Schema-Repair + 2-Shot Rubric Anchors | Small Q4_K_M models frequently drop JSON schema keys under ASR typos; constrained retry + lexical repair closes accuracy gap to `llama3.1:8b` at `<40%` latency | Clean + noisy ASR inputs; Macro-F1, JSON Valid %, Latency, Seed Mean ± Std |
| **Layer C: Stage 5 TTS / Cloning** | `RandomProj-128D` + `Piper VITS` + WSOLA pitch shift | `WeSpeaker CAM++` 256-D Neural Encoder + `Pocket TTS INT8` Zero-Shot Conditioning | Spectral summary statistics + time-scale pitch shifting cannot model vocal tract formants or timbre; neural speaker conditioning preserves target voice identity & prosody | Speaker Cosine Similarity, Roundtrip WER, STOI, CPU RTF |

---

## 4. 10-Session Decoupled Implementation Plan

| Session | Focus & Deliverables | Key Files / Modules | Status |
| --- | --- | --- | --- |
| **Session 1** | Install eval stack & `sherpa-onnx`; create minimal `ml/` scaffold (`metrics`, `configs`), roadmap & tests | `docs/IMPLEMENTATION_ROADMAP.md`, `ml/metrics/eval_metrics.py`, `ml/configs/baseline_pipeline.json` | **Done (Verified 45/45)** |
| **Session 2** | Real Neural Speaker Encoder (`WeSpeaker CAM++` ONNX, 28 MB) in `VoiceCloner` replacing random projection + speaker verification eval | `services/tts/voice_cloner.py`, `ml/experiments/eval_speaker_encoder.py` | **Next** |
| **Session 3** | True CPU Zero-Shot Voice Cloning (`Pocket TTS INT8` ONNX, 94 MB) integrated behind `ITTSService` & engine switcher | `services/tts/tts_services.py`, `server.py`, `web/pages/tts.html` | Pending |
| **Session 4** | Curate clean public + domain-stress benchmark manifests (LibriSpeech/FLEURS `en_in`, SLU intents, AWGN noise) & Dataset Card | `ml/datasets/`, `ml/cards/dataset_card.md` | Pending |
| **Session 5** | Layer A (Stage 2 ASR) baseline reproduction + dynamic context lexical biasing + noise robustness experiments | `services/asr/asr_services.py`, `ml/configs/asr_*.json`, `ml/experiments/run_asr_ablation.py` | Pending |
| **Session 6** | Layer B (Stage 3/4 SLU & Rubric) 1.5B/3B vs. 8B structured prompting + phonetic repair + cost Pareto | `services/nlp/nlp_services.py`, `ml/configs/slu_*.json`, `ml/experiments/run_slu_ablation.py` | Pending |
| **Session 7** | Layer C (Stage 5 TTS) full objective evaluation (`CosSim`, Roundtrip WER, `STOI`, `RTF`) & explainability visuals | `ml/experiments/run_tts_ablation.py`, `ml/experiments/explainability.py` | Pending |
| **Session 8** | End-to-End 6-Stage multi-seed Ablation Matrix (`Baseline`, `+A`, `+B`, `+C`, `Full`, Leave-One-Out) + Paired Bootstrap p-values + Error Analysis | `ml/experiments/run_e2e_ablation.py`, `ml/results/` | Pending |
| **Session 9** | Production hardening: wire winning defaults, GitHub Actions CI, minimal Dockerfile, final `Model Cards` & live E2E soak check | `.github/workflows/ci.yml`, `ml/cards/model_cards.md`, `orchestrator/factory.py` | Pending |
| **Session 10** | Generate final IEEE-style Comprehensive Capstone Report & 8-page Research Paper from verified artifacts (`Shivendra Pratap`, `K23DT`, Batch `2027`, Roll `33`, `Deep Learning`, Prof. `Anjar Hussain Lone`) | `docs/CAPSTONE_REPORT_IEEE.md`, `docs/RESEARCH_PAPER_IEEE.md` | Pending |
