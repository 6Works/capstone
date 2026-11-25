# Model Cards — Decoupled 6-Stage Voice AI Interview Assistant

This document provides structured model cards for every machine learning component evaluated and deployed in the pipeline (`orchestrator/factory.py`), grounded exclusively in directly observed telemetry from `ml/results/*.json`.

---

## 1. Layer A — Whisper `small.en` & `tiny.en` INT8 + Contextual Biasing & Phonetic Repair

- **Backbones**: `Systran/faster-whisper-small.en` (`244M` params, `463.58 MB` disk, `718.36 MB` Peak RSS) and `Systran/faster-whisper-tiny.en` (`39M` params, `74.47 MB` disk, `266.91 MB` Peak RSS) executed on CPU INT8 via `CTranslate2` (`faster-whisper`).
- **License**: `MIT` (OpenAI Whisper weights & Systran CTranslate2 conversion).
- **Layer A Modifications**:
  - `small.en` (`asr_small_improved`): Silero VAD + Role/Candidate dynamic `initial_prompt` + CTranslate2 trie `hotwords` + word-confidence gated Soundex/Rapidfuzz (`phonetic_repair`, bigram join & hyphenated sub-token repair).
  - `tiny.en` (`asr_tiny_improved`): **Capacity-Gated Conditioning** (disables dense `hotwords` to prevent 39M echo hallucination loops, enforces `condition_on_previous_text=False`, `repetition_penalty=1.1`, `no_repeat_ngram_size=4`, `hallucination_silence_threshold=0.4`) + dynamic `initial_prompt` + `phonetic_repair`.
- **Observed Benchmarks**:
  - `asr_small_improved`: Interview Clean WER **`0.0435`** (`-48.8%` vs baseline `0.0849`, `p=0.0`), Proper-Noun Clean Recall **`0.9565`** (`20 dB`: **`0.9783`**, `10 dB`: **`0.9130`**), LibriSpeech Clean WER **`0.0306`**, MINDS-14 Accent WER **`0.2239`**, CPU RTF `0.8857`, Cost `$0.0602 / 1k turns`.
  - `asr_tiny_improved`: Proper-Noun Clean Recall **`0.9130`** (beats `small_baseline` `0.7391` at `p=0.018` with `2.7x` less RAM and `2.3x` faster CPU `RTF=0.2660`), Cost `$0.0181 / 1k turns`.
- **Contamination, Bias & Risks**: Pretrained on public English web speech (overlap risk on LibriSpeech); exhibits higher baseline WER on `en-GB`/`en-AU` accents vs `en-US`; `en-IN` onset stops (`Pooja -> Ujha`) can still bypass Soundex when leading consonant drops. Intended solely for interview transcription assistance with human review.

---

## 2. Layer B — Quantized Local SLMs (`Qwen 2.5 3B`, `Llama 3.2 3B`, `Qwen 2.5 Coder 1.5B`) + Schema Repair

- **Backbones**: `qwen2.5:3b` (`Q4_K_M`, `1.80 GB` disk, `2.23 GB` VRAM), `llama3.2:3b` (`Q4_K_M`, `1.88 GB` disk, `2.56 GB` VRAM), `qwen2.5-coder:1.5b` (`Q4_K_M`, `0.92 GB` disk, `1.31 GB` VRAM), benchmarked against `llama3.1:latest` (`8B`, `4.58 GB` disk, `3.14 GB` VRAM + CPU offload).
- **Licenses**: `Qwen 2.5 (3B)` & `Qwen 2.5 Coder (1.5B)` under **`Apache-2.0`** (permissive commercial startup use); `Llama 3.2 (3B)` & `Llama 3.1 (8B)` under **Meta Llama Community License**. Consequently **`qwen2.5:3b` (`slu_qwen3b_repair`)** is both technically Pareto-optimal and license-safest for commercial packaging.
- **Layer B Modifications**: Shared `extract_and_repair_json` + `repair_and_parse_with_retry` in `services/llm/json_utils.py` (markdown fence stripping, balanced/truncated brace recovery, trailing comma & Python-literal normalization, max 1 constrained retry) + `preprocess_asr_transcript` lexical recovery hook across Stage 3 (`LLMNLPService`) and Stage 4 (`RoleService`).
- **Observed Benchmarks (3 Seeds `[42, 1337, 2026]`)**:
  - `slu_qwen3b_repair`: MINDS-14 Macro-F1 **`0.8667 ± 0.0000`** (`p=0.798` vs `8B`, `p=0.0` vs RuleBased), CLINC-OOS Macro-F1 **`0.7972 ± 0.0192`**, Interview SLU F1 **`0.7606 ± 0.0273`** (beats `8B` `0.6939`), Interview Rubric Spearman rho **`0.6897`** (beats `8B` `0.4188` and `llama3.2:3b` `0.3941`), `100.0%` schema compliance, latency `1258 ms`, Cost **`$0.0172 / 1k turns`** (9.9x cheaper than `8B` `$0.1698`). Cross-layer ASR pre-repair raised noisy `tiny_asr` MINDS-14 F1 `0.9238 -> 0.9619`.
- **Known Limitations**: Closed-list `qwen2.5:3b` prompts over-commit to in-domain labels on true Out-of-Scope inputs (`clinc_oos` OOS F1 = `0.0` vs `0.4444` on Llama); interview scoring outputs must act as co-pilot decision support, never unreviewed automated hiring decisions.

---

## 3. Layer C — Zero-Shot Neural Voice Cloning (`Kyutai Pocket TTS INT8`) & `Piper VITS` Fallback

- **Backbones**: `Kyutai Pocket TTS INT8` via `sherpa-onnx` (`lm_flow.int8.onnx`, `lm_main.int8.onnx`, `encoder.onnx`, `decoder.int8.onnx`, `text_conditioner.onnx`; `~189 MB` ONNX, `239.96 MB` Peak RSS, `24 kHz`), falling back seamlessly to offline `Piper VITS` (`en_US-lessac-low`, `en_US-ryan-low`, `MIT`) or Windows `SAPI5`.
- **License**: `CC-BY-4.0` weights + `Apache-2.0` (`sherpa-onnx`) runtime — permissive commercial readiness.
- **Observed Benchmarks (12 Unseen LibriSpeech Speakers x 3 Utterances)**:
  - Held-out `WeSpeaker CAM++` similarity: **`0.5651`** (`0.5896` vs enroll ref; studio WAV `bria.wav = 0.7444`), outperforming DSP `Piper dual+WSOLA` (`0.4159`, **`+0.1493`**, **`p=0.0`**) and `Pyttsx3` (`0.4776`, `p=0.004`), at CPU `RTF = 0.4384`.
  - Diagnosed why `Piper + WSOLA` failed: WSOLA phase vocoding yielded `STOI = 0.2843` waveform distortion and smeared F2/F3 formants.
- **Ethical Safeguards**: Requires explicit user-uploaded reference audio per cloned voice; custom profiles stored locally (`storage/cloned_voices/`, gitignored) with user rename/delete lifecycle APIs and protected non-impersonation presets.

---

## 4. Speaker Verification Encoder — `WeSpeaker CAM++ LM` (`512-D ONNX`)

- **Model & License**: `wespeaker_en_voxceleb_CAM++_LM.onnx` (`27.93 MB`, `512-D` L2-normalized embedding, `Apache-2.0` / `CC-BY-4.0`).
- **Observed Benchmarks**: Clean separation gap **`0.3016`** (`vs 0.0070` for 128-D Welch RandomProj, `p=0.0`), Clean+EQ-tilt separation gap **`0.2781`** (`vs 0.0060`), warm latency `109.2 ms`.
- **Explainability**: 8-band spectral occlusion (`fig2`) establishes F3/F2 vocal-tract formants (`1.8–3.0 kHz`: `0.769` drop; `1.0–1.8 kHz`: `0.651` drop) carry >2x higher speaker weight than `0–300 Hz` F0 pitch (`0.338`).
- **Misuse Guardrails**: Used purely as a research evaluation & synthesis consistency metric (`morph_profiles` / benchmark scoring), never as a biometric security authentication barrier.
