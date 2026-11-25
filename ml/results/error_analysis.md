# Cross-Layer Error, Failure & Negative Result Analysis

This log documents directly observed failure modes, boundary breakdowns, and negative experimental runs across Sessions 2–8 (`ml/results/*.json`) so every architectural fix and remaining gap is backed by evidence.

---

## 1. Documented Negative Results (Iterated & Diagnosed)

| Experiment / ID | Hypothesis Tested | Observed Failure Metric | Root-Cause Diagnosis | Architectural Remedy Applied |
| :--- | :--- | :--- | :--- | :--- |
| **`tiny_unguarded_hotwords_v1`** (`asr_ablation.json`) | Injecting a 60-term `hotwords` + `initial_prompt` list directly into 39M `whisper-tiny.en` boosts proper-noun recall cleanly | Clean interview WER **regressed `0.1387 -> 0.3292`**, MINDS-14 WER `0.2480 -> 0.3395`, CPU RTF spiked **9.7x (`0.1033 -> 1.0014`)** | 39M tiny decoder lacks cross-attention capacity to separate long logit-biased hotword prefixes from audio, triggering end-of-utterance hotword echoing (`"SAPI5, SAPI5..."`) and YouTube loops (`"Bye. Bye. Bye..."`) | **Capacity-Gated Conditioning**: disabled dense `hotwords` below `small.en`, added `condition_on_previous_text=False`, `repetition_penalty=1.1`, `no_repeat_ngram_size=4`, `hallucination_silence_threshold=0.4` (`R_entity` reached `0.9130` at `RTF=0.266`) |
| **`slu_qwen15b_raw`** (`slu_ablation.json`) | Small 1.5B coder model (`qwen2.5-coder:1.5b`) can serve zero-shot SLU JSON outputs reliably via prompting alone | **0.0% raw schema compliance** across 3 seeds on `MINDS-14` & `Interview QA` (`F1 = 0.0`), falling back to regex defaults | Code-tuned small SLMs wrap JSON blocks inside markdown ````json ... ```` fences or prose preambles, failing strict `json.loads()` | Unified `extract_and_repair_json` + GBNF `json_mode` + 1-shot constrained retry raised schema validity to **`1.000`** (`p = 0.0`, MINDS-14 F1 `0.8294`) |
| **DSP `Piper + WSOLA` Cloning** (`tts_ablation.json`) | Autocorrelation F0 semitone WSOLA pitch-shifting + spectral tilt EQ over dual-gender Piper VITS preserves cloned speaker identity | `F0 MAE` dropped (`50.48 -> 25.24 Hz`), but held-out **`CAM++` identity dropped `0.5297 -> 0.4159`** (`WSOLA STOI = 0.2843`) | Spectral occlusion saliency (`fig2`) proves CAM++ relies peaks at **`1.8–3.0 kHz` F3 (`0.769` drop)** rather than F0 pitch (`0.338`); phase-vocoder WSOLA smears upper formants | Replaced DSP pitch-shifting with flow-matching zero-shot `Kyutai Pocket TTS INT8` (`CAM++ = 0.5651`, `+0.1493` over WSOLA at `p = 0.0`) |
| **OW-Aggregated `RandomProj-128D` Channel Leakage** (`speaker_encoder_eval.json`, `fig3`) | Checking clean LibriSpeech ROC-AUC alone suffices to validate a speaker encoder | `RandomProj-128D` achieved **`0.9707` AUC** on LibriVox read audiobooks while having cosine gap **`0.0070`** | Static Welch Mel PSD memorizes narrator room/microphone EQ signatures inside a tight `>0.99` cosine cone (`0.006` clean+EQ gap vs **`0.2781`** for `WeSpeaker CAM++`) | Evaluated via `separation_gap`, cross-EQ tilt invariance (`fig3`), and same-channel synthetic tests |

---

## 2. Layer-Specific Residual Failure Breakdown

### Layer A (Stage 2 ASR — `asr_small_improved`, Clean Interview WER = `0.0435`, `R_entity = 0.9565`)
Inspection of the top 5 residual error records in `ml/results/asr_ablation.json`:
1. **Phoneme Onset Substitution (`pn_21`)**: `"Pooja tuned Mel spectrogram filterbanks with SciPy."` -> `"Ujha-tuned MEL spectrogram filter banks with SCIPY."` (`WER = 0.5714`). Unaspirated bilabial stop onset (`P-`) weakened under EdgeTTS `en-IN` cadence missed both hotword biasing and unigram length-ratio threshold.
2. **Inverse Text Normalization / Formatting Equivalences (Non-semantic WER)**:
   - `qa_18` (`WER=0.5714`): `"Calculate 95 minus 28 percent error reduction."` -> `"Calculate 95-28% error reduction."`
   - `pn_23` (`WER=0.3333`): `"Python three point ten."` -> `"Python 3.10."`
   - `pn_12` (`WER=0.125`): `"twenty decibels"` -> `"20 decibels"`
   - `qa_38` (`WER=0.25`): `"Alright"` -> `"All right"`
   Four of the five worst `small_improved` WER scores are benign orthographic normalization differences rather than semantic recognition failures.
3. **Homophone Out-of-Scope Digraph Gap (`en-AU_61_9` in E2E)**: `"freeze all transactions"` -> `"phrase all transactions"` (`asr_wer = 0.2308`). Because American Soundex preserves the leading letter (`P620` vs `F620`), `phonetic_repair` did not rewrite `"phrase"` to `"freeze"`, causing Layer B (`qwen2.5:3b`) to misclassify `"freeze"` as `"latest_transactions"`.

### Layer B (Stage 3/4 SLU — `slu_qwen3b_repair` & `slu_llama3b_repair`)
1. **Out-of-Scope (OOS) Over-Commitment (`clinc_oos`, `OOS F1 = 0.0` on Qwen vs `0.4444` on Llama)**: Instructed closed-list SLMs (`qwen2.5:3b`, `qwen2.5-coder:1.5b`) exhibit strong instruction-following bias toward selecting *some* in-domain class even on true OOS prompts, whereas `llama3.2:3b` and `llama3.1:8b` abstain (`oos_f1 = 0.4444`). Production OOS handling should combine `qwen2.5:3b` with a confidence threshold (`<0.75 -> "oos"`) or binary OOS pre-filter.
2. **Open-Domain vs Closed-Schema Ambiguity (`pn_05`, `pn_08`, `pn_10` in E2E Matrix)**: When given declarative engineering resumes (`"Priya benchmarked CTranslate2 quantization..."`) alongside `project_query` and `answer` in a zero-shot prompt without few-shot definitions of `"answer"`, `qwen2.5:3b` reasonably labeled technical architecture summaries as `"project_query"` (and `"beam search"`/`"quantization"` numbers as `"calculation_query"`).

### Layer C (Stage 5 TTS — `Pocket TTS INT8` + `WeSpeaker CAM++`)
1. **Short-Utterance & Noisy-Reference Identity Variance (`librispeech_60`)**:
   - In Session 2 (`speaker_encoder_eval.json`), `Spk 260` clip `260-123440-0001` (`<2.5s` expressive theatrical shift) scored lowest CAM++ similarity (`0.4709`).
   - In Session 7 (`fig3`), `Spk 5639` Pocket TTS clone shifted away from its 5-clip island; in Session 8 (`e2e_ablation_matrix.json`), short truncated replies (`"Our architecture follows SOLID: 1."` on `qa_10`, `CAM++ = 0.4367`; `en-GB_388_5`, `0.2569`; `qa_21`, `0.3276`) fell below the `0.50` CAM++ verification gate despite `0.0–0.28` clean roundtrip Whisper WER. Extending minimum synthesis text duration (`>= 3.0s`) and enforcing clean `>= 5s` studio reference prompts stabilizes zero-shot flow-matching speaker transfer (as demonstrated by `0.7444` CAM++ on studio `bria.wav` in Session 3).

---

## 3. Baseline vs. Full Residual E2E Examples (`ml/results/e2e_ablation_matrix.json`)

| Turn ID | Condition | ASR Hypothesis (`WER` / `R_ent`) | SLU Prediction (`Schema Valid`) | TTS Output & Verification (`CAM++` / `RT_WER`) | E2E Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`pn_05`** | `Baseline` | `"Pre-a-bench marked C-Translate to quantization on Cuba needs nodes."` (`0.857` / `0.0`) | `__fallback__` (`False`) | `Piper WSOLA` (`0.434` / `0.455`) | Failed all 3 layers |
| **`pn_05`** | `Full (A+B+C)` | `"Priya benchmarked CTranslate2 quantization on Kubernetes nodes."` (`0.0` / `1.0`) | `"project_query"` vs `"answer"` (`True`) | `Pocket TTS` (`0.781` / `0.0`) | ASR & TTS perfect; failed closed-intent schema distinction (`project_query` vs `answer`) |
| **`en-AU_236_1`** | `Full (A+B+C)` | `"I was just wondering if you could help me today. I just wanted to update my address details..."` (`0.45`) | `"address"` (`True`) | `Pocket TTS` (`0.578` / `0.0`) | SLU & TTS passed; ASR `WER=0.45` flagged only because reference text had crowd-annotator speech disfluencies (`"how did my name is"`) |
| **`en-AU_61_9`** | `Full (A+B+C)` | `"Hi, I'm just calling because I'd like to phrase all transactions on my card."` (`0.231`) | `"latest_transactions"` vs `"freeze"` (`True`) | `Pocket TTS` (`0.629` / `0.0`) | Cascade error from `"freeze" -> "phrase"` (`ph/f` Soundex boundary) |
| **`qa_01`** | `Baseline -> Full` | `"Hi everyone, great to meet you for today's system check."` (`0.0`) | `__fallback__` (`False`) -> `"greeting"` (`True`) | `Piper` (`0.499`) -> `Pocket TTS` (`>=0.50`) | **Full recovery across SLU + TTS** (`Full: +41.7% E2E vs Baseline: 0.0%`, `p=0.0`) |
