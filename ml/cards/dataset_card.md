# Dataset Card — Decoupled 6-Stage Voice AI Interview Assistant

This card documents all speech and text benchmark corpora curated in `ml/datasets/prepare_benchmarks.py` under `ml/datasets/manifests/`, plus real Indian-English evaluation via `ai4bharat/Svarah`. All local splits are generated deterministically (`seed=42`) with normalized SHA-256 deduplication (`text_sha256`) enforcing zero cross-split text leakage.

Total local cached benchmark corpus size (`storage/benchmarks/`): **491.32 MB** across seven manifests (`librispeech_60`, `librispeech_other_60`, `minds14_en_140`, `minds14_en_full`, `clinc_oos_200`, `clinc_oos_full`, `interview_stress_v1`), complemented on GPU evaluation by **1,000 held-out clips from `ai4bharat/Svarah`**.

---

## 1. Benchmark Inventory & Verified Licensing

| Manifest / Dataset | Source | Verified License | Modality | Total Items | Split Breakdown | Role & Key Audit Finding |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| `librispeech_60.json` | `openslr/librispeech_asr` (`clean/test`) | `CC-BY-4.0` | Speech FLAC | 60 (12 spk x 5) | `enroll: 12`, `val: 12`, `test: 36` | Clean ASR WER, Same/Diff Speaker Verification, TTS Cloning |
| `librispeech_other_60.json` | `openslr/librispeech_asr` (`other/test`) | `CC-BY-4.0` | Speech FLAC | 60 (12 spk x 5) | `enroll: 12`, `val: 12`, `test: 36` | Challenging real-acoustic ASR (`Parakeet=0.0605` vs `small.en=0.0958`) & OOD ASV |
| `minds14_en_140.json` | `PolyAI/minds14` (`en-US`, `en-GB`, `en-AU`) | `CC-BY-4.0` | Speech WAV | 140 (14 x 10) | `dev: 42`, `val: 28`, `test: 70` | Compact accented ASR & cross-layer SLU subset |
| `minds14_en_full.json` | `PolyAI/minds14` (full deduplicated English) | `CC-BY-4.0` | Speech WAV (16 kHz) | 1,662 (`US:541`, `GB:538`, `AU:583`) | `dev: 914`, `val: 250`, `test: 498` | Real human SLU training (`dev`/`val`) & test (`498`); *caveat: reference text retains noisy crowd/Google-ASR strings creating a ~`0.25` WER floor even on `whisper-large-v3-turbo` (`0.2517`)* |
| `clinc_oos_200.json` | `clinc/clinc_oos` (`small`) | `CC-BY-3.0` | Text | 200 (15 classes incl. `oos`) | `val: 100`, `test: 100` | Compact SLU regression manifest |
| `clinc_oos_full.json` | `clinc/clinc_oos` (`small` complete corpus) | `CC-BY-3.0` | Text (151 classes) | 16,196 | `train: 7,596`, `val: 3,100`, `test: 5,500` | Full 150-intent + Out-of-Scope (`oos`) neural LoRA training & `5,500`-item test |
| `interview_stress_v1.json` | External Azure `EdgeTTS` (`en-IN`, `en-US`) | **Research-Only Proxy** (consumer Read-Aloud endpoint; NOT `MIT`) | Speech WAV (16 kHz) | 65 (`25` PN + `40` QA) | `val: 15`, `test: 50` | Controlled South-Asian proper-noun & 50% unseen holdout stress + 10 ordinal rubric answers |
| **`ai4bharat/Svarah`** *(Gated Eval Only)* | HuggingFace `ai4bharat/Svarah` (`test` slice) | **`CC-BY-4.0`** | Human Speech (`en-IN`) | **1,000** sampled (`885` non-digit, `115` digit-rich) | `test-only: 1,000` (never trained on) | Clean human Indian-English ASR evaluation (`whisper-small.en` `0.1521` vs `whisper-large-v3-turbo` **`0.0888`**, `p < 0.001`) |

---

## 2. Anti-Leakage, Holdout & Acoustic Stress Protocols

1. **Deduplication & Split Isolation:** Every manifest enforces zero `text_sha256` intersection across `train`/`dev`/`val`/`test`. Speaker sets (`librispeech_60`, `librispeech_other_60`) partition speaker IDs cleanly across `enroll` (cloning conditioning only), `val` (calibration), and `test` (reporting).
2. **Closed-Roster vs. Zero-Shot Proper-Noun Holdout (`run_asr_ablation.py`):** On `interview_stress_v1` `proper_nouns`, 50% of unique target proper nouns (`25` normalized terms in `_HELD_OUT_LOWER`) are withheld completely from Layer A prompts, hotwords, and repair lexicons in a dedicated holdout pass (`closed_roster_recall` vs `unseen_zero_shot_recall`), alongside `non_entity_wer`, `repair_precision`, `overcorrection_rate`, and ITN-normalized WER (`compute_itn_wer`).
3. **Deterministic AWGN, Cross-Channel Telephony & Narrowband Detection:**
   - `add_awgn(audio, snr_db, seed)`: deterministic `20 dB` and `10 dB` AWGN stress.
   - Cross-channel ASV (`eval_speaker_encoder.py`): 4th-order Butterworth telephony bandpass (`300–3400 Hz`) + `18 dB` AWGN + `-6 dB` gain attenuation. `VoiceCloner` now flags narrowband inputs automatically via Welch 95% spectral rolloff (`<= 3600 Hz`).

---

## 3. Responsible Disclosure & Limitations

- Synthetic audio (`interview_stress_v1`) rendered via EdgeTTS is labeled **Research-Only Proxy**; human `en-IN` ASR claims rest cleanly on gated **`ai4bharat/Svarah` (`CC-BY-4.0`, `N=1,000`)**.
- `MINDS-14` transcripts remain valuable for noisy-ASR SLU transfer (`0.9441–0.9520` F1), whereas raw ASR WER comparisons must account for its crowd-ASR label noise (`~0.25` WER floor) alongside clean human `Svarah` (`0.0888`) and `LibriSpeech` (`0.022–0.060`).
- No private user microphone captures or PII are committed (`storage/cloned_voices/` and `storage/benchmarks/` are gitignored).
