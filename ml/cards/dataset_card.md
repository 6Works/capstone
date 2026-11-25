# Dataset Card — Voice AI Interview Assistant Benchmark Suite

This card documents the four compact, license-clean benchmark manifests curated in `ml/datasets/prepare_benchmarks.py` and stored under `ml/datasets/manifests/`. All splits are generated deterministically (`seed=42`) with normalized SHA-256 text deduplication (`text_sha256`) and zero cross-split text overlap.

## 1. Benchmark Inventory Summary

| Manifest | Source | License | Modality | Total Items | Split Breakdown | Total Audio | Key Metrics Evaluated |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| `librispeech_60.json` | `openslr/librispeech_asr` (`clean/test`) | `CC-BY-4.0` | Speech FLAC | 60 (12 speakers x 5 utts) | `enroll: 12`, `val: 12`, `test: 36` | 10.06 min (`1.74–27.18 s`) | ASR WER/CER, Speaker Cosine Sim, STOI |
| `minds14_en_140.json` | `PolyAI/minds14` (`en-US`, `en-GB`, `en-AU`) | `CC-BY-4.0` | Speech WAV | 140 (14 intents x 10 utts) | `dev: 42`, `val: 28`, `test: 70` | 20.11 min (`1.71–28.33 s`) | Accented ASR WER, Spoken Intent Macro-F1 |
| `clinc_oos_200.json` | `clinc/clinc_oos` (`small` val/test) | `CC-BY-3.0` | Text | 200 (15 classes incl. `oos`) | `val: 100`, `test: 100` | N/A | Intent Macro-F1, OOS Detection, JSON Validity |
| `interview_stress_v1.json` | External Azure `EdgeTTS` (`en-IN`, `en-US`) | `MIT` | Speech WAV (16 kHz mono) | 65 (`25` proper nouns + `40` QA turns) | `val: 15`, `test: 50` | 5.37 min (`3.12–11.09 s`) | Hotword/Noun Recall, Intent F1, Rubric Spearman rho |

Total cached audio footprint on disk (`storage/benchmarks/`): **37.70 MB** across 265 audio clips (`gitignored`; manifests stay version-controlled).

---

## 2. Detailed Dataset Composition

### 2.1 `librispeech_60` (Speaker & Clean Read-Speech Benchmark)
- **Source & Collection**: Drawn from `openslr/librispeech_asr` (`clean`, `test` split via HuggingFace Datasets Server), selecting 12 unseen public-domain LibriVox speakers (`1320`, `2094`, `260`, `2961`, `3575`, `5639`, `6930`, `7127`, `7176`, `7729`, `8230`, `8463`), 5 utterances per speaker.
- **Duration Distribution**: Minimum `1.74 s`, maximum `27.18 s`, mean `10.06 s`.
- **Anti-Leakage & Split Protocol**: Utterance `0` per speaker -> `enroll` (voice-cloning reference only), utterance `1` -> `val` (hyperparameter selection), utterances `2–4` -> `test` (final reporting). Verified `0` `text_hash` collisions across `enroll`, `val`, and `test`.

### 2.2 `minds14_en_140` (Accented Spoken Intent Benchmark)
- **Source & Collection**: Drawn from `PolyAI/minds14` (`train` split) across three native English variants: `en-GB` (`57`), `en-US` (`44`), and `en-AU` (`39`), covering all **14 banking intents** equally (`10` deduplicated items per class: `abroad`, `address`, `app_error`, `atm_limit`, `balance`, `business_loan`, `card_issues`, `cash_deposit`, `direct_debit`, `freeze`, `high_value_payment`, `joint_account`, `latest_transactions`, `pay_bill`).
- **Duration Distribution**: Minimum `1.71 s`, maximum `28.33 s`, mean `8.62 s`.
- **Anti-Leakage & Split Protocol**: Deduplicated globally across accents via `text_sha256`, shuffled with `seed=42`, and stratified `3 dev / 2 val / 5 test` per intent (`42 / 28 / 70`). Zero `text_hash` intersection across splits.

### 2.3 `clinc_oos_200` (Text SLU + Out-of-Scope Benchmark)
- **Source & Collection**: Sampled from disjoint upstream splits of `clinc/clinc_oos` (`small` config): `val: 100` exclusively from HF `validation` and `test: 100` exclusively from HF `test`. Covers 15 classes (`oos` plus 14 in-domain classes: `meaning_of_life`, `improve_credit_score`, `change_language`, `timer`, `translate`, `insurance_change`, `payday`, `find_phone`, `replacement_card_duration`, `pto_request`, `travel_alert`, `fun_fact`, `transfer` at 14 items each, and `definition` at 4 items).
- **Anti-Leakage & Split Protocol**: Cross-split `text_sha256` intersection enforced to `0` (`val & test = 0`).

### 2.4 `interview_stress_v1` (Domain Proper-Noun & Rubric Stress Suite)
- **Source & Circularity Avoidance**: 65 scripted interview utterances synthesized into 16 kHz mono WAV via external Azure **EdgeTTS** (`en-IN-PrabhatNeural`, `en-IN-NeerjaNeural`, `en-US-GuyNeural`, `en-US-JennyNeural`), completely disjoint from our evaluated neural cloning (`Pocket TTS`) and formant baselines so ASR/TTS benchmarks remain free of circular model bias.
- **Subsets**:
  - `proper_nouns` (`25` clips; `5 val`, `20 test`): Tests recognition of South-Asian names and domain technical terms (e.g., `Sachin`, `Aradhya`, `Nandu`, `Bablu`, `Siddharth`, `Krishnendu`, `FastAPI`, `CTranslate2`, `WeSpeaker`, `Kyutai`, `gRPC`).
  - `interview_qa` (`40` clips; `10 val`, `30 test`): Covers `7` SLU intents (`greeting: 5`, `identity_query: 5`, `project_query: 5`, `calculation_query: 5`, `time_query: 3`, `answer: 13`, `farewell: 4`) and `3` rubric quality tiers (`high: 10` `[7–10]`, `mid: 26` `[4–7]`, `low: 4` `[1–5]`) with ordinal `expected_score_range` labels for Spearman correlation checks.
- **Duration Distribution**: Minimum `3.12 s`, maximum `11.09 s`, mean `4.96 s`.

---

## 3. Acoustic Stress Protocol (Deterministic AWGN)

Rather than duplicating noisy WAV files on disk, `ml/datasets/prepare_benchmarks.py:add_awgn(audio, snr_db, seed=42)` injects calibrated zero-mean white Gaussian noise on the fly at target signal-to-noise ratios (`clean`, `20 dB`, `10 dB`) given signal variance $\sigma_s^2 + 10^{-12}$ and noise power $\sigma_s^2 / 10^{\mathrm{SNR}/10}$, clipping softly to `[-1.0, 1.0]`.

---

## 4. Ethics, Bias, Contamination & Licensing Caveats

1. **Pretraining Contamination Disclosure**: Whisper models were exposed to large public English web corpora and LibriSpeech-like read speech; therefore `librispeech_60` should only be interpreted alongside accented `minds14_en_140`, out-of-distribution `interview_stress_v1`, and `20 dB` / `10 dB` AWGN stress runs.
2. **Demographic & Channel Limitations**: `PolyAI/minds14` spans US/UK/Australian accents but lacks Indian English (`en-IN`), motivating the inclusion of Azure `en-IN-PrabhatNeural` / `en-IN-NeerjaNeural` in `interview_stress_v1`. LibriSpeech read audiobooks exhibit distinct recording-channel EQ compared to live headset microphones (established in Session 2 error analysis).
3. **Biometric Privacy & Permissive Commercial Licensing**: No private user voice recordings are stored in the repository; benchmark manifests use `CC-BY-4.0`, `CC-BY-3.0`, or `MIT` sources compatible with startup commercial deployment.
