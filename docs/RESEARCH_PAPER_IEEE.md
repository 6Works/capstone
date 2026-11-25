# Decoupled Pareto-Optimal Conversational Voice AI: Cross-Layer Capacity-Gated ASR, Constrained SLM Schema Recovery, and Formant-Preserving Flow-Matching Synthesis on CPU-Edge Hardware

**Shivendra Pratap**¹\*, **Prof. Anjar Hussain Lone**¹ *(Faculty Mentor)*  
¹Department of Computer Science and Engineering, `[MOCK — University / Institute Name]`, Section K23DT, Batch 2027, Roll No. 33  
*\*Correspondence / Codebase: `6Works` Capstone Repository (`docs/RESEARCH_PAPER_IEEE.md`)*

---

## Abstract

Real-time conversational voice assistants built by chaining standalone speech recognition, compact language modeling, and voice cloning modules suffer catastrophic compounding failures under consumer CPU/low-VRAM budgets (`0.0%` baseline end-to-end task completion). Unconstrained compact ASR decoders miss domain jargon and South-Asian names (`Proper-Noun Recall = 0.4783`), sub-3B quantized small language models (SLMs) emit malformed markdown-wrapped JSON (`0.0%` raw schema validity on `Qwen 2.5 Coder 1.5B`), and classical waveform-similarity overlap-add (`WSOLA`) pitch shifting destroys upper vocal-tract formants vital for speaker identity (`STOI = 0.2843`). We present a SOLID-decoupled six-stage voice pipeline evaluated with strict `text_sha256` zero-leakage benchmarks (`librispeech_60`, `minds14_en_140`, `clinc_oos_200`, `interview_stress_v1`). Layer A introduces capacity-gated contextual prompting, trie `hotwords`, repetition-loop guards, and confidence-gated Soundex/Rapidfuzz repair, reducing `whisper-small.en` interview WER by **48.8%** (`0.0849 -> 0.0435`, $p=0.0$) while lifting `39M tiny.en` entity recall from `0.4783` to **`0.9130`**. Layer B unifies deterministic JSON repair, 1-shot schema retry, and cross-layer ASR lexical pre-repair on `Qwen 2.5 3B Q4_K_M` (`Apache-2.0`), achieving **`100.0%`** schema validity, `MINDS-14` Macro-F1 **`0.8667`** (parity with `Llama 3.1 8B`, $p=0.798$), and **`0.9619`** noisy-ASR transfer F1 at **9.9x lower cost** (`$0.0172` vs. `$0.1698 / 1k turns`). Layer C deploys zero-shot flow-matching **`Kyutai Pocket TTS INT8`** (`CC-BY-4.0`), raising unseen-speaker `WeSpeaker CAM++` cosine similarity from `0.4159` (`WSOLA`) to **`0.5651`** ($p=0.0$; **`0.7444`** studio). Across 3 random seeds in a 9-row ablation matrix, single-layer upgrades stall at `0%–8.33%` E2E pass rate, whereas synchronized optimization reaches **`41.67%` strict / `0.7552` Composite Quality (`Full`, $p=0.0$)** and **`37.50%` / `0.6864` (`Edge`)** at **`$0.0499 / 1k turns`** and **`2.67 GB`** peak memory.

***Index Terms*—** Automatic Speech Recognition, Spoken Language Understanding, Zero-Shot Voice Cloning, Flow Matching, Pareto Optimization, Speaker Verification, Small Language Models.

---

## I. Introduction

Deploying privacy-preserving, offline-capable conversational voice agents on consumer hardware (`4-core CPU`, `16 GB RAM`, `4 GB VRAM`) requires orchestrating Automatic Speech Recognition (ASR), Spoken Language Understanding (SLU), dialogue state tracking, and personalized Text-to-Speech (TTS) under tight compute, latency, and permissive open-source license budgets (`MIT`, `Apache-2.0`, `CC-BY-4.0`). While individual foundation models perform strongly on isolated clean benchmarks [1], [13], [19], assembling unmodified compact components yields `0.0%` end-to-end strict task completion:

1. **Acoustic Entity Blindness & Decoder Capacity Traps:** Unprompted `Whisper` INT8 decoders fail nearly half of unfamiliar South-Asian personal names and technical stack entities (`0.4783` recall on `tiny.en`). Naively injecting dense `hotwords` into `39M`-parameter decoders triggers severe echoing hallucinations, multiplying CPU Real-Time Factor (RTF) by `9.7x` (`0.103 -> 1.001`) and doubling WER (`0.1387 -> 0.3292`).
2. **Quantized SLM Structural Fragility:** Small `Q4_K_M` SLMs frequently wrap structured responses in markdown fences or broken braces (`0.0%` raw schema validity on `Qwen 2.5 Coder 1.5B`), collapsing downstream state machines even when upstream transcripts are exact.
3. **Formant Destruction & Verification Channel Leakage:** Classical DSP F0 pitch-shifting (`WSOLA`) halves fundamental frequency MAE (`50.48 -> 25.24 Hz`) yet degrades unseen speaker similarity below unconditioned `Piper VITS` (`0.4159` vs `0.5297`), while naive Welch Mel PSD verification (`RandomProj-128D`) masks severe room-channel memorization behind a misleading `0.9707` ROC-AUC (`0.0070` cosine separation gap).

This paper resolves all three cross-layer failures within a commercially cleared local architecture, contributing: (i) **capacity-gated ASR conditioning + confidence-gated phonetic repair**; (ii) **deterministic schema repair + ASR lexical pre-repair** on `Apache-2.0` `Qwen 2.5 3B` matching `8B` SLU F1 at `10%` cost; (iii) **spectral saliency explainability** proving why flow-matching `Kyutai Pocket TTS INT8` preserves `1.8–3.0 kHz` speaker identity (`+0.1493` `CAM++`, $p=0.0$); and (iv) a **multi-seed 9-row Pareto evaluation** with paired bootstrap tests ($B=1000$) establishing an optimal `Full` and `Edge` accuracy-cost frontier.

---

## II. Related Work

### A. Contextual Biasing and Generative ASR Error Correction
While weakly supervised Whisper [1] surpassed traditional Conformer [2] and contrastive `wav2vec 2.0` [3] models on zero-shot LibriSpeech [4] evaluation, out-of-vocabulary proper nouns still necessitate biasing. `TCPGen` [5], `PromptASR` [6], and `CB-Whisper` [7] modify model weights or train auxiliary KWS/pointer heads. Post-hoc generative correctors like `N-best T5` [8] and `HyPoradise` [9] repair transcripts via seq2seq/7B LLMs at extra forward latency. In contrast, our Layer A operates zero-training over CPU `INT8` CTranslate2 decoders and documents the sub-`50M` capacity threshold governing prompt vs. trie-hotword stability.

### B. Spoken Language Understanding and Constrained SLMs
Spoken banking benchmarks (`MINDS-14` [11]) and out-of-scope benchmarks (`CLINC-OOS` [10]) reveal sharp degradation under noisy accents and malformed ASR text. `Everson et al.` [12] injected Word Confusion Networks into LLM prompts, while `Outlines` [15] enforces FSM masks at logits. Lightweight local Ollama REST pipelines using `Qwen2.5` [13] and `Llama 3` [14] lack direct logit hooks without custom servers; we pair stateless JSON-mode prompting with deterministic structural repair, 1-shot diagnostic retry, and regex-guided cross-layer ASR normalization.

### C. Speaker Embeddings and Zero-Shot Neural TTS
Speaker verification progressed from `x-vectors` [16] and `GE2E` [17] to channel-attentive `ECAPA-TDNN` [18] and fast multi-granularity `CAM++` [19], while perceptual intelligibility relies on `STOI` [20]. Traditional `WSOLA` [21] time-scale pitch shifting modifies F0 deterministically, whereas modern zero-shot cloning pairs speaker transfer (`SV2TTS` [22], `VITS` [23], `YourTTS` [24]) or codec/flow-matching synthesis (`SPEAR-TTS` [25], `CosyVoice` [26], `Matcha-TTS` [27], `Kyutai Pocket TTS` [28]). Unlike GPU-heavy or `CC-BY-NC` models, we audit `1.8–3.0 kHz` F3 preservation using `CAM++` and demonstrate real-time zero-shot CPU synthesis under `CC-BY-4.0`.

---

## III. Methodology

### A. Problem Formulation & End-to-End Pareto Objective
We structure the voice agent as six strictly decoupled microservices (`orchestrator/pipeline.py`): Audio Input ($f_1$) $\to$ ASR ($f_2$) $\to$ SLU ($f_3$) $\to$ Dialogue Manager ($f_4$) $\to$ TTS ($f_5$) $\to$ Output ($f_6$). Let a turn succeed strictly ($S_i(\theta) = 1$) iff:

$$\text{WER}_i \le 0.25 \;\wedge\; R_{\text{ent},i} \ge 0.80 \;\wedge\; v_{\text{schema},i}=1 \;\wedge\; \text{IntentMatch}_i \;\wedge\; \text{RT-WER}_i \le 0.35 \;\wedge\; \text{sim}_{\text{CAM++},i} \ge 0.50$$

Composite Quality $Q(\theta) \in [0,1]$ aggregates normalized performance across ASR, entity recall, SLU accuracy, speaker similarity, and roundtrip intelligibility:

$$Q(\theta) = 0.25(1 - \text{WER}_{\text{cap}}) + 0.20\,R_{\text{ent}} + 0.30\,\text{IntentAcc} + 0.15\,\text{sim}_{\text{CAM++}} + 0.10(1 - \text{RT-WER}_{\text{cap}})$$

We seek configurations $\theta^* = (\theta_A, \theta_B, \theta_C)$ on the non-dominated Pareto frontier of $(Q(\theta), C(\theta))$ under consumer resource caps ($\le 3.5\text{ GB}$ RAM/VRAM, permissive commercial licenses).

### B. Layer A: Capacity-Gated ASR & Phonetic Repair (`services/asr/asr_services.py`)
Incoming speech enters Silero VAD followed by `faster-whisper` (`INT8`, `beam_size=5`). To cure entity loss without destabilizing small decoders:
1. **Capacity Gate:** On `small.en` (`244M`), both dynamic role/roster `initial_prompt` and trie `hotwords` activate. On `tiny.en` (`39M`), dense `hotwords` are gated off and replaced by loop guards (`condition_on_previous_text=False`, `repetition_penalty=1.1`, `no_repeat_ngram_size=4`, `hallucination_silence_threshold=0.4`).
2. **Word-Confidence Gated Phonetic Repair:** Words with $p_w < 0.85$ (or contiguous sub-token compounds such as `"C-Translate to"` or `"Fast API"`) undergo Soundex + Rapidfuzz weighted matching (`0.55 Soundex + 0.45 Ratio`) against active candidate/tech lexicon $\mathcal{V}_c$, plus bigram `exact_join` concatenation.

### C. Layer B: Schema-Resilient SLU & ASR Pre-Repair (`services/llm/json_utils.py`, `services/nlp/nlp_services.py`)
1. **ASR Pre-Repair:** Before SLM tokenization, `preprocess_asr_transcript` repairs systematic accent substitutions (e.g., banking/code homophones) via zero-latency regex boundaries.
2. **Shared Repair & 1-Shot Retry:** `extract_and_repair_json` strips markdown fences, extracts balanced braces, normalizes trailing commas/Python literals, and repairs unclosed braces. Upon parse failure, `repair_and_parse_with_retry` triggers at most one `temperature=0.0` constrained retry passing exact `JSONDecodeError` offsets before deterministic rule fallback. We adopt **`Qwen 2.5 3B Q4_K_M`** (`Apache-2.0`, `temperature=0.1`).

### D. Layer C: Flow-Matching Cloning & Spectral Biometrics (`services/tts/tts_services.py`, `voice_cloner.py`)
Traditional F0-conditioned `Piper + WSOLA` alters fundamental period via overlap-add autocorrelation windows and spectral-tilt EQ. Instead, **`Kyutai Pocket TTS INT8`** (`sherpa-onnx`, `24 kHz`, 5 Lagrangian self-distillation flow-matching steps) encodes reference speech through the **Mimi** neural codec and generates continuous speaker-consistent acoustic latents natively on CPU. Speaker preservation is audited via `WeSpeaker CAM++ LM` (`512-D ONNX`), avoiding channel leakage found in static Welch PSD encoders.

---

## IV. Experimental Setup

1. **Zero-Leakage Benchmarks (`ml/cards/dataset_card.md`):** Four manifests (`37.70 MB` cached audio, verified `0` cross-split `text_sha256` collisions at `seed=42`): (a) **`librispeech_60`** (`CC-BY-4.0`, 12 unseen speakers × 5 utts: `12 enroll / 12 val / 36 test`, `1.74–27.18 s`); (b) **`minds14_en_140`** (`CC-BY-4.0`, 14 intents across `en-GB:57 / en-US:44 / en-AU:39`, `42/28/70` split); (c) **`clinc_oos_200`** (`CC-BY-3.0`, 15 classes incl. `oos`, `100 val / 100 test`); and (d) **`interview_stress_v1`** (`MIT`, 65 utterances synthesized by *external* Azure `EdgeTTS` `en-IN` and `en-US` voices avoiding circularity: 25 proper-noun utterances + 40 multi-intent/3-tier ordinal rubric QA turns). Stress evaluation injects deterministic AWGN (`clean`, `20 dB`, `10 dB`).
2. **Hardware & Protocol:** All evaluations execute locally on an `Intel Core i5-9300H @ 2.40 GHz` (4C/8T), `16 GB RAM`, `NVIDIA GTX 1650 (4 GB VRAM)` running Python `3.10.22`, `faster-whisper 1.2.1`, `ctranslate2 4.9.1`, `onnxruntime 1.23.2`, and `sherpa-onnx 2.0.3`. SLU and E2E tests run across `3` seeds (`[42, 1337, 2026]`, `mean ± std`). Significance uses paired sample-level bootstrap ($B = 1000$, `seed=42`).

---

## V. Experimental Results

### A. Speaker Biometric Leakage Audit (`fig3`)
On `librispeech_60` (`12` same vs. `54` diff pairs), `RandomProj-128D` achieved `0.9707` ROC-AUC while compressing speakers into a `>0.99` cosine cone (`0.9984 ± 0.0011` same vs `0.9914 ± 0.0043` diff; **`gap = 0.0070`**, dropping to `0.0060` under EQ-tilt over 120 clips). `WeSpeaker CAM++ LM` separated voices clearly (`0.8062` vs `0.5045`, **`gap = 0.3016`**, bootstrap `+0.3634`, **$p=0.0$**) and remained invariant under EQ tilt (**`gap = 0.2781`**, `0.8308` vs `0.5527`) and `20 dB` AWGN (`0.1548`, $p=0.0$).

### B. Layer A: ASR Accuracy, Noise & Negative Findings
| Config (`INT8 CPU`) | Disk / Peak RSS | Interview Clean WER (PN) | Interview `R_ent` (Clean / 20dB / 10dB) | MINDS-14 Accent WER | LibriSpeech (`Clean / 10 dB`) | CPU RTF | Cost / 1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tiny_baseline` | `74.5 / 266.3 MB` | `0.1387` (`0.3636`) | `0.4783 / 0.5000 / 0.3913` | `0.2480` | `0.0548 / 0.1464` | **`0.1033`** | `$0.0070` |
| `tiny_unguarded_hotwords` *(NEG)* | `74.5 MB` | `0.3292` (`0.6039`) | `0.8696 / — / —` | `0.3395` | — | `1.0014` | — |
| **`tiny_improved` (Edge)** | `74.5 / 266.9 MB` | `0.1573` (`0.2208`) | **`0.9130 / 0.8913 / 0.7391`** | `0.2640` | `0.0643 / 0.1402` | `0.2660` | `$0.0181` |
| `small_baseline` | `463.6 / 718.4 MB` | `0.0849` (`0.1623`) | `0.7391 / 0.6739 / 0.6304` | `0.2239` | `0.0348 / 0.0530` | `0.6083` | `$0.0413` |
| **`small_improved` (Full)** | `463.6 / 718.4 MB` | **`0.0435` (`0.0649`)** | **`0.9565 / 0.9783 / 0.9130`** | **`0.2239`** | **`0.0306 / 0.0436`** | `0.8857` | `$0.0602` |

Paired bootstrap confirms `small_improved` lowers interview WER by `+0.0475` (**$p=0.0$**) and lifts `R_ent` by `+0.2083` (**$p=0.006$**) with zero `MINDS-14` regression ($p=0.814$), while `tiny_improved` (`39M`) significantly surpasses `small_baseline` (`244M`) on `R_ent` (`+0.1583`, **$p=0.018$**) at `2.7x` less memory.

### C. Layer B: SLM Schema Repair & Noisy-ASR Transfer (3 Seeds)
| Config | Disk / VRAM | `MINDS-14` F1 (`SchVal`) | `CLINC` F1 (`OOS`) | Interview F1 | Rubric $\rho$ | `M14` (`tiny_asr` $\to$ `+PreRep`) | Cost / 1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rule_based` | `0 / 0 GB` | `0.0000` (`1.0`) | `0.0000` (`0.0`) | `0.6905` | `0.5437` | — | `$0.0000` |
| `qwen15b_raw` *(NEG)* | `0.92 / 1.31 GB` | `0.0000` (`0.0`) | `0.0139` (`0.0`) | `0.0000` | `0.7511` | — | `$0.0067` |
| `qwen15b_repair` | `0.92 / 1.31 GB` | `0.8294` (`1.0`, $p=0.0$) | `0.6321` (`0.0`) | `0.6705` | `0.6568` | `0.6571 -> 0.6571` | `$0.0181` |
| `llama3b_repair` | `1.88 / 2.56 GB` | **`0.9262`** (`1.0`) | `0.7350` (`0.4444`) | `0.3371` | `0.3941` | `0.8619 -> 0.8619` | `$0.0153` |
| **`qwen3b_repair` (`Apache`)** | **`1.80 / 2.23 GB`** | **`0.8667`** (`1.0`, $p=0.798$) | **`0.7972`** (`0.0`) | **`0.7606`** | **`0.6897`** | `0.9238 -> `**`0.9619`** | **`$0.0172`** |
| `llama8b_ref` *(Upper)* | `4.58 / 3.14 GB` | `0.8619` (`1.0`) | **`0.8563`** (`0.4444`) | `0.6939` | `0.4188` | `0.8857 -> 0.8857` | `$0.1698` |

`Qwen 2.5 3B` attains `8B` statistical parity on `MINDS-14` ($p=0.798$) and `CLINC` ($p=0.256$), leads both `Interview SLU` (`0.7606`) and `Rubric` ($\rho=0.6897$, vs `0.4188` on `8B`), improves by `+3.81%` F1 under ASR pre-repair (`0.9619`), and cuts cost by **9.9x**.

### D. Layer C: Flow-Matching Cloning vs. WSOLA Pitch Shifting (`36` Pairs)
`Pocket TTS INT8` reached **`0.5651 ± 0.1426`** held-out `CAM++` (`0.5896` vs enroll; **`0.7444`** on studio prompt `bria.wav`) with `0.0930` roundtrip WER, `0.4384` CPU RTF, and `240.0 MB` Peak RSS—significantly surpassing `Piper+WSOLA` (`0.4159`, **`+0.1493`**, $p=0.0$) and `Pyttsx3` (`0.4776`, `+0.0876`, $p=0.004$) while matching non-cloning cloud `EdgeTTS` identity (`0.5853`, $p=0.412$).

### E. End-to-End 9-Row Ablation & Pareto Front (`3` Seeds × `24` Turns = `72` Evals/Row)

| Row (`A / B / C`) | E2E Strict Pass | Composite $Q$ | Stage A Pass | Stage B Pass | Stage C Pass | Lat (ms) | Mem (MB) | Cost / 1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline` (`tiny / 1.5Braw / WSOLA`) | `0.00%` | `0.4192 ± 0.0005` | `66.67%` | `0.00%` | `25.00%` | `1495.4` | `1727.5` | `$0.0208` |
| `+A` (`small_imp` only) | `0.00%` | `0.4551` | **`95.83%`** | `0.00%` | `25.00%` | `5165.9` | `2179.5` | `$0.0717` |
| `+B` (`qwen3b_rep` only) | `8.33%` | `0.6771 ± 0.0002` | `66.67%` | `75.00%` | `16.67%` | `2308.5` | `2669.6` | `$0.0321` |
| `+C` (`Pocket` only) | `0.00%` | `0.4610 ± 0.0011` | `66.67%` | `0.00%` | **`58.33%`** | `2819.9` | `1727.7` | `$0.0392` |
| **`Full (A+B+C)`** | **`41.67%`** *($p=0.0$)* | **`0.7552 ± 0.0096`** | **`95.83%`** | **`76.39%`** | `56.94%` | `6868.4` | `3121.8` | `$0.0954` |
| `Full - A` | `41.67%` | `0.7221 ± 0.0001` | `66.67%` | `75.00%` | `58.33%` | `3230.9` | `2669.8` | `$0.0449` |
| `Full - B` / `Full - C` | `0.00%` / `8.33%` | `0.4949` / `0.7170` | `95.83%` | `0.00%` / `76.39%` | `58.33%` / `11.11%` | `6526` / `5909` | `2180` / `3122` | `$0.091` / `$0.082` |
| **`Edge (A_tiny+B+C)`** | **`37.50%`** *($p=0.656$ vs Full)* | **`0.6864`** | `75.00%` | `66.67%` | **`58.33%`** | **`3591.7`** | **`2670.4`** | **`$0.0499`** |

As plotted in Figure 1, single-layer improvements (`+A`, `+C`) remain trapped near zero strict E2E utility, whereas `Full` and `Edge` form clean Pareto optima (`Edge` delivers statistical E2E parity at `1.91x` lower latency and `1.91x` lower cost).

![End-to-End Pareto Front](../ml/results/figures/fig4_e2e_pareto_front.png)  
*Fig. 1. End-to-End Composite Quality vs. Operational Cost Pareto Front (`fig4_e2e_pareto_front.png`).*

---

## VI. Explainability & Residual Error Analysis

### A. Spectral Saliency & Token Confidence Visual Diagnostics
To diagnose *why* `Piper + WSOLA` failed despite cutting F0 MAE in half, 8-band spectral occlusion was evaluated over 12 LibriSpeech speakers (`fig2_campp_spectral_saliency.png`, Fig. 2). `WeSpeaker CAM++` exhibits a pronounced peak in upper vocal-tract formants (**`1.8–3.0 kHz` F3 = `0.7687` cosine drop**; `1.0–1.8 kHz` F2 = `0.6515`)—more than **2x** the weight of `0–300 Hz` F0 pitch (`0.3382`). Because WSOLA time-domain pitch shifting distorted envelope continuity (`STOI = 0.2843`), it destroyed identity. Conversely, `Pocket TTS` natively synthesizes coherent F2/F3 timbre. Simultaneously, `fig1_asr_word_confidence_heatmap.png` verifies word-level probabilities drop below `0.85` precisely on corrupted OOV entities (`"CTranslate2"`, `"Kubernetes"`, `"Karthik"`), enabling selective phonetic repair without side effects.

![CAM++ Spectral Saliency](../ml/results/figures/fig2_campp_spectral_saliency.png)  
*Fig. 2. 8-Band Spectral Occlusion Identity Saliency (`fig2_campp_spectral_saliency.png`).*

### B. Taxonomy of Residual Failures
Inspection of `ml/results/error_analysis.md` isolates five concrete remaining gaps:
1. **Unaspirated Onset Drops & Soundex Leading-Letter Lock:** `pn_21` (`"Pooja tuned"` $\to$ `"Ujha-tuned"`, `WER=0.571`) and `en-AU_61_9` (`"freeze all transactions"` $\to$ `"phrase all transactions"`, `WER=0.231` cascading to `"latest_transactions"`) bypass American Soundex because leading characters differ (`F` vs `P`), motivating Double Metaphone matching.
2. **Non-Semantic ITN Differences:** Four of the five highest `small_improved` WER utterances (`qa_18`, `pn_23`, `pn_12`, `qa_38`) reflect valid inverse text normalization (`"95-28%"` vs `"95 minus 28 percent"`).
3. **Closed-Schema Ambiguity & OOS Calibration:** `Qwen 2.5 3B` scored perfect `0.0` WER / `0.781` `CAM++` on `pn_05` yet predicted `"project_query"` instead of generic `"answer"`, and scored `0.0` `OOS F1` on `CLINC-OOS` due to closed-list over-commitment (whereas `Llama` achieved `0.4444`).
4. **Short-Utterance Flow-Matching Variance:** Truncated replies `<2.5 s` (`qa_10`, `en-GB_388_5`) slipped below `0.50 CAM++` (`0.257–0.437`), vs `0.744` on studio references, motivating minimum text-duration buffering.

---

## VII. Limitations, Ethics & Conclusion

This study is bounded by moderate sample counts (`N=24` E2E turns $\times 3$ seeds, `N=10` rubric answers) selected for CPU-only reproducibility, Azure `EdgeTTS` proxies for `en-IN` stress tests, and non-streaming `3.6–6.9 s` CPU turn latency. Responsibly, custom voice cloning stores embeddings locally with instant rename/delete controls, protects fixed system voices, and restricts interview rubric outputs strictly to human-reviewed co-pilot assistance due to residual accent variance (`0.161 US` vs `0.251 AU` WER). All core runtime tests (`58` pytest cases, Docker build, GitHub Actions CI) pass offline cleanly.

**Conclusion:** End-to-end voice agent reliability on edge hardware requires coordinated cross-layer design rather than larger isolated models. Pairing capacity-gated `Whisper INT8` ASR, fault-tolerant `Qwen 2.5 3B (Apache-2.0)` SLU, and flow-matching `Kyutai Pocket TTS INT8 (CC-BY-4.0)` establishes a reproducible, commercially cleared Pareto frontier lifting strict pipeline success from **`0.0%` to `41.67%` (`Full`)** and **`37.50%` (`Edge` at `$0.0499 / 1k turns`)**.

---

## References (IEEE)

[1] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, 2023.  
[2] A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," in *Proc. Interspeech*, 2020.  
[3] A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *NeurIPS*, 2020.  
[4] V. Panayotov et al., "LibriSpeech: An ASR corpus based on public domain audio books," in *Proc. IEEE ICASSP*, 2015, pp. 5206–5210.  
[5] G. Sun et al., "Can Contextual Biasing Remain Effective with Whisper and GPT-2?," in *Proc. Interspeech*, 2023.  
[6] X. Yang et al., "PromptASR for Contextualized ASR with Controllable Style," in *Proc. IEEE ICASSP*, 2024.  
[7] Y. Li et al., "CB-Whisper: Contextual Biasing Whisper Using Open-Vocabulary Keyword-Spotting," in *Proc. LREC-COLING*, 2024.  
[8] R. Ma et al., "N-best T5: Robust ASR Error Correction using Multiple Input Hypotheses and Constrained Decoding Space," in *Proc. Interspeech*, 2023.  
[9] C. Chen et al., "HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models," in *NeurIPS Datasets & Benchmarks*, 2023.  
[10] S. Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction," in *Proc. EMNLP*, 2019.  
[11] D. Gerz et al., "Multilingual and Cross-Lingual Intent Detection from Spoken Data," in *Proc. EMNLP*, 2021.  
[12] K. Everson et al., "Towards ASR Robust Spoken Language Understanding Through In-Context Learning With Word Confusion Networks," in *Proc. IEEE ICASSP*, 2024.  
[13] Qwen Team (A. Yang et al.), "Qwen2.5 Technical Report," *arXiv:2412.15115*, 2024.  
[14] A. Dubey et al., "The Llama 3 Herd of Models," *arXiv:2407.21783*, 2024.  
[15] B. T. Willard and R. Louf, "Efficient Guided Generation for Large Language Models," *arXiv:2307.09702*, 2023.  
[16] D. Snyder et al., "X-Vectors: Robust DNN Embeddings for Speaker Recognition," in *Proc. IEEE ICASSP*, 2018.  
[17] L. Wan et al., "Generalized End-to-End Loss for Speaker Verification," in *Proc. IEEE ICASSP*, 2018.  
[18] B. Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," in *Proc. Interspeech*, 2020.  
[19] H. Wang et al., "CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking," in *Proc. Interspeech*, 2023.  
[20] C. H. Taal et al., "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech," *IEEE TASLP*, vol. 19, no. 7, pp. 2125–2136, Sept. 2011.  
[21] W. Verhelst and M. Roelands, "An Overlap-Add Technique Based on Waveform Similarity (WSOLA) for High Quality Time-Scale Modification of Speech," in *Proc. IEEE ICASSP*, vol. 2, 1993, pp. 554–557.  
[22] Y. Jia et al., "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis," in *NeurIPS*, 2018.  
[23] J. Kim, J. Kong, and J. Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," in *Proc. ICML*, 2021.  
[24] E. Casanova et al., "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone," in *Proc. ICML*, 2022.  
[25] E. Kharitonov et al., "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision," *TACL*, 2023.  
[26] Z. Du et al., "CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens," *arXiv:2407.05407*, 2024.  
[27] S. Mehta et al., "Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching," in *Proc. IEEE ICASSP*, 2024.  
[28] Kyutai, "Pocket TTS: Lightweight Open-Science CPU Flow-Matching Zero-Shot Text-to-Speech," `CC-BY-4.0`, 2026.
