"""
Stage 2: Speech Recognition (ASR) Microservices.
Converts acoustic audio waveform into textual transcript.
Implements IASRService following Open/Closed Principle.
"""

import os
import re
import time
import math
import speech_recognition as sr
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from ..contracts import IASRService, AudioPayload, ASRResult


class GoogleWebSpeechASR(IASRService):
    """ASR implementation using Google Web Speech API."""

    def __init__(self, language: str = "en-US"):
        self.language = language
        self.recognizer = sr.Recognizer()

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        start_time = time.time()
        
        if not payload.file_path or not os.path.exists(payload.file_path):
            return ASRResult(
                transcript="",
                provider="google_web_speech",
                status="error",
                error="Audio file path does not exist",
                latency_sec=0.0
            )

        try:
            with sr.AudioFile(payload.file_path) as source:
                audio_data = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio_data, language=self.language)
            elapsed = round(time.time() - start_time, 3)
            return ASRResult(
                transcript=text,
                confidence=0.92,
                language=self.language,
                latency_sec=elapsed,
                provider="google_web_speech",
                status="success"
            )

        except sr.UnknownValueError:
            elapsed = round(time.time() - start_time, 3)
            return ASRResult(
                transcript="",
                confidence=0.0,
                language=self.language,
                latency_sec=elapsed,
                provider="google_web_speech",
                status="unintelligible",
                error="Audio unintelligible or silent"
            )

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return ASRResult(
                transcript="",
                confidence=0.0,
                language=self.language,
                latency_sec=elapsed,
                provider="google_web_speech",
                status="error",
                error=str(e)
            )


# Module-level model cache to avoid re-loading weights on each turn
_WHISPER_CACHE = {}


def _soundex(word: str) -> str:
    """Computes 4-character American Soundex code for phonetic similarity matching."""
    clean = re.sub(r"[^a-z]", "", (word or "").lower())
    if not clean:
        return ""
    mapping = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3",
        "l": "4",
        "m": "5", "n": "5",
        "r": "6",
    }
    first = clean[0].upper()
    digits: List[str] = []
    prev_code = mapping.get(clean[0], "")
    for ch in clean[1:]:
        code = mapping.get(ch, "")
        if code and code != prev_code:
            digits.append(code)
        prev_code = code if code else ("" if ch not in "hw" else prev_code)
    return (first + "".join(digits) + "000")[:4]


def build_dynamic_bias_prompt(
    role_terms: Optional[Sequence[str]] = None,
    candidate_names: Optional[Sequence[str]] = None,
    base_context: str = "Interview dialogue and technical terms:",
) -> str:
    """Builds a concise deduplicated contextual prompt from role keywords and candidate names."""
    seen = set()
    ordered: List[str] = []
    for term in list(candidate_names or []) + list(role_terms or []):
        t = (term or "").strip()
        low = t.lower()
        if t and low not in seen:
            seen.add(low)
            ordered.append(t)
    if not ordered:
        return WhisperTransformerASR.DEFAULT_PROMPT
    return f"{base_context} {', '.join(ordered)}."


_G2P_INSTANCE = None
_G2P_CACHE: Dict[str, Tuple[str, ...]] = {}

# Guard frequent function/filler words from being accidentally overwritten into domain terms
_COMMON_PROSE_GUARD = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "they", "their", "there",
    "what", "when", "where", "which", "while", "have", "has", "had", "will", "would",
    "could", "should", "about", "into", "only", "more", "some", "very", "just",
    "also", "than", "then", "your", "youre", "been", "being", "over", "under",
    "every", "other", "these", "those", "after", "before", "built", "using",
    "today", "hello", "great", "please", "thank", "thanks", "helping",
    "amount", "fees", "free", "called", "aboard", "dress", "money", "want",
    "need", "know", "make", "take", "give", "keep", "find", "think", "show",
    "work", "right", "first", "still", "around", "much", "well", "many",
})

_FUNCTION_BIGRAM_GUARD = frozenset({
    "a", "an", "the", "to", "in", "on", "of", "at", "by", "my",
    "no", "or", "if", "is", "it", "as", "so", "we", "he", "be", "do", "up",
})

_MORPH_SUFFIXES = ("s", "es", "ed", "ing", "er", "ly")


def _is_morphological_inflection(norm: str, exact_set: set) -> bool:
    """Prevents repairing valid inflected forms (e.g., 'cards', 'debits', 'depositing') into base lemmas."""
    for suf in _MORPH_SUFFIXES:
        if norm.endswith(suf) and len(norm) > len(suf) + 2:
            stem = norm[: -len(suf)]
            if stem in exact_set or (suf == "ing" and (stem + "e") in exact_set) or (suf == "ed" and (stem + "e") in exact_set):
                return True
    return False

_CONFUSABLE_PHONEME_PAIRS = frozenset({
    ("P", "B"), ("P", "Y"), ("P", "F"), ("B", "V"), ("T", "D"), ("T", "TH"),
    ("K", "G"), ("F", "V"), ("F", "TH"), ("S", "Z"), ("S", "SH"), ("SH", "ZH"),
    ("HH", "JH"), ("HH", "Y"), ("JH", "ZH"), ("CH", "JH"), ("N", "NG"), ("M", "N"),
    ("R", "L"), ("R", "ER"), ("W", "V"), ("IY", "EY"), ("IY", "IH"), ("EY", "EH"),
    ("EH", "AE"), ("AA", "AH"), ("AA", "AO"), ("UH", "UW"), ("OW", "AW"), ("ER", "AH"),
})


def _g2p_phonemes(word: str) -> Tuple[str, ...]:
    """Converts a word into stress-stripped ARPAbet phoneme tokens via Apache-2.0 g2p_en."""
    global _G2P_INSTANCE
    clean = re.sub(r"[^a-z]", "", (word or "").lower())
    if not clean:
        return ()
    if clean in _G2P_CACHE:
        return _G2P_CACHE[clean]
    try:
        if _G2P_INSTANCE is None:
            from g2p_en import G2p
            _G2P_INSTANCE = G2p()
        raw_phones = _G2P_INSTANCE(clean)
    except Exception:
        _G2P_CACHE[clean] = ()
        return ()
    phones = tuple(re.sub(r"\d+$", "", p) for p in raw_phones if p.strip() and p.isalpha() or re.match(r"^[A-Z]+\d*$", p))
    _G2P_CACHE[clean] = phones
    return phones


def _phoneme_similarity(p1: Tuple[str, ...], p2: Tuple[str, ...]) -> float:
    """Computes articulatory-weighted normalized phoneme similarity in [0.0, 1.0]."""
    n, m = len(p1), len(p2)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = p1[i - 1], p2[j - 1]
            if a == b:
                cost = 0.0
            elif (a, b) in _CONFUSABLE_PHONEME_PAIRS or (b, a) in _CONFUSABLE_PHONEME_PAIRS:
                cost = 0.40
            else:
                cost = 1.0
            dp[i][j] = min(dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0, dp[i - 1][j - 1] + cost)
    dist = dp[n][m]
    return max(0.0, 1.0 - (dist / float(max(n, m))))


def phonetic_repair(
    transcript: str,
    lexicon: Sequence[str],
    words_meta: Optional[Sequence[Dict[str, Any]]] = None,
    confidence_threshold: float = 0.85,
    phonetic_mode: str = "g2p",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Applies confidence-gated G2P ARPAbet phoneme-edit (or legacy Soundex) + fuzzy edit repair
    on domain lexicon terms, guarding frequent English prose against false over-correction.
    """
    if not transcript or not lexicon:
        return transcript, []

    from rapidfuzz import fuzz

    use_g2p = phonetic_mode.lower() == "g2p"
    lex_entries: List[Tuple[str, str, str, Tuple[str, ...]]] = []
    exact_set = set()
    for term in lexicon:
        t = (term or "").strip()
        low = re.sub(r"[^a-z0-9]", "", t.lower())
        if len(low) >= 3 and low not in exact_set:
            exact_set.add(low)
            phones = _g2p_phonemes(low) if use_g2p else ()
            lex_entries.append((t, low, _soundex(low), phones))

    def _split_punct(tok: str) -> Tuple[str, str, str]:
        m = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$", tok)
        return (m.group(1), m.group(2), m.group(3)) if m else ("", tok, "")

    word_probs = []
    for w in (words_meta or []):
        w_norm = re.sub(r"[^a-z0-9]", "", (w.get("word") or "").lower())
        if w_norm:
            word_probs.append((w_norm, float(w.get("probability", 1.0))))

    no_word_conf = not word_probs
    cursor = 0

    def _next_prob(norm: str) -> float:
        nonlocal cursor
        if not word_probs:
            return 0.65
        for idx in range(cursor, min(cursor + 4, len(word_probs))):
            if word_probs[idx][0] == norm or norm in word_probs[idx][0] or word_probs[idx][0] in norm:
                cursor = idx + 1
                return word_probs[idx][1]
        return 0.65

    def _best_candidate(norm: str, require_strict: bool = False) -> Optional[Tuple[str, str, float]]:
        if not norm or (not require_strict and norm in exact_set) or norm in _COMMON_PROSE_GUARD or _is_morphological_inflection(norm, exact_set):
            return None
        snd = _soundex(norm)
        cand_phones = _g2p_phonemes(norm) if use_g2p else ()
        best = None
        best_score = 0.0
        for orig_term, low_term, term_snd, term_phones in lex_entries:
            if abs(len(norm) - len(low_term)) > max(3, len(low_term) // 2):
                continue
            if norm == low_term:
                return (orig_term, "exact_join", 100.0)
            ratio = float(fuzz.ratio(norm, low_term))
            snd_eq = bool(snd and term_snd and snd == term_snd)
            ph_sim = _phoneme_similarity(cand_phones, term_phones) if (use_g2p and cand_phones and term_phones) else 0.0

            if use_g2p:
                if require_strict:
                    ok = (ph_sim >= 0.86) or (ph_sim >= 0.78 and ratio >= 55.0) or (snd_eq and ratio >= 72.0) or (ratio >= 86.0 and norm[0] == low_term[0])
                else:
                    ok = (
                        (ph_sim >= 0.85 and abs(len(norm) - len(low_term)) <= 2)
                        or (ph_sim >= 0.78 and abs(len(norm) - len(low_term)) <= 2 and ratio >= 40.0)
                        or (ph_sim >= 0.68 and ratio >= 58.0)
                        or (snd_eq and ratio >= 64.0)
                        or (ratio >= 86.0 and abs(len(norm) - len(low_term)) <= 2 and norm[0] == low_term[0])
                    )
                combined = max(ratio, ph_sim * 100.0)
                method = "g2p_phoneme" if ph_sim >= 0.62 else ("soundex+fuzz" if snd_eq else "fuzzy_edit")
            else:
                if require_strict:
                    ok = (snd_eq and ratio >= 72.0) or (ratio >= 86.0 and norm[0] == low_term[0])
                else:
                    ok = (snd_eq and ratio >= 64.0) or (ratio >= 84.0 and abs(len(norm) - len(low_term)) <= 2 and norm[0] == low_term[0])
                combined = ratio
                method = "soundex+fuzz" if snd_eq else "fuzzy_edit"

            if ok and combined > best_score:
                best_score = combined
                best = (orig_term, method, round(combined, 1))
        return best

    raw_tokens = transcript.split()
    out_tokens: List[str] = []
    repairs: List[Dict[str, Any]] = []
    i = 0
    while i < len(raw_tokens):
        pre, core, post = _split_punct(raw_tokens[i])
        norm = re.sub(r"[^a-z0-9]", "", core.lower())
        prob = _next_prob(norm)

        # Pass 1: Adjacent bigram merge (e.g., "Such in" -> "Sachin", "fast API" -> "FastAPI")
        # Never strip short determiners/prepositions like "a payment" -> "payment".
        merged = False
        if (
            i + 1 < len(raw_tokens)
            and not post
            and "-" not in core
            and norm not in _FUNCTION_BIGRAM_GUARD
            and len(norm) >= 2
        ):
            pre2, core2, post2 = _split_punct(raw_tokens[i + 1])
            if not pre2:
                norm2 = re.sub(r"[^a-z0-9]", "", core2.lower())
                joined = norm + norm2
                if (
                    len(joined) >= 5
                    and norm not in exact_set
                    and norm2 not in exact_set
                    and not _is_morphological_inflection(norm2, exact_set)
                ):
                    p2 = _next_prob(norm2) if word_probs else 0.65
                    # Do not over-consume word_probs cursor if bigram isn't merged
                    if word_probs and p2 != 0.65:
                        cursor = max(0, cursor - 1)
                    if min(prob, p2) < confidence_threshold:
                        cand = _best_candidate(joined, require_strict=True)
                        if cand is not None:
                            if word_probs:
                                _next_prob(norm2)
                            repaired_tok = f"{pre}{cand[0]}{post2}"
                            repairs.append({
                                "original": f"{raw_tokens[i]} {raw_tokens[i + 1]}",
                                "repaired": repaired_tok,
                                "method": f"bigram_{cand[1]}",
                                "score": cand[2],
                                "probability": round(min(prob, p2), 3),
                            })
                            out_tokens.append(repaired_tok)
                            i += 2
                            merged = True
        if merged:
            continue

        # Pass 2: Unigram confidence-gated repair (handles hyphenated prefixes like "Babyl-optimized")
        if "-" in core and norm not in exact_set and prob < confidence_threshold:
            parts = core.split("-")
            changed = False
            new_parts = []
            for p_part in parts:
                p_norm = re.sub(r"[^a-z0-9]", "", p_part.lower())
                cand = _best_candidate(p_norm, require_strict=no_word_conf) if len(p_norm) >= 4 else None
                if cand is not None:
                    repairs.append({
                        "original": p_part,
                        "repaired": cand[0],
                        "method": cand[1],
                        "score": cand[2],
                        "probability": round(prob, 3),
                    })
                    new_parts.append(cand[0])
                    changed = True
                else:
                    new_parts.append(p_part)
            if changed:
                out_tokens.append(f"{pre}{'-'.join(new_parts)}{post}")
                i += 1
                continue

        if len(norm) >= 4 and norm not in exact_set and prob < confidence_threshold:
            cand = _best_candidate(norm, require_strict=no_word_conf)
            if cand is not None:
                repaired_tok = f"{pre}{cand[0]}{post}"
                repairs.append({
                    "original": raw_tokens[i],
                    "repaired": repaired_tok,
                    "method": cand[1],
                    "score": cand[2],
                    "probability": round(prob, 3),
                })
                out_tokens.append(repaired_tok)
                i += 1
                continue

        out_tokens.append(raw_tokens[i])
        i += 1

    return " ".join(out_tokens), repairs


class WhisperTransformerASR(IASRService):
    """
    High-Performance Offline Whisper Transformer ASR Service powered by faster-whisper (CTranslate2).
    Supports INT8 CPU quantization, Silero VAD, dynamic prompt/hotword logit biasing, word timestamps,
    and confidence-gated Soundex + fuzzy phonetic repair on proper nouns and technical vocabulary.
    """

    DEFAULT_PROMPT = (
        "Interview dialogue and proper names: "
        "Sachin, Nandu, Bablu, Rahul, Aradhya, Priya, Vikram, Ananya, Rohit, "
        "Rohan, Neha, Amit, Sneha, Aditya, Pooja, Rajesh, Suresh, Deepa, Karthik, "
        "Sanjay, Manish, Alok, Divya, Meera, Shreya, Varun, Gaurav, Tanvi, Ritu, "
        "Nikhil, Harish, Swati, Rakesh, Vishal."
    )

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        vad_filter: bool = True,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        word_timestamps: bool = False,
        repair_lexicon: Optional[Sequence[str]] = None,
        repair_confidence_threshold: float = 0.85,
        phonetic_mode: str = "g2p",
        condition_on_previous_text: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        hallucination_silence_threshold: Optional[float] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt if initial_prompt is not None else self.DEFAULT_PROMPT
        self.hotwords = hotwords
        self.word_timestamps = word_timestamps
        self.repair_lexicon = list(repair_lexicon) if repair_lexicon else None
        self.repair_confidence_threshold = repair_confidence_threshold
        self.phonetic_mode = phonetic_mode
        self.condition_on_previous_text = condition_on_previous_text
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.model = None
        self._is_initialized = False

    def _lazy_init(self):
        """Lazy load and cache Whisper model weights into memory."""
        cache_key = f"{self.model_size}_{self.device}_{self.compute_type}"
        global _WHISPER_CACHE

        if cache_key in _WHISPER_CACHE:
            self.model = _WHISPER_CACHE[cache_key]
            self._is_initialized = True
            return

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            _WHISPER_CACHE[cache_key] = model
            self.model = model
            self._is_initialized = True
        except Exception:
            # Pluggable fallback to openai-whisper if present
            try:
                import whisper
                model = whisper.load_model(self.model_size.split(".")[0], device=self.device)
                _WHISPER_CACHE[cache_key] = model
                self.model = model
                self._is_initialized = True
            except Exception:
                self.model = None
                self._is_initialized = True

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        start_time = time.time()
        self._lazy_init()

        audio_input: Any = None
        if payload.file_path and os.path.exists(payload.file_path):
            audio_input = payload.file_path
        elif payload.raw_bytes and payload.metadata.get("audio_dtype") == "float32":
            audio_input = np.frombuffer(payload.raw_bytes, dtype=np.float32).copy()

        if audio_input is None:
            return ASRResult(
                transcript="",
                confidence=0.0,
                language=self.language,
                latency_sec=0.0,
                provider=f"whisper_{self.model_size}",
                status="error",
                error="Audio file path does not exist"
            )

        if self.model is not None:
            try:
                import faster_whisper
                if isinstance(self.model, faster_whisper.WhisperModel):
                    lang_param = None if self.model_size.endswith(".en") else self.language
                    req_meta = payload.metadata or {}
                    prompt = req_meta.get("initial_prompt", self.initial_prompt)
                    hw = req_meta.get("hotwords", self.hotwords)
                    use_words = bool(req_meta.get("word_timestamps", self.word_timestamps))
                    lexicon = req_meta.get("repair_lexicon", self.repair_lexicon)

                    transcribe_kwargs: Dict[str, Any] = {
                        "language": lang_param,
                        "beam_size": self.beam_size,
                        "vad_filter": self.vad_filter,
                        "initial_prompt": prompt or None,
                        "word_timestamps": use_words or bool(lexicon),
                        "condition_on_previous_text": bool(
                            req_meta.get("condition_on_previous_text", self.condition_on_previous_text)
                        ),
                        "repetition_penalty": float(
                            req_meta.get("repetition_penalty", self.repetition_penalty)
                        ),
                        "no_repeat_ngram_size": int(
                            req_meta.get("no_repeat_ngram_size", self.no_repeat_ngram_size)
                        ),
                    }
                    hst = req_meta.get("hallucination_silence_threshold", self.hallucination_silence_threshold)
                    if hst is not None:
                        transcribe_kwargs["hallucination_silence_threshold"] = float(hst)
                    if hw:
                        transcribe_kwargs["hotwords"] = hw

                    segments, info = self.model.transcribe(audio_input, **transcribe_kwargs)

                    text_parts = []
                    probs = []
                    segments_list = []
                    words_list: List[Dict[str, Any]] = []
                    for seg in segments:
                        txt = seg.text.strip()
                        if txt:
                            text_parts.append(txt)
                            if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                                probs.append(math.exp(seg.avg_logprob))
                            segments_list.append({
                                "start": round(seg.start, 2),
                                "end": round(seg.end, 2),
                                "text": txt
                            })
                        if transcribe_kwargs["word_timestamps"] and getattr(seg, "words", None):
                            for w in seg.words:
                                words_list.append({
                                    "word": (w.word or "").strip(),
                                    "start": round(float(w.start), 2),
                                    "end": round(float(w.end), 2),
                                    "probability": round(float(w.probability), 3),
                                })

                    raw_transcript = " ".join(text_parts).strip()
                    full_transcript = raw_transcript
                    repairs: List[Dict[str, Any]] = []
                    if lexicon and raw_transcript:
                        full_transcript, repairs = phonetic_repair(
                            raw_transcript,
                            lexicon=lexicon,
                            words_meta=words_list,
                            confidence_threshold=self.repair_confidence_threshold,
                            phonetic_mode=req_meta.get("phonetic_mode", self.phonetic_mode),
                        )

                    avg_confidence = round(sum(probs) / len(probs), 2) if probs else 0.95
                    elapsed = round(time.time() - start_time, 3)

                    meta: Dict[str, Any] = {
                        "segments": segments_list,
                        "beam_size": self.beam_size,
                        "vad_filter": self.vad_filter,
                        "hotwords_active": bool(hw),
                        "language_probability": round(float(getattr(info, "language_probability", 1.0)), 3),
                    }
                    if use_words:
                        meta["words"] = words_list
                    if repairs:
                        meta["raw_transcript"] = raw_transcript
                        meta["phonetic_repairs"] = repairs

                    return ASRResult(
                        transcript=full_transcript,
                        confidence=avg_confidence,
                        language=info.language or self.language,
                        latency_sec=elapsed,
                        provider=f"whisper_{self.model_size}_{self.compute_type}",
                        status="success" if full_transcript else "unintelligible",
                        metadata=meta
                    )
                else:
                    # Fallback to standard OpenAI Whisper
                    result = self.model.transcribe(audio_input, language=self.language)
                    elapsed = round(time.time() - start_time, 3)
                    return ASRResult(
                        transcript=result.get("text", "").strip(),
                        confidence=0.92,
                        language=self.language,
                        latency_sec=elapsed,
                        provider=f"whisper_{self.model_size}",
                        status="success"
                    )
            except Exception as e:
                print(f"[Whisper ASR] Transcription error: {e}. Falling back...")

        # Graceful fallback to Google Web Speech ASR
        fallback = GoogleWebSpeechASR(language="en-US")
        res = fallback.transcribe(payload)
        res.provider = f"whisper_{self.model_size} (fallback)"
        return res


class ParakeetTDTASR(IASRService):
    """
    2026 SOTA Offline Transducer ASR Service powered by sherpa-onnx + nvidia/parakeet-tdt-0.6b-v3-int8 (CC-BY-4.0).
    Supports native hotword boosting via modified_beam_search plus optional G2P phoneme repair.
    """

    DEFAULT_MODEL_DIR = "models/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    _recognizer_cache: Dict[str, Any] = {}

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        num_threads: int = 4,
        decoding_method: str = "greedy_search",
        hotwords: Optional[Sequence[str]] = None,
        hotwords_score: float = 1.5,
        repair_lexicon: Optional[Sequence[str]] = None,
        phonetic_mode: str = "g2p",
    ):
        self.model_dir = model_dir
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.hotwords = list(hotwords) if hotwords else None
        self.hotwords_score = hotwords_score
        self.repair_lexicon = list(repair_lexicon) if repair_lexicon else None
        self.phonetic_mode = phonetic_mode

    def _get_recognizer(self, hotwords_file: Optional[str] = None, method: Optional[str] = None):
        import sherpa_onnx
        dec_m = method or self.decoding_method
        hw_key = hotwords_file or ""
        cache_key = f"{self.model_dir}|{dec_m}|{hw_key}"
        if cache_key in self._recognizer_cache:
            return self._recognizer_cache[cache_key]

        bpe_vocab = os.path.join(self.model_dir, "bpe.vocab")
        kwargs: Dict[str, Any] = {
            "tokens": os.path.join(self.model_dir, "tokens.txt"),
            "encoder": os.path.join(self.model_dir, "encoder.int8.onnx"),
            "decoder": os.path.join(self.model_dir, "decoder.int8.onnx"),
            "joiner": os.path.join(self.model_dir, "joiner.int8.onnx"),
            "num_threads": self.num_threads,
            "decoding_method": dec_m,
            "provider": "cpu",
            "model_type": "nemo_transducer",
        }
        if hotwords_file and os.path.exists(bpe_vocab) and dec_m == "modified_beam_search":
            kwargs["hotwords_file"] = hotwords_file
            kwargs["hotwords_score"] = self.hotwords_score
            kwargs["modeling_unit"] = "bpe"
            kwargs["bpe_vocab"] = bpe_vocab

        rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        self._recognizer_cache[cache_key] = rec
        return rec

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        import soundfile as sf
        import scipy.signal as sig

        start_time = time.time()
        if not os.path.exists(os.path.join(self.model_dir, "encoder.int8.onnx")):
            return ASRResult(
                transcript="", confidence=0.0, language="en", latency_sec=0.0,
                provider="parakeet_tdt_0.6b_v3_int8", status="error",
                error=f"Parakeet ONNX model directory missing: {self.model_dir}",
            )

        if payload.file_path and os.path.exists(payload.file_path):
            audio, sr = sf.read(payload.file_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = sig.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
        elif payload.raw_bytes and payload.metadata.get("audio_dtype") == "float32":
            audio = np.frombuffer(payload.raw_bytes, dtype=np.float32).copy()
            sr = int(payload.sample_rate or 16000)
        else:
            return ASRResult(
                transcript="", confidence=0.0, language="en", latency_sec=0.0,
                provider="parakeet_tdt_0.6b_v3_int8", status="error", error="Invalid audio payload",
            )

        req_meta = payload.metadata or {}
        lex = req_meta.get("repair_lexicon", self.repair_lexicon)
        hw_list = req_meta.get("hotwords_list", self.hotwords)
        tmp_hw = None
        dec_m = self.decoding_method
        try:
            if hw_list and os.path.exists(os.path.join(self.model_dir, "bpe.vocab")):
                import tempfile
                dec_m = "modified_beam_search"
                fd, tmp_hw = tempfile.mkstemp(suffix=".txt", text=True)
                with os.fdopen(fd, "w", encoding="utf-8") as hf:
                    for term in hw_list:
                        if str(term).strip():
                            hf.write(str(term).strip() + "\n")
            max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
            if 0.0 < max_abs < 0.05:
                # Guard against quiet-input frontend attenuation documented in parakeet-tdt-0.6b-v3
                audio = np.clip(audio * (0.25 / max_abs), -1.0, 1.0).astype(np.float32)
            rec = self._get_recognizer(hotwords_file=tmp_hw, method=dec_m)
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
            rec.decode_stream(stream)
            raw_text = (stream.result.text or "").strip()
        finally:
            if tmp_hw and os.path.exists(tmp_hw):
                os.remove(tmp_hw)

        full_text = raw_text
        repairs: List[Dict[str, Any]] = []
        if lex and raw_text:
            full_text, repairs = phonetic_repair(
                raw_text, lexicon=lex, phonetic_mode=req_meta.get("phonetic_mode", self.phonetic_mode)
            )
        elapsed = round(time.time() - start_time, 3)
        return ASRResult(
            transcript=full_text,
            confidence=0.95 if full_text else 0.0,
            language="en",
            latency_sec=elapsed,
            provider=f"parakeet_tdt_0.6b_v3_int8_{dec_m}",
            status="success" if full_text else "unintelligible",
            metadata={"raw_transcript": raw_text, "phonetic_repairs": repairs, "decoding_method": dec_m},
        )


class MockASR(IASRService):
    """Deterministic mock ASR for testing without network or audio devices."""

    def __init__(self, fixed_transcript: str = "Hello Mira, what is the time?"):
        self.fixed_transcript = fixed_transcript

    def transcribe(self, payload: AudioPayload) -> ASRResult:
        return ASRResult(
            transcript=self.fixed_transcript,
            confidence=1.0,
            language="en-US",
            latency_sec=0.001,
            provider="mock_asr",
            status="success"
        )
