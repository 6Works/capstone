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


def phonetic_repair(
    transcript: str,
    lexicon: Sequence[str],
    words_meta: Optional[Sequence[Dict[str, Any]]] = None,
    confidence_threshold: float = 0.85,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Applies confidence-gated Soundex + fuzzy edit repair on domain lexicon terms.
    Targets low-confidence ASR tokens and split compound terms without disturbing high-confidence prose.
    """
    if not transcript or not lexicon:
        return transcript, []

    from rapidfuzz import fuzz

    lex_entries: List[Tuple[str, str, str]] = []
    exact_set = set()
    for term in lexicon:
        t = (term or "").strip()
        low = re.sub(r"[^a-z0-9]", "", t.lower())
        if len(low) >= 3 and low not in exact_set:
            exact_set.add(low)
            lex_entries.append((t, low, _soundex(low)))

    def _split_punct(tok: str) -> Tuple[str, str, str]:
        m = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$", tok)
        return (m.group(1), m.group(2), m.group(3)) if m else ("", tok, "")

    word_probs = []
    for w in (words_meta or []):
        w_norm = re.sub(r"[^a-z0-9]", "", (w.get("word") or "").lower())
        if w_norm:
            word_probs.append((w_norm, float(w.get("probability", 1.0))))

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
        if not norm or (not require_strict and norm in exact_set):
            return None
        snd = _soundex(norm)
        best = None
        best_score = 0.0
        for orig_term, low_term, term_snd in lex_entries:
            if abs(len(norm) - len(low_term)) > max(3, len(low_term) // 2):
                continue
            if norm == low_term:
                return (orig_term, "exact_join", 100.0)
            ratio = float(fuzz.ratio(norm, low_term))
            snd_eq = bool(snd and term_snd and snd == term_snd)
            if require_strict:
                ok = (snd_eq and ratio >= 72.0) or (ratio >= 86.0 and norm[0] == low_term[0])
            else:
                ok = (snd_eq and ratio >= 64.0) or (ratio >= 84.0 and abs(len(norm) - len(low_term)) <= 2 and norm[0] == low_term[0])
            if ok and ratio > best_score:
                best_score = ratio
                best = (orig_term, "soundex+fuzz" if snd_eq else "fuzzy_edit", round(ratio, 1))
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
        merged = False
        if i + 1 < len(raw_tokens) and not post and "-" not in core:
            pre2, core2, post2 = _split_punct(raw_tokens[i + 1])
            if not pre2:
                norm2 = re.sub(r"[^a-z0-9]", "", core2.lower())
                joined = norm + norm2
                if len(joined) >= 5 and norm not in exact_set:
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
                cand = _best_candidate(p_norm) if len(p_norm) >= 4 else None
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
            cand = _best_candidate(norm)
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
