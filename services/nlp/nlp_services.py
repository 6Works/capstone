"""
Stage 3: Natural Language Processing (NLP) Microservices.
Extracts intents, entities, sentiment, and orchestrates tool definitions.
Implements INLPService following SOLID.
"""

import os
import re
import time
from typing import Dict, Any, List, Optional, Sequence, Tuple
from ..contracts import INLPService, NLPResult
from services.llm.gateway import llm_gateway
from services.llm.llm_contracts import LLMMessage
from services.llm.json_utils import repair_and_parse_with_retry


def preprocess_asr_transcript(
    text: str,
    repair_lexicon: Optional[Sequence[str]] = None,
    confidence_threshold: float = 0.85,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Optional Layer B pre-repair hook correcting noisy ASR transcripts before SLU parsing."""
    cleaned = (text or "").strip()
    if not cleaned or not repair_lexicon:
        return cleaned, []
    from services.asr import phonetic_repair
    repaired, fixes = phonetic_repair(
        cleaned,
        lexicon=repair_lexicon,
        words_meta=None,
        confidence_threshold=confidence_threshold,
    )
    return repaired, fixes


class RuleBasedNLPService(INLPService):
    """Deterministic, lightweight NLU engine with regex intent & entity extraction."""

    def __init__(self, repair_lexicon: Optional[Sequence[str]] = None):
        self.repair_lexicon = list(repair_lexicon) if repair_lexicon else None
        self.intent_patterns = [
            (r"\b(hello|hi|hey|good\s*(morning|afternoon|evening))\b", "greeting"),
            (r"\b(bye|goodbye|see you|exit|quit|stop)\b", "farewell"),
            (r"\b(who are you|your name|what is this|introduce yourself)\b", "identity_query"),
            (r"\b(what is the time|current time|what time is it)\b", "time_query"),
            (r"\b(weather|temperature|forecast|is it raining)\b", "weather_query"),
            (r"\b(capstone|project|model|pipeline|stages|architecture|solid)\b", "project_query"),
            (r"\b(calculate|what is|how much is)\s+([0-9\+\-\*\/\^\(\)\.\s]+)", "calculation_query"),
            (r"\b(help|commands|what can you do|features)\b", "help_query"),
        ]

    def parse(self, text: str) -> NLPResult:
        start_time = time.time()
        cleaned, asr_fixes = preprocess_asr_transcript(text, self.repair_lexicon)
        lower = cleaned.lower()

        intent = "general_query"
        confidence = 0.5
        entities: Dict[str, Any] = {}
        tool_calls: List[Dict[str, Any]] = []

        # Intent matching
        for pattern, matched_intent in self.intent_patterns:
            m = re.search(pattern, lower)
            if m:
                intent = matched_intent
                confidence = 0.95
                if matched_intent == "calculation_query" and len(m.groups()) > 1:
                    expr = m.group(2).strip()
                    entities["math_expression"] = expr
                    tool_calls.append({"tool": "calculator", "args": {"expression": expr}})
                break

        # Sentiment estimation
        pos_words = {"good", "great", "awesome", "nice", "love", "excellent", "happy"}
        neg_words = {"bad", "sad", "angry", "hate", "terrible", "poor", "wrong"}
        words = set(re.findall(r"\b\w+\b", lower))

        pos_len = len(words.intersection(pos_words))
        neg_len = len(words.intersection(neg_words))
        sentiment = "positive" if pos_len > neg_len else ("negative" if neg_len > pos_len else "neutral")

        elapsed = round(time.time() - start_time, 4)
        return NLPResult(
            cleaned_text=cleaned,
            intent=intent,
            confidence=confidence,
            entities=entities,
            sentiment=sentiment,
            tool_calls=tool_calls,
            latency_sec=elapsed,
            provider="rule_based_nlp",
            metadata={"schema_valid": True, "raw_valid": True, "asr_repairs": asr_fixes},
        )


class LLMNLPService(INLPService):
    """
    Advanced LLM-backed NLP Service orchestrated via LLMGateway.
    Pluggable across Google Gemini, OpenAI, and local Ollama with shared JSON schema repair,
    single constrained retry, and optional ASR phonetic error pre-repair.
    """

    SYSTEM_PROMPT = """You are an NLP NLU microservice in a conversational voice pipeline.
Analyze the user utterance and return ONLY a valid JSON object matching this schema:
{
  "cleaned_text": "sanitized input string",
  "intent": "one of [greeting, farewell, identity_query, time_query, weather_query, project_query, calculation_query, general_query, answer]",
  "confidence": float between 0.0 and 1.0,
  "entities": { "extracted_key": "extracted_val" },
  "sentiment": "one of [positive, neutral, negative]",
  "tool_calls": [ { "tool": "tool_name", "args": {} } ]
}
Do not include markdown fences, thoughts, or commentary."""

    EXPECTED_KEYS = ("intent", "confidence")

    def __init__(
        self,
        model_name: Optional[str] = None,
        allowed_intents: Optional[Sequence[str]] = None,
        repair_lexicon: Optional[Sequence[str]] = None,
        enable_schema_repair: bool = True,
        enable_repair_retry: bool = True,
        use_json_mode: bool = True,
        use_schema_mode: bool = False,
        use_logprobs: bool = False,
        oos_prob_threshold: Optional[float] = None,
        calibrated_prompt: bool = False,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.allowed_intents = list(allowed_intents) if allowed_intents else None
        self.repair_lexicon = list(repair_lexicon) if repair_lexicon else None
        self.enable_schema_repair = enable_schema_repair
        self.enable_repair_retry = enable_repair_retry
        self.use_json_mode = use_json_mode
        self.use_schema_mode = use_schema_mode
        self.use_logprobs = use_logprobs
        self.oos_prob_threshold = oos_prob_threshold
        self.calibrated_prompt = calibrated_prompt
        self.temperature = temperature
        self.seed = seed
        self.fallback = RuleBasedNLPService(repair_lexicon=self.repair_lexicon)

    def _build_response_schema(self) -> Optional[Dict[str, Any]]:
        """Builds a strict JSON Schema dict for Ollama native structured decoding."""
        if not self.use_schema_mode:
            return None
        intent_prop: Dict[str, Any] = {"type": "string"}
        if self.allowed_intents:
            intent_prop["enum"] = list(self.allowed_intents)
        return {
            "type": "object",
            "properties": {
                "intent": intent_prop,
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        }

    def _build_system_prompt(self, uncertain_spans: Optional[Sequence[str]] = None) -> str:
        if not self.allowed_intents:
            return self.SYSTEM_PROMPT
        opts = ", ".join(self.allowed_intents)
        guidance = ""
        if self.calibrated_prompt:
            hints = []
            if "answer" in self.allowed_intents and "project_query" in self.allowed_intents:
                hints.append(
                    'Use "answer" for declarative candidate technical explanations, architecture summaries, or work experience descriptions; '
                    'use "project_query" ONLY when the user explicitly asks a question about the capstone system or pipeline stages.'
                )
            if "oos" in self.allowed_intents:
                hints.append(
                    'Use "oos" when the request falls outside the listed capabilities rather than forcing a weakly related label.'
                )
            if hints:
                guidance = "\nDisambiguation rules:\n- " + "\n- ".join(hints)
        if uncertain_spans:
            guidance += f"\nNote ambiguous low-confidence ASR tokens that may contain homophones: {', '.join(uncertain_spans[:8])}."
        return (
            "You are a strict intent-classification NLU microservice. "
            "Classify the user utterance into EXACTLY ONE intent label from this closed list:\n"
            f"[{opts}]{guidance}\n"
            'Return ONLY a valid JSON object: {"intent": "<label>", "confidence": <0.0-1.0>}. '
            "Do not invent labels outside the list or add prose."
        )

    def parse_with_asr_context(self, text: str, words_meta: Optional[Sequence[Dict[str, Any]]] = None, conf_cutoff: float = 0.85) -> NLPResult:
        """Runs SLU while injecting low-confidence ASR token spans for cross-layer disambiguation."""
        uncertain = [
            str(w.get("word", "")).strip()
            for w in (words_meta or [])
            if str(w.get("word", "")).strip() and float(w.get("probability", 1.0)) < conf_cutoff
        ]
        return self.parse(text, uncertain_spans=uncertain or None)

    def _schema_retry(self, raw_bad: str, expected_keys: Sequence[str], gen_kwargs: Dict[str, Any]) -> str:
        keys_str = ", ".join(expected_keys) if expected_keys else "intent, confidence"
        retry_prompt = (
            f"Convert the following output into ONLY valid JSON containing keys [{keys_str}]. "
            "Do not add markdown or prose."
        )
        res = llm_gateway.generate(
            messages=[LLMMessage(role="user", content=raw_bad[:800])],
            model=self.model_name,
            system_prompt=retry_prompt,
            json_mode=True,
            temperature=0.0,
            max_tokens=220,
            **gen_kwargs,
        )
        return res.content

    def parse(self, text: str, uncertain_spans: Optional[Sequence[str]] = None) -> NLPResult:
        start_time = time.time()
        cleaned, asr_fixes = preprocess_asr_transcript(text, self.repair_lexicon)
        if not cleaned:
            return self.fallback.parse(text)

        messages = [LLMMessage(role="user", content=cleaned)]
        gen_kwargs: Dict[str, Any] = {}
        if self.seed is not None:
            gen_kwargs["seed"] = self.seed
        schema = self._build_response_schema()
        if schema is not None:
            gen_kwargs["response_schema"] = schema
        if self.use_logprobs:
            gen_kwargs["logprobs"] = True

        try:
            llm_res = llm_gateway.generate(
                messages=messages,
                model=self.model_name,
                system_prompt=self._build_system_prompt(uncertain_spans=uncertain_spans),
                json_mode=self.use_json_mode,
                temperature=self.temperature,
                max_tokens=256,
                **gen_kwargs,
            )

            retry_fn = (
                (lambda bad, keys: self._schema_retry(bad, keys, gen_kwargs))
                if (self.enable_schema_repair and self.enable_repair_retry and not llm_res.fallback_used)
                else None
            )
            repair_out = repair_and_parse_with_retry(
                llm_res.content,
                expected_keys=self.EXPECTED_KEYS,
                enable_repair=self.enable_schema_repair,
                retry_fn=retry_fn,
            )
            parsed = repair_out["parsed"]
            if parsed is None:
                raise ValueError("Invalid JSON schema after repair")

            elapsed = round(time.time() - start_time, 4)
            intent_val = str(parsed.get("intent", "general_query")).strip()
            try:
                conf_val = float(parsed.get("confidence", 0.9))
            except (TypeError, ValueError):
                conf_val = 0.75

            seq_prob = None
            raw_meta = llm_res.raw_metadata or {}
            if "seq_prob" in raw_meta:
                seq_prob = float(raw_meta["seq_prob"])
                conf_val = seq_prob

            oos_gated = False
            if (
                self.oos_prob_threshold is not None
                and self.allowed_intents
                and "oos" in self.allowed_intents
                and seq_prob is not None
                and seq_prob < float(self.oos_prob_threshold)
            ):
                intent_val = "oos"
                oos_gated = True

            return NLPResult(
                cleaned_text=str(parsed.get("cleaned_text", cleaned)),
                intent=intent_val,
                confidence=max(0.0, min(1.0, conf_val)),
                entities=parsed.get("entities") if isinstance(parsed.get("entities"), dict) else {},
                sentiment=str(parsed.get("sentiment", "neutral")),
                tool_calls=parsed.get("tool_calls") if isinstance(parsed.get("tool_calls"), list) else [],
                latency_sec=elapsed,
                provider=f"llm_{llm_res.provider}_{llm_res.model}" + (" (fallback)" if llm_res.fallback_used else ""),
                metadata={
                    "raw_valid": repair_out["raw_valid"],
                    "schema_valid": repair_out["schema_valid"],
                    "repaired": repair_out["repaired"],
                    "retried": repair_out["retried"],
                    "fixes": repair_out["fixes"],
                    "asr_repairs": asr_fixes,
                    "fallback_used": llm_res.fallback_used,
                    "tokens_prompt": llm_res.tokens_prompt,
                    "tokens_completion": llm_res.tokens_completion,
                    "schema_mode": bool(schema is not None),
                    "seq_prob": seq_prob,
                    "mean_logprob": raw_meta.get("mean_logprob"),
                    "min_logprob": raw_meta.get("min_logprob"),
                    "oos_gated": oos_gated,
                },
            )

        except Exception as e:
            res = self.fallback.parse(cleaned)
            res.provider = f"rule_based_fallback ({str(e)[:30]})"
            res.latency_sec = round(time.time() - start_time, 4)
            res.metadata.update({
                "raw_valid": False,
                "schema_valid": False,
                "repaired": False,
                "retried": self.enable_schema_repair and self.enable_repair_retry,
                "asr_repairs": asr_fixes,
                "error": str(e)[:120],
            })
            return res


class TrainedDistilBertSLUService(INLPService):
    """
    Lightweight locally trained Apache-2.0 DistilBERT-base LoRA cross-layer SLU + OOS classifier.
    Supports optional ASR word-confidence uncertainty token injection [UNCERTAIN: w(p)].
    """

    _MODEL_CACHE: Dict[str, Any] = {}

    def __init__(
        self,
        model_dir: str = "models/slu_lora_adapter",
        allowed_intents: Optional[Sequence[str]] = None,
        repair_lexicon: Optional[Sequence[str]] = None,
        oos_prob_threshold: Optional[float] = None,
        use_conf_tags: bool = True,
    ):
        self.model_dir = model_dir
        self.allowed_intents = list(allowed_intents) if allowed_intents else None
        self.repair_lexicon = list(repair_lexicon) if repair_lexicon else None
        self.oos_prob_threshold = oos_prob_threshold
        self.use_conf_tags = use_conf_tags

    def _get_loaded(self):
        import json
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        if self.model_dir in self._MODEL_CACHE:
            return self._MODEL_CACHE[self.model_dir]

        lmap = json.load(open(os.path.join(self.model_dir, "label_map.json"), "r", encoding="utf-8"))
        label2id = lmap["label2id"]
        id2label = {int(v): k for k, v in label2id.items()}
        tok = AutoTokenizer.from_pretrained(self.model_dir)
        base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label2id))
        model = PeftModel.from_pretrained(base, self.model_dir)
        model.eval()
        bundle = (model, tok, label2id, id2label, float(lmap.get("oos_threshold", 0.0)))
        self._MODEL_CACHE[self.model_dir] = bundle
        return bundle

    def parse(self, text: str, words_meta: Optional[Sequence[Dict[str, Any]]] = None) -> NLPResult:
        import torch
        import numpy as np

        start_time = time.time()
        cleaned, asr_fixes = preprocess_asr_transcript(text, self.repair_lexicon)
        model, tok, label2id, id2label, saved_tau = self._get_loaded()

        formatted = cleaned
        if self.use_conf_tags and words_meta:
            low_toks = [
                f"{str(w.get('word', '')).strip()}({float(w.get('probability', 1.0)):.2f})"
                for w in words_meta
                if str(w.get("word", "")).strip() and float(w.get("probability", 1.0)) < 0.85
            ]
            if low_toks:
                formatted = f"{cleaned} [UNCERTAIN: {', '.join(low_toks[:8])}]"

        enc = tok([formatted], truncation=True, padding=True, max_length=72, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[0].numpy()

        if self.allowed_intents:
            allowed_ids = [label2id[l] for l in self.allowed_intents if l in label2id]
            if allowed_ids:
                mask = np.full_like(logits, -1e9)
                mask[allowed_ids] = 0.0
                logits = logits + mask
        else:
            allowed_ids = list(range(len(id2label)))

        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 1.0
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        tau = self.oos_prob_threshold if self.oos_prob_threshold is not None else saved_tau
        oos_id = label2id.get("oos")
        oos_gated = False
        if tau > 0.0 and oos_id is not None and oos_id in allowed_ids and best_prob < tau:
            best_idx = oos_id
            oos_gated = True

        intent = id2label[best_idx]
        return NLPResult(
            cleaned_text=cleaned,
            intent=intent,
            confidence=round(best_prob, 4),
            entities={},
            sentiment="neutral",
            tool_calls=[],
            latency_sec=round(time.time() - start_time, 4),
            provider="distilbert_lora_slu",
            metadata={
                "raw_valid": True,
                "schema_valid": True,
                "repaired": False,
                "retried": False,
                "asr_repairs": asr_fixes,
                "oos_gated": oos_gated,
                "oos_threshold": tau,
                "margin": round(margin, 4),
            },
        )


class HybridConfidenceSLUService(INLPService):
    """
    Confidence-gated 2-tier SLU Cascade:
    Routes high-confidence in-taxonomy turns via DistilBERT-LoRA (~30 ms CPU) and escalates
    ambiguous, low-margin, OOS-borderline, or out-of-taxonomy turns to a generative SLM.
    """

    def __init__(
        self,
        fast_service: TrainedDistilBertSLUService,
        slow_service: Any,
        confidence_threshold: float = 0.85,
        margin_threshold: float = 0.25,
        escalate_on_oos: bool = True,
    ):
        self.fast_service = fast_service
        self.slow_service = slow_service
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.escalate_on_oos = escalate_on_oos

    def parse_with_asr_context(
        self,
        text: str,
        words_meta: Optional[Sequence[Dict[str, Any]]] = None,
        conf_cutoff: float = 0.85,
    ) -> NLPResult:
        return self.parse(text, words_meta=words_meta, conf_cutoff=conf_cutoff)

    def parse(
        self,
        text: str,
        words_meta: Optional[Sequence[Dict[str, Any]]] = None,
        conf_cutoff: float = 0.85,
    ) -> NLPResult:
        start_time = time.time()
        _, _, label2id, _, _ = self.fast_service._get_loaded()
        allowed = self.fast_service.allowed_intents
        supported = bool(allowed) and all(lbl in label2id for lbl in allowed)

        if not supported:
            slow_res = (
                self.slow_service.parse_with_asr_context(text, words_meta=words_meta, conf_cutoff=conf_cutoff)
                if words_meta and hasattr(self.slow_service, "parse_with_asr_context")
                else self.slow_service.parse(text)
            )
            slow_res.latency_sec = round(time.time() - start_time, 4)
            slow_res.provider = f"hybrid_slow_fallback({slow_res.provider})"
            slow_res.metadata["cascaded_to_slow"] = True
            slow_res.metadata["cascade_reason"] = "unsupported_taxonomy"
            return slow_res

        fast_res = self.fast_service.parse(text, words_meta=words_meta)
        fast_meta = fast_res.metadata or {}
        margin = float(fast_meta.get("margin", 1.0))
        is_confident = (
            fast_res.confidence >= self.confidence_threshold
            and margin >= self.margin_threshold
            and not (self.escalate_on_oos and fast_res.intent == "oos")
        )
        if is_confident:
            fast_res.latency_sec = round(time.time() - start_time, 4)
            fast_res.provider = "hybrid_fast_distilbert"
            fast_meta["cascaded_to_slow"] = False
            fast_meta["fast_confidence"] = fast_res.confidence
            fast_meta["fast_margin"] = margin
            fast_res.metadata = fast_meta
            return fast_res

        slow_res = (
            self.slow_service.parse_with_asr_context(text, words_meta=words_meta, conf_cutoff=conf_cutoff)
            if words_meta and hasattr(self.slow_service, "parse_with_asr_context")
            else self.slow_service.parse(text)
        )
        slow_res.latency_sec = round(time.time() - start_time, 4)
        slow_res.provider = f"hybrid_slow({slow_res.provider})"
        slow_res.metadata["cascaded_to_slow"] = True
        slow_res.metadata["fast_confidence"] = fast_res.confidence
        slow_res.metadata["fast_margin"] = margin
        slow_res.metadata["fast_intent"] = fast_res.intent
        return slow_res

