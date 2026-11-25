"""
Stage 3: Natural Language Processing (NLP) Microservices.
Extracts intents, entities, sentiment, and orchestrates tool definitions.
Implements INLPService following SOLID.
"""

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
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.allowed_intents = list(allowed_intents) if allowed_intents else None
        self.repair_lexicon = list(repair_lexicon) if repair_lexicon else None
        self.enable_schema_repair = enable_schema_repair
        self.enable_repair_retry = enable_repair_retry
        self.use_json_mode = use_json_mode
        self.temperature = temperature
        self.seed = seed
        self.fallback = RuleBasedNLPService(repair_lexicon=self.repair_lexicon)

    def _build_system_prompt(self) -> str:
        if not self.allowed_intents:
            return self.SYSTEM_PROMPT
        opts = ", ".join(self.allowed_intents)
        return (
            "You are a strict intent-classification NLU microservice. "
            "Classify the user utterance into EXACTLY ONE intent label from this closed list:\n"
            f"[{opts}]\n"
            'Return ONLY a valid JSON object: {"intent": "<exact_label_from_list>", "confidence": 0.95}. '
            "Do not invent labels outside the list or add prose."
        )

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

    def parse(self, text: str) -> NLPResult:
        start_time = time.time()
        cleaned, asr_fixes = preprocess_asr_transcript(text, self.repair_lexicon)
        if not cleaned:
            return self.fallback.parse(text)

        messages = [LLMMessage(role="user", content=cleaned)]
        gen_kwargs: Dict[str, Any] = {}
        if self.seed is not None:
            gen_kwargs["seed"] = self.seed

        try:
            llm_res = llm_gateway.generate(
                messages=messages,
                model=self.model_name,
                system_prompt=self._build_system_prompt(),
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
