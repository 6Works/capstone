"""
Stage 3: Natural Language Processing (NLP) Microservices.
Extracts intents, entities, sentiment, and orchestrates tool definitions.
Implements INLPService following SOLID.
"""

import json
import re
import time
from typing import Dict, Any, List, Optional
from ..contracts import INLPService, NLPResult
from services.llm.gateway import llm_gateway
from services.llm.llm_contracts import LLMMessage


class RuleBasedNLPService(INLPService):
    """Deterministic, lightweight NLU engine with regex intent & entity extraction."""

    def __init__(self):
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
        cleaned = text.strip()
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
            provider="rule_based_nlp"
        )


class LLMNLPService(INLPService):
    """
    Advanced LLM-backed NLP Service orchestrated via LLMGateway.
    Pluggable across Google Gemini, OpenAI, and local Ollama with JSON schema and tool calling.
    Automatically cascades to RuleBasedNLPService if LLM is offline.
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

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.fallback = RuleBasedNLPService()

    def parse(self, text: str) -> NLPResult:
        start_time = time.time()
        cleaned = text.strip()
        if not cleaned:
            return self.fallback.parse(text)

        messages = [LLMMessage(role="user", content=cleaned)]
        try:
            llm_res = llm_gateway.generate(
                messages=messages,
                model=self.model_name,
                system_prompt=self.SYSTEM_PROMPT,
                json_mode=True
            )

            # Parse JSON from response content
            raw_text = llm_res.content.strip()
            # Clean possible markdown fences
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)
            elapsed = round(time.time() - start_time, 4)

            return NLPResult(
                cleaned_text=parsed.get("cleaned_text", cleaned),
                intent=parsed.get("intent", "general_query"),
                confidence=float(parsed.get("confidence", 0.9)),
                entities=parsed.get("entities", {}),
                sentiment=parsed.get("sentiment", "neutral"),
                tool_calls=parsed.get("tool_calls", []),
                latency_sec=elapsed,
                provider=f"llm_{llm_res.provider}_{llm_res.model}" + (" (fallback)" if llm_res.fallback_used else "")
            )

        except Exception as e:
            # Fallback on any JSON decoding or model error
            res = self.fallback.parse(text)
            res.provider = f"rule_based_fallback ({str(e)[:30]})"
            res.latency_sec = round(time.time() - start_time, 4)
            return res
