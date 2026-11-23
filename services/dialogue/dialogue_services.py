"""
Stage 4: Dialogue Management Microservices.
Encapsulates state machine, conversation memory, tool execution, and response policy.
Implements IDialogueManagerService following SOLID.
"""

import time
import datetime
from typing import Dict, Any, List, Optional
from ..contracts import IDialogueManagerService, NLPResult, DialogueTurnResult
from services.llm.gateway import llm_gateway
from services.llm.llm_contracts import LLMMessage


class ConversationMemoryStore:
    """Decoupled memory store with short-term buffers and context memory."""

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.turns: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}

    def append_turn(self, user_text: str, bot_response: str, intent: str, action: str):
        record = {
            "turn_index": len(self.turns) + 1,
            "user_text": user_text,
            "bot_response": bot_response,
            "intent": intent,
            "action": action,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.turns.append(record)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def set_context(self, key: str, value: Any):
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def clear(self):
        self.turns.clear()
        self.context.clear()


class ModularDialogueManager(IDialogueManagerService):
    """
    Decoupled Dialogue Manager combining State Machine, Memory Store, and Policy Engine.
    """

    def __init__(self, bot_name: str = "Mira"):
        self.bot_name = bot_name
        self.state = "INITIAL"
        self.memory = ConversationMemoryStore()
        self.turn_count = 0

    def process_turn(self, nlp_result: NLPResult) -> DialogueTurnResult:
        start_time = time.time()
        self.turn_count += 1
        
        intent = nlp_result.intent
        cleaned = nlp_result.cleaned_text
        entities = nlp_result.entities
        
        # State & policy decisions
        action = "respond"
        reply = ""

        if not cleaned:
            action = "clarification"
            reply = "I didn't quite catch that. Could you please say that again?"

        elif intent == "greeting":
            self.state = "IN_CONVERSATION"
            action = "greet"
            reply = f"Hello! I am {self.bot_name}, your decoupled modular voice assistant. How can I help you today?"

        elif intent == "identity_query":
            self.state = "IN_CONVERSATION"
            action = "inform"
            reply = (
                f"I am {self.bot_name}, built with a decoupled microservice architecture following SOLID principles. "
                "Each stage operates as an independent service with its own interface and zero tight coupling."
            )

        elif intent == "project_query":
            self.state = "IN_CONVERSATION"
            action = "explain"
            reply = (
                "Our architecture follows SOLID: 1. Audio Input captures signal, 2. ASR transcribes speech, "
                "3. NLP extracts intent and entities, 4. Dialogue Management handles state & memory, "
                "5. TTS synthesizes realistic speech, and 6. Audio Output delivers playback. All modules are fully swappable!"
            )

        elif intent == "time_query":
            self.state = "IN_CONVERSATION"
            action = "inform"
            now_str = datetime.datetime.now().strftime("%I:%M %p")
            reply = f"The current time is {now_str}."

        elif intent == "weather_query":
            self.state = "IN_CONVERSATION"
            action = "inform"
            reply = "The weather is currently clear and pleasant, ideal for developing AI systems."

        elif intent == "calculation_query":
            self.state = "IN_CONVERSATION"
            action = "calculate"
            expr = entities.get("math_expression", "")
            if expr:
                try:
                    allowed = set("0123456789+-*/.() ")
                    if all(c in allowed for c in expr):
                        val = eval(expr)
                        reply = f"The answer to {expr} is {val}."
                    else:
                        reply = "I can only calculate basic arithmetic."
                except Exception:
                    reply = f"Could not calculate {expr}."
            else:
                reply = "Please specify the numbers to calculate."

        elif intent == "help_query":
            action = "help"
            reply = "You can ask me the time, request math calculations, learn about our SOLID architecture, or chat freely."

        elif intent == "farewell":
            self.state = "CLOSING"
            action = "farewell"
            reply = "Goodbye! It was a pleasure assisting you."

        else:
            self.state = "IN_CONVERSATION"
            action = "general_reply"
            reply = f"You said: '{cleaned}'. I have stored this in my conversation memory."

        elapsed = round(time.time() - start_time, 4)

        # Update memory store
        self.memory.append_turn(
            user_text=cleaned,
            bot_response=reply,
            intent=intent,
            action=action
        )

        return DialogueTurnResult(
            response_text=reply,
            dialogue_action=action,
            state=self.state,
            turn=self.turn_count,
            memory_context={"total_history_turns": len(self.memory.turns)},
            latency_sec=elapsed,
            provider="modular_dialogue_manager"
        )

    def reset(self) -> None:
        self.state = "INITIAL"
        self.turn_count = 0
        self.memory.clear()


class LLMDialogueManager(IDialogueManagerService):
    """
    Advanced Generative Dialogue Manager driven by LLMGateway.
    Combines state machine tracking, context memory, and deep LLM reasoning.
    """

    def __init__(self, bot_name: str = "Mira", model_name: Optional[str] = None):
        self.bot_name = bot_name
        self.model_name = model_name
        self.state = "INITIAL"
        self.memory = ConversationMemoryStore()
        self.turn_count = 0
        self.fallback = ModularDialogueManager(bot_name=bot_name)

    def process_turn(self, nlp_result: NLPResult) -> DialogueTurnResult:
        start_time = time.time()
        self.turn_count += 1
        cleaned = nlp_result.cleaned_text
        intent = nlp_result.intent

        if not cleaned:
            return self.fallback.process_turn(nlp_result)

        # Update state
        if intent == "farewell":
            self.state = "CLOSING"
        elif self.state == "INITIAL":
            self.state = "IN_CONVERSATION"

        # Build prompt messages including short-term conversation memory
        system_prompt = (
            f"You are {self.bot_name}, a friendly, intelligent voice AI assistant with a decoupled microservice architecture. "
            "Keep your responses natural, conversational, and concise (1-3 sentences), perfectly suited for vocal Text-to-Speech synthesis. "
            "Never use markdown headers, bullets, or asterisks unless specifically requested."
        )

        llm_messages = []
        # Include past 3 turns for context
        for past in self.memory.turns[-3:]:
            llm_messages.append(LLMMessage(role="user", content=past["user_text"]))
            llm_messages.append(LLMMessage(role="assistant", content=past["bot_response"]))

        llm_messages.append(LLMMessage(role="user", content=cleaned))

        try:
            llm_res = llm_gateway.generate(
                messages=llm_messages,
                model=self.model_name,
                system_prompt=system_prompt,
                max_tokens=256
            )
            reply = llm_res.content.strip()
            action = "llm_generate"
            provider_label = f"llm_dialogue_{llm_res.provider}_{llm_res.model}"
            if llm_res.fallback_used:
                provider_label += " (fallback)"

        except Exception as e:
            fallback_res = self.fallback.process_turn(nlp_result)
            reply = fallback_res.response_text
            action = fallback_res.dialogue_action
            provider_label = f"dialogue_fallback ({str(e)[:25]})"

        elapsed = round(time.time() - start_time, 4)

        self.memory.append_turn(
            user_text=cleaned,
            bot_response=reply,
            intent=intent,
            action=action
        )

        return DialogueTurnResult(
            response_text=reply,
            dialogue_action=action,
            state=self.state,
            turn=self.turn_count,
            memory_context={
                "total_history_turns": len(self.memory.turns),
                "intent": intent,
                "sentiment": nlp_result.sentiment
            },
            latency_sec=elapsed,
            provider=provider_label
        )

    def reset(self) -> None:
        self.state = "INITIAL"
        self.turn_count = 0
        self.memory.clear()
        self.fallback.reset()
