"""
Central LLM Gateway and Runtime Provider Switcher.
Orchestrates Gemini, OpenAI, Ollama, Local Endpoints, and Deterministic Fallback.
Adheres strictly to SOLID: Single Responsibility, Open/Closed, Dependency Inversion.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from services.llm.llm_contracts import (
    ILanguageModelProvider, LLMMessage, LLMResponse, LLMHealthStatus
)
from services.llm.providers import (
    GeminiLLMProvider,
    OpenAILLMProvider,
    OllamaLLMProvider,
    LocalOpenAICompatibleProvider,
    RuleBasedLLMProvider
)

logger = logging.getLogger("LLMGateway")


class LLMGateway:
    """
    Central Gateway & Orchestrator for all LLM interactions in the pipeline.
    Supports dynamic hot-swapping between Cloud APIs and Local instances with auto-fallback.
    """

    def __init__(self, bot_name: str = "Mira"):
        self.bot_name = bot_name
        self.fallback_provider = RuleBasedLLMProvider(bot_name=bot_name)

        # Initialize concrete providers
        self.providers: Dict[str, ILanguageModelProvider] = {
            "gemini": GeminiLLMProvider(),
            "openai": OpenAILLMProvider(),
            "ollama": OllamaLLMProvider(),
            "local_openai": LocalOpenAICompatibleProvider(),
            "rule_based": self.fallback_provider
        }

        # Auto-detect default provider based on environment and local models
        ollama_health = self.providers["ollama"].check_health()

        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            self.active_provider_id: str = "gemini"
            self.active_model: str = "gemini-1.5-flash"
        elif os.getenv("OPENAI_API_KEY"):
            self.active_provider_id: str = "openai"
            self.active_model: str = "gpt-4o-mini"
        elif ollama_health.is_available and ollama_health.available_models:
            # Active local Ollama detected on machine!
            self.active_provider_id = "ollama"
            self.active_model = ollama_health.available_models[0]
        else:
            # Safe default fallback
            self.active_provider_id = "rule_based"
            self.active_model = "deterministic-v2"

        self.temperature: float = 0.7
        self.max_tokens: int = 1024
        self.fallback_enabled: bool = True
        self.last_health_cache: Dict[str, Any] = {}
        self.last_health_timestamp: float = 0.0

    def get_active_provider(self) -> ILanguageModelProvider:
        return self.providers.get(self.active_provider_id, self.fallback_provider)

    def configure(
        self,
        provider_id: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        fallback_enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Dynamically reconfigures the active provider and parameters at runtime."""
        if provider_id:
            if provider_id not in self.providers:
                raise ValueError(f"Unknown provider '{provider_id}'. Available: {list(self.providers.keys())}")
            self.active_provider_id = provider_id

        target_provider = self.providers[self.active_provider_id]

        # Update API key if provided
        if api_key is not None and hasattr(target_provider, "set_api_key"):
            target_provider.set_api_key(api_key)

        # Update base URL if provided
        if base_url is not None and hasattr(target_provider, "set_base_url"):
            target_provider.set_base_url(base_url)

        # Update model
        if model:
            self.active_model = model
        else:
            available = target_provider.list_models()
            if self.active_model not in available and available:
                self.active_model = available[0]

        if temperature is not None:
            self.temperature = max(0.0, min(1.5, temperature))

        if fallback_enabled is not None:
            self.fallback_enabled = fallback_enabled

        # Invalidate health cache
        self.last_health_timestamp = 0.0

        return self.get_status()

    def get_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Returns diagnostic health and metadata across all 5 providers."""
        now = time.time()
        if not force_refresh and (now - self.last_health_timestamp < 15.0) and self.last_health_cache:
            status = dict(self.last_health_cache)
            status["active_provider"] = self.active_provider_id
            status["active_model"] = self.active_model
            status["temperature"] = self.temperature
            status["fallback_enabled"] = self.fallback_enabled
            return status

        diagnostics = []
        for pid, provider in self.providers.items():
            try:
                health = provider.check_health()
                diagnostics.append(health.model_dump())
            except Exception as e:
                diagnostics.append({
                    "provider_id": pid,
                    "provider_name": provider.get_display_name(),
                    "is_available": False,
                    "status_message": f"Health check failed: {str(e)}",
                    "available_models": provider.list_models(),
                    "requires_api_key": pid in ["gemini", "openai"],
                    "has_api_key": False,
                    "endpoint_url": None
                })

        active_provider = self.get_active_provider()
        res = {
            "active_provider": self.active_provider_id,
            "active_provider_name": active_provider.get_display_name(),
            "active_model": self.active_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "fallback_enabled": self.fallback_enabled,
            "fallback_provider": "rule_based",
            "providers": diagnostics
        }

        self.last_health_cache = res
        self.last_health_timestamp = now
        return res

    def generate(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Executes generation through the active provider.
        Automatically and gracefully cascades to the deterministic fallback if an error occurs.
        """
        primary_provider = self.get_active_provider()
        chosen_model = model or self.active_model
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            return primary_provider.generate(
                messages=messages,
                model=chosen_model,
                temperature=temp,
                max_tokens=tokens,
                system_prompt=system_prompt,
                json_mode=json_mode,
                **kwargs
            )
        except Exception as e:
            logger.warning(
                f"[LLMGateway] Primary provider '{self.active_provider_id}' failed: {str(e)}. "
                f"Fallback enabled: {self.fallback_enabled}"
            )

            if not self.fallback_enabled or self.active_provider_id == "rule_based":
                raise e

            # Graceful cascade to Fallback Engine
            fb_res = self.fallback_provider.generate(
                messages=messages,
                model="deterministic-v2",
                temperature=temp,
                max_tokens=tokens,
                system_prompt=system_prompt,
                json_mode=json_mode,
                **kwargs
            )
            fb_res.fallback_used = True
            fb_res.fallback_reason = f"Primary provider '{self.active_provider_id}' failed ({str(e)}). Switched to offline deterministic engine."
            return fb_res

    def test_completion(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Convenience method to test prompt execution and benchmark latency."""
        msg = LLMMessage(role="user", content=prompt)
        res = self.generate(messages=[msg], model=model)
        return res.model_dump()


# Central Singleton Gateway Instance
llm_gateway = LLMGateway()
