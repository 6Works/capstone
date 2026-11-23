"""
OpenAI API LLM Provider.
Implements ILanguageModelProvider for OpenAI models (GPT-4o, GPT-4o Mini, GPT-3.5 Turbo).
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
from services.llm.llm_contracts import (
    ILanguageModelProvider, LLMMessage, LLMResponse, LLMHealthStatus
)


class OpenAILLMProvider(ILanguageModelProvider):
    """OpenAI API Provider via standard REST endpoint."""

    DEFAULT_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo"
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.default_model = default_model
        self.base_url = base_url.rstrip("/")

    def get_provider_id(self) -> str:
        return "openai"

    def get_display_name(self) -> str:
        return "OpenAI API"

    def list_models(self) -> List[str]:
        return self.DEFAULT_MODELS

    def set_api_key(self, key: str) -> None:
        self.api_key = key.strip()

    def check_health(self) -> LLMHealthStatus:
        has_key = bool(self.api_key and len(self.api_key) > 5)
        if not has_key:
            return LLMHealthStatus(
                provider_id="openai",
                provider_name="OpenAI API",
                is_available=False,
                status_message="OPENAI_API_KEY is not configured",
                available_models=self.DEFAULT_MODELS,
                requires_api_key=True,
                has_api_key=False,
                endpoint_url=self.base_url
            )

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(f"{self.base_url}/models", headers=headers, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    m["id"] for m in data.get("data", [])
                    if any(prefix in m["id"] for prefix in ["gpt-4", "gpt-3.5", "o1", "o3"])
                ]
                return LLMHealthStatus(
                    provider_id="openai",
                    provider_name="OpenAI API",
                    is_available=True,
                    status_message="Connected and authenticated successfully",
                    available_models=sorted(models[:8]) if models else self.DEFAULT_MODELS,
                    requires_api_key=True,
                    has_api_key=True,
                    endpoint_url=self.base_url
                )
            else:
                err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                return LLMHealthStatus(
                    provider_id="openai",
                    provider_name="OpenAI API",
                    is_available=False,
                    status_message=f"Authentication error: {err_msg}",
                    available_models=self.DEFAULT_MODELS,
                    requires_api_key=True,
                    has_api_key=True,
                    endpoint_url=self.base_url
                )
        except Exception as e:
            return LLMHealthStatus(
                provider_id="openai",
                provider_name="OpenAI API",
                is_available=False,
                status_message=f"Network probe error: {str(e)}",
                available_models=self.DEFAULT_MODELS,
                requires_api_key=True,
                has_api_key=True,
                endpoint_url=self.base_url
            )

    def generate(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        active_model = model or self.default_model

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        url = f"{self.base_url}/chat/completions"

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for m in messages:
            formatted_messages.append({"role": m.role, "content": m.content})

        if not formatted_messages:
            formatted_messages.append({"role": "user", "content": "Hello"})

        payload: Dict[str, Any] = {
            "model": active_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=25.0)

        if resp.status_code != 200:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_text = err_data.get("error", {}).get("message", resp.text)
            raise RuntimeError(f"OpenAI API Error [{resp.status_code}]: {err_text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI returned empty choices array.")

        text_content = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        latency = round(time.time() - start_time, 3)

        return LLMResponse(
            content=text_content,
            provider="openai",
            model=active_model,
            latency_sec=latency,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_completion=usage.get("completion_tokens", 0),
            fallback_used=False,
            raw_metadata=usage
        )
