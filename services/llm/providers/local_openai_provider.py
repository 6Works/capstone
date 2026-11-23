"""
Local OpenAI-Compatible LLM Provider.
Implements ILanguageModelProvider for local inferencing servers exposing /v1/chat/completions
(e.g., LM Studio, vLLM, LocalAI, Jan, text-generation-webui).
"""

import time
import requests
from typing import Dict, Any, List, Optional
from services.llm.llm_contracts import (
    ILanguageModelProvider, LLMMessage, LLMResponse, LLMHealthStatus
)


class LocalOpenAICompatibleProvider(ILanguageModelProvider):
    """Local OpenAI-Compatible Server Provider (e.g. LM Studio on port 1234)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        default_model: str = "local-model",
        api_key: str = "not-needed"
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key
        self.discovered_models: List[str] = []

    def get_provider_id(self) -> str:
        return "local_openai"

    def get_display_name(self) -> str:
        return "Local OpenAI Server (LM Studio / vLLM)"

    def set_base_url(self, url: str) -> None:
        self.base_url = url.rstrip("/")

    def list_models(self) -> List[str]:
        if self.discovered_models:
            return self.discovered_models
        return [self.default_model]

    def check_health(self) -> LLMHealthStatus:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(f"{self.base_url}/models", headers=headers, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                self.discovered_models = models
                return LLMHealthStatus(
                    provider_id="local_openai",
                    provider_name="Local OpenAI Server (LM Studio / vLLM)",
                    is_available=True,
                    status_message=f"Connected to local endpoint ({len(models)} model(s) available)",
                    available_models=models if models else [self.default_model],
                    requires_api_key=False,
                    has_api_key=True,
                    endpoint_url=self.base_url
                )
            else:
                return LLMHealthStatus(
                    provider_id="local_openai",
                    provider_name="Local OpenAI Server (LM Studio / vLLM)",
                    is_available=False,
                    status_message=f"Server returned HTTP status {resp.status_code}",
                    available_models=[self.default_model],
                    requires_api_key=False,
                    has_api_key=True,
                    endpoint_url=self.base_url
                )
        except Exception as e:
            return LLMHealthStatus(
                provider_id="local_openai",
                provider_name="Local OpenAI Server (LM Studio / vLLM)",
                is_available=False,
                status_message=f"Local server not reachable at {self.base_url} ({type(e).__name__})",
                available_models=[self.default_model],
                requires_api_key=False,
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
        active_model = model or (self.discovered_models[0] if self.discovered_models else self.default_model)

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

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=35.0)
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with Local OpenAI server at {url}: {str(e)}")

        if resp.status_code != 200:
            raise RuntimeError(f"Local server error [{resp.status_code}]: {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Local server returned empty choices array.")

        text_content = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        latency = round(time.time() - start_time, 3)

        return LLMResponse(
            content=text_content,
            provider="local_openai",
            model=active_model,
            latency_sec=latency,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_completion=usage.get("completion_tokens", 0),
            fallback_used=False,
            raw_metadata=usage
        )
