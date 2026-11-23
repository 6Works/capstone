"""
Local Ollama LLM Provider.
Implements ILanguageModelProvider for local models running via Ollama (http://127.0.0.1:11434).
Supports automatic dynamic discovery of installed local models and resilient IPv4 fallback.
"""

import time
import requests
from typing import Dict, Any, List, Optional
from services.llm.llm_contracts import (
    ILanguageModelProvider, LLMMessage, LLMResponse, LLMHealthStatus
)


class OllamaLLMProvider(ILanguageModelProvider):
    """Local Ollama Provider communicating via http://127.0.0.1:11434."""

    DEFAULT_FALLBACK_MODELS = [
        "llama3.2:latest",
        "llama3.2:3b",
        "llama3.1:latest",
        "qwen2.5:3b",
        "qwen2.5-coder:1.5b"
    ]

    def __init__(self, base_url: str = "http://127.0.0.1:11434", default_model: str = "llama3.2:latest"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.discovered_models: List[str] = []
        # Attempt immediate local discovery
        self._probe_models()

    def get_provider_id(self) -> str:
        return "ollama"

    def get_display_name(self) -> str:
        return "Local Ollama Instance"

    def set_base_url(self, url: str) -> None:
        clean = url.rstrip("/")
        # Windows IPv4 safeguard: if user supplies localhost, prefer 127.0.0.1 to avoid ::1 WinError 10061
        if "localhost" in clean:
            clean = clean.replace("localhost", "127.0.0.1")
        self.base_url = clean
        self._probe_models()

    def _get_candidate_urls(self) -> List[str]:
        """Returns list of candidate URLs trying IPv4 127.0.0.1 first, then host string."""
        candidates = [self.base_url]
        if "localhost" in self.base_url:
            candidates.insert(0, self.base_url.replace("localhost", "127.0.0.1"))
        elif "127.0.0.1" in self.base_url:
            candidates.append(self.base_url.replace("127.0.0.1", "localhost"))
        return list(dict.fromkeys(candidates))

    def _probe_models(self) -> List[str]:
        for url in self._get_candidate_urls():
            try:
                resp = requests.get(f"{url}/api/tags", timeout=1.5)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if models:
                        self.discovered_models = models
                        self.base_url = url
                        if self.default_model not in models:
                            self.default_model = models[0]
                        return models
            except Exception:
                continue
        return self.discovered_models

    def list_models(self) -> List[str]:
        if not self.discovered_models:
            self._probe_models()
        return self.discovered_models if self.discovered_models else self.DEFAULT_FALLBACK_MODELS

    def check_health(self) -> LLMHealthStatus:
        models = self._probe_models()
        if models:
            return LLMHealthStatus(
                provider_id="ollama",
                provider_name="Local Ollama Instance",
                is_available=True,
                status_message=f"Ollama is running with {len(models)} local model(s) installed",
                available_models=models,
                requires_api_key=False,
                has_api_key=True,
                endpoint_url=self.base_url
            )

        # Probe failed
        return LLMHealthStatus(
            provider_id="ollama",
            provider_name="Local Ollama Instance",
            is_available=False,
            status_message=f"Ollama daemon not reachable at {self.base_url}. Start Ollama or check port 11434.",
            available_models=self.DEFAULT_FALLBACK_MODELS,
            requires_api_key=False,
            has_api_key=True,
            endpoint_url=self.base_url
        )

    def _resolve_model_name(self, requested_model: Optional[str]) -> str:
        """Resolves model name against discovered models to prevent 'model not found' errors."""
        available = self.list_models()
        if not available:
            return requested_model or self.default_model

        if requested_model and requested_model in available:
            return requested_model

        # Substring / family matching (e.g. 'llama3' -> 'llama3.2:latest' or 'llama3.2:3b')
        if requested_model:
            req_clean = requested_model.lower().split(":")[0]
            for m in available:
                if req_clean in m.lower():
                    return m

        return available[0]

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
        active_model = self._resolve_model_name(model)

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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if json_mode:
            payload["format"] = "json"

        last_err = None
        for base in self._get_candidate_urls():
            url = f"{base}/api/chat"
            try:
                resp = requests.post(url, json=payload, timeout=30.0)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "")
                    latency = round(time.time() - start_time, 3)

                    prompt_eval = data.get("prompt_eval_count", 0)
                    eval_count = data.get("eval_count", 0)

                    self.base_url = base
                    return LLMResponse(
                        content=content,
                        provider="ollama",
                        model=active_model,
                        latency_sec=latency,
                        tokens_prompt=prompt_eval,
                        tokens_completion=eval_count,
                        fallback_used=False,
                        raw_metadata={"total_duration": data.get("total_duration", 0)}
                    )
                else:
                    last_err = RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                last_err = e
                continue

        raise ConnectionError(f"Failed to communicate with Ollama at {self.base_url}: {str(last_err)}")
