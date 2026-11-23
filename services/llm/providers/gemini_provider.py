"""
Google Gemini API LLM Provider.
Implements ILanguageModelProvider for Google's Gemini models (Gemini 1.5 Flash, 1.5 Pro, 2.0 Flash).
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
from services.llm.llm_contracts import (
    ILanguageModelProvider, LLMMessage, LLMResponse, LLMHealthStatus
)


class GeminiLLMProvider(ILanguageModelProvider):
    """Google Gemini API Provider via standard REST endpoint."""

    DEFAULT_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash"
    ]

    def __init__(self, api_key: Optional[str] = None, default_model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.default_model = default_model

    def get_provider_id(self) -> str:
        return "gemini"

    def get_display_name(self) -> str:
        return "Google Gemini API"

    def list_models(self) -> List[str]:
        return self.DEFAULT_MODELS

    def set_api_key(self, key: str) -> None:
        self.api_key = key.strip()

    def check_health(self) -> LLMHealthStatus:
        has_key = bool(self.api_key and len(self.api_key) > 5)
        if not has_key:
            return LLMHealthStatus(
                provider_id="gemini",
                provider_name="Google Gemini API",
                is_available=False,
                status_message="GEMINI_API_KEY is not configured",
                available_models=self.DEFAULT_MODELS,
                requires_api_key=True,
                has_api_key=False,
                endpoint_url="https://generativelanguage.googleapis.com/v1beta"
            )

        # Lightweight model probe
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            resp = requests.get(url, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    m["name"].replace("models/", "")
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                return LLMHealthStatus(
                    provider_id="gemini",
                    provider_name="Google Gemini API",
                    is_available=True,
                    status_message="Connected and authenticated successfully",
                    available_models=models[:8] if models else self.DEFAULT_MODELS,
                    requires_api_key=True,
                    has_api_key=True,
                    endpoint_url="https://generativelanguage.googleapis.com/v1beta"
                )
            else:
                err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                return LLMHealthStatus(
                    provider_id="gemini",
                    provider_name="Google Gemini API",
                    is_available=False,
                    status_message=f"Authentication error: {err_msg}",
                    available_models=self.DEFAULT_MODELS,
                    requires_api_key=True,
                    has_api_key=True,
                    endpoint_url="https://generativelanguage.googleapis.com/v1beta"
                )
        except Exception as e:
            return LLMHealthStatus(
                provider_id="gemini",
                provider_name="Google Gemini API",
                is_available=False,
                status_message=f"Network probe error: {str(e)}",
                available_models=self.DEFAULT_MODELS,
                requires_api_key=True,
                has_api_key=True,
                endpoint_url="https://generativelanguage.googleapis.com/v1beta"
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
            raise ValueError("GEMINI_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent?key={self.api_key}"

        # Extract system prompt from messages or argument
        extracted_sys = system_prompt or ""
        contents = []

        for m in messages:
            if m.role == "system":
                extracted_sys += ("\n" + m.content) if extracted_sys else m.content
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": m.content}]
                })

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        if extracted_sys:
            payload["systemInstruction"] = {
                "parts": [{"text": extracted_sys}]
            }

        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=payload, timeout=20.0)

        if resp.status_code != 200:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_text = err_data.get("error", {}).get("message", resp.text)
            raise RuntimeError(f"Gemini API Error [{resp.status_code}]: {err_text}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned empty candidate list.")

        text_content = ""
        parts = candidates[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                text_content += p["text"]

        usage = data.get("usageMetadata", {})
        latency = round(time.time() - start_time, 3)

        return LLMResponse(
            content=text_content,
            provider="gemini",
            model=active_model,
            latency_sec=latency,
            tokens_prompt=usage.get("promptTokenCount", 0),
            tokens_completion=usage.get("candidatesTokenCount", 0),
            fallback_used=False,
            raw_metadata=usage
        )
