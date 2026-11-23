"""
LLM Microservice Contracts and Data Transfer Objects (DTOs).
Defines standard interfaces (ABCs) and strongly typed Pydantic models for Language Model Providers.
Strictly following SOLID: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Standard conversational turn message for LLM context."""
    role: str = Field(description="system, user, assistant, or tool")
    content: str = Field(description="Textual content of the message")
    name: Optional[str] = None


class LLMResponse(BaseModel):
    """Unified result produced by any LLM provider."""
    content: str = Field(description="Generated textual response")
    provider: str = Field(description="Name of provider that generated response (gemini, openai, ollama, etc.)")
    model: str = Field(description="Model identifier used")
    latency_sec: float = Field(default=0.0, description="Inference latency in seconds")
    tokens_prompt: int = Field(default=0, description="Estimated or reported input tokens")
    tokens_completion: int = Field(default=0, description="Estimated or reported output tokens")
    fallback_used: bool = Field(default=False, description="Whether fallback was automatically triggered")
    fallback_reason: Optional[str] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw provider metadata")


class LLMHealthStatus(BaseModel):
    """Health check diagnostic result for a provider."""
    provider_id: str
    provider_name: str
    is_available: bool
    status_message: str
    available_models: List[str] = Field(default_factory=list)
    requires_api_key: bool = False
    has_api_key: bool = False
    endpoint_url: Optional[str] = None


class ILanguageModelProvider(ABC):
    """
    Contract for all LLM providers (Cloud APIs and Local Instances).
    Follows LSP: any provider can seamlessly substitute any other provider.
    """

    @abstractmethod
    def get_provider_id(self) -> str:
        """Unique slug identifier for the provider (gemini, openai, ollama, etc.)."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Human-readable provider label."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Returns list of supported or locally installed model IDs."""
        pass

    @abstractmethod
    def check_health(self) -> LLMHealthStatus:
        """Probes connectivity, API keys, and server readiness."""
        pass

    @abstractmethod
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
        """
        Executes chat completion across the provider.
        Raises exception on network/auth failure so LLMGateway can gracefully cascade to fallback.
        """
        pass

    def set_api_key(self, api_key: str) -> None:
        """Optional hook to configure API key at runtime."""
        pass

    def set_base_url(self, base_url: str) -> None:
        """Optional hook to configure endpoint base URL at runtime."""
        pass
