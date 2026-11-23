"""
Service Contracts and Data Transfer Objects (DTOs).
Defines standard interfaces (ABCs) and strongly typed Pydantic models for microservices.
Following SOLID: Dependency Inversion, Interface Segregation, and Liskov Substitution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 1. Standard Data Transfer Objects (DTOs)
# ============================================================================

class AudioPayload(BaseModel):
    """Payload representing an acoustic audio signal."""
    source_type: str = Field(description="microphone, file_upload, stream_buffer, synthetic")
    format: str = Field(default="wav", description="Audio format (wav, mp3, flac)")
    sample_rate: int = Field(default=16000, description="Sampling rate in Hz")
    channels: int = Field(default=1, description="Audio channels (1=mono, 2=stereo)")
    duration_sec: float = Field(default=0.0, description="Estimated duration in seconds")
    file_path: Optional[str] = Field(default=None, description="Path to audio file on disk")
    raw_bytes: Optional[bytes] = Field(default=None, description="Raw audio byte buffer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ASRResult(BaseModel):
    """Result produced by Automated Speech Recognition (Stage 2)."""
    transcript: str = Field(default="", description="Transcribed text")
    confidence: float = Field(default=1.0, description="Confidence score [0.0 - 1.0]")
    language: str = Field(default="en-US", description="Detected or configured language")
    latency_sec: float = Field(default=0.0, description="Processing latency")
    provider: str = Field(description="Name of the ASR engine (google_speech, whisper, etc.)")
    status: str = Field(default="success", description="success, unintelligible, error")
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Detailed metrics, segments, and audio info")


class NLPResult(BaseModel):
    """Result produced by Natural Language Processing (Stage 3)."""
    cleaned_text: str = Field(description="Sanitized input query")
    intent: str = Field(description="Primary classified intent")
    confidence: float = Field(default=1.0, description="Classification confidence")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    sentiment: str = Field(default="neutral", description="positive, neutral, negative")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Requested tool invocations")
    latency_sec: float = Field(default=0.0, description="Processing latency")
    provider: str = Field(description="Name of NLP engine (rule_based, llm_gemini, etc.)")


class DialogueTurnResult(BaseModel):
    """Result produced by Dialogue Management (Stage 4)."""
    response_text: str = Field(description="Generated dialogue reply")
    dialogue_action: str = Field(description="Selected dialogue action (greet, answer, clarify, etc.)")
    state: str = Field(description="Current conversation state machine state")
    turn: int = Field(description="Dialogue turn counter")
    memory_context: Dict[str, Any] = Field(default_factory=dict, description="Active context snapshot")
    latency_sec: float = Field(default=0.0, description="Processing latency")
    provider: str = Field(description="Name of Dialogue Manager (state_machine, vector_memory, etc.)")


class TTSResult(BaseModel):
    """Result produced by Text-to-Speech (Stage 5)."""
    file_path: str = Field(description="Path to synthesized audio file")
    audio_format: str = Field(default="wav", description="Audio format (wav, mp3)")
    latency_sec: float = Field(default=0.0, description="Synthesis latency")
    provider: str = Field(description="TTS Engine (pyttsx3, elevenlabs, gtts, etc.)")
    speech_rate: int = Field(default=175, description="Speaking rate")
    status: str = Field(default="success", description="success, skipped, error")
    error: Optional[str] = None


class PlaybackResult(BaseModel):
    """Result produced by Audio Output (Stage 6)."""
    status: str = Field(description="completed, skipped, error")
    playback_time_sec: float = Field(default=0.0, description="Duration audio played")
    device: str = Field(default="default_speaker", description="Output audio sink")
    provider: str = Field(description="Audio Output Service (winsound, browser_audio, sounddevice)")
    error: Optional[str] = None


# ============================================================================
# 2. Abstract Service Interfaces (Contracts)
# ============================================================================

class IAudioInputService(ABC):
    """Stage 1 Contract: Captures audio input."""
    @abstractmethod
    def capture(self, **kwargs) -> AudioPayload:
        pass


class IASRService(ABC):
    """Stage 2 Contract: Converts audio payload into textual transcript."""
    @abstractmethod
    def transcribe(self, payload: AudioPayload) -> ASRResult:
        pass


class INLPService(ABC):
    """Stage 3 Contract: Understands transcript, extracts intent, entities, & tools."""
    @abstractmethod
    def parse(self, text: str) -> NLPResult:
        pass


class IDialogueManagerService(ABC):
    """Stage 4 Contract: Manages dialogue state, memory, and determines response."""
    @abstractmethod
    def process_turn(self, nlp_result: NLPResult) -> DialogueTurnResult:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class ITTSService(ABC):
    """Stage 5 Contract: Synthesizes text response into audio waveform."""
    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        pass


class IAudioOutputService(ABC):
    """Stage 6 Contract: Delivers synthesized audio to the user."""
    @abstractmethod
    def play(self, audio_file_path: str, block: bool = True) -> PlaybackResult:
        pass
