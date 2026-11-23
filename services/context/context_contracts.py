"""
Data Transfer Objects & Contracts for Interview Session Context Management.
Following SOLID: Interface Segregation and Open/Closed.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time


class TurnRecord(BaseModel):
    """Represents a single conversational turn in the interview session."""
    turn_index: int
    candidate_utterance: str
    bot_response: Optional[str] = None
    stage_name: str
    scores: Dict[str, int] = Field(default_factory=dict)
    key_claims: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class WorkspacePluginState(BaseModel):
    """Represents the live state snapshot of an active capability module."""
    plugin_id: str
    last_updated: float = Field(default_factory=time.time)
    summary: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


class SessionContextSnapshot(BaseModel):
    """Full serialized snapshot of active interview context."""
    session_id: str = "session-ai-702"
    role_id: str = "interviewer"
    mode: str = "solo"  # solo or copilot
    blueprint_id: str = "sde-dsa-v1"
    seniority: str = "Senior Staff"
    current_stage_name: str = "Stage 1: Background & Architecture"
    current_stage_index: int = 0
    total_turns: int = 0
    turns: List[TurnRecord] = Field(default_factory=list)
    workspace_states: Dict[str, Any] = Field(default_factory=dict)
    competencies_radar: Dict[str, float] = Field(default_factory=lambda: {
        "Technical Depth": 7.5,
        "Communication Clarity": 8.0,
        "System Architecture": 7.5,
        "Problem Solving": 7.0
    })
    discovered_claims: List[str] = Field(default_factory=list)
    pending_follow_ups: List[str] = Field(default_factory=list)
    active_code_snippet: Optional[str] = None
    last_terminal_output: Optional[str] = None


class ContextUpdateRequest(BaseModel):
    """Payload to update session context manually or via events."""
    plugin_id: Optional[str] = None
    workspace_data: Optional[Dict[str, Any]] = None
    candidate_claim: Optional[str] = None
    note: Optional[str] = None
