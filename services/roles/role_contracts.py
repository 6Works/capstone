"""
Role & Persona Domain Models and Contracts.
Following SOLID: Interface Segregation and Single Responsibility.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RoleMode(BaseModel):
    id: str = Field(description="Unique mode identifier, e.g. 'solo' or 'copilot'")
    name: str = Field(description="Display title, e.g. 'Autonomous Solo Interviewer'")
    description: str = Field(description="Operational summary")
    badge: str = Field(description="UI badge tag, e.g. 'Primary AI' or 'Live Co-Pilot'")
    icon: str = Field(description="FontAwesome icon class")


class RoleProfile(BaseModel):
    id: str = Field(description="Role identifier, e.g. 'interviewer', 'tech_screener'")
    name: str = Field(description="Role display name")
    tagline: str = Field(description="One-line description")
    category: str = Field(description="Category e.g. 'Hiring & Interviews'")
    icon: str = Field(description="FontAwesome icon")
    accent_color: str = Field(description="Hex or tailwind color")
    description: str = Field(description="Detailed overview")
    modes: List[RoleMode] = Field(default_factory=list, description="Supported sub-modes")
    system_prompt: str = Field(description="Persona instruction guidelines")


class InterviewEvaluation(BaseModel):
    overall_score: int = Field(default=8, description="Score 1-10")
    clarity_score: int = Field(default=8, description="Communication clarity 1-10")
    depth_score: int = Field(default=8, description="Technical depth 1-10")
    system_design_score: int = Field(default=8, description="System design 1-10")
    strengths: List[str] = Field(default_factory=list)
    areas_for_growth: List[str] = Field(default_factory=list)
    fact_check_note: Optional[str] = None
    executive_summary: str = Field(default="")


class CopilotHUD(BaseModel):
    suggested_probes: List[str] = Field(default_factory=list)
    fact_check: str = Field(default="")
    competency_scores: Dict[str, int] = Field(default_factory=dict)
    key_claims: List[str] = Field(default_factory=list)
    interviewer_notes: str = Field(default="")
