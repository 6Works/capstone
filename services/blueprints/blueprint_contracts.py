"""
Three-Layer Plug-and-Play Interview Architecture Contracts and DTOs.
Adheres strictly to SOLID: Single Responsibility, Open/Closed, and Dependency Inversion.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RuntimeTier(str, Enum):
    TIER_0_ZERO_COMPUTE = "tier_0_zero_compute"       # Browser/WebSocket state (MBA, Case, System Design)
    TIER_1_STATELESS_EXEC = "tier_1_stateless_exec"   # Micro-VM / Judge pool (Core SDE, SQL)
    TIER_2_STATEFUL_SANDBOX = "tier_2_stateful_sandbox" # Isolated container/namespace (ML, Fullstack, Cyber)


class PluginType(str, Enum):
    PROMPT_PANEL = "PromptPanel"
    CASE_BRIEF_READER = "CaseBriefReader"
    CODE_EDITOR = "CodeEditor"
    NOTEBOOK_KERNEL = "NotebookKernel"
    LINUX_TERMINAL = "LinuxTerminal"
    LIVE_WEB_PREVIEW = "LiveWebPreview"
    SPREADSHEET_GRID = "SpreadsheetGrid"
    INFINITE_WHITEBOARD = "InfiniteWhiteboard"
    SQL_WORKBENCH = "SQLWorkbench"
    STRUCTURED_NOTES = "StructuredNotes"
    TEST_RUNNER = "TestRunner"


class PhaseLayout(BaseModel):
    """Layout grid rails for resizable split views."""
    ratio: str = Field(default="50-50", description="50-50, 40-60, 60-40, 30-70, 70-30")
    left: List[str] = Field(default_factory=list, description="Plugin identifiers mounted in left dock")
    right: List[str] = Field(default_factory=list, description="Plugin identifiers mounted in right dock")


class InterviewPhase(BaseModel):
    """Progressive disclosure phase within an interview session."""
    id: str
    name: str
    duration_mins: int
    description: str
    objectives: List[str] = Field(default_factory=list)
    layout: PhaseLayout


class RubricCriterion(BaseModel):
    id: str
    name: str
    weight: float = 1.0  # e.g. 1.0, 1.5, 2.0
    description: str


class EvaluationRubric(BaseModel):
    rubric_id: str
    title: str
    criteria: List[RubricCriterion]


class RuntimeConfig(BaseModel):
    tier: RuntimeTier
    base_image: Optional[str] = None
    warmup_sec: float = 1.2
    compute_description: str


class RoleBlueprint(BaseModel):
    """
    Declarative Role Blueprint manifest defining environment provisioning,
    mounted atomic capability plugins, phase progressive disclosure, and evaluation rubrics.
    """
    blueprint_id: str
    title: str
    domain: str
    seniority: str = "Mid-Senior"
    tagline: str
    icon: str
    accent_color: str
    runtime: RuntimeConfig
    phases: List[InterviewPhase]
    rubric: EvaluationRubric
    all_plugins: List[str] = Field(default_factory=list)
    preset_problem: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: str


class TelemetryEvent(BaseModel):
    """Standardized event emitted by any mounted capability plugin."""
    session_id: str = "default_session"
    plugin_id: str
    event_type: str  # code_diff, cell_run, terminal_exec, formula_update, whiteboard_node, etc.
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default=0.0)


class ArtifactBundle(BaseModel):
    """Exported final state snapshot from all mounted plugins for rubric grading."""
    session_id: str
    blueprint_id: str
    timestamp: float
    plugin_snapshots: Dict[str, Any]
    transcript: List[Dict[str, Any]] = Field(default_factory=list)
