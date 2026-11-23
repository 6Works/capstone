"""
Role Blueprints and Three-Layer Plug-and-Play Interview Architecture.
"""

from .blueprint_contracts import (
    RuntimeTier,
    PluginType,
    PhaseLayout,
    InterviewPhase,
    RubricCriterion,
    EvaluationRubric,
    RuntimeConfig,
    RoleBlueprint,
    TelemetryEvent,
    ArtifactBundle
)
from .blueprint_service import BlueprintService, blueprint_service

__all__ = [
    "RuntimeTier",
    "PluginType",
    "PhaseLayout",
    "InterviewPhase",
    "RubricCriterion",
    "EvaluationRubric",
    "RuntimeConfig",
    "RoleBlueprint",
    "TelemetryEvent",
    "ArtifactBundle",
    "BlueprintService",
    "blueprint_service"
]
