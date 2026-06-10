"""Flask-AppBuilder views and Pydantic schema re-exports for Succession Planning."""
from __future__ import annotations

from typing import Any

from .models import (
	SCPTalentPoolCreate,
	SCPTalentPoolUpdate,
	SCPTalentPoolResponse,
	SCPReadinessAssessmentCreate,
	SCPReadinessAssessmentUpdate,
	SCPReadinessAssessmentResponse,
	SCPNineBoxEntryCreate,
	SCPNineBoxEntryResponse,
	SCPSuccessionScenarioCreate,
	SCPSuccessionScenarioUpdate,
	SCPSuccessionScenarioResponse,
	SCPCriticalRoleCreate,
	SCPCriticalRoleUpdate,
	SCPCriticalRoleResponse,
	SCPFilter,
	SCPAuditEvent,
)

__all__ = [
	"SCPTalentPoolCreate",
	"SCPTalentPoolUpdate",
	"SCPTalentPoolResponse",
	"SCPReadinessAssessmentCreate",
	"SCPReadinessAssessmentUpdate",
	"SCPReadinessAssessmentResponse",
	"SCPNineBoxEntryCreate",
	"SCPNineBoxEntryResponse",
	"SCPSuccessionScenarioCreate",
	"SCPSuccessionScenarioUpdate",
	"SCPSuccessionScenarioResponse",
	"SCPCriticalRoleCreate",
	"SCPCriticalRoleUpdate",
	"SCPCriticalRoleResponse",
	"SCPFilter",
	"SCPAuditEvent",
]


def serialize_record(record: dict[str, Any]) -> dict[str, Any]:
	return {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in record.items()}
