"""Flask-AppBuilder views and Pydantic schema re-exports for Professional Development."""
from __future__ import annotations

from typing import Any

from .models import (
	PRODevelopmentPlanCreate,
	PRODevelopmentPlanUpdate,
	PRODevelopmentPlanResponse,
	PROSkillCreate,
	PROSkillAssessmentCreate,
	PROSkillAssessmentResponse,
	PROMentoringProgrammeCreate,
	PROMentoringProgrammeUpdate,
	PROMentoringProgrammeResponse,
	PROCertificationCreate,
	PROCertificationUpdate,
	PROCertificationResponse,
	PROCareerPathCreate,
	PROCareerPathUpdate,
	PROCareerPathResponse,
	PROFilter,
	PROAuditEvent,
)

__all__ = [
	"PRODevelopmentPlanCreate",
	"PRODevelopmentPlanUpdate",
	"PRODevelopmentPlanResponse",
	"PROSkillCreate",
	"PROSkillAssessmentCreate",
	"PROSkillAssessmentResponse",
	"PROMentoringProgrammeCreate",
	"PROMentoringProgrammeUpdate",
	"PROMentoringProgrammeResponse",
	"PROCertificationCreate",
	"PROCertificationUpdate",
	"PROCertificationResponse",
	"PROCareerPathCreate",
	"PROCareerPathUpdate",
	"PROCareerPathResponse",
	"PROFilter",
	"PROAuditEvent",
]


def serialize_record(record: dict[str, Any]) -> dict[str, Any]:
	return {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in record.items()}
