"""Flask-AppBuilder views and Pydantic schema re-exports for Organizational Management."""
from __future__ import annotations

from typing import Any

# Re-export Pydantic schemas
from .models import (
	ORGUnitCreate,
	ORGUnitUpdate,
	ORGUnitResponse,
	ORGPositionCreate,
	ORGPositionUpdate,
	ORGPositionResponse,
	ORGReportingLineCreate,
	ORGReportingLineResponse,
	ORGRestructuringCreate,
	ORGRestructuringUpdate,
	ORGRestructuringResponse,
	ORGFilter,
	ORGAuditEvent,
)

__all__ = [
	"ORGUnitCreate",
	"ORGUnitUpdate",
	"ORGUnitResponse",
	"ORGPositionCreate",
	"ORGPositionUpdate",
	"ORGPositionResponse",
	"ORGReportingLineCreate",
	"ORGReportingLineResponse",
	"ORGRestructuringCreate",
	"ORGRestructuringUpdate",
	"ORGRestructuringResponse",
	"ORGFilter",
	"ORGAuditEvent",
]


def serialize_record(record: dict[str, Any]) -> dict[str, Any]:
	return {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in record.items()}
