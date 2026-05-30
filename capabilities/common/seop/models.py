"""Security Operations data models."""

from __future__ import annotations

from .ops_runtime import (
	DetectionRecord,
	IncidentRecord,
	OpsAuditEventRecord,
	PlaybookRecord,
	PostureControlRecord,
	ResponseActionRecord,
	SeopAgentRecord,
)


SeopRecord = DetectionRecord


__all__ = [
	"DetectionRecord",
	"IncidentRecord",
	"OpsAuditEventRecord",
	"PlaybookRecord",
	"PostureControlRecord",
	"ResponseActionRecord",
	"SeopAgentRecord",
	"SeopRecord",
]
