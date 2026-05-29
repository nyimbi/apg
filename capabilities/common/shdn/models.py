"""Shutdown and Lifecycle Control data models."""

from __future__ import annotations

from .lifecycle_runtime import (
	BackupSnapshotRecord,
	DrainOperationRecord,
	LifecycleAuditEventRecord,
	RecoveryRecord,
	ShutdownExecutionRecord,
	ShutdownPlanRecord,
	ShutdownTargetRecord,
)


ShdnRecord = ShutdownTargetRecord


__all__ = [
	"BackupSnapshotRecord",
	"DrainOperationRecord",
	"LifecycleAuditEventRecord",
	"RecoveryRecord",
	"ShdnRecord",
	"ShutdownExecutionRecord",
	"ShutdownPlanRecord",
	"ShutdownTargetRecord",
]
