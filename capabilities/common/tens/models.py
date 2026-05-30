"""Tenants Legacy data models."""

from __future__ import annotations

from .tenant_runtime import (
	AccessBoundaryRecord,
	DeprecationPlanRecord,
	LegacyTenantRecord,
	MigrationPlanRecord,
	TensAgentRecord,
	TenantAuditEventRecord,
	TenantMappingRecord,
)


TensRecord = LegacyTenantRecord


__all__ = [
	"AccessBoundaryRecord",
	"DeprecationPlanRecord",
	"LegacyTenantRecord",
	"MigrationPlanRecord",
	"TensAgentRecord",
	"TenantAuditEventRecord",
	"TenantMappingRecord",
	"TensRecord",
]
