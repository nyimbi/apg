"""Flask-AppBuilder compatible views and Pydantic schema re-exports for obs_mtx."""
from __future__ import annotations

# Re-export all Pydantic models for external consumers
from .models import (
	MetricDefinitionCreate,
	MetricDefinitionUpdate,
	MetricDefinitionResponse,
	MetricDataPointCreate,
	MetricDataPointResponse,
	SLOCreate,
	SLOUpdate,
	SLOResponse,
	SLOListResponse,
	BurnRateAlertCreate,
	BurnRateAlertResponse,
	PrometheusExportConfig,
	DashboardCreate,
	DashboardResponse,
	MetricFilterModel,
	REDMetricsSummary,
	AuditEventResponse,
)

__all__ = [
	"MetricDefinitionCreate",
	"MetricDefinitionUpdate",
	"MetricDefinitionResponse",
	"MetricDataPointCreate",
	"MetricDataPointResponse",
	"SLOCreate",
	"SLOUpdate",
	"SLOResponse",
	"SLOListResponse",
	"BurnRateAlertCreate",
	"BurnRateAlertResponse",
	"PrometheusExportConfig",
	"DashboardCreate",
	"DashboardResponse",
	"MetricFilterModel",
	"REDMetricsSummary",
	"AuditEventResponse",
]
