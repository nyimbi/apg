"""Pydantic v2 models for APG Report Builder (bia_rpt)."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7

def uuid7str() -> str: return str(uuid7())

class ReportType(str, Enum):
	TABULAR="tabular"; SUMMARY="summary"; CROSS_TAB="cross_tab"; CHART="chart"
	DASHBOARD_EXPORT="dashboard_export"; LETTER="letter"; INVOICE="invoice"; CUSTOM="custom"

class ReportState(str, Enum):
	DRAFT="draft"; PUBLISHED="published"; SCHEDULED="scheduled"; ARCHIVED="archived"; DEPRECATED="deprecated"

class OutputFormat(str, Enum):
	PDF="pdf"; XLSX="xlsx"; CSV="csv"; HTML="html"; DOCX="docx"; JSON="json"; XML="xml"

class DistributionChannel(str, Enum):
	EMAIL="email"; SFTP="sftp"; S3="s3"; WEBHOOK="webhook"
	IN_APP="in_app"; SHAREPOINT="sharepoint"; API="api"

class ScheduleFrequency(str, Enum):
	ONCE="once"; DAILY="daily"; WEEKLY="weekly"; MONTHLY="monthly"
	QUARTERLY="quarterly"; ANNUAL="annual"; CRON="cron"

_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

class ReportCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; report_type: ReportType; owner_id: str
	datasource_id: str; sections: list[dict[str, Any]] = Field(default_factory=list)
	parameters: list[dict[str, Any]] = Field(default_factory=list)
	default_format: OutputFormat = OutputFormat.PDF
	description: str | None = None; tags: list[str] = Field(default_factory=list)

class ReportUpdate(BaseModel):
	model_config = _CFG
	name: str | None = None; sections: list[dict[str, Any]] | None = None
	parameters: list[dict[str, Any]] | None = None
	default_format: OutputFormat | None = None
	description: str | None = None; tags: list[str] | None = None

class ReportResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	report_type: ReportType; state: ReportState = ReportState.DRAFT; version: str = "1.0.0"
	owner_id: str; datasource_id: str
	sections: list[dict[str, Any]] = Field(default_factory=list)
	parameters: list[dict[str, Any]] = Field(default_factory=list)
	default_format: OutputFormat; description: str | None = None
	tags: list[str] = Field(default_factory=list); published_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class ScheduleCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; report_id: str; frequency: ScheduleFrequency
	cron_expression: str | None = None; output_format: OutputFormat = OutputFormat.PDF
	owner_id: str; notification_targets: list[str] = Field(default_factory=list)

class ScheduleResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; report_id: str
	frequency: ScheduleFrequency; cron_expression: str | None = None
	output_format: OutputFormat; owner_id: str
	notification_targets: list[str] = Field(default_factory=list)
	active: bool = True; last_run_at: datetime | None = None; next_run_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class DistributionCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; report_id: str; channel: DistributionChannel
	recipient: str; output_format: OutputFormat = OutputFormat.PDF
	owner_id: str; config: dict[str, Any] = Field(default_factory=dict)
	is_external: bool = False

class DistributionResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; report_id: str
	channel: DistributionChannel; recipient: str; output_format: OutputFormat
	owner_id: str; config: dict[str, Any] = Field(default_factory=dict)
	is_external: bool; approved: bool = False; approved_by: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class RunRecord(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; report_id: str
	output_format: OutputFormat; parameters: dict[str, Any] = Field(default_factory=dict)
	status: str = "completed"; output_ref: str | None = None
	run_duration_ms: int | None = None; row_count: int | None = None; page_count: int | None = None
	triggered_by: str = "manual"; run_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
