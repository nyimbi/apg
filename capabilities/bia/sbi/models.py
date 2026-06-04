"""Pydantic v2 models for APG Self-Service BI (bia_sbi)."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7

def uuid7str() -> str: return str(uuid7())

class BuilderTool(str, Enum):
	DRAG_DROP="drag_drop_chart"; NLQ="natural_language_query"; WIZARD="guided_wizard"
	TEMPLATE="template_gallery"; SQL="sql_editor"

class ChartType(str, Enum):
	BAR="bar"; LINE="line"; PIE="pie"; SCATTER="scatter"; AREA="area"
	FUNNEL="funnel"; HEATMAP="heatmap"; TABLE="table"; KPI="kpi"
	MAP="map"; TREEMAP="treemap"; GAUGE="gauge"

class DatasourceMode(str, Enum):
	GOVERNED="governed_catalogue"; SANDBOX="sandbox"; FILE="uploaded_file"; API="approved_api"

class CatalogueState(str, Enum):
	DRAFT="draft"; PENDING="pending_approval"; PUBLISHED="published"
	DEPRECATED="deprecated"; RESTRICTED="restricted"

class SandboxState(str, Enum):
	ACTIVE="active"; PAUSED="paused"; EXPIRED="expired"; DELETED="deleted"

class GovernanceTier(str, Enum):
	OPEN="open"; GOVERNED="governed"; RESTRICTED="restricted"; CLASSIFIED="classified"

class AccessLevel(str, Enum):
	PERSONAL="personal"; TEAM="team"; PUBLISHED="published"; EMBEDDED="embedded"

_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

class WorkspaceCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; owner_id: str
	access_level: AccessLevel = AccessLevel.PERSONAL
	description: str | None = None; tags: list[str] = Field(default_factory=list)

class WorkspaceResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	owner_id: str; access_level: AccessLevel
	charts: list[str] = Field(default_factory=list)
	datasource_ids: list[str] = Field(default_factory=list)
	description: str | None = None; tags: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class CatalogueEntryCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; datasource_id: str; owner_id: str
	state: CatalogueState = CatalogueState.DRAFT
	governance_tier: GovernanceTier = GovernanceTier.GOVERNED
	description: str; schema_ref: str | None = None
	tags: list[str] = Field(default_factory=list)

class CatalogueEntryResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	datasource_id: str; owner_id: str; state: CatalogueState
	governance_tier: GovernanceTier; description: str
	schema_ref: str | None = None; tags: list[str] = Field(default_factory=list)
	approved_by: str | None = None; approved_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class SandboxCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; owner_id: str
	datasource_ids: list[str] = Field(default_factory=list)
	description: str | None = None

class SandboxResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	owner_id: str; state: SandboxState = SandboxState.ACTIVE
	datasource_ids: list[str] = Field(default_factory=list)
	row_count: int = 0; description: str | None = None
	expires_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class NLQRequest(BaseModel):
	model_config = _CFG
	tenant_id: str; query_text: str; workspace_id: str | None = None
	nlq_engine: str = "hybrid"; submitted_by: str

class NLQResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; query_text: str
	generated_sql: str | None = None; result_summary: str | None = None
	chart_type_suggestion: ChartType | None = None
	nlq_engine: str; submitted_by: str; confidence: float = 0.0
	submitted_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class ChartCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; workspace_id: str; name: str; chart_type: ChartType
	datasource_id: str; config: dict[str, Any] = Field(default_factory=dict)
	owner_id: str

class ChartResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; workspace_id: str
	name: str; chart_type: ChartType; datasource_id: str
	config: dict[str, Any] = Field(default_factory=dict); owner_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"
