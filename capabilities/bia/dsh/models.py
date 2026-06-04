"""Pydantic v2 models for APG Dashboard Management (bia_dsh)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class WidgetType(str, Enum):
	BAR_CHART = "bar_chart"
	LINE_CHART = "line_chart"
	PIE_CHART = "pie_chart"
	DONUT_CHART = "donut_chart"
	SCATTER_PLOT = "scatter_plot"
	HEATMAP = "heatmap"
	TABLE = "table"
	KPI_CARD = "kpi_card"
	GAUGE = "gauge"
	TREEMAP = "treemap"
	FUNNEL = "funnel"
	MAP = "map"
	TEXT = "text"
	IMAGE = "image"
	IFRAME = "iframe"


class LayoutType(str, Enum):
	GRID = "grid"
	FREEFORM = "freeform"
	RESPONSIVE_GRID = "responsive_grid"
	TABBED = "tabbed"
	STACKED = "stacked"


class DashboardState(str, Enum):
	DRAFT = "draft"
	PUBLISHED = "published"
	ARCHIVED = "archived"
	SCHEDULED = "scheduled"


class AccessLevel(str, Enum):
	PRIVATE = "private"
	TEAM = "team"
	ORGANISATION = "organisation"
	PUBLIC = "public"


class SnapshotFormat(str, Enum):
	PNG = "png"
	PDF = "pdf"
	HTML = "html"


class FilterType(str, Enum):
	DATE_RANGE = "date_range"
	DROPDOWN = "dropdown"
	MULTI_SELECT = "multi_select"
	TEXT_SEARCH = "text_search"
	SLIDER = "slider"
	CHECKBOX = "checkbox"


_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class WidgetCreate(BaseModel):
	model_config = _CFG
	tenant_id: str
	dashboard_id: str
	name: str
	widget_type: WidgetType
	datasource_type: str
	datasource_id: str
	config: dict[str, Any] = Field(default_factory=dict)
	position: dict[str, int] = Field(default_factory=dict)
	size: dict[str, int] = Field(default_factory=dict)
	refresh_interval: str = "manual"
	owner_id: str

	@field_validator("name")
	@classmethod
	def name_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("name must not be blank")
		return v.strip()


class WidgetUpdate(BaseModel):
	model_config = _CFG
	name: str | None = None
	config: dict[str, Any] | None = None
	position: dict[str, int] | None = None
	size: dict[str, int] | None = None
	refresh_interval: str | None = None


class WidgetResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	dashboard_id: str
	name: str
	widget_type: WidgetType
	datasource_type: str
	datasource_id: str
	config: dict[str, Any] = Field(default_factory=dict)
	position: dict[str, int] = Field(default_factory=dict)
	size: dict[str, int] = Field(default_factory=dict)
	refresh_interval: str = "manual"
	owner_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


class DashboardCreate(BaseModel):
	model_config = _CFG
	tenant_id: str
	name: str
	layout_type: LayoutType = LayoutType.RESPONSIVE_GRID
	access_level: AccessLevel = AccessLevel.PRIVATE
	owner_id: str
	description: str | None = None
	tags: list[str] = Field(default_factory=list)


class DashboardUpdate(BaseModel):
	model_config = _CFG
	name: str | None = None
	layout_type: LayoutType | None = None
	access_level: AccessLevel | None = None
	description: str | None = None
	tags: list[str] | None = None


class DashboardResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	layout_type: LayoutType
	access_level: AccessLevel
	state: DashboardState = DashboardState.DRAFT
	owner_id: str
	description: str | None = None
	tags: list[str] = Field(default_factory=list)
	widget_count: int = 0
	published_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


class SnapshotCreate(BaseModel):
	model_config = _CFG
	tenant_id: str
	dashboard_id: str
	format: SnapshotFormat = SnapshotFormat.PNG
	requested_by: str
	label: str | None = None


class SnapshotResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	dashboard_id: str
	format: SnapshotFormat
	storage_ref: str
	label: str | None = None
	requested_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime | None = None
	created_by: str = "system"


class DashboardFilterCreate(BaseModel):
	model_config = _CFG
	tenant_id: str
	dashboard_id: str
	name: str
	filter_type: FilterType
	target_field: str
	config: dict[str, Any] = Field(default_factory=dict)
	owner_id: str


class DashboardFilterResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	dashboard_id: str
	name: str
	filter_type: FilterType
	target_field: str
	config: dict[str, Any] = Field(default_factory=dict)
	owner_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"
