"""Pydantic v2 models for ussd_flo capability."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:  # pragma: no cover
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())


# ── Flow node models ──────────────────────────────────────────────────────────

class FlowNodeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	node_id: str
	node_type: Literal["menu", "input", "decision", "action", "end"]
	title: str
	body: str = ""
	items: list[dict[str, Any]] = Field(default_factory=list)
	position_x: float = 0.0
	position_y: float = 0.0
	metadata: dict[str, Any] = Field(default_factory=dict)


class FlowNodeUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	title: str | None = None
	body: str | None = None
	items: list[dict[str, Any]] | None = None
	position_x: float | None = None
	position_y: float | None = None
	metadata: dict[str, Any] | None = None


class FlowNodeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	flow_id: str
	node_id: str
	node_type: str
	title: str
	body: str
	items: list[dict[str, Any]] = Field(default_factory=list)
	position_x: float
	position_y: float
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


# ── Flow edge (connection) models ─────────────────────────────────────────────

class FlowEdgeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	source_node_id: str
	target_node_id: str
	label: str = ""
	condition: str | None = None  # expression string e.g. "user_input == 1"
	priority: int = 0
	metadata: dict[str, Any] = Field(default_factory=dict)


class FlowEdgeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	flow_id: str
	source_node_id: str
	target_node_id: str
	label: str
	condition: str | None
	priority: int
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str


# ── Flow (full graph) models ──────────────────────────────────────────────────

class FlowCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	description: str = ""
	service_code: str
	root_node_id: str
	tenant_id: str = "default"
	languages: list[str] = Field(default_factory=lambda: ["en"])
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class FlowUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str | None = None
	description: str | None = None
	root_node_id: str | None = None
	languages: list[str] | None = None
	tags: list[str] | None = None
	status: str | None = None
	metadata: dict[str, Any] | None = None


class FlowResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str
	service_code: str
	root_node_id: str
	languages: list[str]
	tags: list[str]
	status: str  # draft | active | archived
	node_count: int = 0
	edge_count: int = 0
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


class FlowList(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	flows: list[FlowResponse]
	total: int
	tenant_id: str


class FlowFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str | None = None
	service_code: str | None = None
	status: str | None = None
	language: str | None = None
	tag: str | None = None


# ── Translation models ────────────────────────────────────────────────────────

class FlowTranslationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	flow_id: str
	language: str
	translations: dict[str, dict[str, str]]  # node_id -> {title, body, item_labels}
	tenant_id: str = "default"


class FlowTranslationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	flow_id: str
	tenant_id: str
	language: str
	translations: dict[str, dict[str, str]]
	created_at: str
	updated_at: str


# ── A/B test models ───────────────────────────────────────────────────────────

class AbTestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	service_code: str
	control_flow_id: str
	variant_flow_id: str
	split_percentage: float = 50.0  # % of traffic to variant
	tenant_id: str = "default"
	metadata: dict[str, Any] = Field(default_factory=dict)


class AbTestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	split_percentage: float | None = None
	status: str | None = None
	metadata: dict[str, Any] | None = None


class AbTestResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	service_code: str
	control_flow_id: str
	variant_flow_id: str
	split_percentage: float
	status: str  # active | paused | concluded
	control_sessions: int = 0
	variant_sessions: int = 0
	control_completions: int = 0
	variant_completions: int = 0
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


# ── Audit model ────────────────────────────────────────────────────────────────

class FloAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	resource_id: str
	resource_type: str
	actor_id: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str
