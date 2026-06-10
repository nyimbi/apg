"""Data Catalog — Pydantic v2 models."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


# ── Dataset models ────────────────────────────────────────────────

class DatasetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	description: str = ""
	schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
	tags: list[str] = Field(default_factory=list)
	owner: str
	source_system: str
	location_uri: str = ""
	format: str = "unknown"
	classification: str = "internal"
	domain: str = "default"

class DatasetUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	description: str | None = None
	tags: list[str] | None = None
	owner: str | None = None
	location_uri: str | None = None
	format: str | None = None
	classification: str | None = None
	domain: str | None = None

class DatasetResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str
	schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
	tags: list[str]
	owner: str
	source_system: str
	location_uri: str
	format: str
	classification: str
	domain: str
	status: str = "active"
	created_at: str
	updated_at: str | None = None

class DatasetListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[dict[str, Any]]
	total: int
	page: int = 1
	page_size: int = 50


# ── Lineage models ────────────────────────────────────────────────

class LineageEdgeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	source_dataset_id: str
	target_dataset_id: str
	transformation: str = ""
	job_name: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)

class LineageEdgeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	source_dataset_id: str
	target_dataset_id: str
	transformation: str
	job_name: str
	metadata: dict[str, Any]
	created_at: str


# ── Tag / glossary models ─────────────────────────────────────────

class TagCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	color: str = "#6366f1"
	description: str = ""

class GlossaryTermCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	term: str
	definition: str
	domain: str = "general"
	synonyms: list[str] = Field(default_factory=list)
	related_terms: list[str] = Field(default_factory=list)

class GlossaryTermResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	term: str
	definition: str
	domain: str
	synonyms: list[str]
	related_terms: list[str]
	status: str = "approved"
	created_at: str


# ── Audit model ───────────────────────────────────────────────────

class CatalogAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	actor: str = "system"
	payload: dict[str, Any] = Field(default_factory=dict)
	created_at: str


# ── Filter model ──────────────────────────────────────────────────

class DatasetFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	owner: str | None = None
	domain: str | None = None
	classification: str | None = None
	format: str | None = None
	tags: list[str] = Field(default_factory=list)
	source_system: str | None = None
	status: str | None = None
