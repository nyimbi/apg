"""Document & eDiscovery — Pydantic v2 models."""
from __future__ import annotations

from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class DscDocumentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	document_type: str  # pleading, brief, contract, correspondence, evidence, internal
	matter_id: str | None = None
	owner_id: str
	file_reference: str  # storage path/key
	file_size_bytes: int = 0
	mime_type: str = "application/pdf"
	description: str = ""
	tags: list[str] = Field(default_factory=list)
	is_privileged: bool = False
	privilege_type: str | None = None  # attorney_client, work_product, common_interest
	metadata: dict[str, Any] = Field(default_factory=dict)


class DscDocumentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str | None = None
	description: str | None = None
	tags: list[str] | None = None
	is_privileged: bool | None = None
	privilege_type: str | None = None
	metadata: dict[str, Any] | None = None


class DscDocumentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	document_type: str
	matter_id: str | None
	owner_id: str
	file_reference: str
	file_size_bytes: int
	mime_type: str
	description: str
	tags: list[str]
	is_privileged: bool
	privilege_type: str | None
	version: int
	status: str
	on_hold: bool
	hold_ids: list[str]
	metadata: dict[str, Any]
	created_at: str
	updated_at: str | None = None


class DscDocumentListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[DscDocumentResponse]
	total: int
	page: int = 1
	page_size: int = 50


class DscDocumentFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str | None = None
	document_type: str | None = None
	owner_id: str | None = None
	is_privileged: bool | None = None
	on_hold: bool | None = None
	tags: list[str] | None = None
	created_after: str | None = None
	created_before: str | None = None


class DscPrivilegeLogEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	document_id: str
	privilege_type: str
	basis: str
	logged_by_id: str
	notes: str = ""


class DscPrivilegeLogResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	document_id: str
	tenant_id: str
	privilege_type: str
	basis: str
	logged_by_id: str
	notes: str
	status: str
	created_at: str


class DscLitigationHoldCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	title: str
	description: str
	custodian_ids: list[str]
	issued_by_id: str
	scope_query: str = ""  # keyword/tag filter for auto-apply


class DscLitigationHoldResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	matter_id: str
	tenant_id: str
	title: str
	description: str
	custodian_ids: list[str]
	issued_by_id: str
	scope_query: str
	document_count: int
	status: str  # active, released
	released_at: str | None = None
	created_at: str


class DscProductionSetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	title: str
	document_ids: list[str]
	production_format: str = "pdf"  # pdf, tiff, native
	bates_prefix: str = ""
	requesting_party: str
	prepared_by_id: str


class DscProductionSetResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	matter_id: str
	tenant_id: str
	title: str
	document_ids: list[str]
	production_format: str
	bates_prefix: str
	bates_start: int
	bates_end: int
	requesting_party: str
	prepared_by_id: str
	status: str
	produced_at: str | None = None
	created_at: str


class DscAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	document_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
