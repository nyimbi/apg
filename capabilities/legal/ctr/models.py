"""Contract Lifecycle Management — Pydantic v2 models."""
from __future__ import annotations

from datetime import date
from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class CtrContractCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	contract_type: str  # nda, msa, sow, lease, employment, vendor, partnership
	counterparty_id: str
	owner_id: str
	effective_date: str
	expiry_date: str | None = None
	auto_renew: bool = False
	renewal_notice_days: int = 30
	value: float | None = None
	currency: str = "KES"
	jurisdiction: str = ""
	governing_law: str = ""
	description: str = ""
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class CtrContractUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str | None = None
	status: str | None = None
	expiry_date: str | None = None
	auto_renew: bool | None = None
	renewal_notice_days: int | None = None
	value: float | None = None
	description: str | None = None
	tags: list[str] | None = None
	metadata: dict[str, Any] | None = None


class CtrContractResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	contract_type: str
	counterparty_id: str
	owner_id: str
	effective_date: str
	expiry_date: str | None
	auto_renew: bool
	renewal_notice_days: int
	value: float | None
	currency: str
	jurisdiction: str
	governing_law: str
	description: str
	status: str
	version: int
	tags: list[str]
	document_ids: list[str]
	obligation_count: int
	metadata: dict[str, Any]
	created_at: str
	updated_at: str | None = None
	signed_at: str | None = None
	executed_at: str | None = None


class CtrContractListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[CtrContractResponse]
	total: int
	page: int = 1
	page_size: int = 50


class CtrContractFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	contract_type: str | None = None
	counterparty_id: str | None = None
	owner_id: str | None = None
	expiring_before: str | None = None
	value_min: float | None = None
	value_max: float | None = None
	tags: list[str] | None = None


class CtrRedlineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	contract_id: str
	reviewer_id: str
	section_ref: str
	original_text: str
	proposed_text: str
	comment: str = ""
	change_type: str = "modification"  # addition, deletion, modification


class CtrRedlineResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	contract_id: str
	tenant_id: str
	reviewer_id: str
	section_ref: str
	original_text: str
	proposed_text: str
	comment: str
	change_type: str
	status: str  # pending, accepted, rejected
	resolved_by_id: str | None = None
	resolved_at: str | None = None
	created_at: str


class CtrObligationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	contract_id: str
	title: str
	description: str
	obligor: str  # us | counterparty
	due_date: str | None = None
	recurrence: str | None = None  # monthly, quarterly, annually
	owner_id: str


class CtrObligationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	contract_id: str
	tenant_id: str
	title: str
	description: str
	obligor: str
	due_date: str | None
	recurrence: str | None
	owner_id: str
	status: str
	last_fulfilled_at: str | None = None
	created_at: str


class CtrApprovalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	contract_id: str
	approver_id: str
	approval_level: int = 1
	comments: str = ""


class CtrApprovalResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	contract_id: str
	tenant_id: str
	approver_id: str
	approval_level: int
	comments: str
	status: str
	decided_at: str | None = None
	created_at: str


class CtrAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	contract_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
