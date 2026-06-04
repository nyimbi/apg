"""Pydantic v2 models for APG Multi-Country Operations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from typing_extensions import Annotated

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

try:
	from .capability_contract import (
		SUPPORTED_COMPLIANCE_DOMAINS,
		SUPPORTED_COMPLIANCE_STATUSES,
		SUPPORTED_COUNTRY_STATUSES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_ENTITY_TYPES,
		SUPPORTED_INTERCOMPANY_STATUSES,
		SUPPORTED_INTERCOMPANY_TYPES,
		SUPPORTED_JURISDICTIONS,
		SUPPORTED_REGULATORY_FRAMEWORKS,
		SUPPORTED_STATUTORY_REPORT_TYPES,
		SUPPORTED_STATUTORY_STATUSES,
		SUPPORTED_TRANSFER_PRICING_METHODS,
		SUPPORTED_APPROVAL_STATUSES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AGENT_ROLES,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore[no-redef]
		SUPPORTED_COMPLIANCE_DOMAINS,
		SUPPORTED_COMPLIANCE_STATUSES,
		SUPPORTED_COUNTRY_STATUSES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_ENTITY_TYPES,
		SUPPORTED_INTERCOMPANY_STATUSES,
		SUPPORTED_INTERCOMPANY_TYPES,
		SUPPORTED_JURISDICTIONS,
		SUPPORTED_REGULATORY_FRAMEWORKS,
		SUPPORTED_STATUTORY_REPORT_TYPES,
		SUPPORTED_STATUTORY_STATUSES,
		SUPPORTED_TRANSFER_PRICING_METHODS,
		SUPPORTED_APPROVAL_STATUSES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AGENT_ROLES,
	)

_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


def _check_jurisdiction(v: str) -> str:
	assert v.lower() in SUPPORTED_JURISDICTIONS, f"jurisdiction '{v}' not in {SUPPORTED_JURISDICTIONS}"
	return v.lower()


def _check_currency(v: str) -> str:
	assert v.upper() in SUPPORTED_CURRENCIES, f"currency '{v}' not in {SUPPORTED_CURRENCIES}"
	return v.upper()


def _check_entity_type(v: str) -> str:
	assert v in SUPPORTED_ENTITY_TYPES, f"entity_type '{v}' not in {SUPPORTED_ENTITY_TYPES}"
	return v


def _check_framework(v: str) -> str:
	assert v in SUPPORTED_REGULATORY_FRAMEWORKS, f"framework '{v}' not in {SUPPORTED_REGULATORY_FRAMEWORKS}"
	return v


def _check_country_status(v: str) -> str:
	assert v in SUPPORTED_COUNTRY_STATUSES, f"status '{v}' not in {SUPPORTED_COUNTRY_STATUSES}"
	return v


def _check_compliance_domain(v: str) -> str:
	assert v in SUPPORTED_COMPLIANCE_DOMAINS, f"domain '{v}' not in {SUPPORTED_COMPLIANCE_DOMAINS}"
	return v


def _check_compliance_status(v: str) -> str:
	assert v in SUPPORTED_COMPLIANCE_STATUSES, f"status '{v}' not in {SUPPORTED_COMPLIANCE_STATUSES}"
	return v


def _check_intercompany_type(v: str) -> str:
	assert v in SUPPORTED_INTERCOMPANY_TYPES, f"type '{v}' not in {SUPPORTED_INTERCOMPANY_TYPES}"
	return v


def _check_intercompany_status(v: str) -> str:
	assert v in SUPPORTED_INTERCOMPANY_STATUSES, f"status '{v}' not in {SUPPORTED_INTERCOMPANY_STATUSES}"
	return v


def _check_tp_method(v: str) -> str:
	assert v in SUPPORTED_TRANSFER_PRICING_METHODS, f"method '{v}' not in {SUPPORTED_TRANSFER_PRICING_METHODS}"
	return v


def _check_report_type(v: str) -> str:
	assert v in SUPPORTED_STATUTORY_REPORT_TYPES, f"type '{v}' not in {SUPPORTED_STATUTORY_REPORT_TYPES}"
	return v


def _check_statutory_status(v: str) -> str:
	assert v in SUPPORTED_STATUTORY_STATUSES, f"status '{v}' not in {SUPPORTED_STATUTORY_STATUSES}"
	return v


def _check_approval_status(v: str) -> str:
	assert v in SUPPORTED_APPROVAL_STATUSES, f"status '{v}' not in {SUPPORTED_APPROVAL_STATUSES}"
	return v


# --- Country models ---

class CountryCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	name: str
	jurisdiction: Annotated[str, AfterValidator(_check_jurisdiction)]
	functional_currency: Annotated[str, AfterValidator(_check_currency)]
	regulatory_framework: Annotated[str, AfterValidator(_check_framework)]
	tax_registration_required: bool = True
	notes: str | None = None


class CountryUpdate(BaseModel):
	model_config = _MODEL_CFG

	name: str | None = None
	regulatory_framework: Annotated[str | None, AfterValidator(lambda v: _check_framework(v) if v else v)] = None
	status: Annotated[str | None, AfterValidator(lambda v: _check_country_status(v) if v else v)] = None
	tax_registration_required: bool | None = None
	notes: str | None = None


class CountryResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	jurisdiction: str
	functional_currency: str
	regulatory_framework: str
	status: str = "active"
	tax_registration_required: bool = True
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Legal Entity models ---

class EntityCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	name: str
	entity_type: Annotated[str, AfterValidator(_check_entity_type)]
	country_id: str
	registration_number: str
	functional_currency: Annotated[str, AfterValidator(_check_currency)]
	parent_entity_id: str | None = None
	tax_id: str | None = None
	incorporation_date: date | None = None
	notes: str | None = None


class EntityUpdate(BaseModel):
	model_config = _MODEL_CFG

	name: str | None = None
	registration_number: str | None = None
	tax_id: str | None = None
	parent_entity_id: str | None = None
	notes: str | None = None
	is_active: bool | None = None


class EntityResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	entity_type: str
	country_id: str
	registration_number: str
	functional_currency: str
	parent_entity_id: str | None = None
	tax_id: str | None = None
	incorporation_date: date | None = None
	is_active: bool = True
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Regulatory Compliance Mapping ---

class ComplianceMappingCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	entity_id: str
	domain: Annotated[str, AfterValidator(_check_compliance_domain)]
	framework: Annotated[str, AfterValidator(_check_framework)]
	owner_id: str
	next_review_date: date
	evidence_reference: str
	notes: str | None = None


class ComplianceMappingUpdate(BaseModel):
	model_config = _MODEL_CFG

	status: Annotated[str | None, AfterValidator(lambda v: _check_compliance_status(v) if v else v)] = None
	owner_id: str | None = None
	next_review_date: date | None = None
	evidence_reference: str | None = None
	notes: str | None = None


class ComplianceMappingResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	entity_id: str
	domain: str
	framework: str
	status: str = "under_review"
	owner_id: str
	next_review_date: date
	evidence_reference: str
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Intercompany Transaction ---

class IntercompanyTransactionCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	transaction_type: Annotated[str, AfterValidator(_check_intercompany_type)]
	originator_entity_id: str
	counterparty_entity_id: str
	amount: float
	currency: Annotated[str, AfterValidator(_check_currency)]
	transaction_date: date
	transfer_pricing_method: Annotated[str, AfterValidator(_check_tp_method)]
	description: str
	documentation_reference: str | None = None


class IntercompanyTransactionUpdate(BaseModel):
	model_config = _MODEL_CFG

	status: Annotated[str | None, AfterValidator(lambda v: _check_intercompany_status(v) if v else v)] = None
	approval_reference: str | None = None
	settlement_date: date | None = None
	documentation_reference: str | None = None
	notes: str | None = None


class IntercompanyTransactionResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	transaction_type: str
	originator_entity_id: str
	counterparty_entity_id: str
	amount: float
	currency: str
	transaction_date: date
	transfer_pricing_method: str
	description: str
	status: str = "draft"
	approval_reference: str | None = None
	settlement_date: date | None = None
	documentation_reference: str | None = None
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Statutory Report ---

class StatutoryReportCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	entity_id: str
	report_type: Annotated[str, AfterValidator(_check_report_type)]
	period_start: date
	period_end: date
	due_date: date
	filer_id: str
	notes: str | None = None


class StatutoryReportUpdate(BaseModel):
	model_config = _MODEL_CFG

	status: Annotated[str | None, AfterValidator(lambda v: _check_statutory_status(v) if v else v)] = None
	filed_date: date | None = None
	acceptance_reference: str | None = None
	rejection_reason: str | None = None
	notes: str | None = None


class StatutoryReportResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	entity_id: str
	report_type: str
	period_start: date
	period_end: date
	due_date: date
	filer_id: str
	status: str = "draft"
	filed_date: date | None = None
	acceptance_reference: str | None = None
	rejection_reason: str | None = None
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- MCO Agent ---

class McoAgentCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	name: str
	runtime: Annotated[str, AfterValidator(lambda v: v if v in SUPPORTED_AGENT_RUNTIMES else (_ for _ in ()).throw(AssertionError(f"runtime '{v}' not supported")))]
	role: Annotated[str, AfterValidator(lambda v: v if v in SUPPORTED_AGENT_ROLES else (_ for _ in ()).throw(AssertionError(f"role '{v}' not supported")))]
	scope: str


class McoAgentResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Audit Event ---

class McoAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	reference_id: str
	actor_id: str = "system"
	payload: dict[str, Any] = Field(default_factory=dict)
	processor: str = "bytewax"
	stream: str = "apg.loc.mco.lifecycle"
	occurred_at: datetime = Field(default_factory=datetime.utcnow)
