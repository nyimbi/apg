"""Pydantic v2 models for APG Remote Workforce."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Work Policy
# ---------------------------------------------------------------------------

class WorkPolicyCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	name: str
	policy_type: str
	description: str | None = None
	effective_date: datetime | None = None
	expiry_date: datetime | None = None
	applicable_roles: list[str] = Field(default_factory=list)
	geographic_scope: list[str] = Field(default_factory=list)
	content: str = ""
	created_by: str

	@field_validator("policy_type")
	@classmethod
	def policy_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_WORK_POLICY_TYPES
		assert v in SUPPORTED_WORK_POLICY_TYPES, f"policy_type must be one of {SUPPORTED_WORK_POLICY_TYPES}"
		return v


class WorkPolicyUpdate(BaseModel):
	model_config = _MODEL_CFG
	name: str | None = None
	description: str | None = None
	content: str | None = None
	applicable_roles: list[str] | None = None
	geographic_scope: list[str] | None = None
	expiry_date: datetime | None = None
	state: str | None = None
	updated_by: str


class WorkPolicyResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	policy_type: str
	description: str | None = None
	content: str = ""
	effective_date: datetime | None = None
	expiry_date: datetime | None = None
	applicable_roles: list[str] = Field(default_factory=list)
	geographic_scope: list[str] = Field(default_factory=list)
	state: str = "draft"
	version: int = 1
	approval_reference: str | None = None
	acknowledgment_count: int = 0
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Policy Acknowledgment
# ---------------------------------------------------------------------------

class PolicyAcknowledgmentCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	policy_id: str
	employee_id: str
	acknowledged_at: datetime = Field(default_factory=datetime.utcnow)
	ip_address: str | None = None
	device_id: str | None = None
	created_by: str


class PolicyAcknowledgmentResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	policy_id: str
	employee_id: str
	acknowledged_at: datetime
	ip_address: str | None = None
	device_id: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# VPN Access
# ---------------------------------------------------------------------------

class VpnAccessCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	employee_id: str
	vpn_protocol: str
	approval_reference: str
	mfa_verified: bool = True
	split_tunneling_requested: bool = False
	allowed_networks: list[str] = Field(default_factory=list)
	created_by: str

	@field_validator("vpn_protocol")
	@classmethod
	def vpn_protocol_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_VPN_PROTOCOLS
		assert v in SUPPORTED_VPN_PROTOCOLS, f"vpn_protocol must be one of {SUPPORTED_VPN_PROTOCOLS}"
		return v


class VpnAccessResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_id: str
	vpn_protocol: str
	approval_reference: str
	mfa_verified: bool
	split_tunneling_enabled: bool = False
	allowed_networks: list[str] = Field(default_factory=list)
	state: str = "active"
	provisioned_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime | None = None
	revoked_at: datetime | None = None
	revocation_reason: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# VPN Session
# ---------------------------------------------------------------------------

class VpnSessionResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	vpn_access_id: str
	employee_id: str
	started_at: datetime = Field(default_factory=datetime.utcnow)
	ended_at: datetime | None = None
	bytes_in: int = 0
	bytes_out: int = 0
	client_ip: str | None = None
	duration_seconds: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Productivity Metric
# ---------------------------------------------------------------------------

class ProductivityMetricCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	employee_id: str
	metric_type: str
	value: float
	period_start: datetime
	period_end: datetime
	consent_given: bool = True
	notes: str | None = None
	created_by: str

	@field_validator("metric_type")
	@classmethod
	def metric_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PRODUCTIVITY_METRICS
		assert v in SUPPORTED_PRODUCTIVITY_METRICS, f"metric_type must be one of {SUPPORTED_PRODUCTIVITY_METRICS}"
		return v


class ProductivityMetricResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_id: str
	metric_type: str
	value: float
	period_start: datetime
	period_end: datetime
	consent_given: bool
	notes: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Equipment Requisition
# ---------------------------------------------------------------------------

class EquipmentRequisitionCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	employee_id: str
	equipment_type: str
	quantity: int = 1
	justification: str
	delivery_address: str
	created_by: str

	@field_validator("equipment_type")
	@classmethod
	def equipment_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_EQUIPMENT_TYPES
		assert v in SUPPORTED_EQUIPMENT_TYPES, f"equipment_type must be one of {SUPPORTED_EQUIPMENT_TYPES}"
		return v

	@field_validator("quantity")
	@classmethod
	def quantity_positive(cls, v: int) -> int:
		assert v >= 1, "quantity must be at least 1"
		return v


class EquipmentRequisitionResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_id: str
	equipment_type: str
	quantity: int
	justification: str
	delivery_address: str
	state: str = "requested"
	approval_reference: str | None = None
	asset_tag: str | None = None
	shipped_at: datetime | None = None
	delivered_at: datetime | None = None
	returned_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Digital Onboarding
# ---------------------------------------------------------------------------

class OnboardingRecordCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	employee_id: str
	manager_id: str
	manager_approval_reference: str
	start_date: datetime
	timezone: str = "UTC"
	collaboration_tools: list[str] = Field(default_factory=list)
	created_by: str


class OnboardingRecordResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_id: str
	manager_id: str
	manager_approval_reference: str
	start_date: datetime
	timezone: str
	collaboration_tools: list[str] = Field(default_factory=list)
	state: str = "not_started"
	completed_steps: list[str] = Field(default_factory=list)
	pending_steps: list[str] = Field(default_factory=list)
	completed_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Onboarding Step
# ---------------------------------------------------------------------------

class OnboardingStepCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	onboarding_id: str
	step_type: str
	notes: str | None = None
	completed_by: str
	created_by: str

	@field_validator("step_type")
	@classmethod
	def step_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_ONBOARDING_STEP_TYPES
		assert v in SUPPORTED_ONBOARDING_STEP_TYPES, f"step_type must be one of {SUPPORTED_ONBOARDING_STEP_TYPES}"
		return v


class OnboardingStepResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	onboarding_id: str
	step_type: str
	notes: str | None = None
	completed_by: str
	completed_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Compliance Check
# ---------------------------------------------------------------------------

class ComplianceCheckCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	employee_id: str
	check_type: str
	result: str  # "pass" | "fail" | "pending"
	evidence_reference: str | None = None
	notes: str | None = None
	created_by: str

	@field_validator("check_type")
	@classmethod
	def check_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_COMPLIANCE_CHECK_TYPES
		assert v in SUPPORTED_COMPLIANCE_CHECK_TYPES, f"check_type must be one of {SUPPORTED_COMPLIANCE_CHECK_TYPES}"
		return v


class ComplianceCheckResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_id: str
	check_type: str
	result: str
	evidence_reference: str | None = None
	notes: str | None = None
	next_due_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Remote Incident
# ---------------------------------------------------------------------------

class RemoteIncidentCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	employee_id: str
	incident_type: str
	description: str
	severity: str
	reported_by: str
	created_by: str

	@field_validator("incident_type")
	@classmethod
	def incident_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_INCIDENT_TYPES
		assert v in SUPPORTED_INCIDENT_TYPES, f"incident_type must be one of {SUPPORTED_INCIDENT_TYPES}"
		return v


class RemoteIncidentResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	employee_id: str
	incident_type: str
	description: str
	severity: str
	reported_by: str
	state: str = "open"
	resolution_notes: str | None = None
	resolved_at: datetime | None = None
	resolved_by: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
