"""Pydantic v2 models for APG Mobile Device Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

class DeviceEnrolmentCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	serial_number: str
	device_type: str
	os_platform: str
	os_version: str
	ownership_type: str
	enrolment_method: str
	approval_reference: str
	assigned_user_id: str | None = None
	asset_tag: str | None = None
	location: str | None = None
	created_by: str

	@field_validator("device_type")
	@classmethod
	def device_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DEVICE_TYPES
		assert v in SUPPORTED_DEVICE_TYPES, f"device_type must be one of {SUPPORTED_DEVICE_TYPES}"
		return v

	@field_validator("os_platform")
	@classmethod
	def os_platform_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_OS_PLATFORMS
		assert v in SUPPORTED_OS_PLATFORMS, f"os_platform must be one of {SUPPORTED_OS_PLATFORMS}"
		return v

	@field_validator("ownership_type")
	@classmethod
	def ownership_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_OWNERSHIP_TYPES
		assert v in SUPPORTED_OWNERSHIP_TYPES, f"ownership_type must be one of {SUPPORTED_OWNERSHIP_TYPES}"
		return v

	@field_validator("enrolment_method")
	@classmethod
	def enrolment_method_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_ENROLMENT_METHODS
		assert v in SUPPORTED_ENROLMENT_METHODS, f"enrolment_method must be one of {SUPPORTED_ENROLMENT_METHODS}"
		return v


class DeviceUpdate(BaseModel):
	model_config = _MODEL_CFG
	enrolment_state: str | None = None
	assigned_user_id: str | None = None
	asset_tag: str | None = None
	location: str | None = None
	os_version: str | None = None
	updated_by: str


class DeviceResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	serial_number: str
	device_type: str
	os_platform: str
	os_version: str
	ownership_type: str
	enrolment_method: str
	enrolment_state: str = "enrolled"
	approval_reference: str
	assigned_user_id: str | None = None
	asset_tag: str | None = None
	location: str | None = None
	compliance_state: str = "pending_evaluation"
	enrolled_at: datetime = Field(default_factory=datetime.utcnow)
	last_seen_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class PolicyCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	name: str
	policy_type: str
	description: str | None = None
	configuration: dict[str, Any] = Field(default_factory=dict)
	platform_targets: list[str] = Field(default_factory=list)
	created_by: str

	@field_validator("policy_type")
	@classmethod
	def policy_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_POLICY_TYPES
		assert v in SUPPORTED_POLICY_TYPES, f"policy_type must be one of {SUPPORTED_POLICY_TYPES}"
		return v


class PolicyUpdate(BaseModel):
	model_config = _MODEL_CFG
	name: str | None = None
	description: str | None = None
	configuration: dict[str, Any] | None = None
	platform_targets: list[str] | None = None
	state: str | None = None
	updated_by: str


class PolicyResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	policy_type: str
	description: str | None = None
	configuration: dict[str, Any] = Field(default_factory=dict)
	platform_targets: list[str] = Field(default_factory=list)
	state: str = "draft"
	version: int = 1
	approval_reference: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Policy Assignment
# ---------------------------------------------------------------------------

class PolicyAssignmentCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	policy_id: str
	device_id: str
	assigned_by: str
	created_by: str


class PolicyAssignmentResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	policy_id: str
	device_id: str
	assigned_by: str
	state: str = "active"
	assigned_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Compliance Record
# ---------------------------------------------------------------------------

class ComplianceEvaluationCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	device_id: str
	evaluator_id: str
	findings: list[dict[str, Any]] = Field(default_factory=list)
	created_by: str


class ComplianceRecordResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	device_id: str
	compliance_state: str
	findings: list[dict[str, Any]] = Field(default_factory=list)
	evaluated_by: str
	evaluated_at: datetime = Field(default_factory=datetime.utcnow)
	next_evaluation_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# App Distribution
# ---------------------------------------------------------------------------

class AppDistributionCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	app_bundle_id: str
	app_name: str
	app_version: str
	device_id: str
	distribution_type: str
	approval_reference: str | None = None
	silent_install: bool = False
	created_by: str

	@field_validator("distribution_type")
	@classmethod
	def distribution_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_APP_DISTRIBUTION_TYPES
		assert v in SUPPORTED_APP_DISTRIBUTION_TYPES, f"distribution_type must be one of {SUPPORTED_APP_DISTRIBUTION_TYPES}"
		return v


class AppDistributionResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	app_bundle_id: str
	app_name: str
	app_version: str
	device_id: str
	distribution_type: str
	approval_reference: str | None = None
	silent_install: bool
	state: str = "pending"
	distributed_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Remote Wipe
# ---------------------------------------------------------------------------

class WipeRequestCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	device_id: str
	wipe_type: str
	approval_reference: str
	second_approval_reference: str
	justification: str
	requested_by: str
	created_by: str

	@field_validator("wipe_type")
	@classmethod
	def wipe_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_WIPE_TYPES
		assert v in SUPPORTED_WIPE_TYPES, f"wipe_type must be one of {SUPPORTED_WIPE_TYPES}"
		return v


class WipeRequestResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	device_id: str
	wipe_type: str
	approval_reference: str
	second_approval_reference: str
	justification: str
	requested_by: str
	state: str = "pending"
	executed_at: datetime | None = None
	completed_at: datetime | None = None
	error_message: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# MDM Profile
# ---------------------------------------------------------------------------

class MdmProfileCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	name: str
	profile_type: str
	platform: str
	payload: dict[str, Any] = Field(default_factory=dict)
	created_by: str

	@field_validator("profile_type")
	@classmethod
	def profile_type_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PROFILE_TYPES
		assert v in SUPPORTED_PROFILE_TYPES, f"profile_type must be one of {SUPPORTED_PROFILE_TYPES}"
		return v


class MdmProfileResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	profile_type: str
	platform: str
	payload: dict[str, Any] = Field(default_factory=dict)
	state: str = "draft"
	deployed_to_count: int = 0
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# MDM Alert
# ---------------------------------------------------------------------------

class MdmAlertResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	device_id: str
	alert_type: str
	severity: str
	message: str
	resolved: bool = False
	resolved_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
