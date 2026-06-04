"""Pydantic v2 models for APG Medical Device Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class DeviceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	device_type: str
	device_class: str
	manufacturer: str
	model_number: str
	serial_number: str
	udi: str | None = None
	udi_format: str | None = None
	location: str
	department: str
	purchase_date: datetime | None = None
	warranty_expiry: datetime | None = None
	created_by: str


class DeviceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	device_type: str
	device_class: str
	manufacturer: str
	model_number: str
	serial_number: str
	udi: str | None = None
	udi_format: str | None = None
	location: str
	department: str
	status: str = "active"
	calibration_status: str = "current"
	last_calibrated_at: datetime | None = None
	next_calibration_due: datetime | None = None
	purchase_date: datetime | None = None
	warranty_expiry: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class MaintenanceScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	device_id: str
	maintenance_type: str
	scheduled_date: datetime
	assigned_to: str
	estimated_hours: float
	instructions: str
	created_by: str

	@field_validator("estimated_hours")
	@classmethod
	def hours_positive(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("estimated_hours must be positive")
		return v


class MaintenanceScheduleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	device_id: str
	maintenance_type: str
	scheduled_date: datetime
	assigned_to: str
	estimated_hours: float
	instructions: str
	status: str = "open"
	completed_at: datetime | None = None
	technician_notes: str = ""
	work_order_id: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class CalibrationRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	device_id: str
	calibrated_by: str
	calibration_date: datetime
	next_due_date: datetime
	certificate_reference: str
	result: str
	notes: str = ""
	created_by: str


class CalibrationRecordResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	device_id: str
	calibrated_by: str
	calibration_date: datetime
	next_due_date: datetime
	certificate_reference: str
	result: str
	notes: str = ""
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AdverseEventCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	device_id: str
	event_type: str
	severity: str
	description: str
	patient_id: str | None = None
	occurred_at: datetime
	reported_by: str
	immediate_action_taken: str
	created_by: str


class AdverseEventResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	device_id: str
	event_type: str
	severity: str
	description: str
	patient_id: str | None = None
	occurred_at: datetime
	reported_by: str
	immediate_action_taken: str
	status: str = "open"
	fda_mdr_reference: str | None = None
	root_cause: str | None = None
	corrective_action: str | None = None
	closed_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
