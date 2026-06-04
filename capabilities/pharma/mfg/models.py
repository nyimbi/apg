"""Pydantic v2 models for APG Pharma Manufacturing."""

from __future__ import annotations

from datetime import datetime
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class MfgBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class BatchRecord(MfgBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	batch_number: str
	product_id: str
	manufacturing_type: str
	master_formula_reference: str
	status: str = "planned"
	line_id: str | None = None
	planned_quantity: float
	actual_quantity: float | None = None
	unit_of_measure: str
	theoretical_yield: float | None = None
	actual_yield: float | None = None
	yield_percentage: float | None = None
	qp_release_reference: str | None = None
	qp_signed_at: datetime | None = None
	start_date: datetime | None = None
	end_date: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_BATCH_STATUSES
		if v not in SUPPORTED_BATCH_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_BATCH_STATUSES}")
		return v

	@field_validator("manufacturing_type")
	@classmethod
	def validate_manufacturing_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_MANUFACTURING_TYPES
		if v not in SUPPORTED_MANUFACTURING_TYPES:
			raise ValueError(f"manufacturing_type must be one of {SUPPORTED_MANUFACTURING_TYPES}")
		return v


class BatchRecordCreate(MfgBase):
	tenant_id: str
	batch_number: str
	product_id: str
	manufacturing_type: str
	master_formula_reference: str
	planned_quantity: float
	unit_of_measure: str
	created_by: str


class Equipment(MfgBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	equipment_id: str
	name: str
	equipment_type: str
	model: str | None = None
	serial_number: str | None = None
	location: str
	status: str = "under_qualification"
	iq_reference: str | None = None
	oq_reference: str | None = None
	pq_reference: str | None = None
	last_calibration_date: datetime | None = None
	next_calibration_due: datetime | None = None
	last_maintenance_date: datetime | None = None
	next_maintenance_due: datetime | None = None
	requalification_due: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_EQUIPMENT_STATUSES
		if v not in SUPPORTED_EQUIPMENT_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_EQUIPMENT_STATUSES}")
		return v


class EquipmentQualification(MfgBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	equipment_id: str
	qualification_type: str
	protocol_reference: str
	report_reference: str | None = None
	status: str = "planned"
	performed_by: str
	approved_by: str | None = None
	start_date: datetime | None = None
	completion_date: datetime | None = None
	next_requalification: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("qualification_type")
	@classmethod
	def validate_qualification_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_QUALIFICATION_TYPES
		if v not in SUPPORTED_QUALIFICATION_TYPES:
			raise ValueError(f"qualification_type must be one of {SUPPORTED_QUALIFICATION_TYPES}")
		return v


class ManufacturingDeviation(MfgBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	deviation_number: str
	batch_id: str | None = None
	equipment_id: str | None = None
	deviation_type: str
	severity: str
	description: str
	status: str = "open"
	root_cause: str | None = None
	capa_reference: str | None = None
	raised_by: str
	raised_date: datetime = Field(default_factory=datetime.utcnow)
	closed_date: datetime | None = None
	gmp_impact: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("deviation_type")
	@classmethod
	def validate_deviation_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DEVIATION_TYPES
		if v not in SUPPORTED_DEVIATION_TYPES:
			raise ValueError(f"deviation_type must be one of {SUPPORTED_DEVIATION_TYPES}")
		return v

	@field_validator("severity")
	@classmethod
	def validate_severity(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DEVIATION_SEVERITIES
		if v not in SUPPORTED_DEVIATION_SEVERITIES:
			raise ValueError(f"severity must be one of {SUPPORTED_DEVIATION_SEVERITIES}")
		return v


class YieldRecord(MfgBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	batch_id: str
	yield_type: str
	step_name: str
	theoretical_quantity: float
	actual_quantity: float
	percentage: float | None = None
	variance_pct: float | None = None
	reconciled: bool = False
	investigation_required: bool = False
	investigation_reference: str | None = None
	recorded_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("yield_type")
	@classmethod
	def validate_yield_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_YIELD_TYPES
		if v not in SUPPORTED_YIELD_TYPES:
			raise ValueError(f"yield_type must be one of {SUPPORTED_YIELD_TYPES}")
		return v


class ProductionLine(MfgBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	line_code: str
	name: str
	manufacturing_type: str
	status: str = "available"
	cleaning_status: str = "cleared_for_use"
	current_batch_id: str | None = None
	last_cleaned_at: datetime | None = None
	last_cleared_at: datetime | None = None
	environmental_monitoring_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_LINE_STATUSES
		if v not in SUPPORTED_LINE_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_LINE_STATUSES}")
		return v

	@field_validator("cleaning_status")
	@classmethod
	def validate_cleaning_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CLEANING_STATUSES
		if v not in SUPPORTED_CLEANING_STATUSES:
			raise ValueError(f"cleaning_status must be one of {SUPPORTED_CLEANING_STATUSES}")
		return v


class RawMaterial(MfgBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	material_code: str
	name: str
	material_type: str
	vendor_id: str
	lot_number: str
	quantity: float
	unit_of_measure: str
	status: str = "quarantine"
	incoming_qc_reference: str | None = None
	expiry_date: datetime | None = None
	storage_condition: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_MATERIAL_STATUSES
		if v not in SUPPORTED_MATERIAL_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_MATERIAL_STATUSES}")
		return v
