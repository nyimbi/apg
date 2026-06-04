"""Pydantic v2 models for APG Pharma Distribution."""

from __future__ import annotations

from datetime import datetime
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class DisBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class Shipment(DisBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	shipment_number: str
	distribution_channel: str
	origin_site: str
	destination_site: str
	transport_mode: str
	transport_condition: str
	status: str = "planned"
	packing_list_reference: str | None = None
	coa_reference: str | None = None
	import_permit_reference: str | None = None
	wda_reference: str | None = None
	dispatch_date: datetime | None = None
	expected_delivery: datetime | None = None
	actual_delivery: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SHIPMENT_STATUSES
		if v not in SUPPORTED_SHIPMENT_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_SHIPMENT_STATUSES}")
		return v

	@field_validator("distribution_channel")
	@classmethod
	def validate_distribution_channel(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DISTRIBUTION_CHANNELS
		if v not in SUPPORTED_DISTRIBUTION_CHANNELS:
			raise ValueError(f"distribution_channel must be one of {SUPPORTED_DISTRIBUTION_CHANNELS}")
		return v

	@field_validator("transport_mode")
	@classmethod
	def validate_transport_mode(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_TRANSPORT_MODES
		if v not in SUPPORTED_TRANSPORT_MODES:
			raise ValueError(f"transport_mode must be one of {SUPPORTED_TRANSPORT_MODES}")
		return v


class ShipmentCreate(DisBase):
	tenant_id: str
	shipment_number: str
	distribution_channel: str
	origin_site: str
	destination_site: str
	transport_mode: str
	transport_condition: str
	expected_delivery: datetime | None = None
	created_by: str


class ColdChainRecord(DisBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	shipment_id: str
	product_id: str
	cold_chain_classification: str
	min_temp_celsius: float
	max_temp_celsius: float
	logger_device_id: str
	qualification_reference: str
	mapping_study_reference: str | None = None
	monitoring_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("cold_chain_classification")
	@classmethod
	def validate_classification(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_COLD_CHAIN_CLASSIFICATIONS
		if v not in SUPPORTED_COLD_CHAIN_CLASSIFICATIONS:
			raise ValueError(f"cold_chain_classification must be one of {SUPPORTED_COLD_CHAIN_CLASSIFICATIONS}")
		return v


class TemperatureExcursion(DisBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	cold_chain_record_id: str
	shipment_id: str
	excursion_start: datetime
	excursion_end: datetime | None = None
	min_recorded: float
	max_recorded: float
	severity: str
	impact_assessment: str | None = None
	disposition: str | None = None
	regulatory_reported: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("severity")
	@classmethod
	def validate_severity(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_EXCURSION_SEVERITIES
		if v not in SUPPORTED_EXCURSION_SEVERITIES:
			raise ValueError(f"severity must be one of {SUPPORTED_EXCURSION_SEVERITIES}")
		return v


class SerialisationRecord(DisBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	product_id: str
	serial_number: str
	batch_number: str
	gtin: str | None = None
	sscc: str | None = None
	standard: str
	aggregation_level: str
	parent_id: str | None = None
	status: str = "active"
	verified: bool = False
	decommissioned: bool = False
	decommission_reason: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("standard")
	@classmethod
	def validate_standard(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SERIALISATION_STANDARDS
		if v not in SUPPORTED_SERIALISATION_STANDARDS:
			raise ValueError(f"standard must be one of {SUPPORTED_SERIALISATION_STANDARDS}")
		return v


class RecallRecord(DisBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	recall_number: str
	recall_class: str
	product_id: str
	batch_numbers: list[str]
	reason: str
	status: str = "initiated"
	initiated_date: datetime = Field(default_factory=datetime.utcnow)
	regulatory_notification_date: datetime | None = None
	effectiveness_check_date: datetime | None = None
	completed_date: datetime | None = None
	recall_scope: str
	units_recalled: int | None = None
	units_returned: int | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("recall_class")
	@classmethod
	def validate_recall_class(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_RECALL_CLASSES
		if v not in SUPPORTED_RECALL_CLASSES:
			raise ValueError(f"recall_class must be one of {SUPPORTED_RECALL_CLASSES}")
		return v

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_RECALL_STATUSES
		if v not in SUPPORTED_RECALL_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_RECALL_STATUSES}")
		return v


class WholesaleDistributionAuthorisation(DisBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	wda_number: str
	market: str
	holder_name: str
	site_address: str
	scope: list[str]
	status: str = "applied"
	granted_date: datetime | None = None
	expiry_date: datetime | None = None
	renewal_submitted_date: datetime | None = None
	issuing_authority: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("status")
	@classmethod
	def validate_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_WDA_STATUSES
		if v not in SUPPORTED_WDA_STATUSES:
			raise ValueError(f"status must be one of {SUPPORTED_WDA_STATUSES}")
		return v


class GdpDeviationRecord(DisBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	deviation_reference: str
	deviation_type: str
	description: str
	gdp_status: str
	affected_products: list[str] = Field(default_factory=list)
	affected_batches: list[str] = Field(default_factory=list)
	root_cause: str | None = None
	capa_reference: str | None = None
	raised_date: datetime = Field(default_factory=datetime.utcnow)
	closed_date: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("gdp_status")
	@classmethod
	def validate_gdp_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_GDP_STATUSES
		if v not in SUPPORTED_GDP_STATUSES:
			raise ValueError(f"gdp_status must be one of {SUPPORTED_GDP_STATUSES}")
		return v
