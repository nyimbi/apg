"""Pydantic v2 models for APG Pharma Supply Chain."""

from __future__ import annotations

from datetime import datetime
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _uuid7str() -> str:
	return str(uuid7())


class SupBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class Supplier(SupBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	supplier_code: str
	name: str
	supplier_type: str
	qualification_status: str = "unqualified"
	quality_agreement_reference: str | None = None
	quality_agreement_signed_date: datetime | None = None
	last_audit_date: datetime | None = None
	next_audit_due: datetime | None = None
	country: str
	regulatory_approval_references: list[str] = Field(default_factory=list)
	approved_materials: list[str] = Field(default_factory=list)
	on_approved_supplier_list: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("supplier_type")
	@classmethod
	def validate_supplier_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SUPPLIER_TYPES
		if v not in SUPPORTED_SUPPLIER_TYPES:
			raise ValueError(f"supplier_type must be one of {SUPPORTED_SUPPLIER_TYPES}")
		return v

	@field_validator("qualification_status")
	@classmethod
	def validate_qualification_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_QUALIFICATION_STATUSES
		if v not in SUPPORTED_QUALIFICATION_STATUSES:
			raise ValueError(f"qualification_status must be one of {SUPPORTED_QUALIFICATION_STATUSES}")
		return v


class SupplierCreate(SupBase):
	tenant_id: str
	supplier_code: str
	name: str
	supplier_type: str
	country: str
	created_by: str


class CmoRecord(SupBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	cmo_code: str
	name: str
	cmo_type: str
	supplier_id: str
	technical_agreement_reference: str | None = None
	technical_agreement_signed_date: datetime | None = None
	quality_agreement_reference: str | None = None
	manufacturing_agreement_reference: str | None = None
	site_audit_date: datetime | None = None
	gmp_certificate_reference: str | None = None
	gmp_certificate_expiry: datetime | None = None
	active: bool = True
	products_manufactured: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("cmo_type")
	@classmethod
	def validate_cmo_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CMO_TYPES
		if v not in SUPPORTED_CMO_TYPES:
			raise ValueError(f"cmo_type must be one of {SUPPORTED_CMO_TYPES}")
		return v


class DemandForecast(SupBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	forecast_number: str
	product_id: str
	method: str
	period: str
	forecast_horizon_months: int
	forecasted_demand: dict[str, float] = Field(default_factory=dict)
	actual_demand: dict[str, float] = Field(default_factory=dict)
	safety_stock: float = 0.0
	created_date: datetime = Field(default_factory=datetime.utcnow)
	reviewed_date: datetime | None = None
	sop_approved: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("method")
	@classmethod
	def validate_method(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_DEMAND_METHODS
		if v not in SUPPORTED_DEMAND_METHODS:
			raise ValueError(f"method must be one of {SUPPORTED_DEMAND_METHODS}")
		return v

	@field_validator("forecast_horizon_months")
	@classmethod
	def validate_horizon(cls, v: int) -> int:
		if v <= 0:
			raise ValueError("forecast_horizon_months must be positive")
		return v


class ImportLicense(SupBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	license_number: str
	license_type: str
	region: str
	product_ids: list[str]
	authority_reference: str
	granted_date: datetime | None = None
	expiry_date: datetime | None = None
	renewal_submitted_date: datetime | None = None
	status: str = "applied"
	issuing_authority: str
	scope: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("license_type")
	@classmethod
	def validate_license_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_IMPORT_LICENSE_TYPES
		if v not in SUPPORTED_IMPORT_LICENSE_TYPES:
			raise ValueError(f"license_type must be one of {SUPPORTED_IMPORT_LICENSE_TYPES}")
		return v


class SupplySecurityRecord(SupBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	product_id: str
	supply_status: str = "secure"
	risk_level: str = "low"
	primary_supplier_id: str | None = None
	alternate_supplier_id: str | None = None
	dual_sourced: bool = False
	inventory_days: float | None = None
	safety_stock_days: float | None = None
	shortage_reported: bool = False
	shortage_report_date: datetime | None = None
	contingency_plan_reference: str | None = None
	last_reviewed: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("supply_status")
	@classmethod
	def validate_supply_status(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SUPPLY_STATUSES
		if v not in SUPPORTED_SUPPLY_STATUSES:
			raise ValueError(f"supply_status must be one of {SUPPORTED_SUPPLY_STATUSES}")
		return v

	@field_validator("risk_level")
	@classmethod
	def validate_risk_level(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SECURITY_RISK_LEVELS
		if v not in SUPPORTED_SECURITY_RISK_LEVELS:
			raise ValueError(f"risk_level must be one of {SUPPORTED_SECURITY_RISK_LEVELS}")
		return v


class PurchaseOrder(SupBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	po_number: str
	order_type: str
	supplier_id: str
	product_id: str
	quantity: float
	unit_of_measure: str
	order_date: datetime = Field(default_factory=datetime.utcnow)
	expected_delivery: datetime | None = None
	actual_delivery: datetime | None = None
	status: str = "placed"
	coa_reference: str | None = None
	quality_released: bool = False
	transport_condition: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("order_type")
	@classmethod
	def validate_order_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_ORDER_TYPES
		if v not in SUPPORTED_ORDER_TYPES:
			raise ValueError(f"order_type must be one of {SUPPORTED_ORDER_TYPES}")
		return v

	@field_validator("quantity")
	@classmethod
	def validate_quantity(cls, v: float) -> float:
		if v <= 0:
			raise ValueError("quantity must be positive")
		return v


class SupplyContract(SupBase):
	id: str = Field(default_factory=_uuid7str)
	tenant_id: str
	contract_number: str
	contract_type: str
	supplier_id: str
	title: str
	version: str = "1.0"
	status: str = "draft"
	approved: bool = False
	approval_reference: str | None = None
	effective_date: datetime | None = None
	expiry_date: datetime | None = None
	renewal_initiated: bool = False
	storage_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str

	@field_validator("contract_type")
	@classmethod
	def validate_contract_type(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CONTRACT_TYPES
		if v not in SUPPORTED_CONTRACT_TYPES:
			raise ValueError(f"contract_type must be one of {SUPPORTED_CONTRACT_TYPES}")
		return v
