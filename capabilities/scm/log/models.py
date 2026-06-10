"""Pydantic v2 models for Logistics & Transportation (scm_log)."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class CarrierCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	carrier_code: str
	carrier_type: str  # air | sea | road | rail | multimodal
	country_of_origin: str
	services_offered: list[str] = Field(default_factory=list)
	contact_email: str | None = None
	contact_phone: str | None = None


class CarrierUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	services_offered: list[str] | None = None
	contact_email: str | None = None
	contact_phone: str | None = None
	status: str | None = None


class CarrierResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	carrier_code: str
	carrier_type: str
	country_of_origin: str
	services_offered: list[str]
	contact_email: str | None
	contact_phone: str | None
	status: str
	created_at: str
	updated_at: str | None = None


class ShipmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	carrier_id: str
	origin_address: dict[str, Any]
	destination_address: dict[str, Any]
	weight_kg: float
	volume_m3: float | None = None
	freight_mode: str  # air | sea | road | rail
	service_level: str = "standard"  # express | standard | economy
	declared_value: float | None = None
	currency: str = "USD"
	special_instructions: str | None = None
	reference_number: str | None = None


class ShipmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	tracking_number: str | None = None
	estimated_delivery: str | None = None
	actual_delivery: str | None = None
	special_instructions: str | None = None


class ShipmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	carrier_id: str
	origin_address: dict[str, Any]
	destination_address: dict[str, Any]
	weight_kg: float
	volume_m3: float | None
	freight_mode: str
	service_level: str
	declared_value: float | None
	currency: str
	tracking_number: str | None
	estimated_delivery: str | None
	actual_delivery: str | None
	special_instructions: str | None
	reference_number: str | None
	status: str
	created_at: str
	updated_at: str | None = None


class FreightAuditCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	shipment_id: str
	carrier_id: str
	invoice_number: str
	invoiced_amount: float
	expected_amount: float
	currency: str = "USD"
	audit_notes: str | None = None


class FreightAuditResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	shipment_id: str
	carrier_id: str
	invoice_number: str
	invoiced_amount: float
	expected_amount: float
	variance: float
	currency: str
	audit_notes: str | None
	status: str  # pending | approved | disputed | resolved
	created_at: str


class RouteCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	origin: str
	destination: str
	waypoints: list[str] = Field(default_factory=list)
	mode: str  # road | rail | sea | air
	distance_km: float | None = None
	estimated_transit_days: int | None = None


class RouteResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	origin: str
	destination: str
	waypoints: list[str]
	mode: str
	distance_km: float | None
	estimated_transit_days: int | None
	optimised_at: str | None
	status: str
	created_at: str


class CustomsDocumentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	shipment_id: str
	document_type: str  # commercial_invoice | packing_list | bill_of_lading | certificate_of_origin | customs_declaration
	country_of_export: str
	country_of_import: str
	hs_codes: list[str] = Field(default_factory=list)
	total_value: float
	currency: str = "USD"
	content: dict[str, Any] = Field(default_factory=dict)


class CustomsDocumentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	shipment_id: str
	document_type: str
	country_of_export: str
	country_of_import: str
	hs_codes: list[str]
	total_value: float
	currency: str
	content: dict[str, Any]
	status: str  # draft | submitted | approved | rejected
	created_at: str


class ThirdPartyLogisticsCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	provider_name: str
	provider_code: str
	service_types: list[str]
	contract_reference: str | None = None
	sla_days: int | None = None


class ThirdPartyLogisticsResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	provider_name: str
	provider_code: str
	service_types: list[str]
	contract_reference: str | None
	sla_days: int | None
	status: str
	created_at: str


class TrackingEventCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	shipment_id: str
	event_type: str  # pickup | in_transit | out_for_delivery | delivered | exception
	location: str
	description: str | None = None
	event_timestamp: str | None = None


class TrackingEventResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	shipment_id: str
	event_type: str
	location: str
	description: str | None
	event_timestamp: str
	created_at: str


class LogAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	status: str
	emitted_at: str
