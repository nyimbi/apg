"""Intellectual Property Registry — Pydantic v2 models."""
from __future__ import annotations

from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class IpAssetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	asset_type: str  # patent, trademark, copyright, trade_secret, design, domain
	owner_id: str
	registration_number: str = ""
	application_number: str = ""
	filing_date: str = ""
	registration_date: str = ""
	expiry_date: str | None = None
	jurisdiction: str
	classes: list[str] = Field(default_factory=list)  # Nice classes for TMs
	description: str = ""
	inventors: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class IpAssetUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str | None = None
	status: str | None = None
	registration_number: str | None = None
	registration_date: str | None = None
	expiry_date: str | None = None
	description: str | None = None
	tags: list[str] | None = None
	metadata: dict[str, Any] | None = None


class IpAssetResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	asset_type: str
	owner_id: str
	registration_number: str
	application_number: str
	filing_date: str
	registration_date: str
	expiry_date: str | None
	jurisdiction: str
	classes: list[str]
	description: str
	inventors: list[str]
	status: str  # pending, registered, lapsed, abandoned, licensed
	renewal_due_date: str | None
	license_count: int
	tags: list[str]
	metadata: dict[str, Any]
	created_at: str
	updated_at: str | None = None


class IpAssetListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[IpAssetResponse]
	total: int
	page: int = 1
	page_size: int = 50


class IpAssetFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	asset_type: str | None = None
	owner_id: str | None = None
	jurisdiction: str | None = None
	status: str | None = None
	expiring_before: str | None = None
	tags: list[str] | None = None


class IpRenewalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	asset_id: str
	renewal_date: str
	renewal_fee: float
	currency: str = "KES"
	official_fee: float = 0.0
	agent_fee: float = 0.0
	submitted_by_id: str
	reference_number: str = ""
	notes: str = ""


class IpRenewalResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	asset_id: str
	tenant_id: str
	renewal_date: str
	renewal_fee: float
	currency: str
	official_fee: float
	agent_fee: float
	submitted_by_id: str
	reference_number: str
	notes: str
	status: str
	new_expiry_date: str | None = None
	created_at: str


class IpLicenseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	asset_id: str
	licensee_id: str
	license_type: str  # exclusive, non_exclusive, sole, sublicense
	territory: str
	start_date: str
	end_date: str | None = None
	royalty_rate: float = 0.0
	royalty_base: str = "revenue"  # revenue, unit, fixed
	upfront_fee: float = 0.0
	currency: str = "KES"
	restrictions: str = ""


class IpLicenseResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	asset_id: str
	tenant_id: str
	licensee_id: str
	license_type: str
	territory: str
	start_date: str
	end_date: str | None
	royalty_rate: float
	royalty_base: str
	upfront_fee: float
	currency: str
	restrictions: str
	status: str
	created_at: str


class IpRoyaltyRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	license_id: str
	period: str  # YYYY-MM
	base_amount: float
	royalty_amount: float
	currency: str = "KES"
	submitted_by_id: str
	notes: str = ""


class IpRoyaltyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	license_id: str
	tenant_id: str
	period: str
	base_amount: float
	royalty_amount: float
	currency: str
	submitted_by_id: str
	notes: str
	status: str
	paid_at: str | None = None
	created_at: str


class IpAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	asset_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
