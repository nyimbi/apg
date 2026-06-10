"""Pydantic v2 models for Supplier Relationship Management (scm_srm)."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class SupplierCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	supplier_code: str
	country: str
	category: str  # raw_material | packaging | services | technology | logistics
	contact_email: str | None = None
	contact_phone: str | None = None
	payment_terms: str = "NET30"
	currency: str = "USD"
	notes: str | None = None


class SupplierUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	contact_email: str | None = None
	contact_phone: str | None = None
	payment_terms: str | None = None
	status: str | None = None
	notes: str | None = None


class SupplierResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	supplier_code: str
	country: str
	category: str
	contact_email: str | None
	contact_phone: str | None
	payment_terms: str
	currency: str
	preferred: bool
	risk_level: str
	overall_score: float | None
	status: str
	notes: str | None
	created_at: str
	updated_at: str | None = None


class ScorecardCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	supplier_id: str
	period: str
	quality_score: float
	delivery_score: float
	responsiveness_score: float
	cost_score: float
	sustainability_score: float | None = None
	reviewed_by: str
	notes: str | None = None


class ScorecardResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	period: str
	quality_score: float
	delivery_score: float
	responsiveness_score: float
	cost_score: float
	sustainability_score: float | None
	overall_score: float
	reviewed_by: str
	notes: str | None
	status: str
	created_at: str


class RiskAssessmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	supplier_id: str
	risk_category: str  # financial | geopolitical | operational | compliance | esg
	risk_level: str  # low | medium | high | critical
	description: str
	mitigation_plan: str | None = None
	assessed_by: str


class RiskAssessmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	risk_category: str
	risk_level: str
	description: str
	mitigation_plan: str | None
	assessed_by: str
	status: str
	created_at: str
	reviewed_at: str | None = None


class CollaborationMessageCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	supplier_id: str
	subject: str
	body: str
	message_type: str = "general"  # general | forecast_share | po_update | complaint | escalation
	attachments: list[str] = Field(default_factory=list)


class CollaborationMessageResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	subject: str
	body: str
	message_type: str
	attachments: list[str]
	status: str
	sent_at: str


class PerformanceReviewCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	supplier_id: str
	review_period: str
	reviewer: str
	summary: str
	action_items: list[str] = Field(default_factory=list)
	next_review_date: str | None = None


class PerformanceReviewResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	supplier_id: str
	review_period: str
	reviewer: str
	summary: str
	action_items: list[str]
	next_review_date: str | None
	status: str
	created_at: str


class SrmAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	status: str
	emitted_at: str
