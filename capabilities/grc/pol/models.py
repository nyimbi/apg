"""Pydantic v2 models for grc_pol capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field

uuid7str = lambda: str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


# ── Enums ─────────────────────────────────────────────────────────────────────

class PolicyStatus(str, Enum):
	draft = "draft"
	in_review = "in_review"
	approved = "approved"
	published = "published"
	under_revision = "under_revision"
	archived = "archived"
	withdrawn = "withdrawn"
	superseded = "superseded"


class PolicyType(str, Enum):
	information_security = "information_security"
	acceptable_use = "acceptable_use"
	data_privacy = "data_privacy"
	hr = "hr"
	finance = "finance"
	operational = "operational"
	compliance = "compliance"
	bcdr = "bcdr"
	third_party = "third_party"
	anti_bribery = "anti_bribery"


class ExceptionStatus(str, Enum):
	pending = "pending"
	approved = "approved"
	rejected = "rejected"
	expired = "expired"


class AcknowledgementStatus(str, Enum):
	pending = "pending"
	completed = "completed"
	overdue = "overdue"


class ReviewAction(str, Enum):
	approve = "approve"
	request_changes = "request_changes"
	reject = "reject"


# ── Core models ───────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class Policy(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	title: str
	category: str
	policy_type: PolicyType
	owner_id: str
	scope: str = "organization_wide"
	description: str = ""
	version: str = "1.0"
	status: PolicyStatus = PolicyStatus.draft
	effective_date: str
	review_cycle_months: int
	next_review_date: str
	review_history: list[str] = Field(default_factory=list)
	revision_history: list[str] = Field(default_factory=list)
	acknowledgement_stats: dict[str, int] = Field(default_factory=dict)
	updated_at: str = Field(default_factory=_now)


class PolicyException(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	policy_id: str
	requestor_id: str
	exception_type: str = "temporary_exemption"
	reason: str
	compensating_controls: str
	risk_level: str
	duration_days: int
	expiry_date: str
	status: ExceptionStatus = ExceptionStatus.pending
	approver_id: str | None = None
	approved_until: str | None = None
	conditions: str | None = None
	requested_at: str = Field(default_factory=_now)


class PolicyAcknowledgement(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	policy_id: str
	employee_id: str
	status: AcknowledgementStatus = AcknowledgementStatus.pending
	method: str = "electronic_signature"
	deadline: str | None = None
	acknowledged_at: str | None = None
	requested_at: str = Field(default_factory=_now)


class PolicyRevision(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str | None = None
	created_at: str = Field(default_factory=_now)
	policy_id: str
	previous_version: str
	new_version: str
	revision_reason: str
	revision_summary: str
	revised_by: str
	revised_at: str = Field(default_factory=_now)


# ── Request / Response ────────────────────────────────────────────────────────

class CreatePolicyRequest(_Base):
	title: str
	category: str
	policy_type: PolicyType
	owner_id: str
	effective_date: str
	review_cycle_months: int
	scope: str = "organization_wide"
	description: str = ""
	version: str = "1.0"


class PolicyReviewRequest(_Base):
	reviewer_id: str
	comments: str
	recommended_action: ReviewAction


class ApprovePolicyRequest(_Base):
	approver_id: str
	approval_date: str
	comments: str = ""


class PublishPolicyRequest(_Base):
	distribution_list: list[str]


class AcknowledgeRequest(_Base):
	employee_id: str
	acknowledgement_date: str
	method: str = "electronic_signature"


class ExceptionRequest(_Base):
	requestor_id: str
	reason: str
	compensating_controls: str
	risk_level: str
	exception_type: str = "temporary_exemption"
	duration_days: int = 90


class ApproveExceptionRequest(_Base):
	approver_id: str
	approved_until: str
	conditions: str


class RevisionRequest(_Base):
	revision_reason: str
	revision_summary: str
	revised_by: str


class RetireRequest(_Base):
	reason: str
	retired_by: str
