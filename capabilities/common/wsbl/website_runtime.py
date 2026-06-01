"""Domain runtime records for the Website Builder capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any


SITE_STATUSES = {"draft", "domain_pending", "ready", "published", "archived"}
PAGE_STATUSES = {"draft", "review_ready", "published", "archived"}
COMPONENT_STATUSES = {"available", "review_required", "approved", "retired"}
PUBLISH_STATUSES = {"approved", "review_required", "denied", "published", "rolled_back"}
AGENT_STATUSES = {"active", "disabled"}


def utc_now() -> str:
	"""Return a stable UTC timestamp string for dependency-light records."""
	return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: object) -> str:
	"""Build a deterministic short identifier from business-key parts."""
	key = "|".join(str(part) for part in parts)
	digest = sha1(key.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_status(value: str, allowed: set[str], default: str) -> str:
	status = (value or default).strip().lower().replace("-", "_")
	return status if status in allowed else default


@dataclass
class WebsiteDomainRecord:
	"""Validated or pending domain binding for a tenant site."""

	id: str
	tenant_id: str
	site_id: str
	domain: str
	validated: bool = False
	validation_method: str = "dns_txt"
	created_at: str = field(default_factory=utc_now)
	validated_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"site_id": self.site_id,
			"domain": self.domain,
			"validated": self.validated,
			"validation_method": self.validation_method,
			"created_at": self.created_at,
			"validated_at": self.validated_at,
		}


@dataclass
class WebsiteSiteRecord:
	"""Tenant-owned website with public-site and privacy governance state."""

	id: str
	tenant_id: str
	name: str
	owner_id: str
	locale: str = "en"
	public_site: bool = True
	privacy_banner_required: bool = True
	status: str = "draft"
	domains: list[str] = field(default_factory=list)
	published_version: int = 0
	required_actions: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner_id": self.owner_id,
			"locale": self.locale,
			"public_site": self.public_site,
			"privacy_banner_required": self.privacy_banner_required,
			"status": self.status,
			"domains": list(self.domains),
			"published_version": self.published_version,
			"required_actions": list(self.required_actions),
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class WebsiteComponentRecord:
	"""Reusable site-builder component governed before page use."""

	id: str
	tenant_id: str
	name: str
	component_type: str = "section"
	custom: bool = False
	status: str = "available"
	reviewed_by: str | None = None
	reviewed_at: str | None = None
	policy_id: str | None = None
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"component_type": self.component_type,
			"custom": self.custom,
			"status": self.status,
			"reviewed_by": self.reviewed_by,
			"reviewed_at": self.reviewed_at,
			"policy_id": self.policy_id,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass
class WebsitePageRecord:
	"""Versioned page definition composed from structured sections."""

	id: str
	tenant_id: str
	site_id: str
	slug: str
	title: str
	status: str = "draft"
	version: int = 1
	sections: list[dict[str, Any]] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"site_id": self.site_id,
			"slug": self.slug,
			"title": self.title,
			"status": self.status,
			"version": self.version,
			"sections": [dict(section) for section in self.sections],
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class WebsitePublishRequestRecord:
	"""Governed publication request for a site environment."""

	id: str
	tenant_id: str
	site_id: str
	requested_by: str
	environment: str = "production"
	status: str = "approved"
	approval_recorded: bool = False
	accessibility_passed: bool = False
	consent_policy_attached: bool = False
	required_actions: list[str] = field(default_factory=list)
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	published_version: int | None = None
	created_at: str = field(default_factory=utc_now)
	published_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"site_id": self.site_id,
			"requested_by": self.requested_by,
			"environment": self.environment,
			"status": self.status,
			"approval_recorded": self.approval_recorded,
			"accessibility_passed": self.accessibility_passed,
			"consent_policy_attached": self.consent_policy_attached,
			"required_actions": list(self.required_actions),
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"published_version": self.published_version,
			"created_at": self.created_at,
			"published_at": self.published_at,
		}


@dataclass
class WebsiteAuditEventRecord:
	"""Append-only audit event for builder and publishing actions."""

	id: str
	tenant_id: str
	action: str
	subject_id: str
	actor_id: str
	details: dict[str, Any] = field(default_factory=dict)
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"action": self.action,
			"subject_id": self.subject_id,
			"actor_id": self.actor_id,
			"details": dict(self.details),
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class WebsiteAgentRecord:
	"""Governed website-builder agent for review and publishing assistance."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	status: str = "active"
	human_approval_required: bool = True
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"owner": self.owner,
			"status": self.status,
			"human_approval_required": self.human_approval_required,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}
