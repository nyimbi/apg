"""Domain models for APG Identity Federation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
	"""Return a timezone-aware UTC timestamp for deterministic package records."""
	return datetime.now(timezone.utc).isoformat()


class ProviderProtocol(str, Enum):
	SAML = "saml"
	OIDC = "oidc"
	LDAP = "ldap"
	SCIM = "scim"


class ProviderStatus(str, Enum):
	DRAFT = "draft"
	ACTIVE = "active"
	DISABLED = "disabled"
	STALE = "stale"


class SessionStatus(str, Enum):
	ACTIVE = "active"
	REVOKED = "revoked"
	EXPIRED = "expired"


@dataclass
class FederationProvider:
	id: str
	tenant_id: str
	name: str
	protocol: ProviderProtocol
	owner_id: str
	signing_key_id: str
	metadata_url: str = ""
	assertion_encrypted: bool = True
	redirect_allowlist: list[str] = field(default_factory=list)
	pkce_required: bool = True
	metadata_refreshed_at: str = field(default_factory=utc_now_iso)
	status: ProviderStatus = ProviderStatus.ACTIVE
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"protocol": self.protocol.value,
			"owner_id": self.owner_id,
			"signing_key_id": self.signing_key_id,
			"metadata_url": self.metadata_url,
			"assertion_encrypted": self.assertion_encrypted,
			"redirect_allowlist": list(self.redirect_allowlist),
			"pkce_required": self.pkce_required,
			"metadata_refreshed_at": self.metadata_refreshed_at,
			"status": self.status.value,
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class ClaimMapping:
	id: str
	tenant_id: str
	provider_id: str
	source_claim: str
	target_claim: str
	transform: str = "copy"
	reviewed: bool = True
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"provider_id": self.provider_id,
			"source_claim": self.source_claim,
			"target_claim": self.target_claim,
			"transform": self.transform,
			"reviewed": self.reviewed,
			"created_at": self.created_at,
		}


@dataclass
class FederatedSession:
	id: str
	tenant_id: str
	provider_id: str
	subject_id: str
	session_privilege: str = "standard"
	mfa_completed: bool = True
	issued_at: str = field(default_factory=utc_now_iso)
	expires_at: str = field(default_factory=utc_now_iso)
	status: SessionStatus = SessionStatus.ACTIVE
	risk_score: float = 0.0
	revoked_at: str | None = None
	revocation_reason: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"provider_id": self.provider_id,
			"subject_id": self.subject_id,
			"session_privilege": self.session_privilege,
			"mfa_completed": self.mfa_completed,
			"issued_at": self.issued_at,
			"expires_at": self.expires_at,
			"status": self.status.value,
			"risk_score": self.risk_score,
			"revoked_at": self.revoked_at,
			"revocation_reason": self.revocation_reason,
		}


@dataclass
class CertificateRecord:
	id: str
	tenant_id: str
	provider_id: str
	key_id: str
	expires_at: str
	rotated_at: str | None = None
	active: bool = True
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"provider_id": self.provider_id,
			"key_id": self.key_id,
			"expires_at": self.expires_at,
			"rotated_at": self.rotated_at,
			"active": self.active,
			"created_at": self.created_at,
		}


@dataclass
class FederationAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	provider_id: str | None = None
	subject_id: str | None = None
	decision: str = "allow"
	reason: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"provider_id": self.provider_id,
			"subject_id": self.subject_id,
			"decision": self.decision,
			"reason": self.reason,
			"created_at": self.created_at,
		}


@dataclass
class FederationHealthReport:
	id: str
	tenant_id: str
	stale_provider_count: int = 0
	active_session_count: int = 0
	expiring_certificate_count: int = 0
	metadata_refresh_required_count: int = 0
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"stale_provider_count": self.stale_provider_count,
			"active_session_count": self.active_session_count,
			"expiring_certificate_count": self.expiring_certificate_count,
			"metadata_refresh_required_count": self.metadata_refresh_required_count,
			"created_at": self.created_at,
		}
