"""Dependency-light data models for APG Know Your Customer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KycProfile:
	id: str
	tenant_id: str
	subject_reference: str
	legal_name: str
	customer_type: str
	country_code: str
	consent_reference: str
	status: str = "open"
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "subject_reference": self.subject_reference, "legal_name": self.legal_name, "customer_type": self.customer_type, "country_code": self.country_code, "consent_reference": self.consent_reference, "status": self.status, "metadata": dict(self.metadata)}


@dataclass
class KycDocument:
	id: str
	tenant_id: str
	profile_id: str
	document_type: str
	token_reference: str
	extracted_subject: str
	confidence: float
	status: str = "verified"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "profile_id": self.profile_id, "document_type": self.document_type, "token_reference": self.token_reference, "extracted_subject": self.extracted_subject, "confidence": self.confidence, "status": self.status}


@dataclass
class KycScreening:
	id: str
	tenant_id: str
	profile_id: str
	sanctions_hit: bool
	pep_hit: bool
	watchlist_hit: bool
	adverse_media_hit: bool
	review_id: str = ""
	status: str = "clear"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "profile_id": self.profile_id, "sanctions_hit": self.sanctions_hit, "pep_hit": self.pep_hit, "watchlist_hit": self.watchlist_hit, "adverse_media_hit": self.adverse_media_hit, "review_id": self.review_id, "status": self.status}


@dataclass
class KycDecision:
	id: str
	tenant_id: str
	profile_id: str
	decision: str
	risk_score: int
	review_id: str = ""
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "profile_id": self.profile_id, "decision": self.decision, "risk_score": self.risk_score, "review_id": self.review_id, "status": self.status}


@dataclass
class KycEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
