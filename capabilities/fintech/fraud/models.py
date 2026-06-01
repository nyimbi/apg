"""Dependency-light data models for APG Fraud Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FraudSignal:
	id: str
	tenant_id: str
	subject_reference: str
	kyc_profile_id: str
	signal_type: str
	channel: str
	source_reference: str
	amount: float
	currency: str
	risk_score: int
	indicators: list[str] = field(default_factory=list)
	status: str = "scored"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "subject_reference": self.subject_reference, "kyc_profile_id": self.kyc_profile_id, "signal_type": self.signal_type, "channel": self.channel, "source_reference": self.source_reference, "amount": self.amount, "currency": self.currency, "risk_score": self.risk_score, "indicators": list(self.indicators), "status": self.status}


@dataclass
class FraudDecision:
	id: str
	tenant_id: str
	signal_id: str
	decision: str
	reason: str = ""
	reviewer_id: str = ""
	challenge_reference: str = ""
	human_approval: str = ""
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "signal_id": self.signal_id, "decision": self.decision, "reason": self.reason, "reviewer_id": self.reviewer_id, "challenge_reference": self.challenge_reference, "human_approval": self.human_approval, "status": self.status}


@dataclass
class FraudCase:
	id: str
	tenant_id: str
	signal_id: str
	case_type: str
	investigator_id: str
	subject_reference: str
	evidence_references: list[str]
	status: str = "open"
	disposition: str = ""
	reviewer_id: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "signal_id": self.signal_id, "case_type": self.case_type, "investigator_id": self.investigator_id, "subject_reference": self.subject_reference, "evidence_references": list(self.evidence_references), "status": self.status, "disposition": self.disposition, "reviewer_id": self.reviewer_id}


@dataclass
class FraudEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
