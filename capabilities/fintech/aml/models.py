"""Dependency-light data models for APG Anti Money Laundering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AmlTransaction:
	id: str
	tenant_id: str
	subject_reference: str
	kyc_profile_id: str
	amount: float
	currency: str
	source_capability: str
	source_reference: str
	risk_score: int
	typology_flags: list[str] = field(default_factory=list)
	status: str = "monitored"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "subject_reference": self.subject_reference, "kyc_profile_id": self.kyc_profile_id, "amount": self.amount, "currency": self.currency, "source_capability": self.source_capability, "source_reference": self.source_reference, "risk_score": self.risk_score, "typology_flags": list(self.typology_flags), "status": self.status}


@dataclass
class AmlAlert:
	id: str
	tenant_id: str
	alert_type: str
	severity: str
	subject_reference: str
	evidence_references: list[str]
	status: str = "open"
	disposition: str = ""
	reviewer_id: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "alert_type": self.alert_type, "severity": self.severity, "subject_reference": self.subject_reference, "evidence_references": list(self.evidence_references), "status": self.status, "disposition": self.disposition, "reviewer_id": self.reviewer_id}


@dataclass
class AmlCase:
	id: str
	tenant_id: str
	alert_id: str
	case_type: str
	investigator_id: str
	subject_reference: str
	status: str = "under_investigation"
	evidence_references: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "alert_id": self.alert_id, "case_type": self.case_type, "investigator_id": self.investigator_id, "subject_reference": self.subject_reference, "status": self.status, "evidence_references": list(self.evidence_references)}


@dataclass
class AmlSarDraft:
	id: str
	tenant_id: str
	case_id: str
	subject_reference: str
	jurisdiction: str
	narrative: str
	evidence_references: list[str]
	approved_by: str
	status: str = "approved_for_filing"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "case_id": self.case_id, "subject_reference": self.subject_reference, "jurisdiction": self.jurisdiction, "narrative": self.narrative, "evidence_references": list(self.evidence_references), "approved_by": self.approved_by, "status": self.status}


@dataclass
class AmlEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
