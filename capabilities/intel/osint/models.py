"""In-memory models for APG Open Source Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CollectionRequirement:
	id: str
	tenant_id: str
	topic: str
	priority: str
	requester_id: str
	classification: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SourceRegistryEntry:
	id: str
	tenant_id: str
	source_type: str
	source_reference: str
	owner_id: str
	terms_review_reference: str
	risk_tier: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CollectionPlan:
	id: str
	tenant_id: str
	requirement_id: str
	source_id: str
	method: str
	cadence: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EvidenceRecord:
	id: str
	tenant_id: str
	plan_id: str
	content_reference: str
	fingerprint: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TriageDecision:
	id: str
	tenant_id: str
	evidence_id: str
	decision: str
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class IntelligenceAssessment:
	id: str
	tenant_id: str
	requirement_id: str
	assessment_type: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DisseminationPackage:
	id: str
	tenant_id: str
	assessment_id: str
	audience: str
	release_marking: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OSINTReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OSINTAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
