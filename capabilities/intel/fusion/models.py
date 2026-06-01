"""In-memory models for APG Intelligence Fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FusionAuthority:
	id: str
	tenant_id: str
	authority_type: str
	scope_reference: str
	classification: str
	approver_id: str
	expires_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionWorkspace:
	id: str
	tenant_id: str
	workspace_type: str
	name: str
	classification: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionSource:
	id: str
	tenant_id: str
	source_type: str
	source_reference: str
	custodian_id: str
	authority_id: str
	lineage_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionArtifact:
	id: str
	tenant_id: str
	workspace_id: str
	source_id: str
	artifact_type: str
	artifact_reference: str
	content_fingerprint: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionCorrelation:
	id: str
	tenant_id: str
	artifact_id: str
	correlation_type: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionHypothesis:
	id: str
	tenant_id: str
	correlation_id: str
	hypothesis_type: str
	claim_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionAssessment:
	id: str
	tenant_id: str
	hypothesis_id: str
	assessment_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionReferral:
	id: str
	tenant_id: str
	assessment_id: str
	referral_type: str
	recipient: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionDissemination:
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
class FusionReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FusionAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
