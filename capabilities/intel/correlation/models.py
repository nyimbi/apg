"""In-memory models for APG Data Correlation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CorrelationAuthority:
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
class CorrelationWorkspace:
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
class CorrelationSource:
	id: str
	tenant_id: str
	workspace_id: str
	source_type: str
	source_reference: str
	custodian_id: str
	lineage_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationEntity:
	id: str
	tenant_id: str
	source_id: str
	entity_type: str
	entity_reference: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationObservation:
	id: str
	tenant_id: str
	entity_id: str
	observation_type: str
	observation_reference: str
	observed_at: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationRule:
	id: str
	tenant_id: str
	workspace_id: str
	rule_type: str
	rule_reference: str
	threshold_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationRun:
	id: str
	tenant_id: str
	rule_id: str
	run_type: str
	result_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationCluster:
	id: str
	tenant_id: str
	run_id: str
	cluster_type: str
	cluster_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationDecision:
	id: str
	tenant_id: str
	cluster_id: str
	decision_type: str
	rationale_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationReferral:
	id: str
	tenant_id: str
	decision_id: str
	referral_type: str
	recipient: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CorrelationAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
