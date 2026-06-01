"""In-memory models for APG Signals Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SignalAuthority:
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
class SignalSource:
	id: str
	tenant_id: str
	source_type: str
	band: str
	source_reference: str
	owner_id: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CollectionTask:
	id: str
	tenant_id: str
	authority_id: str
	source_id: str
	collection_mode: str
	retention_days: int
	minimization_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SignalObservation:
	id: str
	tenant_id: str
	task_id: str
	observation_reference: str
	fingerprint: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProcessingBatch:
	id: str
	tenant_id: str
	observation_id: str
	processing_type: str
	quality_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SignalPattern:
	id: str
	tenant_id: str
	batch_id: str
	pattern_type: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SignalAssessment:
	id: str
	tenant_id: str
	pattern_id: str
	assessment_type: str
	classification: str
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SIGINTReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SIGINTAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
