"""In-memory models for APG Real-Time Monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MonitoringAuthority:
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
class MonitoringPolicy:
	id: str
	tenant_id: str
	policy_type: str
	name: str
	severity_floor: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringSource:
	id: str
	tenant_id: str
	source_type: str
	source_reference: str
	owner_id: str
	authority_id: str
	access_review_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringWatch:
	id: str
	tenant_id: str
	policy_id: str
	source_id: str
	watch_type: str
	watch_expression: str
	retention_class: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringEvent:
	id: str
	tenant_id: str
	watch_id: str
	event_type: str
	event_reference: str
	event_fingerprint: str
	observed_at: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringSignal:
	id: str
	tenant_id: str
	event_id: str
	signal_type: str
	severity: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringIncident:
	id: str
	tenant_id: str
	signal_id: str
	incident_type: str
	severity: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringReferral:
	id: str
	tenant_id: str
	incident_id: str
	referral_type: str
	recipient: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringDissemination:
	id: str
	tenant_id: str
	incident_id: str
	audience: str
	release_marking: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MonitoringAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
