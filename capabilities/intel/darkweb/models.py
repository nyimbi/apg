"""In-memory models for APG Dark Web Monitoring."""

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
class MonitoringProgram:
	id: str
	tenant_id: str
	program_type: str
	name: str
	priority: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class HiddenServiceSource:
	id: str
	tenant_id: str
	source_type: str
	network_type: str
	source_reference: str
	custodian_id: str
	authority_id: str
	access_review_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DarkWebObservation:
	id: str
	tenant_id: str
	program_id: str
	source_id: str
	observation_type: str
	observation_reference: str
	content_fingerprint: str
	observed_at: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ExposureIndicator:
	id: str
	tenant_id: str
	observation_id: str
	indicator_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MarketplaceRiskAssessment:
	id: str
	tenant_id: str
	indicator_id: str
	assessment_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatActorAssessment:
	id: str
	tenant_id: str
	indicator_id: str
	actor_reference: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DarkWebReferral:
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
class DarkWebDissemination:
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
class DarkWebReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DarkWebAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
