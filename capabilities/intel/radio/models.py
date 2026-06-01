"""In-memory models for APG Radio Intelligence Listener."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RadioAuthority:
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
class RadioBandPlan:
	id: str
	tenant_id: str
	band_type: str
	name: str
	frequency_min_mhz: float
	frequency_max_mhz: float
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RadioReceiver:
	id: str
	tenant_id: str
	receiver_type: str
	site_reference: str
	custodian_id: str
	authority_id: str
	calibration_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RadioCollectionSession:
	id: str
	tenant_id: str
	band_id: str
	receiver_id: str
	session_type: str
	started_at: str
	ended_at: str
	collection_plan_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RadioSignalObservation:
	id: str
	tenant_id: str
	session_id: str
	frequency_mhz: float
	signal_type: str
	signal_fingerprint: str
	observed_at: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RadioTransmissionClassification:
	id: str
	tenant_id: str
	observation_id: str
	classification_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RadioEventAssessment:
	id: str
	tenant_id: str
	classification_id: str
	event_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RadioReferral:
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
class RadioDissemination:
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
class RadioReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RadioAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
