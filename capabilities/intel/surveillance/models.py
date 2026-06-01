"""In-memory models for APG Digital Surveillance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SurveillanceAuthority:
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
class SurveillanceProgram:
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
class MonitoredAsset:
	id: str
	tenant_id: str
	asset_type: str
	asset_reference: str
	owner_id: str
	authority_id: str
	privacy_review_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SurveillanceSensor:
	id: str
	tenant_id: str
	sensor_type: str
	asset_id: str
	sensor_reference: str
	custodian_id: str
	calibration_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SurveillanceObservation:
	id: str
	tenant_id: str
	program_id: str
	sensor_id: str
	observation_type: str
	observation_reference: str
	content_fingerprint: str
	observed_at: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SurveillanceAlert:
	id: str
	tenant_id: str
	observation_id: str
	alert_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SurveillanceRiskAssessment:
	id: str
	tenant_id: str
	alert_id: str
	assessment_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SurveillanceReferral:
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
class SurveillanceDissemination:
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
class SurveillanceReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SurveillanceAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
