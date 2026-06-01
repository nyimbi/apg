"""In-memory models for APG FinTech Risk Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RiskAppetite:
	id: str
	tenant_id: str
	risk_domain: str
	threshold_minor: int
	currency: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskProfile:
	id: str
	tenant_id: str
	subject_reference: str
	subject_type: str
	kyc_reference: str
	exposure_minor: int
	currency: str
	risk_score: float
	source_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskExposure:
	id: str
	tenant_id: str
	profile_id: str
	exposure_type: str
	amount_minor: int
	currency: str
	limit_minor: int
	source_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskControl:
	id: str
	tenant_id: str
	profile_id: str
	control_type: str
	owner_id: str
	evidence_reference: str
	effectiveness_score: float

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class StressScenario:
	id: str
	tenant_id: str
	profile_id: str
	scenario_type: str
	impact_minor: int
	probability_bps: int
	mitigation_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LimitBreach:
	id: str
	tenant_id: str
	exposure_id: str
	severity: str
	evidence_reference: str
	remediation_owner: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskEvent:
	id: str
	tenant_id: str
	profile_id: str
	event_type: str
	severity: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RiskEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any]

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
