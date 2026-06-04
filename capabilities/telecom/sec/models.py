"""In-memory models for APG Telecom Security."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SecFraudCase:
	id: str
	tenant_id: str
	fraud_type: str
	msisdn: str
	confidence_score: float
	evidence_reference: str
	status: str
	detected_at: str
	resolved_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SecSs7Attack:
	id: str
	tenant_id: str
	attack_type: str
	source_reference: str
	target_reference: str
	evidence_reference: str
	detected_at: str
	mitigated_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SecDiameterAttack:
	id: str
	tenant_id: str
	attack_type: str
	source_realm: str
	target_realm: str
	evidence_reference: str
	detected_at: str
	mitigated_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SecLawfulIntercept:
	id: str
	tenant_id: str
	intercept_type: str
	target_msisdn: str
	warrant_reference: str
	regulatory_authority: str
	status: str
	activated_at: str | None
	expires_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SecIncident:
	id: str
	tenant_id: str
	incident_type: str
	severity: str
	description: str
	evidence_reference: str
	status: str
	assigned_to: str | None
	opened_at: str
	resolved_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SecThreatIntel:
	id: str
	tenant_id: str
	source: str
	ioc_type: str
	ioc_value: str
	tlp_level: str
	valid_from: str
	valid_to: str | None
	shared: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SecAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
