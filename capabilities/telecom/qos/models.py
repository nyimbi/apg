"""In-memory models for APG Quality of Service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class QosPolicy:
	id: str
	tenant_id: str
	policy_type: str
	qos_class: str
	name: str
	parameters: str
	approval_reference: str
	status: str
	created_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class QosTrafficClassification:
	id: str
	tenant_id: str
	traffic_type: str
	classification: str
	policy_id: str
	flow_reference: str
	classified_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class QosEnforcementRecord:
	id: str
	tenant_id: str
	policy_id: str
	ne_reference: str
	status: str
	enforced_at: str
	last_updated: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class QosSlasMeasurement:
	id: str
	tenant_id: str
	sla_parameter: str
	measured_value: float
	target_value: float
	customer_id: str | None
	is_breach: bool
	measured_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class QosDegradation:
	id: str
	tenant_id: str
	cause: str
	confidence_score: float
	description: str
	affected_resource: str
	evidence_reference: str
	detected_at: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class QosRootCause:
	id: str
	tenant_id: str
	degradation_id: str
	root_cause_description: str
	confidence_score: float
	evidence_reference: str
	identified_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class QosRemediation:
	id: str
	tenant_id: str
	degradation_id: str
	remediation_type: str
	is_disruptive: bool
	approval_reference: str | None
	status: str
	triggered_at: str
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class QosAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
