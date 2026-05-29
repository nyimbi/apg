"""Domain models for the Federated Learning capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Federation:
	"""Tenant-scoped federated learning group and governance boundary."""

	id: str
	tenant_id: str
	name: str
	coordinator: str
	model_family: str
	objective_metric: str
	privacy_epsilon_limit: float
	data_residency_regions: tuple[str, ...]
	status: str = "draft"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"coordinator": self.coordinator,
			"model_family": self.model_family,
			"objective_metric": self.objective_metric,
			"privacy_epsilon_limit": self.privacy_epsilon_limit,
			"data_residency_regions": list(self.data_residency_regions),
			"status": self.status,
		}


@dataclass(frozen=True)
class Participant:
	"""Attested participant node in a federation."""

	id: str
	tenant_id: str
	federation_id: str
	name: str
	region: str
	contract_ref: str
	attested: bool
	compute_profile: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"federation_id": self.federation_id,
			"name": self.name,
			"region": self.region,
			"contract_ref": self.contract_ref,
			"attested": self.attested,
			"compute_profile": self.compute_profile,
			"status": self.status,
		}


@dataclass(frozen=True)
class TrainingRound:
	"""Approved federated training round with privacy-budget allocation."""

	id: str
	tenant_id: str
	federation_id: str
	round_number: int
	participant_ids: tuple[str, ...]
	privacy_epsilon: float
	approval_ref: str
	secure_aggregation: bool
	status: str = "running"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"federation_id": self.federation_id,
			"round_number": self.round_number,
			"participant_ids": list(self.participant_ids),
			"privacy_epsilon": self.privacy_epsilon,
			"approval_ref": self.approval_ref,
			"secure_aggregation": self.secure_aggregation,
			"status": self.status,
		}


@dataclass(frozen=True)
class ModelUpdate:
	"""Participant model update submitted for secure aggregation."""

	id: str
	tenant_id: str
	round_id: str
	participant_id: str
	update_digest: str
	sample_count: int
	quality_score: float
	poisoning_signal: bool = False
	status: str = "accepted"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"round_id": self.round_id,
			"participant_id": self.participant_id,
			"update_digest": self.update_digest,
			"sample_count": self.sample_count,
			"quality_score": self.quality_score,
			"poisoning_signal": self.poisoning_signal,
			"status": self.status,
		}


@dataclass(frozen=True)
class AggregationResult:
	"""Secure aggregate result for a completed federated training round."""

	id: str
	tenant_id: str
	round_id: str
	federation_id: str
	aggregate_digest: str
	participant_count: int
	total_sample_count: int
	privacy_epsilon_spent: float
	model_version: str
	status: str = "aggregated"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"round_id": self.round_id,
			"federation_id": self.federation_id,
			"aggregate_digest": self.aggregate_digest,
			"participant_count": self.participant_count,
			"total_sample_count": self.total_sample_count,
			"privacy_epsilon_spent": self.privacy_epsilon_spent,
			"model_version": self.model_version,
			"status": self.status,
		}


@dataclass(frozen=True)
class FederatedModel:
	"""Model registry entry produced by secure federated aggregation."""

	id: str
	tenant_id: str
	federation_id: str
	model_family: str
	model_version: str
	source_round_id: str
	aggregate_digest: str
	status: str = "registered"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"federation_id": self.federation_id,
			"model_family": self.model_family,
			"model_version": self.model_version,
			"source_round_id": self.source_round_id,
			"aggregate_digest": self.aggregate_digest,
			"status": self.status,
		}


@dataclass(frozen=True)
class FedlAuditEvent:
	"""Governance event emitted by federated learning operations."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
		}
