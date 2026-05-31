"""Service layer for the Federated Learning capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .federated_engine import FederatedLearningEngine
from .models import (
	AggregationResult,
	FederatedModel,
	FederatedModelRelease,
	Federation,
	FederationAgentRecord,
	FedlAuditEvent,
	FedlLifecycleBatchRecord,
	ModelUpdate,
	Participant,
	TrainingRound,
)


class FedlService:
	"""Federation, participant, training-round, and aggregation service."""

	def __init__(self) -> None:
		contract = get_capability_contract()
		self._federations: dict[str, Federation] = {}
		self._participants: dict[str, Participant] = {}
		self._rounds: dict[str, TrainingRound] = {}
		self._updates: dict[str, ModelUpdate] = {}
		self._aggregations: dict[str, AggregationResult] = {}
		self._models: dict[str, FederatedModel] = {}
		self._releases: dict[str, FederatedModelRelease] = {}
		self._federation_agents: dict[str, FederationAgentRecord] = {}
		self._lifecycle_batches: dict[str, FedlLifecycleBatchRecord] = {}
		self._audit_events: dict[str, FedlAuditEvent] = {}
		self._engine = FederatedLearningEngine()
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_federation(
		self,
		federation_id: str,
		tenant_id: str,
		name: str,
		coordinator: str,
		model_family: str,
		objective_metric: str,
		privacy_epsilon_limit: float = 8.0,
		data_residency_regions: list[str] | tuple[str, ...] | None = None,
	) -> dict[str, Any]:
		self._enforce_allow({"tenant_context_present": bool(tenant_id)})
		if not coordinator:
			raise PermissionError("coordinator_required")
		if not model_family:
			raise PermissionError("model_family_required")
		if privacy_epsilon_limit <= 0:
			raise PermissionError("privacy_budget_required")
		regions = tuple(str(region) for region in data_residency_regions or [] if str(region))
		if not regions:
			raise PermissionError("data_residency_required")
		federation = Federation(
			id=federation_id,
			tenant_id=tenant_id,
			name=name,
			coordinator=coordinator,
			model_family=model_family,
			objective_metric=objective_metric,
			privacy_epsilon_limit=float(privacy_epsilon_limit),
			data_residency_regions=regions,
			status="active",
		)
		self._federations[federation_id] = federation
		self._record_audit(tenant_id, federation_id, "federation_created", coordinator, "allow", metadata={"region_count": len(regions)})
		return federation.to_dict()

	def register_participant(
		self,
		participant_id: str,
		tenant_id: str,
		federation_id: str,
		name: str,
		region: str,
		contract_ref: str,
		attested: bool,
		compute_profile: str = "standard",
	) -> dict[str, Any]:
		federation = self._require_federation(federation_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "join_federation",
			"participant_attested": bool(attested),
		})
		self._raise_if_denied(result)
		if not contract_ref:
			raise PermissionError("participant_contract_required")
		if region not in federation.data_residency_regions:
			raise PermissionError("data_residency_required")
		participant = Participant(
			id=participant_id,
			tenant_id=tenant_id,
			federation_id=federation_id,
			name=name,
			region=region,
			contract_ref=contract_ref,
			attested=bool(attested),
			compute_profile=compute_profile,
		)
		self._participants[participant_id] = participant
		self._record_audit(
			tenant_id,
			participant_id,
			"participant_registered",
			name,
			result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"federation_id": federation_id, "region": region},
		)
		return participant.to_dict()

	def start_round(
		self,
		round_id: str,
		tenant_id: str,
		federation_id: str,
		round_number: int,
		privacy_epsilon: float,
		approval_ref: str,
		secure_aggregation: bool = True,
		privacy_review_recorded: bool = True,
	) -> dict[str, Any]:
		federation = self._require_federation(federation_id, tenant_id)
		participant_ids = tuple(participant.id for participant in self._participants_for_federation(federation_id, tenant_id))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "start_round",
			"participant_count": len(participant_ids),
			"privacy_epsilon": float(privacy_epsilon),
			"privacy_review_recorded": bool(privacy_review_recorded),
		})
		self._enforce_allow_result(result)
		if not approval_ref:
			raise PermissionError("round_approval_required")
		if privacy_epsilon > federation.privacy_epsilon_limit:
			raise PermissionError("privacy_budget_exceeds_federation_limit")
		round_model = TrainingRound(
			id=round_id,
			tenant_id=tenant_id,
			federation_id=federation_id,
			round_number=int(round_number),
			participant_ids=participant_ids,
			privacy_epsilon=float(privacy_epsilon),
			approval_ref=approval_ref,
			secure_aggregation=bool(secure_aggregation),
		)
		self._rounds[round_id] = round_model
		self._record_audit(tenant_id, round_id, "round_started", federation.coordinator, result["decision"], metadata={"participant_count": len(participant_ids)})
		return round_model.to_dict()

	def submit_update(
		self,
		update_id: str,
		tenant_id: str,
		round_id: str,
		participant_id: str,
		payload: dict[str, Any],
		sample_count: int,
		quality_score: float,
		poisoning_signal: bool = False,
	) -> dict[str, Any]:
		round_model = self._require_round(round_id, tenant_id)
		participant = self._require_participant(participant_id, tenant_id)
		if participant.federation_id != round_model.federation_id:
			raise PermissionError("participant_not_in_federation")
		if participant_id not in round_model.participant_ids:
			raise PermissionError("participant_not_in_round")
		if round_model.status != "running":
			raise PermissionError("round_not_running")
		if sample_count <= 0:
			raise PermissionError("sample_count_required")
		if not 0.0 <= quality_score <= 1.0:
			raise PermissionError("quality_score_out_of_range")
		detected_signal = self._engine.poisoning_signal(float(quality_score), poisoning_signal)
		update = ModelUpdate(
			id=update_id,
			tenant_id=tenant_id,
			round_id=round_id,
			participant_id=participant_id,
			update_digest=self._engine.update_digest(payload),
			sample_count=int(sample_count),
			quality_score=float(quality_score),
			poisoning_signal=detected_signal,
			status="quarantined" if detected_signal else "accepted",
		)
		self._updates[update_id] = update
		self._record_audit(tenant_id, update_id, "update_submitted", participant.name, "allow", metadata={"round_id": round_id, "poisoning_signal": detected_signal})
		return update.to_dict()

	def aggregate_updates(
		self,
		aggregation_id: str,
		tenant_id: str,
		round_id: str,
		secure_aggregation_enabled: bool,
		privacy_review_recorded: bool = True,
	) -> dict[str, Any]:
		round_model = self._require_round(round_id, tenant_id)
		federation = self._require_federation(round_model.federation_id, tenant_id)
		updates = [update for update in self._updates.values() if update.round_id == round_id and update.tenant_id == tenant_id]
		poisoning_signal = any(update.poisoning_signal for update in updates)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "aggregate_updates",
			"secure_aggregation_enabled": bool(secure_aggregation_enabled),
			"privacy_epsilon": float(round_model.privacy_epsilon),
			"privacy_review_recorded": bool(privacy_review_recorded),
			"poisoning_signal_detected": poisoning_signal,
		})
		self._enforce_allow_result(result)
		accepted_updates = [update for update in updates if update.status == "accepted"]
		if round_model.status != "running":
			raise PermissionError("round_not_running")
		if not round_model.secure_aggregation or not secure_aggregation_enabled:
			raise PermissionError("secure_aggregation_required")
		if len(accepted_updates) < len(round_model.participant_ids):
			raise PermissionError("participant_updates_incomplete")
		aggregate_digest = self._engine.aggregate_digest([update.to_dict() for update in accepted_updates])
		version = self._engine.model_version(federation.id, round_model.round_number, aggregate_digest)
		aggregation = AggregationResult(
			id=aggregation_id,
			tenant_id=tenant_id,
			round_id=round_id,
			federation_id=federation.id,
			aggregate_digest=aggregate_digest,
			participant_count=len(accepted_updates),
			total_sample_count=sum(update.sample_count for update in accepted_updates),
			privacy_epsilon_spent=round_model.privacy_epsilon,
			model_version=version,
		)
		model = FederatedModel(
			id=f"model:{version}",
			tenant_id=tenant_id,
			federation_id=federation.id,
			model_family=federation.model_family,
			model_version=version,
			source_round_id=round_id,
			aggregate_digest=aggregate_digest,
		)
		self._aggregations[aggregation_id] = aggregation
		self._models[model.id] = model
		self._rounds[round_id] = TrainingRound(
			id=round_model.id,
			tenant_id=round_model.tenant_id,
			federation_id=round_model.federation_id,
			round_number=round_model.round_number,
			participant_ids=round_model.participant_ids,
			privacy_epsilon=round_model.privacy_epsilon,
			approval_ref=round_model.approval_ref,
			secure_aggregation=round_model.secure_aggregation,
			status="aggregated",
		)
		self._record_audit(
			tenant_id,
			aggregation_id,
			"updates_aggregated",
			federation.coordinator,
			result["decision"],
			metadata={"round_id": round_id, "model_version": version},
		)
		return aggregation.to_dict()

	def release_model(
		self,
		release_id: str,
		tenant_id: str,
		model_id: str,
		mlcm_model_ref: str,
		release_approval_ref: str,
		privacy_review_ref: str,
		artifact_ref: str = "",
	) -> dict[str, Any]:
		model = self._require_model(model_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "release_model",
			"mlcm_model_ref_present": bool(mlcm_model_ref),
			"release_approval_recorded": bool(release_approval_ref),
			"privacy_review_recorded": bool(privacy_review_ref),
		})
		self._enforce_allow_result(result)
		release = FederatedModelRelease(
			id=release_id,
			tenant_id=tenant_id,
			model_id=model.id,
			federation_id=model.federation_id,
			mlcm_model_ref=mlcm_model_ref,
			release_approval_ref=release_approval_ref,
			privacy_review_ref=privacy_review_ref,
			artifact_ref=artifact_ref,
		)
		self._releases[release_id] = release
		self._record_audit(
			tenant_id,
			release_id,
			"model_released",
			model.federation_id,
			result["decision"],
			metadata={"model_id": model.id, "mlcm_model_ref": mlcm_model_ref},
		)
		return release.to_dict()

	def retire_federation(
		self,
		federation_id: str,
		tenant_id: str,
		impact_review_ref: str,
		retired_by: str = "",
	) -> dict[str, Any]:
		federation = self._require_federation(federation_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_federation",
			"impact_review_recorded": bool(impact_review_ref),
		})
		self._enforce_allow_result(result)
		retired = Federation(
			id=federation.id,
			tenant_id=federation.tenant_id,
			name=federation.name,
			coordinator=federation.coordinator,
			model_family=federation.model_family,
			objective_metric=federation.objective_metric,
			privacy_epsilon_limit=federation.privacy_epsilon_limit,
			data_residency_regions=federation.data_residency_regions,
			status="retired",
		)
		self._federations[federation_id] = retired
		self._record_audit(
			tenant_id,
			federation_id,
			"federation_retired",
			retired_by,
			result["decision"],
			metadata={"impact_review_ref": impact_review_ref},
		)
		return retired.to_dict()

	def register_federation_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		"""Register a first-class federation agent with guardrail evidence."""
		self._enforce_allow({"tenant_context_present": bool(tenant_id)})
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_federation_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		self._raise_if_denied(result)
		if not name:
			raise ValueError("federation_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		record = FederationAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status=status,
		)
		self._federation_agents[self._tenant_record_key(tenant_id, record.id)] = record
		self._record_audit(
			tenant_id,
			record.id,
			"federation_agent_registered",
			owner,
			result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"runtime": runtime_value, "role": role_value, "status": status, "matched_rules": result["matched_rules"]},
		)
		return record.to_dict()

	def validate_fedl_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "federation_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		"""Validate that FEDL lifecycle mutation batches flow through Bytewax."""
		self._enforce_allow({"tenant_context_present": bool(tenant_id)})
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("fedl_lifecycle_batch_empty")
		stream_value = self._normalize_token(event_stream)
		operation_value = self._normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_fedl_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_fedl_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = FedlLifecycleBatchRecord(
			id=batch_id or f"fedlbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_record_key(tenant_id, record.id)] = record
		self._record_audit(
			tenant_id,
			record.id,
			f"fedl_lifecycle_batch_{record.status}",
			"bytewax",
			result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata=record.to_dict(),
		)
		if not accepted:
			self._raise_if_denied(result)
		return record.to_dict()

	def list_federations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._federations, tenant_id)

	def list_participants(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._participants, tenant_id)

	def list_rounds(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._rounds, tenant_id)

	def list_updates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._updates, tenant_id)

	def list_aggregations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._aggregations, tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._models, tenant_id)

	def list_releases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._releases, tenant_id)

	def list_federation_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._federation_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing federations as FEDL records."""
		return self.list_federations(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that creates a federation record."""
		metadata = dict(metadata or {})
		federation = self.create_federation(
			federation_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or "Federated Learning Group"),
			coordinator=str(metadata.get("coordinator") or "fedl-coordinator"),
			model_family=str(metadata.get("model_family") or "tabular"),
			objective_metric=str(metadata.get("objective_metric") or "accuracy"),
			privacy_epsilon_limit=float(metadata.get("privacy_epsilon_limit") or 8.0),
			data_residency_regions=list(metadata.get("data_residency_regions") or ["us-east"]),
		)
		if status != "active":
			self._federations[record_id] = Federation(
				id=federation["id"],
				tenant_id=federation["tenant_id"],
				name=federation["name"],
				coordinator=federation["coordinator"],
				model_family=federation["model_family"],
				objective_metric=federation["objective_metric"],
				privacy_epsilon_limit=federation["privacy_epsilon_limit"],
				data_residency_regions=tuple(federation["data_residency_regions"]),
				status=status,
			)
			return self._federations[record_id].to_dict()
		return federation

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		rounds = self.list_rounds(tenant_id)
		updates = self.list_updates(tenant_id)
		agents = [item for item in self._federation_agents.values() if item.tenant_id == tenant_id]
		batches = [item for item in self._lifecycle_batches.values() if item.tenant_id == tenant_id]
		return {
			"federation_count": len(self.list_federations(tenant_id)),
			"participant_count": len(self.list_participants(tenant_id)),
			"round_count": len(rounds),
			"running_round_count": len([item for item in rounds if item["status"] == "running"]),
			"aggregated_round_count": len([item for item in rounds if item["status"] == "aggregated"]),
			"accepted_update_count": len([item for item in updates if item["status"] == "accepted"]),
			"quarantined_update_count": len([item for item in updates if item["status"] == "quarantined"]),
			"aggregation_count": len(self.list_aggregations(tenant_id)),
			"model_count": len(self.list_models(tenant_id)),
			"release_count": len(self.list_releases(tenant_id)),
			"federation_agent_count": len(agents),
			"pending_agent_review_count": sum(1 for item in agents if item.status == "pending_review"),
			"lifecycle_batch_count": len(batches),
			"denied_lifecycle_batch_count": sum(1 for item in batches if item.status == "denied"),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def privacy_budget_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		rounds = self.list_rounds(tenant_id)
		spent = sum(float(item["privacy_epsilon"]) for item in rounds if item["status"] == "aggregated")
		return {
			"tenant_id": tenant_id,
			"spent_epsilon": spent,
			"active_round_epsilon": sum(float(item["privacy_epsilon"]) for item in rounds if item["status"] == "running"),
			"aggregation_count": len(self.list_aggregations(tenant_id)),
		}

	def _participants_for_federation(self, federation_id: str, tenant_id: str) -> list[Participant]:
		return [
			participant for participant in self._participants.values()
			if participant.tenant_id == tenant_id and participant.federation_id == federation_id and participant.status == "active"
		]

	def _list(self, values: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _require_federation(self, federation_id: str, tenant_id: str) -> Federation:
		federation = self._federations.get(federation_id)
		if federation is None or federation.tenant_id != tenant_id:
			raise KeyError(f"unknown federation: {federation_id}")
		return federation

	def _require_participant(self, participant_id: str, tenant_id: str) -> Participant:
		participant = self._participants.get(participant_id)
		if participant is None or participant.tenant_id != tenant_id:
			raise KeyError(f"unknown participant: {participant_id}")
		return participant

	def _require_round(self, round_id: str, tenant_id: str) -> TrainingRound:
		round_model = self._rounds.get(round_id)
		if round_model is None or round_model.tenant_id != tenant_id:
			raise KeyError(f"unknown training round: {round_id}")
		return round_model

	def _require_model(self, model_id: str, tenant_id: str) -> FederatedModel:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise KeyError(f"unknown federated model: {model_id}")
		return model

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> FedlAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = FedlAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[event_id] = event
		return event

	def _enforce_allow(self, context: dict[str, Any]) -> None:
		self._enforce_allow_result(self.evaluate(context))

	def _enforce_allow_result(self, result: dict[str, Any]) -> None:
		self._raise_if_denied(result)
		if result["decision"] == "require_review":
			reasons = ", ".join(action.get("reason", "fedl_policy_review_required") for action in result["actions"])
			raise PermissionError(reasons or "fedl_policy_review_required")

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "fedl_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "fedl_policy_blocked")

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"
