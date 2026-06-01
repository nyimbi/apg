"""Service layer for executable AI model lifecycle management."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .lifecycle_runtime import (
	deployment_posture,
	drift_status,
	evaluation_status,
	model_card_complete,
	normalize_deployment_status,
	normalize_score,
	normalize_stage,
	promotion_status,
	stable_id,
)
from .models import (
	DeploymentRecord,
	DeploymentTarget,
	DriftSignal,
	EvaluationRun,
	MlcmAuditEvent,
	MlcmLifecycleBatchRecord,
	ModelArtifact,
	ModelLifecycleAgentRecord,
	ModelVersion,
	PromotionRequest,
	RetirementRecord,
	RollbackRecord,
	utc_now_iso,
)


class MlcmService:
	"""In-process model registry, evaluation, promotion, deployment, and drift service."""

	def __init__(self, minimum_eval_score: float | None = None) -> None:
		contract = get_capability_contract()
		self.minimum_eval_score = float(
			minimum_eval_score
			if minimum_eval_score is not None
			else contract["configuration"]["evaluation"]["minimum_eval_score"]
		)
		self._models: dict[str, ModelArtifact] = {}
		self._versions: dict[str, ModelVersion] = {}
		self._evaluations: dict[str, EvaluationRun] = {}
		self._promotion_requests: dict[str, PromotionRequest] = {}
		self._targets: dict[str, DeploymentTarget] = {}
		self._deployments: dict[str, DeploymentRecord] = {}
		self._drift_signals: dict[str, DriftSignal] = {}
		self._rollbacks: dict[str, RollbackRecord] = {}
		self._retirements: dict[str, RetirementRecord] = {}
		self._agents: dict[str, ModelLifecycleAgentRecord] = {}
		self._lifecycle_batches: dict[str, MlcmLifecycleBatchRecord] = {}
		self._audit_events: dict[str, MlcmAuditEvent] = {}
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_model(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		problem_type: str,
		risk_level: str = "medium",
		description: str = "",
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_model",
			"owner_assigned": bool(owner),
			"name_present": bool(name),
			"problem_type_present": bool(problem_type),
			"risk_level_present": bool(risk_level),
		})
		self._raise_if_denied(result)
		model = ModelArtifact(
			id=model_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			problem_type=problem_type,
			risk_level=risk_level,
			description=description,
			tags=list(tags or []),
			metadata=dict(metadata or {}),
		)
		self._models[model.id] = model
		self._audit(tenant_id, "model_registered", model.id, f"Registered model {name}")
		return model.to_dict()

	def create_version(
		self,
		version_id: str,
		tenant_id: str,
		model_id: str,
		version: str,
		artifact_uri: str,
		model_card: dict[str, Any] | None = None,
		training_data_ref: str = "",
		baseline_ref: str = "",
		stage: str = "dev",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		model = self._require_model(model_id, tenant_id)
		model.updated_at = utc_now_iso()
		stage_value = normalize_stage(stage)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_version",
			"model_registered": True,
			"artifact_uri_present": bool(str(artifact_uri or "").strip()),
			"training_data_ref_present": bool(str(training_data_ref or "").strip()),
			"baseline_ref_present": bool(str(baseline_ref or "").strip()),
			"non_dev_stage": stage_value != "dev",
			"model_card_present": bool(model_card),
		})
		self._raise_if_denied(result)
		version_record = ModelVersion(
			id=version_id,
			tenant_id=tenant_id,
			model_id=model.id,
			version=version,
			artifact_uri=artifact_uri,
			stage=stage_value,
			status="pending_review" if result["decision"] == "require_review" else "candidate",
			model_card=dict(model_card or {}),
			training_data_ref=training_data_ref,
			baseline_ref=baseline_ref,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			metadata=dict(metadata or {}),
		)
		self._versions[version_record.id] = version_record
		self._audit(
			tenant_id,
			"model_version_created",
			version_record.id,
			f"Created version {version}",
			metadata={"decision": result["decision"], "matched_rules": result["matched_rules"]},
		)
		return version_record.to_dict()

	def record_evaluation(
		self,
		evaluation_id: str,
		tenant_id: str,
		version_id: str,
		score: float,
		baseline_ref: str,
		metrics: dict[str, float] | None = None,
		evidence_refs: list[str] | None = None,
		evaluator: str = "",
		fairness_review_recorded: bool = False,
		explainability_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		version = self._require_version(version_id, tenant_id)
		model = self._require_model(version.model_id, tenant_id)
		normalized_score = normalize_score(score)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_evaluation",
			"baseline_ref_present": bool(str(baseline_ref or "").strip()),
			"evidence_refs_present": bool(evidence_refs),
			"risk_level": model.risk_level,
			"fairness_review_recorded": bool(fairness_review_recorded),
			"explainability_recorded": bool(explainability_recorded),
		})
		self._raise_if_denied(result)
		evaluation = EvaluationRun(
			id=evaluation_id,
			tenant_id=tenant_id,
			model_id=version.model_id,
			version_id=version.id,
			score=normalized_score,
			baseline_ref=baseline_ref,
			metrics=dict(metrics or {}),
			status="pending_review" if result["decision"] == "require_review" else evaluation_status(normalized_score, self.minimum_eval_score),
			evidence_refs=list(evidence_refs or []),
			evaluator=evaluator,
			fairness_review_recorded=bool(fairness_review_recorded),
			explainability_recorded=bool(explainability_recorded),
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
		)
		self._evaluations[evaluation.id] = evaluation
		version.evaluation_score = normalized_score
		version.evaluation_id = evaluation.id
		version.baseline_ref = baseline_ref or version.baseline_ref
		if evaluation.status == "pending_review":
			version.status = "pending_review"
		else:
			version.status = "candidate" if evaluation.status == "passed" else "needs_improvement"
		self._audit(
			tenant_id,
			"model_evaluated",
			evaluation.id,
			f"Recorded evaluation score {normalized_score:.3f}",
			metadata={"decision": result["decision"], "matched_rules": result["matched_rules"]},
		)
		return evaluation.to_dict()

	def request_promotion(
		self,
		request_id: str,
		tenant_id: str,
		version_id: str,
		target_stage: str,
		requested_by: str,
		approval_recorded: bool = False,
		approval_ref: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		version = self._require_version(version_id, tenant_id)
		target = normalize_stage(target_stage)
		status, reasons = promotion_status(
			target,
			version.evaluation_score,
			self.minimum_eval_score,
			approval_recorded,
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"target_stage": target,
			"approval_recorded": approval_recorded,
			"eval_score": version.evaluation_score or 0.0,
			"promotion_requested": True,
		})
		self._raise_if_denied(result)
		request = PromotionRequest(
			id=request_id,
			tenant_id=tenant_id,
			model_id=version.model_id,
			version_id=version.id,
			source_stage=version.stage,
			target_stage=target,
			requested_by=requested_by,
			approval_recorded=approval_recorded,
			approval_ref=approval_ref,
			status=status,
			reasons=reasons,
			resolved_at=utc_now_iso() if status == "approved" else None,
		)
		self._promotion_requests[request.id] = request
		if status == "approved":
			version.stage = target
			version.status = "promoted"
			version.promoted_at = request.resolved_at
		self._audit(tenant_id, "promotion_requested", request.id, f"Promotion to {target} {status}")
		return request.to_dict()

	def create_target(
		self,
		target_id: str,
		tenant_id: str,
		name: str,
		environment: str,
		serving_runtime: str,
		owner: str,
		status: str = "active",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		target = DeploymentTarget(
			id=target_id,
			tenant_id=tenant_id,
			name=name,
			environment=environment,
			serving_runtime=serving_runtime,
			owner=owner,
			status=status,
			metadata=dict(metadata or {}),
		)
		self._targets[target.id] = target
		self._audit(tenant_id, "deployment_target_created", target.id, f"Created target {name}")
		return target.to_dict()

	def deploy_model(
		self,
		deployment_id: str,
		tenant_id: str,
		version_id: str,
		target_id: str,
		replicas: int = 1,
		canary_percent: int = 0,
		approved_by: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		version = self._require_version(version_id, tenant_id)
		target = self._require_target(target_id, tenant_id)
		if target.status != "active":
			raise PermissionError("deployment_target_inactive")
		unresolved_drift = self._unresolved_drift(version.id, tenant_id)
		posture, posture_reasons = deployment_posture(
			version.stage,
			model_card_complete(version.model_card),
			version.evaluation_score,
			self.minimum_eval_score,
			len(unresolved_drift),
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_model",
			"model_card_present": model_card_complete(version.model_card),
			"eval_score": version.evaluation_score or 0.0,
			"promotion_requested": version.stage == "production",
			"drift_detected": bool(unresolved_drift),
			"drift_review_recorded": not unresolved_drift,
		})
		self._raise_if_denied(result)
		self._raise_if_review_required(result)
		if posture == "blocked":
			raise PermissionError(", ".join(posture_reasons))
		deployment = DeploymentRecord(
			id=deployment_id,
			tenant_id=tenant_id,
			model_id=version.model_id,
			version_id=version.id,
			target_id=target.id,
			stage=version.stage,
			status=normalize_deployment_status("serving"),
			replicas=max(1, int(replicas)),
			canary_percent=max(0, min(100, int(canary_percent))),
			approved_by=approved_by,
			metadata=dict(metadata or {}),
		)
		self._deployments[deployment.id] = deployment
		version.status = "serving"
		self._models[version.model_id].status = "serving"
		self._audit(tenant_id, "model_deployed", deployment.id, f"Deployed {version.id} to {target.name}")
		return deployment.to_dict()

	def record_drift(
		self,
		signal_id: str,
		tenant_id: str,
		version_id: str,
		metric: str,
		score: float,
		threshold: float,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		version = self._require_version(version_id, tenant_id)
		normalized_score = normalize_score(score)
		normalized_threshold = normalize_score(threshold)
		detected, status = drift_status(normalized_score, normalized_threshold)
		signal = DriftSignal(
			id=signal_id,
			tenant_id=tenant_id,
			model_id=version.model_id,
			version_id=version.id,
			metric=metric,
			score=normalized_score,
			threshold=normalized_threshold,
			drift_detected=detected,
			status=status,
			metadata=dict(metadata or {}),
		)
		self._drift_signals[signal.id] = signal
		self._audit(tenant_id, "drift_recorded", signal.id, f"Recorded drift signal {metric}")
		return signal.to_dict()

	def record_drift_review(
		self,
		signal_id: str,
		tenant_id: str,
		review_ref: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		signal = self._require_drift_signal(signal_id, tenant_id)
		signal.review_recorded = True
		signal.review_ref = review_ref
		signal.status = "reviewed" if signal.drift_detected else "within_threshold"
		self._audit(tenant_id, "drift_review_recorded", signal.id, "Recorded drift review")
		return signal.to_dict()

	def rollback_deployment(
		self,
		rollback_id: str,
		tenant_id: str,
		deployment_id: str,
		to_version_id: str,
		reason: str,
		requested_by: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		deployment = self._require_deployment(deployment_id, tenant_id)
		to_version = self._require_version(to_version_id, tenant_id)
		if deployment.model_id != to_version.model_id:
			raise LookupError("rollback_version_model_mismatch")
		rollback = RollbackRecord(
			id=rollback_id,
			tenant_id=tenant_id,
			model_id=deployment.model_id,
			deployment_id=deployment.id,
			from_version_id=deployment.version_id,
			to_version_id=to_version.id,
			reason=reason,
			requested_by=requested_by,
		)
		deployment.version_id = to_version.id
		deployment.stage = to_version.stage
		deployment.status = "rolled_back"
		self._versions[rollback.from_version_id].status = "rolled_back"
		to_version.status = "serving"
		self._rollbacks[rollback.id] = rollback
		self._audit(tenant_id, "deployment_rolled_back", rollback.id, reason)
		return rollback.to_dict()

	def retire_model(
		self,
		retirement_id: str,
		tenant_id: str,
		model_id: str,
		impact_review_ref: str,
		retired_by: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		model = self._require_model(model_id, tenant_id)
		serving_deployments = [
			deployment
			for deployment in self._deployments.values()
			if deployment.tenant_id == tenant_id
			and deployment.model_id == model.id
			and deployment.status == "serving"
		]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_model",
			"impact_review_recorded": bool(impact_review_ref),
			"serving_deployments_present": bool(serving_deployments),
		})
		self._raise_if_denied(result)
		retirement = RetirementRecord(
			id=retirement_id,
			tenant_id=tenant_id,
			model_id=model.id,
			impact_review_ref=impact_review_ref,
			retired_by=retired_by,
			metadata=dict(metadata or {}),
		)
		model.status = "retired"
		model.updated_at = utc_now_iso()
		for version in self._versions.values():
			if version.tenant_id == tenant_id and version.model_id == model.id:
				version.status = "retired"
		self._retirements[retirement.id] = retirement
		self._audit(tenant_id, "model_retired", retirement.id, f"Retired model {model.id}")
		return retirement.to_dict()

	def register_model_lifecycle_agent(
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
		"""Register a first-class model lifecycle agent with guardrail evidence."""
		self._require_tenant(tenant_id)
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_model_lifecycle_agent",
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
			raise ValueError("model_lifecycle_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		record = ModelLifecycleAgentRecord(
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
		self._agents[self._tenant_record_key(tenant_id, record.id)] = record
		self._audit(
			tenant_id,
			"model_lifecycle_agent_registered",
			record.id,
			f"Registered model lifecycle agent {name}",
			metadata={"runtime": runtime_value, "role": role_value, "status": status, "matched_rules": result["matched_rules"]},
		)
		return record.to_dict()

	def validate_mlcm_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "model_lifecycle_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		"""Validate that MLCM lifecycle mutation batches flow through Bytewax."""
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("mlcm_lifecycle_batch_empty")
		stream_value = self._normalize_token(event_stream)
		operation_value = self._normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_mlcm_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_mlcm_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = MlcmLifecycleBatchRecord(
			id=batch_id or stable_id("mlcmbatch", tenant_id, operation_value, len(self._lifecycle_batches)),
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
		self._audit(
			tenant_id,
			f"mlcm_lifecycle_batch_{record.status}",
			record.id,
			f"MLCM lifecycle batch {record.status}",
			metadata=record.to_dict(),
		)
		if not accepted:
			self._raise_if_denied(result)
		return record.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.register_model(
			model_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "unassigned"),
			problem_type=str(metadata.get("problem_type") or "general"),
			risk_level=str(metadata.get("risk_level") or "medium"),
			description=str(metadata.get("description") or ""),
			tags=list(metadata.get("tags") or []),
			metadata=metadata | {"compatibility_status": status or "registered"},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_models(tenant_id)

	def list_models(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._models, tenant_id)

	def list_versions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._versions, tenant_id)

	def list_evaluations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._evaluations, tenant_id)

	def list_promotion_requests(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._promotion_requests, tenant_id)

	def list_targets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._targets, tenant_id)

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployments, tenant_id)

	def list_drift_signals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._drift_signals, tenant_id)

	def list_rollbacks(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._rollbacks, tenant_id)

	def list_retirements(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._retirements, tenant_id)

	def list_model_lifecycle_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		models = [item for item in self._models.values() if item.tenant_id == tenant_id]
		versions = [item for item in self._versions.values() if item.tenant_id == tenant_id]
		deployments = [item for item in self._deployments.values() if item.tenant_id == tenant_id]
		drift = [item for item in self._drift_signals.values() if item.tenant_id == tenant_id]
		promotions = [item for item in self._promotion_requests.values() if item.tenant_id == tenant_id]
		agents = [item for item in self._agents.values() if item.tenant_id == tenant_id]
		batches = [item for item in self._lifecycle_batches.values() if item.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"model_count": len(models),
			"version_count": len(versions),
			"pending_version_review_count": sum(1 for item in versions if item.status == "pending_review"),
			"deployment_count": len(deployments),
			"serving_count": sum(1 for item in deployments if item.status == "serving"),
			"retired_model_count": sum(1 for item in models if item.status == "retired"),
			"production_version_count": sum(1 for item in versions if item.stage == "production"),
			"pending_promotion_count": sum(1 for item in promotions if item.status == "blocked"),
			"pending_evaluation_review_count": sum(1 for item in self._evaluations.values() if item.tenant_id == tenant_id and item.status == "pending_review"),
			"unresolved_drift_count": sum(1 for item in drift if item.drift_detected and not item.review_recorded),
			"model_lifecycle_agent_count": len(agents),
			"pending_agent_review_count": sum(1 for item in agents if item.status == "pending_review"),
			"lifecycle_batch_count": len(batches),
			"denied_lifecycle_batch_count": sum(1 for item in batches if item.status == "denied"),
			"minimum_eval_score": self.minimum_eval_score,
			"audit_event_count": sum(1 for item in self._audit_events.values() if item.tenant_id == tenant_id),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_model(self, model_id: str, tenant_id: str) -> ModelArtifact:
		model = self._models.get(model_id)
		if model is None or model.tenant_id != tenant_id:
			raise LookupError("model_not_found")
		return model

	def _require_version(self, version_id: str, tenant_id: str) -> ModelVersion:
		version = self._versions.get(version_id)
		if version is None or version.tenant_id != tenant_id:
			raise LookupError("model_version_not_found")
		return version

	def _require_target(self, target_id: str, tenant_id: str) -> DeploymentTarget:
		target = self._targets.get(target_id)
		if target is None or target.tenant_id != tenant_id:
			raise LookupError("deployment_target_not_found")
		return target

	def _require_deployment(self, deployment_id: str, tenant_id: str) -> DeploymentRecord:
		deployment = self._deployments.get(deployment_id)
		if deployment is None or deployment.tenant_id != tenant_id:
			raise LookupError("deployment_not_found")
		return deployment

	def _require_drift_signal(self, signal_id: str, tenant_id: str) -> DriftSignal:
		signal = self._drift_signals.get(signal_id)
		if signal is None or signal.tenant_id != tenant_id:
			raise LookupError("drift_signal_not_found")
		return signal

	def _unresolved_drift(self, version_id: str, tenant_id: str) -> list[DriftSignal]:
		return [
			signal
			for signal in self._drift_signals.values()
			if signal.tenant_id == tenant_id
			and signal.version_id == version_id
			and signal.drift_detected
			and not signal.review_recorded
		]

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(self._reasons(result))

	def _raise_if_review_required(self, result: dict[str, Any]) -> None:
		if result["decision"] == "require_review":
			raise PermissionError(self._reasons(result))

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		severity: str = "info",
		metadata: dict[str, Any] | None = None,
	) -> None:
		event = MlcmAuditEvent(
			id=stable_id("mlcmaudit", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self._audit_events[event.id] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> str:
		return ", ".join(
			action.get("reason", "capability_policy_blocked")
			for action in result.get("actions", [])
		) or "capability_policy_blocked"
