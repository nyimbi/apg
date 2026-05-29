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
	ModelArtifact,
	ModelVersion,
	PromotionRequest,
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
		self._audit_events: dict[str, MlcmAuditEvent] = {}

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
		version_record = ModelVersion(
			id=version_id,
			tenant_id=tenant_id,
			model_id=model.id,
			version=version,
			artifact_uri=artifact_uri,
			stage=normalize_stage(stage),
			model_card=dict(model_card or {}),
			training_data_ref=training_data_ref,
			baseline_ref=baseline_ref,
			metadata=dict(metadata or {}),
		)
		self._versions[version_record.id] = version_record
		self._audit(tenant_id, "model_version_created", version_record.id, f"Created version {version}")
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
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		version = self._require_version(version_id, tenant_id)
		normalized_score = normalize_score(score)
		evaluation = EvaluationRun(
			id=evaluation_id,
			tenant_id=tenant_id,
			model_id=version.model_id,
			version_id=version.id,
			score=normalized_score,
			baseline_ref=baseline_ref,
			metrics=dict(metrics or {}),
			status=evaluation_status(normalized_score, self.minimum_eval_score),
			evidence_refs=list(evidence_refs or []),
			evaluator=evaluator,
		)
		self._evaluations[evaluation.id] = evaluation
		version.evaluation_score = normalized_score
		version.evaluation_id = evaluation.id
		version.baseline_ref = baseline_ref or version.baseline_ref
		version.status = "candidate" if evaluation.status == "passed" else "needs_improvement"
		self._audit(tenant_id, "model_evaluated", evaluation.id, f"Recorded evaluation score {normalized_score:.3f}")
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

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		models = [item for item in self._models.values() if item.tenant_id == tenant_id]
		versions = [item for item in self._versions.values() if item.tenant_id == tenant_id]
		deployments = [item for item in self._deployments.values() if item.tenant_id == tenant_id]
		drift = [item for item in self._drift_signals.values() if item.tenant_id == tenant_id]
		promotions = [item for item in self._promotion_requests.values() if item.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"model_count": len(models),
			"version_count": len(versions),
			"deployment_count": len(deployments),
			"serving_count": sum(1 for item in deployments if item.status == "serving"),
			"production_version_count": sum(1 for item in versions if item.stage == "production"),
			"pending_promotion_count": sum(1 for item in promotions if item.status == "blocked"),
			"unresolved_drift_count": sum(1 for item in drift if item.drift_detected and not item.review_recorded),
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
