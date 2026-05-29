"""Service layer for APG Continuous Integration and Delivery."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .cicd_engine import CicdEngine
from .models import BuildArtifact, BuildRun, CicdAuditEvent, PipelineDefinition, PromotionRun, QualityGateResult


class CicdService:
	"""Pipeline registry, build monitor, artifact registry, and release gate console."""

	def __init__(self) -> None:
		self._pipelines: dict[str, PipelineDefinition] = {}
		self._builds: dict[str, BuildRun] = {}
		self._artifacts: dict[str, BuildArtifact] = {}
		self._gates: dict[str, QualityGateResult] = {}
		self._promotions: dict[str, PromotionRun] = {}
		self._audit_events: dict[str, CicdAuditEvent] = {}
		self._engine = CicdEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_pipeline(
		self,
		pipeline_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		source_ref: str,
		worker_pool: str,
		stages: list[str] | tuple[str, ...],
		secret_scope: str,
		cache_policy: str,
		quality_gate: str,
		parallel_job_count: int = 1,
		capacity_review_recorded: bool = True,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_pipeline",
			"pipeline_owner_assigned": bool(owner),
			"parallel_job_count": int(parallel_job_count),
			"capacity_review_recorded": bool(capacity_review_recorded),
		})
		self._raise_if_denied(result)
		if not source_ref:
			raise PermissionError("source_policy_required")
		if not worker_pool:
			raise PermissionError("worker_pool_required")
		if not stages:
			raise PermissionError("pipeline_stages_required")
		if not secret_scope:
			raise PermissionError("secret_scope_required")
		if not cache_policy:
			raise PermissionError("cache_policy_required")
		if not quality_gate:
			raise PermissionError("quality_gate_required")
		review_status = "required" if result["decision"] == "require_review" else "approved"
		status = "pending_review" if review_status == "required" else "active"
		pipeline = PipelineDefinition(
			id=pipeline_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			source_ref=source_ref,
			worker_pool=worker_pool,
			stages=tuple(str(stage) for stage in stages),
			secret_scope=secret_scope,
			cache_policy=cache_policy,
			quality_gate=quality_gate,
			parallel_job_count=int(parallel_job_count),
			status=status,
			review_status=review_status,
		)
		self._pipelines[pipeline_id] = pipeline
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=pipeline_id,
			event_type="pipeline_created",
			actor=owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"stage_count": len(stages), "parallel_job_count": parallel_job_count},
		)
		return pipeline.to_dict()

	def approve_pipeline(self, pipeline_id: str, reviewer: str) -> dict[str, Any]:
		pipeline = self._require_pipeline(pipeline_id)
		if pipeline.status != "pending_review":
			return pipeline.to_dict()
		approved = PipelineDefinition(
			id=pipeline.id,
			tenant_id=pipeline.tenant_id,
			name=pipeline.name,
			owner=pipeline.owner,
			source_ref=pipeline.source_ref,
			worker_pool=pipeline.worker_pool,
			stages=pipeline.stages,
			secret_scope=pipeline.secret_scope,
			cache_policy=pipeline.cache_policy,
			quality_gate=pipeline.quality_gate,
			parallel_job_count=pipeline.parallel_job_count,
			status="active",
			review_status="approved",
		)
		self._pipelines[pipeline_id] = approved
		self._record_audit(approved.tenant_id, pipeline_id, "pipeline_review_approved", reviewer, "allow")
		return approved.to_dict()

	def run_build(
		self,
		build_id: str,
		tenant_id: str,
		pipeline_id: str,
		commit_ref: str,
		triggered_by: str,
		secret_scope_attached: bool = True,
		log_trace_captured: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		pipeline = self._require_pipeline(pipeline_id, tenant_id)
		if pipeline.status != "active":
			raise PermissionError("pipeline_not_active")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "run_build",
			"secret_scope_attached": bool(secret_scope_attached),
		})
		self._raise_if_denied(result)
		if not log_trace_captured:
			raise PermissionError("log_trace_capture_required")
		payload = {"build_id": build_id, "pipeline_id": pipeline_id, "commit_ref": commit_ref}
		build = BuildRun(
			id=build_id,
			tenant_id=tenant_id,
			pipeline_id=pipeline_id,
			commit_ref=commit_ref,
			triggered_by=triggered_by,
			trace_id=self._engine.build_trace_id(payload),
			status="passed",
			log_trace_captured=log_trace_captured,
			secret_scope=pipeline.secret_scope,
			cache_policy=pipeline.cache_policy,
		)
		self._builds[build_id] = build
		self._record_audit(tenant_id, build_id, "build_run_completed", triggered_by, result["decision"], metadata={"pipeline_id": pipeline_id})
		return build.to_dict()

	def publish_artifact(
		self,
		artifact_id: str,
		tenant_id: str,
		build_id: str,
		name: str,
		version: str,
		signed: bool,
	) -> dict[str, Any]:
		build = self._require_build(build_id, tenant_id)
		payload = {"artifact_id": artifact_id, "build_id": build_id, "name": name, "version": version}
		artifact = BuildArtifact(
			id=artifact_id,
			tenant_id=tenant_id,
			build_id=build.id,
			name=name,
			version=version,
			digest=self._engine.artifact_digest(payload),
			signed=bool(signed),
		)
		self._artifacts[artifact_id] = artifact
		self._record_audit(tenant_id, artifact_id, "artifact_published", build.triggered_by, "allow", metadata={"signed": signed})
		return artifact.to_dict()

	def record_quality_gate(
		self,
		gate_id: str,
		tenant_id: str,
		artifact_id: str,
		tests_passed: bool,
		security_scan_passed: bool,
		approval_recorded: bool,
	) -> dict[str, Any]:
		artifact = self._require_artifact(artifact_id, tenant_id)
		findings = self._engine.gate_findings(tests_passed, security_scan_passed, artifact.signed, approval_recorded)
		status = "passed" if not findings else "failed"
		gate = QualityGateResult(
			id=gate_id,
			tenant_id=tenant_id,
			artifact_id=artifact_id,
			status=status,
			tests_passed=bool(tests_passed),
			security_scan_passed=bool(security_scan_passed),
			artifact_signed=artifact.signed,
			approval_recorded=bool(approval_recorded),
			findings=findings,
		)
		self._gates[gate_id] = gate
		self._record_audit(tenant_id, gate_id, "quality_gate_recorded", "quality-gate", status, metadata={"findings": list(findings)})
		return gate.to_dict()

	def promote_artifact(
		self,
		promotion_id: str,
		tenant_id: str,
		artifact_id: str,
		quality_gate_id: str,
		source_environment: str,
		target_environment: str,
		requested_by: str,
		approval_recorded: bool,
	) -> dict[str, Any]:
		artifact = self._require_artifact(artifact_id, tenant_id)
		gate = self._require_gate(quality_gate_id, tenant_id)
		if gate.artifact_id != artifact.id:
			raise PermissionError("quality_gate_artifact_mismatch")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "promote_artifact",
			"artifact_promotion_requested": True,
			"artifact_signed": artifact.signed,
			"quality_gate_passed": gate.status == "passed",
		})
		self._raise_if_denied(result)
		if not approval_recorded:
			raise PermissionError("promotion_approval_required")
		promotion = PromotionRun(
			id=promotion_id,
			tenant_id=tenant_id,
			artifact_id=artifact_id,
			source_environment=source_environment,
			target_environment=target_environment,
			requested_by=requested_by,
			status="promoted",
			quality_gate_id=quality_gate_id,
			approval_recorded=approval_recorded,
		)
		self._promotions[promotion_id] = promotion
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=promotion_id,
			event_type="artifact_promoted",
			actor=requested_by,
			decision=result["decision"],
			metadata={"artifact_id": artifact_id, "target_environment": target_environment},
		)
		return promotion.to_dict()

	def list_pipelines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._pipelines, tenant_id)

	def list_builds(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._builds, tenant_id)

	def list_artifacts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._artifacts, tenant_id)

	def list_gates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._gates, tenant_id)

	def list_promotions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._promotions, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing build runs as CICD records."""
		return self.list_builds(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that records an auditable CI/CD event."""
		self._require_tenant(tenant_id)
		event = self._record_audit(
			tenant_id=tenant_id,
			subject_id=record_id,
			event_type=str((metadata or {}).get("event_type") or "cicd_note"),
			actor=str((metadata or {}).get("actor") or "system"),
			decision=status,
			metadata=dict(metadata or {}),
		)
		return event.to_dict()

	def pipeline_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		pipelines = self.list_pipelines(tenant_id)
		builds = self.list_builds(tenant_id)
		gates = self.list_gates(tenant_id)
		return {
			"pipeline_count": len(pipelines),
			"active_pipeline_count": len([item for item in pipelines if item["status"] == "active"]),
			"pending_review_pipeline_count": len([item for item in pipelines if item["status"] == "pending_review"]),
			"build_count": len(builds),
			"passed_build_count": len([item for item in builds if item["status"] == "passed"]),
			"artifact_count": len(self.list_artifacts(tenant_id)),
			"passed_gate_count": len([item for item in gates if item["status"] == "passed"]),
			"failed_gate_count": len([item for item in gates if item["status"] == "failed"]),
			"promotion_count": len(self.list_promotions(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _list(self, values: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_pipeline(self, pipeline_id: str, tenant_id: str | None = None) -> PipelineDefinition:
		pipeline = self._pipelines.get(pipeline_id)
		if pipeline is None or (tenant_id is not None and pipeline.tenant_id != tenant_id):
			raise KeyError(f"unknown pipeline: {pipeline_id}")
		return pipeline

	def _require_build(self, build_id: str, tenant_id: str) -> BuildRun:
		build = self._builds.get(build_id)
		if build is None or build.tenant_id != tenant_id:
			raise KeyError(f"unknown build: {build_id}")
		return build

	def _require_artifact(self, artifact_id: str, tenant_id: str) -> BuildArtifact:
		artifact = self._artifacts.get(artifact_id)
		if artifact is None or artifact.tenant_id != tenant_id:
			raise KeyError(f"unknown artifact: {artifact_id}")
		return artifact

	def _require_gate(self, gate_id: str, tenant_id: str) -> QualityGateResult:
		gate = self._gates.get(gate_id)
		if gate is None or gate.tenant_id != tenant_id:
			raise KeyError(f"unknown quality gate: {gate_id}")
		return gate

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> CicdAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = CicdAuditEvent(
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

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "cicd_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "cicd_policy_blocked")
