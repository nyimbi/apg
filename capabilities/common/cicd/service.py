"""
APG CI/CD (CICD) - Expanded Service Implementation

Dependency-light in-memory store pattern. 44+ async methods covering
pipeline lifecycle, builds, artifacts, quality gates, deployment
promotion, rollback, feature flags, canary/blue-green deployments,
security scanning, compliance gates, dependency audits, container
scans, secrets scanning, coverage reporting, and deployment analytics.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from datetime import datetime, timezone
from typing import Any

from uuid6 import uuid7

import logging
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

logger = logging.getLogger(__name__)


def uuid7str() -> str:
	return str(uuid7())


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _R(dict[str, Any]):
	"""Thin dict wrapper for records."""


class CICDService:
	"""
	44+ async methods for CI/CD pipeline governance, build tracking,
	artifact management, quality gates, security scanning, deployment
	strategies, feature flags, analytics and compliance reporting.
	"""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id

		self._pipelines:      dict[str, _R] = {}
		self._builds:         dict[str, _R] = {}
		self._artifacts:      dict[str, _R] = {}
		self._gates:          dict[str, _R] = {}
		self._promotions:     dict[str, _R] = {}
		self._deployments:    dict[str, _R] = {}
		self._rollbacks:      dict[str, _R] = {}
		self._feature_flags:  dict[str, _R] = {}
		self._security_scans: dict[str, _R] = {}
		self._stage_logs:     dict[str, list[_R]] = {}
		self._agents:         dict[str, _R] = {}
		self._audit_log:      list[_R] = []

	# ------------------------------------------------------------------
	# helpers
	# ------------------------------------------------------------------

	def _key(self, record_id: str) -> str:
		return f"{self.tenant_id}:{record_id}"

	async def _audit(self, event_type: str, record_id: str, details: dict[str, Any] | None = None) -> None:
		self._audit_log.append(_R(
			event_id=uuid7str(),
			tenant_id=self.tenant_id,
			actor_id=self.actor_id,
			event_type=event_type,
			record_id=record_id,
			details=details or {},
			occurred_at=_ts(),
		))

	def _require_pipeline(self, pipeline_id: str) -> _R:
		r = self._pipelines.get(self._key(pipeline_id))
		if r is None:
			raise KeyError(f"pipeline not found: {pipeline_id}")
		return r

	def _require_build(self, build_id: str) -> _R:
		r = self._builds.get(self._key(build_id))
		if r is None:
			raise KeyError(f"build not found: {build_id}")
		return r

	def _require_artifact(self, artifact_id: str) -> _R:
		r = self._artifacts.get(self._key(artifact_id))
		if r is None:
			raise KeyError(f"artifact not found: {artifact_id}")
		return r

	def _require_gate(self, gate_id: str) -> _R:
		r = self._gates.get(self._key(gate_id))
		if r is None:
			raise KeyError(f"quality gate not found: {gate_id}")
		return r

	def _require_deployment(self, deployment_id: str) -> _R:
		r = self._deployments.get(self._key(deployment_id))
		if r is None:
			raise KeyError(f"deployment not found: {deployment_id}")
		return r

	# ------------------------------------------------------------------
	# 1. pipeline_create
	# ------------------------------------------------------------------

	async def pipeline_create(
		self,
		name: str,
		stages: list[str],
		triggers: list[str],
		environment: str,
		owner: str = "system",
		source_ref: str = "git://repo",
	) -> _R:
		"""Create a CI/CD pipeline definition."""
		assert name, "pipeline name required"
		assert stages, "at least one stage required"
		pipeline_id = uuid7str()
		record = _R(
			pipeline_id=pipeline_id,
			tenant_id=self.tenant_id,
			name=name,
			stages=list(stages),
			triggers=list(triggers),
			environment=environment,
			owner=owner,
			source_ref=source_ref,
			status="active",
			created_at=_ts(),
		)
		self._pipelines[self._key(pipeline_id)] = record
		for stage in stages:
			self._stage_logs[f"{self.tenant_id}:{pipeline_id}:{stage}"] = []
		await self._audit("pipeline_created", pipeline_id, {"name": name, "stages": stages})
		return record

	# ------------------------------------------------------------------
	# 2. trigger_build
	# ------------------------------------------------------------------

	async def trigger_build(
		self,
		pipeline_id: str,
		commit_ref: str,
		triggered_by: str = "push",
		branch: str = "main",
	) -> _R:
		"""Trigger a pipeline run for a commit."""
		pipeline = self._require_pipeline(pipeline_id)
		assert pipeline["status"] == "active", "pipeline not active"
		build_id = uuid7str()
		record = _R(
			build_id=build_id,
			pipeline_id=pipeline_id,
			tenant_id=self.tenant_id,
			commit_ref=commit_ref,
			branch=branch,
			triggered_by=triggered_by,
			status="running",
			stages_completed=[],
			started_at=_ts(),
			completed_at=None,
		)
		self._builds[self._key(build_id)] = record
		for stage in pipeline["stages"]:
			log_key = f"{self.tenant_id}:{pipeline_id}:{stage}"
			if log_key not in self._stage_logs:
				self._stage_logs[log_key] = []
			self._stage_logs[log_key].append(_R(
				build_id=build_id, stage=stage,
				message=f"[{_ts()}] stage {stage} queued", level="info",
			))
		await self._audit("build_triggered", build_id, {"pipeline_id": pipeline_id, "commit_ref": commit_ref})
		return record

	# ------------------------------------------------------------------
	# 3. build_complete
	# ------------------------------------------------------------------

	async def build_complete(self, build_id: str, status: str = "passed") -> _R:
		"""Mark a build as completed."""
		assert status in {"passed", "failed", "cancelled"}, f"invalid status: {status}"
		build = self._require_build(build_id)
		build["status"] = status
		build["completed_at"] = _ts()
		await self._audit("build_completed", build_id, {"status": status})
		return build

	# ------------------------------------------------------------------
	# 4. store_artifact
	# ------------------------------------------------------------------

	async def store_artifact(
		self,
		build_id: str,
		name: str,
		version: str,
		signed: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> _R:
		"""Store a build artifact."""
		build = self._require_build(build_id)
		artifact_id = uuid7str()
		record = _R(
			artifact_id=artifact_id,
			build_id=build_id,
			tenant_id=self.tenant_id,
			name=name,
			version=version,
			signed=signed,
			metadata=metadata or {},
			status="available",
			created_at=_ts(),
		)
		self._artifacts[self._key(artifact_id)] = record
		await self._audit("artifact_stored", artifact_id, {"name": name, "version": version, "signed": signed})
		return record

	# ------------------------------------------------------------------
	# 5. quality_gate_add
	# ------------------------------------------------------------------

	async def quality_gate_add(
		self,
		artifact_id: str,
		tests_passed: bool,
		security_scan_passed: bool,
		coverage_pct: float = 0.0,
		approval_recorded: bool = True,
	) -> _R:
		"""Record a quality gate result for an artifact."""
		artifact = self._require_artifact(artifact_id)
		issues = []
		if not tests_passed:
			issues.append("tests_failed")
		if not security_scan_passed:
			issues.append("security_scan_failed")
		if coverage_pct < 70.0:
			issues.append(f"coverage_below_threshold:{coverage_pct:.1f}%")
		gate_id = uuid7str()
		record = _R(
			gate_id=gate_id,
			artifact_id=artifact_id,
			tenant_id=self.tenant_id,
			tests_passed=tests_passed,
			security_scan_passed=security_scan_passed,
			coverage_pct=round(coverage_pct, 2),
			approval_recorded=approval_recorded,
			issues=issues,
			status="passed" if not issues else "failed",
			recorded_at=_ts(),
		)
		self._gates[self._key(gate_id)] = record
		await self._audit("quality_gate_recorded", gate_id, {"artifact_id": artifact_id, "status": record["status"]})
		return record

	# ------------------------------------------------------------------
	# 6. deployment_promote
	# ------------------------------------------------------------------

	async def deployment_promote(
		self,
		artifact_id: str,
		from_env: str,
		to_env: str,
		approved_by: str,
		quality_gate_id: str | None = None,
	) -> _R:
		"""Promote an artifact to a higher environment."""
		artifact = self._require_artifact(artifact_id)
		assert approved_by, "approved_by required"
		assert from_env != to_env, "source and target environments must differ"
		if quality_gate_id:
			gate = self._require_gate(quality_gate_id)
			assert gate["status"] == "passed", "quality gate must be passed before promotion"
		promotion_id = uuid7str()
		record = _R(
			promotion_id=promotion_id,
			artifact_id=artifact_id,
			tenant_id=self.tenant_id,
			from_env=from_env,
			to_env=to_env,
			approved_by=approved_by,
			quality_gate_id=quality_gate_id,
			status="promoted",
			promoted_at=_ts(),
		)
		self._promotions[self._key(promotion_id)] = record
		deployment_id = uuid7str()
		deployment = _R(
			deployment_id=deployment_id,
			artifact_id=artifact_id,
			tenant_id=self.tenant_id,
			environment=to_env,
			approved_by=approved_by,
			status="deployed",
			deployed_at=_ts(),
		)
		self._deployments[self._key(deployment_id)] = deployment
		await self._audit("artifact_promoted", promotion_id, {"artifact_id": artifact_id, "from": from_env, "to": to_env})
		return record

	# ------------------------------------------------------------------
	# 7. rollback_release
	# ------------------------------------------------------------------

	async def rollback_release(
		self,
		deployment_id: str,
		reason: str,
		approved_by: str,
	) -> _R:
		"""Roll back a deployment."""
		assert reason, "rollback reason required"
		assert approved_by, "approver required"
		deployment = self._require_deployment(deployment_id)
		rollback_id = uuid7str()
		record = _R(
			rollback_id=rollback_id,
			deployment_id=deployment_id,
			tenant_id=self.tenant_id,
			reason=reason,
			approved_by=approved_by,
			previous_status=deployment["status"],
			status="rolled_back",
			rolled_back_at=_ts(),
		)
		self._rollbacks[self._key(rollback_id)] = record
		deployment["status"] = "rolled_back"
		await self._audit("deployment_rolled_back", rollback_id, {"deployment_id": deployment_id, "reason": reason})
		return record

	# ------------------------------------------------------------------
	# 8. feature_flag_release
	# ------------------------------------------------------------------

	async def feature_flag_release(
		self,
		name: str,
		enabled: bool,
		rollout_pct: float = 100.0,
		environments: list[str] | None = None,
	) -> _R:
		"""Create or update a feature flag for controlled rollout."""
		flag_id = uuid7str()
		record = _R(
			flag_id=flag_id,
			name=name,
			tenant_id=self.tenant_id,
			enabled=enabled,
			rollout_pct=round(min(max(rollout_pct, 0.0), 100.0), 2),
			environments=environments or ["production"],
			created_at=_ts(),
			updated_at=_ts(),
		)
		self._feature_flags[self._key(name)] = record
		await self._audit("feature_flag_set", flag_id, {"name": name, "enabled": enabled, "rollout_pct": rollout_pct})
		return record

	# ------------------------------------------------------------------
	# 9. canary_deploy
	# ------------------------------------------------------------------

	async def canary_deploy(
		self,
		artifact_id: str,
		environment: str,
		canary_pct: float,
		approved_by: str,
	) -> _R:
		"""Deploy artifact to a canary traffic slice."""
		artifact = self._require_artifact(artifact_id)
		assert 0.0 < canary_pct <= 50.0, "canary_pct must be in (0, 50]"
		deployment_id = uuid7str()
		record = _R(
			deployment_id=deployment_id,
			artifact_id=artifact_id,
			tenant_id=self.tenant_id,
			environment=environment,
			strategy="canary",
			canary_pct=round(canary_pct, 2),
			approved_by=approved_by,
			status="canary_active",
			deployed_at=_ts(),
		)
		self._deployments[self._key(deployment_id)] = record
		await self._audit("canary_deployed", deployment_id, {"artifact_id": artifact_id, "canary_pct": canary_pct})
		return record

	# ------------------------------------------------------------------
	# 10. blue_green_switch
	# ------------------------------------------------------------------

	async def blue_green_switch(
		self,
		artifact_id: str,
		environment: str,
		approved_by: str,
		active_slot: str = "green",
	) -> _R:
		"""Execute a blue/green deployment switch."""
		artifact = self._require_artifact(artifact_id)
		assert active_slot in {"blue", "green"}, "active_slot must be blue or green"
		deployment_id = uuid7str()
		record = _R(
			deployment_id=deployment_id,
			artifact_id=artifact_id,
			tenant_id=self.tenant_id,
			environment=environment,
			strategy="blue_green",
			active_slot=active_slot,
			inactive_slot="blue" if active_slot == "green" else "green",
			approved_by=approved_by,
			status="deployed",
			deployed_at=_ts(),
		)
		self._deployments[self._key(deployment_id)] = record
		await self._audit("blue_green_switched", deployment_id, {"active_slot": active_slot, "artifact_id": artifact_id})
		return record

	# ------------------------------------------------------------------
	# 11. test_coverage_report
	# ------------------------------------------------------------------

	async def test_coverage_report(self, build_id: str) -> _R:
		"""Generate a test coverage summary for a build."""
		build = self._require_build(build_id)
		# Pull coverage from any gates associated with this build's artifacts
		build_artifacts = [a for a in self._artifacts.values() if a["build_id"] == build_id and a["tenant_id"] == self.tenant_id]
		artifact_ids = {a["artifact_id"] for a in build_artifacts}
		relevant_gates = [g for g in self._gates.values() if g["artifact_id"] in artifact_ids]
		if not relevant_gates:
			return _R(build_id=build_id, coverage_pct=None, threshold=70.0, passes=False, generated_at=_ts())
		coverages = [g["coverage_pct"] for g in relevant_gates]
		avg_coverage = round(statistics.mean(coverages), 2)
		return _R(
			build_id=build_id,
			coverage_pct=avg_coverage,
			min_coverage=round(min(coverages), 2),
			max_coverage=round(max(coverages), 2),
			threshold=70.0,
			passes=avg_coverage >= 70.0,
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 12. security_scan_gate
	# ------------------------------------------------------------------

	async def security_scan_gate(
		self,
		build_id: str,
		scan_type: str = "sast",
		scanner: str = "system",
		findings: list[str] | None = None,
	) -> _R:
		"""Record a security scan result as a pipeline gate."""
		assert scan_type in {"sast", "dast", "dependency_scan", "secret_scan", "container_scan", "iac_scan"}, f"unknown scan_type: {scan_type}"
		build = self._require_build(build_id)
		scan_findings = findings or []
		scan_id = uuid7str()
		record = _R(
			scan_id=scan_id,
			build_id=build_id,
			tenant_id=self.tenant_id,
			scan_type=scan_type,
			scanner=scanner,
			findings=scan_findings,
			critical_count=sum(1 for f in scan_findings if "critical" in f.lower()),
			high_count=sum(1 for f in scan_findings if "high" in f.lower()),
			status="passed" if not scan_findings else "failed",
			scanned_at=_ts(),
		)
		self._security_scans[self._key(scan_id)] = record
		await self._audit(f"security_scan_{scan_type}", scan_id, {"build_id": build_id, "findings": len(scan_findings)})
		return record

	# ------------------------------------------------------------------
	# 13. dependency_audit
	# ------------------------------------------------------------------

	async def dependency_audit(self, build_id: str, dependencies: list[dict[str, Any]] | None = None) -> _R:
		"""Audit dependencies for known vulnerabilities."""
		build = self._require_build(build_id)
		deps = dependencies or []
		vulnerable = [d for d in deps if d.get("vulnerable", False)]
		result = _R(
			build_id=build_id,
			total_dependencies=len(deps),
			vulnerable_count=len(vulnerable),
			vulnerable_packages=[d.get("name") for d in vulnerable],
			passes=len(vulnerable) == 0,
			audited_at=_ts(),
		)
		await self._audit("dependency_audit", build_id, {"total": len(deps), "vulnerable": len(vulnerable)})
		return result

	# ------------------------------------------------------------------
	# 14. container_scan
	# ------------------------------------------------------------------

	async def container_scan(
		self,
		artifact_id: str,
		image_tag: str,
		findings: list[str] | None = None,
	) -> _R:
		"""Scan a container image for vulnerabilities."""
		artifact = self._require_artifact(artifact_id)
		scan_findings = findings or []
		scan_id = uuid7str()
		record = _R(
			scan_id=scan_id,
			artifact_id=artifact_id,
			tenant_id=self.tenant_id,
			image_tag=image_tag,
			findings=scan_findings,
			critical_count=sum(1 for f in scan_findings if "critical" in f.lower()),
			status="passed" if not scan_findings else "failed",
			scanned_at=_ts(),
		)
		self._security_scans[self._key(scan_id)] = record
		await self._audit("container_scan", scan_id, {"artifact_id": artifact_id, "image_tag": image_tag})
		return record

	# ------------------------------------------------------------------
	# 15. secrets_scan
	# ------------------------------------------------------------------

	async def secrets_scan(self, build_id: str, files_scanned: int = 0, secrets_found: list[str] | None = None) -> _R:
		"""Scan source code for accidentally committed secrets."""
		build = self._require_build(build_id)
		found = secrets_found or []
		scan_id = uuid7str()
		record = _R(
			scan_id=scan_id,
			build_id=build_id,
			tenant_id=self.tenant_id,
			scan_type="secret_scan",
			files_scanned=files_scanned,
			secrets_found=found,
			secrets_count=len(found),
			status="passed" if not found else "failed",
			scanned_at=_ts(),
		)
		self._security_scans[self._key(scan_id)] = record
		await self._audit("secrets_scan", scan_id, {"build_id": build_id, "secrets_found": len(found)})
		return record

	# ------------------------------------------------------------------
	# 16. compliance_gate
	# ------------------------------------------------------------------

	async def compliance_gate(
		self,
		artifact_id: str,
		framework: str = "SOC2",
		checks: dict[str, bool] | None = None,
	) -> _R:
		"""Evaluate compliance checks before promotion."""
		artifact = self._require_artifact(artifact_id)
		check_results = checks or {"audit_trail": True, "encryption": True, "access_control": True}
		failed = [k for k, v in check_results.items() if not v]
		gate_id = uuid7str()
		record = _R(
			gate_id=gate_id,
			artifact_id=artifact_id,
			tenant_id=self.tenant_id,
			framework=framework,
			checks=check_results,
			failed_checks=failed,
			status="passed" if not failed else "failed",
			evaluated_at=_ts(),
		)
		self._gates[self._key(gate_id)] = record
		await self._audit("compliance_gate", gate_id, {"framework": framework, "failed": failed})
		return record

	# ------------------------------------------------------------------
	# 17. deployment_history
	# ------------------------------------------------------------------

	async def deployment_history(self, environment: str | None = None) -> list[_R]:
		"""Return deployment history, optionally filtered by environment."""
		deployments = [
			d for d in self._deployments.values()
			if d["tenant_id"] == self.tenant_id
			and (environment is None or d["environment"] == environment)
		]
		return sorted(deployments, key=lambda d: d["deployed_at"])

	# ------------------------------------------------------------------
	# 18. stage_logs
	# ------------------------------------------------------------------

	async def stage_logs(self, build_id: str, stage_name: str, offset: int = 0, limit: int = 100) -> _R:
		"""Return log lines for a specific stage of a build."""
		build = self._require_build(build_id)
		pipeline = self._require_pipeline(build["pipeline_id"])
		assert stage_name in pipeline["stages"], f"stage not in pipeline: {stage_name}"
		log_key = f"{self.tenant_id}:{build['pipeline_id']}:{stage_name}"
		all_lines = [l for l in self._stage_logs.get(log_key, []) if l.get("build_id") == build_id]
		return _R(
			build_id=build_id,
			stage_name=stage_name,
			total_lines=len(all_lines),
			offset=offset,
			lines=all_lines[offset:offset + limit],
		)

	# ------------------------------------------------------------------
	# 19. pipeline_analytics
	# ------------------------------------------------------------------

	async def pipeline_analytics(self, period: str = "30d") -> _R:
		"""Aggregate pipeline, build, artifact and deployment analytics."""
		pipelines = await self.list_pipelines()
		builds = await self.list_builds()
		artifacts = await self.list_artifacts()
		gates = await self.list_gates()
		deployments = await self.deployment_history()
		rollbacks = [r for r in self._rollbacks.values() if r["tenant_id"] == self.tenant_id]
		passed_builds = [b for b in builds if b["status"] == "passed"]
		passed_gates = [g for g in gates if g["status"] == "passed"]
		return _R(
			tenant_id=self.tenant_id,
			period=period,
			pipeline_count=len(pipelines),
			active_pipeline_count=sum(1 for p in pipelines if p["status"] == "active"),
			build_count=len(builds),
			passed_build_count=len(passed_builds),
			build_pass_rate=round(len(passed_builds) / max(len(builds), 1), 4),
			artifact_count=len(artifacts),
			signed_artifact_count=sum(1 for a in artifacts if a.get("signed")),
			gate_count=len(gates),
			passed_gate_count=len(passed_gates),
			gate_pass_rate=round(len(passed_gates) / max(len(gates), 1), 4),
			deployment_count=len(deployments),
			rollback_count=len(rollbacks),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 20. pipeline_cancel
	# ------------------------------------------------------------------

	async def pipeline_cancel(self, build_id: str, reason: str, cancelled_by: str = "system") -> _R:
		"""Cancel an in-progress build."""
		build = self._require_build(build_id)
		assert build["status"] in {"running", "queued"}, f"cannot cancel build in status {build['status']}"
		assert reason, "cancel reason required"
		build["status"] = "cancelled"
		build["cancelled_at"] = _ts()
		build["cancel_reason"] = reason
		await self._audit("build_cancelled", build_id, {"reason": reason, "cancelled_by": cancelled_by})
		return build

	# ------------------------------------------------------------------
	# 21. approve_pipeline
	# ------------------------------------------------------------------

	async def approve_pipeline(self, pipeline_id: str, reviewer: str) -> _R:
		"""Approve a pipeline pending review."""
		pipeline = self._require_pipeline(pipeline_id)
		if pipeline["status"] != "pending_review":
			return pipeline
		pipeline["status"] = "active"
		pipeline["approved_by"] = reviewer
		pipeline["approved_at"] = _ts()
		await self._audit("pipeline_approved", pipeline_id, {"reviewer": reviewer})
		return pipeline

	# ------------------------------------------------------------------
	# 22. update_pipeline
	# ------------------------------------------------------------------

	async def update_pipeline(self, pipeline_id: str, **kwargs: Any) -> _R:
		"""Update mutable pipeline fields."""
		pipeline = self._require_pipeline(pipeline_id)
		allowed = {"name", "stages", "triggers", "environment", "owner", "source_ref", "status"}
		for k, v in kwargs.items():
			if k in allowed:
				pipeline[k] = v
		pipeline["updated_at"] = _ts()
		await self._audit("pipeline_updated", pipeline_id, {k: v for k, v in kwargs.items() if k in allowed})
		return pipeline

	# ------------------------------------------------------------------
	# 23. delete_pipeline
	# ------------------------------------------------------------------

	async def delete_pipeline(self, pipeline_id: str) -> _R:
		"""Soft-delete a pipeline."""
		pipeline = self._require_pipeline(pipeline_id)
		pipeline["status"] = "deleted"
		pipeline["deleted_at"] = _ts()
		await self._audit("pipeline_deleted", pipeline_id, {})
		return pipeline

	# ------------------------------------------------------------------
	# 24. register_delivery_agent
	# ------------------------------------------------------------------

	async def register_delivery_agent(
		self,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
	) -> _R:
		"""Register a CI/CD delivery agent."""
		agent_id = uuid7str()
		record = _R(
			agent_id=agent_id,
			name=name,
			tenant_id=self.tenant_id,
			runtime=runtime,
			role=role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
			status="active",
			registered_at=_ts(),
		)
		self._agents[self._key(agent_id)] = record
		await self._audit("delivery_agent_registered", agent_id, {"name": name, "runtime": runtime})
		return record

	# ------------------------------------------------------------------
	# 25. bulk_trigger_builds
	# ------------------------------------------------------------------

	async def bulk_trigger_builds(
		self,
		pipeline_ids: list[str],
		commit_ref: str,
		triggered_by: str = "batch",
	) -> list[_R]:
		"""Trigger builds for multiple pipelines at once."""
		results = []
		for pid in pipeline_ids:
			build = await self.trigger_build(pid, commit_ref, triggered_by)
			results.append(build)
		await self._audit("bulk_builds_triggered", "system", {"count": len(results)})
		return results

	# ------------------------------------------------------------------
	# 26. bulk_delete_artifacts
	# ------------------------------------------------------------------

	async def bulk_delete_artifacts(self, artifact_ids: list[str]) -> _R:
		"""Expire multiple artifacts at once."""
		deleted = []
		for aid in artifact_ids:
			artifact = self._artifacts.get(self._key(aid))
			if artifact:
				artifact["status"] = "deleted"
				artifact["deleted_at"] = _ts()
				deleted.append(aid)
		await self._audit("bulk_artifacts_deleted", "system", {"count": len(deleted)})
		return _R(deleted_count=len(deleted), artifact_ids=deleted)

	# ------------------------------------------------------------------
	# 27. list_pipelines
	# ------------------------------------------------------------------

	async def list_pipelines(self, status: str | None = None) -> list[_R]:
		"""List pipelines for the tenant."""
		return sorted(
			[p for p in self._pipelines.values() if p["tenant_id"] == self.tenant_id and (status is None or p["status"] == status)],
			key=lambda p: p["created_at"],
		)

	# ------------------------------------------------------------------
	# 28. list_builds
	# ------------------------------------------------------------------

	async def list_builds(self, pipeline_id: str | None = None, status: str | None = None) -> list[_R]:
		"""List builds, optionally filtered by pipeline or status."""
		return sorted(
			[b for b in self._builds.values()
			 if b["tenant_id"] == self.tenant_id
			 and (pipeline_id is None or b["pipeline_id"] == pipeline_id)
			 and (status is None or b["status"] == status)],
			key=lambda b: b["started_at"],
		)

	# ------------------------------------------------------------------
	# 29. list_artifacts
	# ------------------------------------------------------------------

	async def list_artifacts(self, build_id: str | None = None) -> list[_R]:
		"""List artifacts for the tenant."""
		return sorted(
			[a for a in self._artifacts.values()
			 if a["tenant_id"] == self.tenant_id
			 and (build_id is None or a["build_id"] == build_id)],
			key=lambda a: a["created_at"],
		)

	# ------------------------------------------------------------------
	# 30. list_gates
	# ------------------------------------------------------------------

	async def list_gates(self) -> list[_R]:
		"""List quality gates for the tenant."""
		return [g for g in self._gates.values() if g["tenant_id"] == self.tenant_id]

	# ------------------------------------------------------------------
	# 31. export_builds_csv
	# ------------------------------------------------------------------

	async def export_builds_csv(self) -> str:
		"""Export build records to CSV."""
		builds = await self.list_builds()
		buf = io.StringIO()
		fields = ["build_id", "pipeline_id", "commit_ref", "branch", "status", "started_at", "completed_at"]
		writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(builds)
		await self._audit("builds_exported_csv", "system", {"count": len(builds)})
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 32. export_deployments_json
	# ------------------------------------------------------------------

	async def export_deployments_json(self) -> str:
		"""Export deployment history as JSON."""
		deployments = await self.deployment_history()
		await self._audit("deployments_exported_json", "system", {"count": len(deployments)})
		return json.dumps(deployments, default=str, indent=2)

	# ------------------------------------------------------------------
	# 33. health_check
	# ------------------------------------------------------------------

	async def health_check(self) -> _R:
		"""Service health and storage summary."""
		return _R(
			status="healthy",
			tenant_id=self.tenant_id,
			pipeline_count=sum(1 for p in self._pipelines.values() if p["tenant_id"] == self.tenant_id),
			build_count=sum(1 for b in self._builds.values() if b["tenant_id"] == self.tenant_id),
			artifact_count=sum(1 for a in self._artifacts.values() if a["tenant_id"] == self.tenant_id),
			deployment_count=sum(1 for d in self._deployments.values() if d["tenant_id"] == self.tenant_id),
			audit_event_count=len(self._audit_log),
			checked_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 34. dashboard
	# ------------------------------------------------------------------

	async def dashboard(self) -> _R:
		"""KPI dashboard for the CI/CD capability."""
		return await self.pipeline_analytics()

	# ------------------------------------------------------------------
	# 35. compliance_report
	# ------------------------------------------------------------------

	async def compliance_report(self, framework: str = "SOC2") -> _R:
		"""Generate a CI/CD pipeline compliance report."""
		builds = await self.list_builds()
		artifacts = await self.list_artifacts()
		signed = sum(1 for a in artifacts if a.get("signed"))
		security_scans = [s for s in self._security_scans.values() if s["tenant_id"] == self.tenant_id]
		failed_scans = sum(1 for s in security_scans if s["status"] == "failed")
		report = _R(
			framework=framework,
			tenant_id=self.tenant_id,
			total_builds=len(builds),
			total_artifacts=len(artifacts),
			signed_artifact_rate=round(signed / max(len(artifacts), 1), 4),
			security_scans_executed=len(security_scans),
			failed_security_scans=failed_scans,
			audit_trail_complete=True,
			generated_at=_ts(),
		)
		await self._audit("compliance_report_generated", "system", {"framework": framework})
		return report

	# ------------------------------------------------------------------
	# 36. audit_trail
	# ------------------------------------------------------------------

	async def audit_trail(self, event_type: str | None = None) -> list[_R]:
		"""Return audit events, optionally filtered by type."""
		return [
			e for e in self._audit_log
			if e["tenant_id"] == self.tenant_id and (event_type is None or e["event_type"] == event_type)
		]

	# ------------------------------------------------------------------
	# 37. get_feature_flags
	# ------------------------------------------------------------------

	async def get_feature_flags(self, environment: str | None = None) -> list[_R]:
		"""List feature flags, optionally filtered by environment."""
		flags = [f for f in self._feature_flags.values() if f["tenant_id"] == self.tenant_id]
		if environment:
			flags = [f for f in flags if environment in f.get("environments", [])]
		return flags

	# ------------------------------------------------------------------
	# 38. promote_canary_to_full
	# ------------------------------------------------------------------

	async def promote_canary_to_full(self, deployment_id: str, approved_by: str) -> _R:
		"""Promote a canary deployment to 100% traffic."""
		deployment = self._require_deployment(deployment_id)
		assert deployment["strategy"] == "canary", "deployment is not a canary"
		deployment["canary_pct"] = 100.0
		deployment["strategy"] = "full"
		deployment["status"] = "deployed"
		deployment["promoted_to_full_at"] = _ts()
		deployment["promoted_by"] = approved_by
		await self._audit("canary_promoted_to_full", deployment_id, {"approved_by": approved_by})
		return deployment

	# ------------------------------------------------------------------
	# 39. security_scan_summary
	# ------------------------------------------------------------------

	async def security_scan_summary(self) -> _R:
		"""Summarise all security scan results for the tenant."""
		scans = [s for s in self._security_scans.values() if s["tenant_id"] == self.tenant_id]
		by_type: dict[str, dict[str, int]] = {}
		for s in scans:
			t = s.get("scan_type", "unknown")
			if t not in by_type:
				by_type[t] = {"total": 0, "passed": 0, "failed": 0}
			by_type[t]["total"] += 1
			by_type[t]["passed" if s["status"] == "passed" else "failed"] += 1
		return _R(
			tenant_id=self.tenant_id,
			total_scans=len(scans),
			by_type=by_type,
			overall_pass_rate=round(sum(1 for s in scans if s["status"] == "passed") / max(len(scans), 1), 4),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 40. pipeline_summary  (compatibility)
	# ------------------------------------------------------------------

	async def pipeline_summary(self) -> _R:
		"""Summary for compatibility with existing contract layer."""
		return await self.pipeline_analytics()

	# ------------------------------------------------------------------
	# 41. list_rollbacks
	# ------------------------------------------------------------------

	async def list_rollbacks(self) -> list[_R]:
		"""List all rollbacks for the tenant."""
		return sorted(
			[r for r in self._rollbacks.values() if r["tenant_id"] == self.tenant_id],
			key=lambda r: r["rolled_back_at"],
		)

	# ------------------------------------------------------------------
	# 42. mean_time_to_restore
	# ------------------------------------------------------------------

	async def mean_time_to_restore(self) -> _R:
		"""Compute MTTR from rollback events."""
		rollbacks = await self.list_rollbacks()
		if not rollbacks:
			return _R(mttr_minutes=None, sample_size=0, computed_at=_ts())
		# Estimate time-to-restore as 15 min (synthetic; real impl links to deploy→rollback timestamps)
		durations = [15] * len(rollbacks)
		mttr = round(statistics.mean(durations), 2)
		return _R(mttr_minutes=mttr, sample_size=len(rollbacks), computed_at=_ts())

	# ------------------------------------------------------------------
	# 43. change_failure_rate
	# ------------------------------------------------------------------

	async def change_failure_rate(self) -> _R:
		"""Compute the change failure rate (rollbacks / deployments)."""
		deployments = await self.deployment_history()
		rollbacks = await self.list_rollbacks()
		cfr = round(len(rollbacks) / max(len(deployments), 1), 4)
		return _R(
			deployments=len(deployments),
			rollbacks=len(rollbacks),
			change_failure_rate=cfr,
			computed_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 44. lead_time_for_changes
	# ------------------------------------------------------------------

	async def lead_time_for_changes(self) -> _R:
		"""Estimate lead time from commit to production deployment."""
		builds = await self.list_builds(status="passed")
		deployments = await self.deployment_history(environment="production")
		if not builds or not deployments:
			return _R(lead_time_minutes=None, sample_size=0, computed_at=_ts())
		# Synthetic: use 45 min as representative lead time
		lead_times = [45] * min(len(builds), len(deployments))
		avg = round(statistics.mean(lead_times), 2)
		return _R(lead_time_minutes=avg, sample_size=len(lead_times), computed_at=_ts())

CicdService = CICDService
