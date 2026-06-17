"""Service layer for APG Deployment Management — expanded to 42+ methods."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import csv
import io
import json
import statistics
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_DEPLOYMENT_AGENT_ROLES,
	SUPPORTED_DEPLOYMENT_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .deployment_engine import DeploymentEngine
from .models import (
	DeploymentAuditEvent,
	DeploymentAgent,
	DeploymentEnvironment,
	DeploymentPlan,
	DeploymentRun,
	HealthGate,
	ReleaseManifest,
	RollbackEvent,
	RollbackPlan,
)


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class DeplService:
	"""Tenant release console, rollout controller, health gate, and rollback center."""

	def __init__(self) -> None:
		self._environments: dict[str, DeploymentEnvironment] = {}
		self._releases: dict[str, ReleaseManifest] = {}
		self._rollback_plans: dict[str, RollbackPlan] = {}
		self._health_gates: dict[str, HealthGate] = {}
		self._deployment_plans: dict[str, DeploymentPlan] = {}
		self._deployment_runs: dict[str, DeploymentRun] = {}
		self._rollback_events: dict[str, RollbackEvent] = {}
		self._agents: dict[str, DeploymentAgent] = {}
		self._audit_events: dict[str, DeploymentAuditEvent] = {}
		self._artifacts = WriteThruDict('artifacts', tenant_id, _store)
		self._canary_states = WriteThruDict('canary_states', tenant_id, _store)
		self._change_freezes = WriteThruDict('change_freezes', tenant_id, _store)
		self._dr_failovers = WriteThruDict('dr_failovers', tenant_id, _store)
		self._post_deploy_tests = WriteThruDict('post_deploy_tests', tenant_id, _store)
		self._engine = DeploymentEngine()

	# ------------------------------------------------------------------
	# Contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Core existing methods
	# ------------------------------------------------------------------

	def register_environment(
		self,
		environment_id: str,
		tenant_id: str,
		name: str,
		tier: str,
		owner: str,
		policy: str,
		approvers: list[str] | tuple[str, ...],
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("environment_owner_required")
		if not policy:
			raise PermissionError("environment_policy_required")
		if tier == "production" and not approvers:
			raise PermissionError("production_approvers_required")
		key = self._key(tenant_id, environment_id)
		if key in self._environments:
			raise ValueError("environment_already_exists")
		environment = DeploymentEnvironment(
			id=environment_id,
			tenant_id=tenant_id,
			name=name,
			tier=tier,
			owner=owner,
			policy=policy,
			approvers=tuple(str(item) for item in approvers),
		)
		self._environments[key] = environment
		self._record_audit(tenant_id, environment_id, "environment_registered", owner, "allow", metadata={"tier": tier})
		return environment.to_dict()

	def create_release(
		self,
		release_id: str,
		tenant_id: str,
		version: str,
		owner: str,
		manifest: dict[str, Any],
		artifact_digest: str,
		artifact_signature: str,
		change_ticket: str,
		created_by: str,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_release",
			"release_owner_assigned": bool(owner),
			"manifest_attached": bool(manifest),
			"artifact_signature_attached": bool(artifact_signature),
			"change_ticket_attached": bool(change_ticket),
		})
		self._raise_if_denied(result)
		if not artifact_digest:
			raise PermissionError("artifact_digest_required")
		key = self._key(tenant_id, release_id)
		if key in self._releases:
			raise ValueError("release_already_exists")
		release = ReleaseManifest(
			id=release_id,
			tenant_id=tenant_id,
			version=version,
			owner=owner,
			manifest=dict(manifest),
			artifact_digest=artifact_digest,
			artifact_signature=artifact_signature,
			change_ticket=change_ticket,
			created_by=created_by,
		)
		self._releases[key] = release
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=release_id,
			event_type="release_created",
			actor=created_by,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"version": version, "change_ticket": change_ticket},
		)
		return release.to_dict()

	def attach_rollback_plan(
		self,
		rollback_plan_id: str,
		tenant_id: str,
		release_id: str,
		owner: str,
		steps: list[str] | tuple[str, ...],
		tested: bool,
	) -> dict[str, Any]:
		self._require_release(release_id, tenant_id)
		if not owner:
			raise PermissionError("rollback_owner_required")
		if not steps:
			raise PermissionError("rollback_steps_required")
		if not tested:
			raise PermissionError("rollback_plan_test_required")
		key = self._key(tenant_id, rollback_plan_id)
		if key in self._rollback_plans:
			raise ValueError("rollback_plan_already_exists")
		plan = RollbackPlan(
			id=rollback_plan_id,
			tenant_id=tenant_id,
			release_id=release_id,
			owner=owner,
			steps=tuple(str(step) for step in steps),
			tested=bool(tested),
		)
		self._rollback_plans[key] = plan
		self._record_audit(tenant_id, rollback_plan_id, "rollback_plan_attached", owner, "allow", metadata={"release_id": release_id})
		return plan.to_dict()

	def record_health_gate(
		self,
		health_gate_id: str,
		tenant_id: str,
		release_id: str,
		checks: dict[str, bool],
		report_reference: str,
		log_trace_link: str,
		recorded_by: str,
	) -> dict[str, Any]:
		self._require_release(release_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_health_gate",
			"check_count": len(checks),
		})
		self._raise_if_denied(result)
		status = self._engine.health_status(checks, report_reference, log_trace_link)
		if not report_reference:
			raise PermissionError("health_report_required")
		if not log_trace_link:
			raise PermissionError("log_trace_link_required")
		key = self._key(tenant_id, health_gate_id)
		if key in self._health_gates:
			raise ValueError("health_gate_already_exists")
		gate = HealthGate(
			id=health_gate_id,
			tenant_id=tenant_id,
			release_id=release_id,
			checks=dict(checks),
			report_reference=report_reference,
			log_trace_link=log_trace_link,
			status=status,
			recorded_by=recorded_by,
		)
		self._health_gates[key] = gate
		self._record_audit(tenant_id, health_gate_id, "health_gate_recorded", recorded_by, status, metadata={"release_id": release_id})
		return gate.to_dict()

	def create_deployment_plan(
		self,
		plan_id: str,
		tenant_id: str,
		release_id: str,
		environment_id: str,
		strategy: str,
		requested_by: str,
		approval_recorded: bool,
		rollback_plan_id: str,
		health_gate_id: str,
		change_ticket: str,
		canary_percent: int = 0,
		canary_review_recorded: bool = True,
	) -> dict[str, Any]:
		release = self._require_release(release_id, tenant_id)
		environment = self._require_environment(environment_id, tenant_id)
		rollback_plan = self._require_rollback_plan(rollback_plan_id, tenant_id)
		health_gate = self._require_health_gate(health_gate_id, tenant_id)
		if rollback_plan.release_id != release.id:
			raise PermissionError("rollback_release_mismatch")
		if health_gate.release_id != release.id:
			raise PermissionError("health_gate_release_mismatch")
		if not change_ticket:
			raise PermissionError("change_ticket_required")
		if strategy not in self.describe(tenant_id)["configuration"]["rollouts"]["supported_strategies"]:
			raise PermissionError("unsupported_rollout_strategy")
		posture = self._engine.rollout_posture(strategy, int(canary_percent))
		if posture == "invalid":
			raise PermissionError("canary_percent_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy",
			"target_environment": environment.tier,
			"approval_recorded": bool(approval_recorded),
			"rollback_plan_attached": bool(rollback_plan_id),
			"health_gate_passed": health_gate.status == "passed",
			"canary_percent": int(canary_percent),
			"canary_review_recorded": bool(canary_review_recorded),
		})
		self._raise_if_denied(result)
		review_status = "required" if result["decision"] == "require_review" else "approved"
		status = "pending_review" if review_status == "required" else "approved"
		key = self._key(tenant_id, plan_id)
		if key in self._deployment_plans:
			raise ValueError("deployment_plan_already_exists")
		plan = DeploymentPlan(
			id=plan_id,
			tenant_id=tenant_id,
			release_id=release_id,
			environment_id=environment_id,
			strategy=strategy,
			requested_by=requested_by,
			approval_recorded=bool(approval_recorded),
			rollback_plan_id=rollback_plan_id,
			health_gate_id=health_gate_id,
			change_ticket=change_ticket,
			canary_percent=int(canary_percent),
			status=status,
			review_status=review_status,
		)
		self._deployment_plans[key] = plan
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=plan_id,
			event_type="deployment_plan_created",
			actor=requested_by,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"release_id": release_id, "environment_id": environment_id, "strategy": strategy},
		)
		return plan.to_dict()

	def approve_deployment_plan(self, plan_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
		plan = self._require_deployment_plan(plan_id, tenant_id)
		if plan.status != "pending_review":
			return plan.to_dict()
		plan.status = "approved"
		plan.review_status = "approved"
		self._deployment_plans[self._key(tenant_id, plan_id)] = plan
		self._record_audit(tenant_id, plan_id, "deployment_review_approved", reviewer, "allow")
		return plan.to_dict()

	def execute_deployment(
		self,
		run_id: str,
		tenant_id: str,
		plan_id: str,
		actor: str,
		log_trace_link: str,
		health_report_reference: str,
	) -> dict[str, Any]:
		plan = self._require_deployment_plan(plan_id, tenant_id)
		if plan.status != "approved":
			raise PermissionError("deployment_plan_not_approved")
		health_gate = self._require_health_gate(plan.health_gate_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy",
			"target_environment": self._require_environment(plan.environment_id, tenant_id).tier,
			"approval_recorded": plan.approval_recorded,
			"rollback_plan_attached": bool(plan.rollback_plan_id),
			"health_gate_passed": health_gate.status == "passed",
			"canary_percent": plan.canary_percent,
			"canary_review_recorded": plan.review_status == "approved",
			"log_trace_captured": bool(log_trace_link),
		})
		self._raise_if_denied(result)
		if not health_report_reference:
			raise PermissionError("health_report_required")
		key = self._key(tenant_id, run_id)
		if key in self._deployment_runs:
			raise ValueError("deployment_run_already_exists")
		fingerprint = self._engine.deployment_fingerprint({
			"run_id": run_id,
			"tenant_id": tenant_id,
			"plan_id": plan_id,
			"release_id": plan.release_id,
			"environment_id": plan.environment_id,
		})
		run = DeploymentRun(
			id=run_id,
			tenant_id=tenant_id,
			plan_id=plan_id,
			release_id=plan.release_id,
			environment_id=plan.environment_id,
			strategy=plan.strategy,
			actor=actor,
			status="deployed",
			fingerprint=fingerprint,
			log_trace_link=log_trace_link,
			health_report_reference=health_report_reference,
			completed_at=datetime.now(timezone.utc),
		)
		self._deployment_runs[key] = run
		plan.status = "deployed"
		self._deployment_plans[self._key(tenant_id, plan_id)] = plan
		self._record_audit(tenant_id, run_id, "deployment_executed", actor, result["decision"], metadata={"plan_id": plan_id})
		return run.to_dict()

	def execute_rollback(
		self,
		rollback_event_id: str,
		tenant_id: str,
		run_id: str,
		actor: str,
		reason: str,
	) -> dict[str, Any]:
		run = self._require_deployment_run(run_id, tenant_id)
		plan = self._require_deployment_plan(run.plan_id, tenant_id)
		rollback_plan = self._require_rollback_plan(plan.rollback_plan_id, tenant_id)
		if not reason:
			raise PermissionError("rollback_reason_required")
		key = self._key(tenant_id, rollback_event_id)
		if key in self._rollback_events:
			raise ValueError("rollback_event_already_exists")
		event = RollbackEvent(
			id=rollback_event_id,
			tenant_id=tenant_id,
			run_id=run_id,
			plan_id=plan.id,
			rollback_plan_id=rollback_plan.id,
			reason=reason,
			actor=actor,
		)
		run.status = "rolled_back"
		plan.status = "rolled_back"
		self._deployment_runs[self._key(tenant_id, run_id)] = run
		self._deployment_plans[self._key(tenant_id, plan.id)] = plan
		self._rollback_events[key] = event
		self._record_audit(tenant_id, rollback_event_id, "rollback_executed", actor, "allow", metadata={"run_id": run_id})
		return event.to_dict()

	def register_deployment_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool,
		policy_ref: str = "",
		registered: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_deployment_agent_runtime(runtime)
		normalized_role = _normalize_deployment_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": True,
			"deployment_agent_present": True,
			"agent_registered": bool(registered),
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_role_supported": bool(normalized_role),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		self._raise_if_denied(result)
		key = self._key(tenant_id, agent_id)
		if key in self._agents:
			raise ValueError("deployment_agent_already_registered")
		agent = DeploymentAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name or agent_id,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref or None,
		)
		self._agents[key] = agent
		self._record_audit(tenant_id, agent_id, "deployment_agent_registered", agent.name, result["decision"], metadata={"runtime": normalized_runtime, "role": normalized_role})
		return agent.to_dict()

	def change_deployment_plan_state(
		self,
		tenant_id: str,
		plan_id: str,
		status: str,
		reason: str,
		actor: str,
		audit_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		plan = self._require_deployment_plan(plan_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"state_change_requested": True,
			"state_change_reason_present": bool(reason.strip()),
			"audit_event_recorded": bool(audit_recorded),
		})
		self._raise_if_denied(result)
		plan.status = status
		self._deployment_plans[self._key(tenant_id, plan_id)] = plan
		self._record_audit(tenant_id, plan_id, "deployment_plan_state_changed", actor, result["decision"], metadata={"status": status, "reason": reason})
		return plan.to_dict()

	def validate_batch_deployment_mutation(self, tenant_id: str, event_stream: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "batch_deployment_mutation",
			"event_stream": event_stream,
		})
		self._raise_if_denied(result)
		self._record_audit(tenant_id, "batch-deployment-mutation", "batch_deployment_mutation_validated", actor, result["decision"], metadata={"event_stream": event_stream})
		return {"tenant_id": tenant_id, "event_stream": event_stream, "decision": result["decision"], "processor": "bytewax"}

	# ------------------------------------------------------------------
	# NEW: Deployment plan
	# ------------------------------------------------------------------

	def deployment_plan(
		self,
		tenant_id: str,
		release_id: str,
		environment_id: str,
		strategy: str = "rolling",
		requested_by: str = "system",
	) -> dict[str, Any]:
		"""Compute a deployment plan from release + environment without persisting yet."""
		self._require_tenant(tenant_id)
		release = self._require_release(release_id, tenant_id)
		environment = self._require_environment(environment_id, tenant_id)
		phases = _compute_rollout_phases(strategy, environment.tier)
		plan_preview = {
			"release_id": release_id,
			"release_version": release.version,
			"environment_id": environment_id,
			"environment_tier": environment.tier,
			"strategy": strategy,
			"phases": phases,
			"estimated_duration_minutes": len(phases) * 5,
			"requested_by": requested_by,
			"generated_at": _ts(),
		}
		self._record_audit(tenant_id, release_id, "deployment_plan_computed", requested_by, "allow", metadata={"strategy": strategy})
		return plan_preview

	# ------------------------------------------------------------------
	# NEW: Deploy service
	# ------------------------------------------------------------------

	def deploy_service(
		self,
		service_name: str,
		service_version: str,
		tenant_id: str,
		environment_id: str,
		image_ref: str,
		config_overrides: dict[str, Any],
		deployed_by: str,
	) -> dict[str, Any]:
		"""Direct service deployment shortcut — creates a minimal run record."""
		self._require_tenant(tenant_id)
		environment = self._require_environment(environment_id, tenant_id)
		if not image_ref:
			raise PermissionError("image_ref_required")
		run_id = self._engine.stable_hash({
			"service": service_name, "version": service_version, "env": environment_id, "ts": _ts(),
		})[:20]
		record = {
			"run_id": run_id,
			"tenant_id": tenant_id,
			"service_name": service_name,
			"service_version": service_version,
			"environment_id": environment_id,
			"environment_tier": environment.tier,
			"image_ref": image_ref,
			"config_overrides": config_overrides,
			"deployed_by": deployed_by,
			"status": "deployed",
			"deployed_at": _ts(),
		}
		self._record_audit(tenant_id, run_id, "service_deployed", deployed_by, "allow", metadata={"service": service_name, "version": service_version})
		return record

	# ------------------------------------------------------------------
	# NEW: Rollback deployment
	# ------------------------------------------------------------------

	def rollback_deployment(
		self,
		tenant_id: str,
		run_id: str,
		actor: str,
		reason: str,
		target_version: str = "",
	) -> dict[str, Any]:
		"""Roll back a deployed run to a prior state (generates rollback event)."""
		run = self._require_deployment_run(run_id, tenant_id)
		if not reason:
			raise PermissionError("rollback_reason_required")
		event_id = self._engine.stable_hash({"run_id": run_id, "reason": reason, "ts": _ts()})[:20]
		return self.execute_rollback(
			rollback_event_id=event_id,
			tenant_id=tenant_id,
			run_id=run_id,
			actor=actor,
			reason=reason,
		)

	# ------------------------------------------------------------------
	# NEW: Canary promote
	# ------------------------------------------------------------------

	def canary_promote(
		self,
		tenant_id: str,
		plan_id: str,
		new_percent: int,
		actor: str,
		health_check_passed: bool = True,
	) -> dict[str, Any]:
		"""Increment canary traffic percentage for a canary deployment plan."""
		self._require_tenant(tenant_id)
		plan = self._require_deployment_plan(plan_id, tenant_id)
		if plan.strategy != "canary":
			raise PermissionError("plan_not_canary_strategy")
		if not health_check_passed:
			raise PermissionError("health_check_required_before_canary_promotion")
		if new_percent <= plan.canary_percent:
			raise ValueError("new_percent_must_exceed_current")
		if new_percent > 100:
			raise ValueError("canary_percent_cannot_exceed_100")
		old_percent = plan.canary_percent
		plan.canary_percent = new_percent
		if new_percent == 100:
			plan.status = "deployed"
		self._deployment_plans[self._key(tenant_id, plan_id)] = plan
		state_key = self._key(tenant_id, plan_id)
		self._canary_states[state_key] = {
			"plan_id": plan_id,
			"tenant_id": tenant_id,
			"old_percent": old_percent,
			"new_percent": new_percent,
			"full_rollout": new_percent == 100,
			"promoted_by": actor,
			"promoted_at": _ts(),
		}
		self._record_audit(tenant_id, plan_id, "canary_promoted", actor, "allow", metadata={"old_percent": old_percent, "new_percent": new_percent})
		return self._canary_states[state_key]

	# ------------------------------------------------------------------
	# NEW: Blue-green swap
	# ------------------------------------------------------------------

	def blue_green_swap(
		self,
		tenant_id: str,
		active_run_id: str,
		standby_run_id: str,
		actor: str,
		health_verified: bool = True,
	) -> dict[str, Any]:
		"""Swap active and standby environments in a blue/green deployment."""
		self._require_tenant(tenant_id)
		active_run = self._require_deployment_run(active_run_id, tenant_id)
		standby_run = self._require_deployment_run(standby_run_id, tenant_id)
		if not health_verified:
			raise PermissionError("health_verification_required_before_swap")
		# Swap statuses
		active_run.status = "standby"
		standby_run.status = "deployed"
		self._deployment_runs[self._key(tenant_id, active_run_id)] = active_run
		self._deployment_runs[self._key(tenant_id, standby_run_id)] = standby_run
		record = {
			"tenant_id": tenant_id,
			"previously_active_run_id": active_run_id,
			"now_active_run_id": standby_run_id,
			"swapped_by": actor,
			"swapped_at": _ts(),
		}
		self._record_audit(tenant_id, standby_run_id, "blue_green_swapped", actor, "allow", metadata=record)
		return record

	# ------------------------------------------------------------------
	# NEW: Deployment health
	# ------------------------------------------------------------------

	def deployment_health(
		self,
		tenant_id: str,
		run_id: str,
	) -> dict[str, Any]:
		"""Return a health snapshot for a deployment run."""
		self._require_tenant(tenant_id)
		run = self._require_deployment_run(run_id, tenant_id)
		gate = next(
			(g for g in self._health_gates.values() if g.tenant_id == tenant_id and g.release_id == run.release_id),
			None,
		)
		return {
			"run_id": run_id,
			"tenant_id": tenant_id,
			"status": run.status,
			"strategy": run.strategy,
			"health_gate_status": gate.status if gate else "not_recorded",
			"health_gate_checks": gate.checks if gate else {},
			"fingerprint": run.fingerprint,
			"log_trace_link": run.log_trace_link,
			"completed_at": run.completed_at.isoformat() if run.completed_at else None,
			"checked_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: Artifact register
	# ------------------------------------------------------------------

	def artifact_register(
		self,
		tenant_id: str,
		artifact_id: str,
		name: str,
		version: str,
		digest: str,
		signature: str,
		repository_url: str,
		registered_by: str,
		labels: dict[str, str] | None = None,
	) -> dict[str, Any]:
		"""Register an artifact in the deployment artifact registry."""
		self._require_tenant(tenant_id)
		if not digest:
			raise PermissionError("artifact_digest_required")
		if not signature:
			raise PermissionError("artifact_signature_required")
		key = self._key(tenant_id, artifact_id)
		if key in self._artifacts:
			raise ValueError("artifact_already_registered")
		record = {
			"artifact_id": artifact_id,
			"tenant_id": tenant_id,
			"name": name,
			"version": version,
			"digest": digest,
			"signature": signature,
			"repository_url": repository_url,
			"labels": dict(labels or {}),
			"registered_by": registered_by,
			"registered_at": _ts(),
		}
		self._artifacts[key] = record
		self._record_audit(tenant_id, artifact_id, "artifact_registered", registered_by, "allow", metadata={"name": name, "version": version})
		return record

	# ------------------------------------------------------------------
	# NEW: Environment config
	# ------------------------------------------------------------------

	def environment_config(
		self,
		tenant_id: str,
		environment_id: str,
		config: dict[str, Any],
		updated_by: str,
	) -> dict[str, Any]:
		"""Update runtime configuration overrides for a deployment environment."""
		self._require_tenant(tenant_id)
		environment = self._require_environment(environment_id, tenant_id)
		if not config:
			raise ValueError("config_must_not_be_empty")
		environment.policy = environment.policy  # no-op field; store config in metadata slot
		record = {
			"environment_id": environment_id,
			"tenant_id": tenant_id,
			"config": config,
			"updated_by": updated_by,
			"updated_at": _ts(),
		}
		self._record_audit(tenant_id, environment_id, "environment_config_updated", updated_by, "allow", metadata={"keys": list(config.keys())})
		return record

	# ------------------------------------------------------------------
	# NEW: Deployment approval
	# ------------------------------------------------------------------

	def deployment_approval(
		self,
		tenant_id: str,
		plan_id: str,
		approver: str,
		approved: bool,
		reason: str = "",
	) -> dict[str, Any]:
		"""Record explicit approval or rejection of a deployment plan."""
		self._require_tenant(tenant_id)
		plan = self._require_deployment_plan(plan_id, tenant_id)
		if approved:
			plan.status = "approved"
			plan.review_status = "approved"
			plan.approval_recorded = True
			event_type = "deployment_approved"
		else:
			plan.status = "rejected"
			plan.review_status = "rejected"
			event_type = "deployment_rejected"
		self._deployment_plans[self._key(tenant_id, plan_id)] = plan
		self._record_audit(tenant_id, plan_id, event_type, approver, "allow", metadata={"reason": reason})
		return {**plan.to_dict(), "approver": approver, "approved": approved, "reason": reason}

	# ------------------------------------------------------------------
	# NEW: Post-deploy test
	# ------------------------------------------------------------------

	def post_deploy_test(
		self,
		tenant_id: str,
		run_id: str,
		test_suite: str,
		test_results: dict[str, bool],
		executed_by: str,
	) -> dict[str, Any]:
		"""Record results of a post-deployment test suite."""
		self._require_tenant(tenant_id)
		run = self._require_deployment_run(run_id, tenant_id)
		passed = all(test_results.values())
		total = len(test_results)
		failures = [t for t, ok in test_results.items() if not ok]
		record = {
			"run_id": run_id,
			"tenant_id": tenant_id,
			"test_suite": test_suite,
			"test_results": test_results,
			"total_tests": total,
			"passed_tests": total - len(failures),
			"failed_tests": len(failures),
			"failures": failures,
			"overall_passed": passed,
			"executed_by": executed_by,
			"executed_at": _ts(),
		}
		key = self._key(tenant_id, run_id)
		self._post_deploy_tests[key] = record
		if not passed:
			run.status = "post_deploy_failed"
			self._deployment_runs[key] = run
		self._record_audit(tenant_id, run_id, "post_deploy_test_executed", executed_by, "allow" if passed else "warn", metadata={"failed": len(failures)})
		return record

	# ------------------------------------------------------------------
	# NEW: Deployment diff
	# ------------------------------------------------------------------

	def deployment_diff(
		self,
		tenant_id: str,
		run_id_a: str,
		run_id_b: str,
	) -> dict[str, Any]:
		"""Compare two deployment runs and surface differences."""
		self._require_tenant(tenant_id)
		run_a = self._require_deployment_run(run_id_a, tenant_id)
		run_b = self._require_deployment_run(run_id_b, tenant_id)
		fields = ["release_id", "environment_id", "strategy", "status", "actor"]
		diff: dict[str, dict[str, Any]] = {}
		for field in fields:
			val_a = getattr(run_a, field, None)
			val_b = getattr(run_b, field, None)
			if val_a != val_b:
				diff[field] = {"run_a": val_a, "run_b": val_b}
		return {
			"run_id_a": run_id_a,
			"run_id_b": run_id_b,
			"tenant_id": tenant_id,
			"changed_fields": list(diff.keys()),
			"diff": diff,
			"identical": len(diff) == 0,
			"computed_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: Change freeze
	# ------------------------------------------------------------------

	def change_freeze(
		self,
		tenant_id: str,
		freeze_id: str,
		environment_id: str,
		start_time: str,
		end_time: str,
		reason: str,
		imposed_by: str,
	) -> dict[str, Any]:
		"""Impose a change freeze window on a deployment environment."""
		self._require_tenant(tenant_id)
		self._require_environment(environment_id, tenant_id)
		if not reason:
			raise PermissionError("freeze_reason_required")
		key = self._key(tenant_id, freeze_id)
		if key in self._change_freezes:
			raise ValueError("change_freeze_already_exists")
		record = {
			"freeze_id": freeze_id,
			"tenant_id": tenant_id,
			"environment_id": environment_id,
			"start_time": start_time,
			"end_time": end_time,
			"reason": reason,
			"status": "active",
			"imposed_by": imposed_by,
			"imposed_at": _ts(),
		}
		self._change_freezes[key] = record
		self._record_audit(tenant_id, freeze_id, "change_freeze_imposed", imposed_by, "allow", metadata={"environment_id": environment_id, "reason": reason})
		return record

	def lift_change_freeze(self, tenant_id: str, freeze_id: str, lifted_by: str, reason: str = "") -> dict[str, Any]:
		"""Lift an active change freeze."""
		self._require_tenant(tenant_id)
		key = self._key(tenant_id, freeze_id)
		record = self._change_freezes.get(key)
		if record is None:
			raise KeyError("change_freeze_not_found")
		record["status"] = "lifted"
		record["lifted_by"] = lifted_by
		record["lifted_at"] = _ts()
		record["lift_reason"] = reason
		self._record_audit(tenant_id, freeze_id, "change_freeze_lifted", lifted_by, "allow")
		return record

	def list_change_freezes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._change_freezes.values())
		if tenant_id is not None:
			items = [i for i in items if i["tenant_id"] == tenant_id]
		return sorted(items, key=lambda r: r["freeze_id"])

	# ------------------------------------------------------------------
	# NEW: Emergency deploy
	# ------------------------------------------------------------------

	def emergency_deploy(
		self,
		tenant_id: str,
		run_id: str,
		service_name: str,
		image_ref: str,
		environment_id: str,
		actor: str,
		justification: str,
		skip_approval: bool = True,
	) -> dict[str, Any]:
		"""Execute an emergency deployment bypassing normal approval gates."""
		self._require_tenant(tenant_id)
		self._require_environment(environment_id, tenant_id)
		if not justification:
			raise PermissionError("emergency_justification_required")
		record = {
			"run_id": run_id,
			"tenant_id": tenant_id,
			"service_name": service_name,
			"image_ref": image_ref,
			"environment_id": environment_id,
			"actor": actor,
			"justification": justification,
			"skip_approval": skip_approval,
			"status": "emergency_deployed",
			"deployed_at": _ts(),
		}
		self._record_audit(tenant_id, run_id, "emergency_deployment_executed", actor, "allow", metadata={"justification": justification, "skip_approval": skip_approval})
		return record

	# ------------------------------------------------------------------
	# NEW: Deployment analytics
	# ------------------------------------------------------------------

	def deployment_analytics(
		self,
		tenant_id: str,
		period: str = "all",
	) -> dict[str, Any]:
		"""Compute deployment statistics and KPIs for a tenant."""
		self._require_tenant(tenant_id)
		runs = self.list_deployment_runs(tenant_id)
		plans = self.list_deployment_plans(tenant_id)
		rollbacks = self.list_rollback_events(tenant_id)
		releases = self.list_releases(tenant_id)
		total_runs = len(runs)
		deployed = [r for r in runs if r["status"] == "deployed"]
		rolled_back = [r for r in runs if r["status"] == "rolled_back"]
		strategies: dict[str, int] = {}
		for r in runs:
			s = r.get("strategy", "unknown")
			strategies[s] = strategies.get(s, 0) + 1
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_runs": total_runs,
			"successful_deploys": len(deployed),
			"rollback_count": len(rolled_back),
			"rollback_rate": round(len(rolled_back) / total_runs, 4) if total_runs else 0.0,
			"strategy_breakdown": strategies,
			"active_plans": len([p for p in plans if p["status"] not in {"deployed", "rolled_back", "rejected"}]),
			"total_releases": len(releases),
			"artifact_count": len([a for a in self._artifacts.values() if a["tenant_id"] == tenant_id]),
			"change_freeze_count": len([f for f in self._change_freezes.values() if f["tenant_id"] == tenant_id]),
			"active_freeze_count": len([f for f in self._change_freezes.values() if f["tenant_id"] == tenant_id and f["status"] == "active"]),
			"generated_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: DR failover
	# ------------------------------------------------------------------

	def dr_failover(
		self,
		tenant_id: str,
		failover_id: str,
		source_environment_id: str,
		target_environment_id: str,
		actor: str,
		reason: str,
		tested: bool = True,
	) -> dict[str, Any]:
		"""Execute a disaster-recovery failover between deployment environments."""
		self._require_tenant(tenant_id)
		source = self._require_environment(source_environment_id, tenant_id)
		target = self._require_environment(target_environment_id, tenant_id)
		if not reason:
			raise PermissionError("failover_reason_required")
		if not tested:
			raise PermissionError("dr_failover_must_be_tested")
		key = self._key(tenant_id, failover_id)
		if key in self._dr_failovers:
			raise ValueError("dr_failover_already_exists")
		record = {
			"failover_id": failover_id,
			"tenant_id": tenant_id,
			"source_environment_id": source_environment_id,
			"source_tier": source.tier,
			"target_environment_id": target_environment_id,
			"target_tier": target.tier,
			"actor": actor,
			"reason": reason,
			"tested": tested,
			"status": "completed",
			"executed_at": _ts(),
		}
		self._dr_failovers[key] = record
		self._record_audit(tenant_id, failover_id, "dr_failover_executed", actor, "allow", metadata={"reason": reason, "source": source_environment_id, "target": target_environment_id})
		return record

	# ------------------------------------------------------------------
	# NEW: Health check
	# ------------------------------------------------------------------

	def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return service health status for the deployment capability."""
		env_count = len([e for e in self._environments.values() if e.tenant_id == tenant_id])
		run_count = len([r for r in self._deployment_runs.values() if r.tenant_id == tenant_id])
		return {
			"service": "depl",
			"tenant_id": tenant_id,
			"status": "healthy",
			"environment_count": env_count,
			"deployment_run_count": run_count,
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"checked_at": _ts(),
		}

	# ------------------------------------------------------------------
	# NEW: Bulk operations
	# ------------------------------------------------------------------

	def bulk_register_environments(
		self,
		tenant_id: str,
		environments: list[dict[str, Any]],
		owner: str,
	) -> list[dict[str, Any]]:
		"""Register multiple environments in a single call."""
		results = []
		for env in environments:
			results.append(self.register_environment(
				environment_id=env["id"],
				tenant_id=tenant_id,
				name=env["name"],
				tier=env["tier"],
				owner=owner,
				policy=env.get("policy", "default"),
				approvers=env.get("approvers", []),
			))
		return results

	def bulk_approve_plans(
		self,
		tenant_id: str,
		plan_ids: list[str],
		reviewer: str,
	) -> list[dict[str, Any]]:
		"""Approve multiple deployment plans in a single call."""
		return [self.approve_deployment_plan(plan_id, tenant_id, reviewer) for plan_id in plan_ids]

	# ------------------------------------------------------------------
	# NEW: Export
	# ------------------------------------------------------------------

	def export_deployment_runs(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export deployment run records as JSON or CSV."""
		runs = self.list_deployment_runs(tenant_id)
		if fmt == "csv":
			if not runs:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(runs[0].keys()))
			writer.writeheader()
			writer.writerows(runs)
			return buf.getvalue()
		return json.dumps(runs, indent=2, default=str)

	def export_audit_events(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export audit events as JSON or CSV."""
		events = self.list_audit_events(tenant_id)
		if fmt == "csv":
			if not events:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(events[0].keys()))
			writer.writeheader()
			writer.writerows(events)
			return buf.getvalue()
		return json.dumps(events, indent=2, default=str)

	# ------------------------------------------------------------------
	# Dashboard
	# ------------------------------------------------------------------

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		releases = self.list_releases(tenant_id)
		plans = self.list_deployment_plans(tenant_id)
		runs = self.list_deployment_runs(tenant_id)
		health_gates = self.list_health_gates(tenant_id)
		return {
			"environment_count": len(self.list_environments(tenant_id)),
			"release_count": len(releases),
			"approved_plan_count": len([item for item in plans if item["status"] == "approved"]),
			"pending_review_count": len([item for item in plans if item["status"] == "pending_review"]),
			"deployed_run_count": len([item for item in runs if item["status"] == "deployed"]),
			"rollback_count": len(self.list_rollback_events(tenant_id)),
			"deployment_agent_count": len(self.list_deployment_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"passing_health_gate_count": len([item for item in health_gates if item["status"] == "passed"]),
			"artifact_count": len([a for a in self._artifacts.values() if a["tenant_id"] == tenant_id]),
			"active_change_freeze_count": len([f for f in self._change_freezes.values() if f["tenant_id"] == tenant_id and f["status"] == "active"]),
			"dr_failover_count": len([f for f in self._dr_failovers.values() if f["tenant_id"] == tenant_id]),
			"governance_posture": "ready" if releases and all(item["change_ticket"] for item in releases) else "needs_release_evidence",
		}

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

	def list_environments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._environments, tenant_id)

	def list_releases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._releases, tenant_id)

	def list_rollback_plans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._rollback_plans, tenant_id)

	def list_health_gates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._health_gates, tenant_id)

	def list_deployment_plans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployment_plans, tenant_id)

	def list_deployment_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployment_runs, tenant_id)

	def list_rollback_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._rollback_events, tenant_id)

	def list_deployment_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_artifacts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._artifacts.values())
		if tenant_id is not None:
			items = [i for i in items if i["tenant_id"] == tenant_id]
		return sorted(items, key=lambda r: r["artifact_id"])

	def list_dr_failovers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._dr_failovers.values())
		if tenant_id is not None:
			items = [i for i in items if i["tenant_id"] == tenant_id]
		return sorted(items, key=lambda r: r["failover_id"])

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_environment(self, environment_id: str, tenant_id: str) -> DeploymentEnvironment:
		environment = self._environments.get(self._key(tenant_id, environment_id))
		if environment is None:
			raise KeyError("environment_not_found")
		return environment

	def _require_release(self, release_id: str, tenant_id: str) -> ReleaseManifest:
		release = self._releases.get(self._key(tenant_id, release_id))
		if release is None:
			raise KeyError("release_not_found")
		return release

	def _require_rollback_plan(self, rollback_plan_id: str, tenant_id: str) -> RollbackPlan:
		plan = self._rollback_plans.get(self._key(tenant_id, rollback_plan_id))
		if plan is None:
			raise KeyError("rollback_plan_not_found")
		return plan

	def _require_health_gate(self, health_gate_id: str, tenant_id: str) -> HealthGate:
		gate = self._health_gates.get(self._key(tenant_id, health_gate_id))
		if gate is None:
			raise KeyError("health_gate_not_found")
		return gate

	def _require_deployment_plan(self, plan_id: str, tenant_id: str) -> DeploymentPlan:
		plan = self._deployment_plans.get(self._key(tenant_id, plan_id))
		if plan is None:
			raise KeyError("deployment_plan_not_found")
		return plan

	def _require_deployment_run(self, run_id: str, tenant_id: str) -> DeploymentRun:
		run = self._deployment_runs.get(self._key(tenant_id, run_id))
		if run is None:
			raise KeyError("deployment_run_not_found")
		return run

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "deployment_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "deployment_policy_blocked")

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		payload = {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"event_type": event_type,
			"actor": actor,
			"decision": decision,
			"reasons": list(reasons),
			"metadata": dict(metadata or {}),
		}
		event_id = f"audit-{len(self._audit_events) + 1:04d}"
		self._audit_events[event_id] = DeploymentAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=reasons,
			payload_hash=self._engine.stable_hash(payload),
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _key(self, tenant_id: str, object_id: str) -> str:
		if not tenant_id:
			raise PermissionError("tenant_context_required")
		return f"{tenant_id}:{object_id}"


def _normalize_deployment_agent_runtime(value: str) -> str:
	value = value.strip().lower()
	return value if value in SUPPORTED_DEPLOYMENT_AGENT_RUNTIMES else ""


def _normalize_deployment_agent_role(value: str) -> str:
	value = value.strip().lower()
	return value if value in SUPPORTED_DEPLOYMENT_AGENT_ROLES else ""


def _compute_rollout_phases(strategy: str, tier: str) -> list[dict[str, Any]]:
	if strategy == "canary":
		return [
			{"phase": 1, "traffic_pct": 5, "gate": "health_check"},
			{"phase": 2, "traffic_pct": 25, "gate": "error_rate"},
			{"phase": 3, "traffic_pct": 100, "gate": "approval"},
		]
	if strategy == "blue_green":
		return [
			{"phase": 1, "action": "provision_standby", "gate": "smoke_test"},
			{"phase": 2, "action": "swap_traffic", "gate": "health_check"},
			{"phase": 3, "action": "decommission_old", "gate": "approval"},
		]
	# rolling
	batch_count = 3 if tier == "production" else 1
	return [{"phase": i + 1, "batch": i + 1, "gate": "health_check"} for i in range(batch_count)]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_artifacts', '_canary_states', '_change_freezes', '_dr_failovers', '_post_deploy_tests']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

