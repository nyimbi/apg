"""Service layer for APG Deployment Management."""

from __future__ import annotations

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
		self._engine = DeploymentEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

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
			"governance_posture": "ready" if releases and all(item["change_ticket"] for item in releases) else "needs_release_evidence",
		}

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
