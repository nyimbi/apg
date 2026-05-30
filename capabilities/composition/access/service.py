"""Service layer for the composition access-control capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_ACCESS_AGENT_ROLES,
	SUPPORTED_ACCESS_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .models import (
	AccessAgentRecord,
	AccessAuditEventRecord,
	AccessDecisionRecord,
	AccessGrantRecord,
	AccessPolicyRecord,
	AccessProviderRecord,
	AccessResourceRecord,
	AccessSessionRecord,
	stable_id,
	utc_now,
)


class CompositionAccessService:
	"""Dependency-light access-control runtime behind the capability contract."""

	def __init__(self) -> None:
		self._providers: dict[str, AccessProviderRecord] = {}
		self._resources: dict[str, AccessResourceRecord] = {}
		self._policies: dict[str, AccessPolicyRecord] = {}
		self._grants: dict[str, AccessGrantRecord] = {}
		self._sessions: dict[str, AccessSessionRecord] = {}
		self._decisions: dict[str, AccessDecisionRecord] = {}
		self._agents: dict[str, AccessAgentRecord] = {}
		self._audit_events: list[AccessAuditEventRecord] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_provider(
		self,
		provider_key: str,
		tenant_id: str,
		name: str,
		provider_type: str,
		owner_id: str,
		external: bool = True,
		metadata_validated: bool = False,
		secret_reference: str | None = None,
		test_evidence: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_provider",
			"provider_owner_assigned": bool(owner_id),
		})
		provider_id = stable_id("provider", tenant_id, provider_key)
		status = "active" if metadata_validated and (not external or secret_reference) and test_evidence else "draft"
		record = AccessProviderRecord(
			id=provider_id,
			tenant_id=tenant_id,
			name=name,
			provider_type=provider_type,
			owner_id=owner_id,
			status=status,
			external=external,
			metadata_validated=metadata_validated,
			secret_reference=secret_reference,
			test_evidence=test_evidence,
			metadata=dict(metadata or {}),
		)
		self._providers[provider_id] = record
		self._audit(tenant_id, "provider_registered", provider_id, owner_id, {"status": status, "provider_type": provider_type})
		return record.to_dict()

	def activate_provider(
		self,
		provider_id: str,
		actor_id: str,
		metadata_validated: bool = True,
		secret_reference: str | None = None,
		test_evidence: str | None = None,
	) -> dict[str, Any]:
		provider = self._get_provider(provider_id)
		next_secret = secret_reference or provider.secret_reference
		next_evidence = test_evidence or provider.test_evidence
		self._enforce_context({
			"tenant_context_present": bool(provider.tenant_id),
			"operation": "activate_provider",
			"provider_metadata_validated": bool(metadata_validated),
			"external_provider": bool(provider.external),
			"secret_reference_present": bool(next_secret),
		})
		if not next_evidence:
			raise PermissionError("provider_test_evidence_required")
		provider.metadata_validated = True
		provider.secret_reference = next_secret
		provider.test_evidence = next_evidence
		provider.status = "active"
		provider.updated_at = utc_now()
		self._audit(provider.tenant_id, "provider_activated", provider_id, actor_id, {"test_evidence": next_evidence})
		return provider.to_dict()

	def register_resource(
		self,
		resource_key: str,
		tenant_id: str,
		display_name: str,
		owner_id: str,
		scopes: list[str],
		capability_id: str,
		sensitive: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_resource",
			"resource_owner_assigned": bool(owner_id),
			"scope_present": bool(scopes),
		})
		resource_id = stable_id("resource", tenant_id, capability_id, resource_key)
		record = AccessResourceRecord(
			id=resource_id,
			tenant_id=tenant_id,
			resource_key=resource_key,
			display_name=display_name,
			owner_id=owner_id,
			scopes=list(scopes),
			capability_id=capability_id,
			sensitive=sensitive,
			metadata=dict(metadata or {}),
		)
		self._resources[resource_id] = record
		self._audit(tenant_id, "resource_registered", resource_id, owner_id, {"capability_id": capability_id, "sensitive": sensitive})
		return record.to_dict()

	def create_policy(
		self,
		policy_key: str,
		tenant_id: str,
		name: str,
		resource_id: str,
		owner_id: str,
		effect: str,
		conditions: dict[str, Any] | None = None,
		risk_level: str = "standard",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		resource = self._get_resource(resource_id)
		if resource.tenant_id != tenant_id:
			raise ValueError("resource_tenant_mismatch")
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_policy",
			"policy_owner_assigned": bool(owner_id),
			"sensitive_resource": bool(resource.sensitive),
			"policy_conditions_present": bool(conditions),
		})
		if effect not in {"allow", "deny"}:
			raise ValueError("policy_effect_invalid")
		policy_id = stable_id("policy", tenant_id, policy_key)
		record = AccessPolicyRecord(
			id=policy_id,
			tenant_id=tenant_id,
			name=name,
			resource_id=resource_id,
			owner_id=owner_id,
			effect=effect,
			conditions=dict(conditions or {}),
			risk_level=risk_level,
			metadata=dict(metadata or {}),
		)
		self._policies[policy_id] = record
		self._audit(tenant_id, "policy_created", policy_id, owner_id, {"resource_id": resource_id, "effect": effect, "risk_level": risk_level})
		return record.to_dict()

	def activate_policy(
		self,
		policy_id: str,
		actor_id: str,
		simulation_evidence: str | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		policy = self._get_policy(policy_id)
		next_simulation = simulation_evidence or policy.simulation_evidence
		result = self.evaluate({
			"tenant_context_present": bool(policy.tenant_id),
			"operation": "activate_policy",
			"risk_level": policy.risk_level,
			"simulation_evidence_present": bool(next_simulation),
		})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review" and not reviewed_by:
			raise PermissionError(",".join(result["matched_rules"]))
		policy.simulation_evidence = next_simulation
		policy.reviewed_by = reviewed_by or actor_id
		policy.status = "active"
		policy.updated_at = utc_now()
		self._audit(policy.tenant_id, "policy_activated", policy_id, actor_id, {"reviewed_by": policy.reviewed_by})
		return policy.to_dict()

	def create_grant(
		self,
		grant_key: str,
		tenant_id: str,
		subject_id: str,
		resource_id: str,
		scopes: list[str],
		requested_by: str,
		justification: str,
		privileged: bool = False,
		approved_by: str | None = None,
		expires_at: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		resource = self._get_resource(resource_id)
		if resource.tenant_id != tenant_id:
			raise ValueError("resource_tenant_mismatch")
		unknown_scopes = sorted(set(scopes) - set(resource.scopes))
		if unknown_scopes:
			raise ValueError("grant_scope_not_registered")
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_grant",
			"privileged_scope": bool(privileged),
			"approval_recorded": bool(approved_by),
			"expiry_present": bool(expires_at),
			"separation_of_duties_passed": not privileged or (bool(approved_by) and approved_by != requested_by),
			"justification_present": bool(justification),
		})
		grant_id = stable_id("grant", tenant_id, grant_key)
		record = AccessGrantRecord(
			id=grant_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			resource_id=resource_id,
			scopes=list(scopes),
			requested_by=requested_by,
			justification=justification,
			privileged=privileged,
			approved_by=approved_by,
			expires_at=expires_at,
			metadata=dict(metadata or {}),
		)
		self._grants[grant_id] = record
		self._audit(tenant_id, "grant_created", grant_id, requested_by, {"resource_id": resource_id, "privileged": privileged})
		return record.to_dict()

	def revoke_grant(self, grant_id: str, actor_id: str, reason: str) -> dict[str, Any]:
		grant = self._get_grant(grant_id)
		grant.status = "revoked"
		grant.updated_at = utc_now()
		self._audit(grant.tenant_id, "grant_revoked", grant_id, actor_id, {"reason": reason})
		return grant.to_dict()

	def evaluate_session(
		self,
		session_key: str,
		tenant_id: str,
		subject_id: str,
		provider_id: str,
		risk_score: int,
		step_up_completed: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		provider = self._get_provider(provider_id)
		if provider.tenant_id != tenant_id:
			raise ValueError("provider_tenant_mismatch")
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "evaluate_session",
			"risk_score": risk_score,
			"step_up_completed": bool(step_up_completed),
		})
		session_id = stable_id("session", tenant_id, session_key)
		status = "verified" if risk_score <= 74 or step_up_completed else "blocked"
		record = AccessSessionRecord(
			id=session_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			provider_id=provider_id,
			risk_score=risk_score,
			status=status,
			step_up_completed=step_up_completed,
			metadata=dict(metadata or {}),
		)
		self._sessions[session_id] = record
		self._audit(tenant_id, "session_evaluated", session_id, subject_id, {"risk_score": risk_score, "status": status})
		return record.to_dict()

	def record_decision(
		self,
		decision_key: str,
		tenant_id: str,
		subject_id: str,
		resource_id: str,
		action: str,
		decision: str,
		reason: str,
		policy_ids: list[str] | None = None,
		event_stream: str = "bytewax",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		resource = self._get_resource(resource_id)
		if resource.tenant_id != tenant_id:
			raise ValueError("resource_tenant_mismatch")
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_decision",
			"event_stream": event_stream,
		})
		if decision not in {"allow", "deny", "review"}:
			raise ValueError("access_decision_invalid")
		decision_id = stable_id("decision", tenant_id, decision_key)
		record = AccessDecisionRecord(
			id=decision_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			resource_id=resource_id,
			action=action,
			decision=decision,
			reason=reason,
			policy_ids=list(policy_ids or []),
			event_stream=event_stream,
			metadata=dict(metadata or {}),
		)
		self._decisions[decision_id] = record
		self._audit(tenant_id, "access_decision_recorded", decision_id, subject_id, {"decision": decision, "event_stream": event_stream})
		return record.to_dict()

	def register_access_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		instructions: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_access_agent",
			"agent_runtime_supported": runtime in SUPPORTED_ACCESS_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_ACCESS_AGENT_ROLES,
		})
		agent_id = stable_id("access_agent", tenant_id, name, runtime, role)
		record = AccessAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime,
			role=role,
			instructions=instructions,
			metadata=dict(metadata or {}),
		)
		self._agents[agent_id] = record
		self._audit(tenant_id, "access_agent_registered", agent_id, name, {"runtime": runtime, "role": role})
		return record.to_dict()

	def validate_agent_access_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		agent = self._get_agent(agent_id)
		if agent.tenant_id != tenant_id:
			raise ValueError("agent_tenant_mismatch")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_access_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		return {"tenant_id": tenant_id, "agent_id": agent_id, "action": action, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def validate_batch_grant(self, tenant_id: str, grant_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "batch_grant",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant_id, "grant_count": grant_count, "event_stream": event_stream, "stream": event_stream_name(), "processor": "bytewax"}

	def list_providers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._providers, tenant_id)

	def list_resources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._resources, tenant_id)

	def list_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._policies, tenant_id)

	def list_grants(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._grants, tenant_id)

	def list_sessions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sessions, tenant_id)

	def list_decisions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._decisions, tenant_id)

	def list_access_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [event.to_dict() for event in self._audit_events if tenant_id is None or event.tenant_id == tenant_id]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"provider_count": len(self.list_providers(tenant_id)),
			"resource_count": len(self.list_resources(tenant_id)),
			"policy_count": len(self.list_policies(tenant_id)),
			"grant_count": len(self.list_grants(tenant_id)),
			"session_count": len(self.list_sessions(tenant_id)),
			"decision_count": len(self.list_decisions(tenant_id)),
			"access_agent_count": len(self.list_access_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce_context({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached})
		resource = self.register_resource(
			resource_key=record_id,
			tenant_id=tenant_id,
			display_name=str((metadata or {}).get("name") or record_id),
			owner_id=str((metadata or {}).get("owner_id") or "system"),
			scopes=["read"],
			capability_id="composition_access",
			metadata=metadata,
		)
		resource["status"] = status
		return resource

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_resources(tenant_id)

	def _enforce_context(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))

	def _require_tenant(self, tenant_id: str) -> None:
		self._enforce_context({"tenant_context_present": bool(tenant_id)})

	def _audit(self, tenant_id: str, event_type: str, entity_id: str, actor_id: str, metadata: dict[str, Any] | None = None) -> None:
		self._audit_events.append(
			AccessAuditEventRecord(
				id=stable_id("access_audit", tenant_id, event_type, entity_id, str(len(self._audit_events))),
				tenant_id=tenant_id,
				event_type=event_type,
				entity_id=entity_id,
				actor_id=actor_id,
				metadata=dict(metadata or {}),
			)
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in records.values() if tenant_id is None or record.tenant_id == tenant_id]

	def _get_provider(self, provider_id: str) -> AccessProviderRecord:
		try:
			return self._providers[provider_id]
		except KeyError as exc:
			raise KeyError(f"unknown_provider:{provider_id}") from exc

	def _get_resource(self, resource_id: str) -> AccessResourceRecord:
		try:
			return self._resources[resource_id]
		except KeyError as exc:
			raise KeyError(f"unknown_resource:{resource_id}") from exc

	def _get_policy(self, policy_id: str) -> AccessPolicyRecord:
		try:
			return self._policies[policy_id]
		except KeyError as exc:
			raise KeyError(f"unknown_policy:{policy_id}") from exc

	def _get_grant(self, grant_id: str) -> AccessGrantRecord:
		try:
			return self._grants[grant_id]
		except KeyError as exc:
			raise KeyError(f"unknown_grant:{grant_id}") from exc

	def _get_agent(self, agent_id: str) -> AccessAgentRecord:
		try:
			return self._agents[agent_id]
		except KeyError as exc:
			raise KeyError(f"unknown_access_agent:{agent_id}") from exc


__all__ = [
	"CompositionAccessService",
	"AccessProviderRecord",
	"AccessResourceRecord",
	"AccessPolicyRecord",
	"AccessGrantRecord",
	"AccessSessionRecord",
	"AccessDecisionRecord",
	"AccessAgentRecord",
	"AccessAuditEventRecord",
]
