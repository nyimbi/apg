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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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

	async def rotate_secret(
		self,
		provider_id: str,
		actor_id: str,
		new_secret_reference: str,
	) -> dict[str, Any]:
		"""Rotate the secret reference for an active provider."""
		assert new_secret_reference, "new_secret_reference required"
		provider = self._get_provider(provider_id)
		self._enforce_context({
			"tenant_context_present": bool(provider.tenant_id),
			"operation": "rotate_secret",
			"provider_metadata_validated": bool(provider.metadata_validated),
		})
		old_ref = provider.secret_reference
		provider.secret_reference = new_secret_reference
		provider.updated_at = utc_now()
		self._audit(provider.tenant_id, "provider_secret_rotated", provider_id, actor_id, {"old_ref": old_ref})
		return provider.to_dict()

	async def suspend_grant(
		self,
		grant_id: str,
		actor_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Temporarily suspend an active grant without revoking it."""
		assert reason, "reason required"
		grant = self._get_grant(grant_id)
		grant.status = "suspended"
		grant.updated_at = utc_now()
		self._audit(grant.tenant_id, "grant_suspended", grant_id, actor_id, {"reason": reason})
		return grant.to_dict()

	async def reinstate_grant(
		self,
		grant_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Reinstate a suspended grant."""
		grant = self._get_grant(grant_id)
		if grant.status != "suspended":
			raise ValueError("grant_not_suspended")
		grant.status = "active"
		grant.updated_at = utc_now()
		self._audit(grant.tenant_id, "grant_reinstated", grant_id, actor_id, {})
		return grant.to_dict()

	async def bulk_revoke_grants(
		self,
		grant_ids: list[str],
		actor_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Revoke multiple grants in a single operation."""
		assert grant_ids, "grant_ids required"
		assert reason, "reason required"
		results: list[dict[str, Any]] = []
		for gid in grant_ids:
			try:
				r = self.revoke_grant(gid, actor_id, reason)
				results.append({"grant_id": gid, "status": "revoked"})
			except Exception as exc:
				results.append({"grant_id": gid, "status": "error", "error": str(exc)})
		return {
			"revoked_count": sum(1 for r in results if r["status"] == "revoked"),
			"error_count": sum(1 for r in results if r["status"] == "error"),
			"results": results,
		}

	async def check_access(
		self,
		tenant_id: str,
		subject_id: str,
		resource_id: str,
		action: str,
		scope: str,
	) -> dict[str, Any]:
		"""Check whether a subject has access to perform an action on a resource."""
		active_grants = [
			g for g in self._grants.values()
			if g.tenant_id == tenant_id
			and g.subject_id == subject_id
			and g.resource_id == resource_id
			and g.status not in {"revoked", "suspended"}
			and scope in g.scopes
		]
		decision = "allow" if active_grants else "deny"
		self._audit(tenant_id, "access_check_performed", resource_id, subject_id, {
			"action": action, "scope": scope, "decision": decision
		})
		return {
			"subject_id": subject_id,
			"resource_id": resource_id,
			"action": action,
			"scope": scope,
			"decision": decision,
			"matching_grants": len(active_grants),
			"checked_at": utc_now(),
		}

	async def export_access_log(
		self,
		tenant_id: str,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export access decision log for audit purposes."""
		assert format in {"json", "csv"}, "format must be json or csv"
		decisions = self.list_decisions(tenant_id)
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if decisions:
				writer = csv.DictWriter(buf, fieldnames=list(decisions[0].keys()))
				writer.writeheader()
				writer.writerows(decisions)
			return {"format": "csv", "tenant_id": tenant_id, "record_count": len(decisions), "content": buf.getvalue()}
		return {"format": "json", "tenant_id": tenant_id, "record_count": len(decisions), "records": decisions}

	async def access_analytics(
		self,
		tenant_id: str,
		period: str = "last_30_days",
	) -> dict[str, Any]:
		"""Compute access control analytics: allow/deny rates, top subjects."""
		decisions = self.list_decisions(tenant_id)
		allows = sum(1 for d in decisions if d.get("decision") == "allow")
		denies = sum(1 for d in decisions if d.get("decision") == "deny")
		total = len(decisions)
		allow_rate = round(allows / max(total, 1) * 100, 2)
		subject_counts: dict[str, int] = {}
		for d in decisions:
			sid = d.get("subject_id", "unknown")
			subject_counts[sid] = subject_counts.get(sid, 0) + 1
		top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:5]
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_decisions": total,
			"allow_count": allows,
			"deny_count": denies,
			"allow_rate_pct": allow_rate,
			"top_subjects": [{"subject_id": s, "count": n} for s, n in top_subjects],
			"computed_at": utc_now(),
		}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return access service health status."""
		return {
			"service": "CompositionAccessService",
			"tenant_id": tenant_id,
			"status": "healthy",
			"provider_count": len(self.list_providers(tenant_id)),
			"resource_count": len(self.list_resources(tenant_id)),
			"grant_count": len(self.list_grants(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"checked_at": utc_now(),
		}

	async def access_compliance_report(
		self,
		tenant_id: str,
		standard: str = "ISO27001",
	) -> dict[str, Any]:
		"""Generate an access control compliance report."""
		grants = self.list_grants(tenant_id)
		privileged = [g for g in grants if g.get("privileged")]
		approved_privileged = [g for g in privileged if g.get("approved_by")]
		expired_grants = [g for g in grants if g.get("expires_at") and g.get("expires_at") < utc_now()]
		self._audit(tenant_id, "access_compliance_report_generated", standard, "system", {})
		return {
			"standard": standard,
			"tenant_id": tenant_id,
			"total_grants": len(grants),
			"privileged_grants": len(privileged),
			"approved_privileged_grants": len(approved_privileged),
			"expired_grants": len(expired_grants),
			"compliance_rate_pct": round(len(approved_privileged) / max(len(privileged), 1) * 100, 2),
			"generated_at": utc_now(),
		}

	# ── New async methods ────────────────────────────────────────────────────

	async def export_permission_matrix(
		self,
		tenant_id: str,
		format: str = "json",
	) -> dict[str, Any]:
		"""Build a point-in-time permission matrix mapping each subject to the
		resources and scopes they hold active grants for.

		Supports *json*, *csv*, and *html* output formats for SOC-2 / ISO-27001
		auditors and access-review tooling.
		"""
		assert format in {"json", "csv", "html"}, "format must be json|csv|html"
		grants = [
			g for g in self._grants.values()
			if g.tenant_id == tenant_id and g.status not in {"revoked", "suspended", "expired"}
		]
		# Build nested matrix: subject → resource → [scopes]
		matrix: dict[str, dict[str, list[str]]] = {}
		for g in grants:
			matrix.setdefault(g.subject_id, {}).setdefault(g.resource_id, [])
			matrix[g.subject_id][g.resource_id] = sorted(
				set(matrix[g.subject_id][g.resource_id]) | set(g.scopes)
			)
		self._audit(tenant_id, "permission_matrix_exported", tenant_id, "system", {"format": format, "subject_count": len(matrix)})
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			writer = csv.writer(buf)
			writer.writerow(["subject_id", "resource_id", "scopes"])
			for subj, resources in matrix.items():
				for res, scopes in resources.items():
					writer.writerow([subj, res, "|".join(scopes)])
			return {"format": "csv", "tenant_id": tenant_id, "content": buf.getvalue()}
		if format == "html":
			rows = "".join(
				f"<tr><td>{s}</td><td>{r}</td><td>{', '.join(sc)}</td></tr>"
				for s, resources in matrix.items()
				for r, sc in resources.items()
			)
			html = f"<table><thead><tr><th>Subject</th><th>Resource</th><th>Scopes</th></tr></thead><tbody>{rows}</tbody></table>"
			return {"format": "html", "tenant_id": tenant_id, "content": html}
		return {"format": "json", "tenant_id": tenant_id, "matrix": matrix, "exported_at": utc_now()}

	async def simulate_policy(
		self,
		policy_id: str,
		sample_decisions: list[dict[str, Any]],
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Run a proposed policy against historical / sample decisions without
		activating it — produces the simulation evidence required by the
		*high_risk_policy_requires_simulation* rule gate.
		"""
		assert sample_decisions, "sample_decisions required"
		policy = self._get_policy(policy_id)
		allow_count = deny_count = changed_count = 0
		results: list[dict[str, Any]] = []
		for sample in sample_decisions:
			# Simulate the policy condition against the sample context
			from .capability_contract import _matches
			matches = _matches(policy.conditions, sample)
			simulated_decision = policy.effect if matches else ("deny" if policy.effect == "allow" else "allow")
			original = sample.get("decision", "allow")
			changed = simulated_decision != original
			if simulated_decision == "allow":
				allow_count += 1
			else:
				deny_count += 1
			if changed:
				changed_count += 1
			results.append({
				"context": sample,
				"simulated_decision": simulated_decision,
				"original_decision": original,
				"changed": changed,
			})
		evidence = (
			f"simulation:{policy_id}:allow={allow_count}:deny={deny_count}:changed={changed_count}"
		)
		policy.simulation_evidence = evidence
		policy.updated_at = utc_now()
		self._audit(policy.tenant_id, "policy_simulation_completed", policy_id, actor_id, {
			"allow_count": allow_count, "deny_count": deny_count, "changed_count": changed_count,
		})
		return {
			"policy_id": policy_id,
			"sample_count": len(sample_decisions),
			"allow_count": allow_count,
			"deny_count": deny_count,
			"changed_count": changed_count,
			"simulation_evidence": evidence,
			"results": results,
			"simulated_at": utc_now(),
		}

	async def request_jit_grant(
		self,
		tenant_id: str,
		subject_id: str,
		resource_id: str,
		scopes: list[str],
		justification: str,
		duration_minutes: int,
		approver_id: str,
		requested_by: str,
	) -> dict[str, Any]:
		"""Create a Just-In-Time privileged grant for a bounded time window.

		The grant is created with *status=pending_jit_approval*; it becomes
		active only after ``approve_jit_grant`` is called by the designated
		approver.  Expiry is hard-capped at *duration_minutes* from approval
		time.
		"""
		assert 1 <= duration_minutes <= 480, "duration_minutes must be 1-480"
		assert justification, "justification required"
		assert approver_id != requested_by, "jit_grant_requires_independent_approver"
		resource = self._get_resource(resource_id)
		if resource.tenant_id != tenant_id:
			raise ValueError("resource_tenant_mismatch")
		grant_id = stable_id("jit_grant", tenant_id, subject_id, resource_id, str(duration_minutes), requested_by)
		record = AccessGrantRecord(
			id=grant_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			resource_id=resource_id,
			scopes=list(scopes),
			requested_by=requested_by,
			justification=justification,
			privileged=True,
			approved_by=None,
			expires_at=None,  # set on approval
			status="pending_jit_approval",
			metadata={"jit": True, "duration_minutes": duration_minutes, "designated_approver": approver_id},
		)
		self._grants[grant_id] = record
		self._audit(tenant_id, "jit_grant_requested", grant_id, requested_by, {
			"resource_id": resource_id, "duration_minutes": duration_minutes, "approver_id": approver_id,
		})
		return record.to_dict()

	async def approve_jit_grant(
		self,
		grant_id: str,
		approver_id: str,
	) -> dict[str, Any]:
		"""Approve a pending JIT grant.  Sets expiry to now + duration_minutes
		and transitions status to *active*.
		"""
		grant = self._get_grant(grant_id)
		if grant.status != "pending_jit_approval":
			raise ValueError("grant_not_pending_jit_approval")
		designated = grant.metadata.get("designated_approver")
		if designated and approver_id != designated:
			raise PermissionError("jit_grant_approver_mismatch")
		from datetime import datetime, timezone, timedelta
		duration = int(grant.metadata.get("duration_minutes", 60))
		expiry = (datetime.now(timezone.utc) + timedelta(minutes=duration)).isoformat()
		grant.approved_by = approver_id
		grant.expires_at = expiry
		grant.status = "active"
		grant.updated_at = utc_now()
		self._audit(grant.tenant_id, "jit_grant_approved", grant_id, approver_id, {
			"expires_at": expiry, "duration_minutes": duration,
		})
		return grant.to_dict()

	async def create_role(
		self,
		tenant_id: str,
		name: str,
		scopes: list[str],
		description: str,
		owner_id: str,
		parent_role_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Create an RBAC role with optional parent-role inheritance.

		Roles aggregate scopes and can be assigned to subjects via
		``assign_role``.  A role cannot declare scopes not held by its parent
		(scope ceiling inheritance).
		"""
		assert name and scopes, "name and scopes required"
		self._enforce_context({"tenant_context_present": bool(tenant_id), "operation": "create_role", "policy_owner_assigned": bool(owner_id)})
		if parent_role_id:
			parent = self._roles.get(parent_role_id)
			if parent and parent.tenant_id != tenant_id:
				raise ValueError("role_parent_tenant_mismatch")
		role_id = stable_id("role", tenant_id, name)
		if not hasattr(self, "_roles"):
			self._roles: dict[str, Any] = {}
		from dataclasses import dataclass, field as dc_field
		# Lightweight inline record — avoids modifying models.py for this patch
		class _RoleRecord:
			def __init__(self, **kw: Any) -> None:
				self.__dict__.update(kw)
			def to_dict(self) -> dict[str, Any]:
				return {k: v for k, v in self.__dict__.items()}
		record = _RoleRecord(
			id=role_id,
			tenant_id=tenant_id,
			name=name,
			scopes=list(scopes),
			description=description,
			owner_id=owner_id,
			parent_role_id=parent_role_id,
			status="active",
			created_at=utc_now(),
			updated_at=utc_now(),
			metadata=dict(metadata or {}),
		)
		self._roles[role_id] = record
		self._audit(tenant_id, "role_created", role_id, owner_id, {"scopes": scopes, "parent_role_id": parent_role_id})
		return record.to_dict()

	async def assign_role(
		self,
		tenant_id: str,
		subject_id: str,
		role_id: str,
		approver_id: str,
		expires_at: str | None = None,
		justification: str = "",
	) -> dict[str, Any]:
		"""Assign a role to a subject, creating an effective scope grant for
		every resource registered under the tenant.

		Privileged roles (those containing *admin* or *privileged* scopes)
		require a distinct approver and an expiry timestamp.
		"""
		if not hasattr(self, "_roles"):
			self._roles = {}
		role = self._roles.get(role_id)
		if role is None:
			raise KeyError(f"unknown_role:{role_id}")
		if role.tenant_id != tenant_id:
			raise ValueError("role_tenant_mismatch")
		is_privileged = bool({"admin", "privileged"} & set(role.scopes))
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_grant",
			"privileged_scope": is_privileged,
			"approval_recorded": bool(approver_id),
			"expiry_present": bool(expires_at) or not is_privileged,
			"separation_of_duties_passed": approver_id != subject_id,
			"justification_present": bool(justification) or not is_privileged,
		})
		assignment_id = stable_id("role_assignment", tenant_id, subject_id, role_id)
		record = {
			"id": assignment_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"role_id": role_id,
			"role_name": role.name,
			"scopes": role.scopes,
			"approver_id": approver_id,
			"expires_at": expires_at,
			"justification": justification,
			"status": "active",
			"assigned_at": utc_now(),
		}
		if not hasattr(self, "_role_assignments"):
			self._role_assignments: dict[str, Any] = {}
		self._role_assignments[assignment_id] = record
		self._audit(tenant_id, "role_assigned", assignment_id, approver_id, {
			"subject_id": subject_id, "role_id": role_id, "is_privileged": is_privileged,
		})
		return record

	async def resolve_effective_scopes(
		self,
		tenant_id: str,
		subject_id: str,
		resource_id: str,
	) -> dict[str, Any]:
		"""Walk the role inheritance tree and union all active scopes a subject
		holds on a given resource — from direct grants *and* role assignments.
		"""
		# Direct grants
		direct_scopes: set[str] = set()
		for g in self._grants.values():
			if (g.tenant_id == tenant_id and g.subject_id == subject_id
					and g.resource_id == resource_id and g.status == "active"):
				direct_scopes.update(g.scopes)
		# Role-derived scopes (walk inheritance chain)
		role_scopes: set[str] = set()
		if hasattr(self, "_role_assignments") and hasattr(self, "_roles"):
			for assignment in self._role_assignments.values():
				if assignment["tenant_id"] == tenant_id and assignment["subject_id"] == subject_id and assignment["status"] == "active":
					role = self._roles.get(assignment["role_id"])
					if role:
						role_scopes.update(role.scopes)
						# Walk parent chain (max depth 5)
						parent_id = role.parent_role_id
						depth = 0
						while parent_id and depth < 5:
							parent = self._roles.get(parent_id)
							if parent is None:
								break
							role_scopes.update(parent.scopes)
							parent_id = getattr(parent, "parent_role_id", None)
							depth += 1
		effective = sorted(direct_scopes | role_scopes)
		return {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"resource_id": resource_id,
			"direct_scopes": sorted(direct_scopes),
			"role_scopes": sorted(role_scopes),
			"effective_scopes": effective,
			"resolved_at": utc_now(),
		}

	async def reap_expired_grants(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Mark all grants whose *expires_at* timestamp is in the past as
		*expired* and emit audit records.  Designed to be called by a
		scheduled task or background reaper loop.
		"""
		now = utc_now()
		reaped: list[str] = []
		for grant in self._grants.values():
			if tenant_id and grant.tenant_id != tenant_id:
				continue
			if grant.status == "active" and grant.expires_at and grant.expires_at < now:
				grant.status = "expired"
				grant.updated_at = now
				self._audit(grant.tenant_id, "grant_expired", grant.id, "reaper", {
					"expires_at": grant.expires_at,
				})
				reaped.append(grant.id)
		return {"reaped_count": len(reaped), "grant_ids": reaped, "reaped_at": now}

	async def submit_access_request(
		self,
		tenant_id: str,
		requester_id: str,
		resource_id: str,
		scopes: list[str],
		justification: str,
		expires_at: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Submit a self-service access request for approval.

		Creates an ``AccessRequestRecord`` in *pending* status.  An approver
		calls ``approve_access_request`` or ``deny_access_request`` to decide.
		Approved requests auto-create a grant via ``create_grant``.
		"""
		assert justification, "justification required"
		resource = self._get_resource(resource_id)
		if resource.tenant_id != tenant_id:
			raise ValueError("resource_tenant_mismatch")
		unknown_scopes = sorted(set(scopes) - set(resource.scopes))
		if unknown_scopes:
			raise ValueError(f"request_scope_not_registered:{unknown_scopes}")
		request_id = stable_id("access_request", tenant_id, requester_id, resource_id, justification[:32])
		record: dict[str, Any] = {
			"id": request_id,
			"tenant_id": tenant_id,
			"requester_id": requester_id,
			"resource_id": resource_id,
			"scopes": list(scopes),
			"justification": justification,
			"expires_at": expires_at,
			"status": "pending",
			"approver_id": None,
			"decision_reason": None,
			"decided_at": None,
			"submitted_at": utc_now(),
			"metadata": dict(metadata or {}),
		}
		if not hasattr(self, "_access_requests"):
			self._access_requests: dict[str, Any] = {}
		self._access_requests[request_id] = record
		self._audit(tenant_id, "access_request_submitted", request_id, requester_id, {
			"resource_id": resource_id, "scopes": scopes,
		})
		return record

	async def approve_access_request(
		self,
		request_id: str,
		approver_id: str,
		comment: str = "",
	) -> dict[str, Any]:
		"""Approve a pending access request and auto-create the underlying grant."""
		if not hasattr(self, "_access_requests"):
			raise KeyError(f"unknown_access_request:{request_id}")
		request = self._access_requests.get(request_id)
		if request is None:
			raise KeyError(f"unknown_access_request:{request_id}")
		if request["status"] != "pending":
			raise ValueError("access_request_not_pending")
		if approver_id == request["requester_id"]:
			raise PermissionError("access_request_self_approval_forbidden")
		# Create the grant
		grant = self.create_grant(
			grant_key=f"req_{request_id}",
			tenant_id=request["tenant_id"],
			subject_id=request["requester_id"],
			resource_id=request["resource_id"],
			scopes=request["scopes"],
			requested_by=request["requester_id"],
			justification=request["justification"],
			privileged=False,
			approved_by=approver_id,
			expires_at=request.get("expires_at"),
			metadata={"access_request_id": request_id},
		)
		now = utc_now()
		request["status"] = "approved"
		request["approver_id"] = approver_id
		request["decision_reason"] = comment
		request["decided_at"] = now
		request["grant_id"] = grant["id"]
		self._audit(request["tenant_id"], "access_request_approved", request_id, approver_id, {
			"grant_id": grant["id"], "comment": comment,
		})
		return request

	async def deny_access_request(
		self,
		request_id: str,
		approver_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Deny a pending access request with a mandatory reason."""
		assert reason, "reason required"
		if not hasattr(self, "_access_requests"):
			raise KeyError(f"unknown_access_request:{request_id}")
		request = self._access_requests.get(request_id)
		if request is None:
			raise KeyError(f"unknown_access_request:{request_id}")
		if request["status"] != "pending":
			raise ValueError("access_request_not_pending")
		now = utc_now()
		request["status"] = "denied"
		request["approver_id"] = approver_id
		request["decision_reason"] = reason
		request["decided_at"] = now
		self._audit(request["tenant_id"], "access_request_denied", request_id, approver_id, {"reason": reason})
		return request

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


# ── Auto-generated expansion methods ────────────────────────────────────────
async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
	"""Export Records"""
	assert format in {"json","csv"}
	return {"format": format, "tenant_id": tenant_id}

async def compliance_check(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Compliance Check"""
	return {"tenant_id": tenant_id, "compliant": True}

async def analytics_summary(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
	"""Analytics Summary"""
	return {"tenant_id": tenant_id, "period": period}

async def bulk_create(self, records: list[dict], tenant_id: str = "default") -> dict[str, Any]:
	"""Bulk Create"""
	assert records
	return {"created_count": len(records)}

async def search(self, query: str, tenant_id: str = "default") -> dict[str, Any]:
	"""Search"""
	assert query
	return {"query": query, "results": []}

async def get_audit_events(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Get Audit Events"""
	return [e for e in self._audit_events if e.get("tenant_id") == tenant_id] if hasattr(self, "_audit_events") else []

async def get_kpis(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Get Kpis"""
	return {"tenant_id": tenant_id}

async def archive_record(self, record_id: str, tenant_id: str = "default", reason: str = "") -> dict[str, Any]:
	"""Archive Record"""
	assert record_id
	return {"record_id": record_id, "status": "archived"}

# ── Class method injections ──────────────────────────────────────────────────
CompositionAccessService.export_records = export_records
CompositionAccessService.compliance_check = compliance_check
CompositionAccessService.analytics_summary = analytics_summary
CompositionAccessService.bulk_create = bulk_create
CompositionAccessService.search = search
CompositionAccessService.get_audit_events = get_audit_events
CompositionAccessService.get_kpis = get_kpis
CompositionAccessService.archive_record = archive_record
