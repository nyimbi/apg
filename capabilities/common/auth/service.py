"""Dependency-light AUTH service for identity, RBAC, sessions, and privacy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .capability_contract import (
	SUPPORTED_SECURITY_AGENT_ROLES,
	SUPPORTED_SECURITY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	AuthAccessDecision,
	AuthAuditEvent,
	AuthIdentity,
	AuthPrivacyBudgetApproval,
	AuthPrivacyQuery,
	AuthRole,
	AuthRoleAssignment,
	AuthRoleAssignmentApproval,
	AuthSecurityAgent,
	AuthSession,
)


class AuthService:
	"""Tenant identity control plane backed by the executable AUTH contract."""

	def __init__(self) -> None:
		self._identities: dict[tuple[str, str], AuthIdentity] = {}
		self._roles: dict[tuple[str, str], AuthRole] = {}
		self._role_approvals: dict[tuple[str, str], AuthRoleAssignmentApproval] = {}
		self._assignments: dict[tuple[str, str], AuthRoleAssignment] = {}
		self._sessions: dict[tuple[str, str], AuthSession] = {}
		self._access_decisions: dict[tuple[str, str], AuthAccessDecision] = {}
		self._privacy_queries: dict[tuple[str, str], AuthPrivacyQuery] = {}
		self._privacy_approvals: dict[tuple[str, str], AuthPrivacyBudgetApproval] = {}
		self._security_agents: dict[tuple[str, str], AuthSecurityAgent] = {}
		self._audit_events: dict[tuple[str, str], AuthAuditEvent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_identity(
		self,
		user_id: str,
		tenant_id: str,
		email: str,
		display_name: str,
		status: str = "active",
		tenant_memberships: list[str] | tuple[str, ...] | None = None,
		mfa_enabled: bool = False,
		behavioral_trust_score: float = 1.0,
		biometric_enrolled: bool = False,
		quantum_key_registered: bool = False,
		privacy_budget: float = 1.0,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._ensure_new(self._identities, tenant_id, user_id, "identity")
		if "@" not in email:
			raise ValueError("identity_email_required")
		memberships = tuple(dict.fromkeys((tenant_id, *(tenant_memberships or ()))))
		identity = AuthIdentity(
			id=user_id,
			tenant_id=tenant_id,
			email=email,
			display_name=display_name,
			status=status,
			tenant_memberships=memberships,
			mfa_enabled=mfa_enabled,
			behavioral_trust_score=float(behavioral_trust_score),
			biometric_enrolled=biometric_enrolled,
			quantum_key_registered=quantum_key_registered,
			privacy_budget=float(privacy_budget),
			metadata=dict(metadata or {}),
		)
		self._identities[self._tenant_key(tenant_id, user_id)] = identity
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=user_id,
			event_type="identity_registered",
			actor=str((metadata or {}).get("actor") or "system"),
			decision="allow",
			metadata={"status": status, "mfa_enabled": mfa_enabled},
		)
		return identity.to_dict()

	def list_identities(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._identities.values(), tenant_id)

	def define_role(
		self,
		role_id: str,
		tenant_id: str,
		name: str,
		permissions: list[str] | tuple[str, ...],
		tier: str = "standard",
		approval_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._ensure_new(self._roles, tenant_id, role_id, "role")
		if not permissions:
			raise ValueError("role_permissions_required")
		role = AuthRole(
			id=role_id,
			tenant_id=tenant_id,
			name=name,
			permissions=tuple(sorted(set(permissions))),
			tier=tier,
			approval_recorded=approval_recorded,
		)
		self._roles[self._tenant_key(tenant_id, role_id)] = role
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=role_id,
			event_type="role_defined",
			actor="system",
			decision="allow",
			metadata={"tier": tier, "permission_count": len(role.permissions)},
		)
		return role.to_dict()

	def list_roles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._roles.values(), tenant_id)

	def request_role_assignment_approval(
		self,
		approval_id: str,
		tenant_id: str,
		user_id: str,
		role_id: str,
		requested_by: str,
		justification: str,
	) -> dict[str, Any]:
		self._require_identity(user_id, tenant_id)
		self._require_actor_permission(requested_by, tenant_id, "auth:manage_roles")
		role = self._require_role(role_id, tenant_id)
		self._ensure_new(self._role_approvals, tenant_id, approval_id, "role assignment approval")
		if not requested_by:
			raise ValueError("role_approval_requester_required")
		if not justification:
			raise ValueError("role_approval_justification_required")
		approval = AuthRoleAssignmentApproval(
			id=approval_id,
			tenant_id=tenant_id,
			user_id=user_id,
			role_id=role_id,
			requested_by=requested_by,
			justification=justification,
		)
		self._role_approvals[self._tenant_key(tenant_id, approval_id)] = approval
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="role_assignment_approval_requested",
			actor=requested_by,
			decision="require_review",
			metadata={"user_id": user_id, "role_id": role_id, "role_tier": role.tier},
		)
		return approval.to_dict()

	def decide_role_assignment_approval(
		self,
		approval_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		approval = self._require_role_approval(approval_id, tenant_id)
		if approval.status != "pending":
			raise ValueError("role_approval_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("role_approval_decision_invalid")
		if not reviewer:
			raise ValueError("role_approval_reviewer_required")
		if not notes:
			raise ValueError("role_approval_notes_required")
		self._require_actor_permission(reviewer, tenant_id, "auth:approve_roles")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"requested_operation": "approve_role_assignment",
			"reviewer_same_as_requester": reviewer == approval.requested_by,
		})
		self._raise_if_denied(result)
		decided = replace(
			approval,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			status=decision,
		)
		self._role_approvals[self._tenant_key(tenant_id, approval_id)] = decided
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="role_assignment_approval_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"user_id": approval.user_id, "role_id": approval.role_id},
		)
		return decided.to_dict()

	def list_role_assignment_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._role_approvals.values(), tenant_id)

	def assign_role(
		self,
		assignment_id: str,
		tenant_id: str,
		user_id: str,
		role_id: str,
		assigned_by: str,
		approval_recorded: bool = False,
		approval_id: str | None = None,
	) -> dict[str, Any]:
		identity = self._require_identity(user_id, tenant_id)
		role = self._require_role(role_id, tenant_id)
		self._require_actor_permission(assigned_by, tenant_id, "auth:manage_roles")
		self._ensure_new(self._assignments, tenant_id, assignment_id, "role assignment")
		approval = self._approved_role_assignment_approval(tenant_id, approval_id, user_id, role_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"requested_operation": "assign_role",
			"role_tier": role.tier,
			"approval_recorded": approval is not None or (role.tier != "admin" and (approval_recorded or role.approval_recorded)),
			"user_locked": identity.status in {"locked", "suspended"},
		})
		self._raise_if_denied(result)
		assignment = AuthRoleAssignment(
			id=assignment_id,
			tenant_id=tenant_id,
			user_id=user_id,
			role_id=role_id,
			assigned_by=assigned_by,
			approval_id=approval.id if approval else approval_id,
			approval_recorded=approval is not None or approval_recorded or role.approval_recorded,
		)
		self._assignments[self._tenant_key(tenant_id, assignment_id)] = assignment
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=assignment_id,
			event_type="role_assigned",
			actor=assigned_by,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"user_id": user_id, "role_id": role_id, "role_tier": role.tier},
		)
		return assignment.to_dict()

	def list_role_assignments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._assignments.values(), tenant_id)

	def start_session(
		self,
		session_id: str,
		tenant_id: str,
		user_id: str,
		device_id: str,
		auth_source: str = "local",
		issuer_trusted: bool = True,
		mfa_verified: bool = False,
		risk_level: str = "low",
		step_up_completed: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._ensure_new(self._sessions, tenant_id, session_id, "session")
		identity, tenant_membership_confirmed = self._identity_for_tenant_decision(user_id, tenant_id)
		context = {
			"user_locked": identity.status in {"locked", "suspended"},
			"requested_permission_tier": "standard",
			"mfa_verified": bool(mfa_verified),
			"risk_level": risk_level,
			"step_up_completed": bool(step_up_completed),
			"auth_source": auth_source,
			"issuer_trusted": bool(issuer_trusted),
			"tenant_mismatch": tenant_id != identity.tenant_id,
			"tenant_membership_confirmed": tenant_membership_confirmed,
		}
		result = self.evaluate(context)
		self._raise_if_denied(result)
		required_actions = tuple(
			str(action.get("required_action"))
			for action in result["actions"]
			if action.get("required_action")
		)
		trust_score = self._session_trust(identity, risk_level, mfa_verified, step_up_completed)
		session = AuthSession(
			id=session_id,
			tenant_id=tenant_id,
			user_id=user_id,
			device_id=device_id,
			auth_source=auth_source,
			risk_level=risk_level,
			mfa_verified=mfa_verified,
			step_up_completed=step_up_completed,
			issuer_trusted=issuer_trusted,
			trust_score=trust_score,
			required_actions=required_actions,
		)
		self._sessions[self._tenant_key(tenant_id, session_id)] = session
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=session_id,
			event_type="session_started",
			actor=user_id,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"risk_level": risk_level, "auth_source": auth_source, "trust_score": trust_score},
		)
		return session.to_dict()

	def revoke_session(self, session_id: str, actor: str, tenant_id: str | None = None) -> dict[str, Any]:
		session = self._find_session(session_id, tenant_id) if tenant_id else self._require_session(session_id)
		revoked = replace(session, status="revoked")
		self._sessions[self._tenant_key(session.tenant_id, session_id)] = revoked
		self._record_audit(
			tenant_id=revoked.tenant_id,
			subject_id=session_id,
			event_type="session_revoked",
			actor=actor,
			decision="allow",
		)
		return revoked.to_dict()

	def list_sessions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sessions.values(), tenant_id)

	def evaluate_access(
		self,
		decision_id: str,
		tenant_id: str,
		user_id: str,
		permission: str,
		session_id: str | None = None,
		requested_permission_tier: str | None = None,
		mfa_verified: bool | None = None,
		step_up_completed: bool | None = None,
		risk_level: str | None = None,
	) -> dict[str, Any]:
		identity, tenant_membership_confirmed = self._identity_for_tenant_decision(user_id, tenant_id)
		session = self._find_session(session_id, tenant_id) if session_id else None
		role_ids = self._active_role_ids(user_id, tenant_id, permission)
		effective_permission_tier = requested_permission_tier or self._permission_tier(permission, role_ids, tenant_id)
		context = {
			"user_locked": identity.status in {"locked", "suspended"},
			"requested_permission_tier": effective_permission_tier,
			"mfa_verified": session.mfa_verified if mfa_verified is None and session else bool(mfa_verified),
			"risk_level": risk_level or (session.risk_level if session else "low"),
			"step_up_completed": session.step_up_completed if step_up_completed is None and session else bool(step_up_completed),
			"tenant_mismatch": tenant_id != identity.tenant_id,
			"tenant_membership_confirmed": tenant_membership_confirmed,
		}
		result = self.evaluate(context)
		decision = result["decision"]
		reasons = list(self._reasons(result))
		if decision == "allow" and not role_ids:
			decision = "deny"
			reasons.append("permission_not_granted")
		access_decision = AuthAccessDecision(
			id=decision_id,
			tenant_id=tenant_id,
			user_id=user_id,
			permission=permission,
			decision=decision,
			matched_rules=tuple(result["matched_rules"]),
			reasons=tuple(reasons),
			session_id=session_id,
			role_ids=tuple(role_ids),
		)
		self._ensure_new(self._access_decisions, tenant_id, decision_id, "access decision")
		self._access_decisions[self._tenant_key(tenant_id, decision_id)] = access_decision
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=decision_id,
			event_type="access_evaluated",
			actor=user_id,
			decision=decision,
			reasons=tuple(reasons),
			metadata={"permission": permission, "role_ids": role_ids},
		)
		return access_decision.to_dict()

	def list_access_decisions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._access_decisions.values(), tenant_id)

	def request_privacy_budget_approval(
		self,
		approval_id: str,
		tenant_id: str,
		user_id: str,
		query_type: str,
		epsilon_cost: float,
		requested_by: str,
		justification: str,
	) -> dict[str, Any]:
		identity, tenant_membership_confirmed = self._tenant_local_privacy_identity(user_id, tenant_id)
		self._require_actor_permission(requested_by, tenant_id, "auth:manage_privacy")
		result = self.evaluate({
			"tenant_mismatch": tenant_id != identity.tenant_id,
			"tenant_membership_confirmed": tenant_membership_confirmed,
		})
		self._raise_if_denied(result)
		self._ensure_new(self._privacy_approvals, tenant_id, approval_id, "privacy budget approval")
		if float(epsilon_cost) <= 0:
			raise ValueError("privacy_approval_epsilon_cost_required")
		if not requested_by:
			raise ValueError("privacy_approval_requester_required")
		if not justification:
			raise ValueError("privacy_approval_justification_required")
		approval = AuthPrivacyBudgetApproval(
			id=approval_id,
			tenant_id=tenant_id,
			user_id=user_id,
			query_type=query_type,
			epsilon_cost=float(epsilon_cost),
			requested_by=requested_by,
			justification=justification,
		)
		self._privacy_approvals[self._tenant_key(tenant_id, approval_id)] = approval
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="privacy_budget_approval_requested",
			actor=requested_by,
			decision="require_review",
			reasons=self._reasons(result),
			metadata={"user_id": user_id, "query_type": query_type, "epsilon_cost": float(epsilon_cost)},
		)
		return approval.to_dict()

	def decide_privacy_budget_approval(
		self,
		approval_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		approval = self._require_privacy_budget_approval(approval_id, tenant_id)
		if approval.status != "pending":
			raise ValueError("privacy_approval_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("privacy_approval_decision_invalid")
		if not reviewer:
			raise ValueError("privacy_approval_reviewer_required")
		if not notes:
			raise ValueError("privacy_approval_notes_required")
		self._require_actor_permission(reviewer, tenant_id, "auth:approve_privacy")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"requested_operation": "approve_privacy_budget",
			"reviewer_same_as_requester": reviewer == approval.requested_by,
		})
		self._raise_if_denied(result)
		decided = replace(
			approval,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			status=decision,
		)
		self._privacy_approvals[self._tenant_key(tenant_id, approval_id)] = decided
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="privacy_budget_approval_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"user_id": approval.user_id, "query_type": approval.query_type},
		)
		return decided.to_dict()

	def list_privacy_budget_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._privacy_approvals.values(), tenant_id)

	def run_privacy_query(
		self,
		query_id: str,
		tenant_id: str,
		user_id: str,
		query_type: str,
		epsilon_cost: float,
		approval_recorded: bool = False,
		approval_id: str | None = None,
	) -> dict[str, Any]:
		self._ensure_new(self._privacy_queries, tenant_id, query_id, "privacy query")
		identity, tenant_membership_confirmed = self._tenant_local_privacy_identity(user_id, tenant_id)
		if float(epsilon_cost) <= 0:
			raise ValueError("privacy_query_epsilon_cost_required")
		result = self.evaluate({
			"tenant_mismatch": tenant_id != identity.tenant_id,
			"tenant_membership_confirmed": tenant_membership_confirmed,
		})
		self._raise_if_denied(result)
		budget_available = identity.privacy_budget >= float(epsilon_cost)
		privacy_result = self.evaluate({
			"requested_operation": "privacy_analytics_query",
			"privacy_budget_available": budget_available,
		})
		decision = privacy_result["decision"]
		reasons = list(self._reasons(privacy_result))
		approval = (
			self._approved_privacy_budget_approval(tenant_id, approval_id, user_id, query_type, float(epsilon_cost))
			if approval_id else None
		)
		if decision == "require_review" and approval is None:
			status = "review_required"
			remaining_budget = identity.privacy_budget
		else:
			status = "completed"
			remaining_budget = max(identity.privacy_budget - float(epsilon_cost), 0.0)
			self._identities[self._tenant_key(identity.tenant_id, identity.id)] = replace(identity, privacy_budget=remaining_budget)
		query = AuthPrivacyQuery(
			id=query_id,
			tenant_id=tenant_id,
			user_id=user_id,
			query_type=query_type,
			epsilon_cost=float(epsilon_cost),
			status=status,
			remaining_budget=remaining_budget,
			approval_recorded=approval is not None,
			reasons=tuple(reasons),
			approval_id=approval.id if approval else approval_id,
		)
		self._privacy_queries[self._tenant_key(tenant_id, query_id)] = query
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=query_id,
			event_type="privacy_query_evaluated",
			actor=user_id,
			decision=decision,
			reasons=tuple(reasons),
			metadata={"query_type": query_type, "epsilon_cost": float(epsilon_cost)},
		)
		return query.to_dict()

	def list_privacy_queries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._privacy_queries.values(), tenant_id)

	def register_security_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		registered: bool = True,
		contribution_disclosed: bool = True,
		policy_ref: str | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"security_agent_present": True,
			"agent_registered": registered,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_SECURITY_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_SECURITY_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		self._ensure_new(self._security_agents, tenant_id, agent_id, "security agent")
		if not name:
			raise ValueError("security_agent_name_required")
		agent = AuthSecurityAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref,
			status=status,
		)
		self._security_agents[self._tenant_key(tenant_id, agent_id)] = agent
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=agent_id,
			event_type="security_agent_registered",
			actor="system",
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"runtime": agent.runtime, "role": agent.role, "scope": scope},
		)
		return agent.to_dict()

	def list_security_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._security_agents.values(), tenant_id)

	def validate_batch_auth_mutation(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"requested_operation": "batch_auth_mutation",
			"event_stream": event_stream,
			"mutation_count": mutation_count,
		})
		self._raise_if_denied(result)
		return {
			"tenant_id": tenant_id,
			"event_stream": event_stream,
			"mutation_count": mutation_count,
			"accepted": True,
			"rule_result": result,
		}

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		identities = [item for item in self._identities.values() if item.tenant_id == tenant_id]
		sessions = [item for item in self._sessions.values() if item.tenant_id == tenant_id]
		decisions = [item for item in self._access_decisions.values() if item.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"identity_count": len(identities),
			"active_session_count": len([item for item in sessions if item.status == "active"]),
			"role_count": len([item for item in self._roles.values() if item.tenant_id == tenant_id]),
			"admin_assignment_count": len([
				assignment for assignment in self._assignments.values()
				if assignment.tenant_id == tenant_id and self._role_for_assignment(assignment).tier == "admin"
			]),
			"role_approval_count": len([item for item in self._role_approvals.values() if item.tenant_id == tenant_id]),
			"pending_role_approval_count": len([
				item for item in self._role_approvals.values()
				if item.tenant_id == tenant_id and item.status == "pending"
			]),
			"privacy_approval_count": len([item for item in self._privacy_approvals.values() if item.tenant_id == tenant_id]),
			"pending_privacy_approval_count": len([
				item for item in self._privacy_approvals.values()
				if item.tenant_id == tenant_id and item.status == "pending"
			]),
			"security_agent_count": len([item for item in self._security_agents.values() if item.tenant_id == tenant_id]),
			"denied_decision_count": len([item for item in decisions if item.decision == "deny"]),
			"privacy_review_count": len([
				item for item in self._privacy_queries.values()
				if item.tenant_id == tenant_id and item.status == "review_required"
			]),
			"average_trust_score": (
				sum(item.trust_score for item in sessions) / len(sessions)
				if sessions else 0.0
			),
		}

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events.values(), tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing identities as AUTH records."""
		return self.list_identities(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that registers a minimal tenant identity."""
		metadata = dict(metadata or {})
		return self.register_identity(
			user_id=record_id,
			tenant_id=tenant_id,
			email=str(metadata.get("email") or f"{record_id}@example.invalid"),
			display_name=str(metadata.get("display_name") or record_id),
			status=status,
			tenant_memberships=metadata.get("tenant_memberships") or (),
			mfa_enabled=_coerce_bool(metadata.get("mfa_enabled", False)),
			privacy_budget=float(metadata.get("privacy_budget", 1.0)),
			metadata=metadata,
		)

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _ensure_new(self, records: dict[tuple[str, str], Any], tenant_id: str, record_id: str, label: str) -> None:
		if self._tenant_key(tenant_id, record_id) in records:
			raise ValueError(f"{label} already exists for tenant: {record_id}")

	def _require_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			raise PermissionError("tenant_context_required")

	def _require_identity(self, user_id: str, tenant_id: str, allow_membership: bool = False) -> AuthIdentity:
		self._require_tenant(tenant_id)
		identity = self._identities.get(self._tenant_key(tenant_id, user_id))
		if identity is not None:
			return identity
		if allow_membership:
			for candidate in self._identities.values():
				if candidate.id == user_id and tenant_id in candidate.tenant_memberships:
					return candidate
		raise KeyError(f"identity not in tenant: {user_id}")

	def _identity_for_tenant_decision(self, user_id: str, tenant_id: str) -> tuple[AuthIdentity, bool]:
		self._require_tenant(tenant_id)
		identity = self._identities.get(self._tenant_key(tenant_id, user_id))
		if identity is not None:
			return identity, True
		candidates = [candidate for candidate in self._identities.values() if candidate.id == user_id]
		for candidate in candidates:
			if tenant_id in candidate.tenant_memberships:
				return candidate, True
		if len(candidates) == 1:
			return candidates[0], False
		raise KeyError(f"identity not in tenant: {user_id}")

	def _tenant_local_privacy_identity(self, user_id: str, tenant_id: str) -> tuple[AuthIdentity, bool]:
		self._require_tenant(tenant_id)
		identity = self._identities.get(self._tenant_key(tenant_id, user_id))
		if identity is not None:
			return identity, True
		candidates = [candidate for candidate in self._identities.values() if candidate.id == user_id]
		if candidates:
			result = self.evaluate({
				"tenant_mismatch": True,
				"tenant_membership_confirmed": False,
			})
			self._raise_if_denied(result)
		raise KeyError(f"identity not in tenant: {user_id}")

	def _require_actor_permission(self, actor: str, tenant_id: str, permission: str) -> None:
		if not actor:
			raise ValueError("actor_required")
		if actor == "system":
			return
		identity, tenant_membership_confirmed = self._identity_for_tenant_decision(actor, tenant_id)
		result = self.evaluate({
			"tenant_mismatch": tenant_id != identity.tenant_id,
			"tenant_membership_confirmed": tenant_membership_confirmed,
			"user_locked": identity.status in {"locked", "suspended"},
		})
		self._raise_if_denied(result)
		if permission not in self._actor_permissions(actor, tenant_id):
			raise PermissionError(f"{permission.replace(':', '_')}_required")

	def _require_role(self, role_id: str, tenant_id: str) -> AuthRole:
		self._require_tenant(tenant_id)
		role = self._roles.get(self._tenant_key(tenant_id, role_id))
		if role is None:
			raise KeyError(f"unknown role: {role_id}")
		return role

	def _require_role_approval(self, approval_id: str, tenant_id: str) -> AuthRoleAssignmentApproval:
		self._require_tenant(tenant_id)
		approval = self._role_approvals.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise KeyError(f"unknown role assignment approval: {approval_id}")
		return approval

	def _require_privacy_budget_approval(self, approval_id: str, tenant_id: str) -> AuthPrivacyBudgetApproval:
		self._require_tenant(tenant_id)
		approval = self._privacy_approvals.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise KeyError(f"unknown privacy budget approval: {approval_id}")
		return approval

	def _require_session(self, session_id: str) -> AuthSession:
		matches = [session for (_, item_id), session in self._sessions.items() if item_id == session_id]
		if len(matches) > 1:
			raise KeyError(f"session ID is ambiguous across tenants: {session_id}")
		session = matches[0] if matches else None
		if session is None:
			raise KeyError(f"unknown session: {session_id}")
		return session

	def _find_session(self, session_id: str, tenant_id: str) -> AuthSession:
		session = self._sessions.get(self._tenant_key(tenant_id, session_id))
		if session is None:
			raise KeyError(f"session not in tenant: {session_id}")
		return session

	def _approved_role_assignment_approval(
		self,
		tenant_id: str,
		approval_id: str | None,
		user_id: str,
		role_id: str,
	) -> AuthRoleAssignmentApproval | None:
		if approval_id is None:
			return None
		approval = self._role_approvals.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise PermissionError("role_assignment_approval_required")
		if approval.user_id != user_id or approval.role_id != role_id:
			raise PermissionError("role_assignment_approval_mismatch")
		if approval.status != "approved":
			raise PermissionError("role_assignment_approval_not_approved")
		return approval

	def _approved_privacy_budget_approval(
		self,
		tenant_id: str,
		approval_id: str | None,
		user_id: str,
		query_type: str,
		epsilon_cost: float,
	) -> AuthPrivacyBudgetApproval | None:
		if approval_id is None:
			return None
		approval = self._privacy_approvals.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise PermissionError("privacy_budget_approval_required")
		if (
			approval.user_id != user_id
			or approval.query_type != query_type
			or abs(approval.epsilon_cost - float(epsilon_cost)) > 0.000001
		):
			raise PermissionError("privacy_budget_approval_mismatch")
		if approval.status != "approved":
			raise PermissionError("privacy_budget_approval_not_approved")
		return approval

	def _role_for_assignment(self, assignment: AuthRoleAssignment) -> AuthRole:
		return self._require_role(assignment.role_id, assignment.tenant_id)

	def _active_role_ids(self, user_id: str, tenant_id: str, permission: str) -> list[str]:
		role_ids: list[str] = []
		for assignment in self._assignments.values():
			if assignment.user_id != user_id or assignment.tenant_id != tenant_id or assignment.status != "active":
				continue
			role = self._roles.get(self._tenant_key(tenant_id, assignment.role_id))
			if role and role.status == "active" and permission in role.permissions:
				role_ids.append(role.id)
		return sorted(role_ids)

	def _actor_permissions(self, user_id: str, tenant_id: str) -> set[str]:
		permissions: set[str] = set()
		for assignment in self._assignments.values():
			if assignment.user_id != user_id or assignment.tenant_id != tenant_id or assignment.status != "active":
				continue
			role = self._roles.get(self._tenant_key(tenant_id, assignment.role_id))
			if role and role.status == "active":
				permissions.update(role.permissions)
		return permissions

	def _permission_tier(self, permission: str, role_ids: list[str], tenant_id: str) -> str:
		if permission.endswith(":admin") or permission.endswith(":approve") or ":admin" in permission:
			return "privileged"
		for role_id in role_ids:
			role = self._roles.get(self._tenant_key(tenant_id, role_id))
			if role and role.tier in {"admin", "privileged"}:
				return "privileged"
		return "standard"

	def _session_trust(
		self,
		identity: AuthIdentity,
		risk_level: str,
		mfa_verified: bool,
		step_up_completed: bool,
	) -> float:
		risk_penalty = {"low": 0.0, "medium": 0.15, "high": 0.35}.get(risk_level, 0.2)
		mfa_bonus = 0.05 if mfa_verified else 0.0
		step_up_bonus = 0.05 if step_up_completed else 0.0
		return round(max(min(identity.behavioral_trust_score - risk_penalty + mfa_bonus + step_up_bonus, 1.0), 0.0), 3)

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> AuthAuditEvent:
		event = AuthAuditEvent(
			id=f"auth-audit-{len(self._audit_events) + 1:04d}",
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[self._tenant_key(tenant_id, event.id)] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] != "allow":
			reasons = ", ".join(self._reasons(result))
			raise PermissionError(reasons or "capability_policy_blocked")

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			str(action.get("reason") or action.get("required_action") or "capability_policy_blocked")
			for action in result.get("actions", [])
		)

	def _list(self, values: Any, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


def _coerce_bool(value: Any) -> bool:
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")
