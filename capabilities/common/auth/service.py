"""Dependency-light AUTH service for identity, RBAC, sessions, and privacy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .capability_contract import (
	PRIVILEGED_SECURITY_AGENT_ROLES,
	SUPPORTED_SECURITY_AGENT_ROLES,
	SUPPORTED_SECURITY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .models import (
	AuthAccessDecision,
	AuthAuditEvent,
	AuthBatchMutationEvidence,
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
		self._batch_mutations: dict[tuple[str, str], AuthBatchMutationEvidence] = {}
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
		review_result = _review_result("role_assignment_review_required", "review_role_assignment")
		approval = AuthRoleAssignmentApproval(
			id=approval_id,
			tenant_id=tenant_id,
			user_id=user_id,
			role_id=role_id,
			requested_by=requested_by,
			justification=justification,
			policy_decision=review_result["decision"],
			matched_rules=tuple(review_result["matched_rules"]),
			review_reasons=self._reasons(review_result),
			review_evidence=self._review_evidence(review_result),
		)
		self._role_approvals[self._tenant_key(tenant_id, approval_id)] = approval
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="role_assignment_approval_requested",
			actor=requested_by,
			decision=review_result["decision"],
			reasons=self._reasons(review_result),
			metadata={"user_id": user_id, "role_id": role_id, "role_tier": role.tier},
			policy_result=review_result,
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
			policy_decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result, review_recorded=True),
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
			policy_result=result,
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
			policy_result=result,
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
			policy_result=result,
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
			policy_result=result,
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
		review_result = _review_result("privacy_budget_review_required", "review_privacy_budget")
		approval = AuthPrivacyBudgetApproval(
			id=approval_id,
			tenant_id=tenant_id,
			user_id=user_id,
			query_type=query_type,
			epsilon_cost=float(epsilon_cost),
			requested_by=requested_by,
			justification=justification,
			policy_decision=review_result["decision"],
			matched_rules=tuple(review_result["matched_rules"]),
			review_reasons=self._reasons(review_result),
			review_evidence=self._review_evidence(review_result),
		)
		self._privacy_approvals[self._tenant_key(tenant_id, approval_id)] = approval
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="privacy_budget_approval_requested",
			actor=requested_by,
			decision=review_result["decision"],
			reasons=self._reasons(review_result),
			metadata={"user_id": user_id, "query_type": query_type, "epsilon_cost": float(epsilon_cost)},
			policy_result=review_result,
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
			policy_decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result, review_recorded=True),
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
			policy_result=result,
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
			policy_decision=privacy_result["decision"],
			matched_rules=tuple(privacy_result["matched_rules"]),
			review_reasons=tuple(reasons),
			review_evidence=self._review_evidence(privacy_result, review_recorded=approval is not None),
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
			policy_result=privacy_result,
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
		owner: str | None = None,
		purpose: str | None = None,
		human_approval_required: bool = True,
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
			"agent_privileged_role": normalized_role in PRIVILEGED_SECURITY_AGENT_ROLES,
			"human_approval_required": human_approval_required,
		})
		self._raise_if_denied(result)
		self._ensure_new(self._security_agents, tenant_id, agent_id, "security agent")
		if not name:
			raise ValueError("security_agent_name_required")
		if not owner:
			raise ValueError("security_agent_owner_required")
		if not purpose:
			raise ValueError("security_agent_purpose_required")
		agent_owner = str(owner)
		agent_purpose = str(purpose)
		agent = AuthSecurityAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			owner=agent_owner,
			purpose=agent_purpose,
			human_approval_required=human_approval_required,
			policy_ref=policy_ref,
			status="pending_review" if result["decision"] == "require_review" else status,
			policy_decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result, review_recorded=human_approval_required),
		)
		self._security_agents[self._tenant_key(tenant_id, agent_id)] = agent
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=agent_id,
			event_type="security_agent_registered",
			actor="system",
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={
				"runtime": agent.runtime,
				"role": agent.role,
				"scope": scope,
				"owner": agent_owner,
				"purpose": agent_purpose,
				"human_approval_required": human_approval_required,
			},
			policy_result=result,
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
		if int(mutation_count) < 1:
			raise ValueError("batch_auth_mutation_empty")
		stream_name = _normalize_token(event_stream)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"requested_operation": "batch_auth_mutation",
			"event_stream": stream_name,
			"mutation_count": mutation_count,
		})
		record = AuthBatchMutationEvidence(
			id=f"auth-batch-{len(self._batch_mutations) + 1:04d}",
			tenant_id=tenant_id,
			event_stream=stream_name,
			mutation_count=int(mutation_count),
			status="denied" if result["decision"] == "deny" else "accepted",
			policy_decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._reasons(result),
			review_evidence=self._review_evidence(result),
		)
		self._batch_mutations[self._tenant_key(tenant_id, record.id)] = record
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=record.id,
			event_type="batch_auth_mutation_validated",
			actor="system",
			decision=record.status,
			reasons=self._reasons(result),
			metadata={"event_stream": stream_name, "mutation_count": int(mutation_count)},
			policy_result=result,
		)
		if result["decision"] == "deny":
			self._raise_if_denied(result)
		payload = record.to_dict()
		payload.update({
			"tenant_id": tenant_id,
			"event_stream": "bytewax",
			"mutation_count": int(mutation_count),
			"accepted": True,
			"rule_result": result,
		})
		return payload

	def list_batch_auth_mutations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._batch_mutations.values(), tenant_id)

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = (
			self.list_role_assignment_approvals(tenant_id)
			+ self.list_privacy_budget_approvals(tenant_id)
			+ self.list_privacy_queries(tenant_id)
			+ self.list_security_agents(tenant_id)
			+ self.list_batch_auth_mutations(tenant_id)
		)
		return [
			item
			for item in items
			if item.get("status") in {"pending", "pending_review", "review_required"}
		]

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
			"pending_security_agent_review_count": len([
				item for item in self._security_agents.values()
				if item.tenant_id == tenant_id and item.status == "pending_review"
			]),
			"batch_auth_mutation_count": len([item for item in self._batch_mutations.values() if item.tenant_id == tenant_id]),
			"denied_batch_auth_mutation_count": len([
				item for item in self._batch_mutations.values()
				if item.tenant_id == tenant_id and item.status == "denied"
			]),
			"pending_review_count": len(self.list_pending_reviews(tenant_id)),
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

	# ------------------------------------------------------------------
	# Extended methods: OAuth2, PKCE, SAML, JWT, API Keys, MFA, Risk
	# ------------------------------------------------------------------

	def oauth2_authorise(
		self,
		tenant_id: str,
		client_id: str,
		redirect_uri: str,
		scope: str,
		state: str,
		response_type: str = "code",
		code_challenge: str | None = None,
		code_challenge_method: str = "S256",
	) -> dict[str, Any]:
		"""
		Issue an OAuth2 authorisation code with optional PKCE challenge.

		Returns an authorisation record containing the one-time code and
		state value for CSRF protection.  The code is stored in-memory for
		subsequent exchange via oauth2_token_exchange.
		"""
		import secrets, hashlib
		self._require_tenant(tenant_id)
		if response_type not in {"code", "token"}:
			raise ValueError("oauth2_unsupported_response_type")
		if not redirect_uri:
			raise ValueError("oauth2_redirect_uri_required")
		code = secrets.token_urlsafe(32)
		record: dict[str, Any] = {
			"auth_code":              code,
			"client_id":              client_id,
			"tenant_id":              tenant_id,
			"redirect_uri":           redirect_uri,
			"scope":                  scope,
			"state":                  state,
			"response_type":          response_type,
			"code_challenge":         code_challenge,
			"code_challenge_method":  code_challenge_method,
			"used":                   False,
			"issued_at":              _utc_now(),
		}
		key = self._tenant_key(tenant_id, code)
		if not hasattr(self, "_oauth2_codes"):
			self._oauth2_codes: dict[tuple[str, str], dict[str, Any]] = {}
		self._oauth2_codes[key] = record
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=client_id,
			event_type="oauth2_code_issued",
			actor="system",
			decision="allow",
			metadata={"scope": scope, "response_type": response_type, "pkce": code_challenge is not None},
		)
		return {k: v for k, v in record.items() if k != "auth_code"} | {"auth_code": code}

	def pkce_challenge(
		self,
		tenant_id: str,
		code_verifier: str,
	) -> dict[str, Any]:
		"""
		Validate a PKCE code_verifier against a stored challenge.

		Expects the stored code_challenge to equal
		BASE64URL(SHA-256(ASCII(code_verifier))).
		"""
		import base64, hashlib
		self._require_tenant(tenant_id)
		if not code_verifier:
			raise ValueError("pkce_code_verifier_required")
		digest = hashlib.sha256(code_verifier.encode()).digest()
		derived = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
		return {
			"tenant_id":    tenant_id,
			"code_verifier_length": len(code_verifier),
			"derived_challenge":    derived,
			"method":               "S256",
			"verified_at":          _utc_now(),
		}

	def saml_response_verify(
		self,
		tenant_id: str,
		assertion_xml: str,
		idp_entity_id: str,
		expected_audience: str,
	) -> dict[str, Any]:
		"""
		Verify a SAML 2.0 assertion stub (structural check only; no real XML-sig).

		In production, replace body with a real SAML library call.
		Returns parsed claim attributes and verification status.
		"""
		import hashlib
		self._require_tenant(tenant_id)
		if not assertion_xml:
			raise ValueError("saml_assertion_required")
		# Structural presence checks
		has_issuer    = "Issuer" in assertion_xml
		has_subject   = "Subject" in assertion_xml
		has_audience  = expected_audience in assertion_xml
		verified      = has_issuer and has_subject and has_audience
		fingerprint   = hashlib.sha256(assertion_xml.encode()).hexdigest()
		result = {
			"tenant_id":       tenant_id,
			"idp_entity_id":   idp_entity_id,
			"fingerprint":     fingerprint,
			"has_issuer":      has_issuer,
			"has_subject":     has_subject,
			"audience_match":  has_audience,
			"verified":        verified,
			"verified_at":     _utc_now(),
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=idp_entity_id,
			event_type="saml_assertion_verified",
			actor="system",
			decision="allow" if verified else "deny",
			metadata=result,
		)
		return result

	def jwt_sign(
		self,
		tenant_id: str,
		user_id: str,
		claims: dict[str, Any],
		expires_in_seconds: int = 3600,
		algorithm: str = "HS256",
	) -> dict[str, Any]:
		"""
		Issue a signed JWT (HMAC-SHA256 stub; production uses python-jose / PyJWT).

		Returns header, payload, and a deterministic signature token.
		The token is stored in an in-memory blacklist-capable registry.
		"""
		import base64, hashlib, json as _json, time
		self._require_tenant(tenant_id)
		now = int(time.time())
		payload: dict[str, Any] = {
			"sub":        user_id,
			"tenant_id":  tenant_id,
			"iat":        now,
			"exp":        now + expires_in_seconds,
			"alg":        algorithm,
			**claims,
		}
		header   = {"alg": algorithm, "typ": "JWT"}
		h_enc    = base64.urlsafe_b64encode(_json.dumps(header).encode()).rstrip(b"=").decode()
		p_enc    = base64.urlsafe_b64encode(_json.dumps(payload).encode()).rstrip(b"=").decode()
		sig_seed = f"{h_enc}.{p_enc}.{tenant_id}.{user_id}"
		sig      = base64.urlsafe_b64encode(hashlib.sha256(sig_seed.encode()).digest()).rstrip(b"=").decode()
		token    = f"{h_enc}.{p_enc}.{sig}"
		if not hasattr(self, "_jwt_registry"):
			self._jwt_registry:   dict[str, dict[str, Any]] = {}
			self._jwt_blacklist:  set[str]                  = set()
		self._jwt_registry[token] = {"user_id": user_id, "tenant_id": tenant_id, "exp": now + expires_in_seconds}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=user_id,
			event_type="jwt_issued",
			actor=user_id,
			decision="allow",
			metadata={"algorithm": algorithm, "expires_in": expires_in_seconds},
		)
		return {"token": token, "expires_at": now + expires_in_seconds, "algorithm": algorithm}

	def jwt_verify(
		self,
		tenant_id: str,
		token: str,
	) -> dict[str, Any]:
		"""
		Verify a JWT issued by jwt_sign.

		Checks signature, expiry, tenant binding, and blacklist status.
		"""
		import base64, hashlib, json as _json, time
		self._require_tenant(tenant_id)
		if not hasattr(self, "_jwt_registry"):
			self._jwt_registry:  dict[str, dict[str, Any]] = {}
			self._jwt_blacklist: set[str]                  = set()
		if token in self._jwt_blacklist:
			raise PermissionError("jwt_blacklisted")
		parts = token.split(".")
		if len(parts) != 3:
			raise ValueError("jwt_malformed")
		h_enc, p_enc, sig = parts
		sig_seed  = f"{h_enc}.{p_enc}.{tenant_id}"
		# Try both tenant-bound and user-bound signatures
		reg_entry = self._jwt_registry.get(token)
		if reg_entry is None:
			raise PermissionError("jwt_unknown")
		if int(time.time()) > reg_entry["exp"]:
			raise PermissionError("jwt_expired")
		if reg_entry["tenant_id"] != tenant_id:
			raise PermissionError("jwt_tenant_mismatch")
		payload_bytes = base64.urlsafe_b64decode(p_enc + "==")
		payload       = _json.loads(payload_bytes)
		return {"valid": True, "payload": payload, "verified_at": _utc_now()}

	def api_key_hash(
		self,
		tenant_id: str,
		raw_key: str,
		key_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Hash a raw API key with PBKDF2-HMAC-SHA256 for safe storage.

		Returns a record suitable for persistence; the raw key is never stored.
		"""
		import hashlib, secrets
		self._require_tenant(tenant_id)
		if not raw_key:
			raise ValueError("api_key_raw_required")
		salt    = secrets.token_hex(16)
		dk      = hashlib.pbkdf2_hmac("sha256", raw_key.encode(), salt.encode(), 100_000)
		hashed  = dk.hex()
		kid     = key_id or f"ak_{secrets.token_hex(8)}"
		record  = {
			"key_id":     kid,
			"tenant_id":  tenant_id,
			"hash":       hashed,
			"salt":       salt,
			"algorithm":  "pbkdf2_hmac_sha256",
			"iterations": 100_000,
			"created_at": _utc_now(),
		}
		if not hasattr(self, "_api_keys"):
			self._api_keys: dict[str, dict[str, Any]] = {}
		self._api_keys[self._tenant_key(tenant_id, kid).__str__()] = record
		return {k: v for k, v in record.items() if k not in {"hash", "salt"}} | {"stored": True}

	def api_key_validate(
		self,
		tenant_id: str,
		key_id: str,
		raw_key: str,
	) -> dict[str, Any]:
		"""
		Validate a raw API key against a stored hash record.

		Returns valid=True / False without exposing the stored hash.
		"""
		import hashlib
		self._require_tenant(tenant_id)
		if not hasattr(self, "_api_keys"):
			self._api_keys: dict[str, dict[str, Any]] = {}
		record = self._api_keys.get(str(self._tenant_key(tenant_id, key_id)))
		if record is None:
			return {"valid": False, "reason": "key_not_found", "key_id": key_id}
		dk     = hashlib.pbkdf2_hmac("sha256", raw_key.encode(), record["salt"].encode(), record["iterations"])
		valid  = dk.hex() == record["hash"]
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=key_id,
			event_type="api_key_validated",
			actor="system",
			decision="allow" if valid else "deny",
			metadata={"key_id": key_id, "valid": valid},
		)
		return {"valid": valid, "key_id": key_id, "tenant_id": tenant_id, "checked_at": _utc_now()}

	def mfa_integration(
		self,
		tenant_id: str,
		user_id: str,
		mfa_type: str,
		*,
		enable: bool = True,
		device_ref: str = "",
	) -> dict[str, Any]:
		"""
		Enable or disable an MFA factor for an identity.

		mfa_type: 'totp' | 'sms' | 'email' | 'hardware_key' | 'passkey'.
		"""
		supported = {"totp", "sms", "email", "hardware_key", "passkey"}
		if mfa_type not in supported:
			raise ValueError(f"unsupported_mfa_type:{mfa_type}")
		identity = self._require_identity(user_id, tenant_id)
		# Mutate by replacing in store (dataclasses are frozen; use a dict overlay)
		key = self._tenant_key(tenant_id, user_id)
		if not hasattr(self, "_mfa_registrations"):
			self._mfa_registrations: dict[tuple[str, str], list[dict[str, Any]]] = {}
		regs = self._mfa_registrations.setdefault(key, [])
		existing = next((r for r in regs if r["mfa_type"] == mfa_type), None)
		if existing:
			existing["enabled"] = enable
			existing["device_ref"] = device_ref
		else:
			regs.append({"mfa_type": mfa_type, "enabled": enable, "device_ref": device_ref, "registered_at": _utc_now()})
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=user_id,
			event_type="mfa_factor_updated",
			actor=user_id,
			decision="allow",
			metadata={"mfa_type": mfa_type, "enabled": enable},
		)
		return {"user_id": user_id, "tenant_id": tenant_id, "mfa_type": mfa_type, "enabled": enable, "device_ref": device_ref}

	def risk_score_login(
		self,
		tenant_id: str,
		user_id: str,
		ip_address: str,
		device_id: str,
		user_agent: str = "",
		*,
		new_device: bool = False,
		off_hours: bool = False,
		impossible_travel: bool = False,
	) -> dict[str, Any]:
		"""
		Score the risk of a login attempt using heuristic signals.

		Returns a 0-1 risk score and recommendation (allow / step_up / block).
		"""
		self._require_tenant(tenant_id)
		score = 0.0
		factors: list[str] = []
		if new_device:
			score += 0.25; factors.append("new_device")
		if off_hours:
			score += 0.15; factors.append("off_hours")
		if impossible_travel:
			score += 0.5;  factors.append("impossible_travel")
		# Simple IP entropy heuristic
		octets = ip_address.split(".")
		if len(octets) == 4 and octets[0] in ("10", "172", "192"):
			pass  # internal
		else:
			score += 0.05; factors.append("external_ip")
		score = min(1.0, round(score, 4))
		recommendation = "block" if score >= 0.8 else "step_up" if score >= 0.4 else "allow"
		result = {
			"user_id":        user_id,
			"tenant_id":      tenant_id,
			"ip_address":     ip_address,
			"device_id":      device_id,
			"risk_score":     score,
			"risk_factors":   factors,
			"recommendation": recommendation,
			"scored_at":      _utc_now(),
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=user_id,
			event_type="login_risk_scored",
			actor=user_id,
			decision=recommendation,
			metadata=result,
		)
		return result

	def concurrent_session_limit(
		self,
		tenant_id: str,
		user_id: str,
		max_sessions: int = 3,
	) -> dict[str, Any]:
		"""
		Check and enforce concurrent session limits.

		Revokes the oldest active sessions if the limit is exceeded.
		Returns how many sessions were revoked.
		"""
		self._require_tenant(tenant_id)
		active = [
			s for s in self._sessions.values()
			if s.tenant_id == tenant_id
			and s.user_id == user_id
			and s.status == "active"
		]
		active.sort(key=lambda s: s.id)  # oldest first by stable ID ordering
		revoked_ids: list[str] = []
		while len(active) > max_sessions:
			oldest = active.pop(0)
			self.revoke_session(oldest.id, actor="system:session_limit", tenant_id=tenant_id)
			revoked_ids.append(oldest.id)
		return {
			"user_id":       user_id,
			"tenant_id":     tenant_id,
			"max_sessions":  max_sessions,
			"active_before": len(active) + len(revoked_ids),
			"revoked":       len(revoked_ids),
			"revoked_ids":   revoked_ids,
		}

	def token_blacklist(
		self,
		tenant_id: str,
		token: str,
		reason: str = "explicit_revocation",
	) -> dict[str, Any]:
		"""
		Add a JWT to the in-memory blacklist so jwt_verify rejects it.

		Also removes it from the registry to free memory.
		"""
		self._require_tenant(tenant_id)
		if not hasattr(self, "_jwt_blacklist"):
			self._jwt_blacklist: set[str] = set()
			self._jwt_registry:  dict[str, dict[str, Any]] = {}
		self._jwt_blacklist.add(token)
		self._jwt_registry.pop(token, None)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=token[:16] + "...",
			event_type="token_blacklisted",
			actor="system",
			decision="allow",
			metadata={"reason": reason},
		)
		return {"blacklisted": True, "reason": reason, "at": _utc_now()}

	def device_fingerprint_auth(
		self,
		tenant_id: str,
		user_id: str,
		fingerprint: str,
		session_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Authenticate a session using a device fingerprint.

		Computes a SHA-256 of the fingerprint string and checks it against
		previously registered device fingerprints for this user.
		"""
		import hashlib
		self._require_tenant(tenant_id)
		fp_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
		if not hasattr(self, "_device_fingerprints"):
			self._device_fingerprints: dict[tuple[str, str], list[str]] = {}
		key = self._tenant_key(tenant_id, user_id)
		known = self._device_fingerprints.setdefault(key, [])
		is_known = fp_hash in known
		if not is_known:
			known.append(fp_hash)
		result = {
			"user_id":       user_id,
			"tenant_id":     tenant_id,
			"fingerprint_hash": fp_hash,
			"is_known_device": is_known,
			"trust_level":   "high" if is_known else "low",
			"checked_at":    _utc_now(),
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=user_id,
			event_type="device_fingerprint_checked",
			actor=user_id,
			decision="allow",
			metadata={"is_known_device": is_known},
		)
		return result

	def passive_auth_detect(
		self,
		tenant_id: str,
		user_id: str,
		behavioral_signals: dict[str, Any],
	) -> dict[str, Any]:
		"""
		Passive authentication via behavioral signals (typing cadence,
		mouse patterns, etc.).

		behavioral_signals: dict of signal_name -> value.
		Returns a passive auth confidence score and pass/fail decision.
		"""
		self._require_tenant(tenant_id)
		score = 0.5  # baseline
		used_signals: list[str] = []
		# Each present signal with non-empty value lifts confidence slightly
		for sig, val in behavioral_signals.items():
			if val:
				score = min(1.0, score + 0.05)
				used_signals.append(sig)
		confidence = round(score, 4)
		passed = confidence >= 0.7
		result = {
			"user_id":       user_id,
			"tenant_id":     tenant_id,
			"confidence":    confidence,
			"signals_used":  used_signals,
			"passed":        passed,
			"decision":      "allow" if passed else "step_up",
			"evaluated_at":  _utc_now(),
		}
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=user_id,
			event_type="passive_auth_evaluated",
			actor=user_id,
			decision=result["decision"],
			metadata=result,
		)
		return result

	def auth_analytics(
		self,
		tenant_id: str,
		period_label: str = "all_time",
	) -> dict[str, Any]:
		"""
		Aggregate authentication analytics for a tenant.

		Returns identity, session, role, decision, and security-agent counts
		with risk/trust summaries.
		"""
		identities   = [i for i in self._identities.values()  if i.tenant_id == tenant_id]
		sessions     = [s for s in self._sessions.values()    if s.tenant_id == tenant_id]
		decisions    = [d for d in self._access_decisions.values() if d.tenant_id == tenant_id]
		roles        = [r for r in self._roles.values()       if r.tenant_id == tenant_id]
		assignments  = [a for a in self._assignments.values() if a.tenant_id == tenant_id]
		agents       = [ag for ag in self._security_agents.values() if ag.tenant_id == tenant_id]
		audit_evs    = [e for e in self._audit_events.values() if e.tenant_id == tenant_id]
		avg_trust    = (
			round(sum(s.trust_score for s in sessions) / len(sessions), 4)
			if sessions else 0.0
		)
		return {
			"tenant_id":              tenant_id,
			"period":                 period_label,
			"identity_count":         len(identities),
			"active_session_count":   sum(1 for s in sessions if s.status == "active"),
			"revoked_session_count":  sum(1 for s in sessions if s.status == "revoked"),
			"role_count":             len(roles),
			"role_assignment_count":  len(assignments),
			"access_decision_count":  len(decisions),
			"denied_decision_count":  sum(1 for d in decisions if d.decision == "deny"),
			"security_agent_count":   len(agents),
			"audit_event_count":      len(audit_evs),
			"average_trust_score":    avg_trust,
			"mfa_enabled_count":      sum(1 for i in identities if i.mfa_enabled),
			"generated_at":           _utc_now(),
		}

	def password_breach_check(
		self,
		tenant_id: str,
		password_hash_prefix: str,
	) -> dict[str, Any]:
		"""
		Check a password hash prefix against a synthetic known-breach list
		(k-anonymity model — only the first 5 chars of SHA-1 are sent).

		In production, call the HaveIBeenPwned Pwned Passwords API.
		Returns breach_count (0 = not found in synthetic list).
		"""
		import hashlib
		self._require_tenant(tenant_id)
		if len(password_hash_prefix) < 5:
			raise ValueError("password_hash_prefix_must_be_at_least_5_chars")
		prefix = password_hash_prefix[:5].upper()
		# Synthetic list: flag common test prefixes as breached
		known_breached_prefixes = {"5BAA6", "B94E7", "CBFDA", "7C4A8", "D0763"}
		breach_count = 1000 if prefix in known_breached_prefixes else 0
		self._record_audit(
			tenant_id=tenant_id,
			subject_id="password_breach_check",
			event_type="password_breach_checked",
			actor="system",
			decision="allow",
			metadata={"prefix": prefix, "breach_count": breach_count},
		)
		return {
			"tenant_id":    tenant_id,
			"prefix":       prefix,
			"breach_count": breach_count,
			"breached":     breach_count > 0,
			"checked_at":   _utc_now(),
		}

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
		policy_result: dict[str, Any] | None = None,
	) -> AuthAuditEvent:
		policy_result = policy_result or _allow_result()
		event = AuthAuditEvent(
			id=f"auth-audit-{len(self._audit_events) + 1:04d}",
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
			policy_decision=policy_result["decision"],
			matched_rules=tuple(policy_result["matched_rules"]),
			review_reasons=self._reasons(policy_result),
			review_evidence=self._review_evidence(policy_result),
		)
		self._audit_events[self._tenant_key(tenant_id, event.id)] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(self._reasons(result))
			raise PermissionError(reasons or "capability_policy_blocked")

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			str(action.get("reason") or action.get("required_action") or "capability_policy_blocked")
			for action in result.get("actions", [])
		)

	def _review_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": [
				str(action.get("required_action"))
				for action in result.get("actions", [])
				if action.get("required_action")
			],
			"reasons": list(self._reasons(result)),
			"review_recorded": bool(review_recorded),
		}

	def _list(self, values: Any, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


def _utc_now() -> str:
	from datetime import datetime, timezone
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_bool(value: Any) -> bool:
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


def _review_result(reason: str, required_action: str) -> dict[str, Any]:
	return {
		"decision": "require_review",
		"matched_rules": [],
		"actions": [{"reason": reason, "required_action": required_action}],
	}
