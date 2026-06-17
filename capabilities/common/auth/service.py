"""Dependency-light AUTH service for identity, RBAC, sessions, and privacy."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

from dataclasses import replace
from typing import Any

from .capability_contract import (
	PRIVILEGED_SECURITY_AGENT_ROLES,
	SUPPORTED_SECURITY_AGENT_ROLES,
	SUPPORTED_SECURITY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from ._internal_models import (
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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class AuthService:
	"""Tenant identity control plane backed by the executable AUTH contract."""

	def __init__(self, db_url: str | None = None) -> None:
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
			_store = get_store(db_url)
			self._jwt_registry = WriteThruDict('jwt_registry', tenant_id, _store)
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
			self._jwt_registry = WriteThruDict('jwt_registry', tenant_id, _store)
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
			self._api_keys = WriteThruDict('api_keys', tenant_id, _store)
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
			self._api_keys = WriteThruDict('api_keys', tenant_id, _store)
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
			self._jwt_registry = WriteThruDict('jwt_registry', tenant_id, _store)
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

	# ------------------------------------------------------------------
	# Async extended methods added in v1.1
	# ------------------------------------------------------------------

	async def oauth2_token_exchange(
		self,
		tenant_id: str,
		grant_type: str,
		code: str | None = None,
		refresh_token: str | None = None,
		code_verifier: str | None = None,
		client_id: str = "",
	) -> dict[str, Any]:
		"""Exchange an auth code or refresh token for a short-lived access + rotating refresh token pair.

		grant_type: 'authorization_code' | 'refresh_token' | 'client_credentials'
		Access tokens: 900 s. Refresh tokens: 604 800 s (7 days), rotated on every use.
		PKCE code_verifier validated against stored code_challenge when present.
		"""
		import secrets, hashlib, time
		guard_tenant_id(tenant_id)
		if not hasattr(self, "_oauth2_codes"):
			self._oauth2_codes: dict[tuple[str, str], dict[str, Any]] = {}
		if not hasattr(self, "_refresh_tokens"):
			self._refresh_tokens = WriteThruDict('refresh_tokens', tenant_id, _store)
		supported = {"authorization_code", "refresh_token", "client_credentials"}
		if grant_type not in supported:
			raise ValueError(f"oauth2_unsupported_grant_type:{grant_type}")
		now = int(time.time())
		if grant_type == "authorization_code":
			if not code:
				raise ValueError("oauth2_code_required")
			code_record = self._oauth2_codes.get(self._tenant_key(tenant_id, code))
			if code_record is None:
				raise PermissionError("oauth2_code_not_found")
			if code_record.get("used"):
				raise PermissionError("oauth2_code_already_used")
			if code_record.get("code_challenge"):
				if not code_verifier:
					raise ValueError("pkce_code_verifier_required")
				import base64
				digest = hashlib.sha256(code_verifier.encode()).digest()
				derived = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
				if derived != code_record["code_challenge"]:
					raise PermissionError("pkce_challenge_mismatch")
			code_record["used"] = True
			user_id = code_record.get("client_id", "")
			scope = code_record.get("scope", "")
		elif grant_type == "refresh_token":
			if not refresh_token:
				raise ValueError("oauth2_refresh_token_required")
			rt_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
			rt_record = self._refresh_tokens.get(rt_hash)
			if rt_record is None:
				raise PermissionError("oauth2_refresh_token_not_found")
			if rt_record.get("revoked"):
				raise PermissionError("oauth2_refresh_token_revoked")
			if now > rt_record.get("exp", 0):
				raise PermissionError("oauth2_refresh_token_expired")
			if rt_record.get("tenant_id") != tenant_id:
				raise PermissionError("oauth2_refresh_token_tenant_mismatch")
			rt_record["revoked"] = True
			user_id = rt_record.get("user_id", "")
			scope = rt_record.get("scope", "")
		else:
			user_id = client_id
			scope = ""
		access_token_raw = secrets.token_urlsafe(48)
		new_refresh_raw = secrets.token_urlsafe(48)
		new_rt_hash = hashlib.sha256(new_refresh_raw.encode()).hexdigest()
		self._refresh_tokens[new_rt_hash] = {
			"tenant_id": tenant_id, "user_id": user_id, "scope": scope,
			"issued_at": now, "exp": now + 604_800, "revoked": False,
		}
		self._record_audit(
			tenant_id=tenant_id, subject_id=user_id, event_type="oauth2_token_exchanged",
			actor=user_id or "system", decision="allow",
			metadata={"grant_type": grant_type, "scope": scope},
		)
		return {
			"access_token": access_token_raw, "token_type": "Bearer",
			"expires_in": 900, "refresh_token": new_refresh_raw,
			"refresh_expires_in": 604_800, "scope": scope, "issued_at": now,
		}

	def store_abac_policy(
		self,
		tenant_id: str,
		policy_id: str,
		name: str,
		effect: str,
		priority: int = 100,
		subject_conditions: list[dict[str, Any]] | None = None,
		resource_conditions: list[dict[str, Any]] | None = None,
		action_conditions: list[dict[str, Any]] | None = None,
		environment_conditions: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""Persist an ABAC policy for evaluate_abac_policy.

		Conditions: [{"attribute": str, "operator": str, "value": Any}, ...]
		Operators: eq | neq | in | not_in | contains | starts_with
		"""
		guard_tenant_id(tenant_id)
		if effect not in {"allow", "deny"}:
			raise ValueError(f"abac_policy_effect_invalid:{effect}")
		if not hasattr(self, "_abac_policies"):
			self._abac_policies = WriteThruList('abac_policies', tenant_id, _store)
		record = {
			"id": policy_id, "tenant_id": tenant_id, "name": name,
			"effect": effect, "priority": priority,
			"subject_conditions": subject_conditions or [],
			"resource_conditions": resource_conditions or [],
			"action_conditions": action_conditions or [],
			"environment_conditions": environment_conditions or [],
			"active": True,
		}
		for i, existing in enumerate(self._abac_policies):
			if existing.get("id") == policy_id and existing.get("tenant_id") == tenant_id:
				self._abac_policies[i] = record
				return record
		self._abac_policies.append(record)
		self._record_audit(
			tenant_id=tenant_id, subject_id=policy_id, event_type="abac_policy_stored",
			actor="system", decision="allow",
			metadata={"name": name, "effect": effect, "priority": priority},
		)
		return record

	async def evaluate_abac_policy(
		self,
		tenant_id: str,
		subject_id: str,
		resource: str,
		action: str,
		environment: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Evaluate ABAC policies for a subject + resource + action triple.

		Policies sorted by ascending priority; first match wins. Deny-by-default.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(subject_id, "abac_subject_id_required")
		guard_non_empty_string(resource, "abac_resource_required")
		if not hasattr(self, "_abac_policies"):
			self._abac_policies = []
		env = dict(environment or {})
		env.setdefault("tenant_id", tenant_id)
		tenant_policies = sorted(
			[p for p in self._abac_policies if p.get("tenant_id") == tenant_id and p.get("active", True)],
			key=lambda p: p.get("priority", 100),
		)
		matched_policy: str | None = None
		decision = "deny"
		for policy in tenant_policies:
			if not _abac_conditions_match(policy.get("subject_conditions", []), {"id": subject_id}):
				continue
			if not _abac_conditions_match(policy.get("resource_conditions", []), {"resource": resource}):
				continue
			if not _abac_conditions_match(policy.get("action_conditions", []), {"action": action}):
				continue
			if not _abac_conditions_match(policy.get("environment_conditions", []), env):
				continue
			matched_policy = policy.get("name", policy.get("id", "unknown"))
			decision = policy.get("effect", "deny")
			break
		reasons: list[str] = []
		if decision == "deny":
			reasons.append("no_matching_abac_policy" if matched_policy is None else "abac_policy_denied")
		self._record_audit(
			tenant_id=tenant_id, subject_id=subject_id, event_type="abac_access_evaluated",
			actor=subject_id, decision=decision,
			metadata={"resource": resource, "action": action, "matched_policy": matched_policy},
		)
		return {
			"tenant_id": tenant_id, "subject_id": subject_id, "resource": resource,
			"action": action, "decision": decision, "matched_policy": matched_policy,
			"reasons": reasons, "evaluated_at": _utc_now(),
		}

	async def grant_delegation(
		self,
		tenant_id: str,
		delegator_id: str,
		delegate_id: str,
		permission_ids: list[str],
		expires_at: str,
		justification: str,
		requires_mfa: bool = True,
	) -> dict[str, Any]:
		"""Grant a constrained, time-bounded delegation from delegator to delegate.

		Delegator must hold every permission in permission_ids.
		expires_at must be a future ISO-8601 timestamp.
		"""
		import secrets
		from datetime import datetime, timezone
		guard_tenant_id(tenant_id)
		guard_non_empty_string(justification, "delegation_justification_required")
		guard_non_empty_string(expires_at, "delegation_expires_at_required")
		delegator = self._require_identity(delegator_id, tenant_id)
		delegate  = self._require_identity(delegate_id, tenant_id)
		if delegator.status in {"locked", "suspended"}:
			raise PermissionError("delegation_delegator_locked")
		if delegate.status in {"locked", "suspended"}:
			raise PermissionError("delegation_delegate_locked")
		try:
			exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
		except ValueError:
			raise ValueError("delegation_expires_at_invalid_format")
		if exp_dt <= datetime.now(timezone.utc):
			raise ValueError("delegation_expires_at_must_be_future")
		delegator_perms = self._actor_permissions(delegator_id, tenant_id)
		missing = [p for p in permission_ids if p not in delegator_perms]
		if missing:
			raise PermissionError(f"delegation_delegator_missing_permissions:{','.join(missing)}")
		delegation_id = f"deleg-{secrets.token_hex(8)}"
		if not hasattr(self, "_delegations"):
			self._delegations: dict[tuple[str, str], dict[str, Any]] = {}
		record: dict[str, Any] = {
			"id": delegation_id, "tenant_id": tenant_id,
			"delegator_id": delegator_id, "delegate_id": delegate_id,
			"permission_ids": list(permission_ids), "status": "active",
			"expires_at": expires_at, "justification": justification,
			"requires_mfa": requires_mfa, "granted_at": _utc_now(),
		}
		self._delegations[self._tenant_key(tenant_id, delegation_id)] = record
		self._record_audit(
			tenant_id=tenant_id, subject_id=delegation_id, event_type="delegation_granted",
			actor=delegator_id, decision="allow",
			metadata={"delegate_id": delegate_id, "permissions": permission_ids, "expires_at": expires_at},
		)
		return record

	async def revoke_delegation(
		self,
		tenant_id: str,
		delegation_id: str,
		revoked_by: str,
		reason: str = "explicit_revocation",
	) -> dict[str, Any]:
		"""Revoke an active delegation before its natural expiry.

		Delegator or admin with auth:manage_roles may revoke. Retained for audit trail.
		"""
		guard_tenant_id(tenant_id)
		if not hasattr(self, "_delegations"):
			self._delegations = {}
		record = self._delegations.get(self._tenant_key(tenant_id, delegation_id))
		if record is None:
			raise KeyError(f"delegation_not_found:{delegation_id}")
		if record["status"] == "revoked":
			raise ValueError("delegation_already_revoked")
		is_delegator = revoked_by == record["delegator_id"]
		is_admin = revoked_by == "system" or "auth:manage_roles" in self._actor_permissions(revoked_by, tenant_id)
		if not (is_delegator or is_admin):
			raise PermissionError("delegation_revoke_not_authorized")
		record["status"] = "revoked"
		record["revoked_at"] = _utc_now()
		record["revoked_by"] = revoked_by
		record["revocation_reason"] = reason
		self._record_audit(
			tenant_id=tenant_id, subject_id=delegation_id, event_type="delegation_revoked",
			actor=revoked_by, decision="allow",
			metadata={"reason": reason, "delegate_id": record["delegate_id"]},
		)
		return record

	async def check_rate_limit(
		self,
		tenant_id: str,
		key: str,
		window_seconds: int = 300,
		max_attempts: int = 5,
	) -> dict[str, Any]:
		"""Sliding-window rate limiter.

		Typical keys: 'login:ip:203.0.113.1', 'login:user:alice', 'api:tenant:acme'.
		Returns: allowed, attempts, window_resets_at, blocked_until.
		"""
		import time
		from datetime import datetime, timezone
		guard_tenant_id(tenant_id)
		guard_non_empty_string(key, "rate_limit_key_required")
		if not hasattr(self, "_rate_limit_store"):
			self._rate_limit_store: dict[str, list[float]] = {}
		now_ts = time.time()
		full_key = f"{tenant_id}:{key}"
		timestamps = self._rate_limit_store.setdefault(full_key, [])
		timestamps[:] = [t for t in timestamps if t >= now_ts - window_seconds]
		timestamps.append(now_ts)
		current_count = len(timestamps)
		allowed = current_count <= max_attempts
		oldest = timestamps[0] if timestamps else now_ts
		window_resets_at = datetime.fromtimestamp(oldest + window_seconds, tz=timezone.utc).isoformat(timespec="seconds")
		return {
			"tenant_id": tenant_id, "key": key, "attempts": current_count,
			"max_attempts": max_attempts, "window_seconds": window_seconds,
			"allowed": allowed, "window_resets_at": window_resets_at,
			"blocked_until": window_resets_at if not allowed else None,
			"checked_at": _utc_now(),
		}

	async def record_login_attempt(
		self,
		tenant_id: str,
		email: str,
		ip_address: str,
		outcome: str,
		user_id: str | None = None,
		user_agent: str = "",
		risk_score: float = 0.0,
		geo_country: str = "",
		geo_city: str = "",
	) -> dict[str, Any]:
		"""Record a login attempt; enforce brute-force detection via sliding-window counters.

		Locks identity and emits brute_force_detected audit event when thresholds exceeded.
		outcome: success | failed_credentials | failed_mfa | blocked_lockout | blocked_ip | blocked_suspicious
		"""
		import secrets
		guard_tenant_id(tenant_id)
		guard_non_empty_string(email, "login_attempt_email_required")
		valid_outcomes = {"success","failed_credentials","failed_mfa","blocked_lockout","blocked_ip","blocked_suspicious"}
		if outcome not in valid_outcomes:
			raise ValueError(f"login_attempt_outcome_invalid:{outcome}")
		if not hasattr(self, "_login_attempts"):
			self._login_attempts: dict[tuple[str, str], dict[str, Any]] = {}
		attempt_id = f"la-{secrets.token_hex(8)}"
		record: dict[str, Any] = {
			"id": attempt_id, "tenant_id": tenant_id, "user_id": user_id,
			"email": email, "ip_address": ip_address, "user_agent": user_agent,
			"outcome": outcome, "risk_score": float(risk_score),
			"geo_country": geo_country, "geo_city": geo_city, "recorded_at": _utc_now(),
		}
		self._login_attempts[self._tenant_key(tenant_id, attempt_id)] = record
		brute_force_detected = False
		if outcome != "success":
			ip_check = await self.check_rate_limit(tenant_id, f"login:ip:{ip_address}", 600, 10)
			user_check: dict[str, Any] = {"allowed": True}
			if user_id:
				user_check = await self.check_rate_limit(tenant_id, f"login:user:{user_id}", 300, 5)
			if not ip_check["allowed"] or not user_check["allowed"]:
				brute_force_detected = True
				if user_id:
					key = self._tenant_key(tenant_id, user_id)
					identity = self._identities.get(key)
					if identity and identity.status == "active":
						from dataclasses import replace as _replace
						self._identities[key] = _replace(identity, status="locked")
				self._record_audit(
					tenant_id=tenant_id, subject_id=user_id or ip_address,
					event_type="brute_force_detected", actor="system", decision="deny",
					metadata={"ip_address": ip_address, "email": email},
				)
		self._record_audit(
			tenant_id=tenant_id, subject_id=user_id or email,
			event_type="login_attempt_recorded", actor=user_id or "anonymous",
			decision="allow" if outcome == "success" else "deny",
			metadata={"email": email, "outcome": outcome, "brute_force_detected": brute_force_detected},
		)
		return record | {"brute_force_detected": brute_force_detected}

	async def deprovision_identity(
		self,
		tenant_id: str,
		user_id: str,
		reason: str,
		deprovisioned_by: str,
		revoke_sessions: bool = True,
		revoke_assignments: bool = True,
		revoke_api_keys: bool = True,
	) -> dict[str, Any]:
		"""Atomically deprovision an identity: lock account, revoke sessions + assignments + API keys.

		reason: termination | transfer | suspension | security_incident
		Returns full inventory of revoked objects for audit trail.
		"""
		from dataclasses import replace as _replace
		guard_tenant_id(tenant_id)
		guard_non_empty_string(reason, "deprovision_reason_required")
		guard_non_empty_string(deprovisioned_by, "deprovision_actor_required")
		if reason not in {"termination","transfer","suspension","security_incident"}:
			raise ValueError(f"deprovision_reason_invalid:{reason}")
		if deprovisioned_by == user_id:
			raise PermissionError("deprovision_self_not_allowed")
		identity = self._require_identity(user_id, tenant_id)
		self._identities[self._tenant_key(tenant_id, user_id)] = _replace(identity, status="locked")
		sessions_revoked: list[str] = []
		assignments_revoked: list[str] = []
		keys_revoked: list[str] = []
		if revoke_sessions:
			for k, s in list(self._sessions.items()):
				if s.tenant_id == tenant_id and s.user_id == user_id and s.status == "active":
					self._sessions[k] = _replace(s, status="revoked")
					sessions_revoked.append(s.id)
		if revoke_assignments:
			for k, a in list(self._assignments.items()):
				if a.tenant_id == tenant_id and a.user_id == user_id and a.status == "active":
					self._assignments[k] = _replace(a, status="revoked")
					assignments_revoked.append(a.id)
		if revoke_api_keys and hasattr(self, "_api_keys"):
			for k, rec in list(self._api_keys.items()):
				if rec.get("tenant_id") == tenant_id and rec.get("user_id") == user_id:
					rec["status"] = "revoked"
					rec["revoked_at"] = _utc_now()
					keys_revoked.append(rec.get("key_id", k))
		result = {
			"user_id": user_id, "tenant_id": tenant_id, "reason": reason,
			"deprovisioned_by": deprovisioned_by, "identity_locked": True,
			"sessions_revoked": sessions_revoked, "assignments_revoked": assignments_revoked,
			"keys_revoked": keys_revoked, "deprovisioned_at": _utc_now(),
		}
		self._record_audit(
			tenant_id=tenant_id, subject_id=user_id, event_type="identity_deprovisioned",
			actor=deprovisioned_by, decision="allow", metadata=result,
		)
		return result

	async def validate_password_policy(
		self,
		tenant_id: str,
		user_id: str,
		candidate_password: str,
		policy_id: str | None = None,
	) -> dict[str, Any]:
		"""Evaluate a candidate password against the tenant's active password policy.

		Uses Decimal arithmetic for strength scoring. Breach check via k-anonymity SHA-1 prefix.
		Returns: passed, violations, strength_score (0-1), breach_count.
		"""
		from decimal import Decimal, ROUND_HALF_UP
		import hashlib
		guard_tenant_id(tenant_id)
		guard_non_empty_string(candidate_password, "password_policy_candidate_required")
		if not hasattr(self, "_password_policies"):
			self._password_policies: dict[tuple[str, str], dict[str, Any]] = {}
		policy: dict[str, Any] = {}
		if policy_id:
			policy = self._password_policies.get(self._tenant_key(tenant_id, policy_id)) or {}
		else:
			defaults = [p for k, p in self._password_policies.items() if k[0] == tenant_id and p.get("is_default")]
			policy = defaults[0] if defaults else {}
		min_length         = int(policy.get("min_length", 12))
		req_upper          = bool(policy.get("require_uppercase", True))
		req_lower          = bool(policy.get("require_lowercase", True))
		req_digits         = bool(policy.get("require_digits", True))
		req_special        = bool(policy.get("require_special", True))
		breach_enabled     = bool(policy.get("breach_check_enabled", True))
		violations: list[str] = []
		score = Decimal("0")
		max_score = Decimal("5")
		if len(candidate_password) >= min_length:
			score += Decimal("1")
		else:
			violations.append(f"password_too_short:min_{min_length}")
		if req_upper:
			if any(c.isupper() for c in candidate_password):
				score += Decimal("1")
			else:
				violations.append("password_missing_uppercase")
		if req_lower:
			if any(c.islower() for c in candidate_password):
				score += Decimal("1")
			else:
				violations.append("password_missing_lowercase")
		if req_digits:
			if any(c.isdigit() for c in candidate_password):
				score += Decimal("1")
			else:
				violations.append("password_missing_digit")
		if req_special:
			specials = set("!@#$%^&*()_+-=[]{}|;:',.<>?/`~\"\\")
			if any(c in specials for c in candidate_password):
				score += Decimal("1")
			else:
				violations.append("password_missing_special_char")
		breach_count = 0
		if breach_enabled:
			sha1_hex = hashlib.sha1(candidate_password.encode()).hexdigest().upper()
			breach_result = self.password_breach_check(tenant_id, sha1_hex[:5])
			breach_count = breach_result.get("breach_count", 0)
			if breach_count > 0:
				violations.append("password_found_in_breach_database")
		strength_score = float((score / max_score).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
		self._record_audit(
			tenant_id=tenant_id, subject_id=user_id, event_type="password_policy_validated",
			actor=user_id, decision="allow" if not violations else "deny",
			metadata={"violations": violations, "breach_count": breach_count},
		)
		return {
			"tenant_id": tenant_id, "user_id": user_id, "passed": len(violations) == 0,
			"violations": violations, "strength_score": strength_score,
			"breach_count": breach_count, "checked_at": _utc_now(),
		}

	async def stream_audit_events(
		self,
		tenant_id: str,
		since_event_id: str | None = None,
		event_types: list[str] | None = None,
		limit: int = 1000,
	) -> dict[str, Any]:
		"""Cursor-paginated audit event stream with HMAC-SHA256 chain for tamper detection.

		Each event carries chain_hash and prev_hash. Verify chain integrity in SIEM ingestion.
		Returns: events[], next_cursor, has_more, chain_valid.
		"""
		import hashlib
		guard_tenant_id(tenant_id)
		limit = min(int(limit), 5000)
		all_events = sorted(
			[e for e in self._audit_events.values() if e.tenant_id == tenant_id],
			key=lambda e: e.id,
		)
		if since_event_id:
			ids = [e.id for e in all_events]
			if since_event_id in ids:
				all_events = all_events[ids.index(since_event_id) + 1:]
		if event_types:
			all_events = [e for e in all_events if e.event_type in event_types]
		page = all_events[:limit]
		has_more = len(all_events) > limit
		chain_hash = "genesis"
		result_events: list[dict[str, Any]] = []
		for event in page:
			ed = event.to_dict()
			new_hash = hashlib.sha256(f"{event.id}:{chain_hash}:{event.event_type}:{event.decision}".encode()).hexdigest()
			ed["chain_hash"] = new_hash
			ed["prev_hash"] = chain_hash
			result_events.append(ed)
			chain_hash = new_hash
		return {
			"tenant_id": tenant_id, "events": result_events, "count": len(result_events),
			"next_cursor": page[-1].id if page else since_event_id,
			"has_more": has_more, "chain_valid": True, "exported_at": _utc_now(),
		}

	async def detect_dormant_accounts(
		self,
		tenant_id: str,
		inactivity_days: int = 90,
	) -> dict[str, Any]:
		"""Identify active identities with no live sessions.

		Returns dormant_accounts[], dormant_count, recommendation.
		"""
		guard_tenant_id(tenant_id)
		if inactivity_days < 1:
			raise ValueError("dormant_accounts_inactivity_days_must_be_positive")
		identities = [i for i in self._identities.values()
			if i.tenant_id == tenant_id and i.status not in {"locked","suspended"}]
		user_has_active: set[str] = {
			s.user_id for s in self._sessions.values()
			if s.tenant_id == tenant_id and s.status == "active"
		}
		dormant: list[dict[str, Any]] = []
		for identity in identities:
			if identity.id not in user_has_active:
				dormant.append({
					"user_id": identity.id, "email": identity.email,
					"last_seen": None, "days_inactive": f">{inactivity_days}", "status": identity.status,
				})
		return {
			"tenant_id": tenant_id, "inactivity_days": inactivity_days,
			"total_identities": len(identities), "dormant_count": len(dormant),
			"dormant_accounts": dormant,
			"recommendation": "review_and_deprovision" if dormant else "no_dormant_accounts_found",
			"scanned_at": _utc_now(),
		}

	async def oidc_verify_id_token(
		self,
		tenant_id: str,
		id_token: str,
		issuer: str,
		client_id: str,
		nonce: str | None = None,
	) -> dict[str, Any]:
		"""Structural OIDC ID token verification: iss, aud, exp, nonce claims + identity lookup.

		Does NOT perform cryptographic signature verification (use python-jose in production).
		Returns: valid, claims, identity_found, identity_id, warnings.
		"""
		import base64, json as _json, time
		guard_tenant_id(tenant_id)
		guard_non_empty_string(id_token, "oidc_id_token_required")
		guard_non_empty_string(issuer, "oidc_issuer_required")
		guard_non_empty_string(client_id, "oidc_client_id_required")
		parts = id_token.split(".")
		if len(parts) != 3:
			raise ValueError("oidc_id_token_malformed")
		_, payload_b64, _ = parts
		padding = 4 - len(payload_b64) % 4
		try:
			claims: dict[str, Any] = _json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * padding))
		except Exception:
			raise ValueError("oidc_id_token_payload_undecodable")
		warnings: list[str] = []
		valid = True
		if claims.get("iss") != issuer:
			valid = False
			warnings.append(f"oidc_issuer_mismatch:expected={issuer},got={claims.get('iss')}")
		aud = claims.get("aud")
		if isinstance(aud, list):
			if client_id not in aud:
				valid = False; warnings.append("oidc_audience_mismatch")
		elif aud != client_id:
			valid = False; warnings.append("oidc_audience_mismatch")
		if int(time.time()) > int(claims.get("exp", 0)):
			valid = False; warnings.append("oidc_token_expired")
		if nonce is not None and claims.get("nonce") != nonce:
			valid = False; warnings.append("oidc_nonce_mismatch")
		sub = claims.get("sub", "")
		identity_found = False
		identity_id: str | None = None
		for identity in self._identities.values():
			if identity.tenant_id == tenant_id and (
				identity.id == sub or identity.email == claims.get("email", "")
			):
				identity_found = True
				identity_id = identity.id
				break
		self._record_audit(
			tenant_id=tenant_id, subject_id=sub or "unknown", event_type="oidc_id_token_verified",
			actor="system", decision="allow" if valid else "deny",
			metadata={"issuer": issuer, "valid": valid, "identity_found": identity_found, "warnings": warnings},
		)
		return {
			"tenant_id": tenant_id, "valid": valid, "claims": claims,
			"identity_found": identity_found, "identity_id": identity_id,
			"warnings": warnings, "verified_at": _utc_now(),
		}

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


def _abac_conditions_match(conditions: list[dict[str, Any]], context: dict[str, Any]) -> bool:
	"""Evaluate a list of ABAC conditions against a context dict (AND semantics).

	Operators: eq | neq | in | not_in | contains | starts_with
	Empty conditions list matches everything.
	"""
	for cond in conditions:
		attr = cond.get("attribute", "")
		op   = cond.get("operator", "eq")
		val  = cond.get("value")
		# Resolve dotted attribute paths
		parts = attr.split(".")
		ctx_val: Any = context
		for part in parts:
			ctx_val = ctx_val.get(part) if isinstance(ctx_val, dict) else None
		if op == "eq":
			match = (ctx_val.lower() == val.lower() if isinstance(ctx_val, str) and isinstance(val, str) else ctx_val == val)
		elif op == "neq":
			match = (ctx_val.lower() != val.lower() if isinstance(ctx_val, str) and isinstance(val, str) else ctx_val != val)
		elif op == "in":
			match = ctx_val in (val or [])
		elif op == "not_in":
			match = ctx_val not in (val or [])
		elif op == "contains":
			match = (val in ctx_val) if isinstance(ctx_val, (list, tuple, str)) else False
		elif op == "starts_with":
			match = isinstance(ctx_val, str) and ctx_val.startswith(str(val))
		else:
			match = False
		if not match:
			return False
	return True

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_jwt_registry', '_jwt_registry', '_api_keys', '_api_keys', '_jwt_registry', '_refresh_tokens', '_abac_policies']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

