"""Dependency-light AUTH service for identity, RBAC, sessions, and privacy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	AuthAccessDecision,
	AuthAuditEvent,
	AuthIdentity,
	AuthPrivacyQuery,
	AuthRole,
	AuthRoleAssignment,
	AuthSession,
)


class AuthService:
	"""Tenant identity control plane backed by the executable AUTH contract."""

	def __init__(self) -> None:
		self._identities: dict[str, AuthIdentity] = {}
		self._roles: dict[str, AuthRole] = {}
		self._assignments: dict[str, AuthRoleAssignment] = {}
		self._sessions: dict[str, AuthSession] = {}
		self._access_decisions: dict[str, AuthAccessDecision] = {}
		self._privacy_queries: dict[str, AuthPrivacyQuery] = {}
		self._audit_events: dict[str, AuthAuditEvent] = {}

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
		self._identities[user_id] = identity
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
		self._roles[role_id] = role
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

	def assign_role(
		self,
		assignment_id: str,
		tenant_id: str,
		user_id: str,
		role_id: str,
		assigned_by: str,
		approval_recorded: bool = False,
	) -> dict[str, Any]:
		identity = self._require_identity(user_id, tenant_id)
		role = self._require_role(role_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"requested_operation": "assign_role",
			"role_tier": role.tier,
			"approval_recorded": approval_recorded or role.approval_recorded,
			"user_locked": identity.status in {"locked", "suspended"},
		})
		self._raise_if_denied(result)
		assignment = AuthRoleAssignment(
			id=assignment_id,
			tenant_id=tenant_id,
			user_id=user_id,
			role_id=role_id,
			assigned_by=assigned_by,
			approval_recorded=approval_recorded or role.approval_recorded,
		)
		self._assignments[assignment_id] = assignment
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
		identity = self._identities.get(user_id)
		if identity is None:
			raise KeyError(f"unknown identity: {user_id}")
		context = {
			"user_locked": identity.status in {"locked", "suspended"},
			"requested_permission_tier": "standard",
			"mfa_verified": bool(mfa_verified),
			"risk_level": risk_level,
			"step_up_completed": bool(step_up_completed),
			"auth_source": auth_source,
			"issuer_trusted": bool(issuer_trusted),
			"tenant_mismatch": tenant_id != identity.tenant_id,
			"tenant_membership_confirmed": tenant_id in identity.tenant_memberships,
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
		self._sessions[session_id] = session
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

	def revoke_session(self, session_id: str, actor: str) -> dict[str, Any]:
		session = self._require_session(session_id)
		revoked = replace(session, status="revoked")
		self._sessions[session_id] = revoked
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
		requested_permission_tier: str = "standard",
		mfa_verified: bool | None = None,
		step_up_completed: bool | None = None,
		risk_level: str | None = None,
	) -> dict[str, Any]:
		identity = self._require_identity(user_id, tenant_id, allow_membership=True)
		session = self._sessions.get(session_id) if session_id else None
		if session is not None and session.tenant_id != tenant_id:
			raise KeyError(f"session not in tenant: {session_id}")
		context = {
			"user_locked": identity.status in {"locked", "suspended"},
			"requested_permission_tier": requested_permission_tier,
			"mfa_verified": session.mfa_verified if mfa_verified is None and session else bool(mfa_verified),
			"risk_level": risk_level or (session.risk_level if session else "low"),
			"step_up_completed": session.step_up_completed if step_up_completed is None and session else bool(step_up_completed),
			"tenant_mismatch": tenant_id != identity.tenant_id,
			"tenant_membership_confirmed": tenant_id in identity.tenant_memberships,
		}
		result = self.evaluate(context)
		role_ids = self._active_role_ids(user_id, tenant_id, permission)
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
		self._access_decisions[decision_id] = access_decision
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

	def run_privacy_query(
		self,
		query_id: str,
		tenant_id: str,
		user_id: str,
		query_type: str,
		epsilon_cost: float,
		approval_recorded: bool = False,
	) -> dict[str, Any]:
		identity = self._require_identity(user_id, tenant_id, allow_membership=True)
		budget_available = identity.privacy_budget >= float(epsilon_cost)
		result = self.evaluate({
			"requested_operation": "privacy_analytics_query",
			"privacy_budget_available": budget_available,
		})
		decision = result["decision"]
		reasons = list(self._reasons(result))
		if decision == "require_review" and not approval_recorded:
			status = "review_required"
			remaining_budget = identity.privacy_budget
		else:
			status = "completed"
			remaining_budget = max(identity.privacy_budget - float(epsilon_cost), 0.0)
			self._identities[user_id] = replace(identity, privacy_budget=remaining_budget)
		query = AuthPrivacyQuery(
			id=query_id,
			tenant_id=tenant_id,
			user_id=user_id,
			query_type=query_type,
			epsilon_cost=float(epsilon_cost),
			status=status,
			remaining_budget=remaining_budget,
			approval_recorded=approval_recorded,
			reasons=tuple(reasons),
		)
		self._privacy_queries[query_id] = query
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
				if assignment.tenant_id == tenant_id and self._roles[assignment.role_id].tier == "admin"
			]),
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
			mfa_enabled=bool(metadata.get("mfa_enabled", False)),
			privacy_budget=float(metadata.get("privacy_budget", 1.0)),
			metadata=metadata,
		)

	def _require_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			raise PermissionError("tenant_context_required")

	def _require_identity(self, user_id: str, tenant_id: str, allow_membership: bool = False) -> AuthIdentity:
		self._require_tenant(tenant_id)
		identity = self._identities.get(user_id)
		if identity is None:
			raise KeyError(f"unknown identity: {user_id}")
		if identity.tenant_id != tenant_id and not (allow_membership and tenant_id in identity.tenant_memberships):
			raise KeyError(f"identity not in tenant: {user_id}")
		return identity

	def _require_role(self, role_id: str, tenant_id: str) -> AuthRole:
		self._require_tenant(tenant_id)
		role = self._roles.get(role_id)
		if role is None:
			raise KeyError(f"unknown role: {role_id}")
		if role.tenant_id != tenant_id:
			raise KeyError(f"role not in tenant: {role_id}")
		return role

	def _require_session(self, session_id: str) -> AuthSession:
		session = self._sessions.get(session_id)
		if session is None:
			raise KeyError(f"unknown session: {session_id}")
		return session

	def _active_role_ids(self, user_id: str, tenant_id: str, permission: str) -> list[str]:
		role_ids: list[str] = []
		for assignment in self._assignments.values():
			if assignment.user_id != user_id or assignment.tenant_id != tenant_id or assignment.status != "active":
				continue
			role = self._roles.get(assignment.role_id)
			if role and role.status == "active" and permission in role.permissions:
				role_ids.append(role.id)
		return sorted(role_ids)

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
		self._audit_events[event.id] = event
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
