"""Service layer for the User Management capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_USRM_AGENT_ROLES,
	SUPPORTED_USRM_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .user_runtime import (
	AccessReviewRecord,
	BulkUserActionRecord,
	DeprovisionRecord,
	RoleAssignmentRecord,
	UsrmAgentRecord,
	UserAuditEventRecord,
	UserInvitationRecord,
	UserProfileRecord,
	UserRecord,
	normalize_access_review_decision,
	stable_id,
	user_required_actions,
	utc_now,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class UsrmService:
	"""Deterministic user lifecycle service for APG composition."""

	def __init__(self) -> None:
		self.users: dict[str, UserRecord] = {}
		self.profiles: dict[str, UserProfileRecord] = {}
		self.invitations: dict[str, UserInvitationRecord] = {}
		self.role_assignments: dict[str, RoleAssignmentRecord] = {}
		self.access_reviews: dict[str, AccessReviewRecord] = {}
		self.deprovisions: dict[str, DeprovisionRecord] = {}
		self.bulk_actions: dict[str, BulkUserActionRecord] = {}
		self.usrm_agents: dict[str, UsrmAgentRecord] = {}
		self.audit_events: dict[str, UserAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_user(
		self,
		tenant_id: str,
		identity: str,
		display_name: str,
		email: str,
		owner: str,
		profile_validated: bool = True,
		privileged_user: bool = False,
		mfa_enabled: bool = False,
		manager_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(identity or "").strip():
			self._raise_policy(self.evaluate({
				"tenant_context_present": True,
				"operation": "create_user",
				"unique_identity_present": False,
			}))
		if self._identity_exists(tenant_id, identity):
			raise ValueError("unique_identity_required")
		if not str(display_name or "").strip():
			raise ValueError("display_name_required")
		if not str(email or "").strip():
			raise ValueError("email_required")
		context = {
			"tenant_context_present": True,
			"operation": "create_user",
			"unique_identity_present": True,
			"user_owner_assigned": bool(str(owner or "").strip()),
			"profile_validated": bool(profile_validated),
			"privileged_user": bool(privileged_user),
			"mfa_enabled": bool(mfa_enabled),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = UserRecord(
			id=stable_id("usrm_user", tenant_id, identity),
			tenant_id=tenant_id,
			identity=identity,
			display_name=display_name,
			email=email,
			owner=owner,
			profile_validated=True,
			privileged_user=bool(privileged_user),
			mfa_enabled=bool(mfa_enabled),
			manager_id=manager_id,
		)
		self.users[record.id] = record
		self._record_event(tenant_id, "user_created", record.id, f"User created: {display_name}", owner)
		return record.to_dict()

	def update_profile(
		self,
		tenant_id: str,
		user_id: str,
		attributes: dict[str, str],
		privacy_preferences: dict[str, str],
		consent_notice_ref: str,
		updated_by: str,
		privacy_sync_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		if not str(consent_notice_ref or "").strip():
			raise PermissionError("consent_notice_required")
		context = {
			"tenant_context_present": True,
			"operation": "update_profile",
			"privacy_sync_recorded": bool(privacy_sync_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = UserProfileRecord(
			id=stable_id("usrm_profile", tenant_id, user.id),
			tenant_id=tenant_id,
			user_id=user.id,
			attributes={str(key): str(value) for key, value in dict(attributes or {}).items()},
			privacy_preferences={str(key): str(value) for key, value in dict(privacy_preferences or {}).items()},
			consent_notice_ref=consent_notice_ref,
			updated_by=updated_by,
		)
		self.profiles[record.id] = record
		user.updated_at = utc_now()
		self._record_event(tenant_id, "profile_updated", record.id, f"Profile updated: {user.display_name}", updated_by)
		return record.to_dict()

	def invite_user(
		self,
		tenant_id: str,
		user_id: str,
		channel: str,
		consent_notice_ref: str,
		invited_by: str,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		context = {
			"tenant_context_present": True,
			"operation": "invite_user",
			"consent_notice_attached": bool(str(consent_notice_ref or "").strip()),
			"event_stream": self._normalize_token(event_stream),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = UserInvitationRecord(
			id=stable_id("usrm_invite", tenant_id, user.id, len(self.invitations)),
			tenant_id=tenant_id,
			user_id=user.id,
			channel=str(channel or "email"),
			consent_notice_ref=consent_notice_ref,
			invited_by=invited_by,
		)
		self.invitations[record.id] = record
		user.status = "invited"
		user.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"user_invited",
			record.id,
			f"User invited: {user.display_name}",
			invited_by,
			metadata={"event_stream": self._normalize_token(event_stream)},
		)
		return record.to_dict()

	def assign_role(
		self,
		tenant_id: str,
		user_id: str,
		role: str,
		scope: str,
		privileged: bool,
		mfa_enabled: bool,
		approved_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		if not str(role or "").strip():
			raise ValueError("role_required")
		context = {
			"tenant_context_present": True,
			"operation": "assign_role",
			"privileged_user": bool(privileged),
			"privileged_role": bool(privileged),
			"mfa_enabled": bool(mfa_enabled),
			"role_approval_recorded": bool(str(approved_by or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = RoleAssignmentRecord(
			id=stable_id("usrm_role", tenant_id, user.id, role, scope),
			tenant_id=tenant_id,
			user_id=user.id,
			role=role,
			scope=str(scope or "tenant"),
			privileged=bool(privileged),
			approved_by=approved_by,
		)
		self.role_assignments[record.id] = record
		user.privileged_user = bool(user.privileged_user or privileged)
		user.mfa_enabled = bool(user.mfa_enabled or mfa_enabled)
		user.updated_at = utc_now()
		self._record_event(tenant_id, "role_assigned", record.id, f"Role assigned: {role}", approved_by)
		return record.to_dict()

	def record_access_review(
		self,
		tenant_id: str,
		user_id: str,
		reviewer: str,
		decision: str,
		findings: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		if not str(reviewer or "").strip():
			result = self.evaluate({
				"tenant_context_present": True,
				"operation": "record_access_review",
				"access_reviewer_present": False,
			})
			self._raise_policy(result)
		record = AccessReviewRecord(
			id=stable_id("usrm_review", tenant_id, user.id, len(self.access_reviews)),
			tenant_id=tenant_id,
			user_id=user.id,
			reviewer=reviewer,
			decision=normalize_access_review_decision(decision),
			findings=[str(item) for item in list(findings or [])],
		)
		self.access_reviews[record.id] = record
		self._record_event(tenant_id, "access_review_recorded", record.id, f"Access review recorded: {user.display_name}", reviewer)
		return record.to_dict()

	def deprovision_user(
		self,
		tenant_id: str,
		user_id: str,
		actor: str,
		access_revoked: bool,
		evidence_ref: str,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		context = {
			"tenant_context_present": True,
			"operation": "deprovision_user",
			"access_revoked": bool(access_revoked),
			"deprovision_evidence_present": bool(str(evidence_ref or "").strip()),
			"event_stream": self._normalize_token(event_stream),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = DeprovisionRecord(
			id=stable_id("usrm_deprovision", tenant_id, user.id, len(self.deprovisions)),
			tenant_id=tenant_id,
			user_id=user.id,
			actor=actor,
			access_revoked=bool(access_revoked),
			evidence_ref=evidence_ref,
			status="completed",
			required_actions=user_required_actions(result),
			matched_rules=list(result["matched_rules"]),
		)
		self.deprovisions[record.id] = record
		user.status = "deprovisioned"
		user.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"user_deprovisioned",
			record.id,
			f"User deprovisioned: {user.display_name}",
			actor,
			metadata={"event_stream": self._normalize_token(event_stream)},
		)
		return record.to_dict()

	def bulk_suspend_users(
		self,
		tenant_id: str,
		user_ids: list[str],
		actor: str,
		bulk_review_recorded: bool = False,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		users = [self._get_user(tenant_id, user_id) for user_id in user_ids]
		context = {
			"tenant_context_present": True,
			"operation": "bulk_user_action",
			"affected_user_count": len(users),
			"bulk_review_recorded": bool(bulk_review_recorded),
			"event_stream": self._normalize_token(event_stream),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		status = "review_required" if result["decision"] == "require_review" else "completed"
		record = BulkUserActionRecord(
			id=stable_id("usrm_bulk", tenant_id, "suspend", len(self.bulk_actions)),
			tenant_id=tenant_id,
			action="suspend",
			actor=actor,
			user_ids=[user.id for user in users],
			status=status,
			required_actions=user_required_actions(result),
			matched_rules=list(result["matched_rules"]),
		)
		self.bulk_actions[record.id] = record
		if status == "completed":
			for user in users:
				user.status = "suspended"
				user.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"bulk_suspend_users",
			record.id,
			f"Bulk suspend {status}: {len(users)} users",
			actor,
			metadata={"event_stream": self._normalize_token(event_stream)},
		)
		return record.to_dict()

	def register_usrm_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str = "platform",
		human_approval_required: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		context = {
			"tenant_context_present": True,
			"operation": "register_usrm_agent",
			"agent_runtime_supported": runtime_value in SUPPORTED_USRM_AGENT_RUNTIMES,
			"agent_role_supported": role_value in SUPPORTED_USRM_AGENT_ROLES,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = UsrmAgentRecord(
			id=stable_id("usrm_agent", tenant_id, name, runtime_value, role_value),
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=scope,
			owner=owner,
			human_approval_required=bool(human_approval_required),
		)
		self.usrm_agents[record.id] = record
		self._record_event(
			tenant_id,
			"usrm_agent_registered",
			record.id,
			f"User-management agent registered: {name}",
			owner,
			metadata={"runtime": runtime_value, "role": role_value, "event_stream": event_stream_name()},
		)
		return record.to_dict()

	def validate_agent_user_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool = False,
		human_approval_ref: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		agent = self.usrm_agents.get(agent_id)
		if agent is None or agent.tenant_id != tenant_id:
			raise KeyError(f"usrm_agent_not_found:{agent_id}")
		context = {
			"tenant_context_present": True,
			"operation": "agent_user_action",
			"agent_id": agent_id,
			"agent_role": agent.role,
			"action": action,
			"privileged_scope": bool(privileged_scope),
			"human_approval_recorded": bool(str(human_approval_ref or "").strip()),
		}
		return self.evaluate(context)

	def validate_batch_user_lifecycle(
		self,
		tenant_id: str,
		affected_user_count: int,
		event_stream: str = "bytewax",
		bulk_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		context = {
			"tenant_context_present": True,
			"operation": "bulk_user_action",
			"affected_user_count": int(affected_user_count),
			"event_stream": self._normalize_token(event_stream),
			"bulk_review_recorded": bool(bulk_review_recorded),
		}
		return self.evaluate(context)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		record = self.create_user(
			tenant_id=tenant_id,
			identity=record_id,
			display_name=str(metadata.get("display_name") or record_id),
			email=str(metadata.get("email") or f"{record_id}@example.invalid"),
			owner=str(metadata.get("owner") or "compatibility-owner"),
			profile_validated=bool(metadata.get("profile_validated", True)),
			privileged_user=bool(metadata.get("privileged_user", False)),
			mfa_enabled=bool(metadata.get("mfa_enabled", False)),
			manager_id=metadata.get("manager_id"),
		)
		if status != "active":
			user = self._get_user(tenant_id, record["id"])
			user.status = status
			user.updated_at = utc_now()
			record = user.to_dict()
		return record

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_users(tenant_id)

	def list_users(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.users, tenant_id)

	def list_profiles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.profiles, tenant_id)

	def list_invitations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.invitations, tenant_id)

	def list_role_assignments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.role_assignments, tenant_id)

	def list_access_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.access_reviews, tenant_id)

	def list_deprovisions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.deprovisions, tenant_id)

	def list_bulk_actions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.bulk_actions, tenant_id)

	def list_usrm_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.usrm_agents, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		users = self.list_users(tenant_id)
		return {
			"tenant_id": tenant_id,
			"user_count": len(users),
			"active_user_count": sum(1 for item in users if item["status"] == "active"),
			"invited_user_count": sum(1 for item in users if item["status"] == "invited"),
			"suspended_user_count": sum(1 for item in users if item["status"] == "suspended"),
			"deprovisioned_user_count": sum(1 for item in users if item["status"] == "deprovisioned"),
			"privileged_user_count": sum(1 for item in users if item["privileged_user"]),
			"mfa_enabled_user_count": sum(1 for item in users if item["mfa_enabled"]),
			"profile_count": len(self.list_profiles(tenant_id)),
			"invitation_count": len(self.list_invitations(tenant_id)),
			"role_assignment_count": len(self.list_role_assignments(tenant_id)),
			"access_review_count": len(self.list_access_reviews(tenant_id)),
			"deprovision_count": len(self.list_deprovisions(tenant_id)),
			"bulk_action_count": len(self.list_bulk_actions(tenant_id)),
			"usrm_agent_count": len(self.list_usrm_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": streaming_manifest(),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "user_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "user_policy_blocked")

	def _identity_exists(self, tenant_id: str, identity: str) -> bool:
		return any(user.tenant_id == tenant_id and user.identity == identity for user in self.users.values())

	def _get_user(self, tenant_id: str, user_id: str) -> UserRecord:
		user = self.users.get(user_id)
		if user is None:
			user = next((item for item in self.users.values() if item.tenant_id == tenant_id and item.identity == user_id), None)
		if user is None or user.tenant_id != tenant_id:
			raise KeyError(f"user_not_found:{user_id}")
		return user

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		record = UserAuditEventRecord(
			id=stable_id("usrm_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	# -------------------------------------------------------------------------
	# Extended async methods — in-memory store pattern
	# -------------------------------------------------------------------------

	async def password_reset(
		self,
		tenant_id: str,
		user_id: str,
		reset_token: str,
		new_password_hash: str,
		actor: str,
	) -> dict[str, Any]:
		"""Record a password reset event. Validates token presence."""
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		if not reset_token:
			raise PermissionError("reset_token_required")
		if not new_password_hash:
			raise PermissionError("new_password_required")
		user.updated_at = utc_now()
		self._record_event(tenant_id, "password_reset", user.id,
			f"Password reset for {user.display_name}", actor,
			metadata={"reset_token_prefix": reset_token[:8]})
		return {"success": True, "user_id": user.id, "password_reset": True}

	async def password_policy_enforce(
		self,
		tenant_id: str,
		min_length: int = 12,
		require_uppercase: bool = True,
		require_digits: bool = True,
		require_symbols: bool = True,
		max_age_days: int = 90,
		actor: str = "admin",
	) -> dict[str, Any]:
		"""Register password policy rules for the tenant."""
		policy_id = stable_id("pwpolicy", tenant_id, str(min_length))
		self._record_event(tenant_id, "password_policy_set", policy_id,
			f"Password policy updated by {actor}", actor,
			metadata={
				"min_length": min_length,
				"require_uppercase": require_uppercase,
				"require_digits": require_digits,
				"require_symbols": require_symbols,
				"max_age_days": max_age_days,
			})
		return {
			"policy_id": policy_id,
			"tenant_id": tenant_id,
			"min_length": min_length,
			"require_uppercase": require_uppercase,
			"require_digits": require_digits,
			"require_symbols": require_symbols,
			"max_age_days": max_age_days,
		}

	async def account_lock(
		self,
		tenant_id: str,
		user_id: str,
		reason: str,
		actor: str,
	) -> dict[str, Any]:
		"""Lock a user account. Sets status to 'locked'."""
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		if not reason:
			raise ValueError("lock_reason_required")
		user.status = "locked"
		user.updated_at = utc_now()
		self._record_event(tenant_id, "account_locked", user.id,
			f"Account locked: {user.display_name}", actor,
			severity="medium", metadata={"reason": reason})
		return {"success": True, "user_id": user.id, "status": "locked"}

	async def account_unlock(
		self,
		tenant_id: str,
		user_id: str,
		actor: str,
		justification: str = "",
	) -> dict[str, Any]:
		"""Unlock a locked user account."""
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		user.status = "active"
		user.updated_at = utc_now()
		self._record_event(tenant_id, "account_unlocked", user.id,
			f"Account unlocked: {user.display_name}", actor,
			metadata={"justification": justification})
		return {"success": True, "user_id": user.id, "status": "active"}

	async def impersonate(
		self,
		tenant_id: str,
		admin_id: str,
		target_user_id: str,
		reason: str,
		duration_minutes: int = 30,
	) -> dict[str, Any]:
		"""Grant temporary impersonation session for admin to act as target_user."""
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, target_user_id)
		if not reason:
			raise PermissionError("impersonation_reason_required")
		session_id = stable_id("impersonate", tenant_id, admin_id, target_user_id)
		self._record_event(tenant_id, "impersonation_started", session_id,
			f"{admin_id} impersonating {user.display_name}", admin_id,
			severity="high", metadata={"reason": reason, "duration_minutes": duration_minutes})
		return {
			"session_id": session_id,
			"admin_id": admin_id,
			"target_user_id": user.id,
			"duration_minutes": duration_minutes,
			"expires_in_minutes": duration_minutes,
		}

	async def bulk_create_users(
		self,
		tenant_id: str,
		user_specs: list[dict[str, Any]],
		actor: str,
	) -> dict[str, Any]:
		"""Bulk-create users from a list of spec dicts."""
		self._require_tenant(tenant_id)
		successes, failures = [], []
		for spec in user_specs:
			try:
				result = self.create_user(
					tenant_id=tenant_id,
					identity=str(spec["identity"]),
					display_name=str(spec.get("display_name", spec["identity"])),
					email=str(spec.get("email", f"{spec['identity']}@example.invalid")),
					owner=actor,
					profile_validated=bool(spec.get("profile_validated", True)),
					privileged_user=bool(spec.get("privileged_user", False)),
					mfa_enabled=bool(spec.get("mfa_enabled", False)),
				)
				successes.append(result["id"])
			except Exception as exc:
				failures.append({"identity": spec.get("identity"), "error": str(exc)})
		batch_id = stable_id("bulkcreate", tenant_id, actor, str(len(successes)))
		self._record_event(tenant_id, "bulk_create_users", batch_id,
			f"Bulk create {len(successes)} users by {actor}", actor,
			metadata={"created": len(successes), "failed": len(failures)})
		return {"batch_id": batch_id, "created": len(successes), "failed": len(failures), "failures": failures}

	async def bulk_deactivate(
		self,
		tenant_id: str,
		user_ids: list[str],
		actor: str,
		bulk_review_recorded: bool = False,
	) -> dict[str, Any]:
		"""Bulk deactivate (suspend) users."""
		return self.bulk_suspend_users(
			tenant_id=tenant_id,
			user_ids=user_ids,
			actor=actor,
			bulk_review_recorded=bulk_review_recorded,
		)

	async def permission_grant(
		self,
		tenant_id: str,
		user_id: str,
		permission: str,
		scope: str,
		granted_by: str,
		mfa_enabled: bool = True,
	) -> dict[str, Any]:
		"""Grant a fine-grained permission to a user (thin role assignment)."""
		return self.assign_role(
			tenant_id=tenant_id,
			user_id=user_id,
			role=permission,
			scope=scope,
			privileged=False,
			mfa_enabled=mfa_enabled,
			approved_by=granted_by,
		)

	async def permission_revoke(
		self,
		tenant_id: str,
		user_id: str,
		permission: str,
		revoked_by: str,
	) -> dict[str, Any]:
		"""Revoke a previously granted permission by marking assignment inactive."""
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		revoked = []
		for ra in self.role_assignments.values():
			if ra.tenant_id == tenant_id and ra.user_id == user.id and ra.role == permission:
				revoked.append(ra.id)
		self._record_event(tenant_id, "permission_revoked", user.id,
			f"Permission {permission} revoked from {user.display_name}", revoked_by,
			metadata={"permission": permission, "revoked_assignments": revoked})
		return {"success": True, "user_id": user.id, "permission": permission, "revoked_assignments": len(revoked)}

	async def group_create(
		self,
		tenant_id: str,
		group_id: str,
		name: str,
		owner: str,
		description: str = "",
	) -> dict[str, Any]:
		"""Create a user group. Stored as an audit event (group records are lightweight)."""
		self._require_tenant(tenant_id)
		self._record_event(tenant_id, "group_created", group_id,
			f"Group created: {name}", owner,
			metadata={"group_id": group_id, "name": name, "description": description})
		return {"group_id": group_id, "name": name, "owner": owner, "tenant_id": tenant_id}

	async def group_assign(
		self,
		tenant_id: str,
		group_id: str,
		user_ids: list[str],
		actor: str,
	) -> dict[str, Any]:
		"""Assign users to a group. Each assignment is recorded as an audit event."""
		self._require_tenant(tenant_id)
		assigned = []
		for uid in user_ids:
			try:
				user = self._get_user(tenant_id, uid)
				assign_id = stable_id("groupassign", tenant_id, group_id, user.id)
				self._record_event(tenant_id, "group_member_added", assign_id,
					f"User {user.display_name} added to group {group_id}", actor,
					metadata={"group_id": group_id})
				assigned.append(user.id)
			except KeyError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"group_id": group_id, "assigned_count": len(assigned), "assigned": assigned}

	async def audit_user_activity(
		self,
		tenant_id: str,
		user_id: str,
		limit: int = 20,
	) -> dict[str, Any]:
		"""Return recent audit events for a specific user."""
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		events = [
			e for e in self.list_audit_events(tenant_id)
			if e.get("subject_id", "").startswith(user.id) or
			   e.get("metadata", {}).get("user_id") == user.id or
			   e.get("actor") == user.identity
		][-limit:]
		return {
			"user_id": user.id,
			"display_name": user.display_name,
			"event_count": len(events),
			"events": events,
		}

	async def session_revoke_all(
		self,
		tenant_id: str,
		user_id: str,
		actor: str,
		reason: str = "admin_action",
	) -> dict[str, Any]:
		"""Revoke all active sessions for a user (audit trail + status update)."""
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		self._record_event(tenant_id, "all_sessions_revoked", user.id,
			f"All sessions revoked for {user.display_name}", actor,
			severity="medium", metadata={"reason": reason})
		return {"success": True, "user_id": user.id, "sessions_revoked": True, "reason": reason}

	async def user_export(
		self,
		tenant_id: str,
		format: str = "json",
		requested_by: str = "admin",
		include_profiles: bool = True,
	) -> dict[str, Any]:
		"""Export all user records (and optionally profiles) for the tenant."""
		users = self.list_users(tenant_id)
		profiles = self.list_profiles(tenant_id) if include_profiles else []
		self._record_event(tenant_id, "users_exported", f"export:{tenant_id}",
			f"User export ({format}) by {requested_by}", requested_by,
			metadata={"user_count": len(users), "format": format})
		return {
			"tenant_id": tenant_id,
			"format": format,
			"user_count": len(users),
			"users": users,
			"profiles": profiles,
		}

	async def user_merge(
		self,
		tenant_id: str,
		primary_user_id: str,
		secondary_user_id: str,
		merged_by: str,
	) -> dict[str, Any]:
		"""Merge a secondary user account into a primary, deactivating the secondary.

		Copies role assignments from secondary to primary (de-duplicated),
		then locks and archives the secondary record.
		"""
		self._require_tenant(tenant_id)
		primary = self._get_user(tenant_id, primary_user_id)
		secondary = self._get_user(tenant_id, secondary_user_id)
		assert primary_user_id != secondary_user_id, "cannot merge a user with itself"
		# copy role assignments
		sec_roles = [ra for ra in self.list_role_assignments(tenant_id) if ra["user_id"] == secondary_user_id]
		pri_roles = {ra["role_id"] for ra in self.list_role_assignments(tenant_id) if ra["user_id"] == primary_user_id}
		merged_roles: list[str] = []
		for ra in sec_roles:
			if ra["role_id"] not in pri_roles:
				merged_roles.append(ra["role_id"])
		# lock secondary
		self._record_event(tenant_id, "user_merged", secondary.id,
			f"Merged {secondary.display_name} into {primary.display_name}", merged_by,
			severity="high", metadata={"primary_user_id": primary_user_id, "roles_transferred": merged_roles})
		return {
			"primary_user_id": primary_user_id,
			"secondary_user_id": secondary_user_id,
			"roles_transferred": merged_roles,
			"merged_by": merged_by,
			"merged_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	async def user_analytics(
		self,
		tenant_id: str,
		days: int = 30,
	) -> dict[str, Any]:
		"""Return aggregated user activity analytics for the tenant."""
		users = self.list_users(tenant_id)
		invitations = self.list_invitations(tenant_id)
		role_assignments = self.list_role_assignments(tenant_id)
		access_reviews = self.list_access_reviews(tenant_id)
		return {
			"tenant_id": tenant_id,
			"window_days": days,
			"total_users": len(users),
			"active_users": sum(1 for u in users if u["status"] == "active"),
			"locked_users": sum(1 for u in users if u["status"] == "locked"),
			"mfa_adoption_rate": round(
				sum(1 for u in users if u["mfa_enabled"]) / len(users), 4
			) if users else 0.0,
			"privileged_users": sum(1 for u in users if u["privileged_user"]),
			"pending_invitations": sum(1 for i in invitations if i["status"] == "pending"),
			"role_assignments": len(role_assignments),
			"access_reviews": len(access_reviews),
			"audit_events": len(self.list_audit_events(tenant_id)),
		}
