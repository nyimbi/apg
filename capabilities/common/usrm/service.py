"""Service layer for the User Management capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .user_runtime import (
	AccessReviewRecord,
	BulkUserActionRecord,
	DeprovisionRecord,
	RoleAssignmentRecord,
	UserAuditEventRecord,
	UserInvitationRecord,
	UserProfileRecord,
	UserRecord,
	normalize_access_review_decision,
	stable_id,
	user_required_actions,
	utc_now,
)


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
		if not str(owner or "").strip():
			raise ValueError("user_owner_required")
		if not profile_validated:
			raise PermissionError("profile_validation_required")
		context = {
			"tenant_context_present": True,
			"operation": "create_user",
			"unique_identity_present": True,
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
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		if not str(consent_notice_ref or "").strip():
			raise PermissionError("consent_notice_required")
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
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		context = {
			"tenant_context_present": True,
			"operation": "invite_user",
			"consent_notice_attached": bool(str(consent_notice_ref or "").strip()),
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
		self._record_event(tenant_id, "user_invited", record.id, f"User invited: {user.display_name}", invited_by)
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
			"privileged_user": bool(privileged),
			"mfa_enabled": bool(mfa_enabled),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if privileged and not str(approved_by or "").strip():
			raise PermissionError("role_assignment_approval_required")
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
			raise PermissionError("access_reviewer_required")
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
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		user = self._get_user(tenant_id, user_id)
		context = {
			"tenant_context_present": True,
			"operation": "deprovision_user",
			"access_revoked": bool(access_revoked),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(evidence_ref or "").strip():
			raise PermissionError("deprovision_evidence_required")
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
		self._record_event(tenant_id, "user_deprovisioned", record.id, f"User deprovisioned: {user.display_name}", actor)
		return record.to_dict()

	def bulk_suspend_users(
		self,
		tenant_id: str,
		user_ids: list[str],
		actor: str,
		bulk_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		users = [self._get_user(tenant_id, user_id) for user_id in user_ids]
		context = {
			"tenant_context_present": True,
			"operation": "bulk_suspend_users",
			"affected_user_count": len(users),
			"bulk_review_recorded": bool(bulk_review_recorded),
		}
		result = self.evaluate(context)
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
		self._record_event(tenant_id, "bulk_suspend_users", record.id, f"Bulk suspend {status}: {len(users)} users", actor)
		return record.to_dict()

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
	) -> dict[str, Any]:
		record = UserAuditEventRecord(
			id=stable_id("usrm_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
