"""Async service layer for APG Mobile App Platform."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from uuid6 import uuid7
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def uuid7str() -> str:
	return str(uuid7())


try:
	from .capability_contract import (
		SUPPORTED_APP_CATEGORIES,
		SUPPORTED_APP_STATES,
		SUPPORTED_AUTH_METHODS,
		SUPPORTED_CONFLICT_POLICIES,
		SUPPORTED_NOTIFICATION_CHANNELS,
		SUPPORTED_OFFLINE_MODES,
		SUPPORTED_PERMISSION_SCOPES,
		SUPPORTED_PLATFORMS,
		SUPPORTED_SYNC_STATES,
		SUPPORTED_SYNC_STRATEGIES,
		SUPPORTED_UPDATE_POLICIES,
		SUPPORTED_VERSION_CHANNELS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		AppAnalyticsEventCreate,
		AppAnalyticsEventResponse,
		AppVersionCreate,
		AppVersionResponse,
		BiometricEnrollmentCreate,
		BiometricEnrollmentResponse,
		MobileAppCreate,
		MobileAppResponse,
		MobileAppUpdate,
		PermissionScopeCreate,
		PermissionScopeResponse,
		PushNotificationCreate,
		PushNotificationResponse,
		SyncSessionCreate,
		SyncSessionResponse,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_APP_CATEGORIES,
		SUPPORTED_APP_STATES,
		SUPPORTED_AUTH_METHODS,
		SUPPORTED_CONFLICT_POLICIES,
		SUPPORTED_NOTIFICATION_CHANNELS,
		SUPPORTED_OFFLINE_MODES,
		SUPPORTED_PERMISSION_SCOPES,
		SUPPORTED_PLATFORMS,
		SUPPORTED_SYNC_STATES,
		SUPPORTED_SYNC_STRATEGIES,
		SUPPORTED_UPDATE_POLICIES,
		SUPPORTED_VERSION_CHANNELS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		AppAnalyticsEventCreate,
		AppAnalyticsEventResponse,
		AppVersionCreate,
		AppVersionResponse,
		BiometricEnrollmentCreate,
		BiometricEnrollmentResponse,
		MobileAppCreate,
		MobileAppResponse,
		MobileAppUpdate,
		PermissionScopeCreate,
		PermissionScopeResponse,
		PushNotificationCreate,
		PushNotificationResponse,
		SyncSessionCreate,
		SyncSessionResponse,
	)


def _present(v: str | None) -> bool:
	return bool(v and v.strip())


def _norm(v: str) -> str:
	return v.strip().lower()


class MobileAppPlatformService:
	"""Tenant-scoped runtime for the Mobile App Platform capability."""

	def __init__(self) -> None:
		self._apps: dict[tuple[str, str], MobileAppResponse] = {}
		self._versions: dict[tuple[str, str], AppVersionResponse] = {}
		self._sync_sessions: dict[tuple[str, str], SyncSessionResponse] = {}
		self._notifications: dict[tuple[str, str], PushNotificationResponse] = {}
		self._biometrics: dict[tuple[str, str], BiometricEnrollmentResponse] = {}
		self._permissions: dict[tuple[str, str], PermissionScopeResponse] = {}
		self._analytics: dict[tuple[str, str], AppAnalyticsEventResponse] = {}
		self._audit_events: list[dict[str, Any]] = []
		self._notification_counts: dict[tuple[str, str], int] = {}  # (tenant, device) -> hourly count

	# -------------------------------------------------------------------------
	# Contract helpers
	# -------------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return capability contract for tenant."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate rules against an arbitrary context."""
		return evaluate_capability_rules(context)

	# -------------------------------------------------------------------------
	# Mobile Apps
	# -------------------------------------------------------------------------

	async def register_app(self, payload: MobileAppCreate) -> MobileAppResponse:
		"""Register a new mobile application."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_app",
			"platform_supported": payload.platform in SUPPORTED_PLATFORMS,
			"category_supported": payload.category in SUPPORTED_APP_CATEGORIES,
		})
		app = MobileAppResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			name=payload.name,
			bundle_id=payload.bundle_id,
			platform=payload.platform,
			category=payload.category,
			state="draft",
			description=payload.description,
			icon_url=payload.icon_url,
			created_by=payload.created_by,
		)
		self._apps[self._key(payload.tenant_id, app.id)] = app
		self._audit(payload.tenant_id, "app_registered", app.id)
		return app

	async def get_app(self, tenant_id: str, app_id: str) -> MobileAppResponse:
		"""Retrieve a mobile app by ID."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		return self._require_app(tenant_id, app_id)

	async def list_apps(self, tenant_id: str, platform: str | None = None, state: str | None = None) -> list[MobileAppResponse]:
		"""List all apps for a tenant with optional filters."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		apps = [a for a in self._apps.values() if a.tenant_id == tenant_id]
		if platform:
			apps = [a for a in apps if a.platform == platform]
		if state:
			apps = [a for a in apps if a.state == state]
		return sorted(apps, key=lambda a: a.created_at)

	async def update_app(self, tenant_id: str, app_id: str, payload: MobileAppUpdate) -> MobileAppResponse:
		"""Update app metadata or state."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "suspend_app" if payload.state == "suspended" else "update_app",
			"reason_present": _present(payload.suspension_reason) if payload.state == "suspended" else True,
		})
		app = self._require_app(tenant_id, app_id)
		if payload.name:
			app.name = payload.name
		if payload.description is not None:
			app.description = payload.description
		if payload.icon_url is not None:
			app.icon_url = payload.icon_url
		if payload.state:
			assert payload.state in SUPPORTED_APP_STATES, f"state must be one of {SUPPORTED_APP_STATES}"
			app.state = payload.state
		if payload.suspension_reason is not None:
			app.suspension_reason = payload.suspension_reason
		app.updated_at = datetime.utcnow()
		self._audit(tenant_id, "app_state_changed", app_id)
		return app

	async def retire_app(self, tenant_id: str, app_id: str, updated_by: str) -> MobileAppResponse:
		"""Retire an app, preventing future deployments."""
		return await self.update_app(tenant_id, app_id, MobileAppUpdate(state="retired", updated_by=updated_by))

	# -------------------------------------------------------------------------
	# App Versions
	# -------------------------------------------------------------------------

	async def publish_version(self, payload: AppVersionCreate) -> AppVersionResponse:
		"""Publish a new app version."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_version",
			"channel_supported": payload.channel in SUPPORTED_VERSION_CHANNELS,
			"update_policy_supported": payload.update_policy in SUPPORTED_UPDATE_POLICIES,
		})
		app = self._require_app(payload.tenant_id, payload.app_id)
		assert app.state != "retired", "retired_apps_cannot_publish_versions"
		version = AppVersionResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			app_id=payload.app_id,
			version_string=payload.version_string,
			channel=payload.channel,
			update_policy=payload.update_policy,
			build_number=payload.build_number,
			release_notes=payload.release_notes,
			environment=payload.environment,
			state="draft",
			created_by=payload.created_by,
		)
		self._versions[self._key(payload.tenant_id, version.id)] = version
		self._audit(payload.tenant_id, "app_version_published", version.id)
		return version

	async def deploy_version(self, tenant_id: str, version_id: str, approval_reference: str, deployed_by: str) -> AppVersionResponse:
		"""Deploy a version after approval."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "deploy_version",
			"approval_present": _present(approval_reference),
		})
		version = self._require_version(tenant_id, version_id)
		app = self._require_app(tenant_id, version.app_id)
		self._enforce({"operation": "deploy_version", "app_state": app.state, "approval_present": True})
		version.approval_reference = approval_reference
		version.state = "deployed"
		version.deployed_at = datetime.utcnow()
		version.updated_at = datetime.utcnow()
		self._audit(tenant_id, "app_version_deployed", version_id)
		return version

	async def rollback_version(self, tenant_id: str, app_id: str, target_version_id: str, rolled_back_by: str) -> AppVersionResponse:
		"""Roll back to a previously deployed version."""
		versions = [v for v in self._versions.values() if v.tenant_id == tenant_id and v.app_id == app_id and v.state == "deployed"]
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "rollback_version",
			"previous_version_exists": len(versions) > 0,
		})
		target = self._require_version(tenant_id, target_version_id)
		rollback = AppVersionResponse(
			id=uuid7str(),
			tenant_id=tenant_id,
			app_id=app_id,
			version_string=target.version_string,
			channel=target.channel,
			update_policy="mandatory",
			build_number=target.build_number,
			environment=target.environment,
			state="deployed",
			rollback_of=target_version_id,
			deployed_at=datetime.utcnow(),
			created_by=rolled_back_by,
		)
		self._versions[self._key(tenant_id, rollback.id)] = rollback
		self._audit(tenant_id, "app_version_deployed", rollback.id)
		return rollback

	async def list_versions(self, tenant_id: str, app_id: str | None = None) -> list[AppVersionResponse]:
		"""List versions, optionally filtered by app."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		versions = [v for v in self._versions.values() if v.tenant_id == tenant_id]
		if app_id:
			versions = [v for v in versions if v.app_id == app_id]
		return sorted(versions, key=lambda v: v.created_at)

	# -------------------------------------------------------------------------
	# Sync Sessions
	# -------------------------------------------------------------------------

	async def start_sync(self, payload: SyncSessionCreate) -> SyncSessionResponse:
		"""Start an offline sync session."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_sync",
			"encryption_enabled": payload.encryption_enabled,
			"sync_strategy_supported": payload.sync_strategy in SUPPORTED_SYNC_STRATEGIES,
			"offline_mode_supported": payload.offline_mode in SUPPORTED_OFFLINE_MODES,
		})
		self._enforce({
			"operation": "configure_offline",
			"offline_mode_supported": payload.offline_mode in SUPPORTED_OFFLINE_MODES,
		})
		session = SyncSessionResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			app_id=payload.app_id,
			device_id=payload.device_id,
			sync_strategy=payload.sync_strategy,
			offline_mode=payload.offline_mode,
			conflict_policy=payload.conflict_policy,
			encryption_enabled=payload.encryption_enabled,
			compression_algorithm=payload.compression_algorithm,
			state="in_progress",
			started_at=datetime.utcnow(),
			created_by=payload.created_by,
		)
		self._sync_sessions[self._key(payload.tenant_id, session.id)] = session
		self._audit(payload.tenant_id, "sync_session_started", session.id)
		return session

	async def complete_sync(self, tenant_id: str, session_id: str, records_synced: int, conflicts_detected: int, bytes_transferred: int) -> SyncSessionResponse:
		"""Mark a sync session as completed."""
		session = self._require_sync_session(tenant_id, session_id)
		session.state = "completed"
		session.records_synced = records_synced
		session.conflicts_detected = conflicts_detected
		session.bytes_transferred = bytes_transferred
		session.completed_at = datetime.utcnow()
		session.updated_at = datetime.utcnow()
		self._audit(tenant_id, "sync_session_completed", session_id)
		return session

	async def fail_sync(self, tenant_id: str, session_id: str, error_message: str) -> SyncSessionResponse:
		"""Mark a sync session as failed."""
		session = self._require_sync_session(tenant_id, session_id)
		session.state = "failed"
		session.error_message = error_message
		session.completed_at = datetime.utcnow()
		session.updated_at = datetime.utcnow()
		return session

	async def resolve_conflict(self, tenant_id: str, session_id: str, conflict_policy: str) -> SyncSessionResponse:
		"""Resolve sync conflicts using the specified policy."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "resolve_conflict",
			"conflict_policy_supported": conflict_policy in SUPPORTED_CONFLICT_POLICIES,
		})
		session = self._require_sync_session(tenant_id, session_id)
		session.conflicts_resolved = session.conflicts_detected
		session.updated_at = datetime.utcnow()
		self._audit(tenant_id, "sync_conflict_resolved", session_id)
		return session

	async def list_sync_sessions(self, tenant_id: str, app_id: str | None = None, state: str | None = None) -> list[SyncSessionResponse]:
		"""List sync sessions with optional filters."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		sessions = [s for s in self._sync_sessions.values() if s.tenant_id == tenant_id]
		if app_id:
			sessions = [s for s in sessions if s.app_id == app_id]
		if state:
			sessions = [s for s in sessions if s.state == state]
		return sorted(sessions, key=lambda s: s.created_at)

	# -------------------------------------------------------------------------
	# Push Notifications
	# -------------------------------------------------------------------------

	async def send_notification(self, payload: PushNotificationCreate) -> PushNotificationResponse:
		"""Dispatch a push notification."""
		device_key = (payload.tenant_id, payload.target_reference)
		hourly_count = self._notification_counts.get(device_key, 0)
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "send_notification",
			"channel_supported": payload.channel in SUPPORTED_NOTIFICATION_CHANNELS,
			"approval_present": _present(payload.approval_reference) if payload.channel in ("push_apns", "push_fcm") else True,
			"rate_limit_exceeded": hourly_count >= 50,
		})
		notification = PushNotificationResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			app_id=payload.app_id,
			channel=payload.channel,
			title=payload.title,
			body=payload.body,
			target_type=payload.target_type,
			target_reference=payload.target_reference,
			approval_reference=payload.approval_reference,
			deep_link=payload.deep_link,
			payload=payload.payload,
			state="sent",
			sent_at=datetime.utcnow(),
			created_by=payload.created_by,
		)
		self._notifications[self._key(payload.tenant_id, notification.id)] = notification
		self._notification_counts[device_key] = hourly_count + 1
		self._audit(payload.tenant_id, "push_notification_sent", notification.id)
		return notification

	async def list_notifications(self, tenant_id: str, app_id: str | None = None) -> list[PushNotificationResponse]:
		"""List notifications for a tenant."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		notifs = [n for n in self._notifications.values() if n.tenant_id == tenant_id]
		if app_id:
			notifs = [n for n in notifs if n.app_id == app_id]
		return sorted(notifs, key=lambda n: n.created_at)

	# -------------------------------------------------------------------------
	# Biometric Auth
	# -------------------------------------------------------------------------

	async def enroll_biometric(self, payload: BiometricEnrollmentCreate) -> BiometricEnrollmentResponse:
		"""Enroll biometric authentication for a user/device pair."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "enroll_biometric",
			"auth_method_supported": payload.auth_method in SUPPORTED_AUTH_METHODS,
			"device_enrolled": payload.device_enrolled,
		})
		enrollment = BiometricEnrollmentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			app_id=payload.app_id,
			device_id=payload.device_id,
			user_id=payload.user_id,
			auth_method=payload.auth_method,
			biometric_state="enrolled",
			enrolled_at=datetime.utcnow(),
			created_by=payload.created_by,
		)
		self._biometrics[self._key(payload.tenant_id, enrollment.id)] = enrollment
		self._audit(payload.tenant_id, "biometric_enrolled", enrollment.id)
		return enrollment

	async def revoke_biometric(self, tenant_id: str, enrollment_id: str, reason: str, revoked_by: str) -> BiometricEnrollmentResponse:
		"""Revoke a biometric enrollment."""
		enrollment = self._require_biometric(tenant_id, enrollment_id)
		enrollment.biometric_state = "disabled"
		enrollment.revoked_at = datetime.utcnow()
		enrollment.revocation_reason = reason
		enrollment.updated_at = datetime.utcnow()
		self._audit(tenant_id, "biometric_revoked", enrollment_id)
		return enrollment

	async def list_biometrics(self, tenant_id: str, device_id: str | None = None) -> list[BiometricEnrollmentResponse]:
		"""List biometric enrollments."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		enrollments = [e for e in self._biometrics.values() if e.tenant_id == tenant_id]
		if device_id:
			enrollments = [e for e in enrollments if e.device_id == device_id]
		return sorted(enrollments, key=lambda e: e.created_at)

	# -------------------------------------------------------------------------
	# Permission Scopes
	# -------------------------------------------------------------------------

	async def grant_permission(self, payload: PermissionScopeCreate) -> PermissionScopeResponse:
		"""Grant a permission scope to an app/device."""
		self._enforce({
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "grant_permission",
			"scope_supported": payload.scope in SUPPORTED_PERMISSION_SCOPES,
		})
		perm = PermissionScopeResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			app_id=payload.app_id,
			device_id=payload.device_id,
			scope=payload.scope,
			granted_by=payload.granted_by,
			justification=payload.justification,
			state="granted",
			created_by=payload.created_by,
		)
		self._permissions[self._key(payload.tenant_id, perm.id)] = perm
		self._audit(payload.tenant_id, "permission_scope_granted", perm.id)
		return perm

	async def revoke_permission(self, tenant_id: str, permission_id: str, reason: str, revoked_by: str) -> PermissionScopeResponse:
		"""Revoke a previously granted permission scope."""
		perm = self._require_permission(tenant_id, permission_id)
		perm.state = "revoked"
		perm.revoked_at = datetime.utcnow()
		perm.revocation_reason = reason
		perm.updated_at = datetime.utcnow()
		self._audit(tenant_id, "permission_scope_revoked", permission_id)
		return perm

	async def list_permissions(self, tenant_id: str, app_id: str | None = None, scope: str | None = None) -> list[PermissionScopeResponse]:
		"""List permission grants with optional filters."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		perms = [p for p in self._permissions.values() if p.tenant_id == tenant_id]
		if app_id:
			perms = [p for p in perms if p.app_id == app_id]
		if scope:
			perms = [p for p in perms if p.scope == scope]
		return sorted(perms, key=lambda p: p.created_at)

	# -------------------------------------------------------------------------
	# Analytics
	# -------------------------------------------------------------------------

	async def record_analytics_event(self, payload: AppAnalyticsEventCreate) -> AppAnalyticsEventResponse:
		"""Record an app analytics event."""
		self._enforce({"tenant_context_present": _present(payload.tenant_id)})
		event = AppAnalyticsEventResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			app_id=payload.app_id,
			device_id=payload.device_id,
			event_type=payload.event_type,
			event_payload=payload.event_payload,
			session_id=payload.session_id,
			created_by=payload.created_by,
		)
		self._analytics[self._key(payload.tenant_id, event.id)] = event
		self._audit(payload.tenant_id, "app_analytics_event", event.id)
		return event

	async def get_analytics_summary(self, tenant_id: str, app_id: str) -> dict[str, Any]:
		"""Summarise analytics for an app."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		events = [e for e in self._analytics.values() if e.tenant_id == tenant_id and e.app_id == app_id]
		type_counts: dict[str, int] = {}
		for ev in events:
			type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1
		return {
			"app_id": app_id,
			"tenant_id": tenant_id,
			"total_events": len(events),
			"event_type_counts": type_counts,
			"unique_devices": len({e.device_id for e in events}),
		}

	# ── 12 new methods ──────────────────────────────────────────────────────

	async def app_register(
		self, tenant_id: str, platform: str, bundle_id: str, signing_cert: str, app_name: str = "app"
	) -> dict[str, Any]:
		"""Register a new mobile app for a tenant."""
		from .models import MobileAppCreate
		payload = MobileAppCreate(
			tenant_id=tenant_id,
			name=app_name,
			platform=platform,
			bundle_id=bundle_id,
			signing_cert=signing_cert,
			created_by="admin",
		)
		return await self.register_app(payload)

	async def app_version_deploy(
		self, tenant_id: str, app_id: str, version: str, package_url: str, deployed_by: str = "ci"
	) -> dict[str, Any]:
		"""Deploy a new version of a mobile app."""
		from .models import AppVersionCreate
		payload = AppVersionCreate(
			tenant_id=tenant_id,
			app_id=app_id,
			version=version,
			package_url=package_url,
			created_by=deployed_by,
		)
		return await self.create_app_version(payload)

	async def push_campaign(
		self,
		tenant_id: str,
		app_id: str,
		segment: str,
		message: str,
		schedule_at: str | None = None,
		title: str = "Notification",
		sent_by: str = "system",
	) -> dict[str, Any]:
		"""Send a push notification campaign to a user segment."""
		from .models import PushNotificationCreate
		payload = PushNotificationCreate(
			tenant_id=tenant_id,
			app_id=app_id,
			title=title,
			body=message,
			target_segment=segment,
			scheduled_at=schedule_at,
			created_by=sent_by,
		)
		return await self.send_push_notification(payload)

	async def deep_link_resolve(
		self, tenant_id: str, link_token: str
	) -> str:
		"""Resolve a deep link token to its destination URL."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		self._audit(tenant_id, "deep_link_resolved", link_token)
		return f"app://resolved/{link_token}"

	async def ab_test_create(
		self, tenant_id: str, app_id: str, feature: str, variants: list[str], created_by: str = "admin"
	) -> dict[str, Any]:
		"""Create an A/B test for a feature flag."""
		test_id = f"abtest-{app_id[:6]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "ab_test_created", test_id)
		return {
			"test_id": test_id,
			"tenant_id": tenant_id,
			"app_id": app_id,
			"feature": feature,
			"variants": variants,
			"status": "active",
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}

	async def crash_report_ingest(
		self, tenant_id: str, app_id: str, report: dict[str, Any]
	) -> dict[str, Any]:
		"""Ingest a crash report from a mobile device."""
		report_id = f"crash-{app_id[:6]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "crash_report_ingested", report_id)
		return {
			"report_id": report_id,
			"tenant_id": tenant_id,
			"app_id": app_id,
			"crash_type": report.get("type", "unknown"),
			"severity": report.get("severity", "medium"),
			"status": "received",
			"ingested_at": datetime.utcnow().isoformat(),
		}

	async def performance_metric_ingest(
		self, tenant_id: str, app_id: str, metrics: dict[str, Any]
	) -> dict[str, Any]:
		"""Ingest performance metrics from a mobile app session."""
		metric_id = f"perf-{app_id[:6]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "performance_metrics_ingested", metric_id)
		return {
			"metric_id": metric_id,
			"tenant_id": tenant_id,
			"app_id": app_id,
			"metrics": metrics,
			"ingested_at": datetime.utcnow().isoformat(),
		}

	async def user_segment_create(
		self, tenant_id: str, name: str, criteria: dict[str, Any], created_by: str = "admin"
	) -> dict[str, Any]:
		"""Create a user segment for targeting."""
		seg_id = f"seg-{name[:8]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "user_segment_created", seg_id)
		return {
			"segment_id": seg_id,
			"tenant_id": tenant_id,
			"name": name,
			"criteria": criteria,
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}

	async def offline_data_sync(
		self, tenant_id: str, device_id: str, data_bundle: dict[str, Any]
	) -> dict[str, Any]:
		"""Sync offline data from a device back to the server."""
		from .models import SyncSessionCreate
		payload = SyncSessionCreate(
			tenant_id=tenant_id,
			device_id=device_id,
			data_payload=data_bundle,
			created_by="device",
		)
		return await self.create_sync_session(payload)

	async def map_analytics(
		self, tenant_id: str, period: str
	) -> dict[str, Any]:
		"""Return mobile app platform analytics for a period."""
		apps = [a for a in self._apps.values() if a.tenant_id == tenant_id]
		versions = [v for v in self._versions.values() if v.tenant_id == tenant_id]
		notifs = [n for n in self._notifications.values() if n.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_apps": len(apps),
			"total_versions": len(versions),
			"push_notifications_sent": len(notifs),
			"sync_sessions": len([s for s in self._sync_sessions.values() if s.tenant_id == tenant_id]),
			"audit_events": sum(1 for e in self._audit_events if e.get("tenant_id") == tenant_id),
		}

	async def app_update_force(
		self, tenant_id: str, app_id: str, min_version: str, reason: str
	) -> dict[str, Any]:
		"""Force all devices to update to at least min_version."""
		self._enforce({"tenant_context_present": _present(tenant_id)})
		self._require_app(tenant_id, app_id)
		update_id = f"forceupd-{app_id[:6]}-{len(self._audit_events)+1}"
		self._audit(tenant_id, "force_update_issued", update_id)
		return {
			"update_id": update_id,
			"tenant_id": tenant_id,
			"app_id": app_id,
			"min_version": min_version,
			"reason": reason,
			"issued_at": datetime.utcnow().isoformat(),
		}

	async def map_health_check(self) -> dict[str, Any]:
		"""Return mobile app platform health status."""
		return {
			"service": "MobileAppPlatformService",
			"status": "healthy",
			"apps": len(self._apps),
			"versions": len(self._versions),
			"notifications": len(self._notifications),
			"sync_sessions": len(self._sync_sessions),
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return a high-level dashboard summary for a tenant."""
		apps = [a for a in self._apps.values() if a.tenant_id == tenant_id]
		versions = [v for v in self._versions.values() if v.tenant_id == tenant_id]
		sessions = [s for s in self._sync_sessions.values() if s.tenant_id == tenant_id]
		return {
			"total_apps": len(apps),
			"apps_by_state": self._count_by(apps, "state"),
			"apps_by_platform": self._count_by(apps, "platform"),
			"total_versions": len(versions),
			"total_sync_sessions": len(sessions),
			"active_sync_sessions": sum(1 for s in sessions if s.state == "in_progress"),
			"total_notifications": len([n for n in self._notifications.values() if n.tenant_id == tenant_id]),
			"total_biometric_enrollments": len([b for b in self._biometrics.values() if b.tenant_id == tenant_id]),
		}

	# -------------------------------------------------------------------------
	# Private helpers
	# -------------------------------------------------------------------------

	def _log_audit_summary(self, tenant_id: str) -> str:
		count = sum(1 for e in self._audit_events if e.get("tenant_id") == tenant_id)
		return f"tenant={tenant_id} audit_events={count}"

	def _log_pretty_key(self, tenant_id: str, entity_id: str) -> str:
		return f"{tenant_id}/{entity_id}"

	def _key(self, tenant_id: str, entity_id: str) -> tuple[str, str]:
		return (tenant_id, entity_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise ValueError(f"{result['reason']}: {result['required_action']}")

	def _audit(self, tenant_id: str, event_type: str, entity_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"timestamp": datetime.utcnow().isoformat(),
		})

	def _require_app(self, tenant_id: str, app_id: str) -> MobileAppResponse:
		app = self._apps.get((tenant_id, app_id))
		assert app is not None, f"app_not_found: {self._log_pretty_key(tenant_id, app_id)}"
		return app

	def _require_version(self, tenant_id: str, version_id: str) -> AppVersionResponse:
		v = self._versions.get((tenant_id, version_id))
		assert v is not None, f"version_not_found: {self._log_pretty_key(tenant_id, version_id)}"
		return v

	def _require_sync_session(self, tenant_id: str, session_id: str) -> SyncSessionResponse:
		s = self._sync_sessions.get((tenant_id, session_id))
		assert s is not None, f"sync_session_not_found: {self._log_pretty_key(tenant_id, session_id)}"
		return s

	def _require_biometric(self, tenant_id: str, enrollment_id: str) -> BiometricEnrollmentResponse:
		e = self._biometrics.get((tenant_id, enrollment_id))
		assert e is not None, f"biometric_not_found: {self._log_pretty_key(tenant_id, enrollment_id)}"
		return e

	def _require_permission(self, tenant_id: str, perm_id: str) -> PermissionScopeResponse:
		p = self._permissions.get((tenant_id, perm_id))
		assert p is not None, f"permission_not_found: {self._log_pretty_key(tenant_id, perm_id)}"
		return p

	def _count_by(self, items: list[Any], attr: str) -> dict[str, int]:
		counts: dict[str, int] = {}
		for item in items:
			k = getattr(item, attr, "unknown")
			counts[k] = counts.get(k, 0) + 1
		return counts
