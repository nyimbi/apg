"""Flask Blueprint REST API for APG Mobile App Platform."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .capability_contract import get_capability_contract
	from .models import (
		AppVersionCreate,
		BiometricEnrollmentCreate,
		MobileAppCreate,
		MobileAppUpdate,
		PermissionScopeCreate,
		PushNotificationCreate,
		SyncSessionCreate,
		AppAnalyticsEventCreate,
	)
	from .service import MobileAppPlatformService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from models import (  # type: ignore
		AppVersionCreate,
		BiometricEnrollmentCreate,
		MobileAppCreate,
		MobileAppUpdate,
		PermissionScopeCreate,
		PushNotificationCreate,
		SyncSessionCreate,
		AppAnalyticsEventCreate,
	)
	from service import MobileAppPlatformService  # type: ignore


bp = Blueprint("mob_map", __name__, url_prefix="/api/mob/map")
_svc = MobileAppPlatformService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400):
	return jsonify({"status": "error", "message": msg}), status


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@bp.get("/contract")
def get_contract():
	"""Return capability contract.
	---
	GET /api/mob/map/contract
	"""
	return _ok(get_capability_contract(_tenant()))


# ---------------------------------------------------------------------------
# Apps
# ---------------------------------------------------------------------------

@bp.get("/apps")
def list_apps():
	"""List mobile apps.
	---
	GET /api/mob/map/apps
	Query: platform, state
	Permission: mob_map:apps:list
	"""
	platform = request.args.get("platform")
	state = request.args.get("state")
	try:
		apps = _run(_svc.list_apps(_tenant(), platform=platform, state=state))
		return _ok([a.model_dump() for a in apps])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/apps")
def create_app():
	"""Register a new mobile app.
	---
	POST /api/mob/map/apps
	Permission: mob_map:apps:create
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = MobileAppCreate(**body)
		app = _run(_svc.register_app(payload))
		return _ok(app.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))
	except Exception as exc:
		return _err(str(exc))


@bp.get("/apps/<app_id>")
def get_app(app_id: str):
	"""Get a specific app.
	---
	GET /api/mob/map/apps/<app_id>
	Permission: mob_map:apps:view
	"""
	try:
		app = _run(_svc.get_app(_tenant(), app_id))
		return _ok(app.model_dump())
	except AssertionError as exc:
		return _err(str(exc), 404)


@bp.put("/apps/<app_id>")
def update_app(app_id: str):
	"""Update app metadata or state.
	---
	PUT /api/mob/map/apps/<app_id>
	Permission: mob_map:apps:edit
	"""
	body = request.get_json(force=True) or {}
	try:
		payload = MobileAppUpdate(**body)
		app = _run(_svc.update_app(_tenant(), app_id, payload))
		return _ok(app.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.delete("/apps/<app_id>")
def retire_app(app_id: str):
	"""Retire an app.
	---
	DELETE /api/mob/map/apps/<app_id>
	Permission: mob_map:apps:retire
	"""
	updated_by = request.get_json(force=True, silent=True) or {}
	try:
		app = _run(_svc.retire_app(_tenant(), app_id, updated_by.get("updated_by", "system")))
		return _ok(app.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

@bp.get("/versions")
def list_versions():
	"""List app versions.
	---
	GET /api/mob/map/versions
	Query: app_id
	Permission: mob_map:versions:list
	"""
	app_id = request.args.get("app_id")
	try:
		versions = _run(_svc.list_versions(_tenant(), app_id=app_id))
		return _ok([v.model_dump() for v in versions])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/versions")
def publish_version():
	"""Publish a new app version.
	---
	POST /api/mob/map/versions
	Permission: mob_map:versions:publish
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = AppVersionCreate(**body)
		version = _run(_svc.publish_version(payload))
		return _ok(version.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/versions/<version_id>/deploy")
def deploy_version(version_id: str):
	"""Deploy an app version.
	---
	POST /api/mob/map/versions/<version_id>/deploy
	Permission: mob_map:versions:deploy
	"""
	body = request.get_json(force=True) or {}
	try:
		version = _run(_svc.deploy_version(
			_tenant(), version_id,
			body.get("approval_reference", ""),
			body.get("deployed_by", "system"),
		))
		return _ok(version.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/apps/<app_id>/rollback")
def rollback_version(app_id: str):
	"""Rollback to a target version.
	---
	POST /api/mob/map/apps/<app_id>/rollback
	Permission: mob_map:versions:deploy
	"""
	body = request.get_json(force=True) or {}
	try:
		version = _run(_svc.rollback_version(
			_tenant(), app_id,
			body.get("target_version_id", ""),
			body.get("rolled_back_by", "system"),
		))
		return _ok(version.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Sync Sessions
# ---------------------------------------------------------------------------

@bp.get("/sync")
def list_sync():
	"""List sync sessions.
	---
	GET /api/mob/map/sync
	Permission: mob_map:sync:list
	"""
	app_id = request.args.get("app_id")
	state = request.args.get("state")
	try:
		sessions = _run(_svc.list_sync_sessions(_tenant(), app_id=app_id, state=state))
		return _ok([s.model_dump() for s in sessions])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/sync")
def start_sync():
	"""Start a sync session.
	---
	POST /api/mob/map/sync
	Permission: mob_map:sync:start
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = SyncSessionCreate(**body)
		session = _run(_svc.start_sync(payload))
		return _ok(session.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/sync/<session_id>/complete")
def complete_sync(session_id: str):
	"""Mark sync session as completed.
	---
	POST /api/mob/map/sync/<session_id>/complete
	Permission: mob_map:sync:manage
	"""
	body = request.get_json(force=True) or {}
	try:
		session = _run(_svc.complete_sync(
			_tenant(), session_id,
			body.get("records_synced", 0),
			body.get("conflicts_detected", 0),
			body.get("bytes_transferred", 0),
		))
		return _ok(session.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.post("/sync/<session_id>/resolve")
def resolve_conflict(session_id: str):
	"""Resolve conflicts in a sync session.
	---
	POST /api/mob/map/sync/<session_id>/resolve
	Permission: mob_map:sync:resolve
	"""
	body = request.get_json(force=True) or {}
	try:
		session = _run(_svc.resolve_conflict(_tenant(), session_id, body.get("conflict_policy", "server_wins")))
		return _ok(session.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Push Notifications
# ---------------------------------------------------------------------------

@bp.get("/notifications")
def list_notifications():
	"""List notifications.
	---
	GET /api/mob/map/notifications
	Permission: mob_map:notifications:list
	"""
	app_id = request.args.get("app_id")
	try:
		notifs = _run(_svc.list_notifications(_tenant(), app_id=app_id))
		return _ok([n.model_dump() for n in notifs])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/notifications")
def send_notification():
	"""Send a push notification.
	---
	POST /api/mob/map/notifications
	Permission: mob_map:notifications:send
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = PushNotificationCreate(**body)
		notif = _run(_svc.send_notification(payload))
		return _ok(notif.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Biometric Auth
# ---------------------------------------------------------------------------

@bp.get("/auth/biometric")
def list_biometrics():
	"""List biometric enrollments.
	---
	GET /api/mob/map/auth/biometric
	Permission: mob_map:auth:manage
	"""
	device_id = request.args.get("device_id")
	try:
		enrollments = _run(_svc.list_biometrics(_tenant(), device_id=device_id))
		return _ok([e.model_dump() for e in enrollments])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/auth/biometric")
def enroll_biometric():
	"""Enroll biometric auth.
	---
	POST /api/mob/map/auth/biometric
	Permission: mob_map:auth:manage
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = BiometricEnrollmentCreate(**body)
		enrollment = _run(_svc.enroll_biometric(payload))
		return _ok(enrollment.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.delete("/auth/biometric/<enrollment_id>")
def revoke_biometric(enrollment_id: str):
	"""Revoke a biometric enrollment.
	---
	DELETE /api/mob/map/auth/biometric/<enrollment_id>
	Permission: mob_map:auth:manage
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		enrollment = _run(_svc.revoke_biometric(
			_tenant(), enrollment_id,
			body.get("reason", "revoked"),
			body.get("revoked_by", "system"),
		))
		return _ok(enrollment.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Permission Scopes
# ---------------------------------------------------------------------------

@bp.get("/permissions")
def list_permissions():
	"""List permission scope grants.
	---
	GET /api/mob/map/permissions
	Permission: mob_map:permissions:manage
	"""
	app_id = request.args.get("app_id")
	scope = request.args.get("scope")
	try:
		perms = _run(_svc.list_permissions(_tenant(), app_id=app_id, scope=scope))
		return _ok([p.model_dump() for p in perms])
	except Exception as exc:
		return _err(str(exc))


@bp.post("/permissions")
def grant_permission():
	"""Grant a permission scope.
	---
	POST /api/mob/map/permissions
	Permission: mob_map:permissions:manage
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = PermissionScopeCreate(**body)
		perm = _run(_svc.grant_permission(payload))
		return _ok(perm.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


@bp.delete("/permissions/<permission_id>")
def revoke_permission(permission_id: str):
	"""Revoke a permission scope.
	---
	DELETE /api/mob/map/permissions/<permission_id>
	Permission: mob_map:permissions:manage
	"""
	body = request.get_json(force=True, silent=True) or {}
	try:
		perm = _run(_svc.revoke_permission(
			_tenant(), permission_id,
			body.get("reason", "revoked"),
			body.get("revoked_by", "system"),
		))
		return _ok(perm.model_dump())
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@bp.get("/analytics/<app_id>")
def get_analytics(app_id: str):
	"""Get analytics summary for an app.
	---
	GET /api/mob/map/analytics/<app_id>
	Permission: mob_map:analytics:view
	"""
	try:
		summary = _run(_svc.get_analytics_summary(_tenant(), app_id))
		return _ok(summary)
	except Exception as exc:
		return _err(str(exc))


@bp.post("/analytics")
def record_analytics():
	"""Record an analytics event.
	---
	POST /api/mob/map/analytics
	Permission: mob_map:analytics:write
	"""
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant())
	try:
		payload = AppAnalyticsEventCreate(**body)
		event = _run(_svc.record_analytics_event(payload))
		return _ok(event.model_dump(), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc))
