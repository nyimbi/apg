"""View models for APG Mobile App Platform screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import MobileAppPlatformService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import MobileAppPlatformService  # type: ignore


def dashboard_view(service: MobileAppPlatformService, tenant_id: str = "default") -> dict[str, Any]:
	"""Top-level dashboard view model."""
	import asyncio
	contract = get_capability_contract(tenant_id)
	loop = asyncio.new_event_loop()
	try:
		summary = loop.run_until_complete(service.dashboard_summary(tenant_id))
	finally:
		loop.close()
	return {
		"title": "Mobile App Platform",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def app_registry_view(service: MobileAppPlatformService, tenant_id: str = "default", platform: str | None = None, state: str | None = None) -> dict[str, Any]:
	"""App registry list view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		apps = loop.run_until_complete(service.list_apps(tenant_id, platform=platform, state=state))
	finally:
		loop.close()
	return {
		"tenant_id": tenant_id,
		"filters": {"platform": platform, "state": state},
		"apps": [a.model_dump() for a in apps],
		"count": len(apps),
	}


def app_detail_view(service: MobileAppPlatformService, tenant_id: str, app_id: str) -> dict[str, Any]:
	"""Single app detail view with related versions."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		app = loop.run_until_complete(service.get_app(tenant_id, app_id))
		versions = loop.run_until_complete(service.list_versions(tenant_id, app_id=app_id))
		analytics = loop.run_until_complete(service.get_analytics_summary(tenant_id, app_id))
	finally:
		loop.close()
	return {
		"app": app.model_dump(),
		"versions": [v.model_dump() for v in versions],
		"analytics": analytics,
	}


def version_manager_view(service: MobileAppPlatformService, tenant_id: str, app_id: str | None = None) -> dict[str, Any]:
	"""Version manager list view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		versions = loop.run_until_complete(service.list_versions(tenant_id, app_id=app_id))
	finally:
		loop.close()
	return {
		"tenant_id": tenant_id,
		"app_id": app_id,
		"versions": [v.model_dump() for v in versions],
		"count": len(versions),
	}


def sync_monitor_view(service: MobileAppPlatformService, tenant_id: str, app_id: str | None = None, state: str | None = None) -> dict[str, Any]:
	"""Sync session monitor view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		sessions = loop.run_until_complete(service.list_sync_sessions(tenant_id, app_id=app_id, state=state))
	finally:
		loop.close()
	active = [s for s in sessions if s.state == "in_progress"]
	conflicts = [s for s in sessions if s.conflicts_detected > s.conflicts_resolved]
	return {
		"tenant_id": tenant_id,
		"sessions": [s.model_dump() for s in sessions],
		"active_count": len(active),
		"conflict_count": sum(s.conflicts_detected - s.conflicts_resolved for s in conflicts),
	}


def push_notification_console_view(service: MobileAppPlatformService, tenant_id: str, app_id: str | None = None) -> dict[str, Any]:
	"""Push notification console view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		notifs = loop.run_until_complete(service.list_notifications(tenant_id, app_id=app_id))
	finally:
		loop.close()
	return {
		"tenant_id": tenant_id,
		"notifications": [n.model_dump() for n in notifs],
		"count": len(notifs),
	}


def biometric_console_view(service: MobileAppPlatformService, tenant_id: str, device_id: str | None = None) -> dict[str, Any]:
	"""Biometric enrollment console view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		enrollments = loop.run_until_complete(service.list_biometrics(tenant_id, device_id=device_id))
	finally:
		loop.close()
	return {
		"tenant_id": tenant_id,
		"enrollments": [e.model_dump() for e in enrollments],
		"active_count": sum(1 for e in enrollments if e.biometric_state == "enrolled"),
	}


def permission_scope_view(service: MobileAppPlatformService, tenant_id: str, app_id: str | None = None, scope: str | None = None) -> dict[str, Any]:
	"""Permission scope manager view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		perms = loop.run_until_complete(service.list_permissions(tenant_id, app_id=app_id, scope=scope))
	finally:
		loop.close()
	return {
		"tenant_id": tenant_id,
		"permissions": [p.model_dump() for p in perms],
		"granted_count": sum(1 for p in perms if p.state == "granted"),
	}


def analytics_view(service: MobileAppPlatformService, tenant_id: str, app_id: str) -> dict[str, Any]:
	"""App analytics view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		summary = loop.run_until_complete(service.get_analytics_summary(tenant_id, app_id))
	finally:
		loop.close()
	return {
		"tenant_id": tenant_id,
		"app_id": app_id,
		"analytics": summary,
	}
