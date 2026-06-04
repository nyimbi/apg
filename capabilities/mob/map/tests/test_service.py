"""Service layer tests for mob_map Mobile App Platform."""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
	AppVersionCreate,
	BiometricEnrollmentCreate,
	MobileAppCreate,
	MobileAppUpdate,
	PermissionScopeCreate,
	PushNotificationCreate,
	SyncSessionCreate,
	AppAnalyticsEventCreate,
)
from service import MobileAppPlatformService


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def make_service() -> MobileAppPlatformService:
	return MobileAppPlatformService()


# ---------------------------------------------------------------------------
# App registration
# ---------------------------------------------------------------------------

def test_register_app_happy_path():
	svc = make_service()
	payload = MobileAppCreate(tenant_id="t1", name="FieldApp", bundle_id="ke.datacraft.field", platform="android", category="field_ops", created_by="admin")
	app = run(svc.register_app(payload))
	assert app.id
	assert app.state == "draft"
	assert app.platform == "android"
	assert app.tenant_id == "t1"


def test_register_app_invalid_platform():
	svc = make_service()
	try:
		MobileAppCreate(tenant_id="t1", name="X", bundle_id="x", platform="amiga", category="enterprise", created_by="u1")
		assert False, "should have raised"
	except Exception:
		pass


def test_list_apps_filtered_by_platform():
	svc = make_service()
	run(svc.register_app(MobileAppCreate(tenant_id="t1", name="A", bundle_id="a", platform="ios", category="enterprise", created_by="u1")))
	run(svc.register_app(MobileAppCreate(tenant_id="t1", name="B", bundle_id="b", platform="android", category="enterprise", created_by="u1")))
	ios_apps = run(svc.list_apps("t1", platform="ios"))
	assert len(ios_apps) == 1
	assert ios_apps[0].name == "A"


def test_update_app_state():
	svc = make_service()
	app = run(svc.register_app(MobileAppCreate(tenant_id="t1", name="A", bundle_id="a", platform="ios", category="enterprise", created_by="u1")))
	updated = run(svc.update_app("t1", app.id, MobileAppUpdate(state="review", updated_by="u1")))
	assert updated.state == "review"


def test_suspend_app_requires_reason():
	svc = make_service()
	app = run(svc.register_app(MobileAppCreate(tenant_id="t1", name="A", bundle_id="a", platform="ios", category="enterprise", created_by="u1")))
	try:
		run(svc.update_app("t1", app.id, MobileAppUpdate(state="suspended", updated_by="u1")))
		assert False, "should have raised due to missing suspension reason"
	except (ValueError, AssertionError):
		pass


def test_retire_app():
	svc = make_service()
	app = run(svc.register_app(MobileAppCreate(tenant_id="t1", name="A", bundle_id="a", platform="ios", category="enterprise", created_by="u1")))
	retired = run(svc.retire_app("t1", app.id, updated_by="admin"))
	assert retired.state == "retired"


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def test_publish_and_deploy_version():
	svc = make_service()
	app = run(svc.register_app(MobileAppCreate(tenant_id="t1", name="A", bundle_id="a", platform="android", category="enterprise", created_by="u1")))
	version = run(svc.publish_version(AppVersionCreate(tenant_id="t1", app_id=app.id, version_string="1.0.0", channel="stable", update_policy="recommended", build_number=1, created_by="u1")))
	assert version.state == "draft"
	deployed = run(svc.deploy_version("t1", version.id, approval_reference="appr-001", deployed_by="u1"))
	assert deployed.state == "deployed"
	assert deployed.approval_reference == "appr-001"


def test_deploy_requires_approval():
	svc = make_service()
	app = run(svc.register_app(MobileAppCreate(tenant_id="t1", name="A", bundle_id="a", platform="android", category="enterprise", created_by="u1")))
	version = run(svc.publish_version(AppVersionCreate(tenant_id="t1", app_id=app.id, version_string="1.0.0", channel="stable", update_policy="recommended", build_number=1, created_by="u1")))
	try:
		run(svc.deploy_version("t1", version.id, approval_reference="", deployed_by="u1"))
		assert False, "should have raised"
	except (ValueError, AssertionError):
		pass


# ---------------------------------------------------------------------------
# Sync sessions
# ---------------------------------------------------------------------------

def test_sync_session_lifecycle():
	svc = make_service()
	session = run(svc.start_sync(SyncSessionCreate(tenant_id="t1", app_id="app1", device_id="dev1", sync_strategy="incremental", offline_mode="read_write", conflict_policy="server_wins", encryption_enabled=True, created_by="u1")))
	assert session.state == "in_progress"
	completed = run(svc.complete_sync("t1", session.id, records_synced=50, conflicts_detected=2, bytes_transferred=1024))
	assert completed.state == "completed"
	assert completed.records_synced == 50


def test_sync_denies_unencrypted():
	svc = make_service()
	try:
		run(svc.start_sync(SyncSessionCreate(tenant_id="t1", app_id="app1", device_id="dev1", sync_strategy="full", offline_mode="read_only", conflict_policy="server_wins", encryption_enabled=False, created_by="u1")))
		assert False, "should have denied unencrypted sync"
	except (ValueError, AssertionError):
		pass


def test_conflict_resolution():
	svc = make_service()
	session = run(svc.start_sync(SyncSessionCreate(tenant_id="t1", app_id="app1", device_id="dev1", sync_strategy="delta", offline_mode="full_offline", conflict_policy="manual", encryption_enabled=True, created_by="u1")))
	run(svc.complete_sync("t1", session.id, records_synced=10, conflicts_detected=3, bytes_transferred=512))
	resolved = run(svc.resolve_conflict("t1", session.id, conflict_policy="last_write_wins"))
	assert resolved.conflicts_resolved == 3


# ---------------------------------------------------------------------------
# Push notifications
# ---------------------------------------------------------------------------

def test_send_push_notification():
	svc = make_service()
	notif = run(svc.send_notification(PushNotificationCreate(tenant_id="t1", app_id="app1", channel="push_fcm", title="Hello", body="World", target_type="device", target_reference="dev-001", approval_reference="appr-notif-001", created_by="u1")))
	assert notif.state == "sent"
	assert notif.channel == "push_fcm"


# ---------------------------------------------------------------------------
# Biometric
# ---------------------------------------------------------------------------

def test_biometric_enrollment_and_revocation():
	svc = make_service()
	enrollment = run(svc.enroll_biometric(BiometricEnrollmentCreate(tenant_id="t1", app_id="app1", device_id="dev1", user_id="user1", auth_method="biometric_fingerprint", device_enrolled=True, created_by="u1")))
	assert enrollment.biometric_state == "enrolled"
	revoked = run(svc.revoke_biometric("t1", enrollment.id, reason="user request", revoked_by="admin"))
	assert revoked.biometric_state == "disabled"
	assert revoked.revocation_reason == "user request"


def test_biometric_requires_device_enrolled():
	svc = make_service()
	try:
		run(svc.enroll_biometric(BiometricEnrollmentCreate(tenant_id="t1", app_id="app1", device_id="dev1", user_id="user1", auth_method="biometric_face", device_enrolled=False, created_by="u1")))
		assert False, "should have raised"
	except (ValueError, AssertionError):
		pass


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def test_analytics_summary():
	svc = make_service()
	for i in range(3):
		run(svc.record_analytics_event(AppAnalyticsEventCreate(tenant_id="t1", app_id="app1", device_id=f"dev{i}", event_type="screen_view", created_by="sdk")))
	run(svc.record_analytics_event(AppAnalyticsEventCreate(tenant_id="t1", app_id="app1", device_id="dev0", event_type="button_click", created_by="sdk")))
	summary = run(svc.get_analytics_summary("t1", "app1"))
	assert summary["total_events"] == 4
	assert summary["event_type_counts"]["screen_view"] == 3
	assert summary["unique_devices"] == 3


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_summary():
	svc = make_service()
	run(svc.register_app(MobileAppCreate(tenant_id="t1", name="A", bundle_id="a", platform="ios", category="enterprise", created_by="u1")))
	summary = run(svc.dashboard_summary("t1"))
	assert summary["total_apps"] == 1
	assert "apps_by_state" in summary
	assert "apps_by_platform" in summary
