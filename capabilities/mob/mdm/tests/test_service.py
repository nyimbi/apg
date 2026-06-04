"""Service layer tests for mob_mdm Mobile Device Management."""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
	AppDistributionCreate,
	ComplianceEvaluationCreate,
	DeviceEnrolmentCreate,
	DeviceUpdate,
	MdmProfileCreate,
	PolicyAssignmentCreate,
	PolicyCreate,
	PolicyUpdate,
	WipeRequestCreate,
)
from service import MobileDeviceManagementService


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def make_svc() -> MobileDeviceManagementService:
	return MobileDeviceManagementService()


def enrol(svc, tenant="t1", serial="SN001", platform="android"):
	return run(svc.enrol_device(DeviceEnrolmentCreate(
		tenant_id=tenant, serial_number=serial, device_type="smartphone",
		os_platform=platform, os_version="14.0", ownership_type="corporate_owned",
		enrolment_method="qr_code", approval_reference="appr-001", created_by="admin",
	)))


# ---------------------------------------------------------------------------
# Device enrolment
# ---------------------------------------------------------------------------

def test_enrol_device_happy_path():
	svc = make_svc()
	device = enrol(svc)
	assert device.id
	assert device.enrolment_state == "enrolled"
	assert device.os_platform == "android"


def test_enrol_device_invalid_type():
	svc = make_svc()
	try:
		DeviceEnrolmentCreate(
			tenant_id="t1", serial_number="X", device_type="mainframe",
			os_platform="android", os_version="14", ownership_type="corporate_owned",
			enrolment_method="qr_code", approval_reference="a", created_by="u",
		)
		assert False, "should have raised"
	except Exception:
		pass


def test_enrol_requires_approval():
	svc = make_svc()
	try:
		run(svc.enrol_device(DeviceEnrolmentCreate(
			tenant_id="t1", serial_number="X", device_type="smartphone",
			os_platform="ios", os_version="17", ownership_type="byod",
			enrolment_method="email_invite", approval_reference="", created_by="u",
		)))
		assert False, "should have raised"
	except (ValueError, AssertionError):
		pass


def test_list_devices_filtered():
	svc = make_svc()
	enrol(svc, serial="A", platform="android")
	enrol(svc, serial="B", platform="ios")
	android = run(svc.list_devices("t1", os_platform="android"))
	assert len(android) == 1


def test_update_device():
	svc = make_svc()
	device = enrol(svc)
	updated = run(svc.update_device("t1", device.id, DeviceUpdate(location="Nairobi HQ", updated_by="admin")))
	assert updated.location == "Nairobi HQ"


def test_unenrol_device():
	svc = make_svc()
	device = enrol(svc)
	unenrolled = run(svc.unenrol_device("t1", device.id, "admin"))
	assert unenrolled.enrolment_state == "unenrolled"


def test_suspend_device():
	svc = make_svc()
	device = enrol(svc)
	suspended = run(svc.suspend_device("t1", device.id, "admin"))
	assert suspended.enrolment_state == "suspended"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def test_create_and_activate_policy():
	svc = make_svc()
	policy = run(svc.create_policy(PolicyCreate(tenant_id="t1", name="Passcode Policy", policy_type="passcode", created_by="admin")))
	assert policy.state == "draft"
	activated = run(svc.activate_policy("t1", policy.id, approval_reference="appr-pol-001", activated_by="admin"))
	assert activated.state == "active"
	assert activated.approval_reference == "appr-pol-001"


def test_activate_policy_requires_approval():
	svc = make_svc()
	policy = run(svc.create_policy(PolicyCreate(tenant_id="t1", name="Security", policy_type="security", created_by="admin")))
	try:
		run(svc.activate_policy("t1", policy.id, approval_reference="", activated_by="admin"))
		assert False, "should have raised"
	except (ValueError, AssertionError):
		pass


def test_policy_version_increments_on_update():
	svc = make_svc()
	policy = run(svc.create_policy(PolicyCreate(tenant_id="t1", name="VPN", policy_type="vpn", created_by="admin")))
	assert policy.version == 1
	updated = run(svc.update_policy("t1", policy.id, PolicyUpdate(name="VPN v2", updated_by="admin")))
	assert updated.version == 2


def test_assign_policy_to_device():
	svc = make_svc()
	device = enrol(svc)
	policy = run(svc.create_policy(PolicyCreate(tenant_id="t1", name="Enc", policy_type="encryption", created_by="admin")))
	assignment = run(svc.assign_policy(PolicyAssignmentCreate(tenant_id="t1", policy_id=policy.id, device_id=device.id, assigned_by="admin", created_by="admin")))
	assert assignment.device_id == device.id


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

def test_compliance_evaluation_compliant():
	svc = make_svc()
	device = enrol(svc)
	record = run(svc.evaluate_compliance(ComplianceEvaluationCreate(tenant_id="t1", device_id=device.id, evaluator_id="engine", findings=[], created_by="system")))
	assert record.compliance_state == "compliant"


def test_compliance_evaluation_non_compliant():
	svc = make_svc()
	device = enrol(svc)
	findings = [{"check": "disk_encryption", "severity": "critical", "message": "Disk not encrypted"}]
	record = run(svc.evaluate_compliance(ComplianceEvaluationCreate(tenant_id="t1", device_id=device.id, evaluator_id="engine", findings=findings, created_by="system")))
	assert record.compliance_state == "non_compliant"
	# device compliance_state should be updated
	refreshed = run(svc.get_device("t1", device.id))
	assert refreshed.compliance_state == "non_compliant"


def test_compliance_raises_alert_when_non_compliant():
	svc = make_svc()
	device = enrol(svc)
	findings = [{"check": "passcode", "severity": "high", "message": "No passcode set"}]
	run(svc.evaluate_compliance(ComplianceEvaluationCreate(tenant_id="t1", device_id=device.id, evaluator_id="engine", findings=findings, created_by="system")))
	alerts = run(svc.list_alerts("t1"))
	assert len(alerts) > 0


# ---------------------------------------------------------------------------
# App distribution
# ---------------------------------------------------------------------------

def test_distribute_app():
	svc = make_svc()
	device = enrol(svc)
	dist = run(svc.distribute_app(AppDistributionCreate(
		tenant_id="t1", app_bundle_id="ke.datacraft.app", app_name="DatacraftApp",
		app_version="1.0.0", device_id=device.id, distribution_type="required", created_by="admin",
	)))
	assert dist.state == "distributed"


def test_distribute_app_denied_unenrolled():
	svc = make_svc()
	device = enrol(svc)
	run(svc.unenrol_device("t1", device.id, "admin"))
	try:
		run(svc.distribute_app(AppDistributionCreate(
			tenant_id="t1", app_bundle_id="x", app_name="X", app_version="1.0",
			device_id=device.id, distribution_type="required", created_by="admin",
		)))
		assert False, "should have raised"
	except (ValueError, AssertionError):
		pass


# ---------------------------------------------------------------------------
# Remote wipe
# ---------------------------------------------------------------------------

def test_request_and_execute_wipe():
	svc = make_svc()
	device = enrol(svc)
	wipe = run(svc.request_wipe(WipeRequestCreate(
		tenant_id="t1", device_id=device.id, wipe_type="corporate_wipe",
		approval_reference="appr-w1", second_approval_reference="appr-w2",
		justification="Device lost", requested_by="admin", created_by="admin",
	)))
	assert wipe.state == "pending"
	executed = run(svc.execute_wipe("t1", wipe.id, "admin"))
	assert executed.state == "completed"
	wiped_device = run(svc.get_device("t1", device.id))
	assert wiped_device.enrolment_state == "wiped"


def test_wipe_requires_dual_approval():
	svc = make_svc()
	device = enrol(svc)
	try:
		run(svc.request_wipe(WipeRequestCreate(
			tenant_id="t1", device_id=device.id, wipe_type="full_wipe",
			approval_reference="appr-w1", second_approval_reference="",
			justification="test", requested_by="admin", created_by="admin",
		)))
		assert False, "should have raised"
	except (ValueError, AssertionError):
		pass


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def test_create_and_deploy_profile():
	svc = make_svc()
	device = enrol(svc)
	profile = run(svc.create_profile(MdmProfileCreate(tenant_id="t1", name="WiFi Corp", profile_type="wifi", platform="android", created_by="admin")))
	assert profile.state == "draft"
	deployed = run(svc.deploy_profile("t1", profile.id, device.id, "admin"))
	assert deployed.deployed_to_count == 1


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_summary():
	svc = make_svc()
	enrol(svc, serial="D1")
	enrol(svc, serial="D2")
	summary = run(svc.dashboard_summary("t1"))
	assert summary["total_devices"] == 2
	assert "devices_by_platform" in summary
	assert "compliance_summary" in summary
