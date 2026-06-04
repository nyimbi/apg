"""Service layer tests for mob_rwf Remote Workforce."""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
	ComplianceCheckCreate,
	EquipmentRequisitionCreate,
	OnboardingRecordCreate,
	OnboardingStepCreate,
	PolicyAcknowledgmentCreate,
	ProductivityMetricCreate,
	RemoteIncidentCreate,
	VpnAccessCreate,
	WorkPolicyCreate,
	WorkPolicyUpdate,
)
from service import RemoteWorkforceService


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def make_svc() -> RemoteWorkforceService:
	return RemoteWorkforceService()


# ---------------------------------------------------------------------------
# Work Policies
# ---------------------------------------------------------------------------

def test_create_work_policy():
	svc = make_svc()
	policy = run(svc.create_work_policy(WorkPolicyCreate(
		tenant_id="t1", name="Remote Work Policy", policy_type="hybrid",
		content="All employees may work remotely 3 days per week.", created_by="hr",
	)))
	assert policy.id
	assert policy.state == "draft"
	assert policy.version == 1


def test_activate_work_policy():
	svc = make_svc()
	policy = run(svc.create_work_policy(WorkPolicyCreate(tenant_id="t1", name="P", policy_type="fully_remote", created_by="hr")))
	activated = run(svc.activate_work_policy("t1", policy.id, "appr-001", "hr"))
	assert activated.state == "active"
	assert activated.approval_reference == "appr-001"


def test_activate_requires_approval():
	svc = make_svc()
	policy = run(svc.create_work_policy(WorkPolicyCreate(tenant_id="t1", name="P", policy_type="flexible", created_by="hr")))
	try:
		run(svc.activate_work_policy("t1", policy.id, "", "hr"))
		assert False, "should have raised"
	except (ValueError, AssertionError):
		pass


def test_update_policy_increments_version():
	svc = make_svc()
	policy = run(svc.create_work_policy(WorkPolicyCreate(tenant_id="t1", name="P", policy_type="hybrid", created_by="hr")))
	updated = run(svc.update_work_policy("t1", policy.id, WorkPolicyUpdate(name="P v2", updated_by="hr")))
	assert updated.version == 2


def test_acknowledge_active_policy():
	svc = make_svc()
	policy = run(svc.create_work_policy(WorkPolicyCreate(tenant_id="t1", name="P", policy_type="hybrid", created_by="hr")))
	run(svc.activate_work_policy("t1", policy.id, "appr-001", "hr"))
	ack = run(svc.acknowledge_policy(PolicyAcknowledgmentCreate(
		tenant_id="t1", policy_id=policy.id, employee_id="emp1", created_by="emp1",
	)))
	assert ack.employee_id == "emp1"
	refreshed = run(svc.get_work_policy("t1", policy.id))
	assert refreshed.acknowledgment_count == 1


def test_acknowledge_draft_policy_denied():
	svc = make_svc()
	policy = run(svc.create_work_policy(WorkPolicyCreate(tenant_id="t1", name="P", policy_type="hybrid", created_by="hr")))
	try:
		run(svc.acknowledge_policy(PolicyAcknowledgmentCreate(
			tenant_id="t1", policy_id=policy.id, employee_id="emp1", created_by="emp1",
		)))
		assert False, "should have denied draft policy acknowledgment"
	except (ValueError, AssertionError):
		pass


# ---------------------------------------------------------------------------
# VPN
# ---------------------------------------------------------------------------

def test_provision_and_revoke_vpn():
	svc = make_svc()
	access = run(svc.provision_vpn(VpnAccessCreate(
		tenant_id="t1", employee_id="emp1", vpn_protocol="wireguard",
		approval_reference="appr-vpn-001", mfa_verified=True, created_by="it",
	)))
	assert access.state == "active"
	assert access.split_tunneling_enabled is False
	revoked = run(svc.revoke_vpn("t1", access.id, reason="employee offboarded", revoked_by="it"))
	assert revoked.state == "revoked"


def test_vpn_requires_mfa():
	svc = make_svc()
	try:
		run(svc.provision_vpn(VpnAccessCreate(
			tenant_id="t1", employee_id="emp1", vpn_protocol="openvpn",
			approval_reference="appr-001", mfa_verified=False, created_by="it",
		)))
		assert False, "should have denied VPN without MFA"
	except (ValueError, AssertionError):
		pass


def test_vpn_split_tunneling_denied():
	svc = make_svc()
	try:
		run(svc.provision_vpn(VpnAccessCreate(
			tenant_id="t1", employee_id="emp1", vpn_protocol="openvpn",
			approval_reference="appr-001", mfa_verified=True,
			split_tunneling_requested=True, created_by="it",
		)))
		assert False, "should have denied split tunneling"
	except (ValueError, AssertionError):
		pass


def test_vpn_session_lifecycle():
	svc = make_svc()
	access = run(svc.provision_vpn(VpnAccessCreate(
		tenant_id="t1", employee_id="emp1", vpn_protocol="zerotrust",
		approval_reference="appr-001", mfa_verified=True, created_by="it",
	)))
	session = run(svc.start_vpn_session("t1", access.id, client_ip="10.0.0.5"))
	assert session.employee_id == "emp1"
	ended = run(svc.end_vpn_session("t1", session.id, bytes_in=1024, bytes_out=512))
	assert ended.bytes_in == 1024
	assert ended.duration_seconds >= 0


# ---------------------------------------------------------------------------
# Productivity
# ---------------------------------------------------------------------------

def test_record_productivity_requires_consent():
	svc = make_svc()
	try:
		run(svc.record_productivity_metric(ProductivityMetricCreate(
			tenant_id="t1", employee_id="emp1", metric_type="active_hours",
			value=7.5, period_start=datetime.utcnow() - timedelta(hours=8),
			period_end=datetime.utcnow(), consent_given=False, created_by="system",
		)))
		assert False, "should have denied without consent"
	except (ValueError, AssertionError):
		pass


def test_record_productivity_with_consent():
	svc = make_svc()
	metric = run(svc.record_productivity_metric(ProductivityMetricCreate(
		tenant_id="t1", employee_id="emp1", metric_type="active_hours",
		value=7.5, period_start=datetime.utcnow() - timedelta(hours=8),
		period_end=datetime.utcnow(), consent_given=True, created_by="system",
	)))
	assert metric.metric_type == "active_hours"
	assert metric.value == 7.5


def test_productivity_summary():
	svc = make_svc()
	for v in [6.0, 7.0, 8.0]:
		run(svc.record_productivity_metric(ProductivityMetricCreate(
			tenant_id="t1", employee_id="emp1", metric_type="active_hours",
			value=v, period_start=datetime.utcnow() - timedelta(hours=8),
			period_end=datetime.utcnow(), consent_given=True, created_by="system",
		)))
	summary = run(svc.get_productivity_summary("t1", "emp1"))
	assert summary["total_records"] == 3
	assert abs(summary["metric_averages"]["active_hours"] - 7.0) < 0.01


# ---------------------------------------------------------------------------
# Equipment
# ---------------------------------------------------------------------------

def test_equipment_requisition_lifecycle():
	svc = make_svc()
	req = run(svc.request_equipment(EquipmentRequisitionCreate(
		tenant_id="t1", employee_id="emp1", equipment_type="laptop",
		quantity=1, justification="Remote work setup", delivery_address="123 Main St", created_by="emp1",
	)))
	assert req.state == "requested"
	approved = run(svc.approve_equipment("t1", req.id, "appr-eq-001", "manager"))
	assert approved.state == "approved"
	shipped = run(svc.ship_equipment("t1", req.id, "ASSET-001"))
	assert shipped.state == "shipped"
	delivered = run(svc.deliver_equipment("t1", req.id))
	assert delivered.state == "delivered"
	returned = run(svc.return_equipment("t1", req.id, "emp1"))
	assert returned.state == "returned"


def test_equipment_limit_enforced():
	svc = make_svc()
	# Approve 5 items to reach the limit
	for i in range(5):
		req = run(svc.request_equipment(EquipmentRequisitionCreate(
			tenant_id="t1", employee_id="emp1", equipment_type="keyboard",
			quantity=1, justification="Needed", delivery_address="Addr", created_by="emp1",
		)))
		run(svc.approve_equipment("t1", req.id, f"appr-{i}", "manager"))
	# Now requesting one more should fail
	try:
		run(svc.request_equipment(EquipmentRequisitionCreate(
			tenant_id="t1", employee_id="emp1", equipment_type="mouse",
			quantity=1, justification="Extra", delivery_address="Addr", created_by="emp1",
		)))
		assert False, "should have enforced equipment limit"
	except (ValueError, AssertionError):
		pass


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

def test_onboarding_lifecycle():
	svc = make_svc()
	record = run(svc.start_onboarding(OnboardingRecordCreate(
		tenant_id="t1", employee_id="emp_new", manager_id="mgr1",
		manager_approval_reference="appr-mgr-001", start_date=datetime.utcnow(),
		created_by="hr",
	)))
	assert record.state == "in_progress"
	assert len(record.pending_steps) > 0


def test_onboarding_requires_manager_approval():
	svc = make_svc()
	try:
		run(svc.start_onboarding(OnboardingRecordCreate(
			tenant_id="t1", employee_id="emp_new", manager_id="mgr1",
			manager_approval_reference="", start_date=datetime.utcnow(), created_by="hr",
		)))
		assert False, "should have denied without manager approval"
	except (ValueError, AssertionError):
		pass


def test_onboarding_step_completion():
	svc = make_svc()
	record = run(svc.start_onboarding(OnboardingRecordCreate(
		tenant_id="t1", employee_id="emp_new", manager_id="mgr1",
		manager_approval_reference="appr-001", start_date=datetime.utcnow(), created_by="hr",
	)))
	step = run(svc.complete_onboarding_step(OnboardingStepCreate(
		tenant_id="t1", onboarding_id=record.id, step_type="identity_verification",
		completed_by="hr", created_by="hr",
	)))
	assert step.step_type == "identity_verification"
	refreshed = run(svc.get_onboarding_record("t1", record.id))
	assert "identity_verification" in refreshed.completed_steps


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

def test_record_compliance_check():
	svc = make_svc()
	check = run(svc.record_compliance_check(ComplianceCheckCreate(
		tenant_id="t1", employee_id="emp1", check_type="policy_acknowledgment",
		result="pass", created_by="compliance",
	)))
	assert check.result == "pass"
	assert check.next_due_at is not None


def test_compliance_fail_listed():
	svc = make_svc()
	run(svc.record_compliance_check(ComplianceCheckCreate(
		tenant_id="t1", employee_id="emp1", check_type="security_training",
		result="fail", created_by="compliance",
	)))
	fails = run(svc.list_compliance_checks("t1", result="fail"))
	assert len(fails) == 1


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

def test_raise_and_resolve_incident():
	svc = make_svc()
	incident = run(svc.raise_incident(RemoteIncidentCreate(
		tenant_id="t1", employee_id="emp1", incident_type="vpn_anomaly",
		description="Unusual VPN login", severity="high", reported_by="soc", created_by="system",
	)))
	assert incident.state == "open"
	resolved = run(svc.resolve_incident("t1", incident.id, "VPN access reviewed and cleared", "analyst"))
	assert resolved.state == "resolved"
	assert resolved.resolved_by == "analyst"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_summary():
	svc = make_svc()
	run(svc.create_work_policy(WorkPolicyCreate(tenant_id="t1", name="P", policy_type="hybrid", created_by="hr")))
	summary = run(svc.dashboard_summary("t1"))
	assert summary["total_work_policies"] == 1
	assert "open_incidents" in summary
	assert "active_vpn_access" in summary
