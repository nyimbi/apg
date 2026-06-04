"""Service-level tests for telecom_pro."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_describe_returns_contract():
	mod = _load("svc_desc_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_pro"


def test_all_workflow_types_accepted():
	mod = _load("svc_wf_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	for i, wtype in enumerate(["service_activation", "service_modification", "service_termination", "sim_provisioning"]):
		wf = svc.start_workflow(f"wf-{i}", "t1", wtype, f"ORD-{i:03d}", "2026-01-01")
		assert wf["workflow_type"] == wtype


def test_resource_reservation_ttl():
	mod = _load("svc_res_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	svc.start_workflow("wf-1", "t1", "service_activation", "ORD-001", "2026-01-01")
	res = svc.reserve_resource("res-1", "t1", "wf-1", "msisdn", "+254700000001", "2026-01-01T09:00:00", "2026-01-01T09:30:00")
	assert res["expires_at"] == "2026-01-01T09:30:00"
	assert res["released"] is False


def test_config_push_all_methods():
	mod = _load("svc_push_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	for i, method in enumerate(["cli_template", "netconf", "restconf", "rest_api"]):
		svc.start_workflow(f"wf-{i}", "t1", "service_activation", f"ORD-{i:03d}", "2026-01-01")
		push = svc.push_config(f"push-{i}", "t1", f"wf-{i}", f"NE-{i:03d}", method, f"tpl-{i}", "2026-01-01")
		assert push["push_method"] == method


def test_activation_confirmation_stored():
	mod = _load("svc_act_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	svc.start_workflow("wf-1", "t1", "service_activation", "ORD-001", "2026-01-01")
	act = svc.confirm_activation("act-1", "t1", "wf-1", "SVC-001", "2026-01-01T10:00:00", "engineer-1")
	assert act["confirmed_by"] == "engineer-1"
	assert act["e2e_test_passed"] is True


def test_rollback_trigger_types():
	mod = _load("svc_rb_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	for i, trigger in enumerate(["manual", "timeout", "network_error", "verification_failure"]):
		svc.start_workflow(f"wf-{i}", "t1", "service_activation", f"ORD-{i}", "2026-01-01")
		rb = svc.trigger_rollback(f"rb-{i}", "t1", f"wf-{i}", trigger, f"desc-{i}", "2026-01-01")
		assert rb["trigger"] == trigger


def test_bulk_provisioning_requires_approval():
	mod = _load("svc_bulk_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	with pytest.raises(PermissionError, match="bulk_provisioning_approval_required"):
		svc.start_bulk_provisioning("bulk-1", "t1", "service_activation", 100, "", "user", "2026-01-01")


def test_multi_tenant_workflow_isolation():
	mod = _load("svc_iso_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	svc.start_workflow("wf-1", "tenant-a", "service_activation", "ORD-A", "2026-01-01")
	svc.start_workflow("wf-1", "tenant-b", "sim_provisioning", "ORD-B", "2026-01-01")
	assert svc.workflows[("tenant-a", "wf-1")].workflow_type == "service_activation"
	assert svc.workflows[("tenant-b", "wf-1")].workflow_type == "sim_provisioning"


def test_failed_workflows_counted_separately():
	mod = _load("svc_fail_pro", PACKAGE_DIR / "service.py")
	svc = mod.TelecomProService()
	svc.start_workflow("wf-1", "t1", "service_activation", "ORD-001", "2026-01-01")
	svc.update_workflow_status("wf-1", "t1", "failed")
	summary = svc.dashboard_summary("t1")
	assert summary["failed_workflow_count"] == 1
	assert summary["workflow_count"] == 1
