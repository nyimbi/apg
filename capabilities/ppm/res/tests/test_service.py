"""Service tests for PPM Resource Management (res)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	# Evict generic module names that other capabilities may have cached,
	# then prepend this capability's directory so fallback imports resolve correctly.
	_pkg = str(path.parent)
	for _key in ("capability_contract", "models", "service"):
		sys.modules.pop(_key, None)
	if _pkg not in sys.path:
		sys.path.insert(0, _pkg)
	else:
		sys.path.remove(_pkg)
		sys.path.insert(0, _pkg)
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def _svc():
	mod = _load(f"svc_ppm_res_{id(object())}", PACKAGE_DIR / "service.py")
	return mod.ResourceManagementService()


def test_full_resource_lifecycle():
	svc = _svc()
	resource = svc.create_resource("res-1", "t1", "Alice Engineer", "human", "available", "engineering", "manager-1", 150.0, "standard_cost", "evidence-r")
	skill = svc.add_skill("sk-1", "t1", "res-1", "Python", "expert", 5.0, "evidence-sk")
	alloc = svc.create_allocation("al-1", "t1", "res-1", "proj-1", "task-1", "confirmed", "2026-01-01", "2026-03-31", 80.0, "")
	capacity = svc.create_capacity_plan("cp-1", "t1", "staffing_plan", "Q2 Staffing", "medium_term_90d", '{"demand":10}', '{"supply":8}', '{"gap":2}', "planner-1")
	util = svc.take_utilisation_snapshot("us-1", "t1", "res-1", "2026-Q1", 128.0, 160.0)
	demand = svc.forecast_demand("df-1", "t1", "medium_term_90d", "human", "Python", 12.0, 8.0, "system")
	leave = svc.record_leave("lv-1", "t1", "res-1", "annual_leave", "2026-04-01", "2026-04-07", "approval-lv")
	rate = svc.set_cost_rate("cr-1", "t1", "res-1", "billing_rate", 200.0, "USD", "2026-01-01", "finance-approval-1")
	agent = svc.register_agent("ag-1", "t1", "Resource Bot", "codex", "resource_planner", "resource planning")

	assert resource["resource_type"] == "human"
	assert skill["proficiency_level"] == "expert"
	assert alloc["allocation_pct"] == 80.0
	assert capacity["plan_type"] == "staffing_plan"
	assert util["utilisation_pct"] == 80.0
	assert util["utilisation_band"] == "optimal"
	assert demand["gap_fte"] == 4.0
	assert leave["leave_type"] == "annual_leave"
	assert rate["rate_type"] == "billing_rate"
	assert agent["role"] == "resource_planner"


def test_utilisation_band_over_capacity():
	svc = _svc()
	svc.create_resource("res-oc", "t1", "Bob", "human", "available", "engineering", "mgr", 100.0, "standard_cost", "ev")
	snap = svc.take_utilisation_snapshot("us-oc", "t1", "res-oc", "Q1", 180.0, 160.0)
	assert snap["utilisation_band"] == "over_capacity"
	assert snap["utilisation_pct"] > 100.0


def test_skill_matching():
	svc = _svc()
	svc.create_resource("res-sm1", "t1", "Dev A", "human", "available", "engineering", "mgr", 100.0, "standard_cost", "ev")
	svc.create_resource("res-sm2", "t1", "Dev B", "human", "available", "engineering", "mgr", 100.0, "standard_cost", "ev")
	svc.add_skill("sk-sm1a", "t1", "res-sm1", "Python", "expert", 5.0, "ev")
	svc.add_skill("sk-sm1b", "t1", "res-sm1", "Django", "proficient", 3.0, "ev")
	svc.add_skill("sk-sm2", "t1", "res-sm2", "Java", "expert", 4.0, "ev")

	matched = svc.match_skills("t1", ["Python", "Django"])
	assert len(matched) == 1
	assert matched[0]["id"] == "res-sm1"


def test_tenant_isolation():
	svc = _svc()
	svc.create_resource("res-a", "tenant-a", "A", "human", "available", "eng", "mgr", 100.0, "standard_cost", "ev")
	svc.create_resource("res-a", "tenant-b", "A", "human", "available", "eng", "mgr", 100.0, "standard_cost", "ev")
	assert svc.dashboard_summary("tenant-a")["resource_count"] == 1
	assert svc.dashboard_summary("tenant-b")["resource_count"] == 1


def test_guardrail_unsupported_resource_type():
	svc = _svc()
	with pytest.raises(PermissionError, match="resource_type_not_supported"):
		svc.create_resource("r", "t1", "X", "unicorn", "available", "eng", "mgr", 100.0, "standard_cost", "ev")


def test_guardrail_over_allocation_requires_approval():
	svc = _svc()
	svc.create_resource("res-oa", "t1", "R", "human", "available", "eng", "mgr", 100.0, "standard_cost", "ev")
	with pytest.raises(PermissionError, match="over_allocation_requires_manager_approval"):
		svc.create_allocation("al", "t1", "res-oa", "proj", "task", "confirmed", "", "", 120.0, "", over_allocated=True)


def test_guardrail_cost_rate_requires_finance_approval():
	svc = _svc()
	svc.create_resource("res-cr", "t1", "R", "human", "available", "eng", "mgr", 100.0, "standard_cost", "ev")
	with pytest.raises(PermissionError, match="cost_rate_change_requires_finance_approval"):
		svc.set_cost_rate("cr", "t1", "res-cr", "billing_rate", 200.0, "USD", "2026-01-01", "")


def test_guardrail_leave_requires_approval():
	svc = _svc()
	svc.create_resource("res-lv", "t1", "R", "human", "available", "eng", "mgr", 100.0, "standard_cost", "ev")
	with pytest.raises(PermissionError, match="leave_approval_required"):
		svc.record_leave("lv", "t1", "res-lv", "annual_leave", "2026-05-01", "2026-05-05", "")


def test_batch_requires_bytewax():
	svc = _svc()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 5, event_stream="kafka")
