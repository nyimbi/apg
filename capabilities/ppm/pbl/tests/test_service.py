"""Service tests for PPM Project Baseline Management (pbl)."""

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
	mod = _load(f"svc_ppm_pbl_{id(object())}", PACKAGE_DIR / "service.py")
	return mod.ProjectBaselineService()


def test_full_baseline_lifecycle():
	svc = _svc()
	baseline = svc.create_baseline("bl-1", "t1", "proj-1", "cost", "draft", "Cost Baseline v1", "Initial cost baseline", "pm-1", "approval-b", "evidence-b")
	approved = svc.approve_baseline("bl-1", "t1", True, "approval-b2", "evidence-ap")
	cr = svc.submit_change_request("cr-1", "t1", "bl-1", "cost_change", "medium", "Add QA resources", "Need QA budget", "pm-1", "impact-ref", "approval-cr", "evidence-cr")
	impact = svc.assess_change_impact("ia-1", "t1", "cr-1", "cost,schedule", 5, 15000.0, "Add QA to scope", "Low risk", "analyst-1", "evidence-ia")
	implemented = svc.implement_change("cr-1", "t1", "approval-impl")
	ev = svc.take_ev_snapshot("ev-1", "t1", "bl-1", "2026-03-31", 80000.0, 75000.0, 78000.0, 100000.0, "typical_performance", 104000.0, 26000.0)
	variance = svc.generate_variance_report("vr-1", "t1", "bl-1", "2026-Q1", -5000.0, -3000.0, 0.94, 0.96, "standard", "system")
	agent = svc.register_agent("ag-1", "t1", "Baseline Bot", "codex", "baseline_custodian", "baseline management")

	assert baseline["baseline_type"] == "cost"
	assert approved["status"] == "approved"
	assert cr["change_type"] == "cost_change"
	assert impact["schedule_impact_days"] == 5
	assert implemented["status"] == "implemented"
	assert ev["pv"] == 80000.0
	assert ev["forecasting_method"] == "typical_performance"
	assert variance["threshold_breached"] is False
	assert agent["role"] == "baseline_custodian"


def test_ev_spi_cpi_computed():
	svc = _svc()
	svc.create_baseline("bl-ev", "t1", "p", "schedule", "draft", "SBL", "", "pm", "ap", "ev")
	snapshot = svc.take_ev_snapshot("ev-s", "t1", "bl-ev", "2026-04-01", 100.0, 90.0, 95.0, 200.0, "typical_performance", 211.0, 116.0)
	assert snapshot["ev"] == 90.0
	assert snapshot["ac"] == 95.0


def test_variance_threshold_breached():
	svc = _svc()
	svc.create_baseline("bl-vb", "t1", "p", "cost", "draft", "CBL", "", "pm", "ap", "ev")
	vr = svc.generate_variance_report("vr-b", "t1", "bl-vb", "Q1", -10000.0, -8000.0, 0.82, 0.85, "standard", "system")
	assert vr["threshold_breached"] is True


def test_tenant_isolation():
	svc = _svc()
	svc.create_baseline("bl-a", "tenant-a", "p1", "scope", "draft", "A", "", "pm", "ap", "ev")
	svc.create_baseline("bl-a", "tenant-b", "p1", "scope", "draft", "A", "", "pm", "ap", "ev")
	assert svc.dashboard_summary("tenant-a")["baseline_count"] == 1
	assert svc.dashboard_summary("tenant-b")["baseline_count"] == 1


def test_guardrail_unsupported_baseline_type():
	svc = _svc()
	with pytest.raises(PermissionError, match="baseline_type_not_supported"):
		svc.create_baseline("bl", "t1", "p", "financial_forecast", "draft", "Bad", "", "pm", "ap", "ev")


def test_guardrail_baseline_requires_owner():
	svc = _svc()
	with pytest.raises(PermissionError, match="baseline_owner_required"):
		svc.create_baseline("bl", "t1", "p", "cost", "draft", "X", "", "", "ap", "ev")


def test_guardrail_change_request_requires_baseline():
	svc = _svc()
	with pytest.raises(PermissionError, match="baseline_required"):
		svc.submit_change_request("cr", "t1", "nonexistent-bl", "cost_change", "medium", "T", "", "pm", "impact", "", "ev")


def test_guardrail_implement_change_requires_approval():
	svc = _svc()
	svc.create_baseline("bl-ic", "t1", "p", "cost", "draft", "C", "", "pm", "ap", "ev")
	svc.submit_change_request("cr-ic", "t1", "bl-ic", "scope_change", "low", "T", "", "pm", "impact", "", "ev")
	with pytest.raises(PermissionError, match="change_approval_required"):
		svc.implement_change("cr-ic", "t1", "")


def test_guardrail_ev_requires_approved_baseline():
	svc = _svc()
	with pytest.raises(PermissionError, match="approved_baseline_required_for_ev"):
		svc.take_ev_snapshot("ev", "t1", "missing-bl", "2026-01-01", 100.0, 90.0, 95.0, 200.0, "typical_performance", 210.0, 115.0)


def test_batch_requires_bytewax():
	svc = _svc()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 2, event_stream="redis")
