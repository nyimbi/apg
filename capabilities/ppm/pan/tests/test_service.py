"""Service tests for PPM Portfolio Analytics (pan)."""

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
	mod = _load(f"svc_ppm_pan_{id(object())}", PACKAGE_DIR / "service.py")
	return mod.PortfolioAnalyticsService()


def test_full_portfolio_analytics_lifecycle():
	svc = _svc()
	portfolio = svc.create_portfolio("pf-1", "t1", "Innovation Portfolio", "active", "internal", "cpo-1", "approval-pf", "evidence-pf")
	alignment = svc.score_alignment("al-1", "t1", portfolio["id"], "strategic_fit", "weighted_criteria", 8.5, "Strong alignment with digital strategy", "evidence-al")
	risk = svc.analyse_risk_return("rr-1", "t1", portfolio["id"], "market", "npv", 6.0, 1500000.0, "2026-Q1", "evidence-rr")
	heat_map = svc.generate_heat_map("hm-1", "t1", portfolio["id"], "skill_category", "2026-Q1", '{"data":[]}', "system")
	perf = svc.snapshot_performance("ps-1", "t1", portfolio["id"], "current_quarter", '{"kpi":"on_time"}', "target", 0.85, 0.82)
	scenario = svc.run_scenario("sc-1", "t1", portfolio["id"], "Optimistic Growth", '{}', '{}', "analyst-1", "evidence-sc")
	report = svc.generate_report("rp-1", "t1", portfolio["id"], "executive_summary", "dashboard", "system", "{}")
	agent = svc.register_agent("ag-1", "t1", "Portfolio Bot", "claude_code", "portfolio_analyst", "portfolio analysis")

	assert portfolio["status"] == "active"
	assert alignment["score"] == 8.5
	assert risk["risk_category"] == "market"
	assert heat_map["dimension"] == "skill_category"
	assert perf["period"] == "current_quarter"
	assert scenario["analyst_id"] == "analyst-1"
	assert report["dashboard_type"] == "executive_summary"
	assert agent["role"] == "portfolio_analyst"


def test_dashboard_summary():
	svc = _svc()
	svc.create_portfolio("pf-d", "t1", "Dash", "active", "internal", "o", "ap", "ev")
	summary = svc.dashboard_summary("t1")
	assert summary["portfolio_count"] == 1
	assert summary["streaming"]["processor"] == "bytewax"


def test_tenant_isolation():
	svc = _svc()
	svc.create_portfolio("pf-a", "tenant-a", "A", "active", "internal", "o", "ap", "ev")
	svc.create_portfolio("pf-a", "tenant-b", "A", "active", "internal", "o", "ap", "ev")
	assert svc.dashboard_summary("tenant-a")["portfolio_count"] == 1
	assert svc.dashboard_summary("tenant-b")["portfolio_count"] == 1


def test_guardrail_missing_tenant():
	svc = _svc()
	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.create_portfolio("pf", "", "Bad", "active", "internal", "o", "ap", "ev")


def test_guardrail_portfolio_requires_owner():
	svc = _svc()
	with pytest.raises(PermissionError, match="portfolio_owner_required"):
		svc.create_portfolio("pf", "t1", "X", "active", "internal", "", "ap", "ev")


def test_guardrail_unsupported_status():
	svc = _svc()
	with pytest.raises(PermissionError, match="portfolio_status_not_supported"):
		svc.create_portfolio("pf", "t1", "X", "unknown_status", "internal", "o", "ap", "ev")


def test_guardrail_alignment_requires_portfolio():
	svc = _svc()
	with pytest.raises(PermissionError, match="portfolio_required"):
		svc.score_alignment("al", "t1", "nonexistent-pf", "strategic_fit", "weighted_criteria", 7.0, "rationale", "ev")


def test_guardrail_unsupported_risk_category():
	svc = _svc()
	svc.create_portfolio("pf-r", "t1", "R", "active", "internal", "o", "ap", "ev")
	with pytest.raises(PermissionError, match="risk_category_not_supported"):
		svc.analyse_risk_return("rr", "t1", "pf-r", "cosmic_risk", "npv", 5.0, 1000.0, "", "ev")


def test_guardrail_batch_requires_bytewax():
	svc = _svc()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 3, event_stream="kafka")


def test_api_layer():
	mod = _load("api_ppm_pan", PACKAGE_DIR / "api.py")
	pf = mod.create_portfolio({"tenant_id": "api-t", "portfolio_id": "api-pf", "name": "API Portfolio", "owner_id": "cpo", "approval_reference": "ap", "evidence_reference": "ev"})
	dashboard = mod.dashboard({"tenant_id": "api-t"})
	assert pf["name"] == "API Portfolio"
	assert dashboard["portfolio_count"] == 1
