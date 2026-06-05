"""Business Intelligence & Analytics capability integration tests.

All tests are sync; async methods called via asyncio.run().
Uses in-memory store — zero config required.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio


# ── helpers ───────────────────────────────────────────────────────────────────

def _anl():
	from capabilities.bia.anl.service import AnalyticsEngineService
	return AnalyticsEngineService(tenant_id="test-tenant", actor_id="test-actor")


def _dsh():
	from capabilities.bia.dsh.service import DashboardService
	return DashboardService()


def _pda():
	from capabilities.bia.pda.service import PredictiveAnalyticsService
	return PredictiveAnalyticsService()


# ── 1. AnalyticsEngineService instantiable ────────────────────────────────────

def test_analytics_instantiable():
	"""AnalyticsEngineService can be created with no required args."""
	svc = _anl()
	assert svc is not None
	assert svc.tenant_id == "test-tenant"
	assert svc.actor_id == "test-actor"


# ── 2. ad_hoc_query returns dict ──────────────────────────────────────────────

def test_analytics_query():
	"""ad_hoc_query returns a dict with row_count and columns keys."""
	svc = _anl()
	result = asyncio.run(svc.ad_hoc_query(
		tenant_id="test-tenant",
		sql_or_mdx="SELECT revenue, cost FROM sales",
		dataset_id="ds-001",
		actor_id="test-actor",
	))
	assert isinstance(result, dict)
	assert "row_count" in result
	assert "columns" in result
	assert result["tenant_id"] == "test-tenant"
	assert result["dataset_id"] == "ds-001"


# ── 3. DashboardService instantiable ─────────────────────────────────────────

def test_dashboard_instantiable():
	"""DashboardService can be created."""
	svc = _dsh()
	assert svc is not None


# ── 4. PredictiveAnalyticsService instantiable ────────────────────────────────

def test_forecasting_instantiable():
	"""PredictiveAnalyticsService can be created."""
	svc = _pda()
	assert svc is not None


# ── 5. BIA manifest — 8 capabilities ─────────────────────────────────────────

def test_bia_manifest():
	"""BIA domain contains exactly 8 capabilities in the manifest."""
	from capabilities.manifest import get_domain
	caps = get_domain("bia")
	assert len(caps) == 8, f"expected 8 bia capabilities, got {len(caps)}"
	ids = {c.get("capability_id") or c.get("id") for c in caps}
	assert any("anl" in str(cid) for cid in ids), f"bia_anl not found in: {ids}"


# ── 6. BIA rule evaluation ────────────────────────────────────────────────────

def test_bia_rule_evaluation():
	"""evaluate_capability_rules on bia_anl returns dict with 'decision' key."""
	from capabilities.bia.anl.capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "read",
	})
	assert isinstance(result, dict)
	assert "decision" in result
	assert result["decision"] == "allow"
