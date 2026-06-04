"""Service tests for PPM Project Accounting (pac)."""

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
	mod = _load(f"svc_ppm_pac_{id(object())}", PACKAGE_DIR / "service.py")
	return mod.ProjectAccountingService()


def test_full_accounting_lifecycle():
	svc = _svc()
	account = svc.create_account("acc-1", "t1", "proj-1", "Alpha Project", "active", "USD", 100000.0, "pm-1", "evidence-a")
	cost = svc.record_cost("cost-1", "t1", account["id"], "labour", "actual_cost", 5000.0, "Dev work", "2026-Q1", "evidence-c")
	revenue = svc.recognise_revenue("rev-1", "t1", account["id"], "fixed_fee", "percentage_completion", 20000.0, "2026-Q1", "approval-r", "evidence-r")
	wip = svc.post_wip_adjustment("wip-1", "t1", account["id"], 1500.0, "Q1 WIP", "auditor-1", "evidence-w")
	invoice = svc.raise_invoice("inv-1", "t1", account["id"], "milestone", 25000.0, "milestone-1", "approval-i", "evidence-i")
	override = svc.override_budget("ov-1", "t1", account["id"], 100000.0, 120000.0, "scope change", "controller-1", "evidence-ov")
	approval = svc.record_approval("ap-1", "t1", "rev-1", "revenue", "reviewer-1", "approved", "evidence-ap")
	agent = svc.register_agent("ag-1", "t1", "Cost Bot", "codex", "cost_analyst", "cost analysis")

	assert account["status"] == "active"
	assert cost["cost_type"] == "labour"
	assert revenue["revenue_type"] == "fixed_fee"
	assert wip["auditor_id"] == "auditor-1"
	assert invoice["billing_type"] == "milestone"
	assert override["revised_budget"] == 120000.0
	assert approval["status"] == "approved"
	assert agent["runtime"] == "codex"


def test_profitability_report():
	svc = _svc()
	svc.create_account("acc-p", "t1", "proj-p", "Profit Project", "active", "USD", 50000.0, "pm-1", "ev")
	svc.record_cost("c1", "t1", "acc-p", "labour", "actual_cost", 10000.0, "", "", "ev")
	svc.record_cost("c2", "t1", "acc-p", "materials", "actual_cost", 5000.0, "", "", "ev")
	svc.recognise_revenue("r1", "t1", "acc-p", "fixed_fee", "percentage_completion", 30000.0, "", "ap", "ev")

	report = svc.profitability_report("t1", "acc-p")
	assert report["total_costs"] == 15000.0
	assert report["total_revenue"] == 30000.0
	assert report["gross_margin"] == 15000.0
	assert report["margin_pct"] == 50.0


def test_dashboard_summary_counts():
	svc = _svc()
	svc.create_account("acc-d", "t1", "proj-d", "Dash Project", "active", "USD", 10000.0, "pm", "ev")
	svc.record_cost("c-d", "t1", "acc-d", "labour", "actual_cost", 1000.0, "", "", "ev")

	summary = svc.dashboard_summary("t1")
	assert summary["account_count"] == 1
	assert summary["cost_transaction_count"] == 1
	assert summary["streaming"]["processor"] == "bytewax"


def test_tenant_isolation():
	svc = _svc()
	svc.create_account("acc-a", "tenant-a", "p1", "A", "active", "USD", 1000.0, "pm-a", "ev")
	svc.create_account("acc-a", "tenant-b", "p1", "A", "active", "EUR", 2000.0, "pm-b", "ev")

	summary_a = svc.dashboard_summary("tenant-a")
	summary_b = svc.dashboard_summary("tenant-b")
	assert summary_a["account_count"] == 1
	assert summary_b["account_count"] == 1


def test_guardrail_missing_tenant():
	svc = _svc()
	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.create_account("acc", "", "proj", "Bad", "active", "USD", 1000.0, "pm", "ev")


def test_guardrail_unsupported_cost_type():
	svc = _svc()
	svc.create_account("acc-g", "t1", "p", "G", "active", "USD", 1000.0, "pm", "ev")
	with pytest.raises(PermissionError, match="cost_type_not_supported"):
		svc.record_cost("c", "t1", "acc-g", "intergalactic_fuel", "actual_cost", 500.0, "", "", "ev")


def test_guardrail_negative_cost():
	svc = _svc()
	svc.create_account("acc-neg", "t1", "p", "Neg", "active", "USD", 1000.0, "pm", "ev")
	with pytest.raises(PermissionError, match="cost_amount_must_be_positive"):
		svc.record_cost("c", "t1", "acc-neg", "labour", "actual_cost", -100.0, "", "", "ev")


def test_guardrail_revenue_requires_approval():
	svc = _svc()
	svc.create_account("acc-r", "t1", "p", "R", "active", "USD", 1000.0, "pm", "ev")
	with pytest.raises(PermissionError, match="revenue_recognition_requires_approval"):
		svc.recognise_revenue("rev", "t1", "acc-r", "fixed_fee", "percentage_completion", 1000.0, "", "", "ev")


def test_guardrail_wip_requires_auditor():
	svc = _svc()
	svc.create_account("acc-w", "t1", "p", "W", "active", "USD", 1000.0, "pm", "ev")
	with pytest.raises(PermissionError, match="wip_adjustment_requires_auditor"):
		svc.post_wip_adjustment("wip", "t1", "acc-w", 500.0, "desc", "", "ev")


def test_guardrail_backdated_requires_justification():
	svc = _svc()
	svc.create_account("acc-bd", "t1", "p", "BD", "active", "USD", 1000.0, "pm", "ev")
	with pytest.raises(PermissionError, match="backdated_transaction_requires_justification"):
		svc.record_cost("c", "t1", "acc-bd", "labour", "actual_cost", 100.0, "", "", "ev", backdated=True, justification="")


def test_guardrail_budget_override_requires_controller():
	svc = _svc()
	svc.create_account("acc-bo", "t1", "p", "BO", "active", "USD", 1000.0, "pm", "ev")
	with pytest.raises(PermissionError, match="budget_override_requires_controller"):
		svc.override_budget("ov", "t1", "acc-bo", 1000.0, 2000.0, "reason", "", "ev")


def test_batch_requires_bytewax():
	svc = _svc()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 5, event_stream="kafka")


def test_api_layer():
	mod_api = _load("api_ppm_pac", PACKAGE_DIR / "api.py")
	account = mod_api.create_account({"tenant_id": "api-t", "account_id": "api-acc", "project_id": "proj", "name": "API Project", "budget_amount": 50000.0, "owner_id": "pm", "evidence_reference": "ev"})
	cost = mod_api.record_cost({"tenant_id": "api-t", "cost_id": "api-cost", "account_id": account["id"], "cost_type": "labour", "transaction_type": "actual_cost", "amount": 1000.0, "evidence_reference": "ev"})
	dashboard = mod_api.dashboard({"tenant_id": "api-t"})

	assert account["name"] == "API Project"
	assert cost["cost_type"] == "labour"
	assert dashboard["account_count"] == 1
