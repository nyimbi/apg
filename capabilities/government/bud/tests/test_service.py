"""Service layer tests for APG Budget Management."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	"""Load module by path, always (re)registering deps under bare names for fallback imports."""
	# Always overwrite bare-name slots so this capability's deps win even in a multi-cap test run
	for dep in ('capability_contract', 'models'):
		dep_path = PACKAGE_DIR / f"{dep}.py"
		if dep_path.exists():
			dep_spec = importlib.util.spec_from_file_location(f"{name}__{dep}", dep_path)
			dep_mod = importlib.util.module_from_spec(dep_spec)
			sys.modules[f"{name}__{dep}"] = dep_mod
			sys.modules[dep] = dep_mod  # overwrite bare name each time
			dep_spec.loader.exec_module(dep_mod)
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_full_budget_lifecycle():
	svc = _load("svc_bud", PACKAGE_DIR / "service.py").BudgetManagementService()
	budget = svc.record_budget("b1", "t1", "recurrent", "exchequer", "v1", 1_000_000, "2025/26", "approver-1", "evidence-ref")
	assert budget["budget_type"] == "recurrent"
	vote = svc.record_vote("v1", "t1", "VOTE-001", "programme", "b1", 500_000, "vote-evidence")
	assert vote["allocated_amount"] == 500_000
	commitment = svc.record_commitment("c1", "t1", "v1", "lpo", 100_000, "approval-ref", "supplier-1", "comm-evidence")
	assert commitment["amount"] == 100_000
	vote_obj = svc.votes[("t1", "v1")]
	assert vote_obj.available_balance == 400_000
	expenditure = svc.record_expenditure("e1", "t1", "c1", "goods_services", 100_000, "exp-approval", "payee-1", "exp-evidence")
	assert expenditure["expenditure_type"] == "goods_services"
	report = svc.generate_report("r1", "t1", "b1", "budget_outturn", "q1", "author-1", "report-evidence")
	assert report["report_type"] == "budget_outturn"
	summary = svc.dashboard_summary("t1")
	assert summary["budget_count"] == 1
	assert summary["commitment_count"] == 1


def test_tenant_isolation():
	svc = _load("svc_bud_iso", PACKAGE_DIR / "service.py").BudgetManagementService()
	svc.record_budget("b1", "tenant-a", "recurrent", "exchequer", "v1", 100_000, "2025/26", "approver", "ev")
	svc.record_budget("b1", "tenant-b", "development", "donor_grant", "v1", 200_000, "2025/26", "approver", "ev")
	assert svc.dashboard_summary("tenant-a")["budget_count"] == 1
	assert svc.dashboard_summary("tenant-b")["budget_count"] == 1
	assert svc.budgets[("tenant-a", "b1")].budget_type == "recurrent"
	assert svc.budgets[("tenant-b", "b1")].budget_type == "development"


def test_missing_tenant_context_denied():
	svc = _load("svc_bud_auth", PACKAGE_DIR / "service.py").BudgetManagementService()
	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.record_budget("b1", "", "recurrent", "exchequer", "v1", 100, "2025", "approver", "ev")


def test_unsupported_budget_type_denied():
	svc = _load("svc_bud_type", PACKAGE_DIR / "service.py").BudgetManagementService()
	with pytest.raises(PermissionError, match="budget_type_not_supported"):
		svc.record_budget("b1", "t1", "unknown_type", "exchequer", "v1", 100, "2025", "approver", "ev")


def test_commitment_without_balance_denied():
	svc = _load("svc_bud_bal", PACKAGE_DIR / "service.py").BudgetManagementService()
	svc.record_budget("b1", "t1", "recurrent", "exchequer", "v1", 50_000, "2025", "approver", "ev")
	svc.record_vote("v1", "t1", "V001", "programme", "b1", 50_000, "ev")
	with pytest.raises(PermissionError, match="insufficient_vote_balance"):
		svc.record_commitment("c1", "t1", "v1", "lpo", 100_000, "approval", "supplier", "ev")


def test_expenditure_without_commitment_denied():
	svc = _load("svc_bud_exp", PACKAGE_DIR / "service.py").BudgetManagementService()
	with pytest.raises(PermissionError, match="commitment_required"):
		svc.record_expenditure("e1", "t1", "missing-c", "goods_services", 100, "approval", "payee", "ev")


def test_revision_without_treasury_denied():
	svc = _load("svc_bud_rev", PACKAGE_DIR / "service.py").BudgetManagementService()
	svc.record_budget("b1", "t1", "recurrent", "exchequer", "v1", 50_000, "2025", "approver", "ev")
	with pytest.raises(PermissionError, match="treasury_notification_required"):
		svc.record_revision("r1", "t1", "b1", "reallocation", 5000, "approval", "", "ev")


def test_review_requires_reviewer():
	svc = _load("svc_bud_rev2", PACKAGE_DIR / "service.py").BudgetManagementService()
	with pytest.raises(PermissionError, match="reviewer_required"):
		svc.record_review("rev1", "t1", "ref1", "", "approved", "ev")


def test_agent_registration_and_action():
	svc = _load("svc_bud_agent", PACKAGE_DIR / "service.py").BudgetManagementService()
	agent = svc.register_agent("ag1", "t1", "Budget Bot", "codex", "budget_analyst", "read-only analytics")
	assert agent["runtime"] == "codex"
	result = svc.validate_agent_action("t1", privileged_scope=False, human_approval_recorded=False)
	assert result["accepted"] is True


def test_privileged_agent_action_requires_human_approval():
	svc = _load("svc_bud_priv", PACKAGE_DIR / "service.py").BudgetManagementService()
	with pytest.raises(PermissionError, match="human_approval_required"):
		svc.validate_agent_action("t1", privileged_scope=True, human_approval_recorded=False)


def test_bytewax_batch_validation():
	svc = _load("svc_bud_batch", PACKAGE_DIR / "service.py").BudgetManagementService()
	result = svc.validate_batch("t1", 5)
	assert result["processor"] == "bytewax"
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 5, event_stream="kafka")
