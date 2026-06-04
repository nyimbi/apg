"""Service layer tests for APG Government Contracts & Procurement."""

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


def test_full_procurement_lifecycle():
	svc = _load("svc_con", PACKAGE_DIR / "service.py").ProcurementService()
	tender = svc.publish_tender("t1", "ten1", "open_tender", "large", "IT Equipment", "Procurement of laptops", "approver-1", "tender-ev", "", "published")
	assert tender["procurement_method"] == "open_tender"
	eval_ = svc.record_evaluation("ev1", "ten1", "t1", "bidder-1", "combined_score", 85.0, "evaluator-1", "eval-ev")
	assert eval_["score"] == 85.0
	award = svc.record_award("aw1", "ten1", "t1", "bidder-1", 500_000, "PPDA-2025-001", "award-ev")
	assert award["awarded_to"] == "bidder-1"
	contract = svc.record_contract("co1", "ten1", "aw1", "supply", 500_000, "2025-01-01", "2025-12-31", "ps-secretary", "bidder-1", "contract-ev", "signed")
	assert contract["contract_type"] == "supply"
	performance = svc.record_performance("pf1", "ten1", "co1", "on_track", "reviewer-1", "Q1-2025", "All milestones met", "perf-ev")
	assert performance["performance_status"] == "on_track"
	summary = svc.dashboard_summary("ten1")
	assert summary["tender_count"] == 1
	assert summary["contract_count"] == 1


def test_debarred_bidder_blocked():
	svc = _load("svc_con_debar", PACKAGE_DIR / "service.py").ProcurementService()
	svc.debar_bidder("db1", "t1", "bad-bidder", "fraud", "2030-01-01", "ev")
	svc.publish_tender("t1", "t1", "open_tender", "small", "Title", "Desc", "approver", "ev", "")
	with pytest.raises(PermissionError, match="debarred_bidder_denied"):
		svc.record_evaluation("ev1", "t1", "t1", "bad-bidder", "price", 70.0, "evaluator", "ev")


def test_award_without_evaluation_denied():
	svc = _load("svc_con_award", PACKAGE_DIR / "service.py").ProcurementService()
	svc.publish_tender("t1", "t1", "open_tender", "small", "T", "D", "approver", "ev", "")
	with pytest.raises(PermissionError, match="approved_evaluation_required"):
		svc.record_award("aw1", "t1", "t1", "bidder", 100_000, "ppda-ref", "ev")


def test_direct_procurement_requires_justification():
	svc = _load("svc_con_direct", PACKAGE_DIR / "service.py").ProcurementService()
	with pytest.raises(PermissionError, match="single_source_justification_required"):
		svc.publish_tender("t1", "t1", "direct_procurement", "micro", "T", "D", "approver", "ev", "")


def test_contract_variation_requires_approval():
	svc = _load("svc_con_var", PACKAGE_DIR / "service.py").ProcurementService()
	svc.publish_tender("t1", "t1", "open_tender", "small", "T", "D", "approver", "ev", "")
	svc.record_evaluation("ev1", "t1", "t1", "b1", "price", 90.0, "evaluator", "ev")
	svc.record_award("aw1", "t1", "t1", "b1", 100_000, "ppda-ref", "ev")
	svc.record_contract("co1", "t1", "aw1", "supply", 100_000, "2025-01-01", "2025-12-31", "signed-by", "b1", "ev")
	with pytest.raises(PermissionError, match="variation_approval_required"):
		svc.record_variation("v1", "t1", "co1", "scope_change", "Scope expanded", 20_000, "", "ppda-ref", "ev")


def test_unsupported_procurement_method_denied():
	svc = _load("svc_con_method", PACKAGE_DIR / "service.py").ProcurementService()
	with pytest.raises(PermissionError, match="procurement_method_not_supported"):
		svc.publish_tender("t1", "t1", "bribery", "small", "T", "D", "approver", "ev", "")


def test_batch_requires_bytewax():
	svc = _load("svc_con_batch", PACKAGE_DIR / "service.py").ProcurementService()
	result = svc.validate_batch("t1", 2)
	assert result["processor"] == "bytewax"


def test_agent_registration():
	svc = _load("svc_con_agent", PACKAGE_DIR / "service.py").ProcurementService()
	agent = svc.register_agent("ag1", "t1", "Tender Analyst", "codex", "tender_analyst", "tender review scope")
	assert agent["role"] == "tender_analyst"


def test_tenant_isolation():
	svc = _load("svc_con_iso", PACKAGE_DIR / "service.py").ProcurementService()
	svc.publish_tender("t1", "ta", "open_tender", "small", "T-A", "D", "approver", "ev", "")
	svc.publish_tender("t1", "tb", "restricted_tender", "medium", "T-B", "D", "approver", "ev", "")
	assert svc.dashboard_summary("ta")["tender_count"] == 1
	assert svc.dashboard_summary("tb")["tender_count"] == 1
