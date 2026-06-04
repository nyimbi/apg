"""Service layer tests for APG Tax Administration."""

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


def test_full_tax_lifecycle():
	svc = _load("svc_tax", PACKAGE_DIR / "service.py").TaxAdministrationService()
	taxpayer = svc.register_taxpayer("reg1", "t1", "income_tax", "PIN-001", "ID-001", "John Doe", "reg-ev")
	assert taxpayer["tax_pin"] == "PIN-001"
	tax_return = svc.file_return("ret1", "t1", "annual_income", "PIN-001", "2024", 500_000, 75_000, 75_000, "ret-ev")
	assert tax_return["return_type"] == "annual_income"
	assessment = svc.raise_assessment("ass1", "t1", "ret1", "self_assessment", 75_000, "assessor-1", "2025-04-01", "ass-ev")
	assert assessment["assessed_amount"] == 75_000
	objection = svc.file_objection("obj1", "t1", "ass1", "PIN-001", "Double counting of allowances", 10_000, "obj-ev", within_deadline=True)
	assert objection["status"] == "submitted"
	audit = svc.open_audit("aud1", "t1", "PIN-001", "desk_audit", "auditor-1", "2024", "aud-ev")
	assert audit["audit_type"] == "desk_audit"
	svc.complete_audit("aud1", "t1", "Tax correctly computed, no adjustments")
	assert svc.audits[("t1", "aud1")].status == "completed"
	summary = svc.dashboard_summary("t1")
	assert summary["registration_count"] == 1
	assert summary["return_count"] == 1
	assert summary["audit_count"] == 1


def test_duplicate_pin_denied():
	svc = _load("svc_tax_dup", PACKAGE_DIR / "service.py").TaxAdministrationService()
	svc.register_taxpayer("reg1", "t1", "income_tax", "PIN-DUP", "ID-001", "John", "ev")
	with pytest.raises(PermissionError, match="duplicate_pin_denied"):
		svc.register_taxpayer("reg2", "t1", "income_tax", "PIN-DUP", "ID-002", "Jane", "ev")


def test_unsupported_tax_type_denied():
	svc = _load("svc_tax_type", PACKAGE_DIR / "service.py").TaxAdministrationService()
	with pytest.raises(PermissionError, match="tax_type_not_supported"):
		svc.register_taxpayer("reg1", "t1", "lottery_tax", "PIN-001", "ID-001", "John", "ev")


def test_objection_outside_deadline_denied():
	svc = _load("svc_tax_obj", PACKAGE_DIR / "service.py").TaxAdministrationService()
	svc.register_taxpayer("reg1", "t1", "income_tax", "PIN-001", "ID-001", "John", "ev")
	svc.file_return("ret1", "t1", "annual_income", "PIN-001", "2024", 100_000, 15_000, 15_000, "ev")
	svc.raise_assessment("ass1", "t1", "ret1", "self_assessment", 15_000, "assessor", "2025-04-01", "ev")
	with pytest.raises(PermissionError, match="objection_deadline_passed"):
		svc.file_objection("obj1", "t1", "ass1", "PIN-001", "Grounds", 5_000, "ev", within_deadline=False)


def test_debt_collection_without_demand_denied():
	svc = _load("svc_tax_debt", PACKAGE_DIR / "service.py").TaxAdministrationService()
	svc.register_taxpayer("reg1", "t1", "income_tax", "PIN-001", "ID-001", "John", "ev")
	svc.file_return("ret1", "t1", "annual_income", "PIN-001", "2024", 100_000, 15_000, 0, "ev")
	svc.raise_assessment("ass1", "t1", "ret1", "self_assessment", 15_000, "assessor", "2025-04-01", "ev")
	with pytest.raises(PermissionError, match="demand_notice_required"):
		svc.initiate_collection("col1", "t1", "PIN-001", "ass1", "payment_plan", 15_000, "", "approval-ref", "ev")


def test_return_without_pin_denied():
	svc = _load("svc_tax_pin", PACKAGE_DIR / "service.py").TaxAdministrationService()
	with pytest.raises(PermissionError, match="taxpayer_pin_required"):
		svc.file_return("ret1", "t1", "annual_income", "", "2024", 100_000, 15_000, 15_000, "ev")


def test_audit_lifecycle():
	svc = _load("svc_tax_aud", PACKAGE_DIR / "service.py").TaxAdministrationService()
	svc.register_taxpayer("reg1", "t1", "vat", "PIN-VAT", "ID-001", "Business Ltd", "ev")
	audit = svc.open_audit("aud1", "t1", "PIN-VAT", "vat_refund_audit", "auditor-1", "2024", "ev")
	assert audit["status"] == "planned"
	completed = svc.complete_audit("aud1", "t1", "Refund claim verified and valid")
	assert completed["status"] == "completed"


def test_agent_registration():
	svc = _load("svc_tax_agent", PACKAGE_DIR / "service.py").TaxAdministrationService()
	agent = svc.register_agent("ag1", "t1", "Return Processor", "codex", "return_processor", "return processing scope")
	assert agent["role"] == "return_processor"


def test_batch_requires_bytewax():
	svc = _load("svc_tax_batch", PACKAGE_DIR / "service.py").TaxAdministrationService()
	result = svc.validate_batch("t1", 50)
	assert result["processor"] == "bytewax"
	with pytest.raises(PermissionError):
		svc.validate_batch("t1", 50, event_stream="sqs")


def test_tenant_isolation():
	svc = _load("svc_tax_iso", PACKAGE_DIR / "service.py").TaxAdministrationService()
	svc.register_taxpayer("reg1", "ta", "income_tax", "PIN-A", "ID-A", "Tenant A", "ev")
	svc.register_taxpayer("reg1", "tb", "vat", "PIN-B", "ID-B", "Tenant B", "ev")
	assert svc.dashboard_summary("ta")["registration_count"] == 1
	assert svc.dashboard_summary("tb")["registration_count"] == 1
