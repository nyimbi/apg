"""Contract shape and rule evaluation tests for APG Licensing & Permits."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from capabilities.capability_contract_registry import validate_contract_shape

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_contract_shape_is_valid():
	mod = _load("contract_gov_lic", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_lic"
	assert "licence_application_workflow" in contract["provides"]
	assert "licence_revocation_workflow" in contract["provides"]


def test_renewal_blocked_by_failed_inspection():
	mod = _load("rules_gov_lic", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "renew_licence", "last_inspection_failed": True})
	assert result["decision"] == "deny"


def test_duplicate_licence_denied():
	mod = _load("rules_gov_lic2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "issue_licence", "approved_application_present": True, "licence_number_present": True, "expiry_date_present": True, "duplicate_detected": True})
	assert result["decision"] == "deny"


def test_revocation_notice_required():
	mod = _load("rules_gov_lic3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "revoke_licence", "reason_present": True, "approval_present": True, "notice_served": False})
	assert result["decision"] == "deny"


def test_ui_routes_include_revocations():
	mod = _load("ui_gov_lic", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-lic/revocations" in paths
	assert "/government-lic/compliance" in paths
