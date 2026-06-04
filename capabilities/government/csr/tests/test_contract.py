"""Contract shape and rule evaluation tests for APG Citizen Services Portal."""

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
	mod = _load("contract_gov_csr", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_csr"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "citizen_self_service_workflow" in contract["provides"]


def test_unauthenticated_submission_denied():
	mod = _load("rules_gov_csr", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "submit_application", "authenticated": False})
	assert result["decision"] == "deny"


def test_cross_tenant_denied():
	mod = _load("rules_gov_csr2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "submit_application", "cross_tenant": True})
	assert result["decision"] == "deny"


def test_payment_receipt_required():
	mod = _load("rules_gov_csr3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "record_payment", "application_present": True, "payment_method_supported": True, "receipt_present": False})
	assert result["decision"] == "deny"


def test_ui_routes_include_payments():
	mod = _load("ui_gov_csr", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-csr/payments" in paths
	assert "/government-csr/analytics" in paths
