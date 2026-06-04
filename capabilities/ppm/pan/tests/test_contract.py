"""Contract tests for PPM Portfolio Analytics (pan)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
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


def test_contract_shape():
	mod = _load("contract_ppm_pan", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")

	assert contract["capability"] == "ppm_pan"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "tenant_id" in contract["configuration_schema"]["required"]
	assert len(contract["ui"]["routes"]) >= 8
	assert len(contract["rule_engine"]["rules"]) >= 20
	assert "portfolio_performance_dashboard" in contract["provides"]
	assert "auth" in contract["requires"]
	assert "nlpc" in contract["requires"]


def test_rule_tenant_context_required():
	mod = _load("rules_ppm_pan", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"


def test_rule_cross_tenant_denied():
	mod = _load("rules2_ppm_pan", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"cross_tenant_access": True})["decision"] == "deny"


def test_rule_classification_downgrade_denied():
	mod = _load("rules3_ppm_pan", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"operation": "update_portfolio", "classification_downgrade": True})
	assert result["decision"] == "deny"


def test_rule_analytics_batch_requires_bytewax():
	mod = _load("rules4_ppm_pan", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "analytics_batch", "event_stream": "rabbitmq",
	})
	assert result["decision"] == "deny"


def test_rule_scenario_analyst_required():
	mod = _load("rules5_ppm_pan", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "run_scenario", "analyst_present": False,
	})
	assert result["decision"] == "deny"


def test_rule_allow_clean_context():
	mod = _load("rules6_ppm_pan", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"
