"""Contract and service tests for PPM Project Accounting (pac)."""

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


def test_contract_shape_and_required_keys():
	mod = _load("contract_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")

	assert contract["capability"] == "ppm_pac"
	assert contract["version"] == "1.0.0"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "6px"
	assert "tenant_id" in contract["configuration_schema"]["required"]
	assert "ui" in contract["configuration_schema"]["required"]
	assert "theme" in contract["configuration_schema"]["required"]
	assert len(contract["ui"]["routes"]) >= 8
	assert len(contract["rule_engine"]["rules"]) >= 20
	assert contract["rule_engine"]["default_decision"] == "allow"
	assert "project_cost_tracking" in contract["provides"]
	assert "auth" in contract["requires"]
	assert "audl" in contract["requires"]
	assert "comp" in contract["requires"]


def test_rule_engine_tenant_context_required():
	mod = _load("rules_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any("tenant_context_required" in a["reason"] for a in result["actions"])


def test_rule_engine_write_requires_policy():
	mod = _load("rules2_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"


def test_rule_engine_cost_batch_requires_bytewax():
	mod = _load("rules3_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "cost_batch", "event_stream": "queue",
	})
	assert result["decision"] == "deny"
	assert any("bytewax" in a["reason"] for a in result["actions"])


def test_rule_engine_cross_tenant_denied():
	mod = _load("rules4_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"cross_tenant_access": True})
	assert result["decision"] == "deny"


def test_rule_engine_negative_revenue_denied():
	mod = _load("rules5_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "recognise_revenue", "amount_positive": False,
	})
	assert result["decision"] == "deny"


def test_rule_engine_privileged_agent_action():
	mod = _load("rules6_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False,
	})
	assert result["decision"] == "deny"
	assert any("human_approval_required" in a["reason"] for a in result["actions"])


def test_rule_engine_allow_on_clean_context():
	mod = _load("rules7_ppm_pac", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})
	assert result["decision"] == "allow"
