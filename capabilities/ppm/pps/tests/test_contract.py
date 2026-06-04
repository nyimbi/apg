"""Contract tests for PPM Project Planning & Scheduling (pps)."""

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
	mod = _load("contract_ppm_pps", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")

	assert contract["capability"] == "ppm_pps"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "6px"
	assert len(contract["ui"]["routes"]) >= 8
	assert len(contract["rule_engine"]["rules"]) >= 20
	assert "wbs_creation_and_management" in contract["provides"]
	assert "critical_path_analysis" in contract["provides"]
	assert "schd" in contract["requires"]


def test_rule_tenant_required():
	mod = _load("rules_ppm_pps", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"


def test_rule_circular_dependency_denied():
	mod = _load("rules2_ppm_pps", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"circular_dependency": True})
	assert result["decision"] == "deny"
	assert any("circular" in a["reason"] for a in result["actions"])


def test_rule_critical_path_manipulation_denied():
	mod = _load("rules3_ppm_pps", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"critical_path_manipulation": True})
	assert result["decision"] == "deny"


def test_rule_retroactive_edit_requires_change_request():
	mod = _load("rules4_ppm_pps", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"operation": "edit_task", "retroactive": True})
	assert result["decision"] == "deny"


def test_rule_batch_requires_bytewax():
	mod = _load("rules5_ppm_pps", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "schedule_batch", "event_stream": "kafka",
	})
	assert result["decision"] == "deny"


def test_rule_allow_clean():
	mod = _load("rules6_ppm_pps", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"
