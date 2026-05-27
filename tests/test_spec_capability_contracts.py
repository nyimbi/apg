"""Spec-backed capability contract coverage tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


CAPABILITIES_ROOT = Path(__file__).resolve().parents[1] / "capabilities"
SPEC_FILES = sorted(CAPABILITIES_ROOT.glob("*/*/cap_spec.md"))


def _load_contract(path: Path):
	module_name = "apg_spec_contract_" + "_".join(path.parent.parts[-2:])
	spec = importlib.util.spec_from_file_location(module_name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def test_all_spec_backed_capabilities_have_executable_contracts():
	assert len(SPEC_FILES) >= 49

	missing = [
		str(spec_path.parent)
		for spec_path in SPEC_FILES
		if not (spec_path.parent / "capability_contract.py").exists()
	]
	assert missing == []

	for spec_path in SPEC_FILES:
		module = _load_contract(spec_path.parent / "capability_contract.py")
		contract = module.get_capability_contract()
		assert isinstance(contract["configuration"], dict), spec_path
		assert isinstance(contract["configuration_schema"], dict), spec_path
		assert contract["rule_engine"]["type"] == "deterministic", spec_path
		assert contract["rule_engine"]["rules"], spec_path
		assert contract["ui"]["requires_theme"] is True, spec_path
		assert contract["ui"]["routes"], spec_path
		assert contract["theme"]["tokens"], spec_path
