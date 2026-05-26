"""Common capability contract shape regression tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


CONTRACT_ROOT = Path(__file__).resolve().parent
CONTRACT_FILES = sorted(CONTRACT_ROOT.glob("*/capability_contract.py"))


def _load_contract_module(path: Path):
	module_name = f"apg_common_contract_{path.parent.name}"
	spec = importlib.util.spec_from_file_location(module_name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def test_all_common_capabilities_expose_executable_contract_shape():
	assert len(CONTRACT_FILES) >= 80

	for contract_file in CONTRACT_FILES:
		module = _load_contract_module(contract_file)
		assert hasattr(module, "get_capability_contract"), contract_file

		contract = module.get_capability_contract()
		assert isinstance(contract["configuration"], dict), contract_file
		assert isinstance(contract["configuration_schema"], dict), contract_file

		rule_engine = contract["rule_engine"]
		assert rule_engine["type"] == "deterministic", contract_file
		assert isinstance(rule_engine["rules"], list), contract_file
		assert rule_engine["rules"], contract_file

		ui = contract["ui"]
		assert ui["requires_theme"] is True, contract_file
		assert isinstance(ui["routes"], list), contract_file
		assert ui["routes"], contract_file

		theme = contract["theme"]
		assert isinstance(theme["tokens"], dict), contract_file
		assert theme["tokens"], contract_file
