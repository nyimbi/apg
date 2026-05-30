"""CONS package runtime and publish contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_cons_contract_shape_is_valid():
	module = _load_module("cons_contract", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "cons"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["privacy_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_cons_app_entrypoint_is_publishable():
	module = _load_module("cons_app", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "cons" in model["capabilities"]
	assert model["capabilities"]["cons"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["cons"]["screens"]["agents"]["route"] == "/cons/agents"
