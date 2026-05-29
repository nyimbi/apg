"""REGY capability package contract and publish tests."""

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


def test_registry_contract_shape_is_valid():
	module = _load_module("registry_contract_regy", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "regy"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]
	assert contract["configuration"]["registration"]["health_endpoint_required"] is True
	assert contract["rule_engine"]["type"] == "deterministic"


def test_registry_app_entrypoint_is_publishable():
	module = _load_module("registry_app_regy", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "regy" in model["capabilities"]
	assert "service_registration_requires_health_endpoint" in model["rules"]
	assert model["capabilities"]["regy"]["theme"]["name"] == "regy_service_catalog"
