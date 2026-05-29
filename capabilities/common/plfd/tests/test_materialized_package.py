"""Platform Foundation package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib
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


def test_plfd_contract_shape_is_valid():
	module = _load_module("plfd_contract", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "plfd"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_plfd_app_entrypoint_is_publishable():
	module = _load_module("plfd_app", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "plfd" in model["capabilities"]


def test_plfd_compatibility_record_uses_foundation_registry():
	module = importlib.import_module("capabilities.common.plfd.service")
	service = module.PlfdService()

	record = service.create_record("compat-service", "tenant-test", {"owner": "platform-owner", "tier": "shared"})

	assert record["id"] == "compat-service"
	assert record["owner"] == "platform-owner"
	assert service.list_records("tenant-test")[0]["id"] == "compat-service"
