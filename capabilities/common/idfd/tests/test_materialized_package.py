"""IDFD package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.idfd.service import IdfdService
from capabilities.common.idfd.views import dashboard_model


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_idfd_contract_shape_is_valid():
	module = _load_module("materialized_contract_idfd", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "idfd"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_idfd_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_idfd", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "idfd" in model["capabilities"]


def test_idfd_compatibility_record_uses_provider_runtime():
	service = IdfdService()

	record = service.create_record(
		"provider-compat",
		"tenant-test",
		metadata={
			"name": "Compatibility OIDC",
			"protocol": "oidc",
			"redirect_allowlist": ["https://app.example/callback"],
		},
	)
	dashboard = dashboard_model(service, "tenant-test")

	assert record["id"] == "provider-compat"
	assert record["protocol"] == "oidc"
	assert dashboard["summary"]["provider_count"] == 1
	assert dashboard["summary"]["theme"] == "idfd_federation_console"
