"""RCM package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.grc.rcm.service import GrcRcmService
from capabilities.grc.rcm.views import dashboard_model


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_materialized_contract_shape_is_valid():
	module = _load_module("materialized_contract_grc_rcm", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "grc_rcm"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_materialized_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_grc_rcm", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "grc_rcm" in model["capabilities"]


def test_rcm_package_compatibility_runtime_is_executable():
	service = GrcRcmService()
	record = service.create_record(
		record_id="risk-compat",
		tenant_id="tenant-test",
		metadata={
			"title": "Generated package compatibility risk",
			"owner_id": "owner-test",
			"probability": 0.5,
			"impact": 0.5,
		},
	)
	model = dashboard_model(service, "tenant-test")

	assert record["kind"] == "risk"
	assert record["risk_level"] == "low"
	assert model["summary"]["risk_count"] == 1
	assert model["panels"][0]["id"] == "risk_heatmap"
