"""Package contract tests for FEDL."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.fedl.service import FedlService
from capabilities.common.fedl.views import dashboard_model


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_package_contract_shape_is_valid():
	module = _load_module("package_contract_fedl", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fedl"
	assert len(contract["ui"]["routes"]) >= 12
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_fedl", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "fedl" in model["capabilities"]
	assert model["capabilities"]["fedl"]["runtime"]["service"] == "service.FedlService"
	assert model["capabilities"]["fedl"]["streaming"]["engine"] == "bytewax"


def test_package_runtime_compatibility_surface_creates_federation():
	service = FedlService()

	record = service.create_record(
		record_id="fed-compat",
		tenant_id="tenant-test",
		metadata={
			"name": "Compatibility Federation",
			"coordinator": "compat-coordinator",
			"model_family": "tabular",
			"objective_metric": "auc",
			"data_residency_regions": ["ke"],
		},
	)
	model = dashboard_model(service, "tenant-test")

	assert record["id"] == "fed-compat"
	assert record["status"] == "active"
	assert model["summary"]["federation_count"] == 1
	assert model["federations"][0]["coordinator"] == "compat-coordinator"
