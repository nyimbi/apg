"""GEOS package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.geos.service import GeosService
from capabilities.common.geos.views import dashboard_model


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
	module = _load_module("materialized_contract_geos", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "geos"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_materialized_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_geos", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "geos" in model["capabilities"]


def test_package_runtime_compatibility_surface_creates_geofence():
	service = GeosService()

	record = service.create_record(
		record_id="geo-compat",
		tenant_id="tenant-test",
		metadata={
			"name": "Compatibility Geofence",
			"owner": "geo-admin",
			"boundary": {
				"type": "circle",
				"center": {"latitude": -1.286389, "longitude": 36.817223},
				"radius_meters": 1000,
			},
		},
	)
	model = dashboard_model(service, "tenant-test")

	assert record["id"] == "geo-compat"
	assert record["status"] == "active"
	assert model["summary"]["geofence_count"] == 1
	assert model["geofences"][0]["owner"] == "geo-admin"
