"""ESGN package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.esgn.service import EsgnService
from capabilities.common.esgn.views import dashboard_model


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
	module = _load_module("materialized_contract_esgn", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "esgn"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_materialized_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_esgn", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "esgn" in model["capabilities"]


def test_package_runtime_compatibility_surface_creates_submission():
	service = EsgnService()

	record = service.create_record(
		record_id="rec-001",
		tenant_id="tenant-test",
		metadata={
			"template_id": "tpl-compat",
			"template_name": "Compatibility Form",
			"schema_fields": ["name"],
			"data": {"name": "Generated package"},
			"evidence_ref": "audit:rec-001",
		},
	)
	model = dashboard_model(service, "tenant-test")

	assert record["id"] == "rec-001"
	assert record["validation_status"] == "valid"
	assert model["summary"]["template_count"] == 1
	assert model["summary"]["submission_count"] == 1
	assert model["templates"][0]["status"] == "published"
