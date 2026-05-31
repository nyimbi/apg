"""ONTO package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib
import importlib.util
import json
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


def test_onto_contract_shape_is_valid():
	module = _load_module("package_contract_onto", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "onto"
	assert len(contract["ui"]["routes"]) >= 15
	assert len(contract["rule_engine"]["rules"]) >= 55
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_onto_app_entrypoint_is_publishable():
	module = _load_module("package_app_onto", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "onto" in model["capabilities"]
	assert model["capabilities"]["onto"]["agents"]["first_class"] is True
	assert model["capabilities"]["onto"]["streaming"]["required_processor"] == "bytewax"
	assert model["capabilities"]["onto"]["ontology_lifecycle"]["lifecycle_batch"] == "OntoLifecycleBatchRecord"
	assert len(model["capabilities"]["onto"]["ui"]["routes"]) >= 15


def test_onto_package_evidence_matches_entrypoint():
	module = _load_module("package_evidence_app_onto", PACKAGE_DIR / "app.py")
	semantic_json = (PACKAGE_DIR / "semantic_model.json").read_text()

	assert (PACKAGE_DIR / "README.md").exists()
	assert (PACKAGE_DIR / "SPECIFICATION.md").exists()
	assert (PACKAGE_DIR / "PLAN.md").exists()
	assert module.semantic_model() == json.loads(semantic_json)


def test_onto_api_imports_without_production_dependencies():
	module = importlib.import_module("capabilities.common.onto.api")

	status = module.capability_status("tenant-test")

	assert status["capability"] == "onto"
	assert status["rule_count"] >= 55


def test_onto_compatibility_record_uses_ontology_registry():
	module = importlib.import_module("capabilities.common.onto.service")
	service = module.OntoService()

	record = service.create_record(
		record_id="legacy-ontology",
		tenant_id="tenant-test",
		metadata={"name": "Legacy Ontology", "owner": "legacy-owner", "domain": "legacy"},
	)

	assert record["id"] == "legacy-ontology"
	assert record["name"] == "Legacy Ontology"
	assert record["owner"] == "legacy-owner"
	assert service.list_records("tenant-test")[0]["domain"] == "legacy"
