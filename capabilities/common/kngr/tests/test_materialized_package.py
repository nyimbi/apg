"""Knowledge Graph package contract and runtime tests."""

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


def test_kngr_contract_shape_is_valid():
	module = _load_module("materialized_contract_kngr", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "kngr"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_kngr_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_kngr", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "kngr" in model["capabilities"]


def test_kngr_compatibility_record_uses_knowledge_runtime():
	from capabilities.common.kngr.service import KngrService

	service = KngrService()
	record = service.create_record(
		record_id="entity-compat",
		tenant_id="tenant-test",
		metadata={
			"canonical_label": "Compatibility Entity",
			"entity_type": "document",
			"owner": "steward",
			"source_uri": "manual://compat",
			"evidence_refs": ["manual:compat"],
		},
		status="curated",
	)

	assert record["canonical_label"] == "Compatibility Entity"
	assert record["curation_status"] == "curated"
	assert service.dashboard_summary("tenant-test")["source_count"] == 1
