"""Knowledge Graph package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
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


def test_kngr_contract_shape_is_valid():
	module = _load_module("package_contract_kngr", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "kngr"
	assert len(contract["ui"]["routes"]) >= 14
	assert len(contract["rule_engine"]["rules"]) >= 45
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_kngr_app_entrypoint_is_publishable():
	module = _load_module("package_app_kngr", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "kngr" in model["capabilities"]
	assert model["capabilities"]["kngr"]["agents"]["first_class"] is True
	assert model["capabilities"]["kngr"]["streaming"]["required_processor"] == "bytewax"
	assert model["capabilities"]["kngr"]["review_queues"]["relationships"] == "KnowledgeRelationship.status == pending_review"
	assert model["capabilities"]["kngr"]["knowledge_lifecycle"]["lifecycle_batch"] == "KngrLifecycleBatchRecord"
	assert model["composition"]["capability_dependencies"]["kngr"] == ["grph", "nlpc", "meta", "srch", "onto", "aicr", "conf"]
	assert len(model["capabilities"]["kngr"]["ui"]["routes"]) >= 14


def test_kngr_package_evidence_matches_entrypoint():
	module = _load_module("package_evidence_app_kngr", PACKAGE_DIR / "app.py")
	semantic_json = (PACKAGE_DIR / "semantic_model.json").read_text()

	assert (PACKAGE_DIR / "README.md").exists()
	assert (PACKAGE_DIR / "SPECIFICATION.md").exists()
	assert (PACKAGE_DIR / "PLAN.md").exists()
	assert module.semantic_model() == json.loads(semantic_json)
	report = json.loads((PACKAGE_DIR / "release_report.json").read_text())
	assert report["ok"] is True
	assert report["evidence"]["agents"]["first_class"] is True
	assert report["evidence"]["streaming"]["required_processor"] == "bytewax"


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
