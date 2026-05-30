"""SRCH package contract and deterministic runtime tests."""

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


def test_contract_shape_is_valid():
	module = _load_module("package_contract_srch", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "srch"
	assert len(contract["ui"]["routes"]) >= 12
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.SrchService"
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_srch", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text())
	committed_manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
	committed_report = json.loads((PACKAGE_DIR / "release_report.json").read_text())

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "srch" in model["capabilities"]
	assert model["capabilities"]["srch"]["runtime"]["service"] == "service.SrchService"
	assert model["capabilities"]["srch"]["streaming"]["engine"] == "bytewax"
	assert "audit" in model["capabilities"]["srch"]["screens"]
	assert committed_model == model
	for generated_doc in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / generated_doc).is_file()
		assert generated_doc in committed_manifest["generated_artifacts"]
	assert set(committed_manifest["generated_artifacts"]) >= {
		"README.md",
		"SPECIFICATION.md",
		"PLAN.md",
		"capability_contract.py",
		"search_runtime.py",
		"service.py",
		"views.py",
		"app.py",
	}
	assert committed_report["ok"] is True
	assert committed_report["evidence"]["contracts"]["capability_contract"]["route_count"] >= 12
	assert committed_report["evidence"]["contracts"]["capability_contract"]["rule_count"] >= 30
	assert committed_report["evidence"]["runtime"]["event_stream"] == "bytewax"
	assert committed_report["evidence"]["runtime"]["generated_app_runtime"] == "service.SrchService"


def test_package_runtime_compatibility_surface_creates_index():
	service_module = importlib.import_module("capabilities.common.srch.service")
	service = service_module.SrchService()

	record = service.create_record(
		record_id="compat-index",
		tenant_id="tenant-test",
		metadata={
			"owner": "tester",
			"content_type": "document",
			"classification": "internal",
			"source_lineage_ref": "lineage://compat",
		},
	)

	assert record["name"] == "compat-index"
	assert record["owner"] == "tester"
	assert service.list_records("tenant-test")[0]["classification"] == "internal"
