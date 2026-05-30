"""Package contract tests for NLPC."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.nlpc.nlpc_runtime import NlpcService
from capabilities.common.nlpc.view_models import dashboard_model


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
	module = _load_module("package_contract_nlpc", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "nlpc"
	assert len(contract["ui"]["routes"]) >= 12
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "nlpc_runtime.NlpcService"
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_nlpc", PACKAGE_DIR / "app.py")

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
	assert "nlpc" in model["capabilities"]
	assert model["capabilities"]["nlpc"]["runtime"]["service"] == "nlpc_runtime.NlpcService"
	assert model["capabilities"]["nlpc"]["streaming"]["engine"] == "bytewax"
	assert committed_model == model
	assert set(committed_manifest["generated_artifacts"]) >= {
		"README.md",
		"SPECIFICATION.md",
		"PLAN.md",
		"capability_contract.py",
		"nlpc_runtime.py",
		"view_models.py",
		"app.py",
	}
	assert committed_report["ok"] is True
	assert committed_report["evidence"]["contracts"]["capability_contract"]["route_count"] >= 12
	assert committed_report["evidence"]["contracts"]["capability_contract"]["rule_count"] >= 30
	assert committed_report["evidence"]["runtime"]["event_stream"] == "bytewax"
	assert committed_report["evidence"]["runtime"]["generated_app_runtime"] == "nlpc_runtime.NlpcService"


def test_package_runtime_compatibility_surface_creates_document():
	service = NlpcService()

	record = service.create_record(
		record_id="doc-compat",
		tenant_id="tenant-test",
		metadata={
			"content": "Habari from the compatibility surface.",
			"language": "auto",
			"source_ref": "compat://document",
		},
	)
	model = dashboard_model(service, "tenant-test")

	assert record["id"] == "doc-compat"
	assert record["status"] == "active"
	assert record["language"] == "sw"
	assert model["summary"]["document_count"] == 1
	assert model["recent_documents"][0]["source_ref"] == "compat://document"
