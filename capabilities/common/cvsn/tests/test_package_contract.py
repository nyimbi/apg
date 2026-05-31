"""Package contract tests for CVSN."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.cvsn.cvsn_runtime import CvsnService
from capabilities.common.cvsn.view_models import dashboard_model


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
	module = _load_module("package_contract_cvsn", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "cvsn"
	assert len(contract["ui"]["routes"]) >= 15
	assert len(contract["rule_engine"]["rules"]) >= 38
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "cvsn_runtime.CvsnService"
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_cvsn", PACKAGE_DIR / "app.py")

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
	assert "cvsn" in model["capabilities"]
	assert model["capabilities"]["cvsn"]["runtime"]["service"] == "cvsn_runtime.CvsnService"
	assert model["capabilities"]["cvsn"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["cvsn"]["streaming"]["required_processor"] == "bytewax"
	assert model["capabilities"]["cvsn"]["agents"]["first_class"] is True
	assert model["composition"]["capability_dependencies"]["cvsn"] == ["aicr", "mlcm", "conf", "auth"]
	assert model["composition"]["agent_teams"]["cvsn_visual_governance"]["stream"] == "cvsn.lifecycle"
	assert committed_model == model
	assert set(committed_manifest["generated_artifacts"]) >= {
		"README.md",
		"SPECIFICATION.md",
		"PLAN.md",
		"capability_contract.py",
		"cvsn_runtime.py",
		"view_models.py",
		"app.py",
	}
	assert committed_report["ok"] is True
	assert committed_report["evidence"]["contracts"]["capability_contract"]["route_count"] >= 15
	assert committed_report["evidence"]["contracts"]["capability_contract"]["rule_count"] >= 38
	assert committed_report["evidence"]["runtime"]["event_stream"] == "bytewax"
	assert committed_report["evidence"]["runtime"]["generated_app_runtime"] == "cvsn_runtime.CvsnService"
	assert committed_report["evidence"]["agents"]["first_class"] is True
	assert committed_report["evidence"]["streaming"]["required_processor"] == "bytewax"


def test_package_runtime_compatibility_surface_creates_asset():
	service = CvsnService()

	record = service.create_record(
		record_id="asset-compat",
		tenant_id="tenant-test",
		metadata={
			"asset_kind": "image",
			"mime_type": "image/png",
			"file_size_mb": 1,
			"source_ref": "compat://image",
		},
	)
	model = dashboard_model(service, "tenant-test")

	assert record["id"] == "asset-compat"
	assert record["status"] == "active"
	assert record["content_hash"]
	assert model["summary"]["asset_count"] == 1
	assert model["summary"]["vision_agent_count"] == 0
	assert model["recent_assets"][0]["source_ref"] == "compat://image"
