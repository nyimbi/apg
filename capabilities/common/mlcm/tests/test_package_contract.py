"""Package contract tests for MLCM."""

from __future__ import annotations

from pathlib import Path
import importlib
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


def test_package_contract_shape_is_valid():
	module = _load_module("package_contract_mlcm", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "mlcm"
	assert len(contract["ui"]["routes"]) >= 15
	assert len(contract["rule_engine"]["rules"]) >= 38
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_mlcm", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "mlcm" in model["capabilities"]
	assert model["capabilities"]["mlcm"]["runtime"]["service"] == "service.MlcmService"
	assert model["capabilities"]["mlcm"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["mlcm"]["streaming"]["required_processor"] == "bytewax"
	assert model["capabilities"]["mlcm"]["agents"]["first_class"] is True
	assert model["capabilities"]["mlcm"]["review_evidence"]["durable_status"] == "pending_review"
	assert model["capabilities"]["mlcm"]["model_lifecycle"]["model_lifecycle_agent"] == "ModelLifecycleAgentRecord"
	assert model["composition"]["capability_dependencies"]["mlcm"] == ["aicr", "moni", "audl"]


def test_mlcm_compatibility_record_uses_model_registry():
	module = importlib.import_module("capabilities.common.mlcm.service")
	service = module.MlcmService()

	record = service.create_record(
		record_id="legacy-model",
		tenant_id="tenant-test",
		metadata={"name": "Legacy Model", "owner": "legacy-owner", "problem_type": "forecasting"},
	)

	assert record["id"] == "legacy-model"
	assert record["name"] == "Legacy Model"
	assert record["owner"] == "legacy-owner"
	assert service.list_records("tenant-test")[0]["problem_type"] == "forecasting"
