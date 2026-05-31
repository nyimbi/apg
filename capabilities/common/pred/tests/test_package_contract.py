"""Predictive Analytics package contract and runtime tests."""

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


def test_pred_contract_shape_is_valid_and_publishable():
	module = _load_module("package_contract_pred", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "pred"
	assert len(contract["rule_engine"]["rules"]) >= 39
	assert len(contract["ui"]["routes"]) >= 14
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.PredService"
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_pred_app_entrypoint_is_publishable():
	module = _load_module("package_app_pred", PACKAGE_DIR / "app.py")

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
	assert "pred" in model["capabilities"]
	assert model["capabilities"]["pred"]["runtime"]["service"] == "service.PredService"
	assert model["capabilities"]["pred"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["pred"]["streaming"]["required_processor"] == "bytewax"
	assert model["capabilities"]["pred"]["agents"]["first_class"] is True
	assert model["composition"]["capability_dependencies"]["pred"] == ["aicr", "mlcm", "etlp", "conf"]
	assert model["composition"]["agent_teams"]["pred_forecast_governance"]["stream"] == "pred.lifecycle"
	assert "drift" in model["capabilities"]["pred"]["screens"]
	assert committed_model == model
	assert set(committed_manifest["generated_artifacts"]) >= {
		"README.md",
		"SPECIFICATION.md",
		"PLAN.md",
		"capability_contract.py",
		"models.py",
		"predictive_runtime.py",
		"service.py",
		"views.py",
		"app.py",
	}
	assert committed_report["ok"] is True
	assert committed_report["evidence"]["contracts"]["capability_contract"]["route_count"] >= 14
	assert committed_report["evidence"]["contracts"]["capability_contract"]["rule_count"] >= 39
	assert committed_report["evidence"]["runtime"]["event_stream"] == "bytewax"
	assert committed_report["evidence"]["runtime"]["generated_app_runtime"] == "service.PredService"
	assert committed_report["evidence"]["agents"]["first_class"] is True
	assert committed_report["evidence"]["streaming"]["required_processor"] == "bytewax"


def test_pred_compatibility_record_uses_predictive_model_registry():
	service_module = importlib.import_module("capabilities.common.pred.service")
	service = service_module.PredService()

	record = service.create_record(
		record_id="compat-pred",
		tenant_id="tenant-test",
		metadata={
			"owner": "tester",
			"algorithm": "linear",
			"target": "compat",
			"training_history_points": 24,
			"feature_names": ["compat_signal"],
		},
	)

	assert record["id"] == "compat-pred"
	assert record["owner"] == "tester"
	assert service.list_records("tenant-test")[0]["target"] == "compat"
