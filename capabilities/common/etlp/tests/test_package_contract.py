"""ETLP publishable package contract tests."""

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


def test_package_contract_shape_is_valid():
	module = _load_module("package_contract_etlp", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "etlp"
	assert contract["provides"] == ["pipeline_lifecycle", "data_integration_governance", "pipeline_agent_composition"]
	assert contract["requires"] == ["mdm", "meta", "mqeb", "moni"]
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert len(contract["ui"]["routes"]) >= 16
	assert len(contract["rule_engine"]["rules"]) >= 31
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_etlp", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "etlp" in model["capabilities"]
	assert model["capabilities"]["etlp"]["runtime"]["views"] == "view_models.py"
	assert model["capabilities"]["etlp"]["pipeline_lifecycle"]["execution"] == "ETLPExecutionRecord"
	assert model["capabilities"]["etlp"]["approvals"]["pipeline_agent"] == "ETLPPipelineAgentRecord"
	assert model["capabilities"]["etlp"]["streaming"]["required_processor"] == "bytewax"
	assert "codex" in model["capabilities"]["etlp"]["agents"]["pipeline_agent_contract"]["supported_runtimes"]
