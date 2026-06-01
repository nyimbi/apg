"""Package contract tests for CONN."""

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
	module = _load_module("package_contract_conn", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "conn"
	assert len(contract["ui"]["routes"]) >= 14
	assert len(contract["rule_engine"]["rules"]) >= 39
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "conn_runtime.ConnService"
	assert contract["theme"]["tokens"]["border.radius"]
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["provides"] == ["connector_management", "connection_orchestration", "connector_agent_composition", "review_evidence"]
	assert "connector_agents" in contract["review_evidence"]["pending_queues"]
	assert "policy_decision" in contract["review_evidence"]["policy_fields"]
	assert contract["requires"] == ["apig", "auth", "encr", "audl"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_conn", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "conn" in model["capabilities"]
	assert model["capabilities"]["conn"]["runtime"]["service"] == "conn_runtime.py"
	assert model["capabilities"]["conn"]["runtime"]["views"] == "view_models.py"
	assert model["capabilities"]["conn"]["connector_lifecycle"]["connection"] == "ConnectionRecord"
	assert model["capabilities"]["conn"]["connector_lifecycle"]["connector_agent"] == "ConnectorAgentRecord"
	assert model["capabilities"]["conn"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["conn"]["streaming"]["required_processor"] == "bytewax"
	assert model["capabilities"]["conn"]["agents"]["first_class"] is True
	assert model["capabilities"]["conn"]["review_evidence"]["deny_behavior"]
	assert model["contracts"]["conn"]["review_evidence"]["pending_queues"]
