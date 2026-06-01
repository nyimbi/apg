"""DVRL publishable package contract tests."""

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
	module = _load_module("package_contract_dvrl", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "dvrl"
	assert len(contract["ui"]["routes"]) >= 14
	assert len(contract["rule_engine"]["rules"]) >= 28
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "virtualization_agent_composition" in contract["provides"]
	assert "review_evidence" in contract["provides"]
	assert "virtualization_agents" in contract["review_evidence"]["pending_queues"]
	assert "policy_decision" in contract["review_evidence"]["policy_fields"]
	assert contract["requires"] == ["mdm", "etlp", "meta"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_dvrl", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "dvrl" in model["capabilities"]
	assert model["capabilities"]["dvrl"]["runtime"]["views"] == "view_models.py"
	assert model["capabilities"]["dvrl"]["virtualization_lifecycle"]["query"] == "DVRLQueryRecord"
	assert model["capabilities"]["dvrl"]["virtualization_lifecycle"]["virtualization_agent"] == "DVRLVirtualizationAgentRecord"
	assert model["capabilities"]["dvrl"]["streaming"]["required_processor"] == "bytewax"
	assert "virtualization_agents" in model["capabilities"]["dvrl"]["review_evidence"]["pending_queues"]
	assert model["contracts"]["dvrl"]["review_evidence"]["deny_behavior"].startswith("Denied DVRL")
	assert model["agents"]["dvrl_virtualization_agents"]["first_class"] is True
