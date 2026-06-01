"""Package contract tests for HLTH."""

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
	module = _load_module("package_contract_hlth", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "hlth"
	assert contract["display_name"] == "Health Checks and Diagnostics"
	assert contract["provides"] == ["health_governance", "diagnostic_lifecycle", "health_agent_composition", "review_evidence"]
	assert contract["review_evidence"]["pending_queues"]
	assert contract["requires"] == ["moni", "mqeb", "conf"]
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert len(contract["rule_engine"]["rules"]) >= 26
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"components",
		"checks",
		"baselines",
		"predictions",
		"deployment_gates",
		"audit",
		"adapters",
		"agents",
		"lifecycle",
	}
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_hlth", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "hlth" in model["capabilities"]
	assert model["capabilities"]["hlth"]["runtime"]["views"] == "view_models.py"
	assert model["capabilities"]["hlth"]["approvals"]["remediation"] == "HlthRemediationRequestRecord"
	assert model["capabilities"]["hlth"]["approvals"]["deployment_gate"] == "HlthDeploymentGateRecord"
	assert model["capabilities"]["hlth"]["approvals"]["health_agent"] == "HlthAgentRecord"
	assert model["capabilities"]["hlth"]["streaming"]["required_processor"] == "bytewax"
	assert "review_evidence" in model["capabilities"]["hlth"]["provides"]
	assert model["capabilities"]["hlth"]["review_evidence"]["pending_queues"]
	assert "codex" in model["capabilities"]["hlth"]["agents"]["health_agent_contract"]["supported_runtimes"]
	assert "moni" in model["capabilities"]["hlth"]["adapters"]["supported_probe_sources"]
