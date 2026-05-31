"""Package contract tests for MDM."""

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
	module = _load_module("package_contract_mdm", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "mdm"
	assert contract["display_name"] == "Master Data Management"
	assert contract["provides"] == ["master_data_governance", "golden_record_lifecycle", "data_agent_composition"]
	assert contract["requires"] == ["auth", "audl", "conf", "mten"]
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert len(contract["rule_engine"]["rules"]) >= 23
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"entities",
		"golden_records",
		"quality",
		"duplicates",
		"stewardship",
		"lineage",
		"cross_references",
		"publish",
		"audit",
		"adapters",
		"agents",
		"lifecycle",
	}
	assert contract["theme"]["tokens"]["border.radius"]


def test_package_app_entrypoint_is_publishable():
	module = _load_module("package_app_mdm", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "mdm" in model["capabilities"]
	assert model["capabilities"]["mdm"]["runtime"]["views"] == "view_models.py"
	assert model["capabilities"]["mdm"]["approvals"]["duplicate_review"] == "MdmDuplicateCandidateRecord"
	assert model["capabilities"]["mdm"]["approvals"]["golden_record_merge"] == "MdmMergeRequestRecord"
	assert model["capabilities"]["mdm"]["approvals"]["data_agent"] == "MdmDataAgentRecord"
	assert model["capabilities"]["mdm"]["streaming"]["required_processor"] == "bytewax"
	assert "codex" in model["capabilities"]["mdm"]["agents"]["data_agent_contract"]["supported_runtimes"]
	assert model["capabilities"]["mdm"]["adapters"]["event_stream"] == "bytewax"
