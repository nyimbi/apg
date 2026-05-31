"""HELP package contract and runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.help.api import _payload_bool
from capabilities.common.help.service import HelpService
from capabilities.common.help.views import dashboard_model


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_materialized_contract_shape_is_valid():
	module = _load_module("materialized_contract_help", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "help"
	assert contract["ui"]["routes"]
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]


def test_materialized_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_help", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "help" in model["capabilities"]
	assert model["capabilities"]["help"]["agents"]["first_class"] is True
	assert model["capabilities"]["help"]["streaming"]["lifecycle_stream"] == "help.lifecycle"


def test_help_package_compatibility_runtime_is_executable():
	service = HelpService()

	record = service.create_record(
		record_id="article-compat",
		tenant_id="tenant-test",
		metadata={
			"title": "Compatibility article",
			"body": "Compatibility records become editable help articles.",
			"owner_id": "owner-test",
			"topics": ["compatibility"],
		},
	)
	agent = service.register_help_agent(
		tenant_id="tenant-test",
		agent_id="help-agent-compat",
		name="Compatibility Steward",
		runtime="codex",
		role="knowledge_steward",
		scope=record["id"],
		owner="owner-test",
		purpose="Govern generated help compatibility evidence.",
		human_approval_required=True,
	)
	batch = service.validate_help_lifecycle_batch("tenant-test", "bytewax", 1, "help_agent_batch")
	summary = service.dashboard_summary("tenant-test")
	model = dashboard_model(service, "tenant-test")

	assert record["kind"] == "article"
	assert record["title"] == "Compatibility article"
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
	assert summary["article_count"] == 1
	assert summary["help_agent_count"] == 1
	assert summary["lifecycle_batch_count"] == 1
	assert model["summary"]["article_count"] == 1


def test_api_boolean_payloads_parse_readable_string_values():
	assert _payload_bool({"contribution_disclosed": "false"}, "contribution_disclosed", True) is False
	assert _payload_bool({"human_approval_required": "true"}, "human_approval_required", False) is True
	assert _payload_bool({}, "human_approval_required", False) is False
