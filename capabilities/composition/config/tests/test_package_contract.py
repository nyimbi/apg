"""Configuration capability package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_composition_config"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.{name}", PACKAGE_DIR / f"{name}.py")
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_and_streaming_are_valid():
	module = _load_module("capability_contract")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "composition_config"
	assert "config_agents" in contract["provides"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert any(route["path"] == "/composition-config/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_deployment():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "deploy_configuration",
		"environment": "development",
		"impact_level": "standard",
		"approval_recorded": True,
		"canary_evidence_present": True,
		"event_stream": "other",
	})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "deployment_requires_bytewax_stream" in bad_stream["matched_rules"]


def test_service_configuration_release_lifecycle():
	service_module = _load_module("service")
	service = service_module.CompositionConfigService()

	namespace = service.register_namespace("orders-prod", "tenant-test", "Orders", "production", "owner-1", "/orders/prod", "orders")
	config = service.create_configuration(
		"database",
		"tenant-test",
		namespace["id"],
		"/orders/prod/database",
		{"pool_size": 20},
		"owner-1",
		restricted=True,
		schema={"type": "object"},
	)
	validated = service.validate_configuration(config["id"], "owner-1", "schema-check")
	active = service.activate_configuration(config["id"], "owner-1")
	deployment = service.deploy_configuration(
		"release-1",
		"tenant-test",
		config["id"],
		"production",
		"high",
		"release-manager",
		approved_by="owner-2",
		canary_evidence="canary-ok",
	)
	rollback = service.rollback_configuration(deployment["id"], "release-manager", "bad downstream signal")
	template = service.create_template(
		"orders-template",
		"tenant-test",
		"Orders Template",
		"owner-1",
		{"pool_size": 10},
		{"pool_size": {"type": "integer"}},
		shared=True,
		reviewed_by="owner-2",
	)
	drift = service.record_drift("tenant-test", config["id"], 1, 2, "medium")

	assert validated["status"] == "validated"
	assert active["status"] == "active"
	assert deployment["status"] == "deployed"
	assert rollback["status"] == "rolled_back"
	assert template["shared"] is True
	assert drift["severity"] == "medium"
	assert service.dashboard_summary("tenant-test")["audit_event_count"] >= 7


def test_service_enforces_configuration_guardrails():
	service_module = _load_module("service")
	service = service_module.CompositionConfigService()

	try:
		service.register_namespace("bad", "", "Bad", "production", "owner", "/bad", "bad")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	namespace = service.register_namespace("orders-prod", "tenant-test", "Orders", "production", "owner-1", "/orders/prod", "orders")
	try:
		service.create_configuration("restricted", "tenant-test", namespace["id"], "/orders/prod/restricted", {"x": 1}, "owner-1", restricted=True)
	except PermissionError as exc:
		assert "restricted_configuration_requires_schema" in str(exc)
	else:
		raise AssertionError("expected schema guardrail")

	try:
		service.create_configuration("secret", "tenant-test", namespace["id"], "/orders/prod/secret", {"password": "x"}, "owner-1", secret=True)
	except PermissionError as exc:
		assert "secret_configuration_requires_reference" in str(exc)
	else:
		raise AssertionError("expected secret-reference guardrail")

	config = service.create_configuration("feature", "tenant-test", namespace["id"], "/orders/prod/feature", {"enabled": True}, "owner-1")
	try:
		service.activate_configuration(config["id"], "owner-1")
	except PermissionError as exc:
		assert "activation_requires_validation" in str(exc)
	else:
		raise AssertionError("expected validation guardrail")

	service.validate_configuration(config["id"], "owner-1", "schema-check")
	service.activate_configuration(config["id"], "owner-1")
	try:
		service.deploy_configuration("bad-release", "tenant-test", config["id"], "production", "standard", "owner-1")
	except PermissionError as exc:
		assert "production_deployment_requires_approval" in str(exc)
	else:
		raise AssertionError("expected production approval guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.CompositionConfigService()
	agent = service.register_config_agent("tenant-test", "Release Review", "codex", "release_reviewer", "Review releases.")
	agent_result = service.validate_agent_config_action("tenant-test", agent["id"], "recommend_release", True, True)
	batch = service.validate_batch_configuration_change("tenant-test", 2)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"id": "api-config", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["config_agent_count"] == 1
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("configuration_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["capabilities"]["composition_config"]["streaming"]["processor"] == "bytewax"
