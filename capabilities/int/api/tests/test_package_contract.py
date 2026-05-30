"""Executable Integration API Management capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_int_api", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "int_api"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "api_agents" in contract["provides"]
	assert "/int-api/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_external_upstream_gap():
	module = _load_module("rules_int_api", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "api_batch",
		"event_stream": "queue",
	})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "register_api",
		"external_upstream": True,
		"review_recorded": False,
	})["matched_rules"] == ["external_upstream_requires_review"]


def test_service_executes_api_management_lifecycle():
	service_module = _load_module("service_int_api", PACKAGE_DIR / "service.py")
	service = service_module.IntApiService()

	api = service.register_api("api-1", "tenant-test", "payments", "Payments API", "/payments", "internal://payments", "owner-1")
	endpoint = service.register_endpoint("endpoint-1", "tenant-test", api["id"], "/authorize", "POST")
	policy = service.attach_policy("policy-1", "tenant-test", api["id"], "rate_limit", "Tenant rate limit", {"limit": 1000})
	consumer = service.register_consumer("consumer-1", "tenant-test", "Checkout App", "checkout@example.com", "owner-2")
	key = service.issue_api_key("key-1", "tenant-test", consumer["id"], "Checkout key", ["payments:write"], "2026-12-31")
	subscription = service.create_subscription("subscription-1", "tenant-test", consumer["id"], api["id"], "standard", "approver-1")
	approved = service.approve_api(api["id"], "tenant-test", "approver-2")
	deployment = service.deploy_api("deployment-1", "tenant-test", api["id"], "prod", "/gateway/payments", "deployer-1", "approver-3")
	usage = service.record_usage("usage-1", "tenant-test", api["id"], consumer["id"], endpoint["id"], 200, 120)
	agent = service.register_api_agent("tenant-test", "API Review Agent", "codex", "api_designer", "review API lifecycle")

	summary = service.dashboard_summary("tenant-test")
	assert endpoint["method"] == "POST"
	assert policy["policy_type"] == "rate_limit"
	assert key["key_prefix"].startswith("apg_")
	assert subscription["plan"] == "standard"
	assert approved["approved_by"] == "approver-2"
	assert deployment["environment"] == "prod"
	assert usage["latency_ms"] == 120
	assert agent["role"] == "api_designer"
	assert summary["deployment_count"] == 1
	assert summary["audit_event_count"] == 10
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_int_api", PACKAGE_DIR / "service.py")
	service = service_module.IntApiService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_api("api", "", "api", "API", "/api", "internal://api", "owner")
	with pytest.raises(PermissionError, match="api_base_path_invalid"):
		service.register_api("api", "tenant-test", "api", "API", "api", "internal://api", "owner")
	with pytest.raises(PermissionError, match="api_protocol_not_supported"):
		service.register_api("api", "tenant-test", "api", "API", "/api", "internal://api", "owner", protocol="soap")
	with pytest.raises(PermissionError, match="external_upstream_review_required"):
		service.register_api("api", "tenant-test", "api", "API", "/api", "https://example.com", "owner")

	api = service.register_api("api", "tenant-test", "api", "API", "/api", "internal://api", "owner")
	with pytest.raises(PermissionError, match="endpoint_method_not_supported"):
		service.register_endpoint("endpoint", "tenant-test", api["id"], "/x", "TRACE")
	with pytest.raises(PermissionError, match="policy_config_required"):
		service.attach_policy("policy", "tenant-test", api["id"], "rate_limit", "Policy", {})
	with pytest.raises(PermissionError, match="consumer_email_invalid"):
		service.register_consumer("consumer", "tenant-test", "Consumer", "bad-email", "owner")
	consumer = service.register_consumer("consumer", "tenant-test", "Consumer", "consumer@example.com", "owner")
	with pytest.raises(PermissionError, match="api_key_scope_required"):
		service.issue_api_key("key", "tenant-test", consumer["id"], "Key", [], "2026-12-31")
	with pytest.raises(PermissionError, match="api_approver_required"):
		service.approve_api(api["id"], "tenant-test", "")
	with pytest.raises(PermissionError, match="deployer_required"):
		service.deploy_api("deployment", "tenant-test", api["id"], "stage", "/gateway/api", "")
	with pytest.raises(PermissionError, match="production_deployment_approval_required"):
		service.deploy_api("deployment", "tenant-test", api["id"], "prod", "/gateway/api", "deployer")
	with pytest.raises(PermissionError, match="slow_request_review_required"):
		service.record_usage("usage", "tenant-test", api["id"], consumer["id"], None, 200, 2500)


def test_agents_batch_api_views_and_app_are_executable():
	api_module = _load_module("api_int_api", PACKAGE_DIR / "api.py")
	views = _load_module("views_int_api", PACKAGE_DIR / "views.py")
	app = _load_module("app_int_api", PACKAGE_DIR / "app.py")

	api_record = api_module.create_record({"tenant_id": "tenant-api", "id": "api-record"})
	agent = api_module.register_api_agent({
		"tenant_id": "tenant-api",
		"name": "Security Reviewer",
		"runtime": "claude_code",
		"role": "security_reviewer",
	})
	batch = api_module.service().validate_batch("tenant-api", 2)
	model = views.api_registry_model(api_module.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert api_record["id"] == "api-record"
	assert agent["role"] == "security_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["title"] == "API Record"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["int_api"]["screens"]["agents"]["route"] == "/int-api/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_int_api", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["int_api"]["streaming"]["processor"] == "bytewax"
