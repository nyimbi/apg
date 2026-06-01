"""Executable Banking APIs capability package tests."""

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
	module = _load_module("contract_fintech_apis", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_apis"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "api_call_audit_workflow" in contract["provides"]
	assert "/fintech-apis/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_volume_call():
	module = _load_module("rules_fintech_apis", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "apis_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_api_call", "high_volume": True, "human_approval_recorded": False})["decision"] == "require_review"


def test_service_executes_banking_api_lifecycle():
	service_module = _load_module("service_fintech_apis", PACKAGE_DIR / "service.py")
	service = service_module.BankingAPIService()

	product = service.register_api_product("product-1", "tenant-test", "Accounts API", "api-ops", "accounts", "sandbox", ["accounts.read", "balances.read"])
	developer = service.onboard_developer("developer-1", "tenant-test", "Fintech Builder", "kyb-1", "security-1", "risk-1")
	application = service.register_application("app-1", "tenant-test", developer["id"], "PFM App", "sandbox", "https://example.test/callback", "terms-1")
	consent = service.create_consent_grant("consent-1", "tenant-test", application["id"], "customer-1", ["accounts.read"], "2026-12-31")
	client = service.issue_api_client("client-1", "tenant-test", application["id"], "oauth2_auth_code", "key-1", ["accounts.read"])
	endpoint = service.publish_endpoint_policy("endpoint-1", "tenant-test", product["id"], "/accounts", "accounts.read", "throttle-1", "risk-policy-1")
	webhook = service.subscribe_webhook("webhook-1", "tenant-test", application["id"], "account_updated", "https://example.test/webhook", "secret-1")
	bucket = service.update_rate_limit("bucket-1", "tenant-test", client["id"], 100)
	call = service.record_api_call("call-1", "tenant-test", client["id"], product["id"], endpoint["id"], 200, 5, "risk-call-1")
	incident = service.open_sla_incident("incident-1", "tenant-test", "medium", "owner-1", ["trace-1"])
	agent = service.register_api_agent("agent-1", "tenant-test", "API Agent", "codex", "api_ops_reviewer", "review API operations")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert product["product_type"] == "accounts"
	assert consent["scopes"] == ["accounts.read"]
	assert client["public_client_id"].startswith("cli_")
	assert endpoint["route"] == "/accounts"
	assert webhook["event_type"] == "account_updated"
	assert bucket["remaining"] == 100
	assert call["status_code"] == 200
	assert service.rate_limits[client["id"]].remaining == 95
	assert incident["severity"] == "medium"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["call_count"] == 1
	assert summary["audit_event_count"] == 11


def test_service_guardrails_reject_invalid_banking_api_actions():
	service_module = _load_module("guardrail_service_fintech_apis", PACKAGE_DIR / "service.py")
	service = service_module.BankingAPIService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_api_product("product", "", "API", "owner", "accounts", "sandbox", ["accounts.read"])
	with pytest.raises(PermissionError, match="product_owner_required"):
		service.register_api_product("product", "tenant-test", "API", "", "accounts", "sandbox", ["accounts.read"])
	product = service.register_api_product("product-ok", "tenant-test", "API", "owner", "accounts", "sandbox", ["accounts.read"])
	with pytest.raises(PermissionError, match="developer_kyb_required"):
		service.onboard_developer("developer", "tenant-test", "Dev", "", "security", "risk")
	developer = service.onboard_developer("developer-ok", "tenant-test", "Dev", "kyb", "security", "risk")
	with pytest.raises(PermissionError, match="redirect_uri_required"):
		service.register_application("app", "tenant-test", developer["id"], "App", "sandbox", "", "terms")
	application = service.register_application("app-ok", "tenant-test", developer["id"], "App", "sandbox", "https://example.test/callback", "terms")
	with pytest.raises(PermissionError, match="consent_expiry_required"):
		service.create_consent_grant("consent", "tenant-test", application["id"], "customer", ["accounts.read"], "")
	with pytest.raises(PermissionError, match="client_key_reference_required"):
		service.issue_api_client("client", "tenant-test", application["id"], "oauth2_auth_code", "", ["accounts.read"])
	with pytest.raises(PermissionError, match="client_scopes_not_consented"):
		service.issue_api_client("client-unconsented", "tenant-test", application["id"], "oauth2_auth_code", "key", ["payments.write"])
	service.create_consent_grant("consent-ok", "tenant-test", application["id"], "customer", ["accounts.read"], "2026-12-31")
	client = service.issue_api_client("client-ok", "tenant-test", application["id"], "oauth2_auth_code", "key", ["accounts.read"])
	with pytest.raises(PermissionError, match="endpoint_throttle_required"):
		service.publish_endpoint_policy("endpoint", "tenant-test", product["id"], "/accounts", "accounts.read", "", "risk")
	endpoint = service.publish_endpoint_policy("endpoint-ok", "tenant-test", product["id"], "/accounts", "accounts.read", "throttle", "risk")
	other_product = service.register_api_product("product-other", "tenant-test", "Payments API", "owner", "payments", "sandbox", ["payments.write"])
	other_endpoint = service.publish_endpoint_policy("endpoint-other", "tenant-test", other_product["id"], "/payments", "payments.write", "throttle", "risk")
	with pytest.raises(PermissionError, match="webhook_signing_secret_required"):
		service.subscribe_webhook("webhook", "tenant-test", application["id"], "account_updated", "https://example.test/webhook", "")
	service.update_rate_limit("bucket", "tenant-test", client["id"], 1)
	with pytest.raises(PermissionError, match="api_call_endpoint_product_mismatch"):
		service.record_api_call("call-mismatch", "tenant-test", client["id"], product["id"], other_endpoint["id"], 200, 1, "risk")
	with pytest.raises(PermissionError, match="api_rate_limit_exceeded"):
		service.record_api_call("call", "tenant-test", client["id"], product["id"], endpoint["id"], 200, 2, "risk")
	with pytest.raises(PermissionError, match="incident_approval_required"):
		service.open_sla_incident("incident", "tenant-test", "critical", "owner", ["evidence"])
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="api_agent_runtime_not_supported"):
		service.register_api_agent("agent", "tenant-test", "Bad Agent", "unsupported", "api_ops_reviewer", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_apis", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_apis", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_apis", PACKAGE_DIR / "app.py")

	product = api.register_api_product({"tenant_id": "tenant-api", "product_id": "api-product", "name": "Accounts API", "owner_id": "owner", "product_type": "accounts", "environment": "sandbox", "scopes": ["accounts.read"]})
	developer = api.onboard_developer({"tenant_id": "tenant-api", "developer_id": "api-developer", "name": "Dev", "kyb_reference": "kyb", "security_review_reference": "security", "risk_clearance_reference": "risk"})
	application = api.register_application({"tenant_id": "tenant-api", "application_id": "api-app", "developer_id": developer["id"], "name": "App", "environment": "sandbox", "redirect_uri": "https://example.test/callback", "terms_reference": "terms"})
	api.create_consent_grant({"tenant_id": "tenant-api", "consent_id": "api-consent", "application_id": application["id"], "customer_reference": "customer", "scopes": ["accounts.read"], "expiry_date": "2026-12-31"})
	client = api.issue_api_client({"tenant_id": "tenant-api", "client_id": "api-client", "application_id": application["id"], "auth_flow": "client_credentials", "key_reference": "key", "scopes": ["accounts.read"]})
	endpoint = api.publish_endpoint_policy({"tenant_id": "tenant-api", "endpoint_id": "api-endpoint", "product_id": product["id"], "route": "/accounts", "required_scope": "accounts.read", "throttle_policy_reference": "throttle", "risk_policy_reference": "risk"})
	api.record_api_call({"tenant_id": "tenant-api", "call_id": "api-call", "client_id": client["id"], "product_id": product["id"], "endpoint_id": endpoint["id"], "status_code": 200, "call_count": 1, "risk_reference": "risk-call"})
	agent = api.register_api_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "API Agent", "runtime": "claude_code", "role": "webhook_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.apis_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "webhook_reviewer"
	assert dashboard["summary"]["call_count"] == 1
	assert console["calls"][0]["id"] == "api-call"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_apis"]["screens"]["agents"]["route"] == "/fintech-apis/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_apis", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_apis"]["streaming"]["processor"] == "bytewax"
