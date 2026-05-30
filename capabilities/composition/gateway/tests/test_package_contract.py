"""Gateway capability package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_composition_gateway"


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
	assert contract["capability"] == "composition_gateway"
	assert "gateway_agents" in contract["provides"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert any(route["path"] == "/composition-gateway/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_route():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_route",
		"public_route": False,
		"route_policy_attached": True,
		"approval_recorded": True,
		"tls_enabled": True,
		"event_stream": "other",
	})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "route_requires_bytewax_stream" in bad_stream["matched_rules"]


def test_service_route_policy_traffic_certificate_lifecycle():
	service_module = _load_module("service")
	service = service_module.CompositionGatewayService()

	mesh_service = service.register_service(
		"orders",
		"tenant-test",
		"Orders API",
		"owner-1",
		[{"host": "orders.internal", "port": 8080, "protocol": "http"}],
		"/health",
		"orders",
		public_service=True,
	)
	policy = service.attach_policy("orders-policy", "tenant-test", mesh_service["id"], True, True, "owner-1")
	route = service.create_route(
		"orders-public",
		"tenant-test",
		mesh_service["id"],
		"/orders",
		["GET", "POST"],
		public_route=True,
		policy_id=policy["id"],
		approved_by="security-1",
		tls_enabled=True,
	)
	traffic = service.shift_traffic(
		"orders-canary",
		"tenant-test",
		route["id"],
		{"stable": 90, "canary": 10},
		"release-1",
		canary_shift=True,
		canary_evidence="canary-ok",
	)
	certificate = service.register_certificate(
		"orders-cert",
		"tenant-test",
		"orders.example.com",
		"security-1",
		"vault://orders/cert",
		"2026-12-31T23:59:59+00:00",
	)
	health = service.record_health("tenant-test", mesh_service["id"], "healthy", 18)

	assert route["status"] == "active"
	assert traffic["event_stream"] == "bytewax"
	assert certificate["status"] == "active"
	assert health["health_status"] == "healthy"
	assert service.dashboard_summary("tenant-test")["audit_event_count"] >= 5


def test_service_enforces_gateway_guardrails():
	service_module = _load_module("service")
	service = service_module.CompositionGatewayService()

	try:
		service.register_service("bad", "", "Bad", "owner", [{"host": "x"}], "/health", "bad")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	mesh_service = service.register_service("orders", "tenant-test", "Orders", "owner", [{"host": "x"}], "/health", "orders", public_service=True)
	try:
		service.create_route("bad-route", "tenant-test", mesh_service["id"], "/orders", ["GET"], public_route=True)
	except PermissionError as exc:
		assert "public_route_requires_policy" in str(exc)
	else:
		raise AssertionError("expected public route guardrail")

	try:
		service.attach_policy("bad-policy", "tenant-test", mesh_service["id"], False, True, "owner")
	except PermissionError as exc:
		assert "public_service_requires_rate_limit" in str(exc)
	else:
		raise AssertionError("expected rate-limit guardrail")

	try:
		service.register_certificate("bad-cert", "tenant-test", "orders.example.com", "owner", "", "2026-12-31T23:59:59+00:00")
	except PermissionError as exc:
		assert "certificate_requires_secret_reference" in str(exc)
	else:
		raise AssertionError("expected certificate guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.CompositionGatewayService()
	agent = service.register_gateway_agent("tenant-test", "Route Review", "codex", "route_reviewer", "Review routes.")
	agent_result = service.validate_agent_gateway_action("tenant-test", agent["id"], "recommend_route", True, True)
	batch = service.validate_batch_route_change("tenant-test", 2)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"id": "api-service", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["gateway_agent_count"] == 1
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("gateway_service_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["capabilities"]["composition_gateway"]["streaming"]["processor"] == "bytewax"
