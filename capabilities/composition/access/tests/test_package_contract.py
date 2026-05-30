"""Access capability package contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "tested_composition_access"


def _load_module(name: str):
	if PACKAGE_NAME not in sys.modules:
		package = types.ModuleType(PACKAGE_NAME)
		package.__path__ = [str(PACKAGE_DIR)]
		sys.modules[PACKAGE_NAME] = package
	spec = importlib.util.spec_from_file_location(
		f"{PACKAGE_NAME}.{name}",
		PACKAGE_DIR / f"{name}.py",
	)
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
	assert contract["capability"] == "composition_access"
	assert "access_agents" in contract["provides"]
	assert contract["requires"] == ["auth", "audl", "ntfy", "conf", "registry"]
	assert contract["configuration"]["tenant_id"] == "tenant-test"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["stream"] == "apg.composition.access.lifecycle"
	assert any(route["path"] == "/composition-access/agents" for route in contract["ui"]["routes"])


def test_rule_engine_blocks_missing_context_and_non_bytewax_decisions():
	module = _load_module("capability_contract")

	missing_context = module.evaluate_capability_rules({"tenant_context_present": False})
	bad_stream = module.evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "record_decision",
		"event_stream": "other",
	})

	assert missing_context["decision"] == "deny"
	assert "tenant_context_required" in missing_context["matched_rules"]
	assert bad_stream["decision"] == "deny"
	assert "decision_requires_bytewax_stream" in bad_stream["matched_rules"]


def test_service_provider_resource_policy_grant_decision_lifecycle():
	service_module = _load_module("service")
	service = service_module.CompositionAccessService()

	provider = service.register_provider("corp", "tenant-test", "Corporate OIDC", "oidc", "owner-1")
	activated = service.activate_provider(
		provider["id"],
		"owner-1",
		secret_reference="vault://tenant-test/oidc",
		test_evidence="discovery_checked",
	)
	resource = service.register_resource(
		"orders.approve",
		"tenant-test",
		"Approve Orders",
		"orders-owner",
		["read", "approve"],
		"orders",
		sensitive=True,
	)
	policy = service.create_policy(
		"orders-policy",
		"tenant-test",
		"Orders policy",
		resource["id"],
		"owner-1",
		"allow",
		conditions={"mfa": True},
		risk_level="high",
	)
	active_policy = service.activate_policy(
		policy["id"],
		"owner-1",
		simulation_evidence="simulation-1",
		reviewed_by="risk-reviewer",
	)
	grant = service.create_grant(
		"grant-1",
		"tenant-test",
		"user-1",
		resource["id"],
		["approve"],
		"manager-1",
		"coverage",
		privileged=True,
		approved_by="security-1",
		expires_at="2026-06-30T23:59:59+00:00",
	)
	session = service.evaluate_session(
		"session-1",
		"tenant-test",
		"user-1",
		provider["id"],
		risk_score=91,
		step_up_completed=True,
	)
	decision = service.record_decision(
		"decision-1",
		"tenant-test",
		"user-1",
		resource["id"],
		"approve",
		"allow",
		"active_grant",
		[policy["id"]],
		event_stream="bytewax",
	)

	assert activated["status"] == "active"
	assert active_policy["status"] == "active"
	assert grant["status"] == "active"
	assert session["status"] == "verified"
	assert decision["event_stream"] == "bytewax"
	assert service.dashboard_summary("tenant-test")["audit_event_count"] >= 7


def test_service_enforces_access_guardrails():
	service_module = _load_module("service")
	service = service_module.CompositionAccessService()

	try:
		service.register_provider("bad", "", "Bad", "oidc", "owner")
	except PermissionError as exc:
		assert "tenant_context_required" in str(exc)
	else:
		raise AssertionError("expected tenant guardrail")

	provider = service.register_provider("corp", "tenant-test", "Corporate OIDC", "oidc", "owner-1")
	try:
		service.activate_provider(provider["id"], "owner-1", test_evidence="checked")
	except PermissionError as exc:
		assert "provider_requires_secret_reference" in str(exc)
	else:
		raise AssertionError("expected secret-reference guardrail")

	resource = service.register_resource(
		"orders.approve",
		"tenant-test",
		"Approve Orders",
		"orders-owner",
		["approve"],
		"orders",
		sensitive=True,
	)
	try:
		service.create_policy("bad-policy", "tenant-test", "Bad", resource["id"], "owner-1", "allow")
	except PermissionError as exc:
		assert "sensitive_policy_requires_conditions" in str(exc)
	else:
		raise AssertionError("expected policy-condition guardrail")

	try:
		service.create_grant(
			"bad-grant",
			"tenant-test",
			"user-1",
			resource["id"],
			["approve"],
			"manager-1",
			"",
			privileged=True,
		)
	except PermissionError as exc:
		assert "privileged_grant_requires_approval" in str(exc)
	else:
		raise AssertionError("expected privileged-grant guardrail")

	try:
		service.evaluate_session("bad-session", "tenant-test", "user-1", provider["id"], 90)
	except PermissionError as exc:
		assert "high_risk_session_requires_step_up" in str(exc)
	else:
		raise AssertionError("expected session guardrail")

	try:
		service.record_decision(
			"bad-decision",
			"tenant-test",
			"user-1",
			resource["id"],
			"approve",
			"allow",
			"bad_stream",
			event_stream="other",
		)
	except PermissionError as exc:
		assert "decision_requires_bytewax_stream" in str(exc)
	else:
		raise AssertionError("expected stream guardrail")


def test_agents_batch_api_views_and_app_are_executable():
	service_module = _load_module("service")
	api_module = _load_module("api")
	views_module = _load_module("views")
	app_module = _load_module("app")

	service = service_module.CompositionAccessService()
	agent = service.register_access_agent(
		"tenant-test",
		"Grant Review",
		"codex",
		"grant_reviewer",
		"Review privileged grants.",
	)
	agent_result = service.validate_agent_access_action(
		"tenant-test",
		agent["id"],
		"recommend_grant",
		privileged_scope=True,
		human_approval_recorded=True,
	)
	batch = service.validate_batch_grant("tenant-test", 2)
	dashboard = views_module.dashboard_model(service, "tenant-test")
	agent_view = views_module.agent_workbench_model(service, "tenant-test")
	api_record = api_module.create_record({"id": "api-resource", "tenant_id": "tenant-api"})
	status = api_module.capability_status("tenant-api")
	self_test = app_module.self_test()
	model = app_module.semantic_model()

	assert agent_result["decision"] == "allow"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["access_agent_count"] == 1
	assert agent_view["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert api_record["id"].startswith("resource_")
	assert status["streaming"]["processor"] == "bytewax"
	assert self_test["passed"] is True
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["composition_access"]["streaming"]["processor"] == "bytewax"
