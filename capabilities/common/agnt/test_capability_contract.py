"""Regression coverage for the AGNT executable capability contract."""

from agents import DEFAULT_AGENT_INTEGRATIONS
import pytest

from capabilities.common.agnt import api, register_capability, views
from capabilities.common.agnt.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.agnt.service import AgntService


def test_contract_exposes_agent_runtimes_rules_ui_and_theme():
	contract = get_capability_contract("tenant-agnt", {"runtimes": {"default_runtime": "codex"}})

	assert contract["capability"] == "agnt"
	assert contract["configuration"]["tenant_id"] == "tenant-agnt"
	assert contract["configuration"]["runtimes"]["default_runtime"] == "codex"
	assert set(contract["configuration"]["runtimes"]["registered"]) >= {"local", "codex", "claude_code", "opencode", "pi"}
	assert set(DEFAULT_AGENT_INTEGRATIONS.names()) >= {"local", "codex", "claude_code", "opencode", "pi"}
	assert contract["configuration_schema"]["required"] == ["tenant_id", "agents", "teams", "runtimes", "memory", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "agents", "teams", "handoffs", "runtimes", "executions", "memory", "settings"}
	assert contract["theme"]["name"] == "agnt_agent_ops"


def test_rule_engine_enforces_agent_composition_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_agent", "model_present": False, "runtime_registered": False, "handoff_endpoint_resolved": False, "workspace_runtime": True, "sandbox_policy_attached": False})
	team_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_team", "agent_count": 0, "runtime_registered": True, "handoff_endpoint_resolved": True})
	review_result = evaluate_capability_rules({"tenant_context_present": True, "runtime_registered": True, "handoff_endpoint_resolved": True, "external_runtime": True, "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "agent_requires_model", "agent_runtime_must_be_registered", "handoff_endpoint_must_resolve", "workspace_runtime_requires_sandbox"}
	assert team_result["matched_rules"] == ["team_requires_agent"]
	assert review_result["decision"] == "require_review"


def test_registration_includes_full_agent_capability_contract():
	registration = register_capability()

	assert registration["name"] == "agnt"
	assert "aicr" in registration["dependencies"]
	assert registration["ui_components"]["runtimes"] == "/agnt/runtimes"
	assert "agnt:run" in registration["permissions"]
	assert "runtime_adapters" in registration["capabilities"]
	assert "runtime_approval_governance" in registration["capabilities"]


def test_service_registers_agents_teams_and_execution_plans():
	service = AgntService()
	agent = service.register_agent(
		agent_id="builder",
		tenant_id="tenant-agnt",
		name="Builder",
		model="gpt-5.4",
		runtime="codex",
		system_prompt="Build APG capability slices.",
		tool_allowlist=["shell", "pytest"],
		input_contract={"objective": "string"},
		output_contract={"patch": "object"},
		memory_policy={"store": "tenant-vector", "retention_days": 30},
	)
	team = service.register_team(
		team_id="delivery",
		tenant_id="tenant-agnt",
		name="Delivery Team",
		agent_ids=["builder"],
		handoffs=[],
		parallel_execution_enabled=True,
	)
	plan = service.plan_execution("delivery", "Implement a capability package", tenant_id="tenant-agnt")

	assert agent["runtime"] == "codex"
	assert team["agent_ids"] == ["builder"]
	assert plan["team_id"] == "delivery"
	assert plan["runtime_assignments"] == {"builder": "codex"}
	assert plan["steps"][0]["tools"] == ["shell", "pytest"]


def test_service_blocks_invalid_agent_team_and_runtime_changes():
	service = AgntService()

	try:
		service.register_agent(
			agent_id="missing-model",
			tenant_id="tenant-agnt",
			name="Missing Model",
			model="",
			runtime="codex",
			system_prompt="Should fail.",
		)
	except PermissionError as exc:
		assert "agent_model_required" in str(exc)
	else:
		raise AssertionError("expected missing model to be blocked")

	try:
		service.register_runtime(
			name="unapproved_external",
			tenant_id="tenant-agnt",
			external_runtime=True,
			approved=False,
		)
	except PermissionError as exc:
		assert "external_runtime_approval_required" in str(exc)
	else:
		raise AssertionError("expected unapproved external runtime to require review")

	try:
		service.register_team(
			team_id="empty",
			tenant_id="tenant-agnt",
			name="Empty Team",
			agent_ids=[],
		)
	except PermissionError as exc:
		assert "team_agent_required" in str(exc)
	else:
		raise AssertionError("expected empty team to be blocked")


def test_external_runtime_approval_lifecycle_enables_new_provider():
	service = AgntService()
	request = service.request_runtime_approval(
		request_id="runtime-request-1",
		tenant_id="tenant-agnt",
		runtime_name="future_agent",
		requested_by="platform-owner",
		kind="external",
		workspace_runtime=True,
		sandbox_policy="workspace-write",
		capabilities=["code", "analysis"],
		cost_limit=12.5,
	)
	queue = views.runtime_approval_queue_model(service, "tenant-agnt")

	assert request["decision"] == "pending"
	assert queue["pending_requests"][0]["runtime_name"] == "future_agent"

	with pytest.raises(PermissionError, match="external_runtime_approval_required"):
		service.register_runtime(
			name="blocked_future_agent",
			tenant_id="tenant-agnt",
			kind="external",
			external_runtime=True,
			approved=False,
			workspace_runtime=True,
			sandbox_policy="workspace-write",
		)

	decision = service.decide_runtime_approval(
		request_id=request["id"],
		tenant_id="tenant-agnt",
		reviewer="security-reviewer",
		decision="approved",
		notes="Sandbox and cost limits accepted.",
	)
	agent = service.register_agent(
		agent_id="future-builder",
		tenant_id="tenant-agnt",
		name="Future Builder",
		model="future-code-model",
		runtime="future_agent",
		system_prompt="Build governed APG slices.",
		tool_allowlist=["shell"],
	)
	team = service.register_team(
		team_id="future-delivery",
		tenant_id="tenant-agnt",
		name="Future Delivery",
		agent_ids=[agent["id"]],
	)
	plan = service.plan_execution(team["id"], "Ship an approved runtime slice.", tenant_id="tenant-agnt")
	evidence = views.governance_evidence_model(service, "tenant-agnt")

	assert decision["decision"] == "approved"
	assert plan["runtime_assignments"] == {"future-builder": "future_agent"}
	assert evidence["summary"]["runtime_approval_count"] == 1
	assert {event["event_type"] for event in evidence["audit_events"]} >= {
		"runtime_approval_requested",
		"runtime_approval_decided",
		"runtime_registered",
		"agent_registered",
		"team_registered",
		"execution_plan_built",
	}


def test_runtime_approval_blocks_missing_sandbox_rejections_and_tenant_mismatch():
	service = AgntService()

	with pytest.raises(PermissionError, match="workspace_sandbox_required"):
		service.request_runtime_approval(
			request_id="runtime-request-missing-sandbox",
			tenant_id="tenant-agnt",
			runtime_name="unsafe_workspace_agent",
			requested_by="platform-owner",
			workspace_runtime=True,
			sandbox_policy=None,
		)

	request = service.request_runtime_approval(
		request_id="runtime-request-rejected",
		tenant_id="tenant-agnt",
		runtime_name="rejected_agent",
		requested_by="platform-owner",
		workspace_runtime=False,
	)

	with pytest.raises(KeyError, match="unknown runtime approval request for tenant"):
		service.decide_runtime_approval(
			request_id=request["id"],
			tenant_id="other-tenant",
			reviewer="security-reviewer",
			decision="approved",
			notes="Wrong tenant.",
		)

	decision = service.decide_runtime_approval(
		request_id=request["id"],
		tenant_id="tenant-agnt",
		reviewer="security-reviewer",
		decision="rejected",
		notes="Provider risk not accepted.",
	)

	assert decision["decision"] == "rejected"
	assert "rejected_agent" not in {runtime["name"] for runtime in service.list_runtimes()}


def test_api_helpers_expose_runtime_approval_lifecycle():
	request = api.request_runtime_approval({
		"id": "api-runtime-request",
		"tenant_id": "tenant-api-agnt",
		"runtime_name": "api_future_agent",
		"requested_by": "api-owner",
		"workspace_runtime": True,
		"sandbox_policy": "workspace-write",
		"capabilities": ["code"],
	})
	decision = api.decide_runtime_approval({
		"id": request["id"],
		"tenant_id": request["tenant_id"],
		"reviewer": "api-security",
		"decision": "approved",
		"notes": "API runtime accepted.",
	})

	assert decision["decision"] == "approved"
	assert api.list_runtime_approvals(request["tenant_id"])[0]["runtime_name"] == "api_future_agent"
	assert "runtime_approval_requested" in {event["event_type"] for event in api.list_audit_events(request["tenant_id"])}
