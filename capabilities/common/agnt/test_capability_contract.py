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
	assert contract["configuration_schema"]["required"] == ["tenant_id", "agents", "teams", "runtimes", "memory", "governance", "observability", "adapters", "ui", "theme"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert set(contract["provides"]) >= {"agent_registry", "runtime_registry", "execution_plans", "execution_runs", "runtime_approval_governance"}
	assert contract["requires"] == ["aicr", "sbox", "audl"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "agents", "teams", "handoffs", "runtimes", "executions", "runs", "memory", "approvals", "audit", "analytics", "settings"}
	assert contract["theme"]["name"] == "agnt_agent_ops"


def test_rule_engine_enforces_agent_composition_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_agent", "model_present": False, "system_prompt_present": False, "tool_allowlist_present": False, "io_contract_present": False, "memory_policy_present": False, "runtime_registered": False, "handoff_endpoint_resolved": False, "workspace_runtime": True, "sandbox_policy_attached": False})
	team_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_team", "agent_count": 0, "runtime_registered": True, "handoff_endpoint_resolved": True})
	review_result = evaluate_capability_rules({"tenant_context_present": True, "runtime_registered": True, "handoff_endpoint_resolved": True, "external_runtime": True, "approval_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_agent_mutation", "event_stream": "memory"})
	run_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "record_execution_run", "requester_present": False, "trace_sink_present": False, "side_effects_requested": True, "human_approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "agent_requires_model", "agent_requires_system_prompt", "agent_requires_tool_allowlist", "agent_requires_io_contract", "agent_requires_memory_policy", "agent_runtime_must_be_registered", "handoff_endpoint_must_resolve", "workspace_runtime_requires_sandbox"}
	assert team_result["matched_rules"] == ["team_requires_agent"]
	assert review_result["decision"] == "require_review"
	assert batch_result["matched_rules"] == ["batch_agent_mutation_requires_bytewax"]
	assert run_result["decision"] == "deny"
	assert set(run_result["matched_rules"]) == {"execution_run_requires_requester", "execution_run_requires_trace_sink", "execution_side_effect_requires_human_approval"}


def test_registration_includes_full_agent_capability_contract():
	registration = register_capability()

	assert registration["name"] == "agnt"
	assert "aicr" in registration["dependencies"]
	assert registration["ui_components"]["runtimes"] == "/agnt/runtimes"
	assert registration["ui_components"]["approvals"] == "/agnt/approvals"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "agnt:run" in registration["permissions"]
	assert "runtime_adapters" in registration["capabilities"]
	assert "runtime_approval_governance" in registration["capabilities"]
	assert "execution_runs" in registration["capabilities"]
	assert registration["ui_components"]["runs"] == "/agnt/runs"


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


def test_service_records_governed_execution_runs():
	service = AgntService()
	service.register_agent(
		agent_id="runner",
		tenant_id="tenant-agnt-run",
		name="Runner",
		model="gpt-5.4",
		runtime="codex",
		system_prompt="Run governed APG work.",
		tool_allowlist=["shell"],
		input_contract={"objective": "string"},
		output_contract={"trace": "object"},
		memory_policy={"store": "tenant-vector", "retention_days": 7},
	)
	service.register_team(
		team_id="run-team",
		tenant_id="tenant-agnt-run",
		name="Run Team",
		agent_ids=["runner"],
	)

	with pytest.raises(PermissionError, match="execution_requester_required"):
		service.record_execution_run(
			run_id="run-missing-requester",
			tenant_id="tenant-agnt-run",
			team_id="run-team",
			objective="Build an APG slice.",
			requested_by="",
			trace_sink="audl",
		)
	with pytest.raises(PermissionError, match="execution_trace_sink_required"):
		service.record_execution_run(
			run_id="run-missing-trace",
			tenant_id="tenant-agnt-run",
			team_id="run-team",
			objective="Build an APG slice.",
			requested_by="platform-owner",
			trace_sink="",
		)
	with pytest.raises(PermissionError, match="execution_side_effect_approval_required"):
		service.record_execution_run(
			run_id="run-side-effect",
			tenant_id="tenant-agnt-run",
			team_id="run-team",
			objective="Push changes.",
			requested_by="platform-owner",
			trace_sink="audl",
			side_effects_requested=True,
			human_approval_recorded=False,
		)

	run = service.record_execution_run(
		run_id="run-1",
		tenant_id="tenant-agnt-run",
		team_id="run-team",
		objective="Build an APG slice.",
		requested_by="platform-owner",
		trace_sink="audl",
		side_effects_requested=True,
		human_approval_recorded=True,
	)
	console = views.execution_run_console_model(service, "tenant-agnt-run")
	evidence = views.governance_evidence_model(service, "tenant-agnt-run")
	analytics = views.analytics_model(service, "tenant-agnt-run")

	assert run["plan_snapshot"]["runtime_assignments"] == {"runner": "codex"}
	assert run["trace_sink"] == "audl"
	assert console["execution_runs"][0]["id"] == "run-1"
	assert evidence["summary"]["execution_run_count"] == 1
	assert analytics["runs_per_team"] == 1.0
	assert "execution_run_recorded" in {event["event_type"] for event in evidence["audit_events"]}


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
		input_contract={"objective": "string"},
		output_contract={"plan": "object"},
		memory_policy={"store": "tenant-vector", "retention_days": 14},
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


def test_tenant_scope_views_and_bytewax_guardrail():
	service = AgntService()
	for tenant_id, agent_name in (("tenant-a", "agent-a"), ("tenant-b", "agent-b")):
		service.register_agent(
			agent_id="shared-agent",
			tenant_id=tenant_id,
			name=agent_name,
			model="gpt-5.4",
			runtime="codex",
			system_prompt="Execute governed APG work.",
			tool_allowlist=["shell"],
			input_contract={"objective": "string"},
			output_contract={"result": "object"},
			memory_policy={"store": "tenant-vector", "retention_days": 7},
		)
		service.register_team(
			team_id="shared-team",
			tenant_id=tenant_id,
			name=f"{agent_name} team",
			agent_ids=["shared-agent"],
		)
	batch = service.validate_batch_agent_mutation(
		tenant_id="tenant-a",
		event_stream="bytewax",
		mutation_count=2,
	)
	dashboard = views.dashboard_model(service, "tenant-a")
	analytics = views.analytics_model(service, "tenant-a")
	settings = views.settings_model("tenant-a")

	assert batch["accepted"] is True
	assert service.list_agents("tenant-a")[0]["name"] == "agent-a"
	assert service.list_agents("tenant-b")[0]["name"] == "agent-b"
	assert dashboard["streaming"]["processor"] == "bytewax"
	assert analytics["summary"]["agent_count"] == 1
	assert settings["streaming"]["topic"] == "apg.agnt.lifecycle"

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_agent_mutation(
			tenant_id="tenant-a",
			event_stream="memory",
			mutation_count=1,
		)

	with pytest.raises(KeyError, match="unknown agent team for tenant"):
		service.plan_execution("shared-team", "cross tenant attempt", tenant_id="missing-tenant")


def test_api_helpers_expose_runtime_approval_lifecycle():
	request = api.request_runtime_approval({
		"id": "api-runtime-request",
		"tenant_id": "tenant-api-agnt",
		"runtime_name": "api_future_agent",
		"requested_by": "api-owner",
		"workspace_runtime": True,
		"sandbox_policy": "workspace-write",
		"capabilities": ["code"],
		"cost_limit": 10.0,
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


def test_api_helpers_expose_batch_agent_mutation_guardrail():
	batch = api.validate_batch_agent_mutation({
		"tenant_id": "tenant-api-agnt-batch",
		"event_stream": "bytewax",
		"mutation_count": 1,
	})

	assert batch["accepted"] is True


def test_api_helpers_expose_execution_run_lifecycle():
	tenant_id = "tenant-api-agnt-run"
	agent = api.register_agent({
		"id": "api-runner",
		"tenant_id": tenant_id,
		"name": "API Runner",
		"model": "gpt-5.4",
		"runtime": "codex",
		"system_prompt": "Run governed APG work.",
		"tool_allowlist": ["shell"],
		"input_contract": {"objective": "string"},
		"output_contract": {"trace": "object"},
		"memory_policy": {"store": "tenant-vector", "retention_days": 7},
	})
	team = api.register_team({
		"id": "api-run-team",
		"tenant_id": tenant_id,
		"name": "API Run Team",
		"agent_ids": [agent["id"]],
	})
	run = api.record_execution_run({
		"id": "api-run-1",
		"tenant_id": tenant_id,
		"team_id": team["id"],
		"objective": "Build an API-driven APG slice.",
		"requested_by": "api-owner",
		"trace_sink": "audl",
	})

	assert run["requested_by"] == "api-owner"
	assert api.list_execution_runs(tenant_id)[0]["id"] == "api-run-1"
