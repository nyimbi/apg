"""Regression coverage for the AICR executable capability contract."""

import pytest

from capabilities.common.aicr import api_helpers, views
from capabilities.common.aicr import register_capability
from capabilities.common.aicr.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.aicr.service import AicrService


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-ai", {"inference": {"max_concurrent_requests": 32}})

	assert contract["capability"] == "aicr"
	assert contract["configuration"]["tenant_id"] == "tenant-ai"
	assert contract["configuration"]["inference"]["max_concurrent_requests"] == 32
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"services",
		"providers",
		"models",
		"inference",
		"workflows",
		"agent_runtimes",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 38
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.AicrService"
	assert contract["provides"] == ["ai_core", "model_inference", "ai_agent_composition"]
	assert contract["requires"] == ["conf", "auth", "mqeb", "moni"]
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["engine"] == "bytewax"
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"services",
		"providers",
		"models",
		"inference",
		"workflows",
		"agent_runtimes",
		"agents",
		"lifecycle",
		"governance",
		"evaluations",
		"metrics",
		"audit",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/aicr/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "agent_runtime_card" in contract["theme"]["components"]
	assert "ai_agent_roster" in contract["theme"]["components"]
	assert "bytewax_lifecycle_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_ai_core_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "run_inference",
		"owner_assigned": False,
		"workflow_risk": "high",
		"approval_recorded": False,
		"context_tokens": 256000,
		"review_recorded": False,
		"model_policy_attached": False,
		"service_health": "unhealthy",
		"routing_requested": True,
		"pii_detected": True,
		"pii_redaction_enabled": False,
		"tool_call_requested": True,
		"tool_allowlist_attached": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"inference_requires_model_policy",
		"unhealthy_service_blocks_routing",
		"high_risk_workflow_requires_approval",
		"large_context_requires_review",
		"pii_inference_requires_redaction",
		"tool_call_requires_allowlist",
	}


def test_rule_engine_enforces_first_class_agent_and_bytewax_guardrails():
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_ai_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_aicr_lifecycle_batch",
		"event_stream": "kafka",
	})

	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"ai_agent_runtime_supported",
		"ai_agent_role_supported",
		"ai_agent_requires_scope",
		"ai_agent_requires_owner",
		"ai_agent_requires_purpose",
		"ai_agent_requires_contribution_disclosure",
		"ai_agent_privileged_role_requires_human_approval",
	}
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["bytewax_aicr_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "aicr"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "aicr_ai_control_console"
	assert registration["ui_components"]["services"] == "/aicr/services"
	assert "inference_approval_governance" in registration["capabilities"]
	assert "agent_runtime_registry" in registration["capabilities"]
	assert "ai_agent_composition" in registration["capabilities"]
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert "auth" in registration["dependencies"]
	assert "aicr:run_inference" in registration["permissions"]
	assert "aicr:manage_agents" in registration["permissions"]


def test_service_registers_provider_model_workflow_and_agent_runtime():
	service = AicrService()
	provider = service.register_provider(
		"codex-provider",
		"tenant-ai",
		"Codex",
		"codex",
		"ai-platform",
		credential_vault_ref="keym://codex",
		egress_policy_ref="policy://ai-egress",
	)
	service.register_ai_service(
		service_id="llm-router",
		tenant_id="tenant-ai",
		name="LLM Router",
		owner="ai-platform",
		health="healthy",
		model_policy={"policy_id": "safe-gen"},
	)
	model = service.register_model(
		"reasoning-model",
		"tenant-ai",
		"Reasoning Model",
		"codex-provider",
		"ai-platform",
		"text",
		model_policy={"policy_id": "safe-gen"},
	)
	evaluated = service.record_model_evaluation("tenant-ai", "reasoning-model", 0.97, "eval-owner")
	promoted = service.promote_model("tenant-ai", "reasoning-model")
	workflow = service.create_workflow("support-flow", "tenant-ai", "Support Flow", "workflow-owner", ["llm-router"], risk="high")
	agent = service.register_agent_runtime("codex-runtime", "tenant-ai", "Codex Runtime", "codex", "agent-owner", "policy://tools")
	ai_agent = service.register_ai_agent(
		"codex-reviewer",
		"tenant-ai",
		"Codex Reviewer",
		"codex",
		"model_steward",
		"model catalog triage",
		"ai-platform",
		"Keep model metadata consistent.",
	)
	batch = service.validate_aicr_lifecycle_batch("tenant-ai", "bytewax", 3, "ai_agent_batch", "agent-batch-1")
	models = views.model_catalog_model(service, "tenant-ai")
	agents = views.agent_runtime_console_model(service, "tenant-ai")
	roster = views.ai_agent_roster_model(service, "tenant-ai")
	lifecycle = views.lifecycle_batch_model(service, "tenant-ai")

	assert provider["provider_type"] == "codex"
	assert model["status"] == "registered"
	assert evaluated["evaluation_recorded"] is True
	assert promoted["status"] == "promoted"
	assert workflow["service_ids"] == ["llm-router"]
	assert agent["runtime_type"] == "codex"
	assert ai_agent["runtime"] == "codex"
	assert ai_agent["status"] == "active"
	assert batch["accepted"] is True
	assert models["models"][0]["id"] == "reasoning-model"
	assert agents["agent_runtimes"][0]["id"] == "codex-runtime"
	assert roster["agents"][0]["id"] == "codex-reviewer"
	assert lifecycle["batches"][0]["id"] == "agent-batch-1"
	assert service.governance_summary("tenant-ai")["agent_runtime_count"] == 1
	assert service.governance_summary("tenant-ai")["ai_agent_count"] == 1
	assert service.governance_summary("tenant-ai")["lifecycle_batch_count"] == 1


def test_service_blocks_provider_model_workflow_and_agent_guardrail_gaps():
	service = AicrService()

	with pytest.raises(PermissionError, match="provider_credential_vault_required"):
		service.register_provider("external", "tenant-ai", "External", "openai", "ai-owner", external=True)

	service.register_provider("local", "tenant-ai", "Local", "local", "ai-owner", external=False)
	with pytest.raises(PermissionError, match="registered_provider_required"):
		service.register_model("model", "tenant-ai", "Model", "missing", "ai-owner", "text", model_policy={"policy_id": "safe"})
	with pytest.raises(PermissionError, match="unsupported_model_modality"):
		service.register_model("model", "tenant-ai", "Model", "local", "ai-owner", "quantum", model_policy={"policy_id": "safe"})
	service.register_ai_service("svc", "tenant-ai", "Service", "ai-owner", model_policy={"policy_id": "safe"})
	with pytest.raises(PermissionError, match="workflow_service_bindings_required"):
		service.create_workflow("wf", "tenant-ai", "Workflow", "owner", ["missing"])
	with pytest.raises(PermissionError, match="unsupported_agent_runtime"):
		service.register_agent_runtime("runtime", "tenant-ai", "Runtime", "unknown", "owner", "policy://tools")
	with pytest.raises(PermissionError, match="agent_tool_policy_required"):
		service.register_agent_runtime("runtime", "tenant-ai", "Runtime", "codex", "owner", "")
	with pytest.raises(PermissionError, match="unsupported_ai_agent_runtime"):
		service.register_ai_agent("agent", "tenant-ai", "Agent", "unknown", "model_steward", "models", "owner", "purpose")
	with pytest.raises(PermissionError, match="ai_agent_scope_required"):
		service.register_ai_agent("agent", "tenant-ai", "Agent", "codex", "model_steward", "", "owner", "purpose")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_aicr_lifecycle_batch("tenant-ai", "kafka", 1)


def test_privileged_ai_agent_without_human_approval_is_pending_review():
	service = AicrService()

	agent = service.register_ai_agent(
		"cost-reviewer",
		"tenant-ai",
		"Cost Reviewer",
		"claude_code",
		"cost_reviewer",
		"cost gates",
		"finops-owner",
		"Review expensive AI runs.",
	)

	assert agent["status"] == "pending_review"
	assert service.governance_summary("tenant-ai")["ai_agent_count"] == 1


def test_service_runs_high_risk_inference_approval_lifecycle():
	service = AicrService()
	record = service.register_ai_service(
		service_id="llm-router",
		tenant_id="tenant-ai",
		name="LLM Router",
		owner="ai-platform-owner",
		health="healthy",
		model_policy={"policy_id": "safe-gen", "pii_redaction": True},
	)
	approval = service.request_inference(
		request_id="inference-1",
		tenant_id="tenant-ai",
		service_id=record["id"],
		requested_by="workflow-owner",
		prompt_summary="Summarize customer support case.",
		context_tokens=256000,
		workflow_risk="high",
	)
	queue = views.inference_console_model(service, "tenant-ai")

	assert approval["decision"] == "pending"
	assert queue["pending_approvals"][0]["id"] == "inference-1"

	with pytest.raises(PermissionError, match="inference_approval_required"):
		service.run_approved_inference("inference-1", "tenant-ai")

	decision = service.decide_inference_approval(
		request_id="inference-1",
		tenant_id="tenant-ai",
		reviewer="ai-risk-reviewer",
		decision="approved",
		notes="Context and workflow risk accepted for this run.",
	)
	result = service.run_approved_inference("inference-1", "tenant-ai")
	evidence = views.governance_center_model(service, "tenant-ai")

	assert decision["decision"] == "approved"
	assert result["status"] == "completed"
	assert evidence["summary"]["inference_approval_count"] == 1
	assert {event["event_type"] for event in evidence["audit_events"]} >= {
		"ai_service_registered",
		"inference_approval_requested",
		"inference_approval_decided",
		"approved_inference_completed",
	}


def test_service_blocks_missing_policy_unhealthy_routing_and_rejected_approval():
	service = AicrService()

	with pytest.raises(PermissionError, match="service_owner_required"):
		service.register_ai_service(
			service_id="ownerless",
			tenant_id="tenant-ai",
			name="Ownerless",
			owner="",
		)

	service.register_ai_service(
		service_id="unhealthy",
		tenant_id="tenant-ai",
		name="Unhealthy Service",
		owner="ai-owner",
		health="unhealthy",
		model_policy={"policy_id": "safe-gen"},
	)

	with pytest.raises(PermissionError, match="service_unhealthy"):
		service.request_inference(
			request_id="blocked-unhealthy",
			tenant_id="tenant-ai",
			service_id="unhealthy",
			requested_by="workflow-owner",
			prompt_summary="Blocked unhealthy route.",
		)

	service.register_ai_service(
		service_id="healthy",
		tenant_id="tenant-ai",
		name="Healthy Service",
		owner="ai-owner",
		health="healthy",
		model_policy={"policy_id": "safe-gen"},
	)

	with pytest.raises(PermissionError, match="model_policy_required"):
		service.request_inference(
			request_id="missing-policy",
			tenant_id="tenant-ai",
			service_id="healthy",
			requested_by="workflow-owner",
			prompt_summary="Missing model policy.",
			model_policy_attached=False,
		)

	approval = service.request_inference(
		request_id="rejected-risk",
		tenant_id="tenant-ai",
		service_id="healthy",
		requested_by="workflow-owner",
		prompt_summary="High-risk rejected inference.",
		workflow_risk="high",
	)
	service.decide_inference_approval(
		request_id=approval["id"],
		tenant_id="tenant-ai",
		reviewer="ai-risk-reviewer",
		decision="rejected",
		notes="Risk not accepted.",
	)

	with pytest.raises(PermissionError, match="inference_approval_required"):
		service.run_approved_inference(approval["id"], "tenant-ai")

	with pytest.raises(KeyError, match="unknown inference approval for tenant"):
		service.decide_inference_approval(
			request_id=approval["id"],
			tenant_id="other-tenant",
			reviewer="ai-risk-reviewer",
			decision="approved",
			notes="Wrong tenant.",
		)


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = AicrService()
	for tenant_id, owner in [("tenant-a", "owner-a"), ("tenant-b", "owner-b")]:
		service.register_ai_service(
			service_id="shared-router",
			tenant_id=tenant_id,
			name=f"Shared Router {tenant_id}",
			owner=owner,
			health="healthy",
			model_policy={"policy_id": f"policy-{tenant_id}"},
		)
		service.request_inference(
			request_id="shared-request",
			tenant_id=tenant_id,
			service_id="shared-router",
			requested_by=owner,
			prompt_summary=f"High-risk request for {tenant_id}.",
			workflow_risk="high",
		)

	service.decide_inference_approval(
		request_id="shared-request",
		tenant_id="tenant-a",
		reviewer="reviewer-a",
		decision="approved",
		notes="Approve tenant-a only.",
	)
	tenant_a_result = service.run_approved_inference("shared-request", "tenant-a")

	with pytest.raises(PermissionError, match="inference_approval_required"):
		service.run_approved_inference("shared-request", "tenant-b")

	assert tenant_a_result["tenant_id"] == "tenant-a"
	assert service.list_inference_approvals("tenant-a")[0]["decision"] == "approved"
	assert service.list_inference_approvals("tenant-b")[0]["decision"] == "pending"


def test_api_helpers_expose_governed_inference_lifecycle():
	record = api_helpers.register_ai_service({
		"id": "api-llm",
		"tenant_id": "tenant-api-ai",
		"name": "API LLM",
		"owner": "api-owner",
		"model_policy": {"policy_id": "api-safe-gen"},
	})
	approval = api_helpers.request_inference({
		"id": "api-inference",
		"tenant_id": record["tenant_id"],
		"service_id": record["id"],
		"requested_by": "api-workflow",
		"prompt_summary": "API high-risk request.",
		"workflow_risk": "high",
	})
	decision = api_helpers.decide_inference_approval({
		"id": approval["id"],
		"tenant_id": approval["tenant_id"],
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "API approval accepted.",
	})
	result = api_helpers.run_approved_inference({
		"id": approval["id"],
		"tenant_id": approval["tenant_id"],
	})

	assert decision["decision"] == "approved"
	assert result["status"] == "completed"
	assert api_helpers.list_inference_approvals(record["tenant_id"])[0]["id"] == approval["id"]


def test_api_helpers_expose_first_class_ai_agents_and_lifecycle_batches():
	agent = api_helpers.register_ai_agent({
		"id": "api-agent",
		"tenant_id": "tenant-api-agent",
		"name": "API Agent",
		"runtime": "opencode",
		"role": "model_steward",
		"scope": "catalog hygiene",
		"owner": "ai-platform",
		"purpose": "Keep model metadata usable.",
	})
	batch = api_helpers.validate_aicr_lifecycle_batch({
		"id": "api-batch",
		"tenant_id": "tenant-api-agent",
		"event_stream": "bytewax",
		"mutation_count": 2,
		"operation": "ai_agent_batch",
	})

	assert agent["id"] == "api-agent"
	assert batch["accepted"] is True
	assert api_helpers.list_ai_agents("tenant-api-agent")[0]["id"] == "api-agent"
	assert api_helpers.list_lifecycle_batches("tenant-api-agent")[0]["id"] == "api-batch"
