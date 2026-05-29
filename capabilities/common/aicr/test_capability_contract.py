"""Regression coverage for the AICR executable capability contract."""

import pytest

from capabilities.common.aicr import api_helpers, views
from capabilities.common.aicr import register_capability
from capabilities.common.aicr.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.aicr.service import AicrService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-ai", {"inference": {"max_concurrent_requests": 32}})

	assert contract["capability"] == "aicr"
	assert contract["configuration"]["tenant_id"] == "tenant-ai"
	assert contract["configuration"]["inference"]["max_concurrent_requests"] == 32
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"services",
		"inference",
		"orchestration",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"services",
		"inference",
		"models",
		"workflows",
		"governance",
		"metrics",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/aicr/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "workflow_graph" in contract["theme"]["components"]


def test_rule_engine_enforces_ai_core_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_service",
		"owner_assigned": False,
		"workflow_risk": "high",
		"approval_recorded": False,
		"context_tokens": 256000,
		"review_recorded": False
	})
	inference_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "run_inference",
		"model_policy_attached": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"service_registration_requires_owner",
		"high_risk_workflow_requires_approval",
		"large_context_requires_review"
	}
	assert inference_result["decision"] == "deny"
	assert inference_result["matched_rules"] == ["inference_requires_model_policy"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "aicr"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "aicr_ai_control_console"
	assert registration["ui_components"]["services"] == "/aicr/services"
	assert "inference_approval_governance" in registration["capabilities"]
	assert "auth" in registration["dependencies"]
	assert "aicr:run_inference" in registration["permissions"]


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
