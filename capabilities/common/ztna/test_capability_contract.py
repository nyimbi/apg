"""Regression coverage for the ZTNA executable capability contract."""

import pytest

from capabilities.common.ztna import api
from capabilities.common.ztna import register_capability
from capabilities.common.ztna.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.ztna.service import ZtnaService
from capabilities.common.ztna import views


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-zero", {"devices": {"minimum_device_trust": 0.8}})

	assert contract["capability"] == "ztna"
	assert contract["configuration"]["tenant_id"] == "tenant-zero"
	assert contract["configuration"]["devices"]["minimum_device_trust"] == 0.8
	assert set(contract["configuration_schema"]["required"]) >= {
		"tenant_id",
		"identities",
		"devices",
		"resources",
		"access",
		"sessions",
		"segmentation",
		"reviews",
		"security",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert len(contract["rule_engine"]["rules"]) >= 42
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "policies", "identities", "devices", "resources", "access", "sessions", "risk", "reviews", "agents", "lifecycle", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/ztna/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "device_posture" in contract["theme"]["components"]
	assert "review_queue" in contract["theme"]["components"]
	assert "zero_trust_agent_roster" in contract["theme"]["components"]
	assert "bytewax_lifecycle_panel" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["agents"]["first_class"] is True
	assert set(contract["agents"]["supported_runtimes"]) == {"codex", "claude_code", "opencode", "pi"}
	assert "zero_trust_steward" in contract["agents"]["supported_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "ztna_agent_batch" in contract["streaming"]["required_operations"]


def test_rule_engine_enforces_zero_trust_guardrails():
	result = evaluate_capability_rules({
		"operation": "request_access",
		"tenant_context_present": False,
		"identity_verified": False,
		"device_posture_present": False,
		"device_trust_score": 0.2,
		"device_compliant": False,
		"resource_policy_attached": False,
		"access_level": "privileged",
		"mfa_completed": False,
		"access_risk_score": 0.95,
		"access_review_recorded": False,
		"just_in_time_approval_present": False,
		"least_privilege_scope_present": False,
		"explicit_access_decision_present": False,
		"duplicate_pending_review": True,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"identity_must_be_verified",
		"device_posture_required",
		"device_trust_score_requires_threshold",
		"device_compliance_required",
		"resource_policy_required",
		"privileged_access_requires_mfa",
		"privileged_access_requires_approval",
		"least_privilege_scope_required",
		"high_risk_access_requires_review"
	}


def test_rule_engine_requires_bytewax_for_batch_zero_trust_mutations():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "batch_ztna_mutation",
		"event_stream": "kafka",
	})

	assert result["decision"] == "deny"
	assert "batch_ztna_mutation_requires_bytewax" in result["matched_rules"]

	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_zero_trust_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_ztna_lifecycle_batch",
		"event_stream": "kafka",
		"mutation_count": 1,
	})

	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"ztna_agent_runtime_supported",
		"ztna_agent_role_supported",
		"ztna_agent_requires_scope",
		"ztna_agent_requires_owner",
		"ztna_agent_requires_purpose",
		"ztna_agent_requires_contribution_disclosure",
	}
	assert lifecycle_result["matched_rules"] == ["bytewax_ztna_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ztna"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ztna_zero_trust_ops"
	assert registration["ui_components"]["resources"] == "/ztna/resources"
	assert registration["ui_components"]["audit"] == "/ztna/audit"
	assert "mfau" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["endpoints"]["agents"] == "/ztna/api/v1/agents"
	assert registration["endpoints"]["lifecycle"] == "/ztna/api/v1/lifecycle"
	assert "ztna:approve_access" in registration["permissions"]
	assert "ztna:audit" in registration["permissions"]
	assert "ztna:review" in registration["permissions"]


def test_runtime_executes_standard_access_lifecycle_and_view_models():
	service = ZtnaService()
	identity = service.register_identity("analyst", "tenant-zero", "user-1", "Analyst", verified=True)
	device = service.register_device("laptop", "tenant-zero", identity["id"], "Managed Laptop", trust_score=0.94, managed=True, attested=True)
	resource = service.register_resource("crm", "tenant-zero", "CRM", policy_attached=True, policy_id="crm-policy")
	request = service.request_access(identity["id"], device["id"], resource["id"], requested_by="user-1")
	session = service.start_session(request["id"], actor_id="broker")
	closed = service.close_session(session["id"], actor_id="broker")

	assert request["status"] == "approved"
	assert session["status"] == "active"
	assert closed["status"] == "closed"
	assert views.identity_console_model(service, "tenant-zero")["identities"][0]["id"] == identity["id"]
	assert views.review_queue_model(service, "tenant-zero")["review_required"] == []
	assert views.zero_trust_agent_roster_model(service, "tenant-zero")["supported_runtimes"]
	assert views.lifecycle_batch_model(service, "tenant-zero")["streaming"]["required_processor"] == "bytewax"
	assert len(views.audit_model(service, "tenant-zero")["audit_events"]) >= 5


def test_privileged_access_requires_mfa_and_independent_review():
	service = ZtnaService()
	identity = service.register_identity("admin", "tenant-zero", "admin-1", "Admin", verified=True, privileged=True, mfa_completed=True)
	device = service.register_device("admin-laptop", "tenant-zero", identity["id"], "Admin Laptop", trust_score=0.96, managed=True, attested=True)
	resource = service.register_resource("root", "tenant-zero", "Root Console", access_level="privileged", sensitive=True, policy_attached=True)
	request = service.request_access(identity["id"], device["id"], resource["id"], requested_by="admin-1", mfa_completed=True)

	assert request["status"] == "review_required"
	assert "approve_privileged_access" in request["required_actions"]
	with pytest.raises(PermissionError, match="independent_access_review_required"):
		service.approve_access_request(request["id"], reviewer_id="admin-1")

	approved = service.approve_access_request(request["id"], reviewer_id="reviewer-1")
	session = service.start_session(approved["id"], actor_id="broker")

	assert approved["status"] == "approved"
	assert session["status"] == "active"


def test_tenant_local_keys_and_cross_tenant_guardrails_are_enforced():
	service = ZtnaService()
	alpha_identity = service.register_identity("shared", "tenant-alpha", "alpha-user", "Alpha", verified=True)
	beta_identity = service.register_identity("shared", "tenant-beta", "beta-user", "Beta", verified=True)
	alpha_device = service.register_device("device", "tenant-alpha", alpha_identity["id"], "Alpha Device", trust_score=0.93, managed=True)
	beta_device = service.register_device("device", "tenant-beta", beta_identity["id"], "Beta Device", trust_score=0.93, managed=True)
	alpha_resource = service.register_resource("crm", "tenant-alpha", "CRM", policy_attached=True)

	assert alpha_identity["id"] != beta_identity["id"]
	assert service.list_identities("tenant-alpha") == [alpha_identity]
	assert service.list_devices("tenant-beta") == [beta_device]
	with pytest.raises(PermissionError, match="cross_tenant_zero_trust_access_denied"):
		service.request_access(beta_identity["id"], alpha_device["id"], alpha_resource["id"], requested_by="beta-user")


def test_zero_trust_agents_and_lifecycle_batches_are_first_class_runtime_surfaces():
	service = ZtnaService()
	tenant_id = "tenant-agent-ztna"

	agent = service.register_zero_trust_agent(
		"agent-steward",
		tenant_id,
		"Zero Trust Steward",
		"opencode",
		"zero_trust_steward",
		"tenant:tenant-agent-ztna",
		"security-platform",
		"review zero-trust lifecycle batches",
		human_approval_required=True,
	)
	batch = service.validate_ztna_lifecycle_batch(tenant_id, "bytewax", 3, "ztna_agent_batch", "batch-agent")

	assert agent["status"] == "active"
	assert batch["accepted"] is True
	assert service.dashboard_summary(tenant_id)["zero_trust_agent_count"] == 1
	assert service.dashboard_summary(tenant_id)["lifecycle_batch_count"] == 1
	assert views.zero_trust_agent_roster_model(service, tenant_id)["agents"][0]["id"] == agent["id"]
	assert views.lifecycle_batch_model(service, tenant_id)["batches"][0]["id"] == batch["id"]

	pending = service.register_zero_trust_agent(
		"agent-session-risk",
		tenant_id,
		"Session Risk Reviewer",
		"claude_code",
		"session_risk_reviewer",
		"session:*",
		"security-ops",
		"review risky sessions",
	)
	assert pending["status"] == "pending_review"

	with pytest.raises(PermissionError, match="unsupported_ztna_agent_runtime"):
		service.register_zero_trust_agent("agent-unsupported", tenant_id, "Unsupported", "kafka_agent", "policy_reviewer", "policy:*", "owner", "review policies")
	with pytest.raises(ValueError, match="ztna_lifecycle_batch_empty"):
		service.validate_ztna_lifecycle_batch(tenant_id, "bytewax", 0, "policy_batch")
	with pytest.raises(ValueError, match="unsupported_ztna_lifecycle_operation"):
		service.validate_ztna_lifecycle_batch(tenant_id, "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_ztna_lifecycle_batch(tenant_id, "kafka", 1, "policy_batch")

	api_agent = api.register_zero_trust_agent({
		"id": "api-agent",
		"tenant_id": "tenant-api-agent-ztna",
		"name": "API Policy Reviewer",
		"runtime": "codex",
		"role": "policy_reviewer",
		"scope": "policy:*",
		"owner": "security-ops",
		"purpose": "review generated zero-trust policy",
	})
	api_batch = api.validate_lifecycle_batch({
		"id": "api-batch",
		"tenant_id": "tenant-api-agent-ztna",
		"event_stream": "bytewax",
		"mutation_count": 2,
		"operation": "policy_batch",
	})

	assert api_agent["runtime"] == "codex"
	assert api_batch["status"] == "accepted"
	assert api.capability_status("tenant-api-agent-ztna")["zero_trust_agent_count"] == 1
