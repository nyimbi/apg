"""Regression coverage for the CACH executable capability contract."""

import pytest

from capabilities.common.cach import register_capability
from capabilities.common.cach.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.cach.service import CacheGovernanceService
from capabilities.common.cach.view_models import (
	adapter_health_model,
	cache_agent_roster_model,
	dashboard_model,
	eviction_review_model,
	lifecycle_batch_model,
	namespace_inventory_model,
	settings_model,
	warming_console_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-cache",
		{"policy": {"default_ttl_seconds": 120}}
	)

	assert contract["capability"] == "cach"
	assert contract["configuration"]["tenant_id"] == "tenant-cache"
	assert contract["configuration"]["policy"]["default_ttl_seconds"] == 120
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"hierarchy",
		"policy",
		"warming",
		"security",
		"optimization",
		"adapters",
		"agents",
		"streaming",
		"telemetry",
		"ui",
		"theme"
	]
	assert contract["agents"]["first_class"] is True
	assert "codex" in contract["agents"]["supported_runtimes"]
	assert "eviction_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["engine"] == "bytewax"
	assert contract["streaming"]["broker_core_dependency_allowed"] is False
	assert "review_evidence" in contract["provides"]
	assert "cache_agents" in contract["review_evidence"]["pending_queues"]
	assert "policy_decision" in contract["review_evidence"]["policy_fields"]
	assert len(contract["rule_engine"]["rules"]) >= 24
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"namespaces",
		"entries",
		"policies",
		"warming",
		"evictions",
		"hierarchy",
		"analytics",
		"security",
		"adapters",
		"agents",
		"lifecycle",
		"audit",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/cach/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "tier_hierarchy_map" in contract["theme"]["components"]
	assert "cache_agent_roster" in contract["theme"]["components"]


def test_rule_engine_enforces_cache_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "write",
		"namespace_present": False,
		"data_classification": "sensitive",
		"entry_encrypted": False,
		"cross_tenant_access": True,
		"data_criticality": "critical",
		"entry_stale": True,
		"memory_utilization_percent": 95,
		"eviction_plan_ready": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"write_requires_namespace",
		"sensitive_entry_requires_encryption",
		"cross_tenant_cache_access_denied",
		"high_memory_pressure_requires_review"
	}


def test_rule_engine_enforces_cache_agent_and_bytewax_guardrails():
	agent = evaluate_capability_rules({
		"operation": "register_cache_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"agent_scope_present": False,
		"agent_owner_present": False,
		"agent_purpose_present": False,
		"contribution_disclosed": False,
		"privileged_agent_role": True,
		"human_approval_required": False,
	})
	stream = evaluate_capability_rules({
		"operation": "validate_cache_lifecycle_batch",
		"event_stream": "custom-broker",
	})

	assert agent["decision"] == "deny"
	assert {action["reason"] for action in agent["actions"]} == {
		"unsupported_cache_agent_runtime",
		"unsupported_cache_agent_role",
		"cache_agent_scope_required",
		"cache_agent_owner_required",
		"cache_agent_purpose_required",
		"cache_agent_contribution_disclosure_required",
		"cache_agent_human_approval_required",
	}
	assert stream["decision"] == "deny"
	assert stream["actions"][0]["reason"] == "bytewax_cache_stream_required"


def test_rule_engine_preserves_privileged_cache_agent_review_state():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_cache_agent",
		"agent_runtime_supported": True,
		"agent_role_supported": True,
		"agent_scope_present": True,
		"agent_owner_present": True,
		"agent_purpose_present": True,
		"contribution_disclosed": True,
		"privileged_agent_role": True,
		"human_approval_required": False,
	})

	assert result["decision"] == "require_review"
	assert result["matched_rules"] == ["cache_agent_privileged_role_requires_human_approval"]
	assert result["actions"][0]["required_action"] == "require_human_approval_for_agent"


def test_governance_service_enforces_namespace_entry_warming_and_eviction_lifecycle():
	service = CacheGovernanceService("tenant-cache")
	namespace = service.create_namespace(
		tenant_id="tenant-cache",
		namespace="orders",
		owner="platform",
		data_classification="regulated",
		max_ttl_seconds=900,
		encryption_required=True,
	)

	denied_entry = service.write_entry(
		tenant_id="tenant-cache",
		namespace="orders",
		key="order:1001",
		value_ref="memory://orders/order:1001",
		producer="orders-api",
		ttl_seconds=300,
		data_classification="regulated",
		encrypted=False,
	)
	allowed_entry = service.write_entry(
		tenant_id="tenant-cache",
		namespace="orders",
		key="order:1002",
		value_ref="memory://orders/order:1002",
		producer="orders-api",
		ttl_seconds=300,
		data_classification="regulated",
		encrypted=True,
	)
	read_result = service.read_entry(
		tenant_id="tenant-cache",
		namespace="orders",
		key="order:1002",
	)
	warming = service.request_warming_plan(
		tenant_id="tenant-cache",
		namespace="orders",
		source_name="orders-db",
		key_count=25000,
		requester="platform",
		reason="morning order dashboard",
		source_registered=True,
	)
	warming_status = warming.status
	warming_rules = list(warming.matched_rules)
	unknown_warming = service.request_warming_plan(
		tenant_id="tenant-cache",
		namespace="missing",
		source_name="orders-db",
		key_count=10,
		requester="platform",
		reason="missing namespace should fail",
		source_registered=True,
	)
	denied_warming_review = service.decide_warming_plan(
		plan_id=warming.plan_id,
		reviewer="platform",
		decision="approved",
		notes="same reviewer should fail",
	)
	denied_warming_status = denied_warming_review.status
	approved_warming = service.decide_warming_plan(
		plan_id=warming.plan_id,
		reviewer="cache-sre",
		decision="approved",
		notes="source is registered and review limit is understood",
	)
	service.create_namespace(
		tenant_id="tenant-cache",
		namespace="disabled",
		owner="platform",
		status="disabled",
	)
	disabled_warming = service.request_warming_plan(
		tenant_id="tenant-cache",
		namespace="disabled",
		source_name="orders-db",
		key_count=10,
		requester="platform",
		reason="should be denied",
		source_registered=True,
	)
	ttl_review_entry = service.write_entry(
		tenant_id="tenant-cache",
		namespace="orders",
		key="order:long",
		value_ref="memory://orders/order:long",
		producer="orders-api",
		ttl_seconds=1200,
		data_classification="regulated",
		encrypted=True,
	)
	review = service.request_eviction_review(
		tenant_id="tenant-cache",
		namespace="orders",
		requester="platform",
		memory_utilization_percent=94,
		proposed_action="evict cold entries",
		reason="tenant memory pressure",
	)
	denied_review = service.decide_eviction_review(
		review_id=review.review_id,
		reviewer="platform",
		decision="approved",
		notes="same reviewer should fail",
	)
	denied_status = denied_review.status
	denied_rules = list(denied_review.matched_rules)
	approved_review = service.decide_eviction_review(
		review_id=review.review_id,
		reviewer="cache-sre",
		decision="approved",
		notes="cold entries are backed by source of truth",
	)
	review_agent = service.register_cache_agent(
		tenant_id="tenant-cache",
		agent_id="eviction-agent-review",
		name="Eviction Agent Review",
		runtime="codex",
		role="eviction-reviewer",
		scope="eviction review",
		owner="platform",
		purpose="review memory pressure decisions",
		contribution_disclosed=True,
		human_approval_required=False,
	)
	cache_agent = service.register_cache_agent(
		tenant_id="tenant-cache",
		agent_id="warming-agent",
		name="Warming Agent",
		runtime="claude-code",
		role="warming-reviewer",
		scope="warming plan review",
		owner="platform",
		purpose="review warming plans",
		contribution_disclosed=True,
		human_approval_required=True,
	)
	lifecycle_batch = service.validate_cache_lifecycle_batch(
		tenant_id="tenant-cache",
		event_stream="ByteWax",
		mutation_count=3,
	)
	with pytest.raises(PermissionError, match="bytewax_cache_stream_required"):
		service.validate_cache_lifecycle_batch(
			tenant_id="tenant-cache",
			event_stream="custom-broker",
			mutation_count=1,
		)
	with pytest.raises(ValueError, match="cache_lifecycle_batch_empty"):
		service.validate_cache_lifecycle_batch(
			tenant_id="tenant-cache",
			event_stream="bytewax",
			mutation_count=0,
		)

	assert namespace.namespace == "orders"
	assert denied_entry.status == "denied"
	assert "regulated_entry_requires_encryption" in denied_entry.matched_rules
	assert allowed_entry.status == "active"
	assert read_result["hit"] is True
	assert warming_status == "pending_review"
	assert "warming_batch_limit_requires_review" in warming_rules
	assert unknown_warming.status == "denied"
	assert "warming_requires_namespace" in unknown_warming.matched_rules
	assert denied_warming_status == "review_denied"
	assert approved_warming.status == "approved"
	assert disabled_warming.status == "denied"
	assert "disabled_namespace_blocks_cache_warming" in disabled_warming.matched_rules
	assert ttl_review_entry.status == "pending_review"
	assert "ttl_above_namespace_limit_requires_review" in ttl_review_entry.matched_rules
	assert ttl_review_entry.policy_decision == "require_review"
	assert ttl_review_entry.review_reasons == ["ttl_review_required"]
	assert denied_status == "review_denied"
	assert "eviction_review_requires_independent_reviewer" in denied_rules
	assert approved_review.status == "approved"
	assert approved_review.policy_decision == "allow"
	assert review_agent.status == "pending_review"
	assert review_agent.policy_decision == "require_review"
	assert review_agent.review_reasons == ["cache_agent_human_approval_required"]
	assert cache_agent.runtime == "claude_code"
	assert cache_agent.role == "warming_reviewer"
	assert cache_agent.human_approval_required is True
	assert lifecycle_batch.event_stream == "bytewax"
	assert lifecycle_batch.required_processor == "bytewax"
	assert lifecycle_batch.accepted is True
	assert lifecycle_batch.policy_decision == "allow"
	denied_batch = [
		item for item in service.list_records("lifecycle_batches", "tenant-cache")
		if item["status"] == "denied"
	][0]
	assert denied_batch["policy_decision"] == "deny"
	assert denied_batch["review_reasons"] == ["bytewax_cache_stream_required"]
	assert service.dashboard_summary("tenant-cache")["cache_agent_count"] == 2
	assert service.dashboard_summary("tenant-cache")["pending_cache_agent_review_count"] == 1
	assert service.dashboard_summary("tenant-cache")["lifecycle_batch_count"] == 2
	assert service.dashboard_summary("tenant-cache")["denied_lifecycle_batch_count"] == 1
	assert service.dashboard_summary("tenant-cache")["active_entry_count"] == 1
	assert service.dashboard_summary("tenant-cache")["denied_entry_count"] == 1
	assert service.dashboard_summary("tenant-cache")["pending_review_count"] >= 2
	assert {
		record["status"]
		for record in service.list_records("entries", "tenant-cache")
	} >= {"active", "denied", "pending_review"}

	deleted = service.delete_entry(
		tenant_id="tenant-cache",
		namespace="orders",
		key="order:1002",
		actor="orders-api",
	)
	assert deleted["deleted"] is True
	assert {
		record["status"]
		for record in service.list_records("entries", "tenant-cache")
	} >= {"invalidated", "denied", "pending_review"}


def test_generated_view_models_are_operable():
	service = CacheGovernanceService("tenant-cache")
	service.create_namespace(
		tenant_id="tenant-cache",
		namespace="profiles",
		owner="identity",
	)
	service.request_warming_plan(
		tenant_id="tenant-cache",
		namespace="profiles",
		source_name="identity-db",
		key_count=25,
		requester="identity",
		reason="login path",
		source_registered=True,
	)
	service.register_cache_agent(
		tenant_id="tenant-cache",
		agent_id="policy-agent",
		name="Policy Agent",
		runtime="opencode",
		role="namespace-policy-reviewer",
		scope="namespace policy review",
		owner="identity",
		purpose="review cache namespace policy changes",
		contribution_disclosed=True,
	)
	service.validate_cache_lifecycle_batch(
		tenant_id="tenant-cache",
		event_stream="bytewax",
		mutation_count=1,
	)

	assert dashboard_model(service, "tenant-cache")["summary"]["namespace_count"] == 1
	assert namespace_inventory_model(service, "tenant-cache")["rows"][0]["namespace"] == "profiles"
	assert warming_console_model(service, "tenant-cache")["rows"][0]["status"] == "ready"
	assert eviction_review_model(service, "tenant-cache")["review_actions"] == ["approved", "rejected"]
	assert cache_agent_roster_model(service, "tenant-cache")["rows"][0]["runtime"] == "opencode"
	assert lifecycle_batch_model(service, "tenant-cache")["rows"][0]["accepted"] is True
	assert adapter_health_model("tenant-cache")["default_backend"] == "memory"
	assert "redis" in adapter_health_model("tenant-cache")["supported_backends"]
	assert settings_model("tenant-cache")["review_evidence"]["pending_queues"]


def test_governance_service_rejects_invalid_runtime_inputs():
	service = CacheGovernanceService("tenant-cache")
	service.create_namespace(
		tenant_id="tenant-cache",
		namespace="profiles",
		owner="identity",
		allowed_tiers=["memory"],
	)

	with pytest.raises(ValueError, match="ttl_seconds"):
		service.write_entry(
			tenant_id="tenant-cache",
			namespace="profiles",
			key="profile:1",
			value_ref="memory://profiles/profile:1",
			producer="identity",
			ttl_seconds=0,
		)
	with pytest.raises(ValueError, match="not allowed"):
		service.write_entry(
			tenant_id="tenant-cache",
			namespace="profiles",
			key="profile:2",
			value_ref="memory://profiles/profile:2",
			producer="identity",
			tier="edge",
		)
	with pytest.raises(ValueError, match="key_count"):
		service.request_warming_plan(
			tenant_id="tenant-cache",
			namespace="profiles",
			source_name="identity-db",
			key_count=0,
			requester="identity",
			reason="login path",
			source_registered=True,
		)
	with pytest.raises(ValueError, match="memory_utilization_percent"):
		service.request_eviction_review(
			tenant_id="tenant-cache",
			namespace="profiles",
			requester="identity",
			memory_utilization_percent=101,
			proposed_action="evict cold entries",
			reason="invalid metric",
		)


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "cach_memory_fabric"
	assert registration["ui_components"]["warming"] == "/cach/warming"
	assert registration["ui_components"]["evictions"] == "/cach/evictions"
	assert registration["review_evidence"]["deny_behavior"] == "Denied CACH lifecycle batches persist evidence before PermissionError"
	assert "auth" in registration["dependencies"]
