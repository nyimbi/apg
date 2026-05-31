"""Regression coverage for the FEDL executable capability contract."""

import pytest

from capabilities.common.fedl import register_capability
from capabilities.common.fedl import api
from capabilities.common.fedl.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.fedl.service import FedlService
from capabilities.common.fedl.views import (
	aggregation_console_model,
	attestation_center_model,
	audit_timeline_model,
	dashboard_model,
	federation_agent_roster_model,
	lifecycle_batch_model,
	model_registry_model,
	release_console_model,
	security_console_model,
	update_queue_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-fed", {"privacy": {"max_privacy_epsilon": 4.0}})

	assert contract["capability"] == "fedl"
	assert contract["configuration"]["tenant_id"] == "tenant-fed"
	assert contract["configuration"]["privacy"]["max_privacy_epsilon"] == 4.0
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"federation",
		"participants",
		"privacy",
		"training",
		"aggregation",
		"model_release",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 38
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"federations",
		"participants",
		"attestation",
		"rounds",
		"updates",
		"aggregation",
		"privacy",
		"security",
		"models",
		"release",
		"agents",
		"lifecycle",
		"audit",
		"settings"
	}
	assert contract["provides"] == ["federated_learning", "privacy_preserving_training", "federation_agent_composition"]
	assert contract["requires"] == ["aicr", "mlcm", "encr", "mten"]
	assert contract["agents"]["first_class"] is True
	assert {"codex", "claude_code", "opencode", "pi"} <= set(contract["agents"]["supported_runtimes"])
	assert "model_release_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "federation_agent_batch" in contract["streaming"]["required_operations"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.FedlService"
	assert contract["ui"]["api_prefix"] == "/fedl/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "federation_topology" in contract["theme"]["components"]
	assert "federation_agent_roster" in contract["theme"]["components"]


def test_rule_engine_enforces_federated_learning_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "aggregate_updates",
		"participant_attested": False,
		"participant_count": 2,
		"secure_aggregation_enabled": False,
		"privacy_epsilon": 10.0,
		"privacy_review_recorded": False,
		"poisoning_signal_detected": True
	})
	join_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "join_federation",
		"participant_attested": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"secure_aggregation_required",
		"privacy_budget_requires_review",
		"poisoning_signal_blocks_round"
	}
	assert join_result["decision"] == "deny"
	assert join_result["matched_rules"] == ["participant_requires_attestation"]
	stream_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_round_events",
		"event_stream": "legacy_queue",
	})
	assert stream_result["matched_rules"] == ["bytewax_stream_required_for_round_events"]
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_federation_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
	})
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"federation_agent_runtime_supported",
		"federation_agent_role_supported",
		"federation_agent_requires_scope",
		"federation_agent_requires_owner",
		"federation_agent_requires_purpose",
		"federation_agent_requires_contribution_disclosure",
	}
	review_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_federation_agent",
		"agent_runtime_supported": True,
		"agent_role_supported": True,
		"scope_present": True,
		"owner_present": True,
		"purpose_present": True,
		"contribution_disclosed": True,
		"privileged_role": True,
		"human_approval_required": False,
	})
	assert review_result["decision"] == "require_review"
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_fedl_lifecycle_batch",
		"event_stream": "legacy_queue",
	})
	assert lifecycle_result["matched_rules"] == ["bytewax_fedl_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "fedl"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "fedl_privacy_mesh"
	assert registration["ui_components"]["participants"] == "/fedl/participants"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert "federation_agent_composition" in registration["capabilities"]
	assert "mlcm" in registration["dependencies"]
	assert "fedl:run_rounds" in registration["permissions"]
	assert "fedl:govern" in registration["permissions"]


def test_federated_learning_lifecycle_is_executable():
	service = FedlService()

	federation = service.create_federation(
		federation_id="fed-risk",
		tenant_id="tenant-fed",
		name="Risk Model Federation",
		coordinator="ml-platform",
		model_family="tabular-risk",
		objective_metric="auc",
		privacy_epsilon_limit=6.0,
		data_residency_regions=["ke", "za", "ng"],
	)
	for index, region in enumerate(["ke", "za", "ng"], start=1):
		service.register_participant(
			participant_id=f"node-{index}",
			tenant_id="tenant-fed",
			federation_id="fed-risk",
			name=f"Node {index}",
			region=region,
			contract_ref=f"contract-{index}",
			attested=True,
			compute_profile="gpu-small",
		)
	round_model = service.start_round(
		round_id="round-001",
		tenant_id="tenant-fed",
		federation_id="fed-risk",
		round_number=1,
		privacy_epsilon=2.0,
		approval_ref="approval-001",
		secure_aggregation=True,
	)
	for index in range(1, 4):
		service.submit_update(
			update_id=f"upd-{index}",
			tenant_id="tenant-fed",
			round_id="round-001",
			participant_id=f"node-{index}",
			payload={"weights": [index, index + 1], "bias": index / 10},
			sample_count=100 * index,
			quality_score=0.91,
		)
	aggregation = service.aggregate_updates(
		aggregation_id="agg-001",
		tenant_id="tenant-fed",
		round_id="round-001",
		secure_aggregation_enabled=True,
	)
	summary = service.dashboard_summary("tenant-fed")
	budget = service.privacy_budget_summary("tenant-fed")
	models = service.list_models("tenant-fed")
	release = service.release_model(
		release_id="release-001",
		tenant_id="tenant-fed",
		model_id=models[0]["id"],
		mlcm_model_ref="mlcm://risk-model",
		release_approval_ref="approval-release-001",
		privacy_review_ref="privacy-release-001",
		artifact_ref="artifact://fed-risk/round-001",
	)

	assert federation["status"] == "active"
	assert round_model["participant_ids"] == ["node-1", "node-2", "node-3"]
	assert aggregation["participant_count"] == 3
	assert aggregation["total_sample_count"] == 600
	assert aggregation["model_version"].startswith("fed-risk.r1.")
	assert models[0]["model_version"] == aggregation["model_version"]
	assert release["mlcm_model_ref"] == "mlcm://risk-model"
	assert summary == {
		"federation_count": 1,
		"participant_count": 3,
		"round_count": 1,
		"running_round_count": 0,
		"aggregated_round_count": 1,
		"accepted_update_count": 3,
		"quarantined_update_count": 0,
		"aggregation_count": 1,
		"model_count": 1,
		"release_count": 0,
		"federation_agent_count": 0,
		"pending_agent_review_count": 0,
		"lifecycle_batch_count": 0,
		"denied_lifecycle_batch_count": 0,
		"audit_event_count": 9,
	}
	agent = service.register_federation_agent(
		agent_id="fedl-reviewer",
		tenant_id="tenant-fed",
		name="FEDL Privacy Reviewer",
		runtime="codex",
		role="privacy_reviewer",
		scope="fed-risk releases",
		owner="ml-platform",
		purpose="Review privacy budget and release guardrails",
		human_approval_required=True,
	)
	batch = service.validate_fedl_lifecycle_batch(
		tenant_id="tenant-fed",
		event_stream="bytewax",
		mutation_count=3,
		operation="federation_agent_batch",
		batch_id="fedl-batch-001",
	)
	assert agent["runtime"] == "codex"
	assert agent["role"] == "privacy_reviewer"
	assert agent["status"] == "active"
	assert batch["required_processor"] == "bytewax"
	assert budget["spent_epsilon"] == 2.0
	assert dashboard_model(service, "tenant-fed")["summary"]["release_count"] == 1
	assert dashboard_model(service, "tenant-fed")["summary"]["federation_agent_count"] == 1
	assert attestation_center_model(service, "tenant-fed")["missing_attestation"] == []
	assert update_queue_model(service, "tenant-fed")["quarantined"] == []
	assert aggregation_console_model(service, "tenant-fed")["aggregations"][0]["id"] == "agg-001"
	assert model_registry_model(service, "tenant-fed")["models"][0]["id"] == models[0]["id"]
	assert release_console_model(service, "tenant-fed")["releases"][0]["id"] == "release-001"
	assert security_console_model(service, "tenant-fed")["poisoning_signals"] == []
	assert federation_agent_roster_model(service, "tenant-fed")["agents"][0]["id"] == "fedl-reviewer"
	assert lifecycle_batch_model(service, "tenant-fed")["batches"][0]["id"] == "fedl-batch-001"
	assert audit_timeline_model(service, "tenant-fed")["audit_events"]


def test_fedl_service_enforces_policy_guardrails():
	service = FedlService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_federation("fed", "", "No Tenant", "coord", "tabular", "auc", 4.0, ["ke"])
	with pytest.raises(PermissionError, match="data_residency_required"):
		service.create_federation("fed", "tenant-fed", "No Region", "coord", "tabular", "auc", 4.0, [])

	service.create_federation("fed", "tenant-fed", "Risk", "coord", "tabular", "auc", 4.0, ["ke"])
	with pytest.raises(PermissionError, match="participant_attestation_required"):
		service.register_participant("node-1", "tenant-fed", "fed", "Node 1", "ke", "contract", False)
	with pytest.raises(PermissionError, match="participant_contract_required"):
		service.register_participant("node-1", "tenant-fed", "fed", "Node 1", "ke", "", True)
	with pytest.raises(PermissionError, match="data_residency_required"):
		service.register_participant("node-1", "tenant-fed", "fed", "Node 1", "eu", "contract", True)
	service.register_participant("node-1", "tenant-fed", "fed", "Node 1", "ke", "contract-1", True)

	with pytest.raises(PermissionError, match="minimum_participants_required"):
		service.start_round("round-bad", "tenant-fed", "fed", 1, 1.0, "approval", True)
	service.register_participant("node-2", "tenant-fed", "fed", "Node 2", "ke", "contract-2", True)
	service.register_participant("node-3", "tenant-fed", "fed", "Node 3", "ke", "contract-3", True)
	with pytest.raises(PermissionError, match="round_approval_required"):
		service.start_round("round-no-approval", "tenant-fed", "fed", 1, 1.0, "", True)
	with pytest.raises(PermissionError, match="privacy_budget_review_required"):
		service.start_round("round-review", "tenant-fed", "fed", 1, 9.0, "approval", True, privacy_review_recorded=False)
	with pytest.raises(PermissionError, match="privacy_budget_exceeds_federation_limit"):
		service.start_round("round-limit", "tenant-fed", "fed", 1, 5.0, "approval", True)

	service.start_round("round-ok", "tenant-fed", "fed", 1, 2.0, "approval", True)
	with pytest.raises(PermissionError, match="sample_count_required"):
		service.submit_update("upd-bad", "tenant-fed", "round-ok", "node-1", {"weights": [1]}, 0, 0.9)
	service.submit_update("upd-1", "tenant-fed", "round-ok", "node-1", {"weights": [1]}, 100, 0.9)
	service.submit_update("upd-2", "tenant-fed", "round-ok", "node-2", {"weights": [2]}, 100, 0.9)
	service.submit_update("upd-3", "tenant-fed", "round-ok", "node-3", {"weights": [3]}, 100, 0.9)
	with pytest.raises(PermissionError, match="secure_aggregation_required"):
		service.aggregate_updates("agg-bad", "tenant-fed", "round-ok", secure_aggregation_enabled=False)
	service.aggregate_updates("agg-ok", "tenant-fed", "round-ok", secure_aggregation_enabled=True)

	service.start_round("round-poison", "tenant-fed", "fed", 2, 2.0, "approval", True)
	service.submit_update("upd-p1", "tenant-fed", "round-poison", "node-1", {"weights": [1]}, 100, 0.2)
	service.submit_update("upd-p2", "tenant-fed", "round-poison", "node-2", {"weights": [2]}, 100, 0.9)
	service.submit_update("upd-p3", "tenant-fed", "round-poison", "node-3", {"weights": [3]}, 100, 0.9)
	with pytest.raises(PermissionError, match="poisoning_signal_detected"):
		service.aggregate_updates("agg-poison", "tenant-fed", "round-poison", secure_aggregation_enabled=True)

	model = service.list_models("tenant-fed")[0]
	with pytest.raises(PermissionError, match="mlcm_model_ref_required"):
		service.release_model("release-no-mlcm", "tenant-fed", model["id"], "", "approval", "privacy")
	with pytest.raises(PermissionError, match="model_release_approval_required"):
		service.release_model("release-no-approval", "tenant-fed", model["id"], "mlcm://fed", "", "privacy")
	with pytest.raises(PermissionError, match="model_release_privacy_review_required"):
		service.release_model("release-no-privacy", "tenant-fed", model["id"], "mlcm://fed", "approval", "")
	with pytest.raises(PermissionError, match="federation_retirement_review_required"):
		service.retire_federation("fed", "tenant-fed", "", "coord")
	retired = service.retire_federation("fed", "tenant-fed", "impact-review", "coord")
	assert retired["status"] == "retired"

	with pytest.raises(PermissionError, match="unsupported_federation_agent_runtime"):
		service.register_federation_agent("agent-bad-runtime", "tenant-fed", "Bad Runtime", "unknown", "federation_steward", "fed", "owner", "purpose")
	with pytest.raises(PermissionError, match="federation_agent_scope_required"):
		service.register_federation_agent("agent-no-scope", "tenant-fed", "No Scope", "codex", "federation_steward", "", "owner", "purpose")
	agent = service.register_federation_agent("agent-review", "tenant-fed", "Review Agent", "claude-code", "privacy reviewer", "fed", "owner", "purpose")
	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "privacy_reviewer"
	assert agent["status"] == "pending_review"
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_fedl_lifecycle_batch("tenant-fed", "legacy_queue", 1)
	batch = service.validate_fedl_lifecycle_batch("tenant-fed", "bytewax", 1, "privacy_budget_batch")
	assert batch["accepted"] is True


def test_fedl_api_exposes_agent_and_lifecycle_helpers():
	tenant_id = "tenant-fedl-api"

	api.create_federation({
		"id": "fed-api",
		"tenant_id": tenant_id,
		"name": "API Federation",
		"coordinator": "api-coordinator",
		"model_family": "tabular",
		"objective_metric": "auc",
		"data_residency_regions": ["ke"],
	})
	agent = api.register_federation_agent({
		"id": "api-agent",
		"tenant_id": tenant_id,
		"name": "API FEDL Agent",
		"runtime": "opencode",
		"role": "federation_steward",
		"scope": "fed-api",
		"owner": "api-owner",
		"purpose": "Govern generated-app federation changes",
	})
	batch = api.validate_fedl_lifecycle_batch({
		"id": "api-batch",
		"tenant_id": tenant_id,
		"event_stream": "bytewax",
		"mutation_count": 2,
		"operation": "federation_agent_batch",
	})

	assert agent["runtime"] == "opencode"
	assert batch["status"] == "accepted"
	assert api.list_federation_agents(tenant_id)[0]["id"] == "api-agent"
	assert api.list_lifecycle_batches(tenant_id)[0]["id"] == "api-batch"
