"""Regression coverage for the FEDL executable capability contract."""

import pytest

from capabilities.common.fedl import register_capability
from capabilities.common.fedl.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.fedl.service import FedlService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-fed", {"privacy": {"max_privacy_epsilon": 4.0}})

	assert contract["capability"] == "fedl"
	assert contract["configuration"]["tenant_id"] == "tenant-fed"
	assert contract["configuration"]["privacy"]["max_privacy_epsilon"] == 4.0
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"federation",
		"privacy",
		"training",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"federations",
		"participants",
		"rounds",
		"privacy",
		"security",
		"models",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/fedl/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "federation_topology" in contract["theme"]["components"]


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
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"secure_aggregation_required",
		"privacy_budget_requires_review",
		"poisoning_signal_blocks_round"
	}
	assert join_result["decision"] == "deny"
	assert join_result["matched_rules"] == ["participant_requires_attestation"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "fedl"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "fedl_privacy_mesh"
	assert registration["ui_components"]["participants"] == "/fedl/participants"
	assert "mlcm" in registration["dependencies"]
	assert "fedl:run_rounds" in registration["permissions"]


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

	assert federation["status"] == "active"
	assert round_model["participant_ids"] == ["node-1", "node-2", "node-3"]
	assert aggregation["participant_count"] == 3
	assert aggregation["total_sample_count"] == 600
	assert aggregation["model_version"].startswith("fed-risk.r1.")
	assert models[0]["model_version"] == aggregation["model_version"]
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
		"audit_event_count": 9,
	}
	assert budget["spent_epsilon"] == 2.0


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

	service.start_round("round-poison", "tenant-fed", "fed", 2, 2.0, "approval", True)
	service.submit_update("upd-p1", "tenant-fed", "round-poison", "node-1", {"weights": [1]}, 100, 0.2)
	service.submit_update("upd-p2", "tenant-fed", "round-poison", "node-2", {"weights": [2]}, 100, 0.9)
	service.submit_update("upd-p3", "tenant-fed", "round-poison", "node-3", {"weights": [3]}, 100, 0.9)
	with pytest.raises(PermissionError, match="poisoning_signal_detected"):
		service.aggregate_updates("agg-poison", "tenant-fed", "round-poison", secure_aggregation_enabled=True)
