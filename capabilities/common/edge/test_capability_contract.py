"""Regression coverage for the EDGE executable capability contract."""

import pytest

from capabilities.common.edge import register_capability
from capabilities.common.edge.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.edge.service import EdgeService
from capabilities.common.edge.views import dashboard_model, node_manager_model, sync_monitor_model, workload_console_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-edge", {"sync": {"max_offline_hours": 24}})

	assert contract["capability"] == "edge"
	assert contract["configuration"]["tenant_id"] == "tenant-edge"
	assert contract["configuration"]["sync"]["max_offline_hours"] == 24
	assert contract["configuration_schema"]["required"] == ["tenant_id", "nodes", "workloads", "sync", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "edge_operations_console"


def test_rule_engine_enforces_edge_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_node", "node_attested": False, "edge_connection": True, "secure_transport": False, "offline_hours": 100, "offline_review_recorded": False})
	deploy_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "deploy_workload", "artifact_signed": False})
	sync_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "sync_state", "conflict_policy_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "node_requires_attestation", "edge_transport_requires_security", "long_offline_window_requires_review"}
	assert deploy_result["matched_rules"] == ["workload_requires_signed_artifact"]
	assert sync_result["matched_rules"] == ["sync_requires_conflict_policy"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "edge"
	assert "dist" in registration["dependencies"]
	assert registration["ui_components"]["nodes"] == "/edge/nodes"
	assert "edge:deploy_workloads" in registration["permissions"]


def test_service_runs_edge_node_workload_deployment_and_sync_lifecycle():
	service = EdgeService()

	node = service.register_node(
		node_id="node-1",
		tenant_id="tenant-edge",
		name="Plant Gateway",
		owner="edge-ops",
		node_type="gateway",
		location={"site": "plant-a", "zone": "line-1"},
		location_policy="ke-site-policy",
		attested=True,
		capacity={"cpu": 8, "memory": 16384, "storage": 512},
		capabilities=["sensor_aggregation", "local_inference"],
	)
	fleet = service.create_fleet(
		fleet_id="fleet-1",
		tenant_id="tenant-edge",
		name="Plant Fleet",
		owner="edge-ops",
		policy_version="2026.05",
		node_ids=["node-1"],
	)
	workload = service.register_workload(
		workload_id="wl-1",
		tenant_id="tenant-edge",
		name="Line Monitor",
		version="1.0.0",
		owner="automation",
		artifact_payload={"image": "line-monitor:1.0.0"},
		artifact_signed=True,
		deployment_policy="signed-canary",
		resource_quota={"cpu": 2, "memory": 1024, "storage": 10},
	)
	deployment = service.deploy_workload(
		deployment_id="dep-1",
		tenant_id="tenant-edge",
		workload_id="wl-1",
		node_id="node-1",
		deployed_by="release-manager",
	)
	sync = service.sync_state(
		sync_id="sync-1",
		tenant_id="tenant-edge",
		node_id="node-1",
		workload_id="wl-1",
		conflict_policy="last_writer_requires_review",
		cache_policy="bounded_local_cache",
		offline_hours=4,
		event_count=25,
	)

	assert node["attested"] is True
	assert fleet["node_ids"] == ["node-1"]
	assert workload["artifact_signed"] is True
	assert deployment["status"] == "deployed"
	assert sync["status"] == "replayed"
	assert service.node_pressure("node-1", "tenant-edge")["pressure"]["cpu"] == 0.25
	assert service.dashboard_summary("tenant-edge")["deployment_count"] == 1
	assert len(service.list_audit_events("tenant-edge")) >= 5


def test_service_enforces_edge_guardrails():
	service = EdgeService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_node("node-x", "", "No Tenant", "owner", "gateway", {}, "policy", True)

	with pytest.raises(PermissionError, match="node_attestation_required"):
		service.register_node("node-x", "tenant-edge", "Unsigned", "owner", "gateway", {}, "policy", False)

	service.register_node(
		node_id="node-1",
		tenant_id="tenant-edge",
		name="Plant Gateway",
		owner="edge-ops",
		node_type="gateway",
		location={"site": "plant-a"},
		location_policy="site-policy",
		attested=True,
		capacity={"cpu": 1, "memory": 512, "storage": 8},
	)

	with pytest.raises(PermissionError, match="artifact_signature_required"):
		service.register_workload(
			workload_id="wl-unsigned",
			tenant_id="tenant-edge",
			name="Unsigned",
			version="1.0.0",
			owner="automation",
			artifact_payload={},
			artifact_signed=False,
			deployment_policy="signed-only",
			resource_quota={"cpu": 1},
		)

	service.register_workload(
		workload_id="wl-heavy",
		tenant_id="tenant-edge",
		name="Heavy",
		version="1.0.0",
		owner="automation",
		artifact_payload={"image": "heavy:1.0.0"},
		artifact_signed=True,
		deployment_policy="signed-only",
		resource_quota={"cpu": 2},
	)
	with pytest.raises(PermissionError, match="resource_quota_exceeds_node_capacity"):
		service.deploy_workload("dep-heavy", "tenant-edge", "wl-heavy", "node-1", "release-manager")


def test_service_tracks_offline_review_and_view_models():
	service = EdgeService()
	service.register_node(
		node_id="node-1",
		tenant_id="tenant-edge",
		name="Plant Gateway",
		owner="edge-ops",
		node_type="gateway",
		location={"site": "plant-a"},
		location_policy="site-policy",
		attested=True,
		secure_transport=True,
	)
	service.register_workload(
		workload_id="wl-1",
		tenant_id="tenant-edge",
		name="Line Monitor",
		version="1.0.0",
		owner="automation",
		artifact_payload={"image": "line-monitor:1.0.0"},
		artifact_signed=True,
		deployment_policy="signed-only",
		resource_quota={"cpu": 1},
	)
	session = service.sync_state(
		sync_id="sync-1",
		tenant_id="tenant-edge",
		node_id="node-1",
		workload_id="wl-1",
		conflict_policy="manual_review",
		cache_policy="bounded_cache",
		offline_hours=96,
		event_count=50,
		conflicts=["asset-state"],
	)
	reviewed = service.review_offline_window("sync-1", "tenant-edge", "ops-reviewer")

	assert session["review_required"] is True
	assert session["status"] == "review_required"
	assert reviewed["review_required"] is False
	assert reviewed["reviewed_by"] == "ops-reviewer"
	assert reviewed["status"] == "conflict_pending"
	assert dashboard_model(service, "tenant-edge")["summary"]["review_required_sync_count"] == 0
	assert node_manager_model(service, "tenant-edge")["nodes"][0]["id"] == "node-1"
	assert workload_console_model(service, "tenant-edge")["workloads"][0]["id"] == "wl-1"
	assert sync_monitor_model(service, "tenant-edge")["conflicts"][0]["id"] == "sync-1"
