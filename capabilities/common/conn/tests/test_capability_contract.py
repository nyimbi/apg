"""Regression coverage for the CONN executable capability contract."""

import pytest

from capabilities.common.conn import register_capability
from capabilities.common.conn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.conn.conn_runtime import ConnService
from capabilities.common.conn.view_models import (
	connection_workbench_model,
	connector_agent_roster_model,
	connector_catalog_model,
	dashboard_model,
	flow_designer_model,
	lifecycle_batch_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-a")

	assert contract["capability"] == "conn"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"connectors",
		"connections",
		"flows",
		"sync",
		"security",
		"quality",
		"governance",
		"observability",
		"adapters",
		"agents",
		"streaming",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 39
	assert contract["provides"] == ["connector_management", "connection_orchestration", "connector_agent_composition"]
	assert contract["requires"] == ["apig", "auth", "encr", "audl"]
	assert contract["agents"]["first_class"] is True
	assert "codex" in contract["agents"]["supported_runtimes"]
	assert "connector_reviewer" in contract["agents"]["supported_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "conn_runtime.ConnService"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"connectors",
		"connections",
		"designer",
		"sync_runs",
		"quality",
		"lineage",
		"marketplace",
		"security",
		"audit",
		"agents",
		"lifecycle",
		"rules",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "sync_monitor" in contract["theme"]["components"]
	assert "connector_agent_roster" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_activation_and_flow_creation():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_flow",
		"source_connection_active": False,
		"target_connection_active": False,
		"mapping_present": False,
		"lineage_enabled": False,
		"quality_gate_present": False,
		"pii_detected": True,
		"pii_policy_attached": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"flow_requires_source_connection",
		"flow_requires_target_connection",
		"flow_requires_mapping",
		"flow_requires_lineage",
		"flow_requires_quality_gate",
		"pii_requires_policy",
	}


def test_rule_engine_enforces_connector_agent_and_bytewax_guardrails():
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_connector_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	stream_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_conn_lifecycle_batch",
		"event_stream": "kafka",
	})

	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"connector_agent_runtime_supported",
		"connector_agent_role_supported",
		"connector_agent_requires_scope",
		"connector_agent_requires_owner",
		"connector_agent_requires_purpose",
		"connector_agent_requires_contribution_disclosure",
		"connector_agent_privileged_role_requires_human_approval",
	}
	assert stream_result["decision"] == "deny"
	assert stream_result["matched_rules"] == ["bytewax_conn_stream_required"]


def test_conn_runtime_registers_connections_flows_and_syncs():
	service = ConnService()
	service.register_connector("tap-postgres", "tenant-a", "Postgres", "singer", "singer_taps/tap_postgres", "sha256:abc", "platform")
	service.register_connector("target-warehouse", "tenant-a", "Warehouse", "singer", "singer_taps/target_warehouse", "sha256:def", "platform")
	service.register_connection("orders-db", "tenant-a", "Orders DB", "tap-postgres", "data", "development", credential_vault_ref="keym://orders")
	service.register_connection("warehouse", "tenant-a", "Warehouse", "target-warehouse", "data", "development", credential_vault_ref="keym://warehouse")
	service.record_connection_test("tenant-a", "orders-db", True)
	service.record_connection_test("tenant-a", "warehouse", True)
	source = service.activate_connection("tenant-a", "orders-db", secret_rotation_recorded=True)
	target = service.activate_connection("tenant-a", "warehouse", secret_rotation_recorded=True)
	flow = service.create_flow("orders-flow", "tenant-a", "Orders Flow", "orders-db", "warehouse", "data", "maps/orders.json", quality_gate_ref="quality/orders")
	run = service.start_sync("run-1", "tenant-a", "orders-flow", batch_size=5000, monitoring_enabled=True)
	completed = service.complete_sync("tenant-a", "run-1", records_processed=42, quality_score=0.99)

	assert source["status"] == "active"
	assert target["status"] == "active"
	assert flow["status"] == "created"
	assert run["status"] == "running"
	assert completed["status"] == "completed"
	assert service.dashboard_summary("tenant-a")["flow_count"] == 1


def test_conn_runtime_blocks_missing_evidence():
	service = ConnService()

	with pytest.raises(PermissionError, match="connector_owner_required"):
		service.register_connector("tap-postgres", "tenant-a", "Postgres", "singer", "src", "sha256:abc", "")

	service.register_connector("tap-postgres", "tenant-a", "Postgres", "singer", "src", "sha256:abc", "platform")
	with pytest.raises(PermissionError, match="credential_vault_required"):
		service.register_connection("orders-db", "tenant-a", "Orders DB", "tap-postgres", "data", "development", credential_vault_ref="")

	service.register_connection("orders-db", "tenant-a", "Orders DB", "tap-postgres", "data", "development", credential_vault_ref="keym://orders")
	with pytest.raises(PermissionError, match="connection_test_required"):
		service.activate_connection("tenant-a", "orders-db", secret_rotation_recorded=True)


def test_conn_runtime_records_reviews_schedule_replay_and_retirement():
	service = ConnService()
	connector = service.register_connector("tap-custom", "tenant-a", "Custom", "singer", "local/custom", "sha256:abc", "platform", verified_source=False)
	assert connector["status"] == "pending_review"
	assert any(review["review_type"] == "marketplace" for review in service.list_reviews("tenant-a"))

	service.register_connector("tap-postgres", "tenant-a", "Postgres", "singer", "src", "sha256:def", "platform")
	service.register_connector("target-warehouse", "tenant-a", "Warehouse", "singer", "target", "sha256:ghi", "platform")
	for connection_id, connector_id in (("orders-db", "tap-postgres"), ("warehouse", "target-warehouse")):
		service.register_connection(connection_id, "tenant-a", connection_id, connector_id, "data", "production", credential_vault_ref=f"keym://{connection_id}")
		service.record_connection_test("tenant-a", connection_id, True)
		activation = service.activate_connection("tenant-a", connection_id, secret_rotation_recorded=True, activation_review_recorded=False)
		assert activation["status"] == "pending_review"
		service.activate_connection("tenant-a", connection_id, secret_rotation_recorded=True, activation_review_recorded=True)

	service.create_flow("orders-flow", "tenant-a", "Orders Flow", "orders-db", "warehouse", "data", "maps/orders.json", quality_gate_ref="quality/orders")
	with pytest.raises(PermissionError, match="large_batch_requires_monitoring"):
		service.start_sync("run-large", "tenant-a", "orders-flow", batch_size=20000, monitoring_enabled=False)
	with pytest.raises(PermissionError, match="unsupported_sync_mode"):
		service.start_sync("run-mode", "tenant-a", "orders-flow", mode="unsafe_copy")
	schema_run = service.start_sync("run-schema", "tenant-a", "orders-flow", batch_size=1000, schema_change_detected=True, schema_review_recorded=False)
	assert schema_run["status"] == "pending_review"
	assert any(review["review_type"] == "schema_change" for review in service.list_reviews("tenant-a"))
	with pytest.raises(PermissionError, match="sync_run_not_running"):
		service.complete_sync("tenant-a", "run-schema", records_processed=1, quality_score=1.0)
	with pytest.raises(PermissionError, match="timezone_required"):
		service.schedule_flow("tenant-a", "schedule-1", "orders-flow", "0 1 * * *", "")
	service.schedule_flow("tenant-a", "schedule-1", "orders-flow", "0 1 * * *", "Africa/Nairobi")
	with pytest.raises(PermissionError, match="idempotency_required"):
		service.replay_sync("tenant-a", "run-schema", "run-replay", "")
	service.replay_sync("tenant-a", "run-schema", "run-replay", "idem-001")
	with pytest.raises(PermissionError, match="impact_review_required"):
		service.retire_connection("tenant-a", "orders-db", "data", False)
	retired = service.retire_connection("tenant-a", "orders-db", "data", True)
	assert retired["status"] == "retired"


def test_conn_runtime_enforces_connector_runtime_allowlist():
	service = ConnService()

	with pytest.raises(PermissionError, match="unsupported_connector_runtime"):
		service.register_connector("tap-unsafe", "tenant-a", "Unsafe", "shell", "src", "sha256:abc", "platform")


def test_conn_runtime_governs_agents_and_lifecycle_batches():
	service = ConnService()

	with pytest.raises(PermissionError, match="unsupported_connector_agent_runtime"):
		service.register_connector_agent(
			agent_id="unknown-agent",
			tenant_id="tenant-a",
			name="Unknown Agent",
			runtime="unsupported",
			role="connector_reviewer",
			scope="connector package reviews",
			owner="platform",
			purpose="review connector packages",
		)

	pending = service.register_connector_agent(
		agent_id="review-agent",
		tenant_id="tenant-a",
		name="Review Agent",
		runtime="Claude Code",
		role="connector reviewer",
		scope="connector package reviews",
		owner="platform",
		purpose="review connector packages",
		human_approval_required=False,
	)
	active = service.register_connector_agent(
		agent_id="tap-steward",
		tenant_id="tenant-a",
		name="Tap Steward",
		runtime="codex",
		role="tap_steward",
		scope="local singer tap catalog",
		owner="integration-office",
		purpose="maintain local Singer tap metadata",
	)

	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_conn_lifecycle_batch("tenant-a", "kafka", 2)
	batch = service.validate_conn_lifecycle_batch("tenant-a", "bytewax", 4)
	summary = service.dashboard_summary("tenant-a")

	assert pending["status"] == "pending_review"
	assert pending["runtime"] == "claude_code"
	assert active["status"] == "active"
	assert batch["status"] == "accepted"
	assert summary["connector_agent_count"] == 2
	assert summary["lifecycle_batch_count"] == 2
	assert summary["denied_lifecycle_batch_count"] == 1


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "conn_integration_console"
	assert registration["ui_components"]["rules"] == "/conn/rules"
	assert "encr" in registration["dependencies"]
	assert "keym" in registration["optional_dependencies"]
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"


def test_generated_ui_models_are_composable():
	service = ConnService()
	service.register_connector("tap-postgres", "tenant-a", "Postgres", "singer", "src", "sha256:abc", "platform")
	dashboard = dashboard_model(service, "tenant-a")
	catalog = connector_catalog_model(service, "tenant-a")
	workbench = connection_workbench_model(service, "tenant-a")
	designer = flow_designer_model(service, "tenant-a")
	service.register_connector_agent(
		agent_id="tap-steward",
		tenant_id="tenant-a",
		name="Tap Steward",
		runtime="codex",
		role="tap_steward",
		scope="local singer tap catalog",
		owner="integration-office",
		purpose="maintain local Singer tap metadata",
	)
	service.validate_conn_lifecycle_batch("tenant-a", "bytewax", 1)
	agents = connector_agent_roster_model(service, "tenant-a")
	batches = lifecycle_batch_model(service, "tenant-a")

	assert dashboard["summary"]["connector_count"] == 1
	assert catalog["connectors"][0]["id"] == "tap-postgres"
	assert "register_connection" in workbench["actions"]
	assert designer["defaults"]["mapping_required"] is True
	assert agents["agents"][0]["runtime"] == "codex"
	assert batches["required_processor"] == "bytewax"
