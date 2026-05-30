"""Regression coverage for the ETLP executable capability contract."""

from capabilities.common.etlp import register_capability
from capabilities.common.etlp.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.etlp.service import ETLPLifecycleService
from capabilities.common.etlp.view_models import (
	adapter_health_model,
	dashboard_model,
	datasource_manager_model,
	execution_monitor_model,
	pipeline_workbench_model,
	publish_review_model,
	replay_console_model,
	schedule_console_model,
	settings_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-pipeline", {"pipelines": {"max_concurrent_executions": 4}})

	assert contract["capability"] == "etlp"
	assert contract["configuration"]["tenant_id"] == "tenant-pipeline"
	assert contract["configuration"]["pipelines"]["max_concurrent_executions"] == 4
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"pipelines",
		"datasources",
		"mappings",
		"processing",
		"quality",
		"governance",
		"optimization",
		"execution",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 21
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"pipelines",
		"designer",
		"field_mapper",
		"executions",
		"quality",
		"datasources",
		"schedules",
		"publish",
		"lineage",
		"replay",
		"adapters",
		"audit",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/etlp/api/v1"
	assert contract["ui"]["view_module"] == "view_models.py"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "field_mapping_canvas" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"


def test_rule_engine_enforces_pipeline_guardrails():
	execution_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "execute_pipeline",
		"owner_assigned": False,
		"environment": "production",
		"approval_recorded": False,
		"idempotency_key_present": False,
		"transformation_present": True,
		"lineage_emitted": False,
		"estimated_cost": 1500.0,
		"cost_review_recorded": False
	})
	publish_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_output",
		"quality_gate_passed": False,
		"quality_score": 62.0,
		"publish_approval_recorded": False,
	})
	replay_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "replay_execution",
		"reason_present": False,
		"replay_window_hours": 96,
		"replay_review_recorded": False,
	})

	assert execution_result["decision"] == "deny"
	assert set(execution_result["matched_rules"]) == {
		"tenant_context_required",
		"pipeline_execution_requires_owner",
		"production_execution_requires_approval",
		"execution_requires_idempotency_key",
		"lineage_required_for_transformations",
		"high_cost_execution_requires_review"
	}
	assert publish_result["decision"] == "deny"
	assert publish_result["matched_rules"] == [
		"publish_requires_quality_gate",
		"publish_requires_minimum_quality",
		"publish_requires_approval"
	]
	assert replay_result["decision"] == "deny"
	assert "replay_requires_reason" in replay_result["matched_rules"]
	assert "replay_window_requires_review" in replay_result["matched_rules"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "etlp_pipeline_console"
	assert registration["ui_components"]["field_mapper"] == "/etlp/field-mapper"
	assert "metadata" in registration["dependencies"]


def test_lifecycle_service_enforces_datasource_mapping_execution_and_publish_guardrails():
	service = ETLPLifecycleService()
	pipeline = service.register_pipeline(
		tenant_id="tenant-data",
		pipeline_id="customer-sync",
		name="Customer sync",
		mode="elt",
		owner="data-owner",
	)
	source = service.register_datasource(
		tenant_id="tenant-data",
		datasource_id="crm",
		name="CRM",
		datasource_type="api",
		owner="data-owner",
		secret_ref="secret://crm",
		approved=True,
	)
	target = service.register_datasource(
		tenant_id="tenant-data",
		datasource_id="warehouse",
		name="Warehouse",
		datasource_type="warehouse",
		owner="data-owner",
		secret_ref="secret://warehouse",
		approved=True,
	)
	denied_secret = service.register_datasource(
		tenant_id="tenant-data",
		datasource_id="bad",
		name="Bad source",
		datasource_type="api",
		owner="data-owner",
		secret_ref=None,
		approved=True,
		embedded_secret_present=True,
	)
	mapping = service.register_mapping(
		tenant_id="tenant-data",
		mapping_id="customer-map",
		pipeline_id=pipeline.pipeline_id,
		source_datasource_id=source.datasource_id,
		target_datasource_id=target.datasource_id,
		field_mappings=[{"source": "customer_id", "target": "customer_id"}],
		schema_validated=True,
		lineage_emitted=True,
	)
	blocked_execution = service.execute_pipeline(
		tenant_id="tenant-data",
		pipeline_id=pipeline.pipeline_id,
		environment="production",
		triggered_by="data-owner",
		idempotency_key=None,
		approval_recorded=False,
		estimated_cost=1200.0,
	)
	allowed_execution = service.execute_pipeline(
		tenant_id="tenant-data",
		pipeline_id=pipeline.pipeline_id,
		environment="production",
		triggered_by="data-owner",
		idempotency_key="run-001",
		approval_recorded=True,
		estimated_cost=1200.0,
		cost_review_recorded=True,
	)
	allowed_execution_initial_status = allowed_execution.status
	low_quality = service.assess_quality(
		tenant_id="tenant-data",
		execution_id=allowed_execution.execution_id,
		score=76.0,
		dimensions={"completeness": 76.0},
		assessor="quality-engine",
	)
	blocked_publish = service.publish_output(
		tenant_id="tenant-data",
		execution_id=allowed_execution.execution_id,
		requester="data-owner",
		publish_approval_recorded=True,
	)
	high_quality = service.assess_quality(
		tenant_id="tenant-data",
		execution_id=allowed_execution.execution_id,
		score=91.0,
		dimensions={"completeness": 91.0, "lineage": 90.0},
		assessor="quality-engine",
	)
	published = service.publish_output(
		tenant_id="tenant-data",
		execution_id=allowed_execution.execution_id,
		requester="data-owner",
		publish_approval_recorded=True,
		review_notes="Quality and lineage accepted.",
	)

	assert denied_secret.status == "denied"
	assert denied_secret.matched_rules == [
		"datasource_requires_secret_reference",
		"datasource_secrets_must_use_reference",
	]
	assert mapping.status == "active"
	assert blocked_execution.status == "denied"
	assert set(blocked_execution.matched_rules) == {
		"production_execution_requires_approval",
		"execution_requires_idempotency_key",
		"high_cost_execution_requires_review",
	}
	assert allowed_execution_initial_status == "queued"
	assert low_quality.gate_passed is False
	assert blocked_publish.status == "denied"
	assert "publish_requires_quality_gate" in blocked_publish.matched_rules
	assert high_quality.gate_passed is True
	assert published.status == "published"
	assert allowed_execution.status == "published"


def test_lifecycle_service_replay_retry_retire_and_view_models_are_composable():
	service = ETLPLifecycleService()
	pipeline = service.register_pipeline(
		tenant_id="tenant-data",
		pipeline_id="orders-sync",
		name="Orders sync",
		mode="batch",
		owner="ops-owner",
	)
	execution = service.execute_pipeline(
		tenant_id="tenant-data",
		pipeline_id=pipeline.pipeline_id,
		environment="staging",
		triggered_by="ops-owner",
		idempotency_key="orders-run-001",
	)
	retry = service.retry_execution(
		tenant_id="tenant-data",
		execution_id=execution.execution_id,
		retry_count=4,
		retry_review_recorded=False,
	)
	replay = service.replay_execution(
		tenant_id="tenant-data",
		execution_id=execution.execution_id,
		replay_type="backfill",
		reason=None,
		window_hours=96,
	)
	blocked_retire = service.retire_pipeline(
		tenant_id="tenant-data",
		pipeline_id=pipeline.pipeline_id,
		actor="ops-owner",
		impact_review_recorded=False,
	)
	schedule = service.schedule_pipeline(
		tenant_id="tenant-data",
		pipeline_id=pipeline.pipeline_id,
		environment="production",
		schedule="0 2 * * *",
		owner="ops-owner",
		schedule_review_recorded=False,
	)

	assert retry.status == "pending_review"
	assert retry.matched_rules == ["retry_limit_requires_review"]
	assert replay.status == "denied"
	assert "replay_requires_reason" in replay.matched_rules
	assert blocked_retire.status != "retired"
	assert blocked_retire.matched_rules == ["destructive_delete_requires_review"]
	assert schedule.status == "pending_review"
	assert schedule.matched_rules == ["production_schedule_requires_review"]
	assert dashboard_model(service, "tenant-data")["summary"]["pipeline_count"] == 1
	assert pipeline_workbench_model(service, "tenant-data")["rows"][0]["pipeline_id"] == "orders-sync"
	assert datasource_manager_model(service, "tenant-data")["columns"][0] == "datasource_id"
	assert execution_monitor_model(service, "tenant-data")["rows"][0]["execution_id"] == execution.execution_id
	assert schedule_console_model(service, "tenant-data")["rows"][0]["schedule"] == "0 2 * * *"
	assert publish_review_model(service, "tenant-data")["columns"][0] == "created_at"
	assert replay_console_model(service, "tenant-data")["rows"][0]["replay_type"] == "backfill"
	assert adapter_health_model("tenant-data")["event_stream"] == "bytewax"
	assert settings_model("tenant-data")["routes"]
