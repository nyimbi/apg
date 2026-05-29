"""Regression coverage for the LOGT executable capability contract."""

import pytest

from capabilities.common.logt import register_capability
from capabilities.common.logt.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.logt.service import LogtService
from capabilities.common.logt.views import (
	analytics_model,
	dashboard_model,
	log_search_model,
	pipeline_manager_model,
	retention_center_model,
	trace_explorer_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-logs", {"tracing": {"span_retention_days": 7}})

	assert contract["capability"] == "logt"
	assert contract["configuration"]["tenant_id"] == "tenant-logs"
	assert contract["configuration"]["tracing"]["span_retention_days"] == 7
	assert contract["configuration_schema"]["required"] == ["tenant_id", "ingestion", "tracing", "privacy", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "logs", "traces", "spans", "pipelines", "retention", "analytics", "settings"}
	assert contract["theme"]["name"] == "logt_observability_console"


def test_rule_engine_enforces_logging_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_pipeline", "pipeline_owner_assigned": False, "sensitive_log_content": True, "redaction_applied": False, "query_window_hours": 240, "query_review_recorded": False})
	trace_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "ingest_trace", "trace_context_present": False})
	export_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "export_logs", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "pipeline_requires_owner", "sensitive_log_requires_redaction", "large_query_requires_review"}
	assert trace_result["matched_rules"] == ["trace_context_required"]
	assert export_result["matched_rules"] == ["log_export_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "logt"
	assert "moni" in registration["dependencies"]
	assert registration["ui_components"]["traces"] == "/logt/traces"
	assert "logt:query" in registration["permissions"]


def test_logt_lifecycle_is_executable():
	service = LogtService()

	retention = service.create_retention_policy(
		policy_id="retention-primary",
		tenant_id="tenant-logs",
		name="Primary retention",
		log_retention_days=45,
		span_retention_days=14,
	)
	pipeline = service.create_pipeline(
		pipeline_id="pipeline-api",
		tenant_id="tenant-logs",
		name="API pipeline",
		owner="sre-team",
		schema_ref="schema://logs/v1",
		event_bus_ref="mqeb://diagnostics",
		sampling_policy="head-based-10pct",
		retention_policy_id=retention["id"],
	)
	trace = service.ingest_trace(
		trace_record_id="trace-record-1",
		tenant_id="tenant-logs",
		pipeline_id=pipeline["id"],
		trace_id="trace-001",
		root_service="orders-api",
		operation="POST /orders",
		trace_context={"traceparent": "00-trace-001-span-root-01"},
	)
	root_span = service.record_span(
		span_record_id="span-root",
		tenant_id="tenant-logs",
		trace_id=trace["trace_id"],
		span_id="span-root",
		service_name="orders-api",
		operation="POST /orders",
		duration_ms=320,
	)
	slow_span = service.record_span(
		span_record_id="span-db",
		tenant_id="tenant-logs",
		trace_id=trace["trace_id"],
		span_id="span-db",
		parent_span_id=root_span["span_id"],
		service_name="orders-db",
		operation="insert order",
		duration_ms=1250,
	)
	log = service.ingest_log(
		log_id="log-order-created",
		tenant_id="tenant-logs",
		pipeline_id=pipeline["id"],
		service_name="orders-api",
		severity="info",
		message="order created",
		attributes={"order_id": "O-1001"},
		trace_id=trace["trace_id"],
		span_id=root_span["span_id"],
	)
	error_log = service.ingest_log(
		log_id="log-payment-error",
		tenant_id="tenant-logs",
		pipeline_id=pipeline["id"],
		service_name="payments-api",
		severity="error",
		message="payment declined after redaction",
		attributes={"order_id": "O-1001"},
		trace_id=trace["trace_id"],
		span_id=slow_span["span_id"],
		sensitive_log_content=True,
		redaction_applied=True,
	)
	query = service.search_logs(
		query_id="query-order",
		tenant_id="tenant-logs",
		query_text="order",
		requested_by="sre-user",
		query_window_hours=24,
	)
	large_query = service.search_logs(
		query_id="query-quarter",
		tenant_id="tenant-logs",
		query_text="api",
		requested_by="sre-user",
		query_window_hours=720,
		query_review_recorded=True,
	)
	export = service.export_logs(
		export_id="export-order",
		tenant_id="tenant-logs",
		export_type="incident_bundle",
		requested_by="sre-user",
		item_ids=[log["id"], error_log["id"], trace["id"], slow_span["id"]],
		approval_recorded=True,
		approval_ref="approval:incident-1001",
	)

	assert pipeline["owner"] == "sre-team"
	assert trace["sampling_policy"] == "head-based-10pct"
	assert slow_span["status"] == "slow"
	assert error_log["redaction_applied"] is True
	assert query["query"]["result_count"] == 1
	assert large_query["query"]["status"] == "complete"
	assert export["status"] == "approved"
	assert service.dashboard_summary("tenant-logs")["error_log_count"] == 1
	assert service.dashboard_summary("tenant-logs")["slow_span_count"] == 1
	assert service.service_map("tenant-logs")["service_count"] == 2
	assert dashboard_model(service, "tenant-logs")["summary"]["log_count"] == 2
	assert log_search_model(service, "tenant-logs")["queries"][0]["id"] == "query-order"
	assert trace_explorer_model(service, "tenant-logs")["service_map"]["edge_count"] == 1
	assert pipeline_manager_model(service, "tenant-logs")["pipelines"][0]["id"] == "pipeline-api"
	assert retention_center_model(service, "tenant-logs")["exports"][0]["id"] == "export-order"
	assert analytics_model(service, "tenant-logs")["slow_spans"][0]["id"] == "span-db"
	assert len(service.list_audit_events("tenant-logs")) >= 10


def test_logt_service_enforces_policy_guardrails():
	service = LogtService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_retention_policy(
			policy_id="retention-missing-tenant",
			tenant_id="",
			name="Missing tenant",
			log_retention_days=30,
		)

	service.create_retention_policy(
		policy_id="retention-main",
		tenant_id="tenant-logs",
		name="Main retention",
		log_retention_days=30,
	)

	with pytest.raises(PermissionError, match="pipeline_owner_required"):
		service.create_pipeline(
			pipeline_id="pipeline-no-owner",
			tenant_id="tenant-logs",
			name="No owner",
			owner="",
			schema_ref="schema://logs",
			event_bus_ref="mqeb://diagnostics",
			sampling_policy="head-based",
			retention_policy_id="retention-main",
		)

	with pytest.raises(PermissionError, match="schema_validation_required"):
		service.create_pipeline(
			pipeline_id="pipeline-no-schema",
			tenant_id="tenant-logs",
			name="No schema",
			owner="sre-team",
			schema_ref="",
			event_bus_ref="mqeb://diagnostics",
			sampling_policy="head-based",
			retention_policy_id="retention-main",
		)

	pipeline = service.create_pipeline(
		pipeline_id="pipeline-main",
		tenant_id="tenant-logs",
		name="Main",
		owner="sre-team",
		schema_ref="schema://logs",
		event_bus_ref="mqeb://diagnostics",
		sampling_policy="head-based",
		retention_policy_id="retention-main",
	)

	with pytest.raises(PermissionError, match="log_redaction_required"):
		service.ingest_log(
			log_id="log-sensitive",
			tenant_id="tenant-logs",
			pipeline_id=pipeline["id"],
			service_name="auth-api",
			severity="info",
			message="password leaked",
			sensitive_log_content=True,
			redaction_applied=False,
		)

	with pytest.raises(PermissionError, match="trace_context_required"):
		service.ingest_trace(
			trace_record_id="trace-missing-context",
			tenant_id="tenant-logs",
			pipeline_id=pipeline["id"],
			trace_id="trace-missing",
			root_service="orders-api",
			operation="GET /orders",
			trace_context={},
		)

	trace = service.ingest_trace(
		trace_record_id="trace-main",
		tenant_id="tenant-logs",
		pipeline_id=pipeline["id"],
		trace_id="trace-main",
		root_service="orders-api",
		operation="GET /orders",
		trace_context={"traceparent": "00-trace-main-root-01"},
	)
	span = service.record_span(
		span_record_id="span-main",
		tenant_id="tenant-logs",
		trace_id=trace["trace_id"],
		span_id="span-main",
		service_name="orders-api",
		operation="GET /orders",
		duration_ms=10,
	)
	log = service.ingest_log(
		log_id="log-main",
		tenant_id="tenant-logs",
		pipeline_id=pipeline["id"],
		service_name="orders-api",
		severity="info",
		message="order lookup",
	)
	service.create_retention_policy(
		policy_id="retention-other",
		tenant_id="other-tenant",
		name="Other tenant retention",
		log_retention_days=30,
	)
	other_pipeline = service.create_pipeline(
		pipeline_id="pipeline-other",
		tenant_id="other-tenant",
		name="Other tenant",
		owner="other-sre",
		schema_ref="schema://logs",
		event_bus_ref="mqeb://diagnostics",
		sampling_policy="head-based",
		retention_policy_id="retention-other",
	)
	other_log = service.ingest_log(
		log_id="log-other",
		tenant_id="other-tenant",
		pipeline_id=other_pipeline["id"],
		service_name="other-api",
		severity="info",
		message="other tenant event",
	)

	with pytest.raises(PermissionError, match="large_query_review_required"):
		service.search_logs(
			query_id="query-large",
			tenant_id="tenant-logs",
			query_text="orders",
			requested_by="sre-user",
			query_window_hours=240,
			query_review_recorded=False,
		)

	with pytest.raises(PermissionError, match="export_approval_required"):
		service.export_logs(
			export_id="export-no-approval",
			tenant_id="tenant-logs",
			export_type="logs",
			requested_by="sre-user",
			item_ids=[log["id"]],
			approval_recorded=False,
		)

	with pytest.raises(KeyError, match="trace_not_found"):
		service.record_span(
			span_record_id="span-cross-tenant",
			tenant_id="another-tenant",
			trace_id=trace["trace_id"],
			span_id="span-cross",
			service_name="orders-api",
			operation="GET /orders",
			duration_ms=1,
		)

	with pytest.raises(KeyError, match="diagnostic_export_item_not_found"):
		service.export_logs(
			export_id="export-cross-tenant",
			tenant_id="tenant-logs",
			export_type="logs",
			requested_by="sre-user",
			item_ids=[other_log["id"]],
			approval_recorded=True,
			approval_ref="approval:cross-tenant",
		)

	with pytest.raises(KeyError, match="diagnostic_export_item_not_found"):
		service.export_logs(
			export_id="export-missing-item",
			tenant_id="tenant-logs",
			export_type="logs",
			requested_by="sre-user",
			item_ids=["missing"],
			approval_recorded=True,
			approval_ref="approval:1",
		)

	assert span["status"] == "ok"
