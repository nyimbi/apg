"""Regression coverage for the MONI executable capability contract."""

import pytest

from capabilities.common.moni import register_capability
from capabilities.common.moni.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.moni.service import MoniService
from capabilities.common.moni.view_models import (
	adapter_health_model,
	alert_center_model,
	dashboard_model,
	incident_model,
	lifecycle_batch_model,
	monitoring_agent_roster_model,
	remediation_model,
	signal_explorer_model,
	settings_model,
	source_inventory_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-signals",
		{"retention": {"metrics_days": 180}}
	)

	assert contract["capability"] == "moni"
	assert contract["configuration"]["tenant_id"] == "tenant-signals"
	assert contract["configuration"]["retention"]["metrics_days"] == 180
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"collection",
		"slo",
		"alerts",
		"incidents",
		"analytics",
		"retention",
		"remediation",
		"adapters",
		"agents",
		"streaming",
		"security",
		"ui",
		"theme"
	]
	assert contract["provides"] == [
		"observability_governance",
		"metrics_lifecycle",
		"monitoring_agent_composition",
		"review_evidence",
	]
	assert "monitoring_agents" in contract["review_evidence"]["pending_queues"]
	assert "policy_decision" in contract["review_evidence"]["policy_fields"]
	assert contract["requires"] == ["conf", "audl", "mqeb"]
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "slo_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["broker_core_dependency_allowed"] is False
	assert len(contract["rule_engine"]["rules"]) >= 16
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"sources",
		"metrics",
		"logs",
		"alerts",
		"traces",
		"slos",
		"incidents",
		"analytics",
		"rules",
		"remediation",
		"audit",
		"adapters",
		"agents",
		"lifecycle",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/moni/api/v1"
	assert contract["ui"]["view_module"] == "view_models.py"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "alert_correlation_stack" in contract["theme"]["components"]
	assert "monitoring_agent_roster" in contract["theme"]["components"]
	assert "bytewax_lifecycle_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_observability_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "ingest_metric",
		"source_present": False,
		"alert_severity": "critical",
		"notification_route_configured": False,
		"log_contains_pii": True,
		"pii_redacted": False,
		"metric_cardinality": 20000,
		"cardinality_exception_recorded": False,
		"environment": "production",
		"remediation_requested": True,
		"runbook_approved": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"metric_ingestion_requires_source",
		"critical_alert_requires_route",
		"pii_logs_blocked",
		"high_cardinality_metric_requires_review",
		"production_remediation_requires_runbook"
	}


def test_rule_engine_enforces_monitoring_agent_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_monitoring_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"agent_scope_present": False,
		"agent_owner_present": False,
		"agent_purpose_present": False,
		"contribution_disclosed": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"monitoring_agent_runtime_supported",
		"monitoring_agent_role_supported",
		"monitoring_agent_requires_scope",
		"monitoring_agent_requires_owner",
		"monitoring_agent_requires_purpose",
		"monitoring_agent_requires_contribution_disclosure"
	}


def test_rule_engine_preserves_privileged_monitoring_agent_review_state():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_monitoring_agent",
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
	assert result["matched_rules"] == ["monitoring_agent_privileged_role_requires_human_approval"]
	assert result["actions"][0]["required_action"] == "require_human_approval_for_agent"


def test_bytewax_lifecycle_rule_rejects_non_bytewax_streams():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_monitoring_lifecycle_batch",
		"event_stream": "legacy_broker",
	})

	assert result["decision"] == "deny"
	assert "bytewax_monitoring_stream_required" in result["matched_rules"]


def test_moni_service_governs_sources_signals_alerts_incidents_and_remediation():
	service = MoniService("tenant-signals")
	source = service.register_source(
		tenant_id="tenant-signals",
		source_id="orders-api",
		service_name="orders",
		environment="production",
		owner="platform",
		notification_route="pagerduty:orders",
	)
	metric = service.ingest_signal(
		tenant_id="tenant-signals",
		source_id="orders-api",
		signal_type="metric",
		name="orders.latency_ms",
		value=250,
		labels={"route": "/orders"},
		cardinality=250,
	)
	pii_log = service.ingest_signal(
		tenant_id="tenant-signals",
		source_id="orders-api",
		signal_type="log",
		name="orders.request",
		contains_pii=True,
		pii_redacted=False,
	)
	trace = service.ingest_signal(
		tenant_id="tenant-signals",
		source_id="orders-api",
		signal_type="trace",
		name="orders.trace",
		trace_id="trace-1",
	)
	high_cardinality = service.ingest_signal(
		tenant_id="tenant-signals",
		source_id="orders-api",
		signal_type="metric",
		name="orders.customer.metric",
		cardinality=20000,
	)
	slo = service.create_slo(
		tenant_id="tenant-signals",
		service_name="orders",
		objective="p95 latency below 300ms",
		threshold=300,
		window_minutes=60,
		owner="platform",
		notification_route="pagerduty:orders",
	)
	alert = service.create_alert(
		tenant_id="tenant-signals",
		source_id="orders-api",
		severity="critical",
		title="Orders latency burn",
		notification_route="pagerduty:orders",
		owner="platform",
	)
	remediation = service.request_remediation(
		tenant_id="tenant-signals",
		incident_id=alert.incident_id,
		requester="platform",
		environment="production",
		runbook_id="orders-scale-out",
		runbook_approved=True,
		proposed_action="scale orders workers",
		reason="latency burn",
	)
	denied_review = service.decide_remediation(
		request_id=remediation.request_id,
		reviewer="platform",
		decision="approved",
		notes="same reviewer should fail",
	)
	denied_status = denied_review.status
	approved = service.decide_remediation(
		request_id=remediation.request_id,
		reviewer="sre-lead",
		decision="approved",
		notes="approved runbook and capacity available",
	)
	review_agent = service.register_monitoring_agent(
		tenant_id="tenant-signals",
		agent_id="agent-review",
		name="Review Incident Agent",
		runtime="codex",
		role="incident_reviewer",
		scope="production incidents",
		owner="platform",
		purpose="review critical incidents",
	)
	agent = service.register_monitoring_agent(
		tenant_id="tenant-signals",
		agent_id="agent-slo",
		name="SLO Reviewer",
		runtime="Claude Code",
		role="slo reviewer",
		scope="orders service SLOs",
		owner="sre-lead",
		purpose="review SLO burn and alert route quality",
		human_approval_required=True,
	)
	batch = service.validate_monitoring_lifecycle_batch(
		tenant_id="tenant-signals",
		event_stream="bytewax",
		mutation_count=5,
	)

	assert source.source_id == "orders-api"
	assert metric.status == "accepted"
	assert pii_log.status == "denied"
	assert "pii_logs_blocked" in pii_log.matched_rules
	assert trace.status == "accepted"
	assert high_cardinality.status == "pending_review"
	assert "high_cardinality_metric_requires_review" in high_cardinality.matched_rules
	assert high_cardinality.policy_decision == "require_review"
	assert high_cardinality.review_reasons == ["cardinality_review_required"]
	assert slo.status == "active"
	assert alert.status == "open"
	assert alert.policy_decision == "allow"
	assert alert.incident_id in service.incidents
	assert denied_status == "review_denied"
	assert approved.status == "approved"
	assert approved.policy_decision == "allow"
	assert review_agent.status == "pending_review"
	assert review_agent.policy_decision == "require_review"
	assert review_agent.review_reasons == ["monitoring_agent_human_approval_required"]
	assert agent.runtime == "claude_code"
	assert agent.role == "slo_reviewer"
	assert batch.accepted is True
	assert batch.policy_decision == "allow"
	assert batch.required_processor == "bytewax"
	assert service.dashboard_summary("tenant-signals")["source_count"] == 1
	assert service.dashboard_summary("tenant-signals")["open_incident_count"] == 1
	assert service.dashboard_summary("tenant-signals")["monitoring_agent_count"] == 2
	assert service.dashboard_summary("tenant-signals")["pending_monitoring_agent_review_count"] == 1
	assert service.dashboard_summary("tenant-signals")["pending_review_count"] >= 2


def test_moni_service_fails_closed_for_missing_source_disabled_source_and_invalid_inputs():
	service = MoniService("tenant-signals")
	missing = service.ingest_signal(
		tenant_id="tenant-signals",
		source_id="missing",
		signal_type="metric",
		name="orders.latency_ms",
	)
	service.register_source(
		tenant_id="tenant-signals",
		source_id="disabled-api",
		service_name="disabled",
		environment="production",
		owner="platform",
		status="disabled",
	)
	disabled = service.ingest_signal(
		tenant_id="tenant-signals",
		source_id="disabled-api",
		signal_type="metric",
		name="disabled.metric",
	)
	bad_alert = service.create_alert(
		tenant_id="tenant-signals",
		source_id="disabled-api",
		severity="critical",
		title="No route",
	)
	bad_incident = service.create_incident(
		tenant_id="tenant-signals",
		title="No owner",
		severity="critical",
		owner=None,
		notification_route="pagerduty:platform",
	)
	with pytest.raises(ValueError, match="threshold"):
		service.create_slo(
			tenant_id="tenant-signals",
			service_name="orders",
			objective="bad",
			threshold=0,
			window_minutes=60,
			owner="platform",
			notification_route="pagerduty:orders",
		)

	assert missing.status == "denied"
	assert "signal_requires_registered_source" in missing.matched_rules
	assert disabled.status == "denied"
	assert "disabled_source_blocks_ingestion" in disabled.matched_rules
	assert bad_alert.status == "denied"
	assert "critical_alert_requires_route" in bad_alert.matched_rules
	assert "critical_alert_requires_owner" in bad_alert.matched_rules
	assert bad_incident.status == "denied"
	assert "critical_incident_requires_owner" in bad_incident.matched_rules
	with pytest.raises(ValueError, match="existing incident"):
		service.request_remediation(
			tenant_id="tenant-signals",
			incident_id="missing-incident",
			requester="platform",
			environment="production",
			runbook_id="orders-scale-out",
			runbook_approved=True,
			proposed_action="scale orders workers",
			reason="latency burn",
		)
	with pytest.raises(PermissionError, match="unsupported_monitoring_agent_runtime"):
		service.register_monitoring_agent(
			tenant_id="tenant-signals",
			agent_id="bad-agent",
			name="Bad Agent",
			runtime="unsupported",
			role="alert_reviewer",
			scope="alerts",
			owner="platform",
			purpose="review alerts",
			human_approval_required=True,
		)
	with pytest.raises(PermissionError, match="bytewax_monitoring_stream_required"):
		service.validate_monitoring_lifecycle_batch(
			tenant_id="tenant-signals",
			event_stream="legacy_broker",
			mutation_count=1,
		)
	denied_batch = [
		item for item in service.list_records("tenant-signals", "lifecycle_batches")
		if item["status"] == "denied"
	][0]
	assert denied_batch["policy_decision"] == "deny"
	assert denied_batch["review_reasons"] == ["bytewax_monitoring_stream_required"]


def test_view_models_expose_agent_and_lifecycle_surfaces():
	service = MoniService("tenant-signals")
	service.register_monitoring_agent(
		tenant_id="tenant-signals",
		agent_id="metric-quality",
		name="Metric Quality",
		runtime="opencode",
		role="metric_quality_reviewer",
		scope="metric naming and cardinality",
		owner="observability",
		purpose="review metric quality drift",
	)
	service.validate_monitoring_lifecycle_batch(
		tenant_id="tenant-signals",
		event_stream="bytewax",
		mutation_count=3,
	)

	agent_model = monitoring_agent_roster_model(service, "tenant-signals")
	lifecycle_model = lifecycle_batch_model(service, "tenant-signals")

	assert agent_model["rows"][0]["role"] == "metric_quality_reviewer"
	assert "codex" in agent_model["supported_runtimes"]
	assert lifecycle_model["streaming"]["required_processor"] == "bytewax"
	assert lifecycle_model["rows"][0]["mutation_count"] == 3
	assert settings_model("tenant-signals")["review_evidence"]["pending_queues"]


def test_generated_view_models_are_operable():
	service = MoniService("tenant-signals")
	service.register_source(
		tenant_id="tenant-signals",
		source_id="orders-api",
		service_name="orders",
		environment="production",
		owner="platform",
		notification_route="pagerduty:orders",
	)
	service.ingest_signal(
		tenant_id="tenant-signals",
		source_id="orders-api",
		signal_type="metric",
		name="orders.latency_ms",
		value=250,
	)

	assert dashboard_model(service, "tenant-signals")["summary"]["source_count"] == 1
	assert source_inventory_model(service, "tenant-signals")["rows"][0]["source_id"] == "orders-api"
	assert signal_explorer_model(service, "tenant-signals")["rows"][0]["name"] == "orders.latency_ms"
	assert alert_center_model(service, "tenant-signals")["actions"] == ["acknowledge", "resolve", "open_incident"]
	assert incident_model(service, "tenant-signals")["columns"]
	assert remediation_model(service, "tenant-signals")["review_actions"] == ["approved", "rejected"]
	assert "opentelemetry" in adapter_health_model("tenant-signals")["supported_collectors"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "moni_signal_console"
	assert registration["ui_components"]["alerts"] == "/moni/alerts"
	assert registration["ui_components"]["incidents"] == "/moni/incidents"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["review_evidence"]["deny_behavior"] == "Denied MONI lifecycle batches persist evidence before PermissionError"
	assert registration["dependencies"] == ["conf", "audl", "mqeb"]
