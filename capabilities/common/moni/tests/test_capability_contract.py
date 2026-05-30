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
	remediation_model,
	signal_explorer_model,
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
		"security",
		"ui",
		"theme"
	]
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
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/moni/api/v1"
	assert contract["ui"]["view_module"] == "view_models.py"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "alert_correlation_stack" in contract["theme"]["components"]


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

	assert source.source_id == "orders-api"
	assert metric.status == "accepted"
	assert pii_log.status == "denied"
	assert "pii_logs_blocked" in pii_log.matched_rules
	assert trace.status == "accepted"
	assert high_cardinality.status == "pending_review"
	assert "high_cardinality_metric_requires_review" in high_cardinality.matched_rules
	assert slo.status == "active"
	assert alert.status == "open"
	assert alert.incident_id in service.incidents
	assert denied_status == "review_denied"
	assert approved.status == "approved"
	assert service.dashboard_summary("tenant-signals")["source_count"] == 1
	assert service.dashboard_summary("tenant-signals")["open_incident_count"] == 1


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
	assert "auth" in registration["dependencies"]
