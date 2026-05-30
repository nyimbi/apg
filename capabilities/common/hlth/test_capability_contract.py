"""Regression coverage for the HLTH executable capability contract."""

import pytest

from capabilities.common.hlth import register_capability
from capabilities.common.hlth.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.hlth.service import HlthService
from capabilities.common.hlth.view_models import (
	adapter_health_model,
	alert_center_model,
	baseline_model,
	check_timeline_model,
	component_inventory_model,
	dashboard_model,
	deployment_gate_model,
	incident_model,
	prediction_model,
	remediation_model,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-health",
		{"prediction": {"prediction_window_hours": 48}}
	)

	assert contract["capability"] == "hlth"
	assert contract["display_name"] == "Health Checks and Diagnostics"
	assert contract["configuration"]["tenant_id"] == "tenant-health"
	assert contract["configuration"]["prediction"]["prediction_window_hours"] == 48
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"assessment",
		"baselines",
		"alerts",
		"prediction",
		"remediation",
		"incidents",
		"deployment_gates",
		"adapters",
		"security",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 18
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"components",
		"checks",
		"baselines",
		"alerts",
		"incidents",
		"predictions",
		"remediation",
		"deployment_gates",
		"reports",
		"audit",
		"adapters",
		"settings"
	}
	assert contract["ui"]["view_module"] == "views.py"
	assert contract["ui"]["api_prefix"] == "/hlth/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "deployment_gate_panel" in contract["theme"]["components"]
	assert "moni" in contract["configuration"]["adapters"]["supported_probe_sources"]


def test_rule_engine_enforces_health_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "track_component_health",
		"component_id_present": False,
		"component_registered": False,
		"component_status": "disabled",
		"health_score": -1,
		"alert_created": False,
		"alert_severity": "critical",
		"alert_owner_present": False,
		"notification_route_configured": False,
		"incident_severity": "critical",
		"incident_owner_present": False,
		"remediation_requested": True,
		"runbook_attached": False,
		"environment": "production",
		"production_approved": False,
		"baseline_age_days": 45,
		"baseline_review_recorded": False,
		"deployment_requested": True,
		"unresolved_critical_incidents": 1,
		"deployment_waiver_requested": True,
		"waiver_review_recorded": False,
		"reviewer_same_as_requester": True,
		"review_notes_attached": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"component_health_requires_component_id",
		"component_must_be_registered",
		"disabled_component_blocks_health_check",
		"health_score_below_range_denied",
		"critical_health_score_creates_alert",
		"critical_alert_requires_owner",
		"critical_alert_requires_route",
		"critical_incident_requires_owner",
		"critical_incident_requires_route",
		"remediation_requires_runbook",
		"production_remediation_requires_approval",
		"stale_baseline_requires_review",
		"unresolved_critical_incident_blocks_deploy",
		"deployment_waiver_requires_review",
		"remediation_review_requires_independent_reviewer",
		"review_notes_required",
	}
	prediction = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "predict_health",
		"baseline_present": False,
		"prediction_confidence": 0.5,
	})
	assert prediction["decision"] == "deny"
	assert {"prediction_requires_baseline", "low_confidence_prediction_requires_review"} <= set(prediction["matched_rules"])


def test_hlth_service_governs_components_checks_predictions_incidents_and_gates():
	service = HlthService("tenant-health")
	component = service.register_component(
		tenant_id="tenant-health",
		component_id="orders-api",
		name="Orders API",
		component_type="service",
		environment="production",
		owner="platform",
		criticality="critical",
	)
	check = service.record_health_check(
		tenant_id="tenant-health",
		component_id="orders-api",
		dimension="availability",
		score=35,
		summary="Availability below threshold",
		owner="platform",
		notification_route="pagerduty:orders",
	)
	baseline = service.create_baseline(
		tenant_id="tenant-health",
		component_id="orders-api",
		dimension="availability",
		expected_score=95,
		sample_count=50,
	)
	stale_prediction = service.request_prediction(
		tenant_id="tenant-health",
		component_id="orders-api",
		baseline_id=baseline.baseline_id,
		predicted_score=62,
		confidence=0.6,
		baseline_age_days=45,
	)
	remediation = service.request_remediation(
		tenant_id="tenant-health",
		incident_id=check.incident_id,
		requester="platform",
		environment="production",
		runbook_id="orders-restore-availability",
		runbook_attached=True,
		production_approved=True,
		proposed_action="restart unhealthy workers",
		reason="availability score below threshold",
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
		notes="runbook, approval, and blast radius reviewed",
	)
	blocked_gate = service.evaluate_deployment_gate(
		tenant_id="tenant-health",
		deployment_id="orders-2026-05-30",
	)
	waived_gate = service.evaluate_deployment_gate(
		tenant_id="tenant-health",
		deployment_id="orders-2026-05-30-hotfix",
		waiver_recorded=True,
		waiver_review_recorded=True,
	)

	assert component.component_id == "orders-api"
	assert check.status == "critical"
	assert check.alert_id in service.alerts
	assert check.incident_id in service.incidents
	assert baseline.status == "active"
	assert stale_prediction.status == "pending_review"
	assert {"stale_baseline_requires_review", "low_confidence_prediction_requires_review"} <= set(stale_prediction.matched_rules)
	assert denied_status == "review_denied"
	assert approved.status == "approved"
	assert blocked_gate.status == "blocked"
	assert "unresolved_critical_incident_blocks_deploy" in blocked_gate.matched_rules
	assert waived_gate.status == "allowed"
	assert service.dashboard_summary("tenant-health")["component_count"] == 1
	assert service.dashboard_summary("tenant-health")["open_incident_count"] == 1


def test_hlth_service_fails_closed_for_invalid_lifecycle_references():
	service = HlthService("tenant-health")
	missing = service.record_health_check(
		tenant_id="tenant-health",
		component_id="missing",
		dimension="availability",
		score=80,
		summary="No component",
	)
	service.register_component(
		tenant_id="tenant-health",
		component_id="disabled-api",
		name="Disabled API",
		component_type="service",
		environment="production",
		owner="platform",
		status="disabled",
	)
	disabled = service.record_health_check(
		tenant_id="tenant-health",
		component_id="disabled-api",
		dimension="availability",
		score=80,
		summary="Disabled component",
	)
	missing_baseline = service.request_prediction(
		tenant_id="tenant-health",
		component_id="missing",
		baseline_id="missing-baseline",
		predicted_score=70,
		confidence=0.8,
	)
	with pytest.raises(ValueError, match="existing incident"):
		service.request_remediation(
			tenant_id="tenant-health",
			incident_id="missing-incident",
			requester="platform",
			environment="production",
			runbook_id="orders-restore-availability",
			runbook_attached=True,
			production_approved=True,
			proposed_action="restart workers",
			reason="availability score below threshold",
		)
	with pytest.raises(ValueError, match="registered component"):
		service.create_alert(
			tenant_id="tenant-health",
			component_id="missing-alert-component",
			severity="critical",
			title="Missing component",
			owner="platform",
			notification_route="pagerduty:orders",
		)
	with pytest.raises(ValueError, match="expected_score"):
		service.register_component(
			tenant_id="tenant-health",
			component_id="bad-baseline-api",
			name="Bad Baseline API",
			component_type="service",
			environment="production",
			owner="platform",
		)
		service.create_baseline(
			tenant_id="tenant-health",
			component_id="bad-baseline-api",
			dimension="availability",
			expected_score=125,
			sample_count=50,
		)

	assert missing.status == "denied"
	assert "component_must_be_registered" in missing.matched_rules
	assert disabled.status == "denied"
	assert "disabled_component_blocks_health_check" in disabled.matched_rules
	assert missing_baseline.status == "denied"
	assert "prediction_requires_baseline" in missing_baseline.matched_rules


def test_generated_view_models_are_operable():
	service = HlthService("tenant-health")
	service.register_component(
		tenant_id="tenant-health",
		component_id="orders-api",
		name="Orders API",
		component_type="service",
		environment="production",
		owner="platform",
	)
	service.record_health_check(
		tenant_id="tenant-health",
		component_id="orders-api",
		dimension="availability",
		score=90,
		summary="Healthy",
	)
	baseline = service.create_baseline(
		tenant_id="tenant-health",
		component_id="orders-api",
		dimension="availability",
		expected_score=95,
		sample_count=50,
	)
	service.request_prediction(
		tenant_id="tenant-health",
		component_id="orders-api",
		baseline_id=baseline.baseline_id,
		predicted_score=92,
		confidence=0.9,
	)

	assert dashboard_model(service, "tenant-health")["summary"]["component_count"] == 1
	assert component_inventory_model(service, "tenant-health")["rows"][0]["component_id"] == "orders-api"
	assert check_timeline_model(service, "tenant-health")["rows"][0]["summary"] == "Healthy"
	assert baseline_model(service, "tenant-health")["rows"][0]["expected_score"] == 95
	assert prediction_model(service, "tenant-health")["rows"][0]["risk"] == "low"
	assert alert_center_model(service, "tenant-health")["actions"] == ["acknowledge", "resolve", "open_incident"]
	assert incident_model(service, "tenant-health")["columns"]
	assert remediation_model(service, "tenant-health")["review_actions"] == ["approved", "rejected"]
	assert deployment_gate_model(service, "tenant-health")["columns"]
	assert "moni" in adapter_health_model("tenant-health")["supported_probe_sources"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "hlth_health_console"
	assert registration["ui_components"]["predictions"] == "/hlth/predictions"
	assert registration["ui_components"]["deployment_gates"] == "/hlth/deployment-gates"
	assert "moni" in registration["dependencies"]
