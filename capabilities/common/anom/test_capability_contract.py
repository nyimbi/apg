"""Regression coverage for the ANOM executable capability contract."""

import pytest

from capabilities.common.anom import api, register_capability, views
from capabilities.common.anom.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.anom.service import AnomService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-signals", {"detection": {"default_sensitivity": "high"}})

	assert contract["capability"] == "anom"
	assert contract["configuration"]["tenant_id"] == "tenant-signals"
	assert contract["configuration"]["detection"]["default_sensitivity"] == "high"
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"detection",
		"baselines",
		"investigation",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "signals", "baselines", "investigations", "rules", "feedback", "settings"}
	assert contract["ui"]["api_prefix"] == "/anom/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "signal_card" in contract["theme"]["components"]


def test_rule_engine_enforces_anomaly_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "detect",
		"monitoring_source_present": False,
		"history_points": 20,
		"severity": "critical",
		"owner_assigned": False,
		"approval_recorded": False,
		"false_positive_rate": 0.4,
		"tuning_review_recorded": False
	})
	baseline_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_baseline",
		"history_points": 20
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"detection_requires_monitoring_source",
		"critical_anomaly_requires_owner",
		"high_false_positive_rate_requires_tuning"
	}
	assert baseline_result["decision"] == "deny"
	assert baseline_result["matched_rules"] == ["baseline_requires_history"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "anom"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "anom_signal_console"
	assert registration["ui_components"]["investigations"] == "/anom/investigations"
	assert "investigation_closure_governance" in registration["capabilities"]
	assert "pred" in registration["dependencies"]
	assert "anom:investigate" in registration["permissions"]


def test_service_builds_baselines_detects_signals_and_tracks_investigations():
	service = AnomService()
	service.register_source(
		source_id="api_latency",
		tenant_id="tenant-signals",
		name="API Latency",
		kind="metric",
		owner="platform",
	)
	baseline = service.create_baseline(
		baseline_id="api_latency_baseline",
		tenant_id="tenant-signals",
		source_id="api_latency",
		metric="p95_latency_ms",
		values=[100.0 + (index % 5) for index in range(60)],
		sensitivity="medium",
	)
	signal = service.detect(
		detection_id="signal-1",
		tenant_id="tenant-signals",
		source_id="api_latency",
		baseline_id="api_latency_baseline",
		metric="p95_latency_ms",
		value=160.0,
		context={"deployment": "checkout-v2", "region": "ke"},
		owner="sre-lead",
	)
	investigation = service.close_investigation(
		investigation_id="investigate:signal-1",
		tenant_id="tenant-signals",
		resolution="rollback checkout-v2",
		closed_by="sre-lead",
		resolution_evidence=["incident:123", "deployment rollback completed"],
	)
	feedback = service.record_feedback(
		feedback_id="feedback-1",
		tenant_id="tenant-signals",
		signal_id="signal-1",
		label="true_positive",
		reviewer="sre-lead",
	)
	summary = service.signal_summary("tenant-signals")
	investigation_view = views.investigation_queue_model(service, "tenant-signals")

	assert baseline["history_points"] == 60
	assert signal["severity"] == "critical"
	assert "recent deployment present in observation context" in signal["root_cause_hints"]
	assert investigation["status"] == "closed"
	assert investigation["closed_by"] == "sre-lead"
	assert investigation["resolution_evidence"] == ["incident:123", "deployment rollback completed"]
	assert feedback["label"] == "true_positive"
	assert summary["critical_or_high_count"] == 1
	assert summary["investigation_count"] == 1
	assert "closure_required_fields" in investigation_view
	assert {event["event_type"] for event in service.list_audit_events("tenant-signals")} >= {
		"monitoring_source_registered",
		"baseline_created",
		"signal_detected",
		"investigation_opened",
		"investigation_closed",
		"feedback_recorded",
	}


def test_service_blocks_invalid_detection_and_tuning_flows():
	service = AnomService()

	with pytest.raises(PermissionError, match="monitoring_source_required"):
		service.detect(
			detection_id="missing-source",
			tenant_id="tenant-signals",
			source_id="missing",
			baseline_id="missing",
			metric="p95_latency_ms",
			value=150.0,
		)

	service.register_source(
		source_id="errors",
		tenant_id="tenant-signals",
		name="Error Rate",
		kind="metric",
	)

	with pytest.raises(PermissionError, match="insufficient_baseline_history"):
		service.create_baseline(
			baseline_id="short",
			tenant_id="tenant-signals",
			source_id="errors",
			metric="error_rate",
			values=[1.0, 2.0, 3.0],
		)

	service.create_baseline(
		baseline_id="errors_baseline",
		tenant_id="tenant-signals",
		source_id="errors",
		metric="error_rate",
		values=[1.0 + (index % 3) for index in range(60)],
		sensitivity="medium",
	)

	with pytest.raises(PermissionError, match="investigation_owner_required"):
		service.detect(
			detection_id="critical-without-owner",
			tenant_id="tenant-signals",
			source_id="errors",
			baseline_id="errors_baseline",
			metric="error_rate",
			value=25.0,
		)

	signal = service.detect(
		detection_id="critical-with-owner",
		tenant_id="tenant-signals",
		source_id="errors",
		baseline_id="errors_baseline",
		metric="error_rate",
		value=25.0,
		owner="sre-lead",
	)

	with pytest.raises(PermissionError, match="tuning_review_required"):
		service.record_feedback(
			feedback_id="fp-1",
			tenant_id="tenant-signals",
			signal_id=signal["id"],
			label="false_positive",
			reviewer="sre-lead",
		)

	with pytest.raises(PermissionError, match="baseline_reset_approval_required"):
		service.reset_baseline(
			baseline_id="errors_baseline",
			tenant_id="tenant-signals",
			values=[1.0 + (index % 2) for index in range(60)],
			approval_recorded=False,
		)

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.reset_baseline(
			baseline_id="errors_baseline",
			values=[1.0 + (index % 2) for index in range(60)],
			approval_recorded=True,
		)

	with pytest.raises(ValueError, match="investigation resolution evidence is required"):
		service.close_investigation(
			investigation_id="investigate:critical-with-owner",
			tenant_id="tenant-signals",
			resolution="accepted risk",
			closed_by="sre-lead",
			resolution_evidence=[],
		)

	with pytest.raises(ValueError, match="feedback reviewer is required"):
		service.record_feedback(
			feedback_id="reviewer-required",
			tenant_id="tenant-signals",
			signal_id=signal["id"],
			label="true_positive",
			reviewer="",
			tuning_review_recorded=True,
		)


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = AnomService()
	for tenant_id, owner in [("tenant-a", "owner-a"), ("tenant-b", "owner-b")]:
		service.register_source(
			source_id="shared-source",
			tenant_id=tenant_id,
			name=f"Shared Source {tenant_id}",
			owner=owner,
		)
		service.create_baseline(
			baseline_id="shared-baseline",
			tenant_id=tenant_id,
			source_id="shared-source",
			metric="latency",
			values=[100.0 + (index % 4) for index in range(60)],
		)
		service.detect(
			detection_id="shared-signal",
			tenant_id=tenant_id,
			source_id="shared-source",
			baseline_id="shared-baseline",
			metric="latency",
			value=180.0,
			owner=owner,
		)

	closed = service.close_investigation(
		investigation_id="investigate:shared-signal",
		tenant_id="tenant-a",
		resolution="tenant-a rollback",
		closed_by="owner-a",
		resolution_evidence=["incident:a"],
	)

	assert closed["tenant_id"] == "tenant-a"
	assert service.list_investigations("tenant-a")[0]["status"] == "closed"
	assert service.list_investigations("tenant-b")[0]["status"] == "open"
	assert service.list_signals("tenant-a")[0]["tenant_id"] == "tenant-a"
	assert service.list_signals("tenant-b")[0]["tenant_id"] == "tenant-b"


def test_api_helpers_expose_closure_and_audit_lifecycle():
	source = api.register_source({
		"id": "api-source",
		"tenant_id": "tenant-api-anom",
		"name": "API Source",
		"owner": "api-owner",
	})
	baseline = api.create_baseline({
		"id": "api-baseline",
		"tenant_id": source["tenant_id"],
		"source_id": source["id"],
		"metric": "latency",
		"values": [100.0 + (index % 5) for index in range(60)],
	})
	signal = api.detect({
		"id": "api-signal",
		"tenant_id": source["tenant_id"],
		"source_id": source["id"],
		"baseline_id": baseline["id"],
		"metric": "latency",
		"value": 180.0,
		"owner": "api-owner",
	})
	closed = api.close_investigation({
		"id": f"investigate:{signal['id']}",
		"tenant_id": source["tenant_id"],
		"resolution": "scaled API workers",
		"closed_by": "api-owner",
		"resolution_evidence": ["runbook:latency"],
	})

	assert closed["status"] == "closed"
	assert api.list_audit_events(source["tenant_id"])[-1]["event_type"] == "investigation_closed"
