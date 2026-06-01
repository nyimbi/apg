"""Regression coverage for the ANOM executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.anom import api, register_capability, views
from capabilities.common.anom.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
)
from capabilities.common.anom.service import AnomService


def test_contract_exposes_full_lifecycle_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-signals", {"detection": {"default_sensitivity": "high"}})

	assert contract["capability"] == "anom"
	assert contract["configuration"]["tenant_id"] == "tenant-signals"
	assert contract["configuration"]["detection"]["default_sensitivity"] == "high"
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"sources",
		"detection",
		"baselines",
		"signals",
		"investigation",
		"feedback",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 39
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"sources",
		"baselines",
		"detector",
		"signals",
		"investigations",
		"alerts",
		"rules",
		"feedback",
		"quality",
		"agents",
		"lifecycle",
		"audit",
		"settings",
	}
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.AnomService"
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["lifecycle_stream"] == "anom.lifecycle"
	assert next(route for route in contract["ui"]["routes"] if route["name"] == "audit")["permission"] == "anom:audit"
	assert contract["ui"]["api_prefix"] == "/anom/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {
		"signal_card",
		"baseline_chart",
		"alert_queue",
		"quality_dashboard",
		"anomaly_agent_roster",
		"bytewax_lifecycle_panel",
		"audit_timeline",
	} <= set(contract["theme"]["components"])


def test_rule_engine_enforces_anomaly_guardrails():
	detection_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "detect",
		"monitoring_source_present": False,
		"baseline_present": False,
		"metric_present": False,
		"value_present": False,
		"severity": "critical",
		"owner_assigned": False,
	})
	feedback_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "record_feedback",
		"signal_present": True,
		"reviewer_present": True,
		"label_present": True,
		"label_known": True,
		"false_positive_rate": 0.4,
		"tuning_review_recorded": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_batch_detection",
		"event_stream": "legacy_queue",
	})
	state_change_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"state_change_requested": True,
		"audit_event_recorded": False,
	})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_anomaly_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": True,
		"scope_present": True,
		"owner_present": True,
		"purpose_present": True,
		"contribution_disclosed": True,
		"privileged_role": False,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_anom_lifecycle_batch",
		"event_stream": "legacy_queue",
	})

	assert detection_result["decision"] == "deny"
	assert set(detection_result["matched_rules"]) == {
		"tenant_context_required",
		"detection_requires_monitoring_source",
		"detection_requires_baseline",
		"detection_requires_metric",
		"detection_requires_value",
		"critical_anomaly_requires_owner",
	}
	assert feedback_result["decision"] == "require_review"
	assert feedback_result["matched_rules"] == ["high_false_positive_rate_requires_tuning"]
	assert batch_result["matched_rules"] == ["batch_detection_requires_bytewax"]
	assert batch_result["actions"][0]["reason"] == "bytewax_event_stream_required"
	assert state_change_result["matched_rules"] == ["anomaly_state_change_requires_audit"]
	assert agent_result["matched_rules"] == ["anomaly_agent_runtime_supported"]
	assert agent_result["actions"][0]["reason"] == "unsupported_anomaly_agent_runtime"
	assert lifecycle_result["matched_rules"] == ["bytewax_anom_stream_required"]
	assert lifecycle_result["actions"][0]["reason"] == "bytewax_lifecycle_stream_required"


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "anom"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "anom_signal_console"
	assert registration["ui_components"]["investigations"] == "/anom/investigations"
	assert registration["ui_components"]["audit"] == "/anom/audit"
	assert registration["ui_components"]["agents"] == "/anom/agents"
	assert registration["ui_components"]["lifecycle"] == "/anom/lifecycle"
	assert "investigation_closure_governance" in registration["capabilities"]
	assert "feedback_tuning" in registration["capabilities"]
	assert "anomaly_agent_composition" in registration["capabilities"]
	assert "lifecycle_batch_governance" in registration["capabilities"]
	assert "pred" in registration["dependencies"]
	assert "conf" in registration["dependencies"]
	assert "anom:audit" in registration["permissions"]


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
	agent = service.register_anomaly_agent(
		agent_id="anomaly-agent-1",
		tenant_id="tenant-signals",
		name="Anomaly Steward",
		runtime="codex",
		role="anomaly_steward",
		scope="source baseline signal review",
		owner="sre-lead",
		purpose="review anomaly lifecycle changes",
	)
	batch = service.validate_anom_lifecycle_batch(
		tenant_id="tenant-signals",
		event_stream="bytewax",
		mutation_count=2,
		operation="anomaly_agent_batch",
		batch_id="anom-batch-001",
	)
	summary = service.signal_summary("tenant-signals")
	dashboard = views.dashboard_model(service, "tenant-signals")
	source_registry = views.source_registry_model(service, "tenant-signals")
	detector = views.detection_workbench_model(service, "tenant-signals")
	alerts = views.alert_queue_model(service, "tenant-signals")
	rules = views.rule_manager_model(service, "tenant-signals")
	quality = views.quality_model(service, "tenant-signals")
	agent_roster = views.anomaly_agent_roster_model(service, "tenant-signals")
	lifecycle = views.lifecycle_batch_model(service, "tenant-signals")
	audit = views.audit_timeline_model(service, "tenant-signals")

	assert baseline["history_points"] == 60
	assert baseline["status"] == "active"
	assert baseline["decision"] == "allow"
	assert signal["severity"] == "critical"
	assert signal["decision"] == "allow"
	assert "recent deployment present in observation context" in signal["root_cause_hints"]
	assert investigation["status"] == "closed"
	assert investigation["closed_by"] == "sre-lead"
	assert investigation["resolution_evidence"] == ["incident:123", "deployment rollback completed"]
	assert feedback["label"] == "true_positive"
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["required_processor"] == "bytewax"
	assert batch["status"] == "accepted"
	assert summary["critical_or_high_count"] == 1
	assert summary["investigation_count"] == 1
	assert summary["pending_source_review_count"] == 0
	assert summary["pending_baseline_review_count"] == 0
	assert summary["pending_signal_review_count"] == 0
	assert summary["pending_feedback_review_count"] == 0
	assert summary["anomaly_agent_count"] == 1
	assert summary["lifecycle_batch_count"] == 1
	assert dashboard["summary"]["signal_count"] == 1
	assert dashboard["pending_reviews"]["sources"] == []
	assert dashboard["pending_reviews"]["baselines"] == []
	assert dashboard["pending_reviews"]["signals"] == []
	assert dashboard["pending_reviews"]["feedback"] == []
	assert dashboard["anomaly_agents"][0]["id"] == "anomaly-agent-1"
	assert dashboard["lifecycle_batches"][0]["id"] == "anom-batch-001"
	assert source_registry["sources"][0]["id"] == "api_latency"
	assert detector["required_fields"] == ["source_id", "baseline_id", "metric", "value"]
	assert alerts["notification_adapter"] == "ntfy"
	assert len(rules["rules"]) >= 39
	assert rules["agents"]["first_class"] is True
	assert quality["tuning_required"] is False
	assert agent_roster["agents"][0]["role"] == "anomaly_steward"
	assert lifecycle["batches"][0]["operation"] == "anomaly_agent_batch"
	assert audit["audit_events"]
	assert {event["event_type"] for event in service.list_audit_events("tenant-signals")} >= {
		"monitoring_source_registered",
		"baseline_created",
		"signal_detected",
		"investigation_opened",
		"investigation_closed",
		"feedback_recorded",
		"anomaly_agent_registered",
		"anom_lifecycle_batch_accepted",
	}


def test_service_blocks_invalid_detection_and_tuning_flows():
	service = AnomService()

	with pytest.raises(PermissionError, match="source_name_required"):
		service.register_source(
			source_id="missing-name",
			tenant_id="tenant-signals",
			name="",
			kind="metric",
			owner="platform",
		)

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
		owner="platform",
	)
	unknown_source = service.register_source(
		source_id="unknown-kind",
		tenant_id="tenant-signals",
		name="Unknown Source",
		kind="legacy_counter",
		owner="platform",
	)

	assert unknown_source["status"] == "pending_review"
	assert unknown_source["decision"] == "require_review"
	assert unknown_source["matched_rules"] == ["source_kind_requires_review"]
	assert unknown_source["review_reasons"] == ["source_kind_review_required"]

	with pytest.raises(PermissionError, match="source_owner_required"):
		service.register_source(
			source_id="missing-owner",
			tenant_id="tenant-signals",
			name="Missing Owner",
			kind="metric",
			owner="",
		)

	with pytest.raises(PermissionError, match="source_kind_required"):
		service.register_source(
			source_id="missing-kind",
			tenant_id="tenant-signals",
			name="Missing Kind",
			kind="",
			owner="platform",
	)

	with pytest.raises(PermissionError, match="baseline_source_required"):
		service.create_baseline(
			baseline_id="missing-source-baseline",
			tenant_id="tenant-signals",
			source_id="missing",
			metric="error_rate",
			values=[1.0 + (index % 3) for index in range(60)],
			sensitivity="medium",
		)

	with pytest.raises(PermissionError, match="baseline_metric_required"):
		service.create_baseline(
			baseline_id="missing-metric",
			tenant_id="tenant-signals",
			source_id="errors",
			metric="",
			values=[1.0 + (index % 3) for index in range(60)],
			sensitivity="medium",
		)

	with pytest.raises(PermissionError, match="baseline_sensitivity_required"):
		service.create_baseline(
			baseline_id="missing-sensitivity",
			tenant_id="tenant-signals",
			source_id="errors",
			metric="error_rate",
			values=[1.0 + (index % 3) for index in range(60)],
			sensitivity="",
		)

	with pytest.raises(PermissionError, match="insufficient_baseline_history"):
		service.create_baseline(
			baseline_id="short",
			tenant_id="tenant-signals",
			source_id="errors",
			metric="error_rate",
			values=[1.0, 2.0, 3.0],
			sensitivity="medium",
		)

	unknown_sensitivity = service.create_baseline(
		baseline_id="unknown-sensitivity",
		tenant_id="tenant-signals",
		source_id="errors",
		metric="error_rate",
		values=[1.0 + (index % 3) for index in range(60)],
		sensitivity="extreme",
	)
	assert unknown_sensitivity["status"] == "pending_review"
	assert unknown_sensitivity["decision"] == "require_review"
	assert unknown_sensitivity["matched_rules"] == ["baseline_sensitivity_requires_review"]
	assert unknown_sensitivity["review_reasons"] == ["baseline_sensitivity_review_required"]

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

	untriaged = service.create_record(
		record_id="high-without-triage",
		tenant_id="tenant-signals",
		metadata={"severity": "high", "owner": "sre-lead"},
	)
	assert untriaged["status"] == "pending_review"
	assert untriaged["decision"] == "require_review"
	assert untriaged["matched_rules"] == ["high_anomaly_requires_triage"]
	assert untriaged["review_reasons"] == ["high_anomaly_triage_required"]

	triaged = service.create_record(
		record_id="high-with-triage",
		tenant_id="tenant-signals",
		metadata={"severity": "high", "owner": "sre-lead", "triage_recorded": True},
	)
	assert triaged["severity"] == "high"

	review_feedback = service.record_feedback(
		feedback_id="fp-1",
		tenant_id="tenant-signals",
		signal_id=signal["id"],
		label="false_positive",
		reviewer="sre-lead",
	)
	assert review_feedback["status"] == "pending_review"
	assert review_feedback["decision"] == "require_review"
	assert review_feedback["matched_rules"] == ["high_false_positive_rate_requires_tuning"]
	assert review_feedback["review_reasons"] == ["tuning_review_required"]

	summary = service.signal_summary("tenant-signals")
	source_view = views.source_registry_model(service, "tenant-signals")
	baseline_view = views.baseline_console_model(service, "tenant-signals")
	signal_view = views.signal_board_model(service, "tenant-signals")
	feedback_view = views.feedback_review_model(service, "tenant-signals")
	quality_view = views.quality_model(service, "tenant-signals")
	assert summary["pending_source_review_count"] == 1
	assert summary["pending_baseline_review_count"] == 1
	assert summary["pending_signal_review_count"] == 1
	assert summary["pending_feedback_review_count"] == 1
	assert source_view["pending_review"][0]["id"] == "unknown-kind"
	assert baseline_view["pending_review"][0]["id"] == "unknown-sensitivity"
	assert signal_view["pending_review"][0]["id"] == "high-without-triage"
	assert feedback_view["pending_review"][0]["id"] == "fp-1"
	assert quality_view["pending_feedback_review"][0]["id"] == "fp-1"

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

	with pytest.raises(PermissionError, match="investigation_resolution_evidence_required"):
		service.close_investigation(
			investigation_id="investigate:critical-with-owner",
			tenant_id="tenant-signals",
			resolution="accepted risk",
			closed_by="sre-lead",
			resolution_evidence=[],
		)

	with pytest.raises(PermissionError, match="feedback_reviewer_required"):
		service.record_feedback(
			feedback_id="reviewer-required",
			tenant_id="tenant-signals",
			signal_id=signal["id"],
			label="true_positive",
			reviewer="",
			tuning_review_recorded=True,
		)


def test_service_enforces_anomaly_agent_and_lifecycle_guardrails():
	service = AnomService()
	tenant_id = "tenant-agent"

	with pytest.raises(PermissionError, match="unsupported_anomaly_agent_runtime"):
		service.register_anomaly_agent(
			"agent-bad-runtime",
			tenant_id,
			"Bad Runtime",
			"unknown",
			"anomaly_steward",
			"signal review",
			"owner",
			"purpose",
		)

	with pytest.raises(PermissionError, match="anomaly_agent_scope_required"):
		service.register_anomaly_agent(
			"agent-no-scope",
			tenant_id,
			"No Scope",
			"codex",
			"anomaly_steward",
			"",
			"owner",
			"purpose",
		)

	with pytest.raises(PermissionError, match="anomaly_agent_contribution_disclosure_required"):
		service.register_anomaly_agent(
			"agent-no-disclosure",
			tenant_id,
			"No Disclosure",
			"codex",
			"anomaly_steward",
			"signal review",
			"owner",
			"purpose",
			contribution_disclosed=False,
		)

	agent = service.register_anomaly_agent(
		"agent-review",
		tenant_id,
		"Review Agent",
		"claude-code",
		"signal triage reviewer",
		"critical signal triage",
		"owner",
		"purpose",
	)

	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "signal_triage_reviewer"
	assert agent["status"] == "pending_review"
	assert service.signal_summary(tenant_id)["pending_agent_review_count"] == 1

	with pytest.raises(ValueError, match="anom_lifecycle_batch_empty"):
		service.validate_anom_lifecycle_batch(tenant_id, "bytewax", 0)
	with pytest.raises(ValueError, match="unsupported_anom_lifecycle_operation"):
		service.validate_anom_lifecycle_batch(tenant_id, "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_anom_lifecycle_batch(tenant_id, "legacy_queue", 1)

	assert service.list_lifecycle_batches(tenant_id)[0]["status"] == "denied"
	assert service.signal_summary(tenant_id)["denied_lifecycle_batch_count"] == 1


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = AnomService()
	for tenant_id, owner in [("tenant-a", "owner-a"), ("tenant-b", "owner-b")]:
		service.register_source(
			source_id="shared-source",
			tenant_id=tenant_id,
			name=f"Shared Source {tenant_id}",
			kind="metric",
			owner=owner,
		)
		service.create_baseline(
			baseline_id="shared-baseline",
			tenant_id=tenant_id,
			source_id="shared-source",
			metric="latency",
			values=[100.0 + (index % 4) for index in range(60)],
			sensitivity="medium",
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
	with pytest.raises(PermissionError, match="source_name_required"):
		api.register_source({"id": "api-missing-source-fields", "tenant_id": "tenant-api-anom"})

	source = api.register_source({
		"id": "api-source",
		"tenant_id": "tenant-api-anom",
		"name": "API Source",
		"kind": "metric",
		"owner": "api-owner",
	})
	with pytest.raises(PermissionError, match="baseline_metric_required"):
		api.create_baseline({
			"id": "api-missing-metric",
			"tenant_id": source["tenant_id"],
			"source_id": source["id"],
			"values": [100.0 + (index % 5) for index in range(60)],
			"sensitivity": "medium",
		})

	baseline = api.create_baseline({
		"id": "api-baseline",
		"tenant_id": source["tenant_id"],
		"source_id": source["id"],
		"metric": "latency",
		"values": [100.0 + (index % 5) for index in range(60)],
		"sensitivity": "medium",
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
	with pytest.raises(PermissionError, match="feedback_reviewer_required"):
		api.record_feedback({
			"id": "api-feedback-missing-reviewer",
			"tenant_id": source["tenant_id"],
			"signal_id": signal["id"],
			"label": "true_positive",
			"tuning_review_recorded": True,
		})

	closed = api.close_investigation({
		"id": f"investigate:{signal['id']}",
		"tenant_id": source["tenant_id"],
		"resolution": "scaled API workers",
		"closed_by": "api-owner",
		"resolution_evidence": ["runbook:latency"],
	})
	agent = api.register_anomaly_agent({
		"id": "api-agent",
		"tenant_id": source["tenant_id"],
		"name": "API Anomaly Agent",
		"runtime": "opencode",
		"role": "anomaly_steward",
		"scope": "api anomaly review",
		"owner": "api-owner",
		"purpose": "govern API anomaly changes",
	})
	batch = api.validate_anom_lifecycle_batch({
		"id": "api-batch",
		"tenant_id": source["tenant_id"],
		"event_stream": "bytewax",
		"mutation_count": 1,
		"operation": "anomaly_agent_batch",
	})

	assert closed["status"] == "closed"
	assert agent["runtime"] == "opencode"
	assert batch["status"] == "accepted"
	assert api.list_anomaly_agents(source["tenant_id"])[0]["id"] == "api-agent"
	assert api.list_lifecycle_batches(source["tenant_id"])[0]["id"] == "api-batch"
	assert {event["event_type"] for event in api.list_audit_events(source["tenant_id"])} >= {
		"investigation_closed",
		"anomaly_agent_registered",
		"anom_lifecycle_batch_accepted",
	}
