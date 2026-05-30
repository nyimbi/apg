"""Regression coverage for the AUDL executable capability contract."""

import pytest

from .. import get_capability_info
from .. import api_helpers, view_models
from ..audit_runtime import AudlService
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-a",
		{"retention": {"archive_after_days": 180}}
	)

	assert contract["capability"] == "audl"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["retention"]["archive_after_days"] == 180
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"ingestion",
		"agents",
		"retention",
		"compliance",
		"investigations",
		"notifications",
		"ui",
		"theme",
		"streaming"
	]
	assert len(contract["rule_engine"]["rules"]) >= 10
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"events",
		"timeline",
		"investigations",
		"legal_holds",
		"exports",
		"purges",
		"compliance",
		"agents",
		"reports",
		"rules",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/api/v1/audit"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "audit_timeline" in contract["theme"]["components"]
	assert "compliance_scorecard" in contract["theme"]["components"]
	assert "legal_hold_indicator" in contract["theme"]["components"]
	assert "audit_agent_roster" in contract["theme"]["components"]
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["engine"] == "bytewax"


def test_rule_engine_denies_unsafe_audit_operations():
	export_result = evaluate_capability_rules({
		"tenant_id_missing": True,
		"immutable_storage": True,
		"checksum_verified": False,
		"requested_operation": "export",
		"contains_pii": True,
		"masking_enabled": False,
		"event_severity": "critical",
		"escalation_configured": False,
		"batch_size": 20000,
		"stream_processing_enabled": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_id_missing": False,
		"requested_operation": "audit_batch",
		"batch_size": 20000,
		"stream_processing_enabled": False,
		"event_stream": "non_bytewax",
	})

	assert export_result["decision"] == "deny"
	assert set(export_result["matched_rules"]) == {
		"require_tenant_context",
		"immutable_events_require_checksum",
		"regulated_exports_require_masking",
		"critical_events_require_escalation",
		"high_volume_ingestion_requires_stream_processing",
	}
	assert batch_result["decision"] == "deny"
	assert set(batch_result["matched_rules"]) == {
		"high_volume_ingestion_requires_stream_processing",
		"bytewax_event_stream_required",
	}


def test_capability_info_includes_manifest_and_theme():
	info = get_capability_info()

	assert info["metadata"]["capability_id"] == "common/audl"
	assert info["configuration"]["tenant_id"] == "default"
	assert info["rule_engine"]["type"] == "deterministic"
	assert info["ui_manifest"]["requires_theme"] is True
	assert info["theme"]["name"] == "audl_forensics"
	assert info["agents"]["first_class"] is True
	assert info["streaming"]["engine"] == "bytewax"
	assert {route["name"] for route in info["ui_manifest"]["routes"]} >= {
		"timeline",
		"investigations",
		"agents",
		"reports",
	}


def test_service_runs_governed_audit_evidence_lifecycle():
	service = AudlService()
	event = service.append_event(
		event_id="evt-critical",
		tenant_id="tenant-audl",
		actor="security-analyst",
		action="review_access",
		resource_type="account",
		resource_id="acct-001",
		severity="critical",
		contains_pii=True,
		escalation_configured=True,
	)
	batch = service.validate_batch(
		tenant_id="tenant-audl",
		record_count=12000,
		event_stream="bytewax",
		stream_processing_enabled=True,
	)
	agent = service.register_audit_agent(
		agent_id="agent-001",
		tenant_id="tenant-audl",
		name="Audit Evidence Reviewer",
		runtime="codex",
		role="evidence_reviewer",
		purpose="Review chain-of-custody evidence before export.",
		owner="security-lead",
	)
	hold = service.apply_legal_hold(
		hold_id="hold-001",
		tenant_id="tenant-audl",
		scope={"resource_type": "account", "resource_id": "acct-001"},
		reason="Regulatory investigation.",
		approver="legal-counsel",
	)
	export = service.request_export(
		export_id="export-001",
		tenant_id="tenant-audl",
		requested_by="privacy-officer",
		query={"resource_id": "acct-001"},
		contains_pii=True,
		masking_enabled=True,
		reason="Regulator evidence request.",
	)
	export_decision = service.decide_export(
		export_id=export["id"],
		tenant_id="tenant-audl",
		reviewer="compliance-reviewer",
		decision="approved",
		notes="PII masking verified.",
	)
	investigation = service.open_investigation(
		investigation_id="case-001",
		tenant_id="tenant-audl",
		event_ids=[event["id"]],
		owner="security-lead",
	)
	closed = service.close_investigation(
		investigation_id=investigation["id"],
		tenant_id="tenant-audl",
		closed_by="security-lead",
		resolution="Confirmed expected privileged access.",
		evidence={"export_id": export["id"], "hold_id": hold["id"]},
	)

	with pytest.raises(PermissionError, match="legal_hold_active"):
		service.request_purge(
			purge_id="purge-blocked",
			tenant_id="tenant-audl",
			requested_by="records-admin",
			scope={"resource_id": "acct-001"},
			reason="Retention cleanup.",
		)

	service.release_legal_hold(
		hold_id=hold["id"],
		tenant_id="tenant-audl",
		released_by="legal-counsel",
		release_evidence="Matter closed by regulator.",
	)
	purge = service.request_purge(
		purge_id="purge-001",
		tenant_id="tenant-audl",
		requested_by="records-admin",
		scope={"resource_id": "acct-001"},
		reason="Retention cleanup.",
	)
	purge_decision = service.decide_purge(
		purge_id=purge["id"],
		tenant_id="tenant-audl",
		reviewer="records-reviewer",
		decision="approved",
		notes="Legal hold released and retention policy satisfied.",
	)
	dashboard = view_models.dashboard_model(service, "tenant-audl")

	assert event["checksum"]
	assert batch["accepted"] is True
	assert agent["runtime"] == "codex"
	assert hold["status"] == "active"
	assert export_decision["decision"] == "approved"
	assert closed["status"] == "closed"
	assert purge_decision["decision"] == "approved"
	assert dashboard["summary"]["event_count"] == 1
	assert dashboard["summary"]["agent_count"] == 1
	assert dashboard["summary"]["active_legal_hold_count"] == 0
	assert dashboard["streaming"]["engine"] == "bytewax"
	assert view_models.audit_agent_model(service, "tenant-audl")["agents"][0]["id"] == "agent-001"
	assert {event["event_type"] for event in dashboard["governance_events"]} >= {
		"audit_event_appended",
		"audit_agent_registered",
		"legal_hold_applied",
		"export_requested",
		"export_decided",
		"investigation_opened",
		"investigation_closed",
		"legal_hold_released",
		"purge_requested",
		"purge_decided",
	}


def test_service_blocks_audit_guardrail_violations():
	service = AudlService()

	with pytest.raises(PermissionError, match="checksum_verification_required"):
		service.append_event(
			event_id="evt-bad-checksum",
			tenant_id="tenant-audl",
			actor="system",
			action="write",
			resource_type="record",
			resource_id="rec-001",
			checksum="not-the-expected-checksum",
		)

	with pytest.raises(PermissionError, match="critical_escalation_required"):
		service.append_event(
			event_id="evt-critical-no-escalation",
			tenant_id="tenant-audl",
			actor="system",
			action="delete",
			resource_type="record",
			resource_id="rec-001",
			severity="critical",
			escalation_configured=False,
		)

	with pytest.raises(PermissionError, match="pii_masking_required"):
		service.request_export(
			export_id="export-no-mask",
			tenant_id="tenant-audl",
			requested_by="privacy-officer",
			query={"contains": "pii"},
			contains_pii=True,
			masking_enabled=False,
			reason="Unsafe export.",
		)

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch(
			tenant_id="tenant-audl",
			record_count=25000,
			event_stream="queue",
			stream_processing_enabled=True,
		)

	with pytest.raises(PermissionError, match="audit_agent_runtime_unsupported"):
		service.register_audit_agent(
			agent_id="agent-bad-runtime",
			tenant_id="tenant-audl",
			name="Unsupported Agent",
			runtime="unsupported",
			role="audit_reviewer",
			purpose="Invalid runtime proof.",
			owner="security-lead",
		)

	with pytest.raises(PermissionError, match="audit_agent_human_approval_required"):
		service.register_audit_agent(
			agent_id="agent-no-approval",
			tenant_id="tenant-audl",
			name="Privileged Agent",
			runtime="codex",
			role="purge_reviewer",
			purpose="Privileged purge review.",
			owner="records-lead",
			human_approval_required=False,
		)

	event = service.append_event(
		event_id="evt-ok",
		tenant_id="tenant-audl",
		actor="system",
		action="read",
		resource_type="record",
		resource_id="rec-001",
	)
	export = service.request_export(
		export_id="export-ok",
		tenant_id="tenant-audl",
		requested_by="privacy-officer",
		query={"event_id": event["id"]},
		contains_pii=False,
		masking_enabled=False,
		reason="Internal export.",
	)
	with pytest.raises(ValueError, match="export reviewer notes are required"):
		service.decide_export(
			export_id=export["id"],
			tenant_id="tenant-audl",
			reviewer="reviewer",
			decision="approved",
			notes="",
		)

	purge = service.request_purge(
		purge_id="purge-ok",
		tenant_id="tenant-audl",
		requested_by="records-admin",
		scope={"event_id": event["id"]},
		reason="Retention cleanup.",
	)
	service.apply_legal_hold(
		hold_id="hold-after-purge-request",
		tenant_id="tenant-audl",
		scope={"event_id": event["id"]},
		reason="Late legal hold.",
		approver="legal-counsel",
	)
	with pytest.raises(PermissionError, match="legal_hold_active"):
		service.decide_purge(
			purge_id=purge["id"],
			tenant_id="tenant-audl",
			reviewer="records-reviewer",
			decision="approved",
			notes="Legal hold should block approval.",
		)
	service.release_legal_hold(
		hold_id="hold-after-purge-request",
		tenant_id="tenant-audl",
		released_by="legal-counsel",
		release_evidence="Late hold resolved.",
	)
	with pytest.raises(PermissionError, match="dual_control_reviewer_required"):
		service.decide_purge(
			purge_id=purge["id"],
			tenant_id="tenant-audl",
			reviewer="records-admin",
			decision="approved",
			notes="Self review is not allowed.",
		)

	investigation = service.open_investigation(
		investigation_id="case-open",
		tenant_id="tenant-audl",
		event_ids=[event["id"]],
		owner="security-lead",
	)
	with pytest.raises(ValueError, match="investigation closure evidence is required"):
		service.close_investigation(
			investigation_id=investigation["id"],
			tenant_id="tenant-audl",
			closed_by="security-lead",
			resolution="Resolved.",
			evidence={},
		)


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = AudlService()
	for tenant_id, actor in [("tenant-aa", "actor-a"), ("tenant-bb", "actor-b")]:
		service.append_event(
			event_id="shared-event",
			tenant_id=tenant_id,
			actor=actor,
			action="read",
			resource_type="record",
			resource_id="shared",
		)
		service.apply_legal_hold(
			hold_id="shared-hold",
			tenant_id=tenant_id,
			scope={"resource_id": "shared"},
			reason=f"Hold for {tenant_id}.",
			approver=actor,
		)

	assert service.list_events("tenant-aa")[0]["actor"] == "actor-a"
	assert service.list_events("tenant-bb")[0]["actor"] == "actor-b"
	assert service.list_legal_holds("tenant-aa")[0]["tenant_id"] == "tenant-aa"
	assert service.list_legal_holds("tenant-bb")[0]["tenant_id"] == "tenant-bb"

	with pytest.raises(ValueError, match="audit event already exists"):
		service.append_event(
			event_id="shared-event",
			tenant_id="tenant-aa",
			actor="actor-a",
			action="read",
			resource_type="record",
			resource_id="shared",
		)


def test_api_helpers_and_view_models_expose_audit_lifecycle():
	event = api_helpers.append_event({
		"id": "api-event",
		"tenant_id": "tenant-api-audl",
		"actor": "api-user",
		"action": "export",
		"resource_type": "report",
		"resource_id": "report-001",
		"contains_pii": "true",
	})
	export = api_helpers.request_export({
		"id": "api-export",
		"tenant_id": event["tenant_id"],
		"requested_by": "privacy-officer",
		"query": {"event_id": event["id"]},
		"contains_pii": "true",
		"masking_enabled": "true",
		"reason": "Evidence export.",
	})
	agent = api_helpers.register_audit_agent({
		"id": "api-agent",
		"tenant_id": event["tenant_id"],
		"name": "Export Evidence Assistant",
		"runtime": "claude_code",
		"role": "export_reviewer",
		"purpose": "Review masked export evidence before release.",
		"owner": "privacy-reviewer",
		"human_approval_required": "true",
	})
	decision = api_helpers.decide_export({
		"id": export["id"],
		"tenant_id": event["tenant_id"],
		"reviewer": "privacy-reviewer",
		"decision": "approved",
		"notes": "Masking verified.",
	})
	model = view_models.export_review_model(api_helpers.SERVICE, event["tenant_id"])
	agent_model = view_models.audit_agent_model(api_helpers.SERVICE, event["tenant_id"])

	assert decision["decision"] == "approved"
	assert api_helpers.capability_status(event["tenant_id"])["event_count"] == 1
	assert api_helpers.capability_status(event["tenant_id"])["agent_count"] == 1
	assert model["exports"][0]["id"] == "api-export"
	assert agent["role"] == "export_reviewer"
	assert agent_model["agents"][0]["id"] == "api-agent"
