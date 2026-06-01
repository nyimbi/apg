"""SECU package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.secu import api, views
from capabilities.common.secu.service import SecuService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("package_contract_secu", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "secu"
	assert len(contract["ui"]["routes"]) >= 12
	assert len(contract["rule_engine"]["rules"]) >= 14
	assert contract["theme"]["tokens"]["border.radius"]
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["engine"] == "bytewax"
	assert "review_evidence" in contract["provides"]
	assert contract["review_evidence"]["pending_queues"]


def test_app_entrypoint_is_publishable():
	module = _load_module("package_app_secu", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()
	capability = model["capabilities"]["secu"]

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "secu" in model["capabilities"]
	assert len(capability["ui"]["routes"]) >= 11
	assert capability["approvals"]["policy_exception"] == "PolicyExceptionRecord"
	assert capability["approvals"]["incident_response"] == "SecurityIncidentRecord"
	assert capability["approvals"]["security_agent"] == "SecurityAgentRecord"
	assert capability["agents"]["first_class"] is True
	assert capability["streaming"]["engine"] == "bytewax"
	assert "review_evidence" in capability["provides"]
	assert capability["review_evidence"]["pending_queues"]
	assert model["agents"]["secu_agent_contract"]["first_class"] is True


def test_security_lifecycle_records_governed_exception_incident_and_audit_state():
	service = SecuService()

	policy = service.create_policy(
		tenant_id="tenant-a",
		name="Privileged access",
		owner="security-admin",
		security_level="restricted",
		required_controls=["mfa", "device_trust"],
		applies_to=["admin_console"],
		tags=["Privileged Access"],
	)
	device = service.record_device_posture(
		tenant_id="tenant-a",
		device_id="macbook-01",
		user_id="u-123",
		trust_state="trusted",
		risk_score=20,
	)
	threat = service.register_threat_indicator(
		tenant_id="tenant-a",
		name="Known hostile ASN",
		indicator_type="asn",
		value="AS64512",
		severity="high",
		source="manual",
	)
	assessment = service.assess_access(
		tenant_id="tenant-a",
		subject_id="u-123",
		subject_type="user",
		risk_score=75,
		device_id="macbook-01",
		challenge_completed=False,
	)
	control = service.record_compliance_control(
		tenant_id="tenant-a",
		framework="iso_27001",
		control_id="A.5.15",
		owner="security-admin",
		compliant=False,
	)
	exception = service.request_policy_exception(
		tenant_id="tenant-a",
		exception_id="break-glass",
		policy_id=policy["id"],
		requested_by="app-owner",
		reason="Emergency production repair",
		expires_at="2099-01-01T00:00:00Z",
	)
	approved = service.decide_policy_exception(
		tenant_id="tenant-a",
		exception_id=exception["id"],
		reviewer="security-reviewer",
		decision="approved",
		notes="Time-bound exception with compensating controls.",
	)
	incident = service.open_incident(
		tenant_id="tenant-a",
		incident_id="inc-1",
		title="Privileged credential exposure",
		severity="critical",
		opened_by="soc-analyst",
		containment_plan="Disable token and isolate affected host.",
	)
	contained = service.contain_incident(
		tenant_id="tenant-a",
		incident_id=incident["id"],
		actor="incident-commander",
		containment_action="Token disabled and host isolated.",
		containment_evidence="audit://incident/inc-1/containment",
	)
	resolved = service.resolve_incident(
		tenant_id="tenant-a",
		incident_id=contained["id"],
		resolved_by="incident-commander",
		resolution="Credentials rotated and monitoring confirmed clean.",
		notes="Post-incident review attached.",
	)
	agent = service.register_security_agent(
		tenant_id="tenant-a",
		agent_id="incident-agent",
		name="Incident Response Agent",
		runtime="claude-code",
		role="incident-responder",
		scope="summarize containment evidence for human responders",
		owner="security-admin",
		purpose="incident response evidence review",
		human_approval_required=True,
		policy_ref="secu-agent-policy",
	)
	batch = service.validate_security_lifecycle_batch("tenant-a", "ByteWax", 3)
	summary = service.dashboard_summary("tenant-a")

	assert policy["security_level"] == "restricted"
	assert device["quarantined"] is False
	assert threat["severity"] == "high"
	assert assessment["decision"] == "challenge"
	assert assessment["required_actions"] == ["complete_security_challenge"]
	assert control["status"] == "evidence_required"
	assert control["policy_decision"] == "require_review"
	assert approved["status"] == "approved"
	assert approved["policy_decision"] == "allow"
	assert resolved["status"] == "resolved"
	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "incident_responder"
	assert batch["event_stream"] == "bytewax"
	assert batch["accepted"] is True
	assert batch["status"] == "accepted"
	assert summary["policy_count"] == 1
	assert summary["assessment_count"] == 1
	assert summary["compliance_gap_count"] == 1
	assert summary["policy_exception_count"] == 1
	assert summary["open_incident_count"] == 0
	assert summary["security_agent_count"] == 1
	assert summary["security_lifecycle_batch_count"] == 1
	assert summary["pending_review_count"] == 1
	assert {event["event_type"] for event in service.list_audit_events("tenant-a")} >= {
		"policy_exception_requested",
		"policy_exception_decided",
		"security_incident_opened",
		"security_incident_contained",
		"security_incident_resolved",
		"security_agent_registered",
	}


def test_rule_guardrails_deny_quarantine_and_require_tenant_context():
	service = SecuService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_policy("", "No tenant", "owner")
	with pytest.raises(ValueError, match="policy_owner_required"):
		service.create_policy("tenant-a", "No owner", "")
	with pytest.raises(ValueError, match="unsupported_device_trust:rooted"):
		service.record_device_posture("tenant-a", "device-1", "user-1", trust_state="rooted")
	review_agent = service.register_security_agent(
		"tenant-a",
		"unapproved-agent",
		"Unapproved Agent",
		"codex",
		"exception_reviewer",
		"policy exception review",
		"secops",
		"review exceptions",
		human_approval_required=False,
	)
	assert review_agent["status"] == "pending_review"
	assert review_agent["policy_decision"] == "require_review"
	assert review_agent["review_reasons"] == ["security_agent_human_approval_required"]
	with pytest.raises(PermissionError, match="bytewax_security_stream_required"):
		service.validate_security_lifecycle_batch("tenant-a", "memory", 1)
	denied_batch = [
		item for item in service.list_security_lifecycle_batches("tenant-a")
		if item["status"] == "denied"
	][0]
	assert denied_batch["status"] == "denied"
	assert denied_batch["policy_decision"] == "deny"
	assert denied_batch["review_reasons"] == ["bytewax_security_stream_required"]

	service.record_device_posture("tenant-a", "device-2", "user-1", trust_state="compromised")
	quarantined = service.assess_access("tenant-a", "user-1", "user", 60, device_id="device-2")
	denied = service.assess_access("tenant-a", "user-1", "user", 95)
	malicious = service.assess_access("tenant-a", "user-1", "user", 10, is_known_malicious=True)

	assert quarantined["decision"] == "quarantine"
	assert "compromised_device_quarantined" in quarantined["matched_rules"]
	assert denied["decision"] == "deny"
	assert "critical_risk_denied" in denied["matched_rules"]
	assert malicious["decision"] == "deny"
	assert "known_malicious_network_denied" in malicious["matched_rules"]


def test_exception_and_incident_guardrails_fail_closed():
	service = SecuService()
	policy = service.create_policy("tenant-a", "Privileged access", "secops")
	with pytest.raises(ValueError, match="policy_exception_id_required"):
		service.request_policy_exception(
			tenant_id="tenant-a",
			exception_id="",
			policy_id=policy["id"],
			requested_by="requester",
			reason="Missing exception ID.",
			expires_at="2099-01-01T00:00:00Z",
		)
	with pytest.raises(KeyError, match="security_policy_not_found"):
		service.request_policy_exception(
			tenant_id="tenant-a",
			exception_id="missing-policy",
			policy_id="missing-policy",
			requested_by="requester",
			reason="Policy target does not exist.",
			expires_at="2099-01-01T00:00:00Z",
		)
	with pytest.raises(ValueError, match="policy_exception_expiry_invalid"):
		service.request_policy_exception(
			tenant_id="tenant-a",
			exception_id="bad-expiry",
			policy_id=policy["id"],
			requested_by="requester",
			reason="Invalid expiry.",
			expires_at="not-a-timestamp",
		)
	pending = service.request_policy_exception(
		tenant_id="tenant-a",
		exception_id="self-review",
		policy_id=policy["id"],
		requested_by="requester",
		reason="Temporary break glass.",
		expires_at="2099-01-01T00:00:00Z",
	)
	expired = service.request_policy_exception(
		tenant_id="tenant-a",
		exception_id="expired",
		policy_id=policy["id"],
		requested_by="requester",
		reason="Expired exception.",
		expires_at="2000-01-01T00:00:00Z",
	)

	with pytest.raises(PermissionError, match="independent_exception_reviewer_required"):
		service.decide_policy_exception("tenant-a", pending["id"], "requester", "approved", "Self review.")
	with pytest.raises(ValueError, match="policy_exception_notes_required"):
		service.decide_policy_exception("tenant-a", pending["id"], "security-reviewer", "approved", "")
	with pytest.raises(PermissionError, match="policy_exception_expired"):
		service.decide_policy_exception("tenant-a", expired["id"], "security-reviewer", "approved", "Too late.")
	rejected_expired = service.decide_policy_exception(
		"tenant-a",
		expired["id"],
		"security-reviewer",
		"rejected",
		"Expired exceptions are rejected, not approved.",
	)
	with pytest.raises(ValueError, match="incident_id_required"):
		service.open_incident("tenant-a", "", "Credential exposure", "critical", "soc-analyst", "Disable credentials.")
	with pytest.raises(PermissionError, match="critical_incident_containment_required"):
		service.open_incident("tenant-a", "critical-1", "Credential exposure", "critical", "soc-analyst")
	with pytest.raises(PermissionError, match="critical_incident_containment_required"):
		service.open_incident("tenant-a", "critical-whitespace", "Credential exposure", "critical", "soc-analyst", "   ")

	incident = service.open_incident(
		"tenant-a",
		"critical-2",
		"Credential exposure",
		"critical",
		"soc-analyst",
		containment_plan="Disable credentials.",
	)
	with pytest.raises(ValueError, match="incident_containment_evidence_required"):
		service.contain_incident("tenant-a", incident["id"], "soc-lead", "Disabled credentials.", "")
	with pytest.raises(PermissionError, match="incident_containment_evidence_required"):
		service.resolve_incident(
			"tenant-a",
			incident["id"],
			"soc-lead",
			"Credentials rotated.",
			"Resolution before containment should fail.",
		)
	contained = service.contain_incident(
		"tenant-a",
		incident["id"],
		"soc-lead",
		"Disabled credentials.",
		"audit://incident/critical-2/containment",
	)
	with pytest.raises(ValueError, match="incident_resolution_notes_required"):
		service.resolve_incident("tenant-a", contained["id"], "soc-lead", "Credentials rotated.", "")
	resolved = service.resolve_incident(
		"tenant-a",
		contained["id"],
		"soc-lead",
		"Credentials rotated.",
		"Resolution evidence accepted.",
	)
	with pytest.raises(ValueError, match="security_incident_already_resolved"):
		service.resolve_incident("tenant-a", resolved["id"], "soc-lead", "Repeat resolution.", "Duplicate closure.")

	assert rejected_expired["status"] == "rejected"


def test_api_and_view_models_expose_security_posture_surfaces():
	local_api_service = SecuService()
	api.SERVICE = local_api_service

	policy = api.create_policy({"tenant_id": "tenant-b", "name": "Data access", "owner": "secops"})
	api.record_device_posture({"tenant_id": "tenant-b", "device_id": "device-1", "user_id": "user-1"})
	api.register_threat_indicator({
		"tenant_id": "tenant-b",
		"name": "Suspicious host",
		"indicator_type": "host",
		"value": "example.invalid",
		"severity": "medium",
	})
	api.assess_access({"tenant_id": "tenant-b", "subject_id": "user-1", "risk_score": 30})
	api.record_compliance_control({
		"tenant_id": "tenant-b",
		"control_id": "CC6.1",
		"owner": "secops",
		"compliant": True,
		"evidence_ref": "audit://evidence/1",
	})
	exception = api.request_policy_exception({
		"tenant_id": "tenant-b",
		"id": "exception-1",
		"policy_id": policy["id"],
		"requested_by": "owner",
		"reason": "Temporary external auditor access.",
		"expires_at": "2099-01-01T00:00:00Z",
	})
	api.decide_policy_exception({
		"tenant_id": "tenant-b",
		"id": exception["id"],
		"reviewer": "security-reviewer",
		"decision": "approved",
		"notes": "Approved for audit window.",
	})
	expired = api.request_policy_exception({
		"tenant_id": "tenant-b",
		"id": "expired-exception",
		"policy_id": policy["id"],
		"requested_by": "owner",
		"reason": "Expired exception should not be approved.",
		"expires_at": "2000-01-01T00:00:00Z",
	})
	with pytest.raises(PermissionError, match="policy_exception_expired"):
		api.decide_policy_exception({
			"tenant_id": "tenant-b",
			"id": expired["id"],
			"reviewer": "security-reviewer",
			"decision": "approved",
			"notes": "Caller-supplied now must not bypass expiry.",
			"now": "0",
		})
	incident = api.open_incident({
		"tenant_id": "tenant-b",
		"id": "incident-1",
		"title": "Suspicious admin token",
		"severity": "high",
		"opened_by": "soc-analyst",
	})
	api.contain_incident({
		"tenant_id": "tenant-b",
		"id": incident["id"],
		"actor": "soc-lead",
		"containment_action": "Revoked token.",
		"containment_evidence": "audit://incident/incident-1/containment",
	})
	agent = api.register_security_agent({
		"tenant_id": "tenant-b",
		"id": "risk-agent",
		"name": "Risk Review Agent",
		"runtime": "opencode",
		"role": "risk_reviewer",
		"scope": "risk assessment review",
		"owner": "secops",
		"purpose": "risk evidence summarization",
	})
	batch = api.validate_security_lifecycle_batch({
		"tenant_id": "tenant-b",
		"event_stream": "bytewax",
		"mutation_count": 2,
	})

	status = api.capability_status("tenant-b")
	posture = api.list_security_posture("tenant-b")
	dashboard = views.dashboard_model(tenant_id="tenant-b")
	risk = views.risk_console_model(tenant_id="tenant-b")
	threats = views.threat_console_model(tenant_id="tenant-b")
	policies = views.policy_workbench_model(tenant_id="tenant-b")
	exceptions = views.exception_queue_model(tenant_id="tenant-b")
	incidents = views.incident_response_model(tenant_id="tenant-b")
	quarantine = views.quarantine_console_model(tenant_id="tenant-b")
	compliance = views.compliance_console_model(tenant_id="tenant-b")
	agents = views.security_agents_model(tenant_id="tenant-b")
	audit = views.audit_timeline_model(tenant_id="tenant-b")
	rules = views.rule_workbench_model("tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["policy_count"] == 1
	assert status["policy_exception_count"] == 2
	assert status["open_incident_count"] == 1
	assert status["security_agent_count"] == 1
	assert status["pending_review_count"] == 1
	assert posture["summary"]["active_threat_count"] == 1
	assert posture["security_agents"][0]["id"] == agent["id"]
	assert posture["security_lifecycle_batches"][0]["status"] == "accepted"
	assert posture["pending_reviews"][0]["status"] == "pending"
	assert dashboard["summary"]["assessment_count"] == 1
	assert dashboard["review_evidence"]["deny_behavior"] == "Denied security lifecycle batches persist evidence before PermissionError"
	assert risk["route"] == "/secu/risk"
	assert threats["severity_filters"] == ["info", "low", "medium", "high", "critical"]
	assert policies["security_levels"][-1] == "critical"
	assert {item["status"] for item in exceptions["exceptions"]} == {"approved", "pending"}
	assert len(exceptions["pending"]) == 1
	assert incidents["open_incidents"][0]["status"] == "contained"
	assert quarantine["devices"] == []
	assert compliance["controls"][0]["status"] == "implemented"
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert agents["agents"][0]["runtime"] == "opencode"
	assert batch["accepted"] is True
	assert batch["status"] == "accepted"
	assert audit["events"]
	assert rules["decision_order"] == ["deny", "quarantine", "challenge", "require_review", "allow"]
	assert settings["theme"]["name"] == "secu_zero_trust"
	assert settings["review_evidence"]["pending_queues"]
