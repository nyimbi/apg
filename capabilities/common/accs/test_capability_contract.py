"""Regression coverage for the ACCS executable capability contract."""

import pytest

from capabilities.common.accs import register_capability
from capabilities.common.accs import api, views
from capabilities.common.accs.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.accs.service import AccsService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-accs", {"standards": {"default_standard": "EN-301-549"}})

	assert contract["capability"] == "accs"
	assert contract["configuration"]["tenant_id"] == "tenant-accs"
	assert contract["configuration"]["standards"]["default_standard"] == "EN-301-549"
	assert contract["configuration_schema"]["required"] == ["tenant_id", "standards", "audits", "assistive", "accessibility_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert contract["configuration"]["accessibility_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert set(contract["provides"]) >= {"accessibility_audits", "remediation_workflows", "accessibility_exceptions", "accessibility_agents"}
	assert contract["requires"] == ["them", "i18n", "nlpc"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "audits", "findings", "remediation", "exceptions", "assistive", "media", "compliance", "agents", "audit", "analytics", "settings"}
	assert contract["theme"]["name"] == "accs_accessibility_ops"


def test_rule_engine_enforces_accs_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "start_audit", "standard_selected": False, "violation_detected": True, "remediation_owner_assigned": False, "published_ui": True, "contrast_passed": False, "media_content_present": True, "captions_available": False, "issue_severity": "critical", "review_recorded": False})
	review_result = evaluate_capability_rules({"tenant_context_present": True, "issue_severity": "critical", "review_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_accessibility_mutation", "event_stream": "memory"})
	exception_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "record_accessibility_exception", "exception_expiry_present": False, "compensating_controls_present": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "audit_requires_standard", "violation_requires_remediation_owner", "published_ui_requires_contrast", "media_requires_captions", "critical_issue_requires_review"}
	assert review_result["decision"] == "require_review"
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_accessibility_mutation_requires_bytewax"]
	assert exception_result["decision"] == "deny"
	assert set(exception_result["matched_rules"]) == {"accessibility_exception_requires_expiry", "accessibility_exception_requires_compensating_controls"}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "accs"
	assert "nlpc" in registration["dependencies"]
	assert "bytewax" in registration["optional_dependencies"]
	assert registration["ui_components"]["remediation"] == "/accs/remediation"
	assert registration["ui_components"]["exceptions"] == "/accs/exceptions"
	assert registration["ui_components"]["agents"] == "/accs/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "accs:remediate" in registration["permissions"]


def test_service_runs_audits_and_tracks_remediation():
	service = AccsService()
	service.register_target(
		target_id="checkout",
		tenant_id="tenant-accs",
		surface="Checkout Screen",
		route="/checkout",
		owner="product-owner",
		published_ui=True,
		contrast_ratio=3.2,
		media_content_present=True,
		captions_available=False,
	)
	audit = service.run_audit(
		audit_id="audit-1",
		tenant_id="tenant-accs",
		standard_id="wcag_2_2_aa",
		target_ids=["checkout"],
		remediation_owner="accessibility-lead",
	)
	remediation = service.update_remediation(
		finding_id=audit["finding_ids"][0],
		status="in_progress",
		due_date="2026-06-15",
	)
	summary = service.compliance_summary("tenant-accs")

	assert audit["summary"]["finding_count"] == 2
	assert audit["summary"]["critical_or_high_count"] == 2
	assert remediation["status"] == "in_progress"
	assert summary["target_count"] == 1
	assert summary["finding_count"] == 2
	assert summary["remediation_count"] == 2
	assert summary["review_count"] == 0


def test_service_blocks_invalid_audits_and_reports_publication_guardrails():
	service = AccsService()
	service.register_target(
		target_id="media",
		tenant_id="tenant-accs",
		surface="Training Video",
		route="/training",
		owner="learning-owner",
		published_ui=True,
		contrast_ratio=3.8,
		media_content_present=True,
		captions_available=False,
	)

	with pytest.raises(PermissionError, match="audit_standard_required"):
		service.run_audit(
			audit_id="missing-standard",
			tenant_id="tenant-accs",
			standard_id="missing",
			target_ids=["media"],
		)

	with pytest.raises(PermissionError, match="remediation_owner_required"):
		service.run_audit(
			audit_id="missing-owner",
			tenant_id="tenant-accs",
			standard_id="wcag_2_2_aa",
			target_ids=["media"],
		)

	report = service.validate_publication("media", tenant_id="tenant-accs")
	reasons = {action["reason"] for action in report["rule_result"]["actions"]}

	assert report["publishable"] is False
	assert reasons == {"contrast_validation_required", "captions_required"}


def test_critical_finding_review_and_closure_lifecycle():
	service = AccsService()
	service.register_target(
		target_id="admin-panel",
		tenant_id="tenant-accs",
		surface="Admin Panel",
		route="/admin",
		owner="platform-owner",
		keyboard_navigation_present=False,
	)

	audit = service.run_audit(
		audit_id="critical-audit",
		tenant_id="tenant-accs",
		standard_id="wcag_2_2_aa",
		target_ids=["admin-panel"],
		remediation_owner="accessibility-lead",
	)
	finding_id = audit["finding_ids"][0]
	finding = service.list_findings("tenant-accs")[0]
	review_queue = views.review_queue_model(service, "tenant-accs")

	assert finding["severity"] == "critical"
	assert finding["status"] == "review_required"
	assert finding["review_required"] is True
	assert review_queue["findings_requiring_review"][0]["id"] == finding_id

	with pytest.raises(PermissionError, match="critical_accessibility_review_required"):
		service.close_finding(finding_id, tenant_id="tenant-accs", resolution="Keyboard trap fixed.")

	review = service.record_review(
		finding_id=finding_id,
		tenant_id="tenant-accs",
		reviewer="accessibility-reviewer",
		decision="approved",
		notes="Keyboard navigation remediation evidence accepted.",
	)
	closed = service.close_finding(
		finding_id=finding_id,
		tenant_id="tenant-accs",
		resolution="Keyboard navigation verified through deterministic tab-order check.",
	)
	evidence = views.compliance_evidence_model(service, "tenant-accs")

	assert review["decision"] == "approved"
	assert closed["status"] == "closed"
	assert closed["resolution"].startswith("Keyboard navigation verified")
	assert evidence["summary"]["review_count"] == 1
	assert {event["event_type"] for event in evidence["audit_events"]} >= {
		"finding_recorded",
		"finding_review_recorded",
		"finding_closed",
	}


def test_critical_finding_closure_enforces_tenant_and_resolution_guardrails():
	service = AccsService()
	finding = service.record_finding(
		finding_id="manual-critical",
		tenant_id="tenant-accs",
		target_id="manual",
		rule="keyboard_navigation_required",
		severity="critical",
		description="Manual review found keyboard trap.",
		remediation_owner="accessibility-lead",
	)

	with pytest.raises(KeyError, match="unknown accessibility finding for tenant"):
		service.close_finding(finding["id"], tenant_id="other-tenant", resolution="Fixed.")

	service.record_review(
		finding_id=finding["id"],
		tenant_id="tenant-accs",
		reviewer="accessibility-reviewer",
		decision="needs_work",
		notes="Manual evidence incomplete.",
	)

	with pytest.raises(PermissionError, match="critical_accessibility_review_not_approved"):
		service.close_finding(finding["id"], tenant_id="tenant-accs", resolution="Evidence incomplete.")

	assert views.review_queue_model(service, "tenant-accs")["findings_requiring_review"][0]["id"] == finding["id"]

	service.record_review(
		finding_id=finding["id"],
		tenant_id="tenant-accs",
		reviewer="accessibility-reviewer",
		decision="approved",
		notes="Manual evidence accepted.",
	)
	with pytest.raises(ValueError, match="resolution evidence is required"):
		service.close_finding(finding["id"], tenant_id="tenant-accs", resolution="")


def test_accessibility_exception_lifecycle_and_publication_readiness():
	service = AccsService()
	service.register_target(
		target_id="release-page",
		tenant_id="tenant-exception",
		surface="Release Page",
		route="/release",
		owner="release-owner",
		published_ui=True,
		contrast_ratio=3.7,
	)
	audit = service.run_audit(
		audit_id="release-audit",
		tenant_id="tenant-exception",
		standard_id="wcag_2_2_aa",
		target_ids=["release-page"],
		remediation_owner="accessibility-lead",
	)
	finding_id = audit["finding_ids"][0]

	with pytest.raises(PermissionError, match="accessibility_exception_expiry_required"):
		service.record_accessibility_exception(
			exception_id="exception-missing-expiry",
			tenant_id="tenant-exception",
			finding_id=finding_id,
			approver="accessibility-director",
			reason="Awaiting brand palette release.",
			expires_on="",
			compensating_controls=["high contrast mode enabled"],
		)
	with pytest.raises(PermissionError, match="accessibility_exception_compensating_controls_required"):
		service.record_accessibility_exception(
			exception_id="exception-missing-controls",
			tenant_id="tenant-exception",
			finding_id=finding_id,
			approver="accessibility-director",
			reason="Awaiting brand palette release.",
			expires_on="2099-12-31",
			compensating_controls=[],
		)
	with pytest.raises(PermissionError, match="accessibility_exception_expired"):
		service.record_accessibility_exception(
			exception_id="exception-expired",
			tenant_id="tenant-exception",
			finding_id=finding_id,
			approver="accessibility-director",
			reason="Expired risk acceptance.",
			expires_on="2000-01-01",
			compensating_controls=["manual support path enabled"],
		)
	with pytest.raises(KeyError, match="unknown accessibility finding for tenant"):
		service.record_accessibility_exception(
			exception_id="exception-cross-tenant",
			tenant_id="tenant-other",
			finding_id=finding_id,
			approver="accessibility-director",
			reason="Wrong tenant should not see this finding.",
			expires_on="2099-12-31",
			compensating_controls=["manual support path enabled"],
		)

	exception = service.record_accessibility_exception(
		exception_id="exception-1",
		tenant_id="tenant-exception",
		finding_id=finding_id,
		approver="accessibility-director",
		reason="Brand palette fix is scheduled after release freeze.",
		expires_on="2099-12-31",
		compensating_controls=["high contrast mode enabled", "support team release note published"],
	)
	report = service.validate_publication("release-page", tenant_id="tenant-exception")
	exceptions_model = views.accessibility_exceptions_model(service, "tenant-exception")
	evidence = views.compliance_evidence_model(service, "tenant-exception")

	assert exception["status"] == "approved"
	assert report["publishable"] is False
	assert report["publishable_with_exception"] is True
	assert report["active_exceptions"][0]["id"] == "exception-1"
	assert exceptions_model["accessibility_exceptions"][0]["finding_id"] == finding_id
	assert evidence["summary"]["exception_count"] == 1
	assert "accessibility_exception_recorded" in {event["event_type"] for event in evidence["audit_events"]}


def test_accessibility_agents_tenant_scope_and_bytewax_guardrail():
	service = AccsService()
	service.register_target(
		target_id="shared",
		tenant_id="tenant-a",
		surface="Tenant A Screen",
		route="/a",
		owner="owner-a",
	)
	service.register_target(
		target_id="shared",
		tenant_id="tenant-b",
		surface="Tenant B Screen",
		route="/b",
		owner="owner-b",
	)
	agent = service.register_accessibility_agent(
		agent_id="reviewer-1",
		tenant_id="tenant-a",
		name="Accessibility Reviewer",
		runtime="claude-code",
		role="release-reviewer",
		scope="release gates and critical-finding evidence",
		contribution_disclosed=True,
		policy_ref="accs-agent-policy",
	)
	batch = service.validate_batch_accessibility_mutation(
		tenant_id="tenant-a",
		event_stream="bytewax",
		mutation_count=2,
	)
	dashboard = views.dashboard_model(service, "tenant-a")
	agents = views.accessibility_agents_model(service, "tenant-a")
	analytics = views.analytics_model(service, "tenant-a")
	settings = views.settings_model("tenant-a")

	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "release_reviewer"
	assert batch["accepted"] is True
	assert len(service.list_targets("tenant-a")) == 1
	assert len(service.list_targets("tenant-b")) == 1
	assert dashboard["accessibility_agents"][0]["id"] == "reviewer-1"
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert analytics["summary"]["accessibility_agent_count"] == 1
	assert settings["streaming"]["processor"] == "bytewax"

	with pytest.raises(PermissionError, match="accessibility_agent_runtime_not_supported"):
		service.register_accessibility_agent(
			agent_id="bad-runtime",
			tenant_id="tenant-a",
			name="Bad Runtime",
			runtime="unsupported",
			role="audit_reviewer",
			scope="audits",
		)

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_accessibility_mutation(
			tenant_id="tenant-a",
			event_stream="memory",
			mutation_count=1,
		)


def test_api_helpers_expose_review_and_closure_lifecycle():
	target = api.register_target({
		"id": "api-admin",
		"tenant_id": "tenant-api-accs",
		"surface": "API Admin",
		"route": "/api-admin",
		"owner": "api-owner",
		"keyboard_navigation_present": False,
	})
	audit = api.run_audit({
		"id": "api-audit",
		"tenant_id": target["tenant_id"],
		"target_ids": [target["id"]],
		"remediation_owner": "api-accessibility-lead",
	})
	review = api.record_review({
		"finding_id": audit["finding_ids"][0],
		"tenant_id": target["tenant_id"],
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "API lifecycle review accepted.",
	})
	closed = api.close_finding({
		"finding_id": audit["finding_ids"][0],
		"tenant_id": target["tenant_id"],
		"resolution": "API target keyboard path fixed.",
	})

	assert review["reviewer"] == "api-reviewer"
	assert closed["status"] == "closed"
	assert api.list_reviews(target["tenant_id"])[0]["finding_id"] == audit["finding_ids"][0]


def test_api_helpers_expose_accessibility_exceptions():
	target = api.register_target({
		"id": "api-exception-target",
		"tenant_id": "tenant-api-exception",
		"surface": "API Exception Target",
		"route": "/exception",
		"owner": "api-owner",
		"published_ui": True,
		"contrast_ratio": 3.5,
	})
	audit = api.run_audit({
		"id": "api-exception-audit",
		"tenant_id": target["tenant_id"],
		"target_ids": [target["id"]],
		"remediation_owner": "api-accessibility-lead",
	})
	exception = api.record_accessibility_exception({
		"id": "api-exception",
		"tenant_id": target["tenant_id"],
		"finding_id": audit["finding_ids"][0],
		"approver": "api-approver",
		"reason": "Temporary release acceptance.",
		"expires_on": "2099-12-31",
		"compensating_controls": ["release note", "support escalation"],
	})

	assert exception["approver"] == "api-approver"
	assert api.list_accessibility_exceptions(target["tenant_id"])[0]["id"] == "api-exception"


def test_api_helpers_expose_accessibility_agents():
	agent = api.register_accessibility_agent({
		"id": "api-agent",
		"tenant_id": "tenant-api-agent",
		"name": "API Accessibility Agent",
		"runtime": "opencode",
		"role": "audit_reviewer",
		"scope": "audit triage",
		"contribution_disclosed": True,
	})
	batch = api.validate_batch_accessibility_mutation({
		"tenant_id": "tenant-api-agent",
		"event_stream": "bytewax",
		"mutation_count": 1,
	})

	assert agent["runtime"] == "opencode"
	assert api.list_accessibility_agents("tenant-api-agent")[0]["id"] == "api-agent"
	assert batch["accepted"] is True
