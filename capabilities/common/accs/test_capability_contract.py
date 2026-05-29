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
	assert contract["configuration_schema"]["required"] == ["tenant_id", "standards", "audits", "assistive", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "audits", "findings", "remediation", "assistive", "media", "compliance", "settings"}
	assert contract["theme"]["name"] == "accs_accessibility_ops"


def test_rule_engine_enforces_accs_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "start_audit", "standard_selected": False, "violation_detected": True, "remediation_owner_assigned": False, "published_ui": True, "contrast_passed": False, "media_content_present": True, "captions_available": False, "issue_severity": "critical", "review_recorded": False})
	review_result = evaluate_capability_rules({"tenant_context_present": True, "issue_severity": "critical", "review_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "audit_requires_standard", "violation_requires_remediation_owner", "published_ui_requires_contrast", "media_requires_captions", "critical_issue_requires_review"}
	assert review_result["decision"] == "require_review"


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "accs"
	assert "nlpc" in registration["dependencies"]
	assert registration["ui_components"]["remediation"] == "/accs/remediation"
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
