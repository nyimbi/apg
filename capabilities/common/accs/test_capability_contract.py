"""Regression coverage for the ACCS executable capability contract."""

import pytest

from capabilities.common.accs import register_capability
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
