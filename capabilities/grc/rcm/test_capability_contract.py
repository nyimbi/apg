"""Focused executable contract tests for the APG RCM capability."""

from __future__ import annotations

import pytest

from .service import GrcRcmService
from .views import (
	compliance_workbench_model,
	control_testing_model,
	dashboard_model,
	governance_board_model,
	risk_register_model,
)


def test_rcm_lifecycle_is_executable() -> None:
	service = GrcRcmService()

	risk = service.register_risk(
		risk_id="risk-cyber-001",
		tenant_id="tenant-a",
		title="Privileged access compromise",
		category="cyber",
		owner_id="owner-risk",
		probability=0.9,
		impact=0.95,
		control_effectiveness=0.1,
		review_recorded=True,
	)
	control = service.register_control(
		control_id="ctrl-pam-review",
		tenant_id="tenant-a",
		name="Privileged access review",
		owner_id="owner-control",
		control_type="detective",
		mapped_risk_ids=[risk["id"]],
		effectiveness=0.75,
	)
	obligation = service.add_compliance_obligation(
		obligation_id="obl-iso-a9",
		tenant_id="tenant-a",
		framework="ISO27001",
		requirement="Access rights are reviewed at planned intervals.",
		owner_id="owner-compliance",
		jurisdiction="global",
		due_date="2026-06-30",
		mapped_control_ids=[control["id"]],
	)
	evidence = service.collect_evidence(
		evidence_id="ev-pam-q1",
		tenant_id="tenant-a",
		source="pam-review-export.csv",
		linked_control_id=control["id"],
		linked_obligation_id=obligation["id"],
	)
	assessment = service.assess_control(
		assessment_id="assess-pam-q1",
		tenant_id="tenant-a",
		control_id=control["id"],
		assessor_id="assessor-1",
		design_effective=True,
		operating_effective=True,
		evidence_refs=[evidence["id"]],
	)
	decision = service.record_governance_decision(
		decision_id="dec-risk-accept-001",
		tenant_id="tenant-a",
		title="Accept residual privileged access risk through Q2",
		decision_type="risk_acceptance",
		approver_id="board-risk-chair",
		related_risk_ids=[risk["id"]],
		rationale="Residual risk accepted while PAM automation is expanded.",
		review_recorded=True,
	)

	assert risk["risk_level"] == "high"
	assert assessment["status"] == "compliant"
	assert decision["approved"] is True
	assert service.dashboard_summary("tenant-a") == {
		"tenant_id": "tenant-a",
		"risk_count": 1,
		"high_risk_count": 1,
		"control_count": 1,
		"obligation_count": 1,
		"assessment_count": 1,
		"failed_assessment_count": 0,
		"governance_decision_count": 1,
		"evidence_count": 1,
		"audit_event_count": 6,
		"overall_status": "attention_required",
	}
	assert len(risk_register_model(service, "tenant-a")["high_priority"]) == 1
	assert control_testing_model(service, "tenant-a")["evidence"][0]["id"] == "ev-pam-q1"
	assert compliance_workbench_model(service, "tenant-a")["obligations"][0]["id"] == "obl-iso-a9"
	assert governance_board_model(service, "tenant-a")["decisions"][0]["id"] == "dec-risk-accept-001"
	assert dashboard_model(service, "tenant-a")["summary"]["audit_event_count"] == 6


def test_rcm_service_enforces_policy_guardrails() -> None:
	service = GrcRcmService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_risk(
			risk_id="risk-no-tenant",
			tenant_id="",
			title="Missing tenant",
			category="operational",
			owner_id="owner",
			probability=0.1,
			impact=0.1,
		)

	with pytest.raises(PermissionError, match="operation_policy_required"):
		service.register_risk(
			risk_id="risk-no-policy",
			tenant_id="tenant-a",
			title="Policy missing",
			category="operational",
			owner_id="owner",
			probability=0.1,
			impact=0.1,
			policy_attached=False,
		)

	with pytest.raises(PermissionError, match="high_risk_review_required"):
		service.register_risk(
			risk_id="risk-high",
			tenant_id="tenant-a",
			title="High risk",
			category="cyber",
			owner_id="owner",
			probability=0.95,
			impact=0.95,
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="risk_owner_required"):
		service.register_risk(
			risk_id="risk-no-owner",
			tenant_id="tenant-a",
			title="No owner",
			category="operational",
			owner_id="",
			probability=0.2,
			impact=0.2,
		)

	with pytest.raises(ValueError, match="risk_probability_out_of_range"):
		service.register_risk(
			risk_id="risk-bad-probability",
			tenant_id="tenant-a",
			title="Bad probability",
			category="operational",
			owner_id="owner",
			probability=1.5,
			impact=0.2,
		)

	service.register_risk(
		risk_id="risk-ok",
		tenant_id="tenant-a",
		title="Vendor concentration",
		category="third_party",
		owner_id="owner",
		probability=0.4,
		impact=0.5,
	)

	with pytest.raises(PermissionError, match="mapped_risk_missing"):
		service.register_control(
			control_id="ctrl-cross-tenant",
			tenant_id="tenant-b",
			name="Cross tenant control",
			owner_id="owner",
			mapped_risk_ids=["risk-ok"],
			effectiveness=0.5,
		)

	service.register_control(
		control_id="ctrl-ok",
		tenant_id="tenant-a",
		name="Vendor risk review",
		owner_id="owner",
		mapped_risk_ids=["risk-ok"],
		effectiveness=0.5,
	)

	with pytest.raises(PermissionError, match="failed_control_requires_evidence"):
		service.assess_control(
			assessment_id="assess-fail-no-evidence",
			tenant_id="tenant-a",
			control_id="ctrl-ok",
			assessor_id="assessor",
			design_effective=False,
			operating_effective=False,
			evidence_refs=[],
			findings=["sample failure"],
		)

	with pytest.raises(PermissionError, match="evidence_encryption_required"):
		service.collect_evidence(
			evidence_id="ev-plain",
			tenant_id="tenant-a",
			source="plain-export.csv",
			linked_control_id="ctrl-ok",
			encrypted=False,
		)

	with pytest.raises(PermissionError, match="evidence_retention_too_short"):
		service.collect_evidence(
			evidence_id="ev-short",
			tenant_id="tenant-a",
			source="short-retention.csv",
			linked_control_id="ctrl-ok",
			retention_days=30,
		)

	high_risk = service.register_risk(
		risk_id="risk-high-reviewed",
		tenant_id="tenant-a",
		title="Material misstatement",
		category="financial",
		owner_id="owner",
		probability=0.9,
		impact=0.9,
		review_recorded=True,
	)
	assert high_risk["risk_level"] == "high"
	with pytest.raises(PermissionError, match="high_risk_governance_decision_requires_rationale"):
		service.record_governance_decision(
			decision_id="dec-no-rationale",
			tenant_id="tenant-a",
			title="Accept high risk",
			decision_type="risk_acceptance",
			approver_id="approver",
			related_risk_ids=["risk-high-reviewed"],
			rationale="",
			review_recorded=True,
		)
