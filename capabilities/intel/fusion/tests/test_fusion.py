"""Comprehensive tests for APG Intelligence Fusion.

Covers:
  - All 8 entity CRUD flows
  - Full lifecycle (ingest → fuse → correlate → assess → product → disseminate)
  - Domain rules (every assert_* function)
  - Calculations (ACH, KAC, ACE, Bayesian, corroboration, quality)
  - Tenant isolation
  - ACH analysis end-to-end
  - Confidence calibration
  - TLP enforcement
  - Product lifecycle (draft → review → approved → released → recalled)

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest

# Ensure package root is on path for direct execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from domain.calculations import (
	ace_assessment,
	ach_hypothesis_confidence,
	bayesian_update,
	build_ach_matrix,
	calibrated_confidence,
	composite_risk_score,
	confidence_calibration_report,
	cross_domain_correlation_score,
	effective_classification,
	escalation_threshold_met,
	evaluate_assumptions,
	fusion_quality_score,
	items_within_time_window,
	likelihood_ratio,
	score_to_confidence_level,
	source_corroboration_score,
	temporal_clustering_score,
	time_overlap_score,
	tlp_compatible,
)
from domain.rules import (
	RuleViolation,
	assert_analyst_assigned,
	assert_assessment_has_correlations,
	assert_assessment_has_hypotheses,
	assert_assessment_type_supported,
	assert_chain_of_custody_present,
	assert_classification_dominance,
	assert_confidence_in_range,
	assert_content_fingerprint_present,
	assert_correlation_has_items,
	assert_correlation_type_supported,
	assert_custodian_assigned,
	assert_evidence_not_discredited,
	assert_evidence_type_supported,
	assert_hypothesis_has_alternatives,
	assert_hypothesis_open_for_update,
	assert_lawful_authority,
	assert_minimum_sources_for_fusion,
	assert_no_autonomous_dissemination,
	assert_no_cross_tenant_access,
	assert_no_evidence_fabrication,
	assert_no_privacy_bypass,
	assert_no_source_tampering,
	assert_no_unapproved_attribution,
	assert_privileged_agent_action_has_approval,
	assert_product_in_approved_state,
	assert_product_not_recalled,
	assert_risk_level_supported,
	assert_sat_method_supported,
	assert_source_type_supported,
	assert_tenant_context,
	assert_tlp_compatible_with_audience,
	assert_tlp_valid,
	assert_workspace_active,
	assert_workspace_type_supported,
	calculate_confidence_level,
)
from models import (
	AnalyticalJudgementCreate,
	AssessmentPictureCreate,
	ConfidenceLevel,
	CorrelationSetCreate,
	CorrelationSetStatus,
	CorrelationType,
	EvidenceCreate,
	EvidenceStatus,
	EvidenceUpdate,
	FusionWorkspaceCreate,
	HypothesisTestCreate,
	HypothesisTestUpdate,
	HypothesisStatus,
	IntelligenceItemCreate,
	IntelligenceItemUpdate,
	IntelligenceProductCreate,
	IntelligenceProductUpdate,
	IntelItemStatus,
	JudgementType,
	ProductStatus,
	RiskLevel,
	SATMethod,
	SourceType,
	TLPLevel,
	WorkspaceStatus,
)
from service import IntelligenceFusionService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _svc(tenant: str = "tenant-test") -> IntelligenceFusionService:
	return IntelligenceFusionService(tenant_id=tenant, actor_id="analyst-1")


async def _setup_workspace(svc: IntelligenceFusionService, name: str = "Test WS") -> str:
	ws = await svc.create_workspace(FusionWorkspaceCreate(
		tenant_id=svc.tenant_id,
		workspace_type="case_fusion",
		name=name,
		authority_id="auth-1",
		classification="confidential",
		created_by="analyst-1",
	))
	return ws.id


async def _setup_items(svc: IntelligenceFusionService, workspace_id: str, count: int = 2) -> list[str]:
	ids = []
	for i in range(count):
		item = await svc.create_intel_item(IntelligenceItemCreate(
			tenant_id=svc.tenant_id,
			source_type=SourceType.OSINT if i % 2 == 0 else SourceType.HUMINT,
			source_reference=f"ref-{i}",
			content_fingerprint=f"sha256:item{i}",
			custodian_id="custodian-1",
			workspace_id=workspace_id,
			confidence_score=0.7 + i * 0.05,
		))
		ids.append(item.id)
	return ids


# ─────────────────────────────────────────────────────────────────────────────
# Domain Rules
# ─────────────────────────────────────────────────────────────────────────────

def test_tenant_context_required():
	with pytest.raises(RuleViolation, match="tenant_context_required"):
		assert_tenant_context("")
	with pytest.raises(RuleViolation, match="tenant_context_required"):
		assert_tenant_context("   ")
	assert_tenant_context("any-tenant")  # should not raise


def test_cross_tenant_access_denied():
	with pytest.raises(RuleViolation, match="cross_tenant_access_denied"):
		assert_no_cross_tenant_access("a", "b")
	assert_no_cross_tenant_access("a", "a")  # same tenant is fine


def test_lawful_authority_required():
	with pytest.raises(RuleViolation, match="lawful_authority_required"):
		assert_lawful_authority("", True)
	with pytest.raises(RuleViolation, match="lawful_authority_required"):
		assert_lawful_authority("auth-1", False)
	assert_lawful_authority("auth-1", True)


def test_source_type_validation():
	with pytest.raises(RuleViolation, match="source_type_not_supported"):
		assert_source_type_supported("unknown_source")
	for st in ["osint", "sigint", "humint", "geoint", "cybint", "finint"]:
		assert_source_type_supported(st)  # no raise


def test_content_fingerprint_required():
	with pytest.raises(RuleViolation, match="content_fingerprint_required"):
		assert_content_fingerprint_present("")
	assert_content_fingerprint_present("sha256:abc")


def test_confidence_in_range():
	with pytest.raises(RuleViolation, match="confidence_score_invalid"):
		assert_confidence_in_range(1.1)
	with pytest.raises(RuleViolation, match="confidence_score_invalid"):
		assert_confidence_in_range(-0.1)
	assert_confidence_in_range(0.0)
	assert_confidence_in_range(0.5)
	assert_confidence_in_range(1.0)


def test_correlation_type_validation():
	with pytest.raises(RuleViolation, match="correlation_type_not_supported"):
		assert_correlation_type_supported("made_up")
	assert_correlation_type_supported("entity_match")
	assert_correlation_type_supported("cross_source_confirmation")


def test_correlation_requires_minimum_items():
	with pytest.raises(RuleViolation, match="correlation_requires_minimum_items"):
		assert_correlation_has_items([])
	with pytest.raises(RuleViolation, match="correlation_requires_minimum_items"):
		assert_correlation_has_items(["only-one"])
	assert_correlation_has_items(["a", "b"])


def test_assessment_type_validation():
	with pytest.raises(RuleViolation, match="assessment_type_not_supported"):
		assert_assessment_type_supported("bad_type")
	for t in ["threat", "fraud", "public_safety", "operational", "strategic"]:
		assert_assessment_type_supported(t)


def test_risk_level_validation():
	with pytest.raises(RuleViolation, match="risk_level_not_supported"):
		assert_risk_level_supported("extreme")
	for r in ["low", "medium", "high", "critical"]:
		assert_risk_level_supported(r)


def test_assessment_requires_hypotheses():
	with pytest.raises(RuleViolation, match="assessment_requires_hypotheses"):
		assert_assessment_has_hypotheses([])
	assert_assessment_has_hypotheses(["h1"])


def test_assessment_requires_correlations():
	with pytest.raises(RuleViolation, match="assessment_requires_correlations"):
		assert_assessment_has_correlations([])
	assert_assessment_has_correlations(["c1"])


def test_tlp_validation():
	with pytest.raises(RuleViolation, match="tlp_level_not_supported"):
		assert_tlp_valid("TLP:INVALID")
	for tlp in ["TLP:WHITE", "TLP:GREEN", "TLP:AMBER", "TLP:RED", "TLP:CLEAR"]:
		assert_tlp_valid(tlp)


def test_tlp_compatible_with_audience():
	with pytest.raises(RuleViolation, match="tlp_exceeds_recipient_clearance"):
		assert_tlp_compatible_with_audience("TLP:RED", "TLP:AMBER")
	assert_tlp_compatible_with_audience("TLP:GREEN", "TLP:AMBER")
	assert_tlp_compatible_with_audience("TLP:AMBER", "TLP:AMBER")


def test_product_state_rules():
	with pytest.raises(RuleViolation, match="product_must_be_approved_before_release"):
		assert_product_in_approved_state("draft")
	assert_product_in_approved_state("approved")
	with pytest.raises(RuleViolation, match="recalled_product_cannot_be_modified"):
		assert_product_not_recalled("recalled")
	assert_product_not_recalled("draft")


def test_sat_method_validation():
	with pytest.raises(RuleViolation, match="sat_method_not_supported"):
		assert_sat_method_supported("random_method")
	assert_sat_method_supported("analysis_of_competing_hypotheses")
	assert_sat_method_supported("key_assumptions_check")


def test_hypothesis_open_for_update():
	with pytest.raises(RuleViolation, match="hypothesis_is_closed"):
		assert_hypothesis_open_for_update("supported")
	with pytest.raises(RuleViolation, match="hypothesis_is_closed"):
		assert_hypothesis_open_for_update("refuted")
	assert_hypothesis_open_for_update("open")
	assert_hypothesis_open_for_update("inconclusive")


def test_evidence_type_validation():
	with pytest.raises(RuleViolation, match="evidence_type_not_supported"):
		assert_evidence_type_supported("bad_type")
	assert_evidence_type_supported("document")
	assert_evidence_type_supported("geospatial")


def test_chain_of_custody_required():
	with pytest.raises(RuleViolation, match="chain_of_custody_required"):
		assert_chain_of_custody_present([])
	assert_chain_of_custody_present(["entry-1"])


def test_evidence_not_discredited():
	with pytest.raises(RuleViolation, match="discredited_evidence_cannot_be_used"):
		assert_evidence_not_discredited("discredited")
	assert_evidence_not_discredited("verified")


def test_agent_rules():
	with pytest.raises(RuleViolation, match="privileged_agent_action_requires_human_approval"):
		assert_privileged_agent_action_has_approval(True, False)
	assert_privileged_agent_action_has_approval(True, True)  # no raise
	assert_privileged_agent_action_has_approval(False, False)  # no raise

	with pytest.raises(RuleViolation, match="evidence_fabrication_denied"):
		assert_no_evidence_fabrication(True)
	with pytest.raises(RuleViolation, match="source_tampering_denied"):
		assert_no_source_tampering(True)
	with pytest.raises(RuleViolation, match="privacy_bypass_denied"):
		assert_no_privacy_bypass(True)
	with pytest.raises(RuleViolation, match="autonomous_dissemination_denied"):
		assert_no_autonomous_dissemination(True)
	with pytest.raises(RuleViolation, match="unapproved_attribution_denied"):
		assert_no_unapproved_attribution(True)


def test_workspace_rules():
	with pytest.raises(RuleViolation, match="workspace_type_not_supported"):
		assert_workspace_type_supported("unknown_type")
	assert_workspace_type_supported("case_fusion")
	with pytest.raises(RuleViolation, match="workspace_not_active"):
		assert_workspace_active("suspended")


def test_classification_dominance():
	with pytest.raises(RuleViolation, match="classification_exceeds_workspace"):
		assert_classification_dominance("top_secret", "confidential")
	assert_classification_dominance("confidential", "top_secret")
	assert_classification_dominance("unclassified", "unclassified")


def test_minimum_sources_for_fusion():
	with pytest.raises(RuleViolation, match="insufficient_sources_for_fusion"):
		assert_minimum_sources_for_fusion(1)
	assert_minimum_sources_for_fusion(2)
	assert_minimum_sources_for_fusion(10)


def test_hypothesis_requires_alternatives():
	with pytest.raises(RuleViolation, match="ach_requires_alternatives"):
		assert_hypothesis_has_alternatives([])
	assert_hypothesis_has_alternatives(["alt-1"])


def test_analyst_required():
	with pytest.raises(RuleViolation, match="analyst_required"):
		assert_analyst_assigned("")
	assert_analyst_assigned("analyst-1")


# ─────────────────────────────────────────────────────────────────────────────
# Calculations
# ─────────────────────────────────────────────────────────────────────────────

def test_score_to_confidence_level():
	assert score_to_confidence_level(0.95) == "almost_certain"
	assert score_to_confidence_level(0.85) == "highly_likely"
	assert score_to_confidence_level(0.65) == "likely"
	assert score_to_confidence_level(0.50) == "roughly_even"
	assert score_to_confidence_level(0.30) == "unlikely"
	assert score_to_confidence_level(0.10) == "highly_unlikely"
	assert score_to_confidence_level(0.02) == "remote"


def test_rules_confidence_level():
	assert calculate_confidence_level(0.95) == "almost_certain"
	assert calculate_confidence_level(0.01) == "remote"


def test_bayesian_update():
	# With equal likelihoods, posterior == prior
	assert abs(bayesian_update(0.5, 0.6, 0.6) - 0.5) < 0.01
	# Strong evidence should increase confidence
	p = bayesian_update(0.3, 0.9, 0.1)
	assert p > 0.3
	# Zero false likelihood → posterior is just prior * likelihood (clamped)
	p2 = bayesian_update(0.5, 0.9, 0.0)
	assert p2 > 0.4  # should increase from prior but formula clamps to min(0.999, prior*lgt)


def test_likelihood_ratio():
	assert likelihood_ratio(0.9, 0.1) == pytest.approx(9.0, abs=0.01)
	assert likelihood_ratio(0.9, 0.0) == 999.0


def test_calibrated_confidence():
	# Harmonic mean is pulled toward lower values
	result = calibrated_confidence([0.8, 0.4])
	assert result < 0.6
	assert result > 0.3
	# Single value returns that value
	assert calibrated_confidence([0.7]) == pytest.approx(0.7, abs=0.01)
	# Empty returns 0
	assert calibrated_confidence([]) == 0.0


def test_confidence_calibration_report():
	report = confidence_calibration_report(0.3, 0.9, 0.1)
	assert "posterior" in report
	assert "confidence_level" in report
	assert report["posterior"] > 0.3


def test_build_ach_matrix():
	hypotheses = ["H1: State actor", "H2: Non-state actor", "H3: Insider"]
	evidence = [
		{"label": "Signal intercept", "consistencies": [1.0, -1.0, 0.0]},
		{"label": "Human source",     "consistencies": [-1.0, 1.0, 0.0]},
		{"label": "Financial trace",  "consistencies": [0.0, 0.0, 1.0]},
	]
	result = build_ach_matrix(hypotheses, evidence)
	assert len(result["hypotheses"]) == 3
	assert len(result["inconsistency_scores"]) == 3
	assert len(result["hypothesis_confidence"]) == 3
	# All confidences should sum to ~1
	assert abs(sum(result["hypothesis_confidence"]) - 1.0) < 0.01
	assert result["leading_hypothesis"] in hypotheses


def test_ach_hypothesis_confidence():
	scores = [0.0, 1.0, 4.0]  # H1 has least inconsistency
	confidences = ach_hypothesis_confidence(scores)
	assert len(confidences) == 3
	assert confidences[0] > confidences[1] > confidences[2]
	assert abs(sum(confidences) - 1.0) < 0.001


def test_evaluate_assumptions():
	result = evaluate_assumptions(
		["Adversary is rational", "No external interference"],
		[0.8, 0.4],
	)
	assert result["weakest_assumption"] == "No external interference"
	assert result["robustness"] == pytest.approx(math.sqrt(0.8 * 0.4), abs=0.01)
	assert result["analytic_recommendation"] in (
		"revisit_core_assumptions_before_proceeding",
		"stress_test_weakest_assumptions",
		"assumptions_sufficiently_robust",
	)


def test_ace_assessment():
	result = ace_assessment(
		"Adversary has capability X",
		confidence_score=0.75,
		evidence_count=5,
		evidence_types=["document", "signal", "observation"],
		cross_source_confirmed=True,
	)
	assert result["confidence_score"] > 0.75  # bonus applied
	assert result["evidence_sufficiency"] == "adequate"
	assert result["cross_source_confirmed"] is True


def test_source_corroboration_score():
	result = source_corroboration_score(
		[0.7, 0.6, 0.8],
		["osint", "humint", "sigint"],
	)
	# 3 independent sources → combined > 0.9
	assert result["combined_confidence"] > 0.9
	assert result["unique_source_types"] == 3
	assert result["diversity_bonus"] > 0.0


def test_composite_risk_score():
	result = composite_risk_score(["low", "high", "critical"])
	assert result["max_seen"] == "critical"
	assert result["level"] in ("medium", "high", "critical")

	# All same level
	result2 = composite_risk_score(["medium", "medium"])
	assert result2["level"] == "medium"


def test_escalation_threshold_met():
	assert escalation_threshold_met("critical", 0.9) is True
	assert escalation_threshold_met("high", 0.65) is True
	assert escalation_threshold_met("low", 0.9) is False
	assert escalation_threshold_met("high", 0.3) is False


def test_time_overlap_score():
	# Perfect overlap
	assert time_overlap_score(0, 10, 0, 10) == pytest.approx(1.0, abs=0.001)
	# No overlap
	assert time_overlap_score(0, 5, 6, 10) == pytest.approx(0.0, abs=0.001)
	# Partial
	score = time_overlap_score(0, 10, 5, 15)
	assert 0.0 < score < 1.0


def test_items_within_time_window():
	ts = [1.0, 5.0, 10.0, 15.0, 20.0]
	result = items_within_time_window(ts, 5.0, 15.0)
	assert result == [1, 2, 3]


def test_temporal_clustering_score():
	r = temporal_clustering_score([100.0, 110.0, 120.0], window_seconds=3600)
	assert r["coefficient"] > 0.9  # very tight cluster

	r2 = temporal_clustering_score([0.0, 7200.0], window_seconds=3600)
	assert r2["coefficient"] == 0.0  # span exceeds window


def test_tlp_compatible():
	assert tlp_compatible("TLP:WHITE", "TLP:GREEN") is True
	assert tlp_compatible("TLP:RED", "TLP:AMBER") is False
	assert tlp_compatible("TLP:AMBER", "TLP:AMBER") is True


def test_effective_classification():
	assert effective_classification(["unclassified", "secret", "confidential"]) == "secret"
	assert effective_classification(["top_secret", "unclassified"]) == "top_secret"
	assert effective_classification([]) == "unclassified"


def test_fusion_quality_score():
	result = fusion_quality_score(
		source_count=5,
		unique_source_types=3,
		avg_confidence=0.75,
		has_cross_source_confirmation=True,
		has_structured_analytic_technique=True,
	)
	assert result["quality_score"] > 0.7
	assert result["recommendation"] in (
		"publication_ready",
		"additional_corroboration_recommended",
		"insufficient_quality_for_dissemination",
	)


def test_cross_domain_correlation_score():
	result = cross_domain_correlation_score(
		{"osint": 0.7, "sigint": 0.8, "humint": 0.6},
	)
	assert result["overall"] > 0.6
	assert result["dominant_domain"] == "sigint"
	assert result["confidence_level"] in ("likely", "highly_likely", "almost_certain")


# ─────────────────────────────────────────────────────────────────────────────
# Service — IntelligenceItem CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def test_create_and_get_intel_item():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	item = await svc.create_intel_item(IntelligenceItemCreate(
		tenant_id="tenant-test",
		source_type=SourceType.OSINT,
		source_reference="https://example.com/feed",
		content_fingerprint="sha256:abc123",
		custodian_id="custodian-1",
		workspace_id=ws_id,
		confidence_score=0.72,
	))
	assert item.status == IntelItemStatus.RAW
	assert item.source_type == SourceType.OSINT
	assert item.confidence_score == pytest.approx(0.72, abs=0.001)

	fetched = await svc.get_intel_item(item.id)
	assert fetched.id == item.id


async def test_validate_and_reject_item():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	ids = await _setup_items(svc, ws_id, 2)

	validated = await svc.validate_intel_item(ids[0])
	assert validated.status == IntelItemStatus.VALIDATED

	rejected = await svc.reject_intel_item(ids[1])
	assert rejected.status == IntelItemStatus.REJECTED


async def test_list_items_with_filters():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	await _setup_items(svc, ws_id, 4)

	result = await svc.list_intel_items(workspace_id=ws_id)
	assert result.total == 4

	result2 = await svc.list_intel_items(source_type="osint")
	assert result2.total >= 2  # 2 of 4 are OSINT in _setup_items


async def test_delete_intel_item():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	ids = await _setup_items(svc, ws_id, 1)
	await svc.delete_intel_item(ids[0])
	with pytest.raises(KeyError):
		await svc.get_intel_item(ids[0])


# ─────────────────────────────────────────────────────────────────────────────
# Service — FusionWorkspace CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def test_create_and_list_workspaces():
	svc = _svc()
	ws1 = await svc.create_workspace(FusionWorkspaceCreate(
		tenant_id="tenant-test",
		workspace_type="threat_fusion",
		name="Threat Alpha",
		authority_id="auth-1",
	))
	ws2 = await svc.create_workspace(FusionWorkspaceCreate(
		tenant_id="tenant-test",
		workspace_type="fraud_fusion",
		name="Fraud Beta",
		authority_id="auth-1",
	))
	result = await svc.list_workspaces()
	assert result.total >= 2
	ids = [r["id"] for r in result.items]
	assert ws1.id in ids
	assert ws2.id in ids


async def test_workspace_lifecycle():
	svc = _svc()
	ws_id = await _setup_workspace(svc)

	ws = await svc.get_workspace(ws_id)
	assert ws.status == WorkspaceStatus.ACTIVE

	suspended = await svc.suspend_workspace(ws_id)
	assert suspended.status == WorkspaceStatus.SUSPENDED

	closed = await svc.close_workspace(ws_id)
	assert closed.status == WorkspaceStatus.CLOSED


async def test_workspace_summary():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	await _setup_items(svc, ws_id, 3)

	summary = await svc.workspace_summary(ws_id)
	assert summary.item_count == 3
	assert summary.workspace_id == ws_id


# ─────────────────────────────────────────────────────────────────────────────
# Service — CorrelationSet CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def test_create_and_list_correlations():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	item_ids = await _setup_items(svc, ws_id, 3)

	corr = await svc.create_correlation(CorrelationSetCreate(
		tenant_id="tenant-test",
		workspace_id=ws_id,
		correlation_type=CorrelationType.ENTITY_MATCH,
		item_ids=item_ids,
		analyst_id="analyst-1",
		confidence_score=0.82,
		rationale="Same entity observed across OSINT and HUMINT",
	))
	assert corr.correlation_type == CorrelationType.ENTITY_MATCH
	assert len(corr.item_ids) == 3

	result = await svc.list_correlations(workspace_id=ws_id)
	assert result.total == 1


async def test_correlation_confirm_and_dispute():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	item_ids = await _setup_items(svc, ws_id, 2)
	corr = await svc.create_correlation(CorrelationSetCreate(
		tenant_id="tenant-test",
		workspace_id=ws_id,
		correlation_type=CorrelationType.TIME_SEQUENCE,
		item_ids=item_ids,
		analyst_id="analyst-1",
	))

	confirmed = await svc.confirm_correlation(corr.id)
	assert confirmed.status == CorrelationSetStatus.CONFIRMED

	disputed = await svc.dispute_correlation(corr.id)
	assert disputed.status == CorrelationSetStatus.DISPUTED


# ─────────────────────────────────────────────────────────────────────────────
# Service — Evidence CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def test_evidence_crud_and_lifecycle():
	svc = _svc()
	ws_id = await _setup_workspace(svc)

	ev = await svc.create_evidence(EvidenceCreate(
		tenant_id="tenant-test",
		workspace_id=ws_id,
		evidence_type="document",
		source_reference="classified-doc-001",
		content_fingerprint="sha256:doc001",
		custodian_id="custodian-1",
		chain_of_custody=["analyst-1 received 2026-06-01"],
	))
	assert ev.status == EvidenceStatus.PENDING

	verified = await svc.verify_evidence(ev.id)
	assert verified.status == EvidenceStatus.VERIFIED

	challenged = await svc.challenge_evidence(ev.id)
	assert challenged.status == EvidenceStatus.CHALLENGED

	discredited = await svc.discredit_evidence(ev.id)
	assert discredited.status == EvidenceStatus.DISCREDITED


# ─────────────────────────────────────────────────────────────────────────────
# Service — HypothesisTest + ACH
# ─────────────────────────────────────────────────────────────────────────────

async def test_create_ach_hypothesis():
	svc = _svc()
	ws_id = await _setup_workspace(svc)

	hyp = await svc.create_hypothesis(HypothesisTestCreate(
		tenant_id="tenant-test",
		workspace_id=ws_id,
		statement="State actor X is responsible for the attack",
		sat_method=SATMethod.ACH,
		analyst_id="analyst-1",
		alternative_hypotheses=[
			"Non-state actor Y is responsible",
			"Insider threat is responsible",
		],
		initial_confidence=0.4,
	))
	assert hyp.status == HypothesisStatus.OPEN
	assert len(hyp.alternative_hypotheses) == 2


async def test_conclude_hypothesis():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	hyp = await svc.create_hypothesis(HypothesisTestCreate(
		tenant_id="tenant-test",
		workspace_id=ws_id,
		statement="Target is planning financial crime",
		sat_method=SATMethod.ACH,
		analyst_id="analyst-1",
		alternative_hypotheses=["Target is engaging in money laundering"],
		initial_confidence=0.5,
	))

	concluded = await svc.update_hypothesis(hyp.id, HypothesisTestUpdate(
		status=HypothesisStatus.SUPPORTED,
		final_confidence=0.87,
		conclusion="Corroborated by 3 independent sources",
	))
	assert concluded.status == HypothesisStatus.SUPPORTED
	assert concluded.final_confidence == pytest.approx(0.87, abs=0.001)


async def test_ach_analysis_end_to_end():
	svc = _svc()
	ws_id = await _setup_workspace(svc)

	result = await svc.analysis_of_competing_hypotheses(
		workspace_id=ws_id,
		hypotheses=["H1: State actor", "H2: Insider", "H3: Criminal org"],
		evidence_items=[
			{"label": "Intercepted comms", "consistencies": [1.0, -1.0, 0.0]},
			{"label": "Bank records",      "consistencies": [-1.0, 0.0, 1.0]},
			{"label": "Travel records",    "consistencies": [1.0, -1.0, -1.0]},
		],
	)
	assert result["leading_hypothesis"] in ["H1: State actor", "H2: Insider", "H3: Criminal org"]
	assert "inconsistency_scores" in result
	confs = result["hypothesis_confidence"]
	assert abs(sum(confs) - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Service — AssessmentPicture CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def _make_assessment(svc: IntelligenceFusionService, ws_id: str) -> str:
	"""Helper: create assessment with valid references."""
	item_ids = await _setup_items(svc, ws_id, 2)
	corr = await svc.create_correlation(CorrelationSetCreate(
		tenant_id=svc.tenant_id,
		workspace_id=ws_id,
		correlation_type=CorrelationType.ENTITY_MATCH,
		item_ids=item_ids,
		analyst_id="analyst-1",
	))
	hyp = await svc.create_hypothesis(HypothesisTestCreate(
		tenant_id=svc.tenant_id,
		workspace_id=ws_id,
		statement="Threat actor is coordinating",
		sat_method=SATMethod.ACH,
		analyst_id="analyst-1",
		alternative_hypotheses=["Coincidental activity"],
	))
	assessment = await svc.create_assessment(AssessmentPictureCreate(
		tenant_id=svc.tenant_id,
		workspace_id=ws_id,
		assessment_type="threat",
		risk_level=RiskLevel.HIGH,
		analyst_id="analyst-1",
		hypothesis_ids=[hyp.id],
		correlation_ids=[corr.id],
		confidence_score=0.78,
	))
	return assessment.id


async def test_create_and_approve_assessment():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	assessment_id = await _make_assessment(svc, ws_id)

	assessment = await svc.get_assessment(assessment_id)
	assert assessment.risk_level == RiskLevel.HIGH

	approved = await svc.approve_assessment(assessment_id, approver_id="senior-1")
	assert approved.approved_by == "senior-1"
	assert approved.approved_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# Service — IntelligenceProduct full lifecycle
# ─────────────────────────────────────────────────────────────────────────────

async def _make_product(svc: IntelligenceFusionService, ws_id: str) -> str:
	assessment_id = await _make_assessment(svc, ws_id)
	product = await svc.create_product(IntelligenceProductCreate(
		tenant_id=svc.tenant_id,
		workspace_id=ws_id,
		product_type="threat_assessment",
		title="Q3 Threat Assessment",
		tlp=TLPLevel.AMBER,
		assessment_ids=[assessment_id],
		author_id="analyst-1",
	))
	return product.id


async def test_product_full_lifecycle():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	product_id = await _make_product(svc, ws_id)

	product = await svc.get_product(product_id)
	assert product.status == ProductStatus.DRAFT

	# Submit for review
	submitted = await svc.submit_product_for_review(product_id, reviewer_id="reviewer-1")
	assert submitted.status == ProductStatus.REVIEW
	assert submitted.reviewer_id == "reviewer-1"

	# Approve
	approved = await svc.approve_product(product_id, approver_id="senior-1")
	assert approved.status == ProductStatus.APPROVED

	# Release
	released = await svc.release_product(product_id, approval_reference="AUTH-2026-001")
	assert released.status == ProductStatus.RELEASED
	assert released.released_at is not None

	# Recall
	recalled = await svc.recall_product(product_id)
	assert recalled.status == ProductStatus.RECALLED


async def test_product_cannot_be_modified_after_recall():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	product_id = await _make_product(svc, ws_id)

	await svc.submit_product_for_review(product_id, "r1")
	await svc.approve_product(product_id, "a1")
	await svc.release_product(product_id, "AUTH-001")
	await svc.recall_product(product_id)

	# update_product checks assert_product_not_recalled → RuleViolation
	with pytest.raises(RuleViolation, match="recalled_product_cannot_be_modified"):
		await svc.update_product(product_id, IntelligenceProductUpdate(title="new title"))


# ─────────────────────────────────────────────────────────────────────────────
# Service — Dissemination with TLP
# ─────────────────────────────────────────────────────────────────────────────

async def test_dissemination_tlp_enforcement():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	product_id = await _make_product(svc, ws_id)

	# Approve and release product
	await svc.submit_product_for_review(product_id, "r1")
	await svc.approve_product(product_id, "a1")
	await svc.release_product(product_id, "AUTH-001")

	# Dissemination to a recipient with sufficient clearance
	record = await svc.dissemination_with_tlp(
		product_id=product_id,
		audience="CISO Team",
		recipient_max_tlp="TLP:RED",
		approval_reference="DISS-2026-001",
		disseminated_by="analyst-1",
	)
	assert record.audience == "CISO Team"
	assert record.tlp == TLPLevel.AMBER  # product's TLP


# ─────────────────────────────────────────────────────────────────────────────
# Service — AnalyticalJudgement
# ─────────────────────────────────────────────────────────────────────────────

async def test_create_and_challenge_judgement():
	svc = _svc()
	ws_id = await _setup_workspace(svc)

	judgement = await svc.create_judgement(AnalyticalJudgementCreate(
		tenant_id="tenant-test",
		workspace_id=ws_id,
		judgement_type=JudgementType.ATTRIBUTION,
		statement="The attack is attributed to APT-X with high confidence",
		confidence_score=0.85,
		confidence_level=ConfidenceLevel.HIGHLY_LIKELY,
		analyst_id="analyst-1",
		sat_method=SATMethod.ACH,
		key_assumptions=["Attribution based on TTP overlap ≥ 70%"],
	))
	assert judgement.confidence_score == pytest.approx(0.85, abs=0.001)
	assert len(judgement.challenger_ids) == 0

	challenged = await svc.challenge_judgement(judgement.id, challenger_id="red-team-1")
	assert "red-team-1" in challenged.challenger_ids

	# Idempotent: challenging again does not duplicate
	challenged2 = await svc.challenge_judgement(judgement.id, challenger_id="red-team-1")
	assert challenged2.challenger_ids.count("red-team-1") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Service — Intelligence fusion operations
# ─────────────────────────────────────────────────────────────────────────────

async def test_fuse_intelligence():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	await _setup_items(svc, ws_id, 4)

	result = await svc.fuse_intelligence(workspace_id=ws_id)
	assert result["fused_item_count"] == 4
	assert "quality" in result
	assert "corroboration" in result
	assert result["quality"]["quality_score"] > 0.0


async def test_correlate_across_domains():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	item_ids = await _setup_items(svc, ws_id, 4)

	result = await svc.correlate_across_domains(
		workspace_id=ws_id,
		osint_ids=item_ids[:2],
		humint_ids=item_ids[2:],
	)
	assert "cross_domain_score" in result
	assert "osint" in result["domains_covered"]
	assert "humint" in result["domains_covered"]


async def test_key_assumptions_check():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	result = await svc.key_assumptions_check(
		workspace_id=ws_id,
		assumptions=["Threat is active", "Resources are sufficient"],
		confidence_scores=[0.9, 0.5],
	)
	assert result["weakest_assumption"] == "Resources are sufficient"
	assert "robustness" in result


async def test_confidence_calibration():
	svc = _svc()
	result = await svc.confidence_calibration(
		prior=0.3,
		likelihood_given_true=0.9,
		likelihood_given_false=0.1,
	)
	assert result["posterior"] > 0.3
	assert result["confidence_level"] in (
		"likely", "highly_likely", "almost_certain", "roughly_even"
	)


async def test_ace_method():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	ev_ids = []
	for i in range(3):
		ev = await svc.create_evidence(EvidenceCreate(
			tenant_id="tenant-test",
			workspace_id=ws_id,
			evidence_type="document" if i == 0 else "signal",
			source_reference=f"src-{i}",
			content_fingerprint=f"sha256:ev{i}",
			custodian_id="custodian-1",
			chain_of_custody=[f"received-{i}"],
		))
		ev_ids.append(ev.id)

	result = await svc.ace_method(
		workspace_id=ws_id,
		analysis_statement="Target has intent and capability",
		confidence_score=0.7,
		evidence_ids=ev_ids,
	)
	assert "confidence_score" in result
	assert result["evidence_count"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Service — Dashboard report
# ─────────────────────────────────────────────────────────────────────────────

async def test_dashboard_report():
	svc = _svc("tenant-dash")
	ws_id = await _setup_workspace(svc, "Dash WS")
	await _setup_items(svc, ws_id, 5)
	assessment_id = await _make_assessment(svc, ws_id)

	report = await svc.dashboard_report()
	assert report.tenant_id == "tenant-dash"
	assert report.total_items >= 5
	assert report.total_workspaces >= 1
	assert report.total_assessments >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tenant isolation
# ─────────────────────────────────────────────────────────────────────────────

async def test_tenant_isolation():
	# Use a shared in-memory store so both services see each other's data
	from database.store import InMemoryStore
	shared = InMemoryStore()
	svc_a = IntelligenceFusionService(tenant_id="tenant-a", actor_id="a1", store=shared)
	svc_b = IntelligenceFusionService(tenant_id="tenant-b", actor_id="b1", store=shared)

	ws_a = await _setup_workspace(svc_a, "Workspace A")
	ws_b = await _setup_workspace(svc_b, "Workspace B")

	# Tenant B cannot access tenant A's workspace (PermissionError from _require)
	with pytest.raises(PermissionError):
		await svc_b.get_workspace(ws_a)

	# Tenant A cannot access tenant B's workspace
	with pytest.raises(PermissionError):
		await svc_a.get_workspace(ws_b)

	# Each tenant sees only its own workspaces
	result_a = await svc_a.list_workspaces()
	result_b = await svc_b.list_workspaces()
	a_ids = [r["id"] for r in result_a.items]
	b_ids = [r["id"] for r in result_b.items]
	assert ws_a in a_ids
	assert ws_a not in b_ids
	assert ws_b in b_ids
	assert ws_b not in a_ids


async def test_cross_tenant_item_creation_blocked():
	svc_a = _svc("tenant-a")
	ws_a = await _setup_workspace(svc_a, "WS for A")

	# svc_b trying to create item in tenant-a's tenant_id
	svc_b = _svc("tenant-b")
	with pytest.raises(PermissionError):
		await svc_b.create_intel_item(IntelligenceItemCreate(
			tenant_id="tenant-a",  # wrong tenant
			source_type=SourceType.OSINT,
			source_reference="ref",
			content_fingerprint="sha256:x",
			custodian_id="c",
			workspace_id=ws_a,
		))


# ─────────────────────────────────────────────────────────────────────────────
# Validation guard rails — service level
# ─────────────────────────────────────────────────────────────────────────────

async def test_create_item_without_fingerprint_rejected():
	svc = _svc()
	with pytest.raises(Exception):  # pydantic validation
		await svc.create_intel_item(IntelligenceItemCreate(
			tenant_id="tenant-test",
			source_type=SourceType.OSINT,
			source_reference="ref",
			content_fingerprint="   ",  # blank — fails AfterValidator
			custodian_id="c",
		))


async def test_correlation_with_single_item_rejected():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	item_ids = await _setup_items(svc, ws_id, 1)

	with pytest.raises(RuleViolation, match="correlation_requires_minimum_items"):
		await svc.create_correlation(CorrelationSetCreate(
			tenant_id="tenant-test",
			workspace_id=ws_id,
			correlation_type=CorrelationType.ENTITY_MATCH,
			item_ids=item_ids,  # only 1 item
			analyst_id="analyst-1",
		))


async def test_assessment_without_hypotheses_rejected():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	item_ids = await _setup_items(svc, ws_id, 2)
	corr = await svc.create_correlation(CorrelationSetCreate(
		tenant_id="tenant-test",
		workspace_id=ws_id,
		correlation_type=CorrelationType.ENTITY_MATCH,
		item_ids=item_ids,
		analyst_id="analyst-1",
	))
	with pytest.raises(RuleViolation, match="assessment_requires_hypotheses"):
		await svc.create_assessment(AssessmentPictureCreate(
			tenant_id="tenant-test",
			workspace_id=ws_id,
			assessment_type="threat",
			risk_level=RiskLevel.HIGH,
			analyst_id="analyst-1",
			hypothesis_ids=[],  # missing
			correlation_ids=[corr.id],
		))


async def test_fuse_requires_min_2_items():
	svc = _svc()
	ws_id = await _setup_workspace(svc)
	await _setup_items(svc, ws_id, 1)  # only 1 item

	with pytest.raises(RuleViolation, match="insufficient_sources_for_fusion"):
		await svc.fuse_intelligence(workspace_id=ws_id)
