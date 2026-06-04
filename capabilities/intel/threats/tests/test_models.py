"""Async model tests for APG Threat Intelligence.

Tests all Pydantic v2 models: enums, base model, create/update/response
variants, aggregation models, and edge cases.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

import pytest

# Ensure the package is importable
_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
	sys.path.insert(0, str(_PKG))

from models import (
	ThreatActorType, ThreatActorStatus, IndicatorType, IndicatorStatus,
	CampaignType, CampaignStatus, RiskLevel, Classification, ReportType,
	ReportStatus, KillChainPhaseType, MitreTactic, FeedType, FeedStatus,
	AssessmentType, RequirementStatus, EvidenceType,
	TIBase, ThreatActorCreate, ThreatActorUpdate, ThreatActorResponse,
	ThreatIndicatorCreate, ThreatIndicatorUpdate, ThreatIndicatorResponse,
	ThreatCampaignCreate, ThreatCampaignUpdate, ThreatCampaignResponse,
	ThreatReportCreate, ThreatReportUpdate, ThreatReportResponse,
	MITRETechniqueCreate, MITRETechniqueResponse,
	KillChainPhaseCreate, KillChainPhaseResponse,
	ThreatFeedCreate, ThreatFeedUpdate, ThreatFeedResponse,
	IntelRequirementCreate, IntelRequirementUpdate, IntelRequirementResponse,
	ThreatAssessmentCreate, ThreatAssessmentUpdate, ThreatAssessmentResponse,
	AttributionEvidenceCreate, AttributionEvidenceUpdate, AttributionEvidenceResponse,
	ThreatDashboardReport, CorrelationResult, MitreHeatmap,
	ConfidenceScoreBreakdown, STIXBundle, TAXIIShareRequest,
	MISPExportRequest, StalenessSweepResult, FeedIngestResult,
	ThreatReportGenerationRequest, uuid7str,
)


# ── Enum sanity ───────────────────────────────────────────────────────────────

def test_threat_actor_type_values():
	assert ThreatActorType.state_actor == "state_actor"
	assert ThreatActorType.criminal_group == "criminal_group"
	assert ThreatActorType.insider == "insider"
	assert len(ThreatActorType) == 7


def test_indicator_type_values():
	assert IndicatorType.ip_address == "ip_address"
	assert IndicatorType.yara_rule == "yara_rule"
	assert len(IndicatorType) >= 15


def test_mitre_tactic_values():
	assert MitreTactic.initial_access == "TA0001"
	assert MitreTactic.exfiltration == "TA0010"
	assert MitreTactic.impact == "TA0040"


def test_kill_chain_phase_type():
	assert KillChainPhaseType.reconnaissance == "reconnaissance"
	assert KillChainPhaseType.actions_on_objectives == "actions_on_objectives"


def test_risk_level_ordering():
	levels = [RiskLevel.low, RiskLevel.medium, RiskLevel.high, RiskLevel.critical]
	values = [l.value for l in levels]
	assert values == ["low", "medium", "high", "critical"]


def test_classification_values():
	assert Classification.unclassified == "unclassified"
	assert Classification.top_secret == "top_secret"


# ── UUID7 generation ──────────────────────────────────────────────────────────

def test_uuid7str_unique():
	ids = {uuid7str() for _ in range(100)}
	assert len(ids) == 100


def test_uuid7str_format():
	uid = uuid7str()
	parts = uid.split("-")
	assert len(parts) == 5
	assert len(uid) == 36


# ── TIBase ────────────────────────────────────────────────────────────────────

def test_tibase_defaults():
	base = TIBase(tenant_id="t1", created_by="analyst1")
	assert len(base.id) == 36
	assert base.is_deleted is False
	assert base.tenant_id == "t1"
	assert base.created_by == "analyst1"
	assert isinstance(base.created_at, datetime)


def test_tibase_rejects_extra_fields():
	import pydantic
	with pytest.raises(pydantic.ValidationError):
		TIBase(tenant_id="t1", created_by="a1", unknown_field="x")


# ── ThreatActor ───────────────────────────────────────────────────────────────

def test_threat_actor_create_valid():
	actor = ThreatActorCreate(
		tenant_id="t1",
		created_by="analyst1",
		name="APT29",
		actor_type=ThreatActorType.state_actor,
		confidence_score=0.85,
		evidence_reference="ev-001",
	)
	assert actor.name == "APT29"
	assert actor.confidence_score == 0.85
	assert actor.aliases == []


def test_threat_actor_create_confidence_bounds():
	import pydantic
	with pytest.raises(pydantic.ValidationError):
		ThreatActorCreate(
			tenant_id="t1", created_by="a", name="X",
			actor_type=ThreatActorType.unknown,
			confidence_score=1.5, evidence_reference="ev",
		)
	with pytest.raises(pydantic.ValidationError):
		ThreatActorCreate(
			tenant_id="t1", created_by="a", name="X",
			actor_type=ThreatActorType.unknown,
			confidence_score=-0.1, evidence_reference="ev",
		)


def test_threat_actor_update_partial():
	upd = ThreatActorUpdate(name="APT29-v2")
	assert upd.name == "APT29-v2"
	assert upd.status is None
	assert upd.confidence_score is None


def test_threat_actor_response_has_defaults():
	resp = ThreatActorResponse(
		tenant_id="t1", created_by="a",
		name="APT29",
		actor_type=ThreatActorType.state_actor,
		confidence_score=0.8,
		evidence_reference="ev",
	)
	assert resp.status == ThreatActorStatus.suspected
	assert resp.indicator_ids == []
	assert resp.campaign_ids == []


# ── ThreatIndicator ───────────────────────────────────────────────────────────

def test_threat_indicator_create_valid():
	ioc = ThreatIndicatorCreate(
		tenant_id="t1",
		created_by="analyst1",
		indicator_type=IndicatorType.ip_address,
		value="192.168.1.1",
		source_id="src-1",
		confidence_score=0.9,
		evidence_reference="ev-002",
	)
	assert ioc.value == "192.168.1.1"
	assert ioc.tlp == "green"
	assert ioc.tags == []


def test_threat_indicator_tlp_default():
	ioc = ThreatIndicatorCreate(
		tenant_id="t1", created_by="a",
		indicator_type=IndicatorType.domain,
		value="evil.com",
		source_id="src",
		confidence_score=0.7,
		evidence_reference="ev",
	)
	assert ioc.tlp == "green"


def test_threat_indicator_response_staleness_default():
	resp = ThreatIndicatorResponse(
		tenant_id="t1", created_by="a",
		indicator_type=IndicatorType.domain,
		value="evil.com",
		source_id="src",
		confidence_score=0.7,
		valid_from=datetime.now(timezone.utc),
		evidence_reference="ev",
	)
	assert resp.staleness_score == 0.0
	assert resp.status == IndicatorStatus.active


# ── ThreatCampaign ────────────────────────────────────────────────────────────

def test_threat_campaign_create_valid():
	camp = ThreatCampaignCreate(
		tenant_id="t1",
		created_by="analyst",
		name="Operation Ember Bear",
		campaign_type=CampaignType.intrusion_campaign,
		actor_id="actor-1",
		risk_level=RiskLevel.critical,
		classification=Classification.confidential,
		evidence_reference="ev-003",
	)
	assert camp.name == "Operation Ember Bear"
	assert camp.risk_level == RiskLevel.critical
	assert camp.target_sectors == []


def test_threat_campaign_response_defaults():
	resp = ThreatCampaignResponse(
		tenant_id="t1", created_by="a",
		name="Op X",
		campaign_type=CampaignType.ransomware_campaign,
		actor_id="a1",
		risk_level=RiskLevel.high,
		classification=Classification.confidential,
		evidence_reference="ev",
	)
	assert resp.status == CampaignStatus.suspected
	assert resp.indicator_ids == []


# ── ThreatReport ──────────────────────────────────────────────────────────────

def test_threat_report_create_valid():
	rpt = ThreatReportCreate(
		tenant_id="t1",
		created_by="analyst",
		title="Flash Report: APT29",
		report_type=ReportType.flash_report,
		classification=Classification.confidential,
		summary="APT29 active targeting of financial sector",
		assessment_id="assess-1",
		author_id="analyst-1",
		tlp="amber",
		approval_reference="appr-1",
		evidence_reference="ev-004",
	)
	assert rpt.title == "Flash Report: APT29"
	assert rpt.tlp == "amber"
	assert rpt.analyst_ids == []


def test_threat_report_response_status():
	resp = ThreatReportResponse(
		tenant_id="t1", created_by="a",
		title="T",
		report_type=ReportType.advisory,
		classification=Classification.unclassified,
		summary="S",
		assessment_id="a1",
		author_id="au1",
		approval_reference="appr",
		evidence_reference="ev",
	)
	assert resp.status == ReportStatus.draft
	assert resp.published_at is None


# ── MITRE Technique ───────────────────────────────────────────────────────────

def test_mitre_technique_create():
	tech = MITRETechniqueCreate(
		tenant_id="t1",
		created_by="a",
		technique_id="T1059.001",
		name="PowerShell",
		tactic=MitreTactic.execution,
		platforms=["Windows"],
	)
	assert tech.technique_id == "T1059.001"
	assert tech.tactic == MitreTactic.execution


# ── ThreatFeed ────────────────────────────────────────────────────────────────

def test_threat_feed_create_valid():
	feed = ThreatFeedCreate(
		tenant_id="t1",
		created_by="a",
		name="AlienVault OTX",
		feed_type=FeedType.json_api,
		url="https://otx.alienvault.com/api/v1/indicators",
		poll_interval_seconds=3600,
		confidence_weight=0.75,
		custodian_id="custodian-1",
		evidence_reference="ev-005",
	)
	assert feed.name == "AlienVault OTX"
	assert feed.confidence_weight == 0.75


def test_threat_feed_confidence_weight_bounds():
	import pydantic
	with pytest.raises(pydantic.ValidationError):
		ThreatFeedCreate(
			tenant_id="t1", created_by="a",
			name="Bad Feed",
			feed_type=FeedType.csv,
			poll_interval_seconds=3600,
			confidence_weight=1.5,
			custodian_id="c1",
			evidence_reference="ev",
		)


# ── ThreatAssessment ──────────────────────────────────────────────────────────

def test_threat_assessment_create():
	assess = ThreatAssessmentCreate(
		tenant_id="t1",
		created_by="a",
		assessment_type=AssessmentType.risk_assessment,
		campaign_id="camp-1",
		analyst_id="analyst-1",
		risk_level=RiskLevel.high,
		confidence_score=0.82,
		summary="High risk assessed",
		evidence_reference="ev-006",
	)
	assert assess.confidence_score == 0.82
	assert assess.findings == []


# ── AttributionEvidence ───────────────────────────────────────────────────────

def test_attribution_evidence_create():
	ev = AttributionEvidenceCreate(
		tenant_id="t1",
		created_by="a",
		actor_id="actor-1",
		evidence_type=EvidenceType.ttps_match,
		description="TTP overlap with Cozy Bear",
		confidence_score=0.88,
		analyst_id="analyst-1",
	)
	assert ev.evidence_type == EvidenceType.ttps_match
	assert ev.indicator_ids == []


# ── Aggregation models ────────────────────────────────────────────────────────

def test_threat_dashboard_report_defaults():
	rpt = ThreatDashboardReport(tenant_id="t1")
	assert rpt.actor_count == 0
	assert rpt.indicator_count == 0
	assert rpt.critical_actors == []


def test_correlation_result_defaults():
	cr = CorrelationResult(tenant_id="t1")
	assert cr.confidence_weighted_score == 0.0
	assert cr.correlation_count == 0


def test_staleness_sweep_result():
	result = StalenessSweepResult(
		tenant_id="t1",
		reviewed=100,
		marked_stale=15,
		revoked=5,
		still_active=80,
	)
	assert result.reviewed == 100
	assert result.stale_ids == []


def test_stix_bundle_extra_allowed():
	bundle = STIXBundle(
		id="bundle--abc",
		objects=[{"type": "indicator", "id": "indicator--123"}],
		extra_custom_field="allowed",
	)
	assert bundle.type == "bundle"
	assert len(bundle.objects) == 1


def test_taxii_share_request():
	req = TAXIIShareRequest(
		tenant_id="t1",
		taxii_server_url="https://taxii.example.com",
		collection_id="col-1",
		api_root="api21",
	)
	assert req.tlp_max == "green"
	assert req.indicator_ids == []


def test_misp_export_request_defaults():
	req = MISPExportRequest(tenant_id="t1")
	assert req.include_attributes is True
	assert req.distribution == 0


def test_confidence_score_breakdown():
	cb = ConfidenceScoreBreakdown(
		entity_id="e1",
		entity_type="indicator",
		raw_score=0.8,
		source_reliability=0.9,
		recency_decay=0.95,
		corroboration_bonus=0.1,
		final_score=0.85,
	)
	assert cb.final_score == 0.85


def test_feed_ingest_result():
	result = FeedIngestResult(
		feed_id="f1",
		tenant_id="t1",
		total_objects=100,
		indicators_created=80,
	)
	assert result.errors == []
	assert result.skipped_stale == 0


def test_threat_report_generation_request():
	req = ThreatReportGenerationRequest(
		tenant_id="t1",
		created_by="analyst",
		assessment_id="assess-1",
		report_type=ReportType.advisory,
		classification=Classification.confidential,
		title="Q1 Threat Advisory",
	)
	assert req.include_indicators is True
	assert req.tlp == "amber"


def test_mitre_heatmap_defaults():
	hm = MitreHeatmap(tenant_id="t1")
	assert hm.tactic_coverage == {}
	assert hm.total_techniques == 0


# ── IntelRequirement ──────────────────────────────────────────────────────────

def test_intel_requirement_create():
	req = IntelRequirementCreate(
		tenant_id="t1",
		created_by="a",
		title="Attribution of APT29 campaign",
		description="Determine if recent intrusion is linked to APT29",
		requestor_id="manager-1",
		priority=RiskLevel.high,
	)
	assert req.priority == RiskLevel.high
	assert req.related_actor_ids == []


def test_intel_requirement_response_defaults():
	resp = IntelRequirementResponse(
		tenant_id="t1", created_by="a",
		title="T",
		description="D",
		requestor_id="r1",
		priority=RiskLevel.medium,
	)
	assert resp.status == RequirementStatus.open
	assert resp.satisfying_report_ids == []
