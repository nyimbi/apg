"""Tests for OSINTService — full lifecycle coverage.

Run with: uv run pytest -vxs capabilities/intel/osint/tests/
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from repo root or from the package directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
	AgentRole,
	AgentRuntime,
	CollectionTaskCreate,
	CredibilityScoreCreate,
	DisseminationPackageCreate,
	DocumentAnalysisCreate,
	EntityRelationshipCreate,
	IPIntelligenceCreate,
	OSEntityCreate,
	OSINTAgentCreate,
	OSINTReviewCreate,
	OSINTSourceCreate,
	Priority,
	ProcessedIntelligenceCreate,
	RawIntelligenceCreate,
	RelationshipType,
	ReviewStatus,
	RiskTier,
	SocialMediaProfileCreate,
	SourceType,
	TaskType,
	TriageDecision,
	DomainRecordCreate,
	EntityType,
	CollectionMethod,
	ConfidenceLevel,
	ClassificationLevel,
	TLPLevel,
	AssessmentType,
)
from service import OSINTService  # type: ignore


def _svc(tenant: str = "t1") -> OSINTService:
	return OSINTService(db_session=None, tenant_id=tenant, actor_id="analyst-001")


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------

async def test_register_source_success():
	svc = _svc()
	payload = OSINTSourceCreate(
		tenant_id="t1",
		name="Reuters Feed",
		source_type=SourceType.NEWS,
		url="https://reuters.com/rss",
		owner_id="u-001",
		terms_review_reference="tos-review-001",
		risk_tier=RiskTier.LOW,
		collection_method=CollectionMethod.RSS_FEED,
		evidence_reference="ev-001",
	)
	source = await svc.register_source(payload)
	assert source.id
	assert source.name == "Reuters Feed"
	assert source.tenant_id == "t1"
	assert source.risk_tier == RiskTier.LOW
	assert source.total_items_collected == 0


async def test_register_source_missing_terms_review():
	svc = _svc()
	import pytest
	from domain.rules import RuleViolation  # type: ignore
	payload = OSINTSourceCreate(
		tenant_id="t1",
		name="Untrusted Feed",
		source_type=SourceType.FORUM,
		owner_id="u-001",
		terms_review_reference="",  # missing
		risk_tier=RiskTier.MEDIUM,
		collection_method=CollectionMethod.CRAWLER,
		evidence_reference="ev-002",
	)
	try:
		await svc.register_source(payload)
		assert False, "should have raised"
	except RuleViolation as exc:
		assert "terms" in exc.reason


async def test_list_sources():
	svc = _svc()
	for i in range(3):
		await svc.register_source(OSINTSourceCreate(
			tenant_id="t1",
			name=f"Source {i}",
			source_type=SourceType.WEB,
			owner_id="u-001",
			terms_review_reference=f"tos-{i}",
			risk_tier=RiskTier.LOW,
			collection_method=CollectionMethod.CRAWLER,
			evidence_reference=f"ev-{i}",
		))
	sources = await svc.list_sources()
	assert len(sources) == 3


async def test_source_tenant_isolation():
	svc_a = _svc("tenant_a")
	svc_b = _svc("tenant_b")
	await svc_a.register_source(OSINTSourceCreate(
		tenant_id="tenant_a",
		name="Source A",
		source_type=SourceType.WEB,
		owner_id="u-001",
		terms_review_reference="tos-a",
		risk_tier=RiskTier.LOW,
		collection_method=CollectionMethod.CRAWLER,
		evidence_reference="ev-a",
	))
	sources_b = await svc_b.list_sources()
	assert len(sources_b) == 0


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------

async def _create_source_and_task(svc: OSINTService) -> tuple:
	source = await svc.register_source(OSINTSourceCreate(
		tenant_id=svc._tenant_id,
		name="Test Source",
		source_type=SourceType.WEB,
		owner_id="u-001",
		terms_review_reference="tos-001",
		risk_tier=RiskTier.LOW,
		collection_method=CollectionMethod.CRAWLER,
		evidence_reference="ev-src",
	))
	task = await svc.create_task(CollectionTaskCreate(
		tenant_id=svc._tenant_id,
		source_id=source.id,
		task_type=TaskType.WEB_SCRAPE,
		evidence_reference="ev-task",
	))
	return source, task


async def test_create_task():
	svc = _svc()
	source, task = await _create_source_and_task(svc)
	assert task.source_id == source.id
	assert task.status.value == "pending"


async def test_task_lifecycle():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	started = await svc.start_task(task.id)
	assert started.status.value == "running"
	assert started.started_at is not None
	completed = await svc.complete_task(task.id, items_collected=42)
	assert completed.status.value == "completed"
	assert completed.items_collected == 42
	assert completed.completed_at is not None


async def test_task_fail():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	await svc.start_task(task.id)
	failed = await svc.fail_task(task.id, "connection refused")
	assert failed.status.value == "failed"
	assert "connection" in failed.error_message


async def test_high_risk_task_requires_approval():
	svc = _svc()
	source = await svc.register_source(OSINTSourceCreate(
		tenant_id="t1",
		name="Dark Feed",
		source_type=SourceType.DARKWEB,
		owner_id="u-001",
		terms_review_reference="tos-dark",
		risk_tier=RiskTier.HIGH,
		collection_method=CollectionMethod.HEADLESS_BROWSER,
		evidence_reference="ev-dark",
	))
	try:
		await svc.create_task(CollectionTaskCreate(
			tenant_id="t1",
			source_id=source.id,
			task_type=TaskType.DARK_WEB_CRAWL,
			approval_reference=None,  # missing!
			evidence_reference="ev-task",
		))
		assert False, "should have raised"
	except Exception as exc:
		assert "approval" in str(exc).lower() or "high_risk" in str(exc).lower()


# ---------------------------------------------------------------------------
# Raw intelligence tests
# ---------------------------------------------------------------------------

async def test_ingest_raw_intel():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	raw = await svc.ingest_raw_intel(RawIntelligenceCreate(
		tenant_id="t1",
		task_id=task.id,
		source_id=task.source_id,
		content_reference="s3://bucket/file.html",
		content_type="text/html",
		fingerprint="abc123deadbeef",
		confidence_score=0.85,
		evidence_reference="ev-raw",
	))
	assert raw.id
	assert raw.fingerprint == "abc123deadbeef"
	assert raw.status.value == "raw"


async def test_duplicate_fingerprint_rejected():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	base = RawIntelligenceCreate(
		tenant_id="t1",
		task_id=task.id,
		source_id=task.source_id,
		content_reference="s3://bucket/dup.html",
		content_type="text/html",
		fingerprint="dup-fingerprint-xyz",
		confidence_score=0.7,
		evidence_reference="ev-dup",
	)
	await svc.ingest_raw_intel(base)
	try:
		await svc.ingest_raw_intel(base)  # duplicate
		assert False, "should have raised"
	except Exception as exc:
		assert "duplicate" in str(exc).lower() or "fingerprint" in str(exc).lower()


async def test_triage_raw_intel():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	raw = await svc.ingest_raw_intel(RawIntelligenceCreate(
		tenant_id="t1",
		task_id=task.id,
		source_id=task.source_id,
		content_reference="s3://bucket/art.html",
		content_type="text/html",
		fingerprint="triage-test-fp",
		confidence_score=0.8,
		evidence_reference="ev-triage",
	))
	triaged = await svc.triage_raw_intel(raw.id, TriageDecision.RELEVANT, "analyst-001", notes="Key finding")
	assert triaged.status.value == "triaged"
	assert triaged.triage_decision == TriageDecision.RELEVANT
	assert triaged.notes == "Key finding"


# ---------------------------------------------------------------------------
# Processed intelligence tests
# ---------------------------------------------------------------------------

async def test_create_processed_intel():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	raw = await svc.ingest_raw_intel(RawIntelligenceCreate(
		tenant_id="t1",
		task_id=task.id,
		source_id=task.source_id,
		content_reference="s3://bucket/proc.html",
		content_type="text/html",
		fingerprint="proc-fp-001",
		confidence_score=0.9,
		evidence_reference="ev-proc",
	))
	intel = await svc.create_processed_intel(ProcessedIntelligenceCreate(
		tenant_id="t1",
		raw_intel_id=raw.id,
		assessment_type=AssessmentType.THREAT,
		summary="Threat actor X targeting infrastructure",
		key_findings=["C2 server identified", "Malware deployed"],
		confidence_score=0.85,
		confidence_level=ConfidenceLevel.PROBABLE,
		classification=ClassificationLevel.UNCLASSIFIED,
		tlp=TLPLevel.AMBER,
		analyst_id="analyst-001",
		evidence_reference="ev-intel",
	))
	assert intel.assessment_type == AssessmentType.THREAT
	assert len(intel.key_findings) == 2
	assert intel.analyst_id == "analyst-001"
	# Raw intel should be marked processed
	updated_raw = await svc.get_raw_intel(raw.id)
	assert updated_raw.status.value == "processed"


# ---------------------------------------------------------------------------
# Entity and relationship tests
# ---------------------------------------------------------------------------

async def test_extract_entity():
	svc = _svc()
	entity = await svc.extract_entity(OSEntityCreate(
		tenant_id="t1",
		entity_type=EntityType.PERSON,
		name="John Doe",
		aliases=["J. Doe", "JohnD"],
		confidence_score=0.75,
		evidence_reference="ev-entity",
	))
	assert entity.name == "John Doe"
	assert len(entity.aliases) == 2
	assert entity.entity_type == EntityType.PERSON


async def test_entity_name_required():
	svc = _svc()
	from domain.rules import RuleViolation  # type: ignore
	try:
		await svc.extract_entity(OSEntityCreate(
			tenant_id="t1",
			entity_type=EntityType.ORGANIZATION,
			name="   ",  # whitespace only
			confidence_score=0.5,
			evidence_reference="ev-e",
		))
		assert False, "should have raised"
	except RuleViolation as exc:
		assert "name" in exc.reason


async def test_map_relationship():
	svc = _svc()
	e1 = await svc.extract_entity(OSEntityCreate(
		tenant_id="t1",
		entity_type=EntityType.PERSON,
		name="Alice",
		confidence_score=0.8,
		evidence_reference="ev-alice",
	))
	e2 = await svc.extract_entity(OSEntityCreate(
		tenant_id="t1",
		entity_type=EntityType.ORGANIZATION,
		name="Acme Corp",
		confidence_score=0.9,
		evidence_reference="ev-acme",
	))
	rel = await svc.map_relationship(EntityRelationshipCreate(
		tenant_id="t1",
		source_entity_id=e1.id,
		target_entity_id=e2.id,
		relationship_type=RelationshipType.EMPLOYS,
		strength=0.85,
		confidence_score=0.80,
		evidence_reference="ev-rel",
	))
	assert rel.source_entity_id == e1.id
	assert rel.target_entity_id == e2.id
	assert rel.relationship_type == RelationshipType.EMPLOYS
	# Backlinks
	updated_e1 = await svc.get_entity(e1.id)
	assert rel.id in updated_e1.relationship_ids


async def test_self_loop_relationship_denied():
	svc = _svc()
	entity = await svc.extract_entity(OSEntityCreate(
		tenant_id="t1",
		entity_type=EntityType.PERSON,
		name="Self",
		confidence_score=0.5,
		evidence_reference="ev-self",
	))
	from domain.rules import RuleViolation  # type: ignore
	try:
		await svc.map_relationship(EntityRelationshipCreate(
			tenant_id="t1",
			source_entity_id=entity.id,
			target_entity_id=entity.id,  # self-loop!
			relationship_type=RelationshipType.LINKED_TO,
			confidence_score=0.5,
			evidence_reference="ev-loop",
		))
		assert False, "should have raised"
	except RuleViolation as exc:
		assert "self_loop" in exc.rule_name


# ---------------------------------------------------------------------------
# Web scrape tests
# ---------------------------------------------------------------------------

async def test_web_scrape():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	content = await svc.web_scrape(
		url="https://example.com/article",
		task_id=task.id,
		depth=1,
		content="<html><body>Intel article</body></html>",
		title="Intel Article",
	)
	assert content.url == "https://example.com/article"
	assert content.title == "Intel Article"
	assert content.content_hash  # auto-computed


# ---------------------------------------------------------------------------
# Domain intelligence tests
# ---------------------------------------------------------------------------

async def test_domain_intelligence():
	svc = _svc()
	record = await svc.domain_intelligence(DomainRecordCreate(
		tenant_id="t1",
		domain="example.com",
		registrar="GoDaddy",
		registrant_email="admin@example.com",
		a_records=["1.2.3.4"],
		evidence_reference="ev-domain",
	))
	assert record.domain == "example.com"
	assert record.registrar == "GoDaddy"


# ---------------------------------------------------------------------------
# IP intelligence tests
# ---------------------------------------------------------------------------

async def test_ip_geolocation_enrichment():
	svc = _svc()
	ip = await svc.ip_geolocation_enrichment(IPIntelligenceCreate(
		tenant_id="t1",
		ip_address="185.220.101.1",
		ip_version=4,
		country_code="DE",
		country_name="Germany",
		is_tor=True,
		abuse_confidence_score=0.95,
		evidence_reference="ev-ip",
	))
	assert ip.ip_address == "185.220.101.1"
	assert ip.is_tor is True
	assert ip.country_code == "DE"


# ---------------------------------------------------------------------------
# Document analysis tests
# ---------------------------------------------------------------------------

async def test_entity_extraction_nlp():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	raw = await svc.ingest_raw_intel(RawIntelligenceCreate(
		tenant_id="t1",
		task_id=task.id,
		source_id=task.source_id,
		content_reference="s3://bucket/doc.txt",
		content_type="text/plain",
		fingerprint="nlp-test-fp",
		confidence_score=0.7,
		evidence_reference="ev-doc",
	))
	analysis = await svc.entity_extraction_nlp(DocumentAnalysisCreate(
		tenant_id="t1",
		raw_intel_id=raw.id,
		language="en",
		sentiment_score=0.2,
		keywords=["threat", "malware", "infrastructure"],
		topics=["cybersecurity"],
		summary="Threat actor identified",
		person_mentions=["John Doe"],
		org_mentions=["Evil Corp"],
		model_used="ollama/llama3",
		evidence_reference="ev-nlp",
	))
	assert analysis.language == "en"
	assert len(analysis.keywords) == 3
	assert analysis.model_used == "ollama/llama3"


# ---------------------------------------------------------------------------
# Social media monitor tests
# ---------------------------------------------------------------------------

async def test_social_media_monitor():
	svc = _svc()
	profiles = await svc.social_media_monitor(
		handles=["@user1", "@user2"],
		keywords=["hacker", "exploit"],
		platform="twitter",
	)
	assert len(profiles) == 2
	assert all(p.platform == "twitter" for p in profiles)
	assert "@user1" in [p.handle for p in profiles]


# ---------------------------------------------------------------------------
# Dissemination tests
# ---------------------------------------------------------------------------

async def test_dissemination_package():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	raw = await svc.ingest_raw_intel(RawIntelligenceCreate(
		tenant_id="t1",
		task_id=task.id,
		source_id=task.source_id,
		content_reference="s3://bucket/diss.html",
		content_type="text/html",
		fingerprint="diss-fp-001",
		confidence_score=0.9,
		evidence_reference="ev-diss-raw",
	))
	intel = await svc.create_processed_intel(ProcessedIntelligenceCreate(
		tenant_id="t1",
		raw_intel_id=raw.id,
		assessment_type=AssessmentType.THREAT,
		summary="Critical threat identified",
		confidence_score=0.9,
		analyst_id="analyst-001",
		evidence_reference="ev-diss-intel",
	))
	pkg = await svc.intelligence_dissemination(DisseminationPackageCreate(
		tenant_id="t1",
		processed_intel_ids=[intel.id],
		audience="EXEC_TEAM",
		release_marking=TLPLevel.AMBER,
		classification=ClassificationLevel.UNCLASSIFIED,
		title="Critical Threat Brief",
		executive_summary="Immediate action required.",
		approval_reference="approval-001",
		evidence_reference="ev-pkg",
	))
	assert pkg.title == "Critical Threat Brief"
	assert pkg.disseminated_at is not None
	# Intel should be marked disseminated
	updated_intel = await svc.get_processed_intel(intel.id)
	assert updated_intel.status.value == "disseminated"


async def test_dissemination_requires_approval():
	svc = _svc()
	_, task = await _create_source_and_task(svc)
	raw = await svc.ingest_raw_intel(RawIntelligenceCreate(
		tenant_id="t1",
		task_id=task.id,
		source_id=task.source_id,
		content_reference="s3://bucket/noapprove.html",
		content_type="text/html",
		fingerprint="no-approval-fp",
		confidence_score=0.5,
		evidence_reference="ev-na-raw",
	))
	intel = await svc.create_processed_intel(ProcessedIntelligenceCreate(
		tenant_id="t1",
		raw_intel_id=raw.id,
		assessment_type=AssessmentType.TREND,
		summary="Trend analysis",
		confidence_score=0.6,
		analyst_id="analyst-001",
		evidence_reference="ev-na-intel",
	))
	from domain.rules import RuleViolation  # type: ignore
	try:
		await svc.intelligence_dissemination(DisseminationPackageCreate(
			tenant_id="t1",
			processed_intel_ids=[intel.id],
			audience="ALL",
			release_marking=TLPLevel.GREEN,
			classification=ClassificationLevel.UNCLASSIFIED,
			title="Trend Brief",
			executive_summary="Summary",
			approval_reference="",  # missing!
			evidence_reference="ev-na-pkg",
		))
		assert False, "should have raised"
	except RuleViolation as exc:
		assert "approval" in exc.reason


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

async def test_duplicate_deduplication():
	svc = _svc()
	# Create entities with similar names
	for name, alias in [("John Smith", "J Smith"), ("John Smith Jr", "J. Smith"), ("Jane Doe", "J Doe")]:
		await svc.extract_entity(OSEntityCreate(
			tenant_id="t1",
			entity_type=EntityType.PERSON,
			name=name,
			aliases=[alias],
			confidence_score=0.7,
			evidence_reference="ev-dedup",
		))
	result = await svc.duplicate_deduplication(similarity_threshold=0.5)
	assert result["original_entity_count"] == 3
	assert "merged_entity_count" in result
	assert result["similarity_threshold"] == 0.5


# ---------------------------------------------------------------------------
# Relationship mapping (network report)
# ---------------------------------------------------------------------------

async def test_relationship_mapping_report():
	svc = _svc()
	entities = []
	for name in ["Entity A", "Entity B", "Entity C"]:
		e = await svc.extract_entity(OSEntityCreate(
			tenant_id="t1",
			entity_type=EntityType.ORGANIZATION,
			name=name,
			confidence_score=0.8,
			evidence_reference="ev-net",
		))
		entities.append(e)
	await svc.map_relationship(EntityRelationshipCreate(
		tenant_id="t1",
		source_entity_id=entities[0].id,
		target_entity_id=entities[1].id,
		relationship_type=RelationshipType.AFFILIATED_WITH,
		confidence_score=0.9,
		evidence_reference="ev-rel-net",
	))
	report = await svc.relationship_mapping()
	assert report.entity_count == 3
	assert report.relationship_count == 1
	assert len(report.clusters) >= 1


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

async def test_dashboard_summary():
	svc = _svc()
	await svc.register_source(OSINTSourceCreate(
		tenant_id="t1",
		name="Test Source",
		source_type=SourceType.WEB,
		owner_id="u-001",
		terms_review_reference="tos-dash",
		risk_tier=RiskTier.LOW,
		collection_method=CollectionMethod.CRAWLER,
		evidence_reference="ev-dash",
	))
	dashboard = await svc.dashboard_summary()
	assert dashboard.tenant_id == "t1"
	assert dashboard.source_count == 1
	assert dashboard.active_source_count == 1


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

async def test_register_agent():
	svc = _svc()
	agent = await svc.register_agent(OSINTAgentCreate(
		tenant_id="t1",
		name="Scraper Bot",
		runtime=AgentRuntime.CLAUDE_CODE,
		role=AgentRole.SOURCE_SCOUT,
		scope="web collection",
	))
	assert agent.name == "Scraper Bot"
	assert agent.runtime == AgentRuntime.CLAUDE_CODE


async def test_validate_agent_action_privileged_no_approval():
	svc = _svc()
	try:
		await svc.validate_agent_action(privileged_scope=True, human_approval_recorded=False)
		assert False, "should have raised"
	except Exception as exc:
		assert "approval" in str(exc).lower()


async def test_validate_agent_action_approved():
	svc = _svc()
	result = await svc.validate_agent_action(privileged_scope=True, human_approval_recorded=True)
	assert result["accepted"] is True


# ---------------------------------------------------------------------------
# Review tests
# ---------------------------------------------------------------------------

async def test_record_review():
	svc = _svc()
	review = await svc.record_review(OSINTReviewCreate(
		tenant_id="t1",
		reference_id="some-intel-id",
		reference_type="processed_intel",
		reviewer_id="reviewer-001",
		status=ReviewStatus.APPROVED,
		notes="Verified and accurate",
		evidence_reference="ev-review",
	))
	assert review.status == ReviewStatus.APPROVED
	assert review.reviewer_id == "reviewer-001"


# ---------------------------------------------------------------------------
# Calculations tests
# ---------------------------------------------------------------------------

async def test_credibility_scoring():
	svc = _svc()
	score = await svc.credibility_scoring(CredibilityScoreCreate(
		tenant_id="t1",
		reference_id="source-abc",
		reference_type="source",
		score=0.82,
		factors={"domain_age": 0.8, "error_rate": 0.9, "verifications": 0.7},
		analyst_id="analyst-001",
		rationale="High-credibility primary source",
		evidence_reference="ev-cred",
	))
	assert score.score == 0.82
	assert "domain_age" in score.factors


# ---------------------------------------------------------------------------
# Domain calculations unit tests
# ---------------------------------------------------------------------------

def test_timeliness_fresh():
	from domain.calculations import calculate_timeliness_score  # type: ignore
	from datetime import datetime, timezone
	now = datetime.now(timezone.utc)
	score = calculate_timeliness_score(now)
	assert 0.99 <= score <= 1.0


def test_timeliness_old():
	from domain.calculations import calculate_timeliness_score  # type: ignore
	from datetime import datetime, timezone, timedelta
	old = datetime.now(timezone.utc) - timedelta(days=90)
	score = calculate_timeliness_score(old)
	assert score < 0.05


def test_content_fingerprint():
	from domain.calculations import compute_content_fingerprint  # type: ignore
	fp = compute_content_fingerprint("hello world")
	assert len(fp) == 64  # sha256 hex digest


def test_ip_threat_score_tor():
	from domain.calculations import calculate_ip_threat_score  # type: ignore
	score = calculate_ip_threat_score(True, False, False, False, 0, 0)
	assert score == 0.40


def test_ip_threat_score_clean():
	from domain.calculations import calculate_ip_threat_score  # type: ignore
	score = calculate_ip_threat_score(False, False, False, False, 0, 0)
	assert score == 0.0


def test_deduplicate_entities():
	from domain.calculations import deduplicate_entities  # type: ignore
	entities = [
		{"name": "John Smith", "aliases": [], "confidence_score": 0.8},
		{"name": "John Smith", "aliases": ["J. Smith"], "confidence_score": 0.9},
		{"name": "Jane Doe", "aliases": [], "confidence_score": 0.7},
	]
	merged = deduplicate_entities(entities, similarity_threshold=1.0)  # exact match only
	assert len(merged) == 2  # John Smith deduplicated


def test_connected_clusters():
	from domain.calculations import find_connected_clusters  # type: ignore
	entities = ["e1", "e2", "e3", "e4"]
	rels = [
		{"source_entity_id": "e1", "target_entity_id": "e2"},
		{"source_entity_id": "e3", "target_entity_id": "e4"},
	]
	clusters = find_connected_clusters(entities, rels)
	assert len(clusters) == 2


# ---------------------------------------------------------------------------
# Rule unit tests
# ---------------------------------------------------------------------------

def test_assert_confidence_bounds_valid():
	from domain.rules import assert_confidence_bounds  # type: ignore
	assert_confidence_bounds(0.0)
	assert_confidence_bounds(0.5)
	assert_confidence_bounds(1.0)


def test_assert_confidence_bounds_invalid():
	from domain.rules import assert_confidence_bounds, RuleViolation  # type: ignore
	try:
		assert_confidence_bounds(1.5)
		assert False
	except RuleViolation as exc:
		assert "confidence" in exc.rule_name


def test_calculate_source_credibility():
	from domain.rules import calculate_intel_credibility  # type: ignore
	score = calculate_intel_credibility(
		source_credibility=0.8,
		corroboration_count=3,
		analyst_confidence=0.9,
		timeliness_score=0.95,
	)
	assert 0.0 <= score <= 1.0
	assert score > 0.5  # high inputs should yield high score


def test_calculate_relationship_strength():
	from domain.rules import calculate_relationship_strength  # type: ignore
	strength = calculate_relationship_strength(
		evidence_count=5,
		avg_confidence=0.8,
		temporal_consistency=0.9,
	)
	assert 0.0 <= strength <= 1.0


# ---------------------------------------------------------------------------
# Capability contract tests
# ---------------------------------------------------------------------------

def test_get_capability_contract():
	from capability_contract import get_capability_contract  # type: ignore
	contract = get_capability_contract("tenant-x")
	assert contract["capability"] == "intel_osint"
	assert contract["version"] == "2.0.0"
	assert contract["configuration"]["tenant_id"] == "tenant-x"
	assert "sources" in contract["configuration"]
	assert "agents" in contract["configuration"]
	assert len(contract["ui"]["routes"]) >= 10


def test_evaluate_rules_allow():
	from capability_contract import evaluate_capability_rules  # type: ignore
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"


def test_evaluate_rules_deny_no_tenant():
	from capability_contract import evaluate_capability_rules  # type: ignore
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	reasons = [a["reason"] for a in result["actions"]]
	assert "tenant_context_required" in reasons


def test_evaluate_rules_deny_no_policy():
	from capability_contract import evaluate_capability_rules  # type: ignore
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": False,
	})
	assert result["decision"] == "deny"


# ---------------------------------------------------------------------------
# Runner — use asyncio.get_event_loop() pattern per CLAUDE.md
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	loop = asyncio.get_event_loop()

	tests = [
		test_register_source_success,
		test_register_source_missing_terms_review,
		test_list_sources,
		test_source_tenant_isolation,
		test_create_task,
		test_task_lifecycle,
		test_task_fail,
		test_high_risk_task_requires_approval,
		test_ingest_raw_intel,
		test_duplicate_fingerprint_rejected,
		test_triage_raw_intel,
		test_create_processed_intel,
		test_extract_entity,
		test_entity_name_required,
		test_map_relationship,
		test_self_loop_relationship_denied,
		test_web_scrape,
		test_domain_intelligence,
		test_ip_geolocation_enrichment,
		test_entity_extraction_nlp,
		test_social_media_monitor,
		test_dissemination_package,
		test_dissemination_requires_approval,
		test_duplicate_deduplication,
		test_relationship_mapping_report,
		test_dashboard_summary,
		test_register_agent,
		test_validate_agent_action_privileged_no_approval,
		test_validate_agent_action_approved,
		test_record_review,
		test_credibility_scoring,
	]

	sync_tests = [
		test_timeliness_fresh,
		test_timeliness_old,
		test_content_fingerprint,
		test_ip_threat_score_tor,
		test_ip_threat_score_clean,
		test_deduplicate_entities,
		test_connected_clusters,
		test_assert_confidence_bounds_valid,
		test_assert_confidence_bounds_invalid,
		test_calculate_source_credibility,
		test_calculate_relationship_strength,
		test_get_capability_contract,
		test_evaluate_rules_allow,
		test_evaluate_rules_deny_no_tenant,
		test_evaluate_rules_deny_no_policy,
	]

	passed = 0
	failed = 0

	for t in sync_tests:
		try:
			t()
			print(f"  PASS  {t.__name__}")
			passed += 1
		except Exception as exc:
			print(f"  FAIL  {t.__name__}: {exc}")
			failed += 1

	for t in tests:
		try:
			loop.run_until_complete(t())
			print(f"  PASS  {t.__name__}")
			passed += 1
		except Exception as exc:
			print(f"  FAIL  {t.__name__}: {exc}")
			failed += 1

	print(f"\n{passed} passed, {failed} failed")
	sys.exit(0 if failed == 0 else 1)
