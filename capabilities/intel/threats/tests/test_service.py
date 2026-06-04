"""Async service integration tests for APG Threat Intelligence.

Tests cover all major service methods: CRUD, indicator lifecycle, actor
profiling, campaign tracking, MITRE mapping, reporting, feeds, sharing.
No mocks — uses real ThreatIntelligenceService instances.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
	sys.path.insert(0, str(_PKG))

from service import ThreatIntelligenceService


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_service() -> ThreatIntelligenceService:
	return ThreatIntelligenceService()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _setup_base(svc: ThreatIntelligenceService, tenant: str = "t1") -> dict:
	"""Create authority → workspace → source; return ids."""
	auth = svc.record_authority(
		"auth-1", tenant, "mission_order", "scope-ref",
		"confidential", "approver-1", "2027-12-31", "auth-ev",
	)
	ws = svc.record_workspace(
		"ws-1", tenant, "cyber_threat", "CyberOps",
		"confidential", auth["id"], "ws-ev",
	)
	src = svc.register_source(
		"src-1", tenant, ws["id"], "osint",
		"source-ref", "custodian-1", "lineage-ref", "src-ev",
	)
	return {"auth": auth, "ws": ws, "src": src}


# ── Contract / describe / evaluate ───────────────────────────────────────────

def test_describe_returns_contract():
	svc = make_service()
	contract = svc.describe("t1")
	assert contract["capability"] == "intel_threats"
	assert "provides" in contract
	assert "streaming" in contract


def test_evaluate_deny_missing_tenant():
	svc = make_service()
	result = svc.evaluate({"tenant_context_present": False})
	assert result["decision"] == "deny"


def test_evaluate_allow_valid_context():
	svc = make_service()
	result = svc.evaluate({"tenant_id": "t1", "tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"


# ── Authority CRUD ────────────────────────────────────────────────────────────

def test_record_authority_success():
	svc = make_service()
	auth = svc.record_authority(
		"a1", "t1", "mission_order", "scope", "confidential",
		"approver", "2027-01-01", "ev",
	)
	assert auth["authority_type"] == "mission_order"
	assert auth["tenant_id"] == "t1"


def test_record_authority_invalid_type_raises():
	svc = make_service()
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		svc.record_authority("a1", "t1", "INVALID", "scope", "confidential", "approver", "2027-01-01", "ev")


def test_record_authority_missing_tenant_raises():
	svc = make_service()
	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.record_authority("a1", "", "mission_order", "scope", "confidential", "approver", "2027-01-01", "ev")


def test_authority_tenant_isolation():
	svc = make_service()
	svc.record_authority("a1", "t1", "mission_order", "scope", "confidential", "approver", "2027-01-01", "ev")
	svc.record_authority("a1", "t2", "consent", "scope", "unclassified", "approver", "2027-01-01", "ev")
	assert svc.authorities.get(("t1", "a1")).authority_type == "mission_order"
	assert svc.authorities.get(("t2", "a1")).authority_type == "consent"
	assert svc.dashboard_summary("t1")["authority_count"] == 1
	assert svc.dashboard_summary("t2")["authority_count"] == 1


# ── Workspace CRUD ────────────────────────────────────────────────────────────

def test_record_workspace_requires_valid_authority():
	svc = make_service()
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		svc.record_workspace("w1", "t1", "cyber_threat", "CyberOps", "confidential", "missing-auth", "ev")


def test_record_workspace_success():
	svc = make_service()
	auth = svc.record_authority("a1", "t1", "mission_order", "scope", "confidential", "approver", "2027-01-01", "ev")
	ws = svc.record_workspace("w1", "t1", "cyber_threat", "CyberOps", "confidential", auth["id"], "ev")
	assert ws["workspace_type"] == "cyber_threat"
	assert ws["name"] == "CyberOps"


# ── Source CRUD ───────────────────────────────────────────────────────────────

def test_register_source_requires_lineage():
	svc = make_service()
	loop = asyncio.get_event_loop()
	base = loop.run_until_complete(_setup_base(svc))
	with pytest.raises(PermissionError, match="source_lineage_required"):
		svc.register_source("src-bad", "t1", base["ws"]["id"], "osint", "ref", "custodian", "", "ev")


def test_register_source_success():
	svc = make_service()
	loop = asyncio.get_event_loop()
	base = loop.run_until_complete(_setup_base(svc))
	assert base["src"]["source_type"] == "osint"


# ── Indicator lifecycle ───────────────────────────────────────────────────────

def test_record_indicator_success():
	svc = make_service()
	loop = asyncio.get_event_loop()
	base = loop.run_until_complete(_setup_base(svc))
	ind = svc.record_indicator("ind-1", "t1", base["src"]["id"], "ioc", "192.168.1.1", 0.85, "ev")
	assert ind["indicator_type"] == "ioc"
	assert ind["confidence_score"] == 0.85


def test_record_indicator_bad_confidence():
	svc = make_service()
	loop = asyncio.get_event_loop()
	base = loop.run_until_complete(_setup_base(svc))
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		svc.record_indicator("ind-bad", "t1", base["src"]["id"], "ioc", "1.1.1.1", 2.0, "ev")


async def test_create_indicator_async():
	svc = make_service()
	ioc = await svc.create_indicator("ip_address", "10.0.0.1", 0.75, "green", "test-source")
	assert ioc["value"] == "10.0.0.1"
	assert ioc["ioc_type"] == "ip_address"
	assert ioc["status"] == "active"


async def test_enrich_indicator():
	svc = make_service()
	ioc = await svc.create_indicator("ip_address", "185.220.101.1", 0.9, "amber", "test")
	enrichment = await svc.enrich_indicator(ioc["id"])
	assert "geolocation" in enrichment
	assert "asn" in enrichment
	assert "vt_score" in enrichment


async def test_enrich_domain():
	svc = make_service()
	ioc = await svc.create_indicator("domain", "evil-domain.com", 0.8, "green", "test")
	enrichment = await svc.enrich_indicator(ioc["id"])
	assert "whois" in enrichment
	assert "dns_records" in enrichment


async def test_enrich_file_hash():
	svc = make_service()
	ioc = await svc.create_indicator("file_hash_sha256", "abc123deadbeef" * 4, 0.95, "red", "test")
	enrichment = await svc.enrich_indicator(ioc["id"])
	assert "detection" in enrichment
	assert "malware_family" in enrichment


async def test_enrich_cve():
	svc = make_service()
	ioc = await svc.create_indicator("cve_id", "CVE-2024-12345", 0.99, "white", "nvd")
	enrichment = await svc.enrich_indicator(ioc["id"])
	assert "cvss_v3" in enrichment
	assert "epss_score" in enrichment


async def test_retire_indicator():
	svc = make_service()
	ioc = await svc.create_indicator("domain", "bad.com", 0.7, "green", "test")
	retired = await svc.retire_indicator(ioc["id"], "false_positive")
	assert retired["status"] == "retired"
	assert retired["retirement_reason"] == "false_positive"


async def test_retire_already_retired_raises():
	svc = make_service()
	ioc = await svc.create_indicator("domain", "bad.com", 0.7, "green", "test")
	await svc.retire_indicator(ioc["id"], "first_reason")
	with pytest.raises(AssertionError):
		await svc.retire_indicator(ioc["id"], "second_reason")


async def test_search_indicators():
	svc = make_service()
	await svc.create_indicator("domain", "malware.example.com", 0.9, "amber", "osint")
	await svc.create_indicator("ip_address", "1.2.3.4", 0.6, "green", "osint")
	results = await svc.search_indicators("malware")
	assert any("malware" in r["value"] for r in results)


async def test_search_indicators_by_type():
	svc = make_service()
	await svc.create_indicator("domain", "x.com", 0.8, "green", "t")
	await svc.create_indicator("ip_address", "5.5.5.5", 0.8, "green", "t")
	results = await svc.search_indicators("", ioc_types=["domain"])
	assert all(r["ioc_type"] == "domain" for r in results)


async def test_staleness_management():
	svc = make_service()
	# Create 2 indicators; they'll be brand-new so 0-day sweep won't touch them
	await svc.create_indicator("domain", "fresh.com", 0.8, "green", "test")
	result = await svc.staleness_management(older_than_days=3650)  # 10 years — nothing stale
	assert result["evaluated"] >= 1
	assert result["older_than_days"] == 3650


async def test_bulk_import_stix_bundle():
	svc = make_service()
	bundle = {
		"type": "bundle",
		"id": "bundle--test-001",
		"spec_version": "2.1",
		"objects": [
			{"type": "ipv4-addr", "id": "ipv4-addr--1", "value": "10.10.10.1"},
			{"type": "domain-name", "id": "domain-name--1", "value": "c2.evil.com"},
			{"type": "url", "id": "url--1", "value": "http://evil.com/payload"},
			{"type": "email-addr", "id": "email-addr--1", "value": "attacker@evil.com"},
			{"type": "relationship", "id": "relationship--1", "source_ref": "x", "target_ref": "y"},
		],
	}
	result = await svc.bulk_import_indicators(bundle)
	assert result["imported_count"] == 4
	assert result["skipped_count"] == 1  # relationship
	assert result["error_count"] == 0


async def test_bulk_import_deduplication():
	svc = make_service()
	bundle = {
		"type": "bundle",
		"id": "bundle--dup-001",
		"spec_version": "2.1",
		"objects": [
			{"type": "ipv4-addr", "id": "ipv4-addr--a", "value": "10.10.10.1"},
			{"type": "ipv4-addr", "id": "ipv4-addr--b", "value": "10.10.10.1"},  # dup
		],
	}
	result = await svc.bulk_import_indicators(bundle)
	assert result["imported_count"] == 1
	assert result["skipped_count"] == 1


async def test_export_stix():
	svc = make_service()
	await svc.create_indicator("ip_address", "8.8.8.8", 0.9, "green", "test")
	result = await svc.export_indicators(format="stix")
	assert result["format"] == "stix"
	assert "objects" in result["payload"]


async def test_export_misp():
	svc = make_service()
	await svc.create_indicator("domain", "bad.com", 0.8, "amber", "test")
	result = await svc.export_indicators(format="misp")
	assert "Event" in result["payload"]
	assert len(result["payload"]["Event"]["Attribute"]) >= 1


async def test_export_csv():
	svc = make_service()
	await svc.create_indicator("ip_address", "1.1.1.1", 0.5, "white", "test")
	result = await svc.export_indicators(format="csv")
	assert "csv" in result["payload"]
	assert "id,ioc_type" in result["payload"]["csv"]


async def test_indicator_overlap_check():
	svc = make_service()
	ioc = await svc.create_indicator("domain", "overlap.com", 0.9, "amber", "test")
	camp = await svc.create_campaign("TestCamp", "2024-01-01", "espionage", ["finance"], ["EU"])
	await svc.add_campaign_indicator(camp["id"], ioc["id"], "2024-01-01", "2024-06-01")
	overlaps = await svc.indicator_overlap_check("overlap.com")
	assert len(overlaps) == 1
	assert overlaps[0]["campaign_id"] == camp["id"]


# ── Threat Actors ─────────────────────────────────────────────────────────────

def test_record_actor_success():
	svc = make_service()
	loop = asyncio.get_event_loop()
	base = loop.run_until_complete(_setup_base(svc))
	actor = svc.record_actor("act-1", "t1", base["ws"]["id"], "criminal_group", "APT29", 0.88, "ev")
	assert actor["actor_type"] == "criminal_group"


def test_record_actor_invalid_type():
	svc = make_service()
	loop = asyncio.get_event_loop()
	base = loop.run_until_complete(_setup_base(svc))
	with pytest.raises(PermissionError, match="actor_type_not_supported"):
		svc.record_actor("act-bad", "t1", base["ws"]["id"], "ghost_network", "X", 0.5, "ev")


async def test_create_threat_actor_profile():
	svc = make_service()
	actor = await svc.create_threat_actor(
		name="Cozy Bear",
		aliases=["APT29", "The Dukes"],
		motivation="espionage",
		sophistication="nation-state",
		origin_country="RU",
	)
	assert actor["name"] == "Cozy Bear"
	assert actor["origin_country"] == "RU"
	assert "stix_id" in actor


async def test_update_actor_profile():
	svc = make_service()
	actor = await svc.create_threat_actor("FancyBear", [], "espionage", "advanced", "RU")
	updated = await svc.update_actor_profile(
		actor_id=actor["id"],
		ttps=["T1566", "T1059.001", "INVALID-TTP"],
		target_sectors=["government", "defense"],
		known_tools=["Mimikatz", "Cobalt Strike"],
	)
	assert "T1566" in updated["ttps_verified"]
	assert "INVALID-TTP" in updated["ttps_unverified"]
	assert len(updated["target_sectors"]) == 2


async def test_link_actor_to_indicator():
	svc = make_service()
	actor = await svc.create_threat_actor("Actor1", [], "financial", "intermediate", "CN")
	ioc = await svc.create_indicator("domain", "c2.evil.cn", 0.9, "amber", "test")
	link = await svc.link_actor_to_indicator(actor["id"], ioc["id"], "uses", 0.85)
	assert link["actor_id"] == actor["id"]
	assert link["relationship_type"] == "uses"


async def test_link_actor_to_campaign():
	svc = make_service()
	actor = await svc.create_threat_actor("Actor2", [], "hacktivism", "minimal", "US")
	camp = await svc.create_campaign("Campaign1", "2024-01-01", "disruption", [], [])
	link = await svc.link_actor_to_campaign(actor["id"], camp["id"], "operator")
	assert link["role"] == "operator"


async def test_actor_attribution_report():
	svc = make_service()
	actor = await svc.create_threat_actor("APT1", ["Comment Crew"], "espionage", "advanced", "CN")
	ioc = await svc.create_indicator("ip_address", "192.168.100.1", 0.9, "red", "test")
	await svc.link_actor_to_indicator(actor["id"], ioc["id"], "controls", 0.92)
	await svc.update_actor_profile(actor["id"], ["T1566", "T1041"], ["finance"], ["PlugX"])
	report = await svc.actor_attribution_report(actor["id"])
	assert report["linked_indicator_count"] == 1
	assert report["technique_count"] == 2
	assert "attribution_summary" in report


async def test_actor_search():
	svc = make_service()
	await svc.create_threat_actor("Lazarus Group", ["Hidden Cobra"], "financial", "advanced", "KP")
	await svc.create_threat_actor("Sandworm Team", ["Voodoo Bear"], "disruption", "nation-state", "RU")
	results = await svc.actor_search("lazarus")
	assert len(results) == 1
	assert results[0]["name"] == "Lazarus Group"


async def test_actor_search_by_filter():
	svc = make_service()
	await svc.create_threat_actor("ActorA", [], "espionage", "advanced", "RU")
	await svc.create_threat_actor("ActorB", [], "financial", "intermediate", "CN")
	results = await svc.actor_search("", filters={"motivation": "espionage"})
	assert all(a["motivation"] == "espionage" for a in results)


# ── Campaign Tracking ─────────────────────────────────────────────────────────

async def test_create_campaign():
	svc = make_service()
	camp = await svc.create_campaign(
		name="Operation Sandstorm",
		start_date="2024-03-01",
		objective="credential_theft",
		target_sectors=["energy", "utilities"],
		target_regions=["ME", "EU"],
	)
	assert camp["name"] == "Operation Sandstorm"
	assert "energy" in camp["target_sectors"]
	assert camp["status"] == "active"


async def test_add_campaign_technique():
	svc = make_service()
	camp = await svc.create_campaign("Camp1", "2024-01-01", "espionage", [], [])
	result = await svc.add_campaign_technique(camp["id"], "T1566", "Initial phishing vector")
	assert result["technique_id"] == "T1566"
	assert result["is_verified_technique"] is True
	assert result["tactic"] == "initial-access"


async def test_campaign_timeline():
	svc = make_service()
	camp = await svc.create_campaign("Camp2", "2024-01-01", "ransomware", [], [])
	ioc = await svc.create_indicator("ip_address", "10.10.10.10", 0.9, "amber", "test")
	await svc.add_campaign_indicator(camp["id"], ioc["id"], "2024-01-05", "2024-02-10")
	await svc.add_campaign_technique(camp["id"], "T1486", "Ransomware deployment")
	timeline = await svc.campaign_timeline(camp["id"])
	assert timeline["event_count"] >= 3
	assert "impact" in timeline["tactics_observed"]


async def test_campaign_similarity_high():
	svc = make_service()
	camp1 = await svc.create_campaign("SimilarA", "2024-01-01", "espionage", [], [])
	camp2 = await svc.create_campaign("SimilarB", "2024-02-01", "espionage", [], [])
	# Same indicators
	for ip in ["1.1.1.1", "2.2.2.2", "3.3.3.3"]:
		ioc = await svc.create_indicator("ip_address", ip, 0.8, "green", "test")
		await svc.add_campaign_indicator(camp1["id"], ioc["id"], "2024-01-01", "2024-06-01")
		await svc.add_campaign_indicator(camp2["id"], ioc["id"], "2024-02-01", "2024-06-01")
	similarity = await svc.campaign_similarity(camp1["id"], camp2["id"])
	assert similarity["ioc_overlap"]["jaccard_similarity"] == 1.0
	assert similarity["assessment"] == "high_overlap"


async def test_active_campaigns_report():
	svc = make_service()
	await svc.create_campaign("Active1", "2024-01-01", "disruption", [], [])
	await svc.create_campaign("Active2", "2024-02-01", "espionage", [], [])
	report = await svc.active_campaigns_report()
	assert len(report) == 2


# ── MITRE ATT&CK ─────────────────────────────────────────────────────────────

async def test_map_technique_known():
	svc = make_service()
	result = await svc.map_technique("T1566")
	assert result["found"] is True
	assert result["name"] == "Phishing"
	assert result["tactic"] == "initial-access"
	assert "kill_chain_phase" in result


async def test_map_technique_unknown():
	svc = make_service()
	result = await svc.map_technique("T9999")
	assert result["found"] is False
	assert "note" in result


async def test_coverage_analysis():
	svc = make_service()
	result = await svc.coverage_analysis(["T1566", "T1059", "T1486"])
	assert "covered_tactics" in result
	assert result["coverage_ratio"] > 0
	assert "recommended_techniques" in result


async def test_kill_chain_mapping():
	svc = make_service()
	ioc1 = await svc.create_indicator("ip_address", "9.9.9.9", 0.8, "green", "test")
	ioc2 = await svc.create_indicator("url", "http://evil.com/payload", 0.9, "amber", "test")
	result = await svc.kill_chain_mapping([ioc1["id"], ioc2["id"]])
	assert result["mapped_count"] == 2
	assert "delivery" in result["kill_chain_phases"]


async def test_attack_path_analysis():
	svc = make_service()
	result = await svc.attack_path_analysis(["T1566", "T1059", "T1055", "T1041"])
	assert result["valid_techniques"] >= 3
	assert result["current_phase"] is not None
	assert len(result["mitigation_hints"]) > 0


# ── Reporting & Sharing ───────────────────────────────────────────────────────

async def test_generate_threat_report():
	svc = make_service()
	ioc = await svc.create_indicator("domain", "c2.bad.com", 0.9, "amber", "test")
	report = await svc.generate_threat_report(
		classification="tlp:amber",
		report_type="assessment",
		target_audience="SOC Team",
		title="Q2 Assessment",
		summary="Increased APT activity observed",
		indicator_ids=[ioc["id"]],
	)
	assert report["report_type"] == "assessment"
	assert report["indicator_count"] == 1
	assert report["status"] == "draft"


async def test_share_via_taxii():
	svc = make_service()
	ioc = await svc.create_indicator("ip_address", "11.22.33.44", 0.85, "green", "test")
	report = await svc.generate_threat_report(
		"unclassified", "flash_report", "Partner",
		indicator_ids=[ioc["id"]],
	)
	result = await svc.share_via_taxii(
		report["id"],
		"https://taxii.example.com",
		"collection-001",
	)
	assert result["status"] == "submitted"
	assert result["http_status"] == 200
	assert len(svc._taxii_log) == 1


async def test_export_misp_event():
	svc = make_service()
	ioc = await svc.create_indicator("file_hash_sha256", "deadbeef" * 8, 0.95, "red", "malware-lab")
	result = await svc.export_misp_event([ioc["id"]])
	event = result["Event"]
	assert len(event["Attribute"]) == 1
	assert event["Attribute"][0]["type"] == "sha256"


async def test_intelligence_requirement():
	svc = make_service()
	req = await svc.intelligence_requirement(
		"Track APT29 spear-phishing campaigns targeting energy sector",
		"high",
		"CISO",
	)
	assert req["priority"] == "high"
	assert req["status"] == "open"


async def test_dissemination_log():
	svc = make_service()
	report = await svc.generate_threat_report("unclassified", "assessment", "HQ")
	await svc.share_via_taxii(report["id"], "https://taxii.test.com", "col-1")
	log = await svc.dissemination_log(report["id"])
	assert len(log) == 1
	assert log[0]["channel"] == "taxii"


async def test_confidence_calibration_no_data():
	svc = make_service()
	report = await svc.confidence_calibration_report("analyst-x", "2024-Q1")
	assert report["total_assessments"] == 0
	assert report["calibration_score"] is None


# ── Feed Management ───────────────────────────────────────────────────────────

async def test_register_feed():
	svc = make_service()
	feed = await svc.register_feed(
		name="AlienVault OTX",
		url="https://otx.alienvault.com/api/v1",
		format="json",
		auth_method="api_key",
		update_frequency="@hourly",
	)
	assert feed["name"] == "AlienVault OTX"
	assert feed["status"] == "registered"


async def test_ingest_feed():
	svc = make_service()
	feed = await svc.register_feed("TestFeed", "https://feed.test.com", "stix", "none", "@daily")
	batch = await svc.ingest_feed(feed["id"])
	assert batch["status"] == "completed"
	assert batch["imported"] > 0
	assert svc._feeds[feed["id"]]["status"] == "active"


async def test_feed_quality_report():
	svc = make_service()
	feed = await svc.register_feed("QFeed", "https://qfeed.com", "csv", "bearer_token", "@weekly")
	await svc.ingest_feed(feed["id"])
	report = await svc.feed_quality_report(feed["id"])
	assert "quality_score" in report
	assert report["quality_grade"] in ("A", "B", "C", "D")


async def test_deduplicate_from_feed():
	svc = make_service()
	# Two identical indicators (same type + value)
	await svc.create_indicator("domain", "dup-domain.com", 0.8, "green", "f1")
	await svc.create_indicator("domain", "dup-domain.com", 0.8, "green", "f1")
	feed = await svc.register_feed("DupFeed", "https://dup.com", "json", "none", "@hourly")
	result = await svc.deduplicate_from_feed(feed["id"], "batch-001")
	assert result["duplicate_groups_found"] >= 1


async def test_feeds_dashboard():
	svc = make_service()
	await svc.register_feed("Feed1", "https://f1.com", "stix", "none", "@hourly")
	await svc.register_feed("Feed2", "https://f2.com", "misp", "api_key", "@daily")
	result = await svc.feeds_dashboard()
	assert result["total_feeds"] == 2
	assert len(result["feeds"]) == 2


# ── Agent management ──────────────────────────────────────────────────────────

def test_register_threat_agent():
	svc = make_service()
	agent = svc.register_threat_agent("ag-1", "t1", "ThreatAgent", "codex", "actor_analyst", "TI operations")
	assert agent["runtime"] == "codex"
	assert agent["role"] == "actor_analyst"


def test_register_agent_bad_runtime():
	svc = make_service()
	with pytest.raises(PermissionError, match="threat_agent_runtime_not_supported"):
		svc.register_threat_agent("ag", "t1", "Bad", "unsupported_runtime", "actor_analyst", "scope")


def test_validate_agent_privileged_requires_approval():
	svc = make_service()
	with pytest.raises(PermissionError, match="human_approval_required"):
		svc.validate_agent_action("t1", privileged_scope=True, human_approval_recorded=False)


def test_validate_agent_approved_succeeds():
	svc = make_service()
	result = svc.validate_agent_action("t1", privileged_scope=True, human_approval_recorded=True)
	assert result["accepted"] is True


def test_validate_batch_bytewax():
	svc = make_service()
	result = svc.validate_batch("t1", 50)
	assert result["processor"] == "bytewax"
	assert result["accepted"] is True


def test_validate_batch_non_bytewax_raises():
	svc = make_service()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 10, event_stream="kafka")


# ── Dashboard ─────────────────────────────────────────────────────────────────

def test_dashboard_summary_counts():
	svc = make_service()
	loop = asyncio.get_event_loop()
	base = loop.run_until_complete(_setup_base(svc))
	svc.record_indicator("i1", "t1", base["src"]["id"], "ioc", "1.1.1.1", 0.8, "ev")
	actor = svc.record_actor("a1", "t1", base["ws"]["id"], "criminal_group", "G1", 0.7, "ev")
	camp = svc.record_campaign("c1", "t1", actor["id"], "intrusion_campaign", "C1", "high", "ev")
	summary = svc.dashboard_summary("t1")
	assert summary["indicator_count"] == 1
	assert summary["actor_count"] == 1
	assert summary["campaign_count"] == 1


def test_dashboard_tenant_isolation():
	svc = make_service()
	loop = asyncio.get_event_loop()
	loop.run_until_complete(_setup_base(svc, "t1"))
	loop.run_until_complete(_setup_base(svc, "t2"))
	s1 = svc.dashboard_summary("t1")
	s2 = svc.dashboard_summary("t2")
	assert s1["source_count"] == 1
	assert s2["source_count"] == 1
