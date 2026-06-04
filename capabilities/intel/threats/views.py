"""Flask Blueprint UI views for APG Threat Intelligence.

Provides complete UI view models and Flask Blueprint endpoints for all
threat intelligence screens: dashboard, actors, indicators, campaigns,
reports, feeds, assessments, requirements, agents, and MITRE ATT&CK heatmap.

Blueprint prefix: /intel-threats
"""
from __future__ import annotations

import asyncio
from typing import Any

try:
	from flask import Blueprint, jsonify, render_template_string, request
	_FLASK = True
except ImportError:  # pragma: no cover
	_FLASK = False

try:
	from .capability_contract import get_capability_contract
	from .service import ThreatIntelligenceService
	from .domain.calculations import (
		indicator_staleness, admiralty_confidence, threat_risk_score,
		mitre_coverage_percentage, weighted_actor_confidence,
	)
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import ThreatIntelligenceService  # type: ignore
	from domain.calculations import (  # type: ignore
		indicator_staleness, admiralty_confidence, threat_risk_score,
		mitre_coverage_percentage, weighted_actor_confidence,
	)


# ── Async runner ──────────────────────────────────────────────────────────────

def _run(coro):
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


# ── View model builders ───────────────────────────────────────────────────────

def dashboard_model(service: ThreatIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Top-level dashboard: KPIs, recent indicators, active campaigns, threat heatmap."""
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)

	# Recent IOCs from extended store
	ioc_store = service._ioc_store()
	recent_iocs = sorted(
		[{k: v for k, v in r.items() if not k.startswith("_")} for r in ioc_store.values()],
		key=lambda x: x.get("created_at", ""),
		reverse=True,
	)[:10]

	# Active campaign count from extended store
	active_campaigns = [
		c for c in service._campaign_store().values()
		if c.get("status") == "active"
	]

	# Active actor profiles
	active_actors = [
		a for a in service._actor_profiles.values()
		if a.get("status") == "active"
	]

	# Feed status
	feeds_summary = [
		{"name": f["name"], "status": f["status"], "last_ingested_at": f.get("last_ingested_at")}
		for f in service._feeds.values()
	]

	# MITRE tactic coverage across all campaigns
	all_techniques: set[str] = set()
	for ct in service._campaign_techniques:
		all_techniques.add(ct["technique_id"])
	mitre_pct = mitre_coverage_percentage(all_techniques)

	# Open requirements
	open_reqs = [r for r in service._requirements.values() if r.get("status") == "open"]

	kpis = {
		"total_indicators": summary.get("indicator_count", 0) + len(ioc_store),
		"active_actors": summary.get("actor_count", 0) + len(active_actors),
		"active_campaigns": summary.get("campaign_count", 0) + len(active_campaigns),
		"total_reports": summary.get("report_count", 0) + len(service._threat_reports),
		"open_requirements": len(open_reqs),
		"feed_count": len(service._feeds),
		"mitre_coverage_pct": mitre_pct,
		"audit_events_today": summary.get("audit_event_count", 0),
	}

	return {
		"title": "Threat Intelligence Dashboard",
		"tenant_id": tenant_id,
		"kpis": kpis,
		"summary": summary,
		"recent_indicators": recent_iocs,
		"active_campaigns": active_campaigns[:5],
		"active_actors": active_actors[:5],
		"feeds": feeds_summary,
		"open_requirements": open_reqs[:5],
		"mitre_coverage_pct": mitre_pct,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"nav_groups": _build_nav_groups(contract["ui"]["routes"]),
	}


def list_indicators_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
	page: int = 1,
	page_size: int = 50,
	ioc_type: str | None = None,
	status: str | None = None,
) -> dict[str, Any]:
	"""Indicator list view with staleness scores and filters."""
	ioc_store = service._ioc_store()
	items = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ioc_store.values()]

	if ioc_type:
		items = [i for i in items if i.get("ioc_type") == ioc_type]
	if status:
		items = [i for i in items if i.get("status") == status]

	# Also include contract-model indicators for this tenant
	contract_items = [
		v.to_dict() for (tid, _), v in service.indicators.items()
		if tid == tenant_id
	]
	items = contract_items + items
	items.sort(key=lambda x: x.get("confidence", x.get("confidence_score", 0)), reverse=True)

	total = len(items)
	start = (page - 1) * page_size
	paginated = items[start:start + page_size]

	return {
		"title": "Indicators (IOC)",
		"tenant_id": tenant_id,
		"items": paginated,
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, (total + page_size - 1) // page_size),
		"filters": {"ioc_type": ioc_type, "status": status},
		"ioc_types": [
			"ip_address", "domain", "url", "file_hash_md5", "file_hash_sha1",
			"file_hash_sha256", "email_address", "registry_key", "mutex",
			"certificate", "user_agent", "yara_rule",
		],
	}


def detail_indicator_model(
	service: ThreatIntelligenceService,
	indicator_id: str,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Indicator detail with enrichment, linked actors and campaigns."""
	ioc_store = service._ioc_store()
	ioc = {k: v for k, v in ioc_store.get(indicator_id, {}).items() if not k.startswith("_")}

	enrichment = service._enrichments.get(indicator_id, {})

	# Find campaigns that reference this indicator
	linked_campaigns = [
		{"campaign_id": ci["campaign_id"], "first_seen": ci["first_seen"], "last_seen": ci["last_seen"]}
		for ci in service._campaign_indicators
		if ci["indicator_id"] == indicator_id
	]

	# Find actors linked to this indicator
	linked_actors = [
		{"actor_id": lnk["actor_id"], "relationship_type": lnk["relationship_type"], "confidence": lnk["confidence"]}
		for lnk in service._actor_indicator_links
		if lnk["indicator_id"] == indicator_id
	]

	return {
		"title": f"Indicator: {ioc.get('value', indicator_id)}",
		"indicator": ioc,
		"enrichment": enrichment,
		"linked_campaigns": linked_campaigns,
		"linked_actors": linked_actors,
		"tenant_id": tenant_id,
	}


def list_actors_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
	page: int = 1,
	page_size: int = 20,
) -> dict[str, Any]:
	"""Threat actor list with attribution confidence."""
	profiles = list(service._actor_profiles.values())
	contract_actors = [
		v.to_dict() for (tid, _), v in service.actors.items()
		if tid == tenant_id
	]
	combined = contract_actors + profiles
	combined.sort(key=lambda x: x.get("name", ""))

	total = len(combined)
	start = (page - 1) * page_size
	paginated = combined[start:start + page_size]

	return {
		"title": "Threat Actors",
		"tenant_id": tenant_id,
		"items": paginated,
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, (total + page_size - 1) // page_size),
	}


def detail_actor_model(
	service: ThreatIntelligenceService,
	actor_id: str,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Actor detail with attribution dossier, linked indicators and campaigns."""
	profile = service._actor_profiles.get(actor_id, {})

	ioc_store = service._ioc_store()
	linked_indicators = [
		{
			"relationship_type": lnk["relationship_type"],
			"confidence": lnk["confidence"],
			"indicator": {k: v for k, v in ioc_store.get(lnk["indicator_id"], {}).items() if not k.startswith("_")},
		}
		for lnk in service._actor_indicator_links
		if lnk["actor_id"] == actor_id
	]

	linked_campaigns = [
		{"campaign_id": lnk["campaign_id"], "role": lnk["role"]}
		for lnk in service._actor_campaign_links
		if lnk["actor_id"] == actor_id
	]

	# Compute attribution confidence
	evidence_scores = [lnk["confidence"] for lnk in service._actor_indicator_links if lnk["actor_id"] == actor_id]
	from domain.calculations import attribution_confidence  # type: ignore
	try:
		from .domain.calculations import attribution_confidence as _ac
	except ImportError:
		from domain.calculations import attribution_confidence as _ac  # type: ignore
	attr_conf = _ac(evidence_scores) if evidence_scores else 0.0

	return {
		"title": f"Threat Actor: {profile.get('name', actor_id)}",
		"actor": profile,
		"linked_indicators": linked_indicators,
		"linked_campaigns": linked_campaigns,
		"attribution_confidence": attr_conf,
		"tenant_id": tenant_id,
	}


def list_campaigns_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
	page: int = 1,
	page_size: int = 20,
) -> dict[str, Any]:
	"""Campaign list with risk level, indicator count, and actor attribution."""
	ext = list(service._campaign_store().values())
	contract_campaigns = [
		v.to_dict() for (tid, _), v in service.campaigns.items()
		if tid == tenant_id
	]
	combined = contract_campaigns + ext
	combined.sort(key=lambda x: x.get("name", ""))

	total = len(combined)
	start = (page - 1) * page_size
	paginated = combined[start:start + page_size]

	return {
		"title": "Threat Campaigns",
		"tenant_id": tenant_id,
		"items": paginated,
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, (total + page_size - 1) // page_size),
	}


def detail_campaign_model(
	service: ThreatIntelligenceService,
	campaign_id: str,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Campaign detail with timeline, techniques, and linked actors."""
	campaign = service._campaign_store().get(campaign_id, {})

	indicators = [
		{
			"indicator_id": ci["indicator_id"],
			"first_seen": ci["first_seen"],
			"last_seen": ci["last_seen"],
		}
		for ci in service._campaign_indicators
		if ci["campaign_id"] == campaign_id
	]

	techniques = [
		ct for ct in service._campaign_techniques
		if ct["campaign_id"] == campaign_id
	]

	actors = [
		{"actor_id": lnk["actor_id"], "role": lnk["role"]}
		for lnk in service._actor_campaign_links
		if lnk["campaign_id"] == campaign_id
	]

	tactics_observed = {ct["tactic"] for ct in techniques}

	return {
		"title": f"Campaign: {campaign.get('name', campaign_id)}",
		"campaign": campaign,
		"indicators": indicators,
		"techniques": techniques,
		"actors": actors,
		"tactics_observed": sorted(tactics_observed),
		"tenant_id": tenant_id,
	}


def list_reports_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
	page: int = 1,
	page_size: int = 20,
) -> dict[str, Any]:
	"""Report list with classification, status, and type."""
	ext = list(service._threat_reports.values())
	contract_reports = [
		v.to_dict() for (tid, _), v in service.reports.items()
		if tid == tenant_id
	]
	combined = contract_reports + ext
	combined.sort(key=lambda x: x.get("created_at", ""), reverse=True)

	total = len(combined)
	start = (page - 1) * page_size
	paginated = combined[start:start + page_size]

	return {
		"title": "Threat Reports",
		"tenant_id": tenant_id,
		"items": paginated,
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, (total + page_size - 1) // page_size),
	}


def detail_report_model(
	service: ThreatIntelligenceService,
	report_id: str,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Report detail with indicators, actors, campaigns, and dissemination log."""
	report = service._threat_reports.get(report_id, {})
	dissemination = [e for e in service._dissemination_log if e.get("report_id") == report_id]

	return {
		"title": f"Report: {report.get('title', report_id)}",
		"report": report,
		"dissemination_log": dissemination,
		"tenant_id": tenant_id,
	}


def list_feeds_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Feed management view with quality grades and ingestion status."""
	feeds = list(service._feeds.values())
	feeds.sort(key=lambda f: f.get("name", ""))

	batches_by_feed: dict[str, list] = {}
	for b in service._feed_batches:
		batches_by_feed.setdefault(b["feed_id"], []).append(b)

	feed_rows = []
	for f in feeds:
		batches = batches_by_feed.get(f["id"], [])
		total_imported = sum(b.get("imported", 0) for b in batches)
		feed_rows.append({
			**f,
			"batch_count": len(batches),
			"total_imported": total_imported,
		})

	return {
		"title": "Threat Feeds",
		"tenant_id": tenant_id,
		"feeds": feed_rows,
		"total": len(feed_rows),
	}


def list_assessments_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
	page: int = 1,
	page_size: int = 20,
) -> dict[str, Any]:
	"""Assessment list with risk level and confidence scores."""
	items = [
		v.to_dict() for (tid, _), v in service.assessments.items()
		if tid == tenant_id
	]
	items.sort(key=lambda x: x.get("id", ""))

	total = len(items)
	start = (page - 1) * page_size
	paginated = items[start:start + page_size]

	return {
		"title": "Threat Assessments",
		"tenant_id": tenant_id,
		"items": paginated,
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, (total + page_size - 1) // page_size),
	}


def list_requirements_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Priority Intelligence Requirements (PIR) list."""
	items = list(service._requirements.values())
	priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
	items.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 9))

	return {
		"title": "Intelligence Requirements (PIR)",
		"tenant_id": tenant_id,
		"items": items,
		"total": len(items),
		"open_count": sum(1 for i in items if i.get("status") == "open"),
	}


def mitre_heatmap_model(
	service: ThreatIntelligenceService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""MITRE ATT&CK heatmap: tactic coverage and technique frequency."""
	technique_freq: dict[str, int] = {}
	for ct in service._campaign_techniques:
		tid = ct["technique_id"]
		technique_freq[tid] = technique_freq.get(tid, 0) + 1

	tactic_coverage: dict[str, int] = {}
	for ct in service._campaign_techniques:
		tac = ct.get("tactic", "unknown")
		tactic_coverage[tac] = tactic_coverage.get(tac, 0) + 1

	all_tactics = [
		"reconnaissance", "resource-development", "initial-access", "execution",
		"persistence", "privilege-escalation", "defense-evasion", "credential-access",
		"discovery", "lateral-movement", "collection", "command-and-control",
		"exfiltration", "impact",
	]

	coverage_pct = mitre_coverage_percentage(set(technique_freq.keys()))

	return {
		"title": "MITRE ATT&CK Heatmap",
		"tenant_id": tenant_id,
		"tactic_coverage": tactic_coverage,
		"technique_frequency": technique_freq,
		"all_tactics": all_tactics,
		"covered_tactics": list(tactic_coverage.keys()),
		"uncovered_tactics": [t for t in all_tactics if t not in tactic_coverage],
		"coverage_pct": coverage_pct,
		"top_techniques": sorted(technique_freq.items(), key=lambda x: x[1], reverse=True)[:10],
	}


def agent_workbench_model(service: ThreatIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Agent workbench view with supported runtimes and registered agents."""
	contract = get_capability_contract(tenant_id)
	agents = [
		item.to_dict() for (tid, _), item in service.agents.items()
		if tid == tenant_id
	]
	agents.sort(key=lambda a: a.get("id", ""))

	return {
		"title": "Agent Workbench",
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": agents,
		"total": len(agents),
	}


def threat_console_model(service: ThreatIntelligenceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Full console model with all tenant-scoped entities."""
	return {
		"tenant_id": tenant_id,
		"authorities": _tenant_items(service.authorities, tenant_id),
		"workspaces": _tenant_items(service.workspaces, tenant_id),
		"sources": _tenant_items(service.sources, tenant_id),
		"indicators": _tenant_items(service.indicators, tenant_id),
		"actors": _tenant_items(service.actors, tenant_id),
		"campaigns": _tenant_items(service.campaigns, tenant_id),
		"assessments": _tenant_items(service.assessments, tenant_id),
		"reports": _tenant_items(service.reports, tenant_id),
		"mitigations": _tenant_items(service.mitigations, tenant_id),
		"reviews": _tenant_items(service.reviews, tenant_id),
		"agents": _tenant_items(service.agents, tenant_id),
	}


# ── Private helpers ───────────────────────────────────────────────────────────

def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [
		item.to_dict()
		for item in sorted(items.values(), key=lambda v: v.id)
		if item.tenant_id == tenant_id
	]


def _build_nav_groups(routes: list[dict]) -> dict[str, list[dict]]:
	groups: dict[str, list[dict]] = {}
	for route in routes:
		group = route.get("nav_group", "Other")
		groups.setdefault(group, []).append(route)
	return groups


# ── Flask Blueprint UI routes ─────────────────────────────────────────────────

if _FLASK:
	blueprint = Blueprint("intel_threats_views", __name__, url_prefix="/intel-threats")

	_SINGLETON_SERVICE: ThreatIntelligenceService | None = None

	def _svc() -> ThreatIntelligenceService:
		global _SINGLETON_SERVICE
		if _SINGLETON_SERVICE is None:
			try:
				from .api import service
			except ImportError:
				from api import service  # type: ignore
			_SINGLETON_SERVICE = service()
		return _SINGLETON_SERVICE

	def _get_tenant() -> str:
		return (
			request.headers.get("X-Tenant-Id")
			or request.args.get("tenant_id", "default")
		)

	def _json(data):
		return jsonify(data)

	@blueprint.get("/dashboard")
	def view_dashboard():
		return _json(dashboard_model(_svc(), _get_tenant()))

	@blueprint.get("/indicators")
	def view_indicators():
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 50)), 200)
		ioc_type = request.args.get("ioc_type")
		status = request.args.get("status")
		return _json(list_indicators_model(_svc(), _get_tenant(), page, page_size, ioc_type, status))

	@blueprint.get("/indicators/<indicator_id>")
	def view_indicator_detail(indicator_id: str):
		return _json(detail_indicator_model(_svc(), indicator_id, _get_tenant()))

	@blueprint.get("/actors")
	def view_actors():
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		return _json(list_actors_model(_svc(), _get_tenant(), page, page_size))

	@blueprint.get("/actors/<actor_id>")
	def view_actor_detail(actor_id: str):
		return _json(detail_actor_model(_svc(), actor_id, _get_tenant()))

	@blueprint.get("/campaigns")
	def view_campaigns():
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		return _json(list_campaigns_model(_svc(), _get_tenant(), page, page_size))

	@blueprint.get("/campaigns/<campaign_id>")
	def view_campaign_detail(campaign_id: str):
		return _json(detail_campaign_model(_svc(), campaign_id, _get_tenant()))

	@blueprint.get("/reports")
	def view_reports():
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		return _json(list_reports_model(_svc(), _get_tenant(), page, page_size))

	@blueprint.get("/reports/<report_id>")
	def view_report_detail(report_id: str):
		return _json(detail_report_model(_svc(), report_id, _get_tenant()))

	@blueprint.get("/feeds")
	def view_feeds():
		return _json(list_feeds_model(_svc(), _get_tenant()))

	@blueprint.get("/assessments")
	def view_assessments():
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		return _json(list_assessments_model(_svc(), _get_tenant(), page, page_size))

	@blueprint.get("/requirements")
	def view_requirements():
		return _json(list_requirements_model(_svc(), _get_tenant()))

	@blueprint.get("/mitre-heatmap")
	def view_mitre_heatmap():
		return _json(mitre_heatmap_model(_svc(), _get_tenant()))

	@blueprint.get("/agents")
	def view_agents():
		return _json(agent_workbench_model(_svc(), _get_tenant()))

	@blueprint.get("/console")
	def view_console():
		return _json(threat_console_model(_svc(), _get_tenant()))
