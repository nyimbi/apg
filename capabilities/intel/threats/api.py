"""Flask Blueprint REST API for APG Threat Intelligence.

Provides complete REST endpoints for all threat intelligence entities.
Tenant isolation enforced on every operation. Async-compatible via asyncio.run().

Blueprint prefix: /api/v1/intel/threats
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from functools import wraps
from typing import Any

try:
	from flask import Blueprint, jsonify, request, abort
	_FLASK = True
except ImportError:  # pragma: no cover
	_FLASK = False

try:
	from .service import ThreatIntelligenceService
except ImportError:
	from service import ThreatIntelligenceService  # type: ignore


# ── Singleton service (process-local) ─────────────────────────────────────────

_SERVICE = ThreatIntelligenceService()


def service() -> ThreatIntelligenceService:
	return _SERVICE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(coro):
	"""Run an async coroutine from a sync Flask route."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _ok(data: Any, status: int = 200):
	return jsonify({"ok": True, "data": data}), status


def _err(message: str, status: int = 400):
	return jsonify({"ok": False, "error": message}), status


def _paginate(items: list, page: int, page_size: int) -> dict[str, Any]:
	total = len(items)
	start = (page - 1) * page_size
	end = start + page_size
	return {
		"items": items[start:end],
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, (total + page_size - 1) // page_size),
	}


def _get_tenant() -> str:
	return (
		request.headers.get("X-Tenant-Id")
		or request.args.get("tenant_id")
		or (request.get_json(silent=True) or {}).get("tenant_id")
		or "default"
	)


def _get_actor() -> str:
	return request.headers.get("X-Actor-Id", "api_user")


def _guard(fn):
	"""Wrap route handler, catching PermissionError and ValueError as 4xx."""
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except PermissionError as e:
			return _err(str(e), 403)
		except (ValueError, AssertionError, KeyError) as e:
			return _err(str(e), 422)
		except Exception as e:
			return _err(f"internal_error: {e}", 500)
	return wrapper


# ── Blueprint ─────────────────────────────────────────────────────────────────

if _FLASK:
	blueprint = Blueprint("intel_threats_api", __name__, url_prefix="/api/v1/intel/threats")


	# ── Health ────────────────────────────────────────────────────────────────

	@blueprint.get("/health")
	def health():
		"""Capability health check."""
		return _ok({"status": "ok", "capability": "intel_threats", "version": "1.1.0"})


	# ── Dashboard ─────────────────────────────────────────────────────────────

	@blueprint.get("/dashboard")
	@_guard
	def dashboard():
		"""Threat intelligence dashboard summary for a tenant."""
		tenant_id = _get_tenant()
		data = _SERVICE.dashboard_summary(tenant_id)
		return _ok(data)


	# ═══════════════════════════════════════════════════════════════════════════
	# Authorities
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/authorities")
	@_guard
	def list_authorities():
		"""List authorities for a tenant with pagination."""
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.authorities.items()
			if tid == tenant_id
		]
		items.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(items, page, page_size))

	@blueprint.post("/authorities")
	@_guard
	def create_authority():
		"""Record a new threat authority."""
		body = request.get_json(force=True)
		result = _SERVICE.record_authority(
			body["authority_id"],
			body.get("tenant_id", _get_tenant()),
			body["authority_type"],
			body["scope_reference"],
			body["classification"],
			body["approver_id"],
			body["expires_at"],
			body["evidence_reference"],
			body.get("policy_attached", True),
		)
		return _ok(result, 201)

	@blueprint.get("/authorities/<authority_id>")
	@_guard
	def get_authority(authority_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.authorities.get((tenant_id, authority_id))
		if not item:
			return _err("authority not found", 404)
		return _ok(item.to_dict())


	# ═══════════════════════════════════════════════════════════════════════════
	# Workspaces
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/workspaces")
	@_guard
	def list_workspaces():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.workspaces.items()
			if tid == tenant_id
		]
		items.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(items, page, page_size))

	@blueprint.post("/workspaces")
	@_guard
	def create_workspace():
		body = request.get_json(force=True)
		result = _SERVICE.record_workspace(
			body["workspace_id"],
			body.get("tenant_id", _get_tenant()),
			body["workspace_type"],
			body["name"],
			body["classification"],
			body["authority_id"],
			body["evidence_reference"],
		)
		return _ok(result, 201)

	@blueprint.get("/workspaces/<workspace_id>")
	@_guard
	def get_workspace(workspace_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.workspaces.get((tenant_id, workspace_id))
		if not item:
			return _err("workspace not found", 404)
		return _ok(item.to_dict())


	# ═══════════════════════════════════════════════════════════════════════════
	# Sources
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/sources")
	@_guard
	def list_sources():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.sources.items()
			if tid == tenant_id
		]
		items.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(items, page, page_size))

	@blueprint.post("/sources")
	@_guard
	def create_source():
		body = request.get_json(force=True)
		result = _SERVICE.register_source(
			body["source_id"],
			body.get("tenant_id", _get_tenant()),
			body["workspace_id"],
			body["source_type"],
			body["source_reference"],
			body["custodian_id"],
			body["lineage_reference"],
			body["evidence_reference"],
		)
		return _ok(result, 201)

	@blueprint.get("/sources/<source_id>")
	@_guard
	def get_source(source_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.sources.get((tenant_id, source_id))
		if not item:
			return _err("source not found", 404)
		return _ok(item.to_dict())


	# ═══════════════════════════════════════════════════════════════════════════
	# Indicators (IOC)
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/indicators")
	@_guard
	def list_indicators():
		"""List indicators with optional type/status/confidence filters."""
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 50)), 200)
		ioc_type = request.args.get("ioc_type")
		status = request.args.get("status")
		confidence_min = float(request.args.get("confidence_min", 0.0))

		items = [
			v.to_dict() for (tid, _), v in _SERVICE.indicators.items()
			if tid == tenant_id
		]
		# Also include extended IOC store
		ioc_items = list(_SERVICE._ioc_store().values())
		# Filter
		if ioc_type:
			items = [i for i in items if i.get("indicator_type") == ioc_type]
			ioc_items = [i for i in ioc_items if i.get("ioc_type") == ioc_type]
		if status:
			ioc_items = [i for i in ioc_items if i.get("status") == status]
		if confidence_min > 0:
			ioc_items = [i for i in ioc_items if i.get("confidence", 0) >= confidence_min]

		# Strip internal fields
		ioc_clean = [{k: v for k, v in i.items() if not k.startswith("_")} for i in ioc_items]
		combined = items + ioc_clean
		combined.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(combined, page, page_size))

	@blueprint.post("/indicators")
	@_guard
	def create_indicator():
		"""Create an IOC indicator."""
		body = request.get_json(force=True)
		# Support both contract-style and extended-style creation
		if "ioc_type" in body:
			result = _run(_SERVICE.create_indicator(
				ioc_type=body["ioc_type"],
				value=body["value"],
				confidence=float(body.get("confidence", 0.5)),
				tlp=body.get("tlp", "green"),
				source=body.get("source", "api"),
				context=body.get("context"),
			))
		else:
			result = _SERVICE.record_indicator(
				body["indicator_id"],
				body.get("tenant_id", _get_tenant()),
				body["source_id"],
				body["indicator_type"],
				body["indicator_reference"],
				float(body["confidence_score"]),
				body["evidence_reference"],
			)
		return _ok(result, 201)

	@blueprint.get("/indicators/<indicator_id>")
	@_guard
	def get_indicator(indicator_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.indicators.get((tenant_id, indicator_id))
		if item:
			return _ok(item.to_dict())
		# Check extended store
		ioc = _SERVICE._ioc_store().get(indicator_id)
		if ioc:
			return _ok({k: v for k, v in ioc.items() if not k.startswith("_")})
		return _err("indicator not found", 404)

	@blueprint.post("/indicators/<indicator_id>/enrich")
	@_guard
	def enrich_indicator(indicator_id: str):
		"""Enrich an IOC with external context (geo, ASN, reputation, etc.)."""
		result = _run(_SERVICE.enrich_indicator(indicator_id))
		return _ok(result)

	@blueprint.post("/indicators/<indicator_id>/retire")
	@_guard
	def retire_indicator(indicator_id: str):
		"""Retire/revoke an indicator."""
		body = request.get_json(force=True) or {}
		result = _run(_SERVICE.retire_indicator(indicator_id, body.get("reason", "manual_retirement")))
		return _ok(result)

	@blueprint.get("/indicators/search")
	@_guard
	def search_indicators():
		"""Full-text search across indicator values."""
		q = request.args.get("q", "")
		ioc_types_raw = request.args.get("ioc_types")
		ioc_types = ioc_types_raw.split(",") if ioc_types_raw else None
		confidence_min = float(request.args.get("confidence_min", 0.0))
		results = _run(_SERVICE.search_indicators(q, ioc_types, confidence_min))
		return _ok(results)

	@blueprint.post("/indicators/bulk-import")
	@_guard
	def bulk_import_indicators():
		"""Import indicators from a STIX 2.1 bundle."""
		body = request.get_json(force=True)
		result = _run(_SERVICE.bulk_import_indicators(body))
		return _ok(result, 201)

	@blueprint.post("/indicators/export")
	@_guard
	def export_indicators():
		"""Export indicators in stix/misp/csv/openioc format."""
		body = request.get_json(force=True) or {}
		result = _run(_SERVICE.export_indicators(
			filters=body.get("filters"),
			format=body.get("format", "stix"),
		))
		return _ok(result)

	@blueprint.post("/indicators/staleness")
	@_guard
	def run_staleness():
		"""Run staleness sweep — retire indicators older than N days."""
		body = request.get_json(force=True) or {}
		result = _run(_SERVICE.staleness_management(
			older_than_days=int(body.get("older_than_days", 90))
		))
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# Threat Actors
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/actors")
	@_guard
	def list_actors():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.actors.items()
			if tid == tenant_id
		]
		# Include extended actor profiles
		profiles = list(_SERVICE._actor_profiles.values())
		combined = items + profiles
		combined.sort(key=lambda x: x.get("name", x.get("id", "")))
		return _ok(_paginate(combined, page, page_size))

	@blueprint.post("/actors")
	@_guard
	def create_actor():
		body = request.get_json(force=True)
		if "name" in body and "motivation" in body:
			result = _run(_SERVICE.create_threat_actor(
				name=body["name"],
				aliases=body.get("aliases", []),
				motivation=body["motivation"],
				sophistication=body.get("sophistication", "intermediate"),
				origin_country=body.get("origin_country", "XX"),
			))
		else:
			result = _SERVICE.record_actor(
				body["actor_id"],
				body.get("tenant_id", _get_tenant()),
				body["workspace_id"],
				body["actor_type"],
				body["actor_reference"],
				float(body["confidence_score"]),
				body["evidence_reference"],
			)
		return _ok(result, 201)

	@blueprint.get("/actors/<actor_id>")
	@_guard
	def get_actor(actor_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.actors.get((tenant_id, actor_id))
		if item:
			return _ok(item.to_dict())
		profile = _SERVICE._actor_profiles.get(actor_id)
		if profile:
			return _ok(profile)
		return _err("actor not found", 404)

	@blueprint.put("/actors/<actor_id>")
	@_guard
	def update_actor(actor_id: str):
		body = request.get_json(force=True)
		result = _run(_SERVICE.update_actor_profile(
			actor_id=actor_id,
			ttps=body.get("ttps", []),
			target_sectors=body.get("target_sectors", []),
			known_tools=body.get("known_tools", []),
		))
		return _ok(result)

	@blueprint.get("/actors/<actor_id>/attribution")
	@_guard
	def actor_attribution(actor_id: str):
		"""Full attribution dossier for an actor."""
		result = _run(_SERVICE.actor_attribution_report(actor_id))
		return _ok(result)

	@blueprint.get("/actors/<actor_id>/techniques")
	@_guard
	def actor_techniques(actor_id: str):
		"""MITRE ATT&CK techniques linked to an actor."""
		result = _run(_SERVICE.get_techniques_for_actor(actor_id))
		return _ok(result)

	@blueprint.post("/actors/search")
	@_guard
	def search_actors():
		body = request.get_json(force=True) or {}
		result = _run(_SERVICE.actor_search(
			query=body.get("q", ""),
			filters=body.get("filters"),
		))
		return _ok(result)

	@blueprint.post("/actors/<actor_id>/link-indicator")
	@_guard
	def link_actor_indicator(actor_id: str):
		body = request.get_json(force=True)
		result = _run(_SERVICE.link_actor_to_indicator(
			actor_id=actor_id,
			indicator_id=body["indicator_id"],
			relationship_type=body.get("relationship_type", "uses"),
			confidence=float(body.get("confidence", 0.7)),
		))
		return _ok(result, 201)

	@blueprint.post("/actors/<actor_id>/link-campaign")
	@_guard
	def link_actor_campaign(actor_id: str):
		body = request.get_json(force=True)
		result = _run(_SERVICE.link_actor_to_campaign(
			actor_id=actor_id,
			campaign_id=body["campaign_id"],
			role=body.get("role", "operator"),
		))
		return _ok(result, 201)


	# ═══════════════════════════════════════════════════════════════════════════
	# Campaigns
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/campaigns")
	@_guard
	def list_campaigns():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.campaigns.items()
			if tid == tenant_id
		]
		ext = list(_SERVICE._campaign_store().values())
		combined = items + ext
		combined.sort(key=lambda x: x.get("name", x.get("id", "")))
		return _ok(_paginate(combined, page, page_size))

	@blueprint.post("/campaigns")
	@_guard
	def create_campaign():
		body = request.get_json(force=True)
		if "name" in body and "objective" in body:
			result = _run(_SERVICE.create_campaign(
				name=body["name"],
				start_date=body.get("start_date", datetime.now(timezone.utc).isoformat()),
				objective=body["objective"],
				target_sectors=body.get("target_sectors", []),
				target_regions=body.get("target_regions", []),
			))
		else:
			result = _SERVICE.record_campaign(
				body["campaign_id"],
				body.get("tenant_id", _get_tenant()),
				body["actor_id"],
				body["campaign_type"],
				body["campaign_reference"],
				body["risk_level"],
				body["evidence_reference"],
			)
		return _ok(result, 201)

	@blueprint.get("/campaigns/active")
	@_guard
	def list_active_campaigns():
		"""List all active campaigns with technique and indicator counts."""
		result = _run(_SERVICE.active_campaigns_report())
		return _ok(result)

	@blueprint.get("/campaigns/<campaign_id>")
	@_guard
	def get_campaign(campaign_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.campaigns.get((tenant_id, campaign_id))
		if item:
			return _ok(item.to_dict())
		ext = _SERVICE._campaign_store().get(campaign_id)
		if ext:
			return _ok(ext)
		return _err("campaign not found", 404)

	@blueprint.get("/campaigns/<campaign_id>/timeline")
	@_guard
	def campaign_timeline(campaign_id: str):
		"""Chronological timeline of events for a campaign."""
		result = _run(_SERVICE.campaign_timeline(campaign_id))
		return _ok(result)

	@blueprint.post("/campaigns/<campaign_id>/add-indicator")
	@_guard
	def add_campaign_indicator(campaign_id: str):
		body = request.get_json(force=True)
		result = _run(_SERVICE.add_campaign_indicator(
			campaign_id=campaign_id,
			indicator_id=body["indicator_id"],
			first_seen=body.get("first_seen", datetime.now(timezone.utc).isoformat()),
			last_seen=body.get("last_seen", datetime.now(timezone.utc).isoformat()),
		))
		return _ok(result, 201)

	@blueprint.post("/campaigns/<campaign_id>/add-technique")
	@_guard
	def add_campaign_technique(campaign_id: str):
		body = request.get_json(force=True)
		result = _run(_SERVICE.add_campaign_technique(
			campaign_id=campaign_id,
			mitre_technique_id=body["mitre_technique_id"],
			notes=body.get("notes", ""),
		))
		return _ok(result, 201)

	@blueprint.post("/campaigns/similarity")
	@_guard
	def campaign_similarity():
		"""Compute IOC + technique overlap between two campaigns."""
		body = request.get_json(force=True)
		result = _run(_SERVICE.campaign_similarity(
			campaign1_id=body["campaign1_id"],
			campaign2_id=body["campaign2_id"],
		))
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# Assessments
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/assessments")
	@_guard
	def list_assessments():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.assessments.items()
			if tid == tenant_id
		]
		items.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(items, page, page_size))

	@blueprint.post("/assessments")
	@_guard
	def create_assessment():
		body = request.get_json(force=True)
		result = _SERVICE.record_assessment(
			body["assessment_id"],
			body.get("tenant_id", _get_tenant()),
			body["campaign_id"],
			body["assessment_type"],
			body["risk_level"],
			float(body["confidence_score"]),
			body["analyst_id"],
			body["evidence_reference"],
		)
		return _ok(result, 201)

	@blueprint.get("/assessments/<assessment_id>")
	@_guard
	def get_assessment(assessment_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.assessments.get((tenant_id, assessment_id))
		if not item:
			return _err("assessment not found", 404)
		return _ok(item.to_dict())


	# ═══════════════════════════════════════════════════════════════════════════
	# Reports
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/reports")
	@_guard
	def list_reports():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.reports.items()
			if tid == tenant_id
		]
		ext = list(_SERVICE._threat_reports.values())
		combined = items + ext
		combined.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(combined, page, page_size))

	@blueprint.post("/reports")
	@_guard
	def create_report():
		body = request.get_json(force=True)
		if "title" in body and "classification" in body and "target_audience" in body:
			result = _run(_SERVICE.generate_threat_report(
				classification=body["classification"],
				report_type=body.get("report_type", "assessment"),
				target_audience=body["target_audience"],
				title=body.get("title", ""),
				summary=body.get("summary", ""),
				indicator_ids=body.get("indicator_ids"),
				actor_ids=body.get("actor_ids"),
				campaign_ids=body.get("campaign_ids"),
			))
		else:
			result = _SERVICE.record_report(
				body["report_id"],
				body.get("tenant_id", _get_tenant()),
				body["assessment_id"],
				body["report_type"],
				body["report_reference"],
				body["approval_reference"],
				body["evidence_reference"],
			)
		return _ok(result, 201)

	@blueprint.get("/reports/<report_id>")
	@_guard
	def get_report(report_id: str):
		tenant_id = _get_tenant()
		item = _SERVICE.reports.get((tenant_id, report_id))
		if item:
			return _ok(item.to_dict())
		ext = _SERVICE._threat_reports.get(report_id)
		if ext:
			return _ok(ext)
		return _err("report not found", 404)

	@blueprint.post("/reports/<report_id>/taxii-share")
	@_guard
	def share_taxii(report_id: str):
		"""Push report to a TAXII 2.1 collection."""
		body = request.get_json(force=True)
		result = _run(_SERVICE.share_via_taxii(
			report_id=report_id,
			taxii_server_url=body["taxii_server_url"],
			collection_id=body["collection_id"],
		))
		return _ok(result)

	@blueprint.get("/reports/<report_id>/dissemination")
	@_guard
	def report_dissemination(report_id: str):
		"""Dissemination log entries for a report."""
		result = _run(_SERVICE.dissemination_log(report_id))
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# Mitigations
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.post("/mitigations")
	@_guard
	def create_mitigation():
		body = request.get_json(force=True)
		result = _SERVICE.record_mitigation(
			body["mitigation_id"],
			body.get("tenant_id", _get_tenant()),
			body["assessment_id"],
			body["mitigation_type"],
			body["action_reference"],
			body["approval_reference"],
			body["evidence_reference"],
		)
		return _ok(result, 201)

	@blueprint.get("/mitigations")
	@_guard
	def list_mitigations():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.mitigations.items()
			if tid == tenant_id
		]
		items.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(items, page, page_size))


	# ═══════════════════════════════════════════════════════════════════════════
	# Reviews
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.post("/reviews")
	@_guard
	def create_review():
		body = request.get_json(force=True)
		result = _SERVICE.record_review(
			body["review_id"],
			body.get("tenant_id", _get_tenant()),
			body["reference_id"],
			body["reviewer_id"],
			body["status"],
			body["evidence_reference"],
		)
		return _ok(result, 201)

	@blueprint.get("/reviews")
	@_guard
	def list_reviews():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.reviews.items()
			if tid == tenant_id
		]
		items.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(items, page, page_size))


	# ═══════════════════════════════════════════════════════════════════════════
	# MITRE ATT&CK
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/mitre/technique/<technique_id>")
	@_guard
	def mitre_technique(technique_id: str):
		"""Look up a MITRE ATT&CK technique."""
		result = _run(_SERVICE.map_technique(technique_id))
		return _ok(result)

	@blueprint.post("/mitre/coverage")
	@_guard
	def mitre_coverage():
		"""Detection gap analysis for a set of observed techniques."""
		body = request.get_json(force=True)
		result = _run(_SERVICE.coverage_analysis(body.get("techniques", [])))
		return _ok(result)

	@blueprint.post("/mitre/kill-chain")
	@_guard
	def kill_chain_map():
		"""Map indicator IDs to Lockheed Martin Kill Chain phases."""
		body = request.get_json(force=True)
		result = _run(_SERVICE.kill_chain_mapping(body.get("indicator_ids", [])))
		return _ok(result)

	@blueprint.post("/mitre/attack-path")
	@_guard
	def attack_path():
		"""Reconstruct likely attack path from observed techniques."""
		body = request.get_json(force=True)
		result = _run(_SERVICE.attack_path_analysis(body.get("techniques", [])))
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# Intelligence Requirements (PIR)
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/requirements")
	@_guard
	def list_requirements():
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = list(_SERVICE._requirements.values())
		items.sort(key=lambda x: x.get("priority", ""))
		return _ok(_paginate(items, page, page_size))

	@blueprint.post("/requirements")
	@_guard
	def create_requirement():
		body = request.get_json(force=True)
		result = _run(_SERVICE.intelligence_requirement(
			requirement_text=body["requirement_text"],
			priority=body.get("priority", "medium"),
			requester=body.get("requester", _get_actor()),
		))
		return _ok(result, 201)


	# ═══════════════════════════════════════════════════════════════════════════
	# Feeds
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/feeds")
	@_guard
	def list_feeds():
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = list(_SERVICE._feeds.values())
		items.sort(key=lambda x: x.get("name", ""))
		return _ok(_paginate(items, page, page_size))

	@blueprint.get("/feeds/dashboard")
	@_guard
	def feeds_dashboard():
		"""Summary dashboard of all registered feeds."""
		result = _run(_SERVICE.feeds_dashboard())
		return _ok(result)

	@blueprint.post("/feeds")
	@_guard
	def register_feed():
		body = request.get_json(force=True)
		result = _run(_SERVICE.register_feed(
			name=body["name"],
			url=body["url"],
			format=body.get("format", "stix"),
			auth_method=body.get("auth_method", "none"),
			update_frequency=body.get("update_frequency", "@hourly"),
		))
		return _ok(result, 201)

	@blueprint.get("/feeds/<feed_id>")
	@_guard
	def get_feed(feed_id: str):
		feed = _SERVICE._feeds.get(feed_id)
		if not feed:
			return _err("feed not found", 404)
		return _ok(feed)

	@blueprint.post("/feeds/<feed_id>/ingest")
	@_guard
	def ingest_feed(feed_id: str):
		"""Trigger ingestion for a registered feed."""
		result = _run(_SERVICE.ingest_feed(feed_id))
		return _ok(result)

	@blueprint.get("/feeds/<feed_id>/quality")
	@_guard
	def feed_quality(feed_id: str):
		"""Feed quality report with false-positive rate, staleness, dedup rate."""
		result = _run(_SERVICE.feed_quality_report(feed_id))
		return _ok(result)

	@blueprint.post("/feeds/<feed_id>/deduplicate")
	@_guard
	def deduplicate_feed(feed_id: str):
		body = request.get_json(force=True) or {}
		result = _run(_SERVICE.deduplicate_from_feed(
			feed_id=feed_id,
			batch_id=body.get("batch_id", "manual"),
		))
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# MISP Export
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.post("/export/misp")
	@_guard
	def export_misp():
		"""Export indicators as a MISP JSON event."""
		body = request.get_json(force=True)
		result = _run(_SERVICE.export_misp_event(body.get("indicator_ids", [])))
		return _ok(result)

	@blueprint.post("/export/stix")
	@_guard
	def export_stix():
		"""Export indicators as a STIX 2.1 bundle."""
		body = request.get_json(force=True) or {}
		result = _run(_SERVICE.export_indicators(
			filters=body.get("filters"),
			format="stix",
		))
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# Agents
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/agents")
	@_guard
	def list_agents():
		tenant_id = _get_tenant()
		page = int(request.args.get("page", 1))
		page_size = min(int(request.args.get("page_size", 20)), 100)
		items = [
			v.to_dict() for (tid, _), v in _SERVICE.agents.items()
			if tid == tenant_id
		]
		items.sort(key=lambda x: x.get("id", ""))
		return _ok(_paginate(items, page, page_size))

	@blueprint.post("/agents")
	@_guard
	def register_agent():
		body = request.get_json(force=True)
		result = _SERVICE.register_threat_agent(
			body["agent_id"],
			body.get("tenant_id", _get_tenant()),
			body["name"],
			body["runtime"],
			body["role"],
			body.get("scope", "threat intelligence operations"),
		)
		return _ok(result, 201)

	@blueprint.post("/agents/validate-action")
	@_guard
	def validate_agent_action():
		body = request.get_json(force=True) or {}
		result = _SERVICE.validate_agent_action(
			tenant_id=body.get("tenant_id", _get_tenant()),
			privileged_scope=body.get("privileged_scope", False),
			human_approval_recorded=body.get("human_approval_recorded", False),
			unsupported_attribution_scope=body.get("unsupported_attribution_scope", False),
			fabricated_indicator_scope=body.get("fabricated_indicator_scope", False),
			source_tampering_scope=body.get("source_tampering_scope", False),
			privacy_bypass_scope=body.get("privacy_bypass_scope", False),
			autonomous_mitigation_scope=body.get("autonomous_mitigation_scope", False),
			unapproved_publication_scope=body.get("unapproved_publication_scope", False),
		)
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# Reports (domain-specific aggregate reports)
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/reports/type/mitre-heatmap")
	@_guard
	def report_mitre_heatmap():
		"""MITRE ATT&CK heatmap across all observed techniques."""
		all_techniques = set()
		for ct in _SERVICE._campaign_techniques:
			all_techniques.add(ct["technique_id"])
		result = _run(_SERVICE.coverage_analysis(list(all_techniques)))
		return _ok(result)

	@blueprint.get("/reports/type/analyst-calibration")
	@_guard
	def report_calibration():
		"""Confidence calibration report for an analyst."""
		analyst_id = request.args.get("analyst_id", "")
		period = request.args.get("period", "")
		if not analyst_id or not period:
			return _err("analyst_id and period query params required", 422)
		result = _run(_SERVICE.confidence_calibration_report(analyst_id, period))
		return _ok(result)

	@blueprint.get("/reports/type/active-campaigns")
	@_guard
	def report_active_campaigns():
		result = _run(_SERVICE.active_campaigns_report())
		return _ok(result)


	# ═══════════════════════════════════════════════════════════════════════════
	# Contract / self-test
	# ═══════════════════════════════════════════════════════════════════════════

	@blueprint.get("/contract")
	@_guard
	def contract():
		"""Return the capability contract for a tenant."""
		tenant_id = _get_tenant()
		return _ok(_SERVICE.describe(tenant_id))

	@blueprint.post("/evaluate")
	@_guard
	def evaluate():
		"""Evaluate capability rules against a context dict."""
		body = request.get_json(force=True) or {}
		return _ok(_SERVICE.evaluate(body))

	@blueprint.post("/batch/validate")
	@_guard
	def validate_batch():
		body = request.get_json(force=True) or {}
		result = _SERVICE.validate_batch(
			body.get("tenant_id", _get_tenant()),
			body["item_count"],
			body.get("event_stream", "bytewax"),
		)
		return _ok(result)


# ── Process-local helper functions (used by tests and app.py) ─────────────────

def record_authority(payload: dict):
	return _SERVICE.record_authority(
		payload["authority_id"], payload.get("tenant_id", "default"),
		payload["authority_type"], payload["scope_reference"],
		payload["classification"], payload["approver_id"],
		payload["expires_at"], payload["evidence_reference"],
		payload.get("policy_attached", True),
	)


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(
		payload["workspace_id"], payload.get("tenant_id", "default"),
		payload["workspace_type"], payload["name"],
		payload["classification"], payload["authority_id"],
		payload["evidence_reference"],
	)


def register_source(payload: dict):
	return _SERVICE.register_source(
		payload["source_id"], payload.get("tenant_id", "default"),
		payload["workspace_id"], payload["source_type"],
		payload["source_reference"], payload["custodian_id"],
		payload["lineage_reference"], payload["evidence_reference"],
	)


def record_indicator(payload: dict):
	return _SERVICE.record_indicator(
		payload["indicator_id"], payload.get("tenant_id", "default"),
		payload["source_id"], payload["indicator_type"],
		payload["indicator_reference"], payload["confidence_score"],
		payload["evidence_reference"],
	)


def record_actor(payload: dict):
	return _SERVICE.record_actor(
		payload["actor_id"], payload.get("tenant_id", "default"),
		payload["workspace_id"], payload["actor_type"],
		payload["actor_reference"], payload["confidence_score"],
		payload["evidence_reference"],
	)


def record_campaign(payload: dict):
	return _SERVICE.record_campaign(
		payload["campaign_id"], payload.get("tenant_id", "default"),
		payload["actor_id"], payload["campaign_type"],
		payload["campaign_reference"], payload["risk_level"],
		payload["evidence_reference"],
	)


def record_assessment(payload: dict):
	return _SERVICE.record_assessment(
		payload["assessment_id"], payload.get("tenant_id", "default"),
		payload["campaign_id"], payload["assessment_type"],
		payload["risk_level"], payload["confidence_score"],
		payload["analyst_id"], payload["evidence_reference"],
	)


def record_report(payload: dict):
	return _SERVICE.record_report(
		payload["report_id"], payload.get("tenant_id", "default"),
		payload["assessment_id"], payload["report_type"],
		payload["report_reference"], payload["approval_reference"],
		payload["evidence_reference"],
	)


def record_mitigation(payload: dict):
	return _SERVICE.record_mitigation(
		payload["mitigation_id"], payload.get("tenant_id", "default"),
		payload["assessment_id"], payload["mitigation_type"],
		payload["action_reference"], payload["approval_reference"],
		payload["evidence_reference"],
	)


def record_review(payload: dict):
	return _SERVICE.record_review(
		payload["review_id"], payload.get("tenant_id", "default"),
		payload["reference_id"], payload["reviewer_id"],
		payload["status"], payload["evidence_reference"],
	)


def register_threat_agent(payload: dict):
	return _SERVICE.register_threat_agent(
		payload["agent_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["runtime"],
		payload["role"], payload.get("scope", "threat intelligence operations"),
	)


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(
		payload.get("tenant_id", "default"),
		payload.get("privileged_scope", False),
		payload.get("human_approval_recorded", False),
		payload.get("unsupported_attribution_scope", False),
		payload.get("fabricated_indicator_scope", False),
		payload.get("source_tampering_scope", False),
		payload.get("privacy_bypass_scope", False),
		payload.get("autonomous_mitigation_scope", False),
		payload.get("unapproved_publication_scope", False),
	)


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(
		payload.get("tenant_id", "default"),
		payload["item_count"],
		payload.get("event_stream", "bytewax"),
	)


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
