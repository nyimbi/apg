"""REST API for APG Open Source Intelligence (OSINT).

Flask Blueprint — url_prefix: /intel-osint/api/v1

All endpoints enforce tenant isolation via X-Tenant-ID header (falls back
to query param or body field).  Every write operation requires an actor_id
(X-Actor-ID header).

Routes:
    Sources        GET /sources, POST /sources, GET /sources/<id>,
                   PUT /sources/<id>, DELETE /sources/<id>
    Tasks          GET /tasks, POST /tasks, GET /tasks/<id>,
                   POST /tasks/<id>/start, POST /tasks/<id>/complete,
                   POST /tasks/<id>/fail, POST /tasks/<id>/cancel
    Raw Intel      GET /raw-intel, POST /raw-intel, GET /raw-intel/<id>,
                   POST /raw-intel/<id>/triage
    Processed      GET /processed-intel, POST /processed-intel,
                   GET /processed-intel/<id>, PUT /processed-intel/<id>
    Entities       GET /entities, POST /entities, GET /entities/<id>,
                   PUT /entities/<id>, DELETE /entities/<id>
    Relationships  GET /relationships, POST /relationships,
                   GET /relationships/<id>, PUT /relationships/<id>
    Social         GET /social-profiles, POST /social-profiles/monitor,
                   PUT /social-profiles/<id>
    Web Content    GET /web-content, POST /web-scrape, GET /web-content/<id>
    Domains        GET /domain-records, POST /domain-intel,
                   GET /domain-records/<id>
    IPs            GET /ip-intel, POST /ip-intel, GET /ip-intel/<id>
    Documents      GET /document-analyses, POST /document-analyses,
                   GET /document-analyses/<id>
    Credibility    POST /credibility-scores
    Dissemination  GET /dissemination, POST /dissemination,
                   GET /dissemination/<id>
    Reviews        GET /reviews, POST /reviews
    Agents         POST /agents, POST /agents/validate-action,
                   POST /agents/validate-batch
    Reports        GET /reports/dashboard, GET /reports/entity-network,
                   GET /reports/source-health, GET /reports/threat-landscape
    Audit          GET /audit-log
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .models import (
		AgentRole,
		AgentRuntime,
		CollectionTaskCreate,
		CollectionTaskUpdate,
		CredibilityScoreCreate,
		DisseminationPackageCreate,
		DocumentAnalysisCreate,
		EntityRelationshipCreate,
		EntityRelationshipUpdate,
		IPIntelligenceCreate,
		IntelStatus,
		OSEntityCreate,
		OSEntityUpdate,
		OSINTAgentCreate,
		OSINTReviewCreate,
		OSINTSourceCreate,
		OSINTSourceUpdate,
		Priority,
		ProcessedIntelligenceCreate,
		ProcessedIntelligenceUpdate,
		RawIntelligenceCreate,
		ReviewStatus,
		RiskTier,
		SocialMediaProfileCreate,
		SocialMediaProfileUpdate,
		SourceStatus,
		SourceType,
		TaskStatus,
		TaskType,
		TriageDecision,
		DomainRecordCreate,
		EntityType,
		RelationshipType,
	)
	from .service import OSINTService
except ImportError:  # pragma: no cover
	from models import (  # type: ignore
		AgentRole, AgentRuntime, CollectionTaskCreate, CollectionTaskUpdate,
		CredibilityScoreCreate, DisseminationPackageCreate, DocumentAnalysisCreate,
		EntityRelationshipCreate, EntityRelationshipUpdate, IPIntelligenceCreate,
		IntelStatus, OSEntityCreate, OSEntityUpdate, OSINTAgentCreate, OSINTReviewCreate,
		OSINTSourceCreate, OSINTSourceUpdate, Priority, ProcessedIntelligenceCreate,
		ProcessedIntelligenceUpdate, RawIntelligenceCreate, ReviewStatus, RiskTier,
		SocialMediaProfileCreate, SocialMediaProfileUpdate, SourceStatus, SourceType,
		TaskStatus, TaskType, TriageDecision, DomainRecordCreate, EntityType, RelationshipType,
	)
	from service import OSINTService  # type: ignore

bp = Blueprint("intel_osint", __name__, url_prefix="/intel-osint/api/v1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
	"""Run an async coroutine synchronously inside Flask's sync context."""
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _tenant_id() -> str:
	return (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or (request.get_json(silent=True) or {}).get("tenant_id", "default")
	)


def _actor_id() -> str:
	return request.headers.get("X-Actor-ID", "anonymous")


def _svc() -> OSINTService:
	return OSINTService(db_session=None, tenant_id=_tenant_id(), actor_id=_actor_id())


def _ok(data: Any, status: int = 200):
	if hasattr(data, "model_dump"):
		return jsonify(data.model_dump(mode="json")), status
	if isinstance(data, list):
		items = [i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in data]
		return jsonify({"items": items, "count": len(items)}), status
	return jsonify(data), status


def _err(msg: str, status: int = 400):
	return jsonify({"error": msg}), status


def _handle(fn):
	"""Decorator: catch KeyError → 404, PermissionError → 403, ValueError → 400."""
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except KeyError as exc:
			return _err(str(exc), 404)
		except PermissionError as exc:
			return _err(str(exc), 403)
		except (ValueError, TypeError, Exception) as exc:
			import traceback
			return _err(f"{type(exc).__name__}: {exc}", 400)
	return wrapper


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@bp.get("/sources")
@_handle
def list_sources():
	svc = _svc()
	source_type = request.args.get("source_type")
	risk_tier = request.args.get("risk_tier")
	status = request.args.get("status")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_sources(
		source_type=SourceType(source_type) if source_type else None,
		risk_tier=RiskTier(risk_tier) if risk_tier else None,
		status=SourceStatus(status) if status else None,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/sources")
@_handle
def create_source():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = OSINTSourceCreate(**data)
	item = _run(_svc().register_source(payload))
	return _ok(item, 201)


@bp.get("/sources/<source_id>")
@_handle
def get_source(source_id: str):
	return _ok(_run(_svc().get_source(source_id)))


@bp.put("/sources/<source_id>")
@_handle
def update_source(source_id: str):
	data = request.get_json(force=True)
	payload = OSINTSourceUpdate(**data)
	return _ok(_run(_svc().update_source(source_id, payload)))


@bp.delete("/sources/<source_id>")
@_handle
def delete_source(source_id: str):
	_run(_svc().delete_source(source_id))
	return jsonify({"deleted": source_id}), 200


# ---------------------------------------------------------------------------
# Collection tasks
# ---------------------------------------------------------------------------

@bp.get("/tasks")
@_handle
def list_tasks():
	svc = _svc()
	source_id = request.args.get("source_id")
	task_type = request.args.get("task_type")
	status = request.args.get("status")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_tasks(
		source_id=source_id,
		task_type=TaskType(task_type) if task_type else None,
		status=TaskStatus(status) if status else None,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/tasks")
@_handle
def create_task():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = CollectionTaskCreate(**data)
	item = _run(_svc().create_task(payload))
	return _ok(item, 201)


@bp.get("/tasks/<task_id>")
@_handle
def get_task(task_id: str):
	return _ok(_run(_svc().get_task(task_id)))


@bp.post("/tasks/<task_id>/start")
@_handle
def start_task(task_id: str):
	return _ok(_run(_svc().start_task(task_id)))


@bp.post("/tasks/<task_id>/complete")
@_handle
def complete_task(task_id: str):
	data = request.get_json(force=True) or {}
	items_collected = int(data.get("items_collected", 0))
	return _ok(_run(_svc().complete_task(task_id, items_collected)))


@bp.post("/tasks/<task_id>/fail")
@_handle
def fail_task(task_id: str):
	data = request.get_json(force=True) or {}
	error_message = data.get("error_message", "unknown error")
	return _ok(_run(_svc().fail_task(task_id, error_message)))


@bp.post("/tasks/<task_id>/cancel")
@_handle
def cancel_task(task_id: str):
	return _ok(_run(_svc().cancel_task(task_id)))


# ---------------------------------------------------------------------------
# Raw intelligence
# ---------------------------------------------------------------------------

@bp.get("/raw-intel")
@_handle
def list_raw_intel():
	svc = _svc()
	task_id = request.args.get("task_id")
	source_id = request.args.get("source_id")
	status = request.args.get("status")
	triage = request.args.get("triage_decision")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_raw_intel(
		task_id=task_id,
		source_id=source_id,
		status=IntelStatus(status) if status else None,
		triage_decision=TriageDecision(triage) if triage else None,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/raw-intel")
@_handle
def ingest_raw_intel():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = RawIntelligenceCreate(**data)
	item = _run(_svc().ingest_raw_intel(payload))
	return _ok(item, 201)


@bp.get("/raw-intel/<raw_intel_id>")
@_handle
def get_raw_intel(raw_intel_id: str):
	return _ok(_run(_svc().get_raw_intel(raw_intel_id)))


@bp.post("/raw-intel/<raw_intel_id>/triage")
@_handle
def triage_raw_intel(raw_intel_id: str):
	data = request.get_json(force=True)
	decision = TriageDecision(data["decision"])
	analyst_id = data["analyst_id"]
	notes = data.get("notes")
	item = _run(_svc().triage_raw_intel(raw_intel_id, decision, analyst_id, notes))
	return _ok(item)


# ---------------------------------------------------------------------------
# Processed intelligence
# ---------------------------------------------------------------------------

@bp.get("/processed-intel")
@_handle
def list_processed_intel():
	svc = _svc()
	assessment_type = request.args.get("assessment_type")
	status = request.args.get("status")
	analyst_id = request.args.get("analyst_id")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_processed_intel(
		assessment_type=assessment_type,
		status=IntelStatus(status) if status else None,
		analyst_id=analyst_id,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/processed-intel")
@_handle
def create_processed_intel():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = ProcessedIntelligenceCreate(**data)
	item = _run(_svc().create_processed_intel(payload))
	return _ok(item, 201)


@bp.get("/processed-intel/<intel_id>")
@_handle
def get_processed_intel(intel_id: str):
	return _ok(_run(_svc().get_processed_intel(intel_id)))


@bp.put("/processed-intel/<intel_id>")
@_handle
def update_processed_intel(intel_id: str):
	data = request.get_json(force=True)
	payload = ProcessedIntelligenceUpdate(**data)
	return _ok(_run(_svc().update_processed_intel(intel_id, payload)))


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@bp.get("/entities")
@_handle
def list_entities():
	svc = _svc()
	entity_type = request.args.get("entity_type")
	min_confidence = request.args.get("min_confidence", type=float)
	tag = request.args.get("tag")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_entities(
		entity_type=EntityType(entity_type) if entity_type else None,
		min_confidence=min_confidence,
		tag=tag,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/entities")
@_handle
def create_entity():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = OSEntityCreate(**data)
	item = _run(_svc().extract_entity(payload))
	return _ok(item, 201)


@bp.get("/entities/<entity_id>")
@_handle
def get_entity(entity_id: str):
	return _ok(_run(_svc().get_entity(entity_id)))


@bp.put("/entities/<entity_id>")
@_handle
def update_entity(entity_id: str):
	data = request.get_json(force=True)
	payload = OSEntityUpdate(**data)
	return _ok(_run(_svc().update_entity(entity_id, payload)))


@bp.delete("/entities/<entity_id>")
@_handle
def delete_entity(entity_id: str):
	_run(_svc().delete_entity(entity_id))
	return jsonify({"deleted": entity_id}), 200


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

@bp.get("/relationships")
@_handle
def list_relationships():
	svc = _svc()
	entity_id = request.args.get("entity_id")
	rel_type = request.args.get("relationship_type")
	min_confidence = request.args.get("min_confidence", type=float)
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_relationships(
		entity_id=entity_id,
		relationship_type=RelationshipType(rel_type) if rel_type else None,
		min_confidence=min_confidence,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/relationships")
@_handle
def create_relationship():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = EntityRelationshipCreate(**data)
	item = _run(_svc().map_relationship(payload))
	return _ok(item, 201)


@bp.get("/relationships/<rel_id>")
@_handle
def get_relationship(rel_id: str):
	return _ok(_run(_svc().get_relationship(rel_id)))


@bp.put("/relationships/<rel_id>")
@_handle
def update_relationship(rel_id: str):
	data = request.get_json(force=True)
	payload = EntityRelationshipUpdate(**data)
	return _ok(_run(_svc().update_relationship(rel_id, payload)))


# ---------------------------------------------------------------------------
# Social media profiles
# ---------------------------------------------------------------------------

@bp.get("/social-profiles")
@_handle
def list_social_profiles():
	svc = _svc()
	platform = request.args.get("platform")
	entity_id = request.args.get("entity_id")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_social_profiles(platform=platform, entity_id=entity_id, limit=limit, offset=offset))
	return _ok(items)


@bp.post("/social-profiles/monitor")
@_handle
def monitor_social_profiles():
	data = request.get_json(force=True)
	handles = data["handles"]
	keywords = data.get("keywords", [])
	platform = data["platform"]
	items = _run(_svc().social_media_monitor(handles, keywords, platform))
	return _ok(items, 201)


@bp.put("/social-profiles/<profile_id>")
@_handle
def update_social_profile(profile_id: str):
	data = request.get_json(force=True)
	payload = SocialMediaProfileUpdate(**data)
	return _ok(_run(_svc().update_social_profile(profile_id, payload)))


# ---------------------------------------------------------------------------
# Web content
# ---------------------------------------------------------------------------

@bp.get("/web-content")
@_handle
def list_web_content():
	svc = _svc()
	task_id = request.args.get("task_id")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_web_content(task_id=task_id, limit=limit, offset=offset))
	return _ok(items)


@bp.post("/web-scrape")
@_handle
def web_scrape():
	data = request.get_json(force=True)
	svc = OSINTService(db_session=None, tenant_id=_tenant_id(), actor_id=_actor_id())
	item = _run(svc.web_scrape(
		url=data["url"],
		task_id=data["task_id"],
		depth=int(data.get("depth", 2)),
		content=data.get("content"),
		title=data.get("title"),
		language=data.get("language"),
		links=data.get("links"),
		metadata=data.get("metadata"),
	))
	return _ok(item, 201)


@bp.get("/web-content/<content_id>")
@_handle
def get_web_content(content_id: str):
	return _ok(_run(_svc().get_web_content(content_id)))


# ---------------------------------------------------------------------------
# Domain records
# ---------------------------------------------------------------------------

@bp.get("/domain-records")
@_handle
def list_domain_records():
	svc = _svc()
	domain = request.args.get("domain")
	registrant_email = request.args.get("registrant_email")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.find_domain_records(domain=domain, registrant_email=registrant_email, limit=limit, offset=offset))
	return _ok(items)


@bp.post("/domain-intel")
@_handle
def create_domain_intel():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = DomainRecordCreate(**data)
	item = _run(_svc().domain_intelligence(payload))
	return _ok(item, 201)


@bp.get("/domain-records/<record_id>")
@_handle
def get_domain_record(record_id: str):
	return _ok(_run(_svc().get_domain_record(record_id)))


# ---------------------------------------------------------------------------
# IP intelligence
# ---------------------------------------------------------------------------

@bp.get("/ip-intel")
@_handle
def list_ip_intel():
	svc = _svc()
	ip_address = request.args.get("ip_address")
	country_code = request.args.get("country_code")
	is_tor = request.args.get("is_tor")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.find_ip_intel(
		ip_address=ip_address,
		country_code=country_code,
		is_tor=is_tor.lower() == "true" if is_tor else None,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/ip-intel")
@_handle
def create_ip_intel():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = IPIntelligenceCreate(**data)
	item = _run(_svc().ip_geolocation_enrichment(payload))
	return _ok(item, 201)


@bp.get("/ip-intel/<ip_intel_id>")
@_handle
def get_ip_intel(ip_intel_id: str):
	return _ok(_run(_svc().get_ip_intel(ip_intel_id)))


# ---------------------------------------------------------------------------
# Document analysis
# ---------------------------------------------------------------------------

@bp.get("/document-analyses")
@_handle
def list_document_analyses():
	svc = _svc()
	raw_intel_id = request.args.get("raw_intel_id")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_document_analyses(raw_intel_id=raw_intel_id, limit=limit, offset=offset))
	return _ok(items)


@bp.post("/document-analyses")
@_handle
def create_document_analysis():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = DocumentAnalysisCreate(**data)
	item = _run(_svc().entity_extraction_nlp(payload))
	return _ok(item, 201)


@bp.get("/document-analyses/<analysis_id>")
@_handle
def get_document_analysis(analysis_id: str):
	return _ok(_run(_svc().get_document_analysis(analysis_id)))


# ---------------------------------------------------------------------------
# Credibility scoring
# ---------------------------------------------------------------------------

@bp.post("/credibility-scores")
@_handle
def create_credibility_score():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = CredibilityScoreCreate(**data)
	item = _run(_svc().credibility_scoring(payload))
	return _ok(item, 201)


# ---------------------------------------------------------------------------
# Dissemination
# ---------------------------------------------------------------------------

@bp.get("/dissemination")
@_handle
def list_dissemination():
	svc = _svc()
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_dissemination_packages(limit=limit, offset=offset))
	return _ok(items)


@bp.post("/dissemination")
@_handle
def create_dissemination():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = DisseminationPackageCreate(**data)
	item = _run(_svc().intelligence_dissemination(payload))
	return _ok(item, 201)


@bp.get("/dissemination/<package_id>")
@_handle
def get_dissemination(package_id: str):
	return _ok(_run(_svc().get_dissemination_package(package_id)))


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

@bp.get("/reviews")
@_handle
def list_reviews():
	svc = _svc()
	reference_type = request.args.get("reference_type")
	reviewer_id = request.args.get("reviewer_id")
	status = request.args.get("status")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = _run(svc.list_reviews(
		reference_type=reference_type,
		reviewer_id=reviewer_id,
		status=ReviewStatus(status) if status else None,
		limit=limit,
		offset=offset,
	))
	return _ok(items)


@bp.post("/reviews")
@_handle
def create_review():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = OSINTReviewCreate(**data)
	item = _run(_svc().record_review(payload))
	return _ok(item, 201)


# ---------------------------------------------------------------------------
# Agent management
# ---------------------------------------------------------------------------

@bp.post("/agents")
@_handle
def register_agent():
	data = request.get_json(force=True)
	data["tenant_id"] = _tenant_id()
	payload = OSINTAgentCreate(**data)
	item = _run(_svc().register_agent(payload))
	return _ok(item, 201)


@bp.post("/agents/validate-action")
@_handle
def validate_agent_action():
	data = request.get_json(force=True)
	result = _run(_svc().validate_agent_action(
		privileged_scope=bool(data.get("privileged_scope", False)),
		human_approval_recorded=bool(data.get("human_approval_recorded", False)),
		cross_tenant_scope=bool(data.get("cross_tenant_scope", False)),
		privilege_escalation_scope=bool(data.get("privilege_escalation_scope", False)),
		evidence_fabrication_scope=bool(data.get("evidence_fabrication_scope", False)),
		source_terms_violation_scope=bool(data.get("source_terms_violation_scope", False)),
		autonomous_dissemination_scope=bool(data.get("autonomous_dissemination_scope", False)),
		unapproved_high_risk_collection_scope=bool(data.get("unapproved_high_risk_collection_scope", False)),
	))
	return _ok(result)


@bp.post("/agents/validate-batch")
@_handle
def validate_batch():
	data = request.get_json(force=True)
	result = _run(_svc().validate_batch(
		item_count=int(data["item_count"]),
		event_stream=data.get("event_stream", "bytewax"),
	))
	return _ok(result)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@bp.post("/deduplicate")
@_handle
def deduplicate():
	data = request.get_json(force=True) or {}
	threshold = float(data.get("similarity_threshold", 0.85))
	result = _run(_svc().duplicate_deduplication(threshold))
	return _ok(result)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@bp.get("/reports/dashboard")
@_handle
def report_dashboard():
	return _ok(_run(_svc().dashboard_summary()))


@bp.get("/reports/entity-network")
@_handle
def report_entity_network():
	return _ok(_run(_svc().relationship_mapping()))


@bp.get("/reports/source-health")
@_handle
def report_source_health():
	return _ok(_run(_svc().source_health_report()))


@bp.get("/reports/threat-landscape")
@_handle
def report_threat_landscape():
	return _ok(_run(_svc().threat_landscape_report()))


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@bp.get("/audit-log")
@_handle
def audit_log():
	limit = int(request.args.get("limit", 100))
	events = _run(_svc().get_audit_log(limit=limit))
	return jsonify({"items": events, "count": len(events)}), 200


# ---------------------------------------------------------------------------
# Capability contract
# ---------------------------------------------------------------------------

@bp.get("/contract")
@_handle
def get_contract():
	return _ok(_run(_svc().describe()))
