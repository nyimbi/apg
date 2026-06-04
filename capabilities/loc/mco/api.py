"""Process-local API helpers and Flask Blueprint REST endpoints for APG Multi-Country Operations."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import MultiCountryOperationsService
	from .models import (
		ComplianceMappingCreate,
		ComplianceMappingUpdate,
		CountryCreate,
		CountryUpdate,
		EntityCreate,
		EntityUpdate,
		IntercompanyTransactionCreate,
		McoAgentCreate,
		StatutoryReportCreate,
		StatutoryReportUpdate,
	)
except ImportError:  # pragma: no cover
	from service import MultiCountryOperationsService  # type: ignore[no-redef]
	from models import (  # type: ignore[no-redef]
		ComplianceMappingCreate,
		ComplianceMappingUpdate,
		CountryCreate,
		CountryUpdate,
		EntityCreate,
		EntityUpdate,
		IntercompanyTransactionCreate,
		McoAgentCreate,
		StatutoryReportCreate,
		StatutoryReportUpdate,
	)

_SERVICE = MultiCountryOperationsService()

# --- REST Blueprint ---
mco_api = Blueprint("loc_mco_api", __name__, url_prefix="/loc-mco/api/v1")


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _actor() -> str:
	return request.headers.get("X-Actor-ID", "system")


def _ok(data: Any, status: int = 200):
	if hasattr(data, "model_dump"):
		return jsonify(data.model_dump(mode="json")), status
	if isinstance(data, list):
		return jsonify([i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in data]), status
	return jsonify(data), status


def _err(msg: str, status: int = 400):
	return jsonify({"error": msg}), status


# ---- Countries ----

@mco_api.get("/countries")
def api_list_countries():
	"""
	List countries.
	---
	GET /loc-mco/api/v1/countries
	Query: tenant_id, status
	Permission: loc_mco:countries
	"""
	try:
		result = _run(_SERVICE.list_countries(_tenant(), status=request.args.get("status")))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/countries")
def api_create_country():
	"""
	Register a country.
	---
	POST /loc-mco/api/v1/countries
	Body: CountryCreate JSON
	Permission: loc_mco:countries_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		payload = CountryCreate.model_validate(body)
		result = _run(_SERVICE.register_country(payload, actor_id=_actor()))
		return _ok(result, 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.get("/countries/<country_id>")
def api_get_country(country_id: str):
	"""
	Get a country by ID.
	---
	GET /loc-mco/api/v1/countries/<country_id>
	Permission: loc_mco:countries
	"""
	try:
		result = _run(_SERVICE.get_country(_tenant(), country_id))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.put("/countries/<country_id>")
def api_update_country(country_id: str):
	"""
	Update a country record.
	---
	PUT /loc-mco/api/v1/countries/<country_id>
	Body: CountryUpdate JSON
	Permission: loc_mco:countries_write
	"""
	body = request.get_json(silent=True) or {}
	try:
		payload = CountryUpdate.model_validate(body)
		result = _run(_SERVICE.update_country(_tenant(), country_id, payload, actor_id=_actor()))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Entities ----

@mco_api.get("/entities")
def api_list_entities():
	"""
	List legal entities.
	---
	GET /loc-mco/api/v1/entities
	Query: tenant_id, country_id, entity_type, is_active
	Permission: loc_mco:entities
	"""
	is_active_raw = request.args.get("is_active")
	is_active = None if is_active_raw is None else is_active_raw.lower() == "true"
	try:
		result = _run(_SERVICE.list_entities(
			_tenant(),
			country_id=request.args.get("country_id"),
			entity_type=request.args.get("entity_type"),
			is_active=is_active,
		))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/entities")
def api_create_entity():
	"""
	Register a legal entity.
	---
	POST /loc-mco/api/v1/entities
	Body: EntityCreate JSON
	Permission: loc_mco:entities_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		payload = EntityCreate.model_validate(body)
		result = _run(_SERVICE.register_entity(payload, actor_id=_actor()))
		return _ok(result, 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.get("/entities/<entity_id>")
def api_get_entity(entity_id: str):
	"""
	Get a legal entity.
	---
	GET /loc-mco/api/v1/entities/<entity_id>
	Permission: loc_mco:entities
	"""
	try:
		result = _run(_SERVICE.get_entity(_tenant(), entity_id))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.put("/entities/<entity_id>")
def api_update_entity(entity_id: str):
	"""
	Update a legal entity.
	---
	PUT /loc-mco/api/v1/entities/<entity_id>
	Body: EntityUpdate JSON
	Permission: loc_mco:entities_write
	"""
	body = request.get_json(silent=True) or {}
	try:
		payload = EntityUpdate.model_validate(body)
		result = _run(_SERVICE.update_entity(_tenant(), entity_id, payload, actor_id=_actor()))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Compliance ----

@mco_api.get("/compliance")
def api_list_compliance():
	"""
	List compliance mappings.
	---
	GET /loc-mco/api/v1/compliance
	Query: entity_id, domain, status
	Permission: loc_mco:compliance
	"""
	try:
		result = _run(_SERVICE.list_compliance_mappings(
			_tenant(),
			entity_id=request.args.get("entity_id"),
			domain=request.args.get("domain"),
			status=request.args.get("status"),
		))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/compliance")
def api_create_compliance():
	"""
	Record a compliance mapping.
	---
	POST /loc-mco/api/v1/compliance
	Body: ComplianceMappingCreate JSON
	Permission: loc_mco:compliance_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		payload = ComplianceMappingCreate.model_validate(body)
		result = _run(_SERVICE.record_compliance_mapping(payload, actor_id=_actor()))
		return _ok(result, 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.get("/compliance/<mapping_id>")
def api_get_compliance(mapping_id: str):
	"""
	Get a compliance mapping.
	---
	GET /loc-mco/api/v1/compliance/<mapping_id>
	Permission: loc_mco:compliance
	"""
	try:
		result = _run(_SERVICE.get_compliance_mapping(_tenant(), mapping_id))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.put("/compliance/<mapping_id>")
def api_update_compliance(mapping_id: str):
	"""
	Update a compliance mapping.
	---
	PUT /loc-mco/api/v1/compliance/<mapping_id>
	Body: ComplianceMappingUpdate JSON
	Permission: loc_mco:compliance_write
	"""
	body = request.get_json(silent=True) or {}
	try:
		payload = ComplianceMappingUpdate.model_validate(body)
		result = _run(_SERVICE.update_compliance_mapping(_tenant(), mapping_id, payload, actor_id=_actor()))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Intercompany ----

@mco_api.get("/intercompany")
def api_list_intercompany():
	"""
	List intercompany transactions.
	---
	GET /loc-mco/api/v1/intercompany
	Query: entity_id, type, status
	Permission: loc_mco:intercompany
	"""
	try:
		result = _run(_SERVICE.list_intercompany_transactions(
			_tenant(),
			entity_id=request.args.get("entity_id"),
			txn_type=request.args.get("type"),
			status=request.args.get("status"),
		))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/intercompany")
def api_create_intercompany():
	"""
	Create an intercompany transaction.
	---
	POST /loc-mco/api/v1/intercompany
	Body: IntercompanyTransactionCreate JSON
	Permission: loc_mco:intercompany_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		payload = IntercompanyTransactionCreate.model_validate(body)
		result = _run(_SERVICE.create_intercompany_transaction(payload, actor_id=_actor()))
		return _ok(result, 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.get("/intercompany/<txn_id>")
def api_get_intercompany(txn_id: str):
	"""
	Get an intercompany transaction.
	---
	GET /loc-mco/api/v1/intercompany/<txn_id>
	Permission: loc_mco:intercompany
	"""
	try:
		result = _run(_SERVICE.get_intercompany_transaction(_tenant(), txn_id))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/intercompany/<txn_id>/approve")
def api_approve_intercompany(txn_id: str):
	"""
	Approve an intercompany transaction.
	---
	POST /loc-mco/api/v1/intercompany/<txn_id>/approve
	Body: {approver_id, approval_reference}
	Permission: loc_mco:intercompany_write
	"""
	body = request.get_json(silent=True) or {}
	try:
		result = _run(_SERVICE.approve_intercompany_transaction(
			_tenant(), txn_id,
			approver_id=body.get("approver_id", _actor()),
			approval_reference=body["approval_reference"],
		))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Statutory Reports ----

@mco_api.get("/statutory-reports")
def api_list_statutory_reports():
	"""
	List statutory reports.
	---
	GET /loc-mco/api/v1/statutory-reports
	Query: entity_id, report_type, status
	Permission: loc_mco:statutory_reports
	"""
	try:
		result = _run(_SERVICE.list_statutory_reports(
			_tenant(),
			entity_id=request.args.get("entity_id"),
			report_type=request.args.get("report_type"),
			status=request.args.get("status"),
		))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/statutory-reports")
def api_create_statutory_report():
	"""
	Create a statutory report.
	---
	POST /loc-mco/api/v1/statutory-reports
	Body: StatutoryReportCreate JSON
	Permission: loc_mco:statutory_reports_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		payload = StatutoryReportCreate.model_validate(body)
		result = _run(_SERVICE.create_statutory_report(payload, actor_id=_actor()))
		return _ok(result, 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.get("/statutory-reports/<report_id>")
def api_get_statutory_report(report_id: str):
	"""
	Get a statutory report.
	---
	GET /loc-mco/api/v1/statutory-reports/<report_id>
	Permission: loc_mco:statutory_reports
	"""
	try:
		result = _run(_SERVICE.get_statutory_report(_tenant(), report_id))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/statutory-reports/<report_id>/file")
def api_file_statutory_report(report_id: str):
	"""
	File a statutory report.
	---
	POST /loc-mco/api/v1/statutory-reports/<report_id>/file
	Body: {filer_id, filed_date}
	Permission: loc_mco:statutory_reports_write
	"""
	body = request.get_json(silent=True) or {}
	try:
		filed_date = date.fromisoformat(body.get("filed_date", str(date.today())))
		result = _run(_SERVICE.file_statutory_report(
			_tenant(), report_id,
			filer_id=body.get("filer_id", _actor()),
			filed_date=filed_date,
		))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Agents ----

@mco_api.get("/agents")
def api_list_agents():
	"""
	List MCO agents.
	---
	GET /loc-mco/api/v1/agents
	Permission: loc_mco:admin
	"""
	try:
		result = _run(_SERVICE.list_agents(_tenant()))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mco_api.post("/agents")
def api_create_agent():
	"""
	Register an MCO agent.
	---
	POST /loc-mco/api/v1/agents
	Body: McoAgentCreate JSON
	Permission: loc_mco:admin
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		payload = McoAgentCreate.model_validate(body)
		result = _run(_SERVICE.register_agent(payload, actor_id=_actor()))
		return _ok(result, 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Dashboard ----

@mco_api.get("/dashboard")
def api_dashboard():
	"""
	MCO dashboard summary.
	---
	GET /loc-mco/api/v1/dashboard
	Permission: loc_mco:view
	"""
	try:
		result = _run(_SERVICE.dashboard_summary(_tenant()))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Audit Events ----

@mco_api.get("/audit-events")
def api_audit_events():
	"""
	List recent audit events.
	---
	GET /loc-mco/api/v1/audit-events
	Query: limit (default 50)
	Permission: loc_mco:admin
	"""
	try:
		limit = int(request.args.get("limit", 50))
		result = _run(_SERVICE.list_audit_events(_tenant(), limit=limit))
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Process-local helper functions ---

def service() -> MultiCountryOperationsService:
	return _SERVICE


def register_country(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.register_country(CountryCreate.model_validate(payload))).model_dump(mode="json")


def register_entity(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.register_entity(EntityCreate.model_validate(payload))).model_dump(mode="json")


def record_compliance(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.record_compliance_mapping(ComplianceMappingCreate.model_validate(payload))).model_dump(mode="json")


def create_intercompany(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.create_intercompany_transaction(IntercompanyTransactionCreate.model_validate(payload))).model_dump(mode="json")


def create_statutory_report(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.create_statutory_report(StatutoryReportCreate.model_validate(payload))).model_dump(mode="json")


def dashboard(payload: dict) -> dict:
	return _run(_SERVICE.dashboard_summary(payload.get("tenant_id", "default")))
