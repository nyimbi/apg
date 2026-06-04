"""Flask Blueprint views for APG Multi-Country Operations."""

from __future__ import annotations

import asyncio
import json
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, Response, g, jsonify, request

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
		IntercompanyTransactionUpdate,
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
		IntercompanyTransactionUpdate,
		McoAgentCreate,
		StatutoryReportCreate,
		StatutoryReportUpdate,
	)

mco_views = Blueprint("loc_mco", __name__, url_prefix="/loc-mco")
_svc = MultiCountryOperationsService()


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask view."""
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant_id() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _actor_id() -> str:
	return request.headers.get("X-Actor-ID", "system")


def has_access(permission: str) -> Callable:
	"""Decorator: enforce permission check via request context."""
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			# In production this would delegate to the auth capability.
			# Here we check a simple header for testability.
			granted = request.headers.get("X-Permissions", "")
			if permission not in granted and "loc_mco:admin" not in granted:
				return jsonify({"error": "forbidden", "permission_required": permission}), 403
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _json_response(data: Any, status: int = 200) -> Response:
	"""Serialize Pydantic models or plain dicts to JSON response."""
	if hasattr(data, "model_dump"):
		payload = data.model_dump(mode="json")
	elif isinstance(data, list):
		payload = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in data]
	else:
		payload = data
	return jsonify(payload), status  # type: ignore[return-value]


def _error(msg: str, status: int = 400) -> Response:
	return jsonify({"error": msg}), status  # type: ignore[return-value]


# --- Dashboard ---

@mco_views.get("/dashboard")
@has_access("loc_mco:view")
def dashboard() -> Response:
	"""Render MCO dashboard summary."""
	tenant_id = _tenant_id()
	try:
		data = _run(_svc.dashboard_summary(tenant_id))
		return _json_response(data)
	except PermissionError as exc:
		return _error(str(exc), 403)


# --- Countries ---

@mco_views.get("/countries")
@has_access("loc_mco:countries")
def list_countries() -> Response:
	"""List all registered countries for the tenant."""
	tenant_id = _tenant_id()
	status_filter = request.args.get("status")
	try:
		countries = _run(_svc.list_countries(tenant_id, status=status_filter))
		return _json_response(countries)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/countries")
@has_access("loc_mco:countries_write")
def create_country() -> Response:
	"""Register a new country/jurisdiction."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = tenant_id
	try:
		payload = CountryCreate.model_validate(body)
		country = _run(_svc.register_country(payload, actor_id=actor_id))
		return _json_response(country, 201)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.get("/countries/<country_id>")
@has_access("loc_mco:countries")
def get_country(country_id: str) -> Response:
	"""Get a single country record."""
	tenant_id = _tenant_id()
	try:
		country = _run(_svc.get_country(tenant_id, country_id))
		return _json_response(country)
	except KeyError as exc:
		return _error(str(exc), 404)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.put("/countries/<country_id>")
@has_access("loc_mco:countries_write")
def update_country(country_id: str) -> Response:
	"""Update an existing country record."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	try:
		payload = CountryUpdate.model_validate(body)
		country = _run(_svc.update_country(tenant_id, country_id, payload, actor_id=actor_id))
		return _json_response(country)
	except KeyError as exc:
		return _error(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


# --- Entities ---

@mco_views.get("/entities")
@has_access("loc_mco:entities")
def list_entities() -> Response:
	"""List legal entities for the tenant."""
	tenant_id = _tenant_id()
	country_id = request.args.get("country_id")
	entity_type = request.args.get("entity_type")
	is_active_raw = request.args.get("is_active")
	is_active = None if is_active_raw is None else is_active_raw.lower() == "true"
	try:
		entities = _run(_svc.list_entities(tenant_id, country_id=country_id, entity_type=entity_type, is_active=is_active))
		return _json_response(entities)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/entities")
@has_access("loc_mco:entities_write")
def create_entity() -> Response:
	"""Register a new legal entity."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = tenant_id
	try:
		payload = EntityCreate.model_validate(body)
		entity = _run(_svc.register_entity(payload, actor_id=actor_id))
		return _json_response(entity, 201)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.get("/entities/<entity_id>")
@has_access("loc_mco:entities")
def get_entity(entity_id: str) -> Response:
	"""Get a single legal entity."""
	tenant_id = _tenant_id()
	try:
		entity = _run(_svc.get_entity(tenant_id, entity_id))
		return _json_response(entity)
	except KeyError as exc:
		return _error(str(exc), 404)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.put("/entities/<entity_id>")
@has_access("loc_mco:entities_write")
def update_entity(entity_id: str) -> Response:
	"""Update a legal entity."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	try:
		payload = EntityUpdate.model_validate(body)
		entity = _run(_svc.update_entity(tenant_id, entity_id, payload, actor_id=actor_id))
		return _json_response(entity)
	except KeyError as exc:
		return _error(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


# --- Compliance Mappings ---

@mco_views.get("/compliance")
@has_access("loc_mco:compliance")
def list_compliance() -> Response:
	"""List compliance mappings for the tenant."""
	tenant_id = _tenant_id()
	entity_id = request.args.get("entity_id")
	domain = request.args.get("domain")
	status = request.args.get("status")
	try:
		mappings = _run(_svc.list_compliance_mappings(tenant_id, entity_id=entity_id, domain=domain, status=status))
		return _json_response(mappings)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/compliance")
@has_access("loc_mco:compliance_write")
def create_compliance() -> Response:
	"""Record a new compliance mapping."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = tenant_id
	try:
		payload = ComplianceMappingCreate.model_validate(body)
		mapping = _run(_svc.record_compliance_mapping(payload, actor_id=actor_id))
		return _json_response(mapping, 201)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.get("/compliance/<mapping_id>")
@has_access("loc_mco:compliance")
def get_compliance(mapping_id: str) -> Response:
	"""Get a compliance mapping by ID."""
	tenant_id = _tenant_id()
	try:
		mapping = _run(_svc.get_compliance_mapping(tenant_id, mapping_id))
		return _json_response(mapping)
	except KeyError as exc:
		return _error(str(exc), 404)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.put("/compliance/<mapping_id>")
@has_access("loc_mco:compliance_write")
def update_compliance(mapping_id: str) -> Response:
	"""Update a compliance mapping."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	try:
		payload = ComplianceMappingUpdate.model_validate(body)
		mapping = _run(_svc.update_compliance_mapping(tenant_id, mapping_id, payload, actor_id=actor_id))
		return _json_response(mapping)
	except KeyError as exc:
		return _error(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


# --- Intercompany Transactions ---

@mco_views.get("/intercompany")
@has_access("loc_mco:intercompany")
def list_intercompany() -> Response:
	"""List intercompany transactions for the tenant."""
	tenant_id = _tenant_id()
	entity_id = request.args.get("entity_id")
	txn_type = request.args.get("type")
	status = request.args.get("status")
	try:
		txns = _run(_svc.list_intercompany_transactions(tenant_id, entity_id=entity_id, txn_type=txn_type, status=status))
		return _json_response(txns)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/intercompany")
@has_access("loc_mco:intercompany_write")
def create_intercompany() -> Response:
	"""Create an intercompany transaction."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = tenant_id
	try:
		payload = IntercompanyTransactionCreate.model_validate(body)
		txn = _run(_svc.create_intercompany_transaction(payload, actor_id=actor_id))
		return _json_response(txn, 201)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.get("/intercompany/<txn_id>")
@has_access("loc_mco:intercompany")
def get_intercompany(txn_id: str) -> Response:
	"""Get a single intercompany transaction."""
	tenant_id = _tenant_id()
	try:
		txn = _run(_svc.get_intercompany_transaction(tenant_id, txn_id))
		return _json_response(txn)
	except KeyError as exc:
		return _error(str(exc), 404)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/intercompany/<txn_id>/approve")
@has_access("loc_mco:intercompany_write")
def approve_intercompany(txn_id: str) -> Response:
	"""Approve a pending intercompany transaction."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	try:
		txn = _run(_svc.approve_intercompany_transaction(
			tenant_id, txn_id,
			approver_id=body.get("approver_id", actor_id),
			approval_reference=body["approval_reference"],
		))
		return _json_response(txn)
	except KeyError as exc:
		return _error(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


# --- Statutory Reports ---

@mco_views.get("/statutory-reports")
@has_access("loc_mco:statutory_reports")
def list_statutory_reports() -> Response:
	"""List statutory reports for the tenant."""
	tenant_id = _tenant_id()
	entity_id = request.args.get("entity_id")
	report_type = request.args.get("report_type")
	status = request.args.get("status")
	try:
		reports = _run(_svc.list_statutory_reports(tenant_id, entity_id=entity_id, report_type=report_type, status=status))
		return _json_response(reports)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/statutory-reports")
@has_access("loc_mco:statutory_reports_write")
def create_statutory_report() -> Response:
	"""Create a new statutory report."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = tenant_id
	try:
		payload = StatutoryReportCreate.model_validate(body)
		report = _run(_svc.create_statutory_report(payload, actor_id=actor_id))
		return _json_response(report, 201)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.get("/statutory-reports/<report_id>")
@has_access("loc_mco:statutory_reports")
def get_statutory_report(report_id: str) -> Response:
	"""Get a statutory report by ID."""
	tenant_id = _tenant_id()
	try:
		report = _run(_svc.get_statutory_report(tenant_id, report_id))
		return _json_response(report)
	except KeyError as exc:
		return _error(str(exc), 404)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/statutory-reports/<report_id>/file")
@has_access("loc_mco:statutory_reports_write")
def file_statutory_report(report_id: str) -> Response:
	"""Mark a statutory report as filed."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	try:
		from datetime import date
		filed_date = date.fromisoformat(body.get("filed_date", str(date.today())))
		report = _run(_svc.file_statutory_report(
			tenant_id, report_id,
			filer_id=body.get("filer_id", actor_id),
			filed_date=filed_date,
		))
		return _json_response(report)
	except KeyError as exc:
		return _error(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


# --- Agents ---

@mco_views.get("/agents")
@has_access("loc_mco:admin")
def list_agents() -> Response:
	"""List all MCO agents for the tenant."""
	tenant_id = _tenant_id()
	try:
		agents = _run(_svc.list_agents(tenant_id))
		return _json_response(agents)
	except PermissionError as exc:
		return _error(str(exc), 403)


@mco_views.post("/agents")
@has_access("loc_mco:admin")
def create_agent() -> Response:
	"""Register an MCO automation agent."""
	tenant_id = _tenant_id()
	actor_id = _actor_id()
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = tenant_id
	try:
		payload = McoAgentCreate.model_validate(body)
		agent = _run(_svc.register_agent(payload, actor_id=actor_id))
		return _json_response(agent, 201)
	except (AssertionError, ValueError) as exc:
		return _error(str(exc), 422)
	except PermissionError as exc:
		return _error(str(exc), 403)


# --- Audit Events ---

@mco_views.get("/audit-events")
@has_access("loc_mco:admin")
def audit_events() -> Response:
	"""Return recent audit events for the tenant."""
	tenant_id = _tenant_id()
	limit = int(request.args.get("limit", 50))
	try:
		events = _run(_svc.list_audit_events(tenant_id, limit=limit))
		return _json_response(events)
	except PermissionError as exc:
		return _error(str(exc), 403)
