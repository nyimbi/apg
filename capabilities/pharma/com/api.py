"""Flask Blueprint REST API for APG Pharma Commercial Operations."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from .models import (
	CallRecordCreate, CommercialPlanCreate, HcpInteractionCreate,
	SalesRepCreate, SampleDispensingCreate, TerritoryCreate, TerritoryUpdate,
)
from .service import CommercialOperationsService

blueprint = Blueprint("pharma_com", __name__, url_prefix="/pharma-com/api/v1")
_svc = CommercialOperationsService()


def _svc_for(tenant_id: str) -> CommercialOperationsService:
	return _svc


def _err(msg: str, status: int = 400) -> tuple:
	return jsonify({"error": msg}), status


# --- contract ---

@blueprint.get("/contract")
def get_contract():
	"""Return capability contract."""
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).describe(tenant_id))


@blueprint.post("/evaluate")
def evaluate_rules():
	"""Evaluate capability rules against a context payload."""
	body = request.get_json(force=True) or {}
	return jsonify(_svc_for(body.get("tenant_id", "default")).evaluate(body))


# --- dashboard ---

@blueprint.get("/dashboard")
def dashboard():
	"""Dashboard summary."""
	tenant_id = request.args.get("tenant_id", "default")
	svc = _svc_for(tenant_id)
	return jsonify(svc.dashboard_summary(tenant_id))


# --- territories ---

@blueprint.get("/territories")
def list_territories():
	"""List all territories for a tenant."""
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([t.model_dump() for t in _svc_for(tenant_id).list_territories(tenant_id)])


@blueprint.post("/territories")
def create_territory():
	"""Create a new territory."""
	body = request.get_json(force=True) or {}
	try:
		payload = TerritoryCreate(**body)
		result = _svc_for(payload.tenant_id).create_territory(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/territories/<territory_id>")
def get_territory(territory_id: str):
	"""Get a territory by ID."""
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_territory(territory_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.put("/territories/<territory_id>")
def update_territory(territory_id: str):
	"""Update a territory's mutable fields."""
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", request.args.get("tenant_id", "default"))
	try:
		update = TerritoryUpdate(**{k: v for k, v in body.items() if k != "tenant_id"})
		result = _svc_for(tenant_id).update_territory(territory_id, tenant_id, update)
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 400)


# --- reps ---

@blueprint.get("/reps")
def list_reps():
	"""List all sales reps for a tenant."""
	tenant_id = request.args.get("tenant_id", "default")
	territory_id = request.args.get("territory_id")
	svc = _svc_for(tenant_id)
	if territory_id:
		return jsonify([r.model_dump() for r in svc.list_reps_by_territory(territory_id, tenant_id)])
	return jsonify([r.model_dump() for r in svc.list_reps(tenant_id)])


@blueprint.post("/reps")
def assign_rep():
	"""Assign a sales rep."""
	body = request.get_json(force=True) or {}
	try:
		payload = SalesRepCreate(**body)
		result = _svc_for(payload.tenant_id).assign_rep(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/reps/<rep_id>")
def get_rep(rep_id: str):
	"""Get a rep by ID."""
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_rep(rep_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


# --- calls ---

@blueprint.get("/calls")
def list_calls():
	"""List call records."""
	tenant_id = request.args.get("tenant_id", "default")
	rep_id = request.args.get("rep_id")
	return jsonify([c.model_dump() for c in _svc_for(tenant_id).list_calls(tenant_id, rep_id=rep_id)])


@blueprint.post("/calls")
def record_call():
	"""Record a physician call."""
	body = request.get_json(force=True) or {}
	try:
		payload = CallRecordCreate(**body)
		result = _svc_for(payload.tenant_id).record_call(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


# --- samples ---

@blueprint.get("/samples")
def list_samples():
	"""List sample dispensings."""
	tenant_id = request.args.get("tenant_id", "default")
	rep_id = request.args.get("rep_id")
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_samples(tenant_id, rep_id=rep_id)])


@blueprint.post("/samples")
def dispense_sample():
	"""Dispense a product sample."""
	body = request.get_json(force=True) or {}
	try:
		payload = SampleDispensingCreate(**body)
		result = _svc_for(payload.tenant_id).dispense_sample(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/samples/reconcile/<rep_id>")
def reconcile_samples(rep_id: str):
	"""Reconcile sample inventory for a rep."""
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).reconcile_samples(tenant_id, rep_id))


# --- interactions ---

@blueprint.get("/interactions")
def list_interactions():
	"""List HCP interactions."""
	tenant_id = request.args.get("tenant_id", "default")
	hcp_id = request.args.get("hcp_id")
	return jsonify([i.model_dump() for i in _svc_for(tenant_id).list_interactions(tenant_id, hcp_id=hcp_id)])


@blueprint.post("/interactions")
def record_interaction():
	"""Record an HCP interaction."""
	body = request.get_json(force=True) or {}
	try:
		payload = HcpInteractionCreate(**body)
		result = _svc_for(payload.tenant_id).record_interaction(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


# --- plans ---

@blueprint.get("/plans")
def list_plans():
	"""List commercial plans."""
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([p.model_dump() for p in _svc_for(tenant_id).list_plans(tenant_id)])


@blueprint.post("/plans")
def create_plan():
	"""Create a commercial plan."""
	body = request.get_json(force=True) or {}
	try:
		payload = CommercialPlanCreate(**body)
		result = _svc_for(payload.tenant_id).create_plan(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/plans/<plan_id>/approve")
def approve_plan(plan_id: str):
	"""Approve a commercial plan."""
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	approval_reference = body.get("approval_reference", "")
	try:
		result = _svc_for(tenant_id).approve_plan(plan_id, tenant_id, approval_reference)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- spend ---

@blueprint.get("/spend/summary")
def spend_summary():
	"""Get aggregate spend summary for a HCP."""
	tenant_id = request.args.get("tenant_id", "default")
	hcp_id = request.args.get("hcp_id", "")
	fiscal_year = request.args.get("fiscal_year", "")
	if not hcp_id or not fiscal_year:
		return _err("hcp_id and fiscal_year are required")
	return jsonify(_svc_for(tenant_id).get_aggregate_spend_summary(tenant_id, hcp_id, fiscal_year))


# --- targets ---

@blueprint.get("/targets")
def list_targets():
	"""List target physicians."""
	tenant_id = request.args.get("tenant_id", "default")
	territory_id = request.args.get("territory_id")
	return jsonify([t.model_dump() for t in _svc_for(tenant_id).list_targets(tenant_id, territory_id=territory_id)])
