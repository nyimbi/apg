"""Flask Blueprint REST API for Land Registry (gov_lnd)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import LandRegistryService
	from .models import (
		ParcelCreate, ParcelUpdate,
		TitleCreate, TitleUpdate,
		TransferCreate,
		AdjudicationCreate,
		EncumbranceCreate, EncumbranceUpdate,
		ValuationCreate,
	)
except ImportError:
	from service import LandRegistryService  # type: ignore
	from models import (  # type: ignore
		ParcelCreate, ParcelUpdate,
		TitleCreate, TitleUpdate,
		TransferCreate,
		AdjudicationCreate,
		EncumbranceCreate, EncumbranceUpdate,
		ValuationCreate,
	)

_log = logging.getLogger(__name__)

bp = Blueprint("gov_lnd", __name__, url_prefix="/api/government/lnd")

_svc = LandRegistryService()


def _json_response(data: Any, status: int = 200):
	return jsonify(data), status


def _error(message: str, status: int = 400):
	return jsonify({"error": message}), status


# ── Health ─────────────────────────────────────────────────────────────────────

@bp.get("/health")
async def health():
	try:
		return _json_response(await _svc.health_check())
	except Exception as exc:
		_log.error("health_check failed: %s", exc)
		return _error(str(exc), 500)


# ── Parcels ────────────────────────────────────────────────────────────────────

@bp.get("/parcels")
async def list_parcels():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		county = request.args.get("county")
		land_use = request.args.get("land_use")
		status = request.args.get("status")
		result = await _svc.list_parcels(tenant_id=tenant_id, county=county, land_use=land_use, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_parcels failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/parcels/<parcel_id>")
async def get_parcel(parcel_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.get_parcel(parcel_id, tenant_id=tenant_id))
	except KeyError:
		return _error("parcel not found", 404)
	except Exception as exc:
		_log.error("get_parcel failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/parcels")
async def register_parcel():
	try:
		body = ParcelCreate(**request.get_json(force=True))
		result = await _svc.register_parcel(
			parcel_number=body.parcel_number,
			county=body.county,
			sub_county=body.sub_county,
			location=body.location,
			area_hectares=body.area_hectares,
			tenant_id=body.tenant_id,
			land_use=body.land_use,
			coordinates=body.coordinates,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except (ValueError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("register_parcel failed: %s", exc)
		return _error(str(exc), 500)


@bp.put("/parcels/<parcel_id>")
async def update_parcel(parcel_id: str):
	try:
		body = ParcelUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.update_parcel(
			parcel_id=parcel_id,
			tenant_id=tenant_id,
			land_use=body.land_use,
			area_hectares=body.area_hectares,
			coordinates=body.coordinates,
			metadata=body.metadata,
			status=body.status,
		)
		return _json_response(result)
	except KeyError:
		return _error("parcel not found", 404)
	except ValueError as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("update_parcel failed: %s", exc)
		return _error(str(exc), 500)


@bp.delete("/parcels/<parcel_id>")
async def delete_parcel(parcel_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.delete_parcel(parcel_id, tenant_id=tenant_id))
	except KeyError:
		return _error("parcel not found", 404)
	except Exception as exc:
		_log.error("delete_parcel failed: %s", exc)
		return _error(str(exc), 500)


# ── Titles ─────────────────────────────────────────────────────────────────────

@bp.get("/titles")
async def list_titles():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		owner_id = request.args.get("owner_id")
		status = request.args.get("status")
		result = await _svc.list_titles(tenant_id=tenant_id, owner_id=owner_id, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_titles failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/titles/<title_id>")
async def get_title(title_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.get_title(title_id, tenant_id=tenant_id))
	except KeyError:
		return _error("title not found", 404)
	except Exception as exc:
		_log.error("get_title failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/titles")
async def issue_title():
	try:
		body = TitleCreate(**request.get_json(force=True))
		result = await _svc.issue_title(
			parcel_id=body.parcel_id,
			title_number=body.title_number,
			owner_id=body.owner_id,
			owner_name=body.owner_name,
			issue_date=body.issue_date,
			issued_by=body.issued_by,
			tenant_id=body.tenant_id,
			owner_type=body.owner_type,
			tenure_type=body.tenure_type,
			lease_term_years=body.lease_term_years,
		)
		return _json_response(result, 201)
	except (KeyError, PermissionError, ValueError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("issue_title failed: %s", exc)
		return _error(str(exc), 500)


@bp.put("/titles/<title_id>")
async def update_title(title_id: str):
	try:
		body = TitleUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.update_title(
			title_id=title_id,
			tenant_id=tenant_id,
			owner_id=body.owner_id,
			owner_name=body.owner_name,
			status=body.status,
			notes=body.notes,
		)
		return _json_response(result)
	except KeyError:
		return _error("title not found", 404)
	except Exception as exc:
		_log.error("update_title failed: %s", exc)
		return _error(str(exc), 500)


# ── Transfers ──────────────────────────────────────────────────────────────────

@bp.get("/transfers")
async def list_transfers():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		title_id = request.args.get("title_id")
		status = request.args.get("status")
		result = await _svc.list_transfers(tenant_id=tenant_id, title_id=title_id, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_transfers failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/transfers")
async def initiate_transfer():
	try:
		body = TransferCreate(**request.get_json(force=True))
		result = await _svc.initiate_transfer(
			title_id=body.title_id,
			transferor_id=body.transferor_id,
			transferor_name=body.transferor_name,
			transferee_id=body.transferee_id,
			transferee_name=body.transferee_name,
			consideration_kes=body.consideration_kes,
			transfer_date=body.transfer_date,
			instrument_number=body.instrument_number,
			approved_by=body.approved_by,
			tenant_id=body.tenant_id,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except (KeyError, PermissionError, ValueError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("initiate_transfer failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/transfers/<transfer_id>/complete")
async def complete_transfer(transfer_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.complete_transfer(transfer_id, tenant_id=tenant_id))
	except (KeyError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("complete_transfer failed: %s", exc)
		return _error(str(exc), 500)


# ── Adjudications ──────────────────────────────────────────────────────────────

@bp.get("/adjudications")
async def list_adjudications():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		status = request.args.get("status")
		result = await _svc.list_adjudications(tenant_id=tenant_id, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_adjudications failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/adjudications")
async def submit_adjudication():
	try:
		body = AdjudicationCreate(**request.get_json(force=True))
		result = await _svc.submit_adjudication(
			parcel_id=body.parcel_id,
			claimant_id=body.claimant_id,
			claimant_name=body.claimant_name,
			claim_basis=body.claim_basis,
			evidence_reference=body.evidence_reference,
			adjudicator_id=body.adjudicator_id,
			tenant_id=body.tenant_id,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except (ValueError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("submit_adjudication failed: %s", exc)
		return _error(str(exc), 500)


# ── Encumbrances ───────────────────────────────────────────────────────────────

@bp.get("/encumbrances")
async def list_encumbrances():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		title_id = request.args.get("title_id")
		status = request.args.get("status")
		result = await _svc.list_encumbrances(tenant_id=tenant_id, title_id=title_id, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_encumbrances failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/encumbrances")
async def register_encumbrance():
	try:
		body = EncumbranceCreate(**request.get_json(force=True))
		result = await _svc.register_encumbrance(
			title_id=body.title_id,
			encumbrance_type=body.encumbrance_type,
			holder_id=body.holder_id,
			holder_name=body.holder_name,
			start_date=body.start_date,
			instrument_reference=body.instrument_reference,
			registered_by=body.registered_by,
			tenant_id=body.tenant_id,
			amount_kes=body.amount_kes,
			end_date=body.end_date,
		)
		return _json_response(result, 201)
	except (KeyError, ValueError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("register_encumbrance failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/encumbrances/<encumbrance_id>/discharge")
async def discharge_encumbrance(encumbrance_id: str):
	try:
		body = EncumbranceUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.discharge_encumbrance(
			encumbrance_id=encumbrance_id,
			discharge_reference=body.discharge_reference or "",
			discharged_by=body.discharged_by or "",
			tenant_id=tenant_id,
		)
		return _json_response(result)
	except (KeyError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("discharge_encumbrance failed: %s", exc)
		return _error(str(exc), 500)


# ── Valuations ─────────────────────────────────────────────────────────────────

@bp.get("/valuations")
async def list_valuations():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		status = request.args.get("status")
		result = await _svc.list_valuations(tenant_id=tenant_id, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_valuations failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/valuations")
async def record_valuation():
	try:
		body = ValuationCreate(**request.get_json(force=True))
		result = await _svc.record_valuation(
			parcel_id=body.parcel_id,
			valuation_date=body.valuation_date,
			market_value_kes=body.market_value_kes,
			annual_rental_value_kes=body.annual_rental_value_kes,
			unimproved_site_value_kes=body.unimproved_site_value_kes,
			valuer_id=body.valuer_id,
			tenant_id=body.tenant_id,
			valuation_method=body.valuation_method,
		)
		return _json_response(result, 201)
	except (KeyError, ValueError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("record_valuation failed: %s", exc)
		return _error(str(exc), 500)


# ── Dashboard & audit ──────────────────────────────────────────────────────────

@bp.get("/dashboard")
async def dashboard():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.dashboard_summary(tenant_id=tenant_id))
	except Exception as exc:
		_log.error("dashboard failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/audit-events")
async def audit_events():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.get_audit_events(tenant_id=tenant_id)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("audit_events failed: %s", exc)
		return _error(str(exc), 500)
