"""Flask Blueprint views for APG Multi-Currency Management."""

from __future__ import annotations

import asyncio
from datetime import date
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, Response, jsonify, request

try:
	from .service import MultiCurrencyManagementService
	from .models import (
		CurrencyConfigCreate,
		CurrencyConfigUpdate,
		ExchangeRateCreate,
		FxAccountCreate,
		McyAgentCreate,
		RevaluationCreate,
		CurrencyTranslationCreate,
	)
except ImportError:  # pragma: no cover
	from service import MultiCurrencyManagementService  # type: ignore[no-redef]
	from models import (  # type: ignore[no-redef]
		CurrencyConfigCreate,
		CurrencyConfigUpdate,
		ExchangeRateCreate,
		FxAccountCreate,
		McyAgentCreate,
		RevaluationCreate,
		CurrencyTranslationCreate,
	)

mcy_views = Blueprint("loc_mcy", __name__, url_prefix="/loc-mcy")
_svc = MultiCurrencyManagementService()


def _run(coro: Any) -> Any:
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
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			granted = request.headers.get("X-Permissions", "")
			if permission not in granted and "loc_mcy:admin" not in granted:
				return jsonify({"error": "forbidden", "permission_required": permission}), 403
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _ok(data: Any, status: int = 200) -> Response:
	if hasattr(data, "model_dump"):
		return jsonify(data.model_dump(mode="json")), status  # type: ignore[return-value]
	if isinstance(data, list):
		return jsonify([i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in data]), status  # type: ignore[return-value]
	return jsonify(data), status  # type: ignore[return-value]


def _err(msg: str, status: int = 400) -> Response:
	return jsonify({"error": msg}), status  # type: ignore[return-value]


# --- Dashboard ---

@mcy_views.get("/dashboard")
@has_access("loc_mcy:view")
def dashboard() -> Response:
	"""MCY dashboard summary."""
	try:
		return _ok(_run(_svc.dashboard_summary(_tenant_id())))
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Currencies ---

@mcy_views.get("/currencies")
@has_access("loc_mcy:currencies")
def list_currencies() -> Response:
	"""List configured currencies."""
	try:
		return _ok(_run(_svc.list_currencies(_tenant_id(), status=request.args.get("status"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/currencies")
@has_access("loc_mcy:currencies_write")
def create_currency() -> Response:
	"""Configure a new currency."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.configure_currency(CurrencyConfigCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.get("/currencies/<currency_id>")
@has_access("loc_mcy:currencies")
def get_currency(currency_id: str) -> Response:
	"""Get a currency configuration."""
	try:
		return _ok(_run(_svc.get_currency(_tenant_id(), currency_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.put("/currencies/<currency_id>")
@has_access("loc_mcy:currencies_write")
def update_currency(currency_id: str) -> Response:
	"""Update a currency configuration."""
	body = request.get_json(silent=True) or {}
	try:
		return _ok(_run(_svc.update_currency(_tenant_id(), currency_id, CurrencyConfigUpdate.model_validate(body), actor_id=_actor_id())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Exchange Rates ---

@mcy_views.get("/exchange-rates")
@has_access("loc_mcy:exchange_rates")
def list_rates() -> Response:
	"""List exchange rates."""
	try:
		eff_date_raw = request.args.get("effective_date")
		eff_date = date.fromisoformat(eff_date_raw) if eff_date_raw else None
		return _ok(_run(_svc.list_exchange_rates(
			_tenant_id(),
			from_currency=request.args.get("from_currency"),
			to_currency=request.args.get("to_currency"),
			rate_type=request.args.get("rate_type"),
			effective_date=eff_date,
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/exchange-rates")
@has_access("loc_mcy:exchange_rates_write")
def create_rate() -> Response:
	"""Record an exchange rate."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.record_exchange_rate(ExchangeRateCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.get("/exchange-rates/<rate_id>")
@has_access("loc_mcy:exchange_rates")
def get_rate(rate_id: str) -> Response:
	"""Get an exchange rate."""
	try:
		return _ok(_run(_svc.get_exchange_rate(_tenant_id(), rate_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.get("/convert")
@has_access("loc_mcy:exchange_rates")
def convert_amount() -> Response:
	"""Convert an amount between currencies."""
	try:
		amount = float(request.args["amount"])
		from_currency = request.args["from_currency"]
		to_currency = request.args["to_currency"]
		as_of_raw = request.args.get("as_of", str(date.today()))
		as_of = date.fromisoformat(as_of_raw)
		rate_type = request.args.get("rate_type", "spot")
		result = _run(_svc.convert_amount(_tenant_id(), amount, from_currency, to_currency, as_of, rate_type))
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- FX Accounts ---

@mcy_views.get("/fx-accounts")
@has_access("loc_mcy:fx_accounts")
def list_fx_accounts() -> Response:
	"""List FX accounts."""
	try:
		return _ok(_run(_svc.list_fx_accounts(_tenant_id(), account_type=request.args.get("account_type"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/fx-accounts")
@has_access("loc_mcy:fx_accounts")
def create_fx_account() -> Response:
	"""Register an FX account."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.register_fx_account(FxAccountCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.get("/fx-accounts/<account_id>")
@has_access("loc_mcy:fx_accounts")
def get_fx_account(account_id: str) -> Response:
	"""Get an FX account."""
	try:
		return _ok(_run(_svc.get_fx_account(_tenant_id(), account_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Revaluation ---

@mcy_views.get("/revaluation")
@has_access("loc_mcy:revaluation")
def list_revaluations() -> Response:
	"""List revaluation runs."""
	try:
		return _ok(_run(_svc.list_revaluations(
			_tenant_id(),
			entity_id=request.args.get("entity_id"),
			status=request.args.get("status"),
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/revaluation")
@has_access("loc_mcy:revaluation_write")
def create_revaluation() -> Response:
	"""Create a revaluation run."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.create_revaluation(RevaluationCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.get("/revaluation/<revaluation_id>")
@has_access("loc_mcy:revaluation")
def get_revaluation(revaluation_id: str) -> Response:
	"""Get a revaluation run."""
	try:
		return _ok(_run(_svc.get_revaluation(_tenant_id(), revaluation_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/revaluation/<revaluation_id>/post")
@has_access("loc_mcy:revaluation_write")
def post_revaluation(revaluation_id: str) -> Response:
	"""Post an approved revaluation."""
	try:
		return _ok(_run(_svc.post_revaluation(_tenant_id(), revaluation_id, actor_id=_actor_id())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mcy_views.post("/revaluation/<revaluation_id>/reverse")
@has_access("loc_mcy:revaluation_write")
def reverse_revaluation(revaluation_id: str) -> Response:
	"""Reverse a posted revaluation."""
	try:
		return _ok(_run(_svc.reverse_revaluation(_tenant_id(), revaluation_id, actor_id=_actor_id())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


# --- Translation ---

@mcy_views.get("/translation")
@has_access("loc_mcy:translation")
def list_translations() -> Response:
	"""List currency translation runs."""
	try:
		return _ok(_run(_svc.list_translations(
			_tenant_id(),
			entity_id=request.args.get("entity_id"),
			status=request.args.get("status"),
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/translation")
@has_access("loc_mcy:translation_write")
def create_translation() -> Response:
	"""Create a currency translation run."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.create_translation(CurrencyTranslationCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.get("/translation/<translation_id>")
@has_access("loc_mcy:translation")
def get_translation(translation_id: str) -> Response:
	"""Get a translation run."""
	try:
		return _ok(_run(_svc.get_translation(_tenant_id(), translation_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/translation/<translation_id>/post")
@has_access("loc_mcy:translation_write")
def post_translation(translation_id: str) -> Response:
	"""Post an approved translation run."""
	try:
		return _ok(_run(_svc.post_translation(_tenant_id(), translation_id, actor_id=_actor_id())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


# --- FX Reporting ---

@mcy_views.get("/fx-reporting")
@has_access("loc_mcy:fx_reporting")
def fx_report() -> Response:
	"""Generate an FX gain/loss report."""
	try:
		period_start = date.fromisoformat(request.args["period_start"])
		period_end = date.fromisoformat(request.args["period_end"])
		return _ok(_run(_svc.generate_fx_report(
			_tenant_id(),
			period_start=period_start,
			period_end=period_end,
			entity_id=request.args.get("entity_id"),
		)))
	except KeyError as exc:
		return _err(f"missing parameter: {exc}", 400)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Agents ---

@mcy_views.get("/agents")
@has_access("loc_mcy:admin")
def list_agents() -> Response:
	"""List MCY agents."""
	try:
		return _ok(_run(_svc.list_agents(_tenant_id())))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_views.post("/agents")
@has_access("loc_mcy:admin")
def create_agent() -> Response:
	"""Register an MCY agent."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.register_agent(McyAgentCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Audit Events ---

@mcy_views.get("/audit-events")
@has_access("loc_mcy:admin")
def audit_events() -> Response:
	"""Recent audit events."""
	try:
		limit = int(request.args.get("limit", 50))
		return _ok(_run(_svc.list_audit_events(_tenant_id(), limit=limit)))
	except PermissionError as exc:
		return _err(str(exc), 403)
