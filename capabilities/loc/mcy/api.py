"""Process-local API helpers and Flask Blueprint REST endpoints for APG Multi-Currency Management."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import MultiCurrencyManagementService
	from .models import (
		CurrencyConfigCreate,
		CurrencyConfigUpdate,
		CurrencyTranslationCreate,
		ExchangeRateCreate,
		FxAccountCreate,
		McyAgentCreate,
		RevaluationCreate,
	)
except ImportError:  # pragma: no cover
	from service import MultiCurrencyManagementService  # type: ignore[no-redef]
	from models import (  # type: ignore[no-redef]
		CurrencyConfigCreate,
		CurrencyConfigUpdate,
		CurrencyTranslationCreate,
		ExchangeRateCreate,
		FxAccountCreate,
		McyAgentCreate,
		RevaluationCreate,
	)

_SERVICE = MultiCurrencyManagementService()
mcy_api = Blueprint("loc_mcy_api", __name__, url_prefix="/loc-mcy/api/v1")


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


# ---- Currencies ----

@mcy_api.get("/currencies")
def api_list_currencies():
	"""
	List configured currencies.
	---
	GET /loc-mcy/api/v1/currencies
	Query: status
	Permission: loc_mcy:currencies
	"""
	try:
		return _ok(_run(_SERVICE.list_currencies(_tenant(), status=request.args.get("status"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/currencies")
def api_create_currency():
	"""
	Configure a currency.
	---
	POST /loc-mcy/api/v1/currencies
	Body: CurrencyConfigCreate JSON
	Permission: loc_mcy:currencies_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.configure_currency(CurrencyConfigCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.get("/currencies/<currency_id>")
def api_get_currency(currency_id: str):
	"""
	Get a currency configuration.
	---
	GET /loc-mcy/api/v1/currencies/<currency_id>
	Permission: loc_mcy:currencies
	"""
	try:
		return _ok(_run(_SERVICE.get_currency(_tenant(), currency_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.put("/currencies/<currency_id>")
def api_update_currency(currency_id: str):
	"""
	Update a currency.
	---
	PUT /loc-mcy/api/v1/currencies/<currency_id>
	Body: CurrencyConfigUpdate JSON
	Permission: loc_mcy:currencies_write
	"""
	body = request.get_json(silent=True) or {}
	try:
		return _ok(_run(_SERVICE.update_currency(_tenant(), currency_id, CurrencyConfigUpdate.model_validate(body), actor_id=_actor())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Exchange Rates ----

@mcy_api.get("/exchange-rates")
def api_list_rates():
	"""
	List exchange rates.
	---
	GET /loc-mcy/api/v1/exchange-rates
	Query: from_currency, to_currency, rate_type, effective_date
	Permission: loc_mcy:exchange_rates
	"""
	try:
		eff_raw = request.args.get("effective_date")
		eff_date = date.fromisoformat(eff_raw) if eff_raw else None
		return _ok(_run(_SERVICE.list_exchange_rates(
			_tenant(),
			from_currency=request.args.get("from_currency"),
			to_currency=request.args.get("to_currency"),
			rate_type=request.args.get("rate_type"),
			effective_date=eff_date,
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/exchange-rates")
def api_create_rate():
	"""
	Record an exchange rate.
	---
	POST /loc-mcy/api/v1/exchange-rates
	Body: ExchangeRateCreate JSON
	Permission: loc_mcy:exchange_rates_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.record_exchange_rate(ExchangeRateCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.get("/exchange-rates/<rate_id>")
def api_get_rate(rate_id: str):
	"""
	Get an exchange rate.
	---
	GET /loc-mcy/api/v1/exchange-rates/<rate_id>
	Permission: loc_mcy:exchange_rates
	"""
	try:
		return _ok(_run(_SERVICE.get_exchange_rate(_tenant(), rate_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.get("/convert")
def api_convert():
	"""
	Convert amount between currencies.
	---
	GET /loc-mcy/api/v1/convert
	Query: amount, from_currency, to_currency, as_of, rate_type
	Permission: loc_mcy:exchange_rates
	"""
	try:
		amount = float(request.args["amount"])
		from_currency = request.args["from_currency"]
		to_currency = request.args["to_currency"]
		as_of = date.fromisoformat(request.args.get("as_of", str(date.today())))
		rate_type = request.args.get("rate_type", "spot")
		return _ok(_run(_SERVICE.convert_amount(_tenant(), amount, from_currency, to_currency, as_of, rate_type)))
	except KeyError as exc:
		return _err(f"missing parameter or not found: {exc}", 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- FX Accounts ----

@mcy_api.get("/fx-accounts")
def api_list_fx_accounts():
	"""
	List FX accounts.
	---
	GET /loc-mcy/api/v1/fx-accounts
	Query: account_type
	Permission: loc_mcy:fx_accounts
	"""
	try:
		return _ok(_run(_SERVICE.list_fx_accounts(_tenant(), account_type=request.args.get("account_type"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/fx-accounts")
def api_create_fx_account():
	"""
	Register an FX account.
	---
	POST /loc-mcy/api/v1/fx-accounts
	Body: FxAccountCreate JSON
	Permission: loc_mcy:fx_accounts
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.register_fx_account(FxAccountCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.get("/fx-accounts/<account_id>")
def api_get_fx_account(account_id: str):
	"""
	Get an FX account.
	---
	GET /loc-mcy/api/v1/fx-accounts/<account_id>
	Permission: loc_mcy:fx_accounts
	"""
	try:
		return _ok(_run(_SERVICE.get_fx_account(_tenant(), account_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Revaluation ----

@mcy_api.get("/revaluation")
def api_list_revaluations():
	"""
	List revaluation runs.
	---
	GET /loc-mcy/api/v1/revaluation
	Query: entity_id, status
	Permission: loc_mcy:revaluation
	"""
	try:
		return _ok(_run(_SERVICE.list_revaluations(
			_tenant(),
			entity_id=request.args.get("entity_id"),
			status=request.args.get("status"),
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/revaluation")
def api_create_revaluation():
	"""
	Create a revaluation run.
	---
	POST /loc-mcy/api/v1/revaluation
	Body: RevaluationCreate JSON
	Permission: loc_mcy:revaluation_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.create_revaluation(RevaluationCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.get("/revaluation/<revaluation_id>")
def api_get_revaluation(revaluation_id: str):
	"""
	Get a revaluation run.
	---
	GET /loc-mcy/api/v1/revaluation/<revaluation_id>
	Permission: loc_mcy:revaluation
	"""
	try:
		return _ok(_run(_SERVICE.get_revaluation(_tenant(), revaluation_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/revaluation/<revaluation_id>/post")
def api_post_revaluation(revaluation_id: str):
	"""
	Post an approved revaluation.
	---
	POST /loc-mcy/api/v1/revaluation/<revaluation_id>/post
	Permission: loc_mcy:revaluation_write
	"""
	try:
		return _ok(_run(_SERVICE.post_revaluation(_tenant(), revaluation_id, actor_id=_actor())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mcy_api.post("/revaluation/<revaluation_id>/reverse")
def api_reverse_revaluation(revaluation_id: str):
	"""
	Reverse a posted revaluation.
	---
	POST /loc-mcy/api/v1/revaluation/<revaluation_id>/reverse
	Permission: loc_mcy:revaluation_write
	"""
	try:
		return _ok(_run(_SERVICE.reverse_revaluation(_tenant(), revaluation_id, actor_id=_actor())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


# ---- Translation ----

@mcy_api.get("/translation")
def api_list_translations():
	"""
	List currency translation runs.
	---
	GET /loc-mcy/api/v1/translation
	Query: entity_id, status
	Permission: loc_mcy:translation
	"""
	try:
		return _ok(_run(_SERVICE.list_translations(
			_tenant(),
			entity_id=request.args.get("entity_id"),
			status=request.args.get("status"),
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/translation")
def api_create_translation():
	"""
	Create a currency translation run.
	---
	POST /loc-mcy/api/v1/translation
	Body: CurrencyTranslationCreate JSON
	Permission: loc_mcy:translation_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.create_translation(CurrencyTranslationCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.get("/translation/<translation_id>")
def api_get_translation(translation_id: str):
	"""
	Get a translation run.
	---
	GET /loc-mcy/api/v1/translation/<translation_id>
	Permission: loc_mcy:translation
	"""
	try:
		return _ok(_run(_SERVICE.get_translation(_tenant(), translation_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/translation/<translation_id>/post")
def api_post_translation(translation_id: str):
	"""
	Post an approved translation run.
	---
	POST /loc-mcy/api/v1/translation/<translation_id>/post
	Permission: loc_mcy:translation_write
	"""
	try:
		return _ok(_run(_SERVICE.post_translation(_tenant(), translation_id, actor_id=_actor())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


# ---- FX Reporting ----

@mcy_api.get("/fx-reporting")
def api_fx_report():
	"""
	FX gain/loss report.
	---
	GET /loc-mcy/api/v1/fx-reporting
	Query: period_start, period_end, entity_id
	Permission: loc_mcy:fx_reporting
	"""
	try:
		period_start = date.fromisoformat(request.args["period_start"])
		period_end = date.fromisoformat(request.args["period_end"])
		return _ok(_run(_SERVICE.generate_fx_report(
			_tenant(),
			period_start=period_start,
			period_end=period_end,
			entity_id=request.args.get("entity_id"),
		)))
	except KeyError as exc:
		return _err(f"missing parameter: {exc}", 400)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Agents ----

@mcy_api.get("/agents")
def api_list_agents():
	"""
	List MCY agents.
	---
	GET /loc-mcy/api/v1/agents
	Permission: loc_mcy:admin
	"""
	try:
		return _ok(_run(_SERVICE.list_agents(_tenant())))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mcy_api.post("/agents")
def api_create_agent():
	"""
	Register an MCY agent.
	---
	POST /loc-mcy/api/v1/agents
	Body: McyAgentCreate JSON
	Permission: loc_mcy:admin
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.register_agent(McyAgentCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Dashboard ----

@mcy_api.get("/dashboard")
def api_dashboard():
	"""
	MCY dashboard summary.
	---
	GET /loc-mcy/api/v1/dashboard
	Permission: loc_mcy:view
	"""
	try:
		return _ok(_run(_SERVICE.dashboard_summary(_tenant())))
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Audit Events ----

@mcy_api.get("/audit-events")
def api_audit_events():
	"""
	Audit event log.
	---
	GET /loc-mcy/api/v1/audit-events
	Query: limit
	Permission: loc_mcy:admin
	"""
	try:
		return _ok(_run(_SERVICE.list_audit_events(_tenant(), limit=int(request.args.get("limit", 50)))))
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Process-local helper functions ---

def service() -> MultiCurrencyManagementService:
	return _SERVICE


def configure_currency(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.configure_currency(CurrencyConfigCreate.model_validate(payload))).model_dump(mode="json")


def record_exchange_rate(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.record_exchange_rate(ExchangeRateCreate.model_validate(payload))).model_dump(mode="json")


def create_revaluation(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.create_revaluation(RevaluationCreate.model_validate(payload))).model_dump(mode="json")


def create_translation(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.create_translation(CurrencyTranslationCreate.model_validate(payload))).model_dump(mode="json")


def dashboard(payload: dict) -> dict:
	return _run(_SERVICE.dashboard_summary(payload.get("tenant_id", "default")))
