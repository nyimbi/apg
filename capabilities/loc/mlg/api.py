"""Process-local API helpers and Flask Blueprint REST endpoints for APG Multi-Language & Localisation."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import MultiLanguageLocalisationService
	from .models import (
		FormattingRuleCreate,
		LocaleConfigCreate,
		LocaleConfigUpdate,
		MlgAgentCreate,
		TerminologyCreate,
		TranslationCreate,
		TranslationUpdate,
	)
except ImportError:  # pragma: no cover
	from service import MultiLanguageLocalisationService  # type: ignore[no-redef]
	from models import (  # type: ignore[no-redef]
		FormattingRuleCreate,
		LocaleConfigCreate,
		LocaleConfigUpdate,
		MlgAgentCreate,
		TerminologyCreate,
		TranslationCreate,
		TranslationUpdate,
	)

_SERVICE = MultiLanguageLocalisationService()
mlg_api = Blueprint("loc_mlg_api", __name__, url_prefix="/loc-mlg/api/v1")


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


# ---- Locales ----

@mlg_api.get("/locales")
def api_list_locales():
	"""
	List configured locales.
	---
	GET /loc-mlg/api/v1/locales
	Query: language, is_rtl
	Permission: loc_mlg:locales
	"""
	try:
		is_rtl_raw = request.args.get("is_rtl")
		is_rtl = None if is_rtl_raw is None else is_rtl_raw.lower() == "true"
		return _ok(_run(_SERVICE.list_locales(_tenant(), language=request.args.get("language"), is_rtl=is_rtl)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.post("/locales")
def api_create_locale():
	"""
	Configure a locale.
	---
	POST /loc-mlg/api/v1/locales
	Body: LocaleConfigCreate JSON
	Permission: loc_mlg:locales_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.configure_locale(LocaleConfigCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.get("/locales/<locale_id>")
def api_get_locale(locale_id: str):
	"""
	Get a locale by ID.
	---
	GET /loc-mlg/api/v1/locales/<locale_id>
	Permission: loc_mlg:locales
	"""
	try:
		return _ok(_run(_SERVICE.get_locale(_tenant(), locale_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.put("/locales/<locale_id>")
def api_update_locale(locale_id: str):
	"""
	Update a locale configuration.
	---
	PUT /loc-mlg/api/v1/locales/<locale_id>
	Body: LocaleConfigUpdate JSON
	Permission: loc_mlg:locales_write
	"""
	body = request.get_json(silent=True) or {}
	try:
		return _ok(_run(_SERVICE.update_locale(_tenant(), locale_id, LocaleConfigUpdate.model_validate(body), actor_id=_actor())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Translations ----

@mlg_api.get("/translations")
def api_list_translations():
	"""
	List translation entries.
	---
	GET /loc-mlg/api/v1/translations
	Query: target_language, content_type, status, namespace
	Permission: loc_mlg:translations
	"""
	try:
		return _ok(_run(_SERVICE.list_translations(
			_tenant(),
			target_language=request.args.get("target_language"),
			content_type=request.args.get("content_type"),
			status=request.args.get("status"),
			namespace=request.args.get("namespace"),
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.post("/translations")
def api_create_translation():
	"""
	Create a translation entry.
	---
	POST /loc-mlg/api/v1/translations
	Body: TranslationCreate JSON
	Permission: loc_mlg:translations_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.create_translation(TranslationCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.get("/translations/<translation_id>")
def api_get_translation(translation_id: str):
	"""
	Get a translation entry.
	---
	GET /loc-mlg/api/v1/translations/<translation_id>
	Permission: loc_mlg:translations
	"""
	try:
		return _ok(_run(_SERVICE.get_translation(_tenant(), translation_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.post("/translations/<translation_id>/submit")
def api_submit_translation(translation_id: str):
	"""
	Submit a translation for review.
	---
	POST /loc-mlg/api/v1/translations/<translation_id>/submit
	Permission: loc_mlg:translations_write
	"""
	try:
		return _ok(_run(_SERVICE.submit_translation_for_review(_tenant(), translation_id, actor_id=_actor())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mlg_api.post("/translations/<translation_id>/approve")
def api_approve_translation(translation_id: str):
	"""
	Approve a translation.
	---
	POST /loc-mlg/api/v1/translations/<translation_id>/approve
	Body: {reviewer_id}
	Permission: loc_mlg:translations_review
	"""
	body = request.get_json(silent=True) or {}
	reviewer_id = body.get("reviewer_id", _actor())
	try:
		return _ok(_run(_SERVICE.approve_translation(_tenant(), translation_id, reviewer_id=reviewer_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mlg_api.post("/translations/<translation_id>/publish")
def api_publish_translation(translation_id: str):
	"""
	Publish an approved translation.
	---
	POST /loc-mlg/api/v1/translations/<translation_id>/publish
	Permission: loc_mlg:translations_write
	"""
	try:
		return _ok(_run(_SERVICE.publish_translation(_tenant(), translation_id, actor_id=_actor())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mlg_api.get("/translations/lookup")
def api_lookup_translation():
	"""
	Look up a published translation by key + language.
	---
	GET /loc-mlg/api/v1/translations/lookup
	Query: translation_key, target_language, namespace
	Permission: loc_mlg:translations
	"""
	try:
		key = request.args["translation_key"]
		lang = request.args["target_language"]
		ns = request.args.get("namespace", "default")
		result = _run(_SERVICE.lookup_translation(_tenant(), key, lang, namespace=ns))
		if result is None:
			return _err("translation not found", 404)
		return _ok(result)
	except KeyError as exc:
		return _err(f"missing parameter: {exc}", 400)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Formatting Rules ----

@mlg_api.get("/formatting")
def api_list_formatting():
	"""
	List formatting rules.
	---
	GET /loc-mlg/api/v1/formatting
	Query: locale_id
	Permission: loc_mlg:formatting
	"""
	try:
		return _ok(_run(_SERVICE.list_formatting_rules(_tenant(), locale_id=request.args.get("locale_id"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.post("/formatting")
def api_create_formatting():
	"""
	Configure formatting rules.
	---
	POST /loc-mlg/api/v1/formatting
	Body: FormattingRuleCreate JSON
	Permission: loc_mlg:formatting_write
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.configure_formatting(FormattingRuleCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.get("/formatting/<rule_id>")
def api_get_formatting(rule_id: str):
	"""
	Get a formatting rule.
	---
	GET /loc-mlg/api/v1/formatting/<rule_id>
	Permission: loc_mlg:formatting
	"""
	try:
		return _ok(_run(_SERVICE.get_formatting_rule(_tenant(), rule_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Terminology ----

@mlg_api.get("/terminology")
def api_list_terminology():
	"""
	List terminology entries.
	---
	GET /loc-mlg/api/v1/terminology
	Query: language, domain
	Permission: loc_mlg:terminology
	"""
	try:
		return _ok(_run(_SERVICE.list_terminology(_tenant(), language=request.args.get("language"), domain=request.args.get("domain"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.post("/terminology")
def api_create_terminology():
	"""
	Add a terminology entry.
	---
	POST /loc-mlg/api/v1/terminology
	Body: TerminologyCreate JSON
	Permission: loc_mlg:terminology
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.add_terminology(TerminologyCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.get("/terminology/search")
def api_search_terminology():
	"""
	Search terminology.
	---
	GET /loc-mlg/api/v1/terminology/search
	Query: q, language
	Permission: loc_mlg:terminology
	"""
	try:
		query = request.args.get("q", "")
		return _ok(_run(_SERVICE.search_terminology(_tenant(), query, language=request.args.get("language"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Agents ----

@mlg_api.get("/agents")
def api_list_agents():
	"""
	List MLG agents.
	---
	GET /loc-mlg/api/v1/agents
	Permission: loc_mlg:admin
	"""
	try:
		return _ok(_run(_SERVICE.list_agents(_tenant())))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_api.post("/agents")
def api_create_agent():
	"""
	Register an MLG agent.
	---
	POST /loc-mlg/api/v1/agents
	Body: MlgAgentCreate JSON
	Permission: loc_mlg:admin
	"""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant()
	try:
		return _ok(_run(_SERVICE.register_agent(MlgAgentCreate.model_validate(body), actor_id=_actor())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Dashboard ----

@mlg_api.get("/dashboard")
def api_dashboard():
	"""
	MLG dashboard summary.
	---
	GET /loc-mlg/api/v1/dashboard
	Permission: loc_mlg:view
	"""
	try:
		return _ok(_run(_SERVICE.dashboard_summary(_tenant())))
	except PermissionError as exc:
		return _err(str(exc), 403)


# ---- Audit Events ----

@mlg_api.get("/audit-events")
def api_audit_events():
	"""
	Audit event log.
	---
	GET /loc-mlg/api/v1/audit-events
	Query: limit
	Permission: loc_mlg:admin
	"""
	try:
		return _ok(_run(_SERVICE.list_audit_events(_tenant(), limit=int(request.args.get("limit", 50)))))
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Process-local helper functions ---

def service() -> MultiLanguageLocalisationService:
	return _SERVICE


def configure_locale(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.configure_locale(LocaleConfigCreate.model_validate(payload))).model_dump(mode="json")


def create_translation(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.create_translation(TranslationCreate.model_validate(payload))).model_dump(mode="json")


def add_terminology(payload: dict) -> dict:
	payload.setdefault("tenant_id", "default")
	return _run(_SERVICE.add_terminology(TerminologyCreate.model_validate(payload))).model_dump(mode="json")


def dashboard(payload: dict) -> dict:
	return _run(_SERVICE.dashboard_summary(payload.get("tenant_id", "default")))
