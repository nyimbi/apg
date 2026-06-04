"""Flask Blueprint views for APG Multi-Language & Localisation."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, Response, jsonify, request

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

mlg_views = Blueprint("loc_mlg", __name__, url_prefix="/loc-mlg")
_svc = MultiLanguageLocalisationService()


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
			if permission not in granted and "loc_mlg:admin" not in granted:
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

@mlg_views.get("/dashboard")
@has_access("loc_mlg:view")
def dashboard() -> Response:
	"""MLG dashboard summary."""
	try:
		return _ok(_run(_svc.dashboard_summary(_tenant_id())))
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Locales ---

@mlg_views.get("/locales")
@has_access("loc_mlg:locales")
def list_locales() -> Response:
	"""List configured locales."""
	try:
		is_rtl_raw = request.args.get("is_rtl")
		is_rtl = None if is_rtl_raw is None else is_rtl_raw.lower() == "true"
		return _ok(_run(_svc.list_locales(
			_tenant_id(),
			language=request.args.get("language"),
			is_rtl=is_rtl,
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.post("/locales")
@has_access("loc_mlg:locales_write")
def create_locale() -> Response:
	"""Configure a new locale."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.configure_locale(LocaleConfigCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.get("/locales/<locale_id>")
@has_access("loc_mlg:locales")
def get_locale(locale_id: str) -> Response:
	"""Get a locale configuration."""
	try:
		return _ok(_run(_svc.get_locale(_tenant_id(), locale_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.put("/locales/<locale_id>")
@has_access("loc_mlg:locales_write")
def update_locale(locale_id: str) -> Response:
	"""Update a locale configuration."""
	body = request.get_json(silent=True) or {}
	try:
		return _ok(_run(_svc.update_locale(_tenant_id(), locale_id, LocaleConfigUpdate.model_validate(body), actor_id=_actor_id())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Translations ---

@mlg_views.get("/translations")
@has_access("loc_mlg:translations")
def list_translations() -> Response:
	"""List translation entries."""
	try:
		return _ok(_run(_svc.list_translations(
			_tenant_id(),
			target_language=request.args.get("target_language"),
			content_type=request.args.get("content_type"),
			status=request.args.get("status"),
			namespace=request.args.get("namespace"),
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.post("/translations")
@has_access("loc_mlg:translations_write")
def create_translation() -> Response:
	"""Create a translation entry."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.create_translation(TranslationCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.get("/translations/<translation_id>")
@has_access("loc_mlg:translations")
def get_translation(translation_id: str) -> Response:
	"""Get a translation entry."""
	try:
		return _ok(_run(_svc.get_translation(_tenant_id(), translation_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.post("/translations/<translation_id>/submit")
@has_access("loc_mlg:translations_write")
def submit_translation(translation_id: str) -> Response:
	"""Submit a translation for review."""
	try:
		return _ok(_run(_svc.submit_translation_for_review(_tenant_id(), translation_id, actor_id=_actor_id())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mlg_views.post("/translations/<translation_id>/approve")
@has_access("loc_mlg:translations_review")
def approve_translation(translation_id: str) -> Response:
	"""Approve a translation under review."""
	body = request.get_json(silent=True) or {}
	reviewer_id = body.get("reviewer_id", _actor_id())
	try:
		return _ok(_run(_svc.approve_translation(_tenant_id(), translation_id, reviewer_id=reviewer_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mlg_views.post("/translations/<translation_id>/publish")
@has_access("loc_mlg:translations_write")
def publish_translation(translation_id: str) -> Response:
	"""Publish an approved translation."""
	try:
		return _ok(_run(_svc.publish_translation(_tenant_id(), translation_id, actor_id=_actor_id())))
	except KeyError as exc:
		return _err(str(exc), 404)
	except (AssertionError, PermissionError) as exc:
		return _err(str(exc), 403)


@mlg_views.get("/translations/lookup")
@has_access("loc_mlg:translations")
def lookup_translation() -> Response:
	"""Look up a published translation by key and language."""
	try:
		key = request.args["translation_key"]
		lang = request.args["target_language"]
		ns = request.args.get("namespace", "default")
		result = _run(_svc.lookup_translation(_tenant_id(), key, lang, namespace=ns))
		if result is None:
			return _err("translation not found", 404)
		return _ok(result)
	except KeyError as exc:
		return _err(f"missing parameter: {exc}", 400)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Formatting Rules ---

@mlg_views.get("/formatting")
@has_access("loc_mlg:formatting")
def list_formatting() -> Response:
	"""List formatting rules."""
	try:
		return _ok(_run(_svc.list_formatting_rules(_tenant_id(), locale_id=request.args.get("locale_id"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.post("/formatting")
@has_access("loc_mlg:formatting_write")
def create_formatting() -> Response:
	"""Configure formatting rules for a locale."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.configure_formatting(FormattingRuleCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.get("/formatting/<rule_id>")
@has_access("loc_mlg:formatting")
def get_formatting(rule_id: str) -> Response:
	"""Get a formatting rule."""
	try:
		return _ok(_run(_svc.get_formatting_rule(_tenant_id(), rule_id)))
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Terminology ---

@mlg_views.get("/terminology")
@has_access("loc_mlg:terminology")
def list_terminology() -> Response:
	"""List terminology entries."""
	try:
		return _ok(_run(_svc.list_terminology(
			_tenant_id(),
			language=request.args.get("language"),
			domain=request.args.get("domain"),
		)))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.post("/terminology")
@has_access("loc_mlg:terminology")
def create_terminology() -> Response:
	"""Add a terminology entry."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.add_terminology(TerminologyCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.get("/terminology/search")
@has_access("loc_mlg:terminology")
def search_terminology() -> Response:
	"""Search terminology by term text."""
	query = request.args.get("q", "")
	try:
		return _ok(_run(_svc.search_terminology(_tenant_id(), query, language=request.args.get("language"))))
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Agents ---

@mlg_views.get("/agents")
@has_access("loc_mlg:admin")
def list_agents() -> Response:
	"""List MLG agents."""
	try:
		return _ok(_run(_svc.list_agents(_tenant_id())))
	except PermissionError as exc:
		return _err(str(exc), 403)


@mlg_views.post("/agents")
@has_access("loc_mlg:admin")
def create_agent() -> Response:
	"""Register an MLG agent."""
	body = request.get_json(silent=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return _ok(_run(_svc.register_agent(MlgAgentCreate.model_validate(body), actor_id=_actor_id())), 201)
	except (AssertionError, ValueError) as exc:
		return _err(str(exc), 422)
	except PermissionError as exc:
		return _err(str(exc), 403)


# --- Audit Events ---

@mlg_views.get("/audit-events")
@has_access("loc_mlg:admin")
def audit_events() -> Response:
	"""Recent audit events."""
	try:
		limit = int(request.args.get("limit", 50))
		return _ok(_run(_svc.list_audit_events(_tenant_id(), limit=limit)))
	except PermissionError as exc:
		return _err(str(exc), 403)
