"""Flask Blueprint REST API for ussd_eng capability."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import UssdEngService

_log = logging.getLogger(__name__)

bp = Blueprint("ussd_eng", __name__, url_prefix="/api/ussd/eng")
_svc: UssdEngService | None = None


def get_service() -> UssdEngService:
	global _svc
	if _svc is None:
		_svc = UssdEngService()
	return _svc


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask route."""
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				fut = pool.submit(asyncio.run, coro)
				return fut.result()
		return loop.run_until_complete(coro)
	except Exception:
		return asyncio.run(coro)


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	result = _run(get_service().health_check())
	return jsonify(result), 200


# ── Gateway endpoints ─────────────────────────────────────────────────────────

@bp.get("/gateways")
def list_gateways():
	result = _run(get_service().list_gateways(tenant_id=_tenant()))
	return jsonify(result), 200


@bp.post("/gateways")
def create_gateway():
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().create_gateway(
			name=data["name"],
			gateway_type=data["gateway_type"],
			service_code=data["service_code"],
			tenant_id=_tenant(),
			api_key=data.get("api_key"),
			api_secret=data.get("api_secret"),
			username=data.get("username"),
			webhook_url=data.get("webhook_url"),
			environment=data.get("environment", "sandbox"),
			metadata=data.get("metadata"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_gateway error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/gateways/<gateway_id>")
def get_gateway(gateway_id: str):
	try:
		result = _run(get_service().get_gateway(gateway_id, tenant_id=_tenant()))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_gateway error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.put("/gateways/<gateway_id>")
def update_gateway(gateway_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().update_gateway(
			gateway_id=gateway_id,
			tenant_id=_tenant(),
			webhook_url=data.get("webhook_url"),
			environment=data.get("environment"),
			status=data.get("status"),
			metadata=data.get("metadata"),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("update_gateway error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/gateways/<gateway_id>")
def delete_gateway(gateway_id: str):
	try:
		result = _run(get_service().delete_gateway(gateway_id, tenant_id=_tenant()))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_gateway error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Menu endpoints ────────────────────────────────────────────────────────────

@bp.get("/menus")
def list_menus():
	service_code = request.args.get("service_code")
	result = _run(get_service().list_menus(service_code=service_code, tenant_id=_tenant()))
	return jsonify(result), 200


@bp.post("/menus")
def create_menu():
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().create_menu(
			menu_id=data["menu_id"],
			title=data["title"],
			body=data.get("body", ""),
			service_code=data["service_code"],
			tenant_id=_tenant(),
			items=data.get("items"),
			language=data.get("language", "en"),
			is_end_screen=data.get("is_end_screen", False),
			timeout_seconds=data.get("timeout_seconds", 180),
			metadata=data.get("metadata"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_menu error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/menus/<menu_id>")
def get_menu(menu_id: str):
	service_code = request.args.get("service_code", "")
	language = request.args.get("language", "en")
	try:
		result = _run(get_service().get_menu(menu_id, service_code, _tenant(), language))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_menu error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.put("/menus/<menu_id>")
def update_menu(menu_id: str):
	data = request.get_json(force=True) or {}
	service_code = data.get("service_code", request.args.get("service_code", ""))
	language = data.get("language", "en")
	try:
		result = _run(get_service().update_menu(
			menu_id=menu_id,
			service_code=service_code,
			tenant_id=_tenant(),
			language=language,
			title=data.get("title"),
			body=data.get("body"),
			items=data.get("items"),
			is_end_screen=data.get("is_end_screen"),
			timeout_seconds=data.get("timeout_seconds"),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("update_menu error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/menus/<menu_id>")
def delete_menu(menu_id: str):
	service_code = request.args.get("service_code", "")
	language = request.args.get("language", "en")
	try:
		result = _run(get_service().delete_menu(menu_id, service_code, _tenant(), language))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_menu error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Session endpoints ─────────────────────────────────────────────────────────

@bp.get("/sessions")
def list_sessions():
	result = _run(get_service().list_sessions(
		tenant_id=_tenant(),
		phone_number=request.args.get("phone_number"),
		service_code=request.args.get("service_code"),
		session_state=request.args.get("session_state"),
	))
	return jsonify(result), 200


@bp.post("/sessions")
def create_session():
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().create_session(
			phone_number=data["phone_number"],
			service_code=data["service_code"],
			tenant_id=_tenant(),
			gateway=data.get("gateway", "africastalking"),
			language=data.get("language", "en"),
			metadata=data.get("metadata"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_session error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/sessions/<session_id>")
def get_session(session_id: str):
	try:
		result = _run(get_service().get_session(session_id, tenant_id=_tenant()))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_session error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.put("/sessions/<session_id>")
def update_session(session_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().update_session(
			session_id=session_id,
			tenant_id=_tenant(),
			current_menu=data.get("current_menu"),
			language=data.get("language"),
			variables=data.get("variables"),
			status=data.get("status"),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("update_session error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/sessions/<session_id>")
def delete_session(session_id: str):
	try:
		result = _run(get_service().delete_session(session_id, tenant_id=_tenant()))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_session error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── USSD callback endpoint ────────────────────────────────────────────────────

@bp.post("/callback")
def ussd_callback():
	"""Primary USSD callback handler — Africa's Talking and Safaricom."""
	data = request.get_json(force=True) if request.is_json else request.form.to_dict()
	gateway = request.args.get("gateway", "africastalking")
	svc = get_service()
	try:
		if gateway == "africastalking":
			parsed = _run(svc.parse_africastalking_callback(data))
		else:
			parsed = _run(svc.parse_safaricom_callback(data))

		resp = _run(svc.handle_ussd_request(
			session_id=parsed["session_id"],
			service_code=parsed["service_code"],
			phone_number=parsed["phone_number"],
			text=parsed.get("text", ""),
			tenant_id=_tenant(),
			gateway=parsed.get("gateway", gateway),
		))

		if gateway == "africastalking":
			body = _run(svc.format_africastalking_response(resp["response_type"], resp["body"]))
			return body, 200, {"Content-Type": "text/plain"}
		else:
			body = _run(svc.format_safaricom_response(resp["response_type"], resp["body"], resp["session_id"]))
			return jsonify(body), 200

	except Exception as exc:
		_log.error("ussd_callback error: %s", exc)
		if gateway == "africastalking":
			return "END Service error. Please try again.", 200, {"Content-Type": "text/plain"}
		return jsonify({"error": "service_error"}), 500


# ── Analytics endpoint ────────────────────────────────────────────────────────

@bp.get("/analytics")
def session_analytics():
	service_code = request.args.get("service_code")
	result = _run(get_service().get_session_analytics(tenant_id=_tenant(), service_code=service_code))
	return jsonify(result), 200


@bp.get("/analytics/menus")
def menu_analytics():
	service_code = request.args.get("service_code", "")
	result = _run(get_service().get_menu_analytics(service_code=service_code, tenant_id=_tenant()))
	return jsonify(result), 200


@bp.get("/analytics/dropoff")
def drop_off_analysis():
	service_code = request.args.get("service_code", "")
	result = _run(get_service().get_drop_off_analysis(service_code=service_code, tenant_id=_tenant()))
	return jsonify(result), 200


@bp.get("/dashboard")
def dashboard():
	result = _run(get_service().dashboard_summary(tenant_id=_tenant()))
	return jsonify(result), 200


@bp.get("/audit")
def audit_events():
	limit = int(request.args.get("limit", 100))
	result = _run(get_service().get_audit_events(tenant_id=_tenant(), limit=limit))
	return jsonify(result), 200
