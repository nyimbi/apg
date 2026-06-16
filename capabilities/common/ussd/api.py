"""
USSD Engine — Flask API Blueprint.

Endpoints:
  POST /api/common/ussd/session      Africa's Talking webhook
  POST /api/common/ussd/safaricom    Safaricom USSD webhook
  POST /api/common/ussd/flows        Create/register a flow definition
  GET  /api/common/ussd/flows/<id>   Get a flow by ID
  GET  /api/common/ussd/sessions/<id> Get session state

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from __future__ import annotations

import json
import logging
from typing import Any

from flask import Blueprint, Response, jsonify, request

from .models import FlowDefinition, USSDMenu, USSDMenuItem, USSDRequest, USSDResponse
from .service import USSDEngineService

_log = logging.getLogger(__name__)

bp = Blueprint("ussd", __name__, url_prefix="/api/common/ussd")

# Module-level service instance — replaced via init_service() at app startup.
_svc: USSDEngineService = USSDEngineService()


def init_service(svc: USSDEngineService) -> None:
	"""Inject a pre-configured USSDEngineService (call from app factory)."""
	global _svc
	_svc = svc


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a synchronous Flask view."""
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


# ── Africa's Talking webhook ──────────────────────────────────────────────────

@bp.route("/session", methods=["POST"])
def at_session() -> Response:
	"""
	Africa's Talking USSD webhook.

	Accepts form-encoded or JSON body with fields:
	  sessionId, serviceCode, phoneNumber, text, networkCode (optional)

	Returns plain text starting with CON or END.
	"""
	data: dict[str, Any] = {}
	if request.is_json:
		data = request.get_json(force=True) or {}
	else:
		data = request.form.to_dict()

	session_id   = data.get("sessionId") or data.get("session_id", "")
	service_code = data.get("serviceCode") or data.get("service_code", "")
	phone        = data.get("phoneNumber") or data.get("phone_number") or data.get("phone", "")
	text         = data.get("text", "")
	network_code = data.get("networkCode") or data.get("network_code")

	if not session_id or not service_code or not phone:
		return Response("END Missing required fields.", status=400, mimetype="text/plain")

	req = USSDRequest(
		session_id=session_id,
		service_code=service_code,
		msisdn=phone,
		text=text,
		network_code=network_code,
		gateway="africastalking",
		raw=dict(data),
	)

	try:
		resp: USSDResponse = _run(_svc.handle_request(req))
	except Exception as exc:
		_log.error("AT webhook error: %s", exc, exc_info=True)
		return Response("END Service error. Please try again.", status=200, mimetype="text/plain")

	prefix = "CON " if resp.continue_session else "END "
	return Response(prefix + resp.text, status=200, mimetype="text/plain")


# ── Safaricom USSD webhook ────────────────────────────────────────────────────

@bp.route("/safaricom", methods=["POST"])
def safaricom_session() -> Response:
	"""
	Safaricom USSD webhook.

	Accepts JSON body:
	  sessionID, serviceCode, phoneNumber (or msisdn), text (or input)

	Returns JSON: {"sessionID": ..., "responseType": "CON"|"END", "responseMsg": ...}
	"""
	data: dict[str, Any] = request.get_json(force=True) or {}

	session_id   = data.get("sessionId") or data.get("sessionID") or data.get("session_id", "")
	service_code = data.get("serviceCode") or data.get("service_code", "")
	phone        = data.get("msisdn") or data.get("phoneNumber") or data.get("phone", "")
	if phone and not phone.startswith("+"):
		phone = "+" + phone
	text = data.get("input") or data.get("text", "")

	if not session_id or not service_code or not phone:
		body = json.dumps({"sessionID": session_id, "responseType": "END", "responseMsg": "Missing required fields."})
		return Response(body, status=400, mimetype="application/json")

	req = USSDRequest(
		session_id=session_id,
		service_code=service_code,
		msisdn=phone,
		text=text,
		gateway="safaricom",
		raw=dict(data),
	)

	try:
		resp: USSDResponse = _run(_svc.handle_request(req))
	except Exception as exc:
		_log.error("Safaricom webhook error: %s", exc, exc_info=True)
		body = json.dumps({"sessionID": session_id, "responseType": "END", "responseMsg": "Service error."})
		return Response(body, status=200, mimetype="application/json")

	response_type = "CON" if resp.continue_session else "END"
	body = json.dumps({
		"sessionID":    resp.session_id,
		"responseType": response_type,
		"responseMsg":  resp.text,
	})
	return Response(body, status=200, mimetype="application/json")


# ── Flow management ───────────────────────────────────────────────────────────

@bp.route("/flows", methods=["POST"])
def create_flow() -> Response:
	"""
	Register a FlowDefinition.

	Accepts a JSON body matching the FlowDefinition schema.
	Returns {"flow_id": "<id>"} on success.
	"""
	payload = request.get_json(force=True) or {}
	try:
		flow = FlowDefinition.model_validate(payload)
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400

	try:
		flow_id = _run(_svc.create_flow(flow))
	except Exception as exc:
		_log.error("create_flow error: %s", exc, exc_info=True)
		return jsonify({"error": str(exc)}), 500

	return jsonify({"flow_id": flow_id}), 201


@bp.route("/flows/<flow_id>", methods=["GET"])
def get_flow(flow_id: str) -> Response:
	"""Return a flow definition by ID."""
	flow = _svc.get_flow(flow_id)
	if flow is None:
		return jsonify({"error": "not found"}), 404
	return jsonify(flow.model_dump(mode="json")), 200


# ── Session state ─────────────────────────────────────────────────────────────

@bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str) -> Response:
	"""Return current session state."""
	session = _run(_svc.get_session(session_id))
	if session is None:
		return jsonify({"error": "not found"}), 404
	return jsonify(session.model_dump(mode="json")), 200
