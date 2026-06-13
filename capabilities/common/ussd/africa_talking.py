"""
USSD Gateway Adapters — Africa's Talking and Safaricom USSD gateway integration.

Architecture:
  - UssdGatewayAdapter: abstract base with standard interface
  - AfricasTalkingAdapter: normalises AT USSD POST requests → internal format
  - SafaricomUssdAdapter: normalises Safaricom USSD POST requests → internal format

Each adapter exposes:
  handle_incoming(request_data) → UsGatewayResponse
    where request_data is the parsed POST body (dict) from the gateway webhook.

The adapters are stateless — they delegate session logic to the
UssdSessionManager injected at construction time.

'Us' Pydantic model prefix throughout.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_log = logging.getLogger(__name__)


# ── Models ───────────────────────────────────────────────────────────────────

class UsIncomingRequest(BaseModel):
	"""Normalised incoming USSD request (gateway-agnostic internal format)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	session_id: str
	service_code: str
	phone: str
	text: str = ""
	network_code: str | None = None
	gateway: str = "unknown"
	raw: dict[str, Any] = Field(default_factory=dict)


class UsGatewayResponse(BaseModel):
	"""
	Response sent back to the gateway webhook.

	``continue_session=True``  → CON (subscriber sees next screen)
	``continue_session=False`` → END (session terminated)
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	text: str
	continue_session: bool
	session_id: str
	formatted: str = ""   # gateway-specific wire format (populated by adapter)


# ── Abstract base ─────────────────────────────────────────────────────────────

class UssdGatewayAdapter(ABC):
	"""
	Abstract USSD gateway adapter.

	Subclasses implement:
	  - parse(raw_body) → UsIncomingRequest
	  - format_response(response) → str   (gateway wire format)
	  - handle_incoming(raw_body) → UsGatewayResponse

	The session_manager attribute is set at construction; adapters must not
	store per-request mutable state.
	"""

	def __init__(self, session_manager: Any) -> None:
		"""
		Args:
			session_manager: An UssdSessionManager (or compatible duck-type) that
			    exposes start_session() and navigate().
		"""
		self.session_manager = session_manager

	@abstractmethod
	def parse(self, raw_body: dict[str, Any]) -> UsIncomingRequest:
		"""Parse a raw gateway POST body into a normalised UsIncomingRequest."""
		...

	@abstractmethod
	def format_response(self, response: UsGatewayResponse) -> str:
		"""Serialise a UsGatewayResponse to the gateway's expected wire format."""
		...

	@abstractmethod
	async def handle_incoming(self, raw_body: dict[str, Any]) -> UsGatewayResponse:
		"""Full request-response cycle: parse → session logic → format."""
		...

	# ── Shared helpers ────────────────────────────────────────────────────────

	async def _dispatch(self, req: UsIncomingRequest) -> UsGatewayResponse:
		"""
		Core dispatch: look up or create session, call navigate(), build response.
		Shared by all concrete adapters.
		"""
		from capabilities.common.ussd.session_manager import UssdSessionManager  # lazy import

		sm: UssdSessionManager = self.session_manager

		# Africa's Talking sends an empty text on the initial dial-in; both AT
		# and Safaricom re-use the same session_id for subsequent hops.
		existing = await sm.get_session(req.session_id)
		if existing is None:
			await sm.start_session(
				phone=req.phone,
				service_code=req.service_code,
				tenant_id="default",
				metadata={"gateway": req.gateway, "network_code": req.network_code or ""},
			)
			# Override session_id so the gateway's session token is canonical
			# Limitation: in-process store — for distributed deployments inject
			# a Redis-backed session store into UssdSessionManager.

		nav = await sm.navigate(req.session_id, req.text)
		return UsGatewayResponse(
			text=nav.text,
			continue_session=nav.continue_session,
			session_id=req.session_id,
		)


# ── Africa's Talking ───────────────────────────────────────────────────────────

class AfricasTalkingAdapter(UssdGatewayAdapter):
	"""
	Africa's Talking USSD gateway adapter.

	AT sends a POST with form-encoded or JSON body::

		{
		    "sessionId":    "ATxxxxxxx",
		    "serviceCode":  "*123#",
		    "phoneNumber":  "+254700000000",
		    "text":         "1*2",          ← full concatenated input chain
		    "networkCode":  "63902"
		}

	AT expects the response body to be plain text starting with "CON " or "END ".
	"""

	GATEWAY_NAME = "africastalking"

	def parse(self, raw_body: dict[str, Any]) -> UsIncomingRequest:
		session_id = raw_body.get("sessionId") or raw_body.get("session_id", "")
		service_code = raw_body.get("serviceCode") or raw_body.get("service_code", "")
		phone = raw_body.get("phoneNumber") or raw_body.get("phone_number") or raw_body.get("phone", "")
		text = raw_body.get("text", "")
		network_code = raw_body.get("networkCode") or raw_body.get("network_code")

		if not session_id:
			raise ValueError("AT callback missing sessionId")
		if not service_code:
			raise ValueError("AT callback missing serviceCode")
		if not phone:
			raise ValueError("AT callback missing phoneNumber")

		return UsIncomingRequest(
			session_id=session_id,
			service_code=service_code,
			phone=phone,
			text=text,
			network_code=network_code,
			gateway=self.GATEWAY_NAME,
			raw=dict(raw_body),
		)

	def format_response(self, response: UsGatewayResponse) -> str:
		"""Africa's Talking expects 'CON <text>' or 'END <text>'."""
		prefix = "CON " if response.continue_session else "END "
		return prefix + response.text

	async def handle_incoming(self, raw_body: dict[str, Any]) -> UsGatewayResponse:
		"""
		Handle an incoming AT USSD webhook call.

		Parses the body, dispatches through the session manager, formats the
		response in AT wire format, and returns a UsGatewayResponse.

		The caller (Flask/FastAPI view) should return response.formatted as
		the HTTP response body with Content-Type: text/plain.
		"""
		try:
			req = self.parse(raw_body)
		except ValueError as exc:
			_log.warning("AT parse error: %s body=%s", exc, raw_body)
			resp = UsGatewayResponse(
				text="Service error. Please try again.",
				continue_session=False,
				session_id=raw_body.get("sessionId", "unknown"),
			)
			resp.formatted = self.format_response(resp)
			return resp

		resp = await self._dispatch(req)
		resp.formatted = self.format_response(resp)
		_log.debug(
			"AT response: session=%s continue=%s text_len=%d",
			req.session_id, resp.continue_session, len(resp.text),
		)
		return resp


# ── Safaricom USSD ─────────────────────────────────────────────────────────────

class SafaricomUssdAdapter(UssdGatewayAdapter):
	"""
	Safaricom USSD gateway adapter.

	Safaricom USSD POST body (JSON)::

		{
		    "sessionId":   "SAF00001",
		    "serviceCode": "*456#",
		    "msisdn":      "254700000000",   ← no leading '+'
		    "input":       "2",
		    "networkCode": "63902"
		}

	Safaricom expects the response as JSON::

		{
		    "sessionID":    "SAF00001",
		    "responseType": "CON",           ← or "END"
		    "responseMsg":  "<screen text>"
		}
	"""

	GATEWAY_NAME = "safaricom"

	def parse(self, raw_body: dict[str, Any]) -> UsIncomingRequest:
		session_id = raw_body.get("sessionId") or raw_body.get("session_id", "")
		service_code = raw_body.get("serviceCode") or raw_body.get("service_code", "")
		# Safaricom omits the leading '+' on MSISDN
		phone = raw_body.get("msisdn") or raw_body.get("phoneNumber") or raw_body.get("phone", "")
		if phone and not phone.startswith("+"):
			phone = "+" + phone
		text = raw_body.get("input") or raw_body.get("text", "")
		network_code = raw_body.get("networkCode") or raw_body.get("network_code")

		if not session_id:
			raise ValueError("Safaricom callback missing sessionId")
		if not service_code:
			raise ValueError("Safaricom callback missing serviceCode")
		if not phone:
			raise ValueError("Safaricom callback missing msisdn")

		return UsIncomingRequest(
			session_id=session_id,
			service_code=service_code,
			phone=phone,
			text=text,
			network_code=network_code,
			gateway=self.GATEWAY_NAME,
			raw=dict(raw_body),
		)

	def format_response(self, response: UsGatewayResponse) -> str:
		"""Return JSON string in Safaricom's expected response structure."""
		import json
		body = {
			"sessionID": response.session_id,
			"responseType": "CON" if response.continue_session else "END",
			"responseMsg": response.text,
		}
		return json.dumps(body)

	async def handle_incoming(self, raw_body: dict[str, Any]) -> UsGatewayResponse:
		"""
		Handle an incoming Safaricom USSD webhook call.

		The caller should return response.formatted as the HTTP response body
		with Content-Type: application/json.
		"""
		try:
			req = self.parse(raw_body)
		except ValueError as exc:
			_log.warning("Safaricom parse error: %s body=%s", exc, raw_body)
			resp = UsGatewayResponse(
				text="Service error. Please try again.",
				continue_session=False,
				session_id=raw_body.get("sessionId", "unknown"),
			)
			resp.formatted = self.format_response(resp)
			return resp

		resp = await self._dispatch(req)
		resp.formatted = self.format_response(resp)
		_log.debug(
			"Safaricom response: session=%s continue=%s text_len=%d",
			req.session_id, resp.continue_session, len(resp.text),
		)
		return resp
