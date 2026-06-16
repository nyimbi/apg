"""
MTN Mobile Money (MoMo) Africa — Production-Quality APG Connector.

Implements APG's BaseConnector ABC for MTN's MoMo API across all
sub-Saharan Africa markets. Covers C2B collections (request_payment),
B2C disbursements (send_money), balance enquiry, and transaction status.

MTN MoMo uses a three-part credential scheme:
  - Subscription key (Ocp-Apim-Subscription-Key header)
  - API User UUID (provisioned once per environment)
  - API Key (generated from the API User)
  An OAuth2 Basic Auth token is minted from api_user:api_key.

Supported markets (MTN_COUNTRY):
	NG  Nigeria     NGN
	GH  Ghana       GHS
	UG  Uganda      UGX
	CM  Cameroon    XAF
	CI  Côte d'Ivoire XOF
	ZM  Zambia      ZMW

Reference: https://momodeveloper.mtn.com/docs/services/collection

Environment variables (required):
	MTN_SUBSCRIPTION_KEY    Ocp-Apim-Subscription-Key from developer portal
	MTN_API_USER            UUID of the provisioned API user
	MTN_API_KEY             API key generated for the API user
	MTN_COUNTRY             ISO alpha-2 market code

Environment variables (optional):
	MTN_ENV                 "sandbox" | "production"  (default: sandbox)
	MTN_CALLBACK_URL_BASE   Base URL for async result callbacks
	MTN_CURRENCY            ISO 4217 code (auto-derived from MTN_COUNTRY if omitted)

Author: Nyimbi Odero
Company: Datacraft — www.datacraft.co.ke
Copyright: © 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

import httpx
from pydantic import Field, model_validator

from .base_connector import BaseConnector, ConnectorConfiguration, ConnectorStatus

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MTN_ENV_SUBSCRIPTION_KEY	= "MTN_SUBSCRIPTION_KEY"
MTN_ENV_API_USER			= "MTN_API_USER"
MTN_ENV_API_KEY				= "MTN_API_KEY"
MTN_ENV_COUNTRY				= "MTN_COUNTRY"
MTN_ENV_ENVIRONMENT			= "MTN_ENV"
MTN_ENV_CALLBACK_BASE		= "MTN_CALLBACK_URL_BASE"
MTN_ENV_CURRENCY			= "MTN_CURRENCY"

_SANDBOX_BASE		= "https://sandbox.momodeveloper.mtn.com"
_PRODUCTION_BASE	= "https://proxy.momoapi.mtn.com"

_DEFAULT_TIMEOUT		= 30.0
_TOKEN_REFRESH_BUFFER	= 60

SUPPORTED_MARKETS: dict[str, str] = {
	"NG": "NGN",
	"GH": "GHS",
	"UG": "UGX",
	"CM": "XAF",
	"CI": "XOF",
	"ZM": "ZMW",
}


class ConnectorError(Exception):
	"""Unrecoverable MTN MoMo API error."""
	def __init__(self, message: str, *, status_code: int | None = None, body: Any = None) -> None:
		super().__init__(message)
		self.status_code = status_code
		self.body = body


# ── Configuration ─────────────────────────────────────────────────────────────

class MTNMoMoConfiguration(ConnectorConfiguration):
	subscription_key: str	= Field(..., description="Ocp-Apim-Subscription-Key")
	api_user: str			= Field(..., description="Provisioned API User UUID")
	api_key: str			= Field(..., description="API Key for the API User")
	country: str			= Field(..., description="ISO alpha-2 market code, e.g. NG")
	currency: str			= Field(default="", description="ISO 4217 currency (auto-derived if blank)")
	environment: str		= Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str	= Field(default="")

	@model_validator(mode="after")
	def _derive_currency(self) -> "MTNMoMoConfiguration":
		country = self.country.upper()
		if country not in SUPPORTED_MARKETS:
			raise ValueError(f"Unsupported MTN market: {country!r}. Supported: {sorted(SUPPORTED_MARKETS)}")
		if not self.currency:
			self.currency = SUPPORTED_MARKETS[country]
		return self


# ── Connector ─────────────────────────────────────────────────────────────────

class MTNMoMoConnector(BaseConnector):
	"""Production-quality MTN Mobile Money connector.

	Operations
	----------
	send_money(amount, msisdn, currency, note)
		B2C disbursement via the Disbursement product.
	request_payment(amount, msisdn, callback_url)
		C2B collection via the Collection product.
	check_balance()
		Collection account balance.
	transaction_status(tx_id)
		Poll a previously initiated transaction.
	"""

	def __init__(self, config: MTNMoMoConfiguration) -> None:
		super().__init__(config)
		self._cfg = config
		self._base = _SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		self._access_token: str = ""
		self._token_expires_at: float = 0.0
		self._http: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._http = httpx.AsyncClient(base_url=self._base, timeout=_DEFAULT_TIMEOUT)
		await self._ensure_token()
		_log.info("MTNMoMoConnector connected env=%s country=%s", self._cfg.environment, self._cfg.country)

	async def _disconnect(self) -> None:
		if self._http:
			await self._http.aclose()
			self._http = None
		self._access_token = ""
		self._token_expires_at = 0.0

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers: dict[str, Any] = {
			"send_money":			self._send_money,
			"request_payment":		self._request_payment,
			"check_balance":		self._check_balance,
			"transaction_status":	self._transaction_status,
		}
		handler = handlers.get(operation)
		if not handler:
			raise ConnectorError(f"Unknown MTN MoMo operation: {operation!r}. Valid: {sorted(handlers)}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._ensure_token(force=True)
			return bool(self._access_token)
		except ConnectorError:
			return False

	# ── Public API ────────────────────────────────────────────────────────────

	async def send_money(self, amount: str | int | float, msisdn: str, currency: str = "", note: str = "") -> dict[str, Any]:
		tx_id = str(uuid.uuid4())
		return await self._execute_operation("send_money", {
			"amount": str(amount), "msisdn": msisdn,
			"currency": currency or self._cfg.currency,
			"note": note, "transaction_id": tx_id,
		})

	async def request_payment(self, amount: str | int | float, msisdn: str, callback_url: str = "") -> dict[str, Any]:
		tx_id = str(uuid.uuid4())
		return await self._execute_operation("request_payment", {
			"amount": str(amount), "msisdn": msisdn,
			"callback_url": callback_url or f"{self._cfg.callback_url_base}/mtn/payment/callback",
			"transaction_id": tx_id,
		})

	async def check_balance(self) -> dict[str, Any]:
		return await self._execute_operation("check_balance", {})

	async def transaction_status(self, tx_id: str) -> dict[str, Any]:
		return await self._execute_operation("transaction_status", {"tx_id": tx_id})

	# ── Token management ──────────────────────────────────────────────────────

	async def _ensure_token(self, *, force: bool = False) -> None:
		if not force and time.time() < self._token_expires_at - _TOKEN_REFRESH_BUFFER:
			return
		client = self._http or httpx.AsyncClient(base_url=self._base, timeout=_DEFAULT_TIMEOUT)
		try:
			resp = await client.post(
				"/collection/token/",
				auth=(self._cfg.api_user, self._cfg.api_key),
				headers={
					"Ocp-Apim-Subscription-Key": self._cfg.subscription_key,
					"Content-Type": "application/json",
				},
			)
		except httpx.TransportError as exc:
			raise ConnectorError(f"MTN OAuth2 transport error: {exc}") from exc
		if resp.status_code != 200:
			raise ConnectorError(f"MTN OAuth2 failed — HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		data = resp.json()
		self._access_token = data.get("access_token", "")
		if not self._access_token:
			raise ConnectorError("MTN OAuth2 response missing access_token", body=data)
		self._token_expires_at = time.time() + int(data.get("expires_in", 3600))

	def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
		h = {
			"Authorization": f"Bearer {self._access_token}",
			"Ocp-Apim-Subscription-Key": self._cfg.subscription_key,
			"X-Target-Environment": "sandbox" if self._cfg.environment == "sandbox" else "mtncameroon",
			"Content-Type": "application/json",
		}
		if extra:
			h.update(extra)
		return h

	# ── Operation implementations ─────────────────────────────────────────────

	async def _send_money(self, amount: str, msisdn: str, currency: str, note: str, transaction_id: str) -> dict[str, Any]:
		assert self._http is not None
		await self._ensure_token()
		payload = {
			"amount": amount, "currency": currency,
			"externalId": transaction_id,
			"payee": {"partyIdType": "MSISDN", "partyId": msisdn},
			"payerMessage": note, "payeeNote": note,
		}
		try:
			resp = await self._http.post(
				"/disbursement/v1_0/transfer",
				json=payload, headers=self._headers({"X-Reference-Id": transaction_id}),
				timeout=_DEFAULT_TIMEOUT,
			)
		except httpx.TransportError as exc:
			raise ConnectorError(f"MTN send_money transport error: {exc}") from exc
		if resp.status_code not in (200, 202):
			raise ConnectorError(f"MTN send_money HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return {"transaction_id": transaction_id, "status": "pending", "raw": resp.text or "{}"}

	async def _request_payment(self, amount: str, msisdn: str, callback_url: str, transaction_id: str) -> dict[str, Any]:
		assert self._http is not None
		await self._ensure_token()
		payload = {
			"amount": amount, "currency": self._cfg.currency,
			"externalId": transaction_id,
			"payer": {"partyIdType": "MSISDN", "partyId": msisdn},
			"payerMessage": "Payment request", "payeeNote": "Thank you",
		}
		hdrs = self._headers({"X-Reference-Id": transaction_id})
		if callback_url:
			hdrs["X-Callback-Url"] = callback_url
		try:
			resp = await self._http.post("/collection/v1_0/requesttopay", json=payload, headers=hdrs, timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"MTN request_payment transport error: {exc}") from exc
		if resp.status_code not in (200, 202):
			raise ConnectorError(f"MTN request_payment HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return {"transaction_id": transaction_id, "status": "pending"}

	async def _check_balance(self) -> dict[str, Any]:
		assert self._http is not None
		await self._ensure_token()
		try:
			resp = await self._http.get("/collection/v1_0/account/balance", headers=self._headers(), timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"MTN check_balance transport error: {exc}") from exc
		if resp.status_code != 200:
			raise ConnectorError(f"MTN check_balance HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return resp.json()

	async def _transaction_status(self, tx_id: str) -> dict[str, Any]:
		assert self._http is not None
		await self._ensure_token()
		try:
			resp = await self._http.get(f"/collection/v1_0/requesttopay/{tx_id}", headers=self._headers(), timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"MTN transaction_status transport error: {exc}") from exc
		if resp.status_code != 200:
			raise ConnectorError(f"MTN transaction_status HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return resp.json()


# ── Factory ───────────────────────────────────────────────────────────────────

def mtn_momo_connector_from_env(tenant_id: str, user_id: str = "system") -> MTNMoMoConnector:
	"""Construct MTNMoMoConnector from environment variables."""
	config = MTNMoMoConfiguration(
		name="MTN MoMo",
		tenant_id=tenant_id,
		user_id=user_id,
		subscription_key=os.environ[MTN_ENV_SUBSCRIPTION_KEY],
		api_user=os.environ[MTN_ENV_API_USER],
		api_key=os.environ[MTN_ENV_API_KEY],
		country=os.environ[MTN_ENV_COUNTRY],
		currency=os.environ.get(MTN_ENV_CURRENCY, ""),
		environment=os.environ.get(MTN_ENV_ENVIRONMENT, "sandbox"),
		callback_url_base=os.environ.get(MTN_ENV_CALLBACK_BASE, ""),
	)
	return MTNMoMoConnector(config)


__all__ = [
	"MTNMoMoConnector", "MTNMoMoConfiguration", "ConnectorError",
	"mtn_momo_connector_from_env", "SUPPORTED_MARKETS",
]
