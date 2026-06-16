"""
Wave Mobile Money — Production-Quality APG Connector.

Implements APG's BaseConnector ABC for Wave's payment platform across
West Africa. Wave uses a simple API-key bearer token (no OAuth2 flow).

Supported markets (WAVE_COUNTRY):
	SN  Senegal     XOF
	CI  Côte d'Ivoire XOF
	ML  Mali        XOF
	BF  Burkina Faso XOF
	GN  Guinea      GNF

Reference: https://docs.wave.com/

Environment variables (required):
	WAVE_API_KEY     Bearer API key from Wave developer portal
	WAVE_COUNTRY     ISO alpha-2 market code

Environment variables (optional):
	WAVE_ENV                 "sandbox" | "production"  (default: sandbox)
	WAVE_CALLBACK_URL_BASE   Base URL for async result callbacks
	WAVE_CURRENCY            ISO 4217 code (auto-derived from WAVE_COUNTRY if omitted)

Author: Nyimbi Odero
Company: Datacraft — www.datacraft.co.ke
Copyright: © 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import httpx
from pydantic import Field, model_validator

from .base_connector import BaseConnector, ConnectorConfiguration, ConnectorStatus

_log = logging.getLogger(__name__)

WAVE_ENV_API_KEY		= "WAVE_API_KEY"
WAVE_ENV_COUNTRY		= "WAVE_COUNTRY"
WAVE_ENV_ENVIRONMENT	= "WAVE_ENV"
WAVE_ENV_CALLBACK_BASE	= "WAVE_CALLBACK_URL_BASE"
WAVE_ENV_CURRENCY		= "WAVE_CURRENCY"

_SANDBOX_BASE		= "https://api.sandbox.wave.com"
_PRODUCTION_BASE	= "https://api.wave.com"
_DEFAULT_TIMEOUT	= 30.0

SUPPORTED_MARKETS: dict[str, str] = {
	"SN": "XOF",
	"CI": "XOF",
	"ML": "XOF",
	"BF": "XOF",
	"GN": "GNF",
}


class ConnectorError(Exception):
	"""Unrecoverable Wave API error."""
	def __init__(self, message: str, *, status_code: int | None = None, body: Any = None) -> None:
		super().__init__(message)
		self.status_code = status_code
		self.body = body


class WaveConfiguration(ConnectorConfiguration):
	api_key: str			= Field(..., description="Wave Bearer API key")
	country: str			= Field(..., description="ISO alpha-2 market code, e.g. SN")
	currency: str			= Field(default="")
	environment: str		= Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str	= Field(default="")

	@model_validator(mode="after")
	def _derive_currency(self) -> "WaveConfiguration":
		country = self.country.upper()
		if country not in SUPPORTED_MARKETS:
			raise ValueError(f"Unsupported Wave market: {country!r}. Supported: {sorted(SUPPORTED_MARKETS)}")
		if not self.currency:
			self.currency = SUPPORTED_MARKETS[country]
		return self


class WaveConnector(BaseConnector):
	"""Production-quality Wave mobile money connector.

	Operations
	----------
	send_money(amount, msisdn, reference)
		B2C payout — push funds to a Wave subscriber.
	request_payment(amount, msisdn, callback_url)
		C2B — create a checkout session for subscriber to authorise.
	check_balance()
		Wallet balance for the merchant account.
	transaction_status(tx_id)
		Poll a previously initiated transaction.
	"""

	def __init__(self, config: WaveConfiguration) -> None:
		super().__init__(config)
		self._cfg = config
		self._base = _SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		self._http: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._http = httpx.AsyncClient(
			base_url=self._base,
			timeout=_DEFAULT_TIMEOUT,
			headers={
				"Authorization": f"Bearer {self._cfg.api_key}",
				"Content-Type": "application/json",
			},
		)
		_log.info("WaveConnector connected env=%s country=%s", self._cfg.environment, self._cfg.country)

	async def _disconnect(self) -> None:
		if self._http:
			await self._http.aclose()
			self._http = None

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers: dict[str, Any] = {
			"send_money":			self._send_money,
			"request_payment":		self._request_payment,
			"check_balance":		self._check_balance,
			"transaction_status":	self._transaction_status,
		}
		handler = handlers.get(operation)
		if not handler:
			raise ConnectorError(f"Unknown Wave operation: {operation!r}. Valid: {sorted(handlers)}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			result = await self.check_balance()
			return "balance" in result or "amount" in result
		except ConnectorError:
			return False

	# ── Public API ────────────────────────────────────────────────────────────

	async def send_money(self, amount: str | int | float, msisdn: str, reference: str = "") -> dict[str, Any]:
		tx_id = str(uuid.uuid4())
		return await self._execute_operation("send_money", {
			"amount": str(amount), "msisdn": msisdn,
			"reference": reference or tx_id, "transaction_id": tx_id,
		})

	async def request_payment(self, amount: str | int | float, msisdn: str, callback_url: str = "") -> dict[str, Any]:
		tx_id = str(uuid.uuid4())
		return await self._execute_operation("request_payment", {
			"amount": str(amount), "msisdn": msisdn,
			"callback_url": callback_url or f"{self._cfg.callback_url_base}/wave/payment/callback",
			"transaction_id": tx_id,
		})

	async def check_balance(self) -> dict[str, Any]:
		return await self._execute_operation("check_balance", {})

	async def transaction_status(self, tx_id: str) -> dict[str, Any]:
		return await self._execute_operation("transaction_status", {"tx_id": tx_id})

	# ── Implementations ───────────────────────────────────────────────────────

	async def _send_money(self, amount: str, msisdn: str, reference: str, transaction_id: str) -> dict[str, Any]:
		assert self._http is not None
		payload = {
			"currency": self._cfg.currency,
			"receive_amount": amount,
			"mobile": msisdn,
			"name": reference,
			"client_reference": transaction_id,
		}
		try:
			resp = await self._http.post("/v1/payout", json=payload, timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"Wave send_money transport error: {exc}") from exc
		if resp.status_code >= 400:
			raise ConnectorError(f"Wave send_money HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return resp.json()

	async def _request_payment(self, amount: str, msisdn: str, callback_url: str, transaction_id: str) -> dict[str, Any]:
		assert self._http is not None
		payload = {
			"currency": self._cfg.currency,
			"amount": amount,
			"error_url": callback_url,
			"success_url": callback_url,
			"client_reference": transaction_id,
		}
		try:
			resp = await self._http.post("/v1/checkout/sessions", json=payload, timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"Wave request_payment transport error: {exc}") from exc
		if resp.status_code >= 400:
			raise ConnectorError(f"Wave request_payment HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return resp.json()

	async def _check_balance(self) -> dict[str, Any]:
		assert self._http is not None
		try:
			resp = await self._http.get("/v1/balance", timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"Wave check_balance transport error: {exc}") from exc
		if resp.status_code != 200:
			raise ConnectorError(f"Wave check_balance HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return resp.json()

	async def _transaction_status(self, tx_id: str) -> dict[str, Any]:
		assert self._http is not None
		try:
			resp = await self._http.get(f"/v1/transactions/{tx_id}", timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"Wave transaction_status transport error: {exc}") from exc
		if resp.status_code != 200:
			raise ConnectorError(f"Wave transaction_status HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return resp.json()


def wave_connector_from_env(tenant_id: str, user_id: str = "system") -> WaveConnector:
	"""Construct WaveConnector from environment variables."""
	config = WaveConfiguration(
		name="Wave",
		tenant_id=tenant_id,
		user_id=user_id,
		api_key=os.environ[WAVE_ENV_API_KEY],
		country=os.environ[WAVE_ENV_COUNTRY],
		currency=os.environ.get(WAVE_ENV_CURRENCY, ""),
		environment=os.environ.get(WAVE_ENV_ENVIRONMENT, "sandbox"),
		callback_url_base=os.environ.get(WAVE_ENV_CALLBACK_BASE, ""),
	)
	return WaveConnector(config)


__all__ = [
	"WaveConnector", "WaveConfiguration", "ConnectorError",
	"wave_connector_from_env", "SUPPORTED_MARKETS",
]
