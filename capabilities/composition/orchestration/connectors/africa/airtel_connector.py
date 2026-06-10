"""Airtel Money Africa Payment API Connector.

Implements APG's BaseConnector ABC for Airtel Money — the payment product of
Airtel Africa, present in 14 countries. Supports collections (request payment),
disbursements, balance enquiry, transaction status, and subscriber KYC.

OAuth2 client-credentials tokens are cached with a 60-second pre-expiry buffer.

Reference:
    https://developers.airtel.africa/documentation

Configuration via environment variables or AirtelConfiguration:
    AIRTEL_CLIENT_ID      OAuth2 client ID
    AIRTEL_CLIENT_SECRET  OAuth2 client secret
    AIRTEL_COUNTRY        ISO alpha-2 market code (e.g. KE, UG, TZ, RW, ZM)
    AIRTEL_CURRENCY       ISO 4217 currency (e.g. KES, UGX, TZS, RWF, ZMW)
    AIRTEL_ENV            "sandbox" | "production" (default: sandbox)
    AIRTEL_CALLBACK_URL_BASE  Base URL for async result callbacks
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from pydantic import Field

from ..base_connector import BaseConnector, ConnectorConfiguration

_log = logging.getLogger(__name__)

_TIMEOUT = 30.0

_SANDBOX_BASE = "https://openapiuat.airtel.africa"
_PRODUCTION_BASE = "https://openapi.airtel.africa"


class AirtelConfiguration(ConnectorConfiguration):
	"""Configuration for the Airtel Money connector.

	Markets: KE, UG, TZ, RW, ZM, MG, CD (and more).
	"""

	client_id: str = Field(..., description="Airtel OAuth2 client ID")
	client_secret: str = Field(..., description="Airtel OAuth2 client secret")
	country: str = Field(..., description="ISO alpha-2 market code, e.g. KE")
	currency: str = Field(..., description="ISO 4217 currency code, e.g. KES")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str = Field(default="", description="Base URL for async callbacks")


class AirtelConnector(BaseConnector):
	"""Airtel Money Africa connector.

	Supports:
	  - request_payment  — collect money from a subscriber (C2B prompt)
	  - get_transaction_status — poll a transaction by ID
	  - disburse         — send money to a subscriber (B2C)
	  - check_balance    — query merchant wallet balance
	  - get_user_info    — subscriber KYC by MSISDN
	"""

	def __init__(self, config: AirtelConfiguration) -> None:
		super().__init__(config)
		self._config: AirtelConfiguration = config
		self._base_url = (
			_SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		)
		self._access_token: str = ""
		self._token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None

	# ── BaseConnector abstract methods ────────────────────────────────────────

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=_TIMEOUT,
			headers={"Content-Type": "application/json"},
		)
		await self._refresh_token()
		_log.info(
			"Airtel Money connector connected (%s / %s)",
			self._config.environment,
			self._config.country,
		)

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None
		self._access_token = ""
		self._token_expires_at = 0.0

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		handlers: dict[str, Any] = {
			"request_payment": self._request_payment,
			"get_transaction_status": self._get_transaction_status,
			"disburse": self._disburse,
			"check_balance": self._check_balance,
			"get_user_info": self._get_user_info,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(
				f"Unknown Airtel Money operation: {operation!r}. Valid: {list(handlers)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._refresh_token()
			return bool(self._access_token)
		except Exception:
			return False

	# ── Public operation methods ──────────────────────────────────────────────

	async def request_payment(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		transaction_id: str,
	) -> dict[str, Any]:
		"""Initiate a C2B payment request (collect from subscriber).

		Args:
			amount:         Amount as a string
			msisdn:         Subscriber MSISDN in local or international format
			reference:      Merchant order reference
			transaction_id: Unique transaction ID (idempotency key)
		"""
		return await self._execute_operation("request_payment", {
			"amount": amount,
			"msisdn": msisdn,
			"reference": reference,
			"transaction_id": transaction_id,
		})

	async def get_transaction_status(self, transaction_id: str) -> dict[str, Any]:
		"""Poll the status of a payment or disbursement by transaction_id."""
		return await self._execute_operation("get_transaction_status", {
			"transaction_id": transaction_id,
		})

	async def disburse(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		transaction_id: str,
	) -> dict[str, Any]:
		"""Disburse money to a subscriber (B2C payout).

		Args:
			amount:         Amount as a string
			msisdn:         Recipient MSISDN
			reference:      Merchant reference
			transaction_id: Unique transaction ID
		"""
		return await self._execute_operation("disburse", {
			"amount": amount,
			"msisdn": msisdn,
			"reference": reference,
			"transaction_id": transaction_id,
		})

	async def check_balance(self) -> dict[str, Any]:
		"""Query the Airtel Money merchant wallet balance."""
		return await self._execute_operation("check_balance", {})

	async def get_user_info(self, msisdn: str) -> dict[str, Any]:
		"""Return KYC information for a registered Airtel Money subscriber."""
		return await self._execute_operation("get_user_info", {"msisdn": msisdn})

	# ── Private implementation ────────────────────────────────────────────────

	async def _refresh_token(self) -> None:
		if time.time() < self._token_expires_at - 60:
			return
		client = self._client or httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)
		resp = await client.post(
			"/auth/oauth2/token",
			json={
				"client_id": self._config.client_id,
				"client_secret": self._config.client_secret,
				"grant_type": "client_credentials",
			},
			headers={"Content-Type": "application/json"},
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		data = resp.json()
		self._access_token = data["access_token"]
		self._token_expires_at = time.time() + int(data.get("expires_in", 3600))
		_log.debug("Airtel Money OAuth token refreshed")

	def _auth_header(self) -> dict[str, str]:
		return {
			"Authorization": f"Bearer {self._access_token}",
			"X-Country": self._config.country,
			"X-Currency": self._config.currency,
		}

	async def _request_payment(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		transaction_id: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"reference": reference,
			"subscriber": {"country": self._config.country, "currency": self._config.currency, "msisdn": msisdn},
			"transaction": {"amount": amount, "country": self._config.country, "currency": self._config.currency, "id": transaction_id},
		}
		resp = await self._client.post(
			"/merchant/v2/payments/",
			json=payload,
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _get_transaction_status(self, transaction_id: str) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			f"/standard/v1/payments/{transaction_id}",
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _disburse(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		transaction_id: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"payee": {"msisdn": msisdn},
			"reference": reference,
			"pin": "",  # Server-side pin flow — PIN not sent client-side in production
			"transaction": {
				"amount": amount,
				"id": transaction_id,
				"type": "B2C",
			},
		}
		resp = await self._client.post(
			"/standard/v1/disbursements/",
			json=payload,
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _check_balance(self) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			"/standard/v1/users/balance",
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _get_user_info(self, msisdn: str) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			f"/standard/v1/users/{msisdn}",
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()


def airtel_connector_from_env(tenant_id: str, user_id: str = "system") -> AirtelConnector:
	"""Construct AirtelConnector from environment variables.

	Required env vars:
	    AIRTEL_CLIENT_ID, AIRTEL_CLIENT_SECRET, AIRTEL_COUNTRY, AIRTEL_CURRENCY

	Optional:
	    AIRTEL_ENV, AIRTEL_CALLBACK_URL_BASE
	"""
	config = AirtelConfiguration(
		name="Airtel Money",
		tenant_id=tenant_id,
		user_id=user_id,
		client_id=os.environ["AIRTEL_CLIENT_ID"],
		client_secret=os.environ["AIRTEL_CLIENT_SECRET"],
		country=os.environ["AIRTEL_COUNTRY"],
		currency=os.environ["AIRTEL_CURRENCY"],
		environment=os.environ.get("AIRTEL_ENV", "sandbox"),
		callback_url_base=os.environ.get("AIRTEL_CALLBACK_URL_BASE", ""),
	)
	return AirtelConnector(config)
