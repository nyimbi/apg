"""Orange Money Web Payment & Cashout API Connector.

Implements APG's BaseConnector ABC for Orange Money — the fintech arm of the
Orange Group, dominant across Francophone West Africa. Supports web-payment
initiation (redirect flow), payment status polling, cashout (B2C), balance
enquiry, and subscriber validation.

OAuth2 client-credentials tokens are cached with a 60-second pre-expiry buffer.
Each market requires a separate merchant key which is passed as a configuration
parameter.

Reference:
    https://developer.orange.com/apis/om-webpay-prod/getting-started
    https://developer.orange.com/apis/omoney-cashout/getting-started

Configuration via environment variables or OrangeConfiguration:
    ORANGE_CLIENT_ID      OAuth2 client ID
    ORANGE_CLIENT_SECRET  OAuth2 client secret
    ORANGE_MERCHANT_KEY   Market-specific merchant key (X-Auth-Key header)
    ORANGE_COUNTRY        ISO alpha-2 code: CI, SN, CM, ML, BF, MG, NE
    ORANGE_ENV            "sandbox" | "production" (default: sandbox)
    ORANGE_CALLBACK_URL_BASE  Base URL for Orange Money callbacks
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import httpx
from pydantic import Field

from ..base_connector import BaseConnector, ConnectorConfiguration

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent.parent.parent))
    from capabilities.common.reliability.circuit_breaker import get_circuit_breaker as _get_cb, CircuitOpenError
    _HAS_CB = True
except Exception:
    _HAS_CB = False
    CircuitOpenError = RuntimeError

_log = logging.getLogger(__name__)

_TIMEOUT = 30.0

_SANDBOX_BASE = "https://api.orange.com/orange-money-webpay/dev/v1"
_PRODUCTION_BASE = "https://api.orange.com/orange-money-webpay/v1"
_AUTH_URL = "https://api.orange.com/oauth/v3/token"


class OrangeConfiguration(ConnectorConfiguration):
	"""Configuration for the Orange Money connector.

	Markets: CI, SN, CM, ML, BF, MG, NE.
	"""

	client_id: str = Field(..., description="Orange API OAuth2 client ID")
	client_secret: str = Field(..., description="Orange API OAuth2 client secret")
	merchant_key: str = Field(..., description="Market-specific merchant key (X-Auth-Key)")
	country: str = Field(..., description="ISO alpha-2 country code, e.g. CI")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str = Field(default="", description="Base URL for Orange Money callbacks")


class OrangeConnector(BaseConnector):
	"""Orange Money connector (Web Payment + Cashout).

	Supports:
	  - initiate_payment    — create a web-payment session (redirect flow)
	  - check_payment_status — poll a payment by pay_token
	  - cashout             — disburse money to a mobile wallet (B2C)
	  - get_balance         — query merchant account balance
	  - check_user          — validate that a phone is a registered Orange Money user
	"""

	def __init__(self, config: OrangeConfiguration) -> None:
		super().__init__(config)
		self._config: OrangeConfiguration = config
		self._base_url = (
			_SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		)
		self._access_token: str = ""
		self._token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None
		# Dedicated auth client — different base URL from the API
		self._auth_client: httpx.AsyncClient | None = None

	# ── BaseConnector abstract methods ────────────────────────────────────────

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=_TIMEOUT,
			headers={"Content-Type": "application/json"},
		)
		self._auth_client = httpx.AsyncClient(timeout=_TIMEOUT)
		await self._refresh_token()
		_log.info(
			"Orange Money connector connected (%s / %s)",
			self._config.environment,
			self._config.country,
		)

	async def _disconnect(self) -> None:
		import asyncio
		clients = [c for c in [self._client, self._auth_client] if c is not None]
		await asyncio.gather(*[c.aclose() for c in clients], return_exceptions=True)
		self._client = None
		self._auth_client = None
		self._access_token = ""
		self._token_expires_at = 0.0

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		handlers: dict[str, Any] = {
			"initiate_payment": self._initiate_payment,
			"check_payment_status": self._check_payment_status,
			"cashout": self._cashout,
			"get_balance": self._get_balance,
			"check_user": self._check_user,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(
				f"Unknown Orange Money operation: {operation!r}. Valid: {list(handlers)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._refresh_token()
			return bool(self._access_token)
		except Exception:
			return False

	# ── Public operation methods ──────────────────────────────────────────────

	async def initiate_payment(
		self,
		amount: int,
		phone: str,
		order_id: str,
		return_url: str,
		cancel_url: str = "",
		notif_url: str = "",
		lang: str = "fr",
	) -> dict[str, Any]:
		"""Create an Orange Money web payment session.

		Returns a pay_token and payment_url. Redirect the user to payment_url
		to complete the transaction on Orange's hosted page.

		Args:
			amount:     Amount in the market's minor currency unit (integer)
			phone:      Subscriber MSISDN (local or international)
			order_id:   Merchant order reference (unique per transaction)
			return_url: URL Orange redirects back to after payment
			cancel_url: URL on user cancel (defaults to return_url)
			notif_url:  Server callback URL for payment notification
			lang:       UI language — "fr" (default) or "en"
		"""
		return await self._execute_operation("initiate_payment", {
			"amount": amount,
			"phone": phone,
			"order_id": order_id,
			"return_url": return_url,
			"cancel_url": cancel_url or return_url,
			"notif_url": notif_url or f"{self._config.callback_url_base}/orange/payment/notify",
			"lang": lang,
		})

	async def check_payment_status(self, pay_token: str) -> dict[str, Any]:
		"""Poll Orange Money payment status using the pay_token from initiation."""
		return await self._execute_operation("check_payment_status", {"pay_token": pay_token})

	async def cashout(
		self,
		amount: int,
		phone: str,
		pin: str,
		order_id: str,
		description: str = "Cashout",
	) -> dict[str, Any]:
		"""Disburse (cashout) money to a mobile wallet (B2C).

		Args:
			amount:      Amount in the market's minor currency unit
			phone:       Recipient MSISDN
			pin:         Merchant cashout PIN (configured on the portal)
			order_id:    Merchant reference
			description: Transaction description
		"""
		return await self._execute_operation("cashout", {
			"amount": amount,
			"phone": phone,
			"pin": pin,
			"order_id": order_id,
			"description": description,
		})

	async def get_balance(self) -> dict[str, Any]:
		"""Query the Orange Money merchant account balance."""
		return await self._execute_operation("get_balance", {})

	async def check_user(self, phone: str) -> dict[str, Any]:
		"""Validate that a phone number is a registered Orange Money user."""
		return await self._execute_operation("check_user", {"phone": phone})

	# ── Private implementation ────────────────────────────────────────────────

	async def _refresh_token(self) -> None:
		if time.time() < self._token_expires_at - 60:
			return
		creds = base64.b64encode(
			f"{self._config.client_id}:{self._config.client_secret}".encode()
		).decode()
		auth_client = self._auth_client or httpx.AsyncClient(timeout=_TIMEOUT)
		resp = await auth_client.post(
			_AUTH_URL,
			data={"grant_type": "client_credentials"},
			headers={
				"Authorization": f"Basic {creds}",
				"Content-Type": "application/x-www-form-urlencoded",
			},
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		data = resp.json()
		self._access_token = data["access_token"]
		self._token_expires_at = time.time() + int(data.get("expires_in", 3600))
		_log.debug("Orange Money OAuth token refreshed")

	def _auth_header(self) -> dict[str, str]:
		return {
			"Authorization": f"Bearer {self._access_token}",
			"X-Auth-Key": self._config.merchant_key,
			"X-Lang": "fr",
			"X-Country": self._config.country,
		}

	async def _initiate_payment(
		self,
		amount: int,
		phone: str,
		order_id: str,
		return_url: str,
		cancel_url: str,
		notif_url: str,
		lang: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"merchant_key": self._config.merchant_key,
			"currency": "ORA",  # Orange internal token — server maps to local currency
			"order_id": order_id,
			"amount": str(amount),
			"return_url": return_url,
			"cancel_url": cancel_url,
			"notif_url": notif_url,
			"lang": lang,
			"reference": phone,
		}
		resp = await self._client.post(
			"/webpayment",
			json=payload,
			headers={**self._auth_header(), "Content-Type": "application/json"},
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _check_payment_status(self, pay_token: str) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			f"/webpayment/{pay_token}",
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _cashout(
		self,
		amount: int,
		phone: str,
		pin: str,
		order_id: str,
		description: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"merchant_key": self._config.merchant_key,
			"msisdn": phone,
			"pin": pin,
			"amount": str(amount),
			"order_id": order_id,
			"description": description,
		}
		resp = await self._client.post(
			"/cashin",
			json=payload,
			headers={**self._auth_header(), "Content-Type": "application/json"},
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _get_balance(self) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			"/balance",
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _check_user(self, phone: str) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			f"/checkuser/{phone}",
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()


def orange_connector_from_env(tenant_id: str, user_id: str = "system") -> OrangeConnector:
	"""Construct OrangeConnector from environment variables.

	Required env vars:
	    ORANGE_CLIENT_ID, ORANGE_CLIENT_SECRET, ORANGE_MERCHANT_KEY, ORANGE_COUNTRY

	Optional:
	    ORANGE_ENV, ORANGE_CALLBACK_URL_BASE
	"""
	config = OrangeConfiguration(
		name="Orange Money",
		tenant_id=tenant_id,
		user_id=user_id,
		client_id=os.environ["ORANGE_CLIENT_ID"],
		client_secret=os.environ["ORANGE_CLIENT_SECRET"],
		merchant_key=os.environ["ORANGE_MERCHANT_KEY"],
		country=os.environ["ORANGE_COUNTRY"],
		environment=os.environ.get("ORANGE_ENV", "sandbox"),
		callback_url_base=os.environ.get("ORANGE_CALLBACK_URL_BASE", ""),
	)
	return OrangeConnector(config)
