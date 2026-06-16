"""
Orange Money Web Payment & Cashout API Connector — OrangeMoneyConnector

Implements APG's BaseConnector ABC for Orange Money across Francophone West
Africa. Covers OAuth2 client-credentials token management, send_money (B2C
cashout), request_payment (web-payment initiation), check_balance, and
transaction_status polling.

Supported markets: CI (Côte d'Ivoire), SN (Senegal), CM (Cameroon),
                   ML (Mali), BF (Burkina Faso).

Reference:
    https://developer.orange.com/apis/om-webpay-prod/getting-started
    https://developer.orange.com/apis/omoney-cashout/getting-started

Environment variables (set before instantiating via orange_money_connector_from_env):

    ORANGE_MONEY_CLIENT_ID        Required — OAuth2 client ID
    ORANGE_MONEY_CLIENT_SECRET    Required — OAuth2 client secret
    ORANGE_MONEY_MERCHANT_KEY     Required — market-specific merchant key (X-Auth-Key)
    ORANGE_MONEY_COUNTRY          Required — ISO alpha-2 code: CI | SN | CM | ML | BF
    ORANGE_MONEY_ENV              Optional — "sandbox" | "production" (default: sandbox)
    ORANGE_MONEY_CALLBACK_URL     Optional — default notification callback URL

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
Company: Datacraft
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import httpx
from pydantic import Field

from .base_connector import BaseConnector, ConnectorConfiguration

_log = logging.getLogger(__name__)

# ── Module-level env-var constants ────────────────────────────────────────────

ENV_CLIENT_ID = "ORANGE_MONEY_CLIENT_ID"
ENV_CLIENT_SECRET = "ORANGE_MONEY_CLIENT_SECRET"
ENV_MERCHANT_KEY = "ORANGE_MONEY_MERCHANT_KEY"
ENV_COUNTRY = "ORANGE_MONEY_COUNTRY"
ENV_ENVIRONMENT = "ORANGE_MONEY_ENV"
ENV_CALLBACK_URL = "ORANGE_MONEY_CALLBACK_URL"

# ── API endpoints ─────────────────────────────────────────────────────────────

_AUTH_URL = "https://api.orange.com/oauth/v3/token"
_SANDBOX_BASE = "https://api.orange.com/orange-money-webpay/dev/v1"
_PRODUCTION_BASE = "https://api.orange.com/orange-money-webpay/v1"

# Supported markets
SUPPORTED_MARKETS = {"CI", "SN", "CM", "ML", "BF"}

# ── Error class ───────────────────────────────────────────────────────────────


class ConnectorError(Exception):
	"""Raised when an Orange Money API call fails.

	Wraps the underlying httpx or parsing error so callers don't need to
	import httpx to handle failures.

	Attributes:
		status_code: HTTP status code if available, else None
		upstream:    Original exception
	"""

	def __init__(self, message: str, status_code: int | None = None, upstream: Exception | None = None) -> None:
		super().__init__(message)
		self.status_code = status_code
		self.upstream = upstream


# ── Configuration ─────────────────────────────────────────────────────────────


class OrangeMoneyConfiguration(ConnectorConfiguration):
	"""Pydantic v2 configuration for OrangeMoneyConnector.

	Fields map 1-to-1 with the env vars documented in the module docstring.
	"""

	client_id: str = Field(..., description=f"OAuth2 client ID — env: {ENV_CLIENT_ID}")
	client_secret: str = Field(..., description=f"OAuth2 client secret — env: {ENV_CLIENT_SECRET}")
	merchant_key: str = Field(..., description=f"Market merchant key (X-Auth-Key) — env: {ENV_MERCHANT_KEY}")
	country: str = Field(..., description=f"ISO alpha-2 country code — env: {ENV_COUNTRY}")
	environment: str = Field(
		default="sandbox",
		pattern="^(sandbox|production)$",
		description=f"API environment — env: {ENV_ENVIRONMENT}",
	)
	callback_url: str = Field(
		default="",
		description=f"Default notification URL — env: {ENV_CALLBACK_URL}",
	)


# ── Connector ─────────────────────────────────────────────────────────────────


class OrangeMoneyConnector(BaseConnector):
	"""Orange Money connector — send money, request payment, balance, tx status.

	Operations dispatched through execute_request (inherited) → _execute_operation:

	  send_money(amount, msisdn, note)
	      B2C cashout: push money from the merchant wallet to a subscriber.

	  request_payment(amount, msisdn, callback_url)
	      Web-payment initiation: returns pay_token + payment_url for redirect flow.

	  check_balance()
	      Query current merchant wallet balance.

	  transaction_status(tx_id)
	      Poll the status of a transaction by its pay_token / order_id.

	Supported markets: CI, SN, CM, ML, BF.
	"""

	def __init__(self, config: OrangeMoneyConfiguration) -> None:
		super().__init__(config)
		self._cfg: OrangeMoneyConfiguration = config
		self._base_url: str = (
			_SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		)
		self._access_token: str = ""
		self._token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None
		self._auth_client: httpx.AsyncClient | None = None

		if config.country not in SUPPORTED_MARKETS:
			_log.warning(
				"OrangeMoneyConnector: country %r not in supported markets %s",
				config.country,
				SUPPORTED_MARKETS,
			)

	# ── BaseConnector abstract interface ──────────────────────────────────────

	async def _connect(self) -> None:
		"""Open httpx clients and fetch an initial OAuth2 token."""
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=self._cfg.timeout_seconds,
			headers={"Content-Type": "application/json"},
		)
		self._auth_client = httpx.AsyncClient(timeout=self._cfg.timeout_seconds)
		await self._refresh_token()
		_log.info(
			"OrangeMoneyConnector connected — env=%s country=%s",
			self._cfg.environment,
			self._cfg.country,
		)

	async def _disconnect(self) -> None:
		"""Close both httpx clients and wipe the cached token."""
		import asyncio
		clients = [c for c in [self._client, self._auth_client] if c is not None]
		await asyncio.gather(*[c.aclose() for c in clients], return_exceptions=True)
		self._client = None
		self._auth_client = None
		self._access_token = ""
		self._token_expires_at = 0.0
		_log.info("OrangeMoneyConnector disconnected")

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		"""Dispatch to the matching private handler."""
		_dispatch: dict[str, Any] = {
			"send_money": self._send_money,
			"request_payment": self._request_payment,
			"check_balance": self._check_balance,
			"transaction_status": self._transaction_status,
		}
		handler = _dispatch.get(operation)
		if handler is None:
			raise ConnectorError(
				f"Unknown OrangeMoneyConnector operation: {operation!r}. "
				f"Valid: {list(_dispatch)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		"""Treat a successful token refresh as a liveness signal."""
		try:
			await self._refresh_token()
			return bool(self._access_token)
		except Exception as exc:
			_log.warning("OrangeMoneyConnector health check failed: %s", exc)
			return False

	# ── Public API ────────────────────────────────────────────────────────────

	async def send_money(
		self,
		amount: int,
		msisdn: str,
		note: str = "Transfer",
	) -> dict[str, Any]:
		"""Push money from the merchant wallet to a subscriber (B2C cashout).

		Args:
			amount:  Amount in the market's minor currency unit (integer, e.g. XOF cents).
			msisdn:  Recipient phone number — local or international format.
			note:    Human-readable description shown on the subscriber's receipt.

		Returns:
			Orange Money API response dict with at minimum::

			    {
			        "status": {"status": "SUCCESSFULL" | "FAILED" | ...},
			        "data": {"order_id": "...", "txn_id": "..."},
			    }

		Raises:
			ConnectorError: on HTTP error, network failure, or non-2xx response.
		"""
		return await self.execute_request("send_money", {
			"amount": amount,
			"msisdn": msisdn,
			"note": note,
		})

	async def request_payment(
		self,
		amount: int,
		msisdn: str,
		callback_url: str = "",
	) -> dict[str, Any]:
		"""Initiate a web-payment session (redirect / USSD push flow).

		Args:
			amount:       Amount in the market's minor currency unit.
			msisdn:       Payer phone number.
			callback_url: Server notification URL (overrides config default).

		Returns:
			Dict containing ``pay_token`` and ``payment_url`` for the redirect::

			    {
			        "pay_token": "...",
			        "payment_url": "https://...",
			        "notif_token": "...",
			    }

		Raises:
			ConnectorError: on HTTP error or missing pay_token in response.
		"""
		return await self.execute_request("request_payment", {
			"amount": amount,
			"msisdn": msisdn,
			"callback_url": callback_url or self._cfg.callback_url,
		})

	async def check_balance(self) -> dict[str, Any]:
		"""Query the Orange Money merchant account balance.

		Returns:
			Dict with balance and currency::

			    {"balance": 150000, "currency": "XOF", "country": "CI"}

		Raises:
			ConnectorError: on HTTP or network failure.
		"""
		return await self.execute_request("check_balance", {})

	async def transaction_status(self, tx_id: str) -> dict[str, Any]:
		"""Poll the status of a transaction by pay_token or order_id.

		Args:
			tx_id: The ``pay_token`` returned by request_payment, or the
			       merchant ``order_id`` used in send_money.

		Returns:
			Dict with current status from Orange::

			    {
			        "status": "SUCCESSFULL" | "FAILED" | "PENDING" | "CANCELLED",
			        "txn_id": "...",
			        "amount": "...",
			    }

		Raises:
			ConnectorError: on HTTP or network failure.
		"""
		return await self.execute_request("transaction_status", {"tx_id": tx_id})

	# ── Private implementation ────────────────────────────────────────────────

	async def _refresh_token(self) -> None:
		"""Fetch a new OAuth2 token if current one expires within 60 s."""
		if time.time() < self._token_expires_at - 60:
			return
		creds = base64.b64encode(
			f"{self._cfg.client_id}:{self._cfg.client_secret}".encode()
		).decode()
		auth_client = self._auth_client or httpx.AsyncClient(
			timeout=self._cfg.timeout_seconds
		)
		try:
			resp = await auth_client.post(
				_AUTH_URL,
				data={"grant_type": "client_credentials"},
				headers={
					"Authorization": f"Basic {creds}",
					"Content-Type": "application/x-www-form-urlencoded",
				},
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"Orange Money OAuth token fetch failed: {exc}",
				status_code=exc.response.status_code,
				upstream=exc,
			) from exc
		except httpx.RequestError as exc:
			raise ConnectorError(
				f"Orange Money OAuth token fetch network error: {exc}",
				upstream=exc,
			) from exc
		data = resp.json()
		self._access_token = data["access_token"]
		self._token_expires_at = time.time() + int(data.get("expires_in", 3600))
		_log.debug("OrangeMoneyConnector: OAuth token refreshed")

	def _auth_headers(self) -> dict[str, str]:
		"""Common request headers including Bearer token and market context."""
		return {
			"Authorization": f"Bearer {self._access_token}",
			"X-Auth-Key": self._cfg.merchant_key,
			"X-Country": self._cfg.country,
			"X-Lang": "fr",
			"Content-Type": "application/json",
		}

	async def _send_money(
		self,
		amount: int,
		msisdn: str,
		note: str,
	) -> dict[str, Any]:
		"""B2C cashout implementation."""
		await self._refresh_token()
		assert self._client is not None, "_send_money called before _connect"
		payload = {
			"merchant_key": self._cfg.merchant_key,
			"msisdn": msisdn,
			"amount": str(amount),
			"description": note,
		}
		try:
			resp = await self._client.post(
				"/cashin",
				json=payload,
				headers=self._auth_headers(),
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"send_money failed [{exc.response.status_code}]: {exc.response.text}",
				status_code=exc.response.status_code,
				upstream=exc,
			) from exc
		except httpx.RequestError as exc:
			raise ConnectorError(f"send_money network error: {exc}", upstream=exc) from exc
		return resp.json()

	async def _request_payment(
		self,
		amount: int,
		msisdn: str,
		callback_url: str,
	) -> dict[str, Any]:
		"""Web-payment initiation implementation."""
		await self._refresh_token()
		assert self._client is not None, "_request_payment called before _connect"
		import uuid
		order_id = str(uuid.uuid4())
		payload = {
			"merchant_key": self._cfg.merchant_key,
			"currency": "ORA",
			"order_id": order_id,
			"amount": str(amount),
			"return_url": callback_url,
			"cancel_url": callback_url,
			"notif_url": callback_url,
			"lang": "fr",
			"reference": msisdn,
		}
		try:
			resp = await self._client.post(
				"/webpayment",
				json=payload,
				headers=self._auth_headers(),
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"request_payment failed [{exc.response.status_code}]: {exc.response.text}",
				status_code=exc.response.status_code,
				upstream=exc,
			) from exc
		except httpx.RequestError as exc:
			raise ConnectorError(
				f"request_payment network error: {exc}", upstream=exc
			) from exc
		result = resp.json()
		if "pay_token" not in result:
			raise ConnectorError(
				f"request_payment: pay_token missing in response: {result}"
			)
		return result

	async def _check_balance(self) -> dict[str, Any]:
		"""Merchant balance query implementation."""
		await self._refresh_token()
		assert self._client is not None, "_check_balance called before _connect"
		try:
			resp = await self._client.get(
				"/balance",
				headers=self._auth_headers(),
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"check_balance failed [{exc.response.status_code}]: {exc.response.text}",
				status_code=exc.response.status_code,
				upstream=exc,
			) from exc
		except httpx.RequestError as exc:
			raise ConnectorError(f"check_balance network error: {exc}", upstream=exc) from exc
		return resp.json()

	async def _transaction_status(self, tx_id: str) -> dict[str, Any]:
		"""Transaction status poll implementation."""
		await self._refresh_token()
		assert self._client is not None, "_transaction_status called before _connect"
		try:
			resp = await self._client.get(
				f"/webpayment/{tx_id}",
				headers=self._auth_headers(),
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"transaction_status failed [{exc.response.status_code}]: {exc.response.text}",
				status_code=exc.response.status_code,
				upstream=exc,
			) from exc
		except httpx.RequestError as exc:
			raise ConnectorError(
				f"transaction_status network error: {exc}", upstream=exc
			) from exc
		return resp.json()


# ── Factory ───────────────────────────────────────────────────────────────────


def orange_money_connector_from_env(
	tenant_id: str,
	user_id: str = "system",
) -> OrangeMoneyConnector:
	"""Construct OrangeMoneyConnector entirely from environment variables.

	Required:
	    ORANGE_MONEY_CLIENT_ID, ORANGE_MONEY_CLIENT_SECRET,
	    ORANGE_MONEY_MERCHANT_KEY, ORANGE_MONEY_COUNTRY

	Optional:
	    ORANGE_MONEY_ENV (default: "sandbox"),
	    ORANGE_MONEY_CALLBACK_URL (default: "")

	Raises:
	    KeyError: if a required env var is absent.
	"""
	config = OrangeMoneyConfiguration(
		name="Orange Money",
		tenant_id=tenant_id,
		user_id=user_id,
		client_id=os.environ[ENV_CLIENT_ID],
		client_secret=os.environ[ENV_CLIENT_SECRET],
		merchant_key=os.environ[ENV_MERCHANT_KEY],
		country=os.environ[ENV_COUNTRY],
		environment=os.environ.get(ENV_ENVIRONMENT, "sandbox"),
		callback_url=os.environ.get(ENV_CALLBACK_URL, ""),
	)
	return OrangeMoneyConnector(config)


__all__ = [
	"OrangeMoneyConnector",
	"OrangeMoneyConfiguration",
	"ConnectorError",
	"orange_money_connector_from_env",
	"SUPPORTED_MARKETS",
	"ENV_CLIENT_ID",
	"ENV_CLIENT_SECRET",
	"ENV_MERCHANT_KEY",
	"ENV_COUNTRY",
	"ENV_ENVIRONMENT",
	"ENV_CALLBACK_URL",
]
