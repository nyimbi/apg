"""
Airtel Money Africa — Production-Quality APG Connector.

Implements APG's BaseConnector ABC for Airtel Africa's payment platform.
Covers C2B collections (request_payment), B2C disbursements (send_money),
balance enquiry (check_balance), and transaction status polling
(transaction_status) across all Airtel Africa markets.

OAuth2 client-credentials tokens are cached with a 60-second pre-expiry
refresh buffer. Every external call wraps httpx errors in ConnectorError
so callers get a single, predictable exception type.

Supported markets (AIRTEL_COUNTRY):
	KE  Kenya         KES
	UG  Uganda        UGX
	TZ  Tanzania      TZS
	RW  Rwanda        RWF
	ZM  Zambia        ZMW

Reference API docs:
	https://developers.airtel.africa/documentation

Environment variables (required):
	AIRTEL_CLIENT_ID        OAuth2 client ID issued by Airtel Africa developer portal
	AIRTEL_CLIENT_SECRET    OAuth2 client secret
	AIRTEL_COUNTRY          ISO alpha-2 market code (KE | UG | TZ | RW | ZM)
	AIRTEL_CURRENCY         ISO 4217 currency code (KES | UGX | TZS | RWF | ZMW)

Environment variables (optional):
	AIRTEL_ENV              "sandbox" | "production"  (default: sandbox)
	AIRTEL_CALLBACK_URL_BASE  Base URL for async result callbacks

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
Company: Datacraft — www.datacraft.co.ke
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from pydantic import Field, model_validator

from .base_connector import BaseConnector, ConnectorConfiguration, ConnectorStatus

_log = logging.getLogger(__name__)

# ── Module-level constants / env-var names ────────────────────────────────────

AIRTEL_ENV_CLIENT_ID		= "AIRTEL_CLIENT_ID"
AIRTEL_ENV_CLIENT_SECRET	= "AIRTEL_CLIENT_SECRET"
AIRTEL_ENV_COUNTRY			= "AIRTEL_COUNTRY"
AIRTEL_ENV_CURRENCY			= "AIRTEL_CURRENCY"
AIRTEL_ENV_ENVIRONMENT		= "AIRTEL_ENV"
AIRTEL_ENV_CALLBACK_BASE	= "AIRTEL_CALLBACK_URL_BASE"

_SANDBOX_BASE		= "https://openapiuat.airtel.africa"
_PRODUCTION_BASE	= "https://openapi.airtel.africa"

_DEFAULT_TIMEOUT	= 30.0
_TOKEN_REFRESH_BUFFER = 60  # seconds before expiry to pre-refresh

SUPPORTED_MARKETS: dict[str, str] = {
	"KE": "KES",
	"UG": "UGX",
	"TZ": "TZS",
	"RW": "RWF",
	"ZM": "ZMW",
}


class ConnectorError(Exception):
	"""Raised on any unrecoverable error from the Airtel Money API.

	Wraps httpx transport errors, non-2xx HTTP responses, and API-level
	error codes so callers never need to catch httpx internals.
	"""

	def __init__(self, message: str, *, status_code: int | None = None, body: Any = None) -> None:
		super().__init__(message)
		self.status_code = status_code
		self.body = body

	def __repr__(self) -> str:
		return f"ConnectorError({self.args[0]!r}, status_code={self.status_code!r})"


# ── Configuration model ───────────────────────────────────────────────────────

class AirtelMoneyConfiguration(ConnectorConfiguration):
	"""Pydantic v2 configuration for AirtelMoneyConnector.

	Credentials are typically injected from environment variables via
	`airtel_money_connector_from_env()`.
	"""

	client_id: str = Field(..., description="Airtel OAuth2 client ID")
	client_secret: str = Field(..., description="Airtel OAuth2 client secret")
	country: str = Field(..., description="ISO alpha-2 market code, e.g. KE")
	currency: str = Field(..., description="ISO 4217 currency code, e.g. KES")
	# Override base field to restrict to Airtel-valid values
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str = Field(
		default="",
		description="Base URL for Airtel async result callbacks",
	)

	@model_validator(mode="after")
	def _validate_market(self) -> "AirtelMoneyConfiguration":
		country = self.country.upper()
		if country not in SUPPORTED_MARKETS:
			raise ValueError(
				f"Unsupported Airtel market: {country!r}. "
				f"Supported: {sorted(SUPPORTED_MARKETS)}"
			)
		return self


# ── Connector ─────────────────────────────────────────────────────────────────

class AirtelMoneyConnector(BaseConnector):
	"""Production-quality Airtel Money Africa connector.

	Operations
	----------
	send_money(amount, msisdn, reference)
		B2C disbursement — push funds to a subscriber MSISDN.
	request_payment(amount, msisdn, callback_url)
		C2B collection — prompt a subscriber to authorise a debit.
	check_balance()
		Query the Airtel Money merchant wallet balance.
	transaction_status(tx_id)
		Poll the status of any previously initiated transaction.

	All public methods raise ConnectorError on failure.
	"""

	def __init__(self, config: AirtelMoneyConfiguration) -> None:
		super().__init__(config)
		self._cfg: AirtelMoneyConfiguration = config
		self._base_url = (
			_SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		)
		self._access_token: str = ""
		self._token_expires_at: float = 0.0
		self._http: httpx.AsyncClient | None = None

	# ── BaseConnector abstract implementations ────────────────────────────────

	async def _connect(self) -> None:
		"""Open the shared httpx session and obtain the first OAuth2 token."""
		self._http = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=_DEFAULT_TIMEOUT,
			headers={
				"Content-Type": "application/json",
				"Accept": "application/json",
			},
		)
		await self._ensure_token()
		_log.info(
			"AirtelMoneyConnector connected env=%s country=%s",
			self._cfg.environment,
			self._cfg.country,
		)

	async def _disconnect(self) -> None:
		"""Close the httpx session and clear cached credentials."""
		if self._http is not None:
			await self._http.aclose()
			self._http = None
		self._access_token = ""
		self._token_expires_at = 0.0
		_log.info("AirtelMoneyConnector disconnected")

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		"""Dispatch named operation to the appropriate private handler."""
		_handlers: dict[str, Any] = {
			"send_money":			self._send_money,
			"request_payment":		self._request_payment,
			"check_balance":		self._check_balance,
			"transaction_status":	self._transaction_status,
		}
		handler = _handlers.get(operation)
		if handler is None:
			raise ConnectorError(
				f"Unknown Airtel Money operation: {operation!r}. "
				f"Valid operations: {sorted(_handlers)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		"""Health check: attempt token refresh; success means API is reachable."""
		try:
			await self._ensure_token(force=True)
			return bool(self._access_token)
		except ConnectorError:
			return False

	# ── Public high-level API ─────────────────────────────────────────────────

	async def send_money(
		self,
		amount: str | int | float,
		msisdn: str,
		reference: str,
		transaction_id: str = "",
	) -> dict[str, Any]:
		"""Disburse funds to a subscriber (B2C payout).

		Args:
			amount:         Amount in the configured currency.
			msisdn:         Recipient MSISDN (local or international format).
			reference:      Merchant reference / order ID.
			transaction_id: Optional idempotency key; auto-generated if omitted.

		Returns:
			Airtel API response dict.

		Raises:
			ConnectorError: on any transport or API-level failure.
		"""
		if not transaction_id:
			import uuid
			transaction_id = str(uuid.uuid4())
		return await self._execute_operation("send_money", {
			"amount": str(amount),
			"msisdn": msisdn,
			"reference": reference,
			"transaction_id": transaction_id,
		})

	async def request_payment(
		self,
		amount: str | int | float,
		msisdn: str,
		callback_url: str = "",
		reference: str = "",
		transaction_id: str = "",
	) -> dict[str, Any]:
		"""Initiate a C2B payment request (collect from subscriber).

		Sends a push prompt to the subscriber's handset asking them to
		authorise the debit with their Airtel Money PIN.

		Args:
			amount:         Amount in the configured currency.
			msisdn:         Subscriber MSISDN to charge.
			callback_url:   Override the default callback URL for this transaction.
			reference:      Merchant order reference shown to subscriber.
			transaction_id: Optional idempotency key; auto-generated if omitted.

		Returns:
			Airtel API response dict.

		Raises:
			ConnectorError: on any transport or API-level failure.
		"""
		if not transaction_id:
			import uuid
			transaction_id = str(uuid.uuid4())
		cb = callback_url or f"{self._cfg.callback_url_base}/airtel/payment/callback"
		return await self._execute_operation("request_payment", {
			"amount": str(amount),
			"msisdn": msisdn,
			"reference": reference or transaction_id,
			"transaction_id": transaction_id,
			"callback_url": cb,
		})

	async def check_balance(self) -> dict[str, Any]:
		"""Query the Airtel Money merchant wallet balance.

		Returns:
			Airtel API response dict containing balance and currency.

		Raises:
			ConnectorError: on any transport or API-level failure.
		"""
		return await self._execute_operation("check_balance", {})

	async def transaction_status(self, tx_id: str) -> dict[str, Any]:
		"""Poll the status of a previously initiated transaction.

		Args:
			tx_id: The transaction ID (idempotency key) used when initiating.

		Returns:
			Airtel API response dict with current transaction status.

		Raises:
			ConnectorError: on any transport or API-level failure.
		"""
		return await self._execute_operation("transaction_status", {"tx_id": tx_id})

	# ── Private HTTP helpers ──────────────────────────────────────────────────

	async def _ensure_token(self, *, force: bool = False) -> None:
		"""Obtain or refresh the OAuth2 access token.

		Skips the network call if the cached token has more than
		`_TOKEN_REFRESH_BUFFER` seconds of remaining TTL, unless `force=True`.
		"""
		if not force and time.time() < self._token_expires_at - _TOKEN_REFRESH_BUFFER:
			return

		# Use a transient client if _http is not yet initialised (e.g. during
		# health check before _connect completes)
		client = self._http or httpx.AsyncClient(
			base_url=self._base_url, timeout=_DEFAULT_TIMEOUT
		)
		try:
			resp = await client.post(
				"/auth/oauth2/token",
				json={
					"client_id": self._cfg.client_id,
					"client_secret": self._cfg.client_secret,
					"grant_type": "client_credentials",
				},
				headers={"Content-Type": "application/json"},
				timeout=_DEFAULT_TIMEOUT,
			)
		except httpx.TransportError as exc:
			raise ConnectorError(
				f"Airtel OAuth2 token request failed (transport): {exc}"
			) from exc

		if resp.status_code != 200:
			raise ConnectorError(
				f"Airtel OAuth2 token request failed — HTTP {resp.status_code}",
				status_code=resp.status_code,
				body=resp.text,
			)

		data = resp.json()
		self._access_token = data.get("access_token", "")
		if not self._access_token:
			raise ConnectorError(
				"Airtel OAuth2 response did not contain access_token",
				status_code=resp.status_code,
				body=data,
			)
		self._token_expires_at = time.time() + int(data.get("expires_in", 3600))
		_log.debug(
			"AirtelMoneyConnector: OAuth2 token refreshed (expires_in=%s)",
			data.get("expires_in"),
		)

	def _auth_headers(self) -> dict[str, str]:
		"""Return the set of headers required on every authenticated request."""
		return {
			"Authorization": f"Bearer {self._access_token}",
			"X-Country": self._cfg.country.upper(),
			"X-Currency": self._cfg.currency.upper(),
		}

	async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""POST helper — refreshes token, executes request, raises ConnectorError."""
		assert self._http is not None, "_post called before _connect()"
		await self._ensure_token()
		try:
			resp = await self._http.post(
				path,
				json=payload,
				headers=self._auth_headers(),
				timeout=_DEFAULT_TIMEOUT,
			)
		except httpx.TransportError as exc:
			raise ConnectorError(f"Airtel API POST {path} transport error: {exc}") from exc
		except Exception as exc:
			raise ConnectorError(f"Airtel API POST {path} unexpected error: {exc}") from exc

		if resp.status_code >= 400:
			raise ConnectorError(
				f"Airtel API POST {path} — HTTP {resp.status_code}",
				status_code=resp.status_code,
				body=resp.text,
			)
		return resp.json()

	async def _get(self, path: str) -> dict[str, Any]:
		"""GET helper — refreshes token, executes request, raises ConnectorError."""
		assert self._http is not None, "_get called before _connect()"
		await self._ensure_token()
		try:
			resp = await self._http.get(
				path,
				headers=self._auth_headers(),
				timeout=_DEFAULT_TIMEOUT,
			)
		except httpx.TransportError as exc:
			raise ConnectorError(f"Airtel API GET {path} transport error: {exc}") from exc
		except Exception as exc:
			raise ConnectorError(f"Airtel API GET {path} unexpected error: {exc}") from exc

		if resp.status_code >= 400:
			raise ConnectorError(
				f"Airtel API GET {path} — HTTP {resp.status_code}",
				status_code=resp.status_code,
				body=resp.text,
			)
		return resp.json()

	# ── Private operation implementations ────────────────────────────────────

	async def _send_money(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		transaction_id: str,
	) -> dict[str, Any]:
		"""B2C disbursement implementation — POST /standard/v1/disbursements/"""
		payload = {
			"payee": {
				"msisdn": msisdn,
			},
			"reference": reference,
			"pin": "",  # Server-side PIN flow — never send the PIN client-side in production
			"transaction": {
				"amount": amount,
				"id": transaction_id,
				"type": "B2C",
			},
		}
		try:
			return await self._post("/standard/v1/disbursements/", payload)
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(
				f"send_money({msisdn}, {amount}) failed: {exc}"
			) from exc

	async def _request_payment(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		transaction_id: str,
		callback_url: str,
	) -> dict[str, Any]:
		"""C2B collection implementation — POST /merchant/v2/payments/"""
		payload = {
			"reference": reference,
			"subscriber": {
				"country": self._cfg.country.upper(),
				"currency": self._cfg.currency.upper(),
				"msisdn": msisdn,
			},
			"transaction": {
				"amount": amount,
				"country": self._cfg.country.upper(),
				"currency": self._cfg.currency.upper(),
				"id": transaction_id,
			},
		}
		# Only include redirect URL if provided — some markets don't support it
		if callback_url:
			payload["transaction"]["redirect_url"] = callback_url

		try:
			return await self._post("/merchant/v2/payments/", payload)
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(
				f"request_payment({msisdn}, {amount}) failed: {exc}"
			) from exc

	async def _check_balance(self) -> dict[str, Any]:
		"""Merchant wallet balance — GET /standard/v1/users/balance"""
		try:
			return await self._get("/standard/v1/users/balance")
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(f"check_balance() failed: {exc}") from exc

	async def _transaction_status(self, tx_id: str) -> dict[str, Any]:
		"""Transaction status poll — GET /standard/v1/payments/{id}"""
		try:
			return await self._get(f"/standard/v1/payments/{tx_id}")
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(f"transaction_status({tx_id!r}) failed: {exc}") from exc


# ── Env-factory ───────────────────────────────────────────────────────────────

def airtel_money_connector_from_env(
	tenant_id: str,
	user_id: str = "system",
) -> AirtelMoneyConnector:
	"""Construct AirtelMoneyConnector from environment variables.

	Required env vars:
		AIRTEL_CLIENT_ID, AIRTEL_CLIENT_SECRET, AIRTEL_COUNTRY, AIRTEL_CURRENCY

	Optional env vars:
		AIRTEL_ENV               (default: sandbox)
		AIRTEL_CALLBACK_URL_BASE (default: "")

	Raises:
		KeyError:   if a required env var is missing.
		ValueError: if AIRTEL_COUNTRY is not in SUPPORTED_MARKETS.
	"""
	config = AirtelMoneyConfiguration(
		name="Airtel Money",
		tenant_id=tenant_id,
		user_id=user_id,
		client_id=os.environ[AIRTEL_ENV_CLIENT_ID],
		client_secret=os.environ[AIRTEL_ENV_CLIENT_SECRET],
		country=os.environ[AIRTEL_ENV_COUNTRY],
		currency=os.environ[AIRTEL_ENV_CURRENCY],
		environment=os.environ.get(AIRTEL_ENV_ENVIRONMENT, "sandbox"),
		callback_url_base=os.environ.get(AIRTEL_ENV_CALLBACK_BASE, ""),
	)
	return AirtelMoneyConnector(config)


__all__ = [
	"AirtelMoneyConnector",
	"AirtelMoneyConfiguration",
	"ConnectorError",
	"airtel_money_connector_from_env",
	"SUPPORTED_MARKETS",
	"AIRTEL_ENV_CLIENT_ID",
	"AIRTEL_ENV_CLIENT_SECRET",
	"AIRTEL_ENV_COUNTRY",
	"AIRTEL_ENV_CURRENCY",
	"AIRTEL_ENV_ENVIRONMENT",
	"AIRTEL_ENV_CALLBACK_BASE",
]
