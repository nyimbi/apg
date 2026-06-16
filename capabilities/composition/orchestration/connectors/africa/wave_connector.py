"""Wave Mobile Money Business Payments API Connector.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025

Implements APG's BaseConnector ABC for Wave — a rapidly-growing mobile money
provider in Francophone West Africa. Wave uses Bearer-token (API key)
authentication; there is no OAuth2 flow.

Supported markets: SN (Senegal), CI (Côte d'Ivoire), ML (Mali),
                   BF (Burkina Faso), GN (Guinea).

Reference: https://www.wave.com/en/business/api/

Environment variables
---------------------
Required:
    WAVE_API_KEY            Bearer token from the Wave Business dashboard.

Optional:
    WAVE_ENV                "sandbox" | "live"  (default: live — Wave has no
                            public sandbox; this field is kept for consistency).
    WAVE_CALLBACK_URL_BASE  Base URL for Wave result webhooks.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from pydantic import Field

from ..base_connector import BaseConnector, ConnectorConfiguration, ConnectorStatus

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable names (documented as module-level constants)
# ---------------------------------------------------------------------------
ENV_WAVE_API_KEY           = "WAVE_API_KEY"
ENV_WAVE_ENV               = "WAVE_ENV"
ENV_WAVE_CALLBACK_URL_BASE = "WAVE_CALLBACK_URL_BASE"

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
_LIVE_BASE    = "https://api.wave.com/v1"
# Wave has no public sandbox; keep constant for forward-compatibility.
_SANDBOX_BASE = "https://api.wave.com/v1"

_DEFAULT_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# Supported markets
# ---------------------------------------------------------------------------
WAVE_MARKETS: frozenset[str] = frozenset({"SN", "CI", "ML", "BF", "GN"})


# ---------------------------------------------------------------------------
# Connector-specific error
# ---------------------------------------------------------------------------
class ConnectorError(Exception):
	"""Raised when any Wave API call fails (network, HTTP 4xx/5xx, parse)."""

	def __init__(self, message: str, status_code: int | None = None, raw: Any = None) -> None:
		super().__init__(message)
		self.status_code = status_code
		self.raw = raw

	def __repr__(self) -> str:
		return f"ConnectorError(status={self.status_code!r}, msg={str(self)!r})"


# ---------------------------------------------------------------------------
# Pydantic configuration model
# ---------------------------------------------------------------------------
class WaveConfiguration(ConnectorConfiguration):
	"""Configuration for the Wave Mobile Money connector.

	Markets: SN, CI, ML, BF, GN.
	"""

	api_key: str = Field(
		...,
		description="Wave API key — Bearer token from the Wave Business dashboard. "
		            f"Set via {ENV_WAVE_API_KEY}.",
	)
	environment: str = Field(
		default="live",
		pattern="^(sandbox|live)$",
		description=f"Deployment target. Set via {ENV_WAVE_ENV}. Wave has no public "
		            "sandbox so both values point to the live API.",
	)
	callback_url_base: str = Field(
		default="",
		description=f"Base URL for Wave result webhooks. Set via {ENV_WAVE_CALLBACK_URL_BASE}.",
	)


# ---------------------------------------------------------------------------
# Connector implementation
# ---------------------------------------------------------------------------
class WaveConnector(BaseConnector):
	"""Wave Mobile Money connector.

	Operations
	----------
	send_money(amount, msisdn, reference)
		B2C push — send money directly to a Wave wallet.
	request_payment(amount, msisdn, callback_url)
		C2B pull — create a hosted checkout session and return the launch URL.
	check_balance()
		Query the Wave business wallet balance.
	transaction_status(tx_id)
		Retrieve the status of a transaction by its Wave reference ID.

	All public methods raise ConnectorError on any failure.
	"""

	def __init__(self, config: WaveConfiguration) -> None:
		super().__init__(config)
		self._config: WaveConfiguration = config
		self._base_url: str = (
			_SANDBOX_BASE if config.environment == "sandbox" else _LIVE_BASE
		)
		self._client: httpx.AsyncClient | None = None

	# ── BaseConnector abstract interface ──────────────────────────────────────

	async def _connect(self) -> None:
		"""Open the httpx client and validate credentials with a balance probe."""
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=_DEFAULT_TIMEOUT,
			headers=self._auth_headers(),
		)
		# Smoke-test connectivity — raises ConnectorError on failure
		await self._req_balance()
		_log.info(
			self._log_connector_info(f"Wave connector ready (env={self._config.environment})")
		)

	async def _disconnect(self) -> None:
		"""Close the httpx session and release resources."""
		if self._client is not None:
			await self._client.aclose()
			self._client = None

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		"""Dispatch to the correct private handler by operation name."""
		dispatch: dict[str, Any] = {
			"send_money":        self._req_send_money,
			"request_payment":   self._req_request_payment,
			"check_balance":     self._req_balance,
			"transaction_status": self._req_transaction_status,
		}
		handler = dispatch.get(operation)
		if handler is None:
			raise ValueError(
				f"Unknown Wave operation {operation!r}. Valid: {sorted(dispatch)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		"""Return True when the Wave API is reachable and credentials are valid."""
		try:
			result = await self._req_balance()
			return "balance" in result or "amount" in result
		except Exception:
			return False

	# ── Public named operations ───────────────────────────────────────────────

	async def send_money(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		currency: str = "XOF",
	) -> dict[str, Any]:
		"""Send money to a Wave wallet (B2C push / payout).

		Args:
			amount:    Amount in the smallest currency unit (as string, e.g. "1000").
			msisdn:    Recipient Wave phone number in international format (e.g. "+221701234567").
			reference: Merchant idempotency key — must be unique per transaction.
			currency:  ISO 4217 code. Default "XOF" (UEMOA); use "GNF" for Guinea.

		Returns:
			Wave payout response dict containing at minimum ``id`` and ``status``.

		Raises:
			ConnectorError: On any network, HTTP, or parse failure.
		"""
		return await self.execute_request(
			"send_money",
			{"amount": amount, "msisdn": msisdn, "reference": reference, "currency": currency},
		)

	async def request_payment(
		self,
		amount: str,
		msisdn: str,
		callback_url: str,
		currency: str = "XOF",
	) -> dict[str, Any]:
		"""Create a Wave checkout session (C2B collection / pull payment).

		The caller should redirect the payer to the returned ``wave_launch_url``
		to complete payment inside the Wave app or via USSD.

		Args:
			amount:       Amount in the smallest currency unit (as string).
			msisdn:       Payer's Wave phone number in international format.
			callback_url: Webhook URL that Wave will POST the result to.
			currency:     ISO 4217 code. Default "XOF".

		Returns:
			Wave checkout session dict containing ``id`` and ``wave_launch_url``.

		Raises:
			ConnectorError: On any network, HTTP, or parse failure.
		"""
		return await self.execute_request(
			"request_payment",
			{
				"amount": amount,
				"msisdn": msisdn,
				"callback_url": callback_url,
				"currency": currency,
			},
		)

	async def check_balance(self) -> dict[str, Any]:
		"""Query the Wave business wallet balance.

		Returns:
			Dict containing ``balance`` and ``currency`` fields.

		Raises:
			ConnectorError: On any network, HTTP, or parse failure.
		"""
		return await self.execute_request("check_balance", {})

	async def transaction_status(self, tx_id: str) -> dict[str, Any]:
		"""Retrieve the status of a Wave transaction.

		Args:
			tx_id: Wave transaction reference ID (returned by send_money or
			       request_payment).

		Returns:
			Wave transaction dict containing at minimum ``id`` and ``status``.

		Raises:
			ConnectorError: On any network, HTTP, or parse failure.
		"""
		return await self.execute_request("transaction_status", {"tx_id": tx_id})

	# ── Private HTTP helpers ──────────────────────────────────────────────────

	def _auth_headers(self) -> dict[str, str]:
		return {
			"Authorization": f"Bearer {self._config.api_key}",
			"Content-Type":  "application/json",
			"Accept":         "application/json",
		}

	def _ensure_client(self) -> httpx.AsyncClient:
		if self._client is None:
			raise ConnectorError("Wave client not initialised — call initialize() first.")
		return self._client

	async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""POST *path* with *payload*, raise ConnectorError on failure."""
		client = self._ensure_client()
		try:
			resp = await client.post(
				path,
				json=payload,
				headers=self._auth_headers(),
				timeout=_DEFAULT_TIMEOUT,
			)
		except httpx.TimeoutException as exc:
			raise ConnectorError(f"Wave POST {path} timed out: {exc}") from exc
		except httpx.RequestError as exc:
			raise ConnectorError(f"Wave POST {path} network error: {exc}") from exc

		try:
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			body: Any = None
			try:
				body = resp.json()
			except Exception:
				body = resp.text
			raise ConnectorError(
				f"Wave POST {path} returned HTTP {resp.status_code}",
				status_code=resp.status_code,
				raw=body,
			) from exc

		try:
			return resp.json()
		except Exception as exc:
			raise ConnectorError(
				f"Wave POST {path} returned non-JSON body: {resp.text[:200]}"
			) from exc

	async def _get(self, path: str) -> dict[str, Any]:
		"""GET *path*, raise ConnectorError on failure."""
		client = self._ensure_client()
		try:
			resp = await client.get(
				path,
				headers=self._auth_headers(),
				timeout=_DEFAULT_TIMEOUT,
			)
		except httpx.TimeoutException as exc:
			raise ConnectorError(f"Wave GET {path} timed out: {exc}") from exc
		except httpx.RequestError as exc:
			raise ConnectorError(f"Wave GET {path} network error: {exc}") from exc

		try:
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			body: Any = None
			try:
				body = resp.json()
			except Exception:
				body = resp.text
			raise ConnectorError(
				f"Wave GET {path} returned HTTP {resp.status_code}",
				status_code=resp.status_code,
				raw=body,
			) from exc

		try:
			return resp.json()
		except Exception as exc:
			raise ConnectorError(
				f"Wave GET {path} returned non-JSON body: {resp.text[:200]}"
			) from exc

	# ── Private operation implementations ────────────────────────────────────

	async def _req_send_money(
		self,
		amount: str,
		msisdn: str,
		reference: str,
		currency: str,
	) -> dict[str, Any]:
		"""POST /payout — B2C push to a Wave wallet."""
		payload = {
			"currency":         currency,
			"receive_amount":   amount,
			"mobile":           msisdn,
			"client_reference": reference,
		}
		return await self._post("/payout", payload)

	async def _req_request_payment(
		self,
		amount: str,
		msisdn: str,
		callback_url: str,
		currency: str,
	) -> dict[str, Any]:
		"""POST /checkout/sessions — hosted C2B checkout."""
		payload = {
			"currency":         currency,
			"amount":           amount,
			"mobile":           msisdn,
			"success_url":      callback_url,
			"error_url":        callback_url,
		}
		return await self._post("/checkout/sessions", payload)

	async def _req_balance(self) -> dict[str, Any]:
		"""GET /balance — business wallet balance."""
		return await self._get("/balance")

	async def _req_transaction_status(self, tx_id: str) -> dict[str, Any]:
		"""GET /payout/{tx_id} — transaction status by reference."""
		return await self._get(f"/payout/{tx_id}")


# ---------------------------------------------------------------------------
# Env-factory (used by ConnectorRegistry.from_env)
# ---------------------------------------------------------------------------
def wave_connector_from_env(
	tenant_id: str,
	user_id: str = "system",
) -> WaveConnector:
	"""Construct a WaveConnector from environment variables.

	Required env vars
	-----------------
	WAVE_API_KEY            Wave Bearer API key.

	Optional env vars
	-----------------
	WAVE_ENV                "sandbox" | "live"  (default: "live")
	WAVE_CALLBACK_URL_BASE  Webhook base URL    (default: "")

	Raises:
		KeyError: if WAVE_API_KEY is not set.
	"""
	config = WaveConfiguration(
		name="Wave Mobile Money",
		tenant_id=tenant_id,
		user_id=user_id,
		api_key=os.environ[ENV_WAVE_API_KEY],
		environment=os.environ.get(ENV_WAVE_ENV, "live"),
		callback_url_base=os.environ.get(ENV_WAVE_CALLBACK_URL_BASE, ""),
	)
	return WaveConnector(config)


__all__ = [
	"WaveConnector",
	"WaveConfiguration",
	"ConnectorError",
	"wave_connector_from_env",
	"WAVE_MARKETS",
	"ENV_WAVE_API_KEY",
	"ENV_WAVE_ENV",
	"ENV_WAVE_CALLBACK_URL_BASE",
]
