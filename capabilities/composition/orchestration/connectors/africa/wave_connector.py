"""Wave Mobile Money Business Payments API Connector.

Implements APG's BaseConnector ABC for Wave — a rapidly-growing mobile money
provider in Francophone West Africa (Senegal, Côte d'Ivoire, Mali, Burkina Faso,
Guinea). Wave uses a simple API-key (Bearer token) authentication model rather
than OAuth2 flows.

Reference:
    https://www.wave.com/en/business/api/

Configuration via environment variables or WaveConfiguration:
    WAVE_API_KEY    Bearer token from the Wave Business dashboard
    WAVE_ENV        "sandbox" | "live" (default: live — Wave has no sandbox)
    WAVE_CALLBACK_URL_BASE  Base URL for Wave result webhooks
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from pydantic import Field

from ..base_connector import BaseConnector, ConnectorConfiguration

_log = logging.getLogger(__name__)

_TIMEOUT = 30.0

# Wave does not maintain a public sandbox; the live base URL is used for
# both environments. The environment field is preserved for consistency.
_LIVE_BASE = "https://api.wave.com/v1"
_SANDBOX_BASE = "https://api.wave.com/v1"  # same until Wave adds sandbox


class WaveConfiguration(ConnectorConfiguration):
	"""Configuration for the Wave Mobile Money connector.

	Markets: SN, CI, ML, BF, GN.
	"""

	api_key: str = Field(..., description="Wave API key (Bearer token from the dashboard)")
	environment: str = Field(default="live", pattern="^(sandbox|live)$")
	callback_url_base: str = Field(default="", description="Base URL for Wave result webhooks")


class WaveConnector(BaseConnector):
	"""Wave Mobile Money connector.

	Supports:
	  - send_money        — B2C push: send money to a Wave wallet
	  - check_out         — hosted-page checkout for C2B collections
	  - get_balance       — query business wallet balance
	  - get_transaction   — retrieve a transaction by Wave reference
	  - refund            — refund a completed send_money transaction
	"""

	def __init__(self, config: WaveConfiguration) -> None:
		super().__init__(config)
		self._config: WaveConfiguration = config
		self._base_url = _SANDBOX_BASE if config.environment == "sandbox" else _LIVE_BASE
		self._client: httpx.AsyncClient | None = None

	# ── BaseConnector abstract methods ────────────────────────────────────────

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=_TIMEOUT,
			headers={
				"Authorization": f"Bearer {self._config.api_key}",
				"Content-Type": "application/json",
			},
		)
		# Validate connectivity with a balance check
		await self._get_balance()
		_log.info("Wave connector connected (%s)", self._config.environment)

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		handlers: dict[str, Any] = {
			"send_money": self._send_money,
			"check_out": self._check_out,
			"get_balance": self._get_balance,
			"get_transaction": self._get_transaction,
			"refund": self._refund,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(
				f"Unknown Wave operation: {operation!r}. Valid: {list(handlers)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			result = await self._get_balance()
			return "balance" in result or "amount" in result
		except Exception:
			return False

	# ── Public operation methods ──────────────────────────────────────────────

	async def send_money(
		self,
		amount: str,
		phone: str,
		client_reference: str,
		currency: str = "XOF",
	) -> dict[str, Any]:
		"""Send money directly to a Wave wallet (B2C push).

		Args:
			amount:           Amount in the smallest currency unit as string
			phone:            Recipient Wave phone number (international format)
			client_reference: Merchant idempotency key (unique per transaction)
			currency:         ISO 4217 code — XOF (default) or XAF
		"""
		return await self._execute_operation("send_money", {
			"amount": amount,
			"phone": phone,
			"client_reference": client_reference,
			"currency": currency,
		})

	async def check_out(
		self,
		amount: str,
		currency: str,
		client_reference: str,
		success_url: str,
		error_url: str,
	) -> dict[str, Any]:
		"""Create a Wave hosted-page checkout session (C2B collection).

		Returns a wave_launch_url and id. Redirect the user to wave_launch_url to
		complete the payment in the Wave app or via USSD.

		Args:
			amount:           Amount in the smallest currency unit as string
			currency:         ISO 4217 code (XOF or XAF)
			client_reference: Merchant idempotency key
			success_url:      Redirect URL on successful payment
			error_url:        Redirect URL on error or cancellation
		"""
		return await self._execute_operation("check_out", {
			"amount": amount,
			"currency": currency,
			"client_reference": client_reference,
			"success_url": success_url,
			"error_url": error_url,
		})

	async def get_balance(self) -> dict[str, Any]:
		"""Query the Wave business wallet balance."""
		return await self._execute_operation("get_balance", {})

	async def get_transaction(self, wave_ref: str) -> dict[str, Any]:
		"""Retrieve a transaction by its Wave reference ID."""
		return await self._execute_operation("get_transaction", {"wave_ref": wave_ref})

	async def refund(self, wave_ref: str, reason: str = "merchant_refund") -> dict[str, Any]:
		"""Refund a completed send_money transaction.

		Args:
			wave_ref: The Wave transaction reference to refund
			reason:   Reason code — "merchant_refund" (default) or "duplicate"
		"""
		return await self._execute_operation("refund", {"wave_ref": wave_ref, "reason": reason})

	# ── Private implementation ────────────────────────────────────────────────

	def _headers(self) -> dict[str, str]:
		return {
			"Authorization": f"Bearer {self._config.api_key}",
			"Content-Type": "application/json",
		}

	async def _send_money(
		self,
		amount: str,
		phone: str,
		client_reference: str,
		currency: str,
	) -> dict[str, Any]:
		payload = {
			"currency": currency,
			"receive_amount": amount,
			"mobile": phone,
			"client_reference": client_reference,
		}
		resp = await self._client.post(
			"/payout",
			json=payload,
			headers=self._headers(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _check_out(
		self,
		amount: str,
		currency: str,
		client_reference: str,
		success_url: str,
		error_url: str,
	) -> dict[str, Any]:
		payload = {
			"currency": currency,
			"amount": amount,
			"client_reference": client_reference,
			"success_url": success_url,
			"error_url": error_url,
		}
		resp = await self._client.post(
			"/checkout/sessions",
			json=payload,
			headers=self._headers(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _get_balance(self) -> dict[str, Any]:
		resp = await self._client.get(
			"/balance",
			headers=self._headers(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _get_transaction(self, wave_ref: str) -> dict[str, Any]:
		resp = await self._client.get(
			f"/payout/{wave_ref}",
			headers=self._headers(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _refund(self, wave_ref: str, reason: str) -> dict[str, Any]:
		payload = {"wave_ref": wave_ref, "reason": reason}
		resp = await self._client.post(
			"/payout/refund",
			json=payload,
			headers=self._headers(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()


def wave_connector_from_env(tenant_id: str, user_id: str = "system") -> WaveConnector:
	"""Construct WaveConnector from environment variables.

	Required env vars:
	    WAVE_API_KEY

	Optional:
	    WAVE_ENV, WAVE_CALLBACK_URL_BASE
	"""
	config = WaveConfiguration(
		name="Wave Mobile Money",
		tenant_id=tenant_id,
		user_id=user_id,
		api_key=os.environ["WAVE_API_KEY"],
		environment=os.environ.get("WAVE_ENV", "live"),
		callback_url_base=os.environ.get("WAVE_CALLBACK_URL_BASE", ""),
	)
	return WaveConnector(config)
