"""Equity Bank API Connector.

Equity Bank is East Africa's largest bank by customer count, operating in
Kenya, Uganda, Tanzania, Rwanda, DRC, and South Sudan. The Equity Bank API
enables account inquiry, funds transfer, standing orders, and statement download.

Reference: https://developer.equitybankgroup.com/

Authentication: OAuth2 client_credentials
Base URL: https://api.equitybankgroup.com (production)
          https://sandbox.api.equitybankgroup.com (sandbox)
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

_log = logging.getLogger(__name__)

_SANDBOX_BASE = "https://sandbox.api.equitybankgroup.com"
_PRODUCTION_BASE = "https://api.equitybankgroup.com"


class EquityBankConfiguration(ConnectorConfiguration):
	"""Configuration for Equity Bank API connector."""
	client_id: str = Field(..., description="OAuth2 client ID")
	client_secret: str = Field(..., description="OAuth2 client secret")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	merchant_code: str = Field(default="", description="Equity merchant/business code")


class EquityBankConnector(BaseConnector):
	"""Equity Bank API connector.

	Supports:
	  - Account inquiry (balance, details)
	  - Internal funds transfer (Equity to Equity)
	  - PesaLink transfer (to any Kenyan bank)
	  - Standing order creation
	  - Transaction history/statement
	  - MPESA integration (MPESA to Equity, Equity to MPESA)
	"""

	def __init__(self, config: EquityBankConfiguration) -> None:
		super().__init__(config)
		self._config: EquityBankConfiguration = config
		self._base_url = _SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		self._token: str = ""
		self._token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=self._config.timeout_seconds,
			headers={"Content-Type": "application/json"},
		)
		await self._refresh_token()

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None
		self._token = ""
		self._token_expires_at = 0.0

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers = {
			"account_inquiry": self._account_inquiry,
			"internal_transfer": self._internal_transfer,
			"pesalink_transfer": self._pesalink_transfer,
			"mpesa_to_equity": self._mpesa_to_equity,
			"equity_to_mpesa": self._equity_to_mpesa,
			"standing_order": self._standing_order,
			"transaction_history": self._transaction_history,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(f"Unknown Equity Bank operation: {operation!r}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._refresh_token()
			return bool(self._token)
		except Exception:
			return False

	# ── Public operations ──────────────────────────────────────────────

	async def account_inquiry(self, account_number: str) -> dict[str, Any]:
		"""Get account balance and details."""
		return await self._execute_operation("account_inquiry", {"account_number": account_number})

	async def internal_transfer(
		self,
		from_account: str,
		to_account: str,
		amount: float,
		currency: str = "KES",
		narration: str = "",
	) -> dict[str, Any]:
		"""Transfer between two Equity accounts."""
		return await self._execute_operation("internal_transfer", {
			"from_account": from_account, "to_account": to_account,
			"amount": amount, "currency": currency, "narration": narration,
		})

	async def pesalink_transfer(
		self,
		from_account: str,
		to_account: str,
		bank_code: str,
		amount: float,
		narration: str = "",
	) -> dict[str, Any]:
		"""Transfer to any Kenyan bank via PesaLink."""
		return await self._execute_operation("pesalink_transfer", {
			"from_account": from_account, "to_account": to_account,
			"bank_code": bank_code, "amount": amount, "narration": narration,
		})

	async def transaction_history(
		self,
		account_number: str,
		from_date: str,
		to_date: str,
		limit: int = 50,
	) -> dict[str, Any]:
		"""Retrieve transaction history for an account."""
		return await self._execute_operation("transaction_history", {
			"account_number": account_number, "from_date": from_date,
			"to_date": to_date, "limit": limit,
		})

	# ── Private implementation ─────────────────────────────────────────

	async def _refresh_token(self) -> None:
		if time.time() < self._token_expires_at - 60:
			return
		creds = base64.b64encode(
			f"{self._config.client_id}:{self._config.client_secret}".encode()
		).decode()
		client = self._client or httpx.AsyncClient(base_url=self._base_url, timeout=10)
		resp = await client.post(
			"/v1/oauth/token",
			data={"grant_type": "client_credentials"},
			headers={"Authorization": f"Basic {creds}"},
		)
		resp.raise_for_status()
		data = resp.json()
		self._token = data["access_token"]
		self._token_expires_at = time.time() + int(data.get("expires_in", 3600))

	def _auth_header(self) -> dict[str, str]:
		return {"Authorization": f"Bearer {self._token}"}

	async def _account_inquiry(self, account_number: str) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			f"/v3/accounts/{account_number}",
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _internal_transfer(
		self, from_account: str, to_account: str,
		amount: float, currency: str, narration: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"source": {"accountId": from_account},
			"destination": {"accountId": to_account},
			"transfer": {"currency": currency, "amount": str(amount), "narration": narration[:100]},
		}
		resp = await self._client.post(
			"/v3/transfers/internal",
			json=payload, headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _pesalink_transfer(
		self, from_account: str, to_account: str,
		bank_code: str, amount: float, narration: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"source": {"accountId": from_account},
			"destination": {"bankCode": bank_code, "accountId": to_account},
			"transfer": {"amount": str(amount), "narration": narration[:100]},
		}
		resp = await self._client.post(
			"/v3/transfers/pesalink",
			json=payload, headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _mpesa_to_equity(
		self, phone: str, account_number: str, amount: float
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.post(
			"/v3/transfers/mpesa-to-equity",
			json={"phone": phone, "accountId": account_number, "amount": str(amount)},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _equity_to_mpesa(
		self, account_number: str, phone: str, amount: float
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.post(
			"/v3/transfers/equity-to-mpesa",
			json={"accountId": account_number, "phone": phone, "amount": str(amount)},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _standing_order(
		self, from_account: str, to_account: str, amount: float,
		frequency: str, start_date: str, end_date: str = "",
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"source": {"accountId": from_account},
			"destination": {"accountId": to_account},
			"amount": str(amount),
			"frequency": frequency,
			"startDate": start_date,
			"endDate": end_date,
		}
		resp = await self._client.post(
			"/v3/standing-orders",
			json=payload, headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _transaction_history(
		self, account_number: str, from_date: str, to_date: str, limit: int
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			f"/v3/accounts/{account_number}/transactions",
			params={"from": from_date, "to": to_date, "limit": limit},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()


def equity_connector_from_env(tenant_id: str, user_id: str = "system") -> EquityBankConnector:
	"""Construct EquityBankConnector from environment variables."""
	config = EquityBankConfiguration(
		name="Equity Bank",
		tenant_id=tenant_id,
		user_id=user_id,
		client_id=os.environ["EQUITY_CLIENT_ID"],
		client_secret=os.environ["EQUITY_CLIENT_SECRET"],
		environment=os.environ.get("EQUITY_ENV", "sandbox"),
		merchant_code=os.environ.get("EQUITY_MERCHANT_CODE", ""),
	)
	return EquityBankConnector(config)
