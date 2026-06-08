"""KCB Bank API Connector.

Kenya Commercial Bank — Kenya's largest bank by assets. The KCB Connect API
enables corporate banking operations: account inquiry, bulk payroll, MPESA
integration, and corporate transfers.

Reference: https://developer.kcbgroup.com/
Markets: Kenya, Uganda, Tanzania, Rwanda, Burundi, Ethiopia, South Sudan
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

_SANDBOX_BASE = "https://uat.connect.kcbgroup.com"
_PRODUCTION_BASE = "https://connect.kcbgroup.com"


class KCBConfiguration(ConnectorConfiguration):
	consumer_key: str = Field(..., description="KCB Connect consumer key")
	consumer_secret: str = Field(..., description="KCB Connect consumer secret")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	shortcode: str = Field(default="", description="KCB business shortcode (for MPESA→KCB)")


class KCBConnector(BaseConnector):
	"""KCB Bank API connector.

	Supports:
	  - Account inquiry (balance, details)
	  - KCB→KCB internal transfer
	  - MPESA→KCB transfer (Lipa na KCB)
	  - KCB→MPESA transfer (KCB to MPESA)
	  - Bulk payroll (upload + process salary file)
	  - Transaction statement
	"""

	def __init__(self, config: KCBConfiguration) -> None:
		super().__init__(config)
		self._config: KCBConfiguration = config
		self._base_url = _SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		self._token: str = ""
		self._token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url, timeout=self._config.timeout_seconds,
			headers={"Content-Type": "application/json"},
		)
		await self._refresh_token()

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None
		self._token = ""

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers = {
			"account_inquiry": self._account_inquiry,
			"internal_transfer": self._internal_transfer,
			"mpesa_to_kcb": self._mpesa_to_kcb,
			"kcb_to_mpesa": self._kcb_to_mpesa,
			"bulk_payroll": self._bulk_payroll,
			"statement": self._statement,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(f"Unknown KCB operation: {operation!r}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._refresh_token()
			return bool(self._token)
		except Exception:
			return False

	async def account_inquiry(self, account_number: str) -> dict[str, Any]:
		return await self._execute_operation("account_inquiry", {"account_number": account_number})

	async def internal_transfer(
		self, from_account: str, to_account: str, amount: float, narration: str = ""
	) -> dict[str, Any]:
		return await self._execute_operation("internal_transfer", {
			"from_account": from_account, "to_account": to_account,
			"amount": amount, "narration": narration,
		})

	async def kcb_to_mpesa(self, account: str, phone: str, amount: float) -> dict[str, Any]:
		return await self._execute_operation("kcb_to_mpesa", {
			"account": account, "phone": phone, "amount": amount,
		})

	async def bulk_payroll(
		self, from_account: str, payroll_records: list[dict[str, Any]]
	) -> dict[str, Any]:
		"""Upload and process a bulk payroll file.

		Args:
			from_account: Source account for disbursements
			payroll_records: List of {employee_id, account_number, amount, narration}
		"""
		return await self._execute_operation("bulk_payroll", {
			"from_account": from_account, "payroll_records": payroll_records,
		})

	async def _refresh_token(self) -> None:
		if time.time() < self._token_expires_at - 60:
			return
		import base64
		creds = base64.b64encode(
			f"{self._config.consumer_key}:{self._config.consumer_secret}".encode()
		).decode()
		client = self._client or httpx.AsyncClient(base_url=self._base_url, timeout=10)
		resp = await client.post(
			"/oauth/token",
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
			f"/v1/accounts/{account_number}/balance", headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _internal_transfer(
		self, from_account: str, to_account: str, amount: float, narration: str
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.post(
			"/v1/transfers",
			json={
				"sourceAccount": from_account, "destinationAccount": to_account,
				"amount": str(amount), "narration": narration[:100],
			},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _mpesa_to_kcb(
		self, phone: str, account_number: str, amount: float
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.post(
			"/v1/mpesa/to-kcb",
			json={"phone": phone, "accountId": account_number, "amount": str(amount)},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _kcb_to_mpesa(
		self, account: str, phone: str, amount: float
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.post(
			"/v1/mpesa/from-kcb",
			json={"accountId": account, "phone": phone, "amount": str(amount)},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _bulk_payroll(
		self, from_account: str, payroll_records: list[dict[str, Any]]
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.post(
			"/v1/payroll/bulk",
			json={"sourceAccount": from_account, "transactions": payroll_records},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _statement(
		self, account_number: str, from_date: str, to_date: str, limit: int = 50
	) -> dict[str, Any]:
		await self._refresh_token()
		resp = await self._client.get(
			f"/v1/accounts/{account_number}/statement",
			params={"fromDate": from_date, "toDate": to_date, "count": limit},
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()


def kcb_connector_from_env(tenant_id: str, user_id: str = "system") -> KCBConnector:
	config = KCBConfiguration(
		name="KCB Bank",
		tenant_id=tenant_id,
		user_id=user_id,
		consumer_key=os.environ["KCB_CONSUMER_KEY"],
		consumer_secret=os.environ["KCB_CONSUMER_SECRET"],
		environment=os.environ.get("KCB_ENV", "sandbox"),
		shortcode=os.environ.get("KCB_SHORTCODE", ""),
	)
	return KCBConnector(config)
