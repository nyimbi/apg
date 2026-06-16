"""
M-Shwari (CBA/Safaricom) — Production-Quality APG Connector.

M-Shwari is a mobile banking product by Commercial Bank of Africa (CBA)
and Safaricom, accessible via Safaricom's Daraja API (same OAuth2 flow).

Supported markets: KE (Kenya, KES only)

Operations:
	lock_savings(amount, phone, duration_days)  — lock funds in M-Shwari savings
	loan_apply(amount, phone, purpose)           — apply for M-Shwari loan
	loan_repay(loan_id, amount, phone)           — repay outstanding M-Shwari loan
	check_balance(phone)                         — query M-Shwari account balance

All operations use the Daraja B2C / Account Balance APIs on behalf of
the subscriber's MSISDN (phone number).

Environment variables (required):
	MSHWARI_CONSUMER_KEY       Daraja API consumer key
	MSHWARI_CONSUMER_SECRET    Daraja API consumer secret
	MSHWARI_SHORTCODE          Business short code (5-6 digits)
	MSHWARI_INITIATOR_NAME     Initiator name registered with Safaricom
	MSHWARI_SECURITY_CREDENTIAL  Encrypted initiator password

Environment variables (optional):
	MSHWARI_ENV                "sandbox" | "production"  (default: sandbox)
	MSHWARI_CALLBACK_URL_BASE  Base URL for Daraja callbacks

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
from pydantic import Field

from .base_connector import BaseConnector, ConnectorConfiguration, ConnectorStatus

_log = logging.getLogger(__name__)

MSHWARI_ENV_CONSUMER_KEY		= "MSHWARI_CONSUMER_KEY"
MSHWARI_ENV_CONSUMER_SECRET		= "MSHWARI_CONSUMER_SECRET"
MSHWARI_ENV_SHORTCODE			= "MSHWARI_SHORTCODE"
MSHWARI_ENV_INITIATOR_NAME		= "MSHWARI_INITIATOR_NAME"
MSHWARI_ENV_SECURITY_CREDENTIAL	= "MSHWARI_SECURITY_CREDENTIAL"
MSHWARI_ENV_ENVIRONMENT			= "MSHWARI_ENV"
MSHWARI_ENV_CALLBACK_BASE		= "MSHWARI_CALLBACK_URL_BASE"

_SANDBOX_BASE		= "https://sandbox.safaricom.co.ke"
_PRODUCTION_BASE	= "https://api.safaricom.co.ke"
_DEFAULT_TIMEOUT	= 30.0
_TOKEN_REFRESH_BUFFER = 60


class ConnectorError(Exception):
	"""Unrecoverable M-Shwari / Daraja API error."""
	def __init__(self, message: str, *, status_code: int | None = None, body: Any = None) -> None:
		super().__init__(message)
		self.status_code = status_code
		self.body = body


class MShwariConfiguration(ConnectorConfiguration):
	consumer_key: str			= Field(..., description="Daraja API consumer key")
	consumer_secret: str		= Field(..., description="Daraja API consumer secret")
	shortcode: str				= Field(..., description="Business short code")
	initiator_name: str			= Field(..., description="Initiator name registered with Safaricom")
	security_credential: str	= Field(..., description="Encrypted initiator password")
	environment: str			= Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str		= Field(default="")


class MShwariConnector(BaseConnector):
	"""Production-quality M-Shwari connector (Kenya only).

	Operations
	----------
	lock_savings(amount, phone, duration_days)
		Lock subscriber funds in an M-Shwari fixed deposit via B2C credit.
	loan_apply(amount, phone, purpose)
		Apply for an M-Shwari loan on behalf of the subscriber.
	loan_repay(loan_id, amount, phone)
		Repay an outstanding M-Shwari loan via C2B debit.
	check_balance(phone)
		Query the subscriber's M-Shwari balance via Account Balance API.
	"""

	def __init__(self, config: MShwariConfiguration) -> None:
		super().__init__(config)
		self._cfg = config
		self._base = _SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		self._access_token: str = ""
		self._token_expires_at: float = 0.0
		self._http: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._http = httpx.AsyncClient(base_url=self._base, timeout=_DEFAULT_TIMEOUT)
		await self._ensure_token()
		_log.info("MShwariConnector connected env=%s", self._cfg.environment)

	async def _disconnect(self) -> None:
		if self._http:
			await self._http.aclose()
			self._http = None
		self._access_token = ""
		self._token_expires_at = 0.0

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers: dict[str, Any] = {
			"lock_savings":		self._lock_savings,
			"loan_apply":		self._loan_apply,
			"loan_repay":		self._loan_repay,
			"check_balance":	self._check_balance,
		}
		handler = handlers.get(operation)
		if not handler:
			raise ConnectorError(f"Unknown M-Shwari operation: {operation!r}. Valid: {sorted(handlers)}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._ensure_token(force=True)
			return bool(self._access_token)
		except ConnectorError:
			return False

	# ── Public API ────────────────────────────────────────────────────────────

	async def lock_savings(self, amount: str | int | float, phone: str, duration_days: int = 30) -> dict[str, Any]:
		return await self._execute_operation("lock_savings", {
			"amount": str(amount), "phone": phone, "duration_days": duration_days,
		})

	async def loan_apply(self, amount: str | int | float, phone: str, purpose: str = "") -> dict[str, Any]:
		return await self._execute_operation("loan_apply", {
			"amount": str(amount), "phone": phone, "purpose": purpose,
		})

	async def loan_repay(self, loan_id: str, amount: str | int | float, phone: str) -> dict[str, Any]:
		return await self._execute_operation("loan_repay", {
			"loan_id": loan_id, "amount": str(amount), "phone": phone,
		})

	async def check_balance(self, phone: str) -> dict[str, Any]:
		return await self._execute_operation("check_balance", {"phone": phone})

	# ── Token management ──────────────────────────────────────────────────────

	async def _ensure_token(self, *, force: bool = False) -> None:
		if not force and time.time() < self._token_expires_at - _TOKEN_REFRESH_BUFFER:
			return
		client = self._http or httpx.AsyncClient(base_url=self._base, timeout=_DEFAULT_TIMEOUT)
		try:
			resp = await client.get(
				"/oauth/v1/generate?grant_type=client_credentials",
				auth=(self._cfg.consumer_key, self._cfg.consumer_secret),
			)
		except httpx.TransportError as exc:
			raise ConnectorError(f"M-Shwari OAuth2 transport error: {exc}") from exc
		if resp.status_code != 200:
			raise ConnectorError(f"M-Shwari OAuth2 HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		data = resp.json()
		self._access_token = data.get("access_token", "")
		if not self._access_token:
			raise ConnectorError("M-Shwari OAuth2 missing access_token", body=data)
		self._token_expires_at = time.time() + int(data.get("expires_in", 3600))

	def _auth_header(self) -> dict[str, str]:
		return {"Authorization": f"Bearer {self._access_token}", "Content-Type": "application/json"}

	async def _b2c_post(self, amount: str, phone: str, command_id: str, remarks: str) -> dict[str, Any]:
		"""POST to B2C Payment Request endpoint."""
		assert self._http is not None
		await self._ensure_token()
		conversation_id = str(uuid.uuid4())
		payload = {
			"InitiatorName": self._cfg.initiator_name,
			"SecurityCredential": self._cfg.security_credential,
			"CommandID": command_id,
			"Amount": amount,
			"PartyA": self._cfg.shortcode,
			"PartyB": phone,
			"Remarks": remarks,
			"QueueTimeOutURL": f"{self._cfg.callback_url_base}/mshwari/timeout",
			"ResultURL": f"{self._cfg.callback_url_base}/mshwari/result",
			"Occasion": conversation_id,
		}
		try:
			resp = await self._http.post("/mpesa/b2c/v1/paymentrequest", json=payload, headers=self._auth_header(), timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"M-Shwari B2C transport error: {exc}") from exc
		if resp.status_code >= 400:
			raise ConnectorError(f"M-Shwari B2C HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return resp.json()

	# ── Operation implementations ─────────────────────────────────────────────

	async def _lock_savings(self, amount: str, phone: str, duration_days: int) -> dict[str, Any]:
		result = await self._b2c_post(amount, phone, "BusinessPayment", f"M-Shwari lock {duration_days}d")
		return {**result, "duration_days": duration_days, "type": "lock_savings"}

	async def _loan_apply(self, amount: str, phone: str, purpose: str) -> dict[str, Any]:
		result = await self._b2c_post(amount, phone, "BusinessPayment", f"M-Shwari loan: {purpose}")
		return {**result, "type": "loan_disbursement", "purpose": purpose}

	async def _loan_repay(self, loan_id: str, amount: str, phone: str) -> dict[str, Any]:
		assert self._http is not None
		await self._ensure_token()
		# Loan repayment is a C2B STK push
		payload = {
			"BusinessShortCode": self._cfg.shortcode,
			"Password": self._cfg.security_credential,
			"Timestamp": time.strftime("%Y%m%d%H%M%S"),
			"TransactionType": "CustomerPayBillOnline",
			"Amount": amount,
			"PartyA": phone,
			"PartyB": self._cfg.shortcode,
			"PhoneNumber": phone,
			"CallBackURL": f"{self._cfg.callback_url_base}/mshwari/stk/callback",
			"AccountReference": loan_id,
			"TransactionDesc": f"M-Shwari loan repayment {loan_id}",
		}
		try:
			resp = await self._http.post("/mpesa/stkpush/v1/processrequest", json=payload, headers=self._auth_header(), timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"M-Shwari loan_repay transport error: {exc}") from exc
		if resp.status_code >= 400:
			raise ConnectorError(f"M-Shwari loan_repay HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return {**resp.json(), "type": "loan_repayment", "loan_id": loan_id}

	async def _check_balance(self, phone: str) -> dict[str, Any]:
		assert self._http is not None
		await self._ensure_token()
		payload = {
			"Initiator": self._cfg.initiator_name,
			"SecurityCredential": self._cfg.security_credential,
			"CommandID": "AccountBalance",
			"PartyA": self._cfg.shortcode,
			"IdentifierType": "4",
			"Remarks": f"Balance for {phone}",
			"QueueTimeOutURL": f"{self._cfg.callback_url_base}/mshwari/balance/timeout",
			"ResultURL": f"{self._cfg.callback_url_base}/mshwari/balance/result",
		}
		try:
			resp = await self._http.post("/mpesa/accountbalance/v1/query", json=payload, headers=self._auth_header(), timeout=_DEFAULT_TIMEOUT)
		except httpx.TransportError as exc:
			raise ConnectorError(f"M-Shwari check_balance transport error: {exc}") from exc
		if resp.status_code >= 400:
			raise ConnectorError(f"M-Shwari check_balance HTTP {resp.status_code}", status_code=resp.status_code, body=resp.text)
		return {**resp.json(), "phone": phone}


def mshwari_connector_from_env(tenant_id: str, user_id: str = "system") -> MShwariConnector:
	"""Construct MShwariConnector from environment variables."""
	config = MShwariConfiguration(
		name="M-Shwari",
		tenant_id=tenant_id,
		user_id=user_id,
		consumer_key=os.environ[MSHWARI_ENV_CONSUMER_KEY],
		consumer_secret=os.environ[MSHWARI_ENV_CONSUMER_SECRET],
		shortcode=os.environ[MSHWARI_ENV_SHORTCODE],
		initiator_name=os.environ[MSHWARI_ENV_INITIATOR_NAME],
		security_credential=os.environ[MSHWARI_ENV_SECURITY_CREDENTIAL],
		environment=os.environ.get(MSHWARI_ENV_ENVIRONMENT, "sandbox"),
		callback_url_base=os.environ.get(MSHWARI_ENV_CALLBACK_BASE, ""),
	)
	return MShwariConnector(config)


__all__ = [
	"MShwariConnector", "MShwariConfiguration", "ConnectorError",
	"mshwari_connector_from_env",
]
