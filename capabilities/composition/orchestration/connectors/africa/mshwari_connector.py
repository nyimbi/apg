"""M-Shwari (CBA + Safaricom) Connector via Safaricom Daraja API.

Implements APG's BaseConnector ABC for M-Shwari — a mobile banking product
jointly operated by CBA (now NCBA) and Safaricom in Kenya. M-Shwari products
(savings lock/unlock and micro-loans) are accessed through the Safaricom Daraja
API using the same STK Push and B2C mechanisms, distinguished by specific
CommandIDs and BusinessShortCodes.

OAuth2 tokens are obtained from the Safaricom Daraja OAuth endpoint (identical
to the MPESA connector). Tokens are cached with a 60-second pre-expiry buffer.

Reference:
    https://developer.safaricom.co.ke/APIs
    M-Shwari API documentation (available to registered Daraja partners)

IMPORTANT:
    M-Shwari operations require a Daraja API key registered for M-Shwari
    products. Standard MPESA developer credentials do NOT grant M-Shwari
    access. Contact Safaricom's Enterprise Business team for onboarding.

Configuration via environment variables or MShwariConfiguration:
    MSHWARI_CONSUMER_KEY        Daraja consumer key (M-Shwari enabled)
    MSHWARI_CONSUMER_SECRET     Daraja consumer secret
    MSHWARI_SHORTCODE           M-Shwari business shortcode
    MSHWARI_PASSKEY             Lipa na MPESA passkey
    MSHWARI_INITIATOR_NAME      API initiator username
    MSHWARI_INITIATOR_PASSWORD  Encrypted initiator credential
    MSHWARI_ENV                 "sandbox" | "production" (default: sandbox)
    MSHWARI_CALLBACK_URL_BASE   Base URL for Daraja callbacks
"""
from __future__ import annotations

import base64
import logging
import os
import time
from datetime import datetime, timezone
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

_SANDBOX_BASE = "https://sandbox.safaricom.co.ke"
_PRODUCTION_BASE = "https://api.safaricom.co.ke"

# M-Shwari CommandIDs used in STK Push and B2C calls
_CMD_LOCK_SAVINGS = "CustomerLockSavings"
_CMD_UNLOCK_SAVINGS = "CustomerUnlockSavings"
_CMD_LOAN_APPLICATION = "CustomerLoanApplication"
_CMD_LOAN_REPAYMENT = "CustomerLoanRepayment"
_CMD_LOAN_BALANCE = "AccountBalance"
_CMD_SAVINGS_BALANCE = "AccountBalance"


class MShwariConfiguration(ConnectorConfiguration):
	"""Configuration for the M-Shwari connector.

	Kenya only. Requires Daraja API credentials explicitly onboarded for
	M-Shwari products by Safaricom.
	"""

	consumer_key: str = Field(..., description="Daraja API consumer key (M-Shwari enabled)")
	consumer_secret: str = Field(..., description="Daraja API consumer secret")
	shortcode: str = Field(..., description="M-Shwari business shortcode")
	passkey: str = Field(default="", description="Lipa na MPESA passkey for STK operations")
	initiator_name: str = Field(default="", description="API initiator username")
	initiator_password: str = Field(default="", description="Encrypted initiator credential")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str = Field(default="", description="Base URL for Daraja callbacks")


class MShwariConnector(BaseConnector):
	"""M-Shwari connector (Safaricom Daraja API).

	Supports:
	  - lock_savings         — instruct a customer to lock savings (STK Push)
	  - unlock_savings       — instruct a customer to unlock savings
	  - apply_for_loan       — customer loan application via STK Push
	  - repay_loan           — loan repayment via STK Push
	  - check_loan_balance   — query outstanding loan balance (AccountBalance)
	  - check_savings_balance — query locked savings balance (AccountBalance)

	All operations use Daraja STK Push or B2C; the specific product (savings /
	loan) is determined by the CommandID in the request payload.
	"""

	def __init__(self, config: MShwariConfiguration) -> None:
		super().__init__(config)
		self._config: MShwariConfiguration = config
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
		_log.info("M-Shwari connector connected (%s)", self._config.environment)

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
			"lock_savings": self._lock_savings,
			"unlock_savings": self._unlock_savings,
			"apply_for_loan": self._apply_for_loan,
			"repay_loan": self._repay_loan,
			"check_loan_balance": self._check_loan_balance,
			"check_savings_balance": self._check_savings_balance,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(
				f"Unknown M-Shwari operation: {operation!r}. Valid: {list(handlers)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._refresh_token()
			return bool(self._access_token)
		except Exception:
			return False

	# ── Public operation methods ──────────────────────────────────────────────

	async def lock_savings(
		self, amount: int, phone: str, reference: str
	) -> dict[str, Any]:
		"""Instruct a customer to lock savings into M-Shwari via STK Push.

		Args:
			amount:    Amount in KES (integer)
			phone:     Customer MSISDN in 254XXXXXXXXX format
			reference: Merchant transaction reference (max 12 chars)
		"""
		return await self._execute_operation("lock_savings", {
			"amount": amount, "phone": phone, "reference": reference,
		})

	async def unlock_savings(
		self, amount: int, phone: str, reference: str
	) -> dict[str, Any]:
		"""Instruct a customer to unlock (withdraw) M-Shwari savings via STK Push."""
		return await self._execute_operation("unlock_savings", {
			"amount": amount, "phone": phone, "reference": reference,
		})

	async def apply_for_loan(
		self, amount: int, phone: str, reference: str
	) -> dict[str, Any]:
		"""Initiate an M-Shwari loan application via STK Push.

		The customer confirms the loan on their phone. Eligibility is determined
		by Safaricom/NCBA's credit scoring model.
		"""
		return await self._execute_operation("apply_for_loan", {
			"amount": amount, "phone": phone, "reference": reference,
		})

	async def repay_loan(
		self, amount: int, phone: str, reference: str
	) -> dict[str, Any]:
		"""Initiate an M-Shwari loan repayment via STK Push."""
		return await self._execute_operation("repay_loan", {
			"amount": amount, "phone": phone, "reference": reference,
		})

	async def check_loan_balance(self, phone: str) -> dict[str, Any]:
		"""Query the outstanding M-Shwari loan balance for a subscriber.

		Uses Daraja AccountBalance with IdentifierType=1 (MSISDN).
		"""
		return await self._execute_operation("check_loan_balance", {"phone": phone})

	async def check_savings_balance(self, phone: str) -> dict[str, Any]:
		"""Query the M-Shwari savings account balance for a subscriber."""
		return await self._execute_operation("check_savings_balance", {"phone": phone})

	# ── Private implementation ────────────────────────────────────────────────

	async def _refresh_token(self) -> None:
		if time.time() < self._token_expires_at - 60:
			return
		creds = base64.b64encode(
			f"{self._config.consumer_key}:{self._config.consumer_secret}".encode()
		).decode()
		client = self._client or httpx.AsyncClient(base_url=self._base_url, timeout=10)
		resp = await client.get(
			"/oauth/v1/generate?grant_type=client_credentials",
			headers={"Authorization": f"Basic {creds}"},
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		data = resp.json()
		self._access_token = data["access_token"]
		self._token_expires_at = time.time() + int(data.get("expires_in", 3599))
		_log.debug("M-Shwari OAuth token refreshed (expires in %ss)", data.get("expires_in"))

	def _auth_header(self) -> dict[str, str]:
		return {"Authorization": f"Bearer {self._access_token}"}

	def _stk_password_and_timestamp(self) -> tuple[str, str]:
		"""Generate STK Push password (base64 shortcode+passkey+timestamp)."""
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
		raw = f"{self._config.shortcode}{self._config.passkey}{timestamp}"
		password = base64.b64encode(raw.encode()).decode()
		return password, timestamp

	async def _stk_push(
		self,
		amount: int,
		phone: str,
		reference: str,
		transaction_type: str,
		description: str,
		callback_path: str,
	) -> dict[str, Any]:
		"""Common STK Push invocation for all M-Shwari product operations."""
		await self._refresh_token()
		password, timestamp = self._stk_password_and_timestamp()
		cb_url = f"{self._config.callback_url_base}{callback_path}"
		payload = {
			"BusinessShortCode": self._config.shortcode,
			"Password": password,
			"Timestamp": timestamp,
			"TransactionType": transaction_type,
			"Amount": str(amount),
			"PartyA": phone,
			"PartyB": self._config.shortcode,
			"PhoneNumber": phone,
			"CallBackURL": cb_url,
			"AccountReference": reference[:12],
			"TransactionDesc": description[:13],
		}
		resp = await self._client.post(
			"/mpesa/stkpush/v1/processrequest",
			json=payload,
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _account_balance_query(
		self, party_a: str, identifier_type: str, result_url: str, queue_url: str
	) -> dict[str, Any]:
		"""Internal AccountBalance query used for loan and savings balance checks."""
		await self._refresh_token()
		payload = {
			"Initiator": self._config.initiator_name,
			"SecurityCredential": self._config.initiator_password,
			"CommandID": "AccountBalance",
			"PartyA": party_a,
			"IdentifierType": identifier_type,
			"Remarks": "M-Shwari Balance Query",
			"QueueTimeOutURL": queue_url,
			"ResultURL": result_url,
		}
		resp = await self._client.post(
			"/mpesa/accountbalance/v1/query",
			json=payload,
			headers=self._auth_header(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()

	async def _lock_savings(self, amount: int, phone: str, reference: str) -> dict[str, Any]:
		return await self._stk_push(
			amount, phone, reference,
			transaction_type=_CMD_LOCK_SAVINGS,
			description="Lock Savings",
			callback_path="/mshwari/savings/lock/callback",
		)

	async def _unlock_savings(self, amount: int, phone: str, reference: str) -> dict[str, Any]:
		return await self._stk_push(
			amount, phone, reference,
			transaction_type=_CMD_UNLOCK_SAVINGS,
			description="Unlock Savings",
			callback_path="/mshwari/savings/unlock/callback",
		)

	async def _apply_for_loan(self, amount: int, phone: str, reference: str) -> dict[str, Any]:
		return await self._stk_push(
			amount, phone, reference,
			transaction_type=_CMD_LOAN_APPLICATION,
			description="Loan Application",
			callback_path="/mshwari/loan/apply/callback",
		)

	async def _repay_loan(self, amount: int, phone: str, reference: str) -> dict[str, Any]:
		return await self._stk_push(
			amount, phone, reference,
			transaction_type=_CMD_LOAN_REPAYMENT,
			description="Loan Repayment",
			callback_path="/mshwari/loan/repay/callback",
		)

	async def _check_loan_balance(self, phone: str) -> dict[str, Any]:
		result_url = f"{self._config.callback_url_base}/mshwari/loan/balance/result"
		queue_url = f"{self._config.callback_url_base}/mshwari/loan/balance/timeout"
		# IdentifierType "1" = MSISDN — query the subscriber's M-Shwari loan account
		return await self._account_balance_query(phone, "1", result_url, queue_url)

	async def _check_savings_balance(self, phone: str) -> dict[str, Any]:
		result_url = f"{self._config.callback_url_base}/mshwari/savings/balance/result"
		queue_url = f"{self._config.callback_url_base}/mshwari/savings/balance/timeout"
		return await self._account_balance_query(phone, "1", result_url, queue_url)


def mshwari_connector_from_env(tenant_id: str, user_id: str = "system") -> MShwariConnector:
	"""Construct MShwariConnector from environment variables.

	Required env vars:
	    MSHWARI_CONSUMER_KEY, MSHWARI_CONSUMER_SECRET, MSHWARI_SHORTCODE

	Optional:
	    MSHWARI_PASSKEY, MSHWARI_INITIATOR_NAME, MSHWARI_INITIATOR_PASSWORD,
	    MSHWARI_ENV, MSHWARI_CALLBACK_URL_BASE
	"""
	config = MShwariConfiguration(
		name="M-Shwari",
		tenant_id=tenant_id,
		user_id=user_id,
		consumer_key=os.environ["MSHWARI_CONSUMER_KEY"],
		consumer_secret=os.environ["MSHWARI_CONSUMER_SECRET"],
		shortcode=os.environ["MSHWARI_SHORTCODE"],
		passkey=os.environ.get("MSHWARI_PASSKEY", ""),
		initiator_name=os.environ.get("MSHWARI_INITIATOR_NAME", ""),
		initiator_password=os.environ.get("MSHWARI_INITIATOR_PASSWORD", ""),
		environment=os.environ.get("MSHWARI_ENV", "sandbox"),
		callback_url_base=os.environ.get("MSHWARI_CALLBACK_URL_BASE", ""),
	)
	return MShwariConnector(config)
