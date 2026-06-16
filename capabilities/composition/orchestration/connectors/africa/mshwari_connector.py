"""M-Shwari (CBA / NCBA + Safaricom) Connector via Safaricom Daraja API.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025

Implements APG's BaseConnector ABC for M-Shwari — the mobile savings-lock and
micro-loan product jointly operated by NCBA (formerly CBA) and Safaricom in
Kenya. All operations are transacted through the Safaricom Daraja 2.0 API using
STK Push and AccountBalance endpoints, differentiated by M-Shwari CommandIDs
and BusinessShortCodes.

OAuth2 tokens are obtained from the Daraja OAuth endpoint and cached with a
60-second pre-expiry buffer, identical to the MPESA connector pattern.

Reference:
	https://developer.safaricom.co.ke/APIs
	M-Shwari API documentation (available to registered Daraja partners)

IMPORTANT:
	M-Shwari operations require Daraja credentials explicitly onboarded for
	M-Shwari products. Standard MPESA developer keys do NOT grant M-Shwari
	access. Contact Safaricom Enterprise Business for onboarding.

Supported markets: KE

Environment variables (configure before instantiating via mshwari_connector_from_env):

	Required:
		MSHWARI_CONSUMER_KEY        Daraja API consumer key (M-Shwari enabled)
		MSHWARI_CONSUMER_SECRET     Daraja API consumer secret
		MSHWARI_SHORTCODE           M-Shwari business shortcode

	Optional:
		MSHWARI_PASSKEY             Lipa na MPESA passkey for STK password generation
		MSHWARI_INITIATOR_NAME      API initiator username (for balance queries)
		MSHWARI_INITIATOR_PASSWORD  Encrypted initiator credential
		MSHWARI_ENV                 "sandbox" | "production"  (default: sandbox)
		MSHWARI_CALLBACK_URL_BASE   Base URL that Daraja posts callbacks to
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

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MSHWARI_CONSUMER_KEY: str = "MSHWARI_CONSUMER_KEY"
MSHWARI_CONSUMER_SECRET: str = "MSHWARI_CONSUMER_SECRET"
MSHWARI_SHORTCODE: str = "MSHWARI_SHORTCODE"
MSHWARI_PASSKEY: str = "MSHWARI_PASSKEY"
MSHWARI_INITIATOR_NAME: str = "MSHWARI_INITIATOR_NAME"
MSHWARI_INITIATOR_PASSWORD: str = "MSHWARI_INITIATOR_PASSWORD"
MSHWARI_ENV: str = "MSHWARI_ENV"
MSHWARI_CALLBACK_URL_BASE: str = "MSHWARI_CALLBACK_URL_BASE"

_SANDBOX_BASE = "https://sandbox.safaricom.co.ke"
_PRODUCTION_BASE = "https://api.safaricom.co.ke"
_TIMEOUT = 30.0

# Daraja CommandIDs that identify M-Shwari product transactions
_CMD_LOCK_SAVINGS = "CustomerLockSavings"
_CMD_UNLOCK_SAVINGS = "CustomerUnlockSavings"
_CMD_LOAN_APPLICATION = "CustomerLoanApplication"
_CMD_LOAN_REPAYMENT = "CustomerLoanRepayment"


# ── Error type ────────────────────────────────────────────────────────────────

class ConnectorError(Exception):
	"""Raised for any M-Shwari / Daraja API failure.

	Attributes:
		message:    Human-readable error description.
		status_code: HTTP status returned by Daraja, if applicable.
		response:   Raw response body, if available.
	"""

	def __init__(
		self,
		message: str,
		status_code: int | None = None,
		response: Any = None,
	) -> None:
		super().__init__(message)
		self.message = message
		self.status_code = status_code
		self.response = response

	def __repr__(self) -> str:
		return (
			f"ConnectorError(message={self.message!r}, "
			f"status_code={self.status_code!r})"
		)


# ── Configuration ─────────────────────────────────────────────────────────────

class MShwariConfiguration(ConnectorConfiguration):
	"""Pydantic v2 configuration for MShwariConnector.

	Kenya-only. Requires Daraja credentials onboarded for M-Shwari products.
	"""

	consumer_key: str = Field(..., description="Daraja API consumer key (M-Shwari enabled)")
	consumer_secret: str = Field(..., description="Daraja API consumer secret")
	shortcode: str = Field(..., description="M-Shwari business shortcode")
	passkey: str = Field(default="", description="Lipa na MPESA passkey for STK password generation")
	initiator_name: str = Field(default="", description="API initiator username (for balance queries)")
	initiator_password: str = Field(default="", description="Encrypted initiator credential")
	# Override base-class 'environment' with sandbox/production vocabulary
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	callback_url_base: str = Field(default="", description="Base URL for Daraja callbacks")


# ── Connector ─────────────────────────────────────────────────────────────────

class MShwariConnector(BaseConnector):
	"""M-Shwari connector (Safaricom Daraja 2.0).

	Supported markets: KE

	Public operations
	-----------------
	lock_savings(amount, phone, duration_days)
		Lock savings into M-Shwari for the given duration via STK Push.

	loan_apply(amount, phone, purpose)
		Apply for an M-Shwari micro-loan via STK Push.

	loan_repay(loan_id, amount, phone)
		Repay an outstanding M-Shwari loan via STK Push.

	check_balance(phone)
		Query the subscriber's M-Shwari account balance (savings + loan).

	All public methods wrap network calls in try/except and raise ConnectorError
	on any failure, so callers get a uniform error type regardless of whether the
	failure originated in OAuth refresh, HTTP transport, or a Daraja API error.
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

	# ── BaseConnector abstract methods ─────────────────────────────────────────

	async def _connect(self) -> None:
		"""Open the httpx session and obtain an initial OAuth token."""
		try:
			self._client = httpx.AsyncClient(
				base_url=self._base_url,
				timeout=_TIMEOUT,
				headers={"Content-Type": "application/json"},
			)
			await self._refresh_token()
			_log.info(
				"M-Shwari connector connected (%s shortcode=%s)",
				self._config.environment,
				self._config.shortcode,
			)
		except Exception as exc:
			raise ConnectorError(f"M-Shwari _connect failed: {exc}") from exc

	async def _disconnect(self) -> None:
		"""Close the httpx session and clear cached credentials."""
		try:
			if self._client:
				await self._client.aclose()
				self._client = None
			self._access_token = ""
			self._token_expires_at = 0.0
		except Exception as exc:
			_log.warning("M-Shwari _disconnect error (ignored): %s", exc)

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		"""Dispatch to the operation handler by name."""
		handlers: dict[str, Any] = {
			"lock_savings": self._op_lock_savings,
			"loan_apply": self._op_loan_apply,
			"loan_repay": self._op_loan_repay,
			"check_balance": self._op_check_balance,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ConnectorError(
				f"Unknown M-Shwari operation: {operation!r}. Valid: {list(handlers)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		"""Verify connectivity by refreshing the OAuth token."""
		try:
			await self._refresh_token()
			return bool(self._access_token)
		except Exception:
			return False

	# ── Public API (required by task spec) ────────────────────────────────────

	async def lock_savings(
		self, amount: int | float, phone: str, duration_days: int
	) -> dict[str, Any]:
		"""Lock customer savings into M-Shwari.

		Args:
			amount:        Amount in KES.
			phone:         Customer MSISDN in 254XXXXXXXXX format.
			duration_days: Lock duration in days (1–180). Passed as the
			               AccountReference so Daraja/NCBA can route correctly.

		Returns:
			Daraja STK Push initiation response.

		Raises:
			ConnectorError: On any transport or API error.
		"""
		try:
			return await self._execute_operation(
				"lock_savings",
				{"amount": int(amount), "phone": phone, "duration_days": duration_days},
			)
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(f"lock_savings failed: {exc}") from exc

	async def loan_apply(
		self, amount: int | float, phone: str, purpose: str
	) -> dict[str, Any]:
		"""Apply for an M-Shwari micro-loan.

		Args:
			amount:  Loan amount in KES.
			phone:   Customer MSISDN in 254XXXXXXXXX format.
			purpose: Short loan purpose description (max 12 chars after truncation).

		Returns:
			Daraja STK Push initiation response.

		Raises:
			ConnectorError: On any transport or API error.
		"""
		try:
			return await self._execute_operation(
				"loan_apply",
				{"amount": int(amount), "phone": phone, "purpose": purpose},
			)
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(f"loan_apply failed: {exc}") from exc

	async def loan_repay(
		self, loan_id: str, amount: int | float, phone: str
	) -> dict[str, Any]:
		"""Repay an outstanding M-Shwari loan.

		Args:
			loan_id: Daraja / NCBA loan reference identifier (used as
			         AccountReference in the STK Push payload, max 12 chars).
			amount:  Repayment amount in KES.
			phone:   Customer MSISDN in 254XXXXXXXXX format.

		Returns:
			Daraja STK Push initiation response.

		Raises:
			ConnectorError: On any transport or API error.
		"""
		try:
			return await self._execute_operation(
				"loan_repay",
				{"loan_id": loan_id, "amount": int(amount), "phone": phone},
			)
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(f"loan_repay failed: {exc}") from exc

	async def check_balance(self, phone: str) -> dict[str, Any]:
		"""Query the M-Shwari account balance for a subscriber.

		Issues an AccountBalance query against the subscriber's MSISDN.
		The result is delivered asynchronously to the configured callback URL.

		Args:
			phone: Customer MSISDN in 254XXXXXXXXX format.

		Returns:
			Daraja AccountBalance initiation response (contains ConversationID).

		Raises:
			ConnectorError: On any transport or API error.
		"""
		try:
			return await self._execute_operation("check_balance", {"phone": phone})
		except ConnectorError:
			raise
		except Exception as exc:
			raise ConnectorError(f"check_balance failed: {exc}") from exc

	# ── Internal operation implementations ────────────────────────────────────

	async def _op_lock_savings(
		self, amount: int, phone: str, duration_days: int
	) -> dict[str, Any]:
		reference = f"LOCK{duration_days}D"
		return await self._stk_push(
			amount=amount,
			phone=phone,
			reference=reference,
			transaction_type=_CMD_LOCK_SAVINGS,
			description=f"Lock {duration_days}d",
			callback_path="/mshwari/savings/lock/callback",
		)

	async def _op_loan_apply(
		self, amount: int, phone: str, purpose: str
	) -> dict[str, Any]:
		return await self._stk_push(
			amount=amount,
			phone=phone,
			reference=purpose[:12],
			transaction_type=_CMD_LOAN_APPLICATION,
			description="Loan Apply",
			callback_path="/mshwari/loan/apply/callback",
		)

	async def _op_loan_repay(
		self, loan_id: str, amount: int, phone: str
	) -> dict[str, Any]:
		return await self._stk_push(
			amount=amount,
			phone=phone,
			reference=loan_id[:12],
			transaction_type=_CMD_LOAN_REPAYMENT,
			description="Loan Repay",
			callback_path="/mshwari/loan/repay/callback",
		)

	async def _op_check_balance(self, phone: str) -> dict[str, Any]:
		result_url = f"{self._config.callback_url_base}/mshwari/balance/result"
		queue_url = f"{self._config.callback_url_base}/mshwari/balance/timeout"
		# IdentifierType "1" = MSISDN
		return await self._account_balance_query(phone, "1", result_url, queue_url)

	# ── Daraja helpers ─────────────────────────────────────────────────────────

	async def _refresh_token(self) -> None:
		"""Fetch a new OAuth2 access token if the cached one is near expiry."""
		if time.time() < self._token_expires_at - 60:
			return
		creds = base64.b64encode(
			f"{self._config.consumer_key}:{self._config.consumer_secret}".encode()
		).decode()
		client = self._client or httpx.AsyncClient(base_url=self._base_url, timeout=10)
		try:
			resp = await client.get(
				"/oauth/v1/generate?grant_type=client_credentials",
				headers={"Authorization": f"Basic {creds}"},
				timeout=_TIMEOUT,
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"OAuth token refresh failed: HTTP {exc.response.status_code}",
				status_code=exc.response.status_code,
				response=exc.response.text,
			) from exc
		except Exception as exc:
			raise ConnectorError(f"OAuth token refresh failed: {exc}") from exc

		data = resp.json()
		self._access_token = data["access_token"]
		self._token_expires_at = time.time() + int(data.get("expires_in", 3599))
		_log.debug(
			"M-Shwari OAuth token refreshed (expires in %ss)",
			data.get("expires_in"),
		)

	def _auth_header(self) -> dict[str, str]:
		return {"Authorization": f"Bearer {self._access_token}"}

	def _stk_password_and_timestamp(self) -> tuple[str, str]:
		"""Derive STK Push password = base64(shortcode + passkey + timestamp)."""
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
		"""Issue a Daraja STK Push request for any M-Shwari product transaction."""
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
		try:
			resp = await self._client.post(
				"/mpesa/stkpush/v1/processrequest",
				json=payload,
				headers=self._auth_header(),
				timeout=_TIMEOUT,
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"STK Push failed: HTTP {exc.response.status_code} — {exc.response.text}",
				status_code=exc.response.status_code,
				response=exc.response.text,
			) from exc
		except Exception as exc:
			raise ConnectorError(f"STK Push request error: {exc}") from exc

		return resp.json()

	async def _account_balance_query(
		self,
		party_a: str,
		identifier_type: str,
		result_url: str,
		queue_url: str,
	) -> dict[str, Any]:
		"""Issue a Daraja AccountBalance query (async — result delivered to result_url)."""
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
		try:
			resp = await self._client.post(
				"/mpesa/accountbalance/v1/query",
				json=payload,
				headers=self._auth_header(),
				timeout=_TIMEOUT,
			)
			resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise ConnectorError(
				f"AccountBalance query failed: HTTP {exc.response.status_code} — {exc.response.text}",
				status_code=exc.response.status_code,
				response=exc.response.text,
			) from exc
		except Exception as exc:
			raise ConnectorError(f"AccountBalance request error: {exc}") from exc

		return resp.json()


# ── Factory ───────────────────────────────────────────────────────────────────

def mshwari_connector_from_env(
	tenant_id: str, user_id: str = "system"
) -> MShwariConnector:
	"""Construct MShwariConnector from environment variables.

	Required env vars:
		MSHWARI_CONSUMER_KEY, MSHWARI_CONSUMER_SECRET, MSHWARI_SHORTCODE

	Optional env vars:
		MSHWARI_PASSKEY, MSHWARI_INITIATOR_NAME, MSHWARI_INITIATOR_PASSWORD,
		MSHWARI_ENV (sandbox|production), MSHWARI_CALLBACK_URL_BASE

	Raises:
		KeyError: if a required env var is not set.
	"""
	config = MShwariConfiguration(
		name="M-Shwari",
		tenant_id=tenant_id,
		user_id=user_id,
		consumer_key=os.environ[MSHWARI_CONSUMER_KEY],
		consumer_secret=os.environ[MSHWARI_CONSUMER_SECRET],
		shortcode=os.environ[MSHWARI_SHORTCODE],
		passkey=os.environ.get(MSHWARI_PASSKEY, ""),
		initiator_name=os.environ.get(MSHWARI_INITIATOR_NAME, ""),
		initiator_password=os.environ.get(MSHWARI_INITIATOR_PASSWORD, ""),
		environment=os.environ.get(MSHWARI_ENV, "sandbox"),
		callback_url_base=os.environ.get(MSHWARI_CALLBACK_URL_BASE, ""),
	)
	return MShwariConnector(config)


__all__ = [
	"ConnectorError",
	"MShwariConfiguration",
	"MShwariConnector",
	"mshwari_connector_from_env",
]
