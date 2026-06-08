"""Safaricom MPESA Daraja API 2.0 Connector.

Implements APG's BaseConnector ABC for the MPESA payment platform —
East Africa's dominant mobile money network. Supports all Daraja 2.0
APIs including STK Push (Lipa na MPESA), C2B, B2C, B2B, Account Balance,
Transaction Status, and Reversal.

OAuth2 access tokens are cached and refreshed automatically (TTL: 3599s).

Reference: https://developer.safaricom.co.ke/APIs

Configuration via environment variables or ConnectorConfiguration:
    MPESA_CONSUMER_KEY        Daraja app consumer key
    MPESA_CONSUMER_SECRET     Daraja app consumer secret
    MPESA_SHORTCODE           Business shortcode (Paybill or Till)
    MPESA_PASSKEY             Lipa na MPESA passkey (for STK Push)
    MPESA_ENV                 "sandbox" | "production" (default: sandbox)
    MPESA_INITIATOR_NAME      For B2C/B2B/AccountBalance operations
    MPESA_INITIATOR_PASSWORD  Encrypted initiator credential
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import Field

from ..base_connector import BaseConnector, ConnectorConfiguration

_log = logging.getLogger(__name__)

_SANDBOX_BASE = "https://sandbox.safaricom.co.ke"
_PRODUCTION_BASE = "https://api.safaricom.co.ke"


class MPESAConfiguration(ConnectorConfiguration):
	"""Configuration for the MPESA Daraja connector."""

	consumer_key: str = Field(..., description="Daraja API consumer key")
	consumer_secret: str = Field(..., description="Daraja API consumer secret")
	shortcode: str = Field(..., description="Business shortcode (Paybill or Till number)")
	passkey: str = Field(default="", description="Lipa na MPESA online passkey (STK Push)")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	initiator_name: str = Field(default="", description="API initiator username")
	initiator_password: str = Field(default="", description="Encrypted initiator credential")
	callback_url_base: str = Field(default="", description="Base URL for MPESA callbacks")


class MPESAConnector(BaseConnector):
	"""Safaricom MPESA Daraja 2.0 connector.

	Supports:
	  - STK Push (Lipa na MPESA online) — prompt customer to pay on phone
	  - C2B (Customer to Business) — register URLs, simulate/confirm payments
	  - B2C (Business to Customer) — send money to customer phones
	  - Account Balance — query business account balance
	  - Transaction Status — check status of any MPESA transaction
	  - Reversal — reverse a completed transaction
	"""

	def __init__(self, config: MPESAConfiguration) -> None:
		super().__init__(config)
		self._config: MPESAConfiguration = config
		self._base_url = _SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		self._token: str = ""
		self._token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None

	# ── BaseConnector abstract methods ─────────────────────────────────────

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=self._config.timeout_seconds,
			headers={"Content-Type": "application/json"},
		)
		await self._refresh_token()
		_log.info("MPESA connector connected (%s)", self._config.environment)

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None
		self._token = ""
		self._token_expires_at = 0.0

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		"""Route operation name to the appropriate Daraja API call."""
		handlers: dict[str, Any] = {
			"stk_push": self._stk_push,
			"stk_query": self._stk_query,
			"c2b_register_url": self._c2b_register_url,
			"c2b_simulate": self._c2b_simulate,
			"b2c_payment": self._b2c_payment,
			"account_balance": self._account_balance,
			"transaction_status": self._transaction_status,
			"reversal": self._reversal,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(f"Unknown MPESA operation: {operation!r}. Valid: {list(handlers)}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._refresh_token()
			return bool(self._token)
		except Exception:
			return False

	# ── Public operation methods ───────────────────────────────────────────

	async def stk_push(
		self,
		amount: int,
		phone: str,
		account_reference: str,
		transaction_desc: str = "Payment",
		callback_url: str = "",
	) -> dict[str, Any]:
		"""Initiate Lipa na MPESA Online (STK Push) payment.

		Sends an STK push notification to the customer's phone prompting
		them to enter their MPESA PIN to complete the payment.

		Args:
			amount: Amount in KES (integer, no decimals)
			phone: Customer phone in format 254XXXXXXXXX
			account_reference: Paybill account number or order ID (max 12 chars)
			transaction_desc: Description shown on customer's phone (max 13 chars)
			callback_url: Override default callback URL for this transaction

		Returns:
			Daraja STK Push response with MerchantRequestID and CheckoutRequestID
		"""
		return await self._execute_operation("stk_push", {
			"amount": amount,
			"phone": phone,
			"account_reference": account_reference,
			"transaction_desc": transaction_desc,
			"callback_url": callback_url,
		})

	async def check_stk_status(self, checkout_request_id: str) -> dict[str, Any]:
		"""Query STK Push transaction status."""
		return await self._execute_operation("stk_query", {
			"checkout_request_id": checkout_request_id,
		})

	async def b2c_payment(
		self,
		amount: int,
		phone: str,
		command_id: str = "BusinessPayment",
		remarks: str = "Payment",
		occasion: str = "",
		result_url: str = "",
		queue_timeout_url: str = "",
	) -> dict[str, Any]:
		"""Send money from business to customer phone (B2C).

		Args:
			command_id: "SalaryPayment" | "BusinessPayment" | "PromotionPayment"
		"""
		return await self._execute_operation("b2c_payment", {
			"amount": amount,
			"phone": phone,
			"command_id": command_id,
			"remarks": remarks,
			"occasion": occasion,
			"result_url": result_url,
			"queue_timeout_url": queue_timeout_url,
		})

	async def account_balance(
		self,
		party_b: str = "",
		result_url: str = "",
		queue_timeout_url: str = "",
	) -> dict[str, Any]:
		"""Query business account balance."""
		return await self._execute_operation("account_balance", {
			"party_b": party_b or self._config.shortcode,
			"result_url": result_url,
			"queue_timeout_url": queue_timeout_url,
		})

	async def transaction_status(
		self,
		transaction_id: str,
		result_url: str = "",
		queue_timeout_url: str = "",
	) -> dict[str, Any]:
		"""Query the status of any MPESA transaction."""
		return await self._execute_operation("transaction_status", {
			"transaction_id": transaction_id,
			"result_url": result_url,
			"queue_timeout_url": queue_timeout_url,
		})

	async def reversal(
		self,
		transaction_id: str,
		amount: int,
		result_url: str = "",
		queue_timeout_url: str = "",
		remarks: str = "Reversal",
	) -> dict[str, Any]:
		"""Reverse a completed MPESA transaction."""
		return await self._execute_operation("reversal", {
			"transaction_id": transaction_id,
			"amount": amount,
			"result_url": result_url,
			"queue_timeout_url": queue_timeout_url,
			"remarks": remarks,
		})

	def verify_callback_signature(
		self, payload: bytes, signature: str, secret: str
	) -> bool:
		"""Verify MPESA webhook callback signature.

		Used to authenticate incoming C2B confirmation/validation callbacks.
		"""
		expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
		return hmac.compare_digest(expected, signature)

	# ── Private implementation ─────────────────────────────────────────────

	async def _refresh_token(self) -> None:
		if time.time() < self._token_expires_at - 60:
			return  # Token still valid (with 60s buffer)

		creds = base64.b64encode(
			f"{self._config.consumer_key}:{self._config.consumer_secret}".encode()
		).decode()

		client = self._client or httpx.AsyncClient(base_url=self._base_url, timeout=10)
		response = await client.get(
			"/oauth/v1/generate?grant_type=client_credentials",
			headers={"Authorization": f"Basic {creds}"},
		)
		response.raise_for_status()
		data = response.json()
		self._token = data["access_token"]
		self._token_expires_at = time.time() + int(data.get("expires_in", 3599))
		_log.debug("MPESA OAuth token refreshed (expires in %ss)", data.get("expires_in"))

	def _auth_header(self) -> dict[str, str]:
		return {"Authorization": f"Bearer {self._token}"}

	def _stk_password(self) -> str:
		"""Generate STK Push password (base64 of shortcode+passkey+timestamp)."""
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
		raw = f"{self._config.shortcode}{self._config.passkey}{timestamp}"
		return base64.b64encode(raw.encode()).decode(), timestamp

	async def _stk_push(
		self,
		amount: int,
		phone: str,
		account_reference: str,
		transaction_desc: str,
		callback_url: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		password, timestamp = self._stk_password()
		cb = callback_url or f"{self._config.callback_url_base}/mpesa/stk/callback"
		payload = {
			"BusinessShortCode": self._config.shortcode,
			"Password": password,
			"Timestamp": timestamp,
			"TransactionType": "CustomerPayBillOnline",
			"Amount": str(amount),
			"PartyA": phone,
			"PartyB": self._config.shortcode,
			"PhoneNumber": phone,
			"CallBackURL": cb,
			"AccountReference": account_reference[:12],
			"TransactionDesc": transaction_desc[:13],
		}
		resp = await self._client.post(
			"/mpesa/stkpush/v1/processrequest",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _stk_query(self, checkout_request_id: str) -> dict[str, Any]:
		await self._refresh_token()
		password, timestamp = self._stk_password()
		payload = {
			"BusinessShortCode": self._config.shortcode,
			"Password": password,
			"Timestamp": timestamp,
			"CheckoutRequestID": checkout_request_id,
		}
		resp = await self._client.post(
			"/mpesa/stkpushquery/v1/query",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _c2b_register_url(
		self,
		validation_url: str,
		confirmation_url: str,
		response_type: str = "Completed",
	) -> dict[str, Any]:
		await self._refresh_token()
		payload = {
			"ShortCode": self._config.shortcode,
			"ResponseType": response_type,
			"ConfirmationURL": confirmation_url,
			"ValidationURL": validation_url,
		}
		resp = await self._client.post(
			"/mpesa/c2b/v1/registerurl",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _c2b_simulate(
		self,
		amount: int,
		phone: str,
		bill_ref: str = "test",
		command_id: str = "CustomerPayBillOnline",
	) -> dict[str, Any]:
		"""Sandbox-only: simulate a C2B payment."""
		await self._refresh_token()
		payload = {
			"ShortCode": self._config.shortcode,
			"CommandID": command_id,
			"Amount": str(amount),
			"Msisdn": phone,
			"BillRefNumber": bill_ref,
		}
		resp = await self._client.post(
			"/mpesa/c2b/v1/simulate",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _b2c_payment(
		self,
		amount: int,
		phone: str,
		command_id: str,
		remarks: str,
		occasion: str,
		result_url: str,
		queue_timeout_url: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		result_url = result_url or f"{self._config.callback_url_base}/mpesa/b2c/result"
		queue_url = queue_timeout_url or f"{self._config.callback_url_base}/mpesa/b2c/timeout"
		payload = {
			"InitiatorName": self._config.initiator_name,
			"SecurityCredential": self._config.initiator_password,
			"CommandID": command_id,
			"Amount": str(amount),
			"PartyA": self._config.shortcode,
			"PartyB": phone,
			"Remarks": remarks[:100],
			"QueueTimeOutURL": queue_url,
			"ResultURL": result_url,
			"Occasion": occasion[:100],
		}
		resp = await self._client.post(
			"/mpesa/b2c/v1/paymentrequest",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _account_balance(
		self,
		party_b: str,
		result_url: str,
		queue_timeout_url: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		result_url = result_url or f"{self._config.callback_url_base}/mpesa/balance/result"
		queue_url = queue_timeout_url or f"{self._config.callback_url_base}/mpesa/balance/timeout"
		payload = {
			"Initiator": self._config.initiator_name,
			"SecurityCredential": self._config.initiator_password,
			"CommandID": "AccountBalance",
			"PartyA": party_b,
			"IdentifierType": "4",  # Organization shortcode
			"Remarks": "Balance",
			"QueueTimeOutURL": queue_url,
			"ResultURL": result_url,
		}
		resp = await self._client.post(
			"/mpesa/accountbalance/v1/query",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _transaction_status(
		self,
		transaction_id: str,
		result_url: str,
		queue_timeout_url: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		result_url = result_url or f"{self._config.callback_url_base}/mpesa/status/result"
		queue_url = queue_timeout_url or f"{self._config.callback_url_base}/mpesa/status/timeout"
		payload = {
			"Initiator": self._config.initiator_name,
			"SecurityCredential": self._config.initiator_password,
			"CommandID": "TransactionStatusQuery",
			"TransactionID": transaction_id,
			"PartyA": self._config.shortcode,
			"IdentifierType": "4",
			"ResultURL": result_url,
			"QueueTimeOutURL": queue_url,
			"Remarks": "Status",
			"Occasion": "",
		}
		resp = await self._client.post(
			"/mpesa/transactionstatus/v1/query",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()

	async def _reversal(
		self,
		transaction_id: str,
		amount: int,
		result_url: str,
		queue_timeout_url: str,
		remarks: str,
	) -> dict[str, Any]:
		await self._refresh_token()
		result_url = result_url or f"{self._config.callback_url_base}/mpesa/reversal/result"
		queue_url = queue_timeout_url or f"{self._config.callback_url_base}/mpesa/reversal/timeout"
		payload = {
			"Initiator": self._config.initiator_name,
			"SecurityCredential": self._config.initiator_password,
			"CommandID": "TransactionReversal",
			"TransactionID": transaction_id,
			"Amount": str(amount),
			"ReceiverParty": self._config.shortcode,
			"RecieverIdentifierType": "4",
			"ResultURL": result_url,
			"QueueTimeOutURL": queue_url,
			"Remarks": remarks[:100],
			"Occasion": "",
		}
		resp = await self._client.post(
			"/mpesa/reversal/v1/request",
			json=payload,
			headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()


def mpesa_connector_from_env(tenant_id: str, user_id: str = "system") -> MPESAConnector:
	"""Construct MPESAConnector from environment variables."""
	config = MPESAConfiguration(
		name="MPESA Daraja",
		tenant_id=tenant_id,
		user_id=user_id,
		consumer_key=os.environ["MPESA_CONSUMER_KEY"],
		consumer_secret=os.environ["MPESA_CONSUMER_SECRET"],
		shortcode=os.environ["MPESA_SHORTCODE"],
		passkey=os.environ.get("MPESA_PASSKEY", ""),
		environment=os.environ.get("MPESA_ENV", "sandbox"),
		initiator_name=os.environ.get("MPESA_INITIATOR_NAME", ""),
		initiator_password=os.environ.get("MPESA_INITIATOR_PASSWORD", ""),
		callback_url_base=os.environ.get("MPESA_CALLBACK_URL_BASE", ""),
	)
	return MPESAConnector(config)
