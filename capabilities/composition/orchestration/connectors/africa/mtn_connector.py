"""MTN Mobile Money (MoMo) API Connector.

Implements APG's BaseConnector ABC for the MTN MoMo payment platform —
West and Central Africa's largest telecoms operator, active in 17 African
markets. Supports Collections (RequestToPay), Disbursements (Transfer),
account balance, transaction status, and KYC lookups.

OAuth2 API User + API Key credentials are exchanged for a bearer token per
OAuth2 client-credentials flow. Tokens are cached with a 60-second pre-expiry
buffer.

Reference:
    https://momodeveloper.mtn.com/
    https://momodeveloper.mtn.com/api-documentation/collection/
    https://momodeveloper.mtn.com/api-documentation/disbursement/

Configuration via environment variables or MTNConfiguration:
    MTN_API_USER_ID         UUID4 provisioned API user (sandbox or production)
    MTN_API_KEY             API key for the user
    MTN_SUBSCRIPTION_KEY    Ocp-Apim-Subscription-Key from the developer portal
    MTN_ENVIRONMENT         "sandbox" | "production" (default: sandbox)
    MTN_TARGET_ENVIRONMENT  "sandbox" | market-specific (e.g. "mtngh") – same
                            as MTN_ENVIRONMENT unless overridden
    MTN_CALLBACK_URL_BASE   Base URL for asynchronous MoMo callbacks
"""
from __future__ import annotations

import base64
import logging
import os
import time
import uuid
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

_SANDBOX_BASE = "https://sandbox.momodeveloper.mtn.com"
_PRODUCTION_BASE = "https://proxy.momoapi.mtn.com"

# Sub-product paths on the MoMo API
_COLLECTIONS_PREFIX = "/collection/v1_0"
_DISBURSEMENTS_PREFIX = "/disbursement/v1_0"
_REMITTANCES_PREFIX = "/remittance/v1_0"


class MTNConfiguration(ConnectorConfiguration):
	"""Configuration for the MTN MoMo connector.

	Markets: NG, GH, UG, CM, CI, ZM (and 11 more).
	"""

	api_user_id: str = Field(..., description="UUID4 API user provisioned on the developer portal")
	api_key: str = Field(..., description="API key for the provisioned API user")
	subscription_key: str = Field(..., description="Ocp-Apim-Subscription-Key from the portal")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	target_environment: str = Field(
		default="sandbox",
		description="MoMo target environment (sandbox or market code e.g. 'mtngh')",
	)
	callback_url_base: str = Field(default="", description="Base URL for MoMo async callbacks")


class MTNConnector(BaseConnector):
	"""MTN Mobile Money (MoMo) connector.

	Supports:
	  - Collections  — request_to_pay, get_transaction_status, get_account_balance,
	                   get_account_holder_info
	  - Disbursements — transfer, get_transaction_status (disbursement scope),
	                    get_account_balance
	All async operations return the parsed JSON body. Asynchronous delivery
	notifications are handled via the callback_url registered per-request.
	"""

	def __init__(self, config: MTNConfiguration) -> None:
		super().__init__(config)
		self._config: MTNConfiguration = config
		self._base_url = (
			_SANDBOX_BASE if config.environment == "sandbox" else _PRODUCTION_BASE
		)
		# Separate OAuth tokens for each product scope
		self._collections_token: str = ""
		self._collections_token_expires_at: float = 0.0
		self._disbursements_token: str = ""
		self._disbursements_token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None

	# ── BaseConnector abstract methods ────────────────────────────────────────

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=self._base_url,
			timeout=_TIMEOUT,
			headers={"Content-Type": "application/json"},
		)
		await self._refresh_token("collection")
		_log.info("MTN MoMo connector connected (%s)", self._config.environment)

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None
		self._collections_token = ""
		self._collections_token_expires_at = 0.0
		self._disbursements_token = ""
		self._disbursements_token_expires_at = 0.0

	async def _execute_operation(
		self, operation: str, parameters: dict[str, Any]
	) -> dict[str, Any]:
		"""Route operation name to the appropriate MoMo API call."""
		handlers: dict[str, Any] = {
			"request_to_pay": self._request_to_pay,
			"get_account_balance": self._get_account_balance,
			"get_transaction_status": self._get_transaction_status,
			"transfer": self._transfer,
			"get_account_holder_info": self._get_account_holder_info,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(
				f"Unknown MTN MoMo operation: {operation!r}. Valid: {list(handlers)}"
			)
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._refresh_token("collection")
			return bool(self._collections_token)
		except Exception:
			return False

	# ── Public operation methods ──────────────────────────────────────────────

	async def request_to_pay(
		self,
		amount: str,
		phone: str,
		currency: str,
		external_id: str,
		payer_message: str = "Payment",
		payee_note: str = "Payment",
		callback_url: str = "",
	) -> dict[str, Any]:
		"""Initiate a collections RequestToPay.

		Sends an STK-style prompt to the subscriber's phone. Returns the
		reference_id (UUID) which can be used to poll transaction status.

		Args:
			amount:       Amount as a string (e.g. "1000")
			phone:        MSISDN in international format without + (e.g. "256712345678")
			currency:     ISO 4217 code (e.g. "UGX", "GHS", "NGN", "ZMW")
			external_id:  Merchant reference ID (idempotency key)
			payer_message: Message shown to the payer on their phone
			payee_note:   Internal note for the payee
			callback_url: Override default callback URL
		"""
		return await self._execute_operation("request_to_pay", {
			"amount": amount,
			"phone": phone,
			"currency": currency,
			"external_id": external_id,
			"payer_message": payer_message,
			"payee_note": payee_note,
			"callback_url": callback_url,
		})

	async def get_account_balance(self, product: str = "collection") -> dict[str, Any]:
		"""Query account balance for 'collection' or 'disbursement' product."""
		return await self._execute_operation("get_account_balance", {"product": product})

	async def get_transaction_status(
		self, reference_id: str, product: str = "collection"
	) -> dict[str, Any]:
		"""Query the status of a RequestToPay or Transfer by reference_id."""
		return await self._execute_operation("get_transaction_status", {
			"reference_id": reference_id,
			"product": product,
		})

	async def transfer(
		self,
		amount: str,
		phone: str,
		currency: str,
		external_id: str,
		payee_note: str = "Transfer",
		payer_message: str = "Transfer",
		callback_url: str = "",
	) -> dict[str, Any]:
		"""Initiate a disbursement Transfer (B2C payout) to a mobile subscriber."""
		return await self._execute_operation("transfer", {
			"amount": amount,
			"phone": phone,
			"currency": currency,
			"external_id": external_id,
			"payee_note": payee_note,
			"payer_message": payer_message,
			"callback_url": callback_url,
		})

	async def get_account_holder_info(self, msisdn: str) -> dict[str, Any]:
		"""Return KYC details for a registered MoMo subscriber (Collections scope)."""
		return await self._execute_operation("get_account_holder_info", {"msisdn": msisdn})

	# ── Private implementation ────────────────────────────────────────────────

	async def _refresh_token(self, product: str) -> None:
		"""Fetch or reuse a cached OAuth2 bearer token for the given product scope."""
		now = time.time()
		if product == "collection":
			if now < self._collections_token_expires_at - 60:
				return
		else:
			if now < self._disbursements_token_expires_at - 60:
				return

		creds = base64.b64encode(
			f"{self._config.api_user_id}:{self._config.api_key}".encode()
		).decode()
		product_path = (
			"/collection/token/" if product == "collection" else "/disbursement/token/"
		)
		client = self._client or httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)
		resp = await client.post(
			product_path,
			headers={
				"Authorization": f"Basic {creds}",
				"Ocp-Apim-Subscription-Key": self._config.subscription_key,
			},
		)
		resp.raise_for_status()
		data = resp.json()
		token = data["access_token"]
		expires_at = now + int(data.get("expires_in", 3600))
		if product == "collection":
			self._collections_token = token
			self._collections_token_expires_at = expires_at
		else:
			self._disbursements_token = token
			self._disbursements_token_expires_at = expires_at
		_log.debug("MTN MoMo %s token refreshed", product)

	def _collection_headers(self) -> dict[str, str]:
		return {
			"Authorization": f"Bearer {self._collections_token}",
			"Ocp-Apim-Subscription-Key": self._config.subscription_key,
			"X-Target-Environment": self._config.target_environment,
		}

	def _disbursement_headers(self) -> dict[str, str]:
		return {
			"Authorization": f"Bearer {self._disbursements_token}",
			"Ocp-Apim-Subscription-Key": self._config.subscription_key,
			"X-Target-Environment": self._config.target_environment,
		}

	async def _request_to_pay(
		self,
		amount: str,
		phone: str,
		currency: str,
		external_id: str,
		payer_message: str,
		payee_note: str,
		callback_url: str,
	) -> dict[str, Any]:
		await self._refresh_token("collection")
		reference_id = str(uuid.uuid4())
		cb = callback_url or f"{self._config.callback_url_base}/mtn/collection/callback"
		payload = {
			"amount": amount,
			"currency": currency,
			"externalId": external_id,
			"payer": {"partyIdType": "MSISDN", "partyId": phone},
			"payerMessage": payer_message[:160],
			"payeeNote": payee_note[:160],
		}
		headers = {
			**self._collection_headers(),
			"X-Reference-Id": reference_id,
			"X-Callback-Url": cb,
			"Content-Type": "application/json",
		}
		resp = await self._client.post(
			f"{_COLLECTIONS_PREFIX}/requesttopay",
			json=payload,
			headers=headers,
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return {"reference_id": reference_id, "status": resp.status_code}

	async def _get_account_balance(self, product: str) -> dict[str, Any]:
		if product == "collection":
			await self._refresh_token("collection")
			path = f"{_COLLECTIONS_PREFIX}/account/balance"
			headers = self._collection_headers()
		else:
			await self._refresh_token("disbursement")
			path = f"{_DISBURSEMENTS_PREFIX}/account/balance"
			headers = self._disbursement_headers()
		resp = await self._client.get(path, headers=headers, timeout=_TIMEOUT)
		resp.raise_for_status()
		return resp.json()

	async def _get_transaction_status(
		self, reference_id: str, product: str
	) -> dict[str, Any]:
		if product == "collection":
			await self._refresh_token("collection")
			path = f"{_COLLECTIONS_PREFIX}/requesttopay/{reference_id}"
			headers = self._collection_headers()
		else:
			await self._refresh_token("disbursement")
			path = f"{_DISBURSEMENTS_PREFIX}/transfer/{reference_id}"
			headers = self._disbursement_headers()
		resp = await self._client.get(path, headers=headers, timeout=_TIMEOUT)
		resp.raise_for_status()
		return resp.json()

	async def _transfer(
		self,
		amount: str,
		phone: str,
		currency: str,
		external_id: str,
		payee_note: str,
		payer_message: str,
		callback_url: str,
	) -> dict[str, Any]:
		await self._refresh_token("disbursement")
		reference_id = str(uuid.uuid4())
		cb = callback_url or f"{self._config.callback_url_base}/mtn/disbursement/callback"
		payload = {
			"amount": amount,
			"currency": currency,
			"externalId": external_id,
			"payee": {"partyIdType": "MSISDN", "partyId": phone},
			"payerMessage": payer_message[:160],
			"payeeNote": payee_note[:160],
		}
		headers = {
			**self._disbursement_headers(),
			"X-Reference-Id": reference_id,
			"X-Callback-Url": cb,
			"Content-Type": "application/json",
		}
		resp = await self._client.post(
			f"{_DISBURSEMENTS_PREFIX}/transfer",
			json=payload,
			headers=headers,
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return {"reference_id": reference_id, "status": resp.status_code}

	async def _get_account_holder_info(self, msisdn: str) -> dict[str, Any]:
		await self._refresh_token("collection")
		resp = await self._client.get(
			f"{_COLLECTIONS_PREFIX}/accountholder/msisdn/{msisdn}/basicuserinfo",
			headers=self._collection_headers(),
			timeout=_TIMEOUT,
		)
		resp.raise_for_status()
		return resp.json()


def mtn_connector_from_env(tenant_id: str, user_id: str = "system") -> MTNConnector:
	"""Construct MTNConnector from environment variables.

	Required env vars:
	    MTN_API_USER_ID, MTN_API_KEY, MTN_SUBSCRIPTION_KEY

	Optional:
	    MTN_ENVIRONMENT, MTN_TARGET_ENVIRONMENT, MTN_CALLBACK_URL_BASE
	"""
	config = MTNConfiguration(
		name="MTN MoMo",
		tenant_id=tenant_id,
		user_id=user_id,
		api_user_id=os.environ["MTN_API_USER_ID"],
		api_key=os.environ["MTN_API_KEY"],
		subscription_key=os.environ["MTN_SUBSCRIPTION_KEY"],
		environment=os.environ.get("MTN_ENVIRONMENT", "sandbox"),
		target_environment=os.environ.get("MTN_TARGET_ENVIRONMENT", "sandbox"),
		callback_url_base=os.environ.get("MTN_CALLBACK_URL_BASE", ""),
	)
	return MTNConnector(config)
