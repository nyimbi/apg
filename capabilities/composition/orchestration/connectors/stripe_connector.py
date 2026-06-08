"""Stripe API Connector.

Stripe is the world's leading payment infrastructure. The Stripe connector
integrates APG's fintech capabilities (fintech_gwy, fintech_trx) with
Stripe's payment processing, subscription billing, and payout APIs.

Reference: https://stripe.com/docs/api
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from pydantic import Field

from .base_connector import BaseConnector, ConnectorConfiguration

_log = logging.getLogger(__name__)
_STRIPE_BASE = "https://api.stripe.com"


class StripeConfiguration(ConnectorConfiguration):
	secret_key: str = Field(..., description="Stripe secret API key (sk_live_... or sk_test_...)")
	webhook_secret: str = Field(default="", description="Stripe webhook signing secret")
	api_version: str = Field(default="2024-11-20.acacia", description="Stripe API version")


class StripeConnector(BaseConnector):
	"""Stripe API connector.

	Supports:
	  - Payment intents (create, confirm, capture, cancel)
	  - Customers (create, retrieve, update)
	  - Subscriptions (create, update, cancel)
	  - Payouts (to connected accounts)
	  - Refunds
	  - Webhook event verification
	  - Disputes/chargebacks
	"""

	def __init__(self, config: StripeConfiguration) -> None:
		super().__init__(config)
		self._config: StripeConfiguration = config
		self._client: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(
			base_url=_STRIPE_BASE,
			timeout=self._config.timeout_seconds,
			headers={
				"Authorization": f"Bearer {self._config.secret_key}",
				"Stripe-Version": self._config.api_version,
				"Content-Type": "application/x-www-form-urlencoded",
			},
		)

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers = {
			"create_payment_intent": self._create_payment_intent,
			"confirm_payment_intent": self._confirm_payment_intent,
			"capture_payment_intent": self._capture_payment_intent,
			"cancel_payment_intent": self._cancel_payment_intent,
			"create_customer": self._create_customer,
			"retrieve_customer": self._retrieve_customer,
			"create_subscription": self._create_subscription,
			"cancel_subscription": self._cancel_subscription,
			"create_refund": self._create_refund,
			"create_payout": self._create_payout,
			"retrieve_balance": self._retrieve_balance,
			"list_charges": self._list_charges,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(f"Unknown Stripe operation: {operation!r}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			resp = await self._client.get("/v1/balance")
			return resp.status_code == 200
		except Exception:
			return False

	# ── Public helpers ─────────────────────────────────────────────────

	async def create_payment_intent(
		self,
		amount: int,
		currency: str,
		customer_id: str = "",
		metadata: dict[str, str] | None = None,
		capture_method: str = "automatic",
	) -> dict[str, Any]:
		"""Create a Stripe PaymentIntent.

		Args:
			amount: Amount in smallest currency unit (cents/fils/cents — no decimals)
			currency: 3-letter ISO currency code (kes, usd, eur, etc.)
		"""
		return await self._execute_operation("create_payment_intent", {
			"amount": amount, "currency": currency, "customer_id": customer_id,
			"metadata": metadata or {}, "capture_method": capture_method,
		})

	async def create_refund(
		self, payment_intent_id: str, amount: int | None = None, reason: str = ""
	) -> dict[str, Any]:
		return await self._execute_operation("create_refund", {
			"payment_intent_id": payment_intent_id, "amount": amount, "reason": reason,
		})

	async def retrieve_balance(self) -> dict[str, Any]:
		return await self._execute_operation("retrieve_balance", {})

	def verify_webhook_signature(
		self, payload: bytes, signature: str, secret: str = ""
	) -> bool:
		"""Verify Stripe webhook event signature using HMAC-SHA256."""
		import hashlib
		import hmac
		sec = secret or self._config.webhook_secret
		if not sec:
			return False
		# Stripe uses: timestamp.payload format with header 't=...,v1=...'
		try:
			parts = {p.split("=")[0]: p.split("=", 1)[1] for p in signature.split(",")}
			timestamp = parts.get("t", "")
			sig = parts.get("v1", "")
			signed_payload = f"{timestamp}.{payload.decode()}".encode()
			expected = hmac.new(sec.encode(), signed_payload, hashlib.sha256).hexdigest()
			return hmac.compare_digest(expected, sig)
		except Exception:
			return False

	# ── Private implementation ─────────────────────────────────────────

	def _flatten_params(self, d: dict[str, Any], prefix: str = "") -> dict[str, str]:
		"""Flatten nested dict to Stripe's form-encoded format."""
		result: dict[str, str] = {}
		for k, v in d.items():
			key = f"{prefix}[{k}]" if prefix else k
			if isinstance(v, dict):
				result.update(self._flatten_params(v, key))
			elif v is not None:
				result[key] = str(v)
		return result

	async def _create_payment_intent(
		self, amount: int, currency: str, customer_id: str,
		metadata: dict[str, str], capture_method: str,
	) -> dict[str, Any]:
		params = {"amount": str(amount), "currency": currency.lower(),
		          "capture_method": capture_method}
		if customer_id:
			params["customer"] = customer_id
		params.update(self._flatten_params(metadata, "metadata"))
		resp = await self._client.post("/v1/payment_intents", data=params)
		resp.raise_for_status()
		return resp.json()

	async def _confirm_payment_intent(self, payment_intent_id: str, **kwargs: Any) -> dict[str, Any]:
		resp = await self._client.post(
			f"/v1/payment_intents/{payment_intent_id}/confirm", data=kwargs
		)
		resp.raise_for_status()
		return resp.json()

	async def _capture_payment_intent(
		self, payment_intent_id: str, amount_to_capture: int | None = None
	) -> dict[str, Any]:
		data = {}
		if amount_to_capture:
			data["amount_to_capture"] = str(amount_to_capture)
		resp = await self._client.post(
			f"/v1/payment_intents/{payment_intent_id}/capture", data=data
		)
		resp.raise_for_status()
		return resp.json()

	async def _cancel_payment_intent(
		self, payment_intent_id: str, cancellation_reason: str = ""
	) -> dict[str, Any]:
		data = {}
		if cancellation_reason:
			data["cancellation_reason"] = cancellation_reason
		resp = await self._client.post(
			f"/v1/payment_intents/{payment_intent_id}/cancel", data=data
		)
		resp.raise_for_status()
		return resp.json()

	async def _create_customer(self, email: str, name: str = "", metadata: dict | None = None) -> dict[str, Any]:
		data = {"email": email}
		if name:
			data["name"] = name
		if metadata:
			data.update(self._flatten_params(metadata, "metadata"))
		resp = await self._client.post("/v1/customers", data=data)
		resp.raise_for_status()
		return resp.json()

	async def _retrieve_customer(self, customer_id: str) -> dict[str, Any]:
		resp = await self._client.get(f"/v1/customers/{customer_id}")
		resp.raise_for_status()
		return resp.json()

	async def _create_subscription(
		self, customer_id: str, price_id: str, trial_period_days: int = 0
	) -> dict[str, Any]:
		data = {"customer": customer_id, "items[0][price]": price_id}
		if trial_period_days:
			data["trial_period_days"] = str(trial_period_days)
		resp = await self._client.post("/v1/subscriptions", data=data)
		resp.raise_for_status()
		return resp.json()

	async def _cancel_subscription(
		self, subscription_id: str, at_period_end: bool = True
	) -> dict[str, Any]:
		if at_period_end:
			resp = await self._client.post(
				f"/v1/subscriptions/{subscription_id}",
				data={"cancel_at_period_end": "true"},
			)
		else:
			resp = await self._client.delete(f"/v1/subscriptions/{subscription_id}")
		resp.raise_for_status()
		return resp.json()

	async def _create_refund(
		self, payment_intent_id: str, amount: int | None, reason: str
	) -> dict[str, Any]:
		data = {"payment_intent": payment_intent_id}
		if amount:
			data["amount"] = str(amount)
		if reason:
			data["reason"] = reason
		resp = await self._client.post("/v1/refunds", data=data)
		resp.raise_for_status()
		return resp.json()

	async def _create_payout(
		self, amount: int, currency: str, destination: str = ""
	) -> dict[str, Any]:
		data = {"amount": str(amount), "currency": currency.lower()}
		if destination:
			data["destination"] = destination
		resp = await self._client.post("/v1/payouts", data=data)
		resp.raise_for_status()
		return resp.json()

	async def _retrieve_balance(self) -> dict[str, Any]:
		resp = await self._client.get("/v1/balance")
		resp.raise_for_status()
		return resp.json()

	async def _list_charges(
		self, limit: int = 10, customer_id: str = ""
	) -> dict[str, Any]:
		params: dict[str, str] = {"limit": str(limit)}
		if customer_id:
			params["customer"] = customer_id
		resp = await self._client.get("/v1/charges", params=params)
		resp.raise_for_status()
		return resp.json()


def stripe_connector_from_env(tenant_id: str, user_id: str = "system") -> StripeConnector:
	config = StripeConfiguration(
		name="Stripe",
		tenant_id=tenant_id,
		user_id=user_id,
		secret_key=os.environ["STRIPE_SECRET_KEY"],
		webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET", ""),
		api_version=os.environ.get("STRIPE_API_VERSION", "2024-11-20.acacia"),
	)
	return StripeConnector(config)
