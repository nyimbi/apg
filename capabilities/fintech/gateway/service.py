"""Dependency-light Fintech Gateway lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		GATEWAY_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CURRENCIES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_GATEWAY_AGENT_ROLES,
		SUPPORTED_GATEWAY_AGENT_RUNTIMES,
		SUPPORTED_PAYMENT_METHODS,
		SUPPORTED_PROVIDERS,
		SUPPORTED_PROVIDER_TYPES,
		SUPPORTED_RISK_LEVELS,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		GATEWAY_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CURRENCIES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_GATEWAY_AGENT_ROLES,
		SUPPORTED_GATEWAY_AGENT_RUNTIMES,
		SUPPORTED_PAYMENT_METHODS,
		SUPPORTED_PROVIDERS,
		SUPPORTED_PROVIDER_TYPES,
		SUPPORTED_RISK_LEVELS,
		evaluate_capability_rules,
		get_capability_contract,
	)


class FintechGatewayService:
	"""In-memory executable service for the gateway lifecycle packet."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None) -> None:
		if isinstance(tenant_id, dict):
			self.configuration = deepcopy(tenant_id)
			tenant_id = None
		else:
			self.configuration = {}
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.merchants: dict[str, dict[str, Any]] = {}
		self.provider_connections: dict[str, dict[str, Any]] = {}
		self.payment_methods: dict[str, dict[str, Any]] = {}
		self.payment_intents: dict[str, dict[str, Any]] = {}
		self.risk_reviews: dict[str, dict[str, Any]] = {}
		self.authorizations: dict[str, dict[str, Any]] = {}
		self.captures: dict[str, dict[str, Any]] = {}
		self.refunds: dict[str, dict[str, Any]] = {}
		self.webhooks: dict[str, dict[str, Any]] = {}
		self.settlements: dict[str, dict[str, Any]] = {}
		self.disputes: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": GATEWAY_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def onboard_merchant(
		self,
		merchant_id: str,
		tenant_id: str,
		merchant_code: str,
		legal_name: str,
		country: str,
		risk_level: str = "low",
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "onboard_merchant")
		context.update({
			"merchant_code_present": bool(merchant_code),
			"legal_name_present": bool(legal_name),
			"country_present": bool(country),
			"risk_level": risk_level,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("merchant", merchant_id),
			"type": "gateway_merchant",
			"tenant_id": tenant,
			"merchant_code": merchant_code,
			"legal_name": legal_name,
			"country": country,
			"risk_level": risk_level,
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.merchants[record["id"]] = record
		self._emit(tenant, "merchant_onboarded", record)
		return deepcopy(record)

	def connect_provider(
		self,
		connection_id: str,
		tenant_id: str,
		provider: str,
		provider_type: str,
		credential_reference: str,
		priority: int = 100,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "connect_provider")
		context.update({
			"provider_supported": provider in SUPPORTED_PROVIDERS,
			"provider_type_supported": provider_type in SUPPORTED_PROVIDER_TYPES,
			"credential_reference_present": bool(credential_reference),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("provider", connection_id),
			"type": "gateway_provider_connection",
			"tenant_id": tenant,
			"provider": provider,
			"provider_type": provider_type,
			"credential_reference": credential_reference,
			"priority": priority,
			"status": "active",
			"created_at": self._now(),
		}
		self.provider_connections[record["id"]] = record
		self._emit(tenant, "provider_connected", record)
		return deepcopy(record)

	def tokenize_payment_method(
		self,
		method_id: str,
		tenant_id: str,
		merchant_id: str,
		customer_reference: str,
		method_type: str,
		token_reference: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		merchant = self.merchants.get(merchant_id)
		context = self._base_context(tenant, "tokenize_payment_method")
		context.update({
			"merchant_present": bool(merchant and merchant["tenant_id"] == tenant),
			"customer_reference_present": bool(customer_reference),
			"payment_method_type_supported": method_type in SUPPORTED_PAYMENT_METHODS,
			"token_reference_present": bool(token_reference),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("method", method_id),
			"type": "gateway_payment_method",
			"tenant_id": tenant,
			"merchant_id": merchant_id,
			"customer_reference": customer_reference,
			"method_type": method_type,
			"token_reference": token_reference,
			"status": "active",
			"created_at": self._now(),
		}
		self.payment_methods[record["id"]] = record
		self._emit(tenant, "payment_method_tokenized", record)
		return deepcopy(record)

	def create_payment_intent(
		self,
		intent_id: str,
		tenant_id: str,
		merchant_id: str,
		payment_method_id: str,
		amount: float | int | Decimal,
		currency: str,
		description: str = "",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		merchant = self.merchants.get(merchant_id)
		method = self.payment_methods.get(payment_method_id)
		context = self._base_context(tenant, "create_payment_intent")
		context.update({
			"merchant_present": bool(merchant and merchant["tenant_id"] == tenant),
			"payment_method_present": bool(method and method["tenant_id"] == tenant),
			"amount": Decimal(str(amount)),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("intent", intent_id),
			"type": "gateway_payment_intent",
			"tenant_id": tenant,
			"merchant_id": merchant_id,
			"payment_method_id": payment_method_id,
			"amount": Decimal(str(amount)),
			"currency": currency,
			"description": description,
			"risk_level": "low",
			"provider_connection_id": None,
			"authorized_amount": Decimal("0"),
			"captured_amount": Decimal("0"),
			"refunded_amount": Decimal("0"),
			"status": "draft",
			"created_at": self._now(),
		}
		self.payment_intents[record["id"]] = record
		self._emit(tenant, "payment_intent_created", record)
		return deepcopy(record)

	def assess_payment_risk(
		self,
		review_id: str,
		tenant_id: str,
		payment_intent_id: str,
		risk_level: str,
		risk_score: float,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		intent = self.payment_intents.get(payment_intent_id)
		context = self._base_context(tenant, "assess_payment_risk")
		context.update({
			"payment_present": bool(intent and intent["tenant_id"] == tenant),
			"risk_level": risk_level,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		if risk_level not in SUPPORTED_RISK_LEVELS:
			raise PermissionError("risk_level_not_supported")
		record = {
			"id": self._record_id("risk", review_id),
			"type": "gateway_risk_review",
			"tenant_id": tenant,
			"payment_intent_id": payment_intent_id,
			"risk_level": risk_level,
			"risk_score": risk_score,
			"reviewed_by": reviewed_by,
			"status": "reviewed" if reviewed_by else "assessed",
			"created_at": self._now(),
		}
		self.risk_reviews[record["id"]] = record
		intent["risk_level"] = risk_level
		self._emit(tenant, "payment_risk_assessed", record)
		return deepcopy(record)

	def authorize_payment(
		self,
		authorization_id: str,
		tenant_id: str,
		payment_intent_id: str,
		provider_connection_id: str,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		intent = self.payment_intents.get(payment_intent_id)
		provider = self.provider_connections.get(provider_connection_id)
		amount = intent["amount"] if intent else Decimal("0")
		context = self._base_context(tenant, "authorize_payment")
		context.update({
			"provider_present": bool(provider and provider["tenant_id"] == tenant),
			"payment_present": bool(intent and intent["tenant_id"] == tenant),
			"risk_level": intent.get("risk_level") if intent else "blocked",
			"high_value": amount >= Decimal("10000"),
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		if not intent or intent["tenant_id"] != tenant:
			raise PermissionError("payment_intent_required")
		record = {
			"id": self._record_id("auth", authorization_id),
			"type": "gateway_authorization",
			"tenant_id": tenant,
			"payment_intent_id": payment_intent_id,
			"provider_connection_id": provider_connection_id,
			"amount": amount,
			"approved_by": approved_by,
			"status": "authorized",
			"created_at": self._now(),
		}
		self.authorizations[record["id"]] = record
		intent["provider_connection_id"] = provider_connection_id
		intent["authorized_amount"] = amount
		intent["status"] = "authorized"
		self._emit(tenant, "payment_authorized", record)
		return deepcopy(record)

	def capture_payment(self, capture_id: str, tenant_id: str, authorization_id: str, capture_amount: float | int | Decimal) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		authorization = self.authorizations.get(authorization_id)
		amount = Decimal(str(capture_amount))
		intent = self.payment_intents.get(authorization["payment_intent_id"]) if authorization else None
		context = self._base_context(tenant, "capture_payment")
		context.update({
			"authorized_payment_present": bool(authorization and authorization["tenant_id"] == tenant),
			"capture_amount": amount,
			"overcapture": bool(intent and amount > (intent["authorized_amount"] - intent["captured_amount"])),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("capture", capture_id),
			"type": "gateway_capture",
			"tenant_id": tenant,
			"authorization_id": authorization_id,
			"payment_intent_id": authorization["payment_intent_id"],
			"amount": amount,
			"status": "captured",
			"created_at": self._now(),
		}
		self.captures[record["id"]] = record
		intent["captured_amount"] += amount
		intent["status"] = "captured"
		self._emit(tenant, "payment_captured", record)
		return deepcopy(record)

	def refund_payment(
		self,
		refund_id: str,
		tenant_id: str,
		payment_intent_id: str,
		refund_amount: float | int | Decimal,
		reason: str,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		intent = self.payment_intents.get(payment_intent_id)
		amount = Decimal(str(refund_amount))
		refundable = (intent["captured_amount"] - intent["refunded_amount"]) if intent else Decimal("0")
		context = self._base_context(tenant, "refund_payment")
		context.update({
			"captured_payment_present": bool(intent and intent["tenant_id"] == tenant and intent["captured_amount"] > 0),
			"refund_amount": amount,
			"overrefund": amount > refundable,
			"large_refund": amount >= Decimal("5000"),
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("refund", refund_id),
			"type": "gateway_refund",
			"tenant_id": tenant,
			"payment_intent_id": payment_intent_id,
			"amount": amount,
			"reason": reason,
			"reviewed_by": reviewed_by,
			"status": "refunded",
			"created_at": self._now(),
		}
		self.refunds[record["id"]] = record
		intent["refunded_amount"] += amount
		intent["status"] = "refunded" if intent["refunded_amount"] == intent["captured_amount"] else "partially_refunded"
		self._emit(tenant, "payment_refunded", record)
		return deepcopy(record)

	def ingest_webhook(
		self,
		webhook_id: str,
		tenant_id: str,
		provider_connection_id: str,
		event_id: str,
		signature: str,
		idempotency_key: str,
		event_type: str,
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		provider = self.provider_connections.get(provider_connection_id)
		context = self._base_context(tenant, "ingest_webhook")
		context.update({
			"provider_present": bool(provider and provider["tenant_id"] == tenant),
			"event_id_present": bool(event_id),
			"signature_present": bool(signature),
			"idempotency_key_present": bool(idempotency_key),
		})
		self._assert_rules(context)
		if any(record["tenant_id"] == tenant and record["idempotency_key"] == idempotency_key for record in self.webhooks.values()):
			raise PermissionError("webhook_duplicate_idempotency_key")
		record = {
			"id": self._record_id("webhook", webhook_id),
			"type": "gateway_webhook",
			"tenant_id": tenant,
			"provider_connection_id": provider_connection_id,
			"event_id": event_id,
			"signature": signature,
			"idempotency_key": idempotency_key,
			"event_type": event_type,
			"payload": deepcopy(payload or {}),
			"status": "ingested",
			"created_at": self._now(),
		}
		self.webhooks[record["id"]] = record
		self._emit(tenant, "webhook_ingested", record)
		return deepcopy(record)

	def record_settlement(
		self,
		settlement_id: str,
		tenant_id: str,
		provider_connection_id: str,
		settlement_reference: str,
		amount: float | int | Decimal,
		expected_amount: float | int | Decimal | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		provider = self.provider_connections.get(provider_connection_id)
		settlement_amount = Decimal(str(amount))
		expected = Decimal(str(expected_amount)) if expected_amount is not None else settlement_amount
		variance = settlement_amount - expected
		context = self._base_context(tenant, "record_settlement")
		context.update({
			"provider_present": bool(provider and provider["tenant_id"] == tenant),
			"settlement_reference_present": bool(settlement_reference),
			"settlement_amount": settlement_amount,
			"variance_detected": variance != 0,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("settlement", settlement_id),
			"type": "gateway_settlement",
			"tenant_id": tenant,
			"provider_connection_id": provider_connection_id,
			"settlement_reference": settlement_reference,
			"amount": settlement_amount,
			"expected_amount": expected,
			"variance": variance,
			"reviewed_by": reviewed_by,
			"status": "settled",
			"created_at": self._now(),
		}
		self.settlements[record["id"]] = record
		self._emit(tenant, "settlement_recorded", record)
		return deepcopy(record)

	def open_dispute(self, dispute_id: str, tenant_id: str, payment_intent_id: str, reason: str, owner: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		intent = self.payment_intents.get(payment_intent_id)
		context = self._base_context(tenant, "open_dispute")
		context.update({
			"payment_present": bool(intent and intent["tenant_id"] == tenant),
			"dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS,
			"owner_present": bool(owner),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("dispute", dispute_id),
			"type": "gateway_dispute",
			"tenant_id": tenant,
			"payment_intent_id": payment_intent_id,
			"reason": reason,
			"owner": owner,
			"status": "open",
			"created_at": self._now(),
		}
		self.disputes[record["id"]] = record
		intent["status"] = "disputed"
		self._emit(tenant, "payment_dispute_opened", record)
		return deepcopy(record)

	def resolve_dispute(self, dispute_id: str, tenant_id: str, resolution: str, reviewed_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		dispute = self.disputes.get(dispute_id)
		context = self._base_context(tenant, "resolve_dispute")
		context.update({"resolution_review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		if not dispute or dispute["tenant_id"] != tenant:
			raise PermissionError("dispute_required")
		dispute["resolution"] = resolution
		dispute["reviewed_by"] = reviewed_by
		dispute["status"] = "resolved"
		intent = self.payment_intents.get(dispute["payment_intent_id"])
		if intent and intent["tenant_id"] == tenant and intent["status"] == "disputed":
			intent["status"] = "captured" if intent["captured_amount"] else "authorized"
		self._emit(tenant, "payment_dispute_resolved", dispute)
		return deepcopy(dispute)

	def register_gateway_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_gateway_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_GATEWAY_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_GATEWAY_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"),
			"type": "gateway_agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "gateway_agent_registered", record)
		return deepcopy(record)

	def validate_gateway_agent_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("gateway_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "gateway_agent_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "gateway_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant,
			"event_count": event_count,
			"processor": "bytewax",
			"stream": GATEWAY_EVENT_STREAM,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"merchant_count": len([record for record in self.merchants.values() if record["tenant_id"] == tenant]),
			"provider_count": len([record for record in self.provider_connections.values() if record["tenant_id"] == tenant]),
			"payment_method_count": len([record for record in self.payment_methods.values() if record["tenant_id"] == tenant]),
			"payment_intent_count": len([record for record in self.payment_intents.values() if record["tenant_id"] == tenant]),
			"risk_review_count": len([record for record in self.risk_reviews.values() if record["tenant_id"] == tenant]),
			"authorization_count": len([record for record in self.authorizations.values() if record["tenant_id"] == tenant]),
			"capture_count": len([record for record in self.captures.values() if record["tenant_id"] == tenant]),
			"refund_count": len([record for record in self.refunds.values() if record["tenant_id"] == tenant]),
			"webhook_count": len([record for record in self.webhooks.values() if record["tenant_id"] == tenant]),
			"settlement_count": len([record for record in self.settlements.values() if record["tenant_id"] == tenant]),
			"dispute_count": len([record for record in self.disputes.values() if record["tenant_id"] == tenant]),
			"gateway_agent_count": len([record for record in self.agents.values() if record["tenant_id"] == tenant]),
			"audit_event_count": len([event for event in self._audit_events if event["tenant_id"] == tenant]),
			"captured_volume": str(sum((record["amount"] for record in self.captures.values() if record["tenant_id"] == tenant), Decimal("0"))),
			"refund_volume": str(sum((record["amount"] for record in self.refunds.values() if record["tenant_id"] == tenant), Decimal("0"))),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")


GatewayService = FintechGatewayService
PaymentGatewayService = FintechGatewayService
