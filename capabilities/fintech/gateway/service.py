"""Dependency-light Fintech Gateway lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

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
		# Only hard-block on explicit deny; require_review creates an audit flag
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

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

	# ── Additional methods ────────────────────────────────────────────────

	def health_check(self) -> dict[str, Any]:
		"""Return gateway service health status."""
		return {
			"service": "gateway",
			"status": "healthy",
			"merchant_count": len(self.merchants),
			"provider_count": len(self.provider_connections),
			"pending_intents": sum(1 for r in self.payment_intents.values() if r["status"] in {"draft", "authorized"}),
			"open_disputes": sum(1 for r in self.disputes.values() if r["status"] == "open"),
			"checked_at": self._now(),
		}

	def bulk_onboard_merchants(self, tenant_id: str, merchants: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-onboard multiple merchants in one call."""
		tenant = self._tenant(tenant_id)
		results, errors = [], []
		for m in merchants:
			try:
				rec = self.onboard_merchant(
					merchant_id=m.get("merchant_id", self._record_id("merchant")),
					tenant_id=tenant_id,
					merchant_code=m["merchant_code"],
					legal_name=m["legal_name"],
					country=m["country"],
					risk_level=m.get("risk_level", "low"),
					reviewed_by=m.get("reviewed_by"),
				)
				results.append(rec)
			except Exception as exc:
				errors.append({"input": m, "error": str(exc)})
		return {"processed": len(results), "failed": len(errors), "merchants": results, "errors": errors}

	def list_merchants(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all merchants for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.merchants.values() if r["tenant_id"] == tenant]

	def suspend_merchant(self, merchant_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Suspend an active merchant account."""
		tenant = self._tenant(tenant_id)
		merchant = self.merchants.get(merchant_id)
		if not merchant or merchant["tenant_id"] != tenant:
			raise PermissionError("merchant_required")
		merchant["status"] = "suspended"
		merchant["suspension_reason"] = reason
		merchant["suspended_at"] = self._now()
		self._emit(tenant, "merchant_suspended", merchant)
		return deepcopy(merchant)

	def reactivate_merchant(self, merchant_id: str, tenant_id: str, reviewed_by: str) -> dict[str, Any]:
		"""Reactivate a suspended merchant."""
		tenant = self._tenant(tenant_id)
		merchant = self.merchants.get(merchant_id)
		if not merchant or merchant["tenant_id"] != tenant:
			raise PermissionError("merchant_required")
		merchant["status"] = "active"
		merchant["reactivated_by"] = reviewed_by
		merchant["reactivated_at"] = self._now()
		self._emit(tenant, "merchant_reactivated", merchant)
		return deepcopy(merchant)

	def list_payment_intents(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List payment intents, optionally filtered by status."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.payment_intents.values() if r["tenant_id"] == tenant]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	def void_payment_intent(self, intent_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Void an unauthorised payment intent."""
		tenant = self._tenant(tenant_id)
		intent = self.payment_intents.get(intent_id)
		if not intent or intent["tenant_id"] != tenant:
			raise PermissionError("payment_intent_required")
		if intent["status"] not in {"draft"}:
			raise PermissionError("only_draft_intents_can_be_voided")
		intent["status"] = "voided"
		intent["void_reason"] = reason
		intent["voided_at"] = self._now()
		self._emit(tenant, "payment_intent_voided", intent)
		return deepcopy(intent)

	def provider_failover(self, tenant_id: str, primary_provider_id: str, fallback_provider_id: str) -> dict[str, Any]:
		"""Trigger failover from primary to fallback provider for pending intents."""
		tenant = self._tenant(tenant_id)
		primary = self.provider_connections.get(primary_provider_id)
		fallback = self.provider_connections.get(fallback_provider_id)
		if not primary or primary["tenant_id"] != tenant:
			raise PermissionError("primary_provider_required")
		if not fallback or fallback["tenant_id"] != tenant:
			raise PermissionError("fallback_provider_required")
		rerouted = 0
		for intent in self.payment_intents.values():
			if intent["tenant_id"] == tenant and intent.get("provider_connection_id") == primary_provider_id and intent["status"] in {"authorized"}:
				intent["provider_connection_id"] = fallback_provider_id
				intent["failover_at"] = self._now()
				rerouted += 1
		rec = {"type": "gateway_failover", "id": self._record_id("failover"), "tenant_id": tenant,
			   "primary": primary_provider_id, "fallback": fallback_provider_id,
			   "rerouted": rerouted, "status": "active", "created_at": self._now()}
		self._emit(tenant, "provider_failover_activated", rec)
		return rec

	def reconcile_settlements(self, tenant_id: str, period_start: str, period_end: str) -> dict[str, Any]:
		"""Reconcile settlement records vs captures for a date range."""
		tenant = self._tenant(tenant_id)
		caps = [r for r in self.captures.values() if r["tenant_id"] == tenant and period_start <= r["created_at"][:10] <= period_end]
		settlements = [r for r in self.settlements.values() if r["tenant_id"] == tenant and period_start <= r["created_at"][:10] <= period_end]
		captured_total = sum(r["amount"] for r in caps)
		settled_total = sum(r["amount"] for r in settlements)
		variance = captured_total - settled_total
		return {
			"tenant_id": tenant, "period_start": period_start, "period_end": period_end,
			"captured_total": str(captured_total), "settled_total": str(settled_total),
			"variance": str(variance), "status": "balanced" if abs(variance) < Decimal("1") else "variance",
			"generated_at": self._now(),
		}

	def fraud_risk_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate fraud risk metrics from risk reviews."""
		tenant = self._tenant(tenant_id)
		reviews = [r for r in self.risk_reviews.values() if r["tenant_id"] == tenant]
		by_level: dict[str, int] = {}
		for r in reviews:
			by_level[r["risk_level"]] = by_level.get(r["risk_level"], 0) + 1
		high_risk = [r["payment_intent_id"] for r in reviews if r["risk_level"] in {"high", "critical"}]
		return {
			"tenant_id": tenant, "total_reviews": len(reviews),
			"by_risk_level": by_level, "high_risk_intents": high_risk,
			"generated_at": self._now(),
		}

	def export_transactions(self, tenant_id: str, fmt: str = "json") -> dict[str, Any]:
		"""Export transaction records in JSON or CSV format metadata."""
		tenant = self._tenant(tenant_id)
		assert fmt in {"json", "csv", "excel"}, "fmt must be json|csv|excel"
		intents = [r for r in self.payment_intents.values() if r["tenant_id"] == tenant]
		return {
			"tenant_id": tenant, "format": fmt, "record_count": len(intents),
			"export_reference": f"export-{tenant}-{self._now()[:10]}.{fmt}",
			"generated_at": self._now(),
		}

	def mpesa_stk_push(self, tenant_id: str, phone: str, amount: float, reference: str, description: str = "") -> dict[str, Any]:
		"""Initiate an M-Pesa STK Push payment intent."""
		tenant = self._tenant(tenant_id)
		if not phone or not phone.startswith(("07", "01", "254", "+254")):
			raise ValueError("invalid_mpesa_phone_number")
		record = {
			"id": self._record_id("mpesa"),
			"type": "mpesa_stk_push",
			"tenant_id": tenant,
			"phone": phone[-9:].zfill(9),
			"amount": Decimal(str(amount)),
			"reference": reference,
			"description": description,
			"payment_method": "mpesa",
			"status": "pending",
			"created_at": self._now(),
		}
		self.payment_intents[record["id"]] = record
		self._emit(tenant, "mpesa_stk_push_initiated", record)
		return deepcopy(record)

	def mpesa_b2b_transfer(self, tenant_id: str, sender_till: str, receiver_till: str, amount: float, reference: str) -> dict[str, Any]:
		"""Initiate M-Pesa B2B (business to business) transfer."""
		tenant = self._tenant(tenant_id)
		record = {
			"id": self._record_id("b2b"),
			"type": "mpesa_b2b_transfer",
			"tenant_id": tenant,
			"sender_till": sender_till,
			"receiver_till": receiver_till,
			"amount": Decimal(str(amount)),
			"reference": reference,
			"status": "pending",
			"created_at": self._now(),
		}
		self.payment_intents[record["id"]] = record
		self._emit(tenant, "mpesa_b2b_initiated", record)
		return deepcopy(record)

	def pesalink_transfer(self, tenant_id: str, account_number: str, bank_code: str, amount: float, reference: str) -> dict[str, Any]:
		"""Initiate a PesaLink interbank transfer (CBK-cleared)."""
		tenant = self._tenant(tenant_id)
		if amount > 999_999:
			raise ValueError("pesalink_max_999999")
		record = {
			"id": self._record_id("pesalink"),
			"type": "pesalink_transfer",
			"tenant_id": tenant,
			"account_number": account_number,
			"bank_code": bank_code,
			"amount": Decimal(str(amount)),
			"reference": reference,
			"payment_rail": "pesalink",
			"status": "pending",
			"created_at": self._now(),
		}
		self.payment_intents[record["id"]] = record
		self._emit(tenant, "pesalink_transfer_initiated", record)
		return deepcopy(record)

	def rtgs_payment(self, tenant_id: str, beneficiary_account: str, bank_code: str, amount: float, reference: str, approved_by: str) -> dict[str, Any]:
		"""Initiate a high-value RTGS payment (KES 1M+)."""
		tenant = self._tenant(tenant_id)
		if amount < 1_000_000:
			raise ValueError("rtgs_minimum_1000000")
		if not approved_by:
			raise PermissionError("rtgs_requires_approval")
		record = {
			"id": self._record_id("rtgs"),
			"type": "rtgs_payment",
			"tenant_id": tenant,
			"beneficiary_account": beneficiary_account,
			"bank_code": bank_code,
			"amount": Decimal(str(amount)),
			"reference": reference,
			"approved_by": approved_by,
			"payment_rail": "rtgs",
			"status": "pending",
			"created_at": self._now(),
		}
		self.payment_intents[record["id"]] = record
		self._emit(tenant, "rtgs_payment_initiated", record)
		return deepcopy(record)

	def cbk_return_filing(self, tenant_id: str, period: str, return_type: str, submitted_by: str) -> dict[str, Any]:
		"""Generate a CBK gateway regulatory return for a period."""
		tenant = self._tenant(tenant_id)
		intents = [r for r in self.payment_intents.values() if r["tenant_id"] == tenant]
		captured = [r for r in self.captures.values() if r["tenant_id"] == tenant]
		refunded = [r for r in self.refunds.values() if r["tenant_id"] == tenant]
		return {
			"return_id": self._record_id("cbk_return"),
			"tenant_id": tenant,
			"period": period,
			"return_type": return_type,
			"submitted_by": submitted_by,
			"total_intents": len(intents),
			"total_captures": len(captured),
			"total_refunds": len(refunded),
			"captured_volume": str(sum(r["amount"] for r in captured)),
			"refunded_volume": str(sum(r["amount"] for r in refunded)),
			"status": "filed",
			"filed_at": self._now(),
		}

	def gateway_fee_schedule(self, tenant_id: str, provider_id: str) -> dict[str, Any]:
		"""Return the fee schedule for a gateway provider connection."""
		tenant = self._tenant(tenant_id)
		provider = self.provider_connections.get(provider_id)
		if not provider or provider["tenant_id"] != tenant:
			raise PermissionError("provider_required")
		fee_table = {
			"visa": {"rate_pct": 1.5, "flat_kes": 0},
			"mastercard": {"rate_pct": 1.5, "flat_kes": 0},
			"mpesa": {"rate_pct": 1.0, "flat_kes": 0},
			"pesalink": {"rate_pct": 0.0, "flat_kes": 50},
			"rtgs": {"rate_pct": 0.0, "flat_kes": 500},
			"interbank": {"rate_pct": 0.5, "flat_kes": 20},
		}
		ptype = provider.get("provider_type", "interbank")
		fees = fee_table.get(ptype, fee_table["interbank"])
		return {"provider_id": provider_id, "provider_type": ptype, "fees": fees, "currency": "KES", "as_of": self._now()}

	def export_settlements(self, tenant_id: str, period_start: str, period_end: str) -> list[dict[str, Any]]:
		"""Export settlement records for a date range."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.settlements.values() if r["tenant_id"] == tenant and period_start <= r["created_at"][:10] <= period_end]

	def list_disputes(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List disputes for a tenant, optionally filtered by status."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.disputes.values() if r["tenant_id"] == tenant]
		return [r for r in items if r["status"] == status] if status else items

	def payment_method_deactivate(self, method_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Deactivate a tokenised payment method."""
		tenant = self._tenant(tenant_id)
		method = self.payment_methods.get(method_id)
		if not method or method["tenant_id"] != tenant:
			raise PermissionError("payment_method_required")
		method["status"] = "deactivated"
		method["deactivation_reason"] = reason
		method["deactivated_at"] = self._now()
		self._emit(tenant, "payment_method_deactivated", method)
		return deepcopy(method)

	def webhook_retry(self, tenant_id: str, webhook_id: str) -> dict[str, Any]:
		"""Re-dispatch a failed webhook event."""
		tenant = self._tenant(tenant_id)
		webhook = self.webhooks.get(webhook_id)
		if not webhook or webhook["tenant_id"] != tenant:
			raise PermissionError("webhook_required")
		webhook["status"] = "retried"
		webhook["retried_at"] = self._now()
		self._emit(tenant, "webhook_retried", webhook)
		return deepcopy(webhook)


GatewayService = FintechGatewayService
PaymentGatewayService = FintechGatewayService
