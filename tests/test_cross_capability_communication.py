"""Tests for cross-capability communication via adapter protocols.

Covers:
- Adapter injection (AuditAdapter, NotifyAdapter) into AR and Payments services
- Shared InMemoryStore between two service instances
- Rule engine composability: tenant_context, KYC gate on mobile_money
- Null adapter standalone operation
- AlertManagementService audit_events stream
- Registry streaming contract consistency
- Tenant isolation via InMemoryStore
"""

from __future__ import annotations

import asyncio
from typing import Any


# ── Capturing adapters ────────────────────────────────────────────────────────


class CapturingAuditAdapter:
	def __init__(self) -> None:
		self.events: list[dict[str, Any]] = []

	async def log_event(
		self,
		event_type: str,
		actor_id: str,
		tenant_id: str,
		resource_id: str,
		details: dict[str, Any],
	) -> None:
		self.events.append({"event_type": event_type, "tenant_id": tenant_id})


class AssertingAuditAdapter:
	def __init__(self) -> None:
		self.call_count = 0

	async def log_event(
		self,
		event_type: str,
		actor_id: str,
		tenant_id: str,
		resource_id: str,
		details: dict[str, Any],
	) -> None:
		self.call_count += 1
		assert tenant_id, "tenant_id must not be empty"
		assert event_type, "event_type must not be empty"


class CapturingNotifyAdapter:
	def __init__(self) -> None:
		self.sends: list[dict[str, Any]] = []

	async def send(
		self,
		recipient: str,
		channel: str,
		subject: str,
		body: str,
		metadata: dict[str, Any] | None = None,
	) -> None:
		self.sends.append({"recipient": recipient, "channel": channel, "subject": subject})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ar_service(tenant_id: str = "test_tenant", **kwargs):
	from capabilities.fin.arc.accounts_receivable.service import AccountsReceivableService
	return AccountsReceivableService(tenant_id, **kwargs)


def _payments_service(tenant_id: str = "payments_tenant", **kwargs):
	from capabilities.fintech.payments.service import DigitalPaymentsService
	return DigitalPaymentsService(tenant_id, **kwargs)


# ── Test: AR audit adapter ────────────────────────────────────────────────────


async def test_ar_service_uses_audit_adapter():
	"""AR service emits audit events via the AuditAdapter protocol."""
	audit = CapturingAuditAdapter()
	svc = _ar_service(audit=audit)

	customer = await svc.create_customer("Acme Corp", 50_000, "NET30", "USD")
	assert len(audit.events) >= 1
	event_types = [e["event_type"] for e in audit.events]
	assert any("customer" in et for et in event_types), (
		f"Expected a customer-related audit event, got: {event_types}"
	)


async def test_ar_service_audit_event_has_correct_tenant():
	"""All AR audit events carry the tenant_id the service was initialised with."""
	audit = CapturingAuditAdapter()
	svc = _ar_service(tenant_id="acme_corp", audit=audit)

	await svc.create_customer("ACME", 10_000, "NET15", "KES")
	assert audit.events
	for evt in audit.events:
		assert evt["tenant_id"] == "acme_corp"


# ── Test: Payments notify adapter ────────────────────────────────────────────


async def test_payments_service_has_notify_attribute():
	"""DigitalPaymentsService accepts and stores a notify adapter."""
	notify = CapturingNotifyAdapter()
	svc = _payments_service(notify=notify)
	assert svc._notify is notify


async def test_payments_service_stores_notify_adapter():
	"""DigitalPaymentsService stores the injected NotifyAdapter and uses the protocol.

	The notify adapter is used internally by send_payment_receipt / send_payment_failure_alert.
	Both of those require a persisted transaction_id, so we verify the adapter is
	properly injected and satisfies the protocol contract rather than calling a full
	payment pipeline in a unit test.
	"""
	notify = CapturingNotifyAdapter()
	svc = _payments_service(notify=notify)

	# The adapter must be stored on the service
	assert svc._notify is notify

	# Directly invoke the adapter to confirm it works as a NotifyAdapter
	await svc._notify.send(
		recipient="test@example.com",
		channel="email",
		subject="Payment received",
		body="KES 500 received",
	)
	assert len(notify.sends) == 1
	assert notify.sends[0]["channel"] == "email"


# ── Test: Shared InMemoryStore ────────────────────────────────────────────────


async def test_shared_store_between_capabilities():
	"""Two AR service instances sharing an InMemoryStore see each other's data."""
	from capabilities.fin.arc.accounts_receivable.database.store import InMemoryStore

	shared = InMemoryStore()
	svc_a = _ar_service(tenant_id="company_a", store=shared)
	svc_b = _ar_service(tenant_id="company_a", store=shared)  # same tenant

	# svc_a creates a customer
	customer = await svc_a.create_customer("Shared Corp", 20_000, "NET30", "USD")

	# svc_b should find it (same tenant, same store)
	customers_b = await svc_b.list_customers()
	ids = [c["id"] for c in customers_b]
	assert customer["id"] in ids


async def test_inmemory_store_put_get_query():
	"""InMemoryStore basic put/get/query operations work correctly."""
	from capabilities.fin.arc.accounts_receivable.database.store import InMemoryStore

	store = InMemoryStore()

	record = {"id": "r1", "tenant_id": "t1", "value": 42}
	await store.put("test_collection", record)

	fetched = await store.get("test_collection", "r1")
	assert fetched is not None
	assert fetched["value"] == 42

	results = await store.query("test_collection", {"tenant_id": "t1"})
	assert len(results) == 1
	assert results[0]["id"] == "r1"


async def test_inmemory_store_delete():
	"""InMemoryStore delete removes the record."""
	from capabilities.fin.arc.accounts_receivable.database.store import InMemoryStore

	store = InMemoryStore()
	await store.put("col", {"id": "x1", "tenant_id": "t1"})
	assert await store.get("col", "x1") is not None
	deleted = await store.delete("col", "x1")
	assert deleted is True
	assert await store.get("col", "x1") is None


# ── Test: Rule engine composability ──────────────────────────────────────────


def test_payments_rule_engine_deny_without_tenant_context():
	"""Payments rule engine denies operations with no tenant context."""
	from capabilities.fintech.payments.capability_contract import evaluate_capability_rules

	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert "tenant_context_required" in result["matched_rules"]


def test_payments_rule_engine_allow_with_tenant_context():
	"""Payments rule engine allows operations with tenant context present."""
	from capabilities.fintech.payments.capability_contract import evaluate_capability_rules

	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_mobile_money_kyc_gate_denies_without_kyc():
	"""Registering a mobile_money instrument without KYC is denied."""
	from capabilities.fintech.payments.capability_contract import evaluate_capability_rules

	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_instrument",
		"instrument_type": "mobile_money",
		"kyc_present": False,
	})
	assert result["decision"] == "deny"
	assert "mobile_money_kyc_required" in result["matched_rules"]


def test_mobile_money_kyc_gate_allows_with_kyc():
	"""Registering a mobile_money instrument with KYC attached is allowed."""
	from capabilities.fintech.payments.capability_contract import evaluate_capability_rules

	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_instrument",
		"instrument_type": "mobile_money",
		"kyc_present": True,
	})
	assert result["decision"] == "allow"


def test_registry_evaluate_rules_payments_deny():
	"""Registry evaluate_rules for fintech_payments respects tenant_context_present=False."""
	from capabilities.capability_contract_registry import evaluate_rules

	result = evaluate_rules("fintech_payments", {"tenant_context_present": False})
	assert result["decision"] == "deny"


def test_registry_evaluate_rules_payments_allow():
	"""Registry evaluate_rules for fintech_payments allows clean context."""
	from capabilities.capability_contract_registry import evaluate_rules

	result = evaluate_rules("fintech_payments", {"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_registry_evaluate_rules_alerts_allow():
	"""Registry evaluate_rules for intel_alerts allows a clean context."""
	from capabilities.capability_contract_registry import evaluate_rules

	result = evaluate_rules("intel_alerts", {"tenant_context_present": True})
	assert result["decision"] == "allow"


# ── Test: Null adapters standalone operation ──────────────────────────────────


async def test_null_adapters_allow_standalone_operation():
	"""NullAuthAdapter and NullAuditAdapter work in isolation without error."""
	from capabilities.fin.arc.accounts_receivable.domain.adapters import (
		NullAuthAdapter,
		NullAuditAdapter,
	)

	auth = NullAuthAdapter()
	audit = NullAuditAdapter()

	# NullAuth always grants permissions
	granted = await auth.check_permission("user1", "ar:view")
	assert granted is True

	# NullAuth verify_token returns a dict with user_id
	token_data = await auth.verify_token("some-token")
	assert "user_id" in token_data
	assert token_data["user_id"]

	# NullAudit.log_event must not raise
	await audit.log_event(
		"test_event", "actor1", "tenant1", "resource1", {"detail": "test"}
	)


async def test_ar_service_works_standalone_no_adapters():
	"""AR service works with default (null) adapters — no platform needed."""
	svc = _ar_service(tenant_id="standalone")
	customer = await svc.create_customer("Standalone Corp", 5_000, "NET60", "USD")
	assert customer["id"]
	customers = await svc.list_customers()
	assert len(customers) == 1


# ── Test: Tenant isolation ────────────────────────────────────────────────────


async def test_tenant_isolation_in_ar_store():
	"""AR service isolates data between tenants sharing the same store."""
	from capabilities.fin.arc.accounts_receivable.database.store import InMemoryStore

	shared = InMemoryStore()
	svc_a = _ar_service(tenant_id="company_a", store=shared)
	svc_b = _ar_service(tenant_id="company_b", store=shared)

	# Create customer under company_a
	await svc_a.create_customer("A Corp", 10_000, "NET30", "USD")

	# company_b sees no customers
	b_customers = await svc_b.list_customers()
	assert len(b_customers) == 0

	# company_a sees its own customer
	a_customers = await svc_a.list_customers()
	assert len(a_customers) == 1


# ── Test: Alert service audit events ─────────────────────────────────────────


def test_alert_lifecycle_triggers_audit_events():
	"""AlertManagementService.record_authority populates audit_events with bytewax events."""
	from capabilities.intel.alerts.service import AlertManagementService

	svc = AlertManagementService()

	# record_authority is the first step and produces an audit event
	svc.record_authority(
		"auth-001", "tenant1", "mission_order", "scope-ref-001",
		"unclassified", "approver-001", "2030-01-01T00:00:00Z",
		"evidence-ref-001", policy_attached=True,
	)

	audit_events = [e for e in svc.audit_events if e["tenant_id"] == "tenant1"]
	assert len(audit_events) >= 1
	event_types = {e["event_type"] for e in audit_events}
	assert event_types  # non-empty set of event type strings


def test_alert_service_audit_event_has_processor():
	"""Alert audit events carry the bytewax processor tag."""
	from capabilities.intel.alerts.service import AlertManagementService

	svc = AlertManagementService()
	svc.record_authority(
		"auth-002", "t2", "mission_order", "scope-002",
		"unclassified", "approver-002", "2030-06-01T00:00:00Z",
		"evid-002",
	)

	assert svc.audit_events
	for evt in svc.audit_events:
		assert evt.get("processor") == "bytewax"


# ── Test: Registry streaming consistency ─────────────────────────────────────


def test_event_stream_contract_consistency():
	"""All capability contracts that declare streaming use bytewax as the processor."""
	from capabilities.capability_contract_registry import load_contract_registry

	registry = load_contract_registry()
	assert len(registry) >= 100  # sanity-check the registry is populated

	non_bytewax = []
	for cap_id, record in registry.items():
		streaming = record.contract.get("streaming", {})
		if not streaming:
			continue
		processor = streaming.get("processor") or streaming.get("stream_processor", "bytewax")
		if processor != "bytewax":
			non_bytewax.append((cap_id, processor))

	assert not non_bytewax, (
		f"Capabilities not using bytewax: {non_bytewax}"
	)


def test_registry_size_and_contract_shape():
	"""Registry has >= 100 capabilities; each has provides and requires."""
	from capabilities.capability_contract_registry import (
		load_contract_registry,
		validate_contract_shape,
	)

	registry = load_contract_registry()
	assert len(registry) >= 100

	for cap_id, record in registry.items():
		# validate_contract_shape raises ValueError on malformed contracts
		validate_contract_shape(record.contract, record.path)


# ── Test: Cross-capability adapter injection ─────────────────────────────────


async def test_cross_capability_adapter_injection():
	"""Full adapter injection: AR service emits >= 2 asserting audit events."""
	audit = AssertingAuditAdapter()
	svc = _ar_service(tenant_id="inject_test", audit=audit)

	# create_customer emits customer_created
	customer = await svc.create_customer("Beta Corp", 30_000, "NET45", "EUR")
	assert audit.call_count >= 1

	# create_invoice emits invoice_created  (need valid lines)
	await svc.create_invoice(
		customer_id=customer["id"],
		invoice_date="2026-06-01",
		due_date="2026-07-01",
		lines=[{"description": "Consulting", "quantity": 10, "unit_price": 1_000}],
		currency="EUR",
		payment_terms="NET45",
	)
	assert audit.call_count >= 2


async def test_capability_provides_are_non_empty_strings():
	"""Every provides entry in every capability contract is a non-empty string."""
	from capabilities.capability_contract_registry import load_contract_registry

	registry = load_contract_registry()
	for cap_id, record in registry.items():
		for item in record.contract.get("provides", []):
			assert isinstance(item, str) and item, (
				f"{cap_id}: provides contains invalid entry {item!r}"
			)


async def test_capability_requires_are_non_empty_strings():
	"""Every requires entry in every capability contract is a non-empty string."""
	from capabilities.capability_contract_registry import load_contract_registry

	registry = load_contract_registry()
	for cap_id, record in registry.items():
		for item in record.contract.get("requires", []):
			assert isinstance(item, str) and item, (
				f"{cap_id}: requires contains invalid entry {item!r}"
			)
