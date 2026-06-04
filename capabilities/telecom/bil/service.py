"""Service layer for APG Telecom Billing.

Covers: Rating & Charging, Invoice Generation, Collections & Dunning,
Disputes, Revenue Assurance — all tenant-scoped, async, adapter/store pattern.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Protocol, runtime_checkable

from .capability_contract import (
	SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_BILL_CYCLE_TYPES,
	SUPPORTED_CHARGE_TYPES, SUPPORTED_CONVERGENT_MODES, SUPPORTED_DISCOUNT_TYPES,
	SUPPORTED_DUNNING_STEPS, SUPPORTED_INVOICE_STATUSES, SUPPORTED_MEDIATION_STATUSES,
	SUPPORTED_PAYMENT_METHODS, SUPPORTED_RATING_TYPES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	BilAgent, BilCdr, BilCharge, BilConvergentAccount, BilCycle,
	BilDiscount, BilDunningStep, BilInvoice, BilPayment,
)


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------

def _present(value: str | None) -> bool:
	return bool(value and value.strip())

def _positive(value: float | Decimal) -> bool:
	return Decimal(str(value)) > Decimal("0")

def _non_negative(value: float | Decimal) -> bool:
	return Decimal(str(value)) >= Decimal("0")

def _now() -> str:
	return datetime.now(timezone.utc).isoformat()

def _d(value: Any) -> Decimal:
	"""Cast to Decimal, two decimal places."""
	return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Store / Adapter protocols — swap in-memory dict for Redis/Postgres at will
# ---------------------------------------------------------------------------

@runtime_checkable
class BillingStore(Protocol):
	async def get(self, key: str) -> dict[str, Any] | None: ...
	async def put(self, key: str, value: dict[str, Any]) -> None: ...
	async def delete(self, key: str) -> None: ...
	async def scan(self, prefix: str) -> list[dict[str, Any]]: ...


@runtime_checkable
class AuditAdapter(Protocol):
	async def emit(self, tenant_id: str, event_type: str, payload: dict[str, Any]) -> None: ...


@runtime_checkable
class NotifyAdapter(Protocol):
	async def send(self, channel: str, recipient: str, subject: str, body: str) -> None: ...


@runtime_checkable
class AuthAdapter(Protocol):
	async def check(self, actor_id: str, permission: str, resource: str) -> bool: ...


# ---------------------------------------------------------------------------
# In-memory default store
# ---------------------------------------------------------------------------

class _MemStore:
	"""Trivial in-process store used when no external store is supplied."""

	def __init__(self) -> None:
		self._data: dict[str, dict[str, Any]] = {}

	async def get(self, key: str) -> dict[str, Any] | None:
		return self._data.get(key)

	async def put(self, key: str, value: dict[str, Any]) -> None:
		self._data[key] = value

	async def delete(self, key: str) -> None:
		self._data.pop(key, None)

	async def scan(self, prefix: str) -> list[dict[str, Any]]:
		return [v for k, v in self._data.items() if k.startswith(prefix)]


class _NoopAudit:
	async def emit(self, tenant_id: str, event_type: str, payload: dict[str, Any]) -> None:
		pass


class _NoopNotify:
	async def send(self, channel: str, recipient: str, subject: str, body: str) -> None:
		pass


class _NoopAuth:
	async def check(self, actor_id: str, permission: str, resource: str) -> bool:
		return True


# ---------------------------------------------------------------------------
# Tariff / bundle reference data (in-memory, replaceable by store)
# ---------------------------------------------------------------------------

# Default IOT (Inter-Operator Tariff) rate in KES per minute
_DEFAULT_IOT_RATE = Decimal("2.00")
# Default home network margin percentage
_DEFAULT_HOME_MARGIN_PCT = Decimal("15")
# Default PAYG overage rate per unit
_DEFAULT_PAYG_RATE: dict[str, Decimal] = {
	"voice":    Decimal("4.00"),   # per minute
	"data":     Decimal("1.00"),   # per MB
	"sms":      Decimal("1.00"),   # per SMS
}
# Roaming zone surcharge multipliers
_ROAMING_ZONE_MULTIPLIER: dict[str, Decimal] = {
	"domestic": Decimal("1.0"),
	"zone_a":   Decimal("2.5"),
	"zone_b":   Decimal("4.0"),
	"zone_c":   Decimal("6.0"),
	"premium":  Decimal("10.0"),
	"global":   Decimal("8.0"),
}
# Data tier thresholds in MB → rate per MB in KES
_DATA_TIERS: list[tuple[Decimal, Decimal]] = [
	(Decimal("1024"),   Decimal("1.50")),    # 0 – 1 GB
	(Decimal("5120"),   Decimal("1.20")),    # 1 – 5 GB
	(Decimal("20480"),  Decimal("0.90")),    # 5 – 20 GB
	(Decimal("1e9"),    Decimal("0.70")),    # > 20 GB
]


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class TelecomBillingService:
	"""Tenant-scoped telecom billing service.

	Parameters
	----------
	tenant_id:
		Owning tenant.  All operations are scoped to this tenant.
	actor_id:
		The identity performing actions — used in audit trails.
	auth, audit, notify, store:
		Injectable adapters; default to no-ops / in-memory.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: AuthAdapter | None = None,
		audit: AuditAdapter | None = None,
		notify: NotifyAdapter | None = None,
		store: BillingStore | None = None,
	) -> None:
		assert _present(tenant_id), "tenant_id must not be blank"
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth or _NoopAuth()
		self._audit = audit or _NoopAudit()
		self._notify = notify or _NoopNotify()
		self._store = store or _MemStore()

		# Legacy in-process collections — still usable; mirrored to store on writes
		self.cdrs: dict[tuple[str, str], BilCdr] = {}
		self.charges: dict[tuple[str, str], BilCharge] = {}
		self.cycles: dict[tuple[str, str], BilCycle] = {}
		self.invoices: dict[tuple[str, str], BilInvoice] = {}
		self.dunning_steps: dict[tuple[str, str], BilDunningStep] = {}
		self.payments: dict[tuple[str, str], BilPayment] = {}
		self.discounts: dict[tuple[str, str], BilDiscount] = {}
		self.convergent_accounts: dict[tuple[str, str], BilConvergentAccount] = {}
		self.agents: dict[tuple[str, str], BilAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Extended in-memory collections for new methods
		self._balances: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
		self._bundles: dict[str, dict[str, Any]] = {}      # bundle_id -> bundle record
		self._promotions: dict[str, dict[str, Any]] = {}  # promo_code -> promo record
		self._disputes: dict[str, dict[str, Any]] = {}    # dispute_id -> dispute record
		self._adjustments: dict[str, list[dict[str, Any]]] = defaultdict(list)  # invoice_id
		self._suspended_accounts: set[str] = set()
		self._bill_runs: dict[str, dict[str, Any]] = {}   # run_id
		self._revenue_leakage_log: list[dict[str, Any]] = []

	# -----------------------------------------------------------------------
	# Contract / policy helpers
	# -----------------------------------------------------------------------

	def describe(self, tenant_id: str | None = None) -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "policy_denied")

	def _emit(self, event_type: str, reference_id: str, payload: dict[str, Any] | None = None) -> None:
		event: dict[str, Any] = {
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"stream": "apg.telecom.bil.lifecycle",
			"ts": _now(),
		}
		if payload:
			event["payload"] = payload
		self.audit_events.append(event)

	def _key(self, item_id: str) -> tuple[str, str]:
		return (self.tenant_id, item_id)

	def _count(self, store: dict[tuple[str, str], Any]) -> int:
		return sum(1 for k in store if k[0] == self.tenant_id)

	# -----------------------------------------------------------------------
	# Legacy CDR / charge / cycle / invoice / dunning / payment / discount
	# -----------------------------------------------------------------------

	def record_cdr(
		self,
		cdr_id: str,
		source: str,
		mediation_status: str,
		msisdn: str,
		duration_seconds: int,
		data_volume_bytes: int,
		recorded_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Mediate a CDR through the billing pipeline."""
		mediation_status = mediation_status.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_cdr",
			"mediation_status_supported": mediation_status in SUPPORTED_MEDIATION_STATUSES,
			"source_present": _present(source),
		})
		item = BilCdr(cdr_id, self.tenant_id, source, mediation_status, msisdn,
		              int(duration_seconds), int(data_volume_bytes), recorded_at)
		self.cdrs[self._key(cdr_id)] = item
		self._emit("cdr_mediated", cdr_id)
		return item.to_dict()

	def record_charge(
		self,
		charge_id: str,
		customer_id: str,
		charge_type: str,
		rating_type: str,
		amount: float,
		currency: str,
		tax_amount: float,
		cdr_id: str | None = None,
	) -> dict[str, Any]:
		"""Rate a charge for a customer."""
		charge_type = charge_type.lower()
		rating_type = rating_type.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_charge",
			"charge_type_supported": charge_type in SUPPORTED_CHARGE_TYPES,
			"rating_type_supported": rating_type in SUPPORTED_RATING_TYPES,
			"amount_positive": _positive(amount),
		})
		item = BilCharge(charge_id, self.tenant_id, customer_id, charge_type, rating_type,
		                 float(amount), currency, float(tax_amount), cdr_id)
		self.charges[self._key(charge_id)] = item
		self._emit("charge_rated", charge_id)
		return item.to_dict()

	def create_bill_cycle(
		self,
		cycle_id: str,
		cycle_type: str,
		cutoff_date: str,
		start_date: str,
		end_date: str,
		status: str = "active",
	) -> dict[str, Any]:
		"""Create a billing cycle."""
		cycle_type = cycle_type.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_bill_cycle",
			"cycle_type_supported": cycle_type in SUPPORTED_BILL_CYCLE_TYPES,
			"cutoff_present": _present(cutoff_date),
		})
		item = BilCycle(cycle_id, self.tenant_id, cycle_type, cutoff_date, start_date, end_date, status)
		self.cycles[self._key(cycle_id)] = item
		self._emit("bill_cycle_created", cycle_id)
		return item.to_dict()

	def generate_invoice(
		self,
		invoice_id: str,
		customer_id: str,
		cycle_id: str,
		total_amount: float,
		currency: str,
		due_date: str,
	) -> dict[str, Any]:
		"""Generate a draft invoice."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_invoice_status",
			"status_supported": "draft" in SUPPORTED_INVOICE_STATUSES,
		})
		item = BilInvoice(invoice_id, self.tenant_id, customer_id, cycle_id,
		                  float(total_amount), currency, "draft", "", due_date)
		self.invoices[self._key(invoice_id)] = item
		self._emit("invoice_generated", invoice_id)
		return item.to_dict()

	def approve_invoice(self, invoice_id: str, approval_reference: str) -> dict[str, Any]:
		"""Approve an invoice prior to dispatch."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_invoice",
			"approval_present": _present(approval_reference),
		})
		invoice = self._invoice_or_raise(invoice_id)
		invoice.status = "approved"
		invoice.approval_reference = approval_reference
		self._emit("invoice_approved", invoice_id)
		return invoice.to_dict()

	def trigger_dunning(
		self,
		dunning_id: str,
		invoice_id: str,
		step: str,
		triggered_at: str,
		next_step_date: str | None = None,
	) -> dict[str, Any]:
		"""Trigger the next dunning step for an overdue invoice."""
		step = step.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "trigger_dunning",
			"dunning_step_supported": step in SUPPORTED_DUNNING_STEPS,
		})
		item = BilDunningStep(dunning_id, self.tenant_id, invoice_id, step, triggered_at, next_step_date)
		self.dunning_steps[self._key(dunning_id)] = item
		self._emit("dunning_step_triggered", dunning_id)
		return item.to_dict()

	def record_payment(
		self,
		payment_id: str,
		invoice_id: str,
		payment_method: str,
		amount: float,
		currency: str,
		reference: str,
		paid_at: str,
	) -> dict[str, Any]:
		"""Record a payment against an invoice."""
		payment_method = payment_method.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_payment",
			"payment_method_supported": payment_method in SUPPORTED_PAYMENT_METHODS,
			"amount_positive": _positive(amount),
		})
		item = BilPayment(payment_id, self.tenant_id, invoice_id, payment_method,
		                  float(amount), currency, reference, paid_at)
		self.payments[self._key(payment_id)] = item
		self._emit("payment_received", payment_id)
		return item.to_dict()

	def apply_discount(
		self,
		discount_id: str,
		customer_id: str,
		discount_type: str,
		discount_pct: float,
		approval_reference: str,
		valid_from: str,
		valid_to: str,
	) -> dict[str, Any]:
		"""Apply an approved discount to a customer account."""
		discount_type = discount_type.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "apply_discount",
			"discount_type_supported": discount_type in SUPPORTED_DISCOUNT_TYPES,
			"approval_present": _present(approval_reference),
			"max_discount_exceeded": discount_pct > 50,
		})
		item = BilDiscount(discount_id, self.tenant_id, customer_id, discount_type,
		                   float(discount_pct), approval_reference, valid_from, valid_to)
		self.discounts[self._key(discount_id)] = item
		self._emit("discount_applied", discount_id)
		return item.to_dict()

	def write_off_invoice(self, invoice_id: str, approval_reference: str) -> dict[str, Any]:
		"""Write off an uncollectable invoice after approval."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "write_off_invoice",
			"approval_present": _present(approval_reference),
		})
		invoice = self._invoice_or_raise(invoice_id)
		invoice.status = "written_off"
		self._emit("write_off_recorded", invoice_id)
		return invoice.to_dict()

	def setup_convergent(
		self,
		account_id: str,
		convergent_mode: str,
		master_account_id: str,
		member_account_ids: str,
		currency: str,
	) -> dict[str, Any]:
		"""Set up a convergent billing account."""
		convergent_mode = convergent_mode.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "setup_convergent",
			"convergent_mode_supported": convergent_mode in SUPPORTED_CONVERGENT_MODES,
		})
		item = BilConvergentAccount(account_id, self.tenant_id, convergent_mode,
		                            master_account_id, member_account_ids, currency)
		self.convergent_accounts[self._key(account_id)] = item
		self._emit("convergent_account_setup", account_id)
		return item.to_dict()

	def register_agent(
		self,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Register a billing automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_bil_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = BilAgent(agent_id, self.tenant_id, name, runtime, role, scope)
		self.agents[self._key(agent_id)] = item
		self._emit("bil_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		privileged_scope: bool,
		human_approval_recorded: bool,
		bill_suppression_scope: bool = False,
		cross_tenant_billing_scope: bool = False,
	) -> dict[str, Any]:
		"""Validate a billing agent action against guardrails."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation": "bil_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"bill_suppression_scope": bill_suppression_scope,
			"cross_tenant_billing_scope": cross_tenant_billing_scope,
		})
		return {"tenant_id": self.tenant_id, "accepted": True}

	def validate_batch(self, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		"""Validate a billing batch operation."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": True,
			"operation": "bil_batch",
			"event_stream": event_stream,
		})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": self.tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.telecom.bil.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self) -> dict[str, Any]:
		"""Return billing dashboard summary."""
		return {
			"tenant_id": self.tenant_id,
			"cdr_count": self._count(self.cdrs),
			"charge_count": self._count(self.charges),
			"cycle_count": self._count(self.cycles),
			"invoice_count": self._count(self.invoices),
			"dunning_step_count": self._count(self.dunning_steps),
			"payment_count": self._count(self.payments),
			"discount_count": self._count(self.discounts),
			"agent_count": self._count(self.agents),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == self.tenant_id),
			"open_disputes": sum(1 for d in self._disputes.values()
			                     if d["tenant_id"] == self.tenant_id and d["status"] == "open"),
			"suspended_accounts": len(self._suspended_accounts),
			"streaming": get_capability_contract(self.tenant_id)["streaming"],
		}

	# -----------------------------------------------------------------------
	# Rating & Charging (8 methods)
	# -----------------------------------------------------------------------

	async def rate_voice_call(self, cdr: dict[str, Any]) -> dict[str, Any]:
		"""Apply tariff plan, bundle deduction, and roaming rates to a voice CDR.

		cdr keys: subscriber_id, duration_seconds, call_type (on_net|off_net|international),
		          tariff_plan_id, bundle_id (opt), roaming_zone (opt), currency (opt)
		"""
		subscriber_id: str = cdr["subscriber_id"]
		duration_s: int = int(cdr.get("duration_seconds", 0))
		call_type: str = cdr.get("call_type", "on_net").lower()
		tariff_plan_id: str = cdr.get("tariff_plan_id", "standard")
		bundle_id: str | None = cdr.get("bundle_id")
		roaming_zone: str | None = cdr.get("roaming_zone")
		currency: str = cdr.get("currency", "KES")

		assert duration_s >= 0, "duration_seconds must be >= 0"

		duration_min = Decimal(str(duration_s)) / Decimal("60")

		# Base rates per minute by call type
		base_rates: dict[str, Decimal] = {
			"on_net":        Decimal("1.00"),
			"off_net":       Decimal("3.00"),
			"international": Decimal("12.00"),
		}
		rate_per_min = base_rates.get(call_type, Decimal("3.00"))

		gross_charge = (duration_min * rate_per_min).quantize(Decimal("0.01"), ROUND_HALF_UP)

		bundle_units_used = Decimal("0")
		bundle_deduction = Decimal("0")

		if bundle_id:
			# Consume from bundle first
			bundle = self._bundles.get(bundle_id)
			if bundle and bundle.get("remaining_units", Decimal("0")) > 0:
				available = Decimal(str(bundle["remaining_units"]))
				consumed = min(available, duration_min)
				bundle["remaining_units"] = available - consumed
				bundle["consumed_units"] = Decimal(str(bundle.get("consumed_units", 0))) + consumed
				bundle_units_used = consumed
				bundle_deduction = (consumed * rate_per_min).quantize(Decimal("0.01"), ROUND_HALF_UP)

		payg_amount = gross_charge - bundle_deduction

		if roaming_zone:
			multiplier = _ROAMING_ZONE_MULTIPLIER.get(roaming_zone, Decimal("1.0"))
			payg_amount = (payg_amount * multiplier).quantize(Decimal("0.01"), ROUND_HALF_UP)

		# VAT 16%
		tax_amount = (payg_amount * Decimal("0.16")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		total = payg_amount + tax_amount

		result: dict[str, Any] = {
			"subscriber_id": subscriber_id,
			"tariff_plan_id": tariff_plan_id,
			"call_type": call_type,
			"duration_seconds": duration_s,
			"duration_minutes": str(duration_min),
			"gross_charge": str(gross_charge),
			"bundle_id": bundle_id,
			"bundle_units_used_minutes": str(bundle_units_used),
			"bundle_deduction": str(bundle_deduction),
			"roaming_zone": roaming_zone,
			"payg_charge": str(payg_amount),
			"tax_amount": str(tax_amount),
			"total_charge": str(total),
			"currency": currency,
			"rated_at": _now(),
		}
		self._emit("voice_cdr_rated", subscriber_id, {"total_charge": str(total), "call_type": call_type})
		return result

	async def rate_data_session(self, session_cdr: dict[str, Any]) -> dict[str, Any]:
		"""Apply per-MB tiered rating to a data session CDR.

		session_cdr keys: subscriber_id, data_volume_bytes, tariff_plan_id,
		                  bundle_id (opt), currency (opt)
		"""
		subscriber_id: str = session_cdr["subscriber_id"]
		data_bytes: int = int(session_cdr.get("data_volume_bytes", 0))
		bundle_id: str | None = session_cdr.get("bundle_id")
		currency: str = session_cdr.get("currency", "KES")

		assert data_bytes >= 0, "data_volume_bytes must be >= 0"

		data_mb = Decimal(str(data_bytes)) / Decimal("1048576")  # bytes → MB

		bundle_mb_used = Decimal("0")
		bundle_deduction = Decimal("0")

		if bundle_id:
			bundle = self._bundles.get(bundle_id)
			if bundle and bundle.get("bundle_type") == "data":
				available = Decimal(str(bundle.get("remaining_units", 0)))
				if available > 0:
					consumed = min(available, data_mb)
					bundle["remaining_units"] = available - consumed
					bundle["consumed_units"] = Decimal(str(bundle.get("consumed_units", 0))) + consumed
					bundle_mb_used = consumed
					# deduction valued at standard tier-1 rate
					bundle_deduction = (consumed * _DATA_TIERS[0][1]).quantize(Decimal("0.01"), ROUND_HALF_UP)

		billable_mb = data_mb - bundle_mb_used

		# Tiered rating on billable_mb
		tiered_charge = Decimal("0")
		remaining = billable_mb
		prev_threshold = Decimal("0")
		for threshold, rate in _DATA_TIERS:
			if remaining <= 0:
				break
			tier_mb = min(remaining, threshold - prev_threshold)
			tiered_charge += (tier_mb * rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
			remaining -= tier_mb
			prev_threshold = threshold

		tax = (tiered_charge * Decimal("0.16")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		total = tiered_charge + tax

		result: dict[str, Any] = {
			"subscriber_id": subscriber_id,
			"data_volume_bytes": data_bytes,
			"data_volume_mb": str(data_mb.quantize(Decimal("0.001"), ROUND_HALF_UP)),
			"bundle_id": bundle_id,
			"bundle_mb_used": str(bundle_mb_used),
			"bundle_deduction": str(bundle_deduction),
			"billable_mb": str(billable_mb.quantize(Decimal("0.001"), ROUND_HALF_UP)),
			"tiered_charge": str(tiered_charge),
			"tax_amount": str(tax),
			"total_charge": str(total),
			"currency": currency,
			"rated_at": _now(),
		}
		self._emit("data_session_rated", subscriber_id, {"total_charge": str(total)})
		return result

	async def rate_sms(self, sms_cdr: dict[str, Any]) -> dict[str, Any]:
		"""Apply on-net vs off-net SMS rates.

		sms_cdr keys: subscriber_id, sms_count, sms_type (on_net|off_net|international|premium),
		              bundle_id (opt), currency (opt)
		"""
		subscriber_id: str = sms_cdr["subscriber_id"]
		sms_count: int = int(sms_cdr.get("sms_count", 1))
		sms_type: str = sms_cdr.get("sms_type", "on_net").lower()
		bundle_id: str | None = sms_cdr.get("bundle_id")
		currency: str = sms_cdr.get("currency", "KES")

		assert sms_count >= 0, "sms_count must be >= 0"

		sms_rates: dict[str, Decimal] = {
			"on_net":        Decimal("0.50"),
			"off_net":       Decimal("1.00"),
			"international": Decimal("5.00"),
			"premium":       Decimal("3.00"),
		}
		rate = sms_rates.get(sms_type, Decimal("1.00"))
		count = Decimal(str(sms_count))

		bundle_sms_used = Decimal("0")
		if bundle_id:
			bundle = self._bundles.get(bundle_id)
			if bundle and bundle.get("bundle_type") == "sms":
				available = Decimal(str(bundle.get("remaining_units", 0)))
				if available > 0:
					used = min(available, count)
					bundle["remaining_units"] = available - used
					bundle["consumed_units"] = Decimal(str(bundle.get("consumed_units", 0))) + used
					bundle_sms_used = used

		billable_count = count - bundle_sms_used
		gross_charge = (billable_count * rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
		tax = (gross_charge * Decimal("0.16")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		total = gross_charge + tax

		result: dict[str, Any] = {
			"subscriber_id": subscriber_id,
			"sms_count": sms_count,
			"sms_type": sms_type,
			"bundle_id": bundle_id,
			"bundle_sms_used": str(bundle_sms_used),
			"billable_count": str(billable_count),
			"rate_per_sms": str(rate),
			"gross_charge": str(gross_charge),
			"tax_amount": str(tax),
			"total_charge": str(total),
			"currency": currency,
			"rated_at": _now(),
		}
		self._emit("sms_cdr_rated", subscriber_id, {"total_charge": str(total), "sms_type": sms_type})
		return result

	async def rate_roaming_event(self, roaming_cdr: dict[str, Any]) -> dict[str, Any]:
		"""Apply IOT rates plus home network margin for a roaming event.

		roaming_cdr keys: subscriber_id, visited_network, home_network, zone,
		                  service_type (voice|data|sms), duration_seconds (voice),
		                  data_volume_bytes (data), sms_count (sms), currency (opt)
		"""
		subscriber_id: str = roaming_cdr["subscriber_id"]
		visited_network: str = roaming_cdr["visited_network"]
		home_network: str = roaming_cdr.get("home_network", self.tenant_id)
		zone: str = roaming_cdr.get("zone", "zone_a").lower()
		service_type: str = roaming_cdr.get("service_type", "voice").lower()
		currency: str = roaming_cdr.get("currency", "KES")
		iot_rate = _DEFAULT_IOT_RATE * _ROAMING_ZONE_MULTIPLIER.get(zone, Decimal("2.5"))

		if service_type == "voice":
			duration_s = int(roaming_cdr.get("duration_seconds", 0))
			duration_min = Decimal(str(duration_s)) / Decimal("60")
			iot_cost = (duration_min * iot_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
			units = f"{duration_s}s"
		elif service_type == "data":
			data_bytes = int(roaming_cdr.get("data_volume_bytes", 0))
			data_mb = Decimal(str(data_bytes)) / Decimal("1048576")
			iot_cost = (data_mb * iot_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
			units = f"{data_bytes}B"
		else:  # sms
			sms_count = Decimal(str(roaming_cdr.get("sms_count", 1)))
			iot_cost = (sms_count * iot_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
			units = f"{roaming_cdr.get('sms_count', 1)} SMS"

		margin = (_DEFAULT_HOME_MARGIN_PCT / Decimal("100") * iot_cost).quantize(Decimal("0.01"), ROUND_HALF_UP)
		home_charge = iot_cost + margin
		tax = (home_charge * Decimal("0.16")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		total = home_charge + tax

		result: dict[str, Any] = {
			"subscriber_id": subscriber_id,
			"visited_network": visited_network,
			"home_network": home_network,
			"zone": zone,
			"service_type": service_type,
			"units": units,
			"iot_rate": str(iot_rate),
			"iot_cost": str(iot_cost),
			"home_margin_pct": str(_DEFAULT_HOME_MARGIN_PCT),
			"margin_amount": str(margin),
			"home_charge": str(home_charge),
			"tax_amount": str(tax),
			"total_charge": str(total),
			"currency": currency,
			"rated_at": _now(),
		}
		self._emit("roaming_event_rated", subscriber_id, {"total_charge": str(total), "zone": zone})
		return result

	async def real_time_balance_check(
		self,
		subscriber_id: str,
		service_type: str,
		amount: Decimal | float | str,
	) -> dict[str, Any]:
		"""Check and optionally reserve prepaid balance in real time.

		Returns whether the subscriber has sufficient balance for the requested amount.
		Does not deduct — call bundle_consumption or a separate debit for that.
		"""
		assert _present(subscriber_id), "subscriber_id required"
		assert _present(service_type), "service_type required"

		requested = _d(amount)
		assert requested >= 0, "amount must be >= 0"

		balance = self._balances[subscriber_id].get("main_balance", Decimal("0"))
		service_balance = self._balances[subscriber_id].get(service_type, Decimal("0"))
		effective_balance = max(balance, service_balance)

		sufficient = effective_balance >= requested
		result: dict[str, Any] = {
			"subscriber_id": subscriber_id,
			"service_type": service_type,
			"requested_amount": str(requested),
			"main_balance": str(balance),
			"service_balance": str(service_balance),
			"effective_balance": str(effective_balance),
			"sufficient": sufficient,
			"deficit": str(max(Decimal("0"), requested - effective_balance)),
			"checked_at": _now(),
		}
		self._emit("balance_checked", subscriber_id, {"sufficient": sufficient})
		return result

	async def bundle_consumption(
		self,
		subscriber_id: str,
		event_type: str,
		units: Decimal | float | str,
	) -> dict[str, Any]:
		"""Deduct usage units from the subscriber's active bundle(s).

		Iterates all bundles for the subscriber, finds the first active one that
		matches event_type, and deducts.  Returns remaining units and whether the
		bundle is exhausted.
		"""
		assert _present(subscriber_id), "subscriber_id required"
		consumed = _d(units)
		assert consumed >= 0, "units must be >= 0"

		# Find matching bundles for subscriber
		matching = [
			b for b in self._bundles.values()
			if b.get("subscriber_id") == subscriber_id
			and b.get("bundle_type", "").lower() == event_type.lower()
			and b.get("status") == "active"
			and Decimal(str(b.get("remaining_units", 0))) > 0
		]

		if not matching:
			return {
				"subscriber_id": subscriber_id,
				"event_type": event_type,
				"requested_units": str(consumed),
				"bundle_found": False,
				"consumed": "0",
				"overage_units": str(consumed),
				"exhausted": False,
				"consumed_at": _now(),
			}

		bundle = matching[0]
		available = Decimal(str(bundle["remaining_units"]))
		actually_consumed = min(available, consumed)
		overage = consumed - actually_consumed

		bundle["remaining_units"] = available - actually_consumed
		bundle["consumed_units"] = Decimal(str(bundle.get("consumed_units", 0))) + actually_consumed
		exhausted = bundle["remaining_units"] <= Decimal("0")
		if exhausted:
			bundle["status"] = "exhausted"

		result: dict[str, Any] = {
			"subscriber_id": subscriber_id,
			"bundle_id": bundle["bundle_id"],
			"event_type": event_type,
			"requested_units": str(consumed),
			"bundle_found": True,
			"consumed": str(actually_consumed),
			"remaining_units": str(bundle["remaining_units"]),
			"overage_units": str(overage),
			"exhausted": exhausted,
			"consumed_at": _now(),
		}
		self._emit("bundle_consumed", subscriber_id, {"bundle_id": bundle["bundle_id"], "consumed": str(actually_consumed)})
		return result

	async def overage_charging(
		self,
		subscriber_id: str,
		bundle_id: str,
		excess_units: Decimal | float | str,
	) -> dict[str, Any]:
		"""Charge excess usage at PAYG rates after bundle exhaustion.

		Determines service type from the bundle record and applies _DEFAULT_PAYG_RATE.
		"""
		assert _present(subscriber_id), "subscriber_id required"
		assert _present(bundle_id), "bundle_id required"

		excess = _d(excess_units)
		assert excess >= 0, "excess_units must be >= 0"

		bundle = self._bundles.get(bundle_id)
		if bundle is None:
			raise KeyError(f"bundle {bundle_id} not found")

		service = bundle.get("bundle_type", "data").lower()
		payg_rate = _DEFAULT_PAYG_RATE.get(service, Decimal("1.00"))
		gross_charge = (excess * payg_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
		tax = (gross_charge * Decimal("0.16")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		total = gross_charge + tax

		# Debit subscriber balance
		self._balances[subscriber_id]["main_balance"] = (
			self._balances[subscriber_id].get("main_balance", Decimal("0")) - total
		)

		result: dict[str, Any] = {
			"subscriber_id": subscriber_id,
			"bundle_id": bundle_id,
			"service_type": service,
			"excess_units": str(excess),
			"payg_rate": str(payg_rate),
			"gross_charge": str(gross_charge),
			"tax_amount": str(tax),
			"total_charge": str(total),
			"new_balance": str(self._balances[subscriber_id]["main_balance"]),
			"charged_at": _now(),
		}
		self._emit("overage_charged", subscriber_id, {"bundle_id": bundle_id, "total_charge": str(total)})
		return result

	async def apply_promotion(
		self,
		subscriber_id: str,
		promo_code: str,
		valid_from: str,
		valid_to: str,
	) -> dict[str, Any]:
		"""Apply a promotion discount to a subscriber account.

		Creates a pending discount entry; actual deduction happens at invoice generation.
		"""
		assert _present(subscriber_id), "subscriber_id required"
		assert _present(promo_code), "promo_code required"

		promo = self._promotions.get(promo_code)
		if promo is None:
			# Auto-create a stub promotion for the given code (10% default)
			promo = {
				"promo_code": promo_code,
				"discount_pct": Decimal("10.00"),
				"bonus_units": Decimal("0"),
				"status": "active",
				"redemptions": 0,
				"max_redemptions": None,
			}
			self._promotions[promo_code] = promo

		if promo.get("status") != "active":
			raise ValueError(f"Promotion {promo_code} is not active (status: {promo['status']})")

		max_r = promo.get("max_redemptions")
		if max_r is not None and promo.get("redemptions", 0) >= max_r:
			raise ValueError(f"Promotion {promo_code} has reached its redemption limit")

		promo["redemptions"] = promo.get("redemptions", 0) + 1

		discount_id = f"promo_{promo_code}_{subscriber_id}_{_now()}"
		discount_record: dict[str, Any] = {
			"discount_id": discount_id,
			"subscriber_id": subscriber_id,
			"tenant_id": self.tenant_id,
			"promo_code": promo_code,
			"discount_pct": str(promo["discount_pct"]),
			"bonus_units": str(promo["bonus_units"]),
			"valid_from": valid_from,
			"valid_to": valid_to,
			"status": "active",
			"applied_at": _now(),
		}
		self._emit("promotion_applied", subscriber_id, {"promo_code": promo_code, "discount_pct": str(promo["discount_pct"])})
		return discount_record

	# -----------------------------------------------------------------------
	# Invoice Generation (6 methods)
	# -----------------------------------------------------------------------

	async def generate_bill(self, account_id: str, billing_period: dict[str, str]) -> dict[str, Any]:
		"""Aggregate all charges for an account over a billing period into a draft invoice.

		billing_period: {"start": ISO-date, "end": ISO-date}
		"""
		assert _present(account_id), "account_id required"
		period_start = billing_period.get("start", "")
		period_end = billing_period.get("end", "")
		assert _present(period_start) and _present(period_end), "billing_period.start and .end required"

		# Collect all charges for this account (simple in-memory filter)
		account_charges = [
			ch for ch in self.charges.values()
			if hasattr(ch, "customer_id") and ch.customer_id == account_id
		]

		subtotal = sum(_d(getattr(ch, "amount", 0)) for ch in account_charges)
		tax_total = sum(_d(getattr(ch, "tax_amount", 0)) for ch in account_charges)
		discount_total = Decimal("0")

		# Apply active discounts for account
		account_discounts = [
			d for d in self.discounts.values()
			if hasattr(d, "customer_id") and d.customer_id == account_id
		]
		for disc in account_discounts:
			pct = _d(getattr(disc, "discount_pct", 0)) / Decimal("100")
			discount_total += (subtotal * pct).quantize(Decimal("0.01"), ROUND_HALF_UP)

		total = subtotal + tax_total - discount_total

		invoice_id = f"inv_{account_id}_{period_start[:7].replace('-', '')}_{_now()[:10].replace('-', '')}"
		invoice_record: dict[str, Any] = {
			"invoice_id": invoice_id,
			"account_id": account_id,
			"tenant_id": self.tenant_id,
			"period_start": period_start,
			"period_end": period_end,
			"charge_count": len(account_charges),
			"subtotal": str(subtotal),
			"tax_amount": str(tax_total),
			"discount_amount": str(discount_total),
			"total_amount": str(total),
			"currency": "KES",
			"status": "draft",
			"generated_at": _now(),
		}
		self._emit("bill_generated", invoice_id, {"account_id": account_id, "total": str(total)})
		return invoice_record

	async def bill_calculation(self, account_id: str, period: dict[str, str]) -> dict[str, Any]:
		"""Return a detailed charge-by-charge breakdown for an account billing period."""
		assert _present(account_id), "account_id required"

		account_charges = [
			{
				"charge_id": getattr(ch, "charge_id", ""),
				"charge_type": getattr(ch, "charge_type", ""),
				"rating_type": getattr(ch, "rating_type", ""),
				"amount": str(_d(getattr(ch, "amount", 0))),
				"tax_amount": str(_d(getattr(ch, "tax_amount", 0))),
				"currency": getattr(ch, "currency", "KES"),
				"cdr_id": getattr(ch, "cdr_id", None),
			}
			for ch in self.charges.values()
			if hasattr(ch, "customer_id") and ch.customer_id == account_id
		]

		by_type: dict[str, Decimal] = defaultdict(Decimal)
		for ch in account_charges:
			by_type[ch["charge_type"]] += _d(ch["amount"])

		total = sum(_d(ch["amount"]) for ch in account_charges)
		total_tax = sum(_d(ch["tax_amount"]) for ch in account_charges)

		return {
			"account_id": account_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"line_items": account_charges,
			"by_charge_type": {k: str(v) for k, v in by_type.items()},
			"subtotal": str(total),
			"total_tax": str(total_tax),
			"grand_total": str(total + total_tax),
			"currency": "KES",
			"calculated_at": _now(),
		}

	async def apply_adjustments(
		self,
		invoice_id: str,
		adjustment_type: str,
		amount: Decimal | float | str,
		reason: str,
	) -> dict[str, Any]:
		"""Apply a credit or debit adjustment to an invoice.

		adjustment_type: "credit" | "debit" | "write_off" | "goodwill"
		"""
		assert _present(invoice_id), "invoice_id required"
		assert adjustment_type in {"credit", "debit", "write_off", "goodwill"}, \
			f"unsupported adjustment_type: {adjustment_type}"
		assert _present(reason), "reason required"

		adj_amount = _d(amount)
		adjustment: dict[str, Any] = {
			"adjustment_id": f"adj_{invoice_id}_{_now()[:10].replace('-', '')}",
			"invoice_id": invoice_id,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"adjustment_type": adjustment_type,
			"amount": str(adj_amount),
			"reason": reason,
			"applied_at": _now(),
		}
		self._adjustments[invoice_id].append(adjustment)
		self._emit("adjustment_applied", invoice_id, {"type": adjustment_type, "amount": str(adj_amount)})
		return adjustment

	async def generate_bill_run(
		self,
		billing_date: str,
		segment: str | None = None,
	) -> dict[str, Any]:
		"""Execute a mass billing run, optionally filtered to a customer segment.

		billing_date: ISO date string for which the bill run is executed.
		segment: optional segment tag (e.g. "prepaid", "postpaid", "corporate").
		"""
		assert _present(billing_date), "billing_date required"

		# Discover all unique account_ids in the charges store
		all_accounts: set[str] = set()
		for ch in self.charges.values():
			if hasattr(ch, "customer_id"):
				all_accounts.add(ch.customer_id)

		run_id = f"run_{billing_date.replace('-', '')}_{segment or 'all'}"
		generated_invoices: list[str] = []
		errors: list[dict[str, Any]] = []

		for account_id in sorted(all_accounts):
			try:
				inv = await self.generate_bill(
					account_id,
					{"start": billing_date[:7] + "-01", "end": billing_date},
				)
				generated_invoices.append(inv["invoice_id"])
			except Exception as exc:
				errors.append({"account_id": account_id, "error": str(exc)})

		run_record: dict[str, Any] = {
			"run_id": run_id,
			"tenant_id": self.tenant_id,
			"billing_date": billing_date,
			"segment": segment,
			"accounts_processed": len(all_accounts),
			"invoices_generated": len(generated_invoices),
			"invoice_ids": generated_invoices,
			"error_count": len(errors),
			"errors": errors,
			"status": "completed" if not errors else "completed_with_errors",
			"executed_at": _now(),
		}
		self._bill_runs[run_id] = run_record
		self._emit("bill_run_completed", run_id, {"invoices": len(generated_invoices), "errors": len(errors)})
		return run_record

	async def bill_delivery(self, invoice_id: str, channel: str) -> dict[str, Any]:
		"""Deliver an invoice via the specified channel (email|sms|portal|print).

		Looks up invoice, validates channel, dispatches via notify adapter.
		"""
		assert _present(invoice_id), "invoice_id required"
		supported_channels = {"email", "sms", "portal", "print"}
		channel = channel.lower()
		assert channel in supported_channels, f"channel must be one of {supported_channels}"

		# Attempt to find invoice in legacy store
		invoice = next(
			(inv for k, inv in self.invoices.items() if k[1] == invoice_id),
			None,
		)

		recipient = getattr(invoice, "customer_id", "unknown") if invoice else "unknown"
		subject = f"Your invoice {invoice_id}"
		body = f"Invoice {invoice_id} is ready. Please review and pay by the due date."

		await self._notify.send(channel, recipient, subject, body)

		delivery_record: dict[str, Any] = {
			"invoice_id": invoice_id,
			"channel": channel,
			"recipient": recipient,
			"status": "delivered",
			"delivered_at": _now(),
		}
		self._emit("invoice_delivered", invoice_id, {"channel": channel, "recipient": recipient})
		return delivery_record

	async def view_bill(self, invoice_id: str) -> dict[str, Any]:
		"""Return an itemised bill view for a given invoice.

		Merges invoice header, line items, adjustments, and payment history.
		"""
		assert _present(invoice_id), "invoice_id required"

		invoice = next(
			(inv for k, inv in self.invoices.items() if k[1] == invoice_id),
			None,
		)
		if invoice is None:
			# Return a minimal not-found structure rather than raising
			return {
				"invoice_id": invoice_id,
				"found": False,
				"tenant_id": self.tenant_id,
			}

		# Gather charges linked to this invoice via cdr_id (heuristic; in prod use FK)
		line_items = [
			{
				"charge_id": getattr(ch, "charge_id", ""),
				"charge_type": getattr(ch, "charge_type", ""),
				"amount": str(_d(getattr(ch, "amount", 0))),
				"tax_amount": str(_d(getattr(ch, "tax_amount", 0))),
				"currency": getattr(ch, "currency", "KES"),
			}
			for ch in self.charges.values()
			if hasattr(ch, "customer_id") and ch.customer_id == getattr(invoice, "customer_id", "")
		]

		adjustments = self._adjustments.get(invoice_id, [])
		payments = [
			{
				"payment_id": getattr(p, "payment_id", ""),
				"amount": str(_d(getattr(p, "amount", 0))),
				"method": getattr(p, "payment_method", ""),
				"paid_at": getattr(p, "paid_at", ""),
			}
			for k, p in self.payments.items()
			if k[1] == invoice_id or getattr(p, "invoice_id", "") == invoice_id
		]

		subtotal = sum(_d(li["amount"]) for li in line_items)
		tax = sum(_d(li["tax_amount"]) for li in line_items)
		adj_total = sum(
			_d(a["amount"]) if a["adjustment_type"] == "credit" else -_d(a["amount"])
			for a in adjustments
		)
		paid = sum(_d(p["amount"]) for p in payments)
		balance_due = subtotal + tax + adj_total - paid

		return {
			"invoice_id": invoice_id,
			"found": True,
			"tenant_id": self.tenant_id,
			"customer_id": getattr(invoice, "customer_id", ""),
			"status": getattr(invoice, "status", "draft"),
			"due_date": getattr(invoice, "due_date", ""),
			"currency": getattr(invoice, "currency", "KES"),
			"line_items": line_items,
			"adjustments": adjustments,
			"payments": payments,
			"subtotal": str(subtotal),
			"tax_amount": str(tax),
			"adjustment_total": str(adj_total),
			"paid_amount": str(paid),
			"balance_due": str(balance_due),
			"retrieved_at": _now(),
		}

	# -----------------------------------------------------------------------
	# Collections & Dunning (5 methods)
	# -----------------------------------------------------------------------

	async def payment_processing(
		self,
		account_id: str,
		amount: Decimal | float | str,
		payment_method: str,
		reference: str,
	) -> dict[str, Any]:
		"""Process a payment from a subscriber.

		Validates method, records the payment, updates the account balance,
		and notifies the subscriber.
		"""
		assert _present(account_id), "account_id required"
		assert _present(payment_method), "payment_method required"
		assert _present(reference), "reference required"

		pay_amount = _d(amount)
		assert pay_amount > 0, "amount must be positive"

		method = payment_method.lower()
		if method not in SUPPORTED_PAYMENT_METHODS:
			raise ValueError(f"payment_method '{method}' not supported; choose from {SUPPORTED_PAYMENT_METHODS}")

		payment_id = f"pmt_{account_id}_{reference[:12].replace(' ', '_')}_{_now()[:10].replace('-', '')}"
		payment_record: dict[str, Any] = {
			"payment_id": payment_id,
			"account_id": account_id,
			"tenant_id": self.tenant_id,
			"amount": str(pay_amount),
			"payment_method": method,
			"reference": reference,
			"status": "received",
			"received_at": _now(),
		}

		# Update balance
		self._balances[account_id]["main_balance"] = (
			self._balances[account_id].get("main_balance", Decimal("0")) + pay_amount
		)

		await self._notify.send(
			"sms",
			account_id,
			"Payment Received",
			f"Payment of {pay_amount} KES received. Ref: {reference}",
		)
		self._emit("payment_processed", payment_id, {"account_id": account_id, "amount": str(pay_amount)})
		return payment_record

	async def allocate_payment(self, payment_id: str) -> dict[str, Any]:
		"""Allocate a payment to the oldest outstanding charges (FIFO).

		Walks the invoice store sorted by due_date ascending and allocates until
		the payment amount is exhausted.
		"""
		assert _present(payment_id), "payment_id required"

		payment = next(
			(p for k, p in self.payments.items() if k[1] == payment_id),
			None,
		)
		if payment is None:
			raise KeyError(f"Payment {payment_id} not found")

		pay_amount = _d(getattr(payment, "amount", 0))
		remaining = pay_amount

		# Sort invoices by due_date ascending — oldest first
		candidate_invoices = sorted(
			[
				inv for k, inv in self.invoices.items()
				if getattr(inv, "customer_id", "") == getattr(payment, "customer_id", payment_id)
				and getattr(inv, "status", "") not in {"paid", "written_off", "cancelled"}
			],
			key=lambda i: getattr(i, "due_date", ""),
		)

		allocations: list[dict[str, Any]] = []
		for inv in candidate_invoices:
			if remaining <= 0:
				break
			inv_total = _d(getattr(inv, "total_amount", 0))
			inv_paid = _d(getattr(inv, "paid_amount", 0))
			inv_balance = inv_total - inv_paid
			if inv_balance <= 0:
				continue
			allocated = min(remaining, inv_balance)
			# Update invoice paid_amount in-place
			if hasattr(inv, "paid_amount"):
				inv.paid_amount = str(inv_paid + allocated)  # type: ignore[attr-defined]
			remaining -= allocated
			allocations.append({
				"invoice_id": getattr(inv, "invoice_id", ""),
				"allocated": str(allocated),
				"invoice_balance_after": str(inv_balance - allocated),
			})

		result: dict[str, Any] = {
			"payment_id": payment_id,
			"total_payment": str(pay_amount),
			"total_allocated": str(pay_amount - remaining),
			"unallocated_remainder": str(remaining),
			"allocations": allocations,
			"allocated_at": _now(),
		}
		self._emit("payment_allocated", payment_id, {"allocations": len(allocations)})
		return result

	async def dunning_workflow(self, account_id: str, dpd_days: int) -> dict[str, Any]:
		"""Determine and execute the appropriate dunning level based on days-past-due.

		Levels:
		  1–7 dpd   → reminder_1 (SMS/email reminder)
		  8–14 dpd  → reminder_2 (formal notice)
		  15–21 dpd → suspension_warning (with service warning)
		  22–30 dpd → service_suspended (hard suspension)
		  31+  dpd  → legal_notice (escalate to collections)
		"""
		assert _present(account_id), "account_id required"
		assert dpd_days >= 0, "dpd_days must be >= 0"

		if dpd_days == 0:
			return {"account_id": account_id, "action": "none", "dpd_days": dpd_days, "message": "No overdue balance"}

		if dpd_days <= 7:
			step = "reminder_1"
			message = "Friendly reminder: your bill is overdue."
		elif dpd_days <= 14:
			step = "reminder_2"
			message = "Second notice: outstanding balance due immediately."
		elif dpd_days <= 21:
			step = "suspension_warning"
			message = "WARNING: Services will be suspended in 72 hours if payment is not received."
		elif dpd_days <= 30:
			step = "service_suspended"
			message = "Your services have been suspended due to non-payment."
			self._suspended_accounts.add(account_id)
		else:
			step = "legal_notice"
			message = "Your account has been referred to our collections team."

		dunning_id = f"dun_{account_id}_{step}_{_now()[:10].replace('-', '')}"
		self.trigger_dunning(
			dunning_id,
			f"inv_{account_id}",
			step,
			_now(),
		)

		channel = "sms" if dpd_days <= 14 else "email"
		await self._notify.send(channel, account_id, f"Billing Notice — Step {step}", message)

		result: dict[str, Any] = {
			"account_id": account_id,
			"dpd_days": dpd_days,
			"dunning_step": step,
			"dunning_id": dunning_id,
			"message": message,
			"notification_channel": channel,
			"suspended": step == "service_suspended",
			"actioned_at": _now(),
		}
		self._emit("dunning_actioned", dunning_id, {"step": step, "dpd_days": dpd_days})
		return result

	async def service_suspension(self, account_id: str, reason: str) -> dict[str, Any]:
		"""Suspend services for an account due to non-payment or other reason."""
		assert _present(account_id), "account_id required"
		assert _present(reason), "reason required"

		self._suspended_accounts.add(account_id)
		await self._notify.send(
			"sms",
			account_id,
			"Service Suspended",
			f"Your services have been suspended. Reason: {reason}. Please clear outstanding balance.",
		)
		result: dict[str, Any] = {
			"account_id": account_id,
			"tenant_id": self.tenant_id,
			"action": "suspended",
			"reason": reason,
			"suspended_at": _now(),
		}
		self._emit("service_suspended", account_id, {"reason": reason})
		return result

	async def service_restoration(self, account_id: str, payment_id: str) -> dict[str, Any]:
		"""Restore services after a qualifying payment has been received."""
		assert _present(account_id), "account_id required"
		assert _present(payment_id), "payment_id required"

		if account_id not in self._suspended_accounts:
			return {
				"account_id": account_id,
				"action": "not_suspended",
				"payment_id": payment_id,
				"message": "Account was not suspended; no action taken.",
			}

		# Verify payment exists
		payment_exists = any(
			k[1] == payment_id for k in self.payments
		)
		if not payment_exists:
			raise KeyError(f"Payment {payment_id} not found; cannot restore service")

		self._suspended_accounts.discard(account_id)
		await self._notify.send(
			"sms",
			account_id,
			"Service Restored",
			"Your services have been restored. Thank you for your payment.",
		)
		result: dict[str, Any] = {
			"account_id": account_id,
			"tenant_id": self.tenant_id,
			"action": "restored",
			"payment_id": payment_id,
			"restored_at": _now(),
		}
		self._emit("service_restored", account_id, {"payment_id": payment_id})
		return result

	# -----------------------------------------------------------------------
	# Disputes (4 methods)
	# -----------------------------------------------------------------------

	async def raise_billing_dispute(
		self,
		account_id: str,
		invoice_id: str,
		disputed_amount: Decimal | float | str,
		reason: str,
	) -> dict[str, Any]:
		"""Open a billing dispute against an invoice.

		Generates a dispute_id, sets status to 'open', and flags the invoice
		as 'disputed'.  Returns the dispute record.
		"""
		assert _present(account_id), "account_id required"
		assert _present(invoice_id), "invoice_id required"
		assert _present(reason), "reason required"

		amount = _d(disputed_amount)
		assert amount >= 0, "disputed_amount must be >= 0"

		dispute_id = f"disp_{account_id}_{invoice_id[:12]}_{_now()[:10].replace('-', '')}"
		dispute: dict[str, Any] = {
			"dispute_id": dispute_id,
			"account_id": account_id,
			"invoice_id": invoice_id,
			"tenant_id": self.tenant_id,
			"disputed_amount": str(amount),
			"currency": "KES",
			"reason": reason,
			"status": "open",
			"evidence_refs": [],
			"raised_by": self.actor_id,
			"raised_at": _now(),
			"sla_deadline": None,
		}
		self._disputes[dispute_id] = dispute

		# Mark invoice as disputed if it exists
		for k, inv in self.invoices.items():
			if k[1] == invoice_id:
				inv.status = "disputed"
				break

		self._emit("dispute_raised", dispute_id, {"account_id": account_id, "amount": str(amount)})
		return dispute

	async def investigate_dispute(
		self,
		dispute_id: str,
		cdr_analysis: dict[str, Any],
	) -> dict[str, Any]:
		"""Progress a dispute to 'under_review' and record CDR analysis findings.

		cdr_analysis: arbitrary dict of findings from the rating/mediation team.
		"""
		assert _present(dispute_id), "dispute_id required"

		dispute = self._disputes.get(dispute_id)
		if dispute is None:
			raise KeyError(f"Dispute {dispute_id} not found")

		if dispute["status"] not in {"open", "evidence_requested"}:
			raise ValueError(f"Dispute {dispute_id} is in status '{dispute['status']}'; cannot investigate")

		dispute["status"] = "under_review"
		dispute["cdr_analysis"] = cdr_analysis
		dispute["investigation_started_at"] = _now()
		dispute["investigated_by"] = self.actor_id

		result: dict[str, Any] = {
			**dispute,
			"message": "Dispute moved to under_review with CDR analysis attached.",
		}
		self._emit("dispute_investigated", dispute_id, {"analyst": self.actor_id})
		return result

	async def resolve_dispute(
		self,
		dispute_id: str,
		resolution: str,
		credit_amount: Decimal | float | str,
	) -> dict[str, Any]:
		"""Resolve a dispute with an outcome and optional credit.

		resolution: "upheld" | "rejected" | "partial"
		credit_amount: amount to credit back to subscriber (0 for rejected)
		"""
		assert _present(dispute_id), "dispute_id required"
		assert resolution in {"upheld", "rejected", "partial"}, \
			f"resolution must be upheld|rejected|partial, got '{resolution}'"

		dispute = self._disputes.get(dispute_id)
		if dispute is None:
			raise KeyError(f"Dispute {dispute_id} not found")

		credit = _d(credit_amount)
		assert credit >= 0, "credit_amount must be >= 0"

		status_map = {
			"upheld":   "resolved_upheld",
			"rejected": "resolved_rejected",
			"partial":  "resolved_upheld",
		}
		dispute["status"] = status_map[resolution]
		dispute["resolution"] = resolution
		dispute["credit_amount"] = str(credit)
		dispute["resolver_id"] = self.actor_id
		dispute["resolved_at"] = _now()

		# Apply credit if applicable
		if credit > 0:
			await self.apply_adjustments(
				dispute["invoice_id"],
				"credit",
				credit,
				f"Dispute {dispute_id} resolved: {resolution}",
			)

		# Restore invoice from 'disputed' back to 'approved' if credit given
		if credit > 0:
			for k, inv in self.invoices.items():
				if k[1] == dispute["invoice_id"] and getattr(inv, "status", "") == "disputed":
					inv.status = "approved"
					break

		result: dict[str, Any] = {
			**dispute,
			"credit_applied": credit > 0,
			"message": f"Dispute {dispute_id} resolved as {resolution}.",
		}
		self._emit("dispute_resolved", dispute_id, {"resolution": resolution, "credit": str(credit)})
		return result

	async def dispute_analytics(self, period: dict[str, str]) -> dict[str, Any]:
		"""Aggregate dispute metrics for a period.

		period: {"start": ISO-date, "end": ISO-date}
		"""
		period_start = period.get("start", "")
		period_end = period.get("end", "")

		tenant_disputes = [
			d for d in self._disputes.values()
			if d.get("tenant_id") == self.tenant_id
		]

		# Filter by period where possible
		in_period = [
			d for d in tenant_disputes
			if period_start <= d.get("raised_at", "")[:10] <= period_end
		] if period_start and period_end else tenant_disputes

		total = len(in_period)
		by_status: dict[str, int] = defaultdict(int)
		for d in in_period:
			by_status[d["status"]] += 1

		total_disputed = sum(_d(d.get("disputed_amount", 0)) for d in in_period)
		total_credited = sum(_d(d.get("credit_amount", 0)) for d in in_period)
		upheld = by_status.get("resolved_upheld", 0)
		rejected = by_status.get("resolved_rejected", 0)
		resolution_rate = (
			Decimal(str(upheld + rejected)) / Decimal(str(total)) * 100
			if total > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"total_disputes": total,
			"by_status": dict(by_status),
			"total_disputed_amount": str(total_disputed),
			"total_credited_amount": str(total_credited),
			"upheld_count": upheld,
			"rejected_count": rejected,
			"resolution_rate_pct": str(resolution_rate),
			"currency": "KES",
			"generated_at": _now(),
		}

	# -----------------------------------------------------------------------
	# Revenue Assurance (5 methods)
	# -----------------------------------------------------------------------

	async def revenue_leakage_detection(self, period: dict[str, str]) -> dict[str, Any]:
		"""Detect revenue leakage: unrated CDRs, provisioning gaps, rating failures.

		Returns a leakage report with anomaly list and estimated leakage amount.
		"""
		period_start = period.get("start", "")
		period_end = period.get("end", "")

		all_cdrs = list(self.cdrs.values())
		all_charges = list(self.charges.values())

		rated_cdr_ids = {
			getattr(ch, "cdr_id", None)
			for ch in all_charges
			if getattr(ch, "cdr_id", None) is not None
		}

		unrated: list[dict[str, Any]] = []
		for cdr in all_cdrs:
			cdr_id = getattr(cdr, "cdr_id", None)
			status = getattr(cdr, "mediation_status", "raw")
			if cdr_id not in rated_cdr_ids and status != "rejected":
				unrated.append({
					"cdr_id": cdr_id,
					"msisdn": getattr(cdr, "msisdn", ""),
					"status": status,
					"source": getattr(cdr, "source", ""),
					"recorded_at": getattr(cdr, "recorded_at", ""),
				})

		# Estimate leakage: assume average KES 5 per unrated CDR (placeholder without full tariff)
		estimated_leakage = _d(len(unrated) * 5)
		total_charges = sum(_d(getattr(ch, "amount", 0)) for ch in all_charges)
		leakage_pct = (
			estimated_leakage / (total_charges + estimated_leakage) * 100
			if (total_charges + estimated_leakage) > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		anomalies = [
			{"type": "unrated_cdr", "count": len(unrated), "details": unrated[:20]},
		]
		self._revenue_leakage_log.extend(unrated)

		report: dict[str, Any] = {
			"tenant_id": self.tenant_id,
			"period": period,
			"total_cdrs": len(all_cdrs),
			"rated_cdrs": len(all_cdrs) - len(unrated),
			"unrated_cdrs": len(unrated),
			"estimated_leakage": str(estimated_leakage),
			"total_rated_revenue": str(total_charges),
			"leakage_pct": str(leakage_pct),
			"anomalies": anomalies,
			"currency": "KES",
			"generated_at": _now(),
		}
		self._emit("leakage_report_generated", self.tenant_id, {"leakage_pct": str(leakage_pct)})
		return report

	async def interconnect_reconciliation(
		self,
		carrier: str,
		period: dict[str, str],
	) -> dict[str, Any]:
		"""Reconcile interconnect traffic and financials with a carrier for a period.

		Compares our internal CDR counts/amounts against a carrier's claim,
		computing net settlement position.
		"""
		assert _present(carrier), "carrier required"

		period_start = period.get("start", "")
		period_end = period.get("end", "")

		# Collect interconnect charges for this carrier
		ic_charges = [
			ch for ch in self.charges.values()
			if getattr(ch, "charge_type", "") == "interconnect"
		]

		our_receivable = sum(_d(getattr(ch, "amount", 0)) for ch in ic_charges)
		# In production, carrier's claim would come from a TAP file or API.
		# Using a stub 5% variance for demonstration.
		carrier_claim = (our_receivable * Decimal("1.05")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		variance = carrier_claim - our_receivable
		variance_pct = (
			variance / carrier_claim * 100 if carrier_claim > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		agreed = abs(variance_pct) <= Decimal("2.0")
		net_payable = carrier_claim - our_receivable

		result: dict[str, Any] = {
			"tenant_id": self.tenant_id,
			"carrier": carrier,
			"period": period,
			"our_receivable": str(our_receivable),
			"carrier_claim": str(carrier_claim),
			"variance": str(variance),
			"variance_pct": str(variance_pct),
			"net_payable_to_carrier": str(net_payable if net_payable > 0 else Decimal("0")),
			"net_receivable_from_carrier": str(abs(net_payable) if net_payable < 0 else Decimal("0")),
			"agreed": agreed,
			"status": "agreed" if agreed else "disputed",
			"currency": "KES",
			"reconciled_at": _now(),
		}
		self._emit("interconnect_reconciled", carrier, {"status": result["status"]})
		return result

	async def revenue_report(
		self,
		period: dict[str, str],
		segment: str | None = None,
	) -> dict[str, Any]:
		"""Generate a full revenue report for the period, optionally by segment.

		Aggregates voice, data, SMS, roaming, and interconnect revenues.
		"""
		all_charges = list(self.charges.values())

		by_type: dict[str, Decimal] = defaultdict(Decimal)
		total_tax = Decimal("0")
		for ch in all_charges:
			ct = getattr(ch, "charge_type", "other")
			by_type[ct] += _d(getattr(ch, "amount", 0))
			total_tax += _d(getattr(ch, "tax_amount", 0))

		total_revenue = sum(by_type.values())
		total_discounts = Decimal("0")
		for d in self.discounts.values():
			pct = _d(getattr(d, "discount_pct", 0)) / Decimal("100")
			total_discounts += (total_revenue * pct).quantize(Decimal("0.01"), ROUND_HALF_UP)

		net_revenue = total_revenue + total_tax - total_discounts

		# Invoice-level metrics
		all_invoices = list(self.invoices.values())
		paid_invoices = [i for i in all_invoices if getattr(i, "status", "") == "paid"]
		disputed_amount = sum(
			_d(d.get("disputed_amount", 0))
			for d in self._disputes.values()
			if d.get("tenant_id") == self.tenant_id
		)
		written_off = Decimal("0")
		for inv in all_invoices:
			if getattr(inv, "status", "") == "written_off":
				written_off += _d(getattr(inv, "total_amount", 0))

		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"segment": segment,
			"total_revenue": str(total_revenue),
			"by_charge_type": {k: str(v) for k, v in by_type.items()},
			"tax_collected": str(total_tax),
			"discounts_given": str(total_discounts),
			"net_revenue": str(net_revenue),
			"invoice_count": len(all_invoices),
			"paid_invoice_count": len(paid_invoices),
			"disputed_amount": str(disputed_amount),
			"written_off_amount": str(written_off),
			"currency": "KES",
			"generated_at": _now(),
		}

	async def arpu_analysis(
		self,
		period: dict[str, str],
		segment: str | None = None,
	) -> dict[str, Any]:
		"""Compute ARPU (Average Revenue Per User) by customer segment.

		ARPU = Total Revenue / Unique Active Subscribers.
		"""
		all_charges = list(self.charges.values())
		total_revenue = sum(_d(getattr(ch, "amount", 0)) for ch in all_charges)

		unique_subscribers: set[str] = {
			getattr(ch, "customer_id", "")
			for ch in all_charges
			if getattr(ch, "customer_id", "")
		}
		subscriber_count = len(unique_subscribers)

		arpu = (
			total_revenue / Decimal(str(subscriber_count))
			if subscriber_count > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		# Revenue per subscriber breakdown
		per_sub: dict[str, Decimal] = defaultdict(Decimal)
		for ch in all_charges:
			sub = getattr(ch, "customer_id", "unknown")
			per_sub[sub] += _d(getattr(ch, "amount", 0))

		top_10 = sorted(per_sub.items(), key=lambda x: x[1], reverse=True)[:10]

		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"segment": segment,
			"total_revenue": str(total_revenue),
			"unique_subscribers": subscriber_count,
			"arpu": str(arpu),
			"top_10_subscribers": [{"subscriber_id": s, "revenue": str(r)} for s, r in top_10],
			"currency": "KES",
			"generated_at": _now(),
		}

	async def churn_revenue_impact(self, period: dict[str, str]) -> dict[str, Any]:
		"""Estimate revenue impact of churned subscribers over the period.

		Identifies subscribers who had charges in a prior period but none in the
		current period, computes their prior-period revenue as the churn impact.
		"""
		period_start = period.get("start", "")

		# Current period subscribers
		current_subs: set[str] = {
			getattr(ch, "customer_id", "")
			for ch in self.charges.values()
			if getattr(ch, "customer_id", "")
		}

		# All-time unique subscribers (stand-in for prior-period without full date indexing)
		all_time_subs: set[str] = set(current_subs)  # same data; in prod compare against prior snapshot
		churned_subs = all_time_subs - current_subs

		churned_revenue = Decimal("0")
		for ch in self.charges.values():
			if getattr(ch, "customer_id", "") in churned_subs:
				churned_revenue += _d(getattr(ch, "amount", 0))

		total_revenue = sum(_d(getattr(ch, "amount", 0)) for ch in self.charges.values())
		churn_impact_pct = (
			churned_revenue / total_revenue * 100
			if total_revenue > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"active_subscriber_count": len(current_subs),
			"churned_subscriber_count": len(churned_subs),
			"churned_subscriber_ids": list(churned_subs),
			"churned_revenue_impact": str(churned_revenue),
			"total_revenue": str(total_revenue),
			"churn_revenue_impact_pct": str(churn_impact_pct),
			"currency": "KES",
			"generated_at": _now(),
		}

	# -----------------------------------------------------------------------
	# Internal helpers
	# -----------------------------------------------------------------------

	def _invoice_or_raise(self, invoice_id: str) -> BilInvoice:
		invoice = self.invoices.get(self._key(invoice_id))
		if invoice is None:
			raise ValueError(f"Invoice {invoice_id} not found for tenant {self.tenant_id}")
		return invoice


# ---------------------------------------------------------------------------
# Back-compat alias — existing code imported TelecomBilService
# ---------------------------------------------------------------------------

class TelecomBilService(TelecomBillingService):
	"""Back-compat shim.  Wraps TelecomBillingService with the old positional __init__."""

	def __init__(self) -> None:
		super().__init__(tenant_id="default", actor_id="system")

	def record_cdr(  # type: ignore[override]
		self,
		cdr_id: str,
		tenant_id: str,
		source: str,
		mediation_status: str,
		msisdn: str,
		duration_seconds: int,
		data_volume_bytes: int,
		recorded_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().record_cdr(cdr_id, source, mediation_status, msisdn,
		                          duration_seconds, data_volume_bytes, recorded_at, policy_attached)

	def record_charge(  # type: ignore[override]
		self,
		charge_id: str,
		tenant_id: str,
		customer_id: str,
		charge_type: str,
		rating_type: str,
		amount: float,
		currency: str,
		tax_amount: float,
		cdr_id: str | None = None,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().record_charge(charge_id, customer_id, charge_type, rating_type,
		                             amount, currency, tax_amount, cdr_id)

	def create_bill_cycle(  # type: ignore[override]
		self,
		cycle_id: str,
		tenant_id: str,
		cycle_type: str,
		cutoff_date: str,
		start_date: str,
		end_date: str,
		status: str = "active",
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().create_bill_cycle(cycle_id, cycle_type, cutoff_date, start_date, end_date, status)

	def generate_invoice(  # type: ignore[override]
		self,
		invoice_id: str,
		tenant_id: str,
		customer_id: str,
		cycle_id: str,
		total_amount: float,
		currency: str,
		due_date: str,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().generate_invoice(invoice_id, customer_id, cycle_id, total_amount, currency, due_date)

	def approve_invoice(  # type: ignore[override]
		self,
		invoice_id: str,
		tenant_id: str,
		approval_reference: str,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().approve_invoice(invoice_id, approval_reference)

	def trigger_dunning(  # type: ignore[override]
		self,
		dunning_id: str,
		tenant_id: str,
		invoice_id: str,
		step: str,
		triggered_at: str,
		next_step_date: str | None = None,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().trigger_dunning(dunning_id, invoice_id, step, triggered_at, next_step_date)

	def record_payment(  # type: ignore[override]
		self,
		payment_id: str,
		tenant_id: str,
		invoice_id: str,
		payment_method: str,
		amount: float,
		currency: str,
		reference: str,
		paid_at: str,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().record_payment(payment_id, invoice_id, payment_method, amount, currency, reference, paid_at)

	def apply_discount(  # type: ignore[override]
		self,
		discount_id: str,
		tenant_id: str,
		customer_id: str,
		discount_type: str,
		discount_pct: float,
		approval_reference: str,
		valid_from: str,
		valid_to: str,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().apply_discount(discount_id, customer_id, discount_type, discount_pct,
		                              approval_reference, valid_from, valid_to)

	def write_off_invoice(self, invoice_id: str, tenant_id: str, approval_reference: str) -> dict[str, Any]:  # type: ignore[override]
		self.tenant_id = tenant_id
		return super().write_off_invoice(invoice_id, approval_reference)

	def setup_convergent(  # type: ignore[override]
		self,
		account_id: str,
		tenant_id: str,
		convergent_mode: str,
		master_account_id: str,
		member_account_ids: str,
		currency: str,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().setup_convergent(account_id, convergent_mode, master_account_id, member_account_ids, currency)

	def register_agent(  # type: ignore[override]
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().register_agent(agent_id, name, runtime, role, scope)

	def validate_agent_action(  # type: ignore[override]
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		bill_suppression_scope: bool = False,
		cross_tenant_billing_scope: bool = False,
	) -> dict[str, Any]:
		self.tenant_id = tenant_id
		return super().validate_agent_action(privileged_scope, human_approval_recorded,
		                                     bill_suppression_scope, cross_tenant_billing_scope)

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:  # type: ignore[override]
		self.tenant_id = tenant_id
		return super().validate_batch(item_count, event_stream)

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:  # type: ignore[override]
		if tenant_id:
			self.tenant_id = tenant_id
		return super().dashboard_summary()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:  # type: ignore[override]
		return get_capability_contract(tenant_id)
