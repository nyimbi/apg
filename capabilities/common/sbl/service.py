"""SaaS Billing Engine — async service layer for APG common/sbl.

Entry point:  SaaSBillingService (also aliased as CommonSblService).

All public methods are async.  Storage is in-memory for the reference
implementation — swap self._store_* dicts for a real DB adapter.
"""

from __future__ import annotations

import calendar
import math
from datetime import datetime, timedelta, timezone
from typing import Any

try:
	from .capability_contract import (
		PLAN_DEFINITIONS,
		SUPPORTED_BILLING_CYCLES,
		SUPPORTED_CREDIT_NOTE_REASONS,
		SUPPORTED_PAYMENT_METHOD_TYPES,
		SUPPORTED_PLAN_TIERS,
		SUPPORTED_USAGE_METRICS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		SbCreditNote,
		SbInvoice,
		SbInvoiceLineItem,
		SbPaymentMethod,
		SbPlan,
		SbPlanLimits,
		SbSubscription,
		SbTenant,
		SbUsageRecord,
		uuid7str,
	)
except ImportError:  # pragma: no cover — standalone execution
	from capability_contract import (  # type: ignore
		PLAN_DEFINITIONS,
		SUPPORTED_BILLING_CYCLES,
		SUPPORTED_CREDIT_NOTE_REASONS,
		SUPPORTED_PAYMENT_METHOD_TYPES,
		SUPPORTED_PLAN_TIERS,
		SUPPORTED_USAGE_METRICS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		SbCreditNote,
		SbInvoice,
		SbInvoiceLineItem,
		SbPaymentMethod,
		SbPlan,
		SbPlanLimits,
		SbSubscription,
		SbTenant,
		SbUsageRecord,
		uuid7str,
	)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
	return datetime.now(timezone.utc)


def _parse_iso(ts: str | None) -> datetime | None:
	if not ts:
		return None
	try:
		dt = datetime.fromisoformat(ts)
		return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
	except (ValueError, TypeError):
		return None


def _add_months(dt: datetime, months: int) -> datetime:
	"""Advance dt by an integer number of months, clamping to last day."""
	month = dt.month + months
	year  = dt.year + (month - 1) // 12
	month = (month - 1) % 12 + 1
	day   = min(dt.day, calendar.monthrange(year, month)[1])
	return dt.replace(year=year, month=month, day=day)


def _invoice_number(seq: int) -> str:
	return f"INV-{datetime.now(timezone.utc).year}-{seq:05d}"


class SaaSBillingService:
	"""Tenant-scoped SaaS billing runtime.

	Storage convention: all _store_* dicts are keyed (tenant_id, object_id)
	except _plans which is keyed by plan name (global).
	"""

	def __init__(self) -> None:
		# Global plan catalog
		self._plans:          dict[str, SbPlan]                                 = {}
		# Tenant stores — keyed (tenant_id, obj_id)
		self._tenants:        dict[str, SbTenant]                               = {}
		self._subscriptions:  dict[tuple[str, str], SbSubscription]             = {}
		self._usage_records:  dict[tuple[str, str], SbUsageRecord]              = {}
		self._invoices:       dict[tuple[str, str], SbInvoice]                  = {}
		self._payment_methods: dict[tuple[str, str], SbPaymentMethod]           = {}
		self._credit_notes:   dict[tuple[str, str], SbCreditNote]               = {}
		# Dedup: idempotency_key → usage_record_id
		self._usage_idempotency: dict[tuple[str, str], str]                     = {}
		# Invoice sequence counter (global)
		self._invoice_seq:    int                                                = 0
		# Audit log
		self._audit_events:   list[dict[str, Any]]                              = []

		# Seed default plans from PLAN_DEFINITIONS
		self._seed_plans()

	# -----------------------------------------------------------------------
	# Plan seeding (called once on init)
	# -----------------------------------------------------------------------

	def _seed_plans(self) -> None:
		"""Populate the plan catalog from PLAN_DEFINITIONS."""
		for tier, defn in PLAN_DEFINITIONS.items():
			plan = SbPlan(
				id=uuid7str(),
				name=tier,
				display_name=defn["display_name"],
				tier=tier,
				price_monthly_cents=defn["price_monthly_cents"],
				price_annual_cents=defn["price_annual_cents"],
				limits=SbPlanLimits(**defn["limits"]),
				features=list(defn.get("features", [])),
				overage_allowed=defn.get("overage_allowed", False),
				overage_rates=dict(defn.get("overage_rates", {})),
			)
			self._plans[tier] = plan

	# -----------------------------------------------------------------------
	# Internal helpers
	# -----------------------------------------------------------------------

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "billing_policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "billing_policy_denied")

	def _audit(self, tenant_id: str, event_type: str, reference_id: str, metadata: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"tenant_id":    tenant_id,
			"event_type":   event_type,
			"reference_id": reference_id,
			"ts":           _now(),
			"processor":    "bytewax",
			"metadata":     metadata or {},
		})

	def _active_subscription(self, tenant_id: str) -> SbSubscription | None:
		"""Return the active subscription for a tenant or None."""
		for (tid, _), sub in self._subscriptions.items():
			if tid == tenant_id and sub.status == "active":
				return sub
		return None

	def _plan_for_id(self, plan_id: str) -> SbPlan | None:
		"""Look up a plan by its UUID id or by tier name."""
		# Try by tier name first (most common case)
		if plan_id in self._plans:
			return self._plans[plan_id]
		# Fall back to searching by id
		for plan in self._plans.values():
			if plan.id == plan_id:
				return plan
		return None

	def _usage_in_period(self, tenant_id: str, metric: str, period_start: str, period_end: str) -> int:
		"""Sum usage quantity for a tenant/metric within [period_start, period_end]."""
		start_dt = _parse_iso(period_start)
		end_dt   = _parse_iso(period_end)
		total    = 0
		for (tid, _), rec in self._usage_records.items():
			if tid != tenant_id or rec.metric != metric:
				continue
			rec_dt = _parse_iso(rec.timestamp)
			if rec_dt is None:
				continue
			if start_dt and rec_dt < start_dt:
				continue
			if end_dt and rec_dt > end_dt:
				continue
			total += rec.quantity
		return total

	def _next_invoice_number(self) -> str:
		self._invoice_seq += 1
		return _invoice_number(self._invoice_seq)

	# -----------------------------------------------------------------------
	# Tenant lifecycle
	# -----------------------------------------------------------------------

	async def create_tenant(
		self,
		name: str,
		email: str,
		plan_id: str,
		tenant_id: str | None = None,
	) -> SbTenant:
		"""Create a new tenant and start a trial subscription.

		Args:
			name:      Company/organisation display name.
			email:     Billing contact email.
			plan_id:   Tier name (free|starter|professional|enterprise) or plan UUID.
			tenant_id: Optional; supply for deterministic IDs (e.g. tests).

		Returns:
			The newly created SbTenant.
		"""
		self._enforce({
			"tenant_context_present": True,
			"operation_type":         "write",
			"policy_attached":        True,
			"operation":              "create_tenant",
			"tenant_email_present":   bool(email and email.strip()),
			"tenant_plan_present":    bool(plan_id and plan_id.strip()),
		})
		assert name and name.strip(),   "name is required"
		assert email and email.strip(), "email is required"

		plan = self._plan_for_id(plan_id)
		assert plan is not None, f"unknown plan_id {plan_id!r}"

		tid  = tenant_id or uuid7str()
		now  = _now_dt()
		trial_ends = (now + timedelta(days=14)).isoformat() if plan.tier != "enterprise" else None

		tenant = SbTenant(
			id=tid,
			name=name.strip(),
			email=email.strip(),
			plan_id=plan.id,
			status="trial" if trial_ends else "active",
			trial_ends_at=trial_ends,
		)
		self._tenants[tid] = tenant
		self._audit(tid, "tenant_created", tid, {"plan": plan.tier})
		return tenant

	# -----------------------------------------------------------------------
	# Subscription management
	# -----------------------------------------------------------------------

	async def create_subscription(
		self,
		tenant_id:     str,
		plan_id:       str,
		billing_cycle: str = "monthly",
	) -> SbSubscription:
		"""Create an active subscription for a tenant.

		Replaces any existing active subscription (idempotent: calling again
		with the same plan is a no-op returning the existing subscription).
		"""
		self._enforce({
			"tenant_context_present":        bool(tenant_id),
			"operation_type":                "write",
			"policy_attached":               True,
			"operation":                     "create_subscription",
			"subscription_tenant_present":   bool(tenant_id),
			"subscription_plan_present":     bool(plan_id),
			"billing_cycle_supported":       billing_cycle in SUPPORTED_BILLING_CYCLES,
		})
		plan = self._plan_for_id(plan_id)
		assert plan is not None, f"unknown plan_id {plan_id!r}"

		# Idempotency: return existing active sub if same plan
		existing = self._active_subscription(tenant_id)
		if existing and existing.plan_id == plan.id:
			return existing

		now     = _now_dt()
		if billing_cycle == "annual":
			period_end = _add_months(now, 12)
		else:
			period_end = _add_months(now, 1)

		sub = SbSubscription(
			id=uuid7str(),
			tenant_id=tenant_id,
			plan_id=plan.id,
			billing_cycle=billing_cycle,
			status="active",
			current_period_start=now.isoformat(),
			current_period_end=period_end.isoformat(),
			next_renewal_at=period_end.isoformat(),
		)
		self._subscriptions[(tenant_id, sub.id)] = sub

		# Update tenant plan and status
		tenant = self._tenants.get(tenant_id)
		if tenant:
			tenant.plan_id = plan.id
			tenant.status  = "active"
			tenant.updated_at = _now()

		self._audit(tenant_id, "subscription_created", sub.id, {"plan": plan.tier, "cycle": billing_cycle})
		return sub

	async def upgrade_plan(
		self,
		tenant_id:  str,
		new_plan_id: str,
	) -> SbSubscription:
		"""Upgrade (or downgrade) a tenant's plan with prorated billing.

		Prorated credit for unused time in current period is stored on the
		new subscription as proration_credit_cents; it is applied as a
		deduction on the next invoice.

		Returns:
			The new SbSubscription.
		"""
		self._enforce({
			"tenant_context_present":           bool(tenant_id),
			"operation_type":                   "write",
			"policy_attached":                  True,
			"operation":                        "upgrade_plan",
			"new_plan_present":                 bool(new_plan_id),
			"active_subscription_present":      self._active_subscription(tenant_id) is not None,
		})
		new_plan = self._plan_for_id(new_plan_id)
		assert new_plan is not None, f"unknown plan_id {new_plan_id!r}"

		existing = self._active_subscription(tenant_id)
		assert existing is not None, f"no active subscription for tenant {tenant_id!r}"

		# Calculate prorated credit for remaining days in current period
		proration_cents = 0
		now_dt = _now_dt()
		period_end_dt = _parse_iso(existing.current_period_end)
		period_start_dt = _parse_iso(existing.current_period_start)
		old_plan = self._plan_for_id(existing.plan_id)

		if period_end_dt and period_start_dt and old_plan:
			total_period_days = (period_end_dt - period_start_dt).days or 1
			remaining_days    = max(0, (period_end_dt - now_dt).days)
			daily_rate_cents  = old_plan.price_monthly_cents / total_period_days
			proration_cents   = math.floor(daily_rate_cents * remaining_days)

		# Cancel old subscription
		existing.status = "cancelled"
		existing.cancelled_at = _now()
		existing.cancellation_reason = f"upgraded_to:{new_plan.tier}"

		# Create new subscription inheriting the billing cycle
		if existing.billing_cycle == "annual":
			period_end = _add_months(now_dt, 12)
		else:
			period_end = _add_months(now_dt, 1)

		new_sub = SbSubscription(
			id=uuid7str(),
			tenant_id=tenant_id,
			plan_id=new_plan.id,
			billing_cycle=existing.billing_cycle,
			status="active",
			current_period_start=now_dt.isoformat(),
			current_period_end=period_end.isoformat(),
			next_renewal_at=period_end.isoformat(),
			proration_credit_cents=max(0, proration_cents),
		)
		self._subscriptions[(tenant_id, new_sub.id)] = new_sub

		# Update tenant record
		tenant = self._tenants.get(tenant_id)
		if tenant:
			tenant.plan_id   = new_plan.id
			tenant.updated_at = _now()

		self._audit(tenant_id, "subscription_upgraded", new_sub.id, {
			"old_plan":         old_plan.tier if old_plan else "unknown",
			"new_plan":         new_plan.tier,
			"proration_cents":  proration_cents,
		})
		return new_sub

	async def cancel_subscription(
		self,
		tenant_id:  str,
		reason:     str = "",
		immediate:  bool = False,
	) -> SbSubscription:
		"""Cancel a tenant's active subscription.

		Args:
			immediate: If True, cancel now; otherwise cancel at period end.
		"""
		existing = self._active_subscription(tenant_id)
		assert existing is not None, f"no active subscription for tenant {tenant_id!r}"

		if immediate:
			existing.status           = "cancelled"
			existing.cancelled_at     = _now()
		else:
			# Scheduled cancellation — mark as cancelling but stay active until period_end
			existing.metadata["cancel_at_period_end"] = True
			existing.metadata["scheduled_cancel_at"]  = existing.current_period_end

		existing.cancellation_reason = reason

		tenant = self._tenants.get(tenant_id)
		if tenant and immediate:
			tenant.status     = "cancelled"
			tenant.updated_at = _now()

		self._audit(tenant_id, "subscription_cancelled", existing.id, {"immediate": immediate, "reason": reason})
		return existing

	# -----------------------------------------------------------------------
	# Usage metering
	# -----------------------------------------------------------------------

	async def record_usage(
		self,
		tenant_id:        str,
		metric:           str,
		quantity:         int,
		timestamp:        str | None = None,
		idempotency_key:  str | None = None,
		source:           str = "api",
	) -> SbUsageRecord:
		"""Record a metered usage event.

		Args:
			tenant_id:       Owning tenant.
			metric:          Usage metric name (must be in SUPPORTED_USAGE_METRICS).
			quantity:        Positive integer units consumed.
			timestamp:       ISO-8601 event time; defaults to now.
			idempotency_key: If supplied, duplicate submissions are silently ignored.
			source:          Origin tag (api|batch|webhook|internal).

		Returns:
			The persisted SbUsageRecord.
		"""
		self._enforce({
			"tenant_context_present":   bool(tenant_id),
			"operation_type":           "write",
			"policy_attached":          True,
			"operation":                "record_usage",
			"usage_metric_supported":   metric in SUPPORTED_USAGE_METRICS,
			"usage_quantity_positive":  quantity > 0,
			"backdated_beyond_window":  False,   # TODO: add clock-skew check
		})

		# Idempotency dedup
		if idempotency_key:
			ikey = (tenant_id, idempotency_key)
			if ikey in self._usage_idempotency:
				existing_id = self._usage_idempotency[ikey]
				# Return existing record if present
				for (tid, rid), rec in self._usage_records.items():
					if tid == tenant_id and rid == existing_id:
						return rec
				# Record disappeared (shouldn't happen in-memory) — fall through

		sub = self._active_subscription(tenant_id)
		record = SbUsageRecord(
			id=uuid7str(),
			tenant_id=tenant_id,
			subscription_id=sub.id if sub else "",
			metric=metric,
			quantity=quantity,
			timestamp=timestamp or _now(),
			idempotency_key=idempotency_key,
			source=source,
		)
		self._usage_records[(tenant_id, record.id)] = record

		if idempotency_key:
			self._usage_idempotency[(tenant_id, idempotency_key)] = record.id

		self._audit(tenant_id, "usage_recorded", record.id, {"metric": metric, "quantity": quantity})

		# Emit limit-approaching signal
		await self._check_and_signal_limit(tenant_id, metric)

		return record

	async def _check_and_signal_limit(self, tenant_id: str, metric: str) -> None:
		"""Emit an audit event when usage reaches 80% or 100% of limit."""
		usage = await self.get_current_usage(tenant_id)
		metrics = usage.get("metrics", {})
		if metric not in metrics:
			return
		m = metrics[metric]
		if m.get("limit") == -1:  # unlimited
			return
		pct = m.get("percent_used", 0.0)
		if pct >= 100.0:
			self._audit(tenant_id, "usage_limit_exceeded", metric, {"metric": metric, "percent_used": pct})
		elif pct >= 80.0:
			self._audit(tenant_id, "usage_limit_approaching", metric, {"metric": metric, "percent_used": pct})

	# -----------------------------------------------------------------------
	# Usage queries
	# -----------------------------------------------------------------------

	async def get_current_usage(self, tenant_id: str) -> dict[str, Any]:
		"""Return current-period usage totals vs plan limits for all metrics.

		Returns a dict shaped::

			{
			    "tenant_id": ...,
			    "plan":      "starter",
			    "period_start": ...,
			    "period_end":   ...,
			    "metrics": {
			        "api_calls": {
			            "used":         5432,
			            "limit":        10000,
			            "remaining":    4568,
			            "percent_used": 54.32,
			            "overage":      0,
			        },
			        ...
			    }
			}
		"""
		sub  = self._active_subscription(tenant_id)
		plan = self._plan_for_id(sub.plan_id) if sub else None

		if not sub or not plan:
			return {
				"tenant_id":     tenant_id,
				"plan":          None,
				"period_start":  None,
				"period_end":    None,
				"metrics":       {},
				"as_of":         _now(),
			}

		metrics_summary: dict[str, Any] = {}
		for metric in SUPPORTED_USAGE_METRICS:
			limit   = getattr(plan.limits, metric, 0)
			used    = self._usage_in_period(tenant_id, metric, sub.current_period_start, sub.current_period_end)
			if limit == -1:
				remaining   = -1
				pct         = 0.0
				overage     = 0
			else:
				remaining   = max(0, limit - used)
				pct         = round(used / limit * 100, 2) if limit > 0 else 0.0
				overage     = max(0, used - limit)
			metrics_summary[metric] = {
				"used":         used,
				"limit":        limit,
				"remaining":    remaining,
				"percent_used": pct,
				"overage":      overage,
			}

		return {
			"tenant_id":    tenant_id,
			"plan":         plan.name,
			"period_start": sub.current_period_start,
			"period_end":   sub.current_period_end,
			"metrics":      metrics_summary,
			"as_of":        _now(),
		}

	async def check_limit(self, tenant_id: str, metric: str) -> bool:
		"""Return True if tenant is within their plan limit for metric.

		Unlimited plans always return True.  Free plans with no active
		subscription return False.
		"""
		assert metric in SUPPORTED_USAGE_METRICS, f"unsupported metric {metric!r}"
		usage = await self.get_current_usage(tenant_id)
		m = usage.get("metrics", {}).get(metric)
		if not m:
			return False
		if m["limit"] == -1:
			return True
		return m["overage"] == 0

	# -----------------------------------------------------------------------
	# Invoice generation
	# -----------------------------------------------------------------------

	async def generate_invoice(
		self,
		tenant_id:    str,
		period_start: str,
		period_end:   str,
	) -> SbInvoice:
		"""Generate an invoice for a tenant covering a billing period.

		Computes:
		  1. Base subscription fee (prorated if mid-cycle start).
		  2. Overage charges per metric per the plan's overage_rates.
		  3. Applies proration_credit_cents from subscription.

		Returns:
			A draft SbInvoice with all line items populated.
		"""
		self._enforce({
			"tenant_context_present":  bool(tenant_id),
			"operation_type":          "write",
			"policy_attached":         True,
			"operation":               "generate_invoice",
			"invoice_period_present":  bool(period_start and period_end),
		})

		sub  = self._active_subscription(tenant_id)
		plan = self._plan_for_id(sub.plan_id) if sub else None

		assert sub is not None,  f"no active subscription for tenant {tenant_id!r}"
		assert plan is not None, f"plan not found for subscription"

		line_items: list[SbInvoiceLineItem] = []
		invoice_id = uuid7str()

		# --- Subscription fee line
		sub_fee_cents = plan.price_monthly_cents if sub.billing_cycle == "monthly" else plan.price_annual_cents
		if sub_fee_cents > 0:
			line_items.append(SbInvoiceLineItem(
				id=uuid7str(),
				invoice_id=invoice_id,
				description=f"{plan.display_name} subscription ({sub.billing_cycle})",
				item_type="subscription_fee",
				quantity=1.0,
				unit_price_cents=sub_fee_cents,
				amount_cents=sub_fee_cents,
				period_start=period_start,
				period_end=period_end,
			))

		# --- Overage lines
		if plan.overage_allowed:
			for metric, rate in plan.overage_rates.items():
				limit   = getattr(plan.limits, metric, 0)
				used    = self._usage_in_period(tenant_id, metric, period_start, period_end)
				overage = max(0, used - limit)
				if overage > 0:
					overage_cents = math.ceil(overage * rate)
					line_items.append(SbInvoiceLineItem(
						id=uuid7str(),
						invoice_id=invoice_id,
						description=f"{metric} overage ({overage:,} units × ${rate/100:.4f})",
						item_type="overage",
						metric=metric,
						quantity=float(overage),
						unit_price_cents=int(rate),
						amount_cents=overage_cents,
						period_start=period_start,
						period_end=period_end,
					))

		# --- Proration credit line
		proration_credit = sub.proration_credit_cents
		if proration_credit > 0:
			line_items.append(SbInvoiceLineItem(
				id=uuid7str(),
				invoice_id=invoice_id,
				description="Proration credit (unused period from previous plan)",
				item_type="credit",
				quantity=1.0,
				unit_price_cents=proration_credit,
				amount_cents=proration_credit,
			))
			# Zero out after applying
			sub.proration_credit_cents = 0

		# --- Totals
		subtotal = sum(li.amount_cents for li in line_items if li.item_type != "credit")
		credit   = sum(li.amount_cents for li in line_items if li.item_type == "credit")
		total    = max(0, subtotal - credit)
		now_dt   = _now_dt()
		due_date = (now_dt + timedelta(days=30)).isoformat()

		invoice = SbInvoice(
			id=invoice_id,
			tenant_id=tenant_id,
			subscription_id=sub.id,
			invoice_number=self._next_invoice_number(),
			status="open" if total > 0 else "paid",
			period_start=period_start,
			period_end=period_end,
			subtotal_cents=subtotal,
			discount_cents=credit,
			total_cents=total,
			amount_due_cents=total,
			line_items=line_items,
			due_date=due_date,
			paid_at=_now() if total == 0 else None,
		)
		self._invoices[(tenant_id, invoice.id)] = invoice
		self._audit(tenant_id, "invoice_generated", invoice.id, {
			"total_cents":   total,
			"period_start":  period_start,
			"period_end":    period_end,
		})
		return invoice

	async def list_invoices(self, tenant_id: str) -> list[SbInvoice]:
		"""Return all invoices for a tenant, newest first."""
		result = [
			inv for (tid, _), inv in self._invoices.items()
			if tid == tenant_id
		]
		result.sort(key=lambda i: i.created_at, reverse=True)
		return result

	async def mark_invoice_paid(
		self,
		tenant_id:   str,
		invoice_id:  str,
	) -> SbInvoice:
		"""Mark an invoice as paid."""
		inv = self._invoices.get((tenant_id, invoice_id))
		assert inv is not None, f"invoice {invoice_id!r} not found"
		assert inv.status in ("open", "draft"), f"cannot mark invoice with status {inv.status!r} as paid"
		inv.status  = "paid"
		inv.paid_at = _now()
		self._audit(tenant_id, "invoice_paid", invoice_id, {"amount_cents": inv.total_cents})
		return inv

	async def void_invoice(
		self,
		tenant_id:  str,
		invoice_id: str,
		reason:     str = "",
	) -> SbInvoice:
		"""Void an unpaid invoice."""
		self._enforce({
			"tenant_context_present":  True,
			"operation":               "modify_invoice",
			"invoice_status":          self._invoices.get((tenant_id, invoice_id), SbInvoice(tenant_id="x", id=invoice_id)).status,
		})
		inv = self._invoices.get((tenant_id, invoice_id))
		assert inv is not None, f"invoice {invoice_id!r} not found"
		assert inv.status != "paid", "cannot void a paid invoice; issue a credit note instead"
		inv.status   = "void"
		inv.voided_at = _now()
		self._audit(tenant_id, "invoice_voided", invoice_id, {"reason": reason})
		return inv

	# -----------------------------------------------------------------------
	# Payment methods
	# -----------------------------------------------------------------------

	async def attach_payment_method(
		self,
		tenant_id:   str,
		method_type: str,
		token:       str,
		last_four:   str | None = None,
		brand:       str | None = None,
		expiry_month: int | None = None,
		expiry_year:  int | None = None,
		set_as_default: bool = True,
	) -> SbPaymentMethod:
		"""Attach a tokenized payment method to a tenant.

		NEVER passes raw card numbers — only processor tokens are accepted.
		"""
		self._enforce({
			"tenant_context_present":          bool(tenant_id),
			"operation_type":                  "write",
			"policy_attached":                 True,
			"operation":                       "attach_payment_method",
			"payment_method_type_supported":   method_type in SUPPORTED_PAYMENT_METHOD_TYPES,
			"token_present":                   bool(token and token.strip()),
			"raw_card_number_present":         False,  # callers must tokenize before calling this
		})
		if set_as_default:
			# Unset existing defaults
			for (tid, _), pm in self._payment_methods.items():
				if tid == tenant_id:
					pm.is_default = False

		pm = SbPaymentMethod(
			id=uuid7str(),
			tenant_id=tenant_id,
			method_type=method_type,
			token=token.strip(),
			last_four=last_four,
			brand=brand,
			expiry_month=expiry_month,
			expiry_year=expiry_year,
			is_default=set_as_default,
		)
		self._payment_methods[(tenant_id, pm.id)] = pm
		self._audit(tenant_id, "payment_method_attached", pm.id, {"type": method_type})
		return pm

	async def list_payment_methods(self, tenant_id: str) -> list[SbPaymentMethod]:
		"""Return all payment methods for a tenant."""
		return [pm for (tid, _), pm in self._payment_methods.items() if tid == tenant_id]

	# -----------------------------------------------------------------------
	# Credit notes
	# -----------------------------------------------------------------------

	async def issue_credit_note(
		self,
		tenant_id:    str,
		invoice_id:   str,
		amount_cents: int,
		reason:       str,
		description:  str = "",
		approved_by:  str | None = None,
	) -> SbCreditNote:
		"""Issue a credit note against an invoice.

		Requires approval (approved_by must be supplied for amounts > 0).
		"""
		self._enforce({
			"tenant_context_present":           bool(tenant_id),
			"operation_type":                   "write",
			"policy_attached":                  True,
			"operation":                        "issue_credit_note",
			"credit_note_reason_present":       bool(reason and reason.strip()),
			"credit_note_reason_supported":     reason in SUPPORTED_CREDIT_NOTE_REASONS,
			"approval_present":                 bool(approved_by),
		})
		inv = self._invoices.get((tenant_id, invoice_id))
		assert inv is not None, f"invoice {invoice_id!r} not found for tenant {tenant_id!r}"
		assert amount_cents > 0, "amount_cents must be positive"

		cn = SbCreditNote(
			id=uuid7str(),
			tenant_id=tenant_id,
			invoice_id=invoice_id,
			reason=reason,
			amount_cents=amount_cents,
			description=description,
			approved_by=approved_by,
			approved_at=_now() if approved_by else None,
		)
		self._credit_notes[(tenant_id, cn.id)] = cn
		self._audit(tenant_id, "credit_note_issued", cn.id, {
			"invoice_id":    invoice_id,
			"amount_cents":  amount_cents,
			"reason":        reason,
		})
		return cn

	# -----------------------------------------------------------------------
	# Self-provisioning flow
	# -----------------------------------------------------------------------

	async def self_provision(
		self,
		plan_id:        str,
		company_name:   str,
		email:          str,
		payment_method: dict[str, Any] | None = None,
	) -> SbTenant:
		"""Full tenant onboarding in a single call.

		Workflow:
		  1. Create tenant.
		  2. Create active subscription.
		  3. Attach payment method (if supplied).
		  4. Activate tenant.

		Args:
			plan_id:        Tier name or plan UUID.
			company_name:   Organisation name.
			email:          Billing contact.
			payment_method: Optional dict with keys: method_type, token,
			                last_four, brand, expiry_month, expiry_year.

		Returns:
			Fully provisioned SbTenant.
		"""
		plan = self._plan_for_id(plan_id)
		assert plan is not None, f"unknown plan_id {plan_id!r}"

		tenant = await self.create_tenant(
			name=company_name,
			email=email,
			plan_id=plan_id,
		)

		sub = await self.create_subscription(
			tenant_id=tenant.id,
			plan_id=plan.id,
			billing_cycle="monthly",
		)

		if payment_method:
			await self.attach_payment_method(
				tenant_id=tenant.id,
				method_type=payment_method.get("method_type", "card"),
				token=payment_method.get("token", ""),
				last_four=payment_method.get("last_four"),
				brand=payment_method.get("brand"),
				expiry_month=payment_method.get("expiry_month"),
				expiry_year=payment_method.get("expiry_year"),
				set_as_default=True,
			)

		# Activate tenant (move from trial to active if paid plan)
		if plan.price_monthly_cents > 0:
			tenant.status     = "active"
			tenant.updated_at = _now()

		self._audit(tenant.id, "tenant_self_provisioned", tenant.id, {
			"plan":             plan.tier,
			"has_payment_method": payment_method is not None,
		})
		return tenant

	# -----------------------------------------------------------------------
	# Analytics & reporting
	# -----------------------------------------------------------------------

	async def billing_analytics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Platform-level (or tenant-level) billing analytics snapshot.

		Returns:
			MRR, ARR, active tenant count, churn count, plan distribution.
		"""
		tenants      = list(self._tenants.values())
		if tenant_id:
			tenants  = [t for t in tenants if t.id == tenant_id]

		active_count  = sum(1 for t in tenants if t.status == "active")
		trial_count   = sum(1 for t in tenants if t.status == "trial")
		churned_count = sum(1 for t in tenants if t.status == "cancelled")

		mrr_cents = 0
		plan_dist: dict[str, int] = {tier: 0 for tier in SUPPORTED_PLAN_TIERS}

		for tenant in tenants:
			if tenant.status not in ("active", "trial"):
				continue
			plan = self._plan_for_id(tenant.plan_id)
			if not plan:
				continue
			plan_dist[plan.tier] = plan_dist.get(plan.tier, 0) + 1
			sub = self._active_subscription(tenant.id)
			if sub and sub.billing_cycle == "annual":
				mrr_cents += plan.price_annual_cents // 12
			else:
				mrr_cents += plan.price_monthly_cents

		total_invoiced = sum(
			inv.total_cents for (tid, _), inv in self._invoices.items()
			if (tenant_id is None or tid == tenant_id) and inv.status in ("paid", "open")
		)
		total_paid = sum(
			inv.total_cents for (tid, _), inv in self._invoices.items()
			if (tenant_id is None or tid == tenant_id) and inv.status == "paid"
		)

		return {
			"tenant_id":     tenant_id or "platform",
			"active_tenants": active_count,
			"trial_tenants":  trial_count,
			"churned_tenants": churned_count,
			"plan_distribution": plan_dist,
			"mrr_cents":      mrr_cents,
			"arr_cents":      mrr_cents * 12,
			"total_invoiced_cents": total_invoiced,
			"total_paid_cents": total_paid,
			"collection_rate": round(total_paid / total_invoiced, 4) if total_invoiced else 1.0,
			"as_of":          _now(),
		}

	async def dunning_candidates(self) -> list[dict[str, Any]]:
		"""Return tenants with overdue invoices eligible for dunning emails."""
		now_dt = _now_dt()
		result: list[dict[str, Any]] = []
		for (tid, _), inv in self._invoices.items():
			if inv.status != "open":
				continue
			due_dt = _parse_iso(inv.due_date)
			if due_dt and due_dt < now_dt:
				days_overdue = (now_dt - due_dt).days
				tenant = self._tenants.get(tid)
				result.append({
					"tenant_id":     tid,
					"tenant_name":   tenant.name if tenant else "unknown",
					"email":         tenant.email if tenant else "",
					"invoice_id":    inv.id,
					"invoice_number": inv.invoice_number,
					"amount_due_cents": inv.amount_due_cents,
					"due_date":      inv.due_date,
					"days_overdue":  days_overdue,
				})
		result.sort(key=lambda r: r["days_overdue"], reverse=True)
		return result

	# -----------------------------------------------------------------------
	# Describe / evaluate (capability contract interface)
	# -----------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Tenant-level billing dashboard summary."""
		usage  = await self.get_current_usage(tenant_id)
		tenant = self._tenants.get(tenant_id)
		sub    = self._active_subscription(tenant_id)
		plan   = self._plan_for_id(sub.plan_id) if sub else None

		invoices      = await self.list_invoices(tenant_id)
		open_invoices = [i for i in invoices if i.status == "open"]
		paid_invoices = [i for i in invoices if i.status == "paid"]

		return {
			"tenant_id":         tenant_id,
			"tenant_name":       tenant.name if tenant else "unknown",
			"status":            tenant.status if tenant else "unknown",
			"plan":              plan.display_name if plan else "none",
			"billing_cycle":     sub.billing_cycle if sub else "none",
			"next_renewal_at":   sub.next_renewal_at if sub else None,
			"open_invoices":     len(open_invoices),
			"amount_due_cents":  sum(i.amount_due_cents for i in open_invoices),
			"paid_invoices":     len(paid_invoices),
			"total_paid_cents":  sum(i.total_cents for i in paid_invoices),
			"usage_summary":     usage.get("metrics", {}),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant_id),
			"as_of":             _now(),
		}


# Alias for project naming convention
CommonSblService = SaaSBillingService
