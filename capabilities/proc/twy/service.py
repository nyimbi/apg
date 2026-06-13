"""Three-Way Match Engine — service layer (proc_twy).

Business logic:
  1.  Retrieve PO, GR, Invoice by ID (tenant-scoped).
  2.  Resolve applicable TwMatchToleranceRule for each line (specificity cascade:
      line_item > category > vendor > global).
  3.  For each line:
        a. Price check:    |invoice_price − po_price| / po_price  ≤ price_tolerance_pct
        b. Quantity check: |invoice_qty  − gr_qty   | / gr_qty    ≤ qty_tolerance_pct
  4.  Header date check:   invoice_date ≤ po_date + payment_terms_days + date_tolerance_days
  5.  Aggregate variances → outcome: MATCHED | PARTIAL_MATCH | EXCEPTION
        MATCHED        → all lines within tolerance, no missing documents
        PARTIAL_MATCH  → at least one line within tolerance but not all; or minor header variance
        EXCEPTION      → any line outside tolerance, missing document, or header date breach
  6.  If outcome != EXCEPTION and auto_approve flag set → auto-approve and publish "auto_approved".
  7.  If outcome == EXCEPTION → create TwMatchException, publish "exception.raised".

Currency: all amounts are assumed same-currency (currency mismatch is an EXCEPTION).
Multi-currency conversion is handled by fin_arc adapter (out of scope here).

Concurrency: the service is stateless; all state is kept in _store dicts keyed by
  tenant_id → id.  Production deployments replace _store with DB/ORM calls.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

try:
	from .models import (
		TwDocumentLine,
		TwDocumentType,
		TwExceptionResolutionType,
		TwExceptionStatus,
		TwMatchAttempt,
		TwMatchDocument,
		TwMatchException,
		TwMatchOutcome,
		TwMatchResult,
		TwMatchStatus,
		TwMatchToleranceRule,
		TwToleranceScope,
		TwVarianceDetail,
		TwVarianceType,
		uuid7str,
	)
	from .capability_contract import (
		NATS_PUBLISHES,
		NATS_SUBSCRIBES,
		evaluate_capability_rules,
	)
except ImportError:  # pragma: no cover
	from models import (  # type: ignore
		TwDocumentLine,
		TwDocumentType,
		TwExceptionResolutionType,
		TwExceptionStatus,
		TwMatchAttempt,
		TwMatchDocument,
		TwMatchException,
		TwMatchOutcome,
		TwMatchResult,
		TwMatchStatus,
		TwMatchToleranceRule,
		TwToleranceScope,
		TwVarianceDetail,
		TwVarianceType,
		uuid7str,
	)
	from capability_contract import (  # type: ignore
		NATS_PUBLISHES,
		NATS_SUBSCRIBES,
		evaluate_capability_rules,
	)

# ---------------------------------------------------------------------------
# Tolerance cascade order (most specific first)
# ---------------------------------------------------------------------------
_SCOPE_PRIORITY: dict[TwToleranceScope, int] = {
	TwToleranceScope.LINE_ITEM: 10,
	TwToleranceScope.CATEGORY: 20,
	TwToleranceScope.VENDOR: 30,
	TwToleranceScope.GLOBAL: 40,
}

# Default tolerance when no rule matches
_DEFAULT_PRICE_TOL_PCT = 2.0
_DEFAULT_QTY_TOL_PCT = 5.0
_DEFAULT_DATE_TOL_DAYS = 30

# Auto-approve cap: never auto-approve matches where invoice > po_total + x%
_AUTO_APPROVE_MAX_INVOICE_OVER_PO_PCT = 10.0


def _now() -> datetime:
	return datetime.now(timezone.utc)


def _pct(numerator: Decimal, denominator: Decimal) -> float:
	"""Return percentage variance, guarding against division by zero."""
	if denominator == Decimal("0"):
		return 0.0 if numerator == Decimal("0") else 100.0
	return float((abs(numerator) / abs(denominator) * 100).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _audit_entry(event: str, actor: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
	return {
		"event": event,
		"actor": actor,
		"at": _now().isoformat(),
		"detail": detail or {},
	}


# ---------------------------------------------------------------------------
# NATS stub (replaced by real nats.py client in production)
# ---------------------------------------------------------------------------


async def _publish_event(subject: str, payload: dict[str, Any]) -> None:
	"""Stub NATS publisher.  Real implementations inject a nats.aio.client.Client."""
	# In production: await nats_client.publish(subject, json.dumps(payload).encode())
	pass  # noqa: PIE790


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ThreeWayMatchService:
	"""Tenant-scoped Three-Way Match Engine.

	All public methods are async; state stores are in-memory dicts that mirror
	what a PostgreSQL-backed implementation would provide.
	"""

	def __init__(self) -> None:
		# tenant_id → { doc_id: TwMatchDocument }
		self._documents: dict[str, dict[str, TwMatchDocument]] = {}
		# tenant_id → { attempt_id: TwMatchAttempt }
		self._attempts: dict[str, dict[str, TwMatchAttempt]] = {}
		# tenant_id → { result_id: TwMatchResult }
		self._results: dict[str, dict[str, TwMatchResult]] = {}
		# tenant_id → { exception_id: TwMatchException }
		self._exceptions: dict[str, dict[str, TwMatchException]] = {}
		# tenant_id → { rule_id: TwMatchToleranceRule }
		self._tolerance_rules: dict[str, dict[str, TwMatchToleranceRule]] = {}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_docs(self, tenant_id: str) -> dict[str, TwMatchDocument]:
		return self._documents.setdefault(tenant_id, {})

	def _tenant_attempts(self, tenant_id: str) -> dict[str, TwMatchAttempt]:
		return self._attempts.setdefault(tenant_id, {})

	def _tenant_results(self, tenant_id: str) -> dict[str, TwMatchResult]:
		return self._results.setdefault(tenant_id, {})

	def _tenant_exceptions(self, tenant_id: str) -> dict[str, TwMatchException]:
		return self._exceptions.setdefault(tenant_id, {})

	def _tenant_rules(self, tenant_id: str) -> dict[str, TwMatchToleranceRule]:
		return self._tolerance_rules.setdefault(tenant_id, {})

	def _get_doc(self, tenant_id: str, doc_id: str) -> TwMatchDocument | None:
		return self._tenant_docs(tenant_id).get(doc_id)

	def _active_rules(self, tenant_id: str) -> list[TwMatchToleranceRule]:
		"""Return tenant's active rules sorted by specificity (most specific first)."""
		now = _now()
		rules = [
			r for r in self._tenant_rules(tenant_id).values()
			if r.active
			and r.effective_from <= now
			and (r.effective_to is None or r.effective_to >= now)
		]
		return sorted(rules, key=lambda r: (_SCOPE_PRIORITY.get(r.scope, 99), r.priority))

	def _resolve_tolerance(
		self,
		tenant_id: str,
		vendor_id: str,
		category_code: str,
		item_code: str,
		amount: Decimal,
	) -> tuple[float, float, int, str | None]:
		"""Resolve (price_tol_pct, qty_tol_pct, date_tol_days, rule_id) for a line.

		Cascade: LINE_ITEM > CATEGORY > VENDOR > GLOBAL.
		For each matching rule also apply amount-band overrides when set.
		"""
		for rule in self._active_rules(tenant_id):
			match = False
			if rule.scope == TwToleranceScope.LINE_ITEM and rule.item_code == item_code:
				match = True
			elif rule.scope == TwToleranceScope.CATEGORY and rule.category_code == category_code:
				match = True
			elif rule.scope == TwToleranceScope.VENDOR and rule.vendor_id == vendor_id:
				match = True
			elif rule.scope == TwToleranceScope.GLOBAL:
				match = True

			if match:
				# Amount-band override
				if (
					rule.amount_threshold is not None
					and amount > rule.amount_threshold
					and rule.price_tolerance_pct_above_threshold is not None
				):
					price_tol = rule.price_tolerance_pct_above_threshold
					qty_tol = rule.quantity_tolerance_pct_above_threshold or rule.quantity_tolerance_pct
				else:
					price_tol = rule.price_tolerance_pct
					qty_tol = rule.quantity_tolerance_pct
				return price_tol, qty_tol, rule.date_tolerance_days, rule.id

		return _DEFAULT_PRICE_TOL_PCT, _DEFAULT_QTY_TOL_PCT, _DEFAULT_DATE_TOL_DAYS, None

	# ------------------------------------------------------------------
	# Document ingestion
	# ------------------------------------------------------------------

	async def ingest_document(self, doc: TwMatchDocument) -> TwMatchDocument:
		"""Store a PO, GR, or Invoice for later matching."""
		assert doc.tenant_id, "tenant_id required"
		assert doc.document_type in TwDocumentType, f"unknown document_type: {doc.document_type}"
		self._tenant_docs(doc.tenant_id)[doc.id] = doc
		return doc

	async def get_document(self, doc_id: str, tenant_id: str) -> TwMatchDocument | None:
		return self._get_doc(tenant_id, doc_id)

	# ------------------------------------------------------------------
	# Core matching algorithm
	# ------------------------------------------------------------------

	async def match_documents(
		self,
		po_id: str,
		gr_id: str,
		invoice_id: str,
		tenant_id: str,
		initiated_by: str = "system",
	) -> TwMatchResult:
		"""Execute a three-way match.

		Steps:
		  1. Validate all three documents exist and belong to tenant_id.
		  2. Currency sanity check.
		  3. Line-level price and quantity matching.
		  4. Header-level date check.
		  5. Compute outcome.
		  6. Persist attempt and result.
		  7. Raise exception or auto-approve.
		  8. Publish NATS event.
		"""
		assert tenant_id, "tenant_id required"

		# --- 1. Retrieve documents ---
		po = self._get_doc(tenant_id, po_id)
		gr = self._get_doc(tenant_id, gr_id)
		inv = self._get_doc(tenant_id, invoice_id)

		attempt = TwMatchAttempt(
			tenant_id=tenant_id,
			po_id=po_id,
			gr_id=gr_id,
			invoice_id=invoice_id,
			status=TwMatchStatus.IN_PROGRESS,
			initiated_by=initiated_by,
		)
		self._tenant_attempts(tenant_id)[attempt.id] = attempt

		variances: list[TwVarianceDetail] = []
		tolerance_rule_ids: list[str] = []

		try:
			if po is None:
				variances.append(TwVarianceDetail(
					variance_type=TwVarianceType.DOCUMENT_MISSING,
					field_name="purchase_order",
					note=f"Purchase order {po_id!r} not found for tenant {tenant_id!r}",
				))
			if gr is None:
				variances.append(TwVarianceDetail(
					variance_type=TwVarianceType.DOCUMENT_MISSING,
					field_name="goods_receipt",
					note=f"Goods receipt {gr_id!r} not found for tenant {tenant_id!r}",
				))
			if inv is None:
				variances.append(TwVarianceDetail(
					variance_type=TwVarianceType.DOCUMENT_MISSING,
					field_name="vendor_invoice",
					note=f"Invoice {invoice_id!r} not found for tenant {tenant_id!r}",
				))

			if variances:
				# Missing documents → immediate exception
				outcome = TwMatchOutcome.EXCEPTION
				result = await self._build_result(
					tenant_id, attempt, po, gr, inv, variances, outcome,
					tolerance_rule_ids, auto_approved=False,
				)
				await self._raise_exception(result, variances)
				await _publish_event("match.completed", {"result_id": result.id, "outcome": outcome, "tenant_id": tenant_id})
				return result

			# Type sanity (not strictly required since IDs are passed directly)
			assert po.document_type == TwDocumentType.PURCHASE_ORDER, "po_id does not reference a PO"
			assert gr.document_type == TwDocumentType.GOODS_RECEIPT, "gr_id does not reference a GR"
			assert inv.document_type == TwDocumentType.VENDOR_INVOICE, "invoice_id does not reference an Invoice"

			# --- 2. Currency check ---
			if not (po.currency == gr.currency == inv.currency):
				variances.append(TwVarianceDetail(
					variance_type=TwVarianceType.PRICE,
					field_name="currency",
					po_value=po.currency,
					gr_value=gr.currency,
					invoice_value=inv.currency,
					within_tolerance=False,
					note="Currency mismatch across documents — route to fin_arc for FX conversion",
				))

			# --- 3. Line-level matching ---
			po_lines_by_item: dict[str, TwDocumentLine] = {ln.item_code: ln for ln in po.lines}
			gr_lines_by_item: dict[str, TwDocumentLine] = {ln.item_code: ln for ln in gr.lines}
			inv_lines_by_item: dict[str, TwDocumentLine] = {ln.item_code: ln for ln in inv.lines}

			# Check all item codes that appear on ANY document
			all_items = set(po_lines_by_item) | set(gr_lines_by_item) | set(inv_lines_by_item)

			for item_code in sorted(all_items):
				po_line = po_lines_by_item.get(item_code)
				gr_line = gr_lines_by_item.get(item_code)
				inv_line = inv_lines_by_item.get(item_code)

				# Missing-line variances
				if po_line is None:
					variances.append(TwVarianceDetail(
						variance_type=TwVarianceType.LINE_MISSING,
						field_name="purchase_order_line",
						invoice_value=item_code,
						gr_value=item_code if gr_line else None,
						within_tolerance=False,
						note=f"Item {item_code!r} found on invoice/GR but not on PO",
					))
					continue
				if gr_line is None:
					variances.append(TwVarianceDetail(
						variance_type=TwVarianceType.LINE_MISSING,
						field_name="goods_receipt_line",
						po_value=item_code,
						invoice_value=item_code if inv_line else None,
						within_tolerance=False,
						note=f"Item {item_code!r} not received (no GR line)",
					))
					continue
				if inv_line is None:
					# Item on PO+GR but not invoiced — not an exception, just informational
					continue

				# Resolve tolerance for this line
				line_amount = inv_line.line_total
				price_tol, qty_tol, date_tol, rule_id = self._resolve_tolerance(
					tenant_id, po.vendor_id, "", item_code, line_amount,
				)
				if rule_id and rule_id not in tolerance_rule_ids:
					tolerance_rule_ids.append(rule_id)

				# --- 3a. Price variance ---
				# Compare invoice unit_price vs PO unit_price (GR does not carry price)
				price_var_abs = inv_line.unit_price - po_line.unit_price
				price_var_pct = _pct(price_var_abs, po_line.unit_price)
				price_within = price_var_pct <= price_tol
				if not price_within or abs(float(price_var_abs)) > 0:
					variances.append(TwVarianceDetail(
						variance_type=TwVarianceType.PRICE,
						line_number=inv_line.line_number,
						field_name="unit_price",
						po_value=str(po_line.unit_price),
						gr_value=None,
						invoice_value=str(inv_line.unit_price),
						absolute_variance=price_var_abs,
						percentage_variance=price_var_pct,
						within_tolerance=price_within,
						tolerance_rule_id=rule_id,
					))

				# --- 3b. Quantity variance ---
				# Compare invoice qty vs GR qty (what was actually received)
				qty_var_abs = inv_line.quantity - gr_line.quantity
				qty_var_pct = _pct(qty_var_abs, gr_line.quantity)
				qty_within = qty_var_pct <= qty_tol
				if not qty_within or qty_var_abs != Decimal("0"):
					variances.append(TwVarianceDetail(
						variance_type=TwVarianceType.QUANTITY,
						line_number=inv_line.line_number,
						field_name="quantity",
						po_value=str(po_line.quantity),
						gr_value=str(gr_line.quantity),
						invoice_value=str(inv_line.quantity),
						absolute_variance=qty_var_abs,
						percentage_variance=qty_var_pct,
						within_tolerance=qty_within,
						tolerance_rule_id=rule_id,
					))

			# --- 4. Date check ---
			# Payment due date = PO document_date + payment_terms_days
			# Invoice must arrive within payment due date + date_tolerance_days
			_, _, date_tol_days, date_rule_id = self._resolve_tolerance(
				tenant_id, po.vendor_id, "", "", inv.total_amount,
			)
			payment_due = po.document_date + timedelta(days=po.payment_terms_days)
			deadline = payment_due + timedelta(days=date_tol_days)
			# Normalise to UTC-aware for comparison
			inv_date = inv.document_date if inv.document_date.tzinfo else inv.document_date.replace(tzinfo=timezone.utc)
			deadline_aware = deadline if deadline.tzinfo else deadline.replace(tzinfo=timezone.utc)
			date_var_days = max(0, (inv_date - deadline_aware).days)
			date_within = inv_date <= deadline_aware
			if not date_within:
				variances.append(TwVarianceDetail(
					variance_type=TwVarianceType.DATE,
					field_name="invoice_date",
					po_value=payment_due.isoformat(),
					invoice_value=inv_date.isoformat(),
					absolute_variance=Decimal(str(date_var_days)),
					percentage_variance=None,
					within_tolerance=False,
					tolerance_rule_id=date_rule_id,
					note=f"Invoice date {date_var_days}d past allowed deadline",
				))

		except Exception as exc:
			attempt.status = TwMatchStatus.FAILED
			attempt.completed_at = _now()
			attempt.error = str(exc)
			self._tenant_attempts(tenant_id)[attempt.id] = attempt
			raise

		# --- 5. Compute outcome ---
		outside_tol = [v for v in variances if not v.within_tolerance]
		inside_tol = [v for v in variances if v.within_tolerance]

		if not outside_tol:
			outcome = TwMatchOutcome.MATCHED
		elif inside_tol and outside_tol:
			outcome = TwMatchOutcome.PARTIAL_MATCH
		else:
			outcome = TwMatchOutcome.EXCEPTION

		# If only date variance is outside tolerance (but amounts match) → PARTIAL_MATCH
		if outside_tol and all(v.variance_type == TwVarianceType.DATE for v in outside_tol):
			outcome = TwMatchOutcome.PARTIAL_MATCH

		# If any DOCUMENT_MISSING or LINE_MISSING outside tolerance → EXCEPTION
		if any(v.variance_type in (TwVarianceType.DOCUMENT_MISSING, TwVarianceType.LINE_MISSING) for v in outside_tol):
			outcome = TwMatchOutcome.EXCEPTION

		attempt.status = TwMatchStatus.COMPLETED
		attempt.completed_at = _now()
		attempt.variances = variances
		attempt.tolerance_rules_applied = tolerance_rule_ids
		self._tenant_attempts(tenant_id)[attempt.id] = attempt

		# --- 6+7. Result, auto-approve or exception ---
		auto_approved = False
		if outcome != TwMatchOutcome.EXCEPTION:
			# Check auto-approve guard: invoice must not exceed po_total by more than threshold
			if po and inv:
				inv_over_po_pct = _pct(inv.total_amount - po.total_amount, po.total_amount)
				if inv_over_po_pct <= _AUTO_APPROVE_MAX_INVOICE_OVER_PO_PCT:
					auto_approved = True

		result = await self._build_result(
			tenant_id, attempt, po, gr, inv, variances, outcome,
			tolerance_rule_ids, auto_approved=auto_approved,
		)

		if outcome == TwMatchOutcome.EXCEPTION:
			await self._raise_exception(result, variances)
			await _publish_event("exception.raised", {"result_id": result.id, "tenant_id": tenant_id})
		elif auto_approved:
			await _publish_event("auto_approved", {"result_id": result.id, "tenant_id": tenant_id})

		await _publish_event("match.completed", {
			"result_id": result.id,
			"outcome": outcome,
			"tenant_id": tenant_id,
			"auto_approved": auto_approved,
		})
		return result

	async def _build_result(
		self,
		tenant_id: str,
		attempt: TwMatchAttempt,
		po: TwMatchDocument | None,
		gr: TwMatchDocument | None,
		inv: TwMatchDocument | None,
		variances: list[TwVarianceDetail],
		outcome: TwMatchOutcome,
		tolerance_rule_ids: list[str],
		auto_approved: bool,
	) -> TwMatchResult:
		price_vars = [v for v in variances if v.variance_type == TwVarianceType.PRICE and v.percentage_variance is not None]
		qty_vars = [v for v in variances if v.variance_type == TwVarianceType.QUANTITY and v.percentage_variance is not None]
		date_vars = [v for v in variances if v.variance_type == TwVarianceType.DATE and v.absolute_variance is not None]

		result = TwMatchResult(
			tenant_id=tenant_id,
			match_attempt_id=attempt.id,
			po_id=attempt.po_id,
			gr_id=attempt.gr_id,
			invoice_id=attempt.invoice_id,
			outcome=outcome,
			po_total=po.total_amount if po else Decimal("0"),
			gr_total=gr.total_amount if gr else Decimal("0"),
			invoice_total=inv.total_amount if inv else Decimal("0"),
			price_variance_pct=max((v.percentage_variance for v in price_vars), default=0.0),
			quantity_variance_pct=max((v.percentage_variance for v in qty_vars), default=0.0),
			date_variance_days=int(max((float(v.absolute_variance) for v in date_vars), default=0)),
			all_within_tolerance=all(v.within_tolerance for v in variances),
			variances=variances,
			auto_approved=auto_approved,
			matched_by=attempt.initiated_by,
			audit_trail=[_audit_entry("match_completed", attempt.initiated_by, {"outcome": outcome})],
		)
		self._tenant_results(tenant_id)[result.id] = result
		return result

	async def _raise_exception(self, result: TwMatchResult, variances: list[TwVarianceDetail]) -> TwMatchException:
		from datetime import timedelta as _td
		exc = TwMatchException(
			tenant_id=result.tenant_id,
			match_result_id=result.id,
			po_id=result.po_id,
			gr_id=result.gr_id,
			invoice_id=result.invoice_id,
			status=TwExceptionStatus.OPEN,
			variance_summary=variances,
			due_at=_now() + _td(days=30),
			audit_trail=[_audit_entry("exception_raised", "system", {"result_id": result.id})],
		)
		self._tenant_exceptions(result.tenant_id)[exc.id] = exc
		result.exception_id = exc.id
		self._tenant_results(result.tenant_id)[result.id] = result
		return exc

	# ------------------------------------------------------------------
	# Exception management
	# ------------------------------------------------------------------

	async def list_exceptions(
		self,
		tenant_id: str,
		status: TwExceptionStatus | None = None,
	) -> list[TwMatchException]:
		"""Return exceptions for a tenant, optionally filtered by status."""
		assert tenant_id, "tenant_id required"
		excs = list(self._tenant_exceptions(tenant_id).values())
		if status is not None:
			excs = [e for e in excs if e.status == status]
		# Refresh age_days
		now = _now()
		for e in excs:
			e.age_days = (now - e.raised_at).total_seconds() / 86400.0
		return sorted(excs, key=lambda e: e.raised_at)

	async def resolve_exception(
		self,
		exception_id: str,
		resolution: TwExceptionResolutionType,
		resolved_by: str,
		tenant_id: str,
		resolution_note: str = "",
	) -> TwMatchException:
		"""Mark an exception as resolved."""
		assert tenant_id, "tenant_id required"
		assert resolved_by and resolved_by.strip(), "resolved_by required"
		assert resolution_note and resolution_note.strip(), "resolution_note required"

		exc = self._tenant_exceptions(tenant_id).get(exception_id)
		if exc is None:
			raise ValueError(f"Exception {exception_id!r} not found for tenant {tenant_id!r}")
		if exc.status in (TwExceptionStatus.RESOLVED, TwExceptionStatus.CANCELLED):
			raise ValueError(f"Exception {exception_id!r} is already {exc.status}")

		exc.status = TwExceptionStatus.RESOLVED
		exc.resolution_type = resolution
		exc.resolution_note = resolution_note
		exc.resolved_by = resolved_by
		exc.resolved_at = _now()
		exc.audit_trail.append(_audit_entry("exception_resolved", resolved_by, {
			"resolution": resolution,
			"note": resolution_note,
		}))
		self._tenant_exceptions(tenant_id)[exception_id] = exc

		await _publish_event("exception.resolved", {
			"exception_id": exception_id,
			"resolution": resolution,
			"tenant_id": tenant_id,
		})
		return exc

	async def escalate_exception(
		self,
		exception_id: str,
		escalate_to: str,
		reason: str,
		tenant_id: str,
		escalated_by: str = "system",
	) -> TwMatchException:
		"""Escalate an open exception to a named reviewer / manager."""
		assert tenant_id, "tenant_id required"
		assert escalate_to and escalate_to.strip(), "escalate_to required"
		assert reason and reason.strip(), "reason required"

		exc = self._tenant_exceptions(tenant_id).get(exception_id)
		if exc is None:
			raise ValueError(f"Exception {exception_id!r} not found for tenant {tenant_id!r}")
		if exc.status == TwExceptionStatus.RESOLVED:
			raise ValueError(f"Cannot escalate resolved exception {exception_id!r}")

		exc.status = TwExceptionStatus.ESCALATED
		exc.escalated_to = escalate_to
		exc.escalation_reason = reason
		exc.escalated_at = _now()
		exc.audit_trail.append(_audit_entry("exception_escalated", escalated_by, {
			"escalate_to": escalate_to,
			"reason": reason,
		}))
		self._tenant_exceptions(tenant_id)[exception_id] = exc
		return exc

	# ------------------------------------------------------------------
	# Tolerance rules
	# ------------------------------------------------------------------

	async def create_tolerance_rule(self, rule: TwMatchToleranceRule) -> TwMatchToleranceRule:
		"""Persist a new tolerance rule."""
		assert rule.tenant_id, "tenant_id required on rule"
		self._tenant_rules(rule.tenant_id)[rule.id] = rule
		return rule

	async def update_tolerance_rule(self, rule: TwMatchToleranceRule) -> TwMatchToleranceRule:
		"""Replace an existing tolerance rule."""
		assert rule.tenant_id, "tenant_id required"
		store = self._tenant_rules(rule.tenant_id)
		if rule.id not in store:
			raise ValueError(f"Tolerance rule {rule.id!r} not found for tenant {rule.tenant_id!r}")
		rule.updated_at = _now()
		store[rule.id] = rule
		return rule

	async def list_tolerance_rules(
		self,
		tenant_id: str,
		active_only: bool = True,
	) -> list[TwMatchToleranceRule]:
		"""Return tolerance rules for a tenant."""
		assert tenant_id, "tenant_id required"
		rules = list(self._tenant_rules(tenant_id).values())
		if active_only:
			now = _now()
			rules = [
				r for r in rules
				if r.active
				and r.effective_from <= now
				and (r.effective_to is None or r.effective_to >= now)
			]
		return sorted(rules, key=lambda r: (_SCOPE_PRIORITY.get(r.scope, 99), r.priority))

	# ------------------------------------------------------------------
	# Auto-approve
	# ------------------------------------------------------------------

	async def auto_approve_within_tolerance(
		self,
		match_id: str,
		tenant_id: str,
		approved_by: str = "system",
	) -> TwMatchResult:
		"""Re-evaluate a prior result and auto-approve if all variances are within tolerance."""
		assert tenant_id, "tenant_id required"
		result = self._tenant_results(tenant_id).get(match_id)
		if result is None:
			raise ValueError(f"Match result {match_id!r} not found for tenant {tenant_id!r}")

		outside = [v for v in result.variances if not v.within_tolerance]
		if outside:
			raise ValueError(
				f"Match {match_id!r} has {len(outside)} variance(s) outside tolerance — cannot auto-approve"
			)

		result.auto_approved = True
		result.audit_trail.append(_audit_entry("auto_approved", approved_by))
		self._tenant_results(tenant_id)[match_id] = result

		await _publish_event("auto_approved", {
			"result_id": match_id,
			"tenant_id": tenant_id,
			"approved_by": approved_by,
		})
		return result

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	async def get_match_statistics(
		self,
		tenant_id: str,
		date_from: datetime,
		date_to: datetime,
	) -> dict[str, Any]:
		"""Compute match-rate, exception-rate, and average resolution time statistics.

		Returns a dict suitable for dashboard widgets and KPI exports.
		"""
		assert tenant_id, "tenant_id required"
		assert date_from <= date_to, "date_from must be <= date_to"

		# Ensure timezone-aware
		if date_from.tzinfo is None:
			date_from = date_from.replace(tzinfo=timezone.utc)
		if date_to.tzinfo is None:
			date_to = date_to.replace(tzinfo=timezone.utc)

		results = [
			r for r in self._tenant_results(tenant_id).values()
			if date_from <= r.matched_at <= date_to
		]
		exceptions = [
			e for e in self._tenant_exceptions(tenant_id).values()
			if date_from <= e.raised_at <= date_to
		]

		total = len(results)
		matched = sum(1 for r in results if r.outcome == TwMatchOutcome.MATCHED)
		partial = sum(1 for r in results if r.outcome == TwMatchOutcome.PARTIAL_MATCH)
		exception_count = sum(1 for r in results if r.outcome == TwMatchOutcome.EXCEPTION)
		auto_approved = sum(1 for r in results if r.auto_approved)

		match_rate = (matched / total * 100) if total else 0.0
		exception_rate = (exception_count / total * 100) if total else 0.0

		# Average resolution time (hours) for resolved exceptions
		resolved = [e for e in exceptions if e.resolved_at is not None]
		avg_resolution_hours: float | None = None
		if resolved:
			total_hours = sum(
				(e.resolved_at - e.raised_at).total_seconds() / 3600.0  # type: ignore[operator]
				for e in resolved
			)
			avg_resolution_hours = total_hours / len(resolved)

		# Variance breakdown
		price_var_total = sum(
			abs(v.percentage_variance)
			for r in results
			for v in r.variances
			if v.variance_type == TwVarianceType.PRICE and v.percentage_variance is not None
		)
		qty_var_total = sum(
			abs(v.percentage_variance)
			for r in results
			for v in r.variances
			if v.variance_type == TwVarianceType.QUANTITY and v.percentage_variance is not None
		)

		# Open exception ageing buckets
		open_excs = [e for e in exceptions if e.status == TwExceptionStatus.OPEN]
		now = _now()
		for e in open_excs:
			e.age_days = (now - e.raised_at).total_seconds() / 86400.0

		age_buckets = {
			"0_3d": sum(1 for e in open_excs if e.age_days <= 3),
			"4_7d": sum(1 for e in open_excs if 3 < e.age_days <= 7),
			"8_30d": sum(1 for e in open_excs if 7 < e.age_days <= 30),
			"over_30d": sum(1 for e in open_excs if e.age_days > 30),
		}

		return {
			"tenant_id": tenant_id,
			"period": {"from": date_from.isoformat(), "to": date_to.isoformat()},
			"totals": {
				"matches_attempted": total,
				"matched": matched,
				"partial_match": partial,
				"exception": exception_count,
				"auto_approved": auto_approved,
			},
			"rates": {
				"match_rate_pct": round(match_rate, 2),
				"exception_rate_pct": round(exception_rate, 2),
				"auto_approval_rate_pct": round((auto_approved / total * 100) if total else 0.0, 2),
			},
			"variances": {
				"avg_price_variance_pct": round(price_var_total / max(total, 1), 4),
				"avg_qty_variance_pct": round(qty_var_total / max(total, 1), 4),
			},
			"exceptions": {
				"total_raised": len(exceptions),
				"open": len(open_excs),
				"resolved": len(resolved),
				"escalated": sum(1 for e in exceptions if e.status == TwExceptionStatus.ESCALATED),
				"avg_resolution_hours": round(avg_resolution_hours, 2) if avg_resolution_hours is not None else None,
				"age_buckets": age_buckets,
			},
		}

	# ------------------------------------------------------------------
	# NATS event subscription handler (called by event bus wiring layer)
	# ------------------------------------------------------------------

	async def on_invoice_received(self, event: dict[str, Any]) -> None:
		"""Handle fin_arc::invoice.received events.

		Extracts the invoice payload, ingests it as a TwMatchDocument, then
		attempts to locate a matching PO and GR to trigger an automatic match.
		The caller is responsible for resolving po_id and gr_id from the event
		payload or from scm_prc / scm_wms adapters.
		"""
		tenant_id: str = event.get("tenant_id", "")
		invoice_payload: dict[str, Any] = event.get("payload", {})
		po_id: str | None = event.get("po_id")
		gr_id: str | None = event.get("gr_id")

		if not tenant_id or not invoice_payload:
			return  # malformed event — discard

		from decimal import Decimal as _D
		from datetime import datetime as _DT

		# Build a minimal TwMatchDocument from the event
		try:
			doc = TwMatchDocument(
				tenant_id=tenant_id,
				document_type=TwDocumentType.VENDOR_INVOICE,
				external_ref=invoice_payload.get("invoice_number", uuid7str()),
				vendor_id=invoice_payload.get("vendor_id", "unknown"),
				vendor_name=invoice_payload.get("vendor_name", ""),
				currency=invoice_payload.get("currency", "KES"),
				document_date=_DT.fromisoformat(invoice_payload["invoice_date"]) if "invoice_date" in invoice_payload else _now(),
				total_amount=_D(str(invoice_payload.get("total_amount", "0"))),
				lines=[],
				raw_payload=invoice_payload,
			)
			await self.ingest_document(doc)
		except Exception:
			return  # log in production

		# If we have all three IDs, trigger the match
		if po_id and gr_id:
			try:
				await self.match_documents(po_id, gr_id, doc.id, tenant_id, initiated_by="fin_arc")
			except Exception:
				pass  # log in production
