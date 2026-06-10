"""SACCO Dividend & Distribution — full async service."""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_div"
DISTRIBUTION_STATUSES = {"pending", "paid", "failed", "reversed"}
PAYMENT_METHODS = {"cash", "mpesa", "bank_transfer", "savings_credit", "cheque"}
WHT_RATE = Decimal("0.05")   # 5% WHT on dividends per KRA rules


class SaccoDividendService:
	"""Async service for SACCO annual surplus calculation, dividend declaration,
	rebate computation, member distributions, and WHT filing."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.financial_years: dict[str, dict[str, Any]] = {}
		self.surplus_allocations: dict[str, dict[str, Any]] = {}
		self.declarations: dict[str, dict[str, Any]] = {}
		self.distributions: dict[str, dict[str, Any]] = {}
		self.wht_records: dict[str, dict[str, Any]] = {}
		self.payment_runs: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record.get("id", ""),
			"record_type": record.get("type", ""),
			"emitted_at": self._now(),
		})

	def _get_year(self, year_id: str, tenant_id: str) -> dict[str, Any]:
		yr = self.financial_years.get(year_id)
		if not yr or yr["tenant_id"] != tenant_id:
			raise KeyError(f"financial_year_not_found: {year_id}")
		return yr

	def _get_declaration(self, declaration_id: str, tenant_id: str) -> dict[str, Any]:
		d = self.declarations.get(declaration_id)
		if not d or d["tenant_id"] != tenant_id:
			raise KeyError(f"declaration_not_found: {declaration_id}")
		return d

	# ── Health & Describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"financial_years": len(self.financial_years),
			"declarations": len(self.declarations),
			"distributions": len(self.distributions),
			"pending_payments": sum(1 for d in self.distributions.values() if d.get("status") == "pending"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "fintech",
			"description": "SACCO annual surplus, dividend declaration, rebate computation, member distributions, WHT",
			"wht_rate_pct": float(WHT_RATE * 100),
			"payment_methods": list(PAYMENT_METHODS),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Financial Years ───────────────────────────────────────────────────────

	async def create_financial_year(
		self,
		year_code: str,
		start_date: str,
		end_date: str,
		tenant_id: str | None = None,
		description: str | None = None,
	) -> dict[str, Any]:
		"""Open a new financial year."""
		t = self._tenant(tenant_id)
		for yr in self.financial_years.values():
			if yr["tenant_id"] == t and yr["year_code"] == year_code:
				raise ValueError(f"financial_year_exists: {year_code}")
		yr_id = self._record_id("fy")
		record: dict[str, Any] = {
			"id": yr_id,
			"type": "sacco_financial_year",
			"tenant_id": t,
			"year_code": year_code,
			"start_date": start_date,
			"end_date": end_date,
			"description": description,
			"total_income": Decimal("0"),
			"total_expenses": Decimal("0"),
			"gross_surplus": Decimal("0"),
			"status": "open",
			"created_at": self._now(),
		}
		self.financial_years[yr_id] = record
		self._emit(t, "financial_year_created", record)
		_log.info("Financial year created: %s tenant=%s", year_code, t)
		return deepcopy(record)

	async def update_financial_year(
		self,
		year_id: str,
		tenant_id: str | None = None,
		description: str | None = None,
		total_income: float | None = None,
		total_expenses: float | None = None,
	) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		yr = self._get_year(year_id, t)
		if yr["status"] != "open":
			raise ValueError(f"cannot_update_closed_year")
		if description is not None:
			yr["description"] = description
		if total_income is not None:
			yr["total_income"] = Decimal(str(total_income))
		if total_expenses is not None:
			yr["total_expenses"] = Decimal(str(total_expenses))
		yr["gross_surplus"] = yr["total_income"] - yr["total_expenses"]
		yr["updated_at"] = self._now()
		self._emit(t, "financial_year_updated", yr)
		return deepcopy(yr)

	async def list_financial_years(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(yr) for yr in self.financial_years.values() if yr["tenant_id"] == t]
		if status:
			items = [yr for yr in items if yr.get("status") == status]
		return items

	async def get_financial_year(self, year_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		return deepcopy(self._get_year(year_id, t))

	async def delete_financial_year(self, year_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel an open financial year (no declarations attached)."""
		t = self._tenant(tenant_id)
		yr = self._get_year(year_id, t)
		attached = [d for d in self.declarations.values() if d["tenant_id"] == t and d["year_id"] == year_id]
		if attached:
			raise ValueError(f"year_has_declarations: {len(attached)}")
		yr["status"] = "cancelled"
		yr["cancelled_at"] = self._now()
		self._emit(t, "financial_year_cancelled", yr)
		return deepcopy(yr)

	# ── Surplus Allocation ────────────────────────────────────────────────────

	async def allocate_surplus(
		self,
		year_id: str,
		total_income: float,
		total_expenses: float,
		statutory_reserve_pct: float,
		education_fund_pct: float,
		dividend_pool_pct: float,
		rebate_pool_pct: float,
		allocation_approved_by: str,
		allocation_date: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute and record surplus allocation for a financial year."""
		t = self._tenant(tenant_id)
		yr = self._get_year(year_id, t)
		if yr["status"] != "open":
			raise ValueError("year_not_open_for_allocation")
		income = Decimal(str(total_income))
		expenses = Decimal(str(total_expenses))
		gross_surplus = income - expenses
		if gross_surplus < 0:
			raise ValueError(f"negative_surplus: {gross_surplus}")
		pct_total = statutory_reserve_pct + education_fund_pct + dividend_pool_pct + rebate_pool_pct
		if pct_total > 100:
			raise ValueError(f"allocations_exceed_100_pct: {pct_total}")
		statutory_reserve = (gross_surplus * Decimal(str(statutory_reserve_pct)) / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
		education_fund = (gross_surplus * Decimal(str(education_fund_pct)) / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
		dividend_pool = (gross_surplus * Decimal(str(dividend_pool_pct)) / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
		rebate_pool = (gross_surplus * Decimal(str(rebate_pool_pct)) / 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
		retained = gross_surplus - statutory_reserve - education_fund - dividend_pool - rebate_pool
		alloc_id = self._record_id("alloc")
		record: dict[str, Any] = {
			"id": alloc_id,
			"type": "sacco_surplus_allocation",
			"tenant_id": t,
			"year_id": year_id,
			"year_code": yr.get("year_code"),
			"total_income": income,
			"total_expenses": expenses,
			"gross_surplus": gross_surplus,
			"statutory_reserve_pct": Decimal(str(statutory_reserve_pct)),
			"statutory_reserve": statutory_reserve,
			"education_fund_pct": Decimal(str(education_fund_pct)),
			"education_fund": education_fund,
			"dividend_pool_pct": Decimal(str(dividend_pool_pct)),
			"dividend_pool": dividend_pool,
			"rebate_pool_pct": Decimal(str(rebate_pool_pct)),
			"rebate_pool": rebate_pool,
			"retained_surplus": retained,
			"allocation_approved_by": allocation_approved_by,
			"allocation_date": allocation_date,
			"status": "approved",
			"created_at": self._now(),
		}
		self.surplus_allocations[alloc_id] = record
		yr["total_income"] = income
		yr["total_expenses"] = expenses
		yr["gross_surplus"] = gross_surplus
		yr["surplus_allocation_id"] = alloc_id
		yr["updated_at"] = self._now()
		self._emit(t, "surplus_allocated", record)
		_log.info("Surplus allocated: year=%s gross_surplus=%s", yr.get("year_code"), gross_surplus)
		return deepcopy(record)

	async def list_surplus_allocations(self, year_id: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.surplus_allocations.values() if a["tenant_id"] == t]
		if year_id:
			items = [a for a in items if a["year_id"] == year_id]
		return items

	# ── Dividend Declaration ──────────────────────────────────────────────────

	async def declare_dividend(
		self,
		year_id: str,
		dividend_rate_pct: float,
		rebate_rate_pct: float,
		declared_by: str,
		board_resolution_ref: str,
		declaration_date: str,
		payment_date: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Board-level dividend and rebate declaration for the year."""
		t = self._tenant(tenant_id)
		yr = self._get_year(year_id, t)
		if yr["status"] not in {"open"}:
			raise ValueError("year_not_open_for_declaration")
		existing = [d for d in self.declarations.values() if d["tenant_id"] == t and d["year_id"] == year_id and d["status"] not in {"reversed"}]
		if existing:
			raise ValueError(f"declaration_already_exists_for_year: {year_id}")
		decl_id = self._record_id("decl")
		record: dict[str, Any] = {
			"id": decl_id,
			"type": "sacco_dividend_declaration",
			"tenant_id": t,
			"year_id": year_id,
			"year_code": yr.get("year_code"),
			"dividend_rate_pct": Decimal(str(dividend_rate_pct)),
			"rebate_rate_pct": Decimal(str(rebate_rate_pct)),
			"declared_by": declared_by,
			"board_resolution_ref": board_resolution_ref,
			"declaration_date": declaration_date,
			"payment_date": payment_date,
			"total_dividend_paid": Decimal("0"),
			"total_rebate_paid": Decimal("0"),
			"total_wht_withheld": Decimal("0"),
			"members_paid": 0,
			"status": "declared",
			"created_at": self._now(),
		}
		self.declarations[decl_id] = record
		self._emit(t, "dividend_declared", record)
		_log.info("Dividend declared: year=%s rate=%s%%", yr.get("year_code"), dividend_rate_pct)
		return deepcopy(record)

	async def update_declaration(
		self,
		declaration_id: str,
		tenant_id: str | None = None,
		dividend_rate_pct: float | None = None,
		rebate_rate_pct: float | None = None,
		payment_date: str | None = None,
	) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		decl = self._get_declaration(declaration_id, t)
		if decl["status"] != "declared":
			raise ValueError(f"cannot_update_declaration_in_status: {decl['status']}")
		if dividend_rate_pct is not None:
			decl["dividend_rate_pct"] = Decimal(str(dividend_rate_pct))
		if rebate_rate_pct is not None:
			decl["rebate_rate_pct"] = Decimal(str(rebate_rate_pct))
		if payment_date is not None:
			decl["payment_date"] = payment_date
		decl["updated_at"] = self._now()
		self._emit(t, "declaration_updated", decl)
		return deepcopy(decl)

	async def list_declarations(self, tenant_id: str | None = None, year_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.declarations.values() if d["tenant_id"] == t]
		if year_id:
			items = [d for d in items if d["year_id"] == year_id]
		return items

	async def get_declaration(self, declaration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		return deepcopy(self._get_declaration(declaration_id, t))

	async def reverse_declaration(self, declaration_id: str, reversed_by: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reverse a declaration that has not yet been paid."""
		t = self._tenant(tenant_id)
		decl = self._get_declaration(declaration_id, t)
		if decl["status"] not in {"declared"}:
			raise ValueError(f"cannot_reverse_declaration_in_status: {decl['status']}")
		paid_count = sum(1 for d in self.distributions.values() if d["tenant_id"] == t and d["declaration_id"] == declaration_id and d["status"] == "paid")
		if paid_count > 0:
			raise ValueError(f"cannot_reverse_partially_paid_declaration: {paid_count}_payments_made")
		decl["status"] = "reversed"
		decl["reversed_by"] = reversed_by
		decl["reversal_reason"] = reason
		decl["reversed_at"] = self._now()
		self._emit(t, "declaration_reversed", decl)
		return deepcopy(decl)

	# ── Member Distributions ──────────────────────────────────────────────────

	async def compute_member_distribution(
		self,
		declaration_id: str,
		member_id: str,
		share_capital: float,
		savings_balance: float,
		payment_method: str,
		tenant_id: str | None = None,
		member_number: str | None = None,
	) -> dict[str, Any]:
		"""Compute dividend and rebate for one member."""
		t = self._tenant(tenant_id)
		decl = self._get_declaration(declaration_id, t)
		if decl["status"] not in {"declared", "processing"}:
			raise ValueError(f"declaration_not_in_payable_status: {decl['status']}")
		if payment_method not in PAYMENT_METHODS:
			raise ValueError(f"invalid_payment_method: {payment_method}")
		share_cap = Decimal(str(share_capital))
		savings_bal = Decimal(str(savings_balance))
		div_rate = decl["dividend_rate_pct"] / 100
		reb_rate = decl["rebate_rate_pct"] / 100
		dividend_gross = (share_cap * div_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
		rebate_gross = (savings_bal * reb_rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
		gross_total = dividend_gross + rebate_gross
		wht = (gross_total * WHT_RATE).quantize(Decimal("0.01"), ROUND_HALF_UP)
		net_payable = gross_total - wht
		dist_id = self._record_id("dist")
		record: dict[str, Any] = {
			"id": dist_id,
			"type": "sacco_member_distribution",
			"tenant_id": t,
			"declaration_id": declaration_id,
			"member_id": member_id,
			"member_number": member_number,
			"share_capital": share_cap,
			"dividend_rate_pct": str(decl["dividend_rate_pct"]),
			"dividend_gross": dividend_gross,
			"savings_balance": savings_bal,
			"rebate_rate_pct": str(decl["rebate_rate_pct"]),
			"rebate_gross": rebate_gross,
			"gross_total": gross_total,
			"wht_rate_pct": str(WHT_RATE * 100),
			"withholding_tax": wht,
			"net_payable": net_payable,
			"payment_method": payment_method,
			"payment_reference": None,
			"status": "pending",
			"created_at": self._now(),
		}
		self.distributions[dist_id] = record
		self._emit(t, "member_distribution_computed", record)
		return deepcopy(record)

	async def bulk_compute_distributions(
		self,
		declaration_id: str,
		members: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-compute distributions for a list of members."""
		t = self._tenant(tenant_id)
		results, errors = [], []
		for m in members:
			try:
				rec = await self.compute_member_distribution(
					declaration_id=declaration_id,
					member_id=m["member_id"],
					share_capital=m["share_capital"],
					savings_balance=m["savings_balance"],
					payment_method=m.get("payment_method", "savings_credit"),
					tenant_id=t,
					member_number=m.get("member_number"),
				)
				results.append(rec)
			except Exception as exc:
				_log.error("bulk_compute error member=%s: %s", m.get("member_id"), exc)
				errors.append({"member": m, "error": str(exc)})
		return {"computed": len(results), "failed": len(errors), "results": results, "errors": errors}

	async def pay_distribution(
		self,
		distribution_id: str,
		payment_reference: str,
		paid_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark a member distribution as paid."""
		t = self._tenant(tenant_id)
		dist = self.distributions.get(distribution_id)
		if not dist or dist["tenant_id"] != t:
			raise KeyError(f"distribution_not_found: {distribution_id}")
		if dist["status"] != "pending":
			raise ValueError(f"cannot_pay_distribution_in_status: {dist['status']}")
		dist["payment_reference"] = payment_reference
		dist["paid_by"] = paid_by
		dist["paid_at"] = self._now()
		dist["status"] = "paid"
		# Update declaration totals
		decl = self.declarations.get(dist["declaration_id"])
		if decl and decl["tenant_id"] == t:
			decl["total_dividend_paid"] += dist["dividend_gross"]
			decl["total_rebate_paid"] += dist["rebate_gross"]
			decl["total_wht_withheld"] += dist["withholding_tax"]
			decl["members_paid"] += 1
		self._emit(t, "distribution_paid", dist)
		return deepcopy(dist)

	async def run_payment_batch(
		self,
		declaration_id: str,
		run_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Process all pending distributions for a declaration in one batch."""
		t = self._tenant(tenant_id)
		decl = self._get_declaration(declaration_id, t)
		pending = [d for d in self.distributions.values() if d["tenant_id"] == t and d["declaration_id"] == declaration_id and d["status"] == "pending"]
		paid_count = 0
		failed_count = 0
		total_paid = Decimal("0")
		run_id = self._record_id("run")
		for dist in pending:
			try:
				ref = f"BATCH-{run_id}-{dist['id'][-6:]}"
				dist["payment_reference"] = ref
				dist["paid_by"] = run_by
				dist["paid_at"] = self._now()
				dist["status"] = "paid"
				decl["total_dividend_paid"] += dist["dividend_gross"]
				decl["total_rebate_paid"] += dist["rebate_gross"]
				decl["total_wht_withheld"] += dist["withholding_tax"]
				decl["members_paid"] += 1
				total_paid += dist["net_payable"]
				paid_count += 1
			except Exception as exc:
				_log.error("payment_batch error dist=%s: %s", dist["id"], exc)
				dist["status"] = "failed"
				dist["failure_reason"] = str(exc)
				failed_count += 1
		if paid_count > 0 and failed_count == 0:
			decl["status"] = "paid"
		run_record: dict[str, Any] = {
			"id": run_id,
			"type": "sacco_payment_run",
			"tenant_id": t,
			"declaration_id": declaration_id,
			"paid_count": paid_count,
			"failed_count": failed_count,
			"total_paid": str(total_paid),
			"run_by": run_by,
			"status": "completed",
			"created_at": self._now(),
		}
		self.payment_runs[run_id] = run_record
		self._emit(t, "payment_run_completed", run_record)
		return run_record

	async def list_distributions(
		self,
		declaration_id: str | None = None,
		member_id: str | None = None,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.distributions.values() if d["tenant_id"] == t]
		if declaration_id:
			items = [d for d in items if d["declaration_id"] == declaration_id]
		if member_id:
			items = [d for d in items if d["member_id"] == member_id]
		if status:
			items = [d for d in items if d["status"] == status]
		return items

	async def get_distribution(self, distribution_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		dist = self.distributions.get(distribution_id)
		if not dist or dist["tenant_id"] != t:
			raise KeyError(f"distribution_not_found: {distribution_id}")
		return deepcopy(dist)

	async def reverse_distribution(self, distribution_id: str, reversed_by: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reverse a paid distribution (e.g. incorrect payment)."""
		t = self._tenant(tenant_id)
		dist = self.distributions.get(distribution_id)
		if not dist or dist["tenant_id"] != t:
			raise KeyError(f"distribution_not_found: {distribution_id}")
		if dist["status"] != "paid":
			raise ValueError(f"can_only_reverse_paid_distributions")
		dist["status"] = "reversed"
		dist["reversed_by"] = reversed_by
		dist["reversal_reason"] = reason
		dist["reversed_at"] = self._now()
		self._emit(t, "distribution_reversed", dist)
		return deepcopy(dist)

	# ── Withholding Tax ───────────────────────────────────────────────────────

	async def generate_wht_return(
		self,
		declaration_id: str,
		filed_by: str,
		tenant_id: str | None = None,
		kra_return_reference: str | None = None,
	) -> dict[str, Any]:
		"""Generate a WHT return summary for KRA filing."""
		t = self._tenant(tenant_id)
		decl = self._get_declaration(declaration_id, t)
		paid_dists = [d for d in self.distributions.values() if d["tenant_id"] == t and d["declaration_id"] == declaration_id and d["status"] == "paid"]
		total_gross_dividends = sum(d["dividend_gross"] for d in paid_dists)
		total_gross_rebates = sum(d["rebate_gross"] for d in paid_dists)
		total_wht = sum(d["withholding_tax"] for d in paid_dists)
		wht_id = self._record_id("wht")
		record: dict[str, Any] = {
			"id": wht_id,
			"type": "sacco_wht_return",
			"tenant_id": t,
			"declaration_id": declaration_id,
			"year_code": decl.get("year_code"),
			"total_gross_dividends": total_gross_dividends,
			"total_gross_rebates": total_gross_rebates,
			"total_gross_payable": total_gross_dividends + total_gross_rebates,
			"total_wht": total_wht,
			"wht_rate_pct": str(WHT_RATE * 100),
			"beneficiary_count": len(paid_dists),
			"kra_return_reference": kra_return_reference,
			"filed_by": filed_by,
			"filed_at": self._now(),
			"status": "filed",
			"created_at": self._now(),
		}
		self.wht_records[wht_id] = record
		self._emit(t, "wht_return_filed", record)
		_log.info("WHT return filed: declaration=%s total_wht=%s", declaration_id, total_wht)
		return deepcopy(record)

	async def list_wht_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(w) for w in self.wht_records.values() if w["tenant_id"] == t]

	# ── Year Closing ──────────────────────────────────────────────────────────

	async def close_financial_year(
		self,
		year_id: str,
		closed_by: str,
		approved_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Close a financial year after all distributions are settled."""
		t = self._tenant(tenant_id)
		yr = self._get_year(year_id, t)
		if yr["status"] != "open":
			raise ValueError(f"year_not_open: {yr['status']}")
		decls = [d for d in self.declarations.values() if d["tenant_id"] == t and d["year_id"] == year_id]
		unpaid = [d for d in decls if d["status"] not in {"paid", "reversed"}]
		if unpaid:
			raise ValueError(f"unpaid_declarations_exist: {len(unpaid)}")
		yr["status"] = "closed"
		yr["closed_by"] = closed_by
		yr["approved_by"] = approved_by
		yr["closed_at"] = self._now()
		self._emit(t, "financial_year_closed", yr)
		_log.info("Financial year closed: %s", yr.get("year_code"))
		return deepcopy(yr)

	# ── Reporting & Export ────────────────────────────────────────────────────

	async def dividend_summary(self, declaration_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Summary stats for a dividend declaration."""
		t = self._tenant(tenant_id)
		decl = self._get_declaration(declaration_id, t)
		dists = [d for d in self.distributions.values() if d["tenant_id"] == t and d["declaration_id"] == declaration_id]
		total_net = sum(d["net_payable"] for d in dists if d["status"] == "paid")
		total_gross = sum(d["gross_total"] for d in dists if d["status"] == "paid")
		return {
			"declaration_id": declaration_id,
			"year_code": decl.get("year_code"),
			"dividend_rate_pct": str(decl.get("dividend_rate_pct")),
			"rebate_rate_pct": str(decl.get("rebate_rate_pct")),
			"total_distributions": len(dists),
			"paid_distributions": sum(1 for d in dists if d["status"] == "paid"),
			"pending_distributions": sum(1 for d in dists if d["status"] == "pending"),
			"total_gross_paid": str(total_gross),
			"total_net_paid": str(total_net),
			"total_wht_withheld": str(decl.get("total_wht_withheld", Decimal("0"))),
			"generated_at": self._now(),
		}

	async def annual_report(self, year_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Comprehensive annual dividend and distribution report."""
		t = self._tenant(tenant_id)
		yr = self._get_year(year_id, t)
		decls = [d for d in self.declarations.values() if d["tenant_id"] == t and d["year_id"] == year_id]
		alloc = next((a for a in self.surplus_allocations.values() if a["tenant_id"] == t and a["year_id"] == year_id), None)
		all_dists = [d for d in self.distributions.values() if d["tenant_id"] == t and any(decl["id"] == d["declaration_id"] for decl in decls)]
		return {
			"year_id": year_id,
			"year_code": yr.get("year_code"),
			"start_date": yr.get("start_date"),
			"end_date": yr.get("end_date"),
			"status": yr.get("status"),
			"financial_performance": {
				"total_income": str(yr.get("total_income", Decimal("0"))),
				"total_expenses": str(yr.get("total_expenses", Decimal("0"))),
				"gross_surplus": str(yr.get("gross_surplus", Decimal("0"))),
			},
			"surplus_allocation": {
				"statutory_reserve": str(alloc["statutory_reserve"]) if alloc else None,
				"education_fund": str(alloc["education_fund"]) if alloc else None,
				"dividend_pool": str(alloc["dividend_pool"]) if alloc else None,
				"rebate_pool": str(alloc["rebate_pool"]) if alloc else None,
			} if alloc else None,
			"declarations": len(decls),
			"total_members_paid": sum(d.get("members_paid", 0) for d in decls),
			"total_dividend_paid": str(sum(d.get("total_dividend_paid", Decimal("0")) for d in decls)),
			"total_rebate_paid": str(sum(d.get("total_rebate_paid", Decimal("0")) for d in decls)),
			"total_wht_withheld": str(sum(d.get("total_wht_withheld", Decimal("0")) for d in decls)),
			"generated_at": self._now(),
		}

	async def export_distributions(self, declaration_id: str, tenant_id: str | None = None, fmt: str = "json") -> dict[str, Any]:
		t = self._tenant(tenant_id)
		assert fmt in {"json", "csv", "excel"}, "fmt must be json|csv|excel"
		count = sum(1 for d in self.distributions.values() if d["tenant_id"] == t and d["declaration_id"] == declaration_id)
		return {
			"tenant_id": t,
			"declaration_id": declaration_id,
			"format": fmt,
			"record_count": count,
			"export_reference": f"div-dist-{declaration_id[-6:]}-{self._now()[:10]}.{fmt}",
			"generated_at": self._now(),
		}

	async def member_dividend_history(self, member_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Full dividend and rebate history for a member across all years."""
		t = self._tenant(tenant_id)
		dists = [deepcopy(d) for d in self.distributions.values() if d["tenant_id"] == t and d["member_id"] == member_id]
		total_received = sum(d["net_payable"] for d in dists if d["status"] == "paid")
		total_wht = sum(d["withholding_tax"] for d in dists if d["status"] == "paid")
		return {
			"member_id": member_id,
			"distributions": dists,
			"total_distributions": len(dists),
			"total_net_received": str(total_received),
			"total_wht_deducted": str(total_wht),
			"generated_at": self._now(),
		}


# Alias
DividendService = SaccoDividendService
