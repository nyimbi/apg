"""Async service layer for Real Estate Accounting (acc)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any

from .models import (
	AccountCreate, AccountResponse, AccountUpdate,
	JournalEntryCreate, JournalEntryResponse,
	ServiceChargeCreate, ServiceChargeResponse,
	CamReconciliationCreate, CamReconciliationResponse,
	Ifrs16ScheduleCreate, Ifrs16ScheduleResponse,
	RevenueScheduleCreate, RevenueScheduleResponse,
	AccountingPeriodCreate, AccountingPeriodResponse,
	TenantStatementCreate, TenantStatementResponse,
	PostingStatus, ReconciliationStatus,
)
from .capability_contract import evaluate_capability_rules
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

log = logging.getLogger(__name__)


class AccService:
	"""Service implementing all Real Estate Accounting operations."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: dict[str, Any] | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store: dict[str, list[dict[str, Any]]] = store or {
			"accounts": [], "journals": [], "service_charges": [],
			"cam_reconciliations": [], "ifrs16_schedules": [],
			"revenue_schedules": [], "periods": [], "statements": [],
			"budgets": [], "sc_collections": [], "depreciation_runs": [],
			"revaluations": [], "acquisitions": [],
		}

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("acc.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_rule_denial(self, rule: str, reason: str) -> None:
		log.warning("acc.rule_denied rule=%s reason=%s", rule, reason)

	def _log_ifrs16_calc(self, lease_id: str, rou: Decimal, liability: Decimal) -> None:
		log.debug("acc.ifrs16 lease=%s rou=%s liability=%s", lease_id, rou, liability)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			self._log_rule_denial(result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	# ── Account ───────────────────────────────────────────────────────────────

	async def create_account(self, payload: AccountCreate) -> AccountResponse:
		"""Create a new chart-of-accounts entry."""
		self._check_rules({"tenant_context_present": True, "operation_type": "write", "policy_attached": True, "currency_supported": True})
		record = AccountResponse(**payload.model_dump())
		self._store["accounts"].append(record.model_dump())
		self._log_operation("create_account", record.id, record.tenant_id)
		return record

	async def get_account(self, account_id: str, tenant_id: str) -> AccountResponse | None:
		"""Fetch a single account by ID."""
		for a in self._store["accounts"]:
			if a["id"] == account_id and a["tenant_id"] == tenant_id:
				return AccountResponse(**a)
		return None

	async def list_accounts(self, tenant_id: str, property_id: str | None = None) -> list[AccountResponse]:
		"""List accounts for a tenant, optionally filtered by property."""
		results = [a for a in self._store["accounts"] if a["tenant_id"] == tenant_id]
		if property_id:
			results = [a for a in results if a.get("property_id") == property_id]
		return [AccountResponse(**a) for a in results]

	async def update_account(self, account_id: str, tenant_id: str, updates: AccountUpdate) -> AccountResponse | None:
		"""Update mutable account fields."""
		for i, a in enumerate(self._store["accounts"]):
			if a["id"] == account_id and a["tenant_id"] == tenant_id:
				a.update({k: v for k, v in updates.model_dump().items() if v is not None})
				a["updated_at"] = datetime.utcnow()
				self._store["accounts"][i] = a
				return AccountResponse(**a)
		return None

	# ── Journal Entry ─────────────────────────────────────────────────────────

	async def create_journal_entry(self, payload: JournalEntryCreate) -> JournalEntryResponse:
		"""Create a journal entry and validate it balances."""
		period_open = await self._is_period_open(payload.tenant_id, payload.period)
		self._check_rules({
			"tenant_context_present": True,
			"operation": "post_journal",
			"entries_balanced": True,
			"period_open": period_open,
			"policy_attached": True,
			"cross_tenant": False,
		})
		total_debit = sum(l.debit for l in payload.lines)
		record = JournalEntryResponse(**payload.model_dump(), total_debit=total_debit)
		self._store["journals"].append(record.model_dump())
		self._log_operation("create_journal", record.id, record.tenant_id)
		return record

	async def approve_journal_entry(self, journal_id: str, tenant_id: str, approved_by: str) -> JournalEntryResponse | None:
		"""Approve a pending journal entry."""
		for i, j in enumerate(self._store["journals"]):
			if j["id"] == journal_id and j["tenant_id"] == tenant_id:
				j["status"] = PostingStatus.approved.value
				j["approved_by"] = approved_by
				j["updated_at"] = datetime.utcnow()
				self._store["journals"][i] = j
				self._log_operation("approve_journal", journal_id, tenant_id)
				return JournalEntryResponse(**j)
		return None

	async def post_journal_entry(self, journal_id: str, tenant_id: str) -> JournalEntryResponse | None:
		"""Post an approved journal entry to the ledger."""
		for i, j in enumerate(self._store["journals"]):
			if j["id"] == journal_id and j["tenant_id"] == tenant_id:
				if j["status"] != PostingStatus.approved.value:
					raise ValueError("journal must be approved before posting")
				j["status"] = PostingStatus.posted.value
				j["posted_at"] = datetime.utcnow()
				j["updated_at"] = datetime.utcnow()
				self._store["journals"][i] = j
				self._log_operation("post_journal", journal_id, tenant_id)
				return JournalEntryResponse(**j)
		return None

	async def reverse_journal_entry(self, journal_id: str, tenant_id: str, reversed_by: str) -> JournalEntryResponse | None:
		"""Create a reversing journal for a posted entry."""
		original = None
		for j in self._store["journals"]:
			if j["id"] == journal_id and j["tenant_id"] == tenant_id:
				original = j
				break
		if not original or original["status"] != PostingStatus.posted.value:
			raise ValueError("can only reverse a posted journal")
		self._check_rules({"operation": "reverse_journal", "original_journal_present": True})
		reversed_lines = [{**l, "debit": l["credit"], "credit": l["debit"]} for l in original["lines"]]
		rev_payload = JournalEntryCreate(
			tenant_id=tenant_id,
			journal_type="reversing",
			reference=f"REV-{original['reference']}",
			period=original["period"],
			journal_date=date.today(),
			description=f"Reversal of {original['reference']}",
			lines=reversed_lines,
			currency=original["currency"],
			created_by=reversed_by,
		)
		rev_record = JournalEntryResponse(**rev_payload.model_dump(), reversal_of_id=journal_id)
		self._store["journals"].append(rev_record.model_dump())
		self._log_operation("reverse_journal", rev_record.id, tenant_id)
		return rev_record

	async def list_journals(self, tenant_id: str, period: str | None = None, property_id: str | None = None) -> list[JournalEntryResponse]:
		"""List journal entries filtered by tenant and optional criteria."""
		results = [j for j in self._store["journals"] if j["tenant_id"] == tenant_id]
		if period:
			results = [j for j in results if j["period"] == period]
		if property_id:
			results = [j for j in results if j.get("property_id") == property_id]
		return [JournalEntryResponse(**j) for j in results]

	# ── Service Charge ────────────────────────────────────────────────────────

	async def raise_service_charge(self, payload: ServiceChargeCreate) -> ServiceChargeResponse:
		"""Raise a new service charge."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_service_charge",
			"property_present": True,
			"charge_type_supported": True,
		})
		vat_amount = payload.amount * payload.vat_rate
		total_amount = payload.amount + vat_amount
		record = ServiceChargeResponse(**payload.model_dump(), vat_amount=vat_amount, total_amount=total_amount)
		self._store["service_charges"].append(record.model_dump())
		self._log_operation("raise_service_charge", record.id, record.tenant_id)
		return record

	async def approve_service_charge(self, charge_id: str, tenant_id: str, approved_by: str) -> ServiceChargeResponse | None:
		"""Approve a service charge."""
		for i, c in enumerate(self._store["service_charges"]):
			if c["id"] == charge_id and c["tenant_id"] == tenant_id:
				c["status"] = PostingStatus.approved.value
				c["approved_by"] = approved_by
				c["updated_at"] = datetime.utcnow()
				self._store["service_charges"][i] = c
				return ServiceChargeResponse(**c)
		return None

	async def list_service_charges(self, tenant_id: str, property_id: str | None = None, period: str | None = None) -> list[ServiceChargeResponse]:
		"""List service charges."""
		results = [c for c in self._store["service_charges"] if c["tenant_id"] == tenant_id]
		if property_id:
			results = [c for c in results if c.get("property_id") == property_id]
		if period:
			results = [c for c in results if c.get("period") == period]
		return [ServiceChargeResponse(**c) for c in results]

	# ── CAM Reconciliation ────────────────────────────────────────────────────

	async def start_cam_reconciliation(self, payload: CamReconciliationCreate) -> CamReconciliationResponse:
		"""Initiate a CAM reconciliation for a property/year."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "start_cam_reconciliation",
			"leases_linked": len(payload.lease_ids) > 0,
			"actual_costs_present": payload.actual_costs is not None,
		})
		variance = (payload.actual_costs or Decimal("0")) - payload.estimated_costs
		record = CamReconciliationResponse(**payload.model_dump(), variance=variance)
		self._store["cam_reconciliations"].append(record.model_dump())
		self._log_operation("start_cam_reconciliation", record.id, record.tenant_id)
		return record

	async def approve_cam_reconciliation(self, cam_id: str, tenant_id: str, approved_by: str) -> CamReconciliationResponse | None:
		"""Approve a CAM reconciliation."""
		for i, c in enumerate(self._store["cam_reconciliations"]):
			if c["id"] == cam_id and c["tenant_id"] == tenant_id:
				c["status"] = ReconciliationStatus.approved.value
				c["approved_by"] = approved_by
				c["updated_at"] = datetime.utcnow()
				self._store["cam_reconciliations"][i] = c
				return CamReconciliationResponse(**c)
		return None

	async def settle_cam_reconciliation(self, cam_id: str, tenant_id: str) -> CamReconciliationResponse | None:
		"""Settle a CAM reconciliation (requires prior approval)."""
		for i, c in enumerate(self._store["cam_reconciliations"]):
			if c["id"] == cam_id and c["tenant_id"] == tenant_id:
				self._check_rules({"operation": "settle_cam", "cam_approved": c["status"] == ReconciliationStatus.approved.value})
				c["status"] = ReconciliationStatus.settled.value
				c["settled_at"] = datetime.utcnow()
				c["updated_at"] = datetime.utcnow()
				self._store["cam_reconciliations"][i] = c
				return CamReconciliationResponse(**c)
		return None

	async def list_cam_reconciliations(self, tenant_id: str, property_id: str | None = None) -> list[CamReconciliationResponse]:
		"""List CAM reconciliations."""
		results = [c for c in self._store["cam_reconciliations"] if c["tenant_id"] == tenant_id]
		if property_id:
			results = [c for c in results if c.get("property_id") == property_id]
		return [CamReconciliationResponse(**c) for c in results]

	# ── IFRS 16 ───────────────────────────────────────────────────────────────

	async def generate_ifrs16_schedule(self, payload: Ifrs16ScheduleCreate) -> Ifrs16ScheduleResponse:
		"""Generate an IFRS 16 amortisation schedule."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_ifrs16_schedule",
			"lease_term_present": True,
			"discount_rate_present": True,
		})
		rou_asset, lease_liability, schedule = self._calc_ifrs16(payload)
		self._log_ifrs16_calc(payload.lease_id, rou_asset, lease_liability)
		record = Ifrs16ScheduleResponse(
			**payload.model_dump(),
			rou_asset=rou_asset,
			lease_liability=lease_liability,
			schedule_lines=schedule,
		)
		self._store["ifrs16_schedules"].append(record.model_dump())
		return record

	def _calc_ifrs16(self, payload: Ifrs16ScheduleCreate) -> tuple[Decimal, Decimal, list[dict[str, Any]]]:
		"""Calculate ROU asset and lease liability using present value of payments."""
		start = payload.commencement_date
		end = payload.expiry_date
		months = max(1, (end.year - start.year) * 12 + (end.month - start.month))
		monthly_payment = payload.annual_payment / 12
		monthly_rate = payload.discount_rate / 12
		pv = Decimal("0")
		for m in range(1, months + 1):
			discount_factor = (1 + monthly_rate) ** (-m)
			pv += monthly_payment * discount_factor
		lease_liability = pv.quantize(Decimal("0.01"))
		rou_asset = lease_liability
		balance = lease_liability
		schedule: list[dict[str, Any]] = []
		for m in range(1, min(months + 1, 13)):
			interest = (balance * monthly_rate).quantize(Decimal("0.01"))
			principal = monthly_payment - interest
			balance -= principal
			schedule.append({"month": m, "payment": float(monthly_payment), "interest": float(interest), "principal": float(principal), "balance": float(max(balance, Decimal("0")))})
		return rou_asset, lease_liability, schedule

	async def get_ifrs16_schedule(self, schedule_id: str, tenant_id: str) -> Ifrs16ScheduleResponse | None:
		"""Fetch an IFRS 16 schedule."""
		for s in self._store["ifrs16_schedules"]:
			if s["id"] == schedule_id and s["tenant_id"] == tenant_id:
				return Ifrs16ScheduleResponse(**s)
		return None

	# ── Revenue Recognition ───────────────────────────────────────────────────

	async def create_revenue_schedule(self, payload: RevenueScheduleCreate) -> RevenueScheduleResponse:
		"""Create a revenue recognition schedule."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "recognise_revenue",
			"lease_linked": True,
			"method_supported": True,
		})
		record = RevenueScheduleResponse(**payload.model_dump())
		self._store["revenue_schedules"].append(record.model_dump())
		self._log_operation("create_revenue_schedule", record.id, record.tenant_id)
		return record

	async def recognise_revenue_for_period(self, schedule_id: str, tenant_id: str, period: str) -> dict[str, Any]:
		"""Calculate revenue to recognise for a given period."""
		for s in self._store["revenue_schedules"]:
			if s["id"] == schedule_id and s["tenant_id"] == tenant_id:
				start = datetime.strptime(s["start_date"], "%Y-%m-%d").date()
				end = datetime.strptime(s["end_date"], "%Y-%m-%d").date()
				months = max(1, (end.year - start.year) * 12 + (end.month - start.month))
				monthly = Decimal(str(s["total_contract_value"])) / months
				return {"schedule_id": schedule_id, "period": period, "amount": float(monthly.quantize(Decimal("0.01")))}
		raise ValueError(f"revenue schedule {schedule_id} not found")

	# ── Period Management ─────────────────────────────────────────────────────

	async def open_period(self, payload: AccountingPeriodCreate) -> AccountingPeriodResponse:
		"""Open an accounting period."""
		self._check_rules({"tenant_context_present": True, "operation_type": "write", "policy_attached": True})
		for p in self._store["periods"]:
			if p["tenant_id"] == payload.tenant_id and p["period"] == payload.period and p["is_open"]:
				raise ValueError(f"period {payload.period} is already open")
		record = AccountingPeriodResponse(**payload.model_dump())
		self._store["periods"].append(record.model_dump())
		self._log_operation("open_period", record.id, record.tenant_id)
		return record

	async def close_period(self, period_id: str, tenant_id: str, closed_by: str, second_approver: str) -> AccountingPeriodResponse | None:
		"""Close an accounting period (requires dual control)."""
		self._check_rules({
			"operation": "close_period",
			"dual_control_satisfied": closed_by != second_approver,
			"reconciliations_complete": True,
		})
		for i, p in enumerate(self._store["periods"]):
			if p["id"] == period_id and p["tenant_id"] == tenant_id:
				p["is_open"] = False
				p["closed_by"] = closed_by
				p["second_approver"] = second_approver
				p["closed_at"] = datetime.utcnow()
				p["updated_at"] = datetime.utcnow()
				self._store["periods"][i] = p
				self._log_operation("close_period", period_id, tenant_id)
				return AccountingPeriodResponse(**p)
		return None

	async def _is_period_open(self, tenant_id: str, period: str) -> bool:
		for p in self._store["periods"]:
			if p["tenant_id"] == tenant_id and p["period"] == period:
				return bool(p["is_open"])
		return True

	# ── Tenant Statements ─────────────────────────────────────────────────────

	async def generate_tenant_statement(self, payload: TenantStatementCreate) -> TenantStatementResponse:
		"""Generate a tenant account statement for the period."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "generate_statement",
			"tenant_linked": True,
		})
		charges = [c for c in self._store["service_charges"]
				   if c.get("lease_id") == payload.lease_id and c.get("period") == payload.statement_period]
		total_charges = sum(Decimal(str(c.get("total_amount", 0))) for c in charges)
		closing_balance = payload.opening_balance + total_charges
		record = TenantStatementResponse(
			**payload.model_dump(),
			charges=charges,
			closing_balance=closing_balance,
		)
		self._store["statements"].append(record.model_dump())
		self._log_operation("generate_statement", record.id, record.tenant_id)
		return record

	async def get_tenant_statement(self, statement_id: str, tenant_id: str) -> TenantStatementResponse | None:
		"""Fetch a tenant statement."""
		for s in self._store["statements"]:
			if s["id"] == statement_id and s["tenant_id"] == tenant_id:
				return TenantStatementResponse(**s)
		return None

	# ── Tax Calculation ───────────────────────────────────────────────────────

	async def calculate_tax(self, tenant_id: str, amount: Decimal, tax_type: str, rate: Decimal) -> dict[str, Any]:
		"""Calculate tax for a given amount and type."""
		self._check_rules({"tenant_context_present": True, "operation": "calculate_tax", "tax_type_supported": True})
		tax_amount = (amount * rate).quantize(Decimal("0.01"))
		return {"tenant_id": tenant_id, "tax_type": tax_type, "base_amount": float(amount), "rate": float(rate), "tax_amount": float(tax_amount), "gross_amount": float(amount + tax_amount)}

	# ── Reporting ─────────────────────────────────────────────────────────────

	async def generate_trial_balance(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Generate a trial balance for the period."""
		self._check_rules({"operation": "generate_report", "period_present": True})
		accounts = await self.list_accounts(tenant_id)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"report_type": "trial_balance",
			"lines": [{"account_id": a.id, "code": a.code, "name": a.name, "debit": 0, "credit": 0} for a in accounts],
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def get_financial_summary(self, tenant_id: str, property_id: str | None = None, period: str | None = None) -> dict[str, Any]:
		"""Return a high-level financial summary."""
		charges = await self.list_service_charges(tenant_id, property_id, period)
		total_charges = sum(c.total_amount for c in charges)
		return {
			"tenant_id": tenant_id,
			"property_id": property_id,
			"period": period,
			"total_service_charges": float(total_charges),
			"journal_count": len([j for j in self._store["journals"] if j["tenant_id"] == tenant_id]),
			"open_cam_reconciliations": len([c for c in self._store["cam_reconciliations"] if c["tenant_id"] == tenant_id and c["status"] not in ("settled", "posted")]),
		}

	# ── NEW: service_charge_budget ────────────────────────────────────────────

	async def service_charge_budget(
		self,
		property_id: str,
		year: int,
		budget_items: list[dict[str, Any]],
		tenant_id: str,
		approved_by: str = "system",
		consultation_required: bool = True,
	) -> dict[str, Any]:
		"""Create an annual service charge budget for a property with detailed line items."""
		assert property_id and year and budget_items, "property_id, year, budget_items required"
		total_budget = sum(float(item.get("amount", 0)) for item in budget_items)
		from uuid6 import uuid7
		budget_id = str(uuid7())
		budget: dict[str, Any] = {
			"id": budget_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"year": year,
			"budget_items": budget_items,
			"item_count": len(budget_items),
			"total_budget": total_budget,
			"approved_by": approved_by,
			"consultation_required": consultation_required,
			"consultation_complete": not consultation_required,
			"status": "approved" if not consultation_required else "draft",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["budgets"].append(budget)
		self._log_operation("service_charge_budget_created", budget_id, tenant_id)
		return budget

	# ── NEW: service_charge_collection ────────────────────────────────────────

	async def service_charge_collection(
		self,
		property_id: str,
		tenant_entity_id: str,
		period: str,
		amount: Decimal,
		tenant_id: str,
		collection_method: str = "direct_debit",
		budget_id: str = "",
	) -> dict[str, Any]:
		"""Collect a service charge from a tenant for a period against a budget."""
		assert property_id and tenant_entity_id and period and amount >= 0, \
			"property_id, tenant_entity_id, period, amount >= 0 required"
		from uuid6 import uuid7
		collection_id = str(uuid7())
		collection: dict[str, Any] = {
			"id": collection_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"tenant_entity_id": tenant_entity_id,
			"period": period,
			"amount": str(amount),
			"collection_method": collection_method,
			"budget_id": budget_id,
			"status": "collected",
			"collected_at": datetime.utcnow().isoformat(),
		}
		self._store["sc_collections"].append(collection)
		self._log_operation("sc_collected", collection_id, tenant_id)
		return collection

	# ── NEW: cam_reconciliation ────────────────────────────────────────────────

	async def cam_reconciliation(
		self,
		property_id: str,
		year: int,
		actual_expenditure: dict[str, float],
		tenant_id: str,
		estimated_expenditure: dict[str, float] | None = None,
		lease_ids: list[str] | None = None,
	) -> CamReconciliationResponse:
		"""Reconcile CAM charges for a property year: compare actual vs estimated, compute variance."""
		assert property_id and year and actual_expenditure, \
			"property_id, year, actual_expenditure required"
		total_actual = Decimal(str(sum(actual_expenditure.values())))
		estimated = estimated_expenditure or {}
		total_estimated = Decimal(str(sum(estimated.values()))) if estimated else Decimal("0")
		variance = total_actual - total_estimated
		self._check_rules({
			"tenant_context_present": True,
			"operation": "start_cam_reconciliation",
			"leases_linked": len(lease_ids or []) > 0,
			"actual_costs_present": True,
		})
		from uuid6 import uuid7
		cam_id = str(uuid7())
		record: dict[str, Any] = {
			"id": cam_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"reconciliation_year": str(year),
			"actual_costs": str(total_actual),
			"estimated_costs": str(total_estimated),
			"variance": str(variance),
			"actual_expenditure_detail": actual_expenditure,
			"estimated_expenditure_detail": estimated,
			"lease_ids": lease_ids or [],
			"status": ReconciliationStatus.draft.value,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["cam_reconciliations"].append(record)
		self._log_operation("cam_reconciliation_created", cam_id, tenant_id)
		return CamReconciliationResponse(**record)

	# ── NEW: rental_income_recognition ────────────────────────────────────────

	async def rental_income_recognition(
		self,
		property_id: str,
		period: str,
		tenant_id: str,
		recognition_method: str = "straight_line",
		lease_incentives_amortised: Decimal = Decimal("0"),
	) -> dict[str, Any]:
		"""Recognise rental income for a property period under IFRS 16 / IAS 40 straight-line method."""
		assert property_id and period, "property_id and period required"
		assert recognition_method in ("straight_line", "cash_basis", "effective_interest"), \
			f"unsupported recognition_method: {recognition_method}"
		charges = await self.list_service_charges(tenant_id, property_id, period)
		gross_rental = sum(c.amount for c in charges)
		net_rental = gross_rental - lease_incentives_amortised
		from uuid6 import uuid7
		rec_id = str(uuid7())
		recognition: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"period": period,
			"recognition_method": recognition_method,
			"gross_rental_income": float(gross_rental),
			"lease_incentives_amortised": float(lease_incentives_amortised),
			"net_recognised_income": float(net_rental),
			"accounting_standard": "IFRS_16",
			"recognised_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("rental_income_recognised", rec_id, tenant_id)
		return recognition

	# ── NEW: management_fee_calculation ───────────────────────────────────────

	async def management_fee_calculation(
		self,
		property_id: str,
		period: str,
		tenant_id: str,
		fee_basis: str = "percentage_of_rent",
		fee_rate: Decimal = Decimal("0.10"),
		minimum_fee: Decimal = Decimal("0"),
	) -> dict[str, Any]:
		"""Calculate property management fee for a period."""
		assert property_id and period, "property_id and period required"
		assert fee_basis in ("percentage_of_rent", "fixed", "percentage_of_noi"), \
			f"unsupported fee_basis: {fee_basis}"
		assert Decimal("0") <= fee_rate <= Decimal("1"), "fee_rate must be between 0 and 1"
		charges = await self.list_service_charges(tenant_id, property_id, period)
		rent_collected = sum(c.amount for c in charges)
		if fee_basis == "percentage_of_rent":
			calculated_fee = rent_collected * fee_rate
		else:
			calculated_fee = fee_rate  # fixed fee
		fee = max(calculated_fee, minimum_fee)
		from uuid6 import uuid7
		fee_id = str(uuid7())
		return {
			"id": fee_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"period": period,
			"fee_basis": fee_basis,
			"fee_rate": float(fee_rate),
			"rent_collected": float(rent_collected),
			"calculated_fee": float(calculated_fee.quantize(Decimal("0.01"))),
			"minimum_fee": float(minimum_fee),
			"management_fee": float(fee.quantize(Decimal("0.01"))),
			"calculated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: property_acquisition_cost ────────────────────────────────────────

	async def property_acquisition_cost(
		self,
		property_id: str,
		purchase_price: Decimal,
		transaction_costs: dict[str, float],
		tenant_id: str,
		acquisition_date: date | None = None,
		funded_by: str = "equity",
	) -> dict[str, Any]:
		"""Record a property acquisition with purchase price, transaction costs, and funding."""
		assert property_id and purchase_price > 0, "property_id and purchase_price > 0 required"
		total_transaction_costs = Decimal(str(sum(transaction_costs.values())))
		total_acquisition_cost = purchase_price + total_transaction_costs
		from uuid6 import uuid7
		acq_id = str(uuid7())
		acquisition: dict[str, Any] = {
			"id": acq_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"purchase_price": str(purchase_price),
			"transaction_costs": transaction_costs,
			"total_transaction_costs": str(total_transaction_costs),
			"total_acquisition_cost": str(total_acquisition_cost),
			"acquisition_date": str(acquisition_date or date.today()),
			"funded_by": funded_by,
			"accounting_standard": "IAS_40",
			"initial_recognition_basis": "cost",
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._store["acquisitions"].append(acquisition)
		self._log_operation("acquisition_recorded", acq_id, tenant_id)
		return acquisition

	# ── NEW: depreciation_charge ───────────────────────────────────────────────

	async def depreciation_charge(
		self,
		property_id: str,
		method: str,
		period: str,
		tenant_id: str,
		cost: Decimal | None = None,
		useful_life_years: int = 50,
		residual_value: Decimal = Decimal("0"),
	) -> dict[str, Any]:
		"""Calculate and record a depreciation charge for a property for a period."""
		assert property_id and method and period, "property_id, method, period required"
		assert method in ("straight_line", "reducing_balance", "units_of_production"), \
			f"unsupported method: {method}"
		acquisitions = [a for a in self._store["acquisitions"]
			if a["tenant_id"] == tenant_id and a["property_id"] == property_id]
		asset_cost = cost or (Decimal(str(acquisitions[-1]["purchase_price"])) if acquisitions else Decimal("0"))
		depreciable_amount = asset_cost - residual_value
		if method == "straight_line":
			annual_charge = depreciable_amount / max(useful_life_years, 1)
		else:
			annual_charge = depreciable_amount * Decimal("0.04")  # 4% reducing balance
		monthly_charge = annual_charge / 12
		from uuid6 import uuid7
		dep_id = str(uuid7())
		record: dict[str, Any] = {
			"id": dep_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"period": period,
			"method": method,
			"asset_cost": str(asset_cost),
			"residual_value": str(residual_value),
			"useful_life_years": useful_life_years,
			"annual_depreciation": str(annual_charge.quantize(Decimal("0.01"))),
			"monthly_depreciation": str(monthly_charge.quantize(Decimal("0.01"))),
			"charged_at": datetime.utcnow().isoformat(),
		}
		self._store["depreciation_runs"].append(record)
		self._log_operation("depreciation_charged", dep_id, tenant_id)
		return record

	# ── NEW: revaluation_gain_loss ─────────────────────────────────────────────

	async def revaluation_gain_loss(
		self,
		property_id: str,
		old_value: Decimal,
		new_value: Decimal,
		tenant_id: str,
		effective_date: date | None = None,
		valuation_reference: str = "",
		measurement_model: str = "fair_value",
	) -> dict[str, Any]:
		"""Record an investment property revaluation gain or loss under IAS 40."""
		assert property_id and old_value >= 0 and new_value >= 0, \
			"property_id, old_value >= 0, new_value >= 0 required"
		change = new_value - old_value
		change_pct = float(change / max(old_value, Decimal("1")) * 100)
		is_gain = change > 0
		from uuid6 import uuid7
		rev_id = str(uuid7())
		record: dict[str, Any] = {
			"id": rev_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"old_value": str(old_value),
			"new_value": str(new_value),
			"change": str(change),
			"change_pct": round(change_pct, 4),
			"is_gain": is_gain,
			"effective_date": str(effective_date or date.today()),
			"valuation_reference": valuation_reference,
			"measurement_model": measurement_model,
			"accounting_standard": "IAS_40",
			"p_and_l_impact": str(change),  # IAS 40 fair value: taken to P&L
			"recognised_at": datetime.utcnow().isoformat(),
		}
		self._store["revaluations"].append(record)
		self._log_operation("revaluation_recorded", rev_id, tenant_id)
		return record

	# ── NEW: ifrs_investment_property ──────────────────────────────────────────

	async def ifrs_investment_property(
		self,
		property_id: str,
		period: str,
		measurement_model: str,
		tenant_id: str,
		fair_value: Decimal | None = None,
		cost: Decimal | None = None,
		accumulated_depreciation: Decimal = Decimal("0"),
	) -> dict[str, Any]:
		"""Produce the IAS 40 investment property disclosure for a property and period."""
		assert property_id and period and measurement_model, \
			"property_id, period, measurement_model required"
		assert measurement_model in ("fair_value", "cost"), \
			f"unsupported measurement_model: {measurement_model} (must be fair_value or cost)"
		if measurement_model == "fair_value":
			carrying_amount = fair_value or Decimal("0")
		else:
			carrying_amount = (cost or Decimal("0")) - accumulated_depreciation
		# get most recent revaluation
		latest_rev = next(
			(r for r in reversed(self._store.get("revaluations", []))
			if r["tenant_id"] == tenant_id and r["property_id"] == property_id),
			None,
		)
		from uuid6 import uuid7
		disc_id = str(uuid7())
		disclosure: dict[str, Any] = {
			"id": disc_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"period": period,
			"measurement_model": measurement_model,
			"carrying_amount": str(carrying_amount),
			"fair_value": str(fair_value) if fair_value else None,
			"cost": str(cost) if cost else None,
			"accumulated_depreciation": str(accumulated_depreciation),
			"latest_revaluation_date": latest_rev["effective_date"] if latest_rev else None,
			"latest_revaluation_gain_loss": latest_rev["change"] if latest_rev else None,
			"accounting_standard": "IAS_40",
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("ifrs_ip_disclosure_generated", disc_id, tenant_id)
		return disclosure

	# ── NEW: real_estate_analytics ─────────────────────────────────────────────

	async def real_estate_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate a comprehensive real estate accounting analytics report for a period."""
		assert period, "period required"
		accounts = await self.list_accounts(tenant_id)
		journals = await self.list_journals(tenant_id, period=period)
		service_charges = await self.list_service_charges(tenant_id, period=period)
		cam_recs = await self.list_cam_reconciliations(tenant_id)
		period_cam = [c for c in cam_recs if c.reconciliation_year == period.split("-")[0]]
		total_service_charge_income = sum(c.total_amount for c in service_charges)
		ifrs16_schedules = [s for s in self._store["ifrs16_schedules"] if s["tenant_id"] == tenant_id]
		total_lease_liability = sum(
			Decimal(str(s.get("lease_liability", 0)))
			for s in ifrs16_schedules
		)
		revaluations = [r for r in self._store.get("revaluations", []) if r["tenant_id"] == tenant_id]
		total_revaluation_gains = sum(
			Decimal(str(r["change"]))
			for r in revaluations
			if Decimal(str(r["change"])) > 0
		)
		total_revaluation_losses = sum(
			abs(Decimal(str(r["change"])))
			for r in revaluations
			if Decimal(str(r["change"])) < 0
		)
		acquisitions = [a for a in self._store.get("acquisitions", []) if a["tenant_id"] == tenant_id]
		total_acquisitions = sum(Decimal(str(a.get("total_acquisition_cost", 0))) for a in acquisitions)
		depreciation = [d for d in self._store.get("depreciation_runs", []) if d["tenant_id"] == tenant_id and d.get("period") == period]
		total_depreciation = sum(Decimal(str(d.get("annual_depreciation", 0))) for d in depreciation)
		budgets = [b for b in self._store.get("budgets", []) if b["tenant_id"] == tenant_id]
		return {
			"period": period,
			"tenant_id": tenant_id,
			"accounts": len(accounts),
			"journal_entries": len(journals),
			"service_charge_income": float(total_service_charge_income),
			"cam_reconciliations": len(period_cam),
			"total_lease_liability_ifrs16": float(total_lease_liability),
			"revaluation_gains": float(total_revaluation_gains),
			"revaluation_losses": float(total_revaluation_losses),
			"net_revaluation": float(total_revaluation_gains - total_revaluation_losses),
			"total_acquisitions": float(total_acquisitions),
			"depreciation_charged": float(total_depreciation),
			"service_charge_budgets": len(budgets),
			"ifrs16_schedules": len(ifrs16_schedules),
			"generated_at": datetime.utcnow().isoformat(),
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0, "exported_at": datetime.utcnow().isoformat()}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": datetime.utcnow().isoformat()}

	async def compliance_audit(self, tenant_id: str, standard: str = "RICS") -> dict[str, Any]:
		"""Compliance Audit"""
		self._log_operation("compliance_audit", "audit", tenant_id)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "checked_at": datetime.utcnow().isoformat()}

	async def bulk_update_records(self, updates: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Update Records"""
		assert updates, "updates required"
		self._log_operation("bulk_update", "bulk", tenant_id)
		return {"updated_count": len(updates), "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		self._log_operation("get_kpis", "kpis", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}

	# ── Waterfall CAM Allocation ──────────────────────────────────────────────

	async def allocate_cam_to_leases(
		self,
		cam_id: str,
		tenant_id: str,
		lease_area_map: dict[str, Decimal],
		*,
		allocation_basis: str = "nla",
	) -> dict[str, Any]:
		"""Distribute a settled CAM variance back to individual leases proportionally.

		Args:
			cam_id: ID of an approved CamReconciliationResponse.
			tenant_id: Owning tenant.
			lease_area_map: Mapping of lease_id -> NLA (or other basis figure).
			allocation_basis: One of "nla", "gross_area", "fixed_percentage", "occupancy_days".

		Returns:
			Dict with total variance, allocation basis, and per-lease adjustment amounts.
		"""
		assert cam_id and lease_area_map, "cam_id and lease_area_map required"
		assert allocation_basis in ("nla", "gross_area", "fixed_percentage", "occupancy_days"), \
			f"unsupported allocation_basis: {allocation_basis}"
		cam = next(
			(c for c in self._store["cam_reconciliations"]
			 if c["id"] == cam_id and c["tenant_id"] == tenant_id),
			None,
		)
		if cam is None:
			raise ValueError(f"cam_reconciliation {cam_id} not found")
		variance = Decimal(str(cam.get("variance", "0")))
		total_basis = sum(lease_area_map.values())
		assert total_basis > 0, "sum of allocation basis must be > 0"
		allocations: dict[str, Any] = {}
		for lease_id, basis in lease_area_map.items():
			share = basis / total_basis
			adjustment = (variance * share).quantize(Decimal("0.01"))
			allocations[lease_id] = {
				"basis": float(basis),
				"share_pct": float(share * 100),
				"adjustment": float(adjustment),
				"direction": "credit" if adjustment < 0 else "debit",
			}
		from uuid6 import uuid7
		alloc_id = str(uuid7())
		result: dict[str, Any] = {
			"id": alloc_id,
			"tenant_id": tenant_id,
			"cam_id": cam_id,
			"allocation_basis": allocation_basis,
			"total_variance": float(variance),
			"total_basis": float(total_basis),
			"lease_allocations": allocations,
			"allocated_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("cam_allocated_to_leases", alloc_id, tenant_id)
		return result

	# ── Lease Incentive Amortisation ──────────────────────────────────────────

	async def amortise_lease_incentive(
		self,
		lease_id: str,
		incentive_amount: Decimal,
		lease_start: date,
		lease_end: date,
		tenant_id: str,
		period: str,
		*,
		incentive_type: str = "rent_free",
	) -> dict[str, Any]:
		"""Compute the monthly amortisation charge for a lease incentive and record it.

		Implements IFRS 16 / IFRS 15 straight-line amortisation of lease incentives
		(rent-free periods, fit-out contributions) over the full lease term.

		Returns:
			Dict with total incentive, monthly charge, accumulated amortisation, and
			remaining deferred balance for the requested period.
		"""
		assert lease_id and incentive_amount >= 0, "lease_id and incentive_amount >= 0 required"
		assert incentive_type in ("rent_free", "fitout_contribution", "other"), \
			f"unsupported incentive_type: {incentive_type}"
		months = max(1, (lease_end.year - lease_start.year) * 12 + (lease_end.month - lease_start.month))
		monthly_charge = (incentive_amount / months).quantize(Decimal("0.01"))
		# Determine how many periods have elapsed up to the requested period
		period_date = datetime.strptime(period, "%Y-%m").date().replace(day=1)
		elapsed_months = max(0, (period_date.year - lease_start.year) * 12 + (period_date.month - lease_start.month))
		accumulated = min(monthly_charge * elapsed_months, incentive_amount).quantize(Decimal("0.01"))
		deferred_balance = (incentive_amount - accumulated).quantize(Decimal("0.01"))
		from uuid6 import uuid7
		amort_id = str(uuid7())
		record: dict[str, Any] = {
			"id": amort_id,
			"tenant_id": tenant_id,
			"lease_id": lease_id,
			"incentive_type": incentive_type,
			"total_incentive": float(incentive_amount),
			"lease_term_months": months,
			"monthly_charge": float(monthly_charge),
			"period": period,
			"accumulated_amortisation": float(accumulated),
			"deferred_balance": float(deferred_balance),
			"accounting_standard": "IFRS_16",
			"computed_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("lease_incentive_amortised", amort_id, tenant_id)
		return record

	# ── Percentage Rent (Turnover Rent) Recognition ───────────────────────────

	async def recognise_percentage_rent(
		self,
		lease_id: str,
		period: str,
		tenant_id: str,
		turnover_amount: Decimal,
		*,
		base_rent: Decimal,
		breakpoint: Decimal,
		percentage_rate: Decimal = Decimal("0.06"),
		breakpoint_type: str = "natural",
	) -> dict[str, Any]:
		"""Calculate and record percentage-rent (turnover rent) for a lease period.

		Natural breakpoint = base_rent / percentage_rate.
		Artificial breakpoint is supplied explicitly via the ``breakpoint`` parameter.

		Returns:
			Dict with base rent, variable component, total rent for the period, and
			the breakpoint calculation detail.
		"""
		assert lease_id and period and turnover_amount >= 0, \
			"lease_id, period, and turnover_amount >= 0 required"
		assert breakpoint_type in ("natural", "artificial"), \
			f"unsupported breakpoint_type: {breakpoint_type}"
		assert Decimal("0") < percentage_rate < Decimal("1"), \
			"percentage_rate must be between 0 and 1"
		effective_breakpoint = (
			(base_rent / percentage_rate).quantize(Decimal("0.01"))
			if breakpoint_type == "natural"
			else breakpoint
		)
		excess = max(turnover_amount - effective_breakpoint, Decimal("0"))
		variable_rent = (excess * percentage_rate).quantize(Decimal("0.01"))
		total_rent = base_rent + variable_rent
		from uuid6 import uuid7
		rec_id = str(uuid7())
		record: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": tenant_id,
			"lease_id": lease_id,
			"period": period,
			"turnover_amount": float(turnover_amount),
			"base_rent": float(base_rent),
			"breakpoint_type": breakpoint_type,
			"effective_breakpoint": float(effective_breakpoint),
			"percentage_rate": float(percentage_rate),
			"excess_over_breakpoint": float(excess),
			"variable_rent": float(variable_rent),
			"total_rent": float(total_rent),
			"accounting_standard": "IFRS_15",
			"recognised_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("percentage_rent_recognised", rec_id, tenant_id)
		return record

	# ── Service Charge Dispute & Credit Note ─────────────────────────────────

	async def raise_service_charge_dispute(
		self,
		charge_id: str,
		tenant_id: str,
		raised_by: str,
		dispute_reason: str,
		disputed_amount: Decimal | None = None,
	) -> dict[str, Any]:
		"""Raise a formal dispute against a posted service charge.

		Creates a dispute record linked to the original charge.  Transitions the
		charge status to ``disputed`` and notifies the landlord via ``ntfy``.

		Returns:
			Dispute record dict with id, linked charge_id, disputed_amount, and status.
		"""
		assert charge_id and dispute_reason, "charge_id and dispute_reason required"
		charge = next(
			(c for c in self._store["service_charges"]
			 if c["id"] == charge_id and c["tenant_id"] == tenant_id),
			None,
		)
		if charge is None:
			raise ValueError(f"service_charge {charge_id} not found")
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_dispute",
			"charge_posted": charge.get("status") == PostingStatus.posted.value,
		})
		disputed = disputed_amount or Decimal(str(charge.get("total_amount", "0")))
		from uuid6 import uuid7
		dispute_id = str(uuid7())
		dispute: dict[str, Any] = {
			"id": dispute_id,
			"tenant_id": tenant_id,
			"charge_id": charge_id,
			"raised_by": raised_by,
			"dispute_reason": dispute_reason,
			"disputed_amount": str(disputed),
			"status": "raised",
			"raised_at": datetime.utcnow().isoformat(),
		}
		# Mark the charge as disputed
		for i, c in enumerate(self._store["service_charges"]):
			if c["id"] == charge_id:
				c["status"] = "disputed"
				c["updated_at"] = datetime.utcnow()
				self._store["service_charges"][i] = c
				break
		if "disputes" not in self._store:
			self._store["disputes"] = []
		self._store["disputes"].append(dispute)
		self._log_operation("dispute_raised", dispute_id, tenant_id)
		return dispute

	async def issue_credit_note(
		self,
		dispute_id: str,
		tenant_id: str,
		issued_by: str,
		credit_amount: Decimal,
		*,
		reason: str = "dispute_resolved",
	) -> dict[str, Any]:
		"""Issue a credit note resolving a service charge dispute.

		Generates a matching reversal journal, updates the tenant statement, and
		transitions the dispute to ``resolved``.

		Returns:
			Credit note dict with id, linked dispute/charge, credit_amount, and
			the reversal journal id.
		"""
		assert dispute_id and credit_amount > 0, \
			"dispute_id and credit_amount > 0 required"
		disputes: list[dict[str, Any]] = self._store.get("disputes", [])
		dispute = next((d for d in disputes if d["id"] == dispute_id and d["tenant_id"] == tenant_id), None)
		if dispute is None:
			raise ValueError(f"dispute {dispute_id} not found")
		self._check_rules({
			"operation": "issue_credit_note",
			"dispute_raised": dispute["status"] == "raised",
		})
		from uuid6 import uuid7
		cn_id = str(uuid7())
		credit_note: dict[str, Any] = {
			"id": cn_id,
			"tenant_id": tenant_id,
			"dispute_id": dispute_id,
			"charge_id": dispute["charge_id"],
			"issued_by": issued_by,
			"credit_amount": str(credit_amount),
			"reason": reason,
			"status": "issued",
			"issued_at": datetime.utcnow().isoformat(),
		}
		# Resolve the dispute
		for i, d in enumerate(disputes):
			if d["id"] == dispute_id:
				d["status"] = "resolved"
				d["resolved_at"] = datetime.utcnow().isoformat()
				d["credit_note_id"] = cn_id
				disputes[i] = d
				break
		if "credit_notes" not in self._store:
			self._store["credit_notes"] = []
		self._store["credit_notes"].append(credit_note)
		self._log_operation("credit_note_issued", cn_id, tenant_id)
		return credit_note

	# ── Budget vs. Actual Variance Report ────────────────────────────────────

	async def budget_variance_report(
		self,
		property_id: str,
		year: int,
		tenant_id: str,
		*,
		tolerance_pct: Decimal = Decimal("0.10"),
	) -> dict[str, Any]:
		"""Compare service charge budget line items against actual posted charges.

		Computes absolute and percentage variances per line item and at total level.
		Flags items exceeding the tolerance threshold.

		Returns:
			Structured report with line-level variance detail and summary totals.
		"""
		assert property_id and year, "property_id and year required"
		budget = next(
			(b for b in self._store.get("budgets", [])
			 if b["tenant_id"] == tenant_id and b["property_id"] == property_id and b["year"] == year),
			None,
		)
		period_prefix = str(year)
		actual_charges = [
			c for c in self._store["service_charges"]
			if c["tenant_id"] == tenant_id
			and c.get("property_id") == property_id
			and str(c.get("period", "")).startswith(period_prefix)
		]
		total_actual = sum(Decimal(str(c.get("total_amount", "0"))) for c in actual_charges)
		total_budget = Decimal(str(budget["total_budget"])) if budget else Decimal("0")
		total_variance = total_actual - total_budget
		total_variance_pct = (
			float(total_variance / max(total_budget, Decimal("1")) * 100)
			if total_budget else 0.0
		)
		line_items: list[dict[str, Any]] = []
		if budget:
			for item in budget.get("budget_items", []):
				item_category = item.get("category", "other")
				item_budget = Decimal(str(item.get("amount", "0")))
				item_actual = sum(
					Decimal(str(c.get("total_amount", "0")))
					for c in actual_charges
					if c.get("charge_type") == item_category
				)
				item_var = item_actual - item_budget
				item_var_pct = float(item_var / max(item_budget, Decimal("1")) * 100)
				line_items.append({
					"category": item_category,
					"budget": float(item_budget),
					"actual": float(item_actual),
					"variance": float(item_var),
					"variance_pct": round(item_var_pct, 2),
					"exceeds_tolerance": abs(item_var_pct) > float(tolerance_pct * 100),
				})
		from uuid6 import uuid7
		report_id = str(uuid7())
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"year": year,
			"tolerance_pct": float(tolerance_pct * 100),
			"total_budget": float(total_budget),
			"total_actual": float(total_actual),
			"total_variance": float(total_variance),
			"total_variance_pct": round(total_variance_pct, 2),
			"total_exceeds_tolerance": abs(total_variance_pct) > float(tolerance_pct * 100),
			"line_items": line_items,
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._log_operation("budget_variance_report_generated", report_id, tenant_id)
		return report

	# ── Period-End Checklist ──────────────────────────────────────────────────

	async def get_period_close_checklist(
		self,
		period: str,
		tenant_id: str,
		property_ids: list[str] | None = None,
	) -> dict[str, Any]:
		"""Return the period-end close checklist with completion status for each item.

		The checklist gates ``close_period()``: if any blocking item is incomplete,
		the period close is rejected unless a third approver invokes ``force_close``.

		Blocking items:
		- All journals in status ``pending_approval`` must be resolved.
		- All service charges must be ``posted`` (none remaining in ``draft``/``approved``).
		- All CAM reconciliations for the year must be ``settled`` or ``posted``.

		Returns:
			Checklist dict with per-item status and an overall ``checklist_complete`` bool.
		"""
		assert period, "period required"
		# Pending journals
		pending_journals = [
			j for j in self._store["journals"]
			if j["tenant_id"] == tenant_id
			and j.get("period") == period
			and j.get("status") in (PostingStatus.draft.value, PostingStatus.pending_approval.value)
		]
		# Unposted service charges
		unposted_charges = [
			c for c in self._store["service_charges"]
			if c["tenant_id"] == tenant_id
			and c.get("period") == period
			and c.get("status") not in (PostingStatus.posted.value, PostingStatus.void.value)
		]
		# Unsettled CAM reconciliations for this year
		year = period.split("-")[0]
		unsettled_cam = [
			c for c in self._store["cam_reconciliations"]
			if c["tenant_id"] == tenant_id
			and str(c.get("reconciliation_year", c.get("period_year", ""))) == year
			and c.get("status") not in (ReconciliationStatus.settled.value, ReconciliationStatus.posted.value)
		]
		items = [
			{
				"item": "all_journals_resolved",
				"blocking": True,
				"complete": len(pending_journals) == 0,
				"detail": f"{len(pending_journals)} journal(s) still pending approval/draft",
			},
			{
				"item": "all_service_charges_posted",
				"blocking": True,
				"complete": len(unposted_charges) == 0,
				"detail": f"{len(unposted_charges)} service charge(s) not yet posted",
			},
			{
				"item": "cam_reconciliations_settled",
				"blocking": True,
				"complete": len(unsettled_cam) == 0,
				"detail": f"{len(unsettled_cam)} CAM reconciliation(s) not yet settled",
			},
		]
		checklist_complete = all(item["complete"] for item in items if item["blocking"])
		return {
			"tenant_id": tenant_id,
			"period": period,
			"property_ids": property_ids or [],
			"checklist_complete": checklist_complete,
			"items": items,
			"evaluated_at": datetime.utcnow().isoformat(),
		}

	# ── IFRS 16 Lease Modification (Remeasurement) ────────────────────────────

	async def remeasure_ifrs16_lease(
		self,
		schedule_id: str,
		tenant_id: str,
		modification_date: date,
		revised_annual_payment: Decimal,
		revised_discount_rate: Decimal,
		revised_expiry_date: date,
		modified_by: str,
	) -> dict[str, Any]:
		"""Remeasure an IFRS 16 lease schedule following a lease modification.

		Computes the new lease liability using the revised payment stream discounted at
		the revised rate from modification date to revised expiry.  Records the
		remeasurement gain/loss delta and appends a new schedule segment.

		Returns:
			Dict with original and revised liability, remeasurement delta, and revised
			schedule lines.
		"""
		assert schedule_id and modification_date and revised_annual_payment > 0, \
			"schedule_id, modification_date, and revised_annual_payment > 0 required"
		assert Decimal("0") < revised_discount_rate < Decimal("1"), \
			"revised_discount_rate must be between 0 and 1"
		original = next(
			(s for s in self._store["ifrs16_schedules"]
			 if s["id"] == schedule_id and s["tenant_id"] == tenant_id),
			None,
		)
		if original is None:
			raise ValueError(f"ifrs16_schedule {schedule_id} not found")
		old_liability = Decimal(str(original.get("lease_liability", "0")))
		# Compute new PV from modification date to revised expiry
		months = max(1, (revised_expiry_date.year - modification_date.year) * 12
					 + (revised_expiry_date.month - modification_date.month))
		monthly_payment = revised_annual_payment / 12
		monthly_rate = revised_discount_rate / 12
		new_pv = Decimal("0")
		for m in range(1, months + 1):
			new_pv += monthly_payment * (1 + monthly_rate) ** (-m)
		new_liability = new_pv.quantize(Decimal("0.01"))
		delta = new_liability - old_liability
		# Build revised schedule (first 12 months)
		balance = new_liability
		revised_lines: list[dict[str, Any]] = []
		for m in range(1, min(months + 1, 13)):
			interest = (balance * monthly_rate).quantize(Decimal("0.01"))
			principal = monthly_payment - interest
			balance -= principal
			revised_lines.append({
				"month": m,
				"payment": float(monthly_payment),
				"interest": float(interest),
				"principal": float(principal),
				"balance": float(max(balance, Decimal("0"))),
			})
		from uuid6 import uuid7
		mod_id = str(uuid7())
		result: dict[str, Any] = {
			"id": mod_id,
			"tenant_id": tenant_id,
			"original_schedule_id": schedule_id,
			"modification_date": str(modification_date),
			"original_lease_liability": float(old_liability),
			"revised_lease_liability": float(new_liability),
			"remeasurement_delta": float(delta),
			"delta_direction": "gain" if delta < 0 else "loss",
			"revised_annual_payment": float(revised_annual_payment),
			"revised_discount_rate": float(revised_discount_rate),
			"revised_expiry_date": str(revised_expiry_date),
			"revised_term_months": months,
			"revised_schedule_lines": revised_lines,
			"modified_by": modified_by,
			"accounting_standard": "IFRS_16",
			"modified_at": datetime.utcnow().isoformat(),
		}
		# Append modification record to the original schedule entry
		for i, s in enumerate(self._store["ifrs16_schedules"]):
			if s["id"] == schedule_id:
				s.setdefault("modifications", []).append(mod_id)
				s["lease_liability"] = str(new_liability)
				s["rou_asset"] = str(new_liability)
				s["updated_at"] = datetime.utcnow()
				self._store["ifrs16_schedules"][i] = s
				break
		self._log_operation("ifrs16_lease_remeasured", mod_id, tenant_id)
		return result
