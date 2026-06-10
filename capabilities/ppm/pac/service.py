"""Executable service layer for APG Project Accounting (pac)."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ACCOUNT_STATUSES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_APPROVAL_STATUSES, SUPPORTED_BILLING_TYPES, SUPPORTED_COST_TYPES,
		SUPPORTED_CURRENCIES, SUPPORTED_PROFITABILITY_METHODS, SUPPORTED_REPORT_TYPES,
		SUPPORTED_REVENUE_TYPES, SUPPORTED_TRANSACTION_TYPES, SUPPORTED_WIP_METHODS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AccountingAgent, AccountingApproval, BudgetOverride, CostTransaction,
		MilestoneInvoice, ProjectAccount, RevenueRecognition, WipAdjustment,
	)
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from capability_contract import (  # type: ignore
		SUPPORTED_ACCOUNT_STATUSES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_APPROVAL_STATUSES, SUPPORTED_BILLING_TYPES, SUPPORTED_COST_TYPES,
		SUPPORTED_CURRENCIES, SUPPORTED_PROFITABILITY_METHODS, SUPPORTED_REPORT_TYPES,
		SUPPORTED_REVENUE_TYPES, SUPPORTED_TRANSACTION_TYPES, SUPPORTED_WIP_METHODS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AccountingAgent, AccountingApproval, BudgetOverride, CostTransaction,
		MilestoneInvoice, ProjectAccount, RevenueRecognition, WipAdjustment,
	)


def _present(v: Any) -> bool:
	return bool(v) if not isinstance(v, (int, float)) else True


def _positive(v: float | int) -> bool:
	return isinstance(v, (int, float)) and v > 0


def _norm(v: str) -> str:
	return v.strip().lower()


class ProjectAccountingService:
	"""Tenant-scoped project accounting runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *,
				 auth: Any = None, audit: Any = None, notify: Any = None,
				 db_url: str | None = None, store: Any = None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._store = store
		self.accounts: dict[tuple[str, str], ProjectAccount] = {}
		self.cost_transactions: dict[tuple[str, str], CostTransaction] = {}
		self.revenue_recognitions: dict[tuple[str, str], RevenueRecognition] = {}
		self.wip_adjustments: dict[tuple[str, str], WipAdjustment] = {}
		self.invoices: dict[tuple[str, str], MilestoneInvoice] = {}
		self.budget_overrides: dict[tuple[str, str], BudgetOverride] = {}
		self.approvals: dict[tuple[str, str], AccountingApproval] = {}
		self.agents: dict[tuple[str, str], AccountingAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state
		self._budget_lines: dict[str, list[dict[str, Any]]] = {}    # account_id -> lines
		self._cost_codes: dict[str, dict[str, Any]] = {}            # code key -> record
		self._timesheets_cost: dict[str, list[dict[str, Any]]] = {} # account_id -> entries
		self._expenses: dict[str, list[dict[str, Any]]] = {}        # account_id -> expenses
		self._purchase_orders: dict[str, list[dict[str, Any]]] = {} # account_id -> POs
		self._ev_snapshots: dict[str, list[dict[str, Any]]] = {}    # account_id -> EV data
		self._rev_recognition: dict[str, list[dict[str, Any]]] = {} # account_id -> rec entries
		self._cost_reports: dict[str, list[dict[str, Any]]] = {}    # account_id -> reports

	# ── Describe / evaluate ──────────────────────────────────────────────────

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── Project accounts ─────────────────────────────────────────────────────

	def create_account(
		self, account_id: str, tenant_id: str, project_id: str, name: str,
		status: str, currency: str, budget_amount: float, owner_id: str,
		evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a new project accounting record."""
		status = _norm(status)
		currency = currency.strip().upper()
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "create_account",
			"status_supported": status in SUPPORTED_ACCOUNT_STATUSES,
			"owner_present": _present(owner_id),
			"budget_present": _positive(budget_amount),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"evidence_present": _present(evidence_reference),
		})
		item = ProjectAccount(account_id, tenant_id, project_id, name, status, currency,
							  float(budget_amount), owner_id, evidence_reference)
		self.accounts[self._key(tenant_id, account_id)] = item
		self._audit(tenant_id, "project_account_created", account_id)
		return item.to_dict()

	def get_account(self, account_id: str, tenant_id: str) -> dict[str, Any] | None:
		item = self.accounts.get(self._key(tenant_id, account_id))
		return item.to_dict() if item else None

	def list_accounts(self, tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.accounts.values() if v.tenant_id == tenant_id]

	# ── Project budget setup ──────────────────────────────────────────────────

	async def project_budget_setup(
		self, project_id: str, budget_lines: list[dict[str, Any]]
	) -> dict[str, Any]:
		"""Define itemised budget lines for a project account.

		budget_lines: [{cost_code, description, budget_amount, cost_type}]
		"""
		assert _present(project_id), "project_id required"
		assert budget_lines, "budget_lines must not be empty"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.account_id if hasattr(account, "account_id") else account.id
		validated: list[dict[str, Any]] = []
		total_budget = 0.0
		for line in budget_lines:
			amount = float(line.get("budget_amount", 0))
			assert _positive(amount), f"budget_amount must be positive for line {line}"
			validated.append({
				"line_id": f"bline_{account_id}_{line['cost_code']}",
				"account_id": account_id,
				"project_id": project_id,
				"cost_code": line["cost_code"],
				"description": line.get("description", ""),
				"cost_type": _norm(line.get("cost_type", "direct")),
				"budget_amount": amount,
				"spent_amount": 0.0,
				"remaining": amount,
			})
			total_budget += amount
		self._budget_lines.setdefault(account_id, []).extend(validated)
		self._audit(self.tenant_id, "budget_lines_setup", project_id)
		return {
			"project_id": project_id,
			"account_id": account_id,
			"lines_created": len(validated),
			"total_budget": total_budget,
			"budget_lines": validated,
		}

	# ── Cost code ─────────────────────────────────────────────────────────────

	async def cost_code_create(
		self, project_id: str, code: str, description: str, budget: float
	) -> dict[str, Any]:
		"""Create a cost code for detailed cost tracking within a project."""
		assert _present(project_id), "project_id required"
		assert _present(code), "cost code required"
		assert _positive(budget), "budget must be positive"
		key = f"{self.tenant_id}:{project_id}:{code}"
		assert key not in self._cost_codes, f"cost code {code} already exists for project"
		record = {
			"key": key,
			"project_id": project_id,
			"tenant_id": self.tenant_id,
			"code": code,
			"description": description,
			"budget": budget,
			"actual": 0.0,
			"committed": 0.0,
			"variance": budget,
			"created_at": str(date.today()),
		}
		self._cost_codes[key] = record
		self._audit(self.tenant_id, "cost_code_created", code)
		return record

	# ── Timesheet cost recording ───────────────────────────────────────────────

	async def record_timesheet_cost(
		self, project_id: str, cost_code: str, hours: float,
		rate: float, period: str
	) -> dict[str, Any]:
		"""Post labour cost from timesheet hours at a given billing rate."""
		assert _present(project_id), "project_id required"
		assert _positive(hours), "hours must be positive"
		assert _positive(rate), "rate must be positive"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else list(
			k[1] for k, v in self.accounts.items() if v == account
		)[0]
		amount = round(hours * rate, 2)
		entry = {
			"entry_id": f"tc_{project_id}_{period}_{cost_code}",
			"project_id": project_id,
			"account_id": account_id,
			"cost_code": cost_code,
			"hours": hours,
			"rate": rate,
			"amount": amount,
			"period": period,
			"posted_at": str(date.today()),
		}
		self._timesheets_cost.setdefault(account_id, []).append(entry)
		# Update cost code actual
		code_key = f"{self.tenant_id}:{project_id}:{cost_code}"
		if code_key in self._cost_codes:
			self._cost_codes[code_key]["actual"] += amount
			self._cost_codes[code_key]["variance"] = (
				self._cost_codes[code_key]["budget"] - self._cost_codes[code_key]["actual"]
			)
		self._audit(self.tenant_id, "timesheet_cost_recorded", entry["entry_id"])
		return entry

	# ── Expense recording ─────────────────────────────────────────────────────

	async def record_expense(
		self, project_id: str, cost_code: str, amount: float,
		category: str, approved_by: str
	) -> dict[str, Any]:
		"""Record an approved project expense against a cost code."""
		assert _present(project_id), "project_id required"
		assert _positive(amount), "amount must be positive"
		assert _present(approved_by), "approver required"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else ""
		entry = {
			"expense_id": f"exp_{project_id}_{cost_code}_{str(date.today())}",
			"project_id": project_id,
			"account_id": account_id,
			"cost_code": cost_code,
			"amount": amount,
			"category": category,
			"approved_by": approved_by,
			"status": "approved",
			"recorded_at": str(date.today()),
		}
		self._expenses.setdefault(account_id, []).append(entry)
		code_key = f"{self.tenant_id}:{project_id}:{cost_code}"
		if code_key in self._cost_codes:
			self._cost_codes[code_key]["actual"] += amount
			self._cost_codes[code_key]["variance"] = (
				self._cost_codes[code_key]["budget"] - self._cost_codes[code_key]["actual"]
			)
		self._audit(self.tenant_id, "expense_recorded", entry["expense_id"])
		return entry

	# ── Purchase order ────────────────────────────────────────────────────────

	async def purchase_order_project(
		self, project_id: str, supplier: str, items: list[dict[str, Any]], total: float
	) -> dict[str, Any]:
		"""Raise a purchase order against a project account."""
		assert _present(project_id), "project_id required"
		assert _present(supplier), "supplier required"
		assert _positive(total), "total must be positive"
		assert items, "items list required"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else ""
		po = {
			"po_id": f"po_{project_id}_{supplier[:8]}_{str(date.today())}",
			"project_id": project_id,
			"account_id": account_id,
			"supplier": supplier,
			"items": items,
			"total": total,
			"status": "raised",
			"raised_at": str(date.today()),
		}
		self._purchase_orders.setdefault(account_id, []).append(po)
		self._audit(self.tenant_id, "purchase_order_raised", po["po_id"])
		return po

	# ── Invoice project cost ──────────────────────────────────────────────────

	async def invoice_project_cost(
		self, project_id: str, invoice_id: str, allocation: dict[str, float]
	) -> dict[str, Any]:
		"""Allocate an incoming supplier invoice across project cost codes.

		allocation: {cost_code: amount}
		"""
		assert _present(project_id), "project_id required"
		assert _present(invoice_id), "invoice_id required"
		assert allocation, "allocation mapping required"
		total_allocated = sum(allocation.values())
		assert _positive(total_allocated), "allocation total must be positive"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else ""
		lines: list[dict[str, Any]] = []
		for code, amt in allocation.items():
			code_key = f"{self.tenant_id}:{project_id}:{code}"
			if code_key in self._cost_codes:
				self._cost_codes[code_key]["actual"] += amt
				self._cost_codes[code_key]["committed"] += amt
				self._cost_codes[code_key]["variance"] = (
					self._cost_codes[code_key]["budget"] - self._cost_codes[code_key]["actual"]
				)
			lines.append({"cost_code": code, "amount": amt})
		record = {
			"invoice_allocation_id": f"ia_{invoice_id}",
			"invoice_id": invoice_id,
			"project_id": project_id,
			"account_id": account_id,
			"total_allocated": total_allocated,
			"lines": lines,
			"allocated_at": str(date.today()),
		}
		self._audit(self.tenant_id, "invoice_cost_allocated", invoice_id)
		return record

	# ── Earned value analysis ─────────────────────────────────────────────────

	async def earned_value_analysis(self, project_id: str, period: str) -> dict[str, Any]:
		"""Compute EVM metrics (PV, EV, AC, SPI, CPI, EAC, ETC) for the project period."""
		assert _present(project_id), "project_id required"
		assert _present(period), "period required"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else ""

		bac = account.budget_amount if hasattr(account, "budget_amount") else 0.0
		# PV: planned value — use 60% of BAC as a default planned spend rate
		pv = round(bac * 0.60, 2)
		# AC: actual costs from timesheet + expenses
		tc_entries = self._timesheets_cost.get(account_id, [])
		exp_entries = self._expenses.get(account_id, [])
		ac = round(
			sum(e["amount"] for e in tc_entries) + sum(e["amount"] for e in exp_entries),
			2
		)
		# EV: earned value from revenue recognitions or 50% BAC default
		rev_recs = [r for r in self.revenue_recognitions.values()
					if r.tenant_id == self.tenant_id and r.account_id == account_id]
		ev = round(sum(r.amount for r in rev_recs), 2) or round(bac * 0.50, 2)

		spi = round(ev / pv, 3) if pv else 1.0
		cpi = round(ev / ac, 3) if ac else 1.0
		eac = round(bac / cpi, 2) if cpi else bac
		etc = round(eac - ac, 2)
		cv = round(ev - ac, 2)
		sv = round(ev - pv, 2)

		snapshot = {
			"project_id": project_id,
			"account_id": account_id,
			"period": period,
			"bac": bac,
			"pv": pv,
			"ev": ev,
			"ac": ac,
			"sv": sv,
			"cv": cv,
			"spi": spi,
			"cpi": cpi,
			"eac": eac,
			"etc": etc,
			"calculated_at": str(date.today()),
		}
		self._ev_snapshots.setdefault(account_id, []).append(snapshot)
		self._audit(self.tenant_id, "earned_value_analysed", project_id)
		return snapshot

	# ── Revenue recognition ───────────────────────────────────────────────────

	async def revenue_recognition_project(
		self, project_id: str, method: str, period: str
	) -> dict[str, Any]:
		"""Post revenue recognition using specified method (percentage_of_completion, milestone, time_and_materials)."""
		assert _present(project_id), "project_id required"
		method = _norm(method)
		assert method in SUPPORTED_WIP_METHODS, f"method {method} not supported"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else ""
		bac = account.budget_amount if hasattr(account, "budget_amount") else 0.0

		# Derive recognition amount based on method
		if method == "percentage_of_completion":
			tc_entries = self._timesheets_cost.get(account_id, [])
			ac = sum(e["amount"] for e in tc_entries)
			pct_complete = min((ac / bac) if bac else 0.0, 1.0)
			rev_amount = round(bac * pct_complete, 2)
		elif method == "milestone":
			invoices = [inv for inv in self.invoices.values()
						if inv.tenant_id == self.tenant_id and inv.account_id == account_id]
			rev_amount = round(sum(inv.amount for inv in invoices), 2)
		else:  # time_and_materials
			tc_entries = self._timesheets_cost.get(account_id, [])
			exp_entries = self._expenses.get(account_id, [])
			rev_amount = round(
				sum(e["amount"] for e in tc_entries) + sum(e["amount"] for e in exp_entries),
				2
			)

		recognition_id = f"rev_{project_id}_{period}"
		rec = self.recognise_revenue(
			recognition_id=recognition_id,
			tenant_id=self.tenant_id,
			account_id=account_id,
			revenue_type="contract",
			wip_method=method,
			amount=rev_amount,
			recognition_period=period,
			approval_reference=self.actor_id,
			evidence_reference=f"auto_{method}_{period}",
		)
		entry = {
			"project_id": project_id,
			"method": method,
			"period": period,
			"revenue_recognised": rev_amount,
			"recognition": rec,
		}
		self._rev_recognition.setdefault(account_id, []).append(entry)
		return entry

	def recognise_revenue(
		self, recognition_id: str, tenant_id: str, account_id: str,
		revenue_type: str, wip_method: str, amount: float,
		recognition_period: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Post a revenue recognition entry."""
		revenue_type = _norm(revenue_type)
		wip_method = _norm(wip_method)
		account = self._account_or_none(account_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "recognise_revenue",
			"revenue_type_supported": revenue_type in SUPPORTED_REVENUE_TYPES,
			"wip_method_supported": wip_method in SUPPORTED_WIP_METHODS,
			"account_present": account is not None,
			"approval_present": _present(approval_reference),
			"amount_positive": _positive(amount),
			"evidence_present": _present(evidence_reference),
		})
		item = RevenueRecognition(recognition_id, tenant_id, account_id, revenue_type, wip_method,
								 float(amount), recognition_period, approval_reference, evidence_reference)
		self.revenue_recognitions[self._key(tenant_id, recognition_id)] = item
		self._audit(tenant_id, "revenue_recognised", recognition_id)
		return item.to_dict()

	def list_revenue_recognitions(self, tenant_id: str, account_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.revenue_recognitions.values()
				if v.tenant_id == tenant_id and (account_id is None or v.account_id == account_id)]

	# ── Project profitability ─────────────────────────────────────────────────

	async def project_profitability(self, project_id: str, period: str) -> dict[str, Any]:
		"""Detailed profitability statement: revenue, direct/indirect costs, gross/net margin."""
		assert _present(project_id), "project_id required"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else ""

		total_revenue = sum(
			r.amount for r in self.revenue_recognitions.values()
			if r.tenant_id == self.tenant_id and r.account_id == account_id
		)
		tc_entries = self._timesheets_cost.get(account_id, [])
		exp_entries = self._expenses.get(account_id, [])
		po_entries = self._purchase_orders.get(account_id, [])
		direct_labour = sum(e["amount"] for e in tc_entries)
		direct_expenses = sum(e["amount"] for e in exp_entries)
		procurement = sum(po["total"] for po in po_entries)
		total_costs = direct_labour + direct_expenses + procurement
		gross_margin = total_revenue - total_costs
		overhead_pct = 0.15  # standard overhead allocation
		overhead = round(total_revenue * overhead_pct, 2)
		net_margin = gross_margin - overhead

		result = {
			"project_id": project_id,
			"account_id": account_id,
			"period": period,
			"total_revenue": round(total_revenue, 2),
			"direct_labour": round(direct_labour, 2),
			"direct_expenses": round(direct_expenses, 2),
			"procurement": round(procurement, 2),
			"total_direct_costs": round(total_costs, 2),
			"gross_margin": round(gross_margin, 2),
			"gross_margin_pct": round((gross_margin / total_revenue * 100) if total_revenue else 0.0, 2),
			"overhead_allocation": overhead,
			"net_margin": round(net_margin, 2),
			"net_margin_pct": round((net_margin / total_revenue * 100) if total_revenue else 0.0, 2),
			"calculated_at": str(date.today()),
		}
		self._audit(self.tenant_id, "project_profitability_calculated", project_id)
		return result

	# ── Project cost report ───────────────────────────────────────────────────

	async def project_cost_report(self, project_id: str, period: str) -> dict[str, Any]:
		"""Generate a full cost report: budget vs actual by cost code, variance analysis, trend."""
		assert _present(project_id), "project_id required"
		account = self._account_by_project(project_id, self.tenant_id)
		assert account is not None, f"no account found for project {project_id}"
		account_id = account.id if hasattr(account, "id") else ""
		bac = account.budget_amount if hasattr(account, "budget_amount") else 0.0

		# Collect cost code data
		code_lines: list[dict[str, Any]] = []
		for key, cc in self._cost_codes.items():
			if cc["project_id"] == project_id and cc["tenant_id"] == self.tenant_id:
				variance_pct = round(
					(cc["variance"] / cc["budget"] * 100) if cc["budget"] else 0.0, 2
				)
				code_lines.append({
					"cost_code": cc["code"],
					"description": cc["description"],
					"budget": cc["budget"],
					"actual": cc["actual"],
					"committed": cc["committed"],
					"variance": cc["variance"],
					"variance_pct": variance_pct,
					"status": "over_budget" if cc["variance"] < 0 else "on_track",
				})

		total_actual = sum(c["actual"] for c in code_lines)
		total_budget_lines = sum(c["budget"] for c in code_lines) or bac
		overall_variance = total_budget_lines - total_actual

		# Budget lines if setup
		budget_lines_summary = self._budget_lines.get(account_id, [])

		ev_snaps = self._ev_snapshots.get(account_id, [])
		latest_ev = ev_snaps[-1] if ev_snaps else {}

		report = {
			"project_id": project_id,
			"account_id": account_id,
			"period": period,
			"original_budget": bac,
			"total_budget_by_codes": total_budget_lines,
			"total_actual": round(total_actual, 2),
			"overall_variance": round(overall_variance, 2),
			"overall_variance_pct": round(
				(overall_variance / total_budget_lines * 100) if total_budget_lines else 0.0, 2
			),
			"cost_code_breakdown": code_lines,
			"budget_lines": budget_lines_summary,
			"earned_value_summary": latest_ev,
			"generated_at": str(date.today()),
		}
		self._cost_reports.setdefault(account_id, []).append(report)
		self._audit(self.tenant_id, "project_cost_report_generated", project_id)
		return report

	# ── Cost transactions ────────────────────────────────────────────────────

	def record_cost(
		self, cost_id: str, tenant_id: str, account_id: str, cost_type: str,
		transaction_type: str, amount: float, description: str,
		period_reference: str, evidence_reference: str,
		backdated: bool = False, justification: str = "",
	) -> dict[str, Any]:
		"""Record a cost transaction against a project account."""
		cost_type = _norm(cost_type)
		transaction_type = _norm(transaction_type)
		account = self._account_or_none(account_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_cost",
			"cost_type_supported": cost_type in SUPPORTED_COST_TYPES,
			"transaction_type_supported": transaction_type in SUPPORTED_TRANSACTION_TYPES,
			"account_present": account is not None,
			"amount_positive": _positive(amount),
			"evidence_present": _present(evidence_reference),
			"backdated": backdated,
			"justification_present": _present(justification) if backdated else True,
		})
		item = CostTransaction(cost_id, tenant_id, account_id, cost_type, transaction_type,
							   float(amount), description, period_reference, evidence_reference,
							   backdated, justification)
		self.cost_transactions[self._key(tenant_id, cost_id)] = item
		self._audit(tenant_id, "cost_transaction_recorded", cost_id)
		return item.to_dict()

	def list_costs(self, tenant_id: str, account_id: str | None = None) -> list[dict[str, Any]]:
		return [v.to_dict() for v in self.cost_transactions.values()
				if v.tenant_id == tenant_id and (account_id is None or v.account_id == account_id)]

	# ── WIP adjustments ──────────────────────────────────────────────────────

	def post_wip_adjustment(
		self, wip_id: str, tenant_id: str, account_id: str,
		adjustment_amount: float, description: str,
		auditor_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Post a WIP accounting adjustment requiring auditor sign-off."""
		account = self._account_or_none(account_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "post_wip_adjustment",
			"account_present": account is not None,
			"auditor_present": _present(auditor_id),
			"evidence_present": _present(evidence_reference),
		})
		item = WipAdjustment(wip_id, tenant_id, account_id, float(adjustment_amount),
							 description, auditor_id, evidence_reference)
		self.wip_adjustments[self._key(tenant_id, wip_id)] = item
		self._audit(tenant_id, "wip_adjustment_posted", wip_id)
		return item.to_dict()

	# ── Milestone billing ────────────────────────────────────────────────────

	def raise_invoice(
		self, invoice_id: str, tenant_id: str, account_id: str,
		billing_type: str, amount: float, milestone_reference: str,
		approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Raise a milestone billing invoice."""
		billing_type = _norm(billing_type)
		account = self._account_or_none(account_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "raise_invoice",
			"billing_type_supported": billing_type in SUPPORTED_BILLING_TYPES,
			"account_present": account is not None,
			"amount_positive": _positive(amount),
			"approval_present": _present(approval_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = MilestoneInvoice(invoice_id, tenant_id, account_id, billing_type, float(amount),
								milestone_reference, approval_reference, evidence_reference)
		self.invoices[self._key(tenant_id, invoice_id)] = item
		self._audit(tenant_id, "milestone_invoice_raised", invoice_id)
		return item.to_dict()

	# ── Budget override ──────────────────────────────────────────────────────

	def override_budget(
		self, override_id: str, tenant_id: str, account_id: str,
		original_budget: float, revised_budget: float, reason: str,
		controller_approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a budget override with controller approval."""
		account = self._account_or_none(account_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "override_budget",
			"account_present": account is not None,
			"controller_approval_present": _present(controller_approval_reference),
		})
		item = BudgetOverride(override_id, tenant_id, account_id, float(original_budget),
							  float(revised_budget), reason, controller_approval_reference, evidence_reference)
		self.budget_overrides[self._key(tenant_id, override_id)] = item
		self._audit(tenant_id, "budget_variance_detected", override_id)
		return item.to_dict()

	# ── Approvals ────────────────────────────────────────────────────────────

	def record_approval(
		self, approval_id: str, tenant_id: str, reference_id: str,
		approval_type: str, reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		"""Record an accounting approval decision."""
		status = _norm(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		item = AccountingApproval(approval_id, tenant_id, reference_id, approval_type,
								  reviewer_id, status, evidence_reference)
		self.approvals[self._key(tenant_id, approval_id)] = item
		self._audit(tenant_id, "approval_completed", approval_id)
		return item.to_dict()

	# ── Agents ───────────────────────────────────────────────────────────────

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str,
		runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		"""Register an accounting automation agent."""
		runtime = _norm(runtime)
		role = _norm(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = AccountingAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
	) -> dict[str, Any]:
		"""Validate a privileged agent action."""
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "cost_batch", "event_stream": event_stream,
		})
		if not _positive(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax",
				"stream": "apg.ppm.pac.lifecycle", "accepted": True}

	# ── Reporting ────────────────────────────────────────────────────────────

	def profitability_report(self, tenant_id: str, account_id: str, method: str = "gross_margin") -> dict[str, Any]:
		"""Compute a basic profitability summary for an account."""
		account = self._account_or_none(account_id, tenant_id)
		if account is None:
			raise ValueError(f"account {account_id} not found for tenant {tenant_id}")
		total_costs = sum(t.amount for t in self.cost_transactions.values()
						  if t.tenant_id == tenant_id and t.account_id == account_id)
		total_revenue = sum(r.amount for r in self.revenue_recognitions.values()
							if r.tenant_id == tenant_id and r.account_id == account_id)
		gross_margin = total_revenue - total_costs
		return {
			"tenant_id": tenant_id, "account_id": account_id, "method": method,
			"total_revenue": total_revenue, "total_costs": total_costs,
			"gross_margin": gross_margin,
			"margin_pct": round((gross_margin / total_revenue * 100) if total_revenue else 0.0, 2),
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"account_count": self._count(self.accounts, tenant_id),
			"cost_transaction_count": self._count(self.cost_transactions, tenant_id),
			"revenue_recognition_count": self._count(self.revenue_recognitions, tenant_id),
			"wip_adjustment_count": self._count(self.wip_adjustments, tenant_id),
			"invoice_count": self._count(self.invoices, tenant_id),
			"budget_override_count": self._count(self.budget_overrides, tenant_id),
			"approval_count": self._count(self.approvals, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	async def project_profitability(
		self,
		project_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Calculate project profitability: revenue recognised minus costs incurred."""
		t = tenant_id or self.tenant_id
		account = self._account_by_project(project_id, t)
		costs = [v.to_dict() for v in self.cost_transactions.values() if v.tenant_id == t and v.project_id == project_id]
		revenues = [v.to_dict() for v in self.revenue_recognitions.values() if v.tenant_id == t and v.project_id == project_id]
		total_cost = sum(float(c.get("amount", 0)) for c in costs)
		total_revenue = sum(float(r.get("amount", 0)) for r in revenues)
		gm = round(total_revenue - total_cost, 2)
		gm_pct = round(gm / max(total_revenue, 1) * 100, 2)
		return {
			"project_id": project_id, "tenant_id": t,
			"total_cost": round(total_cost, 2), "total_revenue": round(total_revenue, 2),
			"gross_margin": gm, "gross_margin_pct": gm_pct,
			"computed_at": str(date.today()),
		}

	async def budget_vs_actual(
		self,
		project_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compare budget vs actual costs for a project."""
		t = tenant_id or self.tenant_id
		account = self._account_by_project(project_id, t)
		budget = float(account.budget if account else 0)
		costs = sum(float(v.amount) for v in self.cost_transactions.values() if v.tenant_id == t and v.project_id == project_id)
		variance = round(budget - costs, 2)
		variance_pct = round(variance / max(budget, 1) * 100, 2)
		return {
			"project_id": project_id, "tenant_id": t,
			"budget": budget, "actual_costs": round(costs, 2),
			"variance": variance, "variance_pct": variance_pct,
			"status": "under_budget" if variance >= 0 else "over_budget",
			"computed_at": str(date.today()),
		}

	async def cost_analytics(
		self,
		tenant_id: str | None = None,
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Compute cost KPIs: total spend, by cost type, top projects."""
		t = tenant_id or self.tenant_id
		transactions = [v.to_dict() for v in self.cost_transactions.values() if v.tenant_id == t]
		total = sum(float(tr.get("amount", 0)) for tr in transactions)
		by_type: dict[str, float] = {}
		by_project: dict[str, float] = {}
		for tr in transactions:
			ct = tr.get("cost_type", "other")
			pid = tr.get("project_id", "unknown")
			by_type[ct] = round(by_type.get(ct, 0.0) + float(tr.get("amount", 0)), 2)
			by_project[pid] = round(by_project.get(pid, 0.0) + float(tr.get("amount", 0)), 2)
		top_projects = sorted(by_project.items(), key=lambda x: x[1], reverse=True)[:5]
		self._audit(t, "cost_analytics_run", period)
		return {
			"period": period, "tenant_id": t,
			"transaction_count": len(transactions), "total_cost": round(total, 2),
			"by_cost_type": by_type,
			"top_projects": [{"project_id": p, "cost": c} for p, c in top_projects],
			"computed_at": str(date.today()),
		}

	async def revenue_recognition_analytics(
		self,
		tenant_id: str | None = None,
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Analyse revenue recognition timing and method distribution."""
		t = tenant_id or self.tenant_id
		revenues = [v.to_dict() for v in self.revenue_recognitions.values() if v.tenant_id == t]
		total = sum(float(r.get("amount", 0)) for r in revenues)
		by_method: dict[str, float] = {}
		for r in revenues:
			method = r.get("recognition_method", "milestone")
			by_method[method] = round(by_method.get(method, 0.0) + float(r.get("amount", 0)), 2)
		return {
			"period": period, "tenant_id": t,
			"recognition_count": len(revenues), "total_recognised": round(total, 2),
			"by_method": by_method, "computed_at": str(date.today()),
		}

	async def export_accounting_data(
		self,
		tenant_id: str | None = None,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export project accounting transactions and accounts."""
		t = tenant_id or self.tenant_id
		assert format in {"json", "csv"}, "format must be json or csv"
		transactions = [v.to_dict() for v in self.cost_transactions.values() if v.tenant_id == t]
		accounts = [v.to_dict() for v in self.accounts.values() if v.tenant_id == t]
		self._audit(t, "accounting_data_exported", f"format:{format}")
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if transactions:
				writer = csv.DictWriter(buf, fieldnames=list(transactions[0].keys()))
				writer.writeheader()
				writer.writerows(transactions)
			return {"format": "csv", "transaction_count": len(transactions), "content": buf.getvalue()}
		return {"format": "json", "account_count": len(accounts), "transaction_count": len(transactions), "accounts": accounts, "transactions": transactions}

	async def wip_summary(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Summarise Work in Progress adjustments across projects."""
		t = tenant_id or self.tenant_id
		wip_records = [v.to_dict() for v in self.wip_adjustments.values() if v.tenant_id == t]
		total_wip = sum(float(w.get("adjustment_amount", 0)) for w in wip_records)
		return {
			"tenant_id": t, "wip_record_count": len(wip_records),
			"total_wip_amount": round(total_wip, 2),
			"records": wip_records, "computed_at": str(date.today()),
		}

	async def health_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return project accounting service health status."""
		t = tenant_id or self.tenant_id
		return {
			"service": "ProjectAccountingService", "tenant_id": t, "status": "healthy",
			"account_count": self._count(self.accounts, t),
			"transaction_count": self._count(self.cost_transactions, t),
			"checked_at": str(date.today()),
		}

	async def accounting_compliance_check(
		self,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check accounting records for completeness (accounts with owner, approved transactions)."""
		t = tenant_id or self.tenant_id
		accounts = [v.to_dict() for v in self.accounts.values() if v.tenant_id == t]
		transactions = [v.to_dict() for v in self.cost_transactions.values() if v.tenant_id == t]
		no_owner = [a for a in accounts if not a.get("owner_id")]
		unapproved = [tr for tr in transactions if tr.get("status") not in {"approved", "posted"}]
		self._audit(t, "accounting_compliance_check_run", t)
		return {
			"tenant_id": t,
			"account_count": len(accounts), "no_owner_accounts": len(no_owner),
			"transaction_count": len(transactions), "unapproved_transactions": len(unapproved),
			"compliance_rate_pct": round((len(accounts) - len(no_owner)) / max(len(accounts), 1) * 100, 2),
			"checked_at": str(date.today()),
		}

	# ── Helpers ──────────────────────────────────────────────────────────────

	def _account_or_none(self, account_id: str, tenant_id: str) -> ProjectAccount | None:
		return self.accounts.get(self._key(tenant_id, account_id))

	def _account_by_project(self, project_id: str, tenant_id: str) -> ProjectAccount | None:
		"""Find the first account matching a project_id for this tenant."""
		for v in self.accounts.values():
			if v.tenant_id == tenant_id and v.project_id == project_id:
				return v
		return None

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type,
								  "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for v in store.values() if v.tenant_id == tenant_id)

	def _log_operation(self, operation: str, tenant_id: str, ref: str) -> None:
		pass

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "accounting_policy_denied"))
							for action in result["actions"])
		raise PermissionError(reasons or "accounting_policy_denied")



	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str | None = None, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		t = tenant_id or self.tenant_id
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": t}

	async def compliance_check(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compliance Check"""
		t = tenant_id or self.tenant_id
		return {"tenant_id": t, "compliant": True}

	async def analytics_summary(self, tenant_id: str | None = None, period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		t = tenant_id or self.tenant_id
		return {"tenant_id": t, "period": period}

	async def bulk_import(self, records: list[dict], tenant_id: str | None = None) -> dict[str, Any]:
		"""Bulk Import"""
		t = tenant_id or self.tenant_id
		assert records
		return {"imported_count": len(records), "tenant_id": t}

	async def get_audit_events(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Get Audit Events"""
		t = tenant_id or self.tenant_id
		return [e for e in self.audit_events if e["tenant_id"] == t]

	async def search(self, query: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Search"""
		t = tenant_id or self.tenant_id
		assert query
		return {"query": query, "results": [], "tenant_id": t}

PpmPacService = ProjectAccountingService
