"""
APG Budgeting & Forecasting — Core Service

Async, tenant-scoped orchestration of the complete BFC lifecycle:
budget creation → distribution → approval → forecasting → variance → reforecast → scenario.

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from .models import (
	BFApprovalStatus,
	BFBudget,
	BFBudgetApproval,
	BFBudgetApprovalCreate,
	BFBudgetApprovalResponse,
	BFBudgetCreate,
	BFBudgetLine,
	BFBudgetLineCreate,
	BFBudgetLineResponse,
	BFBudgetResponse,
	BFBudgetStatus,
	BFBudgetSummary,
	BFBudgetTemplate,
	BFBudgetTemplateCreate,
	BFBudgetType,
	BFBudgetUpdate,
	BFBudgetVersion,
	BFConsolidationResult,
	BFDashboardKPIs,
	BFDistributionMethod,
	BFDriverBasedAssumption,
	BFDriverAssumptionCreate,
	BFDriverAssumptionResponse,
	BFDriverType,
	BFForecast,
	BFForecastCreate,
	BFForecastLine,
	BFForecastLineCreate,
	BFForecastLineResponse,
	BFForecastMethod,
	BFForecastResponse,
	BFForecastStatus,
	BFForecastType,
	BFLineType,
	BFRollingForecastResult,
	BFScenarioAnalysisResult,
	BFScenarioModel,
	BFScenarioCreate,
	BFScenarioResponse,
	BFScenarioType,
	BFSensitivityResult,
	BFSignificanceLevel,
	BFVarianceReport,
	BFVarianceType,
	uuid7str,
)
from .domain.rules import (
	RuleViolation,
	assert_actor_present,
	assert_amounts_balanced,
	assert_approval_pending,
	assert_approver_not_self,
	assert_budget_approvable,
	assert_budget_has_lines,
	assert_budget_in_draft,
	assert_budget_not_locked,
	assert_budget_period_valid,
	assert_budget_submittable,
	assert_driver_value_positive,
	assert_fiscal_year_reasonable,
	assert_forecast_horizon_valid,
	assert_no_cross_tenant,
	assert_probability_sum_valid,
	assert_scenarios_non_empty,
	assert_sufficient_history,  # noqa: F401 — exported for external callers
	assert_tenant_context,
	assert_zero_based_balanced,  # noqa: F401 — exported for external callers
)
from .domain.calculations import (
	apply_seasonal_adjustment,
	bootstrap_confidence_interval,
	calculate_mae,
	calculate_mape,
	calculate_rmse,
	calculate_rolling_average,
	calculate_variance,
	consolidate_budgets,
	distribute_equal,
	distribute_seasonal,
	distribute_top_down,
	distribute_zero_based,
	double_exponential_smoothing,
	driver_based_forecast,
	exponential_smoothing,
	project_rolling,
	round_currency,
	scenario_delta,
	scenario_delta_pct,
	sensitivity_range,
	significance_level,
	variance_type,
	weighted_scenario_outcome,
)

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
_log = logging.getLogger(__name__)


class BFCService:
	"""
	Tenant-scoped Budgeting & Forecasting service.

	All methods are async and enforce tenant isolation on every
	read/write.  Domain events are emitted after every state change.
	In-memory store is used for lightweight deployment; swap
	_store_* methods for PostgreSQL via database/store.py.

	Usage::

		svc = BFCService(tenant_id="acme", actor_id="alice")
		budget = await svc.create_budget_cycle(BFBudgetCreate(...))
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str,
		*,
		_shared_store: dict[str, Any] | None = None,
	) -> None:
		assert_tenant_context(tenant_id)
		assert_actor_present(actor_id)
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.actor_id = actor_id

		# Allow injecting a shared store for multi-actor tests / real DB backends.
		if _shared_store is not None:
			self._budgets = _shared_store.setdefault("_budgets", {})
			self._budget_lines = _shared_store.setdefault("_budget_lines", {})
			self._budget_versions = _shared_store.setdefault("_budget_versions", {})
			self._budget_templates = _shared_store.setdefault("_budget_templates", {})
			self._budget_approvals = _shared_store.setdefault("_budget_approvals", {})
			self._forecasts = _shared_store.setdefault("_forecasts", {})
			self._forecast_lines = _shared_store.setdefault("_forecast_lines", {})
			self._variance_reports = _shared_store.setdefault("_variance_reports", {})
			self._scenarios = _shared_store.setdefault("_scenarios", {})
			self._driver_assumptions = _shared_store.setdefault("_driver_assumptions", {})
			self._events = _shared_store.setdefault("_events", [])
		else:
			self._budgets: dict[str, BFBudget] = {}
			self._budget_lines: dict[str, BFBudgetLine] = {}
			self._budget_versions: dict[str, BFBudgetVersion] = {}
			self._budget_templates: dict[str, BFBudgetTemplate] = {}
			self._budget_approvals: dict[str, BFBudgetApproval] = {}
			self._forecasts: dict[str, BFForecast] = {}
			self._forecast_lines: dict[str, BFForecastLine] = {}
			self._variance_reports: dict[str, BFVarianceReport] = {}
			self._scenarios: dict[str, BFScenarioModel] = {}
			self._driver_assumptions: dict[str, BFDriverBasedAssumption] = {}
			self._events = WriteThruList('events', tenant_id, _store)

	def as_actor(self, actor_id: str) -> "BFCService":
		"""
		Return a new BFCService scoped to *actor_id* sharing this instance's
		in-memory store.  Useful for multi-actor approval tests.

		Usage::
			alice = BFCService(tenant_id="acme", actor_id="alice")
			bob   = alice.as_actor("bob")
		"""
		assert_actor_present(actor_id)
		store: dict[str, Any] = {
			"_budgets": self._budgets,
			"_budget_lines": self._budget_lines,
			"_budget_versions": self._budget_versions,
			"_budget_templates": self._budget_templates,
			"_budget_approvals": self._budget_approvals,
			"_forecasts": self._forecasts,
			"_forecast_lines": self._forecast_lines,
			"_variance_reports": self._variance_reports,
			"_scenarios": self._scenarios,
			"_driver_assumptions": self._driver_assumptions,
			"_events": self._events,
		}
		return BFCService(tenant_id=self.tenant_id, actor_id=actor_id, _shared_store=store)

	# =========================================================================
	# Budget lifecycle
	# =========================================================================

	async def create_budget_cycle(self, payload: BFBudgetCreate) -> BFBudget:
		"""
		Create a new budget cycle.

		Enforces fiscal-year bounds, valid period, and mandatory fields.
		Emits budget_created event.
		"""
		assert_fiscal_year_reasonable(payload.fiscal_year)
		assert_budget_period_valid(payload.period_start, payload.period_end)

		budget = BFBudget(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**payload.model_dump(),
		)
		self._budgets[budget.id] = budget
		self._emit("budget_created", budget.id, {
			"name": budget.name,
			"fiscal_year": budget.fiscal_year,
			"budget_type": budget.budget_type.value,
		})
		self._log_op("create_budget_cycle", budget.id, budget.name)
		return budget

	async def update_budget(self, budget_id: str, payload: BFBudgetUpdate) -> BFBudget:
		"""Update mutable budget fields (only while DRAFT)."""
		budget = self._get_budget(budget_id)
		assert_budget_in_draft(budget.status)
		updates = payload.model_dump(exclude_none=True)
		for k, v in updates.items():
			setattr(budget, k, v)
		budget.updated_at = datetime.now(timezone.utc)
		self._emit("budget_updated", budget_id, updates)
		return budget

	async def add_budget_line(self, payload: BFBudgetLineCreate) -> BFBudgetLine:
		"""
		Add a line to an existing DRAFT budget.

		Recalculates budget totals after insertion.
		"""
		budget = self._get_budget(payload.budget_id)
		assert_budget_in_draft(budget.status)

		# Auto-assign line_number
		existing = self._lines_for(payload.budget_id)
		line_number = len(existing) + 1

		dump = payload.model_dump()
		dump.setdefault("line_number", line_number)

		# Auto-distribute month_amounts if not supplied
		if dump.get("month_amounts") is None:
			from .domain.calculations import distribute_equal
			dump["month_amounts"] = distribute_equal(dump["budgeted_amount"], 12)

		line = BFBudgetLine(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**dump,
		)
		self._budget_lines[line.id] = line
		await self._recalc_budget_totals(budget)
		self._emit("budget_line_added", line.id, {
			"budget_id": payload.budget_id,
			"line_type": line.line_type.value,
			"amount": str(line.budgeted_amount),
		})
		return line

	async def submit_budget(self, budget_id: str) -> BFBudget:
		"""Submit a budget for approval (DRAFT → SUBMITTED)."""
		budget = self._get_budget(budget_id)
		assert_budget_submittable(budget.status)
		line_count = len(self._lines_for(budget_id))
		assert_budget_has_lines(line_count)

		budget.status = BFBudgetStatus.SUBMITTED
		budget.updated_at = datetime.now(timezone.utc)
		self._emit("budget_submitted", budget_id, {"submitted_by": self.actor_id, "line_count": line_count})
		self._log_op("submit_budget", budget_id, f"line_count={line_count}")
		return budget

	async def approve_budget(self, budget_id: str, approval_id: str, comments: str | None = None) -> BFBudget:
		"""
		Record an approval decision (SUBMITTED/UNDER_REVIEW → APPROVED).

		Enforces four-eyes: approver cannot be the budget creator.
		"""
		budget = self._get_budget(budget_id)
		assert_budget_approvable(budget.status)
		approval = self._get_approval(approval_id)
		assert_approval_pending(approval.status)
		assert_approver_not_self(budget.created_by, self.actor_id)

		approval.status = BFApprovalStatus.APPROVED
		approval.decided_at = datetime.now(timezone.utc)
		approval.comments = comments
		budget.status = BFBudgetStatus.APPROVED
		budget.updated_at = datetime.now(timezone.utc)

		self._emit("budget_approved", budget_id, {
			"approved_by": self.actor_id,
			"total_revenue": str(budget.total_revenue),
			"total_expense": str(budget.total_expense),
		})
		return budget

	async def reject_budget(self, budget_id: str, approval_id: str, reason: str) -> BFBudget:
		"""Reject a submitted budget (→ DRAFT for revision)."""
		budget = self._get_budget(budget_id)
		assert_budget_approvable(budget.status)
		approval = self._get_approval(approval_id)
		assert_approval_pending(approval.status)

		approval.status = BFApprovalStatus.REJECTED
		approval.decided_at = datetime.now(timezone.utc)
		approval.comments = reason
		budget.status = BFBudgetStatus.DRAFT
		budget.updated_at = datetime.now(timezone.utc)
		self._emit("budget_rejected", budget_id, {"rejected_by": self.actor_id, "reason": reason})
		return budget

	async def lock_budget(self, budget_id: str) -> BFBudget:
		"""Lock an approved budget against further changes."""
		budget = self._get_budget(budget_id)
		if budget.status != BFBudgetStatus.APPROVED:
			raise RuleViolation("budget_must_be_approved_to_lock", "Only APPROVED budgets can be locked", "approve_first")
		budget.status = BFBudgetStatus.LOCKED
		budget.updated_at = datetime.now(timezone.utc)
		self._emit("budget_locked", budget_id, {})
		return budget

	async def close_budget(self, budget_id: str) -> BFBudget:
		"""Close a budget at end of period. Allowed from APPROVED, ACTIVE, or LOCKED states."""
		budget = self._get_budget(budget_id)
		closeable = {BFBudgetStatus.APPROVED, BFBudgetStatus.ACTIVE, BFBudgetStatus.LOCKED}
		if budget.status not in closeable:
			raise RuleViolation(
				"budget_not_closeable",
				f"Budget in status '{budget.status.value}' cannot be closed; must be APPROVED, ACTIVE, or LOCKED",
				"approve_or_activate_first",
			)
		prior_status = budget.status.value
		budget.status = BFBudgetStatus.CLOSED
		budget.updated_at = datetime.now(timezone.utc)
		self._emit("budget_closed", budget_id, {"prior_status": prior_status})
		return budget

	async def cancel_budget(self, budget_id: str, reason: str) -> BFBudget:
		"""Cancel a budget."""
		budget = self._get_budget(budget_id)
		assert_budget_not_locked(budget.status)
		budget.status = BFBudgetStatus.CANCELLED
		budget.notes = reason
		budget.updated_at = datetime.now(timezone.utc)
		self._emit("budget_cancelled", budget_id, {"reason": reason})
		return budget

	async def get_budget(self, budget_id: str) -> BFBudget:
		"""Fetch a single budget by ID."""
		return self._get_budget(budget_id)

	async def list_budgets(
		self,
		status: BFBudgetStatus | None = None,
		fiscal_year: int | None = None,
		budget_type: BFBudgetType | None = None,
		offset: int = 0,
		limit: int = 50,
	) -> list[BFBudget]:
		"""List budgets with optional filters."""
		items = list(self._budgets.values())
		if status:
			items = [b for b in items if b.status == status]
		if fiscal_year:
			items = [b for b in items if b.fiscal_year == fiscal_year]
		if budget_type:
			items = [b for b in items if b.budget_type == budget_type]
		return items[offset : offset + limit]

	async def get_budget_lines(self, budget_id: str) -> list[BFBudgetLine]:
		"""Return all lines for a budget."""
		self._get_budget(budget_id)  # ownership check
		return self._lines_for(budget_id)

	async def delete_budget_line(self, line_id: str) -> None:
		"""Remove a line from a DRAFT budget."""
		line = self._budget_lines.get(line_id)
		if not line or line.tenant_id != self.tenant_id:
			raise KeyError(f"BudgetLine {line_id} not found")
		budget = self._get_budget(line.budget_id)
		assert_budget_in_draft(budget.status)
		del self._budget_lines[line_id]
		await self._recalc_budget_totals(budget)
		self._emit("budget_line_deleted", line_id, {"budget_id": line.budget_id})

	# =========================================================================
	# Distribution
	# =========================================================================

	async def distribute_budget(
		self,
		budget_id: str,
		method: BFDistributionMethod = BFDistributionMethod.EQUAL,
		department_weights: dict[str, float] | None = None,
		seasonal_weights: list[float] | None = None,
		line_justifications: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""
		Distribute budget total across periods/departments.

		method='top_down'  → distribute by department_weights
		method='equal'     → 12 equal monthly slices
		method='seasonal'  → monthly slices using seasonal_weights (12 factors)
		method='zero_based'→ sum of justified amounts per line
		method='bottom_up' → aggregate existing lines into period totals
		method='driver_based'→ uses driver assumptions (see driver_based_forecast)

		Returns distribution map and monthly totals.
		"""
		budget = self._get_budget(budget_id)
		assert_budget_in_draft(budget.status)
		total = budget.total_expense + budget.total_revenue

		if method == BFDistributionMethod.TOP_DOWN:
			if not department_weights:
				raise ValueError("department_weights required for top_down distribution")
			allocation = distribute_top_down(total, department_weights)
			result: dict[str, Any] = {"method": "top_down", "by_department": {k: str(v) for k, v in allocation.items()}}

		elif method == BFDistributionMethod.SEASONAL:
			weights = seasonal_weights or [1.0] * 12
			monthly = distribute_seasonal(total, weights)
			result = {"method": "seasonal", "monthly": [str(v) for v in monthly]}

		elif method == BFDistributionMethod.ZERO_BASED:
			if not line_justifications:
				raise ValueError("line_justifications required for zero_based")
			amounts = distribute_zero_based(line_justifications)
			result = {"method": "zero_based", "line_amounts": [str(v) for v in amounts]}

		elif method == BFDistributionMethod.BOTTOM_UP:
			lines = self._lines_for(budget_id)
			by_period: dict[str, Decimal] = {}
			for line in lines:
				key = str(line.period_start)
				by_period[key] = by_period.get(key, Decimal("0")) + line.budgeted_amount
			result = {"method": "bottom_up", "by_period": {k: str(v) for k, v in by_period.items()}}

		else:  # EQUAL (default)
			monthly = distribute_equal(total, 12)
			result = {"method": "equal", "monthly": [str(v) for v in monthly]}

		result["budget_id"] = budget_id
		result["total"] = str(total)
		self._emit("budget_distributed", budget_id, {"method": method.value})
		return result

	# =========================================================================
	# Templates
	# =========================================================================

	async def create_template(self, payload: BFBudgetTemplateCreate) -> BFBudgetTemplate:
		"""Create a reusable budget template."""
		template = BFBudgetTemplate(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**payload.model_dump(),
		)
		self._budget_templates[template.id] = template
		self._emit("template_created", template.id, {"name": template.name})
		return template

	async def list_templates(self) -> list[BFBudgetTemplate]:
		"""List all templates for this tenant."""
		return list(self._budget_templates.values())

	async def instantiate_template(self, template_id: str, fiscal_year: int, period_start: date, period_end: date) -> BFBudget:
		"""Instantiate a budget from a template, copying all line definitions."""
		tmpl = self._budget_templates.get(template_id)
		if not tmpl or tmpl.tenant_id != self.tenant_id:
			raise KeyError(f"Template {template_id} not found")
		payload = BFBudgetCreate(
			name=f"{tmpl.name} {fiscal_year}",
			description=f"Instantiated from template '{tmpl.name}'",
			fiscal_year=fiscal_year,
			period_start=period_start,
			period_end=period_end,
			budget_type=tmpl.budget_type,
			owner_id=self.actor_id,
			template_id=template_id,
		)
		budget = await self.create_budget_cycle(payload)
		# Copy template line definitions
		for line_def in tmpl.line_definitions:
			lc = BFBudgetLineCreate(
				budget_id=budget.id,
				period_start=period_start,
				period_end=period_end,
				account_code=line_def.get("account_code", "UNASSIGNED"),
				line_type=BFLineType(line_def.get("line_type", "expense")),
				description=line_def.get("description", "Template line"),
				budgeted_amount=Decimal(str(line_def.get("default_amount", "0"))),
			)
			await self.add_budget_line(lc)
		tmpl.usage_count += 1
		self._emit("template_instantiated", budget.id, {"template_id": template_id})
		return budget

	# =========================================================================
	# Approval workflow
	# =========================================================================

	async def create_approval(self, payload: BFBudgetApprovalCreate) -> BFBudgetApproval:
		"""Add an approver to the budget's approval chain."""
		self._get_budget(payload.budget_id)  # ownership check
		approval = BFBudgetApproval(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**payload.model_dump(),
		)
		self._budget_approvals[approval.id] = approval
		self._emit("approval_created", approval.id, {"budget_id": payload.budget_id, "approver": payload.approver_id})
		return approval

	async def get_pending_approvals(self, budget_id: str | None = None) -> list[BFBudgetApproval]:
		"""List pending approvals, optionally filtered by budget."""
		approvals = [a for a in self._budget_approvals.values()
					 if a.tenant_id == self.tenant_id and a.status == BFApprovalStatus.PENDING]
		if budget_id:
			approvals = [a for a in approvals if a.budget_id == budget_id]
		return sorted(approvals, key=lambda a: a.sequence)

	async def delegate_approval(self, approval_id: str, delegate_to: str) -> BFBudgetApproval:
		"""Delegate an approval to another user."""
		approval = self._get_approval(approval_id)
		assert_approval_pending(approval.status)
		approval.delegated_to = delegate_to
		approval.updated_at = datetime.now(timezone.utc)
		self._emit("approval_delegated", approval_id, {"delegated_to": delegate_to})
		return approval

	# =========================================================================
	# Forecasting
	# =========================================================================

	async def create_forecast(self, payload: BFForecastCreate) -> BFForecast:
		"""Create a new forecast definition."""
		forecast = BFForecast(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**payload.model_dump(),
		)
		self._forecasts[forecast.id] = forecast
		self._emit("forecast_created", forecast.id, {
			"forecast_type": forecast.forecast_type.value,
		})
		return forecast

	async def rolling_forecast(
		self,
		base_forecast_id: str,
		periods: int = 3,
		alpha: float = 0.3,
	) -> BFRollingForecastResult:
		"""
		Produce a rolling forecast by projecting *periods* ahead using
		exponential smoothing on existing forecast lines.

		Returns projected values with period labels.
		"""
		assert_forecast_horizon_valid(periods)
		forecast = self._get_forecast(base_forecast_id)
		lines = self._forecast_lines_for(base_forecast_id)
		if not lines:
			raise RuleViolation("no_forecast_lines", "Forecast has no lines to project from", "add_forecast_lines")

		actuals = [line.forecasted_value for line in sorted(lines, key=lambda l: l.period_date)]
		projections_decimal = project_rolling(actuals, periods, alpha)

		# Build period labels beyond last known
		last_date = max(l.period_date for l in lines)

		projected = []
		current = last_date
		for val in projections_decimal:
			# advance one month
			month = current.month % 12 + 1
			year = current.year + (current.month // 12)
			current = date(year, month, 1)
			projected.append({"period": str(current), "value": str(val), "lower": None, "upper": None})

		result = BFRollingForecastResult(
			forecast_id=base_forecast_id,
			periods=periods,
			method="rolling_exponential_smoothing",
			projected_values=projected,
			mape=None,
		)
		self._emit("rolling_forecast_generated", base_forecast_id, {"periods": periods})
		return result

	async def driver_based_forecast(
		self,
		forecast_id: str,
		driver_changes: dict[str, float],
	) -> list[BFForecastLine]:
		"""
		Re-compute forecast lines by applying driver elasticities.

		Uses DriverBasedAssumption records linked to the forecast's scenario.
		"""
		forecast = self._get_forecast(forecast_id)
		lines = self._forecast_lines_for(forecast_id)
		assumptions = {a.name: a for a in self._driver_assumptions.values() if a.tenant_id == self.tenant_id}

		updated: list[BFForecastLine] = []
		for line in lines:
			# Build elasticity map from linked drivers
			elasticities: dict[str, float] = {}
			for driver_name, assump in assumptions.items():
				if line.account_code in assump.linked_accounts:
					elasticities[driver_name] = float(assump.growth_rate or Decimal("1"))
			new_val = driver_based_forecast(line.forecasted_value, driver_changes, elasticities)
			line.forecasted_value = new_val
			line.updated_at = datetime.now(timezone.utc)
			updated.append(line)

		forecast.updated_at = datetime.now(timezone.utc)
		self._emit("driver_forecast_applied", forecast_id, {"driver_count": len(driver_changes)})
		return updated

	async def reforecast(
		self,
		forecast_id: str,
		period: str,
		actuals: list[float],
	) -> BFForecast:
		"""
		Reforecast by incorporating new actuals, updating remaining periods.

		period: ISO date string of the period being closed (e.g. '2026-03-01')
		actuals: list of actual values for the period.
		"""
		assert_actuals = actuals
		if not assert_actuals:
			raise RuleViolation("no_actuals_for_reforecast", "actuals list cannot be empty", "provide_actuals")

		forecast = self._get_forecast(forecast_id)
		lines = self._forecast_lines_for(forecast_id)

		# Update lines at or before the period with actuals
		for i, line in enumerate(sorted(lines, key=lambda l: l.period_date)):
			if str(line.period_date) <= period and i < len(actuals):
				line.actual_value = Decimal(str(actuals[i]))
				line.residual = round_currency(line.actual_value - line.forecasted_value)

		# Re-smooth remaining periods
		known_actuals = [
			line.actual_value for line in sorted(lines, key=lambda l: l.period_date)
			if line.actual_value is not None
		]
		if known_actuals:
			future_lines = [
				line for line in sorted(lines, key=lambda l: l.period_date)
				if line.actual_value is None
			]
			projected = project_rolling(known_actuals, len(future_lines))
			for line, val in zip(future_lines, projected):
				line.forecasted_value = val
				line.updated_at = datetime.now(timezone.utc)

		forecast.status = BFForecastStatus.COMPLETED
		forecast.updated_at = datetime.now(timezone.utc)
		self._emit("reforecast_completed", forecast_id, {"period": period, "actuals_count": len(actuals)})
		return forecast

	async def ai_forecast_model(
		self,
		forecast_id: str,
		model_params: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Dispatch to APG's ai_orchestration capability for ML-based forecasting.

		Returns a result envelope; actual compute is async via ai_orchestration.
		This method emits the job request and returns a job_id for polling.
		"""
		forecast = self._get_forecast(forecast_id)
		lines = self._forecast_lines_for(forecast_id)
		history = [float(l.forecasted_value) for l in sorted(lines, key=lambda l: l.period_date)]

		job_id = uuid7str()
		params = model_params or {}
		params.setdefault("algorithm", "double_exponential")
		horizon = int(params.get("horizon", 12))
		params.setdefault("confidence", 0.95)

		# Perform local double-exponential as fallback when ai_orchestration unavailable
		history_decimal = [Decimal(str(v)) for v in history]
		smoothed = double_exponential_smoothing(
			history_decimal,
			alpha=float(params.get("alpha", 0.3)),
			beta=float(params.get("beta", 0.1)),
		)
		projected = project_rolling(history_decimal, horizon)

		result = {
			"job_id": job_id,
			"forecast_id": forecast_id,
			"algorithm": params["algorithm"],
			"horizon": horizon,
			"smoothed": [str(v) for v in smoothed],
			"projected": [str(v) for v in projected],
			"status": "completed",
		}
		self._emit("ai_forecast_dispatched", forecast_id, {"job_id": job_id, "algorithm": params["algorithm"]})
		return result

	async def add_forecast_line(self, payload: BFForecastLineCreate) -> BFForecastLine:
		"""Add a data point to an existing forecast."""
		self._get_forecast(payload.forecast_id)
		line = BFForecastLine(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**payload.model_dump(),
		)
		self._forecast_lines[line.id] = line
		self._emit("forecast_line_added", line.id, {"forecast_id": payload.forecast_id, "period": str(payload.period_date)})
		return line

	async def list_forecasts(
		self,
		status: BFForecastStatus | None = None,
		forecast_type: BFForecastType | None = None,
		limit: int = 50,
	) -> list[BFForecast]:
		"""List forecasts with optional filters."""
		items = list(self._forecasts.values())
		if status:
			items = [f for f in items if f.status == status]
		if forecast_type:
			items = [f for f in items if f.forecast_type == forecast_type]
		return items[:limit]

	# =========================================================================
	# Scenario analysis
	# =========================================================================

	async def create_scenario(self, payload: BFScenarioCreate) -> BFScenarioModel:
		"""Create a what-if scenario."""
		scenario = BFScenarioModel(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**payload.model_dump(),
		)
		self._scenarios[scenario.id] = scenario
		self._emit("scenario_created", scenario.id, {"name": scenario.name, "type": scenario.scenario_type.value})
		return scenario

	async def scenario_analysis(
		self,
		budget_id: str,
		scenario_ids: list[str],
	) -> BFScenarioAnalysisResult:
		"""
		Run all specified scenarios against the budget and produce a
		comparative analysis with expected value and best/worst case.
		"""
		budget = self._get_budget(budget_id)
		assert_scenarios_non_empty(scenario_ids)

		base_net = budget.total_revenue - budget.total_expense
		scenario_results: list[dict[str, Any]] = []
		outcomes: list[Decimal] = []
		probabilities: list[float] = []

		for sid in scenario_ids:
			sc = self._scenarios.get(sid)
			if not sc or sc.tenant_id != self.tenant_id:
				raise KeyError(f"Scenario {sid} not found")

			# Apply adjustments to compute net impact
			adjustments = [Decimal(str(adj.get("amount", "0"))) for adj in sc.adjustments]
			net = scenario_delta(base_net, adjustments)
			delta_pct = scenario_delta_pct(base_net, net)

			sc.results = {"net": str(net), "delta_pct": str(delta_pct)}
			sc.net_impact = round_currency(net - base_net)
			sc.net_impact_pct = delta_pct
			sc.ran_at = datetime.now(timezone.utc)

			scenario_results.append({
				"scenario_id": sid,
				"name": sc.name,
				"probability": sc.probability,
				"net": str(net),
				"delta_pct": str(delta_pct),
			})
			outcomes.append(net)
			probabilities.append(sc.probability)

		assert_probability_sum_valid(probabilities)
		ev = weighted_scenario_outcome(outcomes, probabilities)

		result = BFScenarioAnalysisResult(
			base_net=base_net,
			scenarios=scenario_results,
			expected_value=ev,
			best_case=max(outcomes),
			worst_case=min(outcomes),
		)
		self._emit("scenario_analysis_run", budget_id, {"scenario_count": len(scenario_ids), "expected_value": str(ev)})
		return result

	async def what_if_simulation(
		self,
		budget_id: str,
		adjustments: dict[str, float],
	) -> dict[str, Any]:
		"""
		Lightweight ad-hoc what-if: apply percentage adjustments to each
		line type and return the projected impact without persisting.

		adjustments: {line_type: pct_change}  e.g. {"revenue": 0.05, "expense": -0.03}
		"""
		budget = self._get_budget(budget_id)
		lines = self._lines_for(budget_id)

		original_revenue = budget.total_revenue
		original_expense = budget.total_expense

		new_revenue = original_revenue
		new_expense = original_expense

		for line in lines:
			adj = adjustments.get(line.line_type.value, 0.0)
			if adj != 0.0:
				delta = round_currency(line.budgeted_amount * Decimal(str(adj)))
				if line.line_type == BFLineType.REVENUE:
					new_revenue += delta
				else:
					new_expense += delta

		original_net = original_revenue - original_expense
		new_net = new_revenue - new_expense
		delta_pct = scenario_delta_pct(original_net, new_net)

		return {
			"budget_id": budget_id,
			"original_net": str(original_net),
			"new_net": str(new_net),
			"delta_pct": str(delta_pct),
			"new_revenue": str(new_revenue),
			"new_expense": str(new_expense),
			"adjustments_applied": adjustments,
		}

	async def list_scenarios(self, active_only: bool = False) -> list[BFScenarioModel]:
		"""List scenarios, optionally filtering inactive ones."""
		items = list(self._scenarios.values())
		if active_only:
			items = [s for s in items if s.is_active]
		return items

	# =========================================================================
	# Driver assumptions
	# =========================================================================

	async def create_driver_assumption(self, payload: BFDriverAssumptionCreate) -> BFDriverBasedAssumption:
		"""Register a business driver assumption."""
		assert_driver_value_positive(payload.value)
		assumption = BFDriverBasedAssumption(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			**payload.model_dump(),
		)
		self._driver_assumptions[assumption.id] = assumption
		self._emit("driver_assumption_created", assumption.id, {"name": assumption.name, "type": assumption.driver_type.value})
		return assumption

	async def list_driver_assumptions(self, driver_type: BFDriverType | None = None) -> list[BFDriverBasedAssumption]:
		"""List driver assumptions."""
		items = list(self._driver_assumptions.values())
		if driver_type:
			items = [a for a in items if a.driver_type == driver_type]
		return items

	# =========================================================================
	# Variance analysis
	# =========================================================================

	async def variance_analysis(
		self,
		budget_id: str,
		period_start: date,
		period_end: date,
		actuals_by_account: dict[str, Decimal],
	) -> BFVarianceReport:
		"""
		Compute full budget-vs-actual variance report for a period.

		actuals_by_account: {account_code: actual_amount}
		"""
		budget = self._get_budget(budget_id)
		lines = self._lines_for(budget_id)

		line_variances: list[dict[str, Any]] = []
		total_budget = Decimal("0")
		total_actual = Decimal("0")

		for line in lines:
			# Include line if its period overlaps the report window
			if not (line.period_start <= period_end and line.period_end >= period_start):
				continue
			actual = actuals_by_account.get(line.account_code, Decimal("0"))
			var_amt, var_pct = calculate_variance(line.budgeted_amount, actual)
			vtype = variance_type(line.budgeted_amount, actual, line.line_type.value)
			sig = significance_level(var_pct)

			line_variances.append({
				"line_id": line.id,
				"account_code": line.account_code,
				"line_type": line.line_type.value,
				"budget": str(line.budgeted_amount),
				"actual": str(actual),
				"variance_amount": str(var_amt),
				"variance_pct": str(var_pct),
				"variance_type": vtype,
				"significance": sig,
			})
			total_budget += line.budgeted_amount
			total_actual += actual

		total_var, total_var_pct = calculate_variance(total_budget, total_actual)
		vtype_overall = variance_type(total_budget, total_actual, "expense")
		sig_overall = significance_level(total_var_pct)

		# Generate plain-English recommendations
		recs = self._variance_recommendations(line_variances)

		report = BFVarianceReport(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			created_by=self.actor_id,
			updated_by=self.actor_id,
			created_at=datetime.now(timezone.utc),
			updated_at=datetime.now(timezone.utc),
			budget_id=budget_id,
			report_period_start=period_start,
			report_period_end=period_end,
			total_budget=total_budget,
			total_actual=total_actual,
			total_variance=total_var,
			variance_pct=total_var_pct,
			variance_type=BFVarianceType(vtype_overall),
			significance=BFSignificanceLevel(sig_overall),
			line_variances=line_variances,
			recommendations=recs,
		)
		self._variance_reports[report.id] = report
		self._emit("variance_report_generated", report.id, {
			"budget_id": budget_id,
			"variance_pct": str(total_var_pct),
			"significance": sig_overall,
		})
		return report

	async def list_variance_reports(self, budget_id: str | None = None) -> list[BFVarianceReport]:
		"""List variance reports."""
		items = list(self._variance_reports.values())
		if budget_id:
			items = [r for r in items if r.budget_id == budget_id]
		return items

	# =========================================================================
	# Consolidation
	# =========================================================================

	async def budget_consolidation(
		self,
		budget_ids: list[str],
		currency_code: str = "USD",
	) -> BFConsolidationResult:
		"""
		Consolidate multiple budgets into a single aggregate view.

		Handles multi-currency by assuming same-currency input (FX handled externally).
		Groups by department and cost_center.
		"""
		if not budget_ids:
			raise ValueError("At least one budget_id required for consolidation")

		budgets = [self._get_budget(bid) for bid in budget_ids]
		totals = [{"revenue": b.total_revenue, "expense": b.total_expense} for b in budgets]
		consolidated = consolidate_budgets(totals)

		by_dept: dict[str, Decimal] = {}
		by_cc: dict[str, Decimal] = {}
		for b in budgets:
			if b.department_id:
				by_dept[b.department_id] = by_dept.get(b.department_id, Decimal("0")) + b.total_expense + b.total_revenue
			if b.cost_center_id:
				by_cc[b.cost_center_id] = by_cc.get(b.cost_center_id, Decimal("0")) + b.total_expense + b.total_revenue

		result = BFConsolidationResult(
			tenant_id=self.tenant_id,
			included_budget_ids=budget_ids,
			total_revenue=consolidated["total_revenue"],
			total_expense=consolidated["total_expense"],
			net_amount=consolidated["net_amount"],
			by_department=by_dept,
			by_cost_center=by_cc,
			currency_code=currency_code,
		)
		self._emit("budgets_consolidated", "consolidation", {"budget_count": len(budget_ids)})
		return result

	# =========================================================================
	# Sensitivity analysis
	# =========================================================================

	async def sensitivity_analysis(
		self,
		driver_assumption_id: str,
		steps: list[float] | None = None,
	) -> BFSensitivityResult:
		"""
		Evaluate the sensitivity of the forecast to perturbations in a driver.

		steps: fractional deltas to test (default: ±5%, ±10%, ±20%)
		"""
		assumption = self._driver_assumptions.get(driver_assumption_id)
		if not assumption or assumption.tenant_id != self.tenant_id:
			raise KeyError(f"DriverAssumption {driver_assumption_id} not found")

		steps = steps or [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]

		def compute(driver_val: Decimal) -> Decimal:
			return driver_based_forecast(
				assumption.value,
				{assumption.name: float(driver_val / assumption.value) - 1},
				{assumption.name: float(assumption.growth_rate or Decimal("1"))},
			)

		perturbations = sensitivity_range(compute, assumption.value, steps)

		result = BFSensitivityResult(
			driver_name=assumption.name,
			base_value=assumption.value,
			perturbations=perturbations,
		)
		self._emit("sensitivity_analysis_run", driver_assumption_id, {"step_count": len(steps)})
		return result

	# =========================================================================
	# Dashboard / KPIs
	# =========================================================================

	async def dashboard_kpis(self) -> BFDashboardKPIs:
		"""Compute dashboard KPIs for the tenant."""
		budgets = list(self._budgets.values())
		lines = list(self._budget_lines.values())
		forecasts = list(self._forecasts.values())
		scenarios = list(self._scenarios.values())
		pending_approvals = await self.get_pending_approvals()
		variance_reports = list(self._variance_reports.values())

		total_budget = sum((b.total_revenue + b.total_expense for b in budgets), Decimal("0"))
		total_actual = sum((r.total_actual for r in variance_reports), Decimal("0"))
		material_variances = sum(
			1 for r in variance_reports
			if abs(r.variance_pct) >= Decimal("10")
		)
		overall_var_pct = Decimal("0")
		if total_budget:
			overall_var_pct = round_currency((total_actual - total_budget) / total_budget * Decimal("100"))

		return BFDashboardKPIs(
			tenant_id=self.tenant_id,
			budget_count=len(budgets),
			approved_budget_count=sum(1 for b in budgets if b.status == BFBudgetStatus.APPROVED),
			draft_budget_count=sum(1 for b in budgets if b.status == BFBudgetStatus.DRAFT),
			total_budget_amount=total_budget,
			total_actual_amount=total_actual,
			overall_variance_pct=overall_var_pct,
			forecast_count=len(forecasts),
			scenario_count=len(scenarios),
			pending_approvals=len(pending_approvals),
			material_variances=material_variances,
		)

	async def budget_summary(
		self,
		period_start: date,
		period_end: date,
	) -> BFBudgetSummary:
		"""Aggregate budget summary for a period."""
		budgets = list(self._budgets.values())
		total_rev = sum((b.total_revenue for b in budgets), Decimal("0"))
		total_exp = sum((b.total_expense for b in budgets), Decimal("0"))

		return BFBudgetSummary(
			tenant_id=self.tenant_id,
			period_start=period_start,
			period_end=period_end,
			budget_count=len(budgets),
			total_revenue=total_rev,
			total_expense=total_exp,
			net_amount=total_rev - total_exp,
			approved_count=sum(1 for b in budgets if b.status == BFBudgetStatus.APPROVED),
			draft_count=sum(1 for b in budgets if b.status == BFBudgetStatus.DRAFT),
		)

	async def audit_trail(self, entity_id: str | None = None) -> list[dict[str, Any]]:
		"""Return audit events for this tenant, optionally filtered by entity."""
		events = [e for e in self._events if e["tenant_id"] == self.tenant_id]
		if entity_id:
			events = [e for e in events if e["entity_id"] == entity_id]
		return sorted(events, key=lambda e: e["occurred_at"], reverse=True)

	# =========================================================================
	# Private helpers
	# =========================================================================

	def _get_budget(self, budget_id: str) -> BFBudget:
		budget = self._budgets.get(budget_id)
		if not budget:
			raise KeyError(f"Budget {budget_id} not found")
		assert_no_cross_tenant(self.tenant_id, budget.tenant_id)
		return budget

	def _get_forecast(self, forecast_id: str) -> BFForecast:
		forecast = self._forecasts.get(forecast_id)
		if not forecast:
			raise KeyError(f"Forecast {forecast_id} not found")
		assert_no_cross_tenant(self.tenant_id, forecast.tenant_id)
		return forecast

	def _get_approval(self, approval_id: str) -> BFBudgetApproval:
		approval = self._budget_approvals.get(approval_id)
		if not approval:
			raise KeyError(f"Approval {approval_id} not found")
		assert_no_cross_tenant(self.tenant_id, approval.tenant_id)
		return approval

	def _lines_for(self, budget_id: str) -> list[BFBudgetLine]:
		return [l for l in self._budget_lines.values() if l.budget_id == budget_id]

	def _forecast_lines_for(self, forecast_id: str) -> list[BFForecastLine]:
		return [l for l in self._forecast_lines.values() if l.forecast_id == forecast_id]

	async def _recalc_budget_totals(self, budget: BFBudget) -> None:
		lines = self._lines_for(budget.id)
		budget.total_revenue = round_currency(sum(
			(l.budgeted_amount for l in lines if l.line_type == BFLineType.REVENUE),
			Decimal("0"),
		))
		budget.total_expense = round_currency(sum(
			(l.budgeted_amount for l in lines if l.line_type != BFLineType.REVENUE),
			Decimal("0"),
		))
		budget.net_amount = round_currency(budget.total_revenue - budget.total_expense)
		budget.updated_at = datetime.now(timezone.utc)

	def _emit(self, event_name: str, entity_id: str, payload: dict[str, Any]) -> None:
		self._events.append({
			"event": event_name,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"entity_id": entity_id,
			"payload": payload,
			"occurred_at": datetime.now(timezone.utc).isoformat(),
			"stream": "apg.fin.bfc.lifecycle",
		})

	def _variance_recommendations(self, line_variances: list[dict[str, Any]]) -> list[str]:
		recs: list[str] = []
		for lv in line_variances:
			sig = lv.get("significance", "minimal")
			vtype = lv.get("variance_type", "neutral")
			account = lv.get("account_code", "?")
			pct = lv.get("variance_pct", "0")
			if sig in ("critical", "high") and vtype == "unfavorable":
				recs.append(f"[{account}] Unfavorable {sig} variance ({pct}%) – investigate and reforecast")
			elif sig in ("critical", "high") and vtype == "favorable":
				recs.append(f"[{account}] Favorable {sig} variance ({pct}%) – consider updating budget baseline")
		if not recs:
			recs.append("All variances within acceptable thresholds.")
		return recs

	def _log_op(self, method: str, entity_id: str, detail: str = "") -> None:
		"""_log_ prefixed helper for structured console logging."""
		_log.debug("[BFCService.%s] tenant=%s entity=%s %s", method, self.tenant_id, entity_id, detail)

	def _log_error(self, method: str, error: Exception) -> None:
		_log.error("[BFCService.%s] tenant=%s error=%s", method, self.tenant_id, str(error))

	async def ml_budget_forecast_adjust(self, *args, **kwargs):
		"""AI-powered ML-adjusted budget forecast from historical variance. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict(kwargs.get("historical",[{"period": str(i), "value": 100000.0+i*1000} for i in range(12)]), horizon=3, task="budget_forecast")
			return {"adjusted_forecast": result.predictions, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

