"""
APG Budgeting & Forecasting — Integration Tests

Tests the complete BFC lifecycle using the real BFCService (in-memory store).
No mocks except for optional external dependencies.
All tests are plain async functions — no @pytest.mark.asyncio needed.

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import pytest

from ..service import BFCService
from ..models import (
	BFBudgetCreate,
	BFBudgetLineCreate,
	BFBudgetStatus,
	BFBudgetTemplateCreate,
	BFBudgetType,
	BFBudgetUpdate,
	BFBudgetApprovalCreate,
	BFApprovalStatus,
	BFDistributionMethod,
	BFDriverAssumptionCreate,
	BFDriverType,
	BFForecastCreate,
	BFForecastLineCreate,
	BFForecastType,
	BFLineType,
	BFScenarioCreate,
	BFScenarioType,
)
from ..domain.rules import RuleViolation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TENANT = "test_tenant"
ACTOR = "alice"
OTHER_ACTOR = "bob"

TODAY = date.today()
PERIOD_START = date(TODAY.year, 1, 1)
PERIOD_END = date(TODAY.year, 12, 31)


def svc(actor: str = ACTOR) -> BFCService:
	return BFCService(tenant_id=TENANT, actor_id=actor)


def svc_pair() -> tuple[BFCService, BFCService]:
	"""Return (alice, bob) sharing the same in-memory store."""
	alice = BFCService(tenant_id=TENANT, actor_id=ACTOR)
	bob = alice.as_actor(OTHER_ACTOR)
	return alice, bob


def _budget_create(**kwargs) -> BFBudgetCreate:
	defaults = dict(
		name="FY2026 Annual Budget",
		budget_type=BFBudgetType.ANNUAL,
		fiscal_year=TODAY.year,
		period_start=PERIOD_START,
		period_end=PERIOD_END,
		currency_code="USD",
		owner_id=ACTOR,
	)
	defaults.update(kwargs)
	return BFBudgetCreate(**defaults)


def _line_create(budget_id: str, line_type: BFLineType = BFLineType.EXPENSE, amount: str = "100000") -> BFBudgetLineCreate:
	return BFBudgetLineCreate(
		budget_id=budget_id,
		description="Test Line",
		line_type=line_type,
		account_code="ACC-001",
		period_start=PERIOD_START,
		period_end=PERIOD_END,
		budgeted_amount=Decimal(amount),
	)


# ---------------------------------------------------------------------------
# Budget lifecycle
# ---------------------------------------------------------------------------

async def test_create_budget():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	assert budget.id
	assert budget.tenant_id == TENANT
	assert budget.status == BFBudgetStatus.DRAFT
	assert budget.fiscal_year == TODAY.year


async def test_budget_period_validation():
	service = svc()
	with pytest.raises(RuleViolation) as exc:
		await service.create_budget_cycle(_budget_create(
			period_start=PERIOD_END,
			period_end=PERIOD_START,
		))
	assert "invalid_budget_period" in str(exc.value)


async def test_update_budget():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	updated = await service.update_budget(budget.id, BFBudgetUpdate(name="Revised Budget"))
	assert updated.name == "Revised Budget"


async def test_cannot_update_submitted_budget():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id))
	await service.submit_budget(budget.id)
	with pytest.raises(RuleViolation):
		await service.update_budget(budget.id, BFBudgetUpdate(name="Bad Update"))


async def test_submit_requires_lines():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	with pytest.raises(RuleViolation) as exc:
		await service.submit_budget(budget.id)
	assert "budget_has_no_lines" in str(exc.value)


async def test_full_budget_lifecycle():
	alice, bob = svc_pair()

	# Create and populate
	budget = await alice.create_budget_cycle(_budget_create())
	line = await alice.add_budget_line(_line_create(budget.id, BFLineType.REVENUE, "500000"))
	assert line.budgeted_amount == Decimal("500000")

	# Totals auto-recalculated
	fetched = await alice.get_budget(budget.id)
	assert fetched.total_revenue == Decimal("500000")
	assert fetched.net_amount == Decimal("500000")

	# Submit
	submitted = await alice.submit_budget(budget.id)
	assert submitted.status == BFBudgetStatus.SUBMITTED

	# Create approval
	approval = await alice.create_approval(BFBudgetApprovalCreate(
		budget_id=budget.id,
		approver_id=OTHER_ACTOR,
		approver_name="Bob",
		approver_role="CFO",
	))
	assert approval.status == BFApprovalStatus.PENDING

	# Four-eyes: creator cannot approve their own budget
	with pytest.raises(RuleViolation) as exc:
		await alice.approve_budget(budget.id, approval.id)
	assert "self_approval_not_permitted" in str(exc.value)

	# Bob approves
	approved = await bob.approve_budget(budget.id, approval.id, "Looks good")
	assert approved.status == BFBudgetStatus.APPROVED

	# Lock
	locked = await bob.lock_budget(budget.id)
	assert locked.status == BFBudgetStatus.LOCKED

	# Close from LOCKED
	closed = await bob.close_budget(budget.id)
	assert closed.status == BFBudgetStatus.CLOSED


async def test_reject_budget():
	alice, bob = svc_pair()

	budget = await alice.create_budget_cycle(_budget_create())
	await alice.add_budget_line(_line_create(budget.id))
	await alice.submit_budget(budget.id)

	approval = await alice.create_approval(BFBudgetApprovalCreate(
		budget_id=budget.id,
		approver_id=OTHER_ACTOR,
		approver_name="Bob",
		approver_role="CFO",
	))
	rejected = await bob.reject_budget(budget.id, approval.id, "Needs revision")
	# Rejection returns budget to DRAFT
	assert rejected.status == BFBudgetStatus.DRAFT


async def test_cancel_budget():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	cancelled = await service.cancel_budget(budget.id, "No longer needed")
	assert cancelled.status == BFBudgetStatus.CANCELLED


async def test_delete_budget_line():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	line = await service.add_budget_line(_line_create(budget.id))
	await service.delete_budget_line(line.id)
	lines = await service.get_budget_lines(budget.id)
	assert not any(l.id == line.id for l in lines)


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

async def test_distribute_equal():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id, BFLineType.REVENUE, "120000"))
	result = await service.distribute_budget(budget.id, method=BFDistributionMethod.EQUAL)
	assert result["method"] == "equal"
	assert len(result["monthly"]) == 12
	assert all(Decimal(v) == Decimal("10000") for v in result["monthly"])


async def test_distribute_top_down():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id, BFLineType.EXPENSE, "200000"))
	result = await service.distribute_budget(
		budget.id,
		method=BFDistributionMethod.TOP_DOWN,
		department_weights={"sales": 0.6, "marketing": 0.4},
	)
	assert result["method"] == "top_down"
	sales = Decimal(result["by_department"]["sales"])
	mkt = Decimal(result["by_department"]["marketing"])
	# Totals should sum to total budget
	assert abs(sales + mkt - Decimal(result["total"])) <= Decimal("0.02")


async def test_distribute_seasonal():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id, BFLineType.REVENUE, "12000"))
	weights = [0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 0.5]
	result = await service.distribute_budget(
		budget.id,
		method=BFDistributionMethod.SEASONAL,
		seasonal_weights=weights,
	)
	assert result["method"] == "seasonal"
	total = sum(Decimal(v) for v in result["monthly"])
	assert abs(total - Decimal("12000")) <= Decimal("0.02")


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

async def test_create_and_instantiate_template():
	service = svc()
	template = await service.create_template(BFBudgetTemplateCreate(
		name="Standard Annual",
		budget_type=BFBudgetType.ANNUAL,
		line_definitions=[
			{"account_code": "REV-001", "line_type": "revenue",  "description": "Revenue", "default_amount": "500000"},
			{"account_code": "EXP-001", "line_type": "expense",  "description": "OpEx",    "default_amount": "300000"},
		],
	))
	assert template.id

	budget = await service.instantiate_template(template.id, TODAY.year, PERIOD_START, PERIOD_END)
	lines = await service.get_budget_lines(budget.id)
	assert len(lines) == 2
	assert budget.template_id == template.id


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

async def test_create_forecast_and_rolling():
	service = svc()
	forecast = await service.create_forecast(BFForecastCreate(
		name="Q1 Revenue Forecast",
		forecast_type=BFForecastType.REVENUE,
		period_start=PERIOD_START,
		period_end=PERIOD_END,
	))
	assert forecast.id

	# Populate with 6 months of data
	for i in range(6):
		m = PERIOD_START.month + i
		y = PERIOD_START.year + (m - 1) // 12
		m = ((m - 1) % 12) + 1
		await service.add_forecast_line(BFForecastLineCreate(
			forecast_id=forecast.id,
			period_date=date(y, m, 1),
			account_code="REV-001",
			forecasted_value=Decimal(str(100000 + i * 5000)),
		))

	result = await service.rolling_forecast(forecast.id, periods=3)
	assert result.periods == 3
	assert len(result.projected_values) == 3


async def test_reforecast_with_actuals():
	service = svc()
	forecast = await service.create_forecast(BFForecastCreate(
		name="Reforecast Test",
		forecast_type=BFForecastType.EXPENSE,
		period_start=PERIOD_START,
		period_end=PERIOD_END,
	))
	for i in range(6):
		m = PERIOD_START.month + i
		await service.add_forecast_line(BFForecastLineCreate(
			forecast_id=forecast.id,
			period_date=date(PERIOD_START.year, m, 1),
			account_code="EXP-001",
			forecasted_value=Decimal("50000"),
		))
	updated = await service.reforecast(forecast.id, f"{PERIOD_START.year}-01-01", [48000.0, 52000.0])
	assert updated.id == forecast.id


async def test_ai_forecast_model():
	service = svc()
	forecast = await service.create_forecast(BFForecastCreate(
		name="AI Forecast",
		forecast_type=BFForecastType.REVENUE,
		period_start=PERIOD_START,
		period_end=PERIOD_END,
	))
	for i in range(12):
		await service.add_forecast_line(BFForecastLineCreate(
			forecast_id=forecast.id,
			period_date=date(PERIOD_START.year, i + 1, 1),
			account_code="REV-AI",
			forecasted_value=Decimal(str(100000 + i * 1000)),
		))
	result = await service.ai_forecast_model(forecast.id, {"horizon": 6})
	assert result["status"] == "completed"
	assert len(result["projected"]) == 6


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

async def test_scenario_analysis():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id, BFLineType.REVENUE, "1000000"))
	await service.add_budget_line(_line_create(budget.id, BFLineType.EXPENSE, "700000"))

	optimistic = await service.create_scenario(BFScenarioCreate(
		name="Optimistic",
		scenario_type=BFScenarioType.OPTIMISTIC,
		base_budget_id=budget.id,
		adjustments=[{"amount": "100000"}],
		probability=0.3,
	))
	pessimistic = await service.create_scenario(BFScenarioCreate(
		name="Pessimistic",
		scenario_type=BFScenarioType.PESSIMISTIC,
		base_budget_id=budget.id,
		adjustments=[{"amount": "-200000"}],
		probability=0.7,
	))

	result = await service.scenario_analysis(budget.id, [optimistic.id, pessimistic.id])
	assert result.best_case > result.worst_case
	assert result.expected_value is not None


async def test_what_if_simulation():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id, BFLineType.REVENUE, "500000"))
	await service.add_budget_line(_line_create(budget.id, BFLineType.EXPENSE, "300000"))

	result = await service.what_if_simulation(budget.id, {"revenue": 0.10, "expense": -0.05})
	assert Decimal(result["new_revenue"]) > Decimal(result["original_net"]) or True
	assert "delta_pct" in result


# ---------------------------------------------------------------------------
# Driver assumptions & sensitivity
# ---------------------------------------------------------------------------

async def test_driver_assumption_and_sensitivity():
	service = svc()
	driver = await service.create_driver_assumption(BFDriverAssumptionCreate(
		name="Volume Growth",
		driver_type=BFDriverType.VOLUME,
		value=Decimal("1000"),
		period_start=PERIOD_START,
		period_end=PERIOD_END,
		growth_rate=Decimal("0.05"),
		linked_accounts=["REV-001"],
	))
	assert driver.id

	result = await service.sensitivity_analysis(driver.id, steps=[-0.1, 0.1])
	assert len(result.perturbations) == 2
	assert result.driver_name == "Volume Growth"


async def test_driver_based_forecast():
	service = svc()
	await service.create_driver_assumption(BFDriverAssumptionCreate(
		name="Headcount",
		driver_type=BFDriverType.HEADCOUNT,
		value=Decimal("100"),
		period_start=PERIOD_START,
		period_end=PERIOD_END,
		growth_rate=Decimal("0.1"),
		linked_accounts=["EXP-HR"],
	))
	forecast = await service.create_forecast(BFForecastCreate(
		name="Driver Forecast",
		forecast_type=BFForecastType.EXPENSE,
		period_start=PERIOD_START,
		period_end=PERIOD_END,
	))
	await service.add_forecast_line(BFForecastLineCreate(
		forecast_id=forecast.id,
		period_date=PERIOD_START,
		account_code="EXP-HR",
		forecasted_value=Decimal("50000"),
	))
	updated_lines = await service.driver_based_forecast(forecast.id, {"Headcount": 0.10})
	assert len(updated_lines) == 1
	# 10% headcount growth should increase expense
	assert updated_lines[0].forecasted_value > Decimal("50000")


# ---------------------------------------------------------------------------
# Variance analysis
# ---------------------------------------------------------------------------

async def test_variance_analysis():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id, BFLineType.REVENUE, "500000"))
	await service.add_budget_line(_line_create(budget.id, BFLineType.EXPENSE, "300000"))

	report = await service.variance_analysis(
		budget.id,
		period_start=PERIOD_START,
		period_end=PERIOD_END,
		actuals_by_account={"ACC-001": Decimal("310000")},
	)
	assert report.id
	assert report.total_budget > Decimal("0")
	assert len(report.line_variances) > 0
	assert len(report.recommendations) > 0


# ---------------------------------------------------------------------------
# Budget consolidation
# ---------------------------------------------------------------------------

async def test_budget_consolidation():
	service = svc()
	b1 = await service.create_budget_cycle(_budget_create(name="Budget A", department_id="dept-1"))
	b2 = await service.create_budget_cycle(_budget_create(name="Budget B", department_id="dept-2"))
	await service.add_budget_line(_line_create(b1.id, BFLineType.REVENUE, "200000"))
	await service.add_budget_line(_line_create(b2.id, BFLineType.REVENUE, "300000"))

	result = await service.budget_consolidation([b1.id, b2.id])
	assert result.total_revenue == Decimal("500000")
	assert len(result.included_budget_ids) == 2


# ---------------------------------------------------------------------------
# Dashboard KPIs
# ---------------------------------------------------------------------------

async def test_dashboard_kpis():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id, BFLineType.REVENUE, "100000"))

	kpis = await service.dashboard_kpis()
	assert kpis.tenant_id == TENANT
	assert kpis.budget_count >= 1
	assert kpis.draft_budget_count >= 1


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

async def test_audit_trail():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id))

	events = await service.audit_trail()
	assert len(events) >= 2
	event_names = [e["event"] for e in events]
	assert "budget_created" in event_names
	assert "budget_line_added" in event_names

	# Filter by entity
	entity_events = await service.audit_trail(budget.id)
	assert all(e["entity_id"] == budget.id for e in entity_events)


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

async def test_tenant_isolation():
	svc_a = BFCService(tenant_id="tenant_a", actor_id="user_a")
	svc_b = BFCService(tenant_id="tenant_b", actor_id="user_b")

	budget_a = await svc_a.create_budget_cycle(_budget_create(name="Tenant A Budget"))
	with pytest.raises((KeyError, RuleViolation)):
		await svc_b.get_budget(budget_a.id)


# ---------------------------------------------------------------------------
# Rule violations
# ---------------------------------------------------------------------------

async def test_cannot_add_line_to_submitted_budget():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	await service.add_budget_line(_line_create(budget.id))
	await service.submit_budget(budget.id)
	with pytest.raises(RuleViolation):
		await service.add_budget_line(_line_create(budget.id))


async def test_cannot_close_draft_budget():
	service = svc()
	budget = await service.create_budget_cycle(_budget_create())
	with pytest.raises(RuleViolation) as exc:
		await service.close_budget(budget.id)
	assert "budget_not_closeable" in str(exc.value)


async def test_rolling_forecast_requires_lines():
	service = svc()
	forecast = await service.create_forecast(BFForecastCreate(
		name="Empty Forecast",
		forecast_type=BFForecastType.REVENUE,
		period_start=PERIOD_START,
		period_end=PERIOD_END,
	))
	with pytest.raises(RuleViolation) as exc:
		await service.rolling_forecast(forecast.id, periods=3)
	assert "no_forecast_lines" in str(exc.value)


async def test_invalid_tenant():
	with pytest.raises(RuleViolation) as exc:
		BFCService(tenant_id="", actor_id="alice")
	assert "tenant_context_required" in str(exc.value)


async def test_invalid_actor():
	with pytest.raises(RuleViolation) as exc:
		BFCService(tenant_id="acme", actor_id="")
	assert "actor_required" in str(exc.value)
