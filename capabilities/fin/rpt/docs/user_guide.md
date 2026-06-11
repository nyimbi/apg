# Financial Reporting (fin_rpt) — User Guide

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke

---

## Overview

`fin_rpt` is the APG financial reporting capability. It produces IFRS-compliant P&L, balance sheet, cash flow, equity statements, segment reports, EPS, lease schedules, ESG metrics, variance analysis, and ratio scorecards. All methods are tenant-scoped and emit Bytewax lifecycle events.

---

## Installation

```bash
pip install apg-fin-rpt
# or in the monorepo
uv pip install -e capabilities/fin/rpt
```

---

## Core Concepts

| Concept | Description |
|---|---|
| Template | A named report structure mapping GL account ranges to labelled lines |
| Period | A dated reporting window (month, quarter, year) |
| Generation | A render run of a template against a period |
| Statement | A published, approved generation |
| Consolidation | A group-level parent+subsidiary aggregation |
| Disclosure | A narrative note attached to a published statement |
| Distribution | A delivered statement to a list of recipients |

---

## Quick Start

```python
import asyncio
from capabilities.fin.rpt import FinancialReportingService

svc = FinancialReportingService()

# 1. Build a template
template = svc.create_template(
    "pl-template", "acme", "P&L", "income_statement", "controller"
)
svc.add_report_line("revenue", "acme", template["id"], "Revenue", "4*", 10)
svc.add_report_line("cogs", "acme", template["id"], "Cost of Sales", "5*", 20)

# 2. Open a period
period = svc.open_period("fy2026-q1", "acme", "FY2026 Q1", "2026-01-01", "2026-03-31")

# 3. Generate and publish
gen = svc.generate_report("run-1", "acme", template["id"], period["id"], "pdf")
stmt = svc.publish_statement("stmt-1", "acme", gen["id"], "FY2026 Q1 P&L", True, "cfo", "auditor")
svc.distribute_statement("dist-1", "acme", stmt["id"], ["board@acme.com"], "pdf")

print(svc.dashboard_summary("acme"))
```

---

## IFRS Financial Statements

### Income Statement

```python
income = svc.generate_ifrs_income_statement(
    tenant_id="acme",
    report_id="is-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    revenue={"product_sales": 5_000_000, "service_revenue": 1_200_000},
    expenses={
        "cost_of_sales": 3_100_000,
        "selling_expenses": 400_000,
        "admin_expenses": 300_000,
        "finance_costs": 80_000,
        "income_tax": 0,  # computed at 30% if omitted
    },
)
print(income["profit_after_tax"])  # → float
```

### Balance Sheet

```python
bs = svc.generate_balance_sheet(
    tenant_id="acme",
    report_id="bs-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    assets={"current_assets": 4_500_000, "non_current_assets": 8_000_000},
    liabilities={"current_liabilities": 2_000_000, "non_current_liabilities": 3_500_000},
    equity={"share_capital": 3_000_000, "retained_earnings": 4_000_000},
)
assert bs["balanced"]  # True when assets == liabilities + equity within 0.01
```

### Cash Flow Statement

```python
cf = svc.generate_cash_flow_statement(
    tenant_id="acme",
    report_id="cf-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    operating_activities={"net_profit": 900_000, "depreciation": 250_000, "working_capital_changes": -150_000},
    investing_activities={"capex": -500_000, "asset_disposals": 50_000},
    financing_activities={"dividends_paid": -200_000, "new_borrowings": 300_000},
)
print(cf["net_change_in_cash"])
```

### Statement of Changes in Equity (SOCE)

```python
soce = await svc.generate_equity_statement(
    tenant_id="acme",
    report_id="soce-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    opening_equity={"share_capital": 3_000_000, "retained_earnings": 3_500_000},
    profit_for_period=900_000,
    other_comprehensive_income=50_000,
    dividends_declared=200_000,
    share_issues=500_000,
)
print(soce["closing_total"])
```

---

## Variance Analysis

```python
va = await svc.variance_analysis(
    tenant_id="acme",
    report_id="va-2026-q1",
    entity_id="acme-ke",
    current_period="2026-Q1",
    prior_period="2025-Q1",
    current_figures={"revenue": 6_200_000, "gross_profit": 2_100_000},
    prior_figures={"revenue": 5_500_000, "gross_profit": 2_000_000},
    budget_figures={"revenue": 6_000_000, "gross_profit": 2_050_000},
    threshold_pct=5.0,
)
print(va["flagged_items"])   # lines exceeding 5 % change
print(va["bva_variances"])   # period vs budget
```

---

## Financial Ratio Scorecard

```python
ratios = await svc.compute_financial_ratios(
    tenant_id="acme",
    report_id="ratios-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    current_assets=4_500_000,
    current_liabilities=2_000_000,
    total_assets=12_500_000,
    total_liabilities=5_500_000,
    total_equity=7_000_000,
    revenue=6_200_000,
    gross_profit=2_100_000,
    ebit=1_320_000,
    ebitda=1_570_000,
    net_profit=900_000,
    interest_expense=80_000,
    receivables=1_200_000,
    payables=800_000,
    covenant_thresholds={"current_ratio": 1.5, "interest_coverage": 3.0},
)
print(ratios["ratios"]["gross_margin_pct"])
print(ratios["covenant_breaches"])   # [] if all covenants met
```

---

## Budget vs Actuals

```python
bva = await svc.budget_vs_actuals(
    tenant_id="acme",
    report_id="bva-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    actuals={"revenue": 6_200_000, "opex": 4_100_000},
    budget={"revenue": 6_000_000, "opex": 4_000_000},
    ytd_actuals={"revenue": 11_500_000, "opex": 8_200_000},
    ytd_budget={"revenue": 12_000_000, "opex": 8_000_000},
)
print(bva["adverse_count"])   # number of adverse variance lines
```

---

## IFRS 16 Lease Schedule

```python
lease = await svc.generate_lease_schedule(
    tenant_id="acme",
    schedule_id="lease-nairobi-office",
    entity_id="acme-ke",
    commencement_date="2026-01-01",
    lease_term_months=36,
    monthly_payment=150_000,
    incremental_borrowing_rate=0.12,   # 12 % p.a.
    annual_escalation_pct=5.0,
)
print(lease["initial_lease_liability"])
print(lease["amortisation_table"][0])  # month 1 breakdown
```

---

## Multi-Currency Translation (IAS 21)

```python
fx = await svc.translate_currency(
    tenant_id="acme",
    translation_id="fx-2026-q1",
    entity_id="acme-ug",
    period="2026-Q1",
    functional_currency="UGX",
    presentation_currency="USD",
    spot_rate=0.000265,
    average_rate=0.000270,
    closing_rate=0.000268,
    monetary_items={"cash": 500_000_000, "receivables": 200_000_000},
    income_items={"revenue": 1_200_000_000, "expenses": 900_000_000},
    equity_items={"share_capital": 400_000_000},
)
print(fx["translation_reserve_oci"])  # OCI translation difference
```

---

## Period-Close Checklist

```python
checklist = await svc.create_close_checklist(
    tenant_id="acme",
    checklist_id="close-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    tasks=[
        {"name": "post_accruals", "owner": "controller", "due_date": "2026-04-03", "predecessors": []},
        {"name": "reconcile_subledgers", "owner": "bookkeeper", "due_date": "2026-04-05", "predecessors": ["post_accruals"]},
        {"name": "consolidate", "owner": "group_finance", "due_date": "2026-04-07", "predecessors": ["reconcile_subledgers"]},
        {"name": "cfo_review", "owner": "cfo", "due_date": "2026-04-08", "predecessors": ["consolidate"]},
        {"name": "publish", "owner": "controller", "due_date": "2026-04-10", "predecessors": ["cfo_review"]},
    ],
    close_coordinator="controller",
)

# Advance tasks in dependency order
checklist = await svc.advance_close_task("acme", checklist["id"], "post_accruals", "controller")
checklist = await svc.advance_close_task("acme", checklist["id"], "reconcile_subledgers", "bookkeeper")
print(checklist["completed_count"])  # → 2
```

---

## ESG / Sustainability Reporting

```python
esg = await svc.record_esg_metrics(
    tenant_id="acme",
    report_id="esg-2026",
    entity_id="acme-ke",
    period="2026-FY",
    scope1_co2_tonnes=120.5,
    scope2_co2_tonnes=85.2,
    scope3_co2_tonnes=340.0,
    energy_kwh=1_200_000,
    water_m3=8_500,
    waste_tonnes=42.0,
    female_leadership_pct=45.0,
    employee_turnover_pct=12.5,
    board_independence_pct=60.0,
    reporting_framework="ISSB",
)
print(esg["emissions"]["total_ghg_tonnes"])
print(esg["xbrl_taxonomy"])   # → "IFRS-S2"
```

---

## Segment Reporting (IFRS 8)

```python
segments = svc.segment_reporting(
    tenant_id="acme",
    report_id="seg-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    dimension="product_line",
    segments={
        "enterprise": {"revenue": 4_000_000, "profit": 1_200_000, "assets": 6_000_000},
        "sme": {"revenue": 1_500_000, "profit": 350_000, "assets": 2_000_000},
        "consumer": {"revenue": 700_000, "profit": 80_000, "assets": 800_000},
    },
)
print(segments["segments"])   # list with revenue_pct computed
```

---

## EPS (IAS 33)

```python
eps = svc.earnings_per_share(
    tenant_id="acme",
    report_id="eps-2026-q1",
    entity_id="acme-ke",
    period="2026-Q1",
    net_profit=900_000,
    weighted_avg_shares=10_000_000,
    diluted_shares=10_500_000,
    preferred_dividends=50_000,
)
print(eps["basic_eps"])    # → 0.085
print(eps["diluted_eps"])  # → 0.0809...
```

---

## XBRL Taxonomy Mapping

```python
mapping = svc.xbrl_taxonomy_mapping(
    tenant_id="acme",
    mapping_id="xbrl-2026",
    entity_id="acme-ke",
    period="2026-Q1",
    taxonomy="IFRS",
    line_mappings={
        "Revenue": "ifrs-full:Revenue",
        "Cost of Sales": "ifrs-full:CostOfSales",
    },
)
print(mapping["coverage_pct"])  # % of template lines mapped
```

---

## Regulatory Submission

```python
sub = svc.regulatory_submission(
    tenant_id="acme",
    submission_id="cma-2026-q1",
    entity_id="acme-ke",
    report_type="quarterly_financial_statements",
    regulator="CMA_Kenya",
    period="2026-Q1",
    submitted_by="compliance_officer",
    statement_ids=[stmt["id"]],
    xbrl_mapping_id=mapping["id"],
)
print(sub["status"])  # → "submitted"
```

---

## Group Consolidation

```python
group = svc.consolidation(
    tenant_id="acme",
    consolidation_id="group-2026-q1",
    parent_id="acme-holding",
    subsidiaries=["acme-ke", "acme-ug", "acme-tz"],
    period="2026-Q1",
    method="full",
    eliminations={"intercompany_loans": 500_000, "intercompany_sales": 1_200_000},
    approved_by="group_cfo",
)
print(group["total_eliminations"])
```

---

## AI Agents

Register an LLM agent with a specific role:

```python
agent = svc.register_rpt_agent(
    tenant_id="acme",
    name="variance-narrator",
    runtime="claude_code",
    role="variance_narrative_reviewer",
    instructions="Draft plain-English variance commentary for each flagged line item.",
)

# Validate a privileged action before execution
result = svc.validate_agent_rpt_action(
    tenant_id="acme",
    agent_id=agent["id"],
    action="publish_statement",
    privileged_scope=True,
    human_approval_recorded=True,
)
print(result["decision"])  # → "allow"
```

---

## Dashboard

```python
summary = svc.dashboard_summary("acme")
# Keys: template_count, report_line_count, period_count, generation_count,
#       published_statement_count, consolidation_count, disclosure_count,
#       distribution_count, segment_report_count, eps_report_count,
#       regulatory_submission_count, xbrl_mapping_count, rpt_agent_count,
#       audit_event_count, streaming
```

---

## Guardrail Reference

| Rule | Enforcement |
|---|---|
| Missing tenant context | `deny` |
| Write without policy attachment | `deny` |
| Template missing name or unsupported type | `deny` |
| Report line missing template or account mapping | `deny` |
| Period with invalid date range | `deny` |
| Generation without template lines | `deny` |
| Low data quality without review | `require_review` |
| Statement without balance check or approval | `deny` |
| Consolidation ownership out of bounds | `deny` |
| Distribution to unapproved statement | `deny` |
| Batch not routed through Bytewax | `deny` |
| Unsupported agent runtime or role | `deny` |
| Privileged agent action without human approval | `require_review` |

---

## Testing

```bash
# Compile check
./.venv/bin/python -m py_compile capabilities/fin/rpt/service.py

# Unit tests
./.venv/bin/pytest -q capabilities/fin/rpt/tests/

# Smoke run
./.venv/bin/python capabilities/fin/rpt/app.py
```

---

## Deferred Work

- Durable PostgreSQL persistence (replace in-memory dicts)
- Live ERP/GL CDC ingestion via Debezium + Bytewax
- Ollama-backed narrative commentary generation
- PDF board pack rendering (WeasyPrint/ReportLab)
- Predictive close-date ML model
- Browser-rendered Flask-AppBuilder UI
- Performance and load testing
