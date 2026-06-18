# Financial Reporting

`fin_rpt` is the APG capability for composing financial report templates, report lines, reporting periods, statement generation, statement publication, consolidation, disclosures, and report distribution into generated Python applications. It provides an executable service surface, deterministic guardrails, UI metadata, theme metadata, and Bytewax lifecycle-stream declarations.

## What It Provides

- Report template creation for balance sheet, income statement, cash flow, equity statement, and management reports.
- Report line mapping to GL account ranges.
- Reporting period lifecycle with period date controls.
- Report generation with output-format and data-quality review controls.
- Statement publication with balance check, approval, and narrative review controls.
- Consolidation records with entity, ownership, and elimination-review controls.
- Disclosure management with owner and review controls.
- Statement distribution with approved-statement, recipient, and format controls.
- IFRS-compliant primary statement generators (income statement, balance sheet, cash flow).
- Group consolidation with elimination support.
- IFRS 8 segment reporting and IAS 33 EPS calculation.
- XBRL taxonomy mapping (IFRS, US-GAAP, UK-GAAP, ESRS, FERC).
- Regulatory submission traceability.
- Reporting analytics and KPI aggregation.
- First-class RPT agents for Codex, Claude Code, OpenCode, and Pi.
- Deterministic rules for tenant, policy, reporting, consolidation, disclosure, distribution, agent, and stream guardrails.
- Bytewax lifecycle stream metadata.
- UI route and theme metadata for APG composition.

## Quick Start

```python
from capabilities.fin.rpt import FinancialReportingService

service = FinancialReportingService()
template = service.create_template(
    "income-template",
    "tenant-a",
    "Income Statement",
    "income_statement",
    "controller",
)
service.add_report_line(
    "revenue",
    "tenant-a",
    template["id"],
    "Revenue",
    "4*",
    10,
)
period = service.open_period(
    "fy2026-q1",
    "tenant-a",
    "FY2026 Q1",
    "2026-01-01",
    "2026-03-31",
)
generation = service.generate_report(
    "run-1",
    "tenant-a",
    template["id"],
    period["id"],
    "pdf",
)
statement = service.publish_statement(
    "statement-1",
    "tenant-a",
    generation["id"],
    "FY2026 Q1 Income Statement",
    True,
    "controller",
    "reviewer",
)
service.distribute_statement(
    "dist-1",
    "tenant-a",
    statement["id"],
    ["cfo@example.com"],
    "pdf",
)
summary = service.dashboard_summary("tenant-a")
```

## New Methods

### IFRS Income Statement

```python
stmt = service.generate_ifrs_income_statement(
    tenant_id="tenant-a",
    report_id="is-2026-q1",
    entity_id="entity-ke-001",
    period="FY2026-Q1",
    revenue={"product_revenue": 5_000_000.0, "service_revenue": 1_200_000.0},
    expenses={"cost_of_sales": 2_800_000.0, "operating_expenses": 900_000.0, "finance_costs": 120_000.0},
    prepared_by="controller",
)
# stmt["profit_after_tax"], stmt["gross_profit"], stmt["operating_profit"]
```

### Balance Sheet

```python
bs = service.generate_balance_sheet(
    tenant_id="tenant-a",
    report_id="bs-2026-q1",
    entity_id="entity-ke-001",
    period="FY2026-Q1",
    assets={"current_assets": 3_100_000.0, "non_current_assets": 7_400_000.0},
    liabilities={"current_liabilities": 1_200_000.0, "non_current_liabilities": 2_500_000.0},
    equity={"share_capital": 4_000_000.0, "retained_earnings": 2_800_000.0},
)
# bs["balanced"] — True when assets == liabilities + equity within 0.01
```

### Cash Flow Statement (indirect method)

```python
cf = service.generate_cash_flow_statement(
    tenant_id="tenant-a",
    report_id="cf-2026-q1",
    entity_id="entity-ke-001",
    period="FY2026-Q1",
    operating_activities={"net_profit": 1_380_000.0, "depreciation": 240_000.0, "working_capital_changes": -180_000.0},
    investing_activities={"capex": -600_000.0, "asset_disposals": 50_000.0},
    financing_activities={"dividends_paid": -300_000.0, "new_borrowings": 500_000.0},
)
# cf["net_change_in_cash"]
```

### IFRS 8 Segment Reporting

```python
seg = service.segment_reporting(
    tenant_id="tenant-a",
    report_id="seg-2026-q1",
    entity_id="entity-ke-001",
    period="FY2026-Q1",
    dimension="business_unit",
    segments={
        "retail": {"revenue": 3_200_000.0, "profit": 640_000.0, "assets": 4_100_000.0},
        "wholesale": {"revenue": 2_000_000.0, "profit": 580_000.0, "assets": 3_300_000.0},
    },
)
# seg["segments"][0]["revenue_pct"] — each segment's share of total revenue
```

### XBRL Taxonomy Mapping

```python
xbrl = service.xbrl_taxonomy_mapping(
    tenant_id="tenant-a",
    mapping_id="xbrl-ifrs-2026-q1",
    entity_id="entity-ke-001",
    period="FY2026-Q1",
    taxonomy="IFRS",
    line_mappings={
        "Revenue": "ifrs-full:Revenue",
        "Profit after tax": "ifrs-full:ProfitLoss",
    },
    validated_by="controller",
)
# xbrl["coverage_pct"], xbrl["unmapped_lines"]
```

## Method Reference

| Method | Purpose |
|---|---|
| `create_template` | Define a report template (P&L, BS, CF, equity, management) |
| `add_report_line` | Map a GL account range to a report line |
| `open_period` | Open a reporting period with date controls |
| `generate_report` | Run report generation against a template and period |
| `publish_statement` | Publish with balance check, approval, and narrative review |
| `create_consolidation` | Record a single subsidiary consolidation entry |
| `consolidation` | Full group consolidation across multiple subsidiaries |
| `record_disclosure` | Attach a reviewed disclosure to a published statement |
| `distribute_statement` | Distribute an approved statement to recipients |
| `generate_ifrs_income_statement` | IFRS income statement with P&L breakdown |
| `generate_balance_sheet` | IFRS statement of financial position with balance check |
| `generate_cash_flow_statement` | IFRS cash flow (indirect method) |
| `segment_reporting` | IFRS 8 segment report by any reporting dimension |
| `earnings_per_share` | Basic and diluted EPS per IAS 33 |
| `notes_to_accounts` | Structured notes (policies, estimates, disclosures) |
| `xbrl_taxonomy_mapping` | Map lines to IFRS/US-GAAP/ESRS XBRL concepts |
| `regulatory_submission` | Record a regulatory filing with full statement traceability |
| `reporting_analytics` | Aggregated KPIs across statements, filings, and distributions |
| `register_rpt_agent` | Register a review agent (Codex, Claude Code, OpenCode, Pi) |
| `validate_agent_rpt_action` | Gate privileged agent actions behind human approval |
| `dashboard_summary` | Cross-domain counts and streaming manifest |

## World-Class Enhancements (v2.0)

Roadmap items that extend `fin_rpt` to enterprise-grade coverage:

1. **Variance Analysis Engine** — automated period-over-period and budget-vs-actual variance with Ollama-drafted narrative commentary.
2. **Rolling Forecast Integration** — driver-based 12/18-month rolling forecasts blended with actuals; confidence intervals from historical variance.
3. **Multi-Currency Translation and Revaluation** — IAS 21 FX translation (spot/average/closing rates), translation reserve (OCI), and `fx_translation_completed` events.
4. **Real-Time GL Integration via CDC** — Debezium → Bytewax → Bytewax streaming of journal entries; near-real-time trial balance eliminates batch close delays.
5. **Audit-Trail Immutability with Merkle Chaining** — SHA-256 hash-chained audit ledger with `verify_audit_chain()`; backed by append-only PostgreSQL.
6. **Automated IFRS 16 Lease Schedule Generator** — right-of-use asset, lease liability, interest and depreciation amortisation tables from lease terms.
7. **Statement of Changes in Equity (SOCE)** — IFRS-compliant equity roll-forward across profit, OCI, dividends, and share movements.
8. **Intercompany Elimination Automation** — reference/amount matching of IC transactions; auto-proposed elimination journals; unmatched item flagging.
9. **AI Narrative Generation via Ollama** — MD&A, board pack, and regulatory commentary drafted locally (llama3/mistral); tone-configurable; human approval gate.
10. **Ratio Analysis and Financial Health Scorecard** — liquidity, solvency, profitability, and efficiency ratios; covenant breach flagging against configurable thresholds.
11. **Board Pack / Management Accounts PDF Builder** — cover page, highlights, statements, charts, and commentary assembled via WeasyPrint/ReportLab with Datacraft branding.
12. **Budget vs Actuals with Drill-Down** — hierarchical BvA from entity → cost centre → GL account; YTD tracking; absolute and percentage variance.
13. **Period-Close Checklist and Workflow Orchestration** — DAG-based close task management with owner, due date, predecessor, and completion gates; escalation events.
14. **ESG / Sustainability Reporting Module** — Scope 1/2/3 emissions, energy, water, diversity, and governance metrics mapped to ESRS and IFRS S1/S2 XBRL taxonomy.
15. **Predictive Close Date Estimation via ML** — gradient-boosted regressor on close history (task lag, escalation rate, data quality) returns predicted close date with confidence interval and top-3 risk factors.

## Contract

Use `get_capability_contract()` to inspect the APG composition surface.

```python
from capabilities.fin.rpt import get_capability_contract

contract = get_capability_contract("tenant-a")
print(contract["provides"])
print(contract["streaming"]["processor"])
```

The contract exposes:

- `configuration`
- `configuration_schema`
- `rule_engine`
- `ui`
- `theme`
- `streaming`

## Guardrails

The rule engine blocks or routes review for:

- Missing tenant context.
- Writes without policy attachment.
- Templates without name or supported statement type.
- Report lines without template, account mapping, or sort order.
- Reporting periods without name, dates, or valid period range.
- Report generation without template, period, template lines, or supported output format.
- Low data quality generation without review.
- Statement publication without generated report, balance check, approval, or narrative review.
- Consolidations without parent entity, subsidiary entity, valid ownership, or elimination review.
- Disclosures without statement, owner, or review.
- Distribution without statement, approved statement, recipient, or supported format.
- Batch and lifecycle events not routed through Bytewax.
- Unsupported RPT-agent runtime or role.
- Privileged RPT-agent actions without human approval.

## UI And Theme

The capability publishes route metadata for:

- `/fin-rpt/dashboard`
- `/fin-rpt/templates`
- `/fin-rpt/lines`
- `/fin-rpt/periods`
- `/fin-rpt/generation`
- `/fin-rpt/statements`
- `/fin-rpt/consolidation`
- `/fin-rpt/disclosures`
- `/fin-rpt/distribution`
- `/fin-rpt/agents`
- `/fin-rpt/settings`

The default theme is `fin_rpt_control`. View helpers in `views.py` return dashboard, template, line, period, generation, statement, consolidation, disclosure, distribution, and agent workbench models.

## AI Agents

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `statement_reviewer`
- `consolidation_reviewer`
- `disclosure_reviewer`
- `distribution_reviewer`
- `variance_narrative_reviewer`
- `close_reporting_reviewer`

Register an agent with `register_rpt_agent()` and validate privileged proposals with `validate_agent_rpt_action()`.

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/rpt/__init__.py \
  capabilities/fin/rpt/capability_contract.py \
  capabilities/fin/rpt/service.py \
  capabilities/fin/rpt/api.py \
  capabilities/fin/rpt/views.py \
  capabilities/fin/rpt/app.py \
  capabilities/fin/rpt/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/fin/rpt/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/rpt/app.py
```

Deferred live-system work includes durable stores, live financial/document/BI adapters, report rendering providers, durable Bytewax deployment, rendered browser UI, and performance testing.
