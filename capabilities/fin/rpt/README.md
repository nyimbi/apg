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
