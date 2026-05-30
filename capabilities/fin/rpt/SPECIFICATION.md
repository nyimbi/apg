# Financial Reporting Specification

## Intent

Financial Reporting (`fin_rpt`) makes statement generation and reporting governance a composable APG capability. It provides executable lifecycle surfaces for templates, report lines, reporting periods, report generation, statement publication, consolidation, disclosures, distribution, RPT-agent review, UI routes, theming, deterministic rules, and Bytewax lifecycle streaming.

The capability is designed for generated APG applications that need reporting operations to be executable immediately while still exposing the contract, guardrails, and metadata required for later durable storage, report rendering, and adapter integration.

## Functional Requirements

- Create tenant-scoped report templates with name, supported statement type, and owner.
- Add tenant-scoped report lines with template, label, account mapping, line type, and sort order.
- Open reporting periods with name, start date, end date, and close status.
- Generate reports only when template, reporting period, template lines, supported output format, and data-quality controls pass.
- Publish statements only when generated report, balance check, approval, and narrative review controls pass.
- Create consolidations with parent entity, subsidiary entity, method, ownership percentage, and elimination review.
- Record disclosures with statement linkage, owner, and review evidence.
- Distribute statements only when statement is approved, recipients exist, and output format is supported.
- Register first-class RPT agents for Codex, Claude Code, OpenCode, and Pi.
- Validate privileged AI-agent RPT actions through a human approval guardrail.
- Expose dashboard, template, line, period, generation, statement, consolidation, disclosure, distribution, agent, and settings UI route metadata.
- Emit lifecycle events through a Bytewax-backed stream named `apg.fin.rpt.lifecycle`.

## Rule Engine

The deterministic rule engine evaluates plain context dictionaries and returns `allow`, `deny`, or `require_review`. It enforces tenant context, write policy attachment, template name and statement type, report-line template/account/sort controls, period name/date/range controls, generation template/period/line/output/quality controls, statement balance/approval/narrative controls, consolidation entity/ownership/elimination review controls, disclosure statement/owner/review controls, distribution statement/approval/recipient/format controls, Bytewax routing, supported RPT-agent runtime and role, and human approval for privileged agent actions.

## Configuration

The contract exposes explicit configuration sections:

- `templates`
- `report_lines`
- `periods`
- `generation`
- `statements`
- `consolidation`
- `disclosures`
- `distribution`
- `rpt_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Tenant overrides are passed to `get_capability_contract(tenant_id, overrides)` and deep-merged into the default configuration.

## Composition Interfaces

Provides:

- `financial_report_template_lifecycle`
- `report_line_mapping`
- `reporting_period_lifecycle`
- `financial_statement_generation`
- `statement_publication_workflow`
- `financial_consolidation`
- `disclosure_management`
- `report_distribution`
- `rpt_agents`

Requires:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `general_ledger`
- `accounts_payable`
- `accounts_receivable`
- `cash_management`
- `document_management`
- `business_intelligence`

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, deterministic rules, UI routes, theme tokens, and Bytewax streaming metadata.
- Package import exposes `FinancialReportingService`, `RPTService`, contract helpers, streaming metadata, and registration metadata without requiring optional web, database, AI, or rendering dependencies.
- Service supports template, report-line, period, generation, statement publication, consolidation, disclosure, distribution, RPT-agent, dashboard, statement-summary, distribution-summary, audit, batch-validation, and compatibility record operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes RPT-agent metadata, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths, guardrail failures, API/view execution, app self-test, and semantic metadata.
