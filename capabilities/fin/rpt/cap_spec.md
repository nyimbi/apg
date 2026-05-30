# Financial Reporting Capability Summary

`fin_rpt` provides the APG financial reporting packet for report templates, report lines, reporting periods, statement generation, statement publication, consolidation, disclosures, distribution, and RPT-agent composition.

## Provides

- `financial_report_template_lifecycle`
- `report_line_mapping`
- `reporting_period_lifecycle`
- `financial_statement_generation`
- `statement_publication_workflow`
- `financial_consolidation`
- `disclosure_management`
- `report_distribution`
- `rpt_agents`

## Requires

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

## Execution Model

The package is executable without optional web, database, or AI provider dependencies. `FinancialReportingService` owns the in-memory lifecycle state, evaluates deterministic rules before writes, emits audit events with Bytewax stream metadata, and exposes summaries for generated applications.

## Composition Metadata

- Event processor: `bytewax`
- Stream: `apg.fin.rpt.lifecycle`
- Theme: `fin_rpt_control`
- UI shell: `apg_python`
- App target: `python`

## Deferred Integration

Durable storage, live GL/AP/AR/cash/document/business-intelligence integration, authorization, notification, audit sinks, report rendering engines, and durable Bytewax topologies remain adapter work after the executable package baseline.
