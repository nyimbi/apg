# Accounts Receivable Capability Specification

## Purpose

`arc_accounts_receivable` makes customer receivables executable as an APG composition unit. The capability owns the receivables lifecycle from customer onboarding through credit review, invoicing, payment receipt, cash application, collections, disputes, aging, event emission, and AI-assisted review.

## Users

- Finance operators manage customers, invoices, payments, applications, and disputes.
- Credit reviewers assess limits, holds, and risky customers.
- Collections teams work overdue invoices and record outcomes.
- Controllers inspect aging, controls, and audit evidence.
- AI agents prepare review recommendations under explicit human approval rules.
- Application builders compose ARC into ERP, billing, treasury, CRM, and analytics applications.

## Scope

In scope:

- Tenant-scoped customer receivable records.
- Credit assessments and credit holds.
- Draft invoice creation, issue approval, and receivable status changes.
- Payment receipt capture.
- Cash application to invoices.
- Collection activity records.
- Dispute opening and reviewed resolution.
- Aging summary.
- ARC agent registration, runtime validation, role validation, and privileged-action review.
- Contract metadata for configuration, rules, UI routes, theme, streaming, dependencies, and provided services.
- Dependency-light service, API helpers, view models, semantic app entrypoint, package manifest, and tests.

Out of scope for this packet:

- Durable database migrations.
- Live bank, card, mobile money, GL, CRM, document, BI, auth, audit, and notification adapters.
- Durable Bytewax deployment topology.
- Rendered web UI.
- Large-scale performance and failover verification.

## Lifecycle

1. A customer is created with tenant context, code, legal name, supported customer type, and currency.
2. A credit assessment records limit, score, reviewer where required, and optional credit hold.
3. An invoice is created in draft with customer, number, dates, lines, revenue account, and positive total.
4. The invoice is issued only when approved and when the customer is not on credit hold.
5. A payment is recorded with customer, reference, date, positive amount, supported method, and cash account.
6. Cash is applied to an invoice when payment and invoice exist, allocation is positive, and allocation does not exceed outstanding balance.
7. Collection activity is recorded against an overdue, partially paid, or disputed invoice.
8. A dispute may be opened with a supported reason and owner.
9. A dispute is resolved only after review is recorded.
10. Aging and dashboard summaries expose current receivables state.
11. Lifecycle events are emitted with `processor = "bytewax"`.

## Configuration

The capability contract exposes configuration sections for:

- `customers`
- `credit`
- `invoices`
- `invoice_lines`
- `payments`
- `cash_application`
- `collections`
- `disputes`
- `arc_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Each section is tenant-scoped and can be overridden through APG composition configuration.

## Rules

The deterministic rule engine must return `allow`, `require_review`, or `deny`.

Required rule groups:

- Governance: tenant context and write policy.
- Customer: required code, legal name, supported type.
- Credit: customer, limit, low-score review.
- Invoice: customer, number, dates, due date, lines, total, credit hold, approval.
- Payment: customer, reference, date, amount, method, cash account.
- Cash application: payment, invoice, positive allocation, overapplication, unapplied review.
- Collections: overdue invoice, contact method, priority.
- Disputes: invoice, supported reason, owner, resolution review.
- Streaming: ARC batch and event routing through Bytewax.
- Agents: supported runtimes, supported roles, human approval for privileged scope.

## UI And Theme

The capability must expose screen metadata for:

- Dashboard
- Customers
- Credit
- Invoices
- Payments
- Cash Application
- Collections
- Disputes
- Aging
- Agents
- Settings

The theme name is `arc_accounts_receivable_control`. Screens use compact financial workflow layouts with status chips, review queues, risk bands, allocation grids, and agent review lanes.

## Agent Composition

ARC agents are first-class records. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles cover credit, invoice, cash application, collections, dispute, and revenue-recognition review. Agents can prepare and review work, but privileged actions require human approval.

## Adapter Boundaries

The capability declares dependencies on authorization, audit, notification, composition events/config, GL, cash management, document management, BI, and CRM. The package exposes stable adapter boundaries and does not require those providers for import or focused tests.

## Acceptance Gates

- Contract validates through `validate_contract_shape`.
- Package imports without optional provider dependencies.
- Service executes the full customer-to-cash lifecycle.
- Negative guardrail tests prove denials and review gates.
- Semantic model reports Bytewax streaming and ARC agent support.
- APG inspect, publish-plan, and implementation-audit commands run for the package.
- Documentation matches executable code.
