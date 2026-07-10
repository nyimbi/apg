# APG World-Class Gap Analysis - Post Waves A-L

**Date:** 2026-07-10
**Scope:** APG generator, generated Flask runtime, DSL, CLI, language-server, VS Code extension, and capability surface after the July 2026 world-class uplift work.
**Baseline evidence:** Required inputs from `git log --oneline -25`, `docs/research/world-class-uplift-2026-07/README.md`, `docs/research/app-generator-competitive-landscape-2026/README.md`, and `ls capabilities/`, plus targeted checks of generator code, grammar, tests, language-server fixtures, and VS Code extension metadata.

## Executive Summary

APG is no longer losing on basic generated-app security, operations, API contract, or generated UI polish. Waves A-L moved the generated Flask app from "promising demo generator" toward a credible production scaffold: secure authentication defaults, request IDs, Prometheus-style metrics, OpenAPI 3.1, CSV export, dark/mobile/print UI, SQLite-backed lifecycle patterns, FTS5 search, outbound HMAC webhooks, and row/column data governance.

The remaining world-class blockers are not mostly in page chrome. They are in the APG language and generated runtime contract. Retool, Budibase, Wasp, Appsmith, and Directus all make common business-app primitives immediately expressible: relationships, attachments/files, email or notification automation, automations/jobs, and admin-friendly content localization. APG has pieces of these as platform capabilities or UI affordances, but the generator does not yet expose them as first-class DSL features that reliably compile into runtime behavior, OpenAPI, UI widgets, tests, migrations, and docs.

The shortest path to "world-class" is therefore:

1. Close P1 DSL/runtime primitives: relationships, file upload fields, email notifications, and i18n scaffolding.
2. Close P2 enterprise/developer workflow gaps: generated background jobs, first-class multi-tenancy, VS Code extension hardening, and a Debug Adapter Protocol surface.
3. Close P3 expressiveness gaps: Excel import/export, computed fields, enum types, and field validation rules DSL.

## 1. Shipped: Waves A-L

Evidence caveat: the required `git log --oneline -25` contains named commits for Waves A, B, D, E, F, G, I, J, K, and L. It also contains compiler diagnostics commits without an explicit "Wave H" label. The generated app code and tests contain HTTP efficiency coverage corresponding to the original Wave C plan, but the current worktree also has unrelated uncommitted changes, so this document treats Wave C as "implemented in local generator evidence" and Wave H as "compiler diagnostics wave, not consistently labeled in history."

| Wave | Status | What It Added | Evidence |
| --- | --- | --- | --- |
| A | Shipped | Generated-app security hardening: password hashing support, login throttling/lockout, session fixation defense, request body limits, branded 404/500 handling, timing-safe API-key comparison, and production-mode secure-by-default mutation behavior. | `63730174`, `tests/test_generated_app_hardening.py`, `compiler/code_generator.py` |
| B | Shipped | Operations baseline: structured JSON logging, `X-Request-ID` generation/propagation, `/livez`, `/readyz`, and Prometheus text exposition at `/metrics`. | `fd4d63cd`, `tests/test_generated_app_ops.py` |
| C | Implemented in generator evidence, not named in recent commit subjects | HTTP efficiency: gzip compression, cache-control headers, ETag generation, and conditional GET behavior. | `docs/research/world-class-uplift-2026-07/README.md`, `tests/test_generated_app_http.py`, `compiler/code_generator.py` |
| D | Shipped | Accessibility/UI hardening: table accessibility, density toggle emission, CSP nonce migration away from unsafe inline script/style patterns, and ARIA live status support. | `82e34e76`, `tests/test_generated_app_a11y.py`, generated UI template tests |
| E | Shipped | API completeness: OpenAPI 3.1 document, API docs surface, pagination, filtering, sorting, and CSV export for generated records. | `12d3d594`, `tests/test_generated_app_api.py`, `/openapi.json` generator code |
| F | Shipped | Generated-app production hardening: startup validation, audit log output, rate limiting, and JSON content-type guard with 415 behavior. | `b715b689`, `tests/test_generated_app_hardening2.py` |
| G | Shipped | Generated UI polish: dark mode, mobile layout behavior, and print CSS. | `870af561`, generated UI/CSS tests |
| H | Shipped as compiler diagnostics work, but not consistently labeled | Compiler quality: line/column diagnostics, "Did you mean" suggestions, duplicate entity/field semantic validation, and friendlier compiler error reporting. | `2fd6c840`, `8319ecd3`, `tests/test_compiler_errors.py` |
| I | Shipped | Database/runtime patterns: created/updated/deleted timestamps, soft deletes, auto-migration, bulk create/update/delete operations, and lifecycle metadata. | `06423911`, `tests/test_generated_app_db.py` |
| J | Shipped | Outbound webhooks with request/event payloads and HMAC signing, with delivery failures isolated from primary API responses. | `d9603a78`, `tests/test_generated_app_webhooks.py` |
| K | Shipped | FTS5 full-text search and generated pytest scaffolding for compiled apps. | `2995dbef`, `tests/test_generated_app_search.py`, `tests/test_generated_tests.py` |
| L | Shipped | Data governance in the generated app: column/field ACL filtering, row ownership enforcement, and field-diff audit logging. | `258c0e88`, `tests/test_generated_app_rbac.py` |

### Platform Position After These Waves

APG now has a credible foundation in areas where runtime low-code platforms are weak:

- Ownable generated Python/Flask code rather than proprietary runtime lock-in.
- Stronger generated app contract surface: OpenAPI, smoke tests, Dockerfile, semantic model, component manifest, and release evidence.
- Africa-first capability breadth across fintech, SACCO, government, healthcare, agriculture, telco, retail, transport, and common platform domains.
- Generated UI workspaces beyond basic CRUD: dashboards, kanban, workflow wizard, agent/team console, capability console, database catalog, flow debugger, auth, PWA shell, i18n switcher, command palette, and offline banner.

The gap is that many capability and UI strengths are still metadata or platform-level services rather than language-level promises that compile into generated app behavior every time.

## 2. Remaining Gaps vs Retool, Budibase, Wasp, Appsmith, Directus

### P1 - Blocking World-Class

| Rank | Gap | Current APG State | Competitor Bar | Why It Blocks World-Class |
| ---: | --- | --- | --- | --- |
| P1.1 | DSL relationships: `has_many`, `belongs_to`, many-to-many, cascade, inverse, relationship picker, nested API | APG has DBML-style `ref: >` examples, generic screen relationship metadata, `/relationships`, and database catalog relationship display. Tests explicitly document that `table` is not always mapped into the semantic model `tables` dict; `entity` is more reliable. There is no first-class relationship DSL that drives runtime joins, nested CRUD, OpenAPI schemas, UI relationship pickers, cascade behavior, or generated tests. | Directus has Many-to-One, One-to-Many, Many-to-Many, Many-to-Any, and translations as relationship-backed modeling. Budibase has bidirectional row relationships. Wasp relies on Prisma relations. | Every serious business app needs customer-orders, invoice-lines, users-roles, attachments-owner, and tenant-owned-data relationships. Without this, APG still generates record tables, not a fully relational application model. |
| P1.2 | File upload fields | APG grammar and docs mention `file` as a field/token name, and some capabilities can handle documents, but generated apps do not expose a first-class `file` or `attachment` field with multipart upload routes, storage adapters, access control, download URLs, file metadata, OpenAPI binary schema, or upload tests. | Retool, Appsmith, Budibase, and Directus all have file picker/upload or file-management primitives. Wasp documents file uploads via its app stack. | Internal tools and customer apps routinely require receipts, KYC documents, claims evidence, contracts, profile images, CSV imports, PDFs, and audit attachments. Without file fields, APG cannot replace common Retool/Budibase/Appsmith workflows. |
| P1.3 | Email notifications | APG has notification-related capabilities, outbound webhooks, activity/audit logs, and UI notification affordances, but no generated SMTP/email provider configuration, templates, delivery events, or DSL-triggered email notification scaffold. | Wasp has email-sending support. Appsmith has SMTP plugin docs. Budibase automations can send email on events. Directus Flows can orchestrate email/notification workflows. | Email is the default side effect for approvals, onboarding, password/account flows, invoices, reminders, alerts, and support workflows. Webhooks are not a substitute for generated email templates and provider wiring. |
| P1.4 | i18n scaffolding completeness | APG already emits locale detection, `/locale`, language switcher, RTL direction, and language catalogs. The gap is scaffolding: no extraction workflow, translation file layout per generated app, pluralization rules, per-field translatable content, localized validation messages, route/content localization, or tests that prove every generated string is externalized. | Retool has org-level app localization. Directus has content translations. Budibase exposes translations for user-facing elements. | APG's Africa-first and customer-facing positioning makes i18n table stakes, not polish. A switcher without extraction, translation management, and localized validation cannot support production multilingual deployments. |

### P2 - Significant Gaps

| Rank | Gap | Current APG State | Competitor Bar | Why It Matters |
| ---: | --- | --- | --- | --- |
| P2.1 | Background job scaffolding | APG has workflow surfaces, generated run history, scheduler capability metadata (`schd`), and platform workflow adapters, but generated apps do not emit a worker process, queue backend, retry/dead-letter policy, scheduled job registry, or `job` DSL that compiles into runtime and tests. | Retool Workflows, Budibase Automations, Appsmith Workflows, Directus Flows, and Wasp Jobs all address async/scheduled work directly. | Many generated apps need imports, reconciliation, reminders, webhook retries, report generation, sync jobs, and slow external calls outside request/response. |
| P2.2 | First-class multi-tenancy | APG has an `mten` capability, tenant context patterns, `tenant_id` fields, tenant-scoped generated filtering when `tenant_id` exists, and row ownership. It lacks a declarative tenancy model that generates tenant lifecycle, isolation policy, per-tenant files/jobs/webhooks, tenant-aware audit/metrics, and migration/seed strategy. | Directus and internal tool platforms offer organization/project/user governance; SaaS app generators often rely on manual tenancy. | This is a major differentiator if done well. Today it remains too implicit: "add a `tenant_id` field" is not enough for enterprise SaaS isolation. |
| P2.3 | VS Code extension maturity | The repo has a VS Code extension with syntax grammar, snippets, themes, commands, keybindings, and LSP client configuration. The gap is production packaging and completeness: extension tests, marketplace/publishing pipeline, robust server discovery, command task provider implementation depth, generated project integration, and version compatibility checks. | Wasp has editor setup and a VS Code extension path. VS Code users expect language features, tasks, diagnostics, and installable packaging. | APG's custom DSL needs excellent editor ergonomics to avoid the custom-language adoption penalty identified in the competitive landscape. |
| P2.4 | Debug Adapter Protocol | APG has a generated flow debugger UI with run timelines, replay frames, breakpoint suggestions, and variables in HTML. It does not expose a VS Code DAP adapter with launch configs, breakpoints, step/continue, stack frames, scopes, variables, or expression evaluation. | Mature developer tools integrate debugging into the IDE, while Retool/Appsmith/Directus expose run/flow inspection in-product. | The generated flow debugger is useful, but a world-class DSL should let developers debug generated workflows without leaving the IDE. |

### P3 - Nice-to-Have

| Rank | Gap | Current APG State | Why It Is P3 |
| ---: | --- | --- | --- |
| P3.1 | Excel import/export | CSV export exists and is tested. There is no `.xlsx` import/export scaffold, workbook template generation, type coercion, row error report, or relationship-aware import. | CSV is sufficient for many MVPs. Excel matters for operations teams but does not block core app generation. |
| P3.2 | Computed fields | The grammar has broad keyword coverage including `computed`, and platform capabilities can compute analytics, but generated record models do not have a first-class `computed` field contract with dependencies, persistence policy, recomputation hooks, OpenAPI exposure, and UI read-only rendering. | Apps can work around with service logic initially. First-class computed fields improve ergonomics and data consistency. |
| P3.3 | Enum types | `enum` is a grammar entity type and enum variants exist in grammar, but generated CRUD fields are still mostly primitive strings/numbers. No end-to-end enum handling drives dropdowns, OpenAPI `enum`, DB constraints, validation, or generated tests. | Useful for correctness and UI polish, but can be approximated with strings until validation DSL lands. |
| P3.4 | Validation rules DSL | APG has semantic validation, capability rules, and runtime record validation helpers. It lacks field-level DSL constraints like `required`, `unique`, `min`, `max`, `regex`, `format`, cross-field rules, conditional requirements, localized messages, and generated client/server/OpenAPI/test alignment. | Important for quality, but the P1 primitives unlock more app categories first. Validation should follow relationships/files/email/i18n so it can validate those richer types too. |

## 3. DSL Feature Gaps

The pattern across all major gaps is the same: APG often has a keyword, capability, or UI surface, but not a precise DSL contract that generates a predictable runtime.

### 3.1 Relationships

Missing:

- `belongs_to`, `has_one`, `has_many`, `many_to_many`.
- Inverse relationship validation.
- Foreign-key field generation.
- Cascade/restrict/nullify delete policy.
- Relationship pickers in forms.
- Nested list/detail views.
- Relationship-aware OpenAPI and test scaffolds.

Example target syntax:

```apg
entity Customer {
  name: str required;
  email: str unique;
  orders: has_many<Order> inverse customer;
}

entity Order {
  customer: belongs_to<Customer> required on_delete restrict;
  total: decimal min 0;
}
```

### 3.2 File Upload Fields

Missing:

- `file`/`attachment` field semantics.
- Storage backend declaration.
- Multipart routes and signed download URLs.
- Per-file ACL, size limits, content-type allowlist, and scan hooks.
- File metadata model.
- OpenAPI `format: binary`.

Example target syntax:

```apg
entity Claim {
  claimant_name: str required;
  evidence: file accept ["application/pdf", "image/*"] max_size "10MB" storage "claims";
}

storage claims {
  backend: local;
  path: "./uploads/claims";
  private: true;
}
```

### 3.3 Email Notifications

Missing:

- Email provider configuration.
- Template definitions.
- Event-triggered delivery.
- Retry/dead-letter policy.
- Delivery audit events.
- Local preview/test command.

Example target syntax:

```apg
email_provider primary {
  driver: smtp;
  from: "noreply@example.com";
  env_prefix: "APG_SMTP";
}

email_template WelcomeCustomer {
  subject: "Welcome, {{ customer.name }}";
  body: "templates/welcome_customer.md";
}

on Customer.created {
  send email WelcomeCustomer to Customer.email retry 3;
}
```

### 3.4 i18n Scaffolding

Missing:

- Generated translation catalogs on disk.
- String extraction command.
- Localized validation messages.
- Pluralization and date/currency formatting policy.
- Translatable content fields.
- Locale-aware URLs or route policy where needed.
- Tests proving generated strings are covered by catalogs.

Example target syntax:

```apg
i18n {
  locales: ["en", "sw", "fr", "ar"];
  default: "en";
  fallback: "en";
  extract: true;
  rtl: ["ar"];
}

entity Product {
  name: str translatable required;
  description: text translatable;
}
```

### 3.5 Background Jobs

Missing:

- `job` entity semantics.
- Worker process scaffold.
- Queue backend configuration.
- Schedule syntax.
- Retry/dead-letter policy.
- Job status UI/API.

Example target syntax:

```apg
queue default {
  backend: sqlite;
  retries: 3;
  dead_letter: true;
}

job NightlyReconciliation schedule "0 2 * * *" queue default {
  run ReconcileInvoices;
  timeout: "15m";
}
```

### 3.6 Multi-Tenancy

Missing:

- Declarative tenancy model.
- Generated tenant lifecycle API/UI.
- Tenant-aware files, jobs, webhooks, audit, metrics, and connector credentials.
- Isolation policy validation.

Example target syntax:

```apg
tenant model {
  strategy: row_level;
  key: tenant_id;
  enforce_on: [records, files, jobs, webhooks, audit, metrics];
  default_role: tenant_admin;
}
```

### 3.7 Computed Fields

Missing:

- Field dependency graph.
- Stored vs virtual policy.
- Recompute hooks.
- Cycle detection.
- Read-only UI rendering.

Example target syntax:

```apg
entity Invoice {
  subtotal: decimal min 0;
  tax: decimal min 0;
  total: decimal computed "subtotal + tax" stored;
}
```

### 3.8 Enum Types

Missing:

- Enum type compilation into OpenAPI, DB constraints, generated forms, and validation.
- Labels and i18n for enum display.

Example target syntax:

```apg
enum OrderStatus {
  draft label "Draft";
  submitted label "Submitted";
  approved label "Approved";
  rejected label "Rejected";
}

entity Order {
  status: OrderStatus default draft required;
}
```

### 3.9 Validation Rules DSL

Missing:

- Field-level constraints.
- Cross-field rules.
- Conditional requirements.
- Localized error messages.
- Alignment across UI, API, OpenAPI, and generated tests.

Example target syntax:

```apg
entity Customer {
  email: str required unique format email message "customer.email.invalid";
  age: int min 18 max 120;
  tax_id: str required_if country == "KE";
}
```

## 4. Generated App Gaps

The generated Flask runtime is much stronger after Waves A-L, but it still misses common production features expected from app builders.

### Data and Relationships

- No relationship-aware runtime contract for `belongs_to`, `has_many`, or many-to-many.
- No generated relationship picker widgets.
- No cascade/restrict delete semantics.
- No nested relationship endpoints such as `/customers/{id}/orders`.
- No relationship-aware import/export.

### Files and Attachments

- No multipart upload endpoints.
- No generated storage abstraction.
- No signed/private download URL behavior.
- No attachment metadata table.
- No per-file audit trail.
- No file validation or malware-scanning hook.

### Notifications and Email

- No generated SMTP/provider configuration.
- No email templates.
- No event-triggered email delivery from the DSL.
- No delivery retry/dead-letter store.
- No generated local email preview/test harness.

### i18n Runtime

- Existing locale switcher and catalogs are useful, but incomplete.
- No translation catalog file generation per app.
- No string extraction workflow.
- No localized validation errors.
- No pluralization rules.
- No translatable content fields.

### Background Work

- No generated worker process.
- No queue/scheduler backend.
- No job retry/dead-letter persistence.
- No job status dashboard beyond workflow/debug surfaces.
- No generated async API contract for jobs.

### Multi-Tenancy

- Tenant filtering exists when fields and context are present, but tenant isolation is not generated as a full application mode.
- No tenant lifecycle/admin UI.
- No tenant-scoped files/jobs/webhooks.
- No per-tenant connector credentials.
- No tenant-aware metrics partitioning.

### Import/Export and Data Operations

- CSV export is shipped; Excel import/export is missing.
- No generated import preview, validation report, rollback, or row-level error download.
- Bulk operations are shipped; relationship-aware bulk operations are not.

## 5. Tooling Gaps

### Language Server Completeness

What exists:

- `language_server/` provides a pygls-based APG language server.
- The semantic service exposes completions, diagnostics, document symbols, definitions, references, rename, formatting, and code actions.
- Fixture audit coverage checks completion, document symbols, formatting, rename, diagnostics, code actions, and source immutability.

What remains:

- Completion/diagnostic coverage for the proposed relationship/file/email/job/tenant/i18n/validation DSL.
- Cross-file symbol indexing and import-aware rename beyond simple fixtures.
- Relationship graph navigation in the editor.
- Code actions for scaffolded P1/P2 features.
- Performance tests on large APG workspaces.

### IDE Extension

What exists:

- `vscode-extension/` contributes `.apg` language registration, grammar, snippets, themes, commands, keybindings, problem matcher, task definition, and LSP client startup.

What remains:

- Verified package/publish pipeline.
- Extension integration tests.
- Robust language-server discovery and fallback UX.
- Task provider implementation depth beyond metadata.
- Generated `launch.json` and `tasks.json` scaffolding.
- UI for browsing generated OpenAPI, relationship graph, capability contracts, and diagnostics inside VS Code.

### Debug Adapter

What exists:

- Generated app has a browser flow debugger UI with timelines, replay frames, breakpoint suggestions, and variables.

What remains:

- No VS Code Debug Adapter Protocol implementation.
- No launch configuration for generated apps/workflows.
- No IDE breakpoints, step/continue, stack frames, scopes, variables, or expression evaluation.
- No DAP tests.

### CLI Completeness

What exists:

- The CLI already covers compile, lint, validate, model, graph-suite, release, package, baseline, docs audit, tooling audit, doctor, and capability contract/scaffold workflows.

What remains:

- `apg i18n extract/check`.
- `apg add relationship`.
- `apg add file-field`.
- `apg email test/render`.
- `apg job run/schedule`.
- `apg tenant init/check`.
- `apg debug dap` or equivalent debug adapter launcher.
- `apg import excel` and `apg export excel`.
- Golden-path commands that add tests when adding DSL features.

## 6. Test Coverage Gaps

### Missing P1 Tests

- Relationship DSL parsing, semantic validation, generated OpenAPI, generated UI pickers, nested endpoints, cascade behavior, and migration output.
- Multipart file upload, size/content-type rejection, private download authorization, metadata persistence, and audit events.
- SMTP/email provider configuration, template rendering, event-triggered delivery, retry/dead-letter behavior, and local preview command.
- i18n extraction, catalog completeness, localized validation messages, pluralization, translatable fields, and RTL snapshot coverage.

### Missing P2 Tests

- Generated worker process and queue backend.
- Scheduled job execution, idempotency, retry, timeout, and dead-letter behavior.
- Tenant isolation across records, files, jobs, webhooks, audit logs, metrics, and connector credentials.
- VS Code extension compile/package tests and integration tests with the language server.
- Debug Adapter Protocol contract tests.

### Missing P3 Tests

- Excel import/export round trip with row errors and typed coercion.
- Computed fields: dependency graph, stored/virtual behavior, recomputation, cycle detection.
- Enum types: OpenAPI enum output, DB constraint, form dropdown, validation, i18n labels.
- Field validation DSL: UI errors, API errors, OpenAPI schema, generated tests, and localized messages.

### Existing Coverage That Should Be Preserved

- Generated app hardening tests.
- Ops tests for request IDs, liveness/readiness, metrics, and JSON logs.
- API tests for OpenAPI, pagination, filtering, sorting, and CSV export.
- DB lifecycle tests for timestamps, soft delete, auto-migration, and bulk operations.
- Webhook tests for HMAC and failure isolation.
- Search and generated pytest scaffold tests.
- RBAC tests for field ACL and row ownership.
- Language-server fixture audit and tooling audit.

## 7. Recommended Closure Sequence

1. **Relationship DSL first.** It is the root primitive for file owners, tenant ownership, nested APIs, import/export, and UI pickers.
2. **File upload fields second.** This immediately unlocks KYC, claims, contracts, invoices, HR, healthcare, and government workflows.
3. **Email notifications third.** Add the smallest provider/template/event model before broader automation.
4. **i18n scaffolding fourth.** Preserve the existing locale runtime, then add extraction, catalogs, localized validation, and translatable fields.
5. **Background jobs fifth.** Reuse workflow/scheduler capability ideas, but generate a concrete worker scaffold and tests.
6. **Multi-tenancy sixth.** Promote tenant handling from implicit field/context convention to application mode.
7. **VS Code and DAP seventh.** Once the language contract stabilizes, make the editor experience excellent.
8. **P3 expressiveness last.** Excel, computed fields, enums, and validation rules become simpler after relationships, file fields, i18n, jobs, and tenancy are stable.

## 8. Definition of Done for "World-Class"

APG can claim this gap analysis is closed only when each P1/P2 feature has:

- Grammar and parser support.
- Semantic-model representation.
- Compiler diagnostics with line/column and "Did you mean" quality.
- Generated runtime behavior.
- Generated UI behavior.
- OpenAPI/contract output where relevant.
- CLI support or at least CLI validation.
- Language-server completion/diagnostics.
- Focused regression tests.
- One example APG source file showing the happy path and one showing a validation failure.
