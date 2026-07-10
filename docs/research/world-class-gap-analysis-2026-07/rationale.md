# Prioritisation Rationale

**Date:** 2026-07-10

This note explains why the remaining gaps were ranked P1/P2/P3 and why the recommended closure sequence starts with the DSL rather than more generated UI polish.

## Ranking Method

Each gap was scored against five questions:

1. **Commonness:** Is this needed in ordinary business applications?
2. **Competitor expectation:** Do Retool, Budibase, Wasp, Appsmith, or Directus make this easy or first-class?
3. **Generator leverage:** Does fixing it once improve every generated APG app?
4. **Composability impact:** Does it unlock other missing features?
5. **Workaround quality:** Can users reasonably work around it without abandoning the APG model?

P1 means the gap blocks APG from being credibly world-class for common app-builder workloads. P2 means the gap is important for enterprise adoption or developer experience, but apps can still be built manually around it. P3 means the feature improves correctness or convenience, but does not decide whether APG can replace the benchmark tools.

## Why Relationships Are P1.1

Relationships are the core missing modeling primitive. Without first-class `belongs_to`, `has_many`, and many-to-many semantics, APG cannot reliably generate:

- relational forms and pickers;
- nested CRUD and OpenAPI paths;
- relationship-aware access control;
- cascade/restrict delete behavior;
- joined list/detail views;
- relationship-aware imports;
- tenant ownership graphs;
- file ownership and attachment relationships.

APG currently has relationship-adjacent surfaces: DBML-style refs, database catalog relationships, screen relationship metadata, and relationship routes. The blocker is that these do not form one end-to-end language contract. This is why relationships rank ahead of files, jobs, and validation.

## Why File Upload Fields Are P1

File fields are a standard business-app primitive, not an edge feature. APG's target domains need them constantly:

- KYC documents for fintech and SACCO apps;
- claims evidence for insurance;
- lab reports and patient files for healthcare;
- contracts and matter documents for legal;
- invoices, receipts, and delivery proofs for SCM and finance;
- permits, land documents, and citizen evidence for government.

Competitors expose file upload or file-management surfaces directly. APG capabilities may process documents, but generated apps need a field-level contract that handles upload, storage, metadata, private download, ACL, audit, OpenAPI, and tests.

## Why Email Notifications Are P1

Outbound webhooks are shipped and valuable, but they are developer integration plumbing. Email is the business-user notification primitive. A generated app that can create records but cannot email approval requests, onboarding messages, invoices, reminders, or alerts is not competitive with automation builders.

Email is P1 because it ties together:

- events;
- templates;
- provider configuration;
- retry/dead-letter behavior;
- audit logs;
- local preview/testing;
- i18n message catalogs.

The smallest viable implementation should be boring: SMTP/env configuration, templates, event triggers, delivery audit, and tests.

## Why i18n Scaffolding Is P1 Despite Existing Locale Support

APG already has meaningful i18n runtime work: locale selection, cookie-based switching, `Accept-Language`, RTL direction, and language catalogs. The remaining gap is scaffolding and completeness.

The project positions itself as Africa-first and customer-facing. That makes multilingual output a core product promise. A production multilingual app needs:

- extracted strings;
- generated catalog files;
- localized validation messages;
- pluralization;
- date/currency formatting policy;
- translatable content fields;
- tests that prevent untranslated generated strings from leaking.

This is P1 because APG's geography and customer-facing scope make i18n non-optional.

## Why Background Jobs Are P2, Not P1

Background jobs are important, and every serious app eventually needs async work. They are P2 because APG can still generate useful synchronous CRUD/workflow apps without them, and because the platform already has workflow, scheduler, event, and adapter concepts that can support a staged implementation.

They should be implemented after the P1 primitives because jobs will need:

- relationship-aware data access;
- file processing;
- email delivery retries;
- tenant isolation;
- generated observability.

Building jobs first would force later rework.

## Why Multi-Tenancy Is P2, Not P1

Multi-tenancy could become one of APG's strongest differentiators. The project already has tenant-related capabilities and generated `tenant_id` filtering patterns. However, many generated apps can start as single-tenant or manually scoped deployments.

It is P2 because a complete implementation is larger than a syntax addition. It must cover records, files, jobs, webhooks, audit, metrics, credentials, tenant lifecycle, roles, migrations, and seed data. That breadth makes it significant but not the first blocker after the security/ops waves.

## Why VS Code Extension Maturity Is P2

APG has a custom DSL. The competitive landscape explicitly warns that custom-language adoption suffers when tooling is weak. The repo already has a VS Code extension and a language server, so the gap is maturity rather than absence.

It is P2 because good CLI and tests can carry early adopters, but world-class developer adoption needs installable, tested, predictable IDE support.

## Why Debug Adapter Is P2

The generated browser flow debugger is a strong generated-app feature. The missing piece is IDE debugging through the Debug Adapter Protocol.

This is P2 because it affects developer velocity and trust in generated workflows, but it does not block basic app generation. It should follow language stabilization so the debugger can target durable workflow/job semantics rather than transient implementation details.

## Why Excel, Computed Fields, Enums, and Validation Are P3

These are real gaps but not the first closure wave:

- **Excel import/export:** CSV is already shipped. Excel is operationally useful, but not foundational.
- **Computed fields:** Useful for correctness and UX, but can be approximated in service code until the data model is richer.
- **Enum types:** Important for OpenAPI/forms/validation, but strings can temporarily stand in.
- **Validation rules DSL:** High value, but it should validate the final field universe, including relationships, files, enums, computed fields, and i18n messages.

Validation is intentionally P3 in this document only because the user-specified P1/P2 list puts more structural gaps first. In implementation sequencing, validation work should be threaded into each P1 feature rather than saved as one late mega-feature.

## Why Not Prioritise More UI Polish

The generated UI is already stronger than the generated data model in several areas: command palette, PWA shell, dark mode, mobile layout, print CSS, kanban, dashboards, workflow wizard, capability console, database catalog, and flow debugger.

More UI polish would be visible but lower leverage. The missing world-class claims now live behind the UI:

- Can the DSL express a real relational app?
- Can generated apps accept files?
- Can they notify people?
- Can they run async work?
- Can they isolate tenants?
- Can every generated string be localized?

Those answers matter more than another dashboard widget.

## Recommended Implementation Order

1. **Relationship DSL and runtime.** Define the model before building feature-specific ownership.
2. **File fields.** Use relationships to attach files to owners and tenants.
3. **Email notifications.** Use events and templates, then localize messages.
4. **i18n scaffolding.** Extend the existing runtime into extraction/catalog/test coverage.
5. **Background jobs.** Add worker/queue/schedule after files/email/i18n establish side-effect patterns.
6. **Multi-tenancy.** Promote tenant handling into a full app mode after records/files/jobs/webhooks exist.
7. **IDE/DAP completion.** Update language server, VS Code extension, and debug adapter once the language surface is stable.
8. **P3 expressiveness.** Add Excel, computed fields, enums, and validation-rule completeness.

## Stop Conditions For Each Priority

A gap is not closed when the parser accepts syntax. It is closed only when APG has:

- parser and AST support;
- semantic-model output;
- compiler diagnostics;
- generated runtime behavior;
- generated UI behavior where relevant;
- OpenAPI/contract output where relevant;
- CLI and language-server support;
- regression tests;
- at least one example source file;
- documentation and source references.
