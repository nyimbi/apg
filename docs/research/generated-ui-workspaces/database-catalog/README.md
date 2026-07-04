# Database Catalog

Date: 2026-07-04

## Best-In-Class Patterns

- Data catalogs make database assets discoverable through summary counts, searchable/browsable schemas, table detail pages, column metadata, and trust/validation signals.
- Schema diagram tools make relationships and primary keys visible without forcing users to parse raw DDL or JSON.
- Lineage tools emphasize impact analysis: users should see which tables and columns reference or feed other assets.
- Good database documentation keeps raw schema exports available while making the visual catalog the primary workflow.

## Live Audit

Representative app: `examples/20_enterprise_erp_platform/output/app.py`.

Before server: `127.0.0.1:20901`.

Observed defects:

- `/ui/databases` reported `1 database(s), 0 schema(s), 0 table(s), 0 reference(s)` even though the generated app has Vendor, Customer, and Employee record tables.
- `/databases/ERPDB/schemas` returned an empty `schemas: []` payload.
- The page had no table/column browser, no primary-key visibility, no inferred indexes, and no useful schema map.
- Validation exposed a warning that `ERPDB does not declare schemas`, but the generated app could infer a usable schema from its entity model and database connection config.
- Raw validation JSON was prominent while the operational catalog was empty.

After server: `127.0.0.1:20902`.

After verification:

- `/ui/databases` renders `ERPDB / erp_platform` with Vendor, Customer, and Employee tables.
- The catalog exposes generated columns, type badges, primary-key badges, required/nullable constraints, inferred indexes, and schema summary cards.
- `/databases/ERPDB/schemas` returns the inferred `erp_platform` schema with the same generated table and column metadata.
- Validation is clean and raw validation JSON is available only through a details disclosure.

## Fix List

Must-fix:

- Infer database schemas from generated record entities when a database is declared without explicit schemas.
- Make schema JSON endpoints return the inferred schema, not an empty list.
- Render table and column metadata in the database catalog UI.

High-value polish:

- Add database summary cards and connection metadata.
- Show primary-key, required, nullable, reference, and index details.
- Move validation JSON behind a disclosure while keeping warnings visible.

## Validation

- Regenerated all 20 numbered examples.
- Live after audit: `assets/after-database-catalog.html` and `assets/after-schema-json.json`.
- Targeted tests: `3 passed` across database catalog regression, template route coverage, and CSS class coverage.
- Full suite: `1484 passed, 1 skipped, 3 warnings in 750.06s`.
