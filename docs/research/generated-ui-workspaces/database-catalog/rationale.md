# Rationale

## Decisions

- Infer schemas only for database declarations that have no explicit schema list, preserving authored schemas when present.
- Use the database connection config or schema field default as the inferred schema name.
- Treat generated record entities as database tables and add a synthetic `id` primary-key column because generated records are keyed by `id`.
- Preserve field `required` state as column constraints and derive basic indexes from required/reference columns.
- Keep `/databases/<name>/schemas` aligned with the UI by making both call `list_databases()`.
- Hide raw validation JSON behind a details disclosure and surface warnings as readable cards.

## Rejected Alternatives

- Adding a full ERD/SVG layout engine now: rejected as too broad for this workspace and unnecessary for the main broken empty-catalog defect.
- Showing raw `SEMANTIC_MODEL` tables directly: rejected because the API/UI contract is `list_databases()` and `/databases/<name>/schemas`.
- Treating missing schemas as only a warning: rejected because the generated app can infer a useful schema from its own entity metadata.
- Adding a database documentation dependency: rejected because generated apps must remain self-contained.
