# Rationale

## Decision

Ship a database intelligence strip in `database_catalog.html.j2`: schema diff, ER mini-map, and query playground. Compute it from `database_status()`, `list_databases()`, and `relationship_graph()`.

## Why this beats the benchmark

The benchmark tools excel in specific contexts, but APG can show schema health, relationships, and starter queries in every generated app without requiring a live database editor or external service.

## Rejected alternatives

- SQL execution: rejected because generated apps may be metadata-only and should not gain a write/read execution surface here.
- Full graph library: rejected because it would add JS/CSS weight and a dependency-like surface.
- Migration diff planner: rejected because CLI migration planning already owns that deeper workflow.

## Validation target

Generated database catalog HTML must still render database status, schema links, table columns, reference map, and validation warnings while adding schema diff, ER mini-map, and query playground content.
