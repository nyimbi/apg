# Database Catalog Round-2 Research

## Commercial leader

Prisma Studio is the best-in-class reference for visual database browsing and local developer ergonomics. Supabase's database tooling and dbt lineage/exposures are strong references for schema visibility and dependency context, while Metabase is the benchmark for approachable SQL exploration.

## Leader weaknesses

- Prisma Studio is excellent for browsing and editing records, but it is tied to Prisma projects rather than generated APG semantic/database catalogs.
- Supabase and dbt make schema and lineage visible, but the relationship view and generated app routes are separate surfaces.
- Metabase offers a strong SQL editor, but it is not automatically seeded from generated table metadata.
- None of the leaders combine schema diff, ER mini-map, and safe query snippets in every generated app without a database connection.

## Differentiators proposed

1. Schema Diff: summarize table/reference/warning changes from validation status.
2. ER Mini-map: expose table nodes and relationship edge count with jump links into the catalog.
3. Query Playground: generate safe preview SQL snippets from declared schema metadata.
4. Offline-first Catalog: work entirely from generated metadata with no live database required.

## Shipped verdict

APG now turns the database catalog into a schema operations cockpit. Before, the surface listed databases, schemas, tables, columns, references, and validation. After, it adds diff signals, a relationship mini-map, and generated query snippets that make the catalog actionable.
