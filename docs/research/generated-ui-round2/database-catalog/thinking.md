# Raw Thinking

The database catalog already renders a thorough static catalog. The missing value is scan speed: users need to know what changed, how tables relate, and how to start exploring data. A full SQL execution engine is out of scope and unsafe for generated offline apps, but query snippets are useful and deterministic.

Prisma Studio leads for local database inspection, Supabase/dbt lead for database and lineage visibility, and Metabase leads for approachable SQL authoring. APG can beat the combination for generated apps because the compiler owns the database schema, validation report, relationship graph, and static UI.

Rejected ideas:

- Executing SQL in the generated app. The catalog may not have a live database connection, and execution changes the safety profile.
- Full graph canvas. A compact mini-map is lighter and more robust across generated examples.
- Migration planner. Schema migrations belong in CLI tooling; this pass surfaces diff signals inside the UI.
