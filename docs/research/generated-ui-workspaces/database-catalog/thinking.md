# Raw Reasoning

The database catalog should answer three questions immediately:

1. What database and schema does this generated app use?
2. What tables and columns exist, and which fields are primary keys or required?
3. What references or indexes matter for impact analysis?

The before-state answered none of those for example 20. The DSL declared `ERPDB` with a schema name, and the generated app had record entities, but the database object carried an empty `schemas` array. That made both UI and JSON API surfaces look broken: the catalog was technically valid but operationally empty.

Best-in-class references point to the same repair: treat the catalog as a metadata browser, not a raw JSON dump. DataHub and OpenMetadata foreground assets, schema, and lineage. dbdiagram-style tools make table relationships legible. Database design tools treat primary keys, foreign keys, and indexes as first-class facts.

The strictest implementation is to infer a generated schema only when the database has no explicit schemas. Explicit schemas still win. For generated apps, entity fields are the reliable source for a useful table model; adding a synthetic `id` primary key matches the generated record store and eliminates the false no-primary-key warning.

Rejected: requiring every APG author to manually declare database schemas before the catalog is useful. The compiler already has enough entity metadata to generate a helpful catalog. Rejected: building a new diagram renderer for this pass. A table/column browser plus reference map is the higher-value fix and stays inside the self-contained asset budget.
