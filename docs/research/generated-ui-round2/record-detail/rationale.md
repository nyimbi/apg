# Rationale

## Decision

Ship a schema-derived record intelligence command center with three cards:

- Change Diff Timeline: revision, activity count, and the highest-signal populated fields.
- Related Record Graph: compact relationship nodes built from existing FK-derived related lists.
- Create Sibling Context: safe non-ID fields that help users recreate a similar item without copying internal identifiers.

## Why this beats the benchmark

Notion, Airtable, and Linear each excel in a slice of the problem, but they do not automatically produce an operational context cockpit for arbitrary generated business entities. APG can because the compiler controls the schema, generated routes, activity events, and related-list discovery.

## Rejected alternatives

- Graph canvas: rejected because it would increase JS/CSS complexity and risk layout failures on small generated apps.
- Clone route: rejected because true clone semantics require product policy for unique fields, relationships, and permissions.
- Fake historical diffs: rejected because prior values are not persisted in the generated event log.

## Validation target

The generated `record_detail.html.j2` page must still render existing details, related lists, activity, copy link, and next-record navigation while adding the new intelligence cards.
