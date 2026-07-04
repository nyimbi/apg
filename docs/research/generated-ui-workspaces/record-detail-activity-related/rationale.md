# Rationale

## Decisions

- Compute `related_count` in Python instead of Jinja.
  - Reason: avoids brittle template aggregation and is easier to test.
- Show related candidates even when empty.
  - Reason: modeled relationships should be visible as affordances.
- Prefer human-readable title fields.
  - Reason: users recognize `legal_name` or `full_name` faster than generated numbers.
- Add copy-link and previous/next controls.
  - Reason: record review and handoff workflows need fast navigation.
- Reuse filtered entity list URLs for related records.
  - Reason: it builds on the entity-list workspace and avoids new custom routes.

## Rejected Alternatives

- Custom related-record creation drawers.
  - Rejected because form UX is a separate workspace and should be fixed consistently.
- Keep raw JSON fallback for related failures.
  - Rejected because hiding template errors behind raw JSON breaks the workspace contract.
- Client-side-only related filtering.
  - Rejected because server-rendered links are simpler, testable, and accessible.

