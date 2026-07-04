# Rationale

## Decisions

- Use generated semantic saved views rather than custom persisted views.
  - Reason: every generated app can support this with no new runtime dependency, no storage migration, and no user-account assumptions.
- Use `status`/`state`/`stage`/`phase` as the first semantic filter source.
  - Reason: those fields already power kanban and analytics eligibility, so table views now align with other generated workspaces.
- Generate query URLs server-side for pagination, page size, and table sorting.
  - Reason: a single helper prevents state loss and keeps server-rendered HTML correct without JavaScript.
- Move JSON/API affordances into "Developer exports".
  - Reason: generated apps need inspection surfaces, but operators should see workflow actions first.
- Add a small CSS component block for saved-view tabs and filter chips.
  - Reason: component classes are clearer and easier to validate than more ad hoc utility combinations.

## Rejected Alternatives

- Persist named views in localStorage only.
  - Rejected because it would disappear across browsers and would not be visible in server-rendered HTML or tests.
- Add a generated SQLite/file-backed saved-view store.
  - Rejected as a scope expansion requiring data lifecycle and auth/ownership decisions.
- Keep API JSON in the top nav.
  - Rejected because this workspace should optimize for list/table work, not debugging.

