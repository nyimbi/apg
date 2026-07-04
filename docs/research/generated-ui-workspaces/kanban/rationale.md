# Rationale

## Decisions

- Remove Jinja `{% break %}` instead of enabling extensions.
  - Reason: generated apps should keep the default, dependency-light Jinja environment.
- Add server-rendered move controls to each card.
  - Reason: it gives keyboard and no-custom-JS users a reliable movement path.
- Preserve board context with `return_view=kanban`.
  - Reason: moving a card should keep the user in the board workflow.
- Link columns back to filtered table views.
  - Reason: board and table are complementary views over the same data.
- Use generated WIP guidance rather than persisted limits.
  - Reason: advisory counts are useful without adding a new DSL contract.

## Rejected Alternatives

- Hardcode status columns independent of record values.
  - Rejected because APG should reflect actual data and DSL semantics.
- Add a client-side-only command palette move action.
  - Rejected because the native form gives broader accessibility and simpler tests.
- Store WIP limits in localStorage.
  - Rejected because generated server HTML and tests would not see that state.

