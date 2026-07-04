# Kanban

Workspace: `/ui/entities/<Entity>?view=kanban` via `compiler/templates/kanban_view.html.j2`.

## Best-in-Class References

- Trello: mature Kanban boards support fast keyboard workflows and clear card movement, not only pointer dragging.
- Linear: high-throughput issue boards are optimized for keyboard-first movement and fast status transitions.
- Jira/Atlassian Kanban: visible column counts and WIP limits make bottlenecks obvious.
- Notion board view: board views are just another grouped database surface, so a board should link back to the filtered underlying records.
- MDN drag-and-drop Kanban examples: drag/drop is a progressive enhancement, not the only interaction path.

## Live Audit

Representative app: `examples/20_enterprise_erp_platform/output/app.py`, booted locally at `127.0.0.1:20886`.

Route exercised: `/ui/entities/Vendor?view=kanban`, with Vendor records seeded through the generated API.

Defects found:

- Must-fix: the Kanban template used an unsupported Jinja `{% break %}` tag, causing `_render_template()` to fail and the generated app to silently fall back to the list page.
- Must-fix: card movement was pointer-drag only; keyboard users had no equivalent move workflow.
- Must-fix: successful server-side status changes from UI forms always redirected to the list instead of preserving the board context.
- Must-fix: columns had counts but no filtered list drill-through.
- Polish: no WIP/count guidance to flag overloaded columns.
- Polish: empty columns did not offer a path back to filtered records or creation.

Artifacts:

- `assets/before-vendor-kanban.html`
- `assets/before-vendor-kanban.headers`

## Fix Plan

Must-fix:

- Remove unsupported Jinja loop control and keep the template compatible with the default generated Jinja environment.
- Add native move controls on each card: select status, submit, and return to `?view=kanban`.
- Add route handling for `return_view=kanban` on server-rendered record update forms.
- Add filtered list links per column.

High-value polish:

- Add board summary metrics for total cards, column count, and WIP guide.
- Add WIP-style warnings when a column exceeds the generated guide.
- Improve empty column copy and filtered-list CTA.

## After Verdict

Implemented. The Kanban route now renders the actual board, supports both drag/drop and keyboard/server-rendered moves, preserves board context after moves, and makes column counts/drill-through visible.

