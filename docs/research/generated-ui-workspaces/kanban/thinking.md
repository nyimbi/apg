# Raw Reasoning

The biggest Kanban defect was not visual polish; it was that the page often was not Kanban at all. The renderer swallowed template exceptions and fell back to the entity list. The unsupported `{% break %}` tag was already present in the template, so this workspace needed a compatibility fix before any UX polish mattered.

Pointer drag-and-drop is a good enhancement but cannot be the only way to move a card. Generated apps already have a partial record update route, so the lowest-risk accessible path is a small form on every card that posts the new status. Adding `return_view=kanban` avoids stranding the user back on the list after a move.

WIP limits in real systems are configurable. The generated app has no DSL-level WIP contract yet, so the implementation uses a visible "WIP guide" derived from board size. It is advisory, deterministic, and avoids inventing persistent configuration.

Rejected for this workspace:

- Fully configurable columns or WIP limits. Useful, but it needs DSL/runtime persistence decisions beyond this slice.
- New drag-and-drop library work. SortableJS is already vendored and sufficient.
- Client-only keyboard shortcuts. Native form controls give keyboard access without relying on JavaScript.

