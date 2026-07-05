# Kanban Thinking

The existing APG board already had columns, drag/drop, keyboard move controls, WIP warnings, and filtered-list links. Round 2 should make board health more diagnostic.

Jira is the strongest reference for kanban flow because cumulative-flow diagrams and WIP policies are first-class agile reporting concepts. APG can beat that setup-heavy model by deriving flow rows and swimlane candidates from the generated entity schema.

The implementation should not add a new chart library or route. The board can render useful flow intelligence with server-side counts, preserving offline behavior and the current SortableJS drag/drop path.
