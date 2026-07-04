# Entity List, Table, Filters, Saved Views

Workspace: `/ui/entities/<Entity>` via `compiler/templates/entity_list.html.j2`.

## Best-in-Class References

- Airtable: saved views make filtered/sorted slices first-class workspaces, not ad hoc query strings. The strongest pattern is a fast switcher for named views plus filter/sort controls that visibly explain why the current records are shown.
- Notion databases: one database can be viewed in multiple saved layouts, with filters and sorts attached to that view. The relevant generated-UI lesson is to keep view choice, filters, and sorting in one stable toolbar.
- NN/g data-table guidance: table UIs must support finding records, comparing rows, viewing/editing one row, and taking actions on records. The generated page already had table actions and create/edit affordances, but it under-served finding/orientation because filter state was hidden.
- NN/g filter guidance: filter categories and values should be predictable and visible. Generated `filter.<field>` controls are useful, but the previous surface hid them in an advanced disclosure without active chips or saved presets.

## Live Audit

Representative app: `examples/20_enterprise_erp_platform/output/app.py`, booted locally at `127.0.0.1:20882`.

Route exercised: `/ui/entities/Vendor`.

Defects found:

- Must-fix: no saved views despite the workspace requirement. Users landed on a generic table and had to reconstruct common slices manually.
- Must-fix: raw API JSON was promoted in the primary page navigation, competing with workflow actions.
- Must-fix: active filter state was hard to see. Search had a clear affordance, but field filters and sorting had no chip/summary.
- Must-fix: pagination and table header sort links did not preserve all filter state.
- Must-fix: the table wrapper did not emit the canonical `apg-table-wrap` class when records existed, weakening the mobile/overflow contract from WP2.
- Polish: empty state was serviceable, but it did not orient users to saved views or advanced filters.

Artifacts:

- `assets/before-vendor-list.html`
- `assets/before-vendor-list.headers`

## Fix Plan

Must-fix:

- Generate saved-view metadata per entity from semantic fields and current query state.
- Render saved-view tabs above search/filter controls.
- Render active filter chips for search, field filters, and sort state with direct clear links.
- Preserve current query state across pagination, page-size changes, and table sorting.
- Move API JSON and page record JSON into a developer exports disclosure.
- Emit `apg-table-wrap` around rendered tables.

High-value polish:

- Add a concise table toolbar that groups saved views, search, active chips, and advanced filters.
- Make status/state/stage/phase fields produce an `Active` saved-view preset even before data exists.
- Keep CSV export available but subordinate to the table workflow.

## After Verdict

Implemented. The entity list page now opens as a recognizable table workspace: saved views are visible first, active criteria are explicit, sort/pagination links preserve context, and developer exports are available without dominating the operator path.

