# Entity List Round-2 Research

Accessed: 2026-07-05

## Commercial Leader

Linear is the best-in-class reference for fast operational lists because saved custom views, URL-shareable filtered issue lists, and keyboard-first navigation are core to the product. Airtable is the secondary reference for flexible record grids and alternate views.

## Findings

- Linear custom views create durable filtered views that can be saved and shared with a workspace.
- Airtable kanban/grid surfaces let teams reshape the same records into multiple views, but users still need to configure those views per base.
- Airtable form and prefill documentation reinforces a larger pattern: state is often encoded into URLs or separate setup surfaces instead of being explained inline.

## Leader Weaknesses

- Linear is excellent for issues but not generated arbitrary business entities.
- Airtable is flexible, but view setup can become an admin/configuration activity instead of a runtime user affordance.
- Table products often hide virtualization, bulk actions, and export constraints until the user hits scale.

## Differentiators Shipped

1. **List intelligence strip:** APG explains shareable URL state, column memory, virtual-window size, and keyboard search directly above the table.
2. **Persisted density toggle:** Cmd/Ctrl-D switches compact/comfortable rows and saves the choice per entity in `localStorage`.
3. **Column memory cues:** sortable field chips make column state visible without leaving the list.
4. **Keyboard-first fuzzy focus:** `/` focuses the generated entity search from anywhere on the page.

## Proposed Next Differentiators

- Persist explicit hidden/shown column sets per user and apply them server-side.
- Add client-side fuzzy highlighting within the current virtual window.
- Add offline XLSX-compatible export by generating a tab-separated download beside CSV.
