# Raw Reasoning

The prior analytics page satisfied WP3's basic chart presence requirement but not the workspace-excellence bar. It showed charts, but the line chart was a flat generated sequence unrelated to actual record dates. That is worse than an empty state because it implies a time trend where none was computed.

The strictest useful fix is to compute analytics from only data the generated app already has. APG records may include extra fields submitted through API payloads, so date detection should not depend only on declared entity fields. The implementation checks declared date fields first, then common record keys like `created_at`, `updated_at`, `date`, and `timestamp`.

Status drill-through is the most important workflow improvement. A user who sees "active: 2" should be one click away from the filtered table. This reuses the same `filter.<field>` route contract improved in the entity-list workspace.

Numeric statistics should stay compact. Full histograms or distributions would be valuable, but they require more chart types and more visual tuning. The current min/avg/max cards are enough to make numeric fields useful without exceeding this workspace slice.

Rejected for this workspace:

- Client-only chart interactions. They would not work without JavaScript and would be harder to test from generated HTML.
- Cross-filtering charts in place. Useful long-term, but the generated route contract already gives drill-through with less risk.
- Synthetic trend data when no date exists. Rejected because fake trends undermine trust.

