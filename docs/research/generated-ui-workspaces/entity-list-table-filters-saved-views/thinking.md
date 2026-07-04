# Raw Reasoning

The previous entity-list page was already better than the original audit baseline because WP2 moved record creation into a drawer and WP3/WP4/WP7 had added shell quality. The remaining gap was workflow orientation. A user opening a Vendor or Customer list needs an answer to "which working set am I in?" before they need raw API JSON.

The stricter interpretation of this workspace is that "saved views" must be visible on the page, not merely implied by query parameters. The generated app has no durable user database for custom view definitions, so the self-contained compiler-safe version is generated semantic presets: All records, Recently added, and Active/status-derived views. This avoids adding persistence scope while satisfying the saved-view interaction model.

Query preservation mattered more than a richer filter UI in this slice. Existing advanced filters already expose every field, but users would lose context when paginating or sorting. Fixing URL generation centrally gives a more reliable base for later column-manager or multi-sort work.

Raw JSON remains useful for generated apps, but it should behave like developer tooling. Moving API JSON and rendered record JSON into a disclosure keeps power-user access without making the primary route feel like a debug console.

Rejected for this workspace:

- User-created saved views with persistence. This would require storage/schema decisions outside the current workspace slice.
- Multi-sort and column manager. Valuable, but larger than the must-fix/high-value polish target and better as a later table-v2 package.
- Client-side table state only. It would not satisfy server-rendered/no-JS progressive enhancement and would risk baseline drift without route coverage.

