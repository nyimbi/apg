# Entity List Rationale

## Decisions

- **Leader:** Linear for saved views and keyboard-first list speed; Airtable as the flexible record-grid comparison.
- **Shipped first:** list intelligence because APG already had the underlying saved views, filters, bulk bar, and CSV export.
- **Persisted locally:** density is a per-user preference and belongs in `localStorage`, not generated app server state.

## Rejected Alternatives

- **New virtual scrolling library:** rejected because generated apps already page server-side and no new runtime dependencies are allowed.
- **Full column drag reorder:** rejected for this slice because it needs broader table rendering changes and persistence semantics.
- **XLSX dependency:** rejected because generated `requirements.txt` must not gain new dependencies.

## Verification Intent

Regenerated entity pages should show the list intelligence strip, keep existing saved-view tests green, and stay within the static JS budget because the enhancement is inline and small.
