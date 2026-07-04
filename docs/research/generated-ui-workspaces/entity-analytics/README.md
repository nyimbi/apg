# Entity Analytics

Workspace: `/ui/entities/<Entity>?view=analytics` via `compiler/templates/entity_analytics.html.j2`.

## Best-in-Class References

- Metabase: analytics are strongest when charts support drill-through into the underlying records. Generated entity analytics should not stop at pictures; every visible segment should answer "which rows caused this?"
- Grafana: good dashboards use clear panel hierarchy, variables/links, and concise summaries before detailed panels. For APG, entity-level analytics should start with record, recency, status, and measure summaries.
- Tableau: dashboard actions let one view filter another. The generated equivalent is status rows that deep-link back to the entity table with the matching `filter.<field>` applied.
- NN/g dashboard guidance: visualizations should use preattentive-friendly encodings and make chart meaning easy to understand. The previous generated trend chart did not communicate a real trend because it plotted a flat placeholder.

## Live Audit

Representative app: `examples/20_enterprise_erp_platform/output/app.py`, booted locally at `127.0.0.1:20884`.

Route exercised: `/ui/entities/Vendor?view=analytics`, with three Vendor records seeded through the generated API.

Defects found:

- Must-fix: the "Records Over Time" chart used a placeholder series with 30 identical y-values instead of real date buckets.
- Must-fix: status distribution was visual only; users could not drill from a segment to matching records.
- Must-fix: no headline metrics explained total rows, recent activity, status count, or available measures before the charts.
- Must-fix: no actionable insight summarized the largest segment or trend date window.
- Polish: the no-numeric-fields empty state was technically correct but disconnected from the entity context.

Artifacts:

- `assets/before-vendor-analytics.html`
- `assets/before-vendor-analytics.headers`

## Fix Plan

Must-fix:

- Derive a real 30-day records-over-time series from the first available date-like field or record key (`created_at`, `updated_at`, `date`, etc.).
- Add summary metric tiles for records, recent records, status segments, and numeric measures.
- Add status drill-through rows linking to the entity table with the corresponding filter applied.
- Add insight cards for largest status segment and trend date window.

High-value polish:

- Parse numeric strings as measures when the schema marks a field numeric.
- Explain when trend history is unavailable because no date data exists.
- Keep the analytics page server-rendered and functional without client-side state.

## After Verdict

Implemented. Entity analytics now render as a useful investigation surface: metric tiles summarize the entity, the trend chart is based on date buckets, status rows drill into filtered records, and insight cards surface the largest segment and trend window.

