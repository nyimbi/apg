# Rationale

## Decisions

- Compute the trend chart from the first available date-like field or record key.
  - Reason: generated apps frequently receive API-created records with operational timestamps even when the DSL does not declare them.
- Use 30 buckets ending at the latest record date.
  - Reason: this keeps chart output deterministic for tests and avoids coupling generated examples to wall-clock time.
- Add status drill-through links back to the table.
  - Reason: this turns analytics into a workflow, not a passive report.
- Add headline metric tiles.
  - Reason: users should understand scope and data quality before interpreting charts.
- Preserve empty states when no date/status/numeric data exists.
  - Reason: no chart is better than a fake chart.

## Rejected Alternatives

- Use current system date for the trend window.
  - Rejected because example baselines would drift over time.
- Require declared `created_at` in entity schemas.
  - Rejected because the compiler should improve available records without changing DSL requirements.
- Add new chart dependencies.
  - Rejected because the existing vendored chart pipeline is sufficient for this workspace.

