# Entity Analytics Thinking

The existing analytics page already had a line chart, status donut, numeric statistics, drilldown rows, and insights. Round 2 should move it closer to a decision surface by making uncertainty, comparison, and annotations explicit.

Tableau is the right leader because its analytics story includes annotations, dual-axis comparisons, and forecast prediction intervals. APG can beat it in generated apps by deriving the analytics context from the entity schema and linking every insight back to records.

The implementation should avoid a larger chart runtime change in this slice. Adding metadata to chart JSON plus visible decision cards gives tests and future chart hydration a stable contract without increasing dependency or JS size.
