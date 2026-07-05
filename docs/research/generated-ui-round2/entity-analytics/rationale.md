# Entity Analytics Rationale

## Decisions

- **Leader:** Tableau, with Power BI as a secondary forecast-control reference.
- **Shipped first:** generated decision cards and chart metadata, because the existing chart hydrator already renders safe fallbacks.
- **Kept drilldown-first:** every analytics decision links to the generated entity table instead of becoming static chart decoration.

## Rejected Alternatives

- **Immediate uPlot band rendering:** rejected for this slice to avoid expanding chart JavaScript before a dedicated chart-runtime budget pass.
- **External statistics library:** rejected because generated apps must stay dependency-free beyond Flask and Jinja2.
- **Analyst-authored calculation editor:** rejected because the compiler can already derive useful baseline analytics from entity metadata.

## Verification Intent

The analytics tab should keep existing line/donut specs valid while adding visible decision cards and metadata for forecast, comparison, and annotation behavior.
