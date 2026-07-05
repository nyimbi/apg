# Entity Analytics Round-2 Research

Accessed: 2026-07-05

## Commercial Leader

Tableau is the best-in-class reference for entity analytics because it combines visual analytics, annotations, comparative measures, and forecasting with prediction intervals. Power BI is the secondary reference for forecast controls in an analytics pane.

## Findings

- Tableau supports annotations on visualization marks, points, and areas.
- Tableau forecasting includes configurable prediction intervals, giving users a visible uncertainty band.
- Tableau supports multiple-measure views and dual-axis comparison so related measures can be compared in one analytical pane.
- Power BI’s analytics pane also exposes forecast length and confidence interval controls for time-series visuals.

## Leader Weaknesses

- Forecasting and comparison controls usually require an analyst to configure the visual.
- Business intelligence tools often separate analysis setup from operational record drilldown.
- Prediction intervals and annotations can become chart-only metadata instead of linking back to the underlying records.

## Differentiators Shipped

1. **Generated decision intelligence:** APG analytics pages now expose annotation, comparison, and forecast cards without analyst setup.
2. **Chart metadata contract:** line chart specs carry `compare`, `forecast`, and `annotations` metadata for future richer hydration while preserving fallback rendering.
3. **Record-first drilldown:** each analytics decision links back to the generated entity list, keeping insight-to-record movement one click away.

## Proposed Next Differentiators

- Render forecast bands directly in `apg-charts.js` when uPlot is available.
- Let users save annotation pins as local generated records.
- Add per-field comparative overlays for numeric measures beyond record counts.
