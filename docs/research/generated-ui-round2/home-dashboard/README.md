# Home Dashboard Round-2 Research

Accessed: 2026-07-05

## Commercial Leader

Grafana is the best-in-class reference for operational dashboards because it combines composable panels, alerting, annotations, and scheduled reports in one mature dashboard surface. Tableau is a secondary reference for business-facing annotations and data-driven alerts.

## Findings

- Grafana alerting watches metric or log data for conditions and pushes responders toward action instead of requiring manual dashboard checks.
- Grafana reporting can schedule dashboard exports as PDF/CSV email packets.
- Tableau supports annotations on marks, points, and areas, which makes chart context travel with the visualization.
- Tableau data alerts notify users when dashboard values cross thresholds.

## Leader Weaknesses

- Dashboard composition, annotations, alerts, and reporting are separate setup flows rather than one operator command center.
- Heavy BI/observability products usually assume configured data sources before the surface becomes useful.
- Scheduled reporting often targets administrators or analysts, while frontline app users need offline packets directly in the generated workspace.

## Differentiators Shipped

1. **Dashboard command center:** APG renders tile composition, alert watches, annotation pins, and scheduled export controls above the live stats so the dashboard starts with action controls.
2. **Zero-setup thresholds:** Generated entity tiles get inline threshold watches immediately from local record counts.
3. **Portable annotation pins:** Status charts expose contextual pins linking directly into entity analytics.
4. **Offline export affordance:** Scheduled export rows and a browser-native print/PDF action ship without a server job or third-party service.

## Proposed Next Differentiators

- Persist tile order in localStorage and promote the saved view into URL state.
- Add generated CSV packet assembly that includes dashboard metrics plus per-entity exports.
- Let users pin annotations from chart drilldowns and store them in the local record store.
