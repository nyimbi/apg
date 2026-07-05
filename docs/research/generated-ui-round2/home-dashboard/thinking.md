# Home Dashboard Thinking

The brief asked for composable tiles, threshold alerts, annotations, and scheduled export. Existing APG already had chart hydration, sparkline tiles, status donuts, recent activity, workflows, and agent summary. The highest-leverage Round-2 move is therefore not another chart type; it is integrating the four operator controls into one visible dashboard command center.

Grafana is stronger than generic BI tools for this surface because it treats dashboards as live operational consoles, not static reports. Tableau is still useful as a reference for annotations and business alerts. The weakness common to both is setup burden: users must configure a data source, build a dashboard, configure alert rules, and separately configure reports. APG can beat that because the compiler already knows entities, records, workflows, and local assets.

The implementation should stay dependency-free and generated-app compatible. That rules out a scheduler daemon or chart editor. The first pass should ship visible, testable HTML generated from existing entity metadata and record counts. Future persistence can use localStorage or generated records, but that is not needed to make the differentiator visible and safe.
