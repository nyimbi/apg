# Kanban Round-2 Research

Accessed: 2026-07-05

## Commercial Leader

Jira Software is the best-in-class reference for mature kanban flow management because it combines boards with WIP limits and cumulative-flow reporting. Airtable and Linear are secondary references for lightweight board layouts and record/issue views.

## Findings

- Jira documents cumulative-flow diagrams as a way to understand work items across statuses.
- Atlassian describes WIP limits as constraints that expose bottlenecks before flow breaks down.
- Airtable kanban views make record cards configurable by field.
- Linear board layout is useful but is not available across every Linear surface.

## Leader Weaknesses

- Flow reports are often separate from the board instead of visible where work moves.
- WIP setup can require project administration before teams see bottleneck feedback.
- Lightweight boards often lack swimlane intelligence unless users configure fields manually.

## Differentiators Shipped

1. **Inline cumulative flow:** APG boards show cumulative counts and percentages directly above the kanban.
2. **Generated swimlanes:** APG detects common grouping fields such as owner, priority, country, team, tenant, segment, or type.
3. **Explainable WIP policy:** The generated WIP limit is visible as policy text and over-limit columns stay linked to filtered lists.

## Proposed Next Differentiators

- Render a cumulative-flow sparkline with historical snapshots.
- Allow users to pin a swimlane field in localStorage.
- Add aging limits per card using date fields.
