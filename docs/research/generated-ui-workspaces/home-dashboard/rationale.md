# Rationale

## Decisions

- Use generated `ENTITIES` metadata as the dashboard source of truth. It is present in every generated app and avoids missing optional `describe_application()` keys.
- Show record-owning entities in KPI cards. These are the dashboard objects a user can create, inspect, and analyze.
- Convert quick navigation to workspace actions. The first row now starts with the first primary entity and links to workflows, database catalog, marketplace, metrics, and API contract.
- Make empty activity actionable. The empty state links users to create or inspect the first primary record.
- Keep API/debug links available but secondary. Generated apps still need inspectability, but the Home workspace should lead with user workflows.

## Rejected Alternatives

- Rejected adding seeded demo data to make charts look fuller. That would misrepresent app state and pollute generated runtime behavior.
- Rejected a new dashboard-specific JavaScript module. Existing server-rendered cards and `apg-charts.js` already cover the interaction needs.
- Rejected expanding Home to include every route from the manifest. That recreates the API index problem and belongs in shell search or debug surfaces.
