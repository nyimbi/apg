# Home Dashboard Workspace

## Best-in-Class Patterns

Best commercial dashboards behave like a decision surface, not a site map. Vercel and Stripe-style dashboards lead with the user's most likely next actions, keep metrics close to their drill-down targets, expose search/keyboard shortcuts, and reserve raw API/documentation links for secondary navigation. Dashboard UX guidance also emphasizes preattentive chart comprehension, strong hierarchy, and non-blank empty states.

## Live Audit

Representative app: `examples/20_enterprise_erp_platform/output`, booted locally at `http://127.0.0.1:20881/ui`.

Captured evidence:

- `assets/before-example20-ui.html`
- `assets/before-example20-ui.headers`

Findings:

- Must-fix: quick navigation was API-oriented (`Manifest`, `Component JSON`, `API Contract`) instead of helping the user start work.
- Must-fix: capability and agent summaries used missing `describe_application()` keys, producing empty or incorrect counts on example 20.
- Must-fix: dashboard stat cards included non-record workspace entities before the primary business entities.
- Must-fix: empty recent activity state said only "No activity yet" and gave no next action.
- Polish: summary cards were passive counts without links to Workflows, Marketplace, or Agent Console.
- Polish: entity/capability/agent lists were useful but did not form a clear "start here" hierarchy.

## Fix Plan

Must-fix:

- Compute Home dashboard groups directly from generated entity metadata.
- Restrict top stat cards to record-owning entities.
- Replace API-first shortcuts with workspace-first actions.
- Add actionable empty-state CTA for recent activity.

High-value polish:

- Make stat values and summary cards link to their drill-down surfaces.
- Show accurate agent/team counts and link to the first agent console.
- Keep secondary API links available but de-emphasized.

## Verification

Added `test_generated_home_dashboard_prioritizes_workspace_actions` to lock the user-focused shortcuts, actionable empty state, accurate agent/team counts, and capability summary.
