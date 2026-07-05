# Landing Page And Marketplace

## Verdict

Before: example 20's root landing routes returned `500 Internal Server Error` because the generated landing renderer asked the capability registry for a module-named theme that was not present. The marketplace route rendered, but with no installed connectors it became a passive empty state and a command snippet rather than a discovery surface.

After: `/` and `/home` render a purposeful generated-app landing page with workspace readiness, primary actions, data workspace links, integration readiness, and developer surfaces. `/ui/marketplace` renders marketplace discovery with search, category filters, generated connector blueprints, operation counts, status, and filtered results.

## Live Surface Audit

- Before app: example 20 booted on `127.0.0.1:20907`.
- Before `/` and `/home`: both returned `500`; server traceback showed `KeyError: 'enterprise_erp'` inside `APG_CAPABILITIES.capability_theme(MODULE_NAME)`.
- Before `/ui/marketplace`: returned `200`, but with `No connectors installed` and only `apg connector generate --spec openapi.yaml`.
- After app: regenerated example 20 booted on `127.0.0.1:20908`.
- After `/` and `/home`: both returned `200` and rendered `Generated APG workspace`, `Workspace readiness`, `Start here`, and `Integration readiness`.
- After `/ui/marketplace`: returned `200` and rendered generated blueprints: `Generated API`, `Record sync`, `Workflow webhooks`, and `Agent runtime`.
- After `/ui/marketplace?q=agent`: returned `200` with `1 of 4 shown` and the `Agent runtime` result.

## Must-Fix Items Completed

- Fixed landing route crash by guarding theme lookup failures.
- Replaced generic marketing-style landing with an operational generated workspace entry page.
- Replaced empty marketplace dead end with discoverable generated integration blueprints.
- Added marketplace search and category filters.
- Added regression coverage for landing, marketplace, and filtered marketplace output.

## Evidence

- `assets/before-root.html`
- `assets/before-home.html`
- `assets/before-marketplace.html`
- `assets/after-root.html`
- `assets/after-home.html`
- `assets/after-marketplace.html`
- `assets/after-marketplace-filtered.html`
