# Home Dashboard Rationale

## Decisions

- **Leader:** Grafana, with Tableau as a secondary business-dashboard reference.
- **Shipped first:** an integrated dashboard command center because APG can render it from compile-time and in-memory app knowledge without extra setup.
- **Kept offline:** all controls use generated HTML/CSS and existing local routes.

## Rejected Alternatives

- **Full drag-and-drop layout editor:** rejected for workspace 1 because it requires persistent layout storage and more JavaScript budget. The shipped ordered tile list establishes the product affordance first.
- **Background scheduled email service:** rejected because generated apps must remain stdlib+Flask+Jinja2 with no new runtime dependencies.
- **External chart export library:** rejected because the ground rules prohibit CDN/external runtime dependencies and the browser print path covers PDF/PNG-style export without new assets.

## Verification Intent

Generated examples should include the command center in `/ui`, retain existing chart specs, and keep static JavaScript within the gzip budgets.
