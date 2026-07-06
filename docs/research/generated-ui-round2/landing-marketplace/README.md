# Landing + Marketplace Round-2 Research

Date accessed: 2026-07-06

## Best-in-class reference

Commercial leader: Vercel Marketplace and deploy flow. Vercel is the strongest reference for a generated landing/marketplace surface because it compresses evaluation, integration selection, and deploy handoff into a fast developer experience.

Adjacent references:

- Salesforce AppExchange for listing fundamentals, install confidence, and buyer fit.
- Atlassian Marketplace for trust/security posture and marketplace app assurance.
- Retool templates for one-click demo/start-from-template expectations.

## Leader weaknesses

- Vercel's deploy and marketplace flow is excellent for cloud projects, but it assumes a hosted platform context and does not explain generated application internals.
- Salesforce and Atlassian marketplaces emphasize trust and listing depth, but the buyer often has to infer whether a capability fits their exact generated workspace.
- Retool templates are fast to start, but their demo/install proof depends on the platform. A generated APG app can show its own local proof: routes, OpenAPI, self-test, records, workflows, and agents.

## Differentiators proposed

1. Capability Compare Matrix: compare APG's generated local surfaces against marketplace expectations so users immediately see fit and coverage.
2. Live Demo Boot: provide a first-run path from landing to workspace, marketplace, OpenAPI, and self-test without a hosted deploy dependency.
3. Install Proof Ledger: show concrete generated proof items such as vendored assets, OpenAPI availability, self-test, and blueprint operation count.
4. Marketplace Fit Score: enrich every connector/blueprint card with local fit/proof metadata rather than only title/category/status.

## Prioritized implementation

Ship all four across `landing.html.j2` and `marketplace.html.j2`, backed by generator-derived metadata. Keep all links local and self-contained.
