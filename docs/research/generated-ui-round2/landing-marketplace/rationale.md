# Landing + Marketplace Rationale

## Decisions

- Named Vercel Marketplace/deploy flow as the leader because this workspace is about landing-to-install momentum, not just catalog browsing.
- Added capability compare and install proof using generated APG metadata. This creates evidence users can inspect locally.
- Added live demo boot as local first-run actions rather than hosted deploy. APG generated apps must remain self-contained.
- Added fit score/proof data to marketplace cards so blueprints communicate readiness and coverage.

## Rejected alternatives

- Hosted deploy button: rejected because it implies a specific cloud platform and external runtime.
- External marketplace SDK: rejected by no-new-dependency and offline constraints.
- Reviews, ratings, or social proof: rejected because generated apps should not fabricate marketplace credibility.
- Mutating install flow: rejected because connector installation is a product/runtime decision beyond the generated UI pass.

## Budget note

The implementation is CSS plus Jinja/Python metadata. It adds no generated Python dependencies and no external runtime URLs.
