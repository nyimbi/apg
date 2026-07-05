# Rationale

## Decisions

- Guard landing theme lookup with a broad exception fallback. Capability theme lookup is optional presentation data and should never break the root route.
- Make the landing page an operational entry page rather than a marketing splash. It now foregrounds workspace readiness, start actions, data workspaces, and integration readiness.
- Generate marketplace blueprints when no installed connectors exist. This keeps marketplace useful in a brand-new generated app and points users to the app's real local integration surfaces.
- Add search and category filters directly in the generated route/template. This is lightweight, deterministic, and works with installed connector manifests or generated blueprints.
- Keep all assets local and reuse existing CSS utilities to satisfy the asset and budget gates.

## Rejected Alternatives

- Remote marketplace catalog fetches were rejected because generated apps must be self-contained and offline-safe.
- Installing or configuring connectors from the UI was rejected because current generated apps do not have a connector installation state model.
- Keeping the command-only empty state was rejected because it is a dead end for users who are auditing the generated app through the UI.

## Validation Plan

- Targeted regression for landing and marketplace content.
- Template/CSS route coverage checks.
- Regenerate all 20 numbered examples after compiler/template changes.
- Live after-audit of `/`, `/home`, `/ui/marketplace`, and `/ui/marketplace?q=agent`.
- Full `uv run pytest tests/ -q`.
- PythonCodeGenerator hardcoded-literal tripwire.
