# Access Control Integration Hub Development Plan

## Slice Goal

Deliver a coherent lifecycle and guardrail packet for `composition_access` so the capability is executable, documented, testable, theme-aware, AI-agent aware, and Bytewax-aligned.

## Implementation Steps

1. Replace the generic contract with a domain-specific contract for providers, resources, policies, grants, sessions, decisions, agents, governance, UI, theme, adapters, and Bytewax lifecycle streaming.
2. Replace heavyweight package records with dependency-light runtime records that can compile and run in package tests.
3. Replace the legacy service entrypoint with `CompositionAccessService`, preserving the generated `create_record`/`list_records` compatibility surface.
4. Expand API helpers for each lifecycle operation.
5. Expand view models for dashboard, provider console, policy studio, grant workbench, decision explorer, agent workbench, and audit console.
6. Simplify package registration in `__init__.py` so importing the package does not require optional platform services.
7. Refresh `app.py`, `semantic_model.json`, `package_manifest.json`, and `release_report.json` from the active contract.
8. Add README and specification documents that explain purpose, lifecycle, guardrails, APIs, UI, theming, and verification.
9. Expand focused package tests around contract shape, rule execution, service lifecycle, guardrail failures, API/view surfaces, and semantic metadata.
10. Run battery-conscious verification: compile touched package files, run the focused access tests, check package metadata, and scan touched files for stale marker terms.

## Review Checklist

- Tenant context is enforced.
- Provider activation cannot skip metadata, secret reference, or evidence.
- Sensitive policies require conditions.
- High-risk policy activation requires simulation evidence and review.
- Privileged grants require approval, expiry, justification, and independent approval.
- High-risk sessions require step-up.
- Decision and batch operations require Bytewax.
- Access agents support Codex, Claude Code, OpenCode, and Pi.
- UI models expose operationally useful data without importing unavailable web frameworks.
- Package imports remain dependency-light.

## Known Deferred Work

- Bind provider adapters to live OIDC, SAML, LDAP, vault, and token services.
- Deploy live Bytewax topologies and durable audit sinks.
- Render browser UI screens and run visual checks.
- Add performance and concurrency validation after the full composition layer is stabilized.
