# Central Configuration Management Development Plan

## Slice Goal

Deliver a coherent lifecycle and guardrail packet for `composition_config` so the capability is executable, documented, testable, theme-aware, AI-agent aware, and Bytewax-aligned.

## Implementation Steps

1. Replace the generic contract with a domain-specific contract for namespaces, configurations, deployments, templates, drift, agents, governance, UI, theme, adapters, and Bytewax lifecycle streaming.
2. Replace heavyweight runtime records with dependency-light records that can compile and run in focused package tests.
3. Replace package service, API, views, app, and registration entrypoints with dependency-light lifecycle surfaces.
4. Refresh package evidence from the active contract.
5. Add README, specification, and plan documents.
6. Expand focused tests around contract shape, rules, service lifecycle, guardrail failures, API/view surfaces, app self-test, and semantic metadata.
7. Run focused verification and commit the coherent packet.

## Review Checklist

- Tenant context is enforced.
- Namespace ownership and environment are required.
- Restricted values require schemas.
- Secret values require secret references and return redacted values.
- Activation requires validation evidence.
- Production deployments require approval.
- High-impact deployments require canary evidence.
- Deployments, rollbacks, and batch changes require Bytewax.
- Configuration agents support Codex, Claude Code, OpenCode, and Pi.
- Package imports remain dependency-light.

## Deferred Work

- Bind to live secret managers, external configuration stores, notification systems, and durable audit sinks.
- Deploy live Bytewax topologies.
- Render browser UI screens and run visual checks.
- Add performance and concurrency validation after the composition layer stabilizes.
