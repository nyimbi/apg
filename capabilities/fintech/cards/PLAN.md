# Digital Cards Capability Plan

## Implementation Cycle

1. Replace the placeholder package with a complete executable APG capability.
2. Define a deterministic contract with configuration, dependencies, rules, UI,
   theming, Bytewax metadata, and provider-neutral card-agent composition.
3. Add dependency-light models for card programs, cardholders, cards, tokens,
   authorizations, disputes, and evidence.
4. Add runtime helpers for normalized codes, currencies, amounts, card masks,
   risk bands, and authorization decisions.
5. Add `CardService` methods for program registration, cardholder onboarding,
   card issuance, token provisioning, authorization decisions, dispute filing,
   agent registration, batch validation, and dashboard summaries.
6. Add process-local API helpers and framework-neutral view models.
7. Add publishable `app.py` with semantic model, component manifest, and
   self-test evidence.
8. Add focused tests covering contract shape, rules, lifecycle behavior,
   guardrails, API/view behavior, and publishability.
9. Refresh generated evidence and update shared registry/progress docs.
10. Run focused verification and commit a coherent slice.

## Review Checklist

- Rules and service contexts match exactly.
- Bytewax is the only event-stream processor.
- Card agents remain provider-neutral across Codex, Claude Code, OpenCode, and
  Pi.
- High-impact authorization, limit override, and privileged-agent paths require
  explicit human approval.
- Live issuers, networks, token providers, 3DS, embossers, and PCI systems
  remain behind adapters.
- APG inspect, publish-plan, implementation-audit, lifecycle-audit, and strict
  package audit all pass before commit.
