# Cross-Border Remittance Capability Plan

## Implementation Cycle

1. Replace the placeholder package with a complete executable APG capability.
2. Define a deterministic contract with configuration, dependencies, rules, UI,
   theming, Bytewax metadata, and provider-neutral AI-agent composition.
3. Add dependency-light models for quotes, transfers, payouts/refunds, and
   evidence records.
4. Add runtime helpers for normalization, corridor IDs, transfer amounts, risk
   bands, payout methods, and lifecycle decisions.
5. Add `RemittanceService` methods for quote creation, transfer creation,
   payout release, refund filing, agent registration, batch validation, and
   dashboard summaries.
6. Add process-local API helpers and framework-neutral view models.
7. Add publishable `app.py` with semantic model, component manifest, and
   self-test evidence.
8. Add focused tests covering contract shape, rules, service lifecycle,
   guardrails, API/view behavior, and app publishability.
9. Refresh generated evidence and update shared registry/progress docs.
10. Run focused verification and commit a coherent slice.

## Review Checklist

- Rules and service contexts match exactly.
- Bytewax is the only event-stream processor.
- AI agents remain provider-neutral across Codex, Claude Code, OpenCode, and Pi.
- High-impact money movement requires explicit human approval where configured.
- Live providers remain behind adapters.
- APG inspect, publish-plan, implementation-audit, lifecycle-audit, and strict
  package audit all pass before commit.
