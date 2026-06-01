# Know Your Customer Capability Development Plan

## Current Slice

Promote `capabilities/fintech/kyc` from an empty package into a first-class APG
capability with executable local KYC behavior.

## Implementation Steps

1. Replace empty package metadata with APG package exports.
2. Add a concrete contract with KYC configuration, deterministic rules, routes,
   theme, dependencies, and Bytewax streaming.
3. Add customer profile, document, screening, decision, and evidence models.
4. Add service methods for profile lifecycle, documents, screening, risk,
   verification decisions, agents, batches, and summaries.
5. Add process-local API helpers and framework-neutral view models.
6. Add publishable app self-test and semantic model.
7. Add focused package tests for lifecycle and guardrails.
8. Update package evidence, root catalog, and progress log.

## Adapter Boundary

Keep live document vendors, biometric providers, sanctions/PEP feeds, adverse
media services, government registries, payment providers, wallets, audit sinks,
notifications, identity, consent, key management, and durable Bytewax workers
behind adapters.

## Review Checklist

- Customer profile writes enforce tenant, policy, consent, type, and country.
- Document verification rejects unsupported types, missing token references, and
  low confidence.
- Screening hits and high risk scores require review before verification.
- Verification requires identity, address, screening, risk, and consent
  evidence.
- Non-Bytewax batches are denied.
- Agent runtimes include Codex, Claude Code, OpenCode, and Pi.
- UI routes and theme are contract-backed.
