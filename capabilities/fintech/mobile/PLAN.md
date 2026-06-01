# APG Mobile Banking Capability Build Plan

## Build Sequence

1. Replace the placeholder with a concrete APG capability contract.
2. Add domain models for programs, customers, trusted devices, authentication
   factors, account links, payments, bill payments, airtime purchases, service
   requests, notifications, fraud events, and evidence.
3. Add side-effect-free runtime helpers for code, country, currency, amount,
   device fingerprint, payment direction, and severity normalization.
4. Implement a tenant-scoped in-memory service that enforces deterministic
   rules before mutating state.
5. Add API helpers and view models that generated Python apps can compose.
6. Add an app entrypoint with semantic model, component manifest, and self-test.
7. Add package docs, manifest, release evidence, and focused tests.
8. Run package verification and APG audits.

## Architecture

- `capability_contract.py` owns identity, configuration, rules, UI, theme, and
  Bytewax lifecycle metadata.
- `models.py` contains explicit dataclass records and serializers.
- `mobile_runtime.py` contains normalization and derivation helpers.
- `service.py` owns lifecycle behavior and audit events.
- `api.py` exposes request-dict helpers.
- `views.py` builds dashboard and console view models.
- `app.py` publishes compiler/runtime surfaces.

## Review Criteria

The implementation is acceptable when:

- the APG registry validates the contract;
- lifecycle documents and release evidence exist;
- tests exercise the main mobile-banking lifecycle;
- guardrail tests cover tenant, policy, device, auth, payment, notification,
  fraud, Bytewax, and AI-agent rules;
- no stale placeholder marker remains;
- no disallowed broker terminology appears in the package;
- focused package audits and global capability audits pass.

## Deferred Adapter Work

Follow-up slices can add app-store automation, push gateway delivery, SMS/USSD
gateway delivery, device-attestation providers, live core banking posting,
card-network controls, mobile-money operator adapters, regulator filing,
rendered UI checks, durable Bytewax workers, and performance/load tests.
