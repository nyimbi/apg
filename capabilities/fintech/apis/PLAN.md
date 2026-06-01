# APG Banking APIs Capability Build Plan

## Build Sequence

1. Replace the placeholder with an executable APG capability contract.
2. Add domain models for API products, developers, applications, consents,
   clients, endpoint policies, webhooks, API calls, rate limits, SLA incidents,
   and evidence.
3. Add side-effect-free runtime helpers for normalization, scopes, limits,
   client ids, and rate-limit decisions.
4. Implement a tenant-scoped in-memory service that enforces deterministic
   rules before mutating state.
5. Add API helpers and view models for generated Python applications.
6. Add an app entrypoint with semantic model, component manifest, and self-test.
7. Add package docs, manifest, release evidence, and focused tests.
8. Run package verification and APG audits.

## Architecture

- `capability_contract.py` owns identity, configuration, rules, UI routes,
  theme tokens, and Bytewax lifecycle metadata.
- `models.py` contains explicit dataclass records and serializers.
- `apis_runtime.py` contains normalization and policy helpers.
- `service.py` owns lifecycle behavior and audit events.
- `api.py` exposes request-dict helpers.
- `views.py` builds operational and developer-console view models.
- `app.py` publishes compiler/runtime surfaces.

## Review Criteria

The implementation is acceptable when:

- the APG registry validates the contract;
- lifecycle documents and release evidence exist;
- tests exercise the main banking-API lifecycle;
- guardrail tests cover tenant, policy, developer, consent, key, endpoint,
  webhook, rate-limit, incident, Bytewax, and AI-agent rules;
- no stale placeholder marker remains;
- no disallowed broker terminology appears in the package;
- focused package audits and global capability audits pass.

## Deferred Adapter Work

Follow-up slices can add live API gateway deployment, OAuth/consent screens,
mTLS certificate validation, webhook delivery/retry workers, developer portal
hosting, regulator filing, rendered UI checks, durable Bytewax workers, and
performance/load tests.
