# IDFD Capability Specification

## Purpose

IDFD provides first-class identity federation composition for APG applications.
It lets generated applications define, govern, operate, and inspect SAML, OIDC,
LDAP, and SCIM federation without binding the generated app to one concrete
identity-provider SDK.

The current packet also makes identity-federation governance agents first-class
citizens. Generated applications can register provider-neutral AI agents for
provider review, protocol review, claim mapping review, session-risk review,
certificate rotation review, SCIM review, privacy review, lifecycle batch
review, and federation stewardship. Agent runtimes such as `codex`,
`claude_code`, `opencode`, and `pi` remain adapter-backed; IDFD governs their
declared role, scope, owner, purpose, contribution disclosure, and human
approval posture.

## Scope

IDFD owns:

- Federation provider registration and metadata freshness.
- SAML, OIDC, LDAP, and SCIM protocol guardrails.
- Claim mapping between external identities and APG principals.
- Federated session issuance, revocation, risk handling, and MFA gates.
- Certificate/key lifecycle evidence for federation signing material.
- Health reporting and audit trails.
- Provider-neutral federation governance agents.
- Bytewax-first lifecycle batch validation for provider, protocol, claim,
  session, certificate, SCIM, review, and agent mutations.
- Contract-derived UI routes, view payloads, and theme tokens.
- Bytewax event-stream adapter evidence for batch federation mutation.

IDFD does not own:

- Password authentication or local account storage; that belongs to `auth`.
- MFA factor execution; that belongs to `mfau`.
- Encryption/key custody; those belong to `encr` and `keym`.
- Persistent storage migrations or live IdP protocol handshakes in the
  generated-app runtime.

## Configuration

The contract must expose tenant-scoped configuration sections for providers,
protocols, claims, sessions, SCIM, certificates, reviews, agents, streaming,
security, governance, observability, adapters, UI, and theme.

The `agents` section must define supported runtimes `codex`, `claude_code`,
`opencode`, and `pi`; supported federation-governance roles; privileged roles;
required owner, purpose, scope, contribution disclosure, and privileged-role
human approval; and the provider-neutral AICR adapter contract.

The `streaming` section must define `bytewax`, `idfd.lifecycle`, `event_time`
watermarking, lifecycle operations, lifecycle topics, and no broker-core/Kafka
dependency.

Required adapter evidence:

- `service.IdfdService` for generated runtime execution.
- `federation_runtime.py` for helper runtime functions.
- `api.py` and `views.py` for generated API/view payloads.
- `bytewax` for event-stream composition.
- `auth`, `mfau`, `encr`, `audl`, `secu`, `keym`, `moni`, and `cach` as
  integration adapter points.
- `aicr_provider_neutral_identity_federation_agent_adapter` for AI-assisted
  federation governance agents.

## Runtime Lifecycle

Provider lifecycle:

1. Register a tenant-local provider.
2. Validate owner, signing key, enabled protocol, metadata evidence, protocol
   constraints, and metadata freshness.
3. Refresh metadata or flag stale metadata for review.
4. Disable providers only when a reason is recorded and active sessions are
   revoked.

Claim mapping lifecycle:

1. Attach source and target claims to a tenant-local provider.
2. Require review for claim mappings.
3. Require privacy review for sensitive claims.
4. Audit mapping changes.

Session lifecycle:

1. Issue sessions only from active tenant-local providers.
2. Require MFA for privileged sessions.
3. Enforce session duration limits.
4. Require reauthentication/review for high-risk sessions.
5. Revoke sessions with recorded reasons.

Certificate lifecycle:

1. Register certificates against tenant-local providers.
2. Require key IDs.
3. Surface certificates close to expiry in health reports.
4. Require a new key for rotation.

## Rules

The rule engine is deterministic and returns `allow`, `require_review`, or
`deny`. Rules must cover tenant context, provider registration, protocol
security, claim mapping, SCIM, sessions, certificates, reviews, Bytewax batch
mutation, cross-tenant isolation, and state-change audit.

Rules must also cover unsupported federation-agent runtime, unsupported role,
missing scope, owner, purpose, missing machine-contribution disclosure,
privileged federation-agent registration without human approval, empty
lifecycle batches, unsupported lifecycle operations, and non-Bytewax lifecycle
batch routing.

## UI

The route manifest must include dashboard, providers, protocols, mappings,
sessions, certificates, SCIM, risk, reviews, agents, lifecycle, audit, and
settings. View payloads must be dependency-light dictionaries suitable for
generated Python apps.

## Theming

The default theme is `idfd_federation_console`. It defines compact density,
8px radius, status color tokens, and named theme components for provider grids,
protocol panels, mapping tables, session monitors, certificate timelines, SCIM
directory views, risk consoles, review queues, federation agent rosters,
Bytewax lifecycle panels, and audit timelines.

## Verification Requirements

The focused packet is serviceable when:

- The contract shape validates.
- The package self-test passes.
- The rule count is at least 44.
- The route count is at least 13.
- Bytewax adapter evidence is present.
- The runtime can register providers, map claims, issue sessions, register
  certificates, report health, revoke sessions, register federation agents,
  validate lifecycle batches, and isolate tenants.
- Focused IDFD tests pass without requiring full repository execution.
