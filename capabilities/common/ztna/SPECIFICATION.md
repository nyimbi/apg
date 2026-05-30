# ZTNA Capability Specification

## Purpose

`ztna` provides a composable Zero Trust Network Access capability for APG
applications. It turns identity, device, resource, risk, policy, session, and
audit state into executable access decisions with deterministic guardrails and
UI-ready view models.

The capability does not open network tunnels or talk directly to live identity,
device, MFA, policy, gateway, or SIEM providers. Those integrations are
adapter responsibilities. The local package proves the domain lifecycle,
contract, rules, state transitions, UI payloads, and package evidence.

## Scope

In scope:

- tenant-scoped identity records;
- verified, privileged, suspended, and MFA-complete identity state;
- tenant-scoped device records;
- posture, trust score, managed-device, attestation, compliance, and quarantine
  state;
- protected resource records;
- standard and privileged access levels;
- sensitive resources, resource policies, and network segments;
- deterministic rule evaluation;
- access requests with approved, review-required, active, and denied outcomes;
- independent access review;
- governed session start, reevaluation, revocation, and closure;
- append-only audit events;
- route, permission, view-model, theme, and adapter metadata;
- package self-test, semantic model, manifest, release report, audit, and
  publish-plan evidence.

Out of scope for the local package:

- live tunnel establishment;
- service mesh policy pushes;
- packet inspection;
- live endpoint posture collection;
- live IdP or MFA handshakes;
- browser rendering;
- persistent database migrations;
- live Bytewax execution.

## Users

- Application builders composing APG security capabilities.
- Security operators reviewing access and session state.
- Platform engineers connecting APG adapters to identity, MFA, posture, gateway,
  audit, risk, and monitoring providers.
- Generated applications that need dependable zero-trust access semantics.

## Domain Model

The runtime owns these records:

- `ZeroTrustIdentityRecord`
- `ZeroTrustDeviceRecord`
- `ZeroTrustResourceRecord`
- `ZeroTrustAccessRequestRecord`
- `ZeroTrustSessionRecord`
- `ZeroTrustAuditEventRecord`

All business IDs include tenant context so repeated business keys in different
tenants produce different record IDs.

## Lifecycle

### Identity

1. Register identity with tenant, subject, display name, privilege state, MFA
   state, and optional federated provider.
2. Verify identity.
3. Use verified identity state in access decisions.
4. Deny access for unverified or suspended identities.

### Device

1. Register a device under a tenant-local identity.
2. Capture trust score, posture, compliance, managed-device state, and
   attestation.
3. Mark trusted devices as `trusted`; otherwise mark them `quarantined`.
4. Update posture as runtime signals change.
5. Deny access when posture, trust, compliance, or required attestation fails.

### Resource

1. Register protected resource with tenant, access level, sensitivity, policy
   attachment, policy ID, and segment.
2. Require resource name and network segment.
3. Require policy before access.
4. Require microsegmentation evidence for sensitive resources.

### Access

1. Request access using identity, device, resource, requester, MFA evidence,
   review evidence, least-privilege scope evidence, explicit-decision evidence,
   JIT approval evidence, and optional risk score.
2. Evaluate deterministic rules.
3. Deny on hard guardrail failure.
4. Route to review on high-risk, privileged, unmanaged privileged,
   least-privilege, explicit-decision, or microsegmentation gaps.
5. Approve immediately only when all hard and review guardrails pass.

### Review

1. Review-required requests cannot start sessions.
2. Reviewer must be independent from requester.
3. Review decision records reviewer, timestamp, and audit event.
4. Approved reviews clear required actions.

### Session

1. Start sessions only from approved access requests.
2. Reevaluate sessions when risk or context changes.
3. High-risk reevaluation requires reauthentication.
4. Failed identity or posture context revokes the session.
5. Session closure requires an actor and records audit.

## Deterministic Rules

The contract currently exposes at least 30 rules covering:

- tenant context;
- identity subject and display name;
- identity verification and suspension;
- federated identity provider evidence;
- device ownership, trust, posture, compliance, and attestation;
- resource naming, policy, segment, and microsegmentation;
- privileged MFA and approval;
- least-privilege scope;
- high-risk access review;
- explicit deny-by-default decision evidence;
- session approval, continuous verification, reauthentication, and closure;
- resource policy attachment;
- independent review and review notes;
- duplicate pending review blocking;
- access audit evidence;
- Bytewax for batch zero-trust mutation;
- cross-tenant access denial;
- zero-trust audit evidence for state changes.

Rule decisions are one of:

- `allow`
- `require_review`
- `deny`

`deny` takes precedence over `require_review`.

## Configuration

Required configuration sections:

- `tenant_id`
- `identities`
- `devices`
- `resources`
- `access`
- `sessions`
- `segmentation`
- `reviews`
- `security`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Key defaults:

- verified identity required;
- privileged MFA required;
- device posture required;
- minimum trust score `0.7`;
- attestation for sensitive resources;
- resource policy required;
- least-privilege default enabled;
- high-risk threshold `0.8`;
- deny-by-default posture;
- continuous session verification;
- independent review required;
- Bytewax event stream for batch mutations;
- tenant isolation required;
- audit required for access decisions and state changes.

## UI

Routes:

- `/ztna/dashboard`
- `/ztna/policies`
- `/ztna/identities`
- `/ztna/devices`
- `/ztna/resources`
- `/ztna/access`
- `/ztna/sessions`
- `/ztna/risk`
- `/ztna/reviews`
- `/ztna/audit`
- `/ztna/settings`

View models must remain dependency-light data payloads. They should contain
records, summary counts, required actions, theme component names, and route
metadata. Browser rendering belongs to generated applications.

## Theme

Theme name: `ztna_zero_trust_ops`.

Theme components:

- `access_decision`
- `identity_console`
- `device_posture`
- `resource_map`
- `session_monitor`
- `risk_console`
- `review_queue`
- `audit_timeline`

Generated UIs should use compact density, 8px card radius, status chips, and
clear risk/review/session state indicators.

## Adapter Boundaries

Adapter keys are declared in the capability contract:

- `authentication`: `auth`
- `security_framework`: `secu`
- `mfa_provider`: `mfau`
- `monitoring`: `moni`
- `audit_sink`: `audl`
- `identity_federation`: `idfd`
- `anomaly_detection`: `anom`
- `message_bus`: `mqeb`
- `cache`: `cach`
- `event_stream`: `bytewax`

Adapters must not be required for local package self-tests.

## Acceptance Criteria

- Contract exposes configuration, schema, deterministic rules, UI routes,
  theme, and adapters.
- Rule count is at least 30.
- UI route count is at least 10.
- Bytewax is the event-stream adapter.
- Service executes identity, device, resource, access, review, session, and
  audit lifecycles.
- Privileged access requires MFA and independent review or explicit JIT
  approval.
- High-risk access requires review before session start.
- Tenant-local IDs isolate repeated business keys across tenants.
- Cross-tenant access is denied.
- API helpers expose the lifecycle fields used by the service.
- View models expose all route families.
- `app.self_test()` passes.
- Focused package tests pass.
- Implementation audit and publish-plan pass.
