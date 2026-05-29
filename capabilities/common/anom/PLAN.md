# ANOM Capability Development Plan

## Current State

ANOM already has a deterministic anomaly engine, package contract, service,
API helper facade, view models, and focused tests. The next packet should make
the lifecycle more operational by strengthening tenant isolation and requiring
closure evidence for investigations.

## Packet 1: Governed Investigation Closure

Deliver a focused lifecycle packet:

- add tenant-scoped anomaly audit events;
- make in-memory source, baseline, observation, signal, investigation, and
  feedback stores tenant-qualified;
- detect signals and open investigations without allowing duplicate IDs across
  tenants to collide;
- require tenant, actor, resolution, and evidence when closing investigations;
- expose audit events and closure fields through API helpers and view models;
- replace stale generated-package test naming with package contract tests;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `AnomalyAuditEvent` and closure evidence fields on
   `Investigation`.
2. Update `service.py` to use tenant-qualified keys, emit audit events, and
   enforce investigation closure evidence.
3. Update `api.py` with closure and audit helper functions.
4. Update `views.py` so dashboard and investigation views expose audit evidence.
5. Extend package contract tests with positive detect-investigate-close
   coverage and negative tenant, missing owner, missing resolution evidence,
   feedback review, baseline reset, and duplicate-ID isolation coverage.
6. Rename generated-package tests to package contract naming.
7. Update `cap_spec.md` with the current executable lifecycle and proof
   commands.
8. Run focused package proof, implementation audit, publish-plan, and diff
   checks.

## Review Checklist

- Monitoring source and baseline lookup is tenant-qualified.
- Signal, investigation, and feedback state is tenant-qualified.
- Critical anomalies cannot be accepted without an owner.
- Investigations cannot close without resolution evidence.
- Tenant mismatches are blocked.
- API helpers expose the same behavior as service methods.
- View models expose source, baseline, signal, investigation, feedback, rule,
  theme, and audit-event state.
- Monitoring, incident, workflow, and storage integrations remain adapter
  boundaries.
