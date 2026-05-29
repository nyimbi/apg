# AUDL Capability Development Plan

## Current State

AUDL has a rich production-oriented async service, FastAPI and Flask-AppBuilder
surfaces, a capability contract, package evidence, and contract tests. The
package-level composition gap is a dependency-light lifecycle facade that can
execute audit governance without starting external storage, web servers, ML
providers, or compliance systems.

## Packet 1: Governed Audit Evidence Lifecycle

Deliver a focused lifecycle packet:

- add package-level lifecycle, legal hold, export, purge, investigation, and
  governance-event records;
- add a dependency-light `AudlService` runtime facade;
- append tenant-scoped audit events with checksum verification;
- open and release legal holds with human evidence;
- request, approve, reject, and list regulated exports;
- request and decide purge operations with dual-control evidence;
- open and close investigations with owner and resolution evidence;
- expose API-helper and view-model surfaces for events, timeline, holds,
  exports, purges, investigations, compliance, reporting, rules, and audit
  evidence;
- replace stale generated-package test naming with package contract tests;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `AuditLifecycleEvent`, `AuditLegalHoldRecord`,
   `AuditExportRequest`, `AuditPurgeRequest`, `AuditInvestigationRecord`, and
   `AuditGovernanceEvent`.
2. Add `audit_runtime.py` with a dependency-light `AudlService` facade.
3. Add `api_helpers.py` for generated APG applications that need simple AUDL
   lifecycle calls without importing the production FastAPI app.
4. Add `view_models.py` for dependency-light AUDL UI model surfaces.
5. Update capability metadata with legal hold, export review, purge review,
   and investigation governance surfaces.
6. Extend package contract tests with positive append-hold-export-purge-
   investigation coverage and negative checksum, PII masking, legal hold,
   missing reviewer, missing dual-control, tenant-mismatch, and duplicate-ID
   coverage.
7. Rename generated-package tests to package contract naming.
8. Update `cap_spec.md` with the current executable lifecycle and proof
   commands.
9. Run focused package proof, implementation audit, publish-plan, and diff
   checks.

## Review Checklist

- Event, legal hold, export, purge, investigation, and governance state is
  tenant-qualified.
- Immutable event append verifies checksum when a checksum is supplied.
- PII-bearing exports require masking.
- Export decisions require reviewer identity and notes.
- Legal hold blocks purge.
- Purge decisions require dual-control reviewer evidence.
- Investigation closure requires actor, resolution, and evidence.
- API helpers expose the same behavior as service methods.
- View models expose event, timeline, hold, export, purge, investigation,
  compliance, reporting, rule, theme, and governance-event state.
- Storage, stream processing, SIEM, GRC, DLP, ML, and web-server systems remain
  adapter boundaries.
