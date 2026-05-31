# DLPD Capability Plan

## Delivery Slice

Extend DLPD from its existing policy/classifier/inspection packet into a
coherent AI-agent and Bytewax lifecycle guardrail packet for generated APG
applications.

## Steps

1. Document the packet.
   - Add generated-app README.
   - Add `SPECIFICATION.md`.
   - Add this plan.
   - Convert `cap_spec.md` into a pointer to the active specification.

2. Expand the executable contract.
   - Add policy, classification, quarantine, incident, review, security,
     observability, adapter, UI, and theme sections.
   - Expand deterministic guardrails to the full DLP lifecycle.
   - Add provider-neutral DLP agent manifest for `codex`, `claude_code`,
     `opencode`, and `pi`.
   - Add Bytewax lifecycle stream manifest and DLP lifecycle batch operations.
   - Add Bytewax event-stream adapter evidence.

3. Harden the runtime.
   - Keep `DlpdService` dependency-light.
   - Key state by tenant and record ID.
   - Enforce policy, classifier, inspection, quarantine, incident, and
     cross-tenant guardrails.
   - Register DLP agents with scope, owner, purpose, contribution disclosure,
     and privileged-role review semantics.
   - Validate lifecycle batches through Bytewax-only stream metadata.
   - Preserve hash-only generated runtime behavior for sensitive content.

4. Refresh generated app surfaces.
   - Make `app.py` derive semantic output from the live contract.
   - Update registration metadata, endpoint metadata, adapters, permissions,
     and view helpers.
   - Add agent roster and lifecycle batch view/API surfaces.
   - Refresh JSON package evidence from the live contract.

5. Review and verify.
   - Run py_compile on edited DLPD files.
   - Run focused DLPD tests only.
   - Run implementation audit and publish-plan for DLPD.
   - Scan primary packet files for stale markers.
   - Run `git diff --check`.

## Battery-Conscious Non-Goals

- Full repository tests.
- Live network/email/file/object-store interception.
- Browser-rendered UI.
- Database migrations.
- Live Bytewax execution.
- External SECU, ENCR, NLPC, ANOM, AUDL, MQEB, SRCH, COMP, MONI, or CACH
  adapters.
- Security certification, DLP appliance interoperability, load, latency, and
  throughput benchmarks.
