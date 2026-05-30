# IDFD Capability Plan

## Delivery Slice

Build IDFD as a coherent lifecycle and guardrail packet for generated APG
applications.

## Steps

1. Document the packet.
   - Add this plan.
   - Add a generated-app README.
   - Add `SPECIFICATION.md`.
   - Convert `cap_spec.md` into a pointer to the active specification.

2. Expand the executable contract.
   - Add provider, protocol, claims, sessions, SCIM, certificates, reviews,
     security, governance, observability, adapters, UI, and theme sections.
   - Expand deterministic rules to cover the real federation lifecycle.
   - Add Bytewax event-stream adapter evidence.
   - Add route and theme coverage for operational screens.

3. Harden the runtime.
   - Keep `IdfdService` dependency-light.
   - Key records by tenant and record ID.
   - Enforce provider, protocol, mapping, session, certificate, and
     cross-tenant guardrails.
   - Preserve compatibility helpers used by package probes.

4. Refresh generated app surfaces.
   - Make `app.py` derive semantic output from the live contract.
   - Update registration metadata, endpoint metadata, adapters, permissions,
     and view helpers.
   - Refresh JSON package evidence from the live contract.

5. Review and verify.
   - Run py_compile on edited IDFD files.
   - Run focused IDFD tests only.
   - Run implementation audit and publish-plan for IDFD.
   - Scan primary packet files for stale markers.
   - Run `git diff --check`.

## Battery-Conscious Non-Goals

- Full repository tests.
- Live SAML/OIDC/LDAP/SCIM provider handshakes.
- Browser-rendered UI.
- Database migrations.
- Live Bytewax execution.
- External AUTH, MFAU, ENCR, AUDL, SECU, KEYM, MONI, or CACH adapters.
- Security, interoperability, load, and conformance certification benchmarks.
