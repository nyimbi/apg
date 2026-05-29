# Identity Federation Capability Specification

- **Capability Name**: Identity Federation
- **Capability ID**: `idfd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

IDFD gives generated APG applications an executable identity-federation control
plane. It manages tenant-scoped federation providers, SAML/OIDC/LDAP/SCIM
protocol guardrails, claim mappings, federated sessions, certificate records,
audit events, health reports, UI metadata, and theme metadata without requiring
live identity-provider infrastructure during local APG generation.

The package keeps live SAML, OIDC, LDAP, SCIM, MFA, certificate-store, audit,
and session backends behind adapter boundaries. The in-process runtime is the
deterministic baseline that APG examples, generated applications, and package
publish plans can execute today.

## Provided Services

- `federated_sso`
- `saml_identity_provider`
- `oidc_broker`
- `identity_mapping`
- `certificate_rotation`
- `federation_health`
- `federation_audit`

## Required Services

- `tenant_context`
- `auth`
- `mfau`
- `encr`
- Optional: `audl`, `secu`, `mten`, `ztna`

## Runtime Surfaces

- `models.py` defines provider, claim-mapping, session, certificate, audit, and
  health-report records.
- `federation_runtime.py` provides metadata freshness, session expiry, and
  health-summary helpers.
- `service.py` provides tenant-aware provider registration, metadata refresh,
  claim mapping, session issue/revoke, certificate registration, health
  reporting, compatibility record creation, dashboard summaries, and rule
  enforcement.
- `api.py` exposes dependency-light helpers that mirror generated APG endpoint
  handlers.
- `views.py` exposes dashboard, provider console, protocol workbench, claim
  mapping table, session monitor, certificate center, and audit view models.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The default configuration enables SAML, OIDC,
LDAP, and SCIM; requires provider signing keys; requires encrypted SAML
assertions; requires OIDC redirect allowlists; requires MFA for privileged
federated sessions; audits federation events; and exposes provider, protocol,
session, and certificate UI consoles.

## Rules And Guardrails

IDFD evaluates deterministic contract rules and service-level guardrails:

- `tenant_context_required`
- `provider_requires_signing_key`
- `saml_assertion_requires_encryption`
- `oidc_client_requires_redirect_allowlist`
- `privileged_federation_requires_mfa`
- `stale_metadata_requires_refresh`
- `claim_mapping_review_required`
- `provider_not_active`
- `provider_missing`
- `session_missing`

Service methods raise explicit errors when a guardrail blocks an operation, so
capability users can test negative cases without external federation providers.

## UI And Theme

The package exposes eight APG Python route contracts:

- `/idfd/dashboard`
- `/idfd/providers`
- `/idfd/protocols`
- `/idfd/mappings`
- `/idfd/sessions`
- `/idfd/certificates`
- `/idfd/audit`
- `/idfd/settings`

The theme contract is `idfd_federation_console` with compact provider grids,
protocol panels, mapping tables, and certificate timelines.

## Adapter Boundaries

The current package intentionally does not open network connections, store live
private keys, mint production identity tokens, or call real MFA providers. Those
concerns belong behind explicit adapters that can be verified independently:

- SAML metadata and assertion adapters.
- OIDC client-registration and token adapters.
- LDAP/SCIM directory adapters.
- MFA and privileged-session adapters.
- Certificate store and key-rotation adapters.
- Audit sink adapters.

## Focused Verification

Use focused verification for IDFD changes:

```bash
./.venv/bin/python -m py_compile capabilities/common/idfd/__init__.py capabilities/common/idfd/models.py capabilities/common/idfd/federation_runtime.py capabilities/common/idfd/service.py capabilities/common/idfd/api.py capabilities/common/idfd/views.py capabilities/common/idfd/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/idfd/test_capability_contract.py capabilities/common/idfd/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/idfd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/idfd --json
```
