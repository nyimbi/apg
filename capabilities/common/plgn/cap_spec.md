# Plugin/Extension Framework Capability Specification

- **Capability Name**: Plugin/Extension Framework
- **Capability ID**: `plgn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`plgn` provides executable plugin and extension governance for APG. It owns
tenant plugin manifests, publisher and signature posture, dependency validation,
permission reviews, sandbox policies, curated marketplace listings, release
records, installation and enablement state, audit events, UI view models, and
deterministic rule enforcement behind the capability contract.

The package is intentionally dependency-light. It does not call live package
registries, malware scanners, signing services, secret stores, remote sandboxes,
or marketplace payment systems directly. Those integrations should be composed
through adapters around `auth`, `secu`, `conf`, `regy`, `agnt`, `sbox`, and
`wflo` when production behavior is verified.

## Provided Services

- `plugin_registry`
- `extension_marketplace`
- `permission_review`
- `sandbox_policy`
- `plugin_release_lifecycle`
- `plgn_operations`

## Required Services

- `tenant_context`
- `auth`
- `secu`
- `conf`

Optional composition partners include `regy`, `agnt`, `sbox`, and `wflo`.

## Runtime Behavior

The service layer exposes an in-memory plugin-governance runtime for package
evidence and generated-application composition:

1. Register tenant-scoped plugin manifests with owners, publishers, versions,
   release channels, requested permissions, dependencies, external-plugin
   posture, signatures, manifest validation, dependency validation, and supply
   chain scan evidence.
2. Record permission reviews for requested scopes, denied scopes, sensitive
   scopes, and secret-access posture.
3. Attach sandbox policies covering network, filesystem, secret access, and
   tool allowlists.
4. Publish curated marketplace listings only when publisher verification and
   tenant install policy pass.
5. Create signed releases only when manifests, permission reviews, sandbox
   policy, and listing readiness pass.
6. Install plugins through tenant install policy and enable them only when
   signature and sandbox guardrails pass.
7. Publish dashboard, marketplace, registry, permission review, sandbox,
   release manager, governance, and audit state.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context, marketplace curation, publisher
verification, release channel policy, plugin ownership, manifest schema,
signatures, dependency validation, permission review, sandbox policy, secret
access, supply chain scan, external review, configuration policy, UI, and theme
metadata are explicit contract concerns.

## Rules

- `tenant_context_required`
- `plugin_requires_owner`
- `plugin_requires_signature`
- `permissions_require_review`
- `plugin_requires_sandbox`
- `external_plugin_requires_review`

The service calls the deterministic rule engine before tenant-sensitive writes,
plugin registration, and plugin enablement. Deny decisions raise
`PermissionError`; review decisions block registration until the required review
evidence is recorded.

## UI

The package exposes 8 APG Python UI route contract(s) through `views.py` and the
package semantic model:

- `/plgn/dashboard`
- `/plgn/marketplace`
- `/plgn/plugins`
- `/plgn/manifests`
- `/plgn/permissions`
- `/plgn/sandbox`
- `/plgn/releases`
- `/plgn/settings`

`views.py` provides dashboard, marketplace, plugin registry, permission review,
sandbox policy, release manager, and governance view models.

## Theme

The package uses the `plgn_extension_marketplace` APG theme contract.

## Runtime Surfaces

| File | Responsibility |
| --- | --- |
| `models.py` | Plugin manifests, permission reviews, sandbox policies, marketplace listings, releases, installations, and audit events. |
| `plugin_runtime.py` | Deterministic IDs, release channels, install policy, scope normalization, manifest readiness, and release-readiness helpers. |
| `service.py` | Tenant-scoped lifecycle behavior and APG rule enforcement. |
| `api.py` | Dependency-light API helper functions over `PlgnService`. |
| `views.py` | Route metadata and UI view models. |
| `test_capability_contract.py` | Contract, lifecycle, and guardrail tests. |

## Adapter Boundaries

Production integration should remain behind explicit adapters until directly
verified:

- `auth` for publisher, reviewer, installer, and administrator identity.
- `secu` for permission review, supply chain scanning, malware checks, and
  sensitive-scope policy.
- `conf` for tenant install policy and extension configuration baselines.
- `regy` for service and plugin discovery publication.
- `agnt` for agent-extension tool registration and risk controls.
- `sbox` for remote execution sandbox enforcement.
- `wflo` for review and release approval workflows.
- external signing, registry, billing, and marketplace systems.

## Focused Verification

```bash
./.venv/bin/pytest -q capabilities/common/plgn/test_capability_contract.py capabilities/common/plgn/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/plgn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/plgn --json
```
