# Identity Federation (IDFD)

IDFD is APG's generated-application capability for tenant-scoped identity
federation. It gives composed applications a deterministic, dependency-light
surface for SAML, OIDC, LDAP, SCIM, claim mapping, federated sessions,
certificate rotation, operational health, first-class federation governance
agents, Bytewax lifecycle batch validation, audit, and governance review.

The package is designed for rapid application composition. A generated APG app
can import the contract, register the capability, call the service/runtime
helpers, render route/view-model payloads, and publish package evidence without
requiring Flask, Flask-AppBuilder, a database session, a live identity provider,
or a model server.

## What The Capability Provides

- Provider lifecycle for SAML, OIDC, LDAP, and SCIM federation providers.
- Protocol guardrails for SAML assertion encryption, signed SAML responses,
  OIDC redirect allowlists, OIDC PKCE, LDAP TLS, and SCIM external IDs.
- Claim mapping governance with source/target claim requirements and review
  gates for sensitive mappings.
- Federated session issuance and revocation with MFA, risk, duration, and
  tenant-isolation controls.
- Certificate registration and rotation evidence for federation signing keys.
- Health reporting for stale metadata, active sessions, and expiring
  certificates.
- Provider-neutral federation governance agents for Codex, Claude Code,
  opencode, Pi, and future runtimes through adapter contracts.
- Bytewax-first lifecycle batch validation for provider, protocol, claim,
  session, certificate, SCIM, review, and agent changes.
- Deterministic rule engine, UI route manifest, visual theme tokens, Bytewax
  streaming adapter evidence, and package metadata.

## Generated-App Usage

```python
from capabilities.common.idfd.service import IdfdService, expires_in_days

service = IdfdService()
tenant_id = "tenant-sso"

provider = service.register_provider(
    provider_id="corp-oidc",
    tenant_id=tenant_id,
    name="Corporate OIDC",
    protocol="oidc",
    owner_id="identity",
    signing_key_id="key-1",
    metadata_url="https://idp.example.test/.well-known/openid-configuration",
    redirect_allowlist=["https://app.example.test/callback"],
)

service.add_claim_mapping("map-email", tenant_id, provider["id"], "mail", "email")
session = service.issue_session("session-1", tenant_id, provider["id"], "user-1")
service.register_certificate("cert-1", tenant_id, provider["id"], "key-1", expires_in_days(30))
summary = service.dashboard_summary(tenant_id)
```

## Agent Composition And Lifecycle Batches

IDFD treats AI agents as governed composition records. A generated app can
register an agent that reviews federation evidence while the actual runtime
client remains behind an AICR adapter.

```python
agent = service.register_federation_agent(
    agent_id="agent-federation-review",
    tenant_id=tenant_id,
    name="Federation Review Agent",
    runtime="codex",
    role="provider_reviewer",
    scope="provider metadata and claim mappings",
    owner="identity-governance",
    purpose="review federation rollout evidence",
)

batch = service.validate_idfd_lifecycle_batch(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
    operation="federation_agent_batch",
)

assert agent["status"] == "active"
assert batch["status"] == "accepted"
```

Privileged roles such as `session_risk_reviewer`,
`certificate_rotation_reviewer`, `scim_reviewer`, `privacy_reviewer`,
`lifecycle_batch_reviewer`, and `federation_steward` are marked
`pending_review` unless human approval evidence is recorded. Non-Bytewax
lifecycle batches are intentionally denied by the rule engine.

## Composition Contract

Use `get_capability_contract()` when a compiler, generator, or larger APG
application needs to inspect the capability.

```python
from capabilities.common.idfd.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-sso")
routes = contract["ui"]["routes"]
rules = contract["rule_engine"]["rules"]
adapters = contract["configuration"]["adapters"]
```

Important adapter evidence:

- `generated_app_runtime`: `service.IdfdService`
- `event_stream`: `bytewax`
- `authentication`: `auth`
- `mfa_provider`: `mfau`
- `encryption`: `encr`
- `audit_sink`: `audl`
- `key_management`: `keym`
- `agent_adapter`: `aicr_provider_neutral_identity_federation_agent_adapter`

## Screens

The contract exposes route metadata for:

- dashboard
- providers
- protocols
- mappings
- sessions
- certificates
- SCIM directory
- risk console
- reviews
- agents
- lifecycle
- audit
- settings

The view helpers in `views.py` return dependency-light payloads for those
screens and include the theme component names required by generated UIs.

## Guardrails

IDFD includes deterministic rules for tenant context, provider ownership,
signing keys, metadata review, SAML encryption, OIDC redirect allowlists, PKCE,
LDAP TLS, SCIM deprovisioning, claim mapping review, privileged MFA, high-risk
reauthentication, session duration, certificate rotation review, federation
agent registration, Bytewax lifecycle batch validation, tenant isolation, and
required audit evidence.

Rules are executable through:

```python
from capabilities.common.idfd.capability_contract import evaluate_capability_rules

result = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "batch_federation_mutation",
    "event_stream": "legacy_queue",
})
assert result["decision"] == "deny"
```

## Verification

Focused package checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/idfd/__init__.py capabilities/common/idfd/capability_contract.py capabilities/common/idfd/federation_runtime.py capabilities/common/idfd/models.py capabilities/common/idfd/service.py capabilities/common/idfd/api.py capabilities/common/idfd/views.py capabilities/common/idfd/app.py capabilities/common/idfd/test_capability_contract.py capabilities/common/idfd/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/idfd/test_capability_contract.py capabilities/common/idfd/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/idfd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/idfd --json
```
