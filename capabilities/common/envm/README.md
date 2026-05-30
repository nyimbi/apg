# ENVM Environment Management

`envm` is the APG common environment management capability. It lets generated
applications compose tenant-scoped environment inventory, stage and region
policy, governed promotion paths, promotion runs, configuration drift reports,
secret scopes, audit evidence, Bytewax stream governance, visual theme
metadata, and AI-agent assistance.

The package is dependency-light. It defines the executable service, rule
engine, UI route metadata, theme metadata, Bytewax stream declaration, API
helpers, view models, and semantic evidence. Deployment providers, live
configuration stores, secret vaults, runtime access checks, monitoring
pipelines, and stream-worker deployments are adapter responsibilities.

## What It Provides

- Environment inventory with owner, stage, region, configuration source, RBAC
  policy, and secret-scope policy.
- Production-change approval guardrails and production locking metadata.
- Promotion paths with source, target, deployment link, rollback environment,
  approval state, and promotion runs.
- Configuration drift reports with declared and observed versions, drift
  percentage, review state, and remediation action.
- Secret scopes with policy references, secret references, and access roles.
- AI ENVM-agent registration for Codex, Claude Code, OpenCode, Pi, and future
  runtimes behind the same contract.
- Bytewax stream guardrail for batch environment mutation.
- UI routes and visual theme tokens for generated APG applications.

## Quick Use

```python
from capabilities.common.envm import EnvmService

service = EnvmService()

service.register_environment(
    environment_id="env-dev",
    tenant_id="tenant-acme",
    name="Development",
    stage="development",
    region="ke-nairobi",
    owner="platform",
    configuration_source="git://config/env-dev",
    rbac_policy="rbac-dev",
    secret_scope_policy="secret-dev",
)

service.register_environment(
    environment_id="env-prod",
    tenant_id="tenant-acme",
    name="Production",
    stage="production",
    region="ke-nairobi",
    owner="operations",
    configuration_source="git://config/env-prod",
    rbac_policy="rbac-prod",
    secret_scope_policy="secret-prod",
    approval_recorded=True,
)
```

## AI Agent Registration

AI agents are first-class environment contributors only after registration:

```python
agent = service.register_envm_agent(
    tenant_id="tenant-acme",
    name="Drift reviewer",
    runtime="codex",
    role="drift_reviewer",
    scope="review drift reports and recommend remediation actions",
    contribution_disclosed=True,
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `environment_reviewer`, `promotion_reviewer`, `drift_reviewer`,
`secret_scope_reviewer`, and `policy_reviewer`.

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- environment owner, region policy, configuration source, or RBAC policy is
  missing;
- production change lacks approval evidence;
- promotion lacks a declared path or artifact reference;
- secret scope lacks policy, secret references, or access roles;
- drift exceeds the threshold without review;
- an AI ENVM agent is unregistered, unsupported, unscoped, or undisclosed;
- lifecycle state changes lack audit evidence;
- batch environment mutation does not use Bytewax.

## Bytewax Batch Mutation

Batch environment mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_environment_mutation("bytewax")
blocked = service.validate_batch_environment_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.envm.lifecycle` and state for environments,
promotion paths, promotion runs, drift reports, secret scopes, ENVM agents, and
audit events.

## Composition

Generated APG applications should compose `envm` through:

- capability ID: `envm`;
- provided services: environment inventory, promotion, configuration drift,
  secret scopes, environment policy, and ENVM agents;
- required services: `auth`, `conf`, `audl`, `depl`, `keym`, and `moni`;
- API prefix: `/envm/api/v1`;
- UI routes: dashboard, environments, promotion, drift, secrets, agents,
  policies, rules, analytics, audit, and settings;
- theme: `envm_environment_ops`;
- stream processor: `bytewax`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/envm/__init__.py capabilities/common/envm/capability_contract.py capabilities/common/envm/models.py capabilities/common/envm/service.py capabilities/common/envm/api.py capabilities/common/envm/views.py capabilities/common/envm/app.py capabilities/common/envm/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/envm/test_capability_contract.py
./.venv/bin/python -c "from capabilities.common.envm import EnvmService; service = EnvmService(); service.register_envm_agent('tenant-proof', 'Proof agent', 'codex', 'drift_reviewer', 'review drift'); print(service.dashboard_summary('tenant-proof'))"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/envm --json
./.venv/bin/apg capabilities publish-plan capabilities/common/envm --json
```
