# Composition Configuration — User Guide

© 2025 Datacraft — Nyimbi Odero

---

## Introduction

The `composition_config` capability is the central configuration plane for all APG composed services. It manages the full lifecycle of configuration values from creation through deployment, with built-in versioning, schema validation, drift detection, and a growing set of async operations for advanced orchestration patterns.

This guide covers:
1. Core concepts and the config lifecycle
2. Namespace management
3. Creating, validating, and deploying configs
4. Version history and rollback
5. Async operations: bulk import, diff, scheduling
6. Config linting before deployment
7. Hierarchical config resolution
8. Compliance and audit chain verification
9. Templates and drift detection
10. AI agent integration
11. Streaming with Bytewax + NATS

---

## 1. Core Concepts

### Config Lifecycle

Every configuration value moves through a defined sequence of states:

```
draft → [scheduled] → validated → active → deployed
                                         ↘ rolled_back
                                         ↘ drifted
```

- **draft**: Config created; not yet deployable.
- **scheduled**: Config staged for future activation at `effective_at` time.
- **validated**: Validation evidence attached by a human or automated reviewer.
- **active**: Ready for deployment.
- **deployed**: Active in a target environment.
- **rolled_back**: A previous deployment was reverted.
- **drifted**: Observed version diverges from expected.

### Namespaces

Namespaces are the organisational unit. Each namespace belongs to a tenant, declares an environment, and has an accountable owner. Config key paths are prefixed with the namespace's `path_prefix`.

```
/payment/timeout_ms         → namespace: payment, key: timeout_ms
/payment.prod/timeout_ms    → namespace: payment.prod (production override)
/global/log_level           → namespace: global (inherited baseline)
```

### Tenant Isolation

All operations require a `tenant_id`. Records belonging to different tenants are never visible to each other. Cross-tenant operations (e.g. `config_diff`) are explicit and audited.

---

## 2. Setting Up the Service

```python
from capabilities.composition.config.service import CompositionConfigService

svc = CompositionConfigService()
```

No external dependencies are required for the in-memory service layer. For production, wire the service to the PostgreSQL backend via the `database/store.py` adapter.

---

## 3. Namespace Management

```python
# Register a namespace
ns = svc.register_namespace(
    namespace_key="payment",
    tenant_id="acme",
    name="Payment Service Config",
    environment="production",
    owner_id="payment-team",
    path_prefix="/payment",
    capability_id="composition_config",
)
```

Rules enforced:
- `owner_id` is mandatory — `namespace_requires_owner` will deny otherwise.
- `environment` is mandatory — must be one of `development`, `staging`, `production`, `dr`, `sandbox`.
- `path_prefix` must start with `/`.

---

## 4. Creating and Deploying Configs

### Create and activate

```python
# Create
svc.set_config(
    namespace="payment",
    key="timeout_ms",
    value=5000,
    tenant_id="acme",
    data_type="integer",
    description="HTTP timeout for payment gateway calls",
    owner_id="payment-team",
)

# Attach validation evidence
config_id = svc.get_config("payment", "timeout_ms", "acme")["id"]
svc.validate_configuration(config_id, actor_id="alice", evidence="reviewed in PR #42")

# Activate
svc.activate_configuration(config_id, actor_id="alice")

# Deploy to production (requires approval)
svc.deploy_configuration(
    deployment_key="payment-timeout-deploy-001",
    tenant_id="acme",
    configuration_id=config_id,
    environment="production",
    impact_level="medium",
    actor_id="alice",
    approved_by="bob",
    event_stream="bytewax",
)
```

### Secret configs

Secret values are stored as vault references — never as plaintext.

```python
svc.set_config(
    namespace="payment",
    key="stripe_api_key",
    value="vault:secret/payment/stripe_api_key",
    tenant_id="acme",
    data_type="secret_ref",
    secret=True,
    owner_id="security-team",
)
```

Reading a secret config returns `{"redacted": True, "secret_reference": "vault:..."}` — the actual value is never exposed through the service layer.

---

## 5. Version History and Rollback

### Inspect history

```python
history = svc.config_version_history("payment", "timeout_ms", "acme")
# Returns: version_count, current_version, versions (list of snapshots)
```

### Rollback to a previous version

```python
svc.rollback_config(
    namespace="payment",
    key="timeout_ms",
    version=2,
    tenant_id="acme",
    reason="performance regression observed after v3 deploy",
    rolled_back_by="alice",
)
```

A rollback creates a new version (current + 1) with the values restored from the target snapshot. The current state is always snapshotted before rollback so nothing is permanently lost.

---

## 6. Async Operations

All async methods are safe for use in `asyncio` contexts and can be composed with `asyncio.gather`.

### Parallel read fan-out

```python
import asyncio

results = await asyncio.gather(
    svc.async_get_config("payment", "timeout_ms", "acme"),
    svc.async_get_config("payment", "retry_count", "acme"),
    svc.async_get_config("fraud", "score_threshold", "acme"),
)
```

### Async bulk import

```python
result = await svc.async_bulk_import(
    namespace="payment",
    config_map={
        "timeout_ms": 5000,
        "retry_count": 3,
        "base_url": {"__value": "https://api.stripe.com", "__type": "string", "__description": "Stripe base URL"},
    },
    tenant_id="acme",
    owner_id="payment-team",
)
# result: {created_count, updated_count, failed_count, failures, success}
```

### Cross-tenant diff

```python
diff = await svc.async_config_diff("payment", tenant_a="acme", tenant_b="beta-corp")
# Returns: only_in_a, only_in_b, differing_keys, identical
```

---

## 7. Config Scheduling

Schedule a config to activate at a future UTC timestamp. A bytewax dataflow checks for pending scheduled configs and calls `activate_scheduled_configs` at the appropriate time.

```python
# Schedule a pricing change for midnight UTC
result = await svc.schedule_config_change(
    namespace="pricing",
    key="vat_rate",
    value=0.16,
    tenant_id="acme",
    effective_at="2026-07-01T00:00:00+00:00",
    owner_id="finance-team",
    reason="Kenya VAT rate update effective July 2026",
    data_type="float",
)
# result: {..., "status": "scheduled", "effective_at": "2026-07-01T00:00:00+00:00"}
```

### Activating scheduled configs (called by the scheduler)

```python
activation_result = await svc.activate_scheduled_configs(
    tenant_id="acme",
    reference_time="2026-07-01T00:01:00+00:00",
    actor_id="bytewax-scheduler",
)
# result: {activated_count, skipped_count, activated_ids, processed_at}
```

After activation the config status moves to `active`. It still requires `validate_configuration` + `activate_configuration` if the full lifecycle gate is enforced, or can be deployed directly if pre-approved at schedule time.

---

## 8. Config Linting

Run lint checks against a config value before deployment to catch dangerous settings early.

```python
findings = await svc.lint_config(
    namespace="app",
    key="log_level",
    tenant_id="acme",
    environment="production",
)
# findings: {finding_count, findings: [{rule, severity, message}], passed}
```

Example findings:

```json
{
  "finding_count": 1,
  "findings": [
    {
      "rule": "no_debug_in_production",
      "severity": "warning",
      "message": "log_level='DEBUG' is inappropriate for production environments"
    }
  ],
  "passed": true
}
```

`passed` is `True` when no `error`-severity findings are present. `warning` findings are surfaced for awareness but do not block deployment.

Built-in rules:

| Rule | Severity |
|------|----------|
| `no_debug_in_production` | warning |
| `positive_timeout_required` | error |
| `sensitive_key_must_be_secret` | error |
| `no_null_in_production` | error |

---

## 9. Hierarchical Config Resolution

Use `resolve_config` when a config key may be defined at multiple levels of specificity. The method walks the ancestor chain and returns the first match, with a `resolved_from` provenance field.

```python
# Tries payment.prod → payment → global in order
value = await svc.resolve_config(
    namespace="payment.prod",
    key="timeout_ms",
    tenant_id="acme",
    ancestor_namespaces=["payment", "global"],
)
# value: {..., "resolved_from": "payment", "search_path": ["payment.prod", "payment", "global"]}
```

This supports Spring Cloud Config / Consul KV style inheritance without requiring complex ORM joins.

---

## 10. Compliance and Audit

### Compliance posture check

```python
posture = await svc.compliance_check(tenant_id="acme")
# posture: {compliant, violation_count, violations: [{config_id, key_path, rule}]}
```

Checks performed:
- All `secret=True` configs have a `secret_reference` set.
- All `restricted=True` configs have a JSON schema registered.
- The audit event chain has no structural anomalies.

### Audit chain verification

```python
chain = await svc.verify_audit_chain(tenant_id="acme")
# chain: {event_count, chain_valid, broken_links, verified_at}
```

Broken links are reported with their position in the event sequence and the nature of the anomaly (duplicate ID, malformed ID).

### Export configs

```python
# JSON export
export = await svc.export_records(tenant_id="acme", format="json")

# Env-file export (for 12-factor apps)
env_export = await svc.export_records(tenant_id="acme", format="env")
# env_export["data"] = {"PAYMENT__TIMEOUT_MS": "5000", ...}
```

---

## 11. Templates

Templates let you define reusable config bundles with variable substitution schemas.

```python
template = svc.create_template(
    template_key="fastapi-defaults",
    tenant_id="acme",
    name="FastAPI Service Defaults",
    owner_id="platform-team",
    values={"log_level": "INFO", "workers": 4, "timeout_ms": 30000},
    variable_schema={
        "type": "object",
        "properties": {
            "log_level": {"type": "string"},
            "workers": {"type": "integer"},
            "timeout_ms": {"type": "integer"},
        },
    },
    shared=True,
    reviewed_by="platform-lead",
)
```

Shared templates require a `reviewed_by` actor. Private templates can be created without review.

---

## 12. Drift Detection

Record a drift event when the observed config version in a running service diverges from the expected version.

```python
svc.record_drift(
    tenant_id="acme",
    configuration_id=config_id,
    expected_version=5,
    observed_version=3,
    severity="warning",
    actor_id="drift-monitor",
)
```

Severities: `info`, `warning`, `critical`.

---

## 13. AI Agent Integration

Register an AI config agent with a declared runtime and role. All privileged actions require explicit human approval.

```python
agent = svc.register_config_agent(
    tenant_id="acme",
    name="config-drift-reviewer",
    runtime="claude_code",
    role="drift_reviewer",
    instructions="Review drift records, propose remediation steps, and flag critical drifts for human approval.",
)

# Gate a privileged action
svc.validate_agent_config_action(
    tenant_id="acme",
    agent_id=agent["id"],
    action="rollback_config",
    privileged_scope=True,
    human_approval_recorded=True,
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`

Supported roles: `config_architect`, `schema_reviewer`, `release_reviewer`, `drift_reviewer`, `security_reviewer`, `rollback_reviewer`

---

## 14. Streaming with Bytewax + NATS

All deployment and rollback operations route through Bytewax dataflows publishing to NATS JetStream subjects under `apg.composition.config.*`.

```python
# Batch change validation confirms NATS stream routing
result = svc.validate_batch_configuration_change(
    tenant_id="acme",
    change_count=42,
    event_stream="bytewax",
)
# result: {tenant_id, change_count, event_stream: "bytewax", stream: "apg.composition.config.lifecycle", processor: "bytewax"}
```

The bytewax dataflow subscribes to `apg.composition.config.lifecycle` and:
- Materialises the config change log to PostgreSQL
- Triggers downstream capability config hot-reload via `apg.composition.config.changed.<namespace>.<key>`
- Calls `activate_scheduled_configs` on a timer tick for scheduled changes

---

## 15. Health Check

```python
health = await svc.health_check(tenant_id="acme")
# {service, tenant_id, status: "healthy", namespace_count, configuration_count, deleted_count, audit_event_count, checked_at}
```

---

## Error Reference

| Exception | Cause |
|-----------|-------|
| `ValueError("namespace_and_key_required")` | Empty namespace or key passed to get/set/delete |
| `KeyError("config_not_found:ns/key")` | Key does not exist or belongs to a different tenant |
| `KeyError("config_deleted:ns/key")` | Key was soft-deleted |
| `KeyError("config_version_not_found:ns/key@vN")` | Requested version not in history |
| `PermissionError("rollback_reason_required")` | Rollback called without a reason string |
| `PermissionError("<rule_name>")` | Capability contract rule denied the operation |
| `ValueError("effective_at_required")` | schedule_config_change called without effective_at |

---

## File Reference

| File | Purpose |
|------|---------|
| `service.py` | All lifecycle operations and async methods |
| `models.py` | Dataclass records (no ORM) |
| `capability_contract.py` | Executable rule engine and constants |
| `api.py` | Flask-AppBuilder API helpers |
| `views.py` | UI model helpers |
| `app.py` | Package self-test |
| `database/store.py` | PostgreSQL backend adapter |
| `docs/user_guide.md` | This document |
| `WORLD_CLASS_IMPROVEMENTS.md` | 15 improvement paths toward production-grade capability |
