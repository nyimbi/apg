# CACH - Cache Management

CACH is APG's cache governance and runtime-adapter capability. It gives
generated applications a tenant-aware way to register cache namespaces, enforce
entry admission rules, manage warming and eviction reviews, publish UI metadata,
compose first-class cache agents, validate Bytewax lifecycle batches, and
connect to cache backends without hard-coding a specific store.

The current packet is intentionally dependency-light. It can publish a capability
contract, evaluate guardrails, build view models, and run focused tests without
Redis, Flask-AppBuilder, AI services, or optional compression packages.

## What CACH Provides

- Tenant-scoped namespace registration and policy defaults.
- Deterministic cache guardrails for writes, reads, warming, eviction, tenant
  isolation, encryption, TTL limits, and memory pressure.
- Entry lifecycle records for admitted, denied, expired, invalidated, and
  refresh-required cache entries.
- Warming plans with source evidence, batch limits, and review state.
- Eviction and capacity reviews with independent reviewer evidence.
- First-class cache-agent registration for Codex, Claude Code, opencode, and Pi
  with owner, purpose, scope, contribution disclosure, and human approval
  guardrails.
- Bytewax lifecycle-batch validation for cache policy, warming, agent, and
  eviction mutations.
- Compact generated-application view models for operations UIs.
- Theme tokens and component metadata for cache dashboards and control surfaces.
- Contract-derived semantic-model and release evidence for APG publish tooling.
- Adapter boundaries for memory, Redis-compatible stores, edge caches, CDNs,
  application-local query caches, and future APG integrations.

## Key Files

- `SPECIFICATION.md` - full functional and guardrail specification.
- `PLAN.md` - implementation plan and deferred runtime work.
- `capability_contract.py` - configuration, rule engine, UI routes, and theme.
- `service.py` - existing async cache runtime plus dependency-light governance
  records and lifecycle service.
- `api.py` - FastAPI routes plus direct helper functions for generated apps.
- `view_models.py` - generated-application UI model builders.
- `app.py` - APG package entrypoint and semantic model.
- `semantic_model.json` - publishable semantic-model evidence.
- `release_report.json` - focused release evidence.
- `tests/` - focused package and contract coverage.

## Runtime Shape

CACH has two layers:

1. **Control plane**: namespace policies, rule decisions, lifecycle records,
   review queues, audit events, UI metadata, and semantic evidence.
2. **Data plane adapters**: memory, Redis/Valkey, edge/CDN caches, query caches,
   AI optimizers, monitoring, audit, and security integrations.

Generated APG applications should call the control plane before touching a cache
backend. Cache adapters should honor the decisions returned by CACH.

## Direct Usage

```python
from capabilities.common.cach.api import (
    create_namespace_record,
    write_cache_entry_record,
    read_cache_entry_record,
    request_eviction_review,
    decide_eviction_review,
    register_cache_agent,
    validate_cache_lifecycle_batch,
)

namespace = create_namespace_record(
    tenant_id="tenant-a",
    namespace="orders",
    owner="platform",
    data_classification="regulated",
    max_ttl_seconds=900,
    encryption_required=True,
)

entry = write_cache_entry_record(
    tenant_id="tenant-a",
    namespace="orders",
    key="order:1001",
    value_ref="redis://orders/order:1001",
    producer="order-service",
    ttl_seconds=300,
    encrypted=True,
    data_classification="regulated",
)

read_result = read_cache_entry_record(
    tenant_id="tenant-a",
    namespace="orders",
    key="order:1001",
)

review = request_eviction_review(
    tenant_id="tenant-a",
    namespace="orders",
    requester="platform",
    memory_utilization_percent=94,
    proposed_action="evict cold entries from distributed tier",
    reason="tenant quota pressure",
)

decision = decide_eviction_review(
    review_id=review.review_id,
    reviewer="cache-sre",
    decision="approved",
    notes="Cold entries can be evicted; source of truth is healthy.",
)

agent = register_cache_agent(
    tenant_id="tenant-a",
    agent_id="warming-agent",
    name="Warming Agent",
    runtime="claude-code",
    role="warming-reviewer",
    scope="warming plan review",
    owner="platform",
    purpose="review warming plans",
    contribution_disclosed=True,
    human_approval_required=True,
)

batch = validate_cache_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=3,
)
```

## Rule Evaluation

```python
from capabilities.common.cach.capability_contract import evaluate_capability_rules

decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "write",
    "namespace_present": True,
    "namespace_status": "active",
    "data_classification": "sensitive",
    "entry_encrypted": False,
})

assert decision["decision"] == "deny"
assert "sensitive_entry_requires_encryption" in decision["matched_rules"]
```

## View Models

Generated applications can use `view_models.py` to render CACH without knowing
the internal service representation:

```python
from capabilities.common.cach.api import SERVICE
from capabilities.common.cach.view_models import dashboard_model, warming_console_model

dashboard = dashboard_model(SERVICE)
warming = warming_console_model(SERVICE, tenant_id="tenant-a")
```

## Permissions

The capability registration exposes these permissions:

- `cach:view`
- `cach:read`
- `cach:write`
- `cach:delete`
- `cach:manage_namespaces`
- `cach:manage_policies`
- `cach:warm`
- `cach:review_eviction`
- `cach:view_analytics`
- `cach:manage_agents`
- `cach:admin`

## Adapter Boundary

CACH does not require a specific backend. A production adapter should:

1. Ask CACH for a namespace and rule decision.
2. Store data in the selected backend only when the decision allows it.
3. Emit audit and telemetry events through APG adapters when available.
4. Preserve tenant isolation in backend key construction.
5. Respect encryption, TTL, invalidation, freshness, and eviction decisions.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/cach/capability_contract.py \
  capabilities/common/cach/service.py \
  capabilities/common/cach/api.py \
  capabilities/common/cach/view_models.py \
  capabilities/common/cach/app.py

./.venv/bin/pytest -q \
  capabilities/common/cach/tests/test_capability_contract.py \
  capabilities/common/cach/tests/test_package_contract.py

./.venv/bin/apg capabilities publish-plan capabilities/common/cach --json
```

Full repository tests, live cache backends, production persistence, APG
auth/audit/monitoring adapters, and dashboard browser verification are separate
runtime validation tasks.
