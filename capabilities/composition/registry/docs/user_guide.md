# Composition Registry — User Guide

Complete operational guide for the APG Capability Registry (`composition_registry`).

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Registering Capabilities](#registering-capabilities)
5. [Managing Dependencies](#managing-dependencies)
6. [Composition Blueprints](#composition-blueprints)
7. [Version Governance](#version-governance)
8. [Marketplace Publication](#marketplace-publication)
9. [Discovery and Search](#discovery-and-search)
10. [Health and Observability](#health-and-observability)
11. [Governance and Compliance](#governance-and-compliance)
12. [World-Class Features (v2)](#world-class-features-v2)
13. [Registry Agents](#registry-agents)
14. [Analytics](#analytics)
15. [Reference — All Methods](#reference--all-methods)

---

## Overview

The Composition Registry is the single authoritative catalog for every APG capability. It
enforces tenant isolation, dependency cycle detection, composition validation, version
compatibility governance, and marketplace publication review — all exposed as a pure-Python
service with no external database dependency in testing.

**Key design decisions:**

- All operations are tenant-scoped. Pass a consistent `tenant_id` throughout.
- Write operations require a policy context; the rule engine in `capability_contract.py` enforces
  this automatically.
- Dependency cycles are detected at edge-add time (not at composition time) using DFS.
- The audit log is append-only and can be replayed to reconstruct historical state.

---

## Quick Start

```python
import asyncio
from capabilities.composition.registry.service import CompositionRegistryService

svc = CompositionRegistryService()
TENANT = "acme"

# Register a capability
svc.register_capability(
    capability_id="billing",
    tenant_id=TENANT,
    name="billing",
    display_name="Billing Service",
    owner="finance-team",
    category="fintech",
    version="1.2.0",
    provides=["invoice_generation", "payment_processing"],
    contract_ref="capabilities/fintech/billing/capability_contract.py",
    manifest_path="capabilities/fintech/billing",
)

# View the dashboard
print(svc.dashboard_summary(TENANT))

# Search
results = svc.capability_search(TENANT, "billing")

# Run async features
async def main():
    score = await svc.score_capability(TENANT, "billing")
    print(f"Quality grade: {score['grade']} ({score['score_pct']}%)")

asyncio.run(main())
```

---

## Core Concepts

### Capability

A capability is a discrete unit of business logic with a declared interface (`provides`) and
declared dependencies (`requires`). Every capability must have:

- `capability_id` — stable, unique identifier (slug format, e.g. `fintech_billing`)
- `owner` — accountable team or person
- `category` — domain grouping (e.g. `fintech`, `intel`, `composition`)
- `version` — semver string
- `provides` — list of service surfaces this capability exposes
- `contract_ref` — path to the executable `capability_contract.py`

### Tenant

All data is partitioned by `tenant_id`. Cross-tenant access is not possible without explicit
sharing grants (see [Multi-Tenant Sharing](#multi-tenant-sharing)).

### Lifecycle States

```
discovered → registered → validated → active → deprecated → retired
```

Write operations that change lifecycle state emit events to the Bytewax stream
`apg.composition.registry.lifecycle`.

---

## Registering Capabilities

### Manual Registration

```python
svc.register_capability(
    capability_id="ntfy",
    tenant_id="acme",
    name="ntfy",
    display_name="Notification Service",
    owner="platform-team",
    category="platform",
    version="2.0.0",
    provides=["push_notification", "email_notification"],
    requires=["auth"],
    contract_ref="capabilities/platform/ntfy/capability_contract.py",
    manifest_path="capabilities/platform/ntfy",
)
```

### Filesystem Auto-Discovery

Auto-register all capabilities found under a root directory by scanning for `__init__.py`
files that declare `__capability_code__`, `__capability_name__`, and related metadata fields.

```python
result = await svc.auto_discover_capabilities(
    tenant_id="acme",
    root_path="capabilities/",
    excluded_paths=["__pycache__", ".venv"],
)
print(f"Discovered {result['discovered']}, registered {result['registered']}")
```

The method skips already-registered capabilities and returns a detailed error list for any
files that fail to parse.

### Unregistering

```python
svc.unregister_capability(
    tenant_id="acme",
    capability_id="legacy_module",
    reason="Replaced by billing v2",
)
```

This removes the capability record, all its dependency edges, cached manifests, and health check
records. It does **not** remove composition blueprints that reference it — those will fail
subsequent dry-run validation.

---

## Managing Dependencies

### Adding a Dependency Edge

```python
svc.add_dependency(
    dependency_id="billing-depends-auth",
    tenant_id="acme",
    source_capability_id="billing",
    target_capability_id="auth",
    dependency_type="required",
    version_constraint=">=2.0,<3",
)
```

Valid `dependency_type` values: `required`, `optional`, `recommended`, `conflicting`, `enhancing`.

Cycle detection runs immediately. If the new edge creates a cycle, the add is rolled back and
`ValueError` is raised.

### Topological Resolution Order

```python
order = svc.dependency_resolution_order("acme", ["billing", "auth", "ntfy"])
print(order["resolution_order"])   # e.g. ['auth', 'ntfy', 'billing']
print(order["has_cycle"])          # False
```

### Blast Radius / Impact Analysis

Before deprecating a capability or releasing a breaking version, assess the blast radius:

```python
impact = await svc.impact_analysis("acme", "auth")
print(f"Direct dependents:     {impact['direct_dependents']}")
print(f"Transitive dependents: {impact['transitive_dependents']}")
print(f"Affected compositions: {impact['affected_compositions']}")
print(f"Risk level:            {impact['risk_level']}")  # low / medium / high
```

---

## Composition Blueprints

### Creating a Composition

```python
comp = svc.create_composition(
    composition_id="erp-core",
    tenant_id="acme",
    name="ERP Core Bundle",
    owner="platform-team",
    capability_ids=["billing", "auth", "ntfy", "inventory"],
)
print(comp["validation"]["valid"])  # True if all capabilities registered & deps met
```

### Dry-Run Simulation

Before publishing, simulate execution to catch blockers early:

```python
dry = await svc.dry_run_composition("acme", comp["id"])
if dry["executable"]:
    print("Safe to publish. Order:", dry["simulated_order"])
else:
    for blocker in dry["blockers"]:
        print(f"BLOCKER: {blocker}")
for warning in dry["warnings"]:
    print(f"WARNING: {warning}")
```

### Publishing

```python
svc.publish_composition(
    tenant_id="acme",
    composition_record_id=comp["id"],
    validation_evidence="CI run #4421 passed all contract tests",
)
```

### Comparing Compositions

```python
diff = await svc.diff_compositions("acme", comp_v1["id"], comp_v2["id"])
print("Added:   ", diff["added"])
print("Removed: ", diff["removed"])
for note in diff["migration_notes"]:
    print(note)
```

---

## Version Governance

```python
svc.release_version(
    release_id="billing-v1.3",
    tenant_id="acme",
    capability_id="billing",
    version="1.3.0",
    compatibility_evidence="Backward-compat test matrix v1.2→v1.3 attached in JIRA-4401",
    reviewed_by="alice@acme.com",
)
```

### Deprecating a Capability

Always perform impact analysis first, then deprecate with a migration plan:

```python
impact = await svc.impact_analysis("acme", "billing_v1")
# ... review impact ...

svc.deprecate_capability(
    tenant_id="acme",
    capability_id="billing_v1",
    migration_plan="Migrate consumers to billing v2 by 2026-09-01. See MIGRATION_GUIDE.md.",
)
```

---

## Marketplace Publication

```python
svc.publish_to_marketplace(
    publication_id="billing-pub-001",
    tenant_id="acme",
    capability_id="billing",
    documentation_ref="https://docs.acme.com/capabilities/billing",
    reviewed_by="bob@acme.com",
)
```

`marketplace_publish_requires_review` produces `require_review` — the publication proceeds if
`reviewed_by` is non-empty. Missing documentation raises `PermissionError`.

---

## Discovery and Search

### Full-Text Search

```python
results = svc.capability_search("acme", "invoice payment")
for cap in results:
    print(cap["capability_id"], cap["version"])
```

### Filtered Discovery

```python
# All active fintech capabilities that provide invoice_generation
caps = svc.discover_capabilities(
    tenant_id="acme",
    domain="fintech",
    provides_filter="invoice_generation",
    status_filter="registered",
)
```

### Manifests

```python
manifest = svc.get_capability_manifest("acme", "billing")
print(manifest["provides"])
print(manifest["health_status"])
```

### Compatibility Check

```python
compat = svc.check_compatibility("acme", "billing", "ntfy")
print(compat["compatible"])
print(compat["unresolved_a_requires"])
```

---

## Health and Observability

### Bulk Health Check

```python
health = svc.health_check_all("acme")
print(f"Total: {health['total']}, Healthy: {health['healthy']}, Degraded: {health['degraded']}")
```

### TTL Lease Heartbeat

Capabilities that own long-running processes should renew their lease periodically. Stale leases
are demoted to `failing` on the next `health_check_all` sweep.

```python
# Renew every 20 s with a 30 s TTL
lease = await svc.renew_health_lease("acme", "billing", ttl_s=30)
print(f"Lease expires at: {lease['expires_at']}")
```

### Contract Test Runner

Run synthetic contract validation and update `health_status` automatically:

```python
result = await svc.run_contract_tests("acme", "billing")
print(f"Passed: {result['passed']}/{result['scenarios_run']}")
print(f"Health updated to: {result['health_status_updated_to']}")
```

### Audit Log Replay (Point-in-Time Snapshot)

```python
snapshot = await svc.replay_audit_to_snapshot(
    "acme",
    up_to_iso_timestamp="2026-01-15T00:00:00+00:00",
)
print(f"Replayed {snapshot['event_count_replayed']} events")
print(snapshot["capability_states"])
```

---

## Governance and Compliance

### Capability Quality Scorecard

```python
score = await svc.score_capability("acme", "billing")
print(f"Grade: {score['grade']}  Score: {score['score_pct']}%")
for check in score["checks"]:
    status = "PASS" if check["passed"] else "FAIL"
    print(f"  [{status}] {check['check']}: {check['detail']}")
```

Scoring rubric (max 100 points):

| Check | Points | Condition |
|-------|--------|-----------|
| owner_set | 20 | `owner` field non-empty |
| contract_ref_set | 20 | `contract_ref` present |
| provides_non_empty | 20 | `provides` list non-empty |
| display_name_distinct | 10 | `display_name` != `name` |
| manifest_path_set | 15 | `manifest_path` present |
| health_healthy | 15 | `health_status == "healthy"` |

### Compliance Check

```python
report = await svc.registry_compliance_check("acme")
print(f"Compliance rate: {report['compliance_rate_pct']}%")
print(f"No-owner count:  {report['no_owner_count']}")
```

### Capability Usage Stats

```python
stats = await svc.capability_usage_stats("acme")
print(stats["top_capabilities"])
```

---

## World-Class Features (v2)

### Summary Table

| Method | Description | Analogue |
|--------|-------------|----------|
| `renew_health_lease` | TTL heartbeat; stale → failing | Consul TTL health checks |
| `impact_analysis` | Reverse-BFS blast radius | Maven `dependency:tree` reversed |
| `score_capability` | Quality scorecard (A-D, 0-100) | Backstage Scorecards |
| `diff_compositions` | Structured composition diff + migration notes | `terraform plan` |
| `dry_run_composition` | Simulation: cycles, conflicts, unmet requires | CloudFormation change sets |
| `auto_discover_capabilities` | Filesystem crawl → bulk register | Backstage auto-discovery |
| `replay_audit_to_snapshot` | Point-in-time audit replay | Kafka consumer replay |
| `run_contract_tests` | Synthetic rule evaluation → health update | Pact broker |

All 8 methods are `async` and designed to compose naturally in `asyncio.gather` fan-outs.

---

## Registry Agents

Register an AI agent to perform automated catalog curation or dependency review:

```python
agent = svc.register_registry_agent(
    tenant_id="acme",
    name="curator-bot",
    runtime="openai_functions",
    role="catalog_curator",
    instructions="Review newly registered capabilities and flag missing owners.",
)

# Validate a proposed action before executing
validation = svc.validate_agent_registry_action(
    tenant_id="acme",
    agent_id=agent["id"],
    action="deprecate_capability",
    privileged_scope=True,
    human_approval_recorded=True,
)
print(validation["decision"])  # allow / deny / require_review
```

Supported runtimes and roles are declared in `SUPPORTED_REGISTRY_AGENT_RUNTIMES` and
`SUPPORTED_REGISTRY_AGENT_ROLES` in `capability_contract.py`.

---

## Analytics

```python
analytics = svc.registry_analytics("acme", period="2026-Q2")
print(analytics["capability_count"])
print(analytics["by_category"])
print(analytics["by_status"])
print(analytics["healthy_count"])
```

Export the registry:

```python
export = await svc.export_registry("acme", format="csv")
print(export["content"])  # CSV string
```

---

## Reference — All Methods

### Synchronous Methods

| Method | Description |
|--------|-------------|
| `register_capability(...)` | Add a capability to the catalog |
| `add_dependency(...)` | Add a dependency edge with cycle detection |
| `create_composition(...)` | Create and validate a composition blueprint |
| `publish_composition(...)` | Publish a validated composition |
| `validate_composition(...)` | Check capability set validity |
| `release_version(...)` | Release a new version with compatibility evidence |
| `deprecate_capability(...)` | Deprecate with migration plan |
| `publish_to_marketplace(...)` | Prepare marketplace publication |
| `register_registry_agent(...)` | Register an AI registry agent |
| `validate_agent_registry_action(...)` | Gate an agent's proposed action |
| `validate_import_batch(...)` | Validate a batch import via Bytewax |
| `discover_capabilities(...)` | Filtered capability discovery |
| `get_capability_manifest(...)` | Retrieve full capability manifest |
| `check_compatibility(...)` | Check provides/requires compatibility between two caps |
| `dependency_resolution_order(...)` | Topological sort of a capability set |
| `health_check_all(...)` | Bulk health status sweep |
| `capability_search(...)` | Full-text search over catalog |
| `register_installed_package(...)` | Record an installed Python package |
| `unregister_capability(...)` | Remove capability and clean up related records |
| `registry_analytics(...)` | Adoption and health analytics |
| `dashboard_summary(...)` | High-level dashboard counts |
| `list_capabilities(...)` | List all tenant capabilities |
| `list_dependencies(...)` | List all tenant dependencies |
| `list_compositions(...)` | List all tenant compositions |
| `list_versions(...)` | List all version release records |
| `list_publications(...)` | List all marketplace publications |
| `list_registry_agents(...)` | List registered agents |
| `list_installed_packages(...)` | List installed package records |
| `audit_events(...)` | Retrieve audit event log |

### Async Methods

| Method | Description |
|--------|-------------|
| `renew_health_lease(tenant_id, capability_id, ttl_s)` | Renew TTL health lease |
| `impact_analysis(tenant_id, capability_id)` | Reverse-BFS blast-radius analysis |
| `score_capability(tenant_id, capability_id)` | Quality scorecard evaluation |
| `diff_compositions(tenant_id, base_id, target_id)` | Structured composition diff |
| `dry_run_composition(tenant_id, composition_id)` | Simulate composition execution |
| `auto_discover_capabilities(tenant_id, root_path, excluded_paths)` | Filesystem auto-discovery |
| `replay_audit_to_snapshot(tenant_id, up_to_iso_timestamp)` | Point-in-time audit replay |
| `run_contract_tests(tenant_id, capability_id)` | Synthetic contract test runner |
| `bulk_register_capabilities(tenant_id, capability_specs)` | Bulk register from spec list |
| `export_registry(tenant_id, format)` | Export registry as JSON or CSV |
| `health_check(tenant_id)` | Service-level health check |
| `registry_compliance_check(tenant_id)` | Compliance rate report |
| `capability_usage_stats(tenant_id)` | Usage frequency from audit log |

---

*© 2025 Datacraft — www.datacraft.co.ke*
