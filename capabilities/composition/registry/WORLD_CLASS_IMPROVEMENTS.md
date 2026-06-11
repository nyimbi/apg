# Composition Registry — World Class Improvements

15 targeted improvements drawn from production-grade service registries (Consul, Netflix Eureka,
AWS Service Catalog, Apigee, Backstage) and capability governance literature.

---

## 1. Distributed Lease-Based Health Heartbeat

**Category:** Reliability / Health Tracking

**Justification:**
Current health derives health status from static `status` field values. A production registry must
distinguish between "was healthy at last check" and "is healthy right now." Consul's TTL-based
health checks and Kubernetes liveness probes both use active heartbeat leases — if a capability
fails to renew, it transitions to `failing` automatically without a polling scan.

**Implementation:**
Add `_leases: dict[str, dict]` keyed by `tenant:capability_id` storing `{expires_at, interval_s,
last_heartbeat}`. `async def renew_health_lease(tenant_id, capability_id, ttl_s)` updates
`expires_at = now + ttl_s`. A background sweep (or lazy check at read time) downgrades stale
leases to `failing`. `health_check_all` consults lease expiry first.

**Competitor reference:** HashiCorp Consul TTL health checks; Kubernetes liveness probe with
`failureThreshold`.

---

## 2. Semantic Versioning Constraint Solver

**Category:** Dependency Management

**Justification:**
`version_constraint` is stored as a free-form string. When resolving transitive dependencies,
conflicting constraints (e.g. `>=2.0,<3` vs `>=2.5`) are never evaluated — composition validation
can pass even when no satisfying version exists. npm's semver solver and Poetry's dependency
resolver both handle this.

**Implementation:**
Integrate `packaging.version` (already a transitive dep via pip). Add
`async def resolve_version_constraints(tenant_id, capability_ids) -> dict` that collects all
declared `version_constraint` strings for transitive deps, parses them with
`packaging.specifiers.SpecifierSet`, and reports `satisfiable: bool` plus the intersection range
for each capability. Return conflicts as structured errors.

**Competitor reference:** npm semver, Python `packaging`, Poetry dependency resolver.

---

## 3. Lazy-Loaded Filesystem Auto-Discovery

**Category:** Developer Experience / Automation

**Justification:**
The `CRRegistry.auto_discovery_enabled` field and `discovery_paths` are modeled but never
exercised in `service.py`. Backstage's catalog auto-discovery and AWS Service Catalog portfolio
imports both treat filesystem/git crawling as a first-class path to populate the catalog without
manual `register_capability` calls.

**Implementation:**
`async def auto_discover_capabilities(tenant_id, root_path, excluded_paths=None)` walks
`root_path` using `pathlib.Path.rglob("__init__.py")`, calls the existing
`_extract_capability_metadata` for each file, and bulk-registers results via
`register_capability`. Returns `{discovered, registered, skipped, errors}`.

**Competitor reference:** Backstage Software Catalog auto-discovery; AWS Service Catalog portfolio
import.

---

## 4. Capability Scoring and Quality Gate

**Category:** Governance / Quality

**Justification:**
No mechanism enforces baseline documentation or metadata quality before a capability can be
published. Spotify Backstage's Scorecards and Google's internal capability maturity model both
assign scores across dimensions (docs, tests, owner, SLO) and gate promotion to higher lifecycle
stages.

**Implementation:**
`async def score_capability(tenant_id, capability_id) -> dict` evaluates: owner set (+20),
contract_ref set (+20), provides non-empty (+20), display_name != name (+10), manifest_path set
(+15), health_status healthy (+15). Returns `{score, max_score, grade, passed_checks,
failed_checks}`. Optionally block `publish_to_marketplace` if score < threshold.

**Competitor reference:** Spotify Backstage Scorecards; Google production-readiness reviews.

---

## 5. Transitive Dependency Impact Analysis

**Category:** Dependency Management / Risk

**Justification:**
When a capability is deprecated or a breaking version is released, operators need to know the
blast radius across all downstream consumers — not just direct dependents. Maven's dependency tree
and Snyk's dependency graph both surface transitive impact.

**Implementation:**
`async def impact_analysis(tenant_id, capability_id) -> dict` performs a reverse BFS/DFS from
`capability_id` through the dependency graph, collecting all direct and transitive dependents,
their statuses, and composition memberships. Returns `{direct_dependents, transitive_dependents,
affected_compositions, risk_level}`.

**Competitor reference:** Maven `dependency:tree`; Snyk dependency graph; Renovate bot impact
scoring.

---

## 6. Event Replay and Audit Reconstruction

**Category:** Observability / Compliance

**Justification:**
The audit log is append-only and queryable per-tenant, but there is no mechanism to replay events
to reconstruct the state of the registry at a point in time — essential for compliance audits and
incident post-mortems. Apache Kafka's log replay and EventStore's projections both enable this.

**Implementation:**
`async def replay_audit_to_snapshot(tenant_id, up_to_iso_timestamp) -> dict` iterates
`_audit_events` filtering by `created_at <= up_to_iso_timestamp` and reconstructs a read-only
snapshot dict of capability states at that point. Returns `{snapshot_at, capability_states,
event_count_replayed}`.

**Competitor reference:** Apache Kafka consumer replay; EventStore projections; AWS CloudTrail
log replay.

---

## 7. Composition Diff and Migration Plan Generator

**Category:** Release Management / Developer Experience

**Justification:**
When updating a composition (changing the capability_ids list), operators have no structured view
of what changed — added, removed, or version-bumped capabilities. Terraform's `plan` output and
Kubernetes `kubectl diff` both make incremental changes explicit before applying them.

**Implementation:**
`async def diff_compositions(tenant_id, base_composition_id, target_composition_id) -> dict`
compares `capability_ids` lists and version records for both compositions, returning
`{added, removed, version_changed, unchanged, migration_notes}`. `migration_notes` are
auto-generated strings like "Removed capability X — ensure downstream consumers are updated."

**Competitor reference:** Terraform plan; `kubectl diff`; Helm `helm diff` plugin.

---

## 8. Multi-Tenant Capability Sharing and Visibility Control

**Category:** Multi-Tenancy / Marketplace

**Justification:**
All capabilities are scoped to a single tenant. There is no model for a tenant to share a
capability with selected peers or make it globally visible — a prerequisite for a real marketplace.
Azure Service Catalog and AWS RAM (Resource Access Manager) both allow cross-account resource
sharing with explicit grants.

**Implementation:**
Add `_sharing_grants: dict[str, dict]` keyed by `{tenant_id}:{capability_id}:{grantee_tenant_id}`.
`async def grant_capability_access(tenant_id, capability_id, grantee_tenant_id, access_level)`
creates a grant record. `discover_capabilities` gains a `include_shared=True` parameter that
additionally returns capabilities granted to `tenant_id` from other tenants.

**Competitor reference:** AWS Resource Access Manager; Azure Service Catalog shared galleries;
Apigee portal cross-org sharing.

---

## 9. Circuit-Breaker State Tracking per Capability

**Category:** Reliability / Resilience

**Justification:**
When a capability degrades repeatedly, the registry should surface a circuit-breaker state
(closed/open/half-open) so orchestrators can skip it during composition routing without waiting
for a full health check round-trip. Netflix Hystrix and Resilience4j both implement this pattern.

**Implementation:**
Add `_circuit_states: dict[str, dict]` keyed by `tenant:capability_id` with
`{state, failure_count, last_failure_at, opened_at}`. `async def record_capability_failure` and
`async def record_capability_success` update the counts. State transitions:
`failure_count >= threshold` → `open`; after `recovery_window_s` → `half-open`; success in
half-open → `closed`. `health_check_all` includes `circuit_state` in each capability report.

**Competitor reference:** Netflix Hystrix; Resilience4j circuit breaker; AWS App Mesh outlier
detection.

---

## 10. Signed Capability Manifests with Integrity Verification

**Category:** Security / Supply Chain

**Justification:**
Capability manifests can be tampered between registration and consumption. SLSA provenance,
sigstore, and in-toto attestations all establish a cryptographic chain of custody for software
supply chain artifacts. The registry should store a content hash and optionally a signature for
each registered manifest.

**Implementation:**
`async def sign_capability_manifest(tenant_id, capability_id, signing_key_ref) -> dict`
serialises the manifest to canonical JSON, computes `sha256`, stores `{content_hash, signed_at,
signing_key_ref}` in the capability record. `async def verify_capability_manifest(tenant_id,
capability_id)` recomputes the hash and compares.

**Competitor reference:** sigstore/cosign; SLSA provenance; in-toto attestations; npm package
integrity hashes.

---

## 11. Canary and Staged Rollout Tracking

**Category:** Release Management / Progressive Delivery

**Justification:**
Version releases are binary (released/not released) with no support for staged rollout — deploying
to 5% of tenants first, monitoring, then expanding. Argo Rollouts, Flagger, and LaunchDarkly all
model progressive delivery as a first-class concept in their service registries.

**Implementation:**
Add `_rollout_plans: dict[str, dict]` per version record with `{stages: [{pct, tenant_ids,
promoted_at}], current_stage, status}`.
`async def create_rollout_plan(tenant_id, version_record_id, stages)` initialises the plan.
`async def promote_rollout_stage(tenant_id, version_record_id)` advances to the next stage,
emitting `rollout_stage_promoted` events. `list_versions` gains a `rollout_status` field.

**Competitor reference:** Argo Rollouts; Flagger; Spinnaker canary analysis; LaunchDarkly staged
rollouts.

---

## 12. Capability Deprecation Notification Workflow

**Category:** Operations / Communication

**Justification:**
`deprecate_capability` records a migration plan but does not notify downstream consumers.
In production registries (AWS deprecation notices, GCP API sunset policy), affected consumers
receive structured notifications with timelines. The `ntfy` capability is declared as a dependency
but never invoked.

**Implementation:**
`async def notify_deprecation_consumers(tenant_id, capability_id, sunset_date, migration_guide)`
queries all dependencies where `target_capability_id == capability_id`, collects the owning
tenant/capability records, and emits `deprecation_notice_sent` events with
`{sunset_date, migration_guide, affected_capability_id, consumer_capability_id}`. Plugs into
`ntfy` capability when available.

**Competitor reference:** AWS deprecation notices; GCP API sunset headers; npm deprecation
warnings in `npm install`.

---

## 13. Composition Execution Dry-Run Validation

**Category:** Safety / Testing

**Justification:**
`validate_composition` checks for missing capabilities and unmet dependencies but does not
simulate execution ordering, detect resource conflicts, or verify that capability contracts are
mutually satisfiable. Google Cloud Deployment Manager's preview mode and Terraform's `-plan` both
run a dry-run before any resource is created.

**Implementation:**
`async def dry_run_composition(tenant_id, composition_id) -> dict` performs full topological
sort, simulates contract surface matching for each capability pair in dependency order, checks for
`conflicting` dependency edges within the set, and verifies all `requires` surfaces are satisfied
by some `provides` in the set. Returns `{executable, blockers, warnings, simulated_order}`.

**Competitor reference:** Terraform `plan`; Google Cloud Deployment Manager preview;
AWS CloudFormation change sets.

---

## 14. Registry Federation and Peer Sync

**Category:** Scalability / Multi-Registry

**Justification:**
Large organisations run multiple independent registries (per region, per BU). Without a federation
model, capabilities cannot be discovered across registries. Netflix's Eureka federation, Consul's
WAN gossip, and the OCI Distribution Spec for container registries all support multi-registry
peer replication.

**Implementation:**
Add `_peer_registries: dict[str, dict]` storing peer endpoint configs. `async def sync_from_peer`
pulls capability manifests from a peer registry endpoint (simulated as a dict merge in-process),
marks records with `{source_registry, synced_at}`. `discover_capabilities` gains a
`include_federated=True` flag.

**Competitor reference:** Netflix Eureka server replication; Consul WAN federation; OCI
Distribution Spec mirroring; Helm chart repository federation.

---

## 15. Capability Contract Test Runner

**Category:** Quality / Continuous Validation

**Justification:**
Capability contracts (`capability_contract.py`) declare rules but there is no in-registry
mechanism to run them continuously and surface failures. Pact (consumer-driven contract testing),
Spring Cloud Contract, and Karate DSL all integrate contract execution into the CI/CD lifecycle
and surface results in the service registry.

**Implementation:**
`async def run_contract_tests(tenant_id, capability_id) -> dict` dynamically imports the
capability's `contract_ref` module, invokes `evaluate_capability_rules` against a set of
synthetic test contexts (happy path, missing tenant, missing policy), collects pass/fail results,
stores them in `_contract_test_results`, and updates `health_status` based on outcome.

**Competitor reference:** Pact broker; Spring Cloud Contract verifier; Karate DSL contract
runner; AWS Service Catalog constraint evaluation.
