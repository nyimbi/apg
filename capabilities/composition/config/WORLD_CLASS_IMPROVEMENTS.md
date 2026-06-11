# World-Class Improvements: Composition Configuration

© 2025 Datacraft — Nyimbi Odero

---

### I1. Reactive Config Push via NATS JetStream
**Category**: Streaming Architecture | **Justification**: Current service is pull-only; config consumers must poll or restart to pick up changes. NATS JetStream subjects allow the service to push config-changed events to all subscriber replicas in sub-millisecond latency, eliminating stale-config windows that cause subtle production divergence. 10x better because stale config is the root cause of an entire class of "works on my machine" incidents. | **Implementation**: On every `set_config` / `rollback_config` / `bulk_config_import`, publish a `apg.composition.config.changed.<namespace>.<key>` NATS subject with the new version payload. Consumer SDK subscribes with durable consumers and at-least-once ack policy. Use NATS KV store as the authoritative config mirror so consumers can read the latest value without hitting the service at all. | **Competitor**: HashiCorp Consul watch + notify is the reference; NATS KV is faster and requires zero additional infrastructure in an APG deployment.

---

### I2. Hierarchical Config Inheritance with Override Chains
**Category**: Configuration Model | **Justification**: Flat namespace/key pairs force every environment to duplicate base values. A Google Cloud Config / Spring Cloud Config style hierarchy (`global → capability → environment → tenant`) lets 95 % of keys inherit defaults while overrides stay minimal. Maintenance burden drops proportionally. | **Implementation**: Add `parent_namespace_id: str | None` to `ConfigNamespaceRecord`. Introduce `resolve_config(namespace, key, tenant_id)` that walks the ancestor chain, merges values, and returns a resolved `ConfigValue` with a `resolved_from` provenance list. Cache resolved values in a bounded LRU keyed by `(tenant, namespace, key, version_vector)`. | **Competitor**: Spring Cloud Config hierarchical profiles; AWS AppConfig deployment strategies.

---

### I3. Zero-Downtime Schema Evolution with Migration Engine
**Category**: Schema Management | **Justification**: Schema changes currently break all existing configs that don't match the new shape. A forward/backward compatibility checker (similar to Confluent Schema Registry but without Kafka) validates that new schemas are either fully compatible or provides a migration function. Prevents silent data loss during capability upgrades. | **Implementation**: Store schemas with `schema_version` and `compatibility_mode` (BACKWARD, FORWARD, FULL). On `validate_schema`, run a diff against the previous version and reject breaking changes unless `force=True` with approval. Provide `migrate_config_to_schema(config_id, target_schema_version)` that applies registered migration lambdas. | **Competitor**: Confluent Schema Registry compatibility enforcement; AWS Glue Schema Registry.

---

### I4. Config-as-Code GitOps Round-Trip with bytewax Pipelines
**Category**: GitOps | **Justification**: Manual config updates through the API are error-prone and hard to review. A GitOps round-trip where the canonical source is a git repo and the service syncs state bidirectionally is the gold standard for regulated environments. bytewax pipelines replace the Kafka Connect source connector with a lightweight Python process. | **Implementation**: Extend `gitops_engine.py` to emit to `apg.composition.config.gitops.*` NATS subjects. A bytewax dataflow subscribes, parses YAML config manifests from git push webhooks, calls `bulk_config_import`, and publishes `config_synced` events. Add `sync_from_git(repo_url, branch, namespace_mapping)` async method to the service. | **Competitor**: ArgoCD + Helm values files; Flux CD config management.

---

### I5. Probabilistic Config Canary with Automated Rollback
**Category**: Deployment Safety | **Justification**: Current canary support is evidence-string only — entirely manual. A probabilistic canary system serves the new config to a percentage of traffic, collects error-rate and latency signals from the observability plane, and auto-rolls back if SLOs breach — without human intervention. This is Shopify/Netflix-level deployment safety. | **Implementation**: Add `canary_percentage: float` and `slo_thresholds: dict` to `ConfigDeploymentRecord`. The service publishes canary config versions to a NATS subject with a `canary_slot` header. A bytewax dataflow reads APM metrics from the `apg.moni.metrics` subject, evaluates SLOs, and calls `rollback_configuration` automatically. | **Competitor**: LaunchDarkly progressive rollouts; Netflix Kayenta automated canary analysis.

---

### I6. Immutable Audit Log with Cryptographic Chaining
**Category**: Compliance / Audit | **Justification**: The current audit list is mutable in-memory — a compromised process can erase events. Cryptographically chained audit records (each record includes the SHA-256 hash of the previous record) make tampering detectable, satisfying SOC 2 Type II, PCI-DSS, and GDPR audit requirements. | **Implementation**: Add `previous_hash: str` and `record_hash: str` to `ConfigAuditEventRecord`. The `_audit` helper computes `record_hash = sha256(event_payload + previous_hash)`. Provide `verify_audit_chain(tenant_id)` that replays the hash chain and returns any broken links. Persist to an append-only PostgreSQL table with `GENERATED ALWAYS AS` immutability triggers. | **Competitor**: AWS CloudTrail log file integrity validation; Hyperledger Fabric audit ledger.

---

### I7. Declarative Config Policies via OPA (Open Policy Agent)
**Category**: Policy Engine | **Justification**: The current rule engine is a hand-rolled if/elif structure in `capability_contract.py`. As policy complexity grows, this becomes unmaintainable. OPA Rego policies are composable, testable, version-controlled, and can be loaded at runtime without redeploying the service. | **Implementation**: Add `policy_engine: str = "rego"` to the capability contract. Load `.rego` bundles from a configurable NATS object store bucket. Cache compiled policy modules per tenant. The `evaluate` method dispatches to the OPA SDK `allowed()` call. Fall back to the existing rule engine if OPA is unavailable. | **Competitor**: HashiCorp Sentinel policy-as-code; AWS Cedar authorization language.

---

### I8. Multi-Region Config Replication with CRDT Merge
**Category**: Distribution | **Justification**: Multi-region deployments that write config to a single region create a single point of failure and add cross-region latency to every config read. CRDTs (specifically a Last-Write-Wins register keyed on `(tenant, namespace, key, version)`) allow eventually-consistent multi-region writes with deterministic conflict resolution. | **Implementation**: Represent each config value as a LWW-register. Use NATS JetStream geo-replication to propagate `config_changed` events across regions. On conflict (same key, different value, same version), apply the LWW rule (highest `updated_at` wins) and emit a `config_conflict_resolved` audit event. | **Competitor**: Consul multi-datacenter replication; etcd linearizable replication.

---

### I9. Semantic Config Search with Vector Embeddings
**Category**: Discoverability | **Justification**: Large deployments accumulate thousands of config keys. Finding the right key requires knowing its exact path — there is no fuzzy or semantic search. Embedding config key paths and descriptions enables natural-language queries like "find all timeout settings for the payment service" with sub-100 ms retrieval. | **Implementation**: On `set_config`, generate a vector embedding of `f"{namespace}/{key}: {description}"` using a locally hosted Ollama embedding model (`nomic-embed-text`). Store in pgvector. Add `search_configs(query, tenant_id, top_k=10)` async method that embeds the query and performs an `<=>` cosine similarity search. | **Competitor**: Elastic App Search config discovery; Dynatrace Settings 2.0 semantic search.

---

### I10. Differential Config Snapshots with Storage Compression
**Category**: Storage Efficiency | **Justification**: The current version history stores full copies of the config value at every version. For large JSON configs with minor changes, this inflates storage by 100x over a delta-only store. JSON-patch diffs (RFC 6902) between consecutive versions reduce history storage by 80-90% while enabling exact reconstruction at any version. | **Implementation**: In `_snapshot_version`, compute the JSON-patch diff between the current value and the previous snapshot. Store the patch instead of the full value. In `config_version_history`, reconstruct versions by replaying patches forward from the earliest full snapshot. Add a `compact_history(config_id, keep_full_every=10)` maintenance method. | **Competitor**: Git object store delta compression; PostgreSQL TOAST compression.

---

### I11. Config Dependency Graph with Impact Analysis
**Category**: Change Management | **Justification**: Config keys frequently depend on each other (a `database.pool_size` should not exceed `database.max_connections`). There is currently no way to know what downstream capabilities will be affected before deploying a config change. Dependency graph traversal enables automated blast-radius estimation. | **Implementation**: Add a `DependencyEdge` record linking `(source_config_id, target_config_id, constraint_type)`. Provide `analyze_change_impact(config_id, tenant_id)` that returns a DAG of affected configs, capabilities, and deployments. Integrate with the deployment workflow: high blast-radius changes automatically escalate `impact_level`. | **Competitor**: Backstage dependency graph; ServiceNow Change Impact Analysis.

---

### I12. Hot-Reload Config Subscriptions for Long-Running Services
**Category**: Runtime Integration | **Justification**: Long-running services (Bytewax dataflows, FastAPI applications) must restart to pick up config changes. A subscription model delivers updates in-process without restart, enabling zero-downtime config reloads for database connection strings, feature flags, and rate limits. | **Implementation**: Add `subscribe_config(namespace, key, tenant_id, callback)` that registers an async callback. Internally, a background NATS subscription delivers `config_changed` events and invokes all registered callbacks. Provide a context manager `async with config_watch(namespace, key)` that yields new values as an async iterator. | **Competitor**: Consul watches with long-polling; Spring Cloud Config `@RefreshScope`; LaunchDarkly streaming SDK.

---

### I13. Config Encryption-at-Rest with Key Rotation
**Category**: Security | **Justification**: Secret values are currently stored as vault references (correct) but non-secret values are stored in plaintext. In regulated industries (finance, health), all config values may be subject to encryption requirements. Key rotation without service downtime is a hard requirement for SOC 2 and ISO 27001 compliance. | **Implementation**: Add `encryption_key_id: str | None` to `ConfigurationRecord`. Use AES-256-GCM envelope encryption: a Data Encryption Key (DEK) per namespace encrypted by a Key Encryption Key (KEK) stored in the local Vault instance. Add `rotate_namespace_encryption_key(namespace_id, new_key_id, tenant_id)` that re-encrypts all DEKs without touching the data. | **Competitor**: AWS KMS envelope encryption; HashiCorp Vault Transit secrets engine.

---

### I14. Config Linting and Best-Practice Enforcement
**Category**: Developer Experience | **Justification**: Operators can set configs that are syntactically valid but semantically dangerous (e.g., `log_level=DEBUG` in production, `timeout=0`). A linting layer catches these before deployment, similar to how `pylint` catches Python anti-patterns before they reach production. | **Implementation**: Define a `ConfigLintRule` model with `name`, `condition_expr` (Python expression evaluated against the config value), `severity`, and `environments` (applies only to specific environments). Register rules via `register_lint_rule`. Call `lint_config(config_id, environment)` before any deployment. Rules ship with sensible defaults and are overridable per-tenant. | **Competitor**: `cfn-lint` for CloudFormation; `kube-linter` for Kubernetes manifests; Spectral for OpenAPI.

---

### I15. Time-Locked Config Scheduling with Audit TTL
**Category**: Operations | **Justification**: Many config changes need to take effect at a precise time (maintenance windows, pricing changes at midnight, feature launches at a specific date). Manual coordination is error-prone and causes SLA breaches. Scheduled configs with atomic apply-at semantics eliminate this class of incident. | **Implementation**: Add `effective_at: datetime | None` and `expires_at: datetime | None` to `ConfigurationRecord`. A bytewax dataflow subscribes to a NATS scheduler subject and calls `activate_scheduled_configs(tenant_id)` at the scheduled time. Expired configs are automatically soft-deleted. Add `schedule_config_change(config_id, effective_at, expires_at, reason)` async method. | **Competitor**: LaunchDarkly scheduled flags; AWS AppConfig scheduled deployments; Kubernetes CronJob-triggered config maps.
