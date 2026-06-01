# META Capability Plan

## Objective

Build one coherent metadata-management lifecycle and guardrail packet so APG
applications can compose a governed catalog before production discovery,
classification, lineage, search, catalog-agent, Bytewax lifecycle, and
persistence adapters are attached.

## Sequence

### 1. Specification

- Define asset, discovery, classification, lineage, quality, certification,
  glossary, publication, retirement, catalog-agent, Bytewax lifecycle, audit,
  UI, theme, and adapter boundaries.
- Separate dependency-light generated-app behavior from production runtime
  orchestration.
- Record non-goals to avoid claiming infrastructure that belongs to adapters.

### 2. Contract

- Expand tenant configuration for catalog, discovery, classification, lineage,
  quality, governance, agents, Bytewax lifecycle streams, adapters, UI, and
  theme.
- Add deterministic rules for registration, publication, classification,
  certification, discovery, lineage, glossary ownership, retirement, stale
  assets, review notes, supported catalog-agent runtimes, supported
  catalog-agent roles, agent scope, owner, purpose, machine contribution
  disclosure, privileged-role human approval, and Bytewax lifecycle batches.
- Add contract-level `review_evidence` metadata for durable statuses, policy
  fields, pending-review queues, and denied lifecycle-batch behavior.
- Add UI routes for dashboard, catalog, discovery, lineage, classification,
  quality, certification, glossary, impact, search, audit, adapters, catalog
  agents, lifecycle batches, and settings.

### 3. Control Plane

- Preserve the existing `APGMetadataService`.
- Add `MetaService` for generated applications and focused tests.
- Implement lifecycle records for assets, discovery jobs, classifications,
  lineage edges, quality assessments, certifications, glossary terms, and audit
  events.
- Implement catalog-agent records and lifecycle-batch records.
- Ensure all lifecycle methods evaluate rules and preserve matched-rule
  evidence.
- Persist `policy_decision`, `matched_rules`, `review_reasons`, and
  `review_evidence` on reviewable lifecycle records and audit events.
- Preserve otherwise valid privileged catalog agents without approval as
  `pending_review` records instead of transient exceptions.
- Persist denied non-Bytewax lifecycle batches as `denied` evidence before
  raising `PermissionError`.

### 4. Composition Surfaces

- Add generated-application API helper functions.
- Add `view_models.py` for route-level UI state.
- Add pending-review queue composition for dashboard, catalog-agent roster, and
  settings surfaces.
- Add dependency-light metadata fixture discovery for Oracle, SQL Server,
  Redis, and BigQuery connectors so generated applications can compose catalog
  workflows before live vendor drivers are installed.
- Replace static semantic JSON with contract-derived `app.py` output.
- Refresh `semantic_model.json`, `release_report.json`, and package manifest.

### 5. Documentation

- Replace root `README.md` with current executable scope.
- Add `SPECIFICATION.md` and this plan.
- Replace `cap_spec.md` with practical package summary.

### 6. Review And Proof

- Expand focused tests for contract shape, rule engine behavior, service
  lifecycle, generated UI models, registration metadata, and app evidence.
- Run battery-conscious proof only:
  - `py_compile` for META packet files.
  - focused META pytest files.
  - focused database connector fixture tests.
  - APG implementation audit for `capabilities/common/meta`.
  - APG publish plan for `capabilities/common/meta`.
  - stale-marker search over current packet artifacts.
  - `git diff --check` for META and progress log files.

## Follow-On Work

- Connect `MetaService` decisions to durable persistence in `APGMetadataService`.
- Add production Bytewax flow definitions for metadata catalog and catalog-agent
  lifecycle events.
- Add real runtime adapter shims for Codex, Claude Code, opencode, Pi, and
  later AI-agent providers without making any one runtime mandatory.
- Add live adapter tests for discovery, classification, lineage, search,
  metadata store, auth, audit, MDM, ETL, connector, monitoring, and notification
  integration.
- Replace fixture-backed Oracle, SQL Server, Redis, and BigQuery connectors with
  live driver adapters where deployments provide credentials and dependencies.
- Add rendered UI shells after generated-application targets stabilize.
- Add performance and concurrency benchmarks when running on AC power.
