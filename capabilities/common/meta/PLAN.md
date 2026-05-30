# META Capability Plan

## Objective

Build one coherent metadata-management lifecycle and guardrail packet so APG
applications can compose a governed catalog before production discovery,
classification, lineage, search, and persistence adapters are attached.

## Sequence

### 1. Specification

- Define asset, discovery, classification, lineage, quality, certification,
  glossary, publication, retirement, audit, UI, theme, and adapter boundaries.
- Separate dependency-light generated-app behavior from production runtime
  orchestration.
- Record non-goals to avoid claiming infrastructure that belongs to adapters.

### 2. Contract

- Expand tenant configuration for catalog, discovery, classification, lineage,
  quality, governance, adapters, UI, and theme.
- Add deterministic rules for registration, publication, classification,
  certification, discovery, lineage, glossary ownership, retirement, stale
  assets, and review notes.
- Add UI routes for dashboard, catalog, discovery, lineage, classification,
  quality, certification, glossary, impact, search, audit, adapters, and
  settings.

### 3. Control Plane

- Preserve the existing `APGMetadataService`.
- Add `MetaService` for generated applications and focused tests.
- Implement lifecycle records for assets, discovery jobs, classifications,
  lineage edges, quality assessments, certifications, glossary terms, and audit
  events.
- Ensure all lifecycle methods evaluate rules and preserve matched-rule
  evidence.

### 4. Composition Surfaces

- Add generated-application API helper functions.
- Add `view_models.py` for route-level UI state.
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
  - APG implementation audit for `capabilities/common/meta`.
  - APG publish plan for `capabilities/common/meta`.
  - stale-marker search over current packet artifacts.
  - `git diff --check` for META and progress log files.

## Follow-On Work

- Connect `MetaService` decisions to durable persistence in `APGMetadataService`.
- Add production Bytewax flow definitions for metadata catalog events.
- Add live adapter tests for discovery, classification, lineage, search,
  metadata store, auth, audit, MDM, ETL, connector, monitoring, and notification
  integration.
- Add rendered UI shells after generated-application targets stabilize.
- Add performance and concurrency benchmarks when running on AC power.
