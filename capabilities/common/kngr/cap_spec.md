# Knowledge Graph Capability Specification

- **Capability Name**: Knowledge Graph
- **Capability ID**: `kngr`
- **Category**: common
- **Version**: 1.0.0

## Purpose

KNGR is APG's package-backed knowledge graph runtime. It manages tenant-scoped
knowledge sources, resolved entities, evidence-backed relationships, semantic
enrichments, bounded reasoning paths, curation decisions, graph publications,
audit events, UI route metadata, theme metadata, rule evaluation, semantic-model
publication, and publish-plan evidence.

The package is dependency-light and deterministic. Live graph databases, vector
stores, ontology stores, search indexes, NLPC enrichers, META metadata sources,
streaming ingestion, and external governance stores remain adapter boundaries
until a future slice wires and verifies them directly.

## Provided Services

- `entity_resolution`
- `semantic_enrichment`
- `knowledge_graphs`
- `reasoning_paths`
- `contextual_relationships`
- `kngr_operations`

## Required Services

- `tenant_context`
- `grph` for graph persistence adapters
- `nlpc` for language-derived entity and relation extraction
- `meta` for source metadata and provenance
- optional `audl`, `onto`, and `srch` adapters for audit, ontology, and search integration

## Runtime Surfaces

| File | Runtime responsibility |
| --- | --- |
| `models.py` | Knowledge source, entity, relationship, enrichment, reasoning, curation, publication, and audit dataclasses |
| `knowledge_runtime.py` | Deterministic IDs, confidence normalization, reasoning depth, publication status, and neighborhood helpers |
| `service.py` | Tenant-aware source registration, entity resolution, relationship linking, enrichment, reasoning, curation, publication, context exploration, summaries, and guardrails |
| `api.py` | Dependency-light API helper functions over the service |
| `views.py` | Dashboard, entity browser, curation queue, reasoning, context explorer, and governance view models |
| `app.py` | Publishable APG package entrypoint and semantic-model evidence |

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Runtime operations require tenant context.
Knowledge graph behavior is governed by the `knowledge`, `reasoning`,
`governance`, `ui`, and `theme` configuration sections.

## Rules

KNGR evaluates the deterministic rules from the capability contract:

- `tenant_context_required`
- `entity_resolution_requires_source`
- `semantic_enrichment_requires_confidence`
- `reasoning_requires_evidence`
- `deep_reasoning_requires_review`
- `uncurated_public_graph_blocked`

The service enforces these rules directly. Missing tenant context, missing
source evidence, missing reasoning evidence, unreviewed deep reasoning, and
uncurated publication are blocked. Low-confidence enrichment and relationship
linking require an explicit review record.

## UI And Theme

The package exposes six APG Python UI routes:

- `/kngr/dashboard`
- `/kngr/entities`
- `/kngr/curation`
- `/kngr/reasoning`
- `/kngr/context`
- `/kngr/settings`

View helpers expose summary metrics, entity and relationship inventories,
curation queues, reasoning paths, context neighborhoods, governance rules,
audit events, and graph publication state. The package uses the
`kngr_semantic_console` theme contract with graph, entity-card, reasoning-path,
and context-panel component tokens.

## Adapter Boundaries

This package intentionally does not open network connections or require a graph
database. Production deployments should attach adapters for:

- graph storage and traversal (`grph`);
- ontology and schema alignment (`onto`);
- language extraction and semantic labeling (`nlpc`);
- source metadata and lineage (`meta`);
- audit logging (`audl`);
- search indexing (`srch`);
- Bytewax ingestion flows for streaming source events;
- vector stores or external reasoning engines.

The in-process service remains the executable APG behavior used by generated
apps, tests, publish-plan checks, and local capacity slices.

## Focused Verification

Use battery-conscious verification for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/kngr/__init__.py capabilities/common/kngr/models.py capabilities/common/kngr/knowledge_runtime.py capabilities/common/kngr/service.py capabilities/common/kngr/api.py capabilities/common/kngr/views.py capabilities/common/kngr/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/kngr/test_capability_contract.py capabilities/common/kngr/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/kngr --json
./.venv/bin/apg capabilities publish-plan capabilities/common/kngr --json
```
