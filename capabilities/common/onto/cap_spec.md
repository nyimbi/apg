# Ontology Management Capability Specification

- **Capability Name**: Ontology Management
- **Capability ID**: `onto`
- **Category**: common
- **Version**: 1.0.0

## Purpose

ONTO is APG's package-backed ontology and vocabulary workbench. It gives
composed applications a deterministic ontology registry, term editor, synonym
manager, taxonomy graph, semantic mapping workbench, curation-review ledger,
publication queue, audit stream, UI route model, and theme contract.

The package is executable without external graph, search, metadata, or NLP
infrastructure. Production integrations should be added through explicit
adapters while this package keeps the APG contract and local ontology behavior
testable.

## Provided Services

- `ontology_registry`
- `taxonomy_management`
- `vocabulary_governance`
- `semantic_mapping`
- `term_curation`
- `publication_queue`
- `onto_operations`

## Required Services

- `tenant_context`
- `kngr` for knowledge-graph publication and context links
- `meta` for metadata concept linkage
- `nlpc` for natural-language term assistance
- Optional `audl`, `auth`, `srch`, and `grph` integration from registration metadata

## Runtime Surfaces

| File | Responsibility |
| --- | --- |
| `capability_contract.py` | Configuration schema, deterministic rule engine, UI routes, and theme. |
| `models.py` | Domain dataclasses for ontologies, terms, taxonomy edges, semantic mappings, curation reviews, publications, and audit events. |
| `ontology_runtime.py` | Deterministic IDs, label/status/type/confidence normalization, duplicate detection, taxonomy cycle checks, mapping-review posture, version bumping, and publication readiness helpers. |
| `service.py` | In-process ontology service enforcing tenant, owner, mapping-review, breaking-change-review, duplicate-term, approval, and publication guardrails. |
| `api.py` | Thin payload helpers for ontologies, terms, synonyms, taxonomy edges, mappings, reviews, publications, status, and compatibility calls. |
| `views.py` | Dashboard, ontology registry, term editor, taxonomy, mapping workbench, publication queue, and governance view models. |
| `app.py` | Package entrypoint, manifest, semantic model, and self-test surface. |

## Ontology Behavior

1. Register a tenant-scoped ontology with an accountable owner and domain.
2. Create owned terms with definitions, statuses, synonyms, external references,
   and metadata.
3. Curate terms and record explicit reviews for breaking changes.
4. Add taxonomy edges while preventing cycles.
5. Create semantic mappings to external concepts. Low-confidence mappings require
   review according to the configured confidence threshold.
6. Detect duplicate term labels before publication.
7. Publish only when approval is recorded, no duplicate terms exist, terms are
   curated, and low-confidence mappings have review evidence.

## Rules

- `tenant_context_required`
- `term_requires_owner`
- `publication_requires_approval`
- `breaking_change_requires_review`
- `low_confidence_mapping_requires_review`
- `duplicate_term_blocks_publication`

## UI

The package exposes 7 APG Python UI routes through `views.py` and the package
semantic model:

- `/onto/dashboard`
- `/onto/ontologies`
- `/onto/terms`
- `/onto/mappings`
- `/onto/publication`
- `/onto/governance`
- `/onto/settings`

## Theme

The package uses the `onto_vocabulary_workbench` APG theme contract. The theme
is optimized for compact vocabulary work: term cards, taxonomy trees, mapping
panels, and publication queues.

## Adapter Boundaries

The executable package does not call external systems directly. Production
integrations should be introduced through adapters for:

- Knowledge graph publication and graph traversal through KNGR or GRPH.
- Metadata catalog concept synchronization through META.
- NLPC-assisted term extraction, synonym suggestions, and definition drafting.
- Search indexes, vector indexes, external ontology stores, RDF/OWL exporters,
  SPARQL endpoints, curation workflows, approval systems, RBAC, and audit
  stores.

## Focused Verification

Use focused checks while battery-constrained:

```bash
./.venv/bin/python -m py_compile capabilities/common/onto/__init__.py capabilities/common/onto/models.py capabilities/common/onto/ontology_runtime.py capabilities/common/onto/service.py capabilities/common/onto/api.py capabilities/common/onto/views.py capabilities/common/onto/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/onto/test_capability_contract.py capabilities/common/onto/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/onto --json
./.venv/bin/apg capabilities publish-plan capabilities/common/onto --json
```
