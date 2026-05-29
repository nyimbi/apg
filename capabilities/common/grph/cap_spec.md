# Graph Data Management Capability Specification

- **Capability Name**: Graph Data Management
- **Capability ID**: `grph`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`grph` provides tenant-aware graph schema management, node and edge storage,
bounded traversals, lineage paths, graph-quality reporting, UI metadata, and
publishable package evidence for APG-generated Python applications.

The package is dependency-light and executable in-process. Production graph
databases, search indexes, cache layers, lineage catalog adapters, audit sinks,
and visualization renderers should be attached behind APG adapters rather than
hard-coded into the package facade.

## Runtime Surfaces

- `GraphSchema`: graph kind, node types, edge types, and source asset metadata.
- `GraphNode`: tenant-scoped node with type, owner, labels, properties, and
  optional source asset.
- `GraphEdge`: typed relationship with owner, classification, and properties.
- `GraphTraversalResult`: bounded traversal output with visited node and edge
  identifiers.
- `GraphQualityReport`: orphan-node, missing-owner, and restricted-edge
  summary.
- `graph_runtime.py`: GRPH-specific traversal planning and quality inspection
  algorithms that keep graph behavior outside generic package scaffolding.
- `GrphService`: executable facade for schema creation, node writes, edge
  writes, traversals, lineage paths, quality reports, compatibility records,
  dashboard summaries, and contract rule evaluation.
- `api.py`: dependency-light helpers for generated apps and package probes.
- `views.py`: graph explorer, schema manager, lineage viewer, quality console,
  dashboard, routes, rules, and theme view models.

## Provided Services

- `graph_store`
- `relationship_modeling`
- `graph_traversal`
- `lineage_graphs`
- `graph_quality`

## Required Services

- `tenant_context`

Optional production adapters may use `mdm`, `meta`, `etlp`, `auth`, `audl`,
`cach`, and `moni` as described by package registration metadata.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. The graph configuration defines schema enforcement, maximum default
traversal depth, and lineage graph support.

## Rules

The package enforces the contract rule engine through `GrphService`:

- `tenant_context_required`
- `node_write_requires_owner`
- `edge_write_requires_type`
- `restricted_relationship_requires_review`
- `deep_traversal_requires_review`
- `lineage_graph_requires_source_asset`

Additional service guardrails validate schema existence, same-tenant node and
edge references, schema node type membership, schema edge type membership, and
bounded traversal behavior.

## UI

The package exposes APG Python UI route contracts and dependency-light view
models for:

- dashboard
- graph explorer
- schema manager
- lineage viewer
- graph quality console
- settings

## Theme

The package uses the `grph_relationship_console` APG theme contract with graph
canvas, node panel, edge panel, and lineage path component tokens.

## Verification

Use focused package verification first:

```bash
./.venv/bin/python -m py_compile capabilities/common/grph/__init__.py capabilities/common/grph/models.py capabilities/common/grph/graph_runtime.py capabilities/common/grph/service.py capabilities/common/grph/api.py capabilities/common/grph/views.py capabilities/common/grph/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/grph/test_capability_contract.py capabilities/common/grph/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/grph --json
./.venv/bin/apg capabilities publish-plan capabilities/common/grph --json
```
