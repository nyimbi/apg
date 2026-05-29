# Search Engine Capability Specification

- **Capability Name**: Search Engine
- **Capability ID**: `srch`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG runtime for `srch`.
It gives composed applications a deterministic tenant-aware search surface for
index creation, document indexing, bulk ingestion, embedding readiness,
keyword/semantic/hybrid queries, RBAC-filtered retrieval, facets, query
analytics, governance audit events, UI route metadata, semantic-model
publication, and publish-plan evidence.

## Provided Services

- `search_index_registry`
- `document_indexing`
- `bulk_indexing`
- `embedding_index_readiness`
- `governed_query_execution`
- `facet_aggregation`
- `query_analytics`
- `search_audit_events`

## Required Services

- `tenant_context`
- `source_lineage`
- `identity_rbac`
- `embedding_index_provider`
- `audit_sink`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `indexing_requires_owner`
- `restricted_query_requires_rbac_filter`
- `semantic_query_requires_embeddings`
- `large_result_window_requires_review`
- `bulk_index_requires_lineage`

## UI

The package exposes 7 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The view helpers provide dashboard, search console, index manager, document
indexer, analytics, governance, and settings models.

## Theme

The package uses the `srch_discovery_console` APG theme contract.

## Runtime Behavior

`SrchService` is intentionally dependency-light so it can run inside generated
applications, tests, and publish-plan probes without external infrastructure.
It supports:

- `create_index()` for tenant-scoped search indices with owner, content type,
  classification, source lineage, and embedding readiness metadata.
- `mark_embedding_index_ready()` for semantic/hybrid retrieval readiness.
- `index_document()` and `bulk_index_documents()` for lineage-gated content
  ingestion with facets and metadata.
- `query()` for keyword, semantic, and hybrid retrieval with deterministic
  policy checks for restricted content, RBAC filters, embedding readiness, and
  large result-window review.
- `facets()`, list helpers, and `dashboard_summary()` for API and UI
  composition.

## Adapter Boundaries

The in-package runtime stores records in memory by design. Production adapters
are expected to bind durable index stores, embedding services, tenant context,
identity/RBAC filters, source lineage, telemetry, and audit sinks at the APG
composition layer without changing the deterministic package contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/srch/__init__.py capabilities/common/srch/models.py capabilities/common/srch/search_runtime.py capabilities/common/srch/service.py capabilities/common/srch/api.py capabilities/common/srch/views.py capabilities/common/srch/capability_contract.py capabilities/common/srch/app.py capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/srch --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/srch --json`
