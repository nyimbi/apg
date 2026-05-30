# APG SRCH - Search Engine

SRCH is the APG capability for governed enterprise search across tenant-scoped
indices, documents, facets, keyword retrieval, semantic retrieval, hybrid
retrieval, and query analytics. It lets generated applications create indices,
index documents, run access-filtered queries, review large result windows,
track denied queries, expose UI view models, and publish audit evidence through
deterministic guardrails.

## What It Provides

- Search index creation with tenant, name, owner, content type,
  classification, source lineage, embedding readiness, status, and audit
  evidence.
- Document indexing with document id, title, body, classification, facets,
  metadata, source lineage, and index document counts.
- Bulk indexing with document-count checks, lineage checks, and Bytewax adapter
  evidence.
- Keyword, semantic, and hybrid query execution with RBAC filtering,
  restricted-content enforcement, embedding readiness checks, large-window
  review, query records, facets, and audit events.
- UI view models for dashboard, search, indices, documents, bulk indexing,
  facets, analytics, ranking, access review, governance, audit, and settings.
- Adapter configuration for ETLP, META, NLPC, AICR, AUTH, AUDL, CACH, MONI,
  and Bytewax event streaming.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `service.py` - `SrchService`, the dependency-light generated-app runtime.
- `search_runtime.py` - deterministic record models and helper functions.
- `views.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.

## Generated-App Usage

```python
from capabilities.common.srch.service import SrchService

service = SrchService()
index = service.create_index(
	tenant_id="tenant-a",
	name="knowledge-base",
	owner="search-owner",
	content_type="article",
	classification="internal",
	source_lineage_ref="lineage://kb",
	embedding_index_ready=True,
)
service.index_document(
	tenant_id="tenant-a",
	index_id=index["id"],
	document_id="doc-1",
	title="APG search overview",
	body="APG provides keyword and semantic enterprise search.",
	facets={"module": "platform", "kind": "guide"},
	source_lineage_ref="lineage://kb/doc-1",
)
response = service.query(
	tenant_id="tenant-a",
	query_text="semantic search",
	index_ids=[index["id"]],
	query_type="hybrid",
	result_window=10,
	rbac_filter_applied=True,
)
```

## Guardrails

SRCH blocks missing tenant context, indices without name/owner/content
type/classification, restricted indices without lineage, documents without
index/id/title/body/classification/lineage, empty bulk batches, bulk indexing
without lineage, queries without text/index/type, restricted queries without
RBAC filtering, semantic or hybrid queries without embedding-ready indices,
non-positive result windows, cross-tenant search, non-Bytewax batch indexing,
index retirement without review, and state changes without audit evidence. SRCH
requires review for unknown content types, unknown classifications, large bulk
batches, unknown query types, large result windows, and unapproved facet keys.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/srch/__init__.py capabilities/common/srch/capability_contract.py capabilities/common/srch/models.py capabilities/common/srch/search_runtime.py capabilities/common/srch/service.py capabilities/common/srch/api.py capabilities/common/srch/views.py capabilities/common/srch/app.py capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/srch --json
./.venv/bin/apg capabilities publish-plan capabilities/common/srch --json
```
