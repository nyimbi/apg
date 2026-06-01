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
  evidence, including pending-review state for unfamiliar content types and
  classifications.
- Document indexing with document id, title, body, classification, facets,
  metadata, source lineage, pending-review facet evidence, and index document
  counts.
- Bulk indexing with document-count checks, lineage checks, and Bytewax adapter
  evidence.
- Keyword, semantic, and hybrid query execution with RBAC filtering,
  restricted-content enforcement, embedding readiness checks, large-window
  review, query records, facets, and audit events.
- First-class AI search-agent composition for `codex`, `claude_code`,
  `opencode`, and `pi`, with role, scope, owner, purpose, contribution
  disclosure, and privileged-role review guardrails.
- Bytewax lifecycle batch validation for index, document, bulk indexing, query,
  facet, ranking, access-policy, and search-agent mutations.
- UI view models for dashboard, search, indices, documents, bulk indexing,
  facets, analytics, ranking, access review, governance, agents, lifecycle
  batches, audit, settings, and pending-review queues.
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
agent = service.register_search_agent(
	"agent-001",
	"tenant-a",
	"Search Steward",
	"codex",
	"search_steward",
	"index document query review",
	"search-owner",
	"govern search lifecycle changes",
)
batch = service.validate_srch_lifecycle_batch(
	"tenant-a",
	"bytewax",
	1,
	"search_agent_batch",
	"batch-001",
)
```

Review-required outcomes are persisted as data, not discarded exceptions:

```python
pending_index = service.create_index(
	tenant_id="tenant-a",
	name="custom-content",
	owner="search-owner",
	content_type="briefing",
	classification="internal",
)
assert pending_index["status"] == "pending_review"
assert pending_index["review_reasons"] == ["index_content_type_review_required"]
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
Review-required index, document, query, and privileged search-agent outcomes
are persisted as `pending_review` or `review_required` records with matched
rules and review reasons so generated applications can surface governance
queues without replaying indexing or retrieval work.
AI search-agent guardrails also block unsupported runtimes, unsupported roles,
missing scope, missing owner, missing purpose, missing machine-contribution
disclosure, and route privileged roles through pending human review when
approval evidence is absent. Lifecycle mutation batches are accepted only
through the declared Bytewax processor contract.

## AI Agent Composition

SRCH treats search-governance agents as first-class APG citizens. Generated
applications can compose agents from rapidly changing tool runtimes without
binding indexing, retrieval, ranking, or access-policy logic to a single
provider. The executable contract supports `codex`, `claude_code`, `opencode`,
and `pi`; roles include source curation, index review, document-quality review,
query-relevance review, ranking review, access-policy review, facet-taxonomy
review, lifecycle-batch review, and search stewardship.

The runtime stores provider-neutral agent metadata only. Live CLI/API
invocation, credentials, embedding-provider calls, vector database operations,
and remote agent orchestration belong behind AICR, NLPC, and search-provider
adapter boundaries.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/srch/__init__.py capabilities/common/srch/capability_contract.py capabilities/common/srch/models.py capabilities/common/srch/search_runtime.py capabilities/common/srch/service.py capabilities/common/srch/api.py capabilities/common/srch/views.py capabilities/common/srch/app.py capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/srch --json
./.venv/bin/apg capabilities publish-plan capabilities/common/srch --json
```
