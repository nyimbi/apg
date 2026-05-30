# Intelligence Crawler

`intel_crawler` is the APG capability for composing governed source-collection applications. It wraps source registration, crawl jobs, extraction quality, dataset publication, validation, RAG preparation, graph projection, and crawler-agent review in an executable, dependency-light package surface.

## What It Provides

- Source registry with owner, source type, URLs, allowed domains, and policy review.
- Crawl-job lifecycle with cadence, maximum depth, rate limit, and high-risk approval.
- Extraction records with schema, content fingerprint, and quality score.
- Dataset publication with lineage, validation, and privacy review.
- Validation sessions with reviewer and confidence controls.
- RAG preparation with chunk strategy and embedding model.
- Knowledge-graph projection with entity schema and relationship evidence.
- First-class crawler agents for Codex, Claude Code, OpenCode, and Pi.
- Deterministic rules for operational guardrails.
- Bytewax lifecycle stream metadata.
- UI route and theme metadata for APG composition.

## Quick Start

```python
from capabilities.intel.crawler import IntelligenceCrawlerService

service = IntelligenceCrawlerService()
source = service.register_source(
    "news-feed",
    "tenant-a",
    "News Feed",
    "intel-team",
    "news",
    ["https://example.com/feed"],
    ["example.com"],
    policy_reviewed_by="policy-1",
)
job = service.create_crawl_job("job-1", "tenant-a", source["id"], "hourly", 2, 30)
service.complete_crawl_job("tenant-a", job["id"], fetched_count=12)
extraction = service.record_extraction(
    "ext-1",
    "tenant-a",
    job["id"],
    "article_v1",
    "clean article body",
    0.92,
)
validation = service.open_validation_session("val-1", "tenant-a", extraction["id"], "reviewer-1")
service.complete_validation_session("tenant-a", validation["id"], 0.91, "approve")
dataset = service.publish_dataset("dataset-1", "tenant-a", extraction["id"], validation_recorded=True)
service.record_rag_plan("rag-1", "tenant-a", dataset["id"], "heading-aware", 1200, "text-embedding")
summary = service.dashboard_summary("tenant-a")
```

## Contract

Use `get_capability_contract()` to inspect the APG composition surface.

```python
from capabilities.intel.crawler import get_capability_contract

contract = get_capability_contract("tenant-a")
print(contract["provides"])
print(contract["streaming"]["processor"])
```

## Guardrails

The rule engine blocks or routes review for:

- Missing tenant context.
- Writes without policy attachment.
- Sources without owner, URLs, allowed domains, or crawl-policy review.
- Crawl jobs without source or cadence.
- Non-positive crawl rate limits.
- Crawl depth above the configured limit.
- High-risk crawl jobs without approval.
- Extractions without schema, fingerprint, or sufficient quality.
- Datasets without lineage or validation.
- PII datasets without privacy review.
- Validation without reviewer or sufficient confidence.
- RAG plans without chunk plan, accepted chunk size, or embedding model.
- Graph projections without entity schema or relationship evidence.
- Batch operations and lifecycle events not routed through Bytewax.
- Unsupported crawler-agent runtime or role.
- Privileged agent actions without human approval.

## UI And Theme

The capability publishes route metadata for:

- `/intel-crawler/dashboard`
- `/intel-crawler/sources`
- `/intel-crawler/crawl-jobs`
- `/intel-crawler/extractions`
- `/intel-crawler/datasets`
- `/intel-crawler/validation`
- `/intel-crawler/rag`
- `/intel-crawler/graph`
- `/intel-crawler/agents`
- `/intel-crawler/settings`

The default theme is `intel_crawler_control`.

## AI Agents

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `source_strategy_reviewer`
- `crawl_policy_reviewer`
- `extraction_quality_reviewer`
- `validation_reviewer`
- `rag_pipeline_reviewer`
- `risk_reviewer`

Register an agent with `register_crawler_agent()` and validate privileged proposals with `validate_agent_crawler_action()`.

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/intel/crawler/__init__.py \
  capabilities/intel/crawler/capability_contract.py \
  capabilities/intel/crawler/service.py \
  capabilities/intel/crawler/api.py \
  capabilities/intel/crawler/views.py \
  capabilities/intel/crawler/app.py \
  capabilities/intel/crawler/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/intel/crawler/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/crawler/app.py
```

