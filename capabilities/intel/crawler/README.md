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
- robots.txt compliance enforcement (strict / advisory / disabled modes).
- Content change detection with structural diffing and skip-on-similarity.
- Social media streaming ingest: Twitter, Reddit, Mastodon, Telegram, RSS.
- Multilingual language detection with Unicode block frequency analysis.
- Structured data extraction: JSON-LD, OpenGraph, Microdata.
- PII scrubbing with regex patterns for email, phone, national ID, IP, credit card.
- Source reputation index from extraction quality and validation confidence.
- Resumable crawl checkpointing for fault-tolerant deep crawls.
- Cross-source entity deduplication with fingerprint-blocking near-duplicate detection.
- Outbound webhook bus with HMAC-SHA256 signing for push notifications.
- Semantic near-duplicate report using fingerprint-prefix proximity.

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

# --- new async methods ---
import asyncio

async def run():
    svc = service  # same instance
    schedule = await svc.schedule_crawl("https://example.com", depth=3, frequency="daily", keywords=["AI"])
    robots  = await svc.check_robots_compliance("https://example.com/news", compliance_mode="strict")
    scrub   = await svc.scrub_pii(extraction["id"], "Contact nyimbi@gmail.com or +254712345678")
    rep     = await svc.compute_source_reputation(source["id"])
    ckpt    = await svc.create_crawl_checkpoint(job["id"], visited_urls=["https://example.com/1"], queued_urls=["https://example.com/2"])
    hook    = await svc.register_webhook("hook-1", "https://hooks.example.com/intel", events=["crawl_job_completed"], secret="s3cr3t")
    dedup   = await svc.cross_source_dedup()
    print(dedup)

asyncio.run(run())
```

## New Async Methods

| Method | Description |
|---|---|
| `check_robots_compliance(url, mode)` | Evaluate URL against robots.txt rules; modes: `strict`, `advisory`, `disabled` |
| `detect_content_changes(url, content)` | Diff new content against fingerprint registry; returns similarity and skip recommendation |
| `ingest_social_media(platform, items, source_id)` | Store social-media items (Twitter, Reddit, Mastodon, Telegram, RSS) |
| `detect_language(extraction_id, text)` | Unicode-block language detection; tags extraction record with ISO-639-1 code |
| `extract_structured_data(extraction_id, html)` | Parse JSON-LD, OpenGraph, and Microdata from raw HTML |
| `scrub_pii(extraction_id, text)` | Regex PII detection and placeholder substitution (email, phone, national ID, IP, CC) |
| `compute_source_reputation(source_id)` | Weighted reputation score from extraction quality + validation confidence + HTTPS ratio |
| `create_crawl_checkpoint(job_id, visited, queued)` | Persist resumable frontier checkpoint |
| `resume_from_checkpoint(job_id)` | Return latest checkpoint for frontier rebuild on failure |
| `cross_source_dedup(tenant_id)` | Near-duplicate detection across all sources using fingerprint blocking |
| `register_webhook(id, url, events, secret)` | Register HMAC-SHA256 signed push-notification webhook |
| `semantic_dedup_report(threshold)` | Near-duplicate report using fingerprint-prefix proximity as cosine-similarity proxy |

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

