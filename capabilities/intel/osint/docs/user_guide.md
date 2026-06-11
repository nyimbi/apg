# Open Source Intelligence

**Capability ID**: `intel_osint` | **Domain**: `intel` | **Version**: `2.1.0`

## Description

`intel_osint` is the APG package-backed capability for governed open-source intelligence applications. It composes requirements, sources, collection plans, evidence, triage, assessments, dissemination, reviews, Bytewax lifecycle metadata, UI/view models, visual theming, and provider-neutral AI-agent automation.

## Installation

```bash
pip install apg-intel-osint
```

## Provides

- `osint_source_workflow`
- `osint_collection_task_workflow`
- `osint_raw_intel_workflow`
- `osint_processed_intel_workflow`
- `osint_entity_workflow`
- `osint_pivot_search`
- `osint_bulk_ingest`
- `osint_entity_merge`
- `osint_watchlist_workflow`
- `osint_confidence_decay`
- `osint_task_retry`
- `osint_requirement_lifecycle`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `intel_crawler`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-osint/dashboard` | `intel_osint:view` | Overview |
| `/intel-osint/sources` | `intel_osint:sources` | Collection |
| `/intel-osint/tasks` | `intel_osint:tasks` | Collection |
| `/intel-osint/raw-intel` | `intel_osint:raw_intel` | Processing |
| `/intel-osint/triage` | `intel_osint:triage` | Processing |
| `/intel-osint/processed-intel` | `intel_osint:processed_intel` | Analysis |
| `/intel-osint/entities` | `intel_osint:entities` | Analysis |
| `/intel-osint/relationships` | `intel_osint:relationships` | Analysis |
| `/intel-osint/watchlist` | `intel_osint:watchlist` | Monitoring |

## Key Service Methods

### Source & Task Management

- `register_source()` — register a new intelligence source
- `update_source()` / `get_source()` / `list_sources()` / `delete_source()`
- `create_task()` — create a collection task against a registered source
- `start_task()` / `complete_task()` / `fail_task()` / `cancel_task()`
- `retry_task(task_id, max_retries, backoff_base_seconds)` — reschedule a failed task with exponential back-off

### Raw & Processed Intelligence

- `ingest_raw_intel()` — ingest a single raw intelligence item (fingerprint dedup enforced)
- `bulk_ingest_raw_intel(payloads, max_concurrency=50)` — concurrent batch ingestion with semaphore backpressure
- `triage_raw_intel()` — record triage decision on a raw item
- `create_processed_intel()` / `update_processed_intel()` / `list_processed_intel()`

### Entity & Relationship Graphs

- `extract_entity()` / `update_entity()` / `get_entity()` / `list_entities()` / `delete_entity()`
- `map_relationship()` / `update_relationship()` / `list_relationships()`
- `merge_entities(primary_id, secondary_ids, analyst_id, evidence_reference)` — consolidate duplicate entity records; requires analyst sign-off
- `relationship_mapping()` — full entity network report with cluster detection
- `duplicate_deduplication(similarity_threshold)` — identify merge candidates

### Pivot Search

```python
results = await svc.pivot_search(
    query="APT28",
    pivot_type="entity",   # entity | ip | domain | social | None (all)
    min_confidence=0.6,
    limit=20,
)
# results["results"] is a ranked list across all stores
```

### Intelligence Requirements

```python
req = await svc.create_requirement(
    tenant_id="acme",
    topic="Critical infrastructure threat monitoring",
    priority="high",
    requester_id="u-001",
    classification="confidential",
    evidence_reference="evidence://req-2026",
)
# ... collect and process intel ...
await svc.close_requirement(req["id"], resolution="satisfied", analyst_id="u-001")
```

### Watchlist Management

```python
entry = await svc.add_to_watchlist(
    reference_id=entity.id,
    reference_type="entity",
    reason="Known threat actor alias identified in raw feed",
    analyst_id="u-002",
    evidence_reference="evidence://watchlist-2026",
    alert_threshold=0.75,
)
# Check all active watchlist entries
entries = await svc.list_watchlist(reference_type="entity", active_only=True)
# Remove when no longer needed
await svc.remove_from_watchlist(entry["id"], analyst_id="u-002")
```

### Confidence Decay

Intelligence items age. Apply decay to keep scores current:

```python
report = await svc.apply_confidence_decay(
    decay_model="exponential",  # exponential | linear | step
    half_life_days=90,
    min_score=0.05,
)
print(report["updated_raw"], "raw items decayed")
print(report["updated_processed"], "processed items decayed")
```

### Bulk Ingestion

```python
from capabilities.intel.osint.models import RawIntelligenceCreate

payloads = [RawIntelligenceCreate(...) for item in feed_items]
result = await svc.bulk_ingest_raw_intel(payloads, max_concurrency=50)
print(result["succeeded"], "/", result["total"], "ingested")
print(result["duplicate_skipped"], "duplicates skipped")
```

### Reports

- `dashboard_summary()` — KPI counts for all stores
- `source_health_report()` — credibility and distribution by type/risk
- `threat_landscape_report()` — geographic distribution of threats

## Governance Rules

All write operations enforce policy via `evaluate_capability_rules()`:

| Rule | Trigger |
|------|---------|
| `tenant_context_required` | Missing or empty `tenant_id` |
| `terms_review_required` | Registering source without terms review reference |
| `collection_approval_required` | High/critical risk source task without approval |
| `dissemination_approval_required` | Dissemination without explicit approval reference |
| `human_approval_required` | Agent privileged action without human sign-off |
| `cross_tenant_access_denied` | Payload tenant does not match service tenant |
| `bytewax_event_stream_required` | Batch processing with non-Bytewax stream |

Invalid operations raise `PermissionError` with the rule identifier as the message.

## Interoperability

Reference this capability in `.apg` source files:

```apg
use intel_osint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_OSINT_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `INTEL_OSINT_DEFAULT_CONFIDENCE_DECAY_DAYS` | `90` | Half-life for exponential confidence decay |
| `INTEL_OSINT_BULK_INGEST_CONCURRENCY` | `50` | Default semaphore bound for bulk ingest |
| `INTEL_OSINT_MAX_TASK_RETRIES` | `3` | Maximum retry attempts for failed tasks |
| `OLLAMA_BASE_URL` | _(unset)_ | Enable ML entity extraction via local Ollama |

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Planned improvement roadmap
- `tests/` — Unit and integration tests
