# Product Information Management — User Guide

**APG Platform | Capability: pde_pim | Version 2.2.0 | Updated: June 2026**
**Datacraft | www.datacraft.co.ke**

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Product Catalogue Management](#product-catalogue-management)
4. [Attribute Management](#attribute-management)
5. [Digital Asset Management](#digital-asset-management)
6. [Content Localisation](#content-localisation)
7. [Category Taxonomy](#category-taxonomy)
8. [Data Quality Scoring](#data-quality-scoring)
9. [Channel Publishing and Syndication](#channel-publishing-and-syndication)
10. [Bulk Operations](#bulk-operations)
11. [Async Enrichment Pipelines](#async-enrichment-pipelines)
12. [Quality Remediation Plans](#quality-remediation-plans)
13. [PIM Analytics and KPIs](#pim-analytics-and-kpis)
14. [Product Lifecycle Management](#product-lifecycle-management)
15. [Compliance Management](#compliance-management)
16. [AI Agent Integration](#ai-agent-integration)
17. [API Reference Summary](#api-reference-summary)
18. [Troubleshooting](#troubleshooting)
19. [Frequently Asked Questions](#frequently-asked-questions)

---

## Overview

The PIM capability (`pde_pim`) is the authoritative record for every product in a tenant's catalogue. It governs the complete product data lifecycle:

- **Ingest**: Create products individually or via bulk import from CSV or ERP feeds
- **Enrich**: Add typed attributes, localised content, digital assets, and category assignments
- **Govern**: Score data quality, record compliance, and manage change requests with full audit trails
- **Publish**: Validate products against channel-specific quality gates and syndicate to web, marketplace, and B2B channels
- **Analyse**: Track catalogue health, enrichment coverage, and publication KPIs across time periods

All operations are multi-tenant isolated. Every write emits an audit event to the `apg.pde.pim.lifecycle` Bytewax stream.

---

## Getting Started

### Prerequisites

- APG Platform account with `pde_pim` capability enabled for your tenant
- Permission set: at minimum `pde_pim:view` for read operations, `pde_pim:manage_products` to create/update
- Python 3.11+ if using the service layer directly

### Python Setup

```python
from capabilities.pde.pim.service import ProductInfoManagementService

svc = ProductInfoManagementService(tenant_id="your-tenant-id")
```

Passing `tenant_id` at construction avoids repeating it on every call. You can still override per-call:

```python
svc = ProductInfoManagementService()
product = svc.create_product("SKU-001", "Name", "Category", {}, tenant_id="tenant-a")
```

### Tenant Context

Every operation requires a resolved tenant context. Operations without one raise `PermissionError("tenant_context_required")`. In production, the APG auth layer injects the tenant context automatically.

---

## Product Catalogue Management

### Creating a Catalogue

A catalogue groups products under a code/name. Products auto-create a default catalogue if none is specified.

```python
catalog = svc.create_catalog("cat-main", "tenant-a", "MAIN", "Main Catalogue", "owner-1")
```

Fields: `catalog_id`, `tenant_id`, `code` (uppercase), `name`, `owner_id`.

### Creating a Product

```python
product = svc.create_product(
    sku="SOLAR-20W-BLK",
    name="Solar Charger 20W Black",
    category="Electronics",
    attributes={"weight_g": 180, "panel_watt": 20, "colour": "black"},
    tenant_id="tenant-a",
    product_type="physical",   # physical | digital | service | bundle | component | raw_material
    owner_id="catalog_manager",
    catalog_id="cat-main",     # omit to use auto-default
)
```

SKUs must be unique within a tenant. Duplicate SKUs raise `ValueError("sku_already_exists:<sku>")`.

### Product Types

| Type | Description |
|------|-------------|
| `physical` | Tangible, shippable goods |
| `digital` | Downloads, software, licences |
| `service` | Non-shippable offerings |
| `bundle` | Grouped physical/digital products |
| `component` | Sub-assembly or spare part |
| `raw_material` | Input to manufacturing |

### Retrieving and Searching Products

```python
# Text search across name, SKU, and attribute values
results = svc.product_search("solar charger", tenant_id="tenant-a")

# With filters
results = svc.product_search(
    "solar",
    filters={"category": "Electronics", "status": "published", "channel": "web"},
    tenant_id="tenant-a",
    limit=50,
    offset=0,
)
```

### Product Lifecycle Stages

```python
svc.product_lifecycle("SOLAR-20W-BLK", "launch", tenant_id="tenant-a")
```

Valid stages: `concept → development → launch → growth → maturity → decline → discontinue`

---

## Attribute Management

### Defining Attributes

Define reusable attribute schemas at the tenant level:

```python
attr = svc.define_attribute(
    "attr-colour", "tenant-a", "colour", "Colour", "enum", "owner-1"
)
```

Supported types: `text`, `number`, `boolean`, `date`, `enum`, `money`, `media`, `rich_text`

### Setting Attribute Values

```python
updated = svc.update_attributes(
    "SOLAR-20W-BLK",
    {"colour": "black", "warranty_years": 2, "certifications": ["CE", "RoHS"]},
    tenant_id="tenant-a",
    updated_by="enrichment_agent",
)
```

`update_attributes` merges into the existing attribute dict. Pass the full replacement dict if you need a clean overwrite.

### Rich-Text Attributes

Rich-text attribute values require a locale. Setting one without a locale is denied at the rule engine level before persistence.

---

## Digital Asset Management

### Adding Media

```python
media = svc.add_media(
    sku="SOLAR-20W-BLK",
    media_type="image",          # image | video | document | 3d_model | audio | thumbnail
    url="https://cdn.example.com/solar-20w-hero.jpg",
    alt_text="Solar Charger 20W hero shot",
    tenant_id="tenant-a",
    rights_basis="owned",        # owned | licensed | creative_commons
    sort_order=0,
)
```

### Asset Pipeline Recommendations

For production environments, build an async pipeline around `add_media`:

1. Upload the asset to your CDN/object store and obtain the URL
2. Call `svc.add_media(...)` with the canonical CDN URL
3. Store the returned `media_id` for later reference or deletion

Data quality scoring awards points for each media record: full score at 3+ assets.

---

## Content Localisation

### Enriching Content for a Locale

```python
content = svc.enrich_content(
    content_id="content-en-001",
    tenant_id="tenant-a",
    product_id=product["id"],
    locale="en",
    title="Solar Charger 20W",
    body="Portable solar panel charger with USB-C output. 20W peak output.",
    generated=False,
    reviewed_by="content_reviewer",
)
```

### AI-Generated Content

Set `generated=True` for AI-generated copy. The rule engine holds AI-generated content at `require_review` status until a human reviewer is recorded. Without `reviewed_by`, the content is saved but cannot be published.

### Async Multi-Locale Localisation (v2.2.0)

```python
result = await svc.async_localise_product(
    "SOLAR-20W-BLK",
    localisations=[
        {"locale": "en", "title": "Solar Charger 20W", "description": "..."},
        {"locale": "fr", "title": "Chargeur Solaire 20W", "description": "..."},
        {"locale": "sw", "title": "Chaja ya Jua 20W", "description": "..."},
        {"locale": "ar", "title": "شاحن شمسي 20W", "description": "..."},
    ],
    tenant_id="tenant-a",
    reviewed_by="translation_manager",
)
print(result["coverage_pct"])  # e.g. 100.0
```

All locale coroutines fire concurrently. `coverage_pct` is the ratio of successful locales to total attempted.

---

## Category Taxonomy

### Assigning a Hierarchical Category Path

```python
assignment = svc.product_categorisation(
    sku="SOLAR-20W-BLK",
    category_path=["Electronics", "Energy", "Solar", "Chargers"],
    tenant_id="tenant-a",
    assigned_by="catalog_manager",
)
```

Category nodes are created automatically if they do not exist. The leaf category ID is stored on the assignment and on the product record.

### Bulk Classification

```python
result = svc.bulk_classify(
    skus=["SOLAR-20W-BLK", "SOLAR-10W-WHT", "SOLAR-30W-BLK"],
    category_path=["Electronics", "Energy", "Solar", "Chargers"],
    tenant_id="tenant-a",
)
print(result["classified"], result["failed"])
```

### Managing Taxonomy Nodes

```python
# Create a standalone taxonomy node (without product assignment)
node = svc.taxonomy_manage(
    tenant_id="tenant-a",
    category_name="Accessories",
    parent_id="cat-tenant-a-electronics",
)
```

---

## Data Quality Scoring

### Computing a Score

```python
score = svc.data_quality_score("SOLAR-20W-BLK", tenant_id="tenant-a")
```

Returns a weighted total score (0–100) and per-dimension breakdown:

| Dimension | Weight | Full Score Criteria |
|-----------|--------|---------------------|
| name | 20% | Product name is non-empty |
| attributes | 20% | 5+ attributes present |
| description | 15% | At least one content record exists |
| media | 15% | 3+ media assets attached |
| categorisation | 15% | Category path assigned |
| compliance | 10% | At least one compliance record present |
| channel_listing | 5% | Listed on at least one channel |

Grade: A ≥ 90, B ≥ 75, C ≥ 60, D ≥ 40, F < 40.

### Channel Validation

Before publishing, validate that a product meets the channel's minimum quality threshold:

```python
validation = svc.channel_validate("SOLAR-20W-BLK", "marketplace_amazon", tenant_id="tenant-a")
# {"valid": True/False, "quality_score": 87.5, "min_required": 85, "issues": [...]}
```

Channel quality minimums:

| Channel | Minimum Score |
|---------|--------------|
| `web` | 70 |
| `mobile` | 60 |
| `marketplace_amazon` | 85 |
| `b2b_portal` | 75 |
| other | 70 (default) |

---

## Channel Publishing and Syndication

### Publish to a Single Channel

```python
pub = svc.publish_to_channel(
    sku="SOLAR-20W-BLK",
    channel_id="web",
    tenant_id="tenant-a",
    approved_by="catalog_manager",
)
```

### Publish to Multiple Marketplaces (Synchronous)

```python
result = svc.syndicate_marketplace(
    "SOLAR-20W-BLK",
    marketplaces=["marketplace_amazon", "web", "mobile"],
    tenant_id="tenant-a",
    approved_by="catalog_manager",
)
```

### Async Parallel Syndication (v2.2.0)

Uses quality-gated publish per channel, fired concurrently:

```python
result = await svc.async_syndicate_marketplaces(
    "SOLAR-20W-BLK",
    marketplaces=["marketplace_amazon", "web", "mobile", "b2b_portal"],
    tenant_id="tenant-a",
    approved_by="catalog_manager",
)
print(result["published"], result["failed"])
```

Channels that fail the quality gate surface in `results` with `status: "failed"` — they do not block other channels.

### Scheduling Future Publication

```python
schedule = svc.publication_schedule(
    sku="SOLAR-20W-BLK",
    channel_id="web",
    publish_at="2026-07-01T09:00:00Z",
    tenant_id="tenant-a",
    approved_by="catalog_manager",
)
```

### Unpublishing

```python
svc.unpublish(
    sku="SOLAR-20W-BLK",
    channel_id="web",
    tenant_id="tenant-a",
    reason="seasonal_delisting",
    unpublished_by="catalog_manager",
)
```

---

## Bulk Operations

### Bulk Import from CSV / ERP

```python
rows = [
    {"sku": "P001", "name": "Widget A", "category": "Widgets", "colour": "red"},
    {"sku": "P002", "name": "Widget B", "category": "Widgets", "colour": "blue"},
]
log = svc.bulk_import(rows, tenant_id="tenant-a", owner_id="import_agent")
print(log["created_count"], log["skipped_count"], log["failed_count"])
```

Rows missing `sku` or `name` are recorded as failures and do not block the rest of the batch.

### Import from ERP (domain alias)

```python
log = svc.import_from_erp(erp_rows, tenant_id="tenant-a", owner_id="erp_system")
```

Each row gets `source: "erp"` appended automatically.

---

## Async Enrichment Pipelines

### Concurrent Bulk Attribute Enrichment (v2.2.0)

When processing large catalogues, sequential attribute updates become a bottleneck. Use `async_bulk_enrich` to fire all updates concurrently under a semaphore:

```python
import asyncio

tasks = [
    {"sku": "P001", "attributes": {"colour": "red", "weight_g": 50}},
    {"sku": "P002", "attributes": {"colour": "blue", "weight_g": 65}},
    # ... hundreds of rows
]
result = await svc.async_bulk_enrich(
    tasks,
    tenant_id="tenant-a",
    updated_by="enrichment_pipeline",
    concurrency=20,  # max 20 concurrent updates
)
print(result["succeeded"], result["failed"])
```

`concurrency` defaults to 10. Tune based on your database connection pool size.

### Async Product Creation

```python
product = await svc.async_create_product(
    "SKU-003", "Product C", "Category", {"attr": "val"},
    tenant_id="tenant-a",
)
```

---

## Quality Remediation Plans

The `async_quality_remediation_plan` method returns an actionable, priority-sorted work queue for enrichment agents or human operators:

```python
plan = await svc.async_quality_remediation_plan("SOLAR-20W-BLK", tenant_id="tenant-a")
```

Example response:

```json
{
  "sku": "SOLAR-20W-BLK",
  "total_score": 62.5,
  "grade": "C",
  "plan_items": 4,
  "plan": [
    {
      "priority": 1,
      "dimension": "description",
      "current_score": 0.0,
      "score_gain_if_fixed": 15.0,
      "action": "Write or generate locale-scoped title and body content via enrich_content.",
      "effort": "medium"
    },
    {
      "priority": 2,
      "dimension": "media",
      "current_score": 33.3,
      "score_gain_if_fixed": 10.0,
      "action": "Attach at least 3 media assets: primary image, lifestyle image, and specification sheet.",
      "effort": "medium"
    }
  ]
}
```

AI agents operating within the `inspect_prepare_and_recommend` scope ceiling can consume this plan directly and queue enrichment tasks.

---

## PIM Analytics and KPIs

### Full Analytics Payload

```python
analytics = svc.pim_analytics("2026-Q2", tenant_id="tenant-a")
```

Returns catalogue counts, publication rates, average data quality score, and import activity.

### KPI Summary

```python
kpis = svc.pim_kpi_summary("2026-Q2", tenant_id="tenant-a")
```

Thin wrapper that adds `kpi_summary: True` flag for dashboard routing.

### Dashboard Summary

```python
summary = svc.dashboard_summary("tenant-a")
```

Lightweight count aggregation across all stores — suitable for dashboard card population without computing quality scores.

---

## Product Lifecycle Management

### Variant Creation

```python
variant = svc.product_variant_create(
    sku="SOLAR-20W-BLK",
    variant_attrs={"colour": "white", "sku_suffix": "WHT"},
    tenant_id="tenant-a",
    owner_id="catalog_manager",
)
```

Variants are full product records with a parent relationship stored in `self.variants`. Variant SKU is auto-generated as `{parent_sku}-VAR-{n:03d}`.

### Enrichment Workflows

```python
workflow = svc.enrichment_workflow(
    sku="SOLAR-20W-BLK",
    workflow_steps=["translate_fr", "translate_sw", "add_images", "compliance_check"],
    tenant_id="tenant-a",
    assigned_to="content_team",
)
```

### Version Comparison

```python
diff = svc.version_compare("SOLAR-20W-BLK", tenant_id="tenant-a")
# Returns current version, change count, and latest change record
```

---

## Compliance Management

### Recording Compliance

```python
issue = svc.record_quality_issue(
    issue_id="qi-001",
    tenant_id="tenant-a",
    product_id=product["id"],
    severity="high",           # low | medium | high | critical
    description="Missing CE marking documentation",
    owner_id="compliance_officer",
)
```

High and critical severity issues require an `owner_id` — the rule engine denies without one.

### Registering PIM Agents

```python
agent = svc.register_pim_agent(
    tenant_id="tenant-a",
    name="Enrichment Bot",
    runtime="claude_code",     # codex | claude_code | opencode | pi
    role="enrichment",         # catalog | data_quality | enrichment | channel | compliance | product_query
    purpose="Auto-generate product descriptions from attributes",
    owner_id="platform_admin",
)
```

Agent scope is hard-capped at `inspect_prepare_and_recommend` regardless of runtime. Privileged actions require human approval recorded in the rule engine.

---

## API Reference Summary

All routes are under prefix `/pde/pim/api/v1`.

| Endpoint | Method | Permission | Description |
|----------|--------|------------|-------------|
| `/pde/pim/dashboard` | GET | `pde_pim:view` | Dashboard summary |
| `/pde/pim/products` | GET/POST | `pde_pim:manage_products` | List / create products |
| `/pde/pim/products/<sku>/attributes` | PATCH | `pde_pim:manage_products` | Update attributes |
| `/pde/pim/products/<sku>/media` | POST | `pde_pim:manage_assets` | Add media |
| `/pde/pim/products/<sku>/category` | PUT | `pde_pim:manage_products` | Set category path |
| `/pde/pim/products/<sku>/quality` | GET | `pde_pim:view` | Data quality score |
| `/pde/pim/products/<sku>/quality/plan` | GET | `pde_pim:view` | Remediation plan |
| `/pde/pim/products/<sku>/publish` | POST | `pde_pim:publish` | Publish to channel |
| `/pde/pim/products/<sku>/unpublish` | POST | `pde_pim:publish` | Unpublish from channel |
| `/pde/pim/products/<sku>/localise` | POST | `pde_pim:manage_content` | Multi-locale content |
| `/pde/pim/products/<sku>/syndicate` | POST | `pde_pim:publish` | Syndicate to marketplaces |
| `/pde/pim/products/bulk-import` | POST | `pde_pim:manage_products` | Bulk CSV import |
| `/pde/pim/products/bulk-enrich` | POST | `pde_pim:manage_products` | Async bulk enrichment |
| `/pde/pim/catalogs` | GET/POST | `pde_pim:manage_catalogs` | Catalogue CRUD |
| `/pde/pim/attributes` | GET/POST | `pde_pim:manage_attributes` | Attribute schema |
| `/pde/pim/content` | GET/POST | `pde_pim:manage_content` | Content records |
| `/pde/pim/assets` | GET/POST | `pde_pim:manage_assets` | Digital assets |
| `/pde/pim/compliance` | GET/POST | `pde_pim:govern` | Compliance records |
| `/pde/pim/channels` | GET/POST | `pde_pim:publish` | Channel listings |
| `/pde/pim/quality` | GET/POST | `pde_pim:quality` | Quality issues |
| `/pde/pim/agents` | GET/POST | `pde_pim:agent_manage` | PIM agents |
| `/pde/pim/analytics` | GET | `pde_pim:view` | Analytics payload |
| `/pde/pim/audit` | GET | `pde_pim:govern` | Audit event stream |

---

## Troubleshooting

### `PermissionError: tenant_context_required`

Every operation requires a tenant ID. Pass `tenant_id` at construction or on each call:

```python
svc = ProductInfoManagementService(tenant_id="your-tenant")
```

### `ValueError: sku_already_exists:<sku>`

SKUs must be unique within a tenant. Use `product_search` to find the existing record before retrying with a different SKU.

### `PIMRecordNotFoundError: product_not_found_for_sku:<sku>`

The SKU does not exist for the resolved tenant. Check for typos, tenant mismatch, or that the product was not deleted.

### `ValueError: unsupported_channel:<channel>`

The channel is not in the `SUPPORTED_CHANNELS` contract list. Check `svc.describe()["channels"]["supported_channels"]` for the allowlist.

### `ValueError: channel_quality_gate_failed:<channel>:score=<n>:min=<m>`

Raised by `async_publish_to_channel` when the product's quality score is below the channel minimum. Run `async_quality_remediation_plan` to identify the fastest path to the required score.

### Quality Score Unexpectedly Low

Run the remediation plan:

```python
plan = asyncio.run(svc.async_quality_remediation_plan(sku, tenant_id="tenant-a"))
for item in plan["plan"]:
    print(f"[{item['effort']}] {item['action']} (+{item['score_gain_if_fixed']} pts)")
```

### Audit Events Missing

Audit events are in-memory in the development service. In production, verify the Bytewax stream is running and the `apg.pde.pim.lifecycle` topic is consuming events.

---

## Frequently Asked Questions

**Q: Can I use PIM without setting up a catalogue first?**
A: Yes. If `catalog_id` is omitted from `create_product`, a default catalogue (`default-catalog-{tenant}`) is created automatically.

**Q: What happens to publications when I update attributes?**
A: Existing publication records are not automatically revoked on attribute updates. If the update changes data that affects channel compliance, re-run `channel_validate` and re-publish if needed.

**Q: Can I have multiple category assignments for a product?**
A: Yes. Each call to `product_categorisation` creates a new assignment record. The product's `category` field is updated to the most recent assignment's path string.

**Q: How does the async enrichment concurrency work?**
A: `async_bulk_enrich` uses an `asyncio.Semaphore(concurrency)` to cap simultaneous operations. With an in-memory store this is mostly about maintaining correct async patterns; in a PostgreSQL-backed deployment, set `concurrency` to your connection pool size minus headroom for other operations.

**Q: Are async methods thread-safe?**
A: The service stores are plain Python dicts, which are not thread-safe. Run the service within a single-threaded async event loop (FastAPI, aiohttp) or add explicit locking for multi-threaded deployments.

**Q: Can AI agents publish products autonomously?**
A: No. The `max_autonomous_scope` ceiling is `inspect_prepare_and_recommend`. Publishing is a privileged operation that requires a human-recorded `approved_by` value. An agent can prepare the publication request and queue it for human approval, but cannot self-approve.

**Q: How do I integrate with a real database?**
A: Replace the in-memory dict stores with async SQLAlchemy sessions. The service method signatures remain identical; only the persistence layer changes. See `WORLD_CLASS_IMPROVEMENTS.md` improvement #2 for the migration path.

**Q: What is the Bytewax event stream used for?**
A: All state-changing operations emit events to `apg.pde.pim.lifecycle`. Downstream capabilities (commerce, pricing, analytics, translation) subscribe to this stream rather than polling PIM directly. The stream processor must be Bytewax; routing to a queue is denied by the rule engine.

---

*Maintained by the APG Development Team — Datacraft*
*For corrections or feature requests, submit via the APG Platform feedback system or email nyimbi@gmail.com*

**Document Version**: 2.2.0
**Last Updated**: June 2026
**Next Review**: September 2026
