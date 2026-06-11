# SCRP User Guide

**Capability ID**: `scrp` | **Domain**: `common` | **Version**: `1.1.0`
**Company**: Datacraft | **Author**: Nyimbi Odero | **Copyright**: © 2025

---

## Introduction

SCRP provides governed data harvesting, web scraping, screen capture, OCR,
robotic process automation, and LLM-assisted extraction in a single
tenant-scoped service. Every operation is audited, rate-limited, and subject
to the capability's deterministic rule engine.

---

## Installation

```bash
pip install apg-common-scrp
```

---

## Architecture Overview

```
+---------------------------------------------+
|        ScraperDataHarvestingService         |
|                                             |
|  Source Registry     Extractor Profiles     |
|  Harvest Jobs        Harvest Runs           |
|  Result Batches      Pipeline Handoffs      |
|  Scheduled Tasks     Rate Limits            |
|  Proxy Pool          CAPTCHA Records        |
|  -- Async Layer --------------------------  |
|  Screen Capture      OCR Engine             |
|  RPA Workflows       LLM Extraction         |
|  Cursor Manager      Webhook Dispatch       |
|  Content Diff        Quality Reports        |
+--------------------+------------------------+
                     |
          +----------+---------+
          |   Rule Engine      |
          |   Audit Trail      |
          +--------------------+
```

---

## Quick Start

```python
from capabilities.common.scrp.service import ScrpService

svc = ScrpService()
TENANT = "my-tenant"

source = svc.register_source(
    tenant_id=TENANT,
    name="product-catalogue",
    source_type="web",
    endpoint="https://shop.example.com/products",
    owner="data-eng",
    terms_evidence="tos-accepted:2024-01-15",
    credential_vault_ref="vault://scrp/shop",
    rate_limit_per_minute=30,
    robots_policy_attached=True,
    pii_expected=False,
)

extractor = svc.create_extractor_profile(
    tenant_id=TENANT,
    name="product-html",
    extractor_type="css_selector",
    owner="data-eng",
    schema={"title": ".product-title", "price": ".price", "sku": "[data-sku]"},
    incremental_cursor_field="last_modified",
)

job = svc.create_harvest_job(
    tenant_id=TENANT,
    name="products-daily",
    source_id=source["id"],
    extractor_profile_id=extractor["id"],
    owner="data-eng",
    mode="incremental",
    pipeline_target="etlp:products",
)

run = svc.run_harvest(TENANT, job["id"], "scheduler")
completed = svc.complete_harvest_run(
    tenant_id=TENANT,
    run_id=run["id"],
    records_extracted=150,
    dlp_scanned=True,
    storage_ref="s3://data-lake/products/2024-12-01",
)
```

---

## Screen Capture

`capture_screen` renders a URL in a headless browser and saves a screenshot.
In production this calls Playwright; the current implementation is a stub
returning a synthetic record.

```python
import asyncio

async def main():
    capture = await svc.capture_screen(
        tenant_id=TENANT,
        url="https://shop.example.com/products",
        output_format="png",
        full_page=True,
        viewport_width=1920,
        viewport_height=1080,
        wait_ms=2000,
        owner="analyst",
    )
    # capture["storage_ref"] -> "memory://<tenant>/captures/<id>.png"

asyncio.run(main())
```

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `url` | required | Page URL |
| `output_format` | `"png"` | `"png"` or `"jpeg"` |
| `full_page` | `True` | Capture full scrollable page |
| `viewport_width` | `1280` | Browser viewport width px |
| `viewport_height` | `800` | Browser viewport height px |
| `wait_ms` | `1000` | Post-load wait before capture |

---

## OCR

### Text Extraction

```python
ocr = await svc.ocr_image(
    tenant_id=TENANT,
    image_ref="memory://tenant/captures/abc.png",
    language="eng",
    model="tesseract",         # or "llava", "moondream", "minicpm-v"
    confidence_threshold=0.75,
    owner="analyst",
)
print(ocr["full_text"])
print(ocr["blocks"])           # list of {text, confidence, bbox}
```

### Table Extraction

```python
table = await svc.ocr_extract_table(
    tenant_id=TENANT,
    image_ref="memory://tenant/captures/report.png",
    header_row=True,
    model="tesseract",
    owner="analyst",
)
print(table["headers"])        # ["Column A", "Column B", ...]
print(table["rows"])           # [["val1", "val2"], ...]
```

**Supported OCR Models**

| Model | Notes |
|-------|-------|
| `tesseract` | Fast, offline, good for clean text |
| `llava` | Strong on complex layouts |
| `moondream` | Lightweight vision model |
| `minicpm-v` | Balanced accuracy / speed |
| `llama3.2-vision` | Best accuracy for dense documents |

---

## RPA Workflows

`rpa_workflow_run` executes a sequence of browser actions via Playwright.

```python
result = await svc.rpa_workflow_run(
    tenant_id=TENANT,
    workflow_id="extract-invoice",
    target_url="https://portal.example.com/invoices",
    steps=[
        {"action": "navigate", "value": "https://portal.example.com/login"},
        {"action": "type", "selector": "#email", "value": "user@example.com"},
        {"action": "type", "selector": "#pass", "value": "s3cr3t"},
        {"action": "click", "selector": "button.login"},
        {"action": "wait_for", "selector": ".invoice-list"},
        {"action": "scroll", "selector": ".invoice-list", "value": "bottom"},
        {"action": "extract", "selector": "table.invoices"},
        {"action": "screenshot"},
    ],
    owner="rpa-bot",
    max_retries=2,
    timeout_ms=60000,
)
print(result["status"])          # "completed"
print(result["steps_completed"]) # 8
```

**Supported RPA Actions**

| Action | Selector | Value | Description |
|--------|----------|-------|-------------|
| `navigate` | optional | URL | Go to URL |
| `click` | required | — | Click element |
| `type` | required | text | Type text into element |
| `select` | required | option value | Select dropdown option |
| `wait_for` | required | — | Wait for element to appear |
| `extract` | required | — | Extract element text/HTML |
| `scroll` | required | `"top"` / `"bottom"` | Scroll element |
| `hover` | required | — | Hover over element |
| `screenshot` | optional | — | Take screenshot at step |

---

## LLM-Assisted Extraction

`llm_extract` sends content to an Ollama-hosted model with a structured schema
prompt. Results are cached by content SHA-256 hash.

```python
result = await svc.llm_extract(
    tenant_id=TENANT,
    content="Quarterly Results: Revenue of $4.2B, up 18% YoY. Headcount: 12,400.",
    schema={
        "revenue_usd_b": {"type": "float", "description": "Revenue USD billions"},
        "revenue_growth_pct": {"type": "float", "description": "YoY growth %"},
        "headcount": {"type": "int", "description": "Total employees"},
    },
    model="mistral-nemo",
    cache_results=True,
    source_url="https://ir.example.com/q4-2024",
)
print(result["extracted"])
print(result["cache_hit"])       # True on second identical call
```

---

## Incremental Cursor Management

```python
# Read current cursor (None if never advanced)
cursor = await svc.cursor_read(tenant_id=TENANT, job_id=job["id"])
print(cursor["cursor_value"])

# After a successful harvest run
await svc.cursor_advance(
    tenant_id=TENANT,
    job_id=job["id"],
    new_cursor_value="2024-12-01T12:00:00Z",
    run_id=completed["id"],
)

# Force full re-harvest (schema change, data loss)
await svc.cursor_reset(
    tenant_id=TENANT,
    job_id=job["id"],
    reason="schema_migration_v3",
    reset_by="data-eng",
)
```

---

## Pipeline Handoff Dispatch

```python
handoffs = svc.list_handoffs(TENANT)
for h in handoffs:
    dispatch = await svc.handoff_dispatch(
        tenant_id=TENANT,
        handoff_id=h["id"],
        dispatched_by="scheduler",
        max_retries=3,
        timeout_ms=5000,
    )
    print(dispatch["status"])    # "delivered" or "dead_lettered"

# Retry dead-lettered handoffs
dead = [h for h in handoffs if h["status"] == "dead_lettered"]
for h in dead:
    await svc.handoff_retry(
        tenant_id=TENANT, handoff_id=h["id"], retried_by="ops-team",
    )
```

---

## Content Diff and Change Alerting

```python
diff = await svc.content_diff(
    tenant_id=TENANT,
    source_id=source["id"],
    old_snapshot_ref="memory://snaps/2024-11-30",
    new_snapshot_ref="memory://snaps/2024-12-01",
    diff_format="json_delta",      # or "unified"
    change_threshold_pct=5.0,      # alert if >5% of lines change
    owner="monitor",
)
print(diff["change_pct"])          # e.g. 6.0
print(diff["significant_change"]) # True -> audit warning emitted
print(diff["diff"])                # {"added": 4, "removed": 2, "unchanged": 98}
```

---

## Data Quality Reporting

```python
extraction = svc.extract_structured_data(
    raw_html="<div class='price'>$19.99</div>",
    extraction_schema={"price": {"selector": "price", "type": "text"}},
    tenant_id=TENANT,
    source_url="https://shop.example.com/item/1",
)

report = await svc.quality_report(
    tenant_id=TENANT,
    extraction_id=extraction["extraction_id"],
    owner="analyst",
)
print(report["quality_grade"])          # "A", "B", or "C"
print(report["overall_completeness"])   # 0.0 - 1.0
print(report["field_reports"])          # per-field breakdown
```

**Quality Grades**

| Grade | Completeness | Meaning |
|-------|-------------|---------|
| A | >= 90% | Production-ready |
| B | 70-89% | Review recommended |
| C | < 70% | Remediation required |

---

## Rate Limiting and Proxy Rotation

```python
# Set domain rate limit
svc.rate_limit_management(
    domain="shop.example.com",
    requests_per_minute=20,
    tenant_id=TENANT,
    burst_limit=40,
    backoff_seconds=60,
)

# Rotate proxy for a request
proxy = svc.proxy_rotation(
    request_id="req-001",
    tenant_id=TENANT,
    strategy="round_robin",
    required_country="US",
)
```

---

## CAPTCHA Handling

```python
result = svc.captcha_handling(
    page_url="https://shop.example.com/checkout",
    captcha_type="recaptcha_v2",
    tenant_id=TENANT,
    solver_type="third_party",
)
print(result["solved"])
```

---

## Analytics

```python
analytics = svc.scraping_analytics(period="last_7_days", tenant_id=TENANT)
print(analytics["run_success_rate"])
print(analytics["captcha_solve_rate"])
print(analytics["js_render_count"])
```

---

## Guardrail Reference

| Condition | Decision |
|-----------|----------|
| Missing tenant context | deny |
| Source missing owner / terms / vault | deny |
| PII expected without policy | deny |
| Sensitive source not reviewed | require_review |
| Extractor missing schema | deny |
| Harvest run PII + no DLP scan | deny |
| AI agent not registered | deny |
| State change missing reason | deny |
| Cross-tenant access attempt | deny |
| Cursor reset missing reason | ValueError |
| RPA unsupported action | ValueError |
| Content diff unsupported format | ValueError |

---

## Composition

```apg
use scrp;
```

**Hard dependencies**: `conn`, `etlp`, `auth`
**Optional adapters**: `i18n`, `nlpc`, `schd`, `dlpd`, `bytewax`, `audl`, `moni`

---

## Further Reading

- `service.py` - Business logic and async methods
- `models.py` - Data models
- `api.py` - REST API endpoints
- `views.py` - Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` - 15 planned improvements
- `SPECIFICATION.md` - Full capability specification
