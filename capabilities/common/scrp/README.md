# SCRP - Screen Processing / Scraper & Data Harvesting

SCRP is the APG capability for governed data-source harvesting, screen capture,
OCR, and UI automation. It lets an APG application register tenant-owned
sources, define extractors, schedule harvest jobs, run guarded harvest
lifecycles, record result batches, hand results to ETL pipelines, register
scoped AI harvest agents, expose composable UI models, and audit every
important lifecycle transition.

## What SCRP Provides

- Source registry with endpoint, type, owner, terms evidence, credential vault
  reference, rate limits, robots/terms policy, PII policy, sensitivity review,
  and tags.
- Extractor profiles with parser type, schema, output mapping, and incremental
  cursor metadata.
- Harvest jobs that bind source, extractor, mode, schedule policy, enabled
  state, and downstream pipeline target.
- Harvest runs with requested actor, status, extracted record count, errors,
  DLP status, violations, logs, and timestamps.
- Result batches with schema validity, retention hint, storage reference, and
  pipeline handoff records.
- First-class AI harvest agents for Codex, Claude Code, OpenCode, Pi, and
  compatible runtime adapters.
- Deterministic rule engine for tenant context, source ownership, terms
  evidence, credentials, rate limits, extractor schemas, DLP scans, sensitive
  source review, AI-agent governance, state-change reasons, audit evidence,
  and Bytewax batch mutation streams.
- **Screen capture** via `capture_screen` (PNG/JPEG, full-page, configurable
  viewport).
- **OCR** via `ocr_image` (Tesseract, Ollama vision models) and
  `ocr_extract_table` for tabular data.
- **RPA workflows** via `rpa_workflow_run` (navigate, click, type, select,
  extract, screenshot, scroll, hover).
- **LLM-based extraction** via `llm_extract` (Ollama-hosted models, content
  hash cache).
- **Incremental cursor management** via `cursor_read`, `cursor_advance`,
  `cursor_reset`.
- **Webhook dispatch** via `handoff_dispatch` and `handoff_retry` with
  dead-letter support.
- **Content diff** via `content_diff` (unified or JSON delta, threshold-based
  alerting).
- **Data quality reporting** via `quality_report` (per-field completeness,
  type validity, format conformance).
- View models for dashboard, source registry, job monitor, extractor
  workbench, pipeline handoff, compliance review, results, harvest agents,
  audit trail, analytics, and settings.

## How To Use It

Create the service and operate the lifecycle in order:

```python
from capabilities.common.scrp.service import ScrpService

service = ScrpService()
tenant_id = "tenant-demo"

source = service.register_source(
    tenant_id=tenant_id,
    name="orders-api",
    source_type="api",
    endpoint="https://example.invalid/orders",
    owner="data-owner",
    terms_evidence="contract:orders:v1",
    credential_vault_ref="vault://orders-api",
    rate_limit_per_minute=60,
    pii_expected=True,
    pii_policy_attached=True,
)

extractor = service.create_extractor_profile(
    tenant_id=tenant_id,
    name="orders-json",
    extractor_type="json",
    owner="data-owner",
    schema={"order_id": "str"},
)

agent = service.register_harvest_agent(
    tenant_id=tenant_id,
    agent_id="codex-orders-reviewer",
    name="Codex Orders Reviewer",
    runtime="codex",
    role="source_reviewer",
    scope="Review orders API source terms, schema drift, and run evidence.",
    contribution_disclosed=True,
    policy_ref="policy:scrp:agents:v1",
)

job = service.create_harvest_job(
    tenant_id=tenant_id,
    name="orders-hourly",
    source_id=source["id"],
    extractor_profile_id=extractor["id"],
    owner="data-owner",
    pipeline_target="etlp:orders",
)

run = service.run_harvest(tenant_id, job["id"], "scheduler")
completed = service.complete_harvest_run(
    tenant_id=tenant_id,
    run_id=run["id"],
    records_extracted=25,
    dlp_scanned=True,
    storage_ref="memory://orders",
)
```

### Async: Screen Capture and OCR

```python
import asyncio

async def screen_pipeline(service, tenant_id):
    capture = await service.capture_screen(
        tenant_id=tenant_id, url="https://example.com/report",
        output_format="png", full_page=True, owner="analyst",
    )
    ocr = await service.ocr_image(
        tenant_id=tenant_id, image_ref=capture["storage_ref"],
        model="tesseract", owner="analyst",
    )
    table = await service.ocr_extract_table(
        tenant_id=tenant_id, image_ref=capture["storage_ref"],
        header_row=True, owner="analyst",
    )
    return table["rows"]
```

### Async: RPA Workflow

```python
async def rpa_demo(service, tenant_id):
    return await service.rpa_workflow_run(
        tenant_id=tenant_id,
        workflow_id="login-and-extract",
        target_url="https://app.example.com",
        steps=[
            {"action": "navigate", "value": "https://app.example.com/login"},
            {"action": "type", "selector": "#username", "value": "user@example.com"},
            {"action": "click", "selector": "button[type=submit]"},
            {"action": "wait_for", "selector": ".dashboard"},
            {"action": "extract", "selector": ".report-table"},
        ],
        owner="rpa-agent",
    )
```

### Async: LLM Extraction

```python
async def llm_demo(service, tenant_id):
    return await service.llm_extract(
        tenant_id=tenant_id,
        content="Apple Q4 revenue $94B, up 6% YoY, headcount 164,000",
        schema={
            "company": {"type": "str"},
            "revenue_usd_b": {"type": "float"},
            "headcount": {"type": "int"},
        },
        model="mistral-nemo",
        cache_results=True,
    )
```

Use API helpers from `api.py` when composing the capability into generated APG
applications:

```python
from capabilities.common.scrp import api
from capabilities.common.scrp.service import ScrpService

service = ScrpService()
status = api.capability_status(service, "tenant-demo")
routes = status["routes"]
```

Use view models from `views.py` when a generated UI needs framework-neutral
screen state:

```python
from capabilities.common.scrp.views import dashboard_model, harvest_agents_model

dashboard = dashboard_model(service, "tenant-demo")
agents = harvest_agents_model(service, "tenant-demo")
```

## Async Method Reference

| Method | Description |
|--------|-------------|
| `capture_screen` | Screenshot of a web page (PNG/JPEG, full-page) |
| `ocr_image` | OCR text extraction from image (Tesseract / Ollama vision) |
| `ocr_extract_table` | OCR + table structure reconstruction |
| `rpa_workflow_run` | Execute multi-step RPA workflow via Playwright |
| `llm_extract` | LLM-powered structured extraction with content hash cache |
| `cursor_read` | Read incremental cursor for a harvest job |
| `cursor_advance` | Advance cursor after successful harvest |
| `cursor_reset` | Reset cursor for full re-harvest (requires reason) |
| `handoff_dispatch` | Deliver pipeline handoff via HTTP webhook |
| `handoff_retry` | Retry a dead-lettered handoff |
| `content_diff` | Line diff between source snapshots with alert threshold |
| `quality_report` | Per-field completeness and format conformance report |

## Configuration And Composition

The contract is published by `get_capability_contract()` in
`capability_contract.py`. It includes:

- `configuration` for sources, extraction, compliance, harvest agents,
  governance, observability, adapters, UI, and theme.
- `configuration_schema` for APG validation.
- `rule_engine` for deterministic guardrail evaluation.
- `ui` route metadata and the API prefix `/scrp/api/v1`.
- `theme` tokens and component affordances.
- `streaming` metadata declaring Bytewax as the lifecycle stream processor.

SCRP declares hard dependencies on `conn`, `etlp`, and `auth`. Optional adapter
boundaries are `i18n`, `nlpc`, `schd`, `dlpd`, `bytewax`, `audl`, and `moni`.

## Guardrail Summary

SCRP denies or requires review when:

- tenant context is missing;
- a source has no owner, terms evidence, credential reference, robots/terms
  policy, or positive rate limit;
- PII is expected without a handling policy;
- a sensitive source has not been reviewed;
- an extractor has no schema;
- a job lacks a pipeline target;
- a harvest run for PII-bearing data is completed without DLP evidence;
- an AI harvest agent is unregistered, uses an unsupported runtime or role,
  lacks explicit scope, or has undisclosed contributions;
- a lifecycle state change lacks a reason or audit evidence;
- a cross-tenant access attempt is detected;
- a batch harvest mutation does not declare Bytewax;
- a cursor reset is attempted without a stated reason;
- an RPA workflow contains an unsupported action type;
- a content diff uses an unsupported format.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/scrp/__init__.py capabilities/common/scrp/capability_contract.py capabilities/common/scrp/models.py capabilities/common/scrp/harvest_runtime.py capabilities/common/scrp/service.py capabilities/common/scrp/api.py capabilities/common/scrp/views.py capabilities/common/scrp/test_capability_contract.py capabilities/common/scrp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/scrp/test_capability_contract.py capabilities/common/scrp/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/scrp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/scrp --json
```
