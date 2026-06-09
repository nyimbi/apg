# Scraper/Data Harvesting

**Capability ID**: `scrp` | **Domain**: `common` | **Version**: `1.0.0`

## Description

SCRP is the APG capability for governed data-source harvesting. It lets an APG application register tenant-owned sources, define extractors, schedule harvest jobs, run guarded harvest lifecycles, record result batches, hand results to

## Installation

```bash
pip install apg-common-scrp
```

## Provides

- `source_registry`
- `harvest_jobs`
- `extractor_profiles`
- `compliance_controls`
- `pipeline_handoff`

## Requires

- `conn`
- `etlp`
- `auth`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/scrp/dashboard` | `scrp:view` | Overview |
| `/scrp/sources` | `scrp:configure_sources` | Sources |
| `/scrp/jobs` | `scrp:run_jobs` | Jobs |
| `/scrp/extractors` | `scrp:configure_sources` | Extraction |
| `/scrp/pipelines` | `scrp:view` | Extraction |
| `/scrp/compliance` | `scrp:approve_harvests` | Governance |
| `/scrp/results` | `scrp:view` | Results |
| `/scrp/agents` | `scrp:approve_harvests` | Agents |

## Key Service Methods

- `describe()`
- `evaluate()`
- `schedule_scrape()`
- `run_scrape()`
- `scrape_result()`
- `extract_structured_data()`
- `javascript_rendered_scrape()`
- `rate_limit_management()`
- `proxy_rotation()`
- `captcha_handling()`

_(See `service.py` for complete API.)_

## Interoperability

`scrp` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use scrp;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SCRP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
