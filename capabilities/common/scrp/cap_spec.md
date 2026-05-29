# APG Capability Specification: SCRP - Scraper/Data Harvesting

`scrp` is the APG Scraper/Data Harvesting capability. It provides
tenant-aware source registration, terms and credential evidence, rate controls,
extractor profiles, harvest jobs, deterministic harvest runs, result batches,
pipeline handoffs, compliance controls, audit events, API helpers, route
metadata, and data-harvesting theming.

## Executable Runtime

The package is implemented as a dependency-light Python runtime:

| Surface | File | Responsibility |
| --- | --- | --- |
| Contract | `capability_contract.py` | configuration, deterministic harvesting rules, UI routes, and theme tokens |
| Runtime helpers | `harvest_runtime.py` | stable IDs, source/extractor/mode normalization, DLP status, run status, retention hints |
| Models | `models.py` | sources, extractors, jobs, runs, result batches, pipeline handoffs, and audit events |
| Service | `service.py` | tenant-scoped lifecycle methods and policy enforcement |
| API helpers | `api.py` | callable package API surface for composition and generated apps |
| View models | `views.py` | dashboard, sources, jobs, extractors, pipelines, compliance, results, settings |
| Package entrypoint | `app.py` | publishable semantic model, component manifest, and self-test |

The runtime intentionally does not fetch remote websites, call external APIs,
run browsers, store credentials, or push records to a live ETL system. It
records deterministic harvesting metadata and keeps live scrapers, crawlers,
browser drivers, credential vaults, schedulers, DLP scanners, and ETL pipelines
behind explicit future adapters.

## Domain Model

`ScrpService` manages:

- harvest sources with endpoint, type, owner, terms evidence, credential vault
  reference, rate limits, robots policy, PII policy, sensitivity review, and
  tags
- extractor profiles with parser type, schema, output mapping, validation
  posture, and incremental cursor metadata
- harvest jobs that bind sources to extractors, schedule policy, mode,
  pipeline target, and enabled state
- harvest runs with requested actor, run status, extracted record count,
  error count, DLP status, DLP violations, logs, and timestamps
- result batches with schema validity, storage reference, retention hint, and
  record count
- pipeline handoffs for ETL/data-platform integration
- audit events for source, extractor, job, run, result, and handoff operations

The compatibility `create_record` and `list_records` methods produce and list
sources so existing package tooling can keep treating SCRP as a composable APG
package while richer harvesting APIs are used by new code.

## Rule Engine

SCRP uses deterministic rule evaluation from `capability_contract.py`.

| Rule | Enforced by |
| --- | --- |
| `tenant_context_required` | all service methods that create or mutate tenant-scoped objects |
| `source_requires_owner` | source registration |
| `source_terms_required` | source registration and harvest execution |
| `pii_requires_handling_policy` | PII-bearing source registration and harvest execution |
| `harvest_requires_schedule_policy` | harvest job execution |
| `sensitive_source_requires_review` | sensitive source registration and harvest execution |

Robots policy, credential vault references, rate limits, DLP status, schema
validity, and result-retention metadata are modeled locally so APG can prove
governed harvesting behavior without live scraping side effects.

## UI And Theme Contract

The package publishes APG route metadata for:

- `/scrp/dashboard`
- `/scrp/sources`
- `/scrp/jobs`
- `/scrp/extractors`
- `/scrp/pipelines`
- `/scrp/compliance`
- `/scrp/results`
- `/scrp/settings`

The default theme is `scrp_harvest_ops`. Components include source card, job
monitor, extractor workbench, and compliance panel tokens. View models expose
plain dictionaries so generated Python apps, APG Studio, and future UI adapters
can compose the harvesting capability without framework-specific imports.

## Adapter Boundaries

Future live integrations should attach behind explicit adapters:

- HTTP/API/browser crawling adapters
- credential-vault adapters
- scheduler adapter for `schd`
- DLP scanner adapter for `dlpd`
- NLP/parser adapter for `nlpc`
- ETL pipeline adapter for `etlp`
- connector adapter for `conn`
- audit sink adapter for `audl`

Do not make package import, tests, publish-plan, or implementation audit depend
on those live providers.

## Focused Verification

Use focused checks while developing SCRP:

```bash
rg -n "<generated-baseline marker alternation>" capabilities/common/scrp
./.venv/bin/python -m py_compile capabilities/common/scrp/__init__.py capabilities/common/scrp/models.py capabilities/common/scrp/harvest_runtime.py capabilities/common/scrp/service.py capabilities/common/scrp/api.py capabilities/common/scrp/views.py capabilities/common/scrp/capability_contract.py capabilities/common/scrp/app.py capabilities/common/scrp/test_capability_contract.py capabilities/common/scrp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/scrp/test_capability_contract.py capabilities/common/scrp/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/scrp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/scrp --json
```

The baseline-marker search should return no matches. The implementation audit
should classify `scrp` as `domain_specific` and report no root warnings.
