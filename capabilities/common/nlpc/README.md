# APG NLPC - NLP Core

NLPC is the APG capability for governed text intelligence. It lets generated
applications ingest tenant-scoped documents, detect or declare language,
execute configured NLP tasks, manage pipelines and model releases, coordinate
annotation work, maintain tenant lexicons, expose UI view models, and emit audit
evidence through a deterministic rule engine.

## What It Provides

- Document ingestion with tenant, content, language, size, source, hash, and
  audit evidence.
- Processing runs for sentiment analysis, entity recognition, PII detection,
  summarization, semantic search, translation, classification, topic modeling,
  keyword extraction, and governed text generation.
- Pipeline registration with owner, model linkage, version metadata, enabled
  tasks, and policy checks.
- NLP model registration and release with MLCM linkage, evaluation evidence,
  approval evidence, and audit events.
- Annotation projects with guidelines, consensus thresholds, adjudication
  checks, and annotation records.
- Tenant lexicons with language metadata and term ownership.
- At least 40 African language codes in the language registry.
- Rule-engine metadata, UI route metadata, theme tokens, and generated-app
  semantic package evidence.
- Adapter configuration for AICR, MLCM, CONF, AUTH, AUDL, MONI, SRCH, and
  Bytewax event streaming.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `nlpc_runtime.py` - `NlpcService`, the dependency-light generated-app runtime.
- `view_models.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.

## Generated-App Usage

```python
from capabilities.common.nlpc.nlpc_runtime import NlpcService

service = NlpcService()
document = service.ingest_document(
	"doc-001",
	"tenant-a",
	"Habari Nairobi. This excellent report was prepared by Amina.",
	"auto",
	"case://001",
)
model = service.register_model(
	"model-001",
	"tenant-a",
	"Entity and Sentiment Model",
	"mlcm://nlpc/entity-sentiment",
	"language-team",
	"policy://nlpc/safe",
)
pipeline = service.register_pipeline(
	"pipe-001",
	"tenant-a",
	"Customer Text Pipeline",
	"language-team",
	model["id"],
	"1.0.0",
	["sentiment_analysis", "entity_recognition", "semantic_search"],
)
run = service.process_document(
	"run-001",
	"tenant-a",
	document["id"],
	pipeline["tasks"],
	search_index_attached=True,
)
```

## Guardrails

NLPC blocks missing tenant context, empty documents, oversized documents,
missing language evidence when detection is disabled, unsupported languages,
low-confidence language detection without review, disabled tasks, PII detection
without redaction policy, text generation without safety and model policy,
translation without source and target evidence, semantic search without a search
index, summarization without a length budget, low-confidence results without
review, large batches without async queueing, batch event streams that are not
Bytewax, pipelines without owner/model/version, model registration without MLCM
linkage, model release without evaluation or approval, annotation projects
without guidelines, low annotation consensus without adjudication, lexicons
without language, quality metrics without owner, cross-tenant processing, state
changes without audit evidence, and language registries with fewer than 40
African language codes.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/nlpc/__init__.py capabilities/common/nlpc/capability_contract.py capabilities/common/nlpc/nlpc_runtime.py capabilities/common/nlpc/view_models.py capabilities/common/nlpc/app.py capabilities/common/nlpc/test_capability_contract.py capabilities/common/nlpc/test_language_codes.py capabilities/common/nlpc/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/nlpc/test_capability_contract.py capabilities/common/nlpc/test_language_codes.py capabilities/common/nlpc/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/nlpc --json
./.venv/bin/apg capabilities publish-plan capabilities/common/nlpc --json
```
