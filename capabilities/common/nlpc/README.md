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
  keyword extraction, and governed text generation, including pending-review
  state for low-confidence or budget-incomplete output.
- Fine-grained Ekman 8-class emotion detection (joy, anger, fear, disgust,
  surprise, sadness, anticipation, trust) with VAD axis scores.
- Readability scoring: Flesch-Kincaid Grade Level, Gunning Fog, Coleman-Liau,
  and composite plain-language score with no model dependency.
- Hallucination/faithfulness scoring via NLI entailment (cross-encoder/
  nli-deberta-v3-small) with ROUGE-L fallback.
- MinHash LSH near-duplicate detection across document collections.
- Document structure detection: heading/paragraph/list_item/table_row/code_block
  segmentation with structure_score.
- Semantic Role Labelling (SRL) with PropBank-style ARG0/ARG1/ARG2/ARGM frames.
- Multi-hop question answering with evidence chains across multiple documents.
- Priority-queue parallel batch scheduler with async workers, exponential-backoff
  retry, and configurable concurrency semaphore.
- Pipeline registration with owner, model linkage, version metadata, enabled
  tasks, and policy checks.
- NLP model registration and release with MLCM linkage, evaluation evidence,
  approval evidence, and audit events.
- Annotation projects with guidelines, consensus thresholds, adjudication
  checks, and annotation records.
- Tenant lexicons with language metadata and term ownership.
- At least 40 African language codes in the language registry.
- First-class NLP-agent composition for Codex, Claude Code, OpenCode, Pi, and
  future AICR-adapted runtimes.
- Bytewax lifecycle-batch validation for generated-app and agent-authored text
  mutations.
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
- `processing_pipeline.py` - the advanced dependency-light processing pipeline
  with deterministic handlers for every legacy public `NLPTaskType`.
- `view_models.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.
- `tests/test_processing_pipeline_deterministic.py` - regression coverage for
  content-aware advanced-pipeline task dispatch.

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
service.register_nlp_agent(
	"nlp-reviewer",
	"tenant-a",
	"NLPC Safety Reviewer",
	"codex",
	"generation_safety_reviewer",
	"pipe-001 generation outputs",
	"language-team",
	"Review generated summaries and safety policy drift",
	human_approval_required=True,
)
service.validate_nlpc_lifecycle_batch(
	"tenant-a",
	"bytewax",
	4,
	"nlp_agent_batch",
	"batch-001",
)
```

Review-required processing is retained as executable state:

```python
pending = service.process_document(
	"run-review",
	"tenant-a",
	document["id"],
	"summarization",
	length_budget_present=False,
)
assert pending["status"] == "pending_review"
assert pending["matched_rules"] == ["summarization_requires_length_budget"]
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
Review-required processing outcomes are not dropped. They are stored as
`pending_review` runs with `decision`, `matched_rules`, and `review_reasons` so
the processing console and human review console can route work immediately.
Agent guardrails also block unsupported runtimes, unsupported roles, missing
scope, missing owner, missing purpose, undisclosed machine contribution, and
non-Bytewax lifecycle batches. Privileged NLP-agent roles without explicit
human approval are retained as `pending_review` instead of being silently
activated.

## Agent Composition

NLPC agents are provider-neutral text-governance actors. The contract currently
recognizes `codex`, `claude_code`, `opencode`, and `pi` runtime codes; live
runtime execution remains behind AICR adapter contracts. Generated
applications compose agents through `register_nlp_agent()` and inspect them
through `list_nlp_agents()` or the `/nlpc/agents` route metadata.

Supported roles include document review, language review, PII review,
generation safety, annotation review, pipeline review, model-release review,
semantic-search review, and language steward responsibilities. PII,
generation-safety, annotation, pipeline, model-release, and semantic-search
roles are privileged and require human approval evidence for active status.

## Bytewax Lifecycle Batches

NLPC uses Bytewax for lifecycle mutation governance. The streaming manifest
requires the `nlpc.lifecycle` stream and declares operation names for document,
processing, pipeline, annotation, model, lexicon, language-registry, and
NLP-agent batches. Generated applications validate those batches with
`validate_nlpc_lifecycle_batch()` and inspect accepted or denied evidence
through `list_lifecycle_batches()` or the `/nlpc/lifecycle` route metadata.

## Advanced Pipeline Baseline

`AdvancedProcessingPipeline.process_single()` is executable without live model
services. It dispatches every legacy public `NLPTaskType` to deterministic local
handlers for sentiment, entity extraction, classification, summarization,
language detection, text similarity, question answering, generation, POS tags,
dependency hints, topic extraction, keyword extraction, and clustering. This is
the generated-app baseline: provider-backed AICR or model-registry execution can
replace handlers later, but apps receive content-aware structured output today
instead of placeholder `processed` responses.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/nlpc/__init__.py capabilities/common/nlpc/capability_contract.py capabilities/common/nlpc/nlpc_runtime.py capabilities/common/nlpc/processing_pipeline.py capabilities/common/nlpc/view_models.py capabilities/common/nlpc/app.py capabilities/common/nlpc/test_capability_contract.py capabilities/common/nlpc/test_language_codes.py capabilities/common/nlpc/tests/test_package_contract.py capabilities/common/nlpc/tests/test_processing_pipeline_deterministic.py
./.venv/bin/pytest -q capabilities/common/nlpc/test_capability_contract.py capabilities/common/nlpc/test_language_codes.py capabilities/common/nlpc/tests/test_package_contract.py capabilities/common/nlpc/tests/test_processing_pipeline_deterministic.py
./.venv/bin/python capabilities/common/nlpc/app.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/nlpc --json
./.venv/bin/apg capabilities publish-plan capabilities/common/nlpc --json
```
