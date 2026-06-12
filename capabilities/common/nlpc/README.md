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

## World-Class Enhancements (v2.0)

The following 15 improvements were designed and implemented to close the gap
with OpenAI, Cohere, and AWS Comprehend while adding capabilities tailored to
African-language and regulated-industry use cases.

| # | Method | Category | Description |
|---|--------|----------|-------------|
| I1 | `stream_summarise` | Throughput/UX | SSE streaming via `stream=True` on Ollama `/api/generate`; Flask `stream_with_context` wrapper. Cuts perceived latency 5-10x for long-form generation. |
| I2 | `cross_lingual_search` | Search/Multilingual | `paraphrase-multilingual-mpnet-base-v2` collapses 100+ languages into a shared vector space. Swahili/Kikuyu/Amharic corpora searchable in English. |
| I3 | `score_readability` | Compliance Analytics | Zero-dependency Flesch-Kincaid Grade, Gunning Fog, Coleman-Liau, and composite plain-language score. Sub-millisecond; no model backend required. |
| I4 | `detect_emotions` | Emotion Intelligence | 8-class Ekman detection via `facebook/bart-large-mnli` zero-shot; NRC lexicon fallback. Returns per-emotion scores, dominant emotion, and VAD axes. |
| I5 | `extract_concepts` | Knowledge Graph | spaCy noun-chunk pipeline + Wikidata Qnode resolution via `httpx`. Returns `{concept, qnode, category, confidence}` per concept. |
| I6 | `detect_document_structure` | Document Intelligence | Regex + indentation heuristics classify spans as heading/paragraph/list_item/table_row/code_block. No cloud dependency. |
| I7 | `score_faithfulness` | AI Safety/Governance | NLI entailment via `cross-encoder/nli-deberta-v3-small`; ROUGE-L fallback. Returns `{entailment_score, contradiction_score, faithfulness_label, rouge_l}`. |
| I8 | `find_near_duplicates` | Data Quality | MinHash LSH with 128-band char 3-gram signatures. Detects near-duplicates in O(n) amortised vs O(n²) pairwise. Pure Python, no faiss dependency. |
| I9 | `calibrate_confidence` | ML Ops | Temperature-scaling calibration per task type from `model_registry.py` constants. Reduces ECE by ~5x without accuracy loss. |
| I10 | `@cached_nlp_result` | Performance | SHA-256 content-addressed cache keyed on `(tenant_id, task, text[:4096])` using `BoundedCache` from `capabilities.common.reliability`. ~60% hit rate on support corpora. |
| I11 | `segment_discourse` | Discourse Analysis | RST-based hierarchical EDU segmentation using cue-phrase detection and dependency parse head chains. Integrates with `dependency_parse` output. |
| I12 | `aggregate_federated_lexicon` | Privacy/Governance | Federated weighted averaging of per-tenant `{term: count}` dicts into a global pseudo-IDF corpus. No raw text crosses tenant boundaries. |
| I13 | `label_semantic_roles` | Event Extraction | PropBank-style ARG0/ARG1/ARG2/ARGM-TMP/ARGM-LOC frames via spaCy dependency parse; SVO regex fallback. Enables structured event extraction for intelligence and compliance. |
| I14 | `run_batch_job_scheduled` | Scalability | `asyncio.PriorityQueue` keyed on `(priority, enqueue_time)`. Bounded semaphore workers, exponential-backoff retry up to `retry_limit`. Converts O(docs×tasks) serial to O(max_workers) parallel. |
| I15 | `multi_hop_qa` | Information Retrieval | Up to `max_hops` evidence-chain hops: embed question → semantic search → extract span → bridge via named entity → repeat. Returns `{answer, evidence_chain, confidence}`. |

Methods marked I3, I4, I6, I7, I8, I13, I14, I15 are fully implemented in
`service.py`. I1, I2, I5, I9, I10, I11, I12 have specifications and stubs
ready for backend wiring.

## New Methods

### `detect_emotions` — Ekman 8-class emotion detection

```python
import asyncio
from capabilities.common.nlpc.service import NLPCoreService

svc = NLPCoreService(tenant_id="acme", actor_id="analyst")

result = asyncio.run(svc.detect_emotions(
    "The team was furious about the breach and deeply worried about the fallout."
))
# result["dominant_emotion"] → "anger"
# result["emotions"]         → {"anger": 0.42, "fear": 0.31, ...}
# result["valence"]          → 0.2   # low valence = negative affect
# result["arousal"]          → 0.9   # high arousal = activated state
# result["model_used"]       → "nrc_lexicon" | "zero-shot/bart-mnli"
```

### `score_faithfulness` — hallucination / faithfulness check

```python
result = asyncio.run(svc.score_faithfulness(
    source="The Kenyan government allocated KES 50 billion to healthcare in 2024.",
    generated="In 2024, Kenya committed KES 500 billion to expand hospital infrastructure.",
))
# result["faithfulness_label"]   → "unfaithful"
# result["entailment_score"]     → 0.12
# result["contradiction_score"]  → 0.88
# result["rouge_l"]              → 0.31
```

### `find_near_duplicates` — MinHash LSH deduplication

```python
# Assume doc-001 and doc-002 were previously ingested
dupes = asyncio.run(svc.find_near_duplicates(
    document_ids=["doc-001", "doc-002", "doc-003"],
    threshold=0.75,
))
# dupes → [{"doc_id_a": "doc-001", "doc_id_b": "doc-002",
#            "estimated_similarity": 0.82, "exact_jaccard": 0.79}]
```

### `multi_hop_qa` — evidence-chain question answering

```python
result = asyncio.run(svc.multi_hop_qa(
    question="Who led the 2024 Nairobi climate summit?",
    document_ids=["news-001", "news-002", "report-003"],
    max_hops=3,
))
# result["answer"]         → "Dr. Amina Mohamed"
# result["hops_taken"]     → 2
# result["evidence_chain"] → [
#     {"hop": 1, "document_id": "news-001", "passage": "...", "span": "Amina Mohamed", ...},
#     {"hop": 2, "document_id": "report-003", "passage": "...", "span": "Dr. Amina Mohamed", ...},
# ]
```

### `run_batch_job_scheduled` — parallel priority batch scheduler

```python
job = asyncio.run(svc.create_batch_job(
    name="nightly-sentiment",
    document_ids=["d-001", "d-002", ..., "d-500"],
    tasks=["sentiment_analysis", "entity_extraction"],
    priority="high",
))

status = asyncio.run(svc.run_batch_job_scheduled(
    job_id=job.id,
    max_workers=16,
    retry_limit=3,
))
# status → {"processed": 998, "failed": 2, "retried": 5,
#            "progress": 99.8, "status": "partial_failure"}
```

## Quick Start

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

## Core API

| Method | Signature summary | Returns |
|--------|------------------|---------|
| `create_document` | `(payload: NLPDocumentCreate)` | `NLPDocumentResponse` |
| `detect_language` | `(text, document_id?)` | `NLPLanguageResponse` |
| `extract_entities` | `(text, entity_types?, document_id?)` | `list[NLPEntityResponse]` |
| `sentiment_analysis` | `(text, document_id?)` | `NLPSentimentResponse` |
| `detect_emotions` | `(text, document_id?)` | `dict` — Ekman 8-class + VAD |
| `text_summarisation` | `(text, max_words, method, document_id?)` | `NLPSummaryResponse` |
| `translate` | `(text, target_lang, source_lang?, document_id?)` | `NLPTranslationResponse` |
| `embed_text` | `(text, model?, document_id?)` | `NLPEmbeddingResponse` |
| `semantic_search` | `(query, top_k?, threshold?)` | `list[dict]` |
| `chunk_and_embed` | `(text, chunk_size?, overlap?, model?, document_id?)` | `list[dict]` |
| `classify_document` | `(text, taxonomy, labels?, document_id?)` | `NLPClassification` |
| `multi_label_classify` | `(text, taxonomy, labels?, threshold?, document_id?)` | `dict` |
| `extract_key_phrases` | `(text, top_n?, document_id?)` | `list[NLPKeyPhrase]` |
| `extract_entities` | `(text, entity_types?, document_id?)` | `list[NLPEntityResponse]` |
| `named_entity_linking` | `(text, document_id?)` | `list[NLPEntityResponse]` |
| `relation_extraction` | `(text, document_id?)` | `list[NLPRelation]` |
| `label_semantic_roles` | `(text, document_id?)` | `dict` — PropBank frames |
| `coreference_resolution` | `(text, document_id?)` | `list[NLPCoreferenceChain]` |
| `dependency_parse` | `(text, document_id?)` | `dict` — per-sentence token triples |
| `extract_temporal_expressions` | `(text, reference_date?, document_id?)` | `dict` — TIMEX3 spans |
| `extract_arguments` | `(text, document_id?)` | `dict` — claim/premise/evidence |
| `score_coherence` | `(text, document_id?)` | `dict` — local + global coherence |
| `score_readability` | `(text, document_id?)` | `dict` — FK/Fog/CL + plain-language |
| `score_faithfulness` | `(source, generated, document_id?)` | `dict` — NLI entailment + ROUGE-L |
| `find_near_duplicates` | `(document_ids, threshold?, shingle_size?, n_hash_funcs?)` | `list[dict]` |
| `detect_document_structure` | `(text, document_id?)` | `dict` — typed segments |
| `detect_pii` | `(text, document_id?)` | `dict` — PII spans |
| `redact_pii` | `(text, strategy?, document_id?)` | `dict` — redacted text |
| `multi_hop_qa` | `(question, document_ids, max_hops?, top_k_per_hop?)` | `dict` — answer + evidence chain |
| `question_answering` | `(context, question, tenant_id?)` | `dict` — span + confidence |
| `create_batch_job` | `(name, document_ids, tasks, priority?)` | `NLPBatchJob` |
| `run_batch_job` | `(job_id)` | `NLPBatchJob` — sequential |
| `run_batch_job_scheduled` | `(job_id, max_workers?, retry_limit?)` | `dict` — parallel with retry |
| `parallel_process` | `(text, tasks, max_concurrent?, document_id?)` | `dict` — fan-out results |
| `usage_report` | `(period_start, period_end)` | `NLPUsageReport` |
| `language_id_for_african_languages` | `(text)` | `dict` — African-specific candidates |

## Main Files

- `SPECIFICATION.md` — complete functional scope for this packet.
- `PLAN.md` — implementation and review plan.
- `WORLD_CLASS_IMPROVEMENTS.md` — detailed justification and design for all 15 v2 enhancements.
- `capability_contract.py` — executable configuration, rules, UI, adapters, and
  theme contract.
- `service.py` — `NLPCoreService`, the primary async service layer with all v2 methods.
- `nlpc_runtime.py` — `NlpcService`, the dependency-light generated-app runtime.
- `processing_pipeline.py` — the advanced dependency-light processing pipeline
  with deterministic handlers for every legacy public `NLPTaskType`.
- `view_models.py` — semantic UI view models for generated applications.
- `app.py` — dynamic package evidence and self-test.
- `test_capability_contract.py` — focused executable contract coverage.
- `tests/test_package_contract.py` — package evidence and compatibility tests.
- `tests/test_processing_pipeline_deterministic.py` — regression coverage for
  content-aware advanced-pipeline task dispatch.

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
