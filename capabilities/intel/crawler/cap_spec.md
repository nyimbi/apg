# Intelligence Crawler Capability Summary

`intel_crawler` is a dependency-light APG package for composing governed collection and content-preparation applications. It covers source registry, crawl-job lifecycle, extraction quality, dataset publication, validation workflow, RAG preparation, knowledge-graph projection, crawler-agent review, deterministic guardrails, UI route metadata, theme tokens, and Bytewax lifecycle streaming.

## Lifecycle

1. Register a tenant source with owner, URLs, allowed domains, and crawl-policy review.
2. Create a crawl job with cadence, crawl depth, and rate limit.
3. Complete the crawl job with fetched and error counts.
4. Record extraction output with schema, fingerprint, and quality score.
5. Open and complete validation.
6. Publish a dataset with lineage, validation, and privacy review when needed.
7. Prepare RAG chunking and knowledge-graph projection.
8. Register crawler agents that can recommend, validate, and prepare source operations under human-approval guardrails.

## Composition Surface

Provides:

- `source_intelligence_registry`
- `crawl_job_lifecycle`
- `extraction_pipeline`
- `dataset_quality_control`
- `validation_workflow`
- `rag_graphrag_preparation`
- `crawler_agents`

Requires:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `document_processing`

## Runtime Entry Points

- `capability_contract.py`: APG contract, rules, UI, theme, and Bytewax stream metadata.
- `service.py`: executable crawler domain service.
- `api.py`: dependency-light API helper functions.
- `views.py`: UI view models for APG composition.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused package verification.
