# Intelligence Crawler Specification

## Intent

Intelligence Crawler (`intel_crawler`) makes governed source collection and content preparation a composable APG capability. It coordinates source registration, crawl-job scheduling, extraction quality, dataset publication, validation sessions, RAG preparation, knowledge-graph projection, crawler-agent review, UI routes, theming, deterministic rules, and Bytewax lifecycle streaming.

The capability is designed as a dependency-light package boundary around existing crawler adapters. Live web crawling, social feeds, search adapters, and media-specific collectors can sit behind this boundary without making package import depend on optional web, database, or browser automation services.

## Functional Requirements

- Register tenant-scoped sources with owner, source type, URLs, allowed domains, and crawl-policy review.
- Create crawl jobs with cadence, maximum depth, positive rate limit, and approval for high-risk jobs.
- Complete crawl jobs with fetched and error counts.
- Record extraction batches with schema, content fingerprint, and quality score.
- Open and complete validation sessions with assigned reviewer and confidence score.
- Publish datasets only when lineage and validation are present.
- Require privacy review when publishing datasets containing PII.
- Record RAG preparation plans with chunk strategy, chunk-size limit, and embedding model.
- Record knowledge-graph projections with entity schema and relationship evidence.
- Register first-class crawler agents for Codex, Claude Code, OpenCode, and Pi.
- Validate privileged crawler-agent actions through a human approval guardrail.
- Emit lifecycle events through a Bytewax-backed stream named `apg.intel.crawler.lifecycle`.

## Rule Engine

The deterministic rule engine evaluates plain context dictionaries and returns `allow`, `deny`, or `require_review`. It enforces tenant context, write policy attachment, source ownership, URL and allowed-domain presence, source crawl-policy review, crawl source/cadence/rate/depth/approval, extraction schema/fingerprint/quality, dataset lineage/validation/privacy, validation reviewer/confidence, RAG chunk plan/chunk size/embedding model, graph entity schema/relationship evidence, Bytewax routing, supported crawler-agent runtime and role, and human approval for privileged agent actions.

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, deterministic rules, UI routes, theme tokens, and Bytewax streaming metadata.
- Package import exposes `IntelligenceCrawlerService`, `CrawlerDatabaseService`, `CrawlerService`, contract helpers, streaming metadata, and registration metadata without requiring optional crawler dependencies.
- Service supports source, crawl-job, extraction, validation, dataset, RAG, graph, crawler-agent, dashboard, audit, import-validation, and compatibility record operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes crawler-agent metadata, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths, guardrail failures, API/view execution, app self-test, and semantic metadata.

