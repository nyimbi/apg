# Intelligence Crawler Development Plan

## Slice Goal

Deliver a coherent lifecycle and guardrail packet for `intel_crawler` so the capability is executable, documented, testable, theme-aware, AI-agent aware, and Bytewax-aligned.

## Implementation Steps

1. Replace the generic contract with a domain-specific crawler contract for sources, crawl jobs, extraction, datasets, validation, RAG, graph projection, agents, governance, observability, adapters, UI, theme, provides/requires, and Bytewax lifecycle streaming.
2. Replace dependency-heavy top-level runtime surfaces with dependency-light service, API, view, and app modules that compile without optional web, database, browser, or crawler libraries.
3. Preserve compatibility names such as `CrawlerDatabaseService`, `CrawlerService`, `create_record`, and `list_records`.
4. Add deterministic guardrails for tenant context, write policy, source policy review, crawl rate/depth/approval, extraction quality, dataset privacy, validation confidence, RAG preparation, graph evidence, Bytewax routing, and AI-agent approval.
5. Refresh package metadata and semantic evidence from the active contract.
6. Add README and specification documents that explain purpose, lifecycle, guardrails, APIs, UI, theming, streaming, and verification.
7. Expand focused package tests around contract shape, rule execution, service lifecycle, guardrail failures, API/view surfaces, and semantic metadata.
8. Run battery-conscious verification: compile touched package files, run the focused crawler package tests, inspect package metadata, and scan touched files for stale marker terms.

## Known Deferred Work

- Bind existing Google News, search, GDELT, Twitter/X, YouTube, and news adapters to the service boundary.
- Deploy durable Bytewax topologies and persistent crawl-result stores.
- Add live robots policy checks, browser rendering, and source-specific rate governance.
- Add performance, concurrency, and failure-recovery validation after the capability family is stabilized.

