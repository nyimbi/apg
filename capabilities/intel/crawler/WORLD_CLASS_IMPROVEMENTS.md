# World-Class Improvements: Web Intelligence Crawler (intel_crawler)

**Capability:** `capabilities/intel/crawler`  
**Author:** Datacraft — Nyimbi Odero  
**Date:** 2026-06-11

---

## 1. Adaptive Rate Limiter with Token Bucket + Jitter

Replace the static `rate_limit_per_minute` integer with a runtime adaptive token-bucket that tracks real server response times and backs off exponentially on 429/503. Per-domain buckets prevent one hostile source from starving the whole pool. Jitter (±15 %) defeats fingerprinting by crawl-detection systems.

**Why it matters:** Static limits either over-crawl (bans) or under-crawl (staleness). Adaptive limits maximise throughput within the polite envelope.

---

## 2. robots.txt / crawl-delay Enforcement Layer

Parse and cache `robots.txt` for every registered domain before the first fetch. Honour `Crawl-delay`, `Disallow`, `Allow`, and `Sitemap` directives. Expose `robots_compliance_mode` per source: `strict`, `advisory`, `disabled` (requires explicit approval). Store compliance events in the audit trail.

**Why it matters:** Legal and ethical baseline for any production intelligence product. Strict mode is the default and cannot be disabled without an approval record.

---

## 3. Content Diffing and Change Detection

For recurring crawl schedules, compute a structural diff (line-level for text, key-level for JSON, node-level for HTML) against the previous fingerprinted version. Emit `content_changed` events only when diff similarity falls below a configurable threshold. Skip downstream processing for near-duplicate pages.

**Why it matters:** 60–80 % of recrawled pages are identical. Diffing eliminates wasted extraction, embedding, and storage work.

---

## 4. Dark Web / Tor Proxy Routing

Add a `transport_profile` field to source records: `clearnet`, `tor`, `i2p`. When Tor is selected, route requests through a configurable SOCKS5 proxy (e.g. `localhost:9050`). Include circuit-rotation on domain change and canary URL verification. Requires explicit `high_risk=True` and `approved_by` on the crawl job.

**Why it matters:** Open-source intelligence mandates dark-web coverage for threat tracking. Without first-class Tor support, operators build fragile workarounds outside the governance boundary.

---

## 5. Social Media Streaming Ingest (RSS + API Adapters)

Add adapter slots for Twitter/X firehose (academic API), Reddit pushshift, Mastodon streaming API, and Telegram channel RSS. Each adapter normalises output to the same `CrawledPage` schema consumed by the extraction pipeline. Streaming items land in the Bytewax event stream directly, bypassing the polling scheduler.

**Why it matters:** News velocity on social media is 4–6 hours ahead of web pages. Missing this window degrades time-sensitive intelligence products.

---

## 6. Multilingual Content Detection and Language-Aware Chunking

Detect page language (via `langdetect` or `fasttext`) and store the ISO-639-1 code on every extraction record. Apply language-specific tokenisers for chunking (e.g. no whitespace splitting for CJK). Route language-specific content to matching embedding models (e.g. `multilingual-e5-large` for non-English).

**Why it matters:** Africa and MENA intelligence targets are predominantly non-English. Monolingual chunking destroys semantic coherence for those languages.

---

## 7. Structured Data Extraction (JSON-LD, Microdata, OpenGraph)

Before generic text extraction, scan pages for `<script type="application/ld+json">`, Microdata, and OpenGraph tags. Parse these into typed records (Article, Event, Product, Person, Organisation) and store them as first-class structured extractions alongside free-text. Quality scores for structured extractions start at 0.95 vs the raw-HTML baseline.

**Why it matters:** Structured data is authoritative, parseable without NLP, and already schema-aligned—highest signal-to-noise ratio of any extraction path.

---

## 8. Privacy-Preserving PII Scrubber

Run a regex + NER-based PII detector (names, emails, phone numbers, national IDs, GPS coordinates, credit card numbers) before extraction records are persisted. Replace detected PII with typed placeholders (`[PERSON_NAME]`, `[EMAIL]`, etc.). Store a `pii_scrubbed` flag and scrub report on the extraction record. Block publication of unscrubbed PII datasets.

**Why it matters:** GDPR, Kenya DPA, and APG's own policy mandate PII handling. Automated scrubbing as a gate prevents accidental compliance failures at dataset publication.

---

## 9. Confidence-Scored Source Reputation Index

Maintain a per-domain reputation score (0.0–1.0) derived from: historical extraction quality, factual accuracy signals from validation sessions, known misinformation status (MBFC database), and TLS certificate health. Surface the score on source records and use it to down-weight low-reputation sources in downstream RAG and alert pipelines.

**Why it matters:** Raw aggregation of low-quality sources pollutes knowledge bases. A reputation gate preserves signal integrity without manual curation overhead.

---

## 10. Resumable Crawl Checkpointing

Persist crawl frontier state (URLs visited, URLs queued, extraction status per URL) to an append-only log after every batch of N pages. On restart after failure, resume from the last checkpoint rather than re-crawling from zero. Expose `checkpoint_coverage_pct` in the health report.

**Why it matters:** Long deep-crawls of large sites frequently fail midway. Without checkpointing, operators restart from scratch and waste hours of compute and bandwidth.

---

## 11. Async Batch Extraction Pipeline with Back-Pressure

Replace the synchronous `record_extraction` call with an async pipeline: HTTP fetch -> HTML clean -> text extract -> NER -> embed -> store, each stage running in a bounded async queue. Apply back-pressure when the embedding queue depth exceeds a configurable high-water mark. Expose queue depth in the health report.

**Why it matters:** Current synchronous calls block the event loop and cannot saturate multi-core hardware. An async pipeline with back-pressure achieves 10–20x throughput on typical crawl workloads.

---

## 12. Cross-Source Entity Deduplication (Record Linkage)

After entity extraction, run blocking + similarity matching across all extractions for the same tenant to identify co-referent entities (same person/org mentioned under different surface forms). Merge candidates above a configurable threshold and store a canonical entity ID. Feed the merged entity graph into `record_graph_projection` automatically.

**Why it matters:** Without cross-source deduplication, knowledge graphs fragment into isolated subgraphs per source. Deduplication is the key step that transforms disparate mentions into coherent intelligence.

---

## 13. LLM-Assisted Extraction Verification (Hallucination Guard)

After entity and relationship extraction, optionally route a random sample (configurable %) to a locally-hosted Ollama model (e.g. `mistral:7b`) for spot-check verification. The LLM confirms whether the extracted claim is supported by the source text. Flag extractions that fail verification as `unverified` and exclude them from high-confidence datasets.

**Why it matters:** NER and relation extraction models have systematic error patterns. An LLM spot-check catches the highest-severity errors (hallucinated relationships, wrong entity types) before they reach downstream consumers.

---

## 14. Webhook and Outbound Notification Bus

Add a `notify_webhook` field to source records and crawl jobs. After each completed crawl, extraction, or validation event, POST a signed JSON payload to the configured endpoint. Include HMAC-SHA256 signature for authenticity. Support retry with exponential backoff. Expose delivery status in the audit trail.

**Why it matters:** Intelligence consumers need push notifications, not polling. Webhooks close the latency gap between data availability and consumer action, and eliminate polling load on the API.

---

## 15. Semantic Deduplication via Embedding Cosine Similarity

Augment the SHA-256 fingerprint check with a vector-space near-duplicate detector. Compute an embedding of the first 512 tokens of each new extraction and compare cosine similarity against embeddings of recent extractions from the same domain. Flag pairs above a configurable threshold (e.g. 0.95) as near-duplicates even when the raw text differs (e.g. reformatted article syndication).

**Why it matters:** Syndicated content and reformatted reposts share identical information but different markup—the SHA-256 check misses them entirely. Semantic dedup reduces duplicate knowledge-base entries by an additional 15–30 % over exact-match fingerprinting.

---

*© 2025 Datacraft | www.datacraft.co.ke*
