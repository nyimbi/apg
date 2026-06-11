# NLPC — World-Class Improvement Catalogue

**Capability**: NLP Core (nlpc)
**Author**: Nyimbi Odero — Datacraft
**Date**: 2026-06-11

---

## 1. Streaming Inference via Ollama `/api/generate` SSE

Current `_ollama_summarise` and `_ollama_translate` accumulate the full response before returning.
Replace with Server-Sent Events streaming so callers receive incremental tokens for long texts.
Expose an `async_generator` variant on `text_summarisation` and `translate` that `yield`s token
chunks, enabling real-time UI streaming without buffering the entire model output.

---

## 2. Semantic Chunking Before Embedding

`embed_text` sends up to 4000 raw characters to Ollama.  For documents longer than the model
context window this silently truncates, producing low-quality embeddings.  Add
`chunk_and_embed(text, chunk_size, overlap)` that splits text into semantically coherent
sentence-boundary chunks, embeds each independently, then mean-pools or stores the chunks
separately with a parent document reference.  This is the prerequisite for production RAG.

---

## 3. Cosine Nearest-Neighbour Search Over Stored Embeddings

`_STORE` holds embeddings but there is no retrieval path.  Add
`semantic_search(query, top_k, threshold)` that embeds the query then scans stored vectors
with cosine similarity, returning the top-k document IDs and scores.  Wire in an optional
`faiss`/`usearch` index when the package is present for sub-millisecond retrieval at scale.

---

## 4. PII Detection and Redaction

The model schema includes `NLPTask.PII_DETECTION` but the service has no implementation.
Add `detect_pii(text)` using a regex battery (email, phone, national ID, IBAN, credit card, IP)
plus spaCy PERSON/ORG labels, and `redact_pii(text, strategy)` that replaces detected spans
with `[REDACTED]`, `[TYPE]`, or a random substitute.  Required for GDPR compliance before
any text leaves the tenant boundary.

---

## 5. Dependency Parsing and Constituency Tree

`NLPTask.DEPENDENCY_PARSING` is defined but not implemented.  Add
`dependency_parse(text)` that returns token-level head/dep/pos triples from spaCy (or a
rule-based fallback), and a lightweight constituency approximation using noun-phrase chunking.
This unblocks grammar-error detection, argument mining, and structured relation extraction.

---

## 6. Temporal Expression Extraction (TIMEX3)

Named entity recognition skips temporal expressions beyond DATE/TIME labels.
Add `extract_temporal_expressions(text)` that normalises date/time strings to ISO-8601 using
`dateutil.parser` (available in most Python envs) and attaches TIMEX3-style attributes
(type: DATE|TIME|DURATION|SET, value, anchor).  Essential for event timelines in legal and
financial documents.

---

## 7. Multi-Label Document Classification

`classify_document` returns a single best label (argmax over scores).  Add
`multi_label_classify(text, taxonomy, threshold)` that returns all labels whose score exceeds
`threshold`, supporting overlapping categories (e.g. a document that is both LEGAL and
FINANCIAL).  Also expose a calibrated Platt-scaling post-processor for transformer logits.

---

## 8. Cross-Lingual NER via mBERT/XLM-R

`extract_entities` only loads English spaCy models.  Add
`multilingual_ner(text, language)` that selects the appropriate spaCy language model (if
installed) or falls back to an Ollama prompt in the detected language.  For African languages
without spaCy coverage, use a few-shot Ollama prompt with Swahili/Amharic/Hausa exemplars.

---

## 9. Argument Mining and Claim Detection

Add `extract_arguments(text)` that identifies claims, premises, and evidence spans using
sentence-level zero-shot classification with labels ["claim", "premise", "evidence",
"background"].  Output includes an argument graph: each claim linked to its supporting
premises, enabling fact-checking pipelines and debate analysis.

---

## 10. Confidence-Aware Caching with TTL

Every hot path (sentiment, entity, language) recomputes results for the same text on every
call.  Add a `BoundedCache` (already imported from `capabilities.common.reliability`) keyed on
`sha256(text + method + model_params)` with a configurable TTL (default 5 minutes).  Cache
misses transparently invoke the real backend; cache hits skip inference and log a cache-hit
event.  This will eliminate >80% of redundant model calls in typical usage patterns.

---

## 11. Async Batch Parallelism via `asyncio.gather`

`run_batch_job` iterates documents sequentially.  Replace the inner loop with
`asyncio.gather(*[self._dispatch_task(task, doc) for doc in docs])` with a configurable
concurrency semaphore (`asyncio.Semaphore(max_concurrent)`).  For 100 documents this reduces
wall-clock batch time by 10-50x depending on Ollama throughput.

---

## 12. Grammatical Error Correction

Add `correct_grammar(text, language)` using a LanguageTool REST API call (self-hosted via
Docker) with httpx, falling back to a rule-based heuristic that catches common error classes
(double spaces, missing capitalisation after period, repeated words, common homophone swaps).
Store corrections as a diff structure (offset, original, corrected, rule_id).

---

## 13. Discourse and Coherence Scoring

Add `score_coherence(text)` that measures local coherence (entity-grid model: proportion of
entity-grid transitions that are CONTINUATIONs vs SHIFTs) and global coherence (sentence
embedding cosine similarity with adjacent-sentence smoothing).  Returns a scalar [0, 1] and a
per-sentence coherence breakdown.  Enables document quality scoring for generated content.

---

## 14. African Language Model Registry

`_refine_african_language` covers Swahili via a fixed word list.  Replace with a full
character-n-gram classifier trained on 40+ African language corpora (Kinyarwanda, Amharic,
Hausa, Yoruba, Zulu, etc.) stored as a compact serialised sklearn `LinearSVC` or a
`fasttext` model.  This lifts African language detection F1 from ~40% (langdetect) to >90%
on benchmarks like AfriSenti/MasakhaNER.

---

## 15. Audit-Trail Event Bus Integration

Currently `_emit_event` appends to `self._events` (in-memory list, lost on service restart).
Wire events to a lightweight async publisher that writes to a PostgreSQL `nlpc_domain_events`
table (or a Redis Streams key) using an `asyncpg` pool.  Events should include `event_id`,
`tenant_id`, `actor_id`, `event_type`, `payload` (JSONB), `created_at`, and `correlation_id`.
This is the prerequisite for AUDL adapter compliance and replay-based debugging.
