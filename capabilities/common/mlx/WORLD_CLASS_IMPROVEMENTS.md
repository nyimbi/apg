# MLX — World-Class Improvement Plan

## Current State Audit

The existing service has solid bones: five core ML tools (score, classify, predict, summarize,
extract) backed by Ollama, streaming support, embedding/ranking, and convenience wrappers. But
there are correctness bugs, architectural gaps, and missing high-value capabilities.

---

## Improvements

### 1. Fix Broken Batch Methods (Correctness Bug)
`score_batch`, `classify_batch`, `predict_batch`, `summarize_batch`, `extract_batch` all pass
`model=` as a keyword to the underlying methods, which don't accept it. Same bug plagues
`score_transaction_risk`, `score_lead`, `predict_churn`, `predict_fraud`, `predict_readmission`
which call `self.score(features, task_description=...)` — the real param is `task`. These methods
silently fail or raise `TypeError` at runtime.

### 2. Concurrent Batch Execution via asyncio.gather
All five batch methods iterate serially with `for ... await`. On 10-item batches this serialises
~12 s of Ollama inference. Replace with `asyncio.gather(*coros)` behind a configurable
semaphore (`batch_concurrency: int = 4`) to cap simultaneous connections without flooding the
Ollama server.

### 3. Response Cache with TTL (BoundedCache)
`BoundedCache` is already imported from `capabilities.common.reliability` but never used.
Hash `(model, prompt_hash)` → result. Default TTL 300 s, max 512 entries. Cache hits reduce
latency for repeated identical requests (dashboards polling identical feature vectors).

### 4. Retry with Exponential Back-off
Cold-start model loads cause the first Ollama call to time out. Wrap `_generate` in a retry
loop: up to 3 attempts, back-off `[1, 2, 4]` s, only retry on `httpx.TimeoutException` and
`httpx.ConnectError`. Log each retry at DEBUG.

### 5. Latency & Token Tracking (Real Inference Stats)
`get_inference_stats` returns hardcoded zeros. Add instance-level counters:
`_total_calls`, `_total_latency_ms`, `_total_input_tokens`, `_total_output_tokens`. Ollama
`/api/generate` responses include `prompt_eval_count` and `eval_count`; capture them in
`_generate` and surface them in `MLBaseResult` and stats.

### 6. Multi-Label Classification
Binary/multi-class `classify` forces exactly one label. Many real tasks (document tagging,
compliance flags, content moderation) need multiple labels. Add `classify_multi_label(text,
labels, threshold=0.5)` → `MLMultiLabelResult` with a list of accepted labels each above the
threshold confidence.

### 7. Named Entity Recognition (NER)
`extract_entities` is a stub that calls `extract` with a malformed schema. Implement a proper
`ner(text, entity_types)` method that returns `List[MLEntity(text, type, start, end,
confidence)]`. Useful for PII detection, knowledge-graph construction, compliance scanning.

### 8. Zero-Shot Intent / Hypothesis Scoring
`classify` requires explicit label lists. `zero_shot_classify(text, hypothesis_template,
candidates)` uses NLI-style prompting ("Does this text support: {hypothesis}?") and returns
a ranked list with entailment probabilities. Enables policy-rule engines where labels are
natural-language descriptions.

### 9. Anomaly / Outlier Scoring
Given a baseline statistical description (mean, std, IQR, or exemplar list) and a new
observation, `anomaly_score(observation, baseline)` returns an anomaly score 0–1 and a
list of anomalous dimensions. Critical for fraud, network security, sensor data.

### 10. Structured Chain-of-Thought Scoring
Add `score_with_reasoning(features, task, rubric)` that forces the model to produce a
step-by-step reasoning chain before emitting a score. The rubric maps criteria → max_points;
the method returns `MLScorecardResult` with per-criterion scores summing to a final score.
Auditable, explainable — essential for credit, insurance, healthcare.

### 11. Topic Modelling / Keyword Extraction
`extract_topics(texts, n_topics)` and `extract_keywords(text, n)` — models are already
capable of soft topic assignment given a corpus snippet. Returns dominant topics with
representative terms and per-document topic weights. Feeds search indexing and content routing.

### 12. Language Detection and Translation
`detect_language(text)` returns ISO-639-1 code + confidence. `translate(text, target_lang)`
leverages multilingual Ollama models (mistral, aya, llama3). Africa-facing enterprise software
handles Swahili, Amharic, Hausa alongside English — this is not optional.

### 13. Semantic Chunking for Long Documents
`summarize` truncates at 4000 chars. `summarize_long(text, chunk_size, overlap)` splits on
semantic boundaries (sentence endings near chunk_size), summarises each chunk, then
hierarchically merges summaries. Handles 100-page PDFs without context-window truncation.

### 14. Embedding Batch API with Numpy-compatible Output
Current `embed` joins a list to a single string — semantically wrong for batch embeddings.
Fix to call `/api/embeddings` once per text, gather concurrently, return a `list[list[float]]`
with an optional `as_numpy=True` path. Add `cosine_similarity_matrix(texts)` returning a
symmetric N×N similarity matrix useful for clustering and duplicate detection.

### 15. Model Router / Capability-Aware Dispatch
Different tasks benefit from different model sizes. `auto_route=True` constructor flag enables
the router: scoring/classification → fast small model (e.g. `phi3:mini`), extraction/
summarisation → medium model (e.g. `mistral:7b`), chain-of-thought → large model (e.g.
`llama3:70b`). Router reads `/api/tags` once at construction and selects the best available
model per task category, with a fallback chain. Eliminates the need for callers to reason about
model selection.
