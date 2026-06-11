# NLP Core (nlpc) — World-Class Improvements

**Capability**: NLP Core | **Path**: capabilities/common/nlpc | **Domain**: common
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

### I1. Streaming Incremental NLP via Server-Sent Events

**Category**: Throughput / UX
**Justification**: Latency-to-first-token dominates perceived performance. Current `_ollama_summarise` and `_ollama_translate` block the caller until the full Ollama response arrives. SSE streaming lets the UI render tokens as they arrive, cutting perceived latency 5-10x for long-form generation — matching ChatGPT UX and outperforming any batch-only NLP endpoint.
**Implementation**: Add `stream=True` to Ollama `/api/generate` requests; yield `data:` SSE frames from an async generator `stream_summarise`. Flask route layer wraps the generator in `Response(stream_with_context(...))`.
**Competitor**: OpenAI Streaming API, Cohere Streaming, Anthropic SSE

---

### I2. Cross-Lingual Semantic Search with Multilingual Embeddings

**Category**: Search / Multilingual
**Justification**: Current `semantic_search` only embeds in the query language; a Swahili document cannot be retrieved by an English query. Multilingual embeddings collapse 100+ languages into a shared vector space, enabling 40+ African-language corpora to be searched in English — a feature no incumbent NLP API exposes for Swahili/Kikuyu/Amharic at this depth.
**Implementation**: Add `multilingual_embed_text` using `sentence-transformers` `paraphrase-multilingual-mpnet-base-v2`; expose `cross_lingual_search(query, query_lang, target_lang)` that cosine-ranks stored multilingual embeddings.
**Competitor**: Cohere Multilingual, Azure AI Search, Weaviate hybrid search

---

### I3. Readability and Complexity Scoring (Flesch-Kincaid + Gunning Fog)

**Category**: Text Quality / Compliance Analytics
**Justification**: Legal, financial, and health communications require measurable readability. Flesch-Kincaid Grade Level, Gunning Fog, and Coleman-Liau are pure arithmetic over syllable/word/sentence counts — zero model dependency, sub-millisecond, delivering compliance-grade evidence for regulatory content reviews. Grammarly charges enterprise licences for this; implemented here as `score_readability`.
**Implementation**: Pure-Python `score_readability(text)` computing FK grade, Fog, CL index, and composite plain-language score. Syllable counting via vowel-cluster heuristic; no external dependencies. Already implemented in service.py.
**Competitor**: Grammarly Business, Microsoft Editor, ProWritingAid

---

### I4. Fine-Grained Emotion Detection (Ekman 8-class)

**Category**: Sentiment / Emotion Intelligence
**Justification**: Binary pos/neg sentiment is table-stakes. B2B platforms offer 8 Ekman emotion axes (joy, anger, fear, disgust, surprise, sadness, anticipation, trust). Routing a support ticket tagged `anger+fear` vs `frustration+trust` produces materially different CX outcomes. Current `sentiment_analysis` only returns pos/neg/neutral.
**Implementation**: `detect_emotions(text)` — zero-shot classify against 8-label Ekman set via `facebook/bart-large-mnli`; fallback to NRC Emotion Lexicon word lists embedded as Python dict. Returns per-emotion scores + dominant emotion + VAD axes. Already implemented in service.py.
**Competitor**: IBM Watson Tone Analyzer, AWS Comprehend, Symanto Emotion

---

### I5. Concept Extraction and Ontology Grounding

**Category**: Knowledge Graph / NLP
**Justification**: Named entities resolve to Wikipedia slugs (current NEL). Concepts are broader — "machine learning" is a concept, not a named entity. Concept extraction enables knowledge-graph construction and content recommendation without curating entity lists.
**Implementation**: `extract_concepts(text)` — spaCy noun-chunk pipeline + Wikidata Qnode resolution via `httpx` against `wikidata.org/w/api.php`. Falls back to noun-phrase TF-IDF with BM25 ranking. Returns `{concept, qnode, category, confidence}` per concept.
**Competitor**: Google Natural Language API, Aylien, Dandelion API

---

### I6. Automatic Document Structure Detection

**Category**: Document Intelligence
**Justification**: Unstructured text from PDFs or HTML has implicit structure (headings, bullet lists, tables). Current `create_document` stores raw content with no structural metadata. Structural awareness unlocks segment-level NLP and improves search precision. Matches AWS Textract without a cloud dependency.
**Implementation**: `detect_document_structure(text)` — regex + indentation heuristics classify spans as `heading`, `paragraph`, `list_item`, `table_row`, `code_block`. Returns `{segments, structure_score, heading_count, list_item_count}`. Already implemented in service.py.
**Competitor**: AWS Textract, Google Document AI, unstructured.io

---

### I7. Hallucination Detection for Generated Text

**Category**: AI Safety / Governance
**Justification**: When `text_generation` or `_ollama_summarise` produces output, there is no faithfulness check against the source document. RAG deployments need a faithfulness signal before surfacing generated text to users. Multi-billion-dollar problem — Microsoft Copilot, Google NotebookLM invest here.
**Implementation**: `score_faithfulness(source, generated)` — NLI entailment score via `cross-encoder/nli-deberta-v3-small`; falls back to token-overlap ROUGE-L. Returns `{entailment_score, contradiction_score, faithfulness_label, rouge_l}`. Already implemented in service.py.
**Competitor**: Vectara HHEM, Galileo Hallucination Index, Azure AI Content Safety

---

### I8. Document Deduplication with MinHash LSH

**Category**: Data Quality / Scalability
**Justification**: In high-volume ingestion pipelines (news feeds, customer emails), near-duplicate content degrades model training and biases analytics. MinHash LSH detects near-duplicates in O(n) amortised vs. O(n^2) pairwise Jaccard. Current service has zero deduplication logic.
**Implementation**: `find_near_duplicates(document_ids, threshold=0.8)` — char 3-gram shingle sets, 128-band MinHash signatures, LSH bucketing, returns candidate pairs with estimated and exact Jaccard. Pure Python using `hashlib`. Already implemented in service.py.
**Competitor**: DataRobot Data Prep, AWS Glue dedup, Dedupe.io

---

### I9. Adaptive Confidence Calibration via Temperature Scaling

**Category**: ML Ops / Model Reliability
**Justification**: Raw model softmax probabilities are miscalibrated (ECE > 0.1 for most transformers). The service exposes raw scores as `confidence` without calibration — a compliance liability in regulated use cases. Temperature scaling reduces ECE by ~5x with no accuracy loss, making confidence scores legally defensible.
**Implementation**: `calibrate_confidence(raw_score, task, n_buckets=10)` — stored temperature parameters per task type in `model_registry.py` as constants; calibrated scores replace raw scores in response models.
**Competitor**: Google AutoML Tables calibration, Platt scaling in scikit-learn

---

### I10. Caching Layer with Content-Addressed Results (SHA-256 + TTL)

**Category**: Performance / Cost Reduction
**Justification**: `detect_language`, `sentiment_analysis`, and `embed_text` are pure functions of input text. Re-running on identical text wastes compute. A content-addressed cache keyed on `sha256(tenant_id + task + text)` eliminates redundant inference. At 60% cache hit rate (realistic for support corpora), this halves infrastructure costs.
**Implementation**: Wrap the five most expensive methods with a `@cached_nlp_result` decorator reading/writing `BoundedCache` (imported from `capabilities.common.reliability`) keyed on SHA-256 of (tenant_id, task_name, text[:4096]).
**Competitor**: Cohere caching, OpenAI prompt caching, Anthropic prompt cache

---

### I11. Discourse Segmentation and Rhetorical Structure Theory (RST)

**Category**: Discourse Analysis
**Justification**: RST provides a tree structure that enables structured summarisation respecting document intent — nucleus-satellite, elaboration, contrast. Current `extract_arguments` uses sentence-level zero-shot classification. RST is the foundation for document-level QA and structured summarisation of legal and academic documents.
**Implementation**: `segment_discourse(text)` — hierarchical EDU segmentation using cue-phrase detection + dependency parse head chains. Returns `{edus, depth, root_relation}`. Integrates with `dependency_parse` output.
**Competitor**: RST-DT parsers (CODRA, DPLP), AllenNLP discourse

---

### I12. Privacy-Preserving Federated NLP Analytics

**Category**: Privacy / Multi-Tenant Governance
**Justification**: Enterprise tenants are blocked from pooling data for model improvement by data-residency laws. Federated learning aggregates model updates (not raw text) across tenants, providing better language models without cross-tenant data exposure — differentiator for GDPR/CCPA markets.
**Implementation**: `aggregate_federated_lexicon(tenant_updates: list[dict])` — weighted averaging of per-tenant term frequency deltas into a global pseudo-IDF corpus. Each tenant submits `{term: count}` dicts (no raw text). Improves TF-IDF scoring across `extract_key_phrases` and `score_coherence`.
**Competitor**: Google Federated Learning, PySyft, TensorFlow Federated

---

### I13. Semantic Role Labelling (SRL) for Predicate-Argument Structures

**Category**: Deep Linguistics / Event Extraction
**Justification**: Relation extraction captures binary SVO triples. SRL captures full predicate frames: agent, patient, instrument, location, time — enabling structured event extraction for intelligence and compliance use cases. AllenNLP SRL is the industry reference.
**Implementation**: `label_semantic_roles(text)` — spaCy dependency parse + VerbNet-style argument mapping heuristics. PropBank-style frames `{predicate, args: [{role, text, start, end}]}`. Falls back to SVO regex when spaCy parser unavailable. Already implemented in service.py.
**Competitor**: AllenNLP SRL, Hugging Face SRL models, SENNA

---

### I14. Adaptive Batch Scheduler with Priority Queuing and Back-Pressure

**Category**: Scalability / Operations
**Justification**: Current `run_batch_job` is sequential. A priority-queue scheduler with async semaphore-bounded workers and exponential-backoff retry converts batch throughput from O(docs x tasks) serial to O(max_concurrent) parallel. Critical for SLAs on large corpora (10k+ documents).
**Implementation**: `run_batch_job_scheduled(job_id, max_workers=8, retry_limit=3)` — `asyncio.PriorityQueue` keyed on `(priority, enqueue_time)`; workers claim one (doc, task) pair at a time; failed tasks re-enqueue with backoff; progress updates after each worker cycle. Already implemented in service.py.
**Competitor**: Celery, Prefect, Apache Airflow task scheduling

---

### I15. Multi-Hop Question Answering with Evidence Chain

**Category**: Information Retrieval / QA
**Justification**: Current `question_answering` is single-hop extractive QA over a single context. Complex questions require chaining evidence across multiple retrieved passages. Multi-hop QA powers Google MUM and Microsoft Sydney. For intelligence and legal domains, a traceable evidence chain is a compliance requirement.
**Implementation**: `multi_hop_qa(question, document_ids, max_hops=3)` — (1) embed question, retrieve top-k passages via `semantic_search`; (2) extract answer span; (3) if answer contains a named entity, re-embed as follow-up query and repeat up to `max_hops`; (4) return `{answer, evidence_chain, confidence}`. Already implemented in service.py.
**Competitor**: HotpotQA systems, DeepMind REALM, ColBERT multi-hop
