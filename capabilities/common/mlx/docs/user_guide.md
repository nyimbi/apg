# MLX — Local ML Inference User Guide

## Overview

`mlx` provides Ollama-backed ML tools with 100% data sovereignty — no data leaves the server.
All APG capabilities that declare `ml_tools` in their contracts use this meta-capability as
the inference backend. v1.1 adds 11 extended tools covering multi-label classification, NER,
zero-shot scoring, anomaly detection, chain-of-thought rubric scoring, keyword extraction,
language detection, translation, long-document summarisation, and efficient batch embeddings.

## Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a general model
ollama pull mistral:7b

# Pull a multilingual model (for translation, language detection)
ollama pull aya:8b

# Pull an embedding model
ollama pull nomic-embed-text

# Set env vars (optional — defaults work out of the box)
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=mistral:7b
```

## Constructor

```python
from capabilities.common.mlx.service import MLCapability

ml = MLCapability(
    model="mistral:7b",           # Ollama model tag; env OLLAMA_MODEL overrides
    ollama_url="http://localhost:11434",  # env OLLAMA_BASE_URL overrides
    batch_concurrency=4,          # Max simultaneous Ollama calls for batch ops
    cache_ttl=300,                # TTL in seconds for in-process result cache; 0 = disabled
    auto_route=False,             # Route tasks to best available model size
)
```

## Core Tools

### Score
Returns a float 0–1 representing risk, quality, or likelihood based on a feature dict.

```python
result = await ml.score(
    features={"amount": 50000, "merchant_category": "gambling", "hour": 2},
    task="transaction_fraud_risk",
    context="East African mobile money platform",
)
# MLScoreResult(score=0.87, confidence=0.91, factors=["high amount", "suspicious category"])
print(result.score, result.factors)
```

### Classify
Assigns exactly one label from a provided set with per-label probabilities.

```python
result = await ml.classify(
    "Invoice #1234 for software services",
    labels=["invoice", "contract", "email"],
)
# MLClassifyResult(label="invoice", confidence=0.94)
```

### Predict
Forecasts future values from a time series.

```python
result = await ml.predict(
    series=[{"period": "Jan", "value": 100}, {"period": "Feb", "value": 110}],
    horizon=3,
    task="monthly_sales_forecast",
)
# MLPredictResult(predictions=[{"period": "Mar", "value": 115, "lower": 105, "upper": 125}])
```

### Summarize
Condenses text to key points.

```python
result = await ml.summarize(long_document_text, max_words=100, focus="compliance issues")
# MLSummarizeResult(summary="...", key_points=["...", "..."])
```

### Extract
Pulls structured data from unstructured text given a field schema.

```python
result = await ml.extract(
    "Patient John Doe, DOB 1985-03-15, admitted 2024-01-10 with diagnosis J18.9",
    schema={"name": "patient full name", "dob": "date of birth ISO format", "diagnosis": "ICD-10 code"},
    context="clinical admission notes",
)
# MLExtractResult(extracted={"name": "John Doe", "dob": "1985-03-15", "diagnosis": "J18.9"})
```

---

## Extended Tools (v1.1)

### Multi-Label Classification

Assign zero or more labels that exceed a confidence threshold.
Useful for document tagging, compliance flags, content moderation.

```python
result = await ml.classify_multi_label(
    text="This email discusses overdue payment and threatens legal action.",
    labels=["overdue", "legal_risk", "fraud", "routine", "escalation"],
    threshold=0.5,
    context="accounts receivable",
)
# MLMultiLabelResult(labels=["overdue", "legal_risk", "escalation"])
print(result.probabilities)  # {"overdue": 0.91, "legal_risk": 0.78, ...}
```

### Named Entity Recognition (NER)

Extract typed entity spans from text. Critical for PII detection, knowledge graphs,
compliance scanning.

```python
result = await ml.ner(
    text="Wanjiku Kamau from Nairobi visited Dr. Ochieng at KNH on 12 Jan 2025.",
    entity_types=["PERSON", "LOCATION", "DATE", "ORG"],
    context="clinical notes",
)
for entity in result.entities:
    print(entity.entity_type, entity.text, entity.confidence)
# PERSON  Wanjiku Kamau  0.97
# PERSON  Dr. Ochieng    0.95
# LOCATION  Nairobi      0.92
# DATE  12 Jan 2025      0.99
# ORG  KNH               0.88
```

### Zero-Shot Classification

Score natural-language hypotheses against a text using NLI-style entailment.
Enables policy engines where labels are human-readable descriptions.

```python
result = await ml.zero_shot_classify(
    text="Customer called three times to dispute charges without resolution.",
    candidates=["customer is frustrated", "potential churn risk", "fraud indicator", "routine inquiry"],
    hypothesis_template="This interaction indicates {label}.",
)
# MLZeroShotResult(top_label="potential churn risk", top_score=0.88)
for item in result.ranked:
    print(item["label"], item["score"])
```

### Anomaly / Outlier Scoring

Score a new observation against a statistical baseline description.
Returns an anomaly score 0–1 and the dimensions driving the anomaly.

```python
result = await ml.anomaly_score(
    observation={"amount": 95000, "hour": 3, "location": "foreign_country"},
    baseline={
        "amount": "mean=500 std=200 max_normal=5000",
        "hour": "typically 9-17 local time",
        "location": "domestic transactions only",
    },
    context="mobile money fraud detection",
)
# MLAnomalyResult(anomaly_score=0.94, anomalous_dimensions=["amount", "hour", "location"])
```

### Chain-of-Thought Rubric Scoring

Structured, explainable scoring against a multi-criterion rubric.
Produces per-criterion scores plus a full reasoning chain — required for credit,
insurance, and healthcare decision support.

```python
result = await ml.score_with_reasoning(
    features={"payment_history": "3 late payments", "utilization": 0.45, "age_months": 24},
    task="credit_risk_assessment",
    rubric={
        "payment_history": 35.0,
        "credit_utilization": 30.0,
        "credit_age": 15.0,
        "credit_mix": 10.0,
        "new_credit": 10.0,
    },
    context="consumer credit scoring",
)
print(result.normalized_score)   # e.g. 0.63
print(result.reasoning_chain)    # step-by-step model reasoning
for c in result.criteria:
    print(f"  {c.criterion}: {c.score}/{c.max_score} — {c.reasoning}")
```

### Keyword and Topic Extraction

```python
result = await ml.extract_keywords(
    text=long_article,
    n_keywords=10,
    n_topics=3,
    context="financial news",
)
# MLKeywordResult(keywords=["inflation", "CBK", "rate hike", ...], topics=["monetary policy", ...])
```

### Language Detection

Returns ISO-639-1 code and confidence. Essential for Africa-facing software handling
Swahili, Amharic, Hausa, French, and Arabic alongside English.

```python
result = await ml.detect_language("Habari yako? Niko sawa kabisa.")
# MLLanguageResult(language_code="sw", language_name="Swahili", confidence=0.97)

result = await ml.detect_language("Inọ ọ dị mma.")
# MLLanguageResult(language_code="ig", language_name="Igbo", confidence=0.84)
```

### Translation

Leverages multilingual Ollama models (mistral, aya, llama3). No API keys.

```python
result = await ml.translate("Karibu Kenya", target_language="English")
# MLTranslationResult(translated_text="Welcome to Kenya", source_language="sw")

result = await ml.translate(
    "المدفوعة المتأخرة",
    target_language="English",
    source_language="Arabic",
)
```

### Long-Document Hierarchical Summarisation

Splits on sentence boundaries, summarises chunks concurrently, then merges.
Handles 100-page PDFs without silent 4000-character truncation.

```python
result = await ml.summarize_long(
    text=full_100_page_report,
    chunk_size=3000,     # characters per chunk
    overlap=200,         # context overlap between chunks
    max_words=300,       # final summary word budget
    focus="risk factors and regulatory implications",
)
# MLSummarizeResult with hierarchical rationale and deduplicated key_points
```

### Concurrent Batch Embeddings

Generates one embedding per text, all concurrently. Fixes the v1.0 bug where a
list of texts was joined to a single string.

```python
vectors = await ml.embed_batch(["text A", "text B", "text C"], model="nomic-embed-text")
# [[0.12, -0.34, ...], [0.56, 0.01, ...], [-0.23, 0.78, ...]]
```

### Cosine Similarity Matrix

N×N symmetric matrix for a corpus — useful for clustering and duplicate detection.

```python
matrix = await ml.cosine_similarity_matrix(
    ["invoice for services", "payment receipt", "invoice overdue notice"],
    model="nomic-embed-text",
)
# matrix[0][2] = 0.91  (invoice variants are similar)
# matrix[0][1] = 0.47  (invoice vs. receipt less similar)
```

---

## Batch Operations

All batch methods execute concurrently via `asyncio.gather` with a semaphore:

```python
# Concurrent score batch — (features_dict, task_string) pairs
scores = await ml.score_batch([
    ({"amount": 1000}, "fraud_risk"),
    ({"amount": 80000, "hour": 3}, "fraud_risk"),
    ({"amount": 500, "hour": 14}, "fraud_risk"),
], batch_concurrency=3)

# Concurrent summarise batch
summaries = await ml.summarize_batch(["doc1...", "doc2...", "doc3..."], max_words=80)

# Concurrent extract batch
extractions = await ml.extract_batch(
    ["invoice text 1", "invoice text 2"],
    schema={"vendor": "supplier name", "amount": "invoice total", "date": "invoice date"},
)
```

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/mlx/score` | Score features |
| POST | `/api/mlx/classify` | Single-label classification |
| POST | `/api/mlx/classify/multi` | Multi-label classification |
| POST | `/api/mlx/predict` | Forecast from series |
| POST | `/api/mlx/summarize` | Summarize text |
| POST | `/api/mlx/summarize/long` | Hierarchical long-doc summarise |
| POST | `/api/mlx/extract` | Extract structured fields |
| POST | `/api/mlx/ner` | Named entity recognition |
| POST | `/api/mlx/zero-shot` | Zero-shot hypothesis scoring |
| POST | `/api/mlx/anomaly` | Anomaly/outlier scoring |
| POST | `/api/mlx/score-rubric` | Chain-of-thought rubric scoring |
| POST | `/api/mlx/keywords` | Keyword + topic extraction |
| POST | `/api/mlx/detect-language` | Language detection |
| POST | `/api/mlx/translate` | Translation |
| POST | `/api/mlx/embed` | Single-text embedding |
| POST | `/api/mlx/embed/batch` | Concurrent batch embeddings |
| POST | `/api/mlx/similarity-matrix` | N×N cosine similarity matrix |
| POST | `/api/mlx/rank` | Rank documents by query |
| GET | `/api/mlx/models` | List available models |
| POST | `/api/mlx/models/{name}/pull` | Pull a model |
| GET | `/api/mlx/health` | Health check |
| GET | `/api/mlx/stats` | Inference statistics |

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral:7b` | Default inference model |

---

## Capability Integration

Capabilities declare ML tools in their contracts and delegate to this meta-capability:

```python
from capabilities.common.mlx.service import MLCapability

class MyCapabilityService:
    def __init__(self):
        self._ml = MLCapability()

    async def assess_risk(self, features: dict) -> float:
        result = await self._ml.score(features, task="credit_risk")
        return result.score

    async def classify_incoming_doc(self, text: str) -> str:
        result = await self._ml.classify(text, ["invoice", "contract", "report", "email"])
        return result.label

    async def flag_compliance_issues(self, text: str) -> list[str]:
        result = await self._ml.classify_multi_label(
            text,
            labels=["aml_risk", "data_privacy", "regulatory_breach", "sanctions_hit"],
            threshold=0.6,
        )
        return result.labels
```

---

## Performance Notes

- First call to a model triggers Ollama cold-start (model load). Subsequent calls hit
  the warm model and are significantly faster. Use `warm_up_model()` in app startup.
- The in-process TTL cache (`cache_ttl=300`) eliminates redundant Ollama calls for
  identical (model, prompt) pairs — useful for dashboards polling the same feature vectors.
- `batch_concurrency=4` is a safe default for a single Ollama instance. Increase to 8–16
  if Ollama is running on a machine with sufficient VRAM/RAM.
- `summarize_long` issues chunk summaries concurrently — a 50-chunk document takes roughly
  the same wall-clock time as 1 chunk divided by `batch_concurrency`.
