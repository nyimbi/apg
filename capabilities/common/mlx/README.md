# APG ML Meta-Capability (Ollama) (`mlx`)

**Version**: 1.1.0 | **Domain**: common

## Overview

Ollama-backed ML tools: score, classify, predict, summarize, extract — plus 11 extended
tools added in v1.1. All inference runs locally via the Ollama REST API. No data leaves
the server — 100% data sovereignty for regulated industries.

## Core Tools (v1.0)

| Tool | Method | Returns |
|------|--------|---------|
| Risk/quality scoring | `score(features, task)` | `MLScoreResult` |
| Single-label classification | `classify(text, labels)` | `MLClassifyResult` |
| Time-series forecasting | `predict(series, horizon)` | `MLPredictResult` |
| Text summarisation | `summarize(text)` | `MLSummarizeResult` |
| Structured field extraction | `extract(text, schema)` | `MLExtractResult` |

## Extended Tools (v1.1)

| Tool | Method | Returns |
|------|--------|---------|
| Multi-label classification | `classify_multi_label(text, labels, threshold)` | `MLMultiLabelResult` |
| Named entity recognition | `ner(text, entity_types)` | `MLNERResult` |
| Zero-shot / NLI classification | `zero_shot_classify(text, candidates)` | `MLZeroShotResult` |
| Anomaly / outlier detection | `anomaly_score(observation, baseline)` | `MLAnomalyResult` |
| Chain-of-thought rubric scoring | `score_with_reasoning(features, task, rubric)` | `MLScorecardResult` |
| Keyword + topic extraction | `extract_keywords(text)` | `MLKeywordResult` |
| Language detection | `detect_language(text)` | `MLLanguageResult` |
| Multilingual translation | `translate(text, target_language)` | `MLTranslationResult` |
| Long-doc hierarchical summarisation | `summarize_long(text)` | `MLSummarizeResult` |
| Concurrent batch embeddings | `embed_batch(texts)` | `list[list[float]]` |
| Corpus similarity matrix | `cosine_similarity_matrix(texts)` | `list[list[float]]` |

## Quick Start

```python
from capabilities.common.mlx.service import MLCapability

ml = MLCapability(model="mistral:7b")

# Score fraud risk
result = await ml.score({"amount": 50000, "hour": 2}, task="transaction_fraud_risk")
# MLScoreResult(score=0.87, confidence=0.91)

# Multi-label document tagging
tags = await ml.classify_multi_label(
    "Invoice overdue, compliance violation noted",
    labels=["overdue", "compliance", "fraud", "routine"],
    threshold=0.5,
)
# MLMultiLabelResult(labels=["overdue", "compliance"])

# Language detection
lang = await ml.detect_language("Habari yako? Niko sawa.")
# MLLanguageResult(language_code="sw", language_name="Swahili", confidence=0.96)

# Translate to English
translated = await ml.translate("Habari yako? Niko sawa.", target_language="English")
# MLTranslationResult(translated_text="How are you? I am fine.")

# Anomaly detection
anomaly = await ml.anomaly_score(
    observation={"amount": 95000, "hour": 3},
    baseline={"amount": "mean=500, std=200", "hour": "typically 9-17"},
)
# MLAnomalyResult(anomaly_score=0.92, anomalous_dimensions=["amount", "hour"])
```

## Constructor Options

```python
ml = MLCapability(
    model="mistral:7b",          # Ollama model tag
    ollama_url="http://localhost:11434",
    batch_concurrency=4,         # Max simultaneous Ollama calls for batch ops
    cache_ttl=300,               # Result cache TTL in seconds; 0 disables
    auto_route=False,            # Route tasks to best available model size
)
```

## Concurrent Batch Operations

All batch methods use `asyncio.gather` behind a configurable semaphore:

```python
results = await ml.score_batch([
    ({"amount": 1000, "hour": 10}, "fraud_risk"),
    ({"amount": 80000, "hour": 3}, "fraud_risk"),
])

summaries = await ml.summarize_batch(["doc1 text...", "doc2 text..."], max_words=80)
```

## Governance Rules

- tenant_context_required
- operation_type_required
- audit_logged
- access_controlled

## License

© 2025 Datacraft | nyimbi@gmail.com | www.datacraft.co.ke
