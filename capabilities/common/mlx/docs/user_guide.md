# MLX — Local ML Inference User Guide

## Overview

`mlx` provides Ollama-backed ML tools (score, classify, predict, summarize, extract) with 100% data sovereignty — no data leaves the server. All APG capabilities that declare `ml_tools` in their contracts use this meta-capability as the inference backend.

## Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull mistral:7b

# Set env var
export OLLAMA_BASE_URL=http://localhost:11434
```

## Core Tools

### Score
Returns a float 0–1 representing risk, quality, or likelihood based on a feature dict.

```python
from capabilities.common.mlx.service import MLCapability

ml = MLCapability(ollama_url="http://localhost:11434", model="mistral:7b")
result = await ml.score(
    {"amount": 50000, "merchant_category": "gambling", "hour": 2},
    task_description="Score transaction fraud risk 0-1",
)
# MLScoreResult(score=0.87, confidence=0.91, factors=["high amount", "suspicious category"])
```

### Classify
Assigns a label from a provided set with confidence.

```python
result = await ml.classify("Invoice #1234 for software services", labels=["invoice", "contract", "email"])
# MLClassifyResult(label="invoice", confidence=0.94)
```

### Predict
Forecasts future values from a time series.

```python
result = await ml.predict([100, 110, 105, 115, 120], horizon=3)
# MLPredictResult(predictions=[{"step": 1, "value": 125}, ...])
```

### Summarize
Condenses text to key points.

```python
result = await ml.summarize(long_document_text, max_words=100)
# MLSummarizeResult(summary="...", key_points=["...", "..."])
```

### Extract
Pulls structured data from unstructured text given a JSON Schema.

```python
result = await ml.extract(
    "Patient John Doe, DOB 1985-03-15, admitted 2024-01-10 with diagnosis J18.9",
    schema={"name": "str", "dob": "str", "diagnosis": "str"},
)
# MLExtractResult(extracted={"name": "John Doe", "dob": "1985-03-15", "diagnosis": "J18.9"})
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/mlx/score` | Score features |
| POST | `/api/mlx/classify` | Classify text |
| POST | `/api/mlx/predict` | Predict from series |
| POST | `/api/mlx/summarize` | Summarize text |
| POST | `/api/mlx/extract` | Extract structured data |
| POST | `/api/mlx/embed` | Generate embeddings |
| POST | `/api/mlx/rank` | Rank documents by query |
| GET | `/api/mlx/models` | List available models |
| POST | `/api/mlx/models/{name}/pull` | Pull a model |
| GET | `/api/mlx/health` | Health check |
| GET | `/api/mlx/stats` | Inference statistics |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_DEFAULT_MODEL` | `mistral:7b` | Default model |

## Capability Integration

Capabilities declare ML tools in their contracts:

```python
# In any capability's service.py
from capabilities.common.mlx.service import MLCapability

async def ml_fraud_score(self, features: dict) -> float:
    if os.environ.get("OLLAMA_BASE_URL"):
        ml = MLCapability()
        result = await ml.score(features, task_description="fraud risk 0-1")
        return result.score
    return 0.5  # fallback
```
