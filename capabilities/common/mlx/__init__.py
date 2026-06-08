"""APG MLX — Ollama-backed ML meta-capability.

Exposes local open-weight AI models as typed tools that any APG capability
can declare in its contract. Runs entirely on locally-hosted Ollama —
no data leaves the deployment, satisfying data sovereignty requirements.

Tool types: classify, score, predict, summarize, extract

Usage in APG capability contracts::

    capability FraudDetection {
        contract: {
            ml_tools: [score_transaction_risk, classify_merchant_category];
            model: {provider: ollama, model: "mistral:7b"};
        };
    }

Direct usage::

    from capabilities.common.mlx import MLCapability
    ml = MLCapability(model="mistral:7b")
    result = await ml.score({"amount": 50000, "country": "KE"}, task="fraud_risk")
    print(result.score)       # 0.82
    print(result.rationale)   # "High amount + first transaction from this country"
"""
from .service import MLCapability
from .models import (
    MLToolType,
    MLScoreResult,
    MLClassifyResult,
    MLPredictResult,
    MLSummarizeResult,
    MLExtractResult,
)

__all__ = [
    "MLCapability",
    "MLToolType",
    "MLScoreResult",
    "MLClassifyResult",
    "MLPredictResult",
    "MLSummarizeResult",
    "MLExtractResult",
]
