"""Tests for APG MLX — Ollama-backed ML meta-capability.

Tests validate the tool interfaces, JSON parsing, result shapes,
and error handling — without requiring a live Ollama server.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from capabilities.common.mlx import (
	MLCapability,
	MLClassifyResult,
	MLExtractResult,
	MLPredictResult,
	MLScoreResult,
	MLSummarizeResult,
	MLToolType,
)


def make_ml(model: str = "mistral:7b") -> MLCapability:
	return MLCapability(model=model, ollama_url="http://localhost:11434")


# ── MLCapability.score ────────────────────────────────────────────────────────

async def test_score_returns_typed_result():
	ml = make_ml()
	mock_response = json.dumps({
		"score": 0.82,
		"confidence": 0.75,
		"factors": ["high amount", "new country"],
		"rationale": "First transaction from this country with high amount.",
	})
	with patch.object(ml, "_generate", new=AsyncMock(return_value=mock_response)):
		result = await ml.score(
			{"amount": 50000, "country": "KE", "is_first_tx": True},
			task="fraud_risk",
		)

	assert isinstance(result, MLScoreResult)
	assert result.tool_type == MLToolType.score
	assert result.score == pytest.approx(0.82)
	assert result.confidence == pytest.approx(0.75)
	assert "high amount" in result.factors
	assert result.model == "mistral:7b"
	assert result.latency_ms >= 0


async def test_score_clamps_out_of_range_values():
	ml = make_ml()
	with patch.object(ml, "_generate", new=AsyncMock(return_value='{"score": 1.5, "confidence": -0.3}')):
		result = await ml.score({"x": 1}, task="test")

	assert 0.0 <= result.score <= 1.0
	assert 0.0 <= result.confidence <= 1.0


async def test_score_handles_malformed_json():
	ml = make_ml()
	with patch.object(ml, "_generate", new=AsyncMock(return_value="not json")):
		result = await ml.score({"x": 1}, task="test")

	# Returns default values, does not raise
	assert isinstance(result, MLScoreResult)
	assert 0.0 <= result.score <= 1.0


# ── MLCapability.classify ─────────────────────────────────────────────────────

async def test_classify_returns_typed_result():
	ml = make_ml()
	mock_response = json.dumps({
		"label": "electronics",
		"confidence": 0.91,
		"probabilities": {"electronics": 0.91, "clothing": 0.05, "food": 0.04},
		"rationale": "Product description matches electronics category.",
	})
	with patch.object(ml, "_generate", new=AsyncMock(return_value=mock_response)):
		result = await ml.classify(
			"Samsung 65-inch QLED TV",
			labels=["electronics", "clothing", "food"],
		)

	assert isinstance(result, MLClassifyResult)
	assert result.label == "electronics"
	assert result.confidence == pytest.approx(0.91)
	assert "electronics" in result.probabilities
	assert result.tool_type == MLToolType.classify


async def test_classify_corrects_hallucinated_label():
	"""When model returns a label not in the list, select the closest."""
	ml = make_ml()
	with patch.object(ml, "_generate", new=AsyncMock(return_value='{"label": "Electronics!", "confidence": 0.8}')):
		result = await ml.classify("TV remote", labels=["electronics", "clothing", "food"])

	assert result.label in ["electronics", "clothing", "food"]


# ── MLCapability.predict ──────────────────────────────────────────────────────

async def test_predict_returns_typed_result():
	ml = make_ml()
	mock_response = json.dumps({
		"predictions": [
			{"period": "2025-07", "value": 125000, "lower": 110000, "upper": 140000},
			{"period": "2025-08", "value": 130000, "lower": 115000, "upper": 145000},
		],
		"confidence_interval": {"level": 0.95},
		"rationale": "Based on 12-month seasonal trend.",
	})
	series = [{"month": f"2025-{i:02d}", "value": 100000 + i * 5000} for i in range(1, 7)]

	with patch.object(ml, "_generate", new=AsyncMock(return_value=mock_response)):
		result = await ml.predict(series, horizon=2, task="monthly_revenue")

	assert isinstance(result, MLPredictResult)
	assert result.horizon == 2
	assert len(result.predictions) == 2
	assert result.predictions[0]["value"] == 125000
	assert result.tool_type == MLToolType.predict


# ── MLCapability.summarize ────────────────────────────────────────────────────

async def test_summarize_returns_typed_result():
	ml = make_ml()
	mock_response = json.dumps({
		"summary": "APG is a composable ERP platform with 261 capabilities.",
		"key_points": ["261 capabilities", "composable architecture", "Ollama AI integration"],
	})
	with patch.object(ml, "_generate", new=AsyncMock(return_value=mock_response)):
		result = await ml.summarize(
			"APG Platform provides a comprehensive suite of 261 business capabilities...",
			max_words=50,
		)

	assert isinstance(result, MLSummarizeResult)
	assert "261" in result.summary
	assert len(result.key_points) == 3
	assert result.word_count > 0
	assert result.tool_type == MLToolType.summarize


# ── MLCapability.extract ──────────────────────────────────────────────────────

async def test_extract_returns_typed_result():
	ml = make_ml()
	mock_response = json.dumps({
		"invoice_number": "INV-2025-001",
		"amount": 15000.50,
		"vendor_name": "Safaricom Ltd",
		"due_date": "2025-08-15",
	})
	schema = {
		"invoice_number": "Invoice or reference number",
		"amount": "Total amount due",
		"vendor_name": "Supplier or vendor name",
		"due_date": "Payment due date",
	}
	with patch.object(ml, "_generate", new=AsyncMock(return_value=mock_response)):
		result = await ml.extract(
			"Invoice INV-2025-001 from Safaricom Ltd for KES 15,000.50 due 2025-08-15",
			schema=schema,
		)

	assert isinstance(result, MLExtractResult)
	assert result.extracted["invoice_number"] == "INV-2025-001"
	assert result.extracted["amount"] == 15000.50
	assert "invoice_number" in result.fields_found
	assert result.fields_missing == []
	assert result.tool_type == MLToolType.extract


async def test_extract_reports_missing_fields():
	ml = make_ml()
	with patch.object(ml, "_generate", new=AsyncMock(return_value='{"invoice_number": "INV-001"}')):
		result = await ml.extract("Invoice INV-001", schema={
			"invoice_number": "Invoice number",
			"amount": "Total amount",
		})

	assert "invoice_number" in result.fields_found
	assert "amount" in result.fields_missing


# ── JSON parsing edge cases ───────────────────────────────────────────────────

def test_parse_json_handles_prose_prefix():
	"""Model sometimes prepends prose before the JSON block."""
	ml = make_ml()
	text = 'Sure! Here is the result:\n{"score": 0.7, "confidence": 0.8}'
	result = ml._parse_json(text, {})
	assert result["score"] == pytest.approx(0.7)


def test_parse_json_returns_default_on_empty():
	ml = make_ml()
	assert ml._parse_json("", {"score": 0.5}) == {"score": 0.5}


def test_parse_json_returns_default_on_invalid():
	ml = make_ml()
	assert ml._parse_json("this is not json", {"score": 0.5}) == {"score": 0.5}


# ── Ollama failure handling ───────────────────────────────────────────────────

async def test_score_returns_default_when_ollama_unreachable():
	"""Service degrades gracefully when Ollama is unavailable."""
	ml = make_ml()
	with patch.object(ml, "_generate", new=AsyncMock(return_value="{}")):
		result = await ml.score({"x": 1}, task="test")

	assert isinstance(result, MLScoreResult)
	assert result.score == pytest.approx(0.5)  # default


# ── Health check ──────────────────────────────────────────────────────────────

async def test_health_check_returns_true_when_model_available():
	ml = make_ml(model="mistral:7b")
	mock_resp = MagicMock()
	mock_resp.json.return_value = {
		"models": [{"name": "mistral:7b"}, {"name": "llama3:8b"}]
	}
	mock_resp.raise_for_status = MagicMock()

	with patch.object(ml._client, "get", new=AsyncMock(return_value=mock_resp)):
		assert await ml.health_check() is True


async def test_health_check_returns_false_when_model_unavailable():
	ml = make_ml(model="mistral:7b")
	mock_resp = MagicMock()
	mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
	mock_resp.raise_for_status = MagicMock()

	with patch.object(ml._client, "get", new=AsyncMock(return_value=mock_resp)):
		assert await ml.health_check() is False


async def test_health_check_returns_false_on_connection_error():
	ml = make_ml()
	with patch.object(ml._client, "get", new=AsyncMock(side_effect=Exception("Connection refused"))):
		assert await ml.health_check() is False
