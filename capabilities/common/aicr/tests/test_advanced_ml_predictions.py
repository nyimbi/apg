"""Focused tests for executable AICR advanced-ML prediction helpers."""

import pytest

from capabilities.common.aicr.advanced_ml import AdvancedMLEngine, ExplainabilityEngine


@pytest.mark.asyncio
async def test_advanced_ml_uses_registered_async_model():
	engine = AdvancedMLEngine()

	async def registered_model(payload):
		return {
			"prediction": "approved",
			"predictions": {"class": "approved", "confidence": 0.73},
			"confidence": 0.73
		}

	engine.active_models["credit_model"] = registered_model

	result = await engine._get_model_prediction("credit_model", {"amount": 125})

	assert result["prediction"] == "approved"
	assert result["predictions"]["class"] == "approved"
	assert result["confidence"] == 0.73
	assert result["model_source"] == "registered"


@pytest.mark.asyncio
async def test_advanced_ml_falls_back_to_deterministic_heuristic():
	engine = AdvancedMLEngine()

	result = await engine._get_model_prediction("missing_model", {"features": [0.8, 0.6, 0.7]})

	assert result["prediction"] == "positive"
	assert result["predictions"]["score"] == pytest.approx(0.7)
	assert result["confidence"] == pytest.approx(0.7)
	assert result["model_source"] == "heuristic"


@pytest.mark.asyncio
async def test_fused_inference_delegates_to_prediction_path():
	engine = AdvancedMLEngine()
	engine.active_models["fusion_model"] = lambda payload: {
		"prediction": "matched",
		"confidence": 0.81
	}

	result = await engine._run_inference_with_fusion(
		"fusion_model",
		{"fused_features": [0.1, 0.9]},
		{"tenant_id": "tenant-a"}
	)

	assert result["predictions"]["class"] == "matched"
	assert result["confidence"] == 0.81
	assert result["model_source"] == "registered"
	assert result["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_explainability_prediction_depends_on_input_signal():
	engine = ExplainabilityEngine()

	positive = await engine._get_model_prediction("explainable", {"score": 0.9})
	negative = await engine._get_model_prediction("explainable", {"score": 0.1})

	assert positive["prediction"] == "positive"
	assert negative["prediction"] == "negative"
	assert positive["confidence"] == pytest.approx(0.9)
	assert negative["confidence"] == pytest.approx(0.9)
