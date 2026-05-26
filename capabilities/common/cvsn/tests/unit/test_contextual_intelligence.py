"""Focused regression coverage for contextual trend insights."""

import sys
import types

import pytest


if "torch" not in sys.modules:
	sys.modules["torch"] = types.ModuleType("torch")

if "transformers" not in sys.modules:
	transformers_stub = types.ModuleType("transformers")
	transformers_stub.pipeline = lambda *args, **kwargs: None
	transformers_stub.AutoModel = type(
		"AutoModel",
		(),
		{"from_pretrained": staticmethod(lambda *args, **kwargs: object())}
	)
	transformers_stub.AutoTokenizer = type(
		"AutoTokenizer",
		(),
		{"from_pretrained": staticmethod(lambda *args, **kwargs: object())}
	)
	sys.modules["transformers"] = transformers_stub

if "sklearn" not in sys.modules:
	sklearn_stub = types.ModuleType("sklearn")
	ensemble_stub = types.ModuleType("sklearn.ensemble")
	preprocessing_stub = types.ModuleType("sklearn.preprocessing")

	class _RandomForestClassifier:
		def __init__(self, *args, **kwargs):
			pass

		def fit(self, *args, **kwargs):
			return self

	class _StandardScaler:
		def fit_transform(self, values):
			return values

	ensemble_stub.RandomForestClassifier = _RandomForestClassifier
	preprocessing_stub.StandardScaler = _StandardScaler
	sklearn_stub.ensemble = ensemble_stub
	sklearn_stub.preprocessing = preprocessing_stub
	sys.modules["sklearn"] = sklearn_stub
	sys.modules["sklearn.ensemble"] = ensemble_stub
	sys.modules["sklearn.preprocessing"] = preprocessing_stub

from ...contextual_intelligence import BusinessContext, ContextualIntelligenceEngine


def _business_context(historical_patterns):
	return BusinessContext(
		tenant_id="tenant-test",
		created_by="user-test",
		industry_sector="manufacturing",
		department="quality",
		workflow_stage="inspection",
		historical_patterns=historical_patterns
	)


@pytest.mark.asyncio
async def test_analyze_with_context_emits_improving_trend_insight_from_history():
	engine = ContextualIntelligenceEngine()
	await engine._initialize_insight_generators()
	context = _business_context([
		{"quality_score": 0.68, "processing_time_ms": 1700},
		{"quality_score": 0.71, "processing_time_ms": 1600},
		{"quality_score": 0.73, "processing_time_ms": 1500},
	])

	result = await engine.analyze_with_context(
		b"image-bytes",
		context,
		{
			"quality_score": 0.84,
			"processing_time_ms": 1100,
			"confidence_score": 0.9,
			"detected_objects": [{"class_name": "product"}, {"class_name": "label"}, {"class_name": "packaging"}]
		}
	)

	trend_insight = next(
		insight for insight in result.context_insights if insight.insight_type == "trend_analysis"
	)
	assert "improving" in trend_insight.insight_message
	assert trend_insight.urgency_level == "low"
	assert trend_insight.confidence_score >= 0.7
	assert trend_insight.supporting_evidence[0]["trend_direction"] == "improving"


@pytest.mark.asyncio
async def test_generate_trend_insights_flags_deteriorating_history():
	engine = ContextualIntelligenceEngine()
	context = _business_context([
		{"visual_analysis": {"quality_score": 0.92, "processing_time_ms": 850}},
		{"visual_analysis": {"quality_score": 0.9, "processing_time_ms": 900}},
		{"visual_analysis": {"quality_score": 0.88, "processing_time_ms": 950}},
	])

	insight = await engine._generate_trend_insights(
		{"quality_score": 0.7, "processing_time_ms": 1400},
		context,
		[],
		{}
	)

	assert insight is not None
	assert "deteriorating" in insight.insight_message
	assert insight.urgency_level == "high"
	assert insight.supporting_evidence[0]["trend_direction"] == "deteriorating"


@pytest.mark.asyncio
async def test_generate_trend_insights_returns_none_without_enough_history():
	engine = ContextualIntelligenceEngine()
	context = _business_context([{"quality_score": 0.75, "processing_time_ms": 1200}])

	insight = await engine._generate_trend_insights(
		{"quality_score": 0.76, "processing_time_ms": 1180},
		context,
		[],
		{}
	)

	assert insight is None
