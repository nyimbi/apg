"""Regression coverage for executable NLPC pipeline task handlers."""

from __future__ import annotations

import pytest

from capabilities.common.nlpc.models import ProcessingRequest, NLPTaskType, QualityLevel
from capabilities.common.nlpc.processing_pipeline import AdvancedProcessingPipeline, PipelineConfig


@pytest.mark.asyncio
async def test_advanced_pipeline_returns_content_aware_sentiment_entities_and_qa():
	tenant_id = "tenant-nlpc-pipeline"
	pipeline = AdvancedProcessingPipeline(tenant_id)
	text = (
		"Amina approved invoice INV-204 today. "
		"Contact finance@example.com for the excellent finance report."
	)

	sentiment_request = ProcessingRequest(
		tenant_id=tenant_id,
		task_type=NLPTaskType.SENTIMENT_ANALYSIS,
		text_content=text,
		quality_level=QualityLevel.BALANCED,
	)
	entity_request = ProcessingRequest(
		tenant_id=tenant_id,
		task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
		text_content=text,
	)
	qa_request = ProcessingRequest(
		tenant_id=tenant_id,
		task_type=NLPTaskType.QUESTION_ANSWERING,
		text_content=text,
		parameters={"question": "Who approved invoice INV-204?"},
	)

	sentiment = await pipeline.process_single(sentiment_request)
	entities = await pipeline.process_single(entity_request)
	answer = await pipeline.process_single(qa_request)

	assert sentiment.is_successful is True
	assert sentiment.results["sentiment"] == "positive"
	assert sentiment.results["confidence"] > 0.5
	assert any(entity["text"] == "finance@example.com" for entity in entities.results["entities"])
	assert any(entity["text"] == "INV-204" for entity in entities.results["entities"])
	assert "Amina approved invoice INV-204" in answer.results["answer"]


@pytest.mark.asyncio
async def test_advanced_pipeline_supports_all_legacy_public_task_types():
	tenant_id = "tenant-nlpc-all-tasks"
	pipeline = AdvancedProcessingPipeline(tenant_id)
	config = PipelineConfig(
		name="deterministic-check",
		postprocessing_steps=["confidence_calibration", "result_validation"],
		validation_rules=["confidence_threshold_0.1"],
	)
	text = (
		"Amina approved invoice INV-204 today. "
		"The finance workflow is excellent. "
		"Customer support requested a follow-up tomorrow."
	)
	parameters = {
		"reference_text": "Amina approved a finance invoice today",
		"question": "What did customer support request?",
		"categories": ["finance", "support", "risk"],
		"max_clusters": 2,
		"max_keywords": 5,
	}

	for task_type in NLPTaskType:
		request = ProcessingRequest(
			tenant_id=tenant_id,
			task_type=task_type,
			text_content=text,
			parameters=parameters,
		)
		result = await pipeline.process_single(request, config)

		assert result.is_successful is True, task_type
		assert result.results["task_type"] == task_type.value
		assert result.results["validation_passed"] is True
		assert result.confidence_score is not None
		assert result.confidence_score >= 0.1
