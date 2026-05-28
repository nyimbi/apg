"""Executable coverage for the NLPC core task dispatcher."""

from __future__ import annotations

import asyncio

from capabilities.common.nlpc.models import LanguageCode, NLPDocument, NLPTask, ProcessingRequest, ProcessingStatus
from capabilities.common.nlpc.service import NLPCoreService


TASK_PARAMETERS = {
	NLPTask.SEMANTIC_SIMILARITY: {"reference_text": "Alice paid invoice INV-100 on 2026-05-29."},
	NLPTask.TEXT_SIMILARITY: {"reference_text": "Alice paid invoice INV-100 on 2026-05-29."},
	NLPTask.TEXT_CLASSIFICATION: {"categories": ["billing", "support", "operations"]},
	NLPTask.INTENT_CLASSIFICATION: {"intents": ["question", "billing", "support"]},
	NLPTask.QUESTION_ANSWERING: {"question": "Who paid invoice INV-100?"},
	NLPTask.TEXT_GENERATION: {"prompt": "Draft a short billing update for invoice INV-100.", "prefix": "Update:"},
	NLPTask.TEXT_TRANSLATION: {"target_language": "sw", "source_language": "en"},
	NLPTask.TEXT_CLUSTERING: {"max_clusters": 2},
}


def _request_for(task: NLPTask) -> ProcessingRequest:
	return ProcessingRequest(
		tenant_id="tenant-1",
		tasks=[task],
		parameters=TASK_PARAMETERS.get(task, {}),
	)


def test_nlpc_core_dispatches_every_declared_task_without_notimplemented() -> None:
	service = NLPCoreService()
	document = NLPDocument(
		tenant_id="tenant-1",
		content=(
			"Alice paid invoice INV-100 on 2026-05-29. "
			"She approved the shipment for Nairobi. "
			"Contact alice@example.com for support."
		),
		language=LanguageCode.ENGLISH,
	)

	for task_name, task in NLPTask.__members__.items():
		results = asyncio.run(service.process_document(document, _request_for(task)))

		assert len(results) == 1, task_name
		result = results[0]
		assert result.status == ProcessingStatus.COMPLETED, (task_name, result.error_message)
		assert result.error_message is None, task_name
		assert result.result_data, task_name
		assert result.result_data["model_type"] != "error", task_name


def test_nlpc_core_returns_structured_results_for_extended_tasks() -> None:
	service = NLPCoreService()
	document = NLPDocument(
		tenant_id="tenant-1",
		content=(
			"Alice created a purchase order today. "
			"Bob approved it at 13:45. "
			"The order ships tomorrow."
		),
		language=LanguageCode.ENGLISH,
	)
	request = ProcessingRequest(
		tenant_id="tenant-1",
		tasks=[
			NLPTask.COREFERENCE_RESOLUTION,
			NLPTask.TEMPORAL_EXTRACTION,
			NLPTask.EVENT_EXTRACTION,
			NLPTask.TEXT_CLUSTERING,
		],
		parameters={"max_clusters": 2},
	)

	results = asyncio.run(service.process_document(document, request))
	payloads = {result.task_type: result.result_data for result in results}

	assert payloads[NLPTask.COREFERENCE_RESOLUTION]["coreference_chains"]
	assert payloads[NLPTask.TEMPORAL_EXTRACTION]["temporal_count"] >= 2
	assert payloads[NLPTask.EVENT_EXTRACTION]["event_count"] >= 2
	assert payloads[NLPTask.TEXT_CLUSTERING]["cluster_count"] >= 1
