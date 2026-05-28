"""RAGN conversation turns should pass explicit retrieval results to generation."""

from __future__ import annotations

import pytest

from capabilities.common.ragn.conversation_manager import (
	ConversationConfig,
	ConversationMemory,
	TurnContext,
	TurnProcessor,
)
from capabilities.common.ragn.models import (
	Conversation,
	RetrievalMethod,
	RetrievalRequest,
	RetrievalResult,
)


def make_conversation() -> Conversation:
	return Conversation(
		tenant_id="tenant-a",
		knowledge_base_id="kb-runtime",
		generation_model="qwen3",
		max_context_tokens=4096,
		temperature=0.3,
	)


def make_context(**user_context) -> TurnContext:
	return TurnContext(
		conversation=make_conversation(),
		previous_turns=[],
		memory=ConversationMemory(),
		user_context=user_context,
	)


def make_retrieval_result(query: str, chunk_ids: list[str], scores: list[float]) -> RetrievalResult:
	return RetrievalResult(
		tenant_id="tenant-a",
		query_text=query,
		query_embedding=[0.0] * 1024,
		query_hash="abc123",
		knowledge_base_id="kb-runtime",
		k_retrievals=len(chunk_ids),
		similarity_threshold=0.7,
		retrieved_chunk_ids=chunk_ids,
		similarity_scores=scores,
		retrieval_method=RetrievalMethod.HYBRID_SEARCH,
		retrieval_time_ms=12,
		total_candidates=len(chunk_ids),
		result_quality_score=0.8,
		diversity_score=1.0,
	)


class FakeRetrievalEngine:
	def __init__(self, result: RetrievalResult):
		self.result = result
		self.requests: list[RetrievalRequest] = []

	async def retrieve(self, request: RetrievalRequest, context):
		self.requests.append(request)
		return self.result


class FakeGenerationEngine:
	pass


@pytest.mark.asyncio
async def test_process_user_turn_retains_retrieval_result_for_generation() -> None:
	result = make_retrieval_result("What is APG?", ["chunk-a"], [0.91])
	processor = TurnProcessor(ConversationConfig(auto_retrieve_threshold=0.7), FakeRetrievalEngine(result), FakeGenerationEngine())
	context = make_context()

	user_turn = await processor.process_user_turn("What is APG?", context)

	assert user_turn.retrieved_chunks == ["chunk-a"]
	assert user_turn.retrieval_scores == [0.91]
	assert context.user_context["_last_retrieval_result"] is result


@pytest.mark.asyncio
async def test_retrieval_result_reconstruction_reuses_stored_result() -> None:
	result = make_retrieval_result("Explain APG", ["chunk-a", "chunk-b"], [0.9, 0.8])
	processor = TurnProcessor(ConversationConfig(), FakeRetrievalEngine(result), FakeGenerationEngine())
	context = make_context(_last_retrieval_result=result)

	reconstructed = await processor._create_retrieval_result_from_chunks(
		"Explain APG",
		["chunk-a", "chunk-b"],
		[0.9, 0.8],
		context,
	)

	assert reconstructed is result


@pytest.mark.asyncio
async def test_retrieval_result_reconstruction_builds_valid_result_from_chunks() -> None:
	processor = TurnProcessor(ConversationConfig(auto_retrieve_threshold=0.66), FakeRetrievalEngine(None), FakeGenerationEngine())
	context = make_context()

	reconstructed = await processor._create_retrieval_result_from_chunks(
		"Explain APG composition",
		["chunk-a", "chunk-b", "chunk-a"],
		[0.9],
		context,
	)

	assert isinstance(reconstructed, RetrievalResult)
	assert reconstructed.tenant_id == "tenant-a"
	assert reconstructed.knowledge_base_id == "kb-runtime"
	assert reconstructed.retrieved_chunk_ids == ["chunk-a", "chunk-b", "chunk-a"]
	assert reconstructed.similarity_scores == [0.9, 0.0, 0.0]
	assert reconstructed.similarity_threshold == 0.66
	assert reconstructed.result_quality_score == pytest.approx(0.3)
	assert reconstructed.diversity_score == pytest.approx(2 / 3)
	assert len(reconstructed.query_embedding) == 1024


@pytest.mark.asyncio
async def test_retrieval_result_reconstruction_returns_none_without_chunks() -> None:
	processor = TurnProcessor(ConversationConfig(), FakeRetrievalEngine(None), FakeGenerationEngine())

	reconstructed = await processor._create_retrieval_result_from_chunks(
		"Explain APG",
		[],
		[],
		make_context(),
	)

	assert reconstructed is None
