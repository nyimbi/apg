"""
APG NLP Service Test Suite

Comprehensive tests for NLP service with async patterns, model orchestration,
and APG integration testing.

Uses modern pytest-asyncio patterns without decorators as per APG standards.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json
import time

from ..service import NLPService, ModelConfig
from ..models import (
	ProcessingRequest, ProcessingResult, StreamingSession, StreamingChunk,
	NLPTaskType, ModelProvider, ProcessingStatus, QualityLevel, LanguageCode
)
from . import TEST_CONFIG, TEST_TEXTS, MOCK_MODEL_RESPONSES


class TestNLPServiceInitialization:
	"""Test NLP service initialization and setup"""
	
	def test_service_initialization(self):
		"""Test NLP service proper initialization"""
		tenant_id = TEST_CONFIG['test_tenant_id']
		config = ModelConfig()
		
		service = NLPService(tenant_id, config)
		
		assert service.tenant_id == tenant_id
		assert service.config == config
		assert service._models == {}
		assert service._model_metadata == {}
		assert service._streaming_sessions == {}
	
	def test_service_initialization_requires_tenant_id(self):
		"""Test service initialization requires tenant ID"""
		with pytest.raises(AssertionError):
			NLPService("", ModelConfig())
		
		with pytest.raises(AssertionError):
			NLPService(None, ModelConfig())
	
	def test_service_initialization_tenant_id_must_be_string(self):
		"""Test tenant ID must be string"""
		with pytest.raises(AssertionError):
			NLPService(123, ModelConfig())


class TestModelInitialization:
	"""Test model initialization and loading"""
	
	async def test_initialize_models_success(self):
		"""Test successful model initialization"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Mock the initialization methods
		with patch.object(service, '_initialize_ollama_models') as mock_ollama, \
			 patch.object(service, '_initialize_transformers_models') as mock_transformers, \
			 patch.object(service, '_initialize_spacy_models') as mock_spacy:
			
			mock_ollama.return_value = asyncio.Future()
			mock_ollama.return_value.set_result(None)
			
			mock_transformers.return_value = asyncio.Future()
			mock_transformers.return_value.set_result(None)
			
			mock_spacy.return_value = asyncio.Future()
			mock_spacy.return_value.set_result(None)
			
			await service.initialize_models()
			
			mock_ollama.assert_called_once()
			mock_transformers.assert_called_once()
			mock_spacy.assert_called_once()
	
	async def test_initialize_models_handles_exceptions(self):
		"""Test model initialization handles exceptions gracefully"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		with patch.object(service, '_initialize_ollama_models') as mock_ollama:
			mock_ollama.side_effect = Exception("Ollama not available")
			
			with pytest.raises(Exception, match="Ollama not available"):
				await service.initialize_models()
	
	async def test_register_ollama_model(self):
		"""Test registering Ollama model"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		await service._register_ollama_model("llama3.2:latest")
		
		model_id = "ollama_llama3_2_latest"
		assert model_id in service._model_metadata
		assert model_id in service._models
		assert service._model_health.get(model_id) is True
		
		metadata = service._model_metadata[model_id]
		assert metadata.provider == ModelProvider.OLLAMA
		assert metadata.name == "Ollama llama3.2:latest"
		assert NLPTaskType.TEXT_GENERATION in metadata.supported_tasks
	
	async def test_register_transformers_model(self):
		"""Test registering Transformers model"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		model_config = {
			"name": "BERT Base",
			"model_name": "bert-base-uncased",
			"tasks": [NLPTaskType.SENTIMENT_ANALYSIS]
		}
		
		# Mock transformers components
		mock_tokenizer = MagicMock()
		mock_model = MagicMock()
		
		with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
			 patch('transformers.AutoModel.from_pretrained', return_value=mock_model), \
			 patch('torch.cuda.is_available', return_value=False):
			
			await service._register_transformers_model(model_config)
			
			model_id = "transformers_bert_base_uncased"
			assert model_id in service._model_metadata
			assert model_id in service._models
			
			metadata = service._model_metadata[model_id]
			assert metadata.provider == ModelProvider.TRANSFORMERS
			assert metadata.is_loaded is True


class TestTextProcessing:
	"""Test text processing functionality"""
	
	async def test_process_text_success(self):
		"""Test successful text processing"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Set up mock model
		service._model_metadata["test_model"] = Mock()
		service._model_metadata["test_model"].supported_tasks = [NLPTaskType.SENTIMENT_ANALYSIS]
		service._model_metadata["test_model"].is_available = True
		service._model_metadata["test_model"].provider = ModelProvider.TRANSFORMERS
		service._model_health["test_model"] = True
		
		service._models["test_model"] = {
			"type": "transformers",
			"model": Mock(),
			"tokenizer": Mock(),
			"device": "cpu",
			"name": "test-model"
		}
		
		request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment']
		)
		
		# Mock the processing methods
		with patch.object(service, '_prepare_text_content', return_value=TEST_TEXTS['positive_sentiment']), \
			 patch.object(service, '_select_optimal_model', return_value={
				 "id": "test_model",
				 "provider": ModelProvider.TRANSFORMERS,
				 "model": service._models["test_model"]
			 }), \
			 patch.object(service, '_execute_processing', return_value=MOCK_MODEL_RESPONSES['sentiment_analysis']):
			
			result = await service.process_text(request)
			
			assert result.request_id == request.id
			assert result.tenant_id == TEST_CONFIG['test_tenant_id']
			assert result.task_type == NLPTaskType.SENTIMENT_ANALYSIS
			assert result.status == ProcessingStatus.COMPLETED
			assert result.processing_time_ms > 0
			assert result.is_successful is True
	
	async def test_process_text_validation_error(self):
		"""Test text processing with validation error"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Request with wrong tenant ID
		request = ProcessingRequest(
			tenant_id="wrong_tenant",
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment']
		)
		
		with pytest.raises(AssertionError, match="Request tenant must match service tenant"):
			await service.process_text(request)
	
	async def test_process_text_no_available_models(self):
		"""Test text processing when no models are available"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment']
		)
		
		with patch.object(service, '_prepare_text_content', return_value=TEST_TEXTS['positive_sentiment']):
			result = await service.process_text(request)
			
			assert result.status == ProcessingStatus.FAILED
			assert result.error_message is not None
			assert result.is_successful is False
	
	async def test_select_optimal_model_preferred_model(self):
		"""Test optimal model selection with preferred model"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Set up preferred model
		preferred_model_id = "preferred_model"
		service._model_metadata[preferred_model_id] = Mock()
		service._model_metadata[preferred_model_id].supported_tasks = [NLPTaskType.SENTIMENT_ANALYSIS]
		service._model_metadata[preferred_model_id].is_available = True
		service._model_metadata[preferred_model_id].provider = ModelProvider.TRANSFORMERS
		service._model_health[preferred_model_id] = True
		service._models[preferred_model_id] = {"type": "test"}
		
		request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment'],
			preferred_model=preferred_model_id
		)
		
		selected = await service._select_optimal_model(NLPTaskType.SENTIMENT_ANALYSIS, request)
		
		assert selected["id"] == preferred_model_id
		assert selected["provider"] == ModelProvider.TRANSFORMERS
	
	async def test_select_optimal_model_quality_level(self):
		"""Test optimal model selection based on quality level"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Set up fast model (spaCy)
		fast_model_id = "spacy_model"
		service._model_metadata[fast_model_id] = Mock()
		service._model_metadata[fast_model_id].supported_tasks = [NLPTaskType.SENTIMENT_ANALYSIS]
		service._model_metadata[fast_model_id].is_available = True
		service._model_metadata[fast_model_id].provider = ModelProvider.SPACY
		service._model_metadata[fast_model_id].model_key = "en_core_web_sm"
		service._model_health[fast_model_id] = True
		service._models[fast_model_id] = {"type": "spacy"}
		
		# Set up quality model (BERT)
		quality_model_id = "bert_model"
		service._model_metadata[quality_model_id] = Mock()
		service._model_metadata[quality_model_id].supported_tasks = [NLPTaskType.SENTIMENT_ANALYSIS]
		service._model_metadata[quality_model_id].is_available = True
		service._model_metadata[quality_model_id].provider = ModelProvider.TRANSFORMERS
		service._model_metadata[quality_model_id].model_key = "bert-large-uncased"
		service._model_health[quality_model_id] = True
		service._models[quality_model_id] = {"type": "transformers"}
		
		# Test fast preference
		fast_request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment'],
			quality_level=QualityLevel.FAST
		)
		
		fast_selected = await service._select_optimal_model(NLPTaskType.SENTIMENT_ANALYSIS, fast_request)
		assert fast_selected["id"] == fast_model_id
		
		# Test best quality preference
		best_request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment'],
			quality_level=QualityLevel.BEST
		)
		
		best_selected = await service._select_optimal_model(NLPTaskType.SENTIMENT_ANALYSIS, best_request)
		assert best_selected["id"] == quality_model_id


class TestStreamingProcessing:
	"""Test real-time streaming processing"""
	
	async def test_create_streaming_session(self):
		"""Test creating streaming session"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		config = {
			"user_id": TEST_CONFIG['test_user_id'],
			"task_type": NLPTaskType.SENTIMENT_ANALYSIS,
			"chunk_size": 1000,
			"overlap_size": 100
		}
		
		session = await service.create_streaming_session(config)
		
		assert session.tenant_id == TEST_CONFIG['test_tenant_id']
		assert session.user_id == TEST_CONFIG['test_user_id']
		assert session.task_type == NLPTaskType.SENTIMENT_ANALYSIS
		assert session.chunk_size == 1000
		assert session.overlap_size == 100
		assert session.status == "active"
		
		# Verify session is stored
		assert session.id in service._streaming_sessions
		assert session.id in service._session_queues
	
	async def test_create_streaming_session_validation(self):
		"""Test streaming session creation validation"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Missing user_id
		with pytest.raises(AssertionError, match="User ID is required"):
			await service.create_streaming_session({"task_type": NLPTaskType.SENTIMENT_ANALYSIS})
		
		# Missing task_type
		with pytest.raises(AssertionError, match="Task type is required"):
			await service.create_streaming_session({"user_id": TEST_CONFIG['test_user_id']})
	
	async def test_process_streaming_chunk(self):
		"""Test processing streaming chunk"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Create session
		config = {
			"user_id": TEST_CONFIG['test_user_id'],
			"task_type": NLPTaskType.SENTIMENT_ANALYSIS
		}
		session = await service.create_streaming_session(config)
		
		# Create chunk
		chunk = StreamingChunk(
			session_id=session.id,
			sequence_number=1,
			text_content=TEST_TEXTS['positive_sentiment'],
			start_position=0,
			end_position=len(TEST_TEXTS['positive_sentiment'])
		)
		
		# Mock process_text method
		mock_result = ProcessingResult(
			request_id="mock_request",
			tenant_id=TEST_CONFIG['test_tenant_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_used="mock_model",
			provider_used=ModelProvider.TRANSFORMERS,
			processing_time_ms=45.0,
			total_time_ms=45.0,
			confidence_score=0.89,
			results=MOCK_MODEL_RESPONSES['sentiment_analysis']
		)
		
		with patch.object(service, 'process_text', return_value=mock_result):
			result = await service.process_streaming_chunk(session.id, chunk)
			
			assert result["chunk_id"] == chunk.id
			assert result["processing_time_ms"] > 0
			assert result["confidence"] == 0.89
			assert "session_metrics" in result
			
			# Verify session metrics updated
			updated_session = service._streaming_sessions[session.id]
			assert updated_session.chunks_processed == 1
			assert updated_session.total_characters > 0
	
	async def test_process_streaming_chunk_invalid_session(self):
		"""Test processing chunk with invalid session"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		chunk = StreamingChunk(
			session_id="invalid_session",
			sequence_number=1,
			text_content=TEST_TEXTS['positive_sentiment'],
			start_position=0,
			end_position=len(TEST_TEXTS['positive_sentiment'])
		)
		
		with pytest.raises(ValueError, match="Streaming session not found"):
			await service.process_streaming_chunk("invalid_session", chunk)


class TestSystemHealth:
	"""Test system health monitoring"""
	
	async def test_get_system_health(self):
		"""Test getting system health status"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Add some mock metrics
		service._request_metrics.append({
			"timestamp": datetime.utcnow(),
			"task_type": NLPTaskType.SENTIMENT_ANALYSIS,
			"model_used": "test_model",
			"processing_time_ms": 45.0,
			"success": True,
			"confidence": 0.89
		})
		
		# Add mock models
		service._model_metadata["model1"] = Mock()
		service._model_metadata["model1"].is_active = True
		service._model_metadata["model1"].is_loaded = True
		service._model_metadata["model1"].health_status = "healthy"
		
		service._model_health["model1"] = True
		
		health = await service.get_system_health()
		
		assert health.tenant_id == TEST_CONFIG['test_tenant_id']
		assert health.overall_status in ["healthy", "degraded", "unhealthy"]
		assert health.total_models == 1
		assert health.active_models >= 0
		assert health.loaded_models >= 0
		assert health.performance_rating in ["excellent", "good", "acceptable", "poor"]
	
	async def test_get_available_models(self):
		"""Test getting list of available models"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Add mock model
		from ..models import NLPModel
		mock_model = NLPModel(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Test Model",
			model_key="test-model",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="test-model",
			supported_tasks=[NLPTaskType.SENTIMENT_ANALYSIS],
			supported_languages=[LanguageCode.EN]
		)
		
		service._model_metadata["test_model"] = mock_model
		
		models = await service.get_available_models()
		
		assert len(models) == 1
		assert models[0].name == "Test Model"
		assert models[0].provider == ModelProvider.TRANSFORMERS
	
	async def test_get_model_performance(self):
		"""Test getting model performance metrics"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Add mock model and performance data
		from ..models import NLPModel
		mock_model = NLPModel(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Performance Test Model",
			model_key="perf-test-model",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="perf-test-model",
			supported_tasks=[NLPTaskType.SENTIMENT_ANALYSIS],
			supported_languages=[LanguageCode.EN],
			successful_requests=90,
			failed_requests=10
		)
		
		service._model_metadata["perf_model"] = mock_model
		service._model_performance["perf_model"] = {
			"total_requests": 100,
			"successful_requests": 90,
			"failed_requests": 10,
			"average_latency_ms": 45.2
		}
		
		performance = await service.get_model_performance("perf_model")
		
		assert performance["model_id"] == "perf_model"
		assert performance["model_name"] == "Performance Test Model"
		assert performance["total_requests"] == 100
		assert performance["success_rate"] == 90.0
		assert performance["average_latency_ms"] == 45.2
	
	async def test_get_model_performance_not_found(self):
		"""Test getting performance for non-existent model"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		with pytest.raises(ValueError, match="Model not found"):
			await service.get_model_performance("non_existent_model")


class TestServiceCleanup:
	"""Test service cleanup and resource management"""
	
	async def test_cleanup_streaming_sessions(self):
		"""Test cleanup closes streaming sessions"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Create streaming session
		config = {
			"user_id": TEST_CONFIG['test_user_id'],
			"task_type": NLPTaskType.SENTIMENT_ANALYSIS
		}
		session = await service.create_streaming_session(config)
		
		assert len(service._streaming_sessions) == 1
		assert len(service._session_queues) == 1
		
		# Cleanup
		await service.cleanup()
		
		assert len(service._streaming_sessions) == 0
		assert len(service._session_queues) == 0
	
	async def test_cleanup_models(self):
		"""Test cleanup handles model cleanup"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Add mock transformer model with to() method
		mock_model = Mock()
		mock_model.to = Mock()
		
		service._models["test_model"] = {
			"type": "transformers",
			"model": mock_model
		}
		
		await service.cleanup()
		
		# Verify model was moved to CPU
		mock_model.to.assert_called_once_with("cpu")


class TestIntegrationScenarios:
	"""Test integration scenarios combining multiple features"""
	
	async def test_complete_processing_workflow(self):
		"""Test complete processing workflow from request to result"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Set up mock model
		service._model_metadata["workflow_model"] = Mock()
		service._model_metadata["workflow_model"].supported_tasks = [NLPTaskType.SENTIMENT_ANALYSIS]
		service._model_metadata["workflow_model"].is_available = True
		service._model_metadata["workflow_model"].provider = ModelProvider.TRANSFORMERS
		service._model_health["workflow_model"] = True
		
		service._models["workflow_model"] = {
			"type": "transformers",
			"model": Mock(),
			"tokenizer": Mock(),
			"device": "cpu"
		}
		
		# Create request
		request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment'],
			quality_level=QualityLevel.BALANCED
		)
		
		# Mock transformers pipeline
		with patch('transformers.pipeline') as mock_pipeline:
			mock_pipeline.return_value.return_value = [{"label": "POSITIVE", "score": 0.89}]
			
			result = await service.process_text(request)
			
			assert result.is_successful
			assert result.task_type == NLPTaskType.SENTIMENT_ANALYSIS
			assert result.confidence_score > 0
			assert result.processing_time_ms > 0
	
	async def test_streaming_with_multiple_chunks(self):
		"""Test streaming processing with multiple chunks"""
		loop = asyncio.get_event_loop()
		service = NLPService(TEST_CONFIG['test_tenant_id'], ModelConfig())
		
		# Create streaming session
		config = {
			"user_id": TEST_CONFIG['test_user_id'],
			"task_type": NLPTaskType.SENTIMENT_ANALYSIS,
			"chunk_size": 100
		}
		session = await service.create_streaming_session(config)
		
		# Process multiple chunks
		chunks = [
			StreamingChunk(
				session_id=session.id,
				sequence_number=i,
				text_content=f"Chunk {i}: {TEST_TEXTS['positive_sentiment'][:50]}",
				start_position=i * 50,
				end_position=(i + 1) * 50
			)
			for i in range(3)
		]
		
		with patch.object(service, 'process_text') as mock_process:
			mock_process.return_value = ProcessingResult(
				request_id="mock",
				tenant_id=TEST_CONFIG['test_tenant_id'],
				task_type=NLPTaskType.SENTIMENT_ANALYSIS,
				model_used="mock",
				provider_used=ModelProvider.TRANSFORMERS,
				processing_time_ms=25.0,
				total_time_ms=25.0,
				confidence_score=0.85,
				results={"sentiment": "positive"}
			)
			
			results = []
			for chunk in chunks:
				result = await service.process_streaming_chunk(session.id, chunk)
				results.append(result)
			
			assert len(results) == 3
			
			# Verify session metrics
			updated_session = service._streaming_sessions[session.id]
			assert updated_session.chunks_processed == 3
			assert updated_session.average_latency_ms > 0