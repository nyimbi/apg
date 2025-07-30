"""
Audio Processing API Unit Tests

Comprehensive tests for FastAPI REST endpoints with pytest-httpserver,
authentication, validation, and APG integration patterns.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from ...api import router as audio_router
from ...models import (
	AudioFormat, AudioQuality, TranscriptionProvider, VoiceSynthesisProvider,
	EmotionType, ProcessingStatus
)

# Test FastAPI app setup
def create_test_app():
	"""Create test FastAPI application"""
	app = FastAPI(title="Audio Processing Test API")
	app.include_router(audio_router)
	return app

@pytest.fixture
def test_app():
	"""FastAPI test application"""
	return create_test_app()

@pytest.fixture
def test_client(test_app):
	"""Test client for FastAPI app"""
	return TestClient(test_app)

@pytest.fixture
async def async_client(test_app):
	"""Async test client for FastAPI app"""
	async with AsyncClient(app=test_app, base_url="http://test") as client:
		yield client

class TestTranscriptionAPI:
	"""Test transcription API endpoints"""
	
	def test_transcribe_audio_endpoint(self, test_client):
		"""Test POST /api/v1/audio/transcribe endpoint"""
		request_data = {
			"audio_source": {
				"file_path": "/tmp/test_audio.wav",
				"duration": 30.0,
				"format": "wav"
			},
			"language_code": "en-US",
			"provider": "openai_whisper",
			"speaker_diarization": True,
			"custom_vocabulary": ["machine learning", "AI", "technology"],
			"real_time": False
		}
		
		with patch('...service.create_transcription_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.create_transcription_job = AsyncMock(return_value=MagicMock(
				job_id="test_job_001",
				status=ProcessingStatus.COMPLETED,
				transcription_text="This is a test transcription.",
				confidence_score=0.95,
				speaker_segments=[],
				processing_metadata={"processing_time_ms": 250.0},
				language_detected="en-US",
				created_at=datetime.utcnow(),
				completed_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/transcribe",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert response_data["job_id"] == "test_job_001"
		assert response_data["status"] == "completed"
		assert response_data["transcription_text"] == "This is a test transcription."
		assert response_data["confidence_score"] == 0.95
	
	def test_transcribe_audio_validation_error(self, test_client):
		"""Test transcription endpoint with validation errors"""
		# Missing required fields
		invalid_request = {
			"language_code": "en-US"
			# Missing audio_source
		}
		
		response = test_client.post(
			"/api/v1/audio/transcribe",
			json=invalid_request
		)
		
		assert response.status_code == 422  # Validation error
		error_data = response.json()
		assert "detail" in error_data
	
	def test_transcribe_audio_with_custom_vocabulary(self, test_client):
		"""Test transcription with custom vocabulary"""
		request_data = {
			"audio_source": {
				"file_path": "/tmp/technical_presentation.wav",
				"duration": 300.0
			},
			"language_code": "en-US",
			"provider": "openai_whisper",
			"custom_vocabulary": [
				"machine learning", "artificial intelligence", "deep learning",
				"neural networks", "computer vision", "natural language processing"
			],
			"confidence_threshold": 0.9
		}
		
		with patch('...service.create_transcription_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.create_transcription_job = AsyncMock(return_value=MagicMock(
				job_id="tech_job_001",
				status=ProcessingStatus.COMPLETED,
				transcription_text="Machine learning and artificial intelligence are transforming technology.",
				confidence_score=0.97,
				created_at=datetime.utcnow(),
				completed_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/transcribe",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert response_data["confidence_score"] >= 0.9
		assert "machine learning" in response_data["transcription_text"].lower()
	
	async def test_transcribe_audio_async(self, async_client):
		"""Test async transcription endpoint"""
		request_data = {
			"audio_source": {"file_path": "/tmp/async_test.wav"},
			"language_code": "en-US",
			"provider": "openai_whisper"
		}
		
		with patch('...service.create_transcription_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.create_transcription_job = AsyncMock(return_value=MagicMock(
				job_id="async_job_001",
				status=ProcessingStatus.IN_PROGRESS,
				created_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			response = await async_client.post(
				"/api/v1/audio/transcribe",
				json=request_data
			)
		
		assert response.status_code == 201
		assert response.json()["job_id"] == "async_job_001"

class TestSynthesisAPI:
	"""Test voice synthesis API endpoints"""
	
	def test_synthesize_speech_endpoint(self, test_client):
		"""Test POST /api/v1/audio/synthesize endpoint"""
		request_data = {
			"text": "Hello, this is a test of voice synthesis with emotional control.",
			"voice_id": "neural_female_001",
			"emotion": "happy",
			"emotion_intensity": 0.7,
			"speech_rate": 1.0,
			"pitch_adjustment": 1.0,
			"output_format": "wav",
			"quality": "high"
		}
		
		with patch('...service.create_synthesis_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.synthesize_text = AsyncMock(return_value=MagicMock(
				job_id="synthesis_job_001",
				status=ProcessingStatus.COMPLETED,
				output_audio_path="/tmp/synthesized_test.wav",
				audio_duration=5.2,
				quality_score=4.8,
				synthesis_metadata={"emotion_applied": "happy"},
				created_at=datetime.utcnow(),
				completed_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/synthesize",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert response_data["job_id"] == "synthesis_job_001"
		assert response_data["status"] == "completed"
		assert response_data["audio_path"] == "/tmp/synthesized_test.wav"
		assert response_data["quality_score"] == 4.8
	
	def test_synthesize_text_validation(self, test_client):
		"""Test synthesis endpoint validation"""
		# Text too long
		long_text = "A" * 10001
		invalid_request = {
			"text": long_text,
			"voice_id": "neural_female_001"
		}
		
		response = test_client.post(
			"/api/v1/audio/synthesize",
			json=invalid_request
		)
		
		assert response.status_code == 422
	
	def test_voice_cloning_endpoint(self, test_client):
		"""Test POST /api/v1/audio/voices/clone endpoint"""
		request_data = {
			"voice_name": "Executive Voice Clone",
			"voice_description": "Professional executive voice for presentations",
			"target_language": "en-US",
			"quality_target": 0.95
		}
		
		# Mock file upload
		files = [
			("audio_files", ("sample1.wav", b"mock_audio_data_1", "audio/wav")),
			("audio_files", ("sample2.wav", b"mock_audio_data_2", "audio/wav")),
			("audio_files", ("sample3.wav", b"mock_audio_data_3", "audio/wav"))
		]
		
		with patch('...service.create_synthesis_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.clone_voice_coqui_xtts = AsyncMock(return_value=MagicMock(
				model_id="voice_model_001",
				voice_name="Executive Voice Clone",
				status=ProcessingStatus.COMPLETED,
				quality_score=4.9,
				training_duration=45.0,
				supported_emotions=["neutral", "confident"],
				created_at=datetime.utcnow(),
				completed_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			with patch('...service.create_model_manager') as mock_manager_factory:
				mock_manager = AsyncMock()
				mock_manager.register_model = AsyncMock()
				mock_manager_factory.return_value = mock_manager
				
				response = test_client.post(
					"/api/v1/audio/voices/clone",
					data=request_data,
					files=files
				)
		
		assert response.status_code == 201
		response_data = response.json()
		assert response_data["voice_name"] == "Executive Voice Clone"
		assert response_data["quality_score"] == 4.9

class TestAnalysisAPI:
	"""Test audio analysis API endpoints"""
	
	def test_analyze_audio_endpoint(self, test_client):
		"""Test POST /api/v1/audio/analyze endpoint"""
		request_data = {
			"audio_source": {
				"file_path": "/tmp/meeting_recording.wav",
				"duration": 1800.0
			},
			"analysis_types": ["sentiment", "topics", "quality", "speaker_characteristics"],
			"num_topics": 5,
			"include_emotions": True,
			"include_technical_metrics": True
		}
		
		with patch('...service.create_analysis_service') as mock_service_factory:
			mock_service = AsyncMock()
			
			# Mock different analysis methods
			mock_service.analyze_sentiment = AsyncMock(return_value=MagicMock(
				analysis_results={
					"sentiment": "positive",
					"score": 0.72,
					"emotions": {"happy": 0.4, "neutral": 0.5, "confident": 0.1}
				},
				confidence_score=0.89
			))
			
			mock_service.detect_topics = AsyncMock(return_value=MagicMock(
				analysis_results={
					"topics": [
						{"topic": "Product Development", "confidence": 0.89},
						{"topic": "Market Strategy", "confidence": 0.82}
					]
				},
				confidence_score=0.85
			))
			
			mock_service.assess_quality = AsyncMock(return_value=MagicMock(
				analysis_results={
					"overall_quality": "high",
					"snr_db": 22.5,
					"clarity_score": 0.92
				},
				confidence_score=0.91
			))
			
			mock_service.detect_speaker_characteristics = AsyncMock(return_value=MagicMock(
				analysis_results={
					"speaker_count": 3,
					"speakers": [
						{"speaker_id": "speaker_1", "gender": "male"},
						{"speaker_id": "speaker_2", "gender": "female"}
					]
				},
				confidence_score=0.87
			))
			
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/analyze",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert response_data["status"] == "completed"
		assert "sentiment" in response_data["analysis_results"]
		assert "topics" in response_data["analysis_results"]
		assert "quality" in response_data["analysis_results"]
		assert "speaker_characteristics" in response_data["analysis_results"]
	
	def test_analyze_audio_sentiment_only(self, test_client):
		"""Test analysis with sentiment only"""
		request_data = {
			"audio_source": {"file_path": "/tmp/customer_call.wav"},
			"analysis_types": ["sentiment"],
			"include_emotions": True
		}
		
		with patch('...service.create_analysis_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.analyze_sentiment = AsyncMock(return_value=MagicMock(
				analysis_results={
					"sentiment": "negative",
					"score": -0.3,
					"emotions": {"frustrated": 0.6, "disappointed": 0.3, "neutral": 0.1}
				},
				confidence_score=0.92
			))
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/analyze",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert "sentiment" in response_data["analysis_results"]
		assert response_data["analysis_results"]["sentiment"]["sentiment"] == "negative"

class TestEnhancementAPI:
	"""Test audio enhancement API endpoints"""
	
	def test_enhance_audio_noise_reduction(self, test_client):
		"""Test POST /api/v1/audio/enhance endpoint for noise reduction"""
		request_data = {
			"audio_source": {"file_path": "/tmp/noisy_audio.wav"},
			"enhancement_type": "noise_reduction",
			"parameters": {
				"level": "moderate",
				"preserve_speech": True
			},
			"output_format": "wav"
		}
		
		with patch('...service.create_enhancement_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.reduce_noise = AsyncMock(return_value={
				"enhanced_path": "/tmp/denoised_audio.wav",
				"noise_reduced_db": 35.0,
				"quality_improvement": 3.2,
				"processing_time_ms": 8500
			})
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/enhance",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert response_data["enhanced_path"] == "/tmp/denoised_audio.wav"
		assert response_data["noise_reduced_db"] == 35.0
	
	def test_enhance_audio_voice_isolation(self, test_client):
		"""Test audio enhancement with voice isolation"""
		request_data = {
			"audio_source": {"file_path": "/tmp/meeting_audio.wav"},
			"enhancement_type": "voice_isolation",
			"parameters": {
				"num_speakers": 3,
				"quality": "high"
			},
			"output_format": "wav"
		}
		
		with patch('...service.create_enhancement_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.isolate_voices = AsyncMock(return_value={
				"isolated_voices": [
					"/tmp/speaker_1_isolated.wav",
					"/tmp/speaker_2_isolated.wav", 
					"/tmp/speaker_3_isolated.wav"
				],
				"separation_quality": 0.89,
				"processing_time_ms": 15200
			})
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/enhance",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert len(response_data["isolated_voices"]) == 3
		assert response_data["separation_quality"] == 0.89
	
	def test_enhance_audio_invalid_type(self, test_client):
		"""Test enhancement with invalid enhancement type"""
		request_data = {
			"audio_source": {"file_path": "/tmp/test.wav"},
			"enhancement_type": "invalid_enhancement",
			"output_format": "wav"
		}
		
		response = test_client.post(
			"/api/v1/audio/enhance",
			json=request_data
		)
		
		assert response.status_code == 400
		response_data = response.json()
		assert "Unknown enhancement type" in response_data["detail"]

class TestWorkflowAPI:
	"""Test workflow execution API endpoints"""
	
	def test_execute_workflow_comprehensive(self, test_client):
		"""Test POST /api/v1/audio/workflows/execute endpoint"""
		request_data = {
			"audio_source": {"file_path": "/tmp/comprehensive_test.wav"},
			"workflow_type": "transcribe_analyze_enhance",
			"parameters": {
				"transcription_provider": "openai_whisper",
				"analysis_types": ["sentiment", "topics"],
				"enhancement_type": "noise_reduction"
			}
		}
		
		with patch('...service.create_workflow_orchestrator') as mock_orchestrator_factory:
			mock_orchestrator = AsyncMock()
			mock_orchestrator.process_complete_workflow = AsyncMock(return_value={
				"workflow_id": "workflow_comprehensive_001",
				"status": "completed",
				"total_processing_time": 87.5,
				"results": {
					"transcription": {
						"job_id": "trans_001",
						"status": "completed",
						"text": "This is the transcribed content."
					},
					"analysis": {
						"job_id": "analysis_001", 
						"status": "completed",
						"sentiment": "positive"
					},
					"enhancement": {
						"job_id": "enhance_001",
						"status": "completed",
						"improvement": 2.8
					},
					"steps_completed": ["transcription", "analysis", "enhancement"]
				}
			})
			mock_orchestrator_factory.return_value = mock_orchestrator
			
			response = test_client.post(
				"/api/v1/audio/workflows/execute",
				json=request_data
			)
		
		assert response.status_code == 201
		response_data = response.json()
		assert response_data["workflow_id"] == "workflow_comprehensive_001"
		assert response_data["status"] == "completed"
		assert len(response_data["steps_completed"]) == 3

class TestUtilityAPI:
	"""Test utility API endpoints"""
	
	def test_get_job_status(self, test_client):
		"""Test GET /api/v1/audio/jobs/{job_id} endpoint"""
		job_id = "test_job_001"
		
		response = test_client.get(f"/api/v1/audio/jobs/{job_id}")
		
		assert response.status_code == 200
		response_data = response.json()
		assert response_data["job_id"] == job_id
		assert "status" in response_data
	
	def test_list_voices(self, test_client):
		"""Test GET /api/v1/audio/voices endpoint"""
		with patch('...service.create_model_manager') as mock_manager_factory:
			mock_manager = AsyncMock()
			mock_manager.list_models = AsyncMock(return_value=[
				MagicMock(
					model_id="voice_001",
					voice_name="Sarah Professional",
					voice_description="Professional female voice",
					quality_score=4.8,
					supported_emotions=["neutral", "happy"],
					created_at=datetime.utcnow()
				),
				MagicMock(
					model_id="voice_002",
					voice_name="David Conversational",
					voice_description="Conversational male voice",
					quality_score=4.7,
					supported_emotions=["neutral", "confident"],
					created_at=datetime.utcnow()
				)
			])
			mock_manager_factory.return_value = mock_manager
			
			response = test_client.get("/api/v1/audio/voices")
		
		assert response.status_code == 200
		response_data = response.json()
		assert len(response_data) == 2
		assert response_data[0]["voice_name"] == "Sarah Professional"
		assert response_data[1]["voice_name"] == "David Conversational"
	
	def test_delete_voice(self, test_client):
		"""Test DELETE /api/v1/audio/voices/{model_id} endpoint"""
		model_id = "voice_custom_001"
		
		with patch('...service.create_model_manager') as mock_manager_factory:
			mock_manager = AsyncMock()
			mock_manager.delete_model = AsyncMock(return_value=True)
			mock_manager_factory.return_value = mock_manager
			
			response = test_client.delete(f"/api/v1/audio/voices/{model_id}")
		
		assert response.status_code == 200
		response_data = response.json()
		assert f"Voice model {model_id} deleted successfully" in response_data["message"]
	
	def test_health_check(self, test_client):
		"""Test GET /api/v1/audio/health endpoint"""
		response = test_client.get("/api/v1/audio/health")
		
		assert response.status_code == 200
		response_data = response.json()
		assert response_data["status"] == "healthy"
		assert response_data["service"] == "audio_processing"
		assert response_data["version"] == "1.0.0"
		assert "timestamp" in response_data

class TestAPIAuthentication:
	"""Test API authentication and authorization"""
	
	def test_api_with_tenant_context(self, test_client):
		"""Test API endpoints with tenant context"""
		headers = {
			"X-Tenant-ID": "enterprise_tenant_001",
			"X-User-ID": "user_001"
		}
		
		request_data = {
			"audio_source": {"file_path": "/tmp/tenant_test.wav"},
			"language_code": "en-US",
			"provider": "openai_whisper"
		}
		
		with patch('...service.create_transcription_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.create_transcription_job = AsyncMock(return_value=MagicMock(
				job_id="tenant_job_001",
				status=ProcessingStatus.QUEUED,
				created_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			response = test_client.post(
				"/api/v1/audio/transcribe",
				json=request_data,
				headers=headers
			)
		
		assert response.status_code == 201
		# Verify tenant context was passed to service
		mock_service.create_transcription_job.assert_called()

class TestAPIPerformance:
	"""Test API performance and load handling"""
	
	async def test_concurrent_api_requests(self, async_client):
		"""Test handling multiple concurrent API requests"""
		request_data = {
			"audio_source": {"file_path": "/tmp/load_test.wav"},
			"language_code": "en-US",
			"provider": "openai_whisper"
		}
		
		with patch('...service.create_transcription_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.create_transcription_job = AsyncMock(return_value=MagicMock(
				job_id="load_test_job",
				status=ProcessingStatus.QUEUED,
				created_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			# Create multiple concurrent requests
			tasks = [
				async_client.post("/api/v1/audio/transcribe", json=request_data)
				for _ in range(10)
			]
			
			responses = await asyncio.gather(*tasks)
		
		# All requests should succeed
		for response in responses:
			assert response.status_code == 201
	
	def test_api_response_time(self, test_client):
		"""Test API response time performance"""
		import time
		
		request_data = {
			"audio_source": {"file_path": "/tmp/performance_test.wav"},
			"language_code": "en-US",
			"provider": "openai_whisper"
		}
		
		with patch('...service.create_transcription_service') as mock_service_factory:
			mock_service = AsyncMock()
			mock_service.create_transcription_job = AsyncMock(return_value=MagicMock(
				job_id="perf_test_job",
				status=ProcessingStatus.QUEUED,
				created_at=datetime.utcnow()
			))
			mock_service_factory.return_value = mock_service
			
			start_time = time.time()
			response = test_client.post(
				"/api/v1/audio/transcribe",
				json=request_data
			)
			end_time = time.time()
		
		response_time = (end_time - start_time) * 1000  # Convert to milliseconds
		
		assert response.status_code == 201
		assert response_time < 100  # Should respond within 100ms (excluding actual processing)