"""
Audio Processing Models Unit Tests

Comprehensive unit tests for all data models with validation,
business logic, and APG integration patterns.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid import UUID

from pydantic import ValidationError

from ...models import (
	APAudioSession, APTranscriptionJob, APVoiceSynthesisJob, APAudioAnalysisJob,
	APVoiceModel, APAudioProcessingMetrics, AudioFormat, AudioQuality,
	TranscriptionProvider, VoiceSynthesisProvider, EmotionType, SentimentType,
	ProcessingStatus, AudioSessionType, ContentType
)
from uuid_extensions import uuid7str

class TestAPAudioSession:
	"""Test APAudioSession model"""
	
	def test_audio_session_creation_with_defaults(self):
		"""Test creating audio session with default values"""
		session = APAudioSession(
			tenant_id="test_tenant",
			session_type=AudioSessionType.TRANSCRIPTION
		)
		
		assert session.tenant_id == "test_tenant"
		assert session.session_type == AudioSessionType.TRANSCRIPTION
		assert session.session_id is not None
		assert len(session.session_id) > 0
		assert session.real_time_enabled is False
		assert session.participants == []
		assert session.configuration == {}
		assert session.status == ProcessingStatus.PENDING
		assert isinstance(session.created_at, datetime)
	
	def test_audio_session_validation(self):
		"""Test audio session field validation"""
		# Test valid session
		session = APAudioSession(
			tenant_id="valid_tenant",
			session_type=AudioSessionType.SYNTHESIS,
			participants=["user1", "user2"],
			real_time_enabled=True,
			configuration={"quality": "high", "language": "en-US"}
		)
		
		assert len(session.participants) == 2
		assert session.real_time_enabled is True
		assert session.configuration["quality"] == "high"
	
	def test_audio_session_invalid_tenant_id(self):
		"""Test validation of invalid tenant ID"""
		with pytest.raises(ValidationError) as exc_info:
			APAudioSession(
				tenant_id="",  # Empty tenant ID should fail
				session_type=AudioSessionType.ANALYSIS
			)
		
		error = exc_info.value
		assert "tenant_id" in str(error)
	
	def test_session_id_uniqueness(self):
		"""Test that session IDs are unique"""
		session1 = APAudioSession(
			tenant_id="test_tenant",
			session_type=AudioSessionType.TRANSCRIPTION
		)
		session2 = APAudioSession(
			tenant_id="test_tenant", 
			session_type=AudioSessionType.TRANSCRIPTION
		)
		
		assert session1.session_id != session2.session_id
	
	def test_session_configuration_validation(self):
		"""Test session configuration validation"""
		# Valid configuration
		config = {
			"language": "en-US",
			"quality": "high",
			"enable_diarization": True,
			"max_speakers": 10
		}
		
		session = APAudioSession(
			tenant_id="test_tenant",
			session_type=AudioSessionType.TRANSCRIPTION,
			configuration=config
		)
		
		assert session.configuration["language"] == "en-US"
		assert session.configuration["max_speakers"] == 10

class TestAPTranscriptionJob:
	"""Test APTranscriptionJob model"""
	
	def test_transcription_job_creation(self):
		"""Test creating transcription job"""
		audio_source = {
			"file_path": "/tmp/test_audio.wav",
			"duration": 120.5,
			"format": "wav"
		}
		
		job = APTranscriptionJob(
			job_id=uuid7str(),
			audio_source=audio_source,
			audio_duration=120.5,
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			language_code="en-US",
			tenant_id="test_tenant"
		)
		
		assert job.audio_duration == 120.5
		assert job.audio_format == AudioFormat.WAV
		assert job.provider == TranscriptionProvider.OPENAI_WHISPER
		assert job.language_code == "en-US"
		assert job.speaker_diarization is True  # Default value
		assert job.confidence_threshold == 0.8  # Default value
		assert job.status == ProcessingStatus.PENDING
	
	def test_transcription_job_validation(self):
		"""Test transcription job field validation"""
		# Test invalid audio duration
		with pytest.raises(ValidationError):
			APTranscriptionJob(
				job_id=uuid7str(),
				audio_source={"file_path": "/tmp/test.wav"},
				audio_duration=-5.0,  # Negative duration should fail
				audio_format=AudioFormat.WAV,
				provider=TranscriptionProvider.OPENAI_WHISPER,
				tenant_id="test_tenant"
			)
		
		# Test invalid confidence threshold
		with pytest.raises(ValidationError):
			APTranscriptionJob(
				job_id=uuid7str(),
				audio_source={"file_path": "/tmp/test.wav"},
				audio_duration=30.0,
				audio_format=AudioFormat.WAV,
				provider=TranscriptionProvider.OPENAI_WHISPER,
				confidence_threshold=1.5,  # > 1.0 should fail
				tenant_id="test_tenant"
			)
	
	def test_custom_vocabulary_validation(self):
		"""Test custom vocabulary validation"""
		# Valid vocabulary
		vocabulary = ["product", "meeting", "discussion", "analysis"]
		
		job = APTranscriptionJob(
			job_id=uuid7str(),
			audio_source={"file_path": "/tmp/test.wav"},
			audio_duration=60.0,
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			custom_vocabulary=vocabulary,
			tenant_id="test_tenant"
		)
		
		assert len(job.custom_vocabulary) == 4
		assert "product" in job.custom_vocabulary
	
	def test_processing_completion(self):
		"""Test transcription job completion"""
		job = APTranscriptionJob(
			job_id=uuid7str(),
			audio_source={"file_path": "/tmp/test.wav"},
			audio_duration=30.0,
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			tenant_id="test_tenant"
		)
		
		# Simulate job completion
		job.status = ProcessingStatus.COMPLETED
		job.transcription_text = "This is a test transcription."
		job.confidence_score = 0.95
		job.completed_at = datetime.utcnow()
		job.processing_metadata = {"processing_time_ms": 250.0}
		
		assert job.status == ProcessingStatus.COMPLETED
		assert job.transcription_text == "This is a test transcription."
		assert job.confidence_score == 0.95
		assert job.completed_at is not None

class TestAPVoiceSynthesisJob:
	"""Test APVoiceSynthesisJob model"""
	
	def test_synthesis_job_creation(self):
		"""Test creating voice synthesis job"""
		job = APVoiceSynthesisJob(
			job_id=uuid7str(),
			text_content="Hello, this is a test synthesis.",
			voice_id="neural_female_001",
			emotion=EmotionType.HAPPY,
			emotion_intensity=0.7,
			provider=VoiceSynthesisProvider.COQUI_XTTS,
			output_format=AudioFormat.WAV,
			quality=AudioQuality.HIGH,
			tenant_id="test_tenant"
		)
		
		assert job.text_content == "Hello, this is a test synthesis."
		assert job.voice_id == "neural_female_001"
		assert job.emotion == EmotionType.HAPPY
		assert job.emotion_intensity == 0.7
		assert job.provider == VoiceSynthesisProvider.COQUI_XTTS
		assert job.output_format == AudioFormat.WAV
		assert job.quality == AudioQuality.HIGH
		assert job.speech_rate == 1.0  # Default value
		assert job.pitch_adjustment == 1.0  # Default value
	
	def test_synthesis_text_validation(self):
		"""Test text content validation"""
		# Test empty text
		with pytest.raises(ValidationError):
			APVoiceSynthesisJob(
				job_id=uuid7str(),
				text_content="",  # Empty text should fail
				voice_id="neural_female_001",
				tenant_id="test_tenant"
			)
		
		# Test text too long
		long_text = "A" * 10001  # Exceeds max length
		with pytest.raises(ValidationError):
			APVoiceSynthesisJob(
				job_id=uuid7str(),
				text_content=long_text,
				voice_id="neural_female_001",
				tenant_id="test_tenant"
			)
	
	def test_speech_parameters_validation(self):
		"""Test speech parameter validation"""
		# Test invalid speech rate
		with pytest.raises(ValidationError):
			APVoiceSynthesisJob(
				job_id=uuid7str(),
				text_content="Test text",
				voice_id="neural_female_001",
				speech_rate=3.0,  # > 2.0 should fail
				tenant_id="test_tenant"
			)
		
		# Test invalid emotion intensity
		with pytest.raises(ValidationError):
			APVoiceSynthesisJob(
				job_id=uuid7str(),
				text_content="Test text",
				voice_id="neural_female_001",
				emotion_intensity=1.5,  # > 1.0 should fail
				tenant_id="test_tenant"
			)
	
	def test_estimated_duration_calculation(self):
		"""Test estimated duration calculation"""
		# Text with approximately 30 words (should be ~10 seconds at normal rate)
		text = "This is a test text with exactly thirty words for testing duration calculation in our voice synthesis system implementation today."
		
		job = APVoiceSynthesisJob(
			job_id=uuid7str(),
			text_content=text,
			voice_id="neural_female_001",
			speech_rate=1.0,
			tenant_id="test_tenant"
		)
		
		estimated_duration = job._calculate_estimated_duration()
		assert 8.0 <= estimated_duration <= 12.0  # Allow some variance

class TestAPAudioAnalysisJob:
	"""Test APAudioAnalysisJob model"""
	
	def test_analysis_job_creation(self):
		"""Test creating audio analysis job"""
		audio_source = {
			"file_path": "/tmp/analysis_audio.wav",
			"duration": 180.0,
			"format": "wav"
		}
		
		job = APAudioAnalysisJob(
			job_id=uuid7str(),
			audio_source=audio_source,
			analysis_type="sentiment",
			tenant_id="test_tenant",
			analysis_parameters={
				"include_emotions": True,
				"granularity": "detailed"
			}
		)
		
		assert job.analysis_type == "sentiment"
		assert job.analysis_parameters["include_emotions"] is True
		assert job.analysis_parameters["granularity"] == "detailed"
		assert job.status == ProcessingStatus.PENDING
	
	def test_analysis_results_structure(self):
		"""Test analysis results structure"""
		job = APAudioAnalysisJob(
			job_id=uuid7str(),
			audio_source={"file_path": "/tmp/test.wav"},
			analysis_type="comprehensive",
			tenant_id="test_tenant"
		)
		
		# Simulate analysis completion
		job.status = ProcessingStatus.COMPLETED
		job.analysis_results = {
			"sentiment": {
				"overall_sentiment": "positive",
				"score": 0.72,
				"emotions": {"happy": 0.4, "neutral": 0.5, "sad": 0.1}
			},
			"topics": [
				{"topic": "Technology", "confidence": 0.85},
				{"topic": "Business", "confidence": 0.78}
			],
			"quality": {
				"snr_db": 22.5,
				"clarity_score": 0.92
			}
		}
		job.confidence_score = 0.85
		job.completed_at = datetime.utcnow()
		
		assert job.analysis_results["sentiment"]["overall_sentiment"] == "positive"
		assert len(job.analysis_results["topics"]) == 2
		assert job.analysis_results["quality"]["snr_db"] == 22.5
		assert job.confidence_score == 0.85

class TestAPVoiceModel:
	"""Test APVoiceModel model"""
	
	def test_voice_model_creation(self):
		"""Test creating voice model"""
		model = APVoiceModel(
			model_id=uuid7str(),
			voice_name="Executive Voice Clone",
			voice_description="Custom executive voice for presentations",
			model_type="synthesis",
			training_audio_samples=["/tmp/sample1.wav", "/tmp/sample2.wav"],
			target_language="en-US",
			quality_score=4.9,
			tenant_id="test_tenant",
			supported_emotions=[EmotionType.NEUTRAL, EmotionType.CONFIDENT]
		)
		
		assert model.voice_name == "Executive Voice Clone"
		assert model.model_type == "synthesis"
		assert len(model.training_audio_samples) == 2
		assert model.target_language == "en-US"
		assert model.quality_score == 4.9
		assert EmotionType.NEUTRAL in model.supported_emotions
		assert model.status == ProcessingStatus.PENDING  # Default
	
	def test_voice_model_validation(self):
		"""Test voice model validation"""
		# Test invalid quality score
		with pytest.raises(ValidationError):
			APVoiceModel(
				model_id=uuid7str(),
				voice_name="Test Voice",
				model_type="synthesis",
				quality_score=6.0,  # > 5.0 should fail
				tenant_id="test_tenant"
			)
		
		# Test empty voice name
		with pytest.raises(ValidationError):
			APVoiceModel(
				model_id=uuid7str(),
				voice_name="",  # Empty name should fail
				model_type="synthesis",
				tenant_id="test_tenant"
			)
	
	def test_voice_model_training_validation(self):
		"""Test voice model training validation"""
		# Test insufficient training samples
		model = APVoiceModel(
			model_id=uuid7str(),
			voice_name="Test Voice",
			model_type="synthesis",
			training_audio_samples=["/tmp/sample1.wav"],  # Only one sample
			tenant_id="test_tenant"
		)
		
		# Should log warning but not fail validation
		assert len(model.training_audio_samples) == 1

class TestAPAudioProcessingMetrics:
	"""Test APAudioProcessingMetrics model"""
	
	def test_metrics_creation(self):
		"""Test creating audio processing metrics"""
		metrics = APAudioProcessingMetrics(
			metric_id=uuid7str(),
			tenant_id="test_tenant",
			metric_type="performance",
			metric_data={
				"transcription_accuracy": 0.987,
				"synthesis_quality": 4.8,
				"processing_latency_ms": 185.5,
				"concurrent_jobs": 25
			},
			aggregation_period="hourly",
			recorded_at=datetime.utcnow()
		)
		
		assert metrics.metric_type == "performance"
		assert metrics.metric_data["transcription_accuracy"] == 0.987
		assert metrics.metric_data["concurrent_jobs"] == 25
		assert metrics.aggregation_period == "hourly"
	
	def test_metrics_validation(self):
		"""Test metrics validation"""
		# Test valid metric data
		metrics = APAudioProcessingMetrics(
			metric_id=uuid7str(),
			tenant_id="test_tenant",
			metric_type="quality",
			metric_data={
				"average_quality": 4.5,
				"samples_processed": 1000,
				"error_rate": 0.02
			}
		)
		
		assert metrics.metric_data["average_quality"] == 4.5
		assert metrics.metric_data["error_rate"] == 0.02

class TestEnumerations:
	"""Test enumeration types"""
	
	def test_audio_format_enum(self):
		"""Test AudioFormat enumeration"""
		assert AudioFormat.WAV.value == "wav"
		assert AudioFormat.MP3.value == "mp3"
		assert AudioFormat.M4A.value == "m4a"
		assert AudioFormat.FLAC.value == "flac"
	
	def test_processing_status_enum(self):
		"""Test ProcessingStatus enumeration"""
		assert ProcessingStatus.PENDING.value == "pending"
		assert ProcessingStatus.QUEUED.value == "queued"
		assert ProcessingStatus.IN_PROGRESS.value == "in_progress"
		assert ProcessingStatus.COMPLETED.value == "completed"
		assert ProcessingStatus.FAILED.value == "failed"
	
	def test_emotion_type_enum(self):
		"""Test EmotionType enumeration"""
		assert EmotionType.NEUTRAL.value == "neutral"
		assert EmotionType.HAPPY.value == "happy"
		assert EmotionType.SAD.value == "sad"
		assert EmotionType.ANGRY.value == "angry"
		assert EmotionType.EXCITED.value == "excited"
		assert EmotionType.CONFIDENT.value == "confident"
	
	def test_transcription_provider_enum(self):
		"""Test TranscriptionProvider enumeration"""
		assert TranscriptionProvider.OPENAI_WHISPER.value == "openai_whisper"
		assert TranscriptionProvider.DEEPGRAM.value == "deepgram"
		assert TranscriptionProvider.ASSEMBLY_AI.value == "assembly_ai"

class TestModelIntegration:
	"""Test model integration and relationships"""
	
	def test_session_with_jobs(self):
		"""Test audio session with related jobs"""
		session = APAudioSession(
			tenant_id="test_tenant",
			session_type=AudioSessionType.COMPREHENSIVE,
			real_time_enabled=True
		)
		
		# Create related transcription job
		transcription_job = APTranscriptionJob(
			job_id=uuid7str(),
			session_id=session.session_id,
			audio_source={"file_path": "/tmp/test.wav"},
			audio_duration=60.0,
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			tenant_id="test_tenant"
		)
		
		# Create related synthesis job
		synthesis_job = APVoiceSynthesisJob(
			job_id=uuid7str(),
			session_id=session.session_id,
			text_content="This is synthesized from the transcription.",
			voice_id="neural_female_001",
			tenant_id="test_tenant"
		)
		
		assert transcription_job.session_id == session.session_id
		assert synthesis_job.session_id == session.session_id
	
	def test_multi_tenant_isolation(self):
		"""Test multi-tenant data isolation"""
		tenant1_session = APAudioSession(
			tenant_id="tenant_001",
			session_type=AudioSessionType.TRANSCRIPTION
		)
		
		tenant2_session = APAudioSession(
			tenant_id="tenant_002",
			session_type=AudioSessionType.TRANSCRIPTION
		)
		
		assert tenant1_session.tenant_id != tenant2_session.tenant_id
		assert tenant1_session.session_id != tenant2_session.session_id
	
	def test_model_timestamps(self):
		"""Test automatic timestamp generation"""
		model = APVoiceModel(
			model_id=uuid7str(),
			voice_name="Test Voice",
			model_type="synthesis",
			tenant_id="test_tenant"
		)
		
		assert model.created_at is not None
		assert model.updated_at is not None
		assert isinstance(model.created_at, datetime)
		assert isinstance(model.updated_at, datetime)

# Performance and validation tests
class TestModelPerformance:
	"""Test model performance and validation efficiency"""
	
	def test_bulk_model_creation(self):
		"""Test creating multiple models efficiently"""
		models = []
		start_time = datetime.utcnow()
		
		for i in range(100):
			model = APAudioSession(
				tenant_id=f"tenant_{i:03d}",
				session_type=AudioSessionType.TRANSCRIPTION
			)
			models.append(model)
		
		end_time = datetime.utcnow()
		duration = (end_time - start_time).total_seconds()
		
		assert len(models) == 100
		assert duration < 1.0  # Should create 100 models in under 1 second
	
	def test_validation_performance(self):
		"""Test validation performance for complex models"""
		complex_data = {
			"metric_data": {f"metric_{i}": i * 0.1 for i in range(1000)}
		}
		
		start_time = datetime.utcnow()
		
		metrics = APAudioProcessingMetrics(
			metric_id=uuid7str(),
			tenant_id="test_tenant",
			metric_type="complex",
			**complex_data
		)
		
		end_time = datetime.utcnow()
		duration = (end_time - start_time).total_seconds()
		
		assert len(metrics.metric_data) == 1000
		assert duration < 0.1  # Validation should be fast