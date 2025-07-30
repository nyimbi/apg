"""
Audio Processing Views Unit Tests

Tests for Pydantic v2 view models and Flask-AppBuilder dashboard views
with comprehensive validation and APG integration patterns.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from ...views import (
	# Pydantic v2 View Models
	APGViewModelBase, TranscriptionRequestView, VoiceSynthesisRequestView,
	VoiceCloningRequestView, AudioAnalysisRequestView, AudioEnhancementRequestView,
	AudioProcessingDashboardView, AudioJobStatusView, AudioModelConfigView,
	# Flask-AppBuilder Views  
	AudioProcessingDashboardView as FlaskDashboardView
)
from ...models import (
	AudioFormat, AudioQuality, TranscriptionProvider, VoiceSynthesisProvider,
	EmotionType, ProcessingStatus
)

class TestAPGViewModelBase:
	"""Test APGViewModelBase configuration"""
	
	def test_base_model_config(self):
		"""Test base model configuration settings"""
		# Create a simple test model that inherits from base
		class TestModel(APGViewModelBase):
			name: str
			value: int = 10
		
		# Test valid model
		model = TestModel(name="test")
		assert model.name == "test"
		assert model.value == 10
		
		# Test extra fields forbidden
		with pytest.raises(ValidationError):
			TestModel(name="test", extra_field="not_allowed")
	
	def test_string_stripping(self):
		"""Test string whitespace stripping"""
		class TestModel(APGViewModelBase):
			text: str
		
		model = TestModel(text="  hello world  ")
		assert model.text == "hello world"

class TestTranscriptionRequestView:
	"""Test TranscriptionRequestView model"""
	
	def test_valid_transcription_request(self):
		"""Test valid transcription request creation"""
		request = TranscriptionRequestView(
			audio_file_path="/tmp/test_audio.wav",
			audio_duration_seconds=120.5,
			language_code="en-US",
			provider=TranscriptionProvider.OPENAI_WHISPER,
			enable_speaker_diarization=True,
			max_speakers=5,
			custom_vocabulary=["product", "meeting", "analysis"],
			confidence_threshold=0.85,
			enable_real_time=False,
			priority_level="high"
		)
		
		assert request.audio_file_path == "/tmp/test_audio.wav"
		assert request.audio_duration_seconds == 120.5
		assert request.language_code == "en-US"
		assert request.provider == TranscriptionProvider.OPENAI_WHISPER
		assert request.enable_speaker_diarization is True
		assert request.max_speakers == 5
		assert len(request.custom_vocabulary) == 3
		assert request.confidence_threshold == 0.85
		assert request.priority_level == "high"
	
	def test_transcription_request_defaults(self):
		"""Test transcription request with default values"""
		request = TranscriptionRequestView()
		
		assert request.language_code == "en-US"
		assert request.provider == TranscriptionProvider.OPENAI_WHISPER
		assert request.enable_speaker_diarization is True
		assert request.max_speakers == 10
		assert request.custom_vocabulary == []
		assert request.confidence_threshold == 0.8
		assert request.enable_real_time is False
		assert request.priority_level == "normal"
	
	def test_transcription_request_validation(self):
		"""Test transcription request validation"""
		# Test invalid language code
		with pytest.raises(ValidationError):
			TranscriptionRequestView(language_code="invalid-code")
		
		# Test invalid max speakers (too high)
		with pytest.raises(ValidationError):
			TranscriptionRequestView(max_speakers=100)
		
		# Test invalid confidence threshold
		with pytest.raises(ValidationError):
			TranscriptionRequestView(confidence_threshold=1.5)
		
		# Test invalid priority level
		with pytest.raises(ValidationError):
			TranscriptionRequestView(priority_level="invalid")
	
	def test_custom_vocabulary_validation(self):
		"""Test custom vocabulary validation"""
		# Test valid vocabulary
		request = TranscriptionRequestView(
			custom_vocabulary=["AI", "machine learning", "technology"]
		)
		assert len(request.custom_vocabulary) == 3
		
		# Test vocabulary deduplication
		request = TranscriptionRequestView(
			custom_vocabulary=["AI", "AI", "technology", "technology"]
		)
		assert len(request.custom_vocabulary) == 2  # Duplicates removed
		
		# Test vocabulary too large
		large_vocab = [f"term_{i}" for i in range(1001)]
		with pytest.raises(ValidationError):
			TranscriptionRequestView(custom_vocabulary=large_vocab)
	
	def test_audio_duration_validation(self):
		"""Test audio duration validation"""
		# Test valid duration
		request = TranscriptionRequestView(audio_duration_seconds=3600.0)
		assert request.audio_duration_seconds == 3600.0
		
		# Test duration too long (over 10 hours)
		with pytest.raises(ValidationError):
			TranscriptionRequestView(audio_duration_seconds=40000.0)

class TestVoiceSynthesisRequestView:
	"""Test VoiceSynthesisRequestView model"""
	
	def test_valid_synthesis_request(self):
		"""Test valid synthesis request creation"""
		request = VoiceSynthesisRequestView(
			text_content="Hello, this is a test of voice synthesis.",
			voice_id="neural_female_001",
			emotion=EmotionType.HAPPY,
			emotion_intensity=0.7,
			speech_rate=1.2,
			pitch_adjustment=1.1,
			output_format=AudioFormat.WAV,
			quality=AudioQuality.HIGH,
			sample_rate=44100
		)
		
		assert request.text_content == "Hello, this is a test of voice synthesis."
		assert request.voice_id == "neural_female_001"
		assert request.emotion == EmotionType.HAPPY
		assert request.emotion_intensity == 0.7
		assert request.speech_rate == 1.2
		assert request.pitch_adjustment == 1.1
		assert request.output_format == AudioFormat.WAV
		assert request.quality == AudioQuality.HIGH
	
	def test_synthesis_request_defaults(self):
		"""Test synthesis request defaults"""
		request = VoiceSynthesisRequestView(
			text_content="Test text"
		)
		
		assert request.voice_id == "default_neural_female"
		assert request.emotion == EmotionType.NEUTRAL
		assert request.emotion_intensity == 0.5
		assert request.speech_rate == 1.0
		assert request.pitch_adjustment == 1.0
		assert request.output_format == AudioFormat.WAV
		assert request.quality == AudioQuality.STANDARD
	
	def test_text_content_validation(self):
		"""Test text content validation"""
		# Test empty text
		with pytest.raises(ValidationError):
			VoiceSynthesisRequestView(text_content="")
		
		# Test text too long
		long_text = "A" * 10001
		with pytest.raises(ValidationError):
			VoiceSynthesisRequestView(text_content=long_text)
		
		# Test text with unsupported Unicode
		# This would test for characters outside BMP if implemented
		request = VoiceSynthesisRequestView(text_content="Normal text")
		assert request.text_content == "Normal text"
	
	def test_speech_parameters_validation(self):
		"""Test speech parameters validation"""
		# Test invalid speech rate
		with pytest.raises(ValidationError):
			VoiceSynthesisRequestView(
				text_content="Test",
				speech_rate=3.0  # Too fast
			)
		
		# Test invalid emotion intensity
		with pytest.raises(ValidationError):
			VoiceSynthesisRequestView(
				text_content="Test",
				emotion_intensity=1.5  # Too high
			)
	
	def test_estimated_duration_calculation(self):
		"""Test estimated duration calculation"""
		# Text with 30 words should take ~10 seconds at normal rate
		text = "This is a test text with exactly thirty words for testing duration calculation in our voice synthesis system implementation today."
		
		request = VoiceSynthesisRequestView(
			text_content=text,
			speech_rate=1.0
		)
		
		duration = request.estimated_duration_seconds
		assert 8.0 <= duration <= 12.0  # Allow some variance

class TestVoiceCloningRequestView:
	"""Test VoiceCloningRequestView model"""
	
	def test_valid_voice_cloning_request(self):
		"""Test valid voice cloning request"""
		request = VoiceCloningRequestView(
			voice_name="Executive Voice",
			voice_description="Professional executive voice for presentations",
			training_audio_paths=["/tmp/sample1.wav", "/tmp/sample2.wav"],
			total_training_duration=120.0,
			target_language="en-US",
			voice_gender="male",
			age_category="adult",
			quality_target=0.95,
			enable_emotion_synthesis=True,
			supported_emotions=[EmotionType.NEUTRAL, EmotionType.CONFIDENT]
		)
		
		assert request.voice_name == "Executive Voice"
		assert len(request.training_audio_paths) == 2
		assert request.total_training_duration == 120.0
		assert request.target_language == "en-US"
		assert request.quality_target == 0.95
	
	def test_voice_cloning_validation(self):
		"""Test voice cloning validation"""
		# Test invalid voice name (empty)
		with pytest.raises(ValidationError):
			VoiceCloningRequestView(
				voice_name="",
				training_audio_paths=["/tmp/sample.wav"]
			)
		
		# Test no training audio paths
		with pytest.raises(ValidationError):
			VoiceCloningRequestView(
				voice_name="Test Voice",
				training_audio_paths=[]
			)
		
		# Test too many training files
		many_paths = [f"/tmp/sample_{i}.wav" for i in range(51)]
		with pytest.raises(ValidationError):
			VoiceCloningRequestView(
				voice_name="Test Voice",
				training_audio_paths=many_paths
			)
	
	def test_voice_name_normalization(self):
		"""Test voice name normalization"""
		request = VoiceCloningRequestView(
			voice_name="  Executive   Voice  Clone  ",
			training_audio_paths=["/tmp/sample.wav"]
		)
		
		assert request.voice_name == "Executive Voice Clone"

class TestAudioAnalysisRequestView:
	"""Test AudioAnalysisRequestView model"""
	
	def test_valid_analysis_request(self):
		"""Test valid analysis request"""
		request = AudioAnalysisRequestView(
			audio_source_path="/tmp/meeting.wav",
			enable_sentiment_analysis=True,
			enable_content_analysis=True,
			enable_speaker_analysis=True,
			enable_quality_assessment=True,
			num_topics=10,
			max_keywords=50,
			emotion_granularity="comprehensive",
			analysis_depth="comprehensive"
		)
		
		assert request.audio_source_path == "/tmp/meeting.wav"
		assert request.enable_sentiment_analysis is True
		assert request.num_topics == 10
		assert request.max_keywords == 50
		assert request.emotion_granularity == "comprehensive"
	
	def test_analysis_request_defaults(self):
		"""Test analysis request defaults"""
		request = AudioAnalysisRequestView()
		
		assert request.enable_sentiment_analysis is True
		assert request.enable_content_analysis is True
		assert request.enable_speaker_analysis is True
		assert request.enable_quality_assessment is True
		assert request.enable_music_analysis is False
		assert request.num_topics == 5
		assert request.max_keywords == 20
		assert request.analysis_depth == "standard"
	
	def test_analysis_parameters_validation(self):
		"""Test analysis parameters validation"""
		# Test invalid num_topics
		with pytest.raises(ValidationError):
			AudioAnalysisRequestView(num_topics=0)
		
		with pytest.raises(ValidationError):
			AudioAnalysisRequestView(num_topics=25)
		
		# Test invalid max_keywords
		with pytest.raises(ValidationError):
			AudioAnalysisRequestView(max_keywords=3)

class TestAudioEnhancementRequestView:
	"""Test AudioEnhancementRequestView model"""
	
	def test_valid_enhancement_request(self):
		"""Test valid enhancement request"""
		request = AudioEnhancementRequestView(
			audio_source_path="/tmp/noisy_audio.wav",
			enable_noise_reduction=True,
			noise_reduction_level="aggressive",
			enable_voice_isolation=True,
			num_speakers_to_isolate=3,
			enable_audio_normalization=True,
			target_lufs=-20.0,
			enable_compression=True,
			compression_ratio=3.5,
			output_format=AudioFormat.WAV
		)
		
		assert request.audio_source_path == "/tmp/noisy_audio.wav"
		assert request.enable_noise_reduction is True
		assert request.noise_reduction_level == "aggressive"
		assert request.num_speakers_to_isolate == 3
		assert request.target_lufs == -20.0
		assert request.compression_ratio == 3.5
	
	def test_enhancement_defaults(self):
		"""Test enhancement request defaults"""
		request = AudioEnhancementRequestView(
			audio_source_path="/tmp/test.wav"
		)
		
		assert request.enable_noise_reduction is True
		assert request.noise_reduction_level == "moderate"
		assert request.enable_voice_isolation is False
		assert request.enable_audio_normalization is True
		assert request.target_lufs == -23.0
		assert request.compression_ratio == 3.0
	
	def test_compression_ratio_validation(self):
		"""Test compression ratio validation"""
		# Test valid ratio
		request = AudioEnhancementRequestView(
			audio_source_path="/tmp/test.wav",
			compression_ratio=5.0
		)
		assert request.compression_ratio == 5.0
		
		# Test invalid ratio (too low)
		with pytest.raises(ValidationError):
			AudioEnhancementRequestView(
				audio_source_path="/tmp/test.wav",
				compression_ratio=0.5
			)
		
		# Test invalid ratio (too high)
		with pytest.raises(ValidationError):
			AudioEnhancementRequestView(
				audio_source_path="/tmp/test.wav",
				compression_ratio=15.0
			)

class TestAudioProcessingDashboardView:
	"""Test AudioProcessingDashboardView Pydantic model"""
	
	def test_dashboard_view_creation(self):
		"""Test dashboard view model creation"""
		dashboard = AudioProcessingDashboardView(
			total_jobs_today=156,
			jobs_completed=142,
			jobs_in_progress=8,
			jobs_queued=6,
			average_processing_time=45.2,
			average_transcription_accuracy=0.987,
			average_synthesis_quality=4.8,
			system_health="healthy",
			cpu_utilization=0.72,
			memory_utilization=0.68
		)
		
		assert dashboard.total_jobs_today == 156
		assert dashboard.jobs_completed == 142
		assert dashboard.average_transcription_accuracy == 0.987
		assert dashboard.system_health == "healthy"
	
	def test_dashboard_view_defaults(self):
		"""Test dashboard view defaults"""
		dashboard = AudioProcessingDashboardView()
		
		assert dashboard.total_jobs_today == 0
		assert dashboard.jobs_completed == 0
		assert dashboard.system_health == "healthy"
		assert len(dashboard.hourly_job_counts) == 24
		assert len(dashboard.hourly_success_rates) == 24

class TestAudioJobStatusView:
	"""Test AudioJobStatusView model"""
	
	def test_job_status_creation(self):
		"""Test job status view creation"""
		job_status = AudioJobStatusView(
			job_id="test_job_001",
			job_type="transcription",
			status=ProcessingStatus.COMPLETED,
			progress_percentage=100.0,
			created_at=datetime.utcnow(),
			started_at=datetime.utcnow(),
			completed_at=datetime.utcnow(),
			current_step="Processing complete",
			total_steps=5,
			steps_completed=5
		)
		
		assert job_status.job_id == "test_job_001"
		assert job_status.job_type == "transcription"
		assert job_status.status == ProcessingStatus.COMPLETED
		assert job_status.progress_percentage == 100.0
	
	def test_job_status_computed_fields(self):
		"""Test computed fields in job status"""
		now = datetime.utcnow()
		start_time = now - timedelta(minutes=5)
		
		job_status = AudioJobStatusView(
			job_id="test_job_002",
			job_type="synthesis",
			status=ProcessingStatus.IN_PROGRESS,
			created_at=start_time,
			started_at=start_time
		)
		
		assert job_status.is_active is True
		duration = job_status.processing_duration_seconds
		assert duration is not None
		assert duration > 0

class TestFlaskAppBuilderViews:
	"""Test Flask-AppBuilder dashboard views"""
	
	def test_dashboard_view_initialization(self):
		"""Test dashboard view class initialization"""
		view = FlaskDashboardView()
		assert view.route_base == '/audio_processing'
		assert view.default_view == 'dashboard'
	
	@patch('...views.render_template')
	def test_dashboard_endpoint(self, mock_render):
		"""Test dashboard endpoint"""
		view = FlaskDashboardView()
		mock_render.return_value = "rendered_template"
		
		result = view.dashboard()
		
		mock_render.assert_called_once()
		call_args = mock_render.call_args
		template_name = call_args[0][0]
		context = call_args[1]
		
		assert template_name == 'audio_processing/dashboard.html'
		assert 'stats' in context
		assert 'recent_jobs' in context
		assert 'page_title' in context
	
	@patch('...views.render_template')
	def test_transcription_workspace_endpoint(self, mock_render):
		"""Test transcription workspace endpoint"""
		view = FlaskDashboardView()
		mock_render.return_value = "rendered_template"
		
		result = view.transcription_workspace()
		
		mock_render.assert_called_once()
		call_args = mock_render.call_args
		template_name = call_args[0][0]
		context = call_args[1]
		
		assert template_name == 'audio_processing/transcription_workspace.html'
		assert 'transcriptions' in context
		assert context['page_title'] == "Transcription Workspace"
	
	@patch('...views.render_template')
	def test_synthesis_studio_endpoint(self, mock_render):
		"""Test synthesis studio endpoint"""
		view = FlaskDashboardView()
		mock_render.return_value = "rendered_template"
		
		result = view.synthesis_studio()
		
		mock_render.assert_called_once()
		call_args = mock_render.call_args
		context = call_args[1]
		
		assert 'voices' in context
		assert 'recent_synthesis' in context
		assert len(context['voices']) > 0
	
	@patch('...views.render_template')
	def test_analysis_console_endpoint(self, mock_render):
		"""Test analysis console endpoint"""
		view = FlaskDashboardView()
		mock_render.return_value = "rendered_template"
		
		result = view.analysis_console()
		
		mock_render.assert_called_once()
		call_args = mock_render.call_args
		context = call_args[1]
		
		assert 'analysis_results' in context
		assert 'recent_analysis' in context
		assert 'sentiment_distribution' in context['analysis_results']
	
	@patch('...views.render_template')
	def test_model_management_endpoint(self, mock_render):
		"""Test model management endpoint"""
		view = FlaskDashboardView()
		mock_render.return_value = "rendered_template"
		
		result = view.model_management()
		
		mock_render.assert_called_once()
		call_args = mock_render.call_args
		context = call_args[1]
		
		assert 'models' in context
		assert len(context['models']) > 0
		# Check model types
		model_types = [m['type'] for m in context['models']]
		assert 'transcription' in model_types
		assert 'synthesis' in model_types
	
	@patch('...views.render_template')
	def test_enhancement_tools_endpoint(self, mock_render):
		"""Test enhancement tools endpoint"""
		view = FlaskDashboardView()
		mock_render.return_value = "rendered_template"
		
		result = view.enhancement_tools()
		
		mock_render.assert_called_once()
		call_args = mock_render.call_args
		context = call_args[1]
		
		assert 'presets' in context
		assert 'recent_enhancements' in context
		assert len(context['presets']) == 3  # Three preset types
	
	@patch('...views.render_template')
	def test_job_status_endpoint(self, mock_render):
		"""Test job status detail endpoint"""
		view = FlaskDashboardView()
		mock_render.return_value = "rendered_template"
		
		job_id = "test_job_123"
		result = view.job_status(job_id)
		
		mock_render.assert_called_once()
		call_args = mock_render.call_args
		context = call_args[1]
		
		assert 'job' in context
		assert context['job']['job_id'] == job_id
		assert 'results' in context['job']

class TestViewIntegration:
	"""Test view integration and validation"""
	
	def test_pydantic_view_model_integration(self):
		"""Test integration between different view models"""
		# Create transcription request
		transcription_request = TranscriptionRequestView(
			audio_file_path="/tmp/integration_test.wav",
			language_code="en-US"
		)
		
		# Create corresponding job status
		job_status = AudioJobStatusView(
			job_id="integration_job_001",
			job_type="transcription",
			status=ProcessingStatus.COMPLETED,
			created_at=datetime.utcnow()
		)
		
		# Verify compatibility
		assert job_status.job_type == "transcription"
		assert job_status.status == ProcessingStatus.COMPLETED
	
	def test_dashboard_view_data_consistency(self):
		"""Test dashboard view data consistency"""
		dashboard = AudioProcessingDashboardView(
			total_jobs_today=100,
			jobs_completed=80,
			jobs_in_progress=15,
			jobs_queued=5,
			jobs_failed=0
		)
		
		# Total should equal sum of status categories
		total_accounted = (
			dashboard.jobs_completed + 
			dashboard.jobs_in_progress + 
			dashboard.jobs_queued + 
			dashboard.jobs_failed
		)
		assert total_accounted == dashboard.total_jobs_today
	
	def test_view_model_serialization(self):
		"""Test view model serialization/deserialization"""
		original = VoiceSynthesisRequestView(
			text_content="Test serialization",
			voice_id="test_voice",
			emotion=EmotionType.HAPPY
		)
		
		# Serialize to dict
		data = original.model_dump()
		
		# Deserialize back
		restored = VoiceSynthesisRequestView(**data)
		
		assert restored.text_content == original.text_content
		assert restored.voice_id == original.voice_id
		assert restored.emotion == original.emotion

class TestViewPerformance:
	"""Test view performance and validation efficiency"""
	
	def test_view_model_validation_performance(self):
		"""Test view model validation performance"""
		import time
		
		start_time = time.time()
		
		# Create many view models
		for i in range(1000):
			request = TranscriptionRequestView(
				audio_file_path=f"/tmp/test_{i}.wav",
				language_code="en-US"
			)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Should validate 1000 models quickly
		assert duration < 1.0
	
	def test_complex_view_model_performance(self):
		"""Test complex view model creation performance"""
		import time
		
		complex_data = {
			"audio_source_path": "/tmp/complex_test.wav",
			"enable_sentiment_analysis": True,
			"enable_content_analysis": True,
			"enable_speaker_analysis": True,
			"enable_quality_assessment": True,
			"num_topics": 20,
			"max_keywords": 100
		}
		
		start_time = time.time()
		
		for i in range(100):
			request = AudioAnalysisRequestView(**complex_data)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Should create 100 complex models quickly
		assert duration < 0.5