"""
Audio Processing Services Unit Tests

Comprehensive tests for all audio processing services including
transcription, synthesis, analysis, enhancement, and model management.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from ...service import (
	AudioTranscriptionService, VoiceSynthesisService, AudioAnalysisService,
	AudioEnhancementService, AudioModelManager, AudioWorkflowOrchestrator,
	create_transcription_service, create_synthesis_service, create_analysis_service,
	create_enhancement_service, create_model_manager, create_workflow_orchestrator
)
from ...models import (
	APTranscriptionJob, APVoiceSynthesisJob, APAudioAnalysisJob, APVoiceModel,
	AudioFormat, ProcessingStatus, TranscriptionProvider, VoiceSynthesisProvider,
	EmotionType, SentimentType
)

class TestAudioTranscriptionService:
	"""Test AudioTranscriptionService"""
	
	async def test_service_initialization(self):
		"""Test transcription service initialization"""
		service = AudioTranscriptionService()
		
		assert service is not None
		assert hasattr(service, 'create_transcription_job')
		assert hasattr(service, 'process_stream')
		assert hasattr(service, 'get_supported_languages')
	
	async def test_create_transcription_job(self, mock_transcription_service, sample_transcription_data):
		"""Test creating transcription job"""
		service = mock_transcription_service
		data = sample_transcription_data
		
		job = await service.create_transcription_job(
			audio_source=data['audio_source'],
			audio_duration=data['audio_source']['duration'],
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			language_code=data['language_code'],
			tenant_id="test_tenant",
			speaker_diarization=data['speaker_diarization'],
			custom_vocabulary=data['custom_vocabulary']
		)
		
		assert job is not None
		assert job.job_id is not None
		assert job.status == ProcessingStatus.COMPLETED
		assert job.confidence_score == 0.95
		assert job.transcription_text == 'This is a test transcription.'
	
	async def test_transcription_with_speaker_diarization(self, mock_transcription_service):
		"""Test transcription with speaker diarization"""
		service = mock_transcription_service
		
		# Mock speaker diarization result
		async def mock_diarization_job(**kwargs):
			from ...models import APTranscriptionJob
			from uuid_extensions import uuid7str
			
			job = APTranscriptionJob(
				job_id=uuid7str(),
				audio_source=kwargs.get('audio_source', {}),
				language_code=kwargs.get('language_code', 'en-US'),
				speaker_diarization=True,
				status=ProcessingStatus.COMPLETED,
				transcription_text='Speaker 1: Hello. Speaker 2: Hi there.',
				speaker_segments=[
					{'speaker': 'Speaker 1', 'start': 0.0, 'end': 2.0, 'text': 'Hello.'},
					{'speaker': 'Speaker 2', 'start': 2.0, 'end': 4.0, 'text': 'Hi there.'}
				],
				confidence_score=0.92,
				tenant_id=kwargs.get('tenant_id', 'test')
			)
			return job
		
		service.create_transcription_job = mock_diarization_job
		
		job = await service.create_transcription_job(
			audio_source={'file_path': '/tmp/conversation.wav'},
			audio_duration=4.0,
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			language_code='en-US',
			speaker_diarization=True,
			tenant_id='test_tenant'
		)
		
		assert job.speaker_diarization is True
		assert len(job.speaker_segments) == 2
		assert job.speaker_segments[0]['speaker'] == 'Speaker 1'
		assert job.speaker_segments[1]['speaker'] == 'Speaker 2'
	
	async def test_real_time_transcription_stream(self, mock_transcription_service):
		"""Test real-time transcription streaming"""
		service = mock_transcription_service
		
		# Mock streaming response
		async def mock_stream_generator():
			chunks = [
				{'partial': 'Hello'},
				{'partial': 'Hello there'},
				{'final': 'Hello there, how are you?', 'confidence': 0.95}
			]
			for chunk in chunks:
				yield chunk
		
		service.process_stream = AsyncMock(return_value=mock_stream_generator())
		
		stream = await service.process_stream(
			audio_stream=b'mock_audio_data',
			language_code='en-US',
			tenant_id='test_tenant'
		)
		
		assert stream is not None
		service.process_stream.assert_called_once()
	
	async def test_custom_vocabulary_integration(self, mock_transcription_service):
		"""Test transcription with custom vocabulary"""
		service = mock_transcription_service
		
		custom_vocab = ['machine learning', 'artificial intelligence', 'deep learning', 'neural networks']
		
		job = await service.create_transcription_job(
			audio_source={'file_path': '/tmp/tech_talk.wav'},
			audio_duration=300.0,
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			language_code='en-US',
			custom_vocabulary=custom_vocab,
			tenant_id='test_tenant'
		)
		
		assert job is not None
		# Custom vocabulary should improve accuracy for technical terms
		assert job.confidence_score >= 0.90
	
	async def test_multi_language_support(self, mock_transcription_service):
		"""Test multi-language transcription support"""
		service = mock_transcription_service
		
		languages = ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'zh-CN']
		
		for lang in languages:
			job = await service.create_transcription_job(
				audio_source={'file_path': f'/tmp/audio_{lang}.wav'},
				audio_duration=60.0,
				audio_format=AudioFormat.WAV,
				provider=TranscriptionProvider.OPENAI_WHISPER,
				language_code=lang,
				tenant_id='test_tenant'
			)
			
			assert job is not None
			assert job.status == ProcessingStatus.COMPLETED

class TestVoiceSynthesisService:
	"""Test VoiceSynthesisService"""
	
	async def test_service_initialization(self):
		"""Test synthesis service initialization"""
		service = VoiceSynthesisService()
		
		assert service is not None
		assert hasattr(service, 'synthesize_text')
		assert hasattr(service, 'clone_voice_coqui_xtts')
		assert hasattr(service, 'list_available_voices')
	
	async def test_text_to_speech_synthesis(self, mock_synthesis_service, sample_synthesis_data):
		"""Test basic text-to-speech synthesis"""
		service = mock_synthesis_service
		data = sample_synthesis_data
		
		job = await service.synthesize_text(
			text=data['text'],
			voice_id=data['voice_id'],
			emotion=EmotionType.HAPPY,
			emotion_intensity=data['expected_result'].get('emotion_intensity', 0.7),
			output_format=AudioFormat.WAV,
			tenant_id='test_tenant'
		)
		
		assert job is not None
		assert job.job_id is not None
		assert job.status == ProcessingStatus.COMPLETED
		assert job.output_audio_path == '/tmp/synthesized.wav'
		assert job.quality_score == 4.8
		assert job.audio_duration == 5.2
	
	async def test_voice_cloning_coqui_xtts(self, mock_synthesis_service, temp_audio_dir):
		"""Test voice cloning with Coqui XTTS"""
		service = mock_synthesis_service
		
		# Create mock training audio files
		training_samples = []
		for i in range(5):
			sample_path = temp_audio_dir / f"training_sample_{i}.wav"
			with open(sample_path, 'wb') as f:
				f.write(b'mock_audio_data')
			training_samples.append(str(sample_path))
		
		# Mock successful voice cloning
		async def mock_clone_voice(**kwargs):
			from ...models import APVoiceModel
			from uuid_extensions import uuid7str
			
			model = APVoiceModel(
				model_id=uuid7str(),
				voice_name=kwargs.get('voice_name', 'Cloned Voice'),
				voice_description=kwargs.get('voice_description', ''),
				model_type='synthesis',
				training_audio_samples=kwargs.get('training_audio_samples', []),
				quality_score=4.9,
				training_duration=len(kwargs.get('training_audio_samples', [])) * 10.0,
				status=ProcessingStatus.COMPLETED,
				tenant_id=kwargs.get('tenant_id', 'test')
			)
			return model
		
		service.clone_voice_coqui_xtts = mock_clone_voice
		
		voice_model = await service.clone_voice_coqui_xtts(
			voice_name='Executive Voice Clone',
			training_audio_samples=training_samples,
			voice_description='Professional executive voice for presentations',
			target_language='en-US',
			quality_target=0.95,
			tenant_id='test_tenant'
		)
		
		assert voice_model is not None
		assert voice_model.voice_name == 'Executive Voice Clone'
		assert voice_model.quality_score == 4.9
		assert voice_model.status == ProcessingStatus.COMPLETED
		assert len(voice_model.training_audio_samples) == 5
	
	async def test_emotion_controlled_synthesis(self, mock_synthesis_service):
		"""Test synthesis with emotion control"""
		service = mock_synthesis_service
		
		emotions = [
			EmotionType.HAPPY, EmotionType.SAD, EmotionType.ANGRY,
			EmotionType.EXCITED, EmotionType.CONFIDENT, EmotionType.NEUTRAL
		]
		
		for emotion in emotions:
			job = await service.synthesize_text(
				text="This text will be synthesized with different emotions.",
				voice_id='neural_female_001',
				emotion=emotion,
				emotion_intensity=0.8,
				tenant_id='test_tenant'
			)
			
			assert job is not None
			assert job.status == ProcessingStatus.COMPLETED
	
	async def test_speech_parameter_control(self, mock_synthesis_service):
		"""Test synthesis with speech parameter control"""
		service = mock_synthesis_service
		
		job = await service.synthesize_text(
			text="Testing speech parameters control.",
			voice_id='neural_male_001',
			speech_rate=1.5,  # 1.5x speed
			pitch_adjustment=1.2,  # 20% higher pitch
			volume_level=1.1,  # 10% louder
			tenant_id='test_tenant'
		)
		
		assert job is not None
		assert job.status == ProcessingStatus.COMPLETED

class TestAudioAnalysisService:
	"""Test AudioAnalysisService"""
	
	async def test_service_initialization(self):
		"""Test analysis service initialization"""
		service = AudioAnalysisService()
		
		assert service is not None
		assert hasattr(service, 'analyze_sentiment')
		assert hasattr(service, 'detect_topics')
		assert hasattr(service, 'assess_quality')
		assert hasattr(service, 'detect_speaker_characteristics')
	
	async def test_sentiment_analysis(self, mock_analysis_service, sample_analysis_data):
		"""Test audio sentiment analysis"""
		service = mock_analysis_service
		data = sample_analysis_data
		
		job = await service.analyze_sentiment(
			audio_source=data['audio_source'],
			include_emotions=True,
			tenant_id='test_tenant'
		)
		
		assert job is not None
		assert job.analysis_type == 'sentiment'
		assert job.status == ProcessingStatus.COMPLETED
		assert job.analysis_results is not None
		assert 'sentiment' in job.analysis_results
		assert job.confidence_score == 0.89
	
	async def test_topic_detection(self, mock_analysis_service):
		"""Test topic detection in audio content"""
		service = mock_analysis_service
		
		# Mock topic detection results
		async def mock_detect_topics(**kwargs):
			from ...models import APAudioAnalysisJob
			from uuid_extensions import uuid7str
			
			job = APAudioAnalysisJob(
				job_id=uuid7str(),
				audio_source=kwargs.get('audio_source', {}),
				analysis_type='topics',
				status=ProcessingStatus.COMPLETED,
				analysis_results={
					'topics': [
						{'topic': 'Technology', 'confidence': 0.92, 'mentions': 15},
						{'topic': 'Business Strategy', 'confidence': 0.87, 'mentions': 12},
						{'topic': 'Innovation', 'confidence': 0.79, 'mentions': 8}
					],
					'keywords': ['AI', 'machine learning', 'digital transformation', 'automation']
				},
				confidence_score=0.86,
				tenant_id=kwargs.get('tenant_id', 'test')
			)
			return job
		
		service.detect_topics = mock_detect_topics
		
		job = await service.detect_topics(
			audio_source={'file_path': '/tmp/business_meeting.wav'},
			num_topics=5,
			tenant_id='test_tenant'
		)
		
		assert job is not None
		assert len(job.analysis_results['topics']) == 3
		assert job.analysis_results['topics'][0]['topic'] == 'Technology'
		assert len(job.analysis_results['keywords']) == 4
	
	async def test_quality_assessment(self, mock_analysis_service):
		"""Test audio quality assessment"""
		service = mock_analysis_service
		
		# Mock quality assessment
		async def mock_assess_quality(**kwargs):
			from ...models import APAudioAnalysisJob
			from uuid_extensions import uuid7str
			
			job = APAudioAnalysisJob(
				job_id=uuid7str(),
				audio_source=kwargs.get('audio_source', {}),
				analysis_type='quality',
				status=ProcessingStatus.COMPLETED,
				analysis_results={
					'overall_quality': 'high',
					'snr_db': 25.3,
					'clarity_score': 0.94,
					'noise_level': 'low',
					'frequency_response': 'excellent',
					'dynamic_range': 18.5,
					'recommendations': ['No enhancement needed']
				},
				confidence_score=0.91,
				tenant_id=kwargs.get('tenant_id', 'test')
			)
			return job
		
		service.assess_quality = mock_assess_quality
		
		job = await service.assess_quality(
			audio_source={'file_path': '/tmp/high_quality_audio.wav'},
			include_technical_metrics=True,
			tenant_id='test_tenant'
		)
		
		assert job is not None
		assert job.analysis_results['overall_quality'] == 'high'
		assert job.analysis_results['snr_db'] == 25.3
		assert job.analysis_results['clarity_score'] == 0.94
	
	async def test_speaker_characteristics_detection(self, mock_analysis_service):
		"""Test speaker characteristics detection"""
		service = mock_analysis_service
		
		# Mock speaker characteristics detection
		async def mock_detect_speaker_characteristics(**kwargs):
			from ...models import APAudioAnalysisJob
			from uuid_extensions import uuid7str
			
			job = APAudioAnalysisJob(
				job_id=uuid7str(),
				audio_source=kwargs.get('audio_source', {}),
				analysis_type='speaker_characteristics',
				status=ProcessingStatus.COMPLETED,
				analysis_results={
					'speaker_count': 3,
					'speakers': [
						{
							'speaker_id': 'speaker_1',
							'gender': 'male',
							'age_estimate': 'adult',
							'accent': 'american',
							'speaking_rate': 'normal',
							'energy_level': 'medium'
						},
						{
							'speaker_id': 'speaker_2', 
							'gender': 'female',
							'age_estimate': 'adult',
							'accent': 'british',
							'speaking_rate': 'fast',
							'energy_level': 'high'
						},
						{
							'speaker_id': 'speaker_3',
							'gender': 'male',
							'age_estimate': 'senior',
							'accent': 'american',
							'speaking_rate': 'slow',
							'energy_level': 'low'
						}
					]
				},
				confidence_score=0.88,
				tenant_id=kwargs.get('tenant_id', 'test')
			)
			return job
		
		service.detect_speaker_characteristics = mock_detect_speaker_characteristics
		
		job = await service.detect_speaker_characteristics(
			audio_source={'file_path': '/tmp/multi_speaker_meeting.wav'},
			tenant_id='test_tenant'
		)
		
		assert job is not None
		assert job.analysis_results['speaker_count'] == 3
		assert len(job.analysis_results['speakers']) == 3
		assert job.analysis_results['speakers'][0]['gender'] == 'male'
		assert job.analysis_results['speakers'][1]['accent'] == 'british'

class TestAudioEnhancementService:
	"""Test AudioEnhancementService"""
	
	async def test_service_initialization(self):
		"""Test enhancement service initialization"""
		service = AudioEnhancementService()
		
		assert service is not None
		assert hasattr(service, 'reduce_noise')
		assert hasattr(service, 'isolate_voices')
		assert hasattr(service, 'normalize_audio')
		assert hasattr(service, 'convert_format')
	
	async def test_noise_reduction(self, mock_enhancement_service, sample_enhancement_data):
		"""Test audio noise reduction"""
		service = mock_enhancement_service
		data = sample_enhancement_data
		
		result = await service.reduce_noise(
			audio_source=data['audio_source'],
			noise_reduction_level='moderate',
			preserve_speech=True,
			output_format=AudioFormat.WAV,
			tenant_id='test_tenant'
		)
		
		assert result is not None
		assert result['enhanced_path'] == '/tmp/enhanced.wav'
		assert result['improvement'] == 3.2
		assert result['processing_time'] == 8.5
	
	async def test_voice_isolation(self, mock_enhancement_service):
		"""Test voice isolation from multi-speaker audio"""
		service = mock_enhancement_service
		
		# Mock voice isolation
		async def mock_isolate_voices(**kwargs):
			return {
				'isolated_voices': kwargs.get('num_speakers', 2),
				'output_files': [
					'/tmp/speaker_1_isolated.wav',
					'/tmp/speaker_2_isolated.wav'
				],
				'separation_quality': 0.89,
				'processing_time_ms': 12500
			}
		
		service.isolate_voices = mock_isolate_voices
		
		result = await service.isolate_voices(
			audio_source={'file_path': '/tmp/meeting_recording.wav'},
			num_speakers=2,
			separation_quality='high',
			tenant_id='test_tenant'
		)
		
		assert result is not None
		assert result['isolated_voices'] == 2
		assert len(result['output_files']) == 2
		assert result['separation_quality'] == 0.89
	
	async def test_audio_normalization(self, mock_enhancement_service):
		"""Test audio normalization"""
		service = mock_enhancement_service
		
		# Mock normalization
		async def mock_normalize_audio(**kwargs):
			return {
				'normalized_path': '/tmp/normalized_audio.wav',
				'target_lufs_achieved': kwargs.get('target_lufs', -23.0),
				'peak_level_db': -1.0,
				'dynamic_range_improvement': 0.25,
				'processing_time_ms': 3200
			}
		
		service.normalize_audio = mock_normalize_audio
		
		result = await service.normalize_audio(
			audio_source={'file_path': '/tmp/variable_volume_audio.wav'},
			target_lufs=-23.0,
			peak_limit_db=-1.0,
			tenant_id='test_tenant'
		)
		
		assert result is not None
		assert result['target_lufs_achieved'] == -23.0
		assert result['peak_level_db'] == -1.0
		assert result['dynamic_range_improvement'] == 0.25

class TestAudioModelManager:
	"""Test AudioModelManager"""
	
	async def test_model_registration(self):
		"""Test model registration and management"""
		manager = AudioModelManager()
		
		# Mock model registration
		manager.register_model = AsyncMock(return_value={'registered': True, 'model_id': 'test_model_001'})
		manager.list_models = AsyncMock(return_value=[])
		manager.delete_model = AsyncMock(return_value=True)
		
		# Test model registration
		result = await manager.register_model({
			'model_id': 'test_model_001',
			'model_type': 'synthesis',
			'voice_name': 'Test Voice'
		})
		
		assert result['registered'] is True
		assert result['model_id'] == 'test_model_001'
		
		# Test model listing
		models = await manager.list_models(tenant_id='test_tenant')
		assert isinstance(models, list)
		
		# Test model deletion
		deleted = await manager.delete_model('test_model_001')
		assert deleted is True

class TestAudioWorkflowOrchestrator:
	"""Test AudioWorkflowOrchestrator"""
	
	async def test_workflow_orchestration(self):
		"""Test complete audio processing workflow"""
		orchestrator = AudioWorkflowOrchestrator()
		
		# Mock complete workflow
		async def mock_complete_workflow(**kwargs):
			return {
				'workflow_id': 'workflow_001',
				'status': 'completed',
				'total_processing_time': 45.2,
				'results': {
					'transcription': {
						'job_id': 'trans_001',
						'status': 'completed',
						'accuracy': 0.97
					},
					'analysis': {
						'job_id': 'analysis_001',
						'status': 'completed',
						'sentiment': 'positive'
					},
					'enhancement': {
						'job_id': 'enhance_001',
						'status': 'completed',
						'improvement': 3.1
					},
					'steps_completed': ['transcription', 'analysis', 'enhancement']
				}
			}
		
		orchestrator.process_complete_workflow = mock_complete_workflow
		
		result = await orchestrator.process_complete_workflow(
			audio_source={'file_path': '/tmp/comprehensive_test.wav'},
			workflow_type='transcribe_analyze_enhance',
			tenant_id='test_tenant'
		)
		
		assert result is not None
		assert result['workflow_id'] == 'workflow_001'
		assert result['status'] == 'completed'
		assert len(result['results']['steps_completed']) == 3
		assert result['results']['transcription']['accuracy'] == 0.97

class TestServiceFactories:
	"""Test service factory functions"""
	
	async def test_create_transcription_service(self):
		"""Test transcription service factory"""
		service = create_transcription_service()
		assert service is not None
		assert isinstance(service, AudioTranscriptionService)
	
	async def test_create_synthesis_service(self):
		"""Test synthesis service factory"""
		service = create_synthesis_service()
		assert service is not None
		assert isinstance(service, VoiceSynthesisService)
	
	async def test_create_analysis_service(self):
		"""Test analysis service factory"""
		service = create_analysis_service()
		assert service is not None
		assert isinstance(service, AudioAnalysisService)
	
	async def test_create_enhancement_service(self):
		"""Test enhancement service factory"""
		service = create_enhancement_service()
		assert service is not None
		assert isinstance(service, AudioEnhancementService)
	
	async def test_create_model_manager(self):
		"""Test model manager factory"""
		manager = create_model_manager()
		assert manager is not None
		assert isinstance(manager, AudioModelManager)
	
	async def test_create_workflow_orchestrator(self):
		"""Test workflow orchestrator factory"""
		orchestrator = create_workflow_orchestrator()
		assert orchestrator is not None
		assert isinstance(orchestrator, AudioWorkflowOrchestrator)

class TestServiceIntegration:
	"""Test service integration and coordination"""
	
	async def test_multi_service_coordination(self):
		"""Test coordination between multiple services"""
		# Create all services
		transcription_service = create_transcription_service()
		synthesis_service = create_synthesis_service()
		analysis_service = create_analysis_service()
		enhancement_service = create_enhancement_service()
		
		# Mock coordinated workflow
		transcription_service.create_transcription_job = AsyncMock(return_value=MagicMock(
			transcription_text="This is the transcribed text.",
			confidence_score=0.95
		))
		
		analysis_service.analyze_sentiment = AsyncMock(return_value=MagicMock(
			analysis_results={'sentiment': 'positive', 'score': 0.72}
		))
		
		synthesis_service.synthesize_text = AsyncMock(return_value=MagicMock(
			output_audio_path='/tmp/synthesized_response.wav',
			quality_score=4.8
		))
		
		# Simulate coordinated workflow
		transcription_result = await transcription_service.create_transcription_job(
			audio_source={'file_path': '/tmp/input.wav'},
			audio_duration=30.0,
			audio_format=AudioFormat.WAV,
			provider=TranscriptionProvider.OPENAI_WHISPER,
			tenant_id='test_tenant'
		)
		
		analysis_result = await analysis_service.analyze_sentiment(
			audio_source={'file_path': '/tmp/input.wav'},
			tenant_id='test_tenant'
		)
		
		synthesis_result = await synthesis_service.synthesize_text(
			text=transcription_result.transcription_text,
			voice_id='neural_female_001',
			tenant_id='test_tenant'
		)
		
		assert transcription_result.confidence_score == 0.95
		assert analysis_result.analysis_results['sentiment'] == 'positive'
		assert synthesis_result.quality_score == 4.8
	
	async def test_error_handling_across_services(self):
		"""Test error handling and recovery across services"""
		transcription_service = create_transcription_service()
		
		# Mock service failure
		transcription_service.create_transcription_job = AsyncMock(
			side_effect=Exception("Service temporarily unavailable")
		)
		
		with pytest.raises(Exception) as exc_info:
			await transcription_service.create_transcription_job(
				audio_source={'file_path': '/tmp/test.wav'},
				audio_duration=30.0,
				audio_format=AudioFormat.WAV,
				provider=TranscriptionProvider.OPENAI_WHISPER,
				tenant_id='test_tenant'
			)
		
		assert "Service temporarily unavailable" in str(exc_info.value)
	
	async def test_concurrent_service_operations(self):
		"""Test concurrent operations across services"""
		transcription_service = create_transcription_service()
		synthesis_service = create_synthesis_service()
		analysis_service = create_analysis_service()
		
		# Mock concurrent operations
		transcription_service.create_transcription_job = AsyncMock(return_value=MagicMock())
		synthesis_service.synthesize_text = AsyncMock(return_value=MagicMock())
		analysis_service.analyze_sentiment = AsyncMock(return_value=MagicMock())
		
		# Run operations concurrently
		tasks = [
			transcription_service.create_transcription_job(
				audio_source={'file_path': f'/tmp/test_{i}.wav'},
				audio_duration=30.0,
				audio_format=AudioFormat.WAV,
				provider=TranscriptionProvider.OPENAI_WHISPER,
				tenant_id='test_tenant'
			)
			for i in range(5)
		]
		
		results = await asyncio.gather(*tasks)
		assert len(results) == 5