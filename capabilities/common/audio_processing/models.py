"""
Audio Processing & Intelligence Data Models

APG-compatible data models for comprehensive audio processing, transcription,
synthesis, and analysis with multi-tenant support and modern async patterns.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic.types import PositiveFloat, PositiveInt


class APGBaseModel(BaseModel):
	"""Base model for all APG capabilities with multi-tenant support"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)
	
	# Core APG fields
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="Tenant identifier for multi-tenancy")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str | None = Field(None, description="User ID who created the record")
	updated_by: str | None = Field(None, description="User ID who last updated the record")


# Enumerations for audio processing

class AudioSessionType(str, Enum):
	"""Types of audio processing sessions"""
	REAL_TIME_TRANSCRIPTION = "real_time_transcription"
	BATCH_TRANSCRIPTION = "batch_transcription"
	VOICE_SYNTHESIS = "voice_synthesis"
	AUDIO_ANALYSIS = "audio_analysis"
	AUDIO_ENHANCEMENT = "audio_enhancement"
	COLLABORATIVE_SESSION = "collaborative_session"
	CUSTOM_MODEL_TRAINING = "custom_model_training"


class AudioFormat(str, Enum):
	"""Supported audio formats"""
	MP3 = "mp3"
	WAV = "wav"
	FLAC = "flac"
	AAC = "aac"
	OGG = "ogg"
	M4A = "m4a"
	WEBM = "webm"
	OPUS = "opus"


class AudioQuality(str, Enum):
	"""Audio quality levels"""
	LOW = "low"
	STANDARD = "standard"
	HIGH = "high"
	PREMIUM = "premium"
	LOSSLESS = "lossless"


class TranscriptionProvider(str, Enum):
	"""Speech recognition providers"""
	OPENAI_WHISPER = "openai_whisper"
	GOOGLE_SPEECH = "google_speech"
	AZURE_COGNITIVE = "azure_cognitive"
	AWS_TRANSCRIBE = "aws_transcribe"
	DEEPGRAM = "deepgram"
	ASSEMBLY_AI = "assembly_ai"
	CUSTOM_MODEL = "custom_model"


class VoiceSynthesisProvider(str, Enum):
	"""Text-to-speech providers"""
	OPENAI_TTS = "openai_tts"
	GOOGLE_TTS = "google_tts"
	AZURE_TTS = "azure_tts"
	AWS_POLLY = "aws_polly"
	ELEVEN_LABS = "eleven_labs"
	CUSTOM_VOICE = "custom_voice"


class ProcessingStatus(str, Enum):
	"""Processing status for audio operations"""
	PENDING = "pending"
	QUEUED = "queued"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	TIMEOUT = "timeout"


class EmotionType(str, Enum):
	"""Emotion types for voice synthesis and analysis"""
	NEUTRAL = "neutral"
	HAPPY = "happy"
	SAD = "sad"
	ANGRY = "angry"
	EXCITED = "excited"
	CALM = "calm"
	CONFIDENT = "confident"
	NERVOUS = "nervous"
	FRIENDLY = "friendly"
	PROFESSIONAL = "professional"
	ENTHUSIASTIC = "enthusiastic"
	CONCERNED = "concerned"
	APOLOGETIC = "apologetic"
	CHEERFUL = "cheerful"
	SERIOUS = "serious"
	EMPATHETIC = "empathetic"
	AUTHORITATIVE = "authoritative"
	CONVERSATIONAL = "conversational"
	NEWS_CAST = "news_cast"
	STORYTELLING = "storytelling"


class SentimentType(str, Enum):
	"""Sentiment analysis types"""
	POSITIVE = "positive"
	NEGATIVE = "negative"
	NEUTRAL = "neutral"
	MIXED = "mixed"


class ContentType(str, Enum):
	"""Audio content types"""
	SPEECH = "speech"
	MUSIC = "music"
	MIXED = "mixed"
	NOISE = "noise"
	SILENCE = "silence"
	CONVERSATION = "conversation"
	PRESENTATION = "presentation"
	INTERVIEW = "interview"
	MEETING = "meeting"
	PODCAST = "podcast"
	VOICEMAIL = "voicemail"


# Core Audio Processing Models

class APAudioSession(APGBaseModel):
	"""
	Audio processing session with real-time collaboration support
	
	Manages audio processing sessions with multi-participant support,
	real-time streaming, and comprehensive configuration options.
	"""
	session_id: str = Field(default_factory=uuid7str, description="Session identifier")
	session_name: str = Field(..., min_length=1, max_length=200, description="Session name")
	session_type: AudioSessionType = Field(..., description="Type of audio session")
	
	# Configuration
	configuration: dict[str, Any] = Field(default_factory=dict, description="Session configuration")
	processing_options: dict[str, Any] = Field(default_factory=dict, description="Processing options")
	
	# Participants and Collaboration
	participants: list[str] = Field(default_factory=list, description="User IDs of participants")
	max_participants: int = Field(default=50, ge=1, le=100, description="Maximum participants")
	real_time_enabled: bool = Field(default=False, description="Real-time processing enabled")
	collaborative_editing: bool = Field(default=False, description="Collaborative editing enabled")
	
	# Session Status
	status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Session status")
	started_at: datetime | None = Field(None, description="Session start time")
	ended_at: datetime | None = Field(None, description="Session end time")
	
	# Performance Metrics
	total_processing_time: float = Field(default=0.0, ge=0, description="Total processing time in seconds")
	total_audio_duration: float = Field(default=0.0, ge=0, description="Total audio duration processed")
	quality_score: float = Field(default=0.0, ge=0, le=100, description="Overall quality score")
	
	@field_validator('participants')
	@classmethod
	def validate_participants(cls, v: list[str]) -> list[str]:
		"""Validate participant list"""
		if len(v) > 100:
			raise ValueError("Maximum 100 participants allowed")
		return list(set(v))  # Remove duplicates
	
	def _log_session_start(self) -> None:
		"""Log session start for monitoring"""
		print(f"[AUDIO_SESSION] Started session {self.session_id} with {len(self.participants)} participants")
	
	def _log_session_end(self) -> None:
		"""Log session completion for monitoring"""
		duration = self.get_session_duration()
		print(f"[AUDIO_SESSION] Completed session {self.session_id} in {duration:.2f} seconds")
	
	def get_session_duration(self) -> float:
		"""Get session duration in seconds"""
		if self.started_at and self.ended_at:
			return (self.ended_at - self.started_at).total_seconds()
		elif self.started_at:
			return (datetime.utcnow() - self.started_at).total_seconds()
		return 0.0
	
	def add_participant(self, user_id: str) -> bool:
		"""Add participant to session"""
		if len(self.participants) >= self.max_participants:
			return False
		if user_id not in self.participants:
			self.participants.append(user_id)
			self.updated_at = datetime.utcnow()
		return True
	
	def remove_participant(self, user_id: str) -> bool:
		"""Remove participant from session"""
		if user_id in self.participants:
			self.participants.remove(user_id)
			self.updated_at = datetime.utcnow()
			return True
		return False


class APTranscriptionJob(APGBaseModel):
	"""
	Speech recognition job with advanced features and speaker diarization
	
	Handles speech-to-text conversion with speaker identification,
	custom vocabularies, and comprehensive accuracy metrics.
	"""
	job_id: str = Field(default_factory=uuid7str, description="Job identifier")
	session_id: str | None = Field(None, description="Associated session ID")
	
	# Audio Source
	audio_source: dict[str, Any] = Field(..., description="Audio source configuration")
	audio_duration: float = Field(..., gt=0, description="Audio duration in seconds")
	audio_format: AudioFormat = Field(..., description="Audio format")
	audio_quality_score: float = Field(default=0.0, ge=0, le=100, description="Input audio quality")
	
	# Transcription Configuration
	provider: TranscriptionProvider = Field(..., description="Transcription provider")
	model_name: str | None = Field(None, description="Specific model used")
	language_code: str = Field(..., description="ISO language code (e.g., en-US)")
	language_confidence: float = Field(default=1.0, ge=0, le=1, description="Language detection confidence")
	
	# Advanced Features
	speaker_diarization: bool = Field(default=True, description="Enable speaker diarization")
	custom_vocabulary: list[str] = Field(default_factory=list, description="Custom vocabulary terms")
	enable_punctuation: bool = Field(default=True, description="Enable automatic punctuation")
	enable_timestamps: bool = Field(default=True, description="Enable word-level timestamps")
	profanity_filter: bool = Field(default=False, description="Enable profanity filtering")
	confidence_threshold: float = Field(default=0.8, ge=0, le=1, description="Minimum confidence threshold")
	
	# Results
	transcript_text: str = Field(default="", description="Final transcript text")
	word_count: int = Field(default=0, ge=0, description="Total word count")
	speaker_count: int = Field(default=0, ge=0, description="Number of identified speakers")
	
	# Quality Metrics
	overall_confidence: float = Field(default=0.0, ge=0, le=1, description="Overall transcription confidence")
	accuracy_estimate: float = Field(default=0.0, ge=0, le=1, description="Estimated accuracy")
	
	# Detailed Results
	word_level_data: list[dict[str, Any]] = Field(default_factory=list, description="Word-level timestamps and confidence")
	speaker_segments: list[dict[str, Any]] = Field(default_factory=list, description="Speaker diarization segments")
	sentence_segments: list[dict[str, Any]] = Field(default_factory=list, description="Sentence-level segmentation")
	
	# Processing Performance
	processing_time: float = Field(default=0.0, ge=0, description="Processing time in seconds")
	processing_cost: Decimal = Field(default=Decimal("0.00"), description="Processing cost")
	tokens_used: int = Field(default=0, ge=0, description="API tokens consumed")
	
	# Status and Error Handling
	status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job status")
	error_message: str | None = Field(None, description="Error message if failed")
	processing_warnings: list[str] = Field(default_factory=list, description="Processing warnings")
	
	@field_validator('custom_vocabulary')
	@classmethod
	def validate_vocabulary(cls, v: list[str]) -> list[str]:
		"""Validate custom vocabulary list"""
		if len(v) > 1000:
			raise ValueError("Maximum 1000 custom vocabulary terms allowed")
		return [term.strip() for term in v if term.strip()]
	
	def _log_transcription_start(self) -> None:
		"""Log transcription start for monitoring"""
		print(f"[TRANSCRIPTION] Started job {self.job_id} for {self.audio_duration:.2f}s audio")
	
	def _log_transcription_complete(self) -> None:
		"""Log transcription completion for monitoring"""
		print(f"[TRANSCRIPTION] Completed job {self.job_id} with {self.word_count} words, {self.overall_confidence:.3f} confidence")
	
	def get_processing_speed(self) -> float:
		"""Get processing speed as multiplier of real-time"""
		if self.processing_time > 0:
			return self.audio_duration / self.processing_time
		return 0.0
	
	def get_cost_per_minute(self) -> Decimal:
		"""Get cost per minute of audio"""
		if self.audio_duration > 0:
			return self.processing_cost / Decimal(str(self.audio_duration / 60))
		return Decimal("0.00")
	
	def get_speakers_summary(self) -> dict[str, float]:
		"""Get speaking time summary per speaker"""
		speaker_times: dict[str, float] = {}
		for segment in self.speaker_segments:
			speaker = segment.get('speaker', 'Unknown')
			duration = segment.get('end_time', 0) - segment.get('start_time', 0)
			speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
		return speaker_times
	
	def get_low_confidence_words(self, threshold: float = 0.6) -> list[dict[str, Any]]:
		"""Get words below confidence threshold"""
		return [
			word for word in self.word_level_data
			if word.get('confidence', 1.0) < threshold
		]


class APVoiceSynthesisJob(APGBaseModel):
	"""
	Voice synthesis job with emotion control and voice cloning
	
	Handles text-to-speech generation with advanced voice characteristics,
	emotional expression, and custom voice models.
	"""
	job_id: str = Field(default_factory=uuid7str, description="Job identifier")
	session_id: str | None = Field(None, description="Associated session ID")
	
	# Input Configuration
	input_text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
	input_language: str = Field(..., description="ISO language code")
	input_ssml: str | None = Field(None, description="SSML markup if used")
	text_preprocessing: dict[str, Any] = Field(default_factory=dict, description="Text preprocessing options")
	
	# Voice Configuration
	provider: VoiceSynthesisProvider = Field(..., description="TTS provider")
	voice_id: str = Field(..., description="Voice identifier")
	voice_name: str | None = Field(None, description="Human-readable voice name")
	voice_gender: str | None = Field(None, description="Voice gender")
	voice_age: str | None = Field(None, description="Voice age category")
	voice_style: str | None = Field(None, description="Voice style")
	custom_voice_model: str | None = Field(None, description="Custom voice model ID")
	
	# Audio Parameters
	speaking_rate: float = Field(default=1.0, ge=0.1, le=3.0, description="Speaking rate multiplier")
	pitch_adjustment: float = Field(default=0.0, ge=-20.0, le=20.0, description="Pitch adjustment in semitones")
	volume_level: float = Field(default=1.0, ge=0.1, le=2.0, description="Volume multiplier")
	emphasis_level: str = Field(default="moderate", description="Emphasis level")
	
	# Emotion and Expression
	primary_emotion: EmotionType = Field(default=EmotionType.NEUTRAL, description="Primary emotion")
	emotion_intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Emotion intensity")
	secondary_emotions: list[EmotionType] = Field(default_factory=list, description="Secondary emotions")
	emotional_variation: bool = Field(default=False, description="Enable emotional variation")
	
	# Output Configuration
	output_format: AudioFormat = Field(default=AudioFormat.MP3, description="Output audio format")
	sample_rate: PositiveInt = Field(default=22050, description="Sample rate in Hz")
	bit_depth: int = Field(default=16, ge=8, le=32, description="Bit depth")
	audio_quality: AudioQuality = Field(default=AudioQuality.STANDARD, description="Audio quality level")
	
	# Generated Audio
	output_file_path: str | None = Field(None, description="Path to generated audio file")
	output_file_size: int = Field(default=0, ge=0, description="File size in bytes")
	audio_duration: float = Field(default=0.0, ge=0, description="Generated audio duration")
	audio_hash: str | None = Field(None, description="SHA-256 hash of audio file")
	
	# Quality Metrics
	synthesis_quality: float = Field(default=0.0, ge=0, le=100, description="Synthesis quality score")
	naturalness_score: float = Field(default=0.0, ge=0, le=100, description="Voice naturalness score")
	intelligibility_score: float = Field(default=0.0, ge=0, le=100, description="Speech intelligibility score")
	emotion_accuracy: float = Field(default=0.0, ge=0, le=100, description="Emotion expression accuracy")
	
	# Processing Performance
	processing_time: float = Field(default=0.0, ge=0, description="Processing time in seconds")
	processing_cost: Decimal = Field(default=Decimal("0.00"), description="Processing cost")
	characters_processed: int = Field(default=0, ge=0, description="Characters processed")
	
	# Status and Error Handling
	status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job status")
	error_message: str | None = Field(None, description="Error message if failed")
	processing_warnings: list[str] = Field(default_factory=list, description="Processing warnings")
	
	@field_validator('input_text')
	@classmethod
	def validate_input_text(cls, v: str) -> str:
		"""Validate input text"""
		if len(v.strip()) == 0:
			raise ValueError("Input text cannot be empty")
		return v.strip()
	
	@field_validator('secondary_emotions')
	@classmethod
	def validate_secondary_emotions(cls, v: list[EmotionType]) -> list[EmotionType]:
		"""Validate secondary emotions list"""
		if len(v) > 3:
			raise ValueError("Maximum 3 secondary emotions allowed")
		return list(set(v))  # Remove duplicates
	
	def _log_synthesis_start(self) -> None:
		"""Log synthesis start for monitoring"""
		print(f"[VOICE_SYNTHESIS] Started job {self.job_id} for {len(self.input_text)} characters")
	
	def _log_synthesis_complete(self) -> None:
		"""Log synthesis completion for monitoring"""
		print(f"[VOICE_SYNTHESIS] Completed job {self.job_id}, generated {self.audio_duration:.2f}s audio")
	
	def get_synthesis_speed(self) -> float:
		"""Get synthesis speed as multiple of real-time"""
		if self.processing_time > 0 and self.audio_duration > 0:
			return self.audio_duration / self.processing_time
		return 0.0
	
	def get_cost_per_character(self) -> Decimal:
		"""Get cost per character"""
		if self.characters_processed > 0:
			return self.processing_cost / Decimal(str(self.characters_processed))
		return Decimal("0.00")
	
	def get_voice_characteristics(self) -> dict[str, Any]:
		"""Get voice characteristics summary"""
		return {
			'voice_id': self.voice_id,
			'voice_name': self.voice_name,
			'gender': self.voice_gender,
			'age': self.voice_age,
			'style': self.voice_style,
			'language': self.input_language,
			'provider': self.provider,
			'custom_model': self.custom_voice_model
		}


class APAudioAnalysisJob(APGBaseModel):
	"""
	Audio content analysis with AI-powered insights
	
	Provides comprehensive audio analysis including sentiment detection,
	content classification, speaker characteristics, and intelligent insights.
	"""
	job_id: str = Field(default_factory=uuid7str, description="Job identifier")
	session_id: str | None = Field(None, description="Associated session ID")
	audio_source_id: str = Field(..., description="Source audio identifier")
	
	# Analysis Configuration
	analysis_types: list[str] = Field(..., description="Types of analysis to perform")
	provider: str = Field(..., description="Analysis service provider")
	model_name: str | None = Field(None, description="Specific model used")
	language_hint: str | None = Field(None, description="Language hint for analysis")
	
	# Content Classification
	content_type: ContentType | None = Field(None, description="Detected content type")
	content_category: str | None = Field(None, description="Content category")
	content_confidence: float = Field(default=0.0, ge=0, le=1, description="Classification confidence")
	
	# Content Distribution
	speech_percentage: float = Field(default=0.0, ge=0, le=100, description="Percentage of speech")
	music_percentage: float = Field(default=0.0, ge=0, le=100, description="Percentage of music")
	noise_percentage: float = Field(default=0.0, ge=0, le=100, description="Percentage of noise")
	silence_percentage: float = Field(default=0.0, ge=0, le=100, description="Percentage of silence")
	
	# Sentiment Analysis
	overall_sentiment: SentimentType | None = Field(None, description="Overall sentiment")
	sentiment_confidence: float = Field(default=0.0, ge=0, le=1, description="Sentiment confidence")
	sentiment_scores: dict[str, float] = Field(default_factory=dict, description="Detailed sentiment scores")
	emotional_tone: dict[str, float] = Field(default_factory=dict, description="Emotional tone breakdown")
	
	# Topic and Content Analysis
	detected_topics: list[str] = Field(default_factory=list, description="Identified topics")
	topic_confidence: list[float] = Field(default_factory=list, description="Topic confidence scores")
	key_phrases: list[str] = Field(default_factory=list, description="Important phrases")
	named_entities: list[dict[str, Any]] = Field(default_factory=list, description="Named entities")
	
	# Speaker Analysis
	speaker_count: int = Field(default=0, ge=0, description="Number of speakers detected")
	speaker_characteristics: dict[str, Any] = Field(default_factory=dict, description="Speaker characteristics")
	speaker_emotions: dict[str, Any] = Field(default_factory=dict, description="Speaker emotional states")
	speaking_pace: float = Field(default=0.0, ge=0, description="Speaking pace (words per minute)")
	
	# Content Insights
	summary_text: str | None = Field(None, description="AI-generated summary")
	key_moments: list[dict[str, Any]] = Field(default_factory=list, description="Important timestamps")
	action_items: list[str] = Field(default_factory=list, description="Extracted action items")
	questions_identified: list[str] = Field(default_factory=list, description="Questions found in audio")
	
	# Quality Assessment
	audio_quality_score: float = Field(default=0.0, ge=0, le=100, description="Audio quality score")
	analysis_confidence: float = Field(default=0.0, ge=0, le=1, description="Overall analysis confidence")
	engagement_score: float = Field(default=0.0, ge=0, le=100, description="Content engagement score")
	
	# Processing Performance
	processing_time: float = Field(default=0.0, ge=0, description="Processing time in seconds")
	processing_cost: Decimal = Field(default=Decimal("0.00"), description="Processing cost")
	
	# Status and Error Handling
	status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job status")
	error_message: str | None = Field(None, description="Error message if failed")
	processing_warnings: list[str] = Field(default_factory=list, description="Processing warnings")
	
	@field_validator('analysis_types')
	@classmethod
	def validate_analysis_types(cls, v: list[str]) -> list[str]:
		"""Validate analysis types"""
		valid_types = {
			'sentiment', 'topics', 'entities', 'summary', 'quality',
			'speaker_analysis', 'emotion', 'engagement', 'content_classification'
		}
		invalid_types = set(v) - valid_types
		if invalid_types:
			raise ValueError(f"Invalid analysis types: {invalid_types}")
		return list(set(v))  # Remove duplicates
	
	def _log_analysis_start(self) -> None:
		"""Log analysis start for monitoring"""
		print(f"[AUDIO_ANALYSIS] Started job {self.job_id} with {len(self.analysis_types)} analysis types")
	
	def _log_analysis_complete(self) -> None:
		"""Log analysis completion for monitoring"""
		print(f"[AUDIO_ANALYSIS] Completed job {self.job_id} with {self.analysis_confidence:.3f} confidence")
	
	def get_top_topics(self, limit: int = 5) -> list[dict[str, Any]]:
		"""Get top topics with confidence scores"""
		topics = []
		for i, topic in enumerate(self.detected_topics[:limit]):
			confidence = self.topic_confidence[i] if i < len(self.topic_confidence) else 0.0
			topics.append({'topic': topic, 'confidence': confidence})
		return sorted(topics, key=lambda x: x['confidence'], reverse=True)
	
	def get_sentiment_summary(self) -> dict[str, Any]:
		"""Get comprehensive sentiment summary"""
		return {
			'overall': self.overall_sentiment,
			'confidence': self.sentiment_confidence,
			'scores': self.sentiment_scores,
			'emotions': self.emotional_tone
		}
	
	def calculate_content_distribution(self) -> dict[str, float]:
		"""Calculate content type distribution"""
		total = self.speech_percentage + self.music_percentage + self.noise_percentage + self.silence_percentage
		if total == 0:
			return {}
		
		return {
			'speech': self.speech_percentage / total * 100,
			'music': self.music_percentage / total * 100,
			'noise': self.noise_percentage / total * 100,
			'silence': self.silence_percentage / total * 100
		}


class APVoiceModel(APGBaseModel):
	"""
	Custom voice model for voice cloning and synthesis
	
	Manages custom voice models created from audio samples,
	including training data, quality metrics, and usage permissions.
	"""
	model_id: str = Field(default_factory=uuid7str, description="Model identifier")
	voice_name: str = Field(..., min_length=1, max_length=100, description="Voice name")
	voice_description: str | None = Field(None, max_length=500, description="Voice description")
	
	# Training Configuration
	training_audio_samples: list[str] = Field(..., description="Training audio file IDs")
	training_duration: float = Field(..., gt=0, description="Total training audio duration")
	minimum_sample_duration: float = Field(default=30.0, description="Minimum sample duration in seconds")
	training_language: str = Field(..., description="Primary training language")
	additional_languages: list[str] = Field(default_factory=list, description="Additional supported languages")
	
	# Voice Characteristics
	voice_gender: str | None = Field(None, description="Voice gender")
	voice_age: str | None = Field(None, description="Voice age category")
	voice_accent: str | None = Field(None, description="Voice accent/dialect")
	voice_style: str | None = Field(None, description="Voice style characteristics")
	
	# Model Quality
	training_quality_score: float = Field(default=0.0, ge=0, le=100, description="Training quality score")
	voice_similarity_score: float = Field(default=0.0, ge=0, le=100, description="Similarity to original voice")
	synthesis_quality_score: float = Field(default=0.0, ge=0, le=100, description="Synthesis quality score")
	naturalness_score: float = Field(default=0.0, ge=0, le=100, description="Voice naturalness score")
	
	# Capabilities
	emotion_support: list[EmotionType] = Field(default_factory=list, description="Supported emotions")
	style_variations: list[str] = Field(default_factory=list, description="Supported style variations")
	speaking_rate_range: tuple[float, float] = Field(default=(0.5, 2.0), description="Supported rate range")
	pitch_range: tuple[float, float] = Field(default=(-10.0, 10.0), description="Supported pitch range")
	
	# Training Performance
	training_time: float = Field(default=0.0, ge=0, description="Training time in seconds")
	training_cost: Decimal = Field(default=Decimal("0.00"), description="Training cost")
	training_iterations: int = Field(default=0, ge=0, description="Training iterations")
	
	# Model Status
	training_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Training status")
	model_version: str = Field(default="1.0.0", description="Model version")
	is_active: bool = Field(default=True, description="Model is active")
	is_public: bool = Field(default=False, description="Model is publicly available")
	
	# Usage and Permissions
	usage_permissions: list[str] = Field(default_factory=list, description="User IDs with usage permissions")
	usage_count: int = Field(default=0, ge=0, description="Number of times used")
	last_used: datetime | None = Field(None, description="Last usage timestamp")
	
	# Storage and Management
	model_file_path: str | None = Field(None, description="Path to model file")
	model_file_size: int = Field(default=0, ge=0, description="Model file size in bytes")
	backup_locations: list[str] = Field(default_factory=list, description="Backup storage locations")
	
	@field_validator('training_audio_samples')
	@classmethod
	def validate_training_samples(cls, v: list[str]) -> list[str]:
		"""Validate training audio samples"""
		if len(v) < 1:
			raise ValueError("At least 1 training sample required")
		if len(v) > 100:
			raise ValueError("Maximum 100 training samples allowed")
		return v
	
	@field_validator('emotion_support')
	@classmethod
	def validate_emotion_support(cls, v: list[EmotionType]) -> list[EmotionType]:
		"""Validate emotion support list"""
		return list(set(v))  # Remove duplicates
	
	def _log_training_start(self) -> None:
		"""Log training start for monitoring"""
		print(f"[VOICE_MODEL] Started training {self.model_id} with {len(self.training_audio_samples)} samples")
	
	def _log_training_complete(self) -> None:
		"""Log training completion for monitoring"""
		print(f"[VOICE_MODEL] Completed training {self.model_id} with {self.training_quality_score:.2f} quality score")
	
	def can_user_access(self, user_id: str) -> bool:
		"""Check if user can access this voice model"""
		return (
			user_id == self.created_by or
			user_id in self.usage_permissions or
			self.is_public
		)
	
	def add_usage_permission(self, user_id: str) -> None:
		"""Add usage permission for user"""
		if user_id not in self.usage_permissions:
			self.usage_permissions.append(user_id)
			self.updated_at = datetime.utcnow()
	
	def remove_usage_permission(self, user_id: str) -> None:
		"""Remove usage permission for user"""
		if user_id in self.usage_permissions:
			self.usage_permissions.remove(user_id)
			self.updated_at = datetime.utcnow()
	
	def record_usage(self) -> None:
		"""Record model usage"""
		self.usage_count += 1
		self.last_used = datetime.utcnow()
		self.updated_at = datetime.utcnow()
	
	def get_supported_capabilities(self) -> dict[str, Any]:
		"""Get model capabilities summary"""
		return {
			'emotions': self.emotion_support,
			'styles': self.style_variations,
			'languages': [self.training_language] + self.additional_languages,
			'rate_range': self.speaking_rate_range,
			'pitch_range': self.pitch_range,
			'quality_scores': {
				'training': self.training_quality_score,
				'similarity': self.voice_similarity_score,
				'synthesis': self.synthesis_quality_score,
				'naturalness': self.naturalness_score
			}
		}


class APAudioProcessingMetrics(APGBaseModel):
	"""
	Performance metrics and analytics for audio processing operations
	
	Tracks system performance, quality metrics, and usage analytics
	across all audio processing capabilities.
	"""
	metrics_id: str = Field(default_factory=uuid7str, description="Metrics identifier")
	
	# Time Period
	period_start: datetime = Field(..., description="Metrics period start")
	period_end: datetime = Field(..., description="Metrics period end")
	metric_type: str = Field(..., description="Type of metrics (hourly, daily, weekly)")
	
	# Processing Volume
	total_jobs_processed: int = Field(default=0, ge=0, description="Total jobs processed")
	transcription_jobs: int = Field(default=0, ge=0, description="Transcription jobs")
	synthesis_jobs: int = Field(default=0, ge=0, description="Voice synthesis jobs")
	analysis_jobs: int = Field(default=0, ge=0, description="Audio analysis jobs")
	enhancement_jobs: int = Field(default=0, ge=0, description="Audio enhancement jobs")
	
	# Processing Performance
	average_processing_time: float = Field(default=0.0, ge=0, description="Average processing time")
	total_audio_processed: float = Field(default=0.0, ge=0, description="Total audio duration processed")
	processing_speed_multiplier: float = Field(default=0.0, ge=0, description="Average speed vs real-time")
	
	# Quality Metrics
	average_quality_score: float = Field(default=0.0, ge=0, le=100, description="Average quality score")
	average_confidence_score: float = Field(default=0.0, ge=0, le=1, description="Average confidence score")
	success_rate: float = Field(default=0.0, ge=0, le=100, description="Job success rate")
	
	# Resource Usage
	total_processing_cost: Decimal = Field(default=Decimal("0.00"), description="Total processing cost")
	cpu_hours_used: float = Field(default=0.0, ge=0, description="CPU hours consumed")
	memory_gb_hours: float = Field(default=0.0, ge=0, description="Memory GB-hours consumed")
	storage_gb_used: float = Field(default=0.0, ge=0, description="Storage GB used")
	
	# User Engagement
	active_users: int = Field(default=0, ge=0, description="Number of active users")
	new_users: int = Field(default=0, ge=0, description="Number of new users")
	session_count: int = Field(default=0, ge=0, description="Number of sessions")
	average_session_duration: float = Field(default=0.0, ge=0, description="Average session duration")
	
	# Error and Reliability
	error_count: int = Field(default=0, ge=0, description="Total errors")
	timeout_count: int = Field(default=0, ge=0, description="Timeout occurrences")
	retry_count: int = Field(default=0, ge=0, description="Retry attempts")
	uptime_percentage: float = Field(default=100.0, ge=0, le=100, description="System uptime")
	
	# Feature Usage
	feature_usage: dict[str, int] = Field(default_factory=dict, description="Feature usage statistics")
	provider_usage: dict[str, int] = Field(default_factory=dict, description="Provider usage statistics")
	language_usage: dict[str, int] = Field(default_factory=dict, description="Language usage statistics")
	
	def _log_metrics_calculation(self) -> None:
		"""Log metrics calculation for monitoring"""
		print(f"[METRICS] Calculated metrics for {self.metric_type} period: {self.total_jobs_processed} jobs")
	
	def calculate_throughput(self) -> float:
		"""Calculate jobs per hour throughput"""
		period_hours = (self.period_end - self.period_start).total_seconds() / 3600
		if period_hours > 0:
			return self.total_jobs_processed / period_hours
		return 0.0
	
	def calculate_efficiency_score(self) -> float:
		"""Calculate overall system efficiency score"""
		factors = []
		
		# Success rate factor
		factors.append(self.success_rate)
		
		# Processing speed factor
		if self.processing_speed_multiplier > 0:
			speed_score = min(100, self.processing_speed_multiplier * 10)
			factors.append(speed_score)
		
		# Uptime factor
		factors.append(self.uptime_percentage)
		
		# Quality factor
		factors.append(self.average_quality_score)
		
		return sum(factors) / len(factors) if factors else 0.0
	
	def get_cost_per_minute(self) -> Decimal:
		"""Calculate cost per minute of audio processed"""
		if self.total_audio_processed > 0:
			return self.total_processing_cost / Decimal(str(self.total_audio_processed / 60))
		return Decimal("0.00")
	
	def get_usage_summary(self) -> dict[str, Any]:
		"""Get comprehensive usage summary"""
		return {
			'period': f"{self.period_start.isoformat()} to {self.period_end.isoformat()}",
			'jobs': {
				'total': self.total_jobs_processed,
				'transcription': self.transcription_jobs,
				'synthesis': self.synthesis_jobs,
				'analysis': self.analysis_jobs,
				'enhancement': self.enhancement_jobs
			},
			'performance': {
				'avg_processing_time': self.average_processing_time,
				'speed_multiplier': self.processing_speed_multiplier,
				'success_rate': self.success_rate,
				'efficiency_score': self.calculate_efficiency_score()
			},
			'users': {
				'active': self.active_users,
				'new': self.new_users,
				'sessions': self.session_count,
				'avg_session_duration': self.average_session_duration
			},
			'costs': {
				'total': str(self.total_processing_cost),
				'per_minute': str(self.get_cost_per_minute()),
				'per_job': str(self.total_processing_cost / max(1, self.total_jobs_processed))
			}
		}