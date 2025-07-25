"""
Audio Processing Models

Database models for comprehensive audio processing, transcription,
synthesis, and analysis with multi-provider support and quality tracking.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class APAudioFile(Model, AuditMixin, BaseMixin):
	"""
	Audio file management with metadata and processing status.
	
	Stores information about uploaded audio files including format,
	quality metrics, and processing history.
	"""
	__tablename__ = 'ap_audio_file'
	
	# Identity
	file_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# File Information
	original_filename = Column(String(500), nullable=False)
	file_path = Column(String(1000), nullable=False)  # Storage path
	file_size_bytes = Column(Integer, nullable=False)
	mime_type = Column(String(100), nullable=False)
	file_hash = Column(String(64), nullable=False, index=True)  # SHA-256 for deduplication
	
	# Audio Properties
	format = Column(String(20), nullable=False, index=True)  # mp3, wav, flac, m4a, etc.
	duration_seconds = Column(Float, nullable=False)
	sample_rate = Column(Integer, nullable=True)  # Hz
	bit_depth = Column(Integer, nullable=True)  # bits
	channels = Column(Integer, nullable=True)  # mono=1, stereo=2
	bitrate = Column(Integer, nullable=True)  # kbps
	
	# Quality Metrics
	audio_quality_score = Column(Float, default=0.0)  # 0-100 quality rating
	noise_level = Column(Float, nullable=True)  # dB
	signal_to_noise_ratio = Column(Float, nullable=True)  # dB
	dynamic_range = Column(Float, nullable=True)  # dB
	peak_amplitude = Column(Float, nullable=True)
	rms_level = Column(Float, nullable=True)
	
	# Content Analysis
	detected_language = Column(String(10), nullable=True)  # ISO language code
	speech_detected = Column(Boolean, nullable=True)
	music_detected = Column(Boolean, nullable=True)
	silence_percentage = Column(Float, nullable=True)  # Percentage of silence
	speaker_count = Column(Integer, nullable=True)  # Estimated number of speakers
	
	# Processing Status
	processing_status = Column(String(20), default='uploaded', index=True)  # uploaded, processing, completed, failed
	upload_source = Column(String(50), nullable=True)  # web, api, mobile, integration
	uploaded_by = Column(String(36), nullable=True, index=True)  # User ID
	
	# Processing History
	total_processing_jobs = Column(Integer, default=0)
	successful_jobs = Column(Integer, default=0)
	failed_jobs = Column(Integer, default=0)
	last_processed = Column(DateTime, nullable=True, index=True)
	
	# Storage Management
	storage_class = Column(String(20), default='standard')  # standard, cold, archive
	retention_policy = Column(String(50), nullable=True)
	expires_at = Column(DateTime, nullable=True, index=True)
	is_temporary = Column(Boolean, default=False)
	
	# Privacy and Compliance
	contains_pii = Column(Boolean, nullable=True)
	privacy_level = Column(String(20), default='internal')  # public, internal, confidential, restricted
	consent_obtained = Column(Boolean, default=False)
	gdpr_compliant = Column(Boolean, default=True)
	
	# Relationships
	transcriptions = relationship("APTranscription", back_populates="audio_file", cascade="all, delete-orphan")
	analyses = relationship("APAudioAnalysis", back_populates="audio_file", cascade="all, delete-orphan")
	processing_jobs = relationship("APProcessingJob", back_populates="audio_file", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<APAudioFile {self.original_filename} ({self.format}, {self.duration_seconds}s)>"
	
	def get_duration_formatted(self) -> str:
		"""Get formatted duration string (HH:MM:SS)"""
		hours = int(self.duration_seconds // 3600)
		minutes = int((self.duration_seconds % 3600) // 60)
		seconds = int(self.duration_seconds % 60)
		
		if hours > 0:
			return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
		else:
			return f"{minutes:02d}:{seconds:02d}"
	
	def get_file_size_formatted(self) -> str:
		"""Get formatted file size string"""
		size = self.file_size_bytes
		
		if size < 1024:
			return f"{size} B"
		elif size < 1024 * 1024:
			return f"{size / 1024:.1f} KB"
		elif size < 1024 * 1024 * 1024:
			return f"{size / (1024 * 1024):.1f} MB"
		else:
			return f"{size / (1024 * 1024 * 1024):.1f} GB"
	
	def calculate_processing_success_rate(self) -> float:
		"""Calculate success rate of processing jobs"""
		if self.total_processing_jobs == 0:
			return 0.0
		return (self.successful_jobs / self.total_processing_jobs) * 100
	
	def is_suitable_for_transcription(self) -> bool:
		"""Check if audio file is suitable for transcription"""
		return (self.speech_detected is not False and  # None or True
				self.duration_seconds >= 1.0 and  # At least 1 second
				self.audio_quality_score >= 30)  # Minimum quality threshold
	
	def is_expired(self) -> bool:
		"""Check if file has expired based on retention policy"""
		return self.expires_at is not None and datetime.utcnow() > self.expires_at
	
	def update_quality_metrics(self, quality_data: Dict[str, Any]) -> None:
		"""Update audio quality metrics from analysis"""
		self.audio_quality_score = quality_data.get('quality_score', self.audio_quality_score)
		self.noise_level = quality_data.get('noise_level', self.noise_level)
		self.signal_to_noise_ratio = quality_data.get('snr', self.signal_to_noise_ratio)
		self.dynamic_range = quality_data.get('dynamic_range', self.dynamic_range)
		self.peak_amplitude = quality_data.get('peak_amplitude', self.peak_amplitude)
		self.rms_level = quality_data.get('rms_level', self.rms_level)
		self.silence_percentage = quality_data.get('silence_percentage', self.silence_percentage)


class APTranscription(Model, AuditMixin, BaseMixin):
	"""
	Audio transcription results with accuracy metrics and speaker identification.
	
	Stores speech-to-text results with detailed accuracy metrics,
	speaker diarization, and word-level timestamps.
	"""
	__tablename__ = 'ap_transcription'
	
	# Identity
	transcription_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	audio_file_id = Column(String(36), ForeignKey('ap_audio_file.file_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Transcription Configuration
	provider = Column(String(50), nullable=False, index=True)  # openai, google, azure, aws, custom
	model_used = Column(String(100), nullable=True)
	language = Column(String(10), nullable=False, index=True)  # ISO language code
	language_confidence = Column(Float, nullable=True)  # 0-1 confidence in language detection
	
	# Processing Configuration
	enable_speaker_diarization = Column(Boolean, default=False)
	enable_punctuation = Column(Boolean, default=True)
	enable_word_timestamps = Column(Boolean, default=True)
	custom_vocabulary = Column(JSON, default=list)  # Custom words/phrases
	profanity_filter = Column(Boolean, default=False)
	
	# Transcription Results
	transcript_text = Column(Text, nullable=False)
	confidence_score = Column(Float, nullable=True)  # Overall confidence 0-1
	word_count = Column(Integer, default=0)
	speaker_count = Column(Integer, default=1)
	
	# Detailed Results
	word_level_data = Column(JSON, nullable=True)  # Word timestamps and confidence
	speaker_segments = Column(JSON, nullable=True)  # Speaker diarization results
	sentence_segments = Column(JSON, nullable=True)  # Sentence-level segmentation
	
	# Quality Metrics
	accuracy_estimate = Column(Float, nullable=True)  # Estimated accuracy 0-1
	audio_quality_impact = Column(Float, nullable=True)  # How audio quality affected transcription
	processing_warnings = Column(JSON, default=list)  # Warnings during processing
	recognition_errors = Column(JSON, default=list)  # Detected recognition errors
	
	# Processing Performance
	processing_time_seconds = Column(Float, nullable=True)
	processing_cost = Column(Float, nullable=True)
	tokens_used = Column(Integer, nullable=True)  # For API-based services
	
	# Status and Metadata
	status = Column(String(20), default='completed', index=True)  # processing, completed, failed
	requested_by = Column(String(36), nullable=True, index=True)  # User ID
	processing_metadata = Column(JSON, default=dict)  # Provider-specific metadata
	
	# Export and Integration
	export_formats = Column(JSON, default=list)  # Available export formats
	external_references = Column(JSON, default=dict)  # References to external systems
	
	# Relationships
	audio_file = relationship("APAudioFile", back_populates="transcriptions")
	
	def __repr__(self):
		return f"<APTranscription {self.transcription_id} ({self.language}, {self.provider})>"
	
	def get_transcript_summary(self, max_length: int = 200) -> str:
		"""Get truncated transcript for summary display"""
		if len(self.transcript_text) <= max_length:
			return self.transcript_text
		return self.transcript_text[:max_length] + "..."
	
	def get_speakers_list(self) -> List[str]:
		"""Get list of identified speakers"""
		if not self.speaker_segments:
			return []
		
		speakers = set()
		for segment in self.speaker_segments:
			if 'speaker' in segment:
				speakers.add(segment['speaker'])
		
		return sorted(list(speakers))
	
	def get_words_with_low_confidence(self, threshold: float = 0.6) -> List[Dict[str, Any]]:
		"""Get words with confidence below threshold"""
		if not self.word_level_data:
			return []
		
		low_confidence_words = []
		for word_data in self.word_level_data:
			if word_data.get('confidence', 1.0) < threshold:
				low_confidence_words.append(word_data)
		
		return low_confidence_words
	
	def calculate_speaking_time_per_speaker(self) -> Dict[str, float]:
		"""Calculate speaking time for each speaker"""
		if not self.speaker_segments:
			return {}
		
		speaker_times = {}
		for segment in self.speaker_segments:
			speaker = segment.get('speaker', 'Unknown')
			start_time = segment.get('start_time', 0)
			end_time = segment.get('end_time', 0)
			duration = end_time - start_time
			
			if speaker in speaker_times:
				speaker_times[speaker] += duration
			else:
				speaker_times[speaker] = duration
		
		return speaker_times
	
	def get_transcript_by_speaker(self) -> Dict[str, List[str]]:
		"""Get transcript organized by speaker"""
		if not self.speaker_segments:
			return {'Speaker 1': [self.transcript_text]}
		
		speaker_text = {}
		for segment in self.speaker_segments:
			speaker = segment.get('speaker', 'Unknown')
			text = segment.get('text', '')
			
			if speaker in speaker_text:
				speaker_text[speaker].append(text)
			else:
				speaker_text[speaker] = [text]
		
		return speaker_text
	
	def export_to_format(self, format_type: str) -> str:
		"""Export transcription to specified format"""
		if format_type.lower() == 'srt':
			return self._export_to_srt()
		elif format_type.lower() == 'vtt':
			return self._export_to_vtt()
		elif format_type.lower() == 'txt':
			return self.transcript_text
		elif format_type.lower() == 'json':
			return json.dumps({
				'transcript': self.transcript_text,
				'speakers': self.get_speakers_list(),
				'word_data': self.word_level_data,
				'speaker_segments': self.speaker_segments,
				'confidence': self.confidence_score,
				'language': self.language
			})
		else:
			raise ValueError(f"Unsupported export format: {format_type}")
	
	def _export_to_srt(self) -> str:
		"""Export to SRT subtitle format"""
		if not self.sentence_segments:
			return ""
		
		srt_content = []
		for i, segment in enumerate(self.sentence_segments, 1):
			start_time = self._format_srt_time(segment.get('start_time', 0))
			end_time = self._format_srt_time(segment.get('end_time', 0))
			text = segment.get('text', '')
			
			srt_content.append(f"{i}")
			srt_content.append(f"{start_time} --> {end_time}")
			srt_content.append(text)
			srt_content.append("")  # Empty line between segments
		
		return "\n".join(srt_content)
	
	def _export_to_vtt(self) -> str:
		"""Export to WebVTT format"""
		vtt_content = ["WEBVTT", ""]
		
		if self.sentence_segments:
			for segment in self.sentence_segments:
				start_time = self._format_vtt_time(segment.get('start_time', 0))
				end_time = self._format_vtt_time(segment.get('end_time', 0))
				text = segment.get('text', '')
				
				vtt_content.append(f"{start_time} --> {end_time}")
				vtt_content.append(text)
				vtt_content.append("")
		
		return "\n".join(vtt_content)
	
	def _format_srt_time(self, seconds: float) -> str:
		"""Format time for SRT format (HH:MM:SS,mmm)"""
		hours = int(seconds // 3600)
		minutes = int((seconds % 3600) // 60)
		secs = int(seconds % 60)
		millisecs = int((seconds % 1) * 1000)
		
		return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
	
	def _format_vtt_time(self, seconds: float) -> str:
		"""Format time for VTT format (HH:MM:SS.mmm)"""
		hours = int(seconds // 3600)
		minutes = int((seconds % 3600) // 60)
		secs = int(seconds % 60)
		millisecs = int((seconds % 1) * 1000)
		
		return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


class APVoiceSynthesis(Model, AuditMixin, BaseMixin):
	"""
	Voice synthesis (text-to-speech) results with voice characteristics.
	
	Stores TTS generation results with voice settings, quality metrics,
	and generated audio file information.
	"""
	__tablename__ = 'ap_voice_synthesis'
	
	# Identity
	synthesis_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Input Configuration
	input_text = Column(Text, nullable=False)
	input_language = Column(String(10), nullable=False, index=True)  # ISO language code
	input_ssml = Column(Text, nullable=True)  # SSML markup if used
	text_length = Column(Integer, default=0)  # Character count
	
	# Voice Configuration
	provider = Column(String(50), nullable=False, index=True)  # openai, azure, google, amazon, custom
	voice_id = Column(String(100), nullable=False)  # Provider-specific voice identifier
	voice_name = Column(String(100), nullable=True)  # Human-readable voice name
	voice_gender = Column(String(10), nullable=True)  # male, female, neutral
	voice_age = Column(String(20), nullable=True)  # child, young, adult, elderly
	voice_style = Column(String(50), nullable=True)  # conversational, newscast, cheerful, etc.
	
	# Audio Parameters
	speed = Column(Float, default=1.0)  # Speaking rate multiplier
	pitch = Column(Float, default=0.0)  # Pitch adjustment in semitones
	volume = Column(Float, default=1.0)  # Volume multiplier
	emphasis = Column(String(20), nullable=True)  # none, moderate, strong
	
	# Output Configuration
	output_format = Column(String(20), nullable=False)  # mp3, wav, ogg, aac
	sample_rate = Column(Integer, default=22050)  # Hz
	bit_depth = Column(Integer, default=16)  # bits
	quality = Column(String(20), default='standard')  # low, standard, high, premium
	
	# Generated Audio
	audio_file_path = Column(String(1000), nullable=True)
	audio_file_size = Column(Integer, nullable=True)
	audio_duration = Column(Float, nullable=True)  # seconds
	audio_hash = Column(String(64), nullable=True)  # SHA-256
	
	# Quality Metrics
	synthesis_quality = Column(Float, default=0.0)  # 0-100 quality score
	naturalness_score = Column(Float, nullable=True)  # How natural the voice sounds
	intelligibility_score = Column(Float, nullable=True)  # How clear/understandable
	emotion_accuracy = Column(Float, nullable=True)  # How well emotion is conveyed
	
	# Processing Performance
	processing_time_seconds = Column(Float, nullable=True)
	processing_cost = Column(Float, nullable=True)
	characters_processed = Column(Integer, default=0)
	
	# Status and Metadata
	status = Column(String(20), default='completed', index=True)  # processing, completed, failed
	requested_by = Column(String(36), nullable=True, index=True)  # User ID
	processing_metadata = Column(JSON, default=dict)  # Provider-specific metadata
	error_message = Column(Text, nullable=True)
	
	# Usage and Context
	usage_context = Column(String(100), nullable=True)  # announcement, narration, conversation, etc.
	target_audience = Column(String(100), nullable=True)  # general, children, professional, etc.
	content_type = Column(String(50), nullable=True)  # news, story, instruction, etc.
	
	def __repr__(self):
		return f"<APVoiceSynthesis {self.synthesis_id} ({self.voice_name}, {self.input_language})>"
	
	def get_input_preview(self, max_length: int = 100) -> str:
		"""Get truncated input text for preview"""
		if len(self.input_text) <= max_length:
			return self.input_text
		return self.input_text[:max_length] + "..."
	
	def calculate_cost_per_character(self) -> Optional[float]:
		"""Calculate cost per character if cost data available"""
		if self.processing_cost and self.characters_processed:
			return self.processing_cost / self.characters_processed
		return None
	
	def get_audio_duration_formatted(self) -> str:
		"""Get formatted audio duration"""
		if not self.audio_duration:
			return "00:00"
		
		minutes = int(self.audio_duration // 60)
		seconds = int(self.audio_duration % 60)
		return f"{minutes:02d}:{seconds:02d}"
	
	def get_voice_characteristics(self) -> Dict[str, Any]:
		"""Get voice characteristics summary"""
		return {
			'voice_id': self.voice_id,
			'voice_name': self.voice_name,
			'gender': self.voice_gender,
			'age': self.voice_age,
			'style': self.voice_style,
			'language': self.input_language,
			'provider': self.provider
		}
	
	def get_audio_parameters(self) -> Dict[str, Any]:
		"""Get audio generation parameters"""
		return {
			'speed': self.speed,
			'pitch': self.pitch,
			'volume': self.volume,
			'emphasis': self.emphasis,
			'format': self.output_format,
			'sample_rate': self.sample_rate,
			'bit_depth': self.bit_depth,
			'quality': self.quality
		}


class APAudioAnalysis(Model, AuditMixin, BaseMixin):
	"""
	Audio content analysis results including sentiment, topics, and classification.
	
	Stores comprehensive audio analysis including content classification,
	sentiment analysis, and intelligent insights.
	"""
	__tablename__ = 'ap_audio_analysis'
	
	# Identity
	analysis_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	audio_file_id = Column(String(36), ForeignKey('ap_audio_file.file_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Analysis Configuration
	analysis_types = Column(JSON, nullable=False)  # Types of analysis performed
	provider = Column(String(50), nullable=False, index=True)  # Analysis service provider
	model_used = Column(String(100), nullable=True)
	language = Column(String(10), nullable=True, index=True)
	
	# Content Classification
	content_type = Column(String(50), nullable=True, index=True)  # speech, music, mixed, noise
	content_category = Column(String(100), nullable=True)  # meeting, interview, presentation, etc.
	content_confidence = Column(Float, nullable=True)  # 0-1 confidence in classification
	
	# Speech Analysis
	speech_percentage = Column(Float, nullable=True)  # Percentage of audio that is speech
	music_percentage = Column(Float, nullable=True)  # Percentage that is music
	noise_percentage = Column(Float, nullable=True)  # Percentage that is noise/silence
	
	# Sentiment Analysis
	overall_sentiment = Column(String(20), nullable=True, index=True)  # positive, negative, neutral
	sentiment_confidence = Column(Float, nullable=True)  # 0-1 confidence
	sentiment_scores = Column(JSON, nullable=True)  # Detailed sentiment breakdown
	emotional_tone = Column(JSON, nullable=True)  # Detected emotions
	
	# Topic Analysis
	detected_topics = Column(JSON, nullable=True)  # List of identified topics
	topic_confidence = Column(JSON, nullable=True)  # Confidence scores for topics
	key_phrases = Column(JSON, nullable=True)  # Important phrases/keywords
	named_entities = Column(JSON, nullable=True)  # People, places, organizations
	
	# Speaker Analysis
	speaker_characteristics = Column(JSON, nullable=True)  # Voice characteristics
	speaker_emotions = Column(JSON, nullable=True)  # Emotional state of speakers
	speaking_pace = Column(Float, nullable=True)  # Words per minute
	voice_quality_metrics = Column(JSON, nullable=True)  # Technical voice quality
	
	# Content Insights
	summary_text = Column(Text, nullable=True)  # AI-generated summary
	key_moments = Column(JSON, nullable=True)  # Important timestamps
	action_items = Column(JSON, nullable=True)  # Extracted action items
	questions_asked = Column(JSON, nullable=True)  # Questions identified in audio
	
	# Quality and Technical Metrics
	audio_quality_score = Column(Float, nullable=True)  # Overall audio quality 0-100
	transcription_quality = Column(Float, nullable=True)  # Quality of source transcription
	analysis_confidence = Column(Float, nullable=True)  # Overall analysis confidence
	processing_warnings = Column(JSON, default=list)  # Warnings during analysis
	
	# Processing Performance
	processing_time_seconds = Column(Float, nullable=True)
	processing_cost = Column(Float, nullable=True)
	
	# Status and Metadata
	status = Column(String(20), default='completed', index=True)  # processing, completed, failed
	requested_by = Column(String(36), nullable=True, index=True)  # User ID
	analysis_metadata = Column(JSON, default=dict)  # Provider-specific metadata
	error_message = Column(Text, nullable=True)
	
	# Relationships
	audio_file = relationship("APAudioFile", back_populates="analyses")
	
	def __repr__(self):
		return f"<APAudioAnalysis {self.analysis_id} ({self.content_type}, {self.overall_sentiment})>"
	
	def get_top_topics(self, limit: int = 5) -> List[Dict[str, Any]]:
		"""Get top detected topics with confidence scores"""
		if not self.detected_topics or not self.topic_confidence:
			return []
		
		topic_list = []
		for i, topic in enumerate(self.detected_topics[:limit]):
			confidence = self.topic_confidence[i] if i < len(self.topic_confidence) else 0.0
			topic_list.append({
				'topic': topic,
				'confidence': confidence
			})
		
		return sorted(topic_list, key=lambda x: x['confidence'], reverse=True)
	
	def get_sentiment_summary(self) -> Dict[str, Any]:
		"""Get comprehensive sentiment analysis summary"""
		summary = {
			'overall_sentiment': self.overall_sentiment,
			'confidence': self.sentiment_confidence,
			'emotional_tone': self.emotional_tone or {}
		}
		
		if self.sentiment_scores:
			summary['detailed_scores'] = self.sentiment_scores
		
		return summary
	
	def get_speaker_insights(self) -> Dict[str, Any]:
		"""Get speaker analysis insights"""
		insights = {}
		
		if self.speaker_characteristics:
			insights['characteristics'] = self.speaker_characteristics
		
		if self.speaker_emotions:
			insights['emotions'] = self.speaker_emotions
		
		if self.speaking_pace:
			insights['speaking_pace'] = {
				'words_per_minute': self.speaking_pace,
				'pace_category': self._categorize_speaking_pace(self.speaking_pace)
			}
		
		if self.voice_quality_metrics:
			insights['voice_quality'] = self.voice_quality_metrics
		
		return insights
	
	def _categorize_speaking_pace(self, wpm: float) -> str:
		"""Categorize speaking pace"""
		if wpm < 120:
			return 'slow'
		elif wpm < 160:
			return 'normal'
		elif wpm < 200:
			return 'fast'
		else:
			return 'very_fast'
	
	def get_actionable_insights(self) -> Dict[str, Any]:
		"""Get actionable insights from analysis"""
		insights = {}
		
		if self.action_items:
			insights['action_items'] = self.action_items
		
		if self.questions_asked:
			insights['questions'] = self.questions_asked
		
		if self.key_moments:
			insights['key_moments'] = self.key_moments
		
		if self.key_phrases:
			insights['key_phrases'] = self.key_phrases[:10]  # Top 10 phrases
		
		return insights
	
	def calculate_engagement_score(self) -> float:
		"""Calculate engagement score based on various factors"""
		score = 0.0
		factors = 0
		
		# Factor in sentiment (positive sentiment = higher engagement)
		if self.sentiment_scores and 'positive' in self.sentiment_scores:
			score += self.sentiment_scores['positive'] * 100
			factors += 1
		
		# Factor in speaking pace (normal pace = higher engagement)
		if self.speaking_pace:
			if 120 <= self.speaking_pace <= 160:  # Normal pace
				score += 80
			elif 100 <= self.speaking_pace < 200:  # Acceptable pace
				score += 60
			else:  # Too slow or too fast
				score += 30
			factors += 1
		
		# Factor in content quality
		if self.analysis_confidence:
			score += self.analysis_confidence * 100
			factors += 1
		
		# Factor in topic richness
		if self.detected_topics:
			topic_score = min(100, len(self.detected_topics) * 20)  # More topics = higher engagement
			score += topic_score
			factors += 1
		
		return score / factors if factors > 0 else 0.0


class APProcessingJob(Model, AuditMixin, BaseMixin):
	"""
	Audio processing job tracking with status and performance metrics.
	
	Tracks background audio processing jobs including batch operations,
	complex analysis workflows, and long-running tasks.
	"""
	__tablename__ = 'ap_processing_job'
	
	# Identity
	job_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	audio_file_id = Column(String(36), ForeignKey('ap_audio_file.file_id'), nullable=True, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Job Configuration
	job_type = Column(String(50), nullable=False, index=True)  # transcription, synthesis, analysis, enhancement
	job_name = Column(String(200), nullable=True)
	job_description = Column(Text, nullable=True)
	
	# Job Parameters
	input_parameters = Column(JSON, nullable=False)  # Job configuration
	processing_options = Column(JSON, default=dict)  # Additional processing options
	priority = Column(String(20), default='normal', index=True)  # low, normal, high, urgent
	
	# Job Status
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
	progress_percentage = Column(Float, default=0.0)  # 0-100 completion percentage
	current_step = Column(String(100), nullable=True)  # Current processing step
	total_steps = Column(Integer, nullable=True)
	
	# Scheduling
	scheduled_at = Column(DateTime, nullable=True, index=True)
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True, index=True)
	timeout_at = Column(DateTime, nullable=True)
	
	# Results and Output
	output_data = Column(JSON, nullable=True)  # Job results
	output_files = Column(JSON, default=list)  # Generated files
	processing_log = Column(Text, nullable=True)  # Processing log/messages
	
	# Performance Metrics
	processing_time_seconds = Column(Float, nullable=True)
	cpu_time_seconds = Column(Float, nullable=True)
	memory_used_mb = Column(Float, nullable=True)
	processing_cost = Column(Float, nullable=True)
	
	# Error Handling
	error_message = Column(Text, nullable=True)
	error_code = Column(String(50), nullable=True)
	error_details = Column(JSON, default=dict)
	retry_count = Column(Integer, default=0)
	max_retries = Column(Integer, default=3)
	
	# Job Metadata
	requested_by = Column(String(36), nullable=True, index=True)  # User ID
	worker_id = Column(String(100), nullable=True)  # Processing worker identifier
	worker_metadata = Column(JSON, default=dict)  # Worker-specific information
	
	# Relationships
	audio_file = relationship("APAudioFile", back_populates="processing_jobs")
	
	def __repr__(self):
		return f"<APProcessingJob {self.job_id} ({self.job_type}, {self.status})>"
	
	def is_running(self) -> bool:
		"""Check if job is currently running"""
		return self.status in ['pending', 'running']
	
	def is_completed(self) -> bool:
		"""Check if job completed successfully"""
		return self.status == 'completed'
	
	def is_failed(self) -> bool:
		"""Check if job failed"""
		return self.status == 'failed'
	
	def can_retry(self) -> bool:
		"""Check if job can be retried"""
		return self.is_failed() and self.retry_count < self.max_retries
	
	def get_duration(self) -> Optional[float]:
		"""Get job duration in seconds"""
		if self.started_at and self.completed_at:
			return (self.completed_at - self.started_at).total_seconds()
		elif self.started_at:
			return (datetime.utcnow() - self.started_at).total_seconds()
		return None
	
	def start_job(self, worker_id: str = None) -> None:
		"""Mark job as started"""
		self.status = 'running'
		self.started_at = datetime.utcnow()
		self.worker_id = worker_id
		self.progress_percentage = 0.0
	
	def update_progress(self, percentage: float, current_step: str = None) -> None:
		"""Update job progress"""
		self.progress_percentage = min(100.0, max(0.0, percentage))
		if current_step:
			self.current_step = current_step
	
	def complete_job(self, output_data: Dict[str, Any] = None, 
					 output_files: List[str] = None) -> None:
		"""Mark job as completed with results"""
		self.status = 'completed'
		self.completed_at = datetime.utcnow()
		self.progress_percentage = 100.0
		
		if output_data:
			self.output_data = output_data
		
		if output_files:
			self.output_files = output_files
		
		if self.started_at:
			self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
	
	def fail_job(self, error_message: str, error_code: str = None, 
				 error_details: Dict[str, Any] = None) -> None:
		"""Mark job as failed with error information"""
		self.status = 'failed'
		self.completed_at = datetime.utcnow()
		self.error_message = error_message
		self.error_code = error_code
		self.error_details = error_details or {}
		
		if self.started_at:
			self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
	
	def retry_job(self) -> None:
		"""Prepare job for retry"""
		if not self.can_retry():
			raise ValueError("Job cannot be retried")
		
		self.retry_count += 1
		self.status = 'pending'
		self.started_at = None
		self.completed_at = None
		self.progress_percentage = 0.0
		self.current_step = None
		self.error_message = None
		self.error_code = None
		self.error_details = {}
	
	def cancel_job(self, reason: str = None) -> None:
		"""Cancel job"""
		if self.status in ['completed', 'failed', 'cancelled']:
			return  # Cannot cancel already finished jobs
		
		self.status = 'cancelled'
		self.completed_at = datetime.utcnow()
		
		if reason:
			self.error_message = f"Job cancelled: {reason}"
		
		if self.started_at:
			self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
	
	def add_log_entry(self, message: str, level: str = 'info') -> None:
		"""Add entry to processing log"""
		timestamp = datetime.utcnow().isoformat()
		log_entry = f"[{timestamp}] [{level.upper()}] {message}\n"
		
		if self.processing_log:
			self.processing_log += log_entry
		else:
			self.processing_log = log_entry
	
	def get_estimated_completion_time(self) -> Optional[datetime]:
		"""Estimate completion time based on progress"""
		if not self.started_at or self.progress_percentage <= 0:
			return None
		
		elapsed = (datetime.utcnow() - self.started_at).total_seconds()
		estimated_total = elapsed / (self.progress_percentage / 100)
		remaining = estimated_total - elapsed
		
		return datetime.utcnow() + timedelta(seconds=remaining)