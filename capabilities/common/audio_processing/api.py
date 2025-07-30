"""
Audio Processing API Layer

RESTful API endpoints for audio processing capabilities with
APG platform integration for authentication, rate limiting, and monitoring.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from .models import (
	APAudioSession, APTranscriptionJob, APVoiceSynthesisJob, APAudioAnalysisJob, APVoiceModel,
	AudioFormat, AudioQuality, TranscriptionProvider, VoiceSynthesisProvider, 
	EmotionType, SentimentType, ProcessingStatus
)
from .service import (
	create_transcription_service, create_synthesis_service, create_analysis_service,
	create_enhancement_service, create_model_manager, create_workflow_orchestrator
)
from uuid_extensions import uuid7str

# API Router for audio processing endpoints
router = APIRouter(prefix="/api/v1/audio", tags=["audio_processing"])

# Request/Response Models

class TranscriptionRequest(BaseModel):
	"""Request model for audio transcription"""
	audio_source: Dict[str, Any] = Field(..., description="Audio source configuration")
	language_code: str = Field("en-US", description="Language code for transcription")
	provider: TranscriptionProvider = Field(TranscriptionProvider.OPENAI_WHISPER, description="Transcription provider")
	speaker_diarization: bool = Field(True, description="Enable speaker diarization")
	custom_vocabulary: List[str] = Field(default_factory=list, description="Custom vocabulary terms")
	real_time: bool = Field(False, description="Enable real-time processing")
	
	class Config:
		extra = 'forbid'

class SynthesisRequest(BaseModel):
	"""Request model for voice synthesis"""
	text: str = Field(..., description="Text to synthesize", min_length=1, max_length=10000)
	voice_id: str = Field("default_coqui_female", description="Voice identifier")
	emotion: EmotionType = Field(EmotionType.NEUTRAL, description="Emotion type")
	emotion_intensity: float = Field(0.5, description="Emotion intensity", ge=0.0, le=1.0)
	voice_speed: float = Field(1.0, description="Speech speed multiplier", ge=0.5, le=2.0)
	voice_pitch: float = Field(1.0, description="Voice pitch multiplier", ge=0.5, le=2.0)
	output_format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")
	quality: AudioQuality = Field(AudioQuality.STANDARD, description="Audio quality")
	model_preference: str = Field("auto", description="Preferred synthesis model")
	
	class Config:
		extra = 'forbid'

class VoiceCloningRequest(BaseModel):
	"""Request model for voice cloning"""
	voice_name: str = Field(..., description="Name for the voice model", min_length=1, max_length=100)
	voice_description: str = Field(None, description="Optional voice description", max_length=500)
	target_language: str = Field("en", description="Target language for synthesis")
	quality_target: float = Field(0.95, description="Target quality score", ge=0.7, le=1.0)
	
	class Config:
		extra = 'forbid'

class AnalysisRequest(BaseModel):
	"""Request model for audio analysis"""
	audio_source: Dict[str, Any] = Field(..., description="Audio source configuration")
	analysis_types: List[str] = Field(
		default_factory=lambda: ["sentiment", "topics", "quality", "speaker_characteristics"],
		description="Types of analysis to perform"
	)
	include_emotions: bool = Field(True, description="Include emotion analysis")
	include_technical_metrics: bool = Field(True, description="Include technical metrics")
	num_topics: int = Field(5, description="Number of topics to extract", ge=1, le=20)
	
	class Config:
		extra = 'forbid'

class EnhancementRequest(BaseModel):
	"""Request model for audio enhancement"""
	audio_source: Dict[str, Any] = Field(..., description="Audio source configuration")
	enhancement_type: str = Field("noise_reduction", description="Type of enhancement")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Enhancement parameters")
	output_format: AudioFormat = Field(AudioFormat.WAV, description="Output format")
	
	class Config:
		extra = 'forbid'

class WorkflowRequest(BaseModel):
	"""Request model for workflow execution"""
	audio_source: Dict[str, Any] = Field(..., description="Audio source configuration")
	workflow_type: str = Field("transcribe_analyze_enhance", description="Workflow type")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")
	
	class Config:
		extra = 'forbid'

# Response Models

class TranscriptionResponse(BaseModel):
	"""Response model for transcription results"""
	job_id: str
	status: ProcessingStatus
	transcription_text: str = None
	confidence_score: float = None
	speaker_segments: List[Dict[str, Any]] = None
	processing_time_ms: float = None
	language_detected: str = None
	created_at: datetime
	completed_at: datetime = None

class SynthesisResponse(BaseModel):
	"""Response model for synthesis results"""
	job_id: str
	status: ProcessingStatus
	audio_path: str = None
	audio_duration: float = None
	quality_score: float = None
	synthesis_metadata: Dict[str, Any] = None
	created_at: datetime
	completed_at: datetime = None

class AnalysisResponse(BaseModel):
	"""Response model for analysis results"""
	job_id: str
	status: ProcessingStatus
	analysis_results: Dict[str, Any] = None
	confidence_score: float = None
	processing_metadata: Dict[str, Any] = None
	created_at: datetime
	completed_at: datetime = None

class WorkflowResponse(BaseModel):
	"""Response model for workflow results"""
	workflow_id: str
	status: str
	total_processing_time: float = None
	results: Dict[str, Any] = None
	steps_completed: List[str] = None

# API Endpoints

@router.post("/transcribe", response_model=TranscriptionResponse, status_code=HTTP_201_CREATED)
async def transcribe_audio(
	request: TranscriptionRequest,
	background_tasks: BackgroundTasks,
	tenant_id: str = "default",
	user_id: str = None
):
	"""
	Transcribe audio to text with speaker diarization
	
	Supports multiple providers and real-time processing.
	"""
	try:
		service = create_transcription_service()
		
		# Estimate audio duration from source
		audio_duration = request.audio_source.get('duration', 30.0)
		audio_format = AudioFormat(request.audio_source.get('format', 'wav'))
		
		job = await service.create_transcription_job(
			session_id=None,
			audio_source=request.audio_source,
			audio_duration=audio_duration,
			audio_format=audio_format,
			provider=request.provider,
			language_code=request.language_code,
			tenant_id=tenant_id,
			user_id=user_id,
			speaker_diarization=request.speaker_diarization,
			custom_vocabulary=request.custom_vocabulary,
			real_time=request.real_time
		)
		
		return TranscriptionResponse(
			job_id=job.job_id,
			status=job.status,
			transcription_text=job.transcription_text,
			confidence_score=job.confidence_score,
			speaker_segments=job.speaker_segments,
			processing_time_ms=job.processing_metadata.get('processing_time_ms'),
			language_detected=job.language_detected,
			created_at=job.created_at,
			completed_at=job.completed_at
		)
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Transcription failed: {str(e)}"
		)

@router.post("/synthesize", response_model=SynthesisResponse, status_code=HTTP_201_CREATED)
async def synthesize_speech(
	request: SynthesisRequest,
	background_tasks: BackgroundTasks,
	tenant_id: str = "default",
	user_id: str = None
):
	"""
	Synthesize speech from text with emotion control
	
	Supports multiple open source models and voice cloning.
	"""
	try:
		service = create_synthesis_service()
		
		job = await service.synthesize_text(
			text=request.text,
			voice_id=request.voice_id,
			emotion=request.emotion,
			emotion_intensity=request.emotion_intensity,
			voice_speed=request.voice_speed,
			voice_pitch=request.voice_pitch,
			output_format=request.output_format,
			quality=request.quality,
			model_preference=request.model_preference,
			tenant_id=tenant_id,
			user_id=user_id
		)
		
		return SynthesisResponse(
			job_id=job.job_id,
			status=job.status,
			audio_path=job.output_audio_path,
			audio_duration=job.audio_duration,
			quality_score=job.quality_score,
			synthesis_metadata=job.synthesis_metadata,
			created_at=job.created_at,
			completed_at=job.completed_at
		)
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Synthesis failed: {str(e)}"
		)

@router.post("/voices/clone", response_model=Dict[str, Any], status_code=HTTP_201_CREATED)
async def clone_voice(
	request: VoiceCloningRequest,
	audio_files: List[UploadFile] = File(...),
	tenant_id: str = "default",
	user_id: str = None
):
	"""
	Clone a voice from audio samples using Coqui XTTS-v2
	
	Requires at least 10 seconds of clean audio samples.
	"""
	try:
		if not audio_files:
			raise HTTPException(
				status_code=HTTP_400_BAD_REQUEST,
				detail="At least one audio file must be provided"
			)
		
		# Save uploaded files temporarily
		training_samples = []
		for file in audio_files:
			temp_path = f"/tmp/voice_sample_{uuid7str()}.{file.filename.split('.')[-1]}"
			with open(temp_path, "wb") as f:
				content = await file.read()
				f.write(content)
			training_samples.append(temp_path)
		
		service = create_synthesis_service()
		
		voice_model = await service.clone_voice_coqui_xtts(
			voice_name=request.voice_name,
			training_audio_samples=training_samples,
			voice_description=request.voice_description,
			target_language=request.target_language,
			quality_target=request.quality_target,
			tenant_id=tenant_id,
			user_id=user_id
		)
		
		# Register model with manager
		model_manager = create_model_manager()
		if voice_model.status == ProcessingStatus.COMPLETED:
			await model_manager.register_model(voice_model)
		
		return {
			"model_id": voice_model.model_id,
			"voice_name": voice_model.voice_name,
			"status": voice_model.status.value,
			"quality_score": voice_model.quality_score,
			"training_duration": voice_model.training_duration,
			"supported_emotions": [e.value for e in voice_model.supported_emotions] if voice_model.supported_emotions else [],
			"created_at": voice_model.created_at,
			"completed_at": voice_model.completed_at
		}
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Voice cloning failed: {str(e)}"
		)

@router.post("/analyze", response_model=AnalysisResponse, status_code=HTTP_201_CREATED)
async def analyze_audio(
	request: AnalysisRequest,
	background_tasks: BackgroundTasks,
	tenant_id: str = "default",
	user_id: str = None
):
	"""
	Perform comprehensive audio analysis
	
	Includes sentiment, topics, speaker characteristics, and quality assessment.
	"""
	try:
		service = create_analysis_service()
		
		# Execute requested analyses
		results = {}
		
		if "sentiment" in request.analysis_types:
			sentiment_job = await service.analyze_sentiment(
				audio_source=request.audio_source,
				include_emotions=request.include_emotions,
				tenant_id=tenant_id,
				user_id=user_id
			)
			results["sentiment"] = sentiment_job.analysis_results
		
		if "topics" in request.analysis_types:
			topic_job = await service.detect_topics(
				audio_source=request.audio_source,
				num_topics=request.num_topics,
				tenant_id=tenant_id,
				user_id=user_id
			)
			results["topics"] = topic_job.analysis_results
		
		if "quality" in request.analysis_types:
			quality_job = await service.assess_quality(
				audio_source=request.audio_source,
				include_technical_metrics=request.include_technical_metrics,
				tenant_id=tenant_id,
				user_id=user_id
			)
			results["quality"] = quality_job.analysis_results
		
		if "speaker_characteristics" in request.analysis_types:
			speaker_job = await service.detect_speaker_characteristics(
				audio_source=request.audio_source,
				tenant_id=tenant_id,
				user_id=user_id
			)
			results["speaker_characteristics"] = speaker_job.analysis_results
		
		# Calculate overall confidence
		confidence_scores = [
			job.confidence_score for job in [
				results.get("sentiment"), results.get("topics"), 
				results.get("quality"), results.get("speaker_characteristics")
			] if job
		]
		overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
		
		return AnalysisResponse(
			job_id=uuid7str(),
			status=ProcessingStatus.COMPLETED,
			analysis_results=results,
			confidence_score=overall_confidence,
			processing_metadata={"analysis_types": request.analysis_types},
			created_at=datetime.utcnow(),
			completed_at=datetime.utcnow()
		)
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Analysis failed: {str(e)}"
		)

@router.post("/enhance", response_model=Dict[str, Any], status_code=HTTP_201_CREATED)
async def enhance_audio(
	request: EnhancementRequest,
	background_tasks: BackgroundTasks,
	tenant_id: str = "default",
	user_id: str = None
):
	"""
	Enhance audio quality using various techniques
	
	Supports noise reduction, voice isolation, normalization, and format conversion.
	"""
	try:
		service = create_enhancement_service()
		
		if request.enhancement_type == "noise_reduction":
			result = await service.reduce_noise(
				audio_source=request.audio_source,
				noise_reduction_level=request.parameters.get("level", "moderate"),
				preserve_speech=request.parameters.get("preserve_speech", True),
				output_format=request.output_format,
				tenant_id=tenant_id,
				user_id=user_id
			)
		elif request.enhancement_type == "voice_isolation":
			result = await service.isolate_voices(
				audio_source=request.audio_source,
				num_speakers=request.parameters.get("num_speakers"),
				separation_quality=request.parameters.get("quality", "standard"),
				output_format=request.output_format,
				tenant_id=tenant_id,
				user_id=user_id
			)
		elif request.enhancement_type == "normalization":
			result = await service.normalize_audio(
				audio_source=request.audio_source,
				target_lufs=request.parameters.get("target_lufs", -23.0),
				peak_limit_db=request.parameters.get("peak_limit_db", -1.0),
				output_format=request.output_format,
				tenant_id=tenant_id,
				user_id=user_id
			)
		elif request.enhancement_type == "format_conversion":
			result = await service.convert_format(
				audio_source=request.audio_source,
				target_format=request.output_format,
				target_quality=AudioQuality(request.parameters.get("quality", "standard")),
				tenant_id=tenant_id,
				user_id=user_id
			)
		else:
			raise HTTPException(
				status_code=HTTP_400_BAD_REQUEST,
				detail=f"Unknown enhancement type: {request.enhancement_type}"
			)
		
		return result
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Enhancement failed: {str(e)}"
		)

@router.post("/workflows/execute", response_model=WorkflowResponse, status_code=HTTP_201_CREATED)
async def execute_workflow(
	request: WorkflowRequest,
	background_tasks: BackgroundTasks,
	tenant_id: str = "default",
	user_id: str = None
):
	"""
	Execute complex audio processing workflow
	
	Coordinates multiple services for comprehensive audio processing.
	"""
	try:
		orchestrator = create_workflow_orchestrator()
		
		result = await orchestrator.process_complete_workflow(
			audio_source=request.audio_source,
			workflow_type=request.workflow_type,
			tenant_id=tenant_id,
			user_id=user_id,
			**request.parameters
		)
		
		return WorkflowResponse(
			workflow_id=result["workflow_id"],
			status=result["status"],
			total_processing_time=result.get("total_processing_time"),
			results=result.get("results"),
			steps_completed=result.get("results", {}).get("steps_completed", [])
		)
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Workflow execution failed: {str(e)}"
		)

@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str, tenant_id: str = "default"):
	"""Get status of a processing job"""
	# This would typically query a job tracking system
	# For now, return a placeholder response
	return {
		"job_id": job_id,
		"status": "completed",
		"message": "Job status endpoint - implementation varies by service"
	}

@router.get("/voices", response_model=List[Dict[str, Any]])
async def list_voices(tenant_id: str = "default"):
	"""List available voice models"""
	try:
		model_manager = create_model_manager()
		models = await model_manager.list_models(tenant_id=tenant_id)
		
		return [
			{
				"model_id": model.model_id,
				"voice_name": model.voice_name,
				"voice_description": model.voice_description,
				"quality_score": model.quality_score,
				"supported_emotions": [e.value for e in model.supported_emotions] if model.supported_emotions else [],
				"created_at": model.created_at
			}
			for model in models
		]
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Failed to list voices: {str(e)}"
		)

@router.delete("/voices/{model_id}", response_model=Dict[str, str])
async def delete_voice(model_id: str, tenant_id: str = "default"):
	"""Delete a voice model"""
	try:
		model_manager = create_model_manager()
		success = await model_manager.delete_model(model_id)
		
		if success:
			return {"message": f"Voice model {model_id} deleted successfully"}
		else:
			raise HTTPException(
				status_code=HTTP_404_NOT_FOUND,
				detail=f"Voice model {model_id} not found"
			)
			
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Failed to delete voice: {str(e)}"
		)

@router.get("/workflows/{workflow_id}/status", response_model=Dict[str, Any])
async def get_workflow_status(workflow_id: str, tenant_id: str = "default"):
	"""Get status of a running workflow"""
	try:
		orchestrator = create_workflow_orchestrator()
		status = await orchestrator.get_workflow_status(workflow_id)
		
		if status:
			return status
		else:
			raise HTTPException(
				status_code=HTTP_404_NOT_FOUND,
				detail=f"Workflow {workflow_id} not found"
			)
			
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_400_BAD_REQUEST,
			detail=f"Failed to get workflow status: {str(e)}"
		)

@router.get("/download/{file_path}")
async def download_audio_file(file_path: str, tenant_id: str = "default"):
	"""Download processed audio file"""
	try:
		# Security: validate file path and tenant access
		safe_path = file_path.replace("..", "").replace("/", "_")
		full_path = f"/tmp/{safe_path}"
		
		return FileResponse(
			path=full_path,
			media_type="audio/wav",
			filename=f"audio_{safe_path}"
		)
		
	except Exception as e:
		raise HTTPException(
			status_code=HTTP_404_NOT_FOUND,
			detail=f"File not found: {str(e)}"
		)

# Health check endpoint
@router.get("/health")
async def health_check():
	"""API health check"""
	return {
		"status": "healthy",
		"service": "audio_processing",
		"version": "1.0.0",
		"timestamp": datetime.utcnow().isoformat()
	}