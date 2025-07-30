"""
WebRTC Recording System for APG Real-Time Collaboration

Implements client-side and server-side recording capabilities for video calls,
screen sharing sessions, and audio conversations with AI-powered features.
"""

import asyncio
import json
import logging
import base64
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class RecordingType(Enum):
	"""Recording types"""
	AUDIO_ONLY = "audio_only"
	VIDEO_CALL = "video_call"
	SCREEN_SHARE = "screen_share"
	FULL_MEETING = "full_meeting"
	PRESENTATION = "presentation"


class RecordingStatus(Enum):
	"""Recording status"""
	PREPARING = "preparing"
	RECORDING = "recording"
	PAUSED = "paused"
	STOPPING = "stopping"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


@dataclass
class RecordingSession:
	"""Recording session data"""
	recording_id: str
	call_id: str
	session_id: str
	recording_type: RecordingType
	started_by: str
	participants: List[str] = field(default_factory=list)
	status: RecordingStatus = RecordingStatus.PREPARING
	started_at: Optional[datetime] = None
	ended_at: Optional[datetime] = None
	duration_seconds: int = 0
	file_size_bytes: int = 0
	file_path: Optional[str] = None
	thumbnail_path: Optional[str] = None
	transcript_path: Optional[str] = None
	ai_summary: Optional[str] = None
	ai_highlights: List[str] = field(default_factory=list)
	ai_action_items: List[str] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)
	error_message: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		return {
			"recording_id": self.recording_id,
			"call_id": self.call_id,
			"session_id": self.session_id,
			"recording_type": self.recording_type.value,
			"started_by": self.started_by,
			"participants": self.participants,
			"status": self.status.value,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"ended_at": self.ended_at.isoformat() if self.ended_at else None,
			"duration_seconds": self.duration_seconds,
			"file_size_bytes": self.file_size_bytes,
			"file_path": self.file_path,
			"thumbnail_path": self.thumbnail_path,
			"transcript_path": self.transcript_path,
			"ai_summary": self.ai_summary,
			"ai_highlights": self.ai_highlights,
			"ai_action_items": self.ai_action_items,
			"metadata": self.metadata,
			"error_message": self.error_message
		}


class WebRTCRecordingManager:
	"""Manages WebRTC recording sessions with AI-powered features"""
	
	def __init__(self, storage_path: str = "/tmp/rtc_recordings"):
		self.storage_path = Path(storage_path)
		self.storage_path.mkdir(parents=True, exist_ok=True)
		
		self.active_recordings: Dict[str, RecordingSession] = {}
		self.completed_recordings: Dict[str, RecordingSession] = {}
		
		# AI integration (would connect to actual APG AI service)
		self.ai_service = None
		
		# Configuration
		self.max_recording_duration = 14400  # 4 hours
		self.max_file_size = 5 * 1024 * 1024 * 1024  # 5GB
		self.supported_formats = ["webm", "mp4", "wav", "mp3"]
	
	async def start_recording(self, call_id: str, session_id: str, started_by: str,
							recording_type: RecordingType, participants: List[str],
							options: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Start a new recording session"""
		try:
			recording_id = f"rec_{uuid.uuid4().hex[:12]}"
			
			# Create recording session
			recording = RecordingSession(
				recording_id=recording_id,
				call_id=call_id,
				session_id=session_id,
				recording_type=recording_type,
				started_by=started_by,
				participants=participants.copy(),
				status=RecordingStatus.PREPARING,
				metadata=options or {}
			)
			
			self.active_recordings[recording_id] = recording
			
			# Prepare recording directory
			recording_dir = self.storage_path / recording_id
			recording_dir.mkdir(exist_ok=True)
			
			# Start recording process
			await self._start_recording_process(recording)
			
			logger.info(f"Recording started: {recording_id} for call {call_id}")
			
			return {
				"recording_id": recording_id,
				"status": "started",
				"recording_type": recording_type.value,
				"started_at": recording.started_at.isoformat() if recording.started_at else None
			}
			
		except Exception as e:
			logger.error(f"Error starting recording: {e}")
			return {"error": f"Failed to start recording: {str(e)}"}
	
	async def stop_recording(self, recording_id: str, stopped_by: str) -> Dict[str, Any]:
		"""Stop an active recording"""
		try:
			if recording_id not in self.active_recordings:
				return {"error": "Recording not found or already stopped"}
			
			recording = self.active_recordings[recording_id]
			recording.status = RecordingStatus.STOPPING
			recording.ended_at = datetime.utcnow()
			
			if recording.started_at:
				recording.duration_seconds = int((recording.ended_at - recording.started_at).total_seconds())
			
			# Stop recording process
			await self._stop_recording_process(recording)
			
			# Move to completed recordings
			self.completed_recordings[recording_id] = recording
			del self.active_recordings[recording_id]
			
			logger.info(f"Recording stopped: {recording_id} by {stopped_by}")
			
			# Start post-processing
			asyncio.create_task(self._post_process_recording(recording))
			
			return {
				"recording_id": recording_id,
				"status": "stopped",
				"duration_seconds": recording.duration_seconds,
				"ended_at": recording.ended_at.isoformat()
			}
			
		except Exception as e:
			logger.error(f"Error stopping recording: {e}")
			return {"error": f"Failed to stop recording: {str(e)}"}
	
	async def pause_recording(self, recording_id: str, paused_by: str) -> Dict[str, Any]:
		"""Pause an active recording"""
		try:
			if recording_id not in self.active_recordings:
				return {"error": "Recording not found"}
			
			recording = self.active_recordings[recording_id]
			
			if recording.status != RecordingStatus.RECORDING:
				return {"error": "Recording is not currently recording"}
			
			recording.status = RecordingStatus.PAUSED
			
			# Pause recording process
			await self._pause_recording_process(recording)
			
			logger.info(f"Recording paused: {recording_id} by {paused_by}")
			
			return {
				"recording_id": recording_id,
				"status": "paused",
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Error pausing recording: {e}")
			return {"error": f"Failed to pause recording: {str(e)}"}
	
	async def resume_recording(self, recording_id: str, resumed_by: str) -> Dict[str, Any]:
		"""Resume a paused recording"""
		try:
			if recording_id not in self.active_recordings:
				return {"error": "Recording not found"}
			
			recording = self.active_recordings[recording_id]
			
			if recording.status != RecordingStatus.PAUSED:
				return {"error": "Recording is not currently paused"}
			
			recording.status = RecordingStatus.RECORDING
			
			# Resume recording process
			await self._resume_recording_process(recording)
			
			logger.info(f"Recording resumed: {recording_id} by {resumed_by}")
			
			return {
				"recording_id": recording_id,
				"status": "recording",
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Error resuming recording: {e}")
			return {"error": f"Failed to resume recording: {str(e)}"}
	
	async def cancel_recording(self, recording_id: str, cancelled_by: str) -> Dict[str, Any]:
		"""Cancel an active recording"""
		try:
			if recording_id not in self.active_recordings:
				return {"error": "Recording not found"}
			
			recording = self.active_recordings[recording_id]
			recording.status = RecordingStatus.CANCELLED
			recording.ended_at = datetime.utcnow()
			
			# Cancel recording process
			await self._cancel_recording_process(recording)
			
			# Clean up files
			recording_dir = self.storage_path / recording_id
			if recording_dir.exists():
				# Remove recording files
				for file in recording_dir.iterdir():
					file.unlink()
				recording_dir.rmdir()
			
			# Remove from active recordings
			del self.active_recordings[recording_id]
			
			logger.info(f"Recording cancelled: {recording_id} by {cancelled_by}")
			
			return {
				"recording_id": recording_id,
				"status": "cancelled",
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Error cancelling recording: {e}")
			return {"error": f"Failed to cancel recording: {str(e)}"}
	
	async def _start_recording_process(self, recording: RecordingSession):
		"""Start the actual recording process"""
		try:
			recording.status = RecordingStatus.RECORDING
			recording.started_at = datetime.utcnow()
			
			# Create file paths
			recording_dir = self.storage_path / recording.recording_id
			
			if recording.recording_type == RecordingType.AUDIO_ONLY:
				recording.file_path = str(recording_dir / f"{recording.recording_id}.wav")
			else:
				recording.file_path = str(recording_dir / f"{recording.recording_id}.webm")
			
			# In a real implementation, this would start the actual recording
			# For now, we simulate the recording process
			logger.info(f"Recording process started for {recording.recording_id}")
			
		except Exception as e:
			recording.status = RecordingStatus.FAILED
			recording.error_message = str(e)
			logger.error(f"Failed to start recording process: {e}")
	
	async def _stop_recording_process(self, recording: RecordingSession):
		"""Stop the recording process"""
		try:
			recording.status = RecordingStatus.PROCESSING
			
			# Simulate file creation and size calculation
			if recording.file_path:
				# In a real implementation, this would finalize the recording file
				recording.file_size_bytes = 1024 * 1024  # Simulate 1MB file
			
			logger.info(f"Recording process stopped for {recording.recording_id}")
			
		except Exception as e:
			recording.status = RecordingStatus.FAILED
			recording.error_message = str(e)
			logger.error(f"Failed to stop recording process: {e}")
	
	async def _pause_recording_process(self, recording: RecordingSession):
		"""Pause the recording process"""
		logger.info(f"Recording process paused for {recording.recording_id}")
	
	async def _resume_recording_process(self, recording: RecordingSession):
		"""Resume the recording process"""
		logger.info(f"Recording process resumed for {recording.recording_id}")
	
	async def _cancel_recording_process(self, recording: RecordingSession):
		"""Cancel the recording process"""
		logger.info(f"Recording process cancelled for {recording.recording_id}")
	
	async def _post_process_recording(self, recording: RecordingSession):
		"""Post-process the recording with AI features"""
		try:
			logger.info(f"Starting post-processing for recording {recording.recording_id}")
			
			# Simulate processing time
			await asyncio.sleep(2)
			
			# Generate thumbnail
			if recording.recording_type in [RecordingType.VIDEO_CALL, RecordingType.SCREEN_SHARE, RecordingType.FULL_MEETING]:
				await self._generate_thumbnail(recording)
			
			# Generate transcript
			await self._generate_transcript(recording)
			
			# Generate AI summary and insights
			await self._generate_ai_insights(recording)
			
			recording.status = RecordingStatus.COMPLETED
			
			logger.info(f"Post-processing completed for recording {recording.recording_id}")
			
		except Exception as e:
			recording.status = RecordingStatus.FAILED
			recording.error_message = f"Post-processing failed: {str(e)}"
			logger.error(f"Post-processing failed for recording {recording.recording_id}: {e}")
	
	async def _generate_thumbnail(self, recording: RecordingSession):
		"""Generate video thumbnail"""
		try:
			recording_dir = self.storage_path / recording.recording_id
			thumbnail_path = recording_dir / f"{recording.recording_id}_thumbnail.jpg"
			
			# In a real implementation, this would extract a frame from the video
			# For now, we just create the path
			recording.thumbnail_path = str(thumbnail_path)
			
			logger.info(f"Thumbnail generated for recording {recording.recording_id}")
			
		except Exception as e:
			logger.error(f"Failed to generate thumbnail: {e}")
	
	async def _generate_transcript(self, recording: RecordingSession):
		"""Generate audio transcript using AI"""
		try:
			recording_dir = self.storage_path / recording.recording_id
			transcript_path = recording_dir / f"{recording.recording_id}_transcript.txt"
			
			# In a real implementation, this would use speech-to-text AI
			# For now, we simulate the transcript
			sample_transcript = f"""
Transcript for Recording {recording.recording_id}
Generated at: {datetime.utcnow().isoformat()}
Participants: {', '.join(recording.participants)}

[00:00:10] Participant A: Welcome everyone to today's meeting.
[00:00:15] Participant B: Thank you for organizing this session.
[00:00:20] Participant A: Today we'll be discussing the project updates.
[00:01:00] Participant B: The development phase is proceeding as planned.
[00:01:30] Participant A: Great to hear. Any blockers we should address?
[00:02:00] Participant B: We need to review the API specifications.
[00:02:30] Participant A: I'll schedule a follow-up meeting for that.

End of transcript.
"""
			
			# Write transcript file
			with open(transcript_path, 'w') as f:
				f.write(sample_transcript)
			
			recording.transcript_path = str(transcript_path)
			
			logger.info(f"Transcript generated for recording {recording.recording_id}")
			
		except Exception as e:
			logger.error(f"Failed to generate transcript: {e}")
	
	async def _generate_ai_insights(self, recording: RecordingSession):
		"""Generate AI-powered insights, summary, and action items"""
		try:
			# In a real implementation, this would use APG AI service
			# For now, we simulate AI insights
			
			recording.ai_summary = f"""
Meeting Summary for {recording.recording_id}:

This was a productive collaboration session with {len(recording.participants)} participants. 
The discussion focused on project updates and planning next steps. Key topics included 
development progress review and API specifications discussion.

Duration: {recording.duration_seconds} seconds
Recording Type: {recording.recording_type.value}
"""
			
			recording.ai_highlights = [
				"Development phase is proceeding as planned",
				"API specifications need review",
				"Follow-up meeting scheduled",
				"All participants actively engaged"
			]
			
			recording.ai_action_items = [
				"Review API specifications (Assigned: Participant B)",
				"Schedule follow-up meeting (Assigned: Participant A)",
				"Prepare development status report",
				"Update project timeline"
			]
			
			logger.info(f"AI insights generated for recording {recording.recording_id}")
			
		except Exception as e:
			logger.error(f"Failed to generate AI insights: {e}")
	
	def get_recording_status(self, recording_id: str) -> Optional[Dict[str, Any]]:
		"""Get status of a recording"""
		recording = self.active_recordings.get(recording_id) or self.completed_recordings.get(recording_id)
		
		if not recording:
			return None
		
		return recording.to_dict()
	
	def get_active_recordings(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Get list of active recordings"""
		recordings = []
		
		for recording in self.active_recordings.values():
			if user_id and user_id not in recording.participants and recording.started_by != user_id:
				continue
			
			recordings.append(recording.to_dict())
		
		return recordings
	
	def get_completed_recordings(self, user_id: Optional[str] = None, 
								limit: int = 50) -> List[Dict[str, Any]]:
		"""Get list of completed recordings"""
		recordings = []
		
		for recording in self.completed_recordings.values():
			if user_id and user_id not in recording.participants and recording.started_by != user_id:
				continue
			
			recordings.append(recording.to_dict())
		
		# Sort by completion date (most recent first)
		recordings.sort(key=lambda r: r.get('ended_at', ''), reverse=True)
		
		return recordings[:limit]
	
	def get_recording_statistics(self) -> Dict[str, Any]:
		"""Get recording system statistics"""
		total_completed = len(self.completed_recordings)
		total_active = len(self.active_recordings)
		
		total_duration = sum(
			r.duration_seconds for r in self.completed_recordings.values()
		)
		
		total_size = sum(
			r.file_size_bytes for r in self.completed_recordings.values()
		)
		
		return {
			"active_recordings": total_active,
			"completed_recordings": total_completed,
			"total_duration_seconds": total_duration,
			"total_size_bytes": total_size,
			"average_duration_seconds": total_duration / total_completed if total_completed > 0 else 0,
			"storage_path": str(self.storage_path),
			"timestamp": datetime.utcnow().isoformat()
		}
	
	async def cleanup_old_recordings(self, days_old: int = 30):
		"""Clean up recordings older than specified days"""
		cutoff_date = datetime.utcnow() - timedelta(days=days_old)
		
		recordings_to_remove = []
		
		for recording_id, recording in self.completed_recordings.items():
			if recording.ended_at and recording.ended_at < cutoff_date:
				recordings_to_remove.append(recording_id)
		
		for recording_id in recordings_to_remove:
			try:
				recording = self.completed_recordings[recording_id]
				
				# Remove files
				recording_dir = self.storage_path / recording_id
				if recording_dir.exists():
					for file in recording_dir.iterdir():
						file.unlink()
					recording_dir.rmdir()
				
				# Remove from memory
				del self.completed_recordings[recording_id]
				
				logger.info(f"Cleaned up old recording: {recording_id}")
				
			except Exception as e:
				logger.error(f"Failed to cleanup recording {recording_id}: {e}")
		
		logger.info(f"Cleanup completed: removed {len(recordings_to_remove)} recordings")


# Global recording manager instance
webrtc_recording_manager = WebRTCRecordingManager()


def generate_recording_client_javascript() -> str:
	"""Generate client-side JavaScript for WebRTC recording"""
	return """
/**
 * WebRTC Recording Client for APG Real-Time Collaboration
 * Handles client-side recording with MediaRecorder API
 */

class WebRTCRecordingClient {
	constructor(webrtcClient) {
		this.webrtcClient = webrtcClient;
		this.mediaRecorder = null;
		this.recordedChunks = [];
		this.currentRecording = null;
		this.isRecording = false;
		
		// Configuration
		this.recordingOptions = {
			mimeType: 'video/webm;codecs=vp9',
			audioBitsPerSecond: 128000,
			videoBitsPerSecond: 2500000
		};
		
		// Initialize
		this.initialize();
	}
	
	initialize() {
		// Check browser support
		if (!this.checkRecordingSupport()) {
			console.warn('Recording not supported in this browser');
			return;
		}
		
		console.log('WebRTC Recording Client initialized');
	}
	
	checkRecordingSupport() {
		return !!(window.MediaRecorder && window.MediaRecorder.isTypeSupported);
	}
	
	async startRecording(recordingType = 'full_meeting', options = {}) {
		try {
			if (this.isRecording) {
				throw new Error('Recording already in progress');
			}
			
			// Get media stream based on recording type
			let stream;
			
			switch (recordingType) {
				case 'audio_only':
					stream = await this.getAudioStream();
					break;
				case 'screen_share':
					stream = await this.getScreenShareStream();
					break;
				case 'video_call':
				case 'full_meeting':
				default:
					stream = await this.getVideoCallStream();
					break;
			}
			
			// Set up MediaRecorder
			this.mediaRecorder = new MediaRecorder(stream, this.recordingOptions);
			this.recordedChunks = [];
			
			// Set up event handlers
			this.mediaRecorder.ondataavailable = (event) => {
				if (event.data.size > 0) {
					this.recordedChunks.push(event.data);
				}
			};
			
			this.mediaRecorder.onstop = () => {
				this.handleRecordingStop();
			};
			
			this.mediaRecorder.onerror = (event) => {
				console.error('MediaRecorder error:', event.error);
				this.handleRecordingError(event.error);
			};
			
			// Start recording
			this.mediaRecorder.start(1000); // Collect data every second
			this.isRecording = true;
			
			// Create recording session info
			this.currentRecording = {
				recordingId: this.generateRecordingId(),
				recordingType: recordingType,
				startTime: new Date(),
				stream: stream,
				options: options
			};
			
			// Notify server
			this.notifyServerRecordingStart();
			
			console.log('Recording started:', this.currentRecording.recordingId);
			
			return {
				recordingId: this.currentRecording.recordingId,
				status: 'started',
				recordingType: recordingType
			};
			
		} catch (error) {
			console.error('Error starting recording:', error);
			throw error;
		}
	}
	
	async stopRecording() {
		try {
			if (!this.isRecording || !this.mediaRecorder) {
				throw new Error('No active recording to stop');
			}
			
			// Stop MediaRecorder
			this.mediaRecorder.stop();
			
			// Stop all tracks
			if (this.currentRecording.stream) {
				this.currentRecording.stream.getTracks().forEach(track => track.stop());
			}
			
			this.isRecording = false;
			
			// Notify server
			this.notifyServerRecordingStop();
			
			console.log('Recording stopped:', this.currentRecording.recordingId);
			
		} catch (error) {
			console.error('Error stopping recording:', error);
			throw error;
		}
	}
	
	async pauseRecording() {
		try {
			if (!this.isRecording || !this.mediaRecorder) {
				throw new Error('No active recording to pause');
			}
			
			if (this.mediaRecorder.state === 'recording') {
				this.mediaRecorder.pause();
				console.log('Recording paused');
			}
			
		} catch (error) {
			console.error('Error pausing recording:', error);
			throw error;
		}
	}
	
	async resumeRecording() {
		try {
			if (!this.mediaRecorder) {
				throw new Error('No recording to resume');
			}
			
			if (this.mediaRecorder.state === 'paused') {
				this.mediaRecorder.resume();
				console.log('Recording resumed');
			}
			
		} catch (error) {
			console.error('Error resuming recording:', error);
			throw error;
		}
	}
	
	async getAudioStream() {
		return await navigator.mediaDevices.getUserMedia({
			audio: {
				echoCancellation: true,
				noiseSuppression: true,
				autoGainControl: true
			},
			video: false
		});
	}
	
	async getVideoCallStream() {
		// For full video call recording, we might need to composite multiple streams
		// For now, get the local stream
		return await navigator.mediaDevices.getUserMedia({
			audio: {
				echoCancellation: true,
				noiseSuppression: true,
				autoGainControl: true
			},
			video: {
				width: { ideal: 1280 },
				height: { ideal: 720 },
				frameRate: { ideal: 30 }
			}
		});
	}
	
	async getScreenShareStream() {
		return await navigator.mediaDevices.getDisplayMedia({
			video: {
				cursor: 'always',
				width: { ideal: 1920 },
				height: { ideal: 1080 }
			},
			audio: true
		});
	}
	
	handleRecordingStop() {
		if (this.recordedChunks.length === 0) {
			console.warn('No recorded data available');
			return;
		}
		
		// Create blob from recorded chunks
		const recordedBlob = new Blob(this.recordedChunks, {
			type: this.recordingOptions.mimeType
		});
		
		// Update recording info
		if (this.currentRecording) {
			this.currentRecording.endTime = new Date();
			this.currentRecording.duration = this.currentRecording.endTime - this.currentRecording.startTime;
			this.currentRecording.blob = recordedBlob;
			this.currentRecording.size = recordedBlob.size;
		}
		
		// Process the recording
		this.processRecording(recordedBlob);
		
		console.log('Recording processing completed');
	}
	
	handleRecordingError(error) {
		console.error('Recording error:', error);
		
		this.isRecording = false;
		
		// Clean up
		if (this.currentRecording && this.currentRecording.stream) {
			this.currentRecording.stream.getTracks().forEach(track => track.stop());
		}
		
		// Notify UI of error
		if (typeof this.onRecordingError === 'function') {
			this.onRecordingError(error);
		}
	}
	
	processRecording(blob) {
		// Create download link
		const url = URL.createObjectURL(blob);
		
		// Auto-download or show save dialog
		const a = document.createElement('a');
		a.style.display = 'none';
		a.href = url;
		a.download = this.generateFileName();
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		
		// Clean up
		setTimeout(() => URL.revokeObjectURL(url), 1000);
		
		// Notify completion
		if (typeof this.onRecordingComplete === 'function') {
			this.onRecordingComplete(this.currentRecording);
		}
	}
	
	generateRecordingId() {
		return 'rec_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
	}
	
	generateFileName() {
		const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
		const type = this.currentRecording?.recordingType || 'recording';
		return `rtc_${type}_${timestamp}.webm`;
	}
	
	notifyServerRecordingStart() {
		if (this.webrtcClient && this.webrtcClient.websocketManager) {
			this.webrtcClient.websocketManager.send({
				type: 'webrtc_recording_start',
				recording_id: this.currentRecording.recordingId,
				recording_type: this.currentRecording.recordingType,
				call_id: this.webrtcClient.currentCall?.callId,
				timestamp: new Date().toISOString()
			});
		}
	}
	
	notifyServerRecordingStop() {
		if (this.webrtcClient && this.webrtcClient.websocketManager && this.currentRecording) {
			this.webrtcClient.websocketManager.send({
				type: 'webrtc_recording_stop',
				recording_id: this.currentRecording.recordingId,
				call_id: this.webrtcClient.currentCall?.callId,
				duration: this.currentRecording.duration,
				size: this.currentRecording.size,
				timestamp: new Date().toISOString()
			});
		}
	}
	
	// Public API
	getRecordingStatus() {
		return {
			isRecording: this.isRecording,
			currentRecording: this.currentRecording ? {
				recordingId: this.currentRecording.recordingId,
				recordingType: this.currentRecording.recordingType,
				startTime: this.currentRecording.startTime,
				duration: this.currentRecording.endTime ? 
					this.currentRecording.endTime - this.currentRecording.startTime : 
					Date.now() - this.currentRecording.startTime
			} : null
		};
	}
	
	getSupportedMimeTypes() {
		const types = [
			'video/webm;codecs=vp9',
			'video/webm;codecs=vp8',
			'video/webm',
			'video/mp4',
			'audio/webm',
			'audio/mp4',
			'audio/wav'
		];
		
		return types.filter(type => MediaRecorder.isTypeSupported(type));
	}
	
	// Event handlers (can be overridden)
	onRecordingComplete(recording) {
		console.log('Recording completed:', recording);
	}
	
	onRecordingError(error) {
		console.error('Recording error:', error);
	}
}

// Export for global use
window.WebRTCRecordingClient = WebRTCRecordingClient;
"""


if __name__ == "__main__":
	# Test the recording manager
	async def test_recording():
		print("Testing WebRTC recording manager...")
		
		manager = WebRTCRecordingManager()
		
		# Test recording start
		result = await manager.start_recording(
			call_id="test_call_123",
			session_id="test_session_456",
			started_by="user123",
			recording_type=RecordingType.FULL_MEETING,
			participants=["user123", "user456", "user789"]
		)
		print(f"Recording start result: {result}")
		
		# Get recording status
		if 'recording_id' in result:
			recording_id = result['recording_id']
			status = manager.get_recording_status(recording_id)
			print(f"Recording status: {status}")
			
			# Stop recording after a short delay
			await asyncio.sleep(1)
			stop_result = await manager.stop_recording(recording_id, "user123")
			print(f"Recording stop result: {stop_result}")
			
			# Wait for post-processing
			await asyncio.sleep(3)
			
			# Get final status
			final_status = manager.get_recording_status(recording_id)
			print(f"Final recording status: {final_status}")
		
		# Get statistics
		stats = manager.get_recording_statistics()
		print(f"Recording statistics: {stats}")
		
		# Generate client JavaScript
		js_code = generate_recording_client_javascript()
		print(f"Generated JavaScript: {len(js_code)} characters")
		
		print("✅ WebRTC recording manager test completed")
	
	asyncio.run(test_recording())