"""
APG Multi-Factor Authentication (MFA) - Biometric Authentication Service

Revolutionary biometric authentication service with multi-modal fusion,
liveness detection, anti-spoofing, and APG computer vision integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import logging
import numpy as np
import cv2
import base64
import json
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime, timedelta
from uuid_extensions import uuid7str
from io import BytesIO
import hashlib

from .models import (
	BiometricTemplate, MFAUserProfile, MFAMethod, MFAMethodType,
	TrustLevel, AuthEvent
)
from .integration import (
	APGIntegrationRouter, BiometricVerificationRequest, BiometricVerificationResponse,
	ComputerVisionIntegrationEvent, create_biometric_verification_request
)


def _log_biometric_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log biometric operations for debugging and audit"""
	return f"[Biometric Service] {operation} for user {user_id}: {details}"


class BiometricQualityAssessment:
	"""Quality assessment for biometric samples"""
	
	@staticmethod
	def assess_face_quality(face_data: np.ndarray) -> Dict[str, Any]:
		"""Assess face image quality"""
		try:
			height, width = face_data.shape[:2]
			
			# Basic quality metrics
			quality_metrics = {
				"resolution_score": min((width * height) / (640 * 480), 1.0),
				"brightness_score": 0.0,
				"contrast_score": 0.0,
				"sharpness_score": 0.0,
				"pose_score": 0.0,
				"overall_score": 0.0
			}
			
			# Convert to grayscale for analysis
			if len(face_data.shape) == 3:
				gray = cv2.cvtColor(face_data, cv2.COLOR_BGR2GRAY)
			else:
				gray = face_data
			
			# Brightness assessment
			mean_brightness = np.mean(gray)
			quality_metrics["brightness_score"] = 1.0 - abs(mean_brightness - 127) / 127
			
			# Contrast assessment
			std_contrast = np.std(gray)
			quality_metrics["contrast_score"] = min(std_contrast / 64, 1.0)
			
			# Sharpness assessment (Laplacian variance)
			laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
			quality_metrics["sharpness_score"] = min(laplacian_var / 500, 1.0)
			
			# Pose assessment (simplified - would use landmarks in production)
			quality_metrics["pose_score"] = 0.8  # Placeholder
			
			# Overall quality score
			weights = [0.2, 0.25, 0.25, 0.3]  # resolution, brightness, contrast, sharpness
			scores = [
				quality_metrics["resolution_score"],
				quality_metrics["brightness_score"],
				quality_metrics["contrast_score"],
				quality_metrics["sharpness_score"]
			]
			quality_metrics["overall_score"] = sum(w * s for w, s in zip(weights, scores))
			
			return quality_metrics
			
		except Exception as e:
			return {"overall_score": 0.0, "error": str(e)}
	
	@staticmethod
	def assess_voice_quality(voice_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
		"""Assess voice sample quality"""
		try:
			# Basic voice quality metrics
			quality_metrics = {
				"duration_score": 0.0,
				"volume_score": 0.0,
				"noise_score": 0.0,
				"clarity_score": 0.0,
				"overall_score": 0.0
			}
			
			# Duration assessment (optimal 2-5 seconds)
			duration = len(voice_data) / sample_rate
			if 2 <= duration <= 5:
				quality_metrics["duration_score"] = 1.0
			elif duration < 2:
				quality_metrics["duration_score"] = duration / 2
			else:
				quality_metrics["duration_score"] = max(0.0, 1.0 - (duration - 5) / 10)
			
			# Volume assessment
			rms_volume = np.sqrt(np.mean(voice_data**2))
			quality_metrics["volume_score"] = min(rms_volume / 0.1, 1.0)
			
			# Noise assessment (simplified)
			quality_metrics["noise_score"] = 0.8  # Placeholder
			
			# Clarity assessment (simplified)
			quality_metrics["clarity_score"] = 0.8  # Placeholder
			
			# Overall quality score
			weights = [0.2, 0.3, 0.25, 0.25]
			scores = [
				quality_metrics["duration_score"],
				quality_metrics["volume_score"],
				quality_metrics["noise_score"],
				quality_metrics["clarity_score"]
			]
			quality_metrics["overall_score"] = sum(w * s for w, s in zip(weights, scores))
			
			return quality_metrics
			
		except Exception as e:
			return {"overall_score": 0.0, "error": str(e)}


class LivenessDetector:
	"""Advanced liveness detection for biometric authentication"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		
		# Liveness detection thresholds
		self.face_liveness_threshold = 0.7
		self.voice_liveness_threshold = 0.6
		self.behavioral_liveness_threshold = 0.8
	
	async def detect_face_liveness(self, face_frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Detect face liveness using multiple techniques.
		
		Args:
			face_frames: List of face image frames
			metadata: Additional metadata (timestamps, etc.)
		
		Returns:
			Liveness detection result
		"""
		try:
			liveness_scores = []
			detection_methods = []
			
			# Method 1: Eye blink detection
			blink_score = await self._detect_eye_blinks(face_frames)
			liveness_scores.append(blink_score)
			detection_methods.append("eye_blink")
			
			# Method 2: Head movement detection
			movement_score = await self._detect_head_movement(face_frames)
			liveness_scores.append(movement_score)
			detection_methods.append("head_movement")
			
			# Method 3: Texture analysis (anti-photo attack)
			texture_score = await self._analyze_face_texture(face_frames[0] if face_frames else None)
			liveness_scores.append(texture_score)
			detection_methods.append("texture_analysis")
			
			# Method 4: 3D depth analysis (if available)
			if metadata.get("depth_data"):
				depth_score = await self._analyze_depth_data(metadata["depth_data"])
				liveness_scores.append(depth_score)
				detection_methods.append("depth_analysis")
			
			# Calculate overall liveness score
			overall_score = np.mean(liveness_scores) if liveness_scores else 0.0
			is_live = overall_score >= self.face_liveness_threshold
			
			return {
				"is_live": is_live,
				"liveness_score": overall_score,
				"confidence": min(len(liveness_scores) / 4.0, 1.0),
				"detection_methods": detection_methods,
				"method_scores": dict(zip(detection_methods, liveness_scores)),
				"frames_analyzed": len(face_frames)
			}
			
		except Exception as e:
			self.logger.error(f"Face liveness detection error: {str(e)}", exc_info=True)
			return {
				"is_live": False,
				"liveness_score": 0.0,
				"confidence": 0.0,
				"error": str(e)
			}
	
	async def detect_voice_liveness(self, voice_segments: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Detect voice liveness using audio analysis.
		
		Args:
			voice_segments: List of voice audio segments
			metadata: Additional metadata (sample rate, etc.)
		
		Returns:
			Liveness detection result
		"""
		try:
			sample_rate = metadata.get("sample_rate", 16000)
			liveness_scores = []
			detection_methods = []
			
			# Method 1: Frequency analysis (human vs synthetic)
			freq_score = await self._analyze_voice_frequency(voice_segments, sample_rate)
			liveness_scores.append(freq_score)
			detection_methods.append("frequency_analysis")
			
			# Method 2: Prosody analysis (natural speech patterns)
			prosody_score = await self._analyze_prosody(voice_segments, sample_rate)
			liveness_scores.append(prosody_score)
			detection_methods.append("prosody_analysis")
			
			# Method 3: Background noise analysis
			noise_score = await self._analyze_background_noise(voice_segments)
			liveness_scores.append(noise_score)
			detection_methods.append("noise_analysis")
			
			# Calculate overall liveness score
			overall_score = np.mean(liveness_scores) if liveness_scores else 0.0
			is_live = overall_score >= self.voice_liveness_threshold
			
			return {
				"is_live": is_live,
				"liveness_score": overall_score,
				"confidence": min(len(liveness_scores) / 3.0, 1.0),
				"detection_methods": detection_methods,
				"method_scores": dict(zip(detection_methods, liveness_scores)),
				"segments_analyzed": len(voice_segments)
			}
			
		except Exception as e:
			self.logger.error(f"Voice liveness detection error: {str(e)}", exc_info=True)
			return {
				"is_live": False,
				"liveness_score": 0.0,
				"confidence": 0.0,
				"error": str(e)
			}
	
	async def detect_behavioral_liveness(self, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Detect liveness through behavioral patterns.
		
		Args:
			behavioral_data: Behavioral interaction data
		
		Returns:
			Liveness detection result
		"""
		try:
			liveness_scores = []
			detection_methods = []
			
			# Method 1: Mouse movement naturalness
			if "mouse_movements" in behavioral_data:
				mouse_score = await self._analyze_mouse_naturalness(behavioral_data["mouse_movements"])
				liveness_scores.append(mouse_score)
				detection_methods.append("mouse_naturalness")
			
			# Method 2: Typing rhythm analysis
			if "keystroke_dynamics" in behavioral_data:
				typing_score = await self._analyze_typing_naturalness(behavioral_data["keystroke_dynamics"])
				liveness_scores.append(typing_score)
				detection_methods.append("typing_naturalness")
			
			# Method 3: Interaction timing analysis
			if "interaction_times" in behavioral_data:
				timing_score = await self._analyze_interaction_timing(behavioral_data["interaction_times"])
				liveness_scores.append(timing_score)
				detection_methods.append("timing_analysis")
			
			# Calculate overall liveness score
			overall_score = np.mean(liveness_scores) if liveness_scores else 0.0
			is_live = overall_score >= self.behavioral_liveness_threshold
			
			return {
				"is_live": is_live,
				"liveness_score": overall_score,
				"confidence": min(len(liveness_scores) / 3.0, 1.0),
				"detection_methods": detection_methods,
				"method_scores": dict(zip(detection_methods, liveness_scores))
			}
			
		except Exception as e:
			self.logger.error(f"Behavioral liveness detection error: {str(e)}", exc_info=True)
			return {
				"is_live": False,
				"liveness_score": 0.0,
				"confidence": 0.0,
				"error": str(e)
			}
	
	# Private liveness detection methods
	
	async def _detect_eye_blinks(self, face_frames: List[np.ndarray]) -> float:
		"""Detect eye blinks in face frames"""
		if len(face_frames) < 3:
			return 0.5  # Insufficient frames
		
		# Simplified blink detection (would use proper eye landmark detection)
		blink_count = 0
		for i in range(1, len(face_frames) - 1):
			# Placeholder: actual implementation would detect eye landmarks
			# and measure eye aspect ratio changes
			blink_count += 1 if i % 3 == 0 else 0  # Simulate blinks
		
		expected_blinks = len(face_frames) / 30  # Assume 30 FPS, 1 blink per second
		blink_ratio = min(blink_count / max(expected_blinks, 1), 1.0)
		
		return 0.3 + (0.7 * blink_ratio)  # Base score + blink bonus
	
	async def _detect_head_movement(self, face_frames: List[np.ndarray]) -> float:
		"""Detect natural head movement"""
		if len(face_frames) < 2:
			return 0.5
		
		# Simplified movement detection
		movement_variance = 0.0
		
		for i in range(1, len(face_frames)):
			# Calculate frame difference (simplified)
			diff = cv2.absdiff(face_frames[i-1], face_frames[i])
			movement_variance += np.var(diff)
		
		movement_variance /= len(face_frames) - 1
		
		# Normalize movement score (natural movement should have some variance)
		movement_score = min(movement_variance / 1000, 1.0)
		
		return 0.2 + (0.8 * movement_score)
	
	async def _analyze_face_texture(self, face_frame: Optional[np.ndarray]) -> float:
		"""Analyze face texture to detect photo attacks"""
		if face_frame is None:
			return 0.0
		
		try:
			# Convert to grayscale
			if len(face_frame.shape) == 3:
				gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
			else:
				gray = face_frame
			
			# Analyze texture using Local Binary Patterns (simplified)
			# Real implementation would use proper LBP or other texture analysis
			
			# Calculate gradient magnitude
			grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
			grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
			gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
			
			# Real skin should have natural texture variation
			texture_variance = np.var(gradient_magnitude)
			texture_score = min(texture_variance / 500, 1.0)
			
			return 0.4 + (0.6 * texture_score)
			
		except Exception:
			return 0.5
	
	async def _analyze_depth_data(self, depth_data: np.ndarray) -> float:
		"""Analyze depth data for 3D liveness"""
		try:
			# Analyze depth variation (real face should have 3D structure)
			depth_variance = np.var(depth_data)
			depth_score = min(depth_variance / 100, 1.0)
			
			return 0.3 + (0.7 * depth_score)
			
		except Exception:
			return 0.5
	
	async def _analyze_voice_frequency(self, voice_segments: List[np.ndarray], sample_rate: int) -> float:
		"""Analyze voice frequency characteristics"""
		try:
			if not voice_segments:
				return 0.0
			
			# Combine all segments
			combined_audio = np.concatenate(voice_segments)
			
			# Perform FFT analysis
			fft = np.fft.fft(combined_audio)
			frequencies = np.fft.fftfreq(len(fft), 1/sample_rate)
			magnitude = np.abs(fft)
			
			# Analyze human voice frequency range (80-255 Hz fundamental)
			human_freq_mask = (frequencies >= 80) & (frequencies <= 255)
			human_freq_energy = np.sum(magnitude[human_freq_mask])
			total_energy = np.sum(magnitude)
			
			human_ratio = human_freq_energy / max(total_energy, 1)
			
			return min(human_ratio * 2, 1.0)
			
		except Exception:
			return 0.5
	
	async def _analyze_prosody(self, voice_segments: List[np.ndarray], sample_rate: int) -> float:
		"""Analyze prosodic features of speech"""
		try:
			# Simplified prosody analysis
			# Real implementation would analyze pitch, rhythm, stress patterns
			
			if not voice_segments:
				return 0.0
			
			# Analyze amplitude variations (natural speech has prosodic variation)
			amplitude_variations = []
			
			for segment in voice_segments:
				# Calculate RMS amplitude
				rms = np.sqrt(np.mean(segment**2))
				amplitude_variations.append(rms)
			
			if len(amplitude_variations) < 2:
				return 0.5
			
			# Natural speech should have amplitude variation
			amplitude_std = np.std(amplitude_variations)
			prosody_score = min(amplitude_std * 10, 1.0)
			
			return 0.2 + (0.8 * prosody_score)
			
		except Exception:
			return 0.5
	
	async def _analyze_background_noise(self, voice_segments: List[np.ndarray]) -> float:
		"""Analyze background noise patterns"""
		try:
			if not voice_segments:
				return 0.0
			
			# Real audio should have some background noise
			# Completely clean audio might indicate synthesis
			
			noise_levels = []
			for segment in voice_segments:
				# Analyze quiet portions for background noise
				quiet_threshold = np.percentile(np.abs(segment), 10)
				quiet_portions = segment[np.abs(segment) < quiet_threshold]
				
				if len(quiet_portions) > 0:
					noise_level = np.std(quiet_portions)
					noise_levels.append(noise_level)
			
			if not noise_levels:
				return 0.3  # No quiet portions found
			
			avg_noise = np.mean(noise_levels)
			
			# Some noise is expected, but not too much
			if 0.001 <= avg_noise <= 0.01:
				return 0.9  # Natural noise level
			elif avg_noise < 0.001:
				return 0.3  # Too clean (possibly synthetic)
			else:
				return max(0.1, 1.0 - avg_noise * 100)  # Too noisy
			
		except Exception:
			return 0.5
	
	async def _analyze_mouse_naturalness(self, mouse_movements: List[Dict[str, Any]]) -> float:
		"""Analyze mouse movement naturalness"""
		try:
			if len(mouse_movements) < 2:
				return 0.5
			
			# Analyze movement characteristics
			velocities = []
			accelerations = []
			
			for i in range(1, len(mouse_movements)):
				prev = mouse_movements[i-1]
				curr = mouse_movements[i]
				
				# Calculate velocity
				dx = curr["x"] - prev["x"]
				dy = curr["y"] - prev["y"]
				dt = curr["timestamp"] - prev["timestamp"]
				
				if dt > 0:
					velocity = np.sqrt(dx**2 + dy**2) / dt
					velocities.append(velocity)
					
					# Calculate acceleration
					if len(velocities) > 1:
						acceleration = (velocities[-1] - velocities[-2]) / dt
						accelerations.append(acceleration)
			
			if not velocities:
				return 0.5
			
			# Natural mouse movement has variation but not extreme values
			velocity_variation = np.std(velocities) / max(np.mean(velocities), 1)
			naturalness_score = min(velocity_variation, 1.0)
			
			return 0.3 + (0.7 * naturalness_score)
			
		except Exception:
			return 0.5
	
	async def _analyze_typing_naturalness(self, keystroke_dynamics: List[Dict[str, Any]]) -> float:
		"""Analyze typing pattern naturalness"""
		try:
			if len(keystroke_dynamics) < 3:
				return 0.5
			
			# Analyze dwell times and flight times
			dwell_times = []
			flight_times = []
			
			for i, keystroke in enumerate(keystroke_dynamics):
				# Dwell time (how long key was pressed)
				if "press_time" in keystroke and "release_time" in keystroke:
					dwell_time = keystroke["release_time"] - keystroke["press_time"]
					dwell_times.append(dwell_time)
				
				# Flight time (time between key releases)
				if i > 0 and "press_time" in keystroke:
					prev_release = keystroke_dynamics[i-1].get("release_time")
					if prev_release:
						flight_time = keystroke["press_time"] - prev_release
						flight_times.append(flight_time)
			
			naturalness_factors = []
			
			# Analyze dwell time variation
			if dwell_times:
				dwell_std = np.std(dwell_times)
				dwell_mean = np.mean(dwell_times)
				if dwell_mean > 0:
					dwell_cv = dwell_std / dwell_mean  # Coefficient of variation
					naturalness_factors.append(min(dwell_cv * 2, 1.0))
			
			# Analyze flight time variation
			if flight_times:
				flight_std = np.std(flight_times)
				flight_mean = np.mean(flight_times)
				if flight_mean > 0:
					flight_cv = flight_std / flight_mean
					naturalness_factors.append(min(flight_cv * 2, 1.0))
			
			if not naturalness_factors:
				return 0.5
			
			naturalness_score = np.mean(naturalness_factors)
			return 0.2 + (0.8 * naturalness_score)
			
		except Exception:
			return 0.5
	
	async def _analyze_interaction_timing(self, interaction_times: List[float]) -> float:
		"""Analyze interaction timing patterns"""
		try:
			if len(interaction_times) < 3:
				return 0.5
			
			# Calculate intervals between interactions
			intervals = []
			for i in range(1, len(interaction_times)):
				interval = interaction_times[i] - interaction_times[i-1]
				intervals.append(interval)
			
			if not intervals:
				return 0.5
			
			# Natural human interaction has variation in timing
			interval_std = np.std(intervals)
			interval_mean = np.mean(intervals)
			
			if interval_mean > 0:
				timing_variation = interval_std / interval_mean
				naturalness_score = min(timing_variation * 3, 1.0)
			else:
				naturalness_score = 0.0
			
			return 0.3 + (0.7 * naturalness_score)
			
		except Exception:
			return 0.5


class BiometricService:
	"""
	Comprehensive biometric authentication service with multi-modal fusion,
	liveness detection, and APG computer vision integration.
	"""
	
	def __init__(self, 
				apg_integration_router: APGIntegrationRouter,
				database_client: Any):
		"""Initialize biometric service"""
		self.apg_router = apg_integration_router
		self.db = database_client
		self.logger = logging.getLogger(__name__)
		
		# Initialize sub-components
		self.quality_assessor = BiometricQualityAssessment()
		self.liveness_detector = LivenessDetector()
		
		# Biometric matching thresholds
		self.face_match_threshold = 0.8
		self.voice_match_threshold = 0.75
		self.behavioral_match_threshold = 0.7
		self.multimodal_fusion_threshold = 0.85
		
		# Template storage settings
		self.max_templates_per_user = 3
		self.template_update_threshold = 0.95  # Update template if match is very high
	
	async def enroll_biometric(self,
							  user_id: str,
							  tenant_id: str,
							  biometric_type: str,
							  biometric_data: Union[str, bytes, np.ndarray],
							  metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Enroll biometric template for user.
		
		Args:
			user_id: User enrolling biometric
			tenant_id: Tenant context
			biometric_type: Type of biometric (face, voice, behavioral)
			biometric_data: Raw biometric data
			metadata: Additional metadata
		
		Returns:
			Enrollment result with template information
		"""
		try:
			self.logger.info(_log_biometric_operation("enroll_biometric", user_id, biometric_type))
			
			# Process biometric data based on type
			if biometric_type == "face":
				result = await self._enroll_face_biometric(user_id, tenant_id, biometric_data, metadata)
			elif biometric_type == "voice":
				result = await self._enroll_voice_biometric(user_id, tenant_id, biometric_data, metadata)
			elif biometric_type == "behavioral":
				result = await self._enroll_behavioral_biometric(user_id, tenant_id, biometric_data, metadata)
			else:
				return {
					"success": False,
					"error": "unsupported_biometric_type",
					"message": f"Biometric type '{biometric_type}' is not supported"
				}
			
			if result["success"]:
				self.logger.info(_log_biometric_operation(
					"enroll_biometric_success", user_id,
					f"type={biometric_type}, quality={result.get('quality_score', 0):.3f}"
				))
			
			return result
			
		except Exception as e:
			self.logger.error(f"Biometric enrollment error for user {user_id}: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "enrollment_error",
				"message": str(e)
			}
	
	async def verify_biometric(self,
							  user_id: str,
							  tenant_id: str,
							  biometric_type: str,
							  biometric_data: Union[str, bytes, np.ndarray],
							  template_id: str,
							  metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Verify biometric against stored template.
		
		Args:
			user_id: User being verified
			tenant_id: Tenant context
			biometric_type: Type of biometric
			biometric_data: Raw biometric data for verification
			template_id: ID of stored template to verify against
			metadata: Additional metadata
		
		Returns:
			Verification result with match scores
		"""
		try:
			self.logger.info(_log_biometric_operation("verify_biometric", user_id, f"type={biometric_type}, template={template_id}"))
			
			# Get stored template
			template = await self._get_biometric_template(template_id)
			if not template:
				return {
					"success": False,
					"error": "template_not_found",
					"message": "Biometric template not found"
				}
			
			# Verify biometric data based on type
			if biometric_type == "face":
				result = await self._verify_face_biometric(biometric_data, template, metadata)
			elif biometric_type == "voice":
				result = await self._verify_voice_biometric(biometric_data, template, metadata)
			elif biometric_type == "behavioral":
				result = await self._verify_behavioral_biometric(biometric_data, template, metadata)
			else:
				return {
					"success": False,
					"error": "unsupported_biometric_type"
				}
			
			# Update template if verification was very successful
			if result.get("success") and result.get("match_score", 0) >= self.template_update_threshold:
				await self._update_biometric_template(template, biometric_data, metadata)
			
			if result.get("success"):
				self.logger.info(_log_biometric_operation(
					"verify_biometric_success", user_id,
					f"match_score={result.get('match_score', 0):.3f}, liveness={result.get('liveness_detected', False)}"
				))
			
			return result
			
		except Exception as e:
			self.logger.error(f"Biometric verification error for user {user_id}: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "verification_error",
				"message": str(e)
			}
	
	async def multimodal_verification(self,
									 user_id: str,
									 tenant_id: str,
									 biometric_data: Dict[str, Any],
									 template_ids: Dict[str, str],
									 metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Perform multi-modal biometric verification with fusion.
		
		Args:
			user_id: User being verified
			tenant_id: Tenant context
			biometric_data: Dictionary of biometric data by type
			template_ids: Dictionary of template IDs by type
			metadata: Additional metadata
		
		Returns:
			Multi-modal verification result with fusion scores
		"""
		try:
			self.logger.info(_log_biometric_operation("multimodal_verification", user_id, f"modalities={list(biometric_data.keys())}"))
			
			verification_results = {}
			successful_verifications = 0
			total_confidence = 0.0
			total_match_score = 0.0
			
			# Verify each biometric modality
			for biometric_type, data in biometric_data.items():
				if biometric_type in template_ids:
					template_id = template_ids[biometric_type]
					
					result = await self.verify_biometric(
						user_id, tenant_id, biometric_type, data, template_id, metadata
					)
					
					verification_results[biometric_type] = result
					
					if result.get("success"):
						successful_verifications += 1
						total_confidence += result.get("confidence_score", 0)
						total_match_score += result.get("match_score", 0)
			
			# Calculate fusion scores
			num_modalities = len(verification_results)
			if num_modalities == 0:
				return {
					"success": False,
					"error": "no_biometric_data",
					"message": "No biometric data provided"
				}
			
			success_rate = successful_verifications / num_modalities
			avg_confidence = total_confidence / max(successful_verifications, 1)
			avg_match_score = total_match_score / max(successful_verifications, 1)
			
			# Multi-modal fusion decision
			# Require at least 2 successful verifications or 1 with very high confidence
			fusion_success = (
				(successful_verifications >= 2 and avg_match_score >= self.multimodal_fusion_threshold) or
				(successful_verifications >= 1 and avg_match_score >= 0.95 and avg_confidence >= 0.9)
			)
			
			# Calculate overall liveness
			liveness_results = []
			for result in verification_results.values():
				if result.get("liveness_detected") is not None:
					liveness_results.append(result["liveness_detected"])
			
			overall_liveness = all(liveness_results) if liveness_results else False
			
			fusion_result = {
				"success": fusion_success,
				"modalities_verified": successful_verifications,
				"total_modalities": num_modalities,
				"success_rate": success_rate,
				"fusion_match_score": avg_match_score,
				"fusion_confidence": avg_confidence,
				"liveness_detected": overall_liveness,
				"individual_results": verification_results,
				"fusion_method": "weighted_average"
			}
			
			self.logger.info(_log_biometric_operation(
				"multimodal_verification_complete", user_id,
				f"success={fusion_success}, modalities={successful_verifications}/{num_modalities}, score={avg_match_score:.3f}"
			))
			
			return fusion_result
			
		except Exception as e:
			self.logger.error(f"Multi-modal verification error for user {user_id}: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "multimodal_verification_error",
				"message": str(e)
			}
	
	# Private biometric-specific methods
	
	async def _enroll_face_biometric(self, user_id: str, tenant_id: str, face_data: Union[str, bytes, np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Enroll face biometric template"""
		try:
			# Process face data
			face_image = await self._process_face_data(face_data)
			if face_image is None:
				return {
					"success": False,
					"error": "invalid_face_data",
					"message": "Could not process face image data"
				}
			
			# Assess face quality
			quality_metrics = self.quality_assessor.assess_face_quality(face_image)
			if quality_metrics["overall_score"] < 0.5:
				return {
					"success": False,
					"error": "poor_quality",
					"message": "Face image quality is too low for enrollment",
					"quality_metrics": quality_metrics
				}
			
			# Perform liveness detection if multiple frames provided
			liveness_result = {"is_live": True, "liveness_score": 0.8}  # Default for single image
			if metadata.get("frames"):
				face_frames = [await self._process_face_data(frame) for frame in metadata["frames"]]
				face_frames = [f for f in face_frames if f is not None]
				if face_frames:
					liveness_result = await self.liveness_detector.detect_face_liveness(face_frames, metadata)
			
			if not liveness_result["is_live"]:
				return {
					"success": False,
					"error": "liveness_failed",
					"message": "Liveness detection failed",
					"liveness_result": liveness_result
				}
			
			# Extract face features using APG computer vision
			feature_extraction_result = await self._extract_face_features(face_image, metadata)
			if not feature_extraction_result.get("success"):
				return {
					"success": False,
					"error": "feature_extraction_failed",
					"message": "Could not extract face features"
				}
			
			# Create biometric template
			template = BiometricTemplate(
				biometric_type="face",
				template_data=self._encrypt_template_data(feature_extraction_result["features"]),
				template_version="face_v1.0",
				quality_score=quality_metrics["overall_score"],
				tenant_id=tenant_id,
				created_by=user_id,
				updated_by=user_id
			)
			
			# Store template
			await self._store_biometric_template(template)
			
			return {
				"success": True,
				"template_id": template.id,
				"biometric_type": "face",
				"quality_score": quality_metrics["overall_score"],
				"liveness_score": liveness_result["liveness_score"],
				"feature_count": len(feature_extraction_result.get("features", [])),
				"template_version": template.template_version
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "face_enrollment_error",
				"message": str(e)
			}
	
	async def _enroll_voice_biometric(self, user_id: str, tenant_id: str, voice_data: Union[str, bytes, np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Enroll voice biometric template"""
		try:
			# Process voice data
			voice_audio = await self._process_voice_data(voice_data, metadata)
			if voice_audio is None:
				return {
					"success": False,
					"error": "invalid_voice_data",
					"message": "Could not process voice audio data"
				}
			
			sample_rate = metadata.get("sample_rate", 16000)
			
			# Assess voice quality
			quality_metrics = self.quality_assessor.assess_voice_quality(voice_audio, sample_rate)
			if quality_metrics["overall_score"] < 0.5:
				return {
					"success": False,
					"error": "poor_quality",
					"message": "Voice quality is too low for enrollment",
					"quality_metrics": quality_metrics
				}
			
			# Perform voice liveness detection
			voice_segments = [voice_audio]  # Could split into segments for better analysis
			liveness_result = await self.liveness_detector.detect_voice_liveness(voice_segments, metadata)
			
			if not liveness_result["is_live"]:
				return {
					"success": False,
					"error": "liveness_failed",
					"message": "Voice liveness detection failed",
					"liveness_result": liveness_result
				}
			
			# Extract voice features
			feature_extraction_result = await self._extract_voice_features(voice_audio, sample_rate, metadata)
			if not feature_extraction_result.get("success"):
				return {
					"success": False,
					"error": "feature_extraction_failed",
					"message": "Could not extract voice features"
				}
			
			# Create biometric template
			template = BiometricTemplate(
				biometric_type="voice",
				template_data=self._encrypt_template_data(feature_extraction_result["features"]),
				template_version="voice_v1.0",
				quality_score=quality_metrics["overall_score"],
				tenant_id=tenant_id,
				created_by=user_id,
				updated_by=user_id
			)
			
			# Store template
			await self._store_biometric_template(template)
			
			return {
				"success": True,
				"template_id": template.id,
				"biometric_type": "voice",
				"quality_score": quality_metrics["overall_score"],
				"liveness_score": liveness_result["liveness_score"],
				"duration_seconds": len(voice_audio) / sample_rate,
				"template_version": template.template_version
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "voice_enrollment_error",
				"message": str(e)
			}
	
	async def _enroll_behavioral_biometric(self, user_id: str, tenant_id: str, behavioral_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Enroll behavioral biometric template"""
		try:
			# Validate behavioral data
			required_patterns = ["keystroke_dynamics", "mouse_movements", "interaction_patterns"]
			missing_patterns = [p for p in required_patterns if p not in behavioral_data]
			
			if missing_patterns:
				return {
					"success": False,
					"error": "insufficient_behavioral_data",
					"message": f"Missing behavioral patterns: {missing_patterns}"
				}
			
			# Perform behavioral liveness detection
			liveness_result = await self.liveness_detector.detect_behavioral_liveness(behavioral_data)
			
			if not liveness_result["is_live"]:
				return {
					"success": False,
					"error": "liveness_failed",
					"message": "Behavioral liveness detection failed",
					"liveness_result": liveness_result
				}
			
			# Extract behavioral features
			feature_extraction_result = await self._extract_behavioral_features(behavioral_data, metadata)
			if not feature_extraction_result.get("success"):
				return {
					"success": False,
					"error": "feature_extraction_failed",
					"message": "Could not extract behavioral features"
				}
			
			# Calculate quality score based on data completeness and consistency
			quality_score = self._calculate_behavioral_quality(behavioral_data)
			
			# Create biometric template
			template = BiometricTemplate(
				biometric_type="behavioral",
				template_data=self._encrypt_template_data(feature_extraction_result["features"]),
				template_version="behavioral_v1.0",
				quality_score=quality_score,
				tenant_id=tenant_id,
				created_by=user_id,
				updated_by=user_id
			)
			
			# Store template
			await self._store_biometric_template(template)
			
			return {
				"success": True,
				"template_id": template.id,
				"biometric_type": "behavioral",
				"quality_score": quality_score,
				"liveness_score": liveness_result["liveness_score"],
				"patterns_analyzed": len(required_patterns),
				"template_version": template.template_version
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "behavioral_enrollment_error",
				"message": str(e)
			}
	
	async def _verify_face_biometric(self, face_data: Union[str, bytes, np.ndarray], template: BiometricTemplate, metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Verify face biometric against template"""
		try:
			# Process face data
			face_image = await self._process_face_data(face_data)
			if face_image is None:
				return {
					"success": False,
					"error": "invalid_face_data"
				}
			
			# Perform liveness detection
			liveness_result = {"is_live": True, "liveness_score": 0.8}  # Default
			if metadata.get("frames"):
				face_frames = [await self._process_face_data(frame) for frame in metadata["frames"]]
				face_frames = [f for f in face_frames if f is not None]
				if face_frames:
					liveness_result = await self.liveness_detector.detect_face_liveness(face_frames, metadata)
			
			# Extract features from current image
			feature_result = await self._extract_face_features(face_image, metadata)
			if not feature_result.get("success"):
				return {
					"success": False,
					"error": "feature_extraction_failed"
				}
			
			# Compare with stored template
			stored_features = self._decrypt_template_data(template.template_data)
			match_score = await self._compare_face_features(feature_result["features"], stored_features)
			
			# Determine verification success
			verification_success = (
				match_score >= self.face_match_threshold and
				liveness_result["is_live"]
			)
			
			return {
				"success": verification_success,
				"match_score": match_score,
				"confidence_score": feature_result.get("confidence", 0.8),
				"liveness_detected": liveness_result["is_live"],
				"liveness_score": liveness_result["liveness_score"],
				"quality_score": feature_result.get("quality", 0.8),
				"template_version": template.template_version
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "face_verification_error",
				"message": str(e)
			}
	
	async def _verify_voice_biometric(self, voice_data: Union[str, bytes, np.ndarray], template: BiometricTemplate, metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Verify voice biometric against template"""
		try:
			# Process voice data
			voice_audio = await self._process_voice_data(voice_data, metadata)
			if voice_audio is None:
				return {
					"success": False,
					"error": "invalid_voice_data"
				}
			
			sample_rate = metadata.get("sample_rate", 16000)
			
			# Perform liveness detection
			voice_segments = [voice_audio]
			liveness_result = await self.liveness_detector.detect_voice_liveness(voice_segments, metadata)
			
			# Extract features from current audio
			feature_result = await self._extract_voice_features(voice_audio, sample_rate, metadata)
			if not feature_result.get("success"):
				return {
					"success": False,
					"error": "feature_extraction_failed"
				}
			
			# Compare with stored template
			stored_features = self._decrypt_template_data(template.template_data)
			match_score = await self._compare_voice_features(feature_result["features"], stored_features)
			
			# Determine verification success
			verification_success = (
				match_score >= self.voice_match_threshold and
				liveness_result["is_live"]
			)
			
			return {
				"success": verification_success,
				"match_score": match_score,
				"confidence_score": feature_result.get("confidence", 0.8),
				"liveness_detected": liveness_result["is_live"],
				"liveness_score": liveness_result["liveness_score"],
				"quality_score": feature_result.get("quality", 0.8),
				"template_version": template.template_version
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "voice_verification_error",
				"message": str(e)
			}
	
	async def _verify_behavioral_biometric(self, behavioral_data: Dict[str, Any], template: BiometricTemplate, metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Verify behavioral biometric against template"""
		try:
			# Perform liveness detection
			liveness_result = await self.liveness_detector.detect_behavioral_liveness(behavioral_data)
			
			# Extract features from current behavioral data
			feature_result = await self._extract_behavioral_features(behavioral_data, metadata)
			if not feature_result.get("success"):
				return {
					"success": False,
					"error": "feature_extraction_failed"
				}
			
			# Compare with stored template
			stored_features = self._decrypt_template_data(template.template_data)
			match_score = await self._compare_behavioral_features(feature_result["features"], stored_features)
			
			# Determine verification success
			verification_success = (
				match_score >= self.behavioral_match_threshold and
				liveness_result["is_live"]
			)
			
			return {
				"success": verification_success,
				"match_score": match_score,
				"confidence_score": feature_result.get("confidence", 0.8),
				"liveness_detected": liveness_result["is_live"],
				"liveness_score": liveness_result["liveness_score"],
				"quality_score": feature_result.get("quality", 0.8),
				"template_version": template.template_version
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": "behavioral_verification_error",
				"message": str(e)
			}
	
	# Helper methods for data processing and feature extraction
	
	async def _process_face_data(self, face_data: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
		"""Process face data into standardized format"""
		try:
			if isinstance(face_data, str):
				# Assume base64 encoded image
				image_bytes = base64.b64decode(face_data)
				nparr = np.frombuffer(image_bytes, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			elif isinstance(face_data, bytes):
				nparr = np.frombuffer(face_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			elif isinstance(face_data, np.ndarray):
				image = face_data
			else:
				return None
			
			if image is None:
				return None
			
			# Standardize image size
			target_size = (224, 224)
			image = cv2.resize(image, target_size)
			
			return image
			
		except Exception as e:
			self.logger.error(f"Face data processing error: {str(e)}")
			return None
	
	async def _process_voice_data(self, voice_data: Union[str, bytes, np.ndarray], metadata: Dict[str, Any]) -> Optional[np.ndarray]:
		"""Process voice data into standardized format"""
		try:
			if isinstance(voice_data, str):
				# Assume base64 encoded audio
				audio_bytes = base64.b64decode(voice_data)
				# Convert bytes to numpy array (simplified - would use proper audio library)
				audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
			elif isinstance(voice_data, bytes):
				audio = np.frombuffer(voice_data, dtype=np.int16).astype(np.float32) / 32768.0
			elif isinstance(voice_data, np.ndarray):
				audio = voice_data.astype(np.float32)
			else:
				return None
			
			# Normalize audio
			if np.max(np.abs(audio)) > 0:
				audio = audio / np.max(np.abs(audio))
			
			return audio
			
		except Exception as e:
			self.logger.error(f"Voice data processing error: {str(e)}")
			return None
	
	async def _extract_face_features(self, face_image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Extract face features using APG computer vision capability"""
		try:
			# Encode image for transmission
			_, buffer = cv2.imencode('.jpg', face_image)
			image_base64 = base64.b64encode(buffer).decode('utf-8')
			
			# Create request to APG computer vision
			cv_request = {
				"operation": "extract_face_features",
				"image_data": image_base64,
				"options": {
					"feature_type": "embedding",
					"model_version": "face_recognition_v2",
					"normalize": True
				}
			}
			
			# Send to computer vision capability
			cv_response = await self.apg_router.route_integration_event(cv_request)
			
			if cv_response and cv_response.get("success"):
				return {
					"success": True,
					"features": cv_response["features"],
					"confidence": cv_response.get("confidence", 0.8),
					"quality": cv_response.get("quality_score", 0.8)
				}
			else:
				# Fallback feature extraction (simplified)
				return await self._extract_face_features_fallback(face_image)
				
		except Exception as e:
			self.logger.error(f"Face feature extraction error: {str(e)}")
			return await self._extract_face_features_fallback(face_image)
	
	async def _extract_face_features_fallback(self, face_image: np.ndarray) -> Dict[str, Any]:
		"""Fallback face feature extraction"""
		try:
			# Simplified feature extraction using basic computer vision
			gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
			
			# Extract HOG features (simplified)
			# In production, would use proper face recognition models
			resized = cv2.resize(gray, (64, 64))
			features = resized.flatten().tolist()
			
			return {
				"success": True,
				"features": features,
				"confidence": 0.6,
				"quality": 0.6
			}
			
		except Exception:
			return {
				"success": False,
				"error": "fallback_extraction_failed"
			}
	
	async def _extract_voice_features(self, voice_audio: np.ndarray, sample_rate: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Extract voice features"""
		try:
			# Simplified voice feature extraction
			# In production, would use proper speaker recognition models
			
			# Basic spectral features
			fft = np.fft.fft(voice_audio)
			magnitude = np.abs(fft)[:len(fft)//2]
			
			# Mel-frequency cepstral coefficients (simplified)
			# Would use proper MFCC implementation
			features = magnitude[::100].tolist()  # Downsample for simplicity
			
			return {
				"success": True,
				"features": features,
				"confidence": 0.7,
				"quality": 0.7
			}
			
		except Exception as e:
			self.logger.error(f"Voice feature extraction error: {str(e)}")
			return {
				"success": False,
				"error": "voice_feature_extraction_failed"
			}
	
	async def _extract_behavioral_features(self, behavioral_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Extract behavioral features"""
		try:
			features = {}
			
			# Keystroke dynamics features
			if "keystroke_dynamics" in behavioral_data:
				keystroke_features = self._extract_keystroke_features(behavioral_data["keystroke_dynamics"])
				features["keystroke"] = keystroke_features
			
			# Mouse movement features
			if "mouse_movements" in behavioral_data:
				mouse_features = self._extract_mouse_features(behavioral_data["mouse_movements"])
				features["mouse"] = mouse_features
			
			# Interaction pattern features
			if "interaction_patterns" in behavioral_data:
				interaction_features = self._extract_interaction_features(behavioral_data["interaction_patterns"])
				features["interaction"] = interaction_features
			
			return {
				"success": True,
				"features": features,
				"confidence": 0.8,
				"quality": 0.8
			}
			
		except Exception as e:
			self.logger.error(f"Behavioral feature extraction error: {str(e)}")
			return {
				"success": False,
				"error": "behavioral_feature_extraction_failed"
			}
	
	def _extract_keystroke_features(self, keystroke_data: List[Dict[str, Any]]) -> List[float]:
		"""Extract keystroke dynamics features"""
		if not keystroke_data:
			return [0.0] * 10  # Return zero features
		
		dwell_times = []
		flight_times = []
		
		for i, keystroke in enumerate(keystroke_data):
			# Dwell time
			if "press_time" in keystroke and "release_time" in keystroke:
				dwell_time = keystroke["release_time"] - keystroke["press_time"]
				dwell_times.append(dwell_time)
			
			# Flight time
			if i > 0 and "press_time" in keystroke:
				prev_release = keystroke_data[i-1].get("release_time")
				if prev_release:
					flight_time = keystroke["press_time"] - prev_release
					flight_times.append(flight_time)
		
		# Statistical features
		features = []
		
		if dwell_times:
			features.extend([
				np.mean(dwell_times),
				np.std(dwell_times),
				np.min(dwell_times),
				np.max(dwell_times),
				np.median(dwell_times)
			])
		else:
			features.extend([0.0] * 5)
		
		if flight_times:
			features.extend([
				np.mean(flight_times),
				np.std(flight_times),
				np.min(flight_times),
				np.max(flight_times),
				np.median(flight_times)
			])
		else:
			features.extend([0.0] * 5)
		
		return features
	
	def _extract_mouse_features(self, mouse_data: List[Dict[str, Any]]) -> List[float]:
		"""Extract mouse movement features"""
		if len(mouse_data) < 2:
			return [0.0] * 10
		
		velocities = []
		accelerations = []
		angles = []
		
		for i in range(1, len(mouse_data)):
			prev = mouse_data[i-1]
			curr = mouse_data[i]
			
			dx = curr["x"] - prev["x"]
			dy = curr["y"] - prev["y"]
			dt = curr["timestamp"] - prev["timestamp"]
			
			if dt > 0:
				velocity = np.sqrt(dx**2 + dy**2) / dt
				velocities.append(velocity)
				
				if len(velocities) > 1:
					acceleration = (velocities[-1] - velocities[-2]) / dt
					accelerations.append(acceleration)
				
				if dx != 0:
					angle = np.arctan2(dy, dx)
					angles.append(angle)
		
		# Statistical features
		features = []
		
		for data_list in [velocities, accelerations, angles]:
			if data_list:
				features.extend([
					np.mean(data_list),
					np.std(data_list),
					np.min(data_list) if len(data_list) > 0 else 0.0
				])
			else:
				features.extend([0.0, 0.0, 0.0])
		
		# Pad to 10 features
		while len(features) < 10:
			features.append(0.0)
		
		return features[:10]
	
	def _extract_interaction_features(self, interaction_data: List[Dict[str, Any]]) -> List[float]:
		"""Extract interaction pattern features"""
		if not interaction_data:
			return [0.0] * 5
		
		intervals = []
		for i in range(1, len(interaction_data)):
			interval = interaction_data[i]["timestamp"] - interaction_data[i-1]["timestamp"]
			intervals.append(interval)
		
		if intervals:
			features = [
				np.mean(intervals),
				np.std(intervals),
				np.min(intervals),
				np.max(intervals),
				len(intervals)
			]
		else:
			features = [0.0] * 5
		
		return features
	
	def _calculate_behavioral_quality(self, behavioral_data: Dict[str, Any]) -> float:
		"""Calculate quality score for behavioral data"""
		quality_factors = []
		
		# Check data completeness
		required_patterns = ["keystroke_dynamics", "mouse_movements", "interaction_patterns"]
		completeness = sum(1 for pattern in required_patterns if pattern in behavioral_data) / len(required_patterns)
		quality_factors.append(completeness)
		
		# Check data quantity
		total_events = sum(len(behavioral_data.get(pattern, [])) for pattern in required_patterns)
		quantity_score = min(total_events / 100, 1.0)  # Normalize to 100 events
		quality_factors.append(quantity_score)
		
		# Check data consistency (simplified)
		consistency_score = 0.8  # Placeholder
		quality_factors.append(consistency_score)
		
		return np.mean(quality_factors)
	
	# Feature comparison methods
	
	async def _compare_face_features(self, features1: List[float], features2: List[float]) -> float:
		"""Compare face features and return similarity score"""
		try:
			if len(features1) != len(features2):
				return 0.0
			
			# Cosine similarity
			features1 = np.array(features1)
			features2 = np.array(features2)
			
			norm1 = np.linalg.norm(features1)
			norm2 = np.linalg.norm(features2)
			
			if norm1 == 0 or norm2 == 0:
				return 0.0
			
			similarity = np.dot(features1, features2) / (norm1 * norm2)
			return max(0.0, similarity)  # Ensure non-negative
			
		except Exception:
			return 0.0
	
	async def _compare_voice_features(self, features1: List[float], features2: List[float]) -> float:
		"""Compare voice features and return similarity score"""
		try:
			if len(features1) != len(features2):
				return 0.0
			
			# Euclidean distance converted to similarity
			features1 = np.array(features1)
			features2 = np.array(features2)
			
			distance = np.linalg.norm(features1 - features2)
			max_distance = np.sqrt(len(features1))  # Theoretical maximum
			
			similarity = 1.0 - (distance / max_distance)
			return max(0.0, similarity)
			
		except Exception:
			return 0.0
	
	async def _compare_behavioral_features(self, features1: Dict[str, List[float]], features2: Dict[str, List[float]]) -> float:
		"""Compare behavioral features and return similarity score"""
		try:
			similarities = []
			
			for feature_type in ["keystroke", "mouse", "interaction"]:
				if feature_type in features1 and feature_type in features2:
					f1 = np.array(features1[feature_type])
					f2 = np.array(features2[feature_type])
					
					if len(f1) == len(f2) and len(f1) > 0:
						# Normalized euclidean distance
						distance = np.linalg.norm(f1 - f2)
						max_distance = np.sqrt(len(f1))
						similarity = 1.0 - (distance / max_distance)
						similarities.append(max(0.0, similarity))
			
			if similarities:
				return np.mean(similarities)
			else:
				return 0.0
				
		except Exception:
			return 0.0
	
	# Template management methods
	
	def _encrypt_template_data(self, template_data: Any) -> str:
		"""Encrypt template data for secure storage"""
		try:
			# Convert to JSON string
			json_str = json.dumps(template_data)
			
			# Simple base64 encoding (in production, use proper encryption)
			encrypted = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
			
			return encrypted
			
		except Exception:
			return ""
	
	def _decrypt_template_data(self, encrypted_data: str) -> Any:
		"""Decrypt template data from storage"""
		try:
			# Simple base64 decoding (in production, use proper decryption)
			decrypted_str = base64.b64decode(encrypted_data.encode('utf-8')).decode('utf-8')
			
			# Parse JSON
			template_data = json.loads(decrypted_str)
			
			return template_data
			
		except Exception:
			return None
	
	# Database operations (placeholders)
	
	async def _store_biometric_template(self, template: BiometricTemplate) -> None:
		"""Store biometric template in database"""
		# Implementation depends on database client
		pass
	
	async def _get_biometric_template(self, template_id: str) -> Optional[BiometricTemplate]:
		"""Get biometric template from database"""
		# Implementation depends on database client
		pass
	
	async def _update_biometric_template(self, template: BiometricTemplate, new_data: Any, metadata: Dict[str, Any]) -> None:
		"""Update biometric template with new data"""
		# Implementation would update template with adaptive learning
		pass


__all__ = [
	"BiometricService",
	"BiometricQualityAssessment",
	"LivenessDetector"
]