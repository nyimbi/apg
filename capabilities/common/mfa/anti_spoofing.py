"""
APG Multi-Factor Authentication (MFA) - Anti-Spoofing Security Module

Advanced anti-spoofing mechanisms for biometric authentication including
presentation attack detection, deepfake detection, and security hardening.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import logging
import numpy as np
import cv2
import hashlib
import hmac
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime, timedelta
from uuid_extensions import uuid7str
import base64
import json

from .models import AuthEvent, RiskAssessment, RiskLevel


def _log_antispoofing_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log anti-spoofing operations for debugging and audit"""
	return f"[Anti-Spoofing] {operation} for user {user_id}: {details}"


class PresentationAttackDetector:
	"""Presentation Attack Detection (PAD) for various spoofing attempts"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		
		# Detection thresholds
		self.photo_attack_threshold = 0.7
		self.video_replay_threshold = 0.8
		self.mask_attack_threshold = 0.75
		self.deepfake_threshold = 0.85
		
		# Analysis parameters
		self.min_frames_for_analysis = 3
		self.max_analysis_time_seconds = 10
	
	async def detect_face_presentation_attack(self, 
											 face_frames: List[np.ndarray], 
											 metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Comprehensive face presentation attack detection.
		
		Args:
			face_frames: List of face image frames
			metadata: Additional metadata (timestamps, device info, etc.)
		
		Returns:
			Detection result with attack probability and type
		"""
		try:
			self.logger.info(f"Starting face PAD analysis with {len(face_frames)} frames")
			
			if len(face_frames) < self.min_frames_for_analysis:
				return {
					"is_attack": True,
					"attack_probability": 0.8,
					"attack_type": "insufficient_frames",
					"confidence": 0.9,
					"reason": "Too few frames for reliable analysis"
				}
			
			detection_results = []
			
			# 1. Photo attack detection
			photo_result = await self._detect_photo_attack(face_frames, metadata)
			detection_results.append(("photo_attack", photo_result))
			
			# 2. Video replay attack detection
			replay_result = await self._detect_video_replay_attack(face_frames, metadata)
			detection_results.append(("video_replay", replay_result))
			
			# 3. 3D mask attack detection
			mask_result = await self._detect_mask_attack(face_frames, metadata)
			detection_results.append(("mask_attack", mask_result))
			
			# 4. Deepfake detection
			deepfake_result = await self._detect_deepfake_attack(face_frames, metadata)
			detection_results.append(("deepfake", deepfake_result))
			
			# 5. Environmental consistency checks
			env_result = await self._check_environmental_consistency(face_frames, metadata)
			detection_results.append(("environmental", env_result))
			
			# Combine results
			combined_result = self._combine_detection_results(detection_results)
			
			self.logger.info(f"Face PAD complete: attack={combined_result['is_attack']}, "
						   f"probability={combined_result['attack_probability']:.3f}")
			
			return combined_result
			
		except Exception as e:
			self.logger.error(f"Face PAD error: {str(e)}", exc_info=True)
			return {
				"is_attack": True,
				"attack_probability": 0.9,
				"attack_type": "analysis_error",
				"confidence": 0.5,
				"error": str(e)
			}
	
	async def detect_voice_presentation_attack(self, 
											  voice_segments: List[np.ndarray], 
											  metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Voice presentation attack detection for synthetic and replay attacks.
		
		Args:
			voice_segments: List of voice audio segments
			metadata: Additional metadata (sample rate, device info, etc.)
		
		Returns:
			Detection result with attack analysis
		"""
		try:
			self.logger.info(f"Starting voice PAD analysis with {len(voice_segments)} segments")
			
			if not voice_segments:
				return {
					"is_attack": True,
					"attack_probability": 0.9,
					"attack_type": "no_audio",
					"confidence": 0.9
				}
			
			detection_results = []
			
			# 1. Synthetic voice detection
			synthetic_result = await self._detect_synthetic_voice(voice_segments, metadata)
			detection_results.append(("synthetic_voice", synthetic_result))
			
			# 2. Voice replay attack detection
			replay_result = await self._detect_voice_replay(voice_segments, metadata)
			detection_results.append(("voice_replay", replay_result))
			
			# 3. Voice conversion attack detection
			conversion_result = await self._detect_voice_conversion(voice_segments, metadata)
			detection_results.append(("voice_conversion", conversion_result))
			
			# 4. Audio quality consistency
			quality_result = await self._check_audio_quality_consistency(voice_segments, metadata)
			detection_results.append(("audio_quality", quality_result))
			
			# Combine results
			combined_result = self._combine_detection_results(detection_results)
			
			self.logger.info(f"Voice PAD complete: attack={combined_result['is_attack']}, "
						   f"probability={combined_result['attack_probability']:.3f}")
			
			return combined_result
			
		except Exception as e:
			self.logger.error(f"Voice PAD error: {str(e)}", exc_info=True)
			return {
				"is_attack": True,
				"attack_probability": 0.9,
				"attack_type": "analysis_error",
				"confidence": 0.5,
				"error": str(e)
			}
	
	# Face attack detection methods
	
	async def _detect_photo_attack(self, face_frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect photo/print attacks"""
		try:
			attack_indicators = []
			
			# 1. Motion analysis - photos don't have natural micro-movements
			motion_score = self._analyze_micro_movements(face_frames)
			if motion_score < 0.3:
				attack_indicators.append(("low_motion", 0.6, "Insufficient natural movement"))
			
			# 2. Texture analysis - printed photos have different texture
			texture_scores = []
			for frame in face_frames[:5]:  # Analyze first 5 frames
				texture_score = self._analyze_texture_authenticity(frame)
				texture_scores.append(texture_score)
			
			avg_texture_score = np.mean(texture_scores) if texture_scores else 0.0
			if avg_texture_score < 0.5:
				attack_indicators.append(("artificial_texture", 0.7, "Texture indicates printed material"))
			
			# 3. Frequency domain analysis
			freq_score = self._analyze_frequency_domain(face_frames)
			if freq_score < 0.4:
				attack_indicators.append(("frequency_anomaly", 0.5, "Frequency domain anomalies detected"))
			
			# 4. Color distribution analysis
			color_score = self._analyze_color_distribution(face_frames)
			if color_score < 0.5:
				attack_indicators.append(("color_anomaly", 0.4, "Unnatural color distribution"))
			
			# Calculate overall photo attack probability
			if attack_indicators:
				attack_prob = min(sum(indicator[1] for indicator in attack_indicators) / len(attack_indicators), 1.0)
				is_attack = attack_prob >= self.photo_attack_threshold
			else:
				attack_prob = 0.1
				is_attack = False
			
			return {
				"attack_probability": attack_prob,
				"is_attack": is_attack,
				"indicators": [indicator[2] for indicator in attack_indicators],
				"scores": {
					"motion": motion_score,
					"texture": avg_texture_score,
					"frequency": freq_score,
					"color": color_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.5, "is_attack": True, "error": str(e)}
	
	async def _detect_video_replay_attack(self, face_frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect video replay attacks"""
		try:
			attack_indicators = []
			
			# 1. Temporal consistency analysis
			temporal_score = self._analyze_temporal_consistency(face_frames, metadata)
			if temporal_score < 0.6:
				attack_indicators.append(("temporal_inconsistency", 0.7, "Temporal patterns suggest replay"))
			
			# 2. Screen reflection detection
			screen_score = self._detect_screen_reflections(face_frames)
			if screen_score > 0.7:
				attack_indicators.append(("screen_reflection", 0.8, "Screen reflection patterns detected"))
			
			# 3. Video compression artifacts
			compression_score = self._detect_compression_artifacts(face_frames)
			if compression_score > 0.6:
				attack_indicators.append(("compression_artifacts", 0.6, "Video compression artifacts found"))
			
			# 4. Frame rate inconsistency
			if "frame_timestamps" in metadata:
				framerate_score = self._analyze_frame_rate_consistency(metadata["frame_timestamps"])
				if framerate_score < 0.5:
					attack_indicators.append(("framerate_inconsistency", 0.5, "Inconsistent frame timing"))
			
			# Calculate overall replay attack probability
			if attack_indicators:
				attack_prob = min(sum(indicator[1] for indicator in attack_indicators) / len(attack_indicators), 1.0)
				is_attack = attack_prob >= self.video_replay_threshold
			else:
				attack_prob = 0.1
				is_attack = False
			
			return {
				"attack_probability": attack_prob,
				"is_attack": is_attack,
				"indicators": [indicator[2] for indicator in attack_indicators],
				"scores": {
					"temporal": temporal_score,
					"screen_reflection": screen_score,
					"compression": compression_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.5, "is_attack": True, "error": str(e)}
	
	async def _detect_mask_attack(self, face_frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect 3D mask attacks"""
		try:
			attack_indicators = []
			
			# 1. Depth analysis (if depth data available)
			if metadata.get("depth_data"):
				depth_score = self._analyze_depth_patterns(metadata["depth_data"])
				if depth_score < 0.4:
					attack_indicators.append(("depth_anomaly", 0.8, "Unnatural depth patterns"))
			
			# 2. Material analysis
			material_scores = []
			for frame in face_frames[:3]:
				material_score = self._analyze_material_properties(frame)
				material_scores.append(material_score)
			
			avg_material_score = np.mean(material_scores) if material_scores else 0.5
			if avg_material_score < 0.5:
				attack_indicators.append(("artificial_material", 0.7, "Material properties suggest mask"))
			
			# 3. Facial landmark consistency
			landmark_score = self._analyze_landmark_consistency(face_frames)
			if landmark_score < 0.6:
				attack_indicators.append(("landmark_inconsistency", 0.6, "Inconsistent facial landmarks"))
			
			# 4. Eye movement analysis
			eye_score = self._analyze_eye_movements(face_frames)
			if eye_score < 0.4:
				attack_indicators.append(("unnatural_eye_movement", 0.7, "Unnatural eye movements"))
			
			# Calculate overall mask attack probability
			if attack_indicators:
				attack_prob = min(sum(indicator[1] for indicator in attack_indicators) / len(attack_indicators), 1.0)
				is_attack = attack_prob >= self.mask_attack_threshold
			else:
				attack_prob = 0.1
				is_attack = False
			
			return {
				"attack_probability": attack_prob,
				"is_attack": is_attack,
				"indicators": [indicator[2] for indicator in attack_indicators],
				"scores": {
					"material": avg_material_score,
					"landmarks": landmark_score,
					"eye_movement": eye_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.5, "is_attack": True, "error": str(e)}
	
	async def _detect_deepfake_attack(self, face_frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect deepfake/face swap attacks"""
		try:
			attack_indicators = []
			
			# 1. Temporal inconsistency in facial features
			temporal_face_score = self._analyze_facial_temporal_consistency(face_frames)
			if temporal_face_score < 0.5:
				attack_indicators.append(("facial_temporal_inconsistency", 0.8, "Inconsistent facial features across frames"))
			
			# 2. Blending artifacts analysis
			blending_score = self._detect_blending_artifacts(face_frames)
			if blending_score > 0.6:
				attack_indicators.append(("blending_artifacts", 0.9, "Face blending artifacts detected"))
			
			# 3. Pixel-level inconsistencies
			pixel_score = self._analyze_pixel_inconsistencies(face_frames)
			if pixel_score > 0.7:
				attack_indicators.append(("pixel_inconsistencies", 0.8, "Pixel-level inconsistencies found"))
			
			# 4. Physiological implausibility
			physio_score = self._check_physiological_plausibility(face_frames)
			if physio_score < 0.4:
				attack_indicators.append(("physiological_implausible", 0.9, "Physiologically implausible features"))
			
			# 5. Compression pattern analysis
			if len(face_frames) > 1:
				compression_pattern_score = self._analyze_compression_patterns(face_frames)
				if compression_pattern_score > 0.6:
					attack_indicators.append(("compression_patterns", 0.7, "Suspicious compression patterns"))
			
			# Calculate overall deepfake attack probability
			if attack_indicators:
				attack_prob = min(sum(indicator[1] for indicator in attack_indicators) / len(attack_indicators), 1.0)
				is_attack = attack_prob >= self.deepfake_threshold
			else:
				attack_prob = 0.1
				is_attack = False
			
			return {
				"attack_probability": attack_prob,
				"is_attack": is_attack,
				"indicators": [indicator[2] for indicator in attack_indicators],
				"scores": {
					"temporal_consistency": temporal_face_score,
					"blending": blending_score,
					"pixel_consistency": pixel_score,
					"physiological": physio_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.5, "is_attack": True, "error": str(e)}
	
	async def _check_environmental_consistency(self, face_frames: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Check environmental consistency (lighting, shadows, etc.)"""
		try:
			consistency_issues = []
			
			# 1. Lighting consistency
			lighting_score = self._analyze_lighting_consistency(face_frames)
			if lighting_score < 0.5:
				consistency_issues.append(("lighting_inconsistency", 0.6, "Inconsistent lighting across frames"))
			
			# 2. Shadow consistency
			shadow_score = self._analyze_shadow_consistency(face_frames)
			if shadow_score < 0.5:
				consistency_issues.append(("shadow_inconsistency", 0.5, "Inconsistent shadow patterns"))
			
			# 3. Background consistency
			background_score = self._analyze_background_consistency(face_frames)
			if background_score < 0.6:
				consistency_issues.append(("background_inconsistency", 0.4, "Background inconsistencies"))
			
			# 4. Device sensor consistency
			if metadata.get("device_sensors"):
				sensor_score = self._check_sensor_consistency(metadata["device_sensors"])
				if sensor_score < 0.7:
					consistency_issues.append(("sensor_inconsistency", 0.7, "Device sensor data inconsistent"))
			
			# Calculate environmental attack probability
			if consistency_issues:
				attack_prob = min(sum(issue[1] for issue in consistency_issues) / len(consistency_issues), 1.0)
				is_attack = attack_prob >= 0.5
			else:
				attack_prob = 0.1
				is_attack = False
			
			return {
				"attack_probability": attack_prob,
				"is_attack": is_attack,
				"indicators": [issue[2] for issue in consistency_issues],
				"scores": {
					"lighting": lighting_score,
					"shadows": shadow_score,
					"background": background_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.3, "is_attack": False, "error": str(e)}
	
	# Voice attack detection methods
	
	async def _detect_synthetic_voice(self, voice_segments: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect synthetic/TTS voice"""
		try:
			attack_indicators = []
			sample_rate = metadata.get("sample_rate", 16000)
			
			# 1. Spectral analysis for synthetic patterns
			spectral_score = self._analyze_synthetic_spectral_patterns(voice_segments, sample_rate)
			if spectral_score > 0.7:
				attack_indicators.append(("synthetic_spectral", 0.8, "Synthetic spectral patterns detected"))
			
			# 2. Prosody naturalness
			prosody_score = self._analyze_prosody_naturalness(voice_segments, sample_rate)
			if prosody_score < 0.4:
				attack_indicators.append(("unnatural_prosody", 0.7, "Unnatural prosodic patterns"))
			
			# 3. Vocal tract modeling artifacts
			vocal_tract_score = self._detect_vocal_tract_artifacts(voice_segments, sample_rate)
			if vocal_tract_score > 0.6:
				attack_indicators.append(("vocal_tract_artifacts", 0.8, "Vocal tract modeling artifacts"))
			
			# 4. Phase relationships analysis
			phase_score = self._analyze_phase_relationships(voice_segments)
			if phase_score > 0.6:
				attack_indicators.append(("phase_anomalies", 0.6, "Unnatural phase relationships"))
			
			# Calculate synthetic voice probability
			if attack_indicators:
				attack_prob = min(sum(indicator[1] for indicator in attack_indicators) / len(attack_indicators), 1.0)
			else:
				attack_prob = 0.1
			
			return {
				"attack_probability": attack_prob,
				"is_attack": attack_prob >= 0.6,
				"indicators": [indicator[2] for indicator in attack_indicators],
				"scores": {
					"spectral": spectral_score,
					"prosody": prosody_score,
					"vocal_tract": vocal_tract_score,
					"phase": phase_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.5, "is_attack": True, "error": str(e)}
	
	async def _detect_voice_replay(self, voice_segments: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect voice replay attacks"""
		try:
			attack_indicators = []
			
			# 1. Audio quality consistency
			quality_score = self._analyze_audio_quality_patterns(voice_segments)
			if quality_score > 0.7:
				attack_indicators.append(("quality_inconsistency", 0.6, "Inconsistent audio quality"))
			
			# 2. Background noise analysis
			noise_score = self._analyze_background_noise_patterns(voice_segments)
			if noise_score > 0.6:
				attack_indicators.append(("noise_patterns", 0.7, "Suspicious background noise patterns"))
			
			# 3. Dynamic range compression
			compression_score = self._detect_dynamic_range_compression(voice_segments)
			if compression_score > 0.7:
				attack_indicators.append(("compression_artifacts", 0.6, "Dynamic range compression detected"))
			
			# 4. Temporal consistency
			temporal_score = self._analyze_voice_temporal_consistency(voice_segments, metadata)
			if temporal_score < 0.5:
				attack_indicators.append(("temporal_inconsistency", 0.5, "Temporal inconsistencies"))
			
			# Calculate replay attack probability
			if attack_indicators:
				attack_prob = min(sum(indicator[1] for indicator in attack_indicators) / len(attack_indicators), 1.0)
			else:
				attack_prob = 0.1
			
			return {
				"attack_probability": attack_prob,
				"is_attack": attack_prob >= 0.6,
				"indicators": [indicator[2] for indicator in attack_indicators],
				"scores": {
					"quality": quality_score,
					"noise": noise_score,
					"compression": compression_score,
					"temporal": temporal_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.5, "is_attack": True, "error": str(e)}
	
	async def _detect_voice_conversion(self, voice_segments: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect voice conversion attacks"""
		try:
			attack_indicators = []
			sample_rate = metadata.get("sample_rate", 16000)
			
			# 1. Formant analysis
			formant_score = self._analyze_formant_consistency(voice_segments, sample_rate)
			if formant_score < 0.5:
				attack_indicators.append(("formant_inconsistency", 0.7, "Inconsistent formant patterns"))
			
			# 2. Pitch conversion artifacts
			pitch_score = self._detect_pitch_conversion_artifacts(voice_segments, sample_rate)
			if pitch_score > 0.6:
				attack_indicators.append(("pitch_artifacts", 0.8, "Pitch conversion artifacts detected"))
			
			# 3. Spectral envelope analysis
			envelope_score = self._analyze_spectral_envelope_consistency(voice_segments)
			if envelope_score < 0.5:
				attack_indicators.append(("envelope_inconsistency", 0.6, "Spectral envelope inconsistencies"))
			
			# Calculate voice conversion probability
			if attack_indicators:
				attack_prob = min(sum(indicator[1] for indicator in attack_indicators) / len(attack_indicators), 1.0)
			else:
				attack_prob = 0.1
			
			return {
				"attack_probability": attack_prob,
				"is_attack": attack_prob >= 0.6,
				"indicators": [indicator[2] for indicator in attack_indicators],
				"scores": {
					"formants": formant_score,
					"pitch": pitch_score,
					"envelope": envelope_score
				}
			}
			
		except Exception as e:
			return {"attack_probability": 0.5, "is_attack": True, "error": str(e)}
	
	async def _check_audio_quality_consistency(self, voice_segments: List[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
		"""Check audio quality consistency across segments"""
		try:
			# Simplified quality consistency check
			quality_scores = []
			
			for segment in voice_segments:
				# Calculate SNR estimate
				signal_power = np.mean(segment**2)
				noise_estimate = np.var(segment[:min(len(segment)//10, 1000)])  # First 10% as noise estimate
				snr = 10 * np.log10(signal_power / max(noise_estimate, 1e-10))
				quality_scores.append(snr)
			
			if len(quality_scores) > 1:
				quality_variance = np.var(quality_scores)
				consistency_score = max(0.0, 1.0 - quality_variance / 100)  # Normalize
			else:
				consistency_score = 0.8
			
			is_inconsistent = consistency_score < 0.5
			
			return {
				"attack_probability": 0.4 if is_inconsistent else 0.1,
				"is_attack": is_inconsistent,
				"indicators": ["Quality inconsistency across segments"] if is_inconsistent else [],
				"scores": {"consistency": consistency_score}
			}
			
		except Exception as e:
			return {"attack_probability": 0.3, "is_attack": False, "error": str(e)}
	
	# Helper methods for analysis (simplified implementations)
	
	def _analyze_micro_movements(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze micro-movements between frames"""
		if len(face_frames) < 2:
			return 0.5
		
		movement_scores = []
		for i in range(1, len(face_frames)):
			diff = cv2.absdiff(face_frames[i-1], face_frames[i])
			movement = np.mean(diff)
			movement_scores.append(movement)
		
		# Natural faces should have some micro-movement
		avg_movement = np.mean(movement_scores) if movement_scores else 0
		return min(avg_movement / 10, 1.0)  # Normalize
	
	def _analyze_texture_authenticity(self, face_frame: np.ndarray) -> float:
		"""Analyze texture for authenticity"""
		try:
			gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY) if len(face_frame.shape) == 3 else face_frame
			
			# Local Binary Pattern analysis (simplified)
			laplacian = cv2.Laplacian(gray, cv2.CV_64F)
			texture_variance = np.var(laplacian)
			
			# Real skin should have natural texture variance
			return min(texture_variance / 1000, 1.0)
			
		except Exception:
			return 0.5
	
	def _analyze_frequency_domain(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze frequency domain characteristics"""
		try:
			if not face_frames:
				return 0.5
			
			# Simplified frequency analysis
			frame = face_frames[0]
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
			
			fft = np.fft.fft2(gray)
			magnitude = np.abs(fft)
			
			# Analyze high frequency content
			high_freq_energy = np.sum(magnitude[magnitude.shape[0]//4:, magnitude.shape[1]//4:])
			total_energy = np.sum(magnitude)
			
			high_freq_ratio = high_freq_energy / max(total_energy, 1)
			return min(high_freq_ratio * 5, 1.0)
			
		except Exception:
			return 0.5
	
	def _analyze_color_distribution(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze color distribution naturalness"""
		try:
			if not face_frames:
				return 0.5
			
			frame = face_frames[0]
			if len(frame.shape) != 3:
				return 0.5
			
			# Analyze color channel distributions
			color_variances = []
			for channel in range(3):
				channel_data = frame[:, :, channel]
				variance = np.var(channel_data)
				color_variances.append(variance)
			
			# Natural skin should have balanced color distribution
			color_balance = 1.0 - np.std(color_variances) / max(np.mean(color_variances), 1)
			return max(0.0, min(color_balance, 1.0))
			
		except Exception:
			return 0.5
	
	def _analyze_temporal_consistency(self, face_frames: List[np.ndarray], metadata: Dict[str, Any]) -> float:
		"""Analyze temporal consistency between frames"""
		try:
			if len(face_frames) < 2:
				return 0.5
			
			consistency_scores = []
			for i in range(1, len(face_frames)):
				# Calculate structural similarity
				ssim = self._calculate_ssim(face_frames[i-1], face_frames[i])
				consistency_scores.append(ssim)
			
			# Video replays often have very high consistency
			avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.5
			
			# Too high consistency might indicate replay
			if avg_consistency > 0.95:
				return 0.3  # Suspicious
			elif avg_consistency < 0.2:
				return 0.2  # Too inconsistent
			else:
				return 0.8  # Natural range
				
		except Exception:
			return 0.5
	
	def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
		"""Calculate structural similarity (simplified)"""
		try:
			# Convert to grayscale if needed
			if len(img1.shape) == 3:
				img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
			if len(img2.shape) == 3:
				img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
			
			# Simplified SSIM calculation
			mean1 = np.mean(img1)
			mean2 = np.mean(img2)
			var1 = np.var(img1)
			var2 = np.var(img2)
			cov = np.mean((img1 - mean1) * (img2 - mean2))
			
			c1 = 0.01**2
			c2 = 0.03**2
			
			ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
			
			return max(0.0, min(ssim, 1.0))
			
		except Exception:
			return 0.5
	
	def _detect_screen_reflections(self, face_frames: List[np.ndarray]) -> float:
		"""Detect screen reflection patterns"""
		try:
			reflection_scores = []
			
			for frame in face_frames[:3]:  # Check first 3 frames
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
				
				# Look for rectangular patterns (screen edges)
				edges = cv2.Canny(gray, 50, 150)
				lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
				
				if lines is not None:
					# Check for parallel lines (screen edges)
					reflection_score = min(len(lines) / 20, 1.0)
				else:
					reflection_score = 0.0
				
				reflection_scores.append(reflection_score)
			
			return np.mean(reflection_scores) if reflection_scores else 0.0
			
		except Exception:
			return 0.0
	
	def _detect_compression_artifacts(self, face_frames: List[np.ndarray]) -> float:
		"""Detect video compression artifacts"""
		try:
			artifact_scores = []
			
			for frame in face_frames[:3]:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
				
				# Look for blocking artifacts
				block_artifacts = self._detect_blocking_artifacts(gray)
				artifact_scores.append(block_artifacts)
			
			return np.mean(artifact_scores) if artifact_scores else 0.0
			
		except Exception:
			return 0.0
	
	def _detect_blocking_artifacts(self, gray_image: np.ndarray) -> float:
		"""Detect blocking artifacts in image"""
		try:
			# Simplified blocking artifact detection
			h, w = gray_image.shape
			block_size = 8
			
			blocking_score = 0.0
			block_count = 0
			
			for y in range(0, h - block_size, block_size):
				for x in range(0, w - block_size, block_size):
					block = gray_image[y:y+block_size, x:x+block_size]
					
					# Check for artificial block boundaries
					top_diff = np.mean(np.abs(block[0, :] - block[1, :]))
					left_diff = np.mean(np.abs(block[:, 0] - block[:, 1]))
					
					boundary_strength = (top_diff + left_diff) / 2
					if boundary_strength > 20:  # Threshold for artifact detection
						blocking_score += 1
					
					block_count += 1
			
			return blocking_score / max(block_count, 1)
			
		except Exception:
			return 0.0
	
	def _analyze_frame_rate_consistency(self, frame_timestamps: List[float]) -> float:
		"""Analyze frame rate consistency"""
		try:
			if len(frame_timestamps) < 3:
				return 0.5
			
			intervals = []
			for i in range(1, len(frame_timestamps)):
				interval = frame_timestamps[i] - frame_timestamps[i-1]
				intervals.append(interval)
			
			# Check consistency of intervals
			interval_std = np.std(intervals)
			interval_mean = np.mean(intervals)
			
			if interval_mean > 0:
				consistency = 1.0 - (interval_std / interval_mean)
				return max(0.0, min(consistency, 1.0))
			else:
				return 0.0
				
		except Exception:
			return 0.5
	
	# Simplified implementations for other analysis methods
	def _analyze_depth_patterns(self, depth_data: np.ndarray) -> float:
		"""Analyze depth patterns for mask detection"""
		return 0.7  # Placeholder
	
	def _analyze_material_properties(self, frame: np.ndarray) -> float:
		"""Analyze material properties"""
		return 0.6  # Placeholder
	
	def _analyze_landmark_consistency(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze facial landmark consistency"""
		return 0.7  # Placeholder
	
	def _analyze_eye_movements(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze eye movement patterns"""
		return 0.6  # Placeholder
	
	def _analyze_facial_temporal_consistency(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze facial feature temporal consistency"""
		return 0.6  # Placeholder
	
	def _detect_blending_artifacts(self, face_frames: List[np.ndarray]) -> float:
		"""Detect face blending artifacts"""
		return 0.3  # Placeholder
	
	def _analyze_pixel_inconsistencies(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze pixel-level inconsistencies"""
		return 0.2  # Placeholder
	
	def _check_physiological_plausibility(self, face_frames: List[np.ndarray]) -> float:
		"""Check physiological plausibility"""
		return 0.8  # Placeholder
	
	def _analyze_compression_patterns(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze compression patterns"""
		return 0.3  # Placeholder
	
	def _analyze_lighting_consistency(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze lighting consistency"""
		return 0.7  # Placeholder
	
	def _analyze_shadow_consistency(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze shadow consistency"""
		return 0.7  # Placeholder
	
	def _analyze_background_consistency(self, face_frames: List[np.ndarray]) -> float:
		"""Analyze background consistency"""
		return 0.8  # Placeholder
	
	def _check_sensor_consistency(self, sensor_data: Dict[str, Any]) -> float:
		"""Check device sensor consistency"""
		return 0.8  # Placeholder
	
	# Voice analysis placeholders
	def _analyze_synthetic_spectral_patterns(self, voice_segments: List[np.ndarray], sample_rate: int) -> float:
		return 0.3  # Placeholder
	
	def _analyze_prosody_naturalness(self, voice_segments: List[np.ndarray], sample_rate: int) -> float:
		return 0.7  # Placeholder
	
	def _detect_vocal_tract_artifacts(self, voice_segments: List[np.ndarray], sample_rate: int) -> float:
		return 0.2  # Placeholder
	
	def _analyze_phase_relationships(self, voice_segments: List[np.ndarray]) -> float:
		return 0.3  # Placeholder
	
	def _analyze_audio_quality_patterns(self, voice_segments: List[np.ndarray]) -> float:
		return 0.4  # Placeholder
	
	def _analyze_background_noise_patterns(self, voice_segments: List[np.ndarray]) -> float:
		return 0.3  # Placeholder
	
	def _detect_dynamic_range_compression(self, voice_segments: List[np.ndarray]) -> float:
		return 0.4  # Placeholder
	
	def _analyze_voice_temporal_consistency(self, voice_segments: List[np.ndarray], metadata: Dict[str, Any]) -> float:
		return 0.6  # Placeholder
	
	def _analyze_formant_consistency(self, voice_segments: List[np.ndarray], sample_rate: int) -> float:
		return 0.7  # Placeholder
	
	def _detect_pitch_conversion_artifacts(self, voice_segments: List[np.ndarray], sample_rate: int) -> float:
		return 0.3  # Placeholder
	
	def _analyze_spectral_envelope_consistency(self, voice_segments: List[np.ndarray]) -> float:
		return 0.7  # Placeholder
	
	def _combine_detection_results(self, detection_results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
		"""Combine multiple detection results into final decision"""
		try:
			attack_probabilities = []
			attack_indicators = []
			all_scores = {}
			
			for detection_type, result in detection_results:
				if result.get("is_attack"):
					attack_probabilities.append(result.get("attack_probability", 0.5))
					if result.get("indicators"):
						attack_indicators.extend(result["indicators"])
				
				# Collect scores
				if "scores" in result:
					for score_name, score_value in result["scores"].items():
						all_scores[f"{detection_type}_{score_name}"] = score_value
			
			# Calculate overall attack probability
			if attack_probabilities:
				# Use maximum probability (most conservative)
				overall_attack_prob = max(attack_probabilities)
				is_attack = overall_attack_prob >= 0.6
			else:
				overall_attack_prob = 0.1
				is_attack = False
			
			# Determine primary attack type
			if attack_probabilities:
				max_prob_index = attack_probabilities.index(max(attack_probabilities))
				primary_attack_type = detection_results[max_prob_index][0]
			else:
				primary_attack_type = "none"
			
			# Calculate confidence based on number of detections
			confidence = min(len(detection_results) / 5.0, 1.0)
			
			return {
				"is_attack": is_attack,
				"attack_probability": overall_attack_prob,
				"attack_type": primary_attack_type,
				"confidence": confidence,
				"indicators": list(set(attack_indicators)),  # Remove duplicates
				"detection_scores": all_scores,
				"detections_performed": len(detection_results)
			}
			
		except Exception as e:
			return {
				"is_attack": True,
				"attack_probability": 0.9,
				"attack_type": "analysis_error",
				"confidence": 0.5,
				"error": str(e)
			}


class SecurityHardeningService:
	"""Security hardening and additional protection measures"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		
		# Security policies
		self.max_failed_attempts = 5
		self.lockout_duration_minutes = 30
		self.suspicious_activity_threshold = 0.7
	
	async def validate_device_security(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate device security posture"""
		try:
			security_issues = []
			security_score = 1.0
			
			# Check for rooted/jailbroken device
			if device_info.get("is_rooted") or device_info.get("is_jailbroken"):
				security_issues.append("Device is rooted/jailbroken")
				security_score -= 0.4
			
			# Check app integrity
			if not device_info.get("app_integrity_verified", True):
				security_issues.append("App integrity verification failed")
				security_score -= 0.3
			
			# Check for debugging tools
			if device_info.get("debugging_enabled"):
				security_issues.append("Debugging tools detected")
				security_score -= 0.2
			
			# Check for emulator
			if device_info.get("is_emulator"):
				security_issues.append("Emulator detected")
				security_score -= 0.5
			
			# Check OS version
			if device_info.get("os_outdated"):
				security_issues.append("Outdated operating system")
				security_score -= 0.1
			
			security_score = max(0.0, security_score)
			is_secure = security_score >= 0.6 and len(security_issues) == 0
			
			return {
				"is_secure": is_secure,
				"security_score": security_score,
				"security_issues": security_issues,
				"recommendations": self._generate_security_recommendations(security_issues)
			}
			
		except Exception as e:
			self.logger.error(f"Device security validation error: {str(e)}")
			return {
				"is_secure": False,
				"security_score": 0.0,
				"security_issues": ["Security validation failed"],
				"error": str(e)
			}
	
	async def detect_automation_tools(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect automation tools and bots"""
		try:
			automation_indicators = []
			automation_score = 0.0
			
			# Check mouse movement patterns
			if "mouse_movements" in interaction_data:
				mouse_naturalness = self._analyze_mouse_naturalness(interaction_data["mouse_movements"])
				if mouse_naturalness < 0.3:
					automation_indicators.append("Unnatural mouse movements")
					automation_score += 0.4
			
			# Check timing patterns
			if "interaction_timings" in interaction_data:
				timing_naturalness = self._analyze_timing_naturalness(interaction_data["interaction_timings"])
				if timing_naturalness < 0.3:
					automation_indicators.append("Robotic timing patterns")
					automation_score += 0.3
			
			# Check for automation tool signatures
			if "user_agent" in interaction_data:
				if self._detect_automation_signatures(interaction_data["user_agent"]):
					automation_indicators.append("Automation tool detected in user agent")
					automation_score += 0.6
			
			# Check for headless browser indicators
			if interaction_data.get("headless_indicators"):
				automation_indicators.append("Headless browser detected")
				automation_score += 0.7
			
			is_automation = automation_score >= 0.5
			
			return {
				"is_automation": is_automation,
				"automation_score": min(automation_score, 1.0),
				"indicators": automation_indicators,
				"confidence": 0.8 if automation_indicators else 0.3
			}
			
		except Exception as e:
			self.logger.error(f"Automation detection error: {str(e)}")
			return {
				"is_automation": False,
				"automation_score": 0.0,
				"indicators": [],
				"error": str(e)
			}
	
	async def validate_session_integrity(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate session integrity and detect session attacks"""
		try:
			integrity_issues = []
			integrity_score = 1.0
			
			# Check session token integrity
			if not self._validate_session_token(session_data.get("session_token", "")):
				integrity_issues.append("Invalid session token")
				integrity_score -= 0.5
			
			# Check for session hijacking indicators
			if self._detect_session_hijacking(session_data):
				integrity_issues.append("Potential session hijacking")
				integrity_score -= 0.7
			
			# Check for concurrent sessions
			if session_data.get("concurrent_sessions", 0) > 5:
				integrity_issues.append("Excessive concurrent sessions")
				integrity_score -= 0.3
			
			# Check IP consistency
			if not self._check_ip_consistency(session_data):
				integrity_issues.append("IP address inconsistency")
				integrity_score -= 0.4
			
			integrity_score = max(0.0, integrity_score)
			is_valid = integrity_score >= 0.7 and len(integrity_issues) == 0
			
			return {
				"is_valid": is_valid,
				"integrity_score": integrity_score,
				"integrity_issues": integrity_issues,
				"recommendations": ["Regenerate session", "Verify user identity"] if not is_valid else []
			}
			
		except Exception as e:
			self.logger.error(f"Session integrity validation error: {str(e)}")
			return {
				"is_valid": False,
				"integrity_score": 0.0,
				"integrity_issues": ["Session validation failed"],
				"error": str(e)
			}
	
	# Helper methods
	
	def _generate_security_recommendations(self, security_issues: List[str]) -> List[str]:
		"""Generate security recommendations based on issues"""
		recommendations = []
		
		for issue in security_issues:
			if "rooted" in issue.lower() or "jailbroken" in issue.lower():
				recommendations.append("Use device with unmodified OS")
			elif "integrity" in issue.lower():
				recommendations.append("Reinstall application from official source")
			elif "debugging" in issue.lower():
				recommendations.append("Disable debugging tools")
			elif "emulator" in issue.lower():
				recommendations.append("Use physical device")
			elif "outdated" in issue.lower():
				recommendations.append("Update operating system")
		
		return list(set(recommendations))  # Remove duplicates
	
	def _analyze_mouse_naturalness(self, mouse_movements: List[Dict[str, Any]]) -> float:
		"""Analyze mouse movement naturalness"""
		if len(mouse_movements) < 2:
			return 0.5
		
		# Check for perfectly straight lines (bot indicator)
		straight_line_count = 0
		total_movements = 0
		
		for i in range(1, len(mouse_movements)):
			prev = mouse_movements[i-1]
			curr = mouse_movements[i]
			
			dx = curr["x"] - prev["x"]
			dy = curr["y"] - prev["y"]
			
			if dx != 0 and dy != 0:
				# Check if movement is perfectly diagonal
				if abs(abs(dx) - abs(dy)) < 1:  # Perfect diagonal
					straight_line_count += 1
				total_movements += 1
		
		if total_movements == 0:
			return 0.5
		
		straight_line_ratio = straight_line_count / total_movements
		
		# Natural mouse movement has some randomness
		naturalness = 1.0 - straight_line_ratio
		return max(0.0, min(naturalness, 1.0))
	
	def _analyze_timing_naturalness(self, interaction_timings: List[float]) -> float:
		"""Analyze timing pattern naturalness"""
		if len(interaction_timings) < 3:
			return 0.5
		
		intervals = []
		for i in range(1, len(interaction_timings)):
			interval = interaction_timings[i] - interaction_timings[i-1]
			intervals.append(interval)
		
		# Check for too-regular timing (bot indicator)
		interval_std = np.std(intervals)
		interval_mean = np.mean(intervals)
		
		if interval_mean > 0:
			coefficient_of_variation = interval_std / interval_mean
			# Natural human timing has variation
			naturalness = min(coefficient_of_variation * 2, 1.0)
		else:
			naturalness = 0.0
		
		return max(0.0, min(naturalness, 1.0))
	
	def _detect_automation_signatures(self, user_agent: str) -> bool:
		"""Detect automation tool signatures in user agent"""
		automation_keywords = [
			"selenium", "webdriver", "phantomjs", "headless",
			"automation", "bot", "crawler", "scraper"
		]
		
		user_agent_lower = user_agent.lower()
		return any(keyword in user_agent_lower for keyword in automation_keywords)
	
	def _validate_session_token(self, session_token: str) -> bool:
		"""Validate session token format and integrity"""
		if not session_token or len(session_token) < 32:
			return False
		
		# Basic validation (in production, would use proper cryptographic validation)
		try:
			# Check if token is base64-like
			base64.b64decode(session_token + "==")  # Add padding
			return True
		except Exception:
			return False
	
	def _detect_session_hijacking(self, session_data: Dict[str, Any]) -> bool:
		"""Detect potential session hijacking"""
		# Check for rapid IP changes
		if "ip_history" in session_data:
			ip_history = session_data["ip_history"]
			if len(ip_history) > 1:
				# Check if IPs changed too quickly
				time_diffs = []
				for i in range(1, len(ip_history)):
					time_diff = ip_history[i]["timestamp"] - ip_history[i-1]["timestamp"]
					time_diffs.append(time_diff)
				
				if any(diff < 60 for diff in time_diffs):  # Less than 1 minute
					return True
		
		# Check for impossible geographic locations
		if "location_history" in session_data:
			# Simplified check - would use proper geolocation validation
			locations = session_data["location_history"]
			if len(locations) > 1:
				# Check for impossible travel times
				return False  # Placeholder
		
		return False
	
	def _check_ip_consistency(self, session_data: Dict[str, Any]) -> bool:
		"""Check IP address consistency within session"""
		if "current_ip" in session_data and "session_start_ip" in session_data:
			return session_data["current_ip"] == session_data["session_start_ip"]
		
		return True  # Assume consistent if no data


__all__ = [
	"PresentationAttackDetector",
	"SecurityHardeningService"
]