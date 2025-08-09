"""
Biometric Fusion Engine

Multi-modal biometric authentication system with liveness detection,
template protection, and fusion algorithms for ultimate security and convenience.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import secrets
import json
import math
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass
import cv2
import base64

from .enhanced_models import BiometricType, BiometricTemplate

class BiometricQuality(str, Enum):
	"""Biometric sample quality levels"""
	POOR = "poor"           # Quality < 0.3
	FAIR = "fair"           # Quality 0.3-0.5
	GOOD = "good"           # Quality 0.5-0.7
	EXCELLENT = "excellent" # Quality > 0.7

class FusionMethod(str, Enum):
	"""Biometric fusion methods"""
	SCORE_LEVEL = "score_level"        # Fusion at matching score level
	FEATURE_LEVEL = "feature_level"    # Fusion at feature level
	DECISION_LEVEL = "decision_level"  # Fusion at decision level
	HYBRID = "hybrid"                  # Combination of multiple levels

class LivenessStatus(str, Enum):
	"""Liveness detection results"""
	LIVE = "live"           # Living person detected
	SPOOF = "spoof"         # Spoofing attempt detected
	UNCERTAIN = "uncertain" # Cannot determine conclusively
	ERROR = "error"         # Detection failed

class BiometricSample(BaseModel):
	"""Individual biometric sample"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Sample identifier")
	user_id: str = Field(..., description="User identifier")
	biometric_type: BiometricType = Field(..., description="Type of biometric")
	
	# Sample data (encrypted/protected)
	sample_data: bytes = Field(..., description="Protected biometric sample data")
	feature_vector: Optional[List[float]] = Field(default=None, description="Extracted features")
	
	# Quality assessment
	quality_score: float = Field(..., description="Sample quality (0.0-1.0)", ge=0.0, le=1.0)
	quality_level: BiometricQuality = Field(..., description="Quality classification")
	
	# Acquisition metadata
	captured_at: datetime = Field(default_factory=datetime.utcnow, description="Capture timestamp")
	device_info: Dict[str, str] = Field(default_factory=dict, description="Capture device info")
	environment_conditions: Dict[str, Any] = Field(default_factory=dict, description="Environmental conditions")
	
	# Liveness detection
	liveness_status: LivenessStatus = Field(..., description="Liveness detection result")
	liveness_confidence: float = Field(..., description="Liveness confidence", ge=0.0, le=1.0)
	
	# Processing flags
	is_processed: bool = Field(default=False, description="Sample has been processed")
	processing_errors: List[str] = Field(default_factory=list, description="Processing errors")

class BiometricMatchResult(BaseModel):
	"""Result of biometric matching"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Match result identifier")
	template_id: str = Field(..., description="Template matched against")
	sample_id: str = Field(..., description="Sample that was matched")
	
	# Matching scores
	similarity_score: float = Field(..., description="Similarity score (0.0-1.0)", ge=0.0, le=1.0)
	confidence_score: float = Field(..., description="Confidence in match", ge=0.0, le=1.0)
	
	# Decision
	is_match: bool = Field(..., description="Match decision")
	threshold_used: float = Field(..., description="Threshold used for decision")
	
	# Metadata
	algorithm_used: str = Field(..., description="Matching algorithm used")
	processing_time_ms: float = Field(..., description="Processing time in milliseconds")
	matched_at: datetime = Field(default_factory=datetime.utcnow, description="Match timestamp")

class FusionResult(BaseModel):
	"""Result of multi-modal biometric fusion"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Fusion result identifier")
	user_id: str = Field(..., description="User identifier")
	
	# Input modalities
	individual_results: List[BiometricMatchResult] = Field(
		default_factory=list, description="Individual modality results"
	)
	modalities_used: List[BiometricType] = Field(
		default_factory=list, description="Biometric modalities used"
	)
	
	# Fusion scores
	fused_score: float = Field(..., description="Fused similarity score", ge=0.0, le=1.0)
	fusion_confidence: float = Field(..., description="Confidence in fusion", ge=0.0, le=1.0)
	
	# Decision
	authentication_decision: bool = Field(..., description="Final authentication decision")
	fusion_method: FusionMethod = Field(..., description="Fusion method used")
	
	# Quality metrics
	overall_quality: float = Field(..., description="Overall sample quality", ge=0.0, le=1.0)
	reliability_score: float = Field(..., description="Result reliability", ge=0.0, le=1.0)
	
	# Metadata
	processed_at: datetime = Field(default_factory=datetime.utcnow, description="Fusion timestamp")
	processing_time_ms: float = Field(..., description="Total processing time")
	
	# Security indicators
	spoofing_detected: bool = Field(default=False, description="Spoofing attempt detected")
	liveness_passed: bool = Field(default=True, description="All liveness checks passed")
	anomalies_detected: List[str] = Field(default_factory=list, description="Detected anomalies")

@dataclass
class BiometricProcessor:
	"""Individual biometric modality processor"""
	
	@staticmethod
	async def extract_face_features(image_data: bytes) -> Tuple[List[float], float]:
		"""Extract facial features and assess quality"""
		# Mock implementation - in production would use advanced face recognition
		# Convert image data to feature vector
		image_hash = hashlib.sha256(image_data).digest()
		
		# Generate mock feature vector (128-dimensional)
		feature_vector = []
		for i in range(128):
			# Use hash to generate deterministic features
			feature_val = (image_hash[i % 32] + i) / 255.0
			feature_vector.append(feature_val)
		
		# Mock quality assessment
		# In reality, would analyze image sharpness, lighting, pose, etc.
		quality_factors = [
			len(image_data) / 100000,  # File size indicator
			sum(feature_vector) / len(feature_vector),  # Feature distribution
			1.0 - (abs(feature_vector[0] - 0.5) * 2)  # Normalized first feature
		]
		
		quality_score = min(1.0, max(0.0, np.mean(quality_factors)))
		
		return feature_vector, quality_score
	
	@staticmethod
	async def extract_fingerprint_features(image_data: bytes) -> Tuple[List[float], float]:
		"""Extract fingerprint minutiae and assess quality"""
		# Mock implementation - in production would use minutiae extraction
		image_hash = hashlib.sha256(image_data).digest()
		
		# Generate mock minutiae-based feature vector
		feature_vector = []
		for i in range(0, 64, 2):  # 32 minutiae points with x,y coordinates
			x_coord = (image_hash[i % 32] / 255.0) * 512  # Normalized to image width
			y_coord = (image_hash[(i + 1) % 32] / 255.0) * 512  # Normalized to image height
			feature_vector.extend([x_coord, y_coord])
		
		# Mock quality based on minutiae distribution
		x_coords = feature_vector[::2]
		y_coords = feature_vector[1::2]
		
		# Quality factors: distribution uniformity, edge avoidance
		x_std = np.std(x_coords) / 256  # Normalized standard deviation
		y_std = np.std(y_coords) / 256
		edge_penalty = sum(1 for x in x_coords if x < 50 or x > 462) / len(x_coords)
		
		quality_score = min(1.0, max(0.0, (x_std + y_std) / 2 - edge_penalty))
		
		return feature_vector, quality_score
	
	@staticmethod
	async def extract_voice_features(audio_data: bytes) -> Tuple[List[float], float]:
		"""Extract voice features (MFCC, etc.) and assess quality"""
		# Mock implementation - in production would use audio processing libraries
		audio_hash = hashlib.sha256(audio_data).digest()
		
		# Generate mock MFCC-like features (39-dimensional)
		feature_vector = []
		for i in range(39):
			# Simulate MFCC coefficients
			mfcc_val = (audio_hash[i % 32] - 128) / 128.0  # Normalized to [-1, 1]
			feature_vector.append(mfcc_val)
		
		# Mock quality assessment based on signal characteristics
		signal_energy = sum(abs(val) for val in feature_vector) / len(feature_vector)
		frequency_spread = np.std(feature_vector)
		
		# Quality factors: energy level, frequency distribution
		quality_score = min(1.0, max(0.0, signal_energy * frequency_spread * 2))
		
		return feature_vector, quality_score
	
	@staticmethod
	async def extract_iris_features(image_data: bytes) -> Tuple[List[float], float]:
		"""Extract iris features and assess quality"""
		# Mock implementation - in production would use iris recognition algorithms
		image_hash = hashlib.sha256(image_data).digest()
		
		# Generate mock iris code (256-bit binary code converted to float)
		feature_vector = []
		for i in range(32):  # 32 bytes = 256 bits
			byte_val = image_hash[i]
			# Convert byte to 8 binary features
			for bit in range(8):
				bit_val = float((byte_val >> bit) & 1)
				feature_vector.append(bit_val)
		
		# Mock quality based on iris visibility and focus
		# In reality, would analyze pupil dilation, eyelid occlusion, focus, etc.
		center_features = feature_vector[120:136]  # Central iris region
		peripheral_features = feature_vector[:16] + feature_vector[-16:]  # Outer regions
		
		center_complexity = np.std(center_features)
		peripheral_contrast = abs(np.mean(peripheral_features) - np.mean(center_features))
		
		quality_score = min(1.0, max(0.0, center_complexity + peripheral_contrast))
		
		return feature_vector, quality_score

class LivenessDetector:
	"""Liveness detection for anti-spoofing"""
	
	def __init__(self):
		self._detection_methods = {
			BiometricType.FACE: self._detect_face_liveness,
			BiometricType.FINGERPRINT: self._detect_fingerprint_liveness,
			BiometricType.VOICE: self._detect_voice_liveness,
			BiometricType.IRIS: self._detect_iris_liveness
		}
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[LivenessDetector INFO] {message} {kwargs if kwargs else ''}")
	
	async def detect_liveness(self, biometric_type: BiometricType, 
							  sample_data: bytes,
							  metadata: Dict[str, Any]) -> Tuple[LivenessStatus, float]:
		"""Detect liveness for given biometric sample"""
		if biometric_type not in self._detection_methods:
			return LivenessStatus.UNCERTAIN, 0.5
		
		self._log_info("Detecting liveness", biometric_type=biometric_type.value)
		
		try:
			detection_method = self._detection_methods[biometric_type]
			status, confidence = await detection_method(sample_data, metadata)
			
			self._log_info("Liveness detection complete",
						   biometric_type=biometric_type.value,
						   status=status.value,
						   confidence=confidence)
			
			return status, confidence
			
		except Exception as e:
			self._log_info("Liveness detection failed",
						   biometric_type=biometric_type.value,
						   error=str(e))
			return LivenessStatus.ERROR, 0.0
	
	async def _detect_face_liveness(self, image_data: bytes, 
									metadata: Dict[str, Any]) -> Tuple[LivenessStatus, float]:
		"""Detect face liveness using multiple techniques"""
		liveness_indicators = []
		
		# Mock implementation - in production would use advanced techniques:
		# - Blink detection
		# - Eye movement tracking
		# - 3D depth analysis
		# - Texture analysis
		# - Challenge-response (smile, turn head)
		
		# Simulate texture analysis
		image_hash = hashlib.sha256(image_data).digest()
		texture_complexity = sum(image_hash[:16]) / (16 * 255)
		liveness_indicators.append(texture_complexity)
		
		# Simulate depth information (would come from camera)
		has_depth_info = metadata.get('depth_available', False)
		if has_depth_info:
			depth_consistency = metadata.get('depth_consistency', 0.8)
			liveness_indicators.append(depth_consistency)
		
		# Simulate motion detection
		motion_detected = metadata.get('motion_detected', True)
		if motion_detected:
			motion_score = metadata.get('motion_naturalness', 0.7)
			liveness_indicators.append(motion_score)
		else:
			liveness_indicators.append(0.3)  # Static images are suspicious
		
		# Combine indicators
		overall_liveness = np.mean(liveness_indicators)
		
		# Classify liveness
		if overall_liveness > 0.8:
			return LivenessStatus.LIVE, overall_liveness
		elif overall_liveness < 0.4:
			return LivenessStatus.SPOOF, 1.0 - overall_liveness
		else:
			return LivenessStatus.UNCERTAIN, 0.5
	
	async def _detect_fingerprint_liveness(self, image_data: bytes,
										   metadata: Dict[str, Any]) -> Tuple[LivenessStatus, float]:
		"""Detect fingerprint liveness"""
		liveness_indicators = []
		
		# Mock implementation - in production would analyze:
		# - Pulse detection
		# - Temperature variation
		# - Capacitive response
		# - Ridge flow analysis
		# - Pressure variations
		
		# Simulate pulse detection
		pulse_detected = metadata.get('pulse_detected', True)
		if pulse_detected:
			pulse_regularity = metadata.get('pulse_regularity', 0.8)
			liveness_indicators.append(pulse_regularity)
		else:
			liveness_indicators.append(0.2)
		
		# Simulate capacitive response
		capacitive_response = metadata.get('capacitive_response', 0.75)
		liveness_indicators.append(capacitive_response)
		
		# Simulate temperature
		temperature_appropriate = metadata.get('temperature_normal', True)
		liveness_indicators.append(0.9 if temperature_appropriate else 0.3)
		
		overall_liveness = np.mean(liveness_indicators)
		
		if overall_liveness > 0.7:
			return LivenessStatus.LIVE, overall_liveness
		elif overall_liveness < 0.4:
			return LivenessStatus.SPOOF, 1.0 - overall_liveness
		else:
			return LivenessStatus.UNCERTAIN, 0.5
	
	async def _detect_voice_liveness(self, audio_data: bytes,
									 metadata: Dict[str, Any]) -> Tuple[LivenessStatus, float]:
		"""Detect voice liveness"""
		liveness_indicators = []
		
		# Mock implementation - in production would analyze:
		# - Spectral characteristics
		# - Breathing patterns
		# - Voice naturalness
		# - Challenge-response
		# - Anti-replay detection
		
		# Simulate spectral analysis
		audio_hash = hashlib.sha256(audio_data).digest()
		spectral_naturalness = (audio_hash[0] + audio_hash[1]) / (2 * 255)
		liveness_indicators.append(spectral_naturalness)
		
		# Simulate breathing detection
		breathing_detected = metadata.get('breathing_detected', True)
		liveness_indicators.append(0.8 if breathing_detected else 0.2)
		
		# Simulate anti-replay
		is_live_recording = metadata.get('live_recording', True)
		liveness_indicators.append(0.9 if is_live_recording else 0.1)
		
		overall_liveness = np.mean(liveness_indicators)
		
		if overall_liveness > 0.7:
			return LivenessStatus.LIVE, overall_liveness
		elif overall_liveness < 0.4:
			return LivenessStatus.SPOOF, 1.0 - overall_liveness
		else:
			return LivenessStatus.UNCERTAIN, 0.5
	
	async def _detect_iris_liveness(self, image_data: bytes,
									metadata: Dict[str, Any]) -> Tuple[LivenessStatus, float]:
		"""Detect iris liveness"""
		liveness_indicators = []
		
		# Mock implementation - in production would analyze:
		# - Pupil response to light
		# - Eye movement
		# - Corneal reflection
		# - Focus variation
		
		# Simulate pupil response
		pupil_response = metadata.get('pupil_response_normal', True)
		liveness_indicators.append(0.85 if pupil_response else 0.3)
		
		# Simulate corneal reflection
		reflection_natural = metadata.get('reflection_natural', True)
		liveness_indicators.append(0.8 if reflection_natural else 0.2)
		
		# Simulate eye movement
		eye_movement_detected = metadata.get('eye_movement', False)
		liveness_indicators.append(0.7 if eye_movement_detected else 0.4)
		
		overall_liveness = np.mean(liveness_indicators)
		
		if overall_liveness > 0.7:
			return LivenessStatus.LIVE, overall_liveness
		elif overall_liveness < 0.4:
			return LivenessStatus.SPOOF, 1.0 - overall_liveness
		else:
			return LivenessStatus.UNCERTAIN, 0.5

class BiometricFusionEngine:
	"""Main biometric fusion engine"""
	
	def __init__(self):
		self.processor = BiometricProcessor()
		self.liveness_detector = LivenessDetector()
		
		# Stored templates
		self._templates: Dict[str, BiometricTemplate] = {}
		
		# Fusion configuration
		self._fusion_weights = {
			BiometricType.FACE: 0.35,
			BiometricType.FINGERPRINT: 0.30,
			BiometricType.VOICE: 0.20,
			BiometricType.IRIS: 0.25,
			BiometricType.BEHAVIORAL: 0.15
		}
		
		self._quality_thresholds = {
			BiometricType.FACE: 0.5,
			BiometricType.FINGERPRINT: 0.6,
			BiometricType.VOICE: 0.4,
			BiometricType.IRIS: 0.7
		}
		
		# Performance tracking
		self._operation_times: Dict[str, List[float]] = {}
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[BiometricFusion INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[BiometricFusion WARNING] {message} {kwargs if kwargs else ''}")
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[BiometricFusion ERROR] {message} {kwargs if kwargs else ''}")
	
	async def _time_operation(self, operation_name: str, operation_func):
		"""Time operations for performance monitoring"""
		start_time = asyncio.get_event_loop().time()
		result = await operation_func()
		end_time = asyncio.get_event_loop().time()
		
		duration_ms = (end_time - start_time) * 1000
		
		if operation_name not in self._operation_times:
			self._operation_times[operation_name] = []
		self._operation_times[operation_name].append(duration_ms)
		
		# Keep only last 100 measurements
		self._operation_times[operation_name] = self._operation_times[operation_name][-100:]
		
		return result, duration_ms
	
	async def process_biometric_sample(self, user_id: str, biometric_type: BiometricType,
									   sample_data: bytes, metadata: Dict[str, Any]) -> BiometricSample:
		"""Process individual biometric sample"""
		assert user_id, "User ID is required"
		assert sample_data, "Sample data is required"
		
		self._log_info("Processing biometric sample",
					   user_id=user_id,
					   biometric_type=biometric_type.value,
					   data_size=len(sample_data))
		
		# Detect liveness
		liveness_status, liveness_confidence = await self.liveness_detector.detect_liveness(
			biometric_type, sample_data, metadata
		)
		
		# Extract features based on biometric type
		if biometric_type == BiometricType.FACE:
			feature_vector, quality_score = await self.processor.extract_face_features(sample_data)
		elif biometric_type == BiometricType.FINGERPRINT:
			feature_vector, quality_score = await self.processor.extract_fingerprint_features(sample_data)
		elif biometric_type == BiometricType.VOICE:
			feature_vector, quality_score = await self.processor.extract_voice_features(sample_data)
		elif biometric_type == BiometricType.IRIS:
			feature_vector, quality_score = await self.processor.extract_iris_features(sample_data)
		else:
			self._log_warning("Unsupported biometric type", biometric_type=biometric_type.value)
			feature_vector, quality_score = [], 0.5
		
		# Determine quality level
		if quality_score >= 0.7:
			quality_level = BiometricQuality.EXCELLENT
		elif quality_score >= 0.5:
			quality_level = BiometricQuality.GOOD
		elif quality_score >= 0.3:
			quality_level = BiometricQuality.FAIR
		else:
			quality_level = BiometricQuality.POOR
		
		# Encrypt sample data for storage
		encrypted_sample = self._encrypt_sample_data(sample_data, user_id)
		
		# Create sample object
		sample = BiometricSample(
			user_id=user_id,
			biometric_type=biometric_type,
			sample_data=encrypted_sample,
			feature_vector=feature_vector,
			quality_score=quality_score,
			quality_level=quality_level,
			device_info=metadata.get('device_info', {}),
			environment_conditions=metadata.get('environment', {}),
			liveness_status=liveness_status,
			liveness_confidence=liveness_confidence,
			is_processed=True
		)
		
		# Check for processing errors
		processing_errors = []
		if liveness_status == LivenessStatus.ERROR:
			processing_errors.append("Liveness detection failed")
		if quality_score < self._quality_thresholds.get(biometric_type, 0.5):
			processing_errors.append(f"Quality below threshold ({quality_score:.2f})")
		if not feature_vector:
			processing_errors.append("Feature extraction failed")
		
		sample.processing_errors = processing_errors
		
		self._log_info("Biometric sample processed",
					   sample_id=sample.id,
					   quality_level=quality_level.value,
					   liveness_status=liveness_status.value,
					   feature_count=len(feature_vector),
					   errors=len(processing_errors))
		
		return sample
	
	def _encrypt_sample_data(self, sample_data: bytes, user_id: str) -> bytes:
		"""Encrypt biometric sample data for secure storage"""
		# Simple encryption - in production would use proper biometric template protection
		user_hash = hashlib.sha256(user_id.encode()).digest()
		
		# XOR with user-specific key (simplified)
		encrypted = bytearray()
		for i, byte in enumerate(sample_data):
			key_byte = user_hash[i % len(user_hash)]
			encrypted.append(byte ^ key_byte)
		
		return bytes(encrypted)
	
	def _decrypt_sample_data(self, encrypted_data: bytes, user_id: str) -> bytes:
		"""Decrypt biometric sample data"""
		# Reverse of encryption
		user_hash = hashlib.sha256(user_id.encode()).digest()
		
		decrypted = bytearray()
		for i, byte in enumerate(encrypted_data):
			key_byte = user_hash[i % len(user_hash)]
			decrypted.append(byte ^ key_byte)
		
		return bytes(decrypted)
	
	async def match_biometric(self, sample: BiometricSample, template: BiometricTemplate,
							  threshold: float = 0.7) -> BiometricMatchResult:
		"""Match biometric sample against template"""
		assert sample.biometric_type == template.biometric_type, "Biometric types must match"
		
		self._log_info("Matching biometric sample",
					   sample_id=sample.id,
					   template_id=template.id,
					   biometric_type=sample.biometric_type.value)
		
		start_time = asyncio.get_event_loop().time()
		
		try:
			# Calculate similarity score based on feature vectors
			if not sample.feature_vector:
				raise ValueError("Sample has no feature vector")
			
			# Mock similarity calculation - in production would use proper matching algorithms
			sample_features = np.array(sample.feature_vector)
			
			# For demo, create template features from template hash
			template_hash = hashlib.sha256(template.template_hash.encode()).digest()
			template_features = np.array([
				(template_hash[i % len(template_hash)] / 255.0) 
				for i in range(len(sample.feature_vector))
			])
			
			# Calculate normalized correlation coefficient
			if len(sample_features) == len(template_features):
				correlation = np.corrcoef(sample_features, template_features)[0, 1]
				similarity_score = max(0.0, min(1.0, (correlation + 1) / 2))  # Normalize to [0,1]
			else:
				# Euclidean distance normalized
				min_len = min(len(sample_features), len(template_features))
				sample_subset = sample_features[:min_len]
				template_subset = template_features[:min_len]
				
				distance = np.linalg.norm(sample_subset - template_subset)
				max_distance = np.sqrt(min_len * 2)  # Maximum possible distance
				similarity_score = max(0.0, 1.0 - (distance / max_distance))
			
			# Adjust similarity based on quality scores
			quality_factor = (sample.quality_score + template.quality_score) / 2
			adjusted_similarity = similarity_score * quality_factor
			
			# Calculate confidence based on quality and conditions
			confidence_factors = [
				quality_factor,
				sample.liveness_confidence,
				1.0 - (len(sample.processing_errors) * 0.1)  # Reduce confidence for errors
			]
			confidence_score = max(0.0, min(1.0, np.mean(confidence_factors)))
			
			# Make decision
			is_match = adjusted_similarity >= threshold
			
			end_time = asyncio.get_event_loop().time()
			processing_time = (end_time - start_time) * 1000
			
			match_result = BiometricMatchResult(
				template_id=template.id,
				sample_id=sample.id,
				similarity_score=adjusted_similarity,
				confidence_score=confidence_score,
				is_match=is_match,
				threshold_used=threshold,
				algorithm_used=f"{sample.biometric_type.value}_correlation",
				processing_time_ms=processing_time
			)
			
			self._log_info("Biometric match complete",
						   match_id=match_result.id,
						   similarity=adjusted_similarity,
						   is_match=is_match,
						   processing_time_ms=processing_time)
			
			return match_result
			
		except Exception as e:
			end_time = asyncio.get_event_loop().time()
			processing_time = (end_time - start_time) * 1000
			
			self._log_error("Biometric matching failed",
							sample_id=sample.id,
							template_id=template.id,
							error=str(e))
			
			# Return failed match
			return BiometricMatchResult(
				template_id=template.id,
				sample_id=sample.id,
				similarity_score=0.0,
				confidence_score=0.0,
				is_match=False,
				threshold_used=threshold,
				algorithm_used="error",
				processing_time_ms=processing_time
			)
	
	async def authenticate_multimodal(self, user_id: str, 
									  biometric_samples: List[BiometricSample],
									  fusion_method: FusionMethod = FusionMethod.SCORE_LEVEL,
									  authentication_threshold: float = 0.8) -> FusionResult:
		"""Perform multi-modal biometric authentication with fusion"""
		assert user_id, "User ID is required"
		assert biometric_samples, "Biometric samples are required"
		
		self._log_info("Starting multi-modal biometric authentication",
					   user_id=user_id,
					   sample_count=len(biometric_samples),
					   fusion_method=fusion_method.value)
		
		start_time = asyncio.get_event_loop().time()
		
		# Get user's biometric templates
		user_templates = [
			template for template in self._templates.values() 
			if template.user_id == user_id and template.is_valid()
		]
		
		if not user_templates:
			self._log_warning("No valid templates found for user", user_id=user_id)
			return FusionResult(
				user_id=user_id,
				individual_results=[],
				modalities_used=[],
				fused_score=0.0,
				fusion_confidence=0.0,
				authentication_decision=False,
				fusion_method=fusion_method,
				overall_quality=0.0,
				reliability_score=0.0,
				processing_time_ms=0.0,
				spoofing_detected=True,
				liveness_passed=False
			)
		
		# Process each biometric sample
		individual_results = []
		modalities_used = []
		all_liveness_passed = True
		spoofing_detected = False
		anomalies = []
		quality_scores = []
		
		for sample in biometric_samples:
			# Check liveness
			if sample.liveness_status == LivenessStatus.SPOOF:
				spoofing_detected = True
				anomalies.append(f"Spoofing detected in {sample.biometric_type.value}")
			elif sample.liveness_status != LivenessStatus.LIVE:
				all_liveness_passed = False
				anomalies.append(f"Liveness uncertain for {sample.biometric_type.value}")
			
			# Find matching templates for this modality
			matching_templates = [
				template for template in user_templates
				if template.biometric_type == sample.biometric_type
			]
			
			if not matching_templates:
				anomalies.append(f"No template for {sample.biometric_type.value}")
				continue
			
			# Match against the best template (highest quality)
			best_template = max(matching_templates, key=lambda t: t.quality_score)
			
			# Perform matching
			match_result = await self.match_biometric(sample, best_template)
			individual_results.append(match_result)
			modalities_used.append(sample.biometric_type)
			quality_scores.append(sample.quality_score)
		
		if not individual_results:
			self._log_error("No successful matches for any modality", user_id=user_id)
			return FusionResult(
				user_id=user_id,
				individual_results=[],
				modalities_used=[],
				fused_score=0.0,
				fusion_confidence=0.0,
				authentication_decision=False,
				fusion_method=fusion_method,
				overall_quality=0.0,
				reliability_score=0.0,
				processing_time_ms=0.0,
				spoofing_detected=spoofing_detected,
				liveness_passed=all_liveness_passed,
				anomalies_detected=anomalies
			)
		
		# Perform score-level fusion
		fused_score, fusion_confidence = await self._fuse_scores(
			individual_results, fusion_method
		)
		
		# Calculate overall quality and reliability
		overall_quality = np.mean(quality_scores) if quality_scores else 0.0
		reliability_factors = [
			fusion_confidence,
			overall_quality,
			1.0 if all_liveness_passed else 0.5,
			1.0 if not spoofing_detected else 0.2,
			max(0.0, 1.0 - len(anomalies) * 0.1)
		]
		reliability_score = np.mean(reliability_factors)
		
		# Make final authentication decision
		authentication_decision = (
			fused_score >= authentication_threshold and
			all_liveness_passed and
			not spoofing_detected and
			len(individual_results) >= 1  # At least one successful match
		)
		
		end_time = asyncio.get_event_loop().time()
		processing_time = (end_time - start_time) * 1000
		
		fusion_result = FusionResult(
			user_id=user_id,
			individual_results=individual_results,
			modalities_used=modalities_used,
			fused_score=fused_score,
			fusion_confidence=fusion_confidence,
			authentication_decision=authentication_decision,
			fusion_method=fusion_method,
			overall_quality=overall_quality,
			reliability_score=reliability_score,
			processing_time_ms=processing_time,
			spoofing_detected=spoofing_detected,
			liveness_passed=all_liveness_passed,
			anomalies_detected=anomalies
		)
		
		self._log_info("Multi-modal authentication complete",
					   user_id=user_id,
					   fused_score=fused_score,
					   decision=authentication_decision,
					   processing_time_ms=processing_time,
					   modalities=len(modalities_used))
		
		return fusion_result
	
	async def _fuse_scores(self, match_results: List[BiometricMatchResult],
						   fusion_method: FusionMethod) -> Tuple[float, float]:
		"""Fuse multiple biometric match scores"""
		if not match_results:
			return 0.0, 0.0
		
		if fusion_method == FusionMethod.SCORE_LEVEL:
			# Weighted average of similarity scores
			weighted_scores = []
			weights = []
			
			for result in match_results:
				# Get modality from template (simplified - would be stored properly)
				modality_weight = 1.0  # Default weight
				for biometric_type, weight in self._fusion_weights.items():
					# In practice, would map result to correct modality
					modality_weight = weight
					break
				
				weighted_score = result.similarity_score * result.confidence_score * modality_weight
				weighted_scores.append(weighted_score)
				weights.append(modality_weight)
			
			if sum(weights) > 0:
				fused_score = sum(weighted_scores) / sum(weights)
			else:
				fused_score = np.mean([r.similarity_score for r in match_results])
			
			# Fusion confidence based on individual confidences and agreement
			confidences = [r.confidence_score for r in match_results]
			scores = [r.similarity_score for r in match_results]
			
			avg_confidence = np.mean(confidences)
			score_agreement = 1.0 - np.std(scores)  # Higher agreement = higher confidence
			
			fusion_confidence = (avg_confidence + score_agreement) / 2
			
		elif fusion_method == FusionMethod.DECISION_LEVEL:
			# Majority voting on match decisions
			positive_votes = sum(1 for r in match_results if r.is_match)
			total_votes = len(match_results)
			
			fused_score = positive_votes / total_votes
			fusion_confidence = abs(positive_votes - total_votes/2) / (total_votes/2)
			
		else:  # Default to simple average
			fused_score = np.mean([r.similarity_score for r in match_results])
			fusion_confidence = np.mean([r.confidence_score for r in match_results])
		
		return max(0.0, min(1.0, fused_score)), max(0.0, min(1.0, fusion_confidence))
	
	def add_biometric_template(self, template: BiometricTemplate):
		"""Add biometric template to the engine"""
		self._templates[template.id] = template
		self._log_info("Biometric template added",
					   template_id=template.id,
					   user_id=template.user_id,
					   biometric_type=template.biometric_type.value)
	
	def get_user_templates(self, user_id: str, 
						   biometric_type: Optional[BiometricType] = None) -> List[BiometricTemplate]:
		"""Get biometric templates for user"""
		templates = [
			template for template in self._templates.values()
			if template.user_id == user_id and template.is_valid()
		]
		
		if biometric_type:
			templates = [t for t in templates if t.biometric_type == biometric_type]
		
		return templates
	
	def set_fusion_weights(self, weights: Dict[BiometricType, float]):
		"""Set fusion weights for different biometric modalities"""
		# Normalize weights to sum to 1.0
		total_weight = sum(weights.values())
		if total_weight > 0:
			self._fusion_weights = {
				modality: weight / total_weight
				for modality, weight in weights.items()
			}
			self._log_info("Fusion weights updated", weights=self._fusion_weights)
	
	def set_quality_thresholds(self, thresholds: Dict[BiometricType, float]):
		"""Set quality thresholds for different biometric modalities"""
		self._quality_thresholds = thresholds
		self._log_info("Quality thresholds updated", thresholds=thresholds)
	
	def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
		"""Get performance metrics for biometric operations"""
		metrics = {}
		
		for operation, times in self._operation_times.items():
			if times:
				metrics[operation] = {
					"avg_ms": np.mean(times),
					"min_ms": np.min(times),
					"max_ms": np.max(times),
					"std_ms": np.std(times),
					"count": len(times)
				}
		
		return metrics
	
	def get_fusion_statistics(self, user_id: str) -> Dict[str, Any]:
		"""Get fusion statistics for user"""
		user_templates = self.get_user_templates(user_id)
		
		stats = {
			"user_id": user_id,
			"total_templates": len(user_templates),
			"modalities_enrolled": list(set(t.biometric_type for t in user_templates)),
			"avg_template_quality": np.mean([t.quality_score for t in user_templates]) if user_templates else 0.0,
			"template_success_rate": np.mean([t.get_success_rate() for t in user_templates]) if user_templates else 0.0
		}
		
		return stats
	
	def clear_user_templates(self, user_id: str):
		"""Clear all biometric templates for user (GDPR compliance)"""
		user_template_ids = [
			template_id for template_id, template in self._templates.items()
			if template.user_id == user_id
		]
		
		for template_id in user_template_ids:
			del self._templates[template_id]
		
		self._log_info("User biometric templates cleared",
					   user_id=user_id,
					   templates_removed=len(user_template_ids))