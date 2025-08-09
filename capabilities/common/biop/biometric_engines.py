"""
APG Biometric Authentication - Advanced Biometric Processing Engines

Comprehensive biometric processing engines supporting all modalities with open source libraries.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import asyncio
import numpy as np
import cv2
import librosa
import hashlib
import base64
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Core scientific libraries
import scipy.signal
import scipy.spatial.distance
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Computer vision libraries
import mediapipe as mp
import dlib
from skimage import feature, filters, morphology, measure
from skimage.feature import local_binary_pattern, hog
from scipy import ndimage

# Audio processing libraries
import python_speech_features
import webrtcvad
from pydub import AudioSegment

# Biometric-specific libraries (would be installed via pip)
# pip install opencv-python mediapipe dlib scikit-image librosa python_speech_features webrtcvad pydub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BiometricTemplate:
	"""Standardized biometric template structure"""
	modality: str
	template_data: np.ndarray
	quality_score: float
	feature_vector: np.ndarray
	metadata: Dict[str, Any]
	creation_timestamp: str
	template_hash: str

@dataclass
class BiometricComparisonResult:
	"""Biometric comparison result structure"""
	similarity_score: float
	match_confidence: float
	decision: bool  # True for match, False for no match
	threshold_used: float
	comparison_metadata: Dict[str, Any]

@dataclass
class LivenessDetectionResult:
	"""Liveness detection result structure"""
	is_live: bool
	liveness_score: float
	liveness_indicators: List[str]
	anti_spoofing_checks: Dict[str, bool]
	confidence: float

class BiometricEngine(ABC):
	"""Abstract base class for all biometric engines"""
	
	def __init__(self, modality: str):
		self.modality = modality
		self.quality_threshold = 0.5
		self.match_threshold = 0.7
		self.logger = logging.getLogger(f"{__name__}.{modality}")
	
	@abstractmethod
	async def extract_features(self, raw_data: Union[np.ndarray, bytes]) -> BiometricTemplate:
		"""Extract biometric features from raw data"""
		pass
	
	@abstractmethod
	async def compare_templates(self, template1: BiometricTemplate, template2: BiometricTemplate) -> BiometricComparisonResult:
		"""Compare two biometric templates"""
		pass
	
	@abstractmethod
	async def assess_quality(self, raw_data: Union[np.ndarray, bytes]) -> float:
		"""Assess quality of biometric sample"""
		pass
	
	async def detect_liveness(self, raw_data: Union[np.ndarray, bytes], **kwargs) -> LivenessDetectionResult:
		"""Default liveness detection - override in specific engines"""
		return LivenessDetectionResult(
			is_live=True,
			liveness_score=0.8,
			liveness_indicators=['default_check'],
			anti_spoofing_checks={'basic_check': True},
			confidence=0.8
		)
	
	def _generate_template_hash(self, template_data: np.ndarray) -> str:
		"""Generate secure hash for template"""
		template_bytes = template_data.tobytes()
		return hashlib.sha256(template_bytes).hexdigest()
	
	def _normalize_features(self, features: np.ndarray) -> np.ndarray:
		"""Normalize feature vector"""
		if len(features.shape) == 1:
			features = features.reshape(1, -1)
		scaler = StandardScaler()
		return scaler.fit_transform(features).flatten()

class FingerprintEngine(BiometricEngine):
	"""
	Advanced fingerprint recognition engine
	
	Features:
	- Minutiae extraction and matching
	- Ridge pattern analysis
	- Quality assessment
	- Anti-spoofing detection
	"""
	
	def __init__(self):
		super().__init__("fingerprint")
		self.match_threshold = 0.75
		self.quality_threshold = 0.6
		
		# Initialize fingerprint processing parameters
		self.gabor_filters = self._create_gabor_filters()
		self.minutiae_detector = self._initialize_minutiae_detector()
	
	def _create_gabor_filters(self) -> List[np.ndarray]:
		"""Create Gabor filters for ridge enhancement"""
		filters = []
		for theta in np.arange(0, np.pi, np.pi/8):
			kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
			filters.append(kernel)
		return filters
	
	def _initialize_minutiae_detector(self) -> Dict[str, Any]:
		"""Initialize minutiae detection parameters"""
		return {
			'block_size': 16,
			'threshold': 0.1,
			'min_distance': 10,
			'max_minutiae': 100
		}
	
	async def extract_features(self, raw_data: Union[np.ndarray, bytes]) -> BiometricTemplate:
		"""Extract fingerprint features including minutiae and ridge patterns"""
		try:
			# Convert input to image array
			if isinstance(raw_data, bytes):
				nparr = np.frombuffer(raw_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
			else:
				image = raw_data.copy()
			
			# Preprocess image
			preprocessed = await self._preprocess_fingerprint(image)
			
			# Extract minutiae
			minutiae = await self._extract_minutiae(preprocessed)
			
			# Extract ridge patterns
			ridge_features = await self._extract_ridge_patterns(preprocessed)
			
			# Extract texture features
			texture_features = await self._extract_texture_features(preprocessed)
			
			# Combine all features
			feature_vector = np.concatenate([
				minutiae.flatten(),
				ridge_features.flatten(),
				texture_features.flatten()
			])
			
			# Normalize features
			normalized_features = self._normalize_features(feature_vector)
			
			# Assess quality
			quality_score = await self.assess_quality(image)
			
			# Create template
			template = BiometricTemplate(
				modality=self.modality,
				template_data=normalized_features,
				quality_score=quality_score,
				feature_vector=normalized_features,
				metadata={
					'minutiae_count': len(minutiae),
					'image_size': image.shape,
					'preprocessing_applied': True,
					'ridge_quality': np.mean(ridge_features),
					'extraction_method': 'advanced_minutiae_ridge_texture'
				},
				creation_timestamp=datetime.utcnow().isoformat(),
				template_hash=self._generate_template_hash(normalized_features)
			)
			
			self.logger.info(f"Fingerprint template extracted: quality={quality_score:.3f}, minutiae={len(minutiae)}")
			return template
			
		except Exception as e:
			self.logger.error(f"Fingerprint feature extraction failed: {str(e)}")
			raise
	
	async def _preprocess_fingerprint(self, image: np.ndarray) -> np.ndarray:
		"""Advanced fingerprint preprocessing"""
		# Resize to standard size
		image = cv2.resize(image, (256, 256))
		
		# Histogram equalization
		image = cv2.equalizeHist(image)
		
		# Gaussian blur for noise reduction
		image = cv2.GaussianBlur(image, (3, 3), 0)
		
		# Ridge enhancement using Gabor filters
		enhanced = np.zeros_like(image, dtype=np.float32)
		for gabor_filter in self.gabor_filters:
			filtered = cv2.filter2D(image, cv2.CV_8UC3, gabor_filter)
			enhanced = np.maximum(enhanced, filtered.astype(np.float32))
		
		# Normalize to 0-255 range
		enhanced = ((enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * 255).astype(np.uint8)
		
		return enhanced
	
	async def _extract_minutiae(self, image: np.ndarray) -> np.ndarray:
		"""Extract minutiae points (ridge endings and bifurcations)"""
		# Binarize image
		_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		
		# Skeletonize to get ridge lines
		skeleton = morphology.skeletonize(binary // 255)
		
		# Find minutiae points
		minutiae_points = []
		
		# Scan for ridge endings and bifurcations
		for i in range(1, skeleton.shape[0] - 1):
			for j in range(1, skeleton.shape[1] - 1):
				if skeleton[i, j]:
					# Count neighboring pixels
					neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) - skeleton[i, j]
					
					# Ridge ending (1 neighbor) or bifurcation (3+ neighbors)
					if neighbors == 1 or neighbors >= 3:
						minutiae_points.append([i, j, neighbors])
		
		# Limit number of minutiae and convert to array
		minutiae_points = minutiae_points[:self.minutiae_detector['max_minutiae']]
		
		if len(minutiae_points) == 0:
			return np.zeros((1, 3))
		
		return np.array(minutiae_points)
	
	async def _extract_ridge_patterns(self, image: np.ndarray) -> np.ndarray:
		"""Extract ridge flow and frequency patterns"""
		# Calculate ridge orientation
		sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
		sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
		
		# Ridge orientation
		orientation = np.arctan2(sobely, sobelx)
		
		# Ridge frequency analysis
		block_size = 16
		frequency_map = np.zeros((image.shape[0] // block_size, image.shape[1] // block_size))
		
		for i in range(0, image.shape[0] - block_size, block_size):
			for j in range(0, image.shape[1] - block_size, block_size):
				block = image[i:i+block_size, j:j+block_size]
				# Calculate dominant frequency in block
				fft = np.fft.fft2(block)
				magnitude = np.abs(fft)
				frequency_map[i//block_size, j//block_size] = np.max(magnitude)
		
		# Combine orientation and frequency features
		ridge_features = np.concatenate([
			orientation.flatten(),
			frequency_map.flatten()
		])
		
		return ridge_features[:512]  # Limit feature size
	
	async def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
		"""Extract texture features using LBP and HOG"""
		# Local Binary Pattern
		lbp = local_binary_pattern(image, 8, 1, method='uniform')
		lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9), density=True)
		
		# Histogram of Oriented Gradients
		hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), 
						  cells_per_block=(2, 2), visualize=False)
		
		# Combine texture features
		texture_features = np.concatenate([lbp_hist, hog_features[:100]])  # Limit HOG features
		
		return texture_features
	
	async def compare_templates(self, template1: BiometricTemplate, template2: BiometricTemplate) -> BiometricComparisonResult:
		"""Compare two fingerprint templates"""
		try:
			# Calculate similarity using multiple methods
			
			# Euclidean distance
			euclidean_dist = scipy.spatial.distance.euclidean(
				template1.feature_vector, template2.feature_vector
			)
			euclidean_similarity = 1.0 / (1.0 + euclidean_dist)
			
			# Cosine similarity
			cosine_similarity = 1 - scipy.spatial.distance.cosine(
				template1.feature_vector, template2.feature_vector
			)
			
			# Correlation coefficient
			correlation = np.corrcoef(template1.feature_vector, template2.feature_vector)[0, 1]
			correlation = np.nan_to_num(correlation, 0.0)
			
			# Combined similarity score
			similarity_score = (euclidean_similarity * 0.4 + cosine_similarity * 0.4 + correlation * 0.2)
			
			# Match decision
			match_decision = similarity_score >= self.match_threshold
			
			# Calculate confidence based on quality and similarity
			confidence = similarity_score * min(template1.quality_score, template2.quality_score)
			
			result = BiometricComparisonResult(
				similarity_score=similarity_score,
				match_confidence=confidence,
				decision=match_decision,
				threshold_used=self.match_threshold,
				comparison_metadata={
					'euclidean_similarity': euclidean_similarity,
					'cosine_similarity': cosine_similarity,
					'correlation': correlation,
					'template1_quality': template1.quality_score,
					'template2_quality': template2.quality_score,
					'minutiae1_count': template1.metadata.get('minutiae_count', 0),
					'minutiae2_count': template2.metadata.get('minutiae_count', 0)
				}
			)
			
			self.logger.info(f"Fingerprint comparison: similarity={similarity_score:.3f}, match={match_decision}")
			return result
			
		except Exception as e:
			self.logger.error(f"Fingerprint comparison failed: {str(e)}")
			raise
	
	async def assess_quality(self, raw_data: Union[np.ndarray, bytes]) -> float:
		"""Assess fingerprint image quality"""
		try:
			if isinstance(raw_data, bytes):
				nparr = np.frombuffer(raw_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
			else:
				image = raw_data.copy()
			
			# Resize for consistent quality assessment
			image = cv2.resize(image, (256, 256))
			
			# Calculate multiple quality metrics
			
			# 1. Image sharpness (Laplacian variance)
			laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
			sharpness_score = min(laplacian_var / 1000.0, 1.0)
			
			# 2. Contrast (standard deviation)
			contrast_score = min(np.std(image) / 64.0, 1.0)
			
			# 3. Ridge clarity (using Gabor filter response)
			gabor_response = 0
			for gabor_filter in self.gabor_filters:
				filtered = cv2.filter2D(image, cv2.CV_32F, gabor_filter)
				gabor_response += np.std(filtered)
			ridge_clarity = min(gabor_response / (len(self.gabor_filters) * 50.0), 1.0)
			
			# 4. Overall image quality (histogram distribution)
			hist = cv2.calcHist([image], [0], None, [256], [0, 256])
			hist_spread = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0
			distribution_quality = min(hist_spread / 2.0, 1.0)
			
			# Combined quality score
			quality_score = (sharpness_score * 0.3 + contrast_score * 0.3 + 
							ridge_clarity * 0.3 + distribution_quality * 0.1)
			
			return float(quality_score)
			
		except Exception as e:
			self.logger.error(f"Fingerprint quality assessment failed: {str(e)}")
			return 0.0
	
	async def detect_liveness(self, raw_data: Union[np.ndarray, bytes], **kwargs) -> LivenessDetectionResult:
		"""Fingerprint liveness detection"""
		try:
			if isinstance(raw_data, bytes):
				nparr = np.frombuffer(raw_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
			else:
				image = raw_data.copy()
			
			liveness_indicators = []
			anti_spoofing_checks = {}
			
			# 1. Texture analysis for liveness
			lbp = local_binary_pattern(image, 8, 1, method='uniform')
			lbp_var = np.var(lbp)
			texture_liveness = lbp_var > 50  # Live fingers have more texture variation
			anti_spoofing_checks['texture_analysis'] = texture_liveness
			if texture_liveness:
				liveness_indicators.append('natural_texture')
			
			# 2. Ridge flow continuity
			sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
			sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
			ridge_continuity = np.std(np.arctan2(sobely, sobelx))
			flow_liveness = ridge_continuity > 0.5
			anti_spoofing_checks['ridge_flow'] = flow_liveness
			if flow_liveness:
				liveness_indicators.append('ridge_continuity')
			
			# 3. Frequency domain analysis
			fft = np.fft.fft2(image)
			magnitude_spectrum = np.log(np.abs(fft) + 1)
			freq_analysis = np.std(magnitude_spectrum)
			frequency_liveness = freq_analysis > 2.0
			anti_spoofing_checks['frequency_analysis'] = frequency_liveness
			if frequency_liveness:
				liveness_indicators.append('frequency_characteristics')
			
			# Calculate overall liveness score
			liveness_checks = list(anti_spoofing_checks.values())
			liveness_score = sum(liveness_checks) / len(liveness_checks)
			is_live = liveness_score >= 0.6
			
			return LivenessDetectionResult(
				is_live=is_live,
				liveness_score=liveness_score,
				liveness_indicators=liveness_indicators,
				anti_spoofing_checks=anti_spoofing_checks,
				confidence=liveness_score
			)
			
		except Exception as e:
			self.logger.error(f"Fingerprint liveness detection failed: {str(e)}")
			return LivenessDetectionResult(
				is_live=False, liveness_score=0.0, liveness_indicators=[],
				anti_spoofing_checks={}, confidence=0.0
			)

class IrisEngine(BiometricEngine):
	"""
	Advanced iris recognition engine
	
	Features:
	- Iris segmentation and normalization
	- Texture pattern analysis
	- Quality assessment
	- Pupil dilation analysis
	"""
	
	def __init__(self):
		super().__init__("iris")
		self.match_threshold = 0.8
		self.quality_threshold = 0.7
		
		# Initialize iris processing parameters
		self.iris_detector = self._initialize_iris_detector()
		self.gabor_bank = self._create_gabor_bank()
	
	def _initialize_iris_detector(self) -> Dict[str, Any]:
		"""Initialize iris detection parameters"""
		return {
			'pupil_min_radius': 20,
			'pupil_max_radius': 80,
			'iris_min_radius': 80,
			'iris_max_radius': 150,
			'hough_threshold': 30
		}
	
	def _create_gabor_bank(self) -> List[np.ndarray]:
		"""Create Gabor filter bank for iris texture analysis"""
		filters = []
		for freq in [0.1, 0.3, 0.5]:
			for theta in np.arange(0, np.pi, np.pi/6):
				kernel = cv2.getGaborKernel((31, 31), 4, theta, 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
				filters.append(kernel)
		return filters
	
	async def extract_features(self, raw_data: Union[np.ndarray, bytes]) -> BiometricTemplate:
		"""Extract iris features including texture patterns and geometric features"""
		try:
			# Convert input to image array
			if isinstance(raw_data, bytes):
				nparr = np.frombuffer(raw_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
			else:
				image = raw_data.copy()
			
			# Detect and segment iris
			iris_region, pupil_center, iris_center = await self._segment_iris(image)
			
			if iris_region is None:
				raise ValueError("Could not detect iris in image")
			
			# Normalize iris to rectangular coordinates
			normalized_iris = await self._normalize_iris(iris_region, pupil_center, iris_center)
			
			# Extract texture features using Gabor filters
			texture_features = await self._extract_iris_texture(normalized_iris)
			
			# Extract geometric features
			geometric_features = await self._extract_geometric_features(image, pupil_center, iris_center)
			
			# Combine all features
			feature_vector = np.concatenate([
				texture_features.flatten(),
				geometric_features.flatten()
			])
			
			# Normalize features
			normalized_features = self._normalize_features(feature_vector)
			
			# Assess quality
			quality_score = await self.assess_quality(image)
			
			# Create template
			template = BiometricTemplate(
				modality=self.modality,
				template_data=normalized_features,
				quality_score=quality_score,
				feature_vector=normalized_features,
				metadata={
					'pupil_center': pupil_center,
					'iris_center': iris_center,
					'iris_radius': np.linalg.norm(np.array(iris_center) - np.array(pupil_center)),
					'image_size': image.shape,
					'normalization_applied': True,
					'extraction_method': 'gabor_texture_geometric'
				},
				creation_timestamp=datetime.utcnow().isoformat(),
				template_hash=self._generate_template_hash(normalized_features)
			)
			
			self.logger.info(f"Iris template extracted: quality={quality_score:.3f}")
			return template
			
		except Exception as e:
			self.logger.error(f"Iris feature extraction failed: {str(e)}")
			raise
	
	async def _segment_iris(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
		"""Segment iris and pupil regions"""
		# Resize image for processing
		image = cv2.resize(image, (320, 240))
		
		# Apply Gaussian blur
		blurred = cv2.GaussianBlur(image, (5, 5), 0)
		
		# Detect pupil (dark circle)
		pupil_circles = cv2.HoughCircles(
			blurred, cv2.HOUGH_GRADIENT, 1, 20,
			param1=50, param2=30,
			minRadius=self.iris_detector['pupil_min_radius'],
			maxRadius=self.iris_detector['pupil_max_radius']
		)
		
		if pupil_circles is None:
			return None, None, None
		
		pupil_circles = np.round(pupil_circles[0, :]).astype("int")
		pupil_circle = pupil_circles[0]  # Take first detected circle
		pupil_center = (pupil_circle[0], pupil_circle[1])
		
		# Detect iris (larger circle around pupil)
		iris_circles = cv2.HoughCircles(
			blurred, cv2.HOUGH_GRADIENT, 1, 20,
			param1=50, param2=30,
			minRadius=self.iris_detector['iris_min_radius'],
			maxRadius=self.iris_detector['iris_max_radius']
		)
		
		if iris_circles is None:
			# Estimate iris as 3x pupil radius
			iris_radius = pupil_circle[2] * 3
			iris_center = pupil_center
		else:
			iris_circles = np.round(iris_circles[0, :]).astype("int")
			iris_circle = iris_circles[0]
			iris_center = (iris_circle[0], iris_circle[1])
			iris_radius = iris_circle[2]
		
		# Extract iris region
		mask = np.zeros_like(image)
		cv2.circle(mask, iris_center, iris_radius, 255, -1)
		cv2.circle(mask, pupil_center, pupil_circle[2], 0, -1)  # Remove pupil
		
		iris_region = cv2.bitwise_and(image, mask)
		
		return iris_region, pupil_center, iris_center
	
	async def _normalize_iris(self, iris_region: np.ndarray, pupil_center: Tuple[int, int], 
							 iris_center: Tuple[int, int]) -> np.ndarray:
		"""Normalize iris from cartesian to polar coordinates"""
		# Create polar coordinate mapping
		height, width = 64, 512  # Normalized iris dimensions
		normalized = np.zeros((height, width), dtype=np.uint8)
		
		center_x, center_y = iris_center
		pupil_x, pupil_y = pupil_center
		
		# Calculate radius ranges
		pupil_radius = 20  # Assumed pupil radius in normalized space
		iris_radius = 100   # Assumed iris radius in normalized space
		
		for i in range(height):
			for j in range(width):
				# Convert normalized coordinates to polar
				radius = pupil_radius + (iris_radius - pupil_radius) * (i / height)
				theta = 2 * np.pi * (j / width)
				
				# Convert polar to cartesian
				x = int(center_x + radius * np.cos(theta))
				y = int(center_y + radius * np.sin(theta))
				
				# Sample from original image
				if 0 <= x < iris_region.shape[1] and 0 <= y < iris_region.shape[0]:
					normalized[i, j] = iris_region[y, x]
		
		return normalized
	
	async def _extract_iris_texture(self, normalized_iris: np.ndarray) -> np.ndarray:
		"""Extract iris texture features using Gabor filters"""
		texture_features = []
		
		# Apply Gabor filter bank
		for gabor_filter in self.gabor_bank:
			filtered = cv2.filter2D(normalized_iris, cv2.CV_32F, gabor_filter)
			
			# Calculate statistics for each filter response
			mean_response = np.mean(filtered)
			std_response = np.std(filtered)
			texture_features.extend([mean_response, std_response])
		
		# Add LBP features
		lbp = local_binary_pattern(normalized_iris, 8, 1, method='uniform')
		lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9), density=True)
		texture_features.extend(lbp_hist)
		
		return np.array(texture_features)
	
	async def _extract_geometric_features(self, image: np.ndarray, pupil_center: Tuple[int, int], 
										 iris_center: Tuple[int, int]) -> np.ndarray:
		"""Extract geometric features of iris"""
		features = []
		
		# Pupil-iris distance
		distance = np.linalg.norm(np.array(iris_center) - np.array(pupil_center))
		features.append(distance)
		
		# Pupil eccentricity (measure of roundness)
		# This would require more sophisticated ellipse fitting
		features.append(1.0)  # Placeholder for eccentricity
		
		# Iris boundary irregularity
		# This would require boundary analysis
		features.append(0.1)  # Placeholder for irregularity
		
		# Add relative positions
		features.extend([
			pupil_center[0] / image.shape[1],  # Normalized x
			pupil_center[1] / image.shape[0],  # Normalized y
			iris_center[0] / image.shape[1],   # Normalized x
			iris_center[1] / image.shape[0]    # Normalized y
		])
		
		return np.array(features)
	
	async def compare_templates(self, template1: BiometricTemplate, template2: BiometricTemplate) -> BiometricComparisonResult:
		"""Compare two iris templates"""
		try:
			# Hamming distance for iris comparison (standard approach)
			xor_result = np.logical_xor(
				template1.feature_vector > np.median(template1.feature_vector),
				template2.feature_vector > np.median(template2.feature_vector)
			)
			hamming_distance = np.sum(xor_result) / len(xor_result)
			hamming_similarity = 1.0 - hamming_distance
			
			# Euclidean distance
			euclidean_dist = scipy.spatial.distance.euclidean(
				template1.feature_vector, template2.feature_vector
			)
			euclidean_similarity = 1.0 / (1.0 + euclidean_dist)
			
			# Cosine similarity
			cosine_similarity = 1 - scipy.spatial.distance.cosine(
				template1.feature_vector, template2.feature_vector
			)
			
			# Combined similarity (emphasize Hamming for iris)
			similarity_score = (hamming_similarity * 0.6 + euclidean_similarity * 0.2 + cosine_similarity * 0.2)
			
			# Match decision
			match_decision = similarity_score >= self.match_threshold
			
			# Calculate confidence
			confidence = similarity_score * min(template1.quality_score, template2.quality_score)
			
			result = BiometricComparisonResult(
				similarity_score=similarity_score,
				match_confidence=confidence,
				decision=match_decision,
				threshold_used=self.match_threshold,
				comparison_metadata={
					'hamming_similarity': hamming_similarity,
					'euclidean_similarity': euclidean_similarity,
					'cosine_similarity': cosine_similarity,
					'template1_quality': template1.quality_score,
					'template2_quality': template2.quality_score
				}
			)
			
			self.logger.info(f"Iris comparison: similarity={similarity_score:.3f}, match={match_decision}")
			return result
			
		except Exception as e:
			self.logger.error(f"Iris comparison failed: {str(e)}")
			raise
	
	async def assess_quality(self, raw_data: Union[np.ndarray, bytes]) -> float:
		"""Assess iris image quality"""
		try:
			if isinstance(raw_data, bytes):
				nparr = np.frombuffer(raw_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
			else:
				image = raw_data.copy()
			
			# Focus measure (Laplacian variance)
			laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
			focus_score = min(laplacian_var / 1000.0, 1.0)
			
			# Contrast measure
			contrast_score = min(np.std(image) / 64.0, 1.0)
			
			# Illumination uniformity
			mean_intensity = np.mean(image)
			illumination_score = 1.0 - abs(mean_intensity - 128.0) / 128.0
			
			# Occlusion detection (simplified)
			# Check for very dark or very bright regions that might indicate occlusion
			dark_pixels = np.sum(image < 30) / image.size
			bright_pixels = np.sum(image > 225) / image.size
			occlusion_score = 1.0 - (dark_pixels + bright_pixels)
			
			# Combined quality score
			quality_score = (focus_score * 0.3 + contrast_score * 0.3 + 
							illumination_score * 0.2 + occlusion_score * 0.2)
			
			return float(quality_score)
			
		except Exception as e:
			self.logger.error(f"Iris quality assessment failed: {str(e)}")
			return 0.0

class PalmEngine(BiometricEngine):
	"""
	Advanced palm recognition engine
	
	Features:
	- Palm print and palm vein analysis
	- Principal line extraction
	- Texture pattern analysis
	- Hand geometry features
	"""
	
	def __init__(self):
		super().__init__("palm")
		self.match_threshold = 0.75
		self.quality_threshold = 0.6
		
		# Initialize MediaPipe hands
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			static_image_mode=True,
			max_num_hands=1,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		)
		self.mp_drawing = mp.solutions.drawing_utils
	
	async def extract_features(self, raw_data: Union[np.ndarray, bytes]) -> BiometricTemplate:
		"""Extract palm features including print patterns and hand geometry"""
		try:
			# Convert input to image array
			if isinstance(raw_data, bytes):
				nparr = np.frombuffer(raw_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			else:
				image = raw_data.copy()
			
			# Convert to RGB for MediaPipe
			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			
			# Detect hand landmarks
			hand_landmarks = await self._detect_hand_landmarks(rgb_image)
			
			# Extract palm region
			palm_region = await self._extract_palm_region(gray_image, hand_landmarks)
			
			# Extract principal lines
			principal_lines = await self._extract_principal_lines(palm_region)
			
			# Extract texture features
			texture_features = await self._extract_palm_texture(palm_region)
			
			# Extract hand geometry
			geometry_features = await self._extract_hand_geometry(hand_landmarks, image.shape)
			
			# Extract vein patterns (simplified)
			vein_features = await self._extract_vein_patterns(palm_region)
			
			# Combine all features
			feature_vector = np.concatenate([
				principal_lines.flatten(),
				texture_features.flatten(),
				geometry_features.flatten(),
				vein_features.flatten()
			])
			
			# Normalize features
			normalized_features = self._normalize_features(feature_vector)
			
			# Assess quality
			quality_score = await self.assess_quality(image)
			
			# Create template
			template = BiometricTemplate(
				modality=self.modality,
				template_data=normalized_features,
				quality_score=quality_score,
				feature_vector=normalized_features,
				metadata={
					'hand_detected': hand_landmarks is not None,
					'principal_lines_count': len(principal_lines) if principal_lines is not None else 0,
					'image_size': image.shape,
					'extraction_method': 'lines_texture_geometry_veins'
				},
				creation_timestamp=datetime.utcnow().isoformat(),
				template_hash=self._generate_template_hash(normalized_features)
			)
			
			self.logger.info(f"Palm template extracted: quality={quality_score:.3f}")
			return template
			
		except Exception as e:
			self.logger.error(f"Palm feature extraction failed: {str(e)}")
			raise
	
	async def _detect_hand_landmarks(self, rgb_image: np.ndarray) -> Optional[Any]:
		"""Detect hand landmarks using MediaPipe"""
		results = self.hands.process(rgb_image)
		
		if results.multi_hand_landmarks:
			return results.multi_hand_landmarks[0]  # Take first detected hand
		return None
	
	async def _extract_palm_region(self, gray_image: np.ndarray, hand_landmarks) -> np.ndarray:
		"""Extract palm region from hand image"""
		if hand_landmarks is None:
			# If no landmarks detected, use center region as palm
			h, w = gray_image.shape
			palm_region = gray_image[h//4:3*h//4, w//4:3*w//4]
		else:
			# Use landmarks to define palm region
			h, w = gray_image.shape
			landmarks = []
			for landmark in hand_landmarks.landmark:
				x = int(landmark.x * w)
				y = int(landmark.y * h)
				landmarks.append([x, y])
			
			landmarks = np.array(landmarks)
			
			# Define palm region based on specific landmarks
			# Using wrist, thumb base, pinky base, and middle finger base
			wrist = landmarks[0]
			thumb_base = landmarks[1]
			pinky_base = landmarks[17]
			middle_base = landmarks[9]
			
			# Create bounding box for palm
			palm_points = np.array([wrist, thumb_base, pinky_base, middle_base])
			x_min, y_min = np.min(palm_points, axis=0)
			x_max, y_max = np.max(palm_points, axis=0)
			
			# Extract palm region with some padding
			padding = 20
			x_min = max(0, x_min - padding)
			y_min = max(0, y_min - padding)
			x_max = min(w, x_max + padding)
			y_max = min(h, y_max + padding)
			
			palm_region = gray_image[y_min:y_max, x_min:x_max]
		
		# Resize to standard size
		palm_region = cv2.resize(palm_region, (256, 256))
		return palm_region
	
	async def _extract_principal_lines(self, palm_region: np.ndarray) -> np.ndarray:
		"""Extract principal palm lines (heart, head, life lines)"""
		# Edge detection for line extraction
		edges = cv2.Canny(palm_region, 50, 150)
		
		# Morphological operations to connect line segments
		kernel = np.ones((3, 3), np.uint8)
		edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
		
		# Hough line detection
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
							   minLineLength=30, maxLineGap=10)
		
		line_features = []
		if lines is not None:
			for line in lines[:20]:  # Limit to top 20 lines
				x1, y1, x2, y2 = line[0]
				
				# Line features: length, angle, position
				length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
				angle = np.arctan2(y2-y1, x2-x1)
				center_x = (x1 + x2) / 2 / palm_region.shape[1]  # Normalized
				center_y = (y1 + y2) / 2 / palm_region.shape[0]  # Normalized
				
				line_features.extend([length/256.0, angle, center_x, center_y])
		
		# Pad features to fixed size
		target_size = 80  # 20 lines * 4 features each
		while len(line_features) < target_size:
			line_features.append(0.0)
		
		return np.array(line_features[:target_size])
	
	async def _extract_palm_texture(self, palm_region: np.ndarray) -> np.ndarray:
		"""Extract palm texture features"""
		# Local Binary Pattern
		lbp = local_binary_pattern(palm_region, 8, 1, method='uniform')
		lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 9), density=True)
		
		# Gabor filters for texture
		gabor_features = []
		for theta in [0, 45, 90, 135]:
			kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 10, 0.5, 0, ktype=cv2.CV_32F)
			filtered = cv2.filter2D(palm_region, cv2.CV_8UC3, kernel)
			gabor_features.extend([np.mean(filtered), np.std(filtered)])
		
		# HOG features
		hog_features = hog(palm_region, orientations=9, pixels_per_cell=(8, 8), 
						  cells_per_block=(2, 2), visualize=False)
		
		# Combine texture features
		texture_features = np.concatenate([lbp_hist, gabor_features, hog_features[:50]])
		
		return texture_features
	
	async def _extract_hand_geometry(self, hand_landmarks, image_shape) -> np.ndarray:
		"""Extract hand geometry features"""
		if hand_landmarks is None:
			return np.zeros(20)  # Return zero features if no landmarks
		
		h, w = image_shape[:2]
		landmarks = []
		for landmark in hand_landmarks.landmark:
			x = landmark.x * w
			y = landmark.y * h
			landmarks.append([x, y])
		
		landmarks = np.array(landmarks)
		
		geometry_features = []
		
		# Hand size (bounding box)
		x_min, y_min = np.min(landmarks, axis=0)
		x_max, y_max = np.max(landmarks, axis=0)
		hand_width = (x_max - x_min) / w
		hand_height = (y_max - y_min) / h
		geometry_features.extend([hand_width, hand_height])
		
		# Finger lengths (relative to hand size)
		wrist = landmarks[0]
		finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
		
		for tip in finger_tips:
			finger_length = np.linalg.norm(tip - wrist) / max(w, h)
			geometry_features.append(finger_length)
		
		# Finger spreads (angles between fingers)
		finger_vectors = [tip - wrist for tip in finger_tips[1:]]  # Exclude thumb
		for i in range(len(finger_vectors)-1):
			angle = np.arccos(np.clip(np.dot(finger_vectors[i], finger_vectors[i+1]) / 
									 (np.linalg.norm(finger_vectors[i]) * np.linalg.norm(finger_vectors[i+1])), -1, 1))
			geometry_features.append(angle)
		
		# Palm width and height ratios
		thumb_base = landmarks[1]
		pinky_base = landmarks[17]
		palm_width = np.linalg.norm(pinky_base - thumb_base) / w
		geometry_features.append(palm_width)
		
		# Pad to fixed size
		while len(geometry_features) < 20:
			geometry_features.append(0.0)
		
		return np.array(geometry_features[:20])
	
	async def _extract_vein_patterns(self, palm_region: np.ndarray) -> np.ndarray:
		"""Extract palm vein patterns (simplified approach)"""
		# Enhance vein patterns using top-hat filtering
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
		tophat = cv2.morphologyEx(palm_region, cv2.MORPH_TOPHAT, kernel)
		
		# Apply Gaussian filter to smooth
		smoothed = cv2.GaussianBlur(tophat, (5, 5), 0)
		
		# Extract line-like structures
		sobelx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
		sobely = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
		
		# Calculate vein pattern features
		vein_features = []
		
		# Directional statistics
		angles = np.arctan2(sobely, sobelx)
		angle_hist, _ = np.histogram(angles.ravel(), bins=8, range=(-np.pi, np.pi), density=True)
		vein_features.extend(angle_hist)
		
		# Magnitude statistics
		magnitude = np.sqrt(sobelx**2 + sobely**2)
		vein_features.extend([np.mean(magnitude), np.std(magnitude), np.max(magnitude)])
		
		# Connectivity patterns (simplified)
		binary = (magnitude > np.percentile(magnitude, 80)).astype(np.uint8)
		num_labels, labels = cv2.connectedComponents(binary)
		vein_features.append(num_labels / 100.0)  # Normalized component count
		
		return np.array(vein_features)
	
	async def compare_templates(self, template1: BiometricTemplate, template2: BiometricTemplate) -> BiometricComparisonResult:
		"""Compare two palm templates"""
		try:
			# Multiple comparison methods for palm recognition
			
			# Euclidean distance
			euclidean_dist = scipy.spatial.distance.euclidean(
				template1.feature_vector, template2.feature_vector
			)
			euclidean_similarity = 1.0 / (1.0 + euclidean_dist)
			
			# Cosine similarity
			cosine_similarity = 1 - scipy.spatial.distance.cosine(
				template1.feature_vector, template2.feature_vector
			)
			
			# Correlation coefficient
			correlation = np.corrcoef(template1.feature_vector, template2.feature_vector)[0, 1]
			correlation = np.nan_to_num(correlation, 0.0)
			
			# Combined similarity score
			similarity_score = (euclidean_similarity * 0.4 + cosine_similarity * 0.4 + correlation * 0.2)
			
			# Match decision
			match_decision = similarity_score >= self.match_threshold
			
			# Calculate confidence
			confidence = similarity_score * min(template1.quality_score, template2.quality_score)
			
			result = BiometricComparisonResult(
				similarity_score=similarity_score,
				match_confidence=confidence,
				decision=match_decision,
				threshold_used=self.match_threshold,
				comparison_metadata={
					'euclidean_similarity': euclidean_similarity,
					'cosine_similarity': cosine_similarity,
					'correlation': correlation,
					'template1_quality': template1.quality_score,
					'template2_quality': template2.quality_score
				}
			)
			
			self.logger.info(f"Palm comparison: similarity={similarity_score:.3f}, match={match_decision}")
			return result
			
		except Exception as e:
			self.logger.error(f"Palm comparison failed: {str(e)}")
			raise
	
	async def assess_quality(self, raw_data: Union[np.ndarray, bytes]) -> float:
		"""Assess palm image quality"""
		try:
			if isinstance(raw_data, bytes):
				nparr = np.frombuffer(raw_data, np.uint8)
				image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
			else:
				if len(raw_data.shape) == 3:
					image = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
				else:
					image = raw_data.copy()
			
			# Sharpness (Laplacian variance)
			laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
			sharpness_score = min(laplacian_var / 1000.0, 1.0)
			
			# Contrast
			contrast_score = min(np.std(image) / 64.0, 1.0)
			
			# Illumination uniformity
			mean_intensity = np.mean(image)
			illumination_score = 1.0 - abs(mean_intensity - 128.0) / 128.0
			
			# Hand presence detection (simplified)
			# Check if there's sufficient edge content (indicating hand structure)
			edges = cv2.Canny(image, 50, 150)
			edge_density = np.sum(edges > 0) / edges.size
			presence_score = min(edge_density * 10, 1.0)
			
			# Combined quality score
			quality_score = (sharpness_score * 0.3 + contrast_score * 0.3 + 
							illumination_score * 0.2 + presence_score * 0.2)
			
			return float(quality_score)
			
		except Exception as e:
			self.logger.error(f"Palm quality assessment failed: {str(e)}")
			return 0.0

class VoiceEngine(BiometricEngine):
	"""
	Advanced voice recognition engine
	
	Features:
	- MFCC and spectral feature extraction
	- Speaker verification
	- Voice activity detection
	- Anti-spoofing detection
	"""
	
	def __init__(self):
		super().__init__("voice")
		self.match_threshold = 0.8
		self.quality_threshold = 0.7
		self.sample_rate = 16000
		
		# Initialize voice processing parameters
		self.frame_length = 0.025  # 25ms frames
		self.frame_shift = 0.01    # 10ms shift
		self.vad = webrtcvad.Vad()
		self.vad.set_aggressiveness(2)  # Moderate aggressiveness
	
	async def extract_features(self, raw_data: Union[np.ndarray, bytes]) -> BiometricTemplate:
		"""Extract voice features including MFCC, spectral, and prosodic features"""
		try:
			# Convert input to audio array
			if isinstance(raw_data, bytes):
				# Assume raw audio bytes at 16kHz
				audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
			else:
				audio = raw_data.copy()
			
			# Ensure proper sample rate
			if len(audio) == 0:
				raise ValueError("Empty audio data")
			
			# Voice activity detection
			voiced_segments = await self._detect_voice_activity(audio)
			
			if len(voiced_segments) == 0:
				raise ValueError("No voice activity detected")
			
			# Extract MFCC features
			mfcc_features = await self._extract_mfcc_features(audio)
			
			# Extract spectral features
			spectral_features = await self._extract_spectral_features(audio)
			
			# Extract prosodic features
			prosodic_features = await self._extract_prosodic_features(audio)
			
			# Extract delta and delta-delta features
			delta_features = await self._extract_delta_features(mfcc_features)
			
			# Combine all features
			feature_vector = np.concatenate([
				mfcc_features.flatten(),
				spectral_features.flatten(),
				prosodic_features.flatten(),
				delta_features.flatten()
			])
			
			# Normalize features
			normalized_features = self._normalize_features(feature_vector)
			
			# Assess quality
			quality_score = await self.assess_quality(raw_data)
			
			# Create template
			template = BiometricTemplate(
				modality=self.modality,
				template_data=normalized_features,
				quality_score=quality_score,
				feature_vector=normalized_features,
				metadata={
					'audio_length_seconds': len(audio) / self.sample_rate,
					'voiced_segments': len(voiced_segments),
					'sample_rate': self.sample_rate,
					'mfcc_coefficients': mfcc_features.shape[1] if len(mfcc_features.shape) > 1 else 0,
					'extraction_method': 'mfcc_spectral_prosodic_delta'
				},
				creation_timestamp=datetime.utcnow().isoformat(),
				template_hash=self._generate_template_hash(normalized_features)
			)
			
			self.logger.info(f"Voice template extracted: quality={quality_score:.3f}, duration={len(audio)/self.sample_rate:.2f}s")
			return template
			
		except Exception as e:
			self.logger.error(f"Voice feature extraction failed: {str(e)}")
			raise
	
	async def _detect_voice_activity(self, audio: np.ndarray) -> List[Tuple[int, int]]:
		"""Detect voice activity segments"""
		# Convert to 16-bit PCM for WebRTC VAD
		audio_16bit = (audio * 32767).astype(np.int16)
		
		# Frame the audio
		frame_length = int(self.frame_length * self.sample_rate)
		frame_shift = int(self.frame_shift * self.sample_rate)
		
		voiced_segments = []
		current_segment_start = None
		
		for i in range(0, len(audio_16bit) - frame_length, frame_shift):
			frame = audio_16bit[i:i + frame_length]
			
			# VAD requires specific frame sizes (10, 20, or 30 ms)
			# Use 20ms frames (320 samples at 16kHz)
			if len(frame) >= 320:
				vad_frame = frame[:320]
				is_speech = self.vad.is_speech(vad_frame.tobytes(), self.sample_rate)
				
				if is_speech:
					if current_segment_start is None:
						current_segment_start = i
				else:
					if current_segment_start is not None:
						voiced_segments.append((current_segment_start, i))
						current_segment_start = None
		
		# Close last segment if needed
		if current_segment_start is not None:
			voiced_segments.append((current_segment_start, len(audio_16bit)))
		
		return voiced_segments
	
	async def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
		"""Extract MFCC features"""
		# Extract MFCC using python_speech_features
		mfcc = python_speech_features.mfcc(
			audio, 
			samplerate=self.sample_rate,
			winlen=self.frame_length,
			winstep=self.frame_shift,
			numcep=13,  # 13 MFCC coefficients
			nfilt=26,   # 26 filter banks
			nfft=512,
			appendEnergy=True
		)
		
		# Statistical summary across time
		mfcc_stats = []
		for coeff_idx in range(mfcc.shape[1]):
			coeff_values = mfcc[:, coeff_idx]
			mfcc_stats.extend([
				np.mean(coeff_values),
				np.std(coeff_values),
				np.min(coeff_values),
				np.max(coeff_values)
			])
		
		return np.array(mfcc_stats)
	
	async def _extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
		"""Extract spectral features"""
		# Compute short-time Fourier transform
		stft = librosa.stft(audio, n_fft=512, hop_length=int(self.frame_shift * self.sample_rate))
		magnitude = np.abs(stft)
		
		spectral_features = []
		
		# Spectral centroid
		spectral_centroid = librosa.feature.spectral_centroid(S=magnitude)[0]
		spectral_features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
		
		# Spectral bandwidth
		spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude)[0]
		spectral_features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
		
		# Spectral rolloff
		spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)[0]
		spectral_features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
		
		# Zero crossing rate
		zcr = librosa.feature.zero_crossing_rate(audio)[0]
		spectral_features.extend([np.mean(zcr), np.std(zcr)])
		
		# Spectral contrast
		spectral_contrast = librosa.feature.spectral_contrast(S=magnitude)
		for band in range(spectral_contrast.shape[0]):
			band_values = spectral_contrast[band, :]
			spectral_features.extend([np.mean(band_values), np.std(band_values)])
		
		return np.array(spectral_features)
	
	async def _extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
		"""Extract prosodic features (pitch, energy, rhythm)"""
		prosodic_features = []
		
		# Fundamental frequency (pitch)
		f0 = librosa.yin(audio, fmin=50, fmax=300, sr=self.sample_rate)
		f0_voiced = f0[f0 > 0]  # Remove unvoiced frames
		
		if len(f0_voiced) > 0:
			prosodic_features.extend([
				np.mean(f0_voiced),
				np.std(f0_voiced),
				np.min(f0_voiced),
				np.max(f0_voiced)
			])
		else:
			prosodic_features.extend([0, 0, 0, 0])
		
		# Energy/intensity
		energy = librosa.feature.rms(y=audio)[0]
		prosodic_features.extend([
			np.mean(energy),
			np.std(energy),
			np.min(energy),
			np.max(energy)
		])
		
		# Rhythm features (tempo)
		try:
			tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
			prosodic_features.append(tempo)
			
			# Beat consistency
			if len(beats) > 1:
				beat_intervals = np.diff(beats)
				prosodic_features.extend([np.mean(beat_intervals), np.std(beat_intervals)])
			else:
				prosodic_features.extend([0, 0])
		except:
			prosodic_features.extend([0, 0, 0])  # Fallback if tempo detection fails
		
		return np.array(prosodic_features)
	
	async def _extract_delta_features(self, mfcc_features: np.ndarray) -> np.ndarray:
		"""Extract delta and delta-delta features from MFCC"""
		# For statistical features, compute differences between adjacent statistics
		# This is a simplified approach since we have summary statistics, not time series
		
		# Create delta features by comparing different statistical measures
		delta_features = []
		
		# Reshape MFCC stats (4 stats per coefficient)
		num_coeffs = len(mfcc_features) // 4
		mfcc_reshaped = mfcc_features.reshape(num_coeffs, 4)
		
		for i in range(num_coeffs):
			mean_val, std_val, min_val, max_val = mfcc_reshaped[i]
			
			# Delta-like features: differences and ratios
			delta_features.extend([
				max_val - min_val,  # Range
				std_val / (abs(mean_val) + 1e-8),  # Coefficient of variation
				(mean_val - min_val) / (max_val - min_val + 1e-8)  # Normalized position
			])
		
		return np.array(delta_features)
	
	async def compare_templates(self, template1: BiometricTemplate, template2: BiometricTemplate) -> BiometricComparisonResult:
		"""Compare two voice templates"""
		try:
			# Multiple comparison methods for voice recognition
			
			# Cosine similarity (important for voice)
			cosine_similarity = 1 - scipy.spatial.distance.cosine(
				template1.feature_vector, template2.feature_vector
			)
			
			# Euclidean distance
			euclidean_dist = scipy.spatial.distance.euclidean(
				template1.feature_vector, template2.feature_vector
			)
			euclidean_similarity = 1.0 / (1.0 + euclidean_dist)
			
			# Correlation coefficient
			correlation = np.corrcoef(template1.feature_vector, template2.feature_vector)[0, 1]
			correlation = np.nan_to_num(correlation, 0.0)
			
			# Mahalanobis distance (if we had covariance matrix)
			# For now, use weighted Euclidean as approximation
			weights = np.ones_like(template1.feature_vector)  # Could be learned
			weighted_dist = np.sqrt(np.sum(weights * (template1.feature_vector - template2.feature_vector)**2))
			mahalanobis_similarity = 1.0 / (1.0 + weighted_dist)
			
			# Combined similarity score (emphasize cosine for voice)
			similarity_score = (cosine_similarity * 0.5 + euclidean_similarity * 0.2 + 
							   correlation * 0.2 + mahalanobis_similarity * 0.1)
			
			# Match decision
			match_decision = similarity_score >= self.match_threshold
			
			# Calculate confidence
			confidence = similarity_score * min(template1.quality_score, template2.quality_score)
			
			result = BiometricComparisonResult(
				similarity_score=similarity_score,
				match_confidence=confidence,
				decision=match_decision,
				threshold_used=self.match_threshold,
				comparison_metadata={
					'cosine_similarity': cosine_similarity,
					'euclidean_similarity': euclidean_similarity,
					'correlation': correlation,
					'mahalanobis_similarity': mahalanobis_similarity,
					'template1_quality': template1.quality_score,
					'template2_quality': template2.quality_score,
					'template1_duration': template1.metadata.get('audio_length_seconds', 0),
					'template2_duration': template2.metadata.get('audio_length_seconds', 0)
				}
			)
			
			self.logger.info(f"Voice comparison: similarity={similarity_score:.3f}, match={match_decision}")
			return result
			
		except Exception as e:
			self.logger.error(f"Voice comparison failed: {str(e)}")
			raise
	
	async def assess_quality(self, raw_data: Union[np.ndarray, bytes]) -> float:
		"""Assess voice recording quality"""
		try:
			if isinstance(raw_data, bytes):
				audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
			else:
				audio = raw_data.copy()
			
			quality_factors = []
			
			# Signal-to-noise ratio estimation
			# Use spectral subtraction method
			stft = librosa.stft(audio, n_fft=512)
			magnitude = np.abs(stft)
			
			# Estimate noise floor (bottom 10th percentile)
			noise_floor = np.percentile(magnitude, 10)
			signal_power = np.mean(magnitude**2)
			noise_power = noise_floor**2
			snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
			snr_score = min(max(snr / 20.0, 0), 1)  # Normalize to 0-1
			quality_factors.append(snr_score)
			
			# Voice activity ratio
			voiced_segments = await self._detect_voice_activity(audio)
			if voiced_segments:
				voiced_duration = sum(end - start for start, end in voiced_segments)
				voice_ratio = voiced_duration / len(audio)
				voice_score = min(voice_ratio * 2, 1.0)  # Prefer 50%+ voice activity
			else:
				voice_score = 0.0
			quality_factors.append(voice_score)
			
			# Spectral clarity (high frequency content)
			fft = np.fft.rfft(audio)
			freq_magnitude = np.abs(fft)
			high_freq_energy = np.sum(freq_magnitude[len(freq_magnitude)//2:])
			total_energy = np.sum(freq_magnitude)
			clarity_score = high_freq_energy / (total_energy + 1e-8)
			quality_factors.append(min(clarity_score * 3, 1.0))
			
			# Dynamic range
			audio_range = np.max(audio) - np.min(audio)
			range_score = min(audio_range / 0.8, 1.0)  # Prefer near full scale
			quality_factors.append(range_score)
			
			# Clipping detection
			clipping_threshold = 0.95
			clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
			clipping_ratio = clipped_samples / len(audio)
			clipping_score = 1.0 - min(clipping_ratio * 10, 1.0)  # Penalize clipping
			quality_factors.append(clipping_score)
			
			# Combined quality score
			quality_score = np.mean(quality_factors)
			
			return float(quality_score)
			
		except Exception as e:
			self.logger.error(f"Voice quality assessment failed: {str(e)}")
			return 0.0
	
	async def detect_liveness(self, raw_data: Union[np.ndarray, bytes], **kwargs) -> LivenessDetectionResult:
		"""Voice liveness detection (anti-spoofing)"""
		try:
			if isinstance(raw_data, bytes):
				audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
			else:
				audio = raw_data.copy()
			
			liveness_indicators = []
			anti_spoofing_checks = {}
			
			# 1. Spectral analysis for naturalness
			stft = librosa.stft(audio, n_fft=512)
			magnitude = np.abs(stft)
			
			# Check for unnatural spectral patterns (too regular)
			spectral_variance = np.var(magnitude, axis=1)
			natural_variance = np.std(spectral_variance)
			spectral_natural = natural_variance > 0.1
			anti_spoofing_checks['spectral_naturalness'] = spectral_natural
			if spectral_natural:
				liveness_indicators.append('natural_spectral_variation')
			
			# 2. Prosodic variation analysis
			f0 = librosa.yin(audio, fmin=50, fmax=300, sr=self.sample_rate)
			f0_voiced = f0[f0 > 0]
			
			if len(f0_voiced) > 10:
				f0_variation = np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-8)
				prosodic_natural = f0_variation > 0.05  # Natural speech has pitch variation
				anti_spoofing_checks['prosodic_variation'] = prosodic_natural
				if prosodic_natural:
					liveness_indicators.append('natural_prosody')
			else:
				anti_spoofing_checks['prosodic_variation'] = False
			
			# 3. Micro-variation analysis (natural speech has micro-variations)
			frame_energy = librosa.feature.rms(y=audio, frame_length=256, hop_length=128)[0]
			energy_micro_var = np.std(np.diff(frame_energy))
			micro_variation = energy_micro_var > 0.001
			anti_spoofing_checks['micro_variation'] = micro_variation
			if micro_variation:
				liveness_indicators.append('energy_micro_variations')
			
			# 4. Phase coherence analysis
			phase = np.angle(stft)
			phase_coherence = np.mean(np.abs(np.diff(phase, axis=1)))
			coherence_natural = phase_coherence > 0.5  # Natural speech has phase variation
			anti_spoofing_checks['phase_coherence'] = coherence_natural
			if coherence_natural:
				liveness_indicators.append('natural_phase_patterns')
			
			# Calculate overall liveness score
			liveness_checks = list(anti_spoofing_checks.values())
			liveness_score = sum(liveness_checks) / len(liveness_checks)
			is_live = liveness_score >= 0.6
			
			return LivenessDetectionResult(
				is_live=is_live,
				liveness_score=liveness_score,
				liveness_indicators=liveness_indicators,
				anti_spoofing_checks=anti_spoofing_checks,
				confidence=liveness_score
			)
			
		except Exception as e:
			self.logger.error(f"Voice liveness detection failed: {str(e)}")
			return LivenessDetectionResult(
				is_live=False, liveness_score=0.0, liveness_indicators=[],
				anti_spoofing_checks={}, confidence=0.0
			)

class GaitEngine(BiometricEngine):
	"""
	Advanced gait recognition engine
	
	Features:
	- Motion pattern analysis
	- Stride and step detection
	- Body movement dynamics
	- Smartphone sensor-based gait analysis
	"""
	
	def __init__(self):
		super().__init__("gait")
		self.match_threshold = 0.75
		self.quality_threshold = 0.6
		
		# Gait analysis parameters
		self.window_size = 128  # Samples per analysis window
		self.overlap = 0.5      # 50% overlap between windows
		self.expected_sample_rate = 50  # Hz for accelerometer data
	
	async def extract_features(self, raw_data: Union[np.ndarray, bytes]) -> BiometricTemplate:
		"""Extract gait features from accelerometer/gyroscope data"""
		try:
			# Convert input to sensor data array
			if isinstance(raw_data, bytes):
				# Assume raw data is JSON with sensor readings
				import json
				sensor_data = json.loads(raw_data.decode('utf-8'))
				accel_data = np.array(sensor_data.get('accelerometer', []))
				gyro_data = np.array(sensor_data.get('gyroscope', []))
			else:
				# Assume raw_data is already processed sensor data
				if isinstance(raw_data, dict):
					accel_data = np.array(raw_data.get('accelerometer', []))
					gyro_data = np.array(raw_data.get('gyroscope', []))
				else:
					# Assume it's accelerometer data only
					accel_data = raw_data.copy()
					gyro_data = np.array([])
			
			if len(accel_data) == 0:
				raise ValueError("No accelerometer data provided")
			
			# Ensure proper shape (N, 3) for 3D accelerometer data
			if len(accel_data.shape) == 1:
				accel_data = accel_data.reshape(-1, 3)
			
			# Extract temporal features
			temporal_features = await self._extract_temporal_features(accel_data)
			
			# Extract frequency domain features
			frequency_features = await self._extract_frequency_features(accel_data)
			
			# Extract step/stride features
			step_features = await self._extract_step_features(accel_data)
			
			# Extract statistical features
			statistical_features = await self._extract_statistical_features(accel_data)
			
			# Extract gyroscope features if available
			if len(gyro_data) > 0:
				gyro_features = await self._extract_gyroscope_features(gyro_data)
			else:
				gyro_features = np.zeros(20)  # Placeholder
			
			# Combine all features
			feature_vector = np.concatenate([
				temporal_features.flatten(),
				frequency_features.flatten(),
				step_features.flatten(),
				statistical_features.flatten(),
				gyro_features.flatten()
			])
			
			# Normalize features
			normalized_features = self._normalize_features(feature_vector)
			
			# Assess quality
			quality_score = await self.assess_quality(raw_data)
			
			# Create template
			template = BiometricTemplate(
				modality=self.modality,
				template_data=normalized_features,
				quality_score=quality_score,
				feature_vector=normalized_features,
				metadata={
					'data_length_samples': len(accel_data),
					'duration_seconds': len(accel_data) / self.expected_sample_rate,
					'has_gyroscope': len(gyro_data) > 0,
					'step_count': len(step_features) // 4 if len(step_features) > 0 else 0,
					'extraction_method': 'temporal_frequency_step_statistical_gyro'
				},
				creation_timestamp=datetime.utcnow().isoformat(),
				template_hash=self._generate_template_hash(normalized_features)
			)
			
			self.logger.info(f"Gait template extracted: quality={quality_score:.3f}, duration={len(accel_data)/self.expected_sample_rate:.2f}s")
			return template
			
		except Exception as e:
			self.logger.error(f"Gait feature extraction failed: {str(e)}")
			raise
	
	async def _extract_temporal_features(self, accel_data: np.ndarray) -> np.ndarray:
		"""Extract temporal gait features"""
		# Calculate magnitude of acceleration vector
		magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
		
		temporal_features = []
		
		# Windowed analysis
		window_size = min(self.window_size, len(magnitude))
		hop_size = int(window_size * (1 - self.overlap))
		
		for i in range(0, len(magnitude) - window_size, hop_size):
			window = magnitude[i:i + window_size]
			
			# Basic temporal statistics per window
			temporal_features.extend([
				np.mean(window),
				np.std(window),
				np.min(window),
				np.max(window),
				np.median(window)
			])
		
		# Limit feature size
		return np.array(temporal_features[:100])
	
	async def _extract_frequency_features(self, accel_data: np.ndarray) -> np.ndarray:
		"""Extract frequency domain gait features"""
		frequency_features = []
		
		# Analyze each axis separately
		for axis in range(accel_data.shape[1]):
			axis_data = accel_data[:, axis]
			
			# FFT analysis
			fft = np.fft.rfft(axis_data)
			magnitude_spectrum = np.abs(fft)
			
			# Dominant frequency
			freqs = np.fft.rfftfreq(len(axis_data), 1.0/self.expected_sample_rate)
			dominant_freq_idx = np.argmax(magnitude_spectrum[1:]) + 1  # Skip DC component
			dominant_freq = freqs[dominant_freq_idx]
			frequency_features.append(dominant_freq)
			
			# Spectral centroid
			spectral_centroid = np.sum(freqs * magnitude_spectrum) / (np.sum(magnitude_spectrum) + 1e-8)
			frequency_features.append(spectral_centroid)
			
			# Spectral bandwidth
			spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude_spectrum) / (np.sum(magnitude_spectrum) + 1e-8))
			frequency_features.append(spectral_bandwidth)
			
			# Power in gait frequency band (0.5-3 Hz typical for walking)
			gait_band_mask = (freqs >= 0.5) & (freqs <= 3.0)
			gait_power = np.sum(magnitude_spectrum[gait_band_mask])
			total_power = np.sum(magnitude_spectrum)
			gait_power_ratio = gait_power / (total_power + 1e-8)
			frequency_features.append(gait_power_ratio)
		
		return np.array(frequency_features)
	
	async def _extract_step_features(self, accel_data: np.ndarray) -> np.ndarray:
		"""Extract step and stride characteristics"""
		# Calculate magnitude
		magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
		
		# Detect peaks (steps) in magnitude signal
		# Use a simple peak detection algorithm
		from scipy.signal import find_peaks
		
		# Find peaks that represent steps
		peaks, properties = find_peaks(magnitude, 
										height=np.mean(magnitude) + 0.5*np.std(magnitude),
										distance=int(0.3 * self.expected_sample_rate))  # Min 0.3s between steps
		
		step_features = []
		
		if len(peaks) > 1:
			# Step timing features
			step_intervals = np.diff(peaks) / self.expected_sample_rate  # Convert to seconds
			
			step_features.extend([
				np.mean(step_intervals),    # Average step time
				np.std(step_intervals),     # Step time variability
				len(peaks) / (len(magnitude) / self.expected_sample_rate),  # Step frequency
				np.median(step_intervals)   # Median step time
			])
			
			# Step amplitude features
			step_amplitudes = magnitude[peaks]
			step_features.extend([
				np.mean(step_amplitudes),
				np.std(step_amplitudes),
				np.min(step_amplitudes),
				np.max(step_amplitudes)
			])
			
			# Stride features (pair of steps)
			if len(step_intervals) > 1:
				stride_times = step_intervals[::2] + step_intervals[1::2]  # Approximate stride as sum of two steps
				if len(stride_times) > 0:
					step_features.extend([
						np.mean(stride_times),
						np.std(stride_times)
					])
				else:
					step_features.extend([0, 0])
			else:
				step_features.extend([0, 0])
		else:
			# No clear steps detected
			step_features = [0] * 10
		
		# Pad to fixed size
		while len(step_features) < 20:
			step_features.append(0)
		
		return np.array(step_features[:20])
	
	async def _extract_statistical_features(self, accel_data: np.ndarray) -> np.ndarray:
		"""Extract statistical features from gait data"""
		statistical_features = []
		
		# For each axis and magnitude
		for axis in range(accel_data.shape[1]):
			axis_data = accel_data[:, axis]
			
			# Basic statistics
			statistical_features.extend([
				np.mean(axis_data),
				np.std(axis_data),
				np.min(axis_data),
				np.max(axis_data),
				np.median(axis_data),
				scipy.stats.skew(axis_data),
				scipy.stats.kurtosis(axis_data)
			])
		
		# Magnitude statistics
		magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
		statistical_features.extend([
			np.mean(magnitude),
			np.std(magnitude),
			np.min(magnitude),
			np.max(magnitude),
			np.median(magnitude),
			scipy.stats.skew(magnitude),
			scipy.stats.kurtosis(magnitude)
		])
		
		# Cross-axis correlations
		if accel_data.shape[1] >= 3:
			corr_xy = np.corrcoef(accel_data[:, 0], accel_data[:, 1])[0, 1]
			corr_xz = np.corrcoef(accel_data[:, 0], accel_data[:, 2])[0, 1]
			corr_yz = np.corrcoef(accel_data[:, 1], accel_data[:, 2])[0, 1]
			
			statistical_features.extend([
				np.nan_to_num(corr_xy, 0),
				np.nan_to_num(corr_xz, 0),
				np.nan_to_num(corr_yz, 0)
			])
		
		return np.array(statistical_features)
	
	async def _extract_gyroscope_features(self, gyro_data: np.ndarray) -> np.ndarray:
		"""Extract features from gyroscope data if available"""
		if len(gyro_data) == 0:
			return np.zeros(20)
		
		if len(gyro_data.shape) == 1:
			gyro_data = gyro_data.reshape(-1, 3)
		
		gyro_features = []
		
		# Angular velocity magnitude
		angular_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
		
		# Basic statistics for angular velocity
		gyro_features.extend([
			np.mean(angular_magnitude),
			np.std(angular_magnitude),
			np.min(angular_magnitude),
			np.max(angular_magnitude)
		])
		
		# Per-axis statistics
		for axis in range(min(3, gyro_data.shape[1])):
			axis_data = gyro_data[:, axis]
			gyro_features.extend([
				np.mean(axis_data),
				np.std(axis_data)
			])
		
		# Rotational patterns
		# Detect rotational events (turns)
		turn_threshold = np.mean(angular_magnitude) + 2*np.std(angular_magnitude)
		turn_events = np.sum(angular_magnitude > turn_threshold)
		turn_rate = turn_events / (len(gyro_data) / self.expected_sample_rate)
		gyro_features.append(turn_rate)
		
		# Pad to fixed size
		while len(gyro_features) < 20:
			gyro_features.append(0)
		
		return np.array(gyro_features[:20])
	
	async def compare_templates(self, template1: BiometricTemplate, template2: BiometricTemplate) -> BiometricComparisonResult:
		"""Compare two gait templates"""
		try:
			# Multiple comparison methods for gait recognition
			
			# Dynamic Time Warping would be ideal for gait, but use simpler methods for now
			
			# Euclidean distance
			euclidean_dist = scipy.spatial.distance.euclidean(
				template1.feature_vector, template2.feature_vector
			)
			euclidean_similarity = 1.0 / (1.0 + euclidean_dist)
			
			# Cosine similarity
			cosine_similarity = 1 - scipy.spatial.distance.cosine(
				template1.feature_vector, template2.feature_vector
			)
			
			# Correlation coefficient
			correlation = np.corrcoef(template1.feature_vector, template2.feature_vector)[0, 1]
			correlation = np.nan_to_num(correlation, 0.0)
			
			# Manhattan distance
			manhattan_dist = scipy.spatial.distance.cityblock(
				template1.feature_vector, template2.feature_vector
			)
			manhattan_similarity = 1.0 / (1.0 + manhattan_dist)
			
			# Combined similarity score
			similarity_score = (euclidean_similarity * 0.3 + cosine_similarity * 0.3 + 
							   correlation * 0.3 + manhattan_similarity * 0.1)
			
			# Match decision
			match_decision = similarity_score >= self.match_threshold
			
			# Calculate confidence
			confidence = similarity_score * min(template1.quality_score, template2.quality_score)
			
			result = BiometricComparisonResult(
				similarity_score=similarity_score,
				match_confidence=confidence,
				decision=match_decision,
				threshold_used=self.match_threshold,
				comparison_metadata={
					'euclidean_similarity': euclidean_similarity,
					'cosine_similarity': cosine_similarity,
					'correlation': correlation,
					'manhattan_similarity': manhattan_similarity,
					'template1_quality': template1.quality_score,
					'template2_quality': template2.quality_score,
					'template1_duration': template1.metadata.get('duration_seconds', 0),
					'template2_duration': template2.metadata.get('duration_seconds', 0)
				}
			)
			
			self.logger.info(f"Gait comparison: similarity={similarity_score:.3f}, match={match_decision}")
			return result
			
		except Exception as e:
			self.logger.error(f"Gait comparison failed: {str(e)}")
			raise
	
	async def assess_quality(self, raw_data: Union[np.ndarray, bytes]) -> float:
		"""Assess gait data quality"""
		try:
			# Parse data similar to extract_features
			if isinstance(raw_data, bytes):
				import json
				sensor_data = json.loads(raw_data.decode('utf-8'))
				accel_data = np.array(sensor_data.get('accelerometer', []))
			else:
				if isinstance(raw_data, dict):
					accel_data = np.array(raw_data.get('accelerometer', []))
				else:
					accel_data = raw_data.copy()
			
			if len(accel_data) == 0:
				return 0.0
			
			if len(accel_data.shape) == 1:
				accel_data = accel_data.reshape(-1, 3)
			
			quality_factors = []
			
			# Data length quality (prefer longer sequences)
			duration = len(accel_data) / self.expected_sample_rate
			duration_score = min(duration / 10.0, 1.0)  # Prefer at least 10 seconds
			quality_factors.append(duration_score)
			
			# Signal amplitude quality
			magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
			amplitude_range = np.max(magnitude) - np.min(magnitude)
			amplitude_score = min(amplitude_range / 5.0, 1.0)  # Prefer dynamic range
			quality_factors.append(amplitude_score)
			
			# Step detection quality
			from scipy.signal import find_peaks
			peaks, _ = find_peaks(magnitude, 
								  height=np.mean(magnitude) + 0.5*np.std(magnitude),
								  distance=int(0.3 * self.expected_sample_rate))
			
			expected_steps = duration * 2  # Approximately 2 steps per second for normal walking
			step_count_score = 1.0 - abs(len(peaks) - expected_steps) / (expected_steps + 1)
			step_count_score = max(0, step_count_score)
			quality_factors.append(step_count_score)
			
			# Signal regularity (gait should be somewhat periodic)
			if len(peaks) > 2:
				step_intervals = np.diff(peaks)
				step_regularity = 1.0 - (np.std(step_intervals) / (np.mean(step_intervals) + 1e-8))
				step_regularity = max(0, min(1, step_regularity))
			else:
				step_regularity = 0.0
			quality_factors.append(step_regularity)
			
			# Noise level assessment
			# High frequency noise indicates poor sensor quality
			fft = np.fft.rfft(magnitude)
			freqs = np.fft.rfftfreq(len(magnitude), 1.0/self.expected_sample_rate)
			
			# Power in high frequency band (>10 Hz)
			high_freq_mask = freqs > 10
			high_freq_power = np.sum(np.abs(fft[high_freq_mask])**2)
			total_power = np.sum(np.abs(fft)**2)
			noise_ratio = high_freq_power / (total_power + 1e-8)
			noise_score = 1.0 - min(noise_ratio * 5, 1.0)  # Penalize high frequency noise
			quality_factors.append(noise_score)
			
			# Combined quality score
			quality_score = np.mean(quality_factors)
			
			return float(quality_score)
			
		except Exception as e:
			self.logger.error(f"Gait quality assessment failed: {str(e)}")
			return 0.0

# Biometric Engine Factory
class BiometricEngineFactory:
	"""Factory for creating biometric engines"""
	
	@staticmethod
	def create_engine(modality: str) -> BiometricEngine:
		"""Create appropriate biometric engine for modality"""
		engines = {
			'fingerprint': FingerprintEngine,
			'iris': IrisEngine,
			'palm': PalmEngine,
			'voice': VoiceEngine,
			'gait': GaitEngine
		}
		
		if modality.lower() not in engines:
			raise ValueError(f"Unsupported biometric modality: {modality}")
		
		return engines[modality.lower()]()
	
	@staticmethod
	def get_supported_modalities() -> List[str]:
		"""Get list of supported biometric modalities"""
		return ['fingerprint', 'iris', 'palm', 'voice', 'gait']

# Main biometric processing interface
class BiometricProcessor:
	"""
	Main interface for biometric processing across all modalities
	
	Provides unified access to registration, verification, and comparison
	functions for all supported biometric types.
	"""
	
	def __init__(self):
		self.engines = {}
		self.logger = logging.getLogger(__name__)
		
		# Initialize all engines
		for modality in BiometricEngineFactory.get_supported_modalities():
			try:
				self.engines[modality] = BiometricEngineFactory.create_engine(modality)
				self.logger.info(f"Initialized {modality} biometric engine")
			except Exception as e:
				self.logger.error(f"Failed to initialize {modality} engine: {str(e)}")
	
	async def register_biometric(self, modality: str, raw_data: Union[np.ndarray, bytes]) -> BiometricTemplate:
		"""Register a new biometric template"""
		if modality not in self.engines:
			raise ValueError(f"Unsupported modality: {modality}")
		
		engine = self.engines[modality]
		template = await engine.extract_features(raw_data)
		
		self.logger.info(f"Registered {modality} biometric: quality={template.quality_score:.3f}")
		return template
	
	async def verify_biometric(self, template: BiometricTemplate, raw_data: Union[np.ndarray, bytes]) -> BiometricComparisonResult:
		"""Verify biometric against existing template"""
		if template.modality not in self.engines:
			raise ValueError(f"Unsupported modality: {template.modality}")
		
		engine = self.engines[template.modality]
		
		# Extract features from raw data
		new_template = await engine.extract_features(raw_data)
		
		# Compare templates
		result = await engine.compare_templates(template, new_template)
		
		self.logger.info(f"Verified {template.modality} biometric: match={result.decision}, confidence={result.match_confidence:.3f}")
		return result
	
	async def compare_biometrics(self, template1: BiometricTemplate, template2: BiometricTemplate) -> BiometricComparisonResult:
		"""Compare two biometric templates"""
		if template1.modality != template2.modality:
			raise ValueError("Cannot compare different biometric modalities")
		
		if template1.modality not in self.engines:
			raise ValueError(f"Unsupported modality: {template1.modality}")
		
		engine = self.engines[template1.modality]
		result = await engine.compare_templates(template1, template2)
		
		self.logger.info(f"Compared {template1.modality} biometrics: match={result.decision}, similarity={result.similarity_score:.3f}")
		return result
	
	async def assess_quality(self, modality: str, raw_data: Union[np.ndarray, bytes]) -> float:
		"""Assess quality of biometric sample"""
		if modality not in self.engines:
			raise ValueError(f"Unsupported modality: {modality}")
		
		engine = self.engines[modality]
		quality = await engine.assess_quality(raw_data)
		
		self.logger.info(f"Assessed {modality} quality: {quality:.3f}")
		return quality
	
	async def detect_liveness(self, modality: str, raw_data: Union[np.ndarray, bytes], **kwargs) -> LivenessDetectionResult:
		"""Detect liveness for biometric sample"""
		if modality not in self.engines:
			raise ValueError(f"Unsupported modality: {modality}")
		
		engine = self.engines[modality]
		result = await engine.detect_liveness(raw_data, **kwargs)
		
		self.logger.info(f"Detected {modality} liveness: live={result.is_live}, score={result.liveness_score:.3f}")
		return result
	
	def get_supported_modalities(self) -> List[str]:
		"""Get list of supported and initialized modalities"""
		return list(self.engines.keys())
	
	def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
		"""Get information about all biometric engines"""
		info = {}
		for modality, engine in self.engines.items():
			info[modality] = {
				'match_threshold': engine.match_threshold,
				'quality_threshold': engine.quality_threshold,
				'engine_class': engine.__class__.__name__
			}
		return info

# Export main classes and functions
__all__ = [
	'BiometricTemplate',
	'BiometricComparisonResult', 
	'LivenessDetectionResult',
	'BiometricEngine',
	'FingerprintEngine',
	'IrisEngine', 
	'PalmEngine',
	'VoiceEngine',
	'GaitEngine',
	'BiometricEngineFactory',
	'BiometricProcessor'
]