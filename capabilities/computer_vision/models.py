"""
Computer Vision Models

Database models for comprehensive computer vision processing including
object detection, image analysis, video processing, and ML model management.
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


class CVImage(Model, AuditMixin, BaseMixin):
	"""
	Image records for computer vision processing.
	
	Stores metadata about images including format, dimensions,
	and processing history with quality metrics.
	"""
	__tablename__ = 'cv_image'
	
	# Identity
	image_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Image Information
	original_filename = Column(String(500), nullable=False)
	file_path = Column(String(1000), nullable=False)
	file_size_bytes = Column(Integer, nullable=False)
	mime_type = Column(String(100), nullable=False)
	file_hash = Column(String(64), nullable=False, index=True)  # SHA-256 for deduplication
	
	# Image Properties
	width = Column(Integer, nullable=False)
	height = Column(Integer, nullable=False)
	channels = Column(Integer, nullable=False)  # 1=grayscale, 3=RGB, 4=RGBA
	color_space = Column(String(20), default='RGB')  # RGB, BGR, HSV, LAB, etc.
	bit_depth = Column(Integer, default=8)  # bits per channel
	
	# Image Quality Metrics
	brightness = Column(Float, nullable=True)  # Average brightness 0-255
	contrast = Column(Float, nullable=True)  # Standard deviation of pixel values
	sharpness_score = Column(Float, nullable=True)  # Laplacian variance
	noise_level = Column(Float, nullable=True)  # Estimated noise level
	blur_score = Column(Float, nullable=True)  # Blur detection score
	
	# Color Analysis
	dominant_colors = Column(JSON, default=list)  # RGB values of dominant colors
	color_histogram = Column(JSON, default=dict)  # Color distribution
	color_temperature = Column(Float, nullable=True)  # Kelvin
	
	# Content Classification
	scene_type = Column(String(50), nullable=True)  # indoor, outdoor, portrait, landscape
	content_tags = Column(JSON, default=list)  # Auto-generated content tags
	complexity_score = Column(Float, default=0.0)  # 0-100, based on edges and textures
	
	# Processing Status
	processing_status = Column(String(20), default='uploaded', index=True)  # uploaded, processing, completed, failed
	upload_source = Column(String(50), nullable=True)  # web, api, mobile, batch
	uploaded_by = Column(String(36), nullable=True, index=True)  # User ID
	
	# Processing History
	total_detections = Column(Integer, default=0)
	last_processed = Column(DateTime, nullable=True, index=True)
	processing_count = Column(Integer, default=0)
	
	# Storage and Lifecycle
	storage_class = Column(String(20), default='standard')  # standard, cold, archive
	retention_policy = Column(String(50), nullable=True)
	expires_at = Column(DateTime, nullable=True, index=True)
	
	# Privacy and Compliance
	contains_faces = Column(Boolean, default=False, index=True)
	contains_pii = Column(Boolean, nullable=True)
	privacy_level = Column(String(20), default='internal')  # public, internal, confidential, restricted
	consent_obtained = Column(Boolean, default=False)
	
	# Relationships
	detections = relationship("CVDetection", back_populates="image", cascade="all, delete-orphan")
	analyses = relationship("CVImageAnalysis", back_populates="image", cascade="all, delete-orphan")
	enhancements = relationship("CVImageEnhancement", back_populates="image", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CVImage {self.original_filename} ({self.width}x{self.height})>"
	
	def get_aspect_ratio(self) -> float:
		"""Get image aspect ratio"""
		return self.width / self.height if self.height > 0 else 1.0
	
	def get_megapixels(self) -> float:
		"""Get image size in megapixels"""
		return (self.width * self.height) / 1_000_000
	
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
	
	def calculate_quality_score(self) -> float:
		"""Calculate overall image quality score (0-100)"""
		score = 0.0
		factors = 0
		
		# Factor in sharpness
		if self.sharpness_score is not None:
			score += min(100, self.sharpness_score / 100) * 30  # 30% weight
			factors += 30
		
		# Factor in noise (inverse - less noise is better)
		if self.noise_level is not None:
			noise_score = max(0, 100 - self.noise_level)
			score += noise_score * 25  # 25% weight
			factors += 25
		
		# Factor in blur (inverse - less blur is better)
		if self.blur_score is not None:
			blur_score = max(0, 100 - self.blur_score)
			score += blur_score * 25  # 25% weight
			factors += 25
		
		# Factor in contrast
		if self.contrast is not None:
			contrast_score = min(100, self.contrast / 50 * 100)  # Normalize to 0-100
			score += contrast_score * 20  # 20% weight
			factors += 20
		
		return score / factors * 100 if factors > 0 else 0.0
	
	def is_high_resolution(self) -> bool:
		"""Check if image is high resolution (>= 2MP)"""
		return self.get_megapixels() >= 2.0
	
	def update_processing_stats(self) -> None:
		"""Update processing statistics"""
		self.processing_count += 1
		self.last_processed = datetime.utcnow()
		self.total_detections = len(self.detections)


class CVDetection(Model, AuditMixin, BaseMixin):
	"""
	Object detection results with detailed metrics and confidence scores.
	
	Records individual detected objects with bounding boxes,
	classification confidence, and additional attributes.
	"""
	__tablename__ = 'cv_detection'
	
	# Identity
	detection_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	image_id = Column(String(36), ForeignKey('cv_image.image_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Detection Results
	object_type = Column(String(50), nullable=False, index=True)  # face, person, vehicle, object
	object_class = Column(String(100), nullable=True)  # Specific class like 'car', 'bicycle'
	confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0
	
	# Bounding Box (in pixels)
	bbox_x = Column(Integer, nullable=False)  # Top-left X coordinate
	bbox_y = Column(Integer, nullable=False)  # Top-left Y coordinate
	bbox_width = Column(Integer, nullable=False)  # Width
	bbox_height = Column(Integer, nullable=False)  # Height
	
	# Center Point
	center_x = Column(Integer, nullable=False)
	center_y = Column(Integer, nullable=False)
	
	# Detection Method and Model
	detection_method = Column(String(50), nullable=False)  # haar, hog, yolo, ssd, etc.
	model_name = Column(String(100), nullable=True)  # Specific model used
	model_version = Column(String(20), nullable=True)
	
	# Detection Quality Metrics
	bbox_area = Column(Integer, nullable=False)  # Bounding box area in pixels
	bbox_aspect_ratio = Column(Float, nullable=False)  # Width/Height ratio
	relative_size = Column(Float, nullable=False)  # Size relative to image (0-1)
	edge_proximity = Column(Float, nullable=True)  # Distance to image edges
	
	# Object Attributes (JSON)
	attributes = Column(JSON, default=dict)  # Object-specific attributes
	features = Column(JSON, default=dict)  # Extracted features
	metadata = Column(JSON, default=dict)  # Additional metadata
	
	# Processing Context
	processing_time_ms = Column(Float, nullable=True)
	batch_id = Column(String(36), nullable=True)  # For batch processing
	
	# Verification and Validation
	is_verified = Column(Boolean, default=False)  # Human verification
	verification_confidence = Column(Float, nullable=True)  # Human confidence
	is_false_positive = Column(Boolean, default=False)
	
	# Relationships
	image = relationship("CVImage", back_populates="detections")
	
	def __repr__(self):
		return f"<CVDetection {self.object_type} conf={self.confidence_score:.2f}>"
	
	def get_bbox_dict(self) -> Dict[str, int]:
		"""Get bounding box as dictionary"""
		return {
			'x': self.bbox_x,
			'y': self.bbox_y,
			'width': self.bbox_width,
			'height': self.bbox_height
		}
	
	def get_center_dict(self) -> Dict[str, int]:
		"""Get center point as dictionary"""
		return {
			'x': self.center_x,
			'y': self.center_y
		}
	
	def is_high_confidence(self, threshold: float = 0.8) -> bool:
		"""Check if detection has high confidence"""
		return self.confidence_score >= threshold
	
	def is_large_object(self, threshold: float = 0.1) -> bool:
		"""Check if detected object is large relative to image"""
		return self.relative_size >= threshold
	
	def overlaps_with(self, other_detection: 'CVDetection', threshold: float = 0.5) -> bool:
		"""Check if this detection overlaps with another"""
		# Calculate intersection area
		x1 = max(self.bbox_x, other_detection.bbox_x)
		y1 = max(self.bbox_y, other_detection.bbox_y)
		x2 = min(self.bbox_x + self.bbox_width, other_detection.bbox_x + other_detection.bbox_width)
		y2 = min(self.bbox_y + self.bbox_height, other_detection.bbox_y + other_detection.bbox_height)
		
		if x2 <= x1 or y2 <= y1:
			return False  # No intersection
		
		intersection_area = (x2 - x1) * (y2 - y1)
		union_area = self.bbox_area + other_detection.bbox_area - intersection_area
		
		iou = intersection_area / union_area if union_area > 0 else 0
		return iou >= threshold


class CVImageAnalysis(Model, AuditMixin, BaseMixin):
	"""
	Comprehensive image analysis results including features and insights.
	
	Stores detailed analysis results for images including color analysis,
	texture features, scene classification, and quality assessment.
	"""
	__tablename__ = 'cv_image_analysis'
	
	# Identity
	analysis_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	image_id = Column(String(36), ForeignKey('cv_image.image_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Analysis Configuration
	analysis_type = Column(String(50), nullable=False, index=True)  # quality, content, features, aesthetic
	analysis_version = Column(String(20), default='1.0')
	parameters = Column(JSON, default=dict)  # Analysis parameters used
	
	# Feature Extraction Results
	color_features = Column(JSON, default=dict)  # Color-based features
	texture_features = Column(JSON, default=dict)  # Texture analysis results
	shape_features = Column(JSON, default=dict)  # Shape and edge features
	spatial_features = Column(JSON, default=dict)  # Spatial distribution
	
	# Content Classification
	scene_classification = Column(JSON, default=dict)  # Scene type probabilities
	object_categories = Column(JSON, default=list)  # Detected object categories
	content_tags = Column(JSON, default=list)  # Generated content tags
	aesthetic_score = Column(Float, nullable=True)  # 0-100 aesthetic rating
	
	# Quality Assessment
	technical_quality = Column(JSON, default=dict)  # Technical quality metrics
	visual_quality = Column(JSON, default=dict)  # Perceptual quality metrics
	overall_quality_score = Column(Float, default=0.0)  # 0-100 overall quality
	
	# Statistical Analysis
	histogram_data = Column(JSON, default=dict)  # Color histograms
	statistical_moments = Column(JSON, default=dict)  # Mean, variance, skewness, kurtosis
	entropy_measures = Column(JSON, default=dict)  # Information entropy
	
	# Processing Performance
	processing_time_ms = Column(Float, nullable=True)
	memory_used_mb = Column(Float, nullable=True)
	cpu_usage_percent = Column(Float, nullable=True)
	
	# Analysis Status
	status = Column(String(20), default='completed', index=True)  # processing, completed, failed
	error_message = Column(Text, nullable=True)
	confidence_level = Column(Float, default=0.0)  # Overall analysis confidence
	
	# Relationships
	image = relationship("CVImage", back_populates="analyses")
	
	def __repr__(self):
		return f"<CVImageAnalysis {self.analysis_type} quality={self.overall_quality_score:.1f}>"
	
	def get_dominant_colors(self, count: int = 5) -> List[Dict[str, Any]]:
		"""Get dominant colors from analysis"""
		colors = self.color_features.get('dominant_colors', [])
		return colors[:count]
	
	def get_texture_summary(self) -> Dict[str, float]:
		"""Get texture analysis summary"""
		return {
			'smoothness': self.texture_features.get('smoothness', 0.0),
			'roughness': self.texture_features.get('roughness', 0.0),
			'regularity': self.texture_features.get('regularity', 0.0),
			'contrast': self.texture_features.get('contrast', 0.0)
		}
	
	def is_high_quality(self, threshold: float = 70.0) -> bool:
		"""Check if image is high quality"""
		return self.overall_quality_score >= threshold
	
	def get_scene_predictions(self, top_k: int = 3) -> List[Dict[str, Any]]:
		"""Get top scene classification predictions"""
		predictions = []
		scene_data = self.scene_classification
		
		if isinstance(scene_data, dict):
			sorted_scenes = sorted(scene_data.items(), key=lambda x: x[1], reverse=True)
			for scene, confidence in sorted_scenes[:top_k]:
				predictions.append({
					'scene_type': scene,
					'confidence': confidence
				})
		
		return predictions
	
	def calculate_complexity_score(self) -> float:
		"""Calculate image complexity score based on features"""
		score = 0.0
		factors = 0
		
		# Edge density factor
		if 'edge_density' in self.shape_features:
			score += self.shape_features['edge_density'] * 30
			factors += 30
		
		# Color diversity factor
		if 'color_diversity' in self.color_features:
			score += self.color_features['color_diversity'] * 25
			factors += 25
		
		# Texture complexity factor
		if 'complexity' in self.texture_features:
			score += self.texture_features['complexity'] * 25
			factors += 25
		
		# Entropy factor
		if 'spatial_entropy' in self.entropy_measures:
			score += min(100, self.entropy_measures['spatial_entropy'] * 10) * 20
			factors += 20
		
		return score / factors if factors > 0 else 0.0


class CVImageEnhancement(Model, AuditMixin, BaseMixin):
	"""
	Image enhancement processing records with before/after metrics.
	
	Tracks image enhancement operations including quality improvements,
	filters applied, and performance metrics.
	"""
	__tablename__ = 'cv_image_enhancement'
	
	# Identity
	enhancement_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	image_id = Column(String(36), ForeignKey('cv_image.image_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Enhancement Configuration
	enhancement_type = Column(String(50), nullable=False, index=True)  # auto, brightness, contrast, denoise, sharpen
	algorithm = Column(String(50), nullable=False)  # clahe, gaussian, bilateral, etc.
	parameters = Column(JSON, default=dict)  # Algorithm-specific parameters
	
	# Enhanced Image Information
	enhanced_file_path = Column(String(1000), nullable=True)
	enhanced_file_size = Column(Integer, nullable=True)
	enhanced_hash = Column(String(64), nullable=True)
	
	# Quality Metrics (Before vs After)
	original_quality_score = Column(Float, nullable=True)
	enhanced_quality_score = Column(Float, nullable=True)
	quality_improvement = Column(Float, nullable=True)  # Percentage improvement
	
	# Specific Improvements
	brightness_improvement = Column(Float, nullable=True)
	contrast_improvement = Column(Float, nullable=True)
	sharpness_improvement = Column(Float, nullable=True)
	noise_reduction = Column(Float, nullable=True)
	
	# Processing Performance
	processing_time_ms = Column(Float, nullable=True)
	memory_usage_mb = Column(Float, nullable=True)
	cpu_usage_percent = Column(Float, nullable=True)
	
	# Enhancement Status
	status = Column(String(20), default='completed', index=True)  # processing, completed, failed
	error_message = Column(Text, nullable=True)
	success_score = Column(Float, default=0.0)  # 0-100 enhancement success
	
	# User Feedback
	user_rating = Column(Integer, nullable=True)  # 1-5 user rating
	user_feedback = Column(Text, nullable=True)
	
	# Metadata
	requested_by = Column(String(36), nullable=True, index=True)  # User ID
	enhancement_metadata = Column(JSON, default=dict)
	
	# Relationships
	image = relationship("CVImage", back_populates="enhancements")
	
	def __repr__(self):
		return f"<CVImageEnhancement {self.enhancement_type} improvement={self.quality_improvement:.1f}%>"
	
	def calculate_improvement_percentage(self) -> float:
		"""Calculate overall improvement percentage"""
		if self.original_quality_score and self.enhanced_quality_score:
			return ((self.enhanced_quality_score - self.original_quality_score) 
					/ self.original_quality_score * 100)
		return 0.0
	
	def is_successful_enhancement(self, threshold: float = 5.0) -> bool:
		"""Check if enhancement was successful (improved quality by threshold%)"""
		improvement = self.calculate_improvement_percentage()
		return improvement >= threshold
	
	def get_enhancement_summary(self) -> Dict[str, Any]:
		"""Get enhancement summary with key metrics"""
		return {
			'enhancement_type': self.enhancement_type,
			'algorithm': self.algorithm,
			'quality_improvement': self.quality_improvement,
			'processing_time_ms': self.processing_time_ms,
			'success_score': self.success_score,
			'user_rating': self.user_rating,
			'improvements': {
				'brightness': self.brightness_improvement,
				'contrast': self.contrast_improvement,
				'sharpness': self.sharpness_improvement,
				'noise_reduction': self.noise_reduction
			}
		}


class CVVideo(Model, AuditMixin, BaseMixin):
	"""
	Video records for computer vision processing.
	
	Stores metadata about videos including format, duration,
	frame analysis results, and processing statistics.
	"""
	__tablename__ = 'cv_video'
	
	# Identity
	video_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Video Information
	original_filename = Column(String(500), nullable=False)
	file_path = Column(String(1000), nullable=False)
	file_size_bytes = Column(Integer, nullable=False)
	mime_type = Column(String(100), nullable=False)
	file_hash = Column(String(64), nullable=False, index=True)
	
	# Video Properties
	duration_seconds = Column(Float, nullable=False)
	frame_rate = Column(Float, nullable=False)  # FPS
	total_frames = Column(Integer, nullable=False)
	width = Column(Integer, nullable=False)
	height = Column(Integer, nullable=False)
	bitrate = Column(Integer, nullable=True)  # bits per second
	codec = Column(String(50), nullable=True)
	
	# Processing Status
	processing_status = Column(String(20), default='uploaded', index=True)
	frames_processed = Column(Integer, default=0)
	processing_progress = Column(Float, default=0.0)  # 0-100 percentage
	
	# Analysis Results Summary
	total_detections = Column(Integer, default=0)
	unique_objects_detected = Column(Integer, default=0)
	avg_detections_per_frame = Column(Float, default=0.0)
	detection_confidence_avg = Column(Float, default=0.0)
	
	# Processing Performance
	total_processing_time_ms = Column(Float, nullable=True)
	avg_frame_processing_time = Column(Float, nullable=True)  # ms per frame
	
	# Output Information
	output_video_path = Column(String(1000), nullable=True)  # Enhanced/annotated video
	thumbnail_path = Column(String(1000), nullable=True)
	analysis_report_path = Column(String(1000), nullable=True)
	
	# Upload and User Context
	uploaded_by = Column(String(36), nullable=True, index=True)
	upload_source = Column(String(50), nullable=True)
	
	# Relationships
	frame_analyses = relationship("CVFrameAnalysis", back_populates="video", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CVVideo {self.original_filename} ({self.duration_seconds:.1f}s, {self.total_frames} frames)>"
	
	def get_duration_formatted(self) -> str:
		"""Get formatted duration string (HH:MM:SS)"""
		hours = int(self.duration_seconds // 3600)
		minutes = int((self.duration_seconds % 3600) // 60)
		seconds = int(self.duration_seconds % 60)
		
		if hours > 0:
			return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
		else:
			return f"{minutes:02d}:{seconds:02d}"
	
	def get_resolution_string(self) -> str:
		"""Get video resolution as string"""
		return f"{self.width}x{self.height}"
	
	def get_aspect_ratio(self) -> float:
		"""Get video aspect ratio"""
		return self.width / self.height if self.height > 0 else 1.0
	
	def calculate_processing_efficiency(self) -> float:
		"""Calculate processing efficiency (frames per second processed)"""
		if self.total_processing_time_ms and self.total_processing_time_ms > 0:
			processing_seconds = self.total_processing_time_ms / 1000
			return self.frames_processed / processing_seconds
		return 0.0
	
	def get_file_size_formatted(self) -> str:
		"""Get formatted file size string"""
		size = self.file_size_bytes
		
		if size < 1024 * 1024:
			return f"{size / 1024:.1f} KB"
		elif size < 1024 * 1024 * 1024:
			return f"{size / (1024 * 1024):.1f} MB"
		else:
			return f"{size / (1024 * 1024 * 1024):.1f} GB"
	
	def update_processing_stats(self) -> None:
		"""Update processing statistics based on frame analyses"""
		if self.frame_analyses:
			self.frames_processed = len(self.frame_analyses)
			self.processing_progress = (self.frames_processed / self.total_frames) * 100
			
			# Calculate detection statistics
			total_detections = sum(len(fa.detections) for fa in self.frame_analyses)
			self.total_detections = total_detections
			self.avg_detections_per_frame = total_detections / len(self.frame_analyses)
			
			# Calculate average confidence
			all_confidences = []
			for fa in self.frame_analyses:
				all_confidences.extend([d.confidence_score for d in fa.detections])
			
			if all_confidences:
				self.detection_confidence_avg = sum(all_confidences) / len(all_confidences)


class CVFrameAnalysis(Model, AuditMixin, BaseMixin):
	"""
	Individual video frame analysis results.
	
	Records analysis results for each processed video frame including
	detections, timestamps, and frame-specific metrics.
	"""
	__tablename__ = 'cv_frame_analysis'
	
	# Identity
	frame_analysis_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	video_id = Column(String(36), ForeignKey('cv_video.video_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Frame Information
	frame_number = Column(Integer, nullable=False, index=True)
	timestamp_seconds = Column(Float, nullable=False)  # Time in video
	frame_width = Column(Integer, nullable=False)
	frame_height = Column(Integer, nullable=False)
	
	# Frame Quality Metrics
	brightness = Column(Float, nullable=True)
	contrast = Column(Float, nullable=True)
	blur_score = Column(Float, nullable=True)
	motion_score = Column(Float, nullable=True)  # Amount of motion from previous frame
	
	# Detection Results Summary
	detections_count = Column(Integer, default=0)
	detection_types = Column(JSON, default=list)  # List of detected object types
	max_confidence = Column(Float, default=0.0)
	avg_confidence = Column(Float, default=0.0)
	
	# Processing Performance
	processing_time_ms = Column(Float, nullable=True)
	analysis_timestamp = Column(DateTime, default=datetime.utcnow)
	
	# Frame Analysis Data
	frame_features = Column(JSON, default=dict)  # Extracted frame features
	scene_changes = Column(Boolean, default=False)  # Scene change detection
	motion_vectors = Column(JSON, default=list)  # Motion analysis data
	
	# Status
	status = Column(String(20), default='completed', index=True)
	error_message = Column(Text, nullable=True)
	
	# Relationships
	video = relationship("CVVideo", back_populates="frame_analyses")
	detections = relationship("CVVideoDetection", back_populates="frame_analysis", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CVFrameAnalysis frame={self.frame_number} detections={self.detections_count}>"
	
	def get_timestamp_formatted(self) -> str:
		"""Get formatted timestamp (MM:SS.mmm)"""
		minutes = int(self.timestamp_seconds // 60)
		seconds = self.timestamp_seconds % 60
		return f"{minutes:02d}:{seconds:06.3f}"
	
	def has_significant_detections(self, confidence_threshold: float = 0.7) -> bool:
		"""Check if frame has significant detections above confidence threshold"""
		return self.max_confidence >= confidence_threshold
	
	def get_detection_summary(self) -> Dict[str, int]:
		"""Get summary of detections by type"""
		summary = {}
		for detection in self.detections:
			obj_type = detection.object_type
			summary[obj_type] = summary.get(obj_type, 0) + 1
		return summary


class CVVideoDetection(Model, AuditMixin, BaseMixin):
	"""
	Object detections in video frames.
	
	Extends CVDetection for video-specific detection results with
	temporal tracking and motion analysis.
	"""
	__tablename__ = 'cv_video_detection'
	
	# Identity
	detection_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	frame_analysis_id = Column(String(36), ForeignKey('cv_frame_analysis.frame_analysis_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Detection Results (similar to CVDetection)
	object_type = Column(String(50), nullable=False, index=True)
	object_class = Column(String(100), nullable=True)
	confidence_score = Column(Float, nullable=False)
	
	# Bounding Box
	bbox_x = Column(Integer, nullable=False)
	bbox_y = Column(Integer, nullable=False)
	bbox_width = Column(Integer, nullable=False)
	bbox_height = Column(Integer, nullable=False)
	center_x = Column(Integer, nullable=False)
	center_y = Column(Integer, nullable=False)
	
	# Video-Specific Features
	tracking_id = Column(String(36), nullable=True, index=True)  # For object tracking across frames
	velocity_x = Column(Float, nullable=True)  # Horizontal velocity (pixels/second)
	velocity_y = Column(Float, nullable=True)  # Vertical velocity (pixels/second)
	is_new_object = Column(Boolean, default=True)  # First appearance of object
	is_leaving_frame = Column(Boolean, default=False)  # Object leaving frame
	
	# Temporal Consistency
	consistency_score = Column(Float, default=0.0)  # Consistency with previous frames
	lifetime_frames = Column(Integer, default=1)  # Number of frames object has been tracked
	
	# Detection Method
	detection_method = Column(String(50), nullable=False)
	model_name = Column(String(100), nullable=True)
	
	# Additional Attributes
	attributes = Column(JSON, default=dict)
	metadata = Column(JSON, default=dict)
	
	# Relationships
	frame_analysis = relationship("CVFrameAnalysis", back_populates="detections")
	
	def __repr__(self):
		return f"<CVVideoDetection {self.object_type} track={self.tracking_id}>"
	
	def get_speed(self) -> float:
		"""Calculate object speed in pixels per second"""
		if self.velocity_x is not None and self.velocity_y is not None:
			return (self.velocity_x ** 2 + self.velocity_y ** 2) ** 0.5
		return 0.0
	
	def get_bbox_dict(self) -> Dict[str, int]:
		"""Get bounding box as dictionary"""
		return {
			'x': self.bbox_x,
			'y': self.bbox_y,
			'width': self.bbox_width,
			'height': self.bbox_height
		}
	
	def is_tracked_object(self) -> bool:
		"""Check if object is being tracked across frames"""
		return self.tracking_id is not None and self.lifetime_frames > 1


class CVModelConfiguration(Model, AuditMixin, BaseMixin):
	"""
	Computer vision model configurations and metadata.
	
	Stores information about ML models used for detection and analysis
	including performance metrics and configuration parameters.
	"""
	__tablename__ = 'cv_model_configuration'
	
	# Identity
	model_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Model Information
	model_name = Column(String(100), nullable=False, index=True)
	model_type = Column(String(50), nullable=False, index=True)  # detection, classification, segmentation
	model_version = Column(String(20), nullable=False)
	framework = Column(String(50), nullable=False)  # opencv, tensorflow, pytorch, onnx
	
	# Model Files and Paths
	model_file_path = Column(String(1000), nullable=False)
	config_file_path = Column(String(1000), nullable=True)
	weights_file_path = Column(String(1000), nullable=True)
	labels_file_path = Column(String(1000), nullable=True)
	
	# Model Configuration
	input_size = Column(JSON, default=dict)  # Expected input dimensions
	output_classes = Column(JSON, default=list)  # List of output classes
	preprocessing_config = Column(JSON, default=dict)  # Preprocessing parameters
	postprocessing_config = Column(JSON, default=dict)  # Postprocessing parameters
	
	# Performance Characteristics
	accuracy_score = Column(Float, nullable=True)  # Model accuracy (0-1)
	precision_score = Column(Float, nullable=True)
	recall_score = Column(Float, nullable=True)
	f1_score = Column(Float, nullable=True)
	inference_time_ms = Column(Float, nullable=True)  # Average inference time
	
	# Resource Requirements
	memory_requirement_mb = Column(Integer, nullable=True)
	gpu_required = Column(Boolean, default=False)
	min_gpu_memory_mb = Column(Integer, nullable=True)
	cpu_cores_recommended = Column(Integer, default=1)
	
	# Model Status and Usage
	is_active = Column(Boolean, default=True, index=True)
	is_default = Column(Boolean, default=False)
	usage_count = Column(Integer, default=0)
	last_used = Column(DateTime, nullable=True)
	
	# Model Metadata
	description = Column(Text, nullable=True)
	author = Column(String(200), nullable=True)
	license = Column(String(100), nullable=True)
	training_data_info = Column(JSON, default=dict)
	limitations = Column(Text, nullable=True)
	
	def __repr__(self):
		return f"<CVModelConfiguration {self.model_name} v{self.model_version}>"
	
	def get_performance_summary(self) -> Dict[str, float]:
		"""Get model performance metrics summary"""
		return {
			'accuracy': self.accuracy_score or 0.0,
			'precision': self.precision_score or 0.0,
			'recall': self.recall_score or 0.0,
			'f1_score': self.f1_score or 0.0,
			'inference_time_ms': self.inference_time_ms or 0.0
		}
	
	def is_suitable_for_realtime(self, max_inference_time: float = 100.0) -> bool:
		"""Check if model is suitable for real-time processing"""
		return (self.inference_time_ms is not None and 
				self.inference_time_ms <= max_inference_time)
	
	def update_usage_stats(self) -> None:
		"""Update model usage statistics"""
		self.usage_count += 1
		self.last_used = datetime.utcnow()
	
	def calculate_efficiency_score(self) -> float:
		"""Calculate model efficiency score (performance vs speed)"""
		if self.f1_score and self.inference_time_ms:
			# Balance between accuracy and speed
			accuracy_weight = 0.7
			speed_weight = 0.3
			
			# Normalize speed (lower is better, so invert)
			speed_score = max(0, 1.0 - (self.inference_time_ms / 1000))  # Assume 1s is very slow
			
			return (self.f1_score * accuracy_weight + speed_score * speed_weight) * 100
		
		return 0.0


class CVProcessingJob(Model, AuditMixin, BaseMixin):
	"""
	Computer vision processing job tracking.
	
	Tracks background processing jobs for batch image/video processing
	with status monitoring and performance metrics.
	"""
	__tablename__ = 'cv_processing_job'
	
	# Identity
	job_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Job Configuration
	job_type = Column(String(50), nullable=False, index=True)  # image_detection, video_analysis, batch_enhancement
	job_name = Column(String(200), nullable=True)
	job_description = Column(Text, nullable=True)
	
	# Input Configuration
	input_type = Column(String(20), nullable=False)  # image, video, batch
	input_paths = Column(JSON, nullable=False)  # List of input file paths
	input_count = Column(Integer, default=0)
	
	# Processing Configuration
	detection_types = Column(JSON, default=list)  # Types of detection to perform
	model_configurations = Column(JSON, default=list)  # Models to use
	processing_options = Column(JSON, default=dict)  # Additional options
	
	# Job Status
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
	progress_percentage = Column(Float, default=0.0)
	current_item = Column(String(500), nullable=True)  # Currently processing item
	items_processed = Column(Integer, default=0)
	items_failed = Column(Integer, default=0)
	
	# Scheduling
	scheduled_at = Column(DateTime, nullable=True, index=True)
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True, index=True)
	
	# Results Summary
	total_detections = Column(Integer, default=0)
	output_files = Column(JSON, default=list)
	results_summary = Column(JSON, default=dict)
	
	# Performance Metrics
	total_processing_time_ms = Column(Float, nullable=True)
	avg_item_processing_time = Column(Float, nullable=True)
	peak_memory_usage_mb = Column(Float, nullable=True)
	cpu_usage_avg = Column(Float, nullable=True)
	
	# Error Handling
	error_message = Column(Text, nullable=True)
	error_details = Column(JSON, default=dict)
	retry_count = Column(Integer, default=0)
	max_retries = Column(Integer, default=3)
	
	# Job Metadata
	requested_by = Column(String(36), nullable=True, index=True)
	priority = Column(String(20), default='normal')  # low, normal, high, urgent
	worker_id = Column(String(100), nullable=True)
	
	def __repr__(self):
		return f"<CVProcessingJob {self.job_type} status={self.status}>"
	
	def calculate_success_rate(self) -> float:
		"""Calculate job success rate"""
		total_items = self.items_processed + self.items_failed
		if total_items > 0:
			return (self.items_processed / total_items) * 100
		return 0.0
	
	def get_duration(self) -> Optional[float]:
		"""Get job duration in seconds"""
		if self.started_at and self.completed_at:
			return (self.completed_at - self.started_at).total_seconds()
		elif self.started_at:
			return (datetime.utcnow() - self.started_at).total_seconds()
		return None
	
	def get_estimated_completion(self) -> Optional[datetime]:
		"""Estimate job completion time based on progress"""
		if not self.started_at or self.progress_percentage <= 0:
			return None
		
		elapsed = (datetime.utcnow() - self.started_at).total_seconds()
		estimated_total = elapsed / (self.progress_percentage / 100)
		remaining = estimated_total - elapsed
		
		return datetime.utcnow() + timedelta(seconds=remaining)
	
	def is_running(self) -> bool:
		"""Check if job is currently running"""
		return self.status in ['pending', 'running']
	
	def can_retry(self) -> bool:
		"""Check if job can be retried"""
		return self.status == 'failed' and self.retry_count < self.max_retries
	
	def update_progress(self, percentage: float, current_item: str = None) -> None:
		"""Update job progress"""
		self.progress_percentage = min(100.0, max(0.0, percentage))
		if current_item:
			self.current_item = current_item