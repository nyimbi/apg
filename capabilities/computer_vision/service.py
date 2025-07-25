#!/usr/bin/env python3
"""
APG Computer Vision Capability
==============================

Advanced computer vision capabilities for image processing, object detection,
face recognition, and real-time video analysis.
"""

import asyncio
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import base64
import json
from datetime import datetime
import threading
import queue

# Optional imports for advanced features
try:
	import tensorflow as tf
	HAS_TENSORFLOW = True
except ImportError:
	HAS_TENSORFLOW = False

try:
	import torch
	import torchvision
	HAS_PYTORCH = True
except ImportError:
	HAS_PYTORCH = False

try:
	from PIL import Image
	HAS_PIL = True
except ImportError:
	HAS_PIL = False

class DetectionType(Enum):
	"""Types of object detection"""
	FACE = "face"
	PERSON = "person"
	VEHICLE = "vehicle"
	OBJECT = "object"
	CUSTOM = "custom"

class ProcessingMode(Enum):
	"""Image processing modes"""
	REAL_TIME = "real_time"
	BATCH = "batch"
	STREAM = "stream"

@dataclass
class DetectionResult:
	"""Result of object detection"""
	object_type: str
	confidence: float
	bounding_box: Tuple[int, int, int, int]  # x, y, width, height
	center_point: Tuple[int, int]
	attributes: Dict[str, Any] = field(default_factory=dict)
	timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FrameAnalysis:
	"""Analysis result for a video frame"""
	frame_id: int
	timestamp: datetime
	detections: List[DetectionResult]
	frame_metrics: Dict[str, Any]
	processing_time_ms: float

class ComputerVisionProcessor:
	"""Core computer vision processing engine"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.logger = logging.getLogger("cv_processor")
		
		# Initialize OpenCV cascades
		self.face_cascade = None
		self.eye_cascade = None
		self.body_cascade = None
		
		# ML models
		self.tf_model = None
		self.pytorch_model = None
		
		# Processing state
		self.is_processing = False
		self.frame_buffer = queue.Queue(maxsize=100)
		
		self._initialize_detectors()
	
	def _initialize_detectors(self):
		"""Initialize detection models and cascades"""
		try:
			# Load OpenCV pre-trained cascades
			cv2_data = cv2.data.haarcascades
			self.face_cascade = cv2.CascadeClassifier(cv2_data + 'haarcascade_frontalface_default.xml')
			self.eye_cascade = cv2.CascadeClassifier(cv2_data + 'haarcascade_eye.xml')
			
			# Initialize DNN for object detection
			if 'yolo_weights' in self.config and 'yolo_config' in self.config:
				self.yolo_net = cv2.dnn.readNet(
					self.config['yolo_weights'],
					self.config['yolo_config']
				)
				self.logger.info("YOLO model loaded successfully")
		
		except Exception as e:
			self.logger.error(f"Error initializing detectors: {e}")
	
	async def process_image(
		self, 
		image: Union[np.ndarray, str, Path], 
		detection_types: List[DetectionType] = None
	) -> List[DetectionResult]:
		"""Process a single image for object detection"""
		
		# Load image if path provided
		if isinstance(image, (str, Path)):
			image = cv2.imread(str(image))
		
		if image is None:
			raise ValueError("Invalid image provided")
		
		detection_types = detection_types or [DetectionType.FACE, DetectionType.PERSON]
		results = []
		
		for detection_type in detection_types:
			if detection_type == DetectionType.FACE:
				results.extend(await self._detect_faces(image))
			elif detection_type == DetectionType.PERSON:
				results.extend(await self._detect_persons(image))
			elif detection_type == DetectionType.VEHICLE:
				results.extend(await self._detect_vehicles(image))
			elif detection_type == DetectionType.OBJECT:
				results.extend(await self._detect_objects(image))
		
		return results
	
	async def _detect_faces(self, image: np.ndarray) -> List[DetectionResult]:
		"""Detect faces in image using Haar cascades"""
		if self.face_cascade is None:
			return []
		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(
			gray, 
			scaleFactor=1.1, 
			minNeighbors=5, 
			minSize=(30, 30)
		)
		
		results = []
		for (x, y, w, h) in faces:
			# Calculate confidence based on face size and quality
			face_area = w * h
			confidence = min(0.95, max(0.5, face_area / (image.shape[0] * image.shape[1]) * 10))
			
			# Detect eyes within face region for better confidence
			face_roi = gray[y:y+h, x:x+w]
			eyes = self.eye_cascade.detectMultiScale(face_roi) if self.eye_cascade else []
			
			if len(eyes) >= 2:
				confidence = min(0.98, confidence + 0.2)
			
			result = DetectionResult(
				object_type="face",
				confidence=confidence,
				bounding_box=(x, y, w, h),
				center_point=(x + w//2, y + h//2),
				attributes={
					"eyes_detected": len(eyes),
					"face_area": face_area,
					"face_ratio": w / h if h > 0 else 1.0
				}
			)
			results.append(result)
		
		return results
	
	async def _detect_persons(self, image: np.ndarray) -> List[DetectionResult]:
		"""Detect persons using HOG descriptor"""
		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		
		# Detect people
		(rects, weights) = hog.detectMultiScale(
			image,
			winStride=(4, 4),
			padding=(8, 8),
			scale=1.05
		)
		
		results = []
		for i, (x, y, w, h) in enumerate(rects):
			confidence = float(weights[i]) if i < len(weights) else 0.5
			confidence = max(0.0, min(1.0, confidence))
			
			result = DetectionResult(
				object_type="person",
				confidence=confidence,
				bounding_box=(x, y, w, h),
				center_point=(x + w//2, y + h//2),
				attributes={
					"detection_method": "hog",
					"body_area": w * h,
					"aspect_ratio": w / h if h > 0 else 1.0
				}
			)
			results.append(result)
		
		return results
	
	async def _detect_vehicles(self, image: np.ndarray) -> List[DetectionResult]:
		"""Detect vehicles using basic contour analysis"""
		# Convert to HSV for better color filtering
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# Apply Gaussian blur and edge detection
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		edges = cv2.Canny(blurred, 50, 150)
		
		# Find contours
		contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		results = []
		for contour in contours:
			# Filter contours by area and aspect ratio
			area = cv2.contourArea(contour)
			if area < 1000:  # Minimum area for vehicle
				continue
			
			# Get bounding rectangle
			x, y, w, h = cv2.boundingRect(contour)
			aspect_ratio = w / h if h > 0 else 1.0
			
			# Basic vehicle shape filtering (wider than tall)
			if aspect_ratio > 1.2 and area > 2000:
				confidence = min(0.8, area / 10000)  # Simple confidence based on size
				
				result = DetectionResult(
					object_type="vehicle",
					confidence=confidence,
					bounding_box=(x, y, w, h),
					center_point=(x + w//2, y + h//2),
					attributes={
						"detection_method": "contour",
						"area": area,
						"aspect_ratio": aspect_ratio,
						"contour_points": len(contour)
					}
				)
				results.append(result)
		
		return results
	
	async def _detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
		"""Detect general objects using YOLO or other DNN models"""
		if not hasattr(self, 'yolo_net') or self.yolo_net is None:
			return []
		
		# Create blob from image
		blob = cv2.dnn.blobFromImage(
			image, 1/255.0, (416, 416), swapRB=True, crop=False
		)
		
		# Set input to network
		self.yolo_net.setInput(blob)
		
		# Run forward pass
		layer_outputs = self.yolo_net.forward(self._get_output_layers())
		
		boxes = []
		confidences = []
		class_ids = []
		
		# Process each output layer
		for output in layer_outputs:
			for detection in output:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				
				if confidence > 0.5:  # Confidence threshold
					# Get bounding box coordinates
					center_x = int(detection[0] * image.shape[1])
					center_y = int(detection[1] * image.shape[0])
					width = int(detection[2] * image.shape[1])
					height = int(detection[3] * image.shape[0])
					
					x = int(center_x - width / 2)
					y = int(center_y - height / 2)
					
					boxes.append([x, y, width, height])
					confidences.append(float(confidence))
					class_ids.append(class_id)
		
		# Apply non-maximum suppression
		indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		
		results = []
		if len(indices) > 0:
			for i in indices.flatten():
				x, y, w, h = boxes[i]
				confidence = confidences[i]
				class_id = class_ids[i]
				
				# Map class ID to object name (simplified)
				object_names = {
					0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
					5: "bus", 7: "truck", 15: "cat", 16: "dog"
				}
				object_type = object_names.get(class_id, f"object_{class_id}")
				
				result = DetectionResult(
					object_type=object_type,
					confidence=confidence,
					bounding_box=(x, y, w, h),
					center_point=(x + w//2, y + h//2),
					attributes={
						"detection_method": "yolo",
						"class_id": class_id,
						"nms_applied": True
					}
				)
				results.append(result)
		
		return results
	
	def _get_output_layers(self):
		"""Get YOLO output layer names"""
		if not hasattr(self, 'yolo_net') or self.yolo_net is None:
			return []
		
		layer_names = self.yolo_net.getLayerNames()
		output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
		return output_layers

class VideoProcessor:
	"""Real-time video processing and analysis"""
	
	def __init__(self, cv_processor: ComputerVisionProcessor):
		self.cv_processor = cv_processor
		self.logger = logging.getLogger("video_processor")
		
		# Video capture and processing state
		self.cap = None
		self.is_recording = False
		self.is_processing = False
		self.frame_count = 0
		self.results_buffer = []
		
		# Threading
		self.capture_thread = None
		self.process_thread = None
		self.stop_event = threading.Event()
	
	async def start_camera_stream(
		self, 
		camera_id: int = 0,
		detection_types: List[DetectionType] = None,
		callback: callable = None
	) -> bool:
		"""Start real-time camera processing"""
		
		self.cap = cv2.VideoCapture(camera_id)
		if not self.cap.isOpened():
			self.logger.error(f"Cannot open camera {camera_id}")
			return False
		
		self.is_processing = True
		self.stop_event.clear()
		detection_types = detection_types or [DetectionType.FACE]
		
		# Start processing thread
		self.process_thread = threading.Thread(
			target=self._process_video_stream,
			args=(detection_types, callback)
		)
		self.process_thread.start()
		
		self.logger.info(f"Started camera stream on device {camera_id}")
		return True
	
	def _process_video_stream(
		self, 
		detection_types: List[DetectionType],
		callback: callable = None
	):
		"""Process video stream in separate thread"""
		
		while self.is_processing and not self.stop_event.is_set():
			ret, frame = self.cap.read()
			if not ret:
				break
			
			self.frame_count += 1
			start_time = datetime.utcnow()
			
			# Process frame asynchronously
			try:
				# Run detection in sync mode for threading compatibility
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				detections = loop.run_until_complete(
					self.cv_processor.process_image(frame, detection_types)
				)
				
				loop.close()
				
				# Calculate processing time
				processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
				
				# Create frame analysis
				analysis = FrameAnalysis(
					frame_id=self.frame_count,
					timestamp=start_time,
					detections=detections,
					frame_metrics={
						"width": frame.shape[1],
						"height": frame.shape[0],
						"channels": frame.shape[2],
						"fps": self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0
					},
					processing_time_ms=processing_time
				)
				
				# Store results
				self.results_buffer.append(analysis)
				if len(self.results_buffer) > 100:  # Keep last 100 frames
					self.results_buffer.pop(0)
				
				# Call callback if provided
				if callback:
					callback(frame, analysis)
				
				# Basic visualization
				self._draw_detections(frame, detections)
				cv2.imshow('APG Computer Vision', frame)
				
				# Break on 'q' key
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			
			except Exception as e:
				self.logger.error(f"Error processing frame {self.frame_count}: {e}")
	
	def _draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]):
		"""Draw detection results on frame"""
		
		for detection in detections:
			x, y, w, h = detection.bounding_box
			confidence = detection.confidence
			object_type = detection.object_type
			
			# Color based on object type
			colors = {
				"face": (255, 0, 0),      # Blue
				"person": (0, 255, 0),    # Green
				"vehicle": (0, 0, 255),   # Red
				"object": (255, 255, 0)   # Cyan
			}
			color = colors.get(object_type, (128, 128, 128))
			
			# Draw bounding box
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			
			# Draw label
			label = f"{object_type}: {confidence:.2f}"
			label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
			cv2.rectangle(frame, (x, y - label_size[1] - 10), 
						 (x + label_size[0], y), color, -1)
			cv2.putText(frame, label, (x, y - 5), 
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			
			# Draw center point
			center_x, center_y = detection.center_point
			cv2.circle(frame, (center_x, center_y), 3, color, -1)
	
	async def process_video_file(
		self, 
		video_path: str,
		detection_types: List[DetectionType] = None,
		output_path: str = None
	) -> List[FrameAnalysis]:
		"""Process entire video file"""
		
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise ValueError(f"Cannot open video file: {video_path}")
		
		detection_types = detection_types or [DetectionType.FACE, DetectionType.PERSON]
		results = []
		
		# Get video properties
		fps = cap.get(cv2.CAP_PROP_FPS)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		# Setup output video if requested
		out = None
		if output_path:
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
		
		frame_id = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			
			frame_id += 1
			start_time = datetime.utcnow()
			
			# Process frame
			detections = await self.cv_processor.process_image(frame, detection_types)
			processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			
			# Create analysis
			analysis = FrameAnalysis(
				frame_id=frame_id,
				timestamp=start_time,
				detections=detections,
				frame_metrics={
					"width": frame.shape[1],
					"height": frame.shape[0],
					"fps": fps,
					"progress": frame_id / frame_count
				},
				processing_time_ms=processing_time
			)
			results.append(analysis)
			
			# Draw detections and save frame
			if out:
				self._draw_detections(frame, detections)
				out.write(frame)
			
			# Progress logging
			if frame_id % 30 == 0:  # Log every 30 frames
				self.logger.info(f"Processed frame {frame_id}/{frame_count}")
		
		# Cleanup
		cap.release()
		if out:
			out.release()
		
		self.logger.info(f"Processed {len(results)} frames from {video_path}")
		return results
	
	def stop_processing(self):
		"""Stop video processing"""
		self.is_processing = False
		self.stop_event.set()
		
		if self.process_thread and self.process_thread.is_alive():
			self.process_thread.join(timeout=5)
		
		if self.cap:
			self.cap.release()
		
		cv2.destroyAllWindows()
		self.logger.info("Video processing stopped")

class ImageProcessor:
	"""Advanced image processing and enhancement"""
	
	def __init__(self):
		self.logger = logging.getLogger("image_processor")
	
	async def enhance_image(
		self, 
		image: np.ndarray,
		enhancement_type: str = "auto"
	) -> np.ndarray:
		"""Enhance image quality"""
		
		if enhancement_type == "auto":
			# Auto-enhance based on image characteristics
			enhanced = self._auto_enhance(image)
		elif enhancement_type == "brightness":
			enhanced = self._adjust_brightness(image)
		elif enhancement_type == "contrast":
			enhanced = self._adjust_contrast(image)
		elif enhancement_type == "denoise":
			enhanced = self._denoise_image(image)
		elif enhancement_type == "sharpen":
			enhanced = self._sharpen_image(image)
		else:
			enhanced = image.copy()
		
		return enhanced
	
	def _auto_enhance(self, image: np.ndarray) -> np.ndarray:
		"""Automatically enhance image"""
		# Convert to LAB color space
		lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		
		# Apply CLAHE to L channel
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
		l = clahe.apply(l)
		
		# Merge channels and convert back
		enhanced = cv2.merge([l, a, b])
		enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
		
		return enhanced
	
	def _adjust_brightness(self, image: np.ndarray, value: int = 30) -> np.ndarray:
		"""Adjust image brightness"""
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)
		
		v = cv2.add(v, value)
		v = np.clip(v, 0, 255)
		
		enhanced = cv2.merge([h, s, v])
		enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
		
		return enhanced
	
	def _adjust_contrast(self, image: np.ndarray, alpha: float = 1.3) -> np.ndarray:
		"""Adjust image contrast"""
		return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
	
	def _denoise_image(self, image: np.ndarray) -> np.ndarray:
		"""Remove noise from image"""
		return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
	
	def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
		"""Sharpen image"""
		kernel = np.array([[-1, -1, -1],
						  [-1, 9, -1],
						  [-1, -1, -1]])
		return cv2.filter2D(image, -1, kernel)
	
	async def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
		"""Extract image features for analysis"""
		
		# Convert to different color spaces
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		# Basic statistics
		features = {
			"dimensions": {
				"width": image.shape[1],
				"height": image.shape[0],
				"channels": image.shape[2]
			},
			"color_stats": {
				"mean_bgr": np.mean(image, axis=(0, 1)).tolist(),
				"std_bgr": np.std(image, axis=(0, 1)).tolist(),
				"brightness": np.mean(gray),
				"contrast": np.std(gray)
			},
			"histogram": {
				"blue": cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().tolist(),
				"green": cv2.calcHist([image], [1], None, [256], [0, 256]).flatten().tolist(),
				"red": cv2.calcHist([image], [2], None, [256], [0, 256]).flatten().tolist()
			}
		}
		
		# Edge detection
		edges = cv2.Canny(gray, 50, 150)
		features["edge_density"] = np.sum(edges > 0) / edges.size
		
		# Texture analysis using Local Binary Pattern (simplified)
		features["texture_score"] = self._calculate_texture_score(gray)
		
		return features
	
	def _calculate_texture_score(self, gray_image: np.ndarray) -> float:
		"""Calculate texture score using variance of Laplacian"""
		laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
		return float(laplacian_var)

class ComputerVisionCapability:
	"""Main computer vision capability interface"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.cv_processor = ComputerVisionProcessor(config)
		self.video_processor = VideoProcessor(self.cv_processor)
		self.image_processor = ImageProcessor()
		self.logger = logging.getLogger("cv_capability")
		
		# Initialize capability
		self._initialize()
	
	def _initialize(self):
		"""Initialize the capability"""
		self.logger.info("Initializing Computer Vision Capability")
		
		# Check dependencies
		self._check_dependencies()
		
		# Load configuration
		self._load_models()
	
	def _check_dependencies(self):
		"""Check required dependencies"""
		dependencies = {
			"OpenCV": cv2.__version__ if cv2 else None,
			"NumPy": np.__version__ if np else None,
			"TensorFlow": tf.__version__ if HAS_TENSORFLOW else None,
			"PyTorch": torch.__version__ if HAS_PYTORCH else None,
			"PIL": "Available" if HAS_PIL else None
		}
		
		self.logger.info(f"Dependencies: {dependencies}")
	
	def _load_models(self):
		"""Load pre-trained models"""
		if self.config.get('preload_models', True):
			self.logger.info("Pre-loading computer vision models...")
			# Models loaded in cv_processor initialization
	
	# Public API Methods
	
	async def detect_objects_in_image(
		self,
		image_path: str,
		detection_types: List[str] = None
	) -> Dict[str, Any]:
		"""Detect objects in a single image"""
		
		# Convert string detection types to enum
		if detection_types:
			det_types = [DetectionType(dt) for dt in detection_types if dt in DetectionType._value2member_map_]
		else:
			det_types = [DetectionType.FACE, DetectionType.PERSON]
		
		# Process image
		detections = await self.cv_processor.process_image(image_path, det_types)
		
		# Format results
		return {
			"image_path": str(image_path),
			"detections_count": len(detections),
			"detections": [
				{
					"type": det.object_type,
					"confidence": det.confidence,
					"bounding_box": det.bounding_box,
					"center": det.center_point,
					"attributes": det.attributes,
					"timestamp": det.timestamp.isoformat()
				}
				for det in detections
			],
			"processing_timestamp": datetime.utcnow().isoformat()
		}
	
	async def start_live_detection(
		self,
		camera_id: int = 0,
		detection_types: List[str] = None,
		callback_url: str = None
	) -> Dict[str, Any]:
		"""Start live camera detection"""
		
		# Convert detection types
		det_types = []
		if detection_types:
			det_types = [DetectionType(dt) for dt in detection_types if dt in DetectionType._value2member_map_]
		else:
			det_types = [DetectionType.FACE]
		
		# Define callback function
		def detection_callback(frame, analysis):
			if callback_url:
				# In a real implementation, you'd send results to callback_url
				self.logger.info(f"Frame {analysis.frame_id}: {len(analysis.detections)} detections")
		
		# Start camera stream
		success = await self.video_processor.start_camera_stream(
			camera_id=camera_id,
			detection_types=det_types,
			callback=detection_callback
		)
		
		return {
			"success": success,
			"camera_id": camera_id,
			"detection_types": [dt.value for dt in det_types],
			"status": "streaming" if success else "failed",
			"timestamp": datetime.utcnow().isoformat()
		}
	
	def stop_live_detection(self) -> Dict[str, Any]:
		"""Stop live camera detection"""
		self.video_processor.stop_processing()
		
		return {
			"success": True,
			"status": "stopped",
			"timestamp": datetime.utcnow().isoformat()
		}
	
	async def process_video_file(
		self,
		video_path: str,
		output_path: str = None,
		detection_types: List[str] = None
	) -> Dict[str, Any]:
		"""Process entire video file"""
		
		# Convert detection types
		det_types = []
		if detection_types:
			det_types = [DetectionType(dt) for dt in detection_types if dt in DetectionType._value2member_map_]
		else:
			det_types = [DetectionType.FACE, DetectionType.PERSON]
		
		# Process video
		analyses = await self.video_processor.process_video_file(
			video_path=video_path,
			detection_types=det_types,
			output_path=output_path
		)
		
		# Aggregate results
		total_detections = sum(len(analysis.detections) for analysis in analyses)
		avg_processing_time = np.mean([analysis.processing_time_ms for analysis in analyses])
		
		# Detection summary by type
		detection_summary = {}
		for analysis in analyses:
			for detection in analysis.detections:
				obj_type = detection.object_type
				if obj_type not in detection_summary:
					detection_summary[obj_type] = {"count": 0, "avg_confidence": 0.0}
				detection_summary[obj_type]["count"] += 1
				detection_summary[obj_type]["avg_confidence"] += detection.confidence
		
		# Calculate averages
		for obj_type in detection_summary:
			count = detection_summary[obj_type]["count"]
			detection_summary[obj_type]["avg_confidence"] /= count
		
		return {
			"video_path": str(video_path),
			"output_path": str(output_path) if output_path else None,
			"frames_processed": len(analyses),
			"total_detections": total_detections,
			"avg_processing_time_ms": avg_processing_time,
			"detection_summary": detection_summary,
			"processing_timestamp": datetime.utcnow().isoformat()
		}
	
	async def enhance_image_quality(
		self,
		image_path: str,
		enhancement_type: str = "auto",
		output_path: str = None
	) -> Dict[str, Any]:
		"""Enhance image quality"""
		
		# Load image
		image = cv2.imread(str(image_path))
		if image is None:
			raise ValueError(f"Cannot load image: {image_path}")
		
		# Enhance image
		enhanced = await self.image_processor.enhance_image(image, enhancement_type)
		
		# Save enhanced image
		if output_path:
			cv2.imwrite(str(output_path), enhanced)
		
		# Extract features from both images for comparison
		original_features = await self.image_processor.extract_features(image)
		enhanced_features = await self.image_processor.extract_features(enhanced)
		
		return {
			"original_image": str(image_path),
			"enhanced_image": str(output_path) if output_path else None,
			"enhancement_type": enhancement_type,
			"quality_metrics": {
				"original": {
					"brightness": original_features["color_stats"]["brightness"],
					"contrast": original_features["color_stats"]["contrast"],
					"texture_score": original_features["texture_score"]
				},
				"enhanced": {
					"brightness": enhanced_features["color_stats"]["brightness"],
					"contrast": enhanced_features["color_stats"]["contrast"],
					"texture_score": enhanced_features["texture_score"]
				}
			},
			"timestamp": datetime.utcnow().isoformat()
		}
	
	def get_capability_info(self) -> Dict[str, Any]:
		"""Get capability information"""
		return {
			"name": "computer_vision",
			"version": "1.0.0",
			"description": "Advanced computer vision processing for object detection, face recognition, and video analysis",
			"features": [
				"Object detection (faces, persons, vehicles, general objects)",
				"Real-time video processing",
				"Image enhancement and quality improvement",
				"Feature extraction and analysis",
				"Multi-format support (images and videos)",
				"Configurable detection algorithms"
			],
			"supported_formats": {
				"images": ["jpg", "jpeg", "png", "bmp", "tiff"],
				"videos": ["mp4", "avi", "mov", "mkv", "webm"]
			},
			"dependencies": {
				"opencv-python": ">=4.5.0",
				"numpy": ">=1.19.0",
				"tensorflow": ">=2.6.0 (optional)",
				"torch": ">=1.9.0 (optional)",
				"pillow": ">=8.0.0 (optional)"
			},
			"detection_types": [dt.value for dt in DetectionType],
			"processing_modes": [pm.value for pm in ProcessingMode]
		}

# APG Integration
CAPABILITY_INFO = {
	"name": "computer_vision",
	"version": "1.0.0",
	"provides": ["ComputerVisionCapability"],
	"integrates_with": ["flask", "fastapi", "django", "streamlit"],
	"apg_templates": ["ai_app", "surveillance_system", "image_processor", "video_analytics"],
	"category": "ai_ml",
	"tags": ["computer-vision", "object-detection", "image-processing", "video-analysis", "ai"]
}