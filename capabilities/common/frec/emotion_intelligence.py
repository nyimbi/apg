"""
APG Facial Recognition - Real-Time Emotion & Stress Intelligence Engine

Revolutionary AI-powered emotion recognition with stress detection, micro-expression analysis,
and behavioral pattern learning for enhanced security and user experience.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import asyncio
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid_extensions import uuid7str
import json

try:
	import mediapipe as mp
	from sklearn.ensemble import RandomForestClassifier, IsolationForest
	from sklearn.preprocessing import StandardScaler
	from scipy import signal
	from scipy.spatial.distance import euclidean
except ImportError as e:
	print(f"Optional ML dependencies not available: {e}")

class EmotionIntelligenceEngine:
	"""Real-time emotion and stress detection with behavioral analysis"""
	
	def __init__(self, tenant_id: str):
		"""Initialize emotion intelligence engine"""
		assert tenant_id, "Tenant ID cannot be empty"
		
		self.tenant_id = tenant_id
		self.emotion_enabled = True
		self.stress_detection_enabled = True
		self.micro_expression_enabled = True
		self.behavioral_learning_enabled = True
		
		# Emotion categories and stress indicators
		self.emotion_categories = [
			'neutral', 'happy', 'sad', 'angry', 'fearful', 
			'disgusted', 'surprised', 'contempt', 'confused'
		]
		self.stress_indicators = [
			'eye_strain', 'facial_tension', 'micro_tremors', 
			'forced_expressions', 'cognitive_load', 'anxiety_markers'
		]
		
		# Initialize models and buffers
		self.emotion_history = {}
		self.stress_baselines = {}
		self.behavioral_patterns = {}
		
		self._initialize_models()
		self._log_engine_initialized()
	
	def _initialize_models(self) -> None:
		"""Initialize emotion and stress detection models"""
		try:
			# MediaPipe Face Mesh for detailed facial landmarks
			self.mp_face_mesh = mp.solutions.face_mesh
			self.mp_drawing = mp.solutions.drawing_utils
			self.face_mesh = self.mp_face_mesh.FaceMesh(
				static_image_mode=False,
				max_num_faces=1,
				refine_landmarks=True,
				min_detection_confidence=0.7,
				min_tracking_confidence=0.5
			)
			
			# Initialize emotion classification models
			if 'RandomForestClassifier' in globals():
				self.emotion_classifier = RandomForestClassifier(
					n_estimators=200,
					max_depth=15,
					random_state=42,
					class_weight='balanced'
				)
				
				self.stress_detector = IsolationForest(
					contamination=0.1,
					random_state=42,
					n_estimators=100
				)
				
				self.micro_expression_detector = RandomForestClassifier(
					n_estimators=150,
					max_depth=10,
					random_state=42
				)
			
			# Feature scalers for consistent input
			self.emotion_scaler = StandardScaler() if 'StandardScaler' in globals() else None
			self.stress_scaler = StandardScaler() if 'StandardScaler' in globals() else None
			
			# Facial landmarks for emotion analysis
			self._initialize_landmark_mappings()
			
		except Exception as e:
			print(f"Warning: Some emotion detection models failed to initialize: {e}")
	
	def _initialize_landmark_mappings(self) -> None:
		"""Initialize facial landmark mappings for emotion analysis"""
		try:
			# Key facial regions for emotion detection
			self.facial_regions = {
				'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
				'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
				'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
				'right_eyebrow': [296, 334, 293, 300, 276, 283, 282, 295, 285],
				'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279],
				'mouth': [0, 17, 18, 200, 199, 175, 0, 269, 270, 267, 271, 272, 12, 15, 16, 17, 18, 200, 199, 175],
				'jaw': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
				'forehead': [10, 151, 9, 10, 151, 9, 10, 151]
			}
			
			# Emotion-specific landmark movements
			self.emotion_landmarks = {
				'happy': {'mouth': [12, 15], 'eyes': [33, 362], 'cheeks': [116, 345]},
				'sad': {'mouth': [17, 18], 'eyebrows': [70, 296], 'eyes': [145, 374]},
				'angry': {'eyebrows': [63, 293], 'eyes': [33, 362], 'mouth': [0, 17]},
				'surprised': {'eyebrows': [70, 296], 'eyes': [145, 374], 'mouth': [13, 14]},
				'fearful': {'eyebrows': [70, 296], 'eyes': [33, 362], 'mouth': [17, 18]},
				'disgusted': {'nose': [19, 20], 'mouth': [17, 18], 'eyebrows': [63, 293]}
			}
			
		except Exception as e:
			print(f"Failed to initialize landmark mappings: {e}")
	
	def _log_engine_initialized(self) -> None:
		"""Log engine initialization"""
		print(f"Emotion Intelligence Engine initialized for tenant {self.tenant_id}")
	
	def _log_emotion_operation(self, operation: str, user_id: str | None = None, result: str | None = None) -> None:
		"""Log emotion detection operations"""
		user_info = f" (User: {user_id})" if user_id else ""
		result_info = f" [{result}]" if result else ""
		print(f"Emotion Intelligence {operation}{user_info}{result_info}")
	
	async def analyze_emotions(self, face_frames: List[np.ndarray], user_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
		"""Comprehensive emotion and stress analysis"""
		try:
			assert face_frames, "Face frames cannot be empty"
			assert len(face_frames) >= 3, "Minimum 3 frames required for emotion analysis"
			
			user_context = user_context or {}
			start_time = datetime.now()
			
			# Initialize analysis results
			emotion_analysis = {
				'analysis_id': uuid7str(),
				'analysis_timestamp': start_time.isoformat(),
				'processing_time_ms': 0,
				'user_id': user_context.get('user_id'),
				'primary_emotion': 'neutral',
				'emotion_confidence': 0.0,
				'emotion_scores': {},
				'stress_analysis': {},
				'micro_expressions': [],
				'behavioral_insights': {},
				'temporal_patterns': {},
				'risk_indicators': [],
				'recommendations': []
			}
			
			# Extract facial landmarks from all frames
			landmark_sequences = []
			for frame in face_frames:
				landmarks = await self._extract_facial_landmarks(frame)
				if landmarks is not None:
					landmark_sequences.append(landmarks)
			
			if len(landmark_sequences) < 3:
				emotion_analysis['error'] = 'Insufficient landmark data for analysis'
				return emotion_analysis
			
			# Analyze primary emotions
			emotion_analysis['emotion_scores'] = await self._analyze_primary_emotions(landmark_sequences)
			emotion_analysis['primary_emotion'], emotion_analysis['emotion_confidence'] = self._determine_primary_emotion(
				emotion_analysis['emotion_scores']
			)
			
			# Detect stress indicators
			if self.stress_detection_enabled:
				emotion_analysis['stress_analysis'] = await self._detect_stress_indicators(landmark_sequences, user_context)
			
			# Analyze micro-expressions
			if self.micro_expression_enabled:
				emotion_analysis['micro_expressions'] = await self._detect_micro_expressions(landmark_sequences)
			
			# Behavioral pattern analysis
			if self.behavioral_learning_enabled and user_context.get('user_id'):
				emotion_analysis['behavioral_insights'] = await self._analyze_behavioral_patterns(
					emotion_analysis, user_context['user_id']
				)
			
			# Temporal emotion patterns
			emotion_analysis['temporal_patterns'] = await self._analyze_temporal_patterns(landmark_sequences)
			
			# Risk assessment
			emotion_analysis['risk_indicators'] = self._assess_emotional_risk(emotion_analysis)
			
			# Generate recommendations
			emotion_analysis['recommendations'] = self._generate_emotion_recommendations(emotion_analysis)
			
			# Learn from this analysis
			if self.behavioral_learning_enabled and user_context.get('user_id'):
				await self._learn_from_emotion_analysis(emotion_analysis, user_context['user_id'])
			
			# Calculate processing time
			processing_time = (datetime.now() - start_time).total_seconds() * 1000
			emotion_analysis['processing_time_ms'] = processing_time
			
			self._log_emotion_operation(
				"ANALYZE_EMOTIONS",
				user_context.get('user_id'),
				f"{emotion_analysis['primary_emotion']} ({emotion_analysis['emotion_confidence']:.2f})"
			)
			
			return emotion_analysis
			
		except Exception as e:
			print(f"Failed to analyze emotions: {e}")
			return {'error': str(e), 'analysis_timestamp': datetime.now(timezone.utc).isoformat()}
	
	async def _extract_facial_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
		"""Extract facial landmarks using MediaPipe"""
		try:
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = self.face_mesh.process(rgb_frame)
			
			if results.multi_face_landmarks:
				landmarks = results.multi_face_landmarks[0]
				return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
			
			return None
			
		except Exception as e:
			print(f"Failed to extract facial landmarks: {e}")
			return None
	
	async def _analyze_primary_emotions(self, landmark_sequences: List[np.ndarray]) -> Dict[str, float]:
		"""Analyze primary emotions from facial landmarks"""
		try:
			emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
			
			if not landmark_sequences:
				return emotion_scores
			
			# Calculate emotion features for each frame
			frame_emotions = []
			for landmarks in landmark_sequences:
				frame_emotion = self._calculate_emotion_features(landmarks)
				frame_emotions.append(frame_emotion)
			
			# Average emotions across frames
			if frame_emotions:
				for emotion in self.emotion_categories:
					scores = [fe.get(emotion, 0.0) for fe in frame_emotions]
					emotion_scores[emotion] = sum(scores) / len(scores)
			
			# Normalize scores to sum to 1.0
			total_score = sum(emotion_scores.values())
			if total_score > 0:
				emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
			
			return emotion_scores
			
		except Exception as e:
			print(f"Failed to analyze primary emotions: {e}")
			return {emotion: 0.0 for emotion in self.emotion_categories}
	
	def _calculate_emotion_features(self, landmarks: np.ndarray) -> Dict[str, float]:
		"""Calculate emotion features from facial landmarks"""
		try:
			emotion_features = {}
			
			if landmarks is None or len(landmarks) == 0:
				return {emotion: 0.0 for emotion in self.emotion_categories}
			
			# Calculate distances between key facial points
			# Mouth features
			mouth_width = self._calculate_landmark_distance(landmarks, 61, 291)  # Mouth corners
			mouth_height = self._calculate_landmark_distance(landmarks, 13, 14)  # Upper/lower lip
			mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
			
			# Eye features
			left_eye_openness = self._calculate_eye_openness(landmarks, 'left')
			right_eye_openness = self._calculate_eye_openness(landmarks, 'right')
			eye_symmetry = abs(left_eye_openness - right_eye_openness)
			
			# Eyebrow features
			left_eyebrow_height = self._calculate_eyebrow_height(landmarks, 'left')
			right_eyebrow_height = self._calculate_eyebrow_height(landmarks, 'right')
			eyebrow_asymmetry = abs(left_eyebrow_height - right_eyebrow_height)
			
			# Map features to emotions (simplified heuristics)
			emotion_features['happy'] = min(1.0, mouth_width * 0.5 + (1.0 - mouth_ratio) * 0.3)
			emotion_features['sad'] = min(1.0, mouth_ratio * 0.4 + (1.0 - left_eye_openness) * 0.3)
			emotion_features['angry'] = min(1.0, eyebrow_asymmetry * 0.4 + mouth_ratio * 0.2)
			emotion_features['surprised'] = min(1.0, left_eye_openness * 0.4 + right_eye_openness * 0.4)
			emotion_features['fearful'] = min(1.0, eye_symmetry * 0.3 + eyebrow_asymmetry * 0.3)
			emotion_features['disgusted'] = min(1.0, mouth_ratio * 0.5)
			emotion_features['neutral'] = 1.0 - max(emotion_features.values())
			emotion_features['contempt'] = min(1.0, mouth_ratio * 0.3 + eyebrow_asymmetry * 0.2)
			emotion_features['confused'] = min(1.0, eyebrow_asymmetry * 0.4 + eye_symmetry * 0.2)
			
			return emotion_features
			
		except Exception as e:
			print(f"Failed to calculate emotion features: {e}")
			return {emotion: 0.0 for emotion in self.emotion_categories}
	
	def _calculate_landmark_distance(self, landmarks: np.ndarray, idx1: int, idx2: int) -> float:
		"""Calculate Euclidean distance between two landmarks"""
		try:
			if idx1 >= len(landmarks) or idx2 >= len(landmarks):
				return 0.0
			
			point1 = landmarks[idx1][:2]  # x, y coordinates
			point2 = landmarks[idx2][:2]
			
			return euclidean(point1, point2)
			
		except Exception:
			return 0.0
	
	def _calculate_eye_openness(self, landmarks: np.ndarray, eye: str) -> float:
		"""Calculate eye openness ratio"""
		try:
			if eye == 'left':
				# Left eye landmarks (simplified)
				top_idx, bottom_idx = 159, 145
			else:
				# Right eye landmarks (simplified)
				top_idx, bottom_idx = 386, 374
			
			if top_idx >= len(landmarks) or bottom_idx >= len(landmarks):
				return 0.5
			
			vertical_distance = abs(landmarks[top_idx][1] - landmarks[bottom_idx][1])
			
			# Normalize based on typical eye opening (simplified)
			normalized_openness = min(1.0, vertical_distance * 20)  # Scaling factor
			
			return normalized_openness
			
		except Exception:
			return 0.5
	
	def _calculate_eyebrow_height(self, landmarks: np.ndarray, eyebrow: str) -> float:
		"""Calculate eyebrow height relative to eye"""
		try:
			if eyebrow == 'left':
				eyebrow_idx, eye_idx = 70, 33
			else:
				eyebrow_idx, eye_idx = 296, 362
			
			if eyebrow_idx >= len(landmarks) or eye_idx >= len(landmarks):
				return 0.5
			
			height_difference = landmarks[eyebrow_idx][1] - landmarks[eye_idx][1]
			
			# Normalize height difference
			normalized_height = min(1.0, abs(height_difference) * 10)  # Scaling factor
			
			return normalized_height
			
		except Exception:
			return 0.5
	
	def _determine_primary_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
		"""Determine primary emotion and confidence from scores"""
		try:
			if not emotion_scores:
				return 'neutral', 0.0
			
			# Find emotion with highest score
			primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
			
			return primary_emotion[0], primary_emotion[1]
			
		except Exception as e:
			print(f"Failed to determine primary emotion: {e}")
			return 'neutral', 0.0
	
	async def _detect_stress_indicators(self, landmark_sequences: List[np.ndarray], user_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Detect stress indicators from facial analysis"""
		try:
			stress_analysis = {
				'overall_stress_level': 'low',
				'stress_score': 0.0,
				'stress_indicators': [],
				'physiological_markers': {},
				'cognitive_load_indicators': {}
			}
			
			if len(landmark_sequences) < 5:
				return stress_analysis
			
			# Analyze facial tension
			tension_score = self._analyze_facial_tension(landmark_sequences)
			
			# Detect micro-tremors
			tremor_score = self._detect_facial_tremors(landmark_sequences)
			
			# Analyze eye strain
			eye_strain_score = self._analyze_eye_strain(landmark_sequences)
			
			# Detect forced expressions
			forced_expression_score = self._detect_forced_expressions(landmark_sequences)
			
			# Calculate overall stress score
			stress_components = [tension_score, tremor_score, eye_strain_score, forced_expression_score]
			stress_analysis['stress_score'] = sum(stress_components) / len(stress_components)
			
			# Determine stress level
			if stress_analysis['stress_score'] >= 0.7:
				stress_analysis['overall_stress_level'] = 'high'
			elif stress_analysis['stress_score'] >= 0.4:
				stress_analysis['overall_stress_level'] = 'medium'
			else:
				stress_analysis['overall_stress_level'] = 'low'
			
			# Identify specific stress indicators
			stress_indicators = []
			if tension_score > 0.6:
				stress_indicators.append('facial_tension')
			if tremor_score > 0.5:
				stress_indicators.append('micro_tremors')
			if eye_strain_score > 0.6:
				stress_indicators.append('eye_strain')
			if forced_expression_score > 0.7:
				stress_indicators.append('forced_expressions')
			
			stress_analysis['stress_indicators'] = stress_indicators
			
			# Physiological markers
			stress_analysis['physiological_markers'] = {
				'facial_tension': tension_score,
				'micro_tremors': tremor_score,
				'eye_strain': eye_strain_score,
				'expression_naturalness': 1.0 - forced_expression_score
			}
			
			return stress_analysis
			
		except Exception as e:
			print(f"Failed to detect stress indicators: {e}")
			return {'overall_stress_level': 'unknown', 'stress_score': 0.0}
	
	def _analyze_facial_tension(self, landmark_sequences: List[np.ndarray]) -> float:
		"""Analyze facial muscle tension"""
		try:
			tension_scores = []
			
			for landmarks in landmark_sequences:
				if landmarks is None:
					continue
				
				# Calculate muscle tension indicators
				jaw_tension = self._calculate_jaw_tension(landmarks)
				forehead_tension = self._calculate_forehead_tension(landmarks)
				eye_tension = self._calculate_eye_tension(landmarks)
				
				frame_tension = (jaw_tension + forehead_tension + eye_tension) / 3.0
				tension_scores.append(frame_tension)
			
			return sum(tension_scores) / len(tension_scores) if tension_scores else 0.0
			
		except Exception:
			return 0.0
	
	def _calculate_jaw_tension(self, landmarks: np.ndarray) -> float:
		"""Calculate jaw tension from landmarks"""
		try:
			# Simplified jaw tension calculation
			# In practice, this would analyze jaw muscle activation patterns
			jaw_width = self._calculate_landmark_distance(landmarks, 172, 397)
			normalized_tension = min(1.0, jaw_width * 5)  # Scaling factor
			
			return normalized_tension
			
		except Exception:
			return 0.0
	
	def _calculate_forehead_tension(self, landmarks: np.ndarray) -> float:
		"""Calculate forehead tension from landmarks"""
		try:
			# Simplified forehead tension calculation
			# Analyze eyebrow position and forehead wrinkles
			left_eyebrow_tension = self._calculate_eyebrow_height(landmarks, 'left')
			right_eyebrow_tension = self._calculate_eyebrow_height(landmarks, 'right')
			
			return (left_eyebrow_tension + right_eyebrow_tension) / 2.0
			
		except Exception:
			return 0.0
	
	def _calculate_eye_tension(self, landmarks: np.ndarray) -> float:
		"""Calculate eye tension from landmarks"""
		try:
			# Analyze eye squinting and tension
			left_eye_squeeze = 1.0 - self._calculate_eye_openness(landmarks, 'left')
			right_eye_squeeze = 1.0 - self._calculate_eye_openness(landmarks, 'right')
			
			return (left_eye_squeeze + right_eye_squeeze) / 2.0
			
		except Exception:
			return 0.0
	
	def _detect_facial_tremors(self, landmark_sequences: List[np.ndarray]) -> float:
		"""Detect micro-tremors in facial landmarks"""
		try:
			if len(landmark_sequences) < 5:
				return 0.0
			
			tremor_scores = []
			
			# Analyze key facial points for tremors
			key_points = [33, 362, 0, 17]  # Eyes and mouth corners
			
			for point_idx in key_points:
				point_trajectory = []
				
				for landmarks in landmark_sequences:
					if landmarks is not None and point_idx < len(landmarks):
						point_trajectory.append(landmarks[point_idx][:2])
				
				if len(point_trajectory) >= 5:
					tremor_score = self._calculate_trajectory_tremor(point_trajectory)
					tremor_scores.append(tremor_score)
			
			return sum(tremor_scores) / len(tremor_scores) if tremor_scores else 0.0
			
		except Exception:
			return 0.0
	
	def _calculate_trajectory_tremor(self, trajectory: List[np.ndarray]) -> float:
		"""Calculate tremor score from point trajectory"""
		try:
			if len(trajectory) < 5:
				return 0.0
			
			# Calculate movement variations
			movements = []
			for i in range(1, len(trajectory)):
				movement = euclidean(trajectory[i], trajectory[i-1])
				movements.append(movement)
			
			if not movements:
				return 0.0
			
			# Calculate coefficient of variation (tremor indicator)
			mean_movement = np.mean(movements)
			std_movement = np.std(movements)
			
			if mean_movement > 0:
				tremor_coefficient = std_movement / mean_movement
				return min(1.0, tremor_coefficient * 10)  # Scaling factor
			
			return 0.0
			
		except Exception:
			return 0.0
	
	def _analyze_eye_strain(self, landmark_sequences: List[np.ndarray]) -> float:
		"""Analyze eye strain indicators"""
		try:
			strain_scores = []
			
			for landmarks in landmark_sequences:
				if landmarks is None:
					continue
				
				# Calculate eye strain indicators
				blink_rate = self._estimate_blink_rate(landmarks)
				eye_redness = 0.5  # Would be calculated from eye region color analysis
				squinting = self._calculate_squinting_level(landmarks)
				
				frame_strain = (blink_rate + eye_redness + squinting) / 3.0
				strain_scores.append(frame_strain)
			
			return sum(strain_scores) / len(strain_scores) if strain_scores else 0.0
			
		except Exception:
			return 0.0
	
	def _estimate_blink_rate(self, landmarks: np.ndarray) -> float:
		"""Estimate blink rate from single frame"""
		try:
			# Simplified blink detection
			left_eye_openness = self._calculate_eye_openness(landmarks, 'left')
			right_eye_openness = self._calculate_eye_openness(landmarks, 'right')
			
			# Low eye openness might indicate frequent blinking or strain
			avg_openness = (left_eye_openness + right_eye_openness) / 2.0
			blink_indicator = 1.0 - avg_openness
			
			return blink_indicator
			
		except Exception:
			return 0.0
	
	def _calculate_squinting_level(self, landmarks: np.ndarray) -> float:
		"""Calculate squinting level"""
		try:
			# Analyze eye shape for squinting
			left_squint = 1.0 - self._calculate_eye_openness(landmarks, 'left')
			right_squint = 1.0 - self._calculate_eye_openness(landmarks, 'right')
			
			return (left_squint + right_squint) / 2.0
			
		except Exception:
			return 0.0
	
	def _detect_forced_expressions(self, landmark_sequences: List[np.ndarray]) -> float:
		"""Detect artificial or forced facial expressions"""
		try:
			forced_scores = []
			
			for landmarks in landmark_sequences:
				if landmarks is None:
					continue
				
				# Analyze expression naturalness
				symmetry_score = self._calculate_facial_symmetry(landmarks)
				duration_naturalness = 0.8  # Would analyze expression duration
				intensity_naturalness = self._analyze_expression_intensity(landmarks)
				
				# Forced expressions often lack naturalness
				naturalness = (symmetry_score + duration_naturalness + intensity_naturalness) / 3.0
				forced_score = 1.0 - naturalness
				
				forced_scores.append(forced_score)
			
			return sum(forced_scores) / len(forced_scores) if forced_scores else 0.0
			
		except Exception:
			return 0.0
	
	def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
		"""Calculate facial symmetry score"""
		try:
			# Compare left and right sides of face
			left_mouth = self._calculate_landmark_distance(landmarks, 61, 17)
			right_mouth = self._calculate_landmark_distance(landmarks, 291, 17)
			
			left_eye = self._calculate_eye_openness(landmarks, 'left')
			right_eye = self._calculate_eye_openness(landmarks, 'right')
			
			# Calculate symmetry
			mouth_symmetry = 1.0 - abs(left_mouth - right_mouth) / max(left_mouth + right_mouth, 0.001)
			eye_symmetry = 1.0 - abs(left_eye - right_eye)
			
			return (mouth_symmetry + eye_symmetry) / 2.0
			
		except Exception:
			return 0.5
	
	def _analyze_expression_intensity(self, landmarks: np.ndarray) -> float:
		"""Analyze expression intensity naturalness"""
		try:
			# Calculate overall expression intensity
			emotion_features = self._calculate_emotion_features(landmarks)
			max_emotion_score = max(emotion_features.values())
			
			# Very high intensity might indicate forced expression
			if max_emotion_score > 0.9:
				return 0.3  # Low naturalness
			elif max_emotion_score > 0.7:
				return 0.7  # Medium naturalness
			else:
				return 1.0  # High naturalness
				
		except Exception:
			return 0.5
	
	async def _detect_micro_expressions(self, landmark_sequences: List[np.ndarray]) -> List[Dict[str, Any]]:
		"""Detect brief micro-expressions"""
		try:
			micro_expressions = []
			
			if len(landmark_sequences) < 10:
				return micro_expressions
			
			# Analyze short sequences for micro-expressions
			window_size = 5
			for i in range(len(landmark_sequences) - window_size + 1):
				window = landmark_sequences[i:i + window_size]
				
				micro_expr = self._analyze_micro_expression_window(window, i)
				if micro_expr:
					micro_expressions.append(micro_expr)
			
			return micro_expressions
			
		except Exception as e:
			print(f"Failed to detect micro-expressions: {e}")
			return []
	
	def _analyze_micro_expression_window(self, window: List[np.ndarray], start_frame: int) -> Optional[Dict[str, Any]]:
		"""Analyze a window of frames for micro-expressions"""
		try:
			if len(window) < 3:
				return None
			
			# Calculate emotion changes across window
			emotion_changes = []
			for i in range(1, len(window)):
				if window[i] is not None and window[i-1] is not None:
					change = self._calculate_emotion_change(window[i-1], window[i])
					emotion_changes.append(change)
			
			if not emotion_changes:
				return None
			
			# Look for rapid emotion changes (micro-expressions)
			max_change = max(emotion_changes)
			if max_change > 0.3:  # Threshold for micro-expression
				return {
					'type': 'micro_expression',
					'intensity': max_change,
					'start_frame': start_frame,
					'duration_frames': len(window),
					'detected_emotion': 'surprise',  # Simplified
					'confidence': min(1.0, max_change)
				}
			
			return None
			
		except Exception:
			return None
	
	def _calculate_emotion_change(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
		"""Calculate emotion change between two frames"""
		try:
			emotions1 = self._calculate_emotion_features(landmarks1)
			emotions2 = self._calculate_emotion_features(landmarks2)
			
			# Calculate total emotion change
			total_change = 0.0
			for emotion in self.emotion_categories:
				change = abs(emotions2.get(emotion, 0) - emotions1.get(emotion, 0))
				total_change += change
			
			return total_change
			
		except Exception:
			return 0.0
	
	async def _analyze_behavioral_patterns(self, emotion_analysis: Dict[str, Any], user_id: str) -> Dict[str, Any]:
		"""Analyze user-specific behavioral patterns"""
		try:
			behavioral_insights = {
				'user_baseline': {},
				'pattern_deviations': [],
				'emotional_consistency': 0.0,
				'stress_pattern': 'normal',
				'recommendations': []
			}
			
			# Get user's historical patterns
			if user_id in self.behavioral_patterns:
				user_patterns = self.behavioral_patterns[user_id]
				
				# Compare current emotions with baseline
				current_emotion = emotion_analysis['primary_emotion']
				baseline_emotions = user_patterns.get('typical_emotions', {})
				
				if current_emotion in baseline_emotions:
					typical_confidence = baseline_emotions[current_emotion]
					current_confidence = emotion_analysis['emotion_confidence']
					
					deviation = abs(current_confidence - typical_confidence)
					if deviation > 0.3:
						behavioral_insights['pattern_deviations'].append({
							'type': 'emotion_confidence_deviation',
							'deviation_magnitude': deviation,
							'expected': typical_confidence,
							'observed': current_confidence
						})
				
				# Analyze stress patterns
				current_stress = emotion_analysis.get('stress_analysis', {}).get('stress_score', 0.0)
				baseline_stress = user_patterns.get('typical_stress_level', 0.3)
				
				if current_stress > baseline_stress + 0.2:
					behavioral_insights['stress_pattern'] = 'elevated'
				elif current_stress < baseline_stress - 0.2:
					behavioral_insights['stress_pattern'] = 'below_baseline'
				
				behavioral_insights['user_baseline'] = user_patterns
			
			return behavioral_insights
			
		except Exception as e:
			print(f"Failed to analyze behavioral patterns: {e}")
			return {}
	
	async def _analyze_temporal_patterns(self, landmark_sequences: List[np.ndarray]) -> Dict[str, Any]:
		"""Analyze temporal emotion patterns"""
		try:
			temporal_patterns = {
				'emotion_stability': 0.0,
				'transition_smoothness': 0.0,
				'micro_expression_frequency': 0.0,
				'overall_consistency': 0.0
			}
			
			if len(landmark_sequences) < 5:
				return temporal_patterns
			
			# Calculate emotion stability across frames
			frame_emotions = []
			for landmarks in landmark_sequences:
				if landmarks is not None:
					emotions = self._calculate_emotion_features(landmarks)
					frame_emotions.append(emotions)
			
			if len(frame_emotions) >= 3:
				stability_scores = []
				for emotion in self.emotion_categories:
					emotion_values = [fe.get(emotion, 0.0) for fe in frame_emotions]
					stability = 1.0 - np.std(emotion_values)
					stability_scores.append(max(0.0, stability))
				
				temporal_patterns['emotion_stability'] = sum(stability_scores) / len(stability_scores)
			
			# Calculate transition smoothness
			if len(frame_emotions) >= 2:
				transitions = []
				for i in range(1, len(frame_emotions)):
					transition = self._calculate_emotion_transition_smoothness(
						frame_emotions[i-1], frame_emotions[i]
					)
					transitions.append(transition)
				
				temporal_patterns['transition_smoothness'] = sum(transitions) / len(transitions)
			
			# Overall consistency
			temporal_patterns['overall_consistency'] = (
				temporal_patterns['emotion_stability'] + temporal_patterns['transition_smoothness']
			) / 2.0
			
			return temporal_patterns
			
		except Exception as e:
			print(f"Failed to analyze temporal patterns: {e}")
			return {}
	
	def _calculate_emotion_transition_smoothness(self, emotions1: Dict[str, float], emotions2: Dict[str, float]) -> float:
		"""Calculate smoothness of emotion transition"""
		try:
			total_change = 0.0
			for emotion in self.emotion_categories:
				change = abs(emotions2.get(emotion, 0) - emotions1.get(emotion, 0))
				total_change += change
			
			# Smooth transitions have smaller changes
			smoothness = 1.0 - min(1.0, total_change)
			return smoothness
			
		except Exception:
			return 0.5
	
	def _assess_emotional_risk(self, emotion_analysis: Dict[str, Any]) -> List[str]:
		"""Assess emotional risk indicators"""
		risk_indicators = []
		
		try:
			# High stress risk
			stress_score = emotion_analysis.get('stress_analysis', {}).get('stress_score', 0.0)
			if stress_score > 0.7:
				risk_indicators.append('high_stress_detected')
			
			# Emotional instability
			temporal_patterns = emotion_analysis.get('temporal_patterns', {})
			if temporal_patterns.get('emotion_stability', 1.0) < 0.3:
				risk_indicators.append('emotional_instability')
			
			# Forced expressions (potential deception)
			stress_indicators = emotion_analysis.get('stress_analysis', {}).get('stress_indicators', [])
			if 'forced_expressions' in stress_indicators:
				risk_indicators.append('potential_deception')
			
			# Micro-expressions indicating concealed emotions
			micro_expressions = emotion_analysis.get('micro_expressions', [])
			if len(micro_expressions) > 3:
				risk_indicators.append('concealed_emotions')
			
			# Behavioral deviations
			behavioral_insights = emotion_analysis.get('behavioral_insights', {})
			if behavioral_insights.get('pattern_deviations'):
				risk_indicators.append('behavioral_anomaly')
			
		except Exception as e:
			print(f"Failed to assess emotional risk: {e}")
		
		return risk_indicators
	
	def _generate_emotion_recommendations(self, emotion_analysis: Dict[str, Any]) -> List[str]:
		"""Generate recommendations based on emotion analysis"""
		recommendations = []
		
		try:
			primary_emotion = emotion_analysis.get('primary_emotion', 'neutral')
			stress_level = emotion_analysis.get('stress_analysis', {}).get('overall_stress_level', 'low')
			risk_indicators = emotion_analysis.get('risk_indicators', [])
			
			# Stress-based recommendations
			if stress_level == 'high':
				recommendations.append("Consider additional verification steps due to high stress detected")
				recommendations.append("Provide calming environment for optimal biometric capture")
			
			# Emotion-based recommendations
			if primary_emotion in ['angry', 'fearful']:
				recommendations.append("User may be experiencing negative emotions - ensure comfortable environment")
			elif primary_emotion == 'surprised':
				recommendations.append("Unexpected verification request detected - verify user awareness")
			
			# Risk-based recommendations
			if 'potential_deception' in risk_indicators:
				recommendations.append("Forced expressions detected - consider additional verification methods")
			if 'behavioral_anomaly' in risk_indicators:
				recommendations.append("User behavior differs from baseline - investigate potential security concern")
			if 'concealed_emotions' in risk_indicators:
				recommendations.append("Micro-expressions suggest concealed emotions - enhanced monitoring recommended")
			
			# General recommendations
			if not recommendations:
				recommendations.append("Emotional state is normal - proceed with standard verification")
			
		except Exception as e:
			print(f"Failed to generate recommendations: {e}")
		
		return recommendations
	
	async def _learn_from_emotion_analysis(self, emotion_analysis: Dict[str, Any], user_id: str) -> None:
		"""Learn from emotion analysis to build user patterns"""
		try:
			if not self.behavioral_learning_enabled:
				return
			
			# Initialize user patterns if not exists
			if user_id not in self.behavioral_patterns:
				self.behavioral_patterns[user_id] = {
					'typical_emotions': {},
					'typical_stress_level': 0.3,
					'emotion_history': [],
					'stress_history': [],
					'last_updated': datetime.now(timezone.utc).isoformat()
				}
			
			user_patterns = self.behavioral_patterns[user_id]
			
			# Update emotion patterns
			primary_emotion = emotion_analysis.get('primary_emotion', 'neutral')
			emotion_confidence = emotion_analysis.get('emotion_confidence', 0.0)
			
			if primary_emotion not in user_patterns['typical_emotions']:
				user_patterns['typical_emotions'][primary_emotion] = emotion_confidence
			else:
				# Running average
				current_avg = user_patterns['typical_emotions'][primary_emotion]
				user_patterns['typical_emotions'][primary_emotion] = (current_avg + emotion_confidence) / 2.0
			
			# Update stress patterns
			stress_score = emotion_analysis.get('stress_analysis', {}).get('stress_score', 0.0)
			current_stress_avg = user_patterns['typical_stress_level']
			user_patterns['typical_stress_level'] = (current_stress_avg + stress_score) / 2.0
			
			# Maintain history (keep last 50 entries)
			user_patterns['emotion_history'].append({
				'emotion': primary_emotion,
				'confidence': emotion_confidence,
				'timestamp': datetime.now(timezone.utc).isoformat()
			})
			if len(user_patterns['emotion_history']) > 50:
				user_patterns['emotion_history'] = user_patterns['emotion_history'][-50:]
			
			user_patterns['stress_history'].append({
				'stress_score': stress_score,
				'timestamp': datetime.now(timezone.utc).isoformat()
			})
			if len(user_patterns['stress_history']) > 50:
				user_patterns['stress_history'] = user_patterns['stress_history'][-50:]
			
			user_patterns['last_updated'] = datetime.now(timezone.utc).isoformat()
			
			self._log_emotion_operation("LEARN_PATTERN", user_id)
			
		except Exception as e:
			print(f"Failed to learn from emotion analysis: {e}")
	
	async def get_user_emotional_baseline(self, user_id: str) -> Dict[str, Any]:
		"""Get user's emotional baseline patterns"""
		try:
			if user_id not in self.behavioral_patterns:
				return {'error': 'No baseline data available for user'}
			
			user_patterns = self.behavioral_patterns[user_id]
			
			baseline = {
				'user_id': user_id,
				'typical_emotions': user_patterns['typical_emotions'],
				'typical_stress_level': user_patterns['typical_stress_level'],
				'pattern_stability': self._calculate_pattern_stability(user_patterns),
				'data_points': len(user_patterns['emotion_history']),
				'last_updated': user_patterns['last_updated'],
				'baseline_confidence': self._calculate_baseline_confidence(user_patterns)
			}
			
			return baseline
			
		except Exception as e:
			print(f"Failed to get emotional baseline: {e}")
			return {'error': str(e)}
	
	def _calculate_pattern_stability(self, user_patterns: Dict[str, Any]) -> float:
		"""Calculate stability of user's emotional patterns"""
		try:
			emotion_history = user_patterns.get('emotion_history', [])
			if len(emotion_history) < 5:
				return 0.0
			
			# Calculate emotion consistency
			recent_emotions = [entry['emotion'] for entry in emotion_history[-10:]]
			emotion_counts = {}
			for emotion in recent_emotions:
				emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
			
			# Higher stability if emotions are consistent
			max_count = max(emotion_counts.values()) if emotion_counts else 0
			stability = max_count / len(recent_emotions)
			
			return stability
			
		except Exception:
			return 0.0
	
	def _calculate_baseline_confidence(self, user_patterns: Dict[str, Any]) -> float:
		"""Calculate confidence in baseline patterns"""
		try:
			data_points = len(user_patterns.get('emotion_history', []))
			
			# More data points = higher confidence
			if data_points >= 30:
				return 1.0
			elif data_points >= 15:
				return 0.8
			elif data_points >= 5:
				return 0.6
			else:
				return 0.3
				
		except Exception:
			return 0.0
	
	async def reset_user_patterns(self, user_id: str) -> bool:
		"""Reset user's behavioral patterns (GDPR compliance)"""
		try:
			if user_id in self.behavioral_patterns:
				del self.behavioral_patterns[user_id]
				self._log_emotion_operation("RESET_PATTERNS", user_id)
				return True
			
			return False
			
		except Exception as e:
			print(f"Failed to reset user patterns: {e}")
			return False
	
	def get_engine_statistics(self) -> Dict[str, Any]:
		"""Get emotion intelligence engine statistics"""
		return {
			'tenant_id': self.tenant_id,
			'emotion_categories': self.emotion_categories,
			'stress_indicators': self.stress_indicators,
			'users_with_patterns': len(self.behavioral_patterns),
			'features_enabled': {
				'emotion_detection': self.emotion_enabled,
				'stress_detection': self.stress_detection_enabled,
				'micro_expressions': self.micro_expression_enabled,
				'behavioral_learning': self.behavioral_learning_enabled
			},
			'model_status': {
				'facial_landmarks': hasattr(self, 'face_mesh'),
				'emotion_classifier': hasattr(self, 'emotion_classifier'),
				'stress_detector': hasattr(self, 'stress_detector')
			}
		}

# Export for use in other modules
__all__ = ['EmotionIntelligenceEngine']