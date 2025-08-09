"""
Behavioral Authentication Engine

ML-powered continuous behavioral authentication system providing seamless
security through pattern analysis and anomaly detection.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import numpy as np
import hashlib
import json
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, validator
from dataclasses import dataclass, field

class BehavioralEventType(str, Enum):
	"""Types of behavioral events to track"""
	KEYSTROKE = "keystroke"
	MOUSE_MOVEMENT = "mouse_movement"
	CLICK_PATTERN = "click_pattern"
	NAVIGATION = "navigation"
	TYPING_RHYTHM = "typing_rhythm"
	SCROLL_BEHAVIOR = "scroll_behavior"
	DEVICE_ORIENTATION = "device_orientation"
	TOUCH_PRESSURE = "touch_pressure"
	INTERACTION_TIMING = "interaction_timing"

class BehavioralRiskLevel(str, Enum):
	"""Risk levels for behavioral analysis"""
	VERY_LOW = "very_low"
	LOW = "low"
	MODERATE = "moderate"
	HIGH = "high"
	VERY_HIGH = "very_high"
	CRITICAL = "critical"

class AuthScore(BaseModel):
	"""Authentication confidence score"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	confidence: float = Field(..., description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
	method: str = Field(..., description="Authentication method used")
	risk_level: BehavioralRiskLevel = Field(..., description="Risk level assessment")
	factors: Dict[str, float] = Field(default_factory=dict, description="Individual factor scores")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Score timestamp")

class BehavioralPattern(BaseModel):
	"""Individual behavioral pattern model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Pattern identifier")
	user_id: str = Field(..., description="User identifier")
	event_type: BehavioralEventType = Field(..., description="Type of behavioral event")
	
	# Pattern features
	features: Dict[str, float] = Field(..., description="Extracted behavioral features")
	feature_weights: Dict[str, float] = Field(default_factory=dict, description="Feature importance weights")
	
	# Statistical measures
	mean_values: Dict[str, float] = Field(default_factory=dict, description="Mean feature values")
	std_values: Dict[str, float] = Field(default_factory=dict, description="Standard deviation values")
	confidence_interval: Dict[str, Tuple[float, float]] = Field(default_factory=dict, description="95% confidence intervals")
	
	# Metadata
	sample_count: int = Field(default=0, description="Number of samples used to build pattern")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last pattern update")
	stability_score: float = Field(default=0.0, description="Pattern stability (0.0-1.0)")

class BehavioralBaseline(BaseModel):
	"""User's behavioral baseline model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Baseline identifier")
	user_id: str = Field(..., description="User identifier")
	
	# Behavioral patterns by type
	patterns: Dict[BehavioralEventType, BehavioralPattern] = Field(
		default_factory=dict, description="Behavioral patterns by event type"
	)
	
	# Baseline metadata
	establishment_date: datetime = Field(default_factory=datetime.utcnow, description="When baseline was established")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last baseline update")
	maturity_level: float = Field(default=0.0, description="Baseline maturity (0.0-1.0)")
	confidence_level: float = Field(default=0.0, description="Overall confidence in baseline")
	
	# Learning parameters
	learning_samples_required: int = Field(default=100, description="Samples needed for stable baseline")
	current_sample_count: int = Field(default=0, description="Current number of samples")
	adaptive_learning_enabled: bool = Field(default=True, description="Enable continuous learning")
	
	def is_mature(self) -> bool:
		"""Check if baseline is mature enough for authentication"""
		return (self.maturity_level >= 0.7 and 
				self.current_sample_count >= self.learning_samples_required)
	
	def update_timestamp(self):
		"""Update the last updated timestamp"""
		self.last_updated = datetime.utcnow()

class BehavioralEvent(BaseModel):
	"""Single behavioral event record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Event identifier")
	user_id: str = Field(..., description="User identifier")
	session_id: str = Field(..., description="Session identifier")
	event_type: BehavioralEventType = Field(..., description="Event type")
	
	# Event data
	raw_data: Dict[str, Any] = Field(..., description="Raw event data")
	processed_features: Dict[str, float] = Field(default_factory=dict, description="Processed feature vector")
	
	# Context
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
	device_info: Dict[str, str] = Field(default_factory=dict, description="Device context")
	page_context: Optional[str] = Field(default=None, description="Page/view context")
	
	# Analysis results
	anomaly_score: Optional[float] = Field(default=None, description="Anomaly detection score")
	confidence_score: Optional[float] = Field(default=None, description="Pattern matching confidence")

@dataclass
class FeatureExtractor:
	"""Behavioral feature extraction utilities"""
	
	@staticmethod
	def extract_keystroke_features(keystroke_data: List[Dict[str, Any]]) -> Dict[str, float]:
		"""Extract features from keystroke dynamics"""
		if not keystroke_data:
			return {}
		
		# Extract timing features
		dwell_times = [event.get('dwell_time', 0) for event in keystroke_data if event.get('dwell_time')]
		flight_times = [event.get('flight_time', 0) for event in keystroke_data if event.get('flight_time')]
		
		features = {}
		
		if dwell_times:
			features.update({
				'avg_dwell_time': np.mean(dwell_times),
				'std_dwell_time': np.std(dwell_times),
				'min_dwell_time': np.min(dwell_times),
				'max_dwell_time': np.max(dwell_times)
			})
		
		if flight_times:
			features.update({
				'avg_flight_time': np.mean(flight_times),
				'std_flight_time': np.std(flight_times),
				'min_flight_time': np.min(flight_times),
				'max_flight_time': np.max(flight_times)
			})
		
		# Rhythm analysis
		if len(keystroke_data) >= 5:
			intervals = [keystroke_data[i+1]['timestamp'] - keystroke_data[i]['timestamp'] 
						for i in range(len(keystroke_data)-1)]
			features['typing_rhythm_variance'] = np.var(intervals) if intervals else 0.0
		
		return features
	
	@staticmethod
	def extract_mouse_features(mouse_data: List[Dict[str, Any]]) -> Dict[str, float]:
		"""Extract features from mouse movement patterns"""
		if not mouse_data:
			return {}
		
		# Calculate movement vectors
		movements = []
		velocities = []
		accelerations = []
		
		for i in range(1, len(mouse_data)):
			prev = mouse_data[i-1]
			curr = mouse_data[i]
			
			dx = curr['x'] - prev['x']
			dy = curr['y'] - prev['y']
			dt = curr['timestamp'] - prev['timestamp']
			
			if dt > 0:
				distance = np.sqrt(dx**2 + dy**2)
				velocity = distance / dt
				movements.append(distance)
				velocities.append(velocity)
		
		# Calculate accelerations
		for i in range(1, len(velocities)):
			dt = mouse_data[i+1]['timestamp'] - mouse_data[i]['timestamp']
			if dt > 0:
				accel = (velocities[i] - velocities[i-1]) / dt
				accelerations.append(accel)
		
		features = {}
		
		if movements:
			features.update({
				'avg_movement_distance': np.mean(movements),
				'std_movement_distance': np.std(movements),
				'total_movement': np.sum(movements)
			})
		
		if velocities:
			features.update({
				'avg_velocity': np.mean(velocities),
				'std_velocity': np.std(velocities),
				'max_velocity': np.max(velocities)
			})
		
		if accelerations:
			features.update({
				'avg_acceleration': np.mean(accelerations),
				'std_acceleration': np.std(accelerations)
			})
		
		# Click patterns
		clicks = [event for event in mouse_data if event.get('event_type') == 'click']
		if len(clicks) >= 2:
			click_intervals = [clicks[i+1]['timestamp'] - clicks[i]['timestamp']
							  for i in range(len(clicks)-1)]
			features['avg_click_interval'] = np.mean(click_intervals)
			features['std_click_interval'] = np.std(click_intervals)
		
		return features
	
	@staticmethod
	def extract_navigation_features(navigation_data: List[Dict[str, Any]]) -> Dict[str, float]:
		"""Extract features from navigation patterns"""
		if not navigation_data:
			return {}
		
		features = {}
		
		# Page visit patterns
		page_visits = [event['page'] for event in navigation_data if event.get('page')]
		unique_pages = len(set(page_visits))
		total_visits = len(page_visits)
		
		features.update({
			'unique_pages_visited': unique_pages,
			'total_page_visits': total_visits,
			'page_revisit_ratio': (total_visits - unique_pages) / max(total_visits, 1)
		})
		
		# Time on page analysis
		page_times = [event.get('time_on_page', 0) for event in navigation_data if event.get('time_on_page')]
		if page_times:
			features.update({
				'avg_time_on_page': np.mean(page_times),
				'std_time_on_page': np.std(page_times),
				'median_time_on_page': np.median(page_times)
			})
		
		# Navigation sequence analysis
		if len(navigation_data) >= 3:
			# Calculate session depth
			features['session_depth'] = len(navigation_data)
			
			# Back/forward button usage
			back_actions = len([event for event in navigation_data if event.get('action') == 'back'])
			forward_actions = len([event for event in navigation_data if event.get('action') == 'forward'])
			features['back_forward_ratio'] = (back_actions + forward_actions) / len(navigation_data)
		
		return features

class BehavioralAuthenticator:
	"""Main behavioral authentication engine"""
	
	def __init__(self, anomaly_threshold: float = 0.3, learning_rate: float = 0.1):
		self.anomaly_threshold = anomaly_threshold
		self.learning_rate = learning_rate
		self._baselines: Dict[str, BehavioralBaseline] = {}
		self._recent_events: Dict[str, List[BehavioralEvent]] = {}
		self._feature_extractor = FeatureExtractor()
		
		# Risk assessment thresholds
		self.risk_thresholds = {
			BehavioralRiskLevel.VERY_LOW: 0.0,
			BehavioralRiskLevel.LOW: 0.1, 
			BehavioralRiskLevel.MODERATE: 0.3,
			BehavioralRiskLevel.HIGH: 0.6,
			BehavioralRiskLevel.VERY_HIGH: 0.8,
			BehavioralRiskLevel.CRITICAL: 0.95
		}
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[BehavioralAuth INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[BehavioralAuth WARNING] {message} {kwargs if kwargs else ''}")
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[BehavioralAuth ERROR] {message} {kwargs if kwargs else ''}")
	
	async def establish_baseline(self, user_id: str, initial_events: List[BehavioralEvent]) -> BehavioralBaseline:
		"""Establish initial behavioral baseline for new user"""
		assert user_id, "User ID is required"
		assert initial_events, "Initial events are required"
		
		self._log_info("Establishing behavioral baseline", user_id=user_id, event_count=len(initial_events))
		
		baseline = BehavioralBaseline(user_id=user_id)
		
		# Group events by type
		events_by_type: Dict[BehavioralEventType, List[BehavioralEvent]] = {}
		for event in initial_events:
			if event.event_type not in events_by_type:
				events_by_type[event.event_type] = []
			events_by_type[event.event_type].append(event)
		
		# Create pattern for each event type
		for event_type, events in events_by_type.items():
			pattern = await self._create_pattern_from_events(user_id, event_type, events)
			baseline.patterns[event_type] = pattern
		
		# Update baseline metadata
		baseline.current_sample_count = len(initial_events)
		baseline.maturity_level = min(len(initial_events) / baseline.learning_samples_required, 1.0)
		baseline.confidence_level = baseline.maturity_level * 0.8  # Conservative confidence
		
		# Store baseline
		self._baselines[user_id] = baseline
		
		self._log_info("Baseline established", user_id=user_id, maturity=baseline.maturity_level)
		return baseline
	
	async def _create_pattern_from_events(self, user_id: str, event_type: BehavioralEventType, 
										  events: List[BehavioralEvent]) -> BehavioralPattern:
		"""Create behavioral pattern from event list"""
		assert events, "Events list cannot be empty"
		
		# Extract features from all events
		all_features = []
		for event in events:
			if event.processed_features:
				all_features.append(event.processed_features)
			else:
				# Process raw data based on event type
				features = await self._extract_features_from_raw_data(event_type, event.raw_data)
				event.processed_features = features
				all_features.append(features)
		
		if not all_features:
			# Create empty pattern
			return BehavioralPattern(
				user_id=user_id,
				event_type=event_type,
				features={},
				sample_count=0
			)
		
		# Calculate statistical measures
		feature_names = set()
		for features in all_features:
			feature_names.update(features.keys())
		
		mean_values = {}
		std_values = {}
		confidence_interval = {}
		
		for feature_name in feature_names:
			values = [features.get(feature_name, 0.0) for features in all_features]
			mean_val = np.mean(values)
			std_val = np.std(values)
			
			mean_values[feature_name] = mean_val
			std_values[feature_name] = std_val
			
			# 95% confidence interval
			margin = 1.96 * std_val  # 95% CI for normal distribution
			confidence_interval[feature_name] = (mean_val - margin, mean_val + margin)
		
		# Calculate stability score based on variance
		stability_scores = []
		for feature_name in feature_names:
			std_val = std_values[feature_name]
			mean_val = mean_values[feature_name]
			if mean_val != 0:
				cv = std_val / abs(mean_val)  # Coefficient of variation
				stability = max(0.0, 1.0 - cv)  # Convert to stability score
				stability_scores.append(stability)
		
		stability_score = np.mean(stability_scores) if stability_scores else 0.0
		
		return BehavioralPattern(
			user_id=user_id,
			event_type=event_type,
			features=mean_values,
			mean_values=mean_values,
			std_values=std_values,
			confidence_interval=confidence_interval,
			sample_count=len(events),
			stability_score=stability_score
		)
	
	async def _extract_features_from_raw_data(self, event_type: BehavioralEventType, 
											  raw_data: Dict[str, Any]) -> Dict[str, float]:
		"""Extract features from raw event data based on event type"""
		if event_type == BehavioralEventType.KEYSTROKE:
			return self._feature_extractor.extract_keystroke_features(raw_data.get('keystrokes', []))
		elif event_type == BehavioralEventType.MOUSE_MOVEMENT:
			return self._feature_extractor.extract_mouse_features(raw_data.get('mouse_events', []))
		elif event_type == BehavioralEventType.NAVIGATION:
			return self._feature_extractor.extract_navigation_features(raw_data.get('navigation', []))
		else:
			# For other event types, return raw data as features if numeric
			features = {}
			for key, value in raw_data.items():
				if isinstance(value, (int, float)):
					features[key] = float(value)
			return features
	
	async def analyze_user_patterns(self, user_id: str, session_data: Dict[str, Any]) -> AuthScore:
		"""Analyze user behavioral patterns and generate authentication score"""
		assert user_id, "User ID is required"
		assert session_data, "Session data is required"
		
		self._log_info("Analyzing user patterns", user_id=user_id)
		
		# Get user baseline
		baseline = self._baselines.get(user_id)
		if not baseline:
			# No baseline exists - create one or return low confidence
			self._log_warning("No baseline found for user", user_id=user_id)
			return AuthScore(
				confidence=0.1,
				method="behavioral",
				risk_level=BehavioralRiskLevel.HIGH,
				factors={"no_baseline": 0.1}
			)
		
		if not baseline.is_mature():
			self._log_info("Baseline not mature enough", user_id=user_id, maturity=baseline.maturity_level)
			return AuthScore(
				confidence=baseline.maturity_level * 0.5,
				method="behavioral", 
				risk_level=BehavioralRiskLevel.MODERATE,
				factors={"immature_baseline": baseline.maturity_level}
			)
		
		# Extract current behavioral features
		current_events = await self._process_session_data(user_id, session_data)
		if not current_events:
			return AuthScore(
				confidence=0.2,
				method="behavioral",
				risk_level=BehavioralRiskLevel.HIGH,
				factors={"no_events": 0.2}
			)
		
		# Calculate confidence scores for each event type
		factor_scores = {}
		total_confidence = 0.0
		pattern_count = 0
		
		for event_type, events in current_events.items():
			if event_type in baseline.patterns:
				baseline_pattern = baseline.patterns[event_type]
				confidence = await self._compare_with_baseline(baseline_pattern, events)
				factor_scores[f"{event_type.value}_confidence"] = confidence
				total_confidence += confidence
				pattern_count += 1
		
		# Calculate overall confidence
		if pattern_count > 0:
			overall_confidence = total_confidence / pattern_count
		else:
			overall_confidence = 0.1
		
		# Determine risk level
		risk_level = self._calculate_risk_level(overall_confidence)
		
		# Check for anomalies that might trigger step-up auth
		if overall_confidence < self.anomaly_threshold:
			self._log_warning("Behavioral anomaly detected", user_id=user_id, confidence=overall_confidence)
		
		auth_score = AuthScore(
			confidence=overall_confidence,
			method="behavioral",
			risk_level=risk_level,
			factors=factor_scores
		)
		
		self._log_info("Pattern analysis complete", 
					   user_id=user_id, 
					   confidence=overall_confidence,
					   risk_level=risk_level.value)
		
		return auth_score
	
	async def _process_session_data(self, user_id: str, session_data: Dict[str, Any]) -> Dict[BehavioralEventType, List[BehavioralEvent]]:
		"""Process session data into behavioral events"""
		events_by_type: Dict[BehavioralEventType, List[BehavioralEvent]] = {}
		
		# Process different types of session data
		if 'keystrokes' in session_data:
			keystroke_event = BehavioralEvent(
				user_id=user_id,
				session_id=session_data.get('session_id', 'unknown'),
				event_type=BehavioralEventType.KEYSTROKE,
				raw_data={'keystrokes': session_data['keystrokes']},
				device_info=session_data.get('device_info', {}),
				page_context=session_data.get('page_context')
			)
			keystroke_event.processed_features = await self._extract_features_from_raw_data(
				BehavioralEventType.KEYSTROKE, keystroke_event.raw_data
			)
			events_by_type[BehavioralEventType.KEYSTROKE] = [keystroke_event]
		
		if 'mouse_events' in session_data:
			mouse_event = BehavioralEvent(
				user_id=user_id,
				session_id=session_data.get('session_id', 'unknown'),
				event_type=BehavioralEventType.MOUSE_MOVEMENT,
				raw_data={'mouse_events': session_data['mouse_events']},
				device_info=session_data.get('device_info', {}),
				page_context=session_data.get('page_context')
			)
			mouse_event.processed_features = await self._extract_features_from_raw_data(
				BehavioralEventType.MOUSE_MOVEMENT, mouse_event.raw_data
			)
			events_by_type[BehavioralEventType.MOUSE_MOVEMENT] = [mouse_event]
		
		if 'navigation' in session_data:
			nav_event = BehavioralEvent(
				user_id=user_id,
				session_id=session_data.get('session_id', 'unknown'),
				event_type=BehavioralEventType.NAVIGATION,
				raw_data={'navigation': session_data['navigation']},
				device_info=session_data.get('device_info', {}),
				page_context=session_data.get('page_context')
			)
			nav_event.processed_features = await self._extract_features_from_raw_data(
				BehavioralEventType.NAVIGATION, nav_event.raw_data
			)
			events_by_type[BehavioralEventType.NAVIGATION] = [nav_event]
		
		return events_by_type
	
	async def _compare_with_baseline(self, baseline_pattern: BehavioralPattern, 
									 current_events: List[BehavioralEvent]) -> float:
		"""Compare current events with baseline pattern"""
		if not current_events or not baseline_pattern.features:
			return 0.0
		
		# Extract features from current events
		current_features_list = [event.processed_features for event in current_events if event.processed_features]
		if not current_features_list:
			return 0.0
		
		# Calculate mean of current features
		all_feature_names = set()
		for features in current_features_list:
			all_feature_names.update(features.keys())
		
		current_mean_features = {}
		for feature_name in all_feature_names:
			values = [features.get(feature_name, 0.0) for features in current_features_list]
			current_mean_features[feature_name] = np.mean(values)
		
		# Calculate similarity to baseline
		similarities = []
		for feature_name in baseline_pattern.features.keys():
			if feature_name in current_mean_features:
				baseline_val = baseline_pattern.features[feature_name]
				current_val = current_mean_features[feature_name]
				baseline_std = baseline_pattern.std_values.get(feature_name, 1.0)
				
				# Calculate normalized distance
				if baseline_std > 0:
					distance = abs(current_val - baseline_val) / baseline_std
					# Convert distance to similarity (inverse exponential)
					similarity = np.exp(-distance)
				else:
					# If no variance in baseline, check for exact match
					similarity = 1.0 if abs(current_val - baseline_val) < 1e-6 else 0.0
				
				similarities.append(similarity)
		
		if not similarities:
			return 0.0
		
		# Return weighted average similarity
		confidence = np.mean(similarities)
		return min(1.0, max(0.0, confidence))  # Clamp to [0, 1]
	
	def _calculate_risk_level(self, confidence: float) -> BehavioralRiskLevel:
		"""Calculate risk level based on confidence score"""
		# Invert confidence to get risk score
		risk_score = 1.0 - confidence
		
		for risk_level, threshold in reversed(list(self.risk_thresholds.items())):
			if risk_score >= threshold:
				return risk_level
		
		return BehavioralRiskLevel.VERY_LOW
	
	async def continuous_monitoring(self, user_id: str, session_id: str, 
									real_time_data: Dict[str, Any]) -> Optional[AuthScore]:
		"""Continuous behavioral monitoring during active session"""
		assert user_id, "User ID is required"
		assert session_id, "Session ID is required"
		
		# Store recent events for this user
		if user_id not in self._recent_events:
			self._recent_events[user_id] = []
		
		# Process real-time data into events
		current_events = await self._process_session_data(user_id, real_time_data)
		
		# Add to recent events (keep last 50 events)
		for event_type, events in current_events.items():
			self._recent_events[user_id].extend(events)
		
		self._recent_events[user_id] = self._recent_events[user_id][-50:]  # Keep last 50
		
		# Only analyze if we have sufficient recent data
		if len(self._recent_events[user_id]) < 5:
			return None
		
		# Create session data from recent events
		session_data = {'session_id': session_id}
		
		# Group recent events by type
		for event in self._recent_events[user_id]:
			if event.event_type not in session_data:
				session_data[event.event_type.value] = []
			session_data[event.event_type.value].append(event.raw_data)
		
		# Analyze patterns
		auth_score = await self.analyze_user_patterns(user_id, session_data)
		
		# Update events with analysis results
		for event in self._recent_events[user_id]:
			event.confidence_score = auth_score.confidence
			event.anomaly_score = 1.0 - auth_score.confidence
		
		return auth_score
	
	async def update_baseline(self, user_id: str, new_events: List[BehavioralEvent], 
							  adaptation_rate: float = 0.1) -> bool:
		"""Update user's behavioral baseline with new authentic events"""
		assert user_id, "User ID is required"
		assert 0.0 <= adaptation_rate <= 1.0, "Adaptation rate must be between 0.0 and 1.0"
		
		baseline = self._baselines.get(user_id)
		if not baseline:
			self._log_warning("No baseline to update", user_id=user_id)
			return False
		
		if not baseline.adaptive_learning_enabled:
			return False
		
		self._log_info("Updating baseline", user_id=user_id, new_event_count=len(new_events))
		
		# Group new events by type
		events_by_type: Dict[BehavioralEventType, List[BehavioralEvent]] = {}
		for event in new_events:
			if event.event_type not in events_by_type:
				events_by_type[event.event_type] = []
			events_by_type[event.event_type].append(event)
		
		# Update patterns for each event type
		updated = False
		for event_type, events in events_by_type.items():
			if event_type in baseline.patterns:
				# Update existing pattern
				success = await self._update_pattern(baseline.patterns[event_type], events, adaptation_rate)
				if success:
					updated = True
			else:
				# Create new pattern for this event type
				new_pattern = await self._create_pattern_from_events(user_id, event_type, events)
				baseline.patterns[event_type] = new_pattern
				updated = True
		
		if updated:
			baseline.current_sample_count += len(new_events)
			baseline.update_timestamp()
			
			# Recalculate maturity and confidence
			baseline.maturity_level = min(
				baseline.current_sample_count / baseline.learning_samples_required, 1.0
			)
			baseline.confidence_level = baseline.maturity_level * 0.9
			
			self._log_info("Baseline updated successfully", user_id=user_id)
		
		return updated
	
	async def _update_pattern(self, pattern: BehavioralPattern, new_events: List[BehavioralEvent],
							  adaptation_rate: float) -> bool:
		"""Update existing behavioral pattern with new events"""
		if not new_events:
			return False
		
		# Extract features from new events
		new_features_list = []
		for event in new_events:
			if event.processed_features:
				new_features_list.append(event.processed_features)
			else:
				features = await self._extract_features_from_raw_data(
					pattern.event_type, event.raw_data
				)
				new_features_list.append(features)
		
		if not new_features_list:
			return False
		
		# Calculate new feature statistics
		all_feature_names = set(pattern.features.keys())
		for features in new_features_list:
			all_feature_names.update(features.keys())
		
		# Update each feature with exponential moving average
		for feature_name in all_feature_names:
			new_values = [features.get(feature_name, 0.0) for features in new_features_list]
			new_mean = np.mean(new_values)
			new_std = np.std(new_values) if len(new_values) > 1 else 0.0
			
			if feature_name in pattern.mean_values:
				# Update existing feature
				old_mean = pattern.mean_values[feature_name]
				old_std = pattern.std_values[feature_name]
				
				# Exponential moving average
				updated_mean = (1 - adaptation_rate) * old_mean + adaptation_rate * new_mean
				updated_std = (1 - adaptation_rate) * old_std + adaptation_rate * new_std
				
				pattern.mean_values[feature_name] = updated_mean
				pattern.std_values[feature_name] = updated_std
				pattern.features[feature_name] = updated_mean
				
				# Update confidence interval
				margin = 1.96 * updated_std
				pattern.confidence_interval[feature_name] = (
					updated_mean - margin, updated_mean + margin
				)
			else:
				# New feature
				pattern.mean_values[feature_name] = new_mean
				pattern.std_values[feature_name] = new_std
				pattern.features[feature_name] = new_mean
				
				margin = 1.96 * new_std
				pattern.confidence_interval[feature_name] = (new_mean - margin, new_mean + margin)
		
		# Update pattern metadata
		pattern.sample_count += len(new_events)
		pattern.last_updated = datetime.utcnow()
		
		# Recalculate stability score
		stability_scores = []
		for feature_name, mean_val in pattern.mean_values.items():
			std_val = pattern.std_values[feature_name]
			if mean_val != 0:
				cv = std_val / abs(mean_val)
				stability = max(0.0, 1.0 - cv)
				stability_scores.append(stability)
		
		pattern.stability_score = np.mean(stability_scores) if stability_scores else 0.0
		
		return True
	
	async def detect_anomalies(self, user_id: str, current_score: AuthScore,
							   threshold: Optional[float] = None) -> Dict[str, Any]:
		"""Detect behavioral anomalies requiring step-up authentication"""
		if threshold is None:
			threshold = self.anomaly_threshold
		
		anomalies = {
			"anomaly_detected": current_score.confidence < threshold,
			"confidence_score": current_score.confidence,
			"risk_level": current_score.risk_level,
			"threshold_used": threshold,
			"factors": current_score.factors,
			"recommendations": []
		}
		
		# Generate specific recommendations based on risk level
		if current_score.risk_level in [BehavioralRiskLevel.HIGH, BehavioralRiskLevel.VERY_HIGH]:
			anomalies["recommendations"].extend([
				"require_mfa",
				"additional_verification"
			])
		
		if current_score.risk_level == BehavioralRiskLevel.CRITICAL:
			anomalies["recommendations"].extend([
				"require_mfa",
				"biometric_verification",
				"security_questions",
				"admin_review"
			])
		
		# Check specific factor anomalies
		for factor_name, factor_score in current_score.factors.items():
			if factor_score < 0.2:  # Very low confidence in specific factor
				anomalies["recommendations"].append(f"verify_{factor_name}")
		
		return anomalies
	
	async def trigger_step_up_auth(self, user_id: str, anomaly_reason: str) -> Dict[str, Any]:
		"""Trigger step-up authentication due to behavioral anomaly"""
		self._log_warning("Triggering step-up authentication", 
						  user_id=user_id, reason=anomaly_reason)
		
		# Generate step-up authentication requirements
		step_up_auth = {
			"user_id": user_id,
			"reason": anomaly_reason,
			"timestamp": datetime.utcnow().isoformat(),
			"required_methods": ["mfa"],  # Default requirement
			"challenge_id": uuid7str(),
			"expires_in": 300  # 5 minutes
		}
		
		# Determine additional requirements based on anomaly reason
		if "critical" in anomaly_reason.lower():
			step_up_auth["required_methods"].extend(["biometric", "security_questions"])
		elif "high" in anomaly_reason.lower():
			step_up_auth["required_methods"].append("biometric")
		
		# TODO: Integrate with APG notification system
		# await self._send_security_alert(user_id, step_up_auth)
		
		return step_up_auth
	
	def get_user_baseline(self, user_id: str) -> Optional[BehavioralBaseline]:
		"""Get user's behavioral baseline"""
		return self._baselines.get(user_id)
	
	def get_baseline_maturity(self, user_id: str) -> float:
		"""Get baseline maturity level for user"""
		baseline = self._baselines.get(user_id)
		return baseline.maturity_level if baseline else 0.0
	
	async def export_baseline(self, user_id: str) -> Optional[Dict[str, Any]]:
		"""Export user baseline for backup or transfer"""
		baseline = self._baselines.get(user_id)
		if not baseline:
			return None
		
		return baseline.model_dump()
	
	async def import_baseline(self, baseline_data: Dict[str, Any]) -> bool:
		"""Import user baseline from backup"""
		try:
			baseline = BehavioralBaseline.model_validate(baseline_data)
			self._baselines[baseline.user_id] = baseline
			self._log_info("Baseline imported successfully", user_id=baseline.user_id)
			return True
		except Exception as e:
			self._log_error("Failed to import baseline", error=str(e))
			return False
	
	def clear_user_data(self, user_id: str):
		"""Clear all behavioral data for user (GDPR compliance)"""
		if user_id in self._baselines:
			del self._baselines[user_id]
		if user_id in self._recent_events:
			del self._recent_events[user_id]
		self._log_info("User behavioral data cleared", user_id=user_id)