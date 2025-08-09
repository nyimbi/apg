#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Predictive Engine
ML-driven predictive content delivery and intelligent prefetching

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from collections import defaultdict, deque

from .models import CacheEntry, CacheAccessPattern, CacheTier


class PredictionType(str, Enum):
	"""Types of cache predictions"""
	ACCESS_TIME = "access_time"
	ACCESS_PROBABILITY = "access_probability" 
	CONTENT_RELATIONSHIP = "content_relationship"
	USER_BEHAVIOR = "user_behavior"
	TEMPORAL_PATTERN = "temporal_pattern"
	GEOGRAPHIC_PATTERN = "geographic_pattern"


@dataclass 
class PredictionResult:
	"""Result of a cache prediction"""
	prediction_type: PredictionType
	target_key: str
	confidence_score: float
	predicted_value: Any
	prediction_time: datetime = field(default_factory=datetime.utcnow)
	features_used: Dict[str, Any] = field(default_factory=dict)
	model_version: str = "1.0.0"


@dataclass
class ContentRelationship:
	"""Relationship between cached content items"""
	source_key: str
	related_key: str
	relationship_type: str  # "sequential", "categorical", "user_session", "temporal"
	strength: float  # 0.0 to 1.0
	frequency: int
	last_seen: datetime


@dataclass
class UserBehaviorPattern:
	"""User behavior pattern for predictive caching"""
	user_id: str
	session_patterns: List[List[str]]  # Sequences of keys accessed
	temporal_preferences: Dict[int, float]  # Hour -> preference score
	content_preferences: Dict[str, float]  # Category -> preference score
	geographic_context: Optional[str] = None
	device_context: Optional[str] = None


class PredictiveEngine:
	"""
	Revolutionary predictive content delivery engine
	Revolutionary Differentiator #2: Predictive Content Delivery
	"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.logger = logging.getLogger('cach.predictive_engine')
		
		# Prediction models
		self.access_predictor = AccessTimePredictor()
		self.relationship_analyzer = ContentRelationshipAnalyzer()
		self.behavior_analyzer = UserBehaviorAnalyzer()
		self.temporal_predictor = TemporalPatternPredictor()
		
		# Prediction state
		self.predictions_history: deque = deque(maxlen=10000)
		self.content_relationships: Dict[str, List[ContentRelationship]] = defaultdict(list)
		self.user_behaviors: Dict[str, UserBehaviorPattern] = {}
		self.prefetch_queue: deque = deque(maxlen=1000)
		
		# Configuration
		self.prediction_window = timedelta(hours=24)
		self.confidence_threshold = 0.7
		self.relationship_strength_threshold = 0.5
		self.max_prefetch_candidates = 100
		
		# Performance tracking
		self.prediction_accuracy_cache = deque(maxlen=1000)
		self.prefetch_hit_rate_cache = deque(maxlen=1000)
	
	async def initialize(self) -> None:
		"""Initialize predictive engine"""
		self.logger.info("Initializing predictive content delivery engine...")
		
		# Initialize prediction models
		await self.access_predictor.initialize()
		await self.relationship_analyzer.initialize()
		await self.behavior_analyzer.initialize()
		await self.temporal_predictor.initialize()
		
		self.logger.info("Predictive engine initialized")
	
	async def predict_access_probability(self, key: str, context: Dict[str, Any] = None) -> PredictionResult:
		"""
		Predict the probability that a key will be accessed soon
		Core component of Revolutionary Differentiator #2
		"""
		
		features = await self._extract_prediction_features(key, context or {})
		
		# Get prediction from access predictor
		probability = await self.access_predictor.predict_access_probability(features)
		
		# Adjust based on content relationships
		relationship_boost = await self._calculate_relationship_boost(key, context)
		adjusted_probability = min(probability + relationship_boost, 1.0)
		
		# Calculate confidence based on feature completeness and model accuracy
		confidence = await self._calculate_prediction_confidence(features, PredictionType.ACCESS_PROBABILITY)
		
		result = PredictionResult(
			prediction_type=PredictionType.ACCESS_PROBABILITY,
			target_key=key,
			confidence_score=confidence,
			predicted_value=adjusted_probability,
			features_used=features
		)
		
		# Store prediction for accuracy tracking
		self.predictions_history.append(result)
		
		return result
	
	async def predict_next_access_time(self, key: str, context: Dict[str, Any] = None) -> PredictionResult:
		"""Predict when a key will next be accessed"""
		
		features = await self._extract_prediction_features(key, context or {})
		
		# Get temporal prediction
		next_access_time = await self.temporal_predictor.predict_next_access(features)
		
		# Adjust based on user behavior patterns
		if context and context.get('user_id'):
			user_adjustment = await self._get_user_temporal_adjustment(
				context['user_id'], next_access_time
			)
			next_access_time += user_adjustment
		
		confidence = await self._calculate_prediction_confidence(features, PredictionType.ACCESS_TIME)
		
		return PredictionResult(
			prediction_type=PredictionType.ACCESS_TIME,
			target_key=key,
			confidence_score=confidence,
			predicted_value=next_access_time,
			features_used=features
		)
	
	async def analyze_content_relationships(self, entries: Dict[str, CacheEntry]) -> List[ContentRelationship]:
		"""
		Analyze relationships between cached content
		Enables intelligent prefetching of related content
		"""
		
		new_relationships = []
		
		# Analyze sequential access patterns
		sequential_relationships = await self.relationship_analyzer.find_sequential_patterns(entries)
		new_relationships.extend(sequential_relationships)
		
		# Analyze categorical relationships
		categorical_relationships = await self.relationship_analyzer.find_categorical_patterns(entries)
		new_relationships.extend(categorical_relationships)
		
		# Analyze user session relationships
		session_relationships = await self.relationship_analyzer.find_session_patterns(entries)
		new_relationships.extend(session_relationships)
		
		# Update relationship graph
		for relationship in new_relationships:
			self.content_relationships[relationship.source_key].append(relationship)
		
		# Clean up old relationships
		await self._cleanup_old_relationships()
		
		self.logger.debug(f"Analyzed {len(new_relationships)} content relationships")
		return new_relationships
	
	async def generate_prefetch_candidates(self, recently_accessed: List[str],
										   context: Dict[str, Any] = None) -> List[Tuple[str, float]]:
		"""
		Generate intelligent prefetch candidates based on recent access
		Revolutionary Differentiator #2: Predictive Content Delivery
		"""
		
		candidates = []
		context = context or {}
		
		# Find related content based on relationships
		for key in recently_accessed:
			related_keys = await self._find_related_keys(key, context)
			candidates.extend(related_keys)
		
		# Add user behavior-based predictions
		if context.get('user_id'):
			behavior_predictions = await self._predict_user_behavior_keys(
				context['user_id'], recently_accessed
			)
			candidates.extend(behavior_predictions)
		
		# Add temporal pattern predictions
		temporal_predictions = await self._predict_temporal_keys(recently_accessed, context)
		candidates.extend(temporal_predictions)
		
		# Remove duplicates and sort by prediction score
		unique_candidates = {}
		for key, score in candidates:
			if key not in recently_accessed:  # Don't prefetch already cached items
				if key in unique_candidates:
					unique_candidates[key] = max(unique_candidates[key], score)
				else:
					unique_candidates[key] = score
		
		# Sort by prediction score
		sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
		
		# Filter by confidence threshold and limit
		filtered_candidates = [
			(key, score) for key, score in sorted_candidates 
			if score >= self.confidence_threshold
		][:self.max_prefetch_candidates]
		
		self.logger.debug(f"Generated {len(filtered_candidates)} prefetch candidates")
		return filtered_candidates
	
	async def update_user_behavior(self, user_id: str, accessed_keys: List[str],
								   context: Dict[str, Any] = None) -> None:
		"""Update user behavior patterns for better predictions"""
		
		if user_id not in self.user_behaviors:
			self.user_behaviors[user_id] = UserBehaviorPattern(
				user_id=user_id,
				session_patterns=[],
				temporal_preferences={},
				content_preferences={}
			)
		
		behavior = self.user_behaviors[user_id]
		
		# Update session patterns
		if accessed_keys:
			behavior.session_patterns.append(accessed_keys.copy())
			
			# Keep only recent session patterns
			if len(behavior.session_patterns) > 100:
				behavior.session_patterns = behavior.session_patterns[-100:]
		
		# Update temporal preferences
		current_hour = datetime.utcnow().hour
		if current_hour not in behavior.temporal_preferences:
			behavior.temporal_preferences[current_hour] = 0.0
		
		behavior.temporal_preferences[current_hour] += len(accessed_keys)
		
		# Update content preferences (simplified category analysis)
		for key in accessed_keys:
			category = self._extract_content_category(key)
			if category not in behavior.content_preferences:
				behavior.content_preferences[category] = 0.0
			behavior.content_preferences[category] += 1.0
		
		# Update context information
		if context:
			if 'location' in context:
				behavior.geographic_context = context['location']
			if 'device' in context:
				behavior.device_context = context['device']
	
	async def validate_prediction_accuracy(self, actual_accesses: List[str],
										   timeframe: timedelta) -> Dict[str, float]:
		"""Validate accuracy of recent predictions"""
		
		cutoff_time = datetime.utcnow() - timeframe
		recent_predictions = [
			p for p in self.predictions_history 
			if p.prediction_time >= cutoff_time
		]
		
		if not recent_predictions:
			return {'accuracy': 0.0, 'predictions_count': 0}
		
		correct_predictions = 0
		total_predictions = len(recent_predictions)
		
		for prediction in recent_predictions:
			if prediction.prediction_type == PredictionType.ACCESS_PROBABILITY:
				# Consider prediction correct if key was actually accessed
				# and prediction probability was > threshold
				if (prediction.target_key in actual_accesses and 
					prediction.predicted_value >= self.confidence_threshold):
					correct_predictions += 1
				elif (prediction.target_key not in actual_accesses and 
					  prediction.predicted_value < self.confidence_threshold):
					correct_predictions += 1
		
		accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
		
		# Cache accuracy for model improvement
		self.prediction_accuracy_cache.append(accuracy)
		
		return {
			'accuracy': accuracy,
			'predictions_count': total_predictions,
			'correct_predictions': correct_predictions
		}
	
	# Private helper methods
	
	async def _extract_prediction_features(self, key: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Extract features for prediction models"""
		
		features = {
			'key': key,
			'key_length': len(key),
			'has_numeric': any(c.isdigit() for c in key),
			'has_separators': '.' in key or '/' in key or '_' in key,
			'timestamp': datetime.utcnow().timestamp(),
			'hour_of_day': datetime.utcnow().hour,
			'day_of_week': datetime.utcnow().weekday()
		}
		
		# Add context features
		features.update(context)
		
		# Add relationship features
		relationship_count = len(self.content_relationships.get(key, []))
		features['relationship_count'] = relationship_count
		
		# Add user behavior features if available
		if context.get('user_id'):
			user_behavior = self.user_behaviors.get(context['user_id'])
			if user_behavior:
				current_hour = datetime.utcnow().hour
				features['user_temporal_preference'] = user_behavior.temporal_preferences.get(current_hour, 0.0)
				
				category = self._extract_content_category(key)
				features['user_content_preference'] = user_behavior.content_preferences.get(category, 0.0)
		
		return features
	
	async def _calculate_relationship_boost(self, key: str, context: Dict[str, Any]) -> float:
		"""Calculate relationship-based probability boost"""
		
		boost = 0.0
		
		# Check if related keys were recently accessed
		recently_accessed = context.get('recently_accessed', [])
		
		for accessed_key in recently_accessed:
			relationships = self.content_relationships.get(accessed_key, [])
			for relationship in relationships:
				if relationship.related_key == key:
					# Boost based on relationship strength and recency
					time_decay = self._calculate_time_decay(relationship.last_seen)
					boost += relationship.strength * time_decay * 0.2
		
		return min(boost, 0.3)  # Cap boost at 30%
	
	async def _calculate_prediction_confidence(self, features: Dict[str, Any],
											   prediction_type: PredictionType) -> float:
		"""Calculate confidence score for prediction"""
		
		base_confidence = 0.7
		
		# Adjust based on feature completeness
		feature_completeness = len(features) / 10.0  # Normalize to 0-1
		
		# Adjust based on historical accuracy
		if self.prediction_accuracy_cache:
			historical_accuracy = sum(self.prediction_accuracy_cache) / len(self.prediction_accuracy_cache)
		else:
			historical_accuracy = 0.7
		
		# Adjust based on relationship data availability
		relationship_factor = 1.0
		if features.get('relationship_count', 0) > 0:
			relationship_factor = 1.1
		
		# Combine factors
		confidence = base_confidence * feature_completeness * historical_accuracy * relationship_factor
		
		return min(confidence, 1.0)
	
	async def _find_related_keys(self, key: str, context: Dict[str, Any]) -> List[Tuple[str, float]]:
		"""Find keys related to the given key"""
		
		related = []
		relationships = self.content_relationships.get(key, [])
		
		for relationship in relationships:
			if relationship.strength >= self.relationship_strength_threshold:
				# Calculate prediction score based on relationship strength and recency
				time_decay = self._calculate_time_decay(relationship.last_seen)
				score = relationship.strength * time_decay
				related.append((relationship.related_key, score))
		
		return related
	
	async def _predict_user_behavior_keys(self, user_id: str, recently_accessed: List[str]) -> List[Tuple[str, float]]:
		"""Predict keys based on user behavior patterns"""
		
		if user_id not in self.user_behaviors:
			return []
		
		behavior = self.user_behaviors[user_id]
		predictions = []
		
		# Find patterns in user's session history
		for session in behavior.session_patterns[-10:]:  # Last 10 sessions
			for i, key in enumerate(session[:-1]):  # Don't include last key
				if key in recently_accessed:
					# Predict next key in sequence
					next_key = session[i + 1]
					score = 0.8 * (1.0 - i / len(session))  # Earlier in session = higher score
					predictions.append((next_key, score))
		
		return predictions
	
	async def _predict_temporal_keys(self, recently_accessed: List[str], 
									 context: Dict[str, Any]) -> List[Tuple[str, float]]:
		"""Predict keys based on temporal patterns"""
		
		predictions = []
		current_hour = datetime.utcnow().hour
		
		# Simple temporal pattern: predict keys typically accessed at this hour
		# Would be more sophisticated with actual temporal analysis
		for key in recently_accessed:
			# Simulate temporal correlation (would use actual historical data)
			if current_hour >= 9 and current_hour <= 17:  # Business hours
				score = 0.6
			else:
				score = 0.3
			
			# Predict related keys might be accessed
			predictions.append((key + "_related", score))
		
		return predictions
	
	async def _get_user_temporal_adjustment(self, user_id: str, base_time: datetime) -> timedelta:
		"""Get user-specific temporal adjustment"""
		
		if user_id not in self.user_behaviors:
			return timedelta()
		
		behavior = self.user_behaviors[user_id]
		target_hour = base_time.hour
		
		# Find user's preference for this hour
		preference = behavior.temporal_preferences.get(target_hour, 0.0)
		
		# Adjust timing based on preference (simplified)
		if preference > 10:  # High activity hour
			return timedelta(minutes=-15)  # Predict earlier
		elif preference < 2:  # Low activity hour  
			return timedelta(minutes=30)   # Predict later
		
		return timedelta()
	
	def _extract_content_category(self, key: str) -> str:
		"""Extract content category from key (simplified)"""
		
		# Simple category extraction based on key patterns
		if 'user' in key.lower():
			return 'user_data'
		elif 'product' in key.lower():
			return 'product_data'
		elif 'image' in key.lower() or 'img' in key.lower():
			return 'media'
		elif 'api' in key.lower():
			return 'api_data'
		else:
			return 'general'
	
	def _calculate_time_decay(self, timestamp: datetime) -> float:
		"""Calculate time decay factor for relationships"""
		
		age = datetime.utcnow() - timestamp
		age_hours = age.total_seconds() / 3600
		
		# Exponential decay: recent relationships are much stronger
		return math.exp(-age_hours / 24.0)  # Half-life of 24 hours
	
	async def _cleanup_old_relationships(self) -> None:
		"""Clean up old content relationships"""
		
		cutoff_time = datetime.utcnow() - timedelta(days=7)
		
		for key in list(self.content_relationships.keys()):
			relationships = self.content_relationships[key]
			active_relationships = [
				rel for rel in relationships 
				if rel.last_seen >= cutoff_time
			]
			
			if active_relationships:
				self.content_relationships[key] = active_relationships
			else:
				del self.content_relationships[key]


# Simplified ML model classes

class AccessTimePredictor:
	"""Predict when content will be accessed"""
	
	async def initialize(self) -> None:
		"""Initialize access time prediction model"""
		pass
	
	async def predict_access_probability(self, features: Dict[str, Any]) -> float:
		"""Predict probability of access in near future"""
		
		# Simplified prediction logic (would use actual ML model)
		base_prob = 0.1
		
		# Factor in relationship count
		if features.get('relationship_count', 0) > 0:
			base_prob *= 1.5
		
		# Factor in user preferences
		if features.get('user_content_preference', 0) > 5:
			base_prob *= 2.0
		
		# Factor in temporal patterns
		hour = features.get('hour_of_day', 12)
		if 9 <= hour <= 17:  # Business hours
			base_prob *= 1.3
		
		return min(base_prob, 0.95)


class ContentRelationshipAnalyzer:
	"""Analyze relationships between content items"""
	
	async def initialize(self) -> None:
		"""Initialize relationship analyzer"""
		pass
	
	async def find_sequential_patterns(self, entries: Dict[str, CacheEntry]) -> List[ContentRelationship]:
		"""Find sequential access patterns"""
		
		relationships = []
		
		# Simple sequential pattern detection (would be more sophisticated)
		sorted_entries = sorted(entries.items(), key=lambda x: x[1].last_accessed or datetime.min)
		
		for i in range(len(sorted_entries) - 1):
			current_key, current_entry = sorted_entries[i]
			next_key, next_entry = sorted_entries[i + 1]
			
			# If accessed within 5 minutes of each other, consider sequential
			if (next_entry.last_accessed and current_entry.last_accessed and
				(next_entry.last_accessed - current_entry.last_accessed).total_seconds() < 300):
				
				relationships.append(ContentRelationship(
					source_key=current_key,
					related_key=next_key,
					relationship_type="sequential",
					strength=0.7,
					frequency=1,
					last_seen=datetime.utcnow()
				))
		
		return relationships
	
	async def find_categorical_patterns(self, entries: Dict[str, CacheEntry]) -> List[ContentRelationship]:
		"""Find categorical relationships"""
		
		relationships = []
		
		# Group by categories
		categories = defaultdict(list)
		for key, entry in entries.items():
			category = self._get_category(key)
			categories[category].append((key, entry))
		
		# Create relationships within categories
		for category, items in categories.items():
			if len(items) > 1:
				for i, (key1, entry1) in enumerate(items):
					for key2, entry2 in items[i+1:]:
						relationships.append(ContentRelationship(
							source_key=key1,
							related_key=key2,
							relationship_type="categorical",
							strength=0.5,
							frequency=1,
							last_seen=datetime.utcnow()
						))
		
		return relationships
	
	async def find_session_patterns(self, entries: Dict[str, CacheEntry]) -> List[ContentRelationship]:
		"""Find user session-based relationships"""
		
		# Simplified session pattern detection
		# Would analyze actual user session data in production
		return []
	
	def _get_category(self, key: str) -> str:
		"""Get category for a key"""
		if 'user' in key:
			return 'user'
		elif 'product' in key:
			return 'product'
		elif 'api' in key:
			return 'api'
		else:
			return 'other'


class UserBehaviorAnalyzer:
	"""Analyze user behavior patterns"""
	
	async def initialize(self) -> None:
		"""Initialize user behavior analyzer"""
		pass


class TemporalPatternPredictor:
	"""Predict temporal access patterns"""
	
	async def initialize(self) -> None:
		"""Initialize temporal pattern predictor"""
		pass
	
	async def predict_next_access(self, features: Dict[str, Any]) -> datetime:
		"""Predict next access time"""
		
		# Simple temporal prediction (would use actual ML model)
		current_hour = features.get('hour_of_day', 12)
		
		if 9 <= current_hour <= 17:  # Business hours
			next_access = datetime.utcnow() + timedelta(minutes=30)
		else:
			next_access = datetime.utcnow() + timedelta(hours=2)
		
		return next_access


# Export main components
__all__ = [
	'PredictiveEngine',
	'PredictionType',
	'PredictionResult',
	'ContentRelationship',
	'UserBehaviorPattern',
	'AccessTimePredictor',
	'ContentRelationshipAnalyzer',
	'UserBehaviorAnalyzer',
	'TemporalPatternPredictor'
]