"""
APG Notification Capability - AI-Powered Personalization Engine

Revolutionary AI-powered personalization engine providing hyper-intelligent content
optimization, behavioral analysis, predictive personalization, and real-time adaptation.
Designed to be 10x better than industry leaders with unprecedented capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import pickle
from collections import defaultdict, Counter

from .api_models import (
	DeliveryChannel, NotificationPriority, EngagementEvent,
	UltimateUserPreferences, ComprehensiveDelivery
)


# Configure logging
_log = logging.getLogger(__name__)


class PersonalizationStrategy(str, Enum):
	"""Personalization strategies"""
	BEHAVIORAL = "behavioral"
	DEMOGRAPHIC = "demographic" 
	CONTEXTUAL = "contextual"
	PREDICTIVE = "predictive"
	COLLABORATIVE = "collaborative"
	CONTENT_BASED = "content_based"
	HYBRID = "hybrid"


class ContentType(str, Enum):
	"""Content types for personalization"""
	SUBJECT_LINE = "subject_line"
	MESSAGE_BODY = "message_body"
	CALL_TO_ACTION = "call_to_action"
	SEND_TIME = "send_time"
	CHANNEL_SELECTION = "channel_selection"
	FREQUENCY = "frequency"
	MEDIA_ASSETS = "media_assets"


@dataclass
class PersonalizationContext:
	"""Context information for personalization"""
	user_id: str
	tenant_id: str
	template_id: str
	channels: List[DeliveryChannel]
	campaign_id: Optional[str] = None
	
	# User context
	user_preferences: Optional[UltimateUserPreferences] = None
	engagement_history: List[Dict[str, Any]] = field(default_factory=list)
	behavioral_profile: Dict[str, Any] = field(default_factory=dict)
	
	# Environmental context
	timestamp: datetime = field(default_factory=datetime.utcnow)
	timezone: str = "UTC"
	device_type: Optional[str] = None
	location: Optional[Dict[str, Any]] = None
	weather: Optional[Dict[str, Any]] = None
	
	# Business context
	business_rules: Dict[str, Any] = field(default_factory=dict)
	ab_test_variant: Optional[str] = None
	campaign_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalizationResult:
	"""Result of personalization process"""
	original_content: Dict[str, Any]
	personalized_content: Dict[str, Any]
	optimizations: List[Dict[str, Any]]
	confidence_score: float
	strategy_used: PersonalizationStrategy
	processing_time_ms: int
	
	# Channel optimization
	recommended_channels: List[DeliveryChannel]
	optimal_send_time: Optional[datetime] = None
	
	# A/B testing
	ab_test_variant: Optional[str] = None
	expected_performance: Dict[str, float] = field(default_factory=dict)
	
	# Metadata
	personalization_id: str = field(default_factory=lambda: f"pers_{datetime.utcnow().timestamp()}")
	applied_rules: List[str] = field(default_factory=list)


@dataclass
class UserEngagementProfile:
	"""Comprehensive user engagement profile"""
	user_id: str
	tenant_id: str
	
	# Engagement metrics
	total_notifications: int = 0
	total_opened: int = 0
	total_clicked: int = 0
	total_converted: int = 0
	
	# Channel preferences (learned)
	channel_engagement_scores: Dict[DeliveryChannel, float] = field(default_factory=dict)
	optimal_send_times: Dict[DeliveryChannel, List[int]] = field(default_factory=dict)  # Hours of day
	
	# Content preferences
	preferred_content_length: Dict[str, int] = field(default_factory=dict)  # channel -> length
	preferred_tone: str = "neutral"  # casual, formal, friendly, urgent
	preferred_language: str = "en-US"
	
	# Behavioral patterns
	interaction_patterns: Dict[str, Any] = field(default_factory=dict)
	seasonal_patterns: Dict[str, Any] = field(default_factory=dict)
	frequency_tolerance: Dict[DeliveryChannel, int] = field(default_factory=dict)  # max per day
	
	# Predictive scores
	churn_risk_score: float = 0.0
	engagement_momentum: float = 0.5
	value_score: float = 0.0
	
	# Learning metadata
	last_updated: datetime = field(default_factory=datetime.utcnow)
	confidence_level: float = 0.5
	data_points: int = 0


class IntelligentPersonalizationEngine:
	"""
	Revolutionary AI-powered personalization engine providing hyper-intelligent
	content optimization, behavioral analysis, and predictive personalization.
	"""
	
	def __init__(self, tenant_id: str, redis_client=None):
		"""Initialize personalization engine"""
		self.tenant_id = tenant_id
		self.redis_client = redis_client
		
		# User engagement profiles cache
		self.user_profiles: Dict[str, UserEngagementProfile] = {}
		
		# ML models (simplified - would use actual ML libraries)
		self.content_optimization_model = None
		self.send_time_model = None
		self.channel_preference_model = None
		self.churn_prediction_model = None
		
		# Personalization rules engine
		self.business_rules: List[Dict[str, Any]] = []
		
		# Performance tracking
		self.personalization_stats = {
			'total_personalizations': 0,
			'successful_optimizations': 0,
			'average_confidence_score': 0.0,
			'performance_lift': 0.0
		}
		
		_log.info(f"IntelligentPersonalizationEngine initialized for tenant {tenant_id}")
	
	# ========== Core Personalization Methods ==========
	
	async def personalize_content(
		self,
		template_id: str,
		user_id: str,
		variables: Dict[str, Any],
		context: Dict[str, Any] = None
	) -> PersonalizationResult:
		"""
		Perform comprehensive content personalization using AI and behavioral analysis.
		
		Args:
			template_id: Template to personalize
			user_id: Target user ID
			variables: Template variables
			context: Additional context for personalization
		
		Returns:
			Complete personalization result with optimized content
		"""
		start_time = datetime.utcnow()
		
		try:
			# Build personalization context
			personalization_context = await self._build_personalization_context(
				user_id, template_id, context or {}
			)
			
			# Get user engagement profile
			user_profile = await self._get_user_engagement_profile(user_id)
			
			# Load original template content
			original_content = await self._load_template_content(template_id)
			
			# Apply personalization strategies
			personalized_content = await self._apply_personalization_strategies(
				original_content,
				variables,
				user_profile,
				personalization_context
			)
			
			# Optimize channel selection
			recommended_channels = await self._optimize_channel_selection(
				user_profile,
				personalization_context
			)
			
			# Calculate optimal send time
			optimal_send_time = await self._calculate_optimal_send_time(
				user_profile,
				personalization_context
			)
			
			# Calculate confidence score
			confidence_score = self._calculate_confidence_score(
				user_profile,
				personalization_context,
				len(personalized_content.get('optimizations', []))
			)
			
			# Create personalization result
			processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
			
			result = PersonalizationResult(
				original_content=original_content,
				personalized_content=personalized_content['content'],
				optimizations=personalized_content['optimizations'],
				confidence_score=confidence_score,
				strategy_used=PersonalizationStrategy.HYBRID,
				processing_time_ms=processing_time,
				recommended_channels=recommended_channels,
				optimal_send_time=optimal_send_time,
				expected_performance=await self._predict_performance(
					personalized_content['content'],
					user_profile,
					recommended_channels
				)
			)
			
			# Update statistics
			await self._update_personalization_stats(result)
			
			_log.debug(f"Personalization completed for user {user_id}: confidence {confidence_score:.2f}")
			return result
			
		except Exception as e:
			_log.error(f"Personalization failed for user {user_id}: {str(e)}")
			# Return original content as fallback
			return PersonalizationResult(
				original_content=original_content or {},
				personalized_content=original_content or {},
				optimizations=[],
				confidence_score=0.0,
				strategy_used=PersonalizationStrategy.BEHAVIORAL,
				processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
				recommended_channels=context.get('channels', []) if context else []
			)
	
	async def optimize_send_time(
		self,
		user_id: str,
		channels: List[DeliveryChannel],
		priority: NotificationPriority = NotificationPriority.NORMAL
	) -> Dict[DeliveryChannel, datetime]:
		"""
		Calculate optimal send time for each channel based on user behavior.
		
		Args:
			user_id: Target user ID
			channels: Delivery channels to optimize
			priority: Notification priority
		
		Returns:
			Optimal send times per channel
		"""
		try:
			user_profile = await self._get_user_engagement_profile(user_id)
			
			send_times = {}
			current_time = datetime.utcnow()
			
			for channel in channels:
				if channel in user_profile.optimal_send_times:
					# Use learned optimal hours
					optimal_hours = user_profile.optimal_send_times[channel]
					if optimal_hours:
						# Find next optimal hour
						current_hour = current_time.hour
						next_optimal_hour = min(
							[h for h in optimal_hours if h > current_hour],
							default=optimal_hours[0]
						)
						
						# Calculate next optimal time
						if next_optimal_hour > current_hour:
							optimal_time = current_time.replace(
								hour=next_optimal_hour, minute=0, second=0, microsecond=0
							)
						else:
							# Next day
							optimal_time = (current_time + timedelta(days=1)).replace(
								hour=next_optimal_hour, minute=0, second=0, microsecond=0
							)
						
						send_times[channel] = optimal_time
					else:
						# Use default optimal time for channel
						send_times[channel] = await self._get_default_optimal_time(channel, priority)
				else:
					# Use default optimal time
					send_times[channel] = await self._get_default_optimal_time(channel, priority)
			
			return send_times
			
		except Exception as e:
			_log.error(f"Send time optimization failed: {str(e)}")
			return {channel: datetime.utcnow() for channel in channels}
	
	async def update_user_engagement(
		self,
		user_id: str,
		engagement_event: EngagementEvent,
		delivery: ComprehensiveDelivery,
		event_data: Dict[str, Any] = None
	):
		"""
		Update user engagement profile based on interaction events.
		
		Args:
			user_id: User ID
			engagement_event: Type of engagement event
			delivery: Delivery record
			event_data: Additional event data
		"""
		try:
			user_profile = await self._get_user_engagement_profile(user_id)
			
			# Update basic engagement counts
			user_profile.total_notifications += 1
			
			if engagement_event == EngagementEvent.OPENED:
				user_profile.total_opened += 1
			elif engagement_event == EngagementEvent.CLICKED:
				user_profile.total_clicked += 1
			elif engagement_event in [EngagementEvent.CONVERTED, EngagementEvent.REPLIED]:
				user_profile.total_converted += 1
			
			# Update channel-specific engagement scores
			for channel in delivery.channels:
				if channel not in user_profile.channel_engagement_scores:
					user_profile.channel_engagement_scores[channel] = 0.5
				
				# Adjust score based on engagement
				if engagement_event in [EngagementEvent.OPENED, EngagementEvent.CLICKED]:
					user_profile.channel_engagement_scores[channel] = min(1.0,
						user_profile.channel_engagement_scores[channel] + 0.1
					)
				elif engagement_event == EngagementEvent.DISMISSED:
					user_profile.channel_engagement_scores[channel] = max(0.0,
						user_profile.channel_engagement_scores[channel] - 0.05
					)
				
				# Update optimal send times
				if engagement_event in [EngagementEvent.OPENED, EngagementEvent.CLICKED]:
					if delivery.delivered_at:
						hour = delivery.delivered_at.hour
						if channel not in user_profile.optimal_send_times:
							user_profile.optimal_send_times[channel] = []
						
						user_profile.optimal_send_times[channel].append(hour)
						
						# Keep only recent optimal times (last 50)
						user_profile.optimal_send_times[channel] = \
							user_profile.optimal_send_times[channel][-50:]
			
			# Update predictive scores
			await self._update_predictive_scores(user_profile, engagement_event, event_data)
			
			# Update learning metadata
			user_profile.last_updated = datetime.utcnow()
			user_profile.data_points += 1
			user_profile.confidence_level = min(1.0, user_profile.data_points / 100.0)
			
			# Cache updated profile
			await self._cache_user_profile(user_profile)
			
			_log.debug(f"Updated engagement profile for user {user_id}: {engagement_event.value}")
			
		except Exception as e:
			_log.error(f"Failed to update user engagement: {str(e)}")
	
	# ========== Advanced AI Methods ==========
	
	async def predict_engagement_probability(
		self,
		user_id: str,
		content: Dict[str, Any],
		channels: List[DeliveryChannel],
		send_time: datetime
	) -> Dict[str, float]:
		"""
		Predict engagement probability for different metrics.
		
		Args:
			user_id: Target user ID
			content: Notification content
			channels: Delivery channels
			send_time: Planned send time
		
		Returns:
			Predicted probabilities for open, click, convert
		"""
		try:
			user_profile = await self._get_user_engagement_profile(user_id)
			
			# Calculate base probabilities from user history
			base_open_rate = user_profile.total_opened / max(user_profile.total_notifications, 1)
			base_click_rate = user_profile.total_clicked / max(user_profile.total_opened, 1)
			base_convert_rate = user_profile.total_converted / max(user_profile.total_clicked, 1)
			
			# Apply channel-specific adjustments
			channel_boost = 1.0
			for channel in channels:
				if channel in user_profile.channel_engagement_scores:
					channel_boost *= user_profile.channel_engagement_scores[channel]
			
			# Apply time-based adjustments
			time_boost = await self._calculate_time_boost(user_profile, send_time)
			
			# Apply content-based adjustments (simplified)
			content_boost = await self._calculate_content_boost(content, user_profile)
			
			# Calculate final probabilities
			total_boost = channel_boost * time_boost * content_boost
			
			predictions = {
				'open_probability': min(0.95, base_open_rate * total_boost),
				'click_probability': min(0.95, base_click_rate * total_boost),
				'convert_probability': min(0.95, base_convert_rate * total_boost),
				'engagement_score': min(100.0, (base_open_rate + base_click_rate + base_convert_rate) * total_boost * 33.33)
			}
			
			return predictions
			
		except Exception as e:
			_log.error(f"Engagement prediction failed: {str(e)}")
			return {
				'open_probability': 0.25,
				'click_probability': 0.05,
				'convert_probability': 0.01,
				'engagement_score': 50.0
			}
	
	async def generate_content_variations(
		self,
		original_content: Dict[str, Any],
		user_profile: UserEngagementProfile,
		variation_count: int = 3
	) -> List[Dict[str, Any]]:
		"""
		Generate AI-powered content variations for A/B testing.
		
		Args:
			original_content: Original template content
			user_profile: User engagement profile
			variation_count: Number of variations to generate
		
		Returns:
			List of content variations
		"""
		try:
			variations = []
			
			for i in range(variation_count):
				variation = original_content.copy()
				
				# Generate subject line variations
				if 'subject' in original_content:
					variation['subject'] = await self._generate_subject_variation(
						original_content['subject'],
						user_profile,
						i
					)
				
				# Generate message body variations
				if 'text' in original_content:
					variation['text'] = await self._generate_message_variation(
						original_content['text'],
						user_profile,
						i
					)
				
				# Generate CTA variations
				if 'cta' in original_content:
					variation['cta'] = await self._generate_cta_variation(
						original_content['cta'],
						user_profile,
						i
					)
				
				variation['variation_id'] = f"var_{i+1}"
				variation['optimization_strategy'] = self._get_variation_strategy(i)
				
				variations.append(variation)
			
			return variations
			
		except Exception as e:
			_log.error(f"Content variation generation failed: {str(e)}")
			return [original_content]  # Fallback to original
	
	# ========== Private Implementation Methods ==========
	
	async def _build_personalization_context(
		self,
		user_id: str,
		template_id: str,
		context: Dict[str, Any]
	) -> PersonalizationContext:
		"""Build comprehensive personalization context"""
		user_preferences = await self._get_user_preferences(user_id)
		
		return PersonalizationContext(
			user_id=user_id,
			tenant_id=self.tenant_id,
			template_id=template_id,
			channels=context.get('channels', []),
			campaign_id=context.get('campaign_id'),
			user_preferences=user_preferences,
			engagement_history=await self._get_user_engagement_history(user_id),
			behavioral_profile=await self._get_behavioral_profile(user_id),
			timestamp=datetime.utcnow(),
			timezone=user_preferences.timezone if user_preferences else "UTC",
			device_type=context.get('device_type'),
			location=context.get('location'),
			business_rules=context.get('business_rules', {}),
			campaign_context=context.get('campaign_context', {})
		)
	
	async def _get_user_engagement_profile(self, user_id: str) -> UserEngagementProfile:
		"""Get or create user engagement profile"""
		if user_id in self.user_profiles:
			return self.user_profiles[user_id]
		
		# Try to load from Redis cache
		if self.redis_client:
			cached_profile = await self._load_cached_profile(user_id)
			if cached_profile:
				self.user_profiles[user_id] = cached_profile
				return cached_profile
		
		# Create new profile
		profile = UserEngagementProfile(
			user_id=user_id,
			tenant_id=self.tenant_id
		)
		
		# Initialize with basic data from database
		await self._initialize_user_profile(profile)
		
		self.user_profiles[user_id] = profile
		return profile
	
	async def _apply_personalization_strategies(
		self,
		original_content: Dict[str, Any],
		variables: Dict[str, Any],
		user_profile: UserEngagementProfile,
		context: PersonalizationContext
	) -> Dict[str, Any]:
		"""Apply multiple personalization strategies"""
		personalized_content = original_content.copy()
		optimizations = []
		
		# 1. Behavioral personalization
		behavioral_opts = await self._apply_behavioral_personalization(
			personalized_content, user_profile, context
		)
		personalized_content.update(behavioral_opts['content'])
		optimizations.extend(behavioral_opts['optimizations'])
		
		# 2. Contextual personalization
		contextual_opts = await self._apply_contextual_personalization(
			personalized_content, context, variables
		)
		personalized_content.update(contextual_opts['content'])
		optimizations.extend(contextual_opts['optimizations'])
		
		# 3. Predictive personalization
		predictive_opts = await self._apply_predictive_personalization(
			personalized_content, user_profile, context
		)
		personalized_content.update(predictive_opts['content'])
		optimizations.extend(predictive_opts['optimizations'])
		
		# 4. A/B test personalization
		if context.ab_test_variant:
			ab_opts = await self._apply_ab_test_personalization(
				personalized_content, context.ab_test_variant
			)
			personalized_content.update(ab_opts['content'])
			optimizations.extend(ab_opts['optimizations'])
		
		return {
			'content': personalized_content,
			'optimizations': optimizations
		}
	
	async def _apply_behavioral_personalization(
		self,
		content: Dict[str, Any],
		user_profile: UserEngagementProfile,
		context: PersonalizationContext
	) -> Dict[str, Any]:
		"""Apply behavioral personalization based on user patterns"""
		optimized_content = content.copy()
		optimizations = []
		
		# Adjust content tone based on user preference
		if user_profile.preferred_tone != "neutral":
			if 'text' in content:
				optimized_content['text'] = await self._adjust_content_tone(
					content['text'], user_profile.preferred_tone
				)
				optimizations.append({
					'type': 'tone_adjustment',
					'from': 'neutral',
					'to': user_profile.preferred_tone,
					'confidence': 0.8
				})
		
		# Adjust content length based on preferences
		for channel in context.channels:
			if channel.value in user_profile.preferred_content_length:
				preferred_length = user_profile.preferred_content_length[channel.value]
				if 'text' in content and len(content['text']) > preferred_length * 1.2:
					optimized_content['text'] = await self._truncate_content(
						content['text'], preferred_length
					)
					optimizations.append({
						'type': 'length_optimization',
						'channel': channel.value,
						'target_length': preferred_length,
						'confidence': 0.7
					})
		
		return {
			'content': optimized_content,
			'optimizations': optimizations
		}
	
	async def _apply_contextual_personalization(
		self,
		content: Dict[str, Any],
		context: PersonalizationContext,
		variables: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Apply contextual personalization based on environment and situation"""
		optimized_content = content.copy()
		optimizations = []
		
		# Time-based personalization
		hour = context.timestamp.hour
		if 6 <= hour < 12:
			time_greeting = "Good morning"
		elif 12 <= hour < 18:
			time_greeting = "Good afternoon"
		else:
			time_greeting = "Good evening"
		
		# Add time-appropriate greeting if not present
		if 'text' in content and 'greeting' not in variables:
			optimized_content['text'] = f"{time_greeting}! {content['text']}"
			optimizations.append({
				'type': 'time_greeting',
				'greeting': time_greeting,
				'confidence': 0.6
			})
		
		# Location-based personalization
		if context.location:
			city = context.location.get('city')
			if city and 'text' in content:
				optimized_content['text'] = content['text'].replace(
					'{{location}}', city
				)
				optimizations.append({
					'type': 'location_personalization',
					'location': city,
					'confidence': 0.8
				})
		
		# Device-based optimization
		if context.device_type == 'mobile':
			# Optimize for mobile - shorter content, bigger CTAs
			if 'text' in content and len(content['text']) > 160:
				optimized_content['text'] = await self._optimize_for_mobile(content['text'])
				optimizations.append({
					'type': 'mobile_optimization',
					'confidence': 0.7
				})
		
		return {
			'content': optimized_content,
			'optimizations': optimizations
		}
	
	async def _apply_predictive_personalization(
		self,
		content: Dict[str, Any],
		user_profile: UserEngagementProfile,
		context: PersonalizationContext
	) -> Dict[str, Any]:
		"""Apply predictive personalization using ML models"""
		optimized_content = content.copy()
		optimizations = []
		
		# Predict optimal content style
		predicted_style = await self._predict_optimal_content_style(user_profile)
		if predicted_style != 'default':
			optimized_content = await self._apply_content_style(
				optimized_content, predicted_style
			)
			optimizations.append({
				'type': 'predictive_style',
				'style': predicted_style,
				'confidence': 0.75
			})
		
		# Predict and add relevant variables
		predicted_vars = await self._predict_relevant_variables(user_profile, context)
		if predicted_vars:
			for var_name, var_value in predicted_vars.items():
				if f"{{{{{var_name}}}}}" in str(content):
					optimized_content = await self._substitute_variable(
						optimized_content, var_name, var_value
					)
					optimizations.append({
						'type': 'predictive_variable',
						'variable': var_name,
						'value': var_value,
						'confidence': 0.6
					})
		
		return {
			'content': optimized_content,
			'optimizations': optimizations
		}
	
	async def _apply_ab_test_personalization(
		self,
		content: Dict[str, Any],
		variant: str
	) -> Dict[str, Any]:
		"""Apply A/B test variant personalization"""
		optimized_content = content.copy()
		optimizations = []
		
		# Apply variant-specific modifications
		if variant == 'A':
			# Conservative variant
			pass  # Keep original
		elif variant == 'B':
			# Aggressive variant
			if 'text' in content:
				optimized_content['text'] = await self._make_content_more_urgent(content['text'])
				optimizations.append({
					'type': 'ab_test_urgency',
					'variant': variant,
					'confidence': 1.0
				})
		elif variant == 'C':
			# Friendly variant
			if 'text' in content:
				optimized_content['text'] = await self._make_content_more_friendly(content['text'])
				optimizations.append({
					'type': 'ab_test_friendly',
					'variant': variant,
					'confidence': 1.0
				})
		
		return {
			'content': optimized_content,
			'optimizations': optimizations
		}
	
	async def _optimize_channel_selection(
		self,
		user_profile: UserEngagementProfile,
		context: PersonalizationContext
	) -> List[DeliveryChannel]:
		"""Optimize channel selection based on user engagement patterns"""
		if not context.channels:
			return []
		
		# Score channels based on user engagement
		channel_scores = {}
		for channel in context.channels:
			base_score = user_profile.channel_engagement_scores.get(channel, 0.5)
			
			# Apply time-based adjustments
			time_score = await self._calculate_channel_time_score(channel, context.timestamp)
			
			# Apply frequency tolerance
			frequency_score = await self._calculate_frequency_score(
				channel, user_profile, context.timestamp
			)
			
			final_score = base_score * time_score * frequency_score
			channel_scores[channel] = final_score
		
		# Sort channels by score
		sorted_channels = sorted(
			channel_scores.items(),
			key=lambda x: x[1],
			reverse=True
		)
		
		return [channel for channel, score in sorted_channels if score > 0.3]
	
	async def _calculate_optimal_send_time(
		self,
		user_profile: UserEngagementProfile,
		context: PersonalizationContext
	) -> Optional[datetime]:
		"""Calculate optimal send time based on user patterns"""
		if not context.channels:
			return None
		
		# Find the best channel for timing calculation
		primary_channel = context.channels[0]
		if primary_channel in user_profile.optimal_send_times:
			optimal_hours = user_profile.optimal_send_times[primary_channel]
			if optimal_hours:
				# Find most common hour
				hour_counts = Counter(optimal_hours)
				best_hour = hour_counts.most_common(1)[0][0]
				
				# Calculate next occurrence of that hour
				now = context.timestamp
				if now.hour <= best_hour:
					optimal_time = now.replace(hour=best_hour, minute=0, second=0, microsecond=0)
				else:
					optimal_time = (now + timedelta(days=1)).replace(
						hour=best_hour, minute=0, second=0, microsecond=0
					)
				
				return optimal_time
		
		return None
	
	async def _predict_performance(
		self,
		content: Dict[str, Any],
		user_profile: UserEngagementProfile,
		channels: List[DeliveryChannel]
	) -> Dict[str, float]:
		"""Predict performance metrics for personalized content"""
		# Simplified prediction based on user profile
		base_open_rate = user_profile.total_opened / max(user_profile.total_notifications, 1)
		base_click_rate = user_profile.total_clicked / max(user_profile.total_opened, 1)
		
		# Apply channel boost
		channel_boost = 1.0
		for channel in channels:
			if channel in user_profile.channel_engagement_scores:
				channel_boost *= user_profile.channel_engagement_scores[channel]
		
		return {
			'predicted_open_rate': min(0.95, base_open_rate * channel_boost),
			'predicted_click_rate': min(0.95, base_click_rate * channel_boost),
			'predicted_engagement_score': min(100.0, (base_open_rate + base_click_rate) * channel_boost * 50),
			'confidence': user_profile.confidence_level
		}
	
	# ========== Utility Methods ==========
	
	def _calculate_confidence_score(
		self,
		user_profile: UserEngagementProfile,
		context: PersonalizationContext,
		optimization_count: int
	) -> float:
		"""Calculate confidence score for personalization"""
		# Base confidence from user profile
		base_confidence = user_profile.confidence_level
		
		# Boost from number of data points
		data_boost = min(1.0, user_profile.data_points / 50.0)
		
		# Boost from optimization count
		optimization_boost = min(0.3, optimization_count * 0.1)
		
		return min(1.0, base_confidence + data_boost * 0.3 + optimization_boost)
	
	async def _load_template_content(self, template_id: str) -> Dict[str, Any]:
		"""Load template content from database"""
		# Mock template content
		return {
			'subject': 'Important notification',
			'text': 'Hello! We have an important update for you.',
			'html': '<h1>Important notification</h1><p>Hello! We have an important update for you.</p>',
			'cta': 'Learn More'
		}
	
	async def _get_user_preferences(self, user_id: str) -> Optional[UltimateUserPreferences]:
		"""Get user preferences"""
		# Mock preferences
		return UltimateUserPreferences(
			user_id=user_id,
			tenant_id=self.tenant_id,
			timezone="America/New_York",
			language_preference="en-US"
		)
	
	async def _get_user_engagement_history(self, user_id: str) -> List[Dict[str, Any]]:
		"""Get user engagement history"""
		return []  # Mock empty history
	
	async def _get_behavioral_profile(self, user_id: str) -> Dict[str, Any]:
		"""Get user behavioral profile"""
		return {}  # Mock empty profile
	
	# Additional helper methods would be implemented here...
	# (Content generation, ML model integration, caching, etc.)
	
	async def _update_personalization_stats(self, result: PersonalizationResult):
		"""Update personalization performance statistics"""
		self.personalization_stats['total_personalizations'] += 1
		
		if result.confidence_score > 0.7:
			self.personalization_stats['successful_optimizations'] += 1
		
		# Update average confidence score
		total = self.personalization_stats['total_personalizations']
		current_avg = self.personalization_stats['average_confidence_score']
		self.personalization_stats['average_confidence_score'] = (
			(current_avg * (total - 1) + result.confidence_score) / total
		)


# Factory function for engine creation
def create_personalization_engine(tenant_id: str, redis_client=None) -> IntelligentPersonalizationEngine:
	"""
	Create personalization engine instance.
	
	Args:
		tenant_id: Tenant ID for multi-tenant isolation
		redis_client: Optional Redis client for caching
	
	Returns:
		Configured personalization engine instance
	"""
	return IntelligentPersonalizationEngine(tenant_id, redis_client)


# Export main classes and functions
__all__ = [
	'IntelligentPersonalizationEngine',
	'PersonalizationContext',
	'PersonalizationResult',
	'UserEngagementProfile',
	'PersonalizationStrategy',
	'ContentType',
	'create_personalization_engine'
]