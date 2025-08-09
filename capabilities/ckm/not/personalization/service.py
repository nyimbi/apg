"""
APG Notification/Personalization Subcapability - Service Layer

Comprehensive service layer providing high-level personalization operations,
integration with parent notification capability, and enterprise-grade
personalization management for hyper-intelligent message customization.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import redis
import uuid

# Import parent capability components
from ..api_models import (
	DeliveryChannel, NotificationPriority, EngagementEvent,
	ComprehensiveDelivery, AdvancedCampaign, UltimateNotificationTemplate
)
from ..service import NotificationService

# Import subcapability components
from .core import (
	DeepPersonalizationEngine, PersonalizationOrchestrator,
	PersonalizationRequest, PersonalizationResponse, PersonalizationProfile,
	PersonalizationStrategy, PersonalizationTrigger
)
from .ai_models import (
	ContentGenerationModel, BehavioralAnalysisModel,
	create_content_generation_model, create_behavioral_analysis_model
)


# Configure logging
_log = logging.getLogger(__name__)


class PersonalizationServiceLevel(str, Enum):
	"""Personalization service levels"""
	BASIC = "basic"				# Template-based personalization
	STANDARD = "standard"		# AI-enhanced personalization
	PREMIUM = "premium"			# Advanced AI with behavioral analysis
	ENTERPRISE = "enterprise"	# Full AI suite with real-time adaptation
	QUANTUM = "quantum"			# Revolutionary quantum-level personalization


@dataclass
class PersonalizationConfig:
	"""Personalization service configuration"""
	service_level: PersonalizationServiceLevel = PersonalizationServiceLevel.STANDARD
	enable_real_time: bool = True
	enable_predictive: bool = True
	enable_emotional_intelligence: bool = False
	enable_cross_channel_sync: bool = True
	
	# AI model configuration
	content_generation_enabled: bool = True
	behavioral_analysis_enabled: bool = True
	sentiment_analysis_enabled: bool = False
	
	# Performance configuration
	max_response_time_ms: int = 100
	min_quality_score: float = 0.7
	cache_ttl_seconds: int = 3600
	
	# Enterprise features
	compliance_mode: bool = False
	audit_logging: bool = True
	data_retention_days: int = 365


class PersonalizationService:
	"""
	Comprehensive personalization service providing enterprise-grade
	personalization capabilities for the notification system.
	"""
	
	def __init__(
		self,
		tenant_id: str,
		config: PersonalizationConfig = None,
		redis_client: redis.Redis = None,
		notification_service: NotificationService = None
	):
		"""Initialize personalization service"""
		self.tenant_id = tenant_id
		self.config = config or PersonalizationConfig()
		self.redis_client = redis_client
		self.notification_service = notification_service
		
		# Initialize core engine
		self.personalization_engine = DeepPersonalizationEngine(
			tenant_id, 
			config={'service_level': self.config.service_level.value}
		)
		
		# Initialize AI models based on configuration
		self.ai_models = {}
		if self.config.content_generation_enabled:
			self.ai_models['content_generator'] = create_content_generation_model()
		if self.config.behavioral_analysis_enabled:
			self.ai_models['behavioral_analyzer'] = create_behavioral_analysis_model()
		
		# Service statistics
		self.service_stats = {
			'total_personalizations': 0,
			'successful_personalizations': 0,
			'cache_hits': 0,
			'ai_model_calls': 0,
			'avg_quality_score': 0.0,
			'avg_response_time_ms': 0.0
		}
		
		# Integration registry
		self.integration_hooks = {
			'pre_personalization': [],
			'post_personalization': [],
			'profile_updated': [],
			'quality_threshold_failed': []
		}
		
		_log.info(f"PersonalizationService initialized for tenant {tenant_id} "
				 f"(level: {self.config.service_level.value})")
	
	# ========== Core Personalization Operations ==========
	
	async def personalize_notification(
		self,
		notification_template: UltimateNotificationTemplate,
		user_id: str,
		context: Dict[str, Any] = None,
		channels: List[DeliveryChannel] = None
	) -> Dict[str, Any]:
		"""
		Personalize a notification template for a specific user.
		Main entry point for single notification personalization.
		"""
		try:
			# Execute pre-personalization hooks
			await self._execute_hooks('pre_personalization', {
				'template': notification_template,
				'user_id': user_id,
				'context': context
			})
			
			# Build personalization context
			personalization_context = await self._build_personalization_context(
				user_id, notification_template, context, channels
			)
			
			# Determine personalization strategies based on service level
			strategies = self._determine_personalization_strategies(
				self.config.service_level, personalization_context
			)
			
			# Execute personalization
			response = await self.personalization_engine.personalize_message(
				user_id=user_id,
				content=self._extract_template_content(notification_template),
				context=personalization_context,
				strategies=strategies
			)
			
			# Validate quality threshold
			if response.quality_score < self.config.min_quality_score:
				await self._execute_hooks('quality_threshold_failed', {
					'response': response,
					'threshold': self.config.min_quality_score
				})
			
			# Execute post-personalization hooks
			await self._execute_hooks('post_personalization', {
				'response': response,
				'original_template': notification_template
			})
			
			# Update service statistics
			await self._update_service_stats(response)
			
			# Convert response to notification format
			personalized_notification = await self._convert_to_notification_format(
				response, notification_template, channels
			)
			
			_log.debug(f"Notification personalized for user {user_id}: "
					  f"quality {response.quality_score:.2f}, "
					  f"time {response.processing_time_ms}ms")
			
			return personalized_notification
			
		except Exception as e:
			_log.error(f"Notification personalization failed for user {user_id}: {str(e)}")
			raise
	
	async def personalize_campaign(
		self,
		campaign: AdvancedCampaign,
		target_users: List[str],
		context: Dict[str, Any] = None
	) -> Dict[str, List[Dict[str, Any]]]:
		"""
		Personalize campaign content for multiple users.
		Optimized for bulk campaign operations with intelligent batching.
		"""
		try:
			_log.info(f"Starting campaign personalization: {campaign.id} "
					 f"for {len(target_users)} users")
			
			# Extract campaign content
			campaign_content = self._extract_campaign_content(campaign)
			
			# Build batch personalization context
			batch_context = await self._build_batch_personalization_context(
				campaign, target_users, context
			)
			
			# Execute batch personalization
			responses = await self.personalization_engine.personalize_campaign(
				campaign_id=campaign.id,
				user_ids=target_users,
				content=campaign_content,
				context=batch_context
			)
			
			# Process responses and organize by user
			personalized_content = {}
			successful_personalizations = 0
			
			for response in responses:
				if response.quality_score >= self.config.min_quality_score:
					personalized_content[response.user_id] = response.to_dict()
					successful_personalizations += 1
				else:
					# Use fallback personalization
					personalized_content[response.user_id] = await self._create_fallback_personalization(
						response.user_id, campaign_content
					)
			
			# Update campaign statistics
			campaign_stats = {
				'total_users': len(target_users),
				'successful_personalizations': successful_personalizations,
				'success_rate': successful_personalizations / len(target_users) if target_users else 0,
				'avg_quality_score': sum(r.quality_score for r in responses) / len(responses) if responses else 0
			}
			
			_log.info(f"Campaign personalization completed: {campaign.id} "
					 f"({successful_personalizations}/{len(target_users)} successful)")
			
			return {
				'personalized_content': personalized_content,
				'campaign_stats': campaign_stats,
				'responses': [r.to_dict() for r in responses]
			}
			
		except Exception as e:
			_log.error(f"Campaign personalization failed for {campaign.id}: {str(e)}")
			raise
	
	async def get_personalization_insights(
		self,
		user_id: str,
		include_predictions: bool = True
	) -> Dict[str, Any]:
		"""
		Get comprehensive personalization insights for a user.
		Includes behavioral analysis, preferences, and predictions.
		"""
		try:
			# Get user personalization profile
			profile = await self.personalization_engine.orchestrator.get_user_profile(user_id)
			
			# Get behavioral insights if enabled
			behavioral_insights = {}
			if self.config.behavioral_analysis_enabled and 'behavioral_analyzer' in self.ai_models:
				behavioral_model = self.ai_models['behavioral_analyzer']
				behavioral_prediction = await behavioral_model.predict({
					'user_data': asdict(profile),
					'analysis_type': 'comprehensive'
				})
				behavioral_insights = behavioral_prediction.prediction
			
			# Get content generation insights if enabled
			content_insights = {}
			if self.config.content_generation_enabled and 'content_generator' in self.ai_models:
				content_insights = await self._get_content_generation_insights(profile)
			
			# Compile comprehensive insights
			insights = {
				'user_id': user_id,
				'profile_summary': {
					'completeness': profile.profile_completeness,
					'confidence': profile.model_confidence,
					'data_quality': profile.data_quality_score,
					'last_updated': profile.last_updated.isoformat()
				},
				'personalization_preferences': {
					'content_preferences': profile.content_preferences,
					'channel_preferences': profile.channel_preferences,
					'timing_preferences': profile.timing_preferences,
					'frequency_preferences': profile.frequency_preferences
				},
				'behavioral_insights': behavioral_insights,
				'content_insights': content_insights,
				'predictive_scores': {
					'engagement_prediction': profile.engagement_prediction,
					'churn_risk': profile.churn_risk_score,
					'lifetime_value': profile.lifetime_value_score,
					'personalization_receptivity': profile.personalization_receptivity
				}
			}
			
			# Add predictions if requested
			if include_predictions and self.config.enable_predictive:
				insights['predictions'] = await self._generate_user_predictions(profile)
			
			return insights
			
		except Exception as e:
			_log.error(f"Failed to get personalization insights for user {user_id}: {str(e)}")
			raise
	
	async def update_user_preferences(
		self,
		user_id: str,
		preferences: Dict[str, Any],
		trigger: PersonalizationTrigger = PersonalizationTrigger.USER_INTERACTION
	):
		"""
		Update user personalization preferences and profile.
		Triggers profile recalculation and model updates.
		"""
		try:
			# Update user profile through orchestrator
			await self.personalization_engine.orchestrator.update_user_profile(
				user_id, preferences, trigger
			)
			
			# Execute profile updated hooks
			await self._execute_hooks('profile_updated', {
				'user_id': user_id,
				'preferences': preferences,
				'trigger': trigger
			})
			
			# Invalidate cache for user
			await self._invalidate_user_cache(user_id)
			
			_log.debug(f"User preferences updated: {user_id} ({trigger.value})")
			
		except Exception as e:
			_log.error(f"Failed to update user preferences for {user_id}: {str(e)}")
			raise
	
	# ========== AI Model Integration ==========
	
	async def generate_personalized_content(
		self,
		user_id: str,
		content_type: str,
		context: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""
		Generate completely new personalized content using AI models.
		"""
		if not self.config.content_generation_enabled or 'content_generator' not in self.ai_models:
			raise ValueError("Content generation not enabled")
		
		try:
			# Get user profile for personalization
			profile = await self.personalization_engine.orchestrator.get_user_profile(user_id)
			
			# Prepare features for content generation
			features = {
				'user_profile': asdict(profile),
				'content_type': content_type,
				'context': context or {},
				'tone': context.get('tone', 'friendly') if context else 'friendly',
				'length': context.get('length', 'medium') if context else 'medium',
				'language': context.get('language', 'en') if context else 'en'
			}
			
			# Generate content using AI model
			content_model = self.ai_models['content_generator']
			prediction = await content_model.predict(features)
			
			# Update model call statistics
			self.service_stats['ai_model_calls'] += 1
			
			return {
				'generated_content': prediction.prediction,
				'confidence_score': prediction.confidence_score,
				'generation_metadata': {
					'model_id': prediction.model_id,
					'model_version': prediction.model_version,
					'processing_time_ms': prediction.processing_time_ms,
					'explanation': prediction.explanation
				}
			}
			
		except Exception as e:
			_log.error(f"Content generation failed for user {user_id}: {str(e)}")
			raise
	
	async def analyze_user_behavior(
		self,
		user_id: str,
		analysis_type: str = "comprehensive"
	) -> Dict[str, Any]:
		"""
		Analyze user behavior using advanced AI models.
		"""
		if not self.config.behavioral_analysis_enabled or 'behavioral_analyzer' not in self.ai_models:
			raise ValueError("Behavioral analysis not enabled")
		
		try:
			# Get user profile and behavioral data
			profile = await self.personalization_engine.orchestrator.get_user_profile(user_id)
			
			# Prepare features for behavioral analysis
			features = {
				'user_data': asdict(profile),
				'analysis_type': analysis_type
			}
			
			# Analyze behavior using AI model
			behavioral_model = self.ai_models['behavioral_analyzer']
			prediction = await behavioral_model.predict(features)
			
			# Update model call statistics
			self.service_stats['ai_model_calls'] += 1
			
			return {
				'behavioral_analysis': prediction.prediction,
				'confidence_score': prediction.confidence_score,
				'analysis_metadata': {
					'model_id': prediction.model_id,
					'model_version': prediction.model_version,
					'processing_time_ms': prediction.processing_time_ms,
					'explanation': prediction.explanation
				}
			}
			
		except Exception as e:
			_log.error(f"Behavioral analysis failed for user {user_id}: {str(e)}")
			raise
	
	# ========== Integration & Hooks ==========
	
	def register_hook(self, hook_type: str, callback):
		"""Register integration hook"""
		if hook_type in self.integration_hooks:
			self.integration_hooks[hook_type].append(callback)
			_log.info(f"Hook registered: {hook_type}")
		else:
			raise ValueError(f"Unknown hook type: {hook_type}")
	
	async def _execute_hooks(self, hook_type: str, data: Dict[str, Any]):
		"""Execute registered hooks"""
		if hook_type in self.integration_hooks:
			for callback in self.integration_hooks[hook_type]:
				try:
					if asyncio.iscoroutinefunction(callback):
						await callback(data)
					else:
						callback(data)
				except Exception as e:
					_log.error(f"Hook execution failed ({hook_type}): {str(e)}")
	
	# ========== Service Management ==========
	
	def get_service_stats(self) -> Dict[str, Any]:
		"""Get comprehensive service statistics"""
		engine_stats = self.personalization_engine.get_performance_stats()
		
		return {
			'service_stats': self.service_stats,
			'engine_stats': engine_stats,
			'ai_model_stats': {
				model_id: model.get_model_info() 
				for model_id, model in self.ai_models.items()
			},
			'service_config': {
				'service_level': self.config.service_level.value,
				'features_enabled': {
					'real_time': self.config.enable_real_time,
					'predictive': self.config.enable_predictive,
					'emotional_intelligence': self.config.enable_emotional_intelligence,
					'cross_channel_sync': self.config.enable_cross_channel_sync,
					'content_generation': self.config.content_generation_enabled,
					'behavioral_analysis': self.config.behavioral_analysis_enabled
				}
			}
		}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform comprehensive health check"""
		health_status = {
			'status': 'healthy',
			'timestamp': datetime.utcnow().isoformat(),
			'components': {}
		}
		
		# Check personalization engine
		try:
			engine_stats = self.personalization_engine.get_performance_stats()
			health_status['components']['personalization_engine'] = {
				'status': 'healthy',
				'total_personalizations': engine_stats['total_personalizations']
			}
		except Exception as e:
			health_status['components']['personalization_engine'] = {
				'status': 'unhealthy',
				'error': str(e)
			}
			health_status['status'] = 'degraded'
		
		# Check AI models
		for model_id, model in self.ai_models.items():
			try:
				model_info = model.get_model_info()
				health_status['components'][f'ai_model_{model_id}'] = {
					'status': model_info['status'],
					'version': model_info['version']
				}
			except Exception as e:
				health_status['components'][f'ai_model_{model_id}'] = {
					'status': 'error',
					'error': str(e)
				}
				health_status['status'] = 'degraded'
		
		# Check Redis connection if available
		if self.redis_client:
			try:
				self.redis_client.ping()
				health_status['components']['redis'] = {'status': 'healthy'}
			except Exception as e:
				health_status['components']['redis'] = {
					'status': 'unhealthy',
					'error': str(e)
				}
				health_status['status'] = 'degraded'
		
		return health_status
	
	# ========== Private Helper Methods ==========
	
	async def _build_personalization_context(
		self,
		user_id: str,
		template: UltimateNotificationTemplate,
		context: Dict[str, Any] = None,
		channels: List[DeliveryChannel] = None
	) -> Dict[str, Any]:
		"""Build comprehensive personalization context"""
		base_context = context or {}
		
		# Add template context
		base_context.update({
			'template_id': template.id,
			'template_name': template.name,
			'template_type': template.template_type.value if hasattr(template, 'template_type') else 'standard',
			'supported_channels': template.supported_channels
		})
		
		# Add channel context
		if channels:
			base_context['target_channels'] = [c.value for c in channels]
		
		# Add service context
		base_context.update({
			'service_level': self.config.service_level.value,
			'tenant_id': self.tenant_id,
			'timestamp': datetime.utcnow().isoformat()
		})
		
		return base_context
	
	def _determine_personalization_strategies(
		self,
		service_level: PersonalizationServiceLevel,
		context: Dict[str, Any]
	) -> List[PersonalizationStrategy]:
		"""Determine personalization strategies based on service level"""
		if service_level == PersonalizationServiceLevel.BASIC:
			return [PersonalizationStrategy.BEHAVIORAL_ADAPTIVE]
		elif service_level == PersonalizationServiceLevel.STANDARD:
			return [
				PersonalizationStrategy.NEURAL_CONTENT,
				PersonalizationStrategy.BEHAVIORAL_ADAPTIVE
			]
		elif service_level == PersonalizationServiceLevel.PREMIUM:
			return [
				PersonalizationStrategy.NEURAL_CONTENT,
				PersonalizationStrategy.BEHAVIORAL_ADAPTIVE,
				PersonalizationStrategy.CONTEXTUAL_INTELLIGENCE,
				PersonalizationStrategy.PREDICTIVE_OPTIMIZATION
			]
		elif service_level == PersonalizationServiceLevel.ENTERPRISE:
			return [
				PersonalizationStrategy.NEURAL_CONTENT,
				PersonalizationStrategy.BEHAVIORAL_ADAPTIVE,
				PersonalizationStrategy.EMOTIONAL_RESONANCE,
				PersonalizationStrategy.CONTEXTUAL_INTELLIGENCE,
				PersonalizationStrategy.PREDICTIVE_OPTIMIZATION,
				PersonalizationStrategy.CROSS_CHANNEL_SYNC
			]
		else:  # QUANTUM
			return [
				PersonalizationStrategy.QUANTUM_PERSONALIZATION,
				PersonalizationStrategy.NEURAL_CONTENT,
				PersonalizationStrategy.BEHAVIORAL_ADAPTIVE,
				PersonalizationStrategy.EMOTIONAL_RESONANCE,
				PersonalizationStrategy.CONTEXTUAL_INTELLIGENCE,
				PersonalizationStrategy.PREDICTIVE_OPTIMIZATION,
				PersonalizationStrategy.CROSS_CHANNEL_SYNC,
				PersonalizationStrategy.EMPATHY_DRIVEN,
				PersonalizationStrategy.REAL_TIME_ADAPTATION,
				PersonalizationStrategy.MULTI_DIMENSIONAL
			]
	
	def _extract_template_content(self, template: UltimateNotificationTemplate) -> Dict[str, Any]:
		"""Extract content from notification template"""
		return {
			'subject': template.subject_template or '',
			'html': template.html_template or '',
			'text': template.text_template or '',
			'sms': template.sms_template or '',
			'push': template.push_template or '',
			'metadata': {
				'template_id': template.id,
				'template_name': template.name,
				'version': template.version
			}
		}
	
	async def _update_service_stats(self, response: PersonalizationResponse):
		"""Update service statistics"""
		self.service_stats['total_personalizations'] += 1
		
		if response.quality_score >= self.config.min_quality_score:
			self.service_stats['successful_personalizations'] += 1
		
		if response.cache_hit:
			self.service_stats['cache_hits'] += 1
		
		# Update averages
		total = self.service_stats['total_personalizations']
		
		# Update average quality score
		current_avg_quality = self.service_stats['avg_quality_score']
		new_avg_quality = ((current_avg_quality * (total - 1)) + response.quality_score) / total
		self.service_stats['avg_quality_score'] = new_avg_quality
		
		# Update average response time
		current_avg_time = self.service_stats['avg_response_time_ms']
		new_avg_time = ((current_avg_time * (total - 1)) + response.processing_time_ms) / total
		self.service_stats['avg_response_time_ms'] = new_avg_time


# Factory function
def create_personalization_service(
	tenant_id: str,
	config: PersonalizationConfig = None,
	redis_client: redis.Redis = None,
	notification_service: NotificationService = None
) -> PersonalizationService:
	"""Create personalization service instance"""
	return PersonalizationService(tenant_id, config, redis_client, notification_service)


# Export main classes and functions
__all__ = [
	'PersonalizationService',
	'PersonalizationConfig',
	'PersonalizationServiceLevel',
	'create_personalization_service'
]