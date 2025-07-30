"""
APG Notification/Personalization Subcapability - REST API

Comprehensive REST API for personalization operations providing enterprise-grade
endpoints for AI-powered content personalization, behavioral analysis, user profiling,
and personalization management with OpenAPI documentation.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from flask import Flask, request, jsonify, g
from flask_restx import Api, Resource, Namespace, fields, marshal_with
from flask_restx.errors import abort
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from functools import wraps

# Import parent capability components
from ..api_models import DeliveryChannel, NotificationPriority

# Import personalization components
from .service import (
	PersonalizationService, PersonalizationConfig, PersonalizationServiceLevel,
	create_personalization_service
)
from .core import PersonalizationStrategy, PersonalizationTrigger


# Configure logging
_log = logging.getLogger(__name__)


# ========== API Models for OpenAPI Documentation ==========

def create_personalization_api_models(api: Api):
	"""Create API models for OpenAPI documentation"""
	
	# Request models
	personalization_request = api.model('PersonalizationRequest', {
		'user_id': fields.String(required=True, description='Target user ID'),
		'content': fields.Raw(required=True, description='Content to personalize'),
		'context': fields.Raw(description='Additional context for personalization'),
		'strategies': fields.List(fields.String, description='Personalization strategies to apply'),
		'channels': fields.List(fields.String, description='Target delivery channels'),
		'priority': fields.String(description='Notification priority', enum=['low', 'normal', 'high', 'urgent', 'critical']),
		'min_quality_score': fields.Float(description='Minimum quality score threshold', min=0.0, max=1.0),
		'require_real_time': fields.Boolean(description='Require real-time processing')
	})
	
	campaign_personalization_request = api.model('CampaignPersonalizationRequest', {
		'campaign_id': fields.String(required=True, description='Campaign ID'),
		'target_users': fields.List(fields.String, required=True, description='List of target user IDs'),
		'content': fields.Raw(required=True, description='Campaign content to personalize'),
		'context': fields.Raw(description='Campaign context'),
		'batch_size': fields.Integer(description='Batch processing size', default=100)
	})
	
	user_preferences_update = api.model('UserPreferencesUpdate', {
		'content_preferences': fields.Raw(description='Content preferences'),
		'channel_preferences': fields.Raw(description='Channel preferences'),
		'timing_preferences': fields.Raw(description='Timing preferences'),
		'frequency_preferences': fields.Raw(description='Frequency preferences'),
		'personalization_enabled': fields.Boolean(description='Enable personalization'),
		'trigger': fields.String(description='Update trigger', enum=[
			'user_interaction', 'behavioral_pattern', 'emotional_state_change',
			'context_shift', 'engagement_threshold', 'predictive_signal'
		])
	})
	
	content_generation_request = api.model('ContentGenerationRequest', {
		'user_id': fields.String(required=True, description='Target user ID'),
		'content_type': fields.String(required=True, description='Type of content to generate'),
		'tone': fields.String(description='Content tone', enum=['casual', 'formal', 'friendly', 'urgent', 'professional']),
		'length': fields.String(description='Content length', enum=['short', 'medium', 'long']),
		'language': fields.String(description='Content language', default='en'),
		'context': fields.Raw(description='Generation context')
	})
	
	behavioral_analysis_request = api.model('BehavioralAnalysisRequest', {
		'user_id': fields.String(required=True, description='Target user ID'),
		'analysis_type': fields.String(description='Type of analysis', default='comprehensive', enum=[
			'comprehensive', 'engagement_patterns', 'content_preferences', 'timing_patterns'
		])
	})
	
	# Response models
	personalization_response = api.model('PersonalizationResponse', {
		'request_id': fields.String(description='Request ID'),
		'user_id': fields.String(description='User ID'),
		'personalized_content': fields.Raw(description='Personalized content'),
		'original_content': fields.Raw(description='Original content'),
		'strategies_applied': fields.List(fields.String, description='Applied strategies'),
		'quality_score': fields.Float(description='Personalization quality score'),
		'confidence_score': fields.Float(description='Confidence score'),
		'personalization_level': fields.String(description='Personalization level'),
		'processing_time_ms': fields.Integer(description='Processing time in milliseconds'),
		'recommendations': fields.List(fields.String, description='Optimization recommendations'),
		'predicted_engagement': fields.Raw(description='Predicted engagement metrics'),
		'optimal_channels': fields.List(fields.String, description='Optimal delivery channels'),
		'optimal_timing': fields.String(description='Optimal send time'),
		'cache_hit': fields.Boolean(description='Whether result was served from cache')
	})
	
	user_insights_response = api.model('UserInsightsResponse', {
		'user_id': fields.String(description='User ID'),
		'profile_summary': fields.Raw(description='Profile summary'),
		'personalization_preferences': fields.Raw(description='Personalization preferences'),
		'behavioral_insights': fields.Raw(description='Behavioral insights'),
		'content_insights': fields.Raw(description='Content insights'),
		'predictive_scores': fields.Raw(description='Predictive scores'),
		'predictions': fields.Raw(description='Future predictions')
	})
	
	service_stats_response = api.model('ServiceStatsResponse', {
		'service_stats': fields.Raw(description='Service statistics'),
		'engine_stats': fields.Raw(description='Engine statistics'),
		'ai_model_stats': fields.Raw(description='AI model statistics'),
		'service_config': fields.Raw(description='Service configuration')
	})
	
	health_check_response = api.model('HealthCheckResponse', {
		'status': fields.String(description='Overall health status', enum=['healthy', 'degraded', 'unhealthy']),
		'timestamp': fields.String(description='Check timestamp'),
		'components': fields.Raw(description='Component health details')
	})
	
	return {
		'personalization_request': personalization_request,
		'campaign_personalization_request': campaign_personalization_request,
		'user_preferences_update': user_preferences_update,
		'content_generation_request': content_generation_request,
		'behavioral_analysis_request': behavioral_analysis_request,
		'personalization_response': personalization_response,
		'user_insights_response': user_insights_response,
		'service_stats_response': service_stats_response,
		'health_check_response': health_check_response
	}


# ========== Authentication & Authorization ==========

def require_auth(f):
	"""Decorator for authentication requirement"""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		# Mock authentication - would integrate with actual auth system
		auth_header = request.headers.get('Authorization')
		if not auth_header or not auth_header.startswith('Bearer '):
			abort(401, message='Authentication required')
		
		# Extract tenant_id from token (mock)
		g.tenant_id = request.headers.get('X-Tenant-ID', 'default_tenant')
		g.user_id = request.headers.get('X-User-ID', 'system')
		
		return f(*args, **kwargs)
	return decorated_function


def require_personalization_permission(permission: str):
	"""Decorator for personalization-specific permissions"""
	def decorator(f):
		@wraps(f)
		def decorated_function(*args, **kwargs):
			# Mock permission check - would integrate with actual permission system
			permissions = request.headers.get('X-Permissions', 'read,write').split(',')
			if permission not in permissions:
				abort(403, message=f'Permission required: {permission}')
			return f(*args, **kwargs)
		return decorated_function
	return decorator


# ========== API Namespaces ==========

def create_personalization_api(app: Flask) -> Api:
	"""Create and configure personalization API"""
	
	api = Api(
		app,
		title='APG Deep Personalization API',
		version='1.0.0',
		description='Revolutionary AI-powered personalization API for hyper-intelligent message customization',
		doc='/personalization/docs/',
		prefix='/api/v1/personalization'
	)
	
	# Create API models
	models = create_personalization_api_models(api)
	
	# Create namespaces
	personalization_ns = Namespace(
		'personalization',
		description='Core personalization operations',
		path='/personalization'
	)
	
	insights_ns = Namespace(
		'insights',
		description='User insights and behavioral analysis',
		path='/insights'
	)
	
	ai_ns = Namespace(
		'ai',
		description='AI model operations and content generation',
		path='/ai'
	)
	
	management_ns = Namespace(
		'management',
		description='Service management and administration',
		path='/management'
	)
	
	# Initialize personalization service (would be injected in production)
	personalization_service = None
	
	def get_personalization_service() -> PersonalizationService:
		"""Get personalization service instance"""
		nonlocal personalization_service
		if not personalization_service:
			config = PersonalizationConfig(
				service_level=PersonalizationServiceLevel.ENTERPRISE,
				enable_real_time=True,
				enable_predictive=True,
				enable_emotional_intelligence=True
			)
			personalization_service = create_personalization_service(
				tenant_id=g.get('tenant_id', 'default_tenant'),
				config=config
			)
		return personalization_service
	
	# ========== Personalization Endpoints ==========
	
	@personalization_ns.route('/personalize')
	class PersonalizeResource(Resource):
		@personalization_ns.doc('personalize_content')
		@personalization_ns.expect(models['personalization_request'])
		@personalization_ns.marshal_with(models['personalization_response'])
		@require_auth
		@require_personalization_permission('personalize')
		def post(self):
			"""Personalize content for a single user"""
			try:
				data = request.get_json()
				service = get_personalization_service()
				
				# Mock template creation from request data
				from ..api_models import UltimateNotificationTemplate
				template = UltimateNotificationTemplate(
					id=f"temp_{datetime.utcnow().timestamp()}",
					name="API Template",
					subject_template=data['content'].get('subject', ''),
					text_template=data['content'].get('text', ''),
					html_template=data['content'].get('html', ''),
					tenant_id=g.tenant_id
				)
				
				# Parse channels
				channels = []
				if data.get('channels'):
					channels = [DeliveryChannel(c) for c in data['channels']]
				
				# Run personalization (async wrapper)
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					result = loop.run_until_complete(
						service.personalize_notification(
							notification_template=template,
							user_id=data['user_id'],
							context=data.get('context', {}),
							channels=channels
						)
					)
					return result, 200
				finally:
					loop.close()
				
			except Exception as e:
				_log.error(f"Personalization API error: {str(e)}")
				abort(500, message=f"Personalization failed: {str(e)}")
	
	@personalization_ns.route('/campaigns/<string:campaign_id>/personalize')
	class CampaignPersonalizeResource(Resource):
		@personalization_ns.doc('personalize_campaign')
		@personalization_ns.expect(models['campaign_personalization_request'])
		@require_auth
		@require_personalization_permission('personalize')
		def post(self, campaign_id):
			"""Personalize campaign content for multiple users"""
			try:
				data = request.get_json()
				service = get_personalization_service()
				
				# Mock campaign creation
				from ..api_models import AdvancedCampaign, CampaignType
				campaign = AdvancedCampaign(
					id=campaign_id,
					name=f"Campaign {campaign_id}",
					campaign_type=CampaignType.PROMOTIONAL,
					tenant_id=g.tenant_id
				)
				
				# Run campaign personalization
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					result = loop.run_until_complete(
						service.personalize_campaign(
							campaign=campaign,
							target_users=data['target_users'],
							context=data.get('context', {})
						)
					)
					return result, 200
				finally:
					loop.close()
				
			except Exception as e:
				_log.error(f"Campaign personalization API error: {str(e)}")
				abort(500, message=f"Campaign personalization failed: {str(e)}")
	
	# ========== User Insights Endpoints ==========
	
	@insights_ns.route('/users/<string:user_id>')
	class UserInsightsResource(Resource):
		@insights_ns.doc('get_user_insights')
		@insights_ns.marshal_with(models['user_insights_response'])
		@require_auth
		@require_personalization_permission('read_insights')
		def get(self, user_id):
			"""Get comprehensive personalization insights for a user"""
			try:
				service = get_personalization_service()
				include_predictions = request.args.get('include_predictions', 'true').lower() == 'true'
				
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					insights = loop.run_until_complete(
						service.get_personalization_insights(
							user_id=user_id,
							include_predictions=include_predictions
						)
					)
					return insights, 200
				finally:
					loop.close()
				
			except Exception as e:
				_log.error(f"User insights API error: {str(e)}")
				abort(500, message=f"Failed to get insights: {str(e)}")
	
	@insights_ns.route('/users/<string:user_id>/preferences')
	class UserPreferencesResource(Resource):
		@insights_ns.doc('update_user_preferences')
		@insights_ns.expect(models['user_preferences_update'])
		@require_auth
		@require_personalization_permission('update_preferences')
		def put(self, user_id):
			"""Update user personalization preferences"""
			try:
				data = request.get_json()
				service = get_personalization_service()
				
				# Parse trigger
				trigger = PersonalizationTrigger.USER_INTERACTION
				if data.get('trigger'):
					trigger = PersonalizationTrigger(data['trigger'])
				
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					loop.run_until_complete(
						service.update_user_preferences(
							user_id=user_id,
							preferences=data,
							trigger=trigger
						)
					)
					return {'message': 'Preferences updated successfully'}, 200
				finally:
					loop.close()
				
			except Exception as e:
				_log.error(f"User preferences API error: {str(e)}")
				abort(500, message=f"Failed to update preferences: {str(e)}")
	
	# ========== AI Model Endpoints ==========
	
	@ai_ns.route('/generate-content')
	class ContentGenerationResource(Resource):
		@ai_ns.doc('generate_content')
		@ai_ns.expect(models['content_generation_request'])
		@require_auth
		@require_personalization_permission('generate_content')
		def post(self):
			"""Generate personalized content using AI models"""
			try:
				data = request.get_json()
				service = get_personalization_service()
				
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					result = loop.run_until_complete(
						service.generate_personalized_content(
							user_id=data['user_id'],
							content_type=data['content_type'],
							context=data.get('context', {})
						)
					)
					return result, 200
				finally:
					loop.close()
				
			except Exception as e:
				_log.error(f"Content generation API error: {str(e)}")
				abort(500, message=f"Content generation failed: {str(e)}")
	
	@ai_ns.route('/analyze-behavior')
	class BehavioralAnalysisResource(Resource):
		@ai_ns.doc('analyze_behavior')
		@ai_ns.expect(models['behavioral_analysis_request'])
		@require_auth
		@require_personalization_permission('analyze_behavior')
		def post(self):
			"""Analyze user behavior using AI models"""
			try:
				data = request.get_json()
				service = get_personalization_service()
				
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					result = loop.run_until_complete(
						service.analyze_user_behavior(
							user_id=data['user_id'],
							analysis_type=data.get('analysis_type', 'comprehensive')
						)
					)
					return result, 200
				finally:
					loop.close()
				
			except Exception as e:
				_log.error(f"Behavioral analysis API error: {str(e)}")
				abort(500, message=f"Behavioral analysis failed: {str(e)}")
	
	# ========== Management Endpoints ==========
	
	@management_ns.route('/stats')
	class ServiceStatsResource(Resource):
		@management_ns.doc('get_service_stats')
		@management_ns.marshal_with(models['service_stats_response'])
		@require_auth
		@require_personalization_permission('admin')
		def get(self):
			"""Get comprehensive service statistics"""
			try:
				service = get_personalization_service()
				stats = service.get_service_stats()
				return stats, 200
				
			except Exception as e:
				_log.error(f"Service stats API error: {str(e)}")
				abort(500, message=f"Failed to get stats: {str(e)}")
	
	@management_ns.route('/health')
	class HealthCheckResource(Resource):
		@management_ns.doc('health_check')
		@management_ns.marshal_with(models['health_check_response'])
		def get(self):
			"""Perform comprehensive health check"""
			try:
				# Don't require auth for health check
				service = get_personalization_service()
				
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				try:
					health = loop.run_until_complete(service.health_check())
					status_code = 200 if health['status'] == 'healthy' else 503
					return health, status_code
				finally:
					loop.close()
				
			except Exception as e:
				_log.error(f"Health check API error: {str(e)}")
				return {
					'status': 'unhealthy',
					'timestamp': datetime.utcnow().isoformat(),
					'error': str(e)
				}, 503
	
	@management_ns.route('/config')
	class ConfigurationResource(Resource):
		@management_ns.doc('get_config')
		@require_auth
		@require_personalization_permission('admin')
		def get(self):
			"""Get service configuration"""
			try:
				service = get_personalization_service()
				config_info = {
					'service_level': service.config.service_level.value,
					'features': {
						'real_time': service.config.enable_real_time,
						'predictive': service.config.enable_predictive,
						'emotional_intelligence': service.config.enable_emotional_intelligence,
						'cross_channel_sync': service.config.enable_cross_channel_sync,
						'content_generation': service.config.content_generation_enabled,
						'behavioral_analysis': service.config.behavioral_analysis_enabled
					},
					'performance': {
						'max_response_time_ms': service.config.max_response_time_ms,
						'min_quality_score': service.config.min_quality_score,
						'cache_ttl_seconds': service.config.cache_ttl_seconds
					},
					'tenant_id': service.tenant_id
				}
				return config_info, 200
				
			except Exception as e:
				_log.error(f"Config API error: {str(e)}")
				abort(500, message=f"Failed to get config: {str(e)}")
	
	# Add namespaces to API
	api.add_namespace(personalization_ns)
	api.add_namespace(insights_ns)
	api.add_namespace(ai_ns)
	api.add_namespace(management_ns)
	
	return api


# Factory function
def create_personalization_api_blueprint(app: Flask):
	"""Create personalization API blueprint"""
	api = create_personalization_api(app)
	return api


# Export main functions
__all__ = [
	'create_personalization_api',
	'create_personalization_api_blueprint'
]