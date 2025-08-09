"""
APG Multi-Factor Authentication (MFA) - Flask Blueprint Integration

Flask Blueprint for APG MFA capability providing seamless integration
with the APG platform's Flask-AppBuilder architecture.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import logging
from flask import Blueprint, current_app, request, jsonify
from flask_appbuilder import AppBuilder
from typing import Optional, Dict, Any

from .views import register_mfa_views
from .api import create_mfa_api, MFAWebSocketEvents
from .service import MFAService
from .integration import APGIntegrationRouter
from .notification_service import MFANotificationService


def _log_blueprint_operation(operation: str, details: str = "") -> str:
	"""Log blueprint operations for debugging"""
	return f"[MFA Blueprint] {operation}: {details}"


class MFABlueprint:
	"""
	APG MFA capability blueprint providing complete integration
	with Flask-AppBuilder and the APG platform ecosystem.
	"""
	
	def __init__(self, name: str = 'mfa'):
		"""Initialize MFA blueprint"""
		self.name = name
		self.blueprint = Blueprint(
			name,
			__name__,
			url_prefix='/mfa',
			template_folder='templates',
			static_folder='static'
		)
		self.logger = logging.getLogger(__name__)
		
		# Service instances
		self.mfa_service: Optional[MFAService] = None
		self.integration_router: Optional[APGIntegrationRouter] = None
		self.notification_service: Optional[MFANotificationService] = None
		self.websocket_events: Optional[MFAWebSocketEvents] = None
		
		# Register blueprint routes
		self._register_routes()
	
	def init_app(self, app, appbuilder: AppBuilder, **kwargs):
		"""Initialize MFA capability with Flask app and AppBuilder"""
		try:
			self.logger.info(_log_blueprint_operation("initializing", "MFA capability"))
			
			# Get configuration
			config = kwargs.get('config', {})
			database_client = kwargs.get('database_client')
			integration_router = kwargs.get('integration_router')
			encryption_key = kwargs.get('encryption_key')
			socketio = kwargs.get('socketio')
			
			# Validate required dependencies
			if not database_client:
				raise ValueError("Database client is required for MFA capability")
			
			if not integration_router:
				raise ValueError("APG integration router is required for MFA capability")
			
			if not encryption_key:
				raise ValueError("Encryption key is required for MFA capability")
			
			# Initialize services
			self.integration_router = integration_router
			self.mfa_service = MFAService(
				database_client=database_client,
				integration_router=integration_router,
				encryption_key=encryption_key
			)
			
			self.notification_service = MFANotificationService(integration_router)
			
			# Initialize WebSocket events if available
			if socketio:
				self.websocket_events = MFAWebSocketEvents(socketio, self.mfa_service)
				self.logger.info(_log_blueprint_operation("websocket_initialized", "Real-time support enabled"))
			
			# Register Flask-AppBuilder views
			register_mfa_views(appbuilder)
			self.logger.info(_log_blueprint_operation("views_registered", "Flask-AppBuilder views"))
			
			# Create REST API
			api = create_mfa_api(app, self.mfa_service)
			self.logger.info(_log_blueprint_operation("api_created", "REST API endpoints"))
			
			# Register MFA notification templates
			self._register_notification_templates()
			
			# Register with APG composition engine
			self._register_with_apg_composition(app)
			
			# Register blueprint with app
			app.register_blueprint(self.blueprint)
			
			# Add MFA service to app context
			app.extensions = getattr(app, 'extensions', {})
			app.extensions['mfa'] = {
				'service': self.mfa_service,
				'integration': self.integration_router,
				'notifications': self.notification_service
			}
			
			self.logger.info(_log_blueprint_operation("initialization_complete", "MFA capability ready"))
			
		except Exception as e:
			self.logger.error(f"MFA blueprint initialization failed: {str(e)}", exc_info=True)
			raise
	
	def _register_routes(self):
		"""Register additional blueprint routes"""
		
		@self.blueprint.route('/health')
		def health_check():
			"""Health check endpoint"""
			try:
				# Basic health check
				status = {
					'status': 'healthy',
					'capability': 'mfa',
					'version': '1.0.0',
					'timestamp': '2025-01-29T12:00:00Z'
				}
				
				# Check service health if available
				if self.mfa_service:
					try:
						metrics = self.mfa_service.get_service_metrics()
						status['metrics'] = {
							'system_health': metrics.get('system_health', 'unknown'),
							'active_users': metrics.get('active_users', 0)
						}
					except Exception as e:
						status['service_status'] = f'degraded: {str(e)}'
				
				return jsonify(status)
				
			except Exception as e:
				return jsonify({
					'status': 'unhealthy',
					'error': str(e),
					'capability': 'mfa'
				}), 500
		
		@self.blueprint.route('/info')
		def capability_info():
			"""Get MFA capability information"""
			return jsonify({
				'name': 'Multi-Factor Authentication',
				'capability_id': 'mfa',
				'version': '1.0.0',
				'description': 'Enterprise-grade multi-factor authentication with biometric support',
				'features': [
					'Intelligent Adaptive Authentication',
					'Multi-Modal Biometric Authentication',
					'AI-Powered Risk Assessment',
					'Account Recovery System',
					'Real-Time Security Analytics',
					'Enterprise Policy Management'
				],
				'endpoints': {
					'api': '/api/mfa',
					'dashboard': '/mfadashboardview/dashboard/',
					'documentation': '/mfa/docs/'
				},
				'integrations': [
					'auth_rbac',
					'notification',
					'audit_compliance',
					'ai_orchestration',
					'computer_vision'
				]
			})
		
		@self.blueprint.route('/config')
		def get_configuration():
			"""Get MFA configuration (admin only)"""
			try:
				# This would include configuration checks in production
				config = {
					'methods_available': [
						'TOTP', 'SMS', 'EMAIL', 'FACE_RECOGNITION', 
						'VOICE_RECOGNITION', 'HARDWARE_TOKEN'
					],
					'biometric_enabled': True,
					'recovery_methods': [
						'email_verification', 'backup_codes', 'admin_override'
					],
					'policy_enforcement': True,
					'real_time_support': self.websocket_events is not None
				}
				
				return jsonify(config)
				
			except Exception as e:
				return jsonify({'error': str(e)}), 500
	
	async def _register_notification_templates(self):
		"""Register MFA notification templates with the notification service"""
		try:
			if self.notification_service:
				success = await self.notification_service.register_mfa_notification_templates()
				if success:
					self.logger.info(_log_blueprint_operation("templates_registered", "MFA notification templates"))
				else:
					self.logger.warning(_log_blueprint_operation("templates_failed", "Failed to register notification templates"))
			
		except Exception as e:
			self.logger.error(f"Template registration error: {str(e)}", exc_info=True)
	
	def _register_with_apg_composition(self, app):
		"""Register MFA capability with APG composition engine"""
		try:
			# Get APG composition engine if available
			composition_engine = getattr(app, 'apg_composition', None)
			
			if composition_engine:
				# Register capability metadata
				capability_metadata = {
					'id': 'mfa',
					'name': 'Multi-Factor Authentication',
					'version': '1.0.0',
					'type': 'security',
					'description': 'Enterprise-grade multi-factor authentication capability',
					'dependencies': [
						'auth_rbac',
						'notification', 
						'audit_compliance',
						'ai_orchestration',
						'computer_vision'
					],
					'provides': [
						'authentication',
						'biometric_authentication',
						'risk_assessment',
						'account_recovery',
						'security_analytics'
					],
					'endpoints': {
						'api': '/api/mfa',
						'health': '/mfa/health',
						'dashboard': '/mfadashboardview/dashboard/'
					},
					'events': {
						'publishes': [
							'mfa.authentication.success',
							'mfa.authentication.failure',
							'mfa.method.enrolled',
							'mfa.method.removed',
							'mfa.security.alert',
							'mfa.recovery.initiated'
						],
						'subscribes': [
							'auth.user.login',
							'auth.user.logout',
							'system.security.threat_detected'
						]
					}
				}
				
				composition_engine.register_capability('mfa', capability_metadata, self.mfa_service)
				self.logger.info(_log_blueprint_operation("composition_registered", "APG composition engine"))
			else:
				self.logger.warning(_log_blueprint_operation("composition_unavailable", "APG composition engine not found"))
			
		except Exception as e:
			self.logger.error(f"APG composition registration error: {str(e)}", exc_info=True)
	
	def get_service(self) -> Optional[MFAService]:
		"""Get MFA service instance"""
		return self.mfa_service
	
	def get_integration_router(self) -> Optional[APGIntegrationRouter]:
		"""Get APG integration router"""
		return self.integration_router


# Factory function for creating MFA blueprint
def create_mfa_blueprint(**kwargs) -> MFABlueprint:
	"""
	Factory function to create MFA blueprint with configuration.
	
	Args:
		**kwargs: Configuration parameters
	
	Returns:
		Configured MFA blueprint instance
	"""
	blueprint = MFABlueprint(kwargs.get('name', 'mfa'))
	return blueprint


# APG Capability Registration Helper
def register_mfa_capability(app, appbuilder: AppBuilder, **config):
	"""
	Convenience function to register MFA capability with APG platform.
	
	Args:
		app: Flask application instance
		appbuilder: Flask-AppBuilder instance
		**config: Configuration parameters
	
	Returns:
		Initialized MFA blueprint
	"""
	try:
		# Create and initialize MFA blueprint
		mfa_blueprint = create_mfa_blueprint()
		mfa_blueprint.init_app(app, appbuilder, **config)
		
		logging.info("MFA capability registered successfully with APG platform")
		return mfa_blueprint
		
	except Exception as e:
		logging.error(f"Failed to register MFA capability: {str(e)}", exc_info=True)
		raise


# Context processor for templates
def mfa_context_processor():
	"""Provide MFA context to templates"""
	return {
		'mfa_service': current_app.extensions.get('mfa', {}).get('service'),
		'mfa_enabled': True
	}


__all__ = [
	'MFABlueprint',
	'create_mfa_blueprint', 
	'register_mfa_capability',
	'mfa_context_processor'
]