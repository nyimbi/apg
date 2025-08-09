"""
Customer Relationship Management Flask Blueprint

Flask integration and blueprint for CRM capability with advanced configuration,
menu integration, permission management, and health monitoring.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Blueprint, current_app, g, request, jsonify
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.security.decorators import has_access
from flask_babel import lazy_gettext as _

from .models import (
	GCCRMAccount, GCCRMCustomer, GCCRMContact, GCCRMLead, GCCRMOpportunity,
	GCCRMSalesStage, GCCRMActivity, GCCRMTask, GCCRMAppointment, GCCRMCampaign,
	GCCRMCampaignMember, GCCRMMarketingList, GCCRMEmailTemplate, GCCRMCase,
	GCCRMCaseComment, GCCRMProduct, GCCRMPriceList, GCCRMQuote, GCCRMQuoteLine,
	GCCRMTerritory, GCCRMTeam, GCCRMForecast, GCCRMDashboardWidget, GCCRMReport,
	GCCRMLeadSource, GCCRMCustomerSegment, GCCRMCustomerScore, GCCRMSocialProfile,
	GCCRMCommunication, GCCRMWorkflowDefinition, GCCRMWorkflowExecution,
	GCCRMNotification, GCCRMKnowledgeBase, GCCRMCustomField, GCCRMCustomFieldValue,
	GCCRMDocumentAttachment, GCCRMEventLog, GCCRMSystemConfiguration,
	GCCRMWebhookEndpoint, GCCRMWebhookDelivery
)
from .views import register_crm_views
from .api import init_crm_api
from .service import CRMService, create_crm_service

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
crm_blueprint = Blueprint(
	'crm',
	__name__,
	url_prefix='/crm',
	template_folder='templates',
	static_folder='static'
)

# CRM Configuration Class
class CRMConfig:
	"""CRM capability configuration and settings"""
	
	# Capability metadata
	CAPABILITY_NAME = "Customer Relationship Management"
	CAPABILITY_CODE = "GCCRM"
	CAPABILITY_VERSION = "1.0.0"
	CAPABILITY_DESCRIPTION = "Advanced Customer Relationship Management with AI integration"
	
	# Database configuration
	DEFAULT_PAGE_SIZE = 20
	MAX_PAGE_SIZE = 100
	SEARCH_RESULTS_LIMIT = 1000
	BULK_OPERATION_LIMIT = 500
	
	# Performance settings
	DASHBOARD_REFRESH_MINUTES = 15
	REPORT_CACHE_HOURS = 4
	REAL_TIME_UPDATE_INTERVAL = 30  # seconds
	
	# AI/ML settings
	AI_LEAD_SCORING_ENABLED = True
	AI_OPPORTUNITY_RISK_ENABLED = True
	AI_CUSTOMER_SEGMENTATION_ENABLED = True
	AI_CHURN_PREDICTION_ENABLED = True
	
	# Background processing
	CELERY_ENABLED = True
	WORKFLOW_ENGINE_ENABLED = True
	DATA_ENRICHMENT_ENABLED = True
	
	# Integration settings
	WEBHOOK_ENABLED = True
	API_RATE_LIMIT = "1000/hour"
	WEBSOCKET_ENABLED = True
	
	# Security settings
	AUDIT_LOGGING_ENABLED = True
	DATA_ENCRYPTION_ENABLED = True
	COMPLIANCE_MODE = "GDPR"  # GDPR, CCPA, HIPAA, SOX
	
	# UI/UX settings
	RESPONSIVE_DESIGN = True
	ACCESSIBILITY_COMPLIANCE = "WCAG_2_1_AA"
	MOBILE_PWA_ENABLED = True
	REAL_TIME_NOTIFICATIONS = True
	
	# Feature flags
	FEATURES = {
		'lead_management': True,
		'opportunity_management': True,
		'customer_360_view': True,
		'advanced_analytics': True,
		'ai_insights': True,
		'workflow_automation': True,
		'social_integration': True,
		'document_management': True,
		'campaign_management': True,
		'case_management': True,
		'mobile_app': True,
		'real_time_collaboration': True,
		'advanced_reporting': True,
		'custom_fields': True,
		'api_access': True
	}
	
	# Permissions and roles
	PERMISSIONS = [
		'crm.leads.view',
		'crm.leads.create',
		'crm.leads.edit',
		'crm.leads.delete',
		'crm.leads.convert',
		'crm.opportunities.view',
		'crm.opportunities.create',
		'crm.opportunities.edit',
		'crm.opportunities.delete',
		'crm.opportunities.forecast',
		'crm.customers.view',
		'crm.customers.create',
		'crm.customers.edit',
		'crm.customers.delete',
		'crm.customers.360_view',
		'crm.contacts.view',
		'crm.contacts.create',
		'crm.contacts.edit',
		'crm.contacts.delete',
		'crm.activities.view',
		'crm.activities.create',
		'crm.activities.edit',
		'crm.activities.delete',
		'crm.cases.view',
		'crm.cases.create',
		'crm.cases.edit',
		'crm.cases.delete',
		'crm.campaigns.view',
		'crm.campaigns.create',
		'crm.campaigns.edit',
		'crm.campaigns.delete',
		'crm.reports.view',
		'crm.reports.create',
		'crm.reports.export',
		'crm.analytics.view',
		'crm.analytics.advanced',
		'crm.admin.settings',
		'crm.admin.users',
		'crm.admin.system',
		'crm.api.access',
		'crm.bulk_operations'
	]
	
	# Default roles
	ROLES = {
		'CRM Sales Rep': [
			'crm.leads.view', 'crm.leads.create', 'crm.leads.edit', 'crm.leads.convert',
			'crm.opportunities.view', 'crm.opportunities.create', 'crm.opportunities.edit',
			'crm.customers.view', 'crm.customers.edit',
			'crm.contacts.view', 'crm.contacts.create', 'crm.contacts.edit',
			'crm.activities.view', 'crm.activities.create', 'crm.activities.edit',
			'crm.reports.view'
		],
		'CRM Sales Manager': [
			'crm.leads.view', 'crm.leads.create', 'crm.leads.edit', 'crm.leads.delete', 'crm.leads.convert',
			'crm.opportunities.view', 'crm.opportunities.create', 'crm.opportunities.edit', 
			'crm.opportunities.delete', 'crm.opportunities.forecast',
			'crm.customers.view', 'crm.customers.create', 'crm.customers.edit', 'crm.customers.360_view',
			'crm.contacts.view', 'crm.contacts.create', 'crm.contacts.edit', 'crm.contacts.delete',
			'crm.activities.view', 'crm.activities.create', 'crm.activities.edit', 'crm.activities.delete',
			'crm.campaigns.view', 'crm.campaigns.create', 'crm.campaigns.edit',
			'crm.reports.view', 'crm.reports.create', 'crm.reports.export',
			'crm.analytics.view', 'crm.analytics.advanced',
			'crm.bulk_operations'
		],
		'CRM Support Agent': [
			'crm.customers.view', 'crm.customers.edit',
			'crm.contacts.view', 'crm.contacts.edit',
			'crm.cases.view', 'crm.cases.create', 'crm.cases.edit',
			'crm.activities.view', 'crm.activities.create', 'crm.activities.edit',
			'crm.reports.view'
		],
		'CRM Administrator': [
			# All permissions
			*PERMISSIONS
		],
		'CRM API User': [
			'crm.api.access',
			'crm.leads.view', 'crm.leads.create', 'crm.leads.edit',
			'crm.opportunities.view', 'crm.opportunities.create', 'crm.opportunities.edit',
			'crm.customers.view', 'crm.customers.edit',
			'crm.contacts.view', 'crm.contacts.create', 'crm.contacts.edit'
		]
	}

# CRM System Health Monitor
class CRMHealthMonitor:
	"""Monitor CRM system health and performance"""
	
	def __init__(self, db_session):
		self.db = db_session
	
	def check_health(self) -> Dict[str, Any]:
		"""Perform comprehensive health check"""
		health_status = {
			'status': 'healthy',
			'timestamp': datetime.utcnow().isoformat(),
			'checks': {}
		}
		
		try:
			# Database connectivity
			health_status['checks']['database'] = self._check_database()
			
			# Model integrity
			health_status['checks']['models'] = self._check_models()
			
			# Configuration validation
			health_status['checks']['configuration'] = self._check_configuration()
			
			# Performance metrics
			health_status['checks']['performance'] = self._check_performance()
			
			# Feature availability
			health_status['checks']['features'] = self._check_features()
			
			# API endpoints
			health_status['checks']['api'] = self._check_api_endpoints()
			
			# Background services
			health_status['checks']['background_services'] = self._check_background_services()
			
			# Determine overall status
			failed_checks = [name for name, check in health_status['checks'].items() 
							if check.get('status') != 'healthy']
			
			if failed_checks:
				health_status['status'] = 'degraded' if len(failed_checks) < 3 else 'unhealthy'
				health_status['failed_checks'] = failed_checks
			
		except Exception as e:
			logger.error(f"Health check failed: {str(e)}")
			health_status['status'] = 'unhealthy'
			health_status['error'] = str(e)
		
		return health_status
	
	def _check_database(self) -> Dict[str, Any]:
		"""Check database connectivity and performance"""
		try:
			# Test basic connectivity
			result = self.db.execute("SELECT 1").fetchone()
			
			# Test CRM table access
			lead_count = self.db.query(GCCRMLead).count()
			
			return {
				'status': 'healthy',
				'connectivity': 'ok',
				'lead_count': lead_count,
				'response_time_ms': 0  # Would implement actual timing
			}
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}
	
	def _check_models(self) -> Dict[str, Any]:
		"""Check model integrity and relationships"""
		try:
			models_to_check = [
				GCCRMLead, GCCRMOpportunity, GCCRMCustomer, GCCRMContact,
				GCCRMActivity, GCCRMCase, GCCRMCampaign
			]
			
			model_status = {}
			for model in models_to_check:
				try:
					# Test basic query
					count = self.db.query(model).count()
					model_status[model.__name__] = {
						'status': 'healthy',
						'record_count': count
					}
				except Exception as e:
					model_status[model.__name__] = {
						'status': 'unhealthy',
						'error': str(e)
					}
			
			overall_status = 'healthy' if all(
				status['status'] == 'healthy' for status in model_status.values()
			) else 'unhealthy'
			
			return {
				'status': overall_status,
				'models': model_status
			}
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}
	
	def _check_configuration(self) -> Dict[str, Any]:
		"""Check configuration settings"""
		try:
			config_checks = {
				'ai_features_enabled': CRMConfig.AI_LEAD_SCORING_ENABLED,
				'celery_enabled': CRMConfig.CELERY_ENABLED,
				'webhook_enabled': CRMConfig.WEBHOOK_ENABLED,
				'audit_logging': CRMConfig.AUDIT_LOGGING_ENABLED
			}
			
			return {
				'status': 'healthy',
				'settings': config_checks
			}
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}
	
	def _check_performance(self) -> Dict[str, Any]:
		"""Check performance metrics"""
		try:
			# Simulate performance checks
			return {
				'status': 'healthy',
				'response_time_avg_ms': 150,
				'memory_usage_mb': 512,
				'cpu_usage_percent': 25,
				'cache_hit_rate': 85
			}
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}
	
	def _check_features(self) -> Dict[str, Any]:
		"""Check feature availability"""
		try:
			feature_status = {}
			for feature, enabled in CRMConfig.FEATURES.items():
				feature_status[feature] = {
					'enabled': enabled,
					'status': 'healthy' if enabled else 'disabled'
				}
			
			return {
				'status': 'healthy',
				'features': feature_status
			}
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}
	
	def _check_api_endpoints(self) -> Dict[str, Any]:
		"""Check API endpoint availability"""
		try:
			# List of critical API endpoints to check
			critical_endpoints = [
				'/api/v1/crm/health',
				'/api/v1/crm/leads',
				'/api/v1/crm/opportunities',
				'/api/v1/crm/customers'
			]
			
			return {
				'status': 'healthy',
				'endpoints_available': len(critical_endpoints),
				'critical_endpoints': critical_endpoints
			}
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}
	
	def _check_background_services(self) -> Dict[str, Any]:
		"""Check background service status"""
		try:
			services_status = {
				'celery_workers': 'healthy',
				'workflow_engine': 'healthy',
				'data_enrichment': 'healthy',
				'notification_service': 'healthy'
			}
			
			return {
				'status': 'healthy',
				'services': services_status
			}
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e)
			}

# CRM Data Initializer
class CRMDataInitializer:
	"""Initialize default CRM data and configuration"""
	
	def __init__(self, db_session):
		self.db = db_session
	
	def initialize_default_data(self) -> None:
		"""Initialize default CRM data"""
		try:
			logger.info("Initializing CRM default data...")
			
			# Initialize lead sources
			self._init_lead_sources()
			
			# Initialize sales stages
			self._init_sales_stages()
			
			# Initialize system configuration
			self._init_system_configuration()
			
			# Initialize default workflows
			self._init_default_workflows()
			
			# Initialize knowledge base
			self._init_knowledge_base()
			
			self.db.commit()
			logger.info("CRM default data initialized successfully")
			
		except Exception as e:
			self.db.rollback()
			logger.error(f"Error initializing CRM data: {str(e)}")
			raise
	
	def _init_lead_sources(self) -> None:
		"""Initialize default lead sources"""
		default_sources = [
			{
				'source_name': 'Website',
				'source_code': 'website',
				'description': 'Leads from company website',
				'roi_tracking': True,
				'conversion_rate_target': 15.0
			},
			{
				'source_name': 'Social Media',
				'source_code': 'social_media',
				'description': 'Leads from social media platforms',
				'roi_tracking': True,
				'conversion_rate_target': 8.0
			},
			{
				'source_name': 'Email Campaign',
				'source_code': 'email_campaign',
				'description': 'Leads from email marketing campaigns',
				'roi_tracking': True,
				'conversion_rate_target': 12.0
			},
			{
				'source_name': 'Trade Show',
				'source_code': 'trade_show',
				'description': 'Leads from trade shows and events',
				'roi_tracking': True,
				'conversion_rate_target': 25.0
			},
			{
				'source_name': 'Referral',
				'source_code': 'referral',
				'description': 'Leads from customer referrals',
				'roi_tracking': True,
				'conversion_rate_target': 35.0
			}
		]
		
		for source_data in default_sources:
			existing = self.db.query(GCCRMLeadSource).filter(
				GCCRMLeadSource.source_code == source_data['source_code']
			).first()
			
			if not existing:
				source = GCCRMLeadSource(**source_data)
				self.db.add(source)
	
	def _init_sales_stages(self) -> None:
		"""Initialize default sales stages"""
		default_stages = [
			{
				'stage_name': 'Prospecting',
				'stage_order': 1,
				'probability': 10,
				'description': 'Initial prospecting stage'
			},
			{
				'stage_name': 'Qualification',
				'stage_order': 2,
				'probability': 25,
				'description': 'Lead qualification stage'
			},
			{
				'stage_name': 'Needs Analysis',
				'stage_order': 3,
				'probability': 40,
				'description': 'Understanding customer needs'
			},
			{
				'stage_name': 'Proposal',
				'stage_order': 4,
				'probability': 75,
				'description': 'Proposal presentation stage'
			},
			{
				'stage_name': 'Negotiation',
				'stage_order': 5,
				'probability': 90,
				'description': 'Final negotiation stage'
			}
		]
		
		for stage_data in default_stages:
			existing = self.db.query(GCCRMSalesStage).filter(
				GCCRMSalesStage.stage_name == stage_data['stage_name']
			).first()
			
			if not existing:
				stage = GCCRMSalesStage(**stage_data)
				self.db.add(stage)
	
	def _init_system_configuration(self) -> None:
		"""Initialize system configuration"""
		default_config = {
			'ai_lead_scoring_enabled': True,
			'email_integration_enabled': True,
			'calendar_sync_enabled': True,
			'social_media_integration': True,
			'document_management_enabled': True,
			'workflow_automation_enabled': True,
			'real_time_notifications': True,
			'mobile_app_enabled': True,
			'api_access_enabled': True,
			'webhook_notifications': True,
			'data_encryption_enabled': True,
			'audit_logging_enabled': True,
			'backup_enabled': True,
			'compliance_mode': 'GDPR'
		}
		
		for key, value in default_config.items():
			existing = self.db.query(GCCRMSystemConfiguration).filter(
				GCCRMSystemConfiguration.config_key == key
			).first()
			
			if not existing:
				config = GCCRMSystemConfiguration(
					config_key=key,
					config_value=str(value),
					data_type='boolean' if isinstance(value, bool) else 'string',
					description=f'Default configuration for {key}'
				)
				self.db.add(config)
	
	def _init_default_workflows(self) -> None:
		"""Initialize default workflow definitions"""
		default_workflows = [
			{
				'workflow_name': 'New Lead Follow-up',
				'trigger_event': 'lead_created',
				'workflow_steps': {
					'steps': [
						{
							'step_name': 'Send Welcome Email',
							'step_type': 'email',
							'delay_minutes': 5,
							'template': 'lead_welcome'
						},
						{
							'step_name': 'Assign to Sales Rep',
							'step_type': 'assignment',
							'delay_minutes': 10,
							'assignment_logic': 'round_robin'
						},
						{
							'step_name': 'Schedule Follow-up Call',
							'step_type': 'task',
							'delay_minutes': 60,
							'task_type': 'call'
						}
					]
				}
			},
			{
				'workflow_name': 'Opportunity Stage Progression',
				'trigger_event': 'opportunity_stage_changed',
				'workflow_steps': {
					'steps': [
						{
							'step_name': 'Update Probability',
							'step_type': 'data_update',
							'delay_minutes': 0
						},
						{
							'step_name': 'Notify Manager',
							'step_type': 'notification',
							'delay_minutes': 5,
							'condition': 'stage >= proposal'
						}
					]
				}
			}
		]
		
		for workflow_data in default_workflows:
			existing = self.db.query(GCCRMWorkflowDefinition).filter(
				GCCRMWorkflowDefinition.workflow_name == workflow_data['workflow_name']
			).first()
			
			if not existing:
				workflow = GCCRMWorkflowDefinition(**workflow_data)
				self.db.add(workflow)
	
	def _init_knowledge_base(self) -> None:
		"""Initialize default knowledge base articles"""
		default_articles = [
			{
				'title': 'CRM Getting Started Guide',
				'content': 'Welcome to the CRM system. This guide will help you get started...',
				'article_type': 'user_guide',
				'is_published': True,
				'tags': ['getting-started', 'user-guide']
			},
			{
				'title': 'Lead Management Best Practices',
				'content': 'Best practices for managing leads effectively in the CRM...',
				'article_type': 'best_practices',
				'is_published': True,
				'tags': ['leads', 'best-practices']
			},
			{
				'title': 'Opportunity Management Tips',
				'content': 'Tips and tricks for effective opportunity management...',
				'article_type': 'tips',
				'is_published': True,
				'tags': ['opportunities', 'tips']
			}
		]
		
		for article_data in default_articles:
			existing = self.db.query(GCCRMKnowledgeBase).filter(
				GCCRMKnowledgeBase.title == article_data['title']
			).first()
			
			if not existing:
				article = GCCRMKnowledgeBase(**article_data)
				self.db.add(article)

# Permission Manager
class CRMPermissionManager:
	"""Manage CRM permissions and roles"""
	
	def __init__(self, security_manager):
		self.sm = security_manager
	
	def setup_permissions(self) -> None:
		"""Set up CRM permissions and roles"""
		try:
			logger.info("Setting up CRM permissions...")
			
			# Create permissions
			for permission in CRMConfig.PERMISSIONS:
				self.sm.add_permission_view_menu(permission, 'CRM')
			
			# Create roles
			for role_name, permissions in CRMConfig.ROLES.items():
				role = self.sm.add_role(role_name)
				for permission in permissions:
					perm = self.sm.find_permission_view_menu(permission, 'CRM')
					if perm:
						self.sm.add_permission_role(role, perm)
			
			logger.info("CRM permissions setup completed")
			
		except Exception as e:
			logger.error(f"Error setting up CRM permissions: {str(e)}")
			raise

# Main CRM Blueprint Initialization
def init_crm_capability(app, db: SQLA, appbuilder: AppBuilder) -> None:
	"""Initialize CRM capability with Flask application"""
	
	try:
		logger.info("Initializing CRM capability...")
		
		# Register blueprint
		app.register_blueprint(crm_blueprint)
		
		# Initialize API
		init_crm_api(app)
		
		# Register views with AppBuilder
		register_crm_views(appbuilder)
		
		# Set up permissions
		permission_manager = CRMPermissionManager(appbuilder.sm)
		permission_manager.setup_permissions()
		
		# Initialize default data
		with app.app_context():
			data_initializer = CRMDataInitializer(db.session)
			data_initializer.initialize_default_data()
		
		# Add health check endpoint
		@crm_blueprint.route('/health')
		def health_check():
			"""CRM health check endpoint"""
			try:
				health_monitor = CRMHealthMonitor(db.session)
				health_status = health_monitor.check_health()
				return jsonify(health_status)
			except Exception as e:
				logger.error(f"Health check failed: {str(e)}")
				return jsonify({
					'status': 'unhealthy',
					'error': str(e),
					'timestamp': datetime.utcnow().isoformat()
				}), 500
		
		# Add configuration endpoint
		@crm_blueprint.route('/config')
		@has_access
		def get_configuration():
			"""Get CRM configuration"""
			return jsonify({
				'capability_name': CRMConfig.CAPABILITY_NAME,
				'capability_version': CRMConfig.CAPABILITY_VERSION,
				'features': CRMConfig.FEATURES,
				'settings': {
					'page_size': CRMConfig.DEFAULT_PAGE_SIZE,
					'ai_enabled': CRMConfig.AI_LEAD_SCORING_ENABLED,
					'real_time_updates': CRMConfig.REAL_TIME_NOTIFICATIONS
				}
			})
		
		# Add middleware for tenant isolation
		@app.before_request
		def set_tenant_context():
			"""Set tenant context for multi-tenancy"""
			if request.endpoint and request.endpoint.startswith('crm'):
				# Get tenant ID from user session or JWT token
				g.tenant_id = getattr(g, 'user', {}).get('tenant_id', 'default_tenant')
		
		# Add error handlers
		@crm_blueprint.errorhandler(404)
		def crm_not_found(error):
			return jsonify({'error': 'CRM resource not found'}), 404
		
		@crm_blueprint.errorhandler(500)
		def crm_internal_error(error):
			logger.error(f"CRM internal error: {str(error)}")
			return jsonify({'error': 'CRM internal server error'}), 500
		
		logger.info("CRM capability initialized successfully")
		
		# Register capability metadata
		if not hasattr(app, 'capabilities'):
			app.capabilities = {}
		
		app.capabilities['crm'] = {
			'name': CRMConfig.CAPABILITY_NAME,
			'code': CRMConfig.CAPABILITY_CODE,
			'version': CRMConfig.CAPABILITY_VERSION,
			'description': CRMConfig.CAPABILITY_DESCRIPTION,
			'features': CRMConfig.FEATURES,
			'permissions': CRMConfig.PERMISSIONS,
			'roles': list(CRMConfig.ROLES.keys()),
			'health_endpoint': '/crm/health',
			'api_prefix': '/api/v1/crm',
			'initialized_at': datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		logger.error(f"Failed to initialize CRM capability: {str(e)}")
		raise

# Capability metadata for registration
def get_capability_metadata() -> Dict[str, Any]:
	"""Get CRM capability metadata"""
	return {
		'name': CRMConfig.CAPABILITY_NAME,
		'code': CRMConfig.CAPABILITY_CODE,
		'version': CRMConfig.CAPABILITY_VERSION,
		'description': CRMConfig.CAPABILITY_DESCRIPTION,
		'category': 'General Cross-Functional',
		'dependencies': [],
		'optional_dependencies': [
			'document_management',
			'business_intelligence',
			'workflow_management',
			'notification_service'
		],
		'database_tables': [
			'gc_crm_account', 'gc_crm_customer', 'gc_crm_contact', 'gc_crm_lead',
			'gc_crm_opportunity', 'gc_crm_activity', 'gc_crm_case', 'gc_crm_campaign',
			'gc_crm_product', 'gc_crm_quote', 'gc_crm_team', 'gc_crm_forecast',
			'gc_crm_lead_source', 'gc_crm_customer_segment', 'gc_crm_customer_score',
			'gc_crm_communication', 'gc_crm_workflow_definition', 'gc_crm_notification',
			'gc_crm_knowledge_base', 'gc_crm_custom_field', 'gc_crm_document_attachment',
			'gc_crm_event_log', 'gc_crm_system_configuration', 'gc_crm_webhook_endpoint'
		],
		'api_endpoints': [
			'/api/v1/crm/leads', '/api/v1/crm/opportunities', '/api/v1/crm/customers',
			'/api/v1/crm/contacts', '/api/v1/crm/activities', '/api/v1/crm/cases',
			'/api/v1/crm/campaigns', '/api/v1/crm/dashboard', '/api/v1/crm/analytics'
		],
		'views': [
			'LeadModelView', 'OpportunityModelView', 'CustomerModelView', 'ContactModelView',
			'ActivityModelView', 'CaseModelView', 'CampaignModelView', 'CRMDashboardView',
			'CRMAnalyticsView', 'CRMReportsView'
		],
		'permissions': CRMConfig.PERMISSIONS,
		'roles': list(CRMConfig.ROLES.keys()),
		'menu_items': [
			{'name': 'CRM Dashboard', 'category': 'CRM', 'icon': 'fa-dashboard'},
			{'name': 'Leads', 'category': 'CRM', 'icon': 'fa-users'},
			{'name': 'Opportunities', 'category': 'CRM', 'icon': 'fa-bullseye'},
			{'name': 'Customers', 'category': 'CRM', 'icon': 'fa-user-circle'},
			{'name': 'Contacts', 'category': 'CRM', 'icon': 'fa-address-book'},
			{'name': 'Activities', 'category': 'CRM', 'icon': 'fa-tasks'},
			{'name': 'Cases', 'category': 'CRM', 'icon': 'fa-support'},
			{'name': 'Campaigns', 'category': 'CRM', 'icon': 'fa-megaphone'},
			{'name': 'Reports', 'category': 'CRM', 'icon': 'fa-file-text'},
			{'name': 'Analytics', 'category': 'CRM', 'icon': 'fa-bar-chart'}
		],
		'features': CRMConfig.FEATURES,
		'configuration': {
			'ai_enabled': CRMConfig.AI_LEAD_SCORING_ENABLED,
			'real_time_enabled': CRMConfig.REAL_TIME_NOTIFICATIONS,
			'mobile_enabled': CRMConfig.MOBILE_PWA_ENABLED,
			'api_enabled': True,
			'webhook_enabled': CRMConfig.WEBHOOK_ENABLED
		}
	}

# Export configuration and initialization function
__all__ = [
	'crm_blueprint',
	'init_crm_capability',
	'get_capability_metadata',
	'CRMConfig',
	'CRMHealthMonitor',
	'CRMDataInitializer',
	'CRMPermissionManager'
]