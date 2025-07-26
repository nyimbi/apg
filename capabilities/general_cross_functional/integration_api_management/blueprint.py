"""
APG Integration API Management - Flask Blueprint

Flask-AppBuilder blueprint registration for all API management views,
including menu structure and security configuration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	# API Management Views
	APIManagementView,
	EndpointManagementView,
	PolicyManagementView,
	
	# Consumer Management Views
	ConsumerManagementView,
	APIKeyManagementView,
	
	# Analytics Views
	AnalyticsDashboardView,
	UsageRecordsView,
	
	# Developer Portal Views
	DeveloperPortalView,
	
	# Deployment Views
	DeploymentManagementView
)

# Create the blueprint
integration_api_management_bp = Blueprint(
	'integration_api_management',
	__name__,
	url_prefix='/iam',
	template_folder='templates',
	static_folder='static'
)

def register_views(appbuilder: AppBuilder):
	"""Register all Integration API Management views with Flask-AppBuilder."""
	
	# =============================================================================
	# API Management Menu Section
	# =============================================================================
	
	# Main API Management
	appbuilder.add_view(
		APIManagementView,
		"APIs",
		icon="fa-code",
		category="API Management",
		category_icon="fa-cogs"
	)
	
	# API Endpoints
	appbuilder.add_view(
		EndpointManagementView,
		"Endpoints",
		icon="fa-sitemap",
		category="API Management"
	)
	
	# API Policies
	appbuilder.add_view(
		PolicyManagementView,
		"Policies",
		icon="fa-shield",
		category="API Management"
	)
	
	# API Deployments
	appbuilder.add_view(
		DeploymentManagementView,
		"Deployments",
		icon="fa-rocket",
		category="API Management"
	)
	
	# =============================================================================
	# Consumer Management Menu Section
	# =============================================================================
	
	# API Consumers
	appbuilder.add_view(
		ConsumerManagementView,
		"Consumers",
		icon="fa-users",
		category="Consumer Management",
		category_icon="fa-user-circle"
	)
	
	# API Keys
	appbuilder.add_view(
		APIKeyManagementView,
		"API Keys",
		icon="fa-key",
		category="Consumer Management"
	)
	
	# =============================================================================
	# Analytics & Monitoring Menu Section
	# =============================================================================
	
	# Analytics Dashboard
	appbuilder.add_view_no_menu(AnalyticsDashboardView)
	appbuilder.add_link(
		"Dashboard",
		href="/analytics/dashboard/",
		icon="fa-dashboard",
		category="Analytics & Monitoring",
		category_icon="fa-bar-chart"
	)
	
	# API Usage Analytics
	appbuilder.add_link(
		"API Usage",
		href="/analytics/api-usage/",
		icon="fa-line-chart",
		category="Analytics & Monitoring"
	)
	
	# Performance Analytics
	appbuilder.add_link(
		"Performance",
		href="/analytics/performance/",
		icon="fa-tachometer",
		category="Analytics & Monitoring"
	)
	
	# Usage Records
	appbuilder.add_view(
		UsageRecordsView,
		"Usage Records",
		icon="fa-list",
		category="Analytics & Monitoring"
	)
	
	# =============================================================================
	# Developer Portal Menu Section
	# =============================================================================
	
	# Developer Portal
	appbuilder.add_view_no_menu(DeveloperPortalView)
	appbuilder.add_link(
		"Developer Portal",
		href="/developer/portal/",
		icon="fa-code",
		category="Developer Tools",
		category_icon="fa-laptop"
	)
	
	# API Documentation
	appbuilder.add_link(
		"API Documentation",
		href="/developer/portal/",
		icon="fa-book",
		category="Developer Tools"
	)
	
	# API Testing
	appbuilder.add_link(
		"API Testing",
		href="/developer/portal/",
		icon="fa-flask",
		category="Developer Tools"
	)

def register_permissions(appbuilder: AppBuilder):
	"""Register custom permissions for Integration API Management."""
	
	# Define permission categories
	permission_categories = [
		# API Management Permissions
		{
			'name': 'API_ADMIN',
			'description': 'Full API management access',
			'views': [
				'APIManagementView',
				'EndpointManagementView', 
				'PolicyManagementView',
				'DeploymentManagementView'
			],
			'permissions': ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
		},
		{
			'name': 'API_OPERATOR',
			'description': 'API operational access (no create/delete)',
			'views': [
				'APIManagementView',
				'EndpointManagementView',
				'PolicyManagementView',
				'DeploymentManagementView'
			],
			'permissions': ['can_list', 'can_show', 'can_edit']
		},
		{
			'name': 'API_VIEWER',
			'description': 'Read-only API access',
			'views': [
				'APIManagementView',
				'EndpointManagementView',
				'PolicyManagementView',
				'DeploymentManagementView'
			],
			'permissions': ['can_list', 'can_show']
		},
		
		# Consumer Management Permissions
		{
			'name': 'CONSUMER_ADMIN',
			'description': 'Full consumer management access',
			'views': [
				'ConsumerManagementView',
				'APIKeyManagementView'
			],
			'permissions': ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
		},
		{
			'name': 'CONSUMER_MANAGER',
			'description': 'Consumer management access (no delete)',
			'views': [
				'ConsumerManagementView',
				'APIKeyManagementView'
			],
			'permissions': ['can_list', 'can_show', 'can_add', 'can_edit']
		},
		
		# Analytics Permissions
		{
			'name': 'ANALYTICS_ADMIN',
			'description': 'Full analytics access',
			'views': [
				'AnalyticsDashboardView',
				'UsageRecordsView'
			],
			'permissions': ['can_list', 'can_show']
		},
		
		# Developer Portal Permissions
		{
			'name': 'DEVELOPER',
			'description': 'Developer portal access',
			'views': [
				'DeveloperPortalView'
			],
			'permissions': ['can_list', 'can_show']
		}
	]
	
	# Register permissions with security manager
	security_manager = appbuilder.sm
	
	for category in permission_categories:
		# Create role if it doesn't exist
		role = security_manager.find_role(category['name'])
		if not role:
			role = security_manager.add_role(category['name'])
			role.description = category['description']
			
			# Add permissions to role
			for view_name in category['views']:
				for permission in category['permissions']:
					perm = security_manager.find_permission_on_view(permission, view_name)
					if perm:
						security_manager.add_permission_role(role, perm)

def create_default_users(appbuilder: AppBuilder):
	"""Create default users for testing and initial setup."""
	
	security_manager = appbuilder.sm
	
	# Default users configuration
	default_users = [
		{
			'username': 'api_admin',
			'email': 'api.admin@company.com',
			'first_name': 'API',
			'last_name': 'Administrator',
			'role': 'API_ADMIN',
			'password': 'api_admin_2025!'
		},
		{
			'username': 'api_operator',
			'email': 'api.operator@company.com',
			'first_name': 'API',
			'last_name': 'Operator',
			'role': 'API_OPERATOR',
			'password': 'api_operator_2025!'
		},
		{
			'username': 'consumer_manager',
			'email': 'consumer.manager@company.com',
			'first_name': 'Consumer',
			'last_name': 'Manager',
			'role': 'CONSUMER_ADMIN',
			'password': 'consumer_mgr_2025!'
		},
		{
			'username': 'developer',
			'email': 'developer@company.com',
			'first_name': 'API',
			'last_name': 'Developer',
			'role': 'DEVELOPER',
			'password': 'developer_2025!'
		}
	]
	
	for user_config in default_users:
		# Check if user already exists
		user = security_manager.find_user(username=user_config['username'])
		if not user:
			# Find role
			role = security_manager.find_role(user_config['role'])
			if role:
				# Create user
				user = security_manager.add_user(
					username=user_config['username'],
					email=user_config['email'],
					first_name=user_config['first_name'],
					last_name=user_config['last_name'],
					role=role,
					password=user_config['password']
				)

def init_integration_api_management(appbuilder: AppBuilder):
	"""Initialize the Integration API Management capability."""
	
	# Register all views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Create default users (for development/testing)
	# Comment out in production
	# create_default_users(appbuilder)
	
	# Add custom CSS/JS resources
	appbuilder.app.static_folder = 'static'
	
	# Register custom Jinja filters for the views
	@appbuilder.app.template_filter('format_json')
	def format_json_filter(value):
		"""Format JSON for display."""
		if isinstance(value, (dict, list)):
			import json
			return json.dumps(value, indent=2)
		return value
	
	@appbuilder.app.template_filter('format_bytes')
	def format_bytes_filter(bytes_value):
		"""Format bytes into human-readable format."""
		if not bytes_value:
			return ''
		
		for unit in ['B', 'KB', 'MB', 'GB']:
			if bytes_value < 1024:
				return f'{bytes_value:.1f} {unit}'
			bytes_value /= 1024
		return f'{bytes_value:.1f} TB'
	
	@appbuilder.app.template_filter('time_ago')
	def time_ago_filter(datetime_value):
		"""Format datetime as time ago."""
		if not datetime_value:
			return ''
		
		from datetime import datetime, timezone
		
		if datetime_value.tzinfo is None:
			datetime_value = datetime_value.replace(tzinfo=timezone.utc)
		
		now = datetime.now(timezone.utc)
		diff = now - datetime_value
		
		if diff.days > 0:
			return f'{diff.days} days ago'
		elif diff.seconds > 3600:
			hours = diff.seconds // 3600
			return f'{hours} hours ago'
		elif diff.seconds > 60:
			minutes = diff.seconds // 60
			return f'{minutes} minutes ago'
		else:
			return 'Just now'

# =============================================================================
# Blueprint Factory Function
# =============================================================================

def create_integration_api_management_blueprint(appbuilder: AppBuilder) -> Blueprint:
	"""Create and configure the Integration API Management blueprint."""
	
	# Initialize the capability
	init_integration_api_management(appbuilder)
	
	# Add any blueprint-specific routes
	@integration_api_management_bp.route('/health')
	def health_check():
		"""Health check endpoint for the capability."""
		return {
			'status': 'healthy',
			'capability': 'integration_api_management',
			'version': '1.0.0',
			'timestamp': datetime.now(timezone.utc).isoformat()
		}
	
	@integration_api_management_bp.route('/info')
	def capability_info():
		"""Get capability information."""
		return {
			'capability_id': 'integration_api_management',
			'capability_name': 'Integration API Management',
			'version': '1.0.0',
			'description': 'Comprehensive API gateway and management platform',
			'features': [
				'API Gateway & Routing',
				'Security & Access Control', 
				'API Lifecycle Management',
				'Developer Experience Platform',
				'Analytics & Monitoring'
			],
			'endpoints': {
				'admin': '/iam/',
				'analytics': '/analytics/',
				'developer_portal': '/developer/',
				'api': '/api/v1/',
				'health': '/iam/health'
			}
		}
	
	return integration_api_management_bp

# Export for use in main application
__all__ = [
	'integration_api_management_bp',
	'create_integration_api_management_blueprint',
	'init_integration_api_management'
]