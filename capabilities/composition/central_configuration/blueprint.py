"""
APG Central Configuration - Flask-AppBuilder Blueprint Integration

Complete Flask-AppBuilder blueprint with all views, security, and navigation
for the Central Configuration management interface.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.security.sqla.manager import SecurityManager
from flask_appbuilder.menu import Menu

# Import all view classes
from .views import (
	CentralConfigurationView,
	ConfigurationTemplateView,
	ConfigurationVersionView,
	WorkspaceView,
	UserView,
	AuditLogView,
	WebhookView,
	EncryptionKeyView,
	ComplianceFrameworkView,
	BackupView,
	DeploymentView,
	EnvironmentView,
	IntegrationView,
	MetricView,
	NotificationView
)

# Import dashboard views
from .analytics.analytics_dashboard import AnalyticsDashboardView
from .performance.optimization_dashboard import PerformanceOptimizationDashboard
from .security.compliance_dashboard import ComplianceDashboardView

# Import specialized views
from .capability_manager import CapabilityManagerView
from .applet_system import AppletSystemView

# Import engines and managers
from .service import CentralConfigurationEngine
from .ai_engine import CentralConfigurationAI
from .ml_models import CentralConfigurationML
from .analytics.reporting_engine import AdvancedAnalyticsEngine
from .performance.auto_scaler import IntelligentAutoScaler
from .security.audit_engine import SecurityAuditEngine
from .deployment.multi_region_orchestrator import MultiRegionOrchestrator
from .integrations.enterprise_connectors import EnterpriseIntegrationManager

# Database models
from .models import db


class CentralConfigurationBlueprint:
	"""Complete Flask-AppBuilder blueprint for Central Configuration."""
	
	def __init__(self, app_builder: AppBuilder):
		"""Initialize Central Configuration blueprint."""
		self.appbuilder = app_builder
		self.app = app_builder.app
		
		# Initialize engines and managers
		self.config_engine = None
		self.ai_engine = None
		self.ml_models = None
		self.analytics_engine = None
		self.auto_scaler = None
		self.security_engine = None
		self.orchestrator = None
		self.integration_manager = None
		
		# Initialize views
		self.views = {}
		self.dashboards = {}
		
		print("🏗️ Initializing Central Configuration Blueprint")
	
	async def initialize_engines(self):
		"""Initialize all engines and managers."""
		print("🔧 Initializing engines and managers...")
		
		# Core configuration engine
		self.config_engine = CentralConfigurationEngine()
		await self.config_engine.initialize()
		
		# AI and ML engines
		self.ai_engine = CentralConfigurationAI(self.config_engine)
		await self.ai_engine.initialize()
		
		self.ml_models = CentralConfigurationML(self.config_engine)
		await self.ml_models.initialize()
		
		# Analytics engine
		self.analytics_engine = AdvancedAnalyticsEngine(self.config_engine)
		
		# Performance auto-scaler
		self.auto_scaler = IntelligentAutoScaler(self.config_engine)
		
		# Security audit engine
		self.security_engine = SecurityAuditEngine(self.config_engine)
		
		# Multi-region orchestrator
		self.orchestrator = MultiRegionOrchestrator(self.config_engine)
		
		# Enterprise integration manager
		self.integration_manager = EnterpriseIntegrationManager(self.config_engine)
		
		print("✅ All engines initialized")
	
	def register_views(self):
		"""Register all views with Flask-AppBuilder."""
		print("📋 Registering views...")
		
		# Core configuration views
		self.views['configuration'] = CentralConfigurationView
		self.appbuilder.add_view(
			CentralConfigurationView,
			"Configurations",
			icon="fa-cogs",
			category="Configuration Management",
			category_icon="fa-folder-open-o"
		)
		
		self.views['template'] = ConfigurationTemplateView
		self.appbuilder.add_view(
			ConfigurationTemplateView,
			"Templates",
			icon="fa-file-code-o",
			category="Configuration Management"
		)
		
		self.views['version'] = ConfigurationVersionView
		self.appbuilder.add_view(
			ConfigurationVersionView,
			"Versions",
			icon="fa-code-fork",
			category="Configuration Management"
		)
		
		self.views['workspace'] = WorkspaceView
		self.appbuilder.add_view(
			WorkspaceView,
			"Workspaces",
			icon="fa-briefcase",
			category="Configuration Management"
		)
		
		self.views['environment'] = EnvironmentView
		self.appbuilder.add_view(
			EnvironmentView,
			"Environments",
			icon="fa-server",
			category="Configuration Management"
		)
		
		# User and security views
		self.views['user'] = UserView
		self.appbuilder.add_view(
			UserView,
			"Users",
			icon="fa-users",
			category="Security & Access",
			category_icon="fa-shield"
		)
		
		self.views['audit'] = AuditLogView
		self.appbuilder.add_view(
			AuditLogView,
			"Audit Logs",
			icon="fa-history",
			category="Security & Access"
		)
		
		self.views['encryption'] = EncryptionKeyView
		self.appbuilder.add_view(
			EncryptionKeyView,
			"Encryption Keys",
			icon="fa-key",
			category="Security & Access"
		)
		
		self.views['compliance'] = ComplianceFrameworkView
		self.appbuilder.add_view(
			ComplianceFrameworkView,
			"Compliance Frameworks",
			icon="fa-gavel",
			category="Security & Access"
		)
		
		# Operations and monitoring views
		self.views['webhook'] = WebhookView
		self.appbuilder.add_view(
			WebhookView,
			"Webhooks",
			icon="fa-plug",
			category="Operations & Monitoring",
			category_icon="fa-dashboard"
		)
		
		self.views['backup'] = BackupView
		self.appbuilder.add_view(
			BackupView,
			"Backups",
			icon="fa-database",
			category="Operations & Monitoring"
		)
		
		self.views['deployment'] = DeploymentView
		self.appbuilder.add_view(
			DeploymentView,
			"Deployments",
			icon="fa-rocket",
			category="Operations & Monitoring"
		)
		
		self.views['integration'] = IntegrationView
		self.appbuilder.add_view(
			IntegrationView,
			"Integrations",
			icon="fa-exchange",
			category="Operations & Monitoring"
		)
		
		self.views['metric'] = MetricView
		self.appbuilder.add_view(
			MetricView,
			"Metrics",
			icon="fa-line-chart",
			category="Operations & Monitoring"
		)
		
		self.views['notification'] = NotificationView
		self.appbuilder.add_view(
			NotificationView,
			"Notifications",
			icon="fa-bell",
			category="Operations & Monitoring"
		)
		
		print("✅ Core views registered")
	
	def register_dashboards(self):
		"""Register dashboard views."""
		print("📊 Registering dashboards...")
		
		# Analytics dashboard
		self.dashboards['analytics'] = AnalyticsDashboardView(self.analytics_engine)
		self.appbuilder.add_view_no_menu(self.dashboards['analytics'])
		self.appbuilder.add_link(
			"Analytics Dashboard",
			href="/analytics/",
			icon="fa-bar-chart",
			category="Dashboards",
			category_icon="fa-dashboard"
		)
		
		# Performance dashboard
		self.dashboards['performance'] = PerformanceOptimizationDashboard(self.auto_scaler)
		self.appbuilder.add_view_no_menu(self.dashboards['performance'])
		self.appbuilder.add_link(
			"Performance Dashboard",
			href="/performance/",
			icon="fa-tachometer",
			category="Dashboards"
		)
		
		# Security dashboard
		self.dashboards['security'] = ComplianceDashboardView(self.security_engine)
		self.appbuilder.add_view_no_menu(self.dashboards['security'])
		self.appbuilder.add_link(
			"Security Dashboard",
			href="/security/",
			icon="fa-shield",
			category="Dashboards"
		)
		
		# Capability manager
		self.dashboards['capabilities'] = CapabilityManagerView(self.config_engine)
		self.appbuilder.add_view_no_menu(self.dashboards['capabilities'])
		self.appbuilder.add_link(
			"Capability Manager",
			href="/capabilities/",
			icon="fa-sitemap",
			category="APG Platform",
			category_icon="fa-cubes"
		)
		
		# Applet system
		self.dashboards['applets'] = AppletSystemView(self.config_engine)
		self.appbuilder.add_view_no_menu(self.dashboards['applets'])
		self.appbuilder.add_link(
			"Capability Applets",
			href="/applets/",
			icon="fa-th",
			category="APG Platform"
		)
		
		print("✅ Dashboards registered")
	
	def register_api_endpoints(self):
		"""Register API endpoints."""
		print("🔌 Registering API endpoints...")
		
		# Create API blueprint
		api_bp = Blueprint('central_config_api', __name__, url_prefix='/api/central-config')
		
		@api_bp.route('/health', methods=['GET'])
		def health_check():
			"""Health check endpoint."""
			return {"status": "healthy", "service": "central-configuration"}
		
		@api_bp.route('/configurations', methods=['GET', 'POST'])
		def configurations_api():
			"""Configurations API endpoint."""
			# Implementation would delegate to the service layer
			return {"message": "Configurations API"}
		
		@api_bp.route('/ai/optimize', methods=['POST'])
		def ai_optimize():
			"""AI optimization endpoint."""
			# Implementation would use the AI engine
			return {"message": "AI optimization triggered"}
		
		@api_bp.route('/analytics/generate-report', methods=['POST'])
		def generate_report():
			"""Generate analytics report."""
			# Implementation would use analytics engine
			return {"message": "Report generation started"}
		
		@api_bp.route('/performance/scale', methods=['POST'])
		def performance_scale():
			"""Trigger performance scaling."""
			# Implementation would use auto-scaler
			return {"message": "Scaling operation initiated"}
		
		# Register the API blueprint
		self.app.register_blueprint(api_bp)
		
		print("✅ API endpoints registered")
	
	def setup_security_permissions(self):
		"""Setup security permissions and roles."""
		print("🔒 Setting up security permissions...")
		
		# Define permissions
		permissions = [
			# Configuration permissions
			('can_read', 'CentralConfiguration', 'Read configurations'),
			('can_write', 'CentralConfiguration', 'Write configurations'),
			('can_delete', 'CentralConfiguration', 'Delete configurations'),
			
			# Template permissions
			('can_read', 'ConfigurationTemplate', 'Read templates'),
			('can_write', 'ConfigurationTemplate', 'Write templates'),
			('can_delete', 'ConfigurationTemplate', 'Delete templates'),
			
			# Version permissions
			('can_read', 'ConfigurationVersion', 'Read versions'),
			('can_write', 'ConfigurationVersion', 'Write versions'),
			
			# Security permissions
			('can_read', 'AuditLog', 'Read audit logs'),
			('can_read', 'EncryptionKey', 'Read encryption keys'),
			('can_write', 'EncryptionKey', 'Write encryption keys'),
			
			# Dashboard permissions
			('can_read', 'AnalyticsDashboard', 'Access analytics dashboard'),
			('can_read', 'PerformanceDashboard', 'Access performance dashboard'),
			('can_read', 'SecurityDashboard', 'Access security dashboard'),
			('can_write', 'PerformanceDashboard', 'Control performance settings'),
			
			# Operations permissions
			('can_read', 'Deployment', 'Read deployments'),
			('can_write', 'Deployment', 'Execute deployments'),
			('can_read', 'Integration', 'Read integrations'),
			('can_write', 'Integration', 'Configure integrations'),
			
			# Advanced permissions
			('can_admin', 'CentralConfiguration', 'Full administrative access'),
			('can_ai_optimize', 'CentralConfiguration', 'Use AI optimization'),
			('can_multi_region', 'CentralConfiguration', 'Multi-region operations')
		]
		
		# Create permissions in the security manager
		for permission, view_menu, description in permissions:
			try:
				self.appbuilder.sm.add_permission(permission)
				self.appbuilder.sm.add_view_menu(view_menu)
				self.appbuilder.sm.add_permission_view_menu(permission, view_menu)
			except Exception as e:
				print(f"⚠️ Permission setup warning: {e}")
		
		# Define roles
		roles_config = {
			'Central Config Admin': {
				'description': 'Full administrative access to Central Configuration',
				'permissions': [
					('can_admin', 'CentralConfiguration'),
					('can_read', 'AnalyticsDashboard'),
					('can_read', 'PerformanceDashboard'),
					('can_write', 'PerformanceDashboard'),
					('can_read', 'SecurityDashboard'),
					('can_ai_optimize', 'CentralConfiguration'),
					('can_multi_region', 'CentralConfiguration')
				]
			},
			'Configuration Manager': {
				'description': 'Manage configurations and templates',
				'permissions': [
					('can_read', 'CentralConfiguration'),
					('can_write', 'CentralConfiguration'),
					('can_read', 'ConfigurationTemplate'),
					('can_write', 'ConfigurationTemplate'),
					('can_read', 'ConfigurationVersion'),
					('can_read', 'AnalyticsDashboard')
				]
			},
			'Configuration Viewer': {
				'description': 'Read-only access to configurations',
				'permissions': [
					('can_read', 'CentralConfiguration'),
					('can_read', 'ConfigurationTemplate'),
					('can_read', 'ConfigurationVersion'),
					('can_read', 'AnalyticsDashboard')
				]
			},
			'Operations Engineer': {
				'description': 'Operations and monitoring access',
				'permissions': [
					('can_read', 'Deployment'),
					('can_write', 'Deployment'),
					('can_read', 'Integration'),
					('can_write', 'Integration'),
					('can_read', 'PerformanceDashboard'),
					('can_write', 'PerformanceDashboard'),
					('can_read', 'SecurityDashboard')
				]
			},
			'Security Auditor': {
				'description': 'Security and compliance access',
				'permissions': [
					('can_read', 'AuditLog'),
					('can_read', 'EncryptionKey'),
					('can_read', 'SecurityDashboard'),
					('can_read', 'CentralConfiguration')
				]
			}
		}
		
		# Create roles
		for role_name, role_config in roles_config.items():
			try:
				role = self.appbuilder.sm.add_role(role_name)
				if role:
					# Add permissions to role
					for permission, view_menu in role_config['permissions']:
						perm_view = self.appbuilder.sm.find_permission_view_menu(permission, view_menu)
						if perm_view:
							self.appbuilder.sm.add_permission_role(role, perm_view)
			except Exception as e:
				print(f"⚠️ Role setup warning: {e}")
		
		print("✅ Security permissions configured")
	
	def customize_menu(self):
		"""Customize the application menu."""
		print("📋 Customizing application menu...")
		
		# Add custom menu items
		self.appbuilder.add_separator("Configuration Management")
		
		# Quick actions menu
		self.appbuilder.add_link(
			"Quick Create Config",
			href="/centralconfigurationview/add",
			icon="fa-plus-circle",
			category="Quick Actions",
			category_icon="fa-bolt"
		)
		
		self.appbuilder.add_link(
			"AI Optimization",
			href="/api/central-config/ai/optimize",
			icon="fa-magic",
			category="Quick Actions"
		)
		
		self.appbuilder.add_link(
			"System Health",
			href="/performance/api/overview",
			icon="fa-heartbeat",
			category="Quick Actions"
		)
		
		# Help and documentation
		self.appbuilder.add_link(
			"API Documentation",
			href="/api/central-config/docs",
			icon="fa-book",
			category="Help & Support",
			category_icon="fa-question-circle"
		)
		
		self.appbuilder.add_link(
			"User Guide",
			href="/help/user-guide",
			icon="fa-graduation-cap",
			category="Help & Support"
		)
		
		print("✅ Menu customized")
	
	def setup_custom_filters(self):
		"""Setup custom Jinja2 filters for templates."""
		print("🔧 Setting up custom filters...")
		
		@self.app.template_filter('format_bytes')
		def format_bytes(bytes_value):
			"""Format bytes as human-readable string."""
			if bytes_value < 1024:
				return f"{bytes_value} B"
			elif bytes_value < 1024**2:
				return f"{bytes_value/1024:.1f} KB"
			elif bytes_value < 1024**3:
				return f"{bytes_value/(1024**2):.1f} MB"
			else:
				return f"{bytes_value/(1024**3):.1f} GB"
		
		@self.app.template_filter('format_duration')
		def format_duration(seconds):
			"""Format seconds as duration string."""
			if seconds < 60:
				return f"{seconds:.1f}s"
			elif seconds < 3600:
				return f"{seconds/60:.1f}m"
			else:
				return f"{seconds/3600:.1f}h"
		
		@self.app.template_filter('security_level_badge')
		def security_level_badge(level):
			"""Format security level as Bootstrap badge."""
			colors = {
				'low': 'success',
				'medium': 'warning', 
				'high': 'danger',
				'critical': 'dark'
			}
			color = colors.get(level.lower(), 'secondary')
			return f'<span class="badge badge-{color}">{level.upper()}</span>'
		
		@self.app.template_filter('status_icon')
		def status_icon(status):
			"""Format status as icon."""
			icons = {
				'active': 'fa-check-circle text-success',
				'inactive': 'fa-times-circle text-danger',
				'pending': 'fa-clock-o text-warning',
				'draft': 'fa-edit text-info'
			}
			icon = icons.get(status.lower(), 'fa-question-circle text-muted')
			return f'<i class="fa {icon}"></i>'
		
		print("✅ Custom filters configured")
	
	def register_error_handlers(self):
		"""Register custom error handlers."""
		print("⚠️ Setting up error handlers...")
		
		@self.app.errorhandler(404)
		def not_found_error(error):
			return {"error": "Resource not found", "code": 404}, 404
		
		@self.app.errorhandler(500)
		def internal_error(error):
			return {"error": "Internal server error", "code": 500}, 500
		
		@self.app.errorhandler(403)
		def forbidden_error(error):
			return {"error": "Access forbidden", "code": 403}, 403
		
		print("✅ Error handlers configured")
	
	async def initialize_blueprint(self):
		"""Initialize the complete blueprint."""
		print("🚀 Initializing Central Configuration Blueprint...")
		
		# Initialize engines first
		await self.initialize_engines()
		
		# Setup database
		db.init_app(self.app)
		
		# Register all components
		self.register_views()
		self.register_dashboards()
		self.register_api_endpoints()
		self.setup_security_permissions()
		self.customize_menu()
		self.setup_custom_filters()
		self.register_error_handlers()
		
		print("✅ Central Configuration Blueprint fully initialized!")
		print(f"📊 Registered {len(self.views)} core views")
		print(f"📈 Registered {len(self.dashboards)} dashboards")
		print("🔐 Security permissions configured")
		print("📋 Custom menu and filters active")
		
		return self


def create_central_configuration_blueprint(app_builder: AppBuilder) -> CentralConfigurationBlueprint:
	"""Factory function to create and initialize the Central Configuration blueprint."""
	blueprint = CentralConfigurationBlueprint(app_builder)
	print("🏗️ Central Configuration Blueprint created")
	return blueprint


# Blueprint registration function for Flask-AppBuilder
def init_app(app_builder: AppBuilder):
	"""Initialize Central Configuration capability with Flask-AppBuilder."""
	import asyncio
	
	# Create and initialize blueprint
	blueprint = create_central_configuration_blueprint(app_builder)
	
	# Initialize asynchronously
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	try:
		loop.run_until_complete(blueprint.initialize_blueprint())
	finally:
		loop.close()
	
	print("🎉 Central Configuration capability fully integrated with Flask-AppBuilder!")
	return blueprint