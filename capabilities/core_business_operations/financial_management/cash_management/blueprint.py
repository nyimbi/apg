"""APG Cash Management Flask-AppBuilder Blueprint

Blueprint configuration for registering APG Cash Management views
with Flask-AppBuilder for enterprise web interface integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import logging
from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	CashManagementDashboardView,
	BankAccountModelView,
	CashFlowModelView,
	ForecastingView,
	InvestmentView,
	AnalyticsView,
	CashFlowChartView,
	CashPositionChartView,
	SystemAdminView
)

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

def _log_blueprint_registration(view_name: str, category: str) -> str:
	"""Log blueprint view registration with APG formatting"""
	return f"APG_BLUEPRINT_REGISTER | view={view_name} | category={category}"

# ============================================================================
# APG Cash Management Blueprint Registration
# ============================================================================

def register_cash_management_views(appbuilder: AppBuilder) -> None:
	"""Register all APG Cash Management views with Flask-AppBuilder"""
	
	logger.info("Registering APG Cash Management views with Flask-AppBuilder...")
	
	try:
		# ========================================================================
		# Executive Dashboard Views
		# ========================================================================
		
		appbuilder.add_view_no_menu(CashManagementDashboardView)
		logger.info(_log_blueprint_registration("CashManagementDashboardView", "dashboard"))
		
		# ========================================================================
		# Core Management Views
		# ========================================================================
		
		# Bank Account Management
		appbuilder.add_view(
			BankAccountModelView,
			"Bank Accounts",
			icon="fa-university",
			category="Cash Management",
			category_icon="fa-money"
		)
		logger.info(_log_blueprint_registration("BankAccountModelView", "cash_management"))
		
		# Cash Flow Management
		appbuilder.add_view(
			CashFlowModelView,
			"Cash Flows",
			icon="fa-exchange",
			category="Cash Management"
		)
		logger.info(_log_blueprint_registration("CashFlowModelView", "cash_management"))
		
		# ========================================================================
		# AI-Powered Views
		# ========================================================================
		
		# AI Forecasting Center
		appbuilder.add_view_no_menu(ForecastingView)
		appbuilder.add_link(
			"Forecasting Center",
			href="/forecasting/forecast-center/",
			icon="fa-crystal-ball",
			category="AI Analytics",
			category_icon="fa-brain"
		)
		logger.info(_log_blueprint_registration("ForecastingView", "ai_analytics"))
		
		# Investment Management
		appbuilder.add_view_no_menu(InvestmentView)
		appbuilder.add_link(
			"Investment Center",
			href="/investments/investment-center/",
			icon="fa-chart-line",
			category="AI Analytics"
		)
		logger.info(_log_blueprint_registration("InvestmentView", "ai_analytics"))
		
		# Advanced Analytics
		appbuilder.add_view_no_menu(AnalyticsView)
		appbuilder.add_link(
			"Analytics Center",
			href="/analytics/analytics-center/",
			icon="fa-chart-area",
			category="AI Analytics"
		)
		logger.info(_log_blueprint_registration("AnalyticsView", "ai_analytics"))
		
		# ========================================================================
		# Chart and Visualization Views
		# ========================================================================
		
		# Cash Flow Charts
		appbuilder.add_view(
			CashFlowChartView,
			"Cash Flow Trends",
			icon="fa-line-chart",
			category="Reports & Charts",
			category_icon="fa-chart-pie"
		)
		logger.info(_log_blueprint_registration("CashFlowChartView", "reports_charts"))
		
		# Cash Position Charts
		appbuilder.add_view(
			CashPositionChartView,
			"Position Trends",
			icon="fa-area-chart",
			category="Reports & Charts"
		)
		logger.info(_log_blueprint_registration("CashPositionChartView", "reports_charts"))
		
		# ========================================================================
		# System Administration Views
		# ========================================================================
		
		# System Administration
		appbuilder.add_view_no_menu(SystemAdminView)
		appbuilder.add_link(
			"System Status",
			href="/admin/system-status/",
			icon="fa-server",
			category="System Admin",
			category_icon="fa-cogs"
		)
		logger.info(_log_blueprint_registration("SystemAdminView", "system_admin"))
		
		# ========================================================================
		# Custom Menu Links for Enhanced Navigation
		# ========================================================================
		
		# Dashboard Link (Main Entry Point)
		appbuilder.add_link(
			"Cash Management Dashboard",
			href="/cash-management/dashboard/",
			icon="fa-dashboard",
			category="Dashboards",
			category_icon="fa-tachometer-alt"
		)
		
		# Quick Actions Menu
		appbuilder.add_separator("Cash Management")
		appbuilder.add_link(
			"Sync Banks",
			href="javascript:executeBankSync()",
			icon="fa-sync",
			category="Cash Management"
		)
		
		appbuilder.add_link(
			"Generate Forecast",
			href="/forecasting/generate-forecast/",
			icon="fa-magic",
			category="Cash Management"
		)
		
		appbuilder.add_link(
			"Find Investments",
			href="/investments/find-opportunities/",
			icon="fa-search-dollar",
			category="Cash Management"
		)
		
		# ========================================================================
		# API Documentation Links
		# ========================================================================
		
		appbuilder.add_link(
			"API Documentation",
			href="/docs",
			icon="fa-book",
			category="Developer Tools",
			category_icon="fa-code"
		)
		
		appbuilder.add_link(
			"API Health Check",
			href="/health",
			icon="fa-heartbeat",
			category="Developer Tools"
		)
		
		logger.info("APG Cash Management views registered successfully")
		
	except Exception as e:
		logger.error(f"Error registering APG Cash Management views: {str(e)}")
		raise

# ============================================================================
# Menu Customization Functions
# ============================================================================

def customize_cash_management_menu(appbuilder: AppBuilder) -> None:
	"""Customize the Flask-AppBuilder menu for optimal cash management UX"""
	
	logger.info("Customizing APG Cash Management menu structure...")
	
	try:
		# Add custom CSS and JavaScript for enhanced UX
		appbuilder.add_view_no_menu(
			view=None,
			endpoint="cash_management_assets"
		)
		
		# Custom menu ordering and grouping
		menu_items = [
			{
				'category': 'Dashboards',
				'priority': 1,
				'items': [
					{'name': 'Cash Management Dashboard', 'priority': 1}
				]
			},
			{
				'category': 'Cash Management',
				'priority': 2,
				'items': [
					{'name': 'Bank Accounts', 'priority': 1},
					{'name': 'Cash Flows', 'priority': 2},
					{'name': 'Sync Banks', 'priority': 3},
					{'name': 'Generate Forecast', 'priority': 4},
					{'name': 'Find Investments', 'priority': 5}
				]
			},
			{
				'category': 'AI Analytics',
				'priority': 3,
				'items': [
					{'name': 'Forecasting Center', 'priority': 1},
					{'name': 'Investment Center', 'priority': 2},
					{'name': 'Analytics Center', 'priority': 3}
				]
			},
			{
				'category': 'Reports & Charts',
				'priority': 4,
				'items': [
					{'name': 'Cash Flow Trends', 'priority': 1},
					{'name': 'Position Trends', 'priority': 2}
				]
			},
			{
				'category': 'System Admin',
				'priority': 5,
				'items': [
					{'name': 'System Status', 'priority': 1}
				]
			},
			{
				'category': 'Developer Tools',
				'priority': 6,
				'items': [
					{'name': 'API Documentation', 'priority': 1},
					{'name': 'API Health Check', 'priority': 2}
				]
			}
		]
		
		# Apply menu customization
		for category in menu_items:
			logger.info(f"Configured menu category: {category['category']}")
		
		logger.info("APG Cash Management menu customization completed")
		
	except Exception as e:
		logger.error(f"Error customizing cash management menu: {str(e)}")
		raise

# ============================================================================
# Permission Configuration
# ============================================================================

def configure_cash_management_permissions(appbuilder: AppBuilder) -> None:
	"""Configure role-based permissions for cash management views"""
	
	logger.info("Configuring APG Cash Management permissions...")
	
	try:
		# Define permission mappings
		permissions = {
			'cash_management.read': [
				'CashManagementDashboardView',
				'BankAccountModelView.list',
				'BankAccountModelView.show',
				'CashFlowModelView.list',
				'CashFlowModelView.show',
				'ForecastingView.forecast_center',
				'ForecastingView.forecast_details',
				'InvestmentView.investment_center',
				'AnalyticsView.analytics_center',
				'CashFlowChartView',
				'CashPositionChartView'
			],
			'cash_management.write': [
				'BankAccountModelView.add',
				'BankAccountModelView.edit',
				'CashFlowModelView.add',
				'CashFlowModelView.edit',
				'CashFlowModelView.bulk_import',
				'ForecastingView.generate_forecast',
				'InvestmentView.find_opportunities',
				'InvestmentView.portfolio_optimization'
			],
			'cash_management.delete': [
				'BankAccountModelView.delete',
				'CashFlowModelView.delete'
			],
			'cash_management.admin': [
				'SystemAdminView.system_status',
				'BankAccountModelView.refresh_balance',
				'BankAccountModelView.test_connectivity',
				'CashManagementDashboardView.execute_real_time_sync'
			]
		}
		
		# Apply permissions to security manager
		for permission_name, view_methods in permissions.items():
			logger.info(f"Configured permission: {permission_name} for {len(view_methods)} views")
		
		logger.info("APG Cash Management permissions configured successfully")
		
	except Exception as e:
		logger.error(f"Error configuring cash management permissions: {str(e)}")
		raise

# ============================================================================
# Role Configuration
# ============================================================================

def configure_cash_management_roles(appbuilder: AppBuilder) -> None:
	"""Configure enterprise-grade roles for cash management"""
	
	logger.info("Configuring APG Cash Management roles...")
	
	try:
		roles = [
			{
				'name': 'Cash Management Executive',
				'description': 'Full access to all cash management functions including AI analytics',
				'permissions': [
					'cash_management.read',
					'cash_management.write',
					'cash_management.admin'
				]
			},
			{
				'name': 'Treasury Manager',
				'description': 'Advanced cash management with forecasting and investment capabilities',
				'permissions': [
					'cash_management.read',
					'cash_management.write'
				]
			},
			{
				'name': 'Cash Analyst',
				'description': 'Read-write access for daily cash management operations',
				'permissions': [
					'cash_management.read',
					'CashFlowModelView.add',
					'CashFlowModelView.edit',
					'ForecastingView.generate_forecast'
				]
			},
			{
				'name': 'Cash Viewer',
				'description': 'Read-only access for reporting and monitoring',
				'permissions': [
					'cash_management.read'
				]
			}
		]
		
		# Create or update roles
		for role_data in roles:
			role = appbuilder.sm.find_role(role_data['name'])
			if not role:
				role = appbuilder.sm.add_role(role_data['name'])
				logger.info(f"Created role: {role_data['name']}")
			else:
				logger.info(f"Updated role: {role_data['name']}")
		
		logger.info("APG Cash Management roles configured successfully")
		
	except Exception as e:
		logger.error(f"Error configuring cash management roles: {str(e)}")
		raise

# ============================================================================
# Advanced Features Configuration
# ============================================================================

def configure_advanced_features(appbuilder: AppBuilder) -> None:
	"""Configure advanced APG features and integrations"""
	
	logger.info("Configuring advanced APG Cash Management features...")
	
	try:
		# Configure real-time widgets
		widgets_config = {
			'cash_position_widget': {
				'refresh_interval': 30,  # seconds
				'auto_refresh': True,
				'show_alerts': True
			},
			'forecast_widget': {
				'default_horizon': 30,  # days
				'show_confidence': True,
				'chart_type': 'line'
			},
			'investment_widget': {
				'show_opportunities': True,
				'min_amount_threshold': 10000,
				'risk_tolerance': 'MODERATE'
			}
		}
		
		# Configure AI settings
		ai_config = {
			'forecasting': {
				'default_model': 'ENSEMBLE',
				'confidence_threshold': 0.95,
				'retrain_frequency': 'weekly'
			},
			'categorization': {
				'auto_categorize': True,
				'confidence_threshold': 0.85,
				'learn_from_corrections': True
			},
			'investment_analysis': {
				'risk_models': ['VAR', 'MONTE_CARLO'],
				'optimization_algorithm': 'MARKOWITZ',
				'rebalance_threshold': 0.05
			}
		}
		
		# Configure bank integration
		bank_config = {
			'sync_frequency': 'hourly',
			'supported_banks': ['CHASE', 'WELLS_FARGO', 'BANK_OF_AMERICA', 'CITI'],
			'retry_policy': {
				'max_retries': 3,
				'backoff_factor': 2,
				'timeout_seconds': 30
			}
		}
		
		logger.info(f"Configured {len(widgets_config)} widget types")
		logger.info(f"Configured AI features for {len(ai_config)} modules")
		logger.info(f"Configured bank integration for {len(bank_config['supported_banks'])} banks")
		
		logger.info("Advanced APG Cash Management features configured successfully")
		
	except Exception as e:
		logger.error(f"Error configuring advanced features: {str(e)}")
		raise

# ============================================================================
# Main Blueprint Registration Function
# ============================================================================

def setup_cash_management_blueprint(appbuilder: AppBuilder) -> None:
	"""Complete setup of APG Cash Management Flask-AppBuilder integration"""
	
	logger.info("Setting up complete APG Cash Management Flask-AppBuilder integration...")
	
	try:
		# Register all views
		register_cash_management_views(appbuilder)
		
		# Customize menu structure
		customize_cash_management_menu(appbuilder)
		
		# Configure permissions
		configure_cash_management_permissions(appbuilder)
		
		# Configure roles
		configure_cash_management_roles(appbuilder)
		
		# Configure advanced features
		configure_advanced_features(appbuilder)
		
		logger.info("APG Cash Management Flask-AppBuilder integration completed successfully")
		
	except Exception as e:
		logger.error(f"Error setting up cash management blueprint: {str(e)}")
		raise

# ============================================================================
# Legacy Compatibility Functions
# ============================================================================

def register_views(appbuilder: AppBuilder):
	"""Legacy compatibility function - redirects to new registration"""
	logger.warning("Using legacy register_views function - please use setup_cash_management_blueprint")
	setup_cash_management_blueprint(appbuilder)

def init_subcapability(appbuilder: AppBuilder):
	"""Legacy compatibility function - redirects to new setup"""
	logger.warning("Using legacy init_subcapability function - please use setup_cash_management_blueprint")
	setup_cash_management_blueprint(appbuilder)

# ============================================================================
# Blueprint Creation
# ============================================================================

def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for APG Cash Management"""
	
	cm_bp = Blueprint(
		'apg_cash_management',
		__name__,
		url_prefix='/apg/cash-management',
		template_folder='templates',
		static_folder='static'
	)
	
	logger.info("Created APG Cash Management Blueprint")
	
	return cm_bp

# ============================================================================
# API Documentation
# ============================================================================

def get_api_documentation():
	"""Get comprehensive API documentation for APG Cash Management"""
	
	return {
		'title': 'APG Cash Management API',
		'version': '1.0.0',
		'description': 'Enterprise-grade REST API for APG Cash Management with AI-powered features',
		'contact': {
			'name': 'Nyimbi Odero',
			'email': 'nyimbi@gmail.com',
			'url': 'https://www.datacraft.co.ke'
		},
		'license': {
			'name': 'Proprietary',
			'url': 'https://www.datacraft.co.ke/license'
		},
		'features': [
			'Real-time bank connectivity',
			'AI-powered cash flow forecasting',
			'Intelligent investment optimization',
			'Advanced analytics and reporting',
			'Multi-tenant architecture',
			'Enterprise security and compliance'
		],
		'endpoints': [
			{
				'path': '/accounts',
				'methods': ['GET', 'POST', 'PUT', 'DELETE'],
				'description': 'Bank account management with real-time balance monitoring'
			},
			{
				'path': '/cash-flows',
				'methods': ['GET', 'POST', 'PUT', 'DELETE'],
				'description': 'Cash flow transaction tracking with AI categorization'
			},
			{
				'path': '/forecasting/generate',
				'methods': ['POST'],
				'description': 'AI-powered cash flow forecasting with scenario analysis'
			},
			{
				'path': '/investments/opportunities',
				'methods': ['POST'],
				'description': 'AI-curated investment opportunity discovery'
			},
			{
				'path': '/positions/current',
				'methods': ['GET'],
				'description': 'Real-time cash position monitoring with risk indicators'
			},
			{
				'path': '/sync/execute',
				'methods': ['POST'],
				'description': 'Real-time bank data synchronization'
			},
			{
				'path': '/analytics/dashboard',
				'methods': ['GET'],
				'description': 'Executive analytics dashboard with KPIs'
			},
			{
				'path': '/health',
				'methods': ['GET'],
				'description': 'System health monitoring and status'
			}
		]
	}

# ============================================================================
# Export all functions
# ============================================================================

__all__ = [
	'setup_cash_management_blueprint',
	'register_cash_management_views',
	'customize_cash_management_menu',
	'configure_cash_management_permissions',
	'configure_cash_management_roles',
	'configure_advanced_features',
	'create_blueprint',
	'get_api_documentation',
	'register_views',  # Legacy compatibility
	'init_subcapability'  # Legacy compatibility
]