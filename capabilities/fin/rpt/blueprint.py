"""
Financial Reporting Blueprint

Flask blueprint registration for Financial Reporting sub-capability
including all views and URL routing configuration.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

from .views import (
	CFRFReportTemplateModelView,
	CFRFFinancialStatementModelView,
	CFRFConsolidationModelView,
	CFRFAnalyticalReportModelView,
	CFRFReportGenerationView,
	CFRFFinancialDashboardView,
	CFRFReportPeriodModelView,
	CFRFNotesModelView,
	CFRFDisclosureModelView,
	CFRFReportDistributionModelView
)

# Create blueprint
financial_reporting_bp = Blueprint(
	'financial_reporting',
	__name__,
	url_prefix='/core_financials/financial_reporting',
	template_folder='templates',
	static_folder='static'
)


def register_views(appbuilder: AppBuilder):
	"""Register all Financial Reporting views with Flask-AppBuilder"""
	
	# Core Financial Reporting Views
	appbuilder.add_view(
		CFRFReportTemplateModelView,
		"Report Templates",
		icon="fa-file-text-o",
		category="Financial Reporting",
		category_icon="fa-calculator"
	)
	
	appbuilder.add_view(
		CFRFFinancialStatementModelView,
		"Financial Statements",
		icon="fa-file-text",
		category="Financial Reporting"
	)
	
	appbuilder.add_view(
		CFRFConsolidationModelView,
		"Consolidations",
		icon="fa-sitemap",
		category="Financial Reporting"
	)
	
	appbuilder.add_view(
		CFRFAnalyticalReportModelView,
		"Analytical Reports",
		icon="fa-line-chart",
		category="Financial Reporting"
	)
	
	# Report Generation and Dashboard Views
	appbuilder.add_view(
		CFRFReportGenerationView,
		"Report Generation",
		icon="fa-play-circle",
		category="Financial Reporting"
	)
	
	appbuilder.add_view(
		CFRFFinancialDashboardView,
		"Financial Dashboard",
		icon="fa-dashboard",
		category="Financial Reporting"
	)
	
	# Supporting Configuration Views
	appbuilder.add_view(
		CFRFReportPeriodModelView,
		"Reporting Periods",
		icon="fa-calendar",
		category="Financial Reporting - Setup",
		category_icon="fa-cog"
	)
	
	appbuilder.add_view(
		CFRFNotesModelView,
		"Statement Notes",
		icon="fa-sticky-note-o",
		category="Financial Reporting - Setup"
	)
	
	appbuilder.add_view(
		CFRFDisclosureModelView,
		"Regulatory Disclosures",
		icon="fa-legal",
		category="Financial Reporting - Setup"
	)
	
	appbuilder.add_view(
		CFRFReportDistributionModelView,
		"Distribution Lists",
		icon="fa-share-alt",
		category="Financial Reporting - Setup"
	)
	
	# Add separator for better menu organization
	appbuilder.add_separator("Financial Reporting")


def register_menu_links(appbuilder: AppBuilder):
	"""Register additional menu links for Financial Reporting"""
	
	# Quick access links
	appbuilder.add_link(
		"Generate Balance Sheet",
		href="/core_financials/financial_reporting/generate?template_type=balance_sheet",
		icon="fa-balance-scale",
		category="Financial Reporting - Quick Actions",
		category_icon="fa-bolt"
	)
	
	appbuilder.add_link(
		"Generate Income Statement",
		href="/core_financials/financial_reporting/generate?template_type=income_statement",
		icon="fa-line-chart",
		category="Financial Reporting - Quick Actions"
	)
	
	appbuilder.add_link(
		"Generate Cash Flow",
		href="/core_financials/financial_reporting/generate?template_type=cash_flow",
		icon="fa-money",
		category="Financial Reporting - Quick Actions"
	)
	
	# Reports and Analytics
	appbuilder.add_link(
		"Financial Metrics",
		href="/core_financials/financial_reporting/api/metrics",
		icon="fa-bar-chart",
		category="Financial Reporting - Analytics",
		category_icon="fa-area-chart"
	)
	
	appbuilder.add_link(
		"Period Comparison",
		href="/core_financials/financial_reporting/analytics/period_comparison",
		icon="fa-calendar-check-o",
		category="Financial Reporting - Analytics"
	)
	
	appbuilder.add_link(
		"Consolidation Analysis",
		href="/core_financials/financial_reporting/analytics/consolidation",
		icon="fa-sitemap",
		category="Financial Reporting - Analytics"
	)


def register_api_endpoints(appbuilder: AppBuilder):
	"""Register API endpoints for Financial Reporting"""
	
	# These would typically be registered with Flask-RESTX or similar
	# Here we're just documenting the intended API structure
	
	api_endpoints = [
		# Report Templates
		{
			'endpoint': '/api/core_financials/fr/templates',
			'methods': ['GET', 'POST', 'PUT', 'DELETE'],
			'description': 'Manage report templates'
		},
		
		# Financial Statements
		{
			'endpoint': '/api/core_financials/fr/statements',
			'methods': ['GET', 'POST'],
			'description': 'Manage financial statements'
		},
		
		# Report Generation
		{
			'endpoint': '/api/core_financials/fr/generate',
			'methods': ['POST'],
			'description': 'Generate financial reports'
		},
		
		# Consolidation
		{
			'endpoint': '/api/core_financials/fr/consolidations',
			'methods': ['GET', 'POST', 'PUT', 'DELETE'],
			'description': 'Manage consolidation rules'
		},
		
		# Analytical Reports
		{
			'endpoint': '/api/core_financials/fr/reports',
			'methods': ['GET', 'POST', 'PUT', 'DELETE'],
			'description': 'Manage analytical reports'
		},
		
		# Distribution
		{
			'endpoint': '/api/core_financials/fr/distributions',
			'methods': ['GET', 'POST', 'PUT', 'DELETE'],
			'description': 'Manage report distributions'
		},
		
		# Status and Monitoring
		{
			'endpoint': '/api/core_financials/fr/status/<generation_id>',
			'methods': ['GET'],
			'description': 'Get report generation status'
		},
		
		# Dashboard Metrics
		{
			'endpoint': '/api/core_financials/fr/metrics',
			'methods': ['GET'],
			'description': 'Get financial dashboard metrics'
		}
	]
	
	return api_endpoints


def configure_permissions(appbuilder: AppBuilder):
	"""Configure permissions for Financial Reporting"""
	
	# Define permission categories
	permission_categories = [
		{
			'name': 'Financial Reporting - Read',
			'permissions': [
				'can_list_CFRFReportTemplateModelView',
				'can_show_CFRFReportTemplateModelView',
				'can_list_CFRFFinancialStatementModelView',
				'can_show_CFRFFinancialStatementModelView',
				'can_list_CFRFAnalyticalReportModelView',
				'can_show_CFRFAnalyticalReportModelView',
				'can_index_CFRFFinancialDashboardView',
				'can_api_metrics_CFRFFinancialDashboardView'
			]
		},
		{
			'name': 'Financial Reporting - Write',
			'permissions': [
				'can_add_CFRFReportTemplateModelView',
				'can_edit_CFRFReportTemplateModelView',
				'can_add_CFRFAnalyticalReportModelView',
				'can_edit_CFRFAnalyticalReportModelView',
				'can_add_CFRFNotesModelView',
				'can_edit_CFRFNotesModelView'
			]
		},
		{
			'name': 'Financial Reporting - Generate',
			'permissions': [
				'can_index_CFRFReportGenerationView',
				'can_generate_CFRFReportGenerationView',
				'can_status_CFRFReportGenerationView',
				'can_api_status_CFRFReportGenerationView',
				'can_generate_reports_CFRFAnalyticalReportModelView'
			]
		},
		{
			'name': 'Financial Reporting - Consolidate',
			'permissions': [
				'can_list_CFRFConsolidationModelView',
				'can_show_CFRFConsolidationModelView',
				'can_add_CFRFConsolidationModelView',
				'can_edit_CFRFConsolidationModelView'
			]
		},
		{
			'name': 'Financial Reporting - Distribute',
			'permissions': [
				'can_list_CFRFReportDistributionModelView',
				'can_show_CFRFReportDistributionModelView',
				'can_add_CFRFReportDistributionModelView',
				'can_edit_CFRFReportDistributionModelView'
			]
		},
		{
			'name': 'Financial Reporting - Admin',
			'permissions': [
				'can_delete_CFRFReportTemplateModelView',
				'can_delete_CFRFConsolidationModelView',
				'can_delete_CFRFAnalyticalReportModelView',
				'can_delete_CFRFReportDistributionModelView',
				'can_clone_template_CFRFReportTemplateModelView',
				'can_activate_templates_CFRFReportTemplateModelView',
				'can_publish_statements_CFRFFinancialStatementModelView',
				'can_finalize_statements_CFRFFinancialStatementModelView'
			]
		}
	]
	
	return permission_categories


def register_tasks(appbuilder: AppBuilder):
	"""Register background tasks for Financial Reporting"""
	
	# These would typically be registered with Celery or similar task queue
	# Here we're just documenting the intended task structure
	
	scheduled_tasks = [
		{
			'name': 'auto_generate_monthly_statements',
			'schedule': 'monthly',
			'description': 'Automatically generate monthly financial statements'
		},
		{
			'name': 'auto_generate_quarterly_statements',
			'schedule': 'quarterly',
			'description': 'Automatically generate quarterly financial statements'
		},
		{
			'name': 'distribute_scheduled_reports',
			'schedule': 'daily',
			'description': 'Distribute scheduled analytical reports'
		},
		{
			'name': 'cleanup_old_generations',
			'schedule': 'weekly',
			'description': 'Clean up old report generation files'
		},
		{
			'name': 'update_consolidation_rates',
			'schedule': 'daily',
			'description': 'Update currency exchange rates for consolidation'
		},
		{
			'name': 'validate_financial_data',
			'schedule': 'daily',
			'description': 'Validate financial data integrity and completeness'
		}
	]
	
	return scheduled_tasks


def get_blueprint_info():
	"""Get blueprint information for registration"""
	return {
		'blueprint': financial_reporting_bp,
		'views': [
			CFRFReportTemplateModelView,
			CFRFFinancialStatementModelView,
			CFRFConsolidationModelView,
			CFRFAnalyticalReportModelView,
			CFRFReportGenerationView,
			CFRFFinancialDashboardView,
			CFRFReportPeriodModelView,
			CFRFNotesModelView,
			CFRFDisclosureModelView,
			CFRFReportDistributionModelView
		],
		'register_views': register_views,
		'register_menu_links': register_menu_links,
		'configure_permissions': configure_permissions,
		'register_tasks': register_tasks,
		'api_endpoints': register_api_endpoints
	}