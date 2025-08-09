"""
APG Capability Registry - Flask-AppBuilder Blueprint

Flask-AppBuilder blueprint for APG capability registry web interface
with comprehensive views, forms, and dashboard integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, render_template, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, IntegerField, FloatField
from wtforms import BooleanField, SelectMultipleField, validators
from wtforms.widgets import TextArea, Select

from .models import (
	CRCapability, CRComposition, CRVersion, CRDependency, 
	CRRegistry, CRUsageAnalytics, CRHealthMetrics
)
from .views import (
	CapabilityListView, CapabilityDetailView, CapabilityCreateForm,
	CompositionListView, CompositionDetailView, CompositionCreateForm,
	RegistryDashboardData, CapabilitySearchForm, APG_UI_CONFIG
)
from .service import get_registry_service

# Create Flask Blueprint
capability_registry_bp = Blueprint(
	'capability_registry',
	__name__,
	template_folder='templates',
	static_folder='static',
	url_prefix='/capability-registry'
)

# =============================================================================
# Custom Widgets for APG UI Framework
# =============================================================================

class APGListWidget(ListWidget):
	"""Custom list widget for APG UI framework."""
	template = 'capability_registry/widgets/list.html'

class APGShowWidget(ShowWidget):
	"""Custom show widget for APG UI framework."""
	template = 'capability_registry/widgets/show.html'

class APGEditWidget(EditWidget):
	"""Custom edit widget for APG UI framework."""
	template = 'capability_registry/widgets/edit.html'

class APGSearchWidget(SearchWidget):
	"""Custom search widget for APG UI framework."""
	template = 'capability_registry/widgets/search.html'

# =============================================================================
# Custom Forms
# =============================================================================

class CapabilityForm(DynamicForm):
	"""Dynamic form for capability creation and editing."""
	capability_code = StringField(
		'Capability Code',
		validators=[validators.DataRequired(), validators.Length(min=3, max=100)],
		description='Unique identifier for the capability (e.g., USER_MANAGEMENT)'
	)
	capability_name = StringField(
		'Capability Name',
		validators=[validators.DataRequired(), validators.Length(min=5, max=255)],
		description='Human-readable name for the capability'
	)
	description = TextAreaField(
		'Description',
		validators=[validators.DataRequired(), validators.Length(min=10)],
		widget=TextArea(),
		description='Detailed description of the capability'
	)
	category = SelectField(
		'Category',
		validators=[validators.DataRequired()],
		choices=[
			('foundation_infrastructure', 'Foundation Infrastructure'),
			('business_operations', 'Business Operations'),
			('analytics_intelligence', 'Analytics & Intelligence'),
			('manufacturing_operations', 'Manufacturing Operations'),
			('industry_verticals', 'Industry Verticals'),
			('emerging_technologies', 'Emerging Technologies')
		],
		description='Primary category for the capability'
	)
	subcategory = StringField(
		'Subcategory',
		validators=[validators.Optional(), validators.Length(max=100)],
		description='Optional subcategory for more specific classification'
	)
	version = StringField(
		'Version',
		validators=[validators.DataRequired()],
		default='1.0.0',
		description='Semantic version number (e.g., 1.0.0)'
	)
	target_users = SelectMultipleField(
		'Target Users',
		choices=[
			('developers', 'Developers'),
			('business_users', 'Business Users'),
			('administrators', 'Administrators'),
			('end_users', 'End Users'),
			('analysts', 'Analysts')
		],
		description='Primary user types for this capability'
	)
	business_value = TextAreaField(
		'Business Value',
		validators=[validators.Optional()],
		widget=TextArea(),
		description='Business value proposition and benefits'
	)

class CompositionForm(DynamicForm):
	"""Dynamic form for composition creation and editing."""
	name = StringField(
		'Composition Name',
		validators=[validators.DataRequired(), validators.Length(min=3, max=255)],
		description='Name for the composition'
	)
	description = TextAreaField(
		'Description',
		validators=[validators.DataRequired(), validators.Length(min=10)],
		widget=TextArea(),
		description='Detailed description of the composition'
	)
	composition_type = SelectField(
		'Composition Type',
		validators=[validators.DataRequired()],
		choices=[
			('erp_enterprise', 'ERP Enterprise'),
			('industry_vertical', 'Industry Vertical'),
			('departmental', 'Departmental'),
			('microservice', 'Microservice'),
			('hybrid', 'Hybrid'),
			('custom', 'Custom')
		],
		default='custom',
		description='Type of composition'
	)
	industry_template = SelectField(
		'Industry Template',
		validators=[validators.Optional()],
		choices=[
			('', 'None'),
			('manufacturing', 'Manufacturing'),
			('healthcare', 'Healthcare'),
			('finance', 'Finance'),
			('retail', 'Retail'),
			('education', 'Education')
		],
		description='Industry-specific template'
	)
	is_template = BooleanField(
		'Save as Template',
		default=False,
		description='Save this composition as a reusable template'
	)
	is_public = BooleanField(
		'Make Public',
		default=False,
		description='Make this composition publicly available'
	)

# =============================================================================
# Model Views
# =============================================================================

class CapabilityModelView(ModelView):
	"""Model view for capabilities with APG UI integration."""
	datamodel = SQLAInterface(CRCapability)
	
	# Custom widgets
	list_widget = APGListWidget
	show_widget = APGShowWidget
	edit_widget = APGEditWidget
	search_widget = APGSearchWidget
	
	# Form configuration
	add_form = CapabilityForm
	edit_form = CapabilityForm
	
	# List view configuration
	list_columns = [
		'capability_code', 'capability_name', 'category', 'version', 
		'status', 'quality_score', 'usage_count', 'created_at'
	]
	
	# Show view configuration
	show_columns = [
		'capability_code', 'capability_name', 'description', 'category', 
		'subcategory', 'version', 'status', 'target_users', 'business_value',
		'composition_keywords', 'provides_services', 'quality_score',
		'popularity_score', 'usage_count', 'created_at', 'updated_at'
	]
	
	# Search configuration
	search_columns = ['capability_code', 'capability_name', 'description', 'category']
	
	# Filters
	base_filters = [['tenant_id', lambda: 'default']]  # Multi-tenant filter
	
	# Order
	base_order = ('capability_name', 'asc')
	
	# Labels
	label_columns = {
		'capability_code': 'Code',
		'capability_name': 'Name',
		'description': 'Description',
		'category': 'Category',
		'subcategory': 'Subcategory',
		'version': 'Version',
		'status': 'Status',
		'quality_score': 'Quality',
		'popularity_score': 'Popularity',
		'usage_count': 'Usage',
		'created_at': 'Created',
		'updated_at': 'Updated'
	}
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']

class CompositionModelView(ModelView):
	"""Model view for compositions with APG UI integration."""
	datamodel = SQLAInterface(CRComposition)
	
	# Custom widgets
	list_widget = APGListWidget
	show_widget = APGShowWidget
	edit_widget = APGEditWidget
	search_widget = APGSearchWidget
	
	# Form configuration
	add_form = CompositionForm
	edit_form = CompositionForm
	
	# List view configuration
	list_columns = [
		'name', 'composition_type', 'version', 'validation_status',
		'estimated_complexity', 'estimated_cost', 'is_template', 'created_at'
	]
	
	# Show view configuration
	show_columns = [
		'name', 'description', 'composition_type', 'version', 'validation_status',
		'validation_results', 'estimated_complexity', 'estimated_cost',
		'business_requirements', 'is_template', 'is_public', 'created_at'
	]
	
	# Search configuration
	search_columns = ['name', 'description', 'composition_type']
	
	# Filters
	base_filters = [['tenant_id', lambda: 'default']]
	
	# Order
	base_order = ('name', 'asc')
	
	# Labels
	label_columns = {
		'name': 'Name',
		'description': 'Description',
		'composition_type': 'Type',
		'version': 'Version',
		'validation_status': 'Status',
		'estimated_complexity': 'Complexity',
		'estimated_cost': 'Est. Cost',
		'is_template': 'Template',
		'is_public': 'Public',
		'created_at': 'Created'
	}

class VersionModelView(ModelView):
	"""Model view for capability versions."""
	datamodel = SQLAInterface(CRVersion)
	
	# List view configuration
	list_columns = [
		'version_number', 'release_date', 'backward_compatible',
		'quality_score', 'test_coverage', 'status'
	]
	
	# Show view configuration
	show_columns = [
		'version_number', 'release_date', 'release_notes', 'breaking_changes',
		'new_features', 'backward_compatible', 'forward_compatible',
		'quality_score', 'test_coverage', 'security_audit_passed', 'status'
	]
	
	# Search configuration
	search_columns = ['version_number', 'release_notes']
	
	# Order
	base_order = ('release_date', 'desc')
	
	# Read-only (versions created through service)
	base_permissions = ['can_list', 'can_show']

# =============================================================================
# Custom Views
# =============================================================================

class RegistryDashboardView(BaseView):
	"""Dashboard view for capability registry overview."""
	
	route_base = '/dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard view."""
		# In real implementation, would get data from service
		dashboard_data = {
			'total_capabilities': 47,
			'active_capabilities': 42,
			'total_compositions': 23,
			'total_versions': 156,
			'registry_health_score': 0.94,
			'avg_quality_score': 0.78,
			'recent_capabilities': [],
			'recent_compositions': [],
			'category_stats': [
				{'category': 'Foundation Infrastructure', 'count': 12},
				{'category': 'Business Operations', 'count': 15},
				{'category': 'Analytics & Intelligence', 'count': 8},
				{'category': 'Manufacturing Operations', 'count': 7},
				{'category': 'Industry Verticals', 'count': 5}
			],
			'marketplace_enabled': True,
			'published_capabilities': 15,
			'pending_submissions': 3
		}
		
		return self.render_template(
			'capability_registry/dashboard.html',
			dashboard_data=dashboard_data,
			ui_config=APG_UI_CONFIG
		)

class CompositionDesignerView(BaseView):
	"""Interactive composition designer view."""
	
	route_base = '/compose'
	default_view = 'designer'
	
	@expose('/')
	@has_access
	def designer(self):
		"""Composition designer interface."""
		return self.render_template(
			'capability_registry/composition_designer.html',
			ui_config=APG_UI_CONFIG
		)
	
	@expose('/validate', methods=['POST'])
	@has_access
	def validate_composition(self):
		"""AJAX endpoint for composition validation."""
		data = request.get_json()
		capability_ids = data.get('capability_ids', [])
		
		# In real implementation, would call service
		validation_result = {
			'is_valid': True,
			'validation_score': 0.85,
			'conflicts': [],
			'recommendations': [
				{
					'type': 'optimization',
					'title': 'Consider adding caching',
					'description': 'Adding a caching layer could improve performance'
				}
			],
			'performance_impact': {
				'memory_usage_mb': 250,
				'response_time_ms': 120,
				'scalability_score': 0.9
			}
		}
		
		return jsonify(validation_result)

class MarketplaceView(BaseView):
	"""Marketplace integration view."""
	
	route_base = '/marketplace'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Marketplace overview."""
		marketplace_data = {
			'marketplace_url': 'https://marketplace.apg.platform',
			'published_capabilities': 15,
			'pending_submissions': 3,
			'total_downloads': 1247,
			'featured_capabilities': []
		}
		
		return self.render_template(
			'capability_registry/marketplace.html',
			marketplace_data=marketplace_data,
			ui_config=APG_UI_CONFIG
		)
	
	@expose('/publish/<capability_id>')
	@has_access
	def publish_capability(self, capability_id):
		"""Publish capability to marketplace."""
		return self.render_template(
			'capability_registry/publish_capability.html',
			capability_id=capability_id,
			ui_config=APG_UI_CONFIG
		)

class AnalyticsView(BaseView):
	"""Analytics and reporting view."""
	
	route_base = '/analytics'
	default_view = 'overview'
	
	@expose('/')
	@has_access
	def overview(self):
		"""Analytics overview."""
		analytics_data = {
			'usage_trends': [],
			'performance_metrics': [],
			'popular_capabilities': [],
			'composition_patterns': []
		}
		
		return self.render_template(
			'capability_registry/analytics.html',
			analytics_data=analytics_data,
			ui_config=APG_UI_CONFIG
		)

class MobileView(BaseView):
	"""Mobile-optimized view with PWA support."""
	
	route_base = '/mobile'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Mobile app interface."""
		return self.render_template(
			'capability_registry/mobile.html',
			ui_config=APG_UI_CONFIG
		)
	
	@expose('/manifest.json')
	def pwa_manifest(self):
		"""PWA manifest endpoint."""
		from .mobile_service import generate_pwa_manifest
		manifest = generate_pwa_manifest()
		return jsonify(manifest.model_dump())
	
	@expose('/sw.js')
	def service_worker(self):
		"""Service worker endpoint."""
		from .mobile_service import generate_service_worker
		sw_content = generate_service_worker()
		response = self.appbuilder.app.response_class(
			sw_content,
			mimetype='application/javascript'
		)
		response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
		return response

# =============================================================================
# Chart Views
# =============================================================================

class CapabilityUsageChartView(DirectByChartView):
	"""Chart view for capability usage analytics."""
	chart_title = 'Capability Usage Over Time'
	chart_type = 'LineChart'
	direct_columns = {
		'Usage Date': ('usage_date', None),
		'Usage Count': ('usage_count', 'sum')
	}
	base_order = ('usage_date', 'asc')
	
	def query_obj(self):
		# In real implementation, would query CRUsageAnalytics
		return None

class QualityScoreChartView(DirectByChartView):
	"""Chart view for quality score distribution."""
	chart_title = 'Quality Score Distribution'
	chart_type = 'ColumnChart'
	direct_columns = {
		'Category': ('category', None),
		'Average Quality': ('quality_score', 'avg')
	}
	group_by_columns = ['category']

# =============================================================================
# API Endpoints for AJAX
# =============================================================================

@capability_registry_bp.route('/api/capabilities/search', methods=['POST'])
def api_search_capabilities():
	"""AJAX endpoint for capability search."""
	data = request.get_json()
	query = data.get('query', '')
	category = data.get('category')
	
	# In real implementation, would call service
	results = {
		'capabilities': [
			{
				'capability_id': 'cap_001',
				'capability_code': 'USER_MANAGEMENT',
				'capability_name': 'User Management',
				'description': 'Complete user management system',
				'category': 'foundation_infrastructure',
				'quality_score': 0.92
			}
		],
		'total_count': 1,
		'search_time_ms': 45.2
	}
	
	return jsonify(results)

@capability_registry_bp.route('/api/compositions/validate', methods=['POST'])
def api_validate_composition():
	"""AJAX endpoint for composition validation."""
	data = request.get_json()
	capability_ids = data.get('capability_ids', [])
	
	# In real implementation, would call service
	result = {
		'is_valid': True,
		'validation_score': 0.88,
		'conflicts': [],
		'recommendations': [],
		'performance_impact': {
			'memory_usage_mb': 180,
			'response_time_ms': 95
		}
	}
	
	return jsonify(result)

@capability_registry_bp.route('/api/marketplace/prepare', methods=['POST'])
def api_prepare_marketplace():
	"""AJAX endpoint for marketplace preparation."""
	data = request.get_json()
	capability_id = data.get('capability_id')
	
	# In real implementation, would call service
	result = {
		'success': True,
		'quality_score': 0.85,
		'compliance_passed': True,
		'documentation_complete': True,
		'validation_errors': []
	}
	
	return jsonify(result)

# =============================================================================
# Template Filters
# =============================================================================

@capability_registry_bp.app_template_filter('quality_badge')
def quality_badge_filter(score):
	"""Template filter for quality score badges."""
	if score >= 0.9:
		return '<span class="badge badge-success">Excellent</span>'
	elif score >= 0.8:
		return '<span class="badge badge-primary">Good</span>'
	elif score >= 0.7:
		return '<span class="badge badge-warning">Fair</span>'
	else:
		return '<span class="badge badge-danger">Poor</span>'

@capability_registry_bp.app_template_filter('status_badge')
def status_badge_filter(status):
	"""Template filter for status badges."""
	status_map = {
		'active': 'success',
		'discovered': 'info',
		'registered': 'primary',
		'validated': 'success',
		'deprecated': 'warning',
		'retired': 'danger'
	}
	badge_class = status_map.get(status.lower(), 'secondary')
	return f'<span class="badge badge-{badge_class}">{status.title()}</span>'

# =============================================================================
# Blueprint Registration Helper
# =============================================================================

def register_views(appbuilder):
	"""Register all views with Flask-AppBuilder."""
	
	# Model views
	appbuilder.add_view(
		CapabilityModelView,
		"Capabilities",
		icon="fa-cogs",
		category="Registry",
		category_icon="fa-database"
	)
	
	appbuilder.add_view(
		CompositionModelView,
		"Compositions",
		icon="fa-cubes",
		category="Registry"
	)
	
	appbuilder.add_view(
		VersionModelView,
		"Versions",
		icon="fa-tags",
		category="Registry"
	)
	
	# Custom views
	appbuilder.add_view(
		RegistryDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Registry"
	)
	
	appbuilder.add_view(
		CompositionDesignerView,
		"Composition Designer",
		icon="fa-magic",
		category="Tools",
		category_icon="fa-wrench"
	)
	
	appbuilder.add_view(
		MarketplaceView,
		"Marketplace",
		icon="fa-shopping-cart",
		category="Tools"
	)
	
	appbuilder.add_view(
		AnalyticsView,
		"Analytics",
		icon="fa-bar-chart",
		category="Tools"
	)
	
	appbuilder.add_view(
		MobileView,
		"Mobile App",
		icon="fa-mobile",
		category="Tools"
	)
	
	# Chart views
	appbuilder.add_view(
		CapabilityUsageChartView,
		"Usage Chart",
		icon="fa-line-chart",
		category="Analytics",
		category_icon="fa-bar-chart"
	)
	
	appbuilder.add_view(
		QualityScoreChartView,
		"Quality Chart", 
		icon="fa-pie-chart",
		category="Analytics"
	)

# Export blueprint and registration function
__all__ = [
	'capability_registry_bp',
	'register_views',
	'CapabilityModelView',
	'CompositionModelView',
	'RegistryDashboardView',
	'CompositionDesignerView',
	'MarketplaceView',
	'AnalyticsView',
	'MobileView'
]