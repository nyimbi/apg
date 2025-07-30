"""
APG Vendor Management - Flask-AppBuilder Views
Comprehensive web interface for AI-powered vendor lifecycle management

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from flask import flash, jsonify, redirect, request, url_for
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import protect
from wtforms import Form, StringField, SelectField, DecimalField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, Email, NumberRange, Optional as WTFOptional

from .models import (
	VMVendor, VMPerformance, VMRisk, VMIntelligence,
	VendorStatus, VendorType, VendorSizeClassification, 
	StrategicImportance, RiskSeverity
)
from .service import VendorManagementService, VMDatabaseContext
from .intelligence_service import VendorIntelligenceEngine


# ============================================================================
# FORMS FOR VENDOR MANAGEMENT
# ============================================================================

class VendorForm(Form):
	"""Form for creating/editing vendors"""
	
	vendor_code = StringField('Vendor Code', validators=[DataRequired()])
	name = StringField('Vendor Name', validators=[DataRequired()])
	legal_name = StringField('Legal Name', validators=[WTFOptional()])
	display_name = StringField('Display Name', validators=[WTFOptional()])
	
	vendor_type = SelectField(
		'Vendor Type',
		choices=[(e.value, e.value.title()) for e in VendorType],
		validators=[DataRequired()]
	)
	
	category = StringField('Category', validators=[DataRequired()])
	subcategory = StringField('Subcategory', validators=[WTFOptional()])
	industry = StringField('Industry', validators=[WTFOptional()])
	
	size_classification = SelectField(
		'Size Classification',
		choices=[(e.value, e.value.title()) for e in VendorSizeClassification],
		validators=[DataRequired()]
	)
	
	email = StringField('Email', validators=[WTFOptional(), Email()])
	phone = StringField('Phone', validators=[WTFOptional()])
	website = StringField('Website', validators=[WTFOptional()])
	
	address_line1 = StringField('Address Line 1', validators=[WTFOptional()])
	address_line2 = StringField('Address Line 2', validators=[WTFOptional()])
	city = StringField('City', validators=[WTFOptional()])
	state_province = StringField('State/Province', validators=[WTFOptional()])
	postal_code = StringField('Postal Code', validators=[WTFOptional()])
	country = StringField('Country', validators=[WTFOptional()])
	
	credit_rating = StringField('Credit Rating', validators=[WTFOptional()])
	payment_terms = StringField('Payment Terms', validators=[WTFOptional()])
	currency = StringField('Currency', validators=[WTFOptional()])
	tax_id = StringField('Tax ID', validators=[WTFOptional()])
	duns_number = StringField('DUNS Number', validators=[WTFOptional()])
	
	strategic_importance = SelectField(
		'Strategic Importance',
		choices=[(e.value, e.value.title()) for e in StrategicImportance],
		validators=[DataRequired()]
	)
	
	preferred_vendor = BooleanField('Preferred Vendor')
	strategic_partner = BooleanField('Strategic Partner')
	diversity_category = StringField('Diversity Category', validators=[WTFOptional()])


class PerformanceForm(Form):
	"""Form for recording vendor performance"""
	
	vendor_id = StringField('Vendor ID', validators=[DataRequired()])
	measurement_period = SelectField(
		'Measurement Period',
		choices=[('monthly', 'Monthly'), ('quarterly', 'Quarterly'), ('annual', 'Annual')],
		validators=[DataRequired()]
	)
	
	overall_score = DecimalField(
		'Overall Score', 
		validators=[DataRequired(), NumberRange(min=0, max=100)]
	)
	quality_score = DecimalField(
		'Quality Score', 
		validators=[DataRequired(), NumberRange(min=0, max=100)]
	)
	delivery_score = DecimalField(
		'Delivery Score', 
		validators=[DataRequired(), NumberRange(min=0, max=100)]
	)
	cost_score = DecimalField(
		'Cost Score', 
		validators=[DataRequired(), NumberRange(min=0, max=100)]
	)
	service_score = DecimalField(
		'Service Score', 
		validators=[DataRequired(), NumberRange(min=0, max=100)]
	)
	
	on_time_delivery_rate = DecimalField(
		'On-Time Delivery Rate (%)', 
		validators=[WTFOptional(), NumberRange(min=0, max=100)]
	)
	quality_rejection_rate = DecimalField(
		'Quality Rejection Rate (%)', 
		validators=[WTFOptional(), NumberRange(min=0, max=100)]
	)


class RiskForm(Form):
	"""Form for recording vendor risks"""
	
	vendor_id = StringField('Vendor ID', validators=[DataRequired()])
	risk_type = StringField('Risk Type', validators=[DataRequired()])
	risk_category = StringField('Risk Category', validators=[DataRequired()])
	
	severity = SelectField(
		'Severity',
		choices=[(e.value, e.value.title()) for e in RiskSeverity],
		validators=[DataRequired()]
	)
	
	title = StringField('Risk Title', validators=[DataRequired()])
	description = TextAreaField('Description', validators=[DataRequired()])
	root_cause = TextAreaField('Root Cause', validators=[WTFOptional()])
	potential_impact = TextAreaField('Potential Impact', validators=[WTFOptional()])
	
	overall_risk_score = DecimalField(
		'Overall Risk Score', 
		validators=[DataRequired(), NumberRange(min=0, max=100)]
	)
	financial_impact = DecimalField('Financial Impact ($)', validators=[WTFOptional()])
	
	mitigation_strategy = TextAreaField('Mitigation Strategy', validators=[WTFOptional()])


# ============================================================================
# BASE VIEW CLASS WITH COMMON FUNCTIONALITY
# ============================================================================

class BaseVendorView(BaseView):
	"""Base view class with common vendor management functionality"""
	
	def __init__(self):
		super().__init__()
		self.db_context = None
		self.vendor_service = None
		self.intelligence_engine = None
	
	def _get_tenant_id(self) -> UUID:
		"""Get current tenant ID from session/context"""
		# In production, this would get the tenant from user session
		return UUID('00000000-0000-0000-0000-000000000000')
	
	def _get_current_user_id(self) -> UUID:
		"""Get current user ID from session/context"""
		# In production, this would get the user from session
		return UUID('00000000-0000-0000-0000-000000000000')
	
	async def _get_vendor_service(self) -> VendorManagementService:
		"""Get vendor service instance"""
		if not self.vendor_service:
			if not self.db_context:
				# In production, get from app config
				self.db_context = VMDatabaseContext("postgresql://localhost/apg")
			
			self.vendor_service = VendorManagementService(
				self._get_tenant_id(), 
				self.db_context
			)
			self.vendor_service.set_current_user(self._get_current_user_id())
		
		return self.vendor_service
	
	async def _get_intelligence_engine(self) -> VendorIntelligenceEngine:
		"""Get intelligence engine instance"""
		if not self.intelligence_engine:
			if not self.db_context:
				self.db_context = VMDatabaseContext("postgresql://localhost/apg")
			
			self.intelligence_engine = VendorIntelligenceEngine(
				self._get_tenant_id(),
				self.db_context
			)
			self.intelligence_engine.set_current_user(self._get_current_user_id())
		
		return self.intelligence_engine
	
	def _format_currency(self, amount: float) -> str:
		"""Format currency values"""
		return f"${amount:,.2f}" if amount else "$0.00"
	
	def _format_percentage(self, value: float) -> str:
		"""Format percentage values"""
		return f"{value:.1f}%" if value else "0.0%"
	
	def _format_score(self, score: float) -> str:
		"""Format score values"""
		return f"{score:.1f}" if score else "N/A"


# ============================================================================
# VENDOR DASHBOARD VIEW
# ============================================================================

class VendorDashboardView(BaseVendorView):
	"""Advanced vendor management dashboard with AI insights"""
	
	route_base = "/vendor_management"
	default_view = 'dashboard'
	
	@expose('/dashboard')
	@has_access
	def dashboard(self):
		"""Main vendor management dashboard"""
		
		try:
			# Run async operations
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			dashboard_data = loop.run_until_complete(self._get_dashboard_data())
			
			return self.render_template(
				'vendor_dashboard.html',
				dashboard_data=dashboard_data,
				title="AI-Powered Vendor Management Dashboard"
			)
			
		except Exception as e:
			flash(f"Error loading dashboard: {str(e)}", "danger")
			return self.render_template('error.html', error=str(e))
		finally:
			loop.close()
	
	async def _get_dashboard_data(self) -> Dict[str, Any]:
		"""Collect comprehensive dashboard data"""
		
		service = await self._get_vendor_service()
		
		# Get basic analytics
		analytics = await service.get_vendor_analytics()
		
		# Get vendor list for summary
		vendor_list = await service.list_vendors(page_size=10)
		
		# Get top performers
		top_performers = []
		for vendor in vendor_list.vendors[:5]:
			perf_summary = await service.get_vendor_performance_summary(vendor.id)
			if perf_summary:
				top_performers.append(perf_summary)
		
		return {
			'analytics': analytics,
			'recent_vendors': vendor_list.vendors[:5],
			'top_performers': top_performers,
			'total_vendors': analytics['vendor_counts']['total_vendors'],
			'active_vendors': analytics['vendor_counts']['active_vendors'],
			'preferred_vendors': analytics['vendor_counts']['preferred_vendors'],
			'strategic_partners': analytics['vendor_counts']['strategic_partners'],
			'avg_performance': self._format_score(analytics['performance_metrics']['avg_performance']),
			'avg_risk': self._format_score(analytics['performance_metrics']['avg_risk']),
			'recent_activities': analytics['recent_activities']
		}


# ============================================================================
# VENDOR LIST AND CRUD VIEWS
# ============================================================================

class VendorListView(BaseVendorView):
	"""Vendor list view with advanced filtering and AI insights"""
	
	route_base = "/vendor_management/vendors"
	default_view = 'list'
	
	@expose('/list')
	@expose('/list/<int:page>')
	@has_access
	def list(self, page=1):
		"""List vendors with pagination and filtering"""
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			# Get filter parameters
			filters = {
				'status': request.args.get('status'),
				'category': request.args.get('category'),
				'vendor_type': request.args.get('vendor_type'),
				'search': request.args.get('search')
			}
			
			# Remove None values
			filters = {k: v for k, v in filters.items() if v}
			
			sort_by = request.args.get('sort_by', 'name')
			sort_order = request.args.get('sort_order', 'asc')
			
			vendor_data = loop.run_until_complete(
				self._get_vendor_list(page, filters, sort_by, sort_order)
			)
			
			return self.render_template(
				'vendor_list.html',
				vendors=vendor_data['vendors'],
				pagination=vendor_data['pagination'],
				filters=filters,
				sort_by=sort_by,
				sort_order=sort_order,
				title="Vendor Management"
			)
			
		except Exception as e:
			flash(f"Error loading vendors: {str(e)}", "danger")
			return redirect(url_for('VendorDashboardView.dashboard'))
		finally:
			loop.close()
	
	async def _get_vendor_list(
		self, 
		page: int, 
		filters: Dict[str, str],
		sort_by: str,
		sort_order: str
	) -> Dict[str, Any]:
		"""Get paginated vendor list with metadata"""
		
		service = await self._get_vendor_service()
		
		vendor_response = await service.list_vendors(
			page=page,
			page_size=25,
			filters=filters,
			sort_by=sort_by,
			sort_order=sort_order
		)
		
		# Add formatted data for display
		formatted_vendors = []
		for vendor in vendor_response.vendors:
			formatted_vendors.append({
				'vendor': vendor,
				'performance_score': self._format_score(float(vendor.performance_score)),
				'risk_score': self._format_score(float(vendor.risk_score)),
				'intelligence_score': self._format_score(float(vendor.intelligence_score)),
				'relationship_score': self._format_score(float(vendor.relationship_score)),
				'status_badge': self._get_status_badge(vendor.status),
				'type_badge': self._get_type_badge(vendor.vendor_type)
			})
		
		return {
			'vendors': formatted_vendors,
			'pagination': {
				'page': vendor_response.page,
				'page_size': vendor_response.page_size,
				'total_count': vendor_response.total_count,
				'has_next': vendor_response.has_next,
				'has_prev': page > 1,
				'total_pages': (vendor_response.total_count + vendor_response.page_size - 1) // vendor_response.page_size
			}
		}
	
	def _get_status_badge(self, status: VendorStatus) -> Dict[str, str]:
		"""Get Bootstrap badge class for vendor status"""
		badge_classes = {
			VendorStatus.ACTIVE: {'class': 'success', 'text': 'Active'},
			VendorStatus.INACTIVE: {'class': 'secondary', 'text': 'Inactive'},
			VendorStatus.PENDING: {'class': 'warning', 'text': 'Pending'},
			VendorStatus.SUSPENDED: {'class': 'danger', 'text': 'Suspended'},
			VendorStatus.TERMINATED: {'class': 'dark', 'text': 'Terminated'},
			VendorStatus.UNDER_REVIEW: {'class': 'info', 'text': 'Under Review'}
		}
		return badge_classes.get(status, {'class': 'secondary', 'text': status.value})
	
	def _get_type_badge(self, vendor_type: VendorType) -> Dict[str, str]:
		"""Get Bootstrap badge class for vendor type"""
		return {'class': 'primary', 'text': vendor_type.value.replace('_', ' ').title()}


# ============================================================================
# VENDOR DETAIL VIEW
# ============================================================================

class VendorDetailView(BaseVendorView):
	"""Detailed vendor view with AI insights and analytics"""
	
	route_base = "/vendor_management/vendor"
	
	@expose('/<vendor_id>')
	@has_access
	def detail(self, vendor_id):
		"""Show detailed vendor information with AI insights"""
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			vendor_data = loop.run_until_complete(
				self._get_vendor_detail(vendor_id)
			)
			
			if not vendor_data:
				flash("Vendor not found", "warning")
				return redirect(url_for('VendorListView.list'))
			
			return self.render_template(
				'vendor_detail.html',
				**vendor_data,
				title=f"Vendor: {vendor_data['vendor'].name}"
			)
			
		except Exception as e:
			flash(f"Error loading vendor details: {str(e)}", "danger")
			return redirect(url_for('VendorListView.list'))
		finally:
			loop.close()
	
	async def _get_vendor_detail(self, vendor_id: str) -> Optional[Dict[str, Any]]:
		"""Get comprehensive vendor details with AI insights"""
		
		service = await self._get_vendor_service()
		intelligence_engine = await self._get_intelligence_engine()
		
		# Get vendor basic info
		vendor = await service.get_vendor_by_id(vendor_id)
		if not vendor:
			return None
		
		# Get performance summary
		performance_summary = await service.get_vendor_performance_summary(vendor_id)
		
		# Get behavior patterns
		behavior_patterns = await intelligence_engine.analyze_vendor_behavior_patterns(vendor_id)
		
		# Get predictive insights
		predictive_insights = await intelligence_engine.generate_predictive_insights(vendor_id)
		
		return {
			'vendor': vendor,
			'performance_summary': performance_summary,
			'behavior_patterns': behavior_patterns,
			'predictive_insights': predictive_insights,
			'formatted_scores': {
				'performance': self._format_score(float(vendor.performance_score)),
				'risk': self._format_score(float(vendor.risk_score)),
				'intelligence': self._format_score(float(vendor.intelligence_score)),
				'relationship': self._format_score(float(vendor.relationship_score))
			}
		}


# ============================================================================
# VENDOR PERFORMANCE VIEW
# ============================================================================

class VendorPerformanceView(BaseVendorView):
	"""Vendor performance tracking and analytics"""
	
	route_base = "/vendor_management/performance"
	
	@expose('/')
	@expose('/<vendor_id>')
	@has_access
	def performance(self, vendor_id=None):
		"""Show vendor performance analytics"""
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			if vendor_id:
				performance_data = loop.run_until_complete(
					self._get_vendor_performance(vendor_id)
				)
				template = 'vendor_performance_detail.html'
				title = f"Performance: {performance_data['vendor'].name}"
			else:
				performance_data = loop.run_until_complete(
					self._get_performance_overview()
				)
				template = 'vendor_performance_overview.html'
				title = "Vendor Performance Overview"
			
			return self.render_template(
				template,
				**performance_data,
				title=title
			)
			
		except Exception as e:
			flash(f"Error loading performance data: {str(e)}", "danger")
			return redirect(url_for('VendorDashboardView.dashboard'))
		finally:
			loop.close()
	
	async def _get_vendor_performance(self, vendor_id: str) -> Dict[str, Any]:
		"""Get detailed performance data for specific vendor"""
		
		service = await self._get_vendor_service()
		
		vendor = await service.get_vendor_by_id(vendor_id)
		performance_summary = await service.get_vendor_performance_summary(vendor_id)
		
		return {
			'vendor': vendor,
			'performance_summary': performance_summary
		}
	
	async def _get_performance_overview(self) -> Dict[str, Any]:
		"""Get performance overview for all vendors"""
		
		service = await self._get_vendor_service()
		analytics = await service.get_vendor_analytics()
		
		return {
			'analytics': analytics,
			'performance_metrics': analytics['performance_metrics']
		}


# ============================================================================
# VENDOR INTELLIGENCE VIEW
# ============================================================================

class VendorIntelligenceView(BaseVendorView):
	"""AI-powered vendor intelligence and insights"""
	
	route_base = "/vendor_management/intelligence"
	
	@expose('/')
	@expose('/<vendor_id>')
	@has_access
	def intelligence(self, vendor_id=None):
		"""Show AI-powered vendor intelligence"""
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			if vendor_id:
				intelligence_data = loop.run_until_complete(
					self._get_vendor_intelligence(vendor_id)
				)
				template = 'vendor_intelligence_detail.html'
				title = f"Intelligence: {intelligence_data['vendor'].name}"
			else:
				intelligence_data = loop.run_until_complete(
					self._get_intelligence_overview()
				)
				template = 'vendor_intelligence_overview.html'
				title = "Vendor Intelligence Overview"
			
			return self.render_template(
				template,
				**intelligence_data,
				title=title
			)
			
		except Exception as e:
			flash(f"Error loading intelligence data: {str(e)}", "danger")
			return redirect(url_for('VendorDashboardView.dashboard'))
		finally:
			loop.close()
	
	async def _get_vendor_intelligence(self, vendor_id: str) -> Dict[str, Any]:
		"""Get AI intelligence for specific vendor"""
		
		service = await self._get_vendor_service()
		intelligence_engine = await self._get_intelligence_engine()
		
		vendor = await service.get_vendor_by_id(vendor_id)
		
		# Generate fresh intelligence
		intelligence = await service.generate_vendor_intelligence(vendor_id)
		
		# Get behavior patterns
		behavior_patterns = await intelligence_engine.analyze_vendor_behavior_patterns(vendor_id)
		
		# Get predictive insights
		predictive_insights = await intelligence_engine.generate_predictive_insights(vendor_id)
		
		return {
			'vendor': vendor,
			'intelligence': intelligence,
			'behavior_patterns': behavior_patterns,
			'predictive_insights': predictive_insights
		}
	
	async def _get_intelligence_overview(self) -> Dict[str, Any]:
		"""Get intelligence overview for all vendors"""
		
		service = await self._get_vendor_service()
		analytics = await service.get_vendor_analytics()
		
		return {
			'analytics': analytics
		}


# ============================================================================
# API ENDPOINTS FOR AJAX CALLS
# ============================================================================

class VendorAPIView(BaseVendorView):
	"""API endpoints for AJAX operations"""
	
	route_base = "/api/vendor_management"
	
	@expose('/generate_intelligence/<vendor_id>')
	@has_access
	def generate_intelligence(self, vendor_id):
		"""Generate fresh AI intelligence for vendor"""
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			service = loop.run_until_complete(self._get_vendor_service())
			intelligence = loop.run_until_complete(
				service.generate_vendor_intelligence(vendor_id)
			)
			
			return jsonify({
				'success': True,
				'intelligence_score': float(intelligence.confidence_score * 100),
				'generated_at': intelligence.intelligence_date.isoformat(),
				'patterns_count': len(intelligence.behavior_patterns),
				'insights_count': len(intelligence.predictive_insights)
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
		finally:
			loop.close()
	
	@expose('/optimization_plan/<vendor_id>')
	@has_access  
	def get_optimization_plan(self, vendor_id):
		"""Get optimization plan for vendor"""
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			intelligence_engine = loop.run_until_complete(self._get_intelligence_engine())
			
			objectives = request.args.getlist('objectives') or [
				'performance_improvement', 'cost_reduction', 'risk_mitigation'
			]
			
			optimization_plan = loop.run_until_complete(
				intelligence_engine.generate_optimization_plan(vendor_id, objectives)
			)
			
			return jsonify({
				'success': True,
				'plan': {
					'objectives': optimization_plan.optimization_objectives,
					'actions_count': len(optimization_plan.recommended_actions),
					'predicted_improvement': optimization_plan.predicted_outcomes,
					'created_at': optimization_plan.created_at.isoformat()
				}
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
		finally:
			loop.close()