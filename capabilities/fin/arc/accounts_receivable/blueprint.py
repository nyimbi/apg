"""
APG Accounts Receivable - Flask-AppBuilder Blueprint
Comprehensive web interface for accounts receivable management

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Union
from functools import wraps

from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for, current_app
from flask_appbuilder import BaseView, ModelView, SimpleFormView, expose, has_access, AppBuilder
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget, ShowWidget, FormWidget
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.actions import action
from flask_appbuilder.security.decorators import protect
from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, SelectField, DateField, TextAreaField, IntegerField, BooleanField
from wtforms.validators import DataRequired, NumberRange, Length, Optional as OptionalValidator
from wtforms.widgets import TextArea, Select

from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import joinedload

from apg.core.flask_integration import APGBaseView, APGModelView, APGFormView
from apg.auth_rbac import require_permission
from apg.audit_compliance import audit_action

from .models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity, ARCreditAssessment, ARDispute,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus, ARCollectionPriority
)
from .service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService, 
	ARCashApplicationService, ARAnalyticsService
)
from .ai_credit_scoring import APGCreditScoringService
from .ai_collections_optimization import APGCollectionsAIService
from .ai_cashflow_forecasting import APGCashFlowForecastingService
from .views import (
	ARCustomerCreateView, ARCustomerUpdateView, ARCustomerDetailView,
	ARInvoiceCreateView, ARInvoiceUpdateView, ARInvoiceDetailView,
	ARPaymentCreateView, ARPaymentUpdateView, ARPaymentDetailView,
	CreditScoringRequestView, CollectionsOptimizationView, CashFlowForecastView
)


# =============================================================================
# Blueprint Configuration
# =============================================================================

ar_blueprint = Blueprint(
	'accounts_receivable',
	__name__,
	url_prefix='/ar',
	template_folder='templates',
	static_folder='static'
)


# =============================================================================
# Custom Widgets and Forms
# =============================================================================

class ARCustomerWidget(ListWidget):
	"""Custom widget for AR customer list view."""
	template = 'ar/widgets/customer_list.html'


class ARInvoiceWidget(ListWidget):
	"""Custom widget for AR invoice list view."""
	template = 'ar/widgets/invoice_list.html'


class ARDashboardWidget(ShowWidget):
	"""Custom widget for AR dashboard."""
	template = 'ar/widgets/dashboard.html'


class CustomerSearchForm(FlaskForm):
	"""Customer search form."""
	customer_code = StringField('Customer Code', validators=[OptionalValidator(), Length(max=50)])
	legal_name = StringField('Legal Name', validators=[OptionalValidator(), Length(max=200)])
	customer_type = SelectField('Customer Type', 
		choices=[('', 'All')] + [(t.value, t.value.title()) for t in ARCustomerType],
		validators=[OptionalValidator()]
	)
	status = SelectField('Status',
		choices=[('', 'All')] + [(s.value, s.value.title()) for s in ARCustomerStatus],
		validators=[OptionalValidator()]
	)
	min_outstanding = DecimalField('Min Outstanding', validators=[OptionalValidator(), NumberRange(min=0)])
	max_outstanding = DecimalField('Max Outstanding', validators=[OptionalValidator(), NumberRange(min=0)])


class InvoiceSearchForm(FlaskForm):
	"""Invoice search form."""
	invoice_number = StringField('Invoice Number', validators=[OptionalValidator(), Length(max=50)])
	customer_id = SelectField('Customer', coerce=str, validators=[OptionalValidator()])
	status = SelectField('Status',
		choices=[('', 'All')] + [(s.value, s.value.title()) for s in ARInvoiceStatus],
		validators=[OptionalValidator()]
	)
	date_from = DateField('Date From', validators=[OptionalValidator()])
	date_to = DateField('Date To', validators=[OptionalValidator()])
	min_amount = DecimalField('Min Amount', validators=[OptionalValidator(), NumberRange(min=0)])
	max_amount = DecimalField('Max Amount', validators=[OptionalValidator(), NumberRange(min=0)])


class CreditAssessmentForm(FlaskForm):
	"""Credit assessment request form."""
	customer_id = SelectField('Customer', coerce=str, validators=[DataRequired()])
	assessment_type = SelectField('Assessment Type',
		choices=[
			('standard', 'Standard Assessment'),
			('comprehensive', 'Comprehensive Assessment'),
			('monitoring', 'Risk Monitoring')
		],
		validators=[DataRequired()]
	)
	include_explanations = BooleanField('Include AI Explanations', default=True)
	generate_recommendations = BooleanField('Generate Recommendations', default=True)
	update_customer_record = BooleanField('Update Customer Record', default=False)
	notes = TextAreaField('Assessment Notes', validators=[OptionalValidator(), Length(max=1000)])


class CollectionsOptimizationForm(FlaskForm):
	"""Collections optimization request form."""
	customer_ids = SelectField('Customers', coerce=str, validators=[OptionalValidator()])
	optimization_scope = SelectField('Optimization Scope',
		choices=[
			('single', 'Single Customer'),
			('batch', 'Multiple Customers'),
			('campaign', 'Campaign Planning')
		],
		validators=[DataRequired()]
	)
	scenario_type = SelectField('Scenario Type',
		choices=[
			('realistic', 'Realistic'),
			('optimistic', 'Optimistic'), 
			('pessimistic', 'Pessimistic'),
			('custom', 'Custom')
		],
		validators=[DataRequired()]
	)
	include_ai_recommendations = BooleanField('Include AI Recommendations', default=True)
	generate_campaign_plan = BooleanField('Generate Campaign Plan', default=False)


class CashFlowForecastForm(FlaskForm):
	"""Cash flow forecast request form."""
	forecast_start_date = DateField('Start Date', validators=[DataRequired()], default=date.today)
	forecast_end_date = DateField('End Date', validators=[DataRequired()], 
		default=lambda: date.today() + timedelta(days=30))
	forecast_period = SelectField('Forecast Period',
		choices=[
			('daily', 'Daily'),
			('weekly', 'Weekly'),
			('monthly', 'Monthly')
		],
		validators=[DataRequired()],
		default='daily'
	)
	scenario_type = SelectField('Scenario Type',
		choices=[
			('realistic', 'Realistic'),
			('optimistic', 'Optimistic'),
			('pessimistic', 'Pessimistic'),
			('comparison', 'Multi-Scenario Comparison')
		],
		validators=[DataRequired()],
		default='realistic'
	)
	include_seasonal_trends = BooleanField('Include Seasonal Analysis', default=True)
	include_external_factors = BooleanField('Include Economic Factors', default=True)
	confidence_level = SelectField('Confidence Level',
		choices=[
			('0.90', '90%'),
			('0.95', '95%'),
			('0.99', '99%')
		],
		validators=[DataRequired()],
		default='0.95'
	)


# =============================================================================
# Utility Decorators and Functions
# =============================================================================

def async_route(f):
	"""Decorator to handle async route functions."""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			return loop.run_until_complete(f(*args, **kwargs))
		finally:
			loop.close()
	return decorated_function


def get_current_tenant_id():
	"""Get current tenant ID from session or request context."""
	# This would integrate with APG auth_rbac to get tenant context
	return request.headers.get('X-Tenant-ID', 'default_tenant')


def get_current_user_id():
	"""Get current user ID from session or request context."""
	# This would integrate with APG auth_rbac to get user context
	return request.headers.get('X-User-ID', 'system_user')


# =============================================================================
# Customer Management Views
# =============================================================================

class ARCustomerModelView(APGModelView):
	"""AR Customer management view."""
	
	datamodel = SQLAInterface(ARCustomer)
	
	# List view configuration
	list_title = "Accounts Receivable - Customers"
	list_columns = [
		'customer_code', 'legal_name', 'customer_type', 'status',
		'credit_limit', 'total_outstanding', 'overdue_amount', 'created_at'
	]
	list_widget = ARCustomerWidget
	
	# Search configuration
	search_columns = ['customer_code', 'legal_name', 'customer_type', 'status']
	search_form = CustomerSearchForm
	
	# Show view configuration
	show_title = "Customer Details"
	show_columns = [
		'customer_code', 'legal_name', 'display_name', 'customer_type', 'status',
		'credit_limit', 'payment_terms_days', 'total_outstanding', 'overdue_amount',
		'contact_email', 'contact_phone', 'billing_address',
		'created_at', 'updated_at', 'created_by', 'updated_by'
	]
	
	# Edit/Add form configuration
	add_form = ARCustomerCreateView
	edit_form = ARCustomerUpdateView
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']
	
	@expose('/list/')
	@has_access
	@require_permission('ar_customer_view')
	def list(self):
		"""Customer list view with enhanced filtering."""
		search_form = CustomerSearchForm(request.args)
		
		# Build query with filters
		query = self.datamodel.get_query()
		
		if search_form.customer_code.data:
			query = query.filter(ARCustomer.customer_code.ilike(f"%{search_form.customer_code.data}%"))
		
		if search_form.legal_name.data:
			query = query.filter(ARCustomer.legal_name.ilike(f"%{search_form.legal_name.data}%"))
		
		if search_form.customer_type.data:
			query = query.filter(ARCustomer.customer_type == search_form.customer_type.data)
		
		if search_form.status.data:
			query = query.filter(ARCustomer.status == search_form.status.data)
		
		if search_form.min_outstanding.data:
			query = query.filter(ARCustomer.total_outstanding >= search_form.min_outstanding.data)
		
		if search_form.max_outstanding.data:
			query = query.filter(ARCustomer.total_outstanding <= search_form.max_outstanding.data)
		
		# Add tenant filtering
		tenant_id = get_current_tenant_id()
		query = query.filter(ARCustomer.tenant_id == tenant_id)
		
		# Apply ordering
		query = query.order_by(desc(ARCustomer.created_at))
		
		customers = query.all()
		
		return self.render_template(
			'ar/customer_list.html',
			customers=customers,
			search_form=search_form,
			title=self.list_title
		)
	
	@expose('/show/<pk>')
	@has_access
	@require_permission('ar_customer_view')
	def show(self, pk):
		"""Customer detail view with related data."""
		customer = self.datamodel.get(pk)
		if not customer:
			flash("Customer not found", "error")
			return redirect(url_for('ARCustomerModelView.list'))
		
		# Get related data
		invoices = ARInvoice.query.filter_by(customer_id=pk).order_by(desc(ARInvoice.created_at)).limit(10).all()
		payments = ARPayment.query.filter_by(customer_id=pk).order_by(desc(ARPayment.created_at)).limit(10).all() 
		collections = ARCollectionActivity.query.filter_by(customer_id=pk).order_by(desc(ARCollectionActivity.created_at)).limit(10).all()
		
		# Calculate summary statistics
		total_invoices = ARInvoice.query.filter_by(customer_id=pk).count()
		total_payments = ARPayment.query.filter_by(customer_id=pk).count()
		avg_payment_days = self._calculate_avg_payment_days(pk)
		
		return self.render_template(
			'ar/customer_detail.html',
			customer=customer,
			invoices=invoices,
			payments=payments,
			collections=collections,
			stats={
				'total_invoices': total_invoices,
				'total_payments': total_payments,
				'avg_payment_days': avg_payment_days
			},
			title=f"Customer: {customer.legal_name}"
		)
	
	@action('assess_credit', 'Assess Credit', 'Perform AI credit assessment')
	@require_permission('ar_credit_assessment')
	@async_route
	async def assess_credit(self, items):
		"""Perform AI credit assessment for selected customers."""
		if not items:
			flash("No customers selected", "warning")
			return redirect(url_for('ARCustomerModelView.list'))
		
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		try:
			customer_service = ARCustomerService(tenant_id, user_id)
			results = []
			
			for customer in items:
				assessment = await customer_service.assess_customer_credit(customer.id)
				results.append({
					'customer': customer,
					'assessment': assessment
				})
			
			return self.render_template(
				'ar/credit_assessment_results.html',
				results=results,
				title="Credit Assessment Results"
			)
			
		except Exception as e:
			flash(f"Credit assessment failed: {str(e)}", "error")
			return redirect(url_for('ARCustomerModelView.list'))
	
	@action('optimize_collections', 'Optimize Collections', 'Generate AI collection strategies')
	@require_permission('ar_collections_optimization')
	@async_route
	async def optimize_collections(self, items):
		"""Generate AI-optimized collection strategies."""
		if not items:
			flash("No customers selected", "warning")
			return redirect(url_for('ARCustomerModelView.list'))
		
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		try:
			collections_service = APGCollectionsAIService(tenant_id, user_id, None)
			strategies = await collections_service.batch_optimize_strategies([c.id for c in items])
			
			return self.render_template(
				'ar/collections_optimization_results.html',
				strategies=strategies,
				customers={c.id: c for c in items},
				title="Collections Optimization Results"
			)
			
		except Exception as e:
			flash(f"Collections optimization failed: {str(e)}", "error")
			return redirect(url_for('ARCustomerModelView.list'))
	
	def _calculate_avg_payment_days(self, customer_id):
		"""Calculate average payment days for customer."""
		# This would involve complex SQL query to calculate payment timing
		# Simplified implementation for blueprint demonstration
		return 28.5


# =============================================================================
# AI-Powered Views
# =============================================================================

class ARCreditAssessmentView(APGFormView):
	"""AI Credit Assessment view."""
	
	form = CreditAssessmentForm
	form_title = "AI Credit Assessment"
	
	@expose('/assess', methods=['GET', 'POST'])
	@has_access
	@require_permission('ar_credit_assessment')
	@async_route
	async def assess(self):
		"""Perform AI credit assessment."""
		form = self.form()
		
		# Populate customer choices
		tenant_id = get_current_tenant_id()
		customers = ARCustomer.query.filter_by(tenant_id=tenant_id).all()
		form.customer_id.choices = [(c.id, c.legal_name) for c in customers]
		
		if request.method == 'POST' and form.validate():
			try:
				user_id = get_current_user_id()
				credit_service = APGCreditScoringService(tenant_id, user_id, None)
				
				assessment = await credit_service.assess_customer_credit(
					customer_id=form.customer_id.data,
					assessment_options={
						'include_explanations': form.include_explanations.data,
						'generate_recommendations': form.generate_recommendations.data,
						'update_customer_record': form.update_customer_record.data
					}
				)
				
				return self.render_template(
					'ar/credit_assessment_result.html',
					assessment=assessment,
					customer=ARCustomer.query.get(form.customer_id.data),
					title="Credit Assessment Result"
				)
				
			except Exception as e:
				flash(f"Credit assessment failed: {str(e)}", "error")
		
		return self.render_template(
			'ar/credit_assessment_form.html',
			form=form,
			title=self.form_title
		)


class ARDashboardView(APGBaseView):
	"""AR Dashboard with key metrics and insights."""
	
	@expose('/dashboard')
	@has_access
	@require_permission('ar_dashboard_view')
	@async_route
	async def dashboard(self):
		"""Main AR dashboard with KPIs and analytics."""
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		try:
			analytics_service = ARAnalyticsService(tenant_id, user_id)
			
			# Get dashboard metrics
			metrics = await analytics_service.get_ar_dashboard_metrics()
			aging_analysis = await analytics_service.get_aging_analysis()
			collection_performance = await analytics_service.get_collection_performance_metrics()
			
			# Get recent activities
			recent_invoices = ARInvoice.query.filter_by(tenant_id=tenant_id)\
				.order_by(desc(ARInvoice.created_at)).limit(5).all()
			recent_payments = ARPayment.query.filter_by(tenant_id=tenant_id)\
				.order_by(desc(ARPayment.created_at)).limit(5).all()
			
			return self.render_template(
				'ar/dashboard.html',
				metrics=metrics,
				aging_analysis=aging_analysis,
				collection_performance=collection_performance,
				recent_invoices=recent_invoices,
				recent_payments=recent_payments,
				title="Accounts Receivable Dashboard"
			)
			
		except Exception as e:
			flash(f"Dashboard loading failed: {str(e)}", "error")
			return self.render_template(
				'ar/dashboard_error.html',
				error=str(e),
				title="Dashboard Error"
			)


def register_views(appbuilder: AppBuilder):
	"""Register Accounts Receivable views with Flask-AppBuilder"""
	
	# Core model views
	appbuilder.add_view(
		ARCustomerModelView,
		"Customers",
		icon="fa-users",
		category="Accounts Receivable",
		category_icon="fa-money"
	)
	
	# AI-powered views  
	appbuilder.add_view(
		ARCreditAssessmentView,
		"Credit Assessment",
		icon="fa-calculator",
		category="AR Intelligence"
	)
	
	# Dashboard
	appbuilder.add_view(
		ARDashboardView,
		"AR Dashboard",
		icon="fa-dashboard",
		category="Accounts Receivable"
	)
	
	# Register API blueprint
	appbuilder.get_app.register_blueprint(ar_blueprint)


def register_api_blueprint(app):
	"""Register API blueprint with Flask app"""
	from .api import create_api_blueprint
	api_bp = create_api_blueprint()
	app.register_blueprint(api_bp)


def create_blueprint() -> Blueprint:
	"""Create Flask blueprint for Accounts Receivable"""
	
	ar_bp = Blueprint(
		'accounts_receivable',
		__name__,
		url_prefix='/ar',
		template_folder='templates',
		static_folder='static'
	)
	
	return ar_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register Accounts Receivable permissions"""
	
	permissions = [
		# Core AR permissions
		('can_list', 'ARCustomerModelView'),
		('can_show', 'ARCustomerModelView'),
		('can_add', 'ARCustomerModelView'),
		('can_edit', 'ARCustomerModelView'),
		('can_delete', 'ARCustomerModelView'),
		('assess_credit', 'ARCustomerModelView'),
		('optimize_collections', 'ARCustomerModelView'),
		
		# AI-powered permissions
		('ar_credit_assessment', 'system'),
		('ar_collections_optimization', 'system'),
		('ar_cashflow_forecast', 'system'),
		('ar_dashboard_view', 'system'),
		('ar_analytics_view', 'system'),
		
		# API permissions
		('ar_customer_view', 'system'),
		('ar_invoice_view', 'system'),
		('ar_payment_view', 'system'),
		('ar_collections_view', 'system'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Accounts Receivable"""
	
	return {
		'name': 'Accounts Receivable',
		'icon': 'fa-money',
		'items': [
			{
				'name': 'AR Dashboard',
				'href': '/ar/dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'ar_dashboard_view'
			},
			{
				'name': 'Customers',
				'href': '/arcustomermodelview/list/',
				'icon': 'fa-users',
				'permission': 'can_list on ARCustomerModelView'
			},
			{
				'name': 'Credit Assessment',
				'href': '/ar/assess/',
				'icon': 'fa-calculator',
				'permission': 'ar_credit_assessment'
			},
			{
				'name': 'Collections Optimization',
				'href': '/ar/optimize/',
				'icon': 'fa-bullseye',
				'permission': 'ar_collections_optimization'
			},
			{
				'name': 'Cash Flow Forecast',
				'href': '/ar/forecast/',
				'icon': 'fa-line-chart',
				'permission': 'ar_cashflow_forecast'
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Accounts Receivable sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default AR data if needed"""
	
	try:
		from .models import ARCustomer
		from apg.auth_rbac.models import db
		
		# Check if customers already exist (use a default tenant for now)
		existing_customers = ARCustomer.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_customers == 0:
			print("No existing customers found - would initialize sample data in production")
			
	except Exception as e:
		print(f"Error initializing default AR data: {e}")


def get_ar_configuration():
	"""Get AR configuration settings"""
	
	return {
		'ai_features_enabled': True,
		'credit_scoring_model': 'ar_credit_scoring_v2',
		'collections_optimization_model': 'ar_collections_optimizer_v1',
		'cashflow_forecasting_model': 'ar_cashflow_predictor_v1',
		'performance_targets': {
			'credit_scoring_accuracy': 0.87,
			'collections_success_rate': 0.70,
			'forecast_accuracy_30_day': 0.90
		}
	}


def validate_ar_setup(tenant_id: str) -> dict[str, Any]:
	"""Validate AR setup for a tenant"""
	
	validation_results = {
		'valid': True,
		'errors': [],
		'warnings': []
	}
	
	try:
		from .models import ARCustomer
		
		# Check if customers exist
		customers = ARCustomer.query.filter_by(tenant_id=tenant_id).count()
		if customers == 0:
			validation_results['warnings'].append("No customers configured")
		
		# Check AI service configurations
		config = get_ar_configuration()
		if not config.get('ai_features_enabled'):
			validation_results['warnings'].append("AI features are disabled")
		
	except Exception as e:
		validation_results['errors'].append(f"Validation error: {str(e)}")
		validation_results['valid'] = False
	
	return validation_results


def get_ar_dashboard_widgets():
	"""Get dashboard widgets for AR"""
	
	return [
		{
			'name': 'Total AR Balance',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/total_balance',
			'icon': 'fa-dollar-sign',
			'color': 'primary'
		},
		{
			'name': 'AI Credit Assessments',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/credit_assessments',
			'icon': 'fa-brain',
			'color': 'info'
		},
		{
			'name': 'Collections Success Rate',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/collections_success',
			'icon': 'fa-bullseye',
			'color': 'success'
		},
		{
			'name': 'Cash Flow Forecast',
			'widget_type': 'chart',
			'chart_type': 'line',
			'api_endpoint': '/api/ar/dashboard/cashflow_chart',
			'height': 300
		}
	]


def get_ar_reports():
	"""Get available AR reports"""
	
	return [
		{
			'name': 'AR Dashboard',
			'description': 'Comprehensive accounts receivable dashboard with AI insights',
			'endpoint': '/ar/dashboard/',
			'parameters': [],
			'formats': ['HTML']
		},
		{
			'name': 'Credit Assessment Report',
			'description': 'AI-powered credit risk assessment results',
			'endpoint': '/ar/assess/',
			'parameters': ['customer_id', 'assessment_type'],
			'formats': ['HTML', 'PDF']
		},
		{
			'name': 'Collections Optimization',
			'description': 'AI-optimized collection strategies and campaigns',
			'endpoint': '/ar/optimize/',
			'parameters': ['optimization_scope', 'scenario_type'],
			'formats': ['HTML', 'PDF']
		},
		{
			'name': 'Cash Flow Forecast',
			'description': 'Predictive cash flow forecasting with multiple scenarios',
			'endpoint': '/ar/forecast/',
			'parameters': ['forecast_period', 'scenario_type'],
			'formats': ['HTML', 'Excel']
		}
	]
	
	ar_bp = Blueprint(
		'accounts_receivable',
		__name__,
		url_prefix='/ar',
		template_folder='templates',
		static_folder='static'
	)
	
	return ar_bp


def register_permissions(appbuilder: AppBuilder):
	"""Register Accounts Receivable permissions"""
	
	permissions = [
		# Customer permissions
		('can_list', 'ARCustomerModelView'),
		('can_show', 'ARCustomerModelView'),
		('can_add', 'ARCustomerModelView'),
		('can_edit', 'ARCustomerModelView'),
		('can_delete', 'ARCustomerModelView'),
		('can_place_on_hold', 'ARCustomerModelView'),
		('can_release_hold', 'ARCustomerModelView'),
		('can_customer_summary', 'ARCustomerModelView'),
		
		# Invoice permissions
		('can_list', 'ARInvoiceModelView'),
		('can_show', 'ARInvoiceModelView'),
		('can_add', 'ARInvoiceModelView'),
		('can_edit', 'ARInvoiceModelView'),
		('can_delete', 'ARInvoiceModelView'),
		('can_post_invoice', 'ARInvoiceModelView'),
		('can_hold_invoice', 'ARInvoiceModelView'),
		('can_invoice_lines', 'ARInvoiceModelView'),
		
		# Payment permissions
		('can_list', 'ARPaymentModelView'),
		('can_show', 'ARPaymentModelView'),
		('can_add', 'ARPaymentModelView'),
		('can_edit', 'ARPaymentModelView'),
		('can_delete', 'ARPaymentModelView'),
		('can_post_payment', 'ARPaymentModelView'),
		('can_auto_apply', 'ARPaymentModelView'),
		('can_void_payment', 'ARPaymentModelView'),
		('can_payment_lines', 'ARPaymentModelView'),
		
		# Credit Memo permissions
		('can_list', 'ARCreditMemoModelView'),
		('can_show', 'ARCreditMemoModelView'),
		('can_add', 'ARCreditMemoModelView'),
		('can_edit', 'ARCreditMemoModelView'),
		('can_delete', 'ARCreditMemoModelView'),
		('can_post_credit_memo', 'ARCreditMemoModelView'),
		
		# Statement permissions
		('can_list', 'ARStatementModelView'),
		('can_show', 'ARStatementModelView'),
		('can_add', 'ARStatementModelView'),
		('can_edit', 'ARStatementModelView'),
		('can_delete', 'ARStatementModelView'),
		('can_generate_statements', 'ARStatementModelView'),
		
		# Collection permissions
		('can_list', 'ARCollectionModelView'),
		('can_show', 'ARCollectionModelView'),
		('can_add', 'ARCollectionModelView'),
		('can_edit', 'ARCollectionModelView'),
		('can_delete', 'ARCollectionModelView'),
		('can_mark_promise_kept', 'ARCollectionModelView'),
		('can_mark_promise_broken', 'ARCollectionModelView'),
		
		# Recurring Billing permissions
		('can_list', 'ARRecurringBillingModelView'),
		('can_show', 'ARRecurringBillingModelView'),
		('can_add', 'ARRecurringBillingModelView'),
		('can_edit', 'ARRecurringBillingModelView'),
		('can_delete', 'ARRecurringBillingModelView'),
		('can_pause_billing', 'ARRecurringBillingModelView'),
		('can_resume_billing', 'ARRecurringBillingModelView'),
		('can_process_billing', 'ARRecurringBillingModelView'),
		
		# Tax Code permissions
		('can_list', 'ARTaxCodeModelView'),
		('can_show', 'ARTaxCodeModelView'),
		('can_add', 'ARTaxCodeModelView'),
		('can_edit', 'ARTaxCodeModelView'),
		('can_delete', 'ARTaxCodeModelView'),
		
		# Report permissions
		('can_index', 'ARAgingView'),
		('can_export', 'ARAgingView'),
		('can_index', 'ARDashboardView'),
		('can_api_summary', 'ARDashboardView'),
		('can_api_cash_flow', 'ARDashboardView'),
		
		# API permissions
		('can_get_list', 'ARCustomerApi'),
		('can_get_customer', 'ARCustomerApi'),
		('can_create_customer', 'ARCustomerApi'),
		('can_update_customer', 'ARCustomerApi'),
		('can_get_customer_summary', 'ARCustomerApi'),
		('can_get_customer_invoices', 'ARCustomerApi'),
		('can_get_customer_payments', 'ARCustomerApi'),
		('can_place_customer_on_hold', 'ARCustomerApi'),
		('can_release_customer_hold', 'ARCustomerApi'),
		
		('can_get_list', 'ARInvoiceApi'),
		('can_get_invoice', 'ARInvoiceApi'),
		('can_create_invoice', 'ARInvoiceApi'),
		('can_update_invoice', 'ARInvoiceApi'),
		('can_post_invoice', 'ARInvoiceApi'),
		('can_get_invoice_lines', 'ARInvoiceApi'),
		('can_add_invoice_line', 'ARInvoiceApi'),
		('can_update_invoice_line', 'ARInvoiceApi'),
		('can_delete_invoice_line', 'ARInvoiceApi'),
		
		('can_get_list', 'ARPaymentApi'),
		('can_get_payment', 'ARPaymentApi'),
		('can_create_payment', 'ARPaymentApi'),
		('can_update_payment', 'ARPaymentApi'),
		('can_post_payment', 'ARPaymentApi'),
		('can_auto_apply_payment', 'ARPaymentApi'),
		('can_void_payment', 'ARPaymentApi'),
		('can_get_payment_lines', 'ARPaymentApi'),
		('can_add_payment_line', 'ARPaymentApi'),
		
		('can_get_list', 'ARCreditMemoApi'),
		('can_get_credit_memo', 'ARCreditMemoApi'),
		('can_create_credit_memo', 'ARCreditMemoApi'),
		('can_update_credit_memo', 'ARCreditMemoApi'),
		('can_post_credit_memo', 'ARCreditMemoApi'),
		('can_get_credit_memo_lines', 'ARCreditMemoApi'),
		
		('can_get_list', 'ARStatementApi'),
		('can_get_statement', 'ARStatementApi'),
		('can_generate_statement', 'ARStatementApi'),
		('can_generate_batch_statements', 'ARStatementApi'),
		
		('can_get_list', 'ARCollectionApi'),
		('can_get_collection', 'ARCollectionApi'),
		('can_create_collection', 'ARCollectionApi'),
		('can_update_collection', 'ARCollectionApi'),
		('can_get_customers_for_collections', 'ARCollectionApi'),
		('can_generate_dunning_letters', 'ARCollectionApi'),
		
		('can_get_list', 'ARRecurringBillingApi'),
		('can_get_recurring_billing', 'ARRecurringBillingApi'),
		('can_create_recurring_billing', 'ARRecurringBillingApi'),
		('can_update_recurring_billing', 'ARRecurringBillingApi'),
		('can_process_recurring_billing', 'ARRecurringBillingApi'),
		('can_pause_recurring_billing', 'ARRecurringBillingApi'),
		('can_resume_recurring_billing', 'ARRecurringBillingApi'),
		
		('can_get_list', 'ARTaxCodeApi'),
		('can_get_tax_code', 'ARTaxCodeApi'),
		('can_create_tax_code', 'ARTaxCodeApi'),
		('can_update_tax_code', 'ARTaxCodeApi'),
		
		('can_get_aging_report', 'ARAgingApi'),
		('can_get_aging_summary', 'ARAgingApi'),
		('can_generate_aging_report', 'ARAgingApi'),
		
		('can_get_dashboard_summary', 'ARDashboardApi'),
		('can_get_cash_flow_projection', 'ARDashboardApi'),
		('can_get_collection_dashboard', 'ARDashboardApi'),
	]
	
	# Create permissions if they don't exist
	for permission_name, view_name in permissions:
		perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
		if not perm:
			appbuilder.sm.add_permission_view_menu(permission_name, view_name)


def get_menu_structure():
	"""Get menu structure for Accounts Receivable"""
	
	return {
		'name': 'Accounts Receivable',
		'icon': 'fa-money-check-alt',
		'items': [
			{
				'name': 'AR Dashboard',
				'href': '/ar_dashboard/',
				'icon': 'fa-dashboard',
				'permission': 'can_index on ARDashboardView'
			},
			{
				'name': 'Customers',
				'href': '/arcustomermodelview/list/',
				'icon': 'fa-users',
				'permission': 'can_list on ARCustomerModelView'
			},
			{
				'name': 'Customer Invoices',
				'href': '/arinvoicemodelview/list/',
				'icon': 'fa-file-invoice-dollar',
				'permission': 'can_list on ARInvoiceModelView'
			},
			{
				'name': 'Customer Payments',
				'href': '/arpaymentmodelview/list/',
				'icon': 'fa-money-check',
				'permission': 'can_list on ARPaymentModelView'
			},
			{
				'name': 'Credit Memos',
				'href': '/arcreditmemomodelview/list/',
				'icon': 'fa-undo',
				'permission': 'can_list on ARCreditMemoModelView'
			},
			{
				'name': 'Customer Statements',
				'href': '/arstatementmodelview/list/',
				'icon': 'fa-file-alt',
				'permission': 'can_list on ARStatementModelView'
			},
			{
				'name': 'Collections',
				'href': '/arcollectionmodelview/list/',
				'icon': 'fa-phone',
				'permission': 'can_list on ARCollectionModelView'
			},
			{
				'name': 'Recurring Billing',
				'href': '/arrecurringbillingmodelview/list/',
				'icon': 'fa-repeat',
				'permission': 'can_list on ARRecurringBillingModelView'
			},
			{
				'name': 'AR Aging Report',
				'href': '/ar_aging/',
				'icon': 'fa-calendar-alt',
				'permission': 'can_index on ARAgingView'
			},
			{
				'name': 'AR Tax Codes',
				'href': '/artaxcodemodelview/list/',
				'icon': 'fa-percentage',
				'permission': 'can_list on ARTaxCodeModelView'
			}
		]
	}


def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Accounts Receivable sub-capability"""
	
	# Register views
	register_views(appbuilder)
	
	# Register permissions
	register_permissions(appbuilder)
	
	# Initialize default data if needed
	_init_default_data(appbuilder)


def _init_default_data(appbuilder: AppBuilder):
	"""Initialize default AR data if needed"""
	
	from .models import CFARTaxCode
	from ...auth_rbac.models import db
	from . import get_default_tax_codes
	
	try:
		# Check if tax codes already exist (use a default tenant for now)
		existing_tax_codes = CFARTaxCode.query.filter_by(tenant_id='default_tenant').count()
		
		if existing_tax_codes == 0:
			# Create default tax codes
			default_tax_codes = get_default_tax_codes()
			
			for tax_data in default_tax_codes:
				tax_code = CFARTaxCode(
					tenant_id='default_tenant',
					code=tax_data['code'],
					name=tax_data['name'],
					description=tax_data['description'],
					tax_rate=tax_data['rate'],
					is_active=tax_data['is_active']
				)
				db.session.add(tax_code)
			
			db.session.commit()
			print("Default AR tax codes created")
			
	except Exception as e:
		print(f"Error initializing default AR data: {e}")
		db.session.rollback()


def create_default_customers(tenant_id: str, appbuilder: AppBuilder):
	"""Create default customers for a tenant"""
	
	from .service import AccountsReceivableService
	from . import get_default_customer_types
	
	try:
		ar_service = AccountsReceivableService(tenant_id)
		
		# Check if customers already exist
		existing_customers = ar_service.get_customers()
		if len(existing_customers) > 0:
			return
		
		# Create sample customers for each type
		customer_types = get_default_customer_types()
		
		sample_customers = [
			{
				'customer_number': '001001',
				'customer_name': 'ABC Manufacturing Corp',
				'customer_type': 'CORPORATE',
				'contact_name': 'John Smith',
				'email': 'accounting@abcmanufacturing.com',
				'phone': '555-0101',
				'billing_address_line1': '123 Business Ave',
				'billing_city': 'Business City',
				'billing_state_province': 'BC',
				'billing_postal_code': '12345',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_30',
				'payment_method': 'ACH',
				'credit_limit': 50000.00,
				'is_active': True
			},
			{
				'customer_number': '002001',
				'customer_name': 'XYZ Retail Stores',
				'customer_type': 'RETAIL',
				'contact_name': 'Jane Doe',
				'email': 'payments@xyzretail.com',
				'phone': '555-0202',
				'billing_address_line1': '456 Commerce St',
				'billing_city': 'Commerce City',
				'billing_state_province': 'CC',
				'billing_postal_code': '67890',
				'billing_country': 'USA',
				'payment_terms_code': '2_10_NET_30',
				'payment_method': 'CHECK',
				'credit_limit': 25000.00,
				'is_active': True
			},
			{
				'customer_number': '003001',
				'customer_name': 'Global Distributors Inc',
				'customer_type': 'WHOLESALE',
				'contact_name': 'Mike Johnson',
				'email': 'ap@globaldist.com',
				'phone': '555-0303',
				'billing_address_line1': '789 Distribution Blvd',
				'billing_city': 'Distribution Center',
				'billing_state_province': 'DC',
				'billing_postal_code': '13579',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_60',
				'payment_method': 'WIRE',
				'credit_limit': 100000.00,
				'is_active': True
			},
			{
				'customer_number': '004001',
				'customer_name': 'City Government',
				'customer_type': 'GOVERNMENT',
				'contact_name': 'Sarah Wilson',
				'email': 'procurement@citygovt.gov',
				'phone': '555-0404',
				'billing_address_line1': '101 City Hall Plaza',
				'billing_city': 'Government City',
				'billing_state_province': 'GC',
				'billing_postal_code': '24680',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_30',
				'payment_method': 'ACH',
				'credit_limit': 75000.00,
				'tax_exempt': True,
				'tax_exempt_number': 'EXEMPT-001',
				'is_active': True
			},
			{
				'customer_number': '005001',
				'customer_name': 'Community Foundation',
				'customer_type': 'NONPROFIT',
				'contact_name': 'David Brown',
				'email': 'finance@communityfoundation.org',
				'phone': '555-0505',
				'billing_address_line1': '202 Charity Lane',
				'billing_city': 'Foundation City',
				'billing_state_province': 'FC',
				'billing_postal_code': '35791',
				'billing_country': 'USA',
				'payment_terms_code': 'NET_30',
				'payment_method': 'CHECK',
				'credit_limit': 15000.00,
				'tax_exempt': True,
				'tax_exempt_number': 'NONPROFIT-501C3',
				'is_active': True
			}
		]
		
		for customer_data in sample_customers:
			ar_service.create_customer(customer_data)
		
		print(f"Default customers created for tenant {tenant_id}")
		
	except Exception as e:
		print(f"Error creating default customers: {e}")


def setup_ar_integration(appbuilder: AppBuilder):
	"""Set up AR integration with other modules"""
	
	try:
		# Set up GL integration
		from ..general_ledger.models import CFGLAccount
		from ...auth_rbac.models import db
		from . import get_default_gl_account_mappings
		
		# Ensure required GL accounts exist
		gl_mappings = get_default_gl_account_mappings()
		
		for account_type, account_code in gl_mappings.items():
			existing_account = CFGLAccount.query.filter_by(
				tenant_id='default_tenant',
				account_code=account_code
			).first()
			
			if not existing_account:
				print(f"Warning: GL account {account_code} for {account_type} not found")
		
		print("AR-GL integration check completed")
		
	except Exception as e:
		print(f"Error setting up AR integration: {e}")


def get_ar_configuration():
	"""Get AR configuration settings"""
	
	from . import SUBCAPABILITY_META
	
	return SUBCAPABILITY_META['configuration']


def validate_ar_setup(tenant_id: str) -> dict[str, Any]:
	"""Validate AR setup for a tenant"""
	
	from .service import AccountsReceivableService
	from ..general_ledger.models import CFGLAccount
	from . import get_default_gl_account_mappings
	
	validation_results = {
		'valid': True,
		'errors': [],
		'warnings': []
	}
	
	try:
		ar_service = AccountsReceivableService(tenant_id)
		
		# Check if required GL accounts exist
		gl_mappings = get_default_gl_account_mappings()
		missing_accounts = []
		
		for account_type, account_code in gl_mappings.items():
			account = CFGLAccount.query.filter_by(
				tenant_id=tenant_id,
				account_code=account_code
			).first()
			
			if not account:
				missing_accounts.append(f"{account_type} ({account_code})")
		
		if missing_accounts:
			validation_results['errors'].append(
				f"Missing required GL accounts: {', '.join(missing_accounts)}"
			)
			validation_results['valid'] = False
		
		# Check if customers exist
		customers = ar_service.get_customers()
		if len(customers) == 0:
			validation_results['warnings'].append("No customers configured")
		
		# Check tax codes
		from .models import CFARTaxCode
		tax_codes = CFARTaxCode.query.filter_by(tenant_id=tenant_id).count()
		if tax_codes == 0:
			validation_results['warnings'].append("No tax codes configured")
		
		# Check payment terms configuration
		payment_terms_count = len([c for c in customers if c.payment_terms_code])
		if payment_terms_count == 0 and len(customers) > 0:
			validation_results['warnings'].append("No payment terms configured for customers")
		
	except Exception as e:
		validation_results['errors'].append(f"Validation error: {str(e)}")
		validation_results['valid'] = False
	
	return validation_results


def get_ar_dashboard_widgets():
	"""Get dashboard widgets for AR"""
	
	return [
		{
			'name': 'Total AR Balance',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/total_balance',
			'icon': 'fa-dollar-sign',
			'color': 'primary'
		},
		{
			'name': 'Past Due Amount',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/past_due',
			'icon': 'fa-exclamation-triangle',
			'color': 'warning'
		},
		{
			'name': 'Collection Required',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/collection_required',
			'icon': 'fa-phone',
			'color': 'danger'
		},
		{
			'name': 'Current Month Sales',
			'widget_type': 'metric',
			'api_endpoint': '/api/ar/dashboard/current_month_sales',
			'icon': 'fa-chart-line',
			'color': 'success'
		},
		{
			'name': 'AR Aging',
			'widget_type': 'chart',
			'chart_type': 'pie',
			'api_endpoint': '/api/ar/dashboard/aging_chart',
			'height': 300
		},
		{
			'name': 'Cash Flow Projection',
			'widget_type': 'chart',
			'chart_type': 'bar',
			'api_endpoint': '/api/ar/dashboard/cash_flow_chart',
			'height': 250
		}
	]


def get_ar_reports():
	"""Get available AR reports"""
	
	return [
		{
			'name': 'AR Aging Report',
			'description': 'Customer aging analysis by due date buckets',
			'endpoint': '/ar_aging/',
			'parameters': ['as_of_date'],
			'formats': ['HTML', 'PDF', 'Excel']
		},
		{
			'name': 'Customer Statement',
			'description': 'Individual customer account statement',
			'endpoint': '/api/ar/reports/customer_statement',
			'parameters': ['customer_id', 'statement_date'],
			'formats': ['PDF']
		},
		{
			'name': 'Sales Analysis',
			'description': 'Sales analysis by customer, product, territory',
			'endpoint': '/api/ar/reports/sales_analysis',
			'parameters': ['date_from', 'date_to', 'group_by'],
			'formats': ['HTML', 'Excel']
		},
		{
			'name': 'Collection Report',
			'description': 'Collection activities and effectiveness',
			'endpoint': '/api/ar/reports/collection_report',
			'parameters': ['date_from', 'date_to'],
			'formats': ['HTML', 'PDF']
		},
		{
			'name': 'Cash Receipts Journal',
			'description': 'Detailed cash receipts for a period',
			'endpoint': '/api/ar/reports/cash_receipts',
			'parameters': ['date_from', 'date_to'],
			'formats': ['HTML', 'Excel']
		},
		{
			'name': 'Invoice Register',
			'description': 'List of invoices for a period',
			'endpoint': '/api/ar/reports/invoice_register',
			'parameters': ['date_from', 'date_to'],
			'formats': ['HTML', 'Excel']
		}
	]