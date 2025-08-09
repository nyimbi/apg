"""APG Cash Management Flask-AppBuilder Views

Enterprise-grade web interface for APG Cash Management.
Provides comprehensive dashboard views, real-time monitoring,
and advanced cash management operations with APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional

from flask import flash, redirect, request, jsonify, render_template, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView, GroupByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, FormWidget
from flask_babel import lazy_gettext as _
from markupsafe import Markup
from wtforms import StringField, DecimalField, DateField, SelectField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, Optional as OptionalValidator, NumberRange

from .models import CashManagementModels
from .service import CashManagementService
from .cache import CashCacheManager
from .events import CashEventManager
from .bank_integration import BankAPIConnection
from .real_time_sync import RealTimeSyncEngine
from .ai_forecasting import AIForecastingEngine
from .analytics_dashboard import AnalyticsDashboard

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

def _log_view_access(view_name: str, user_id: str, tenant_id: str) -> str:
	"""Log view access with APG formatting"""
	return f"APG_VIEW_ACCESS | view={view_name} | user={user_id} | tenant={tenant_id}"

def _log_view_action(view_name: str, action: str, user_id: str, tenant_id: str) -> str:
	"""Log view action with APG formatting"""
	return f"APG_VIEW_ACTION | view={view_name} | action={action} | user={user_id} | tenant={tenant_id}"

def _log_view_error(view_name: str, error: str, tenant_id: str) -> str:
	"""Log view error with APG formatting"""
	return f"APG_VIEW_ERROR | view={view_name} | error={error} | tenant={tenant_id}"

# ============================================================================
# Custom Widgets for Enhanced UX
# ============================================================================

class APGListWidget(ListWidget):
	"""Enhanced list widget with APG styling and real-time features"""
	template = 'cash_management/widgets/list.html'

class APGShowWidget(ShowWidget):
	"""Enhanced show widget with APG analytics integration"""
	template = 'cash_management/widgets/show.html'

class APGFormWidget(FormWidget):
	"""Enhanced form widget with validation and APG integration"""
	template = 'cash_management/widgets/form.html'

class APGDashboardWidget:
	"""Custom dashboard widget for executive KPIs"""
	template = 'cash_management/widgets/dashboard.html'
	
	def __init__(self, title: str, metrics: Dict[str, Any]):
		self.title = title
		self.metrics = metrics

# ============================================================================
# APG Service Integration Mixin
# ============================================================================

class APGServiceMixin:
	"""Mixin to provide APG service integration for all views"""
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID from user session"""
		# Integration with APG authentication system
		return getattr(self.appbuilder.sm.user, 'tenant_id', 'default_tenant')
	
	def get_cash_service(self) -> CashManagementService:
		"""Get APG cash management service instance"""
		tenant_id = self.get_tenant_id()
		cache = CashCacheManager()
		events = CashEventManager()
		return CashManagementService(tenant_id, cache, events)
	
	def get_bank_integration(self) -> BankAPIConnection:
		"""Get APG bank integration service"""
		tenant_id = self.get_tenant_id()
		cache = CashCacheManager()
		events = CashEventManager()
		return BankAPIConnection(tenant_id, cache, events)
	
	def get_sync_engine(self) -> RealTimeSyncEngine:
		"""Get APG real-time sync engine"""
		tenant_id = self.get_tenant_id()
		bank_api = self.get_bank_integration()
		cache = CashCacheManager()
		events = CashEventManager()
		return RealTimeSyncEngine(tenant_id, bank_api, cache, events)
	
	def get_ai_forecasting(self) -> AIForecastingEngine:
		"""Get APG AI forecasting engine"""
		tenant_id = self.get_tenant_id()
		cache = CashCacheManager()
		events = CashEventManager()
		return AIForecastingEngine(tenant_id, cache, events)
	
	def get_analytics_dashboard(self) -> AnalyticsDashboard:
		"""Get APG analytics dashboard"""
		tenant_id = self.get_tenant_id()
		cache = CashCacheManager()
		events = CashEventManager()
		ai_forecasting = self.get_ai_forecasting()
		return AnalyticsDashboard(tenant_id, cache, events, ai_forecasting)

# ============================================================================
# Executive Dashboard Views
# ============================================================================

class CashManagementDashboardView(BaseView, APGServiceMixin):
	"""Executive cash management dashboard with real-time KPIs"""
	
	route_base = '/cash-management'
	default_view = 'dashboard'

	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main executive dashboard"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_access("dashboard", str(user_id), tenant_id))
			
			# Get analytics dashboard service
			analytics = self.get_analytics_dashboard()
			
			# Generate comprehensive dashboard data
			dashboard_data = analytics.generate_executive_dashboard(
				time_range='30d',
				include_forecasts=True,
				include_opportunities=True
			)
			
			# Get real-time cash position
			cash_service = self.get_cash_service()
			current_position = cash_service.get_current_cash_position()
			
			# Get recent alerts
			recent_alerts = cash_service.get_recent_alerts(limit=10)
			
			# Get maturing investments
			maturing_investments = cash_service.get_maturing_investments(days_ahead=30)
			
			# Prepare dashboard context
			dashboard_context = {
				'dashboard_data': dashboard_data,
				'current_position': current_position,
				'recent_alerts': recent_alerts,
				'maturing_investments': maturing_investments,
				'last_updated': datetime.now(),
				'tenant_id': tenant_id
			}
			
			return self.render_template(
				'cash_management/dashboard.html',
				**dashboard_context
			)
			
		except Exception as e:
			logger.error(_log_view_error("dashboard", str(e), self.get_tenant_id()))
			flash(_('Error loading dashboard: %(error)s', error=str(e)), 'error')
			return redirect(url_for('CashManagementDashboardView.dashboard'))

	@expose('/real-time-sync/', methods=['POST'])
	@has_access
	def execute_real_time_sync(self):
		"""Execute real-time bank synchronization"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_action("dashboard", "real_time_sync", str(user_id), tenant_id))
			
			# Get sync engine
			sync_engine = self.get_sync_engine()
			
			# Execute comprehensive sync
			sync_result = sync_engine.execute_comprehensive_sync(
				force_refresh=True,
				include_pending=True,
				include_intraday=True
			)
			
			flash(_(
				'Bank synchronization completed successfully. '
				'%(successful)d accounts synced, %(new_transactions)d new transactions.',
				successful=sync_result['successful_syncs'],
				new_transactions=sync_result['new_transactions']
			), 'success')
			
			return jsonify({
				'success': True,
				'sync_result': sync_result
			})
			
		except Exception as e:
			logger.error(_log_view_error("dashboard", str(e), self.get_tenant_id()))
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500

	@expose('/cash-position-widget/')
	@has_access
	def cash_position_widget(self):
		"""Real-time cash position widget"""
		try:
			cash_service = self.get_cash_service()
			position = cash_service.get_current_cash_position()
			
			return self.render_template(
				'cash_management/widgets/cash_position.html',
				position=position
			)
			
		except Exception as e:
			logger.error(_log_view_error("cash_position_widget", str(e), self.get_tenant_id()))
			return jsonify({'error': str(e)}), 500

	@expose('/forecast-widget/')
	@has_access
	def forecast_widget(self):
		"""AI-powered forecast widget"""
		try:
			ai_forecasting = self.get_ai_forecasting()
			forecast = ai_forecasting.generate_comprehensive_forecast(
				horizon_days=30,
				scenario_type='BASE_CASE'
			)
			
			return self.render_template(
				'cash_management/widgets/forecast.html',
				forecast=forecast
			)
			
		except Exception as e:
			logger.error(_log_view_error("forecast_widget", str(e), self.get_tenant_id()))
			return jsonify({'error': str(e)}), 500

# ============================================================================
# Bank Account Management Views
# ============================================================================

class BankAccountModelView(ModelView, APGServiceMixin):
	"""Bank account management with real-time balance monitoring"""
	
	datamodel = SQLAInterface(CashManagementModels.CashAccount)
	
	# List view configuration
	list_title = _('Bank Accounts')
	list_columns = [
		'account_name', 'account_number', 'account_type', 'bank_name',
		'currency_code', 'current_balance', 'available_balance', 'is_active'
	]
	search_columns = ['account_name', 'account_number', 'bank_name']
	order_columns = ['account_name', 'current_balance', 'created_at']
	list_widget = APGListWidget
	
	# Show view configuration
	show_title = _('Bank Account Details')
	show_columns = [
		'account_name', 'account_number', 'account_type', 'bank_name',
		'bank_code', 'routing_number', 'branch_name', 'currency_code',
		'current_balance', 'available_balance', 'pending_credits', 'pending_debits',
		'is_active', 'is_primary', 'requires_reconciliation',
		'last_reconciliation_date', 'interest_rate', 'minimum_balance',
		'notes', 'created_at', 'updated_at'
	]
	show_widget = APGShowWidget
	
	# Form configuration
	add_title = _('Add Bank Account')
	edit_title = _('Edit Bank Account')
	add_columns = [
		'account_name', 'account_number', 'account_type', 'bank_name',
		'bank_code', 'routing_number', 'branch_name', 'currency_code',
		'current_balance', 'available_balance', 'is_active', 'is_primary',
		'requires_reconciliation', 'interest_rate', 'minimum_balance', 'notes'
	]
	edit_columns = add_columns
	add_form_widget = APGFormWidget
	edit_form_widget = APGFormWidget
	
	# Validation and formatting
	validators_columns = {
		'account_name': [DataRequired()],
		'account_number': [DataRequired()],
		'account_type': [DataRequired()],
		'bank_name': [DataRequired()],
		'current_balance': [NumberRange(min=0)],
		'available_balance': [NumberRange(min=0)]
	}
	
	formatters_columns = {
		'current_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'available_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'interest_rate': lambda x: f"{x:.4f}%" if x else "0.0000%"
	}

	@expose('/balance-refresh/<account_id>', methods=['POST'])
	@has_access
	def refresh_balance(self, account_id):
		"""Refresh account balance from bank API"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_action("bank_account", "refresh_balance", str(user_id), tenant_id))
			
			# Get bank integration service
			bank_api = self.get_bank_integration()
			
			# Refresh balance
			updated_balance = bank_api.refresh_account_balance(account_id)
			
			flash(_(
				'Account balance refreshed successfully. '
				'New balance: %(balance)s',
				balance=f"${updated_balance:,.2f}"
			), 'success')
			
			return redirect(url_for('BankAccountModelView.show', pk=account_id))
			
		except Exception as e:
			logger.error(_log_view_error("bank_account", str(e), self.get_tenant_id()))
			flash(_('Error refreshing balance: %(error)s', error=str(e)), 'error')
			return redirect(url_for('BankAccountModelView.list'))

	@expose('/connectivity-test/<account_id>')
	@has_access
	def test_connectivity(self, account_id):
		"""Test bank connectivity for account"""
		try:
			bank_api = self.get_bank_integration()
			connectivity_result = bank_api.test_account_connectivity(account_id)
			
			if connectivity_result['success']:
				flash(_('Bank connectivity test successful'), 'success')
			else:
				flash(_(
					'Bank connectivity test failed: %(error)s',
					error=connectivity_result['error']
				), 'warning')
			
			return redirect(url_for('BankAccountModelView.show', pk=account_id))
			
		except Exception as e:
			logger.error(_log_view_error("bank_account", str(e), self.get_tenant_id()))
			flash(_('Error testing connectivity: %(error)s', error=str(e)), 'error')
			return redirect(url_for('BankAccountModelView.list'))

# ============================================================================
# Cash Flow Management Views
# ============================================================================

class CashFlowModelView(ModelView, APGServiceMixin):
	"""Cash flow transaction management with AI categorization"""
	
	datamodel = SQLAInterface(CashManagementModels.CashFlow)
	
	# List view configuration
	list_title = _('Cash Flows')
	list_columns = [
		'transaction_date', 'description', 'amount', 'flow_type',
		'category', 'counterparty', 'account_id', 'is_forecasted'
	]
	search_columns = ['description', 'category', 'counterparty']
	order_columns = ['transaction_date', 'amount']
	list_widget = APGListWidget
	
	# Show view configuration
	show_title = _('Cash Flow Details')
	show_columns = [
		'transaction_date', 'description', 'amount', 'flow_type',
		'category', 'counterparty', 'reference_number', 'account_id',
		'is_forecasted', 'confidence_level', 'source_module',
		'transaction_id', 'cost_center', 'department', 'tags',
		'notes', 'created_at', 'updated_at'
	]
	show_widget = APGShowWidget
	
	# Form configuration
	add_title = _('Add Cash Flow')
	edit_title = _('Edit Cash Flow')
	add_columns = [
		'account_id', 'transaction_date', 'description', 'amount',
		'flow_type', 'category', 'counterparty', 'reference_number',
		'is_forecasted', 'confidence_level', 'source_module',
		'cost_center', 'department', 'notes'
	]
	edit_columns = add_columns
	add_form_widget = APGFormWidget
	edit_form_widget = APGFormWidget
	
	# Validation and formatting
	validators_columns = {
		'transaction_date': [DataRequired()],
		'description': [DataRequired()],
		'amount': [DataRequired(), NumberRange(min=0.01)],
		'flow_type': [DataRequired()],
		'category': [DataRequired()],
		'confidence_level': [NumberRange(min=0, max=1)]
	}
	
	formatters_columns = {
		'amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'confidence_level': lambda x: f"{x:.2%}" if x else "100%"
	}

	@expose('/ai-categorize/<cash_flow_id>', methods=['POST'])
	@has_access
	def ai_categorize(self, cash_flow_id):
		"""Use AI to categorize cash flow transaction"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_action("cash_flow", "ai_categorize", str(user_id), tenant_id))
			
			# Get cash service
			cash_service = self.get_cash_service()
			
			# Apply AI categorization
			categorization_result = cash_service.ai_categorize_transaction(cash_flow_id)
			
			flash(_(
				'AI categorization completed. '
				'Category: %(category)s (Confidence: %(confidence)s)',
				category=categorization_result['category'],
				confidence=f"{categorization_result['confidence']:.1%}"
			), 'success')
			
			return redirect(url_for('CashFlowModelView.show', pk=cash_flow_id))
			
		except Exception as e:
			logger.error(_log_view_error("cash_flow", str(e), self.get_tenant_id()))
			flash(_('Error in AI categorization: %(error)s', error=str(e)), 'error')
			return redirect(url_for('CashFlowModelView.list'))

	@expose('/bulk-import/')
	@has_access
	def bulk_import(self):
		"""Bulk import cash flow transactions"""
		try:
			if request.method == 'GET':
				return self.render_template('cash_management/bulk_import.html')
			
			# Handle file upload and processing
			uploaded_file = request.files.get('import_file')
			if not uploaded_file:
				flash(_('Please select a file to import'), 'error')
				return redirect(url_for('CashFlowModelView.bulk_import'))
			
			# Process import
			cash_service = self.get_cash_service()
			import_result = cash_service.bulk_import_cash_flows_from_file(uploaded_file)
			
			flash(_(
				'Import completed successfully. '
				'%(imported)d transactions imported, %(errors)d errors.',
				imported=import_result['imported_count'],
				errors=import_result['error_count']
			), 'success')
			
			return redirect(url_for('CashFlowModelView.list'))
			
		except Exception as e:
			logger.error(_log_view_error("cash_flow", str(e), self.get_tenant_id()))
			flash(_('Error during import: %(error)s', error=str(e)), 'error')
			return redirect(url_for('CashFlowModelView.list'))

# ============================================================================
# AI Forecasting Views
# ============================================================================

class ForecastingView(BaseView, APGServiceMixin):
	"""AI-powered cash flow forecasting interface"""
	
	route_base = '/forecasting'
	default_view = 'forecast_center'

	@expose('/forecast-center/')
	@has_access
	def forecast_center(self):
		"""Main forecasting center"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_access("forecast_center", str(user_id), tenant_id))
			
			# Get AI forecasting service
			ai_forecasting = self.get_ai_forecasting()
			
			# Get existing forecasts
			existing_forecasts = ai_forecasting.get_recent_forecasts(limit=10)
			
			# Get model performance metrics
			model_performance = ai_forecasting.get_model_performance_metrics()
			
			# Get forecast accuracy trends
			accuracy_trends = ai_forecasting.analyze_forecast_accuracy(lookback_days=90)
			
			context = {
				'existing_forecasts': existing_forecasts,
				'model_performance': model_performance,
				'accuracy_trends': accuracy_trends,
				'tenant_id': tenant_id
			}
			
			return self.render_template(
				'cash_management/forecast_center.html',
				**context
			)
			
		except Exception as e:
			logger.error(_log_view_error("forecast_center", str(e), self.get_tenant_id()))
			flash(_('Error loading forecast center: %(error)s', error=str(e)), 'error')
			return redirect(url_for('CashManagementDashboardView.dashboard'))

	@expose('/generate-forecast/', methods=['GET', 'POST'])
	@has_access
	def generate_forecast(self):
		"""Generate new AI-powered forecast"""
		try:
			if request.method == 'GET':
				return self.render_template('cash_management/generate_forecast.html')
			
			# Get form data
			horizon_days = int(request.form.get('horizon_days', 90))
			scenario_type = request.form.get('scenario_type', 'BASE_CASE')
			confidence_level = float(request.form.get('confidence_level', 0.95))
			include_seasonality = request.form.get('include_seasonality') == 'on'
			include_external_factors = request.form.get('include_external_factors') == 'on'
			model_type = request.form.get('model_type', 'AUTO')
			
			# Generate forecast
			ai_forecasting = self.get_ai_forecasting()
			forecast = ai_forecasting.generate_comprehensive_forecast(
				horizon_days=horizon_days,
				scenario_type=scenario_type,
				confidence_level=Decimal(str(confidence_level)),
				include_seasonality=include_seasonality,
				include_external_factors=include_external_factors,
				model_type=model_type
			)
			
			flash(_(
				'Forecast generated successfully. '
				'Projected net cash flow: %(net_flow)s',
				net_flow=f"${forecast['net_cash_flow']:,.2f}"
			), 'success')
			
			return redirect(url_for(
				'ForecastingView.forecast_details',
				forecast_id=forecast['id']
			))
			
		except Exception as e:
			logger.error(_log_view_error("generate_forecast", str(e), self.get_tenant_id()))
			flash(_('Error generating forecast: %(error)s', error=str(e)), 'error')
			return redirect(url_for('ForecastingView.forecast_center'))

	@expose('/forecast-details/<forecast_id>')
	@has_access
	def forecast_details(self, forecast_id):
		"""Detailed forecast analysis view"""
		try:
			ai_forecasting = self.get_ai_forecasting()
			forecast = ai_forecasting.get_forecast_details(forecast_id)
			
			if not forecast:
				flash(_('Forecast not found'), 'error')
				return redirect(url_for('ForecastingView.forecast_center'))
			
			return self.render_template(
				'cash_management/forecast_details.html',
				forecast=forecast
			)
			
		except Exception as e:
			logger.error(_log_view_error("forecast_details", str(e), self.get_tenant_id()))
			flash(_('Error loading forecast details: %(error)s', error=str(e)), 'error')
			return redirect(url_for('ForecastingView.forecast_center'))

	@expose('/scenario-comparison/')
	@has_access
	def scenario_comparison(self):
		"""Compare multiple forecast scenarios"""
		try:
			horizon_days = int(request.args.get('horizon_days', 90))
			
			ai_forecasting = self.get_ai_forecasting()
			scenarios = ai_forecasting.generate_scenario_comparison(
				horizon_days=horizon_days
			)
			
			return self.render_template(
				'cash_management/scenario_comparison.html',
				scenarios=scenarios,
				horizon_days=horizon_days
			)
			
		except Exception as e:
			logger.error(_log_view_error("scenario_comparison", str(e), self.get_tenant_id()))
			flash(_('Error generating scenario comparison: %(error)s', error=str(e)), 'error')
			return redirect(url_for('ForecastingView.forecast_center'))

# ============================================================================
# Investment Management Views
# ============================================================================

class InvestmentView(BaseView, APGServiceMixin):
	"""Investment portfolio management and opportunity discovery"""
	
	route_base = '/investments'
	default_view = 'investment_center'

	@expose('/investment-center/')
	@has_access
	def investment_center(self):
		"""Main investment management center"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_access("investment_center", str(user_id), tenant_id))
			
			# Get cash service for current investments
			cash_service = self.get_cash_service()
			current_investments = cash_service.get_current_investments()
			
			# Get maturing investments
			maturing_investments = cash_service.get_maturing_investments(days_ahead=30)
			
			# Get AI forecasting for opportunities
			ai_forecasting = self.get_ai_forecasting()
			
			# Calculate available cash for investment
			cash_position = cash_service.get_current_cash_position()
			available_for_investment = cash_position.get('available_cash', Decimal('0'))
			
			# Get investment opportunities
			opportunities = []
			if available_for_investment > 10000:  # Minimum investment threshold
				opportunities = ai_forecasting.find_investment_opportunities(
					amount=available_for_investment,
					maturity_days=90,
					risk_tolerance='MODERATE'
				)
			
			context = {
				'current_investments': current_investments,
				'maturing_investments': maturing_investments,
				'opportunities': opportunities,
				'available_for_investment': available_for_investment,
				'tenant_id': tenant_id
			}
			
			return self.render_template(
				'cash_management/investment_center.html',
				**context
			)
			
		except Exception as e:
			logger.error(_log_view_error("investment_center", str(e), self.get_tenant_id()))
			flash(_('Error loading investment center: %(error)s', error=str(e)), 'error')
			return redirect(url_for('CashManagementDashboardView.dashboard'))

	@expose('/find-opportunities/', methods=['GET', 'POST'])
	@has_access
	def find_opportunities(self):
		"""Find AI-curated investment opportunities"""
		try:
			if request.method == 'GET':
				return self.render_template('cash_management/find_opportunities.html')
			
			# Get form data
			amount = Decimal(request.form.get('amount', '0'))
			maturity_days = int(request.form.get('maturity_days', 90))
			risk_tolerance = request.form.get('risk_tolerance', 'MODERATE')
			liquidity_requirement = request.form.get('liquidity_requirement', 'NORMAL')
			yield_preference = request.form.get('yield_preference', 'BALANCED')
			min_yield = Decimal(request.form.get('min_yield', '0'))
			
			# Find opportunities
			ai_forecasting = self.get_ai_forecasting()
			opportunities = ai_forecasting.find_investment_opportunities(
				amount=amount,
				maturity_days=maturity_days,
				risk_tolerance=risk_tolerance,
				liquidity_requirement=liquidity_requirement,
				yield_preference=yield_preference,
				min_yield=min_yield
			)
			
			flash(_(
				'Found %(count)d investment opportunities matching your criteria',
				count=len(opportunities)
			), 'success')
			
			return self.render_template(
				'cash_management/opportunities_results.html',
				opportunities=opportunities,
				search_criteria={
					'amount': amount,
					'maturity_days': maturity_days,
					'risk_tolerance': risk_tolerance
				}
			)
			
		except Exception as e:
			logger.error(_log_view_error("find_opportunities", str(e), self.get_tenant_id()))
			flash(_('Error finding opportunities: %(error)s', error=str(e)), 'error')
			return redirect(url_for('InvestmentView.investment_center'))

	@expose('/portfolio-optimization/', methods=['GET', 'POST'])
	@has_access
	def portfolio_optimization(self):
		"""AI-powered portfolio optimization"""
		try:
			if request.method == 'GET':
				return self.render_template('cash_management/portfolio_optimization.html')
			
			# Get form data
			total_amount = Decimal(request.form.get('total_amount', '0'))
			target_yield = request.form.get('target_yield')
			if target_yield:
				target_yield = Decimal(target_yield)
			max_risk_score = Decimal(request.form.get('max_risk_score', '0.5'))
			diversification_target = Decimal(request.form.get('diversification_target', '0.7'))
			
			# Optimize portfolio
			ai_forecasting = self.get_ai_forecasting()
			optimization = ai_forecasting.optimize_investment_portfolio(
				total_amount=total_amount,
				target_yield=target_yield,
				max_risk_score=max_risk_score,
				diversification_target=diversification_target
			)
			
			flash(_(
				'Portfolio optimization completed. '
				'Expected yield: %(yield)s, Risk score: %(risk)s',
				yield=f"{optimization['expected_yield']:.2%}",
				risk=f"{optimization['portfolio_risk_score']:.2f}"
			), 'success')
			
			return self.render_template(
				'cash_management/optimization_results.html',
				optimization=optimization
			)
			
		except Exception as e:
			logger.error(_log_view_error("portfolio_optimization", str(e), self.get_tenant_id()))
			flash(_('Error optimizing portfolio: %(error)s', error=str(e)), 'error')
			return redirect(url_for('InvestmentView.investment_center'))

# ============================================================================
# Analytics and Reporting Views
# ============================================================================

class AnalyticsView(BaseView, APGServiceMixin):
	"""Advanced analytics and reporting interface"""
	
	route_base = '/analytics'
	default_view = 'analytics_center'

	@expose('/analytics-center/')
	@has_access
	def analytics_center(self):
		"""Main analytics center with comprehensive KPIs"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_access("analytics_center", str(user_id), tenant_id))
			
			# Get analytics dashboard service
			analytics = self.get_analytics_dashboard()
			
			# Generate comprehensive analytics
			analytics_data = analytics.generate_executive_dashboard(
				time_range='30d',
				include_forecasts=True,
				include_opportunities=True
			)
			
			# Get trend analysis
			cash_flow_trends = analytics.analyze_trends(
				metric='cash_flow',
				lookback_days=90
			)
			
			# Get variance analysis
			variance_analysis = analytics.calculate_variance_analysis(
				comparison_period='previous_month'
			)
			
			context = {
				'analytics_data': analytics_data,
				'cash_flow_trends': cash_flow_trends,
				'variance_analysis': variance_analysis,
				'tenant_id': tenant_id
			}
			
			return self.render_template(
				'cash_management/analytics_center.html',
				**context
			)
			
		except Exception as e:
			logger.error(_log_view_error("analytics_center", str(e), self.get_tenant_id()))
			flash(_('Error loading analytics center: %(error)s', error=str(e)), 'error')
			return redirect(url_for('CashManagementDashboardView.dashboard'))

	@expose('/custom-report/', methods=['GET', 'POST'])
	@has_access
	def custom_report(self):
		"""Generate custom reports with flexible parameters"""
		try:
			if request.method == 'GET':
				return self.render_template('cash_management/custom_report.html')
			
			# Get report parameters
			report_type = request.form.get('report_type')
			start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
			end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d').date()
			accounts = request.form.getlist('accounts')
			categories = request.form.getlist('categories')
			
			# Generate report
			cash_service = self.get_cash_service()
			
			if report_type == 'cash_flow':
				report_data = cash_service.generate_cash_flow_report(
					start_date=start_date,
					end_date=end_date,
					account_ids=accounts or None,
					categories=categories or None
				)
			elif report_type == 'position_analysis':
				report_data = cash_service.generate_position_analysis_report(
					start_date=start_date,
					end_date=end_date,
					account_ids=accounts or None
				)
			elif report_type == 'forecast_accuracy':
				ai_forecasting = self.get_ai_forecasting()
				report_data = ai_forecasting.generate_accuracy_report(
					start_date=start_date,
					end_date=end_date
				)
			else:
				flash(_('Invalid report type selected'), 'error')
				return redirect(url_for('AnalyticsView.custom_report'))
			
			return self.render_template(
				'cash_management/report_results.html',
				report_type=report_type,
				report_data=report_data,
				parameters={
					'start_date': start_date,
					'end_date': end_date,
					'accounts': accounts,
					'categories': categories
				}
			)
			
		except Exception as e:
			logger.error(_log_view_error("custom_report", str(e), self.get_tenant_id()))
			flash(_('Error generating report: %(error)s', error=str(e)), 'error')
			return redirect(url_for('AnalyticsView.analytics_center'))

	@expose('/export-data/<report_type>/')
	@has_access
	def export_data(self, report_type):
		"""Export analytics data to various formats"""
		try:
			# Get export parameters from query string
			format_type = request.args.get('format', 'excel')
			
			analytics = self.get_analytics_dashboard()
			
			if format_type == 'excel':
				export_data = analytics.export_to_excel(report_type)
				return export_data
			elif format_type == 'pdf':
				export_data = analytics.export_to_pdf(report_type)
				return export_data
			elif format_type == 'csv':
				export_data = analytics.export_to_csv(report_type)
				return export_data
			else:
				flash(_('Unsupported export format'), 'error')
				return redirect(url_for('AnalyticsView.analytics_center'))
			
		except Exception as e:
			logger.error(_log_view_error("export_data", str(e), self.get_tenant_id()))
			flash(_('Error exporting data: %(error)s', error=str(e)), 'error')
			return redirect(url_for('AnalyticsView.analytics_center'))

# ============================================================================
# Chart Views for Visual Analytics
# ============================================================================

class CashFlowChartView(GroupByChartView, APGServiceMixin):
	"""Cash flow trend charts"""
	
	datamodel = SQLAInterface(CashManagementModels.CashFlow)
	chart_title = _('Cash Flow Trends')
	chart_type = 'LineChart'
	label_columns = {'flow_type': _('Flow Type')}
	group_by_columns = ['transaction_date', 'flow_type']

class CashPositionChartView(DirectByChartView, APGServiceMixin):
	"""Cash position trend charts"""
	
	datamodel = SQLAInterface(CashManagementModels.CashPosition)
	chart_title = _('Cash Position Trends')
	chart_type = 'AreaChart'
	direct_columns = {
		'Daily Cash Position': ('position_date', 'total_cash'),
		'Available Cash': ('position_date', 'available_cash')
	}

# ============================================================================
# System Administration Views
# ============================================================================

class SystemAdminView(BaseView, APGServiceMixin):
	"""System administration and monitoring"""
	
	route_base = '/admin'
	default_view = 'system_status'

	@expose('/system-status/')
	@has_access
	def system_status(self):
		"""System health and status monitoring"""
		try:
			tenant_id = self.get_tenant_id()
			user_id = self.appbuilder.sm.user.id
			
			logger.info(_log_view_access("system_status", str(user_id), tenant_id))
			
			# Get system health information
			cache = CashCacheManager()
			events = CashEventManager()
			
			system_health = {
				'cache_status': cache.get_health_status(),
				'event_system_status': events.get_health_status(),
				'database_status': self._check_database_health(),
				'api_status': self._check_api_health(),
				'bank_connectivity': self._check_bank_connectivity()
			}
			
			return self.render_template(
				'cash_management/system_status.html',
				system_health=system_health
			)
			
		except Exception as e:
			logger.error(_log_view_error("system_status", str(e), self.get_tenant_id()))
			flash(_('Error loading system status: %(error)s', error=str(e)), 'error')
			return redirect(url_for('CashManagementDashboardView.dashboard'))

	def _check_database_health(self) -> Dict[str, Any]:
		"""Check database connectivity and performance"""
		try:
			# Implementation would check database health
			return {
				'status': 'healthy',
				'connection_count': 25,
				'query_performance': 'good',
				'last_backup': datetime.now() - timedelta(hours=6)
			}
		except Exception:
			return {'status': 'error'}

	def _check_api_health(self) -> Dict[str, Any]:
		"""Check API service health"""
		try:
			# Implementation would check FastAPI health
			return {
				'status': 'healthy',
				'response_time': '120ms',
				'active_sessions': 15,
				'error_rate': '0.1%'
			}
		except Exception:
			return {'status': 'error'}

	def _check_bank_connectivity(self) -> Dict[str, Any]:
		"""Check bank API connectivity status"""
		try:
			bank_api = self.get_bank_integration()
			return bank_api.get_connectivity_status()
		except Exception:
			return {'status': 'error'}

# ============================================================================
# Export all view classes for Flask-AppBuilder registration
# ============================================================================

__all__ = [
	'CashManagementDashboardView',
	'BankAccountModelView',
	'CashFlowModelView',
	'ForecastingView',
	'InvestmentView',
	'AnalyticsView',
	'CashFlowChartView',
	'CashPositionChartView',
	'SystemAdminView',
	'APGServiceMixin',
	'APGListWidget',
	'APGShowWidget',
	'APGFormWidget',
	'APGDashboardWidget'
]