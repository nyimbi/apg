"""
APG Accounts Receivable - Blueprint Tests
Unit tests for Flask-AppBuilder blueprint integration

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal

from flask import Flask
from flask_appbuilder import AppBuilder
from flask_testing import TestCase
from wtforms.form import Form

from uuid_extensions import uuid7str

from ..blueprint import (
	ar_blueprint, ARCustomerWidget, ARInvoiceWidget, ARDashboardWidget,
	CustomerSearchForm, InvoiceSearchForm, CreditAssessmentForm,
	CollectionsOptimizationForm, CashFlowForecastForm,
	ARCustomerModelView, ARCreditAssessmentView, ARDashboardView,
	async_route, get_current_tenant_id, get_current_user_id,
	register_views, register_api_blueprint, create_blueprint,
	register_permissions, get_menu_structure, init_subcapability,
	get_ar_configuration, validate_ar_setup, get_ar_dashboard_widgets,
	get_ar_reports
)
from ..models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus
)


class TestBlueprintConfiguration:
	"""Test blueprint configuration and setup."""
	
	def test_ar_blueprint_configuration(self):
		"""Test AR blueprint is properly configured."""
		assert ar_blueprint.name == 'accounts_receivable'
		assert ar_blueprint.url_prefix == '/ar'
		assert ar_blueprint.template_folder == 'templates'
		assert ar_blueprint.static_folder == 'static'
	
	def test_create_blueprint(self):
		"""Test blueprint creation function."""
		blueprint = create_blueprint()
		
		assert blueprint.name == 'accounts_receivable'
		assert blueprint.url_prefix == '/ar'
		assert blueprint.template_folder == 'templates'
		assert blueprint.static_folder == 'static'


class TestCustomWidgets:
	"""Test custom widgets for AR views."""
	
	def test_ar_customer_widget(self):
		"""Test AR customer widget configuration."""
		widget = ARCustomerWidget()
		assert widget.template == 'ar/widgets/customer_list.html'
	
	def test_ar_invoice_widget(self):
		"""Test AR invoice widget configuration."""
		widget = ARInvoiceWidget()
		assert widget.template == 'ar/widgets/invoice_list.html'
	
	def test_ar_dashboard_widget(self):
		"""Test AR dashboard widget configuration."""
		widget = ARDashboardWidget()
		assert widget.template == 'ar/widgets/dashboard.html'


class TestSearchForms:
	"""Test search form configurations."""
	
	def test_customer_search_form_fields(self):
		"""Test customer search form has correct fields."""
		form = CustomerSearchForm()
		
		# Check form has expected fields
		assert hasattr(form, 'customer_code')
		assert hasattr(form, 'legal_name')
		assert hasattr(form, 'customer_type')
		assert hasattr(form, 'status')
		assert hasattr(form, 'min_outstanding')
		assert hasattr(form, 'max_outstanding')
		
		# Check select field choices
		customer_type_choices = form.customer_type.choices
		assert ('', 'All') in customer_type_choices
		assert len(customer_type_choices) > 1  # Should have enum values
		
		status_choices = form.status.choices
		assert ('', 'All') in status_choices
		assert len(status_choices) > 1  # Should have enum values
	
	def test_invoice_search_form_fields(self):
		"""Test invoice search form has correct fields."""
		form = InvoiceSearchForm()
		
		# Check form has expected fields
		assert hasattr(form, 'invoice_number')
		assert hasattr(form, 'customer_id')
		assert hasattr(form, 'status')
		assert hasattr(form, 'date_from')
		assert hasattr(form, 'date_to')
		assert hasattr(form, 'min_amount')
		assert hasattr(form, 'max_amount')
		
		# Check select field choices
		status_choices = form.status.choices
		assert ('', 'All') in status_choices
		assert len(status_choices) > 1  # Should have enum values
	
	def test_customer_search_form_validation(self):
		"""Test customer search form validation."""
		# Test valid data
		form_data = {
			'customer_code': 'CUST001',
			'legal_name': 'Test Customer',
			'min_outstanding': '1000.00',
			'max_outstanding': '5000.00'
		}
		form = CustomerSearchForm(data=form_data)
		assert form.validate()
		
		# Test invalid decimal
		invalid_form_data = {
			'min_outstanding': 'invalid_decimal'
		}
		invalid_form = CustomerSearchForm(data=invalid_form_data)
		assert not invalid_form.validate()


class TestAIForms:
	"""Test AI-powered form configurations."""
	
	def test_credit_assessment_form_fields(self):
		"""Test credit assessment form has correct fields."""
		form = CreditAssessmentForm()
		
		# Check form has expected fields
		assert hasattr(form, 'customer_id')
		assert hasattr(form, 'assessment_type')
		assert hasattr(form, 'include_explanations')
		assert hasattr(form, 'generate_recommendations')
		assert hasattr(form, 'update_customer_record')
		assert hasattr(form, 'notes')
		
		# Check select field choices
		assessment_choices = form.assessment_type.choices
		expected_choices = [
			('standard', 'Standard Assessment'),
			('comprehensive', 'Comprehensive Assessment'),
			('monitoring', 'Risk Monitoring')
		]
		for choice in expected_choices:
			assert choice in assessment_choices
		
		# Check boolean field defaults
		assert form.include_explanations.default is True
		assert form.generate_recommendations.default is True
		assert form.update_customer_record.default is False
	
	def test_collections_optimization_form_fields(self):
		"""Test collections optimization form has correct fields."""
		form = CollectionsOptimizationForm()
		
		# Check form has expected fields
		assert hasattr(form, 'customer_ids')
		assert hasattr(form, 'optimization_scope')
		assert hasattr(form, 'scenario_type')
		assert hasattr(form, 'include_ai_recommendations')
		assert hasattr(form, 'generate_campaign_plan')
		
		# Check select field choices
		scope_choices = form.optimization_scope.choices
		expected_scopes = [
			('single', 'Single Customer'),
			('batch', 'Multiple Customers'),
			('campaign', 'Campaign Planning')
		]
		for choice in expected_scopes:
			assert choice in scope_choices
		
		scenario_choices = form.scenario_type.choices
		expected_scenarios = [
			('realistic', 'Realistic'),
			('optimistic', 'Optimistic'),
			('pessimistic', 'Pessimistic'),
			('custom', 'Custom')
		]
		for choice in expected_scenarios:
			assert choice in scenario_choices
	
	def test_cash_flow_forecast_form_fields(self):
		"""Test cash flow forecast form has correct fields."""
		form = CashFlowForecastForm()
		
		# Check form has expected fields
		assert hasattr(form, 'forecast_start_date')
		assert hasattr(form, 'forecast_end_date')
		assert hasattr(form, 'forecast_period')
		assert hasattr(form, 'scenario_type')
		assert hasattr(form, 'include_seasonal_trends')
		assert hasattr(form, 'include_external_factors')
		assert hasattr(form, 'confidence_level')
		
		# Check select field choices and defaults
		period_choices = form.forecast_period.choices
		expected_periods = [
			('daily', 'Daily'),
			('weekly', 'Weekly'),
			('monthly', 'Monthly')
		]
		for choice in expected_periods:
			assert choice in period_choices
		
		assert form.forecast_period.default == 'daily'
		
		confidence_choices = form.confidence_level.choices
		expected_confidence = [
			('0.90', '90%'),
			('0.95', '95%'),
			('0.99', '99%')
		]
		for choice in expected_confidence:
			assert choice in confidence_choices
		
		assert form.confidence_level.default == '0.95'


class TestUtilityFunctions:
	"""Test utility functions and decorators."""
	
	def test_async_route_decorator(self):
		"""Test async route decorator functionality."""
		
		@async_route
		async def test_async_function():
			await asyncio.sleep(0.001)  # Minimal async operation
			return "test_result"
		
		# Call the decorated function
		result = test_async_function()
		assert result == "test_result"
	
	@patch('flask.request')
	def test_get_current_tenant_id(self, mock_request):
		"""Test getting current tenant ID from request context."""
		# Test with header present
		mock_request.headers.get.return_value = 'test_tenant_123'
		tenant_id = get_current_tenant_id()
		assert tenant_id == 'test_tenant_123'
		mock_request.headers.get.assert_called_with('X-Tenant-ID', 'default_tenant')
		
		# Test with header absent (default value)
		mock_request.headers.get.return_value = None
		mock_request.headers.get.side_effect = lambda key, default: default if key == 'X-Tenant-ID' else None
		tenant_id = get_current_tenant_id()
		assert tenant_id == 'default_tenant'
	
	@patch('flask.request')
	def test_get_current_user_id(self, mock_request):
		"""Test getting current user ID from request context."""
		# Test with header present
		mock_request.headers.get.return_value = 'user_456'
		user_id = get_current_user_id()
		assert user_id == 'user_456'
		mock_request.headers.get.assert_called_with('X-User-ID', 'system_user')
		
		# Test with header absent (default value)
		mock_request.headers.get.return_value = None
		mock_request.headers.get.side_effect = lambda key, default: default if key == 'X-User-ID' else None
		user_id = get_current_user_id()
		assert user_id == 'system_user'


class TestModelViews:
	"""Test Flask-AppBuilder model view configurations."""
	
	def test_ar_customer_model_view_configuration(self):
		"""Test AR customer model view configuration."""
		view = ARCustomerModelView()
		
		# Check basic configuration
		assert view.list_title == "Accounts Receivable - Customers"
		assert isinstance(view.list_widget, type) and issubclass(view.list_widget, ARCustomerWidget)
		
		# Check list columns
		expected_columns = [
			'customer_code', 'legal_name', 'customer_type', 'status',
			'credit_limit', 'total_outstanding', 'overdue_amount', 'created_at'
		]
		assert view.list_columns == expected_columns
		
		# Check search configuration
		expected_search_columns = ['customer_code', 'legal_name', 'customer_type', 'status']
		assert view.search_columns == expected_search_columns
		assert view.search_form == CustomerSearchForm
		
		# Check show view configuration
		assert view.show_title == "Customer Details"
		assert len(view.show_columns) > 8  # Should have multiple columns
		
		# Check permissions
		expected_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']
		assert view.base_permissions == expected_permissions
	
	def test_ar_customer_model_view_calculate_avg_payment_days(self):
		"""Test average payment days calculation."""
		view = ARCustomerModelView()
		
		# Test simplified implementation
		avg_days = view._calculate_avg_payment_days('test_customer_id')
		assert avg_days == 28.5  # Expected simplified value


class TestAIPoweredViews:
	"""Test AI-powered view configurations."""
	
	def test_ar_credit_assessment_view_configuration(self):
		"""Test AR credit assessment view configuration."""
		view = ARCreditAssessmentView()
		
		assert view.form == CreditAssessmentForm
		assert view.form_title == "AI Credit Assessment"
	
	def test_ar_dashboard_view_configuration(self):
		"""Test AR dashboard view configuration."""
		view = ARDashboardView()
		
		# Check that it's a proper base view
		assert hasattr(view, 'dashboard')  # Should have dashboard method


class TestRegistrationFunctions:
	"""Test view and permission registration functions."""
	
	def test_register_permissions(self):
		"""Test permission registration function."""
		mock_appbuilder = Mock()
		mock_security_manager = Mock()
		mock_appbuilder.sm = mock_security_manager
		
		# Mock find_permission_view_menu to return None (permission doesn't exist)
		mock_security_manager.find_permission_view_menu.return_value = None
		
		register_permissions(mock_appbuilder)
		
		# Check that permissions were created
		assert mock_security_manager.add_permission_view_menu.call_count > 0
		
		# Check specific permission calls
		call_args_list = mock_security_manager.add_permission_view_menu.call_args_list
		permission_calls = [(args[0], args[1]) for args, kwargs in call_args_list]
		
		# Check some expected permissions exist
		expected_permissions = [
			('can_list', 'ARCustomerModelView'),
			('assess_credit', 'ARCustomerModelView'),
			('ar_credit_assessment', 'system'),
			('ar_collections_optimization', 'system'),
			('ar_cashflow_forecast', 'system')
		]
		
		for perm in expected_permissions:
			assert perm in permission_calls
	
	def test_register_views(self):
		"""Test view registration function."""
		mock_appbuilder = Mock()
		mock_app = Mock()
		mock_appbuilder.get_app = mock_app
		
		register_views(mock_appbuilder)
		
		# Check that views were added
		assert mock_appbuilder.add_view.call_count >= 3  # At least 3 views
		
		# Check that blueprint was registered
		mock_app.register_blueprint.assert_called_once_with(ar_blueprint)
	
	def test_register_api_blueprint(self):
		"""Test API blueprint registration."""
		mock_app = Mock()
		
		with patch('..blueprint.create_api_blueprint') as mock_create_api:
			mock_api_bp = Mock()
			mock_create_api.return_value = mock_api_bp
			
			register_api_blueprint(mock_app)
			
			mock_create_api.assert_called_once()
			mock_app.register_blueprint.assert_called_once_with(mock_api_bp)


class TestConfigurationAndStructure:
	"""Test configuration and structure functions."""
	
	def test_get_menu_structure(self):
		"""Test menu structure generation."""
		menu = get_menu_structure()
		
		assert menu['name'] == 'Accounts Receivable'
		assert menu['icon'] == 'fa-money'
		assert 'items' in menu
		assert len(menu['items']) >= 5  # Should have multiple menu items
		
		# Check specific menu items
		menu_items = {item['name']: item for item in menu['items']}
		
		assert 'AR Dashboard' in menu_items
		assert 'Customers' in menu_items
		assert 'Credit Assessment' in menu_items
		assert 'Collections Optimization' in menu_items
		assert 'Cash Flow Forecast' in menu_items
		
		# Check menu item structure
		dashboard_item = menu_items['AR Dashboard']
		assert dashboard_item['icon'] == 'fa-dashboard'
		assert dashboard_item['permission'] == 'ar_dashboard_view'
		assert '/ar/dashboard/' in dashboard_item['href']
	
	def test_get_ar_configuration(self):
		"""Test AR configuration retrieval."""
		config = get_ar_configuration()
		
		assert isinstance(config, dict)
		assert config['ai_features_enabled'] is True
		assert 'credit_scoring_model' in config
		assert 'collections_optimization_model' in config
		assert 'cashflow_forecasting_model' in config
		assert 'performance_targets' in config
		
		# Check performance targets
		targets = config['performance_targets']
		assert 'credit_scoring_accuracy' in targets
		assert 'collections_success_rate' in targets
		assert 'forecast_accuracy_30_day' in targets
		
		assert targets['credit_scoring_accuracy'] == 0.87
		assert targets['collections_success_rate'] == 0.70
		assert targets['forecast_accuracy_30_day'] == 0.90
	
	def test_validate_ar_setup(self):
		"""Test AR setup validation."""
		
		with patch('..blueprint.ARCustomer') as mock_customer:
			mock_query = Mock()
			mock_customer.query.filter_by.return_value = mock_query
			mock_query.count.return_value = 5  # Some customers exist
			
			result = validate_ar_setup('test_tenant')
			
			assert isinstance(result, dict)
			assert 'valid' in result
			assert 'errors' in result
			assert 'warnings' in result
			
			assert result['valid'] is True
			assert len(result['errors']) == 0
			
			# Test with no customers
			mock_query.count.return_value = 0
			result_no_customers = validate_ar_setup('test_tenant')
			assert len(result_no_customers['warnings']) > 0
			assert "No customers configured" in result_no_customers['warnings'][0]
	
	def test_get_ar_dashboard_widgets(self):
		"""Test dashboard widgets configuration."""
		widgets = get_ar_dashboard_widgets()
		
		assert isinstance(widgets, list)
		assert len(widgets) >= 4  # Should have multiple widgets
		
		# Check widget structure
		for widget in widgets:
			assert 'name' in widget
			assert 'widget_type' in widget
			assert 'api_endpoint' in widget
			
			if widget['widget_type'] == 'metric':
				assert 'icon' in widget
				assert 'color' in widget
			elif widget['widget_type'] == 'chart':
				assert 'chart_type' in widget
				assert 'height' in widget
		
		# Check specific widgets
		widget_names = [w['name'] for w in widgets]
		assert 'Total AR Balance' in widget_names
		assert 'AI Credit Assessments' in widget_names
		assert 'Collections Success Rate' in widget_names
		assert 'Cash Flow Forecast' in widget_names
	
	def test_get_ar_reports(self):
		"""Test AR reports configuration."""
		reports = get_ar_reports()
		
		assert isinstance(reports, list)
		assert len(reports) >= 4  # Should have multiple reports
		
		# Check report structure
		for report in reports:
			assert 'name' in report
			assert 'description' in report
			assert 'endpoint' in report
			assert 'parameters' in report
			assert 'formats' in report
			assert isinstance(report['parameters'], list)
			assert isinstance(report['formats'], list)
		
		# Check specific reports
		report_names = [r['name'] for r in reports]
		assert 'AR Dashboard' in report_names
		assert 'Credit Assessment Report' in report_names
		assert 'Collections Optimization' in report_names
		assert 'Cash Flow Forecast' in report_names


class TestInitialization:
	"""Test initialization functions."""
	
	def test_init_subcapability(self):
		"""Test subcapability initialization."""
		mock_appbuilder = Mock()
		
		with patch('..blueprint.register_views') as mock_register_views, \
			 patch('..blueprint.register_permissions') as mock_register_permissions, \
			 patch('..blueprint._init_default_data') as mock_init_data:
			
			init_subcapability(mock_appbuilder)
			
			mock_register_views.assert_called_once_with(mock_appbuilder)
			mock_register_permissions.assert_called_once_with(mock_appbuilder)
			mock_init_data.assert_called_once_with(mock_appbuilder)
	
	def test_init_default_data(self):
		"""Test default data initialization."""
		mock_appbuilder = Mock()
		
		with patch('..blueprint.ARCustomer') as mock_customer:
			mock_query = Mock()
			mock_customer.query.filter_by.return_value = mock_query
			mock_query.count.return_value = 0  # No existing customers
			
			# Import and call the private function
			from ..blueprint import _init_default_data
			
			# Should not raise exception
			_init_default_data(mock_appbuilder)
			
			# Should have checked for existing customers
			mock_customer.query.filter_by.assert_called_with(tenant_id='default_tenant')


class TestIntegrationScenarios:
	"""Test realistic integration scenarios."""
	
	def test_full_blueprint_registration_workflow(self):
		"""Test complete blueprint registration workflow."""
		mock_appbuilder = Mock()
		mock_app = Mock()
		mock_appbuilder.get_app = mock_app
		mock_appbuilder.sm = Mock()
		mock_appbuilder.sm.find_permission_view_menu.return_value = None
		
		# Test complete initialization
		init_subcapability(mock_appbuilder)
		
		# Verify views were registered
		assert mock_appbuilder.add_view.call_count >= 3
		
		# Verify permissions were registered
		assert mock_appbuilder.sm.add_permission_view_menu.call_count > 0
		
		# Verify blueprint was registered
		mock_app.register_blueprint.assert_called_with(ar_blueprint)
	
	def test_ai_features_configuration_validation(self):
		"""Test AI features are properly configured."""
		config = get_ar_configuration()
		
		# Verify AI features are enabled
		assert config['ai_features_enabled'] is True
		
		# Verify all required AI models are configured
		required_models = [
			'credit_scoring_model',
			'collections_optimization_model', 
			'cashflow_forecasting_model'
		]
		
		for model in required_models:
			assert model in config
			assert config[model] is not None
			assert len(config[model]) > 0
		
		# Verify performance targets meet minimum requirements
		targets = config['performance_targets']
		assert targets['credit_scoring_accuracy'] >= 0.85
		assert targets['collections_success_rate'] >= 0.60
		assert targets['forecast_accuracy_30_day'] >= 0.85
	
	def test_menu_and_permissions_alignment(self):
		"""Test menu items align with registered permissions."""
		menu = get_menu_structure()
		
		# Extract required permissions from menu items
		menu_permissions = set()
		for item in menu['items']:
			if 'permission' in item:
				menu_permissions.add(item['permission'])
		
		# Check that all menu permissions are in the registration list
		# This is a basic check - in a real test you'd mock the permission registration
		assert len(menu_permissions) > 0
		
		# Verify key permissions exist
		expected_permissions = [
			'ar_dashboard_view',
			'ar_credit_assessment',
			'ar_collections_optimization',
			'ar_cashflow_forecast'
		]
		
		for perm in expected_permissions:
			assert perm in menu_permissions or any(perm in item.get('permission', '') for item in menu['items'])


# Run tests
if __name__ == "__main__":
	pytest.main([__file__, "-v"])