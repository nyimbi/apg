#!/usr/bin/env python3
"""
Unit Tests for Flask-AppBuilder Views
Tests for DVRL web interface components

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from flask import Flask
from flask_appbuilder import AppBuilder
from werkzeug.test import Client
from werkzeug.wrappers import Response
from datetime import datetime, timezone

from capabilities.common.dvrl.views import (
	DVRLDashboardView, DataSourceForm, NaturalLanguageQueryForm, 
	SQLQueryForm, dvrl_bp
)
from capabilities.common.dvrl.models import DataSource, DataSourceType, DataSourceStatus


class TestDataSourceForm:
	"""Test suite for Data Source Form"""
	
	def test_form_validation_success(self):
		"""Test successful form validation"""
		form_data = {
			'name': 'Test PostgreSQL',
			'type': 'postgresql',
			'host': 'localhost',
			'port': '5432',
			'database': 'testdb',
			'username': 'testuser',
			'password': 'testpass',
			'description': 'Test database connection'
		}
		
		form = DataSourceForm(data=form_data)
		
		assert form.validate() is True
		assert form.name.data == 'Test PostgreSQL'
		assert form.type.data == 'postgresql'
		assert form.host.data == 'localhost'
	
	def test_form_validation_missing_required(self):
		"""Test form validation with missing required fields"""
		form_data = {
			'host': 'localhost',  # Missing name and type
			'port': '5432'
		}
		
		form = DataSourceForm(data=form_data)
		
		assert form.validate() is False
		assert 'name' in form.errors
		assert 'type' in form.errors
	
	def test_form_validation_invalid_name_length(self):
		"""Test form validation with invalid name length"""
		form_data = {
			'name': '',  # Empty name
			'type': 'postgresql'
		}
		
		form = DataSourceForm(data=form_data)
		
		assert form.validate() is False
		assert 'name' in form.errors


class TestNaturalLanguageQueryForm:
	"""Test suite for Natural Language Query Form"""
	
	def test_form_validation_success(self):
		"""Test successful NL form validation"""
		form_data = {
			'query': 'Show me all users created last week',
			'data_sources': 'users_db, orders_db'
		}
		
		form = NaturalLanguageQueryForm(data=form_data)
		
		assert form.validate() is True
		assert form.query.data == 'Show me all users created last week'
		assert form.data_sources.data == 'users_db, orders_db'
	
	def test_form_validation_query_too_short(self):
		"""Test form validation with query too short"""
		form_data = {
			'query': 'Hi'  # Too short
		}
		
		form = NaturalLanguageQueryForm(data=form_data)
		
		assert form.validate() is False
		assert 'query' in form.errors
	
	def test_form_validation_query_too_long(self):
		"""Test form validation with query too long"""
		form_data = {
			'query': 'A' * 1001  # Too long
		}
		
		form = NaturalLanguageQueryForm(data=form_data)
		
		assert form.validate() is False
		assert 'query' in form.errors


class TestSQLQueryForm:
	"""Test suite for SQL Query Form"""
	
	def test_form_validation_success(self):
		"""Test successful SQL form validation"""
		form_data = {
			'sql': 'SELECT * FROM users WHERE created_at > CURRENT_DATE - INTERVAL 7 DAY',
			'parameters': '{"limit": 100, "offset": 0}'
		}
		
		form = SQLQueryForm(data=form_data)
		
		assert form.validate() is True
		assert 'SELECT' in form.sql.data
		assert form.parameters.data == '{"limit": 100, "offset": 0}'
	
	def test_form_validation_sql_too_short(self):
		"""Test form validation with SQL too short"""
		form_data = {
			'sql': 'SEL'  # Too short
		}
		
		form = SQLQueryForm(data=form_data)
		
		assert form.validate() is False
		assert 'sql' in form.errors


class TestDVRLDashboardView:
	"""Test suite for DVRL Dashboard View"""
	
	@pytest.fixture
	def app(self):
		"""Create test Flask application"""
		app = Flask(__name__)
		app.config['SECRET_KEY'] = 'test_secret_key'
		app.config['TESTING'] = True
		return app
	
	@pytest.fixture
	def mock_dvrl_service(self):
		"""Create mock DVRL service"""
		service = Mock()
		service.data_sources = {}
		service.query_history = []
		
		# Mock async methods
		service.get_health_status = AsyncMock(return_value={
			'status': 'healthy',
			'uptime': '1d 2h 3m',
			'components': {
				'database': 'healthy',
				'connectors': 'healthy',
				'api': 'healthy'
			}
		})
		
		service.get_performance_metrics = AsyncMock(return_value={
			'queries_per_second': 10.5,
			'avg_response_time': 150.2,
			'connector_framework': {
				'total_connectors': 5,
				'active_connectors': 4
			}
		})
		
		service.register_data_source = AsyncMock()
		service.execute_natural_language_query = AsyncMock()
		service.execute_federated_query = AsyncMock()
		
		return service
	
	@pytest.fixture
	def dashboard_view(self, mock_dvrl_service):
		"""Create dashboard view instance"""
		return DVRLDashboardView(mock_dvrl_service)
	
	def test_run_async_helper(self, dashboard_view):
		"""Test async helper function"""
		async def test_coro():
			return "test_result"
		
		result = dashboard_view._run_async(test_coro())
		
		assert result == "test_result"
	
	def test_dashboard_success(self, app, dashboard_view):
		"""Test successful dashboard page load"""
		with app.test_request_context():
			with patch.object(dashboard_view, 'render_template', return_value='dashboard_html'):
				result = dashboard_view.dashboard()
				
				assert result == 'dashboard_html'
				dashboard_view.dvrl_service.get_health_status.assert_called_once()
				dashboard_view.dvrl_service.get_performance_metrics.assert_called_once()
	
	def test_dashboard_error_handling(self, app, dashboard_view):
		"""Test dashboard error handling"""
		# Mock service to raise exception
		dashboard_view.dvrl_service.get_health_status.side_effect = Exception("Service unavailable")
		
		with app.test_request_context():
			with patch.object(dashboard_view, 'render_template', return_value='error_html'):
				result = dashboard_view.dashboard()
				
				assert result == 'error_html'
	
	def test_data_sources_success(self, app, dashboard_view):
		"""Test successful data sources page load"""
		# Mock data sources
		mock_source = Mock()
		mock_source.id = 'source1'
		mock_source.name = 'Test DB'
		mock_source.type.value = 'postgresql'
		mock_source.status.value = 'active'
		mock_source.query_count = 10
		mock_source.created_at = datetime.now()
		
		dashboard_view.dvrl_service.data_sources = {'source1': mock_source}
		
		# Mock connector manager
		mock_connector = Mock()
		mock_connector.get_connection_stats = AsyncMock(return_value={
			'health_status': 'healthy',
			'capabilities': ['read', 'write']
		})
		
		dashboard_view.dvrl_service.connector_manager.get_connector = AsyncMock(return_value=mock_connector)
		
		with app.test_request_context():
			with patch.object(dashboard_view, 'render_template', return_value='data_sources_html'):
				result = dashboard_view.data_sources()
				
				assert result == 'data_sources_html'
	
	def test_query_interface(self, app, dashboard_view):
		"""Test query interface page"""
		with app.test_request_context():
			with patch.object(dashboard_view, 'render_template', return_value='query_interface_html'):
				result = dashboard_view.query_interface()
				
				assert result == 'query_interface_html'
	
	def test_execute_nl_query_success(self, app, dashboard_view):
		"""Test successful natural language query execution"""
		# Mock NL query result
		dashboard_view.dvrl_service.execute_natural_language_query.return_value = {
			'query_id': 'q123',
			'generated_sql': 'SELECT COUNT(*) FROM users',
			'confidence': 0.9,
			'results': [{'count': 42}],
			'explanation': 'This counts all users'
		}
		
		form_data = {
			'query': 'How many users are there?',
			'data_sources': ''
		}
		
		with app.test_request_context(method='POST', data=form_data):
			result = dashboard_view.execute_nl_query()
			
			# Should return JSON response
			assert result.status_code == 200
			data = json.loads(result.get_data(as_text=True))
			assert data['success'] is True
			assert data['query_id'] == 'q123'
			assert data['confidence'] == 0.9
	
	def test_execute_nl_query_validation_error(self, app, dashboard_view):
		"""Test NL query execution with form validation error"""
		form_data = {
			'query': 'Hi',  # Too short
			'data_sources': ''
		}
		
		with app.test_request_context(method='POST', data=form_data):
			result = dashboard_view.execute_nl_query()
			
			assert result[1] == 400  # HTTP 400 Bad Request
			data = json.loads(result[0].get_data(as_text=True))
			assert data['success'] is False
			assert 'form_errors' in data
	
	def test_execute_sql_query_success(self, app, dashboard_view):
		"""Test successful SQL query execution"""
		# Mock query result
		mock_result = Mock()
		mock_result.id = 'q456'
		mock_result.status = 'completed'
		mock_result.results = [{'id': 1, 'name': 'John'}]
		mock_result.rows_returned = 1
		mock_result.duration_ms = 150
		
		dashboard_view.dvrl_service.execute_federated_query.return_value = mock_result
		
		form_data = {
			'sql': 'SELECT * FROM users LIMIT 10',
			'parameters': '{"limit": 10}'
		}
		
		with app.test_request_context(method='POST', data=form_data):
			result = dashboard_view.execute_sql_query()
			
			assert result.status_code == 200
			data = json.loads(result.get_data(as_text=True))
			assert data['success'] is True
			assert data['query_id'] == 'q456'
			assert data['rows_returned'] == 1
	
	def test_execute_sql_query_invalid_json_params(self, app, dashboard_view):
		"""Test SQL query execution with invalid JSON parameters"""
		form_data = {
			'sql': 'SELECT * FROM users',
			'parameters': 'invalid json'
		}
		
		with app.test_request_context(method='POST', data=form_data):
			result = dashboard_view.execute_sql_query()
			
			assert result[1] == 500  # Should handle JSON parse error
			data = json.loads(result[0].get_data(as_text=True))
			assert data['success'] is False
	
	def test_add_data_source_success(self, app, dashboard_view):
		"""Test successful data source addition"""
		# Mock successful data source creation
		mock_data_source = Mock()
		mock_data_source.name = 'New Test DB'
		dashboard_view.dvrl_service.register_data_source.return_value = mock_data_source
		
		form_data = {
			'name': 'New Test DB',
			'type': 'postgresql',
			'host': 'localhost',
			'port': '5432',
			'database': 'testdb',
			'username': 'testuser',
			'password': 'testpass',
			'description': 'Test database'
		}
		
		with app.test_request_context(method='POST', data=form_data):
			with patch('flask.flash') as mock_flash, \
				 patch('flask.redirect') as mock_redirect, \
				 patch('flask.url_for', return_value='/data-sources'):
				
				result = dashboard_view.add_data_source()
				
				mock_flash.assert_called()
				mock_redirect.assert_called()
				dashboard_view.dvrl_service.register_data_source.assert_called_once()
	
	def test_add_data_source_with_json_config(self, app, dashboard_view):
		"""Test data source addition with additional JSON config"""
		mock_data_source = Mock()
		mock_data_source.name = 'Advanced DB'
		dashboard_view.dvrl_service.register_data_source.return_value = mock_data_source
		
		form_data = {
			'name': 'Advanced DB',
			'type': 'postgresql',
			'connection_config': '{"ssl_mode": "require", "timeout": 30}'
		}
		
		with app.test_request_context(method='POST', data=form_data):
			with patch('flask.flash'), patch('flask.redirect'), patch('flask.url_for'):
				
				result = dashboard_view.add_data_source()
				
				# Verify the config was parsed and merged
				call_args = dashboard_view.dvrl_service.register_data_source.call_args[0][0]
				assert call_args['ssl_mode'] == 'require'
				assert call_args['timeout'] == 30
	
	def test_add_data_source_invalid_json(self, app, dashboard_view):
		"""Test data source addition with invalid JSON config"""
		form_data = {
			'name': 'Test DB',
			'type': 'postgresql',
			'connection_config': 'invalid json'
		}
		
		with app.test_request_context(method='POST', data=form_data):
			with patch('flask.flash') as mock_flash, \
				 patch('flask.redirect') as mock_redirect, \
				 patch('flask.url_for'):
				
				result = dashboard_view.add_data_source()
				
				# Should flash error message
				mock_flash.assert_called()
	
	def test_singer_taps_with_manager(self, app, dashboard_view):
		"""Test Singer taps page when manager is available"""
		# Mock Singer manager
		mock_manager = Mock()
		mock_manager.available_taps = {
			'tap-postgres': {
				'description': 'PostgreSQL tap',
				'category': 'database'
			}
		}
		mock_manager.installed_taps = {
			'tap-postgres': {
				'version': '1.0.0',
				'status': 'installed'
			}
		}
		
		dashboard_view.dvrl_service.singer_manager = mock_manager
		
		with app.test_request_context():
			with patch.object(dashboard_view, 'render_template', return_value='singer_taps_html'):
				result = dashboard_view.singer_taps()
				
				assert result == 'singer_taps_html'
	
	def test_singer_taps_without_manager(self, app, dashboard_view):
		"""Test Singer taps page when manager is not available"""
		# Remove singer_manager attribute
		if hasattr(dashboard_view.dvrl_service, 'singer_manager'):
			delattr(dashboard_view.dvrl_service, 'singer_manager')
		
		with app.test_request_context():
			with patch.object(dashboard_view, 'render_template', return_value='singer_error_html'):
				result = dashboard_view.singer_taps()
				
				assert result == 'singer_error_html'
	
	def test_performance_metrics(self, app, dashboard_view):
		"""Test performance metrics page"""
		# Mock connector manager
		dashboard_view.dvrl_service.connector_manager.get_connector_stats = AsyncMock(return_value={
			'total_connectors': 5,
			'healthy_connectors': 4,
			'total_queries': 1000,
			'avg_query_time': 125.5
		})
		
		with app.test_request_context():
			with patch.object(dashboard_view, 'render_template', return_value='metrics_html'):
				result = dashboard_view.performance_metrics()
				
				assert result == 'metrics_html'


# Integration tests for view components
class TestViewsIntegration:
	"""Integration tests for Flask-AppBuilder views"""
	
	@pytest.fixture
	def flask_app(self):
		"""Create Flask app with AppBuilder"""
		app = Flask(__name__)
		app.config['SECRET_KEY'] = 'test_secret'
		app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
		app.config['TESTING'] = True
		
		appbuilder = AppBuilder(app)
		return app, appbuilder
	
	def test_blueprint_registration(self, flask_app):
		"""Test DVRL blueprint registration"""
		app, appbuilder = flask_app
		
		app.register_blueprint(dvrl_bp)
		
		# Check blueprint is registered
		assert 'dvrl' in app.blueprints
		assert app.blueprints['dvrl'].url_prefix == '/dvrl'
	
	def test_view_integration_with_appbuilder(self, flask_app):
		"""Test view integration with Flask-AppBuilder"""
		app, appbuilder = flask_app
		
		# Mock service
		mock_service = Mock()
		mock_service.get_health_status = AsyncMock(return_value={'status': 'ok'})
		mock_service.get_performance_metrics = AsyncMock(return_value={'metrics': {}})
		mock_service.data_sources = {}
		
		# Create view and add to AppBuilder
		dashboard_view = DVRLDashboardView(mock_service)
		
		# This would normally be done by AppBuilder
		dashboard_view.appbuilder = appbuilder
		
		# Test that view has proper AppBuilder integration
		assert hasattr(dashboard_view, 'appbuilder')
		assert dashboard_view.route_base == '/dvrl'
		assert dashboard_view.default_view == 'dashboard'
	
	def test_form_integration_with_wtforms(self):
		"""Test forms integration with WTForms"""
		# Test that forms work with WTForms validation
		from wtforms import validators
		
		# Test DataSourceForm
		form = DataSourceForm()
		assert hasattr(form, 'name')
		assert hasattr(form, 'type')
		assert hasattr(form, 'validate')
		
		# Test form field validators
		name_field = form.name
		assert any(isinstance(v, validators.DataRequired) for v in name_field.validators)
		assert any(isinstance(v, validators.Length) for v in name_field.validators)


if __name__ == '__main__':
	pytest.main([__file__])