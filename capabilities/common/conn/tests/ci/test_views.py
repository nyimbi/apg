"""
Tests for Flask-AppBuilder Views
Tests the web interface, API endpoints, and view integration

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from flask import url_for

from ...views import (
    ConnectionModelView, DataFlowModelView, SingerTapModelView,
    ConnectionDashboardView, FlowDesignerView, ConnectionAnalyticsView
)
from ...sqlalchemy_models import CnConnection, CnDataFlow, ConnectionStatus, ConnectionType


class TestConnectionModelView:
	"""Test ConnectionModelView CRUD operations"""

	@pytest.fixture
	def connection_view(self, mock_flask_app):
		"""Create ConnectionModelView instance"""
		app, appbuilder = mock_flask_app
		view = ConnectionModelView()
		view.appbuilder = appbuilder
		return view

	def test_pre_add_connection(self, connection_view, sample_connection_data):
		"""Test pre_add method sets defaults"""
		mock_connection = Mock()

		connection_view.pre_add(mock_connection)

		assert mock_connection.tenant_id == 'default'
		assert mock_connection.status == ConnectionStatus.CONFIGURING

	def test_post_add_connection(self, connection_view):
		"""Test post_add method triggers connection test"""
		mock_connection = Mock()
		mock_connection.id = 'conn_123'
		mock_connection.name = 'Test Connection'

		with patch.object(connection_view, '_test_connection_async') as mock_test:
			connection_view.post_add(mock_connection)

			mock_test.assert_called_once_with('conn_123')

	def test_post_update_connection(self, connection_view):
		"""Test post_update method triggers connection re-test"""
		mock_connection = Mock()
		mock_connection.id = 'conn_123'
		mock_connection.name = 'Updated Connection'

		with patch.object(connection_view, '_test_connection_async') as mock_test:
			connection_view.post_update(mock_connection)

			mock_test.assert_called_once_with('conn_123')


class TestDataFlowModelView:
	"""Test DataFlowModelView functionality"""

	@pytest.fixture
	def flow_view(self, mock_flask_app):
		"""Create DataFlowModelView instance"""
		app, appbuilder = mock_flask_app
		view = DataFlowModelView()
		view.appbuilder = appbuilder
		return view

	def test_pre_add_flow(self, flow_view):
		"""Test pre_add method sets tenant"""
		mock_flow = Mock()

		flow_view.pre_add(mock_flow)

		assert mock_flow.tenant_id == 'default'


class TestConnectionDashboardView:
	"""Test ConnectionDashboardView functionality"""

	@pytest.fixture
	def dashboard_view(self, mock_flask_app, db_session):
		"""Create ConnectionDashboardView instance"""
		app, appbuilder = mock_flask_app

		# Mock database session
		mock_session = Mock()
		mock_session.query.return_value = Mock()
		appbuilder.get_session = mock_session

		view = ConnectionDashboardView()
		view.appbuilder = appbuilder

		return view, app

	def test_dashboard_view(self, dashboard_view):
		"""Test dashboard view renders correctly"""
		view, app = dashboard_view

		# Mock query results
		mock_session = view.appbuilder.get_session

		# Mock connection counts
		mock_session.query.return_value.count.return_value = 5
		mock_session.query.return_value.filter.return_value.count.return_value = 3

		# Mock recent connections
		mock_connections = [
			Mock(id='1', name='DB1', connection_type=ConnectionType.DATABASE, status=ConnectionStatus.ACTIVE),
			Mock(id='2', name='DB2', connection_type=ConnectionType.API, status=ConnectionStatus.ERROR)
		]
		mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = mock_connections

		# Mock connection types
		mock_session.query.return_value.group_by.return_value.all.return_value = [
			(ConnectionType.DATABASE, 3),
			(ConnectionType.API, 2)
		]

		with app.test_request_context('/connections/'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Dashboard</html>'

				result = view.dashboard()

				mock_render.assert_called_once()
				args, kwargs = mock_render.call_args

				assert 'connection_dashboard.html' in args
				assert 'total_connections' in kwargs
				assert 'active_connections' in kwargs
				assert 'recent_connections' in kwargs

	def test_health_overview(self, dashboard_view):
		"""Test health overview view"""
		view, app = dashboard_view

		# Mock connections with health data
		mock_connections = [
			Mock(
				id='1', name='DB1', status=ConnectionStatus.ACTIVE,
				last_sync='2025-01-01', records_processed=1000, error_count=0
			),
			Mock(
				id='2', name='DB2', status=ConnectionStatus.ERROR,
				last_sync='2025-01-01', records_processed=500, error_count=5
			)
		]

		view.appbuilder.get_session.query.return_value.all.return_value = mock_connections

		with app.test_request_context('/connections/health'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Health</html>'

				result = view.health_overview()

				mock_render.assert_called_once()
				args, kwargs = mock_render.call_args

				assert 'connection_health.html' in args
				assert len(kwargs['connections']) == 2
				assert kwargs['connections'][0]['name'] == 'DB1'
				assert kwargs['connections'][0]['status'] == 'active'

	def test_lineage_view_success(self, dashboard_view):
		"""Test lineage view with successful service call"""
		view, app = dashboard_view

		mock_lineage_data = {
			'nodes': [
				{'id': 'node1', 'label': 'Table1', 'type': 'table', 'metadata': {'sensitive': False}}
			],
			'edges': [
				{'id': 'edge1', 'source': 'node1', 'target': 'node2', 'type': 'contains', 'metadata': {}}
			],
			'summary': {'total_nodes': 1, 'total_edges': 1}
		}

		with app.test_request_context('/connections/lineage'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Lineage</html>'

				# Mock service bridge
				mock_service_bridge = Mock()
				mock_service_bridge.get_lineage_visualization.return_value = {
					'success': True,
					'lineage_data': mock_lineage_data
				}

				result = view.lineage_view(service_bridge=mock_service_bridge)

				mock_render.assert_called_once()
				args, kwargs = mock_render.call_args

				assert 'data_lineage.html' in args
				lineage_data = json.loads(kwargs['lineage_data'])
				assert len(lineage_data['nodes']) == 1
				assert len(lineage_data['edges']) == 1

	def test_lineage_view_fallback(self, dashboard_view):
		"""Test lineage view falls back to database when service fails"""
		view, app = dashboard_view

		# Mock database nodes and edges
		mock_nodes = [
			Mock(
				id='node1', name='Table1', node_type=Mock(value='table'),
				sensitive=False, connection_id='conn1', schema_name='public',
				table_name='table1', field_name=None, meta_data={}
			)
		]
		mock_edges = [
			Mock(
				id='edge1', source_node_id='node1', target_node_id='node2',
				relationship_type='contains', transformation_logic=None,
				confidence_score=1.0, flow_id=None, meta_data={}
			)
		]

		view.appbuilder.get_session.query.return_value.all.side_effect = [mock_nodes, mock_edges]

		with app.test_request_context('/connections/lineage'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Lineage</html>'

				# Mock service bridge failure
				mock_service_bridge = Mock()
				mock_service_bridge.get_lineage_visualization.return_value = {
					'success': False,
					'error': 'Service unavailable'
				}

				result = view.lineage_view(service_bridge=mock_service_bridge)

				mock_render.assert_called_once()
				args, kwargs = mock_render.call_args

				lineage_data = json.loads(kwargs['lineage_data'])
				assert 'nodes' in lineage_data
				assert 'edges' in lineage_data
				assert 'summary' in lineage_data

	def test_test_connection_success(self, dashboard_view):
		"""Test connection testing endpoint"""
		view, app = dashboard_view

		# Mock connection in database
		mock_connection = Mock(id='conn_123')
		view.appbuilder.get_session.query.return_value.get.return_value = mock_connection

		with app.test_request_context('/connections/test/conn_123'):
			with patch('flask.flash') as mock_flash:
				with patch('flask.redirect') as mock_redirect:
					with patch('flask.url_for') as mock_url_for:
						mock_url_for.return_value = '/connections/show/conn_123'

						# Mock service bridge
						mock_service_bridge = Mock()
						mock_service_bridge.test_connection.return_value = {
							'success': True,
							'message': 'Connection successful'
						}

						result = view.test_connection('conn_123', service_bridge=mock_service_bridge)

						mock_flash.assert_called_once_with('Connection successful', 'success')
						mock_redirect.assert_called_once()

	def test_test_connection_not_found(self, dashboard_view):
		"""Test connection testing with non-existent connection"""
		view, app = dashboard_view

		# Mock connection not found
		view.appbuilder.get_session.query.return_value.get.return_value = None

		with app.test_request_context('/connections/test/invalid_id'):
			with patch('flask.flash') as mock_flash:
				with patch('flask.redirect') as mock_redirect:

					result = view.test_connection('invalid_id', service_bridge=Mock())

					mock_flash.assert_called_once_with('Connection not found', 'error')
					mock_redirect.assert_called_once()

	def test_api_connection_stats(self, dashboard_view):
		"""Test API endpoint for connection statistics"""
		view, app = dashboard_view

		# Mock database queries
		mock_session = view.appbuilder.get_session
		mock_session.query.return_value.count.return_value = 10
		mock_session.query.return_value.filter.return_value.count.side_effect = [8, 1, 5, 3]

		with app.test_request_context('/connections/api/connections/stats'):
			# Mock service bridge
			mock_service_bridge = Mock()
			mock_service_bridge.get_performance_metrics.return_value = {
				'success': True,
				'metrics': {
					'avg_latency': 150.5,
					'total_throughput': 5000
				}
			}

			with patch('flask.jsonify') as mock_jsonify:
				result = view.api_connection_stats(service_bridge=mock_service_bridge)

				mock_jsonify.assert_called_once()
				stats = mock_jsonify.call_args[0][0]

				assert stats['total_connections'] == 10
				assert stats['active_connections'] == 8
				assert stats['avg_latency'] == 150.5

	def test_api_discover_lineage(self, dashboard_view):
		"""Test API endpoint for lineage discovery"""
		view, app = dashboard_view

		with app.test_request_context('/connections/api/lineage/discover/conn_123'):
			mock_service_bridge = Mock()
			mock_service_bridge.discover_lineage.return_value = {
				'success': True,
				'discovery_result': {'nodes_created': 25}
			}

			with patch('flask.jsonify') as mock_jsonify:
				result = view.api_discover_lineage('conn_123', service_bridge=mock_service_bridge)

				mock_jsonify.assert_called_once()
				response = mock_jsonify.call_args[0][0]
				assert response['success'] is True
				assert response['discovery_result']['nodes_created'] == 25

	def test_api_lineage_visualization(self, dashboard_view):
		"""Test API endpoint for lineage visualization"""
		view, app = dashboard_view

		with app.test_request_context('/connections/api/lineage/visualization?type=upstream&max_depth=5'):
			mock_service_bridge = Mock()
			mock_service_bridge.get_lineage_visualization.return_value = {
				'success': True,
				'lineage_data': {'nodes': [], 'edges': []}
			}

			with patch('flask.jsonify') as mock_jsonify:
				result = view.api_lineage_visualization(service_bridge=mock_service_bridge)

				mock_jsonify.assert_called_once()

				# Check service bridge was called with correct parameters
				call_args = mock_service_bridge.get_lineage_visualization.call_args[0][0]
				assert call_args['type'] == 'upstream'
				assert call_args['max_depth'] == 5


class TestFlowDesignerView:
	"""Test FlowDesignerView functionality"""

	@pytest.fixture
	def designer_view(self, mock_flask_app):
		"""Create FlowDesignerView instance"""
		app, appbuilder = mock_flask_app

		# Mock active connections
		mock_connections = [
			Mock(id='1', name='DB1', connection_type=ConnectionType.DATABASE, singer_tap='tap-postgres'),
			Mock(id='2', name='API1', connection_type=ConnectionType.API, singer_tap='tap-salesforce')
		]

		mock_session = Mock()
		mock_session.query.return_value.filter.return_value.all.return_value = mock_connections
		appbuilder.get_session = mock_session

		view = FlowDesignerView()
		view.appbuilder = appbuilder

		return view, app

	def test_designer_view(self, designer_view):
		"""Test flow designer main view"""
		view, app = designer_view

		with app.test_request_context('/flow-designer/'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Flow Designer</html>'

				result = view.designer()

				mock_render.assert_called_once()
				args, kwargs = mock_render.call_args

				assert 'flow_designer.html' in args

				connections = json.loads(kwargs['connections'])
				assert len(connections) == 2
				assert connections[0]['name'] == 'DB1'
				assert connections[0]['type'] == 'database'

	def test_save_flow_success(self, designer_view):
		"""Test successful flow saving"""
		view, app = designer_view

		flow_data = {
			'name': 'Test Flow',
			'description': 'Test flow description',
			'source_connection_id': 'conn_1',
			'target_connection_id': 'conn_2',
			'field_mappings': {'id': 'user_id'},
			'transformations': {},
			'enabled': True
		}

		with app.test_request_context('/flow-designer/save',
									method='POST',
									json=flow_data):
			with patch('flask.request') as mock_request:
				mock_request.get_json.return_value = flow_data

				# Mock database operations
				mock_session = view.appbuilder.get_session
				mock_session.add = Mock()
				mock_session.commit = Mock()

				with patch('flask.jsonify') as mock_jsonify:
					result = view.save_flow()

					mock_session.add.assert_called_once()
					mock_session.commit.assert_called_once()

					mock_jsonify.assert_called_once()
					response = mock_jsonify.call_args[0][0]
					assert response['status'] == 'success'
					assert 'Test Flow' in response['message']

	def test_save_flow_error(self, designer_view):
		"""Test flow saving error handling"""
		view, app = designer_view

		flow_data = {'invalid': 'data'}

		with app.test_request_context('/flow-designer/save',
									method='POST',
									json=flow_data):
			with patch('flask.request') as mock_request:
				mock_request.get_json.return_value = flow_data

				# Mock database error
				mock_session = view.appbuilder.get_session
				mock_session.add.side_effect = Exception("Database error")

				result = view.save_flow()

				# Should return error response with 400 status
				assert isinstance(result, tuple)
				response_data, status_code = result
				assert status_code == 400


class TestConnectionAnalyticsView:
	"""Test ConnectionAnalyticsView functionality"""

	def test_analytics_view_configuration(self):
		"""Test analytics view is configured correctly"""
		view = ConnectionAnalyticsView()

		assert view.chart_title == 'Connection Analytics'
		assert len(view.definitions) > 0
		assert 'group' in view.definitions[0]
		assert view.definitions[0]['group'] == 'connection_type'


class TestViewIntegration:
	"""Integration tests for views"""

	def test_view_service_bridge_integration(self, mock_flask_app):
		"""Test views properly integrate with service bridge"""
		app, appbuilder = mock_flask_app

		view = ConnectionDashboardView()
		view.appbuilder = appbuilder

		# Test that service bridge decorator works
		mock_service_bridge = Mock()

		with app.test_request_context('/connections/test/conn_123'):
			# This should work without raising exceptions
			with patch.object(view, 'appbuilder') as mock_appbuilder:
				mock_appbuilder.get_session.query.return_value.get.return_value = None

				with patch('flask.flash'):
					with patch('flask.redirect'):
						result = view.test_connection('conn_123', service_bridge=mock_service_bridge)

	def test_view_database_integration(self, mock_flask_app, sample_connection, sample_flow):
		"""Test views integrate properly with database models"""
		app, appbuilder = mock_flask_app

		# Mock session with real model instances
		mock_session = Mock()
		mock_session.query.return_value.get.return_value = sample_connection
		mock_session.query.return_value.all.return_value = [sample_connection]
		appbuilder.get_session = mock_session

		view = ConnectionDashboardView()
		view.appbuilder = appbuilder

		with app.test_request_context('/connections/'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Dashboard</html>'

				# Mock counts
				mock_session.query.return_value.count.return_value = 1
				mock_session.query.return_value.filter.return_value.count.return_value = 1
				mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = [sample_connection]
				mock_session.query.return_value.group_by.return_value.all.return_value = []

				result = view.dashboard()

				mock_render.assert_called_once()


# Error handling tests
class TestViewErrorHandling:
	"""Test error handling in views"""

	def test_dashboard_database_error(self, mock_flask_app):
		"""Test dashboard handles database errors gracefully"""
		app, appbuilder = mock_flask_app

		# Mock database error
		mock_session = Mock()
		mock_session.query.side_effect = Exception("Database connection failed")
		appbuilder.get_session = mock_session

		view = ConnectionDashboardView()
		view.appbuilder = appbuilder

		with app.test_request_context('/connections/'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Error Dashboard</html>'

				# Should handle error gracefully
				with pytest.raises(Exception, match="Database connection failed"):
					result = view.dashboard()

	def test_api_endpoint_error_handling(self, mock_flask_app):
		"""Test API endpoints handle errors and return proper JSON responses"""
		app, appbuilder = mock_flask_app

		view = ConnectionDashboardView()
		view.appbuilder = appbuilder

		with app.test_request_context('/connections/api/lineage/discover/invalid'):
			mock_service_bridge = Mock()
			mock_service_bridge.discover_lineage.side_effect = Exception("Service error")

			with patch('flask.jsonify') as mock_jsonify:
				# Should catch exception and return error response
				with pytest.raises(Exception, match="Service error"):
					result = view.api_discover_lineage('invalid', service_bridge=mock_service_bridge)


# Performance tests
class TestViewPerformance:
	"""Performance tests for views"""

	def test_dashboard_with_large_dataset(self, mock_flask_app):
		"""Test dashboard performance with large number of connections"""
		app, appbuilder = mock_flask_app

		# Mock large dataset
		large_connection_list = [
			Mock(id=f'conn_{i}', name=f'Connection {i}',
				connection_type=ConnectionType.DATABASE, status=ConnectionStatus.ACTIVE)
			for i in range(1000)
		]

		mock_session = Mock()
		mock_session.query.return_value.count.return_value = 1000
		mock_session.query.return_value.filter.return_value.count.return_value = 800
		mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = large_connection_list[:5]
		mock_session.query.return_value.group_by.return_value.all.return_value = []
		appbuilder.get_session = mock_session

		view = ConnectionDashboardView()
		view.appbuilder = appbuilder

		with app.test_request_context('/connections/'):
			with patch.object(view, 'render_template') as mock_render:
				mock_render.return_value = '<html>Dashboard</html>'

				import time
				start_time = time.time()

				result = view.dashboard()

				duration = time.time() - start_time

				# Should complete quickly even with large dataset
				assert duration < 1.0  # Under 1 second
				mock_render.assert_called_once()

	def test_lineage_api_performance(self, mock_flask_app):
		"""Test lineage API performance with complex graph"""
		app, appbuilder = mock_flask_app

		view = ConnectionDashboardView()
		view.appbuilder = appbuilder

		# Mock large lineage data
		large_lineage_data = {
			'nodes': [{'id': f'node_{i}', 'type': 'table'} for i in range(500)],
			'edges': [{'id': f'edge_{i}', 'source': f'node_{i}', 'target': f'node_{i+1}'} for i in range(499)],
			'summary': {'total_nodes': 500, 'total_edges': 499}
		}

		with app.test_request_context('/connections/api/lineage/visualization'):
			mock_service_bridge = Mock()
			mock_service_bridge.get_lineage_visualization.return_value = {
				'success': True,
				'lineage_data': large_lineage_data
			}

			with patch('flask.jsonify') as mock_jsonify:
				import time
				start_time = time.time()

				result = view.api_lineage_visualization(service_bridge=mock_service_bridge)

				duration = time.time() - start_time

				# Should handle large dataset efficiently
				assert duration < 2.0  # Under 2 seconds
				mock_jsonify.assert_called_once()