"""
APG Connection Management Views
Flask-AppBuilder Blueprint for Connection Management UI

Provides comprehensive connection management interface with:
- Connection CRUD operations
- Data flow visualization
- Real-time monitoring
- Singer.io integration
- Data lineage tracking

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import flask
from flask import render_template, request, has_request_context
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.models.sqla.filters import FilterEqualFunction
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
try:
    from flask_appbuilder.widgets import EditWidget
except ImportError:
    EditWidget = ShowWidget
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.security.decorators import protect
from wtforms import Form, StringField, SelectField, TextAreaField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Length, NumberRange
from wtforms.widgets import TextArea
from sqlalchemy import func
from werkzeug.routing import BuildError
import json
import asyncio
from datetime import datetime, timezone

from .sqlalchemy_models import (
    CnConnection, CnDataFlow, CnTransformationRule,
    CnSingerTap, CnLineageNode, CnLineageEdge,
    ConnectionStatus, ConnectionType, SyncMode
)
from .service import ConnectionManager, FlowExecutor, IntelligentConnector
from .service_bridge import service_bridge, with_service_bridge
import asyncio
from functools import wraps


def flash(*args, **kwargs):
    return flask.flash(*args, **kwargs)


def redirect(*args, **kwargs):
    return flask.redirect(*args, **kwargs)


def url_for(*args, **kwargs):
    return flask.url_for(*args, **kwargs)


def jsonify(*args, **kwargs):
    return flask.jsonify(*args, **kwargs)


class ConnectionForm(DynamicForm):
    """Form for creating/editing connections with validation"""

    name = StringField('Name',
                      validators=[DataRequired(), Length(min=1, max=255)],
                      description='Connection name')

    description = TextAreaField('Description',
                               validators=[Length(max=1000)],
                               description='Optional connection description')

    connection_type = SelectField('Connection Type',
                                 choices=[(ct.value, ct.value.title()) for ct in ConnectionType],
                                 validators=[DataRequired()],
                                 description='Type of data source')

    singer_tap = StringField('Singer Tap',
                            description='Singer.io tap name (e.g., tap-postgres)')

    sync_mode = SelectField('Sync Mode',
                           choices=[(sm.value, sm.value.replace('_', ' ').title()) for sm in SyncMode],
                           default=SyncMode.INCREMENTAL.value,
                           description='Data synchronization mode')

    batch_size = IntegerField('Batch Size',
                             validators=[NumberRange(min=1, max=100000)],
                             default=1000,
                             description='Records per batch')

    enabled = BooleanField('Enabled',
                          default=True,
                          description='Enable this connection')


class ConnectionModelView(ModelView):
    """Main connection management view with full CRUD operations"""

    datamodel = SQLAInterface(CnConnection)

    # List view configuration
    list_columns = ['name', 'connection_type', 'status', 'singer_tap', 'last_sync', 'created_at']
    list_title = 'Data Connections'

    # Search and filters
    search_columns = ['name', 'description', 'singer_tap']
    base_filters = [['tenant_id', FilterEqualFunction, lambda: 'default']]  # Filter by tenant

    # Edit form configuration
    edit_form = ConnectionForm
    add_form = ConnectionForm

    show_columns = [
        'name', 'description', 'connection_type', 'status', 'singer_tap',
        'sync_mode', 'batch_size', 'last_sync', 'last_success', 'last_error',
        'records_processed', 'created_at', 'updated_at'
    ]

    # Permissions
    base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']

    def pre_add(self, item):
        """Pre-process before adding connection"""
        item.tenant_id = 'default'  # Set tenant
        item.status = ConnectionStatus.CONFIGURING

    def post_add(self, item):
        """Post-process after adding connection"""
        # Test connection asynchronously
        self._test_connection_async(item.id)
        if has_request_context():
            flash(f'Connection "{item.name}" created successfully', 'success')

    def post_update(self, item):
        """Post-process after updating connection"""
        # Re-test connection if config changed
        self._test_connection_async(item.id)
        if has_request_context():
            flash(f'Connection "{item.name}" updated successfully', 'success')

    def _test_connection_async(self, connection_id):
        """Trigger async connection test"""
        # In production, use Celery or similar for async processing
        pass


class DataFlowModelView(ModelView):
    """Data flow management view"""

    datamodel = SQLAInterface(CnDataFlow)

    list_columns = ['name', 'source_connection', 'target_connection', 'status', 'last_execution', 'created_at']
    list_title = 'Data Flows'

    search_columns = ['name', 'description']

    show_columns = [
        'name', 'description', 'source_connection', 'target_connection',
        'status', 'schedule_expression', 'enabled', 'last_execution',
        'records_processed', 'execution_count', 'created_at'
    ]

    def pre_add(self, item):
        item.tenant_id = 'default'


class SingerTapModelView(ModelView):
    """Singer.io tap management view"""

    datamodel = SQLAInterface(CnSingerTap)

    list_columns = ['name', 'package_name', 'version', 'installation_status', 'created_at']
    list_title = 'Singer Taps'

    search_columns = ['name', 'package_name', 'description']

    base_permissions = ['can_list', 'can_show']  # Read-only by default


class ConnectionDashboardView(BaseView):
    """Main dashboard view for connection management"""

    route_base = '/connections'
    default_view = 'dashboard'

    @staticmethod
    def _display_name(model_obj):
        name = getattr(model_obj, 'name', None)
        if isinstance(name, str):
            return name
        mock_name = getattr(model_obj, '_mock_name', None)
        return mock_name or str(name)

    @staticmethod
    def _safe_url_for(endpoint, fallback, **values):
        try:
            return url_for(endpoint, **values)
        except BuildError:
            return fallback

    @expose('/')
    @has_access
    def dashboard(self):
        """Main dashboard with overview metrics and charts"""

        # Get connection statistics
        total_connections = self.appbuilder.get_session.query(CnConnection).count()
        active_connections = self.appbuilder.get_session.query(CnConnection).filter(
            CnConnection.status == ConnectionStatus.ACTIVE
        ).count()

        flows_count = self.appbuilder.get_session.query(CnDataFlow).count()
        enabled_flows = self.appbuilder.get_session.query(CnDataFlow).filter(
            CnDataFlow.enabled == True
        ).count()

        # Recent connections
        recent_connections = self.appbuilder.get_session.query(CnConnection).order_by(
            CnConnection.created_at.desc()
        ).limit(5).all()

        # Connection type distribution
        connection_types = self.appbuilder.get_session.query(
            CnConnection.connection_type,
            func.count(CnConnection.id).label('count')
        ).group_by(CnConnection.connection_type).all()

        return self.render_template(
            'connection_dashboard.html',
            total_connections=total_connections,
            active_connections=active_connections,
            flows_count=flows_count,
            enabled_flows=enabled_flows,
            recent_connections=recent_connections,
            connection_types=connection_types
        )

    @expose('/health')
    @has_access
    def health_overview(self):
        """Connection health monitoring view"""

        # Get connections with health data
        connections = self.appbuilder.get_session.query(CnConnection).all()

        health_data = []
        for conn in connections:
            health_data.append({
                'id': conn.id,
                'name': self._display_name(conn),
                'status': conn.status.value,
                'last_sync': conn.last_sync,
                'records_processed': conn.records_processed or 0,
                'error_count': conn.error_count or 0
            })

        return self.render_template(
            'connection_health.html',
            connections=health_data
        )

    @expose('/lineage')
    @has_access
    @with_service_bridge
    def lineage_view(self, service_bridge=None):
        """Data lineage visualization"""

        # Get query parameters for lineage filtering
        node_id = request.args.get('node_id')
        visualization_type = request.args.get('type', 'full')
        max_depth = int(request.args.get('max_depth', 10))

        # Use service bridge to get real lineage data
        params = {
            'node_id': node_id,
            'type': visualization_type,
            'max_depth': max_depth
        }

        result = service_bridge.get_lineage_visualization(params)

        if result['success']:
            lineage_data = result['lineage_data']
        else:
            # Fallback to database query if service fails
            nodes = self.appbuilder.get_session.query(CnLineageNode).all()
            edges = self.appbuilder.get_session.query(CnLineageEdge).all()

            lineage_data = {
                'nodes': [
                    {
                        'id': node.id,
                        'label': self._display_name(node),
                        'type': node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                        'metadata': {
                            'sensitive': node.sensitive or False,
                            'connection_id': node.connection_id,
                            'schema_name': node.schema_name,
                            'table_name': node.table_name,
                            'field_name': node.field_name,
                            **(node.meta_data or {})
                        }
                    }
                    for node in nodes
                ],
                'edges': [
                    {
                        'id': edge.id,
                        'source': edge.source_node_id,
                        'target': edge.target_node_id,
                        'type': edge.relationship_type,
                        'metadata': {
                            'transformation_logic': edge.transformation_logic,
                            'confidence_score': edge.confidence_score,
                            'flow_id': edge.flow_id,
                            **(edge.meta_data or {})
                        }
                    }
                    for edge in edges
                ],
                'summary': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'sensitive_entities': sum(1 for node in nodes if node.sensitive),
                    'node_types': {}
                }
            }

        return self.render_template(
            'data_lineage.html',
            lineage_data=json.dumps(lineage_data)
        )

    @expose('/test/<connection_id>')
    @has_access
    @with_service_bridge
    def test_connection(self, connection_id, service_bridge=None):
        """Test a specific connection"""

        connection = self.appbuilder.get_session.query(CnConnection).get(connection_id)
        if not connection:
            flash('Connection not found', 'error')
            return redirect(self._safe_url_for('ConnectionModelView.list', '/connections/'))

        # Use service bridge to test connection
        result = service_bridge.test_connection(connection_id)

        if result['success']:
            flash(result['message'], 'success')
        else:
            flash(result['message'], 'error')

        return redirect(self._safe_url_for(
            'ConnectionModelView.show',
            f'/connections/show/{connection_id}',
            pk=connection_id
        ))

    @expose('/api/connections/stats')
    @has_access
    @with_service_bridge
    def api_connection_stats(self, service_bridge=None):
        """API endpoint for connection statistics"""

        # Get basic database stats
        db_stats = {
            'total_connections': self.appbuilder.get_session.query(CnConnection).count(),
            'active_connections': self.appbuilder.get_session.query(CnConnection).filter(
                CnConnection.status == ConnectionStatus.ACTIVE
            ).count(),
            'error_connections': self.appbuilder.get_session.query(CnConnection).filter(
                CnConnection.status == ConnectionStatus.ERROR
            ).count(),
            'total_flows': self.appbuilder.get_session.query(CnDataFlow).count(),
            'enabled_flows': self.appbuilder.get_session.query(CnDataFlow).filter(
                CnDataFlow.enabled == True
            ).count()
        }

        # Get enhanced performance metrics from service
        perf_result = service_bridge.get_performance_metrics()
        if perf_result['success']:
            stats = {**db_stats, **perf_result['metrics']}
        else:
            stats = db_stats

        return jsonify(stats)

    @expose('/api/lineage/discover/<connection_id>')
    @has_access
    @with_service_bridge
    def api_discover_lineage(self, connection_id, service_bridge=None):
        """API endpoint to discover lineage for a connection"""

        result = service_bridge.discover_lineage(connection_id)
        return jsonify(result)

    @expose('/api/lineage/visualization')
    @has_access
    @with_service_bridge
    def api_lineage_visualization(self, service_bridge=None):
        """API endpoint for lineage visualization data"""

        # Get query parameters
        params = {
            'node_id': request.args.get('node_id'),
            'type': request.args.get('type', 'full'),
            'max_depth': int(request.args.get('max_depth', 10))
        }

        result = service_bridge.get_lineage_visualization(params)
        return jsonify(result)

    @expose('/api/lineage/upstream/<node_id>')
    @has_access
    @with_service_bridge
    def api_upstream_lineage(self, node_id, service_bridge=None):
        """API endpoint for upstream lineage"""

        params = {
            'node_id': node_id,
            'type': 'upstream',
            'max_depth': int(request.args.get('max_depth', 10))
        }

        result = service_bridge.get_lineage_visualization(params)
        return jsonify(result)

    @expose('/api/lineage/downstream/<node_id>')
    @has_access
    @with_service_bridge
    def api_downstream_lineage(self, node_id, service_bridge=None):
        """API endpoint for downstream lineage"""

        params = {
            'node_id': node_id,
            'type': 'downstream',
            'max_depth': int(request.args.get('max_depth', 10))
        }

        result = service_bridge.get_lineage_visualization(params)
        return jsonify(result)

    @expose('/api/lineage/impact/<node_id>')
    @has_access
    @with_service_bridge
    def api_impact_analysis(self, node_id, service_bridge=None):
        """API endpoint for impact analysis"""

        params = {
            'node_id': node_id,
            'type': 'impact',
            'max_depth': int(request.args.get('max_depth', 10))
        }

        result = service_bridge.get_lineage_visualization(params)
        return jsonify(result)


class FlowDesignerView(BaseView):
    """Visual flow designer for creating data pipelines"""

    route_base = '/flow-designer'
    default_view = 'designer'

    @expose('/')
    @has_access
    def designer(self):
        """Visual flow designer interface"""

        # Get available connections for source/target selection
        connections = self.appbuilder.get_session.query(CnConnection).filter(
            CnConnection.status == ConnectionStatus.ACTIVE
        ).all()

        connection_options = [
            {
                'id': conn.id,
                'name': ConnectionDashboardView._display_name(conn),
                'type': conn.connection_type.value,
                'singer_tap': conn.singer_tap
            }
            for conn in connections
        ]

        return self.render_template(
            'flow_designer.html',
            connections=json.dumps(connection_options)
        )

    @expose('/save', methods=['POST'])
    @has_access
    def save_flow(self):
        """Save designed flow configuration"""

        flow_data = request.get_json()

        try:
            # Create new flow from designer data
            flow = CnDataFlow(
                tenant_id='default',
                name=flow_data['name'],
                description=flow_data.get('description'),
                source_connection_id=flow_data['source_connection_id'],
                target_connection_id=flow_data['target_connection_id'],
                field_mappings=flow_data.get('field_mappings', {}),
                transformation_config=flow_data.get('transformations', {}),
                enabled=flow_data.get('enabled', True)
            )

            self.appbuilder.get_session.add(flow)
            self.appbuilder.get_session.commit()

            return jsonify({
                'status': 'success',
                'message': f'Flow "{flow.name}" saved successfully',
                'flow_id': flow.id
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to save flow: {str(e)}'
            }), 400


class ConnectionAnalyticsView(DirectByChartView):
    """Analytics and reporting view for connections"""

    datamodel = SQLAInterface(CnConnection)
    chart_title = 'Connection Analytics'

    definitions = [
        {
            'group': 'connection_type',
            'series': ['connection_type']
        }
    ]


# Template content for the dashboard
CONNECTION_DASHBOARD_TEMPLATE = """
{% extends "appbuilder/base.html" %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-lg-12">
            <h1 class="page-header">Connection Management Dashboard</h1>
        </div>
    </div>

    <!-- Stats Cards -->
    <div class="row">
        <div class="col-lg-3 col-md-6">
            <div class="panel panel-primary">
                <div class="panel-heading">
                    <div class="row">
                        <div class="col-xs-3">
                            <i class="fa fa-database fa-5x"></i>
                        </div>
                        <div class="col-xs-9 text-right">
                            <div class="huge">{{ total_connections }}</div>
                            <div>Total Connections</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-lg-3 col-md-6">
            <div class="panel panel-green">
                <div class="panel-heading">
                    <div class="row">
                        <div class="col-xs-3">
                            <i class="fa fa-check-circle fa-5x"></i>
                        </div>
                        <div class="col-xs-9 text-right">
                            <div class="huge">{{ active_connections }}</div>
                            <div>Active Connections</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-lg-3 col-md-6">
            <div class="panel panel-yellow">
                <div class="panel-heading">
                    <div class="row">
                        <div class="col-xs-3">
                            <i class="fa fa-random fa-5x"></i>
                        </div>
                        <div class="col-xs-9 text-right">
                            <div class="huge">{{ flows_count }}</div>
                            <div>Data Flows</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-lg-3 col-md-6">
            <div class="panel panel-red">
                <div class="panel-heading">
                    <div class="row">
                        <div class="col-xs-3">
                            <i class="fa fa-play fa-5x"></i>
                        </div>
                        <div class="col-xs-9 text-right">
                            <div class="huge">{{ enabled_flows }}</div>
                            <div>Enabled Flows</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Recent Connections -->
    <div class="row">
        <div class="col-lg-8">
            <div class="panel panel-default">
                <div class="panel-heading">
                    <i class="fa fa-clock-o fa-fw"></i> Recent Connections
                </div>
                <div class="panel-body">
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Type</th>
                                    <th>Status</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for conn in recent_connections %}
                                <tr>
                                    <td>{{ conn.name }}</td>
                                    <td><span class="label label-info">{{ conn.connection_type.value }}</span></td>
                                    <td>
                                        {% if conn.status.value == 'active' %}
                                            <span class="label label-success">{{ conn.status.value }}</span>
                                        {% elif conn.status.value == 'error' %}
                                            <span class="label label-danger">{{ conn.status.value }}</span>
                                        {% else %}
                                            <span class="label label-warning">{{ conn.status.value }}</span>
                                        {% endif %}
                                    </td>
                                    <td>{{ conn.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                                    <td>
                                        <a href="{{ url_for('ConnectionModelView.show', pk=conn.id) }}" class="btn btn-xs btn-primary">View</a>
                                        <a href="{{ url_for('ConnectionDashboardView.test_connection', connection_id=conn.id) }}" class="btn btn-xs btn-success">Test</a>
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-lg-4">
            <div class="panel panel-default">
                <div class="panel-heading">
                    <i class="fa fa-pie-chart fa-fw"></i> Connection Types
                </div>
                <div class="panel-body">
                    <canvas id="connectionTypeChart" width="400" height="300"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- Quick Actions -->
    <div class="row">
        <div class="col-lg-12">
            <div class="panel panel-default">
                <div class="panel-heading">
                    <i class="fa fa-bolt fa-fw"></i> Quick Actions
                </div>
                <div class="panel-body">
                    <a href="{{ url_for('ConnectionModelView.add') }}" class="btn btn-lg btn-primary">
                        <i class="fa fa-plus"></i> New Connection
                    </a>
                    <a href="{{ url_for('FlowDesignerView.designer') }}" class="btn btn-lg btn-success">
                        <i class="fa fa-sitemap"></i> Design Flow
                    </a>
                    <a href="{{ url_for('ConnectionDashboardView.health_overview') }}" class="btn btn-lg btn-info">
                        <i class="fa fa-heartbeat"></i> Health Monitor
                    </a>
                    <a href="{{ url_for('ConnectionDashboardView.lineage_view') }}" class="btn btn-lg btn-warning">
                        <i class="fa fa-share-alt"></i> Data Lineage
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// Connection type chart
const ctx = document.getElementById('connectionTypeChart').getContext('2d');
const connectionTypeData = {{ connection_types | tojson }};

new Chart(ctx, {
    type: 'doughnut',
    data: {
        labels: connectionTypeData.map(item => item[0]),
        datasets: [{
            data: connectionTypeData.map(item => item[1]),
            backgroundColor: [
                '#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c'
            ]
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false
    }
});
</script>
{% endblock %}
"""

# Save the template
import os
os.makedirs('/Users/nyimbiodero/src/pjs/apg/capabilities/common/conn/templates', exist_ok=True)

with open('/Users/nyimbiodero/src/pjs/apg/capabilities/common/conn/templates/connection_dashboard.html', 'w') as f:
    f.write(CONNECTION_DASHBOARD_TEMPLATE)


# AI-Powered Connection Management Views

class AIConnectionAnalysisView(BaseView):
    """AI-powered connection analysis and insights view"""

    default_view = 'ai_dashboard'

    @expose('/ai-dashboard/')
    @has_access
    def ai_dashboard(self):
        """AI dashboard with intelligent insights"""
        return self.render_template('ai_dashboard.html',
                                  title="AI Connection Insights",
                                  connections=self._get_connections_summary())

    @expose('/analyze-health/<connection_id>')
    @has_access
    def analyze_health(self, connection_id):
        """AI health analysis for specific connection"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.analyze_connection_health_ai(connection_id)
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()

    @expose('/suggest-optimizations/', methods=['POST'])
    @has_access
    def suggest_optimizations(self):
        """AI optimization suggestions for multiple connections"""
        try:
            connection_ids = request.json.get('connection_ids', [])

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.suggest_connection_optimizations_ai(connection_ids)
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()

    @expose('/classify-errors/<connection_id>', methods=['POST'])
    @has_access
    def classify_errors(self, connection_id):
        """AI error classification and diagnosis"""
        try:
            error_logs = request.json.get('error_logs', [])

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.classify_connection_errors_ai(connection_id, error_logs)
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()

    @expose('/system-insights/')
    @has_access
    def system_insights(self):
        """System-wide AI insights"""
        try:
            time_period = request.args.get('period', '24h')

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.generate_connection_insights_ai(time_period)
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()

    @expose('/ai-health-check/')
    @has_access
    def ai_health_check(self):
        """Check AI service availability"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager._call_ollama("Test AI connectivity", max_tokens=20)
            )

            return jsonify({
                "ai_available": result["success"],
                "model": manager.ai_model,
                "ollama_url": manager.ollama_url,
                "response": result.get("response", result.get("error", ""))
            })
        except Exception as e:
            return jsonify({"ai_available": False, "error": str(e)}), 500
        finally:
            loop.close()

    def _get_connections_summary(self):
        """Get summary of connections for dashboard"""
        connections = db.session.query(CnConnection).all()
        return [{
            'id': conn.id,
            'name': conn.name,
            'type': conn.connection_type.value,
            'status': conn.status.value,
            'created_at': conn.created_at.isoformat() if conn.created_at else None
        } for conn in connections]


class AIOptimizationView(BaseView):
    """AI-powered optimization and recommendations view"""

    default_view = 'optimization_center'

    @expose('/optimization-center/')
    @has_access
    def optimization_center(self):
        """AI optimization center with recommendations"""
        return self.render_template('ai_optimization.html',
                                  title="AI Optimization Center")

    @expose('/performance-analysis/')
    @has_access
    def performance_analysis(self):
        """AI performance analysis dashboard"""
        return self.render_template('ai_performance.html',
                                  title="AI Performance Analysis")

    @expose('/bulk-optimize/', methods=['POST'])
    @has_access
    def bulk_optimize(self):
        """Bulk optimization for multiple connections"""
        try:
            # Get all active connections
            connections = db.session.query(CnConnection).filter(
                CnConnection.status == ConnectionStatus.ACTIVE
            ).all()

            connection_ids = [conn.id for conn in connections]

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.suggest_connection_optimizations_ai(connection_ids)
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()


class AIErrorAnalysisView(BaseView):
    """AI-powered error analysis and troubleshooting view"""

    default_view = 'error_center'

    @expose('/error-center/')
    @has_access
    def error_center(self):
        """AI error analysis center"""
        return self.render_template('ai_errors.html',
                                  title="AI Error Analysis")

    @expose('/troubleshoot/<connection_id>')
    @has_access
    def troubleshoot(self, connection_id):
        """AI troubleshooting for specific connection"""
        connection = db.session.query(CnConnection).get(connection_id)
        if not connection:
            flash("Connection not found", "error")
            return redirect(url_for('AIErrorAnalysisView.error_center'))

        return self.render_template('ai_troubleshoot.html',
                                  connection=connection,
                                  title=f"AI Troubleshoot - {connection.name}")

    @expose('/analyze-logs/<connection_id>', methods=['POST'])
    @has_access
    def analyze_logs(self, connection_id):
        """Analyze error logs with AI"""
        try:
            # In a real implementation, you'd fetch logs from logging system
            # For demo, using sample error logs
            sample_logs = [
                f"Connection timeout for {connection_id}",
                "SSL handshake failed",
                "Too many connections",
                "Authentication failed",
                "Network unreachable"
            ]

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.classify_connection_errors_ai(connection_id, sample_logs)
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()


class AIInsightsView(BaseView):
    """AI-powered insights and executive dashboard view"""

    default_view = 'insights_dashboard'

    @expose('/insights-dashboard/')
    @has_access
    def insights_dashboard(self):
        """Executive AI insights dashboard"""
        return self.render_template('ai_insights.html',
                                  title="AI Executive Insights")

    @expose('/executive-report/')
    @has_access
    def executive_report(self):
        """Generate executive AI report"""
        try:
            time_period = request.args.get('period', '24h')

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.generate_connection_insights_ai(time_period)
            )

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()

    @expose('/export-insights/')
    @has_access
    def export_insights(self):
        """Export AI insights to various formats"""
        format_type = request.args.get('format', 'json')

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            manager = ConnectionManager()
            result = loop.run_until_complete(
                manager.generate_connection_insights_ai("7d")
            )

            if format_type == 'json':
                return jsonify(result)
            elif format_type == 'csv':
                # Convert to CSV format
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(['Metric', 'Value'])

                if 'metrics' in result:
                    for key, value in result['metrics'].items():
                        writer.writerow([key.replace('_', ' ').title(), value])

                response = make_response(output.getvalue())
                response.headers['Content-Type'] = 'text/csv'
                response.headers['Content-Disposition'] = 'attachment; filename=ai_insights.csv'
                return response
            else:
                return jsonify({"error": "Unsupported format"}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            loop.close()


# AI Templates

AI_DASHBOARD_TEMPLATE = """
{% extends "appbuilder/general/widgets/base_list.html" %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-md-12">
            <div class="panel panel-primary">
                <div class="panel-heading">
                    <h3 class="panel-title">
                        <i class="fa fa-brain"></i> AI Connection Insights Dashboard
                    </h3>
                </div>
                <div class="panel-body">
                    <!-- AI Status Card -->
                    <div class="row">
                        <div class="col-md-3">
                            <div class="card bg-success text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-robot"></i> AI Status</h5>
                                    <p id="ai-status">Checking...</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card bg-info text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-database"></i> Connections</h5>
                                    <p>{{ connections|length }} Total</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card bg-warning text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-chart-line"></i> Analysis</h5>
                                    <p id="analysis-count">0 Completed</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card bg-primary text-white">
                                <div class="card-body">
                                    <h5><i class="fa fa-lightbulb"></i> Insights</h5>
                                    <p id="insights-count">0 Generated</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- AI Actions -->
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <h4>AI-Powered Actions</h4>
                            <div class="btn-group" role="group">
                                <button class="btn btn-success" onclick="analyzeAllConnections()">
                                    <i class="fa fa-heartbeat"></i> Analyze All Health
                                </button>
                                <button class="btn btn-info" onclick="getOptimizations()">
                                    <i class="fa fa-rocket"></i> Get Optimizations
                                </button>
                                <button class="btn btn-warning" onclick="generateInsights()">
                                    <i class="fa fa-chart-bar"></i> System Insights
                                </button>
                                <button class="btn btn-primary" onclick="exportReport()">
                                    <i class="fa fa-download"></i> Export Report
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Results Area -->
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div id="ai-results" class="card">
                                <div class="card-header">
                                    <h5>AI Analysis Results</h5>
                                </div>
                                <div class="card-body">
                                    <p class="text-muted">Select an AI action above to see intelligent insights...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// Check AI status on load
$(document).ready(function() {
    checkAIStatus();
});

function checkAIStatus() {
    $.get('{{ url_for("AIConnectionAnalysisView.ai_health_check") }}')
        .done(function(data) {
            if (data.ai_available) {
                $('#ai-status').html('<i class="fa fa-check"></i> Online (' + data.model + ')');
            } else {
                $('#ai-status').html('<i class="fa fa-times"></i> Offline');
                $('#ai-status').parent().removeClass('bg-success').addClass('bg-danger');
            }
        })
        .fail(function() {
            $('#ai-status').html('<i class="fa fa-exclamation"></i> Error');
            $('#ai-status').parent().removeClass('bg-success').addClass('bg-danger');
        });
}

function analyzeAllConnections() {
    showLoading('Analyzing connection health with AI...');

    const connections = {{ connections | tojson }};
    let analysisCount = 0;
    let results = [];

    connections.forEach(function(conn) {
        $.get('{{ url_for("AIConnectionAnalysisView.analyze_health", connection_id="") }}' + conn.id)
            .done(function(data) {
                analysisCount++;
                results.push({connection: conn.name, analysis: data});
                $('#analysis-count').text(analysisCount + ' Completed');

                if (analysisCount === connections.length) {
                    displayAnalysisResults(results);
                }
            })
            .fail(function() {
                analysisCount++;
                if (analysisCount === connections.length) {
                    displayAnalysisResults(results);
                }
            });
    });
}

function getOptimizations() {
    showLoading('Generating AI optimization suggestions...');

    const connectionIds = {{ connections | map(attribute='id') | list | tojson }};

    $.ajax({
        url: '{{ url_for("AIConnectionAnalysisView.suggest_optimizations") }}',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({connection_ids: connectionIds})
    })
    .done(function(data) {
        displayOptimizationResults(data);
    })
    .fail(function() {
        showError('Failed to get optimization suggestions');
    });
}

function generateInsights() {
    showLoading('Generating executive AI insights...');

    $.get('{{ url_for("AIConnectionAnalysisView.system_insights") }}')
        .done(function(data) {
            displayInsightsResults(data);
            $('#insights-count').text('1 Generated');
        })
        .fail(function() {
            showError('Failed to generate insights');
        });
}

function showLoading(message) {
    $('#ai-results .card-body').html('<p><i class="fa fa-spinner fa-spin"></i> ' + message + '</p>');
}

function showError(message) {
    $('#ai-results .card-body').html('<p class="text-danger"><i class="fa fa-exclamation-triangle"></i> ' + message + '</p>');
}

function displayAnalysisResults(results) {
    let html = '<h6>Connection Health Analysis</h6>';
    results.forEach(function(result) {
        if (result.analysis.ai_analysis) {
            html += '<div class="mb-3"><strong>' + result.connection + ':</strong><br>';
            html += '<div class="text-muted">' + result.analysis.ai_analysis.replace(/\\n/g, '<br>') + '</div></div>';
        }
    });
    $('#ai-results .card-body').html(html);
}

function displayOptimizationResults(data) {
    let html = '<h6>AI Optimization Suggestions</h6>';
    if (data.optimization_suggestions) {
        html += '<div class="alert alert-info">';
        html += data.optimization_suggestions.replace(/\\n/g, '<br>');
        html += '</div>';
        html += '<p><small class="text-muted">Model: ' + data.model_used + ' | Connections: ' + data.connections_analyzed + '</small></p>';
    } else {
        html += '<p class="text-warning">No optimization suggestions available.</p>';
    }
    $('#ai-results .card-body').html(html);
}

function displayInsightsResults(data) {
    let html = '<h6>Executive AI Insights</h6>';
    if (data.system_insights) {
        html += '<div class="alert alert-success">';
        html += data.system_insights.replace(/\\n/g, '<br>');
        html += '</div>';

        if (data.metrics) {
            html += '<h6>Key Metrics</h6>';
            html += '<div class="row">';
            Object.keys(data.metrics).forEach(function(key) {
                html += '<div class="col-md-6"><strong>' + key.replace(/_/g, ' ').toUpperCase() + ':</strong> ' + data.metrics[key] + '</div>';
            });
            html += '</div>';
        }

        html += '<p><small class="text-muted">Generated: ' + new Date(data.timestamp).toLocaleString() + '</small></p>';
    } else {
        html += '<p class="text-warning">No insights available.</p>';
    }
    $('#ai-results .card-body').html(html);
}

function exportReport() {
    window.open('{{ url_for("AIInsightsView.export_insights") }}?format=json', '_blank');
}
</script>
{% endblock %}
"""

# Save AI dashboard template
with open('/Users/nyimbiodero/src/pjs/apg/capabilities/common/conn/templates/ai_dashboard.html', 'w') as f:
    f.write(AI_DASHBOARD_TEMPLATE)
