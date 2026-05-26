"""
APG Connection Management Capability
Flask-AppBuilder Blueprint Registration

This capability provides comprehensive data connection management with:
- Visual connection designer
- Singer.io integration
- Real-time monitoring
- Data lineage tracking
- Advanced transformation engine

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

# Import views and models
from .views import (
    ConnectionModelView,
    DataFlowModelView,
    SingerTapModelView,
    ConnectionDashboardView,
    FlowDesignerView,
    ConnectionAnalyticsView,
    AIConnectionAnalysisView,
    AIOptimizationView,
    AIErrorAnalysisView,
    AIInsightsView
)

from .composition_views import (
    CapabilityCompositionView,
    CapabilityTestView
)

from .monitoring_views import (
    MonitoringDashboardView,
    LogsView
)

from .sqlalchemy_models import (
    CnConnection,
    CnDataFlow,
    CnSingerTap,
    CnSingerTarget,
    CnTransformationRule,
    CnLineageNode,
    CnLineageEdge,
    CnHealthCheck,
    CnFlowExecution
)

# Capability information
CAPABILITY_INFO = {
    'name': 'Connection Management',
    'code': 'conn',
    'version': '1.0.0',
    'description': 'Comprehensive data connection management with Singer.io integration',
    'author': 'Nyimbi Odero',
    'company': 'Datacraft',
    'category': 'Data Integration',
    'keywords': ['connections', 'data-integration', 'singer.io', 'etl', 'data-lineage'],
    'capabilities': [
        'connection_management',
        'data_flows',
        'singer_integration',
        'lineage_tracking',
        'health_monitoring',
        'visual_designer',
        'capability_composition',
        'security_controls',
        'production_monitoring'
    ],
    'composition_keywords': [
        'connect_data_source',
        'create_data_flow',
        'track_lineage',
        'monitor_health',
        'transform_data',
        'compose_capabilities',
        'register_capability',
        'create_composition'
    ]
}


def create_connection_blueprint(name='connection_mgmt'):
    """Create Flask blueprint for connection management capability"""

    blueprint = Blueprint(
        name,
        __name__,
        url_prefix='/conn',
        template_folder='templates',
        static_folder='static'
    )

    return blueprint


def register_views(appbuilder: AppBuilder):
    """Register all views with Flask-AppBuilder"""

    # Main dashboard view (default route)
    appbuilder.add_view(
        ConnectionDashboardView,
        "Connection Dashboard",
        icon="fa-dashboard",
        category="Connections",
        category_icon="fa-database"
    )

    # Connection management
    appbuilder.add_view(
        ConnectionModelView,
        "Manage Connections",
        icon="fa-plug",
        category="Connections"
    )

    # Data flow management
    appbuilder.add_view(
        DataFlowModelView,
        "Data Flows",
        icon="fa-random",
        category="Connections"
    )

    # Flow designer
    appbuilder.add_view(
        FlowDesignerView,
        "Flow Designer",
        icon="fa-sitemap",
        category="Connections"
    )

    # Singer.io management
    appbuilder.add_view(
        SingerTapModelView,
        "Singer Taps",
        icon="fa-music",
        category="Connections"
    )

    # Analytics
    appbuilder.add_view(
        ConnectionAnalyticsView,
        "Analytics",
        icon="fa-bar-chart",
        category="Connections"
    )

    # Capability Composition
    appbuilder.add_view(
        CapabilityCompositionView,
        "Capability Composition",
        icon="fa-puzzle-piece",
        category="Composition",
        category_icon="fa-link"
    )

    # Capability Testing
    appbuilder.add_view(
        CapabilityTestView,
        "Composition Testing",
        icon="fa-flask",
        category="Composition"
    )

    # Monitoring Dashboard
    appbuilder.add_view(
        MonitoringDashboardView,
        "Monitoring Dashboard",
        icon="fa-chart-line",
        category="Monitoring",
        category_icon="fa-eye"
    )

    # Logs View
    appbuilder.add_view(
        LogsView,
        "System Logs",
        icon="fa-file-alt",
        category="Monitoring"
    )

    # AI-Powered Views
    appbuilder.add_view(
        AIConnectionAnalysisView,
        "AI Insights Dashboard",
        icon="fa-brain",
        category="AI Intelligence",
        category_icon="fa-robot"
    )

    appbuilder.add_view(
        AIOptimizationView,
        "AI Optimization Center",
        icon="fa-rocket",
        category="AI Intelligence"
    )

    appbuilder.add_view(
        AIErrorAnalysisView,
        "AI Error Analysis",
        icon="fa-bug",
        category="AI Intelligence"
    )

    appbuilder.add_view(
        AIInsightsView,
        "Executive AI Insights",
        icon="fa-chart-pie",
        category="AI Intelligence"
    )


def register_menu_links(appbuilder: AppBuilder):
    """Register additional menu links"""

    # Quick action links
    appbuilder.add_link(
        "New Connection",
        href="/connectionmodelview/add",
        icon="fa-plus",
        category="Quick Actions"
    )

    appbuilder.add_link(
        "Connection Health",
        href="/connections/health",
        icon="fa-heartbeat",
        category="Quick Actions"
    )

    appbuilder.add_link(
        "Data Lineage",
        href="/connections/lineage",
        icon="fa-share-alt",
        category="Quick Actions"
    )

    appbuilder.add_link(
        "Register Capability",
        href="/composition/register",
        icon="fa-plus-circle",
        category="Quick Actions"
    )

    appbuilder.add_link(
        "Create Composition",
        href="/composition/compose",
        icon="fa-link",
        category="Quick Actions"
    )

    appbuilder.add_link(
        "System Health",
        href="/monitoring/health",
        icon="fa-heartbeat",
        category="Quick Actions"
    )

    appbuilder.add_link(
        "View Metrics",
        href="/monitoring/api/metrics",
        icon="fa-chart-bar",
        category="Quick Actions"
    )


def init_capability(appbuilder: AppBuilder):
    """Initialize the connection management capability"""

    # Register views
    register_views(appbuilder)

    # Register menu links
    register_menu_links(appbuilder)

    # Initialize database tables if needed
    try:
        # This would typically be handled by Alembic migrations
        from .sqlalchemy_models import Base
        Base.metadata.create_all(appbuilder.get_session.get_bind())
    except Exception as e:
        print(f"Warning: Could not initialize database tables: {e}")

    print(f"✓ Connection Management capability initialized")
    print(f"  Version: {CAPABILITY_INFO['version']}")
    print(f"  Features: {', '.join(CAPABILITY_INFO['capabilities'])}")


# Export key components
__all__ = [
    'CAPABILITY_INFO',
    'create_connection_blueprint',
    'register_views',
    'register_menu_links',
    'init_capability',
    # Models
    'CnConnection',
    'CnDataFlow',
    'CnSingerTap',
    'CnSingerTarget',
    'CnTransformationRule',
    'CnLineageNode',
    'CnLineageEdge',
    'CnHealthCheck',
    'CnFlowExecution',
    # Views
    'ConnectionModelView',
    'DataFlowModelView',
    'SingerTapModelView',
    'ConnectionDashboardView',
    'FlowDesignerView',
    'ConnectionAnalyticsView',
    'CapabilityCompositionView',
    'CapabilityTestView'
]