#!/usr/bin/env python3
"""
Create Business Intelligence and Reporting Capabilities
=======================================================

Create comprehensive BI and reporting capabilities for the APG composable template system.
"""

import json
from pathlib import Path
from templates.composable.capability import Capability, CapabilityCategory, CapabilityDependency, CapabilityIntegration

def create_advanced_analytics_capability():
    """Create advanced analytics capability"""
    return Capability(
        name="Advanced Analytics",
        category=CapabilityCategory.ANALYTICS,
        description="Advanced data analytics with statistical analysis and data mining",
        version="1.0.0",
        python_requirements=[
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "scikit-learn>=1.3.0",
            "plotly>=5.15.0",
            "dash>=2.11.0",
            "seaborn>=0.12.0"
        ],
        features=[
            "Statistical Analysis",
            "Correlation Analysis",
            "Regression Analysis",
            "Clustering",
            "Dimensionality Reduction",
            "A/B Testing",
            "Cohort Analysis",
            "Interactive Dashboards"
        ],
        compatible_bases=["flask_webapp", "dashboard"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store analytical data and results"),
            CapabilityDependency("analytics/basic_analytics", reason="Build upon basic analytics")
        ],
        integration=CapabilityIntegration(
            models=["AnalyticsJob", "StatisticalResult", "Experiment", "Cohort"],
            views=["AdvancedAnalyticsView", "ExperimentView", "CohortView"],
            apis=["analytics/advanced", "analytics/experiment", "analytics/cohort"],
            templates=["advanced_analytics.html", "experiment_results.html"]
        )
    )

def create_data_warehouse_capability():
    """Create data warehouse capability"""
    return Capability(
        name="Data Warehouse",
        category=CapabilityCategory.ANALYTICS,
        description="Data warehousing with ETL pipelines and dimensional modeling",
        version="1.0.0",
        python_requirements=[
            "sqlalchemy>=2.0.0",
            "alembic>=1.11.0",
            "apache-airflow>=2.6.0",
            "great-expectations>=0.17.0",
            "dbt-core>=1.5.0"
        ],
        features=[
            "ETL Pipeline Management",
            "Dimensional Modeling",
            "Data Quality Checks",
            "Incremental Loading",
            "Change Data Capture",
            "Data Lineage Tracking",
            "Automated Documentation",
            "Performance Optimization"
        ],
        compatible_bases=["microservice", "api_only"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Primary data storage")
        ],
        integration=CapabilityIntegration(
            models=["DataSource", "ETLJob", "DimensionTable", "FactTable", "DataQualityCheck"],
            views=["ETLView", "DataQualityView", "LineageView"],
            apis=["etl/run", "etl/status", "warehouse/query"],
            config_additions={
                "AIRFLOW_HOME": "/opt/airflow",
                "DBT_PROFILES_DIR": "/opt/dbt"
            }
        )
    )

def create_kpi_dashboard_capability():
    """Create KPI dashboard capability"""
    return Capability(
        name="KPI Dashboard",
        category=CapabilityCategory.ANALYTICS,
        description="Key Performance Indicator dashboards with real-time monitoring",
        version="1.0.0",
        python_requirements=[
            "plotly>=5.15.0",
            "dash>=2.11.0",
            "dash-bootstrap-components>=1.4.0",
            "pandas>=2.0.0",
            "redis>=4.6.0"
        ],
        features=[
            "Real-time KPI Monitoring",
            "Custom KPI Definitions",
            "Threshold Alerts",
            "Trend Analysis",
            "Goal Tracking",
            "Executive Dashboards",
            "Mobile Responsive",
            "Drill-down Capabilities"
        ],
        compatible_bases=["dashboard", "flask_webapp"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store KPI data and configurations"),
            CapabilityDependency("communication/websocket_communication", reason="Real-time updates", optional=True)
        ],
        integration=CapabilityIntegration(
            models=["KPI", "KPITarget", "KPIValue", "Dashboard", "Widget"],
            views=["KPIDashboardView", "KPIConfigView"],
            apis=["kpi/current", "kpi/historical", "kpi/alert"],
            templates=["kpi_dashboard.html", "kpi_widget.html"],
            static_files=["css/kpi.css", "js/kpi_realtime.js"]
        )
    )

def create_reporting_engine_capability():
    """Create reporting engine capability"""
    return Capability(
        name="Reporting Engine",
        category=CapabilityCategory.ANALYTICS,
        description="Automated report generation with scheduling and distribution",
        version="1.0.0",
        python_requirements=[
            "reportlab>=4.0.0",
            "openpyxl>=3.1.0",
            "jinja2>=3.1.0",
            "weasyprint>=59.0",
            "celery>=5.3.0",
            "schedule>=1.2.0"
        ],
        features=[
            "PDF Report Generation",
            "Excel Report Export",
            "HTML Reports",
            "Scheduled Reports",
            "Email Distribution",
            "Template Engine",
            "Parameter Queries",
            "Batch Processing"
        ],
        compatible_bases=["flask_webapp", "microservice"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Query data for reports"),
            CapabilityDependency("communication/email", reason="Email report distribution", optional=True)
        ],
        integration=CapabilityIntegration(
            models=["Report", "ReportTemplate", "ReportSchedule", "ReportRun"],
            views=["ReportView", "TemplateView", "ScheduleView"],
            apis=["reports/generate", "reports/schedule", "reports/download"],
            templates=["report_builder.html", "report_viewer.html"],
            config_additions={
                "REPORTS_OUTPUT_DIR": "/var/reports",
                "CELERY_BROKER_URL": "redis://localhost:6379/0"
            }
        )
    )

def create_business_metrics_capability():
    """Create business metrics capability"""
    return Capability(
        name="Business Metrics",
        category=CapabilityCategory.BUSINESS,
        description="Comprehensive business metrics tracking and analysis",
        version="1.0.0",
        python_requirements=[
            "pandas>=2.0.0",
            "plotly>=5.15.0",
            "sqlalchemy>=2.0.0"
        ],
        features=[
            "Revenue Tracking",
            "Customer Metrics",
            "Sales Funnel Analysis",
            "Churn Analysis",
            "LTV Calculation",
            "Unit Economics",
            "Growth Metrics",
            "Profitability Analysis"
        ],
        compatible_bases=["flask_webapp", "dashboard"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store business data"),
            CapabilityDependency("analytics/basic_analytics", reason="Data visualization")
        ],
        integration=CapabilityIntegration(
            models=["Customer", "Sale", "Revenue", "Metric", "Funnel"],
            views=["BusinessMetricsView", "RevenueView", "CustomerView"],
            apis=["metrics/revenue", "metrics/customers", "metrics/funnel"],
            templates=["business_dashboard.html", "revenue_analysis.html"]
        )
    )

def create_data_visualization_capability():
    """Create advanced data visualization capability"""
    return Capability(
        name="Advanced Data Visualization",
        category=CapabilityCategory.ANALYTICS,
        description="Advanced interactive data visualizations and charts",
        version="1.0.0",
        python_requirements=[
            "plotly>=5.15.0",
            "bokeh>=3.2.0",
            "altair>=5.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "d3py>=0.2.8"
        ],
        features=[
            "Interactive Charts",
            "3D Visualizations",
            "Geographic Maps",
            "Network Graphs",
            "Animated Charts",
            "Custom Widgets",
            "Export Capabilities",
            "Responsive Design"
        ],
        compatible_bases=["flask_webapp", "dashboard"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Data source for visualizations")
        ],
        integration=CapabilityIntegration(
            models=["Visualization", "Chart", "Dataset"],
            views=["VisualizationView", "ChartBuilderView"],
            apis=["viz/create", "viz/data", "viz/export"],
            templates=["chart_builder.html", "visualization_gallery.html"],
            static_files=["js/d3.min.js", "js/chart_interactions.js", "css/visualizations.css"]
        )
    )

def save_bi_capabilities():
    """Save all business intelligence capabilities to the filesystem"""
    print("📊 Creating Business Intelligence and Reporting Capabilities")
    print("=" * 70)
    
    # Create capabilities - mixing analytics and business categories
    capabilities = [
        create_advanced_analytics_capability(),
        create_data_warehouse_capability(), 
        create_kpi_dashboard_capability(),
        create_reporting_engine_capability(),
        create_business_metrics_capability(),
        create_data_visualization_capability()
    ]
    
    # Save each capability to appropriate category
    base_dir = Path(__file__).parent / 'templates' / 'composable' / 'capabilities'
    
    for capability in capabilities:
        # Determine directory based on category
        category_dir = base_dir / capability.category.value
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create capability directory
        cap_name = capability.name.lower().replace(' ', '_')
        cap_dir = category_dir / cap_name
        cap_dir.mkdir(exist_ok=True)
        
        # Create standard directories
        for subdir in ['models', 'views', 'templates', 'static', 'tests', 'config', 'scripts']:
            (cap_dir / subdir).mkdir(exist_ok=True)
        
        # Save capability.json
        with open(cap_dir / 'capability.json', 'w') as f:
            json.dump(capability.to_dict(), f, indent=2)
        
        # Create integration template
        create_bi_integration_template(cap_dir, capability)
        
        print(f"  ✅ Created {capability.name} ({capability.category.value})")
    
    print(f"\n📁 BI capabilities saved to: {base_dir}")
    return capabilities

def create_bi_integration_template(cap_dir: Path, capability: Capability):
    """Create integration template for BI capability"""
    cap_name_snake = capability.name.lower().replace(' ', '_')
    cap_name_class = capability.name.replace(' ', '').replace('/', '')
    
    integration_content = f'''"""
{capability.name} Integration
{'=' * (len(capability.name) + 12)}

Integration logic for the {capability.name} capability.
Handles BI/analytics-specific setup and configuration.
"""

import logging
from flask import Blueprint
from flask_appbuilder import BaseView

# Configure logging
log = logging.getLogger(__name__)

# Create capability blueprint
{cap_name_snake}_bp = Blueprint(
    '{cap_name_snake}',
    __name__,
    url_prefix='/{capability.category.value}/{cap_name_snake}',
    template_folder='templates',
    static_folder='static'
)


def integrate_{cap_name_snake}(app, appbuilder, db):
    """
    Integrate {capability.name} capability into the application.
    
    Args:
        app: Flask application instance
        appbuilder: Flask-AppBuilder instance
        db: SQLAlchemy database instance
    """
    try:
        # Register blueprint
        app.register_blueprint({cap_name_snake}_bp)
        
        # Import and register models
        from .models import *  # noqa
        
        # Import and register views
        from .views import *  # noqa
        
        # Apply BI-specific configuration
        config_additions = {repr(capability.integration.config_additions)}
        for key, value in config_additions.items():
            app.config[key] = value
        
        # Initialize BI service
        bi_service = {cap_name_class}Service(app, appbuilder, db)
        app.extensions['{cap_name_snake}_service'] = bi_service
        
        # Register views with AppBuilder
        appbuilder.add_view(
            {cap_name_class}View,
            "{capability.name}",
            icon="fa-chart-bar",
            category="Analytics",
            category_icon="fa-analytics"
        )
        
        log.info(f"Successfully integrated {capability.name} capability")
        
    except Exception as e:
        log.error(f"Failed to integrate {capability.name} capability: {{e}}")
        raise


class {cap_name_class}Service:
    """
    Main service class for {capability.name}.
    
    Handles BI/analytics processing and data operations.
    """
    
    def __init__(self, app, appbuilder, db):
        self.app = app
        self.appbuilder = appbuilder
        self.db = db
        self.initialize_service()
    
    def initialize_service(self):
        """Initialize BI service"""
        log.info(f"Initializing {capability.name} service")
        
        try:
            # Setup BI-specific components
            self.setup_analytics_engine()
            
            # Initialize data connections
            self.setup_data_connections()
            
        except Exception as e:
            log.error(f"Error initializing BI service: {{e}}")
    
    def setup_analytics_engine(self):
        """Setup analytics processing engine"""
        # Analytics engine setup logic
        pass
    
    def setup_data_connections(self):
        """Setup data source connections"""
        # Data connection setup logic
        pass
    
    def generate_insights(self, data):
        """Generate business insights from data"""
        # Insight generation logic
        return {{"insights": [], "recommendations": []}}


class {cap_name_class}View(BaseView):
    """
    Main view for {capability.name} capability.
    """
    
    route_base = "/{cap_name_snake}"
    
    @expose("/")
    def index(self):
        """Main dashboard view"""
        return self.render_template("{cap_name_snake}_dashboard.html")
    
    @expose("/analytics")
    def analytics(self):
        """Analytics view"""
        return self.render_template("{cap_name_snake}_analytics.html")
'''
    
    # Save integration template
    with open(cap_dir / 'integration.py.template', 'w') as f:
        f.write(integration_content)
    
    # Create models template for BI
    models_content = f'''"""
{capability.name} Models
{'=' * (len(capability.name) + 7)}

Database models for {capability.name} capability.
"""

from flask_appbuilder import Model
from flask_appbuilder.models.mixins import AuditMixin
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime


class BIBaseModel(AuditMixin, Model):
    """Base model for BI entities"""
    __abstract__ = True
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Add BI-specific models based on capability
{generate_bi_models(capability)}
'''
    
    with open(cap_dir / 'models' / '__init__.py.template', 'w') as f:
        f.write(models_content)

def generate_bi_models(capability: Capability) -> str:
    """Generate BI-specific models based on capability type"""
    if "KPI" in capability.name:
        return '''
class KPI(BIBaseModel):
    """Key Performance Indicator model"""
    __tablename__ = 'bi_kpis'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text)
    formula = Column(Text)
    unit = Column(String(32))
    target_value = Column(Float)
    current_value = Column(Float)
    status = Column(String(32))  # green, yellow, red
    
    values = relationship("KPIValue", back_populates="kpi")


class KPIValue(BIBaseModel):
    """KPI historical values"""
    __tablename__ = 'bi_kpi_values'
    
    id = Column(Integer, primary_key=True)
    kpi_id = Column(Integer, ForeignKey('bi_kpis.id'))
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    kpi = relationship("KPI", back_populates="values")
'''
    elif "Report" in capability.name:
        return '''
class Report(BIBaseModel):
    """Report definition model"""
    __tablename__ = 'bi_reports'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text)
    query = Column(Text)
    template_path = Column(String(512))
    output_format = Column(String(32))
    
    schedules = relationship("ReportSchedule", back_populates="report")


class ReportSchedule(BIBaseModel):
    """Report scheduling model"""
    __tablename__ = 'bi_report_schedules'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey('bi_reports.id'))
    cron_expression = Column(String(128))
    recipients = Column(Text)  # JSON array of email addresses
    last_run = Column(DateTime)
    next_run = Column(DateTime)
    active = Column(Boolean, default=True)
    
    report = relationship("Report", back_populates="schedules")
'''
    elif "Warehouse" in capability.name:
        return '''
class DataSource(BIBaseModel):
    """Data source configuration"""
    __tablename__ = 'bi_data_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    source_type = Column(String(64))  # database, api, file
    connection_string = Column(String(512))
    configuration = Column(JSON)
    last_sync = Column(DateTime)
    
    etl_jobs = relationship("ETLJob", back_populates="data_source")


class ETLJob(BIBaseModel):
    """ETL job tracking"""
    __tablename__ = 'bi_etl_jobs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    data_source_id = Column(Integer, ForeignKey('bi_data_sources.id'))
    status = Column(String(32))  # pending, running, completed, failed
    records_processed = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    data_source = relationship("DataSource", back_populates="etl_jobs")
'''
    else:
        return '''
# Generic BI analytics model
class AnalyticsJob(BIBaseModel):
    """Generic analytics job"""
    __tablename__ = 'bi_analytics_jobs'
    
    id = Column(Integer, primary_key=True)
    job_name = Column(String(256), nullable=False)
    job_type = Column(String(64))
    parameters = Column(JSON)
    results = Column(JSON)
    status = Column(String(32), default='pending')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
'''

def main():
    """Create all business intelligence capabilities"""
    try:
        capabilities = save_bi_capabilities()
        
        print(f"\n🎉 Successfully created {len(capabilities)} BI capabilities!")
        print(f"\n📋 Business Intelligence Capabilities Created:")
        for cap in capabilities:
            print(f"   • {cap.name} ({cap.category.value}) - {cap.description}")
        
        print(f"\n🚀 These capabilities enable:")
        print(f"   • Advanced statistical analysis and data mining")
        print(f"   • Enterprise data warehousing with ETL pipelines")
        print(f"   • Real-time KPI monitoring and dashboards")
        print(f"   • Automated report generation and distribution")
        print(f"   • Comprehensive business metrics tracking")
        print(f"   • Interactive data visualizations and charts")
        
        return True
        
    except Exception as e:
        print(f"💥 Error creating BI capabilities: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)