#!/usr/bin/env python3
"""
Create Template Structure
========================

Creates the complete directory structure and executable starter files for all APG application templates.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]

# Template definitions with metadata
TEMPLATES = {
    'basic/simple_agent': {
        'name': 'Simple Agent',
        'description': 'Basic agent with simple methods and web interface',
        'complexity': 'Beginner',
        'domain': 'General',
        'agents': ['BasicAgent'],
        'features': ['Agent Runtime', 'Basic Methods', 'Web Dashboard'],
        'databases': ['simple_data'],
    },
    
    'basic/crud_app': {
        'name': 'CRUD Application',
        'description': 'Complete Create, Read, Update, Delete application',
        'complexity': 'Beginner',
        'domain': 'General',
        'agents': ['DataManagerAgent'],
        'features': ['CRUD Operations', 'Database Management', 'Admin Interface'],
        'databases': ['main_data'],
    },
    
    'basic/dashboard': {
        'name': 'Analytics Dashboard',
        'description': 'Real-time analytics dashboard with charts and metrics',
        'complexity': 'Intermediate',
        'domain': 'Analytics',
        'agents': ['AnalyticsAgent', 'MetricsCollector'],
        'features': ['Real-time Charts', 'KPI Tracking', 'Data Visualization'],
        'databases': ['analytics_data', 'metrics'],
    },
    
    'intelligence/ai_platform': {
        'name': 'AI Intelligence Platform',
        'description': 'Complete AI platform with multiple AI agents and services',
        'complexity': 'Expert',
        'domain': 'Artificial Intelligence',
        'agents': ['AIOrchestrator', 'ModelManager', 'InferenceAgent'],
        'features': ['Model Management', 'Multi-Agent AI', 'Inference Pipeline', 'AI Analytics'],
        'databases': ['models', 'inference_logs', 'training_data'],
    },
    
    'intelligence/knowledge_base': {
        'name': 'Knowledge Base System',
        'description': 'Intelligent knowledge management with semantic search',
        'complexity': 'Advanced',
        'domain': 'Knowledge Management',
        'agents': ['KnowledgeAgent', 'SearchAgent', 'IndexingAgent'],
        'features': ['Semantic Search', 'Document Processing', 'Knowledge Graph'],
        'databases': ['documents', 'knowledge_graph', 'search_index'],
    },
    
    'intelligence/chat_assistant': {
        'name': 'Conversational AI Assistant',
        'description': 'Advanced chatbot with context awareness and learning',
        'complexity': 'Advanced',
        'domain': 'Conversational AI',
        'agents': ['ChatAgent', 'ContextManager', 'LearningAgent'],
        'features': ['Natural Language Processing', 'Context Awareness', 'Multi-turn Conversations'],
        'databases': ['conversations', 'context_memory', 'user_profiles'],
    },
    
    'intelligence/nlp_processor': {
        'name': 'NLP Processing Pipeline',
        'description': 'Complete natural language processing pipeline',
        'complexity': 'Advanced',
        'domain': 'Natural Language Processing',
        'agents': ['NLPAgent', 'TextAnalyzer', 'SentimentAnalyzer'],
        'features': ['Text Processing', 'Sentiment Analysis', 'Entity Recognition'],
        'databases': ['text_corpus', 'processing_results', 'models'],
    },
    
    'marketplace/ecommerce': {
        'name': 'E-Commerce Platform',
        'description': 'Complete online marketplace with products, orders, and payments',
        'complexity': 'Advanced',
        'domain': 'E-Commerce',
        'agents': ['ProductAgent', 'OrderAgent', 'PaymentAgent', 'InventoryAgent'],
        'features': ['Product Catalog', 'Shopping Cart', 'Order Management', 'Payment Processing'],
        'databases': ['products', 'orders', 'customers', 'inventory'],
    },
    
    'marketplace/b2b_platform': {
        'name': 'B2B Trading Platform',
        'description': 'Business-to-business marketplace with advanced trading features',
        'complexity': 'Expert',
        'domain': 'B2B Commerce',
        'agents': ['TradingAgent', 'MatchingAgent', 'ContractAgent', 'ComplianceAgent'],
        'features': ['Business Matching', 'Contract Management', 'Bulk Trading', 'Compliance Tracking'],
        'databases': ['businesses', 'trades', 'contracts', 'compliance_records'],
    },
    
    'marketplace/service_marketplace': {
        'name': 'Service Marketplace',
        'description': 'Platform for service providers and consumers',
        'complexity': 'Advanced',
        'domain': 'Service Economy',
        'agents': ['ServiceAgent', 'BookingAgent', 'RatingAgent', 'DisputeAgent'],
        'features': ['Service Listings', 'Booking System', 'Rating & Reviews', 'Dispute Resolution'],
        'databases': ['services', 'bookings', 'reviews', 'disputes'],
    },
    
    'marketplace/gig_economy': {
        'name': 'Gig Economy Platform',
        'description': 'Platform for freelancers and gig workers',
        'complexity': 'Advanced',
        'domain': 'Gig Economy',
        'agents': ['GigAgent', 'WorkerAgent', 'ClientAgent', 'PayoutAgent'],
        'features': ['Gig Matching', 'Skill Verification', 'Project Management', 'Automated Payouts'],
        'databases': ['gigs', 'workers', 'clients', 'transactions'],
    },
    
    'iot/device_monitor': {
        'name': 'IoT Device Monitor',
        'description': 'Monitor and manage IoT devices with real-time data',
        'complexity': 'Intermediate',
        'domain': 'Internet of Things',
        'agents': ['DeviceAgent', 'MonitoringAgent', 'AlertAgent'],
        'features': ['Device Management', 'Real-time Monitoring', 'Alert System', 'Data Visualization'],
        'databases': ['devices', 'sensor_data', 'alerts'],
        'digital_twins': ['SensorTwin', 'DeviceTwin'],
    },
    
    'iot/smart_factory': {
        'name': 'Smart Factory System',
        'description': 'Industrial IoT platform for smart manufacturing',
        'complexity': 'Expert',
        'domain': 'Industrial IoT',
        'agents': ['ProductionAgent', 'QualityAgent', 'MaintenanceAgent', 'EfficiencyAgent'],
        'features': ['Production Monitoring', 'Quality Control', 'Predictive Maintenance', 'Efficiency Analytics'],
        'databases': ['production_data', 'quality_metrics', 'maintenance_logs', 'efficiency_reports'],
        'digital_twins': ['MachineTwin', 'ProductionLineTwin', 'FactoryTwin'],
    },
    
    'iot/environmental_sensor': {
        'name': 'Environmental Monitoring',
        'description': 'Environmental sensor network with data analysis',
        'complexity': 'Intermediate',
        'domain': 'Environmental Monitoring',
        'agents': ['SensorAgent', 'AnalyticsAgent', 'PredictionAgent'],
        'features': ['Multi-sensor Integration', 'Environmental Analytics', 'Trend Prediction'],
        'databases': ['sensor_readings', 'environmental_data', 'predictions'],
        'digital_twins': ['SensorNetworkTwin', 'EnvironmentTwin'],
    },
    
    'iot/fleet_management': {
        'name': 'Fleet Management System',
        'description': 'Vehicle fleet tracking and management platform',
        'complexity': 'Advanced',
        'domain': 'Fleet Management',
        'agents': ['VehicleAgent', 'RoutingAgent', 'MaintenanceAgent', 'FuelAgent'],
        'features': ['GPS Tracking', 'Route Optimization', 'Maintenance Scheduling', 'Fuel Management'],
        'databases': ['vehicles', 'routes', 'maintenance', 'fuel_data'],
        'digital_twins': ['VehicleTwin', 'FleetTwin'],
    },
    
    'fintech/trading_platform': {
        'name': 'Trading Platform',
        'description': 'Financial trading platform with real-time market data',
        'complexity': 'Expert',
        'domain': 'Financial Services',
        'agents': ['TradingAgent', 'RiskAgent', 'MarketDataAgent', 'ComplianceAgent'],
        'features': ['Real-time Trading', 'Risk Management', 'Market Analysis', 'Regulatory Compliance'],
        'databases': ['trades', 'market_data', 'risk_metrics', 'compliance_logs'],
    },
    
    'fintech/payment_processor': {
        'name': 'Payment Processing System',
        'description': 'Secure payment processing with fraud detection',
        'complexity': 'Expert',
        'domain': 'Payment Processing',
        'agents': ['PaymentAgent', 'FraudAgent', 'SettlementAgent', 'ReconciliationAgent'],
        'features': ['Payment Processing', 'Fraud Detection', 'Settlement', 'Reconciliation'],
        'databases': ['payments', 'fraud_scores', 'settlements', 'reconciliation'],
    },
    
    'fintech/loan_origination': {
        'name': 'Loan Origination System',
        'description': 'Automated loan processing and underwriting',
        'complexity': 'Advanced',
        'domain': 'Lending',
        'agents': ['LoanAgent', 'UnderwritingAgent', 'CreditAgent', 'DocumentAgent'],
        'features': ['Loan Applications', 'Credit Scoring', 'Automated Underwriting', 'Document Processing'],
        'databases': ['loan_applications', 'credit_reports', 'underwriting_results', 'documents'],
    },
    
    'fintech/compliance_monitor': {
        'name': 'Financial Compliance Monitor',
        'description': 'Regulatory compliance monitoring and reporting',
        'complexity': 'Expert',
        'domain': 'Financial Compliance',
        'agents': ['ComplianceAgent', 'AuditAgent', 'ReportingAgent', 'AlertAgent'],
        'features': ['Compliance Monitoring', 'Audit Trails', 'Regulatory Reporting', 'Risk Alerts'],
        'databases': ['compliance_rules', 'audit_logs', 'reports', 'violations'],
    },
    
    'healthcare/patient_management': {
        'name': 'Patient Management System',
        'description': 'Complete electronic health records and patient management',
        'complexity': 'Expert',
        'domain': 'Healthcare',
        'agents': ['PatientAgent', 'AppointmentAgent', 'MedicalRecordsAgent', 'BillingAgent'],
        'features': ['Electronic Health Records', 'Appointment Scheduling', 'Medical History', 'Billing Integration'],
        'databases': ['patients', 'appointments', 'medical_records', 'billing'],
    },
    
    'healthcare/telemedicine': {
        'name': 'Telemedicine Platform',
        'description': 'Remote healthcare delivery platform',
        'complexity': 'Advanced',
        'domain': 'Telemedicine',
        'agents': ['ConsultationAgent', 'DiagnosticAgent', 'PrescriptionAgent', 'MonitoringAgent'],
        'features': ['Video Consultations', 'Remote Diagnostics', 'Digital Prescriptions', 'Health Monitoring'],
        'databases': ['consultations', 'diagnostics', 'prescriptions', 'health_data'],
    },
    
    'healthcare/clinical_trials': {
        'name': 'Clinical Trials Management',
        'description': 'Clinical research and trial management system',
        'complexity': 'Expert',
        'domain': 'Clinical Research',
        'agents': ['TrialAgent', 'ParticipantAgent', 'DataAgent', 'ComplianceAgent'],
        'features': ['Trial Management', 'Participant Tracking', 'Data Collection', 'Regulatory Compliance'],
        'databases': ['trials', 'participants', 'trial_data', 'compliance'],
    },
    
    'healthcare/health_analytics': {
        'name': 'Health Analytics Platform',
        'description': 'Healthcare data analytics and insights',
        'complexity': 'Advanced',
        'domain': 'Health Analytics',
        'agents': ['AnalyticsAgent', 'PopulationAgent', 'OutcomeAgent', 'PredictiveAgent'],
        'features': ['Health Analytics', 'Population Health', 'Outcome Analysis', 'Predictive Modeling'],
        'databases': ['health_metrics', 'population_data', 'outcomes', 'predictions'],
    },
    
    'logistics/supply_chain': {
        'name': 'Supply Chain Management',
        'description': 'End-to-end supply chain visibility and optimization',
        'complexity': 'Expert',
        'domain': 'Supply Chain',
        'agents': ['SupplyAgent', 'DemandAgent', 'LogisticsAgent', 'OptimizationAgent'],
        'features': ['Supply Planning', 'Demand Forecasting', 'Logistics Optimization', 'Supply Chain Visibility'],
        'databases': ['suppliers', 'inventory', 'shipments', 'demand_forecasts'],
    },
    
    'logistics/warehouse_management': {
        'name': 'Warehouse Management System',
        'description': 'Advanced warehouse operations and inventory management',
        'complexity': 'Advanced',
        'domain': 'Warehouse Management',
        'agents': ['InventoryAgent', 'PickingAgent', 'ShippingAgent', 'OptimizationAgent'],
        'features': ['Inventory Management', 'Order Picking', 'Shipping Management', 'Layout Optimization'],
        'databases': ['inventory', 'orders', 'shipments', 'warehouse_layout'],
    },
    
    'logistics/shipping_tracker': {
        'name': 'Shipping Tracker',
        'description': 'Real-time package tracking and delivery management',
        'complexity': 'Intermediate',
        'domain': 'Shipping & Delivery',
        'agents': ['TrackingAgent', 'DeliveryAgent', 'RouteAgent', 'NotificationAgent'],
        'features': ['Package Tracking', 'Delivery Management', 'Route Optimization', 'Customer Notifications'],
        'databases': ['packages', 'deliveries', 'routes', 'tracking_events'],
    },
    
    'logistics/inventory_optimizer': {
        'name': 'Inventory Optimization',
        'description': 'AI-powered inventory optimization and demand planning',
        'complexity': 'Advanced',
        'domain': 'Inventory Management',
        'agents': ['InventoryAgent', 'DemandAgent', 'OptimizationAgent', 'ReplenishmentAgent'],
        'features': ['Demand Planning', 'Inventory Optimization', 'Automated Replenishment', 'Cost Analysis'],
        'databases': ['inventory_levels', 'demand_history', 'optimization_models', 'cost_analysis'],
    },
    
    'enterprise/erp_system': {
        'name': 'Enterprise Resource Planning',
        'description': 'Complete ERP system for enterprise resource management',
        'complexity': 'Expert',
        'domain': 'Enterprise Management',
        'agents': ['ERPAgent', 'FinanceAgent', 'HRAgent', 'OperationsAgent'],
        'features': ['Financial Management', 'Human Resources', 'Operations Management', 'Reporting'],
        'databases': ['financials', 'employees', 'operations', 'reports'],
    },
    
    'enterprise/crm_platform': {
        'name': 'Customer Relationship Management',
        'description': 'Comprehensive CRM platform for customer management',
        'complexity': 'Advanced',
        'domain': 'Customer Management',
        'agents': ['CustomerAgent', 'SalesAgent', 'MarketingAgent', 'ServiceAgent'],
        'features': ['Customer Management', 'Sales Pipeline', 'Marketing Campaigns', 'Customer Service'],
        'databases': ['customers', 'sales', 'marketing', 'service_tickets'],
    },
    
    'enterprise/hr_management': {
        'name': 'Human Resources Management',
        'description': 'Complete HR management system with employee lifecycle',
        'complexity': 'Advanced',
        'domain': 'Human Resources',
        'agents': ['HRAgent', 'RecruitmentAgent', 'PerformanceAgent', 'PayrollAgent'],
        'features': ['Employee Management', 'Recruitment', 'Performance Management', 'Payroll'],
        'databases': ['employees', 'recruitment', 'performance', 'payroll'],
    },
    
    'enterprise/business_intelligence': {
        'name': 'Business Intelligence Platform',
        'description': 'Enterprise BI platform with advanced analytics',
        'complexity': 'Expert',
        'domain': 'Business Intelligence',
        'agents': ['BIAgent', 'AnalyticsAgent', 'ReportingAgent', 'DashboardAgent'],
        'features': ['Data Analytics', 'Interactive Dashboards', 'Automated Reporting', 'Predictive Analytics'],
        'databases': ['business_data', 'analytics_results', 'reports', 'dashboards'],
    },
}

def create_template_structure():
    """Create the complete template structure"""
    base_path = REPO_ROOT / 'templates' / 'application_templates'
    
    for template_id, metadata in TEMPLATES.items():
        template_path = base_path / template_id
        template_path.mkdir(parents=True, exist_ok=True)
        
        # Create template.json
        create_template_json(template_path, template_id, metadata)
        
        # Create directory structure
        create_template_directories(template_path, metadata)
        
        # Create executable starter files
        create_template_files(template_path, metadata)
        
        print(f"✅ Created template: {template_id}")

def create_template_json(template_path: Path, template_id: str, metadata: Dict):
    """Create template.json metadata file"""
    template_json = {
        "template_id": template_id,
        "name": metadata['name'],
        "description": metadata['description'],
        "complexity": metadata['complexity'],
        "domain": metadata['domain'],
        "version": "1.0.0",
        "apg_version": ">=1.0.0",
        "agents": metadata.get('agents', []),
        "digital_twins": metadata.get('digital_twins', []),
        "features": metadata['features'],
        "databases": metadata.get('databases', []),
        "requirements": [],
        "target": "python",
        "variables": {
            "project_name": "{{project_name}}",
            "project_description": "{{project_description}}",
            "author": "{{author}}",
            "database_url": "{{database_url}}",
            "secret_key": "{{secret_key}}"
        },
        "files": template_file_manifest(metadata)
    }
    
    with open(template_path / 'template.json', 'w') as f:
        json.dump(template_json, f, indent=2)

def create_template_directories(template_path: Path, metadata: Dict):
    """Create template directory structure"""
    directories = [
        'agents',
        'models', 
        'views',
        'templates/html',
        'static/css',
        'static/js',
        'tests',
        'docs'
    ]
    
    # Add digital twins directory if needed
    if metadata.get('digital_twins'):
        directories.append('digital_twins')
    
    # Add workflows directory for complex templates
    if metadata['complexity'] in ['Advanced', 'Expert']:
        directories.append('workflows')
    
    for directory in directories:
        (template_path / directory).mkdir(parents=True, exist_ok=True)

def template_file_manifest(metadata: Dict) -> List[str]:
    """Return every template file registered for a generated template."""
    files_to_create = [
        'app.py.template',
        'config.py.template', 
        'requirements.txt.template',
        'README.md.template',
        'agents/__init__.py.template',
        'models/__init__.py.template',
        'views/__init__.py.template',
        'tests/__init__.py.template',
        'tests/__main__.py.template'
    ]
    
    if metadata.get('digital_twins'):
        files_to_create.append('digital_twins/__init__.py.template')

    return sorted(files_to_create)


def _slug(value: str) -> str:
    return value.lower().replace('&', 'and').replace(' ', '_').replace('-', '_')


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=True, indent=4)


def _app_template(metadata: Dict) -> str:
    return f'''"""Executable APG starter for {metadata['name']}."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agents import build_agent_registry
from models import seed_records
from views import build_dashboard_payload, list_routes


TEMPLATE_NAME = {json.dumps(metadata['name'])}
DESCRIPTION = {json.dumps(metadata['description'])}
FEATURES = {_json(metadata['features'])}
DATABASES = {_json(metadata.get('databases', []))}


def create_application(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a dependency-free application descriptor."""
    runtime_config = config or {{}}
    records = seed_records()
    agents = build_agent_registry()
    return {{
        "name": runtime_config.get("project_name", "{{{{project_name}}}}"),
        "description": runtime_config.get("project_description", DESCRIPTION),
        "template": TEMPLATE_NAME,
        "features": list(FEATURES),
        "databases": list(DATABASES),
        "agents": list(agents),
        "routes": list_routes(),
        "records": [asdict(record) for record in records],
        "dashboard": build_dashboard_payload(records, agents, runtime_config),
    }}


def health_check() -> dict[str, Any]:
    app = create_application()
    return {{
        "status": "ready",
        "template": TEMPLATE_NAME,
        "feature_count": len(app["features"]),
        "agent_count": len(app["agents"]),
    }}


if __name__ == "__main__":
    status = health_check()
    print(f"{{status['template']}}: {{status['status']}} ({{status['feature_count']}} features)")
'''


def _config_template(metadata: Dict) -> str:
    return f'''"""Configuration defaults for {metadata['name']}."""

from __future__ import annotations

import os


class Config:
    PROJECT_NAME = os.environ.get("PROJECT_NAME", "{{{{project_name}}}}")
    PROJECT_DESCRIPTION = os.environ.get("PROJECT_DESCRIPTION", "{{{{project_description}}}}")
    AUTHOR = os.environ.get("PROJECT_AUTHOR", "{{{{author}}}}")
    DATABASE_URL = os.environ.get("DATABASE_URL", "{{{{database_url}}}}")
    SECRET_KEY = os.environ.get("SECRET_KEY", "{{{{secret_key}}}}")
    TEMPLATE_NAME = {json.dumps(metadata['name'])}
    DOMAIN = {json.dumps(metadata['domain'])}
    FEATURES = {_json(metadata['features'])}
    AGENTS = {_json(metadata.get('agents', []))}
'''


def _models_template(metadata: Dict) -> str:
    items = [(_slug(feature), feature) for feature in metadata['features']] or [('core', 'Core Workflow')]
    return f'''"""Domain records for {metadata['name']}."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


DATABASES = {_json(metadata.get('databases', []))}
SEED_ITEMS = {_json(items)}


@dataclass(slots=True)
class TemplateRecord:
    key: str
    label: str
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def activate(self) -> None:
        self.status = "active"


def seed_records() -> list[TemplateRecord]:
    return [
        TemplateRecord(key=key, label=label, metadata={{"database": DATABASES[index % len(DATABASES)] if DATABASES else "default"}})
        for index, (key, label) in enumerate(SEED_ITEMS)
    ]
'''


def _agents_template(metadata: Dict) -> str:
    agents = metadata.get('agents', []) or ['ApplicationAgent']
    return f'''"""Agent registry for {metadata['name']}."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


AGENTS = {_json(agents)}


@dataclass(slots=True)
class TemplateAgent:
    name: str
    capabilities: list[str] = field(default_factory=list)

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        return {{
            "agent": self.name,
            "accepted": True,
            "event": dict(event),
            "handled_at": datetime.now(timezone.utc).isoformat(),
        }}


def build_agent_registry() -> dict[str, TemplateAgent]:
    return {{
        agent_name: TemplateAgent(agent_name, ["observe", "validate", "act"])
        for agent_name in AGENTS
    }}
'''


def _views_template(metadata: Dict) -> str:
    routes = [
        {"name": _slug(feature), "path": "/" + _slug(feature).replace('_', '-'), "feature": feature}
        for feature in metadata['features']
    ]
    return f'''"""View payload builders for {metadata['name']}."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


ROUTES = {_json(routes)}


def list_routes() -> list[dict[str, str]]:
    return [dict(route) for route in ROUTES]


def build_dashboard_payload(records: list[Any], agents: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime_config = config or {{}}
    return {{
        "title": runtime_config.get("project_name", "{{{{project_name}}}}"),
        "summary": {json.dumps(metadata['description'])},
        "record_count": len(records),
        "agent_count": len(agents),
        "routes": list_routes(),
        "records": [asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record) for record in records],
    }}
'''


def _tests_template(metadata: Dict) -> str:
    return f'''"""Smoke tests for generated {metadata['name']} projects."""

from __future__ import annotations


def smoke_test() -> bool:
    from app import create_application, health_check

    application = create_application()
    status = health_check()
    assert status["status"] == "ready"
    assert application["features"]
    assert application["routes"]
    return True
'''


def _tests_main_template(metadata: Dict) -> str:
    return f'''"""Command-line smoke test runner for generated {metadata['name']} projects."""

from __future__ import annotations

from . import smoke_test


if __name__ == "__main__":
    raise SystemExit(0 if smoke_test() else 1)
'''


def _digital_twins_template(metadata: Dict) -> str:
    twins = metadata.get('digital_twins') or ['AssetTwin']
    return f'''"""Digital twin definitions for generated {metadata['name']} projects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


TWIN_TYPES = {_json(twins)}


@dataclass(slots=True)
class DigitalTwin:
    twin_id: str
    twin_type: str
    state: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        self.state.update(changes)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return dict(self.state)


def seed_twins() -> list[DigitalTwin]:
    return [DigitalTwin(twin_id=f"{{twin_type.lower()}}-001", twin_type=twin_type) for twin_type in TWIN_TYPES]
'''


def _readme_template(metadata: Dict) -> str:
    features = '\n'.join(f'- {feature}' for feature in metadata['features'])
    agents = '\n'.join(f'- {agent}' for agent in metadata.get('agents', [])) or '- ApplicationAgent'
    databases = '\n'.join(f'- {database}' for database in metadata.get('databases', [])) or '- default'
    return f'''# {{{{project_name}}}}

Generated from the APG **{metadata['name']}** template.

{metadata['description']}

## Features

{features}

## Agents

{agents}

## Data Stores

{databases}

## Run

```bash
python generated/app.py
```

## Smoke Test

```bash
python -c "from tests import smoke_test; smoke_test()"
```
'''


def template_content(file_path: str, metadata: Dict) -> str:
    """Build executable starter content for a template file."""
    if file_path == 'app.py.template':
        return _app_template(metadata)
    if file_path == 'config.py.template':
        return _config_template(metadata)
    if file_path == 'requirements.txt.template':
        return '\n'.join([
            "# APG Generated Application Requirements",
            "# The default Python compiler target uses only the Python standard library.",
        ]) + '\n'
    if file_path == 'README.md.template':
        return _readme_template(metadata)
    if file_path == 'agents/__init__.py.template':
        return _agents_template(metadata)
    if file_path == 'models/__init__.py.template':
        return _models_template(metadata)
    if file_path == 'views/__init__.py.template':
        return _views_template(metadata)
    if file_path == 'tests/__init__.py.template':
        return _tests_template(metadata)
    if file_path == 'tests/__main__.py.template':
        return _tests_main_template(metadata)
    if file_path == 'digital_twins/__init__.py.template':
        return _digital_twins_template(metadata)
    raise ValueError(f"Unsupported template file: {file_path}")


def create_template_files(template_path: Path, metadata: Dict):
    """Create executable starter template files."""
    for file_path in template_file_manifest(metadata):
        full_path = template_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(full_path, 'w') as f:
            f.write(template_content(file_path, metadata))

if __name__ == '__main__':
    print("🚀 Creating APG Application Template Structure")
    print("=" * 60)
    create_template_structure()
    print(f"\n✅ Created {len(TEMPLATES)} application templates")
    print("📁 Template structure ready for use")
    print("\nNext steps:")
    print("1. Review generated template metadata")
    print("2. Run generated starter smoke tests")
    print("3. Integrate with APG code generator")
