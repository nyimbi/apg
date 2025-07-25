#!/usr/bin/env python3
"""
Create Template Structure
========================

Creates the complete directory structure and placeholder files for all APG application templates.
"""

import os
from pathlib import Path
from typing import Dict, List

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
    base_path = Path(__file__).parent / 'templates' / 'application_templates'
    
    for template_id, metadata in TEMPLATES.items():
        template_path = base_path / template_id
        template_path.mkdir(parents=True, exist_ok=True)
        
        # Create template.json
        create_template_json(template_path, template_id, metadata)
        
        # Create directory structure
        create_template_directories(template_path, metadata)
        
        # Create placeholder files
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
        "requirements": [
            "Flask-AppBuilder>=4.3.0",
            "Flask>=2.3.0",
            "SQLAlchemy>=2.0.0"
        ],
        "variables": {
            "project_name": "{{project_name}}",
            "project_description": "{{project_description}}",
            "author": "{{author}}",
            "database_url": "{{database_url}}",
            "secret_key": "{{secret_key}}"
        },
        "files": [
            "app.py.template",
            "config.py.template",
            "requirements.txt.template",
            "README.md.template"
        ]
    }
    
    with open(template_path / 'template.json', 'w') as f:
        import json
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

def create_template_files(template_path: Path, metadata: Dict):
    """Create template placeholder files"""
    # Main files
    files_to_create = [
        'app.py.template',
        'config.py.template', 
        'requirements.txt.template',
        'README.md.template',
        'agents/__init__.py.template',
        'models/__init__.py.template',
        'views/__init__.py.template',
        'tests/__init__.py.template'
    ]
    
    # Add digital twins files if needed
    if metadata.get('digital_twins'):
        files_to_create.append('digital_twins/__init__.py.template')
    
    for file_path in files_to_create:
        full_path = template_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder content
        content = f"""# APG Template File: {file_path}
# Template: {metadata['name']}
# Domain: {metadata['domain']}
#
# This file will contain the complete implementation for {metadata['name']}.
# TODO: Implement {file_path} for {metadata['name']}
"""
        
        with open(full_path, 'w') as f:
            f.write(content)

if __name__ == '__main__':
    print("🚀 Creating APG Application Template Structure")
    print("=" * 60)
    create_template_structure()
    print(f"\n✅ Created {len(TEMPLATES)} application templates")
    print("📁 Template structure ready for individual implementation")
    print("\nNext steps:")
    print("1. Implement each template individually")
    print("2. Test each template as a working Flask-AppBuilder application")
    print("3. Integrate with APG code generator")