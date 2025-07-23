#!/usr/bin/env python3
"""
APG Template Types and Configuration
===================================

Defines available project templates and configuration options.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


class TemplateType(Enum):
	"""Available APG project templates"""
	
	# Basic Templates
	BASIC_AGENT = "basic_agent"
	SIMPLE_WORKFLOW = "simple_workflow"
	DATABASE_CRUD = "database_crud"
	
	# Advanced Templates
	TASK_MANAGEMENT = "task_management"
	E_COMMERCE = "e_commerce"
	DIGITAL_TWIN = "digital_twin"
	AI_ASSISTANT = "ai_assistant"
	
	# Enterprise Templates
	MICROSERVICES = "microservices"
	ENTERPRISE_DASHBOARD = "enterprise_dashboard"
	DATA_PIPELINE = "data_pipeline"
	IOT_PLATFORM = "iot_platform"
	
	# Industry-Specific
	FINTECH_PLATFORM = "fintech_platform"
	HEALTHCARE_SYSTEM = "healthcare_system"
	LOGISTICS_TRACKER = "logistics_tracker"
	SOCIAL_NETWORK = "social_network"


@dataclass
class ProjectConfig:
	"""Configuration for APG project generation"""
	
	# Basic Project Info
	name: str
	description: str
	author: str = "APG Developer"
	version: str = "1.0.0"
	license: str = "MIT"
	
	# Template Configuration
	template_type: TemplateType = TemplateType.BASIC_AGENT
	target_framework: str = "flask-appbuilder"
	database_type: str = "sqlite"
	
	# Feature Flags
	enable_authentication: bool = True
	enable_api: bool = True
	enable_database: bool = True
	enable_testing: bool = True
	enable_docker: bool = False
	enable_ai_features: bool = False
	enable_real_time: bool = False
	
	# Advanced Options
	python_version: str = "3.12"
	use_async: bool = True
	include_examples: bool = True
	generate_docs: bool = True
	
	# Dependencies
	custom_dependencies: List[str] = field(default_factory=list)
	
	# Output Configuration
	output_directory: Optional[Path] = None
	overwrite_existing: bool = False
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			'name': self.name,
			'description': self.description,
			'author': self.author,
			'version': self.version,
			'license': self.license,
			'template_type': self.template_type.value,
			'target_framework': self.target_framework,
			'database_type': self.database_type,
			'enable_authentication': self.enable_authentication,
			'enable_api': self.enable_api,
			'enable_database': self.enable_database,
			'enable_testing': self.enable_testing,
			'enable_docker': self.enable_docker,
			'enable_ai_features': self.enable_ai_features,
			'enable_real_time': self.enable_real_time,
			'python_version': self.python_version,
			'use_async': self.use_async,
			'include_examples': self.include_examples,
			'generate_docs': self.generate_docs,
			'custom_dependencies': self.custom_dependencies,
			'output_directory': str(self.output_directory) if self.output_directory else None,
			'overwrite_existing': self.overwrite_existing
		}
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
		"""Create from dictionary"""
		template_type = TemplateType(data.get('template_type', 'basic_agent'))
		output_dir = Path(data['output_directory']) if data.get('output_directory') else None
		
		return cls(
			name=data['name'],
			description=data['description'],
			author=data.get('author', 'APG Developer'),
			version=data.get('version', '1.0.0'),
			license=data.get('license', 'MIT'),
			template_type=template_type,
			target_framework=data.get('target_framework', 'flask-appbuilder'),
			database_type=data.get('database_type', 'sqlite'),
			enable_authentication=data.get('enable_authentication', True),
			enable_api=data.get('enable_api', True),
			enable_database=data.get('enable_database', True),
			enable_testing=data.get('enable_testing', True),
			enable_docker=data.get('enable_docker', False),
			enable_ai_features=data.get('enable_ai_features', False),
			enable_real_time=data.get('enable_real_time', False),
			python_version=data.get('python_version', '3.12'),
			use_async=data.get('use_async', True),
			include_examples=data.get('include_examples', True),
			generate_docs=data.get('generate_docs', True),
			custom_dependencies=data.get('custom_dependencies', []),
			output_directory=output_dir,
			overwrite_existing=data.get('overwrite_existing', False)
		)


# Template metadata for each template type
TEMPLATE_METADATA = {
	TemplateType.BASIC_AGENT: {
		'name': 'Basic Agent',
		'description': 'Simple agent with basic functionality and Flask-AppBuilder interface',
		'complexity': 'Beginner',
		'features': ['Agent Runtime', 'Basic Methods', 'Web Dashboard'],
		'use_cases': ['Learning APG', 'Simple Automation', 'Proof of Concept']
	},
	
	TemplateType.SIMPLE_WORKFLOW: {
		'name': 'Simple Workflow',
		'description': 'Basic workflow automation with step-by-step processing',
		'complexity': 'Beginner',
		'features': ['Workflow Engine', 'Step Management', 'Progress Tracking'],
		'use_cases': ['Process Automation', 'Task Orchestration', 'Business Logic']
	},
	
	TemplateType.DATABASE_CRUD: {
		'name': 'Database CRUD',
		'description': 'Complete database management with CRUD operations',
		'complexity': 'Intermediate',
		'features': ['Database Models', 'CRUD Operations', 'Admin Interface'],
		'use_cases': ['Data Management', 'Content Management', 'Record Keeping']
	},
	
	TemplateType.TASK_MANAGEMENT: {
		'name': 'Task Management System',
		'description': 'Full-featured task management with assignments and tracking',
		'complexity': 'Intermediate', 
		'features': ['Task Agents', 'User Management', 'Progress Tracking', 'Notifications'],
		'use_cases': ['Project Management', 'Team Coordination', 'Task Tracking']
	},
	
	TemplateType.E_COMMERCE: {
		'name': 'E-Commerce Platform',
		'description': 'Complete e-commerce solution with products, orders, and payments',
		'complexity': 'Advanced',
		'features': ['Product Catalog', 'Shopping Cart', 'Order Management', 'Payment Integration'],
		'use_cases': ['Online Store', 'Marketplace', 'B2B Commerce']
	},
	
	TemplateType.DIGITAL_TWIN: {
		'name': 'Digital Twin System',
		'description': 'IoT-enabled digital twin with real-time monitoring',
		'complexity': 'Advanced',
		'features': ['Device Modeling', 'Real-time Data', 'Simulation', 'Analytics'],
		'use_cases': ['IoT Monitoring', 'Predictive Maintenance', 'Smart Manufacturing']
	},
	
	TemplateType.AI_ASSISTANT: {
		'name': 'AI Assistant',
		'description': 'Intelligent assistant with natural language processing',
		'complexity': 'Advanced',
		'features': ['NLP Processing', 'Knowledge Base', 'Chat Interface', 'AI Integration'],
		'use_cases': ['Customer Support', 'Personal Assistant', 'Knowledge Management']
	},
	
	TemplateType.MICROSERVICES: {
		'name': 'Microservices Architecture',
		'description': 'Distributed microservices with API gateway and service discovery',
		'complexity': 'Expert',
		'features': ['Service Mesh', 'API Gateway', 'Load Balancing', 'Health Monitoring'],
		'use_cases': ['Enterprise Applications', 'Scalable Systems', 'Cloud-Native Apps']
	},
	
	TemplateType.ENTERPRISE_DASHBOARD: {
		'name': 'Enterprise Dashboard',
		'description': 'Executive dashboard with analytics and reporting',
		'complexity': 'Advanced',
		'features': ['Real-time Analytics', 'Custom Widgets', 'Report Generation', 'KPI Tracking'],
		'use_cases': ['Business Intelligence', 'Executive Reporting', 'Data Visualization']
	},
	
	TemplateType.DATA_PIPELINE: {
		'name': 'Data Pipeline',
		'description': 'ETL pipeline for data processing and analytics',
		'complexity': 'Advanced',
		'features': ['Data Ingestion', 'Transformation Engine', 'Analytics', 'Scheduling'],
		'use_cases': ['Data Processing', 'Analytics Workflows', 'Business Intelligence']
	},
	
	TemplateType.IOT_PLATFORM: {
		'name': 'IoT Platform',
		'description': 'Complete IoT platform with device management and analytics',
		'complexity': 'Expert',
		'features': ['Device Management', 'Data Collection', 'Real-time Analytics', 'Alerts'],
		'use_cases': ['Smart Cities', 'Industrial IoT', 'Consumer IoT']
	},
	
	TemplateType.FINTECH_PLATFORM: {
		'name': 'FinTech Platform',
		'description': 'Financial services platform with transactions and compliance',
		'complexity': 'Expert',
		'features': ['Transaction Processing', 'Compliance Tools', 'Risk Management', 'Reporting'],
		'use_cases': ['Digital Banking', 'Payment Processing', 'Financial Services']
	},
	
	TemplateType.HEALTHCARE_SYSTEM: {
		'name': 'Healthcare Management',
		'description': 'Healthcare management system with patient records and scheduling',
		'complexity': 'Expert',
		'features': ['Patient Management', 'Appointment Scheduling', 'Medical Records', 'HIPAA Compliance'],
		'use_cases': ['Hospital Management', 'Clinic Operations', 'Telemedicine']
	},
	
	TemplateType.LOGISTICS_TRACKER: {
		'name': 'Logistics Tracker',
		'description': 'Supply chain and logistics management with real-time tracking',
		'complexity': 'Advanced',
		'features': ['Shipment Tracking', 'Route Optimization', 'Inventory Management', 'Analytics'],
		'use_cases': ['Supply Chain', 'Delivery Services', 'Warehouse Management']
	},
	
	TemplateType.SOCIAL_NETWORK: {
		'name': 'Social Network',
		'description': 'Social networking platform with posts, connections, and messaging',
		'complexity': 'Advanced',
		'features': ['User Profiles', 'Social Feed', 'Messaging', 'Content Moderation'],
		'use_cases': ['Social Media', 'Community Platform', 'Professional Networks']
	}
}


def get_template_info(template_type: TemplateType) -> Dict[str, Any]:
	"""Get detailed information about a template type"""
	return TEMPLATE_METADATA.get(template_type, {})


def list_available_templates() -> List[Dict[str, Any]]:
	"""List all available templates with metadata"""
	templates = []
	for template_type in TemplateType:
		info = get_template_info(template_type)
		templates.append({
			'type': template_type.value,
			'template_type': template_type,
			**info
		})
	return templates


def get_recommended_templates(complexity: str = None, use_case: str = None) -> List[TemplateType]:
	"""Get recommended templates based on criteria"""
	recommendations = []
	
	for template_type in TemplateType:
		info = get_template_info(template_type)
		
		# Filter by complexity
		if complexity and info.get('complexity', '').lower() != complexity.lower():
			continue
		
		# Filter by use case
		if use_case:
			use_cases = [uc.lower() for uc in info.get('use_cases', [])]
			if not any(use_case.lower() in uc for uc in use_cases):
				continue
		
		recommendations.append(template_type)
	
	return recommendations