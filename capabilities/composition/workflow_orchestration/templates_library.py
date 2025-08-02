#!/usr/bin/env python3
"""
APG Workflow Orchestration Templates Library

Extensive collection of workflow templates for various industries and use cases.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

# APG Framework imports
from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase

# Local imports
from .models import WorkflowDefinition, TaskDefinition


logger = logging.getLogger(__name__)


class TemplateCategory(str, Enum):
	"""Template categories."""
	
	# Business Process Templates
	BUSINESS_PROCESS = "business_process"
	FINANCE_ACCOUNTING = "finance_accounting"
	HUMAN_RESOURCES = "human_resources"
	CUSTOMER_SERVICE = "customer_service"
	SALES_MARKETING = "sales_marketing"
	SUPPLY_CHAIN = "supply_chain"
	COMPLIANCE_AUDIT = "compliance_audit"
	
	# Technical Templates
	DATA_PROCESSING = "data_processing"
	ETL_PIPELINE = "etl_pipeline"
	CI_CD_PIPELINE = "ci_cd_pipeline"
	MONITORING_ALERTING = "monitoring_alerting"
	BACKUP_RECOVERY = "backup_recovery"
	SECURITY_INCIDENT = "security_incident"
	
	# Industry-Specific Templates
	HEALTHCARE = "healthcare"
	FINANCIAL_SERVICES = "financial_services"
	MANUFACTURING = "manufacturing"
	RETAIL_ECOMMERCE = "retail_ecommerce"
	EDUCATION = "education"
	GOVERNMENT = "government"
	
	# Integration Templates
	API_INTEGRATION = "api_integration"
	DATABASE_SYNC = "database_sync"
	FILE_PROCESSING = "file_processing"
	NOTIFICATION_SYSTEM = "notification_system"
	REPORTING_ANALYTICS = "reporting_analytics"
	
	# Specialized Templates
	MACHINE_LEARNING = "machine_learning"
	IOT_AUTOMATION = "iot_automation"
	BLOCKCHAIN = "blockchain"
	MICROSERVICES = "microservices"


class TemplateTags(str, Enum):
	"""Template tags for filtering and search."""
	
	# Complexity levels
	BEGINNER = "beginner"
	INTERMEDIATE = "intermediate"
	ADVANCED = "advanced"
	EXPERT = "expert"
	
	# Process types
	APPROVAL = "approval"
	AUTOMATION = "automation"
	INTEGRATION = "integration"
	NOTIFICATION = "notification"
	PROCESSING = "processing"
	ANALYSIS = "analysis"
	MONITORING = "monitoring"
	
	# Technical characteristics
	REAL_TIME = "real_time"
	BATCH = "batch"
	SCHEDULED = "scheduled"
	EVENT_DRIVEN = "event_driven"
	PARALLEL = "parallel"
	SEQUENTIAL = "sequential"
	
	# Industry standards
	SOX_COMPLIANT = "sox_compliant"
	GDPR_COMPLIANT = "gdpr_compliant"
	HIPAA_COMPLIANT = "hipaa_compliant"
	ISO_27001 = "iso_27001"
	PCI_DSS = "pci_dss"


@dataclass
class WorkflowTemplate:
	"""Workflow template definition."""
	
	id: str
	name: str
	description: str
	category: TemplateCategory
	tags: List[TemplateTags]
	version: str
	author: str
	organization: str
	created_at: datetime
	updated_at: datetime
	workflow_definition: Dict[str, Any]
	configuration_schema: Dict[str, Any]
	documentation: str
	use_cases: List[str]
	prerequisites: List[str]
	estimated_duration: Optional[int] = None  # in seconds
	complexity_score: int = 1  # 1-10 scale
	popularity_score: int = 0
	usage_count: int = 0
	is_verified: bool = False
	is_featured: bool = False
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.metadata is None:
			self.metadata = {}


class TemplatesLibraryService(APGBaseService):
	"""Service for managing workflow templates."""
	
	def __init__(self):
		super().__init__()
		self.database = APGDatabase()
		self.templates: Dict[str, WorkflowTemplate] = {}
		self.category_index: Dict[TemplateCategory, List[str]] = {}
		self.tag_index: Dict[TemplateTags, List[str]] = {}
		
		# Initialize with built-in templates
		self._initialize_templates()
	
	async def start(self):
		"""Start the templates library service."""
		await super().start()
		await self._load_templates_from_database()
		logger.info(f"Templates library service started with {len(self.templates)} templates")
	
	def _initialize_templates(self):
		"""Initialize with built-in templates."""
		# Business Process Templates
		self._add_employee_onboarding_template()
		self._add_expense_approval_template()
		self._add_invoice_processing_template()
		self._add_customer_support_ticket_template()
		self._add_purchase_order_template()
		
		# Data Processing Templates
		self._add_data_ingestion_template()
		self._add_etl_pipeline_template()
		self._add_report_generation_template()
		self._add_data_quality_check_template()
		self._add_backup_workflow_template()
		
		# Integration Templates
		self._add_api_sync_template()
		self._add_database_migration_template()
		self._add_file_processing_template()
		self._add_notification_workflow_template()
		
		# Industry-Specific Templates
		self._add_loan_approval_template()
		self._add_patient_admission_template()
		self._add_order_fulfillment_template()
		self._add_incident_response_template()
		
		# Technical Templates
		self._add_ci_cd_pipeline_template()
		self._add_monitoring_template()
		self._add_ml_training_template()
		self._add_security_scan_template()
		
		# Specialized Templates
		self._add_iot_data_processing_template()
		self._add_blockchain_transaction_template()
		self._add_microservice_deployment_template()
		
		logger.info(f"Initialized {len(self.templates)} built-in templates")
	
	# Employee Onboarding Template
	def _add_employee_onboarding_template(self):
		"""Employee onboarding workflow template."""
		template = WorkflowTemplate(
			id="template_employee_onboarding_001",
			name="Employee Onboarding Process",
			description="Complete employee onboarding workflow including account setup, equipment provisioning, training assignments, and orientation scheduling.",
			category=TemplateCategory.HUMAN_RESOURCES,
			tags=[TemplateTags.BEGINNER, TemplateTags.APPROVAL, TemplateTags.AUTOMATION, TemplateTags.SEQUENTIAL],
			version="1.0.0",
			author="APG Team",
			organization="Datacraft",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			workflow_definition={
				"name": "Employee Onboarding Process",
				"description": "Streamlined employee onboarding workflow",
				"tasks": [
					{
						"id": "hr_approval",
						"name": "HR Approval",
						"type": "approval",
						"description": "HR manager approves new employee onboarding",
						"config": {
							"approvers": ["hr_manager"],
							"timeout_hours": 24,
							"escalation_enabled": True
						},
						"next_tasks": ["create_accounts"]
					},
					{
						"id": "create_accounts",
						"name": "Create System Accounts",
						"type": "integration",
						"description": "Create employee accounts in all systems",
						"config": {
							"integration_id": "active_directory",
							"action": "create_user",
							"parallel_execution": True
						},
						"next_tasks": ["provision_equipment", "assign_training"]
					},
					{
						"id": "provision_equipment",
						"name": "Equipment Provisioning",
						"type": "task",
						"description": "Assign and prepare equipment for new employee",
						"config": {
							"integration_id": "asset_management",
							"equipment_types": ["laptop", "phone", "access_card"]
						},
						"next_tasks": ["schedule_orientation"]
					},
					{
						"id": "assign_training",
						"name": "Assign Training Modules",
						"type": "integration",
						"description": "Assign mandatory training courses",
						"config": {
							"integration_id": "learning_management",
							"training_modules": ["security_awareness", "company_policies", "role_specific"]
						},
						"next_tasks": ["schedule_orientation"]
					},
					{
						"id": "schedule_orientation",
						"name": "Schedule Orientation",
						"type": "integration",
						"description": "Schedule orientation sessions",
						"config": {
							"integration_id": "calendar_system",
							"session_types": ["company_overview", "team_introduction", "it_setup"],
							"duration_hours": 8
						},
						"next_tasks": ["send_welcome_package"]
					},
					{
						"id": "send_welcome_package",
						"name": "Send Welcome Package",
						"type": "notification",
						"description": "Send welcome email and documentation",
						"config": {
							"notification_type": "email",
							"template": "welcome_package",
							"attachments": ["employee_handbook", "benefits_guide", "org_chart"]
						},
						"next_tasks": []
					}
				],
				"error_handling": {
					"retry_attempts": 3,
					"escalation_enabled": True,
					"fallback_actions": ["manual_intervention"]
				}
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"employee_info": {
						"type": "object",
						"properties": {
							"first_name": {"type": "string"},
							"last_name": {"type": "string"},
							"email": {"type": "string", "format": "email"},
							"department": {"type": "string"},
							"position": {"type": "string"},
							"start_date": {"type": "string", "format": "date"},
							"manager_email": {"type": "string", "format": "email"}
						},
						"required": ["first_name", "last_name", "email", "department", "position", "start_date"]
					},
					"equipment_requirements": {
						"type": "array",
						"items": {"type": "string"},
						"default": ["laptop", "phone", "access_card"]
					},
					"training_modules": {
						"type": "array",
						"items": {"type": "string"},
						"default": ["security_awareness", "company_policies"]
					}
				},
				"required": ["employee_info"]
			},
			documentation="""
# Employee Onboarding Process Template

This template automates the complete employee onboarding process from initial HR approval to welcome package delivery.

## Features
- Automated account creation across multiple systems
- Equipment provisioning and tracking
- Training assignment and scheduling
- Orientation scheduling with calendar integration
- Welcome package delivery with documentation

## Prerequisites
- Active Directory integration configured
- Asset management system integration
- Learning management system integration
- Calendar system integration
- Email notification system configured

## Configuration
Configure employee information, equipment requirements, and training modules through the configuration schema.

## Estimated Duration
2-3 business days for complete onboarding process.
			""",
			use_cases=[
				"New hire onboarding for corporate environments",
				"Temporary employee setup",
				"Contractor onboarding with limited access",
				"Remote employee onboarding",
				"Bulk employee onboarding for new offices"
			],
			prerequisites=[
				"Active Directory or equivalent identity management system",
				"Asset management system with API access",
				"Learning management system integration",
				"Calendar system with scheduling API",
				"Email system for notifications"
			],
			estimated_duration=259200,  # 3 days
			complexity_score=3,
			is_verified=True,
			is_featured=True
		)
		
		self._register_template(template)
	
	# Expense Approval Template
	def _add_expense_approval_template(self):
		"""Expense approval workflow template."""
		template = WorkflowTemplate(
			id="template_expense_approval_001",
			name="Expense Approval Workflow",
			description="Multi-level expense approval process with automatic routing, policy validation, and reimbursement processing.",
			category=TemplateCategory.FINANCE_ACCOUNTING,
			tags=[TemplateTags.INTERMEDIATE, TemplateTags.APPROVAL, TemplateTags.AUTOMATION, TemplateTags.SOX_COMPLIANT],
			version="1.2.0",
			author="APG Team",
			organization="Datacraft",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			workflow_definition={
				"name": "Expense Approval Workflow",
				"description": "Automated expense approval with policy validation",
				"tasks": [
					{
						"id": "validate_expense",
						"name": "Validate Expense Policy",
						"type": "validation",
						"description": "Validate expense against company policy",
						"config": {
							"validation_rules": [
								{"field": "amount", "rule": "max_value", "value": 5000},
								{"field": "category", "rule": "allowed_values", "values": ["travel", "meals", "supplies", "software"]},
								{"field": "receipt", "rule": "required", "min_amount": 25}
							]
						},
						"next_tasks": ["manager_approval"]
					},
					{
						"id": "manager_approval",
						"name": "Manager Approval",
						"type": "approval",
						"description": "Direct manager approval for expenses",
						"config": {
							"approver_type": "manager",
							"auto_approve_threshold": 100,
							"timeout_hours": 48,
							"escalation_enabled": True
						},
						"next_tasks": ["finance_approval"]
					},
					{
						"id": "finance_approval",
						"name": "Finance Team Approval",
						"type": "approval",
						"description": "Finance team approval for expenses over threshold",
						"config": {
							"approver_group": "finance_team",
							"required_threshold": 1000,
							"timeout_hours": 72,
							"parallel_approval": True
						},
						"next_tasks": ["process_reimbursement"]
					},
					{
						"id": "process_reimbursement",
						"name": "Process Reimbursement",
						"type": "integration",
						"description": "Process payment through accounting system",
						"config": {
							"integration_id": "accounting_system",
							"action": "create_payment",
							"payment_method": "bank_transfer"
						},
						"next_tasks": ["notify_employee"]
					},
					{
						"id": "notify_employee",
						"name": "Notify Employee",
						"type": "notification",
						"description": "Send reimbursement confirmation to employee",
						"config": {
							"notification_type": "email",
							"template": "reimbursement_processed",
							"include_payment_details": True
						},
						"next_tasks": []
					}
				],
				"conditions": [
					{
						"task_id": "finance_approval",
						"condition": "expense_amount > 1000"
					}
				]
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"expense_details": {
						"type": "object",
						"properties": {
							"amount": {"type": "number", "minimum": 0},
							"currency": {"type": "string", "default": "USD"},
							"category": {"type": "string", "enum": ["travel", "meals", "supplies", "software", "other"]},
							"description": {"type": "string"},
							"date": {"type": "string", "format": "date"},
							"receipt_url": {"type": "string", "format": "uri"}
						},
						"required": ["amount", "category", "description", "date"]
					},
					"employee_info": {
						"type": "object",
						"properties": {
							"employee_id": {"type": "string"},
							"name": {"type": "string"},
							"email": {"type": "string", "format": "email"},
							"department": {"type": "string"},
							"manager_email": {"type": "string", "format": "email"}
						},
						"required": ["employee_id", "name", "email", "manager_email"]
					},
					"approval_thresholds": {
						"type": "object",
						"properties": {
							"auto_approve_limit": {"type": "number", "default": 100},
							"finance_approval_threshold": {"type": "number", "default": 1000}
						}
					}
				},
				"required": ["expense_details", "employee_info"]
			},
			documentation="""
# Expense Approval Workflow Template

Automated expense approval process with policy validation, multi-level approvals, and reimbursement processing.

## Features
- Automatic policy validation against company rules
- Dynamic approval routing based on amount and category
- Integration with accounting systems for payment processing
- Automated notifications and status updates
- Audit trail for compliance requirements

## Approval Levels
1. **Automatic Approval**: Expenses under $100 (configurable)
2. **Manager Approval**: All expenses require manager approval
3. **Finance Approval**: Expenses over $1000 require finance team approval

## Prerequisites
- Employee directory with manager relationships
- Accounting system integration for payment processing
- Document storage for receipt management
- Email notification system

## Compliance
- SOX compliant with audit trail
- Configurable approval thresholds
- Receipt requirements based on amount
			""",
			use_cases=[
				"Employee expense reimbursement",
				"Travel expense processing",
				"Vendor payment approvals",
				"Petty cash reimbursements",
				"Conference and training expenses"
			],
			prerequisites=[
				"Employee directory with manager hierarchy",
				"Accounting system with payment API",
				"Document management system for receipts",
				"Email notification system",
				"Expense policy configuration"
			],
			estimated_duration=172800,  # 2 days
			complexity_score=4,
			is_verified=True,
			is_featured=True
		)
		
		self._register_template(template)
	
	# Data Processing Templates
	def _add_data_ingestion_template(self):
		"""Data ingestion workflow template."""
		template = WorkflowTemplate(
			id="template_data_ingestion_001",
			name="Data Ingestion Pipeline",
			description="Comprehensive data ingestion workflow with validation, transformation, and loading into data warehouse.",
			category=TemplateCategory.DATA_PROCESSING,
			tags=[TemplateTags.INTERMEDIATE, TemplateTags.PROCESSING, TemplateTags.BATCH, TemplateTags.SCHEDULED],
			version="2.1.0",
			author="APG Team",
			organization="Datacraft",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			workflow_definition={
				"name": "Data Ingestion Pipeline",
				"description": "ETL pipeline for data ingestion and processing",
				"tasks": [
					{
						"id": "extract_data",
						"name": "Extract Data",
						"type": "integration",
						"description": "Extract data from source systems",
						"config": {
							"integration_id": "source_database",
							"query_type": "incremental",
							"batch_size": 10000,
							"parallel_connections": 4
						},
						"next_tasks": ["validate_data"]
					},
					{
						"id": "validate_data",
						"name": "Data Validation",
						"type": "validation",
						"description": "Validate data quality and completeness",
						"config": {
							"validation_rules": [
								{"type": "not_null", "columns": ["id", "created_at"]},
								{"type": "unique", "columns": ["id"]},
								{"type": "data_type", "validations": {"amount": "decimal", "date": "datetime"}},
								{"type": "range", "column": "amount", "min": 0, "max": 1000000}
							],
							"error_threshold": 0.05
						},
						"next_tasks": ["transform_data"]
					},
					{
						"id": "transform_data",
						"name": "Data Transformation",
						"type": "processing",
						"description": "Apply business rules and transformations",
						"config": {
							"transformations": [
								{"type": "standardize_format", "columns": ["phone", "email"]},
								{"type": "calculate_derived", "formula": "amount * tax_rate", "result_column": "tax_amount"},
								{"type": "categorize", "column": "amount", "ranges": [{"min": 0, "max": 100, "category": "small"}]},
								{"type": "anonymize", "columns": ["ssn", "credit_card"], "method": "hash"}
							]
						},
						"next_tasks": ["load_data"]
					},
					{
						"id": "load_data",
						"name": "Load Data",
						"type": "integration",
						"description": "Load transformed data into data warehouse",
						"config": {
							"integration_id": "data_warehouse",
							"load_strategy": "upsert",
							"batch_size": 5000,
							"create_backup": True
						},
						"next_tasks": ["update_metadata", "generate_report"]
					},
					{
						"id": "update_metadata",
						"name": "Update Metadata",
						"type": "integration",
						"description": "Update data catalog and lineage information",
						"config": {
							"integration_id": "data_catalog",
							"update_lineage": True,
							"update_statistics": True
						},
						"next_tasks": ["send_notification"]
					},
					{
						"id": "generate_report",
						"name": "Generate Processing Report",
						"type": "reporting",
						"description": "Generate data processing summary report",
						"config": {
							"report_type": "processing_summary",
							"include_metrics": True,
							"include_errors": True
						},
						"next_tasks": ["send_notification"]
					},
					{
						"id": "send_notification",
						"name": "Send Completion Notification",
						"type": "notification",
						"description": "Notify stakeholders of processing completion",
						"config": {
							"notification_type": "email",
							"recipients": ["data_team", "business_users"],
							"include_report": True
						},
						"next_tasks": []
					}
				],
				"error_handling": {
					"retry_attempts": 3,
					"retry_delay": 300,
					"escalation_enabled": True,
					"data_quality_alerts": True
				},
				"scheduling": {
					"type": "cron",
					"schedule": "0 2 * * *",
					"timezone": "UTC"
				}
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"source_config": {
						"type": "object",
						"properties": {
							"connection_string": {"type": "string"},
							"table_name": {"type": "string"},
							"incremental_column": {"type": "string", "default": "updated_at"},
							"batch_size": {"type": "integer", "default": 10000}
						},
						"required": ["connection_string", "table_name"]
					},
					"transformation_config": {
						"type": "object",
						"properties": {
							"business_rules": {"type": "array", "items": {"type": "object"}},
							"data_quality_threshold": {"type": "number", "default": 0.95}
						}
					},
					"destination_config": {
						"type": "object",
						"properties": {
							"warehouse_connection": {"type": "string"},
							"target_schema": {"type": "string"},
							"target_table": {"type": "string"},
							"load_strategy": {"type": "string", "enum": ["insert", "upsert", "replace"], "default": "upsert"}
						},
						"required": ["warehouse_connection", "target_schema", "target_table"]
					},
					"notification_config": {
						"type": "object",
						"properties": {
							"recipients": {"type": "array", "items": {"type": "string"}},
							"notify_on_success": {"type": "boolean", "default": true},
							"notify_on_failure": {"type": "boolean", "default": true}
						}
					}
				},
				"required": ["source_config", "destination_config"]
			},
			documentation="""
# Data Ingestion Pipeline Template

Enterprise-grade data ingestion workflow with comprehensive validation, transformation, and loading capabilities.

## Features
- Incremental data extraction with configurable batch sizes
- Comprehensive data quality validation
- Flexible transformation engine with business rules
- Reliable loading with error handling and rollback
- Metadata management and data lineage tracking
- Automated reporting and notifications

## Data Quality Checks
- Null value validation
- Data type verification
- Uniqueness constraints
- Range and format validation
- Custom business rule validation

## Transformation Capabilities
- Data standardization and cleansing
- Derived column calculations
- Data categorization and enrichment
- Privacy and anonymization features
- Custom transformation functions

## Prerequisites
- Source system with API or database access
- Data warehouse or lake destination
- Data catalog system for metadata management
- Notification system for alerts and reports

## Scheduling
Supports cron-based scheduling for automated execution.
			""",
			use_cases=[
				"Daily sales data ingestion",
				"Customer data synchronization",
				"Financial data processing",
				"IoT sensor data ingestion",
				"Log file processing and analysis",
				"Third-party API data integration"
			],
			prerequisites=[
				"Source database or API access",
				"Data warehouse with write permissions",
				"Data catalog system",
				"Notification system",
				"Data quality monitoring tools"
			],
			estimated_duration=3600,  # 1 hour
			complexity_score=6,
			is_verified=True,
			is_featured=True
		)
		
		self._register_template(template)
	
	# Continue with more templates...
	def _add_ci_cd_pipeline_template(self):
		"""CI/CD pipeline workflow template."""
		template = WorkflowTemplate(
			id="template_cicd_pipeline_001",
			name="CI/CD Pipeline",
			description="Complete continuous integration and deployment pipeline with testing, security scanning, and multi-environment deployment.",
			category=TemplateCategory.CI_CD_PIPELINE,
			tags=[TemplateTags.ADVANCED, TemplateTags.AUTOMATION, TemplateTags.EVENT_DRIVEN, TemplateTags.PARALLEL],
			version="3.0.0",
			author="APG Team",
			organization="Datacraft",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			workflow_definition={
				"name": "CI/CD Pipeline",
				"description": "Automated build, test, and deployment pipeline",
				"tasks": [
					{
						"id": "checkout_code",
						"name": "Checkout Source Code",
						"type": "integration",
						"description": "Checkout code from version control",
						"config": {
							"integration_id": "git_repository",
							"action": "checkout",
							"branch": "main",
							"clean_workspace": True
						},
						"next_tasks": ["install_dependencies", "code_analysis"]
					},
					{
						"id": "install_dependencies",
						"name": "Install Dependencies",
						"type": "build",
						"description": "Install project dependencies",
						"config": {
							"package_manager": "npm",
							"cache_enabled": True,
							"production_only": False
						},
						"next_tasks": ["run_tests"]
					},
					{
						"id": "code_analysis",
						"name": "Static Code Analysis",
						"type": "analysis",
						"description": "Run static code analysis and linting",
						"config": {
							"tools": ["eslint", "sonarqube", "prettier"],
							"fail_on_error": True,
							"quality_gate": True
						},
						"next_tasks": ["run_tests"]
					},
					{
						"id": "run_tests",
						"name": "Run Test Suite",
						"type": "testing",
						"description": "Execute unit, integration, and e2e tests",
						"config": {
							"test_types": ["unit", "integration", "e2e"],
							"parallel_execution": True,
							"coverage_threshold": 80,
							"generate_report": True
						},
						"next_tasks": ["security_scan"]
					},
					{
						"id": "security_scan",
						"name": "Security Vulnerability Scan",
						"type": "security",
						"description": "Scan for security vulnerabilities",
						"config": {
							"tools": ["snyk", "npm_audit", "docker_scan"],
							"severity_threshold": "medium",
							"fail_on_high": True
						},
						"next_tasks": ["build_application"]
					},
					{
						"id": "build_application",
						"name": "Build Application",
						"type": "build",
						"description": "Build application artifacts",
						"config": {
							"build_tool": "webpack",
							"optimization": True,
							"generate_sourcemaps": False,
							"output_path": "dist/"
						},
						"next_tasks": ["build_docker_image"]
					},
					{
						"id": "build_docker_image",
						"name": "Build Docker Image",
						"type": "containerization",
						"description": "Build and tag Docker image",
						"config": {
							"dockerfile_path": "Dockerfile",
							"image_name": "${project_name}",
							"tag_strategy": "git_commit",
							"push_to_registry": True
						},
						"next_tasks": ["deploy_staging"]
					},
					{
						"id": "deploy_staging",
						"name": "Deploy to Staging",
						"type": "deployment",
						"description": "Deploy to staging environment",
						"config": {
							"environment": "staging",
							"deployment_strategy": "rolling",
							"health_check_enabled": True,
							"rollback_on_failure": True
						},
						"next_tasks": ["staging_tests"]
					},
					{
						"id": "staging_tests",
						"name": "Staging Environment Tests",
						"type": "testing",
						"description": "Run tests against staging environment",
						"config": {
							"test_types": ["smoke", "integration", "performance"],
							"environment_url": "${staging_url}",
							"timeout_minutes": 30
						},
						"next_tasks": ["approve_production"]
					},
					{
						"id": "approve_production",
						"name": "Production Deployment Approval",
						"type": "approval",
						"description": "Manual approval for production deployment",
						"config": {
							"approvers": ["tech_lead", "product_owner"],
							"timeout_hours": 168,
							"auto_approve_hotfix": True
						},
						"next_tasks": ["deploy_production"]
					},
					{
						"id": "deploy_production",
						"name": "Deploy to Production",
						"type": "deployment",
						"description": "Deploy to production environment",
						"config": {
							"environment": "production",
							"deployment_strategy": "blue_green",
							"health_check_enabled": True,
							"smoke_test_enabled": True,
							"monitoring_enabled": True
						},
						"next_tasks": ["notify_team"]
					},
					{
						"id": "notify_team",
						"name": "Notify Team",
						"type": "notification",
						"description": "Send deployment notification",
						"config": {
							"notification_type": "slack",
							"channels": ["#deployments", "#team-updates"],
							"include_deployment_info": True,
							"mention_on_failure": True
						},
						"next_tasks": []
					}
				],
				"parallel_tasks": [
					["install_dependencies", "code_analysis"]
				],
				"error_handling": {
					"retry_attempts": 2,
					"rollback_enabled": True,
					"notify_on_failure": True
				}
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"project_config": {
						"type": "object",
						"properties": {
							"project_name": {"type": "string"},
							"repository_url": {"type": "string"},
							"branch": {"type": "string", "default": "main"},
							"build_tool": {"type": "string", "enum": ["npm", "yarn", "maven", "gradle"], "default": "npm"}
						},
						"required": ["project_name", "repository_url"]
					},
					"environment_config": {
						"type": "object",
						"properties": {
							"staging_url": {"type": "string"},
							"production_url": {"type": "string"},
							"docker_registry": {"type": "string"}
						},
						"required": ["staging_url", "production_url"]
					},
					"quality_gates": {
						"type": "object",
						"properties": {
							"test_coverage_threshold": {"type": "number", "default": 80},
							"security_severity_threshold": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
							"code_quality_gate": {"type": "boolean", "default": true}
						}
					},
					"notification_config": {
						"type": "object",
						"properties": {
							"slack_webhook": {"type": "string"},
							"email_recipients": {"type": "array", "items": {"type": "string"}},
							"notify_on_success": {"type": "boolean", "default": true},
							"notify_on_failure": {"type": "boolean", "default": true}
						}
					}
				},
				"required": ["project_config"]
			},
			documentation="""
# CI/CD Pipeline Template

Production-ready continuous integration and deployment pipeline with comprehensive testing, security scanning, and multi-environment deployment.

## Pipeline Stages

### Continuous Integration
1. **Source Code Checkout**: Clone repository and prepare workspace
2. **Dependency Installation**: Install project dependencies with caching
3. **Static Code Analysis**: ESLint, SonarQube, and code formatting checks
4. **Test Execution**: Unit, integration, and end-to-end tests
5. **Security Scanning**: Vulnerability scanning with Snyk and npm audit
6. **Build Process**: Application build with optimization

### Continuous Deployment
1. **Container Building**: Docker image creation and registry push
2. **Staging Deployment**: Automated deployment to staging environment
3. **Staging Tests**: Smoke, integration, and performance tests
4. **Production Approval**: Manual approval gate for production
5. **Production Deployment**: Blue-green deployment with health checks
6. **Notifications**: Team notifications via Slack and email

## Quality Gates
- Minimum test coverage threshold (configurable)
- Security vulnerability thresholds
- Code quality metrics from static analysis
- Performance benchmarks in staging

## Deployment Strategies
- **Rolling Deployment**: For staging environments
- **Blue-Green Deployment**: For production environments
- **Automatic Rollback**: On health check failures

## Prerequisites
- Git repository with proper branching strategy
- Container registry (Docker Hub, ECR, etc.)
- Staging and production environments
- Monitoring and alerting systems
			""",
			use_cases=[
				"Web application deployment",
				"Microservices deployment",
				"Mobile application CI/CD",
				"API service deployment",
				"Infrastructure as code deployment",
				"Multi-environment application delivery"
			],
			prerequisites=[
				"Git repository with webhook support",
				"Docker registry access",
				"Kubernetes or container orchestration platform",
				"Testing framework and tools",
				"Security scanning tools",
				"Notification systems (Slack, email)"
			],
			estimated_duration=1800,  # 30 minutes
			complexity_score=8,
			is_verified=True,
			is_featured=True
		)
		
		self._register_template(template)
	
	# Add more specialized templates
	def _add_loan_approval_template(self):
		"""Loan approval workflow for financial services."""
		template = WorkflowTemplate(
			id="template_loan_approval_001",
			name="Loan Approval Process",
			description="Comprehensive loan approval workflow with credit checks, risk assessment, document verification, and multi-level approvals.",
			category=TemplateCategory.FINANCIAL_SERVICES,
			tags=[TemplateTags.ADVANCED, TemplateTags.APPROVAL, TemplateTags.AUTOMATION, TemplateTags.SOX_COMPLIANT],
			version="2.5.0",
			author="APG Team",
			organization="Datacraft",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			workflow_definition={
				"name": "Loan Approval Process",
				"description": "Automated loan processing with risk assessment",
				"tasks": [
					{
						"id": "validate_application",
						"name": "Validate Loan Application",
						"type": "validation",
						"description": "Validate completeness and accuracy of loan application",
						"config": {
							"required_documents": ["application_form", "income_proof", "identity_proof", "bank_statements"],
							"data_validation_rules": [
								{"field": "income", "rule": "minimum", "value": 25000},
								{"field": "age", "rule": "range", "min": 18, "max": 70},
								{"field": "employment_status", "rule": "allowed_values", "values": ["employed", "self_employed", "retired"]}
							]
						},
						"next_tasks": ["credit_check", "document_verification"]
					},
					{
						"id": "credit_check",
						"name": "Credit Bureau Check",
						"type": "integration",
						"description": "Retrieve credit score and history from credit bureaus",
						"config": {
							"integration_id": "credit_bureau_api",
							"bureaus": ["experian", "equifax", "transunion"],
							"detailed_report": True
						},
						"next_tasks": ["risk_assessment"]
					},
					{
						"id": "document_verification",
						"name": "Document Verification",
						"type": "verification",
						"description": "Verify authenticity of submitted documents",
						"config": {
							"verification_methods": ["ocr_analysis", "third_party_verification", "manual_review"],
							"required_confidence": 0.85
						},
						"next_tasks": ["risk_assessment"]
					},
					{
						"id": "risk_assessment",
						"name": "Risk Assessment",
						"type": "analysis",
						"description": "Calculate loan risk score and recommendation",
						"config": {
							"risk_model": "advanced_ml_model",
							"factors": ["credit_score", "income", "debt_ratio", "employment_history", "collateral"],
							"scoring_algorithm": "ensemble_model"
						},
						"next_tasks": ["underwriter_review"]
					},
					{
						"id": "underwriter_review",
						"name": "Underwriter Review",
						"type": "approval",
						"description": "Underwriter review and initial approval",
						"config": {
							"approver_role": "underwriter",
							"auto_approve_threshold": 750,
							"auto_reject_threshold": 500,
							"timeout_hours": 48,
							"escalation_enabled": True
						},
						"next_tasks": ["senior_approval"]
					},
					{
						"id": "senior_approval",
						"name": "Senior Underwriter Approval",
						"type": "approval",
						"description": "Senior approval for high-value or complex loans",
						"config": {
							"approver_role": "senior_underwriter",
							"required_conditions": [
								{"field": "loan_amount", "condition": ">", "value": 500000},
								{"field": "risk_score", "condition": "<", "value": 650}
							],
							"timeout_hours": 72
						},
						"next_tasks": ["generate_loan_terms"]
					},
					{
						"id": "generate_loan_terms",
						"name": "Generate Loan Terms",
						"type": "processing",
						"description": "Generate loan terms and conditions",
						"config": {
							"interest_rate_model": "risk_based_pricing",
							"term_options": [12, 24, 36, 48, 60],
							"fee_calculation": "standard_schedule"
						},
						"next_tasks": ["customer_acceptance"]
					},
					{
						"id": "customer_acceptance",
						"name": "Customer Acceptance",
						"type": "approval",
						"description": "Customer accepts loan terms and conditions",
						"config": {
							"approver_type": "customer",
							"timeout_hours": 168,
							"digital_signature_required": True
						},
						"next_tasks": ["fund_disbursement"]
					},
					{
						"id": "fund_disbursement",
						"name": "Fund Disbursement",
						"type": "integration",
						"description": "Disburse loan funds to customer account",
						"config": {
							"integration_id": "core_banking_system",
							"verification_required": True,
							"compliance_checks": ["aml", "kyc", "sanctions"]
						},
						"next_tasks": ["setup_repayment", "notify_customer"]
					},
					{
						"id": "setup_repayment",
						"name": "Setup Repayment Schedule",
						"type": "integration",
						"description": "Setup automatic repayment schedule",
						"config": {
							"integration_id": "loan_servicing_system",
							"repayment_method": "auto_debit",
							"schedule_type": "monthly"
						},
						"next_tasks": ["notify_customer"]
					},
					{
						"id": "notify_customer",
						"name": "Customer Notification",
						"type": "notification",
						"description": "Send loan approval and disbursement notification",
						"config": {
							"notification_types": ["email", "sms", "postal_mail"],
							"documents": ["loan_agreement", "repayment_schedule", "terms_conditions"],
							"delivery_confirmation": True
						},
						"next_tasks": []
					}
				],
				"conditions": [
					{
						"task_id": "senior_approval",
						"condition": "loan_amount > 500000 OR risk_score < 650"
					}
				],
				"sla": {
					"total_processing_time": 5,
					"unit": "business_days"
				}
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"loan_application": {
						"type": "object",
						"properties": {
							"applicant_info": {
								"type": "object",
								"properties": {
									"name": {"type": "string"},
									"ssn": {"type": "string"},
									"date_of_birth": {"type": "string", "format": "date"},
									"email": {"type": "string", "format": "email"},
									"phone": {"type": "string"},
									"address": {"type": "object"}
								},
								"required": ["name", "ssn", "date_of_birth", "email"]
							},
							"loan_details": {
								"type": "object",
								"properties": {
									"amount": {"type": "number", "minimum": 1000},
									"purpose": {"type": "string"},
									"term_months": {"type": "integer", "minimum": 12, "maximum": 360},
									"collateral": {"type": "object"}
								},
								"required": ["amount", "purpose", "term_months"]
							},
							"financial_info": {
								"type": "object",
								"properties": {
									"annual_income": {"type": "number"},
									"employment_status": {"type": "string"},
									"employer": {"type": "string"},
									"existing_debts": {"type": "number"}
								},
								"required": ["annual_income", "employment_status"]
							}
						},
						"required": ["applicant_info", "loan_details", "financial_info"]
					},
					"risk_settings": {
						"type": "object",
						"properties": {
							"auto_approve_score": {"type": "number", "default": 750},
							"auto_reject_score": {"type": "number", "default": 500},
							"max_debt_to_income_ratio": {"type": "number", "default": 0.43}
						}
					},
					"approval_settings": {
						"type": "object",
						"properties": {
							"underwriter_timeout_hours": {"type": "number", "default": 48},
							"senior_approval_threshold": {"type": "number", "default": 500000},
							"customer_acceptance_timeout_hours": {"type": "number", "default": 168}
						}
					}
				},
				"required": ["loan_application"]
			},
			documentation="""
# Loan Approval Process Template

Comprehensive loan approval workflow designed for financial institutions with automated risk assessment, regulatory compliance, and multi-level approvals.

## Process Overview

### Stage 1: Application Processing
- Application validation and completeness check
- Document verification using OCR and third-party services
- Credit bureau checks from multiple sources

### Stage 2: Risk Assessment
- Advanced ML-based risk scoring
- Income and employment verification
- Debt-to-income ratio calculation
- Collateral evaluation (if applicable)

### Stage 3: Approval Process
- Automated approval/rejection for clear cases
- Underwriter review for marginal cases
- Senior approval for high-value or high-risk loans
- Regulatory compliance checks

### Stage 4: Loan Origination
- Loan terms generation based on risk profile
- Customer acceptance and digital signature
- Fund disbursement with compliance checks
- Repayment schedule setup

## Key Features
- **Automated Decision Making**: AI-powered risk assessment and automated approvals
- **Regulatory Compliance**: Built-in AML, KYC, and sanctions screening
- **Multi-Bureau Credit Checks**: Comprehensive credit history analysis
- **Document Verification**: OCR and third-party verification services
- **Flexible Approval Workflows**: Configurable approval thresholds and routing
- **Audit Trail**: Complete audit trail for regulatory requirements

## Risk Management
- Credit score thresholds for automated decisions
- Debt-to-income ratio validation
- Employment and income verification
- Collateral valuation and assessment
- Fraud detection and prevention

## Compliance Features
- SOX compliance with audit trail
- Fair lending practices compliance
- Data privacy and security controls
- Regulatory reporting capabilities
			""",
			use_cases=[
				"Personal loan approval",
				"Mortgage loan processing",
				"Business loan evaluation",
				"Auto loan approval",
				"Credit line applications",
				"Refinancing applications"
			],
			prerequisites=[
				"Credit bureau API access",
				"Core banking system integration",
				"Document verification services",
				"Risk assessment models and algorithms",
				"Loan servicing system",
				"Digital signature platform",
				"Compliance and regulatory frameworks"
			],
			estimated_duration=432000,  # 5 business days
			complexity_score=9,
			is_verified=True,
			is_featured=True
		)
		
		self._register_template(template)
	
	# Add ML Training Template
	def _add_ml_training_template(self):
		"""Machine learning model training workflow."""
		template = WorkflowTemplate(
			id="template_ml_training_001",
			name="ML Model Training Pipeline",
			description="End-to-end machine learning model training workflow with data preparation, feature engineering, model training, validation, and deployment.",
			category=TemplateCategory.MACHINE_LEARNING,
			tags=[TemplateTags.ADVANCED, TemplateTags.PROCESSING, TemplateTags.BATCH, TemplateTags.SCHEDULED],
			version="1.8.0",
			author="APG Team",
			organization="Datacraft",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			workflow_definition={
				"name": "ML Model Training Pipeline",
				"description": "Automated ML model training with MLOps best practices",
				"tasks": [
					{
						"id": "data_extraction",
						"name": "Extract Training Data",
						"type": "integration",
						"description": "Extract data from feature store or data warehouse",
						"config": {
							"integration_id": "feature_store",
							"query": "SELECT * FROM training_dataset WHERE date >= CURRENT_DATE - INTERVAL '90 days'",
							"format": "parquet"
						},
						"next_tasks": ["data_validation"]
					},
					{
						"id": "data_validation",
						"name": "Data Quality Validation",
						"type": "validation",
						"description": "Validate data quality and schema compliance",
						"config": {
							"validation_rules": [
								{"type": "schema_validation", "schema_file": "training_schema.json"},
								{"type": "data_drift", "baseline_dataset": "production_baseline"},
								{"type": "missing_values", "threshold": 0.05},
								{"type": "outlier_detection", "method": "isolation_forest"}
							]
						},
						"next_tasks": ["feature_engineering"]
					},
					{
						"id": "feature_engineering",
						"name": "Feature Engineering",
						"type": "processing",
						"description": "Apply feature transformations and engineering",
						"config": {
							"transformations": [
								{"type": "scaling", "method": "standard_scaler", "columns": ["numeric_features"]},
								{"type": "encoding", "method": "one_hot", "columns": ["categorical_features"]},
								{"type": "feature_selection", "method": "recursive_feature_elimination", "n_features": 50},
								{"type": "dimensionality_reduction", "method": "pca", "n_components": 0.95}
							],
							"feature_store_update": True
						},
						"next_tasks": ["data_splitting"]
					},
					{
						"id": "data_splitting",
						"name": "Data Splitting",
						"type": "processing",
						"description": "Split data into training, validation, and test sets",
						"config": {
							"split_strategy": "stratified",
							"train_ratio": 0.7,
							"validation_ratio": 0.15,
							"test_ratio": 0.15,
							"random_seed": 42
						},
						"next_tasks": ["hyperparameter_tuning"]
					},
					{
						"id": "hyperparameter_tuning",
						"name": "Hyperparameter Tuning",
						"type": "ml_training",
						"description": "Optimize model hyperparameters using grid search or Bayesian optimization",
						"config": {
							"optimization_method": "bayesian",
							"model_types": ["random_forest", "xgboost", "neural_network"],
							"parameter_space": {
								"random_forest": {
									"n_estimators": [100, 200, 500],
									"max_depth": [10, 20, None],
									"min_samples_split": [2, 5, 10]
								}
							},
							"cv_folds": 5,
							"max_trials": 100,
							"metric": "f1_score"
						},
						"next_tasks": ["model_training"]
					},
					{
						"id": "model_training",
						"name": "Model Training",
						"type": "ml_training",
						"description": "Train final model with optimized hyperparameters",
						"config": {
							"model_type": "ensemble",
							"use_gpu": True,
							"early_stopping": True,
							"checkpoint_enabled": True,
							"experiment_tracking": "mlflow"
						},
						"next_tasks": ["model_evaluation"]
					},
					{
						"id": "model_evaluation",
						"name": "Model Evaluation",
						"type": "ml_evaluation",
						"description": "Evaluate model performance on test set",
						"config": {
							"metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
							"evaluation_methods": ["confusion_matrix", "feature_importance", "shap_analysis"],
							"benchmark_models": ["baseline_model", "previous_production_model"],
							"performance_threshold": 0.85
						},
						"next_tasks": ["model_validation"]
					},
					{
						"id": "model_validation",
						"name": "Model Validation",
						"type": "validation",
						"description": "Validate model meets business and technical requirements",
						"config": {
							"validation_checks": [
								{"type": "performance_threshold", "metric": "f1_score", "threshold": 0.85},
								{"type": "bias_detection", "protected_attributes": ["gender", "age_group"]},
								{"type": "robustness_test", "perturbation_methods": ["noise", "adversarial"]},
								{"type": "explainability_check", "method": "lime"}
							]
						},
						"next_tasks": ["model_registration"]
					},
					{
						"id": "model_registration",
						"name": "Model Registration",
						"type": "integration",
						"description": "Register model in model registry",
						"config": {
							"integration_id": "model_registry",
							"model_name": "${project_name}_model",
							"version_strategy": "semantic",
							"metadata": {
								"framework": "scikit-learn",
								"metrics": "${evaluation_metrics}",
								"training_data": "${training_dataset_id}"
							}
						},
						"next_tasks": ["deployment_approval"]
					},
					{
						"id": "deployment_approval",
						"name": "Deployment Approval",
						"type": "approval",
						"description": "Approve model for production deployment",
						"config": {
							"approvers": ["ml_engineer", "data_scientist", "product_owner"],
							"auto_approve_threshold": 0.9,
							"required_documents": ["model_card", "evaluation_report", "bias_report"]
						},
						"next_tasks": ["model_deployment"]
					},
					{
						"id": "model_deployment",
						"name": "Model Deployment",
						"type": "deployment",
						"description": "Deploy model to production serving infrastructure",
						"config": {
							"deployment_target": "kubernetes",
							"serving_framework": "seldon",
							"auto_scaling_enabled": True,
							"monitoring_enabled": True,
							"canary_deployment": True
						},
						"next_tasks": ["setup_monitoring"]
					},
					{
						"id": "setup_monitoring",
						"name": "Setup Model Monitoring",
						"type": "integration",
						"description": "Setup model performance and drift monitoring",
						"config": {
							"integration_id": "model_monitoring",
							"metrics_to_track": ["prediction_accuracy", "data_drift", "model_drift", "latency"],
							"alert_thresholds": {
								"accuracy_drop": 0.05,
								"data_drift_score": 0.3,
								"latency_p95": 100
							}
						},
						"next_tasks": ["notify_stakeholders"]
					},
					{
						"id": "notify_stakeholders",
						"name": "Notify Stakeholders",
						"type": "notification",
						"description": "Send training completion and deployment notification",
						"config": {
							"notification_type": "email",
							"recipients": ["ml_team", "product_team", "engineering_team"],
							"include_reports": True,
							"include_model_card": True
						},
						"next_tasks": []
					}
				],
				"error_handling": {
					"retry_attempts": 2,
					"rollback_enabled": True,
					"checkpoint_recovery": True
				}
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"project_config": {
						"type": "object",
						"properties": {
							"project_name": {"type": "string"},
							"model_type": {"type": "string", "enum": ["classification", "regression", "clustering"]},
							"target_variable": {"type": "string"},
							"feature_columns": {"type": "array", "items": {"type": "string"}}
						},
						"required": ["project_name", "model_type", "target_variable"]
					},
					"data_config": {
						"type": "object",
						"properties": {
							"data_source": {"type": "string"},
							"training_period_days": {"type": "integer", "default": 90},
							"min_samples": {"type": "integer", "default": 1000}
						},
						"required": ["data_source"]
					},
					"training_config": {
						"type": "object",
						"properties": {
							"algorithms": {
								"type": "array",
								"items": {"type": "string"},
								"default": ["random_forest", "xgboost"]
							},
							"cross_validation_folds": {"type": "integer", "default": 5},
							"hyperparameter_trials": {"type": "integer", "default": 100},
							"early_stopping_patience": {"type": "integer", "default": 10}
						}
					},
					"evaluation_config": {
						"type": "object",
						"properties": {
							"primary_metric": {"type": "string", "default": "f1_score"},
							"performance_threshold": {"type": "number", "default": 0.85},
							"bias_detection_enabled": {"type": "boolean", "default": true}
						}
					},
					"deployment_config": {
						"type": "object",
						"properties": {
							"deployment_environment": {"type": "string", "default": "production"},
							"auto_scaling_min_replicas": {"type": "integer", "default": 2},
							"auto_scaling_max_replicas": {"type": "integer", "default": 10},
							"monitoring_enabled": {"type": "boolean", "default": true}
						}
					}
				},
				"required": ["project_config", "data_config"]
			},
			documentation="""
# ML Model Training Pipeline Template

Comprehensive machine learning model training workflow implementing MLOps best practices for production-ready model development.

## Pipeline Stages

### Data Preparation
- **Data Extraction**: Retrieve training data from feature store or data warehouse
- **Data Validation**: Schema validation, data drift detection, and quality checks
- **Feature Engineering**: Automated feature transformations and selection
- **Data Splitting**: Stratified splitting into train/validation/test sets

### Model Development
- **Hyperparameter Tuning**: Bayesian optimization for optimal parameters
- **Model Training**: Multi-algorithm training with experiment tracking
- **Model Evaluation**: Comprehensive performance evaluation with multiple metrics
- **Model Validation**: Bias detection, robustness testing, and explainability

### Model Deployment
- **Model Registration**: Version control and metadata management
- **Deployment Approval**: Multi-stakeholder approval process
- **Production Deployment**: Containerized deployment with auto-scaling
- **Monitoring Setup**: Real-time performance and drift monitoring

## Key Features
- **AutoML Capabilities**: Automated algorithm selection and hyperparameter tuning
- **Experiment Tracking**: Integration with MLflow for experiment management
- **Model Versioning**: Semantic versioning and model lineage tracking
- **Bias Detection**: Automated fairness and bias assessment
- **Explainability**: SHAP and LIME integration for model interpretability
- **Continuous Monitoring**: Real-time model performance monitoring

## Supported Algorithms
- Random Forest
- XGBoost
- Neural Networks
- Support Vector Machines
- Linear/Logistic Regression
- Ensemble Methods

## MLOps Integration
- Feature Store integration
- Model Registry management
- Continuous Integration/Deployment
- A/B testing capabilities
- Model performance monitoring
- Automated retraining triggers

## Prerequisites
- Feature store or data warehouse access
- Model registry (MLflow, Kubeflow, etc.)
- Container orchestration platform (Kubernetes)
- Model serving infrastructure (Seldon, KFServing)
- Monitoring and alerting systems
			""",
			use_cases=[
				"Customer churn prediction",
				"Fraud detection models",
				"Recommendation systems",
				"Price optimization models",
				"Demand forecasting",
				"Risk assessment models",
				"Image classification",
				"Natural language processing",
				"Time series forecasting"
			],
			prerequisites=[
				"Feature store or data warehouse",
				"Model registry (MLflow, Kubeflow)",
				"Container orchestration (Kubernetes)",
				"Model serving platform",
				"Experiment tracking system",
				"Monitoring and alerting infrastructure",
				"GPU resources for training (optional)"
			],
			estimated_duration=14400,  # 4 hours
			complexity_score=9,
			is_verified=True,
			is_featured=True
		)
		
		self._register_template(template)
	
	# Continue with additional utility methods...
	def _register_template(self, template: WorkflowTemplate):
		"""Register a template in the library."""
		self.templates[template.id] = template
		
		# Update category index
		if template.category not in self.category_index:
			self.category_index[template.category] = []
		self.category_index[template.category].append(template.id)
		
		# Update tag index
		for tag in template.tags:
			if tag not in self.tag_index:
				self.tag_index[tag] = []
			self.tag_index[tag].append(template.id)
	
	# Add remaining template methods...
	def _add_invoice_processing_template(self):
		"""Add invoice processing workflow template."""
		from .additional_templates import create_invoice_processing_workflow
		template = create_invoice_processing_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added invoice processing template: {template.name}")
	
	def _add_customer_support_ticket_template(self):
		"""Add customer support ticket workflow template."""
		from .additional_templates import create_customer_support_ticket_workflow
		template = create_customer_support_ticket_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added customer support ticket template: {template.name}")
	
	def _add_purchase_order_template(self):
		"""Add purchase order processing workflow template."""
		from .additional_templates import create_purchase_order_workflow
		template = create_purchase_order_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added purchase order template: {template.name}")
	
	def _add_etl_pipeline_template(self):
		"""Add ETL pipeline workflow template."""
		from .additional_templates import create_etl_pipeline_workflow
		template = create_etl_pipeline_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added ETL pipeline template: {template.name}")
	
	def _add_report_generation_template(self):
		"""Add report generation workflow template."""
		from .additional_templates import create_report_generation_workflow
		template = create_report_generation_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added report generation template: {template.name}")
	
	def _add_data_quality_check_template(self):
		"""Add data quality check workflow template."""
		from .additional_templates import create_data_quality_check_workflow
		template = create_data_quality_check_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added data quality check template: {template.name}")
	
	def _add_backup_workflow_template(self):
		"""Add backup workflow template."""
		from .additional_templates import create_backup_workflow
		template = create_backup_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added backup workflow template: {template.name}")
	
	def _add_api_sync_template(self):
		"""Add API synchronization workflow template."""
		from .additional_templates import create_api_sync_workflow
		template = create_api_sync_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added API sync template: {template.name}")
	
	def _add_database_migration_template(self):
		"""Add database migration workflow template."""
		from .additional_templates import create_database_migration_workflow
		template = create_database_migration_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added database migration template: {template.name}")
	
	def _add_file_processing_template(self):
		"""Add file processing workflow template."""
		from .additional_templates import create_file_processing_workflow
		template = create_file_processing_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added file processing template: {template.name}")
	
	def _add_notification_workflow_template(self):
		"""Add notification workflow template."""
		from .additional_templates import create_notification_workflow
		template = create_notification_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added notification workflow template: {template.name}")
	
	def _add_patient_admission_template(self):
		"""Add patient admission workflow template."""
		from .additional_templates import create_patient_admission_workflow
		template = create_patient_admission_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added patient admission template: {template.name}")
	
	def _add_order_fulfillment_template(self):
		"""Add order fulfillment workflow template."""
		from .additional_templates import create_order_fulfillment_workflow
		template = create_order_fulfillment_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added order fulfillment template: {template.name}")
	
	def _add_incident_response_template(self):
		"""Add incident response workflow template."""
		from .additional_templates import create_incident_response_workflow
		template = create_incident_response_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added incident response template: {template.name}")
	
	def _add_monitoring_template(self):
		"""Add monitoring workflow template."""
		from .additional_templates import create_monitoring_workflow
		template = create_monitoring_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added monitoring template: {template.name}")
	
	def _add_security_scan_template(self):
		"""Add security scan workflow template."""
		from .additional_templates import create_security_scan_workflow
		template = create_security_scan_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added security scan template: {template.name}")
	
	def _add_iot_data_processing_template(self):
		"""Add IoT data processing workflow template."""
		from .additional_templates import create_iot_data_processing_workflow
		template = create_iot_data_processing_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added IoT data processing template: {template.name}")
	
	def _add_blockchain_transaction_template(self):
		"""Add blockchain transaction workflow template."""
		from .additional_templates import create_blockchain_transaction_workflow
		template = create_blockchain_transaction_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added blockchain transaction template: {template.name}")
	
	def _add_microservice_deployment_template(self):
		"""Add microservice deployment workflow template."""
		from .additional_templates import create_microservice_deployment_workflow
		template = create_microservice_deployment_workflow()
		self.workflow_templates[template.id] = template
		logger.info(f"Added microservice deployment template: {template.name}")
	
	async def _load_templates_from_database(self):
		"""Load custom templates from database."""
		try:
			# Load custom templates from database
			templates_query = """
			SELECT 
				id, name, description, definition, category, tags, version,
				complexity_score, estimated_duration, is_featured, is_active,
				created_by, created_at, updated_at, metadata
			FROM cr_workflow_templates 
			WHERE tenant_id = %s 
			AND is_active = true
			ORDER BY name ASC
			"""
			
			template_rows = await self.database.fetch_all(templates_query, (self.tenant_id,))
			
			if not template_rows:
				logger.info("No custom templates found in database")
				return
			
			for row in template_rows:
				try:
					# Parse template definition
					definition = json.loads(row['definition']) if isinstance(row['definition'], str) else row['definition']
					tags = json.loads(row['tags']) if isinstance(row['tags'], str) else (row['tags'] or [])
					metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else (row['metadata'] or {})
					
					# Create WorkflowTemplate instance
					template = WorkflowTemplate(
						id=row['id'],
						name=row['name'],
						description=row['description'] or "",
						definition=definition,
						category=TemplateCategory(row['category']) if row['category'] else TemplateCategory.BUSINESS_PROCESS,
						tags=[TemplateTags(tag) for tag in tags if tag in [t.value for t in TemplateTags]],
						version=row['version'] or "1.0.0",
						complexity_score=float(row['complexity_score'] or 5.0),
						estimated_duration=int(row['estimated_duration'] or 3600),
						is_featured=bool(row['is_featured']),
						prerequisites=metadata.get('prerequisites', []),
						configuration_schema=metadata.get('configuration_schema', {}),
						documentation=metadata.get('documentation', {}),
						use_cases=metadata.get('use_cases', []),
						created_at=row['created_at'],
						updated_at=row['updated_at']
					)
					
					# Add to template library
					self.workflow_templates[template.id] = template
					logger.debug(f"Loaded custom template: {template.name}")
					
				except Exception as template_error:
					logger.error(f"Failed to load template {row.get('id', 'unknown')}: {template_error}")
					continue
			
			logger.info(f"Loaded {len(template_rows)} custom templates from database")
			
		except Exception as e:
			logger.error(f"Failed to load templates from database: {str(e)}")
	
	# Public API methods
	
	async def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
		"""Get template by ID."""
		return self.templates.get(template_id)
	
	async def get_templates_by_category(self, category: TemplateCategory) -> List[WorkflowTemplate]:
		"""Get templates by category."""
		template_ids = self.category_index.get(category, [])
		return [self.templates[template_id] for template_id in template_ids]
	
	async def get_templates_by_tag(self, tag: TemplateTags) -> List[WorkflowTemplate]:
		"""Get templates by tag."""
		template_ids = self.tag_index.get(tag, [])
		return [self.templates[template_id] for template_id in template_ids]
	
	async def search_templates(self, query: str, filters: Dict[str, Any] = None) -> List[WorkflowTemplate]:
		"""Search templates by query and filters."""
		results = []
		
		for template in self.templates.values():
			# Text search in name, description, and use cases
			if (query.lower() in template.name.lower() or 
				query.lower() in template.description.lower() or
				any(query.lower() in use_case.lower() for use_case in template.use_cases)):
				
				# Apply filters
				if filters:
					if filters.get('category') and template.category != filters['category']:
						continue
					if filters.get('tags') and not any(tag in template.tags for tag in filters['tags']):
						continue
					if filters.get('complexity_min') and template.complexity_score < filters['complexity_min']:
						continue
					if filters.get('complexity_max') and template.complexity_score > filters['complexity_max']:
						continue
					if filters.get('verified_only') and not template.is_verified:
						continue
				
				results.append(template)
		
		# Sort by popularity and relevance
		results.sort(key=lambda t: (t.is_featured, t.popularity_score, t.usage_count), reverse=True)
		
		return results
	
	async def get_featured_templates(self, limit: int = 10) -> List[WorkflowTemplate]:
		"""Get featured templates."""
		featured = [t for t in self.templates.values() if t.is_featured]
		featured.sort(key=lambda t: (t.popularity_score, t.usage_count), reverse=True)
		return featured[:limit]
	
	async def get_popular_templates(self, limit: int = 10) -> List[WorkflowTemplate]:
		"""Get popular templates."""
		popular = sorted(self.templates.values(), key=lambda t: t.usage_count, reverse=True)
		return popular[:limit]
	
	async def increment_usage_count(self, template_id: str):
		"""Increment template usage count."""
		if template_id in self.templates:
			self.templates[template_id].usage_count += 1
	
	async def create_workflow_from_template(self, template_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
		"""Create workflow instance from template."""
		template = await self.get_template(template_id)
		if not template:
			raise ValueError(f"Template not found: {template_id}")
		
		# Increment usage count
		await self.increment_usage_count(template_id)
		
		# Create workflow definition from template
		workflow_definition = template.workflow_definition.copy()
		
		# Apply configuration overrides
		if config:
			# Merge configuration with template defaults
			try:
				# Deep merge configuration into workflow definition
				if isinstance(config, dict):
					# Apply configuration to workflow-level settings
					if 'workflow' in config:
						workflow_config = config['workflow']
						if 'name' in workflow_config:
							workflow_definition['name'] = workflow_config['name']
						if 'description' in workflow_config:
							workflow_definition['description'] = workflow_config['description']
						if 'timeout' in workflow_config:
							workflow_definition['timeout'] = workflow_config['timeout']
						if 'retry_policy' in workflow_config:
							workflow_definition['retry_policy'] = workflow_config['retry_policy']
					
					# Apply configuration to task-level settings
					if 'tasks' in config and 'tasks' in workflow_definition:
						task_configs = config['tasks']
						for i, task in enumerate(workflow_definition['tasks']):
							if str(i) in task_configs:
								task_config = task_configs[str(i)]
								# Merge task configuration
								task.update(task_config)
							elif task.get('id') in task_configs:
								task_config = task_configs[task['id']]
								task.update(task_config)
					
					# Apply global parameter overrides
					if 'parameters' in config:
						workflow_definition.setdefault('parameters', {})
						workflow_definition['parameters'].update(config['parameters'])
					
					# Apply environment-specific settings
					if 'environment' in config:
						env_config = config['environment']
						workflow_definition.setdefault('environment', {})
						workflow_definition['environment'].update(env_config)
					
					# Apply connector configurations
					if 'connectors' in config:
						connector_configs = config['connectors']
						for task in workflow_definition.get('tasks', []):
							if 'connector' in task and task['connector'] in connector_configs:
								task.setdefault('config', {})
								task['config'].update(connector_configs[task['connector']])
					
					logger.debug(f"Applied configuration overrides to template {template.name}")
				
			except Exception as config_error:
				logger.error(f"Failed to apply configuration to template {template.name}: {config_error}")
				# Continue with original template if config merge fails
		
		return workflow_definition


# Global service instance
templates_library_service = TemplatesLibraryService()