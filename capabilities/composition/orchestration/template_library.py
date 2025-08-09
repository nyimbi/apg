#!/usr/bin/env python3
"""
APG Workflow Orchestration Template Library

Extensive templates library with pre-built workflow templates, component collections,
industry-specific templates, and template management system.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field, validator
import yaml

from apg.framework.base_service import APGBaseService
from apg.framework.database import APGDatabase
from apg.framework.audit_compliance import APGAuditLogger

from .config import get_config
from .models import WorkflowStatus, TaskStatus
from .component_library import ComponentType, ComponentCategory, component_library


logger = logging.getLogger(__name__)


class TemplateCategory(str, Enum):
	"""Template categories for organization."""
	DATA_PROCESSING = "data_processing"
	BUSINESS_PROCESS = "business_process"
	INTEGRATION = "integration"
	AUTOMATION = "automation"
	ANALYTICS = "analytics"
	MONITORING = "monitoring"
	NOTIFICATION = "notification"
	APPROVAL = "approval"
	MACHINE_LEARNING = "machine_learning"
	DEVOPS = "devops"
	FINANCE = "finance"
	HR = "hr"
	MARKETING = "marketing"
	CUSTOMER_SERVICE = "customer_service"
	SUPPLY_CHAIN = "supply_chain"
	SECURITY = "security"
	COMPLIANCE = "compliance"
	HEALTHCARE = "healthcare"
	EDUCATION = "education"
	CUSTOM = "custom"


class TemplateComplexity(str, Enum):
	"""Template complexity levels."""
	BEGINNER = "beginner"
	INTERMEDIATE = "intermediate"
	ADVANCED = "advanced"
	EXPERT = "expert"


class TemplateType(str, Enum):
	"""Template types."""
	WORKFLOW = "workflow"
	COMPONENT_SET = "component_set"
	PATTERN = "pattern"
	SNIPPET = "snippet"
	BLUEPRINT = "blueprint"


@dataclass
class TemplateMetadata:
	"""Template metadata."""
	id: str
	name: str
	description: str
	category: TemplateCategory
	complexity: TemplateComplexity
	template_type: TemplateType
	version: str = "1.0.0"
	author: str = "APG System"
	
	# Usage information
	use_cases: List[str] = field(default_factory=list)
	tags: List[str] = field(default_factory=list)
	industry: Optional[str] = None
	
	# Requirements
	required_capabilities: List[str] = field(default_factory=list)
	required_connectors: List[str] = field(default_factory=list)
	estimated_runtime: Optional[str] = None
	
	# Metrics
	popularity_score: float = 0.0
	usage_count: int = 0
	rating: float = 0.0
	rating_count: int = 0
	
	# Timestamps
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	
	# Configuration
	customizable_params: List[str] = field(default_factory=list)
	preview_image: Optional[str] = None
	documentation_url: Optional[str] = None


@dataclass
class WorkflowTemplate:
	"""Complete workflow template."""
	metadata: TemplateMetadata
	workflow_definition: Dict[str, Any]
	components: List[Dict[str, Any]]
	connections: List[Dict[str, Any]]
	parameters: Dict[str, Any] = field(default_factory=dict)
	configuration: Dict[str, Any] = field(default_factory=dict)
	test_data: Dict[str, Any] = field(default_factory=dict)
	validation_rules: List[str] = field(default_factory=list)


@dataclass
class ComponentSet:
	"""Collection of related components."""
	metadata: TemplateMetadata
	components: List[Dict[str, Any]]
	component_relationships: Dict[str, List[str]] = field(default_factory=dict)
	usage_patterns: List[Dict[str, Any]] = field(default_factory=list)
	best_practices: List[str] = field(default_factory=list)


class TemplateLibrary:
	"""Manages workflow templates and component sets."""
	
	def __init__(self):
		self.workflow_templates: Dict[str, WorkflowTemplate] = {}
		self.component_sets: Dict[str, ComponentSet] = {}
		self.categories: Dict[TemplateCategory, List[str]] = {}
		self.tag_index: Dict[str, Set[str]] = {}
		
		# Initialize with built-in templates
		self._load_builtin_templates()
	
	def _load_builtin_templates(self):
		"""Load built-in templates."""
		self._create_data_processing_templates()
		self._create_business_process_templates()
		self._create_integration_templates()
		self._create_automation_templates()
		self._create_analytics_templates()
		self._create_monitoring_templates()
		self._create_ml_templates()
		self._create_devops_templates()
		self._create_industry_templates()
	
	def _create_data_processing_templates(self):
		"""Create data processing templates."""
		
		# ETL Pipeline Template
		etl_metadata = TemplateMetadata(
			id="etl_pipeline",
			name="ETL Pipeline",
			description="Extract, Transform, Load data pipeline with validation and error handling",
			category=TemplateCategory.DATA_PROCESSING,
			complexity=TemplateComplexity.INTERMEDIATE,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Data migration between systems",
				"Regular data synchronization",
				"Data warehouse loading",
				"Data cleaning and transformation"
			],
			tags=["ETL", "data", "pipeline", "transformation", "validation"],
			required_connectors=["database", "file_system"],
			estimated_runtime="5-30 minutes"
		)
		
		etl_workflow = WorkflowTemplate(
			metadata=etl_metadata,
			workflow_definition={
				"name": "ETL Pipeline",
				"description": "Extract, Transform, Load data pipeline",
				"trigger": "manual"
			},
			components=[
				{
					"id": "extract_source",
					"type": "database_query",
					"name": "Extract Data",
					"config": {
						"query": "SELECT * FROM source_table WHERE updated_at > :last_sync",
						"connection": "source_db"
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "validate_data",
					"type": "task",
					"name": "Validate Data",
					"config": {
						"task_type": "validation",
						"validation_rules": [
							"data.length > 0",
							"all(record.get('id') for record in data)"
						]
					},
					"position": {"x": 300, "y": 100}
				},
				{
					"id": "transform_data",
					"type": "transform",
					"name": "Transform Data",
					"config": {
						"transformations": [
							{
								"type": "map",
								"mapping": {
									"customer_id": "$.id",
									"full_name": "$.first_name + ' ' + $.last_name",
									"email_normalized": "$.email.lower()"
								}
							}
						]
					},
					"position": {"x": 500, "y": 100}
				},
				{
					"id": "load_target",
					"type": "database_query",
					"name": "Load Data",
					"config": {
						"query": "INSERT INTO target_table VALUES (:data)",
						"connection": "target_db",
						"batch_size": 1000
					},
					"position": {"x": 700, "y": 100}
				},
				{
					"id": "error_handler",
					"type": "email",
					"name": "Error Notification",
					"config": {
						"default_to": "admin@example.com",
						"default_subject": "ETL Pipeline Error"
					},
					"position": {"x": 400, "y": 250}
				}
			],
			connections=[
				{"from": "extract_source", "to": "validate_data"},
				{"from": "validate_data", "to": "transform_data", "condition": "success"},
				{"from": "validate_data", "to": "error_handler", "condition": "error"},
				{"from": "transform_data", "to": "load_target"},
				{"from": "transform_data", "to": "error_handler", "condition": "error"}
			],
			parameters={
				"source_connection": {
					"type": "string",
					"description": "Source database connection",
					"required": True
				},
				"target_connection": {
					"type": "string",
					"description": "Target database connection",
					"required": True
				},
				"batch_size": {
					"type": "integer",
					"description": "Batch size for data processing",
					"default": 1000
				}
			}
		)
		
		self.add_workflow_template(etl_workflow)
		
		# Data Quality Assessment Template
		dq_metadata = TemplateMetadata(
			id="data_quality_assessment",
			name="Data Quality Assessment",
			description="Comprehensive data quality checks and reporting",
			category=TemplateCategory.DATA_PROCESSING,
			complexity=TemplateComplexity.ADVANCED,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Data quality monitoring",
				"Pre-migration data assessment",
				"Compliance reporting",
				"Data governance"
			],
			tags=["data quality", "assessment", "validation", "reporting"],
			required_connectors=["database", "email"],
			estimated_runtime="10-60 minutes"
		)
		
		dq_workflow = WorkflowTemplate(
			metadata=dq_metadata,
			workflow_definition={
				"name": "Data Quality Assessment",
				"description": "Assess and report on data quality metrics",
				"trigger": "scheduled"
			},
			components=[
				{
					"id": "extract_sample",
					"type": "database_query",
					"name": "Extract Data Sample",
					"config": {
						"query": "SELECT * FROM {{table_name}} TABLESAMPLE SYSTEM (10)",
						"connection": "data_source"
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "completeness_check",
					"type": "script",
					"name": "Completeness Check",
					"config": {
						"script_type": "python",
						"script_code": """
completeness_scores = {}
for column in input_data[0].keys():
	non_null_count = sum(1 for row in input_data if row.get(column) is not None)
	completeness_scores[column] = non_null_count / len(input_data)

result = {
	'metric': 'completeness',
	'scores': completeness_scores,
	'overall_score': sum(completeness_scores.values()) / len(completeness_scores)
}
						"""
					},
					"position": {"x": 300, "y": 100}
				},
				{
					"id": "accuracy_check",
					"type": "script",
					"name": "Accuracy Check",
					"config": {
						"script_type": "python",
						"script_code": """
import re

accuracy_scores = {}
patterns = {
	'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
	'phone': r'^\+?1?[-.\s()]?\d{3}[-.\s()]?\d{3}[-.\s()]?\d{4}$'
}

for column, pattern in patterns.items():
	if column in input_data[0]:
		valid_count = sum(1 for row in input_data 
						 if row.get(column) and re.match(pattern, str(row[column])))
		accuracy_scores[column] = valid_count / len(input_data)

result = {
	'metric': 'accuracy',
	'scores': accuracy_scores,
	'overall_score': sum(accuracy_scores.values()) / len(accuracy_scores) if accuracy_scores else 1.0
}
						"""
					},
					"position": {"x": 500, "y": 100}
				},
				{
					"id": "generate_report",
					"type": "script",
					"name": "Generate Quality Report",
					"config": {
						"script_type": "python",
						"script_code": """
completeness = context.get('completeness_result', {})
accuracy = context.get('accuracy_result', {})

report = {
	'assessment_date': datetime.now().isoformat(),
	'data_sample_size': len(input_data),
	'completeness': completeness,
	'accuracy': accuracy,
	'overall_quality_score': (
		completeness.get('overall_score', 0) + 
		accuracy.get('overall_score', 0)
	) / 2,
	'recommendations': []
}

# Add recommendations based on scores
if report['overall_quality_score'] < 0.8:
	report['recommendations'].append('Data quality improvement needed')
if completeness.get('overall_score', 0) < 0.9:
	report['recommendations'].append('Address missing data issues')
if accuracy.get('overall_score', 0) < 0.95:
	report['recommendations'].append('Implement data validation rules')

result = report
						"""
					},
					"position": {"x": 700, "y": 150}
				},
				{
					"id": "send_report",
					"type": "email",
					"name": "Send Quality Report",
					"config": {
						"default_to": "data-team@example.com",
						"default_subject": "Data Quality Assessment Report"
					},
					"position": {"x": 900, "y": 150}
				}
			],
			connections=[
				{"from": "extract_sample", "to": "completeness_check"},
				{"from": "extract_sample", "to": "accuracy_check"},
				{"from": "completeness_check", "to": "generate_report"},
				{"from": "accuracy_check", "to": "generate_report"},
				{"from": "generate_report", "to": "send_report"}
			],
			parameters={
				"table_name": {
					"type": "string",
					"description": "Table to assess",
					"required": True
				},
				"sample_percentage": {
					"type": "integer",
					"description": "Percentage of data to sample",
					"default": 10,
					"minimum": 1,
					"maximum": 100
				}
			}
		)
		
		self.add_workflow_template(dq_workflow)
	
	def _create_business_process_templates(self):
		"""Create business process templates."""
		
		# Approval Workflow Template
		approval_metadata = TemplateMetadata(
			id="approval_workflow",
			name="Multi-Step Approval Workflow",
			description="Configurable multi-step approval process with escalation",
			category=TemplateCategory.BUSINESS_PROCESS,
			complexity=TemplateComplexity.INTERMEDIATE,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Purchase order approval",
				"Document approval",
				"Budget approval",
				"Project approval"
			],
			tags=["approval", "workflow", "escalation", "notification"],
			required_capabilities=["user_management", "notifications"],
			estimated_runtime="Variable (depends on approvers)"
		)
		
		approval_workflow = WorkflowTemplate(
			metadata=approval_metadata,
			workflow_definition={
				"name": "Multi-Step Approval Workflow",
				"description": "Configurable approval process",
				"trigger": "manual"
			},
			components=[
				{
					"id": "submit_request",
					"type": "start",
					"name": "Submit Request",
					"config": {
						"initial_data": {
							"status": "pending",
							"submission_date": "{{current_timestamp}}"
						}
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "manager_approval",
					"type": "human_task",
					"name": "Manager Approval",
					"config": {
						"assignee_role": "manager",
						"timeout_hours": 48,
						"form_fields": [
							{"name": "decision", "type": "radio", "options": ["approve", "reject", "request_changes"]},
							{"name": "comments", "type": "textarea", "required": False}
						]
					},
					"position": {"x": 300, "y": 100}
				},
				{
					"id": "director_approval",
					"type": "human_task",
					"name": "Director Approval",
					"config": {
						"assignee_role": "director",
						"timeout_hours": 72,
						"condition": "amount > 10000",
						"form_fields": [
							{"name": "decision", "type": "radio", "options": ["approve", "reject"]},
							{"name": "comments", "type": "textarea", "required": False}
						]
					},
					"position": {"x": 500, "y": 100}
				},
				{
					"id": "notify_approved",
					"type": "notification",
					"name": "Notify Approval",
					"config": {
						"recipients": ["{{requester_email}}"],
						"template": "approval_granted",
						"channels": ["email", "in_app"]
					},
					"position": {"x": 700, "y": 100}
				},
				{
					"id": "notify_rejected",
					"type": "notification",
					"name": "Notify Rejection",
					"config": {
						"recipients": ["{{requester_email}}"],
						"template": "approval_rejected",
						"channels": ["email", "in_app"]
					},
					"position": {"x": 700, "y": 200}
				},
				{
					"id": "escalation_timeout",
					"type": "notification",
					"name": "Escalation Alert",
					"config": {
						"recipients": ["{{escalation_email}}"],
						"template": "approval_timeout",
						"trigger": "timeout"
					},
					"position": {"x": 400, "y": 300}
				}
			],
			connections=[
				{"from": "submit_request", "to": "manager_approval"},
				{"from": "manager_approval", "to": "director_approval", "condition": "decision == 'approve' && amount > 10000"},
				{"from": "manager_approval", "to": "notify_approved", "condition": "decision == 'approve' && amount <= 10000"},
				{"from": "manager_approval", "to": "notify_rejected", "condition": "decision == 'reject'"},
				{"from": "director_approval", "to": "notify_approved", "condition": "decision == 'approve'"},
				{"from": "director_approval", "to": "notify_rejected", "condition": "decision == 'reject'"},
				{"from": "manager_approval", "to": "escalation_timeout", "condition": "timeout"},
				{"from": "director_approval", "to": "escalation_timeout", "condition": "timeout"}
			],
			parameters={
				"approval_threshold": {
					"type": "number",
					"description": "Amount requiring director approval",
					"default": 10000
				},
				"manager_timeout_hours": {
					"type": "integer",
					"description": "Hours before manager approval times out",
					"default": 48
				},
				"director_timeout_hours": {
					"type": "integer",
					"description": "Hours before director approval times out",
					"default": 72
				}
			}
		)
		
		self.add_workflow_template(approval_workflow)
	
	def _create_integration_templates(self):
		"""Create integration templates."""
		
		# API Integration Template
		api_metadata = TemplateMetadata(
			id="api_integration_sync",
			name="API Integration & Sync",
			description="Bidirectional data synchronization between APIs with conflict resolution",
			category=TemplateCategory.INTEGRATION,
			complexity=TemplateComplexity.ADVANCED,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"CRM to ERP synchronization",
				"Multi-system data sync",
				"API data aggregation",
				"Real-time integration"
			],
			tags=["API", "integration", "sync", "conflict resolution"],
			required_connectors=["http_request"],
			estimated_runtime="2-15 minutes"
		)
		
		api_workflow = WorkflowTemplate(
			metadata=api_metadata,
			workflow_definition={
				"name": "API Integration & Sync",
				"description": "Synchronize data between multiple APIs",
				"trigger": "scheduled"
			},
			components=[
				{
					"id": "fetch_source_a",
					"type": "http_request",
					"name": "Fetch from API A",
					"config": {
						"method": "GET",
						"url": "{{api_a_endpoint}}/records",
						"headers": {
							"Authorization": "Bearer {{api_a_token}}",
							"Content-Type": "application/json"
						},
						"timeout": 30
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "fetch_source_b", 
					"type": "http_request",
					"name": "Fetch from API B",
					"config": {
						"method": "GET",
						"url": "{{api_b_endpoint}}/records",
						"headers": {
							"Authorization": "Bearer {{api_b_token}}",
							"Content-Type": "application/json"
						},
						"timeout": 30
					},
					"position": {"x": 100, "y": 200}
				},
				{
					"id": "conflict_resolution",
					"type": "script",
					"name": "Resolve Conflicts",
					"config": {
						"script_type": "python",
						"script_code": """
source_a = context.get('fetch_source_a_result', {}).get('data', [])
source_b = context.get('fetch_source_b_result', {}).get('data', [])

# Create lookup maps
a_map = {record['id']: record for record in source_a}
b_map = {record['id']: record for record in source_b}

conflicts = []
resolved_records = []

# Find conflicts and resolve
for record_id in set(a_map.keys()) | set(b_map.keys()):
	a_record = a_map.get(record_id)
	b_record = b_map.get(record_id)
	
	if a_record and b_record:
		# Both exist - check for conflicts
		if a_record.get('updated_at') > b_record.get('updated_at'):
			resolved_records.append(a_record)
		else:
			resolved_records.append(b_record)
			
		if a_record != b_record:
			conflicts.append({
				'id': record_id,
				'source_a': a_record,
				'source_b': b_record,
				'resolved_to': 'most_recent'
			})
	elif a_record:
		resolved_records.append(a_record)
	elif b_record:
		resolved_records.append(b_record)

result = {
	'resolved_records': resolved_records,
	'conflicts': conflicts,
	'conflict_count': len(conflicts)
}
						"""
					},
					"position": {"x": 400, "y": 150}
				},
				{
					"id": "sync_to_a",
					"type": "http_request",
					"name": "Sync to API A",
					"config": {
						"method": "POST",
						"url": "{{api_a_endpoint}}/batch_update",
						"headers": {
							"Authorization": "Bearer {{api_a_token}}",
							"Content-Type": "application/json"
						}
					},
					"position": {"x": 600, "y": 100}
				},
				{
					"id": "sync_to_b",
					"type": "http_request", 
					"name": "Sync to API B",
					"config": {
						"method": "POST",
						"url": "{{api_b_endpoint}}/batch_update",
						"headers": {
							"Authorization": "Bearer {{api_b_token}}",
							"Content-Type": "application/json"
						}
					},
					"position": {"x": 600, "y": 200}
				},
				{
					"id": "conflict_notification",
					"type": "email",
					"name": "Conflict Report",
					"config": {
						"default_to": "integration-team@example.com",
						"default_subject": "API Sync Conflicts Detected",
						"condition": "conflict_count > 0"
					},
					"position": {"x": 800, "y": 150}
				}
			],
			connections=[
				{"from": "fetch_source_a", "to": "conflict_resolution"},
				{"from": "fetch_source_b", "to": "conflict_resolution"},
				{"from": "conflict_resolution", "to": "sync_to_a"},
				{"from": "conflict_resolution", "to": "sync_to_b"},
				{"from": "conflict_resolution", "to": "conflict_notification", "condition": "conflict_count > 0"}
			],
			parameters={
				"api_a_endpoint": {
					"type": "string",
					"description": "API A base endpoint",
					"required": True
				},
				"api_b_endpoint": {
					"type": "string", 
					"description": "API B base endpoint",
					"required": True
				},
				"sync_interval": {
					"type": "integer",
					"description": "Sync interval in minutes",
					"default": 60
				}
			}
		)
		
		self.add_workflow_template(api_workflow)
	
	def _create_automation_templates(self):
		"""Create automation templates."""
		
		# File Processing Automation
		file_metadata = TemplateMetadata(
			id="file_processing_automation",
			name="File Processing Automation",
			description="Automated file processing with validation, transformation, and archiving",
			category=TemplateCategory.AUTOMATION,
			complexity=TemplateComplexity.INTERMEDIATE,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Batch file processing",
				"Document workflow automation",
				"Data import automation",
				"File format conversion"
			],
			tags=["files", "automation", "processing", "validation"],
			required_connectors=["file_system"],
			estimated_runtime="1-30 minutes"
		)
		
		file_workflow = WorkflowTemplate(
			metadata=file_metadata,
			workflow_definition={
				"name": "File Processing Automation",
				"description": "Process files automatically with validation and archiving",
				"trigger": "file_watcher"
			},
			components=[
				{
					"id": "file_detected",
					"type": "start",
					"name": "File Detected",
					"config": {
						"trigger_type": "file_watcher",
						"watch_path": "{{input_directory}}",
						"file_pattern": "{{file_pattern}}"
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "validate_file",
					"type": "script",
					"name": "Validate File",
					"config": {
						"script_type": "python",
						"script_code": """
import os
import magic

file_path = input_data.get('file_path')
if not file_path or not os.path.exists(file_path):
	result = {'valid': False, 'error': 'File not found'}
else:
	file_size = os.path.getsize(file_path)
	file_type = magic.from_file(file_path, mime=True)
	
	# Validation rules
	max_size = context.get('max_file_size', 100 * 1024 * 1024)  # 100MB
	allowed_types = context.get('allowed_file_types', ['text/csv', 'application/json', 'text/plain'])
	
	valid = file_size <= max_size and file_type in allowed_types
	
	result = {
		'valid': valid,
		'file_size': file_size,
		'file_type': file_type,
		'file_path': file_path
	}
						"""
					},
					"position": {"x": 300, "y": 100}
				},
				{
					"id": "process_file",
					"type": "script",
					"name": "Process File",
					"config": {
						"script_type": "python",
						"script_code": """
import csv
import json

file_path = input_data.get('file_path')
file_type = input_data.get('file_type')

processed_data = []

if file_type == 'text/csv':
	with open(file_path, 'r') as f:
		reader = csv.DictReader(f)
		processed_data = list(reader)
elif file_type == 'application/json':
	with open(file_path, 'r') as f:
		processed_data = json.load(f)

# Apply transformations
for record in processed_data:
	# Normalize data
	if isinstance(record, dict):
		for key, value in record.items():
			if isinstance(value, str):
				record[key] = value.strip()

result = {
	'processed_data': processed_data,
	'record_count': len(processed_data),
	'processing_completed': True
}
						"""
					},
					"position": {"x": 500, "y": 100}
				},
				{
					"id": "save_output",
					"type": "file_operation",
					"name": "Save Processed Data",
					"config": {
						"operation": "write",
						"output_path": "{{output_directory}}/processed_{{timestamp}}.json",
						"format": "json"
					},
					"position": {"x": 700, "y": 100}
				},
				{
					"id": "archive_file",
					"type": "file_operation",
					"name": "Archive Original",
					"config": {
						"operation": "move",
						"destination": "{{archive_directory}}/{{filename}}_{{timestamp}}"
					},
					"position": {"x": 900, "y": 100}
				},
				{
					"id": "error_handler",
					"type": "email",
					"name": "Error Notification",
					"config": {
						"default_to": "admin@example.com",
						"default_subject": "File Processing Error"
					},
					"position": {"x": 500, "y": 250}
				}
			],
			connections=[
				{"from": "file_detected", "to": "validate_file"},
				{"from": "validate_file", "to": "process_file", "condition": "valid == true"},
				{"from": "validate_file", "to": "error_handler", "condition": "valid == false"},
				{"from": "process_file", "to": "save_output"},
				{"from": "process_file", "to": "error_handler", "condition": "error"},
				{"from": "save_output", "to": "archive_file"},
				{"from": "save_output", "to": "error_handler", "condition": "error"}
			],
			parameters={
				"input_directory": {
					"type": "string",
					"description": "Directory to watch for files",
					"required": True
				},
				"output_directory": {
					"type": "string", 
					"description": "Directory for processed files",
					"required": True
				},
				"archive_directory": {
					"type": "string",
					"description": "Directory for archived files", 
					"required": True
				},
				"file_pattern": {
					"type": "string",
					"description": "File pattern to match",
					"default": "*.csv"
				}
			}
		)
		
		self.add_workflow_template(file_workflow)
	
	def _create_analytics_templates(self):
		"""Create analytics templates."""
		
		# Sales Analytics Dashboard
		analytics_metadata = TemplateMetadata(
			id="sales_analytics_pipeline",
			name="Sales Analytics Pipeline",
			description="Automated sales data analysis and reporting pipeline",
			category=TemplateCategory.ANALYTICS,
			complexity=TemplateComplexity.ADVANCED,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Sales performance analysis",
				"Revenue reporting",
				"Trend analysis",
				"KPI monitoring"
			],
			tags=["analytics", "sales", "reporting", "KPI"],
			required_connectors=["database", "email"],
			estimated_runtime="15-45 minutes"
		)
		
		analytics_workflow = WorkflowTemplate(
			metadata=analytics_metadata,
			workflow_definition={
				"name": "Sales Analytics Pipeline",
				"description": "Generate comprehensive sales analytics",
				"trigger": "scheduled"
			},
			components=[
				{
					"id": "extract_sales_data",
					"type": "database_query",
					"name": "Extract Sales Data",
					"config": {
						"query": """
						SELECT 
							DATE(created_at) as sale_date,
							sales_rep_id,
							customer_id,
							product_id,
							amount,
							quantity,
							region
						FROM sales 
						WHERE created_at >= :start_date 
						AND created_at <= :end_date
						""",
						"connection": "sales_db"
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "calculate_metrics",
					"type": "script",
					"name": "Calculate KPIs",
					"config": {
						"script_type": "python",
						"script_code": """
import pandas as pd
from datetime import datetime, timedelta

# Convert to DataFrame for easier analysis
df = pd.DataFrame(input_data)
df['sale_date'] = pd.to_datetime(df['sale_date'])

# Calculate key metrics
total_revenue = df['amount'].sum()
total_sales = len(df)
avg_deal_size = df['amount'].mean()
unique_customers = df['customer_id'].nunique()

# Sales by rep
sales_by_rep = df.groupby('sales_rep_id').agg({
	'amount': ['sum', 'count', 'mean'],
	'customer_id': 'nunique'
}).round(2)

# Sales by region
sales_by_region = df.groupby('region').agg({
	'amount': ['sum', 'count'],
	'customer_id': 'nunique'
}).round(2)

# Daily trends
daily_sales = df.groupby('sale_date').agg({
	'amount': 'sum',
	'customer_id': 'count'
}).round(2)

# Top products
top_products = df.groupby('product_id')['amount'].sum().sort_values(ascending=False).head(10)

result = {
	'summary_metrics': {
		'total_revenue': float(total_revenue),
		'total_sales': int(total_sales),
		'avg_deal_size': float(avg_deal_size),
		'unique_customers': int(unique_customers)
	},
	'sales_by_rep': sales_by_rep.to_dict(),
	'sales_by_region': sales_by_region.to_dict(),
	'daily_trends': daily_sales.to_dict(),
	'top_products': top_products.to_dict(),
	'analysis_date': datetime.now().isoformat()
}
						"""
					},
					"position": {"x": 350, "y": 100}
				},
				{
					"id": "generate_insights",
					"type": "script",
					"name": "Generate Insights",
					"config": {
						"script_type": "python",
						"script_code": """
metrics = input_data.get('summary_metrics', {})
sales_by_rep = input_data.get('sales_by_rep', {})
daily_trends = input_data.get('daily_trends', {})

insights = []

# Revenue insights
total_revenue = metrics.get('total_revenue', 0)
if total_revenue > 1000000:
	insights.append(f"Excellent performance: ${total_revenue:,.2f} in revenue")
elif total_revenue > 500000:
	insights.append(f"Good performance: ${total_revenue:,.2f} in revenue")
else:
	insights.append(f"Revenue below target: ${total_revenue:,.2f}")

# Deal size insights
avg_deal = metrics.get('avg_deal_size', 0)
if avg_deal > 10000:
	insights.append("High-value deals are performing well")
elif avg_deal < 1000:
	insights.append("Focus on increasing average deal size")

# Rep performance insights
if sales_by_rep:
	top_rep_revenue = max([rep.get('amount', {}).get('sum', 0) for rep in sales_by_rep.values()])
	if top_rep_revenue > total_revenue * 0.3:
		insights.append("Revenue heavily concentrated in top performer")

# Trend insights
if daily_trends:
	recent_sales = list(daily_trends.get('amount', {}).values())[-7:]  # Last 7 days
	if len(recent_sales) >= 2:
		if recent_sales[-1] > recent_sales[-2]:
			insights.append("Positive momentum in recent sales")
		else:
			insights.append("Sales momentum declining")

result = {
	'insights': insights,
	'recommendations': [
		"Continue monitoring daily trends",
		"Focus on rep development programs",
		"Analyze top-performing products for expansion"
	],
	'generated_at': datetime.now().isoformat()
}
						"""
					},
					"position": {"x": 600, "y": 100)
				},
				{
					"id": "create_report",
					"type": "script",
					"name": "Create HTML Report",
					"config": {
						"script_type": "python",
						"script_code": """
metrics = context.get('calculate_metrics_result', {})
insights = context.get('generate_insights_result', {})

html_report = f'''
<html>
<head>
	<title>Sales Analytics Report</title>
	<style>
		body {{ font-family: Arial, sans-serif; margin: 40px; }}
		.metric {{ background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 5px; }}
		.insight {{ background: #e3f2fd; padding: 15px; margin: 10px 0; border-left: 4px solid #2196f3; }}
		table {{ border-collapse: collapse; width: 100%; }}
		th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
		th {{ background-color: #f2f2f2; }}
	</style>
</head>
<body>
	<h1>Sales Analytics Report</h1>
	<h2>Summary Metrics</h2>
	<div class="metric">
		<h3>Total Revenue: ${metrics.get('summary_metrics', {}).get('total_revenue', 0):,.2f}</h3>
		<p>Total Sales: {metrics.get('summary_metrics', {}).get('total_sales', 0)}</p>
		<p>Average Deal Size: ${metrics.get('summary_metrics', {}).get('avg_deal_size', 0):,.2f}</p>
		<p>Unique Customers: {metrics.get('summary_metrics', {}).get('unique_customers', 0)}</p>
	</div>
	
	<h2>Key Insights</h2>
	{chr(10).join([f'<div class="insight">{insight}</div>' for insight in insights.get('insights', [])])}
	
	<h2>Recommendations</h2>
	<ul>
		{chr(10).join([f'<li>{rec}</li>' for rec in insights.get('recommendations', [])])}
	</ul>
	
	<p><small>Generated on: {insights.get('generated_at', '')}</small></p>
</body>
</html>
'''

result = {
	'html_report': html_report,
	'report_generated': True
}
						"""
					},
					"position": {"x": 850, "y": 100}
				},
				{
					"id": "email_report",
					"type": "email",
					"name": "Email Report",
					"config": {
						"default_to": "sales-team@example.com",
						"default_subject": "Daily Sales Analytics Report",
						"html": True
					},
					"position": {"x": 1100, "y": 100}
				}
			],
			connections=[
				{"from": "extract_sales_data", "to": "calculate_metrics"},
				{"from": "calculate_metrics", "to": "generate_insights"},
				{"from": "generate_insights", "to": "create_report"},
				{"from": "create_report", "to": "email_report"}
			],
			parameters={
				"start_date": {
					"type": "string",
					"description": "Analysis start date (YYYY-MM-DD)",
					"required": True
				},
				"end_date": {
					"type": "string",
					"description": "Analysis end date (YYYY-MM-DD)",
					"required": True
				},
				"report_recipients": {
					"type": "array",
					"description": "Email recipients for the report",
					"default": ["sales-manager@example.com"]
				}
			}
		)
		
		self.add_workflow_template(analytics_workflow)
	
	def _create_monitoring_templates(self):
		"""Create monitoring templates."""
		
		# System Health Check Template
		health_metadata = TemplateMetadata(
			id="system_health_monitoring",
			name="System Health Monitoring",
			description="Comprehensive system health checks with alerting",
			category=TemplateCategory.MONITORING,
			complexity=TemplateComplexity.INTERMEDIATE,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Infrastructure monitoring",
				"Service health checks",
				"Performance monitoring",
				"Uptime monitoring"
			],
			tags=["monitoring", "health check", "alerts", "infrastructure"],
			required_connectors=["http_request", "email"],
			estimated_runtime="2-10 minutes"
		)
		
		health_workflow = WorkflowTemplate(
			metadata=health_metadata,
			workflow_definition={
				"name": "System Health Monitoring",
				"description": "Monitor system health and send alerts",
				"trigger": "scheduled"
			},
			components=[
				{
					"id": "check_services",
					"type": "script",
					"name": "Check Service Health",
					"config": {
						"script_type": "python",
						"script_code": """
import requests
import time

services = input_data.get('services', [
	{'name': 'API Gateway', 'url': 'https://api.example.com/health'},
	{'name': 'Database', 'url': 'https://db.example.com/ping'},
	{'name': 'Redis Cache', 'url': 'https://cache.example.com/ping'}
])

health_results = []

for service in services:
	start_time = time.time()
	try:
		response = requests.get(service['url'], timeout=10)
		response_time = (time.time() - start_time) * 1000  # Convert to ms
		
		status = 'healthy' if response.status_code == 200 else 'unhealthy'
		health_results.append({
			'name': service['name'],
			'status': status,
			'response_time': round(response_time, 2),
			'status_code': response.status_code,
			'url': service['url']
		})
	except Exception as e:
		health_results.append({
			'name': service['name'],
			'status': 'error',
			'error': str(e),
			'url': service['url']
		})

# Calculate overall health
healthy_count = sum(1 for result in health_results if result['status'] == 'healthy')
overall_health = 'healthy' if healthy_count == len(health_results) else 'degraded'

result = {
	'services': health_results,
	'overall_health': overall_health,
	'healthy_services': healthy_count,
	'total_services': len(health_results),
	'check_timestamp': time.time()
}
						"""
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "check_system_metrics",
					"type": "script",
					"name": "Check System Metrics",
					"config": {
						"script_type": "python",
						"script_code": """
import psutil
import time

# Get system metrics
cpu_percent = psutil.cpu_percent(interval=1)
memory = psutil.virtual_memory()
disk = psutil.disk_usage('/')

# Define thresholds
cpu_warning = 80
cpu_critical = 95
memory_warning = 80
memory_critical = 95
disk_warning = 80
disk_critical = 95

# Determine status levels
cpu_status = 'critical' if cpu_percent > cpu_critical else ('warning' if cpu_percent > cpu_warning else 'ok')
memory_status = 'critical' if memory.percent > memory_critical else ('warning' if memory.percent > memory_warning else 'ok')
disk_status = 'critical' if disk.percent > disk_critical else ('warning' if disk.percent > disk_warning else 'ok')

result = {
	'cpu': {
		'percent': cpu_percent,
		'status': cpu_status
	},
	'memory': {
		'percent': memory.percent,
		'used_gb': round(memory.used / (1024**3), 2),
		'total_gb': round(memory.total / (1024**3), 2),
		'status': memory_status
	},
	'disk': {
		'percent': disk.percent,
		'used_gb': round(disk.used / (1024**3), 2),
		'total_gb': round(disk.total / (1024**3), 2),
		'status': disk_status
	},
	'overall_status': 'critical' if 'critical' in [cpu_status, memory_status, disk_status] 
					else ('warning' if 'warning' in [cpu_status, memory_status, disk_status] else 'ok'),
	'timestamp': time.time()
}
						"""
					},
					"position": {"x": 100, "y": 250}
				},
				{
					"id": "evaluate_alerts",
					"type": "script",
					"name": "Evaluate Alert Conditions",
					"config": {
						"script_type": "python",
						"script_code": """
service_health = context.get('check_services_result', {})
system_metrics = context.get('check_system_metrics_result', {})

alerts = []

# Service health alerts
if service_health.get('overall_health') == 'degraded':
	unhealthy_services = [s['name'] for s in service_health.get('services', []) if s['status'] != 'healthy']
	alerts.append({
		'level': 'critical',
		'type': 'service_health',
		'message': f"Services unhealthy: {', '.join(unhealthy_services)}",
		'affected_services': unhealthy_services
	})

# System metrics alerts
system_status = system_metrics.get('overall_status', 'ok')
if system_status in ['warning', 'critical']:
	alerts.append({
		'level': system_status,
		'type': 'system_metrics',
		'message': f"System resources at {system_status} level",
		'cpu_percent': system_metrics.get('cpu', {}).get('percent'),
		'memory_percent': system_metrics.get('memory', {}).get('percent'),
		'disk_percent': system_metrics.get('disk', {}).get('percent')
	})

# Response time alerts
slow_services = [s for s in service_health.get('services', []) 
				if s.get('response_time', 0) > 5000]  # > 5 seconds
if slow_services:
	alerts.append({
		'level': 'warning',
		'type': 'performance',
		'message': f"Slow response times detected",
		'slow_services': [(s['name'], s['response_time']) for s in slow_services]
	})

result = {
	'alerts': alerts,
	'alert_count': len(alerts),
	'requires_notification': len(alerts) > 0,
	'highest_severity': 'critical' if any(a['level'] == 'critical' for a in alerts) 
					   else ('warning' if any(a['level'] == 'warning' for a in alerts) else 'ok')
}
						"""
					},
					"position": {"x": 400, "y": 175}
				},
				{
					"id": "send_alert",
					"type": "email",
					"name": "Send Alert Email",
					"config": {
						"default_to": "ops-team@example.com",
						"default_subject": "System Health Alert - {{highest_severity}}",
						"condition": "requires_notification == true"
					},
					"position": {"x": 700, "y": 175}
				},
				{
					"id": "log_metrics",
					"type": "database_query",
					"name": "Log Health Metrics",
					"config": {
						"query": """
						INSERT INTO health_metrics 
						(timestamp, overall_health, cpu_percent, memory_percent, disk_percent, alert_count)
						VALUES (:timestamp, :overall_health, :cpu_percent, :memory_percent, :disk_percent, :alert_count)
						""",
						"connection": "monitoring_db"
					},
					"position": {"x": 700, "y": 300}
				}
			],
			connections=[
				{"from": "check_services", "to": "evaluate_alerts"},
				{"from": "check_system_metrics", "to": "evaluate_alerts"},
				{"from": "evaluate_alerts", "to": "send_alert", "condition": "requires_notification == true"},
				{"from": "evaluate_alerts", "to": "log_metrics"}
			],
			parameters={
				"services_to_check": {
					"type": "array",
					"description": "List of services to monitor",
					"default": [
						{"name": "API Gateway", "url": "https://api.example.com/health"},
						{"name": "Database", "url": "https://db.example.com/ping"}
					]
				},
				"alert_recipients": {
					"type": "array",
					"description": "Email recipients for alerts",
					"default": ["ops-team@example.com"]
				}
			}
		)
		
		self.add_workflow_template(health_workflow)
	
	def _create_ml_templates(self):
		"""Create machine learning templates."""
		
		# ML Model Training Pipeline
		ml_metadata = TemplateMetadata(
			id="ml_training_pipeline",
			name="ML Model Training Pipeline",
			description="End-to-end machine learning model training and deployment pipeline",
			category=TemplateCategory.MACHINE_LEARNING,
			complexity=TemplateComplexity.EXPERT,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Model training automation",
				"ML pipeline deployment",
				"Model validation",
				"Automated retraining"
			],
			tags=["machine learning", "training", "pipeline", "automation"],
			required_connectors=["database", "file_system"],
			estimated_runtime="30-180 minutes"
		)
		
		ml_workflow = WorkflowTemplate(
			metadata=ml_metadata,
			workflow_definition={
				"name": "ML Model Training Pipeline",
				"description": "Train and deploy ML models automatically",
				"trigger": "scheduled"
			},
			components=[
				{
					"id": "extract_training_data",
					"type": "database_query",
					"name": "Extract Training Data",
					"config": {
						"query": """
						SELECT 
							feature1, feature2, feature3, feature4, feature5,
							target_variable,
							data_quality_score
						FROM training_data 
						WHERE created_at >= :start_date 
						AND data_quality_score > 0.8
						ORDER BY created_at
						""",
						"connection": "ml_data_db"
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "data_preprocessing",
					"type": "script",
					"name": "Preprocess Data",
					"config": {
						"script_type": "python",
						"script_code": """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Convert to DataFrame
df = pd.DataFrame(input_data)

# Handle missing values
df = df.dropna()

# Separate features and target
feature_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
X = df[feature_columns]
y = df['target_variable']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
	X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

result = {
	'X_train': X_train.tolist(),
	'X_test': X_test.tolist(), 
	'y_train': y_train.tolist(),
	'y_test': y_test.tolist(),
	'feature_columns': feature_columns,
	'scaler_params': {
		'mean': scaler.mean_.tolist(),
		'scale': scaler.scale_.tolist()
	},
	'train_size': len(X_train),
	'test_size': len(X_test)
}
						"""
					},
					"position": {"x": 350, "y": 100}
				},
				{
					"id": "train_model",
					"type": "script",
					"name": "Train ML Model",
					"config": {
						"script_type": "python",
						"script_code": """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Get preprocessed data
X_train = np.array(input_data['X_train'])
X_test = np.array(input_data['X_test'])
y_train = np.array(input_data['y_train'])
y_test = np.array(input_data['y_test'])

# Train Random Forest model
model = RandomForestClassifier(
	n_estimators=100,
	max_depth=10,
	random_state=42,
	n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
feature_importance = model.feature_importances_

# Generate classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

result = {
	'model_trained': True,
	'accuracy': float(accuracy),
	'feature_importance': feature_importance.tolist(),
	'classification_report': class_report,
	'predictions': y_pred.tolist(),
	'prediction_probabilities': y_pred_proba.tolist(),
	'model_params': model.get_params(),
	'training_completed': True
}

# Save model (in production, save to persistent storage)
# joblib.dump(model, 'trained_model.pkl')
						"""
					},
					"position": {"x": 600, "y": 100}
				},
				{
					"id": "evaluate_model",
					"type": "script",
					"name": "Evaluate Model Performance",
					"config": {
						"script_type": "python",
						"script_code": """
accuracy = input_data.get('accuracy', 0)
classification_report = input_data.get('classification_report', {})

# Define performance thresholds
accuracy_threshold = 0.85
precision_threshold = 0.80
recall_threshold = 0.80

# Get weighted averages
weighted_avg = classification_report.get('weighted avg', {})
precision = weighted_avg.get('precision', 0)
recall = weighted_avg.get('recall', 0)
f1_score = weighted_avg.get('f1-score', 0)

# Determine if model meets criteria
meets_accuracy = accuracy >= accuracy_threshold
meets_precision = precision >= precision_threshold  
meets_recall = recall >= recall_threshold

model_approved = meets_accuracy and meets_precision and meets_recall

# Generate evaluation summary
evaluation = {
	'model_approved': model_approved,
	'performance_metrics': {
		'accuracy': accuracy,
		'precision': precision,
		'recall': recall,
		'f1_score': f1_score
	},
	'threshold_checks': {
		'accuracy_check': meets_accuracy,
		'precision_check': meets_precision,
		'recall_check': meets_recall
	},
	'evaluation_timestamp': datetime.now().isoformat()
}

if not model_approved:
	evaluation['rejection_reasons'] = []
	if not meets_accuracy:
		evaluation['rejection_reasons'].append(f'Accuracy {accuracy:.3f} below threshold {accuracy_threshold}')
	if not meets_precision:
		evaluation['rejection_reasons'].append(f'Precision {precision:.3f} below threshold {precision_threshold}')
	if not meets_recall:
		evaluation['rejection_reasons'].append(f'Recall {recall:.3f} below threshold {recall_threshold}')

result = evaluation
						"""
					},
					"position": {"x": 850, "y": 100}
				},
				{
					"id": "deploy_model",
					"type": "script",
					"name": "Deploy Model",
					"config": {
						"script_type": "python",
						"script_code": """
if input_data.get('model_approved', False):
	# Simulate model deployment
	deployment_result = {
		'deployed': True,
		'deployment_id': f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
		'endpoint_url': 'https://api.example.com/ml/predict',
		'deployment_timestamp': datetime.now().isoformat(),
		'status': 'active'
	}
else:
	deployment_result = {
		'deployed': False,
		'reason': 'Model did not meet performance criteria',
		'rejection_reasons': input_data.get('rejection_reasons', [])
	}

result = deployment_result
						"""
					},
					"position": {"x": 1100, "y": 100}
				},
				{
					"id": "send_training_report",
					"type": "email",
					"name": "Send Training Report",
					"config": {
						"default_to": "ml-team@example.com",
						"default_subject": "ML Model Training Report - {{model_approved}}",
						"html": True
					},
					"position": {"x": 850, "y": 250}
				}
			],
			connections=[
				{"from": "extract_training_data", "to": "data_preprocessing"},
				{"from": "data_preprocessing", "to": "train_model"},
				{"from": "train_model", "to": "evaluate_model"},
				{"from": "evaluate_model", "to": "deploy_model", "condition": "model_approved == true"},
				{"from": "evaluate_model", "to": "send_training_report"},
				{"from": "deploy_model", "to": "send_training_report"}
			],
			parameters={
				"training_data_days": {
					"type": "integer",
					"description": "Days of training data to use",
					"default": 30
				},
				"model_type": {
					"type": "string",
					"description": "Type of ML model to train",
					"default": "random_forest",
					"enum": ["random_forest", "gradient_boosting", "neural_network"]
				},
				"accuracy_threshold": {
					"type": "number",
					"description": "Minimum accuracy for model approval",
					"default": 0.85
				}
			}
		)
		
		self.add_workflow_template(ml_workflow)
	
	def _create_devops_templates(self):
		"""Create DevOps templates."""
		
		# CI/CD Pipeline Template
		cicd_metadata = TemplateMetadata(
			id="cicd_deployment_pipeline",
			name="CI/CD Deployment Pipeline",
			description="Continuous integration and deployment pipeline with testing and rollback",
			category=TemplateCategory.DEVOPS,
			complexity=TemplateComplexity.ADVANCED,
			template_type=TemplateType.WORKFLOW,
			use_cases=[
				"Application deployment",
				"Infrastructure deployment",
				"Automated testing",
				"Release management"
			],
			tags=["CI/CD", "deployment", "testing", "automation"],
			required_connectors=["http_request", "file_system"],
			estimated_runtime="10-60 minutes"
		)
		
		cicd_workflow = WorkflowTemplate(
			metadata=cicd_metadata,
			workflow_definition={
				"name": "CI/CD Deployment Pipeline",
				"description": "Automated build, test, and deployment pipeline",
				"trigger": "webhook"
			},
			components=[
				{
					"id": "trigger_build",
					"type": "start",
					"name": "Deployment Triggered",
					"config": {
						"trigger_type": "webhook",
						"webhook_path": "/deploy"
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "run_tests",
					"type": "script",
					"name": "Run Test Suite",
					"config": {
						"script_type": "python",
						"script_code": """
import subprocess
import time

# Simulate running test suite
test_commands = [
	"pytest tests/unit/",
	"pytest tests/integration/", 
	"npm run test:e2e"
]

test_results = []
overall_success = True

for cmd in test_commands:
	start_time = time.time()
	try:
		# In real implementation, actually run the command
		# result = subprocess.run(cmd.split(), capture_output=True, text=True)
		
		# Simulate test results
		duration = time.time() - start_time + 2  # Add 2 seconds simulation
		test_results.append({
			'command': cmd,
			'status': 'passed',
			'duration': round(duration, 2),
			'output': f'All tests passed for {cmd}'
		})
	except Exception as e:
		overall_success = False
		test_results.append({
			'command': cmd,
			'status': 'failed',
			'error': str(e),
			'duration': time.time() - start_time
		})

result = {
	'tests_passed': overall_success,
	'test_results': test_results,
	'total_tests': len(test_commands),
	'passed_tests': sum(1 for t in test_results if t['status'] == 'passed'),
	'test_duration': sum(t['duration'] for t in test_results)
}
						"""
					},
					"position": {"x": 300, "y": 100}
				},
				{
					"id": "build_application",
					"type": "script",
					"name": "Build Application",
					"config": {
						"script_type": "python",
						"script_code": """
import time

build_start = time.time()

# Simulate build process
build_steps = [
	"Installing dependencies",
	"Compiling source code", 
	"Running linting",
	"Creating build artifacts",
	"Generating documentation"
]

build_log = []
for step in build_steps:
	build_log.append(f"[{time.strftime('%H:%M:%S')}] {step}")
	time.sleep(0.1)  # Simulate build time

build_duration = time.time() - build_start

result = {
	'build_successful': True,
	'build_duration': round(build_duration, 2),
	'build_log': build_log,
	'artifacts': [
		'app-v1.2.3.tar.gz',
		'docker-image:latest',
		'deployment.yaml'
	],
	'build_id': f'build_{int(time.time())}'
}
						"""
					},
					"position": {"x": 500, "y": 100}
				},
				{
					"id": "deploy_staging",
					"type": "http_request",
					"name": "Deploy to Staging",
					"config": {
						"method": "POST",
						"url": "{{staging_deploy_url}}",
						"headers": {
							"Authorization": "Bearer {{deploy_token}}",
							"Content-Type": "application/json"
						},
						"timeout": 300
					},
					"position": {"x": 700, "y": 100}
				},
				{
					"id": "staging_smoke_tests",
					"type": "script",
					"name": "Staging Smoke Tests",
					"config": {
						"script_type": "python",
						"script_code": """
import requests
import time

staging_url = context.get('staging_url', 'https://staging.example.com')

smoke_tests = [
	{'name': 'Health Check', 'url': f'{staging_url}/health'},
	{'name': 'API Status', 'url': f'{staging_url}/api/status'},
	{'name': 'Database Connection', 'url': f'{staging_url}/api/db/ping'}
]

test_results = []
all_passed = True

for test in smoke_tests:
	try:
		response = requests.get(test['url'], timeout=10)
		passed = response.status_code == 200
		test_results.append({
			'name': test['name'],
			'url': test['url'],
			'status': 'passed' if passed else 'failed',
			'response_code': response.status_code,
			'response_time': response.elapsed.total_seconds()
		})
		if not passed:
			all_passed = False
	except Exception as e:
		all_passed = False
		test_results.append({
			'name': test['name'],
			'url': test['url'],
			'status': 'error',
			'error': str(e)
		})

result = {
	'smoke_tests_passed': all_passed,
	'test_results': test_results,
	'ready_for_production': all_passed
}
						"""
					},
					"position": {"x": 900, "y": 100}
				},
				{
					"id": "deploy_production",
					"type": "http_request",
					"name": "Deploy to Production",
					"config": {
						"method": "POST",
						"url": "{{production_deploy_url}}",
						"headers": {
							"Authorization": "Bearer {{deploy_token}}",
							"Content-Type": "application/json"
						},
						"timeout": 600,
						"condition": "ready_for_production == true"
					},
					"position": {"x": 1100, "y": 100}
				},
				{
					"id": "rollback_deployment",
					"type": "http_request",
					"name": "Rollback Deployment",
					"config": {
						"method": "POST",
						"url": "{{rollback_url}}",
						"headers": {
							"Authorization": "Bearer {{deploy_token}}",
							"Content-Type": "application/json"
						}
					},
					"position": {"x": 900, "y": 300}
				},
				{
					"id": "notify_team",
					"type": "email",
					"name": "Notify Deployment Result",
					"config": {
						"default_to": "dev-team@example.com",
						"default_subject": "Deployment {{deployment_status}} - Build {{build_id}}"
					},
					"position": {"x": 1300, "y": 150}
				}
			],
			connections=[
				{"from": "trigger_build", "to": "run_tests"},
				{"from": "run_tests", "to": "build_application", "condition": "tests_passed == true"},
				{"from": "build_application", "to": "deploy_staging", "condition": "build_successful == true"},
				{"from": "deploy_staging", "to": "staging_smoke_tests"},
				{"from": "staging_smoke_tests", "to": "deploy_production", "condition": "ready_for_production == true"},
				{"from": "staging_smoke_tests", "to": "rollback_deployment", "condition": "ready_for_production == false"},
				{"from": "deploy_production", "to": "notify_team"},
				{"from": "rollback_deployment", "to": "notify_team"},
				{"from": "run_tests", "to": "notify_team", "condition": "tests_passed == false"},
				{"from": "build_application", "to": "notify_team", "condition": "build_successful == false"}
			],
			parameters={
				"staging_deploy_url": {
					"type": "string",
					"description": "Staging deployment endpoint",
					"required": True
				},
				"production_deploy_url": {
					"type": "string",
					"description": "Production deployment endpoint",
					"required": True
				},
				"rollback_url": {
					"type": "string",
					"description": "Rollback endpoint",
					"required": True
				},
				"deploy_token": {
					"type": "string",
					"description": "Deployment authentication token",
					"required": True
				}
			}
		)
		
		self.add_workflow_template(cicd_workflow)
	
	def _create_industry_templates(self):
		"""Create industry-specific templates."""
		
		# E-commerce Order Processing
		ecommerce_metadata = TemplateMetadata(
			id="ecommerce_order_processing",
			name="E-commerce Order Processing",
			description="Complete order processing workflow with inventory, payment, and fulfillment",
			category=TemplateCategory.BUSINESS_PROCESS,
			complexity=TemplateComplexity.ADVANCED,
			template_type=TemplateType.WORKFLOW,
			industry="E-commerce",
			use_cases=[
				"Order fulfillment",
				"Inventory management",
				"Payment processing",
				"Customer communication"
			],
			tags=["e-commerce", "orders", "inventory", "payment", "fulfillment"],
			required_connectors=["database", "email", "http_request"],
			estimated_runtime="5-15 minutes"
		)
		
		ecommerce_workflow = WorkflowTemplate(
			metadata=ecommerce_metadata,
			workflow_definition={
				"name": "E-commerce Order Processing",
				"description": "Process e-commerce orders end-to-end",
				"trigger": "webhook"
			},
			components=[
				{
					"id": "order_received",
					"type": "start",
					"name": "Order Received",
					"config": {
						"trigger_type": "webhook",
						"webhook_path": "/orders/new"
					},
					"position": {"x": 100, "y": 100}
				},
				{
					"id": "validate_order",
					"type": "script",
					"name": "Validate Order",
					"config": {
						"script_type": "python",
						"script_code": """
order = input_data

# Validation checks
validation_errors = []

# Required fields
required_fields = ['customer_id', 'items', 'shipping_address', 'payment_method']
for field in required_fields:
	if not order.get(field):
		validation_errors.append(f'Missing required field: {field}')

# Validate items
items = order.get('items', [])
if not items:
	validation_errors.append('Order must contain at least one item')
else:
	for i, item in enumerate(items):
		if not item.get('product_id'):
			validation_errors.append(f'Item {i+1}: Missing product_id')
		if not item.get('quantity') or item['quantity'] <= 0:
			validation_errors.append(f'Item {i+1}: Invalid quantity')

# Validate payment method
payment_method = order.get('payment_method', {})
if not payment_method.get('type'):
	validation_errors.append('Payment method type is required')

result = {
	'order_valid': len(validation_errors) == 0,
	'validation_errors': validation_errors,
	'order_data': order
}
						"""
					},
					"position": {"x": 300, "y": 100}
				},
				{
					"id": "check_inventory",
					"type": "database_query",
					"name": "Check Inventory",
					"config": {
						"query": """
						SELECT 
							product_id,
							available_quantity,
							reserved_quantity,
							price
						FROM inventory 
						WHERE product_id IN :product_ids
						""",
						"connection": "inventory_db"
					},
					"position": {"x": 500, "y": 100}
				},
				{
					"id": "reserve_inventory",
					"type": "script",
					"name": "Reserve Items",
					"config": {
						"script_type": "python",
						"script_code": """
order_items = context.get('validate_order_result', {}).get('order_data', {}).get('items', [])
inventory_data = input_data

# Create inventory lookup
inventory_map = {item['product_id']: item for item in inventory_data}

reservation_results = []
can_fulfill = True
total_amount = 0

for order_item in order_items:
	product_id = order_item['product_id']
	requested_qty = order_item['quantity']
	
	inventory_item = inventory_map.get(product_id)
	if not inventory_item:
		can_fulfill = False
		reservation_results.append({
			'product_id': product_id,
			'requested': requested_qty,
			'reserved': 0,
			'status': 'not_found'
		})
		continue
	
	available = inventory_item['available_quantity']
	if available >= requested_qty:
		reservation_results.append({
			'product_id': product_id,
			'requested': requested_qty,
			'reserved': requested_qty,
			'status': 'reserved',
			'unit_price': inventory_item['price']
		})
		total_amount += requested_qty * inventory_item['price']
	else:
		can_fulfill = False
		reservation_results.append({
			'product_id': product_id,
			'requested': requested_qty,
			'reserved': 0,
			'available': available,
			'status': 'insufficient_stock'
		})

result = {
	'can_fulfill_order': can_fulfill,
	'reservations': reservation_results,
	'total_amount': round(total_amount, 2),
	'fulfillment_status': 'confirmed' if can_fulfill else 'backordered'
}
						"""
					},
					"position": {"x": 700, "y": 100}
				},
				{
					"id": "process_payment",
					"type": "http_request",
					"name": "Process Payment",
					"config": {
						"method": "POST",
						"url": "{{payment_gateway_url}}/charges",
						"headers": {
							"Authorization": "Bearer {{payment_api_key}}",
							"Content-Type": "application/json"
						},
						"condition": "can_fulfill_order == true"
					},
					"position": {"x": 900, "y": 100}
				},
				{
					"id": "create_shipment",
					"type": "script",
					"name": "Create Shipment",
					"config": {
						"script_type": "python",
						"script_code": """
order_data = context.get('validate_order_result', {}).get('order_data', {})
payment_result = input_data

if payment_result.get('status') == 'succeeded':
	shipment = {
		'order_id': order_data.get('order_id'),
		'customer_id': order_data.get('customer_id'),
		'shipping_address': order_data.get('shipping_address'),
		'items': order_data.get('items'),
		'tracking_number': f"TRK{int(time.time())}",
		'status': 'preparing',
		'estimated_delivery': (datetime.now() + timedelta(days=3)).isoformat(),
		'created_at': datetime.now().isoformat()
	}
	
	result = {
		'shipment_created': True,
		'shipment_data': shipment,
		'tracking_number': shipment['tracking_number']
	}
else:
	result = {
		'shipment_created': False,
		'error': 'Payment failed'
	}
						"""
					},
					"position": {"x": 1100, "y": 100}
				},
				{
					"id": "send_confirmation",
					"type": "email",
					"name": "Order Confirmation",
					"config": {
						"default_subject": "Order Confirmation - {{order_id}}",
						"html": True
					},
					"position": {"x": 1300, "y": 100}
				},
				{
					"id": "backorder_notification",
					"type": "email",
					"name": "Backorder Notification",
					"config": {
						"default_subject": "Order Backordered - {{order_id}}",
						"condition": "can_fulfill_order == false"
					},
					"position": {"x": 700, "y": 300}
				}
			],
			connections=[
				{"from": "order_received", "to": "validate_order"},
				{"from": "validate_order", "to": "check_inventory", "condition": "order_valid == true"},
				{"from": "check_inventory", "to": "reserve_inventory"},
				{"from": "reserve_inventory", "to": "process_payment", "condition": "can_fulfill_order == true"},
				{"from": "reserve_inventory", "to": "backorder_notification", "condition": "can_fulfill_order == false"},
				{"from": "process_payment", "to": "create_shipment"},
				{"from": "create_shipment", "to": "send_confirmation"}
			],
			parameters={
				"payment_gateway_url": {
					"type": "string",
					"description": "Payment gateway API endpoint",
					"required": True
				},
				"payment_api_key": {
					"type": "string",
					"description": "Payment gateway API key",
					"required": True
				},
				"inventory_threshold": {
					"type": "integer",
					"description": "Minimum inventory level for auto-fulfillment",
					"default": 1
				}
			}
		)
		
		self.add_workflow_template(ecommerce_workflow)
	
	def add_workflow_template(self, template: WorkflowTemplate):
		"""Add a workflow template to the library."""
		self.workflow_templates[template.metadata.id] = template
		
		# Update category index
		category = template.metadata.category
		if category not in self.categories:
			self.categories[category] = []
		if template.metadata.id not in self.categories[category]:
			self.categories[category].append(template.metadata.id)
		
		# Update tag index
		for tag in template.metadata.tags:
			if tag not in self.tag_index:
				self.tag_index[tag] = set()
			self.tag_index[tag].add(template.metadata.id)
	
	def add_component_set(self, component_set: ComponentSet):
		"""Add a component set to the library."""
		self.component_sets[component_set.metadata.id] = component_set
		
		# Update category index
		category = component_set.metadata.category
		if category not in self.categories:
			self.categories[category] = []
		if component_set.metadata.id not in self.categories[category]:
			self.categories[category].append(component_set.metadata.id)
		
		# Update tag index
		for tag in component_set.metadata.tags:
			if tag not in self.tag_index:
				self.tag_index[tag] = set()
			self.tag_index[tag].add(component_set.metadata.id)
	
	def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
		"""Get a workflow template by ID."""
		return self.workflow_templates.get(template_id)
	
	def get_component_set(self, set_id: str) -> Optional[ComponentSet]:
		"""Get a component set by ID."""
		return self.component_sets.get(set_id)
	
	def search_templates(self, query: str = None, category: TemplateCategory = None,
						complexity: TemplateComplexity = None, tags: List[str] = None,
						template_type: TemplateType = None) -> List[WorkflowTemplate]:
		"""Search templates with filters."""
		results = list(self.workflow_templates.values())
		
		# Filter by category
		if category:
			results = [t for t in results if t.metadata.category == category]
		
		# Filter by complexity
		if complexity:
			results = [t for t in results if t.metadata.complexity == complexity]
		
		# Filter by template type
		if template_type:
			results = [t for t in results if t.metadata.template_type == template_type]
		
		# Filter by tags
		if tags:
			results = [t for t in results if any(tag in t.metadata.tags for tag in tags)]
		
		# Filter by query (search in name, description, use cases)
		if query:
			query_lower = query.lower()
			results = [t for t in results if (
				query_lower in t.metadata.name.lower() or
				query_lower in t.metadata.description.lower() or
				any(query_lower in use_case.lower() for use_case in t.metadata.use_cases)
			)]
		
		# Sort by popularity score (descending)
		results.sort(key=lambda t: t.metadata.popularity_score, reverse=True)
		
		return results
	
	def get_popular_templates(self, limit: int = 10) -> List[WorkflowTemplate]:
		"""Get most popular templates."""
		templates = list(self.workflow_templates.values())
		templates.sort(key=lambda t: t.metadata.popularity_score, reverse=True)
		return templates[:limit]
	
	def get_templates_by_category(self, category: TemplateCategory) -> List[WorkflowTemplate]:
		"""Get all templates in a category."""
		template_ids = self.categories.get(category, [])
		return [self.workflow_templates[tid] for tid in template_ids if tid in self.workflow_templates]
	
	def get_templates_by_tags(self, tags: List[str]) -> List[WorkflowTemplate]:
		"""Get templates matching any of the provided tags."""
		matching_ids = set()
		for tag in tags:
			if tag in self.tag_index:
				matching_ids.update(self.tag_index[tag])
		
		return [self.workflow_templates[tid] for tid in matching_ids if tid in self.workflow_templates]
	
	def get_all_categories(self) -> List[TemplateCategory]:
		"""Get all available categories."""
		return list(self.categories.keys())
	
	def get_all_tags(self) -> List[str]:
		"""Get all available tags."""
		return list(self.tag_index.keys())
	
	def export_template(self, template_id: str, format: str = "json") -> Optional[str]:
		"""Export template in specified format."""
		template = self.get_template(template_id)
		if not template:
			return None
		
		if format == "json":
			import json
			template_dict = {
				"metadata": {
					"id": template.metadata.id,
					"name": template.metadata.name,
					"description": template.metadata.description,
					"category": template.metadata.category.value,
					"complexity": template.metadata.complexity.value,
					"template_type": template.metadata.template_type.value,
					"version": template.metadata.version,
					"author": template.metadata.author,
					"use_cases": template.metadata.use_cases,
					"tags": template.metadata.tags,
					"industry": template.metadata.industry,
					"required_capabilities": template.metadata.required_capabilities,
					"required_connectors": template.metadata.required_connectors,
					"estimated_runtime": template.metadata.estimated_runtime
				},
				"workflow_definition": template.workflow_definition,
				"components": template.components,
				"connections": template.connections,
				"parameters": template.parameters,
				"configuration": template.configuration
			}
			return json.dumps(template_dict, indent=2)
		
		elif format == "yaml":
			import yaml
			template_dict = {
				"metadata": {
					"id": template.metadata.id,
					"name": template.metadata.name,
					"description": template.metadata.description,
					"category": template.metadata.category.value,
					"complexity": template.metadata.complexity.value,
					"template_type": template.metadata.template_type.value,
					"version": template.metadata.version,
					"author": template.metadata.author,
					"use_cases": template.metadata.use_cases,
					"tags": template.metadata.tags,
					"industry": template.metadata.industry,
					"required_capabilities": template.metadata.required_capabilities,
					"required_connectors": template.metadata.required_connectors,
					"estimated_runtime": template.metadata.estimated_runtime
				},
				"workflow_definition": template.workflow_definition,
				"components": template.components,
				"connections": template.connections,
				"parameters": template.parameters,
				"configuration": template.configuration
			}
			return yaml.dump(template_dict, default_flow_style=False, indent=2)
		
		return None


# Template Library Service

class TemplateLibraryService(APGBaseService):
	"""Service for managing workflow template library."""
	
	def __init__(self):
		super().__init__()
		self.library = TemplateLibrary()
		self.database = APGDatabase()
		self.audit = APGAuditLogger()
	
	async def start(self):
		"""Start template library service."""
		await super().start()
		await self._load_custom_templates()
		logger.info("Template library service started")
	
	async def get_templates(self, category: TemplateCategory = None, 
						  complexity: TemplateComplexity = None,
						  tags: List[str] = None, 
						  query: str = None) -> List[Dict[str, Any]]:
		"""Get templates with filtering."""
		try:
			templates = self.library.search_templates(
				query=query,
				category=category,
				complexity=complexity,
				tags=tags
			)
			
			return [self._template_to_dict(template) for template in templates]
			
		except Exception as e:
			logger.error(f"Failed to get templates: {e}")
			return []
	
	async def get_template_by_id(self, template_id: str) -> Optional[Dict[str, Any]]:
		"""Get specific template by ID."""
		try:
			template = self.library.get_template(template_id)
			if template:
				return self._template_to_dict(template)
			return None
			
		except Exception as e:
			logger.error(f"Failed to get template {template_id}: {e}")
			return None
	
	async def create_workflow_from_template(self, template_id: str, 
										   parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
		"""Create a workflow instance from a template."""
		try:
			template = self.library.get_template(template_id)
			if not template:
				raise ValueError(f"Template not found: {template_id}")
			
			# Create workflow instance with parameter substitution
			workflow_instance = self._instantiate_template(template, parameters or {})
			
			# Update usage statistics
			template.metadata.usage_count += 1
			
			# Log template usage
			await self.audit.log_event({
				'event_type': 'template_used',
				'template_id': template_id,
				'template_name': template.metadata.name,
				'parameters_provided': bool(parameters)
			})
			
			return workflow_instance
			
		except Exception as e:
			logger.error(f"Failed to create workflow from template {template_id}: {e}")
			return None
	
	async def export_template(self, template_id: str, format: str = "json") -> Optional[str]:
		"""Export template in specified format."""
		try:
			return self.library.export_template(template_id, format)
		except Exception as e:
			logger.error(f"Failed to export template {template_id}: {e}")
			return None
	
	async def get_categories(self) -> List[str]:
		"""Get all available template categories."""
		return [cat.value for cat in self.library.get_all_categories()]
	
	async def get_tags(self) -> List[str]:
		"""Get all available template tags."""
		return self.library.get_all_tags()
	
	async def get_popular_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
		"""Get most popular templates."""
		try:
			templates = self.library.get_popular_templates(limit)
			return [self._template_to_dict(template) for template in templates]
		except Exception as e:
			logger.error(f"Failed to get popular templates: {e}")
			return []
	
	def _template_to_dict(self, template: WorkflowTemplate) -> Dict[str, Any]:
		"""Convert template to dictionary."""
		return {
			'id': template.metadata.id,
			'name': template.metadata.name,
			'description': template.metadata.description,
			'category': template.metadata.category.value,
			'complexity': template.metadata.complexity.value,
			'template_type': template.metadata.template_type.value,
			'version': template.metadata.version,
			'author': template.metadata.author,
			'use_cases': template.metadata.use_cases,
			'tags': template.metadata.tags,
			'industry': template.metadata.industry,
			'required_capabilities': template.metadata.required_capabilities,
			'required_connectors': template.metadata.required_connectors,
			'estimated_runtime': template.metadata.estimated_runtime,
			'popularity_score': template.metadata.popularity_score,
			'usage_count': template.metadata.usage_count,
			'rating': template.metadata.rating,
			'rating_count': template.metadata.rating_count,
			'customizable_params': template.metadata.customizable_params,
			'preview_image': template.metadata.preview_image,
			'created_at': template.metadata.created_at.isoformat(),
			'updated_at': template.metadata.updated_at.isoformat(),
			'parameters': template.parameters
		}
	
	def _instantiate_template(self, template: WorkflowTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Create workflow instance from template with parameter substitution."""
		import re
		
		def replace_parameters(obj, params):
			"""Recursively replace parameter placeholders."""
			if isinstance(obj, str):
				# Replace {{param_name}} with actual values
				for param_name, param_value in params.items():
					placeholder = f"{{{{{param_name}}}}}"
					obj = obj.replace(placeholder, str(param_value))
				return obj
			elif isinstance(obj, dict):
				return {k: replace_parameters(v, params) for k, v in obj.items()}
			elif isinstance(obj, list):
				return [replace_parameters(item, params) for item in obj]
			else:
				return obj
		
		# Create base workflow instance
		workflow_instance = {
			'id': uuid7str(),
			'name': f"{template.metadata.name} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
			'description': template.workflow_definition.get('description', ''),
			'template_id': template.metadata.id,
			'template_version': template.metadata.version,
			'status': 'draft',
			'created_at': datetime.utcnow().isoformat(),
			'components': replace_parameters(template.components, parameters),
			'connections': replace_parameters(template.connections, parameters),
			'configuration': replace_parameters(template.configuration, parameters),
			'parameters': parameters
		}
		
		return workflow_instance
	
	async def _load_custom_templates(self):
		"""Load custom templates from database."""
		try:
			# In a real implementation, this would load from database
			logger.info("Loading custom templates...")
		except Exception as e:
			logger.error(f"Failed to load custom templates: {e}")
	
	async def health_check(self) -> bool:
		"""Health check for template library service."""
		try:
			# Check if templates are available
			templates = await self.get_templates()
			return len(templates) > 0
		except Exception:
			return False


# Global template library instance
template_library = TemplateLibrary()
template_library_service = TemplateLibraryService()