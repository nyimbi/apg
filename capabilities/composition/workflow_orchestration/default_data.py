#!/usr/bin/env python3
"""
APG Workflow Orchestration Default Data

Default configuration data, sample workflows, and system setup.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .models import WorkflowStatus, TaskStatus
from .database import WorkflowDB, WorkflowInstanceDB, ConnectorDB
from .templates_library import WorkflowTemplate, TemplateCategory, TemplateTags


logger = logging.getLogger(__name__)


def get_default_system_workflows() -> List[Dict[str, Any]]:
	"""Get default system workflows."""
	return [
		{
			"id": "system_health_check_001",
			"name": "System Health Check",
			"description": "Comprehensive system health monitoring workflow",
			"definition": {
				"version": "1.0.0",
				"metadata": {
					"author": "APG System",
					"description": "Monitors system health and performance",
					"tags": ["system", "monitoring", "health"]
				},
				"variables": {
					"check_interval": {"type": "integer", "default": 300},
					"alert_threshold": {"type": "number", "default": 0.8}
				},
				"tasks": [
					{
						"id": "check_database",
						"name": "Database Health Check",
						"type": "health_check",
						"description": "Verify database connectivity and performance",
						"config": {
							"target": "database",
							"timeout": 30,
							"checks": ["connectivity", "response_time", "disk_space"]
						},
						"next_tasks": ["check_redis"]
					},
					{
						"id": "check_redis",
						"name": "Redis Health Check",
						"type": "health_check",
						"description": "Verify Redis connectivity and memory usage",
						"config": {
							"target": "redis",
							"timeout": 15,
							"checks": ["connectivity", "memory_usage", "keys_count"]
						},
						"next_tasks": ["check_services"]
					},
					{
						"id": "check_services",
						"name": "Services Health Check",
						"type": "health_check",
						"description": "Verify all workflow services are running",
						"config": {
							"services": ["workflow_service", "template_service", "integration_service"],
							"timeout": 20
						},
						"next_tasks": ["check_connectors"]
					},
					{
						"id": "check_connectors",
						"name": "Connectors Health Check",
						"type": "connector_health",
						"description": "Test active connector endpoints",
						"config": {
							"test_active_only": True,
							"timeout_per_connector": 10,
							"parallel_execution": True
						},
						"next_tasks": ["generate_report"]
					},
					{
						"id": "generate_report",
						"name": "Generate Health Report",
						"type": "reporting",
						"description": "Generate comprehensive health report",
						"config": {
							"report_format": "json",
							"include_metrics": True,
							"store_history": True
						},
						"next_tasks": ["send_alerts"]
					},
					{
						"id": "send_alerts",
						"name": "Send Alerts if Needed",
						"type": "conditional_notification",
						"description": "Send alerts for any health issues",
						"config": {
							"condition": "health_issues_detected",
							"notification_channels": ["email", "slack"],
							"escalation_levels": ["warning", "critical"]
						},
						"next_tasks": []
					}
				],
				"schedule": {
					"cron": "0 */15 * * * *",  # Every 15 minutes
					"timezone": "UTC",
					"enabled": True
				},
				"error_handling": {
					"default_retry_count": 2,
					"default_retry_delay": 60,
					"alert_on_failure": True
				}
			},
			"version": "1.0.0",
			"tenant_id": "system",
			"created_by": "system",
			"is_active": True,
			"is_system": True,
			"tags": ["system", "monitoring", "health", "scheduled"]
		},
		
		{
			"id": "cleanup_old_data_001",
			"name": "Cleanup Old Data",
			"description": "Automated cleanup of old workflow instances and logs",
			"definition": {
				"version": "1.0.0",
				"metadata": {
					"author": "APG System",
					"description": "Maintains database by cleaning up old data",
					"tags": ["system", "maintenance", "cleanup"]
				},
				"variables": {
					"completed_retention_days": {"type": "integer", "default": 30},
					"failed_retention_days": {"type": "integer", "default": 90},
					"logs_retention_days": {"type": "integer", "default": 30}
				},
				"tasks": [
					{
						"id": "cleanup_completed_instances",
						"name": "Cleanup Completed Instances",
						"type": "database_cleanup",
						"description": "Remove old completed workflow instances",
						"config": {
							"table": "wo_workflow_instances",
							"condition": "status = 'completed' AND completed_at < NOW() - INTERVAL '{{variables.completed_retention_days}} days'",
							"cascade_tables": ["wo_task_executions"],
							"dry_run": False
						},
						"next_tasks": ["cleanup_failed_instances"]
					},
					{
						"id": "cleanup_failed_instances",
						"name": "Cleanup Failed Instances",
						"type": "database_cleanup",
						"description": "Remove old failed workflow instances",
						"config": {
							"table": "wo_workflow_instances",
							"condition": "status = 'failed' AND completed_at < NOW() - INTERVAL '{{variables.failed_retention_days}} days'",
							"cascade_tables": ["wo_task_executions"],
							"dry_run": False
						},
						"next_tasks": ["cleanup_logs"]
					},
					{
						"id": "cleanup_logs",
						"name": "Cleanup Old Logs",
						"type": "log_cleanup",
						"description": "Archive and remove old log files",
						"config": {
							"log_directories": ["/var/log/workflow", "/var/log/apg"],
							"archive_before_delete": True,
							"archive_location": "/var/archive/logs",
							"retention_days": "{{variables.logs_retention_days}}"
						},
						"next_tasks": ["vacuum_database"]
					},
					{
						"id": "vacuum_database",
						"name": "Database Maintenance",
						"type": "database_maintenance",
						"description": "Optimize database performance",
						"config": {
							"operations": ["vacuum", "analyze", "reindex"],
							"tables": ["wo_workflows", "wo_workflow_instances", "wo_task_executions"],
							"full_vacuum": False
						},
						"next_tasks": ["generate_cleanup_report"]
					},
					{
						"id": "generate_cleanup_report",
						"name": "Generate Cleanup Report",
						"type": "reporting",
						"description": "Generate cleanup summary report",
						"config": {
							"report_type": "cleanup_summary",
							"include_metrics": True,
							"store_history": True
						},
						"next_tasks": []
					}
				],
				"schedule": {
					"cron": "0 2 * * 0",  # Weekly on Sunday at 2 AM
					"timezone": "UTC",
					"enabled": True
				}
			},
			"version": "1.0.0",
			"tenant_id": "system",
			"created_by": "system",
			"is_active": True,
			"is_system": True,
			"tags": ["system", "maintenance", "cleanup", "scheduled"]
		},
		
		{
			"id": "backup_workflow_data_001",
			"name": "Backup Workflow Data",
			"description": "Automated backup of critical workflow data",
			"definition": {
				"version": "1.0.0",
				"metadata": {
					"author": "APG System",
					"description": "Creates backups of workflow definitions and data",
					"tags": ["system", "backup", "data_protection"]
				},
				"variables": {
					"backup_location": {"type": "string", "default": "/var/backups/workflow"},
					"compression_enabled": {"type": "boolean", "default": True},
					"retention_copies": {"type": "integer", "default": 7}
				},
				"tasks": [
					{
						"id": "backup_workflows",
						"name": "Backup Workflow Definitions",
						"type": "database_backup",
						"description": "Export all workflow definitions",
						"config": {
							"tables": ["wo_workflows", "wo_workflow_templates"],
							"format": "json",
							"compression": "{{variables.compression_enabled}}",
							"destination": "{{variables.backup_location}}/workflows"
						},
						"next_tasks": ["backup_instances"]
					},
					{
						"id": "backup_instances",
						"name": "Backup Active Instances",
						"type": "database_backup",
						"description": "Export active workflow instances",
						"config": {
							"tables": ["wo_workflow_instances"],
							"condition": "status IN ('running', 'paused')",
							"format": "json",
							"destination": "{{variables.backup_location}}/instances"
						},
						"next_tasks": ["backup_connectors"]
					},
					{
						"id": "backup_connectors",
						"name": "Backup Connector Configurations",
						"type": "database_backup",
						"description": "Export connector configurations",
						"config": {
							"tables": ["wo_connectors"],
							"exclude_fields": ["auth_config"],  # Exclude sensitive data
							"format": "json",
							"destination": "{{variables.backup_location}}/connectors"
						},
						"next_tasks": ["cleanup_old_backups"]
					},
					{
						"id": "cleanup_old_backups",
						"name": "Cleanup Old Backups",
						"type": "file_cleanup",
						"description": "Remove old backup files",
						"config": {
							"directory": "{{variables.backup_location}}",
							"retention_count": "{{variables.retention_copies}}",
							"sort_by": "creation_time"
						},
						"next_tasks": ["verify_backup"]
					},
					{
						"id": "verify_backup",
						"name": "Verify Backup Integrity",
						"type": "backup_verification",
						"description": "Verify backup file integrity",
						"config": {
							"backup_directory": "{{variables.backup_location}}",
							"verify_checksums": True,
							"test_restore": False
						},
						"next_tasks": []
					}
				],
				"schedule": {
					"cron": "0 1 * * *",  # Daily at 1 AM
					"timezone": "UTC",
					"enabled": True
				}
			},
			"version": "1.0.0",
			"tenant_id": "system",
			"created_by": "system",
			"is_active": True,
			"is_system": True,
			"tags": ["system", "backup", "data_protection", "scheduled"]
		}
	]


def get_default_demo_workflows() -> List[Dict[str, Any]]:
	"""Get default demonstration workflows."""
	return [
		{
			"id": "demo_hello_world_001",
			"name": "Hello World Demo",
			"description": "Simple demonstration workflow for new users",
			"definition": {
				"version": "1.0.0",
				"metadata": {
					"author": "APG Demo",
					"description": "Basic workflow to demonstrate core concepts",
					"tags": ["demo", "tutorial", "beginner"]
				},
				"variables": {
					"greeting_name": {"type": "string", "default": "World"},
					"notification_email": {"type": "email", "required": False}
				},
				"tasks": [
					{
						"id": "generate_greeting",
						"name": "Generate Greeting",
						"type": "processing",
						"description": "Create a personalized greeting message",
						"config": {
							"operation": "string_template",
							"template": "Hello, {{variables.greeting_name}}! Welcome to APG Workflow Orchestration.",
							"output_variable": "greeting_message"
						},
						"next_tasks": ["log_message"]
					},
					{
						"id": "log_message",
						"name": "Log Message",
						"type": "logging",
						"description": "Log the greeting message",
						"config": {
							"level": "info",
							"message": "{{generate_greeting.greeting_message}}",
							"category": "demo"
						},
						"next_tasks": ["send_notification"]
					},
					{
						"id": "send_notification",
						"name": "Send Notification (Optional)",
						"type": "conditional_notification",
						"description": "Send email notification if address provided",
						"config": {
							"condition": "variables.notification_email is not null",
							"connector": "email_smtp",
							"to": "{{variables.notification_email}}",
							"subject": "Workflow Completed: Hello World Demo",
							"body": "Your demo workflow has completed successfully!\n\nMessage: {{generate_greeting.greeting_message}}"
						},
						"next_tasks": []
					}
				]
			},
			"version": "1.0.0",
			"tenant_id": "demo",
			"created_by": "system",
			"is_active": True,
			"is_system": False,
			"tags": ["demo", "tutorial", "beginner"]
		},
		
		{
			"id": "demo_data_processing_001",
			"name": "Data Processing Demo",
			"description": "Demonstrates data processing and transformation capabilities",
			"definition": {
				"version": "1.0.0",
				"metadata": {
					"author": "APG Demo",
					"description": "Shows data ingestion, processing, and output",
					"tags": ["demo", "data_processing", "transformation"]
				},
				"variables": {
					"input_data": {
						"type": "array",
						"default": [
							{"name": "Alice", "age": 30, "department": "Engineering"},
							{"name": "Bob", "age": 25, "department": "Marketing"},
							{"name": "Carol", "age": 35, "department": "Sales"}
						]
					},
					"min_age": {"type": "integer", "default": 18}
				},
				"tasks": [
					{
						"id": "validate_data",
						"name": "Validate Input Data",
						"type": "data_validation",
						"description": "Validate the structure and content of input data",
						"config": {
							"data": "{{variables.input_data}}",
							"schema": {
								"type": "array",
								"items": {
									"type": "object",
									"properties": {
										"name": {"type": "string"},
										"age": {"type": "integer", "minimum": 0},
										"department": {"type": "string"}
									},
									"required": ["name", "age", "department"]
								}
							}
						},
						"next_tasks": ["filter_data"]
					},
					{
						"id": "filter_data",
						"name": "Filter Data by Age",
						"type": "data_transformation",
						"description": "Filter records based on minimum age requirement",
						"config": {
							"operation": "filter",
							"data": "{{variables.input_data}}",
							"condition": "age >= {{variables.min_age}}",
							"output_variable": "filtered_data"
						},
						"next_tasks": ["transform_data"]
					},
					{
						"id": "transform_data",
						"name": "Transform Data",
						"type": "data_transformation",
						"description": "Add computed fields and format data",
						"config": {
							"operation": "map",
							"data": "{{filter_data.filtered_data}}",
							"transformations": {
								"full_name": "{{name}}",
								"age_group": "{{age < 30 ? 'Young' : 'Experienced'}}",
								"department_code": "{{department | upper | substring(0, 3)}}",
								"processed_at": "{{now()}}"
							},
							"output_variable": "transformed_data"
						},
						"next_tasks": ["aggregate_data"]
					},
					{
						"id": "aggregate_data",
						"name": "Aggregate Data",
						"type": "data_aggregation",
						"description": "Generate summary statistics",
						"config": {
							"data": "{{transform_data.transformed_data}}",
							"operations": {
								"total_count": "count(*)",
								"avg_age": "avg(age)",
								"departments": "distinct(department)",
								"age_groups": "group_by(age_group).count()"
							},
							"output_variable": "summary_stats"
						},
						"next_tasks": ["generate_report"]
					},
					{
						"id": "generate_report",
						"name": "Generate Processing Report",
						"type": "reporting",
						"description": "Create a summary report of the data processing",
						"config": {
							"template": "data_processing_report",
							"data": {
								"original_count": "{{variables.input_data | length}}",
								"filtered_count": "{{filter_data.filtered_data | length}}",
								"summary": "{{aggregate_data.summary_stats}}",
								"processed_data": "{{transform_data.transformed_data}}"
							},
							"format": "json",
							"output_variable": "processing_report"
						},
						"next_tasks": []
					}
				]
			},
			"version": "1.0.0",
			"tenant_id": "demo",
			"created_by": "system",
			"is_active": True,
			"is_system": False,
			"tags": ["demo", "data_processing", "transformation", "analytics"]
		}
	]


def get_default_connectors() -> List[Dict[str, Any]]:
	"""Get default system connectors."""
	return [
		{
			"id": "apg_user_management_001",
			"name": "APG User Management",
			"connector_type": "apg_capability",
			"version": "1.0.0",
			"endpoint": "internal://apg/user_management",
			"configuration": {
				"capability_id": "user_management",
				"operations": ["create_user", "update_user", "get_user", "list_users", "delete_user"],
				"authentication": {"type": "internal"},
				"timeout_seconds": 30,
				"retry_attempts": 3
			},
			"auth_config": {},
			"tenant_id": "system",
			"created_by": "system",
			"is_active": True,
			"is_system": True,
			"health_status": "unknown"
		},
		
		{
			"id": "apg_notifications_001",
			"name": "APG Notification Service",
			"connector_type": "apg_capability",
			"version": "1.0.0",
			"endpoint": "internal://apg/notifications",
			"configuration": {
				"capability_id": "notifications",
				"operations": ["send_email", "send_sms", "send_push", "create_notification"],
				"supported_channels": ["email", "sms", "push", "slack", "teams"],
				"authentication": {"type": "internal"},
				"timeout_seconds": 30,
				"retry_attempts": 3
			},
			"auth_config": {},
			"tenant_id": "system",
			"created_by": "system",
			"is_active": True,
			"is_system": True,
			"health_status": "unknown"
		},
		
		{
			"id": "apg_file_management_001",
			"name": "APG File Management",
			"connector_type": "apg_capability",
			"version": "1.0.0",
			"endpoint": "internal://apg/file_management",
			"configuration": {
				"capability_id": "file_management",
				"operations": ["upload_file", "download_file", "delete_file", "list_files", "get_file_info"],
				"supported_storages": ["local", "s3", "azure_blob", "gcs"],
				"authentication": {"type": "internal"},
				"timeout_seconds": 60,
				"retry_attempts": 3
			},
			"auth_config": {},
			"tenant_id": "system",
			"created_by": "system",
			"is_active": True,
			"is_system": True,
			"health_status": "unknown"
		},
		
		{
			"id": "http_rest_template_001",
			"name": "HTTP REST API Template",
			"connector_type": "rest_api",
			"version": "1.0.0",
			"endpoint": "https://api.example.com",
			"configuration": {
				"methods": ["GET", "POST", "PUT", "DELETE"],
				"content_types": ["application/json", "application/xml", "text/plain"],
				"timeout_seconds": 30,
				"retry_attempts": 3,
				"retry_delay_seconds": 5,
				"follow_redirects": True,
				"verify_ssl": True
			},
			"auth_config": {
				"type": "none",
				"api_key_header": "X-API-Key",
				"bearer_token_header": "Authorization"
			},
			"tenant_id": "template",
			"created_by": "system",
			"is_active": False,
			"is_system": True,
			"health_status": "unknown"
		},
		
		{
			"id": "email_smtp_template_001",
			"name": "Email SMTP Template",
			"connector_type": "email",
			"version": "1.0.0",
			"endpoint": "smtp://smtp.gmail.com:587",
			"configuration": {
				"smtp_host": "smtp.gmail.com",
				"smtp_port": 587,
				"use_tls": True,
				"use_ssl": False,
				"timeout_seconds": 30,
				"max_message_size": 25000000,  # 25MB
				"supported_formats": ["text", "html", "multipart"]
			},
			"auth_config": {
				"type": "basic",
				"username": "",
				"password": "",
				"oauth2_enabled": False
			},
			"tenant_id": "template",
			"created_by": "system",
			"is_active": False,
			"is_system": True,
			"health_status": "unknown"
		},
		
		{
			"id": "database_postgresql_template_001",
			"name": "PostgreSQL Database Template",
			"connector_type": "database",
			"version": "1.0.0",
			"endpoint": "postgresql://localhost:5432/database",
			"configuration": {
				"database_type": "postgresql",
				"connection_pool_size": 10,
				"connection_timeout": 30,
				"query_timeout": 300,
				"ssl_mode": "prefer",
				"supported_operations": ["select", "insert", "update", "delete", "execute"]
			},
			"auth_config": {
				"type": "basic",
				"username": "",
				"password": "",
				"ssl_cert": "",
				"ssl_key": ""
			},
			"tenant_id": "template",
			"created_by": "system",
			"is_active": False,
			"is_system": True,
			"health_status": "unknown"
		}
	]


def get_default_configuration() -> Dict[str, Any]:
	"""Get default system configuration."""
	return {
		"system": {
			"name": "APG Workflow Orchestration",
			"version": "1.0.0",
			"description": "Advanced workflow orchestration and automation platform",
			"default_tenant": "default",
			"max_concurrent_workflows": 100,
			"default_workflow_timeout": 3600,
			"cleanup_interval": 86400  # 24 hours
		},
		
		"features": {
			"real_time_collaboration": True,
			"visual_designer": True,
			"template_library": True,
			"advanced_analytics": True,
			"ml_optimization": True,
			"blockchain_integration": True,
			"quantum_computing": False,  # Experimental
			"multi_tenant": True,
			"audit_logging": True,
			"compliance_reporting": True
		},
		
		"security": {
			"encryption_enabled": True,
			"audit_trail": True,
			"rbac_enabled": True,
			"session_timeout": 3600,
			"max_login_attempts": 5,
			"password_policy": {
				"min_length": 8,
				"require_uppercase": True,
				"require_lowercase": True,
				"require_numbers": True,
				"require_special_chars": True
			}
		},
		
		"performance": {
			"enable_caching": True,
			"cache_ttl": 300,
			"enable_compression": True,
			"enable_connection_pooling": True,
			"max_memory_usage": "1GB",
			"enable_metrics": True,
			"metrics_retention_days": 30
		},
		
		"integrations": {
			"enable_webhooks": True,
			"webhook_timeout": 30,
			"webhook_retry_attempts": 3,
			"enable_api_gateway": True,
			"rate_limiting": {
				"enabled": True,
				"requests_per_minute": 1000,
				"burst_limit": 100
			}
		},
		
		"ui_preferences": {
			"theme": "light",
			"default_language": "en",
			"timezone": "UTC",
			"date_format": "YYYY-MM-DD",
			"time_format": "24h",
			"enable_animations": True,
			"enable_sound_notifications": False
		}
	}


def get_sample_template_data() -> List[WorkflowTemplate]:
	"""Get sample workflow templates for demonstration."""
	return [
		WorkflowTemplate(
			id="sample_approval_workflow",
			name="Document Approval Workflow",
			description="Multi-stage document approval process with parallel reviews",
			category=TemplateCategory.BUSINESS_PROCESS,
			tags=[TemplateTags.BEGINNER, TemplateTags.APPROVAL, TemplateTags.PARALLEL],
			version="1.0.0",
			author="APG Templates",
			organization="Datacraft",
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			workflow_definition={
				"name": "Document Approval Workflow",
				"description": "Automated document approval with multiple reviewers",
				"tasks": [
					{
						"id": "submit_document",
						"name": "Document Submission",
						"type": "form_input",
						"description": "Submit document for approval",
						"config": {
							"form_fields": [
								{"name": "document_title", "type": "text", "required": True},
								{"name": "document_content", "type": "file", "required": True},
								{"name": "urgency", "type": "select", "options": ["low", "medium", "high"]}
							]
						},
						"next_tasks": ["parallel_review"]
					},
					{
						"id": "parallel_review",
						"name": "Parallel Review Process",
						"type": "parallel",
						"description": "Multiple reviewers assess document simultaneously",
						"branches": [
							{
								"name": "technical_review",
								"tasks": [
									{
										"id": "technical_reviewer",
										"name": "Technical Review",
										"type": "human_task",
										"config": {
											"assignee_role": "technical_reviewer",
											"task_form": "technical_review_form",
											"sla_hours": 24
										}
									}
								]
							},
							{
								"name": "business_review",
								"tasks": [
									{
										"id": "business_reviewer",
										"name": "Business Review",
										"type": "human_task",
										"config": {
											"assignee_role": "business_reviewer",
											"task_form": "business_review_form",
											"sla_hours": 24
										}
									}
								]
							}
						],
						"next_tasks": ["final_approval"]
					},
					{
						"id": "final_approval",
						"name": "Final Approval",
						"type": "decision",
						"description": "Make final approval decision based on reviews",
						"config": {
							"decision_logic": "all_reviews_approved",
							"approver_role": "manager",
							"escalation_hours": 48
						},
						"next_tasks": ["notify_result"]
					},
					{
						"id": "notify_result",
						"name": "Notify Approval Result",
						"type": "notification",
						"description": "Send approval result to all stakeholders",
						"config": {
							"recipients": ["submitter", "reviewers", "manager"],
							"template": "approval_result",
							"channels": ["email", "dashboard"]
						},
						"next_tasks": []
					}
				]
			},
			configuration_schema={
				"type": "object",
				"properties": {
					"document_info": {
						"type": "object",
						"properties": {
							"title": {"type": "string"},
							"category": {"type": "string"},
							"priority": {"type": "string", "enum": ["low", "medium", "high"]}
						},
						"required": ["title"]
					},
					"reviewers": {
						"type": "object",
						"properties": {
							"technical_reviewer": {"type": "string"},
							"business_reviewer": {"type": "string"},
							"final_approver": {"type": "string"}
						},
						"required": ["technical_reviewer", "business_reviewer", "final_approver"]
					}
				},
				"required": ["document_info", "reviewers"]
			},
			documentation="""
# Document Approval Workflow Template

A comprehensive document approval process template with parallel reviews.

## Features
- Multi-stage approval process
- Parallel technical and business reviews
- Automated notifications
- SLA tracking and escalation
- Configurable approval criteria

## Use Cases
- Document approval processes
- Policy review workflows
- Contract approval
- Change request approval
""",
			use_cases=[
				"Corporate document approval",
				"Policy and procedure reviews",
				"Contract and agreement approval",
				"Change request processes"
			],
			prerequisites=[
				"User management system",
				"Notification service",
				"Document storage system"
			],
			estimated_duration=172800,  # 48 hours
			complexity_score=4,
			is_verified=True,
			is_featured=True
		)
	]


def get_default_user_roles() -> List[Dict[str, Any]]:
	"""Get default user roles and permissions."""
	return [
		{
			"name": "workflow_viewer",
			"display_name": "Workflow Viewer",
			"description": "Can view workflows and instances but cannot modify them",
			"permissions": [
				"workflow.read",
				"workflow_instance.read",
				"connector.read",
				"template.read"
			],
			"is_system": True,
			"created_by": "system"
		},
		
		{
			"name": "workflow_operator",
			"display_name": "Workflow Operator",  
			"description": "Can execute and monitor workflows",
			"permissions": [
				"workflow.read",
				"workflow.execute",
				"workflow_instance.read",
				"workflow_instance.control",
				"connector.read",
				"template.read"
			],
			"is_system": True,
			"created_by": "system"
		},
		
		{
			"name": "workflow_developer",
			"display_name": "Workflow Developer",
			"description": "Can create and modify workflows and templates",
			"permissions": [
				"workflow.create",
				"workflow.read", 
				"workflow.update",
				"workflow.execute",
				"workflow_instance.read",
				"workflow_instance.control",
				"connector.read",
				"template.read",
				"template.create"
			],
			"is_system": True,
			"created_by": "system"
		},
		
		{
			"name": "workflow_admin",
			"display_name": "Workflow Administrator",
			"description": "Full administrative access to workflow orchestration",
			"permissions": [
				"workflow.create",
				"workflow.read",
				"workflow.update", 
				"workflow.delete",
				"workflow.execute",
				"workflow_instance.read",
				"workflow_instance.control",
				"connector.create",
				"connector.read",
				"connector.update",
				"connector.delete",
				"template.read",
				"template.create",
				"system.admin"
			],
			"is_system": True,
			"created_by": "system"
		}
	]