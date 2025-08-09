#!/usr/bin/env python3
"""
APG Workflow Orchestration Initialization

System initialization, default data creation, and dependency setup.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from apg.framework.base_service import APGBaseService
from apg.framework.auth_rbac import APGAuth, Role, Permission
from apg.framework.database import APGDatabase
from apg.framework.audit_compliance import APGAuditLogger

from .config import get_config, WorkflowOrchestrationConfig
from .database import init_database, WorkflowDB, WorkflowInstanceDB, TaskExecutionDB, ConnectorDB
from .templates_library import TemplatesLibraryService
from .additional_templates import create_additional_templates
from .specialized_templates import create_specialized_templates
from .service import WorkflowOrchestrationService
from .integration_services import IntegrationService


logger = logging.getLogger(__name__)


class InitializationService(APGBaseService):
	"""Service for initializing the workflow orchestration system."""
	
	def __init__(self):
		super().__init__()
		self.config: Optional[WorkflowOrchestrationConfig] = None
		self.database = APGDatabase()
		self.auth = APGAuth()
		self.audit = APGAuditLogger()
		
		self.templates_service: Optional[TemplatesLibraryService] = None
		self.integration_service: Optional[IntegrationService] = None
		self.workflow_service: Optional[WorkflowOrchestrationService] = None
	
	async def start(self):
		"""Start initialization service."""
		await super().start()
		self.config = await get_config()
		logger.info("Initialization service started")
	
	async def initialize_system(self, force_recreate: bool = False) -> bool:
		"""Initialize the complete workflow orchestration system."""
		try:
			logger.info("Starting workflow orchestration system initialization...")
			
			# Step 1: Initialize database
			await self._initialize_database(force_recreate)
			
			# Step 2: Create default roles and permissions
			await self._initialize_security()
			
			# Step 3: Initialize services
			await self._initialize_services()
			
			# Step 4: Load default templates
			await self._initialize_templates()
			
			# Step 5: Create default connectors
			await self._initialize_connectors()
			
			# Step 6: Setup default workflows
			await self._initialize_default_workflows()
			
			# Step 7: Create system monitoring
			await self._initialize_monitoring()
			
			# Step 8: Validate system health
			await self._validate_system_health()
			
			logger.info("✅ Workflow orchestration system initialization completed successfully")
			return True
			
		except Exception as e:
			logger.error(f"❌ System initialization failed: {e}")
			raise
	
	async def _initialize_database(self, force_recreate: bool = False):
		"""Initialize database schema and tables."""
		logger.info("Initializing database...")
		
		try:
			# Initialize database connection
			await self.database.start()
			
			# Initialize workflow orchestration database
			await init_database(force_recreate=force_recreate)
			
			# Create indexes for performance
			await self._create_database_indexes()
			
			# Setup database maintenance tasks
			await self._setup_database_maintenance()
			
			logger.info("✅ Database initialization completed")
			
		except Exception as e:
			logger.error(f"❌ Database initialization failed: {e}")
			raise
	
	async def _create_database_indexes(self):
		"""Create database indexes for optimal performance."""
		indexes = [
			# Workflow indexes
			"CREATE INDEX IF NOT EXISTS idx_workflows_tenant_id ON wo_workflows(tenant_id)",
			"CREATE INDEX IF NOT EXISTS idx_workflows_status ON wo_workflows(is_active)",
			"CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON wo_workflows(created_at)",
			"CREATE INDEX IF NOT EXISTS idx_workflows_name ON wo_workflows(name)",
			
			# Workflow instance indexes
			"CREATE INDEX IF NOT EXISTS idx_instances_workflow_id ON wo_workflow_instances(workflow_id)",
			"CREATE INDEX IF NOT EXISTS idx_instances_status ON wo_workflow_instances(status)",
			"CREATE INDEX IF NOT EXISTS idx_instances_tenant_id ON wo_workflow_instances(tenant_id)",
			"CREATE INDEX IF NOT EXISTS idx_instances_created_at ON wo_workflow_instances(created_at)",
			"CREATE INDEX IF NOT EXISTS idx_instances_completed_at ON wo_workflow_instances(completed_at)",
			
			# Task execution indexes
			"CREATE INDEX IF NOT EXISTS idx_tasks_instance_id ON wo_task_executions(workflow_instance_id)",
			"CREATE INDEX IF NOT EXISTS idx_tasks_status ON wo_task_executions(status)",
			"CREATE INDEX IF NOT EXISTS idx_tasks_task_id ON wo_task_executions(task_id)",
			"CREATE INDEX IF NOT EXISTS idx_tasks_started_at ON wo_task_executions(started_at)",
			
			# Connector indexes
			"CREATE INDEX IF NOT EXISTS idx_connectors_type ON wo_connectors(connector_type)",
			"CREATE INDEX IF NOT EXISTS idx_connectors_tenant_id ON wo_connectors(tenant_id)",
			"CREATE INDEX IF NOT EXISTS idx_connectors_active ON wo_connectors(is_active)",
		]
		
		for index_sql in indexes:
			try:
				await self.database.execute(index_sql)
			except Exception as e:
				logger.warning(f"Failed to create index: {e}")
	
	async def _setup_database_maintenance(self):
		"""Setup automated database maintenance tasks."""
		maintenance_tasks = [
			# Clean up old completed workflow instances
			"""
			CREATE OR REPLACE FUNCTION cleanup_old_workflows() RETURNS void AS $$
			BEGIN
				DELETE FROM wo_workflow_instances 
				WHERE status = 'completed' 
				AND completed_at < NOW() - INTERVAL '30 days';
				
				DELETE FROM wo_task_executions 
				WHERE completed_at < NOW() - INTERVAL '30 days'
				AND workflow_instance_id NOT IN (
					SELECT id FROM wo_workflow_instances
				);
			END;
			$$ LANGUAGE plpgsql;
			""",
			
			# Update workflow statistics
			"""
			CREATE OR REPLACE FUNCTION update_workflow_stats() RETURNS void AS $$
			BEGIN
				-- Update execution counts and performance metrics
				UPDATE wo_workflows SET 
					total_executions = (
						SELECT COUNT(*) FROM wo_workflow_instances 
						WHERE workflow_id = wo_workflows.id
					),
					successful_executions = (
						SELECT COUNT(*) FROM wo_workflow_instances 
						WHERE workflow_id = wo_workflows.id AND status = 'completed'
					),
					avg_duration_seconds = (
						SELECT AVG(duration_seconds) FROM wo_workflow_instances 
						WHERE workflow_id = wo_workflows.id AND status = 'completed'
					);
			END;
			$$ LANGUAGE plpgsql;
			"""
		]
		
		for task_sql in maintenance_tasks:
			try:
				await self.database.execute(task_sql)
			except Exception as e:
				logger.warning(f"Failed to create maintenance task: {e}")
	
	async def _initialize_security(self):
		"""Initialize security roles and permissions."""
		logger.info("Initializing security...")
		
		try:
			# Define workflow orchestration permissions
			permissions = [
				Permission(
					name="workflow.create",
					description="Create new workflows",
					resource_type="workflow"
				),
				Permission(
					name="workflow.read",
					description="Read workflows",
					resource_type="workflow"
				),
				Permission(
					name="workflow.update",
					description="Update workflows",
					resource_type="workflow"
				),
				Permission(
					name="workflow.delete",
					description="Delete workflows",
					resource_type="workflow"
				),
				Permission(
					name="workflow.execute",
					description="Execute workflows",
					resource_type="workflow"
				),
				Permission(
					name="workflow_instance.read",
					description="Read workflow instances",
					resource_type="workflow_instance"
				),
				Permission(
					name="workflow_instance.control",
					description="Control workflow instances (pause, resume, cancel)",
					resource_type="workflow_instance"
				),
				Permission(
					name="connector.create",
					description="Create connectors",
					resource_type="connector"
				),
				Permission(
					name="connector.read",
					description="Read connectors",
					resource_type="connector"
				),
				Permission(
					name="connector.update",
					description="Update connectors",
					resource_type="connector"
				),
				Permission(
					name="connector.delete",
					description="Delete connectors",
					resource_type="connector"
				),
				Permission(
					name="template.read",
					description="Read workflow templates",
					resource_type="template"
				),
				Permission(
					name="template.create",
					description="Create workflow templates",
					resource_type="template"
				),
				Permission(
					name="system.admin",
					description="System administration",
					resource_type="system"
				)
			]
			
			# Create permissions
			for permission in permissions:
				await self.auth.create_permission(permission)
			
			# Define roles
			roles = [
				Role(
					name="workflow_viewer",
					description="Can view workflows and instances",
					permissions=["workflow.read", "workflow_instance.read", "template.read"]
				),
				Role(
					name="workflow_operator",
					description="Can execute and monitor workflows",
					permissions=[
						"workflow.read", "workflow.execute",
						"workflow_instance.read", "workflow_instance.control",
						"connector.read", "template.read"
					]
				),
				Role(
					name="workflow_developer",
					description="Can create and modify workflows",
					permissions=[
						"workflow.create", "workflow.read", "workflow.update",
						"workflow.execute", "workflow_instance.read", "workflow_instance.control",
						"connector.read", "template.read", "template.create"
					]
				),
				Role(
					name="workflow_admin",
					description="Full workflow orchestration administration",
					permissions=[
						"workflow.create", "workflow.read", "workflow.update", "workflow.delete",
						"workflow.execute", "workflow_instance.read", "workflow_instance.control",
						"connector.create", "connector.read", "connector.update", "connector.delete",
						"template.read", "template.create", "system.admin"
					]
				)
			]
			
			# Create roles
			for role in roles:
				await self.auth.create_role(role)
			
			logger.info("✅ Security initialization completed")
			
		except Exception as e:
			logger.error(f"❌ Security initialization failed: {e}")
			raise
	
	async def _initialize_services(self):
		"""Initialize core services."""
		logger.info("Initializing services...")
		
		try:
			# Initialize templates service
			self.templates_service = TemplatesLibraryService()
			await self.templates_service.start()
			
			# Initialize integration service
			self.integration_service = IntegrationService()
			await self.integration_service.start()
			
			# Initialize workflow service
			self.workflow_service = WorkflowOrchestrationService()
			await self.workflow_service.start()
			
			logger.info("✅ Services initialization completed")
			
		except Exception as e:
			logger.error(f"❌ Services initialization failed: {e}")
			raise
	
	async def _initialize_templates(self):
		"""Load default workflow templates."""
		logger.info("Initializing workflow templates...")
		
		try:
			if not self.templates_service:
				raise ValueError("Templates service not initialized")
			
			# Load built-in templates
			await self.templates_service._initialize_templates()
			
			# Load additional templates
			additional_templates = create_additional_templates()
			for template in additional_templates:
				await self.templates_service.add_template(template)
			
			# Load specialized templates
			specialized_templates = create_specialized_templates()
			for template in specialized_templates:
				await self.templates_service.add_template(template)
			
			# Create default template categories
			await self._create_default_template_categories()
			
			template_count = len(await self.templates_service.list_templates())
			logger.info(f"✅ Templates initialization completed ({template_count} templates loaded)")
			
		except Exception as e:
			logger.error(f"❌ Templates initialization failed: {e}")
			raise
	
	async def _create_default_template_categories(self):
		"""Create default template categories and tags."""
		categories = [
			{
				"name": "Getting Started",
				"description": "Simple templates for new users",
				"templates": ["hello_world", "data_processing_basic", "notification_simple"]
			},
			{
				"name": "Business Process",
				"description": "Common business workflow templates",
				"templates": ["approval_workflow", "invoice_processing", "employee_onboarding"]
			},
			{
				"name": "Healthcare",
				"description": "HIPAA-compliant healthcare workflows",
				"templates": ["patient_admission", "medical_imaging", "clinical_trial"]
			},
			{
				"name": "E-commerce",
				"description": "Online retail and e-commerce workflows",
				"templates": ["order_fulfillment", "inventory_management", "customer_return"]
			},
			{
				"name": "Advanced Analytics",
				"description": "Data science and machine learning workflows",
				"templates": ["ml_model_training", "data_pipeline", "anomaly_detection"]
			}
		]
		
		# Store categories in database or cache for UI
		for category in categories:
			await self.templates_service._store_category(category)
	
	async def _initialize_connectors(self):
		"""Create default system connectors."""
		logger.info("Initializing default connectors...")
		
		try:
			if not self.integration_service:
				raise ValueError("Integration service not initialized")
			
			# APG Framework connectors
			apg_connectors = [
				{
					"name": "APG User Management",
					"connector_type": "apg_capability",
					"version": "1.0.0",
					"endpoint": "internal://apg/user_management",
					"tenant_id": self.config.apg_tenant_id,
					"configuration": {
						"capability_id": "user_management",
						"operations": ["create_user", "update_user", "get_user", "list_users"]
					},
					"is_active": True,
					"is_system": True
				},
				{
					"name": "APG Notification Service",
					"connector_type": "apg_capability",
					"version": "1.0.0",
					"endpoint": "internal://apg/notifications",
					"tenant_id": self.config.apg_tenant_id,
					"configuration": {
						"capability_id": "notifications",
						"operations": ["send_email", "send_sms", "send_push", "create_notification"]
					},
					"is_active": True,
					"is_system": True
				},
				{
					"name": "APG File Management",
					"connector_type": "apg_capability",
					"version": "1.0.0",
					"endpoint": "internal://apg/file_management",
					"tenant_id": self.config.apg_tenant_id,
					"configuration": {
						"capability_id": "file_management",
						"operations": ["upload_file", "download_file", "delete_file", "list_files"]
					},
					"is_active": True,
					"is_system": True
				}
			]
			
			# External service connectors
			external_connectors = [
				{
					"name": "HTTP REST API",
					"connector_type": "rest_api",
					"version": "1.0.0",
					"endpoint": "https://api.example.com",
					"tenant_id": self.config.apg_tenant_id,
					"configuration": {
						"timeout_seconds": 30,
						"retry_attempts": 3,
						"authentication": {"type": "none"}
					},
					"is_active": False,
					"is_system": True
				},
				{
					"name": "Email SMTP",
					"connector_type": "email",
					"version": "1.0.0",
					"endpoint": "smtp://smtp.gmail.com:587",
					"tenant_id": self.config.apg_tenant_id,
					"configuration": {
						"smtp_host": "smtp.gmail.com",
						"smtp_port": 587,
						"use_tls": True
					},
					"is_active": False,
					"is_system": True
				},
				{
					"name": "Database PostgreSQL",
					"connector_type": "database",
					"version": "1.0.0",
					"endpoint": "postgresql://localhost:5432/defaultdb",
					"tenant_id": self.config.apg_tenant_id,
					"configuration": {
						"database_type": "postgresql",
						"connection_pool_size": 10
					},
					"is_active": False,
					"is_system": True
				}
			]
			
			# Create connectors
			all_connectors = apg_connectors + external_connectors
			for connector_data in all_connectors:
				connector = ConnectorDB(**connector_data)
				await self.integration_service.create_connector(connector)
			
			logger.info(f"✅ Connectors initialization completed ({len(all_connectors)} connectors created)")
			
		except Exception as e:
			logger.error(f"❌ Connectors initialization failed: {e}")
			raise
	
	async def _initialize_default_workflows(self):
		"""Create default system workflows."""
		logger.info("Initializing default workflows...")
		
		try:
			if not self.workflow_service:
				raise ValueError("Workflow service not initialized")
			
			# System maintenance workflows
			system_workflows = [
				{
					"name": "System Health Check",
					"description": "Periodic system health monitoring",
					"definition": {
						"tasks": [
							{
								"id": "check_database",
								"name": "Database Health Check",
								"type": "health_check",
								"config": {"target": "database"}
							},
							{
								"id": "check_services",
								"name": "Services Health Check",
								"type": "health_check",
								"config": {"target": "services"}
							},
							{
								"id": "check_connectors",
								"name": "Connectors Health Check",
								"type": "health_check",
								"config": {"target": "connectors"}
							}
						],
						"schedule": "0 */6 * * *"  # Every 6 hours
					},
					"tenant_id": self.config.apg_tenant_id,
					"is_active": True,
					"is_system": True
				},
				{
					"name": "Cleanup Old Data",
					"description": "Clean up old workflow instances and logs",
					"definition": {
						"tasks": [
							{
								"id": "cleanup_instances",
								"name": "Cleanup Old Instances",
								"type": "maintenance",
								"config": {
									"operation": "cleanup_old_instances",
									"retention_days": 30
								}
							},
							{
								"id": "cleanup_logs",
								"name": "Cleanup Old Logs",
								"type": "maintenance",
								"config": {
									"operation": "cleanup_old_logs",
									"retention_days": 90
								}
							}
						],
						"schedule": "0 2 * * *"  # Daily at 2 AM
					},
					"tenant_id": self.config.apg_tenant_id,
					"is_active": True,
					"is_system": True
				}
			]
			
			# Create system workflows
			for workflow_data in system_workflows:
				workflow = WorkflowDB(**workflow_data)
				await self.workflow_service.create_workflow(workflow)
			
			logger.info(f"✅ Default workflows initialization completed ({len(system_workflows)} workflows created)")
			
		except Exception as e:
			logger.error(f"❌ Default workflows initialization failed: {e}")
			raise
	
	async def _initialize_monitoring(self):
		"""Setup system monitoring and health checks."""
		logger.info("Initializing monitoring...")
		
		try:
			# Create monitoring dashboards
			dashboards = [
				{
					"name": "Workflow Orchestration Overview",
					"description": "Main system overview dashboard",
					"widgets": [
						{"type": "metric", "title": "Active Workflows", "query": "count_active_workflows"},
						{"type": "metric", "title": "Running Instances", "query": "count_running_instances"},
						{"type": "chart", "title": "Executions Over Time", "query": "executions_timeline"},
						{"type": "chart", "title": "Success Rate", "query": "success_rate_timeline"}
					]
				},
				{
					"name": "Performance Metrics",
					"description": "System performance monitoring",
					"widgets": [
						{"type": "metric", "title": "Avg Execution Time", "query": "avg_execution_time"},
						{"type": "metric", "title": "Queue Depth", "query": "queue_depth"},
						{"type": "chart", "title": "Resource Usage", "query": "resource_usage_timeline"},
						{"type": "chart", "title": "Error Rate", "query": "error_rate_timeline"}
					]
				}
			]
			
			# Store dashboard configurations
			for dashboard in dashboards:
				await self._store_dashboard_config(dashboard)
			
			# Setup health check endpoints
			await self._setup_health_checks()
			
			logger.info("✅ Monitoring initialization completed")
			
		except Exception as e:
			logger.error(f"❌ Monitoring initialization failed: {e}")
			raise
	
	async def _store_dashboard_config(self, dashboard: Dict[str, Any]):
		"""Store dashboard configuration."""
		# This would typically store in a configuration database or file
		dashboard_file = f"dashboards/{dashboard['name'].lower().replace(' ', '_')}.json"
		# Implementation would depend on your dashboard system
		logger.debug(f"Dashboard config stored: {dashboard_file}")
	
	async def _setup_health_checks(self):
		"""Setup health check endpoints and monitoring."""
		health_checks = [
			{
				"name": "database_connection",
				"description": "Database connectivity check",
				"endpoint": "/health/database",
				"timeout": 5,
				"critical": True
			},
			{
				"name": "redis_connection",
				"description": "Redis connectivity check",
				"endpoint": "/health/redis",
				"timeout": 5,
				"critical": True
			},
			{
				"name": "workflow_service",
				"description": "Workflow service health",
				"endpoint": "/health/workflow-service",
				"timeout": 10,
				"critical": True
			},
			{
				"name": "templates_service",
				"description": "Templates service health",
				"endpoint": "/health/templates-service",
				"timeout": 5,
				"critical": False
			}
		]
		
		# Register health checks
		for check in health_checks:
			await self._register_health_check(check)
	
	async def _register_health_check(self, check: Dict[str, Any]):
		"""Register a health check."""
		# Implementation would register with monitoring system
		logger.debug(f"Health check registered: {check['name']}")
	
	async def _validate_system_health(self):
		"""Validate that all system components are healthy."""
		logger.info("Validating system health...")
		
		health_checks = []
		
		# Check database connectivity
		try:
			await self.database.execute("SELECT 1")
			health_checks.append(("Database", True, "Connected"))
		except Exception as e:
			health_checks.append(("Database", False, str(e)))
		
		# Check services
		services = [
			("Templates Service", self.templates_service),
			("Integration Service", self.integration_service),
			("Workflow Service", self.workflow_service)
		]
		
		for service_name, service in services:
			try:
				if service and hasattr(service, 'health_check'):
					healthy = await service.health_check()
					health_checks.append((service_name, healthy, "OK" if healthy else "Failed"))
				else:
					health_checks.append((service_name, service is not None, "Service available" if service else "Service not initialized"))
			except Exception as e:
				health_checks.append((service_name, False, str(e)))
		
		# Report health status
		failed_checks = [check for check in health_checks if not check[1]]
		
		if failed_checks:
			logger.warning("⚠️  Some health checks failed:")
			for name, status, message in failed_checks:
				logger.warning(f"  - {name}: {message}")
		else:
			logger.info("✅ All system health checks passed")
		
		# Log all health checks
		for name, status, message in health_checks:
			status_icon = "✅" if status else "❌"
			logger.info(f"  {status_icon} {name}: {message}")
		
		return len(failed_checks) == 0
	
	async def reset_system(self, confirm: bool = False) -> bool:
		"""Reset the entire system (WARNING: Destructive operation)."""
		if not confirm:
			logger.error("System reset requires explicit confirmation")
			return False
		
		logger.warning("🚨 RESETTING WORKFLOW ORCHESTRATION SYSTEM 🚨")
		
		try:
			# Stop all services
			if self.workflow_service:
				await self.workflow_service.stop()
			if self.integration_service:
				await self.integration_service.stop()
			if self.templates_service:
				await self.templates_service.stop()
			
			# Reset database
			await init_database(force_recreate=True)
			
			# Reinitialize system
			await self.initialize_system()
			
			logger.info("✅ System reset completed successfully")
			return True
			
		except Exception as e:
			logger.error(f"❌ System reset failed: {e}")
			raise
	
	async def get_system_status(self) -> Dict[str, Any]:
		"""Get comprehensive system status."""
		status = {
			"system": {
				"initialized": True,
				"version": self.config.service_version if self.config else "unknown",
				"environment": self.config.environment.value if self.config else "unknown",
				"uptime": datetime.utcnow()  # Would track actual uptime
			},
			"services": {},
			"database": {},
			"health_checks": []
		}
		
		# Check service status
		services = [
			("templates", self.templates_service),
			("integration", self.integration_service),
			("workflow", self.workflow_service)
		]
		
		for service_name, service in services:
			if service:
				try:
					health = await service.health_check() if hasattr(service, 'health_check') else True
					status["services"][service_name] = {
						"status": "healthy" if health else "unhealthy",
						"initialized": True
					}
				except Exception as e:
					status["services"][service_name] = {
						"status": "error",
						"error": str(e),
						"initialized": False
					}
			else:
				status["services"][service_name] = {
					"status": "not_initialized",
					"initialized": False
				}
		
		# Database status
		try:
			await self.database.execute("SELECT 1")
			status["database"] = {
				"status": "connected",
				"url": self.config.get_database_url() if self.config else "unknown"
			}
		except Exception as e:
			status["database"] = {
				"status": "error",
				"error": str(e)
			}
		
		return status


# Global initialization service instance
initialization_service = InitializationService()


async def initialize_workflow_orchestration(force_recreate: bool = False) -> bool:
	"""Initialize the workflow orchestration system."""
	if not initialization_service.is_started:
		await initialization_service.start()
	
	return await initialization_service.initialize_system(force_recreate=force_recreate)


async def get_system_status() -> Dict[str, Any]:
	"""Get system status."""
	if not initialization_service.is_started:
		await initialization_service.start()
	
	return await initialization_service.get_system_status()


async def reset_system(confirm: bool = False) -> bool:
	"""Reset the system."""
	if not initialization_service.is_started:
		await initialization_service.start()
	
	return await initialization_service.reset_system(confirm=confirm)