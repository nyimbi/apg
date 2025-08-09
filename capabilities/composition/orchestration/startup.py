#!/usr/bin/env python3
"""
APG Workflow Orchestration Startup

Application startup, initialization, and dependency management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from contextual import AsyncContextManager

from apg.framework.base_service import APGBaseService
from apg.framework.messaging import APGEventBus
from apg.framework.monitoring import APGMonitoring

from .config import get_config, ConfigurationManager, config_manager
from .initialization import initialize_workflow_orchestration, initialization_service
from .migrations import apply_migrations, migration_manager
from .service import WorkflowOrchestrationService
from .integration_services import IntegrationService
from .templates_library import TemplatesLibraryService
from .realtime_api import realtime_api_service
from .realtime_handlers import RealTimeEventManager


logger = logging.getLogger(__name__)


class WorkflowOrchestrationApp:
	"""Main application class for workflow orchestration."""
	
	def __init__(self):
		self.config: Optional[Any] = None
		self.services: Dict[str, APGBaseService] = {}
		self.event_bus: Optional[APGEventBus] = None
		self.monitoring: Optional[APGMonitoring] = None
		self.realtime_manager: Optional[RealTimeEventManager] = None
		self.is_running = False
		self.shutdown_event = asyncio.Event()
	
	async def initialize(self) -> bool:
		"""Initialize the application and all services."""
		try:
			logger.info("🚀 Starting APG Workflow Orchestration initialization...")
			
			# Step 1: Load configuration
			await self._load_configuration()
			
			# Step 2: Setup logging
			await self._setup_logging()
			
			# Step 3: Apply database migrations
			await self._apply_migrations()
			
			# Step 4: Initialize core services
			await self._initialize_core_services()
			
			# Step 5: Initialize workflow orchestration system
			await self._initialize_workflow_system()
			
			# Step 6: Setup monitoring and health checks
			await self._setup_monitoring()
			
			# Step 7: Setup real-time features
			await self._setup_realtime_features()
			
			# Step 8: Setup event bus and messaging
			await self._setup_event_bus()
			
			# Step 9: Register signal handlers
			await self._setup_signal_handlers()
			
			# Step 10: Perform health checks
			await self._perform_startup_health_checks()
			
			logger.info("✅ APG Workflow Orchestration initialization completed successfully")
			return True
			
		except Exception as e:
			logger.error(f"❌ Application initialization failed: {e}")
			await self.shutdown()
			raise
	
	async def _load_configuration(self):
		"""Load and validate configuration."""
		logger.info("Loading configuration...")
		
		if not config_manager.is_started:
			await config_manager.start()
		
		self.config = await get_config()
		
		logger.info(f"Configuration loaded for environment: {self.config.environment.value}")
	
	async def _setup_logging(self):
		"""Setup application logging."""
		log_level = self.config.monitoring.log_level.value
		log_format = self.config.monitoring.log_format
		
		if log_format == "json":
			import json_logging
			json_logging.init_non_web(enable_json=True)
		
		logging.basicConfig(
			level=getattr(logging, log_level),
			format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		)
		
		# Set specific logger levels
		logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
		logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
		
		logger.info(f"Logging configured: level={log_level}, format={log_format}")
	
	async def _apply_migrations(self):
		"""Apply database migrations."""
		logger.info("Applying database migrations...")
		
		if not migration_manager.is_started:
			await migration_manager.start()
		
		applied_count = await apply_migrations()
		
		if applied_count > 0:
			logger.info(f"Applied {applied_count} database migrations")
		else:
			logger.info("Database schema is up to date")
	
	async def _initialize_core_services(self):
		"""Initialize core APG services."""
		logger.info("Initializing core services...")
		
		# Configuration manager
		self.services['config'] = config_manager
		
		# Migration manager  
		self.services['migrations'] = migration_manager
		
		# Initialization service
		if not initialization_service.is_started:
			await initialization_service.start()
		self.services['initialization'] = initialization_service
		
		logger.info("Core services initialized")
	
	async def _initialize_workflow_system(self):
		"""Initialize the workflow orchestration system."""
		logger.info("Initializing workflow orchestration system...")
		
		# Initialize the complete system
		success = await initialize_workflow_orchestration(
			force_recreate=self.config.debug and self.config.is_development()
		)
		
		if not success:
			raise RuntimeError("Failed to initialize workflow orchestration system")
		
		# Get service instances
		self.services['workflow'] = initialization_service.workflow_service
		self.services['templates'] = initialization_service.templates_service
		self.services['integration'] = initialization_service.integration_service
		
		logger.info("Workflow orchestration system initialized")
	
	async def _setup_monitoring(self):
		"""Setup monitoring and metrics collection."""
		if not self.config.monitoring.enable_metrics:
			logger.info("Monitoring disabled in configuration")
			return
		
		logger.info("Setting up monitoring...")
		
		try:
			self.monitoring = APGMonitoring(
				service_name=self.config.service_name,
				service_version=self.config.service_version,
				environment=self.config.environment.value
			)
			
			await self.monitoring.start()
			self.services['monitoring'] = self.monitoring
			
			# Setup custom metrics
			await self._setup_custom_metrics()
			
			logger.info("Monitoring setup completed")
			
		except Exception as e:
			logger.warning(f"Failed to setup monitoring: {e}")
			if self.config.is_production():
				raise
	
	async def _setup_custom_metrics(self):
		"""Setup custom workflow orchestration metrics."""
		if not self.monitoring:
			return
		
		# Workflow metrics
		await self.monitoring.create_gauge(
			'active_workflows_total',
			'Total number of active workflows'
		)
		
		await self.monitoring.create_gauge(
			'running_instances_total',
			'Total number of running workflow instances'
		)
		
		await self.monitoring.create_counter(
			'workflow_executions_total',
			'Total number of workflow executions',
			labels=['workflow_id', 'status']
		)
		
		await self.monitoring.create_histogram(
			'workflow_execution_duration_seconds',
			'Duration of workflow executions',
			labels=['workflow_id']
		)
		
		await self.monitoring.create_gauge(
			'connector_health_status',
			'Health status of connectors',
			labels=['connector_id', 'connector_type']
		)
	
	async def _setup_realtime_features(self):
		"""Setup real-time API and collaboration features."""
		if not self.config.integration.enable_webhooks:
			logger.info("Real-time features disabled in configuration")
			return
		
		logger.info("Setting up real-time features...")
		
		try:
			# Start real-time API service
			await realtime_api_service.start()
			self.services['realtime_api'] = realtime_api_service
			
			# Setup real-time event manager
			workflow_service = self.services.get('workflow')
			if workflow_service:
				self.realtime_manager = RealTimeEventManager(workflow_service)
				await self.realtime_manager.start()
				self.services['realtime_manager'] = self.realtime_manager
			
			logger.info("Real-time features setup completed")
			
		except Exception as e:
			logger.warning(f"Failed to setup real-time features: {e}")
			if self.config.is_production():
				raise
	
	async def _setup_event_bus(self):
		"""Setup event bus for inter-service communication."""
		logger.info("Setting up event bus...")
		
		try:
			self.event_bus = APGEventBus(
				redis_url=self.config.get_redis_url(),
				service_name=self.config.service_name
			)
			
			await self.event_bus.start()
			self.services['event_bus'] = self.event_bus
			
			# Setup event handlers
			await self._setup_event_handlers()
			
			logger.info("Event bus setup completed")
			
		except Exception as e:
			logger.warning(f"Failed to setup event bus: {e}")
			if self.config.is_production():
				raise
	
	async def _setup_event_handlers(self):
		"""Setup event handlers for various system events."""
		if not self.event_bus:
			return
		
		# Workflow execution events
		await self.event_bus.subscribe('workflow.executed', self._handle_workflow_executed)
		await self.event_bus.subscribe('workflow.completed', self._handle_workflow_completed)
		await self.event_bus.subscribe('workflow.failed', self._handle_workflow_failed)
		
		# System events
		await self.event_bus.subscribe('system.health_check', self._handle_health_check)
		await self.event_bus.subscribe('system.shutdown', self._handle_shutdown_request)
	
	async def _handle_workflow_executed(self, event: Dict[str, Any]):
		"""Handle workflow execution event."""
		if self.monitoring:
			await self.monitoring.increment_counter(
				'workflow_executions_total',
				labels={
					'workflow_id': event.get('workflow_id', 'unknown'),
					'status': 'started'
				}
			)
	
	async def _handle_workflow_completed(self, event: Dict[str, Any]):
		"""Handle workflow completion event."""
		if self.monitoring:
			await self.monitoring.increment_counter(
				'workflow_executions_total',
				labels={
					'workflow_id': event.get('workflow_id', 'unknown'),
					'status': 'completed'
				}
			)
			
			duration = event.get('duration_seconds', 0)
			if duration > 0:
				await self.monitoring.observe_histogram(
					'workflow_execution_duration_seconds',
					duration,
					labels={'workflow_id': event.get('workflow_id', 'unknown')}
				)
	
	async def _handle_workflow_failed(self, event: Dict[str, Any]):
		"""Handle workflow failure event."""
		if self.monitoring:
			await self.monitoring.increment_counter(
				'workflow_executions_total',
				labels={
					'workflow_id': event.get('workflow_id', 'unknown'),
					'status': 'failed'
				}
			)
	
	async def _handle_health_check(self, event: Dict[str, Any]):
		"""Handle health check request."""
		health_status = await self.get_health_status()
		
		if self.event_bus:
			await self.event_bus.publish('system.health_response', {
				'request_id': event.get('request_id'),
				'health_status': health_status
			})
	
	async def _handle_shutdown_request(self, event: Dict[str, Any]):
		"""Handle graceful shutdown request."""
		logger.info("Received shutdown request via event bus")
		self.shutdown_event.set()
	
	async def _setup_signal_handlers(self):
		"""Setup signal handlers for graceful shutdown."""
		if sys.platform != 'win32':
			loop = asyncio.get_event_loop()
			
			for sig in (signal.SIGTERM, signal.SIGINT):
				loop.add_signal_handler(sig, self._signal_handler, sig)
	
	def _signal_handler(self, sig):
		"""Handle shutdown signals."""
		logger.info(f"Received signal {sig}, initiating graceful shutdown...")
		self.shutdown_event.set()
	
	async def _perform_startup_health_checks(self):
		"""Perform health checks after startup."""
		logger.info("Performing startup health checks...")
		
		health_status = await initialization_service.get_system_status()
		
		# Check critical services
		critical_services = ['database', 'workflow', 'templates']
		failed_services = []
		
		for service_name in critical_services:
			service_status = health_status.get('services', {}).get(service_name, {})
			if service_status.get('status') != 'healthy':
				failed_services.append(service_name)
		
		if failed_services:
			error_msg = f"Critical services failed health check: {', '.join(failed_services)}"
			logger.error(error_msg)
			if self.config.is_production():
				raise RuntimeError(error_msg)
			else:
				logger.warning("Continuing startup despite failed health checks (development mode)")
		else:
			logger.info("✅ All startup health checks passed")
	
	async def run(self):
		"""Run the application."""
		if not await self.initialize():
			return False
		
		self.is_running = True
		logger.info(f"🎉 APG Workflow Orchestration is running on {self.config.service_host}:{self.config.service_port}")
		
		try:
			# Start background tasks
			await self._start_background_tasks()
			
			# Wait for shutdown signal
			await self.shutdown_event.wait()
			
		except Exception as e:
			logger.error(f"Runtime error: {e}")
			raise
		finally:
			await self.shutdown()
	
	async def _start_background_tasks(self):
		"""Start background tasks."""
		background_tasks = [
			self._metrics_collection_task(),
			self._health_monitoring_task(),
			self._cleanup_task()
		]
		
		for task in background_tasks:
			asyncio.create_task(task)
	
	async def _metrics_collection_task(self):
		"""Background task for metrics collection."""
		if not self.monitoring:
			return
		
		while self.is_running:
			try:
				# Update workflow metrics
				workflow_service = self.services.get('workflow')
				if workflow_service:
					active_workflows = await workflow_service.get_active_workflows_count()
					running_instances = await workflow_service.get_running_instances_count()
					
					await self.monitoring.set_gauge('active_workflows_total', active_workflows)
					await self.monitoring.set_gauge('running_instances_total', running_instances)
				
				# Update connector health metrics
				integration_service = self.services.get('integration')
				if integration_service:
					connector_health = await integration_service.get_all_connectors_health()
					for connector_id, health in connector_health.items():
						await self.monitoring.set_gauge(
							'connector_health_status',
							1 if health.get('status') == 'healthy' else 0,
							labels={
								'connector_id': connector_id,
								'connector_type': health.get('type', 'unknown')
							}
						)
				
				await asyncio.sleep(30)  # Collect metrics every 30 seconds
				
			except Exception as e:
				logger.error(f"Metrics collection error: {e}")
				await asyncio.sleep(60)
	
	async def _health_monitoring_task(self):
		"""Background task for health monitoring."""
		while self.is_running:
			try:
				# Perform health checks
				health_status = await self.get_health_status()
				
				# Log unhealthy services
				unhealthy_services = [
					name for name, status in health_status.get('services', {}).items()
					if status.get('status') != 'healthy'
				]
				
				if unhealthy_services:
					logger.warning(f"Unhealthy services detected: {', '.join(unhealthy_services)}")
				
				await asyncio.sleep(300)  # Check health every 5 minutes
				
			except Exception as e:
				logger.error(f"Health monitoring error: {e}")
				await asyncio.sleep(600)
	
	async def _cleanup_task(self):
		"""Background task for system cleanup."""
		while self.is_running:
			try:
				# Cleanup old workflow instances
				workflow_service = self.services.get('workflow')
				if workflow_service:
					cleaned_count = await workflow_service.cleanup_old_instances()
					if cleaned_count > 0:
						logger.info(f"Cleaned up {cleaned_count} old workflow instances")
				
				# Sleep for cleanup interval (default 24 hours)
				await asyncio.sleep(self.config.workflow.cleanup_completed_after)
				
			except Exception as e:
				logger.error(f"Cleanup task error: {e}")
				await asyncio.sleep(3600)  # Retry in 1 hour
	
	async def get_health_status(self) -> Dict[str, Any]:
		"""Get comprehensive health status."""
		return await initialization_service.get_system_status()
	
	async def shutdown(self):
		"""Graceful shutdown of the application."""
		if not self.is_running:
			return
		
		logger.info("🛑 Initiating graceful shutdown...")
		self.is_running = False
		
		# Stop services in reverse order
		service_shutdown_order = [
			'realtime_manager',
			'realtime_api',
			'event_bus',
			'monitoring',
			'integration',
			'templates',
			'workflow',
			'initialization',
			'migrations',
			'config'
		]
		
		for service_name in service_shutdown_order:
			service = self.services.get(service_name)
			if service and hasattr(service, 'stop'):
				try:
					await service.stop()
					logger.info(f"✅ {service_name} service stopped")
				except Exception as e:
					logger.error(f"❌ Error stopping {service_name} service: {e}")
		
		logger.info("✅ APG Workflow Orchestration shutdown completed")


# Global application instance
app = WorkflowOrchestrationApp()


async def main():
	"""Main application entry point."""
	try:
		await app.run()
	except KeyboardInterrupt:
		logger.info("Received keyboard interrupt")
	except Exception as e:
		logger.error(f"Application error: {e}")
		return 1
	
	return 0


if __name__ == "__main__":
	# Run the application
	exit_code = asyncio.run(main())
	sys.exit(exit_code)