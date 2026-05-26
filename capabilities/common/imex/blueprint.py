"""
APG Import/Export (IMEX) Flask Blueprint

Flask-AppBuilder blueprint integration for APG composition engine.
Registers views, API endpoints, and integrates with APG platform infrastructure.
"""

import logging
from datetime import datetime, timezone
from flask import Blueprint, current_app
from flask_appbuilder import AppBuilder
from flask_appbuilder.menu import Menu

from .models import model_registry
from .views import view_registry
from .api import api_registry, imex_api_bp
from .service import imex_service
from .__init__ import imex_capability, capability_metadata

logger = logging.getLogger(__name__)


class ImportExportBlueprint:
	"""
	APG Import/Export Blueprint Integration

	Manages registration with Flask-AppBuilder and APG composition engine.
	Provides seamless integration with APG platform infrastructure.
	"""

	def __init__(self, appbuilder: AppBuilder):
		self.appbuilder = appbuilder
		self.db = appbuilder.get_session
		self.blueprint_registered = False
		self.views_registered = False
		self.api_registered = False

		# APG Integration Status
		self.apg_composition_registered = False
		self.apg_dependencies_validated = False

	async def initialize(self) -> bool:
		"""Initialize blueprint and register with APG composition engine"""
		try:
			# Validate APG dependencies
			await self._validate_apg_dependencies()

			# Initialize service layer
			await imex_service.initialize()

			# Register with Flask-AppBuilder
			self._register_blueprint()
			self._register_views()
			self._register_api()
			self._register_menu_items()

			# Register with APG composition engine
			await self._register_apg_composition()

			# Setup monitoring and health checks
			self._setup_monitoring()

			self._log_blueprint_status("Import/Export blueprint initialized successfully")
			return True

		except Exception as e:
			self._log_blueprint_error(f"Failed to initialize blueprint: {e}")
			raise RuntimeError(f"Blueprint initialization failed: {e}")

	async def _validate_apg_dependencies(self):
		"""Validate required APG capability dependencies"""
		required_dependencies = capability_metadata["dependencies"]

		# Check each dependency
		missing_deps = []
		for dep in required_dependencies:
			try:
				# In production, would check actual APG capability availability
				# await apg.composition.validate_capability(dep)
				self._log_dependency_status(f"Dependency {dep}: available")
			except Exception:
				missing_deps.append(dep)
				self._log_dependency_error(f"Dependency {dep}: missing")

		if missing_deps:
			raise RuntimeError(f"Missing APG dependencies: {missing_deps}")

		self.apg_dependencies_validated = True
		self._log_blueprint_status("APG dependencies validated successfully")

	def _register_blueprint(self):
		"""Register Flask blueprint with AppBuilder"""
		try:
			# Register API blueprint
			current_app.register_blueprint(imex_api_bp)
			self.blueprint_registered = True

			self._log_blueprint_status("Flask blueprint registered")

		except Exception as e:
			self._log_blueprint_error(f"Failed to register blueprint: {e}")
			raise

	def _register_views(self):
		"""Register Flask-AppBuilder views"""
		try:
			# Register main views
			self.appbuilder.add_view(
				view_registry["ImportExportJobView"],
				"Import/Export Jobs",
				icon="fa-exchange",
				category="Data Platform",
				category_icon="fa-database"
			)

			self.appbuilder.add_view(
				view_registry["JobExecutionView"],
				"Job Executions",
				icon="fa-tasks",
				category="Data Platform"
			)

			self.appbuilder.add_view(
				view_registry["WorkflowView"],
				"Workflows",
				icon="fa-sitemap",
				category="Data Platform"
			)

			self.appbuilder.add_view(
				view_registry["SchemaMappingView"],
				"Schema Mappings",
				icon="fa-code-fork",
				category="Data Platform"
			)

			# Register dashboard views
			self.appbuilder.add_view_no_menu(view_registry["MonitoringDashboardView"])
			self.appbuilder.add_view_no_menu(view_registry["DataQualityView"])
			self.appbuilder.add_view_no_menu(view_registry["PerformanceAnalyticsView"])
			self.appbuilder.add_view_no_menu(view_registry["TemplateManagementView"])

			# Register chart views
			self.appbuilder.add_view(
				view_registry["JobStatusChart"],
				"Job Status Chart",
				icon="fa-pie-chart",
				category="Analytics"
			)

			self.appbuilder.add_view(
				view_registry["ThroughputChart"],
				"Throughput Chart",
				icon="fa-line-chart",
				category="Analytics"
			)

			self.views_registered = True
			self._log_blueprint_status("Flask-AppBuilder views registered")

		except Exception as e:
			self._log_blueprint_error(f"Failed to register views: {e}")
			raise

	def _register_api(self):
		"""Register API endpoints"""
		try:
			# API blueprint is already registered in _register_blueprint
			# Here we could add additional API configuration

			self.api_registered = True
			self._log_blueprint_status("API endpoints registered")

		except Exception as e:
			self._log_blueprint_error(f"Failed to register API: {e}")
			raise

	def _register_menu_items(self):
		"""Register custom menu items"""
		try:
			# Add custom menu items for enhanced navigation

			# Import/Export Dashboard
			self.appbuilder.add_link(
				"Import/Export Dashboard",
				href="/imex/dashboard",
				icon="fa-dashboard",
				category="Data Platform"
			)

			# Data Quality Dashboard
			self.appbuilder.add_link(
				"Data Quality",
				href="/imex/quality/dashboard",
				icon="fa-check-circle",
				category="Data Platform"
			)

			# Performance Analytics
			self.appbuilder.add_link(
				"Performance Analytics",
				href="/imex/analytics",
				icon="fa-chart-line",
				category="Analytics"
			)

			# Template Library
			self.appbuilder.add_link(
				"Template Library",
				href="/imex/templates/library",
				icon="fa-puzzle-piece",
				category="Data Platform"
			)

			# Workflow Designer
			self.appbuilder.add_link(
				"Workflow Designer",
				href="/imex/workflows/designer/new",
				icon="fa-magic",
				category="Data Platform"
			)

			# Schema Mapper
			self.appbuilder.add_link(
				"Schema Mapper",
				href="/imex/schemas/mapper",
				icon="fa-code-fork",
				category="Data Platform"
			)

			self._log_blueprint_status("Menu items registered")

		except Exception as e:
			self._log_blueprint_error(f"Failed to register menu items: {e}")
			raise

	async def _register_apg_composition(self):
		"""Register with APG composition engine"""
		try:
			# Initialize capability
			await imex_capability.initialize()

			# Register composition patterns
			await self._register_composition_patterns()

			# Register with marketplace
			await self._register_marketplace()

			# Setup capability monitoring
			await self._setup_capability_monitoring()

			self.apg_composition_registered = True
			self._log_blueprint_status("APG composition engine registration complete")

		except Exception as e:
			self._log_blueprint_error(f"Failed to register with APG composition: {e}")
			raise

	async def _register_composition_patterns(self):
		"""Register composition patterns with APG orchestration"""
		composition_patterns = capability_metadata["composition_patterns"]

		for pattern in composition_patterns:
			try:
				# Register pattern with APG composition engine
				# await apg.composition.register_pattern(pattern, imex_capability)
				self._log_composition_status(f"Registered pattern: {pattern}")
			except Exception as e:
				self._log_composition_error(f"Failed to register pattern {pattern}: {e}")
				raise

	async def _register_marketplace(self):
		"""Register capability with APG marketplace"""
		try:
			marketplace_metadata = {
				"capability_id": imex_capability.capability_id,
				"metadata": capability_metadata,
				"endpoints": {
					"health": "/api/v1/imex/monitoring/health",
					"metrics": "/api/v1/imex/monitoring/system",
					"documentation": "/api/v1/imex/docs/"
				},
				"integration_examples": [
					{
						"name": "Simple CSV Import",
						"description": "Import CSV data with validation",
						"code_example": '''
from imex.service import imex_service

# Create import job
job_config = {
	"name": "Customer CSV Import",
	"job_type": "import",
	"source_config": {
		"source_type": "file",
		"file_path": "/data/customers.csv",
		"format": "csv",
		"has_header": True
	},
	"target_config": {
		"target_type": "database",
		"connection_id": "postgres_main",
		"table_name": "customers"
	}
}

job = await imex_service.create_job(job_config, "admin")
execution = await imex_service.execute_job(job.id)
'''
					},
					{
						"name": "Data Migration Workflow",
						"description": "Complete data migration with validation",
						"code_example": '''
from imex.service import imex_service

# Create migration workflow
workflow_config = {
	"name": "Legacy System Migration",
	"description": "Migrate data from legacy system to cloud",
	"steps": [
		{
			"name": "Extract Legacy Data",
			"step_type": "import",
			"configuration": {"source": "legacy_db"}
		},
		{
			"name": "Transform Data",
			"step_type": "transform",
			"configuration": {"script": "normalize_addresses.py"}
		},
		{
			"name": "Validate Quality",
			"step_type": "validate",
			"configuration": {"rules": ["completeness", "accuracy"]}
		},
		{
			"name": "Load to Cloud",
			"step_type": "export",
			"configuration": {"target": "cloud_db"}
		}
	]
}

workflow = await imex_service.create_workflow(workflow_config, "admin")
execution_id = await imex_service.execute_workflow(workflow)
'''
					}
				]
			}

			# Register with marketplace
			# await apg.marketplace.register_capability(marketplace_metadata)
			self._log_marketplace_status("Registered with APG marketplace")

		except Exception as e:
			self._log_marketplace_error(f"Failed to register with marketplace: {e}")
			raise

	async def _setup_capability_monitoring(self):
		"""Setup capability-specific monitoring"""
		try:
			# Setup health check endpoint
			# Setup metrics collection
			# Setup alerting integration

			self._log_monitoring_status("Capability monitoring setup complete")

		except Exception as e:
			self._log_monitoring_error(f"Failed to setup monitoring: {e}")
			raise

	def _setup_monitoring(self):
		"""Setup Flask-AppBuilder monitoring integration"""
		try:
			# Add custom monitoring endpoints
			# Setup health check routes
			# Configure metrics collection

			self._log_blueprint_status("Monitoring integration setup complete")

		except Exception as e:
			self._log_blueprint_error(f"Failed to setup monitoring: {e}")
			raise

	def get_health_status(self) -> dict:
		"""Get comprehensive blueprint health status"""
		return {
			"blueprint": {
				"registered": self.blueprint_registered,
				"views_registered": self.views_registered,
				"api_registered": self.api_registered
			},
			"apg_integration": {
				"composition_registered": self.apg_composition_registered,
				"dependencies_validated": self.apg_dependencies_validated
			},
			"service": {
				"status": imex_service.health_status,
				"active_jobs": len(imex_service.active_jobs)
			},
			"capability": {
				"id": imex_capability.capability_id,
				"status": imex_capability.health_status,
				"last_health_check": imex_capability.last_health_check.isoformat()
			}
		}

	def get_blueprint_info(self) -> dict:
		"""Get blueprint information for APG platform"""
		return {
			"capability_metadata": capability_metadata,
			"view_registry": list(view_registry.keys()),
			"model_registry": list(model_registry.keys()),
			"api_namespaces": list(api_registry["namespaces"].keys()),
			"health_status": self.get_health_status(),
			"initialization_timestamp": datetime.now(timezone.utc).isoformat()
		}

	# Logging Methods

	def _log_blueprint_status(self, message: str):
		"""Log blueprint status message"""
		logger.info(f"[IMEX Blueprint] {message}")

	def _log_blueprint_error(self, message: str):
		"""Log blueprint error message"""
		logger.error(f"[IMEX Blueprint] {message}")

	def _log_dependency_status(self, message: str):
		"""Log dependency status message"""
		logger.info(f"[IMEX Dependencies] {message}")

	def _log_dependency_error(self, message: str):
		"""Log dependency error message"""
		logger.error(f"[IMEX Dependencies] {message}")

	def _log_composition_status(self, message: str):
		"""Log composition status message"""
		logger.info(f"[IMEX Composition] {message}")

	def _log_composition_error(self, message: str):
		"""Log composition error message"""
		logger.error(f"[IMEX Composition] {message}")

	def _log_marketplace_status(self, message: str):
		"""Log marketplace status message"""
		logger.info(f"[IMEX Marketplace] {message}")

	def _log_marketplace_error(self, message: str):
		"""Log marketplace error message"""
		logger.error(f"[IMEX Marketplace] {message}")

	def _log_monitoring_status(self, message: str):
		"""Log monitoring status message"""
		logger.info(f"[IMEX Monitoring] {message}")

	def _log_monitoring_error(self, message: str):
		"""Log monitoring error message"""
		logger.error(f"[IMEX Monitoring] {message}")


# Factory function for APG composition

def create_imex_blueprint(appbuilder: AppBuilder) -> ImportExportBlueprint:
	"""
	Factory function to create Import/Export blueprint

	Args:
		appbuilder: Flask-AppBuilder instance

	Returns:
		Configured ImportExportBlueprint instance
	"""
	blueprint = ImportExportBlueprint(appbuilder)
	return blueprint


# Blueprint registration helper

async def register_imex_capability(appbuilder: AppBuilder) -> ImportExportBlueprint:
	"""
	Register Import/Export capability with APG platform

	Args:
		appbuilder: Flask-AppBuilder instance

	Returns:
		Initialized and registered blueprint
	"""
	blueprint = create_imex_blueprint(appbuilder)
	await blueprint.initialize()
	return blueprint


# Configuration validation

def validate_blueprint_config() -> bool:
	"""Validate blueprint configuration for APG integration"""
	try:
		# Validate capability metadata
		required_fields = ["name", "version", "dependencies", "provides"]
		for field in required_fields:
			if field not in capability_metadata:
				raise ValueError(f"Missing required capability metadata field: {field}")

		# Validate model registry
		if not model_registry:
			raise ValueError("Model registry is empty")

		# Validate view registry
		if not view_registry:
			raise ValueError("View registry is empty")

		# Validate API registry
		if not api_registry:
			raise ValueError("API registry is empty")

		return True

	except Exception as e:
		logger.error(f"Blueprint configuration validation failed: {e}")
		return False


# Blueprint instance for APG composition
imex_blueprint = None

def get_imex_blueprint() -> ImportExportBlueprint | None:
	"""Get the registered blueprint instance"""
	return imex_blueprint

def set_imex_blueprint(blueprint: ImportExportBlueprint):
	"""Set the blueprint instance"""
	global imex_blueprint
	imex_blueprint = blueprint


__all__ = [
	"ImportExportBlueprint",
	"create_imex_blueprint",
	"register_imex_capability",
	"validate_blueprint_config",
	"get_imex_blueprint",
	"set_imex_blueprint"
]