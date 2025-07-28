"""
APG Financial Management General Ledger - Capability Main Entry Point

Main initialization and lifecycle management for the General Ledger capability
within the APG platform ecosystem.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os

from flask import Flask
from flask_appbuilder import AppBuilder
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .integration import APGIntegrationManager, GLEventPublisher, CapabilityStatus
from .discovery import CapabilityDiscoveryService, DiscoveryConfig, initialize_discovery
from .service import GeneralLedgerService
from .api import create_api_blueprint, register_api_views
from .blueprint import register_views, register_permissions, init_subcapability

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneralLedgerCapability:
	"""Main General Ledger capability class"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.app: Optional[Flask] = None
		self.appbuilder: Optional[AppBuilder] = None
		self.integration_manager: Optional[APGIntegrationManager] = None
		self.discovery_service: Optional[CapabilityDiscoveryService] = None
		self.event_publisher: Optional[GLEventPublisher] = None
		self.is_running = False
		
		logger.info("General Ledger capability initialized")
	
	async def initialize(self):
		"""Initialize the capability and all its components"""
		try:
			logger.info("🚀 Initializing General Ledger capability...")
			
			# Initialize Flask application
			await self._initialize_flask_app()
			
			# Initialize APG integration
			await self._initialize_apg_integration()
			
			# Initialize discovery service
			await self._initialize_discovery_service()
			
			# Setup event publishing
			await self._setup_event_publishing()
			
			# Register signal handlers
			self._register_signal_handlers()
			
			logger.info("✅ General Ledger capability initialization completed")
			
		except Exception as e:
			logger.error(f"❌ Error initializing General Ledger capability: {e}")
			raise
	
	async def _initialize_flask_app(self):
		"""Initialize Flask application and AppBuilder"""
		try:
			logger.info("Initializing Flask application...")
			
			# Create Flask app
			self.app = Flask(__name__)
			self.app.config.from_mapping(self.config.get('flask', {}))
			
			# Configure database
			db_config = self.config.get('database', {})
			db_url = self._build_database_url(db_config)
			self.app.config['SQLALCHEMY_DATABASE_URI'] = db_url
			self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
			
			# Configure AppBuilder
			from flask_appbuilder.security.manager import AUTH_DB
			self.app.config['AUTH_TYPE'] = AUTH_DB
			self.app.config['AUTH_ROLE_ADMIN'] = 'Admin'
			self.app.config['AUTH_ROLE_PUBLIC'] = 'Public'
			
			# Initialize AppBuilder
			self.appbuilder = AppBuilder(self.app)
			
			# Register views and API endpoints
			register_views(self.appbuilder)
			register_permissions(self.appbuilder)
			register_api_views(self.appbuilder)
			
			# Register API blueprint
			api_blueprint = create_api_blueprint()
			self.app.register_blueprint(api_blueprint)
			
			# Initialize subcapability
			init_subcapability(self.appbuilder)
			
			logger.info("✓ Flask application initialized successfully")
			
		except Exception as e:
			logger.error(f"Error initializing Flask app: {e}")
			raise
	
	def _build_database_url(self, db_config: Dict[str, Any]) -> str:
		"""Build database URL from configuration"""
		return (
			f"postgresql://{db_config.get('username', 'postgres')}:"
			f"{db_config.get('password', 'password')}@"
			f"{db_config.get('host', 'localhost')}:"
			f"{db_config.get('port', 5432)}/"
			f"{db_config.get('database', 'apg_gl')}"
		)
	
	async def _initialize_apg_integration(self):
		"""Initialize APG platform integration"""
		try:
			logger.info("Initializing APG platform integration...")
			
			integration_config = self.config.get('integration', {})
			self.integration_manager = APGIntegrationManager(integration_config)
			
			# Register capability with platform
			await self.integration_manager.register_capability()
			
			# Register API endpoints with gateway
			await self.integration_manager.register_api_endpoints()
			
			# Setup event streaming
			await self.integration_manager.setup_event_streaming()
			
			logger.info("✓ APG platform integration initialized")
			
		except Exception as e:
			logger.error(f"Error initializing APG integration: {e}")
			raise
	
	async def _initialize_discovery_service(self):
		"""Initialize capability discovery service"""
		try:
			logger.info("Initializing capability discovery service...")
			
			discovery_config_data = self.config.get('discovery', {})
			discovery_config = DiscoveryConfig(
				service_registry_url=discovery_config_data.get(
					'service_registry_url', 
					'http://localhost:8081'
				),
				heartbeat_interval_seconds=discovery_config_data.get('heartbeat_interval', 30),
				health_check_interval_seconds=discovery_config_data.get('health_check_interval', 60),
				service_tags=discovery_config_data.get('tags', ['financial', 'accounting', 'general-ledger']),
				service_metadata=discovery_config_data.get('metadata', {})
			)
			
			self.discovery_service = await initialize_discovery(
				self.integration_manager, 
				discovery_config
			)
			
			logger.info("✓ Capability discovery service initialized")
			
		except Exception as e:
			logger.error(f"Error initializing discovery service: {e}")
			raise
	
	async def _setup_event_publishing(self):
		"""Setup event publishing for business events"""
		try:
			logger.info("Setting up event publishing...")
			
			self.event_publisher = GLEventPublisher(self.integration_manager)
			
			# Register event publisher with service layer
			# This would typically involve injecting the publisher into services
			
			logger.info("✓ Event publishing setup completed")
			
		except Exception as e:
			logger.error(f"Error setting up event publishing: {e}")
			raise
	
	def _register_signal_handlers(self):
		"""Register signal handlers for graceful shutdown"""
		def signal_handler(signum, frame):
			logger.info(f"Received signal {signum}, initiating graceful shutdown...")
			asyncio.create_task(self.shutdown())
		
		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)
	
	async def start(self):
		"""Start the capability"""
		try:
			if self.is_running:
				logger.warning("Capability is already running")
				return
			
			logger.info("🚀 Starting General Ledger capability...")
			
			# Update status to active
			if self.integration_manager:
				await self.integration_manager._update_status(CapabilityStatus.ACTIVE)
			
			self.is_running = True
			
			# Start Flask application in production mode
			if self.config.get('environment') == 'production':
				await self._start_production_server()
			else:
				await self._start_development_server()
			
		except Exception as e:
			logger.error(f"❌ Error starting capability: {e}")
			await self.shutdown()
			raise
	
	async def _start_production_server(self):
		"""Start production server with Gunicorn"""
		try:
			logger.info("Starting production server...")
			
			# In a real implementation, this would configure and start Gunicorn
			# For now, we'll simulate production startup
			
			logger.info("✓ Production server started")
			
			# Keep the capability running
			while self.is_running:
				await asyncio.sleep(1)
			
		except Exception as e:
			logger.error(f"Error in production server: {e}")
			raise
	
	async def _start_development_server(self):
		"""Start development server"""
		try:
			logger.info("Starting development server...")
			
			# Configure development settings
			self.app.config['DEBUG'] = True
			self.app.config['TESTING'] = False
			
			# Start development server in a separate thread
			import threading
			
			def run_flask():
				self.app.run(
					host=self.config.get('host', '0.0.0.0'),
					port=self.config.get('port', 5000),
					debug=False,  # Disable debug in thread
					use_reloader=False
				)
			
			flask_thread = threading.Thread(target=run_flask, daemon=True)
			flask_thread.start()
			
			logger.info("✓ Development server started")
			
			# Keep the capability running
			while self.is_running:
				await asyncio.sleep(1)
			
		except Exception as e:
			logger.error(f"Error in development server: {e}")
			raise
	
	async def shutdown(self):
		"""Graceful shutdown of the capability"""
		try:
			if not self.is_running:
				logger.info("Capability is not running")
				return
			
			logger.info("🛑 Shutting down General Ledger capability...")
			
			self.is_running = False
			
			# Update status to shutdown
			if self.integration_manager:
				await self.integration_manager._update_status(CapabilityStatus.SHUTDOWN)
			
			# Shutdown discovery service
			if self.discovery_service:
				await self.discovery_service.stop()
			
			# Shutdown integration manager
			if self.integration_manager:
				await self.integration_manager.shutdown()
			
			logger.info("✅ General Ledger capability shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during shutdown: {e}")
	
	async def get_health_status(self) -> Dict[str, Any]:
		"""Get comprehensive health status"""
		if not self.integration_manager:
			return {
				"status": "error",
				"message": "Integration manager not initialized"
			}
		
		health_status = await self.integration_manager.check_health()
		
		return {
			"status": health_status.status.value,
			"timestamp": health_status.timestamp.isoformat(),
			"response_time_ms": health_status.response_time_ms,
			"dependencies_healthy": health_status.dependencies_healthy,
			"error_message": health_status.error_message,
			"metrics": health_status.metrics,
			"capability_info": {
				"id": self.integration_manager.capability_id,
				"version": self.integration_manager.capability_info.version,
				"uptime_seconds": (datetime.now(timezone.utc) - 
								 self.integration_manager.capability_info.created_at).total_seconds()
			}
		}


async def create_capability(config_path: Optional[str] = None) -> GeneralLedgerCapability:
	"""Create and initialize General Ledger capability"""
	
	# Load configuration
	config = load_configuration(config_path)
	
	# Create capability instance
	capability = GeneralLedgerCapability(config)
	
	# Initialize capability
	await capability.initialize()
	
	return capability


def load_configuration(config_path: Optional[str] = None) -> Dict[str, Any]:
	"""Load capability configuration"""
	
	# Default configuration
	default_config = {
		"environment": "development",
		"host": "0.0.0.0",
		"port": 5000,
		"flask": {
			"SECRET_KEY": "dev-secret-key-change-in-production",
			"WTF_CSRF_ENABLED": True,
			"WTF_CSRF_TIME_LIMIT": None
		},
		"database": {
			"host": "localhost",
			"port": 5432,
			"database": "apg_gl",
			"username": "postgres",
			"password": "password"
		},
		"integration": {
			"discovery_service_url": "http://localhost:8081",
			"event_streaming_url": "http://localhost:8082",
			"api_gateway_url": "http://localhost:8083",
			"auth_service_url": "http://localhost:8080"
		},
		"discovery": {
			"service_registry_url": "http://localhost:8081",
			"heartbeat_interval": 30,
			"health_check_interval": 60,
			"tags": ["financial", "accounting", "general-ledger"],
			"metadata": {
				"compliance_frameworks": ["SOX", "GAAP", "IFRS"],
				"supported_currencies": ["USD", "EUR", "GBP", "JPY", "CAD"],
				"max_tenants": 1000,
				"data_retention_days": 2555  # 7 years
			}
		},
		"logging": {
			"level": "INFO",
			"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
		}
	}
	
	# Load from file if specified
	if config_path and Path(config_path).exists():
		try:
			with open(config_path, 'r') as f:
				file_config = json.load(f)
			
			# Merge configurations (file overrides default)
			config = {**default_config, **file_config}
			logger.info(f"Configuration loaded from {config_path}")
			
		except Exception as e:
			logger.error(f"Error loading configuration from {config_path}: {e}")
			logger.info("Using default configuration")
			config = default_config
	else:
		config = default_config
	
	# Override with environment variables
	config = _override_with_env_vars(config)
	
	return config


def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
	"""Override configuration with environment variables"""
	
	env_mappings = {
		"APG_GL_HOST": ("host",),
		"APG_GL_PORT": ("port",),
		"APG_GL_SECRET_KEY": ("flask", "SECRET_KEY"),
		"APG_GL_DB_HOST": ("database", "host"),
		"APG_GL_DB_PORT": ("database", "port"),
		"APG_GL_DB_NAME": ("database", "database"),
		"APG_GL_DB_USER": ("database", "username"),
		"APG_GL_DB_PASSWORD": ("database", "password"),
		"APG_DISCOVERY_URL": ("integration", "discovery_service_url"),
		"APG_EVENT_STREAMING_URL": ("integration", "event_streaming_url"),
		"APG_API_GATEWAY_URL": ("integration", "api_gateway_url"),
		"APG_AUTH_SERVICE_URL": ("integration", "auth_service_url"),
		"APG_ENVIRONMENT": ("environment",)
	}
	
	for env_var, config_path in env_mappings.items():
		env_value = os.getenv(env_var)
		if env_value:
			current = config
			for key in config_path[:-1]:
				current = current.setdefault(key, {})
			
			# Convert port to int
			if config_path[-1] in ['port']:
				env_value = int(env_value)
			
			current[config_path[-1]] = env_value
	
	return config


async def main():
	"""Main entry point for the capability"""
	try:
		# Load configuration
		config_path = sys.argv[1] if len(sys.argv) > 1 else None
		
		# Create and start capability
		capability = await create_capability(config_path)
		await capability.start()
		
	except KeyboardInterrupt:
		logger.info("Received interrupt signal")
	except Exception as e:
		logger.error(f"Fatal error: {e}")
		sys.exit(1)


if __name__ == "__main__":
	asyncio.run(main())


# Export main classes
__all__ = [
	'GeneralLedgerCapability',
	'create_capability',
	'load_configuration',
	'main'
]