"""
APG Integration API Management - Gateway Runner

Main application runner that initializes and starts the API gateway with
full monitoring, configuration management, and graceful shutdown capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import signal
import sys
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

import aioredis
import uvloop
from aiohttp import web
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import (
	create_configuration, APIManagementSettings, Environment,
	get_database_url, get_redis_url
)
from .models import Base
from .service import (
	APILifecycleService, ConsumerManagementService,
	PolicyManagementService, AnalyticsService
)
from .gateway import APIGateway
from .monitoring import MetricsCollector, HealthMonitor, AlertManager
from .api import register_api_endpoints

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(config: APIManagementSettings):
	"""Setup application logging."""
	
	# Configure logging format
	if config.monitoring.log_format == "json":
		import json_logging
		json_logging.init_flask(enable_json=True)
		json_logging.init_request_instrument()
		
		formatter = json_logging.JSONLogFormatter()
	else:
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		)
	
	# Configure root logger
	root_logger = logging.getLogger()
	root_logger.setLevel(getattr(logging, config.monitoring.log_level.value))
	
	# Console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(formatter)
	root_logger.addHandler(console_handler)
	
	# File handler if configured
	if config.monitoring.log_file:
		from logging.handlers import RotatingFileHandler
		
		file_handler = RotatingFileHandler(
			config.monitoring.log_file,
			maxBytes=config.monitoring.log_max_size_mb * 1024 * 1024,
			backupCount=config.monitoring.log_backup_count
		)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)
	
	# Configure specific loggers
	logging.getLogger('aiohttp.access').setLevel(logging.INFO)
	logging.getLogger('aioredis').setLevel(logging.WARNING)
	logging.getLogger('sqlalchemy.engine').setLevel(
		logging.INFO if config.database.echo else logging.WARNING
	)

# =============================================================================
# Database Initialization
# =============================================================================

async def initialize_database(config: APIManagementSettings):
	"""Initialize database connection and create tables."""
	
	logger = logging.getLogger(__name__)
	
	try:
		# Create database engine
		engine = create_engine(
			get_database_url(config),
			pool_size=config.database.pool_size,
			max_overflow=config.database.max_overflow,
			pool_timeout=config.database.pool_timeout,
			pool_recycle=config.database.pool_recycle,
			echo=config.database.echo
		)
		
		# Create all tables
		Base.metadata.create_all(engine)
		
		# Test connection
		with engine.connect() as conn:
			conn.execute("SELECT 1")
		
		logger.info("Database initialized successfully")
		return engine
		
	except Exception as e:
		logger.error(f"Database initialization failed: {e}")
		raise

# =============================================================================
# Service Initialization
# =============================================================================

async def initialize_services(config: APIManagementSettings, 
							  database_engine, redis_client) -> Dict[str, Any]:
	"""Initialize all application services."""
	
	logger = logging.getLogger(__name__)
	
	try:
		# Create session factory
		Session = sessionmaker(bind=database_engine)
		
		# Initialize core services
		services = {
			'api_lifecycle': APILifecycleService(),
			'consumer_management': ConsumerManagementService(),
			'policy_management': PolicyManagementService(),
			'analytics': AnalyticsService()
		}
		
		# Initialize monitoring services
		metrics_collector = MetricsCollector(redis_client)
		health_monitor = HealthMonitor(redis_client, services['analytics'], metrics_collector)
		alert_manager = AlertManager(redis_client)
		
		monitoring_services = {
			'metrics_collector': metrics_collector,
			'health_monitor': health_monitor,
			'alert_manager': alert_manager
		}
		
		# Add alert handlers
		if config.monitoring.alert_email_enabled:
			alert_manager.add_notification_channel(
				create_email_alert_handler(config)
			)
		
		if config.monitoring.alert_slack_enabled:
			alert_manager.add_notification_channel(
				create_slack_alert_handler(config)
			)
		
		logger.info("Services initialized successfully")
		
		return {
			**services,
			**monitoring_services,
			'session_factory': Session
		}
		
	except Exception as e:
		logger.error(f"Service initialization failed: {e}")
		raise

def create_email_alert_handler(config: APIManagementSettings):
	"""Create email alert handler."""
	
	async def send_email_alert(alert_data: Dict[str, Any]):
		"""Send alert via email."""
		# Implementation would use SMTP to send alert emails
		logger = logging.getLogger(__name__)
		logger.info(f"EMAIL ALERT: {alert_data.get('message', 'Unknown alert')}")
	
	return send_email_alert

def create_slack_alert_handler(config: APIManagementSettings):
	"""Create Slack alert handler."""
	
	async def send_slack_alert(alert_data: Dict[str, Any]):
		"""Send alert via Slack webhook."""
		# Implementation would use Slack webhook to send alerts
		logger = logging.getLogger(__name__)
		logger.info(f"SLACK ALERT: {alert_data.get('message', 'Unknown alert')}")
	
	return send_slack_alert

# =============================================================================
# Gateway Application
# =============================================================================

class GatewayApplication:
	"""Main gateway application with lifecycle management."""
	
	def __init__(self, config: APIManagementSettings):
		self.config = config
		self.logger = logging.getLogger(__name__)
		
		# Components
		self.database_engine = None
		self.redis_client = None
		self.services = {}
		self.gateway = None
		self.gateway_runner = None
		
		# Shutdown flag
		self.shutdown_event = asyncio.Event()
		
	async def initialize(self):
		"""Initialize all application components."""
		
		self.logger.info("Initializing API Gateway Application...")
		
		try:
			# Initialize database
			self.database_engine = await initialize_database(self.config)
			
			# Initialize Redis
			self.redis_client = aioredis.from_url(get_redis_url(self.config))
			await self.redis_client.ping()
			self.logger.info("Redis connection established")
			
			# Initialize services
			self.services = await initialize_services(
				self.config, self.database_engine, self.redis_client
			)
			
			# Initialize API gateway
			self.gateway = APIGateway(
				host=self.config.gateway.host,
				port=self.config.gateway.port,
				redis_url=get_redis_url(self.config),
				database_url=get_database_url(self.config)
			)
			
			# Setup signal handlers
			self._setup_signal_handlers()
			
			self.logger.info("Application initialization complete")
			
		except Exception as e:
			self.logger.error(f"Application initialization failed: {e}")
			await self.cleanup()
			raise
	
	async def start(self):
		"""Start the gateway application."""
		
		self.logger.info("Starting API Gateway Application...")
		
		try:
			# Start monitoring services
			if self.config.monitoring.health_check_enabled:
				await self.services['health_monitor'].start_monitoring()
				self.logger.info("Health monitoring started")
			
			# Start the gateway server
			self.gateway_runner = await self.gateway.start()
			self.logger.info(f"API Gateway started on {self.config.gateway.host}:{self.config.gateway.port}")
			
			# Log startup information
			self._log_startup_info()
			
			# Wait for shutdown signal
			await self.shutdown_event.wait()
			
		except Exception as e:
			self.logger.error(f"Application startup failed: {e}")
			raise
		finally:
			await self.cleanup()
	
	async def cleanup(self):
		"""Cleanup application resources."""
		
		self.logger.info("Shutting down API Gateway Application...")
		
		try:
			# Stop monitoring services
			if 'health_monitor' in self.services:
				await self.services['health_monitor'].stop_monitoring()
			
			# Stop gateway server
			if self.gateway_runner:
				await self.gateway.stop(self.gateway_runner)
			
			# Close Redis connection
			if self.redis_client:
				await self.redis_client.close()
			
			# Close database connections
			if self.database_engine:
				self.database_engine.dispose()
			
			self.logger.info("Application shutdown complete")
			
		except Exception as e:
			self.logger.error(f"Error during cleanup: {e}")
	
	def _setup_signal_handlers(self):
		"""Setup signal handlers for graceful shutdown."""
		
		def signal_handler(signum, frame):
			self.logger.info(f"Received signal {signum}, initiating shutdown...")
			asyncio.create_task(self._shutdown())
		
		signal.signal(signal.SIGTERM, signal_handler)
		signal.signal(signal.SIGINT, signal_handler)
		
		if hasattr(signal, 'SIGHUP'):
			signal.signal(signal.SIGHUP, signal_handler)
	
	async def _shutdown(self):
		"""Initiate graceful shutdown."""
		self.shutdown_event.set()
	
	def _log_startup_info(self):
		"""Log application startup information."""
		
		self.logger.info("=" * 80)
		self.logger.info(f"APG Integration API Management v{self.config.app_version}")
		self.logger.info(f"Environment: {self.config.environment.value}")
		self.logger.info(f"Gateway URL: http://{self.config.gateway.host}:{self.config.gateway.port}")
		self.logger.info(f"Health Check: http://{self.config.gateway.host}:{self.config.gateway.port}/health")
		self.logger.info(f"Multi-tenant: {'Enabled' if self.config.multi_tenant_enabled else 'Disabled'}")
		self.logger.info(f"Features: {', '.join([k for k, v in self.config.features.items() if v])}")
		self.logger.info("=" * 80)

# =============================================================================
# Command Line Interface
# =============================================================================

async def run_gateway(environment: Optional[Environment] = None,
					 config_dir: str = "config",
					 debug: bool = False):
	"""Run the API gateway application."""
	
	# Load configuration
	config = create_configuration(environment=environment, config_dir=config_dir)
	
	if debug:
		config.debug = True
		config.monitoring.log_level = "DEBUG"
	
	# Setup logging
	setup_logging(config)
	
	# Create and run application
	app = GatewayApplication(config)
	
	try:
		await app.initialize()
		await app.start()
	except KeyboardInterrupt:
		logging.getLogger(__name__).info("Received keyboard interrupt")
	except Exception as e:
		logging.getLogger(__name__).error(f"Application error: {e}")
		logging.getLogger(__name__).debug(traceback.format_exc())
		sys.exit(1)

def main():
	"""Main entry point for the gateway application."""
	
	import argparse
	
	parser = argparse.ArgumentParser(
		description="APG Integration API Management Gateway"
	)
	
	parser.add_argument(
		"--environment", "-e",
		type=str,
		choices=[env.value for env in Environment],
		help="Deployment environment"
	)
	
	parser.add_argument(
		"--config-dir", "-c",
		type=str,
		default="config",
		help="Configuration directory path"
	)
	
	parser.add_argument(
		"--debug", "-d",
		action="store_true",
		help="Enable debug mode"
	)
	
	parser.add_argument(
		"--generate-config",
		action="store_true",
		help="Generate configuration templates and exit"
	)
	
	args = parser.parse_args()
	
	# Generate configuration templates if requested
	if args.generate_config:
		from .config import generate_config_templates
		generate_config_templates(args.config_dir)
		return
	
	# Parse environment
	environment = None
	if args.environment:
		environment = Environment(args.environment)
	
	# Use uvloop for better performance
	if sys.platform != 'win32':
		uvloop.install()
	
	# Run the gateway
	asyncio.run(run_gateway(
		environment=environment,
		config_dir=args.config_dir,
		debug=args.debug
	))

# =============================================================================
# Development and Testing Utilities
# =============================================================================

async def run_development_server():
	"""Run development server with hot reloading."""
	
	config = create_configuration(Environment.DEVELOPMENT)
	config.debug = True
	config.monitoring.log_level = "DEBUG"
	
	setup_logging(config)
	
	app = GatewayApplication(config)
	await app.initialize()
	await app.start()

async def run_health_check():
	"""Run health check and exit."""
	
	import aiohttp
	
	config = create_configuration()
	
	url = f"http://{config.gateway.host}:{config.gateway.port}/health"
	
	try:
		async with aiohttp.ClientSession() as session:
			async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
				health_data = await response.json()
				
				print(f"Health Check Status: {health_data.get('status', 'unknown')}")
				print(f"Timestamp: {health_data.get('timestamp', 'unknown')}")
				print(f"Version: {health_data.get('version', 'unknown')}")
				
				if health_data.get('checks'):
					print("\nComponent Health:")
					for component, status in health_data['checks'].items():
						print(f"  {component}: {status}")
				
				sys.exit(0 if health_data.get('status') == 'healthy' else 1)
				
	except Exception as e:
		print(f"Health check failed: {e}")
		sys.exit(1)

def run_load_test(target_rps: int = 1000, duration_seconds: int = 60):
	"""Run basic load test against the gateway."""
	
	import aiohttp
	import time
	
	async def load_test():
		config = create_configuration()
		base_url = f"http://{config.gateway.host}:{config.gateway.port}"
		
		start_time = time.time()
		request_count = 0
		error_count = 0
		
		async with aiohttp.ClientSession() as session:
			while time.time() - start_time < duration_seconds:
				try:
					async with session.get(f"{base_url}/health") as response:
						if response.status >= 400:
							error_count += 1
						request_count += 1
				except Exception:
					error_count += 1
				
				# Control rate
				await asyncio.sleep(1.0 / target_rps)
		
		elapsed = time.time() - start_time
		actual_rps = request_count / elapsed
		error_rate = (error_count / request_count) * 100 if request_count > 0 else 0
		
		print(f"Load Test Results:")
		print(f"  Duration: {elapsed:.2f} seconds")
		print(f"  Total Requests: {request_count}")
		print(f"  Actual RPS: {actual_rps:.2f}")
		print(f"  Error Count: {error_count}")
		print(f"  Error Rate: {error_rate:.2f}%")
	
	asyncio.run(load_test())

# =============================================================================
# Export Runner Components
# =============================================================================

__all__ = [
	'GatewayApplication',
	'run_gateway',
	'main',
	'run_development_server',
	'run_health_check',
	'run_load_test'
]

if __name__ == "__main__":
	main()