#!/usr/bin/env python3
"""
APG Financial Management General Ledger - Capability Runner

Production startup script for the General Ledger capability with
comprehensive initialization, health checks, and lifecycle management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import sys
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from capability import create_capability, GeneralLedgerCapability
from integration import CapabilityStatus

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CapabilityRunner:
	"""Manages the lifecycle of the General Ledger capability"""
	
	def __init__(self):
		self.capability: Optional[GeneralLedgerCapability] = None
		self.shutdown_event = asyncio.Event()
		
	async def run(self, config_path: Optional[str] = None, wait_for_dependencies: bool = True):
		"""Run the capability with proper lifecycle management"""
		try:
			logger.info("🚀 Starting APG Financial Management General Ledger capability...")
			
			# Create and initialize capability
			self.capability = await create_capability(config_path)
			
			# Wait for dependencies if requested
			if wait_for_dependencies:
				await self._wait_for_dependencies()
			
			# Register signal handlers
			self._register_signal_handlers()
			
			# Start the capability
			logger.info("Starting capability services...")
			await self.capability.start()
			
			# Wait for shutdown signal
			await self.shutdown_event.wait()
			
		except KeyboardInterrupt:
			logger.info("Received interrupt signal")
		except Exception as e:
			logger.error(f"Fatal error: {e}")
			raise
		finally:
			await self._cleanup()
	
	async def _wait_for_dependencies(self):
		"""Wait for required dependencies to be available"""
		try:
			logger.info("Checking dependencies...")
			
			if self.capability and self.capability.discovery_service:
				dependencies_ready = await self.capability.discovery_service.wait_for_dependencies(
					max_wait_seconds=300
				)
				
				if dependencies_ready:
					logger.info("✅ All dependencies are ready")
				else:
					logger.warning("⚠️  Some dependencies are not ready - continuing anyway")
			
		except Exception as e:
			logger.error(f"Error checking dependencies: {e}")
			# Continue anyway - the capability can handle missing dependencies
	
	def _register_signal_handlers(self):
		"""Register signal handlers for graceful shutdown"""
		def signal_handler(signum, frame):
			logger.info(f"Received signal {signum}, initiating graceful shutdown...")
			asyncio.create_task(self._shutdown())
		
		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)
		
		# Unix-specific signals
		if hasattr(signal, 'SIGHUP'):
			signal.signal(signal.SIGHUP, signal_handler)
		if hasattr(signal, 'SIGUSR1'):
			signal.signal(signal.SIGUSR1, self._handle_status_signal)
	
	def _handle_status_signal(self, signum, frame):
		"""Handle status signal (SIGUSR1) to print current status"""
		asyncio.create_task(self._print_status())
	
	async def _print_status(self):
		"""Print current capability status"""
		try:
			if self.capability:
				health_status = await self.capability.get_health_status()
				
				logger.info("📊 Capability Status Report:")
				logger.info(f"   Status: {health_status['status']}")
				logger.info(f"   Response Time: {health_status['response_time_ms']:.2f}ms")
				logger.info(f"   Dependencies Healthy: {health_status['dependencies_healthy']}")
				
				if health_status.get('metrics'):
					metrics = health_status['metrics']
					logger.info(f"   Active Tenants: {metrics.get('active_tenants', 'N/A')}")
					logger.info(f"   Total Accounts: {metrics.get('total_accounts', 'N/A')}")
					logger.info(f"   Daily Journal Entries: {metrics.get('daily_journal_entries', 'N/A')}")
			else:
				logger.info("📊 Capability not initialized")
				
		except Exception as e:
			logger.error(f"Error printing status: {e}")
	
	async def _shutdown(self):
		"""Initiate graceful shutdown"""
		logger.info("🛑 Initiating graceful shutdown...")
		self.shutdown_event.set()
	
	async def _cleanup(self):
		"""Cleanup resources"""
		try:
			if self.capability:
				await self.capability.shutdown()
			logger.info("✅ Cleanup completed")
		except Exception as e:
			logger.error(f"Error during cleanup: {e}")


async def health_check(config_path: Optional[str] = None) -> bool:
	"""Perform a health check on the capability"""
	try:
		logger.info("🔍 Performing health check...")
		
		# Create capability instance
		capability = await create_capability(config_path)
		
		# Get health status
		health_status = await capability.get_health_status()
		
		status = health_status['status']
		response_time = health_status['response_time_ms']
		
		if status == CapabilityStatus.ACTIVE.value:
			logger.info(f"✅ Health check passed - Status: {status}, Response time: {response_time:.2f}ms")
			return True
		else:
			logger.warning(f"⚠️  Health check warning - Status: {status}, Response time: {response_time:.2f}ms")
			return False
		
	except Exception as e:
		logger.error(f"❌ Health check failed: {e}")
		return False
	finally:
		if 'capability' in locals():
			await capability.shutdown()


async def validate_config(config_path: str) -> bool:
	"""Validate configuration file"""
	try:
		logger.info(f"🔍 Validating configuration: {config_path}")
		
		config_file = Path(config_path)
		if not config_file.exists():
			logger.error(f"Configuration file does not exist: {config_path}")
			return False
		
		# Try to load configuration
		from capability import load_configuration
		config = load_configuration(config_path)
		
		# Validate required sections
		required_sections = ['database', 'integration', 'discovery']
		for section in required_sections:
			if section not in config:
				logger.error(f"Missing required configuration section: {section}")
				return False
		
		# Validate database configuration
		db_config = config['database']
		required_db_fields = ['host', 'port', 'database', 'username', 'password']
		for field in required_db_fields:
			if field not in db_config:
				logger.error(f"Missing required database field: {field}")
				return False
		
		logger.info("✅ Configuration validation passed")
		return True
		
	except Exception as e:
		logger.error(f"❌ Configuration validation failed: {e}")
		return False


def main():
	"""Main entry point with command line argument parsing"""
	parser = argparse.ArgumentParser(
		description='APG Financial Management General Ledger Capability',
		epilog='''
Examples:
  %(prog)s                              # Run with default configuration
  %(prog)s --config config.json         # Run with custom configuration
  %(prog)s --health-check               # Perform health check only
  %(prog)s --validate-config config.json # Validate configuration file
		''',
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	
	parser.add_argument(
		'--config',
		type=str,
		help='Path to configuration file (default: config.json)'
	)
	
	parser.add_argument(
		'--health-check',
		action='store_true',
		help='Perform health check and exit'
	)
	
	parser.add_argument(
		'--validate-config',
		type=str,
		help='Validate configuration file and exit'
	)
	
	parser.add_argument(
		'--no-wait-dependencies',
		action='store_true',
		help='Do not wait for dependencies to be ready'
	)
	
	parser.add_argument(
		'--log-level',
		choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
		default='INFO',
		help='Set logging level (default: INFO)'
	)
	
	args = parser.parse_args()
	
	# Set logging level
	logging.getLogger().setLevel(getattr(logging, args.log_level))
	
	# Handle validation mode
	if args.validate_config:
		success = asyncio.run(validate_config(args.validate_config))
		sys.exit(0 if success else 1)
	
	# Handle health check mode
	if args.health_check:
		success = asyncio.run(health_check(args.config))
		sys.exit(0 if success else 1)
	
	# Normal run mode
	runner = CapabilityRunner()
	
	try:
		asyncio.run(runner.run(
			config_path=args.config,
			wait_for_dependencies=not args.no_wait_dependencies
		))
	except KeyboardInterrupt:
		logger.info("Received interrupt, exiting...")
		sys.exit(0)
	except Exception as e:
		logger.error(f"Fatal error: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()