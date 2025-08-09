"""
APG Financial Management General Ledger - Capability Discovery and Registration

Handles automatic capability discovery, registration, and lifecycle management
within the APG platform ecosystem.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import aiohttp
import socket
import psutil
from dataclasses import dataclass, asdict

from .integration import APGIntegrationManager, CapabilityStatus, CapabilityInfo
from .service import GeneralLedgerService

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
	"""Service endpoint information"""
	name: str
	url: str
	health_path: str
	timeout_seconds: int = 5
	retry_count: int = 3


@dataclass
class DiscoveryConfig:
	"""Discovery service configuration"""
	service_registry_url: str
	heartbeat_interval_seconds: int = 30
	health_check_interval_seconds: int = 60
	registration_retry_count: int = 5
	registration_retry_delay_seconds: int = 10
	auto_deregister_on_shutdown: bool = True
	service_tags: List[str] = None
	service_metadata: Dict[str, Any] = None


class CapabilityDiscoveryService:
	"""Manages capability discovery and registration with APG platform"""
	
	def __init__(self, integration_manager: APGIntegrationManager, config: DiscoveryConfig):
		self.integration_manager = integration_manager
		self.config = config
		self.session: Optional[aiohttp.ClientSession] = None
		self.registration_id: Optional[str] = None
		self.heartbeat_task: Optional[asyncio.Task] = None
		self.health_check_task: Optional[asyncio.Task] = None
		self.is_registered = False
		
		logger.info("Capability Discovery Service initialized")
	
	async def start(self):
		"""Start discovery service and register capability"""
		try:
			logger.info("Starting Capability Discovery Service")
			
			# Initialize HTTP session
			timeout = aiohttp.ClientTimeout(total=30)
			self.session = aiohttp.ClientSession(timeout=timeout)
			
			# Register capability
			await self._register_capability()
			
			# Start background tasks
			if self.is_registered:
				self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
				self.health_check_task = asyncio.create_task(self._health_check_loop())
				
				logger.info("✓ Capability Discovery Service started successfully")
			else:
				logger.error("Failed to start Discovery Service - registration failed")
				
		except Exception as e:
			logger.error(f"Error starting discovery service: {e}")
			raise
	
	async def stop(self):
		"""Stop discovery service and deregister capability"""
		try:
			logger.info("Stopping Capability Discovery Service")
			
			# Cancel background tasks
			if self.heartbeat_task:
				self.heartbeat_task.cancel()
				try:
					await self.heartbeat_task
				except asyncio.CancelledError:
					pass
			
			if self.health_check_task:
				self.health_check_task.cancel()
				try:
					await self.health_check_task
				except asyncio.CancelledError:
					pass
			
			# Deregister capability
			if self.config.auto_deregister_on_shutdown and self.is_registered:
				await self._deregister_capability()
			
			# Close HTTP session
			if self.session:
				await self.session.close()
			
			logger.info("✓ Capability Discovery Service stopped")
			
		except Exception as e:
			logger.error(f"Error stopping discovery service: {e}")
	
	async def _register_capability(self):
		"""Register capability with discovery service"""
		capability_info = self.integration_manager.capability_info
		
		registration_data = {
			"capability_id": capability_info.capability_id,
			"name": capability_info.name,
			"version": capability_info.version,
			"description": capability_info.description,
			"category": capability_info.category,
			"subcategory": capability_info.subcategory,
			"provider": capability_info.provider,
			"author": capability_info.author,
			"status": capability_info.status.value,
			"dependencies": capability_info.dependencies,
			"endpoints": {
				"api": capability_info.api_endpoints,
				"ui": capability_info.ui_routes,
				"health": capability_info.health_check_url,
				"metrics": capability_info.metrics_url,
				"docs": capability_info.documentation_url
			},
			"features": capability_info.features,
			"service_info": self._get_service_info(),
			"tags": self.config.service_tags or ["financial", "accounting", "general-ledger"],
			"metadata": {
				**(self.config.service_metadata or {}),
				"registration_timestamp": datetime.now(timezone.utc).isoformat(),
				"auto_discovery": True,
				"heartbeat_interval": self.config.heartbeat_interval_seconds,
				"health_check_interval": self.config.health_check_interval_seconds
			}
		}
		
		for attempt in range(self.config.registration_retry_count):
			try:
				logger.info(f"Attempting capability registration (attempt {attempt + 1}/{self.config.registration_retry_count})")
				
				async with self.session.post(
					f"{self.config.service_registry_url}/api/v1/capabilities/register",
					json=registration_data
				) as response:
					
					if response.status == 201:
						result = await response.json()
						self.registration_id = result.get('registration_id')
						self.is_registered = True
						
						logger.info(f"✓ Capability registered successfully with ID: {self.registration_id}")
						
						# Update capability status to active
						await self.integration_manager._update_status(CapabilityStatus.ACTIVE)
						
						return
					
					elif response.status == 409:
						# Already registered
						logger.info("Capability already registered, updating registration")
						await self._update_registration(registration_data)
						return
					
					else:
						error_text = await response.text()
						logger.error(f"Registration failed with status {response.status}: {error_text}")
				
			except Exception as e:
				logger.error(f"Registration attempt {attempt + 1} failed: {e}")
				
				if attempt < self.config.registration_retry_count - 1:
					await asyncio.sleep(self.config.registration_retry_delay_seconds)
				else:
					logger.error("All registration attempts failed")
					raise
	
	async def _update_registration(self, registration_data: Dict[str, Any]):
		"""Update existing capability registration"""
		try:
			async with self.session.put(
				f"{self.config.service_registry_url}/api/v1/capabilities/{self.integration_manager.capability_id}",
				json=registration_data
			) as response:
				
				if response.status == 200:
					result = await response.json()
					self.registration_id = result.get('registration_id')
					self.is_registered = True
					
					logger.info("✓ Capability registration updated successfully")
					
					# Update capability status to active
					await self.integration_manager._update_status(CapabilityStatus.ACTIVE)
				else:
					error_text = await response.text()
					logger.error(f"Registration update failed: {error_text}")
					
		except Exception as e:
			logger.error(f"Error updating registration: {e}")
			raise
	
	async def _deregister_capability(self):
		"""Deregister capability from discovery service"""
		try:
			if not self.registration_id:
				logger.warning("No registration ID available for deregistration")
				return
			
			async with self.session.delete(
				f"{self.config.service_registry_url}/api/v1/capabilities/{self.registration_id}"
			) as response:
				
				if response.status == 200:
					logger.info("✓ Capability deregistered successfully")
					self.is_registered = False
					self.registration_id = None
				else:
					error_text = await response.text()
					logger.error(f"Deregistration failed: {error_text}")
					
		except Exception as e:
			logger.error(f"Error during deregistration: {e}")
	
	async def _heartbeat_loop(self):
		"""Send periodic heartbeat to discovery service"""
		while True:
			try:
				await asyncio.sleep(self.config.heartbeat_interval_seconds)
				
				if not self.is_registered:
					continue
				
				heartbeat_data = {
					"capability_id": self.integration_manager.capability_id,
					"registration_id": self.registration_id,
					"status": self.integration_manager.status.value,
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"service_info": self._get_service_info()
				}
				
				async with self.session.post(
					f"{self.config.service_registry_url}/api/v1/capabilities/{self.registration_id}/heartbeat",
					json=heartbeat_data
				) as response:
					
					if response.status == 200:
						logger.debug("Heartbeat sent successfully")
					else:
						logger.warning(f"Heartbeat failed with status: {response.status}")
						
						# If heartbeat fails, try to re-register
						if response.status == 404:
							logger.info("Registration not found, attempting re-registration")
							self.is_registered = False
							await self._register_capability()
				
			except asyncio.CancelledError:
				logger.info("Heartbeat loop cancelled")
				break
			except Exception as e:
				logger.error(f"Error in heartbeat loop: {e}")
				await asyncio.sleep(5)  # Short delay before retry
	
	async def _health_check_loop(self):
		"""Perform periodic health checks and report status"""
		while True:
			try:
				await asyncio.sleep(self.config.health_check_interval_seconds)
				
				if not self.is_registered:
					continue
				
				# Perform health check
				health_status = await self.integration_manager.check_health()
				
				# Report health status
				health_data = {
					"capability_id": self.integration_manager.capability_id,
					"registration_id": self.registration_id,
					"status": health_status.status.value,
					"timestamp": health_status.timestamp.isoformat(),
					"response_time_ms": health_status.response_time_ms,
					"dependencies_healthy": health_status.dependencies_healthy,
					"error_message": health_status.error_message,
					"metrics": health_status.metrics
				}
				
				async with self.session.post(
					f"{self.config.service_registry_url}/api/v1/capabilities/{self.registration_id}/health",
					json=health_data
				) as response:
					
					if response.status == 200:
						logger.debug(f"Health status reported: {health_status.status.value}")
					else:
						logger.warning(f"Health report failed with status: {response.status}")
				
			except asyncio.CancelledError:
				logger.info("Health check loop cancelled")
				break
			except Exception as e:
				logger.error(f"Error in health check loop: {e}")
				await asyncio.sleep(10)  # Delay before retry
	
	def _get_service_info(self) -> Dict[str, Any]:
		"""Get current service information"""
		try:
			return {
				"hostname": socket.gethostname(),
				"ip_address": socket.gethostbyname(socket.gethostname()),
				"process_id": psutil.Process().pid,
				"cpu_percent": psutil.cpu_percent(interval=1),
				"memory_percent": psutil.virtual_memory().percent,
				"disk_usage_percent": psutil.disk_usage('/').percent,
				"uptime_seconds": (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds(),
				"load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
		except Exception as e:
			logger.error(f"Error gathering service info: {e}")
			return {
				"error": str(e),
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
	
	async def discover_dependencies(self) -> Dict[str, ServiceEndpoint]:
		"""Discover required service dependencies"""
		try:
			dependencies = self.integration_manager.capability_info.dependencies
			discovered_services = {}
			
			for dependency in dependencies:
				logger.info(f"Discovering service: {dependency}")
				
				async with self.session.get(
					f"{self.config.service_registry_url}/api/v1/capabilities/search",
					params={"capability_id": dependency, "status": "active"}
				) as response:
					
					if response.status == 200:
						result = await response.json()
						services = result.get('capabilities', [])
						
						if services:
							service = services[0]  # Use first available instance
							endpoint = ServiceEndpoint(
								name=dependency,
								url=service.get('base_url', ''),
								health_path=service.get('health_check_url', '/health')
							)
							discovered_services[dependency] = endpoint
							logger.info(f"✓ Discovered service: {dependency} at {endpoint.url}")
						else:
							logger.warning(f"Service not found: {dependency}")
					else:
						logger.error(f"Failed to discover service {dependency}: {response.status}")
			
			return discovered_services
			
		except Exception as e:
			logger.error(f"Error discovering dependencies: {e}")
			return {}
	
	async def wait_for_dependencies(self, max_wait_seconds: int = 300) -> bool:
		"""Wait for required dependencies to become available"""
		try:
			logger.info("Waiting for dependencies to become available")
			
			dependencies = await self.discover_dependencies()
			start_time = datetime.now()
			
			while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
				all_healthy = True
				
				for name, endpoint in dependencies.items():
					try:
						async with self.session.get(
							f"{endpoint.url}{endpoint.health_path}",
							timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
						) as response:
							
							if response.status == 200:
								logger.debug(f"✓ Dependency healthy: {name}")
							else:
								logger.warning(f"Dependency unhealthy: {name} (status: {response.status})")
								all_healthy = False
								
					except Exception as e:
						logger.warning(f"Cannot reach dependency {name}: {e}")
						all_healthy = False
				
				if all_healthy:
					logger.info("✓ All dependencies are healthy")
					return True
				
				logger.info("Waiting for dependencies...")
				await asyncio.sleep(10)
			
			logger.error(f"Timeout waiting for dependencies after {max_wait_seconds} seconds")
			return False
			
		except Exception as e:
			logger.error(f"Error waiting for dependencies: {e}")
			return False


async def initialize_discovery(integration_manager: APGIntegrationManager, 
							   discovery_config: DiscoveryConfig) -> CapabilityDiscoveryService:
	"""Initialize and start capability discovery service"""
	discovery_service = CapabilityDiscoveryService(integration_manager, discovery_config)
	
	try:
		# Wait for dependencies first
		dependencies_ready = await discovery_service.wait_for_dependencies()
		
		if not dependencies_ready:
			logger.warning("Starting discovery service without all dependencies ready")
		
		# Start discovery service
		await discovery_service.start()
		
		return discovery_service
		
	except Exception as e:
		logger.error(f"Failed to initialize discovery service: {e}")
		await discovery_service.stop()
		raise


# Export classes for external use
__all__ = [
	'ServiceEndpoint',
	'DiscoveryConfig',
	'CapabilityDiscoveryService',
	'initialize_discovery'
]