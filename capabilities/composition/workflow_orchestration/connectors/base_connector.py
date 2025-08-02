"""
APG Workflow Orchestration Base Connector

Abstract base class for all external system connectors with standardized
interface, error handling, metrics collection, and APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

class ConnectorStatus(Enum):
	"""Connector status enumeration."""
	INITIALIZING = "initializing"
	CONNECTED = "connected"
	DISCONNECTED = "disconnected"
	ERROR = "error"
	MAINTENANCE = "maintenance"

@dataclass
class ConnectorMetrics:
	"""Connector performance and health metrics."""
	total_requests: int = 0
	successful_requests: int = 0
	failed_requests: int = 0
	average_response_time: float = 0.0
	last_request_time: Optional[datetime] = None
	connection_uptime: float = 0.0
	error_rate: float = 0.0
	throughput_per_second: float = 0.0
	
	def update_request_metrics(self, success: bool, response_time: float) -> None:
		"""Update request metrics."""
		self.total_requests += 1
		self.last_request_time = datetime.now(timezone.utc)
		
		if success:
			self.successful_requests += 1
		else:
			self.failed_requests += 1
		
		# Calculate rolling average response time
		self.average_response_time = (
			(self.average_response_time * (self.total_requests - 1) + response_time) / 
			self.total_requests
		)
		
		# Update error rate
		self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0

class ConnectorConfiguration(BaseModel):
	"""Base configuration for all connectors."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., min_length=1, max_length=200)
	description: str = Field("", max_length=1000)
	enabled: bool = Field(default=True)
	timeout_seconds: int = Field(default=30, ge=1, le=300)
	retry_attempts: int = Field(default=3, ge=0, le=10)
	retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
	health_check_interval_seconds: int = Field(default=60, ge=10, le=3600)
	tenant_id: str = Field(..., description="APG tenant identifier")
	user_id: str = Field(..., description="User who configured the connector")
	tags: Dict[str, str] = Field(default_factory=dict)
	custom_headers: Dict[str, str] = Field(default_factory=dict)
	environment: str = Field(default="production", regex="^(development|staging|production)$")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BaseConnector(ABC):
	"""Abstract base class for all external system connectors."""
	
	def __init__(self, config: ConnectorConfiguration):
		self.config = config
		self.status = ConnectorStatus.INITIALIZING
		self.metrics = ConnectorMetrics()
		self.connection_start_time = datetime.now(timezone.utc)
		self._health_check_task: Optional[asyncio.Task] = None
		self._event_callbacks: Dict[str, List[Callable]] = {
			"connected": [],
			"disconnected": [],
			"error": [],
			"request_completed": [],
			"health_check": []
		}
		
		logger.info(f"Initialized {self.__class__.__name__} connector: {config.name}")
	
	async def initialize(self) -> bool:
		"""Initialize the connector and establish connection."""
		try:
			await self._connect()
			self.status = ConnectorStatus.CONNECTED
			self.connection_start_time = datetime.now(timezone.utc)
			
			# Start health check monitoring
			self._health_check_task = asyncio.create_task(self._health_check_loop())
			
			await self._notify_event("connected", {"connector_id": self.config.id})
			logger.info(f"Successfully initialized connector: {self.config.name}")
			return True
			
		except Exception as e:
			self.status = ConnectorStatus.ERROR
			await self._notify_event("error", {"connector_id": self.config.id, "error": str(e)})
			logger.error(f"Failed to initialize connector {self.config.name}: {e}")
			return False
	
	async def disconnect(self) -> None:
		"""Disconnect from the external system and cleanup resources."""
		try:
			# Cancel health check task
			if self._health_check_task:
				self._health_check_task.cancel()
				await asyncio.gather(self._health_check_task, return_exceptions=True)
			
			await self._disconnect()
			self.status = ConnectorStatus.DISCONNECTED
			
			await self._notify_event("disconnected", {"connector_id": self.config.id})
			logger.info(f"Disconnected connector: {self.config.name}")
			
		except Exception as e:
			logger.error(f"Error during connector disconnect {self.config.name}: {e}")
	
	async def execute_request(
		self,
		operation: str,
		parameters: Dict[str, Any],
		timeout: Optional[int] = None
	) -> Dict[str, Any]:
		"""Execute a request against the external system."""
		if self.status != ConnectorStatus.CONNECTED:
			raise ConnectionError(f"Connector {self.config.name} is not connected")
		
		start_time = time.time()
		timeout = timeout or self.config.timeout_seconds
		success = False
		result = {}
		
		try:
			# Execute with retry logic
			for attempt in range(self.config.retry_attempts + 1):
				try:
					result = await asyncio.wait_for(
						self._execute_operation(operation, parameters),
						timeout=timeout
					)
					success = True
					break
					
				except asyncio.TimeoutError:
					if attempt < self.config.retry_attempts:
						await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
						continue
					raise
				except Exception as e:
					if attempt < self.config.retry_attempts:
						await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
						continue
					raise
			
			response_time = time.time() - start_time
			self.metrics.update_request_metrics(success, response_time)
			
			await self._notify_event("request_completed", {
				"connector_id": self.config.id,
				"operation": operation,
				"success": success,
				"response_time": response_time
			})
			
			return result
			
		except Exception as e:
			response_time = time.time() - start_time
			self.metrics.update_request_metrics(False, response_time)
			
			await self._notify_event("error", {
				"connector_id": self.config.id,
				"operation": operation,
				"error": str(e),
				"response_time": response_time
			})
			
			logger.error(f"Request failed for connector {self.config.name}: {e}")
			raise
	
	async def health_check(self) -> bool:
		"""Perform health check on the connector."""
		try:
			result = await self._health_check()
			
			if result:
				if self.status == ConnectorStatus.ERROR:
					self.status = ConnectorStatus.CONNECTED
					logger.info(f"Connector {self.config.name} recovered")
			else:
				if self.status == ConnectorStatus.CONNECTED:
					self.status = ConnectorStatus.ERROR
					logger.warning(f"Connector {self.config.name} health check failed")
			
			await self._notify_event("health_check", {
				"connector_id": self.config.id,
				"healthy": result
			})
			
			return result
			
		except Exception as e:
			self.status = ConnectorStatus.ERROR
			logger.error(f"Health check error for connector {self.config.name}: {e}")
			return False
	
	def get_metrics(self) -> Dict[str, Any]:
		"""Get connector performance metrics."""
		uptime = (datetime.now(timezone.utc) - self.connection_start_time).total_seconds()
		self.metrics.connection_uptime = uptime
		
		# Calculate throughput
		if uptime > 0:
			self.metrics.throughput_per_second = self.metrics.total_requests / uptime
		
		return {
			"connector_id": self.config.id,
			"name": self.config.name,
			"status": self.status.value,
			"total_requests": self.metrics.total_requests,
			"successful_requests": self.metrics.successful_requests,
			"failed_requests": self.metrics.failed_requests,
			"error_rate": self.metrics.error_rate,
			"average_response_time": self.metrics.average_response_time,
			"connection_uptime": self.metrics.connection_uptime,
			"throughput_per_second": self.metrics.throughput_per_second,
			"last_request_time": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None,
			"tenant_id": self.config.tenant_id
		}
	
	def add_event_callback(self, event_type: str, callback: Callable) -> None:
		"""Add event callback for connector events."""
		if event_type in self._event_callbacks:
			self._event_callbacks[event_type].append(callback)
	
	def remove_event_callback(self, event_type: str, callback: Callable) -> None:
		"""Remove event callback."""
		if event_type in self._event_callbacks and callback in self._event_callbacks[event_type]:
			self._event_callbacks[event_type].remove(callback)
	
	@abstractmethod
	async def _connect(self) -> None:
		"""Connect to the external system. Must be implemented by subclasses."""
		pass
	
	@abstractmethod
	async def _disconnect(self) -> None:
		"""Disconnect from the external system. Must be implemented by subclasses."""
		pass
	
	@abstractmethod
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute operation against the external system. Must be implemented by subclasses."""
		pass
	
	@abstractmethod
	async def _health_check(self) -> bool:
		"""Perform health check. Must be implemented by subclasses."""
		pass
	
	async def _health_check_loop(self) -> None:
		"""Background health check loop."""
		while True:
			try:
				await asyncio.sleep(self.config.health_check_interval_seconds)
				await self.health_check()
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Health check loop error for connector {self.config.name}: {e}")
				await asyncio.sleep(self.config.health_check_interval_seconds)
	
	async def _notify_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
		"""Notify event callbacks."""
		if event_type in self._event_callbacks:
			for callback in self._event_callbacks[event_type]:
				try:
					if asyncio.iscoroutinefunction(callback):
						await callback(event_data)
					else:
						callback(event_data)
				except Exception as e:
					logger.error(f"Event callback error for {event_type}: {e}")
	
	def _log_connector_info(self, message: str) -> str:
		"""Log connector information with standardized format."""
		return f"[{self.config.name}:{self.config.id[:8]}] {message}"

# Export base connector classes
__all__ = ["BaseConnector", "ConnectorConfiguration", "ConnectorStatus", "ConnectorMetrics"]