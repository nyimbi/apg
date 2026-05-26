"""
APG Connection Management Capability Composition API
Enables integration with other APG capabilities through standardized interfaces

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict

from .service import ConnectionManager
from .models import ConnectionType


class CapabilityType(str, Enum):
	"""Types of capabilities that can integrate with connection management"""
	DATA_PROCESSING = "data_processing"
	ANALYTICS = "analytics"
	MONITORING = "monitoring"
	WORKFLOW = "workflow"
	INTELLIGENCE = "intelligence"
	SECURITY = "security"
	STORAGE = "storage"
	TRANSFORMATION = "transformation"


class IntegrationMethod(str, Enum):
	"""Methods of capability integration"""
	EVENT_DRIVEN = "event_driven"
	API_CALL = "api_call"
	DATA_STREAM = "data_stream"
	BATCH_PROCESS = "batch_process"
	REAL_TIME = "real_time"


@dataclass
class CapabilityEvent:
	"""Event structure for capability communication"""
	event_id: str
	source_capability: str
	target_capability: Optional[str]
	event_type: str
	timestamp: str
	payload: Dict[str, Any]
	metadata: Dict[str, Any]
	correlation_id: Optional[str] = None


class CapabilityInterface(BaseModel):
	"""Interface definition for capability integration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	capability_id: str = Field(default_factory=uuid7str)
	name: str
	version: str
	capability_type: CapabilityType
	supported_methods: List[IntegrationMethod]
	endpoints: Dict[str, str]
	event_types: List[str]
	data_formats: List[str]
	requirements: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class CompositionContract(BaseModel):
	"""Contract defining how capabilities compose together"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	contract_id: str = Field(default_factory=uuid7str)
	source_capability: str
	target_capability: str
	integration_method: IntegrationMethod
	data_flow_direction: str  # "bidirectional", "source_to_target", "target_to_source"
	event_mappings: Dict[str, str]
	data_transformations: List[Dict[str, Any]]
	validation_rules: List[Dict[str, Any]]
	error_handling: Dict[str, Any]
	performance_requirements: Dict[str, Any] = Field(default_factory=dict)


class ICapabilityComposer(ABC):
	"""Abstract interface for capability composition"""

	@abstractmethod
	async def register_capability(self, interface: CapabilityInterface) -> bool:
		"""Register a capability for composition"""
		pass

	@abstractmethod
	async def create_composition(self, contract: CompositionContract) -> str:
		"""Create a new capability composition"""
		pass

	@abstractmethod
	async def execute_composition(self, composition_id: str, event: CapabilityEvent) -> Any:
		"""Execute a capability composition"""
		pass

	@abstractmethod
	async def validate_composition(self, contract: CompositionContract) -> List[str]:
		"""Validate a composition contract"""
		pass


class ConnectionCapabilityComposer(ICapabilityComposer):
	"""Connection Management capability composer implementation"""

	def __init__(self, connection_manager: ConnectionManager, tenant_id: str):
		self.connection_manager = connection_manager
		self.tenant_id = tenant_id
		self.registered_capabilities: Dict[str, CapabilityInterface] = {}
		self.active_compositions: Dict[str, CompositionContract] = {}
		self.event_handlers: Dict[str, List[Callable]] = {}

		# Register our own interface
		self.own_interface = self._create_connection_interface()

	def _create_connection_interface(self) -> CapabilityInterface:
		"""Create interface definition for connection management capability"""
		return CapabilityInterface(
			name="connection_management",
			version="1.0.0",
			capability_type=CapabilityType.DATA_PROCESSING,
			supported_methods=[
				IntegrationMethod.EVENT_DRIVEN,
				IntegrationMethod.API_CALL,
				IntegrationMethod.DATA_STREAM,
				IntegrationMethod.REAL_TIME
			],
			endpoints={
				"create_connection": "/api/connections/create",
				"test_connection": "/api/connections/test",
				"get_schema": "/api/connections/schema",
				"execute_flow": "/api/flows/execute",
				"get_lineage": "/api/lineage/get",
				"health_check": "/api/health"
			},
			event_types=[
				"connection.created",
				"connection.tested",
				"connection.failed",
				"flow.started",
				"flow.completed",
				"flow.failed",
				"schema.discovered",
				"lineage.updated"
			],
			data_formats=["json", "avro", "parquet", "csv"],
			requirements={
				"async_support": True,
				"tenant_isolation": True,
				"schema_validation": True
			},
			metadata={
				"supported_databases": list(ConnectionType.__members__.keys()),
				"max_concurrent_flows": 100,
				"supports_streaming": True
			}
		)

	async def register_capability(self, interface: CapabilityInterface) -> bool:
		"""Register a capability for composition"""
		try:
			# Validate interface
			validation_errors = await self._validate_interface(interface)
			if validation_errors:
				return False

			# Store interface
			self.registered_capabilities[interface.capability_id] = interface

			# Initialize event handlers for this capability
			for event_type in interface.event_types:
				if event_type not in self.event_handlers:
					self.event_handlers[event_type] = []

			return True
		except Exception as e:
			print(f"Failed to register capability: {e}")
			return False

	async def create_composition(self, contract: CompositionContract) -> str:
		"""Create a new capability composition"""
		try:
			# Validate contract
			validation_errors = await self.validate_composition(contract)
			if validation_errors:
				raise ValueError(f"Invalid composition contract: {validation_errors}")

			# Store contract
			self.active_compositions[contract.contract_id] = contract

			# Set up event routing
			await self._setup_event_routing(contract)

			return contract.contract_id
		except Exception as e:
			print(f"Failed to create composition: {e}")
			raise

	async def execute_composition(self, composition_id: str, event: CapabilityEvent) -> Any:
		"""Execute a capability composition"""
		try:
			contract = self.active_compositions.get(composition_id)
			if not contract:
				raise ValueError(f"Composition {composition_id} not found")

			# Route event based on contract
			result = await self._route_event(contract, event)

			# Apply data transformations
			if contract.data_transformations:
				result = await self._apply_transformations(result, contract.data_transformations)

			# Validate result
			await self._validate_result(result, contract.validation_rules)

			return result
		except Exception as e:
			print(f"Failed to execute composition: {e}")
			await self._handle_composition_error(composition_id, event, e)
			raise

	async def validate_composition(self, contract: CompositionContract) -> List[str]:
		"""Validate a composition contract"""
		errors = []

		# Check if capabilities exist
		if contract.source_capability not in self.registered_capabilities:
			errors.append(f"Source capability '{contract.source_capability}' not registered")

		if contract.target_capability not in self.registered_capabilities:
			errors.append(f"Target capability '{contract.target_capability}' not registered")

		if errors:
			return errors

		source_interface = self.registered_capabilities[contract.source_capability]
		target_interface = self.registered_capabilities[contract.target_capability]

		# Validate integration method compatibility
		if contract.integration_method not in source_interface.supported_methods:
			errors.append(f"Source capability doesn't support {contract.integration_method}")

		if contract.integration_method not in target_interface.supported_methods:
			errors.append(f"Target capability doesn't support {contract.integration_method}")

		# Validate event mappings
		for source_event, target_event in contract.event_mappings.items():
			if source_event not in source_interface.event_types:
				errors.append(f"Source event '{source_event}' not supported")
			if target_event not in target_interface.event_types:
				errors.append(f"Target event '{target_event}' not supported")

		return errors

	async def get_registered_capabilities(self) -> List[CapabilityInterface]:
		"""Get list of registered capabilities"""
		return list(self.registered_capabilities.values())

	async def get_active_compositions(self) -> List[CompositionContract]:
		"""Get list of active compositions"""
		return list(self.active_compositions.values())

	async def handle_connection_event(self, event_type: str, connection_id: str, data: Dict[str, Any]):
		"""Handle connection-related events and propagate to composed capabilities"""
		event = CapabilityEvent(
			event_id=uuid7str(),
			source_capability="connection_management",
			target_capability=None,
			event_type=event_type,
			timestamp=asyncio.get_event_loop().time(),
			payload={
				"connection_id": connection_id,
				"tenant_id": self.tenant_id,
				**data
			},
			metadata={
				"source": "connection_manager",
				"capability_type": "data_processing"
			}
		)

		# Propagate to all relevant compositions
		for composition_id, contract in self.active_compositions.items():
			if event_type in contract.event_mappings:
				try:
					await self.execute_composition(composition_id, event)
				except Exception as e:
					print(f"Failed to propagate event to composition {composition_id}: {e}")

	async def provide_connection_services(self, requesting_capability: str, service_type: str, parameters: Dict[str, Any]) -> Any:
		"""Provide connection services to other capabilities"""
		try:
			if service_type == "create_connection":
				return await self.connection_manager.create_connection(
					tenant_id=self.tenant_id,
					connection_data=parameters
				)

			elif service_type == "test_connection":
				return await self.connection_manager.test_connection(
					connection_id=parameters.get("connection_id")
				)

			elif service_type == "get_schema":
				return await self.connection_manager.discover_schema(
					connection_id=parameters.get("connection_id")
				)

			elif service_type == "execute_query":
				# Custom query execution service
				return await self._execute_custom_query(
					connection_id=parameters.get("connection_id"),
					query=parameters.get("query"),
					parameters=parameters.get("query_parameters", {})
				)

			else:
				raise ValueError(f"Unknown service type: {service_type}")

		except Exception as e:
			print(f"Failed to provide service {service_type} to {requesting_capability}: {e}")
			raise

	async def _validate_interface(self, interface: CapabilityInterface) -> List[str]:
		"""Validate a capability interface"""
		errors = []

		if not interface.name:
			errors.append("Capability name is required")

		if not interface.version:
			errors.append("Capability version is required")

		if not interface.supported_methods:
			errors.append("At least one integration method must be supported")

		return errors

	async def _setup_event_routing(self, contract: CompositionContract):
		"""Set up event routing for a composition contract"""
		for source_event, target_event in contract.event_mappings.items():
			if source_event not in self.event_handlers:
				self.event_handlers[source_event] = []

			# Create event handler for this mapping
			handler = lambda event, se=source_event, te=target_event, c=contract: self._transform_and_forward_event(event, se, te, c)
			self.event_handlers[source_event].append(handler)

	async def _route_event(self, contract: CompositionContract, event: CapabilityEvent) -> Any:
		"""Route an event according to composition contract"""
		# Get target event type
		target_event_type = contract.event_mappings.get(event.event_type)
		if not target_event_type:
			raise ValueError(f"No mapping for event type {event.event_type}")

		# Create target event
		target_event = CapabilityEvent(
			event_id=uuid7str(),
			source_capability=contract.source_capability,
			target_capability=contract.target_capability,
			event_type=target_event_type,
			timestamp=event.timestamp,
			payload=event.payload.copy(),
			metadata=event.metadata.copy(),
			correlation_id=event.event_id
		)

		# Execute based on integration method
		if contract.integration_method == IntegrationMethod.EVENT_DRIVEN:
			return await self._handle_event_driven_integration(target_event, contract)
		elif contract.integration_method == IntegrationMethod.API_CALL:
			return await self._handle_api_call_integration(target_event, contract)
		elif contract.integration_method == IntegrationMethod.DATA_STREAM:
			return await self._handle_data_stream_integration(target_event, contract)
		else:
			raise ValueError(f"Unsupported integration method: {contract.integration_method}")

	async def _apply_transformations(self, data: Any, transformations: List[Dict[str, Any]]) -> Any:
		"""Apply data transformations"""
		result = data

		for transformation in transformations:
			transform_type = transformation.get("type")

			if transform_type == "map_fields":
				result = await self._map_fields(result, transformation.get("mappings", {}))
			elif transform_type == "filter_data":
				result = await self._filter_data(result, transformation.get("conditions", []))
			elif transform_type == "aggregate":
				result = await self._aggregate_data(result, transformation.get("operations", []))
			elif transform_type == "validate":
				await self._validate_data(result, transformation.get("schema", {}))

		return result

	async def _validate_result(self, result: Any, validation_rules: List[Dict[str, Any]]):
		"""Validate composition result"""
		for rule in validation_rules:
			rule_type = rule.get("type")

			if rule_type == "required_fields":
				await self._validate_required_fields(result, rule.get("fields", []))
			elif rule_type == "data_types":
				await self._validate_data_types(result, rule.get("types", {}))
			elif rule_type == "value_ranges":
				await self._validate_value_ranges(result, rule.get("ranges", {}))

	async def _handle_composition_error(self, composition_id: str, event: CapabilityEvent, error: Exception):
		"""Handle errors in capability composition"""
		contract = self.active_compositions.get(composition_id)
		if not contract or not contract.error_handling:
			return

		error_strategy = contract.error_handling.get("strategy", "log")

		if error_strategy == "retry":
			retry_count = contract.error_handling.get("retry_count", 3)
			# Implement retry logic
			pass
		elif error_strategy == "fallback":
			fallback_action = contract.error_handling.get("fallback_action")
			# Implement fallback logic
			pass
		elif error_strategy == "notify":
			# Send error notification
			await self._send_error_notification(composition_id, event, error)

	async def _execute_custom_query(self, connection_id: str, query: str, parameters: Dict[str, Any]) -> Any:
		"""Execute a custom query for other capabilities"""
		# This would integrate with the actual connection to execute queries
		# Implementation depends on the specific connection type and requirements
		pass

	async def _transform_and_forward_event(self, event: CapabilityEvent, source_event: str, target_event: str, contract: CompositionContract):
		"""Transform and forward an event according to composition contract"""
		pass

	async def _handle_event_driven_integration(self, event: CapabilityEvent, contract: CompositionContract) -> Any:
		"""Handle event-driven integration"""
		pass

	async def _handle_api_call_integration(self, event: CapabilityEvent, contract: CompositionContract) -> Any:
		"""Handle API call integration"""
		pass

	async def _handle_data_stream_integration(self, event: CapabilityEvent, contract: CompositionContract) -> Any:
		"""Handle data stream integration"""
		pass

	async def _map_fields(self, data: Any, mappings: Dict[str, str]) -> Any:
		"""Map fields according to transformation rules"""
		pass

	async def _filter_data(self, data: Any, conditions: List[Dict[str, Any]]) -> Any:
		"""Filter data according to conditions"""
		pass

	async def _aggregate_data(self, data: Any, operations: List[Dict[str, Any]]) -> Any:
		"""Aggregate data according to operations"""
		pass

	async def _validate_data(self, data: Any, schema: Dict[str, Any]):
		"""Validate data against schema"""
		pass

	async def _validate_required_fields(self, data: Any, fields: List[str]):
		"""Validate required fields are present"""
		pass

	async def _validate_data_types(self, data: Any, types: Dict[str, str]):
		"""Validate data types"""
		pass

	async def _validate_value_ranges(self, data: Any, ranges: Dict[str, Any]):
		"""Validate value ranges"""
		pass

	async def _send_error_notification(self, composition_id: str, event: CapabilityEvent, error: Exception):
		"""Send error notification"""
		pass


class CompositionRegistry:
	"""Registry for managing capability compositions across the APG platform"""

	def __init__(self):
		self.composers: Dict[str, ICapabilityComposer] = {}
		self.global_compositions: Dict[str, CompositionContract] = {}

	def register_composer(self, capability_name: str, composer: ICapabilityComposer):
		"""Register a capability composer"""
		self.composers[capability_name] = composer

	async def create_cross_capability_composition(self, source_capability: str, target_capability: str, contract: CompositionContract) -> str:
		"""Create a composition between different capabilities"""
		if source_capability not in self.composers or target_capability not in self.composers:
			raise ValueError("One or both capabilities not registered")

		# Validate contract with both composers
		source_errors = await self.composers[source_capability].validate_composition(contract)
		target_errors = await self.composers[target_capability].validate_composition(contract)

		if source_errors or target_errors:
			raise ValueError(f"Contract validation failed: {source_errors + target_errors}")

		# Create composition in both composers
		source_id = await self.composers[source_capability].create_composition(contract)
		target_id = await self.composers[target_capability].create_composition(contract)

		# Store global reference
		self.global_compositions[contract.contract_id] = contract

		return contract.contract_id

	async def get_capability_interfaces(self) -> Dict[str, List[CapabilityInterface]]:
		"""Get all registered capability interfaces"""
		interfaces = {}
		for name, composer in self.composers.items():
			interfaces[name] = await composer.get_registered_capabilities()
		return interfaces