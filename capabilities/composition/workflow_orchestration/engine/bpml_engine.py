"""
APG Workflow Orchestration BPML Engine

Comprehensive Business Process Modeling Language (BPML) parser and execution engine
with support for full BPML 1.0 specification and simplified variants.

© 2025 Datacraft. All rights reserved.
Author: APG Development Team
"""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from uuid_extensions import uuid7str
import json
import logging

from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, TaskType, Priority,
	_log_workflow_operation, _log_task_execution, _log_audit_event
)

logger = logging.getLogger(__name__)

class BPMLVersion(str, Enum):
	"""Supported BPML versions."""
	FULL_1_0 = "1.0"
	SIMPLIFIED = "simplified"
	APG_EXTENDED = "apg-extended"

class BPMLElementType(str, Enum):
	"""BPML element types."""
	PROCESS = "process"
	ACTIVITY = "activity"
	SEQUENCE = "sequence"
	PARALLEL = "parallel"
	CHOICE = "choice"
	CONDITION = "condition"
	JOIN = "join"
	SPLIT = "split"
	START_EVENT = "startEvent"
	END_EVENT = "endEvent"
	INTERMEDIATE_EVENT = "intermediateEvent"
	USER_TASK = "userTask"
	SERVICE_TASK = "serviceTask"
	SCRIPT_TASK = "scriptTask"
	MANUAL_TASK = "manualTask"
	BUSINESS_RULE_TASK = "businessRuleTask"
	CALL_ACTIVITY = "callActivity"
	SUB_PROCESS = "subProcess"
	GATEWAY_EXCLUSIVE = "exclusiveGateway"
	GATEWAY_PARALLEL = "parallelGateway"
	GATEWAY_INCLUSIVE = "inclusiveGateway"
	GATEWAY_EVENT_BASED = "eventBasedGateway"
	MESSAGE_FLOW = "messageFlow"
	SEQUENCE_FLOW = "sequenceFlow"

class BPMLGatewayType(str, Enum):
	"""Gateway types for flow control."""
	EXCLUSIVE = "exclusive"  # XOR - one path only
	PARALLEL = "parallel"    # AND - all paths
	INCLUSIVE = "inclusive"  # OR - one or more paths
	EVENT_BASED = "event"    # Wait for events

@dataclass
class BPMLElement:
	"""Base BPML element."""
	id: str
	name: str
	element_type: BPMLElementType
	attributes: Dict[str, Any] = field(default_factory=dict)
	incoming: List[str] = field(default_factory=list)  # Incoming flow IDs
	outgoing: List[str] = field(default_factory=list)  # Outgoing flow IDs
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BPMLFlow:
	"""Sequence or message flow between elements."""
	id: str
	name: str
	source_ref: str
	target_ref: str
	flow_type: str = "sequence"  # sequence, message
	condition: Optional[str] = None
	is_default: bool = False
	attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BPMLProcess:
	"""BPML process definition."""
	id: str
	name: str
	elements: Dict[str, BPMLElement] = field(default_factory=dict)
	flows: Dict[str, BPMLFlow] = field(default_factory=dict)
	start_events: List[str] = field(default_factory=list)
	end_events: List[str] = field(default_factory=list)
	variables: Dict[str, Any] = field(default_factory=dict)
	is_executable: bool = True
	version: str = "1.0"
	documentation: str = ""
	extensions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BPMLExecutionState:
	"""BPML execution state tracking."""
	process_id: str
	instance_id: str
	current_tokens: Dict[str, List[str]] = field(default_factory=dict)  # element_id -> token_ids
	completed_elements: Set[str] = field(default_factory=set)
	failed_elements: Set[str] = field(default_factory=set)
	waiting_elements: Set[str] = field(default_factory=set)
	variables: Dict[str, Any] = field(default_factory=dict)
	execution_path: List[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class BPMLParser:
	"""BPML XML parser supporting full BPML 1.0 and simplified variants."""
	
	def __init__(self, version: BPMLVersion = BPMLVersion.FULL_1_0):
		self.version = version
		self.namespaces = {
			'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
			'bpml': 'http://www.bpml.org/bpml',
			'apg': 'http://apg.datacraft.co.ke/workflow',
			'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
		}
	
	def parse_xml(self, xml_content: str) -> BPMLProcess:
		"""Parse BPML XML content into process definition."""
		try:
			root = ET.fromstring(xml_content)
			
			# Register namespaces
			for prefix, uri in self.namespaces.items():
				ET.register_namespace(prefix, uri)
			
			# Find process element
			process_element = self._find_process_element(root)
			if process_element is None:
				raise ValueError("No process element found in BPML XML")
			
			# Parse process
			process = self._parse_process(process_element)
			
			_log_workflow_operation("bpml_parsed", process.id, {
				"version": self.version.value,
				"elements_count": len(process.elements),
				"flows_count": len(process.flows)
			})
			
			return process
			
		except ET.ParseError as e:
			logger.error(f"BPML XML parsing error: {e}")
			raise ValueError(f"Invalid BPML XML: {e}")
		except Exception as e:
			logger.error(f"BPML parsing error: {e}")
			raise ValueError(f"BPML parsing failed: {e}")
	
	def parse_json(self, json_content: str) -> BPMLProcess:
		"""Parse simplified JSON BPML format."""
		try:
			data = json.loads(json_content)
			
			# Create process
			process = BPMLProcess(
				id=data.get('id', uuid7str()),
				name=data.get('name', 'Unnamed Process'),
				version=data.get('version', '1.0'),
				documentation=data.get('documentation', ''),
				variables=data.get('variables', {}),
				is_executable=data.get('is_executable', True)
			)
			
			# Parse elements
			for element_data in data.get('elements', []):
				element = self._parse_json_element(element_data)
				process.elements[element.id] = element
				
				# Track start and end events
				if element.element_type == BPMLElementType.START_EVENT:
					process.start_events.append(element.id)
				elif element.element_type == BPMLElementType.END_EVENT:
					process.end_events.append(element.id)
			
			# Parse flows
			for flow_data in data.get('flows', []):
				flow = self._parse_json_flow(flow_data)
				process.flows[flow.id] = flow
				
				# Update element connections
				if flow.source_ref in process.elements:
					process.elements[flow.source_ref].outgoing.append(flow.id)
				if flow.target_ref in process.elements:
					process.elements[flow.target_ref].incoming.append(flow.id)
			
			self._validate_process(process)
			
			_log_workflow_operation("bpml_json_parsed", process.id, {
				"version": self.version.value,
				"elements_count": len(process.elements),
				"flows_count": len(process.flows)
			})
			
			return process
			
		except json.JSONDecodeError as e:
			logger.error(f"BPML JSON parsing error: {e}")
			raise ValueError(f"Invalid BPML JSON: {e}")
		except Exception as e:
			logger.error(f"BPML JSON parsing error: {e}")
			raise ValueError(f"BPML JSON parsing failed: {e}")
	
	def _find_process_element(self, root: ET.Element) -> Optional[ET.Element]:
		"""Find the main process element in XML."""
		# Try different namespace combinations
		for ns in ['bpmn', 'bpml', '']:
			prefix = f"{ns}:" if ns else ""
			process_elem = root.find(f".//{prefix}process")
			if process_elem is not None:
				return process_elem
		return None
	
	def _parse_process(self, process_element: ET.Element) -> BPMLProcess:
		"""Parse XML process element."""
		process = BPMLProcess(
			id=process_element.get('id', uuid7str()),
			name=process_element.get('name', 'Unnamed Process'),
			version=process_element.get('version', '1.0'),
			is_executable=process_element.get('isExecutable', 'true').lower() == 'true'
		)
		
		# Parse documentation
		doc_elem = process_element.find('.//documentation')
		if doc_elem is not None and doc_elem.text:
			process.documentation = doc_elem.text.strip()
		
		# Parse all child elements
		for child in process_element:
			if child.tag.endswith('startEvent'):
				element = self._parse_start_event(child)
				process.elements[element.id] = element
				process.start_events.append(element.id)
			elif child.tag.endswith('endEvent'):
				element = self._parse_end_event(child)
				process.elements[element.id] = element
				process.end_events.append(element.id)
			elif child.tag.endswith('task') or 'Task' in child.tag:
				element = self._parse_task(child)
				process.elements[element.id] = element
			elif 'Gateway' in child.tag:
				element = self._parse_gateway(child)
				process.elements[element.id] = element
			elif child.tag.endswith('sequenceFlow'):
				flow = self._parse_sequence_flow(child)
				process.flows[flow.id] = flow
		
		# Update element connections
		self._update_element_connections(process)
		
		# Validate process
		self._validate_process(process)
		
		return process
	
	def _parse_start_event(self, element: ET.Element) -> BPMLElement:
		"""Parse start event element."""
		return BPMLElement(
			id=element.get('id', uuid7str()),
			name=element.get('name', 'Start'),
			element_type=BPMLElementType.START_EVENT,
			attributes=dict(element.attrib)
		)
	
	def _parse_end_event(self, element: ET.Element) -> BPMLElement:
		"""Parse end event element."""
		return BPMLElement(
			id=element.get('id', uuid7str()),
			name=element.get('name', 'End'),
			element_type=BPMLElementType.END_EVENT,
			attributes=dict(element.attrib)
		)
	
	def _parse_task(self, element: ET.Element) -> BPMLElement:
		"""Parse task element."""
		tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
		
		# Map XML task types to BPML element types
		task_type_mapping = {
			'task': BPMLElementType.ACTIVITY,
			'userTask': BPMLElementType.USER_TASK,
			'serviceTask': BPMLElementType.SERVICE_TASK,
			'scriptTask': BPMLElementType.SCRIPT_TASK,
			'manualTask': BPMLElementType.MANUAL_TASK,
			'businessRuleTask': BPMLElementType.BUSINESS_RULE_TASK,
			'callActivity': BPMLElementType.CALL_ACTIVITY
		}
		
		element_type = task_type_mapping.get(tag_name, BPMLElementType.ACTIVITY)
		
		return BPMLElement(
			id=element.get('id', uuid7str()),
			name=element.get('name', f'Task {element.get("id", "")}'),
			element_type=element_type,
			attributes=dict(element.attrib)
		)
	
	def _parse_gateway(self, element: ET.Element) -> BPMLElement:
		"""Parse gateway element."""
		tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
		
		# Map XML gateway types to BPML element types
		gateway_type_mapping = {
			'exclusiveGateway': BPMLElementType.GATEWAY_EXCLUSIVE,
			'parallelGateway': BPMLElementType.GATEWAY_PARALLEL,
			'inclusiveGateway': BPMLElementType.GATEWAY_INCLUSIVE,
			'eventBasedGateway': BPMLElementType.GATEWAY_EVENT_BASED
		}
		
		element_type = gateway_type_mapping.get(tag_name, BPMLElementType.GATEWAY_EXCLUSIVE)
		
		return BPMLElement(
			id=element.get('id', uuid7str()),
			name=element.get('name', f'Gateway {element.get("id", "")}'),
			element_type=element_type,
			attributes=dict(element.attrib)
		)
	
	def _parse_sequence_flow(self, element: ET.Element) -> BPMLFlow:
		"""Parse sequence flow element."""
		return BPMLFlow(
			id=element.get('id', uuid7str()),
			name=element.get('name', ''),
			source_ref=element.get('sourceRef', ''),
			target_ref=element.get('targetRef', ''),
			flow_type='sequence',
			condition=self._extract_condition(element),
			is_default=element.get('default', 'false').lower() == 'true',
			attributes=dict(element.attrib)
		)
	
	def _extract_condition(self, element: ET.Element) -> Optional[str]:
		"""Extract condition expression from flow element."""
		condition_elem = element.find('.//conditionExpression')
		if condition_elem is not None and condition_elem.text:
			return condition_elem.text.strip()
		return None
	
	def _parse_json_element(self, element_data: Dict[str, Any]) -> BPMLElement:
		"""Parse JSON element data."""
		element_type_str = element_data.get('type', 'activity')
		
		# Map string types to enum
		type_mapping = {
			'start': BPMLElementType.START_EVENT,
			'end': BPMLElementType.END_EVENT,
			'activity': BPMLElementType.ACTIVITY,
			'userTask': BPMLElementType.USER_TASK,
			'serviceTask': BPMLElementType.SERVICE_TASK,
			'scriptTask': BPMLElementType.SCRIPT_TASK,
			'manualTask': BPMLElementType.MANUAL_TASK,
			'businessRuleTask': BPMLElementType.BUSINESS_RULE_TASK,
			'exclusiveGateway': BPMLElementType.GATEWAY_EXCLUSIVE,
			'parallelGateway': BPMLElementType.GATEWAY_PARALLEL,
			'inclusiveGateway': BPMLElementType.GATEWAY_INCLUSIVE
		}
		
		element_type = type_mapping.get(element_type_str, BPMLElementType.ACTIVITY)
		
		return BPMLElement(
			id=element_data.get('id', uuid7str()),
			name=element_data.get('name', f'Element {element_data.get("id", "")}'),
			element_type=element_type,
			attributes=element_data.get('attributes', {}),
			metadata=element_data.get('metadata', {})
		)
	
	def _parse_json_flow(self, flow_data: Dict[str, Any]) -> BPMLFlow:
		"""Parse JSON flow data."""
		return BPMLFlow(
			id=flow_data.get('id', uuid7str()),
			name=flow_data.get('name', ''),
			source_ref=flow_data['from'],
			target_ref=flow_data['to'],
			flow_type=flow_data.get('type', 'sequence'),
			condition=flow_data.get('condition'),
			is_default=flow_data.get('default', False),
			attributes=flow_data.get('attributes', {})
		)
	
	def _update_element_connections(self, process: BPMLProcess) -> None:
		"""Update element incoming/outgoing connections."""
		for flow in process.flows.values():
			if flow.source_ref in process.elements:
				process.elements[flow.source_ref].outgoing.append(flow.id)
			if flow.target_ref in process.elements:
				process.elements[flow.target_ref].incoming.append(flow.id)
	
	def _validate_process(self, process: BPMLProcess) -> None:
		"""Validate BPML process structure."""
		errors = []
		
		# Check for start events
		if not process.start_events:
			errors.append("Process must have at least one start event")
		
		# Check for end events
		if not process.end_events:
			errors.append("Process must have at least one end event")
		
		# Validate flows
		for flow in process.flows.values():
			if flow.source_ref not in process.elements:
				errors.append(f"Flow {flow.id} references unknown source: {flow.source_ref}")
			if flow.target_ref not in process.elements:
				errors.append(f"Flow {flow.id} references unknown target: {flow.target_ref}")
		
		# Check for orphaned elements (no incoming or outgoing flows)
		for element in process.elements.values():
			if (element.element_type not in [BPMLElementType.START_EVENT, BPMLElementType.END_EVENT] and
				not element.incoming and not element.outgoing):
				errors.append(f"Element {element.id} has no connections")
		
		if errors:
			raise ValueError(f"BPML validation errors: {'; '.join(errors)}")

class BPMLExecutionEngine:
	"""BPML execution engine with token-based flow control."""
	
	def __init__(self, workflow_executor: 'WorkflowExecutor'):
		self.workflow_executor = workflow_executor
		self.active_executions: Dict[str, BPMLExecutionState] = {}
		self.parser = BPMLParser()
	
	async def execute_bpml_process(
		self,
		process: BPMLProcess,
		instance_id: str,
		initial_variables: Optional[Dict[str, Any]] = None
	) -> BPMLExecutionState:
		"""Execute BPML process with token-based flow control."""
		try:
			# Create execution state
			execution_state = BPMLExecutionState(
				process_id=process.id,
				instance_id=instance_id,
				variables=initial_variables or {}
			)
			
			# Merge process variables
			execution_state.variables.update(process.variables)
			
			# Store execution state
			self.active_executions[instance_id] = execution_state
			
			_log_workflow_operation("bpml_execution_started", process.id, {
				"instance_id": instance_id,
				"start_events": len(process.start_events)
			})
			
			# Create tokens at start events
			for start_event_id in process.start_events:
				await self._create_token(execution_state, start_event_id)
			
			# Execute process
			await self._execute_process_flow(process, execution_state)
			
			return execution_state
			
		except Exception as e:
			logger.error(f"BPML execution error: {e}")
			if instance_id in self.active_executions:
				del self.active_executions[instance_id]
			raise
	
	async def _create_token(self, execution_state: BPMLExecutionState, element_id: str) -> str:
		"""Create execution token at element."""
		token_id = uuid7str()
		
		if element_id not in execution_state.current_tokens:
			execution_state.current_tokens[element_id] = []
		
		execution_state.current_tokens[element_id].append(token_id)
		
		_log_workflow_operation("bpml_token_created", execution_state.process_id, {
			"instance_id": execution_state.instance_id,
			"element_id": element_id,
			"token_id": token_id
		})
		
		return token_id
	
	async def _consume_token(self, execution_state: BPMLExecutionState, element_id: str) -> Optional[str]:
		"""Consume execution token from element."""
		if element_id not in execution_state.current_tokens:
			return None
		
		tokens = execution_state.current_tokens[element_id]
		if not tokens:
			return None
		
		token_id = tokens.pop(0)
		
		if not tokens:
			del execution_state.current_tokens[element_id]
		
		_log_workflow_operation("bpml_token_consumed", execution_state.process_id, {
			"instance_id": execution_state.instance_id,
			"element_id": element_id,
			"token_id": token_id
		})
		
		return token_id
	
	async def _execute_process_flow(self, process: BPMLProcess, execution_state: BPMLExecutionState) -> None:
		"""Execute the process flow with token-based control."""
		max_iterations = 1000  # Prevent infinite loops
		iteration = 0
		
		while execution_state.current_tokens and iteration < max_iterations:
			iteration += 1
			
			# Get elements with tokens
			elements_with_tokens = list(execution_state.current_tokens.keys())
			
			for element_id in elements_with_tokens:
				if element_id not in process.elements:
					continue
				
				element = process.elements[element_id]
				
				# Check if element is ready to execute
				if await self._is_element_ready(process, element, execution_state):
					await self._execute_element(process, element, execution_state)
			
			# Small delay to prevent tight loops
			await asyncio.sleep(0.01)
		
		if iteration >= max_iterations:
			logger.warning(f"BPML execution reached max iterations for {execution_state.instance_id}")
	
	async def _is_element_ready(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> bool:
		"""Check if element is ready to execute."""
		# Element must have tokens
		if element.id not in execution_state.current_tokens:
			return False
		
		# Check gateway join conditions
		if element.element_type in [
			BPMLElementType.GATEWAY_PARALLEL,
			BPMLElementType.GATEWAY_INCLUSIVE
		]:
			return await self._check_gateway_join_condition(process, element, execution_state)
		
		return True
	
	async def _check_gateway_join_condition(
		self,
		process: BPMLProcess,
		gateway: BPMLElement,
		execution_state: BPMLExecutionState
	) -> bool:
		"""Check if gateway join condition is met."""
		if gateway.element_type == BPMLElementType.GATEWAY_PARALLEL:
			# Parallel gateway requires tokens from all incoming flows
			required_tokens = len(gateway.incoming)
			actual_tokens = len(execution_state.current_tokens.get(gateway.id, []))
			return actual_tokens >= required_tokens
		elif gateway.element_type == BPMLElementType.GATEWAY_INCLUSIVE:
			# Inclusive gateway requires at least one token
			return len(execution_state.current_tokens.get(gateway.id, [])) > 0
		
		return True
	
	async def _execute_element(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute BPML element."""
		try:
			# Consume token
			token_id = await self._consume_token(execution_state, element.id)
			if not token_id:
				return
			
			# Add to execution path
			execution_state.execution_path.append(element.id)
			
			_log_workflow_operation("bpml_element_executing", process.id, {
				"instance_id": execution_state.instance_id,
				"element_id": element.id,
				"element_type": element.element_type.value,
				"token_id": token_id
			})
			
			# Execute based on element type
			if element.element_type == BPMLElementType.START_EVENT:
				await self._execute_start_event(process, element, execution_state)
			elif element.element_type == BPMLElementType.END_EVENT:
				await self._execute_end_event(process, element, execution_state)
			elif element.element_type in [
				BPMLElementType.ACTIVITY,
				BPMLElementType.USER_TASK,
				BPMLElementType.SERVICE_TASK,
				BPMLElementType.SCRIPT_TASK,
				BPMLElementType.MANUAL_TASK,
				BPMLElementType.BUSINESS_RULE_TASK
			]:
				await self._execute_task_element(process, element, execution_state)
			elif element.element_type in [
				BPMLElementType.GATEWAY_EXCLUSIVE,
				BPMLElementType.GATEWAY_PARALLEL,
				BPMLElementType.GATEWAY_INCLUSIVE,
				BPMLElementType.GATEWAY_EVENT_BASED
			]:
				await self._execute_gateway(process, element, execution_state)
			
			# Mark element as completed
			execution_state.completed_elements.add(element.id)
			
		except Exception as e:
			logger.error(f"BPML element execution error: {e}")
			execution_state.failed_elements.add(element.id)
			raise
	
	async def _execute_start_event(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute start event."""
		# Start events just pass tokens to outgoing flows
		await self._follow_outgoing_flows(process, element, execution_state)
	
	async def _execute_end_event(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute end event."""
		# End events consume tokens and don't create new ones
		_log_workflow_operation("bpml_end_event_reached", process.id, {
			"instance_id": execution_state.instance_id,
			"element_id": element.id
		})
	
	async def _execute_task_element(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute task element by converting to APG task."""
		# Convert BPML element to APG TaskDefinition
		task_definition = self._convert_bpml_to_apg_task(element, execution_state)
		
		# Create execution context
		from .executor import ExecutionContext
		context = ExecutionContext(
			instance_id=execution_state.instance_id,
			workflow_id=process.id,
			tenant_id=self.workflow_executor.tenant_id,
			user_id="bpml_engine",
			variables=execution_state.variables,
			correlation_id=uuid7str()
		)
		
		# Execute using workflow executor task handlers
		task_handler = self.workflow_executor.task_handlers.get(task_definition.task_type)
		if task_handler:
			# Create task execution record
			from ..models import TaskExecution
			task_execution = TaskExecution(
				instance_id=execution_state.instance_id,
				task_id=element.id,
				task_name=element.name,
				created_by="bpml_engine"
			)
			
			# Execute task
			result = await task_handler.execute(task_definition, task_execution, context)
			
			# Update execution state with results
			if task_execution.output_data:
				execution_state.variables.update(task_execution.output_data)
		
		# Follow outgoing flows
		await self._follow_outgoing_flows(process, element, execution_state)
	
	async def _execute_gateway(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute gateway element."""
		if element.element_type == BPMLElementType.GATEWAY_EXCLUSIVE:
			await self._execute_exclusive_gateway(process, element, execution_state)
		elif element.element_type == BPMLElementType.GATEWAY_PARALLEL:
			await self._execute_parallel_gateway(process, element, execution_state)
		elif element.element_type == BPMLElementType.GATEWAY_INCLUSIVE:
			await self._execute_inclusive_gateway(process, element, execution_state)
		elif element.element_type == BPMLElementType.GATEWAY_EVENT_BASED:
			await self._execute_event_based_gateway(process, element, execution_state)
	
	async def _execute_exclusive_gateway(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute exclusive gateway (XOR)."""
		# Evaluate conditions on outgoing flows
		selected_flow = None
		default_flow = None
		
		for flow_id in element.outgoing:
			flow = process.flows.get(flow_id)
			if not flow:
				continue
			
			if flow.is_default:
				default_flow = flow
			elif flow.condition:
				if await self._evaluate_condition(flow.condition, execution_state):
					selected_flow = flow
					break
			else:
				# No condition means always take this flow
				selected_flow = flow
				break
		
		# Use default flow if no condition matched
		if not selected_flow and default_flow:
			selected_flow = default_flow
		
		# Create token on selected flow target
		if selected_flow:
			await self._create_token(execution_state, selected_flow.target_ref)
	
	async def _execute_parallel_gateway(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute parallel gateway (AND)."""
		# Create tokens on all outgoing flows
		for flow_id in element.outgoing:
			flow = process.flows.get(flow_id)
			if flow:
				await self._create_token(execution_state, flow.target_ref)
	
	async def _execute_inclusive_gateway(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute inclusive gateway (OR)."""
		# Create tokens on flows where conditions are met
		flows_taken = []
		default_flow = None
		
		for flow_id in element.outgoing:
			flow = process.flows.get(flow_id)
			if not flow:
				continue
			
			if flow.is_default:
				default_flow = flow
			elif flow.condition:
				if await self._evaluate_condition(flow.condition, execution_state):
					flows_taken.append(flow)
			else:
				# No condition means always take this flow
				flows_taken.append(flow)
		
		# If no flows taken, use default
		if not flows_taken and default_flow:
			flows_taken.append(default_flow)
		
		# Create tokens on selected flows
		for flow in flows_taken:
			await self._create_token(execution_state, flow.target_ref)
	
	async def _execute_event_based_gateway(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Execute event-based gateway."""
		# For now, just follow the first outgoing flow
		# In a full implementation, this would wait for events
		if element.outgoing:
			flow = process.flows.get(element.outgoing[0])
			if flow:
				await self._create_token(execution_state, flow.target_ref)
	
	async def _follow_outgoing_flows(
		self,
		process: BPMLProcess,
		element: BPMLElement,
		execution_state: BPMLExecutionState
	) -> None:
		"""Follow all outgoing flows from element."""
		for flow_id in element.outgoing:
			flow = process.flows.get(flow_id)
			if flow:
				await self._create_token(execution_state, flow.target_ref)
	
	async def _evaluate_condition(self, condition: str, execution_state: BPMLExecutionState) -> bool:
		"""Evaluate flow condition expression."""
		try:
			# Simple condition evaluation - in production would use expression engine
			# For now, support basic variable comparisons
			if '==' in condition:
				left, right = condition.split('==', 1)
				left_val = execution_state.variables.get(left.strip())
				right_val = right.strip().strip('"\'')
				return str(left_val) == right_val
			elif '!=' in condition:
				left, right = condition.split('!=', 1)
				left_val = execution_state.variables.get(left.strip())
				right_val = right.strip().strip('"\'')
				return str(left_val) != right_val
			elif condition.strip() in execution_state.variables:
				# Boolean variable
				return bool(execution_state.variables[condition.strip()])
			
			return False
		except Exception as e:
			logger.error(f"Condition evaluation error: {e}")
			return False
	
	def _convert_bpml_to_apg_task(self, element: BPMLElement, execution_state: BPMLExecutionState) -> TaskDefinition:
		"""Convert BPML element to APG TaskDefinition."""
		# Map BPML element types to APG task types
		task_type_mapping = {
			BPMLElementType.ACTIVITY: TaskType.AUTOMATED,
			BPMLElementType.USER_TASK: TaskType.HUMAN,
			BPMLElementType.SERVICE_TASK: TaskType.INTEGRATION,
			BPMLElementType.SCRIPT_TASK: TaskType.AUTOMATED,
			BPMLElementType.MANUAL_TASK: TaskType.HUMAN,
			BPMLElementType.BUSINESS_RULE_TASK: TaskType.AUTOMATED
		}
		
		task_type = task_type_mapping.get(element.element_type, TaskType.AUTOMATED)
		
		# Create task definition
		task_definition = TaskDefinition(
			id=element.id,
			name=element.name,
			task_type=task_type,
			configuration=element.attributes,
			metadata=element.metadata
		)
		
		# Set assignment from attributes
		if 'assignee' in element.attributes:
			task_definition.assigned_to = element.attributes['assignee']
		elif 'candidateUsers' in element.attributes:
			# Take first candidate user
			candidates = element.attributes['candidateUsers'].split(',')
			if candidates:
				task_definition.assigned_to = candidates[0].strip()
		
		return task_definition
	
	def get_execution_state(self, instance_id: str) -> Optional[BPMLExecutionState]:
		"""Get BPML execution state."""
		return self.active_executions.get(instance_id)
	
	def cleanup_execution(self, instance_id: str) -> None:
		"""Clean up BPML execution state."""
		if instance_id in self.active_executions:
			del self.active_executions[instance_id]
			_log_workflow_operation("bpml_execution_cleaned", "system", {
				"instance_id": instance_id
			})

__all__ = [
	"BPMLVersion",
	"BPMLElementType",
	"BPMLGatewayType",
	"BPMLElement",
	"BPMLFlow",
	"BPMLProcess",
	"BPMLExecutionState",
	"BPMLParser",
	"BPMLExecutionEngine"
]