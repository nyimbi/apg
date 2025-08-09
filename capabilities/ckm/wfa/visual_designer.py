"""
APG Workflow & Business Process Management - Visual Process Design Studio

Browser-based BPMN 2.0 visual designer with real-time collaboration,
intelligent validation, and seamless APG platform integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from models import (
	APGTenantContext, WBPMProcessDefinition, WBPMProcessActivity, WBPMProcessFlow,
	WBPMServiceResponse, ActivityType, GatewayDirection, EventType,
	WBPMProcessTemplate, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Visual Designer Core Classes
# =============================================================================

class ValidationSeverity(str, Enum):
	"""Validation issue severity levels."""
	ERROR = "error"
	WARNING = "warning"
	INFO = "info"
	SUGGESTION = "suggestion"


class DiagramElementType(str, Enum):
	"""BPMN diagram element types."""
	PROCESS = "process"
	START_EVENT = "startEvent"
	END_EVENT = "endEvent"
	INTERMEDIATE_EVENT = "intermediateEvent"
	USER_TASK = "userTask"
	SERVICE_TASK = "serviceTask"
	SCRIPT_TASK = "scriptTask"
	MANUAL_TASK = "manualTask"
	EXCLUSIVE_GATEWAY = "exclusiveGateway"
	PARALLEL_GATEWAY = "parallelGateway"
	INCLUSIVE_GATEWAY = "inclusiveGateway"
	SEQUENCE_FLOW = "sequenceFlow"
	MESSAGE_FLOW = "messageFlow"
	TEXT_ANNOTATION = "textAnnotation"


@dataclass
class ValidationIssue:
	"""Process validation issue."""
	issue_id: str
	severity: ValidationSeverity
	element_id: Optional[str]
	element_type: Optional[str]
	message: str
	description: str
	suggested_fix: Optional[str] = None
	auto_fixable: bool = False


@dataclass
class DiagramPosition:
	"""Element position in diagram."""
	x: float
	y: float
	width: Optional[float] = None
	height: Optional[float] = None


@dataclass
class DiagramElement:
	"""Visual diagram element."""
	element_id: str
	element_type: DiagramElementType
	name: Optional[str]
	position: DiagramPosition
	properties: Dict[str, Any] = field(default_factory=dict)
	connections: List[str] = field(default_factory=list)
	visual_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessDiagram:
	"""Complete process diagram representation."""
	diagram_id: str
	process_id: str
	tenant_id: str
	diagram_name: str
	elements: Dict[str, DiagramElement] = field(default_factory=dict)
	canvas_properties: Dict[str, Any] = field(default_factory=dict)
	version: str = "1.0.0"
	created_by: str = ""
	updated_by: str = ""
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DesignSession:
	"""Active design session state."""
	session_id: str
	tenant_context: APGTenantContext
	process_diagram: ProcessDiagram
	participants: List[str] = field(default_factory=list)
	active_collaborators: Set[str] = field(default_factory=set)
	session_state: Dict[str, Any] = field(default_factory=dict)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	auto_save_enabled: bool = True
	validation_enabled: bool = True


# =============================================================================
# BPMN Validation Engine
# =============================================================================

class BPMNValidator:
	"""Comprehensive BPMN 2.0 validation engine."""
	
	def __init__(self):
		self.validation_rules = self._initialize_validation_rules()
	
	async def validate_diagram(self, diagram: ProcessDiagram) -> List[ValidationIssue]:
		"""Perform comprehensive diagram validation."""
		issues = []
		
		try:
			# Structural validation
			issues.extend(await self._validate_structure(diagram))
			
			# Flow validation
			issues.extend(await self._validate_flows(diagram))
			
			# Gateway validation
			issues.extend(await self._validate_gateways(diagram))
			
			# Task validation
			issues.extend(await self._validate_tasks(diagram))
			
			# Event validation
			issues.extend(await self._validate_events(diagram))
			
			# Best practices validation
			issues.extend(await self._validate_best_practices(diagram))
			
			logger.info(f"Diagram validation completed: {len(issues)} issues found")
			return issues
			
		except Exception as e:
			logger.error(f"Error during diagram validation: {e}")
			return [ValidationIssue(
				issue_id=f"validation_error_{uuid.uuid4().hex[:8]}",
				severity=ValidationSeverity.ERROR,
				element_id=None,
				element_type=None,
				message="Validation engine error",
				description=f"Internal validation error: {e}"
			)]
	
	async def _validate_structure(self, diagram: ProcessDiagram) -> List[ValidationIssue]:
		"""Validate basic diagram structure."""
		issues = []
		
		# Check for start events
		start_events = [
			elem for elem in diagram.elements.values()
			if elem.element_type == DiagramElementType.START_EVENT
		]
		
		if not start_events:
			issues.append(ValidationIssue(
				issue_id="missing_start_event",
				severity=ValidationSeverity.ERROR,
				element_id=None,
				element_type=None,
				message="Process must have at least one start event",
				description="Every BPMN process must begin with a start event to define the entry point.",
				suggested_fix="Add a start event to begin the process flow"
			))
		elif len(start_events) > 1:
			issues.append(ValidationIssue(
				issue_id="multiple_start_events",
				severity=ValidationSeverity.WARNING,
				element_id=None,
				element_type=None,
				message="Process has multiple start events",
				description="Multiple start events may cause confusion. Consider using intermediate events or event-based gateways.",
				suggested_fix="Consolidate to a single start event or use event-based gateway"
			))
		
		# Check for end events
		end_events = [
			elem for elem in diagram.elements.values()
			if elem.element_type == DiagramElementType.END_EVENT
		]
		
		if not end_events:
			issues.append(ValidationIssue(
				issue_id="missing_end_event",
				severity=ValidationSeverity.ERROR,
				element_id=None,
				element_type=None,
				message="Process must have at least one end event",
				description="Every BPMN process must have at least one end event to define completion.",
				suggested_fix="Add an end event to complete the process flow"
			))
		
		# Check for unreachable elements
		reachable_elements = await self._find_reachable_elements(diagram)
		all_elements = set(diagram.elements.keys())
		unreachable = all_elements - reachable_elements
		
		for element_id in unreachable:
			element = diagram.elements[element_id]
			issues.append(ValidationIssue(
				issue_id=f"unreachable_element_{element_id}",
				severity=ValidationSeverity.WARNING,
				element_id=element_id,
				element_type=element.element_type.value,
				message=f"Element '{element.name or element_id}' is unreachable",
				description="This element cannot be reached from any start event in the process.",
				suggested_fix="Connect this element to the main process flow or remove it"
			))
		
		return issues
	
	async def _validate_flows(self, diagram: ProcessDiagram) -> List[ValidationIssue]:
		"""Validate sequence flows and connections."""
		issues = []
		
		sequence_flows = [
			elem for elem in diagram.elements.values()
			if elem.element_type == DiagramElementType.SEQUENCE_FLOW
		]
		
		for flow in sequence_flows:
			flow_props = flow.properties
			source_id = flow_props.get('source_ref')
			target_id = flow_props.get('target_ref')
			
			# Validate source and target exist
			if not source_id or source_id not in diagram.elements:
				issues.append(ValidationIssue(
					issue_id=f"invalid_flow_source_{flow.element_id}",
					severity=ValidationSeverity.ERROR,
					element_id=flow.element_id,
					element_type=flow.element_type.value,
					message="Sequence flow has invalid source",
					description=f"Flow '{flow.name or flow.element_id}' references non-existent source element.",
					suggested_fix="Connect flow to a valid source element"
				))
			
			if not target_id or target_id not in diagram.elements:
				issues.append(ValidationIssue(
					issue_id=f"invalid_flow_target_{flow.element_id}",
					severity=ValidationSeverity.ERROR,
					element_id=flow.element_id,
					element_type=flow.element_type.value,
					message="Sequence flow has invalid target",
					description=f"Flow '{flow.name or flow.element_id}' references non-existent target element.",
					suggested_fix="Connect flow to a valid target element"
				))
			
			# Validate flow conditions on exclusive gateways
			if (source_id and source_id in diagram.elements and
				diagram.elements[source_id].element_type == DiagramElementType.EXCLUSIVE_GATEWAY):
				
				condition = flow_props.get('condition_expression')
				is_default = flow_props.get('is_default_flow', False)
				
				if not condition and not is_default:
					issues.append(ValidationIssue(
						issue_id=f"missing_flow_condition_{flow.element_id}",
						severity=ValidationSeverity.WARNING,
						element_id=flow.element_id,
						element_type=flow.element_type.value,
						message="Outgoing flow from exclusive gateway missing condition",
						description="Flows from exclusive gateways should have conditions or be marked as default.",
						suggested_fix="Add a condition expression or mark as default flow"
					))
		
		return issues
	
	async def _validate_gateways(self, diagram: ProcessDiagram) -> List[ValidationIssue]:
		"""Validate gateway configurations."""
		issues = []
		
		gateway_types = [
			DiagramElementType.EXCLUSIVE_GATEWAY,
			DiagramElementType.PARALLEL_GATEWAY,
			DiagramElementType.INCLUSIVE_GATEWAY
		]
		
		gateways = [
			elem for elem in diagram.elements.values()
			if elem.element_type in gateway_types
		]
		
		for gateway in gateways:
			incoming_flows = self._get_incoming_flows(gateway.element_id, diagram)
			outgoing_flows = self._get_outgoing_flows(gateway.element_id, diagram)
			
			# Check for single flow gateways
			if len(incoming_flows) <= 1 and len(outgoing_flows) <= 1:
				issues.append(ValidationIssue(
					issue_id=f"unnecessary_gateway_{gateway.element_id}",
					severity=ValidationSeverity.SUGGESTION,
					element_id=gateway.element_id,
					element_type=gateway.element_type.value,
					message="Gateway has only one incoming and outgoing flow",
					description="This gateway serves no purpose and can be removed.",
					suggested_fix="Remove gateway and connect flows directly",
					auto_fixable=True
				))
			
			# Exclusive gateway specific validation
			if gateway.element_type == DiagramElementType.EXCLUSIVE_GATEWAY:
				default_flows = [
					flow for flow in outgoing_flows
					if flow.properties.get('is_default_flow', False)
				]
				
				if len(default_flows) > 1:
					issues.append(ValidationIssue(
						issue_id=f"multiple_default_flows_{gateway.element_id}",
						severity=ValidationSeverity.ERROR,
						element_id=gateway.element_id,
						element_type=gateway.element_type.value,
						message="Exclusive gateway has multiple default flows",
						description="An exclusive gateway can have only one default flow.",
						suggested_fix="Mark only one outgoing flow as default"
					))
		
		return issues
	
	async def _validate_tasks(self, diagram: ProcessDiagram) -> List[ValidationIssue]:
		"""Validate task configurations."""
		issues = []
		
		task_types = [
			DiagramElementType.USER_TASK,
			DiagramElementType.SERVICE_TASK,
			DiagramElementType.SCRIPT_TASK,
			DiagramElementType.MANUAL_TASK
		]
		
		tasks = [
			elem for elem in diagram.elements.values()
			if elem.element_type in task_types
		]
		
		for task in tasks:
			# Check for missing names
			if not task.name or task.name.strip() == "":
				issues.append(ValidationIssue(
					issue_id=f"unnamed_task_{task.element_id}",
					severity=ValidationSeverity.WARNING,
					element_id=task.element_id,
					element_type=task.element_type.value,
					message="Task is missing a name",
					description="Tasks should have descriptive names for clarity.",
					suggested_fix="Add a descriptive name to this task"
				))
			
			# User task specific validation
			if task.element_type == DiagramElementType.USER_TASK:
				props = task.properties
				assignee = props.get('assignee')
				candidate_users = props.get('candidate_users', [])
				candidate_groups = props.get('candidate_groups', [])
				
				if not assignee and not candidate_users and not candidate_groups:
					issues.append(ValidationIssue(
						issue_id=f"unassigned_user_task_{task.element_id}",
						severity=ValidationSeverity.WARNING,
						element_id=task.element_id,
						element_type=task.element_type.value,
						message="User task has no assignment configuration",
						description="User tasks should specify assignee, candidate users, or candidate groups.",
						suggested_fix="Configure task assignment through properties panel"
					))
			
			# Service task specific validation
			elif task.element_type == DiagramElementType.SERVICE_TASK:
				props = task.properties
				implementation = (
					props.get('class_name') or
					props.get('expression') or
					props.get('delegate_expression')
				)
				
				if not implementation:
					issues.append(ValidationIssue(
						issue_id=f"unconfigured_service_task_{task.element_id}",
						severity=ValidationSeverity.ERROR,
						element_id=task.element_id,
						element_type=task.element_type.value,
						message="Service task has no implementation configured",
						description="Service tasks require class name, expression, or delegate expression.",
						suggested_fix="Configure service implementation in properties panel"
					))
		
		return issues
	
	async def _validate_events(self, diagram: ProcessDiagram) -> List[ValidationIssue]:
		"""Validate event configurations."""
		issues = []
		
		event_types = [
			DiagramElementType.START_EVENT,
			DiagramElementType.END_EVENT,
			DiagramElementType.INTERMEDIATE_EVENT
		]
		
		events = [
			elem for elem in diagram.elements.values()
			if elem.element_type in event_types
		]
		
		for event in events:
			# Start events should have no incoming flows
			if event.element_type == DiagramElementType.START_EVENT:
				incoming_flows = self._get_incoming_flows(event.element_id, diagram)
				if incoming_flows:
					issues.append(ValidationIssue(
						issue_id=f"start_event_incoming_flow_{event.element_id}",
						severity=ValidationSeverity.ERROR,
						element_id=event.element_id,
						element_type=event.element_type.value,
						message="Start event has incoming sequence flows",
						description="Start events cannot have incoming sequence flows.",
						suggested_fix="Remove incoming flows from start event"
					))
			
			# End events should have no outgoing flows
			elif event.element_type == DiagramElementType.END_EVENT:
				outgoing_flows = self._get_outgoing_flows(event.element_id, diagram)
				if outgoing_flows:
					issues.append(ValidationIssue(
						issue_id=f"end_event_outgoing_flow_{event.element_id}",
						severity=ValidationSeverity.ERROR,
						element_id=event.element_id,
						element_type=event.element_type.value,
						message="End event has outgoing sequence flows",
						description="End events cannot have outgoing sequence flows.",
						suggested_fix="Remove outgoing flows from end event"
					))
		
		return issues
	
	async def _validate_best_practices(self, diagram: ProcessDiagram) -> List[ValidationIssue]:
		"""Validate against BPMN best practices."""
		issues = []
		
		# Check for overly complex processes
		total_elements = len(diagram.elements)
		if total_elements > 50:
			issues.append(ValidationIssue(
				issue_id="complex_process",
				severity=ValidationSeverity.SUGGESTION,
				element_id=None,
				element_type=None,
				message="Process is very complex",
				description=f"Process has {total_elements} elements. Consider breaking into subprocesses.",
				suggested_fix="Split complex process into smaller, manageable subprocesses"
			))
		
		# Check for processes without documentation
		tasks_without_docs = [
			elem for elem in diagram.elements.values()
			if elem.element_type in [DiagramElementType.USER_TASK, DiagramElementType.SERVICE_TASK]
			and not elem.properties.get('documentation')
		]
		
		if len(tasks_without_docs) > 3:
			issues.append(ValidationIssue(
				issue_id="missing_documentation",
				severity=ValidationSeverity.INFO,
				element_id=None,
				element_type=None,
				message="Many tasks lack documentation",
				description="Consider adding documentation to tasks for better maintainability.",
				suggested_fix="Add documentation to key tasks and activities"
			))
		
		return issues
	
	def _get_incoming_flows(self, element_id: str, diagram: ProcessDiagram) -> List[DiagramElement]:
		"""Get incoming sequence flows for element."""
		return [
			elem for elem in diagram.elements.values()
			if (elem.element_type == DiagramElementType.SEQUENCE_FLOW and
				elem.properties.get('target_ref') == element_id)
		]
	
	def _get_outgoing_flows(self, element_id: str, diagram: ProcessDiagram) -> List[DiagramElement]:
		"""Get outgoing sequence flows for element."""
		return [
			elem for elem in diagram.elements.values()
			if (elem.element_type == DiagramElementType.SEQUENCE_FLOW and
				elem.properties.get('source_ref') == element_id)
		]
	
	async def _find_reachable_elements(self, diagram: ProcessDiagram) -> Set[str]:
		"""Find all elements reachable from start events."""
		reachable = set()
		
		# Find all start events
		start_events = [
			elem.element_id for elem in diagram.elements.values()
			if elem.element_type == DiagramElementType.START_EVENT
		]
		
		# Traverse from each start event
		for start_id in start_events:
			await self._traverse_reachable(start_id, diagram, reachable)
		
		return reachable
	
	async def _traverse_reachable(self, element_id: str, diagram: ProcessDiagram, visited: Set[str]) -> None:
		"""Recursively traverse reachable elements."""
		if element_id in visited:
			return
		
		visited.add(element_id)
		
		# Get outgoing flows
		outgoing_flows = self._get_outgoing_flows(element_id, diagram)
		
		for flow in outgoing_flows:
			target_id = flow.properties.get('target_ref')
			if target_id:
				await self._traverse_reachable(target_id, diagram, visited)
	
	def _initialize_validation_rules(self) -> Dict[str, Any]:
		"""Initialize validation rules configuration."""
		return {
			'structural': {
				'require_start_event': True,
				'require_end_event': True,
				'check_unreachable_elements': True
			},
			'flows': {
				'validate_connections': True,
				'check_gateway_conditions': True
			},
			'tasks': {
				'require_task_names': True,
				'validate_user_task_assignment': True,
				'validate_service_task_implementation': True
			},
			'best_practices': {
				'complexity_threshold': 50,
				'documentation_recommendation_threshold': 3
			}
		}


# =============================================================================
# Visual Designer Service
# =============================================================================

class VisualDesignerService:
	"""Core visual designer service with collaboration and validation."""
	
	def __init__(self):
		self.active_sessions: Dict[str, DesignSession] = {}
		self.validator = BPMNValidator()
		self.auto_save_interval = 30  # seconds
		self.session_timeout = 3600  # 1 hour
	
	async def create_design_session(
		self,
		process_id: Optional[str],
		context: APGTenantContext,
		template_id: Optional[str] = None
	) -> WBPMServiceResponse:
		"""Create new design session."""
		try:
			session_id = f"design_{uuid.uuid4().hex}"
			
			# Create or load process diagram
			if process_id:
				# Load existing process
				diagram = await self._load_process_diagram(process_id, context)
			elif template_id:
				# Create from template
				diagram = await self._create_from_template(template_id, context)
			else:
				# Create blank diagram
				diagram = await self._create_blank_diagram(context)
			
			# Create design session
			session = DesignSession(
				session_id=session_id,
				tenant_context=context,
				process_diagram=diagram,
				participants=[context.user_id],
				active_collaborators={context.user_id}
			)
			
			self.active_sessions[session_id] = session
			
			# Start background tasks
			asyncio.create_task(self._session_auto_save(session_id))
			asyncio.create_task(self._session_cleanup(session_id))
			
			logger.info(f"Design session created: {session_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Design session created successfully",
				data={
					"session_id": session_id,
					"diagram_id": diagram.diagram_id,
					"process_id": diagram.process_id,
					"session_info": {
						"participants": session.participants,
						"auto_save_enabled": session.auto_save_enabled,
						"validation_enabled": session.validation_enabled
					}
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating design session: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create design session: {e}",
				errors=[str(e)]
			)
	
	async def join_design_session(
		self,
		session_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Join existing design session for collaboration."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Design session not found",
					errors=["Session not found"]
				)
			
			# Verify tenant access
			if session.tenant_context.tenant_id != context.tenant_id:
				return WBPMServiceResponse(
					success=False,
					message="Access denied to design session",
					errors=["Tenant access denied"]
				)
			
			# Add participant
			if context.user_id not in session.participants:
				session.participants.append(context.user_id)
			
			session.active_collaborators.add(context.user_id)
			session.last_activity = datetime.utcnow()
			
			# Notify other collaborators
			await self._notify_collaborators(session_id, {
				"event": "user_joined",
				"user_id": context.user_id,
				"timestamp": datetime.utcnow().isoformat()
			})
			
			return WBPMServiceResponse(
				success=True,
				message="Joined design session successfully",
				data={
					"session_id": session_id,
					"diagram": session.process_diagram.dict() if hasattr(session.process_diagram, 'dict') else session.process_diagram.__dict__,
					"collaborators": list(session.active_collaborators),
					"session_state": session.session_state
				}
			)
			
		except Exception as e:
			logger.error(f"Error joining design session {session_id}: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to join design session: {e}",
				errors=[str(e)]
			)
	
	async def update_diagram_element(
		self,
		session_id: str,
		element_update: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Update diagram element in design session."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Design session not found",
					errors=["Session not found"]
				)
			
			# Verify access
			if context.user_id not in session.participants:
				return WBPMServiceResponse(
					success=False,
					message="Not a participant in this session",
					errors=["Access denied"]
				)
			
			element_id = element_update.get('element_id')
			if not element_id:
				return WBPMServiceResponse(
					success=False,
					message="Element ID required",
					errors=["Missing element_id"]
				)
			
			# Apply update
			if element_id in session.process_diagram.elements:
				element = session.process_diagram.elements[element_id]
				
				# Update properties
				if 'name' in element_update:
					element.name = element_update['name']
				
				if 'position' in element_update:
					pos_data = element_update['position']
					element.position = DiagramPosition(
						x=pos_data.get('x', element.position.x),
						y=pos_data.get('y', element.position.y),
						width=pos_data.get('width', element.position.width),
						height=pos_data.get('height', element.position.height)
					)
				
				if 'properties' in element_update:
					element.properties.update(element_update['properties'])
				
				if 'visual_properties' in element_update:
					element.visual_properties.update(element_update['visual_properties'])
			
			else:
				# Create new element
				element = self._create_diagram_element(element_update)
				session.process_diagram.elements[element_id] = element
			
			# Update session metadata
			session.process_diagram.updated_by = context.user_id
			session.process_diagram.updated_at = datetime.utcnow()
			session.last_activity = datetime.utcnow()
			
			# Real-time validation if enabled
			validation_issues = []
			if session.validation_enabled:
				validation_issues = await self.validator.validate_diagram(session.process_diagram)
			
			# Notify collaborators
			await self._notify_collaborators(session_id, {
				"event": "element_updated",
				"element_id": element_id,
				"updated_by": context.user_id,
				"update_data": element_update,
				"validation_issues": [issue.__dict__ for issue in validation_issues],
				"timestamp": datetime.utcnow().isoformat()
			})
			
			return WBPMServiceResponse(
				success=True,
				message="Element updated successfully",
				data={
					"element_id": element_id,
					"validation_issues": [issue.__dict__ for issue in validation_issues]
				}
			)
			
		except Exception as e:
			logger.error(f"Error updating diagram element: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to update element: {e}",
				errors=[str(e)]
			)
	
	async def validate_diagram(
		self,
		session_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Validate complete diagram."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Design session not found",
					errors=["Session not found"]
				)
			
			# Perform validation
			validation_issues = await self.validator.validate_diagram(session.process_diagram)
			
			# Categorize issues
			errors = [issue for issue in validation_issues if issue.severity == ValidationSeverity.ERROR]
			warnings = [issue for issue in validation_issues if issue.severity == ValidationSeverity.WARNING]
			suggestions = [issue for issue in validation_issues if issue.severity == ValidationSeverity.SUGGESTION]
			
			is_valid = len(errors) == 0
			
			return WBPMServiceResponse(
				success=True,
				message=f"Diagram validation completed. Valid: {is_valid}",
				data={
					"is_valid": is_valid,
					"validation_summary": {
						"total_issues": len(validation_issues),
						"errors": len(errors),
						"warnings": len(warnings),
						"suggestions": len(suggestions)
					},
					"issues": [issue.__dict__ for issue in validation_issues],
					"validation_timestamp": datetime.utcnow().isoformat()
				}
			)
			
		except Exception as e:
			logger.error(f"Error validating diagram: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to validate diagram: {e}",
				errors=[str(e)]
			)
	
	async def export_bpmn_xml(
		self,
		session_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Export diagram as BPMN XML."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Design session not found",
					errors=["Session not found"]
				)
			
			# Generate BPMN XML
			bpmn_xml = await self._generate_bpmn_xml(session.process_diagram)
			
			return WBPMServiceResponse(
				success=True,
				message="BPMN XML exported successfully",
				data={
					"bpmn_xml": bpmn_xml,
					"diagram_id": session.process_diagram.diagram_id,
					"export_timestamp": datetime.utcnow().isoformat()
				}
			)
			
		except Exception as e:
			logger.error(f"Error exporting BPMN XML: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to export BPMN XML: {e}",
				errors=[str(e)]
			)
	
	async def save_process_definition(
		self,
		session_id: str,
		process_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Save diagram as process definition."""
		try:
			session = self.active_sessions.get(session_id)
			if not session:
				return WBPMServiceResponse(
					success=False,
					message="Design session not found",
					errors=["Session not found"]
				)
			
			# Validate before saving
			validation_issues = await self.validator.validate_diagram(session.process_diagram)
			errors = [issue for issue in validation_issues if issue.severity == ValidationSeverity.ERROR]
			
			if errors:
				return WBPMServiceResponse(
					success=False,
					message="Cannot save process with validation errors",
					errors=[f"Validation error: {error.message}" for error in errors]
				)
			
			# Generate BPMN XML
			bpmn_xml = await self._generate_bpmn_xml(session.process_diagram)
			
			# Create process definition
			process_definition = WBPMProcessDefinition(
				tenant_id=context.tenant_id,
				process_key=process_data['process_key'],
				process_name=process_data['process_name'],
				process_description=process_data.get('process_description'),
				bpmn_xml=bpmn_xml,
				category=process_data.get('category'),
				tags=process_data.get('tags', []),
				created_by=context.user_id,
				updated_by=context.user_id
			)
			
			# In production, save to database via process service
			# For now, simulate successful save
			process_id = process_definition.id
			
			# Update session
			session.process_diagram.process_id = process_id
			session.process_diagram.updated_by = context.user_id
			session.process_diagram.updated_at = datetime.utcnow()
			
			logger.info(f"Process definition saved: {process_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Process definition saved successfully",
				data={
					"process_id": process_id,
					"process_key": process_definition.process_key,
					"process_name": process_definition.process_name,
					"version": process_definition.process_version,
					"validation_summary": {
						"warnings": len([i for i in validation_issues if i.severity == ValidationSeverity.WARNING]),
						"suggestions": len([i for i in validation_issues if i.severity == ValidationSeverity.SUGGESTION])
					}
				}
			)
			
		except Exception as e:
			logger.error(f"Error saving process definition: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to save process definition: {e}",
				errors=[str(e)]
			)
	
	async def _load_process_diagram(self, process_id: str, context: APGTenantContext) -> ProcessDiagram:
		"""Load existing process as diagram."""
		# In production, load from database
		# For now, create a sample diagram
		return ProcessDiagram(
			diagram_id=f"diagram_{uuid.uuid4().hex}",
			process_id=process_id,
			tenant_id=context.tenant_id,
			diagram_name="Loaded Process",
			created_by=context.user_id,
			updated_by=context.user_id
		)
	
	async def _create_from_template(self, template_id: str, context: APGTenantContext) -> ProcessDiagram:
		"""Create diagram from template."""
		# In production, load template and create diagram
		return ProcessDiagram(
			diagram_id=f"diagram_{uuid.uuid4().hex}",
			process_id=f"process_{uuid.uuid4().hex}",
			tenant_id=context.tenant_id,
			diagram_name="New Process from Template",
			created_by=context.user_id,
			updated_by=context.user_id
		)
	
	async def _create_blank_diagram(self, context: APGTenantContext) -> ProcessDiagram:
		"""Create blank diagram with start and end events."""
		diagram = ProcessDiagram(
			diagram_id=f"diagram_{uuid.uuid4().hex}",
			process_id=f"process_{uuid.uuid4().hex}",
			tenant_id=context.tenant_id,
			diagram_name="New Process",
			created_by=context.user_id,
			updated_by=context.user_id
		)
		
		# Add default start event
		start_event = DiagramElement(
			element_id="start_1",
			element_type=DiagramElementType.START_EVENT,
			name="Start",
			position=DiagramPosition(x=100, y=100, width=36, height=36)
		)
		
		# Add default end event
		end_event = DiagramElement(
			element_id="end_1",
			element_type=DiagramElementType.END_EVENT,
			name="End",
			position=DiagramPosition(x=300, y=100, width=36, height=36)
		)
		
		diagram.elements[start_event.element_id] = start_event
		diagram.elements[end_event.element_id] = end_event
		
		return diagram
	
	def _create_diagram_element(self, element_data: Dict[str, Any]) -> DiagramElement:
		"""Create diagram element from update data."""
		return DiagramElement(
			element_id=element_data['element_id'],
			element_type=DiagramElementType(element_data['element_type']),
			name=element_data.get('name'),
			position=DiagramPosition(**element_data.get('position', {})),
			properties=element_data.get('properties', {}),
			connections=element_data.get('connections', []),
			visual_properties=element_data.get('visual_properties', {})
		)
	
	async def _generate_bpmn_xml(self, diagram: ProcessDiagram) -> str:
		"""Generate BPMN 2.0 XML from diagram."""
		# Create BPMN XML structure
		root = ET.Element("definitions")
		root.set("xmlns", "http://www.omg.org/spec/BPMN/20100524/MODEL")
		root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
		root.set("xmlns:bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
		root.set("xmlns:omgdc", "http://www.omg.org/spec/DD/20100524/DC")
		root.set("xmlns:omgdi", "http://www.omg.org/spec/DD/20100524/DI")
		root.set("id", diagram.diagram_id)
		root.set("targetNamespace", f"http://datacraft.co.ke/workflows/{diagram.tenant_id}")
		
		# Create process element
		process = ET.SubElement(root, "process")
		process.set("id", diagram.process_id)
		process.set("name", diagram.diagram_name)
		process.set("isExecutable", "true")
		
		# Add elements
		for element in diagram.elements.values():
			self._add_bpmn_element(process, element)
		
		# Add diagram interchange
		bpmn_diagram = ET.SubElement(root, "bpmndi:BPMNDiagram")
		bpmn_diagram.set("id", f"{diagram.diagram_id}_di")
		
		bpmn_plane = ET.SubElement(bpmn_diagram, "bpmndi:BPMNPlane")
		bpmn_plane.set("id", f"{diagram.process_id}_plane")
		bpmn_plane.set("bpmnElement", diagram.process_id)
		
		# Add shapes and edges
		for element in diagram.elements.values():
			self._add_bpmn_di_element(bpmn_plane, element)
		
		# Convert to string
		return ET.tostring(root, encoding='unicode')
	
	def _add_bpmn_element(self, process: ET.Element, element: DiagramElement) -> None:
		"""Add BPMN element to process."""
		if element.element_type == DiagramElementType.START_EVENT:
			elem = ET.SubElement(process, "startEvent")
		elif element.element_type == DiagramElementType.END_EVENT:
			elem = ET.SubElement(process, "endEvent")
		elif element.element_type == DiagramElementType.USER_TASK:
			elem = ET.SubElement(process, "userTask")
		elif element.element_type == DiagramElementType.SERVICE_TASK:
			elem = ET.SubElement(process, "serviceTask")
		elif element.element_type == DiagramElementType.EXCLUSIVE_GATEWAY:
			elem = ET.SubElement(process, "exclusiveGateway")
		elif element.element_type == DiagramElementType.SEQUENCE_FLOW:
			elem = ET.SubElement(process, "sequenceFlow")
		else:
			return  # Skip unsupported elements
		
		elem.set("id", element.element_id)
		if element.name:
			elem.set("name", element.name)
		
		# Add element-specific attributes
		if element.element_type == DiagramElementType.SEQUENCE_FLOW:
			if 'source_ref' in element.properties:
				elem.set("sourceRef", element.properties['source_ref'])
			if 'target_ref' in element.properties:
				elem.set("targetRef", element.properties['target_ref'])
	
	def _add_bpmn_di_element(self, plane: ET.Element, element: DiagramElement) -> None:
		"""Add BPMN DI element to diagram plane."""
		if element.element_type == DiagramElementType.SEQUENCE_FLOW:
			# Add edge
			edge = ET.SubElement(plane, "bpmndi:BPMNEdge")
			edge.set("id", f"{element.element_id}_di")
			edge.set("bpmnElement", element.element_id)
			
			# Add waypoints (simplified)
			waypoint1 = ET.SubElement(edge, "omgdi:waypoint")
			waypoint1.set("x", str(element.position.x))
			waypoint1.set("y", str(element.position.y))
			
			waypoint2 = ET.SubElement(edge, "omgdi:waypoint")
			waypoint2.set("x", str(element.position.x + 100))
			waypoint2.set("y", str(element.position.y))
		else:
			# Add shape
			shape = ET.SubElement(plane, "bpmndi:BPMNShape")
			shape.set("id", f"{element.element_id}_di")
			shape.set("bpmnElement", element.element_id)
			
			bounds = ET.SubElement(shape, "omgdc:Bounds")
			bounds.set("x", str(element.position.x))
			bounds.set("y", str(element.position.y))
			bounds.set("width", str(element.position.width or 100))
			bounds.set("height", str(element.position.height or 80))
	
	async def _notify_collaborators(self, session_id: str, event_data: Dict[str, Any]) -> None:
		"""Notify collaborators of session changes."""
		session = self.active_sessions.get(session_id)
		if not session:
			return
		
		# In production, use WebSocket or APG real-time collaboration service
		logger.info(f"Collaborator notification for session {session_id}: {event_data['event']}")
	
	async def _session_auto_save(self, session_id: str) -> None:
		"""Auto-save session periodically."""
		while session_id in self.active_sessions:
			session = self.active_sessions[session_id]
			if session.auto_save_enabled:
				# In production, save to database
				logger.debug(f"Auto-saving session {session_id}")
			
			await asyncio.sleep(self.auto_save_interval)
	
	async def _session_cleanup(self, session_id: str) -> None:
		"""Clean up inactive sessions."""
		await asyncio.sleep(self.session_timeout)
		
		session = self.active_sessions.get(session_id)
		if session:
			inactive_time = datetime.utcnow() - session.last_activity
			if inactive_time.total_seconds() > self.session_timeout:
				# Save final state and remove session
				logger.info(f"Cleaning up inactive session: {session_id}")
				del self.active_sessions[session_id]


# =============================================================================
# Service Factory
# =============================================================================

def create_visual_designer_service() -> VisualDesignerService:
	"""Create and configure visual designer service."""
	service = VisualDesignerService()
	logger.info("Visual designer service created and configured")
	return service


# Export main classes
__all__ = [
	'VisualDesignerService',
	'BPMNValidator',
	'ProcessDiagram',
	'DiagramElement',
	'DesignSession',
	'ValidationIssue',
	'DiagramElementType',
	'ValidationSeverity',
	'create_visual_designer_service'
]