"""
APG Workflow & Business Process Management - APG Workflow Definition Parser

Comprehensive parser and implementer for textual APG workflow definitions,
enabling automatic workflow creation from structured text descriptions.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import re
import yaml
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from uuid_extensions import uuid7str

from models import (
	APGTenantContext, WBPMProcessDefinition, WBPMProcessActivity, WBPMProcessFlow,
	WBPMServiceResponse, ActivityType, GatewayDirection, EventType, APGBaseModel,
	TaskPriority
)

from enhanced_visual_designer import (
	EnhancedVisualDesignerService, ProcessDiagramCanvas, EnhancedDiagramElement,
	VisualPosition, TimingConfiguration, ElementPermissions
)

from workflow_scheduler import WorkflowScheduler, ProcessTimer, WorkflowSchedule, ScheduleType

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# APG Workflow Definition Structures
# =============================================================================

class APGWorkflowFormat(str, Enum):
	"""Supported APG workflow definition formats."""
	YAML = "yaml"
	JSON = "json"
	APG_DSL = "apg_dsl"  # APG Domain Specific Language
	NATURAL_LANGUAGE = "natural_language"
	BPMN_XML = "bpmn_xml"


class APGElementType(str, Enum):
	"""APG workflow element types."""
	START = "start"
	END = "end"
	TASK = "task"
	USER_TASK = "user_task"
	SERVICE_TASK = "service_task"
	DECISION = "decision"
	PARALLEL = "parallel"
	WAIT = "wait"
	TIMER = "timer"
	SUBPROCESS = "subprocess"
	LOOP = "loop"
	CONDITION = "condition"


@dataclass
class APGWorkflowElement:
	"""APG workflow element definition."""
	id: str = ""
	type: APGElementType = APGElementType.TASK
	name: str = ""
	description: str = ""
	
	# Task Configuration
	assignee: Optional[str] = None
	candidate_groups: List[str] = field(default_factory=list)
	form_fields: List[Dict[str, Any]] = field(default_factory=list)
	
	# Service Task Configuration
	service_type: Optional[str] = None
	service_url: Optional[str] = None
	service_method: str = "POST"
	service_headers: Dict[str, str] = field(default_factory=dict)
	service_payload: Dict[str, Any] = field(default_factory=dict)
	
	# Decision Configuration
	conditions: List[Dict[str, Any]] = field(default_factory=list)
	default_path: Optional[str] = None
	
	# Timing Configuration
	estimated_duration: Optional[str] = None  # e.g., "2h", "30m", "1d"
	max_duration: Optional[str] = None
	sla_target: Optional[str] = None
	
	# Permissions
	view_roles: List[str] = field(default_factory=list)
	execute_roles: List[str] = field(default_factory=list)
	
	# Visual Properties
	position: Optional[Dict[str, float]] = None
	style: Optional[Dict[str, Any]] = None
	
	# Connections
	next_elements: List[str] = field(default_factory=list)
	previous_elements: List[str] = field(default_factory=list)
	
	# Additional Properties
	properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APGWorkflowDefinition:
	"""Complete APG workflow definition."""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	version: str = "1.0"
	category: str = ""
	
	# Metadata
	author: str = ""
	created_date: Optional[datetime] = None
	tags: List[str] = field(default_factory=list)
	
	# Process Configuration
	start_element: str = ""
	elements: Dict[str, APGWorkflowElement] = field(default_factory=dict)
	flows: List[Dict[str, str]] = field(default_factory=list)
	
	# Global Settings
	process_timeout: Optional[str] = None
	retry_policy: Dict[str, Any] = field(default_factory=dict)
	error_handling: Dict[str, Any] = field(default_factory=dict)
	
	# Scheduling
	schedule_config: Optional[Dict[str, Any]] = None
	
	# Permissions
	process_permissions: Dict[str, List[str]] = field(default_factory=dict)
	
	# Integration
	integrations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	
	# Variables
	process_variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# APG Workflow Parser
# =============================================================================

class APGWorkflowParser:
	"""Parser for APG workflow definitions in various formats."""
	
	def __init__(self):
		self.supported_formats = {
			APGWorkflowFormat.YAML: self._parse_yaml,
			APGWorkflowFormat.JSON: self._parse_json,
			APGWorkflowFormat.APG_DSL: self._parse_apg_dsl,
			APGWorkflowFormat.NATURAL_LANGUAGE: self._parse_natural_language,
			APGWorkflowFormat.BPMN_XML: self._parse_bpmn_xml
		}
		
		# DSL Keywords and patterns
		self.dsl_keywords = self._initialize_dsl_keywords()
		self.duration_pattern = re.compile(r'(\d+)([smhd])')  # seconds, minutes, hours, days
		
		# Natural language processing patterns
		self.nl_patterns = self._initialize_nl_patterns()
	
	
	async def parse_workflow_definition(
		self,
		definition_text: str,
		format_type: APGWorkflowFormat,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Parse workflow definition from text."""
		try:
			# Detect format if not specified
			if format_type == APGWorkflowFormat.NATURAL_LANGUAGE:
				detected_format = await self._detect_format(definition_text)
				if detected_format != APGWorkflowFormat.NATURAL_LANGUAGE:
					format_type = detected_format
			
			# Parse using appropriate parser
			parser_func = self.supported_formats.get(format_type)
			if not parser_func:
				return WBPMServiceResponse(
					success=False,
					message=f"Unsupported format: {format_type}"
				)
			
			workflow_definition = await parser_func(definition_text, context)
			
			# Validate parsed definition
			validation_result = await self._validate_workflow_definition(workflow_definition)
			if not validation_result.success:
				return validation_result
			
			return WBPMServiceResponse(
				success=True,
				message="Workflow definition parsed successfully",
				data={
					"workflow_definition": workflow_definition,
					"format": format_type,
					"element_count": len(workflow_definition.elements),
					"flow_count": len(workflow_definition.flows)
				}
			)
			
		except Exception as e:
			logger.error(f"Error parsing workflow definition: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to parse workflow definition: {str(e)}"
			)
	
	
	# =============================================================================
	# Format-Specific Parsers
	# =============================================================================
	
	async def _parse_yaml(self, yaml_text: str, context: APGTenantContext) -> APGWorkflowDefinition:
		"""Parse YAML workflow definition."""
		try:
			data = yaml.safe_load(yaml_text)
			return await self._convert_dict_to_workflow(data, context)
		except yaml.YAMLError as e:
			raise ValueError(f"Invalid YAML format: {e}")
	
	
	async def _parse_json(self, json_text: str, context: APGTenantContext) -> APGWorkflowDefinition:
		"""Parse JSON workflow definition."""
		try:
			data = json.loads(json_text)
			return await self._convert_dict_to_workflow(data, context)
		except json.JSONDecodeError as e:
			raise ValueError(f"Invalid JSON format: {e}")
	
	
	async def _parse_apg_dsl(self, dsl_text: str, context: APGTenantContext) -> APGWorkflowDefinition:
		"""Parse APG Domain Specific Language workflow definition."""
		workflow = APGWorkflowDefinition()
		workflow.created_date = datetime.utcnow()
		workflow.author = context.user_id
		
		lines = [line.strip() for line in dsl_text.split('\n') if line.strip()]
		current_element = None
		
		for line in lines:
			# Process metadata
			if line.startswith('WORKFLOW'):
				workflow.name = self._extract_quoted_text(line) or "Unnamed Workflow"
			elif line.startswith('DESCRIPTION'):
				workflow.description = self._extract_quoted_text(line) or ""
			elif line.startswith('VERSION'):
				workflow.version = self._extract_quoted_text(line) or "1.0"
			elif line.startswith('CATEGORY'):
				workflow.category = self._extract_quoted_text(line) or ""
			
			# Process elements
			elif line.startswith('START'):
				element = APGWorkflowElement(
					id=f"start_{uuid7str()[:8]}",
					type=APGElementType.START,
					name=self._extract_quoted_text(line) or "Start"
				)
				workflow.elements[element.id] = element
				workflow.start_element = element.id
				current_element = element
			
			elif line.startswith('END'):
				element = APGWorkflowElement(
					id=f"end_{uuid7str()[:8]}",
					type=APGElementType.END,
					name=self._extract_quoted_text(line) or "End"
				)
				workflow.elements[element.id] = element
				current_element = element
			
			elif line.startswith('TASK'):
				element = await self._parse_dsl_task(line)
				workflow.elements[element.id] = element
				current_element = element
			
			elif line.startswith('USER_TASK'):
				element = await self._parse_dsl_user_task(line)
				workflow.elements[element.id] = element
				current_element = element
			
			elif line.startswith('SERVICE_TASK'):
				element = await self._parse_dsl_service_task(line)
				workflow.elements[element.id] = element
				current_element = element
			
			elif line.startswith('DECISION'):
				element = await self._parse_dsl_decision(line)
				workflow.elements[element.id] = element
				current_element = element
			
			elif line.startswith('PARALLEL'):
				element = await self._parse_dsl_parallel(line)
				workflow.elements[element.id] = element
				current_element = element
			
			# Process connections
			elif line.startswith('CONNECT') or '→' in line or '->' in line:
				flow = await self._parse_dsl_connection(line, workflow.elements)
				if flow:
					workflow.flows.append(flow)
			
			# Process element properties
			elif current_element and line.startswith('  '):
				await self._parse_dsl_element_property(line.strip(), current_element)
		
		return workflow
	
	
	async def _parse_natural_language(self, nl_text: str, context: APGTenantContext) -> APGWorkflowDefinition:
		"""Parse natural language workflow description."""
		workflow = APGWorkflowDefinition()
		workflow.created_date = datetime.utcnow()
		workflow.author = context.user_id
		
		# Extract workflow metadata from natural language
		workflow.name = await self._extract_workflow_name(nl_text)
		workflow.description = await self._extract_workflow_description(nl_text)
		
		# Parse steps and create elements
		steps = await self._extract_workflow_steps(nl_text)
		
		# Create start element
		start_element = APGWorkflowElement(
			id=f"start_{uuid7str()[:8]}",
			type=APGElementType.START,
			name="Process Start"
		)
		workflow.elements[start_element.id] = start_element
		workflow.start_element = start_element.id
		
		previous_element_id = start_element.id
		
		# Process each step
		for i, step in enumerate(steps):
			element = await self._create_element_from_step(step, i)
			workflow.elements[element.id] = element
			
			# Create flow from previous element
			workflow.flows.append({
				"from": previous_element_id,
				"to": element.id,
				"condition": ""
			})
			
			previous_element_id = element.id
		
		# Create end element
		end_element = APGWorkflowElement(
			id=f"end_{uuid7str()[:8]}",
			type=APGElementType.END,
			name="Process End"
		)
		workflow.elements[end_element.id] = end_element
		
		# Connect last step to end
		workflow.flows.append({
			"from": previous_element_id,
			"to": end_element.id,
			"condition": ""
		})
		
		return workflow
	
	
	async def _parse_bpmn_xml(self, xml_text: str, context: APGTenantContext) -> APGWorkflowDefinition:
		"""Parse BPMN 2.0 XML workflow definition."""
		import xml.etree.ElementTree as ET
		
		workflow = APGWorkflowDefinition()
		workflow.created_date = datetime.utcnow()
		workflow.author = context.user_id
		
		try:
			root = ET.fromstring(xml_text)
			
			# Find process element
			process = root.find('.//{http://www.omg.org/spec/BPMN/20100524/MODEL}process')
			if process is not None:
				workflow.name = process.get('name', 'Imported Process')
				workflow.id = process.get('id', workflow.id)
				
				# Parse elements
				for elem in process:
					await self._parse_bpmn_element(elem, workflow)
				
				# Parse sequence flows
				for flow in process.findall('.//{http://www.omg.org/spec/BPMN/20100524/MODEL}sequenceFlow'):
					workflow.flows.append({
						"from": flow.get('sourceRef', ''),
						"to": flow.get('targetRef', ''),
						"condition": flow.get('conditionExpression', '')
					})
		
		except ET.ParseError as e:
			raise ValueError(f"Invalid BPMN XML: {e}")
		
		return workflow
	
	
	# =============================================================================
	# Helper Methods for DSL Parsing
	# =============================================================================
	
	async def _parse_dsl_task(self, line: str) -> APGWorkflowElement:
		"""Parse DSL task definition."""
		name = self._extract_quoted_text(line) or "Task"
		return APGWorkflowElement(
			id=f"task_{uuid7str()[:8]}",
			type=APGElementType.TASK,
			name=name
		)
	
	
	async def _parse_dsl_user_task(self, line: str) -> APGWorkflowElement:
		"""Parse DSL user task definition."""
		name = self._extract_quoted_text(line) or "User Task"
		return APGWorkflowElement(
			id=f"user_task_{uuid7str()[:8]}",
			type=APGElementType.USER_TASK,
			name=name
		)
	
	
	async def _parse_dsl_service_task(self, line: str) -> APGWorkflowElement:
		"""Parse DSL service task definition."""
		name = self._extract_quoted_text(line) or "Service Task"
		return APGWorkflowElement(
			id=f"service_task_{uuid7str()[:8]}",
			type=APGElementType.SERVICE_TASK,
			name=name
		)
	
	
	async def _parse_dsl_decision(self, line: str) -> APGWorkflowElement:
		"""Parse DSL decision definition."""
		name = self._extract_quoted_text(line) or "Decision"
		return APGWorkflowElement(
			id=f"decision_{uuid7str()[:8]}",
			type=APGElementType.DECISION,
			name=name
		)
	
	
	async def _parse_dsl_parallel(self, line: str) -> APGWorkflowElement:
		"""Parse DSL parallel gateway definition."""
		name = self._extract_quoted_text(line) or "Parallel Gateway"
		return APGWorkflowElement(
			id=f"parallel_{uuid7str()[:8]}",
			type=APGElementType.PARALLEL,
			name=name
		)
	
	
	async def _parse_dsl_connection(self, line: str, elements: Dict[str, APGWorkflowElement]) -> Optional[Dict[str, str]]:
		"""Parse DSL connection definition."""
		# Handle different connection formats
		if '→' in line:
			parts = line.split('→')
		elif '->' in line:
			parts = line.split('->')
		elif 'CONNECT' in line:
			# Format: CONNECT "element1" TO "element2"
			match = re.search(r'CONNECT\s+"([^"]+)"\s+TO\s+"([^"]+)"', line)
			if match:
				from_name, to_name = match.groups()
				from_id = self._find_element_by_name(from_name, elements)
				to_id = self._find_element_by_name(to_name, elements)
				if from_id and to_id:
					return {"from": from_id, "to": to_id, "condition": ""}
		else:
			return None
		
		if len(parts) == 2:
			from_name = parts[0].strip().strip('"')
			to_name = parts[1].strip().strip('"')
			from_id = self._find_element_by_name(from_name, elements)
			to_id = self._find_element_by_name(to_name, elements)
			if from_id and to_id:
				return {"from": from_id, "to": to_id, "condition": ""}
		
		return None
	
	
	async def _parse_dsl_element_property(self, line: str, element: APGWorkflowElement) -> None:
		"""Parse element property from DSL."""
		if line.startswith('assignee:'):
			element.assignee = line.split(':', 1)[1].strip().strip('"')
		elif line.startswith('groups:'):
			groups_text = line.split(':', 1)[1].strip()
			element.candidate_groups = [g.strip().strip('"') for g in groups_text.split(',')]
		elif line.startswith('duration:'):
			element.estimated_duration = line.split(':', 1)[1].strip().strip('"')
		elif line.startswith('sla:'):
			element.sla_target = line.split(':', 1)[1].strip().strip('"')
		elif line.startswith('url:'):
			element.service_url = line.split(':', 1)[1].strip().strip('"')
		elif line.startswith('method:'):
			element.service_method = line.split(':', 1)[1].strip().strip('"')
		elif line.startswith('description:'):
			element.description = line.split(':', 1)[1].strip().strip('"')
	
	
	# =============================================================================
	# Helper Methods for Natural Language Parsing
	# =============================================================================
	
	async def _extract_workflow_name(self, text: str) -> str:
		"""Extract workflow name from natural language text."""
		# Look for patterns like "Process for...", "Workflow to...", etc.
		patterns = [
			r'(?:process|workflow)\s+(?:for|to)\s+([^.]+)',
			r'([^.]+)\s+(?:process|workflow)',
			r'^([^.]+)'  # First sentence
		]
		
		for pattern in patterns:
			match = re.search(pattern, text, re.IGNORECASE)
			if match:
				name = match.group(1).strip()
				if len(name) > 5:  # Reasonable length
					return name[:100]  # Limit length
		
		return "Imported Workflow"
	
	
	async def _extract_workflow_description(self, text: str) -> str:
		"""Extract workflow description from natural language text."""
		# Use first paragraph or first few sentences
		sentences = text.split('.')
		description = '. '.join(sentences[:3])
		return description[:500]  # Limit length
	
	
	async def _extract_workflow_steps(self, text: str) -> List[str]:
		"""Extract workflow steps from natural language text."""
		steps = []
		
		# Look for numbered lists
		numbered_pattern = r'^\s*(\d+)\.?\s+(.+)$'
		lines = text.split('\n')
		
		for line in lines:
			match = re.match(numbered_pattern, line)
			if match:
				steps.append(match.group(2).strip())
		
		# If no numbered list found, look for bullet points
		if not steps:
			bullet_pattern = r'^\s*[-•*]\s+(.+)$'
			for line in lines:
				match = re.match(bullet_pattern, line)
				if match:
					steps.append(match.group(1).strip())
		
		# If still no steps, try to identify action words
		if not steps:
			action_words = ['create', 'send', 'review', 'approve', 'process', 'generate', 'update', 'delete']
			sentences = text.split('.')
			
			for sentence in sentences:
				for action in action_words:
					if action in sentence.lower():
						steps.append(sentence.strip())
						break
		
		return steps[:20]  # Limit to 20 steps
	
	
	async def _create_element_from_step(self, step: str, index: int) -> APGWorkflowElement:
		"""Create workflow element from natural language step."""
		# Determine element type based on keywords
		step_lower = step.lower()
		
		if any(word in step_lower for word in ['approve', 'review', 'verify', 'check']):
			element_type = APGElementType.USER_TASK
		elif any(word in step_lower for word in ['send', 'email', 'notify', 'generate', 'create']):
			element_type = APGElementType.SERVICE_TASK
		elif any(word in step_lower for word in ['if', 'when', 'decide', 'choose']):
			element_type = APGElementType.DECISION
		elif any(word in step_lower for word in ['wait', 'pause', 'delay']):
			element_type = APGElementType.WAIT
		else:
			element_type = APGElementType.TASK
		
		return APGWorkflowElement(
			id=f"step_{index}_{uuid7str()[:8]}",
			type=element_type,
			name=step[:50],  # Limit name length
			description=step
		)
	
	
	# =============================================================================
	# Helper Methods for BPMN Parsing
	# =============================================================================
	
	async def _parse_bpmn_element(self, elem, workflow: APGWorkflowDefinition) -> None:
		"""Parse BPMN element and add to workflow."""
		tag_name = elem.tag.split('}')[-1]  # Remove namespace
		element_id = elem.get('id', '')
		element_name = elem.get('name', '')
		
		if tag_name == 'startEvent':
			element = APGWorkflowElement(
				id=element_id,
				type=APGElementType.START,
				name=element_name or "Start"
			)
			workflow.elements[element_id] = element
			workflow.start_element = element_id
		
		elif tag_name == 'endEvent':
			element = APGWorkflowElement(
				id=element_id,
				type=APGElementType.END,
				name=element_name or "End"
			)
			workflow.elements[element_id] = element
		
		elif tag_name == 'userTask':
			element = APGWorkflowElement(
				id=element_id,
				type=APGElementType.USER_TASK,
				name=element_name or "User Task"
			)
			workflow.elements[element_id] = element
		
		elif tag_name == 'serviceTask':
			element = APGWorkflowElement(
				id=element_id,
				type=APGElementType.SERVICE_TASK,
				name=element_name or "Service Task"
			)
			workflow.elements[element_id] = element
		
		elif tag_name in ['exclusiveGateway', 'inclusiveGateway']:
			element = APGWorkflowElement(
				id=element_id,
				type=APGElementType.DECISION,
				name=element_name or "Decision"
			)
			workflow.elements[element_id] = element
		
		elif tag_name == 'parallelGateway':
			element = APGWorkflowElement(
				id=element_id,
				type=APGElementType.PARALLEL,
				name=element_name or "Parallel Gateway"
			)
			workflow.elements[element_id] = element
	
	
	# =============================================================================
	# Utility Methods
	# =============================================================================
	
	async def _detect_format(self, text: str) -> APGWorkflowFormat:
		"""Detect workflow definition format."""
		text = text.strip()
		
		if text.startswith('{') or text.startswith('['):
			return APGWorkflowFormat.JSON
		elif text.startswith('<?xml') or '<definitions' in text:
			return APGWorkflowFormat.BPMN_XML
		elif any(keyword in text.upper() for keyword in ['WORKFLOW', 'START', 'END', 'TASK']):
			return APGWorkflowFormat.APG_DSL
		elif text.startswith('---') or re.search(r'^\w+:', text, re.MULTILINE):
			return APGWorkflowFormat.YAML
		else:
			return APGWorkflowFormat.NATURAL_LANGUAGE
	
	
	def _extract_quoted_text(self, line: str) -> Optional[str]:
		"""Extract text within quotes."""
		match = re.search(r'"([^"]*)"', line)
		return match.group(1) if match else None
	
	
	def _find_element_by_name(self, name: str, elements: Dict[str, APGWorkflowElement]) -> Optional[str]:
		"""Find element ID by name."""
		for element_id, element in elements.items():
			if element.name == name:
				return element_id
		return None
	
	
	async def _convert_dict_to_workflow(self, data: Dict[str, Any], context: APGTenantContext) -> APGWorkflowDefinition:
		"""Convert dictionary data to workflow definition."""
		workflow = APGWorkflowDefinition()
		workflow.created_date = datetime.utcnow()
		workflow.author = context.user_id
		
		# Basic metadata
		workflow.name = data.get('name', 'Unnamed Workflow')
		workflow.description = data.get('description', '')
		workflow.version = data.get('version', '1.0')
		workflow.category = data.get('category', '')
		
		# Process elements
		elements_data = data.get('elements', {})
		for element_id, element_data in elements_data.items():
			element = APGWorkflowElement(id=element_id)
			
			# Basic properties
			element.type = APGElementType(element_data.get('type', 'task'))
			element.name = element_data.get('name', '')
			element.description = element_data.get('description', '')
			
			# Task configuration
			element.assignee = element_data.get('assignee')
			element.candidate_groups = element_data.get('candidate_groups', [])
			element.form_fields = element_data.get('form_fields', [])
			
			# Service configuration
			element.service_type = element_data.get('service_type')
			element.service_url = element_data.get('service_url')
			element.service_method = element_data.get('service_method', 'POST')
			
			# Timing
			element.estimated_duration = element_data.get('estimated_duration')
			element.max_duration = element_data.get('max_duration')
			element.sla_target = element_data.get('sla_target')
			
			# Permissions
			element.view_roles = element_data.get('view_roles', [])
			element.execute_roles = element_data.get('execute_roles', [])
			
			# Properties
			element.properties = element_data.get('properties', {})
			
			workflow.elements[element_id] = element
		
		# Process flows
		flows_data = data.get('flows', [])
		for flow_data in flows_data:
			workflow.flows.append({
				'from': flow_data.get('from', ''),
				'to': flow_data.get('to', ''),
				'condition': flow_data.get('condition', '')
			})
		
		# Start element
		workflow.start_element = data.get('start_element', '')
		
		# Global settings
		workflow.process_timeout = data.get('process_timeout')
		workflow.retry_policy = data.get('retry_policy', {})
		workflow.error_handling = data.get('error_handling', {})
		workflow.schedule_config = data.get('schedule_config')
		workflow.process_permissions = data.get('process_permissions', {})
		workflow.integrations = data.get('integrations', {})
		workflow.process_variables = data.get('process_variables', {})
		
		return workflow
	
	
	async def _validate_workflow_definition(self, workflow: APGWorkflowDefinition) -> WBPMServiceResponse:
		"""Validate parsed workflow definition."""
		errors = []
		
		# Basic validation
		if not workflow.name:
			errors.append("Workflow name is required")
		
		if not workflow.elements:
			errors.append("Workflow must have at least one element")
		
		if not workflow.start_element:
			errors.append("Workflow must have a start element")
		elif workflow.start_element not in workflow.elements:
			errors.append("Start element not found in elements")
		
		# Validate elements
		for element_id, element in workflow.elements.items():
			if not element.name:
				errors.append(f"Element {element_id} must have a name")
		
		# Validate flows
		for flow in workflow.flows:
			if flow['from'] not in workflow.elements:
				errors.append(f"Flow source '{flow['from']}' not found in elements")
			if flow['to'] not in workflow.elements:
				errors.append(f"Flow target '{flow['to']}' not found in elements")
		
		if errors:
			return WBPMServiceResponse(
				success=False,
				message="Workflow validation failed",
				data={"errors": errors}
			)
		
		return WBPMServiceResponse(
			success=True,
			message="Workflow validation passed"
		)
	
	
	def _initialize_dsl_keywords(self) -> Dict[str, Any]:
		"""Initialize DSL keywords and patterns."""
		return {
			'workflow_keywords': ['WORKFLOW', 'PROCESS'],
			'element_keywords': ['START', 'END', 'TASK', 'USER_TASK', 'SERVICE_TASK', 'DECISION', 'PARALLEL'],
			'connection_keywords': ['CONNECT', 'TO', '→', '->'],
			'property_keywords': ['assignee', 'groups', 'duration', 'sla', 'url', 'method', 'description']
		}
	
	
	def _initialize_nl_patterns(self) -> Dict[str, Any]:
		"""Initialize natural language processing patterns."""
		return {
			'action_words': ['create', 'send', 'review', 'approve', 'process', 'generate', 'update', 'delete'],
			'user_actions': ['approve', 'review', 'verify', 'check', 'validate'],
			'system_actions': ['send', 'email', 'notify', 'generate', 'create', 'update'],
			'decision_words': ['if', 'when', 'decide', 'choose', 'determine'],
			'wait_words': ['wait', 'pause', 'delay', 'hold']
		}


# =============================================================================
# APG Workflow Implementer
# =============================================================================

class APGWorkflowImplementer:
	"""Implements parsed APG workflow definitions into executable workflows."""
	
	def __init__(self, visual_designer: EnhancedVisualDesignerService, scheduler: WorkflowScheduler):
		self.visual_designer = visual_designer
		self.scheduler = scheduler
		
		# Element type mapping to BPMN types
		self.type_mapping = {
			APGElementType.START: "startEvent",
			APGElementType.END: "endEvent",
			APGElementType.TASK: "task",
			APGElementType.USER_TASK: "userTask",
			APGElementType.SERVICE_TASK: "serviceTask",
			APGElementType.DECISION: "exclusiveGateway",
			APGElementType.PARALLEL: "parallelGateway",
			APGElementType.WAIT: "intermediateCatchEvent",
			APGElementType.TIMER: "intermediateTimerEvent",
			APGElementType.SUBPROCESS: "subProcess"
		}
	
	
	async def implement_workflow(
		self,
		workflow_definition: APGWorkflowDefinition,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Implement APG workflow definition as executable workflow."""
		try:
			# Create visual canvas
			canvas_result = await self.visual_designer.create_canvas(
				name=workflow_definition.name,
				context=context
			)
			
			if not canvas_result.success:
				return canvas_result
			
			canvas_id = canvas_result.data['canvas_id']
			
			# Position elements for visual layout
			element_positions = await self._calculate_element_positions(workflow_definition)
			
			# Create visual elements
			element_mapping = {}
			for element_id, element_def in workflow_definition.elements.items():
				visual_element_result = await self._create_visual_element(
					canvas_id, element_def, element_positions.get(element_id), context
				)
				
				if visual_element_result.success:
					visual_element_id = visual_element_result.data['element_id']
					element_mapping[element_id] = visual_element_id
			
			# Create connections
			for flow in workflow_definition.flows:
				source_id = element_mapping.get(flow['from'])
				target_id = element_mapping.get(flow['to'])
				
				if source_id and target_id:
					await self.visual_designer.create_connection(
						canvas_id=canvas_id,
						source_id=source_id,
						target_id=target_id,
						context=context,
						properties={'condition': flow.get('condition', '')}
					)
			
			# Configure process-level settings
			if workflow_definition.process_timeout:
				process_timing = await self._parse_timing_configuration(workflow_definition.process_timeout)
				await self.visual_designer.configure_process_timing(
					canvas_id=canvas_id,
					timing_config=process_timing,
					context=context
				)
			
			# Create schedule if configured
			schedule_id = None
			if workflow_definition.schedule_config:
				schedule_result = await self._create_workflow_schedule(
					workflow_definition, canvas_id, context
				)
				if schedule_result.success:
					schedule_id = schedule_result.data.get('schedule_id')
			
			# Save the canvas
			await self.visual_designer.save_canvas(canvas_id, context)
			
			logger.info(f"Implemented workflow {workflow_definition.name} as canvas {canvas_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Workflow implemented successfully",
				data={
					"workflow_id": workflow_definition.id,
					"canvas_id": canvas_id,
					"schedule_id": schedule_id,
					"element_count": len(workflow_definition.elements),
					"element_mapping": element_mapping
				}
			)
			
		except Exception as e:
			logger.error(f"Error implementing workflow: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to implement workflow: {str(e)}"
			)
	
	
	async def _create_visual_element(
		self,
		canvas_id: str,
		element_def: APGWorkflowElement,
		position: Optional[VisualPosition],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create visual element from APG element definition."""
		# Map APG element type to BPMN type
		bpmn_type = self.type_mapping.get(element_def.type, "task")
		
		# Create visual position
		if not position:
			position = VisualPosition(x=100, y=100, width=100, height=80)
		
		# Create the element
		element_result = await self.visual_designer.add_element(
			canvas_id=canvas_id,
			element_type=bpmn_type,
			position=position,
			context=context,
			element_name=element_def.name,
			properties=element_def.properties
		)
		
		if not element_result.success:
			return element_result
		
		visual_element_id = element_result.data['element_id']
		
		# Configure timing if specified
		if element_def.estimated_duration or element_def.sla_target:
			timing_config = TimingConfiguration()
			
			if element_def.estimated_duration:
				duration_minutes = await self._parse_duration(element_def.estimated_duration)
				timing_config.estimated_duration_minutes = duration_minutes
			
			if element_def.max_duration:
				max_duration_minutes = await self._parse_duration(element_def.max_duration)
				timing_config.max_duration_minutes = max_duration_minutes
			
			if element_def.sla_target:
				sla_minutes = await self._parse_duration(element_def.sla_target)
				timing_config.sla_target_minutes = sla_minutes
			
			await self.visual_designer.configure_element_timing(
				canvas_id=canvas_id,
				element_id=visual_element_id,
				timing_config=timing_config,
				context=context
			)
		
		# Configure permissions if specified
		if element_def.view_roles or element_def.execute_roles:
			permissions = ElementPermissions(
				view_roles=element_def.view_roles,
				execute_roles=element_def.execute_roles,
				edit_roles=element_def.execute_roles,  # Same as execute for now
				assign_roles=element_def.execute_roles
			)
			
			await self.visual_designer.configure_element_permissions(
				canvas_id=canvas_id,
				element_id=visual_element_id,
				permissions=permissions,
				context=context
			)
		
		return element_result
	
	
	async def _calculate_element_positions(self, workflow_definition: APGWorkflowDefinition) -> Dict[str, VisualPosition]:
		"""Calculate visual positions for workflow elements."""
		positions = {}
		
		# Simple grid layout algorithm
		elements = list(workflow_definition.elements.keys())
		cols = max(3, int(len(elements) ** 0.5))
		
		for i, element_id in enumerate(elements):
			row = i // cols
			col = i % cols
			
			x = 100 + col * 200
			y = 100 + row * 150
			
			positions[element_id] = VisualPosition(x=x, y=y, width=120, height=80)
		
		return positions
	
	
	async def _parse_duration(self, duration_str: str) -> int:
		"""Parse duration string to minutes."""
		if not duration_str:
			return 0
		
		# Handle formats like "2h", "30m", "1d", "90s"
		match = re.match(r'(\d+)([smhd])', duration_str.lower())
		if match:
			value, unit = match.groups()
			value = int(value)
			
			if unit == 's':
				return max(1, value // 60)  # Convert seconds to minutes
			elif unit == 'm':
				return value
			elif unit == 'h':
				return value * 60
			elif unit == 'd':
				return value * 24 * 60
		
		# Try to parse as plain number (assume minutes)
		try:
			return int(duration_str)
		except ValueError:
			return 60  # Default to 1 hour
	
	
	async def _parse_timing_configuration(self, timeout_str: str) -> TimingConfiguration:
		"""Parse timeout string to timing configuration."""
		timeout_minutes = await self._parse_duration(timeout_str)
		
		return TimingConfiguration(
			max_duration_minutes=timeout_minutes,
			warning_threshold_percent=80,
			escalation_threshold_percent=100
		)
	
	
	async def _create_workflow_schedule(
		self,
		workflow_definition: APGWorkflowDefinition,
		canvas_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create workflow schedule from definition."""
		schedule_config = workflow_definition.schedule_config
		if not schedule_config:
			return WBPMServiceResponse(success=False, message="No schedule configuration")
		
		from workflow_scheduler import WorkflowSchedule, ScheduleType
		
		# Create schedule based on configuration
		schedule = WorkflowSchedule(
			tenant_id=context.tenant_id,
			created_by=context.user_id,
			updated_by=context.user_id,
			name=f"Schedule for {workflow_definition.name}",
			process_definition_id=canvas_id,
			input_variables=schedule_config.get('variables', {})
		)
		
		# Configure schedule type
		if 'cron' in schedule_config:
			schedule.schedule_type = ScheduleType.CRON
			schedule.cron_expression = schedule_config['cron']
		elif 'interval' in schedule_config:
			schedule.schedule_type = ScheduleType.RECURRING
			schedule.interval_minutes = schedule_config['interval']
		elif 'start_time' in schedule_config:
			schedule.schedule_type = ScheduleType.ONE_TIME
			schedule.start_time = datetime.fromisoformat(schedule_config['start_time'])
		
		# Create the schedule
		return await self.scheduler.create_schedule(schedule)


# =============================================================================
# Combined Parser and Implementer Service
# =============================================================================

class APGWorkflowService:
	"""Combined service for parsing and implementing APG workflow definitions."""
	
	def __init__(self, visual_designer: EnhancedVisualDesignerService, scheduler: WorkflowScheduler):
		self.parser = APGWorkflowParser()
		self.implementer = APGWorkflowImplementer(visual_designer, scheduler)
	
	
	async def create_workflow_from_definition(
		self,
		definition_text: str,
		format_type: APGWorkflowFormat,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Parse and implement workflow definition in one operation."""
		try:
			# Parse the definition
			parse_result = await self.parser.parse_workflow_definition(
				definition_text, format_type, context
			)
			
			if not parse_result.success:
				return parse_result
			
			workflow_definition = parse_result.data['workflow_definition']
			
			# Implement the workflow
			implement_result = await self.implementer.implement_workflow(
				workflow_definition, context
			)
			
			if not implement_result.success:
				return implement_result
			
			return WBPMServiceResponse(
				success=True,
				message="Workflow created successfully from definition",
				data={
					"workflow_definition": workflow_definition,
					"implementation": implement_result.data,
					"format": format_type
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating workflow from definition: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create workflow from definition: {str(e)}"
			)


# =============================================================================
# Example Usage
# =============================================================================

async def example_apg_workflow_usage():
	"""Example usage of APG workflow parser and implementer."""
	from enhanced_visual_designer import EnhancedVisualDesignerService
	from workflow_scheduler import SchedulerFactory
	
	# Create tenant context
	context = APGTenantContext(
		tenant_id="example_tenant",
		user_id="admin@example.com",
		user_roles=["admin"],
		permissions=["workflow_design", "workflow_execute"]
	)
	
	# Create services
	scheduler = await SchedulerFactory.get_scheduler(context)
	visual_designer = EnhancedVisualDesignerService(scheduler)
	workflow_service = APGWorkflowService(visual_designer, scheduler)
	
	# Example 1: Parse APG DSL
	apg_dsl = '''
WORKFLOW "Employee Onboarding Process"
DESCRIPTION "Complete onboarding process for new employees"
VERSION "1.0"

START "New Employee"
USER_TASK "Complete HR Forms"
  assignee: "hr@company.com"
  duration: "2h"
  sla: "4h"
SERVICE_TASK "Create Email Account"
  url: "https://api.company.com/accounts"
  method: "POST"
DECISION "Manager Approval Required?"
USER_TASK "Manager Approval"
  groups: "managers"
  duration: "1h"
END "Onboarding Complete"

"New Employee" → "Complete HR Forms"
"Complete HR Forms" → "Create Email Account"
"Create Email Account" → "Manager Approval Required?"
"Manager Approval Required?" → "Manager Approval"
"Manager Approval" → "Onboarding Complete"
'''
	
	result = await workflow_service.create_workflow_from_definition(
		apg_dsl, APGWorkflowFormat.APG_DSL, context
	)
	print(f"APG DSL workflow created: {result.success}")
	
	# Example 2: Parse Natural Language
	natural_language = '''
Employee Onboarding Workflow

1. New employee completes HR forms
2. System creates email account
3. Manager reviews and approves setup
4. Send welcome email to employee
5. Process is complete
'''
	
	result = await workflow_service.create_workflow_from_definition(
		natural_language, APGWorkflowFormat.NATURAL_LANGUAGE, context
	)
	print(f"Natural language workflow created: {result.success}")
	
	# Example 3: Parse YAML
	yaml_definition = '''
name: "Invoice Processing"
description: "Automated invoice processing workflow"
version: "1.0"

elements:
  start_1:
    type: "start"
    name: "Invoice Received"
  
  validate_invoice:
    type: "service_task"
    name: "Validate Invoice"
    service_url: "https://api.company.com/validate"
    estimated_duration: "5m"
  
  approve_invoice:
    type: "user_task"
    name: "Approve Invoice"
    candidate_groups: ["finance"]
    sla_target: "2h"
  
  end_1:
    type: "end"
    name: "Invoice Processed"

flows:
  - from: "start_1"
    to: "validate_invoice"
  - from: "validate_invoice"
    to: "approve_invoice"
  - from: "approve_invoice"
    to: "end_1"

start_element: "start_1"
'''
	
	result = await workflow_service.create_workflow_from_definition(
		yaml_definition, APGWorkflowFormat.YAML, context
	)
	print(f"YAML workflow created: {result.success}")


if __name__ == "__main__":
	asyncio.run(example_apg_workflow_usage())