"""
APG Workflow Validation Engine

Real-time validation engine for workflow integrity and correctness.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

class ValidationSeverity(str, Enum):
	"""Validation issue severity levels."""
	ERROR = "error"
	WARNING = "warning"
	INFO = "info"

class ValidationIssue(BaseModel):
	"""Represents a validation issue."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Issue ID")
	severity: ValidationSeverity = Field(..., description="Issue severity")
	message: str = Field(..., description="Issue message")
	description: str = Field(default="", description="Detailed description")
	
	# Location
	node_id: Optional[str] = Field(default=None, description="Associated node ID")
	connection_id: Optional[str] = Field(default=None, description="Associated connection ID")
	property_name: Optional[str] = Field(default=None, description="Associated property name")
	
	# Fix suggestions
	fix_suggestion: Optional[str] = Field(default=None, description="Suggested fix")
	auto_fixable: bool = Field(default=False, description="Whether issue can be auto-fixed")
	
	# Metadata
	rule_id: str = Field(..., description="Validation rule ID")
	category: str = Field(..., description="Issue category")

class ValidationResult(BaseModel):
	"""Result of workflow validation."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	valid: bool = Field(..., description="Overall validation result")
	issues: List[ValidationIssue] = Field(default_factory=list, description="Validation issues")
	
	# Statistics
	error_count: int = Field(default=0, description="Number of errors")
	warning_count: int = Field(default=0, description="Number of warnings")
	info_count: int = Field(default=0, description="Number of info messages")
	
	# Metadata
	validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	validation_duration: float = Field(default=0.0, description="Validation duration in seconds")

class ValidationRule(BaseModel):
	"""Defines a validation rule."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Rule ID")
	name: str = Field(..., description="Rule name")
	description: str = Field(..., description="Rule description")
	category: str = Field(..., description="Rule category")
	severity: ValidationSeverity = Field(..., description="Default severity")
	enabled: bool = Field(default=True, description="Whether rule is enabled")

class ValidationEngine:
	"""
	Real-time validation engine for workflow integrity.
	
	Features:
	- Structural validation (nodes, connections, data flow)
	- Component configuration validation
	- Performance analysis and optimization suggestions
	- Security and compliance checks
	- Custom validation rules
	- Auto-fix suggestions
	"""
	
	def __init__(self, config):
		self.config = config
		self.validation_rules: Dict[str, ValidationRule] = {}
		self.is_initialized = False
		
		logger.info("Validation engine initialized")
	
	async def initialize(self) -> None:
		"""Initialize the validation engine."""
		try:
			# Load built-in validation rules
			await self._load_builtin_rules()
			
			# Load custom rules from database
			await self._load_custom_rules()
			
			self.is_initialized = True
			logger.info(f"Validation engine initialized with {len(self.validation_rules)} rules")
			
		except Exception as e:
			logger.error(f"Failed to initialize validation engine: {e}")
			raise
	
	async def shutdown(self) -> None:
		"""Shutdown the validation engine."""
		try:
			self.validation_rules.clear()
			self.is_initialized = False
			logger.info("Validation engine shutdown completed")
		except Exception as e:
			logger.error(f"Error during validation engine shutdown: {e}")
	
	async def validate_workflow(self, workflow_data: Dict[str, Any]) -> ValidationResult:
		"""Validate complete workflow."""
		try:
			start_time = datetime.now(timezone.utc)
			issues = []
			
			# Extract workflow components
			nodes = workflow_data.get('definition', {}).get('nodes', [])
			connections = workflow_data.get('definition', {}).get('connections', [])
			
			# Run validation rules
			issues.extend(await self._validate_structure(nodes, connections))
			issues.extend(await self._validate_nodes(nodes))
			issues.extend(await self._validate_connections(nodes, connections))
			issues.extend(await self._validate_data_flow(nodes, connections))
			issues.extend(await self._validate_performance(nodes, connections))
			issues.extend(await self._validate_security(workflow_data))
			
			# Calculate statistics
			error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
			warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
			info_count = len([i for i in issues if i.severity == ValidationSeverity.INFO])
			
			end_time = datetime.now(timezone.utc)
			duration = (end_time - start_time).total_seconds()
			
			result = ValidationResult(
				valid=error_count == 0,
				issues=issues,
				error_count=error_count,
				warning_count=warning_count,
				info_count=info_count,
				validated_at=start_time,
				validation_duration=duration
			)
			
			logger.debug(f"Validated workflow with {len(issues)} issues in {duration:.3f}s")
			return result
			
		except Exception as e:
			logger.error(f"Failed to validate workflow: {e}")
			return ValidationResult(
				valid=False,
				issues=[ValidationIssue(
					id="validation_error",
					severity=ValidationSeverity.ERROR,
					message=f"Validation failed: {e}",
					rule_id="system_error",
					category="system"
				)]
			)
	
	async def validate_node(self, node_data: Dict[str, Any], component_definition: Optional[Dict[str, Any]] = None) -> ValidationResult:
		"""Validate a single node."""
		try:
			issues = []
			
			# Basic node validation
			issues.extend(await self._validate_single_node(node_data, component_definition))
			
			# Calculate result
			error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
			
			return ValidationResult(
				valid=error_count == 0,
				issues=issues,
				error_count=error_count,
				warning_count=len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
				info_count=len([i for i in issues if i.severity == ValidationSeverity.INFO])
			)
			
		except Exception as e:
			logger.error(f"Failed to validate node: {e}")
			return ValidationResult(
				valid=False,
				issues=[ValidationIssue(
					id="node_validation_error",
					severity=ValidationSeverity.ERROR,
					message=f"Node validation failed: {e}",
					rule_id="system_error",
					category="system"
				)]
			)
	
	async def auto_fix_issues(self, workflow_data: Dict[str, Any], issue_ids: List[str]) -> Dict[str, Any]:
		"""Automatically fix specified validation issues."""
		try:
			fixed_issues = []
			failed_fixes = []
			
			# Validate workflow first to get current issues
			validation_result = await self.validate_workflow(workflow_data)
			
			for issue in validation_result.issues:
				if issue.id in issue_ids and issue.auto_fixable:
					try:
						await self._apply_auto_fix(workflow_data, issue)
						fixed_issues.append(issue.id)
					except Exception as e:
						failed_fixes.append({'issue_id': issue.id, 'error': str(e)})
			
			return {
				'fixed_issues': fixed_issues,
				'failed_fixes': failed_fixes,
				'workflow_data': workflow_data
			}
			
		except Exception as e:
			logger.error(f"Failed to auto-fix issues: {e}")
			raise
	
	async def get_validation_rules(self) -> List[ValidationRule]:
		"""Get all validation rules."""
		return list(self.validation_rules.values())
	
	async def enable_rule(self, rule_id: str, enabled: bool = True) -> None:
		"""Enable or disable a validation rule."""
		if rule_id in self.validation_rules:
			self.validation_rules[rule_id].enabled = enabled
	
	async def add_custom_rule(self, rule: ValidationRule) -> None:
		"""Add a custom validation rule."""
		self.validation_rules[rule.id] = rule
		logger.info(f"Added custom validation rule: {rule.id}")
	
	# Private validation methods
	
	async def _validate_structure(self, nodes: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> List[ValidationIssue]:
		"""Validate workflow structure."""
		issues = []
		
		try:
			# Check for empty workflow
			if not nodes:
				issues.append(ValidationIssue(
					id="empty_workflow",
					severity=ValidationSeverity.WARNING,
					message="Workflow is empty",
					description="Add components to create a functional workflow",
					rule_id="structure_empty",
					category="structure"
				))
			
			# Check for disconnected nodes
			connected_nodes = set()
			for conn in connections:
				connected_nodes.add(conn.get('source_node_id'))
				connected_nodes.add(conn.get('target_node_id'))
			
			node_ids = {node.get('id') for node in nodes}
			disconnected = node_ids - connected_nodes
			
			if len(disconnected) > 1:  # Allow single start node
				for node_id in disconnected:
					node = next((n for n in nodes if n.get('id') == node_id), None)
					if node and not self._is_trigger_node(node):
						issues.append(ValidationIssue(
							id=f"disconnected_node_{node_id}",
							severity=ValidationSeverity.WARNING,
							message="Node is not connected",
							description="Connect this node to the workflow",
							node_id=node_id,
							rule_id="structure_disconnected",
							category="structure",
							auto_fixable=True,
							fix_suggestion="Connect this node to other components"
						))
			
			# Check for cycles (simple detection)
			if self._has_cycles(connections):
				issues.append(ValidationIssue(
					id="workflow_cycle",
					severity=ValidationSeverity.ERROR,
					message="Workflow contains cycles",
					description="Remove circular dependencies to prevent infinite loops",
					rule_id="structure_cycle",
					category="structure"
				))
			
		except Exception as e:
			logger.error(f"Structure validation error: {e}")
		
		return issues
	
	async def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> List[ValidationIssue]:
		"""Validate individual nodes."""
		issues = []
		
		try:
			for node in nodes:
				node_issues = await self._validate_single_node(node)
				issues.extend(node_issues)
		except Exception as e:
			logger.error(f"Node validation error: {e}")
		
		return issues
	
	async def _validate_single_node(self, node: Dict[str, Any], component_def: Optional[Dict[str, Any]] = None) -> List[ValidationIssue]:
		"""Validate a single node."""
		issues = []
		node_id = node.get('id')
		
		try:
			# Required fields
			if not node.get('component_type'):
				issues.append(ValidationIssue(
					id=f"missing_component_type_{node_id}",
					severity=ValidationSeverity.ERROR,
					message="Component type is missing",
					node_id=node_id,
					rule_id="node_component_type",
					category="node"
				))
			
			# Configuration validation
			config = node.get('config', {})
			if component_def:
				# Validate against component definition
				required_props = [
					prop for prop in component_def.get('properties', [])
					if prop.get('required', False)
				]
				
				for prop in required_props:
					prop_name = prop.get('name')
					if prop_name not in config or config[prop_name] in [None, '']:
						issues.append(ValidationIssue(
							id=f"missing_required_prop_{node_id}_{prop_name}",
							severity=ValidationSeverity.ERROR,
							message=f"Required property '{prop.get('label', prop_name)}' is missing",
							node_id=node_id,
							property_name=prop_name,
							rule_id="node_required_property",
							category="configuration",
							auto_fixable=True,
							fix_suggestion=f"Set a value for {prop.get('label', prop_name)}"
						))
			
			# Position validation
			position = node.get('position', {})
			if not isinstance(position, dict) or 'x' not in position or 'y' not in position:
				issues.append(ValidationIssue(
					id=f"invalid_position_{node_id}",
					severity=ValidationSeverity.WARNING,
					message="Node position is invalid",
					node_id=node_id,
					rule_id="node_position",
					category="layout",
					auto_fixable=True,
					fix_suggestion="Reset node position"
				))
			
		except Exception as e:
			logger.error(f"Single node validation error: {e}")
		
		return issues
	
	async def _validate_connections(self, nodes: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> List[ValidationIssue]:
		"""Validate connections between nodes."""
		issues = []
		
		try:
			node_ids = {node.get('id') for node in nodes}
			
			for conn in connections:
				conn_id = conn.get('id')
				source_id = conn.get('source_node_id')
				target_id = conn.get('target_node_id')
				
				# Check if referenced nodes exist
				if source_id not in node_ids:
					issues.append(ValidationIssue(
						id=f"invalid_source_{conn_id}",
						severity=ValidationSeverity.ERROR,
						message="Connection source node not found",
						connection_id=conn_id,
						rule_id="connection_source",
						category="connection"
					))
				
				if target_id not in node_ids:
					issues.append(ValidationIssue(
						id=f"invalid_target_{conn_id}",
						severity=ValidationSeverity.ERROR,
						message="Connection target node not found",
						connection_id=conn_id,
						rule_id="connection_target",
						category="connection"
					))
				
				# Check for self-connections
				if source_id == target_id:
					issues.append(ValidationIssue(
						id=f"self_connection_{conn_id}",
						severity=ValidationSeverity.WARNING,
						message="Node connected to itself",
						connection_id=conn_id,
						rule_id="connection_self",
						category="connection"
					))
			
		except Exception as e:
			logger.error(f"Connection validation error: {e}")
		
		return issues
	
	async def _validate_data_flow(self, nodes: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> List[ValidationIssue]:
		"""Validate data flow through the workflow."""
		issues = []
		
		try:
			# Check for nodes with no inputs (except triggers)
			node_inputs = {}
			for conn in connections:
				target_id = conn.get('target_node_id')
				if target_id not in node_inputs:
					node_inputs[target_id] = []
				node_inputs[target_id].append(conn)
			
			for node in nodes:
				node_id = node.get('id')
				if node_id not in node_inputs and not self._is_trigger_node(node):
					issues.append(ValidationIssue(
						id=f"no_input_{node_id}",
						severity=ValidationSeverity.WARNING,
						message="Node has no input connections",
						description="This node will not receive data unless it's a trigger",
						node_id=node_id,
						rule_id="dataflow_no_input",
						category="dataflow"
					))
			
			# Check for nodes with no outputs (potential dead ends)
			node_outputs = {}
			for conn in connections:
				source_id = conn.get('source_node_id')
				if source_id not in node_outputs:
					node_outputs[source_id] = []
				node_outputs[source_id].append(conn)
			
			for node in nodes:
				node_id = node.get('id')
				if node_id not in node_outputs:
					issues.append(ValidationIssue(
						id=f"no_output_{node_id}",
						severity=ValidationSeverity.INFO,
						message="Node has no output connections",
						description="This node's output will not be used",
						node_id=node_id,
						rule_id="dataflow_no_output",
						category="dataflow"
					))
			
		except Exception as e:
			logger.error(f"Data flow validation error: {e}")
		
		return issues
	
	async def _validate_performance(self, nodes: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> List[ValidationIssue]:
		"""Validate workflow performance characteristics."""
		issues = []
		
		try:
			# Check workflow complexity
			if len(nodes) > 100:
				issues.append(ValidationIssue(
					id="high_complexity",
					severity=ValidationSeverity.WARNING,
					message="Workflow has high complexity",
					description=f"Workflow has {len(nodes)} nodes. Consider breaking it into smaller workflows",
					rule_id="performance_complexity",
					category="performance"
				))
			
			# Check for potential bottlenecks
			node_fan_in = {}
			for conn in connections:
				target_id = conn.get('target_node_id')
				node_fan_in[target_id] = node_fan_in.get(target_id, 0) + 1
			
			for node_id, fan_in in node_fan_in.items():
				if fan_in > 10:
					issues.append(ValidationIssue(
						id=f"high_fan_in_{node_id}",
						severity=ValidationSeverity.WARNING,
						message="Node has many input connections",
						description=f"Node has {fan_in} inputs, which may cause performance issues",
						node_id=node_id,
						rule_id="performance_fan_in",
						category="performance"
					))
			
		except Exception as e:
			logger.error(f"Performance validation error: {e}")
		
		return issues
	
	async def _validate_security(self, workflow_data: Dict[str, Any]) -> List[ValidationIssue]:
		"""Validate security aspects of the workflow."""
		issues = []
		
		try:
			# Check for sensitive data in configuration
			nodes = workflow_data.get('definition', {}).get('nodes', [])
			
			for node in nodes:
				config = node.get('config', {})
				node_id = node.get('id')
				
				# Look for potential secrets in plain text
				sensitive_patterns = ['password', 'secret', 'token', 'key', 'credential']
				
				for key, value in config.items():
					if isinstance(value, str):
						key_lower = key.lower()
						if any(pattern in key_lower for pattern in sensitive_patterns):
							if len(value) > 0 and not value.startswith('${'):  # Not a variable reference
								issues.append(ValidationIssue(
									id=f"plaintext_secret_{node_id}_{key}",
									severity=ValidationSeverity.WARNING,
									message="Potential secret in plain text",
									description=f"Property '{key}' may contain sensitive data",
									node_id=node_id,
									property_name=key,
									rule_id="security_plaintext_secret",
									category="security",
									fix_suggestion="Use environment variables or secret management"
								))
			
		except Exception as e:
			logger.error(f"Security validation error: {e}")
		
		return issues
	
	# Helper methods
	
	def _is_trigger_node(self, node: Dict[str, Any]) -> bool:
		"""Check if node is a trigger node."""
		component_type = node.get('component_type', '')
		trigger_types = ['http_trigger', 'schedule_trigger', 'file_trigger', 'manual_trigger']
		return any(trigger in component_type for trigger in trigger_types)
	
	def _has_cycles(self, connections: List[Dict[str, Any]]) -> bool:
		"""Detect cycles in the workflow graph."""
		try:
			# Build adjacency list
			graph = {}
			for conn in connections:
				source = conn.get('source_node_id')
				target = conn.get('target_node_id')
				if source and target:
					if source not in graph:
						graph[source] = []
					graph[source].append(target)
			
			# DFS cycle detection
			visited = set()
			rec_stack = set()
			
			def has_cycle_dfs(node):
				if node in rec_stack:
					return True
				if node in visited:
					return False
				
				visited.add(node)
				rec_stack.add(node)
				
				for neighbor in graph.get(node, []):
					if has_cycle_dfs(neighbor):
						return True
				
				rec_stack.remove(node)
				return False
			
			for node in graph:
				if node not in visited:
					if has_cycle_dfs(node):
						return True
			
			return False
			
		except Exception as e:
			logger.error(f"Cycle detection error: {e}")
			return False
	
	async def _apply_auto_fix(self, workflow_data: Dict[str, Any], issue: ValidationIssue) -> None:
		"""Apply automatic fix for an issue."""
		try:
			if issue.rule_id == "structure_disconnected":
				# Auto-fix: Remove disconnected non-trigger nodes
				nodes = workflow_data.get('definition', {}).get('nodes', [])
				workflow_data['definition']['nodes'] = [
					n for n in nodes if n.get('id') != issue.node_id or self._is_trigger_node(n)
				]
			
			elif issue.rule_id == "node_required_property":
				# Auto-fix: Set default value for required property
				nodes = workflow_data.get('definition', {}).get('nodes', [])
				for node in nodes:
					if node.get('id') == issue.node_id:
						if 'config' not in node:
							node['config'] = {}
						node['config'][issue.property_name] = ""
			
			elif issue.rule_id == "node_position":
				# Auto-fix: Reset node position
				nodes = workflow_data.get('definition', {}).get('nodes', [])
				for node in nodes:
					if node.get('id') == issue.node_id:
						node['position'] = {'x': 0, 'y': 0}
			
		except Exception as e:
			logger.error(f"Auto-fix error for issue {issue.id}: {e}")
			raise
	
	async def _load_builtin_rules(self) -> None:
		"""Load built-in validation rules."""
		try:
			builtin_rules = [
				ValidationRule(
					id="structure_empty",
					name="Empty Workflow",
					description="Check for empty workflows",
					category="structure",
					severity=ValidationSeverity.WARNING
				),
				ValidationRule(
					id="structure_disconnected",
					name="Disconnected Nodes",
					description="Check for disconnected nodes",
					category="structure",
					severity=ValidationSeverity.WARNING
				),
				ValidationRule(
					id="structure_cycle",
					name="Workflow Cycles",
					description="Check for circular dependencies",
					category="structure",
					severity=ValidationSeverity.ERROR
				),
				ValidationRule(
					id="node_component_type",
					name="Component Type",
					description="Check for missing component types",
					category="node",
					severity=ValidationSeverity.ERROR
				),
				ValidationRule(
					id="node_required_property",
					name="Required Properties",
					description="Check for missing required properties",
					category="configuration",
					severity=ValidationSeverity.ERROR
				),
				ValidationRule(
					id="connection_source",
					name="Connection Source",
					description="Check for invalid connection sources",
					category="connection",
					severity=ValidationSeverity.ERROR
				),
				ValidationRule(
					id="connection_target",
					name="Connection Target",
					description="Check for invalid connection targets",
					category="connection",
					severity=ValidationSeverity.ERROR
				),
				ValidationRule(
					id="dataflow_no_input",
					name="No Input Connections",
					description="Check for nodes without inputs",
					category="dataflow",
					severity=ValidationSeverity.WARNING
				),
				ValidationRule(
					id="performance_complexity",
					name="High Complexity",
					description="Check for overly complex workflows",
					category="performance",
					severity=ValidationSeverity.WARNING
				),
				ValidationRule(
					id="security_plaintext_secret",
					name="Plaintext Secrets",
					description="Check for secrets in plain text",
					category="security",
					severity=ValidationSeverity.WARNING
				)
			]
			
			for rule in builtin_rules:
				self.validation_rules[rule.id] = rule
			
		except Exception as e:
			logger.error(f"Failed to load builtin rules: {e}")
			raise
	
	async def _load_custom_rules(self) -> None:
		"""Load custom validation rules from database."""
		try:
			# Implementation would load custom rules from database
			# For now, we'll just log that we attempted to load them
			logger.info("Custom validation rules loading completed")
		except Exception as e:
			logger.error(f"Failed to load custom rules: {e}")