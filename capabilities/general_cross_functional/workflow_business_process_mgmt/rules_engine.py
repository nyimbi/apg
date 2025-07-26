"""
APG Workflow & Business Process Management - Business Rules Engine

Advanced business rules engine with decision tables, rule sets, and 
intelligent rule evaluation for dynamic workflow control.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict
import operator
import ast

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Rules Engine Core Classes
# =============================================================================

class RuleType(str, Enum):
	"""Types of business rules."""
	DECISION_TABLE = "decision_table"
	CONDITIONAL_RULE = "conditional_rule"
	CALCULATION_RULE = "calculation_rule"
	VALIDATION_RULE = "validation_rule"
	ROUTING_RULE = "routing_rule"
	SLA_RULE = "sla_rule"
	ESCALATION_RULE = "escalation_rule"
	ASSIGNMENT_RULE = "assignment_rule"


class OperatorType(str, Enum):
	"""Rule condition operators."""
	EQUALS = "equals"
	NOT_EQUALS = "not_equals"
	GREATER_THAN = "greater_than"
	GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
	LESS_THAN = "less_than"
	LESS_THAN_OR_EQUAL = "less_than_or_equal"
	CONTAINS = "contains"
	NOT_CONTAINS = "not_contains"
	STARTS_WITH = "starts_with"
	ENDS_WITH = "ends_with"
	MATCHES_REGEX = "matches_regex"
	IN_LIST = "in_list"
	NOT_IN_LIST = "not_in_list"
	IS_NULL = "is_null"
	IS_NOT_NULL = "is_not_null"
	BETWEEN = "between"


class ActionType(str, Enum):
	"""Rule action types."""
	SET_VARIABLE = "set_variable"
	SEND_NOTIFICATION = "send_notification"
	ROUTE_TO_USER = "route_to_user"
	ROUTE_TO_GROUP = "route_to_group"
	SET_PRIORITY = "set_priority"
	SET_DUE_DATE = "set_due_date"
	TRIGGER_ESCALATION = "trigger_escalation"
	CALL_SERVICE = "call_service"
	EXECUTE_SCRIPT = "execute_script"
	CREATE_TASK = "create_task"
	COMPLETE_TASK = "complete_task"


@dataclass
class RuleCondition:
	"""Individual rule condition."""
	condition_id: str = field(default_factory=lambda: f"cond_{uuid.uuid4().hex[:8]}")
	field_name: str = ""
	operator: OperatorType = OperatorType.EQUALS
	value: Any = None
	value_type: str = "string"  # string, number, boolean, date, list
	case_sensitive: bool = True
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"condition_id": self.condition_id,
			"field_name": self.field_name,
			"operator": self.operator.value,
			"value": self.value,
			"value_type": self.value_type,
			"case_sensitive": self.case_sensitive
		}


@dataclass
class RuleAction:
	"""Individual rule action."""
	action_id: str = field(default_factory=lambda: f"action_{uuid.uuid4().hex[:8]}")
	action_type: ActionType = ActionType.SET_VARIABLE
	target: str = ""
	value: Any = None
	parameters: Dict[str, Any] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"action_id": self.action_id,
			"action_type": self.action_type.value,
			"target": self.target,
			"value": self.value,
			"parameters": self.parameters
		}


@dataclass
class BusinessRule:
	"""Individual business rule."""
	rule_id: str = field(default_factory=lambda: f"rule_{uuid.uuid4().hex}")
	tenant_id: str = ""
	rule_name: str = ""
	rule_description: str = ""
	rule_type: RuleType = RuleType.CONDITIONAL_RULE
	priority: int = 100  # Lower number = higher priority
	is_active: bool = True
	conditions: List[RuleCondition] = field(default_factory=list)
	condition_logic: str = "AND"  # AND, OR, custom expression
	actions: List[RuleAction] = field(default_factory=list)
	tags: List[str] = field(default_factory=list)
	created_by: str = ""
	updated_by: str = ""
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	execution_count: int = 0
	last_executed: Optional[datetime] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"rule_id": self.rule_id,
			"tenant_id": self.tenant_id,
			"rule_name": self.rule_name,
			"rule_description": self.rule_description,
			"rule_type": self.rule_type.value,
			"priority": self.priority,
			"is_active": self.is_active,
			"conditions": [c.to_dict() for c in self.conditions],
			"condition_logic": self.condition_logic,
			"actions": [a.to_dict() for a in self.actions],
			"tags": self.tags,
			"created_by": self.created_by,
			"updated_by": self.updated_by,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat(),
			"execution_count": self.execution_count,
			"last_executed": self.last_executed.isoformat() if self.last_executed else None
		}


@dataclass
class DecisionTableRow:
	"""Decision table row."""
	row_id: str = field(default_factory=lambda: f"row_{uuid.uuid4().hex[:8]}")
	input_values: Dict[str, Any] = field(default_factory=dict)
	output_values: Dict[str, Any] = field(default_factory=dict)
	priority: int = 100
	is_active: bool = True
	description: str = ""


@dataclass
class DecisionTable:
	"""Decision table for complex decision logic."""
	table_id: str = field(default_factory=lambda: f"table_{uuid.uuid4().hex}")
	tenant_id: str = ""
	table_name: str = ""
	table_description: str = ""
	input_columns: List[Dict[str, Any]] = field(default_factory=list)  # Column definitions
	output_columns: List[Dict[str, Any]] = field(default_factory=list)  # Column definitions
	rows: List[DecisionTableRow] = field(default_factory=list)
	hit_policy: str = "FIRST"  # FIRST, UNIQUE, PRIORITY, ANY, COLLECT
	tags: List[str] = field(default_factory=list)
	created_by: str = ""
	updated_by: str = ""
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"table_id": self.table_id,
			"tenant_id": self.tenant_id,
			"table_name": self.table_name,
			"table_description": self.table_description,
			"input_columns": self.input_columns,
			"output_columns": self.output_columns,
			"rows": [
				{
					"row_id": row.row_id,
					"input_values": row.input_values,
					"output_values": row.output_values,
					"priority": row.priority,
					"is_active": row.is_active,
					"description": row.description
				}
				for row in self.rows
			],
			"hit_policy": self.hit_policy,
			"tags": self.tags,
			"created_by": self.created_by,
			"updated_by": self.updated_by,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat()
		}


@dataclass
class RuleSet:
	"""Collection of related rules."""
	ruleset_id: str = field(default_factory=lambda: f"ruleset_{uuid.uuid4().hex}")
	tenant_id: str = ""
	ruleset_name: str = ""
	ruleset_description: str = ""
	rule_ids: List[str] = field(default_factory=list)
	decision_table_ids: List[str] = field(default_factory=list)
	execution_order: str = "PRIORITY"  # PRIORITY, SEQUENTIAL, PARALLEL
	tags: List[str] = field(default_factory=list)
	created_by: str = ""
	updated_by: str = ""
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RuleExecutionContext:
	"""Context for rule execution."""
	execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex}")
	tenant_context: APGTenantContext
	process_instance: Optional[WBPMProcessInstance] = None
	task: Optional[WBPMTask] = None
	variables: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RuleExecutionResult:
	"""Result of rule execution."""
	execution_id: str = ""
	rule_id: str = ""
	matched: bool = False
	actions_executed: List[str] = field(default_factory=list)
	variables_changed: Dict[str, Any] = field(default_factory=dict)
	errors: List[str] = field(default_factory=list)
	execution_time_ms: float = 0.0
	timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Condition Evaluator
# =============================================================================

class ConditionEvaluator:
	"""Evaluate rule conditions against data."""
	
	def __init__(self):
		self.operators = {
			OperatorType.EQUALS: self._equals,
			OperatorType.NOT_EQUALS: self._not_equals,
			OperatorType.GREATER_THAN: self._greater_than,
			OperatorType.GREATER_THAN_OR_EQUAL: self._greater_than_or_equal,
			OperatorType.LESS_THAN: self._less_than,
			OperatorType.LESS_THAN_OR_EQUAL: self._less_than_or_equal,
			OperatorType.CONTAINS: self._contains,
			OperatorType.NOT_CONTAINS: self._not_contains,
			OperatorType.STARTS_WITH: self._starts_with,
			OperatorType.ENDS_WITH: self._ends_with,
			OperatorType.MATCHES_REGEX: self._matches_regex,
			OperatorType.IN_LIST: self._in_list,
			OperatorType.NOT_IN_LIST: self._not_in_list,
			OperatorType.IS_NULL: self._is_null,
			OperatorType.IS_NOT_NULL: self._is_not_null,
			OperatorType.BETWEEN: self._between
		}
	
	async def evaluate_condition(
		self,
		condition: RuleCondition,
		context: RuleExecutionContext
	) -> bool:
		"""Evaluate single condition."""
		try:
			# Get field value from context
			field_value = self._get_field_value(condition.field_name, context)
			
			# Get operator function
			operator_func = self.operators.get(condition.operator)
			if not operator_func:
				logger.error(f"Unknown operator: {condition.operator}")
				return False
			
			# Evaluate condition
			result = operator_func(field_value, condition.value, condition)
			
			logger.debug(f"Condition {condition.condition_id}: {field_value} {condition.operator.value} {condition.value} = {result}")
			
			return result
			
		except Exception as e:
			logger.error(f"Error evaluating condition {condition.condition_id}: {e}")
			return False
	
	async def evaluate_conditions(
		self,
		conditions: List[RuleCondition],
		logic: str,
		context: RuleExecutionContext
	) -> bool:
		"""Evaluate multiple conditions with logic."""
		if not conditions:
			return True
		
		try:
			# Evaluate individual conditions
			results = []
			for condition in conditions:
				result = await self.evaluate_condition(condition, context)
				results.append(result)
			
			# Apply logic
			if logic.upper() == "AND":
				return all(results)
			elif logic.upper() == "OR":
				return any(results)
			else:
				# Custom expression logic
				return await self._evaluate_custom_logic(logic, results, conditions)
			
		except Exception as e:
			logger.error(f"Error evaluating conditions: {e}")
			return False
	
	def _get_field_value(self, field_name: str, context: RuleExecutionContext) -> Any:
		"""Get field value from execution context."""
		# Try variables first
		if field_name in context.variables:
			return context.variables[field_name]
		
		# Try process instance fields
		if context.process_instance:
			if hasattr(context.process_instance, field_name):
				return getattr(context.process_instance, field_name)
			
			# Try process variables
			if field_name in context.process_instance.process_variables:
				return context.process_instance.process_variables[field_name]
		
		# Try task fields
		if context.task:
			if hasattr(context.task, field_name):
				return getattr(context.task, field_name)
			
			# Try task variables
			if field_name in context.task.task_variables:
				return context.task.task_variables[field_name]
		
		# Try metadata
		if field_name in context.metadata:
			return context.metadata[field_name]
		
		# Return None if not found
		return None
	
	def _equals(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Equals operator."""
		if field_value is None and condition_value is None:
			return True
		if field_value is None or condition_value is None:
			return False
		
		if isinstance(field_value, str) and isinstance(condition_value, str):
			if not condition.case_sensitive:
				return field_value.lower() == condition_value.lower()
		
		return field_value == condition_value
	
	def _not_equals(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Not equals operator."""
		return not self._equals(field_value, condition_value, condition)
	
	def _greater_than(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Greater than operator."""
		try:
			return float(field_value) > float(condition_value)
		except (TypeError, ValueError):
			return False
	
	def _greater_than_or_equal(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Greater than or equal operator."""
		try:
			return float(field_value) >= float(condition_value)
		except (TypeError, ValueError):
			return False
	
	def _less_than(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Less than operator."""
		try:
			return float(field_value) < float(condition_value)
		except (TypeError, ValueError):
			return False
	
	def _less_than_or_equal(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Less than or equal operator."""
		try:
			return float(field_value) <= float(condition_value)
		except (TypeError, ValueError):
			return False
	
	def _contains(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Contains operator."""
		if field_value is None or condition_value is None:
			return False
		
		field_str = str(field_value)
		condition_str = str(condition_value)
		
		if not condition.case_sensitive:
			field_str = field_str.lower()
			condition_str = condition_str.lower()
		
		return condition_str in field_str
	
	def _not_contains(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Not contains operator."""
		return not self._contains(field_value, condition_value, condition)
	
	def _starts_with(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Starts with operator."""
		if field_value is None or condition_value is None:
			return False
		
		field_str = str(field_value)
		condition_str = str(condition_value)
		
		if not condition.case_sensitive:
			field_str = field_str.lower()
			condition_str = condition_str.lower()
		
		return field_str.startswith(condition_str)
	
	def _ends_with(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Ends with operator."""
		if field_value is None or condition_value is None:
			return False
		
		field_str = str(field_value)
		condition_str = str(condition_value)
		
		if not condition.case_sensitive:
			field_str = field_str.lower()
			condition_str = condition_str.lower()
		
		return field_str.endswith(condition_str)
	
	def _matches_regex(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Regex match operator."""
		if field_value is None or condition_value is None:
			return False
		
		try:
			flags = 0 if condition.case_sensitive else re.IGNORECASE
			pattern = re.compile(str(condition_value), flags)
			return bool(pattern.search(str(field_value)))
		except re.error:
			return False
	
	def _in_list(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""In list operator."""
		if field_value is None or condition_value is None:
			return False
		
		if not isinstance(condition_value, (list, tuple)):
			condition_value = [condition_value]
		
		if not condition.case_sensitive and isinstance(field_value, str):
			field_value = field_value.lower()
			condition_value = [str(v).lower() if isinstance(v, str) else v for v in condition_value]
		
		return field_value in condition_value
	
	def _not_in_list(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Not in list operator."""
		return not self._in_list(field_value, condition_value, condition)
	
	def _is_null(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Is null operator."""
		return field_value is None or (isinstance(field_value, str) and field_value.strip() == "")
	
	def _is_not_null(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Is not null operator."""
		return not self._is_null(field_value, condition_value, condition)
	
	def _between(self, field_value: Any, condition_value: Any, condition: RuleCondition) -> bool:
		"""Between operator."""
		if field_value is None or condition_value is None:
			return False
		
		if not isinstance(condition_value, (list, tuple)) or len(condition_value) != 2:
			return False
		
		try:
			value = float(field_value)
			min_val = float(condition_value[0])
			max_val = float(condition_value[1])
			return min_val <= value <= max_val
		except (TypeError, ValueError):
			return False
	
	async def _evaluate_custom_logic(
		self,
		logic: str,
		results: List[bool],
		conditions: List[RuleCondition]
	) -> bool:
		"""Evaluate custom logic expression."""
		try:
			# Replace condition IDs with results
			expression = logic
			for i, condition in enumerate(conditions):
				expression = expression.replace(condition.condition_id, str(results[i]))
			
			# Evaluate the expression safely
			# This is a simplified implementation - production would use a proper expression parser
			expression = expression.replace("AND", " and ").replace("OR", " or ").replace("NOT", " not ")
			
			# Only allow specific operators for security
			allowed_chars = set("() andornotTrueFalse ")
			if all(c in allowed_chars or c.isspace() for c in expression):
				return bool(eval(expression))
			
			return False
			
		except Exception as e:
			logger.error(f"Error evaluating custom logic '{logic}': {e}")
			return False


# =============================================================================
# Action Executor
# =============================================================================

class ActionExecutor:
	"""Execute rule actions."""
	
	def __init__(self):
		self.action_handlers = {
			ActionType.SET_VARIABLE: self._set_variable,
			ActionType.SEND_NOTIFICATION: self._send_notification,
			ActionType.ROUTE_TO_USER: self._route_to_user,
			ActionType.ROUTE_TO_GROUP: self._route_to_group,
			ActionType.SET_PRIORITY: self._set_priority,
			ActionType.SET_DUE_DATE: self._set_due_date,
			ActionType.TRIGGER_ESCALATION: self._trigger_escalation,
			ActionType.CALL_SERVICE: self._call_service,
			ActionType.EXECUTE_SCRIPT: self._execute_script,
			ActionType.CREATE_TASK: self._create_task,
			ActionType.COMPLETE_TASK: self._complete_task
		}
	
	async def execute_actions(
		self,
		actions: List[RuleAction],
		context: RuleExecutionContext
	) -> Tuple[List[str], Dict[str, Any]]:
		"""Execute list of actions."""
		executed_actions = []
		variables_changed = {}
		
		for action in actions:
			try:
				handler = self.action_handlers.get(action.action_type)
				if handler:
					result = await handler(action, context)
					if result:
						executed_actions.append(action.action_id)
						if isinstance(result, dict) and "variables" in result:
							variables_changed.update(result["variables"])
				else:
					logger.warning(f"No handler for action type: {action.action_type}")
				
			except Exception as e:
				logger.error(f"Error executing action {action.action_id}: {e}")
		
		return executed_actions, variables_changed
	
	async def _set_variable(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
		"""Set variable action."""
		variable_name = action.target
		variable_value = action.value
		
		# Support dynamic values
		if isinstance(variable_value, str) and variable_value.startswith("${") and variable_value.endswith("}"):
			# Extract field reference
			field_name = variable_value[2:-1]
			variable_value = self._get_context_value(field_name, context)
		
		context.variables[variable_name] = variable_value
		
		logger.debug(f"Set variable {variable_name} = {variable_value}")
		
		return {"variables": {variable_name: variable_value}}
	
	async def _send_notification(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Send notification action."""
		recipient = action.target
		message = action.value
		notification_type = action.parameters.get("type", "info")
		
		# In production, integrate with APG notification service
		logger.info(f"Notification sent to {recipient}: {message} (type: {notification_type})")
		
		return True
	
	async def _route_to_user(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Route to user action."""
		user_id = action.target
		
		if context.task:
			context.task.assignee = user_id
			logger.info(f"Task {context.task.id} routed to user {user_id}")
		
		return True
	
	async def _route_to_group(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Route to group action."""
		group_id = action.target
		
		if context.task:
			if group_id not in context.task.candidate_groups:
				context.task.candidate_groups.append(group_id)
			logger.info(f"Task {context.task.id} routed to group {group_id}")
		
		return True
	
	async def _set_priority(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Set priority action."""
		priority = action.value
		
		if context.task:
			context.task.priority = TaskPriority(priority)
			logger.info(f"Task {context.task.id} priority set to {priority}")
		
		if context.process_instance:
			context.process_instance.priority = TaskPriority(priority)
			logger.info(f"Process {context.process_instance.id} priority set to {priority}")
		
		return True
	
	async def _set_due_date(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Set due date action."""
		due_date_spec = action.value
		
		# Parse due date specification
		if isinstance(due_date_spec, str):
			if due_date_spec.startswith("+"):
				# Relative date (e.g., "+3d", "+2h")
				amount = int(due_date_spec[1:-1])
				unit = due_date_spec[-1]
				
				if unit == "d":
					due_date = datetime.utcnow() + timedelta(days=amount)
				elif unit == "h":
					due_date = datetime.utcnow() + timedelta(hours=amount)
				elif unit == "m":
					due_date = datetime.utcnow() + timedelta(minutes=amount)
				else:
					due_date = datetime.utcnow() + timedelta(days=1)
			else:
				# Absolute date
				try:
					due_date = datetime.fromisoformat(due_date_spec)
				except ValueError:
					due_date = datetime.utcnow() + timedelta(days=1)
		else:
			due_date = due_date_spec
		
		if context.task:
			context.task.due_date = due_date
			logger.info(f"Task {context.task.id} due date set to {due_date}")
		
		return True
	
	async def _trigger_escalation(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Trigger escalation action."""
		escalation_type = action.parameters.get("type", "standard")
		escalation_target = action.target
		
		# In production, integrate with escalation engine
		logger.info(f"Escalation triggered: {escalation_type} to {escalation_target}")
		
		return True
	
	async def _call_service(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Call external service action."""
		service_url = action.target
		service_params = action.parameters
		
		# In production, make actual HTTP call
		logger.info(f"Service call to {service_url} with params: {service_params}")
		
		return True
	
	async def _execute_script(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Execute script action."""
		script_code = action.value
		
		# In production, use secure script execution environment
		logger.info(f"Script execution: {script_code[:100]}...")
		
		return True
	
	async def _create_task(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Create task action."""
		task_name = action.value
		task_params = action.parameters
		
		# In production, create actual task
		logger.info(f"Task created: {task_name} with params: {task_params}")
		
		return True
	
	async def _complete_task(self, action: RuleAction, context: RuleExecutionContext) -> bool:
		"""Complete task action."""
		if context.task:
			# In production, actually complete the task
			logger.info(f"Task {context.task.id} completed via rule action")
		
		return True
	
	def _get_context_value(self, field_name: str, context: RuleExecutionContext) -> Any:
		"""Get value from context (same as in ConditionEvaluator)."""
		# Try variables first
		if field_name in context.variables:
			return context.variables[field_name]
		
		# Try process instance fields
		if context.process_instance:
			if hasattr(context.process_instance, field_name):
				return getattr(context.process_instance, field_name)
			
			if field_name in context.process_instance.process_variables:
				return context.process_instance.process_variables[field_name]
		
		# Try task fields
		if context.task:
			if hasattr(context.task, field_name):
				return getattr(context.task, field_name)
			
			if field_name in context.task.task_variables:
				return context.task.task_variables[field_name]
		
		# Try metadata
		if field_name in context.metadata:
			return context.metadata[field_name]
		
		return None


# =============================================================================
# Decision Table Engine
# =============================================================================

class DecisionTableEngine:
	"""Decision table evaluation engine."""
	
	def __init__(self, condition_evaluator: ConditionEvaluator):
		self.condition_evaluator = condition_evaluator
	
	async def evaluate_decision_table(
		self,
		table: DecisionTable,
		context: RuleExecutionContext
	) -> Dict[str, Any]:
		"""Evaluate decision table and return results."""
		matching_rows = []
		
		# Evaluate each row
		for row in table.rows:
			if not row.is_active:
				continue
			
			if await self._evaluate_row(row, table.input_columns, context):
				matching_rows.append(row)
		
		# Apply hit policy
		if table.hit_policy == "FIRST":
			return matching_rows[0].output_values if matching_rows else {}
		elif table.hit_policy == "UNIQUE":
			if len(matching_rows) > 1:
				logger.warning(f"Decision table {table.table_id} has multiple matches for UNIQUE hit policy")
			return matching_rows[0].output_values if matching_rows else {}
		elif table.hit_policy == "PRIORITY":
			if matching_rows:
				# Sort by priority (lower number = higher priority)
				matching_rows.sort(key=lambda r: r.priority)
				return matching_rows[0].output_values
			return {}
		elif table.hit_policy == "ANY":
			# Return any matching row (typically all should have same output)
			return matching_rows[0].output_values if matching_rows else {}
		elif table.hit_policy == "COLLECT":
			# Collect all outputs
			collected_outputs = defaultdict(list)
			for row in matching_rows:
				for key, value in row.output_values.items():
					collected_outputs[key].append(value)
			return dict(collected_outputs)
		
		return {}
	
	async def _evaluate_row(
		self,
		row: DecisionTableRow,
		input_columns: List[Dict[str, Any]],
		context: RuleExecutionContext
	) -> bool:
		"""Evaluate if a row matches the input context."""
		for column in input_columns:
			column_name = column["name"]
			column_type = column.get("type", "string")
			
			# Get expected value from row
			expected_value = row.input_values.get(column_name)
			if expected_value is None:
				continue  # Skip if no value specified
			
			# Get actual value from context
			actual_value = self._get_context_value(column_name, context)
			
			# Compare values
			if not await self._compare_values(actual_value, expected_value, column_type):
				return False
		
		return True
	
	async def _compare_values(self, actual: Any, expected: Any, value_type: str) -> bool:
		"""Compare actual and expected values."""
		# Handle special expressions
		if isinstance(expected, str):
			if expected == "*":  # Wildcard - matches anything
				return True
			elif expected == "-":  # No value - matches null
				return actual is None
			elif expected.startswith(">="):
				try:
					return float(actual) >= float(expected[2:])
				except (TypeError, ValueError):
					return False
			elif expected.startswith("<="):
				try:
					return float(actual) <= float(expected[2:])
				except (TypeError, ValueError):
					return False
			elif expected.startswith(">"):
				try:
					return float(actual) > float(expected[1:])
				except (TypeError, ValueError):
					return False
			elif expected.startswith("<"):
				try:
					return float(actual) < float(expected[1:])
				except (TypeError, ValueError):
					return False
			elif "," in expected:  # List of values
				expected_list = [v.strip() for v in expected.split(",")]
				return str(actual) in expected_list
		
		# Direct comparison
		if value_type == "number":
			try:
				return float(actual) == float(expected)
			except (TypeError, ValueError):
				return False
		elif value_type == "boolean":
			return bool(actual) == bool(expected)
		else:  # string or other
			return str(actual) == str(expected)
	
	def _get_context_value(self, field_name: str, context: RuleExecutionContext) -> Any:
		"""Get value from context."""
		# Try variables first
		if field_name in context.variables:
			return context.variables[field_name]
		
		# Try process instance fields
		if context.process_instance:
			if hasattr(context.process_instance, field_name):
				return getattr(context.process_instance, field_name)
			
			if field_name in context.process_instance.process_variables:
				return context.process_instance.process_variables[field_name]
		
		# Try task fields
		if context.task:
			if hasattr(context.task, field_name):
				return getattr(context.task, field_name)
			
			if field_name in context.task.task_variables:
				return context.task.task_variables[field_name]
		
		# Try metadata
		if field_name in context.metadata:
			return context.metadata[field_name]
		
		return None


# =============================================================================
# Business Rules Engine
# =============================================================================

class BusinessRulesEngine:
	"""Main business rules engine."""
	
	def __init__(self):
		self.condition_evaluator = ConditionEvaluator()
		self.action_executor = ActionExecutor()
		self.decision_table_engine = DecisionTableEngine(self.condition_evaluator)
		
		# Storage (in production would be database)
		self.rules: Dict[str, BusinessRule] = {}
		self.decision_tables: Dict[str, DecisionTable] = {}
		self.rule_sets: Dict[str, RuleSet] = {}
		self.execution_history: List[RuleExecutionResult] = []
	
	async def execute_rules(
		self,
		context: RuleExecutionContext,
		ruleset_id: Optional[str] = None,
		rule_types: Optional[List[RuleType]] = None
	) -> WBPMServiceResponse:
		"""Execute business rules against context."""
		try:
			start_time = datetime.utcnow()
			execution_results = []
			
			# Get rules to execute
			rules_to_execute = await self._get_rules_to_execute(
				context, ruleset_id, rule_types
			)
			
			# Execute rules
			for rule in rules_to_execute:
				result = await self._execute_single_rule(rule, context)
				execution_results.append(result)
			
			# Execute decision tables if any
			decision_table_results = await self._execute_decision_tables(context, ruleset_id)
			
			# Calculate total execution time
			execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			
			# Store execution history
			self.execution_history.extend(execution_results)
			
			# Keep history manageable
			if len(self.execution_history) > 10000:
				self.execution_history = self.execution_history[-5000:]
			
			# Summary
			matched_rules = [r for r in execution_results if r.matched]
			total_actions = sum(len(r.actions_executed) for r in execution_results)
			
			logger.info(f"Rules execution completed: {len(matched_rules)}/{len(rules_to_execute)} rules matched, {total_actions} actions executed")
			
			return WBPMServiceResponse(
				success=True,
				message="Rules executed successfully",
				data={
					"execution_id": context.execution_id,
					"rules_evaluated": len(rules_to_execute),
					"rules_matched": len(matched_rules),
					"actions_executed": total_actions,
					"execution_time_ms": execution_time,
					"rule_results": [
						{
							"rule_id": r.rule_id,
							"matched": r.matched,
							"actions_executed": len(r.actions_executed),
							"execution_time_ms": r.execution_time_ms
						}
						for r in execution_results
					],
					"decision_table_results": decision_table_results,
					"variables_changed": {
						key: value for result in execution_results
						for key, value in result.variables_changed.items()
					}
				}
			)
			
		except Exception as e:
			logger.error(f"Error executing rules: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to execute rules: {e}",
				errors=[str(e)]
			)
	
	async def _execute_single_rule(
		self,
		rule: BusinessRule,
		context: RuleExecutionContext
	) -> RuleExecutionResult:
		"""Execute a single business rule."""
		start_time = datetime.utcnow()
		
		result = RuleExecutionResult(
			execution_id=context.execution_id,
			rule_id=rule.rule_id
		)
		
		try:
			# Evaluate conditions
			conditions_met = await self.condition_evaluator.evaluate_conditions(
				rule.conditions, rule.condition_logic, context
			)
			
			result.matched = conditions_met
			
			if conditions_met:
				# Execute actions
				executed_actions, variables_changed = await self.action_executor.execute_actions(
					rule.actions, context
				)
				
				result.actions_executed = executed_actions
				result.variables_changed = variables_changed
				
				# Update rule statistics
				rule.execution_count += 1
				rule.last_executed = datetime.utcnow()
				
				logger.debug(f"Rule {rule.rule_id} matched and executed {len(executed_actions)} actions")
			else:
				logger.debug(f"Rule {rule.rule_id} conditions not met")
			
		except Exception as e:
			error_msg = f"Error executing rule {rule.rule_id}: {e}"
			logger.error(error_msg)
			result.errors.append(error_msg)
		
		# Calculate execution time
		execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
		result.execution_time_ms = execution_time
		result.timestamp = datetime.utcnow()
		
		return result
	
	async def _get_rules_to_execute(
		self,
		context: RuleExecutionContext,
		ruleset_id: Optional[str] = None,
		rule_types: Optional[List[RuleType]] = None
	) -> List[BusinessRule]:
		"""Get rules that should be executed."""
		if ruleset_id:
			# Execute rules from specific ruleset
			ruleset = self.rule_sets.get(ruleset_id)
			if not ruleset:
				return []
			
			rules = [
				self.rules[rule_id] for rule_id in ruleset.rule_ids
				if rule_id in self.rules
			]
		else:
			# Execute all applicable rules
			rules = [
				rule for rule in self.rules.values()
				if rule.tenant_id == context.tenant_context.tenant_id and rule.is_active
			]
		
		# Filter by rule types if specified
		if rule_types:
			rules = [rule for rule in rules if rule.rule_type in rule_types]
		
		# Sort by priority (lower number = higher priority)
		rules.sort(key=lambda r: r.priority)
		
		return rules
	
	async def _execute_decision_tables(
		self,
		context: RuleExecutionContext,
		ruleset_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""Execute decision tables."""
		results = {}
		
		if ruleset_id:
			ruleset = self.rule_sets.get(ruleset_id)
			if ruleset:
				table_ids = ruleset.decision_table_ids
			else:
				table_ids = []
		else:
			table_ids = [
				table_id for table_id, table in self.decision_tables.items()
				if table.tenant_id == context.tenant_context.tenant_id
			]
		
		for table_id in table_ids:
			table = self.decision_tables.get(table_id)
			if table:
				table_result = await self.decision_table_engine.evaluate_decision_table(
					table, context
				)
				results[table_id] = table_result
				
				# Apply decision table results to context variables
				context.variables.update(table_result)
		
		return results
	
	# Rule Management Methods
	
	async def create_rule(
		self,
		rule_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create new business rule."""
		try:
			# Create conditions
			conditions = []
			for cond_data in rule_data.get("conditions", []):
				condition = RuleCondition(
					field_name=cond_data["field_name"],
					operator=OperatorType(cond_data["operator"]),
					value=cond_data["value"],
					value_type=cond_data.get("value_type", "string"),
					case_sensitive=cond_data.get("case_sensitive", True)
				)
				conditions.append(condition)
			
			# Create actions
			actions = []
			for action_data in rule_data.get("actions", []):
				action = RuleAction(
					action_type=ActionType(action_data["action_type"]),
					target=action_data["target"],
					value=action_data["value"],
					parameters=action_data.get("parameters", {})
				)
				actions.append(action)
			
			# Create rule
			rule = BusinessRule(
				tenant_id=context.tenant_id,
				rule_name=rule_data["rule_name"],
				rule_description=rule_data.get("rule_description", ""),
				rule_type=RuleType(rule_data.get("rule_type", RuleType.CONDITIONAL_RULE)),
				priority=rule_data.get("priority", 100),
				conditions=conditions,
				condition_logic=rule_data.get("condition_logic", "AND"),
				actions=actions,
				tags=rule_data.get("tags", []),
				created_by=context.user_id,
				updated_by=context.user_id
			)
			
			# Store rule
			self.rules[rule.rule_id] = rule
			
			logger.info(f"Business rule created: {rule.rule_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Business rule created successfully",
				data={
					"rule_id": rule.rule_id,
					"rule_name": rule.rule_name,
					"rule_type": rule.rule_type.value
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating business rule: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create business rule: {e}",
				errors=[str(e)]
			)
	
	async def create_decision_table(
		self,
		table_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create new decision table."""
		try:
			# Create rows
			rows = []
			for row_data in table_data.get("rows", []):
				row = DecisionTableRow(
					input_values=row_data.get("input_values", {}),
					output_values=row_data.get("output_values", {}),
					priority=row_data.get("priority", 100),
					is_active=row_data.get("is_active", True),
					description=row_data.get("description", "")
				)
				rows.append(row)
			
			# Create decision table
			table = DecisionTable(
				tenant_id=context.tenant_id,
				table_name=table_data["table_name"],
				table_description=table_data.get("table_description", ""),
				input_columns=table_data.get("input_columns", []),
				output_columns=table_data.get("output_columns", []),
				rows=rows,
				hit_policy=table_data.get("hit_policy", "FIRST"),
				tags=table_data.get("tags", []),
				created_by=context.user_id,
				updated_by=context.user_id
			)
			
			# Store table
			self.decision_tables[table.table_id] = table
			
			logger.info(f"Decision table created: {table.table_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Decision table created successfully",
				data={
					"table_id": table.table_id,
					"table_name": table.table_name,
					"rows_count": len(table.rows)
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating decision table: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create decision table: {e}",
				errors=[str(e)]
			)


# =============================================================================
# Service Factory
# =============================================================================

def create_business_rules_engine() -> BusinessRulesEngine:
	"""Create and configure business rules engine."""
	engine = BusinessRulesEngine()
	logger.info("Business rules engine created and configured")
	return engine


# Export main classes
__all__ = [
	'BusinessRulesEngine',
	'ConditionEvaluator',
	'ActionExecutor',
	'DecisionTableEngine',
	'BusinessRule',
	'DecisionTable',
	'RuleSet',
	'RuleCondition',
	'RuleAction',
	'RuleExecutionContext',
	'RuleExecutionResult',
	'RuleType',
	'OperatorType',
	'ActionType',
	'create_business_rules_engine'
]