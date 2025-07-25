"""
Attribute-Based Access Control (ABAC) Models

Comprehensive ABAC implementation with policy-based authorization,
dynamic attribute evaluation, and context-aware access control.
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, Float, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
from flask_appbuilder import Model
from enum import Enum

from ..common.base import BaseCapabilityModel, AuditMixin, BaseMixin

# UUID7-like string generator for consistent ID generation
def uuid7str() -> str:
	"""Generate UUID7-like string for consistent ID generation"""
	return str(uuid.uuid4())


class AttributeType(str, Enum):
	"""Attribute data types for ABAC"""
	STRING = "string"
	INTEGER = "integer"
	FLOAT = "float"
	BOOLEAN = "boolean"
	DATE = "date"
	DATETIME = "datetime"
	LIST = "list"
	OBJECT = "object"


class PolicyEffect(str, Enum):
	"""Policy decision effects"""
	PERMIT = "permit"
	DENY = "deny"


class PolicyAlgorithm(str, Enum):
	"""Policy combination algorithms"""
	PERMIT_OVERRIDES = "permit_overrides"
	DENY_OVERRIDES = "deny_overrides"
	FIRST_APPLICABLE = "first_applicable"
	ONLY_ONE_APPLICABLE = "only_one_applicable"
	PERMIT_UNLESS_DENY = "permit_unless_deny"
	DENY_UNLESS_PERMIT = "deny_unless_permit"


class ARAttribute(Model, AuditMixin, BaseMixin):
	"""
	ABAC Attribute definitions for subjects, resources, actions, and environment.
	
	Defines the attributes that can be used in ABAC policies with
	comprehensive metadata and validation rules.
	"""
	
	__tablename__ = 'ar_abac_attribute'
	
	# Identity
	attribute_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Attribute Definition
	category = Column(String(50), nullable=False, index=True)  # subject, resource, action, environment
	name = Column(String(100), nullable=False, index=True)
	display_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Data Type and Constraints
	data_type = Column(String(20), nullable=False)  # AttributeType enum values
	is_required = Column(Boolean, default=False)
	is_multi_valued = Column(Boolean, default=False)
	default_value = Column(JSON, nullable=True)
	
	# Validation Rules
	validation_rules = Column(JSON, default=dict)  # Format, range, pattern, etc.
	allowed_values = Column(JSON, default=list)  # For enumerated values
	
	# Attribute Source
	source_type = Column(String(50), nullable=False)  # static, dynamic, computed, external
	source_config = Column(JSON, default=dict)  # Configuration for dynamic/external sources
	cache_duration = Column(Integer, default=300)  # Cache duration in seconds
	
	# Metadata
	is_system_attribute = Column(Boolean, default=False)
	is_pii = Column(Boolean, default=False)  # Contains personally identifiable information
	sensitivity_level = Column(String(20), default='public')  # public, internal, confidential, restricted
	
	# Usage Analytics
	usage_count = Column(Integer, default=0)
	last_used = Column(DateTime, nullable=True)
	
	# Relationships
	policy_conditions = relationship("ARPolicyCondition", back_populates="attribute", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'category', 'name', name='uq_ar_attribute_category_name'),
		Index('ix_ar_attribute_tenant_category', 'tenant_id', 'category'),
		Index('ix_ar_attribute_source_type', 'source_type'),
		Index('ix_ar_attribute_data_type', 'data_type'),
	)
	
	@validates('category')
	def validate_category(self, key, category):
		"""Validate attribute category"""
		allowed_categories = ['subject', 'resource', 'action', 'environment']
		if category not in allowed_categories:
			raise ValueError(f"Category must be one of: {allowed_categories}")
		return category
	
	@validates('data_type')
	def validate_data_type(self, key, data_type):
		"""Validate data type"""
		if data_type not in [t.value for t in AttributeType]:
			raise ValueError(f"Invalid data type: {data_type}")
		return data_type
	
	@validates('source_type')
	def validate_source_type(self, key, source_type):
		"""Validate source type"""
		allowed_sources = ['static', 'dynamic', 'computed', 'external']
		if source_type not in allowed_sources:
			raise ValueError(f"Source type must be one of: {allowed_sources}")
		return source_type
	
	def validate_value(self, value: Any) -> Tuple[bool, str]:
		"""Validate attribute value against constraints"""
		if value is None:
			if self.is_required:
				return False, "Attribute is required"
			return True, ""
		
		# Type validation
		if self.data_type == AttributeType.STRING.value:
			if not isinstance(value, str):
				return False, f"Expected string, got {type(value).__name__}"
		elif self.data_type == AttributeType.INTEGER.value:
			if not isinstance(value, int):
				return False, f"Expected integer, got {type(value).__name__}"
		elif self.data_type == AttributeType.FLOAT.value:
			if not isinstance(value, (int, float)):
				return False, f"Expected float, got {type(value).__name__}"
		elif self.data_type == AttributeType.BOOLEAN.value:
			if not isinstance(value, bool):
				return False, f"Expected boolean, got {type(value).__name__}"
		elif self.data_type == AttributeType.LIST.value:
			if not isinstance(value, list):
				return False, f"Expected list, got {type(value).__name__}"
		elif self.data_type == AttributeType.OBJECT.value:
			if not isinstance(value, dict):
				return False, f"Expected object, got {type(value).__name__}"
		
		# Multi-value validation
		if self.is_multi_valued and not isinstance(value, list):
			return False, "Multi-valued attribute must be a list"
		
		# Allowed values validation
		if self.allowed_values:
			if self.is_multi_valued:
				for v in value:
					if v not in self.allowed_values:
						return False, f"Value '{v}' not in allowed values"
			else:
				if value not in self.allowed_values:
					return False, f"Value '{value}' not in allowed values"
		
		# Validation rules
		if self.validation_rules:
			for rule_name, rule_config in self.validation_rules.items():
				if rule_name == 'min_length' and isinstance(value, str):
					if len(value) < rule_config:
						return False, f"String too short (min: {rule_config})"
				elif rule_name == 'max_length' and isinstance(value, str):
					if len(value) > rule_config:
						return False, f"String too long (max: {rule_config})"
				elif rule_name == 'min_value' and isinstance(value, (int, float)):
					if value < rule_config:
						return False, f"Value too small (min: {rule_config})"
				elif rule_name == 'max_value' and isinstance(value, (int, float)):
					if value > rule_config:
						return False, f"Value too large (max: {rule_config})"
				elif rule_name == 'pattern' and isinstance(value, str):
					import re
					if not re.match(rule_config, value):
						return False, f"Value doesn't match pattern: {rule_config}"
		
		return True, ""
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert attribute to dictionary representation"""
		return {
			'attribute_id': self.attribute_id,
			'tenant_id': self.tenant_id,
			'category': self.category,
			'name': self.name,
			'display_name': self.display_name,
			'description': self.description,
			'data_type': self.data_type,
			'is_required': self.is_required,
			'is_multi_valued': self.is_multi_valued,
			'default_value': self.default_value,
			'validation_rules': self.validation_rules,
			'allowed_values': self.allowed_values,
			'source_type': self.source_type,
			'source_config': self.source_config,
			'cache_duration': self.cache_duration,
			'is_system_attribute': self.is_system_attribute,
			'is_pii': self.is_pii,
			'sensitivity_level': self.sensitivity_level,
			'usage_count': self.usage_count,
			'last_used': self.last_used.isoformat() if self.last_used else None,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARPolicy(Model, AuditMixin, BaseMixin):
	"""
	ABAC Policy definitions with comprehensive rule sets.
	
	Defines access control policies using attribute-based conditions
	with support for complex rule combinations and algorithms.
	"""
	
	__tablename__ = 'ar_abac_policy'
	
	# Identity
	policy_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Policy Definition
	name = Column(String(200), nullable=False)
	display_name = Column(String(300), nullable=True)
	description = Column(Text, nullable=True)
	version = Column(String(20), default='1.0.0')
	
	# Policy Configuration
	effect = Column(String(10), nullable=False, default=PolicyEffect.PERMIT.value)  # permit, deny
	algorithm = Column(String(50), default=PolicyAlgorithm.DENY_OVERRIDES.value)
	priority = Column(Integer, default=100)  # Higher = higher priority
	
	# Target Definition (what this policy applies to)
	target_subjects = Column(JSON, default=list)  # Subject attribute conditions
	target_resources = Column(JSON, default=list)  # Resource attribute conditions
	target_actions = Column(JSON, default=list)  # Action attribute conditions
	target_environments = Column(JSON, default=list)  # Environment attribute conditions
	
	# Policy Status
	is_active = Column(Boolean, default=True)
	is_system_policy = Column(Boolean, default=False)
	
	# Time Constraints
	effective_from = Column(DateTime, nullable=True)
	effective_until = Column(DateTime, nullable=True)
	
	# Policy Metadata
	tags = Column(JSON, default=list)
	compliance_frameworks = Column(JSON, default=list)  # GDPR, HIPAA, SOX, etc.
	risk_level = Column(String(20), default='medium')  # low, medium, high, critical
	
	# Performance and Analytics
	evaluation_count = Column(Integer, default=0)
	permit_count = Column(Integer, default=0)
	deny_count = Column(Integer, default=0)
	last_evaluated = Column(DateTime, nullable=True)
	average_evaluation_time_ms = Column(Float, default=0.0)
	
	# Relationships
	conditions = relationship("ARPolicyCondition", back_populates="policy", cascade="all, delete-orphan")
	rules = relationship("ARPolicyRule", back_populates="policy", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'name', 'version', name='uq_ar_policy_name_version'),
		Index('ix_ar_policy_tenant_active', 'tenant_id', 'is_active'),
		Index('ix_ar_policy_priority', 'priority'),
		Index('ix_ar_policy_effective', 'effective_from', 'effective_until'),
	)
	
	@validates('effect')
	def validate_effect(self, key, effect):
		"""Validate policy effect"""
		if effect not in [e.value for e in PolicyEffect]:
			raise ValueError(f"Invalid policy effect: {effect}")
		return effect
	
	@validates('algorithm')
	def validate_algorithm(self, key, algorithm):
		"""Validate policy algorithm"""
		if algorithm not in [a.value for a in PolicyAlgorithm]:
			raise ValueError(f"Invalid policy algorithm: {algorithm}")
		return algorithm
	
	def is_effective(self) -> bool:
		"""Check if policy is currently effective"""
		if not self.is_active:
			return False
		
		now = datetime.utcnow()
		
		if self.effective_from and now < self.effective_from:
			return False
		
		if self.effective_until and now > self.effective_until:
			return False
		
		return True
	
	def matches_target(self, request_context: Dict[str, Any]) -> bool:
		"""Check if policy target matches the request context"""
		
		# Check subject targets
		if self.target_subjects:
			subject_attrs = request_context.get('subject', {})
			if not self._matches_target_conditions(self.target_subjects, subject_attrs):
				return False
		
		# Check resource targets
		if self.target_resources:
			resource_attrs = request_context.get('resource', {})
			if not self._matches_target_conditions(self.target_resources, resource_attrs):
				return False
		
		# Check action targets
		if self.target_actions:
			action_attrs = request_context.get('action', {})
			if not self._matches_target_conditions(self.target_actions, action_attrs):
				return False
		
		# Check environment targets
		if self.target_environments:
			environment_attrs = request_context.get('environment', {})
			if not self._matches_target_conditions(self.target_environments, environment_attrs):
				return False
		
		return True
	
	def _matches_target_conditions(self, target_conditions: List[Dict], attributes: Dict[str, Any]) -> bool:
		"""Check if attributes match target conditions"""
		for condition in target_conditions:
			attr_name = condition.get('attribute')
			operator = condition.get('operator', 'equals')
			expected_value = condition.get('value')
			
			attr_value = attributes.get(attr_name)
			
			if not self._evaluate_condition(attr_value, operator, expected_value):
				return False
		
		return True
	
	def _evaluate_condition(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
		"""Evaluate a single condition"""
		if operator == 'equals':
			return actual_value == expected_value
		elif operator == 'not_equals':
			return actual_value != expected_value
		elif operator == 'in':
			return actual_value in expected_value if isinstance(expected_value, list) else False
		elif operator == 'not_in':
			return actual_value not in expected_value if isinstance(expected_value, list) else True
		elif operator == 'contains':
			return expected_value in actual_value if isinstance(actual_value, (str, list)) else False
		elif operator == 'starts_with':
			return actual_value.startswith(expected_value) if isinstance(actual_value, str) else False
		elif operator == 'ends_with':
			return actual_value.endswith(expected_value) if isinstance(actual_value, str) else False
		elif operator == 'greater_than':
			return actual_value > expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'less_than':
			return actual_value < expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'greater_equal':
			return actual_value >= expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'less_equal':
			return actual_value <= expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'regex_match':
			import re
			return bool(re.match(expected_value, str(actual_value))) if expected_value else False
		elif operator == 'exists':
			return actual_value is not None
		elif operator == 'not_exists':
			return actual_value is None
		
		return False
	
	def update_evaluation_stats(self, evaluation_time_ms: float, decision: str) -> None:
		"""Update policy evaluation statistics"""
		self.evaluation_count += 1
		self.last_evaluated = datetime.utcnow()
		
		if decision == PolicyEffect.PERMIT.value:
			self.permit_count += 1
		elif decision == PolicyEffect.DENY.value:
			self.deny_count += 1
		
		# Update average evaluation time
		if self.evaluation_count > 1:
			self.average_evaluation_time_ms = (
				(self.average_evaluation_time_ms * (self.evaluation_count - 1) + evaluation_time_ms) / 
				self.evaluation_count
			)
		else:
			self.average_evaluation_time_ms = evaluation_time_ms
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert policy to dictionary representation"""
		return {
			'policy_id': self.policy_id,
			'tenant_id': self.tenant_id,
			'name': self.name,
			'display_name': self.display_name,
			'description': self.description,
			'version': self.version,
			'effect': self.effect,
			'algorithm': self.algorithm,
			'priority': self.priority,
			'target_subjects': self.target_subjects,
			'target_resources': self.target_resources,
			'target_actions': self.target_actions,
			'target_environments': self.target_environments,
			'is_active': self.is_active,
			'is_system_policy': self.is_system_policy,
			'effective_from': self.effective_from.isoformat() if self.effective_from else None,
			'effective_until': self.effective_until.isoformat() if self.effective_until else None,
			'tags': self.tags,
			'compliance_frameworks': self.compliance_frameworks,
			'risk_level': self.risk_level,
			'is_effective': self.is_effective(),
			'evaluation_count': self.evaluation_count,
			'permit_count': self.permit_count,
			'deny_count': self.deny_count,
			'last_evaluated': self.last_evaluated.isoformat() if self.last_evaluated else None,
			'average_evaluation_time_ms': self.average_evaluation_time_ms,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARPolicyRule(Model, AuditMixin, BaseMixin):
	"""
	Individual rules within ABAC policies.
	
	Defines specific conditions and logic for policy evaluation
	with support for complex boolean expressions.
	"""
	
	__tablename__ = 'ar_abac_policy_rule'
	
	# Identity
	rule_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	policy_id = Column(String(36), ForeignKey('ar_abac_policy.policy_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule Definition
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	rule_order = Column(Integer, default=0)  # Order within policy
	
	# Rule Logic
	rule_expression = Column(Text, nullable=False)  # Boolean expression using conditions
	combination_logic = Column(String(10), default='AND')  # AND, OR
	
	# Rule Configuration
	effect = Column(String(10), nullable=False, default=PolicyEffect.PERMIT.value)
	is_active = Column(Boolean, default=True)
	
	# Performance Tracking
	evaluation_count = Column(Integer, default=0)
	true_count = Column(Integer, default=0)
	false_count = Column(Integer, default=0)
	
	# Relationships
	policy = relationship("ARPolicy", back_populates="rules")
	conditions = relationship("ARPolicyCondition", back_populates="rule", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		Index('ix_ar_policy_rule_policy', 'policy_id'),
		Index('ix_ar_policy_rule_order', 'rule_order'),
	)
	
	def evaluate(self, request_context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
		"""Evaluate rule against request context"""
		if not self.is_active:
			return False, {'reason': 'Rule is inactive'}
		
		evaluation_details = {
			'rule_id': self.rule_id,
			'rule_name': self.name,
			'conditions_evaluated': [],
			'result': False,
			'reason': ''
		}
		
		try:
			# Evaluate all conditions
			condition_results = {}
			for condition in self.conditions:
				if condition.is_active:
					result, details = condition.evaluate(request_context)
					condition_results[condition.condition_id] = result
					evaluation_details['conditions_evaluated'].append({
						'condition_id': condition.condition_id,
						'condition_name': condition.name,
						'result': result,
						'details': details
					})
			
			# Evaluate rule expression
			if self.rule_expression:
				# Simple expression evaluation (in production, use a proper expression parser)
				rule_result = self._evaluate_expression(self.rule_expression, condition_results)
			else:
				# Default combination logic
				if self.combination_logic == 'AND':
					rule_result = all(condition_results.values())
				else:  # OR
					rule_result = any(condition_results.values())
			
			evaluation_details['result'] = rule_result
			evaluation_details['reason'] = f"Rule evaluated to {rule_result}"
			
			# Update statistics
			self.evaluation_count += 1
			if rule_result:
				self.true_count += 1
			else:
				self.false_count += 1
			
			return rule_result, evaluation_details
			
		except Exception as e:
			evaluation_details['result'] = False
			evaluation_details['reason'] = f"Rule evaluation error: {str(e)}"
			return False, evaluation_details
	
	def _evaluate_expression(self, expression: str, condition_results: Dict[str, bool]) -> bool:
		"""
		Evaluate rule expression using condition results.
		This is a simplified implementation - production should use a proper parser.
		"""
		# Replace condition IDs with their boolean results
		evaluated_expression = expression
		for condition_id, result in condition_results.items():
			evaluated_expression = evaluated_expression.replace(condition_id, str(result))
		
		# Simple boolean expression evaluation
		try:
			# This is unsafe for production - use a proper expression evaluator
			return eval(evaluated_expression.replace('AND', ' and ').replace('OR', ' or ').replace('NOT', ' not '))
		except:
			return False
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert rule to dictionary representation"""
		return {
			'rule_id': self.rule_id,
			'policy_id': self.policy_id,
			'tenant_id': self.tenant_id,
			'name': self.name,
			'description': self.description,
			'rule_order': self.rule_order,
			'rule_expression': self.rule_expression,
			'combination_logic': self.combination_logic,
			'effect': self.effect,
			'is_active': self.is_active,
			'evaluation_count': self.evaluation_count,
			'true_count': self.true_count,
			'false_count': self.false_count,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARPolicyCondition(Model, AuditMixin, BaseMixin):
	"""
	Individual conditions within policy rules.
	
	Defines specific attribute-based conditions for policy evaluation
	with comprehensive comparison operators and value types.
	"""
	
	__tablename__ = 'ar_abac_policy_condition'
	
	# Identity
	condition_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	policy_id = Column(String(36), ForeignKey('ar_abac_policy.policy_id'), nullable=True, index=True)
	rule_id = Column(String(36), ForeignKey('ar_abac_policy_rule.rule_id'), nullable=True, index=True)
	attribute_id = Column(String(36), ForeignKey('ar_abac_attribute.attribute_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Condition Definition
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Condition Logic
	operator = Column(String(50), nullable=False)  # equals, not_equals, in, contains, etc.
	expected_value = Column(JSON, nullable=True)  # The value to compare against
	
	# Advanced Conditions
	function_name = Column(String(100), nullable=True)  # Custom function for complex logic
	function_params = Column(JSON, default=dict)
	
	# Condition Configuration
	is_active = Column(Boolean, default=True)
	negate_result = Column(Boolean, default=False)  # Negate the condition result
	
	# Performance Tracking
	evaluation_count = Column(Integer, default=0)
	true_count = Column(Integer, default=0)
	false_count = Column(Integer, default=0)
	last_evaluated = Column(DateTime, nullable=True)
	
	# Relationships
	policy = relationship("ARPolicy", back_populates="conditions")
	rule = relationship("ARPolicyRule", back_populates="conditions")
	attribute = relationship("ARAttribute", back_populates="policy_conditions")
	
	# Table constraints
	__table_args__ = (
		Index('ix_ar_policy_condition_policy', 'policy_id'),
		Index('ix_ar_policy_condition_rule', 'rule_id'),
		Index('ix_ar_policy_condition_attribute', 'attribute_id'),
	)
	
	@validates('operator')
	def validate_operator(self, key, operator):
		"""Validate condition operator"""
		allowed_operators = [
			'equals', 'not_equals', 'in', 'not_in', 'contains', 'not_contains',
			'starts_with', 'ends_with', 'greater_than', 'less_than', 
			'greater_equal', 'less_equal', 'regex_match', 'exists', 'not_exists',
			'between', 'not_between', 'is_null', 'is_not_null'
		]
		if operator not in allowed_operators:
			raise ValueError(f"Operator must be one of: {allowed_operators}")
		return operator
	
	def evaluate(self, request_context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
		"""Evaluate condition against request context"""
		evaluation_details = {
			'condition_id': self.condition_id,
			'condition_name': self.name,
			'attribute_name': self.attribute.name if self.attribute else 'unknown',
			'attribute_category': self.attribute.category if self.attribute else 'unknown',
			'operator': self.operator,
			'expected_value': self.expected_value,
			'actual_value': None,
			'result': False,
			'reason': ''
		}
		
		if not self.is_active:
			evaluation_details['reason'] = 'Condition is inactive'
			return False, evaluation_details
		
		try:
			# Get attribute value from context
			attr_category = self.attribute.category if self.attribute else 'subject'
			attr_name = self.attribute.name if self.attribute else 'unknown'
			
			category_context = request_context.get(attr_category, {})
			actual_value = category_context.get(attr_name)
			evaluation_details['actual_value'] = actual_value
			
			# Apply custom function if specified
			if self.function_name:
				result = self._apply_custom_function(actual_value, request_context)
			else:
				# Standard operator evaluation
				result = self._evaluate_operator(actual_value, self.operator, self.expected_value)
			
			# Apply negation if configured
			if self.negate_result:
				result = not result
			
			evaluation_details['result'] = result
			evaluation_details['reason'] = f"Condition evaluated to {result}"
			
			# Update statistics
			self.evaluation_count += 1
			self.last_evaluated = datetime.utcnow()
			if result:
				self.true_count += 1
			else:
				self.false_count += 1
			
			return result, evaluation_details
			
		except Exception as e:
			evaluation_details['result'] = False
			evaluation_details['reason'] = f"Condition evaluation error: {str(e)}"
			return False, evaluation_details
	
	def _evaluate_operator(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
		"""Evaluate condition using specified operator"""
		if operator == 'equals':
			return actual_value == expected_value
		elif operator == 'not_equals':
			return actual_value != expected_value
		elif operator == 'in':
			return actual_value in expected_value if isinstance(expected_value, list) else False
		elif operator == 'not_in':
			return actual_value not in expected_value if isinstance(expected_value, list) else True
		elif operator == 'contains':
			return expected_value in actual_value if isinstance(actual_value, (str, list)) else False
		elif operator == 'not_contains':
			return expected_value not in actual_value if isinstance(actual_value, (str, list)) else True
		elif operator == 'starts_with':
			return actual_value.startswith(expected_value) if isinstance(actual_value, str) else False
		elif operator == 'ends_with':
			return actual_value.endswith(expected_value) if isinstance(actual_value, str) else False
		elif operator == 'greater_than':
			return actual_value > expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'less_than':
			return actual_value < expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'greater_equal':
			return actual_value >= expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'less_equal':
			return actual_value <= expected_value if isinstance(actual_value, (int, float)) else False
		elif operator == 'regex_match':
			import re
			return bool(re.match(expected_value, str(actual_value))) if expected_value else False
		elif operator == 'exists':
			return actual_value is not None
		elif operator == 'not_exists':
			return actual_value is None
		elif operator == 'between':
			if isinstance(expected_value, list) and len(expected_value) == 2:
				return expected_value[0] <= actual_value <= expected_value[1]
			return False
		elif operator == 'not_between':
			if isinstance(expected_value, list) and len(expected_value) == 2:
				return not (expected_value[0] <= actual_value <= expected_value[1])
			return True
		elif operator == 'is_null':
			return actual_value is None
		elif operator == 'is_not_null':
			return actual_value is not None
		
		return False
	
	def _apply_custom_function(self, actual_value: Any, request_context: Dict[str, Any]) -> bool:
		"""Apply custom function for complex condition logic"""
		# This would integrate with a custom function registry
		# For now, implement some common custom functions
		
		if self.function_name == 'time_based_access':
			# Check if current time is within allowed hours
			from datetime import datetime
			current_hour = datetime.utcnow().hour
			allowed_hours = self.function_params.get('allowed_hours', [])
			return current_hour in allowed_hours
		
		elif self.function_name == 'location_based':
			# Check if user location is in allowed regions
			user_location = request_context.get('environment', {}).get('user_location')
			allowed_locations = self.function_params.get('allowed_locations', [])
			return user_location in allowed_locations
		
		elif self.function_name == 'data_classification':
			# Check data classification level
			data_classification = actual_value
			max_classification = self.function_params.get('max_classification', 'public')
			classification_levels = ['public', 'internal', 'confidential', 'restricted']
			actual_level = classification_levels.index(data_classification) if data_classification in classification_levels else 0
			max_level = classification_levels.index(max_classification) if max_classification in classification_levels else 0
			return actual_level <= max_level
		
		elif self.function_name == 'role_hierarchy':
			# Check role hierarchy permissions
			user_roles = request_context.get('subject', {}).get('roles', [])
			required_role_level = self.function_params.get('required_level', 0)
			# This would integrate with role hierarchy logic
			return True  # Placeholder
		
		return False
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert condition to dictionary representation"""
		return {
			'condition_id': self.condition_id,
			'policy_id': self.policy_id,
			'rule_id': self.rule_id,
			'attribute_id': self.attribute_id,
			'tenant_id': self.tenant_id,
			'name': self.name,
			'description': self.description,
			'operator': self.operator,
			'expected_value': self.expected_value,
			'function_name': self.function_name,
			'function_params': self.function_params,
			'is_active': self.is_active,
			'negate_result': self.negate_result,
			'evaluation_count': self.evaluation_count,
			'true_count': self.true_count,
			'false_count': self.false_count,
			'last_evaluated': self.last_evaluated.isoformat() if self.last_evaluated else None,
			'attribute_name': self.attribute.name if self.attribute else None,
			'attribute_category': self.attribute.category if self.attribute else None,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARPolicySet(Model, AuditMixin, BaseMixin):
	"""
	ABAC Policy Sets for organizing and combining related policies.
	
	Groups policies together with combination algorithms
	for complex authorization scenarios.
	"""
	
	__tablename__ = 'ar_abac_policy_set'
	
	# Identity
	policy_set_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Policy Set Definition
	name = Column(String(200), nullable=False)
	display_name = Column(String(300), nullable=True)
	description = Column(Text, nullable=True)
	version = Column(String(20), default='1.0.0')
	
	# Combination Algorithm
	combination_algorithm = Column(String(50), default=PolicyAlgorithm.DENY_OVERRIDES.value)
	
	# Target Definition
	target_subjects = Column(JSON, default=list)
	target_resources = Column(JSON, default=list)
	target_actions = Column(JSON, default=list)
	target_environments = Column(JSON, default=list)
	
	# Policy Set Configuration
	is_active = Column(Boolean, default=True)
	priority = Column(Integer, default=100)
	
	# Time Constraints
	effective_from = Column(DateTime, nullable=True)
	effective_until = Column(DateTime, nullable=True)
	
	# Relationships
	policy_mappings = relationship("ARPolicySetMapping", back_populates="policy_set", cascade="all, delete-orphan")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'name', 'version', name='uq_ar_policy_set_name_version'),
		Index('ix_ar_policy_set_tenant_active', 'tenant_id', 'is_active'),
		Index('ix_ar_policy_set_priority', 'priority'),
	)
	
	def is_effective(self) -> bool:
		"""Check if policy set is currently effective"""
		if not self.is_active:
			return False
		
		now = datetime.utcnow()
		
		if self.effective_from and now < self.effective_from:
			return False
		
		if self.effective_until and now > self.effective_until:
			return False
		
		return True
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert policy set to dictionary representation"""
		return {
			'policy_set_id': self.policy_set_id,
			'tenant_id': self.tenant_id,
			'name': self.name,
			'display_name': self.display_name,
			'description': self.description,
			'version': self.version,
			'combination_algorithm': self.combination_algorithm,
			'target_subjects': self.target_subjects,
			'target_resources': self.target_resources,
			'target_actions': self.target_actions,
			'target_environments': self.target_environments,
			'is_active': self.is_active,
			'priority': self.priority,
			'effective_from': self.effective_from.isoformat() if self.effective_from else None,
			'effective_until': self.effective_until.isoformat() if self.effective_until else None,
			'is_effective': self.is_effective(),
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARPolicySetMapping(Model, AuditMixin, BaseMixin):
	"""
	Mapping between Policy Sets and individual Policies.
	"""
	
	__tablename__ = 'ar_abac_policy_set_mapping'
	
	# Identity
	mapping_id = Column(String(36), unique=True, nullable=False, default=uuid7str, primary_key=True)
	policy_set_id = Column(String(36), ForeignKey('ar_abac_policy_set.policy_set_id'), nullable=False, index=True)
	policy_id = Column(String(36), ForeignKey('ar_abac_policy.policy_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Mapping Configuration
	policy_order = Column(Integer, default=0)
	is_active = Column(Boolean, default=True)
	
	# Relationships
	policy_set = relationship("ARPolicySet", back_populates="policy_mappings")
	policy = relationship("ARPolicy")
	
	# Table constraints
	__table_args__ = (
		UniqueConstraint('policy_set_id', 'policy_id', name='uq_ar_policy_set_policy'),
		Index('ix_ar_policy_set_mapping_order', 'policy_order'),
	)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert mapping to dictionary representation"""
		return {
			'mapping_id': self.mapping_id,
			'policy_set_id': self.policy_set_id,
			'policy_id': self.policy_id,
			'tenant_id': self.tenant_id,
			'policy_order': self.policy_order,
			'is_active': self.is_active,
			'policy_name': self.policy.name if self.policy else None,
			'created_on': self.created_on.isoformat() if self.created_on else None
		}


class ARAccessRequest(Model, AuditMixin, BaseMixin):
	"""
	ABAC Access Request logging for audit and analysis.
	
	Records all access requests with complete context
	for security monitoring and compliance reporting.
	"""
	
	__tablename__ = 'ar_abac_access_request'
	
	# Identity
	request_id = Column(String(36), unique=True, nullable=False, default=uuid7str, primary_key=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Request Context
	subject_id = Column(String(36), nullable=False, index=True)  # User ID
	resource_type = Column(String(100), nullable=False, index=True)
	resource_id = Column(String(200), nullable=False, index=True)
	action = Column(String(50), nullable=False, index=True)
	
	# Full Request Context
	subject_attributes = Column(JSON, default=dict)
	resource_attributes = Column(JSON, default=dict)
	action_attributes = Column(JSON, default=dict)
	environment_attributes = Column(JSON, default=dict)
	
	# Decision Information
	decision = Column(String(10), nullable=False, index=True)  # permit, deny
	decision_reason = Column(Text, nullable=True)
	policies_evaluated = Column(JSON, default=list)
	evaluation_time_ms = Column(Float, nullable=False)
	
	# Request Metadata
	session_id = Column(String(128), nullable=True, index=True)
	ip_address = Column(String(45), nullable=True)
	user_agent = Column(Text, nullable=True)
	request_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	
	# Table constraints
	__table_args__ = (
		Index('ix_ar_access_request_subject_resource', 'subject_id', 'resource_type', 'resource_id'),
		Index('ix_ar_access_request_decision_time', 'decision', 'request_timestamp'),
		Index('ix_ar_access_request_tenant_time', 'tenant_id', 'request_timestamp'),
	)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert access request to dictionary representation"""
		return {
			'request_id': self.request_id,
			'tenant_id': self.tenant_id,
			'subject_id': self.subject_id,
			'resource_type': self.resource_type,
			'resource_id': self.resource_id,
			'action': self.action,
			'subject_attributes': self.subject_attributes,
			'resource_attributes': self.resource_attributes,
			'action_attributes': self.action_attributes,
			'environment_attributes': self.environment_attributes,
			'decision': self.decision,
			'decision_reason': self.decision_reason,
			'policies_evaluated': self.policies_evaluated,
			'evaluation_time_ms': self.evaluation_time_ms,
			'session_id': self.session_id,
			'ip_address': self.ip_address,
			'request_timestamp': self.request_timestamp.isoformat(),
			'created_on': self.created_on.isoformat() if self.created_on else None
		}