"""
APG Employee Data Management - Data Quality Rules Configuration

Comprehensive data quality rules engine with configurable validation,
correction rules, and AI-powered quality assessment policies.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import json
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import re

from .data_quality_engine import DataQualityDimension, ValidationSeverity, AutoCorrectionAction


class RuleCategory(str, Enum):
	"""Categories of data quality rules."""
	BUSINESS_RULE = "business_rule"
	COMPLIANCE_RULE = "compliance_rule"
	TECHNICAL_RULE = "technical_rule"
	INTEGRITY_RULE = "integrity_rule"
	ENRICHMENT_RULE = "enrichment_rule"


class RuleOperator(str, Enum):
	"""Operators for rule conditions."""
	EQUALS = "equals"
	NOT_EQUALS = "not_equals"
	GREATER_THAN = "greater_than"
	LESS_THAN = "less_than"
	GREATER_EQUAL = "greater_equal"
	LESS_EQUAL = "less_equal"
	CONTAINS = "contains"
	NOT_CONTAINS = "not_contains"
	STARTS_WITH = "starts_with"
	ENDS_WITH = "ends_with"
	MATCHES_PATTERN = "matches_pattern"
	IS_NULL = "is_null"
	IS_NOT_NULL = "is_not_null"
	IN_LIST = "in_list"
	NOT_IN_LIST = "not_in_list"
	LENGTH_EQUALS = "length_equals"
	LENGTH_BETWEEN = "length_between"
	DATE_BEFORE = "date_before"
	DATE_AFTER = "date_after"
	DATE_BETWEEN = "date_between"


@dataclass
class RuleCondition:
	"""Individual condition within a data quality rule."""
	field_name: str
	operator: RuleOperator
	expected_value: Any = None
	error_message: str = ""
	severity: ValidationSeverity = ValidationSeverity.MEDIUM


@dataclass
class QualityRule:
	"""Comprehensive data quality rule definition."""
	rule_id: str
	rule_name: str
	description: str
	category: RuleCategory
	dimension: DataQualityDimension
	conditions: List[RuleCondition] = field(default_factory=list)
	
	# Rule behavior
	enabled: bool = True
	auto_correct: bool = False
	correction_action: Optional[AutoCorrectionAction] = None
	
	# Rule metadata
	priority: int = 1
	business_impact: str = ""
	compliance_requirement: str = ""
	
	# AI enhancement
	ai_enhanced: bool = False
	ai_confidence_threshold: float = 0.7
	
	# Rule execution
	pre_conditions: List[str] = field(default_factory=list)
	post_actions: List[str] = field(default_factory=list)
	
	# Audit and tracking
	created_date: datetime = field(default_factory=datetime.utcnow)
	last_modified: datetime = field(default_factory=datetime.utcnow)
	version: str = "1.0"


class DataQualityRulesEngine:
	"""Comprehensive data quality rules management and execution engine."""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(f"QualityRulesEngine.{tenant_id}")
		
		# Rules registry
		self.rules_registry: Dict[str, QualityRule] = {}
		self.rules_by_category: Dict[RuleCategory, List[QualityRule]] = {}
		self.rules_by_dimension: Dict[DataQualityDimension, List[QualityRule]] = {}
		
		# Rule execution statistics
		self.execution_stats: Dict[str, Dict[str, int]] = {}
		
		# Custom validators
		self.custom_validators: Dict[str, Callable] = {}
		
		# Initialize with default rules
		self._initialize_default_rules()

	def _initialize_default_rules(self) -> None:
		"""Initialize the engine with comprehensive default data quality rules."""
		
		# ================================================================
		# COMPLETENESS RULES
		# ================================================================
		
		# Required Fields Rule
		self.add_rule(QualityRule(
			rule_id="completeness_required_fields",
			rule_name="Required Fields Completeness",
			description="Ensure all required fields are populated",
			category=RuleCategory.BUSINESS_RULE,
			dimension=DataQualityDimension.COMPLETENESS,
			conditions=[
				RuleCondition(
					field_name="first_name",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="First name is required",
					severity=ValidationSeverity.CRITICAL
				),
				RuleCondition(
					field_name="last_name",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Last name is required",
					severity=ValidationSeverity.CRITICAL
				),
				RuleCondition(
					field_name="work_email",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Work email is required",
					severity=ValidationSeverity.CRITICAL
				),
				RuleCondition(
					field_name="department_id",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Department assignment is required",
					severity=ValidationSeverity.CRITICAL
				),
				RuleCondition(
					field_name="position_id",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Position assignment is required",
					severity=ValidationSeverity.CRITICAL
				),
				RuleCondition(
					field_name="hire_date",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Hire date is required",
					severity=ValidationSeverity.CRITICAL
				)
			],
			business_impact="Critical for employee identification and organizational structure",
			priority=1
		))
		
		# Important Fields Rule
		self.add_rule(QualityRule(
			rule_id="completeness_important_fields",
			rule_name="Important Fields Completeness",
			description="Ensure important fields are populated for comprehensive employee profiles",
			category=RuleCategory.BUSINESS_RULE,
			dimension=DataQualityDimension.COMPLETENESS,
			conditions=[
				RuleCondition(
					field_name="phone_mobile",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Mobile phone number should be provided",
					severity=ValidationSeverity.MEDIUM
				),
				RuleCondition(
					field_name="manager_id",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Manager assignment improves organizational clarity",
					severity=ValidationSeverity.MEDIUM
				),
				RuleCondition(
					field_name="base_salary",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Salary information important for compensation analysis",
					severity=ValidationSeverity.MEDIUM
				)
			],
			auto_correct=True,
			correction_action=AutoCorrectionAction.DATA_ENRICHMENT,
			business_impact="Improves data richness and analytics capability",
			priority=2
		))
		
		# ================================================================
		# ACCURACY RULES
		# ================================================================
		
		# Email Format Validation
		self.add_rule(QualityRule(
			rule_id="accuracy_email_format",
			rule_name="Email Format Validation",
			description="Validate email addresses follow correct format",
			category=RuleCategory.TECHNICAL_RULE,
			dimension=DataQualityDimension.ACCURACY,
			conditions=[
				RuleCondition(
					field_name="work_email",
					operator=RuleOperator.MATCHES_PATTERN,
					expected_value=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
					error_message="Work email format is invalid",
					severity=ValidationSeverity.HIGH
				),
				RuleCondition(
					field_name="personal_email",
					operator=RuleOperator.MATCHES_PATTERN,
					expected_value=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
					error_message="Personal email format is invalid",
					severity=ValidationSeverity.MEDIUM
				)
			],
			auto_correct=True,
			correction_action=AutoCorrectionAction.FORMAT_STANDARDIZATION,
			business_impact="Ensures reliable communication and system integration",
			priority=1
		))
		
		# Phone Number Validation
		self.add_rule(QualityRule(
			rule_id="accuracy_phone_format",
			rule_name="Phone Number Format Validation",
			description="Validate phone numbers follow acceptable formats",
			category=RuleCategory.TECHNICAL_RULE,
			dimension=DataQualityDimension.ACCURACY,
			conditions=[
				RuleCondition(
					field_name="phone_mobile",
					operator=RuleOperator.MATCHES_PATTERN,
					expected_value=r'^[\+]?[1-9][\d]{0,15}$',
					error_message="Mobile phone number format is invalid",
					severity=ValidationSeverity.MEDIUM
				),
				RuleCondition(
					field_name="phone_work",
					operator=RuleOperator.MATCHES_PATTERN,
					expected_value=r'^[\+]?[1-9][\d]{0,15}$',
					error_message="Work phone number format is invalid",
					severity=ValidationSeverity.MEDIUM
				)
			],
			auto_correct=True,
			correction_action=AutoCorrectionAction.FORMAT_STANDARDIZATION,
			business_impact="Ensures reliable contact information",
			priority=2
		))
		
		# Salary Range Validation
		self.add_rule(QualityRule(
			rule_id="accuracy_salary_range",
			rule_name="Salary Range Validation",
			description="Validate salary amounts are within reasonable ranges",
			category=RuleCategory.BUSINESS_RULE,
			dimension=DataQualityDimension.ACCURACY,
			conditions=[
				RuleCondition(
					field_name="base_salary",
					operator=RuleOperator.GREATER_THAN,
					expected_value=15000,  # Minimum wage consideration
					error_message="Base salary below minimum threshold",
					severity=ValidationSeverity.HIGH
				),
				RuleCondition(
					field_name="base_salary",
					operator=RuleOperator.LESS_THAN,
					expected_value=10000000,  # 10M cap
					error_message="Base salary exceeds maximum threshold",
					severity=ValidationSeverity.MEDIUM
				),
				RuleCondition(
					field_name="hourly_rate",
					operator=RuleOperator.GREATER_THAN,
					expected_value=7.25,  # Federal minimum wage
					error_message="Hourly rate below minimum wage",
					severity=ValidationSeverity.HIGH
				),
				RuleCondition(
					field_name="hourly_rate",
					operator=RuleOperator.LESS_THAN,
					expected_value=1000,  # $1000/hour cap
					error_message="Hourly rate exceeds reasonable maximum",
					severity=ValidationSeverity.MEDIUM
				)
			],
			business_impact="Prevents payroll errors and ensures compliance",
			priority=1
		))
		
		# ================================================================
		# CONSISTENCY RULES
		# ================================================================
		
		# Name Consistency
		self.add_rule(QualityRule(
			rule_id="consistency_name_fields",
			rule_name="Name Fields Consistency",
			description="Ensure name fields are consistent with each other",
			category=RuleCategory.BUSINESS_RULE,
			dimension=DataQualityDimension.CONSISTENCY,
			conditions=[],  # This requires custom validation logic
			auto_correct=True,
			correction_action=AutoCorrectionAction.FORMAT_STANDARDIZATION,
			business_impact="Maintains consistent employee identification",
			priority=2
		))
		
		# Date Logic Consistency
		self.add_rule(QualityRule(
			rule_id="consistency_employment_dates",
			rule_name="Employment Dates Logic",
			description="Ensure employment dates follow logical sequence",
			category=RuleCategory.BUSINESS_RULE,
			dimension=DataQualityDimension.CONSISTENCY,
			conditions=[],  # Custom validation for date logic
			business_impact="Ensures accurate tenure and employment calculations",
			priority=1
		))
		
		# Employment Status Consistency
		self.add_rule(QualityRule(
			rule_id="consistency_employment_status",
			rule_name="Employment Status Consistency",
			description="Ensure employment status matches termination dates",
			category=RuleCategory.BUSINESS_RULE,
			dimension=DataQualityDimension.CONSISTENCY,
			conditions=[],  # Custom validation
			auto_correct=True,
			correction_action=AutoCorrectionAction.REFERENCE_VALIDATION,
			business_impact="Prevents status inconsistencies affecting payroll and access",
			priority=1
		))
		
		# ================================================================
		# VALIDITY RULES
		# ================================================================
		
		# Date Validity
		self.add_rule(QualityRule(
			rule_id="validity_date_ranges",
			rule_name="Date Range Validity",
			description="Validate dates are within reasonable ranges",
			category=RuleCategory.TECHNICAL_RULE,
			dimension=DataQualityDimension.VALIDITY,
			conditions=[
				RuleCondition(
					field_name="hire_date",
					operator=RuleOperator.DATE_AFTER,
					expected_value="1950-01-01",
					error_message="Hire date unrealistically early",
					severity=ValidationSeverity.HIGH
				),
				RuleCondition(
					field_name="hire_date",
					operator=RuleOperator.DATE_BEFORE,
					expected_value=datetime.now().date(),
					error_message="Hire date cannot be in the future",
					severity=ValidationSeverity.HIGH
				),
				RuleCondition(
					field_name="date_of_birth",
					operator=RuleOperator.DATE_AFTER,
					expected_value="1920-01-01",
					error_message="Birth date unrealistically early",
					severity=ValidationSeverity.MEDIUM
				),
				RuleCondition(
					field_name="date_of_birth",
					operator=RuleOperator.DATE_BEFORE,
					expected_value=datetime.now().date(),
					error_message="Birth date cannot be in the future",
					severity=ValidationSeverity.HIGH
				)
			],
			business_impact="Prevents impossible dates affecting calculations",
			priority=1
		))
		
		# Employee Number Format
		self.add_rule(QualityRule(
			rule_id="validity_employee_number",
			rule_name="Employee Number Format",
			description="Validate employee number follows organizational format",
			category=RuleCategory.TECHNICAL_RULE,
			dimension=DataQualityDimension.VALIDITY,
			conditions=[
				RuleCondition(
					field_name="employee_number",
					operator=RuleOperator.MATCHES_PATTERN,
					expected_value=r'^EMP\d{6}$',
					error_message="Employee number must follow EMP###### format",
					severity=ValidationSeverity.HIGH
				)
			],
			auto_correct=True,
			correction_action=AutoCorrectionAction.FORMAT_STANDARDIZATION,
			business_impact="Ensures consistent employee identification",
			priority=2
		))
		
		# ================================================================
		# UNIQUENESS RULES
		# ================================================================
		
		# Unique Identifiers
		self.add_rule(QualityRule(
			rule_id="uniqueness_identifiers",
			rule_name="Unique Employee Identifiers",
			description="Ensure employee identifiers are unique",
			category=RuleCategory.INTEGRITY_RULE,
			dimension=DataQualityDimension.UNIQUENESS,
			conditions=[],  # Requires database queries
			business_impact="Prevents duplicate employee records",
			priority=1
		))
		
		# ================================================================
		# COMPLIANCE RULES
		# ================================================================
		
		# Age Compliance
		self.add_rule(QualityRule(
			rule_id="compliance_minimum_age",
			rule_name="Minimum Age Compliance",
			description="Ensure employees meet minimum age requirements",
			category=RuleCategory.COMPLIANCE_RULE,
			dimension=DataQualityDimension.VALIDITY,
			conditions=[],  # Custom age calculation
			compliance_requirement="Fair Labor Standards Act",
			business_impact="Ensures compliance with labor laws",
			priority=1
		))
		
		# Tax Information Compliance
		self.add_rule(QualityRule(
			rule_id="compliance_tax_info",
			rule_name="Tax Information Completeness",
			description="Ensure required tax information is provided",
			category=RuleCategory.COMPLIANCE_RULE,
			dimension=DataQualityDimension.COMPLETENESS,
			conditions=[
				RuleCondition(
					field_name="tax_id",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Tax ID required for payroll compliance",
					severity=ValidationSeverity.HIGH
				),
				RuleCondition(
					field_name="tax_country",
					operator=RuleOperator.IS_NOT_NULL,
					error_message="Tax country required for compliance",
					severity=ValidationSeverity.HIGH
				)
			],
			compliance_requirement="IRS and tax authority requirements",
			business_impact="Ensures payroll tax compliance",
			priority=1
		))

	def add_rule(self, rule: QualityRule) -> None:
		"""Add a new quality rule to the engine."""
		self.rules_registry[rule.rule_id] = rule
		
		# Index by category
		if rule.category not in self.rules_by_category:
			self.rules_by_category[rule.category] = []
		self.rules_by_category[rule.category].append(rule)
		
		# Index by dimension
		if rule.dimension not in self.rules_by_dimension:
			self.rules_by_dimension[rule.dimension] = []
		self.rules_by_dimension[rule.dimension].append(rule)
		
		self.logger.info(f"Added quality rule: {rule.rule_id}")

	def remove_rule(self, rule_id: str) -> bool:
		"""Remove a quality rule from the engine."""
		if rule_id in self.rules_registry:
			rule = self.rules_registry[rule_id]
			
			# Remove from indices
			if rule.category in self.rules_by_category:
				self.rules_by_category[rule.category] = [
					r for r in self.rules_by_category[rule.category] if r.rule_id != rule_id
				]
			
			if rule.dimension in self.rules_by_dimension:
				self.rules_by_dimension[rule.dimension] = [
					r for r in self.rules_by_dimension[rule.dimension] if r.rule_id != rule_id
				]
			
			# Remove from registry
			del self.rules_registry[rule_id]
			
			self.logger.info(f"Removed quality rule: {rule_id}")
			return True
		
		return False

	def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
		"""Update an existing quality rule."""
		if rule_id in self.rules_registry:
			rule = self.rules_registry[rule_id]
			
			# Update allowed fields
			allowed_updates = [
				'rule_name', 'description', 'enabled', 'auto_correct',
				'correction_action', 'priority', 'business_impact',
				'compliance_requirement', 'ai_enhanced', 'ai_confidence_threshold'
			]
			
			for field, value in updates.items():
				if field in allowed_updates:
					setattr(rule, field, value)
			
			rule.last_modified = datetime.utcnow()
			rule.version = f"{float(rule.version) + 0.1:.1f}"
			
			self.logger.info(f"Updated quality rule: {rule_id}")
			return True
		
		return False

	def get_rules_for_dimension(self, dimension: DataQualityDimension) -> List[QualityRule]:
		"""Get all rules for a specific quality dimension."""
		return [rule for rule in self.rules_by_dimension.get(dimension, []) if rule.enabled]

	def get_rules_for_category(self, category: RuleCategory) -> List[QualityRule]:
		"""Get all rules for a specific category."""
		return [rule for rule in self.rules_by_category.get(category, []) if rule.enabled]

	def evaluate_rule_condition(self, condition: RuleCondition, field_value: Any) -> Tuple[bool, str]:
		"""Evaluate a single rule condition against a field value."""
		try:
			operator = condition.operator
			expected = condition.expected_value
			
			# Handle null checks first
			if operator == RuleOperator.IS_NULL:
				return field_value is None or field_value == "", ""
			
			if operator == RuleOperator.IS_NOT_NULL:
				is_valid = field_value is not None and str(field_value).strip() != ""
				return is_valid, "" if is_valid else condition.error_message
			
			# If field is null/empty and we're not checking for null, it's invalid
			if field_value is None or str(field_value).strip() == "":
				return False, f"Field {condition.field_name} is empty"
			
			# Convert to string for most operations
			str_value = str(field_value).strip()
			
			# Equality operators
			if operator == RuleOperator.EQUALS:
				is_valid = str_value == str(expected)
				return is_valid, "" if is_valid else condition.error_message
			
			if operator == RuleOperator.NOT_EQUALS:
				is_valid = str_value != str(expected)
				return is_valid, "" if is_valid else condition.error_message
			
			# Numeric comparisons
			if operator in [RuleOperator.GREATER_THAN, RuleOperator.LESS_THAN, 
							RuleOperator.GREATER_EQUAL, RuleOperator.LESS_EQUAL]:
				try:
					num_value = float(field_value)
					num_expected = float(expected)
					
					if operator == RuleOperator.GREATER_THAN:
						is_valid = num_value > num_expected
					elif operator == RuleOperator.LESS_THAN:
						is_valid = num_value < num_expected
					elif operator == RuleOperator.GREATER_EQUAL:
						is_valid = num_value >= num_expected
					elif operator == RuleOperator.LESS_EQUAL:
						is_valid = num_value <= num_expected
					
					return is_valid, "" if is_valid else condition.error_message
					
				except (ValueError, TypeError):
					return False, f"Cannot compare non-numeric value: {field_value}"
			
			# String operations
			if operator == RuleOperator.CONTAINS:
				is_valid = str(expected) in str_value
				return is_valid, "" if is_valid else condition.error_message
			
			if operator == RuleOperator.NOT_CONTAINS:
				is_valid = str(expected) not in str_value
				return is_valid, "" if is_valid else condition.error_message
			
			if operator == RuleOperator.STARTS_WITH:
				is_valid = str_value.startswith(str(expected))
				return is_valid, "" if is_valid else condition.error_message
			
			if operator == RuleOperator.ENDS_WITH:
				is_valid = str_value.endswith(str(expected))
				return is_valid, "" if is_valid else condition.error_message
			
			# Pattern matching
			if operator == RuleOperator.MATCHES_PATTERN:
				try:
					is_valid = bool(re.match(str(expected), str_value))
					return is_valid, "" if is_valid else condition.error_message
				except re.error:
					return False, f"Invalid regex pattern: {expected}"
			
			# List operations
			if operator == RuleOperator.IN_LIST:
				is_valid = field_value in expected if isinstance(expected, (list, tuple, set)) else False
				return is_valid, "" if is_valid else condition.error_message
			
			if operator == RuleOperator.NOT_IN_LIST:
				is_valid = field_value not in expected if isinstance(expected, (list, tuple, set)) else True
				return is_valid, "" if is_valid else condition.error_message
			
			# Length operations
			if operator == RuleOperator.LENGTH_EQUALS:
				is_valid = len(str_value) == int(expected)
				return is_valid, "" if is_valid else condition.error_message
			
			if operator == RuleOperator.LENGTH_BETWEEN:
				if isinstance(expected, (list, tuple)) and len(expected) == 2:
					length = len(str_value)
					is_valid = expected[0] <= length <= expected[1]
					return is_valid, "" if is_valid else condition.error_message
				return False, "LENGTH_BETWEEN requires [min, max] values"
			
			# Date operations
			if operator in [RuleOperator.DATE_BEFORE, RuleOperator.DATE_AFTER, RuleOperator.DATE_BETWEEN]:
				try:
					if isinstance(field_value, str):
						field_date = datetime.strptime(field_value, '%Y-%m-%d').date()
					elif isinstance(field_value, datetime):
						field_date = field_value.date()
					elif isinstance(field_value, date):
						field_date = field_value
					else:
						return False, f"Invalid date value: {field_value}"
					
					if operator == RuleOperator.DATE_BEFORE:
						expected_date = datetime.strptime(str(expected), '%Y-%m-%d').date() if isinstance(expected, str) else expected
						is_valid = field_date < expected_date
						return is_valid, "" if is_valid else condition.error_message
					
					elif operator == RuleOperator.DATE_AFTER:
						expected_date = datetime.strptime(str(expected), '%Y-%m-%d').date() if isinstance(expected, str) else expected
						is_valid = field_date > expected_date
						return is_valid, "" if is_valid else condition.error_message
					
					elif operator == RuleOperator.DATE_BETWEEN:
						if isinstance(expected, (list, tuple)) and len(expected) == 2:
							start_date = datetime.strptime(str(expected[0]), '%Y-%m-%d').date() if isinstance(expected[0], str) else expected[0]
							end_date = datetime.strptime(str(expected[1]), '%Y-%m-%d').date() if isinstance(expected[1], str) else expected[1]
							is_valid = start_date <= field_date <= end_date
							return is_valid, "" if is_valid else condition.error_message
						return False, "DATE_BETWEEN requires [start_date, end_date] values"
				
				except (ValueError, TypeError) as e:
					return False, f"Date parsing error: {str(e)}"
			
			# Unknown operator
			return False, f"Unknown operator: {operator}"
			
		except Exception as e:
			return False, f"Rule evaluation error: {str(e)}"

	def evaluate_rule(self, rule: QualityRule, employee_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Evaluate a complete quality rule against employee data."""
		rule_result = {
			'rule_id': rule.rule_id,
			'rule_name': rule.rule_name,
			'dimension': rule.dimension.value,
			'category': rule.category.value,
			'passed': True,
			'violations': [],
			'auto_correctable': rule.auto_correct,
			'correction_action': rule.correction_action.value if rule.correction_action else None,
			'business_impact': rule.business_impact
		}
		
		try:
			# Handle rules with no conditions (require custom logic)
			if not rule.conditions:
				if rule.rule_id == "consistency_name_fields":
					violations = self._check_name_consistency(employee_data)
					rule_result['violations'].extend(violations)
				elif rule.rule_id == "consistency_employment_dates":
					violations = self._check_date_consistency(employee_data)
					rule_result['violations'].extend(violations)
				elif rule.rule_id == "consistency_employment_status":
					violations = self._check_status_consistency(employee_data)
					rule_result['violations'].extend(violations)
				elif rule.rule_id == "uniqueness_identifiers":
					violations = self._check_identifier_uniqueness(employee_data)
					rule_result['violations'].extend(violations)
				elif rule.rule_id == "compliance_minimum_age":
					violations = self._check_minimum_age(employee_data)
					rule_result['violations'].extend(violations)
			else:
				# Evaluate standard conditions
				for condition in rule.conditions:
					field_value = employee_data.get(condition.field_name)
					is_valid, error_message = self.evaluate_rule_condition(condition, field_value)
					
					if not is_valid:
						rule_result['violations'].append({
							'field_name': condition.field_name,
							'field_value': field_value,
							'error_message': error_message,
							'severity': condition.severity.value
						})
			
			# Rule passes if no violations
			rule_result['passed'] = len(rule_result['violations']) == 0
			
			# Update execution statistics
			if rule.rule_id not in self.execution_stats:
				self.execution_stats[rule.rule_id] = {'executed': 0, 'passed': 0, 'failed': 0}
			
			self.execution_stats[rule.rule_id]['executed'] += 1
			if rule_result['passed']:
				self.execution_stats[rule.rule_id]['passed'] += 1
			else:
				self.execution_stats[rule.rule_id]['failed'] += 1
			
			return rule_result
			
		except Exception as e:
			self.logger.error(f"Rule evaluation failed for {rule.rule_id}: {str(e)}")
			rule_result['passed'] = False
			rule_result['violations'].append({
				'field_name': 'system',
				'field_value': None,
				'error_message': f"Rule evaluation error: {str(e)}",
				'severity': ValidationSeverity.HIGH.value
			})
			return rule_result

	# ================================================================
	# CUSTOM VALIDATION METHODS
	# ================================================================

	def _check_name_consistency(self, employee_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Check consistency between name fields."""
		violations = []
		
		first_name = employee_data.get('first_name', '').strip()
		middle_name = employee_data.get('middle_name', '').strip()
		last_name = employee_data.get('last_name', '').strip()
		full_name = employee_data.get('full_name', '').strip()
		
		if first_name and last_name and full_name:
			expected_full_name = f"{first_name} {middle_name} {last_name}".replace('  ', ' ').strip()
			if full_name.lower() != expected_full_name.lower():
				violations.append({
					'field_name': 'full_name',
					'field_value': full_name,
					'error_message': f"Full name '{full_name}' doesn't match first/last name '{expected_full_name}'",
					'severity': ValidationSeverity.MEDIUM.value
				})
		
		return violations

	def _check_date_consistency(self, employee_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Check logical consistency of employment dates."""
		violations = []
		
		hire_date = employee_data.get('hire_date')
		start_date = employee_data.get('start_date')
		termination_date = employee_data.get('termination_date')
		
		try:
			# Convert string dates to date objects
			if isinstance(hire_date, str):
				hire_date = datetime.strptime(hire_date, '%Y-%m-%d').date()
			if isinstance(start_date, str):
				start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
			if isinstance(termination_date, str):
				termination_date = datetime.strptime(termination_date, '%Y-%m-%d').date()
			
			# Check hire_date vs start_date
			if hire_date and start_date and start_date < hire_date:
				violations.append({
					'field_name': 'start_date',
					'field_value': start_date,
					'error_message': 'Start date cannot be before hire date',
					'severity': ValidationSeverity.HIGH.value
				})
			
			# Check hire_date vs termination_date
			if hire_date and termination_date and termination_date < hire_date:
				violations.append({
					'field_name': 'termination_date',
					'field_value': termination_date,
					'error_message': 'Termination date cannot be before hire date',
					'severity': ValidationSeverity.CRITICAL.value
				})
		
		except (ValueError, TypeError):
			pass  # Date parsing errors handled elsewhere
		
		return violations

	def _check_status_consistency(self, employee_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Check consistency between employment status and termination date."""
		violations = []
		
		employment_status = employee_data.get('employment_status', '').lower()
		termination_date = employee_data.get('termination_date')
		
		try:
			current_date = date.today()
			
			# Active employee with past termination date
			if employment_status == 'active' and termination_date:
				if isinstance(termination_date, str):
					term_date = datetime.strptime(termination_date, '%Y-%m-%d').date()
				else:
					term_date = termination_date
				
				if term_date <= current_date:
					violations.append({
						'field_name': 'employment_status',
						'field_value': employment_status,
						'error_message': 'Employee marked as active but has past termination date',
						'severity': ValidationSeverity.HIGH.value
					})
			
			# Terminated employee without termination date
			if employment_status == 'terminated' and not termination_date:
				violations.append({
					'field_name': 'termination_date',
					'field_value': termination_date,
					'error_message': 'Terminated employee must have termination date',
					'severity': ValidationSeverity.CRITICAL.value
				})
		
		except (ValueError, TypeError):
			pass
		
		return violations

	def _check_identifier_uniqueness(self, employee_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Check uniqueness of employee identifiers."""
		violations = []
		
		# This would typically query the database to check for duplicates
		# For demo purposes, we'll create a simplified check
		
		unique_fields = ['employee_number', 'work_email', 'badge_id']
		
		for field in unique_fields:
			field_value = employee_data.get(field)
			if field_value:
				# Simulate duplicate check (in production, this would query the database)
				if self._simulate_duplicate_check(field, field_value):
					violations.append({
						'field_name': field,
						'field_value': field_value,
						'error_message': f'Duplicate {field} found in system',
						'severity': ValidationSeverity.HIGH.value
					})
		
		return violations

	def _check_minimum_age(self, employee_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Check minimum age compliance."""
		violations = []
		
		date_of_birth = employee_data.get('date_of_birth')
		if date_of_birth:
			try:
				if isinstance(date_of_birth, str):
					birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
				else:
					birth_date = date_of_birth
				
				age = (date.today() - birth_date).days / 365.25
				
				if age < 16:  # Minimum working age
					violations.append({
						'field_name': 'date_of_birth',
						'field_value': date_of_birth,
						'error_message': f'Employee age ({age:.1f}) below minimum working age',
						'severity': ValidationSeverity.CRITICAL.value
					})
			
			except (ValueError, TypeError):
				pass
		
		return violations

	def _simulate_duplicate_check(self, field: str, value: Any) -> bool:
		"""Simulate database duplicate check."""
		# In production, this would query the database
		# For demo, return False (no duplicates)
		return False

	# ================================================================
	# RULE MANAGEMENT AND REPORTING
	# ================================================================

	def get_all_rules(self) -> Dict[str, QualityRule]:
		"""Get all registered quality rules."""
		return self.rules_registry.copy()

	def get_enabled_rules(self) -> Dict[str, QualityRule]:
		"""Get only enabled quality rules."""
		return {rule_id: rule for rule_id, rule in self.rules_registry.items() if rule.enabled}

	def get_rule_statistics(self) -> Dict[str, Any]:
		"""Get execution statistics for all rules."""
		total_rules = len(self.rules_registry)
		enabled_rules = len([r for r in self.rules_registry.values() if r.enabled])
		
		stats = {
			'total_rules': total_rules,
			'enabled_rules': enabled_rules,
			'disabled_rules': total_rules - enabled_rules,
			'rules_by_category': {cat.value: len(rules) for cat, rules in self.rules_by_category.items()},
			'rules_by_dimension': {dim.value: len(rules) for dim, rules in self.rules_by_dimension.items()},
			'execution_stats': self.execution_stats.copy()
		}
		
		return stats

	def export_rules_configuration(self) -> Dict[str, Any]:
		"""Export rules configuration for backup or migration."""
		config = {
			'tenant_id': self.tenant_id,
			'export_timestamp': datetime.utcnow().isoformat(),
			'rules': []
		}
		
		for rule in self.rules_registry.values():
			rule_config = {
				'rule_id': rule.rule_id,
				'rule_name': rule.rule_name,
				'description': rule.description,
				'category': rule.category.value,
				'dimension': rule.dimension.value,
				'enabled': rule.enabled,
				'auto_correct': rule.auto_correct,
				'correction_action': rule.correction_action.value if rule.correction_action else None,
				'priority': rule.priority,
				'business_impact': rule.business_impact,
				'compliance_requirement': rule.compliance_requirement,
				'ai_enhanced': rule.ai_enhanced,
				'ai_confidence_threshold': rule.ai_confidence_threshold,
				'conditions': [
					{
						'field_name': cond.field_name,
						'operator': cond.operator.value,
						'expected_value': cond.expected_value,
						'error_message': cond.error_message,
						'severity': cond.severity.value
					}
					for cond in rule.conditions
				],
				'version': rule.version,
				'created_date': rule.created_date.isoformat(),
				'last_modified': rule.last_modified.isoformat()
			}
			config['rules'].append(rule_config)
		
		return config

	def import_rules_configuration(self, config: Dict[str, Any]) -> bool:
		"""Import rules configuration from backup or migration."""
		try:
			for rule_config in config.get('rules', []):
				# Create conditions
				conditions = []
				for cond_config in rule_config.get('conditions', []):
					condition = RuleCondition(
						field_name=cond_config['field_name'],
						operator=RuleOperator(cond_config['operator']),
						expected_value=cond_config['expected_value'],
						error_message=cond_config['error_message'],
						severity=ValidationSeverity(cond_config['severity'])
					)
					conditions.append(condition)
				
				# Create rule
				rule = QualityRule(
					rule_id=rule_config['rule_id'],
					rule_name=rule_config['rule_name'],
					description=rule_config['description'],
					category=RuleCategory(rule_config['category']),
					dimension=DataQualityDimension(rule_config['dimension']),
					conditions=conditions,
					enabled=rule_config['enabled'],
					auto_correct=rule_config['auto_correct'],
					correction_action=AutoCorrectionAction(rule_config['correction_action']) if rule_config['correction_action'] else None,
					priority=rule_config['priority'],
					business_impact=rule_config['business_impact'],
					compliance_requirement=rule_config['compliance_requirement'],
					ai_enhanced=rule_config['ai_enhanced'],
					ai_confidence_threshold=rule_config['ai_confidence_threshold'],
					version=rule_config['version'],
					created_date=datetime.fromisoformat(rule_config['created_date']),
					last_modified=datetime.fromisoformat(rule_config['last_modified'])
				)
				
				self.add_rule(rule)
			
			self.logger.info(f"Imported {len(config.get('rules', []))} quality rules")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to import rules configuration: {str(e)}")
			return False