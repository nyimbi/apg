"""
Attribute-Based Access Control (ABAC) Service

Comprehensive ABAC implementation with policy evaluation engine,
attribute resolution, and decision caching for high-performance
enterprise authorization.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from enum import Enum

from .abac_models import (
	ARAttribute, ARPolicy, ARPolicyRule, ARPolicyCondition, 
	ARPolicySet, ARPolicySetMapping, ARAccessRequest,
	PolicyEffect, PolicyAlgorithm, AttributeType
)
from .models import ARUser, ARRole, ARUserRole, ARUserSession
from .exceptions import AuthRBACError, AuthorizationError, PolicyEvaluationError

# Set up logging
logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
	"""ABAC decision types"""
	PERMIT = "permit"
	DENY = "deny"
	NOT_APPLICABLE = "not_applicable"
	INDETERMINATE = "indeterminate"


@dataclass
class AccessRequestContext:
	"""ABAC access request context structure"""
	subject: Dict[str, Any]
	resource: Dict[str, Any]
	action: Dict[str, Any]
	environment: Dict[str, Any]
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'subject': self.subject,
			'resource': self.resource,
			'action': self.action,
			'environment': self.environment
		}


@dataclass
class PolicyDecision:
	"""Policy evaluation decision result"""
	decision: DecisionType
	reason: str
	policy_id: Optional[str] = None
	policy_name: Optional[str] = None
	evaluation_time_ms: float = 0.0
	rule_evaluations: List[Dict[str, Any]] = None
	
	def __post_init__(self):
		if self.rule_evaluations is None:
			self.rule_evaluations = []


@dataclass
class AuthorizationDecision:
	"""Final authorization decision result"""
	decision: DecisionType
	reason: str
	policies_evaluated: List[PolicyDecision]
	evaluation_time_ms: float
	request_id: str
	cache_hit: bool = False
	
	def is_permitted(self) -> bool:
		"""Check if decision permits access"""
		return self.decision == DecisionType.PERMIT


class ABACService:
	"""
	Comprehensive ABAC service for policy-based authorization.
	
	Provides high-performance policy evaluation with caching,
	attribute resolution, and comprehensive audit logging.
	"""
	
	def __init__(self, db_session: Session, cache_service=None):
		self.db = db_session
		self.cache = cache_service  # Redis or similar cache
		self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
		
		# Performance configuration
		self.cache_ttl = 300  # 5 minutes default
		self.max_policy_evaluation_time = 5000  # 5 seconds max
		self.attribute_cache = {}  # In-memory attribute cache
		
	def authorize(self, subject_id: str, resource_type: str, resource_id: str, 
				 action: str, tenant_id: str, context: Dict[str, Any] = None) -> AuthorizationDecision:
		"""
		Main authorization entry point using ABAC policies.
		
		Args:
			subject_id: User/subject identifier
			resource_type: Type of resource being accessed
			resource_id: Specific resource identifier
			action: Action being performed
			tenant_id: Tenant context
			context: Additional context attributes
			
		Returns:
			AuthorizationDecision with permit/deny and reasoning
		"""
		start_time = time.time()
		request_id = self._generate_request_id()
		
		try:
			# Build request context
			request_context = self._build_request_context(
				subject_id, resource_type, resource_id, action, tenant_id, context
			)
			
			# Check cache first
			cache_key = self._generate_cache_key(request_context)
			cached_decision = self._get_cached_decision(cache_key)
			
			if cached_decision:
				cached_decision.request_id = request_id
				cached_decision.cache_hit = True
				return cached_decision
			
			# Evaluate policies
			decision = self._evaluate_policies(request_context, tenant_id)
			decision.request_id = request_id
			decision.evaluation_time_ms = (time.time() - start_time) * 1000
			
			# Cache decision
			self._cache_decision(cache_key, decision)
			
			# Log access request
			self._log_access_request(request_context, decision, tenant_id)
			
			return decision
			
		except Exception as e:
			self.logger.error(f"Authorization failed for subject {subject_id}: {str(e)}")
			
			# Return secure deny decision on error
			decision = AuthorizationDecision(
				decision=DecisionType.DENY,
				reason=f"Authorization error: {str(e)}",
				policies_evaluated=[],
				evaluation_time_ms=(time.time() - start_time) * 1000,
				request_id=request_id
			)
			
			# Still log the request for audit
			try:
				if 'request_context' in locals():
					self._log_access_request(request_context, decision, tenant_id)
			except:
				pass
			
			return decision
	
	def bulk_authorize(self, requests: List[Dict[str, Any]], tenant_id: str) -> List[AuthorizationDecision]:
		"""
		Bulk authorization for multiple requests with optimized evaluation.
		
		Args:
			requests: List of authorization requests
			tenant_id: Tenant context
			
		Returns:
			List of AuthorizationDecision objects
		"""
		decisions = []
		
		# Pre-load common policies and attributes for efficiency
		self._preload_tenant_policies(tenant_id)
		
		for request in requests:
			decision = self.authorize(
				subject_id=request['subject_id'],
				resource_type=request['resource_type'],
				resource_id=request['resource_id'],
				action=request['action'],
				tenant_id=tenant_id,
				context=request.get('context', {})
			)
			decisions.append(decision)
		
		return decisions
	
	def evaluate_policy(self, policy_id: str, request_context: AccessRequestContext) -> PolicyDecision:
		"""
		Evaluate a specific policy against request context.
		
		Args:
			policy_id: Policy to evaluate
			request_context: Request context with attributes
			
		Returns:
			PolicyDecision with evaluation result
		"""
		start_time = time.time()
		
		try:
			# Get policy
			policy = self.db.query(ARPolicy).filter(
				ARPolicy.policy_id == policy_id
			).first()
			
			if not policy:
				return PolicyDecision(
					decision=DecisionType.NOT_APPLICABLE,
					reason=f"Policy {policy_id} not found"
				)
			
			if not policy.is_effective():
				return PolicyDecision(
					decision=DecisionType.NOT_APPLICABLE,
					reason=f"Policy {policy.name} is not currently effective"
				)
			
			# Check if policy target matches request
			if not policy.matches_target(request_context.to_dict()):
				return PolicyDecision(
					decision=DecisionType.NOT_APPLICABLE,
					reason=f"Policy {policy.name} target does not match request",
					policy_id=policy_id,
					policy_name=policy.name
				)
			
			# Evaluate policy rules
			rule_results = []
			for rule in policy.rules:
				if rule.is_active:
					rule_result, rule_details = rule.evaluate(request_context.to_dict())
					rule_results.append({
						'rule_id': rule.rule_id,
						'rule_name': rule.name,
						'result': rule_result,
						'details': rule_details
					})
			
			# Apply policy algorithm to combine rule results
			final_decision = self._apply_policy_algorithm(
				policy.algorithm, rule_results, policy.effect
			)
			
			# Update policy statistics
			evaluation_time_ms = (time.time() - start_time) * 1000
			policy.update_evaluation_stats(evaluation_time_ms, final_decision.value)
			
			return PolicyDecision(
				decision=final_decision,
				reason=f"Policy {policy.name} evaluated to {final_decision.value}",
				policy_id=policy_id,
				policy_name=policy.name,
				evaluation_time_ms=evaluation_time_ms,
				rule_evaluations=rule_results
			)
			
		except Exception as e:
			self.logger.error(f"Policy evaluation failed for {policy_id}: {str(e)}")
			return PolicyDecision(
				decision=DecisionType.INDETERMINATE,
				reason=f"Policy evaluation error: {str(e)}",
				policy_id=policy_id
			)
	
	def create_policy(self, policy_data: Dict[str, Any], tenant_id: str) -> str:
		"""
		Create a new ABAC policy.
		
		Args:
			policy_data: Policy configuration data
			tenant_id: Tenant context
			
		Returns:
			Policy ID of created policy
		"""
		try:
			# Create policy
			policy = ARPolicy(
				tenant_id=tenant_id,
				name=policy_data['name'],
				display_name=policy_data.get('display_name'),
				description=policy_data.get('description'),
				version=policy_data.get('version', '1.0.0'),
				effect=policy_data.get('effect', PolicyEffect.PERMIT.value),
				algorithm=policy_data.get('algorithm', PolicyAlgorithm.DENY_OVERRIDES.value),
				priority=policy_data.get('priority', 100),
				target_subjects=policy_data.get('target_subjects', []),
				target_resources=policy_data.get('target_resources', []),
				target_actions=policy_data.get('target_actions', []),
				target_environments=policy_data.get('target_environments', []),
				effective_from=policy_data.get('effective_from'),
				effective_until=policy_data.get('effective_until'),
				tags=policy_data.get('tags', []),
				compliance_frameworks=policy_data.get('compliance_frameworks', []),
				risk_level=policy_data.get('risk_level', 'medium')
			)
			
			self.db.add(policy)
			self.db.flush()  # Get policy ID
			
			# Create rules if provided
			rules_data = policy_data.get('rules', [])
			for rule_data in rules_data:
				self._create_policy_rule(policy.policy_id, rule_data, tenant_id)
			
			self.db.commit()
			
			# Clear relevant caches
			self._invalidate_policy_cache(tenant_id)
			
			self.logger.info(f"Created ABAC policy {policy.name} with ID {policy.policy_id}")
			return policy.policy_id
			
		except Exception as e:
			self.db.rollback()
			self.logger.error(f"Failed to create policy: {str(e)}")
			raise PolicyEvaluationError(f"Policy creation failed: {str(e)}")
	
	def create_attribute(self, attribute_data: Dict[str, Any], tenant_id: str) -> str:
		"""
		Create a new ABAC attribute definition.
		
		Args:
			attribute_data: Attribute configuration data
			tenant_id: Tenant context
			
		Returns:
			Attribute ID of created attribute
		"""
		try:
			attribute = ARAttribute(
				tenant_id=tenant_id,
				category=attribute_data['category'],
				name=attribute_data['name'],
				display_name=attribute_data['display_name'],
				description=attribute_data.get('description'),
				data_type=attribute_data['data_type'],
				is_required=attribute_data.get('is_required', False),
				is_multi_valued=attribute_data.get('is_multi_valued', False),
				default_value=attribute_data.get('default_value'),
				validation_rules=attribute_data.get('validation_rules', {}),
				allowed_values=attribute_data.get('allowed_values', []),
				source_type=attribute_data.get('source_type', 'static'),
				source_config=attribute_data.get('source_config', {}),
				cache_duration=attribute_data.get('cache_duration', 300),
				is_pii=attribute_data.get('is_pii', False),
				sensitivity_level=attribute_data.get('sensitivity_level', 'public')
			)
			
			self.db.add(attribute)
			self.db.commit()
			
			# Clear attribute cache
			self._invalidate_attribute_cache(tenant_id)
			
			self.logger.info(f"Created ABAC attribute {attribute.name} with ID {attribute.attribute_id}")
			return attribute.attribute_id
			
		except Exception as e:
			self.db.rollback()
			self.logger.error(f"Failed to create attribute: {str(e)}")
			raise PolicyEvaluationError(f"Attribute creation failed: {str(e)}")
	
	def get_user_attributes(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
		"""
		Get subject attributes for a user.
		
		Args:
			user_id: User identifier
			tenant_id: Tenant context
			
		Returns:
			Dictionary of user attributes
		"""
		try:
			# Get user with roles and sessions
			user = self.db.query(ARUser).filter(
				and_(
					ARUser.user_id == user_id,
					ARUser.tenant_id == tenant_id
				)
			).first()
			
			if not user:
				return {}
			
			# Build base user attributes
			attributes = {
				'user_id': user.user_id,
				'email': user.email,
				'username': user.username,
				'account_type': user.account_type,
				'security_level': user.security_level,
				'is_active': user.is_active,
				'is_verified': user.is_verified,
				'email_verified': user.email_verified,
				'mfa_enabled': user.mfa_enabled,
				'require_mfa': user.require_mfa,
				'failed_login_attempts': user.failed_login_attempts,
				'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None,
				'created_on': user.created_on.isoformat() if user.created_on else None
			}
			
			# Add role attributes
			active_roles = []
			role_names = []
			role_priorities = []
			
			for user_role in user.roles:
				if user_role.is_active() and user_role.role:
					role_info = {
						'role_id': user_role.role.role_id,
						'role_name': user_role.role.name,
						'role_display_name': user_role.role.display_name,
						'hierarchy_level': user_role.role.hierarchy_level,
						'priority': user_role.role.priority,
						'assigned_at': user_role.assigned_at.isoformat()
					}
					active_roles.append(role_info)
					role_names.append(user_role.role.name)
					role_priorities.append(user_role.role.priority)
			
			attributes.update({
				'roles': active_roles,
				'role_names': role_names,
				'role_count': len(active_roles),
				'highest_role_priority': max(role_priorities) if role_priorities else 0,
				'lowest_role_priority': min(role_priorities) if role_priorities else 0
			})
			
			# Add session attributes if available
			active_session = self.db.query(ARUserSession).filter(
				and_(
					ARUserSession.user_id == user_id,
					ARUserSession.logout_at.is_(None),
					ARUserSession.expires_at > datetime.utcnow()
				)
			).first()
			
			if active_session:
				attributes.update({
					'session_id': active_session.session_id,
					'login_method': active_session.login_method,
					'device_type': active_session.device_type,
					'ip_address': active_session.ip_address,
					'ip_country': active_session.ip_country,
					'ip_city': active_session.ip_city,
					'is_trusted_device': active_session.is_trusted_device,
					'anomaly_score': active_session.anomaly_score,
					'session_duration_minutes': int((datetime.utcnow() - active_session.login_at).total_seconds() / 60)
				})
			
			# Add computed attributes
			attributes.update({
				'account_age_days': (datetime.utcnow() - user.created_on).days if user.created_on else 0,
				'days_since_last_login': (datetime.utcnow() - user.last_login_at).days if user.last_login_at else 999999,
				'password_age_days': (datetime.utcnow() - user.password_changed_at).days if user.password_changed_at else 999999,
				'is_account_locked': user.is_account_locked(),
				'requires_password_change': user.require_password_change
			})
			
			return attributes
			
		except Exception as e:
			self.logger.error(f"Failed to get user attributes for {user_id}: {str(e)}")
			return {}
	
	def get_resource_attributes(self, resource_type: str, resource_id: str, 
							  tenant_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""
		Get resource attributes for authorization.
		
		Args:
			resource_type: Type of resource
			resource_id: Resource identifier
			tenant_id: Tenant context
			context: Additional context
			
		Returns:
			Dictionary of resource attributes
		"""
		try:
			attributes = {
				'resource_type': resource_type,
				'resource_id': resource_id,
				'tenant_id': tenant_id
			}
			
			# Add context attributes
			if context:
				attributes.update(context)
			
			# Resource-specific attribute resolution
			if resource_type == 'profile_management':
				# Get profile-specific attributes
				attributes.update(self._get_profile_resource_attributes(resource_id, tenant_id))
			elif resource_type == 'capability':
				# Get capability-specific attributes
				attributes.update(self._get_capability_resource_attributes(resource_id, tenant_id))
			elif resource_type == 'api_endpoint':
				# Get API endpoint attributes
				attributes.update(self._get_api_resource_attributes(resource_id, tenant_id))
			
			# Add common computed attributes
			attributes.update({
				'current_time': datetime.utcnow().isoformat(),
				'current_hour': datetime.utcnow().hour,
				'current_day_of_week': datetime.utcnow().weekday(),
				'is_business_hours': 9 <= datetime.utcnow().hour <= 17,
				'is_weekend': datetime.utcnow().weekday() >= 5
			})
			
			return attributes
			
		except Exception as e:
			self.logger.error(f"Failed to get resource attributes for {resource_type}:{resource_id}: {str(e)}")
			return {
				'resource_type': resource_type,
				'resource_id': resource_id,
				'tenant_id': tenant_id
			}
	
	def get_environment_attributes(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
		"""
		Get environment attributes for authorization context.
		
		Args:
			context: Additional context information
			
		Returns:
			Dictionary of environment attributes
		"""
		now = datetime.utcnow()
		
		attributes = {
			'current_time': now.isoformat(),
			'current_timestamp': int(now.timestamp()),
			'current_hour': now.hour,
			'current_minute': now.minute,
			'current_day_of_week': now.weekday(),
			'current_day_of_month': now.day,
			'current_month': now.month,
			'current_year': now.year,
			'is_business_hours': 9 <= now.hour <= 17,
			'is_weekend': now.weekday() >= 5,
			'is_holiday': False,  # Could integrate with holiday calendar
			'timezone': 'UTC'
		}
		
		# Add context attributes
		if context:
			attributes.update(context)
		
		return attributes
	
	def _build_request_context(self, subject_id: str, resource_type: str, resource_id: str,
							  action: str, tenant_id: str, context: Dict[str, Any] = None) -> AccessRequestContext:
		"""Build complete request context for policy evaluation"""
		
		# Get subject attributes
		subject_attributes = self.get_user_attributes(subject_id, tenant_id)
		
		# Get resource attributes
		resource_context = context.get('resource', {}) if context else {}
		resource_attributes = self.get_resource_attributes(resource_type, resource_id, tenant_id, resource_context)
		
		# Get action attributes
		action_attributes = {
			'action': action,
			'action_type': action,
			'requested_at': datetime.utcnow().isoformat()
		}
		
		if context and 'action' in context:
			action_attributes.update(context['action'])
		
		# Get environment attributes
		environment_context = context.get('environment', {}) if context else {}
		environment_attributes = self.get_environment_attributes(environment_context)
		
		return AccessRequestContext(
			subject=subject_attributes,
			resource=resource_attributes,
			action=action_attributes,
			environment=environment_attributes
		)
	
	def _evaluate_policies(self, request_context: AccessRequestContext, tenant_id: str) -> AuthorizationDecision:
		"""Evaluate all applicable policies for the request"""
		start_time = time.time()
		
		try:
			# Get applicable policies
			policies = self._get_applicable_policies(request_context, tenant_id)
			
			if not policies:
				return AuthorizationDecision(
					decision=DecisionType.DENY,
					reason="No applicable policies found",
					policies_evaluated=[],
					evaluation_time_ms=0
				)
			
			# Evaluate each policy
			policy_decisions = []
			for policy in policies:
				decision = self.evaluate_policy(policy.policy_id, request_context)
				policy_decisions.append(decision)
			
			# Apply policy combination algorithm
			final_decision = self._combine_policy_decisions(policy_decisions)
			
			evaluation_time_ms = (time.time() - start_time) * 1000
			
			return AuthorizationDecision(
				decision=final_decision.decision,
				reason=final_decision.reason,
				policies_evaluated=policy_decisions,
				evaluation_time_ms=evaluation_time_ms
			)
			
		except Exception as e:
			self.logger.error(f"Policy evaluation failed: {str(e)}")
			return AuthorizationDecision(
				decision=DecisionType.DENY,
				reason=f"Policy evaluation error: {str(e)}",
				policies_evaluated=[],
				evaluation_time_ms=(time.time() - start_time) * 1000
			)
	
	def _get_applicable_policies(self, request_context: AccessRequestContext, tenant_id: str) -> List[ARPolicy]:
		"""Get policies that might apply to the request"""
		
		# Get all active policies for tenant
		policies = self.db.query(ARPolicy).filter(
			and_(
				ARPolicy.tenant_id == tenant_id,
				ARPolicy.is_active == True,
				or_(
					ARPolicy.effective_from.is_(None),
					ARPolicy.effective_from <= datetime.utcnow()
				),
				or_(
					ARPolicy.effective_until.is_(None),
					ARPolicy.effective_until > datetime.utcnow()
				)
			)
		).order_by(ARPolicy.priority.desc()).all()
		
		# Filter policies that match the request target
		applicable_policies = []
		for policy in policies:
			if policy.matches_target(request_context.to_dict()):
				applicable_policies.append(policy)
		
		return applicable_policies
	
	def _apply_policy_algorithm(self, algorithm: str, rule_results: List[Dict], policy_effect: str) -> DecisionType:
		"""Apply policy combination algorithm to rule results"""
		
		if not rule_results:
			return DecisionType.NOT_APPLICABLE
		
		rule_decisions = [r['result'] for r in rule_results]
		
		if algorithm == PolicyAlgorithm.PERMIT_OVERRIDES.value:
			# If any rule permits, the policy permits
			if any(rule_decisions):
				return DecisionType.PERMIT if policy_effect == PolicyEffect.PERMIT.value else DecisionType.DENY
			else:
				return DecisionType.DENY if policy_effect == PolicyEffect.PERMIT.value else DecisionType.PERMIT
		
		elif algorithm == PolicyAlgorithm.DENY_OVERRIDES.value:
			# If any rule denies, the policy denies
			if not all(rule_decisions):
				return DecisionType.DENY if policy_effect == PolicyEffect.PERMIT.value else DecisionType.PERMIT
			else:
				return DecisionType.PERMIT if policy_effect == PolicyEffect.PERMIT.value else DecisionType.DENY
		
		elif algorithm == PolicyAlgorithm.FIRST_APPLICABLE.value:
			# Use the first applicable rule result
			if rule_decisions:
				first_result = rule_decisions[0]
				return DecisionType.PERMIT if (first_result and policy_effect == PolicyEffect.PERMIT.value) else DecisionType.DENY
		
		elif algorithm == PolicyAlgorithm.ONLY_ONE_APPLICABLE.value:
			# Only one rule should be applicable
			true_count = sum(rule_decisions)
			if true_count == 1:
				return DecisionType.PERMIT if policy_effect == PolicyEffect.PERMIT.value else DecisionType.DENY
			else:
				return DecisionType.INDETERMINATE
		
		elif algorithm == PolicyAlgorithm.PERMIT_UNLESS_DENY.value:
			# Permit unless explicitly denied
			if all(rule_decisions):
				return DecisionType.PERMIT if policy_effect == PolicyEffect.PERMIT.value else DecisionType.DENY
			else:
				return DecisionType.DENY if policy_effect == PolicyEffect.PERMIT.value else DecisionType.PERMIT
		
		elif algorithm == PolicyAlgorithm.DENY_UNLESS_PERMIT.value:
			# Deny unless explicitly permitted
			if any(rule_decisions):
				return DecisionType.PERMIT if policy_effect == PolicyEffect.PERMIT.value else DecisionType.DENY
			else:
				return DecisionType.DENY if policy_effect == PolicyEffect.PERMIT.value else DecisionType.PERMIT
		
		return DecisionType.NOT_APPLICABLE
	
	def _combine_policy_decisions(self, policy_decisions: List[PolicyDecision]) -> PolicyDecision:
		"""Combine multiple policy decisions using deny-overrides algorithm"""
		
		if not policy_decisions:
			return PolicyDecision(
				decision=DecisionType.DENY,
				reason="No policies evaluated"
			)
		
		# Apply deny-overrides: if any policy denies, final decision is deny
		for decision in policy_decisions:
			if decision.decision == DecisionType.DENY:
				return PolicyDecision(
					decision=DecisionType.DENY,
					reason=f"Access denied by policy: {decision.reason}"
				)
		
		# Check for permits
		permit_decisions = [d for d in policy_decisions if d.decision == DecisionType.PERMIT]
		if permit_decisions:
			return PolicyDecision(
				decision=DecisionType.PERMIT,
				reason=f"Access permitted by {len(permit_decisions)} policy(ies)"
			)
		
		# Check for indeterminate
		indeterminate_decisions = [d for d in policy_decisions if d.decision == DecisionType.INDETERMINATE]
		if indeterminate_decisions:
			return PolicyDecision(
				decision=DecisionType.INDETERMINATE,
				reason="Policy evaluation was indeterminate"
			)
		
		# All policies were not applicable
		return PolicyDecision(
			decision=DecisionType.DENY,
			reason="No applicable policies found"
		)
	
	def _create_policy_rule(self, policy_id: str, rule_data: Dict[str, Any], tenant_id: str) -> str:
		"""Create a policy rule with conditions"""
		
		rule = ARPolicyRule(
			policy_id=policy_id,
			tenant_id=tenant_id,
			name=rule_data['name'],
			description=rule_data.get('description'),
			rule_order=rule_data.get('rule_order', 0),
			rule_expression=rule_data.get('rule_expression', ''),
			combination_logic=rule_data.get('combination_logic', 'AND'),
			effect=rule_data.get('effect', PolicyEffect.PERMIT.value)
		)
		
		self.db.add(rule)
		self.db.flush()
		
		# Create conditions
		conditions_data = rule_data.get('conditions', [])
		for condition_data in conditions_data:
			condition = ARPolicyCondition(
				rule_id=rule.rule_id,
				policy_id=policy_id,
				attribute_id=condition_data['attribute_id'],
				tenant_id=tenant_id,
				name=condition_data['name'],
				description=condition_data.get('description'),
				operator=condition_data['operator'],
				expected_value=condition_data.get('expected_value'),
				function_name=condition_data.get('function_name'),
				function_params=condition_data.get('function_params', {}),
				negate_result=condition_data.get('negate_result', False)
			)
			self.db.add(condition)
		
		return rule.rule_id
	
	def _get_profile_resource_attributes(self, resource_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Get profile-specific resource attributes"""
		# This would integrate with the profile management capability
		return {
			'resource_category': 'profile',
			'is_personal_data': True,
			'data_classification': 'confidential',
			'requires_consent': True
		}
	
	def _get_capability_resource_attributes(self, resource_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Get capability-specific resource attributes"""
		return {
			'resource_category': 'capability',
			'is_system_resource': True,
			'data_classification': 'internal',
			'requires_elevated_access': True
		}
	
	def _get_api_resource_attributes(self, resource_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Get API endpoint resource attributes"""
		return {
			'resource_category': 'api',
			'is_public_api': False,
			'rate_limited': True,
			'requires_authentication': True
		}
	
	def _generate_request_id(self) -> str:
		"""Generate unique request ID"""
		import uuid
		return str(uuid.uuid4())
	
	def _generate_cache_key(self, request_context: AccessRequestContext) -> str:
		"""Generate cache key for request context"""
		import hashlib
		
		# Create a stable hash of the request context
		context_str = json.dumps(request_context.to_dict(), sort_keys=True)
		return f"abac:{hashlib.sha256(context_str.encode()).hexdigest()}"
	
	def _get_cached_decision(self, cache_key: str) -> Optional[AuthorizationDecision]:
		"""Get cached authorization decision"""
		if not self.cache:
			return None
		
		try:
			cached_data = self.cache.get(cache_key)
			if cached_data:
				# Deserialize cached decision
				return AuthorizationDecision(**json.loads(cached_data))
		except Exception as e:
			self.logger.warning(f"Cache retrieval failed: {str(e)}")
		
		return None
	
	def _cache_decision(self, cache_key: str, decision: AuthorizationDecision) -> None:
		"""Cache authorization decision"""
		if not self.cache:
			return
		
		try:
			# Only cache permit decisions to avoid security issues
			if decision.decision == DecisionType.PERMIT:
				cached_data = {
					'decision': decision.decision.value,
					'reason': decision.reason,
					'policies_evaluated': [
						{
							'decision': p.decision.value,
							'reason': p.reason,
							'policy_id': p.policy_id,
							'policy_name': p.policy_name
						} for p in decision.policies_evaluated
					],
					'evaluation_time_ms': decision.evaluation_time_ms,
					'request_id': decision.request_id
				}
				
				self.cache.setex(cache_key, self.cache_ttl, json.dumps(cached_data))
		except Exception as e:
			self.logger.warning(f"Cache storage failed: {str(e)}")
	
	def _log_access_request(self, request_context: AccessRequestContext, 
						   decision: AuthorizationDecision, tenant_id: str) -> None:
		"""Log access request for audit and analysis"""
		try:
			access_request = ARAccessRequest(
				request_id=decision.request_id,
				tenant_id=tenant_id,
				subject_id=request_context.subject.get('user_id', 'unknown'),
				resource_type=request_context.resource.get('resource_type', 'unknown'),
				resource_id=request_context.resource.get('resource_id', 'unknown'),
				action=request_context.action.get('action', 'unknown'),
				subject_attributes=request_context.subject,
				resource_attributes=request_context.resource,
				action_attributes=request_context.action,
				environment_attributes=request_context.environment,
				decision=decision.decision.value,
				decision_reason=decision.reason,
				policies_evaluated=[
					{
						'policy_id': p.policy_id,
						'policy_name': p.policy_name,
						'decision': p.decision.value,
						'evaluation_time_ms': p.evaluation_time_ms
					} for p in decision.policies_evaluated
				],
				evaluation_time_ms=decision.evaluation_time_ms,
				session_id=request_context.subject.get('session_id'),
				ip_address=request_context.subject.get('ip_address'),
				user_agent=request_context.environment.get('user_agent')
			)
			
			self.db.add(access_request)
			self.db.commit()
			
		except Exception as e:
			self.logger.error(f"Failed to log access request: {str(e)}")
			# Don't fail authorization due to logging issues
			self.db.rollback()
	
	def _preload_tenant_policies(self, tenant_id: str) -> None:
		"""Preload tenant policies for bulk operations"""
		# This would implement policy preloading and caching
		pass
	
	def _invalidate_policy_cache(self, tenant_id: str) -> None:
		"""Invalidate policy-related caches"""
		if self.cache:
			try:
				# This would implement cache invalidation logic
				cache_pattern = f"abac:policy:{tenant_id}:*"
				# Implementation depends on cache backend
			except Exception as e:
				self.logger.warning(f"Cache invalidation failed: {str(e)}")
	
	def _invalidate_attribute_cache(self, tenant_id: str) -> None:
		"""Invalidate attribute-related caches"""
		if self.cache:
			try:
				# Clear in-memory attribute cache
				self.attribute_cache.clear()
				
				# This would implement distributed cache invalidation
				cache_pattern = f"abac:attr:{tenant_id}:*"
				# Implementation depends on cache backend
			except Exception as e:
				self.logger.warning(f"Attribute cache invalidation failed: {str(e)}")


# Capability composition functions
def get_abac_service(db_session: Session, cache_service=None) -> ABACService:
	"""Capability composition function to get ABAC service"""
	return ABACService(db_session, cache_service)


def create_default_attributes(tenant_id: str, db_session: Session) -> None:
	"""Create default ABAC attributes for a tenant"""
	
	abac_service = ABACService(db_session)
	
	# Default subject attributes
	subject_attributes = [
		{
			'category': 'subject',
			'name': 'user_id',
			'display_name': 'User ID',
			'description': 'Unique user identifier',
			'data_type': 'string',
			'is_required': True,
			'source_type': 'static'
		},
		{
			'category': 'subject',
			'name': 'role_names',
			'display_name': 'Role Names',
			'description': 'List of user role names',
			'data_type': 'list',
			'is_multi_valued': True,
			'source_type': 'dynamic'
		},
		{
			'category': 'subject',
			'name': 'security_level',
			'display_name': 'Security Level',
			'description': 'User security clearance level',
			'data_type': 'string',
			'allowed_values': ['basic', 'standard', 'high', 'critical'],
			'source_type': 'static'
		}
	]
	
	# Default resource attributes
	resource_attributes = [
		{
			'category': 'resource',
			'name': 'resource_type',
			'display_name': 'Resource Type',
			'description': 'Type of resource being accessed',
			'data_type': 'string',
			'is_required': True,
			'source_type': 'static'
		},
		{
			'category': 'resource',
			'name': 'data_classification',
			'display_name': 'Data Classification',
			'description': 'Data sensitivity classification',
			'data_type': 'string',
			'allowed_values': ['public', 'internal', 'confidential', 'restricted'],
			'source_type': 'static'
		}
	]
	
	# Default action attributes
	action_attributes = [
		{
			'category': 'action',
			'name': 'action',
			'display_name': 'Action',
			'description': 'Action being performed',
			'data_type': 'string',
			'is_required': True,
			'allowed_values': ['create', 'read', 'update', 'delete', 'execute', 'admin'],
			'source_type': 'static'
		}
	]
	
	# Default environment attributes
	environment_attributes = [
		{
			'category': 'environment',
			'name': 'current_time',
			'display_name': 'Current Time',
			'description': 'Current timestamp',
			'data_type': 'datetime',
			'source_type': 'computed'
		},
		{
			'category': 'environment',
			'name': 'is_business_hours',
			'display_name': 'Business Hours',
			'description': 'Whether current time is within business hours',
			'data_type': 'boolean',
			'source_type': 'computed'
		}
	]
	
	# Create all default attributes
	all_attributes = subject_attributes + resource_attributes + action_attributes + environment_attributes
	
	for attr_data in all_attributes:
		try:
			abac_service.create_attribute(attr_data, tenant_id)
		except Exception as e:
			logger.warning(f"Failed to create default attribute {attr_data['name']}: {str(e)}")