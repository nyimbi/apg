"""
APG Multi-Factor Authentication (MFA) - Business Logic Orchestration

Advanced business logic orchestration handling complex MFA workflows,
policy enforcement, and intelligent decision making.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict

from .models import (
	MFAUserProfile, MFAMethod, MFAMethodType, AuthEvent,
	TrustLevel, AuthenticationStatus
)
from .integration import APGIntegrationRouter


def _log_business_operation(operation: str, user_id: str, details: str = "") -> str:
	"""Log business logic operations for debugging and audit"""
	return f"[Business Logic] {operation} for user {user_id}: {details}"


class PolicyDecisionType(str, Enum):
	"""Policy decision types"""
	ALLOW = "allow"
	DENY = "deny"
	REQUIRE_ADDITIONAL_AUTH = "require_additional_auth"
	REQUIRE_STEP_UP = "require_step_up"
	REQUIRE_ADMIN_APPROVAL = "require_admin_approval"


class WorkflowStage(str, Enum):
	"""MFA workflow stages"""
	INITIAL_ASSESSMENT = "initial_assessment"
	METHOD_SELECTION = "method_selection"
	PRIMARY_AUTHENTICATION = "primary_authentication"
	RISK_EVALUATION = "risk_evaluation"
	ADDITIONAL_VERIFICATION = "additional_verification"
	STEP_UP_AUTHENTICATION = "step_up_authentication"
	FINAL_AUTHORIZATION = "final_authorization"
	COMPLETED = "completed"


class BusinessRule(BaseModel):
	"""Business rule model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rule_name: str
	rule_type: str  # "authentication", "enrollment", "recovery"
	
	# Rule conditions
	conditions: Dict[str, Any] = {}
	risk_threshold: float = 0.5
	user_groups: List[str] = []
	time_restrictions: Dict[str, Any] = {}
	
	# Rule actions
	actions: Dict[str, Any] = {}
	required_methods: List[str] = []
	notification_settings: Dict[str, Any] = {}
	
	# Rule metadata
	priority: int = 100
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowContext(BaseModel):
	"""Workflow execution context"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	session_id: str = Field(default_factory=uuid7str)
	user_id: str
	tenant_id: str
	
	# Workflow state
	current_stage: WorkflowStage = WorkflowStage.INITIAL_ASSESSMENT
	completed_stages: List[WorkflowStage] = []
	
	# Authentication data
	provided_methods: List[Dict[str, Any]] = []
	verified_methods: List[str] = []
	failed_methods: List[str] = []
	
	# Risk and trust
	risk_score: float = 0.0
	trust_score: float = 0.0
	risk_factors: List[str] = []
	
	# Decision tracking
	policy_decisions: List[Dict[str, Any]] = []
	workflow_decisions: List[Dict[str, Any]] = []
	
	# Context information
	request_context: Dict[str, Any] = {}
	business_context: Dict[str, Any] = {}
	
	# Timestamps
	started_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class BusinessLogicOrchestrator:
	"""
	Advanced business logic orchestrator handling complex MFA workflows,
	policy enforcement, and intelligent decision making for enterprise scenarios.
	"""
	
	def __init__(self,
				 database_client: Any,
				 integration_router: APGIntegrationRouter):
		"""Initialize business logic orchestrator"""
		self.db = database_client
		self.integration = integration_router
		self.logger = logging.getLogger(__name__)
		
		# Business rule cache
		self._rule_cache = {}
		self._cache_expiry = {}
		
		# Workflow tracking
		self._active_workflows = {}
		
		# Default policies
		self.default_risk_threshold = 0.6
		self.max_workflow_duration_hours = 2
		self.step_up_validity_minutes = 15

	async def evaluate_authentication_policy(self,
											  user_id: str,
											  tenant_id: str,
											  context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Evaluate authentication policies for user and context.
		
		Args:
			user_id: User attempting authentication
			tenant_id: Tenant context
			context: Request context and user data
		
		Returns:
			Policy evaluation result with decisions and requirements
		"""
		try:
			self.logger.info(_log_business_operation("evaluate_auth_policy", user_id))
			
			# Get applicable business rules
			rules = await self._get_applicable_rules(tenant_id, "authentication", context)
			
			# Initialize policy evaluation context
			evaluation_context = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"timestamp": datetime.utcnow().isoformat(),
				"request_context": context,
				"applicable_rules": [rule.rule_name for rule in rules]
			}
			
			# Evaluate each rule
			policy_decisions = []
			overall_decision = PolicyDecisionType.ALLOW
			required_methods = []
			additional_requirements = []
			
			for rule in rules:
				decision = await self._evaluate_rule(rule, evaluation_context)
				policy_decisions.append(decision)
				
				# Aggregate decisions (most restrictive wins)
				if decision["decision"] == PolicyDecisionType.DENY:
					overall_decision = PolicyDecisionType.DENY
				elif decision["decision"] == PolicyDecisionType.REQUIRE_ADMIN_APPROVAL:
					overall_decision = PolicyDecisionType.REQUIRE_ADMIN_APPROVAL
				elif decision["decision"] == PolicyDecisionType.REQUIRE_STEP_UP and overall_decision == PolicyDecisionType.ALLOW:
					overall_decision = PolicyDecisionType.REQUIRE_STEP_UP
				elif decision["decision"] == PolicyDecisionType.REQUIRE_ADDITIONAL_AUTH and overall_decision == PolicyDecisionType.ALLOW:
					overall_decision = PolicyDecisionType.REQUIRE_ADDITIONAL_AUTH
				
				# Collect requirements
				if "required_methods" in decision:
					required_methods.extend(decision["required_methods"])
				if "additional_requirements" in decision:
					additional_requirements.extend(decision["additional_requirements"])
			
			return {
				"decision": overall_decision,
				"required_methods": list(set(required_methods)),
				"additional_requirements": additional_requirements,
				"policy_decisions": policy_decisions,
				"evaluation_context": evaluation_context
			}
			
		except Exception as e:
			self.logger.error(f"Policy evaluation error for user {user_id}: {str(e)}", exc_info=True)
			return {
				"decision": PolicyDecisionType.DENY,
				"reason": "policy_evaluation_error",
				"error": str(e)
			}

	async def orchestrate_authentication_workflow(self,
												  user_id: str,
												  tenant_id: str,
												  provided_methods: List[Dict[str, Any]],
												  context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Orchestrate complex authentication workflow with business logic.
		
		Args:
			user_id: User authenticating
			tenant_id: Tenant context
			provided_methods: Authentication methods provided
			context: Request context
		
		Returns:
			Workflow execution result
		"""
		try:
			self.logger.info(_log_business_operation("orchestrate_workflow", user_id))
			
			# Create or resume workflow context
			workflow_context = await self._get_or_create_workflow_context(
				user_id, tenant_id, provided_methods, context
			)
			
			# Execute workflow stages
			while workflow_context.current_stage != WorkflowStage.COMPLETED:
				stage_result = await self._execute_workflow_stage(workflow_context)
				
				if not stage_result["success"]:
					return stage_result
				
				# Update workflow context
				workflow_context = stage_result["workflow_context"]
				
				# Check for workflow completion or failure
				if stage_result.get("workflow_complete"):
					break
				
				# Prevent infinite loops
				if len(workflow_context.completed_stages) > 10:
					return {
						"success": False,
						"error": "workflow_too_long",
						"message": "Workflow exceeded maximum stages"
					}
			
			# Finalize workflow
			return await self._finalize_workflow(workflow_context)
			
		except Exception as e:
			self.logger.error(f"Workflow orchestration error for user {user_id}: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "workflow_orchestration_error",
				"message": str(e)
			}

	async def evaluate_enrollment_policy(self,
										 user_id: str,
										 tenant_id: str,
										 method_type: MFAMethodType,
										 context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Evaluate enrollment policies for MFA method.
		
		Args:
			user_id: User enrolling method
			tenant_id: Tenant context
			method_type: Type of method being enrolled
			context: Request context
		
		Returns:
			Enrollment policy decision
		"""
		try:
			self.logger.info(_log_business_operation("evaluate_enrollment_policy", user_id, f"method={method_type}"))
			
			# Get enrollment rules
			rules = await self._get_applicable_rules(tenant_id, "enrollment", context)
			
			# Check method-specific policies
			method_policies = await self._get_method_policies(tenant_id, method_type)
			
			# Evaluate enrollment constraints
			enrollment_evaluation = {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"method_type": method_type.value,
				"context": context,
				"existing_methods": await self._get_user_method_types(user_id, tenant_id)
			}
			
			# Apply business rules
			for rule in rules:
				rule_result = await self._evaluate_rule(rule, enrollment_evaluation)
				if rule_result["decision"] == PolicyDecisionType.DENY:
					return {
						"allowed": False,
						"reason": rule_result.get("reason", "policy_violation"),
						"rule": rule.rule_name
					}
			
			# Check method limits and constraints
			constraints_check = await self._check_enrollment_constraints(enrollment_evaluation)
			if not constraints_check["allowed"]:
				return constraints_check
			
			return {
				"allowed": True,
				"requirements": method_policies.get("requirements", []),
				"verification_steps": method_policies.get("verification_steps", [])
			}
			
		except Exception as e:
			self.logger.error(f"Enrollment policy evaluation error: {str(e)}", exc_info=True)
			return {
				"allowed": False,
				"reason": "policy_evaluation_error",
				"error": str(e)
			}

	async def determine_risk_based_requirements(self,
												user_id: str,
												tenant_id: str,
												risk_assessment: Dict[str, Any],
												context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Determine authentication requirements based on risk assessment.
		
		Args:
			user_id: User ID
			tenant_id: Tenant context
			risk_assessment: Risk assessment results
			context: Request context
		
		Returns:
			Risk-based authentication requirements
		"""
		try:
			risk_score = risk_assessment.get("risk_score", 0.0)
			risk_factors = risk_assessment.get("risk_factors", [])
			
			self.logger.info(_log_business_operation(
				"determine_risk_requirements", user_id, f"risk_score={risk_score:.2f}"
			))
			
			# Get user's current methods
			user_methods = await self._get_user_method_types(user_id, tenant_id)
			
			# Determine base requirements
			requirements = {
				"minimum_methods": 1,
				"required_method_types": [],
				"forbidden_method_types": [],
				"additional_verifications": [],
				"time_constraints": {},
				"approval_required": False
			}
			
			# Apply risk-based escalation
			if risk_score >= 0.8:
				# Very high risk
				requirements.update({
					"minimum_methods": 3,
					"required_method_types": [MFAMethodType.TOTP, MFAMethodType.FACE_RECOGNITION],
					"additional_verifications": ["admin_approval", "manual_review"],
					"approval_required": True
				})
			elif risk_score >= 0.6:
				# High risk
				requirements.update({
					"minimum_methods": 2,
					"required_method_types": [MFAMethodType.TOTP],
					"additional_verifications": ["device_verification"]
				})
			elif risk_score >= 0.4:
				# Medium risk
				requirements.update({
					"minimum_methods": 2,
					"additional_verifications": ["location_verification"]
				})
			
			# Apply factor-specific requirements
			if "unusual_location" in risk_factors:
				requirements["additional_verifications"].append("location_confirmation")
			
			if "new_device" in risk_factors:
				requirements["additional_verifications"].append("device_registration")
			
			if "suspicious_pattern" in risk_factors:
				requirements["additional_verifications"].append("behavioral_verification")
			
			# Check if user has required methods
			missing_methods = []
			for required_type in requirements["required_method_types"]:
				if required_type not in user_methods:
					missing_methods.append(required_type)
			
			if missing_methods:
				requirements["enrollment_required"] = missing_methods
			
			return {
				"requirements": requirements,
				"risk_score": risk_score,
				"risk_factors": risk_factors,
				"escalation_reason": self._get_escalation_reason(risk_score, risk_factors)
			}
			
		except Exception as e:
			self.logger.error(f"Risk requirements determination error: {str(e)}", exc_info=True)
			return {
				"requirements": {"minimum_methods": 1},
				"error": "requirements_determination_failed"
			}

	async def validate_business_constraints(self,
											user_id: str,
											tenant_id: str,
											operation: str,
											operation_data: Dict[str, Any],
											context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Validate business constraints for MFA operations.
		
		Args:
			user_id: User performing operation
			tenant_id: Tenant context
			operation: Operation being performed
			operation_data: Operation-specific data
			context: Request context
		
		Returns:
			Constraint validation result
		"""
		try:
			self.logger.info(_log_business_operation("validate_constraints", user_id, f"operation={operation}"))
			
			# Get tenant constraints
			constraints = await self._get_tenant_constraints(tenant_id)
			
			# Validate operation-specific constraints
			if operation == "authentication":
				return await self._validate_auth_constraints(user_id, tenant_id, operation_data, constraints, context)
			elif operation == "enrollment":
				return await self._validate_enrollment_constraints(user_id, tenant_id, operation_data, constraints, context)
			elif operation == "recovery":
				return await self._validate_recovery_constraints(user_id, tenant_id, operation_data, constraints, context)
			else:
				return {"valid": True, "message": "No specific constraints for operation"}
			
		except Exception as e:
			self.logger.error(f"Constraint validation error: {str(e)}", exc_info=True)
			return {
				"valid": False,
				"reason": "constraint_validation_error",
				"error": str(e)
			}

	# Private helper methods

	async def _execute_workflow_stage(self, workflow_context: WorkflowContext) -> Dict[str, Any]:
		"""Execute a specific workflow stage"""
		stage = workflow_context.current_stage
		
		try:
			if stage == WorkflowStage.INITIAL_ASSESSMENT:
				return await self._stage_initial_assessment(workflow_context)
			elif stage == WorkflowStage.METHOD_SELECTION:
				return await self._stage_method_selection(workflow_context)
			elif stage == WorkflowStage.PRIMARY_AUTHENTICATION:
				return await self._stage_primary_authentication(workflow_context)
			elif stage == WorkflowStage.RISK_EVALUATION:
				return await self._stage_risk_evaluation(workflow_context)
			elif stage == WorkflowStage.ADDITIONAL_VERIFICATION:
				return await self._stage_additional_verification(workflow_context)
			elif stage == WorkflowStage.STEP_UP_AUTHENTICATION:
				return await self._stage_step_up_authentication(workflow_context)
			elif stage == WorkflowStage.FINAL_AUTHORIZATION:
				return await self._stage_final_authorization(workflow_context)
			else:
				return {
					"success": False,
					"error": "unknown_stage",
					"message": f"Unknown workflow stage: {stage}"
				}
				
		except Exception as e:
			self.logger.error(f"Workflow stage execution error: {str(e)}", exc_info=True)
			return {
				"success": False,
				"error": "stage_execution_error",
				"message": str(e)
			}

	async def _stage_initial_assessment(self, context: WorkflowContext) -> Dict[str, Any]:
		"""Execute initial assessment stage"""
		# Evaluate user profile and basic eligibility
		user_profile = await self._get_user_profile(context.user_id, context.tenant_id)
		if not user_profile or not user_profile.mfa_enabled:
			return {
				"success": False,
				"error": "mfa_not_enabled",
				"message": "MFA is not enabled for this user"
			}
		
		# Move to next stage
		context.completed_stages.append(context.current_stage)
		context.current_stage = WorkflowStage.METHOD_SELECTION
		context.updated_at = datetime.utcnow()
		
		return {
			"success": True,
			"workflow_context": context,
			"stage_result": {"user_eligible": True}
		}

	async def _stage_method_selection(self, context: WorkflowContext) -> Dict[str, Any]:
		"""Execute method selection stage"""
		# Determine required and available methods
		available_methods = await self._get_user_method_types(context.user_id, context.tenant_id)
		provided_methods = [method["type"] for method in context.provided_methods]
		
		# Check if sufficient methods provided
		if not provided_methods:
			return {
				"success": False,
				"error": "no_methods_provided",
				"available_methods": available_methods
			}
		
		# Move to authentication
		context.completed_stages.append(context.current_stage)
		context.current_stage = WorkflowStage.PRIMARY_AUTHENTICATION
		context.updated_at = datetime.utcnow()
		
		return {
			"success": True,
			"workflow_context": context,
			"stage_result": {"methods_selected": provided_methods}
		}

	async def _stage_primary_authentication(self, context: WorkflowContext) -> Dict[str, Any]:
		"""Execute primary authentication stage"""
		# This would integrate with the MFA engine for actual verification
		# For now, simulate authentication
		
		# Move to risk evaluation
		context.completed_stages.append(context.current_stage)
		context.current_stage = WorkflowStage.RISK_EVALUATION
		context.updated_at = datetime.utcnow()
		
		return {
			"success": True,
			"workflow_context": context,
			"stage_result": {"auth_successful": True}
		}

	async def _stage_risk_evaluation(self, context: WorkflowContext) -> Dict[str, Any]:
		"""Execute risk evaluation stage"""
		# Evaluate risk and determine if additional verification needed
		if context.risk_score > 0.6:
			context.current_stage = WorkflowStage.ADDITIONAL_VERIFICATION
		else:
			context.current_stage = WorkflowStage.FINAL_AUTHORIZATION
		
		context.completed_stages.append(WorkflowStage.RISK_EVALUATION)
		context.updated_at = datetime.utcnow()
		
		return {
			"success": True,
			"workflow_context": context,
			"stage_result": {"risk_evaluated": True}
		}

	async def _stage_final_authorization(self, context: WorkflowContext) -> Dict[str, Any]:
		"""Execute final authorization stage"""
		context.completed_stages.append(context.current_stage)
		context.current_stage = WorkflowStage.COMPLETED
		context.updated_at = datetime.utcnow()
		
		return {
			"success": True,
			"workflow_context": context,
			"workflow_complete": True,
			"stage_result": {"authorized": True}
		}

	async def _get_applicable_rules(self, tenant_id: str, rule_type: str, context: Dict[str, Any]) -> List[BusinessRule]:
		"""Get applicable business rules for tenant and context"""
		# This would query the database for applicable rules
		# For now, return empty list
		return []

	async def _evaluate_rule(self, rule: BusinessRule, evaluation_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Evaluate a single business rule"""
		# Implement rule evaluation logic based on rule conditions
		return {
			"rule_id": rule.id,
			"rule_name": rule.rule_name,
			"decision": PolicyDecisionType.ALLOW,
			"reason": "rule_evaluation_not_implemented"
		}

	def _get_escalation_reason(self, risk_score: float, risk_factors: List[str]) -> str:
		"""Get human-readable escalation reason"""
		if risk_score >= 0.8:
			return "Very high risk detected - multiple risk factors present"
		elif risk_score >= 0.6:
			return "High risk detected - additional verification required"
		elif risk_score >= 0.4:
			return "Medium risk detected - enhanced security measures applied"
		else:
			return "Standard security measures applied"

	# Database operations (placeholders)

	async def _get_user_profile(self, user_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get user profile"""
		pass

	async def _get_user_method_types(self, user_id: str, tenant_id: str) -> List[MFAMethodType]:
		"""Get user's enrolled method types"""
		pass

	async def _get_tenant_constraints(self, tenant_id: str) -> Dict[str, Any]:
		"""Get tenant-specific constraints"""
		pass

	async def _get_method_policies(self, tenant_id: str, method_type: MFAMethodType) -> Dict[str, Any]:
		"""Get method-specific policies"""
		pass


__all__ = [
	"BusinessLogicOrchestrator",
	"BusinessRule",
	"WorkflowContext",
	"PolicyDecisionType",
	"WorkflowStage"
]