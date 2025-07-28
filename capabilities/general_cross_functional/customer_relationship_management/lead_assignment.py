"""
APG Customer Relationship Management - Lead Assignment Rules Module

Advanced lead assignment system with intelligent routing, workload balancing,
territory-based assignment, and performance-driven distribution for optimal
sales team productivity and lead conversion rates.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from decimal import Decimal
from uuid_extensions import uuid7str
import json

from pydantic import BaseModel, Field, validator, ConfigDict

from .database import DatabaseManager


logger = logging.getLogger(__name__)


class AssignmentType(str, Enum):
	"""Types of lead assignment rules"""
	ROUND_ROBIN = "round_robin"
	TERRITORY_BASED = "territory_based"
	SKILL_BASED = "skill_based"
	WORKLOAD_BALANCED = "workload_balanced"
	PERFORMANCE_BASED = "performance_based"
	COMPANY_SIZE = "company_size"
	INDUSTRY_BASED = "industry_based"
	LEAD_SCORE_BASED = "lead_score_based"
	CUSTOM_RULE = "custom_rule"


class AssignmentStatus(str, Enum):
	"""Assignment rule status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	PAUSED = "paused"
	ARCHIVED = "archived"


class AssignmentPriority(str, Enum):
	"""Assignment rule priority"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


class WorkloadMetric(str, Enum):
	"""Workload calculation metrics"""
	ACTIVE_LEADS = "active_leads"
	OPEN_OPPORTUNITIES = "open_opportunities"
	MONTHLY_QUOTA = "monthly_quota"
	RESPONSE_TIME = "response_time"
	CONVERSION_RATE = "conversion_rate"
	WEIGHTED_PIPELINE = "weighted_pipeline"


class AssignmentCondition(BaseModel):
	"""Lead assignment condition configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	field_name: str = Field(..., description="Field to evaluate")
	operator: str = Field(..., description="Comparison operator")
	value: Union[str, int, float, List[str]] = Field(..., description="Comparison value")
	weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Condition weight")
	is_required: bool = Field(default=False, description="Whether condition is mandatory")
	created_at: datetime = Field(default_factory=datetime.now)


class AssignmentTarget(BaseModel):
	"""Assignment target configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(..., description="Target user ID")
	user_name: str = Field(..., description="Target user name")
	user_email: str = Field(..., description="Target user email")
	team: Optional[str] = Field(None, description="User team/department")
	territory: Optional[str] = Field(None, description="User territory")
	skills: List[str] = Field(default_factory=list, description="User skills/specializations")
	capacity: int = Field(default=100, ge=0, description="User capacity percentage")
	max_leads_per_month: Optional[int] = Field(None, ge=0, description="Monthly lead limit")
	priority_score: float = Field(default=5.0, ge=0.0, le=10.0, description="Assignment priority")
	workload_weight: float = Field(default=1.0, ge=0.1, le=5.0, description="Workload calculation weight")
	performance_score: float = Field(default=5.0, ge=0.0, le=10.0, description="Performance-based score")
	is_active: bool = Field(default=True, description="Whether target is active")
	backup_assignee: Optional[str] = Field(None, description="Backup assignee user ID")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)


class LeadAssignmentRule(BaseModel):
	"""Lead assignment rule configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Rule name")
	description: Optional[str] = Field(None, description="Rule description")
	assignment_type: AssignmentType = Field(..., description="Type of assignment logic")
	status: AssignmentStatus = Field(default=AssignmentStatus.ACTIVE, description="Rule status")
	priority: AssignmentPriority = Field(default=AssignmentPriority.MEDIUM, description="Rule priority")
	
	# Rule configuration
	conditions: List[AssignmentCondition] = Field(default_factory=list, description="Assignment conditions")
	targets: List[AssignmentTarget] = Field(..., min_items=1, description="Assignment targets")
	
	# Assignment settings
	round_robin_position: int = Field(default=0, ge=0, description="Current round-robin position")
	workload_metrics: List[WorkloadMetric] = Field(default_factory=list, description="Workload calculation metrics")
	rebalance_frequency: int = Field(default=24, ge=1, description="Rebalance frequency in hours")
	max_assignments_per_hour: Optional[int] = Field(None, ge=1, description="Rate limiting per assignee")
	
	# Business rules
	business_hours_only: bool = Field(default=False, description="Assign only during business hours")
	time_zone: str = Field(default="UTC", description="Time zone for business hours")
	exclude_weekends: bool = Field(default=False, description="Exclude weekend assignments")
	escalation_timeout_hours: int = Field(default=24, ge=1, description="Escalation timeout")
	
	# Analytics and tracking
	total_assignments: int = Field(default=0, ge=0, description="Total assignments made")
	successful_assignments: int = Field(default=0, ge=0, description="Successful assignments")
	failed_assignments: int = Field(default=0, ge=0, description="Failed assignments")
	avg_assignment_time_ms: float = Field(default=0.0, ge=0.0, description="Average assignment time")
	last_assignment_at: Optional[datetime] = Field(None, description="Last assignment timestamp")
	
	# Configuration
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional rule metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")


class LeadAssignment(BaseModel):
	"""Lead assignment record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	lead_id: str = Field(..., description="Lead identifier")
	rule_id: str = Field(..., description="Assignment rule used")
	
	# Assignment details
	assigned_to: str = Field(..., description="Assigned user ID")
	assigned_to_name: str = Field(..., description="Assigned user name")
	assigned_to_email: str = Field(..., description="Assigned user email")
	assigned_team: Optional[str] = Field(None, description="Assigned team")
	assigned_territory: Optional[str] = Field(None, description="Assigned territory")
	
	# Assignment metadata
	assignment_reason: str = Field(..., description="Reason for assignment")
	assignment_score: float = Field(default=0.0, description="Assignment confidence score")
	assignment_method: AssignmentType = Field(..., description="Assignment method used")
	assignment_duration_ms: int = Field(default=0, ge=0, description="Assignment processing time")
	
	# Lead context at assignment
	lead_score: Optional[float] = Field(None, description="Lead score at assignment")
	lead_source: Optional[str] = Field(None, description="Lead source")
	lead_industry: Optional[str] = Field(None, description="Lead industry")
	lead_company_size: Optional[str] = Field(None, description="Lead company size")
	lead_territory: Optional[str] = Field(None, description="Lead territory")
	
	# Status tracking
	is_accepted: bool = Field(default=False, description="Whether assignment was accepted")
	accepted_at: Optional[datetime] = Field(None, description="Assignment acceptance time")
	is_reassigned: bool = Field(default=False, description="Whether lead was reassigned")
	reassigned_at: Optional[datetime] = Field(None, description="Reassignment time")
	reassignment_reason: Optional[str] = Field(None, description="Reassignment reason")
	
	# Performance tracking
	first_contact_at: Optional[datetime] = Field(None, description="First contact timestamp")
	qualified_at: Optional[datetime] = Field(None, description="Lead qualification time")
	converted_at: Optional[datetime] = Field(None, description="Lead conversion time")
	response_time_hours: Optional[float] = Field(None, ge=0.0, description="Response time in hours")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional assignment metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)


class AssignmentAnalytics(BaseModel):
	"""Lead assignment analytics data"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	tenant_id: str = Field(..., description="Tenant identifier")
	period_start: datetime = Field(..., description="Analytics period start")
	period_end: datetime = Field(..., description="Analytics period end")
	
	# Overall metrics
	total_assignments: int = Field(default=0, description="Total assignments in period")
	successful_assignments: int = Field(default=0, description="Successful assignments")
	failed_assignments: int = Field(default=0, description="Failed assignments")
	avg_assignment_time_ms: float = Field(default=0.0, description="Average assignment processing time")
	
	# Performance metrics
	assignment_acceptance_rate: float = Field(default=0.0, description="Assignment acceptance rate")
	avg_response_time_hours: float = Field(default=0.0, description="Average response time")
	conversion_rate: float = Field(default=0.0, description="Assignment to conversion rate")
	reassignment_rate: float = Field(default=0.0, description="Reassignment rate")
	
	# Distribution metrics
	assignments_by_rule: Dict[str, int] = Field(default_factory=dict, description="Assignments by rule")
	assignments_by_user: Dict[str, int] = Field(default_factory=dict, description="Assignments by user")
	assignments_by_team: Dict[str, int] = Field(default_factory=dict, description="Assignments by team")
	assignments_by_territory: Dict[str, int] = Field(default_factory=dict, description="Assignments by territory")
	
	# Performance by assignee
	user_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance by user")
	team_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance by team")
	
	# Rule effectiveness
	rule_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance by rule")
	optimization_suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="Optimization recommendations")
	
	created_at: datetime = Field(default_factory=datetime.now)


class LeadAssignmentManager:
	"""Advanced lead assignment management system"""
	
	def __init__(self, db_manager: DatabaseManager):
		self.db_manager = db_manager
		self._initialized = False
		self._assignment_cache = {}
		self._workload_cache = {}
		self._performance_cache = {}
	
	async def initialize(self):
		"""Initialize the lead assignment manager"""
		try:
			logger.info("🚀 Initializing Lead Assignment Manager...")
			
			# Initialize database connection
			await self.db_manager.initialize()
			
			# Load active assignment rules
			await self._load_active_rules()
			
			# Initialize workload tracking
			await self._initialize_workload_tracking()
			
			self._initialized = True
			logger.info("✅ Lead Assignment Manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize Lead Assignment Manager: {str(e)}")
			raise
	
	async def create_assignment_rule(self, rule_data: Dict[str, Any], tenant_id: str, created_by: str) -> LeadAssignmentRule:
		"""Create a new lead assignment rule"""
		try:
			if not self._initialized:
				await self.initialize()
			
			# Validate rule data
			rule = LeadAssignmentRule(
				tenant_id=tenant_id,
				created_by=created_by,
				**rule_data
			)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_lead_assignment_rules (
						id, tenant_id, name, description, assignment_type, status, priority,
						conditions, targets, round_robin_position, workload_metrics,
						rebalance_frequency, max_assignments_per_hour, business_hours_only,
						time_zone, exclude_weekends, escalation_timeout_hours, metadata,
						created_by, created_at, updated_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
				""", 
				rule.id, rule.tenant_id, rule.name, rule.description, rule.assignment_type.value,
				rule.status.value, rule.priority.value, json.dumps([c.model_dump() for c in rule.conditions]),
				json.dumps([t.model_dump() for t in rule.targets]), rule.round_robin_position,
				json.dumps([m.value for m in rule.workload_metrics]), rule.rebalance_frequency,
				rule.max_assignments_per_hour, rule.business_hours_only, rule.time_zone,
				rule.exclude_weekends, rule.escalation_timeout_hours, json.dumps(rule.metadata),
				rule.created_by, rule.created_at, rule.updated_at
				)
			
			# Update cache
			self._assignment_cache[rule.id] = rule
			
			logger.info(f"✅ Created assignment rule: {rule.name} ({rule.id})")
			return rule
			
		except Exception as e:
			logger.error(f"Failed to create assignment rule: {str(e)}")
			raise
	
	async def assign_lead(self, lead_data: Dict[str, Any], tenant_id: str) -> Optional[LeadAssignment]:
		"""Assign a lead using configured rules"""
		try:
			if not self._initialized:
				await self.initialize()
			
			lead_id = lead_data.get('id')
			if not lead_id:
				raise ValueError("Lead ID is required for assignment")
			
			logger.info(f"🎯 Assigning lead: {lead_id}")
			
			# Get active assignment rules sorted by priority
			rules = await self._get_active_rules(tenant_id)
			
			if not rules:
				logger.warning(f"No active assignment rules found for tenant: {tenant_id}")
				return None
			
			# Evaluate rules and find best match
			best_assignment = None
			best_score = 0.0
			
			for rule in rules:
				assignment = await self._evaluate_rule(rule, lead_data, tenant_id)
				if assignment and assignment.assignment_score > best_score:
					best_assignment = assignment
					best_score = assignment.assignment_score
			
			if not best_assignment:
				logger.warning(f"No suitable assignment found for lead: {lead_id}")
				return None
			
			# Store assignment
			await self._store_assignment(best_assignment)
			
			# Update rule statistics
			await self._update_rule_stats(best_assignment.rule_id, True)
			
			# Send assignment notification
			await self._send_assignment_notification(best_assignment)
			
			logger.info(f"✅ Lead {lead_id} assigned to {best_assignment.assigned_to_name}")
			return best_assignment
			
		except Exception as e:
			logger.error(f"Failed to assign lead: {str(e)}")
			if 'rule_id' in locals():
				await self._update_rule_stats(rule_id, False)
			raise
	
	async def _evaluate_rule(self, rule: LeadAssignmentRule, lead_data: Dict[str, Any], tenant_id: str) -> Optional[LeadAssignment]:
		"""Evaluate a specific assignment rule against lead data"""
		try:
			start_time = datetime.now()
			
			# Check if rule conditions match
			if not await self._check_rule_conditions(rule, lead_data):
				return None
			
			# Get available targets
			available_targets = await self._get_available_targets(rule, tenant_id)
			
			if not available_targets:
				return None
			
			# Select best target based on assignment type
			selected_target = await self._select_target(rule, available_targets, lead_data, tenant_id)
			
			if not selected_target:
				return None
			
			# Calculate assignment score
			assignment_score = await self._calculate_assignment_score(rule, selected_target, lead_data)
			
			# Create assignment record
			assignment = LeadAssignment(
				tenant_id=tenant_id,
				lead_id=lead_data['id'],
				rule_id=rule.id,
				assigned_to=selected_target.user_id,
				assigned_to_name=selected_target.user_name,
				assigned_to_email=selected_target.user_email,
				assigned_team=selected_target.team,
				assigned_territory=selected_target.territory,
				assignment_reason=f"Matched rule: {rule.name}",
				assignment_score=assignment_score,
				assignment_method=rule.assignment_type,
				assignment_duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
				lead_score=lead_data.get('score'),
				lead_source=lead_data.get('source'),
				lead_industry=lead_data.get('industry'),
				lead_company_size=lead_data.get('company_size'),
				lead_territory=lead_data.get('territory')
			)
			
			return assignment
			
		except Exception as e:
			logger.error(f"Failed to evaluate assignment rule {rule.id}: {str(e)}")
			return None
	
	async def _check_rule_conditions(self, rule: LeadAssignmentRule, lead_data: Dict[str, Any]) -> bool:
		"""Check if lead data matches rule conditions"""
		try:
			if not rule.conditions:
				return True
			
			required_conditions_met = 0
			required_conditions_total = sum(1 for c in rule.conditions if c.is_required)
			optional_score = 0.0
			optional_total = 0
			
			for condition in rule.conditions:
				field_value = lead_data.get(condition.field_name)
				condition_met = self._evaluate_condition(condition, field_value)
				
				if condition.is_required:
					if condition_met:
						required_conditions_met += 1
					else:
						return False  # Required condition not met
				else:
					optional_total += 1
					if condition_met:
						optional_score += condition.weight
			
			# All required conditions must be met
			if required_conditions_total > 0 and required_conditions_met < required_conditions_total:
				return False
			
			# Optional conditions contribute to overall match quality
			if optional_total > 0:
				optional_match_rate = optional_score / (optional_total * 10.0)  # Normalize to 0-1
				return optional_match_rate >= 0.3  # At least 30% optional match
			
			return True
			
		except Exception as e:
			logger.error(f"Failed to check rule conditions: {str(e)}")
			return False
	
	def _evaluate_condition(self, condition: AssignmentCondition, field_value: Any) -> bool:
		"""Evaluate a single assignment condition"""
		try:
			if field_value is None:
				return False
			
			operator = condition.operator.lower()
			target_value = condition.value
			
			if operator == "equals":
				return field_value == target_value
			elif operator == "not_equals":
				return field_value != target_value
			elif operator == "greater_than":
				return float(field_value) > float(target_value)
			elif operator == "less_than":
				return float(field_value) < float(target_value)
			elif operator == "greater_equal":
				return float(field_value) >= float(target_value)
			elif operator == "less_equal":
				return float(field_value) <= float(target_value)
			elif operator == "contains":
				return str(target_value).lower() in str(field_value).lower()
			elif operator == "not_contains":
				return str(target_value).lower() not in str(field_value).lower()
			elif operator == "starts_with":
				return str(field_value).lower().startswith(str(target_value).lower())
			elif operator == "ends_with":
				return str(field_value).lower().endswith(str(target_value).lower())
			elif operator == "in":
				if isinstance(target_value, list):
					return field_value in target_value
				return field_value == target_value
			elif operator == "not_in":
				if isinstance(target_value, list):
					return field_value not in target_value
				return field_value != target_value
			elif operator == "regex":
				import re
				return bool(re.search(str(target_value), str(field_value), re.IGNORECASE))
			else:
				logger.warning(f"Unknown condition operator: {operator}")
				return False
				
		except Exception as e:
			logger.error(f"Failed to evaluate condition: {str(e)}")
			return False
	
	async def _get_available_targets(self, rule: LeadAssignmentRule, tenant_id: str) -> List[AssignmentTarget]:
		"""Get available assignment targets for a rule"""
		try:
			available_targets = []
			
			for target in rule.targets:
				if not target.is_active:
					continue
				
				# Check capacity and workload limits
				if target.capacity <= 0:
					continue
				
				# Check monthly lead limits
				if target.max_leads_per_month:
					current_month_assignments = await self._get_monthly_assignments(target.user_id, tenant_id)
					if current_month_assignments >= target.max_leads_per_month:
						continue
				
				# Check business hours if required
				if rule.business_hours_only and not self._is_business_hours(rule.time_zone):
					continue
				
				# Check rate limiting
				if rule.max_assignments_per_hour:
					recent_assignments = await self._get_recent_assignments(target.user_id, tenant_id, hours=1)
					if recent_assignments >= rule.max_assignments_per_hour:
						continue
				
				available_targets.append(target)
			
			return available_targets
			
		except Exception as e:
			logger.error(f"Failed to get available targets: {str(e)}")
			return []
	
	async def _select_target(self, rule: LeadAssignmentRule, targets: List[AssignmentTarget], lead_data: Dict[str, Any], tenant_id: str) -> Optional[AssignmentTarget]:
		"""Select the best target based on assignment type"""
		try:
			if not targets:
				return None
			
			if rule.assignment_type == AssignmentType.ROUND_ROBIN:
				return await self._round_robin_selection(rule, targets)
			
			elif rule.assignment_type == AssignmentType.WORKLOAD_BALANCED:
				return await self._workload_balanced_selection(rule, targets, tenant_id)
			
			elif rule.assignment_type == AssignmentType.PERFORMANCE_BASED:
				return await self._performance_based_selection(targets, tenant_id)
			
			elif rule.assignment_type == AssignmentType.SKILL_BASED:
				return await self._skill_based_selection(targets, lead_data)
			
			elif rule.assignment_type == AssignmentType.TERRITORY_BASED:
				return await self._territory_based_selection(targets, lead_data)
			
			elif rule.assignment_type == AssignmentType.LEAD_SCORE_BASED:
				return await self._score_based_selection(targets, lead_data)
			
			else:
				# Default to first available target
				return targets[0]
				
		except Exception as e:
			logger.error(f"Failed to select target: {str(e)}")
			return targets[0] if targets else None
	
	async def _round_robin_selection(self, rule: LeadAssignmentRule, targets: List[AssignmentTarget]) -> AssignmentTarget:
		"""Round-robin target selection"""
		try:
			# Get current position and increment
			current_position = rule.round_robin_position % len(targets)
			selected_target = targets[current_position]
			
			# Update position in database
			new_position = (current_position + 1) % len(targets)
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_lead_assignment_rules 
					SET round_robin_position = $2, updated_at = NOW()
					WHERE id = $1
				""", rule.id, new_position)
			
			# Update cache
			rule.round_robin_position = new_position
			
			return selected_target
			
		except Exception as e:
			logger.error(f"Failed round-robin selection: {str(e)}")
			return targets[0]
	
	async def _workload_balanced_selection(self, rule: LeadAssignmentRule, targets: List[AssignmentTarget], tenant_id: str) -> AssignmentTarget:
		"""Workload-balanced target selection"""
		try:
			target_workloads = []
			
			for target in targets:
				workload = await self._calculate_workload(target, rule.workload_metrics, tenant_id)
				adjusted_workload = workload * target.workload_weight
				target_workloads.append((target, adjusted_workload))
			
			# Select target with lowest adjusted workload
			target_workloads.sort(key=lambda x: x[1])
			return target_workloads[0][0]
			
		except Exception as e:
			logger.error(f"Failed workload-balanced selection: {str(e)}")
			return targets[0]
	
	async def _calculate_workload(self, target: AssignmentTarget, metrics: List[WorkloadMetric], tenant_id: str) -> float:
		"""Calculate current workload for a target"""
		try:
			total_workload = 0.0
			
			for metric in metrics:
				if metric == WorkloadMetric.ACTIVE_LEADS:
					count = await self._get_active_leads_count(target.user_id, tenant_id)
					total_workload += count * 1.0
				
				elif metric == WorkloadMetric.OPEN_OPPORTUNITIES:
					count = await self._get_open_opportunities_count(target.user_id, tenant_id)
					total_workload += count * 2.0  # Opportunities weighted higher
				
				elif metric == WorkloadMetric.RESPONSE_TIME:
					avg_response = await self._get_avg_response_time(target.user_id, tenant_id)
					total_workload += max(0, avg_response - 4.0)  # Penalty for slow response
				
				elif metric == WorkloadMetric.CONVERSION_RATE:
					conversion_rate = await self._get_conversion_rate(target.user_id, tenant_id)
					total_workload -= conversion_rate * 10.0  # Bonus for high conversion
			
			return max(0.0, total_workload)
			
		except Exception as e:
			logger.error(f"Failed to calculate workload: {str(e)}")
			return 0.0
	
	async def get_assignment_analytics(self, tenant_id: str, period_days: int = 30) -> AssignmentAnalytics:
		"""Get assignment analytics for a tenant"""
		try:
			if not self._initialized:
				await self.initialize()
			
			period_start = datetime.now() - timedelta(days=period_days)
			period_end = datetime.now()
			
			async with self.db_manager.get_connection() as conn:
				# Get assignment statistics
				assignments = await conn.fetch("""
					SELECT * FROM crm_lead_assignments 
					WHERE tenant_id = $1 AND created_at >= $2 AND created_at <= $3
				""", tenant_id, period_start, period_end)
				
				# Calculate metrics
				total_assignments = len(assignments)
				successful_assignments = sum(1 for a in assignments if a.get('is_accepted', False))
				failed_assignments = total_assignments - successful_assignments
				
				# Build analytics
				analytics = AssignmentAnalytics(
					tenant_id=tenant_id,
					period_start=period_start,
					period_end=period_end,
					total_assignments=total_assignments,
					successful_assignments=successful_assignments,
					failed_assignments=failed_assignments
				)
				
				if assignments:
					# Calculate rates and averages
					analytics.assignment_acceptance_rate = successful_assignments / total_assignments * 100
					
					response_times = [a.get('response_time_hours', 0) for a in assignments if a.get('response_time_hours')]
					if response_times:
						analytics.avg_response_time_hours = sum(response_times) / len(response_times)
					
					# Group by various dimensions
					analytics.assignments_by_user = {}
					analytics.assignments_by_rule = {}
					
					for assignment in assignments:
						user_id = assignment.get('assigned_to', 'unknown')
						rule_id = assignment.get('rule_id', 'unknown')
						
						analytics.assignments_by_user[user_id] = analytics.assignments_by_user.get(user_id, 0) + 1
						analytics.assignments_by_rule[rule_id] = analytics.assignments_by_rule.get(rule_id, 0) + 1
				
				logger.info(f"📊 Generated assignment analytics for {period_days} days")
				return analytics
				
		except Exception as e:
			logger.error(f"Failed to get assignment analytics: {str(e)}")
			raise
	
	async def _load_active_rules(self):
		"""Load active assignment rules into cache"""
		try:
			async with self.db_manager.get_connection() as conn:
				rules_data = await conn.fetch("""
					SELECT * FROM crm_lead_assignment_rules 
					WHERE status = 'active'
					ORDER BY priority DESC, created_at ASC
				""")
				
				for rule_data in rules_data:
					rule = LeadAssignmentRule(
						id=rule_data['id'],
						tenant_id=rule_data['tenant_id'],
						name=rule_data['name'],
						description=rule_data['description'],
						assignment_type=AssignmentType(rule_data['assignment_type']),
						status=AssignmentStatus(rule_data['status']),
						priority=AssignmentPriority(rule_data['priority']),
						conditions=[AssignmentCondition(**c) for c in json.loads(rule_data['conditions'] or '[]')],
						targets=[AssignmentTarget(**t) for t in json.loads(rule_data['targets'] or '[]')],
						round_robin_position=rule_data['round_robin_position'],
						workload_metrics=[WorkloadMetric(m) for m in json.loads(rule_data['workload_metrics'] or '[]')],
						rebalance_frequency=rule_data['rebalance_frequency'],
						max_assignments_per_hour=rule_data['max_assignments_per_hour'],
						business_hours_only=rule_data['business_hours_only'],
						time_zone=rule_data['time_zone'],
						exclude_weekends=rule_data['exclude_weekends'],
						escalation_timeout_hours=rule_data['escalation_timeout_hours'],
						metadata=json.loads(rule_data['metadata'] or '{}'),
						created_by=rule_data['created_by'],
						created_at=rule_data['created_at'],
						updated_at=rule_data['updated_at']
					)
					
					self._assignment_cache[rule.id] = rule
			
			logger.info(f"📋 Loaded {len(self._assignment_cache)} active assignment rules")
			
		except Exception as e:
			logger.error(f"Failed to load active rules: {str(e)}")
			raise
	
	async def _get_active_rules(self, tenant_id: str) -> List[LeadAssignmentRule]:
		"""Get active rules for a tenant, sorted by priority"""
		rules = [rule for rule in self._assignment_cache.values() 
				if rule.tenant_id == tenant_id and rule.status == AssignmentStatus.ACTIVE]
		
		# Sort by priority (critical > high > medium > low)
		priority_order = {
			AssignmentPriority.CRITICAL: 4,
			AssignmentPriority.HIGH: 3,
			AssignmentPriority.MEDIUM: 2,
			AssignmentPriority.LOW: 1
		}
		
		return sorted(rules, key=lambda r: priority_order.get(r.priority, 0), reverse=True)
	
	async def _initialize_workload_tracking(self):
		"""Initialize workload tracking system"""
		try:
			# This would integrate with existing CRM data to track workloads
			# For now, we'll implement basic tracking
			logger.info("📊 Workload tracking initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize workload tracking: {str(e)}")
			raise
	
	# Helper methods for database operations
	async def _store_assignment(self, assignment: LeadAssignment):
		"""Store assignment in database"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				INSERT INTO crm_lead_assignments (
					id, tenant_id, lead_id, rule_id, assigned_to, assigned_to_name, assigned_to_email,
					assigned_team, assigned_territory, assignment_reason, assignment_score, assignment_method,
					assignment_duration_ms, lead_score, lead_source, lead_industry, lead_company_size,
					lead_territory, metadata, created_at, updated_at
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
			""", 
			assignment.id, assignment.tenant_id, assignment.lead_id, assignment.rule_id,
			assignment.assigned_to, assignment.assigned_to_name, assignment.assigned_to_email,
			assignment.assigned_team, assignment.assigned_territory, assignment.assignment_reason,
			assignment.assignment_score, assignment.assignment_method.value, assignment.assignment_duration_ms,
			assignment.lead_score, assignment.lead_source, assignment.lead_industry,
			assignment.lead_company_size, assignment.lead_territory, json.dumps(assignment.metadata),
			assignment.created_at, assignment.updated_at
			)
	
	async def _update_rule_stats(self, rule_id: str, success: bool):
		"""Update rule assignment statistics"""
		async with self.db_manager.get_connection() as conn:
			if success:
				await conn.execute("""
					UPDATE crm_lead_assignment_rules SET
						total_assignments = total_assignments + 1,
						successful_assignments = successful_assignments + 1,
						last_assignment_at = NOW(),
						updated_at = NOW()
					WHERE id = $1
				""", rule_id)
			else:
				await conn.execute("""
					UPDATE crm_lead_assignment_rules SET
						total_assignments = total_assignments + 1,
						failed_assignments = failed_assignments + 1,
						updated_at = NOW()
					WHERE id = $1
				""", rule_id)
	
	async def _send_assignment_notification(self, assignment: LeadAssignment):
		"""Send assignment notification (placeholder for notification system)"""
		logger.info(f"📧 Assignment notification sent to {assignment.assigned_to_email}")
	
	# Placeholder methods for workload calculations
	async def _get_monthly_assignments(self, user_id: str, tenant_id: str) -> int:
		"""Get monthly assignments count for user"""
		return 0  # Placeholder
	
	async def _get_recent_assignments(self, user_id: str, tenant_id: str, hours: int) -> int:
		"""Get recent assignments count for user"""
		return 0  # Placeholder
	
	async def _get_active_leads_count(self, user_id: str, tenant_id: str) -> int:
		"""Get active leads count for user"""
		return 0  # Placeholder
	
	async def _get_open_opportunities_count(self, user_id: str, tenant_id: str) -> int:
		"""Get open opportunities count for user"""
		return 0  # Placeholder
	
	async def _get_avg_response_time(self, user_id: str, tenant_id: str) -> float:
		"""Get average response time for user"""
		return 4.0  # Placeholder
	
	async def _get_conversion_rate(self, user_id: str, tenant_id: str) -> float:
		"""Get conversion rate for user"""
		return 0.2  # Placeholder
	
	def _is_business_hours(self, timezone: str) -> bool:
		"""Check if current time is within business hours"""
		return True  # Placeholder
	
	async def _calculate_assignment_score(self, rule: LeadAssignmentRule, target: AssignmentTarget, lead_data: Dict[str, Any]) -> float:
		"""Calculate assignment confidence score"""
		return target.priority_score  # Simplified scoring
	
	async def _performance_based_selection(self, targets: List[AssignmentTarget], tenant_id: str) -> AssignmentTarget:
		"""Select target based on performance metrics"""
		return max(targets, key=lambda t: t.performance_score)
	
	async def _skill_based_selection(self, targets: List[AssignmentTarget], lead_data: Dict[str, Any]) -> AssignmentTarget:
		"""Select target based on skills matching"""
		# Simplified skill matching
		lead_industry = lead_data.get('industry', '').lower()
		for target in targets:
			if any(skill.lower() in lead_industry for skill in target.skills):
				return target
		return targets[0]
	
	async def _territory_based_selection(self, targets: List[AssignmentTarget], lead_data: Dict[str, Any]) -> AssignmentTarget:
		"""Select target based on territory matching"""
		lead_territory = lead_data.get('territory', '')
		for target in targets:
			if target.territory == lead_territory:
				return target
		return targets[0]
	
	async def _score_based_selection(self, targets: List[AssignmentTarget], lead_data: Dict[str, Any]) -> AssignmentTarget:
		"""Select target based on lead score thresholds"""
		lead_score = lead_data.get('score', 0)
		# High-score leads go to high-performing reps
		if lead_score >= 80:
			return max(targets, key=lambda t: t.performance_score)
		return targets[0]