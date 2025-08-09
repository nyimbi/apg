"""
APG Customer Relationship Management - Lead Scoring Module

Advanced lead scoring system with AI-powered algorithms for automatic
lead qualification and prioritization based on multiple factors.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ValidationError

from .models import CRMContact, CRMLead, ContactType, LeadSource, LeadStatus
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class ScoreCategory(str, Enum):
	"""Lead scoring categories"""
	DEMOGRAPHIC = "demographic"
	FIRMOGRAPHIC = "firmographic"
	BEHAVIORAL = "behavioral"
	ENGAGEMENT = "engagement"
	FIT = "fit"
	INTENT = "intent"


class ScoreWeight(str, Enum):
	"""Score weight levels"""
	CRITICAL = "critical"		# 40-50 points
	HIGH = "high"				# 20-30 points
	MEDIUM = "medium"			# 10-15 points
	LOW = "low"					# 1-5 points


class LeadScoreRule(BaseModel):
	"""Individual lead scoring rule"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Rule definition
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	category: ScoreCategory
	weight: ScoreWeight
	
	# Scoring logic
	field: str = Field(..., description="Field to evaluate")
	operator: str = Field(..., description="Comparison operator")
	value: Union[str, int, float, bool, List[Any]] = Field(..., description="Value to compare against")
	score_points: int = Field(..., description="Points awarded when rule matches")
	
	# Rule conditions
	is_active: bool = Field(True, description="Whether rule is active")
	applies_to_lead_sources: List[LeadSource] = Field(default_factory=list, description="Specific lead sources this rule applies to")
	applies_to_contact_types: List[ContactType] = Field(default_factory=list, description="Specific contact types this rule applies to")
	
	# Time-based conditions
	valid_from: Optional[datetime] = Field(None, description="Rule valid from date")
	valid_until: Optional[datetime] = Field(None, description="Rule valid until date")
	
	# Metadata
	usage_count: int = Field(0, description="Number of times rule has been applied")
	last_used_at: Optional[datetime] = None
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class LeadScore(BaseModel):
	"""Lead scoring result"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	lead_id: str
	contact_id: Optional[str] = None
	
	# Score details
	total_score: int = Field(0, description="Total calculated score")
	max_possible_score: int = Field(0, description="Maximum possible score")
	score_percentage: float = Field(0.0, description="Score as percentage of maximum")
	grade: str = Field("F", description="Letter grade (A, B, C, D, F)")
	
	# Category breakdown
	demographic_score: int = 0
	firmographic_score: int = 0
	behavioral_score: int = 0
	engagement_score: int = 0
	fit_score: int = 0
	intent_score: int = 0
	
	# Rule applications
	applied_rules: List[str] = Field(default_factory=list, description="IDs of rules that were applied")
	rule_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed rule application results")
	
	# Recommendations
	recommended_action: Optional[str] = Field(None, description="Recommended next action")
	priority_level: str = Field("medium", description="Priority level (hot, warm, cold)")
	qualification_status: str = Field("unqualified", description="Qualification status")
	
	# Timing and freshness
	calculated_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: Optional[datetime] = Field(None, description="When score expires")
	calculation_duration_ms: Optional[int] = None
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)


class LeadScoringAnalytics(BaseModel):
	"""Analytics for lead scoring performance"""
	tenant_id: str
	
	# Score distribution
	total_leads_scored: int = 0
	average_score: float = 0.0
	median_score: float = 0.0
	score_distribution: Dict[str, int] = Field(default_factory=dict)  # Grade -> count
	
	# Category performance
	category_averages: Dict[str, float] = Field(default_factory=dict)
	top_performing_rules: List[Dict[str, Any]] = Field(default_factory=list)
	underperforming_rules: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Conversion metrics
	high_score_conversions: int = 0
	low_score_conversions: int = 0
	conversion_by_grade: Dict[str, float] = Field(default_factory=dict)
	
	# Time-based metrics
	scores_calculated_today: int = 0
	scores_calculated_this_week: int = 0
	scores_calculated_this_month: int = 0
	
	# Performance metrics
	average_calculation_time_ms: float = 0.0
	calculation_errors: int = 0
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_period_days: int = 30


class LeadScoringManager:
	"""
	Advanced lead scoring management system
	
	Provides intelligent lead scoring capabilities with configurable rules,
	real-time scoring, analytics, and automated qualification workflows.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize lead scoring manager
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
		self._default_rules_created = False
	
	async def initialize(self):
		"""Initialize the lead scoring manager"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Lead Scoring Manager...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		self._initialized = True
		logger.info("✅ Lead Scoring Manager initialized successfully")
	
	async def create_scoring_rule(
		self,
		rule_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> LeadScoreRule:
		"""
		Create a new lead scoring rule
		
		Args:
			rule_data: Rule configuration data
			tenant_id: Tenant identifier
			created_by: User creating the rule
			
		Returns:
			Created scoring rule
		"""
		try:
			logger.info(f"📏 Creating lead scoring rule: {rule_data.get('name')}")
			
			# Add required fields
			rule_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create rule object
			rule = LeadScoreRule(**rule_data)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_lead_score_rules (
						id, tenant_id, name, description, category, weight,
						field, operator, value, score_points, is_active,
						applies_to_lead_sources, applies_to_contact_types,
						valid_from, valid_until, usage_count, last_used_at,
						created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
						$12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
					)
				""", 
				rule.id, rule.tenant_id, rule.name, rule.description,
				rule.category.value, rule.weight.value,
				rule.field, rule.operator, rule.value, rule.score_points,
				rule.is_active, [source.value for source in rule.applies_to_lead_sources],
				[ctype.value for ctype in rule.applies_to_contact_types],
				rule.valid_from, rule.valid_until, rule.usage_count, rule.last_used_at,
				rule.created_at, rule.updated_at, rule.created_by, rule.updated_by, rule.version
				)
			
			logger.info(f"✅ Lead scoring rule created successfully: {rule.id}")
			return rule
			
		except Exception as e:
			logger.error(f"Failed to create lead scoring rule: {str(e)}", exc_info=True)
			raise
	
	async def calculate_lead_score(
		self,
		lead_id: str,
		tenant_id: str,
		force_recalculate: bool = False
	) -> LeadScore:
		"""
		Calculate comprehensive score for a lead
		
		Args:
			lead_id: Lead identifier
			tenant_id: Tenant identifier
			force_recalculate: Force recalculation even if recent score exists
			
		Returns:
			Calculated lead score
		"""
		try:
			logger.info(f"🎯 Calculating lead score: {lead_id}")
			
			start_time = datetime.utcnow()
			
			# Check for existing recent score
			if not force_recalculate:
				existing_score = await self._get_recent_score(lead_id, tenant_id)
				if existing_score:
					logger.info(f"📊 Using existing recent score for lead: {lead_id}")
					return existing_score
			
			# Get lead and contact data
			lead_data = await self._get_lead_data(lead_id, tenant_id)
			if not lead_data:
				raise ValueError(f"Lead not found: {lead_id}")
			
			# Get active scoring rules
			rules = await self._get_active_scoring_rules(tenant_id)
			
			# Initialize score
			score = LeadScore(
				tenant_id=tenant_id,
				lead_id=lead_id,
				contact_id=lead_data.get('contact_id')
			)
			
			# Apply each rule
			total_possible = 0
			for rule in rules:
				if self._should_apply_rule(rule, lead_data):
					rule_result = await self._apply_scoring_rule(rule, lead_data)
					if rule_result['applies']:
						# Add to appropriate category
						category_score = getattr(score, f"{rule.category.value}_score")
						setattr(score, f"{rule.category.value}_score", category_score + rule_result['points'])
						
						# Track rule application
						score.applied_rules.append(rule.id)
						score.rule_details.append(rule_result)
						
						# Update rule usage
						await self._update_rule_usage(rule.id, tenant_id)
				
				total_possible += rule.score_points
			
			# Calculate totals
			score.total_score = (
				score.demographic_score + score.firmographic_score +
				score.behavioral_score + score.engagement_score +
				score.fit_score + score.intent_score
			)
			score.max_possible_score = total_possible
			score.score_percentage = (score.total_score / total_possible * 100) if total_possible > 0 else 0
			
			# Assign grade and recommendations
			score.grade = self._calculate_grade(score.score_percentage)
			score.priority_level = self._determine_priority_level(score.total_score, score.score_percentage)
			score.qualification_status = self._determine_qualification_status(score.total_score, score.grade)
			score.recommended_action = self._get_recommended_action(score)
			
			# Set expiration (scores expire after 7 days)
			score.expires_at = datetime.utcnow() + timedelta(days=7)
			
			# Calculate duration
			end_time = datetime.utcnow()
			score.calculation_duration_ms = int((end_time - start_time).total_seconds() * 1000)
			
			# Store score
			await self._store_lead_score(score)
			
			logger.info(f"✅ Lead score calculated: {score.total_score}/{score.max_possible_score} ({score.grade})")
			return score
			
		except Exception as e:
			logger.error(f"Failed to calculate lead score: {str(e)}", exc_info=True)
			raise
	
	async def get_lead_score(
		self,
		lead_id: str,
		tenant_id: str
	) -> Optional[LeadScore]:
		"""
		Get current lead score
		
		Args:
			lead_id: Lead identifier
			tenant_id: Tenant identifier
			
		Returns:
			Current lead score if exists
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_lead_scores 
					WHERE lead_id = $1 AND tenant_id = $2
					AND (expires_at IS NULL OR expires_at > NOW())
					ORDER BY calculated_at DESC
					LIMIT 1
				""", lead_id, tenant_id)
				
				if not row:
					return None
				
				return LeadScore(**dict(row))
				
		except Exception as e:
			logger.error(f"Failed to get lead score: {str(e)}", exc_info=True)
			raise
	
	async def batch_score_leads(
		self,
		lead_ids: List[str],
		tenant_id: str,
		force_recalculate: bool = False
	) -> Dict[str, LeadScore]:
		"""
		Calculate scores for multiple leads in batch
		
		Args:
			lead_ids: List of lead identifiers
			tenant_id: Tenant identifier
			force_recalculate: Force recalculation for all leads
			
		Returns:
			Dict mapping lead IDs to their scores
		"""
		try:
			logger.info(f"🎯 Batch scoring {len(lead_ids)} leads")
			
			results = {}
			for lead_id in lead_ids:
				try:
					score = await self.calculate_lead_score(lead_id, tenant_id, force_recalculate)
					results[lead_id] = score
				except Exception as e:
					logger.error(f"Failed to score lead {lead_id}: {str(e)}")
					# Continue with other leads
			
			logger.info(f"✅ Batch scoring completed: {len(results)}/{len(lead_ids)} successful")
			return results
			
		except Exception as e:
			logger.error(f"Failed to batch score leads: {str(e)}", exc_info=True)
			raise
	
	async def get_scoring_analytics(
		self,
		tenant_id: str,
		period_days: int = 30
	) -> LeadScoringAnalytics:
		"""
		Get comprehensive lead scoring analytics
		
		Args:
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Scoring analytics data
		"""
		try:
			logger.info(f"📊 Generating lead scoring analytics for {period_days} days")
			
			analytics = LeadScoringAnalytics(
				tenant_id=tenant_id,
				analysis_period_days=period_days
			)
			
			async with self.db_manager.get_connection() as conn:
				# Basic score statistics
				stats_row = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_leads,
						AVG(total_score) as avg_score,
						PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_score) as median_score
					FROM crm_lead_scores 
					WHERE tenant_id = $1 
					AND calculated_at >= NOW() - INTERVAL '%s days'
				""", tenant_id, period_days)
				
				if stats_row:
					analytics.total_leads_scored = stats_row['total_leads'] or 0
					analytics.average_score = float(stats_row['avg_score'] or 0)
					analytics.median_score = float(stats_row['median_score'] or 0)
				
				# Grade distribution
				grade_rows = await conn.fetch("""
					SELECT grade, COUNT(*) as count
					FROM crm_lead_scores 
					WHERE tenant_id = $1 
					AND calculated_at >= NOW() - INTERVAL '%s days'
					GROUP BY grade
				""", tenant_id, period_days)
				
				for row in grade_rows:
					analytics.score_distribution[row['grade']] = row['count']
				
				# Category averages
				category_rows = await conn.fetch("""
					SELECT 
						AVG(demographic_score) as avg_demographic,
						AVG(firmographic_score) as avg_firmographic,
						AVG(behavioral_score) as avg_behavioral,
						AVG(engagement_score) as avg_engagement,
						AVG(fit_score) as avg_fit,
						AVG(intent_score) as avg_intent
					FROM crm_lead_scores 
					WHERE tenant_id = $1 
					AND calculated_at >= NOW() - INTERVAL '%s days'
				""", tenant_id, period_days)
				
				if category_rows:
					row = category_rows[0]
					analytics.category_averages = {
						'demographic': float(row['avg_demographic'] or 0),
						'firmographic': float(row['avg_firmographic'] or 0),
						'behavioral': float(row['avg_behavioral'] or 0),
						'engagement': float(row['avg_engagement'] or 0),
						'fit': float(row['avg_fit'] or 0),
						'intent': float(row['avg_intent'] or 0)
					}
				
				# Time-based metrics
				time_metrics = await conn.fetchrow("""
					SELECT 
						COUNT(*) FILTER (WHERE calculated_at >= CURRENT_DATE) as today,
						COUNT(*) FILTER (WHERE calculated_at >= DATE_TRUNC('week', NOW())) as this_week,
						COUNT(*) FILTER (WHERE calculated_at >= DATE_TRUNC('month', NOW())) as this_month,
						AVG(calculation_duration_ms) as avg_duration
					FROM crm_lead_scores 
					WHERE tenant_id = $1
				""", tenant_id)
				
				if time_metrics:
					analytics.scores_calculated_today = time_metrics['today'] or 0
					analytics.scores_calculated_this_week = time_metrics['this_week'] or 0
					analytics.scores_calculated_this_month = time_metrics['this_month'] or 0
					analytics.average_calculation_time_ms = float(time_metrics['avg_duration'] or 0)
			
			logger.info(f"✅ Generated analytics for {analytics.total_leads_scored} scored leads")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate scoring analytics: {str(e)}", exc_info=True)
			raise
	
	async def create_default_scoring_rules(
		self,
		tenant_id: str,
		created_by: str
	) -> List[LeadScoreRule]:
		"""
		Create default lead scoring rules for a new tenant
		
		Args:
			tenant_id: Tenant identifier
			created_by: User creating the rules
			
		Returns:
			List of created default rules
		"""
		try:
			logger.info("🏗️ Creating default lead scoring rules")
			
			default_rules = [
				# Demographic Rules
				{
					'name': 'Job Title - Decision Maker',
					'description': 'High score for decision-making job titles',
					'category': ScoreCategory.DEMOGRAPHIC,
					'weight': ScoreWeight.HIGH,
					'field': 'job_title',
					'operator': 'contains_any',
					'value': ['CEO', 'CTO', 'VP', 'Director', 'Manager', 'Head of'],
					'score_points': 25
				},
				{
					'name': 'Company Size - Enterprise',
					'description': 'High score for large companies',
					'category': ScoreCategory.FIRMOGRAPHIC,
					'weight': ScoreWeight.HIGH,
					'field': 'company_size',
					'operator': 'greater_than',
					'value': 1000,
					'score_points': 30
				},
				{
					'name': 'Lead Source - Referral',
					'description': 'High score for referral leads',
					'category': ScoreCategory.FIT,
					'weight': ScoreWeight.CRITICAL,
					'field': 'lead_source',
					'operator': 'equals',
					'value': LeadSource.REFERRAL.value,
					'score_points': 40
				},
				{
					'name': 'Email Engagement - High',
					'description': 'High score for email engagement',
					'category': ScoreCategory.ENGAGEMENT,
					'weight': ScoreWeight.MEDIUM,
					'field': 'email_opens',
					'operator': 'greater_than',
					'value': 5,
					'score_points': 15
				},
				{
					'name': 'Website Behavior - Multiple Visits',
					'description': 'Score for multiple website visits',
					'category': ScoreCategory.BEHAVIORAL,
					'weight': ScoreWeight.MEDIUM,
					'field': 'website_visits',
					'operator': 'greater_than',
					'value': 3,
					'score_points': 12
				},
				# Intent Rules
				{
					'name': 'Downloaded Premium Content',
					'description': 'High intent signal',
					'category': ScoreCategory.INTENT,
					'weight': ScoreWeight.HIGH,
					'field': 'downloaded_content',
					'operator': 'contains',
					'value': 'whitepaper',
					'score_points': 20
				},
				{
					'name': 'Requested Demo',
					'description': 'Very high intent signal',
					'category': ScoreCategory.INTENT,
					'weight': ScoreWeight.CRITICAL,
					'field': 'requested_demo',
					'operator': 'equals',
					'value': True,
					'score_points': 45
				}
			]
			
			created_rules = []
			for rule_data in default_rules:
				rule = await self.create_scoring_rule(rule_data, tenant_id, created_by)
				created_rules.append(rule)
			
			self._default_rules_created = True
			logger.info(f"✅ Created {len(created_rules)} default scoring rules")
			return created_rules
			
		except Exception as e:
			logger.error(f"Failed to create default scoring rules: {str(e)}", exc_info=True)
			raise
	
	async def _get_recent_score(
		self,
		lead_id: str,
		tenant_id: str,
		max_age_hours: int = 24
	) -> Optional[LeadScore]:
		"""Get recent score if exists"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_lead_scores 
					WHERE lead_id = $1 AND tenant_id = $2
					AND calculated_at > NOW() - INTERVAL '%s hours'
					ORDER BY calculated_at DESC
					LIMIT 1
				""", lead_id, tenant_id, max_age_hours)
				
				if row:
					return LeadScore(**dict(row))
				return None
				
		except Exception as e:
			logger.error(f"Failed to get recent score: {str(e)}", exc_info=True)
			return None
	
	async def _get_lead_data(self, lead_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get comprehensive lead data for scoring"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Get lead and contact data
				row = await conn.fetchrow("""
					SELECT 
						l.*,
						c.first_name, c.last_name, c.email, c.phone,
						c.job_title, c.company_name, c.industry,
						c.contact_type, c.lead_source,
						c.created_at as contact_created_at,
						c.metadata as contact_metadata
					FROM crm_leads l
					LEFT JOIN crm_contacts c ON l.contact_id = c.id
					WHERE l.id = $1 AND l.tenant_id = $2
				""", lead_id, tenant_id)
				
				if row:
					return dict(row)
				return None
				
		except Exception as e:
			logger.error(f"Failed to get lead data: {str(e)}", exc_info=True)
			return None
	
	async def _get_active_scoring_rules(self, tenant_id: str) -> List[LeadScoreRule]:
		"""Get all active scoring rules for tenant"""
		try:
			async with self.db_manager.get_connection() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_lead_score_rules 
					WHERE tenant_id = $1 AND is_active = true
					AND (valid_from IS NULL OR valid_from <= NOW())
					AND (valid_until IS NULL OR valid_until > NOW())
					ORDER BY category, score_points DESC
				""", tenant_id)
				
				return [LeadScoreRule(**dict(row)) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get scoring rules: {str(e)}", exc_info=True)
			return []
	
	def _should_apply_rule(self, rule: LeadScoreRule, lead_data: Dict[str, Any]) -> bool:
		"""Determine if a rule should be applied to this lead"""
		# Check lead source restrictions
		if rule.applies_to_lead_sources:
			lead_source = lead_data.get('lead_source')
			if lead_source not in [source.value for source in rule.applies_to_lead_sources]:
				return False
		
		# Check contact type restrictions
		if rule.applies_to_contact_types:
			contact_type = lead_data.get('contact_type')
			if contact_type not in [ctype.value for ctype in rule.applies_to_contact_types]:
				return False
		
		return True
	
	async def _apply_scoring_rule(
		self,
		rule: LeadScoreRule,
		lead_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Apply a single scoring rule to lead data"""
		result = {
			'rule_id': rule.id,
			'rule_name': rule.name,
			'applies': False,
			'points': 0,
			'reason': '',
			'field_value': None
		}
		
		try:
			field_value = lead_data.get(rule.field)
			result['field_value'] = field_value
			
			# Apply operator logic
			applies = False
			
			if rule.operator == 'equals':
				applies = field_value == rule.value
			elif rule.operator == 'not_equals':
				applies = field_value != rule.value
			elif rule.operator == 'contains':
				applies = rule.value in str(field_value or '')
			elif rule.operator == 'contains_any':
				if isinstance(rule.value, list) and field_value:
					applies = any(item.lower() in str(field_value).lower() for item in rule.value)
			elif rule.operator == 'greater_than':
				applies = (field_value or 0) > rule.value
			elif rule.operator == 'less_than':
				applies = (field_value or 0) < rule.value
			elif rule.operator == 'is_not_null':
				applies = field_value is not None
			elif rule.operator == 'is_null':
				applies = field_value is None
			
			if applies:
				result['applies'] = True
				result['points'] = rule.score_points
				result['reason'] = f"Rule '{rule.name}' matched: {field_value}"
			else:
				result['reason'] = f"Rule '{rule.name}' did not match: {field_value}"
			
			return result
			
		except Exception as e:
			logger.error(f"Error applying rule {rule.id}: {str(e)}", exc_info=True)
			result['reason'] = f"Error applying rule: {str(e)}"
			return result
	
	async def _update_rule_usage(self, rule_id: str, tenant_id: str):
		"""Update rule usage statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_lead_score_rules 
					SET usage_count = usage_count + 1, 
						last_used_at = NOW(),
						updated_at = NOW()
					WHERE id = $1 AND tenant_id = $2
				""", rule_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to update rule usage: {str(e)}", exc_info=True)
	
	def _calculate_grade(self, score_percentage: float) -> str:
		"""Calculate letter grade from score percentage"""
		if score_percentage >= 90:
			return "A"
		elif score_percentage >= 80:
			return "B"
		elif score_percentage >= 70:
			return "C"
		elif score_percentage >= 60:
			return "D"
		else:
			return "F"
	
	def _determine_priority_level(self, total_score: int, score_percentage: float) -> str:
		"""Determine priority level from score"""
		if total_score >= 80 or score_percentage >= 85:
			return "hot"
		elif total_score >= 50 or score_percentage >= 65:
			return "warm"
		else:
			return "cold"
	
	def _determine_qualification_status(self, total_score: int, grade: str) -> str:
		"""Determine qualification status"""
		if grade in ["A", "B"] and total_score >= 60:
			return "qualified"
		elif grade == "C" and total_score >= 40:
			return "marketing_qualified"
		else:
			return "unqualified"
	
	def _get_recommended_action(self, score: LeadScore) -> str:
		"""Get recommended action based on score"""
		if score.grade == "A":
			return "Contact immediately - high priority lead"
		elif score.grade == "B":
			return "Contact within 24 hours - good potential"
		elif score.grade == "C":
			return "Add to nurturing campaign"
		elif score.grade == "D":
			return "Continue monitoring and nurturing"
		else:
			return "Low priority - minimal follow-up"
	
	async def _store_lead_score(self, score: LeadScore):
		"""Store calculated lead score"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_lead_scores (
						id, tenant_id, lead_id, contact_id, total_score,
						max_possible_score, score_percentage, grade,
						demographic_score, firmographic_score, behavioral_score,
						engagement_score, fit_score, intent_score,
						applied_rules, rule_details, recommended_action,
						priority_level, qualification_status, calculated_at,
						expires_at, calculation_duration_ms, metadata
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
						$12, $13, $14, $15, $16, $17, $18, $19, $20,
						$21, $22, $23
					)
					ON CONFLICT (lead_id, tenant_id) DO UPDATE SET
						total_score = EXCLUDED.total_score,
						max_possible_score = EXCLUDED.max_possible_score,
						score_percentage = EXCLUDED.score_percentage,
						grade = EXCLUDED.grade,
						demographic_score = EXCLUDED.demographic_score,
						firmographic_score = EXCLUDED.firmographic_score,
						behavioral_score = EXCLUDED.behavioral_score,
						engagement_score = EXCLUDED.engagement_score,
						fit_score = EXCLUDED.fit_score,
						intent_score = EXCLUDED.intent_score,
						applied_rules = EXCLUDED.applied_rules,
						rule_details = EXCLUDED.rule_details,
						recommended_action = EXCLUDED.recommended_action,
						priority_level = EXCLUDED.priority_level,
						qualification_status = EXCLUDED.qualification_status,
						calculated_at = EXCLUDED.calculated_at,
						expires_at = EXCLUDED.expires_at,
						calculation_duration_ms = EXCLUDED.calculation_duration_ms,
						metadata = EXCLUDED.metadata
				""", 
				score.id, score.tenant_id, score.lead_id, score.contact_id,
				score.total_score, score.max_possible_score, score.score_percentage,
				score.grade, score.demographic_score, score.firmographic_score,
				score.behavioral_score, score.engagement_score, score.fit_score,
				score.intent_score, score.applied_rules, score.rule_details,
				score.recommended_action, score.priority_level,
				score.qualification_status, score.calculated_at, score.expires_at,
				score.calculation_duration_ms, score.metadata
				)
		except Exception as e:
			logger.error(f"Failed to store lead score: {str(e)}", exc_info=True)
			raise