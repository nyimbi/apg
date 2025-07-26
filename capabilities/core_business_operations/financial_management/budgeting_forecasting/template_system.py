"""
APG Budgeting & Forecasting - Template System

Advanced template management system with inheritance, sharing,
and intelligent template recommendations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from uuid import UUID
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum

import asyncpg
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict

from .models import (
	APGBaseModel, BFBudgetType, BFLineType,
	PositiveAmount, CurrencyCode, FiscalYear, NonEmptyString
)
from .service import APGTenantContext, BFServiceConfig, ServiceResponse, APGServiceBase
from uuid_extensions import uuid7str


# =============================================================================
# Template System Enumerations and Models
# =============================================================================

class TemplateCategory(str, Enum):
	"""Template category enumeration."""
	ANNUAL_BUDGET = "annual_budget"
	QUARTERLY_BUDGET = "quarterly_budget"
	DEPARTMENT_BUDGET = "department_budget"
	PROJECT_BUDGET = "project_budget"
	CAPITAL_BUDGET = "capital_budget"
	OPERATIONAL_BUDGET = "operational_budget"
	CUSTOM = "custom"


class TemplateScope(str, Enum):
	"""Template scope enumeration."""
	PRIVATE = "private"
	TENANT = "tenant"
	PUBLIC = "public"
	SYSTEM = "system"


class TemplateComplexity(str, Enum):
	"""Template complexity level enumeration."""
	SIMPLE = "simple"
	INTERMEDIATE = "intermediate"
	ADVANCED = "advanced"
	ENTERPRISE = "enterprise"


class BudgetTemplateModel(APGBaseModel):
	"""Comprehensive budget template model with inheritance and sharing capabilities."""
	
	# Template identification
	template_name: NonEmptyString = Field(..., max_length=255)
	template_description: Optional[str] = Field(None, max_length=1000)
	template_category: TemplateCategory = Field(...)
	template_scope: TemplateScope = Field(default=TemplateScope.PRIVATE)
	template_complexity: TemplateComplexity = Field(default=TemplateComplexity.SIMPLE)
	
	# Template metadata
	industry_type: Optional[str] = Field(None, max_length=100)
	organization_size: Optional[str] = Field(None, max_length=50)  # small, medium, large, enterprise
	applicable_regions: List[str] = Field(default_factory=list)
	supported_currencies: List[CurrencyCode] = Field(default_factory=lambda: ["USD"])
	
	# Ownership and sharing
	owner_tenant_id: str = Field(...)
	owner_user_id: str = Field(...)
	is_system_template: bool = Field(default=False)
	is_certified: bool = Field(default=False)
	certification_date: Optional[datetime] = Field(None)
	certified_by: Optional[str] = Field(None)
	
	# Template inheritance
	parent_template_id: Optional[str] = Field(None)
	derived_templates: List[str] = Field(default_factory=list)
	inheritance_level: int = Field(default=0, ge=0, le=5)
	allows_inheritance: bool = Field(default=True)
	
	# Template structure and configuration
	template_structure: Dict[str, Any] = Field(default_factory=dict)
	line_item_templates: List[Dict[str, Any]] = Field(default_factory=list)
	default_settings: Dict[str, Any] = Field(default_factory=dict)
	calculation_rules: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Customization controls
	customizable_fields: List[str] = Field(default_factory=list)
	locked_fields: List[str] = Field(default_factory=list)
	required_fields: List[str] = Field(default_factory=list)
	field_constraints: Dict[str, Any] = Field(default_factory=dict)
	
	# Sharing and permissions
	sharing_permissions: Dict[str, Any] = Field(default_factory=dict)
	shared_with_tenants: List[str] = Field(default_factory=list)
	shared_with_users: List[str] = Field(default_factory=list)
	access_level: str = Field(default="view_only", max_length=20)  # view_only, use, modify, full
	
	# Usage and analytics
	usage_count: int = Field(default=0, ge=0)
	last_used_date: Optional[datetime] = Field(None)
	success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
	average_customization_time: Optional[int] = Field(None, ge=0)  # in minutes
	user_ratings: List[Dict[str, Any]] = Field(default_factory=list)
	average_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
	
	# Lifecycle management
	template_status: str = Field(default="active", max_length=20)  # draft, active, deprecated, archived
	deprecation_date: Optional[datetime] = Field(None)
	replacement_template_id: Optional[str] = Field(None)
	maintenance_schedule: Optional[str] = Field(None)
	
	# Validation and compliance
	validation_rules: List[Dict[str, Any]] = Field(default_factory=list)
	compliance_requirements: List[str] = Field(default_factory=list)
	audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
	
	# AI and ML enhancements
	ai_generated: bool = Field(default=False)
	ai_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	prediction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
	recommended_improvements: List[str] = Field(default_factory=list)
	
	# Performance metrics
	template_size_kb: Optional[int] = Field(None, ge=0)
	load_time_ms: Optional[int] = Field(None, ge=0)
	customization_complexity_score: Optional[float] = Field(None, ge=0.0, le=10.0)

	@validator('parent_template_id')
	def validate_parent_template(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
		"""Validate parent template relationship."""
		if v and v == values.get('id'):
			raise ValueError("Template cannot be its own parent")
		return v

	@root_validator
	def validate_template_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate template data consistency."""
		# Validate inheritance level
		parent_id = values.get('parent_template_id')
		inheritance_level = values.get('inheritance_level', 0)
		
		if parent_id and inheritance_level == 0:
			values['inheritance_level'] = 1
		elif not parent_id and inheritance_level > 0:
			raise ValueError("Inheritance level > 0 requires parent template")
		
		# Validate sharing permissions
		template_scope = values.get('template_scope')
		shared_tenants = values.get('shared_with_tenants', [])
		
		if template_scope == TemplateScope.PRIVATE and shared_tenants:
			raise ValueError("Private templates cannot be shared with other tenants")
		
		return values


class TemplateLineItem(BaseModel):
	"""Template line item with intelligent defaults and validation."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	line_id: str = Field(default_factory=uuid7str)
	line_number: int = Field(..., ge=1)
	line_description: NonEmptyString = Field(..., max_length=500)
	line_category: NonEmptyString = Field(..., max_length=100)
	line_type: BFLineType = Field(...)
	
	# Account mapping template
	account_code: NonEmptyString = Field(..., max_length=50)
	account_category: Optional[str] = Field(None, max_length=100)
	gl_account_template: Optional[str] = Field(None, max_length=50)
	
	# Amount templates and formulas
	amount_template: str = Field(..., max_length=200)  # Formula or fixed amount
	amount_type: str = Field(default="fixed", max_length=20)  # fixed, percentage, formula, driver_based
	calculation_formula: Optional[str] = Field(None)
	
	# Driver-based templates
	quantity_driver_template: Optional[str] = Field(None, max_length=100)
	unit_price_template: Optional[str] = Field(None)
	escalation_template: Optional[str] = Field(None)
	
	# Period allocation templates
	allocation_method: str = Field(default="equal", max_length=50)
	seasonality_pattern: Optional[str] = Field(None, max_length=50)  # constant, seasonal, custom
	monthly_distribution: Optional[List[float]] = Field(None)
	
	# Organizational allocation
	department_codes: List[str] = Field(default_factory=list)
	cost_center_codes: List[str] = Field(default_factory=list)
	project_codes: List[str] = Field(default_factory=list)
	
	# Validation and constraints
	min_amount: Optional[Decimal] = Field(None, ge=0)
	max_amount: Optional[Decimal] = Field(None, ge=0)
	amount_constraints: Dict[str, Any] = Field(default_factory=dict)
	validation_rules: List[str] = Field(default_factory=list)
	
	# Customization options
	is_required: bool = Field(default=True)
	is_customizable: bool = Field(default=True)
	customization_options: Dict[str, Any] = Field(default_factory=dict)
	help_text: Optional[str] = Field(None)
	
	# AI and suggestions
	ai_suggested: bool = Field(default=False)
	confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	historical_variance: Optional[float] = Field(None)
	benchmark_data: Optional[Dict[str, Any]] = Field(None)

	@validator('monthly_distribution')
	def validate_monthly_distribution(cls, v: Optional[List[float]]) -> Optional[List[float]]:
		"""Validate monthly distribution sums to 1.0."""
		if v is not None:
			if len(v) != 12:
				raise ValueError("Monthly distribution must have exactly 12 values")
			if abs(sum(v) - 1.0) > 0.01:
				raise ValueError("Monthly distribution must sum to 1.0")
		return v


class TemplateUsageHistory(BaseModel):
	"""Template usage history for analytics and improvement."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	usage_id: str = Field(default_factory=uuid7str)
	template_id: str = Field(...)
	used_by_tenant_id: str = Field(...)
	used_by_user_id: str = Field(...)
	
	# Usage details
	usage_date: datetime = Field(default_factory=datetime.utcnow)
	budget_created_id: Optional[str] = Field(None)
	customizations_made: List[Dict[str, Any]] = Field(default_factory=list)
	time_to_complete_minutes: Optional[int] = Field(None, ge=0)
	
	# User feedback
	user_rating: Optional[int] = Field(None, ge=1, le=5)
	user_feedback: Optional[str] = Field(None)
	suggested_improvements: List[str] = Field(default_factory=list)
	reported_issues: List[str] = Field(default_factory=list)
	
	# Success metrics
	budget_approved: Optional[bool] = Field(None)
	approval_time_days: Optional[int] = Field(None, ge=0)
	variance_from_template: Optional[float] = Field(None, ge=0.0)
	success_indicators: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Template Management Service
# =============================================================================

class TemplateManagementService(APGServiceBase):
	"""
	Comprehensive template management service with intelligent
	recommendations, inheritance, and collaborative template development.
	"""
	
	async def create_template(self, template_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new budget template with comprehensive validation."""
		try:
			# Validate permissions
			if not await self._validate_permissions('template.create'):
				raise PermissionError("Insufficient permissions to create template")
			
			# Inject tenant and user context
			template_data.update({
				'tenant_id': self.context.tenant_id,
				'owner_tenant_id': self.context.tenant_id,
				'owner_user_id': self.context.user_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Create template model
			template = BudgetTemplateModel(**template_data)
			
			# Validate template structure
			validation_result = await self._validate_template_structure(template)
			if not validation_result['is_valid']:
				return ServiceResponse(
					success=False,
					message="Template validation failed",
					errors=validation_result['errors']
				)
			
			# Check for duplicate template names in tenant
			existing = await self._check_template_name_exists(template.template_name, self.context.tenant_id)
			if existing:
				raise ValueError(f"Template name '{template.template_name}' already exists")
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert template
				template_id = await self._insert_template(template)
				
				# Insert line item templates
				if template.line_item_templates:
					await self._insert_template_line_items(template_id, template.line_item_templates)
				
				# Create initial version if versioning is enabled
				await self._create_template_version(template_id, 1, "Initial template creation", template.dict())
				
				# Index template for search if search service is available
				await self._index_template_for_search(template_id, template)
				
				# Audit the creation
				await self._audit_action('create', 'template', template_id, new_data=template.dict())
			
			return ServiceResponse(
				success=True,
				message=f"Template '{template.template_name}' created successfully",
				data={'template_id': template_id, 'template_name': template.template_name}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_template')
	
	async def get_template_recommendations(self, criteria: Dict[str, Any]) -> ServiceResponse:
		"""Get intelligent template recommendations based on criteria."""
		try:
			# Validate permissions
			if not await self._validate_permissions('template.view'):
				raise PermissionError("Insufficient permissions to view templates")
			
			# Extract recommendation criteria
			budget_type = criteria.get('budget_type')
			department = criteria.get('department')
			industry = criteria.get('industry')
			organization_size = criteria.get('organization_size', 'medium')
			fiscal_year = criteria.get('fiscal_year')
			amount_range = criteria.get('amount_range')
			
			# Build recommendation query with ML scoring
			recommendations = await self._get_ai_template_recommendations(criteria)
			
			# Get usage-based recommendations
			popular_templates = await self._get_popular_templates(criteria)
			
			# Get tenant-specific recommendations
			tenant_templates = await self._get_tenant_templates()
			
			# Combine and rank recommendations
			combined_recommendations = await self._rank_template_recommendations(
				recommendations, popular_templates, tenant_templates, criteria
			)
			
			# Add template details and metadata
			detailed_recommendations = []
			for rec in combined_recommendations[:10]:  # Top 10 recommendations
				template_details = await self._get_template_details(rec['template_id'])
				if template_details:
					template_details.update({
						'recommendation_score': rec['score'],
						'recommendation_reason': rec['reason'],
						'match_criteria': rec['match_criteria']
					})
					detailed_recommendations.append(template_details)
			
			return ServiceResponse(
				success=True,
				message=f"Found {len(detailed_recommendations)} template recommendations",
				data={
					'recommendations': detailed_recommendations,
					'criteria_used': criteria,
					'total_available_templates': await self._count_available_templates()
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'get_template_recommendations')
	
	async def create_template_from_budget(self, budget_id: str, template_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a template from an existing successful budget."""
		try:
			# Validate permissions
			if not await self._validate_permissions('template.create_from_budget'):
				raise PermissionError("Insufficient permissions to create template from budget")
			
			# Get budget data
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			# Check if user has access to this budget
			if budget['tenant_id'] != self.context.tenant_id:
				raise PermissionError("No access to this budget")
			
			# Get budget line items
			budget_lines = await self._get_budget_lines(budget_id)
			
			# Extract template structure from budget
			template_structure = await self._extract_template_structure(budget, budget_lines)
			
			# Create line item templates
			line_item_templates = await self._create_line_item_templates(budget_lines)
			
			# Merge with provided template data
			template_data.update({
				'template_structure': template_structure,
				'line_item_templates': line_item_templates,
				'tenant_id': self.context.tenant_id,
				'owner_tenant_id': self.context.tenant_id,
				'owner_user_id': self.context.user_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id,
				'source_budget_id': budget_id
			})
			
			# Create template
			template = BudgetTemplateModel(**template_data)
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert template
				template_id = await self._insert_template(template)
				
				# Link to source budget for provenance
				await self._link_template_to_source_budget(template_id, budget_id)
				
				# Record template derivation
				await self._record_template_derivation(template_id, 'budget', budget_id)
				
				# Audit the creation
				await self._audit_action('create_from_budget', 'template', template_id, 
										new_data={'source_budget_id': budget_id})
			
			return ServiceResponse(
				success=True,
				message=f"Template created successfully from budget '{budget['budget_name']}'",
				data={'template_id': template_id, 'source_budget_id': budget_id}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_template_from_budget')
	
	async def share_template(self, template_id: str, sharing_config: Dict[str, Any]) -> ServiceResponse:
		"""Share template with other tenants or make it public."""
		try:
			# Validate permissions
			if not await self._validate_permissions('template.share', template_id):
				raise PermissionError("Insufficient permissions to share template")
			
			# Get template
			template = await self._get_template(template_id)
			if not template:
				raise ValueError("Template not found")
			
			# Check ownership
			if template['owner_tenant_id'] != self.context.tenant_id:
				raise PermissionError("Only template owner can share template")
			
			# Validate sharing configuration
			share_with_tenants = sharing_config.get('share_with_tenants', [])
			make_public = sharing_config.get('make_public', False)
			access_level = sharing_config.get('access_level', 'view_only')
			expiration_date = sharing_config.get('expiration_date')
			
			# Start database transaction
			async with self._connection.transaction():
				# Update template sharing settings
				await self._connection.execute("""
					UPDATE budget_templates 
					SET shared_with_tenants = $1,
						template_scope = $2,
						access_level = $3,
						sharing_permissions = $4,
						updated_at = NOW(),
						updated_by = $5
					WHERE id = $6
				""", 
					share_with_tenants,
					'public' if make_public else 'tenant',
					access_level,
					json.dumps(sharing_config),
					self.context.user_id,
					template_id
				)
				
				# Create sharing notifications
				if share_with_tenants:
					await self._send_template_sharing_notifications(template_id, share_with_tenants)
				
				# Record sharing history
				await self._record_template_sharing(template_id, sharing_config)
				
				# Audit the sharing
				await self._audit_action('share', 'template', template_id, 
										new_data=sharing_config)
			
			return ServiceResponse(
				success=True,
				message=f"Template shared successfully with {len(share_with_tenants)} tenant(s)",
				data={
					'template_id': template_id,
					'shared_with': share_with_tenants,
					'is_public': make_public,
					'access_level': access_level
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'share_template')
	
	async def inherit_template(self, parent_template_id: str, inheritance_config: Dict[str, Any]) -> ServiceResponse:
		"""Create a new template by inheriting from an existing template."""
		try:
			# Validate permissions
			if not await self._validate_permissions('template.inherit'):
				raise PermissionError("Insufficient permissions to inherit template")
			
			# Get parent template
			parent_template = await self._get_template(parent_template_id)
			if not parent_template:
				raise ValueError("Parent template not found")
			
			# Check access to parent template
			if not await self._can_access_template(parent_template, self.context.tenant_id):
				raise PermissionError("No access to parent template")
			
			# Check if parent allows inheritance
			if not parent_template.get('allows_inheritance', True):
				raise ValueError("Parent template does not allow inheritance")
			
			# Check inheritance depth limit
			parent_inheritance_level = parent_template.get('inheritance_level', 0)
			if parent_inheritance_level >= 5:  # Max inheritance depth
				raise ValueError("Maximum inheritance depth exceeded")
			
			# Create inherited template data
			inherited_data = await self._create_inherited_template_data(parent_template, inheritance_config)
			inherited_data.update({
				'parent_template_id': parent_template_id,
				'inheritance_level': parent_inheritance_level + 1,
				'tenant_id': self.context.tenant_id,
				'owner_tenant_id': self.context.tenant_id,
				'owner_user_id': self.context.user_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Create inherited template
			template = BudgetTemplateModel(**inherited_data)
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert template
				template_id = await self._insert_template(template)
				
				# Update parent template with derived template reference
				await self._connection.execute("""
					UPDATE budget_templates 
					SET derived_templates = array_append(
						COALESCE(derived_templates, '{}'), $1
					)
					WHERE id = $2
				""", template_id, parent_template_id)
				
				# Copy and customize line items
				await self._inherit_template_line_items(template_id, parent_template_id, inheritance_config)
				
				# Record inheritance relationship
				await self._record_template_inheritance(template_id, parent_template_id)
				
				# Audit the inheritance
				await self._audit_action('inherit', 'template', template_id, 
										new_data={'parent_template_id': parent_template_id})
			
			return ServiceResponse(
				success=True,
				message=f"Template inherited successfully from '{parent_template['template_name']}'",
				data={
					'template_id': template_id,
					'parent_template_id': parent_template_id,
					'inheritance_level': template.inheritance_level
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'inherit_template')
	
	# =============================================================================
	# Helper Methods
	# =============================================================================
	
	async def _validate_template_structure(self, template: BudgetTemplateModel) -> Dict[str, Any]:
		"""Validate template structure and content."""
		errors = []
		
		# Validate template structure
		if not template.template_structure:
			errors.append("Template structure cannot be empty")
		
		# Validate line item templates
		if template.line_item_templates:
			for i, line_template in enumerate(template.line_item_templates):
				try:
					TemplateLineItem(**line_template)
				except Exception as e:
					errors.append(f"Line item template {i+1}: {str(e)}")
		
		# Validate customizable fields
		for field in template.customizable_fields:
			if field in template.locked_fields:
				errors.append(f"Field '{field}' cannot be both customizable and locked")
		
		return {
			'is_valid': len(errors) == 0,
			'errors': errors
		}
	
	async def _check_template_name_exists(self, template_name: str, tenant_id: str) -> bool:
		"""Check if template name already exists for tenant."""
		result = await self._connection.fetchval("""
			SELECT EXISTS(
				SELECT 1 FROM budget_templates 
				WHERE template_name = $1 AND owner_tenant_id = $2 AND is_deleted = FALSE
			)
		""", template_name, tenant_id)
		return result
	
	async def _insert_template(self, template: BudgetTemplateModel) -> str:
		"""Insert template into database."""
		template_dict = template.dict()
		columns = list(template_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(template_dict.values())
		
		query = f"""
			INSERT INTO budget_templates ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)
	
	async def _insert_template_line_items(self, template_id: str, line_items: List[Dict[str, Any]]) -> None:
		"""Insert template line items."""
		for line_item in line_items:
			line_item.update({
				'template_id': template_id,
				'created_by': self.context.user_id
			})
			
			line_template = TemplateLineItem(**line_item)
			await self._insert_template_line_item(line_template)
	
	async def _insert_template_line_item(self, line_item: TemplateLineItem) -> None:
		"""Insert single template line item."""
		line_dict = line_item.dict()
		columns = list(line_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(line_dict.values())
		
		query = f"""
			INSERT INTO template_line_items ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
		"""
		
		await self._connection.execute(query, *values)
	
	async def _get_ai_template_recommendations(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Get AI-powered template recommendations."""
		# This would integrate with APG ai_orchestration for intelligent recommendations
		# For now, return rule-based recommendations
		
		budget_type = criteria.get('budget_type')
		industry = criteria.get('industry')
		organization_size = criteria.get('organization_size')
		
		query = """
			SELECT id, template_name, template_category, success_rate, usage_count,
				   average_rating, template_complexity
			FROM budget_templates
			WHERE template_scope IN ('public', 'tenant')
			  AND template_status = 'active'
			  AND ($1::text IS NULL OR template_category = $1)
			  AND ($2::text IS NULL OR industry_type = $2)
			  AND ($3::text IS NULL OR organization_size = $3)
			ORDER BY 
				CASE WHEN success_rate IS NOT NULL THEN success_rate ELSE 0.5 END DESC,
				usage_count DESC,
				average_rating DESC NULLS LAST
			LIMIT 20
		"""
		
		results = await self._connection.fetch(query, budget_type, industry, organization_size)
		
		recommendations = []
		for row in results:
			score = await self._calculate_recommendation_score(dict(row), criteria)
			recommendations.append({
				'template_id': row['id'],
				'score': score,
				'reason': f"Matches {budget_type or 'general'} budget needs",
				'match_criteria': ['budget_type', 'industry', 'organization_size']
			})
		
		return recommendations
	
	async def _calculate_recommendation_score(self, template: Dict[str, Any], criteria: Dict[str, Any]) -> float:
		"""Calculate recommendation score for template."""
		score = 0.0
		
		# Success rate weight (40%)
		success_rate = template.get('success_rate', 0.5)
		score += success_rate * 0.4
		
		# Usage popularity weight (20%)
		usage_count = template.get('usage_count', 0)
		usage_score = min(usage_count / 100, 1.0)  # Normalize to max 100 uses
		score += usage_score * 0.2
		
		# User rating weight (20%)
		rating = template.get('average_rating', 3.0)
		rating_score = rating / 5.0
		score += rating_score * 0.2
		
		# Criteria match weight (20%)
		match_score = await self._calculate_criteria_match_score(template, criteria)
		score += match_score * 0.2
		
		return min(score, 1.0)
	
	async def _calculate_criteria_match_score(self, template: Dict[str, Any], criteria: Dict[str, Any]) -> float:
		"""Calculate how well template matches search criteria."""
		matches = 0
		total_criteria = 0
		
		# Check budget type match
		if criteria.get('budget_type'):
			total_criteria += 1
			if template.get('template_category') == criteria.get('budget_type'):
				matches += 1
		
		# Check industry match
		if criteria.get('industry'):
			total_criteria += 1
			if template.get('industry_type') == criteria.get('industry'):
				matches += 1
		
		# Check organization size match
		if criteria.get('organization_size'):
			total_criteria += 1
			if template.get('organization_size') == criteria.get('organization_size'):
				matches += 1
		
		return matches / max(total_criteria, 1)
	
	async def _get_popular_templates(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Get popular templates based on usage statistics."""
		query = """
			SELECT id, usage_count, last_used_date, average_rating
			FROM budget_templates
			WHERE template_scope IN ('public', 'tenant')
			  AND template_status = 'active'
			  AND usage_count > 0
			ORDER BY usage_count DESC, average_rating DESC NULLS LAST
			LIMIT 10
		"""
		
		results = await self._connection.fetch(query)
		
		popular = []
		for row in results:
			popular.append({
				'template_id': row['id'],
				'score': 0.8,  # High score for popular templates
				'reason': f"Popular template with {row['usage_count']} uses",
				'match_criteria': ['popularity']
			})
		
		return popular
	
	async def _get_tenant_templates(self) -> List[Dict[str, Any]]:
		"""Get templates created by current tenant."""
		query = """
			SELECT id, template_name, usage_count, created_at
			FROM budget_templates
			WHERE owner_tenant_id = $1
			  AND template_status = 'active'
			ORDER BY created_at DESC
			LIMIT 5
		"""
		
		results = await self._connection.fetch(query, self.context.tenant_id)
		
		tenant_templates = []
		for row in results:
			tenant_templates.append({
				'template_id': row['id'],
				'score': 0.9,  # High score for own templates
				'reason': "Created by your organization",
				'match_criteria': ['ownership']
			})
		
		return tenant_templates
	
	async def _rank_template_recommendations(self, *recommendation_lists) -> List[Dict[str, Any]]:
		"""Rank and combine multiple recommendation lists."""
		all_recommendations = {}
		
		for rec_list in recommendation_lists:
			for rec in rec_list:
				template_id = rec['template_id']
				if template_id in all_recommendations:
					# Boost score for templates appearing in multiple lists
					all_recommendations[template_id]['score'] = min(
						all_recommendations[template_id]['score'] + rec['score'] * 0.1,
						1.0
					)
				else:
					all_recommendations[template_id] = rec
		
		# Sort by score
		ranked = sorted(all_recommendations.values(), key=lambda x: x['score'], reverse=True)
		
		return ranked


# =============================================================================
# Service Factory and Export
# =============================================================================

def create_template_service(context: APGTenantContext, config: BFServiceConfig) -> TemplateManagementService:
	"""Factory function to create template management service."""
	return TemplateManagementService(context, config)


# Export template system classes
__all__ = [
	'TemplateCategory',
	'TemplateScope', 
	'TemplateComplexity',
	'BudgetTemplateModel',
	'TemplateLineItem',
	'TemplateUsageHistory',
	'TemplateManagementService',
	'create_template_service'
]


def _log_template_system_summary() -> str:
	"""Log summary of template system capabilities."""
	return f"Template System loaded: {len(__all__)} components with AI recommendations and inheritance support"