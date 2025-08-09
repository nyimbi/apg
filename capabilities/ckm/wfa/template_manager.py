"""
APG Workflow & Business Process Management - Template & Library Manager

Comprehensive process template management with categorization, versioning,
sharing, and intelligent template recommendations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict
import hashlib

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessTemplate, ProcessStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Template Core Classes
# =============================================================================

class TemplateCategory(str, Enum):
	"""Process template categories."""
	BUSINESS_PROCESS = "business_process"
	APPROVAL_WORKFLOW = "approval_workflow"
	DATA_PROCESSING = "data_processing"
	CUSTOMER_SERVICE = "customer_service"
	HR_PROCESS = "hr_process"
	FINANCE_PROCESS = "finance_process"
	IT_PROCESS = "it_process"
	COMPLIANCE_PROCESS = "compliance_process"
	QUALITY_ASSURANCE = "quality_assurance"
	PROJECT_MANAGEMENT = "project_management"
	CUSTOM = "custom"


class TemplateScope(str, Enum):
	"""Template sharing scope."""
	PRIVATE = "private"
	TEAM = "team"
	ORGANIZATION = "organization"
	PUBLIC = "public"


class TemplateComplexity(str, Enum):
	"""Template complexity levels."""
	SIMPLE = "simple"
	MODERATE = "moderate"
	COMPLEX = "complex"
	ADVANCED = "advanced"


class UsageTracking(str, Enum):
	"""Template usage tracking types."""
	VIEW = "view"
	DOWNLOAD = "download"
	INSTANTIATE = "instantiate"
	CUSTOMIZE = "customize"
	SHARE = "share"


@dataclass
class TemplateMetadata:
	"""Extended template metadata."""
	estimated_duration: Optional[str] = None
	participant_roles: List[str] = field(default_factory=list)
	required_skills: List[str] = field(default_factory=list)
	business_value: str = ""
	use_cases: List[str] = field(default_factory=list)
	prerequisites: List[str] = field(default_factory=list)
	success_metrics: List[str] = field(default_factory=list)
	related_templates: List[str] = field(default_factory=list)


@dataclass
class TemplateUsage:
	"""Template usage record."""
	usage_id: str = field(default_factory=lambda: f"usage_{uuid.uuid4().hex}")
	template_id: str = ""
	user_id: str = ""
	usage_type: UsageTracking = UsageTracking.VIEW
	timestamp: datetime = field(default_factory=datetime.utcnow)
	context_data: Dict[str, Any] = field(default_factory=dict)
	tenant_id: str = ""


@dataclass
class TemplateRating:
	"""Template rating and review."""
	rating_id: str = field(default_factory=lambda: f"rating_{uuid.uuid4().hex}")
	template_id: str = ""
	user_id: str = ""
	rating: int = 5  # 1-5 stars
	review: Optional[str] = None
	helpful_votes: int = 0
	timestamp: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


@dataclass
class TemplateRecommendation:
	"""AI-powered template recommendation."""
	recommendation_id: str = field(default_factory=lambda: f"rec_{uuid.uuid4().hex}")
	template_id: str = ""
	user_id: str = ""
	recommendation_score: float = 0.0  # 0-1 scale
	recommendation_reason: str = ""
	context_factors: List[str] = field(default_factory=list)
	generated_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


@dataclass
class TemplateCollection:
	"""Curated collection of templates."""
	collection_id: str = field(default_factory=lambda: f"collection_{uuid.uuid4().hex}")
	collection_name: str = ""
	description: str = ""
	template_ids: List[str] = field(default_factory=list)
	tags: List[str] = field(default_factory=list)
	created_by: str = ""
	visibility: TemplateScope = TemplateScope.PRIVATE
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


# =============================================================================
# Template Analytics
# =============================================================================

class TemplateAnalytics:
	"""Analyze template usage and performance."""
	
	def __init__(self):
		self.usage_data: Dict[str, List[TemplateUsage]] = defaultdict(list)
		self.rating_data: Dict[str, List[TemplateRating]] = defaultdict(list)
	
	async def track_usage(self, usage: TemplateUsage) -> None:
		"""Track template usage."""
		self.usage_data[usage.template_id].append(usage)
		
		# Keep only recent usage data (last 10000 records per template)
		if len(self.usage_data[usage.template_id]) > 10000:
			self.usage_data[usage.template_id] = self.usage_data[usage.template_id][-5000:]
	
	async def add_rating(self, rating: TemplateRating) -> None:
		"""Add template rating."""
		self.rating_data[rating.template_id].append(rating)
	
	async def get_template_analytics(
		self,
		template_id: str,
		time_window: Optional[timedelta] = None
	) -> Dict[str, Any]:
		"""Get comprehensive template analytics."""
		if time_window:
			cutoff_time = datetime.utcnow() - time_window
			usage_records = [
				usage for usage in self.usage_data.get(template_id, [])
				if usage.timestamp >= cutoff_time
			]
			rating_records = [
				rating for rating in self.rating_data.get(template_id, [])
				if rating.timestamp >= cutoff_time
			]
		else:
			usage_records = self.usage_data.get(template_id, [])
			rating_records = self.rating_data.get(template_id, [])
		
		# Calculate usage metrics
		total_views = len([u for u in usage_records if u.usage_type == UsageTracking.VIEW])
		total_downloads = len([u for u in usage_records if u.usage_type == UsageTracking.DOWNLOAD])
		total_instantiations = len([u for u in usage_records if u.usage_type == UsageTracking.INSTANTIATE])
		
		# Calculate rating metrics
		if rating_records:
			average_rating = sum(r.rating for r in rating_records) / len(rating_records)
			rating_distribution = defaultdict(int)
			for rating in rating_records:
				rating_distribution[rating.rating] += 1
		else:
			average_rating = 0.0
			rating_distribution = {}
		
		# Calculate popularity score
		popularity_score = self._calculate_popularity_score(usage_records, rating_records)
		
		return {
			"template_id": template_id,
			"usage_metrics": {
				"total_views": total_views,
				"total_downloads": total_downloads,
				"total_instantiations": total_instantiations,
				"unique_users": len(set(u.user_id for u in usage_records)),
				"conversion_rate": total_instantiations / max(total_views, 1) * 100
			},
			"rating_metrics": {
				"average_rating": round(average_rating, 2),
				"total_ratings": len(rating_records),
				"rating_distribution": dict(rating_distribution)
			},
			"popularity_score": popularity_score,
			"trend_data": await self._calculate_usage_trends(usage_records)
		}
	
	def _calculate_popularity_score(
		self,
		usage_records: List[TemplateUsage],
		rating_records: List[TemplateRating]
	) -> float:
		"""Calculate template popularity score (0-100)."""
		# Base score from usage
		usage_score = min(50, len(usage_records) / 10)  # Max 50 points for usage
		
		# Rating score
		if rating_records:
			avg_rating = sum(r.rating for r in rating_records) / len(rating_records)
			rating_score = (avg_rating / 5.0) * 30  # Max 30 points for rating
		else:
			rating_score = 0
		
		# Recency score (recent usage gets higher score)
		recent_usage = [
			u for u in usage_records
			if u.timestamp >= datetime.utcnow() - timedelta(days=30)
		]
		recency_score = min(20, len(recent_usage) / 5)  # Max 20 points for recency
		
		return min(100, usage_score + rating_score + recency_score)
	
	async def _calculate_usage_trends(self, usage_records: List[TemplateUsage]) -> List[Dict[str, Any]]:
		"""Calculate usage trends over time."""
		# Group usage by day
		daily_usage = defaultdict(int)
		for usage in usage_records:
			day_key = usage.timestamp.date()
			daily_usage[day_key] += 1
		
		# Return last 30 days
		trend_data = []
		for i in range(30):
			date = datetime.utcnow().date() - timedelta(days=i)
			trend_data.append({
				"date": date.isoformat(),
				"usage_count": daily_usage.get(date, 0)
			})
		
		return list(reversed(trend_data))


# =============================================================================
# Template Recommendation Engine
# =============================================================================

class TemplateRecommendationEngine:
	"""AI-powered template recommendation system."""
	
	def __init__(self):
		self.user_profiles: Dict[str, Dict[str, Any]] = {}
		self.template_features: Dict[str, Dict[str, Any]] = {}
		self.recommendation_cache: Dict[str, List[TemplateRecommendation]] = {}
	
	async def generate_recommendations(
		self,
		user_id: str,
		context: APGTenantContext,
		limit: int = 10,
		category_filter: Optional[TemplateCategory] = None
	) -> List[TemplateRecommendation]:
		"""Generate personalized template recommendations."""
		try:
			# Get or create user profile
			user_profile = await self._get_user_profile(user_id, context)
			
			# Get available templates
			available_templates = await self._get_available_templates(context, category_filter)
			
			# Calculate recommendation scores
			recommendations = []
			for template_id, template_data in available_templates.items():
				score = await self._calculate_recommendation_score(
					user_profile, template_data, context
				)
				
				if score > 0.1:  # Minimum threshold
					reason = await self._generate_recommendation_reason(
						user_profile, template_data, score
					)
					
					recommendations.append(TemplateRecommendation(
						template_id=template_id,
						user_id=user_id,
						recommendation_score=score,
						recommendation_reason=reason,
						context_factors=self._extract_context_factors(user_profile, template_data),
						tenant_id=context.tenant_id
					))
			
			# Sort by score and limit
			recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
			
			return recommendations[:limit]
			
		except Exception as e:
			logger.error(f"Error generating recommendations: {e}")
			return []
	
	async def _get_user_profile(self, user_id: str, context: APGTenantContext) -> Dict[str, Any]:
		"""Get or create user profile for recommendations."""
		if user_id not in self.user_profiles:
			# Create default profile
			self.user_profiles[user_id] = {
				"preferred_categories": [],
				"complexity_preference": "moderate",
				"recent_templates": [],
				"skill_areas": [],
				"role": "analyst",
				"experience_level": "intermediate"
			}
		
		return self.user_profiles[user_id]
	
	async def _get_available_templates(
		self,
		context: APGTenantContext,
		category_filter: Optional[TemplateCategory] = None
	) -> Dict[str, Dict[str, Any]]:
		"""Get available templates with features."""
		# In production, this would query the template database
		# For demonstration, we'll create sample template data
		templates = {
			"template_1": {
				"name": "Employee Onboarding",
				"category": TemplateCategory.HR_PROCESS,
				"complexity": TemplateComplexity.MODERATE,
				"tags": ["hr", "onboarding", "employee"],
				"popularity_score": 85.0,
				"average_rating": 4.2,
				"recent_usage": 45
			},
			"template_2": {
				"name": "Invoice Approval",
				"category": TemplateCategory.FINANCE_PROCESS,
				"complexity": TemplateComplexity.SIMPLE,
				"tags": ["finance", "approval", "invoice"],
				"popularity_score": 92.0,
				"average_rating": 4.5,
				"recent_usage": 67
			},
			"template_3": {
				"name": "Software Development Lifecycle",
				"category": TemplateCategory.IT_PROCESS,
				"complexity": TemplateComplexity.COMPLEX,
				"tags": ["development", "software", "lifecycle"],
				"popularity_score": 78.0,
				"average_rating": 4.0,
				"recent_usage": 23
			}
		}
		
		if category_filter:
			templates = {
				tid: tdata for tid, tdata in templates.items()
				if tdata["category"] == category_filter
			}
		
		return templates
	
	async def _calculate_recommendation_score(
		self,
		user_profile: Dict[str, Any],
		template_data: Dict[str, Any],
		context: APGTenantContext
	) -> float:
		"""Calculate recommendation score for template."""
		score = 0.0
		
		# Category preference score (0.3 weight)
		preferred_categories = user_profile.get("preferred_categories", [])
		if template_data["category"] in preferred_categories:
			score += 0.3
		elif not preferred_categories:  # No strong preference
			score += 0.15
		
		# Complexity match score (0.2 weight)
		user_complexity_pref = user_profile.get("complexity_preference", "moderate")
		template_complexity = template_data["complexity"]
		
		complexity_match = {
			("simple", TemplateComplexity.SIMPLE): 1.0,
			("moderate", TemplateComplexity.MODERATE): 1.0,
			("complex", TemplateComplexity.COMPLEX): 1.0,
			("moderate", TemplateComplexity.SIMPLE): 0.8,
			("moderate", TemplateComplexity.COMPLEX): 0.8,
		}
		
		complexity_score = complexity_match.get((user_complexity_pref, template_complexity), 0.5)
		score += complexity_score * 0.2
		
		# Popularity and rating score (0.3 weight)
		popularity_normalized = template_data["popularity_score"] / 100.0
		rating_normalized = template_data["average_rating"] / 5.0
		quality_score = (popularity_normalized + rating_normalized) / 2
		score += quality_score * 0.3
		
		# Recent usage trend (0.2 weight)
		recent_usage = template_data["recent_usage"]
		usage_score = min(1.0, recent_usage / 50.0)  # Normalize to 50 as high usage
		score += usage_score * 0.2
		
		return min(1.0, score)
	
	async def _generate_recommendation_reason(
		self,
		user_profile: Dict[str, Any],
		template_data: Dict[str, Any],
		score: float
	) -> str:
		"""Generate human-readable recommendation reason."""
		reasons = []
		
		# Category match
		if template_data["category"] in user_profile.get("preferred_categories", []):
			reasons.append("matches your preferred category")
		
		# Popularity
		if template_data["popularity_score"] > 80:
			reasons.append("highly popular among users")
		
		# Rating
		if template_data["average_rating"] > 4.0:
			reasons.append("excellent user ratings")
		
		# Recent activity
		if template_data["recent_usage"] > 30:
			reasons.append("actively used by others")
		
		if not reasons:
			reasons = ["good match for your profile"]
		
		return f"Recommended because it {' and '.join(reasons[:2])}."
	
	def _extract_context_factors(
		self,
		user_profile: Dict[str, Any],
		template_data: Dict[str, Any]
	) -> List[str]:
		"""Extract context factors that influenced recommendation."""
		factors = []
		
		if template_data["category"] in user_profile.get("preferred_categories", []):
			factors.append("category_preference")
		
		if template_data["popularity_score"] > 80:
			factors.append("high_popularity")
		
		if template_data["average_rating"] > 4.0:
			factors.append("high_rating")
		
		if template_data["recent_usage"] > 30:
			factors.append("trending")
		
		return factors
	
	async def update_user_profile(
		self,
		user_id: str,
		template_usage: TemplateUsage,
		template_data: Dict[str, Any]
	) -> None:
		"""Update user profile based on template interaction."""
		if user_id not in self.user_profiles:
			self.user_profiles[user_id] = {
				"preferred_categories": [],
				"complexity_preference": "moderate",
				"recent_templates": [],
				"skill_areas": [],
				"role": "analyst",
				"experience_level": "intermediate"
			}
		
		profile = self.user_profiles[user_id]
		
		# Update preferred categories
		category = template_data.get("category")
		if category and category not in profile["preferred_categories"]:
			profile["preferred_categories"].append(category)
			
			# Keep only top 5 categories
			if len(profile["preferred_categories"]) > 5:
				profile["preferred_categories"] = profile["preferred_categories"][-5:]
		
		# Update recent templates
		profile["recent_templates"].insert(0, template_usage.template_id)
		profile["recent_templates"] = profile["recent_templates"][:10]  # Keep last 10


# =============================================================================
# Template Library Manager
# =============================================================================

class TemplateLibraryManager:
	"""Comprehensive template library management."""
	
	def __init__(self):
		self.templates: Dict[str, WBPMProcessTemplate] = {}
		self.collections: Dict[str, TemplateCollection] = {}
		self.analytics = TemplateAnalytics()
		self.recommendation_engine = TemplateRecommendationEngine()
		self.template_index = self._build_search_index()
	
	async def create_template(
		self,
		template_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create new process template."""
		try:
			# Create template
			template = WBPMProcessTemplate(
				tenant_id=context.tenant_id,
				template_name=template_data["template_name"],
				template_description=template_data.get("template_description", ""),
				category=TemplateCategory(template_data.get("category", TemplateCategory.CUSTOM)),
				tags=template_data.get("tags", []),
				template_bpmn=template_data["template_bpmn"],
				template_variables=template_data.get("template_variables", {}),
				complexity_level=TemplateComplexity(template_data.get("complexity_level", TemplateComplexity.MODERATE)),
				estimated_duration=template_data.get("estimated_duration"),
				required_roles=template_data.get("required_roles", []),
				scope=TemplateScope(template_data.get("scope", TemplateScope.PRIVATE)),
				created_by=context.user_id,
				updated_by=context.user_id
			)
			
			# Store template
			self.templates[template.id] = template
			
			# Update search index
			await self._update_search_index(template)
			
			# Track creation
			await self.analytics.track_usage(TemplateUsage(
				template_id=template.id,
				user_id=context.user_id,
				usage_type=UsageTracking.DOWNLOAD,
				context_data={"action": "create"},
				tenant_id=context.tenant_id
			))
			
			logger.info(f"Template created: {template.id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Template created successfully",
				data={
					"template_id": template.id,
					"template_name": template.template_name,
					"category": template.category,
					"scope": template.scope
				}
			)
			
		except Exception as e:
			logger.error(f"Error creating template: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create template: {e}",
				errors=[str(e)]
			)
	
	async def get_template(
		self,
		template_id: str,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Get template by ID."""
		try:
			template = self.templates.get(template_id)
			if not template:
				return WBPMServiceResponse(
					success=False,
					message="Template not found",
					errors=["Template not found"]
				)
			
			# Verify access permissions
			if not await self._check_template_access(template, context):
				return WBPMServiceResponse(
					success=False,
					message="Access denied to template",
					errors=["Access denied"]
				)
			
			# Track view
			await self.analytics.track_usage(TemplateUsage(
				template_id=template_id,
				user_id=context.user_id,
				usage_type=UsageTracking.VIEW,
				tenant_id=context.tenant_id
			))
			
			# Get analytics
			analytics_data = await self.analytics.get_template_analytics(template_id)
			
			return WBPMServiceResponse(
				success=True,
				message="Template retrieved successfully",
				data={
					"template": template.dict() if hasattr(template, 'dict') else template.__dict__,
					"analytics": analytics_data
				}
			)
			
		except Exception as e:
			logger.error(f"Error getting template {template_id}: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get template: {e}",
				errors=[str(e)]
			)
	
	async def search_templates(
		self,
		query: str,
		context: APGTenantContext,
		filters: Optional[Dict[str, Any]] = None,
		page: int = 1,
		page_size: int = 20
	) -> WBPMPagedResponse:
		"""Search templates with advanced filtering."""
		try:
			# Get accessible templates
			accessible_templates = [
				template for template in self.templates.values()
				if await self._check_template_access(template, context)
			]
			
			# Apply text search
			if query:
				search_results = await self._perform_text_search(query, accessible_templates)
			else:
				search_results = accessible_templates
			
			# Apply filters
			if filters:
				search_results = await self._apply_filters(search_results, filters)
			
			# Sort results
			sort_by = filters.get("sort_by", "relevance") if filters else "relevance"
			search_results = await self._sort_results(search_results, sort_by)
			
			# Pagination
			total_count = len(search_results)
			start_index = (page - 1) * page_size
			end_index = start_index + page_size
			page_results = search_results[start_index:end_index]
			
			# Convert to dict format with analytics
			result_data = []
			for template in page_results:
				analytics = await self.analytics.get_template_analytics(template.id)
				template_dict = template.dict() if hasattr(template, 'dict') else template.__dict__
				template_dict["analytics_summary"] = {
					"popularity_score": analytics["popularity_score"],
					"average_rating": analytics["rating_metrics"]["average_rating"],
					"usage_count": analytics["usage_metrics"]["total_views"]
				}
				result_data.append(template_dict)
			
			return WBPMPagedResponse(
				items=result_data,
				total_count=total_count,
				page=page,
				page_size=page_size,
				has_next=end_index < total_count,
				has_previous=page > 1
			)
			
		except Exception as e:
			logger.error(f"Error searching templates: {e}")
			return WBPMPagedResponse(
				items=[],
				total_count=0,
				page=page,
				page_size=page_size,
				has_next=False,
				has_previous=False
			)
	
	async def get_recommendations(
		self,
		context: APGTenantContext,
		category_filter: Optional[TemplateCategory] = None,
		limit: int = 10
	) -> WBPMServiceResponse:
		"""Get personalized template recommendations."""
		try:
			recommendations = await self.recommendation_engine.generate_recommendations(
				context.user_id, context, limit, category_filter
			)
			
			# Enrich with template data
			enriched_recommendations = []
			for rec in recommendations:
				template = self.templates.get(rec.template_id)
				if template:
					rec_data = {
						"recommendation": rec.__dict__,
						"template": template.dict() if hasattr(template, 'dict') else template.__dict__
					}
					enriched_recommendations.append(rec_data)
			
			return WBPMServiceResponse(
				success=True,
				message="Recommendations generated successfully",
				data={
					"recommendations": enriched_recommendations,
					"total_count": len(enriched_recommendations)
				}
			)
			
		except Exception as e:
			logger.error(f"Error getting recommendations: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get recommendations: {e}",
				errors=[str(e)]
			)
	
	async def instantiate_template(
		self,
		template_id: str,
		instance_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Instantiate process from template."""
		try:
			template = self.templates.get(template_id)
			if not template:
				return WBPMServiceResponse(
					success=False,
					message="Template not found",
					errors=["Template not found"]
				)
			
			# Verify access
			if not await self._check_template_access(template, context):
				return WBPMServiceResponse(
					success=False,
					message="Access denied to template",
					errors=["Access denied"]
				)
			
			# Create process instance from template
			process_instance_data = {
				"process_key": f"{template.template_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
				"process_name": instance_data.get("process_name", template.template_name),
				"process_description": instance_data.get("process_description", template.template_description),
				"bpmn_xml": template.template_bpmn,
				"category": template.category,
				"tags": template.tags + instance_data.get("additional_tags", [])
			}
			
			# In production, this would create actual process definition
			process_id = f"process_{uuid.uuid4().hex}"
			
			# Track instantiation
			await self.analytics.track_usage(TemplateUsage(
				template_id=template_id,
				user_id=context.user_id,
				usage_type=UsageTracking.INSTANTIATE,
				context_data={
					"process_id": process_id,
					"customizations": instance_data.get("customizations", {})
				},
				tenant_id=context.tenant_id
			))
			
			# Update recommendation engine
			template_data = {
				"category": template.category,
				"complexity": template.complexity_level,
				"tags": template.tags
			}
			await self.recommendation_engine.update_user_profile(
				context.user_id,
				TemplateUsage(
					template_id=template_id,
					user_id=context.user_id,
					usage_type=UsageTracking.INSTANTIATE,
					tenant_id=context.tenant_id
				),
				template_data
			)
			
			return WBPMServiceResponse(
				success=True,
				message="Process instantiated from template successfully",
				data={
					"process_id": process_id,
					"template_id": template_id,
					"instance_name": process_instance_data["process_name"]
				}
			)
			
		except Exception as e:
			logger.error(f"Error instantiating template {template_id}: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to instantiate template: {e}",
				errors=[str(e)]
			)
	
	async def rate_template(
		self,
		template_id: str,
		rating_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Rate and review template."""
		try:
			template = self.templates.get(template_id)
			if not template:
				return WBPMServiceResponse(
					success=False,
					message="Template not found",
					errors=["Template not found"]
				)
			
			# Create rating
			rating = TemplateRating(
				template_id=template_id,
				user_id=context.user_id,
				rating=rating_data["rating"],
				review=rating_data.get("review"),
				tenant_id=context.tenant_id
			)
			
			# Store rating
			await self.analytics.add_rating(rating)
			
			return WBPMServiceResponse(
				success=True,
				message="Template rated successfully",
				data={
					"rating_id": rating.rating_id,
					"rating": rating.rating
				}
			)
			
		except Exception as e:
			logger.error(f"Error rating template {template_id}: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to rate template: {e}",
				errors=[str(e)]
			)
	
	async def _check_template_access(
		self,
		template: WBPMProcessTemplate,
		context: APGTenantContext
	) -> bool:
		"""Check if user has access to template."""
		# Tenant check
		if template.tenant_id != context.tenant_id:
			return False
		
		# Scope check
		if template.scope == TemplateScope.PRIVATE:
			return template.created_by == context.user_id
		elif template.scope == TemplateScope.TEAM:
			# In production, check team membership
			return True  # Simplified for demo
		elif template.scope in [TemplateScope.ORGANIZATION, TemplateScope.PUBLIC]:
			return True
		
		return False
	
	async def _perform_text_search(
		self,
		query: str,
		templates: List[WBPMProcessTemplate]
	) -> List[WBPMProcessTemplate]:
		"""Perform text search on templates."""
		query_lower = query.lower()
		results = []
		
		for template in templates:
			score = 0
			
			# Name match (highest priority)
			if query_lower in template.template_name.lower():
				score += 10
			
			# Description match
			if template.template_description and query_lower in template.template_description.lower():
				score += 5
			
			# Tag match
			for tag in template.tags:
				if query_lower in tag.lower():
					score += 3
			
			# Category match
			if query_lower in template.category.value.lower():
				score += 2
			
			if score > 0:
				results.append(template)
		
		return results
	
	async def _apply_filters(
		self,
		templates: List[WBPMProcessTemplate],
		filters: Dict[str, Any]
	) -> List[WBPMProcessTemplate]:
		"""Apply search filters."""
		filtered = templates
		
		if "category" in filters:
			category = TemplateCategory(filters["category"])
			filtered = [t for t in filtered if t.category == category]
		
		if "complexity" in filters:
			complexity = TemplateComplexity(filters["complexity"])
			filtered = [t for t in filtered if t.complexity_level == complexity]
		
		if "scope" in filters:
			scope = TemplateScope(filters["scope"])
			filtered = [t for t in filtered if t.scope == scope]
		
		if "tags" in filters:
			required_tags = filters["tags"]
			if isinstance(required_tags, str):
				required_tags = [required_tags]
			filtered = [
				t for t in filtered
				if any(tag in t.tags for tag in required_tags)
			]
		
		if "min_rating" in filters:
			min_rating = float(filters["min_rating"])
			# This would need to join with rating data in production
			filtered = [t for t in filtered]  # Placeholder
		
		return filtered
	
	async def _sort_results(
		self,
		templates: List[WBPMProcessTemplate],
		sort_by: str
	) -> List[WBPMProcessTemplate]:
		"""Sort search results."""
		if sort_by == "name":
			return sorted(templates, key=lambda t: t.template_name.lower())
		elif sort_by == "created_date":
			return sorted(templates, key=lambda t: t.created_at, reverse=True)
		elif sort_by == "updated_date":
			return sorted(templates, key=lambda t: t.updated_at, reverse=True)
		elif sort_by == "popularity":
			# This would need analytics data in production
			return templates
		else:  # relevance (default)
			return templates
	
	def _build_search_index(self) -> Dict[str, Any]:
		"""Build search index for templates."""
		return {
			"name_index": {},
			"tag_index": {},
			"category_index": {},
			"full_text_index": {}
		}
	
	async def _update_search_index(self, template: WBPMProcessTemplate) -> None:
		"""Update search index with new template."""
		# In production, this would update a proper search index
		# For demonstration, we'll just log the update
		logger.debug(f"Search index updated for template {template.id}")


# =============================================================================
# Service Factory
# =============================================================================

def create_template_library_manager() -> TemplateLibraryManager:
	"""Create and configure template library manager."""
	manager = TemplateLibraryManager()
	logger.info("Template library manager created and configured")
	return manager


# Export main classes
__all__ = [
	'TemplateLibraryManager',
	'TemplateAnalytics',
	'TemplateRecommendationEngine',
	'TemplateMetadata',
	'TemplateUsage',
	'TemplateRating',
	'TemplateRecommendation',
	'TemplateCollection',
	'TemplateCategory',
	'TemplateScope',
	'TemplateComplexity',
	'UsageTracking',
	'create_template_library_manager'
]