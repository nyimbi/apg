"""
Advanced Template System for Multi-Tenant Management

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Comprehensive template system with inheritance, versioning, marketplace,
and composition capabilities for enterprise-grade tenant management.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from uuid_extensions import uuid7str

from .models import (
	TenantTier, CloudProvider, ResourceAllocation, 
	TenantConfiguration, TenantTemplate
)


class TemplateType(str, Enum):
	"""Types of tenant templates"""
	BASE = "base"
	APPLICATION = "application"
	INFRASTRUCTURE = "infrastructure"
	SECURITY = "security"
	COMPLIANCE = "compliance"
	INDUSTRY = "industry"
	CUSTOM = "custom"


class TemplateStatus(str, Enum):
	"""Template lifecycle status"""
	DRAFT = "draft"
	TESTING = "testing"
	PUBLISHED = "published"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"


class TemplateCategory(str, Enum):
	"""Template categorization"""
	WEB_APPLICATION = "web_application"
	MICROSERVICES = "microservices"
	DATA_ANALYTICS = "data_analytics"
	AI_ML_PLATFORM = "ai_ml_platform"
	E_COMMERCE = "e_commerce"
	HEALTHCARE = "healthcare"
	FINTECH = "fintech"
	ENTERPRISE_SaaS = "enterprise_saas"
	DEVELOPMENT = "development"
	COMPLIANCE = "compliance"


class VersioningStrategy(str, Enum):
	"""Template versioning strategies"""
	SEMANTIC = "semantic"  # Major.Minor.Patch
	DATE_BASED = "date_based"  # YYYY.MM.DD
	SEQUENTIAL = "sequential"  # v1, v2, v3
	CUSTOM = "custom"


@dataclass
class TemplateVersion:
	"""Template version information"""
	version: str
	created_at: datetime
	created_by: str
	changelog: List[str]
	breaking_changes: List[str] = field(default_factory=list)
	deprecated_features: List[str] = field(default_factory=list)
	migration_notes: str = ""
	compatibility: Dict[str, str] = field(default_factory=dict)


@dataclass
class TemplateMetadata:
	"""Comprehensive template metadata"""
	id: str
	name: str
	display_name: str
	description: str
	template_type: TemplateType
	category: TemplateCategory
	status: TemplateStatus
	
	# Versioning
	current_version: str
	versions: List[TemplateVersion]
	versioning_strategy: VersioningStrategy = VersioningStrategy.SEMANTIC
	
	# Inheritance
	parent_template_id: Optional[str] = None
	child_templates: List[str] = field(default_factory=list)
	inheritance_depth: int = 0
	
	# Marketplace
	author: str = "System"
	organization: Optional[str] = None
	license: str = "MIT"
	homepage_url: Optional[str] = None
	documentation_url: Optional[str] = None
	support_url: Optional[str] = None
	
	# Ratings and usage
	rating: float = 0.0
	rating_count: int = 0
	download_count: int = 0
	usage_count: int = 0
	
	# Constraints and requirements
	min_tier: TenantTier = TenantTier.FREE
	supported_clouds: List[CloudProvider] = field(default_factory=list)
	required_features: List[str] = field(default_factory=list)
	optional_features: List[str] = field(default_factory=list)
	
	# Timestamps
	created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
	updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
	published_at: Optional[datetime] = None


@dataclass
class TemplateConfiguration:
	"""Template configuration with inheritance support"""
	base_configuration: Dict[str, Any]
	overrides: Dict[str, Any] = field(default_factory=dict)
	computed_configuration: Optional[Dict[str, Any]] = None
	
	def resolve_configuration(self, parent_config: Optional['TemplateConfiguration'] = None) -> Dict[str, Any]:
		"""Resolve final configuration with inheritance"""
		if self.computed_configuration:
			return self.computed_configuration
		
		# Start with base configuration
		resolved = self.base_configuration.copy()
		
		# Apply parent configuration if available
		if parent_config:
			parent_resolved = parent_config.resolve_configuration()
			resolved = self._deep_merge(parent_resolved, resolved)
		
		# Apply local overrides
		resolved = self._deep_merge(resolved, self.overrides)
		
		# Cache computed result
		self.computed_configuration = resolved
		return resolved
	
	def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
		"""Deep merge two dictionaries"""
		result = base.copy()
		
		for key, value in override.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = self._deep_merge(result[key], value)
			else:
				result[key] = value
		
		return result


@dataclass
class EnterpriseTemplate:
	"""Comprehensive enterprise template with all features"""
	metadata: TemplateMetadata
	configuration: TemplateConfiguration
	resource_allocation: ResourceAllocation
	
	# Advanced features
	pre_deployment_hooks: List[str] = field(default_factory=list)
	post_deployment_hooks: List[str] = field(default_factory=list)
	health_checks: List[Dict[str, Any]] = field(default_factory=list)
	monitoring_rules: List[Dict[str, Any]] = field(default_factory=list)
	
	# Composition support
	composed_templates: List[str] = field(default_factory=list)
	dependencies: List[str] = field(default_factory=list)
	conflicts: List[str] = field(default_factory=list)
	
	def get_effective_configuration(self, parent_template: Optional['EnterpriseTemplate'] = None) -> Dict[str, Any]:
		"""Get effective configuration with inheritance resolved"""
		parent_config = parent_template.configuration if parent_template else None
		return self.configuration.resolve_configuration(parent_config)
	
	def validate_compatibility(self, tenant_tier: TenantTier, cloud_provider: CloudProvider) -> List[str]:
		"""Validate template compatibility with tenant requirements"""
		issues = []
		
		# Check tier requirements
		tier_order = {TenantTier.FREE: 0, TenantTier.STANDARD: 1, TenantTier.PREMIUM: 2, TenantTier.ENTERPRISE: 3}
		if tier_order[tenant_tier] < tier_order[self.metadata.min_tier]:
			issues.append(f"Template requires {self.metadata.min_tier.value} tier or higher, got {tenant_tier.value}")
		
		# Check cloud provider support
		if self.metadata.supported_clouds and cloud_provider not in self.metadata.supported_clouds:
			supported = [cp.value for cp in self.metadata.supported_clouds]
			issues.append(f"Template not supported on {cloud_provider.value}, supports: {', '.join(supported)}")
		
		return issues


class TemplateRepository:
	"""Template storage and retrieval interface"""
	
	@abstractmethod
	async def save_template(self, template: EnterpriseTemplate) -> str:
		"""Save template and return ID"""
		pass
	
	@abstractmethod
	async def get_template(self, template_id: str, version: Optional[str] = None) -> Optional[EnterpriseTemplate]:
		"""Get template by ID and optional version"""
		pass
	
	@abstractmethod
	async def list_templates(
		self,
		template_type: Optional[TemplateType] = None,
		category: Optional[TemplateCategory] = None,
		status: TemplateStatus = TemplateStatus.PUBLISHED
	) -> List[EnterpriseTemplate]:
		"""List templates with optional filtering"""
		pass
	
	@abstractmethod
	async def delete_template(self, template_id: str) -> bool:
		"""Delete template"""
		pass


class InMemoryTemplateRepository(TemplateRepository):
	"""In-memory template repository for development/testing"""
	
	def __init__(self):
		self._templates: Dict[str, EnterpriseTemplate] = {}
		self._template_versions: Dict[str, Dict[str, EnterpriseTemplate]] = {}
	
	async def save_template(self, template: EnterpriseTemplate) -> str:
		"""Save template to memory"""
		template_id = template.metadata.id
		version = template.metadata.current_version
		
		# Store main template
		self._templates[template_id] = template
		
		# Store versioned template
		if template_id not in self._template_versions:
			self._template_versions[template_id] = {}
		self._template_versions[template_id][version] = template
		
		return template_id
	
	async def get_template(self, template_id: str, version: Optional[str] = None) -> Optional[EnterpriseTemplate]:
		"""Get template from memory"""
		if version:
			return self._template_versions.get(template_id, {}).get(version)
		return self._templates.get(template_id)
	
	async def list_templates(
		self,
		template_type: Optional[TemplateType] = None,
		category: Optional[TemplateCategory] = None,
		status: TemplateStatus = TemplateStatus.PUBLISHED
	) -> List[EnterpriseTemplate]:
		"""List templates with filtering"""
		templates = []
		
		for template in self._templates.values():
			# Apply filters
			if template_type and template.metadata.template_type != template_type:
				continue
			if category and template.metadata.category != category:
				continue
			if status and template.metadata.status != status:
				continue
			
			templates.append(template)
		
		# Sort by rating and usage
		templates.sort(key=lambda t: (t.metadata.rating, t.metadata.usage_count), reverse=True)
		return templates
	
	async def delete_template(self, template_id: str) -> bool:
		"""Delete template from memory"""
		if template_id in self._templates:
			del self._templates[template_id]
			if template_id in self._template_versions:
				del self._template_versions[template_id]
			return True
		return False


class TemplateInheritanceResolver:
	"""Resolves template inheritance chains"""
	
	def __init__(self, repository: TemplateRepository):
		self._repository = repository
	
	async def resolve_inheritance_chain(self, template: EnterpriseTemplate) -> List[EnterpriseTemplate]:
		"""Resolve complete inheritance chain from root to template"""
		chain = []
		current_template = template
		visited = set()
		
		while current_template:
			# Prevent infinite loops
			if current_template.metadata.id in visited:
				break
			
			visited.add(current_template.metadata.id)
			chain.insert(0, current_template)  # Insert at beginning for root-first order
			
			# Get parent template
			if current_template.metadata.parent_template_id:
				current_template = await self._repository.get_template(current_template.metadata.parent_template_id)
			else:
				break
		
		return chain
	
	async def get_effective_configuration(self, template: EnterpriseTemplate) -> Dict[str, Any]:
		"""Get template configuration with full inheritance resolved"""
		chain = await self.resolve_inheritance_chain(template)
		
		# Merge configurations from root to leaf
		effective_config = {}
		for tmpl in chain:
			config = tmpl.configuration.resolve_configuration()
			effective_config = template.configuration._deep_merge(effective_config, config)
		
		return effective_config
	
	async def validate_inheritance_depth(self, template: EnterpriseTemplate, max_depth: int = 10) -> List[str]:
		"""Validate inheritance depth doesn't exceed limits"""
		chain = await self.resolve_inheritance_chain(template)
		issues = []
		
		if len(chain) > max_depth:
			issues.append(f"Inheritance depth {len(chain)} exceeds maximum {max_depth}")
		
		return issues


class TemplateComposer:
	"""Composes multiple templates into unified configurations"""
	
	def __init__(self, repository: TemplateRepository):
		self._repository = repository
		self._inheritance_resolver = TemplateInheritanceResolver(repository)
	
	async def compose_templates(self, template_ids: List[str]) -> Dict[str, Any]:
		"""Compose multiple templates into unified configuration"""
		composed_config = {}
		template_order = []
		
		# Get all templates
		templates = []
		for template_id in template_ids:
			template = await self._repository.get_template(template_id)
			if template:
				templates.append(template)
		
		# Validate composition compatibility
		conflicts = await self._validate_composition_compatibility(templates)
		if conflicts:
			raise ValueError(f"Template composition conflicts: {', '.join(conflicts)}")
		
		# Sort templates by dependency order
		ordered_templates = await self._resolve_composition_order(templates)
		
		# Compose configurations
		for template in ordered_templates:
			effective_config = await self._inheritance_resolver.get_effective_configuration(template)
			composed_config = self._merge_compositions(composed_config, effective_config, template)
			template_order.append(template.metadata.id)
		
		return {
			"composed_configuration": composed_config,
			"template_order": template_order,
			"composition_metadata": {
				"composed_at": datetime.now(UTC).isoformat(),
				"template_count": len(templates),
				"total_features": len(self._extract_features(composed_config))
			}
		}
	
	async def _validate_composition_compatibility(self, templates: List[EnterpriseTemplate]) -> List[str]:
		"""Validate templates can be composed together"""
		conflicts = []
		
		for i, template1 in enumerate(templates):
			for template2 in templates[i+1:]:
				# Check explicit conflicts
				if template1.metadata.id in template2.conflicts:
					conflicts.append(f"{template1.metadata.name} conflicts with {template2.metadata.name}")
				
				# Check resource conflicts (simplified)
				if (template1.resource_allocation.cpu_cores + template2.resource_allocation.cpu_cores > 64):
					conflicts.append(f"Combined CPU requirements exceed limits")
		
		return conflicts
	
	async def _resolve_composition_order(self, templates: List[EnterpriseTemplate]) -> List[EnterpriseTemplate]:
		"""Resolve template application order based on dependencies"""
		ordered = []
		remaining = templates.copy()
		template_map = {t.metadata.id: t for t in templates}
		
		while remaining:
			# Find templates with no unresolved dependencies
			ready = []
			for template in remaining:
				dependencies_met = True
				for dep_id in template.dependencies:
					if dep_id in template_map and dep_id not in [t.metadata.id for t in ordered]:
						dependencies_met = False
						break
				
				if dependencies_met:
					ready.append(template)
			
			if not ready:
				# Circular dependency or missing dependency
				ready = remaining[:1]  # Take first remaining to avoid infinite loop
			
			for template in ready:
				ordered.append(template)
				remaining.remove(template)
		
		return ordered
	
	def _merge_compositions(
		self,
		base_config: Dict[str, Any],
		template_config: Dict[str, Any],
		template: EnterpriseTemplate
	) -> Dict[str, Any]:
		"""Merge template configuration into base with composition rules"""
		# For now, use simple deep merge
		# In production, would implement sophisticated merge strategies
		return self._deep_merge(base_config, template_config)
	
	def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
		"""Deep merge dictionaries"""
		result = base.copy()
		
		for key, value in override.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = self._deep_merge(result[key], value)
			else:
				result[key] = value
		
		return result
	
	def _extract_features(self, config: Dict[str, Any]) -> Set[str]:
		"""Extract feature list from configuration"""
		features = set()
		
		def extract_recursive(obj, prefix=""):
			if isinstance(obj, dict):
				for key, value in obj.items():
					feature_key = f"{prefix}.{key}" if prefix else key
					features.add(feature_key)
					extract_recursive(value, feature_key)
		
		extract_recursive(config)
		return features


class TemplateMarketplace:
	"""Template marketplace with rating, discovery, and sharing"""
	
	def __init__(self, repository: TemplateRepository):
		self._repository = repository
		self._user_ratings: Dict[str, Dict[str, int]] = {}  # template_id -> user_id -> rating
		self._featured_templates: List[str] = []
		self._trending_templates: List[str] = []
	
	async def search_templates(
		self,
		query: str = "",
		template_type: Optional[TemplateType] = None,
		category: Optional[TemplateCategory] = None,
		min_rating: float = 0.0,
		sort_by: str = "rating",
		limit: int = 50
	) -> List[EnterpriseTemplate]:
		"""Search templates in marketplace"""
		templates = await self._repository.list_templates(template_type, category, TemplateStatus.PUBLISHED)
		
		# Filter by query
		if query:
			query_lower = query.lower()
			templates = [
				t for t in templates
				if (query_lower in t.metadata.name.lower() or
				    query_lower in t.metadata.description.lower())
			]
		
		# Filter by minimum rating
		templates = [t for t in templates if t.metadata.rating >= min_rating]
		
		# Sort templates
		if sort_by == "rating":
			templates.sort(key=lambda t: (t.metadata.rating, t.metadata.rating_count), reverse=True)
		elif sort_by == "popularity":
			templates.sort(key=lambda t: t.metadata.usage_count, reverse=True)
		elif sort_by == "newest":
			templates.sort(key=lambda t: t.metadata.created_at, reverse=True)
		elif sort_by == "name":
			templates.sort(key=lambda t: t.metadata.name)
		
		return templates[:limit]
	
	async def get_featured_templates(self) -> List[EnterpriseTemplate]:
		"""Get featured templates"""
		templates = []
		for template_id in self._featured_templates:
			template = await self._repository.get_template(template_id)
			if template:
				templates.append(template)
		return templates
	
	async def get_trending_templates(self, days: int = 7) -> List[EnterpriseTemplate]:
		"""Get trending templates based on recent usage"""
		all_templates = await self._repository.list_templates(status=TemplateStatus.PUBLISHED)
		
		# Sort by recent usage (simplified - in production would track actual usage metrics)
		trending = sorted(all_templates, key=lambda t: t.metadata.download_count, reverse=True)
		return trending[:10]
	
	async def rate_template(self, template_id: str, user_id: str, rating: int) -> bool:
		"""Rate a template (1-5 stars)"""
		if not (1 <= rating <= 5):
			return False
		
		template = await self._repository.get_template(template_id)
		if not template:
			return False
		
		# Store user rating
		if template_id not in self._user_ratings:
			self._user_ratings[template_id] = {}
		
		old_rating = self._user_ratings[template_id].get(user_id)
		self._user_ratings[template_id][user_id] = rating
		
		# Update template average rating
		user_ratings = list(self._user_ratings[template_id].values())
		template.metadata.rating = sum(user_ratings) / len(user_ratings)
		template.metadata.rating_count = len(user_ratings)
		
		# Save updated template
		await self._repository.save_template(template)
		
		return True
	
	async def get_template_stats(self, template_id: str) -> Dict[str, Any]:
		"""Get comprehensive template statistics"""
		template = await self._repository.get_template(template_id)
		if not template:
			return {}
		
		user_ratings = self._user_ratings.get(template_id, {})
		rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		
		for rating in user_ratings.values():
			rating_distribution[rating] += 1
		
		return {
			"template_id": template_id,
			"name": template.metadata.name,
			"average_rating": template.metadata.rating,
			"total_ratings": template.metadata.rating_count,
			"rating_distribution": rating_distribution,
			"download_count": template.metadata.download_count,
			"usage_count": template.metadata.usage_count,
			"created_at": template.metadata.created_at.isoformat(),
			"last_updated": template.metadata.updated_at.isoformat(),
			"author": template.metadata.author,
			"category": template.metadata.category.value,
			"template_type": template.metadata.template_type.value
		}


class AdvancedTemplateSystem:
	"""
	Advanced template system with inheritance, versioning, marketplace, and composition
	
	Provides enterprise-grade template management with sophisticated features
	for complex multi-tenant deployments.
	"""
	
	def __init__(self, repository: Optional[TemplateRepository] = None):
		self._repository = repository or InMemoryTemplateRepository()
		self._inheritance_resolver = TemplateInheritanceResolver(self._repository)
		self._composer = TemplateComposer(self._repository)
		self._marketplace = TemplateMarketplace(self._repository)
		
		# Built-in templates
		self._builtin_templates_loaded = False
	
	def _log_template_operation(self, operation: str, template_id: str = None) -> str:
		"""Log template operations"""
		template_info = f" for template {template_id}" if template_id else ""
		return f"[Template] {operation}{template_info}"
	
	async def initialize_builtin_templates(self) -> None:
		"""Initialize built-in template library"""
		if self._builtin_templates_loaded:
			return
		
		builtin_templates = await self._create_builtin_templates()
		
		for template in builtin_templates:
			await self._repository.save_template(template)
		
		self._builtin_templates_loaded = True
		print(self._log_template_operation(f"Loaded {len(builtin_templates)} built-in templates"))
	
	async def create_template(
		self,
		name: str,
		template_type: TemplateType,
		category: TemplateCategory,
		configuration: Dict[str, Any],
		resource_allocation: ResourceAllocation,
		parent_template_id: Optional[str] = None,
		author: str = "System",
		description: str = ""
	) -> EnterpriseTemplate:
		"""Create a new template"""
		
		# Validate parent template if specified
		parent_template = None
		if parent_template_id:
			parent_template = await self._repository.get_template(parent_template_id)
			if not parent_template:
				raise ValueError(f"Parent template {parent_template_id} not found")
		
		# Create template metadata
		template_id = uuid7str()
		metadata = TemplateMetadata(
			id=template_id,
			name=name,
			display_name=name.replace("_", " ").title(),
			description=description,
			template_type=template_type,
			category=category,
			status=TemplateStatus.DRAFT,
			current_version="1.0.0",
			versions=[TemplateVersion(
				version="1.0.0",
				created_at=datetime.now(UTC),
				created_by=author,
				changelog=["Initial template creation"]
			)],
			parent_template_id=parent_template_id,
			inheritance_depth=parent_template.metadata.inheritance_depth + 1 if parent_template else 0,
			author=author
		)
		
		# Create template configuration
		template_config = TemplateConfiguration(
			base_configuration=configuration
		)
		
		# Create enterprise template
		template = EnterpriseTemplate(
			metadata=metadata,
			configuration=template_config,
			resource_allocation=resource_allocation
		)
		
		# Save template
		await self._repository.save_template(template)
		
		print(self._log_template_operation("Template created", template_id))
		return template
	
	async def update_template(
		self,
		template_id: str,
		updates: Dict[str, Any],
		version_increment: str = "patch"
	) -> EnterpriseTemplate:
		"""Update existing template with versioning"""
		template = await self._repository.get_template(template_id)
		if not template:
			raise ValueError(f"Template {template_id} not found")
		
		# Create new version
		old_version = template.metadata.current_version
		new_version = self._increment_version(old_version, version_increment)
		
		# Apply updates
		for key, value in updates.items():
			if key == "configuration":
				template.configuration.base_configuration.update(value)
			elif key == "resource_allocation":
				for attr, val in value.items():
					if hasattr(template.resource_allocation, attr):
						setattr(template.resource_allocation, attr, val)
			elif hasattr(template.metadata, key):
				setattr(template.metadata, key, value)
		
		# Update version information
		template.metadata.current_version = new_version
		template.metadata.updated_at = datetime.now(UTC)
		template.metadata.versions.append(TemplateVersion(
			version=new_version,
			created_at=datetime.now(UTC),
			created_by=updates.get("updated_by", "System"),
			changelog=updates.get("changelog", ["Template updated"])
		))
		
		# Save updated template
		await self._repository.save_template(template)
		
		print(self._log_template_operation(f"Template updated to version {new_version}", template_id))
		return template
	
	async def get_template(self, template_id: str, version: Optional[str] = None) -> Optional[EnterpriseTemplate]:
		"""Get template by ID and optional version"""
		return await self._repository.get_template(template_id, version)
	
	async def list_templates(
		self,
		template_type: Optional[TemplateType] = None,
		category: Optional[TemplateCategory] = None,
		status: TemplateStatus = TemplateStatus.PUBLISHED
	) -> List[EnterpriseTemplate]:
		"""List templates with filtering"""
		return await self._repository.list_templates(template_type, category, status)
	
	async def inherit_template(
		self,
		name: str,
		parent_template_id: str,
		overrides: Dict[str, Any],
		author: str = "System"
	) -> EnterpriseTemplate:
		"""Create template inheriting from parent"""
		parent_template = await self._repository.get_template(parent_template_id)
		if not parent_template:
			raise ValueError(f"Parent template {parent_template_id} not found")
		
		# Validate inheritance depth
		issues = await self._inheritance_resolver.validate_inheritance_depth(parent_template)
		if issues:
			raise ValueError(f"Inheritance validation failed: {', '.join(issues)}")
		
		# Create child template
		child_template = await self.create_template(
			name=name,
			template_type=parent_template.metadata.template_type,
			category=parent_template.metadata.category,
			configuration=overrides,
			resource_allocation=parent_template.resource_allocation,
			parent_template_id=parent_template_id,
			author=author,
			description=f"Inherits from {parent_template.metadata.name}"
		)
		
		# Update parent's child list
		parent_template.metadata.child_templates.append(child_template.metadata.id)
		await self._repository.save_template(parent_template)
		
		return child_template
	
	async def compose_templates(self, template_ids: List[str], name: str, author: str = "System") -> EnterpriseTemplate:
		"""Compose multiple templates into a new template"""
		composition_result = await self._composer.compose_templates(template_ids)
		
		# Create composed template
		composed_template = await self.create_template(
			name=name,
			template_type=TemplateType.CUSTOM,
			category=TemplateCategory.ENTERPRISE_SaaS,
			configuration=composition_result["composed_configuration"],
			resource_allocation=ResourceAllocation(
				cpu_cores=4,
				memory_gb=8,
				storage_gb=100,
				bandwidth_mbps=1000,
				database_connections=50
			),
			author=author,
			description=f"Composed from {len(template_ids)} templates"
		)
		
		# Store composition metadata
		composed_template.composed_templates = template_ids
		await self._repository.save_template(composed_template)
		
		return composed_template
	
	async def publish_template(self, template_id: str) -> bool:
		"""Publish template to marketplace"""
		template = await self._repository.get_template(template_id)
		if not template:
			return False
		
		template.metadata.status = TemplateStatus.PUBLISHED
		template.metadata.published_at = datetime.now(UTC)
		await self._repository.save_template(template)
		
		print(self._log_template_operation("Template published to marketplace", template_id))
		return True
	
	async def search_marketplace(
		self,
		query: str = "",
		template_type: Optional[TemplateType] = None,
		category: Optional[TemplateCategory] = None,
		min_rating: float = 0.0,
		sort_by: str = "rating"
	) -> List[EnterpriseTemplate]:
		"""Search templates in marketplace"""
		return await self._marketplace.search_templates(query, template_type, category, min_rating, sort_by)
	
	async def get_effective_configuration(self, template_id: str) -> Dict[str, Any]:
		"""Get template configuration with inheritance resolved"""
		template = await self._repository.get_template(template_id)
		if not template:
			return {}
		
		return await self._inheritance_resolver.get_effective_configuration(template)
	
	def _increment_version(self, current_version: str, increment_type: str) -> str:
		"""Increment semantic version"""
		try:
			parts = current_version.split(".")
			major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
			
			if increment_type == "major":
				major += 1
				minor = 0
				patch = 0
			elif increment_type == "minor":
				minor += 1
				patch = 0
			else:  # patch
				patch += 1
			
			return f"{major}.{minor}.{patch}"
		except:
			# Fallback to sequential
			return f"{current_version}.1"
	
	async def _create_builtin_templates(self) -> List[EnterpriseTemplate]:
		"""Create built-in template library"""
		templates = []
		
		# Base Web Application Template
		web_app_template = await self._create_web_application_template()
		templates.append(web_app_template)
		
		# Microservices Template  
		microservices_template = await self._create_microservices_template()
		templates.append(microservices_template)
		
		# Data Analytics Template
		analytics_template = await self._create_data_analytics_template()
		templates.append(analytics_template)
		
		# AI/ML Platform Template
		aiml_template = await self._create_aiml_template()
		templates.append(aiml_template)
		
		# Healthcare Compliance Template
		healthcare_template = await self._create_healthcare_template()
		templates.append(healthcare_template)
		
		return templates
	
	async def _create_web_application_template(self) -> EnterpriseTemplate:
		"""Create base web application template"""
		metadata = TemplateMetadata(
			id="builtin-web-app",
			name="web_application_basic",
			display_name="Basic Web Application",
			description="Standard web application with database, caching, and monitoring",
			template_type=TemplateType.APPLICATION,
			category=TemplateCategory.WEB_APPLICATION,
			status=TemplateStatus.PUBLISHED,
			current_version="1.0.0",
			versions=[TemplateVersion(
				version="1.0.0",
				created_at=datetime.now(UTC),
				created_by="System",
				changelog=["Initial web application template"]
			)],
			supported_clouds=[CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP],
			min_tier=TenantTier.STANDARD
		)
		
		configuration = TemplateConfiguration(
			base_configuration={
				"application": {
					"framework": "react",
					"backend": "node.js",
					"database": "postgresql",
					"cache": "redis",
					"storage": "s3_compatible"
				},
				"infrastructure": {
					"load_balancer": True,
					"auto_scaling": True,
					"monitoring": True,
					"logging": True
				},
				"security": {
					"https_only": True,
					"cors_enabled": True,
					"rate_limiting": True,
					"auth_required": True
				}
			}
		)
		
		resource_allocation = ResourceAllocation(
			cpu_cores=2,
			memory_gb=4,
			storage_gb=50,
			bandwidth_mbps=100,
			database_connections=25
		)
		
		return EnterpriseTemplate(
			metadata=metadata,
			configuration=configuration,
			resource_allocation=resource_allocation
		)
	
	async def _create_microservices_template(self) -> EnterpriseTemplate:
		"""Create microservices template"""
		metadata = TemplateMetadata(
			id="builtin-microservices",
			name="microservices_platform",
			display_name="Microservices Platform",
			description="Kubernetes-based microservices platform with service mesh",
			template_type=TemplateType.INFRASTRUCTURE,
			category=TemplateCategory.MICROSERVICES,
			status=TemplateStatus.PUBLISHED,
			current_version="1.0.0",
			versions=[TemplateVersion(
				version="1.0.0",
				created_at=datetime.now(UTC),
				created_by="System",
				changelog=["Initial microservices template"]
			)],
			supported_clouds=[CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP],
			min_tier=TenantTier.PREMIUM
		)
		
		configuration = TemplateConfiguration(
			base_configuration={
				"orchestration": {
					"platform": "kubernetes",
					"service_mesh": "istio",
					"ingress": "nginx",
					"registry": "docker_hub"
				},
				"services": {
					"api_gateway": True,
					"config_service": True,
					"discovery_service": True,
					"monitoring_service": True
				},
				"observability": {
					"distributed_tracing": True,
					"metrics_collection": True,
					"log_aggregation": True,
					"health_checks": True
				}
			}
		)
		
		resource_allocation = ResourceAllocation(
			cpu_cores=8,
			memory_gb=16,
			storage_gb=200,
			bandwidth_mbps=500,
			database_connections=100
		)
		
		return EnterpriseTemplate(
			metadata=metadata,
			configuration=configuration,
			resource_allocation=resource_allocation
		)
	
	async def _create_data_analytics_template(self) -> EnterpriseTemplate:
		"""Create data analytics template"""
		metadata = TemplateMetadata(
			id="builtin-data-analytics",
			name="data_analytics_platform",
			display_name="Data Analytics Platform",
			description="Big data analytics platform with real-time processing",
			template_type=TemplateType.APPLICATION,
			category=TemplateCategory.DATA_ANALYTICS,
			status=TemplateStatus.PUBLISHED,
			current_version="1.0.0",
			versions=[TemplateVersion(
				version="1.0.0",
				created_at=datetime.now(UTC),
				created_by="System",
				changelog=["Initial data analytics template"]
			)],
			min_tier=TenantTier.ENTERPRISE
		)
		
		configuration = TemplateConfiguration(
			base_configuration={
				"data_processing": {
					"streaming": "apache_kafka",
					"batch": "apache_spark",
					"storage": "data_lake",
					"warehouse": "clickhouse"
				},
				"visualization": {
					"dashboards": "grafana",
					"notebooks": "jupyter",
					"reporting": "custom_api"
				}
			}
		)
		
		resource_allocation = ResourceAllocation(
			cpu_cores=16,
			memory_gb=64,
			storage_gb=1000,
			bandwidth_mbps=1000,
			database_connections=200
		)
		
		return EnterpriseTemplate(
			metadata=metadata,
			configuration=configuration,
			resource_allocation=resource_allocation
		)
	
	async def _create_aiml_template(self) -> EnterpriseTemplate:
		"""Create AI/ML template"""
		metadata = TemplateMetadata(
			id="builtin-aiml-platform",
			name="aiml_platform",
			display_name="AI/ML Platform",
			description="Machine learning platform with model training and inference",
			template_type=TemplateType.APPLICATION,
			category=TemplateCategory.AI_ML_PLATFORM,
			status=TemplateStatus.PUBLISHED,
			current_version="1.0.0",
			versions=[TemplateVersion(
				version="1.0.0",
				created_at=datetime.now(UTC),
				created_by="System",
				changelog=["Initial AI/ML platform template"]
			)],
			min_tier=TenantTier.ENTERPRISE
		)
		
		configuration = TemplateConfiguration(
			base_configuration={
				"ml_platform": {
					"framework": "pytorch",
					"training": "distributed",
					"inference": "model_serving",
					"data": "feature_store"
				},
				"infrastructure": {
					"gpu_support": True,
					"auto_scaling": True,
					"model_registry": True,
					"experiment_tracking": True
				}
			}
		)
		
		resource_allocation = ResourceAllocation(
			cpu_cores=32,
			memory_gb=128,
			storage_gb=2000,
			bandwidth_mbps=2000,
			database_connections=150
		)
		
		return EnterpriseTemplate(
			metadata=metadata,
			configuration=configuration,
			resource_allocation=resource_allocation
		)
	
	async def _create_healthcare_template(self) -> EnterpriseTemplate:
		"""Create healthcare compliance template"""
		metadata = TemplateMetadata(
			id="builtin-healthcare",
			name="healthcare_compliant",
			display_name="Healthcare Compliant Platform",
			description="HIPAA-compliant healthcare application platform",
			template_type=TemplateType.COMPLIANCE,
			category=TemplateCategory.HEALTHCARE,
			status=TemplateStatus.PUBLISHED,
			current_version="1.0.0",
			versions=[TemplateVersion(
				version="1.0.0",
				created_at=datetime.now(UTC),
				created_by="System",
				changelog=["Initial healthcare template"]
			)],
			min_tier=TenantTier.ENTERPRISE,
			required_features=["hipaa_compliance", "audit_logging", "encryption"]
		)
		
		configuration = TemplateConfiguration(
			base_configuration={
				"compliance": {
					"framework": "hipaa",
					"encryption": "end_to_end",
					"audit_logging": "comprehensive",
					"access_controls": "rbac"
				},
				"security": {
					"data_encryption": True,
					"secure_communications": True,
					"access_logging": True,
					"data_retention": "compliant"
				}
			}
		)
		
		resource_allocation = ResourceAllocation(
			cpu_cores=4,
			memory_gb=8,
			storage_gb=100,
			bandwidth_mbps=200,
			database_connections=50
		)
		
		return EnterpriseTemplate(
			metadata=metadata,
			configuration=configuration,
			resource_allocation=resource_allocation
		)


# Export key classes and functions
__all__ = [
	'AdvancedTemplateSystem',
	'EnterpriseTemplate',
	'TemplateMetadata',
	'TemplateConfiguration',
	'TemplateVersion',
	'TemplateInheritanceResolver',
	'TemplateComposer',
	'TemplateMarketplace',
	'TemplateType',
	'TemplateCategory',
	'TemplateStatus',
	'VersioningStrategy'
]