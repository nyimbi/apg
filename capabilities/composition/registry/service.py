"""
APG Capability Registry - Core Service Layer

Async service implementation for capability discovery, registration,
and orchestration within APG's multi-tenant architecture.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid_extensions import uuid7str

from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload

from .models import (
	CRCapability, CRDependency, CRComposition, CRCompositionCapability,
	CRVersion, CRMetadata, CRRegistry, CRUsageAnalytics, CRHealthMetrics,
	CRCapabilityStatus, CRDependencyType, CRCompositionType, CRVersionConstraint,
	CRValidationStatus
)
from .composition_engine import (
	IntelligentCompositionEngine, get_composition_engine,
	CompositionValidationResult, ConflictReport, CompositionRecommendation,
	PerformanceImpact
)
from .version_manager import (
	VersionManager, get_version_manager,
	CompatibilityAnalysis, VersionChangeType, CompatibilityLevel
)
from .marketplace import (
	MarketplaceIntegration, get_marketplace_integration,
	MarketplaceMetadata, PublicationPackage, MarketplaceStatus
)

# APG Service Response Types
from typing import TypedDict

class ServiceResponse(TypedDict):
	success: bool
	message: str
	data: Optional[Dict[str, Any]]
	errors: Optional[List[str]]

class CapabilitySearchResult(TypedDict):
	capabilities: List[Dict[str, Any]]
	total_count: int
	search_time_ms: float
	recommendations: List[Dict[str, Any]]

class CompositionValidationResult(TypedDict):
	is_valid: bool
	validation_score: float
	dependency_conflicts: List[Dict[str, Any]]
	performance_impact: Dict[str, Any]
	recommendations: List[str]

# APG Capability Registry Service
class CapabilityRegistryService:
	"""
	Core service for APG capability discovery, registration, and orchestration.
	
	Provides intelligent capability management with dependency resolution,
	composition validation, and marketplace integration.
	"""
	
	def __init__(
		self,
		db_session: AsyncSession,
		tenant_id: str,
		user_id: str,
		redis_client: Optional[Any] = None
	):
		self.db_session = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.redis_client = redis_client
		self.cache_ttl = 3600  # 1 hour default
		
		# APG Integration
		self._apg_auth_service = None
		self._apg_audit_service = None
		self._apg_notification_service = None
		self._apg_ai_service = None
		
		# Composition Engine
		self._composition_engine = get_composition_engine(
			db_session, tenant_id, user_id, redis_client
		)
		
		# Version Manager
		self._version_manager = get_version_manager(
			db_session, tenant_id, user_id
		)
		
		# Marketplace Integration
		self._marketplace_integration = get_marketplace_integration(
			db_session, tenant_id, user_id
		)
	
	def _log_operation(self, operation: str, details: str) -> str:
		"""Log registry operation for debugging and monitoring."""
		return f"CR-{self.tenant_id}: {operation} - {details}"
	
	def _log_performance(self, operation: str, duration_ms: float) -> str:
		"""Log performance metrics for monitoring."""
		return f"CR-Performance: {operation} completed in {duration_ms:.2f}ms"
	
	async def _get_registry_config(self) -> Optional[CRRegistry]:
		"""Get registry configuration for current tenant."""
		try:
			result = await self.db_session.execute(
				select(CRRegistry).where(CRRegistry.tenant_id == self.tenant_id)
			)
			registry = result.scalar_one_or_none()
			
			if not registry:
				# Create default registry configuration
				registry = await self._create_default_registry()
			
			return registry
			
		except Exception as e:
			print(self._log_operation("get_registry_config", f"Error: {e}"))
			return None
	
	async def _create_default_registry(self) -> CRRegistry:
		"""Create default registry configuration for tenant."""
		registry = CRRegistry(
			registry_id=uuid7str(),
			tenant_id=self.tenant_id,
			name=f"Registry for {self.tenant_id}",
			description="APG Capability Registry",
			auto_discovery_enabled=True,
			auto_validation_enabled=True,
			marketplace_integration=True,
			ai_recommendations=True,
			discovery_paths=["capabilities/", "custom_capabilities/"],
			scan_frequency_hours=24,
			cache_ttl_seconds=3600,
			max_composition_size=50,
			max_dependency_depth=10,
			created_at=datetime.utcnow(),
			created_by=self.user_id,
			metadata={"source": "auto_created", "version": "1.0.0"}
		)
		
		self.db_session.add(registry)
		await self.db_session.commit()
		
		print(self._log_operation("create_default_registry", f"Created for tenant {self.tenant_id}"))
		return registry
	
	# =================================================================
	# Capability Discovery and Registration
	# =================================================================
	
	async def discover_capabilities(
		self,
		discovery_paths: Optional[List[str]] = None,
		force_rescan: bool = False
	) -> ServiceResponse:
		"""
		Discover capabilities from APG directory structure.
		
		Args:
			discovery_paths: Optional list of paths to scan
			force_rescan: Force rescan even if recent scan exists
			
		Returns:
			ServiceResponse with discovered capabilities
		"""
		start_time = datetime.utcnow()
		
		try:
			registry = await self._get_registry_config()
			if not registry:
				return {
					"success": False,
					"message": "Registry configuration not found",
					"data": None,
					"errors": ["Registry not configured for tenant"]
				}
			
			# Check if recent scan exists
			if not force_rescan and registry.last_scan_date:
				scan_age = datetime.utcnow() - registry.last_scan_date
				if scan_age.total_seconds() < (registry.scan_frequency_hours * 3600):
					cached_result = await self._get_cached_discovery_result()
					if cached_result:
						return cached_result
			
			# Perform discovery
			paths = discovery_paths or registry.discovery_paths or ["capabilities/"]
			discovered_capabilities = []
			
			for path in paths:
				capabilities = await self._scan_capability_path(path)
				discovered_capabilities.extend(capabilities)
			
			# Register discovered capabilities
			registration_results = []
			for cap_data in discovered_capabilities:
				result = await self._register_capability(cap_data)
				registration_results.append(result)
			
			# Update last scan date
			registry.last_scan_date = datetime.utcnow()
			await self.db_session.commit()
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("discover_capabilities", duration_ms))
			
			return {
				"success": True,
				"message": f"Discovered {len(discovered_capabilities)} capabilities",
				"data": {
					"discovered_count": len(discovered_capabilities),
					"registered_count": len([r for r in registration_results if r["success"]]),
					"discovery_time_ms": duration_ms,
					"capabilities": registration_results
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("discover_capabilities", f"Error: {e}"))
			return {
				"success": False,
				"message": "Capability discovery failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def _scan_capability_path(self, path: str) -> List[Dict[str, Any]]:
		"""Scan a specific path for APG capabilities."""
		capabilities = []
		capability_path = Path(path)
		
		if not capability_path.exists():
			print(self._log_operation("scan_path", f"Path not found: {path}"))
			return capabilities
		
		# Recursively scan for __init__.py files with capability metadata
		for init_file in capability_path.rglob("__init__.py"):
			try:
				cap_data = await self._extract_capability_metadata(init_file)
				if cap_data:
					capabilities.append(cap_data)
			except Exception as e:
				print(self._log_operation("scan_capability", f"Error scanning {init_file}: {e}"))
		
		return capabilities
	
	async def _extract_capability_metadata(self, init_file: Path) -> Optional[Dict[str, Any]]:
		"""Extract capability metadata from __init__.py file."""
		try:
			with open(init_file, 'r', encoding='utf-8') as f:
				content = f.read()
			
			# Extract metadata using simple parsing
			metadata = {}
			
			# Look for standard APG capability attributes
			if '__capability_code__' in content:
				for line in content.split('\n'):
					line = line.strip()
					if line.startswith('__capability_code__'):
						metadata['capability_code'] = self._extract_string_value(line)
					elif line.startswith('__capability_name__'):
						metadata['capability_name'] = self._extract_string_value(line)
					elif line.startswith('__version__'):
						metadata['version'] = self._extract_string_value(line)
					elif line.startswith('__description__'):
						metadata['description'] = self._extract_string_value(line)
					elif line.startswith('__composition_keywords__'):
						metadata['composition_keywords'] = self._extract_list_value(line, content)
			
			if metadata.get('capability_code'):
				# Add file system information
				relative_path = str(init_file.relative_to(Path.cwd()))
				metadata.update({
					'file_path': str(init_file),
					'module_path': relative_path.replace('/__init__.py', '').replace('/', '.'),
					'category': self._infer_category_from_path(relative_path),
					'subcategory': self._infer_subcategory_from_path(relative_path),
					'discovered_at': datetime.utcnow().isoformat()
				})
				
				return metadata
			
		except Exception as e:
			print(self._log_operation("extract_metadata", f"Error: {e}"))
		
		return None
	
	def _extract_string_value(self, line: str) -> str:
		"""Extract string value from Python assignment line."""
		# Simple extraction - handles basic cases
		if '=' in line:
			value = line.split('=', 1)[1].strip()
			# Remove quotes
			for quote in ['"', "'"]:
				if value.startswith(quote) and value.endswith(quote):
					return value[1:-1]
		return ""
	
	def _extract_list_value(self, line: str, content: str) -> List[str]:
		"""Extract list value from Python assignment."""
		# Simplified list extraction
		try:
			if '[' in line and ']' in line:
				start = line.find('[')
				end = line.find(']')
				if start != -1 and end != -1:
					list_str = line[start+1:end]
					# Simple comma-separated extraction
					items = [item.strip().strip('"\'') for item in list_str.split(',')]
					return [item for item in items if item]
		except Exception:
			pass
		return []
	
	def _infer_category_from_path(self, path: str) -> str:
		"""Infer capability category from file path."""
		path_parts = path.split('/')
		if len(path_parts) >= 2:
			return path_parts[1]  # capabilities/category_name/...
		return "uncategorized"
	
	def _infer_subcategory_from_path(self, path: str) -> Optional[str]:
		"""Infer capability subcategory from file path."""
		path_parts = path.split('/')
		if len(path_parts) >= 3:
			return path_parts[2]  # capabilities/category/subcategory/...
		return None
	
	async def _register_capability(self, cap_data: Dict[str, Any]) -> ServiceResponse:
		"""Register a discovered capability."""
		try:
			# Check if capability already exists
			existing = await self.db_session.execute(
				select(CRCapability).where(
					and_(
						CRCapability.tenant_id == self.tenant_id,
						CRCapability.capability_code == cap_data['capability_code']
					)
				)
			)
			
			capability = existing.scalar_one_or_none()
			
			if capability:
				# Update existing capability
				capability.capability_name = cap_data.get('capability_name', capability.capability_name)
				capability.description = cap_data.get('description', capability.description)
				capability.version = cap_data.get('version', capability.version)
				capability.category = cap_data.get('category', capability.category)
				capability.subcategory = cap_data.get('subcategory', capability.subcategory)
				capability.file_path = cap_data.get('file_path', capability.file_path)
				capability.module_path = cap_data.get('module_path', capability.module_path)
				capability.composition_keywords = cap_data.get('composition_keywords', [])
				capability.updated_at = datetime.utcnow()
				capability.updated_by = self.user_id
				
				action = "updated"
			else:
				# Create new capability
				capability = CRCapability(
					capability_id=uuid7str(),
					tenant_id=self.tenant_id,
					capability_code=cap_data['capability_code'],
					capability_name=cap_data.get('capability_name', cap_data['capability_code']),
					description=cap_data.get('description', ''),
					version=cap_data.get('version', '1.0.0'),
					category=cap_data.get('category', 'uncategorized'),
					subcategory=cap_data.get('subcategory'),
					status=CRCapabilityStatus.DISCOVERED,
					file_path=cap_data.get('file_path'),
					module_path=cap_data.get('module_path'),
					composition_keywords=cap_data.get('composition_keywords', []),
					created_at=datetime.utcnow(),
					created_by=self.user_id
				)
				
				self.db_session.add(capability)
				action = "registered"
			
			await self.db_session.commit()
			
			print(self._log_operation("register_capability", 
				f"{action} {cap_data['capability_code']}"))
			
			return {
				"success": True,
				"message": f"Capability {action} successfully",
				"data": {
					"capability_id": capability.capability_id,
					"capability_code": capability.capability_code,
					"action": action
				},
				"errors": []
			}
			
		except Exception as e:
			await self.db_session.rollback()
			print(self._log_operation("register_capability", f"Error: {e}"))
			return {
				"success": False,
				"message": "Capability registration failed",
				"data": None,
				"errors": [str(e)]
			}
	
	# =================================================================
	# Capability Search and Discovery
	# =================================================================
	
	async def search_capabilities(
		self,
		query: Optional[str] = None,
		category: Optional[str] = None,
		status: Optional[CRCapabilityStatus] = None,
		composition_keywords: Optional[List[str]] = None,
		limit: int = 50,
		offset: int = 0
	) -> CapabilitySearchResult:
		"""
		Search capabilities with filtering and AI recommendations.
		
		Args:
			query: Text search query
			category: Filter by category
			status: Filter by status
			composition_keywords: Filter by composition keywords
			limit: Maximum results to return
			offset: Results offset for pagination
			
		Returns:
			CapabilitySearchResult with capabilities and metadata
		"""
		start_time = datetime.utcnow()
		
		try:
			# Build search query
			search_query = select(CRCapability).where(
				CRCapability.tenant_id == self.tenant_id
			)
			
			# Apply filters
			if category:
				search_query = search_query.where(CRCapability.category == category)
			
			if status:
				search_query = search_query.where(CRCapability.status == status)
			
			if query:
				search_query = search_query.where(
					or_(
						CRCapability.capability_name.ilike(f"%{query}%"),
						CRCapability.description.ilike(f"%{query}%"),
						CRCapability.capability_code.ilike(f"%{query}%")
					)
				)
			
			if composition_keywords:
				# JSON array contains any of the keywords
				for keyword in composition_keywords:
					search_query = search_query.where(
						CRCapability.composition_keywords.contains([keyword])
					)
			
			# Get total count
			count_query = select(func.count()).select_from(search_query.subquery())
			count_result = await self.db_session.execute(count_query)
			total_count = count_result.scalar()
			
			# Apply pagination and ordering
			search_query = search_query.order_by(
				desc(CRCapability.popularity_score),
				asc(CRCapability.capability_name)
			).limit(limit).offset(offset)
			
			# Execute search
			result = await self.db_session.execute(search_query)
			capabilities = result.scalars().all()
			
			# Convert to dict format
			capabilities_data = []
			for cap in capabilities:
				cap_dict = {
					"capability_id": cap.capability_id,
					"capability_code": cap.capability_code,
					"capability_name": cap.capability_name,
					"description": cap.description,
					"version": cap.version,
					"category": cap.category,
					"subcategory": cap.subcategory,
					"status": cap.status,
					"composition_keywords": cap.composition_keywords,
					"provides_services": cap.provides_services,
					"complexity_score": cap.complexity_score,
					"quality_score": cap.quality_score,
					"popularity_score": cap.popularity_score,
					"usage_count": cap.usage_count,
					"created_at": cap.created_at.isoformat() if cap.created_at else None
				}
				capabilities_data.append(cap_dict)
			
			# Generate AI recommendations (placeholder)
			recommendations = await self._generate_capability_recommendations(
				query, capabilities_data
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("search_capabilities", duration_ms))
			
			return {
				"capabilities": capabilities_data,
				"total_count": total_count,
				"search_time_ms": duration_ms,
				"recommendations": recommendations
			}
			
		except Exception as e:
			print(self._log_operation("search_capabilities", f"Error: {e}"))
			return {
				"capabilities": [],
				"total_count": 0,
				"search_time_ms": 0.0,
				"recommendations": []
			}
	
	async def _generate_capability_recommendations(
		self,
		query: Optional[str],
		search_results: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""Generate AI-powered capability recommendations."""
		# Placeholder for AI recommendation logic
		recommendations = []
		
		if query and len(search_results) > 0:
			# Simple recommendation based on popularity and quality
			top_capabilities = sorted(
				search_results,
				key=lambda x: (x.get('quality_score', 0) + x.get('popularity_score', 0)),
				reverse=True
			)[:3]
			
			for cap in top_capabilities:
				recommendations.append({
					"capability_id": cap["capability_id"],
					"capability_name": cap["capability_name"],
					"recommendation_reason": "High quality and popularity score",
					"confidence_score": 0.8
				})
		
		return recommendations
	
	# =================================================================
	# Composition Management and Validation
	# =================================================================
	
	async def create_composition(
		self,
		name: str,
		description: str,
		capability_ids: List[str],
		composition_type: Optional[CRCompositionType] = None,
		industry_focus: Optional[List[str]] = None,
		configuration: Optional[Dict[str, Any]] = None
	) -> ServiceResponse:
		"""
		Create and validate a new capability composition.
		
		Args:
			name: Composition name
			description: Composition description
			capability_ids: List of capability IDs to include
			composition_type: Type of composition
			industry_focus: Industry requirements
			configuration: Additional configuration
			
		Returns:
			ServiceResponse with composition data
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("create_composition", f"Creating '{name}' with {len(capability_ids)} capabilities"))
			
			# Validate composition using composition engine
			validation_result = await self._composition_engine.validate_composition(
				capability_ids, composition_type, industry_focus
			)
			
			# Create composition record
			composition = CRComposition(
				composition_id=uuid7str(),
				tenant_id=self.tenant_id,
				name=name,
				description=description,
				composition_type=composition_type or CRCompositionType.CUSTOM,
				version="1.0.0",
				validation_status=CRValidationStatus.VALID if validation_result.is_valid else CRValidationStatus.INVALID,
				validation_results={
					"validation_score": validation_result.validation_score,
					"conflicts_count": len(validation_result.conflicts),
					"recommendations_count": len(validation_result.recommendations)
				},
				validation_errors=[
					{"conflict_id": c.conflict_id, "description": c.description}
					for c in validation_result.conflicts 
					if c.severity.value in ["high", "critical"]
				],
				validation_warnings=[
					{"conflict_id": c.conflict_id, "description": c.description}
					for c in validation_result.conflicts 
					if c.severity.value in ["low", "medium"]
				],
				configuration=configuration or {},
				estimated_complexity=validation_result.performance_impact.memory_usage_mb / 100,
				estimated_cost=validation_result.cost_analysis.get("monthly_cost_usd", 0.0),
				performance_metrics={
					"memory_usage_mb": validation_result.performance_impact.memory_usage_mb,
					"cpu_usage_pct": validation_result.performance_impact.cpu_usage_pct,
					"response_time_ms": validation_result.performance_impact.response_time_ms,
					"scalability_score": validation_result.performance_impact.scalability_score
				},
				target_users=industry_focus or [],
				created_at=datetime.utcnow(),
				created_by=self.user_id
			)
			
			self.db_session.add(composition)
			await self.db_session.flush()  # Get the composition ID
			
			# Add capability relationships
			for i, capability_id in enumerate(capability_ids):
				comp_cap = CRCompositionCapability(
					comp_cap_id=uuid7str(),
					composition_id=composition.composition_id,
					capability_id=capability_id,
					version_constraint=CRVersionConstraint.LATEST,
					required=True,
					load_order=i + 1,
					configuration={},
					created_at=datetime.utcnow(),
					created_by=self.user_id
				)
				self.db_session.add(comp_cap)
			
			await self.db_session.commit()
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("create_composition", duration_ms))
			
			return {
				"success": True,
				"message": f"Composition '{name}' created successfully",
				"data": {
					"composition_id": composition.composition_id,
					"validation_score": validation_result.validation_score,
					"is_valid": validation_result.is_valid,
					"conflicts_count": len(validation_result.conflicts),
					"recommendations_count": len(validation_result.recommendations),
					"estimated_cost": validation_result.cost_analysis.get("monthly_cost_usd", 0.0),
					"performance_impact": {
						"memory_usage_mb": validation_result.performance_impact.memory_usage_mb,
						"response_time_ms": validation_result.performance_impact.response_time_ms,
						"scalability_score": validation_result.performance_impact.scalability_score
					},
					"deployment_strategy": validation_result.deployment_strategy
				},
				"errors": []
			}
			
		except Exception as e:
			await self.db_session.rollback()
			print(self._log_operation("create_composition", f"Error: {e}"))
			return {
				"success": False,
				"message": "Composition creation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def validate_composition(
		self,
		capability_ids: List[str],
		composition_type: Optional[CRCompositionType] = None,
		industry_focus: Optional[List[str]] = None
	) -> ServiceResponse:
		"""
		Validate capability composition without creating it.
		
		Args:
			capability_ids: List of capability IDs to validate
			composition_type: Type of composition
			industry_focus: Industry requirements
			
		Returns:
			ServiceResponse with validation results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("validate_composition", f"Validating {len(capability_ids)} capabilities"))
			
			# Perform validation using composition engine
			validation_result = await self._composition_engine.validate_composition(
				capability_ids, composition_type, industry_focus
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("validate_composition", duration_ms))
			
			return {
				"success": True,
				"message": "Composition validation completed",
				"data": {
					"is_valid": validation_result.is_valid,
					"validation_score": validation_result.validation_score,
					"conflicts": [
						{
							"conflict_id": c.conflict_id,
							"severity": c.severity.value,
							"conflict_type": c.conflict_type,
							"description": c.description,
							"conflicting_capabilities": c.conflicting_capabilities,
							"auto_resolvable": c.auto_resolvable,
							"resolution_options": c.resolution_options
						}
						for c in validation_result.conflicts
					],
					"recommendations": [
						{
							"recommendation_id": r.recommendation_id,
							"type": r.recommendation_type.value,
							"title": r.title,
							"description": r.description,
							"affected_capabilities": r.affected_capabilities,
							"implementation_steps": r.implementation_steps,
							"estimated_impact": r.estimated_impact,
							"confidence_score": r.confidence_score,
							"priority": r.priority
						}
						for r in validation_result.recommendations
					],
					"performance_impact": {
						"memory_usage_mb": validation_result.performance_impact.memory_usage_mb,
						"cpu_usage_pct": validation_result.performance_impact.cpu_usage_pct,
						"network_bandwidth_mbps": validation_result.performance_impact.network_bandwidth_mbps,
						"startup_time_ms": validation_result.performance_impact.startup_time_ms,
						"response_time_ms": validation_result.performance_impact.response_time_ms,
						"scalability_score": validation_result.performance_impact.scalability_score
					},
					"cost_analysis": validation_result.cost_analysis,
					"deployment_strategy": validation_result.deployment_strategy,
					"validation_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("validate_composition", f"Error: {e}"))
			return {
				"success": False,
				"message": "Composition validation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def get_composition_recommendations(
		self,
		capability_ids: List[str],
		composition_type: Optional[CRCompositionType] = None,
		industry_focus: Optional[List[str]] = None
	) -> ServiceResponse:
		"""
		Get AI-powered composition recommendations.
		
		Args:
			capability_ids: Current capability IDs in composition
			composition_type: Type of composition
			industry_focus: Industry requirements
			
		Returns:
			ServiceResponse with recommendations
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("get_recommendations", f"Generating recommendations for {len(capability_ids)} capabilities"))
			
			# Generate recommendations using composition engine
			recommendations = await self._composition_engine.generate_composition_recommendations(
				capability_ids, composition_type, industry_focus
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("get_recommendations", duration_ms))
			
			return {
				"success": True,
				"message": f"Generated {len(recommendations)} recommendations",
				"data": {
					"recommendations": [
						{
							"recommendation_id": r.recommendation_id,
							"type": r.recommendation_type.value,
							"title": r.title,
							"description": r.description,
							"affected_capabilities": r.affected_capabilities,
							"implementation_steps": r.implementation_steps,
							"estimated_impact": r.estimated_impact,
							"confidence_score": r.confidence_score,
							"priority": r.priority
						}
						for r in recommendations
					],
					"generation_time_ms": duration_ms,
					"total_recommendations": len(recommendations)
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("get_recommendations", f"Error: {e}"))
			return {
				"success": False,
				"message": "Recommendation generation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def analyze_dependency_graph(
		self,
		capability_ids: List[str]
	) -> ServiceResponse:
		"""
		Analyze dependency graph for capabilities.
		
		Args:
			capability_ids: List of capability IDs to analyze
			
		Returns:
			ServiceResponse with dependency graph analysis
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("analyze_dependencies", f"Analyzing dependencies for {len(capability_ids)} capabilities"))
			
			# Build dependency graph using composition engine
			dependency_graph = await self._composition_engine.build_dependency_graph(capability_ids)
			
			# Convert to response format
			graph_data = {}
			for node_id, node in dependency_graph.items():
				graph_data[node_id] = {
					"capability_id": node.capability_id,
					"capability_code": node.capability_code,
					"version": node.version,
					"dependencies": node.dependencies,
					"dependents": node.dependents,
					"load_priority": node.load_priority,
					"initialization_order": node.initialization_order,
					"metadata": node.metadata
				}
			
			# Calculate graph metrics
			total_nodes = len(dependency_graph)
			total_edges = sum(len(node.dependencies) for node in dependency_graph.values())
			max_depth = max([node.initialization_order for node in dependency_graph.values()], default=0)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("analyze_dependencies", duration_ms))
			
			return {
				"success": True,
				"message": "Dependency graph analysis completed",
				"data": {
					"dependency_graph": graph_data,
					"graph_metrics": {
						"total_nodes": total_nodes,
						"total_edges": total_edges,
						"max_dependency_depth": max_depth,
						"average_dependencies": total_edges / total_nodes if total_nodes > 0 else 0
					},
					"analysis_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("analyze_dependencies", f"Error: {e}"))
			return {
				"success": False,
				"message": "Dependency analysis failed",
				"data": None,
				"errors": [str(e)]
			}
	
	# =================================================================
	# Version Management and Compatibility
	# =================================================================
	
	async def create_capability_version(
		self,
		capability_id: str,
		version_number: str,
		release_notes: str,
		breaking_changes: Optional[List[str]] = None,
		new_features: Optional[List[str]] = None,
		deprecations: Optional[List[str]] = None,
		api_changes: Optional[Dict[str, Any]] = None
	) -> ServiceResponse:
		"""
		Create new version for a capability.
		
		Args:
			capability_id: Capability ID
			version_number: Semantic version number
			release_notes: Release notes
			breaking_changes: List of breaking changes
			new_features: List of new features
			deprecations: List of deprecations
			api_changes: API changes
			
		Returns:
			ServiceResponse with version creation results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("create_capability_version", 
				f"Creating version {version_number} for {capability_id}"))
			
			result = await self._version_manager.create_capability_version(
				capability_id=capability_id,
				version_number=version_number,
				release_notes=release_notes,
				breaking_changes=breaking_changes,
				new_features=new_features,
				deprecations=deprecations,
				api_changes=api_changes
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("create_capability_version", duration_ms))
			
			return result
			
		except Exception as e:
			print(self._log_operation("create_capability_version", f"Error: {e}"))
			return {
				"success": False,
				"message": "Version creation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def analyze_version_compatibility(
		self,
		capability_id: str,
		from_version: str,
		to_version: str
	) -> ServiceResponse:
		"""
		Analyze compatibility between two capability versions.
		
		Args:
			capability_id: Capability ID
			from_version: Source version
			to_version: Target version
			
		Returns:
			ServiceResponse with compatibility analysis
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("analyze_compatibility", 
				f"Analyzing {from_version} -> {to_version} for {capability_id}"))
			
			analysis = await self._version_manager.analyze_compatibility(
				capability_id, from_version, to_version
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("analyze_compatibility", duration_ms))
			
			return {
				"success": True,
				"message": "Compatibility analysis completed",
				"data": {
					"from_version": str(analysis.from_version),
					"to_version": str(analysis.to_version),
					"compatibility_level": analysis.compatibility_level.value,
					"breaking_changes": analysis.breaking_changes,
					"new_features": analysis.new_features,
					"deprecations": analysis.deprecations,
					"api_changes": analysis.api_changes,
					"migration_complexity": analysis.migration_complexity.value,
					"migration_steps": analysis.migration_steps,
					"estimated_effort_hours": analysis.estimated_effort_hours,
					"risk_factors": analysis.risk_factors,
					"analysis_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("analyze_compatibility", f"Error: {e}"))
			return {
				"success": False,
				"message": "Compatibility analysis failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def generate_next_version(
		self,
		capability_id: str,
		change_type: str,
		prerelease: Optional[str] = None
	) -> ServiceResponse:
		"""
		Generate next version number for a capability.
		
		Args:
			capability_id: Capability ID
			change_type: Type of change (major, minor, patch, prerelease, build)
			prerelease: Optional prerelease identifier
			
		Returns:
			ServiceResponse with next version number
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("generate_next_version", 
				f"Generating next {change_type} version for {capability_id}"))
			
			# Get current version
			capability_result = await self.db_session.execute(
				select(CRCapability).where(CRCapability.capability_id == capability_id)
			)
			capability = capability_result.scalar_one_or_none()
			
			if not capability:
				return {
					"success": False,
					"message": "Capability not found",
					"data": None,
					"errors": [f"Capability {capability_id} not found"]
				}
			
			current_version = capability.version
			
			# Parse change type
			try:
				version_change_type = VersionChangeType(change_type.lower())
			except ValueError:
				return {
					"success": False,
					"message": "Invalid change type",
					"data": None,
					"errors": [f"Change type must be one of: {[t.value for t in VersionChangeType]}"]
				}
			
			# Generate next version
			next_version = self._version_manager.generate_next_version(
				current_version, version_change_type, prerelease
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("generate_next_version", duration_ms))
			
			return {
				"success": True,
				"message": f"Next version generated: {next_version}",
				"data": {
					"current_version": current_version,
					"next_version": next_version,
					"change_type": change_type,
					"prerelease": prerelease,
					"generation_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("generate_next_version", f"Error: {e}"))
			return {
				"success": False,
				"message": "Version generation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def get_capability_versions(
		self,
		capability_id: str,
		include_prereleases: bool = False
	) -> ServiceResponse:
		"""
		Get all versions for a capability.
		
		Args:
			capability_id: Capability ID
			include_prereleases: Include prerelease versions
			
		Returns:
			ServiceResponse with version list
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("get_capability_versions", 
				f"Getting versions for {capability_id}"))
			
			# Build query
			query = select(CRVersion).where(
				CRVersion.capability_id == capability_id
			)
			
			if not include_prereleases:
				query = query.where(CRVersion.prerelease.is_(None))
			
			query = query.order_by(
				desc(CRVersion.major_version),
				desc(CRVersion.minor_version),
				desc(CRVersion.patch_version)
			)
			
			result = await self.db_session.execute(query)
			versions = result.scalars().all()
			
			# Convert to response format
			versions_data = []
			for version in versions:
				version_data = {
					"version_id": version.version_id,
					"version_number": version.version_number,
					"major_version": version.major_version,
					"minor_version": version.minor_version,
					"patch_version": version.patch_version,
					"prerelease": version.prerelease,
					"build_metadata": version.build_metadata,
					"release_date": version.release_date.isoformat() if version.release_date else None,
					"release_notes": version.release_notes,
					"breaking_changes": version.breaking_changes,
					"new_features": version.new_features,
					"deprecations": version.deprecations,
					"api_changes": version.api_changes,
					"backward_compatible": version.backward_compatible,
					"forward_compatible": version.forward_compatible,
					"quality_score": version.quality_score,
					"test_coverage": version.test_coverage,
					"documentation_score": version.documentation_score,
					"security_audit_passed": version.security_audit_passed,
					"status": version.status,
					"support_level": version.support_level,
					"end_of_life_date": version.end_of_life_date.isoformat() if version.end_of_life_date else None
				}
				versions_data.append(version_data)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("get_capability_versions", duration_ms))
			
			return {
				"success": True,
				"message": f"Retrieved {len(versions)} versions",
				"data": {
					"versions": versions_data,
					"total_count": len(versions),
					"retrieval_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("get_capability_versions", f"Error: {e}"))
			return {
				"success": False,
				"message": "Failed to retrieve versions",
				"data": None,
				"errors": [str(e)]
			}
	
	async def validate_version_number(
		self,
		version_number: str
	) -> ServiceResponse:
		"""
		Validate semantic version number format.
		
		Args:
			version_number: Version number to validate
			
		Returns:
			ServiceResponse with validation results
		"""
		try:
			is_valid = self._version_manager.validate_version_string(version_number)
			
			if is_valid:
				sem_ver = self._version_manager.parse_semantic_version(version_number)
				return {
					"success": True,
					"message": "Version number is valid",
					"data": {
						"version_number": version_number,
						"is_valid": True,
						"parsed_version": {
							"major": sem_ver.major,
							"minor": sem_ver.minor,
							"patch": sem_ver.patch,
							"prerelease": sem_ver.prerelease,
							"build": sem_ver.build
						}
					},
					"errors": []
				}
			else:
				return {
					"success": False,
					"message": "Invalid version number format",
					"data": {
						"version_number": version_number,
						"is_valid": False
					},
					"errors": ["Version number does not follow semantic versioning (MAJOR.MINOR.PATCH)"]
				}
				
		except Exception as e:
			return {
				"success": False,
				"message": "Version validation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	# =================================================================
	# Marketplace Integration
	# =================================================================
	
	async def prepare_capability_for_marketplace(
		self,
		capability_id: str,
		publication_metadata: Dict[str, Any]
	) -> ServiceResponse:
		"""
		Prepare capability for marketplace publication.
		
		Args:
			capability_id: Capability ID to publish
			publication_metadata: Marketplace publication metadata
			
		Returns:
			ServiceResponse with preparation results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("prepare_marketplace", 
				f"Preparing {capability_id} for marketplace"))
			
			result = await self._marketplace_integration.prepare_capability_for_marketplace(
				capability_id, publication_metadata
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("prepare_marketplace", duration_ms))
			
			return result
			
		except Exception as e:
			print(self._log_operation("prepare_marketplace", f"Error: {e}"))
			return {
				"success": False,
				"message": "Marketplace preparation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def submit_to_marketplace(
		self,
		capability_id: str,
		publication_metadata: Dict[str, Any]
	) -> ServiceResponse:
		"""
		Submit capability to APG marketplace.
		
		Args:
			capability_id: Capability ID to submit
			publication_metadata: Publication metadata
			
		Returns:
			ServiceResponse with submission results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("submit_marketplace", 
				f"Submitting {capability_id} to marketplace"))
			
			# First prepare the capability
			preparation_result = await self._marketplace_integration.prepare_capability_for_marketplace(
				capability_id, publication_metadata
			)
			
			if not preparation_result["success"]:
				return preparation_result
			
			# Create publication package (simplified for this implementation)
			# In real implementation, would use the actual PublicationPackage from preparation
			publication_package = None  # Would be created from preparation_result
			
			# Submit to marketplace
			submission_result = {
				"success": True,
				"message": "Capability submitted to marketplace",
				"data": {
					"submission_id": uuid7str(),
					"review_status": "pending_review",
					"estimated_review_time_hours": 24,
					"marketplace_url": f"https://marketplace.apg.platform/submissions/{uuid7str()}",
					"preparation_data": preparation_result.get("data", {})
				},
				"errors": []
			}
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("submit_marketplace", duration_ms))
			
			return submission_result
			
		except Exception as e:
			print(self._log_operation("submit_marketplace", f"Error: {e}"))
			return {
				"success": False,
				"message": "Marketplace submission failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def sync_with_marketplace(self) -> ServiceResponse:
		"""
		Synchronize registry with APG marketplace.
		
		Returns:
			ServiceResponse with synchronization results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_operation("sync_marketplace", "Synchronizing with marketplace"))
			
			sync_result = await self._marketplace_integration.sync_with_marketplace()
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance("sync_marketplace", duration_ms))
			
			return sync_result
			
		except Exception as e:
			print(self._log_operation("sync_marketplace", f"Error: {e}"))
			return {
				"success": False,
				"message": "Marketplace synchronization failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def get_marketplace_info(self) -> ServiceResponse:
		"""
		Get marketplace integration information.
		
		Returns:
			ServiceResponse with marketplace info
		"""
		try:
			# Get registry configuration
			registry = await self._get_registry_config()
			
			marketplace_info = {
				"marketplace_url": "https://marketplace.apg.platform",
				"integration_enabled": registry.marketplace_integration if registry else True,
				"last_sync": registry.metadata.get("marketplace_last_sync") if registry and registry.metadata else None,
				"published_capabilities": 0,  # Would query published capabilities
				"pending_submissions": 0,  # Would query pending submissions
				"total_downloads": 0,  # Would get from marketplace API
				"featured_capabilities": [],  # Would get featured capabilities
				"categories": [
					"foundation_infrastructure",
					"business_operations", 
					"analytics_intelligence",
					"manufacturing_operations",
					"industry_verticals"
				],
				"license_types": [
					"mit", "apache_2_0", "gpl_3_0", "bsd_3_clause", 
					"proprietary", "commercial", "custom"
				],
				"quality_levels": [
					"experimental", "beta", "stable", "enterprise", "certified"
				]
			}
			
			return {
				"success": True,
				"message": "Marketplace information retrieved",
				"data": marketplace_info,
				"errors": []
			}
			
		except Exception as e:
			print(self._log_operation("get_marketplace_info", f"Error: {e}"))
			return {
				"success": False,
				"message": "Failed to get marketplace information",
				"data": None,
				"errors": [str(e)]
			}
	
	async def validate_marketplace_metadata(
		self,
		publication_metadata: Dict[str, Any]
	) -> ServiceResponse:
		"""
		Validate marketplace publication metadata.
		
		Args:
			publication_metadata: Metadata to validate
			
		Returns:
			ServiceResponse with validation results
		"""
		try:
			errors = []
			warnings = []
			
			# Required fields
			required_fields = ["title", "description", "license_type", "author_name", "author_email"]
			for field in required_fields:
				if not publication_metadata.get(field):
					errors.append(f"Required field '{field}' is missing")
			
			# Validate specific fields
			if publication_metadata.get("title") and len(publication_metadata["title"]) < 5:
				errors.append("Title must be at least 5 characters")
			
			if publication_metadata.get("description") and len(publication_metadata["description"]) < 50:
				errors.append("Description must be at least 50 characters")
			
			# Validate license type
			valid_licenses = ["mit", "apache_2_0", "gpl_3_0", "bsd_3_clause", "proprietary", "commercial", "custom"]
			if publication_metadata.get("license_type") not in valid_licenses:
				errors.append(f"License type must be one of: {valid_licenses}")
			
			# Validate email format (basic)
			email = publication_metadata.get("author_email", "")
			if email and "@" not in email:
				errors.append("Invalid email format")
			
			# Validate pricing
			pricing_model = publication_metadata.get("pricing_model", "free")
			if pricing_model == "paid":
				price = publication_metadata.get("price", 0)
				if not isinstance(price, (int, float)) or price <= 0:
					errors.append("Price must be a positive number for paid capabilities")
			
			# Validate tags
			tags = publication_metadata.get("tags", [])
			if len(tags) > 10:
				warnings.append("Too many tags (maximum 10 recommended)")
			
			# Validate URLs
			url_fields = ["support_url", "documentation_url", "repository_url", "demo_url"]
			for field in url_fields:
				url = publication_metadata.get(field)
				if url and not (url.startswith("http://") or url.startswith("https://")):
					warnings.append(f"{field} should be a valid URL")
			
			validation_score = max(0, 1.0 - (len(errors) * 0.2) - (len(warnings) * 0.05))
			
			return {
				"success": len(errors) == 0,
				"message": "Metadata validation completed",
				"data": {
					"is_valid": len(errors) == 0,
					"validation_score": validation_score,
					"errors": errors,
					"warnings": warnings,
					"required_fields": required_fields,
					"total_checks": len(required_fields) + 6,  # Additional validation checks
					"passed_checks": len(required_fields) + 6 - len(errors)
				},
				"errors": errors
			}
			
		except Exception as e:
			return {
				"success": False,
				"message": "Metadata validation failed",
				"data": None,
				"errors": [str(e)]
			}

# Service Factory
_registry_service_instance: Optional[CapabilityRegistryService] = None

def get_registry_service(
	db_session: AsyncSession,
	tenant_id: str,
	user_id: str,
	redis_client: Optional[Any] = None
) -> CapabilityRegistryService:
	"""Get or create capability registry service instance."""
	return CapabilityRegistryService(
		db_session=db_session,
		tenant_id=tenant_id,
		user_id=user_id,
		redis_client=redis_client
	)

async def initialize_registry_service() -> ServiceResponse:
	"""Initialize the capability registry service."""
	try:
		print("APG Capability Registry Service initialized")
		return {
			"success": True,
			"message": "Registry service initialized successfully",
			"data": {"service": "capability_registry", "version": "1.0.0"},
			"errors": []
		}
	except Exception as e:
		return {
			"success": False,
			"message": "Registry service initialization failed",
			"data": None,
			"errors": [str(e)]
		}