"""
APG Capability Registry - Intelligent Composition Engine

AI-powered composition engine for dependency resolution, conflict detection,
and intelligent capability orchestration within APG's architecture.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid_extensions import uuid7str
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
	CRCapability, CRDependency, CRComposition, CRCompositionCapability,
	CRVersion, CRRegistry, CRUsageAnalytics, CRHealthMetrics,
	CRCapabilityStatus, CRDependencyType, CRCompositionType, 
	CRVersionConstraint, CRValidationStatus
)

# =============================================================================
# Composition Engine Data Structures
# =============================================================================

class ConflictSeverity(str, Enum):
	"""Severity levels for dependency conflicts."""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

class RecommendationType(str, Enum):
	"""Types of composition recommendations."""
	OPTIMIZATION = "optimization"
	SECURITY = "security"
	PERFORMANCE = "performance"
	COMPATIBILITY = "compatibility"
	COST_REDUCTION = "cost_reduction"
	FEATURE_ENHANCEMENT = "feature_enhancement"

@dataclass
class DependencyNode:
	"""Represents a capability node in the dependency graph."""
	capability_id: str
	capability_code: str
	version: str
	dependencies: List[str]
	dependents: List[str]
	load_priority: int
	initialization_order: int
	metadata: Dict[str, Any]

@dataclass
class ConflictReport:
	"""Represents a dependency conflict."""
	conflict_id: str
	severity: ConflictSeverity
	conflicting_capabilities: List[str]
	conflict_type: str
	description: str
	resolution_options: List[Dict[str, Any]]
	auto_resolvable: bool

@dataclass
class CompositionRecommendation:
	"""Represents an AI-generated composition recommendation."""
	recommendation_id: str
	recommendation_type: RecommendationType
	title: str
	description: str
	affected_capabilities: List[str]
	implementation_steps: List[str]
	estimated_impact: Dict[str, Any]
	confidence_score: float
	priority: int

@dataclass
class PerformanceImpact:
	"""Represents performance impact analysis."""
	memory_usage_mb: float
	cpu_usage_pct: float
	network_bandwidth_mbps: float
	disk_io_ops: int
	startup_time_ms: float
	response_time_ms: float
	scalability_score: float

@dataclass
class CompositionValidationResult:
	"""Complete composition validation results."""
	is_valid: bool
	validation_score: float
	conflicts: List[ConflictReport]
	recommendations: List[CompositionRecommendation]
	performance_impact: PerformanceImpact
	cost_analysis: Dict[str, Any]
	deployment_strategy: Dict[str, Any]

# =============================================================================
# Intelligent Composition Engine
# =============================================================================

class IntelligentCompositionEngine:
	"""
	AI-powered composition engine for APG capability orchestration.
	
	Provides dependency resolution, conflict detection, performance optimization,
	and intelligent recommendations for capability compositions.
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
		
		# AI Integration (placeholder for APG AI service)
		self._ai_service = None
		self._analytics_service = None
		self._performance_service = None
		
		# Caching
		self._dependency_cache: Dict[str, DependencyNode] = {}
		self._validation_cache: Dict[str, CompositionValidationResult] = {}
		
	def _log_composition_operation(self, operation: str, details: str) -> str:
		"""Log composition engine operations."""
		return f"CE-{self.tenant_id}: {operation} - {details}"
	
	def _log_performance_metric(self, operation: str, duration_ms: float) -> str:
		"""Log composition engine performance metrics."""
		return f"CE-Performance: {operation} completed in {duration_ms:.2f}ms"
	
	# =================================================================
	# Dependency Graph Management
	# =================================================================
	
	async def build_dependency_graph(
		self,
		capability_ids: List[str]
	) -> Dict[str, DependencyNode]:
		"""
		Build complete dependency graph for given capabilities.
		
		Args:
			capability_ids: List of capability IDs to include
			
		Returns:
			Dictionary mapping capability_id to DependencyNode
		"""
		start_time = datetime.utcnow()
		dependency_graph = {}
		
		try:
			# Get all capabilities with their dependencies
			capabilities_query = select(CRCapability).options(
				selectinload(CRCapability.dependencies),
				selectinload(CRCapability.dependent_on),
				selectinload(CRCapability.versions)
			).where(
				and_(
					CRCapability.tenant_id == self.tenant_id,
					CRCapability.capability_id.in_(capability_ids)
				)
			)
			
			result = await self.db_session.execute(capabilities_query)
			capabilities = result.scalars().all()
			
			# Build nodes
			for cap in capabilities:
				node = DependencyNode(
					capability_id=cap.capability_id,
					capability_code=cap.capability_code,
					version=cap.version,
					dependencies=[dep.depends_on_id for dep in cap.dependencies],
					dependents=[dep.capability_id for dep in cap.dependent_on],
					load_priority=1,
					initialization_order=1,
					metadata={
						"name": cap.capability_name,
						"category": cap.category,
						"status": cap.status,
						"complexity_score": cap.complexity_score,
						"quality_score": cap.quality_score
					}
				)
				dependency_graph[cap.capability_id] = node
			
			# Resolve transitive dependencies
			await self._resolve_transitive_dependencies(dependency_graph)
			
			# Calculate load order
			await self._calculate_load_order(dependency_graph)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance_metric("build_dependency_graph", duration_ms))
			
			return dependency_graph
			
		except Exception as e:
			print(self._log_composition_operation("build_dependency_graph", f"Error: {e}"))
			return {}
	
	async def _resolve_transitive_dependencies(
		self,
		dependency_graph: Dict[str, DependencyNode]
	):
		"""Resolve transitive dependencies in the graph."""
		for node_id, node in dependency_graph.items():
			visited = set()
			transitive_deps = set()
			
			async def _collect_dependencies(cap_id: str, depth: int = 0):
				if cap_id in visited or depth > 10:  # Prevent infinite loops
					return
				
				visited.add(cap_id)
				
				if cap_id in dependency_graph:
					for dep_id in dependency_graph[cap_id].dependencies:
						transitive_deps.add(dep_id)
						await _collect_dependencies(dep_id, depth + 1)
			
			await _collect_dependencies(node_id)
			node.dependencies = list(transitive_deps)
	
	async def _calculate_load_order(
		self,
		dependency_graph: Dict[str, DependencyNode]
	):
		"""Calculate optimal load order using topological sorting."""
		# Simplified topological sort
		in_degree = {node_id: 0 for node_id in dependency_graph}
		
		# Calculate in-degrees
		for node in dependency_graph.values():
			for dep_id in node.dependencies:
				if dep_id in in_degree:
					in_degree[dep_id] += 1
		
		# Assign load order
		queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
		order = 1
		
		while queue:
			current = queue.pop(0)
			if current in dependency_graph:
				dependency_graph[current].initialization_order = order
				order += 1
				
				# Update dependents
				for dependent_id in dependency_graph[current].dependents:
					if dependent_id in in_degree:
						in_degree[dependent_id] -= 1
						if in_degree[dependent_id] == 0:
							queue.append(dependent_id)
	
	# =================================================================
	# Conflict Detection and Resolution
	# =================================================================
	
	async def detect_conflicts(
		self,
		capability_ids: List[str]
	) -> List[ConflictReport]:
		"""
		Detect conflicts in capability composition.
		
		Args:
			capability_ids: List of capability IDs to validate
			
		Returns:
			List of conflict reports
		"""
		start_time = datetime.utcnow()
		conflicts = []
		
		try:
			dependency_graph = await self.build_dependency_graph(capability_ids)
			
			# Check for circular dependencies
			circular_conflicts = await self._detect_circular_dependencies(dependency_graph)
			conflicts.extend(circular_conflicts)
			
			# Check for version conflicts
			version_conflicts = await self._detect_version_conflicts(capability_ids)
			conflicts.extend(version_conflicts)
			
			# Check for resource conflicts
			resource_conflicts = await self._detect_resource_conflicts(capability_ids)
			conflicts.extend(resource_conflicts)
			
			# Check for configuration conflicts
			config_conflicts = await self._detect_configuration_conflicts(capability_ids)
			conflicts.extend(config_conflicts)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance_metric("detect_conflicts", duration_ms))
			
			return conflicts
			
		except Exception as e:
			print(self._log_composition_operation("detect_conflicts", f"Error: {e}"))
			return []
	
	async def _detect_circular_dependencies(
		self,
		dependency_graph: Dict[str, DependencyNode]
	) -> List[ConflictReport]:
		"""Detect circular dependencies in the graph."""
		conflicts = []
		visited = set()
		rec_stack = set()
		
		def _has_cycle(node_id: str, path: List[str]) -> Optional[List[str]]:
			if node_id in rec_stack:
				# Found cycle - return the cycle path
				cycle_start = path.index(node_id)
				return path[cycle_start:]
			
			if node_id in visited:
				return None
			
			visited.add(node_id)
			rec_stack.add(node_id)
			path.append(node_id)
			
			if node_id in dependency_graph:
				for dep_id in dependency_graph[node_id].dependencies:
					cycle = _has_cycle(dep_id, path.copy())
					if cycle:
						return cycle
			
			rec_stack.remove(node_id)
			return None
		
		for node_id in dependency_graph:
			if node_id not in visited:
				cycle = _has_cycle(node_id, [])
				if cycle:
					conflict = ConflictReport(
						conflict_id=uuid7str(),
						severity=ConflictSeverity.CRITICAL,
						conflicting_capabilities=cycle,
						conflict_type="circular_dependency",
						description=f"Circular dependency detected: {' -> '.join(cycle)}",
						resolution_options=[
							{
								"option": "remove_dependency",
								"description": "Remove one dependency to break the cycle",
								"impact": "May affect functionality"
							}
						],
						auto_resolvable=False
					)
					conflicts.append(conflict)
		
		return conflicts
	
	async def _detect_version_conflicts(
		self,
		capability_ids: List[str]
	) -> List[ConflictReport]:
		"""Detect version compatibility conflicts."""
		conflicts = []
		
		try:
			# Get all dependencies with version constraints
			dependencies_query = select(CRDependency).where(
				CRDependency.capability_id.in_(capability_ids)
			)
			result = await self.db_session.execute(dependencies_query)
			dependencies = result.scalars().all()
			
			# Group by dependency target
			dependency_groups = {}
			for dep in dependencies:
				target_id = dep.depends_on_id
				if target_id not in dependency_groups:
					dependency_groups[target_id] = []
				dependency_groups[target_id].append(dep)
			
			# Check for conflicting version requirements
			for target_id, deps in dependency_groups.items():
				if len(deps) > 1:
					version_requirements = []
					for dep in deps:
						if dep.version_constraint and dep.version_constraint != CRVersionConstraint.LATEST:
							version_requirements.append({
								"from_capability": dep.capability_id,
								"constraint": dep.version_constraint,
								"version": dep.version_exact or dep.version_min or dep.version_max
							})
					
					if len(version_requirements) > 1:
						# Simplified conflict detection - check for exact version mismatches
						exact_versions = [req for req in version_requirements 
										if req["constraint"] == CRVersionConstraint.EXACT]
						
						if len(set(req["version"] for req in exact_versions)) > 1:
							conflict = ConflictReport(
								conflict_id=uuid7str(),
								severity=ConflictSeverity.HIGH,
								conflicting_capabilities=[req["from_capability"] for req in exact_versions],
								conflict_type="version_conflict",
								description=f"Conflicting version requirements for capability {target_id}",
								resolution_options=[
									{
										"option": "use_compatible_version",
										"description": "Find a version compatible with all requirements",
										"impact": "May require capability updates"
									}
								],
								auto_resolvable=True
							)
							conflicts.append(conflict)
			
		except Exception as e:
			print(self._log_composition_operation("detect_version_conflicts", f"Error: {e}"))
		
		return conflicts
	
	async def _detect_resource_conflicts(
		self,
		capability_ids: List[str]
	) -> List[ConflictReport]:
		"""Detect resource usage conflicts."""
		conflicts = []
		
		# Placeholder for resource conflict detection
		# This would analyze CPU, memory, network, and storage requirements
		
		return conflicts
	
	async def _detect_configuration_conflicts(
		self,
		capability_ids: List[str]
	) -> List[ConflictReport]:
		"""Detect configuration conflicts between capabilities."""
		conflicts = []
		
		# Placeholder for configuration conflict detection
		# This would analyze environment variables, port usage, and configuration overlaps
		
		return conflicts
	
	# =================================================================
	# AI-Powered Recommendations
	# =================================================================
	
	async def generate_composition_recommendations(
		self,
		capability_ids: List[str],
		composition_type: Optional[CRCompositionType] = None,
		industry_focus: Optional[List[str]] = None
	) -> List[CompositionRecommendation]:
		"""
		Generate AI-powered composition recommendations.
		
		Args:
			capability_ids: Current capabilities in composition
			composition_type: Type of composition being created
			industry_focus: Industry-specific requirements
			
		Returns:
			List of composition recommendations
		"""
		start_time = datetime.utcnow()
		recommendations = []
		
		try:
			# Get capability analytics
			analytics = await self._get_capability_analytics(capability_ids)
			
			# Generate optimization recommendations
			optimization_recs = await self._generate_optimization_recommendations(
				capability_ids, analytics
			)
			recommendations.extend(optimization_recs)
			
			# Generate security recommendations
			security_recs = await self._generate_security_recommendations(
				capability_ids, analytics
			)
			recommendations.extend(security_recs)
			
			# Generate performance recommendations
			performance_recs = await self._generate_performance_recommendations(
				capability_ids, analytics
			)
			recommendations.extend(performance_recs)
			
			# Generate industry-specific recommendations
			if industry_focus:
				industry_recs = await self._generate_industry_recommendations(
					capability_ids, industry_focus
				)
				recommendations.extend(industry_recs)
			
			# Sort by confidence score and priority
			recommendations.sort(key=lambda x: (x.confidence_score, x.priority), reverse=True)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance_metric("generate_recommendations", duration_ms))
			
			return recommendations[:10]  # Return top 10 recommendations
			
		except Exception as e:
			print(self._log_composition_operation("generate_recommendations", f"Error: {e}"))
			return []
	
	async def _get_capability_analytics(
		self,
		capability_ids: List[str]
	) -> Dict[str, Any]:
		"""Get analytics data for capabilities."""
		analytics = {}
		
		try:
			# Get usage analytics
			usage_query = select(CRUsageAnalytics).where(
				CRUsageAnalytics.capability_id.in_(capability_ids)
			)
			result = await self.db_session.execute(usage_query)
			usage_data = result.scalars().all()
			
			# Get health metrics
			health_query = select(CRHealthMetrics).where(
				CRHealthMetrics.capability_id.in_(capability_ids)
			)
			result = await self.db_session.execute(health_query)
			health_data = result.scalars().all()
			
			analytics = {
				"usage_data": [
					{
						"capability_id": usage.capability_id,
						"usage_count": usage.usage_count,
						"error_count": usage.error_count,
						"avg_response_time": usage.avg_response_time_ms
					}
					for usage in usage_data
				],
				"health_data": [
					{
						"capability_id": health.capability_id,
						"health_score": health.health_score,
						"performance_score": health.performance_score,
						"security_score": health.security_score
					}
					for health in health_data
				]
			}
			
		except Exception as e:
			print(self._log_composition_operation("get_analytics", f"Error: {e}"))
		
		return analytics
	
	async def _generate_optimization_recommendations(
		self,
		capability_ids: List[str],
		analytics: Dict[str, Any]
	) -> List[CompositionRecommendation]:
		"""Generate optimization recommendations."""
		recommendations = []
		
		# Analyze for redundant capabilities
		recommendation = CompositionRecommendation(
			recommendation_id=uuid7str(),
			recommendation_type=RecommendationType.OPTIMIZATION,
			title="Optimize Capability Dependencies",
			description="Consider consolidating similar capabilities to reduce complexity",
			affected_capabilities=capability_ids[:2],  # Simplified
			implementation_steps=[
				"Analyze overlapping functionality",
				"Identify consolidation opportunities",
				"Update composition configuration"
			],
			estimated_impact={
				"complexity_reduction": 0.15,
				"maintenance_reduction": 0.20,
				"performance_improvement": 0.10
			},
			confidence_score=0.75,
			priority=2
		)
		recommendations.append(recommendation)
		
		return recommendations
	
	async def _generate_security_recommendations(
		self,
		capability_ids: List[str],
		analytics: Dict[str, Any]
	) -> List[CompositionRecommendation]:
		"""Generate security recommendations."""
		recommendations = []
		
		# Check for auth_rbac integration
		recommendation = CompositionRecommendation(
			recommendation_id=uuid7str(),
			recommendation_type=RecommendationType.SECURITY,
			title="Enhance Security Integration",
			description="Ensure all capabilities integrate with APG auth_rbac",
			affected_capabilities=capability_ids,
			implementation_steps=[
				"Verify auth_rbac dependency",
				"Implement security policies",
				"Add audit logging"
			],
			estimated_impact={
				"security_improvement": 0.30,
				"compliance_score": 0.25
			},
			confidence_score=0.85,
			priority=1
		)
		recommendations.append(recommendation)
		
		return recommendations
	
	async def _generate_performance_recommendations(
		self,
		capability_ids: List[str],
		analytics: Dict[str, Any]
	) -> List[CompositionRecommendation]:
		"""Generate performance recommendations."""
		recommendations = []
		
		# Analyze response times
		high_latency_caps = []
		for usage in analytics.get("usage_data", []):
			if usage.get("avg_response_time", 0) > 500:  # >500ms
				high_latency_caps.append(usage["capability_id"])
		
		if high_latency_caps:
			recommendation = CompositionRecommendation(
				recommendation_id=uuid7str(),
				recommendation_type=RecommendationType.PERFORMANCE,
				title="Optimize High-Latency Capabilities",
				description="Some capabilities show high response times",
				affected_capabilities=high_latency_caps,
				implementation_steps=[
					"Enable caching",
					"Optimize database queries",
					"Consider async processing"
				],
				estimated_impact={
					"response_time_improvement": 0.40,
					"user_experience": 0.25
				},
				confidence_score=0.70,
				priority=2
			)
			recommendations.append(recommendation)
		
		return recommendations
	
	async def _generate_industry_recommendations(
		self,
		capability_ids: List[str],
		industry_focus: List[str]
	) -> List[CompositionRecommendation]:
		"""Generate industry-specific recommendations."""
		recommendations = []
		
		# Industry-specific capability suggestions
		if "healthcare" in industry_focus:
			recommendation = CompositionRecommendation(
				recommendation_id=uuid7str(),
				recommendation_type=RecommendationType.FEATURE_ENHANCEMENT,
				title="Healthcare Compliance Integration",
				description="Add HIPAA compliance and healthcare data management",
				affected_capabilities=[],
				implementation_steps=[
					"Add healthcare_compliance capability",
					"Implement data encryption",
					"Setup audit trails"
				],
				estimated_impact={
					"compliance_score": 0.50,
					"industry_alignment": 0.40
				},
				confidence_score=0.80,
				priority=1
			)
			recommendations.append(recommendation)
		
		return recommendations
	
	# =================================================================
	# Performance Impact Analysis
	# =================================================================
	
	async def analyze_performance_impact(
		self,
		capability_ids: List[str]
	) -> PerformanceImpact:
		"""
		Analyze performance impact of capability composition.
		
		Args:
			capability_ids: List of capability IDs
			
		Returns:
			PerformanceImpact analysis
		"""
		try:
			# Get capability metadata
			capabilities_query = select(CRCapability).where(
				and_(
					CRCapability.tenant_id == self.tenant_id,
					CRCapability.capability_id.in_(capability_ids)
				)
			)
			result = await self.db_session.execute(capabilities_query)
			capabilities = result.scalars().all()
			
			# Calculate estimated performance impact
			total_complexity = sum(cap.complexity_score for cap in capabilities)
			capability_count = len(capabilities)
			
			# Simplified performance calculations
			estimated_memory = capability_count * 50 + total_complexity * 25  # MB
			estimated_cpu = min(capability_count * 5 + total_complexity * 10, 100)  # %
			estimated_startup = capability_count * 100 + total_complexity * 50  # ms
			estimated_response = capability_count * 10 + total_complexity * 20  # ms
			
			scalability_score = max(0, 1.0 - (total_complexity / 100))
			
			return PerformanceImpact(
				memory_usage_mb=estimated_memory,
				cpu_usage_pct=estimated_cpu,
				network_bandwidth_mbps=capability_count * 2.5,
				disk_io_ops=capability_count * 100,
				startup_time_ms=estimated_startup,
				response_time_ms=estimated_response,
				scalability_score=scalability_score
			)
			
		except Exception as e:
			print(self._log_composition_operation("analyze_performance", f"Error: {e}"))
			return PerformanceImpact(
				memory_usage_mb=0.0,
				cpu_usage_pct=0.0,
				network_bandwidth_mbps=0.0,
				disk_io_ops=0,
				startup_time_ms=0.0,
				response_time_ms=0.0,
				scalability_score=1.0
			)
	
	# =================================================================
	# Complete Composition Validation
	# =================================================================
	
	async def validate_composition(
		self,
		capability_ids: List[str],
		composition_type: Optional[CRCompositionType] = None,
		industry_focus: Optional[List[str]] = None
	) -> CompositionValidationResult:
		"""
		Perform complete composition validation.
		
		Args:
			capability_ids: List of capability IDs to validate
			composition_type: Type of composition
			industry_focus: Industry requirements
			
		Returns:
			Complete validation results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_composition_operation("validate_composition", 
				f"Validating {len(capability_ids)} capabilities"))
			
			# Detect conflicts
			conflicts = await self.detect_conflicts(capability_ids)
			
			# Generate recommendations
			recommendations = await self.generate_composition_recommendations(
				capability_ids, composition_type, industry_focus
			)
			
			# Analyze performance impact
			performance_impact = await self.analyze_performance_impact(capability_ids)
			
			# Calculate validation score
			validation_score = await self._calculate_validation_score(
				conflicts, performance_impact, len(capability_ids)
			)
			
			# Determine if composition is valid
			is_valid = len([c for c in conflicts if c.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]]) == 0
			
			# Generate cost analysis (placeholder)
			cost_analysis = await self._generate_cost_analysis(capability_ids, performance_impact)
			
			# Generate deployment strategy
			deployment_strategy = await self._generate_deployment_strategy(capability_ids, conflicts)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_performance_metric("validate_composition", duration_ms))
			
			return CompositionValidationResult(
				is_valid=is_valid,
				validation_score=validation_score,
				conflicts=conflicts,
				recommendations=recommendations,
				performance_impact=performance_impact,
				cost_analysis=cost_analysis,
				deployment_strategy=deployment_strategy
			)
			
		except Exception as e:
			print(self._log_composition_operation("validate_composition", f"Error: {e}"))
			return CompositionValidationResult(
				is_valid=False,
				validation_score=0.0,
				conflicts=[],
				recommendations=[],
				performance_impact=PerformanceImpact(0,0,0,0,0,0,0),
				cost_analysis={},
				deployment_strategy={}
			)
	
	async def _calculate_validation_score(
		self,
		conflicts: List[ConflictReport],
		performance_impact: PerformanceImpact,
		capability_count: int
	) -> float:
		"""Calculate overall validation score."""
		base_score = 1.0
		
		# Deduct for conflicts
		for conflict in conflicts:
			if conflict.severity == ConflictSeverity.CRITICAL:
				base_score -= 0.25
			elif conflict.severity == ConflictSeverity.HIGH:
				base_score -= 0.15
			elif conflict.severity == ConflictSeverity.MEDIUM:
				base_score -= 0.10
			else:
				base_score -= 0.05
		
		# Adjust for performance
		if performance_impact.response_time_ms > 1000:
			base_score -= 0.10
		if performance_impact.memory_usage_mb > 1000:
			base_score -= 0.05
		
		# Adjust for scalability
		base_score *= performance_impact.scalability_score
		
		return max(0.0, min(1.0, base_score))
	
	async def _generate_cost_analysis(
		self,
		capability_ids: List[str],
		performance_impact: PerformanceImpact
	) -> Dict[str, Any]:
		"""Generate cost analysis for composition."""
		# Simplified cost calculation
		base_cost_per_capability = 10.0  # USD per month
		memory_cost_per_gb = 5.0
		cpu_cost_factor = 0.1
		
		base_cost = len(capability_ids) * base_cost_per_capability
		memory_cost = (performance_impact.memory_usage_mb / 1024) * memory_cost_per_gb
		cpu_cost = performance_impact.cpu_usage_pct * cpu_cost_factor
		
		total_monthly_cost = base_cost + memory_cost + cpu_cost
		
		return {
			"monthly_cost_usd": round(total_monthly_cost, 2),
			"cost_breakdown": {
				"base_cost": round(base_cost, 2),
				"memory_cost": round(memory_cost, 2),
				"cpu_cost": round(cpu_cost, 2)
			},
			"cost_per_capability": round(total_monthly_cost / len(capability_ids), 2),
			"optimization_potential": 0.15  # 15% potential savings
		}
	
	async def _generate_deployment_strategy(
		self,
		capability_ids: List[str],
		conflicts: List[ConflictReport]
	) -> Dict[str, Any]:
		"""Generate deployment strategy for composition."""
		strategy = {
			"deployment_type": "rolling",
			"phases": [
				{
					"phase": 1,
					"name": "Core Infrastructure",
					"capabilities": capability_ids[:len(capability_ids)//3],
					"estimated_time_minutes": 15
				},
				{
					"phase": 2, 
					"name": "Business Logic",
					"capabilities": capability_ids[len(capability_ids)//3:2*len(capability_ids)//3],
					"estimated_time_minutes": 20
				},
				{
					"phase": 3,
					"name": "User Interface",
					"capabilities": capability_ids[2*len(capability_ids)//3:],
					"estimated_time_minutes": 10
				}
			],
			"rollback_strategy": "automatic",
			"health_checks_enabled": True,
			"conflicts_to_resolve": len([c for c in conflicts if not c.auto_resolvable])
		}
		
		return strategy


# Service Factory
def get_composition_engine(
	db_session: AsyncSession,
	tenant_id: str,
	user_id: str,
	redis_client: Optional[Any] = None
) -> IntelligentCompositionEngine:
	"""Get or create composition engine instance."""
	return IntelligentCompositionEngine(
		db_session=db_session,
		tenant_id=tenant_id,
		user_id=user_id,
		redis_client=redis_client
	)