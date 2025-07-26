"""
APG Capability Registry - Version Management and Compatibility

Semantic versioning, compatibility analysis, and automated migration
path generation for APG capability evolution.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid_extensions import uuid7str
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
	CRCapability, CRVersion, CRDependency, CRComposition,
	CRCapabilityStatus, CRVersionConstraint
)

# =============================================================================
# Version Management Data Structures
# =============================================================================

class VersionChangeType(str, Enum):
	"""Types of version changes."""
	MAJOR = "major"
	MINOR = "minor"
	PATCH = "patch"
	PRERELEASE = "prerelease"
	BUILD = "build"

class CompatibilityLevel(str, Enum):
	"""Compatibility levels between versions."""
	FULLY_COMPATIBLE = "fully_compatible"
	BACKWARD_COMPATIBLE = "backward_compatible"
	FORWARD_COMPATIBLE = "forward_compatible"
	BREAKING_CHANGES = "breaking_changes"
	INCOMPATIBLE = "incompatible"

class MigrationComplexity(str, Enum):
	"""Migration complexity levels."""
	AUTOMATIC = "automatic"
	SIMPLE = "simple"
	MODERATE = "moderate"
	COMPLEX = "complex"
	MANUAL = "manual"

@dataclass
class SemanticVersion:
	"""Semantic version representation."""
	major: int
	minor: int
	patch: int
	prerelease: Optional[str] = None
	build: Optional[str] = None
	
	def __str__(self) -> str:
		version = f"{self.major}.{self.minor}.{self.patch}"
		if self.prerelease:
			version += f"-{self.prerelease}"
		if self.build:
			version += f"+{self.build}"
		return version
	
	def to_tuple(self) -> Tuple[int, int, int, str, str]:
		"""Convert to tuple for comparison."""
		return (
			self.major,
			self.minor, 
			self.patch,
			self.prerelease or "",
			self.build or ""
		)
	
	def __lt__(self, other: 'SemanticVersion') -> bool:
		return self.to_tuple() < other.to_tuple()
	
	def __eq__(self, other: 'SemanticVersion') -> bool:
		return self.to_tuple() == other.to_tuple()

@dataclass
class CompatibilityAnalysis:
	"""Results of version compatibility analysis."""
	from_version: SemanticVersion
	to_version: SemanticVersion
	compatibility_level: CompatibilityLevel
	breaking_changes: List[str]
	new_features: List[str]
	deprecations: List[str]
	api_changes: Dict[str, Any]
	migration_complexity: MigrationComplexity
	migration_steps: List[str]
	estimated_effort_hours: float
	risk_factors: List[str]

@dataclass
class MigrationPath:
	"""Migration path between versions."""
	path_id: str
	from_version: SemanticVersion
	to_version: SemanticVersion
	intermediate_versions: List[SemanticVersion]
	total_steps: int
	total_effort_hours: float
	requires_manual_intervention: bool
	migration_script: Optional[str]
	rollback_script: Optional[str]

# =============================================================================
# Version Manager
# =============================================================================

class VersionManager:
	"""
	Manages capability versions, compatibility analysis, and migration paths.
	
	Provides semantic versioning, automated compatibility analysis,
	and intelligent migration path generation.
	"""
	
	def __init__(
		self,
		db_session: AsyncSession,
		tenant_id: str,
		user_id: str
	):
		self.db_session = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id
		
		# Version comparison cache
		self._compatibility_cache: Dict[str, CompatibilityAnalysis] = {}
		self._migration_cache: Dict[str, MigrationPath] = {}
	
	def _log_version_operation(self, operation: str, details: str) -> str:
		"""Log version management operations."""
		return f"VM-{self.tenant_id}: {operation} - {details}"
	
	def _log_version_performance(self, operation: str, duration_ms: float) -> str:
		"""Log version management performance metrics."""
		return f"VM-Performance: {operation} completed in {duration_ms:.2f}ms"
	
	# =================================================================
	# Semantic Version Parsing and Validation
	# =================================================================
	
	def parse_semantic_version(self, version_string: str) -> Optional[SemanticVersion]:
		"""
		Parse semantic version string into SemanticVersion object.
		
		Args:
			version_string: Version string (e.g., "1.2.3-alpha.1+build.123")
			
		Returns:
			SemanticVersion object or None if invalid
		"""
		try:
			# Regex pattern for semantic versioning
			pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
			match = re.match(pattern, version_string.strip())
			
			if not match:
				return None
			
			major, minor, patch, prerelease, build = match.groups()
			
			return SemanticVersion(
				major=int(major),
				minor=int(minor),
				patch=int(patch),
				prerelease=prerelease,
				build=build
			)
			
		except Exception as e:
			print(self._log_version_operation("parse_version", f"Error parsing '{version_string}': {e}"))
			return None
	
	def validate_version_string(self, version_string: str) -> bool:
		"""Validate if version string follows semantic versioning."""
		return self.parse_semantic_version(version_string) is not None
	
	def generate_next_version(
		self,
		current_version: str,
		change_type: VersionChangeType,
		prerelease: Optional[str] = None
	) -> str:
		"""
		Generate next version based on change type.
		
		Args:
			current_version: Current version string
			change_type: Type of version change
			prerelease: Optional prerelease identifier
			
		Returns:
			Next version string
		"""
		try:
			sem_ver = self.parse_semantic_version(current_version)
			if not sem_ver:
				return "1.0.0"
			
			if change_type == VersionChangeType.MAJOR:
				next_ver = SemanticVersion(sem_ver.major + 1, 0, 0)
			elif change_type == VersionChangeType.MINOR:
				next_ver = SemanticVersion(sem_ver.major, sem_ver.minor + 1, 0)
			elif change_type == VersionChangeType.PATCH:
				next_ver = SemanticVersion(sem_ver.major, sem_ver.minor, sem_ver.patch + 1)
			elif change_type == VersionChangeType.PRERELEASE:
				next_ver = SemanticVersion(
					sem_ver.major, sem_ver.minor, sem_ver.patch,
					prerelease or "alpha.1"
				)
			else:  # BUILD
				next_ver = SemanticVersion(
					sem_ver.major, sem_ver.minor, sem_ver.patch,
					sem_ver.prerelease,
					f"build.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
				)
			
			return str(next_ver)
			
		except Exception as e:
			print(self._log_version_operation("generate_next_version", f"Error: {e}"))
			return "1.0.0"
	
	# =================================================================
	# Capability Version Management
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
	) -> Dict[str, Any]:
		"""
		Create new version for a capability.
		
		Args:
			capability_id: Capability ID
			version_number: Version number (semantic versioning)
			release_notes: Release notes
			breaking_changes: List of breaking changes
			new_features: List of new features
			deprecations: List of deprecations
			api_changes: API changes description
			
		Returns:
			Version creation result
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_version_operation("create_version", 
				f"Creating version {version_number} for {capability_id}"))
			
			# Validate version format
			sem_ver = self.parse_semantic_version(version_number)
			if not sem_ver:
				return {
					"success": False,
					"message": "Invalid semantic version format",
					"errors": [f"Version '{version_number}' does not follow semantic versioning"]
				}
			
			# Check if version already exists
			existing_version = await self.db_session.execute(
				select(CRVersion).where(
					and_(
						CRVersion.capability_id == capability_id,
						CRVersion.version_number == version_number
					)
				)
			)
			
			if existing_version.scalar_one_or_none():
				return {
					"success": False,
					"message": "Version already exists",
					"errors": [f"Version {version_number} already exists for capability"]
				}
			
			# Get capability
			capability_result = await self.db_session.execute(
				select(CRCapability).where(CRCapability.capability_id == capability_id)
			)
			capability = capability_result.scalar_one_or_none()
			
			if not capability:
				return {
					"success": False,
					"message": "Capability not found",
					"errors": [f"Capability {capability_id} not found"]
				}
			
			# Analyze compatibility with previous versions
			compatibility_analysis = await self._analyze_version_compatibility(
				capability_id, version_number, breaking_changes or [], api_changes or {}
			)
			
			# Create version record
			version_record = CRVersion(
				version_id=uuid7str(),
				capability_id=capability_id,
				version_number=version_number,
				major_version=sem_ver.major,
				minor_version=sem_ver.minor,
				patch_version=sem_ver.patch,
				prerelease=sem_ver.prerelease,
				build_metadata=sem_ver.build,
				release_date=datetime.utcnow(),
				release_notes=release_notes,
				breaking_changes=breaking_changes or [],
				deprecations=deprecations or [],
				new_features=new_features or [],
				compatible_versions=compatibility_analysis.get("compatible_versions", []),
				incompatible_versions=compatibility_analysis.get("incompatible_versions", []),
				api_changes=api_changes or {},
				backward_compatible=compatibility_analysis.get("backward_compatible", True),
				forward_compatible=compatibility_analysis.get("forward_compatible", False),
				quality_score=0.0,  # To be updated by quality analysis
				test_coverage=0.0,  # To be updated by test analysis
				documentation_score=0.0,  # To be updated by doc analysis
				security_audit_passed=False,  # To be updated by security audit
				status="active",
				support_level="full",
				created_at=datetime.utcnow(),
				created_by=self.user_id,
				metadata={"source": "version_manager", "analysis": compatibility_analysis}
			)
			
			self.db_session.add(version_record)
			
			# Update capability's current version
			capability.version = version_number
			capability.updated_at = datetime.utcnow()
			capability.updated_by = self.user_id
			
			await self.db_session.commit()
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_version_performance("create_version", duration_ms))
			
			return {
				"success": True,
				"message": f"Version {version_number} created successfully",
				"data": {
					"version_id": version_record.version_id,
					"version_number": version_number,
					"compatibility_analysis": compatibility_analysis,
					"creation_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			await self.db_session.rollback()
			print(self._log_version_operation("create_version", f"Error: {e}"))
			return {
				"success": False,
				"message": "Version creation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def _analyze_version_compatibility(
		self,
		capability_id: str,
		new_version: str,
		breaking_changes: List[str],
		api_changes: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Analyze compatibility with existing versions."""
		try:
			# Get all existing versions
			versions_result = await self.db_session.execute(
				select(CRVersion).where(
					CRVersion.capability_id == capability_id
				).order_by(desc(CRVersion.major_version, CRVersion.minor_version, CRVersion.patch_version))
			)
			existing_versions = versions_result.scalars().all()
			
			compatible_versions = []
			incompatible_versions = []
			
			new_sem_ver = self.parse_semantic_version(new_version)
			if not new_sem_ver:
				return {"compatible_versions": [], "incompatible_versions": []}
			
			for version in existing_versions:
				existing_sem_ver = self.parse_semantic_version(version.version_number)
				if not existing_sem_ver:
					continue
				
				# Determine compatibility
				is_backward_compatible = self._is_backward_compatible(
					existing_sem_ver, new_sem_ver, breaking_changes
				)
				
				if is_backward_compatible:
					compatible_versions.append(version.version_number)
				else:
					incompatible_versions.append(version.version_number)
			
			# Determine overall backward compatibility
			backward_compatible = len(breaking_changes) == 0
			forward_compatible = False  # Conservative approach
			
			return {
				"compatible_versions": compatible_versions,
				"incompatible_versions": incompatible_versions,
				"backward_compatible": backward_compatible,
				"forward_compatible": forward_compatible,
				"analysis_timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			print(self._log_version_operation("analyze_compatibility", f"Error: {e}"))
			return {"compatible_versions": [], "incompatible_versions": []}
	
	def _is_backward_compatible(
		self,
		old_version: SemanticVersion,
		new_version: SemanticVersion,
		breaking_changes: List[str]
	) -> bool:
		"""Determine if new version is backward compatible with old version."""
		# If there are breaking changes, not backward compatible
		if breaking_changes:
			return False
		
		# Major version change is not backward compatible
		if new_version.major > old_version.major:
			return False
		
		# Same major version with minor/patch increases are compatible
		if (new_version.major == old_version.major and 
			new_version.minor >= old_version.minor):
			return True
		
		return False
	
	# =================================================================
	# Compatibility Analysis
	# =================================================================
	
	async def analyze_compatibility(
		self,
		capability_id: str,
		from_version: str,
		to_version: str
	) -> CompatibilityAnalysis:
		"""
		Perform detailed compatibility analysis between two versions.
		
		Args:
			capability_id: Capability ID
			from_version: Source version
			to_version: Target version
			
		Returns:
			Detailed compatibility analysis
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_version_operation("analyze_compatibility", 
				f"Analyzing {from_version} -> {to_version} for {capability_id}"))
			
			# Parse versions
			from_sem_ver = self.parse_semantic_version(from_version)
			to_sem_ver = self.parse_semantic_version(to_version)
			
			if not from_sem_ver or not to_sem_ver:
				return CompatibilityAnalysis(
					from_version=from_sem_ver or SemanticVersion(0, 0, 0),
					to_version=to_sem_ver or SemanticVersion(0, 0, 0),
					compatibility_level=CompatibilityLevel.INCOMPATIBLE,
					breaking_changes=["Invalid version format"],
					new_features=[],
					deprecations=[],
					api_changes={},
					migration_complexity=MigrationComplexity.MANUAL,
					migration_steps=[],
					estimated_effort_hours=0.0,
					risk_factors=["Invalid version format"]
				)
			
			# Get version records
			from_version_record = await self._get_version_record(capability_id, from_version)
			to_version_record = await self._get_version_record(capability_id, to_version)
			
			# Determine compatibility level
			compatibility_level = self._determine_compatibility_level(
				from_sem_ver, to_sem_ver, 
				to_version_record.breaking_changes if to_version_record else []
			)
			
			# Extract changes
			breaking_changes = to_version_record.breaking_changes if to_version_record else []
			new_features = to_version_record.new_features if to_version_record else []
			deprecations = to_version_record.deprecations if to_version_record else []
			api_changes = to_version_record.api_changes if to_version_record else {}
			
			# Analyze migration complexity
			migration_complexity = self._analyze_migration_complexity(
				from_sem_ver, to_sem_ver, breaking_changes, api_changes
			)
			
			# Generate migration steps
			migration_steps = self._generate_migration_steps(
				from_sem_ver, to_sem_ver, breaking_changes, api_changes
			)
			
			# Estimate effort
			estimated_effort = self._estimate_migration_effort(
				migration_complexity, len(breaking_changes), len(api_changes)
			)
			
			# Identify risk factors
			risk_factors = self._identify_risk_factors(
				from_sem_ver, to_sem_ver, breaking_changes, deprecations
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_version_performance("analyze_compatibility", duration_ms))
			
			return CompatibilityAnalysis(
				from_version=from_sem_ver,
				to_version=to_sem_ver,
				compatibility_level=compatibility_level,
				breaking_changes=breaking_changes,
				new_features=new_features,
				deprecations=deprecations,
				api_changes=api_changes,
				migration_complexity=migration_complexity,
				migration_steps=migration_steps,
				estimated_effort_hours=estimated_effort,
				risk_factors=risk_factors
			)
			
		except Exception as e:
			print(self._log_version_operation("analyze_compatibility", f"Error: {e}"))
			return CompatibilityAnalysis(
				from_version=SemanticVersion(0, 0, 0),
				to_version=SemanticVersion(0, 0, 0),
				compatibility_level=CompatibilityLevel.INCOMPATIBLE,
				breaking_changes=[],
				new_features=[],
				deprecations=[],
				api_changes={},
				migration_complexity=MigrationComplexity.MANUAL,
				migration_steps=[],
				estimated_effort_hours=0.0,
				risk_factors=[]
			)
	
	async def _get_version_record(
		self,
		capability_id: str,
		version: str
	) -> Optional[CRVersion]:
		"""Get version record from database."""
		try:
			result = await self.db_session.execute(
				select(CRVersion).where(
					and_(
						CRVersion.capability_id == capability_id,
						CRVersion.version_number == version
					)
				)
			)
			return result.scalar_one_or_none()
		except Exception:
			return None
	
	def _determine_compatibility_level(
		self,
		from_version: SemanticVersion,
		to_version: SemanticVersion,
		breaking_changes: List[str]
	) -> CompatibilityLevel:
		"""Determine compatibility level between versions."""
		# If there are breaking changes, check severity
		if breaking_changes:
			if to_version.major > from_version.major:
				return CompatibilityLevel.INCOMPATIBLE
			else:
				return CompatibilityLevel.BREAKING_CHANGES
		
		# Major version increase
		if to_version.major > from_version.major:
			return CompatibilityLevel.BREAKING_CHANGES
		
		# Minor version increase (backward compatible)
		if (to_version.major == from_version.major and 
			to_version.minor > from_version.minor):
			return CompatibilityLevel.BACKWARD_COMPATIBLE
		
		# Patch version increase (fully compatible)
		if (to_version.major == from_version.major and 
			to_version.minor == from_version.minor and
			to_version.patch > from_version.patch):
			return CompatibilityLevel.FULLY_COMPATIBLE
		
		# Same version
		if from_version == to_version:
			return CompatibilityLevel.FULLY_COMPATIBLE
		
		return CompatibilityLevel.INCOMPATIBLE
	
	def _analyze_migration_complexity(
		self,
		from_version: SemanticVersion,
		to_version: SemanticVersion,
		breaking_changes: List[str],
		api_changes: Dict[str, Any]
	) -> MigrationComplexity:
		"""Analyze migration complexity."""
		# Automatic for patch versions with no breaking changes
		if (to_version.major == from_version.major and
			to_version.minor == from_version.minor and
			len(breaking_changes) == 0):
			return MigrationComplexity.AUTOMATIC
		
		# Simple for minor versions with no breaking changes
		if (to_version.major == from_version.major and
			len(breaking_changes) == 0):
			return MigrationComplexity.SIMPLE
		
		# Moderate for minor breaking changes
		if len(breaking_changes) <= 3:
			return MigrationComplexity.MODERATE
		
		# Complex for major version changes or many breaking changes
		if (to_version.major > from_version.major or 
			len(breaking_changes) > 5):
			return MigrationComplexity.COMPLEX
		
		return MigrationComplexity.MANUAL
	
	def _generate_migration_steps(
		self,
		from_version: SemanticVersion,
		to_version: SemanticVersion,
		breaking_changes: List[str],
		api_changes: Dict[str, Any]
	) -> List[str]:
		"""Generate migration steps."""
		steps = []
		
		# Basic steps
		steps.append(f"Backup current configuration")
		steps.append(f"Update capability from {from_version} to {to_version}")
		
		# Handle breaking changes
		for change in breaking_changes:
			steps.append(f"Address breaking change: {change}")
		
		# Handle API changes
		if api_changes:
			steps.append("Update API usage according to changes")
			for change_type, changes in api_changes.items():
				if isinstance(changes, list):
					for change in changes:
						steps.append(f"Update {change_type}: {change}")
		
		# Final steps
		steps.append("Test updated functionality")
		steps.append("Validate integration")
		steps.append("Update documentation")
		
		return steps
	
	def _estimate_migration_effort(
		self,
		complexity: MigrationComplexity,
		breaking_changes_count: int,
		api_changes_count: int
	) -> float:
		"""Estimate migration effort in hours."""
		base_effort = {
			MigrationComplexity.AUTOMATIC: 0.5,
			MigrationComplexity.SIMPLE: 2.0,
			MigrationComplexity.MODERATE: 8.0,
			MigrationComplexity.COMPLEX: 24.0,
			MigrationComplexity.MANUAL: 40.0
		}
		
		effort = base_effort.get(complexity, 8.0)
		
		# Add effort for breaking changes
		effort += breaking_changes_count * 2.0
		
		# Add effort for API changes
		effort += api_changes_count * 1.0
		
		return round(effort, 1)
	
	def _identify_risk_factors(
		self,
		from_version: SemanticVersion,
		to_version: SemanticVersion,
		breaking_changes: List[str],
		deprecations: List[str]
	) -> List[str]:
		"""Identify migration risk factors."""
		risks = []
		
		# Major version jump
		if to_version.major > from_version.major:
			risks.append("Major version upgrade may introduce significant changes")
		
		# Many breaking changes
		if len(breaking_changes) > 5:
			risks.append("High number of breaking changes")
		
		# Deprecations
		if len(deprecations) > 0:
			risks.append(f"{len(deprecations)} features deprecated")
		
		# Large version gap
		version_gap = (to_version.major - from_version.major) * 100 + \
					 (to_version.minor - from_version.minor) * 10 + \
					 (to_version.patch - from_version.patch)
		
		if version_gap > 50:
			risks.append("Large version gap may require intermediate upgrades")
		
		return risks

# Service Factory
def get_version_manager(
	db_session: AsyncSession,
	tenant_id: str,
	user_id: str
) -> VersionManager:
	"""Get or create version manager instance."""
	return VersionManager(
		db_session=db_session,
		tenant_id=tenant_id,
		user_id=user_id
	)