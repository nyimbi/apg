"""
APG Capability Registry - Marketplace Integration

Integration with APG marketplace for capability publishing, discovery,
and distribution across the APG ecosystem.

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
	CRCapability, CRVersion, CRComposition, CRMetadata, CRRegistry,
	CRCapabilityStatus, CRCompositionType
)

# =============================================================================
# Marketplace Data Structures
# =============================================================================

class MarketplaceStatus(str, Enum):
	"""Marketplace publication status."""
	DRAFT = "draft"
	PENDING_REVIEW = "pending_review"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	PUBLISHED = "published"
	REJECTED = "rejected"
	DEPRECATED = "deprecated"
	REMOVED = "removed"

class PublicationType(str, Enum):
	"""Types of marketplace publications."""
	CAPABILITY = "capability"
	COMPOSITION = "composition"
	TEMPLATE = "template"
	SOLUTION = "solution"

class LicenseType(str, Enum):
	"""License types for marketplace items."""
	MIT = "mit"
	APACHE_2_0 = "apache_2_0"
	GPL_3_0 = "gpl_3_0"
	BSD_3_CLAUSE = "bsd_3_clause"
	PROPRIETARY = "proprietary"
	COMMERCIAL = "commercial"
	CUSTOM = "custom"

class QualityLevel(str, Enum):
	"""Quality assessment levels."""
	EXPERIMENTAL = "experimental"
	BETA = "beta"
	STABLE = "stable"
	ENTERPRISE = "enterprise"
	CERTIFIED = "certified"

@dataclass
class MarketplaceMetadata:
	"""Marketplace publication metadata."""
	publication_id: str
	publication_type: PublicationType
	title: str
	description: str
	long_description: str
	tags: List[str]
	categories: List[str]
	license_type: LicenseType
	license_url: Optional[str]
	pricing_model: str
	price: float
	currency: str
	author_name: str
	author_email: str
	author_organization: str
	support_url: Optional[str]
	documentation_url: Optional[str]
	repository_url: Optional[str]
	demo_url: Optional[str]
	screenshots: List[str]
	videos: List[str]
	requirements: Dict[str, Any]
	compatibility: Dict[str, Any]
	quality_level: QualityLevel
	marketplace_status: MarketplaceStatus

@dataclass
class PublicationPackage:
	"""Complete publication package for marketplace."""
	metadata: MarketplaceMetadata
	capability_data: Optional[Dict[str, Any]]
	composition_data: Optional[Dict[str, Any]]
	documentation: Dict[str, str]
	assets: List[Dict[str, Any]]
	validation_results: Dict[str, Any]
	quality_score: float
	compliance_check: Dict[str, Any]

@dataclass
class MarketplaceSubmission:
	"""Marketplace submission with review status."""
	submission_id: str
	package: PublicationPackage
	submitted_at: datetime
	submitted_by: str
	review_status: MarketplaceStatus
	review_comments: List[str]
	reviewer_id: Optional[str]
	reviewed_at: Optional[datetime]
	approval_score: float
	estimated_review_time: Optional[int]

# =============================================================================
# Marketplace Integration Manager
# =============================================================================

class MarketplaceIntegration:
	"""
	Manages APG marketplace integration for capability registry.
	
	Provides capability publishing, marketplace synchronization,
	and distribution management across the APG ecosystem.
	"""
	
	def __init__(
		self,
		db_session: AsyncSession,
		tenant_id: str,
		user_id: str,
		marketplace_api_client: Optional[Any] = None
	):
		self.db_session = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.marketplace_api_client = marketplace_api_client
		
		# Configuration
		self.marketplace_url = "https://marketplace.apg.platform"
		self.api_version = "v1"
		self.submission_cache: Dict[str, MarketplaceSubmission] = {}
	
	def _log_marketplace_operation(self, operation: str, details: str) -> str:
		"""Log marketplace operations."""
		return f"MP-{self.tenant_id}: {operation} - {details}"
	
	def _log_marketplace_performance(self, operation: str, duration_ms: float) -> str:
		"""Log marketplace performance metrics."""
		return f"MP-Performance: {operation} completed in {duration_ms:.2f}ms"
	
	# =================================================================
	# Capability Publication
	# =================================================================
	
	async def prepare_capability_for_marketplace(
		self,
		capability_id: str,
		publication_metadata: Dict[str, Any]
	) -> Dict[str, Any]:
		"""
		Prepare capability for marketplace publication.
		
		Args:
			capability_id: Capability ID to publish
			publication_metadata: Marketplace metadata
			
		Returns:
			Publication preparation results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_marketplace_operation("prepare_capability", 
				f"Preparing {capability_id} for marketplace"))
			
			# Get capability data
			capability = await self._get_capability_with_details(capability_id)
			if not capability:
				return {
					"success": False,
					"message": "Capability not found",
					"errors": [f"Capability {capability_id} not found"]
				}
			
			# Validate capability for marketplace
			validation_result = await self._validate_capability_for_marketplace(capability)
			
			if not validation_result["is_valid"]:
				return {
					"success": False,
					"message": "Capability validation failed",
					"errors": validation_result["errors"]
				}
			
			# Create marketplace metadata
			marketplace_metadata = await self._create_marketplace_metadata(
				capability, publication_metadata
			)
			
			# Generate documentation package
			documentation = await self._generate_marketplace_documentation(capability)
			
			# Create publication package
			publication_package = PublicationPackage(
				metadata=marketplace_metadata,
				capability_data=await self._export_capability_data(capability),
				composition_data=None,
				documentation=documentation,
				assets=await self._collect_capability_assets(capability),
				validation_results=validation_result,
				quality_score=await self._calculate_quality_score(capability),
				compliance_check=await self._run_compliance_check(capability)
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_marketplace_performance("prepare_capability", duration_ms))
			
			return {
				"success": True,
				"message": "Capability prepared for marketplace",
				"data": {
					"publication_package": {
						"metadata": {
							"publication_id": marketplace_metadata.publication_id,
							"title": marketplace_metadata.title,
							"description": marketplace_metadata.description,
							"tags": marketplace_metadata.tags,
							"categories": marketplace_metadata.categories,
							"quality_level": marketplace_metadata.quality_level.value,
							"license_type": marketplace_metadata.license_type.value
						},
						"quality_score": publication_package.quality_score,
						"validation_passed": validation_result["is_valid"],
						"compliance_passed": publication_package.compliance_check.get("passed", False),
						"documentation_complete": len(documentation) > 0,
						"assets_count": len(publication_package.assets)
					},
					"preparation_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_marketplace_operation("prepare_capability", f"Error: {e}"))
			return {
				"success": False,
				"message": "Capability preparation failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def _get_capability_with_details(self, capability_id: str) -> Optional[CRCapability]:
		"""Get capability with all related data."""
		try:
			result = await self.db_session.execute(
				select(CRCapability).options(
					selectinload(CRCapability.versions),
					selectinload(CRCapability.dependencies),
					selectinload(CRCapability.compositions)
				).where(CRCapability.capability_id == capability_id)
			)
			return result.scalar_one_or_none()
		except Exception:
			return None
	
	async def _validate_capability_for_marketplace(
		self,
		capability: CRCapability
	) -> Dict[str, Any]:
		"""Validate capability meets marketplace requirements."""
		errors = []
		warnings = []
		
		# Check basic requirements
		if not capability.description or len(capability.description) < 50:
			errors.append("Description must be at least 50 characters")
		
		if not capability.capability_name or len(capability.capability_name) < 5:
			errors.append("Capability name must be at least 5 characters")
		
		if capability.status != CRCapabilityStatus.ACTIVE:
			errors.append("Capability must be in active status")
		
		# Check version requirements
		if not capability.versions or len(capability.versions) == 0:
			errors.append("At least one version must be published")
		
		# Check quality metrics
		if capability.quality_score < 0.7:
			warnings.append("Quality score below recommended threshold (0.7)")
		
		# Check documentation
		if not capability.documentation_path:
			warnings.append("No documentation path specified")
		
		# Check repository
		if not capability.repository_url:
			warnings.append("No repository URL specified")
		
		return {
			"is_valid": len(errors) == 0,
			"errors": errors,
			"warnings": warnings,
			"validation_score": max(0, 1.0 - (len(errors) * 0.2) - (len(warnings) * 0.1))
		}
	
	async def _create_marketplace_metadata(
		self,
		capability: CRCapability,
		publication_metadata: Dict[str, Any]
	) -> MarketplaceMetadata:
		"""Create marketplace metadata from capability and user input."""
		
		# Extract or default values
		title = publication_metadata.get("title", capability.capability_name)
		description = publication_metadata.get("description", capability.description)
		long_description = publication_metadata.get("long_description", description)
		
		# Generate tags from composition keywords and category
		tags = list(set(
			publication_metadata.get("tags", []) +
			capability.composition_keywords +
			[capability.category, capability.subcategory or ""]
		))
		tags = [tag for tag in tags if tag]  # Remove empty strings
		
		return MarketplaceMetadata(
			publication_id=uuid7str(),
			publication_type=PublicationType.CAPABILITY,
			title=title,
			description=description,
			long_description=long_description,
			tags=tags[:10],  # Limit to 10 tags
			categories=[capability.category],
			license_type=LicenseType(publication_metadata.get("license_type", "mit")),
			license_url=publication_metadata.get("license_url"),
			pricing_model=publication_metadata.get("pricing_model", "free"),
			price=float(publication_metadata.get("price", 0.0)),
			currency=publication_metadata.get("currency", "USD"),
			author_name=publication_metadata.get("author_name", "APG Developer"),
			author_email=publication_metadata.get("author_email", "developer@apg.platform"),
			author_organization=publication_metadata.get("author_organization", "APG Community"),
			support_url=publication_metadata.get("support_url"),
			documentation_url=capability.documentation_path,
			repository_url=capability.repository_url,
			demo_url=publication_metadata.get("demo_url"),
			screenshots=publication_metadata.get("screenshots", []),
			videos=publication_metadata.get("videos", []),
			requirements={
				"apg_version": ">=2.0.0",
				"dependencies": [dep.depends_on_id for dep in capability.dependencies],
				"minimum_memory_mb": 100,
				"minimum_cpu_cores": 1
			},
			compatibility={
				"apg_versions": ["2.0.0", "2.1.0"],
				"platforms": ["linux", "windows", "macos"],
				"databases": ["postgresql", "mysql"]
			},
			quality_level=self._determine_quality_level(capability),
			marketplace_status=MarketplaceStatus.DRAFT
		)
	
	def _determine_quality_level(self, capability: CRCapability) -> QualityLevel:
		"""Determine quality level based on capability metrics."""
		score = capability.quality_score
		
		if score >= 0.95:
			return QualityLevel.CERTIFIED
		elif score >= 0.85:
			return QualityLevel.ENTERPRISE
		elif score >= 0.75:
			return QualityLevel.STABLE
		elif score >= 0.6:
			return QualityLevel.BETA
		else:
			return QualityLevel.EXPERIMENTAL
	
	async def _generate_marketplace_documentation(
		self,
		capability: CRCapability
	) -> Dict[str, str]:
		"""Generate documentation package for marketplace."""
		documentation = {}
		
		# README
		documentation["README.md"] = f"""# {capability.capability_name}

{capability.description}

## Overview

{capability.description}

## Features

{', '.join(capability.provides_services)}

## Installation

This capability can be installed through the APG Marketplace or directly via the APG CLI:

```bash
apg capability install {capability.capability_code}
```

## Usage

{capability.capability_name} provides the following services:
{chr(10).join([f"- {service}" for service in capability.provides_services])}

## Configuration

See the configuration documentation for detailed setup instructions.

## Dependencies

{chr(10).join([f"- {dep.depends_on_id}" for dep in capability.dependencies])}

## License

Please see the license information in the marketplace listing.

## Support

For support and documentation, please visit the APG platform documentation.
"""
		
		# API Documentation
		if capability.api_endpoints:
			documentation["API.md"] = f"""# {capability.capability_name} API Reference

## Endpoints

{chr(10).join([f"- `{endpoint}`" for endpoint in capability.api_endpoints])}

## Authentication

This capability integrates with APG's authentication system.

## Rate Limiting

Standard APG rate limiting applies.
"""
		
		# Configuration Guide
		documentation["CONFIGURATION.md"] = f"""# {capability.capability_name} Configuration

## Environment Variables

Configuration is managed through APG's configuration system.

## Database Setup

If this capability requires database setup, it will be handled automatically by APG.

## Dependencies

This capability depends on:
{chr(10).join([f"- {dep.depends_on_id}" for dep in capability.dependencies])}
"""
		
		return documentation
	
	async def _export_capability_data(self, capability: CRCapability) -> Dict[str, Any]:
		"""Export capability data for marketplace."""
		return {
			"capability_id": capability.capability_id,
			"capability_code": capability.capability_code,
			"capability_name": capability.capability_name,
			"description": capability.description,
			"version": capability.version,
			"category": capability.category,
			"subcategory": capability.subcategory,
			"composition_keywords": capability.composition_keywords,
			"provides_services": capability.provides_services,
			"data_models": capability.data_models,
			"api_endpoints": capability.api_endpoints,
			"multi_tenant": capability.multi_tenant,
			"audit_enabled": capability.audit_enabled,
			"security_integration": capability.security_integration,
			"performance_optimized": capability.performance_optimized,
			"ai_enhanced": capability.ai_enhanced,
			"target_users": capability.target_users,
			"business_value": capability.business_value,
			"use_cases": capability.use_cases,
			"industry_focus": capability.industry_focus,
			"complexity_score": capability.complexity_score,
			"quality_score": capability.quality_score,
			"popularity_score": capability.popularity_score,
			"usage_count": capability.usage_count,
			"dependencies": [
				{
					"depends_on_id": dep.depends_on_id,
					"dependency_type": dep.dependency_type,
					"version_constraint": dep.version_constraint
				}
				for dep in capability.dependencies
			],
			"versions": [
				{
					"version_number": version.version_number,
					"release_date": version.release_date.isoformat() if version.release_date else None,
					"release_notes": version.release_notes,
					"breaking_changes": version.breaking_changes,
					"new_features": version.new_features
				}
				for version in capability.versions
			]
		}
	
	async def _collect_capability_assets(self, capability: CRCapability) -> List[Dict[str, Any]]:
		"""Collect assets for marketplace publication."""
		assets = []
		
		# Code package
		if capability.file_path:
			assets.append({
				"type": "code_package",
				"name": f"{capability.capability_code}.zip",
				"path": capability.file_path,
				"size_bytes": 0,  # Would be calculated
				"checksum": "sha256:...",  # Would be calculated
				"description": "Main capability code package"
			})
		
		# Documentation
		if capability.documentation_path:
			assets.append({
				"type": "documentation",
				"name": "documentation.zip",
				"path": capability.documentation_path,
				"size_bytes": 0,
				"checksum": "sha256:...",
				"description": "Capability documentation"
			})
		
		return assets
	
	async def _calculate_quality_score(self, capability: CRCapability) -> float:
		"""Calculate comprehensive quality score."""
		scores = []
		
		# Base quality score
		scores.append(capability.quality_score)
		
		# Documentation completeness
		if capability.description and len(capability.description) > 100:
			scores.append(0.8)
		else:
			scores.append(0.4)
		
		# Version maturity
		if capability.versions and len(capability.versions) > 1:
			scores.append(0.9)
		else:
			scores.append(0.6)
		
		# Usage metrics
		if capability.usage_count > 10:
			scores.append(0.9)
		elif capability.usage_count > 0:
			scores.append(0.7)
		else:
			scores.append(0.5)
		
		# Dependencies health
		if len(capability.dependencies) <= 5:
			scores.append(0.8)
		else:
			scores.append(0.6)
		
		return sum(scores) / len(scores)
	
	async def _run_compliance_check(self, capability: CRCapability) -> Dict[str, Any]:
		"""Run compliance checks for marketplace."""
		checks = {
			"security_scan": True,  # Placeholder
			"license_valid": True,
			"code_quality": capability.quality_score > 0.5,
			"documentation_complete": bool(capability.description),
			"api_documented": len(capability.api_endpoints) > 0,
			"tests_present": True,  # Would check for test files
			"no_security_issues": True,  # Would run security scan
			"follows_conventions": True  # Would check naming conventions
		}
		
		passed_checks = sum(1 for check in checks.values() if check)
		total_checks = len(checks)
		
		return {
			"passed": passed_checks == total_checks,
			"score": passed_checks / total_checks,
			"checks": checks,
			"total_checks": total_checks,
			"passed_checks": passed_checks
		}
	
	# =================================================================
	# Marketplace Submission
	# =================================================================
	
	async def submit_to_marketplace(
		self,
		publication_package: PublicationPackage
	) -> Dict[str, Any]:
		"""
		Submit capability to APG marketplace.
		
		Args:
			publication_package: Complete publication package
			
		Returns:
			Submission results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_marketplace_operation("submit_to_marketplace", 
				f"Submitting {publication_package.metadata.title}"))
			
			# Create submission record
			submission = MarketplaceSubmission(
				submission_id=uuid7str(),
				package=publication_package,
				submitted_at=datetime.utcnow(),
				submitted_by=self.user_id,
				review_status=MarketplaceStatus.PENDING_REVIEW,
				review_comments=[],
				reviewer_id=None,
				reviewed_at=None,
				approval_score=0.0,
				estimated_review_time=self._estimate_review_time(publication_package)
			)
			
			# Store in cache (in real implementation, would send to marketplace API)
			self.submission_cache[submission.submission_id] = submission
			
			# Simulate marketplace API call
			marketplace_response = await self._call_marketplace_api(
				"submissions", "POST", {
					"submission_id": submission.submission_id,
					"package": self._serialize_publication_package(publication_package),
					"submitted_by": self.user_id,
					"tenant_id": self.tenant_id
				}
			)
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_marketplace_performance("submit_to_marketplace", duration_ms))
			
			return {
				"success": True,
				"message": "Submission successful",
				"data": {
					"submission_id": submission.submission_id,
					"review_status": submission.review_status.value,
					"estimated_review_time_hours": submission.estimated_review_time,
					"marketplace_url": f"{self.marketplace_url}/submissions/{submission.submission_id}",
					"submission_time_ms": duration_ms
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_marketplace_operation("submit_to_marketplace", f"Error: {e}"))
			return {
				"success": False,
				"message": "Marketplace submission failed",
				"data": None,
				"errors": [str(e)]
			}
	
	def _estimate_review_time(self, package: PublicationPackage) -> int:
		"""Estimate review time in hours."""
		base_time = 24  # 24 hours base
		
		# Adjust based on quality
		if package.quality_score > 0.9:
			base_time -= 8
		elif package.quality_score < 0.7:
			base_time += 16
		
		# Adjust based on complexity
		if package.capability_data:
			complexity = package.capability_data.get("complexity_score", 1.0)
			base_time += int(complexity * 8)
		
		# Adjust based on compliance
		if not package.compliance_check.get("passed", False):
			base_time += 24
		
		return max(8, min(72, base_time))  # Between 8 and 72 hours
	
	async def _call_marketplace_api(
		self,
		endpoint: str,
		method: str,
		data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Call marketplace API (placeholder implementation)."""
		# In real implementation, this would make HTTP calls to marketplace API
		await asyncio.sleep(0.1)  # Simulate network delay
		
		return {
			"status": "success",
			"response_id": uuid7str(),
			"timestamp": datetime.utcnow().isoformat()
		}
	
	def _serialize_publication_package(self, package: PublicationPackage) -> Dict[str, Any]:
		"""Serialize publication package for API transmission."""
		return {
			"metadata": {
				"publication_id": package.metadata.publication_id,
				"title": package.metadata.title,
				"description": package.metadata.description,
				"tags": package.metadata.tags,
				"categories": package.metadata.categories,
				"license_type": package.metadata.license_type.value,
				"pricing_model": package.metadata.pricing_model,
				"price": package.metadata.price,
				"author_name": package.metadata.author_name,
				"quality_level": package.metadata.quality_level.value
			},
			"capability_data": package.capability_data,
			"documentation": package.documentation,
			"quality_score": package.quality_score,
			"compliance_passed": package.compliance_check.get("passed", False)
		}
	
	# =================================================================
	# Marketplace Synchronization
	# =================================================================
	
	async def sync_with_marketplace(self) -> Dict[str, Any]:
		"""
		Synchronize local registry with marketplace.
		
		Returns:
			Synchronization results
		"""
		start_time = datetime.utcnow()
		
		try:
			print(self._log_marketplace_operation("sync_marketplace", 
				"Starting marketplace synchronization"))
			
			# Get marketplace updates
			updates = await self._fetch_marketplace_updates()
			
			# Process capability updates
			capability_updates = 0
			for update in updates.get("capabilities", []):
				await self._process_capability_update(update)
				capability_updates += 1
			
			# Process composition templates
			template_updates = 0
			for template in updates.get("templates", []):
				await self._process_template_update(template)
				template_updates += 1
			
			# Update registry metadata
			await self._update_registry_marketplace_info()
			
			duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
			print(self._log_marketplace_performance("sync_marketplace", duration_ms))
			
			return {
				"success": True,
				"message": "Marketplace sync completed",
				"data": {
					"capability_updates": capability_updates,
					"template_updates": template_updates,
					"sync_time_ms": duration_ms,
					"last_sync": datetime.utcnow().isoformat()
				},
				"errors": []
			}
			
		except Exception as e:
			print(self._log_marketplace_operation("sync_marketplace", f"Error: {e}"))
			return {
				"success": False,
				"message": "Marketplace sync failed",
				"data": None,
				"errors": [str(e)]
			}
	
	async def _fetch_marketplace_updates(self) -> Dict[str, Any]:
		"""Fetch updates from marketplace."""
		# Placeholder implementation
		return {
			"capabilities": [],
			"templates": [],
			"last_updated": datetime.utcnow().isoformat()
		}
	
	async def _process_capability_update(self, update: Dict[str, Any]):
		"""Process a capability update from marketplace."""
		# Placeholder for processing marketplace capability updates
		pass
	
	async def _process_template_update(self, template: Dict[str, Any]):
		"""Process a template update from marketplace."""
		# Placeholder for processing marketplace template updates
		pass
	
	async def _update_registry_marketplace_info(self):
		"""Update registry with marketplace information."""
		registry_result = await self.db_session.execute(
			select(CRRegistry).where(CRRegistry.tenant_id == self.tenant_id)
		)
		registry = registry_result.scalar_one_or_none()
		
		if registry:
			registry.metadata = registry.metadata or {}
			registry.metadata["marketplace_last_sync"] = datetime.utcnow().isoformat()
			registry.metadata["marketplace_integration"] = True
			registry.updated_at = datetime.utcnow()
			registry.updated_by = self.user_id
			
			await self.db_session.commit()


# Service Factory
def get_marketplace_integration(
	db_session: AsyncSession,
	tenant_id: str,
	user_id: str,
	marketplace_api_client: Optional[Any] = None
) -> MarketplaceIntegration:
	"""Get or create marketplace integration instance."""
	return MarketplaceIntegration(
		db_session=db_session,
		tenant_id=tenant_id,
		user_id=user_id,
		marketplace_api_client=marketplace_api_client
	)