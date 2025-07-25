#!/usr/bin/env python3
"""
APG Capability Marketplace
==========================

Community-driven marketplace for discovering, sharing, and managing APG capabilities.
"""

import asyncio
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

def uuid7str() -> str:
	"""Generate a UUID7-style string using uuid4 as fallback"""
	return str(uuid.uuid4())

class CapabilityCategory(Enum):
	"""Capability categories for organization"""
	WEB_DEVELOPMENT = "web_development"
	AI_ML = "ai_ml" 
	IOT_HARDWARE = "iot_hardware"
	BUSINESS_INTELLIGENCE = "business_intelligence"
	CLOUD_INTEGRATION = "cloud_integration"
	SECURITY_COMPLIANCE = "security_compliance"
	PERFORMANCE_MONITORING = "performance_monitoring"
	DEVOPS_DEPLOYMENT = "devops_deployment"
	DATA_PROCESSING = "data_processing"
	MOBILE_DEVELOPMENT = "mobile_development"
	BLOCKCHAIN = "blockchain"
	GAMING = "gaming"
	HEALTHCARE = "healthcare"
	FINANCE = "finance"
	EDUCATION = "education"
	CUSTOM = "custom"

class CapabilityStatus(Enum):
	"""Capability lifecycle status"""
	DRAFT = "draft"
	PUBLISHED = "published"
	VERIFIED = "verified"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"

class LicenseType(Enum):
	"""Capability licensing options"""
	MIT = "mit"
	APACHE_2 = "apache_2"
	GPL_V3 = "gpl_v3"
	BSD = "bsd"
	PROPRIETARY = "proprietary"
	CUSTOM = "custom"

@dataclass
class CapabilityVersion:
	"""Version information for capabilities"""
	version: str
	changelog: str = ""
	compatibility: List[str] = field(default_factory=list)
	breaking_changes: bool = False
	release_date: datetime = field(default_factory=datetime.utcnow)
	download_count: int = 0

@dataclass
class CapabilityRating:
	"""User rating and review for capabilities"""
	id: str = field(default_factory=uuid7str)
	user_id: str = ""
	capability_id: str = ""
	rating: int = 5  # 1-5 stars
	review: str = ""
	helpful_votes: int = 0
	created_at: datetime = field(default_factory=datetime.utcnow)
	verified_purchase: bool = False

@dataclass
class CapabilityDependency:
	"""Dependency specification for capabilities"""
	name: str
	version_constraint: str = "*"
	optional: bool = False
	description: str = ""

@dataclass
class CapabilityMetrics:
	"""Usage and performance metrics"""
	download_count: int = 0
	active_installations: int = 0
	average_rating: float = 0.0
	rating_count: int = 0
	last_updated: datetime = field(default_factory=datetime.utcnow)
	compatibility_score: float = 1.0
	performance_score: float = 1.0

@dataclass
class MarketplaceCapability:
	"""Complete capability package for marketplace"""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	display_name: str = ""
	description: str = ""
	detailed_description: str = ""
	
	# Categorization
	category: CapabilityCategory = CapabilityCategory.CUSTOM
	tags: List[str] = field(default_factory=list)
	keywords: List[str] = field(default_factory=list)
	
	# Author and publishing
	author: str = ""
	author_email: str = ""
	organization: str = ""
	license: LicenseType = LicenseType.MIT
	homepage: str = ""
	repository: str = ""
	
	# Versioning
	current_version: str = "1.0.0"
	versions: List[CapabilityVersion] = field(default_factory=list)
	
	# Dependencies and compatibility
	dependencies: List[CapabilityDependency] = field(default_factory=list)
	apg_version_min: str = "1.0.0"
	apg_version_max: str = "*"
	platforms: List[str] = field(default_factory=lambda: ["linux", "windows", "macos"])
	
	# Content
	capability_code: str = ""
	example_usage: str = ""
	documentation: str = ""
	test_cases: List[str] = field(default_factory=list)
	
	# Marketplace metadata
	status: CapabilityStatus = CapabilityStatus.DRAFT
	featured: bool = False
	verified: bool = False
	premium: bool = False
	price: float = 0.0  # 0 for free
	
	# Analytics
	metrics: CapabilityMetrics = field(default_factory=CapabilityMetrics)
	ratings: List[CapabilityRating] = field(default_factory=list)
	
	# Timestamps
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	published_at: Optional[datetime] = None

class CapabilityValidator:
	"""Validates capability packages for security and quality"""
	
	def __init__(self):
		self.logger = logging.getLogger("capability_validator")
		self.security_patterns = self._load_security_patterns()
		self.quality_checks = self._load_quality_checks()
	
	def _load_security_patterns(self) -> List[str]:
		"""Load security anti-patterns to check for"""
		return [
			r"eval\s*\(",
			r"exec\s*\(",
			r"__import__\s*\(",
			r"subprocess\.call",
			r"os\.system",
			r"shell=True",
			r"pickle\.loads",
			r"input\s*\(",  # Potentially dangerous in certain contexts
		]
	
	def _load_quality_checks(self) -> Dict[str, Any]:
		"""Load quality check configuration"""
		return {
			'min_description_length': 50,
			'max_code_lines': 10000,
			'required_fields': ['name', 'description', 'author', 'capability_code'],
			'max_dependencies': 50,
			'min_test_coverage': 0.7
		}
	
	async def validate_capability(self, capability: MarketplaceCapability) -> Dict[str, Any]:
		"""Comprehensive capability validation"""
		results = {
			'valid': True,
			'errors': [],
			'warnings': [],
			'score': 0.0,
			'security_issues': [],
			'quality_issues': []
		}
		
		# Security validation
		security_results = await self._validate_security(capability)
		results['security_issues'] = security_results['issues']
		if security_results['critical_issues']:
			results['valid'] = False
			results['errors'].extend(security_results['critical_issues'])
		
		# Quality validation
		quality_results = await self._validate_quality(capability)
		results['quality_issues'] = quality_results['issues']
		results['warnings'].extend(quality_results['warnings'])
		
		# Dependency validation
		dependency_results = await self._validate_dependencies(capability)
		results['warnings'].extend(dependency_results['warnings'])
		if dependency_results['errors']:
			results['errors'].extend(dependency_results['errors'])
		
		# Calculate overall score
		results['score'] = self._calculate_quality_score(capability, results)
		
		return results
	
	async def _validate_security(self, capability: MarketplaceCapability) -> Dict[str, Any]:
		"""Security validation checks"""
		import re
		
		issues = []
		critical_issues = []
		
		code = capability.capability_code
		
		# Check for dangerous patterns
		for pattern in self.security_patterns:
			matches = re.findall(pattern, code, re.IGNORECASE)
			if matches:
				issue = f"Potentially dangerous pattern found: {pattern}"
				if pattern in [r"eval\s*\(", r"exec\s*\(", r"os\.system"]:
					critical_issues.append(issue)
				else:
					issues.append(issue)
		
		# Check for hardcoded secrets
		secret_patterns = [
			r"password\s*=\s*['\"][^'\"]+['\"]",
			r"api_key\s*=\s*['\"][^'\"]+['\"]",
			r"secret\s*=\s*['\"][^'\"]+['\"]",
			r"token\s*=\s*['\"][^'\"]+['\"]"
		]
		
		for pattern in secret_patterns:
			if re.search(pattern, code, re.IGNORECASE):
				issues.append(f"Potential hardcoded secret detected: {pattern}")
		
		# Check imports
		import_pattern = r"(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)"
		imports = re.findall(import_pattern, code)
		
		dangerous_imports = ['os', 'subprocess', 'sys', 'importlib', '__builtin__']
		for imp in imports:
			if imp in dangerous_imports:
				issues.append(f"Potentially dangerous import: {imp}")
		
		return {
			'issues': issues,
			'critical_issues': critical_issues,
			'safe': len(critical_issues) == 0
		}
	
	async def _validate_quality(self, capability: MarketplaceCapability) -> Dict[str, Any]:
		"""Quality validation checks"""
		issues = []
		warnings = []
		
		# Check required fields
		for field in self.quality_checks['required_fields']:
			if not getattr(capability, field, None):
				issues.append(f"Missing required field: {field}")
		
		# Check description length
		if len(capability.description) < self.quality_checks['min_description_length']:
			warnings.append("Description is too short, consider adding more details")
		
		# Check code length
		code_lines = len(capability.capability_code.split('\n'))
		if code_lines > self.quality_checks['max_code_lines']:
			warnings.append(f"Code is very long ({code_lines} lines), consider splitting into modules")
		
		# Check dependencies
		if len(capability.dependencies) > self.quality_checks['max_dependencies']:
			warnings.append("Too many dependencies, consider reducing complexity")
		
		# Check for documentation
		if not capability.documentation:
			warnings.append("No documentation provided")
		
		if not capability.example_usage:
			warnings.append("No example usage provided")
		
		# Check for tests
		if not capability.test_cases:
			warnings.append("No test cases provided")
		
		return {
			'issues': issues,
			'warnings': warnings,
			'quality_score': self._calculate_code_quality(capability)
		}
	
	async def _validate_dependencies(self, capability: MarketplaceCapability) -> Dict[str, Any]:
		"""Validate capability dependencies"""
		warnings = []
		errors = []
		
		for dep in capability.dependencies:
			# Check if dependency name is valid
			if not dep.name or not dep.name.replace('_', '').replace('-', '').isalnum():
				errors.append(f"Invalid dependency name: {dep.name}")
			
			# Check version constraint format
			if dep.version_constraint and dep.version_constraint != "*":
				# Simple version validation
				if not any(op in dep.version_constraint for op in ['>=', '<=', '==', '>', '<', '~', '^']):
					warnings.append(f"Version constraint for {dep.name} may be invalid: {dep.version_constraint}")
		
		return {
			'warnings': warnings,
			'errors': errors
		}
	
	def _calculate_code_quality(self, capability: MarketplaceCapability) -> float:
		"""Calculate code quality score"""
		score = 1.0
		
		# Deduct for missing documentation
		if not capability.documentation:
			score -= 0.2
		
		if not capability.example_usage:
			score -= 0.1
		
		if not capability.test_cases:
			score -= 0.2
		
		# Code analysis
		code = capability.capability_code
		if code:
			lines = code.split('\n')
			non_empty_lines = [line for line in lines if line.strip()]
			
			# Check comment ratio
			comment_lines = [line for line in non_empty_lines if line.strip().startswith('#')]
			if non_empty_lines:
				comment_ratio = len(comment_lines) / len(non_empty_lines)
				if comment_ratio < 0.1:
					score -= 0.1
		
		return max(0.0, score)
	
	def _calculate_quality_score(self, capability: MarketplaceCapability, validation_results: Dict[str, Any]) -> float:
		"""Calculate overall quality score"""
		base_score = 100.0
		
		# Deduct for errors and warnings
		base_score -= len(validation_results['errors']) * 20
		base_score -= len(validation_results['warnings']) * 5
		base_score -= len(validation_results['security_issues']) * 10
		
		# Quality bonus
		quality_results = next((r for r in validation_results.get('quality_issues', []) 
							   if isinstance(r, dict) and 'quality_score' in r), {})
		if quality_results:
			base_score *= quality_results['quality_score']
		
		return max(0.0, min(100.0, base_score))

class CapabilityDiscovery:
	"""Intelligent capability discovery and recommendation system"""
	
	def __init__(self, marketplace: 'CapabilityMarketplace'):
		self.marketplace = marketplace
		self.logger = logging.getLogger("capability_discovery")
	
	async def search_capabilities(
		self, 
		query: str = "",
		category: Optional[CapabilityCategory] = None,
		tags: List[str] = None,
		min_rating: float = 0.0,
		max_results: int = 50
	) -> List[MarketplaceCapability]:
		"""Search capabilities with intelligent ranking"""
		
		capabilities = await self.marketplace.list_capabilities()
		
		# Apply filters
		filtered = []
		for cap in capabilities:
			# Category filter
			if category and cap.category != category:
				continue
			
			# Rating filter
			if cap.metrics.average_rating < min_rating:
				continue
			
			# Tag filter
			if tags and not any(tag in cap.tags for tag in tags):
				continue
			
			# Query filter
			if query:
				score = self._calculate_relevance_score(cap, query)
				if score > 0.1:  # Minimum relevance threshold
					filtered.append((cap, score))
			else:
				filtered.append((cap, 1.0))
		
		# Sort by relevance and popularity
		if query:
			filtered.sort(key=lambda x: (x[1], x[0].metrics.average_rating, x[0].metrics.download_count), reverse=True)
		else:
			filtered.sort(key=lambda x: (x[0].metrics.average_rating, x[0].metrics.download_count), reverse=True)
		
		return [cap for cap, score in filtered[:max_results]]
	
	def _calculate_relevance_score(self, capability: MarketplaceCapability, query: str) -> float:
		"""Calculate relevance score for search query"""
		query_lower = query.lower()
		score = 0.0
		
		# Name match (highest weight)
		if query_lower in capability.name.lower():
			score += 1.0
		
		# Display name match
		if query_lower in capability.display_name.lower():
			score += 0.8
		
		# Description match
		if query_lower in capability.description.lower():
			score += 0.5
		
		# Tags match
		for tag in capability.tags:
			if query_lower in tag.lower():
				score += 0.6
		
		# Keywords match
		for keyword in capability.keywords:
			if query_lower in keyword.lower():
				score += 0.7
		
		# Detailed description match (lower weight)
		if query_lower in capability.detailed_description.lower():
			score += 0.3
		
		return min(score, 2.0)  # Cap at 2.0
	
	async def get_recommendations(
		self, 
		based_on_capability: Optional[str] = None,
		user_history: List[str] = None,
		project_context: Dict[str, Any] = None,
		limit: int = 10
	) -> List[MarketplaceCapability]:
		"""Get personalized capability recommendations"""
		
		recommendations = []
		capabilities = await self.marketplace.list_capabilities()
		
		if based_on_capability:
			# Find similar capabilities
			base_cap = await self.marketplace.get_capability(based_on_capability)
			if base_cap:
				similar = self._find_similar_capabilities(base_cap, capabilities)
				recommendations.extend(similar[:limit//2])
		
		if user_history:
			# Recommend based on user's download history
			history_based = self._recommend_from_history(user_history, capabilities)
			recommendations.extend(history_based[:limit//2])
		
		if project_context:
			# Recommend based on project requirements
			context_based = self._recommend_from_context(project_context, capabilities)
			recommendations.extend(context_based[:limit//3])
		
		# Add trending capabilities
		trending = self._get_trending_capabilities(capabilities)
		recommendations.extend(trending[:limit//3])
		
		# Remove duplicates and sort
		seen = set()
		unique_recommendations = []
		for cap in recommendations:
			if cap.id not in seen:
				seen.add(cap.id)
				unique_recommendations.append(cap)
		
		# Sort by popularity and rating
		unique_recommendations.sort(
			key=lambda x: (x.metrics.average_rating, x.metrics.download_count), 
			reverse=True
		)
		
		return unique_recommendations[:limit]
	
	def _find_similar_capabilities(self, base_cap: MarketplaceCapability, all_caps: List[MarketplaceCapability]) -> List[MarketplaceCapability]:
		"""Find capabilities similar to the base capability"""
		similar = []
		
		for cap in all_caps:
			if cap.id == base_cap.id:
				continue
			
			similarity_score = 0.0
			
			# Category match
			if cap.category == base_cap.category:
				similarity_score += 0.4
			
			# Tag overlap
			common_tags = set(cap.tags) & set(base_cap.tags)
			if base_cap.tags:
				tag_similarity = len(common_tags) / len(set(cap.tags) | set(base_cap.tags))
				similarity_score += tag_similarity * 0.3
			
			# Keyword overlap
			common_keywords = set(cap.keywords) & set(base_cap.keywords)
			if base_cap.keywords:
				keyword_similarity = len(common_keywords) / len(set(cap.keywords) | set(base_cap.keywords))
				similarity_score += keyword_similarity * 0.2
			
			# Dependencies overlap
			cap_dep_names = {dep.name for dep in cap.dependencies}
			base_dep_names = {dep.name for dep in base_cap.dependencies}
			if base_dep_names:
				dep_similarity = len(cap_dep_names & base_dep_names) / len(cap_dep_names | base_dep_names)
				similarity_score += dep_similarity * 0.1
			
			if similarity_score > 0.3:  # Minimum similarity threshold
				similar.append((cap, similarity_score))
		
		similar.sort(key=lambda x: x[1], reverse=True)
		return [cap for cap, score in similar]
	
	def _recommend_from_history(self, user_history: List[str], all_caps: List[MarketplaceCapability]) -> List[MarketplaceCapability]:
		"""Recommend capabilities based on user's download history"""
		# Get categories and tags from user's history
		user_categories = set()
		user_tags = set()
		
		for cap_id in user_history:
			cap = next((c for c in all_caps if c.id == cap_id), None)
			if cap:
				user_categories.add(cap.category)
				user_tags.update(cap.tags)
		
		# Find capabilities in similar categories with similar tags
		recommendations = []
		for cap in all_caps:
			if cap.id in user_history:
				continue
			
			score = 0.0
			if cap.category in user_categories:
				score += 0.5
			
			tag_overlap = len(set(cap.tags) & user_tags)
			if user_tags:
				score += (tag_overlap / len(user_tags)) * 0.5
			
			if score > 0.3:
				recommendations.append((cap, score))
		
		recommendations.sort(key=lambda x: x[1], reverse=True)
		return [cap for cap, score in recommendations]
	
	def _recommend_from_context(self, project_context: Dict[str, Any], all_caps: List[MarketplaceCapability]) -> List[MarketplaceCapability]:
		"""Recommend capabilities based on project context"""
		project_type = project_context.get('type', '').lower()
		required_features = project_context.get('features', [])
		tech_stack = project_context.get('tech_stack', [])
		
		recommendations = []
		
		for cap in all_caps:
			score = 0.0
			
			# Match project type with capability category
			if project_type in cap.category.value.lower():
				score += 0.4
			
			# Match required features with capability tags
			for feature in required_features:
				if any(feature.lower() in tag.lower() for tag in cap.tags):
					score += 0.2
			
			# Match tech stack with capability keywords
			for tech in tech_stack:
				if any(tech.lower() in keyword.lower() for keyword in cap.keywords):
					score += 0.1
			
			if score > 0.2:
				recommendations.append((cap, score))
		
		recommendations.sort(key=lambda x: x[1], reverse=True)
		return [cap for cap, score in recommendations]
	
	def _get_trending_capabilities(self, all_caps: List[MarketplaceCapability]) -> List[MarketplaceCapability]:
		"""Get trending capabilities based on recent activity"""
		# Calculate trending score based on recent downloads and ratings
		recent_cutoff = datetime.utcnow() - timedelta(days=30)
		
		trending = []
		for cap in all_caps:
			# Simple trending calculation (would be more sophisticated in real system)
			recent_downloads = cap.metrics.download_count  # Simplified
			rating_momentum = cap.metrics.average_rating * cap.metrics.rating_count
			
			trend_score = recent_downloads * 0.7 + rating_momentum * 0.3
			trending.append((cap, trend_score))
		
		trending.sort(key=lambda x: x[1], reverse=True)
		return [cap for cap, score in trending]

class CapabilityMarketplace:
	"""Main marketplace system for managing capabilities"""
	
	def __init__(self, storage_path: str = "./marketplace_data"):
		self.storage_path = Path(storage_path)
		self.storage_path.mkdir(exist_ok=True)
		
		self.capabilities: Dict[str, MarketplaceCapability] = {}
		self.validator = CapabilityValidator()
		self.discovery = CapabilityDiscovery(self)
		
		# Indexes for fast searching
		self.category_index: Dict[CapabilityCategory, Set[str]] = {}
		self.tag_index: Dict[str, Set[str]] = {}
		self.author_index: Dict[str, Set[str]] = {}
		
		self.logger = logging.getLogger("capability_marketplace")
		
		# Load existing capabilities
		asyncio.create_task(self._load_capabilities())
	
	async def _load_capabilities(self):
		"""Load capabilities from storage"""
		capabilities_file = self.storage_path / "capabilities.json"
		if capabilities_file.exists():
			try:
				with open(capabilities_file, 'r') as f:
					data = json.load(f)
				
				for cap_data in data:
					capability = self._deserialize_capability(cap_data)
					self.capabilities[capability.id] = capability
					self._update_indexes(capability)
				
				self.logger.info(f"Loaded {len(self.capabilities)} capabilities from storage")
			except Exception as e:
				self.logger.error(f"Failed to load capabilities: {e}")
	
	async def _save_capabilities(self):
		"""Save capabilities to storage"""
		capabilities_file = self.storage_path / "capabilities.json"
		try:
			data = [self._serialize_capability(cap) for cap in self.capabilities.values()]
			with open(capabilities_file, 'w') as f:
				json.dump(data, f, indent=2, default=str)
		except Exception as e:
			self.logger.error(f"Failed to save capabilities: {e}")
	
	def _serialize_capability(self, capability: MarketplaceCapability) -> Dict[str, Any]:
		"""Serialize capability to JSON-compatible format"""
		return {
			'id': capability.id,
			'name': capability.name,
			'display_name': capability.display_name,
			'description': capability.description,
			'detailed_description': capability.detailed_description,
			'category': capability.category.value,
			'tags': capability.tags,
			'keywords': capability.keywords,
			'author': capability.author,
			'author_email': capability.author_email,
			'organization': capability.organization,
			'license': capability.license.value,
			'homepage': capability.homepage,
			'repository': capability.repository,
			'current_version': capability.current_version,
			'versions': [self._serialize_version(v) for v in capability.versions],
			'dependencies': [self._serialize_dependency(d) for d in capability.dependencies],
			'apg_version_min': capability.apg_version_min,
			'apg_version_max': capability.apg_version_max,
			'platforms': capability.platforms,
			'capability_code': capability.capability_code,
			'example_usage': capability.example_usage,
			'documentation': capability.documentation,
			'test_cases': capability.test_cases,
			'status': capability.status.value,
			'featured': capability.featured,
			'verified': capability.verified,
			'premium': capability.premium,
			'price': capability.price,
			'metrics': self._serialize_metrics(capability.metrics),
			'ratings': [self._serialize_rating(r) for r in capability.ratings],
			'created_at': capability.created_at.isoformat(),
			'updated_at': capability.updated_at.isoformat(),
			'published_at': capability.published_at.isoformat() if capability.published_at else None
		}
	
	def _deserialize_capability(self, data: Dict[str, Any]) -> MarketplaceCapability:
		"""Deserialize capability from JSON format"""
		return MarketplaceCapability(
			id=data['id'],
			name=data['name'],
			display_name=data.get('display_name', data['name']),
			description=data['description'],
			detailed_description=data.get('detailed_description', ''),
			category=CapabilityCategory(data.get('category', 'custom')),
			tags=data.get('tags', []),
			keywords=data.get('keywords', []),
			author=data.get('author', ''),
			author_email=data.get('author_email', ''),
			organization=data.get('organization', ''),
			license=LicenseType(data.get('license', 'mit')),
			homepage=data.get('homepage', ''),
			repository=data.get('repository', ''),
			current_version=data.get('current_version', '1.0.0'),
			versions=[self._deserialize_version(v) for v in data.get('versions', [])],
			dependencies=[self._deserialize_dependency(d) for d in data.get('dependencies', [])],
			apg_version_min=data.get('apg_version_min', '1.0.0'),
			apg_version_max=data.get('apg_version_max', '*'),
			platforms=data.get('platforms', ['linux', 'windows', 'macos']),
			capability_code=data.get('capability_code', ''),
			example_usage=data.get('example_usage', ''),
			documentation=data.get('documentation', ''),
			test_cases=data.get('test_cases', []),
			status=CapabilityStatus(data.get('status', 'draft')),
			featured=data.get('featured', False),
			verified=data.get('verified', False),
			premium=data.get('premium', False),
			price=data.get('price', 0.0),
			metrics=self._deserialize_metrics(data.get('metrics', {})),
			ratings=[self._deserialize_rating(r) for r in data.get('ratings', [])],
			created_at=datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat())),
			updated_at=datetime.fromisoformat(data.get('updated_at', datetime.utcnow().isoformat())),
			published_at=datetime.fromisoformat(data['published_at']) if data.get('published_at') else None
		)
	
	def _serialize_version(self, version: CapabilityVersion) -> Dict[str, Any]:
		"""Serialize capability version"""
		return {
			'version': version.version,
			'changelog': version.changelog,
			'compatibility': version.compatibility,
			'breaking_changes': version.breaking_changes,
			'release_date': version.release_date.isoformat(),
			'download_count': version.download_count
		}
	
	def _deserialize_version(self, data: Dict[str, Any]) -> CapabilityVersion:
		"""Deserialize capability version"""
		return CapabilityVersion(
			version=data['version'],
			changelog=data.get('changelog', ''),
			compatibility=data.get('compatibility', []),
			breaking_changes=data.get('breaking_changes', False),
			release_date=datetime.fromisoformat(data.get('release_date', datetime.utcnow().isoformat())),
			download_count=data.get('download_count', 0)
		)
	
	def _serialize_dependency(self, dependency: CapabilityDependency) -> Dict[str, Any]:
		"""Serialize capability dependency"""
		return {
			'name': dependency.name,
			'version_constraint': dependency.version_constraint,
			'optional': dependency.optional,
			'description': dependency.description
		}
	
	def _deserialize_dependency(self, data: Dict[str, Any]) -> CapabilityDependency:
		"""Deserialize capability dependency"""
		return CapabilityDependency(
			name=data['name'],
			version_constraint=data.get('version_constraint', '*'),
			optional=data.get('optional', False),
			description=data.get('description', '')
		)
	
	def _serialize_metrics(self, metrics: CapabilityMetrics) -> Dict[str, Any]:
		"""Serialize capability metrics"""
		return {
			'download_count': metrics.download_count,
			'active_installations': metrics.active_installations,
			'average_rating': metrics.average_rating,
			'rating_count': metrics.rating_count,
			'last_updated': metrics.last_updated.isoformat(),
			'compatibility_score': metrics.compatibility_score,
			'performance_score': metrics.performance_score
		}
	
	def _deserialize_metrics(self, data: Dict[str, Any]) -> CapabilityMetrics:
		"""Deserialize capability metrics"""
		return CapabilityMetrics(
			download_count=data.get('download_count', 0),
			active_installations=data.get('active_installations', 0),
			average_rating=data.get('average_rating', 0.0),
			rating_count=data.get('rating_count', 0),
			last_updated=datetime.fromisoformat(data.get('last_updated', datetime.utcnow().isoformat())),
			compatibility_score=data.get('compatibility_score', 1.0),
			performance_score=data.get('performance_score', 1.0)
		)
	
	def _serialize_rating(self, rating: CapabilityRating) -> Dict[str, Any]:
		"""Serialize capability rating"""
		return {
			'id': rating.id,
			'user_id': rating.user_id,
			'capability_id': rating.capability_id,
			'rating': rating.rating,
			'review': rating.review,
			'helpful_votes': rating.helpful_votes,
			'created_at': rating.created_at.isoformat(),
			'verified_purchase': rating.verified_purchase
		}
	
	def _deserialize_rating(self, data: Dict[str, Any]) -> CapabilityRating:
		"""Deserialize capability rating"""
		return CapabilityRating(
			id=data['id'],
			user_id=data['user_id'],
			capability_id=data['capability_id'],
			rating=data['rating'],
			review=data.get('review', ''),
			helpful_votes=data.get('helpful_votes', 0),
			created_at=datetime.fromisoformat(data['created_at']),
			verified_purchase=data.get('verified_purchase', False)
		)
	
	def _update_indexes(self, capability: MarketplaceCapability):
		"""Update search indexes for capability"""
		# Category index
		if capability.category not in self.category_index:
			self.category_index[capability.category] = set()
		self.category_index[capability.category].add(capability.id)
		
		# Tag index
		for tag in capability.tags:
			if tag not in self.tag_index:
				self.tag_index[tag] = set()
			self.tag_index[tag].add(capability.id)
		
		# Author index
		if capability.author not in self.author_index:
			self.author_index[capability.author] = set()
		self.author_index[capability.author].add(capability.id)
	
	async def submit_capability(self, capability: MarketplaceCapability) -> Dict[str, Any]:
		"""Submit a new capability to the marketplace"""
		self.logger.info(f"Submitting capability: {capability.name}")
		
		# Validate capability
		validation_results = await self.validator.validate_capability(capability)
		
		if not validation_results['valid']:
			return {
				'success': False,
				'errors': validation_results['errors'],
				'validation_results': validation_results
			}
		
		# Generate ID if not provided
		if not capability.id:
			capability.id = uuid7str()
		
		# Set timestamps
		capability.created_at = datetime.utcnow()
		capability.updated_at = datetime.utcnow()
		
		# Store capability
		self.capabilities[capability.id] = capability
		self._update_indexes(capability)
		
		# Save to storage
		await self._save_capabilities()
		
		self.logger.info(f"Successfully submitted capability: {capability.name} ({capability.id})")
		
		return {
			'success': True,
			'capability_id': capability.id,
			'validation_results': validation_results
		}
	
	async def publish_capability(self, capability_id: str) -> bool:
		"""Publish a capability to make it publicly available"""
		if capability_id not in self.capabilities:
			return False
		
		capability = self.capabilities[capability_id]
		
		# Re-validate before publishing
		validation_results = await self.validator.validate_capability(capability)
		if not validation_results['valid']:
			return False
		
		capability.status = CapabilityStatus.PUBLISHED
		capability.published_at = datetime.utcnow()
		capability.updated_at = datetime.utcnow()
		
		await self._save_capabilities()
		
		self.logger.info(f"Published capability: {capability.name}")
		return True
	
	async def get_capability(self, capability_id: str) -> Optional[MarketplaceCapability]:
		"""Get a specific capability by ID"""
		return self.capabilities.get(capability_id)
	
	async def list_capabilities(
		self, 
		status: Optional[CapabilityStatus] = None,
		category: Optional[CapabilityCategory] = None,
		author: Optional[str] = None
	) -> List[MarketplaceCapability]:
		"""List capabilities with optional filters"""
		capabilities = list(self.capabilities.values())
		
		if status:
			capabilities = [c for c in capabilities if c.status == status]
		
		if category:
			capabilities = [c for c in capabilities if c.category == category]
		
		if author:
			capabilities = [c for c in capabilities if c.author == author]
		
		return capabilities
	
	async def search_capabilities(self, query: str, **kwargs) -> List[MarketplaceCapability]:
		"""Search capabilities using the discovery engine"""
		return await self.discovery.search_capabilities(query, **kwargs)
	
	async def get_recommendations(self, **kwargs) -> List[MarketplaceCapability]:
		"""Get capability recommendations"""
		return await self.discovery.get_recommendations(**kwargs)
	
	async def download_capability(self, capability_id: str, user_id: str = None) -> Optional[Dict[str, Any]]:
		"""Download a capability and update metrics"""
		capability = self.capabilities.get(capability_id)
		if not capability or capability.status != CapabilityStatus.PUBLISHED:
			return None
		
		# Update download metrics
		capability.metrics.download_count += 1
		capability.metrics.last_updated = datetime.utcnow()
		
		# Save updated metrics
		await self._save_capabilities()
		
		# Return capability package
		return {
			'capability': capability,
			'code': capability.capability_code,
			'documentation': capability.documentation,
			'example_usage': capability.example_usage,
			'test_cases': capability.test_cases,
			'dependencies': capability.dependencies
		}
	
	async def add_rating(self, rating: CapabilityRating) -> bool:
		"""Add a rating for a capability"""
		capability = self.capabilities.get(rating.capability_id)
		if not capability:
			return False
		
		# Add rating
		capability.ratings.append(rating)
		
		# Update average rating
		total_rating = sum(r.rating for r in capability.ratings)
		capability.metrics.average_rating = total_rating / len(capability.ratings)
		capability.metrics.rating_count = len(capability.ratings)
		capability.metrics.last_updated = datetime.utcnow()
		
		await self._save_capabilities()
		return True
	
	def get_marketplace_stats(self) -> Dict[str, Any]:
		"""Get marketplace statistics"""
		capabilities = list(self.capabilities.values())
		
		stats = {
			'total_capabilities': len(capabilities),
			'published_capabilities': len([c for c in capabilities if c.status == CapabilityStatus.PUBLISHED]),
			'verified_capabilities': len([c for c in capabilities if c.verified]),
			'total_downloads': sum(c.metrics.download_count for c in capabilities),
			'categories': {},
			'top_authors': {},
			'recent_activity': self._get_recent_activity()
		}
		
		# Category breakdown
		for category in CapabilityCategory:
			count = len([c for c in capabilities if c.category == category])
			if count > 0:
				stats['categories'][category.value] = count
		
		# Top authors
		author_counts = {}
		for cap in capabilities:
			if cap.author:
				author_counts[cap.author] = author_counts.get(cap.author, 0) + 1
		
		stats['top_authors'] = dict(sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10])
		
		return stats
	
	def _get_recent_activity(self) -> List[Dict[str, Any]]:
		"""Get recent marketplace activity"""
		capabilities = list(self.capabilities.values())
		
		# Sort by update time
		recent = sorted(capabilities, key=lambda x: x.updated_at, reverse=True)[:10]
		
		return [
			{
				'capability_name': cap.name,
				'author': cap.author,
				'action': 'published' if cap.published_at else 'updated',
				'timestamp': cap.updated_at.isoformat()
			}
			for cap in recent
		]