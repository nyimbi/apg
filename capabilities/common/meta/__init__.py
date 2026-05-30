#!/usr/bin/env python3
"""
APG Metadata Management Capability
Tenant-scoped metadata catalog, classification, lineage, and governance.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from typing import Dict, List, Any, Optional, Union
import asyncio

from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)

from .service import (
	APGMetadataService,
	MetaAssetRecord,
	MetaAuditEventRecord,
	MetaCertificationRecord,
	MetaClassificationRecord,
	MetaDiscoveryJobRecord,
	MetaGlossaryTermRecord,
	MetaLineageRecord,
	MetaQualityRecord,
	MetaService,
	ServiceHealth,
	ServiceStatus,
	create_metadata_service,
	get_metadata_service,
	shutdown_metadata_service
)

try:
	# Database and models
	from .database import MetaDatabaseManager, create_database_manager
	from .models import (
		MetaAsset,
		MetaColumn,
		MetaLineage,
		MetaClassification,
		MetaQualityAssessment,
		MetaDiscoveryJob,
		MetaDiscoverySchedule,
		AssetType,
		AssetStatus,
		DataType,
		ClassificationType,
		LineageType,
		QualityDimension
	)

	# AI and discovery components
	from .ai_classifier import AIClassificationEngine, create_ai_classifier
	from .discovery import MetadataDiscoveryService, DiscoverySchedule, create_discovery_service
	from .lineage_engine import DataLineageEngine, LineageEdge, create_lineage_engine
	from .search_engine import MetadataSearchEngine, SearchQuery, create_search_engine

	# Connectors
	from .connectors import (
		ConnectorConfig,
		ConnectorType,
		BaseConnector,
		DatabaseConnector,
		PostgreSQLConnector,
		MySQLConnector,
		MongoDBConnector
	)

	# Integration framework
	from .integrations import APGMetadataIntegrationManager, create_apg_integration_manager
	_RUNTIME_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	_RUNTIME_IMPORT_ERROR = exc
	MetaDatabaseManager = None
	MetaAsset = MetaColumn = MetaLineage = MetaClassification = MetaQualityAssessment = None
	MetaDiscoveryJob = MetaDiscoverySchedule = None
	AssetType = AssetStatus = DataType = ClassificationType = LineageType = QualityDimension = None
	AIClassificationEngine = MetadataDiscoveryService = DataLineageEngine = MetadataSearchEngine = None
	ConnectorConfig = ConnectorType = BaseConnector = DatabaseConnector = None
	PostgreSQLConnector = MySQLConnector = MongoDBConnector = None
	APGMetadataIntegrationManager = None
	DiscoverySchedule = LineageEdge = SearchQuery = None

	def _runtime_unavailable(*args, **kwargs):
		"""Require optional META runtime dependencies before use."""
		raise ModuleNotFoundError(
			"META runtime requires optional database/search dependencies such as asyncpg"
		) from _RUNTIME_IMPORT_ERROR

	create_metadata_service = _runtime_unavailable
	get_metadata_service = _runtime_unavailable
	shutdown_metadata_service = _runtime_unavailable
	create_database_manager = _runtime_unavailable
	create_ai_classifier = _runtime_unavailable
	create_discovery_service = _runtime_unavailable
	create_lineage_engine = _runtime_unavailable
	create_search_engine = _runtime_unavailable
	create_apg_integration_manager = _runtime_unavailable

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@gmail.com"
__company__ = "Datacraft"

# Capability metadata
CAPABILITY_INFO = {
	"name": "metadata_management",
	"display_name": "APG Metadata Management",
	"version": __version__,
	"description": "Tenant-scoped metadata catalog, classification, lineage, search, and governance",
	"category": "data_management",
	"tags": ["metadata", "ai", "discovery", "lineage", "search"],
	"capabilities": [
		"auto_discovery",
		"ai_classification", 
		"lineage_tracking",
		"natural_language_search",
		"impact_analysis",
		"data_quality_assessment",
		"real_time_monitoring"
	],
	"integrations": [
		"auth_rbac",
		"audit_compliance", 
		"ai_orchestration",
		"mdm"
	],
	"supported_sources": [
		"postgresql",
		"mysql",
		"mongodb",
		"file_systems",
		"cloud_storage"
	]
}

# Global service instance
_service_instance: Optional[APGMetadataService] = None


async def initialize_capability(config: Dict[str, Any] = None) -> APGMetadataService:
	"""
	Initialize the metadata management capability
	
	Args:
		config: Configuration dictionary for the capability
		
	Returns:
		Initialized metadata service instance
	"""
	global _service_instance
	
	if _service_instance is not None:
		return _service_instance
	if APGMetadataService is None:
		raise ModuleNotFoundError(
			"META runtime requires optional database/search dependencies such as asyncpg"
		) from _RUNTIME_IMPORT_ERROR
	
	# Default configuration
	default_config = {
		"database": {
			"postgresql_url": "postgresql://localhost/apg_metadata",
			"neo4j_url": "bolt://localhost:7687",
			"redis_url": "redis://localhost:6379"
		},
		"discovery": {
			"enable_auto_discovery": True,
			"discovery_interval_hours": 24,
			"max_concurrent_jobs": 5
		},
		"ai_classifier": {
			"enable_ai_classification": True,
			"ollama_base_url": "http://localhost:11434",
			"classification_model": "llama3.2:latest"
		},
		"lineage": {
			"enable_lineage_tracking": True,
			"real_time_processing": True
		},
		"search": {
			"enable_advanced_search": True,
			"enable_natural_language": True
		},
		"integrations": {
			"enable_apg_integration": True,
			"auth_service_url": "http://localhost:8001",
			"audit_service_url": "http://localhost:8002"
		}
	}
	
	# Merge with provided config
	if config:
		def deep_merge(default: dict, override: dict) -> dict:
			result = default.copy()
			for key, value in override.items():
				if key in result and isinstance(result[key], dict) and isinstance(value, dict):
					result[key] = deep_merge(result[key], value)
				else:
					result[key] = value
			return result
		
		final_config = deep_merge(default_config, config)
	else:
		final_config = default_config
	
	# Create and initialize service
	_service_instance = await create_metadata_service(final_config)
	
	return _service_instance


async def get_capability_instance() -> Optional[APGMetadataService]:
	"""Get the current capability instance"""
	return _service_instance


async def shutdown_capability():
	"""Shutdown the metadata management capability"""
	global _service_instance
	
	if _service_instance:
		await shutdown_metadata_service()
		_service_instance = None


def get_capability_info() -> Dict[str, Any]:
	"""Get capability information"""
	info = CAPABILITY_INFO.copy()
	info["contract"] = get_capability_contract()
	return info


def register_capability() -> Dict[str, Any]:
	"""Register metadata management with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "meta",
		"aliases": ["metadata_management", "catalog", "data_catalog"],
		"display_name": CAPABILITY_INFO["display_name"],
		"description": CAPABILITY_INFO["description"],
		"version": CAPABILITY_INFO["version"],
		"dependencies": ["mdm", "auth", "audl"],
		"optional_dependencies": ["aicr", "conn", "etlp", "mqeb", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"asset_catalog": "Register, search, and govern metadata assets",
			"auto_discovery": "Discover metadata from approved data sources",
			"ai_classification": "Classify sensitive assets with reviewable confidence",
			"lineage_tracking": "Capture upstream/downstream asset lineage",
			"impact_analysis": "Analyze downstream impact from asset changes",
			"capability_rules": "Evaluate deterministic metadata governance rules",
			"visual_theming": "Apply catalog-console theme tokens and components"
		},
		"endpoints": {
			"assets": "/meta/api/v1/assets",
			"discovery": "/meta/api/v1/discovery",
			"classification": "/meta/api/v1/classification",
			"lineage": "/meta/api/v1/lineage",
			"quality": "/meta/api/v1/quality",
		"search": "/meta/api/v1/search"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"meta:view",
			"meta:view_assets",
			"meta:run_discovery",
			"meta:view_lineage",
			"meta:classify",
			"meta:view_quality",
			"meta:search",
			"meta:admin"
		]
	}


async def quick_start(
	postgresql_url: str = "postgresql://localhost/apg_metadata",
	tenant_id: str = "default",
	enable_all_features: bool = True
) -> APGMetadataService:
	"""
	Quick start the metadata management capability with minimal configuration
	
	Args:
		postgresql_url: PostgreSQL connection URL
		tenant_id: Tenant ID for multi-tenant setup
		enable_all_features: Whether to enable all features
		
	Returns:
		Initialized metadata service
	"""
	config = {
		"database": {
			"postgresql_url": postgresql_url,
			"default_tenant_id": tenant_id
		},
		"enable_auto_discovery": enable_all_features,
		"enable_ai_classification": enable_all_features,
		"enable_lineage_tracking": enable_all_features,
		"enable_advanced_search": enable_all_features
	}
	
	return await initialize_capability(config)


# Convenience functions for common operations
async def discover_database(
	connection_config: Dict[str, Any],
	tenant_id: str = "default"
) -> str:
	"""
	Convenience function to discover a database
	
	Args:
		connection_config: Database connection configuration
		tenant_id: Tenant ID
		
	Returns:
		Discovery job ID
	"""
	service = await get_capability_instance()
	if not service:
		raise RuntimeError("Metadata service not initialized. Call initialize_capability() first.")
	
	from .discovery import DiscoverySchedule
	from .connectors import ConnectorConfig
	
	connector_config = ConnectorConfig(
		name=connection_config.get("name", "database_discovery"),
		connector_type=connection_config.get("type", "postgresql"),
		connection_params=connection_config,
		tenant_id=tenant_id
	)
	
	schedule = DiscoverySchedule(
		name="Quick Database Discovery",
		connector_config=connector_config,
		tenant_id=tenant_id,
		is_one_time=True
	)
	
	schedule_id = await service.create_discovery_schedule(schedule)
	return await service.run_discovery(schedule_id)


async def search_assets(
	query_text: str,
	tenant_id: str = "default",
	filters: Dict[str, Any] = None,
	limit: int = 50
) -> Dict[str, Any]:
	"""
	Convenience function to search metadata assets
	
	Args:
		query_text: Search query text
		tenant_id: Tenant ID
		filters: Additional filters
		limit: Maximum results
		
	Returns:
		Search results
	"""
	service = await get_capability_instance()
	if not service:
		raise RuntimeError("Metadata service not initialized. Call initialize_capability() first.")
	
	from .search_engine import SearchQuery
	
	search_query = SearchQuery(
		query_text=query_text,
		tenant_id=tenant_id,
		filters=filters or {},
		limit=limit,
		enable_natural_language=True
	)
	
	return await service.search_metadata(search_query)


async def get_asset_lineage(
	asset_id: str,
	tenant_id: str = "default",
	direction: str = "both",
	max_depth: int = 5
) -> List[Dict[str, Any]]:
	"""
	Convenience function to get asset lineage
	
	Args:
		asset_id: Asset ID to get lineage for
		tenant_id: Tenant ID
		direction: Lineage direction (upstream, downstream, both)
		max_depth: Maximum depth to traverse
		
	Returns:
		Lineage paths
	"""
	service = await get_capability_instance()
	if not service:
		raise RuntimeError("Metadata service not initialized. Call initialize_capability() first.")
	
	return await service.get_lineage_path(asset_id, tenant_id, direction, max_depth)


# Export main classes and functions
__all__ = [
	# Core service
	"APGMetadataService",
	"ServiceStatus", 
	"ServiceHealth",
	"create_metadata_service",
	"get_metadata_service",
	"shutdown_metadata_service",
	
	# Database and models
	"MetaDatabaseManager",
	"create_database_manager",
	"MetaAsset",
	"MetaColumn",
	"MetaLineage",
	"MetaClassification",
	"MetaQualityAssessment",
	"MetaDiscoveryJob",
	"MetaDiscoverySchedule",
	
	# Enums
	"AssetType",
	"AssetStatus", 
	"DataType",
	"ClassificationType",
	"LineageType",
	"QualityDimension",
	
	# Components
	"AIClassificationEngine",
	"MetadataDiscoveryService", 
	"DataLineageEngine",
	"MetadataSearchEngine",
	"APGMetadataIntegrationManager",
	
	# Connectors
	"ConnectorConfig",
	"ConnectorType",
	"BaseConnector",
	"DatabaseConnector",
	"PostgreSQLConnector",
	"MySQLConnector",
	"MongoDBConnector",
	
	# Data structures
	"DiscoverySchedule",
	"LineageEdge",
	"SearchQuery",
	
	# Capability management
	"initialize_capability",
	"get_capability_instance",
	"shutdown_capability",
	"get_capability_info",
	"register_capability",
	"get_capability_contract",
	"evaluate_capability_rules",
	"quick_start",
	
	# Convenience functions
	"discover_database",
	"search_assets",
	"get_asset_lineage",
	
	# Metadata
	"CAPABILITY_INFO",
	"__version__",
	"__author__"
]
