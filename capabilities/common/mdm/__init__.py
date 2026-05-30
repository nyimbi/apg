#!/usr/bin/env python3
"""
APG Master Data Management (MDM) Capability
Tenant-scoped master-data governance with generated-app composition hooks.

This package provides comprehensive MDM capabilities including:
- Entity lifecycle management with multi-tenant isolation
- Data quality assessment with 6-dimensional scoring
- Duplicate detection evidence with steward review
- Golden record management with survivorship policies
- Bytewax-ready event streaming and APG ecosystem integration
- Enterprise-grade security and comprehensive audit trails

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
Email: nyimbi@gmail.com

Usage:
    >>> from apg.capabilities.common.mdm import MdmService
    >>> mdm_service = MdmService()
    >>> status = mdm_service.dashboard_summary()
    >>> print(f"MDM entities: {status['entity_count']}")
"""

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@gmail.com"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"

from .capability_contract import (
    get_capability_contract,
    evaluate_capability_rules
)

from .service import (
    MDMService,
    MdmService,
    EntityService,
    QualityService,
    MatchingService,
    AuditService,
    MDMOperationType,
    MDMOperationContext,
    MdmEntityRecord,
    MdmQualityRecord,
    MdmDuplicateCandidateRecord,
    MdmGoldenRecord,
    MdmMergeRequestRecord,
    MdmCrossReferenceRecord,
    MdmPublishRecord,
    MdmAuditEventRecord
)

try:
    from .database import (
        MDMDatabaseManager,
        DatabaseHealthStatus
    )

    # Data model imports
    from .models import (
        # Core entity models
        MdEntity,
        MdEntityVersion,
        MdGoldenRecord,
        MdDataQualityAssessment,
        MdCrossReference,

        # Pydantic models for API
        MdEntityCreate,
        MdEntityUpdate,

        # Enums
        EntityType,
        EntityStatus,
        DataQualityStatus,
        MatchConfidence
    )

    # View models for serialization
    from .views import (
        # Response containers
        EntityResponse,
        EntityListResponse,
        QualityAssessmentResponse,
        DuplicateDetectionResponse,

        # View models
        EntityDetailView,
        EntitySummaryView,
        QualityAssessmentView,
        DuplicateCandidateView
    )

    # AI engines
    from .ai_engines import (
        EntityMatchingEngine,
        DataQualityEngine,
        AnomalyDetectionEngine,
        AIEngineConfig
    )

    # APG integration
    from .integrations import (
        APGIntegrationManager,
        EventPublisher,
        CacheManager,
        APGAuditLogger,
        ConfigurationManager,
        MDMEvent
    )

    # API components
    from .api import (
        create_mdm_app,
        MDMRouter,
        get_mdm_service
    )

    # Flask blueprint
    from .blueprint import (
        mdm_bp,
        register_mdm_views,
        MDMDashboardView,
        EntityManagementView,
        QualityManagementView
    )
    _RUNTIME_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _RUNTIME_IMPORT_ERROR = exc
    MDMDatabaseManager = DatabaseHealthStatus = None
    MdEntity = MdEntityVersion = MdGoldenRecord = MdDataQualityAssessment = MdCrossReference = None
    MdEntityCreate = MdEntityUpdate = None
    EntityType = EntityStatus = DataQualityStatus = MatchConfidence = None
    EntityResponse = EntityListResponse = QualityAssessmentResponse = DuplicateDetectionResponse = None
    EntityDetailView = EntitySummaryView = QualityAssessmentView = DuplicateCandidateView = None
    EntityMatchingEngine = DataQualityEngine = AnomalyDetectionEngine = AIEngineConfig = None
    APGIntegrationManager = EventPublisher = CacheManager = APGAuditLogger = ConfigurationManager = MDMEvent = None
    MDMRouter = None
    mdm_bp = None
    MDMDashboardView = EntityManagementView = QualityManagementView = None

    def _runtime_unavailable(*args, **kwargs):
        """Require optional MDM runtime dependencies before use."""
        raise ModuleNotFoundError(
            "MDM runtime requires optional database/UI dependencies such as asyncpg"
        ) from _RUNTIME_IMPORT_ERROR

    create_mdm_app = _runtime_unavailable
    get_mdm_service = _runtime_unavailable
    register_mdm_views = _runtime_unavailable

# Configuration constants
MDM_DEFAULT_CONFIG = {
    "enable_ai": True,
    "enable_caching": True,
    "enable_events": True,
    "quality_thresholds": {
        "excellent": 95.0,
        "good": 80.0,
        "fair": 60.0,
        "poor": 40.0
    },
    "matching_thresholds": {
        "exact_match": 100.0,
        "high_confidence": 90.0,
        "medium_confidence": 70.0,
        "minimum_match": 50.0
    },
    "performance_targets": {
        "measure_entity_creation_ms": True,
        "measure_entity_retrieval_ms": True,
        "measure_quality_assessment_ms": True,
        "measure_duplicate_detection_ms": True,
        "measure_batch_throughput": True
    }
}

# Public API
__all__ = [
    # Version and metadata
    "__version__",
    "__author__", 
    "__email__",
    "__company__",
    "__website__",
    
    # Core services
    "MDMService",
    "MdmService",
    "EntityService",
    "QualityService", 
    "MatchingService",
    "AuditService",
    "MDMDatabaseManager",
    "MdmEntityRecord",
    "MdmQualityRecord",
    "MdmDuplicateCandidateRecord",
    "MdmGoldenRecord",
    "MdmMergeRequestRecord",
    "MdmCrossReferenceRecord",
    "MdmPublishRecord",
    "MdmAuditEventRecord",
    
    # Models
    "MdEntity",
    "MdEntityVersion",
    "MdGoldenRecord", 
    "MdDataQualityAssessment",
    "MdCrossReference",
    "MdEntityCreate",
    "MdEntityUpdate",
    
    # Enums
    "EntityType",
    "EntityStatus",
    "DataQualityStatus",
    "MatchConfidence",
    
    # Views
    "EntityResponse",
    "EntityListResponse",
    "QualityAssessmentResponse",
    "DuplicateDetectionResponse",
    "EntityDetailView",
    "EntitySummaryView",
    "QualityAssessmentView",
    "DuplicateCandidateView",
    
    # AI Engines
    "EntityMatchingEngine",
    "DataQualityEngine", 
    "AnomalyDetectionEngine",
    "AIEngineConfig",
    
    # APG Integration
    "APGIntegrationManager",
    "EventPublisher",
    "CacheManager",
    "APGAuditLogger",
    "ConfigurationManager",
    "MDMEvent",
    
    # API
    "create_mdm_app",
    "MDMRouter",
    "get_mdm_service",
    
    # Flask Blueprint
    "mdm_bp",
    "register_mdm_views",
    "MDMDashboardView",
    "EntityManagementView", 
    "QualityManagementView",
    
    # Operation types
    "MDMOperationType",
    "MDMOperationContext",
    "DatabaseHealthStatus",
    
    # Configuration
    "MDM_DEFAULT_CONFIG",
    "register_capability",
    "get_capability_info",
    "get_capability_contract",
    "evaluate_capability_rules"
]


def register_capability() -> dict:
    """Register master data management with the APG composition engine."""
    contract = get_capability_contract()
    return {
        "name": "mdm",
        "aliases": ["master_data_management", "golden_records", "data_stewardship"],
        "display_name": "Master Data Management",
        "description": "Entity lifecycle, golden records, quality scoring, and stewardship governance",
        "version": __version__,
        "dependencies": ["auth", "audl", "conf", "mten"],
        "optional_dependencies": ["meta", "mqeb", "moni", "cach", "aicr"],
        "configuration": contract["configuration"],
        "configuration_schema": contract["configuration_schema"],
        "rule_engine": contract["rule_engine"],
        "capabilities": {
            "entity_lifecycle": "Create, version, publish, and retire tenant-aware master entities",
            "golden_records": "Manage survivorship and canonical entity views",
            "data_quality": "Score master data across quality dimensions",
            "duplicate_review": "Detect and review duplicate entity candidates",
            "stewardship": "Route governance work to data stewards and owners",
            "capability_rules": "Evaluate deterministic master-data governance rules",
            "visual_theming": "Apply golden-record console theme tokens and components"
        },
        "endpoints": {
            "entities": "/mdm/api/v1/entities",
            "golden_records": "/mdm/api/v1/golden-records",
            "quality": "/mdm/api/v1/quality",
            "duplicates": "/mdm/api/v1/duplicates",
            "stewardship": "/mdm/api/v1/stewardship",
            "analytics": "/mdm/api/v1/analytics"
        },
        "ui_components": {
            route["name"]: route["path"]
            for route in contract["ui"]["routes"]
        },
        "ui_manifest": contract["ui"],
        "theme": contract["theme"],
        "permissions": [
            "mdm:view",
            "mdm:manage_entities",
            "mdm:manage_golden_records",
            "mdm:view_quality",
            "mdm:review_duplicates",
            "mdm:steward",
            "mdm:view_analytics",
            "mdm:admin"
        ]
    }


def get_capability_info() -> dict:
    """Get MDM capability information for composition and marketplace discovery."""
    return {
        "metadata": {
            "name": "mdm",
            "display_name": "Master Data Management",
            "version": __version__,
            "author": __author__,
            "company": __company__,
            "dependencies": ["auth", "audl", "conf", "mten"]
        },
        "contract": get_capability_contract(),
        "features": [
            "Entity lifecycle management",
            "AI-powered quality assessment",
            "Semantic duplicate detection",
            "Golden record management",
            "Multi-tenant security",
            "Comprehensive audit trails"
        ]
    }

# Package-level convenience functions
async def create_default_mdm_service(config: dict = None) -> MDMService:
    """
    Create and initialize a default MDM service with standard configuration.
    
    Args:
        config: Optional configuration dictionary to override defaults
        
    Returns:
        Initialized MDM service ready for use
        
    Example:
        >>> mdm = await create_default_mdm_service()
        >>> health = await mdm.health_check()
        >>> print(f"Status: {health['status']}")
    """
    # Merge provided config with defaults
    final_config = {**MDM_DEFAULT_CONFIG, **(config or {})}

    service = MDMService(
        database_url=final_config.get("database_url"),
        config=final_config,
    )
    await service.initialize()
    return service


def get_version_info() -> dict:
    """
    Get comprehensive version and build information.
    
    Returns:
        Dictionary with version, author, and build information
    """
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__, 
        "company": __company__,
        "website": __website__,
        "build_date": "2025-01-09",
        "python_min_version": "3.11",
        "capabilities": [
            "Entity Lifecycle Management",
            "AI-Powered Quality Assessment", 
            "Semantic Duplicate Detection",
            "Golden Record Management",
            "APG Ecosystem Integration",
            "Multi-tenant Security",
            "Real-time Event Streaming",
            "Comprehensive Audit Trails"
        ]
    }
