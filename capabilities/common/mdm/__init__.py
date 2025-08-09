#!/usr/bin/env python3
"""
APG Master Data Management (MDM) Capability
World-class Master Data Management with AI-powered intelligence and APG ecosystem integration

This package provides comprehensive MDM capabilities including:
- Entity lifecycle management with multi-tenant isolation
- AI-powered data quality assessment with 6-dimensional scoring
- Semantic duplicate detection with explainable confidence
- Golden record management with automated survivorship
- Real-time event streaming and APG ecosystem integration
- Enterprise-grade security and comprehensive audit trails

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
Email: nyimbi@gmail.com

Usage:
    >>> from apg.capabilities.common.mdm import MDMService
    >>> mdm_service = MDMService()
    >>> await mdm_service.initialize()
    >>> health = await mdm_service.health_check()
    >>> print(f"MDM Status: {health['status']}")
"""

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@gmail.com"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"

# Core service and manager imports
from .service import (
    MDMService,
    EntityService, 
    QualityService,
    MatchingService,
    AuditService,
    MDMOperationType,
    MDMOperationContext
)

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
        "entity_creation_max_ms": 50,
        "entity_retrieval_max_ms": 25,
        "quality_assessment_max_ms": 100,
        "duplicate_detection_max_ms": 500,
        "batch_operation_min_per_second": 100
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
    "EntityService",
    "QualityService", 
    "MatchingService",
    "AuditService",
    "MDMDatabaseManager",
    
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
    "MDM_DEFAULT_CONFIG"
]

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
    from .database import MDMDatabaseManager
    from .integrations import APGIntegrationManager
    
    # Merge provided config with defaults
    final_config = {**MDM_DEFAULT_CONFIG, **(config or {})}
    
    # Initialize components
    db_manager = MDMDatabaseManager(final_config)
    integration_manager = APGIntegrationManager(final_config)
    
    # Create and initialize service
    service = MDMService(
        db_manager=db_manager,
        integration_manager=integration_manager,
        config=final_config
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
