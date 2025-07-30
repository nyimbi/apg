"""
Enhanced Google News Implementation
===================================

This module provides the main implementation for the enhanced Google News crawler,
integrating all components and providing backward compatibility with GNews API.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

# Re-export all classes and functions from the API module
from .api.google_news_client import (
    # Main client classes
    EnhancedGoogleNewsClient,
    GNewsCompatibilityWrapper,

    # Configuration and utility classes
    NewsSource,
    EnhancedNewsSource,
    SiteFilteringEngine,
    NewsSourceConfig,

    # Data classes and enums
    SourceType,
    ContentFormat,
    CrawlPriority,
    GeographicalFocus,
    SourceCredibilityMetrics,

    # Factory functions
    create_enhanced_gnews_client,
    create_basic_gnews_client,
    create_sample_configuration,
    load_source_configuration,
)

# Re-export optimization components
from .optimization.performance_optimizer import (
    PerformanceMetrics,
    AdvancedCacheManager,
    DistributedTaskManager,
    PerformanceMonitor,
    OptimizedNewsIntelligenceOrchestrator,
    performance_timing,
)

# All exports for easy import
__all__ = [
    # Main classes
    'EnhancedGoogleNewsClient',
    'GNewsCompatibilityWrapper',
    'NewsSource',
    'EnhancedNewsSource',
    'SiteFilteringEngine',
    'NewsSourceConfig',

    # Enums and data classes
    'SourceType',
    'ContentFormat',
    'CrawlPriority',
    'GeographicalFocus',
    'SourceCredibilityMetrics',

    # Factory functions
    'create_enhanced_gnews_client',
    'create_basic_gnews_client',
    'create_sample_configuration',
    'load_source_configuration',

    # Optimization classes
    'PerformanceMetrics',
    'AdvancedCacheManager',
    'DistributedTaskManager',
    'PerformanceMonitor',
    'OptimizedNewsIntelligenceOrchestrator',
    'performance_timing',
]
