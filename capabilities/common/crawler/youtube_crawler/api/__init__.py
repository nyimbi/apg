"""
YouTube Crawler API Module
===========================

Main API components for YouTube content crawling and data extraction.

This module provides:
- Enhanced YouTube API client
- Web scraping capabilities
- Data models and utilities
- Integration with YouTube Data API v3

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

from .youtube_client import (
    # Main client classes
    YouTubeClient,
    YouTubeAPIWrapper,
    YouTubeScrapingClient,

    # Data models
    YouTubeVideo,
    YouTubeChannel,
    YouTubePlaylist,
    VideoMetrics,
    ChannelCredibilityMetrics,

    # Configuration classes
    YouTubeSourceConfig,
    ChannelFilteringEngine,

    # Enums
    ContentType,
    VideoQuality,
    CrawlPriority,
    GeographicalFocus,

    # Factory functions
    create_youtube_client,
    create_basic_youtube_client,
    create_sample_configuration,
    load_source_configuration,
)

from .data_models import (
    # Core data structures
    VideoData,
    ChannelData,
    PlaylistData,
    CommentData,
    TranscriptData,
    ThumbnailData,

    # Metrics and analytics
    EngagementMetrics,
    PerformanceMetrics,
    AudienceMetrics,

    # Status and result classes
    CrawlResult,
    ExtractResult,
    ValidationResult,
)

from .exceptions import (
    YouTubeCrawlerError,
    APIQuotaExceededError,
    VideoNotFoundError,
    ChannelNotFoundError,
    AccessRestrictedError,
    RateLimitExceededError,
    ConfigurationError,
    ParsingError,
)

__all__ = [
    # Main client classes
    'YouTubeClient',
    'YouTubeAPIWrapper',
    'YouTubeScrapingClient',

    # Data models
    'YouTubeVideo',
    'YouTubeChannel',
    'YouTubePlaylist',
    'VideoMetrics',
    'ChannelCredibilityMetrics',
    'VideoData',
    'ChannelData',
    'PlaylistData',
    'CommentData',
    'TranscriptData',
    'ThumbnailData',

    # Metrics
    'EngagementMetrics',
    'PerformanceMetrics',
    'AudienceMetrics',

    # Results
    'CrawlResult',
    'ExtractResult',
    'ValidationResult',

    # Configuration
    'YouTubeSourceConfig',
    'ChannelFilteringEngine',

    # Enums
    'ContentType',
    'VideoQuality',
    'CrawlPriority',
    'GeographicalFocus',
    'ChannelCredibilityMetrics',

    # Factory functions
    'create_youtube_client',
    'create_basic_youtube_client',
    'create_sample_configuration',
    'load_source_configuration',

    # Exceptions
    'YouTubeCrawlerError',
    'APIQuotaExceededError',
    'VideoNotFoundError',
    'ChannelNotFoundError',
    'AccessRestrictedError',
    'RateLimitExceededError',
    'ConfigurationError',
    'ParsingError',
]
