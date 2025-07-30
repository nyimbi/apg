"""
YouTube Crawler Parsers Module
===============================

Comprehensive parsing components for YouTube content extraction.
Provides specialized parsers for different types of YouTube data.

This module includes:
- Base parser framework
- Video metadata parser
- Channel information parser
- Comment parser
- Transcript parser
- Thumbnail parser
- Parser registry system

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

from .base_parser import (
    BaseParser,
    ParseResult,
    ParseStatus,
    ParserConfig,
    ParserRegistry,
    parser_registry
)

from .video_parser import (
    VideoParser,
    VideoMetadataParser,
    VideoStatisticsParser,
    VideoContentParser
)

from .channel_parser import (
    ChannelParser,
    ChannelMetadataParser,
    ChannelStatisticsParser,
    ChannelContentParser
)

from .comment_parser import (
    CommentParser,
    CommentThreadParser,
    CommentAnalyzer
)

from .transcript_parser import (
    TranscriptParser,
    SubtitleParser,
    CaptionParser,
    TranscriptAnalyzer
)

from .thumbnail_parser import (
    ThumbnailParser,
    ThumbnailProcessor,
    ThumbnailAnalyzer
)

from .metadata_parser import (
    MetadataParser,
    ContentMetadataParser,
    TechnicalMetadataParser
)

# Import data models from the main API module to avoid duplication
from ..api.data_models import (
    VideoData,
    ChannelData,
    CommentData,
    TranscriptData,
    ThumbnailData,
    PlaylistData,
    CrawlResult,
    ExtractResult,
    ValidationResult
)

# Import base classes and enums needed for parsers
from .base_parser import ContentType, MediaType

__all__ = [
    # Base parser framework
    'BaseParser',
    'ParseResult',
    'ParseStatus',
    'ParserConfig',
    'ParserRegistry',
    'parser_registry',

    # Video parsers
    'VideoParser',
    'VideoMetadataParser',
    'VideoStatisticsParser',
    'VideoContentParser',

    # Channel parsers
    'ChannelParser',
    'ChannelMetadataParser',
    'ChannelStatisticsParser',
    'ChannelContentParser',

    # Comment parsers
    'CommentParser',
    'CommentThreadParser',
    'CommentAnalyzer',

    # Transcript parsers
    'TranscriptParser',
    'SubtitleParser',
    'CaptionParser',
    'TranscriptAnalyzer',

    # Thumbnail parsers
    'ThumbnailParser',
    'ThumbnailProcessor',
    'ThumbnailAnalyzer',

    # Metadata parsers
    'MetadataParser',
    'ContentMetadataParser',
    'TechnicalMetadataParser',

    # Data models
    'VideoData',
    'ChannelData',
    'CommentData',
    'TranscriptData',
    'ThumbnailData',
    'PlaylistData',
    'CrawlResult',
    'ExtractResult',
    'ValidationResult',
    'ContentType',
    'MediaType',
]

# Parser type constants
PARSER_TYPES = {
    'video': 'video_parser',
    'channel': 'channel_parser',
    'comment': 'comment_parser',
    'transcript': 'transcript_parser',
    'thumbnail': 'thumbnail_parser',
    'metadata': 'metadata_parser'
}

# Content type mappings
CONTENT_TYPE_MAPPINGS = {
    'video': ContentType.VIDEO,
    'channel': ContentType.CHANNEL,
    'playlist': ContentType.PLAYLIST,
    'comment': ContentType.COMMENT,
    'transcript': ContentType.TRANSCRIPT,
    'thumbnail': ContentType.THUMBNAIL
}

def get_parser_for_content_type(content_type: str) -> BaseParser:
    """Get appropriate parser for content type."""
    if content_type in PARSER_TYPES:
        parser_name = PARSER_TYPES[content_type]
        return parser_registry.get_parser(parser_name)
    else:
        raise ValueError(f"Unknown content type: {content_type}")

def register_custom_parser(name: str, parser_class: BaseParser):
    """Register a custom parser."""
    parser_registry.register_parser(name, parser_class)

def list_available_parsers():
    """List all available parsers."""
    return parser_registry.list_parsers()

def get_parser_info(parser_name: str):
    """Get information about a specific parser."""
    return parser_registry.get_parser_info(parser_name)

# Initialize default parsers
def initialize_default_parsers():
    """Initialize all default parsers in the registry."""
    try:
        # Register video parsers
        parser_registry.register_parser('video_parser', VideoParser)
        parser_registry.register_parser('video_metadata_parser', VideoMetadataParser)
        parser_registry.register_parser('video_statistics_parser', VideoStatisticsParser)
        parser_registry.register_parser('video_content_parser', VideoContentParser)

        # Register channel parsers
        parser_registry.register_parser('channel_parser', ChannelParser)
        parser_registry.register_parser('channel_metadata_parser', ChannelMetadataParser)
        parser_registry.register_parser('channel_statistics_parser', ChannelStatisticsParser)
        parser_registry.register_parser('channel_content_parser', ChannelContentParser)

        # Register comment parsers
        parser_registry.register_parser('comment_parser', CommentParser)
        parser_registry.register_parser('comment_thread_parser', CommentThreadParser)
        parser_registry.register_parser('comment_analyzer', CommentAnalyzer)

        # Register transcript parsers
        parser_registry.register_parser('transcript_parser', TranscriptParser)
        parser_registry.register_parser('subtitle_parser', SubtitleParser)
        parser_registry.register_parser('caption_parser', CaptionParser)
        parser_registry.register_parser('transcript_analyzer', TranscriptAnalyzer)

        # Register thumbnail parsers
        parser_registry.register_parser('thumbnail_parser', ThumbnailParser)
        parser_registry.register_parser('thumbnail_processor', ThumbnailProcessor)
        parser_registry.register_parser('thumbnail_analyzer', ThumbnailAnalyzer)

        # Register metadata parsers
        parser_registry.register_parser('metadata_parser', MetadataParser)
        parser_registry.register_parser('content_metadata_parser', ContentMetadataParser)
        parser_registry.register_parser('technical_metadata_parser', TechnicalMetadataParser)

        return True

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to initialize default parsers: {e}")
        return False

# Initialize parsers on module import
initialize_default_parsers()

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@datacraft.co.ke"
__company__ = "Datacraft"

# Parser capabilities
PARSER_CAPABILITIES = {
    'video_parser': {
        'input_types': ['api_response', 'html', 'json'],
        'output_types': ['video_data'],
        'features': ['metadata', 'statistics', 'content_analysis'],
        'async_support': True,
        'batch_processing': True
    },
    'channel_parser': {
        'input_types': ['api_response', 'html', 'json'],
        'output_types': ['channel_data'],
        'features': ['metadata', 'statistics', 'content_analysis'],
        'async_support': True,
        'batch_processing': True
    },
    'comment_parser': {
        'input_types': ['api_response', 'html', 'json'],
        'output_types': ['comment_data'],
        'features': ['sentiment_analysis', 'threading', 'moderation'],
        'async_support': True,
        'batch_processing': True
    },
    'transcript_parser': {
        'input_types': ['vtt', 'srt', 'json', 'xml'],
        'output_types': ['transcript_data'],
        'features': ['language_detection', 'timing', 'formatting'],
        'async_support': True,
        'batch_processing': False
    },
    'thumbnail_parser': {
        'input_types': ['url', 'binary', 'base64'],
        'output_types': ['thumbnail_data'],
        'features': ['image_analysis', 'optimization', 'format_conversion'],
        'async_support': True,
        'batch_processing': True
    },
    'metadata_parser': {
        'input_types': ['api_response', 'html', 'json'],
        'output_types': ['metadata'],
        'features': ['extraction', 'validation', 'enrichment'],
        'async_support': True,
        'batch_processing': True
    }
}

def get_parser_capabilities(parser_name: str = None):
    """Get parser capabilities information."""
    if parser_name:
        return PARSER_CAPABILITIES.get(parser_name, {})
    return PARSER_CAPABILITIES

def validate_parser_compatibility(parser_name: str, input_type: str, output_type: str):
    """Validate if parser supports given input/output types."""
    capabilities = get_parser_capabilities(parser_name)

    if not capabilities:
        return False

    input_supported = input_type in capabilities.get('input_types', [])
    output_supported = output_type in capabilities.get('output_types', [])

    return input_supported and output_supported

# Parser performance metrics
PARSER_PERFORMANCE_METRICS = {
    'parsing_time': 'average_parsing_time_ms',
    'success_rate': 'parsing_success_rate_percent',
    'error_rate': 'parsing_error_rate_percent',
    'throughput': 'items_parsed_per_second',
    'memory_usage': 'average_memory_usage_mb',
    'cpu_usage': 'average_cpu_usage_percent'
}

def get_parser_performance_metrics():
    """Get parser performance metrics definitions."""
    return PARSER_PERFORMANCE_METRICS

# Error handling utilities
def handle_parser_error(parser_name: str, error: Exception, context: dict = None):
    """Handle parser errors with context."""
    import logging
    logger = logging.getLogger(__name__)

    error_context = {
        'parser_name': parser_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {}
    }

    logger.error(f"Parser error in {parser_name}: {error}", extra=error_context)

    # Could implement error reporting, metrics collection, etc.
    return error_context

# Parser configuration utilities
def create_parser_config(parser_type: str, **kwargs):
    """Create parser configuration for specific parser type."""
    config = ParserConfig()

    # Apply parser-specific defaults
    if parser_type == 'video_parser':
        config.extract_metadata = kwargs.get('extract_metadata', True)
        config.extract_statistics = kwargs.get('extract_statistics', True)
        config.extract_content = kwargs.get('extract_content', True)
    elif parser_type == 'channel_parser':
        config.extract_metadata = kwargs.get('extract_metadata', True)
        config.extract_statistics = kwargs.get('extract_statistics', True)
        config.extract_videos = kwargs.get('extract_videos', False)
    elif parser_type == 'comment_parser':
        config.analyze_sentiment = kwargs.get('analyze_sentiment', False)
        config.extract_threads = kwargs.get('extract_threads', True)
        config.max_comments = kwargs.get('max_comments', 100)
    elif parser_type == 'transcript_parser':
        config.detect_language = kwargs.get('detect_language', True)
        config.preserve_timing = kwargs.get('preserve_timing', True)
        config.clean_text = kwargs.get('clean_text', True)

    return config

# Health check for parsers
def health_check():
    """Perform health check on all parsers."""
    results = {
        'status': 'healthy',
        'parsers': {},
        'total_parsers': 0,
        'healthy_parsers': 0,
        'unhealthy_parsers': 0
    }

    try:
        available_parsers = list_available_parsers()
        results['total_parsers'] = len(available_parsers)

        for parser_name in available_parsers:
            try:
                parser_info = get_parser_info(parser_name)
                results['parsers'][parser_name] = {
                    'status': 'healthy',
                    'info': parser_info
                }
                results['healthy_parsers'] += 1
            except Exception as e:
                results['parsers'][parser_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                results['unhealthy_parsers'] += 1

        if results['unhealthy_parsers'] > 0:
            results['status'] = 'degraded'

        if results['healthy_parsers'] == 0:
            results['status'] = 'unhealthy'

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)

    return results
