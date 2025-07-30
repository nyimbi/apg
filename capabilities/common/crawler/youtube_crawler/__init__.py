"""
YouTube Crawler Package
========================

Enterprise-grade YouTube content crawler with filtering, metadata extraction,
and integration with existing systems.

This package provides:
- YouTube API client with multi-source aggregation
- Video/channel filtering and credibility assessment
- Stealth crawling capabilities for public content
- PostgreSQL integration for data storage
- Comment and metadata extraction
- Video transcript analysis
- Channel analytics and monitoring

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@datacraft.co.ke"
__company__ = "Datacraft"

# Import main classes for easy access
try:
    from .api.youtube_client import (
        # Main client classes
        YouTubeClient,
        EnhancedYouTubeClient,  # backward compatibility
        YouTubeAPIWrapper,
        YouTubeScrapingClient,

        # Configuration and utility classes
        YouTubeChannel,
        YouTubeVideo,
        YouTubePlaylist,
        ChannelFilteringEngine,
        YouTubeSourceConfig,

        # Data classes and enums
        ContentType,
        VideoQuality,
        CrawlPriority,
        GeographicalFocus,
        ChannelCredibilityMetrics,
        VideoMetrics,

        # Factory functions
        create_youtube_client,
        create_basic_youtube_client,
        create_sample_configuration,
        load_source_configuration,
    )

    # Import unified configuration system
    from .unified_config import (
        YouTubeConfigurationManager,
        create_youtube_crawler_config,
        get_youtube_crawler_config_manager,
        UNIFIED_CONFIG_AVAILABLE
    )
    
    # Import legacy configuration classes for compatibility
    from .config import CrawlerConfig, ConfigurationManager, get_config, load_config

    # Import parser classes
    from .parsers import (
        BaseParser, ParseResult, VideoData, ParseStatus, MediaType,
        ParserRegistry, parser_registry, TranscriptParser, MetadataParser,
        CommentParser
    )

    # Import optimization classes
    from .optimization import (
        CacheManager, RateLimiter, RequestOptimizer, BatchProcessor,
        PerformanceMonitor
    )
    
    # Alias for backward compatibility and naming consistency
    create_enhanced_youtube_client = create_youtube_client

    # Define what gets imported with "from youtube_crawler import *"
    __all__ = [
        # Main classes
        'YouTubeClient',
        'EnhancedYouTubeClient',  # backward compatibility
        'YouTubeAPIWrapper',
        'YouTubeScrapingClient',
        'YouTubeChannel',
        'YouTubeVideo',
        'YouTubePlaylist',
        'ChannelFilteringEngine',
        'YouTubeSourceConfig',

        # Enums and data classes
        'ContentType',
        'VideoQuality',
        'CrawlPriority',
        'GeographicalFocus',
        'ChannelCredibilityMetrics',
        'VideoMetrics',

        # Factory functions
        'create_youtube_client',
        'create_enhanced_youtube_client',  # Alias for compatibility
        'create_basic_youtube_client',
        'create_sample_configuration',
        'load_source_configuration',

        # Unified Configuration
        'YouTubeConfigurationManager',
        'create_youtube_crawler_config',
        'get_youtube_crawler_config_manager',
        'UNIFIED_CONFIG_AVAILABLE',
        
        # Legacy Configuration
        'CrawlerConfig',
        'ConfigurationManager',
        'get_config',
        'load_config',

        # Parsers
        'BaseParser',
        'ParseResult',
        'VideoData',
        'ParseStatus',
        'MediaType',
        'ParserRegistry',
        'parser_registry',
        'TranscriptParser',
        'MetadataParser',
        'CommentParser',

        # Optimization
        'CacheManager',
        'RateLimiter',
        'RequestOptimizer',
        'BatchProcessor',
        'PerformanceMonitor',
    ]

except ImportError as e:
    # If imports fail, provide a helpful error message
    import warnings
    warnings.warn(
        f"Could not import core components: {e}\n"
        "Please ensure all dependencies are installed:\n"
        "pip install aiohttp asyncpg youtube-dl yt-dlp google-api-python-client",
        ImportWarning
    )

    # Define empty __all__ to prevent import errors
    __all__ = []
    
    # Try to create minimal fallback for the alias
    try:
        from .api.youtube_client import create_youtube_client
        create_enhanced_youtube_client = create_youtube_client
        __all__.append('create_enhanced_youtube_client')
    except ImportError:
        # Create a placeholder function if nothing works
        def create_enhanced_youtube_client(*args, **kwargs):
            raise ImportError("YouTube crawler components not available")
        __all__.append('create_enhanced_youtube_client')
    
    # Create placeholder classes to prevent import errors
    from dataclasses import dataclass, field
    from typing import Optional, Dict, Any, List
    from datetime import datetime
    
    @dataclass
    class VideoData:
        """Placeholder VideoData class when main components are not available."""
        video_id: str = ""
        title: str = ""
        description: str = ""
        duration: Optional[int] = None
        view_count: Optional[int] = None
        like_count: Optional[int] = None
        channel_id: str = ""
        channel_name: str = ""
        upload_date: Optional[datetime] = None
        thumbnail_url: str = ""
        video_url: str = ""
        transcript: Optional[str] = None
        comments: List[str] = None
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.comments is None:
                self.comments = []
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass
    class ChannelData:
        """Placeholder ChannelData class when main components are not available."""
        channel_id: str = ""
        channel_name: str = ""
        description: str = ""
        subscriber_count: Optional[int] = None
        video_count: Optional[int] = None
        view_count: Optional[int] = None
        created_date: Optional[datetime] = None
        channel_url: str = ""
        thumbnail_url: str = ""
        country: str = ""
        language: str = ""
        topics: List[str] = None
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.topics is None:
                self.topics = []
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass
    class CommentData:
        """Placeholder CommentData class when main components are not available."""
        comment_id: str = ""
        video_id: str = ""
        author: str = ""
        text: str = ""
        like_count: Optional[int] = None
        reply_count: Optional[int] = None
        published_date: Optional[datetime] = None
        parent_comment_id: Optional[str] = None
        is_reply: bool = False
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass
    class TranscriptData:
        """Placeholder TranscriptData class when main components are not available."""
        video_id: str = ""
        language: str = "en"
        transcript_text: str = ""
        segments: List[Dict[str, Any]] = None
        duration: Optional[float] = None
        auto_generated: bool = False
        confidence_score: Optional[float] = None
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.segments is None:
                self.segments = []
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass
    class CrawlResult:
        """Placeholder CrawlResult class when main components are not available."""
        success: bool = False
        data: Any = None
        error: Optional[str] = None
        url: str = ""
        timestamp: datetime = field(default_factory=datetime.now)
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass
    class ExtractResult:
        """Placeholder ExtractResult class when main components are not available."""
        success: bool = False
        extracted_data: Any = None
        source_type: str = ""
        confidence: float = 0.0
        error: Optional[str] = None
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    # Add to __all__
    __all__.extend(['YouTubeClient', 'EnhancedYouTubeClient', 'VideoData', 'ChannelData', 'CommentData', 'TranscriptData', 'CrawlResult', 'ExtractResult'])

# Package metadata
PACKAGE_INFO = {
    'name': 'YouTube Crawler',
    'version': __version__,
    'description': 'Enterprise-grade YouTube content crawler with filtering and metadata extraction',
    'author': __author__,
    'company': __company__,
    'license': 'MIT',
    'python_requires': '>=3.8',
    'keywords': ['youtube', 'crawler', 'video', 'metadata', 'scraping', 'api', 'social-media'],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia :: Video',
        'Topic :: Text Processing :: General',
    ]
}

def get_version():
    """Get the package version."""
    return __version__

def get_package_info():
    """Get complete package information."""
    return PACKAGE_INFO.copy()

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    optional_deps = []

    # Core dependencies
    core_deps = [
        ('aiohttp', 'HTTP client for async operations'),
        ('asyncpg', 'PostgreSQL async driver'),
        ('google-api-python-client', 'YouTube Data API client'),
        ('yt-dlp', 'YouTube video/metadata downloader'),
    ]

    # Optional dependencies
    opt_deps = [
        ('beautifulsoup4', 'HTML parsing for scraping'),
        ('pandas', 'Data manipulation and analysis'),
        ('textblob', 'Text analysis and sentiment'),
        ('scikit-learn', 'Machine learning features'),
        ('pydantic', 'Data validation'),
        ('PyYAML', 'Configuration file support'),
        ('Pillow', 'Image processing for thumbnails'),
        ('opencv-python', 'Video processing capabilities'),
        ('ffmpeg-python', 'Video/audio processing'),
        ('speech-recognition', 'Audio transcription'),
        ('matplotlib', 'Data visualization'),
        ('seaborn', 'Statistical data visualization'),
    ]

    for dep_name, description in core_deps:
        try:
            __import__(dep_name.replace('-', '_'))
        except ImportError:
            missing_deps.append((dep_name, description))

    for dep_name, description in opt_deps:
        try:
            __import__(dep_name.replace('-', '_'))
        except ImportError:
            optional_deps.append((dep_name, description))

    return {
        'missing_required': missing_deps,
        'missing_optional': optional_deps,
        'all_available': len(missing_deps) == 0
    }

# Convenience function for quick setup
def quick_setup_guide():
    """Print a quick setup guide for the package."""
    print(f"""
YouTube Crawler v{__version__}
===============================

Quick Setup Guide:
1. Install dependencies: pip install -r requirements.txt
2. Set up YouTube Data API credentials (optional for API access)
3. Set up PostgreSQL database
4. Configure your database connection
5. Initialize the crawler:

   from youtube_crawler import create_youtube_client

   client = await create_youtube_client(
       db_manager=your_db_manager,
       config=your_config,
       youtube_api_key=your_api_key  # Optional
   )

Features:
- Video metadata extraction
- Channel analytics
- Comment analysis
- Transcript extraction
- Thumbnail processing
- Batch video processing
- Rate limiting and optimization
- Stealth scraping capabilities

For detailed documentation, see the README.md file.
For examples, check the test files and example scripts.

Dependencies Status:
""")

    deps = check_dependencies()
    if deps['all_available']:
        print("✅ All required dependencies are available!")
    else:
        print("❌ Missing required dependencies:")
        for dep, desc in deps['missing_required']:
            print(f"   - {dep}: {desc}")

    if deps['missing_optional']:
        print("\n⚠️  Missing optional dependencies (some features may be limited):")
        for dep, desc in deps['missing_optional']:
            print(f"   - {dep}: {desc}")

# YouTube-specific constants
YOUTUBE_CONSTANTS = {
    'BASE_URL': 'https://www.youtube.com',
    'API_BASE_URL': 'https://www.googleapis.com/youtube/v3',
    'MAX_RESULTS_PER_REQUEST': 50,
    'MAX_COMMENT_THREADS': 100,
    'DEFAULT_VIDEO_QUALITY': 'medium',
    'SUPPORTED_VIDEO_FORMATS': ['mp4', 'webm', 'flv'],
    'SUPPORTED_AUDIO_FORMATS': ['mp3', 'aac', 'ogg'],
    'MAX_TRANSCRIPT_LENGTH': 10000,
    'RATE_LIMIT_REQUESTS_PER_MINUTE': 60,
    'CACHE_EXPIRY_HOURS': 24,
}

def get_youtube_constants():
    """Get YouTube-specific constants."""
    return YOUTUBE_CONSTANTS.copy()

# Health check function
def health_check():
    """Perform a health check of the YouTube crawler package."""
    results = {
        'package_version': __version__,
        'dependencies': check_dependencies(),
        'timestamp': None,
        'status': 'unknown'
    }

    try:
        from datetime import datetime
        results['timestamp'] = datetime.utcnow().isoformat()

        if results['dependencies']['all_available']:
            results['status'] = 'healthy'
        else:
            results['status'] = 'degraded'

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)

    return results
