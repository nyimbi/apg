"""
News Crawler Parsers Module
============================

Content parsing and extraction utilities for news articles.
Integrates with packages_enhanced/utils for enhanced processing.

Components:
- HTMLParser: HTML content parsing with BeautifulSoup integration
- ArticleExtractor: Article content extraction using multiple strategies
- MetadataExtractor: Metadata extraction from news articles
- ContentParser: Unified content parsing interface
- MLContentAnalyzer: ML-based content analysis using existing scorers

Features:
- Multi-strategy content extraction with fallbacks
- Integration with utils/nlp for text processing
- ML-based content scoring using existing scorers
- Metadata extraction and validation
- Content quality assessment

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

# Version information
__version__ = "4.0.0"
__author__ = "Lindela Development Team"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import parser components
try:
    from .html_parser import HTMLParser, HTMLParseResult
    from .article_extractor import ArticleExtractor, ArticleContent
    from .metadata_extractor import MetadataExtractor, ArticleMetadata
    from .content_parser import ContentParser, ParsedContent
    from .ml_content_analyzer import MLContentAnalyzer, ContentAnalysis
    _PARSER_COMPONENTS_AVAILABLE = True
    logger.debug("Parser components loaded successfully")
except ImportError as e:
    logger.warning(f"Some parser components not available: {e}")
    _PARSER_COMPONENTS_AVAILABLE = False
    
    # Placeholder classes
    class HTMLParser:
        def __init__(self, *args, **kwargs):
            raise ImportError("Parser components not available")
    
    class ArticleExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError("Parser components not available")
    
    class MetadataExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError("Parser components not available")
    
    class ContentParser:
        def __init__(self, *args, **kwargs):
            raise ImportError("Parser components not available")
    
    class MLContentAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Parser components not available")


class ParserFactory:
    """Factory for creating parser instances."""
    
    @staticmethod
    def create_parser(
        parser_type: str = "unified",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'ContentParser':
        """
        Create a parser instance based on type and configuration.
        
        Args:
            parser_type: Type of parser ('html', 'article', 'metadata', 'unified')
            config: Optional configuration dictionary
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured parser instance
        """
        if not _PARSER_COMPONENTS_AVAILABLE:
            raise ImportError("Parser components not available")
        
        config = config or {}
        config.update(kwargs)
        
        if parser_type == "html":
            return HTMLParser(config=config)
        elif parser_type == "article":
            return ArticleExtractor(config=config)
        elif parser_type == "metadata":
            return MetadataExtractor(config=config)
        elif parser_type == "unified":
            return ContentParser(config=config)
        else:
            raise ValueError(f"Unknown parser type: {parser_type}")


# Utility functions
def get_parser_health() -> Dict[str, Any]:
    """Get health status of parser components."""
    return {
        'status': 'healthy' if _PARSER_COMPONENTS_AVAILABLE else 'degraded',
        'components_available': _PARSER_COMPONENTS_AVAILABLE,
        'version': __version__,
        'supported_parsers': [
            'html', 'article', 'metadata', 'unified'
        ] if _PARSER_COMPONENTS_AVAILABLE else []
    }


def validate_content(content: str) -> bool:
    """Validate content for parsing."""
    if not content or not isinstance(content, str):
        return False
    
    # Basic content validation
    if len(content.strip()) < 10:
        return False
    
    return True


# Quick parse functions
async def quick_parse_html(html: str, **kwargs) -> Dict[str, Any]:
    """Quick HTML parsing function."""
    if not _PARSER_COMPONENTS_AVAILABLE:
        raise ImportError("Parser components not available")
    
    parser = HTMLParser(**kwargs)
    return await parser.parse(html)


async def quick_extract_article(html: str, url: str = "", **kwargs) -> Dict[str, Any]:
    """Quick article extraction function."""
    if not _PARSER_COMPONENTS_AVAILABLE:
        raise ImportError("Parser components not available")
    
    extractor = ArticleExtractor(**kwargs)
    return await extractor.extract(html, url)


async def quick_analyze_content(content: str, **kwargs) -> Dict[str, Any]:
    """Quick ML content analysis function."""
    if not _PARSER_COMPONENTS_AVAILABLE:
        raise ImportError("Parser components not available")
    
    analyzer = MLContentAnalyzer(**kwargs)
    return await analyzer.analyze(content)


# Export all public components
__all__ = [
    # Core parser classes
    'HTMLParser',
    'ArticleExtractor', 
    'MetadataExtractor',
    'ContentParser',
    'MLContentAnalyzer',
    'ParserFactory',
    
    # Data structures
    'HTMLParseResult',
    'ArticleContent',
    'ArticleMetadata', 
    'ParsedContent',
    'ContentAnalysis',
    
    # Utility functions
    'get_parser_health',
    'validate_content',
    'quick_parse_html',
    'quick_extract_article',
    'quick_analyze_content',
    
    # Version info
    '__version__',
    '__author__',
    '__license__'
]

# Module initialization
logger.info(f"News Crawler Parsers Module v{__version__} initialized")
logger.info(f"Parser components available: {_PARSER_COMPONENTS_AVAILABLE}")