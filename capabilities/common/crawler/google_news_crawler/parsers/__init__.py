"""
News Parsers Package
===================

This package provides comprehensive parsing capabilities for various news sources,
including Google News RSS feeds, direct news site scraping, and content extraction.

The parsers are designed to handle:
- RSS/Atom feed parsing
- HTML content extraction
- Article metadata extraction
- Content cleaning and normalization
- Multi-language support
- Error handling and recovery

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ParseStatus(Enum):
    """Status of parsing operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

class ContentType(Enum):
    """Type of content being parsed."""
    RSS_FEED = "rss_feed"
    ATOM_FEED = "atom_feed"
    HTML_ARTICLE = "html_article"
    JSON_ARTICLE = "json_article"
    PLAIN_TEXT = "plain_text"

@dataclass
class ParseResult:
    """Result of a parsing operation."""
    status: ParseStatus
    content: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    extracted_at: datetime = None

    def __post_init__(self):
        if self.extracted_at is None:
            self.extracted_at = datetime.utcnow()

@dataclass
class ArticleData:
    """Structured article data."""
    title: str
    url: str
    published_date: Optional[datetime] = None
    description: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    source_domain: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    images: Optional[List[str]] = None
    sentiment: Optional[Dict[str, float]] = None
    readability_score: Optional[float] = None
    word_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'title': self.title,
            'url': self.url,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'description': self.description,
            'content': self.content,
            'author': self.author,
            'publisher': self.publisher,
            'source_domain': self.source_domain,
            'language': self.language,
            'tags': self.tags or [],
            'images': self.images or [],
            'sentiment': self.sentiment,
            'readability_score': self.readability_score,
            'word_count': self.word_count,
        }

class BaseParser(ABC):
    """Abstract base class for all parsers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize parser with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def parse(self, content: str, source_url: str = None, **kwargs) -> ParseResult:
        """Parse content and return structured data."""
        pass

    @abstractmethod
    def can_parse(self, content: str, content_type: ContentType = None) -> bool:
        """Check if this parser can handle the given content."""
        pass

    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content."""
        return {
            'content_length': len(content),
            'content_type': self._detect_content_type(content),
            'extracted_at': datetime.utcnow().isoformat(),
        }

    def _detect_content_type(self, content: str) -> ContentType:
        """Detect content type from content."""
        content_lower = content.lower().strip()

        if content_lower.startswith('<?xml') or '<rss' in content_lower:
            return ContentType.RSS_FEED
        elif '<feed' in content_lower or 'xmlns="http://www.w3.org/2005/Atom"' in content_lower:
            return ContentType.ATOM_FEED
        elif content_lower.startswith('{') and content_lower.endswith('}'):
            return ContentType.JSON_ARTICLE
        elif '<html' in content_lower or '<body' in content_lower:
            return ContentType.HTML_ARTICLE
        else:
            return ContentType.PLAIN_TEXT

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove common unwanted characters
        text = text.replace('\x00', '').replace('\r', ' ').replace('\n', ' ')

        return text.strip()

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return None

class ParserRegistry:
    """Registry for managing different parsers."""

    def __init__(self):
        self._parsers = {}
        self._default_parser = None

    def register(self, name: str, parser: BaseParser, is_default: bool = False):
        """Register a parser."""
        self._parsers[name] = parser
        if is_default:
            self._default_parser = parser
        logger.info(f"Registered parser: {name}")

    def get_parser(self, name: str) -> Optional[BaseParser]:
        """Get parser by name."""
        return self._parsers.get(name)

    def get_suitable_parser(self, content: str, content_type: ContentType = None) -> Optional[BaseParser]:
        """Get the most suitable parser for the content."""
        for parser in self._parsers.values():
            if parser.can_parse(content, content_type):
                return parser
        return self._default_parser

    def list_parsers(self) -> List[str]:
        """List all registered parsers."""
        return list(self._parsers.keys())

# Global parser registry
parser_registry = ParserRegistry()

# Import and register parsers when available
def _register_parsers():
    """Register parsers with the global registry."""
    try:
        from .rss_parser import RSSParser
        parser_registry.register("rss", RSSParser(), is_default=True)
    except ImportError:
        logger.warning("RSSParser not available")

    try:
        from .html_parser import HTMLParser
        parser_registry.register("html", HTMLParser())
    except ImportError:
        logger.warning("HTMLParser not available")

    try:
        from .json_parser import JSONParser
        parser_registry.register("json", JSONParser())
    except ImportError:
        logger.warning("JSONParser not available")

    try:
        from .intelligent_parser import IntelligentParser
        parser_registry.register("intelligent", IntelligentParser())
    except ImportError:
        logger.warning("IntelligentParser not available")

# Register parsers on module import
_register_parsers()

__all__ = [
    'BaseParser',
    'ParseResult',
    'ArticleData',
    'ParseStatus',
    'ContentType',
    'ParserRegistry',
    'parser_registry',
]
