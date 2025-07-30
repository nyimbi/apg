"""
JSON Article Parser
===================

Parser for JSON-structured article data, supporting various JSON formats
commonly used by news APIs and structured data endpoints.

Features:
- JSON-LD structured data parsing
- News API response parsing
- Custom JSON schema support
- Nested object traversal
- Data validation and normalization
- Error handling for malformed JSON
- Multi-format support (single articles, article arrays)

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    date_parser = None

from . import BaseParser, ParseResult, ArticleData, ParseStatus, ContentType

logger = logging.getLogger(__name__)

class JSONParser(BaseParser):
    """Parser for JSON article data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize JSON parser."""
        super().__init__(config)

        # Configuration options
        self.max_articles = config.get('max_articles', 50) if config else 50
        self.validate_urls = config.get('validate_urls', True) if config else True
        self.extract_nested = config.get('extract_nested', True) if config else True
        self.strict_validation = config.get('strict_validation', False) if config else False

        # Common JSON field mappings
        self.field_mappings = {
            'title': ['title', 'headline', 'name', 'subject'],
            'url': ['url', 'link', 'permalink', 'canonical_url'],
            'description': ['description', 'summary', 'excerpt', 'abstract', 'lead'],
            'content': ['content', 'body', 'text', 'article_body', 'full_text'],
            'author': ['author', 'byline', 'writer', 'creator'],
            'published_date': ['published_date', 'date_published', 'publish_date', 'created_at', 'timestamp'],
            'source': ['source', 'publisher', 'publication', 'site_name'],
            'tags': ['tags', 'keywords', 'categories', 'topics'],
            'images': ['images', 'image_urls', 'photos', 'pictures'],
        }

        # Schema.org contexts
        self.schema_contexts = [
            'https://schema.org',
            'http://schema.org',
            'schema.org'
        ]

        # Supported article types
        self.article_types = [
            'Article', 'NewsArticle', 'BlogPosting', 'ScholarlyArticle',
            'TechArticle', 'Report', 'WebPage'
        ]

    def can_parse(self, content: str, content_type: ContentType = None) -> bool:
        """Check if content can be parsed as JSON."""
        if content_type == ContentType.JSON_ARTICLE:
            return True

        content = content.strip()

        # Check for JSON indicators
        if (content.startswith('{') and content.endswith('}')) or \
           (content.startswith('[') and content.endswith(']')):
            try:
                json.loads(content)
                return True
            except json.JSONDecodeError:
                return False

        return False

    async def parse(self, content: str, source_url: str = None, **kwargs) -> ParseResult:
        """Parse JSON content and extract article data."""
        try:
            # Parse JSON
            data = json.loads(content)

            # Handle different JSON structures
            articles = []

            if isinstance(data, dict):
                # Single article or structured data
                if self._is_article_data(data):
                    article = await self._parse_single_article(data, source_url)
                    if article:
                        articles.append(article)
                else:
                    # Try to find articles in nested structure
                    nested_articles = await self._extract_nested_articles(data, source_url)
                    articles.extend(nested_articles)

            elif isinstance(data, list):
                # Array of articles
                for item in data[:self.max_articles]:
                    if isinstance(item, dict):
                        article = await self._parse_single_article(item, source_url)
                        if article:
                            articles.append(article)

            if not articles:
                return ParseResult(
                    status=ParseStatus.FAILED,
                    error="No valid articles found in JSON data",
                    metadata=self.extract_metadata(content)
                )

            return ParseResult(
                status=ParseStatus.SUCCESS,
                content={'articles': [article.to_dict() for article in articles]},
                metadata={
                    **self.extract_metadata(content),
                    'parser_used': 'json',
                    'total_articles': len(articles),
                    'json_structure': 'array' if isinstance(data, list) else 'object',
                }
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error=f"Invalid JSON: {e}",
                metadata=self.extract_metadata(content)
            )
        except Exception as e:
            logger.error(f"JSON article parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error=str(e),
                metadata=self.extract_metadata(content)
            )

    async def _parse_single_article(self, data: Dict[str, Any], source_url: str = None) -> Optional[ArticleData]:
        """Parse a single article from JSON data."""
        try:
            # Determine parsing strategy based on data structure
            if self._is_schema_org_data(data):
                return await self._parse_schema_org_article(data, source_url)
            elif self._is_news_api_data(data):
                return await self._parse_news_api_article(data, source_url)
            else:
                return await self._parse_generic_article(data, source_url)

        except Exception as e:
            logger.error(f"Failed to parse single article: {e}")
            return None

    async def _parse_schema_org_article(self, data: Dict[str, Any], source_url: str = None) -> Optional[ArticleData]:
        """Parse Schema.org structured data."""
        try:
            # Extract basic fields
            title = self._get_field_value(data, ['headline', 'name', 'title'])
            url = self._get_field_value(data, ['url', 'mainEntityOfPage']) or source_url
            description = self._get_field_value(data, ['description'])
            content = self._get_field_value(data, ['articleBody', 'text'])

            # Extract author
            author = None
            author_data = data.get('author')
            if author_data:
                if isinstance(author_data, dict):
                    author = author_data.get('name')
                elif isinstance(author_data, list) and author_data:
                    first_author = author_data[0]
                    if isinstance(first_author, dict):
                        author = first_author.get('name')
                    else:
                        author = str(first_author)
                elif isinstance(author_data, str):
                    author = author_data

            # Extract publisher
            publisher = None
            publisher_data = data.get('publisher')
            if publisher_data:
                if isinstance(publisher_data, dict):
                    publisher = publisher_data.get('name')
                elif isinstance(publisher_data, str):
                    publisher = publisher_data

            # Extract publish date
            pub_date = None
            date_published = data.get('datePublished')
            if date_published:
                pub_date = self._parse_date(date_published)

            # Extract images
            images = []
            image_data = data.get('image')
            if image_data:
                images = self._extract_images_from_data(image_data)

            # Extract keywords/tags
            tags = []
            keywords = data.get('keywords')
            if keywords:
                if isinstance(keywords, str):
                    tags = [k.strip() for k in keywords.split(',')]
                elif isinstance(keywords, list):
                    tags = [str(k) for k in keywords]

            # Basic validation
            if not title:
                return None

            return ArticleData(
                title=self._clean_text(title),
                url=url,
                description=self._clean_text(description) if description else None,
                content=self._clean_text(content) if content else None,
                author=self._clean_text(author) if author else None,
                publisher=self._clean_text(publisher) if publisher else None,
                published_date=pub_date,
                source_domain=self._extract_domain(url) if url else None,
                tags=tags,
                images=images,
                word_count=len(content.split()) if content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse Schema.org article: {e}")
            return None

    async def _parse_news_api_article(self, data: Dict[str, Any], source_url: str = None) -> Optional[ArticleData]:
        """Parse News API formatted article."""
        try:
            title = data.get('title')
            url = data.get('url') or source_url
            description = data.get('description')
            content = data.get('content')

            # Handle source object
            source_data = data.get('source', {})
            publisher = source_data.get('name') if isinstance(source_data, dict) else None

            # Parse publish date
            pub_date = None
            published_at = data.get('publishedAt')
            if published_at:
                pub_date = self._parse_date(published_at)

            # Extract author
            author = data.get('author')

            # Extract image
            images = []
            url_to_image = data.get('urlToImage')
            if url_to_image:
                images = [url_to_image]

            # Basic validation
            if not title:
                return None

            return ArticleData(
                title=self._clean_text(title),
                url=url,
                description=self._clean_text(description) if description else None,
                content=self._clean_text(content) if content else None,
                author=self._clean_text(author) if author else None,
                publisher=self._clean_text(publisher) if publisher else None,
                published_date=pub_date,
                source_domain=self._extract_domain(url) if url else None,
                images=images,
                word_count=len(content.split()) if content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse News API article: {e}")
            return None

    async def _parse_generic_article(self, data: Dict[str, Any], source_url: str = None) -> Optional[ArticleData]:
        """Parse generic JSON article using field mappings."""
        try:
            # Extract fields using mappings
            title = self._extract_mapped_field(data, 'title')
            url = self._extract_mapped_field(data, 'url') or source_url
            description = self._extract_mapped_field(data, 'description')
            content = self._extract_mapped_field(data, 'content')
            author = self._extract_mapped_field(data, 'author')
            publisher = self._extract_mapped_field(data, 'source')

            # Extract publish date
            pub_date = None
            date_str = self._extract_mapped_field(data, 'published_date')
            if date_str:
                pub_date = self._parse_date(str(date_str))

            # Extract tags
            tags = []
            tags_data = self._extract_mapped_field(data, 'tags')
            if tags_data:
                if isinstance(tags_data, list):
                    tags = [str(tag) for tag in tags_data]
                elif isinstance(tags_data, str):
                    tags = [tag.strip() for tag in tags_data.split(',')]

            # Extract images
            images = []
            images_data = self._extract_mapped_field(data, 'images')
            if images_data:
                images = self._extract_images_from_data(images_data)

            # Basic validation
            if not title:
                return None

            return ArticleData(
                title=self._clean_text(title),
                url=url,
                description=self._clean_text(description) if description else None,
                content=self._clean_text(content) if content else None,
                author=self._clean_text(author) if author else None,
                publisher=self._clean_text(publisher) if publisher else None,
                published_date=pub_date,
                source_domain=self._extract_domain(url) if url else None,
                tags=tags,
                images=images,
                word_count=len(content.split()) if content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse generic article: {e}")
            return None

    async def _extract_nested_articles(self, data: Dict[str, Any], source_url: str = None) -> List[ArticleData]:
        """Extract articles from nested JSON structure."""
        articles = []

        # Common nested structures
        nested_keys = ['articles', 'items', 'results', 'data', 'entries', 'posts']

        for key in nested_keys:
            if key in data:
                nested_data = data[key]
                if isinstance(nested_data, list):
                    for item in nested_data[:self.max_articles]:
                        if isinstance(item, dict):
                            article = await self._parse_single_article(item, source_url)
                            if article:
                                articles.append(article)
                    break

        # If no nested articles found, try to traverse all keys
        if not articles and self.extract_nested:
            for value in data.values():
                if isinstance(value, list):
                    for item in value[:self.max_articles]:
                        if isinstance(item, dict) and self._is_article_data(item):
                            article = await self._parse_single_article(item, source_url)
                            if article:
                                articles.append(article)
                elif isinstance(value, dict) and self._is_article_data(value):
                    article = await self._parse_single_article(value, source_url)
                    if article:
                        articles.append(article)

        return articles

    def _is_article_data(self, data: Dict[str, Any]) -> bool:
        """Check if data appears to be article data."""
        if not isinstance(data, dict):
            return False

        # Check for common article fields
        article_indicators = [
            'title', 'headline', 'name',
            'url', 'link', 'permalink',
            'content', 'body', 'text',
            'description', 'summary'
        ]

        return any(indicator in data for indicator in article_indicators)

    def _is_schema_org_data(self, data: Dict[str, Any]) -> bool:
        """Check if data is Schema.org structured data."""
        if not isinstance(data, dict):
            return False

        # Check for Schema.org context
        context = data.get('@context', '')
        if any(ctx in str(context) for ctx in self.schema_contexts):
            return True

        # Check for Schema.org type
        schema_type = data.get('@type', '')
        if schema_type in self.article_types:
            return True

        return False

    def _is_news_api_data(self, data: Dict[str, Any]) -> bool:
        """Check if data is from News API."""
        if not isinstance(data, dict):
            return False

        # News API specific fields
        news_api_fields = ['publishedAt', 'urlToImage', 'source']
        return any(field in data for field in news_api_fields)

    def _extract_mapped_field(self, data: Dict[str, Any], field_type: str) -> Any:
        """Extract field value using field mappings."""
        if field_type not in self.field_mappings:
            return None

        for field_name in self.field_mappings[field_type]:
            if field_name in data:
                return data[field_name]

        return None

    def _get_field_value(self, data: Dict[str, Any], field_names: List[str]) -> Any:
        """Get field value from data using multiple possible field names."""
        for field_name in field_names:
            if field_name in data:
                value = data[field_name]
                if value is not None:
                    return value
        return None

    def _extract_images_from_data(self, image_data: Any) -> List[str]:
        """Extract image URLs from various data formats."""
        images = []

        if isinstance(image_data, str):
            images = [image_data]
        elif isinstance(image_data, list):
            for item in image_data:
                if isinstance(item, str):
                    images.append(item)
                elif isinstance(item, dict):
                    # Look for URL in nested object
                    url = item.get('url') or item.get('src') or item.get('href')
                    if url:
                        images.append(url)
        elif isinstance(image_data, dict):
            # Single image object
            url = image_data.get('url') or image_data.get('src') or image_data.get('href')
            if url:
                images.append(url)

        # Validate URLs if configured
        if self.validate_urls:
            images = [img for img in images if self._is_valid_url(img)]

        return images

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        if not url or not isinstance(url, str):
            return False

        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None

        try:
            if DATEUTIL_AVAILABLE and date_parser:
                return date_parser.parse(date_str)
            else:
                # Basic ISO format parsing
                import datetime as dt

                # Try common formats
                formats = [
                    '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%dT%H:%M:%S%z',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                ]

                for fmt in formats:
                    try:
                        return dt.datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue

        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")

        return None
