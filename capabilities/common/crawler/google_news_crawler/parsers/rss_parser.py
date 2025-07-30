"""
RSS/Atom Feed Parser
===================

Comprehensive RSS and Atom feed parser with support for various feed formats,
error handling, and content extraction capabilities.

Features:
- RSS 2.0, RSS 1.0, and Atom feed parsing
- Robust error handling and recovery
- Content extraction and cleaning
- Media enclosure handling
- Dublin Core metadata support
- Content deduplication
- Multi-language support

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    date_parser = None

from . import BaseParser, ParseResult, ArticleData, ParseStatus, ContentType

logger = logging.getLogger(__name__)

class RSSParser(BaseParser):
    """Parser for RSS and Atom feeds."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RSS parser."""
        super().__init__(config)

        # Configuration options
        self.max_entries = config.get('max_entries', 100) if config else 100
        self.extract_content = config.get('extract_content', True) if config else True
        self.clean_html = config.get('clean_html', True) if config else True
        self.extract_images = config.get('extract_images', True) if config else True
        self.parse_enclosures = config.get('parse_enclosures', True) if config else True

        # Namespace mappings for XML parsing
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'content': 'http://purl.org/rss/1.0/modules/content/',
            'media': 'http://search.yahoo.com/mrss/',
            'georss': 'http://www.georss.org/georss',
        }

    def can_parse(self, content: str, content_type: ContentType = None) -> bool:
        """Check if content can be parsed as RSS/Atom feed."""
        if content_type == ContentType.RSS_FEED or content_type == ContentType.ATOM_FEED:
            return True

        content_lower = content.lower().strip()

        # Check for RSS indicators
        rss_indicators = [
            '<rss', '<feed', 'xmlns="http://www.w3.org/2005/Atom"',
            'version="2.0"', 'version="1.0"', '<channel>', '<item>'
        ]

        return any(indicator in content_lower for indicator in rss_indicators)

    async def parse(self, content: str, source_url: str = None, **kwargs) -> ParseResult:
        """Parse RSS/Atom feed content."""
        try:
            # Try feedparser first if available
            if FEEDPARSER_AVAILABLE and feedparser:
                result = await self._parse_with_feedparser(content, source_url)
                if result.status == ParseStatus.SUCCESS:
                    return result

            # Fallback to manual XML parsing
            return await self._parse_with_xml(content, source_url)

        except Exception as e:
            logger.error(f"RSS parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error=str(e),
                metadata=self.extract_metadata(content)
            )

    async def _parse_with_feedparser(self, content: str, source_url: str = None) -> ParseResult:
        """Parse using feedparser library."""
        try:
            feed = feedparser.parse(content)

            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed has issues: {feed.bozo_exception}")

            if not feed.entries:
                return ParseResult(
                    status=ParseStatus.FAILED,
                    error="No entries found in feed",
                    metadata=self.extract_metadata(content)
                )

            articles = []

            # Extract feed metadata
            feed_info = self._extract_feed_info(feed, source_url)

            # Parse entries
            for entry in feed.entries[:self.max_entries]:
                article = await self._parse_entry_feedparser(entry, feed_info)
                if article:
                    articles.append(article)

            return ParseResult(
                status=ParseStatus.SUCCESS,
                content={'articles': articles, 'feed_info': feed_info},
                metadata={
                    **self.extract_metadata(content),
                    'parser_used': 'feedparser',
                    'total_entries': len(feed.entries),
                    'parsed_entries': len(articles),
                }
            )

        except Exception as e:
            logger.error(f"Feedparser parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error=f"Feedparser error: {e}",
                metadata=self.extract_metadata(content)
            )

    async def _parse_with_xml(self, content: str, source_url: str = None) -> ParseResult:
        """Parse using manual XML parsing."""
        try:
            # Clean content for XML parsing
            content = self._clean_xml_content(content)

            root = ET.fromstring(content)

            # Determine feed type
            if root.tag == 'rss' or root.find('channel') is not None:
                return await self._parse_rss(root, source_url)
            elif root.tag.endswith('feed') or 'atom' in root.tag.lower():
                return await self._parse_atom(root, source_url)
            else:
                return ParseResult(
                    status=ParseStatus.FAILED,
                    error="Unknown feed format",
                    metadata=self.extract_metadata(content)
                )

        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error=f"XML parse error: {e}",
                metadata=self.extract_metadata(content)
            )

    async def _parse_rss(self, root: ET.Element, source_url: str = None) -> ParseResult:
        """Parse RSS feed."""
        articles = []

        # Find channel element
        channel = root.find('channel')
        if channel is None:
            channel = root

        # Extract feed info
        feed_info = {
            'title': self._get_text(channel, 'title'),
            'description': self._get_text(channel, 'description'),
            'link': self._get_text(channel, 'link'),
            'language': self._get_text(channel, 'language'),
            'last_build_date': self._parse_date(self._get_text(channel, 'lastBuildDate')),
            'source_url': source_url,
        }

        # Parse items
        items = channel.findall('item')
        for item in items[:self.max_entries]:
            article = await self._parse_rss_item(item, feed_info)
            if article:
                articles.append(article)

        return ParseResult(
            status=ParseStatus.SUCCESS,
            content={'articles': articles, 'feed_info': feed_info},
            metadata={
                **self.extract_metadata(str(ET.tostring(root, encoding='unicode'))),
                'parser_used': 'xml',
                'feed_type': 'rss',
                'total_items': len(items),
                'parsed_items': len(articles),
            }
        )

    async def _parse_atom(self, root: ET.Element, source_url: str = None) -> ParseResult:
        """Parse Atom feed."""
        articles = []

        # Extract feed info
        feed_info = {
            'title': self._get_text_ns(root, 'atom:title'),
            'description': self._get_text_ns(root, 'atom:subtitle'),
            'link': self._get_atom_link(root),
            'language': root.get('{http://www.w3.org/XML/1998/namespace}lang'),
            'updated': self._parse_date(self._get_text_ns(root, 'atom:updated')),
            'source_url': source_url,
        }

        # Parse entries
        entries = root.findall('atom:entry', self.namespaces)
        for entry in entries[:self.max_entries]:
            article = await self._parse_atom_entry(entry, feed_info)
            if article:
                articles.append(article)

        return ParseResult(
            status=ParseStatus.SUCCESS,
            content={'articles': articles, 'feed_info': feed_info},
            metadata={
                **self.extract_metadata(str(ET.tostring(root, encoding='unicode'))),
                'parser_used': 'xml',
                'feed_type': 'atom',
                'total_entries': len(entries),
                'parsed_entries': len(articles),
            }
        )

    async def _parse_entry_feedparser(self, entry: Any, feed_info: Dict[str, Any]) -> Optional[ArticleData]:
        """Parse single entry using feedparser."""
        try:
            # Extract basic information
            title = self._clean_text(getattr(entry, 'title', ''))
            url = getattr(entry, 'link', '')

            if not title or not url:
                return None

            description = self._clean_text(getattr(entry, 'summary', ''))

            # Extract content
            content = None
            if hasattr(entry, 'content'):
                content = ' '.join([c.value for c in entry.content if hasattr(c, 'value')])
            elif hasattr(entry, 'description'):
                content = entry.description

            if content:
                content = self._clean_html_content(content) if self.clean_html else content

            # Extract metadata
            published_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

            author = getattr(entry, 'author', None)
            if not author and hasattr(entry, 'authors'):
                author = ', '.join([a.get('name', '') for a in entry.authors if a.get('name')])

            # Extract images
            images = []
            if self.extract_images:
                images = self._extract_images_feedparser(entry)

            # Extract tags
            tags = []
            if hasattr(entry, 'tags'):
                tags = [tag.term for tag in entry.tags if hasattr(tag, 'term')]

            return ArticleData(
                title=title,
                url=url,
                published_date=published_date,
                description=description,
                content=content,
                author=author,
                publisher=feed_info.get('title'),
                source_domain=self._extract_domain(url),
                language=feed_info.get('language'),
                tags=tags,
                images=images,
                word_count=len(content.split()) if content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse entry: {e}")
            return None

    async def _parse_rss_item(self, item: ET.Element, feed_info: Dict[str, Any]) -> Optional[ArticleData]:
        """Parse RSS item."""
        try:
            title = self._clean_text(self._get_text(item, 'title'))
            url = self._get_text(item, 'link')

            if not title or not url:
                return None

            description = self._clean_text(self._get_text(item, 'description'))

            # Extract content
            content = self._get_text_ns(item, 'content:encoded')
            if not content:
                content = description

            if content and self.clean_html:
                content = self._clean_html_content(content)

            # Parse date
            pub_date = self._parse_date(self._get_text(item, 'pubDate'))

            # Extract author
            author = self._get_text(item, 'author') or self._get_text_ns(item, 'dc:creator')

            # Extract images
            images = []
            if self.extract_images:
                images = self._extract_images_rss(item)

            # Extract categories as tags
            tags = [cat.text for cat in item.findall('category') if cat.text]

            return ArticleData(
                title=title,
                url=url,
                published_date=pub_date,
                description=description,
                content=content,
                author=author,
                publisher=feed_info.get('title'),
                source_domain=self._extract_domain(url),
                language=feed_info.get('language'),
                tags=tags,
                images=images,
                word_count=len(content.split()) if content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse RSS item: {e}")
            return None

    async def _parse_atom_entry(self, entry: ET.Element, feed_info: Dict[str, Any]) -> Optional[ArticleData]:
        """Parse Atom entry."""
        try:
            title = self._clean_text(self._get_text_ns(entry, 'atom:title'))
            url = self._get_atom_link(entry)

            if not title or not url:
                return None

            description = self._clean_text(self._get_text_ns(entry, 'atom:summary'))

            # Extract content
            content = self._get_text_ns(entry, 'atom:content')
            if not content:
                content = description

            if content and self.clean_html:
                content = self._clean_html_content(content)

            # Parse date
            pub_date = self._parse_date(self._get_text_ns(entry, 'atom:published'))
            if not pub_date:
                pub_date = self._parse_date(self._get_text_ns(entry, 'atom:updated'))

            # Extract author
            author_elem = entry.find('atom:author', self.namespaces)
            author = None
            if author_elem is not None:
                author = self._get_text_ns(author_elem, 'atom:name')

            # Extract images
            images = []
            if self.extract_images:
                images = self._extract_images_atom(entry)

            # Extract categories as tags
            categories = entry.findall('atom:category', self.namespaces)
            tags = [cat.get('term') for cat in categories if cat.get('term')]

            return ArticleData(
                title=title,
                url=url,
                published_date=pub_date,
                description=description,
                content=content,
                author=author,
                publisher=feed_info.get('title'),
                source_domain=self._extract_domain(url),
                language=feed_info.get('language'),
                tags=tags,
                images=images,
                word_count=len(content.split()) if content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse Atom entry: {e}")
            return None

    def _extract_feed_info(self, feed: Any, source_url: str = None) -> Dict[str, Any]:
        """Extract feed information from feedparser feed."""
        return {
            'title': getattr(feed.feed, 'title', ''),
            'description': getattr(feed.feed, 'description', ''),
            'link': getattr(feed.feed, 'link', ''),
            'language': getattr(feed.feed, 'language', ''),
            'last_build_date': getattr(feed.feed, 'updated', ''),
            'source_url': source_url,
        }

    def _extract_images_feedparser(self, entry: Any) -> List[str]:
        """Extract images from feedparser entry."""
        images = []

        # Check enclosures
        if hasattr(entry, 'enclosures'):
            for enclosure in entry.enclosures:
                if hasattr(enclosure, 'type') and enclosure.type.startswith('image/'):
                    images.append(enclosure.href)

        # Check media content
        if hasattr(entry, 'media_content'):
            for media in entry.media_content:
                if media.get('type', '').startswith('image/'):
                    images.append(media['url'])

        # Extract from content
        if hasattr(entry, 'content'):
            for content in entry.content:
                if hasattr(content, 'value'):
                    images.extend(self._extract_images_from_html(content.value))

        return list(set(images))  # Remove duplicates

    def _extract_images_rss(self, item: ET.Element) -> List[str]:
        """Extract images from RSS item."""
        images = []

        # Check enclosures
        for enclosure in item.findall('enclosure'):
            if enclosure.get('type', '').startswith('image/'):
                images.append(enclosure.get('url'))

        # Check media elements
        for media in item.findall('media:content', self.namespaces):
            if media.get('type', '').startswith('image/'):
                images.append(media.get('url'))

        # Extract from content
        content = self._get_text_ns(item, 'content:encoded') or self._get_text(item, 'description')
        if content:
            images.extend(self._extract_images_from_html(content))

        return list(set(images))

    def _extract_images_atom(self, entry: ET.Element) -> List[str]:
        """Extract images from Atom entry."""
        images = []

        # Check links with image type
        for link in entry.findall('atom:link', self.namespaces):
            if link.get('type', '').startswith('image/'):
                images.append(link.get('href'))

        # Extract from content
        content = self._get_text_ns(entry, 'atom:content') or self._get_text_ns(entry, 'atom:summary')
        if content:
            images.extend(self._extract_images_from_html(content))

        return list(set(images))

    def _extract_images_from_html(self, html_content: str) -> List[str]:
        """Extract image URLs from HTML content."""
        if not html_content:
            return []

        # Simple regex-based extraction
        img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        matches = re.findall(img_pattern, html_content, re.IGNORECASE)

        return matches

    def _clean_xml_content(self, content: str) -> str:
        """Clean XML content for parsing."""
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]

        # Fix common XML issues
        content = content.replace('&nbsp;', ' ')
        content = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', content)

        return content

    def _clean_html_content(self, content: str) -> str:
        """Clean HTML content."""
        if not content:
            return ""

        # Remove HTML tags (simple approach)
        clean_text = re.sub(r'<[^>]+>', '', content)

        # Decode HTML entities
        import html
        clean_text = html.unescape(clean_text)

        return self._clean_text(clean_text)

    def _get_text(self, element: ET.Element, tag: str) -> Optional[str]:
        """Get text content from XML element."""
        child = element.find(tag)
        return child.text if child is not None else None

    def _get_text_ns(self, element: ET.Element, tag: str) -> Optional[str]:
        """Get text content from XML element with namespace."""
        child = element.find(tag, self.namespaces)
        return child.text if child is not None else None

    def _get_atom_link(self, element: ET.Element) -> Optional[str]:
        """Get link from Atom element."""
        link = element.find('atom:link[@rel="alternate"]', self.namespaces)
        if link is None:
            link = element.find('atom:link', self.namespaces)

        return link.get('href') if link is not None else None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None

        try:
            if DATEUTIL_AVAILABLE and date_parser:
                return date_parser.parse(date_str)
            else:
                # Basic RFC 2822 parsing
                from email.utils import parsedate_to_datetime
                return parsedate_to_datetime(date_str)
        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")
            return None
