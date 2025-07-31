"""
HTML Article Parser
===================

Comprehensive HTML parser for extracting article content from web pages.

Features:
- Article content extraction using multiple algorithms
- Metadata extraction (title, author, publish date, etc.)
- Image extraction and processing
- Content cleaning and normalization
- Readability scoring
- Multi-language support
- Schema.org microdata support
- Open Graph and Twitter Card extraction

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urljoin, urlparse
import html

try:
    from bs4 import BeautifulSoup, Comment
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    BeautifulSoup = None

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    Document = None

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    date_parser = None

from . import BaseParser, ParseResult, ArticleData, ParseStatus, ContentType

logger = logging.getLogger(__name__)

class HTMLParser(BaseParser):
    """Parser for HTML article content."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HTML parser."""
        super().__init__(config)

        # Configuration options
        self.extract_images = config.get('extract_images', True) if config else True
        self.extract_links = config.get('extract_links', False) if config else False
        self.clean_content = config.get('clean_content', True) if config else True
        self.calculate_readability = config.get('calculate_readability', True) if config else True
        self.min_content_length = config.get('min_content_length', 100) if config else 100
        self.max_content_length = config.get('max_content_length', 50000) if config else 50000

        # Content extraction selectors (in order of preference)
        self.content_selectors = [
            'article',
            '[role="main"]',
            '.post-content',
            '.article-content',
            '.entry-content',
            '.content',
            '.post-body',
            '.article-body',
            '#content',
            '#main-content',
            '.main-content',
        ]

        # Elements to remove during cleaning
        self.noise_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            '.ad', '.advertisement', '.ads', '.social-share',
            '.related-posts', '.comments', '.comment-form',
            '.newsletter-signup', '.sidebar', '.widget',
            '[class*="ad-"]', '[id*="ad-"]', '[class*="social"]'
        ]

        # Schema.org article types
        self.article_schemas = [
            'Article', 'NewsArticle', 'BlogPosting', 'ScholarlyArticle',
            'TechArticle', 'Report'
        ]

    def can_parse(self, content: str, content_type: ContentType = None) -> bool:
        """Check if content can be parsed as HTML."""
        if content_type == ContentType.HTML_ARTICLE:
            return True

        content_lower = content.lower().strip()

        # Check for HTML indicators
        html_indicators = ['<html', '<body', '<head', '<div', '<article', '<p>']
        return any(indicator in content_lower for indicator in html_indicators)

    async def parse(self, content: str, source_url: str = None, **kwargs) -> ParseResult:
        """Parse HTML content and extract article data."""
        try:
            if not BEAUTIFULSOUP_AVAILABLE:
                return ParseResult(
                    status=ParseStatus.FAILED,
                    error="BeautifulSoup not available for HTML parsing",
                    metadata=self.extract_metadata(content)
                )

            soup = BeautifulSoup(content, 'html.parser')

            # Extract article data using multiple methods
            article_data = await self._extract_article_data(soup, source_url)

            if not article_data:
                return ParseResult(
                    status=ParseStatus.FAILED,
                    error="Could not extract article data",
                    metadata=self.extract_metadata(content)
                )

            # Validate content quality
            if not self._validate_content_quality(article_data):
                return ParseResult(
                    status=ParseStatus.PARTIAL,
                    content={'articles': [article_data.to_dict()]},
                    error="Content quality below threshold",
                    metadata=self.extract_metadata(content)
                )

            return ParseResult(
                status=ParseStatus.SUCCESS,
                content={'articles': [article_data.to_dict()]},
                metadata={
                    **self.extract_metadata(content),
                    'parser_used': 'html',
                    'extraction_method': article_data.to_dict().get('_extraction_method', 'unknown'),
                    'content_quality_score': self._calculate_content_quality_score(article_data),
                }
            )

        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error=str(e),
                metadata=self.extract_metadata(content)
            )

    async def _extract_article_data(self, soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Extract article data using multiple extraction methods."""

        # Try different extraction methods in order of preference
        extraction_methods = [
            ('schema_org', self._extract_from_schema_org),
            ('open_graph', self._extract_from_open_graph),
            ('readability', self._extract_with_readability),
            ('semantic', self._extract_semantic_content),
            ('heuristic', self._extract_heuristic_content),
        ]

        best_article = None
        best_score = 0

        for method_name, method in extraction_methods:
            try:
                article = await method(soup, source_url)
                if article:
                    # Score the extraction quality
                    score = self._score_extraction(article)
                    if score > best_score:
                        best_article = article
                        best_score = score
                        # Store extraction method for debugging
                        best_article.__dict__['_extraction_method'] = method_name

            except Exception as e:
                logger.debug(f"Extraction method {method_name} failed: {e}")

        return best_article

    async def _extract_from_schema_org(self, soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Extract article data from Schema.org structured data."""

        # Find JSON-LD structured data
        scripts = soup.find_all('script', type='application/ld+json')

        for script in scripts:
            try:
                import json
                data = json.loads(script.string)

                # Handle both single objects and arrays
                if isinstance(data, list):
                    data = data[0] if data else {}

                # Check if it's an article type
                schema_type = data.get('@type', '')
                if schema_type in self.article_schemas:
                    return await self._build_article_from_schema(data, soup, source_url)

            except (json.JSONDecodeError, AttributeError) as e:
                logger.debug(f"Failed to parse JSON-LD: {e}")

        # Try microdata
        article_elements = soup.find_all(attrs={'itemtype': True})
        for element in article_elements:
            itemtype = element.get('itemtype', '')
            if any(schema in itemtype for schema in self.article_schemas):
                return await self._build_article_from_microdata(element, soup, source_url)

        return None

    async def _extract_from_open_graph(self, soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Extract article data from Open Graph meta tags."""

        og_data = {}

        # Extract Open Graph tags
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            if content:
                og_data[property_name] = content

        # Extract Twitter Card tags as fallback
        twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            content = tag.get('content', '')
            if content and name not in og_data:
                og_data[name] = content

        if not og_data:
            return None

        # Build article from Open Graph data
        title = og_data.get('title') or self._extract_title(soup)
        description = og_data.get('description')
        url = og_data.get('url') or source_url
        image = og_data.get('image')

        # Try to extract article content
        content = await self._extract_main_content(soup)

        # Extract other metadata
        author = self._extract_author(soup)
        pub_date = self._extract_publish_date(soup)

        return ArticleData(
            title=title,
            url=url,
            description=description,
            content=content,
            author=author,
            published_date=pub_date,
            source_domain=self._extract_domain(url) if url else None,
            images=[image] if image else [],
            word_count=len(content.split()) if content else None,
        )

    async def _extract_with_readability(self, soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Extract article using python-readability."""

        if not READABILITY_AVAILABLE:
            return None

        try:
            # Get the original HTML
            html_content = str(soup)

            # Extract with readability
            doc = Document(html_content)
            content = doc.summary()
            title = doc.title()

            if not content or len(content.strip()) < self.min_content_length:
                return None

            # Clean the extracted content
            content_soup = BeautifulSoup(content, 'html.parser')
            clean_content = self._clean_content_text(content_soup.get_text())

            # Extract other metadata from original soup
            author = self._extract_author(soup)
            pub_date = self._extract_publish_date(soup)
            images = self._extract_images(soup, source_url) if self.extract_images else []

            return ArticleData(
                title=title,
                url=source_url,
                content=clean_content,
                author=author,
                published_date=pub_date,
                source_domain=self._extract_domain(source_url) if source_url else None,
                images=images,
                word_count=len(clean_content.split()) if clean_content else None,
            )

        except Exception as e:
            logger.debug(f"Readability extraction failed: {e}")
            return None

    async def _extract_semantic_content(self, soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Extract content using semantic HTML elements."""

        # Try to find the main article content
        content_element = None

        # Try semantic elements first
        for selector in self.content_selectors:
            element = soup.select_one(selector)
            if element and self._has_substantial_text(element):
                content_element = element
                break

        if not content_element:
            return None

        # Extract content
        content = self._extract_text_from_element(content_element)
        if not content or len(content.strip()) < self.min_content_length:
            return None

        # Extract metadata
        title = self._extract_title(soup)
        author = self._extract_author(soup)
        pub_date = self._extract_publish_date(soup)
        description = self._extract_description(soup)
        images = self._extract_images(soup, source_url) if self.extract_images else []

        return ArticleData(
            title=title,
            url=source_url,
            description=description,
            content=content,
            author=author,
            published_date=pub_date,
            source_domain=self._extract_domain(source_url) if source_url else None,
            images=images,
            word_count=len(content.split()) if content else None,
        )

    async def _extract_heuristic_content(self, soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Extract content using heuristic methods."""

        # Remove noise elements
        self._remove_noise_elements(soup)

        # Find paragraphs and score them
        paragraphs = soup.find_all('p')
        if not paragraphs:
            return None

        # Score paragraphs based on various factors
        scored_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 20:  # Minimum length threshold
                score = self._score_paragraph(p)
                scored_paragraphs.append((score, text))

        if not scored_paragraphs:
            return None

        # Sort by score and take top paragraphs
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        # Combine top-scoring paragraphs
        content_parts = []
        total_chars = 0

        for score, text in scored_paragraphs:
            if total_chars + len(text) > self.max_content_length:
                break
            content_parts.append(text)
            total_chars += len(text)

        if not content_parts:
            return None

        content = '\n\n'.join(content_parts)

        # Extract metadata
        title = self._extract_title(soup)
        author = self._extract_author(soup)
        pub_date = self._extract_publish_date(soup)
        description = self._extract_description(soup)
        images = self._extract_images(soup, source_url) if self.extract_images else []

        return ArticleData(
            title=title,
            url=source_url,
            description=description,
            content=content,
            author=author,
            published_date=pub_date,
            source_domain=self._extract_domain(source_url) if source_url else None,
            images=images,
            word_count=len(content.split()),
        )

    async def _build_article_from_schema(self, data: Dict[str, Any], soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Build article from Schema.org JSON-LD data."""

        title = data.get('headline') or data.get('name')
        description = data.get('description')
        content = data.get('articleBody')
        url = data.get('url') or source_url

        # Extract author
        author = None
        author_data = data.get('author')
        if author_data:
            if isinstance(author_data, dict):
                author = author_data.get('name')
            elif isinstance(author_data, list) and author_data:
                author = author_data[0].get('name') if isinstance(author_data[0], dict) else str(author_data[0])

        # Extract publish date
        pub_date = None
        date_published = data.get('datePublished')
        if date_published:
            pub_date = self._parse_date(date_published)

        # Extract images
        images = []
        image_data = data.get('image')
        if image_data:
            if isinstance(image_data, str):
                images = [image_data]
            elif isinstance(image_data, list):
                images = [img if isinstance(img, str) else img.get('url') for img in image_data]
            elif isinstance(image_data, dict):
                images = [image_data.get('url')]

        # If content is not in schema, try to extract from HTML
        if not content:
            content = await self._extract_main_content(soup)

        return ArticleData(
            title=title,
            url=url,
            description=description,
            content=content,
            author=author,
            published_date=pub_date,
            source_domain=self._extract_domain(url) if url else None,
            images=[img for img in images if img],
            word_count=len(content.split()) if content else None,
        )

    async def _build_article_from_microdata(self, element, soup: BeautifulSoup, source_url: str = None) -> Optional[ArticleData]:
        """Build article from microdata."""

        # Extract microdata properties
        title = self._get_microdata_property(element, 'headline') or self._get_microdata_property(element, 'name')
        description = self._get_microdata_property(element, 'description')
        content = self._get_microdata_property(element, 'articleBody')
        author = self._get_microdata_property(element, 'author')

        # Extract date
        pub_date = None
        date_str = self._get_microdata_property(element, 'datePublished')
        if date_str:
            pub_date = self._parse_date(date_str)

        # If content not found in microdata, extract from element
        if not content:
            content = self._extract_text_from_element(element)

        return ArticleData(
            title=title,
            url=source_url,
            description=description,
            content=content,
            author=author,
            published_date=pub_date,
            source_domain=self._extract_domain(source_url) if source_url else None,
            word_count=len(content.split()) if content else None,
        )

    async def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main content from HTML."""

        # Try content selectors
        for selector in self.content_selectors:
            element = soup.select_one(selector)
            if element and self._has_substantial_text(element):
                return self._extract_text_from_element(element)

        # Fallback to body content
        body = soup.find('body')
        if body:
            # Remove noise elements
            body_copy = BeautifulSoup(str(body), 'html.parser')
            self._remove_noise_elements(body_copy)
            return self._extract_text_from_element(body_copy)

        return None

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article title."""

        # Try different title sources
        title_selectors = [
            'h1',
            '.article-title',
            '.post-title',
            '.entry-title',
            '[itemprop="headline"]',
            'title'
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title and len(title) > 5:  # Reasonable title length
                    return self._clean_text(title)

        return None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article author."""

        # Try different author sources
        author_selectors = [
            '[itemprop="author"]',
            '[rel="author"]',
            '.author',
            '.byline',
            '.post-author',
            '.article-author',
        ]

        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text().strip()
                if author:
                    return self._clean_text(author)

        return None

    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publish date."""

        # Try different date sources
        date_selectors = [
            '[itemprop="datePublished"]',
            '[property="article:published_time"]',
            'time[datetime]',
            '.publish-date',
            '.post-date',
            '.article-date',
        ]

        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get('content') or element.get_text()
                if date_str:
                    date_obj = self._parse_date(date_str.strip())
                    if date_obj:
                        return date_obj

        return None

    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article description/summary."""

        # Try meta description first
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            desc = meta_desc.get('content', '').strip()
            if desc:
                return self._clean_text(desc)

        # Try other description sources
        desc_selectors = [
            '[itemprop="description"]',
            '.article-summary',
            '.post-excerpt',
            '.excerpt',
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                desc = element.get_text().strip()
                if desc:
                    return self._clean_text(desc)

        return None

    def _extract_images(self, soup: BeautifulSoup, base_url: str = None) -> List[str]:
        """Extract images from HTML."""

        images = []

        # Find all img tags
        img_tags = soup.find_all('img')

        for img in img_tags:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src:
                # Convert relative URLs to absolute
                if base_url and not src.startswith(('http://', 'https://')):
                    src = urljoin(base_url, src)

                # Basic image validation
                if self._is_valid_image_url(src):
                    images.append(src)

        return list(set(images))  # Remove duplicates

    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from HTML element."""

        if not element:
            return ""

        # Remove script and style elements
        for script in element(["script", "style"]):
            script.decompose()

        # Get text and clean it
        text = element.get_text()
        return self._clean_content_text(text)

    def _clean_content_text(self, text: str) -> str:
        """Clean and normalize content text."""

        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove extra newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Clean up common issues
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u200b', '')   # Zero-width space

        return text.strip()

    def _remove_noise_elements(self, soup: BeautifulSoup):
        """Remove noise elements from soup."""

        for selector in self.noise_selectors:
            elements = soup.select(selector)
            for element in elements:
                element.decompose()

        # Remove comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

    def _has_substantial_text(self, element) -> bool:
        """Check if element has substantial text content."""

        if not element:
            return False

        text = element.get_text().strip()
        return len(text) >= self.min_content_length

    def _score_paragraph(self, paragraph) -> float:
        """Score a paragraph for content quality."""

        text = paragraph.get_text().strip()
        score = 0.0

        # Length bonus
        if len(text) > 100:
            score += 1.0
        elif len(text) > 50:
            score += 0.5

        # Word count bonus
        words = text.split()
        if len(words) > 20:
            score += 1.0
        elif len(words) > 10:
            score += 0.5

        # Sentence structure bonus
        sentences = text.split('.')
        if len(sentences) > 2:
            score += 0.5

        # Class name penalties/bonuses
        class_names = paragraph.get('class', [])
        class_str = ' '.join(class_names).lower()

        if any(keyword in class_str for keyword in ['content', 'article', 'text']):
            score += 0.5

        if any(keyword in class_str for keyword in ['ad', 'social', 'share', 'related']):
            score -= 1.0

        return score

    def _score_extraction(self, article: ArticleData) -> float:
        """Score the quality of extracted article data."""

        score = 0.0

        # Title quality
        if article.title:
            if len(article.title) > 10:
                score += 2.0
            else:
                score += 1.0

        # Content quality
        if article.content:
            word_count = len(article.content.split())
            if word_count > 500:
                score += 3.0
            elif word_count > 200:
                score += 2.0
            elif word_count > 100:
                score += 1.0

        # Metadata bonuses
        if article.author:
            score += 1.0
        if article.published_date:
            score += 1.0
        if article.description:
            score += 0.5
        if article.images:
            score += 0.5

        return score

    def _calculate_content_quality_score(self, article: ArticleData) -> float:
        """Calculate overall content quality score."""

        return self._score_extraction(article) / 10.0  # Normalize to 0-1

    def _validate_content_quality(self, article: ArticleData) -> bool:
        """Validate if extracted content meets quality thresholds."""

        if not article.title or len(article.title) < 5:
            return False

        if not article.content or len(article.content) < self.min_content_length:
            return False

        return True

    def _get_microdata_property(self, element, property_name: str) -> Optional[str]:
        """Get microdata property value."""

        prop_element = element.find(attrs={'itemprop': property_name})
        if prop_element:
            return prop_element.get_text().strip()
        return None

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL appears to be a valid image."""

        if not url:
            return False

        # Check for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
        url_lower = url.lower()

        if any(ext in url_lower for ext in image_extensions):
            return True

        # Check for image in URL path
        if 'image' in url_lower or 'img' in url_lower:
            return True

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
                    '%Y-%m-%dT%H:%M:%S%z',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                ]

                for fmt in formats:
                    try:
                        return dt.datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue

        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")

        return None
