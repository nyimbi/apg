"""
HTML Parser with Utils Integration
===================================

HTML parsing and content extraction using BeautifulSoup and utils packages.
Integrates with packages_enhanced/utils for enhanced functionality.

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlparse

# Import utils packages
try:
    from ....utils.validation import validate_url, validate_text
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    
    def validate_url(url): return True
    def validate_text(text): return True

try:
    from ....utils.caching import CacheManager, CacheConfig, CacheStrategy, CacheBackend
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    CacheManager = None
    CacheConfig = None

try:
    from ....utils.monitoring import PerformanceMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PerformanceMonitor = None

# Content parsing libraries
try:
    from bs4 import BeautifulSoup, Comment
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    import lxml
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HTMLParseResult:
    """Result of HTML parsing operation."""
    success: bool
    title: str = ""
    content: str = ""
    metadata: Dict[str, Any] = None
    links: List[str] = None
    images: List[str] = None
    errors: List[str] = None
    processing_time: float = 0.0
    parser_used: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.links is None:
            self.links = []
        if self.images is None:
            self.images = []
        if self.errors is None:
            self.errors = []


class HTMLParser:
    """HTML parser with utils integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HTML parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize utils components
        self.cache_manager = None
        self.performance_monitor = None
        
        if CACHING_AVAILABLE and self.config.get('enable_caching', True):
            cache_config = CacheConfig(
                strategy=CacheStrategy.LRU,
                backend=CacheBackend.MEMORY,
                ttl=self.config.get('cache_ttl', 3600),
                max_size=self.config.get('cache_max_size', 1000)
            )
            self.cache_manager = CacheManager(cache_config)
        
        if MONITORING_AVAILABLE and self.config.get('enable_monitoring', True):
            self.performance_monitor = PerformanceMonitor()
        
        # Parser configuration
        self.parser_preference = self.config.get('parser', 'lxml' if LXML_AVAILABLE else 'html.parser')
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.extract_links = self.config.get('extract_links', True)
        self.extract_images = self.config.get('extract_images', True)
        self.clean_html = self.config.get('clean_html', True)
        
        logger.info(f"HTMLParser initialized with parser: {self.parser_preference}")
    
    async def parse(self, html: str, url: str = "", **kwargs) -> HTMLParseResult:
        """
        Parse HTML content.
        
        Args:
            html: HTML content to parse
            url: Source URL for link resolution
            **kwargs: Additional parameters
            
        Returns:
            HTMLParseResult with parsed content
        """
        if not BS4_AVAILABLE:
            return HTMLParseResult(
                success=False,
                errors=["BeautifulSoup not available"]
            )
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"html_parse_{hash(html)}"
        if self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug("Retrieved HTML parse result from cache")
                return cached_result
        
        try:
            # Validate input
            if not html or not isinstance(html, str):
                return HTMLParseResult(
                    success=False,
                    errors=["Invalid HTML content"]
                )
            
            # Parse HTML
            soup = BeautifulSoup(html, self.parser_preference)
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Extract metadata
            metadata = {}
            if self.extract_metadata:
                metadata = self._extract_metadata(soup)
            
            # Extract links
            links = []
            if self.extract_links:
                links = self._extract_links(soup, url)
            
            # Extract images
            images = []
            if self.extract_images:
                images = self._extract_images(soup, url)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = HTMLParseResult(
                success=True,
                title=title,
                content=content,
                metadata=metadata,
                links=links,
                images=images,
                processing_time=processing_time,
                parser_used=self.parser_preference
            )
            
            # Cache result
            if self.cache_manager:
                await self.cache_manager.set(cache_key, result)
            
            # Record performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_timing('html_parse', processing_time * 1000)
            
            return result
            
        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return HTMLParseResult(
                success=False,
                errors=[str(e)],
                processing_time=processing_time,
                parser_used=self.parser_preference
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        try:
            # Try title tag first
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                return title_tag.string.strip()
            
            # Try meta property title
            meta_title = soup.find('meta', property='og:title')
            if meta_title and meta_title.get('content'):
                return meta_title['content'].strip()
            
            # Try h1 tag
            h1_tag = soup.find('h1')
            if h1_tag:
                return h1_tag.get_text(strip=True)
            
            return ""
            
        except Exception as e:
            logger.warning(f"Title extraction failed: {e}")
            return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        try:
            # Remove unwanted elements
            if self.clean_html:
                self._clean_soup(soup)
            
            # Try to find main content areas
            content_selectors = [
                'article', 'main', '.content', '.article', '.post',
                '[role="main"]', '.entry-content', '.post-content'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    return content_element.get_text(separator=' ', strip=True)
            
            # Fallback to body content
            body = soup.find('body')
            if body:
                return body.get_text(separator=' ', strip=True)
            
            # Final fallback to all text
            return soup.get_text(separator=' ', strip=True)
            
        except Exception as e:
            logger.warning(f"Content extraction failed: {e}")
            return ""
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}
        
        try:
            # Meta tags
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name') or tag.get('property') or tag.get('http-equiv')
                content = tag.get('content')
                
                if name and content:
                    metadata[name] = content
            
            # Structured data (JSON-LD)
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            if json_ld_scripts:
                import json
                structured_data = []
                for script in json_ld_scripts:
                    try:
                        data = json.loads(script.string)
                        structured_data.append(data)
                    except:
                        continue
                if structured_data:
                    metadata['structured_data'] = structured_data
            
            # Language
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                metadata['language'] = html_tag['lang']
            
            # Canonical URL
            canonical = soup.find('link', rel='canonical')
            if canonical and canonical.get('href'):
                metadata['canonical_url'] = canonical['href']
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str = "") -> List[str]:
        """Extract links from HTML."""
        links = []
        
        try:
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href and not href.startswith('#'):
                    # Resolve relative URLs
                    if base_url and not href.startswith(('http://', 'https://')):
                        href = urljoin(base_url, href)
                    links.append(href)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_links = []
            for link in links:
                if link not in seen:
                    seen.add(link)
                    unique_links.append(link)
            
            return unique_links
            
        except Exception as e:
            logger.warning(f"Link extraction failed: {e}")
            return []
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str = "") -> List[str]:
        """Extract image URLs from HTML."""
        images = []
        
        try:
            for img in soup.find_all('img', src=True):
                src = img['src']
                if src:
                    # Resolve relative URLs
                    if base_url and not src.startswith(('http://', 'https://', 'data:')):
                        src = urljoin(base_url, src)
                    images.append(src)
            
            # Remove duplicates
            return list(set(images))
            
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
            return []
    
    def _clean_soup(self, soup: BeautifulSoup):
        """Clean soup by removing unwanted elements."""
        try:
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove elements with common ad/nav classes
            unwanted_classes = [
                'advertisement', 'ads', 'navigation', 'nav', 'sidebar',
                'footer', 'header', 'menu', 'social', 'share'
            ]
            
            for class_name in unwanted_classes:
                for element in soup.find_all(class_=re.compile(class_name, re.I)):
                    element.decompose()
                    
        except Exception as e:
            logger.warning(f"HTML cleaning failed: {e}")
    
    async def parse_batch(self, html_list: List[str], urls: List[str] = None) -> List[HTMLParseResult]:
        """Parse multiple HTML documents."""
        urls = urls or [""] * len(html_list)
        results = []
        
        for i, html in enumerate(html_list):
            url = urls[i] if i < len(urls) else ""
            result = await self.parse(html, url)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics."""
        stats = {
            'parser_preference': self.parser_preference,
            'bs4_available': BS4_AVAILABLE,
            'lxml_available': LXML_AVAILABLE,
            'caching_enabled': self.cache_manager is not None,
            'monitoring_enabled': self.performance_monitor is not None
        }
        
        if self.performance_monitor:
            stats['performance_stats'] = self.performance_monitor.get_summary()
        
        return stats


# Utility functions
def quick_parse_html(html: str, url: str = "", **kwargs) -> HTMLParseResult:
    """Quick HTML parsing function."""
    parser = HTMLParser(kwargs)
    import asyncio
    return asyncio.run(parser.parse(html, url))


def extract_text_only(html: str) -> str:
    """Extract only text content from HTML."""
    if not BS4_AVAILABLE:
        return ""
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except:
        return ""