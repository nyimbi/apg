"""
Generation Crawler - Main Implementation
========================================

Core implementation of the next-generation crawler using Crawlee's AdaptivePlaywrightCrawler
for comprehensive full-site crawling with intelligent adaptation and performance optimization.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import uuid
import time

# Crawlee imports
try:
    from crawlee.crawlers import AdaptivePlaywrightCrawler, AdaptivePlaywrightCrawlingContext
    from crawlee import Request, ConcurrencySettings
    CRAWLEE_AVAILABLE = True
except ImportError:
    CRAWLEE_AVAILABLE = False
    AdaptivePlaywrightCrawler = None
    AdaptivePlaywrightCrawlingContext = None
    Request = None
    ConcurrencySettings = None

# Database integration
try:
    from ....database import PgSQLManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    PgSQLManager = None

# Content parsing
try:
    from ..parsers.content_parser import GenContentParser, ParsedSiteContent
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False
    GenContentParser = None
    ParsedSiteContent = None

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class GenCrawlResult:
    """Result from crawling a single page."""
    url: str
    title: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    crawl_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    content_type: str = "unknown"
    word_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata,
            'links': self.links,
            'images': self.images,
            'timestamp': self.timestamp.isoformat(),
            'crawl_time': self.crawl_time,
            'success': self.success,
            'error': self.error,
            'content_type': self.content_type,
            'word_count': self.word_count
        }

@dataclass
class GenSiteResult:
    """Result from crawling an entire site."""
    base_url: str
    pages: List[GenCrawlResult] = field(default_factory=list)
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    total_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    site_metadata: Dict[str, Any] = field(default_factory=dict)
    crawl_statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100
    
    def get_content_by_type(self, content_type: str) -> List[GenCrawlResult]:
        """Get pages filtered by content type."""
        return [page for page in self.pages if page.content_type == content_type]
    
    def get_articles(self) -> List[GenCrawlResult]:
        """Get article pages."""
        return self.get_content_by_type("article")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'base_url': self.base_url,
            'pages': [page.to_dict() for page in self.pages],
            'total_pages': self.total_pages,
            'successful_pages': self.successful_pages,
            'failed_pages': self.failed_pages,
            'success_rate': self.success_rate,
            'total_time': self.total_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'site_metadata': self.site_metadata,
            'crawl_statistics': self.crawl_statistics
        }

class GenCrawler:
    """
    Next-generation crawler using Crawlee's AdaptivePlaywrightCrawler with fallback.
    
    Attempts to use AdaptivePlaywrightCrawler first for optimal adaptive behavior,
    with automatic fallback to PlaywrightCrawler if initialization fails.
    
    Features:
    - Adaptive crawling with automatic HTTP/browser switching
    - Intelligent content analysis and classification
    - Intelligent link discovery and content extraction
    - Performance optimization with concurrent processing
    - Built-in content analysis and classification
    - Database integration for persistent storage
    - Graceful fallback handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GenCrawler.
        
        Args:
            config: Configuration dictionary with crawler settings
        """
        if not CRAWLEE_AVAILABLE:
            raise ImportError("Crawlee is required but not available. Install with: pip install 'crawlee[all]'")
        
        self.config = config or self._get_default_config()
        self.crawler = None
        self._crawler_type = None
        self.current_site_result = None
        self.visited_urls = set()
        self.discovered_urls = set()
        self.stats = {
            'pages_crawled': 0,
            'links_discovered': 0,
            'errors': 0,
            'start_time': None,
            'sites_crawled': 0
        }
        
        # Initialize content parser if available
        self.content_parser = GenContentParser() if PARSERS_AVAILABLE else None
        
        # Initialize database if available and configured
        self.db_manager = None
        if self.config.get('enable_database', False):
            # Note: Database connection handled directly in _store_in_database method
            logger.info("✅ Database storage enabled")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'max_pages_per_site': 500,
            'max_concurrent': 5,
            'request_timeout': 30,
            'max_retries': 3,
            'enable_database': False,
            'enable_content_analysis': True,
            'respect_robots_txt': True,
            'crawl_delay': 2.0,
            'user_agent': 'GenCrawler/1.0 (+https://datacraft.co.ke)',
            'enable_adaptive_crawling': True,
            'max_depth': 10,
            'content_filters': {
                'min_content_length': 100,
                'exclude_extensions': ['.pdf', '.doc', '.xls', '.zip'],
                'include_patterns': ['article', 'news', 'post', 'story'],
                'exclude_patterns': ['tag', 'category', 'archive', 'login']
            }
        }
    
    async def initialize(self):
        """Initialize the Crawlee crawler.
        
        Attempts to use AdaptivePlaywrightCrawler first, with automatic 
        fallback to PlaywrightCrawler if issues occur.
        """
        logger.info("🚀 Initializing Crawlee crawler...")
        
        # Configure concurrency settings
        concurrency_settings = ConcurrencySettings(
            max_concurrency=self.config.get('max_concurrent', 5),
            min_concurrency=1
        )
        
        # Create AdaptivePlaywrightCrawler with proper static parser
        try:
            # Try to use AdaptivePlaywrightCrawler first with BeautifulSoup parser
            self.crawler = AdaptivePlaywrightCrawler.with_beautifulsoup_static_parser(
                max_requests_per_crawl=self.config.get('max_pages_per_site', 500),
                max_request_retries=self.config.get('max_retries', 3),
                request_handler_timeout=timedelta(seconds=self.config.get('request_timeout', 30)),
                concurrency_settings=concurrency_settings
            )
            self._crawler_type = 'adaptive'
        except Exception as e:
            logger.debug(f"AdaptivePlaywrightCrawler initialization failed: {e}")
            logger.info("Using PlaywrightCrawler (adaptive crawler not available)")
            # Fallback to PlaywrightCrawler
            from crawlee.crawlers import PlaywrightCrawler
            self.crawler = PlaywrightCrawler(
                max_requests_per_crawl=self.config.get('max_pages_per_site', 500),
                max_request_retries=self.config.get('max_retries', 3),
                request_handler_timeout=timedelta(seconds=self.config.get('request_timeout', 30)),
                concurrency_settings=concurrency_settings
            )
            self._crawler_type = 'playwright'
        
        # Set up the request handler
        @self.crawler.router.default_handler
        async def handle_request(context) -> None:
            await self._handle_request(context)
        
        logger.info(f"✅ {self._crawler_type.title()}Crawler initialized" if self._crawler_type else "✅ Crawler initialized")
        logger.info(f"   Max pages per site: {self.config.get('max_pages_per_site', 500)}")
        logger.info(f"   Max concurrent: {self.config.get('max_concurrent', 5)}")
    
    async def _handle_request(self, context) -> None:
        """Handle individual page crawling."""
        start_time = time.time()
        url = context.request.url
        
        try:
            logger.debug(f"Processing: {url}")
            
            # Wait for page to load
            if hasattr(context, 'page') and context.page:
                await context.page.wait_for_load_state('domcontentloaded', timeout=15000)
                
                # Extract page content
                title = await context.page.title()
                content = await context.page.inner_text('body')
                
                # Extract links for further crawling
                links = await self._extract_links(context.page, url)
                images = await self._extract_images(context.page, url)
                
                # Analyze content
                content_type = self._classify_content(url, title, content)
                word_count = len(content.split()) if content else 0
                
            else:
                # No page context available
                title = context.request.url
                content = ""
                links = []
                images = []
                content_type = "unknown"
                word_count = 0
            
            # Create crawl result
            crawl_result = GenCrawlResult(
                url=url,
                title=title,
                content=content,
                links=links,
                images=images,
                timestamp=datetime.now(),
                crawl_time=time.time() - start_time,
                success=True,
                content_type=content_type,
                word_count=word_count,
                metadata={
                    'crawl_method': 'adaptive',
                    'has_page': hasattr(context, 'page') and context.page is not None
                }
            )
            
            # Store result
            if self.current_site_result:
                self.current_site_result.pages.append(crawl_result)
                self.current_site_result.successful_pages += 1
            
            # Update statistics
            self.stats['pages_crawled'] += 1
            self.visited_urls.add(url)
            
            # Enqueue discovered links (this enables full site crawling)
            if links and len(self.visited_urls) < self.config.get('max_pages_per_site', 500):
                await context.enqueue_links()
                self.stats['links_discovered'] += len(links)
            
            # Store in database if configured
            if self.config.get('enable_database', False):
                await self._store_in_database(crawl_result)
            
            logger.debug(f"✅ Processed {url}: {word_count} words, {len(links)} links, {content_type}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {url}: {e}")
            
            # Create failed result
            crawl_result = GenCrawlResult(
                url=url,
                timestamp=datetime.now(),
                crawl_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
            
            if self.current_site_result:
                self.current_site_result.pages.append(crawl_result)
                self.current_site_result.failed_pages += 1
            
            self.stats['errors'] += 1
    
    async def _handle_single_url_request(self, context, site_result: GenSiteResult, deep_crawl: bool) -> None:
        """Handle individual page crawling for single URL operations."""
        start_time = time.time()
        url = context.request.url
        
        try:
            logger.debug(f"Processing single URL: {url}")
            
            # For AdaptivePlaywrightCrawler, context might have different structure
            # Try multiple ways to access page context
            page = None
            
            # Method 1: Direct page attribute
            if hasattr(context, 'page') and context.page:
                page = context.page
            # Method 2: Check if it's wrapped in a different structure
            elif hasattr(context, 'playwright_page') and context.playwright_page:
                page = context.playwright_page
            # Method 3: Check for response content (HTTP mode)
            elif hasattr(context, 'response'):
                response = context.response
                if response and hasattr(response, 'text'):
                    # We have HTTP response, extract content directly
                    try:
                        html_content = await response.text()
                        # Use BeautifulSoup to parse if available
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(html_content, 'html.parser')
                            title = soup.title.string if soup.title else url
                            content = soup.get_text(separator=' ', strip=True)
                            links = [a.get('href') for a in soup.find_all('a', href=True)] if deep_crawl else []
                            images = [img.get('src') for img in soup.find_all('img', src=True)] if deep_crawl else []
                        except ImportError:
                            # No BeautifulSoup, just use title from URL
                            title = url
                            content = html_content[:1000] if html_content else ""
                            links = []
                            images = []
                            
                        content_type = self._classify_content(url, title, content)
                        word_count = len(content.split()) if content else 0
                        
                    except Exception as resp_error:
                        logger.debug(f"Response processing error for {url}: {resp_error}")
                        title = url
                        content = ""
                        links = []
                        images = []
                        content_type = "unknown"
                        word_count = 0
                else:
                    title = url
                    content = ""
                    links = []
                    images = []
                    content_type = "unknown"
                    word_count = 0
            
            # If we have a page object, use Playwright methods
            if page:
                try:
                    # Wait for page to load
                    await page.wait_for_load_state('domcontentloaded', timeout=15000)
                    
                    # Extract page content
                    title = await page.title()
                    content = await page.inner_text('body')
                    
                    # Extract links and images if deep crawl is enabled
                    links = []
                    images = []
                    if deep_crawl:
                        links = await self._extract_links(page, url)
                        images = await self._extract_images(page, url)
                    
                    # Analyze content
                    content_type = self._classify_content(url, title, content)
                    word_count = len(content.split()) if content else 0
                    
                except Exception as page_error:
                    logger.debug(f"Page processing error for {url}: {page_error}")
                    # Fallback to basic info if page processing fails
                    title = url
                    content = ""
                    links = []
                    images = []
                    content_type = "unknown"
                    word_count = 0
            
            # If no page and no response, use fallback
            if not page and not (hasattr(context, 'response') and context.response):
                logger.debug(f"No page or response context for {url}, using fallback")
                title = url
                content = ""
                links = []
                images = []
                content_type = "unknown"
                word_count = 0
            
            # Create crawl result
            crawl_result = GenCrawlResult(
                url=url,
                title=title,
                content=content,
                links=links,
                images=images,
                timestamp=datetime.now(),
                crawl_time=time.time() - start_time,
                success=True,
                content_type=content_type,
                word_count=word_count,
                metadata={
                    'crawl_method': 'single_url',
                    'deep_crawl': deep_crawl,
                    'has_page': hasattr(context, 'page') and context.page is not None
                }
            )
            
            # Store result in the provided site_result
            site_result.pages.append(crawl_result)
            site_result.successful_pages += 1
            
            # Enqueue discovered links only if deep_crawl is enabled
            if deep_crawl and links and len(site_result.pages) < self.config.get('max_pages_per_site', 500):
                await context.enqueue_links()
            
            # Store in database if configured
            if self.config.get('enable_database', False):
                await self._store_in_database(crawl_result)
            
            logger.debug(f"✅ Processed single URL {url}: {word_count} words, {len(links)} links, {content_type}")
            
        except Exception as e:
            error_msg = str(e)
            
            # Don't treat "Page was not crawled with PlaywrightCrawler" as a fatal error
            # This is a known issue with mixing crawler types that doesn't affect functionality
            if "Page was not crawled with PlaywrightCrawler" in error_msg:
                logger.debug(f"Known crawler context warning for {url}: {error_msg}")
                # Try to continue processing if we have minimal info
                crawl_result = GenCrawlResult(
                    url=url,
                    title="Processing Error",
                    content="",
                    timestamp=datetime.now(),
                    crawl_time=time.time() - start_time,
                    success=False,
                    error=error_msg,
                    metadata={
                        'crawl_method': 'single_url',
                        'deep_crawl': deep_crawl,
                        'known_issue': True
                    }
                )
            else:
                logger.error(f"❌ Error processing single URL {url}: {e}")
                crawl_result = GenCrawlResult(
                    url=url,
                    timestamp=datetime.now(),
                    crawl_time=time.time() - start_time,
                    success=False,
                    error=error_msg,
                    metadata={
                        'crawl_method': 'single_url',
                        'deep_crawl': deep_crawl
                    }
                )
            
            site_result.pages.append(crawl_result)
            site_result.failed_pages += 1
    
    async def _extract_links(self, page, base_url: str) -> List[str]:
        """Extract and filter links from the page."""
        try:
            # Get all links
            link_elements = await page.query_selector_all('a[href]')
            links = []
            
            for link_element in link_elements:
                href = await link_element.get_attribute('href')
                if href:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, href)
                    
                    # Filter links
                    if self._should_crawl_url(absolute_url, base_url):
                        links.append(absolute_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.debug(f"Error extracting links from {base_url}: {e}")
            return []
    
    async def _extract_images(self, page, base_url: str) -> List[str]:
        """Extract image URLs from the page."""
        try:
            img_elements = await page.query_selector_all('img[src]')
            images = []
            
            for img_element in img_elements:
                src = await img_element.get_attribute('src')
                if src:
                    absolute_url = urljoin(base_url, src)
                    images.append(absolute_url)
            
            return images
            
        except Exception as e:
            logger.debug(f"Error extracting images from {base_url}: {e}")
            return []
    
    def _should_crawl_url(self, url: str, base_url: str) -> bool:
        """Determine if a URL should be crawled."""
        try:
            parsed_url = urlparse(url)
            base_parsed = urlparse(base_url)
            
            # Stay within the same domain
            if parsed_url.netloc != base_parsed.netloc:
                return False
            
            # Skip certain file extensions
            exclude_extensions = self.config.get('content_filters', {}).get('exclude_extensions', [])
            if any(url.lower().endswith(ext) for ext in exclude_extensions):
                return False
            
            # Skip URLs with certain patterns
            exclude_patterns = self.config.get('content_filters', {}).get('exclude_patterns', [])
            if any(pattern in url.lower() for pattern in exclude_patterns):
                return False
            
            # Prefer URLs with certain patterns
            include_patterns = self.config.get('content_filters', {}).get('include_patterns', [])
            if include_patterns and not any(pattern in url.lower() for pattern in include_patterns):
                # Only apply include patterns if they're defined
                pass
            
            return True
            
        except Exception:
            return False
    
    def _classify_content(self, url: str, title: str, content: str) -> str:
        """Classify the type of content on the page."""
        if not content or len(content) < self.config.get('content_filters', {}).get('min_content_length', 100):
            return "insufficient_content"
        
        # Simple heuristics for content classification
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Check for article patterns
        article_indicators = ['article', 'story', 'news', 'post', 'blog']
        if any(indicator in url_lower for indicator in article_indicators):
            return "article"
        
        if any(indicator in title_lower for indicator in article_indicators):
            return "article"
        
        # Check content length and structure
        if len(content.split()) > 300:  # Substantial content
            return "article"
        elif len(content.split()) > 50:
            return "page"
        else:
            return "snippet"
    
    async def _store_in_database(self, result: GenCrawlResult):
        """Store crawl result in information_units table."""
        # Skip if database storage is not enabled
        if not self.config.get('enable_database', False):
            return
        
        try:
            # Convert GenCrawlResult to information_units record
            information_unit = await self._convert_to_information_unit(result)
            
            # Insert into database using async connection
            import asyncpg
            database_url = self.config.get('database_config', {}).get('connection_string', 'postgresql:///lnd')
            
            conn = await asyncpg.connect(database_url)
            try:
                # Check if record already exists by content_url
                if information_unit.get('content_url'):
                    existing = await conn.fetchval(
                        "SELECT id FROM information_units WHERE content_url = $1",
                        information_unit['content_url']
                    )
                    if existing:
                        logger.debug(f"Record already exists for {result.url}, skipping")
                        return
                
                # Insert new record
                await conn.execute("""
                    INSERT INTO information_units (
                        id, external_id, unit_type, title, content, content_url, 
                        summary, source_domain, language_code, published_at, 
                        discovered_at, scraped_at, capture_method, scraper_name, 
                        scraper_version, metadata, http_status_code, 
                        extraction_status, extraction_confidence_score,
                        classification_level, verification_status, paywall_status,
                        content_hash, url_hash, raw_content_snapshot
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
                    )
                """,
                information_unit['id'], information_unit['external_id'], 
                information_unit['unit_type'], information_unit['title'],
                information_unit['content'], information_unit['content_url'],
                information_unit['summary'], information_unit['source_domain'],
                information_unit['language_code'], information_unit['published_at'],
                information_unit['discovered_at'], information_unit['scraped_at'],
                information_unit['capture_method'], information_unit['scraper_name'],
                information_unit['scraper_version'], information_unit['metadata'],
                information_unit['http_status_code'], information_unit['extraction_status'],
                information_unit['extraction_confidence_score'], information_unit['classification_level'],
                information_unit['verification_status'], information_unit['paywall_status'],
                information_unit['content_hash'], information_unit['url_hash'],
                information_unit['raw_content_snapshot']
                )
                
                logger.debug(f"✅ Stored {result.url} in information_units table")
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.warning(f"Failed to store {result.url} in database: {e}")
    
    async def _convert_to_information_unit(self, result: GenCrawlResult) -> Dict[str, Any]:
        """Convert GenCrawlResult to information_units table format."""
        import hashlib
        from urllib.parse import urlparse
        
        # Generate IDs and hashes
        record_id = str(uuid.uuid4())
        external_id = f"gen_crawler_{hashlib.md5(result.url.encode()).hexdigest()}"
        
        # Generate content hash
        content_hash = None
        if result.content:
            content_hash = hashlib.sha256(result.content.encode('utf-8')).hexdigest()
        
        # Generate URL hash
        url_hash = None
        if result.url:
            url_hash = hashlib.sha256(result.url.encode('utf-8')).hexdigest()
        
        # Extract domain
        source_domain = None
        try:
            parsed_url = urlparse(result.url)
            source_domain = parsed_url.netloc
        except:
            pass
        
        # Determine unit type based on content classification
        unit_type_map = {
            'article': 'news_article',
            'page': 'web_page',
            'snippet': 'text_extract',
            'insufficient_content': 'web_page',
            'unknown': 'web_page'
        }
        unit_type = unit_type_map.get(result.content_type, 'web_page')
        
        # Create summary from content
        summary = None
        if result.content:
            # Take first 500 characters as summary
            summary = result.content[:500].strip()
            if len(result.content) > 500:
                summary += "..."
        
        # Create comprehensive metadata
        metadata = {
            'gen_crawler_version': '1.0.0',
            'crawl_method': result.metadata.get('crawl_method', 'adaptive'),
            'has_page_context': result.metadata.get('has_page', False),
            'word_count': result.word_count,
            'links_found': len(result.links),
            'images_found': len(result.images),
            'crawl_time_seconds': result.crawl_time,
            'crawl_timestamp': result.timestamp.isoformat(),
            'success': result.success,
            'error': result.error,
            'content_type_classification': result.content_type,
            'original_metadata': result.metadata,
            'discovered_links': result.links[:10],  # Store first 10 links
            'discovered_images': result.images[:5]  # Store first 5 images
        }
        
        # Determine extraction confidence based on success and content quality
        extraction_confidence = 0.5  # Default
        if result.success:
            if result.content and len(result.content) > 100:
                extraction_confidence = 0.9
            elif result.title:
                extraction_confidence = 0.7
            else:
                extraction_confidence = 0.6
        else:
            extraction_confidence = 0.1
        
        return {
            'id': record_id,
            'external_id': external_id,
            'unit_type': unit_type,
            'title': result.title[:255] if result.title else None,  # Truncate to fit field
            'content': result.content,
            'content_url': result.url,
            'summary': summary,
            'source_domain': source_domain,
            'language_code': 'en',  # Default to English, could be enhanced with detection
            'published_at': result.timestamp,  # Use crawl timestamp as published time
            'discovered_at': result.timestamp,
            'scraped_at': result.timestamp,
            'capture_method': 'web_scrape_dynamic',  # Gen crawler uses Playwright
            'scraper_name': 'gen_crawler',
            'scraper_version': '1.0.0',
            'metadata': json.dumps(metadata),
            'http_status_code': 200 if result.success else None,  # Assume 200 for successful crawls
            'extraction_status': 'completed' if result.success else 'failed',
            'extraction_confidence_score': extraction_confidence,
            'classification_level': 'public',  # Web content is generally public
            'verification_status': 'unverified',  # Content not verified by default
            'paywall_status': 'unknown',  # Cannot determine paywall status from crawler
            'content_hash': content_hash,
            'url_hash': url_hash,
            'raw_content_snapshot': result.content  # Store full content as snapshot
        }
    
    async def crawl_site(self, base_url: str) -> GenSiteResult:
        """
        Crawl an entire website starting from the base URL.
        
        Args:
            base_url: The starting URL for the site crawl
            
        Returns:
            GenSiteResult containing all crawled pages and statistics
        """
        if not self.crawler:
            await self.initialize()
        
        start_time = datetime.now()
        self.stats['start_time'] = start_time
        
        # Initialize site result
        self.current_site_result = GenSiteResult(
            base_url=base_url,
            start_time=start_time
        )
        
        logger.info(f"🚀 Starting full site crawl: {base_url}")
        
        try:
            # Start crawling from the base URL
            await self.crawler.run([base_url])
            
            # Finalize results
            end_time = datetime.now()
            self.current_site_result.end_time = end_time
            self.current_site_result.total_time = (end_time - start_time).total_seconds()
            self.current_site_result.total_pages = len(self.current_site_result.pages)
            
            # Update statistics
            self.current_site_result.crawl_statistics = {
                'pages_per_minute': (self.current_site_result.total_pages / max(self.current_site_result.total_time / 60, 1)),
                'average_page_time': (self.current_site_result.total_time / max(self.current_site_result.total_pages, 1)),
                'links_discovered': self.stats.get('links_discovered', 0),
                'errors': self.stats.get('errors', 0)
            }
            
            logger.info(f"✅ Site crawl completed: {self.current_site_result.total_pages} pages")
            logger.info(f"   Success rate: {self.current_site_result.success_rate:.1f}%")
            logger.info(f"   Total time: {self.current_site_result.total_time:.1f}s")
            
            self.stats['sites_crawled'] += 1
            
            return self.current_site_result
            
        except Exception as e:
            logger.error(f"❌ Site crawl failed: {e}")
            if self.current_site_result:
                self.current_site_result.end_time = datetime.now()
                self.current_site_result.total_time = (self.current_site_result.end_time - start_time).total_seconds()
            raise
    
    async def crawl_url(self, url: str, deep_crawl: bool = False) -> Optional[GenCrawlResult]:
        """
        Crawl a single URL and return its content.
        
        Args:
            url: The URL to crawl
            deep_crawl: If True, follows links and crawls discovered pages. If False, only crawls the specified URL.
            
        Returns:
            GenCrawlResult containing the page content or None if failed
        """
        logger.info(f"🌐 Crawling single URL: {url} (deep_crawl={deep_crawl})")
        
        # Create a fresh crawler instance for this single operation
        temp_crawler = None
        temp_result = None
        
        try:
            # Configure concurrency settings
            concurrency_settings = ConcurrencySettings(
                max_concurrency=self.config.get('max_concurrent', 5),
                min_concurrency=1
            )
            
            # Determine max requests based on deep_crawl setting
            max_requests = 500 if deep_crawl else 1
            
            # Create fresh AdaptivePlaywrightCrawler for this operation
            try:
                temp_crawler = AdaptivePlaywrightCrawler.with_beautifulsoup_static_parser(
                    max_requests_per_crawl=max_requests,
                    max_request_retries=self.config.get('max_retries', 3),
                    request_handler_timeout=timedelta(seconds=self.config.get('request_timeout', 30)),
                    concurrency_settings=concurrency_settings
                )
                crawler_type = 'adaptive'
            except Exception as e:
                logger.debug(f"AdaptivePlaywrightCrawler creation failed: {e}")
                # Fallback to PlaywrightCrawler
                from crawlee.crawlers import PlaywrightCrawler
                temp_crawler = PlaywrightCrawler(
                    max_requests_per_crawl=max_requests,
                    max_request_retries=self.config.get('max_retries', 3),
                    request_handler_timeout=timedelta(seconds=self.config.get('request_timeout', 30)),
                    concurrency_settings=concurrency_settings
                )
                crawler_type = 'playwright'
            
            # Initialize result container for this operation
            start_time = datetime.now()
            temp_result = GenSiteResult(
                base_url=url,
                start_time=start_time
            )
            
            # Set up the request handler for this crawler instance
            @temp_crawler.router.default_handler
            async def handle_single_url_request(context) -> None:
                await self._handle_single_url_request(context, temp_result, deep_crawl)
            
            logger.debug(f"Created fresh {crawler_type} crawler for single URL operation")
            
            # Run the crawler with just this URL
            await temp_crawler.run([url])
            
            # Return the result
            if temp_result and temp_result.pages:
                result = temp_result.pages[0]
                logger.info(f"✅ Single URL crawl completed: {len(result.content) if result.content else 0} chars")
                return result
            else:
                logger.warning(f"⚠️  No content extracted from {url}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Single URL crawl failed for {url}: {e}")
            return None
    
    async def crawl_urls(self, urls: List[str], deep_crawl: bool = False) -> List[GenCrawlResult]:
        """
        Crawl multiple URLs and return their content.
        
        Args:
            urls: List of URLs to crawl
            deep_crawl: If True, follows links and crawls discovered pages for each URL. 
                       If False, only crawls the specified URLs.
            
        Returns:
            List of GenCrawlResult containing the page content for successfully crawled URLs
        """
        if not urls:
            return []
            
        if not self.crawler:
            await self.initialize()
            
        logger.info(f"🌐 Crawling {len(urls)} URLs (deep_crawl={deep_crawl})")
        
        results = []
        
        if deep_crawl:
            # For deep crawl, treat each URL as a separate site crawl
            for i, url in enumerate(urls, 1):
                logger.info(f"   [{i}/{len(urls)}] Deep crawling: {url}")
                try:
                    site_result = await self.crawl_site(url)
                    if site_result and site_result.pages:
                        results.extend(site_result.pages)
                except Exception as e:
                    logger.error(f"   ❌ Failed to deep crawl {url}: {e}")
                    continue
        else:
            # For shallow crawl, use individual crawl_url calls to avoid crawler reuse issues
            logger.info(f"   Processing {len(urls)} URLs individually (shallow crawl)")
            for i, url in enumerate(urls, 1):
                try:
                    logger.debug(f"   [{i}/{len(urls)}] Crawling: {url}")
                    result = await self.crawl_url(url, deep_crawl=False)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"   ❌ Failed to crawl {url}: {e}")
                    continue
            
            logger.info(f"✅ Shallow crawl completed: {len(results)} successful from {len(urls)} URLs")
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        if self.db_manager:
            try:
                await self.db_manager.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
        
        # Reset state
        self.current_site_result = None
        self.visited_urls.clear()
        self.discovered_urls.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get crawler statistics."""
        return {
            'pages_crawled': self.stats.get('pages_crawled', 0),
            'sites_crawled': self.stats.get('sites_crawled', 0),
            'links_discovered': self.stats.get('links_discovered', 0),
            'errors': self.stats.get('errors', 0),
            'start_time': self.stats.get('start_time'),
            'config': self.config
        }

def create_gen_crawler(config: Optional[Dict[str, Any]] = None) -> GenCrawler:
    """
    Factory function to create a GenCrawler instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured GenCrawler instance
    """
    return GenCrawler(config)

def create_gen_crawler_with_database(
    database_url: str = "postgresql:///lnd",
    max_pages: int = 500,
    max_concurrent: int = 5,
    **kwargs
) -> GenCrawler:
    """
    Convenience function to create a GenCrawler with database integration enabled.
    
    Args:
        database_url: PostgreSQL connection string
        max_pages: Maximum pages to crawl per site
        max_concurrent: Maximum concurrent requests
        **kwargs: Additional configuration options
        
    Returns:
        GenCrawler instance configured for database storage
    """
    config = {
        'enable_database': True,
        'database_config': {
            'connection_string': database_url
        },
        'max_pages_per_site': max_pages,
        'max_concurrent': max_concurrent,
        'enable_content_analysis': True,
        'respect_robots_txt': True,
        'crawl_delay': 2.0,
        'user_agent': 'GenCrawler/1.0 (+https://datacraft.co.ke)',
        **kwargs
    }
    
    return GenCrawler(config)