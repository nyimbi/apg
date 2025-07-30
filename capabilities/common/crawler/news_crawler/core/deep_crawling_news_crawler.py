"""
Deep Crawling News Crawler with CloudScraper Stealth
====================================================

Advanced news crawler that implements deep site crawling with CloudScraper-based stealth.
Capable of crawling entire news sites with a minimum target of 100 articles per site.

Features:
- CloudScraper integration for stealth and Cloudflare bypass
- Deep site crawling with intelligent URL discovery
- Minimum 100 articles per site target
- Advanced anti-detection mechanisms
- Comprehensive error handling and retry logic
- Real-time progress monitoring
- Database integration for article storage

Author: Lindela Development Team
Version: 6.0.0 (Deep Crawling)
License: MIT
"""

import asyncio
import logging
import random
import time
import ssl
import re
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin, parse_qs, urlunparse
from urllib.robotparser import RobotFileParser
import uuid
import json
from collections import deque, defaultdict

# CloudScraper for stealth
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False
    cloudscraper = None

# Browser automation for infinite scroll
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

# Content extraction libraries
try:
    from newspaper import Article
    import trafilatura
    from bs4 import BeautifulSoup
    import requests
    EXTRACTION_LIBS_AVAILABLE = True
except ImportError:
    EXTRACTION_LIBS_AVAILABLE = False

# Import unified config system
try:
    from ..config.unified_config import (
        NewsConfigurationManager,
        create_news_crawler_config,
        UNIFIED_CONFIG_AVAILABLE
    )
    from ....utils.config import UnifiedCrawlerConfiguration
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    NewsConfigurationManager = None
    create_news_crawler_config = None
    UnifiedCrawlerConfiguration = None

# Import database components
try:
    from ....database import (
        PgSQLManager,
        create_enhanced_postgresql_manager,
        EventExtractionManager
    )
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    PgSQLManager = None
    create_enhanced_postgresql_manager = None
    EventExtractionManager = None

# Import parsers
try:
    from ..parsers import ContentParser, ParsedContent
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False
    ContentParser = None
    ParsedContent = None

# Import bypass components
try:
    from ..bypass import BypassManager, BypassConfig, create_bypass_manager
    BYPASS_AVAILABLE = True
except ImportError:
    BYPASS_AVAILABLE = False
    BypassManager = None
    BypassConfig = None
    create_bypass_manager = None

# Import enhanced crawler components
try:
    from .enhanced_news_crawler import (
        NewsCrawler, StealthConfig, NewsArticle as EnhancedNewsArticle,
        create_news_crawler, create_stealth_crawler
    )
    ENHANCED_CRAWLER_AVAILABLE = True
except ImportError:
    ENHANCED_CRAWLER_AVAILABLE = False
    NewsCrawler = None
    StealthConfig = None
    EnhancedNewsArticle = None
    create_news_crawler = None
    create_stealth_crawler = None

logger = logging.getLogger(__name__)


@dataclass
class CrawlTarget:
    """Configuration for a deep crawl target."""
    url: str
    name: str = ""
    priority: int = 1  # 1=high, 2=medium, 3=low
    max_articles: int = 100  # Minimum target articles
    max_depth: int = 5  # Maximum crawl depth
    max_pages: int = 1000  # Maximum pages to crawl
    follow_external_links: bool = False
    respect_robots_txt: bool = True
    custom_selectors: Dict[str, str] = field(default_factory=dict)
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class NewsArticle:
    """Enhanced news article data structure - unified with enhanced crawler."""
    title: str = ""
    content: str = ""
    url: str = ""
    summary: str = ""
    authors: List[str] = field(default_factory=list)
    publish_date: Optional[datetime] = None
    source_domain: str = ""
    image_url: str = ""
    keywords: List[str] = field(default_factory=list)
    language: str = "en"

    # Crawl metadata (deep crawler specific)
    crawl_depth: int = 0
    discovery_method: str = ""  # 'sitemap', 'link_discovery', 'rss'
    parent_url: str = ""

    # Quality metrics (enhanced)
    quality_score: float = 0.0
    confidence_score: float = 0.0
    word_count: int = 0

    # Geographic information
    location: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    country: str = ""
    region: str = ""

    # Analysis results (from enhanced crawler)
    ml_analysis: Optional[Dict[str, Any]] = None
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing metadata (unified)
    extraction_time: datetime = field(default_factory=datetime.now)
    extraction_method: str = ""
    bypass_method: str = ""
    response_time: float = 0.0


@dataclass
class CrawlSession:
    """Deep crawl session tracking."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target: CrawlTarget = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Progress tracking
    urls_discovered: int = 0
    urls_processed: int = 0
    articles_extracted: int = 0
    target_reached: bool = False

    # Crawl state
    urls_queue: deque = field(default_factory=deque)
    processed_urls: Set[str] = field(default_factory=set)
    failed_urls: Set[str] = field(default_factory=set)
    articles: List[NewsArticle] = field(default_factory=list)

    # Performance metrics
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Stealth metrics
    cloudflare_bypasses: int = 0
    stealth_activations: int = 0
    user_agent_rotations: int = 0
    delay_total: float = 0.0


class CloudScraperStealthCrawler:
    """
    Deep crawling news crawler with CloudScraper stealth capabilities.

    This crawler can discover and extract articles from entire news sites
    with a target minimum of 100 articles per site.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], UnifiedCrawlerConfiguration]] = None):
        """
        Initialize deep crawling news crawler with CloudScraper stealth.

        Args:
            config: Configuration for the crawler
        """
        # Handle configuration
        self.unified_config = None
        self.legacy_config = {}

        if UnifiedCrawlerConfiguration and isinstance(config, UnifiedCrawlerConfiguration):
            self.unified_config = config
            self.legacy_config = self._convert_unified_to_legacy_dict(config)
        elif isinstance(config, dict):
            self.legacy_config = config
        else:
            self.legacy_config = self._get_default_config()

        self.config = self.legacy_config

        # Initialize CloudScraper session
        self.scraper_session = None
        self._initialize_cloudscraper()

        # Initialize enhanced crawler for stealth capabilities
        self.enhanced_crawler = None
        if ENHANCED_CRAWLER_AVAILABLE and self.config.get('enable_enhanced_stealth', True):
            self.enhanced_crawler = NewsCrawler(self.config)
            logger.info("Enhanced stealth crawler integrated")

        # Initialize components
        self.content_parser = None
        self.bypass_manager = None
        self.database_manager = None

        if PARSERS_AVAILABLE:
            self.content_parser = ContentParser(self.config.get('parser_config', {}))

        if BYPASS_AVAILABLE and self.config.get('enable_bypass', True):
            self.bypass_manager = create_bypass_manager(self.config.get('bypass_config', {}))

        if DATABASE_AVAILABLE and self.config.get('enable_database', True):
            self._initialize_database()

        # URL discovery patterns
        self.article_patterns = [
            r'/news/',
            r'/article/',
            r'/story/',
            r'/post/',
            r'/content/',
            r'/blog/',
            r'/press/',
            r'/media/',
            r'/breaking/',
            r'/politics/',
            r'/world/',
            r'/local/',
            r'/sports/',
            r'/business/',
            r'/technology/',
            r'/health/',
            r'/entertainment/'
        ]

        # Exclude patterns
        self.exclude_patterns = [
            r'/search',
            r'/category',
            r'/tag/',
            r'/author/',
            r'/archive/',
            r'/contact',
            r'/about',
            r'/privacy',
            r'/terms',
            r'/login',
            r'/register',
            r'/subscribe',
            r'/newsletter',
            r'/advertisement',
            r'/ads/',
            r'\.css$',
            r'\.js$',
            r'\.jpg$',
            r'\.png$',
            r'\.gif$',
            r'\.pdf$',
            r'\.zip$',
            r'\.exe$'
        ]

        # User agent rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0'
        ]

        # Performance settings
        self.delay_range = (
            self.config.get('min_delay', 2.0),
            self.config.get('max_delay', 5.0)
        )
        self.max_concurrent = self.config.get('max_concurrent', 3)
        self.timeout = self.config.get('timeout', 30)

        logger.info("Deep Crawling News Crawler v6.0.0 initialized with enhanced stealth integration")
        logger.info(f"Enhanced crawler: {ENHANCED_CRAWLER_AVAILABLE and self.config.get('enable_enhanced_stealth', True)}")
        logger.info(f"CloudScraper available: {CLOUDSCRAPER_AVAILABLE}")
        logger.info(f"Target articles per site: {self.config.get('min_articles_per_site', 100)}")

    def _initialize_cloudscraper(self):
        """Initialize CloudScraper session with stealth configuration."""
        if not CLOUDSCRAPER_AVAILABLE:
            logger.warning("CloudScraper not available - falling back to requests")
            return

        try:
            # Create CloudScraper session with anti-detection
            self.scraper_session = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'mobile': False
                },
                delay=random.uniform(*self.delay_range),
                debug=self.config.get('debug', False)
            )

            # Set additional headers for stealth
            self.scraper_session.headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            })

            logger.info("CloudScraper session initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CloudScraper: {e}")
            self.scraper_session = None

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for deep crawling."""
        return {
            'min_articles_per_site': 100,
            'max_depth': 5,
            'max_pages': 1000,
            'min_delay': 2.0,
            'max_delay': 5.0,
            'max_concurrent': 3,
            'timeout': 30,
            'enable_database': True,
            'enable_bypass': True,
            'enable_enhanced_stealth': True,
            'enable_ml_analysis': True,
            'respect_robots_txt': True,
            'follow_redirects': True,
            'extract_images': False,
            'handle_infinite_scroll': False,
            'max_infinite_scrolls': 5,
            'debug': False,
            'bypass_config': {
                'enable_cloudflare_bypass': True,
                'enable_anti_403': True,
                'max_retries': 3,
                'retry_delay': 2.0
            },
            'parser_config': {
                'enable_ml_analysis': True,
                'quality_threshold': 0.3,
                'min_content_length': 100
            }
        }

    def _convert_unified_to_legacy_dict(self, unified_config: UnifiedCrawlerConfiguration) -> Dict[str, Any]:
        """Convert unified configuration to legacy dictionary format."""
        return {
            'min_articles_per_site': 100,  # Fixed target
            'max_depth': 5,
            'max_pages': 1000,
            'min_delay': unified_config.stealth.min_request_delay,
            'max_delay': unified_config.stealth.max_request_delay,
            'max_concurrent': unified_config.performance.max_concurrent_requests,
            'timeout': unified_config.performance.request_timeout,
            'enable_database': True,
            'enable_enhanced_stealth': True,
            'enable_ml_analysis': True,
            'respect_robots_txt': True,
            'follow_redirects': True,
            'extract_images': unified_config.content_extraction.download_images,
            'debug': False
        }

    def _initialize_database(self):
        """Initialize database manager for article storage."""
        try:
            connection_string = self.config.get(
                'database_url',
                'postgresql://postgres:@localhost:5432/lindela'
            )
            self.database_manager = create_enhanced_postgresql_manager(
                connection_string=connection_string
            )
            logger.info("Database manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.database_manager = None

    async def deep_crawl_site(self, target: CrawlTarget) -> CrawlSession:
        """
        Perform deep crawling of a news site with target article count.

        Args:
            target: Crawl target configuration

        Returns:
            CrawlSession with results
        """
        session = CrawlSession(target=target)

        logger.info(f"Starting deep crawl of {target.url}")
        logger.info(f"Target: {target.max_articles} articles, Max depth: {target.max_depth}")

        try:
            # Phase 1: URL Discovery
            await self._discover_urls(session)

            # Phase 2: Content Extraction
            await self._extract_articles(session)

            # Phase 3: Quality Assessment
            await self._assess_quality(session)

            # Phase 4: Database Storage
            if self.database_manager:
                await self._store_articles(session)

            session.end_time = datetime.now()
            session.target_reached = session.articles_extracted >= target.max_articles

            logger.info(f"Deep crawl completed: {session.articles_extracted} articles extracted")
            logger.info(f"Target reached: {session.target_reached}")

        except Exception as e:
            logger.error(f"Deep crawl failed: {e}")
            session.errors.append(str(e))
            session.end_time = datetime.now()

        return session

    async def _discover_urls(self, session: CrawlSession):
        """Discover URLs for article extraction."""
        logger.info("Phase 1: URL Discovery")

        # Add initial URL
        session.urls_queue.append((session.target.url, 0))

        # Try different discovery methods
        await self._discover_from_sitemap(session)
        await self._discover_from_rss(session)
        await self._discover_from_links(session)

        # Handle infinite scroll pages if enabled
        if self.config.get('handle_infinite_scroll', False):
            await self._discover_from_infinite_scroll(session)

        logger.info(f"URL discovery completed: {session.urls_discovered} URLs found")

    async def _discover_from_sitemap(self, session: CrawlSession):
        """Discover URLs from sitemap."""
        try:
            base_url = f"{urlparse(session.target.url).scheme}://{urlparse(session.target.url).netloc}"
            sitemap_urls = [
                f"{base_url}/sitemap.xml",
                f"{base_url}/sitemap_index.xml",
                f"{base_url}/news-sitemap.xml",
                f"{base_url}/sitemaps/news.xml"
            ]

            for sitemap_url in sitemap_urls:
                try:
                    response = await self._make_request(sitemap_url, session)
                    if response and response.status_code == 200:
                        urls = self._extract_urls_from_sitemap(response.text, session.target.url)
                        for url in urls:
                            if self._is_article_url(url):
                                session.urls_queue.append((url, 1))
                                session.urls_discovered += 1

                        logger.info(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
                        break

                except Exception as e:
                    logger.debug(f"Failed to process sitemap {sitemap_url}: {e}")

        except Exception as e:
            logger.warning(f"Sitemap discovery failed: {e}")

    async def _discover_from_rss(self, session: CrawlSession):
        """Discover URLs from RSS feeds."""
        try:
            base_url = f"{urlparse(session.target.url).scheme}://{urlparse(session.target.url).netloc}"
            rss_urls = [
                f"{base_url}/rss",
                f"{base_url}/feed",
                f"{base_url}/rss.xml",
                f"{base_url}/feed.xml",
                f"{base_url}/news/rss",
                f"{base_url}/news/feed"
            ]

            for rss_url in rss_urls:
                try:
                    response = await self._make_request(rss_url, session)
                    if response and response.status_code == 200:
                        urls = self._extract_urls_from_rss(response.text)
                        for url in urls:
                            if self._is_article_url(url):
                                session.urls_queue.append((url, 1))
                                session.urls_discovered += 1

                        logger.info(f"Found {len(urls)} URLs in RSS: {rss_url}")
                        break

                except Exception as e:
                    logger.debug(f"Failed to process RSS {rss_url}: {e}")

        except Exception as e:
            logger.warning(f"RSS discovery failed: {e}")

    async def _discover_from_links(self, session: CrawlSession):
        """Discover URLs by following links."""
        try:
            processed_count = 0
            max_discovery_pages = min(50, session.target.max_pages // 10)  # Limit discovery phase

            while (session.urls_queue and
                   processed_count < max_discovery_pages and
                   session.urls_discovered < session.target.max_pages):

                url, depth = session.urls_queue.popleft()

                if url in session.processed_urls or depth > session.target.max_depth:
                    continue

                session.processed_urls.add(url)
                processed_count += 1

                try:
                    response = await self._make_request(url, session)
                    if response and response.status_code == 200:
                        links = self._extract_links_from_html(response.text, url)

                        for link in links:
                            if (link not in session.processed_urls and
                                self._is_same_domain(link, session.target.url) and
                                self._is_article_url(link)):

                                session.urls_queue.append((link, depth + 1))
                                session.urls_discovered += 1

                    # Add delay between requests
                    await self._apply_delay(session)

                except Exception as e:
                    logger.debug(f"Failed to process discovery URL {url}: {e}")
                    session.failed_urls.add(url)

            logger.info(f"Link discovery completed: {processed_count} pages processed")

        except Exception as e:
            logger.warning(f"Link discovery failed: {e}")

    async def _discover_from_infinite_scroll(self, session: CrawlSession):
        """Discover URLs from infinite scroll pages using Playwright."""
        logger.info("Discovering URLs from infinite scroll pages")

        # Check for infinite scroll indicators on main page
        infinite_scroll_urls = [session.target.url]

        # Also check category/section pages that might have infinite scroll
        base_url = f"{urlparse(session.target.url).scheme}://{urlparse(session.target.url).netloc}"
        potential_scroll_pages = [
            f"{base_url}/news",
            f"{base_url}/latest",
            f"{base_url}/breaking",
            f"{base_url}/world",
            f"{base_url}/politics"
        ]

        for scroll_url in infinite_scroll_urls + potential_scroll_pages:
            try:
                discovered_urls = await self._handle_infinite_scroll(
                    scroll_url,
                    max_scrolls=self.config.get('max_infinite_scrolls', 5)
                )

                # Add discovered URLs to queue
                for url in discovered_urls:
                    if (url not in session.processed_urls and
                        self._is_same_domain(url, session.target.url)):
                        session.urls_queue.append((url, 1))
                        session.urls_discovered += 1

                logger.info(f"Infinite scroll on {scroll_url}: {len(discovered_urls)} URLs found")

                # Add delay between infinite scroll sessions
                await asyncio.sleep(random.uniform(3, 6))

            except Exception as e:
                logger.debug(f"Infinite scroll failed for {scroll_url}: {e}")

    async def _extract_articles(self, session: CrawlSession):
        """Extract articles from discovered URLs."""
        logger.info("Phase 2: Article Extraction")

        # Convert queue to list for article extraction
        article_urls = []
        while session.urls_queue:
            url, depth = session.urls_queue.popleft()
            if self._is_article_url(url):
                article_urls.append((url, depth))

        # Shuffle for better distribution
        random.shuffle(article_urls)

        # Extract articles with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []

        for url, depth in article_urls[:session.target.max_articles * 2]:  # Get extra for filtering
            task = self._extract_single_article(url, depth, session, semaphore)
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Article extraction completed: {session.articles_extracted} articles")

    async def _extract_single_article(self, url: str, depth: int, session: CrawlSession, semaphore: asyncio.Semaphore):
        """Extract a single article with stealth."""
        async with semaphore:
            try:
                # Skip if we've reached target
                if session.articles_extracted >= session.target.max_articles:
                    return

                response = await self._make_request(url, session)
                if not response or response.status_code != 200:
                    return

                # Extract article content
                article = await self._parse_article_content(url, response.text, depth, session)

                if article and self._is_valid_article(article):
                    session.articles.append(article)
                    session.articles_extracted += 1

                    logger.debug(f"Extracted article [{session.articles_extracted}]: {article.title[:50]}...")

                session.urls_processed += 1

                # Apply delay
                await self._apply_delay(session)

            except Exception as e:
                logger.debug(f"Failed to extract article from {url}: {e}")
                session.failed_urls.add(url)

    async def _make_request(self, url: str, session: CrawlSession) -> Optional[Any]:
        """Make HTTP request with integrated bypass capabilities."""
        try:
            # Use bypass manager if available, otherwise use CloudScraper directly
            if self.bypass_manager and BYPASS_AVAILABLE:
                return await self._make_request_with_bypass(url, session)
            else:
                return await self._make_request_with_cloudscraper(url, session)

        except Exception as e:
            logger.debug(f"Request failed for {url}: {e}")
            return None

    async def _make_request_with_bypass(self, url: str, session: CrawlSession) -> Optional[Any]:
        """Make request using the integrated bypass manager."""
        try:
            # Make request through bypass manager
            bypass_result = await self.bypass_manager.bypass_request(url)

            if bypass_result.success:
                # Update session metrics
                if bypass_result.cloudflare_bypassed:
                    session.cloudflare_bypasses += 1
                if bypass_result.user_agent_rotated:
                    session.user_agent_rotations += 1

                # Create response-like object
                class BypassResponse:
                    def __init__(self, content, status_code, headers):
                        self.text = content
                        self.status_code = status_code
                        self.headers = headers

                return BypassResponse(
                    bypass_result.content,
                    bypass_result.status_code,
                    bypass_result.headers
                )
            else:
                logger.debug(f"Bypass manager failed for {url}: {bypass_result.error}")
                return None

        except Exception as e:
            logger.debug(f"Bypass request failed for {url}: {e}")
            # Fallback to CloudScraper
            return await self._make_request_with_cloudscraper(url, session)

    async def _make_request_with_cloudscraper(self, url: str, session: CrawlSession) -> Optional[Any]:
        """Fallback request using CloudScraper directly."""
        try:
            if not self.scraper_session:
                logger.warning("CloudScraper not available, cannot make request")
                return None

            # Rotate user agent occasionally
            if random.random() < 0.1:  # 10% chance
                self.scraper_session.headers['User-Agent'] = random.choice(self.user_agents)
                session.user_agent_rotations += 1

            start_time = time.time()

            # Make request with CloudScraper
            response = self.scraper_session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True
            )

            response_time = time.time() - start_time

            # Check if Cloudflare bypass was used
            if 'cf-ray' in response.headers or 'cloudflare' in str(response.headers).lower():
                session.cloudflare_bypasses += 1

            logger.debug(f"Request to {url}: {response.status_code} ({response_time:.2f}s)")

            return response

        except Exception as e:
            logger.debug(f"CloudScraper request failed for {url}: {e}")
            return None

    async def _parse_article_content(self, url: str, html_content: str, depth: int, session: CrawlSession) -> Optional[NewsArticle]:
        """Parse article content using integrated parser modules."""
        try:
            # Use ContentParser if available, otherwise fallback to built-in
            if self.content_parser and PARSERS_AVAILABLE:
                return await self._parse_with_content_parser(url, html_content, depth, session)
            else:
                return await self._parse_with_builtin_methods(url, html_content, depth, session)

        except Exception as e:
            logger.debug(f"Article parsing failed for {url}: {e}")
            return None

    async def _parse_with_content_parser(self, url: str, html_content: str, depth: int, session: CrawlSession) -> Optional[NewsArticle]:
        """Parse article using the unified ContentParser module."""
        try:
            # Parse content using unified parser
            parsed_result = await self.content_parser.parse_content(url, html_content)

            if not parsed_result.success:
                logger.debug(f"ContentParser failed for {url}")
                return None

            # Create NewsArticle from parsed results
            article = NewsArticle(
                url=url,
                crawl_depth=depth,
                discovery_method="link_discovery",
                source_domain=urlparse(url).netloc
            )

            # Populate from extraction results
            if parsed_result.extraction_result:
                ext = parsed_result.extraction_result
                article.title = ext.title or ""
                article.content = ext.content or ""
                article.summary = ext.summary or ""
                article.authors = ext.authors or []
                article.publish_date = ext.publish_date
                article.image_url = ext.image_url or ""
                article.keywords = ext.keywords or []
                article.word_count = len(article.content.split()) if article.content else 0

            # Add metadata
            if parsed_result.metadata:
                meta = parsed_result.metadata
                if not article.authors and meta.authors:
                    article.authors = meta.authors
                if not article.publish_date and meta.publish_date:
                    article.publish_date = meta.publish_date
                if not article.keywords and meta.keywords:
                    article.keywords = meta.keywords

            # Add ML analysis results
            if parsed_result.ml_analysis:
                article.ml_analysis = parsed_result.ml_analysis.to_dict()

            # Store extraction metadata
            article.extraction_metadata = {
                'extraction_method': 'content_parser',
                'parser_version': parsed_result.parser_version or 'unknown',
                'extraction_time': parsed_result.extraction_time,
                'confidence_score': parsed_result.confidence_score,
                'stealth_used': True
            }

            return article if article.content else None

        except Exception as e:
            logger.debug(f"ContentParser integration failed for {url}: {e}")
            # Fallback to built-in methods
            return await self._parse_with_builtin_methods(url, html_content, depth, session)

    async def _parse_with_builtin_methods(self, url: str, html_content: str, depth: int, session: CrawlSession) -> Optional[NewsArticle]:
        """Fallback parsing using built-in extraction methods."""
        try:
            article = NewsArticle(
                url=url,
                crawl_depth=depth,
                discovery_method="link_discovery",
                source_domain=urlparse(url).netloc
            )

            # Try newspaper3k first
            if EXTRACTION_LIBS_AVAILABLE:
                try:
                    news_article = Article(url)
                    news_article.set_html(html_content)
                    news_article.parse()

                    article.title = news_article.title or ""
                    article.content = news_article.text or ""
                    article.authors = news_article.authors or []
                    article.publish_date = news_article.publish_date
                    article.image_url = news_article.top_image or ""
                    article.keywords = news_article.keywords or []
                    article.summary = news_article.summary or ""

                except Exception as e:
                    logger.debug(f"Newspaper3k parsing failed for {url}: {e}")

            # Fallback to trafilatura
            if not article.content and EXTRACTION_LIBS_AVAILABLE:
                try:
                    extracted = trafilatura.extract(html_content, include_comments=False)
                    if extracted:
                        article.content = extracted

                        # Extract title from HTML if not found
                        if not article.title:
                            soup = BeautifulSoup(html_content, 'html.parser')
                            title_tag = soup.find('title')
                            if title_tag:
                                article.title = title_tag.get_text().strip()

                except Exception as e:
                    logger.debug(f"Trafilatura parsing failed for {url}: {e}")

            # Basic HTML parsing fallback
            if not article.content:
                try:
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Extract title
                    if not article.title:
                        title_tag = soup.find('title')
                        if title_tag:
                            article.title = title_tag.get_text().strip()

                    # Extract content from common article selectors
                    content_selectors = [
                        'article', '.article', '#article',
                        '.content', '#content', '.post-content',
                        '.entry-content', '.article-content'
                    ]

                    for selector in content_selectors:
                        content_elem = soup.select_one(selector)
                        if content_elem:
                            article.content = content_elem.get_text().strip()
                            break

                except Exception as e:
                    logger.debug(f"HTML parsing failed for {url}: {e}")

            # Calculate word count
            if article.content:
                article.word_count = len(article.content.split())

            # Store extraction metadata
            article.extraction_metadata = {
                'extraction_method': 'builtin_fallback',
                'stealth_used': True
            }

            return article if article.content else None

        except Exception as e:
            logger.debug(f"Builtin parsing failed for {url}: {e}")
            return None

    def _is_valid_article(self, article: NewsArticle) -> bool:
        """Check if article meets quality criteria."""
        # Minimum content length
        if article.word_count < 50:
            return False

        # Must have title
        if not article.title or len(article.title.strip()) < 10:
            return False

        # Content quality checks
        content_lower = article.content.lower()

        # Skip if too much boilerplate
        boilerplate_keywords = [
            'cookie policy', 'privacy policy', 'terms of service',
            'subscribe to newsletter', 'follow us on', 'advertisement',
            'sponsored content', 'this website uses cookies'
        ]

        boilerplate_count = sum(1 for keyword in boilerplate_keywords if keyword in content_lower)
        if boilerplate_count > 2:
            return False

        return True

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article."""
        url_lower = url.lower()

        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, url_lower):
                return False

        # Check include patterns
        for pattern in self.article_patterns:
            if pattern in url_lower:
                return True

        # Check for date patterns (common in news URLs)
        date_patterns = [
            r'/20\d{2}/',  # Year
            r'/\d{4}/\d{2}/',  # Year/month
            r'/\d{4}/\d{2}/\d{2}/'  # Year/month/day
        ]

        for pattern in date_patterns:
            if re.search(pattern, url):
                return True

        # Check URL structure
        path = urlparse(url).path
        segments = [s for s in path.split('/') if s]

        # Likely article if has multiple path segments
        if len(segments) >= 2:
            return True

        return False

    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain."""
        try:
            domain1 = urlparse(url1).netloc.lower()
            domain2 = urlparse(url2).netloc.lower()

            # Remove www prefix
            domain1 = domain1.replace('www.', '')
            domain2 = domain2.replace('www.', '')

            return domain1 == domain2
        except:
            return False

    def _extract_urls_from_sitemap(self, xml_content: str, base_url: str) -> List[str]:
        """Extract URLs from sitemap XML."""
        urls = []
        try:
            # Simple regex extraction for URLs
            url_pattern = r'<loc>(.*?)</loc>'
            matches = re.findall(url_pattern, xml_content)

            for match in matches:
                if self._is_same_domain(match, base_url):
                    urls.append(match)
        except Exception as e:
            logger.debug(f"Sitemap parsing error: {e}")

        return urls

    def _extract_urls_from_rss(self, xml_content: str) -> List[str]:
        """Extract URLs from RSS feed."""
        urls = []
        try:
            # Simple regex extraction for RSS links
            link_patterns = [
                r'<link>(.*?)</link>',
                r'<link[^>]*href=["\']([^"\']*)["\']',
                r'<guid[^>]*>(.*?)</guid>'
            ]

            for pattern in link_patterns:
                matches = re.findall(pattern, xml_content)
                for match in matches:
                    if match.startswith('http'):
                        urls.append(match)
        except Exception as e:
            logger.debug(f"RSS parsing error: {e}")

        return urls

    def _extract_links_from_html(self, html_content: str, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        links = []
        try:
            if not EXTRACTION_LIBS_AVAILABLE:
                return links

            soup = BeautifulSoup(html_content, 'html.parser')

            for link in soup.find_all('a', href=True):
                href = link['href']

                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(base_url, href)
                elif not href.startswith('http'):
                    continue

                # Clean up URL
                parsed = urlparse(href)
                clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

                if clean_url not in links:
                    links.append(clean_url)

        except Exception as e:
            logger.debug(f"Link extraction error: {e}")

        return links

    async def _apply_delay(self, session: CrawlSession):
        """Apply random delay between requests."""
        delay = random.uniform(*self.delay_range)
        session.delay_total += delay
        await asyncio.sleep(delay)

    async def _assess_quality(self, session: CrawlSession):
        """Assess quality of extracted articles."""
        logger.info("Phase 3: Quality Assessment")

        for article in session.articles:
            # Calculate quality score based on various factors
            score = 0.0

            # Content length score
            if article.word_count >= 300:
                score += 0.3
            elif article.word_count >= 150:
                score += 0.2
            elif article.word_count >= 50:
                score += 0.1

            # Title quality score
            if article.title and len(article.title) >= 20:
                score += 0.2

            # Author information
            if article.authors:
                score += 0.2

            # Publication date
            if article.publish_date:
                score += 0.1

            # Content structure
            if '\n' in article.content and len(article.content.split('\n')) > 3:
                score += 0.1

            # Image presence
            if article.image_url:
                score += 0.1

            article.quality_score = min(score, 1.0)
            article.confidence_score = score * 0.8  # Slightly lower confidence

        # Sort articles by quality
        session.articles.sort(key=lambda x: x.quality_score, reverse=True)

        logger.info(f"Quality assessment completed. Average score: {sum(a.quality_score for a in session.articles) / len(session.articles):.2f}")

    async def _store_articles(self, session: CrawlSession):
        """Store articles in database."""
        logger.info("Phase 4: Database Storage")

        if not self.database_manager:
            logger.warning("Database not available for storage")
            return

        try:
            stored_count = 0
            for article in session.articles:
                # Convert article to database format
                article_data = {
                    'url': article.url,
                    'title': article.title,
                    'content': article.content,
                    'summary': article.summary,
                    'authors': json.dumps(article.authors),
                    'publish_date': article.publish_date,
                    'source_domain': article.source_domain,
                    'image_url': article.image_url,
                    'keywords': json.dumps(article.keywords),
                    'quality_score': article.quality_score,
                    'word_count': article.word_count,
                    'crawl_depth': article.crawl_depth,
                    'discovery_method': article.discovery_method,
                    'extraction_metadata': json.dumps(article.extraction_metadata),
                    'crawl_session_id': session.session_id,
                    'created_at': datetime.now()
                }

                # Store in database
                try:
                    await self.database_manager.store_data([article_data])
                    stored_count += 1
                except Exception as e:
                    logger.debug(f"Failed to store article {article.url}: {e}")

            logger.info(f"Database storage completed: {stored_count} articles stored")

        except Exception as e:
            logger.error(f"Database storage failed: {e}")

    async def crawl_multiple_sites(self, targets: List[CrawlTarget]) -> List[CrawlSession]:
        """Crawl multiple news sites."""
        logger.info(f"Starting crawl of {len(targets)} sites")

        sessions = []
        for target in targets:
            if target.enabled:
                session = await self.deep_crawl_site(target)
                sessions.append(session)

                # Log results
                logger.info(f"Site: {target.name or target.url}")
                logger.info(f"  Articles: {session.articles_extracted}")
                logger.info(f"  Target reached: {session.target_reached}")
                logger.info(f"  Success rate: {(session.articles_extracted / session.urls_processed * 100):.1f}%")

        return sessions

    async def _handle_infinite_scroll(self, url: str, max_scrolls: int = 5) -> List[str]:
        """
        Handle infinite scroll pages using Playwright.
        Returns list of URLs discovered from infinite scroll.
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available for infinite scroll handling")
            return []

        discovered_urls = []

        try:
            async with async_playwright() as p:
                # Launch browser with stealth
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled'
                    ]
                )

                # Create context with stealth headers
                context = await browser.new_context(
                    user_agent=random.choice(self.user_agents),
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                    }
                )

                page = await context.new_page()

                # Navigate to page
                await page.goto(url, wait_until='networkidle')
                await asyncio.sleep(random.uniform(2, 4))

                # Scroll and collect URLs
                for scroll_count in range(max_scrolls):
                    # Get current URLs
                    links = await page.evaluate("""
                        () => {
                            const links = Array.from(document.querySelectorAll('a[href]'));
                            return links.map(link => link.href);
                        }
                    """)

                    # Filter article URLs
                    for link in links:
                        if self._is_article_url(link) and link not in discovered_urls:
                            discovered_urls.append(link)

                    # Scroll down
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

                    # Wait for new content
                    await page.wait_for_timeout(random.randint(2000, 4000))

                    # Check if more content loaded
                    new_height = await page.evaluate("document.body.scrollHeight")
                    if scroll_count > 0:
                        prev_height = await page.evaluate("document.body.scrollHeight")
                        if new_height == prev_height:
                            logger.info(f"No new content after scroll {scroll_count}, stopping")
                            break

                await browser.close()

        except Exception as e:
            logger.error(f"Infinite scroll handling failed for {url}: {e}")

        logger.info(f"Infinite scroll discovered {len(discovered_urls)} URLs from {url}")
        return discovered_urls

    def _is_article_url(self, url: str) -> bool:
        """Check if URL matches article patterns."""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        # Check article patterns
        for pattern in self.article_patterns:
            if re.search(pattern, path):
                return True

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, path):
                return False

        return False


# Factory functions
async def create_deep_crawler(config: Optional[Dict[str, Any]] = None) -> CloudScraperStealthCrawler:
    """Create a deep crawling news crawler with default stealth configuration."""
    default_config = {
        'min_articles_per_site': 100,
        'max_depth': 5,
        'max_pages': 1000,
        'min_delay': 2.0,
        'max_delay': 5.0,
        'max_concurrent': 3,
        'timeout': 30,
        'enable_database': True,
        'enable_ml_analysis': True,
        'respect_robots_txt': True,
        'debug': False
    }

    if config:
        default_config.update(config)

    return CloudScraperStealthCrawler(default_config)


async def create_production_deep_crawler() -> CloudScraperStealthCrawler:
    """Create a production-ready deep crawler with optimized settings."""
    production_config = {
        'min_articles_per_site': 150,  # Higher target for production
        'max_depth': 6,
        'max_pages': 1500,
        'min_delay': 3.0,  # More conservative delays
        'max_delay': 7.0,
        'max_concurrent': 2,  # Lower concurrency for stealth
        'timeout': 45,
        'enable_database': True,
        'enable_ml_analysis': True,
        'respect_robots_txt': True,
        'debug': False
    }

    return CloudScraperStealthCrawler(production_config)


# Export classes and functions
__all__ = [
    'CloudScraperStealthCrawler',
    'CrawlTarget',
    'NewsArticle',
    'CrawlSession',
    'create_deep_crawler',
    'create_production_deep_crawler'
]
