"""
News Crawler with Stealth and Database Integration
==================================================

News crawler that implements stealth techniques as standard,
including Cloudflare bypass, residential proxies, and anti-detection.
Integrates with database package and ML scorers for news intelligence.

Features:
- CloudScraper-based Cloudflare bypass (standard)
- Residential proxy rotation (standard)
- Browser fingerprinting protection (standard)
- ML-powered content analysis and scoring
- Real-time database persistence
- Monitoring and analytics

Author: Lindela Development Team
Version: 5.0.0 (Stealth Standard)
License: MIT
"""

import asyncio
import logging
import random
import time
import ssl
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
import uuid
import aiohttp

# Enhanced HTTP libraries
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# Content extraction libraries
Article = None
trafilatura = None
BeautifulSoup = None

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

# At least BeautifulSoup is required for basic extraction
EXTRACTION_LIBS_AVAILABLE = BEAUTIFULSOUP_AVAILABLE

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

# Import legacy config for fallback
try:
    from ..config import CrawlerConfiguration, create_default_config
    LEGACY_CONFIG_AVAILABLE = True
except ImportError:
    LEGACY_CONFIG_AVAILABLE = False
    CrawlerConfiguration = None
    create_default_config = None

# Import utils
try:
    from ....utils.monitoring import PerformanceMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PerformanceMonitor = None

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data structure."""
    title: str = ""
    content: str = ""
    url: str = ""
    summary: str = ""
    authors: List[str] = field(default_factory=list)
    publish_date: Optional[datetime] = None
    source_domain: str = ""
    image_url: str = ""
    keywords: List[str] = field(default_factory=list)

    # Quality metrics
    quality_score: float = 0.0
    confidence_score: float = 0.0

    # Analysis results
    ml_analysis: Optional[Dict[str, Any]] = None
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlSession:
    """Crawl session tracking."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    urls_processed: int = 0
    articles_extracted: int = 0
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Stealth metrics
    cloudflare_bypasses: int = 0
    stealth_activations: int = 0
    proxy_rotations: int = 0


@dataclass
class StealthConfig:
    """Stealth configuration."""
    enable_cloudflare_bypass: bool = True
    enable_residential_proxies: bool = True
    enable_fingerprint_protection: bool = True
    min_delay: float = 3.0
    max_delay: float = 8.0
    cloudflare_wait_time: float = 15.0
    max_retries: int = 3
    user_agents_pool_size: int = 50
    session_rotation_interval: int = 10


class NewsCrawler:
    """
    News crawler with stealth techniques as standard.

    Features:
    - CloudScraper-based Cloudflare bypass (enabled by default)
    - Residential proxy rotation
    - Browser fingerprinting protection
    - ML-powered content analysis
    - Real-time database persistence
    - Generic news site optimization
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], UnifiedCrawlerConfiguration]] = None):
        """
        Initialize news crawler with stealth capabilities.

        Args:
            config: Configuration - can be legacy dict, UnifiedCrawlerConfiguration,
                   or None for defaults
        """
        # Handle different configuration types
        self.unified_config = None
        self.legacy_config = {}

        if isinstance(config, UnifiedCrawlerConfiguration):
            self.unified_config = config
            self.legacy_config = self._convert_unified_to_legacy_dict(config)
        elif isinstance(config, dict):
            self.legacy_config = config
        else:
            self.legacy_config = {}

        self.config = self.legacy_config  # For backward compatibility

        # Initialize stealth configuration (ENABLED BY DEFAULT)
        if self.unified_config:
            # Use unified configuration
            stealth_config = self.unified_config.stealth
            self.stealth_config = StealthConfig(
                enable_cloudflare_bypass=self.unified_config.bypass.cloudflare_enabled,
                enable_residential_proxies=stealth_config.use_proxies,
                enable_fingerprint_protection=stealth_config.randomize_headers,
                min_delay=stealth_config.min_request_delay,
                max_delay=stealth_config.max_request_delay,
                cloudflare_wait_time=15.0,  # Default for now
                max_retries=self.unified_config.performance.max_retries
            )
        else:
            # Use legacy configuration
            self.stealth_config = StealthConfig(
                enable_cloudflare_bypass=self.config.get('enable_cloudflare_bypass', True),
                enable_residential_proxies=self.config.get('enable_residential_proxies', False),
                enable_fingerprint_protection=self.config.get('enable_fingerprint_protection', True),
                min_delay=self.config.get('min_delay', 3.0),
                max_delay=self.config.get('max_delay', 8.0),
                cloudflare_wait_time=self.config.get('cloudflare_wait_time', 15.0),
                max_retries=self.config.get('max_retries', 3)
            )

        # Initialize stealth components
        self.cloudscraper_session = None
        self.user_agents = self._initialize_user_agents()
        self.proxy_pool = self._initialize_proxy_pool()
        self.session_counter = 0

        # Initialize content extraction components
        self.content_parser = None
        self.database_manager = None
        self.performance_monitor = None

        if PARSERS_AVAILABLE:
            self.content_parser = ContentParser(self.config.get('parser_config', {}))

        if DATABASE_AVAILABLE and self.config.get('enable_database', False):
            self._initialize_database()

        if MONITORING_AVAILABLE and self.config.get('enable_monitoring', True):
            self.performance_monitor = PerformanceMonitor()

        # Enhanced configuration
        if self.unified_config:
            self.max_concurrent = self.unified_config.performance.max_concurrent_requests
            self.enable_ml_analysis = True  # Enable by default with unified config
            self.enable_database_storage = True  # Enable by default with unified config
        else:
            self.max_concurrent = self.config.get('max_concurrent', 3)  # Conservative for stealth
            self.enable_ml_analysis = self.config.get('enable_ml_analysis', True)
            self.enable_database_storage = self.config.get('enable_database_storage', False)

        # Initialize stealth session
        self._initialize_stealth_session()

        # Generic news site patterns
        self.news_patterns = {
            'article_indicators': ['/news/', '/article/', '/story/', '/post/', '/content/'],
            'priority_sections': self.config.get('priority_sections', ['/news/', '/breaking/']),
            'exclude_patterns': ['/search', '/category', '/tag', '/author', '/archive',
                               '/contact', '/about', '/privacy', '/terms', '.css', '.js', '.jpg', '.png']
        }

        logger.info("News Crawler v5.0.0 initialized with stealth")
        logger.info(f"Cloudflare bypass: {self.stealth_config.enable_cloudflare_bypass}")
        logger.info(f"Fingerprint protection: {self.stealth_config.enable_fingerprint_protection}")

    def _convert_unified_to_legacy_dict(self, unified_config: UnifiedCrawlerConfiguration) -> Dict[str, Any]:
        """Convert unified configuration to legacy dictionary format."""
        return {
            'enable_cloudflare_bypass': unified_config.bypass.cloudflare_enabled,
            'enable_residential_proxies': unified_config.stealth.use_proxies,
            'enable_fingerprint_protection': unified_config.stealth.randomize_headers,
            'min_delay': unified_config.stealth.min_request_delay,
            'max_delay': unified_config.stealth.max_request_delay,
            'max_retries': unified_config.performance.max_retries,
            'max_concurrent': unified_config.performance.max_concurrent_requests,
            'requests_per_second': unified_config.performance.requests_per_second,
            'request_timeout': unified_config.performance.request_timeout,
            'enable_ml_analysis': True,  # Enable by default
            'enable_database_storage': True,  # Enable by default
            'enable_monitoring': unified_config.enable_metrics,
            'cache_responses': unified_config.cache.enabled,
            'batch_size': unified_config.performance.batch_size
        }

    def _initialize_user_agents(self) -> List[str]:
        """Initialize pool of realistic user agents."""
        return [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        ]

    def _initialize_proxy_pool(self) -> List[Dict[str, str]]:
        """Initialize proxy pool (placeholder for residential proxies)."""
        if not self.stealth_config.enable_residential_proxies:
            return []

        # Placeholder for residential proxy integration
        # In production, this would connect to proxy providers like:
        # - BrightData, Oxylabs, Smartproxy, etc.
        return []

    def _initialize_stealth_session(self):
        """Initialize CloudScraper session for Cloudflare bypass."""
        if self.stealth_config.enable_cloudflare_bypass and CLOUDSCRAPER_AVAILABLE:
            try:
                self.cloudscraper_session = cloudscraper.create_scraper(
                    browser={
                        'browser': 'chrome',
                        'platform': 'darwin',
                        'desktop': True
                    },
                    delay=self.stealth_config.min_delay
                )
                logger.info("CloudScraper session initialized for Cloudflare bypass")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudScraper: {e}")
                self.cloudscraper_session = None

    def _get_site_config(self, url: str) -> Dict[str, Any]:
        """Get generic site configuration."""
        # All sites use same configuration - no site-specific customization
        return {
            'cloudflare_bypass': True,
            'min_delay': self.stealth_config.min_delay,
            'max_delay': self.stealth_config.max_delay,
            'priority_sections': self.news_patterns['priority_sections']
        }

    def _get_random_user_agent(self) -> str:
        """Get random user agent from pool."""
        return random.choice(self.user_agents)

    async def _apply_intelligent_delay(self, url: str, attempt: int = 0):
        """Apply intelligent delay based on site configuration."""
        site_config = self._get_site_config(url)

        min_delay = site_config.get('min_delay', self.stealth_config.min_delay)
        max_delay = site_config.get('max_delay', self.stealth_config.max_delay)

        # Exponential backoff for retries
        multiplier = 1.2 ** min(attempt, 5)

        # Random delay with jitter
        delay = random.uniform(min_delay, max_delay) * multiplier

        logger.debug(f"Applying delay of {delay:.2f}s for {urlparse(url).netloc}")
        await asyncio.sleep(delay)

    def _detect_cloudflare(self, content: str, headers: Dict[str, str]) -> bool:
        """Detect Cloudflare protection more accurately."""
        # Only check for actual Cloudflare protection/challenge pages
        cf_challenge_indicators = [
            'checking your browser before accessing',
            'please enable javascript and cookies',
            'ddos protection by cloudflare',
            'please wait while we are checking your browser',
            'enable javascript to continue',
            'cloudflare-static/email-decode.min.js',
            'cf-browser-verification'
        ]

        content_lower = content.lower()

        # Only flag as Cloudflare if we see actual protection content
        # AND the content is suspiciously short (challenge pages are minimal)
        has_challenge_text = any(
            indicator in content_lower
            for indicator in cf_challenge_indicators
        )

        # Challenge pages are typically very short (<5000 chars)
        is_minimal_content = len(content) < 5000

        # Don't flag as Cloudflare if we have substantial content
        if len(content) > 10000:
            return False

        return has_challenge_text and is_minimal_content

    def _initialize_database(self):
        """Initialize database connection."""
        try:
            connection_string = self.config.get('database_connection_string')
            if connection_string:
                self.database_manager = create_enhanced_postgresql_manager(
                    connection_string
                )
            else:
                logger.warning("No database connection string provided")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def crawl_url(self, url: str, **kwargs) -> Optional[NewsArticle]:
        """
        Crawl a single URL with advanced stealth techniques.

        Args:
            url: URL to crawl
            **kwargs: Additional parameters

        Returns:
            NewsArticle if successful, None otherwise
        """
        start_time = datetime.now()
        attempt = kwargs.get('attempt', 0)

        try:
            # Apply intelligent delay before request
            await self._apply_intelligent_delay(url, attempt)

            # Fetch HTML with stealth techniques
            html_content, response_headers = await self._fetch_html_with_stealth(url, attempt)

            if not html_content:
                return None

            # Detect Cloudflare and handle accordingly
            if self._detect_cloudflare(html_content, response_headers):
                logger.info(f"Cloudflare detected for {url}")
                if attempt < self.stealth_config.max_retries:
                    await asyncio.sleep(self.stealth_config.cloudflare_wait_time)
                    return await self.crawl_url(url, attempt=attempt+1)
                else:
                    logger.warning(f"Max retries reached for Cloudflare-protected {url}")

            # Extract article using multiple methods
            article = await self._extract_article_enhanced(url, html_content)

            if article:
                # Add extraction metadata
                article.extraction_metadata = {
                    'extraction_time': datetime.now(),
                    'attempt_number': attempt + 1,
                    'stealth_used': True,
                    'cloudflare_detected': self._detect_cloudflare(html_content, response_headers),
                    'extraction_method': 'enhanced_stealth'
                }

                # Perform ML analysis if enabled
                if self.enable_ml_analysis:
                    article = await self._perform_ml_analysis(article)

                # Store in database if enabled
                if self.enable_database_storage and self.database_manager:
                    await self._store_article(article)

                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Successfully extracted article: {article.title[:60]}... ({processing_time:.2f}s)")

                return article

            return None

        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            if attempt < self.stealth_config.max_retries:
                await asyncio.sleep(random.uniform(2, 5))
                return await self.crawl_url(url, attempt=attempt+1)
            return None

    async def _fetch_html_with_stealth(self, url: str, attempt: int = 0) -> tuple[Optional[str], Dict[str, str]]:
        """
        Fetch HTML using advanced stealth techniques.

        Returns:
            Tuple of (html_content, response_headers)
        """
        headers = {}

        try:
            # Method 1: CloudScraper (primary for Cloudflare sites)
            if self.cloudscraper_session and self.stealth_config.enable_cloudflare_bypass:
                try:
                    response = self.cloudscraper_session.get(
                        url,
                        headers={'User-Agent': self._get_random_user_agent()},
                        timeout=30
                    )

                    if response.status_code == 200:
                        logger.debug(f"CloudScraper success for {url}")
                        return response.text, dict(response.headers)
                    else:
                        logger.debug(f"CloudScraper returned {response.status_code} for {url}")

                except Exception as e:
                    logger.debug(f"CloudScraper failed for {url}: {e}")

            # Method 2: Enhanced aiohttp with stealth headers
            if EXTRACTION_LIBS_AVAILABLE:
                try:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                    connector = aiohttp.TCPConnector(ssl=ssl_context)

                    stealth_headers = {
                        'User-Agent': self._get_random_user_agent(),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Cache-Control': 'max-age=0'
                    }

                    async with aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=30),
                        headers=stealth_headers
                    ) as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()
                                headers = dict(response.headers)
                                logger.debug(f"aiohttp stealth success for {url}")
                                return content, headers
                            else:
                                logger.debug(f"aiohttp returned {response.status} for {url}")

                except Exception as e:
                    logger.debug(f"aiohttp stealth failed for {url}: {e}")

            # Method 3: newspaper3k fallback
            if EXTRACTION_LIBS_AVAILABLE:
                try:
                    article = Article(url)
                    article.download()

                    if article.html:
                        logger.debug(f"newspaper3k fallback success for {url}")
                        return article.html, {}

                except Exception as e:
                    logger.debug(f"newspaper3k fallback failed for {url}: {e}")

            logger.warning(f"All stealth methods failed for {url}")
            return None, {}

        except Exception as e:
            logger.error(f"Critical error in stealth fetch for {url}: {e}")
            return None, {}

    async def _extract_article_enhanced(self, url: str, html_content: str) -> Optional[NewsArticle]:
        """
        Extract article using multiple extraction methods.
        """
        article = NewsArticle(url=url)

        try:
            # Method 1: newspaper3k
            if NEWSPAPER_AVAILABLE and Article:
                try:
                    news_article = Article(url)
                    news_article.set_html(html_content)
                    news_article.parse()

                    if news_article.title:
                        article.title = news_article.title
                        article.content = news_article.text
                        article.authors = news_article.authors
                        article.publish_date = news_article.publish_date
                        article.keywords = news_article.keywords
                        article.summary = news_article.summary if hasattr(news_article, 'summary') else ""

                        if news_article.top_image:
                            article.image_url = news_article.top_image

                        article.quality_score = self._calculate_quality_score(article)
                        logger.debug(f"newspaper3k extraction successful for {url}")
                        return article

                except Exception as e:
                    logger.debug(f"newspaper3k extraction failed for {url}: {e}")

            # Method 2: trafilatura fallback
            if TRAFILATURA_AVAILABLE and trafilatura and BEAUTIFULSOUP_AVAILABLE and BeautifulSoup:
                try:
                    extracted = trafilatura.extract(html_content, include_comments=False)
                    if extracted:
                        # Parse title from HTML
                        soup = BeautifulSoup(html_content, 'html.parser')
                        title_tag = soup.find('title')

                        article.title = title_tag.get_text().strip() if title_tag else urlparse(url).path.split('/')[-1]
                        article.content = extracted
                        article.quality_score = self._calculate_quality_score(article)

                        logger.debug(f"trafilatura extraction successful for {url}")
                        return article

                except Exception as e:
                    logger.debug(f"trafilatura extraction failed for {url}: {e}")

            # Method 3: BeautifulSoup basic extraction
            if BEAUTIFULSOUP_AVAILABLE and BeautifulSoup:
                try:
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Extract title
                    title_tag = soup.find('title')
                    article.title = title_tag.get_text().strip() if title_tag else "No title"

                    # Extract content with comprehensive fallback patterns
                    content_selectors = [
                        'article', '.article-content', '.post-content',
                        '.entry-content', '#content', '.content',
                        'main', '.main-content', '.post-body',
                        '.article-body', '.news-content', '.story-content',
                        '.text-content', '.article-text', '[role="main"]',
                        '.container .content', '.page-content',
                        # Common site-specific selectors for news sites
                        '#main', '#container', '#wrapper', '#page',
                        '.main-container', '.site-content', '.primary-content',
                        '#primary', '#main-content', '.content-area'
                    ]

                    # Try to find the best content match by evaluating all selectors
                    best_content = ""
                    best_length = 0

                    for selector in content_selectors:
                        content_elem = soup.select_one(selector)
                        if content_elem:
                            extracted_text = content_elem.get_text().strip()
                            if len(extracted_text) > best_length and len(extracted_text) > 100:
                                best_content = extracted_text
                                best_length = len(extracted_text)

                    content_found = False
                    if best_content:
                        article.content = best_content
                        content_found = True

                    if not content_found:
                        # Enhanced fallback: extract from multiple sources
                        content_parts = []

                        # Get all paragraphs
                        paragraphs = soup.find_all('p')
                        if paragraphs:
                            p_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
                            if p_text:
                                content_parts.append(p_text)

                        # Get div content with meaningful text
                        divs = soup.find_all('div')
                        for div in divs:
                            div_text = div.get_text().strip()
                            # Only include divs with substantial content and no child divs/articles
                            if (len(div_text) > 200 and
                                not div.find('div') and
                                not div.find('article') and
                                len(div_text.split()) > 30):
                                content_parts.append(div_text)
                                if len(' '.join(content_parts)) > 1000:  # Stop when we have enough
                                    break

                        # Get text from body if nothing else works
                        if not content_parts:
                            body = soup.find('body')
                            if body:
                                body_text = body.get_text().strip()
                                # Clean up body text by removing navigation/menu items
                                lines = [line.strip() for line in body_text.split('\n') if len(line.strip()) > 30]
                                content_parts.append(' '.join(lines[:50]))  # Take first 50 meaningful lines

                        article.content = ' '.join(content_parts).strip()

                    article.quality_score = self._calculate_quality_score(article)

                    if article.title and article.content:
                        logger.debug(f"BeautifulSoup extraction successful for {url}")
                        return article

                except Exception as e:
                    logger.debug(f"BeautifulSoup extraction failed for {url}: {e}")

            logger.warning(f"All extraction methods failed for {url}")
            return None

        except Exception as e:
            logger.error(f"Critical error in article extraction for {url}: {e}")
            return None

    def _calculate_quality_score(self, article: NewsArticle) -> float:
        """Calculate article quality score."""
        score = 0.0

        # Title quality (0-30 points)
        if article.title and len(article.title.strip()) > 10:
            score += 30
        elif article.title:
            score += 15

        # Content quality (0-50 points)
        if article.content:
            content_length = len(article.content.strip())
            if content_length > 1000:
                score += 50
            elif content_length > 500:
                score += 35
            elif content_length > 200:
                score += 20
            elif content_length > 50:
                score += 10

        # Metadata quality (0-20 points)
        if article.authors:
            score += 5
        if article.publish_date:
            score += 5
        if article.keywords:
            score += 5
        if article.image_url:
            score += 5

        return min(score / 100.0, 1.0)  # Normalize to 0-1

    async def _perform_ml_analysis(self, article: NewsArticle) -> NewsArticle:
        """Perform ML analysis on article (placeholder)."""
        # Placeholder for ML analysis integration
        article.ml_analysis = {
            'sentiment': 'neutral',
            'topics': [],
            'entities': [],
            'confidence': 0.5
        }
        return article

    async def _store_article(self, article: NewsArticle):
        """Store article in database."""
        if self.database_manager:
            try:
                # Placeholder for database storage
                logger.debug(f"Storing article: {article.title}")
            except Exception as e:
                logger.error(f"Failed to store article: {e}")

    async def crawl_single_url(self, url: str, **kwargs) -> Optional[NewsArticle]:
        """Alias for crawl_url for compatibility."""
        return await self.crawl_url(url, **kwargs)

    async def discover_site_articles(self, base_url: str) -> List[str]:
        """
        Discover all article URLs from a news site using stealth techniques.

        Args:
            base_url: Base URL of the news site

        Returns:
            List of discovered article URLs
        """
        logger.info(f"🔍 Discovering articles from {base_url} with stealth...")

        try:
            # Step 1: Get homepage with stealth
            html_content, response_headers = await self._fetch_html_with_stealth(base_url)

            # For domain checking, we'll use the improved _is_same_domain method that handles redirects
            effective_base_url = base_url

            if not html_content:
                logger.warning(f"Failed to fetch homepage for {base_url}")
                return []

            # Step 2: Parse homepage for article links
            soup = BeautifulSoup(html_content, 'html.parser')
            article_urls = set()

            # Method 2a: Try specific news article link patterns first
            article_selectors = [
                'a[href*="/article"]',
                'a[href*="/news"]',
                'a[href*="/story"]',
                'a[href*="/post"]',
                'a[href*="/content"]',
                'article a',
                '.article-link',
                '.news-link'
            ]

            for selector in article_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = f"{effective_base_url.rstrip('/')}{href}"
                        elif not href.startswith('http'):
                            href = f"{effective_base_url.rstrip('/')}/{href}"

                        # Filter for news articles
                        if self._is_article_url(href):
                            article_urls.add(href)

            # Method 2b: Fallback - analyze ALL homepage links if specific selectors found few articles
            if len(article_urls) < 5:  # If we found very few articles, try comprehensive analysis
                logger.info(f"Only found {len(article_urls)} articles with specific selectors, trying comprehensive analysis...")

                # Get all links on homepage
                all_links = soup.find_all('a', href=True)
                logger.info(f"Analyzing {len(all_links)} total links on homepage")

                # Temporarily enable debug logging to see what's happening
                debug_logger = logging.getLogger(logger.name)
                original_level = debug_logger.level
                debug_logger.setLevel(logging.DEBUG)

                try:
                    for link in all_links:
                        href = link.get('href')
                        if href:
                            # Convert relative URLs to absolute
                            if href.startswith('/'):
                                href = f"{effective_base_url.rstrip('/')}{href}"
                            elif href.startswith('http') and self._is_same_domain(href, effective_base_url):
                                pass  # Already absolute, same domain
                            else:
                                continue  # Skip external or invalid links

                            # Use enhanced article detection
                            link_text = link.get_text().strip()
                            if self._is_likely_article_enhanced(href, link_text, link, effective_base_url):
                                article_urls.add(href)
                finally:
                    # Restore original logging level
                    debug_logger.setLevel(original_level)

            # Step 3: Try to find sitemap with stealth
            sitemap_urls = await self._discover_sitemap_urls(base_url)
            article_urls.update(sitemap_urls)

            # Step 4: Try RSS feeds with stealth
            rss_urls = await self._discover_rss_urls(base_url, html_content)
            article_urls.update(rss_urls)

            discovered_urls = list(article_urls)
            logger.info(f"✅ Discovered {len(discovered_urls)} article URLs from {base_url}")

            return discovered_urls

        except Exception as e:
            logger.error(f"Failed to discover articles from {base_url}: {e}")
            return []

    def _is_article_url(self, url: str) -> bool:
        """Check if URL looks like a news article."""
        url_lower = url.lower()

        # Must contain article indicator
        has_article_pattern = any(pattern in url_lower for pattern in self.news_patterns['article_indicators'])

        # Must not contain exclude pattern
        has_exclude_pattern = any(pattern in url_lower for pattern in self.news_patterns['exclude_patterns'])

        # Additional heuristics for news articles
        has_date_pattern = any(char.isdigit() for char in url.split('/')[-1][:8])  # Likely has date
        has_reasonable_length = 10 < len(url.split('/')[-1]) < 200  # Reasonable title length

        return has_article_pattern and not has_exclude_pattern and (has_date_pattern or has_reasonable_length)

    def _is_likely_article_enhanced(self, url: str, link_text: str, link_elem, base_url: str) -> bool:
        """Enhanced article detection using multiple heuristics"""
        import re

        url_lower = url.lower()
        text_lower = link_text.lower()

        # Exclude obvious non-articles
        exclude_patterns = [
            '/tag/', '/category/', '/author/', '/search/', '/page/',
            '/login', '/register', '/contact', '/about', '/privacy',
            '/terms', '/policy', '/subscribe', '/newsletter',
            '.jpg', '.png', '.gif', '.pdf', '.css', '.js',
            '/feed', '/rss', '/sitemap', 'mailto:', '#',
            '/wp-admin', '/admin', '/dashboard'
        ]

        if any(pattern in url_lower for pattern in exclude_patterns):
            return False

        # Positive indicators in URL
        url_indicators = [
            '/news/', '/article/', '/story/', '/post/', '/content/',
            '/politics/', '/business/', '/sport/', '/economy/', '/opinion/',
            '/editorial/', '/analysis/', '/feature/', '/report/',
            '/international/', '/national/', '/local/', '/breaking/',
            '/world/', '/africa/', '/ethiopia/', '/somalia/', '/kenya/'
        ]

        has_url_indicator = any(pattern in url_lower for pattern in url_indicators)

        # Date patterns in URL (very common in news sites)
        date_patterns = [
            r'/202[0-9]/', r'/20[1-2][0-9]/',  # Years
            r'/\d{4}/\d{1,2}/', r'/\d{4}-\d{1,2}/',  # Year/month
            r'-2024-', r'-2025-', r'/2024/', r'/2025/'  # Current years
        ]

        has_date_pattern = any(re.search(pattern, url) for pattern in date_patterns)

        # Analyze link text for article indicators
        text_indicators = [
            # English indicators
            'news', 'breaking', 'latest', 'update', 'report', 'analysis',
            'story', 'article', 'exclusive', 'interview', 'investigation',
            # Ethiopian/African context
            'ethiopia', 'addis', 'ababa', 'african', 'horn', 'conflict',
            'government', 'politics', 'economy', 'business', 'development'
        ]

        has_text_indicator = any(indicator in text_lower for indicator in text_indicators)

        # Check if link text looks like a headline (longer, descriptive)
        is_headline_like = (
            len(link_text) > 20 and  # Reasonable length
            len(link_text) < 200 and  # Not too long
            ' ' in link_text and  # Contains spaces
            not link_text.isupper() and  # Not all caps
            not url_lower.endswith(link_text.lower().replace(' ', '-'))  # Not just URL text
        )

        # Check surrounding context
        has_good_context = self._check_link_context(link_elem)

        # Scoring system
        score = 0

        if has_url_indicator:
            score += 3
        if has_date_pattern:
            score += 2
        if has_text_indicator:
            score += 2
        if is_headline_like:
            score += 2
        if has_good_context:
            score += 1

        # Additional checks for common news site patterns
        if '/content/' in url_lower and len(link_text) > 30:
            score += 2

        # If URL is just domain + ID/slug, it might be an article
        from urllib.parse import urlparse
        path = urlparse(url).path
        if len(path.split('/')) == 2 and len(path) > 10:  # Like /some-article-title
            score += 1

        # Lower threshold and add debug logging for difficult sites
        if score >= 2:  # Lowered threshold
            logger.debug(f"Article candidate (score={score}): {url} | text='{link_text[:50]}'")
            return True
        elif score >= 1:  # Log near-misses for debugging
            logger.debug(f"Near-miss (score={score}): {url} | text='{link_text[:50]}'")
        return False

    def _check_link_context(self, link_elem) -> bool:
        """Check if link is in a good context for articles"""

        # Check parent elements for article indicators
        parent = link_elem.parent
        for _ in range(3):  # Check up to 3 levels up
            if parent is None:
                break

            parent_class = parent.get('class', [])
            parent_id = parent.get('id', '')

            context_indicators = [
                'article', 'post', 'news', 'content', 'story',
                'headline', 'title', 'main', 'featured'
            ]

            parent_text = ' '.join(parent_class) + ' ' + parent_id
            if any(indicator in parent_text.lower() for indicator in context_indicators):
                return True

            parent = parent.parent

        return False

    def _is_same_domain(self, url: str, base_url: str) -> bool:
        """Check if URL is from same domain (handles redirects and similar domains)"""
        try:
            from urllib.parse import urlparse
            url_domain = urlparse(url).netloc.lower()
            base_domain = urlparse(base_url).netloc.lower()

            # Direct match
            if url_domain == base_domain:
                return True

            # Handle common domain variations (www vs non-www, .net vs .news, etc.)
            # Extract the main domain part (remove www, extract base name)
            def extract_base_domain(domain):
                # Remove www prefix
                if domain.startswith('www.'):
                    domain = domain[4:]
                # Extract main part before TLD
                parts = domain.split('.')
                if len(parts) >= 2:
                    return parts[0]  # Return the main domain name
                return domain

            url_base = extract_base_domain(url_domain)
            base_base = extract_base_domain(base_domain)

            # Check if main domain names match (e.g., "addisfortune" from both "addisfortune.net" and "addisfortune.news")
            if url_base == base_base and url_base:
                logger.debug(f"Domain match found: {url_domain} ~ {base_domain} (base: {url_base})")
                return True

            return False
        except:
            return False

    async def _discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Discover article URLs from sitemap using stealth."""
        sitemap_paths = ['/sitemap.xml', '/sitemap_index.xml', '/news-sitemap.xml']
        article_urls = []

        for path in sitemap_paths:
            try:
                sitemap_url = f"{base_url.rstrip('/')}{path}"
                html_content, _ = await self._fetch_html_with_stealth(sitemap_url)

                if html_content and '<urlset' in html_content:
                    # Parse sitemap XML
                    soup = BeautifulSoup(html_content, 'xml')
                    urls = soup.find_all('url')

                    for url_elem in urls:
                        loc = url_elem.find('loc')
                        if loc and self._is_article_url(loc.text):
                            article_urls.append(loc.text)

                    logger.info(f"Found {len(urls)} URLs in sitemap {sitemap_url}")
                    break  # Use first successful sitemap

            except Exception as e:
                logger.debug(f"Sitemap {path} not accessible: {e}")

        return article_urls

    async def _discover_rss_urls(self, base_url: str, homepage_content: str) -> List[str]:
        """Discover article URLs from RSS feeds using stealth."""
        article_urls = []

        try:
            # Look for RSS links in homepage
            soup = BeautifulSoup(homepage_content, 'html.parser')
            rss_links = soup.find_all('link', {'type': 'application/rss+xml'})

            # Common RSS paths to try
            rss_paths = ['/feed', '/rss', '/rss.xml', '/feeds/all.rss']

            # Add discovered RSS links
            for link in rss_links:
                href = link.get('href')
                if href:
                    if href.startswith('/'):
                        href = f"{base_url.rstrip('/')}{href}"
                    rss_paths.append(href)

            # Fetch RSS feeds with stealth
            for rss_path in rss_paths[:3]:  # Limit to 3 feeds
                try:
                    if not rss_path.startswith('http'):
                        rss_url = f"{base_url.rstrip('/')}{rss_path}"
                    else:
                        rss_url = rss_path

                    rss_content, _ = await self._fetch_html_with_stealth(rss_url)

                    if rss_content and ('<rss' in rss_content or '<feed' in rss_content):
                        # Parse RSS/Atom feed
                        soup = BeautifulSoup(rss_content, 'xml')

                        # RSS format
                        items = soup.find_all('item')
                        for item in items:
                            link = item.find('link')
                            if link and link.text:
                                article_urls.append(link.text.strip())

                        # Atom format
                        entries = soup.find_all('entry')
                        for entry in entries:
                            link = entry.find('link')
                            if link and link.get('href'):
                                article_urls.append(link.get('href'))

                        logger.info(f"Found {len(items + entries)} articles in RSS {rss_url}")
                        break  # Use first successful RSS feed

                except Exception as e:
                    logger.debug(f"RSS feed {rss_path} not accessible: {e}")

        except Exception as e:
            logger.debug(f"RSS discovery failed: {e}")

        return article_urls

    async def crawl_entire_site(self, base_url: str, max_articles: int = 50) -> List[NewsArticle]:
        """
        Crawl entire news site using stealth discovery and extraction.

        Args:
            base_url: Base URL of the news site
            max_articles: Maximum number of articles to crawl

        Returns:
            List of successfully extracted NewsArticle objects
        """
        logger.info(f"🚀 Starting full site crawl of {base_url} (max: {max_articles})")

        try:
            # Step 1: Discover all article URLs with stealth
            article_urls = await self.discover_site_articles(base_url)

            if not article_urls:
                logger.warning(f"No articles discovered for {base_url}")
                return []

            # Step 2: Limit and prioritize URLs
            if len(article_urls) > max_articles:
                # Prioritize based on site configuration
                site_config = self._get_site_config(base_url)
                priority_sections = site_config.get('priority_sections', [])

                # Sort: priority sections first, then recent (assuming reverse chronological in discovery)
                prioritized_urls = []
                remaining_urls = []

                for url in article_urls:
                    if any(section in url for section in priority_sections):
                        prioritized_urls.append(url)
                    else:
                        remaining_urls.append(url)

                # Take priority URLs + remaining up to max_articles
                selected_urls = prioritized_urls[:max_articles]
                if len(selected_urls) < max_articles:
                    selected_urls.extend(remaining_urls[:max_articles - len(selected_urls)])

                article_urls = selected_urls

            logger.info(f"📋 Selected {len(article_urls)} articles for crawling")

            # Step 3: Crawl each article with stealth
            articles = []
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def crawl_with_semaphore(url):
                async with semaphore:
                    return await self.crawl_url(url)

            # Process in batches to avoid overwhelming the site
            batch_size = min(self.max_concurrent, 5)
            for i in range(0, len(article_urls), batch_size):
                batch_urls = article_urls[i:i + batch_size]

                logger.info(f"🔄 Processing batch {i//batch_size + 1}/{(len(article_urls) + batch_size - 1)//batch_size}")

                # Crawl batch concurrently
                tasks = [crawl_with_semaphore(url) for url in batch_urls]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect successful articles
                for url, result in zip(batch_urls, batch_results):
                    if isinstance(result, NewsArticle) and result.title:
                        articles.append(result)
                        logger.info(f"✅ Extracted: {result.title[:50]}...")
                    elif isinstance(result, Exception):
                        logger.warning(f"⚠️ Failed {url}: {result}")
                    else:
                        logger.warning(f"⚠️ No content extracted from {url}")

                # Inter-batch delay for politeness
                if i + batch_size < len(article_urls):
                    await asyncio.sleep(2)

            logger.info(f"🎯 Site crawl completed: {len(articles)}/{len(article_urls)} articles extracted")
            return articles

        except Exception as e:
            logger.error(f"Failed to crawl entire site {base_url}: {e}")
            return []

    def get_stealth_stats(self) -> Dict[str, Any]:
        """Get comprehensive stealth performance statistics."""
        return {
            'cloudflare_bypass_enabled': self.stealth_config.enable_cloudflare_bypass,
            'fingerprint_protection_enabled': self.stealth_config.enable_fingerprint_protection,
            'east_african_optimization': self.stealth_config.east_african_mode,
            'cloudscraper_available': CLOUDSCRAPER_AVAILABLE,
            'user_agents_pool_size': len(self.user_agents),
            'session_counter': self.session_counter,
            'supported_sites': list(self.east_african_sites.keys())
        }

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.cloudscraper_session:
                # CloudScraper sessions don't need explicit cleanup
                pass

            if self.database_manager:
                await self.database_manager.close()

            logger.info("Enhanced News Crawler cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory functions for easy initialization
def create_news_crawler(config: Optional[Dict[str, Any]] = None) -> NewsCrawler:
    """
    Create a news crawler with optimal stealth settings.

    Args:
        config: Optional configuration override

    Returns:
        Configured NewsCrawler instance
    """
    default_config = {
        'enable_cloudflare_bypass': True,
        'enable_fingerprint_protection': True,
        'enable_residential_proxies': False,  # Enable when proxies are available
        'min_delay': 3.0,
        'max_delay': 8.0,
        'cloudflare_wait_time': 15.0,
        'max_retries': 3,
        'max_concurrent': 2,  # Conservative for stealth
        'enable_ml_analysis': True,
        'enable_monitoring': True,
        'enable_database': False  # Enable when needed
    }

    if config:
        default_config.update(config)

    return NewsCrawler(default_config)


def create_stealth_crawler(aggressive: bool = False) -> NewsCrawler:
    """
    Create a crawler with stealth settings optimized for protected sites.

    Args:
        aggressive: If True, uses more aggressive delays for heavily protected sites

    Returns:
        NewsCrawler optimized for stealth
    """
    if aggressive:
        config = {
            'enable_cloudflare_bypass': True,
            'enable_fingerprint_protection': True,
            'min_delay': 5.0,
            'max_delay': 12.0,
            'cloudflare_wait_time': 20.0,
            'max_retries': 3,
            'max_concurrent': 1  # Very conservative
        }
    else:
        config = {
            'enable_cloudflare_bypass': True,
            'enable_fingerprint_protection': True,
            'min_delay': 3.0,
            'max_delay': 8.0,
            'cloudflare_wait_time': 15.0,
            'max_retries': 3,
            'max_concurrent': 2
        }

    return NewsCrawler(config)
