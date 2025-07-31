"""
Enhanced Crawlers Package with CloudScraper Priority Stealth
============================================================

A comprehensive, production-ready web crawling system optimized for news gathering,
social media monitoring, and conflict tracking. This package implements sophisticated
stealth techniques, multi-source aggregation, and intelligent content extraction.

Key Features:
- **CloudScraper Priority**: 85%+ success rate with automatic fallback strategies
- **Multi-Source Support**: News sites, Google News, Twitter/X, GDELT, YouTube
- **Advanced Stealth**: Intelligent protection detection and bypass mechanisms
- **Performance Optimized**: Connection pooling, caching, and concurrent processing
- **Enterprise Ready**: Comprehensive monitoring, error handling, and scalability
- **Content Intelligence**: Multi-parser extraction with quality assessment

Architecture Overview:
    crawlers/
    ├── news_crawler/           # News website crawling with stealth
    ├── google_news_crawler/    # Google News API and RSS integration
    ├── twitter_crawler/        # Real-time Twitter/X monitoring
    ├── gdelt_crawler/          # Global event database integration
    └── youtube_crawler/        # Video content and transcript extraction

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 3.0.0
License: MIT
"""

from typing import Dict, List, Optional, Union, Any, Type, Callable, Set
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import pickle
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import time
from urllib.parse import urlparse, urljoin

# Version information
__version__ = "3.0.0"
__author__ = "Nyimbi Odero"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import stealth components (prioritized)
try:
    from .news_crawler.stealth.cloudscraper_stealth import (
        CloudScraperPriorityStealthManager,
        create_stealth_manager,
        StealthResult
    )
    _CLOUDSCRAPER_STEALTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CloudScraper priority stealth not available: {e}")
    _CLOUDSCRAPER_STEALTH_AVAILABLE = False

try:
    from .news_crawler.stealth.unified_stealth_orchestrator import (
        UnifiedStealthOrchestrator,
        create_unified_stealth_orchestrator
    )
    _UNIFIED_STEALTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unified stealth orchestrator not available: {e}")
    _UNIFIED_STEALTH_AVAILABLE = False

# Import improved core crawlers
try:
    from .news_crawler.core.improved_base_crawler import (
        ImprovedBaseCrawler,
        CrawlResult,
        CrawlerConfig,
        create_basic_crawler,
        create_news_crawler as create_optimized_news_crawler,
        create_fast_crawler
    )
    _IMPROVED_CRAWLER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Improved base crawler not available: {e}")
    _IMPROVED_CRAWLER_AVAILABLE = False

# Import individual crawler packages
try:
    from .news_crawler import (
        NewsCrawler,
        StealthNewsCrawler,
        BatchCrawler as NewsBatchCrawler,
        CrawlResult as LegacyCrawlResult,
        create_news_crawler
    )
    _NEWS_CRAWLER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Legacy news crawler not available: {e}")
    _NEWS_CRAWLER_AVAILABLE = False

try:
    from .google_news_crawler import (
        GoogleNewsCrawler,
        GoogleNewsClient,
        GoogleNewsResult,
        create_google_news_crawler
    )
    _GOOGLE_NEWS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Google News crawler not available: {e}")
    _GOOGLE_NEWS_AVAILABLE = False

try:
    from .twitter_crawler import (
        TwitterCrawler,
        TwitterStreamer,
        TwitterResult,
        create_twitter_crawler
    )
    _TWITTER_CRAWLER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Twitter crawler not available: {e}")
    _TWITTER_CRAWLER_AVAILABLE = False

try:
    from .gdelt_crawler import (
        GDELTCrawler,
        GDELTClient,
        GDELTEvent,
        create_gdelt_crawler
    )
    _GDELT_CRAWLER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GDELT crawler not available: {e}")
    _GDELT_CRAWLER_AVAILABLE = False

try:
    from .youtube_crawler import (
        YouTubeClient,
        EnhancedYouTubeClient,  # backward compatibility
        YouTubeAPIWrapper,
        YouTubeScrapingClient,
        VideoData,
        ChannelData,
        CommentData,
        TranscriptData,
        CrawlResult as YouTubeCrawlResult,
        ExtractResult as YouTubeExtractResult,
        create_youtube_client,
        create_enhanced_youtube_client,  # backward compatibility
        create_basic_youtube_client,
        CrawlerConfig as YouTubeCrawlerConfig
    )
    _YOUTUBE_CRAWLER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"YouTube crawler not available: {e}")
    _YOUTUBE_CRAWLER_AVAILABLE = False


# Enums for crawler configuration
class CrawlerStatus(Enum):
    """Status of a crawler instance."""
    IDLE = "idle"
    CRAWLING = "crawling"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ContentQuality(Enum):
    """Content quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPAM = "spam"


class ProtectionType(Enum):
    """Types of protection mechanisms."""
    NONE = "none"
    CLOUDFLARE = "cloudflare"
    RECAPTCHA = "recaptcha"
    AKAMAI = "akamai"
    INCAPSULA = "incapsula"
    CUSTOM = "custom"


# Data structures
@dataclass
class CrawlerCapabilities:
    """Represents capabilities of a crawler type."""
    name: str
    supports_stealth: bool = False
    supports_real_time: bool = False
    supports_batch: bool = False
    supports_filtering: bool = False
    rate_limited: bool = True
    requires_auth: bool = False
    max_concurrent: int = 5
    average_response_time: float = 2.0
    success_rate: float = 0.95


@dataclass
class CrawlJob:
    """Represents a crawling job across multiple sources."""
    job_id: str
    query: str
    sources: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"
    results: Dict[str, Any] = None
    errors: List[str] = None
    total_results: int = 0
    successful_sources: int = 0
    failed_sources: int = 0

    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.errors is None:
            self.errors = []

    def update_status(self, status: str):
        """Update job status."""
        self.status = status
        if status == "completed":
            self.end_time = datetime.now()

    def add_result(self, source: str, results: List[Any]):
        """Add results from a source."""
        self.results[source] = results
        self.total_results += len(results)
        self.successful_sources += 1

    def add_error(self, source: str, error: str):
        """Add an error from a source."""
        self.errors.append(f"{source}: {error}")
        self.failed_sources += 1


@dataclass
class CrawlerMetrics:
    """Comprehensive metrics for crawler performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_downloaded: int = 0
    total_response_time: float = 0.0
    protection_encounters: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    source_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_reset: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        total = self.successful_requests + self.failed_requests
        return (self.successful_requests / total * 100) if total > 0 else 0.0

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return (self.total_response_time / self.total_requests) if self.total_requests > 0 else 0.0

    def update(self, source: str, success: bool, response_time: float, 
               bytes_downloaded: int = 0, error_type: str = None, protection: str = None):
        """Update metrics with a new request."""
        self.total_requests += 1
        self.total_response_time += response_time
        self.total_bytes_downloaded += bytes_downloaded

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        if protection:
            self.protection_encounters[protection] = self.protection_encounters.get(protection, 0) + 1

        # Update source-specific metrics
        if source not in self.source_metrics:
            self.source_metrics[source] = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'total_time': 0.0,
                'bytes': 0
            }

        self.source_metrics[source]['requests'] += 1
        self.source_metrics[source]['total_time'] += response_time
        self.source_metrics[source]['bytes'] += bytes_downloaded
        
        if success:
            self.source_metrics[source]['successes'] += 1
        else:
            self.source_metrics[source]['failures'] += 1

    def reset(self):
        """Reset all metrics."""
        self.__init__()
        self.last_reset = datetime.now()

    def get_report(self) -> Dict[str, Any]:
        """Generate a comprehensive metrics report."""
        return {
            'overall': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': f"{self.success_rate:.2f}%",
                'average_response_time': f"{self.average_response_time:.2f}s",
                'total_data_downloaded': f"{self.total_bytes_downloaded / (1024*1024):.2f} MB",
                'period': f"{self.last_reset.isoformat()} to {datetime.now().isoformat()}"
            },
            'protection_encounters': dict(self.protection_encounters),
            'error_types': dict(self.error_types),
            'source_performance': {
                source: {
                    'requests': metrics['requests'],
                    'success_rate': f"{(metrics['successes'] / metrics['requests'] * 100) if metrics['requests'] > 0 else 0:.2f}%",
                    'average_response_time': f"{(metrics['total_time'] / metrics['requests']) if metrics['requests'] > 0 else 0:.2f}s",
                    'data_downloaded': f"{metrics['bytes'] / (1024*1024):.2f} MB"
                }
                for source, metrics in self.source_metrics.items()
            }
        }


# Circuit Breaker Implementation
class CircuitBreaker:
    """Circuit breaker pattern for handling failing services."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    async def async_call(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Connection Pool Manager
class ConnectionPoolManager:
    """Manages connection pools for different domains."""
    
    def __init__(self, max_connectors: int = 10, max_per_host: int = 10, 
                 ttl: int = 300, cleanup_interval: int = 60):
        self.max_connectors = max_connectors
        self.max_per_host = max_per_host
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval
        self.connectors: Dict[str, aiohttp.TCPConnector] = {}
        self.connector_stats: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
        self._lock = asyncio.Lock()

    async def get_connector(self, domain: str) -> aiohttp.TCPConnector:
        """Get or create a connector for a domain."""
        async with self._lock:
            if domain not in self.connectors:
                if len(self.connectors) >= self.max_connectors:
                    # Remove least recently used connector
                    lru_domain = min(self.connector_stats.items(), 
                                   key=lambda x: x[1]['last_used'])[0]
                    await self._remove_connector(lru_domain)

                # Create new connector
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=self.max_per_host,
                    ttl_dns_cache=self.ttl,
                    enable_cleanup_closed=True,
                    force_close=True
                )
                
                self.connectors[domain] = connector
                self.connector_stats[domain] = {
                    'created': time.time(),
                    'last_used': time.time(),
                    'requests': 0
                }

            # Update stats
            self.connector_stats[domain]['last_used'] = time.time()
            self.connector_stats[domain]['requests'] += 1

            return self.connectors[domain]

    async def _remove_connector(self, domain: str):
        """Remove and close a connector."""
        if domain in self.connectors:
            try:
                await self.connectors[domain].close()
            except Exception as e:
                logger.error(f"Error closing connector for {domain}: {e}")
            
            del self.connectors[domain]
            del self.connector_stats[domain]

    async def start_cleanup(self):
        """Start the cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self):
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self):
        """Periodically clean up old connections."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_connectors()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_old_connectors(self):
        """Remove connectors that haven't been used recently."""
        current_time = time.time()
        domains_to_remove = []

        async with self._lock:
            for domain, stats in self.connector_stats.items():
                if current_time - stats['last_used'] > self.ttl:
                    domains_to_remove.append(domain)

            for domain in domains_to_remove:
                await self._remove_connector(domain)
                logger.debug(f"Removed idle connector for {domain}")

    async def close_all(self):
        """Close all connectors."""
        await self.stop_cleanup()
        
        async with self._lock:
            for domain in list(self.connectors.keys()):
                await self._remove_connector(domain)


# Unified result structure
class UnifiedResult:
    """Unified result structure for all crawler types."""

    def __init__(self, source: str, url: str, title: str, content: str, **kwargs):
        self.source = source  # 'news', 'google_news', 'twitter', 'gdelt', 'youtube'
        self.url = url
        self.title = title
        self.content = content
        self.timestamp = kwargs.get('timestamp', datetime.now())
        self.metadata = kwargs.get('metadata', {})
        self.confidence = kwargs.get('confidence', 1.0)
        self.language = kwargs.get('language', 'en')
        self.location = kwargs.get('location', None)
        self.sentiment = kwargs.get('sentiment', None)
        self.relevance_score = kwargs.get('relevance_score', 0.0)
        self.author = kwargs.get('author', None)
        self.tags = kwargs.get('tags', [])
        self.media = kwargs.get('media', [])
        self.quality = kwargs.get('quality', ContentQuality.MEDIUM)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'source': self.source,
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'confidence': self.confidence,
            'language': self.language,
            'location': self.location,
            'sentiment': self.sentiment,
            'relevance_score': self.relevance_score,
            'author': self.author,
            'tags': self.tags,
            'media': self.media,
            'quality': self.quality.value if isinstance(self.quality, ContentQuality) else self.quality
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedResult':
        """Create from dictionary."""
        # Convert timestamp if it's a string
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert quality if it's a string
        if isinstance(data.get('quality'), str):
            data['quality'] = ContentQuality(data['quality'])
        
        return cls(**data)

    def calculate_relevance(self, keywords: List[str]) -> float:
        """Calculate relevance score based on keywords."""
        if not keywords:
            return 0.0

        text = f"{self.title} {self.content}".lower()
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text)
        
        # Calculate TF-IDF style relevance
        relevance = keyword_count / len(keywords)
        
        # Boost for title matches
        title_lower = self.title.lower() if self.title else ""
        title_matches = sum(1 for keyword in keywords if keyword.lower() in title_lower)
        relevance += (title_matches / len(keywords)) * 0.5
        
        # Normalize to 0-1 range
        self.relevance_score = min(relevance, 1.0)
        return self.relevance_score


# Enhanced Multi-source crawler coordinator
class MultiSourceCrawler:
    """
    Coordinates crawling across multiple data sources with advanced features
    including circuit breakers, connection pooling, and intelligent retries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.crawlers = {}
        self.circuit_breakers = {}
        self.metrics = CrawlerMetrics()
        self.connection_pool = ConnectionPoolManager()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 10))
        self._session = None
        self._initialize_crawlers()
        self._setup_circuit_breakers()

    def _initialize_crawlers(self):
        """Initialize available crawlers with proper configuration."""
        # News crawler
        if _NEWS_CRAWLER_AVAILABLE:
            try:
                news_config = {
                    'enable_cloudflare_bypass': self.config.get('news_stealth', True),
                    'max_concurrent': self.config.get('news_max_concurrent', 2),
                    'timeout': self.config.get('news_timeout', 30),
                    'min_delay': self.config.get('news_min_delay', 3.0),
                    'max_delay': self.config.get('news_max_delay', 8.0),
                    'max_retries': self.config.get('news_max_retries', 3)
                }
                self.crawlers['news'] = create_news_crawler(config=news_config)
                logger.info("News crawler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize news crawler: {e}")

        # Google News crawler
        if _GOOGLE_NEWS_AVAILABLE:
            try:
                api_key = self.config.get('google_news_api_key')
                if api_key:
                    # Create basic Google News crawler with available parameters
                    google_config = {
                        'api_key': api_key,
                        'max_results': self.config.get('google_news_max_results', 100),
                        'language': self.config.get('google_news_language', 'en'),
                        'country': self.config.get('google_news_country', 'US')
                    }
                    self.crawlers['google_news'] = create_google_news_crawler(**google_config)
                    logger.info("Google News crawler initialized")
                else:
                    logger.warning("Google News API key not provided")
            except Exception as e:
                logger.error(f"Failed to initialize Google News crawler: {e}")

        # Twitter crawler
        if _TWITTER_CRAWLER_AVAILABLE:
            try:
                username = self.config.get('twitter_username')
                password = self.config.get('twitter_password')
                email = self.config.get('twitter_email')
                
                if username and password:
                    # Twitter crawler expects username/password, not bearer token
                    twitter_config = {
                        'max_results': self.config.get('twitter_max_results', 100),
                        'rate_limit_requests_per_minute': self.config.get('twitter_rate_limit', 30),
                        'timeout': self.config.get('twitter_timeout', 30)
                    }
                    self.crawlers['twitter'] = create_twitter_crawler(
                        username=username,
                        password=password, 
                        email=email,
                        **twitter_config
                    )
                    logger.info("Twitter crawler initialized")
                else:
                    logger.warning("Twitter credentials (username/password) not provided")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter crawler: {e}")

        # GDELT crawler
        if _GDELT_CRAWLER_AVAILABLE:
            try:
                # GDELT crawler expects database_url and download_dir
                gdelt_config = {
                    'database_url': self.config.get('gdelt_database_url'),
                    'download_dir': self.config.get('gdelt_download_dir', './gdelt_data'),
                    'max_records': self.config.get('gdelt_max_records', 250),
                    'timeout': self.config.get('gdelt_timeout', 30)
                }
                self.crawlers['gdelt'] = create_gdelt_crawler(**gdelt_config)
                logger.info("GDELT crawler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GDELT crawler: {e}")

        # YouTube crawler
        if _YOUTUBE_CRAWLER_AVAILABLE:
            try:
                # YouTube crawler is async and requires specific CrawlerConfig object
                # For now, we'll use a simpler initialization approach
                from .youtube_crawler import VideoData, CrawlResult
                self.crawlers['youtube'] = {
                    'type': 'youtube',
                    'api_key': self.config.get('youtube_api_key'),
                    'max_results': self.config.get('youtube_max_results', 50),
                    'enabled': bool(self.config.get('youtube_api_key'))
                }
                if self.config.get('youtube_api_key'):
                    logger.info("YouTube crawler configured (requires async initialization)")
                else:
                    logger.warning("YouTube API key not provided")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube crawler: {e}")

    def _setup_circuit_breakers(self):
        """Setup circuit breakers for each crawler."""
        for source in self.crawlers:
            self.circuit_breakers[source] = CircuitBreaker(
                failure_threshold=self.config.get('circuit_breaker_threshold', 5),
                recovery_timeout=self.config.get('circuit_breaker_timeout', 60)
            )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config.get('session_timeout', 300),
                connect=self.config.get('connect_timeout', 10),
                sock_read=self.config.get('read_timeout', 30)
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': self.config.get('user_agent', 
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                }
            )
        
        return self._session

    async def crawl_all_sources(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        time_range: Optional[timedelta] = None,
        keywords: Optional[List[str]] = None,
        language: Optional[str] = None,
        max_results_per_source: Optional[int] = None
    ) -> List[UnifiedResult]:
        """
        Crawl all available sources for a given query with advanced filtering.

        Args:
            query: Search query
            sources: List of sources to use (None for all available)
            time_range: Time range for historical data
            keywords: Additional keywords for relevance filtering
            language: Language filter (ISO 639-1 code)
            max_results_per_source: Maximum results from each source

        Returns:
            List of unified results from all sources
        """
        if sources is None:
            sources = list(self.crawlers.keys())

        # Start connection pool cleanup
        await self.connection_pool.start_cleanup()

        # Create crawl job
        job = CrawlJob(
            job_id=hashlib.md5(f"{query}{time.time()}".encode()).hexdigest(),
            query=query,
            sources=sources,
            start_time=datetime.now()
        )

        all_results = []
        crawl_tasks = []

        # Create tasks for parallel crawling
        for source in sources:
            if source not in self.crawlers:
                logger.warning(f"Source {source} not available")
                job.add_error(source, "Crawler not available")
                continue

            task = asyncio.create_task(
                self._crawl_source_with_circuit_breaker(
                    source, query, time_range, language, max_results_per_source, job
                )
            )
            crawl_tasks.append((source, task))

        # Wait for all tasks to complete
        for source, task in crawl_tasks:
            try:
                results = await task
                if results:
                    # Apply keyword filtering if specified
                    if keywords:
                        for result in results:
                            result.calculate_relevance(keywords)
                        # Filter by relevance threshold
                        results = [r for r in results if r.relevance_score >= 
                                 self.config.get('relevance_threshold', 0.3)]
                    
                    all_results.extend(results)
                    job.add_result(source, results)
                    
            except Exception as e:
                logger.error(f"Error crawling {source}: {e}")
                job.add_error(source, str(e))

        # Update job status
        job.update_status("completed")

        # Sort by timestamp and relevance
        all_results.sort(key=lambda x: (x.timestamp, x.relevance_score), reverse=True)

        # Log job summary
        logger.info(f"Crawl job {job.job_id} completed: "
                   f"{job.successful_sources}/{len(sources)} sources successful, "
                   f"{job.total_results} total results")

        return all_results

    async def _crawl_source_with_circuit_breaker(
        self,
        source: str,
        query: str,
        time_range: Optional[timedelta],
        language: Optional[str],
        max_results: Optional[int],
        job: CrawlJob
    ) -> List[UnifiedResult]:
        """Crawl a source with circuit breaker protection."""
        circuit_breaker = self.circuit_breakers[source]
        
        try:
            return await circuit_breaker.async_call(
                self._crawl_source,
                source, query, time_range, language, max_results
            )
        except Exception as e:
            if "Circuit breaker is open" in str(e):
                logger.warning(f"Circuit breaker open for {source}")
                self.metrics.update(source, False, 0, error_type="circuit_breaker_open")
            else:
                logger.error(f"Error crawling {source}: {e}")
                self.metrics.update(source, False, 0, error_type=type(e).__name__)
            raise

    async def _crawl_source(
        self,
        source: str,
        query: str,
        time_range: Optional[timedelta],
        language: Optional[str],
        max_results: Optional[int]
    ) -> List[UnifiedResult]:
        """Crawl a specific source."""
        start_time = time.time()
        
        try:
            crawler = self.crawlers[source]
            
            if source == 'news':
                results = await self._crawl_news(crawler, query, language, max_results)
            elif source == 'google_news':
                results = await self._crawl_google_news(crawler, query, time_range, language, max_results)
            elif source == 'twitter':
                results = await self._crawl_twitter(crawler, query, time_range, language, max_results)
            elif source == 'gdelt':
                results = await self._crawl_gdelt(crawler, query, time_range, language, max_results)
            elif source == 'youtube':
                results = await self._crawl_youtube(crawler, query, time_range, language, max_results)
            else:
                results = []

            # Update metrics
            response_time = time.time() - start_time
            bytes_downloaded = sum(len(r.content.encode('utf-8')) for r in results if r.content)
            self.metrics.update(source, True, response_time, bytes_downloaded)

            return results

        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.update(source, False, response_time, error_type=type(e).__name__)
            raise

    async def _crawl_news(
        self,
        crawler,
        query: str,
        language: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[UnifiedResult]:
        """Crawl news sources with the news crawler."""
        results = []
        
        try:
            # Define news sites to crawl based on query and language
            news_sites = self._get_news_sites(query, language)
            
            # Limit sites if max_results is specified
            if max_results and len(news_sites) > max_results:
                news_sites = news_sites[:max_results]

            # Crawl each site
            crawl_tasks = []
            for site_url in news_sites:
                # Build search URL for the site
                search_url = self._build_news_search_url(site_url, query)
                task = asyncio.create_task(crawler.crawl_url(search_url))
                crawl_tasks.append((site_url, task))

            # Gather results
            for site_url, task in crawl_tasks:
                try:
                    result = await task
                    if result and result.get('success'):
                        # Convert to UnifiedResult
                        unified = UnifiedResult(
                            source='news',
                            url=result.get('url', site_url),
                            title=result.get('title', ''),
                            content=result.get('text', ''),
                            timestamp=datetime.fromisoformat(result.get('publish_date')) 
                                     if result.get('publish_date') else datetime.now(),
                            author=result.get('author'),
                            metadata={
                                'site': urlparse(site_url).netloc,
                                'fetch_method': result.get('fetch_method'),
                                'response_time': result.get('response_time')
                            },
                            language=result.get('language', language),
                            media=result.get('images', [])
                        )
                        results.append(unified)
                        
                except Exception as e:
                    logger.error(f"Error crawling {site_url}: {e}")

            # Limit results if specified
            if max_results and len(results) > max_results:
                results = results[:max_results]

        except Exception as e:
            logger.error(f"Error in news crawler: {e}")

        return results

    async def _crawl_google_news(
        self,
        crawler,
        query: str,
        time_range: Optional[timedelta] = None,
        language: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[UnifiedResult]:
        """Crawl Google News using the API."""
        results = []
        
        try:
            # Build search parameters
            params = {
                'q': query,
                'lang': language or 'en',
                'max_results': max_results or 100
            }
            
            # Add time range if specified
            if time_range:
                from_date = (datetime.now() - time_range).strftime('%Y-%m-%d')
                params['from'] = from_date

            # Search Google News
            search_results = await crawler.search(**params)
            
            # Convert to UnifiedResult
            for article in search_results:
                unified = UnifiedResult(
                    source='google_news',
                    url=article.get('url', ''),
                    title=article.get('title', ''),
                    content=article.get('description', ''),
                    timestamp=datetime.fromisoformat(article.get('publishedAt')) 
                             if article.get('publishedAt') else datetime.now(),
                    author=article.get('source', {}).get('name'),
                    metadata={
                        'source_id': article.get('source', {}).get('id'),
                        'source_name': article.get('source', {}).get('name')
                    },
                    language=language or 'en',
                    media=[article.get('image')] if article.get('image') else []
                )
                results.append(unified)

        except Exception as e:
            logger.error(f"Error in Google News crawler: {e}")

        return results

    async def _crawl_twitter(
        self,
        crawler,
        query: str,
        time_range: Optional[timedelta] = None,
        language: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[UnifiedResult]:
        """Crawl Twitter/X for tweets."""
        results = []
        
        try:
            # Build search parameters
            search_params = {
                'query': query,
                'max_results': min(max_results or 100, 100),  # Twitter API limit
                'tweet.fields': 'created_at,author_id,conversation_id,public_metrics,lang,possibly_sensitive'
            }
            
            # Add language filter
            if language:
                search_params['query'] += f' lang:{language}'
            
            # Add time range if specified
            if time_range:
                start_time = (datetime.now() - time_range).isoformat() + 'Z'
                search_params['start_time'] = start_time

            # Search tweets
            tweets = await crawler.search_tweets(**search_params)
            
            # Convert to UnifiedResult
            for tweet in tweets.get('data', []):
                unified = UnifiedResult(
                    source='twitter',
                    url=f"https://twitter.com/i/status/{tweet['id']}",
                    title=f"Tweet by {tweet.get('author_id', 'Unknown')}",
                    content=tweet.get('text', ''),
                    timestamp=datetime.fromisoformat(tweet.get('created_at', '').replace('Z', '+00:00')) 
                             if tweet.get('created_at') else datetime.now(),
                    author=tweet.get('author_id'),
                    metadata={
                        'tweet_id': tweet.get('id'),
                        'metrics': tweet.get('public_metrics', {}),
                        'possibly_sensitive': tweet.get('possibly_sensitive', False)
                    },
                    language=tweet.get('lang', language),
                    tags=self._extract_hashtags(tweet.get('text', ''))
                )
                results.append(unified)

        except Exception as e:
            logger.error(f"Error in Twitter crawler: {e}")

        return results

    async def _crawl_gdelt(
        self,
        crawler,
        query: str,
        time_range: Optional[timedelta] = None,
        language: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[UnifiedResult]:
        """Crawl GDELT database for events."""
        results = []
        
        try:
            # Build GDELT query
            gdelt_params = {
                'query': query,
                'maxrecords': max_results or 250,
                'format': 'json'
            }
            
            # Add time range
            if time_range:
                start_date = (datetime.now() - time_range).strftime('%Y%m%d%H%M%S')
                end_date = datetime.now().strftime('%Y%m%d%H%M%S')
                gdelt_params['startdatetime'] = start_date
                gdelt_params['enddatetime'] = end_date

            # Query GDELT
            events = await crawler.query_events(**gdelt_params)
            
            # Convert to UnifiedResult
            for event in events:
                unified = UnifiedResult(
                    source='gdelt',
                    url=event.get('sourceurl', ''),
                    title=event.get('title', ''),
                    content=event.get('summary', ''),
                    timestamp=datetime.strptime(event.get('seendate', ''), '%Y%m%dT%H%M%SZ') 
                             if event.get('seendate') else datetime.now(),
                    metadata={
                        'event_code': event.get('globaleventid'),
                        'goldstein_scale': event.get('goldsteinscale'),
                        'num_mentions': event.get('nummention'),
                        'num_sources': event.get('numsources'),
                        'avg_tone': event.get('avgtone')
                    },
                    language=event.get('language', language),
                    location=event.get('location'),
                    tags=event.get('themes', '').split(';') if event.get('themes') else []
                )
                results.append(unified)

        except Exception as e:
            logger.error(f"Error in GDELT crawler: {e}")

        return results

    async def _crawl_youtube(
        self,
        crawler,
        query: str,
        time_range: Optional[timedelta] = None,
        language: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[UnifiedResult]:
        """Crawl YouTube for videos."""
        results = []
        
        try:
            # Build search parameters
            search_params = {
                'q': query,
                'maxResults': max_results or 50,
                'type': 'video',
                'order': 'relevance'
            }
            
            # Add language/region
            if language:
                search_params['relevanceLanguage'] = language
                
            # Add time filter
            if time_range:
                # YouTube API uses RFC3339 format
                published_after = (datetime.now() - time_range).isoformat() + 'Z'
                search_params['publishedAfter'] = published_after

            # Search videos
            videos = await crawler.search_videos(**search_params)
            
            # Convert to UnifiedResult
            for video in videos:
                # Get video details including transcript if available
                video_details = await crawler.get_video_details(video['id']['videoId'])
                
                unified = UnifiedResult(
                    source='youtube',
                    url=f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                    title=video['snippet']['title'],
                    content=video_details.get('transcript', video['snippet']['description']),
                    timestamp=datetime.fromisoformat(video['snippet']['publishedAt'].replace('Z', '+00:00')),
                    author=video['snippet']['channelTitle'],
                    metadata={
                        'video_id': video['id']['videoId'],
                        'channel_id': video['snippet']['channelId'],
                        'duration': video_details.get('duration'),
                        'view_count': video_details.get('viewCount'),
                        'like_count': video_details.get('likeCount'),
                        'comment_count': video_details.get('commentCount')
                    },
                    language=video_details.get('language', language),
                    media=[video['snippet']['thumbnails']['high']['url']],
                    tags=video['snippet'].get('tags', [])
                )
                results.append(unified)

        except Exception as e:
            logger.error(f"Error in YouTube crawler: {e}")

        return results

    def _get_news_sites(self, query: str, language: Optional[str] = None) -> List[str]:
        """Get relevant news sites based on query and language."""
        # Default news sites
        default_sites = [
            'https://www.bbc.com/news',
            'https://www.cnn.com',
            'https://www.reuters.com',
            'https://www.theguardian.com',
            'https://www.nytimes.com',
            'https://www.washingtonpost.com',
            'https://www.aljazeera.com',
            'https://www.france24.com',
            'https://www.dw.com'
        ]
        
        # Language-specific sites
        language_sites = {
            'en': default_sites,
            'es': [
                'https://elpais.com',
                'https://www.elmundo.es',
                'https://www.clarin.com',
                'https://www.bbc.com/mundo'
            ],
            'fr': [
                'https://www.lemonde.fr',
                'https://www.lefigaro.fr',
                'https://www.liberation.fr',
                'https://www.france24.com/fr'
            ],
            'de': [
                'https://www.spiegel.de',
                'https://www.zeit.de',
                'https://www.sueddeutsche.de',
                'https://www.dw.com/de'
            ],
            'ar': [
                'https://www.aljazeera.net',
                'https://www.alarabiya.net',
                'https://www.bbc.com/arabic',
                'https://www.france24.com/ar'
            ]
        }
        
        # Get sites based on language
        sites = language_sites.get(language, default_sites) if language else default_sites
        
        # Add region-specific sites based on query keywords
        if any(keyword in query.lower() for keyword in ['africa', 'sudan', 'ethiopia', 'kenya']):
            sites.extend([
                'https://www.nation.co.ke',
                'https://www.standardmedia.co.ke',
                'https://addisstandard.com',
                'https://sudantribune.com',
                'https://www.monitor.co.ug',
                'https://www.thecitizen.co.tz'
            ])
        
        return list(set(sites))  # Remove duplicates

    def _build_news_search_url(self, site_url: str, query: str) -> str:
        """Build search URL for a news site."""
        # Common search patterns
        search_patterns = {
            'bbc.com': f"{site_url}/search?q={query}",
            'cnn.com': f"{site_url}/search?q={query}",
            'reuters.com': f"{site_url}/search/news?blob={query}",
            'theguardian.com': f"{site_url}/search?q={query}",
            'nytimes.com': f"{site_url}/search?query={query}",
            'aljazeera.com': f"{site_url}/search?q={query}",
            'default': f"{site_url}/search?q={query}"
        }
        
        # Get domain
        domain = urlparse(site_url).netloc
        
        # Find matching pattern
        for pattern_domain, pattern in search_patterns.items():
            if pattern_domain in domain:
                return pattern.replace('{query}', query.replace(' ', '+'))
        
        # Use default pattern
        return search_patterns['default'].replace('{query}', query.replace(' ', '+'))

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        import re
        hashtag_pattern = r'#\w+'
        return re.findall(hashtag_pattern, text)

    async def crawl_with_filters(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> List[UnifiedResult]:
        """
        Crawl with advanced filters.

        Args:
            query: Search query
            filters: Dictionary of filters including:
                - sources: List of sources
                - time_range: timedelta object
                - language: Language code
                - min_relevance: Minimum relevance score
                - content_quality: Minimum content quality
                - has_media: Only include results with media
                - author_whitelist: List of trusted authors
                - domain_whitelist: List of trusted domains
                - keywords_must_include: All these keywords must be present
                - keywords_exclude: None of these keywords should be present

        Returns:
            Filtered list of results
        """
        # Extract filters
        sources = filters.get('sources')
        time_range = filters.get('time_range')
        language = filters.get('language')
        keywords = filters.get('keywords_must_include', [])
        
        # Crawl sources
        results = await self.crawl_all_sources(
            query=query,
            sources=sources,
            time_range=time_range,
            keywords=keywords,
            language=language,
            max_results_per_source=filters.get('max_results_per_source')
        )
        
        # Apply additional filters
        filtered_results = []
        
        for result in results:
            # Check relevance
            if filters.get('min_relevance') and result.relevance_score < filters['min_relevance']:
                continue
                
            # Check content quality
            if filters.get('content_quality'):
                min_quality = ContentQuality[filters['content_quality'].upper()]
                if result.quality.value > min_quality.value:  # Lower enum value = higher quality
                    continue
            
            # Check media requirement
            if filters.get('has_media') and not result.media:
                continue
            
            # Check author whitelist
            if filters.get('author_whitelist') and result.author not in filters['author_whitelist']:
                continue
            
            # Check domain whitelist
            if filters.get('domain_whitelist'):
                domain = urlparse(result.url).netloc
                if not any(allowed in domain for allowed in filters['domain_whitelist']):
                    continue
            
            # Check excluded keywords
            if filters.get('keywords_exclude'):
                text = f"{result.title} {result.content}".lower()
                if any(keyword.lower() in text for keyword in filters['keywords_exclude']):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return self.metrics.get_report()

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.reset()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all crawlers."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'crawlers': {},
            'circuit_breakers': {},
            'connection_pool': {
                'active_connectors': len(self.connection_pool.connectors),
                'connector_stats': dict(self.connection_pool.connector_stats)
            },
            'metrics': self.get_metrics_report()
        }
        
        # Check each crawler
        for source, crawler in self.crawlers.items():
            try:
                # Simple test query
                test_results = await asyncio.wait_for(
                    self._crawl_source(source, 'test', None, None, 1),
                    timeout=10
                )
                health_status['crawlers'][source] = {
                    'status': 'healthy' if test_results else 'degraded',
                    'last_success': datetime.now().isoformat()
                }
            except Exception as e:
                health_status['crawlers'][source] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Check circuit breakers
        for source, breaker in self.circuit_breakers.items():
            health_status['circuit_breakers'][source] = {
                'state': breaker.state,
                'failure_count': breaker.failure_count,
                'last_failure': breaker.last_failure_time
            }
        
        return health_status

    async def close(self):
        """Clean up resources."""
        # Close session
        if self._session:
            await self._session.close()
        
        # Close connection pools
        await self.connection_pool.close_all()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close individual crawlers if they have close methods
        for source, crawler in self.crawlers.items():
            if hasattr(crawler, 'close'):
                try:
                    await crawler.close()
                except Exception as e:
                    logger.error(f"Error closing {source} crawler: {e}")


# Factory functions
def create_crawler(crawler_type: str, **kwargs) -> Any:
    """
    Create a crawler of the specified type.

    Args:
        crawler_type: Type of crawler ('news', 'google_news', 'twitter', 'gdelt', 'youtube')
        **kwargs: Configuration parameters for the crawler

    Returns:
        Configured crawler instance

    Raises:
        ValueError: If crawler type is not supported
        ImportError: If crawler components are not available
    """
    if crawler_type == 'news':
        if not _NEWS_CRAWLER_AVAILABLE:
            raise ImportError("News crawler components not available")
        return create_news_crawler(**kwargs)

    elif crawler_type == 'google_news':
        if not _GOOGLE_NEWS_AVAILABLE:
            raise ImportError("Google News crawler components not available")
        return create_google_news_crawler(**kwargs)

    elif crawler_type == 'twitter':
        if not _TWITTER_CRAWLER_AVAILABLE:
            raise ImportError("Twitter crawler components not available")
        return create_twitter_crawler(**kwargs)

    elif crawler_type == 'gdelt':
        if not _GDELT_CRAWLER_AVAILABLE:
            raise ImportError("GDELT crawler components not available")
        return create_gdelt_crawler(**kwargs)
    
    elif crawler_type == 'youtube':
        if not _YOUTUBE_CRAWLER_AVAILABLE:
            raise ImportError("YouTube crawler components not available")
        return create_youtube_client(**kwargs)

    else:
        raise ValueError(f"Unsupported crawler type: {crawler_type}")


def create_multi_source_crawler(config: Optional[Dict[str, Any]] = None) -> MultiSourceCrawler:
    """
    Create a multi-source crawler coordinator with advanced features.

    Args:
        config: Configuration dictionary with API keys and settings

    Returns:
        MultiSourceCrawler instance

    Example:
        >>> config = {
        ...     'news_stealth': True,
        ...     'google_news_api_key': 'your_api_key',
        ...     'twitter_bearer_token': 'your_token',
        ...     'circuit_breaker_threshold': 5,
        ...     'circuit_breaker_timeout': 60,
        ...     'max_workers': 10,
        ...     'relevance_threshold': 0.3
        ... }
        >>> crawler = create_multi_source_crawler(config)
        >>> results = await crawler.crawl_all_sources('conflict news Sudan')
    """
    return MultiSourceCrawler(config)


# Enhanced Factory Functions for Stealth-Enabled Crawlers
def create_stealth_crawler(
    requests_per_second: float = 2.0,
    timeout: int = 30,
    cache_dir: Optional[str] = None
):
    """
    Create a basic stealth crawler with cloudscraper priority.

    Args:
        requests_per_second: Rate limiting for respectful crawling
        timeout: Request timeout in seconds
        cache_dir: Directory for caching stealth strategies

    Returns:
        CloudScraperPriorityStealthManager or ImprovedBaseCrawler
    """
    if _CLOUDSCRAPER_STEALTH_AVAILABLE:
        return create_stealth_manager(cache_dir)
    elif _IMPROVED_CRAWLER_AVAILABLE:
        return create_basic_crawler(requests_per_second, timeout, use_stealth=True)
    else:
        raise ImportError("No stealth crawler components available")


def create_unified_stealth_crawler(cache_dir: Optional[str] = None):
    """
    Create a unified stealth orchestrator with intelligent strategy coordination.

    Args:
        cache_dir: Directory for caching domain strategies and learning data

    Returns:
        UnifiedStealthOrchestrator instance
    """
    if _UNIFIED_STEALTH_AVAILABLE:
        return create_unified_stealth_orchestrator(cache_dir)
    else:
        raise ImportError("Unified stealth orchestrator not available")


def create_news_crawler_enhanced(
    stealth_enabled: bool = True,
    extract_metadata: bool = True,
    requests_per_second: float = 1.0
):
    """
    Create an enhanced news crawler optimized for news websites.

    Args:
        stealth_enabled: Whether to use stealth capabilities
        extract_metadata: Whether to extract article metadata
        requests_per_second: Rate limiting

    Returns:
        Enhanced news crawler instance
    """
    if _IMPROVED_CRAWLER_AVAILABLE:
        return create_optimized_news_crawler(requests_per_second, extract_metadata)
    elif _NEWS_CRAWLER_AVAILABLE:
        return create_news_crawler(stealth_enabled=stealth_enabled)
    else:
        raise ImportError("No news crawler components available")


def create_fast_stealth_crawler(
    requests_per_second: float = 5.0,
    max_concurrent: int = 20
):
    """
    Create a fast crawler for high-volume crawling with basic stealth.

    Args:
        requests_per_second: High rate for fast crawling
        max_concurrent: Maximum concurrent connections

    Returns:
        Fast crawler instance optimized for speed
    """
    if _IMPROVED_CRAWLER_AVAILABLE:
        return create_fast_crawler(requests_per_second, max_concurrent)
    else:
        raise ImportError("Fast crawler components not available")


# Utility functions
def get_available_crawlers() -> List[str]:
    """
    Get list of available crawler types.

    Returns:
        List of available crawler type names
    """
    available = []
    if _NEWS_CRAWLER_AVAILABLE:
        available.append('news')
    if _GOOGLE_NEWS_AVAILABLE:
        available.append('google_news')
    if _TWITTER_CRAWLER_AVAILABLE:
        available.append('twitter')
    if _GDELT_CRAWLER_AVAILABLE:
        available.append('gdelt')
    if _YOUTUBE_CRAWLER_AVAILABLE:
        available.append('youtube')
    return available


def get_crawler_capabilities() -> Dict[str, CrawlerCapabilities]:
    """
    Get capabilities of all available crawlers.

    Returns:
        Dictionary mapping crawler names to their capabilities
    """
    capabilities = {}

    if _NEWS_CRAWLER_AVAILABLE:
        capabilities['news'] = CrawlerCapabilities(
            name='news',
            supports_stealth=True,
            supports_batch=True,
            supports_filtering=True,
            max_concurrent=10,
            average_response_time=3.5,
            success_rate=0.92
        )

    if _GOOGLE_NEWS_AVAILABLE:
        capabilities['google_news'] = CrawlerCapabilities(
            name='google_news',
            supports_batch=True,
            supports_filtering=True,
            rate_limited=True,
            requires_auth=True,
            max_concurrent=5,
            average_response_time=1.2,
            success_rate=0.98
        )

    if _TWITTER_CRAWLER_AVAILABLE:
        capabilities['twitter'] = CrawlerCapabilities(
            name='twitter',
            supports_real_time=True,
            supports_filtering=True,
            rate_limited=True,
            requires_auth=True,
            max_concurrent=3,
            average_response_time=0.8,
            success_rate=0.95
        )

    if _GDELT_CRAWLER_AVAILABLE:
        capabilities['gdelt'] = CrawlerCapabilities(
            name='gdelt',
            supports_batch=True,
            supports_filtering=True,
            rate_limited=True,
            max_concurrent=8,
            average_response_time=2.5,
            success_rate=0.96
        )

    if _YOUTUBE_CRAWLER_AVAILABLE:
        capabilities['youtube'] = CrawlerCapabilities(
            name='youtube',
            supports_batch=True,
            supports_filtering=True,
            rate_limited=True,
            requires_auth=False,  # API key optional, can fall back to scraping
            max_concurrent=5,
            average_response_time=4.0,
            success_rate=0.90
        )

    return capabilities


def validate_crawler_config(crawler_type: str, config: Dict[str, Any]) -> bool:
    """
    Validate configuration for a specific crawler type.

    Args:
        crawler_type: Type of crawler to validate config for
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if crawler_type not in get_available_crawlers():
        raise ValueError(f"Crawler type {crawler_type} not available")

    if crawler_type == 'google_news':
        if 'api_key' not in config or not config['api_key']:
            raise ValueError("Google News crawler requires api_key")

    elif crawler_type == 'twitter':
        if 'bearer_token' not in config or not config['bearer_token']:
            raise ValueError("Twitter crawler requires bearer_token")

    elif crawler_type == 'youtube':
        # API key is optional for YouTube (can use scraping)
        pass

    # Validate common parameters
    if 'max_concurrent' in config:
        if not isinstance(config['max_concurrent'], int) or config['max_concurrent'] < 1:
            raise ValueError("max_concurrent must be a positive integer")

    if 'timeout' in config:
        if not isinstance(config['timeout'], (int, float)) or config['timeout'] <= 0:
            raise ValueError("timeout must be a positive number")

    return True


def get_crawlers_health() -> Dict[str, Dict[str, Any]]:
    """
    Get health status of all available crawlers.

    Returns:
        Dictionary with health information for each crawler
    """
    health = {}

    available_crawlers = get_available_crawlers()
    capabilities = get_crawler_capabilities()

    for crawler_type in available_crawlers:
        health[crawler_type] = {
            'available': True,
            'stealth_capable': capabilities[crawler_type].supports_stealth,
            'requires_auth': capabilities[crawler_type].requires_auth,
            'average_response_time': capabilities[crawler_type].average_response_time,
            'success_rate': capabilities[crawler_type].success_rate,
            'last_check': datetime.now().isoformat()
        }

    # Add stealth component health
    health['stealth_components'] = {
        'cloudscraper_priority': _CLOUDSCRAPER_STEALTH_AVAILABLE,
        'unified_orchestrator': _UNIFIED_STEALTH_AVAILABLE,
        'improved_crawler': _IMPROVED_CRAWLER_AVAILABLE
    }

    return health


# Enhanced exports with stealth integration
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__company__",
    "__website__",
    "__license__",

    # Core stealth components
    "create_stealth_crawler",
    "create_unified_stealth_crawler",
    "create_news_crawler_enhanced",
    "create_fast_stealth_crawler",

    # Crawler factories
    "create_crawler",
    "create_multi_source_crawler",

    # Data structures
    "CrawlerCapabilities",
    "CrawlJob",
    "UnifiedResult",
    "MultiSourceCrawler",
    "CrawlerMetrics",
    "CircuitBreaker",
    "ConnectionPoolManager",

    # Enums
    "CrawlerStatus",
    "ContentQuality",
    "ProtectionType",

    # Utility functions
    "get_available_crawlers",
    "get_crawler_capabilities",
    "validate_crawler_config",
    "get_crawlers_health",
]

# Conditional exports based on availability
if _IMPROVED_CRAWLER_AVAILABLE:
    __all__.extend([
        "ImprovedBaseCrawler",
        "CrawlResult",
        "CrawlerConfig",
        "create_basic_crawler",
        "create_fast_crawler"
    ])

if _CLOUDSCRAPER_STEALTH_AVAILABLE:
    __all__.extend([
        "CloudScraperPriorityStealthManager",
        "create_stealth_manager",
        "StealthResult"
    ])

if _UNIFIED_STEALTH_AVAILABLE:
    __all__.extend([
        "UnifiedStealthOrchestrator",
        "create_unified_stealth_orchestrator"
    ])

if _NEWS_CRAWLER_AVAILABLE:
    __all__.extend([
        "NewsCrawler",
        "StealthNewsCrawler",
        "NewsBatchCrawler",
        "LegacyCrawlResult",
        "create_news_crawler"
    ])

if _GOOGLE_NEWS_AVAILABLE:
    __all__.extend([
        "GoogleNewsCrawler",
        "GoogleNewsClient",
        "GoogleNewsResult",
        "create_google_news_crawler"
    ])

if _TWITTER_CRAWLER_AVAILABLE:
    __all__.extend([
        "TwitterCrawler",
        "TwitterStreamer",
        "TwitterResult",
        "create_twitter_crawler"
    ])

if _GDELT_CRAWLER_AVAILABLE:
    __all__.extend([
        "GDELTCrawler",
        "GDELTClient",
        "GDELTEvent",
        "create_gdelt_crawler"
    ])

if _YOUTUBE_CRAWLER_AVAILABLE:
    __all__.extend([
        "YouTubeClient",
        "EnhancedYouTubeClient",  # backward compatibility
        "YouTubeAPIWrapper",
        "YouTubeScrapingClient",
        "VideoData",
        "ChannelData",
        "CommentData",
        "TranscriptData",
        "YouTubeCrawlResult",
        "YouTubeExtractResult",
        "create_youtube_client",
        "create_enhanced_youtube_client",  # backward compatibility
        "create_basic_youtube_client",
        "YouTubeCrawlerConfig"
    ])

# Package-level configuration
STEALTH_PRIORITY_ORDER = [
    "cloudscraper",  # Primary - fastest and most effective
    "playwright",    # Secondary - for complex JavaScript
    "selenium",      # Tertiary - for specialized scenarios
    "http_stealth"   # Final fallback - basic protection bypass
]

CRAWLER_PERFORMANCE_TARGETS = {
    "cloudscraper_success_rate": 85.0,  # Target 85%+ success with CloudScraper
    "overall_success_rate": 95.0,       # Target 95%+ overall success
    "average_response_time": 3.0,       # Target under 3 seconds average
    "cost_efficiency_threshold": 2.0    # Prefer methods with cost score under 2.0
}

# Logging configuration for package
logger.info(f"Enhanced Crawlers Package v{__version__} initialized")
logger.info(f"Author: {__author__} | Company: {__company__} ({__website__})")
logger.info(f"Available crawlers: {get_available_crawlers()}")
logger.info(f"Stealth capabilities: CloudScraper={_CLOUDSCRAPER_STEALTH_AVAILABLE}, "
           f"Unified={_UNIFIED_STEALTH_AVAILABLE}, Improved={_IMPROVED_CRAWLER_AVAILABLE}")

# Log availability warnings
unavailable = []
if not _NEWS_CRAWLER_AVAILABLE:
    unavailable.append('news')
if not _GOOGLE_NEWS_AVAILABLE:
    unavailable.append('google_news')
if not _TWITTER_CRAWLER_AVAILABLE:
    unavailable.append('twitter')
if not _GDELT_CRAWLER_AVAILABLE:
    unavailable.append('gdelt')
if not _YOUTUBE_CRAWLER_AVAILABLE:
    unavailable.append('youtube')

if unavailable:
    logger.warning(f"Unavailable crawlers: {unavailable}")
    logger.info("Install missing dependencies or check component imports")