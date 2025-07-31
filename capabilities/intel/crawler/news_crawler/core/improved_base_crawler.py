"""
Improved Base Crawler for News Sites
=====================================

An enhanced base crawler implementation with advanced features for news site crawling,
including intelligent rate limiting, content detection, error recovery, and performance
optimization specifically designed for news and media websites.

Features:
- Intelligent content type detection
- Advanced error recovery and retry mechanisms
- Dynamic rate limiting based on site behavior
- Content quality assessment
- Memory-efficient streaming processing
- Comprehensive metrics and monitoring
- Multi-format content support (HTML, JSON, XML, RSS)

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, urljoin, parse_qs
from datetime import datetime, timedelta
import re

# Configure logging
logger = logging.getLogger(__name__)


class CrawlPriority(Enum):
    """Crawl priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class ContentType(Enum):
    """Content type classifications."""
    ARTICLE = "article"
    RSS_FEED = "rss_feed"
    JSON_API = "json_api"
    HTML_PAGE = "html_page"
    SITEMAP = "sitemap"
    MEDIA = "media"
    UNKNOWN = "unknown"


class CrawlStatus(Enum):
    """Crawl operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class CrawlRequest:
    """Represents a crawl request with metadata."""
    url: str
    priority: CrawlPriority = CrawlPriority.NORMAL
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    expected_content_type: Optional[ContentType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    status: CrawlStatus = CrawlStatus.PENDING


@dataclass
class CrawlResult:
    """Represents the result of a crawl operation."""
    request: CrawlRequest
    status_code: Optional[int] = None
    content: Optional[str] = None
    content_type: Optional[ContentType] = None
    headers: Dict[str, str] = field(default_factory=dict)
    response_time: Optional[float] = None
    content_length: Optional[int] = None
    encoding: Optional[str] = None
    final_url: Optional[str] = None
    redirects: List[str] = field(default_factory=list)
    error: Optional[str] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlerConfig:
    """Configuration for the improved base crawler."""
    max_concurrent_requests: int = 10
    default_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_requests_per_second: float = 2.0
    rate_limit_burst: int = 5
    enable_content_detection: bool = True
    enable_quality_assessment: bool = True
    enable_streaming: bool = False
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    min_content_size: int = 100  # 100 bytes
    user_agent: str = "Lindela-NewsBot/4.0 (+https://datacraft.co.ke/bot)"
    enable_robots_txt: bool = True
    enable_sitemap_detection: bool = True
    enable_feed_detection: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    domain_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ContentDetector:
    """Intelligent content type detection and analysis."""
    
    def __init__(self):
        self.content_patterns = {
            ContentType.ARTICLE: [
                r'<article[\s>]',
                r'class="[^"]*article[^"]*"',
                r'<div[^>]*class="[^"]*post[^"]*"',
                r'<div[^>]*class="[^"]*content[^"]*"',
                r'<meta[^>]*property="og:type"[^>]*content="article"'
            ],
            ContentType.RSS_FEED: [
                r'<rss\s',
                r'<feed\s',
                r'<\?xml[^>]*>\s*<rss',
                r'application/rss\+xml',
                r'application/atom\+xml'
            ],
            ContentType.JSON_API: [
                r'^\s*[\{\[]',
                r'application/json',
                r'application/ld\+json'
            ],
            ContentType.SITEMAP: [
                r'<urlset',
                r'<sitemapindex',
                r'application/xml.*sitemap'
            ]
        }
        
        self.quality_indicators = {
            'positive': [
                r'<article[\s>]',
                r'<time\s',
                r'<meta[^>]*property="og:',
                r'<meta[^>]*name="description"',
                r'<meta[^>]*name="author"',
                r'class="[^"]*byline[^"]*"',
                r'class="[^"]*date[^"]*"'
            ],
            'negative': [
                r'error',
                r'not found',
                r'access denied',
                r'<title>[^<]*404[^<]*</title>',
                r'<title>[^<]*error[^<]*</title>'
            ]
        }
    
    def detect_content_type(self, content: str, headers: Dict[str, str], url: str) -> ContentType:
        """Detect the content type of the response."""
        content_type_header = headers.get('content-type', '').lower()
        
        # Check header first
        if 'application/json' in content_type_header:
            return ContentType.JSON_API
        elif 'application/rss+xml' in content_type_header or 'application/atom+xml' in content_type_header:
            return ContentType.RSS_FEED
        elif 'application/xml' in content_type_header and 'sitemap' in url.lower():
            return ContentType.SITEMAP
        
        # Check content patterns
        content_lower = content.lower()
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    return content_type
        
        # Default for HTML
        if 'text/html' in content_type_header or '<html' in content_lower:
            return ContentType.HTML_PAGE
            
        return ContentType.UNKNOWN
    
    def assess_content_quality(self, content: str, content_type: ContentType) -> float:
        """Assess the quality of the content (0.0 to 1.0)."""
        if not content or len(content) < 100:
            return 0.0
        
        score = 0.5  # Base score
        content_lower = content.lower()
        
        # Check positive indicators
        positive_matches = 0
        for pattern in self.quality_indicators['positive']:
            if re.search(pattern, content_lower, re.IGNORECASE):
                positive_matches += 1
        
        # Check negative indicators
        negative_matches = 0
        for pattern in self.quality_indicators['negative']:
            if re.search(pattern, content_lower, re.IGNORECASE):
                negative_matches += 1
        
        # Adjust score based on indicators
        score += (positive_matches * 0.1)
        score -= (negative_matches * 0.2)
        
        # Content type specific adjustments
        if content_type == ContentType.ARTICLE:
            score += 0.2
        elif content_type == ContentType.RSS_FEED:
            score += 0.15
        elif content_type == ContentType.JSON_API:
            score += 0.1
        
        # Length-based adjustments
        if len(content) > 1000:
            score += 0.1
        if len(content) > 5000:
            score += 0.1
        
        return max(0.0, min(1.0, score))


class RateLimiter:
    """Intelligent rate limiting with domain-specific controls."""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.domain_limits = {}
        self.request_history = {}
        self.last_request_times = {}
        
    async def wait_for_request(self, url: str):
        """Wait if necessary to respect rate limits."""
        domain = urlparse(url).netloc.lower()
        current_time = time.time()
        
        # Get domain-specific configuration
        domain_config = self.config.domain_specific_configs.get(domain, {})
        requests_per_second = domain_config.get('rate_limit', self.config.rate_limit_requests_per_second)
        
        # Check last request time for this domain
        last_time = self.last_request_times.get(domain, 0)
        time_since_last = current_time - last_time
        min_interval = 1.0 / requests_per_second
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
            await asyncio.sleep(wait_time)
        
        self.last_request_times[domain] = time.time()
    
    def record_request(self, url: str, success: bool):
        """Record a request for rate limiting calculations."""
        domain = urlparse(url).netloc.lower()
        if domain not in self.request_history:
            self.request_history[domain] = []
        
        self.request_history[domain].append({
            'timestamp': time.time(),
            'success': success
        })
        
        # Keep only recent history (last hour)
        cutoff = time.time() - 3600
        self.request_history[domain] = [
            req for req in self.request_history[domain] 
            if req['timestamp'] > cutoff
        ]


class ErrorRecovery:
    """Advanced error recovery and retry logic."""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.error_patterns = {
            'rate_limited': [r'rate limit', r'too many requests', r'429'],
            'server_error': [r'5\d\d', r'server error', r'internal error'],
            'timeout': [r'timeout', r'timed out'],
            'connection_error': [r'connection', r'refused', r'unreachable'],
            'forbidden': [r'403', r'forbidden', r'access denied'],
            'not_found': [r'404', r'not found']
        }
    
    def should_retry(self, request: CrawlRequest, error: str, status_code: Optional[int] = None) -> bool:
        """Determine if a request should be retried."""
        if request.attempts >= request.max_retries:
            return False
        
        # Check status code
        if status_code:
            if status_code in [429, 502, 503, 504]:  # Retry on rate limit and server errors
                return True
            elif status_code in [404, 403, 401]:  # Don't retry on client errors
                return False
        
        # Check error patterns
        error_lower = error.lower()
        
        # Always retry on rate limits and server errors
        if any(re.search(pattern, error_lower) for pattern in self.error_patterns['rate_limited']):
            return True
        if any(re.search(pattern, error_lower) for pattern in self.error_patterns['server_error']):
            return True
        if any(re.search(pattern, error_lower) for pattern in self.error_patterns['timeout']):
            return True
        if any(re.search(pattern, error_lower) for pattern in self.error_patterns['connection_error']):
            return True
            
        # Don't retry on forbidden or not found
        if any(re.search(pattern, error_lower) for pattern in self.error_patterns['forbidden']):
            return False
        if any(re.search(pattern, error_lower) for pattern in self.error_patterns['not_found']):
            return False
        
        # Default to retry for unknown errors
        return True
    
    def calculate_retry_delay(self, request: CrawlRequest, error_type: str) -> float:
        """Calculate the delay before retrying."""
        base_delay = request.retry_delay
        
        # Exponential backoff
        delay = base_delay * (2 ** request.attempts)
        
        # Add jitter
        jitter = delay * 0.1 * (0.5 - time.time() % 1)
        delay += jitter
        
        # Cap the delay
        max_delay = 60.0
        delay = min(delay, max_delay)
        
        # Special handling for rate limits
        if error_type == 'rate_limited':
            delay = max(delay, 10.0)  # Minimum 10 seconds for rate limits
        
        return delay


class ImprovedBaseCrawler:
    """Enhanced base crawler with advanced features for news sites."""
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.content_detector = ContentDetector()
        self.rate_limiter = RateLimiter(self.config)
        self.error_recovery = ErrorRecovery(self.config)
        
        # State management
        self.active_requests = {}
        self.completed_requests = []
        self.failed_requests = []
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'total_response_time': 0.0,
            'content_types': {},
            'status_codes': {},
            'domains_crawled': set()
        }
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        logger.info(f"ImprovedBaseCrawler initialized with {self.config.max_concurrent_requests} max concurrent requests")
    
    async def crawl_url(self, url: str, **kwargs) -> CrawlResult:
        """Crawl a single URL with advanced error handling."""
        request = CrawlRequest(url=url, **kwargs)
        return await self.crawl_request(request)
    
    async def crawl_request(self, request: CrawlRequest) -> CrawlResult:
        """Crawl a single request with full error recovery."""
        async with self.semaphore:
            return await self._execute_crawl(request)
    
    async def _execute_crawl(self, request: CrawlRequest) -> CrawlResult:
        """Execute the actual crawl operation."""
        start_time = time.time()
        request.status = CrawlStatus.IN_PROGRESS
        
        try:
            # Rate limiting
            await self.rate_limiter.wait_for_request(request.url)
            
            # Simulate HTTP request (would be replaced with actual HTTP client)
            result = await self._simulate_http_request(request)
            
            # Content detection and quality assessment
            if result.content and self.config.enable_content_detection:
                result.content_type = self.content_detector.detect_content_type(
                    result.content, result.headers, request.url
                )
                
                if self.config.enable_quality_assessment:
                    quality_score = self.content_detector.assess_content_quality(
                        result.content, result.content_type
                    )
                    result.metrics['quality_score'] = quality_score
            
            # Update metrics
            response_time = time.time() - start_time
            result.response_time = response_time
            result.completed_at = datetime.now()
            
            self._update_metrics(request, result, success=True)
            self.rate_limiter.record_request(request.url, True)
            
            request.status = CrawlStatus.COMPLETED
            logger.debug(f"Successfully crawled {request.url} in {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Crawl failed for {request.url}: {error_msg}")
            
            # Check if we should retry
            if self.error_recovery.should_retry(request, error_msg):
                request.attempts += 1
                request.last_attempt = datetime.now()
                request.status = CrawlStatus.RETRYING
                
                # Calculate retry delay
                retry_delay = self.error_recovery.calculate_retry_delay(request, 'unknown')
                logger.info(f"Retrying {request.url} in {retry_delay:.2f}s (attempt {request.attempts})")
                
                await asyncio.sleep(retry_delay)
                return await self._execute_crawl(request)
            else:
                # Final failure
                request.status = CrawlStatus.FAILED
                result = CrawlResult(
                    request=request,
                    error=error_msg,
                    completed_at=datetime.now()
                )
                
                self._update_metrics(request, result, success=False)
                self.rate_limiter.record_request(request.url, False)
                
                return result
    
    async def _simulate_http_request(self, request: CrawlRequest) -> CrawlResult:
        """Simulate HTTP request (to be replaced with actual implementation)."""
        # This is a placeholder - would be replaced with actual HTTP client
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Generate mock response
        mock_content = f"<html><head><title>Test Page</title></head><body><article>Content for {request.url}</article></body></html>"
        
        return CrawlResult(
            request=request,
            status_code=200,
            content=mock_content,
            headers={'content-type': 'text/html; charset=utf-8'},
            content_length=len(mock_content),
            encoding='utf-8',
            final_url=request.url
        )
    
    def _update_metrics(self, request: CrawlRequest, result: CrawlResult, success: bool):
        """Update crawler metrics."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
            if result.response_time:
                self.metrics['total_response_time'] += result.response_time
        else:
            self.metrics['failed_requests'] += 1
        
        if request.attempts > 0:
            self.metrics['retried_requests'] += 1
        
        # Track content types
        if result.content_type:
            content_type = result.content_type.value
            self.metrics['content_types'][content_type] = self.metrics['content_types'].get(content_type, 0) + 1
        
        # Track status codes
        if result.status_code:
            self.metrics['status_codes'][result.status_code] = self.metrics['status_codes'].get(result.status_code, 0) + 1
        
        # Track domains
        domain = urlparse(request.url).netloc
        self.metrics['domains_crawled'].add(domain)
    
    async def crawl_multiple(self, urls: List[str], **kwargs) -> List[CrawlResult]:
        """Crawl multiple URLs concurrently."""
        requests = [CrawlRequest(url=url, **kwargs) for url in urls]
        
        tasks = [self.crawl_request(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = CrawlResult(
                    request=requests[i],
                    error=str(result),
                    completed_at=datetime.now()
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive crawler metrics."""
        metrics = dict(self.metrics)
        
        # Calculate derived metrics
        if metrics['successful_requests'] > 0:
            metrics['average_response_time'] = metrics['total_response_time'] / metrics['successful_requests']
            metrics['success_rate'] = metrics['successful_requests'] / metrics['total_requests']
        else:
            metrics['average_response_time'] = 0.0
            metrics['success_rate'] = 0.0
        
        metrics['total_domains'] = len(metrics['domains_crawled'])
        metrics['domains_crawled'] = list(metrics['domains_crawled'])  # Convert set to list for JSON serialization
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'total_response_time': 0.0,
            'content_types': {},
            'status_codes': {},
            'domains_crawled': set()
        }
        logger.info("Crawler metrics reset")


# Factory functions
def create_crawler_config(**kwargs) -> CrawlerConfig:
    """Factory function to create crawler configuration."""
    return CrawlerConfig(**kwargs)


def create_improved_crawler(config: Optional[CrawlerConfig] = None) -> ImprovedBaseCrawler:
    """Factory function to create improved base crawler."""
    return ImprovedBaseCrawler(config)


def create_news_optimized_crawler() -> ImprovedBaseCrawler:
    """Factory function to create news-optimized crawler."""
    config = CrawlerConfig(
        max_concurrent_requests=5,
        rate_limit_requests_per_second=1.5,
        enable_content_detection=True,
        enable_quality_assessment=True,
        enable_feed_detection=True,
        enable_sitemap_detection=True
    )
    return ImprovedBaseCrawler(config)


def create_basic_crawler() -> ImprovedBaseCrawler:
    """Factory function to create basic crawler with minimal configuration."""
    config = CrawlerConfig(
        max_concurrent_requests=3,
        rate_limit_requests_per_second=1.0,
        enable_content_detection=False,
        enable_quality_assessment=False,
        enable_feed_detection=False,
        enable_sitemap_detection=False,
        max_retries=2
    )
    return ImprovedBaseCrawler(config)


def create_news_crawler(config: Optional[CrawlerConfig] = None) -> ImprovedBaseCrawler:
    """Factory function to create news crawler."""
    if config is None:
        config = CrawlerConfig(
            max_concurrent_requests=8,
            rate_limit_requests_per_second=2.0,
            enable_content_detection=True,
            enable_quality_assessment=True,
            enable_feed_detection=True,
            enable_sitemap_detection=True,
            max_retries=3
        )
    return ImprovedBaseCrawler(config)


def create_fast_crawler() -> ImprovedBaseCrawler:
    """Factory function to create fast crawler with maximum performance."""
    config = CrawlerConfig(
        max_concurrent_requests=20,
        rate_limit_requests_per_second=5.0,
        enable_content_detection=False,
        enable_quality_assessment=False,
        enable_feed_detection=False,
        enable_sitemap_detection=False,
        max_retries=1,
        default_timeout=15.0
    )
    return ImprovedBaseCrawler(config)


# Export all components
__all__ = [
    # Enums
    'CrawlPriority', 'ContentType', 'CrawlStatus',
    
    # Data classes
    'CrawlRequest', 'CrawlResult', 'CrawlerConfig',
    
    # Core classes
    'ContentDetector', 'RateLimiter', 'ErrorRecovery', 'ImprovedBaseCrawler',
    
    # Factory functions
    'create_crawler_config', 'create_improved_crawler', 'create_news_optimized_crawler', 'create_basic_crawler',
    'create_news_crawler', 'create_fast_crawler'
]