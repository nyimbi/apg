"""
Fixed GDELT DOC 2.0 API Client with Content Download and Timespan Support

This implementation fixes:
1. CSV parsing to handle actual GDELT CSV format (URL, MobileURL, Date, Title)
2. Domain field made optional and auto-extracted from URL
3. Content downloading with HTML to markdown conversion
4. Timespan parameter support alongside datetime parameters
5. Robust error handling and validation

Author: Advanced Data Engineering Framework
Version: 4.0.0
"""

import asyncio
import aiohttp
import csv
import io
import logging
import hashlib
import re
import httpx
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple, Any, Union, AsyncIterator, Set
from urllib.parse import urlencode, quote, unquote, urlparse
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import time
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, field_validator, HttpUrl, ValidationError, ConfigDict
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)

# HTML to Markdown conversion
try:
    from markdownify import markdownify as md
except ImportError:
    # Fallback if markdownify not available
    def md(html, **kwargs):
        """Simple HTML to text fallback."""
        import re
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        html = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        html = re.sub(r'\s+', ' ', html)
        return html.strip()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)


class GDELTMode(str, Enum):
    """GDELT API query modes."""
    ARTLIST = "artlist"
    TIMELINEVOL = "timelinevol"
    TIMELINETONE = "timelinetone"
    TIMELINELANG = "timelinelang"
    TIMELINESOURCECOUNTRY = "timelinesourcecountry"
    TONECHART = "tonechart"
    TIMELINE = "timeline"


class GDELTFormat(str, Enum):
    """Output format options."""
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    RSS = "rss"
    JSONFEED = "jsonfeed"


class GDELTSort(str, Enum):
    """Sorting options."""
    DATE = "date"
    REL = "rel"
    HYBRID = "hybridrel"


class GDELTDedup(str, Enum):
    """Deduplication options."""
    NO_DEDUP = "0"
    DEDUP_TITLE = "1"
    DEDUP_URL = "2"
    DEDUP_FULL = "3"


class GDELTImageFacet(str, Enum):
    """Image faceting options."""
    INCLUDE = "include"
    ONLY = "only"
    EXCLUDE = "exclude"


@dataclass
class GDELTDateRange:
    """
    Represents a date range for GDELT queries with validation.
    """
    start_date: datetime
    end_date: datetime

    def __post_init__(self):
        """Validate date range constraints."""
        if self.start_date > self.end_date:
            raise ValueError(f"Start date {self.start_date} must be before end date {self.end_date}")

        # GDELT has data from 2017 onwards for DOC API
        gdelt_start = datetime(2017, 1, 1, tzinfo=timezone.utc)
        if self.start_date < gdelt_start:
            logger.warning(f"Start date {self.start_date} is before GDELT DOC availability. Adjusting to {gdelt_start}")
            self.start_date = gdelt_start

        # Ensure timezone awareness
        if self.start_date.tzinfo is None:
            self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        if self.end_date.tzinfo is None:
            self.end_date = self.end_date.replace(tzinfo=timezone.utc)

    async def iter_days(self):
        """Asynchronously iterate over days in the range."""
        current = self.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = self.end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        while current <= end:
            yield current
            current += timedelta(days=1)
            await asyncio.sleep(0)  # Yield control

    def to_gdelt_datetime_params(self, specific_date: Optional[datetime] = None) -> Dict[str, str]:
        """
        Convert to GDELT startdatetime/enddatetime parameters.

        Args:
            specific_date: If provided, return params for this specific date only

        Returns:
            Dictionary with startdatetime and enddatetime
        """
        if specific_date:
            start = specific_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = specific_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            start = self.start_date
            end = self.end_date

        return {
            'startdatetime': start.strftime("%Y%m%d%H%M%S"),
            'enddatetime': end.strftime("%Y%m%d%H%M%S")
        }

    @property
    def total_days(self) -> int:
        """Calculate total days in range."""
        return (self.end_date.date() - self.start_date.date()).days + 1


@dataclass
class GDELTQueryParameters:
    """
    GDELT query parameters with proper date handling and timespan support.
    """
    query: str
    mode: GDELTMode = GDELTMode.ARTLIST
    format: GDELTFormat = GDELTFormat.CSV
    timespan: Optional[str] = None  # For relative offsets (e.g., "7d", "24h")
    startdatetime: Optional[str] = None  # For specific dates
    enddatetime: Optional[str] = None  # For specific dates
    maxrecords: int = 250
    sort: GDELTSort = GDELTSort.HYBRID
    dropdup: GDELTDedup = GDELTDedup.DEDUP_TITLE
    translation: bool = False
    transpose: bool = False
    timelinesmooth: int = 0
    imagewebtagonly: bool = False
    imagefacetonly: Optional[GDELTImageFacet] = None
    domain: Optional[str] = None
    domainis: Optional[str] = None
    theme: Optional[str] = None
    near: Optional[Tuple[int, str]] = None
    repeat: Optional[str] = None

    def __post_init__(self):
        """Validate parameter constraints."""
        if self.maxrecords < 1 or self.maxrecords > 250:
            raise ValueError("maxrecords must be between 1 and 250")

        if self.timelinesmooth < 0:
            raise ValueError("timelinesmooth must be non-negative")

        # Ensure we don't use both timespan and datetime parameters
        if self.timespan and (self.startdatetime or self.enddatetime):
            raise ValueError("Cannot use both timespan and startdatetime/enddatetime")

    def set_datetime_params(self, startdatetime: str, enddatetime: str):
        """Set datetime parameters and clear timespan to avoid conflicts."""
        self.startdatetime = startdatetime
        self.enddatetime = enddatetime
        self.timespan = None  # Clear timespan to avoid parameter conflicts
        
    def set_timespan_param(self, timespan: str):
        """Set timespan parameter and clear datetime params to avoid conflicts."""
        self.timespan = timespan
        self.startdatetime = None
        self.enddatetime = None

    def ensure_time_params(self, default_timespan: str = "24h"):
        """Ensure time parameters are set, defaulting to timespan if none provided."""
        if not self.timespan and not self.startdatetime and not self.enddatetime:
            self.timespan = default_timespan
            logger.info(f"No time parameters specified, defaulting to timespan={default_timespan}")

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API request."""
        params = {
            "query": self.query,
            "mode": self.mode.value,
            "format": self.format.value,
            "maxrecords": str(self.maxrecords),
            "sort": self.sort.value,
            "dropdup": self.dropdup.value
        }

        # Add optional parameters
        if self.timespan:
            params["timespan"] = self.timespan
        if self.startdatetime:
            params["startdatetime"] = self.startdatetime
        if self.enddatetime:
            params["enddatetime"] = self.enddatetime
        if self.translation:
            params["translation"] = "1"
        if self.transpose:
            params["transpose"] = "1"
        if self.timelinesmooth > 0:
            params["timelinesmooth"] = str(self.timelinesmooth)
        if self.imagewebtagonly:
            params["imagewebtagonly"] = "1"
        if self.imagefacetonly:
            params["imagefacetonly"] = self.imagefacetonly.value
        if self.domain:
            params["domain"] = self.domain
        if self.domainis:
            params["domainis"] = self.domainis
        if self.theme:
            params["theme"] = self.theme
        if self.near:
            distance, location = self.near
            params["near"] = f"{distance}:{location}"
        if self.repeat:
            params["repeat"] = self.repeat

        return params


class GDELTArticle(BaseModel):
    """Pydantic model for GDELT article with content downloading."""
    url: HttpUrl
    url_mobile: Optional[HttpUrl] = Field(None, alias='mobileurl')
    title: str
    seendate: datetime = Field(alias='date')
    domain: Optional[str] = None  # Made optional, will be auto-extracted
    content: Optional[str] = None  # Downloaded content
    
    # Additional fields that might be present in enhanced responses
    socialimage: Optional[HttpUrl] = None
    language: Optional[str] = Field(None, max_length=10)
    sourcecountry: Optional[str] = Field(None, max_length=10)
    tone: Optional[float] = None
    pos_score: Optional[float] = None
    neg_score: Optional[float] = None
    polarity: Optional[float] = None
    activity_density: Optional[float] = None
    self_density: Optional[float] = None
    word_count: Optional[int] = None

    @field_validator('seendate', mode='before')
    @classmethod
    def parse_seendate(cls, v: Union[str, datetime]) -> datetime:
        """Parse GDELT datetime format."""
        if isinstance(v, datetime):
            return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v

        if not isinstance(v, str):
            raise ValueError(f"Unexpected date type: {type(v)}")

        # Handle various date formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S',     # Standard CSV format: 2025-04-12 12:30:00
            '%Y%m%d%H%M%S',          # GDELT format: 20250412123000
            '%Y-%m-%dT%H:%M:%SZ',    # ISO format with Z
            '%Y-%m-%dT%H:%M:%S',     # ISO format without Z
            '%Y-%m-%d',              # Date only
        ]

        for fmt in date_formats:
            try:
                dt = datetime.strptime(v, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # Try ISO format parsing
        try:
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            pass

        raise ValueError(f"Unparseable date format: {v}")

    @field_validator('title')
    @classmethod
    def sanitize_title(cls, v: str) -> str:
        """Sanitize title."""
        if not v:
            return "Untitled Article"
        
        import html
        v = html.unescape(v)
        v = ''.join(char for char in v if ord(char) >= 32 or char in '\n\t')
        v = re.sub(r'\s+', ' ', v)
        v = v.strip()
        
        return v if len(v) >= 3 else "Untitled Article"

    @field_validator('domain', mode='before')
    @classmethod
    def extract_domain(cls, v: Optional[str]) -> str:
        """Extract and normalize domain from URL if not provided."""
        if v and isinstance(v, str) and len(v) > 0:
            domain = v.lower().strip()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain

        return 'unknown'

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            HttpUrl: str
        }
    )


class ContentDownloader:
    """Handles downloading and converting article content."""
    
    def __init__(self, timeout: int = 30, max_content_length: int = 5 * 1024 * 1024):  # 5MB max
        """Initialize content downloader."""
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = None
        
    async def __aenter__(self):
        """Create HTTP session."""
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                'User-Agent': 'GDELT-Content-Downloader/1.0 (News Aggregator)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP session."""
        if self.session:
            await self.session.aclose()
    
    async def download_content(self, url: str) -> Optional[str]:
        """Download and convert article content to markdown."""
        try:
            logger.debug(f"Downloading content from: {url}")
            
            response = await self.session.get(url)
            response.raise_for_status()
            
            # Check content length
            content_length = len(response.content)
            if content_length > self.max_content_length:
                logger.warning(f"Content too large ({content_length} bytes), truncating")
                html_content = response.content[:self.max_content_length].decode('utf-8', errors='ignore')
            else:
                html_content = response.text
            
            # Convert HTML to markdown
            markdown_content = await self._html_to_markdown(html_content)
            
            # Clean and validate
            cleaned_content = self._clean_content(markdown_content)
            
            if len(cleaned_content) < 50:  # Too short to be useful
                logger.debug(f"Content too short after cleaning: {len(cleaned_content)} chars")
                return None
                
            logger.debug(f"Successfully downloaded content: {len(cleaned_content)} chars")
            return cleaned_content
            
        except httpx.TimeoutException:
            logger.debug(f"Timeout downloading content from: {url}")
            return None
        except httpx.HTTPStatusError as e:
            logger.debug(f"HTTP error {e.response.status_code} downloading from: {url}")
            return None
        except Exception as e:
            logger.debug(f"Error downloading content from {url}: {e}")
            return None
    
    async def _html_to_markdown(self, html_content: str) -> str:
        """Convert HTML to markdown asynchronously."""
        try:
            # Run the conversion in a thread pool to avoid blocking
            markdown = await asyncio.to_thread(
                md, 
                html_content,
                heading_style="ATX",
                bullets="-",
                strong_mark="**",
                em_mark="*",
                strip=['script', 'style', 'meta', 'link', 'noscript', 'nav', 'footer', 'header']
            )
            return markdown
        except Exception as e:
            logger.debug(f"Error converting HTML to markdown: {e}")
            # Fallback to simple text extraction
            return self._html_to_text_fallback(html_content)
    
    def _html_to_text_fallback(self, html_content: str) -> str:
        """Fallback HTML to text conversion."""
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Decode HTML entities
        import html
        html_content = html.unescape(html_content)
        
        # Clean up whitespace
        html_content = re.sub(r'\s+', ' ', html_content)
        
        return html_content.strip()
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces
        
        # Remove common boilerplate
        boilerplate_patterns = [
            r'cookie\s+policy',
            r'privacy\s+policy',
            r'terms\s+of\s+service',
            r'subscribe\s+to\s+our\s+newsletter',
            r'follow\s+us\s+on',
            r'share\s+this\s+article',
            r'advertisement',
            r'related\s+articles?',
        ]
        
        for pattern in boilerplate_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Trim to reasonable length (keep first ~5000 chars)
        if len(content) > 5000:
            content = content[:5000] + "..."
        
        return content.strip()


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                return wait_time
            else:
                self.tokens -= tokens
                return 0.0


class GDELTClient:
    """
    GDELT DOC 2.0 API client with content download and timespan support.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        rate_limit: float = 10.0,
        timeout: int = 60,
        max_concurrent: int = 5,
        cache_results: bool = True,
        download_content: bool = True,
        retry_config: Optional[Dict[str, Any]] = None
    ):
        self.rate_limiter = RateLimiter(rate=rate_limit, capacity=rate_limit * 2)
        self.timeout_seconds = timeout
        self.max_concurrent = max_concurrent
        self.cache_results = cache_results
        self.download_content = download_content

        # Defer connector creation until async context
        self.connector: Optional[aiohttp.TCPConnector] = None
        self.timeout: Optional[aiohttp.ClientTimeout] = None

        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.content_downloader: Optional[ContentDownloader] = None

        self._cache: Dict[str, Tuple[List[GDELTArticle], float]] = {}
        self._cache_ttl = 3600

        self.stats = defaultdict(int)
        
        # Store raw CSV data for saving
        self.last_raw_csv_data: Optional[str] = None

        self.retry_config = retry_config or {
            'stop': stop_after_attempt(3),
            'wait': wait_exponential(multiplier=2, min=4, max=30),
            'retry': retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            'before_sleep': before_sleep_log(logger, logging.WARNING)
        }

    async def __aenter__(self):
        """Initialize session and connector on context entry."""
        # Create connector in async context
        self.connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent,
            force_close=True,
            enable_cleanup_closed=True,
            ttl_dns_cache=300
        )

        # Create timeout object
        self.timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        # Create semaphore
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create session
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'GDELT-Advanced-Client/4.0.0',
                'Accept': 'text/csv, application/json, text/html, application/rss+xml',
                'Accept-Encoding': 'gzip, deflate, br',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )

        # Initialize content downloader if enabled
        if self.download_content:
            self.content_downloader = ContentDownloader()
            await self.content_downloader.__aenter__()

        logger.info("GDELT client session initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        await self.close()

    async def close(self):
        """Close the GDELT client and cleanup resources."""
        if self.content_downloader:
            await self.content_downloader.__aexit__(None, None, None)
        
        if self.session:
            await self.session.close()
            await asyncio.sleep(0.5)
        
        logger.info(f"GDELT client closed. Statistics: {dict(self.stats)}")

    def _get_cache_key(self, params: Dict[str, str]) -> str:
        """Generate cache key from parameters."""
        sorted_params = sorted(params.items())
        param_str = urlencode(sorted_params)
        return hashlib.sha256(param_str.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached entry is still valid."""
        return (time.time() - timestamp) < self._cache_ttl

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _fetch_with_params(
        self,
        query_params: GDELTQueryParameters
    ) -> List[GDELTArticle]:
        """
        Fetch articles with the given parameters (supports both timespan and datetime).
        """
        # Get parameters dictionary
        params_dict = query_params.to_dict()
        cache_key = self._get_cache_key(params_dict)

        # Check cache
        if self.cache_results and cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.stats['cache_hits'] += 1
                logger.debug("Cache hit for query")
                return cached_data

        # Rate limiting
        async with self._semaphore:
            wait_time = await self.rate_limiter.acquire()
            if wait_time > 0:
                logger.debug(f"Rate limited, waited {wait_time:.2f}s")

            # Make request
            self.stats['api_requests'] += 1

            # Log the URL for debugging
            full_url = f"{self.BASE_URL}?{urlencode(params_dict)}"
            logger.info(f"GDELT API URL: {full_url}")

            async with self.session.get(
                self.BASE_URL,
                params=params_dict,
                raise_for_status=True
            ) as response:

                if query_params.format == GDELTFormat.CSV:
                    content = await response.text()
                    # Store raw CSV data for potential saving
                    self.last_raw_csv_data = content
                    # Log the raw CSV content for debugging parsing issues
                    logger.debug(f"Raw CSV response (first 1000 chars): {content[:1000]}")
                    articles = await self._parse_csv_response(content)
                elif query_params.format == GDELTFormat.JSON:
                    data = await response.json()
                    articles = await self._parse_json_response(data)
                else:
                    raise ValueError(f"Unsupported format: {query_params.format}")

                # Download content for articles if enabled
                if self.download_content and self.content_downloader:
                    articles = await self._download_content_for_articles(articles)

                # Cache results
                if self.cache_results:
                    self._cache[cache_key] = (articles, time.time())

                logger.info(f"Fetched {len(articles)} articles")
                self.stats['articles_fetched'] += len(articles)

                return articles

    async def _fetch_single_day(
        self,
        query_params: GDELTQueryParameters,
        date: datetime
    ) -> List[GDELTArticle]:
        """
        Fetch articles for a single day using datetime parameters.
        """
        # Create a copy of query params for this specific day
        day_params = GDELTQueryParameters(
            query=query_params.query,
            mode=query_params.mode,
            format=query_params.format,
            maxrecords=query_params.maxrecords,
            sort=query_params.sort,
            dropdup=query_params.dropdup,
            translation=query_params.translation,
            transpose=query_params.transpose,
            timelinesmooth=query_params.timelinesmooth,
            imagewebtagonly=query_params.imagewebtagonly,
            imagefacetonly=query_params.imagefacetonly,
            domain=query_params.domain,
            domainis=query_params.domainis,
            theme=query_params.theme,
            near=query_params.near,
            repeat=query_params.repeat
            # NOTE: Intentionally NOT copying timespan, startdatetime, enddatetime
        )

        # Set specific date parameters (NOT timespan) to avoid conflicts
        date_range = GDELTDateRange(date, date)
        datetime_params = date_range.to_gdelt_datetime_params(specific_date=date)
        day_params.set_datetime_params(datetime_params['startdatetime'], datetime_params['enddatetime'])
        
        logger.info(f"Fetching articles for {date.date()} using datetime params: {datetime_params['startdatetime']} to {datetime_params['enddatetime']}")

        return await self._fetch_with_params(day_params)

    async def _parse_csv_response(self, content: str) -> List[GDELTArticle]:
        """Parse CSV response with the actual GDELT CSV format."""
        articles = []
        seen_urls = set()

        if not content.strip():
            logger.info("Empty CSV response - no articles found")
            return articles

        csv_buffer = io.StringIO(content)
        
        # Count total rows first for logging
        content_lines = content.strip().split('\n')
        total_rows = len(content_lines) - 1 if len(content_lines) > 1 else 0  # Subtract header row
        logger.info(f"CSV response contains {total_rows} data rows (plus header)")
        
        # Reset buffer for actual parsing
        csv_buffer = io.StringIO(content)
        reader = csv.DictReader(csv_buffer)

        processed_rows = 0
        for row_idx, row in enumerate(reader):
            processed_rows += 1
            try:
                # Clean field names and values
                cleaned_row = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    clean_key = key.strip().replace('\ufeff', '').lower()
                    clean_value = value.strip() if value else None
                    cleaned_row[clean_key] = clean_value

                # Map CSV fields to article fields
                # Actual GDELT CSV format: URL, MobileURL, Date, Title
                article_data = {}
                
                # Required fields
                if 'url' in cleaned_row and cleaned_row['url']:
                    article_data['url'] = cleaned_row['url']
                else:
                    logger.debug(f"Row {row_idx}: Missing URL")
                    continue

                if 'title' in cleaned_row and cleaned_row['title']:
                    article_data['title'] = cleaned_row['title']
                else:
                    logger.debug(f"Row {row_idx}: Missing title")
                    continue

                if 'date' in cleaned_row and cleaned_row['date']:
                    article_data['date'] = cleaned_row['date']
                else:
                    logger.debug(f"Row {row_idx}: Missing date")
                    continue

                # Optional fields
                if 'mobileurl' in cleaned_row and cleaned_row['mobileurl']:
                    article_data['mobileurl'] = cleaned_row['mobileurl']

                # Additional fields that might be present in enhanced responses
                optional_fields = {
                    'socialimage': 'socialimage',
                    'language': 'language', 
                    'sourcecountry': 'sourcecountry',
                    'tone': 'tone',
                    'posscore': 'pos_score',
                    'negscore': 'neg_score',
                    'polarity': 'polarity',
                    'activitydensity': 'activity_density',
                    'selfdensity': 'self_density',
                    'wordcount': 'word_count'
                }

                for csv_field, model_field in optional_fields.items():
                    if csv_field in cleaned_row and cleaned_row[csv_field]:
                        value = cleaned_row[csv_field]
                        # Handle numeric conversions
                        if model_field in ['tone', 'pos_score', 'neg_score', 'polarity', 'activity_density', 'self_density']:
                            try:
                                article_data[model_field] = float(value)
                            except (ValueError, TypeError):
                                pass  # Skip invalid numeric values
                        elif model_field == 'word_count':
                            try:
                                article_data[model_field] = int(value)
                            except (ValueError, TypeError):
                                pass  # Skip invalid numeric values
                        else:
                            article_data[model_field] = value

                # Create article (domain will be auto-extracted from URL)
                article = GDELTArticle(**article_data)

                # Deduplication
                url_str = str(article.url)
                if url_str not in seen_urls:
                    seen_urls.add(url_str)
                    articles.append(article)
                else:
                    logger.debug(f"Duplicate URL found: {url_str}")

            except ValidationError as e:
                logger.debug(f"Row {row_idx} validation error: {e}")
                self.stats['parse_errors'] += 1
            except Exception as e:
                logger.debug(f"Row {row_idx} parse error: {e}")
                self.stats['parse_errors'] += 1

        logger.info(f"Successfully parsed {len(articles)} articles from {processed_rows} CSV rows "
                   f"(with {self.stats.get('parse_errors', 0)} parse errors)")
        return articles

    async def _parse_json_response(self, data: Dict) -> List[GDELTArticle]:
        """Parse JSON response."""
        articles = []

        if isinstance(data, dict) and 'articles' in data:
            article_list = data['articles']
        elif isinstance(data, list):
            article_list = data
        else:
            logger.warning(f"Unexpected JSON structure: {type(data)}")
            return articles

        for item in article_list:
            try:
                article = GDELTArticle(**item)
                articles.append(article)
            except ValidationError as e:
                logger.debug(f"JSON validation error: {e}")
                self.stats['parse_errors'] += 1

        return articles

    async def _download_content_for_articles(self, articles: List[GDELTArticle]) -> List[GDELTArticle]:
        """Download content for all articles concurrently."""
        if not articles or not self.content_downloader:
            return articles

        logger.info(f"Downloading content for {len(articles)} articles...")

        # Limit concurrent downloads to avoid overwhelming servers
        semaphore = asyncio.Semaphore(5)
        
        async def download_single_article_content(article: GDELTArticle) -> GDELTArticle:
            async with semaphore:
                try:
                    content = await self.content_downloader.download_content(str(article.url))
                    article.content = content
                    if content:
                        self.stats['content_downloads_success'] += 1
                    else:
                        self.stats['content_downloads_failed'] += 1
                except Exception as e:
                    logger.debug(f"Failed to download content for {article.url}: {e}")
                    self.stats['content_downloads_failed'] += 1
                    article.content = None
                
                return article

        # Download content for all articles concurrently
        updated_articles = await asyncio.gather(
            *[download_single_article_content(article) for article in articles],
            return_exceptions=True
        )

        # Filter out exceptions and return valid articles
        valid_articles = []
        for result in updated_articles:
            if isinstance(result, GDELTArticle):
                valid_articles.append(result)
            else:
                logger.error(f"Content download failed with exception: {result}")

        successful_downloads = sum(1 for article in valid_articles if article.content)
        logger.info(f"Content download complete: {successful_downloads}/{len(valid_articles)} successful")

        return valid_articles

    async def fetch_with_timespan(
        self,
        query: str,
        timespan: str = "24h",
        mode: GDELTMode = GDELTMode.ARTLIST,
        format: GDELTFormat = GDELTFormat.CSV,
        max_records: int = 250,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> List[GDELTArticle]:
        """
        Fetch articles using timespan parameter (e.g., "24h", "7d").
        """
        query_params = GDELTQueryParameters(
            query=query,
            mode=mode,
            format=format,
            timespan=timespan,
            maxrecords=min(max_records, 250),
            **(additional_params or {})
        )

        return await self._fetch_with_params(query_params)

    async def fetch_date_range(
        self,
        query: str,
        date_range: GDELTDateRange,
        mode: GDELTMode = GDELTMode.ARTLIST,
        format: GDELTFormat = GDELTFormat.CSV,
        max_records_per_day: int = 250,
        additional_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> AsyncIterator[Tuple[datetime, List[GDELTArticle]]]:
        """
        Fetch articles for a date range with daily batching.
        """
        # Build base query parameters
        base_params = GDELTQueryParameters(
            query=query,
            mode=mode,
            format=format,
            maxrecords=min(max_records_per_day, 250),
            **(additional_params or {})
        )

        total_days = date_range.total_days
        processed_days = 0

        # Process each day
        async for current_date in date_range.iter_days():
            try:
                # Fetch articles for current day
                articles = await self._fetch_single_day(base_params, current_date)

                # Update progress
                processed_days += 1
                if progress_callback:
                    await progress_callback(processed_days, total_days, current_date)

                yield (current_date, articles)

            except Exception as e:
                logger.error(f"Failed to fetch data for {current_date.date()}: {e}")
                self.stats['failed_days'] += 1
                # Yield empty list to maintain iteration
                yield (current_date, [])

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = dict(self.stats)

        if stats.get('api_requests', 0) > 0:
            stats['avg_articles_per_request'] = (
                stats.get('articles_fetched', 0) / stats['api_requests']
            )
            stats['cache_hit_rate'] = (
                stats.get('cache_hits', 0) /
                (stats.get('cache_hits', 0) + stats['api_requests'])
            )

        if self.download_content:
            total_downloads = stats.get('content_downloads_success', 0) + stats.get('content_downloads_failed', 0)
            if total_downloads > 0:
                stats['content_download_success_rate'] = (
                    stats.get('content_downloads_success', 0) / total_downloads
                )

        return stats


# Example usage and testing
async def test_gdelt_client():
    """Test the enhanced GDELT client with content download."""
    async with ComprehensiveGDELTClient(
        rate_limit=5.0, 
        download_content=True
    ) as client:
        
        # Test 1: Fetch with timespan (default behavior)
        print("Testing timespan query...")
        articles = await client.fetch_with_timespan(
            query="(Fight OR Killed) (sourcecountry:ET OR sourcecountry:SO)",
            timespan="24h",
            max_records=50
        )
        print(f"Timespan query returned {len(articles)} articles")
        
        if articles:
            sample = articles[0]
            print(f"Sample article:")
            print(f"  Title: {sample.title}")
            print(f"  URL: {sample.url}")
            print(f"  Domain: {sample.domain}")
            print(f"  Date: {sample.seendate}")
            print(f"  Content: {'Available' if sample.content else 'Not downloaded'}")
            if sample.content:
                print(f"  Content preview: {sample.content[:200]}...")

        # Test 2: Fetch with date range
        print("\nTesting date range query...")
        date_range = GDELTDateRange(
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 3, tzinfo=timezone.utc)
        )

        async for date, daily_articles in client.fetch_date_range(
            query="(Fight OR Killed) (sourcecountry:ET OR sourcecountry:SO)",
            date_range=date_range,
            max_records_per_day=25
        ):
            print(f"{date.date()}: {len(daily_articles)} articles")
            if daily_articles and daily_articles[0].content:
                print(f"  Sample content length: {len(daily_articles[0].content)} chars")

        # Print statistics
        stats = client.get_statistics()
        print(f"\nStatistics: {stats}")


# Alias for backward compatibility
ComprehensiveGDELTClient = GDELTClient

# Export all components
__all__ = [
    'GDELTArticle',
    'GDELTDateRange', 
    'GDELTQueryParameters',
    'GDELTMode',
    'GDELTFormat',
    'GDELTSort',
    'GDELTDedup',
    'GDELTImageFacet',
    'ComprehensiveGDELTClient',
    'GDELTClient',
    'ContentDownloader',
    'RateLimiter'
]

if __name__ == "__main__":
    asyncio.run(test_gdelt_client())
