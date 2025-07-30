#!/usr/bin/env python3
"""
Enhanced GNews Implementation - Enterprise Grade News Intelligence
================================================================

Complete reimplementation of GNews with advanced site filtering, stealth capabilities,
and integration with existing crawler and PostgreSQL manager components.

Theoretical Foundation:
- Information retrieval optimization using TF-IDF and semantic similarity
- Distributed crawler coordination with consistent hashing
- Adaptive rate limiting using token bucket algorithms
- Site reliability assessment using PageRank variants
- Content quality scoring using linguistic feature analysis

Mathematical Models:
- Site credibility: C(s) = α × authority(s) + β × freshness(s) + γ × relevance(s)
- Content quality: Q(c) = Σ(w_i × feature_i) where features include readability, completeness
- Crawl priority: P(url) = urgency × credibility × (1 - staleness)
- Rate limiting: tokens = min(capacity, tokens + (current_time - last_time) × rate)

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import json
import logging
import hashlib
import time
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin, quote_plus

from enum import Enum

import aiohttp
try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    import pydantic
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    pydantic = None
    BaseModel = None
    Field = None
    field_validator = None

try:
    import yaml
except ImportError:
    yaml = None

# Import Crawlee integration
try:
    from ..crawlee_integration import CrawleeNewsEnhancer, ArticleResult
    CRAWLEE_INTEGRATION_AVAILABLE = True
except ImportError:
    CrawleeNewsEnhancer = None
    ArticleResult = None
    CRAWLEE_INTEGRATION_AVAILABLE = False

# Configure logging early
logger = logging.getLogger(__name__)

# Import your existing components - flexible import handling
try:
    # Try importing from packages_enhanced database module (correct path)
    from ....database.postgresql_manager import PgSQLManager
except ImportError:
    try:
        # Try importing from managers subdirectory
        from ....database.managers.postgres_manager import HybridIntegratedPostgreSQLManager
    except ImportError:
        try:
            # Try absolute import
            from packages_enhanced.database.postgresql_manager import PgSQLManager
        except ImportError:
            try:
                # Try importing from the packages directory
                from lindela.packages.pgmgr import HybridIntegratedPostgreSQLManager
            except ImportError:
                try:
                    # Try relative import for direct access
                    import sys
                    import os
                    # Add the packages directory to the path
                    packages_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'packages')
                    if os.path.exists(packages_path):
                        sys.path.insert(0, packages_path)
                    from pgmgr.hybrid_postgresql_manager import HybridIntegratedPostgreSQLManager
                except ImportError:
                    # Last resort - create a placeholder
                    logger.debug("HybridIntegratedPostgreSQLManager not found. Using placeholder.")
                    class PgSQLManager:
                        def __init__(self, *args, **kwargs):
                            raise NotImplementedError("PostgreSQL manager not available")


class SourceType(Enum):
    """News source classification types."""
    MAINSTREAM_MEDIA = "mainstream_media"
    NEWSPAPER = "newspaper"
    GOVERNMENT_OFFICIAL = "government_official"
    NEWS_AGENCY = "news_agency"
    INDEPENDENT_MEDIA = "independent_media"
    BLOG_OPINION = "blog_opinion"
    SOCIAL_MEDIA = "social_media"
    ACADEMIC_RESEARCH = "academic_research"
    NGO_REPORT = "ngo_report"
    RADIO = "radio"
    MAGAZINE = "magazine"
    JOURNAL = "journal"
    ACADEMIC_JOURNAL = "academic_journal"
    UNKNOWN = "unknown"


class ContentFormat(Enum):
    """Content format types."""
    RSS = "rss"
    RSS_FEED = "rss_feed"
    HTML_PAGE = "html_page"
    JSON_API = "json_api"
    XML_SITEMAP = "xml_sitemap"
    SOCIAL_MEDIA_API = "social_media_api"


class CrawlPriority(Enum):
    """Crawl priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class GeographicalFocus:
    """Geographical focus configuration."""
    countries: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    coordinates: Optional[Tuple[float, float]] = None
    radius_km: Optional[float] = None
    priority_multiplier: float = 1.0


@dataclass
class SourceCredibilityMetrics:
    """Comprehensive source credibility assessment."""
    domain: str
    source_type: SourceType
    authority_score: float = 0.5
    freshness_score: float = 0.5
    reliability_score: float = 0.5
    bias_rating: str = "NEUTRAL"  # LEFT, LEAN_LEFT, CENTER, LEAN_RIGHT, RIGHT
    fact_check_rating: Optional[str] = None
    language_quality_score: float = 0.5
    update_frequency_score: float = 0.5
    geographical_relevance: Dict[str, float] = field(default_factory=dict)

    # Historical performance
    successful_crawls: int = 0
    failed_crawls: int = 0
    avg_response_time_ms: float = 0.0
    content_quality_avg: float = 0.5

    # Temporal patterns
    active_hours: List[int] = field(default_factory=lambda: list(range(24)))
    peak_update_times: List[str] = field(default_factory=list)

    def calculate_composite_score(self) -> float:
        """Calculate weighted composite credibility score."""
        weights = {
            'authority': 0.25,
            'freshness': 0.20,
            'reliability': 0.25,
            'language_quality': 0.15,
            'update_frequency': 0.15
        }

        return (
            weights['authority'] * self.authority_score +
            weights['freshness'] * self.freshness_score +
            weights['reliability'] * self.reliability_score +
            weights['language_quality'] * self.language_quality_score +
            weights['update_frequency'] * self.update_frequency_score
        )

    def get_success_rate(self) -> float:
        """Calculate crawl success rate."""
        total = self.successful_crawls + self.failed_crawls
        return self.successful_crawls / max(1, total)


@dataclass
class NewsSource:
    """Enhanced news source configuration."""
    name: str
    domain: str
    source_type: SourceType
    content_format: ContentFormat

    # URLs and endpoints
    rss_feeds: List[str] = field(default_factory=list)
    sitemap_urls: List[str] = field(default_factory=list)
    api_endpoints: List[str] = field(default_factory=list)
    base_urls: List[str] = field(default_factory=list)

    # Geographical and topical focus
    geographical_focus: Optional[GeographicalFocus] = None
    topic_categories: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])

    # Crawling configuration
    crawl_priority: CrawlPriority = CrawlPriority.NORMAL
    crawl_interval_minutes: int = 60
    max_articles_per_crawl: int = 100
    respect_robots_txt: bool = True

    # Rate limiting and politeness
    requests_per_minute: int = 30
    concurrent_connections: int = 2
    crawl_delay_seconds: float = 2.0

    # Content filtering
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    excluded_keywords: List[str] = field(default_factory=list)

    # Technical configuration
    user_agent: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    proxy_config: Optional[Dict[str, str]] = None
    auth_config: Optional[Dict[str, str]] = None

    # Quality thresholds
    min_content_length: int = 200
    max_content_length: int = 50000
    min_readability_score: float = 30.0

    # Metadata
    credibility_metrics: Optional[SourceCredibilityMetrics] = None
    last_crawl_time: Optional[datetime] = None
    last_successful_crawl: Optional[datetime] = None
    is_active: bool = True
    notes: str = ""


class SiteFilteringEngine:
    """
    Advanced site filtering engine with multi-criteria evaluation
    and dynamic credibility assessment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize site filtering engine."""
        self.config = config or {}

        # Filtering criteria
        self.geographical_filters = {}
        self.topical_filters = {}
        self.quality_thresholds = {
            'min_authority_score': 0.3,
            'min_reliability_score': 0.4,
            'max_bias_deviation': 0.7,  # From center
            'min_success_rate': 0.7,
            'max_avg_response_time': 10000  # ms
        }

        # Credibility databases
        self.known_credible_sources = set()
        self.known_unreliable_sources = set()
        self.bias_ratings_db = {}

        # Content analysis
        if TfidfVectorizer:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = None

        # Load predefined source lists
        self._load_source_databases()

        logger.info("Site filtering engine initialized")

    def _load_source_databases(self):
        """Load predefined source credibility databases."""
        # Horn of Africa focused credible sources
        horn_africa_sources = {
            'nation.co.ke', 'standardmedia.co.ke', 'thestar.co.ke',  # Kenya
            'fanabc.com', 'ena.et', 'addisstandard.com',  # Ethiopia
            'hiiraan.com', 'somaliguardian.com', 'garowe-online.com',  # Somalia
            'sudantribune.com', 'smc.sd',  # Sudan
            'radiotamazuj.org', 'eyeradio.org',  # South Sudan
            'monitor.co.ug', 'newvision.co.ug',  # Uganda
            'thecitizen.co.tz', 'dailynews.co.tz',  # Tanzania
            'shabait.com',  # Eritrea
            'djiboutitimes.com'  # Djibouti
        }

        international_credible_sources = {
            'reuters.com', 'ap.org', 'bbc.com', 'cnn.com',
            'aljazeera.com', 'france24.com', 'dw.com',
            'voanews.com', 'theguardian.com', 'npr.org'
        }

        self.known_credible_sources.update(horn_africa_sources)
        self.known_credible_sources.update(international_credible_sources)

        # Known unreliable or problematic sources
        self.known_unreliable_sources.update({
            'example-fake-news.com',  # Placeholder - would be populated with actual unreliable sources
        })

        logger.info(f"Loaded {len(self.known_credible_sources)} credible sources")

    async def evaluate_source(self, source: NewsSource) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Comprehensive source evaluation using multiple criteria.

        Args:
            source: NewsSource to evaluate

        Returns:
            Tuple of (should_include, score, detailed_metrics)
        """
        evaluation_start = time.time()

        # Initialize evaluation metrics
        metrics = {
            'domain_credibility': 0.0,
            'geographical_relevance': 0.0,
            'topical_relevance': 0.0,
            'technical_reliability': 0.0,
            'content_quality': 0.0,
            'historical_performance': 0.0,
            'bias_assessment': 0.0,
            'overall_score': 0.0,
            'evaluation_time_ms': 0.0,
            'reasons_included': [],
            'reasons_excluded': []
        }

        # 1. Domain credibility assessment
        domain_score = await self._assess_domain_credibility(source)
        metrics['domain_credibility'] = domain_score

        if domain_score > 0.7:
            metrics['reasons_included'].append(f"High domain credibility: {domain_score:.2f}")
        elif domain_score < 0.3:
            metrics['reasons_excluded'].append(f"Low domain credibility: {domain_score:.2f}")

        # 2. Geographical relevance
        geo_score = self._assess_geographical_relevance(source)
        metrics['geographical_relevance'] = geo_score

        if geo_score > 0.6:
            metrics['reasons_included'].append(f"High geographical relevance: {geo_score:.2f}")

        # 3. Topical relevance for conflict/CEWS
        topic_score = self._assess_topical_relevance(source)
        metrics['topical_relevance'] = topic_score

        # 4. Technical reliability
        tech_score = self._assess_technical_reliability(source)
        metrics['technical_reliability'] = tech_score

        if tech_score < 0.4:
            metrics['reasons_excluded'].append(f"Poor technical reliability: {tech_score:.2f}")

        # 5. Historical performance
        if source.credibility_metrics:
            hist_score = source.credibility_metrics.get_success_rate()
            metrics['historical_performance'] = hist_score

            if hist_score < self.quality_thresholds['min_success_rate']:
                metrics['reasons_excluded'].append(f"Low success rate: {hist_score:.2f}")

        # 6. Bias assessment
        bias_score = self._assess_bias_level(source)
        metrics['bias_assessment'] = bias_score

        # Calculate weighted overall score
        weights = {
            'domain_credibility': 0.25,
            'geographical_relevance': 0.20,
            'topical_relevance': 0.15,
            'technical_reliability': 0.15,
            'historical_performance': 0.15,
            'bias_assessment': 0.10
        }

        overall_score = sum(
            weights[key] * metrics[key]
            for key in weights.keys()
        )

        metrics['overall_score'] = overall_score
        metrics['evaluation_time_ms'] = (time.time() - evaluation_start) * 1000

        # Decision logic
        should_include = (
            overall_score >= 0.5 and
            domain_score >= 0.3 and
            tech_score >= 0.4 and
            len(metrics['reasons_excluded']) == 0
        )

        if should_include:
            metrics['reasons_included'].append(f"Overall score threshold met: {overall_score:.2f}")
        else:
            metrics['reasons_excluded'].append(f"Overall score below threshold: {overall_score:.2f}")

        logger.debug(f"Source evaluation for {source.domain}: {overall_score:.2f} ({'included' if should_include else 'excluded'})")

        return should_include, overall_score, metrics

    async def _assess_domain_credibility(self, source: NewsSource) -> float:
        """Assess domain credibility using multiple indicators."""
        score = 0.5  # Default neutral score

        # Check against known credible sources
        if source.domain in self.known_credible_sources:
            score += 0.3

        # Check against known unreliable sources
        if source.domain in self.known_unreliable_sources:
            score -= 0.4

        # Domain age and authority (simplified heuristics)
        if source.domain.endswith('.gov') or source.domain.endswith('.edu'):
            score += 0.2
        elif source.domain.endswith('.org'):
            score += 0.1

        # TLD credibility assessment
        credible_tlds = {'.com', '.org', '.net', '.gov', '.edu', '.mil'}
        if any(source.domain.endswith(tld) for tld in credible_tlds):
            score += 0.1

        # Use existing credibility metrics if available
        if source.credibility_metrics:
            existing_score = source.credibility_metrics.calculate_composite_score()
            score = (score + existing_score) / 2  # Average with existing assessment

        return min(max(score, 0.0), 1.0)

    def _assess_geographical_relevance(self, source: NewsSource) -> float:
        """Assess geographical relevance for CEWS focus areas."""
        if not source.geographical_focus:
            return 0.3  # Neutral score for non-geographically focused sources

        relevance_score = 0.0

        # Horn of Africa countries (high priority for CEWS)
        horn_countries = {
            'kenya', 'ethiopia', 'somalia', 'sudan', 'south sudan',
            'uganda', 'tanzania', 'eritrea', 'djibouti', 'rwanda', 'burundi'
        }

        # Check country relevance
        for country in source.geographical_focus.countries:
            if country.lower() in horn_countries:
                relevance_score += 0.3
            elif country.lower() in {'egypt', 'libya', 'chad', 'car', 'drc'}:  # Neighboring regions
                relevance_score += 0.2
            else:
                relevance_score += 0.1

        # Bonus for regional coverage
        if any(region.lower() in ['east africa', 'horn of africa', 'africa']
               for region in source.geographical_focus.regions):
            relevance_score += 0.2

        # Apply priority multiplier
        relevance_score *= source.geographical_focus.priority_multiplier

        return min(relevance_score, 1.0)

    def _assess_topical_relevance(self, source: NewsSource) -> float:
        """Assess topical relevance for conflict and early warning content."""
        conflict_topics = {
            'politics', 'security', 'military', 'conflict', 'peace',
            'humanitarian', 'refugees', 'displacement', 'crisis',
            'elections', 'governance', 'human rights', 'justice'
        }

        relevance_score = 0.5  # Default for general news

        # Check topic overlap
        source_topics = [topic.lower() for topic in source.topic_categories]
        overlap = len(set(source_topics) & conflict_topics)

        if overlap > 0:
            relevance_score += min(overlap * 0.15, 0.4)

        # Check for explicit conflict/CEWS keywords
        cews_keywords = ['conflict', 'peace', 'security', 'crisis', 'humanitarian']
        for keyword in cews_keywords:
            if any(keyword in topic.lower() for topic in source.topic_categories):
                relevance_score += 0.1

        return min(relevance_score, 1.0)

    def _assess_technical_reliability(self, source: NewsSource) -> float:
        """Assess technical reliability based on configuration and history."""
        score = 0.5

        # RSS feed availability (positive indicator)
        if source.rss_feeds:
            score += 0.2

        # Multiple content formats (resilience)
        format_count = sum([
            bool(source.rss_feeds),
            bool(source.sitemap_urls),
            bool(source.api_endpoints),
            bool(source.base_urls)
        ])
        score += min(format_count * 0.1, 0.3)

        # Reasonable rate limiting (indicates professionalism)
        if 10 <= source.requests_per_minute <= 60:
            score += 0.1

        # Robots.txt respect (indicates good practices)
        if source.respect_robots_txt:
            score += 0.1

        # Historical performance
        if source.credibility_metrics:
            if source.credibility_metrics.avg_response_time_ms < 5000:
                score += 0.1
            if source.credibility_metrics.get_success_rate() > 0.8:
                score += 0.1

        return min(score, 1.0)

    def _assess_bias_level(self, source: NewsSource) -> float:
        """Assess bias level (higher score = less biased)."""
        if not source.credibility_metrics:
            return 0.5  # Neutral assumption

        bias_mapping = {
            'CENTER': 1.0,
            'LEAN_LEFT': 0.8,
            'LEAN_RIGHT': 0.8,
            'LEFT': 0.6,
            'RIGHT': 0.6,
            'NEUTRAL': 0.9
        }

        return bias_mapping.get(source.credibility_metrics.bias_rating, 0.5)

    def filter_sources(self, sources: List[NewsSource]) -> Tuple[List[NewsSource], List[NewsSource]]:
        """
        Filter sources synchronously for bulk operations.

        Returns:
            Tuple of (included_sources, excluded_sources)
        """
        included = []
        excluded = []

        for source in sources:
            # Quick pre-filters
            if not source.is_active:
                excluded.append(source)
                continue

            if source.domain in self.known_unreliable_sources:
                excluded.append(source)
                continue

            # Basic quality checks
            if (source.credibility_metrics and
                source.credibility_metrics.get_success_rate() < 0.3):
                excluded.append(source)
                continue

            included.append(source)

        logger.info(f"Pre-filtered {len(sources)} sources: {len(included)} included, {len(excluded)} excluded")
        return included, excluded


class NewsSource:
    """News source with dynamic capability detection."""

    def __init__(self, source_config: NewsSource):
        """Initialize news source."""
        self.config = source_config
        self.capabilities = {
            'rss_available': bool(source_config.rss_feeds),
            'sitemap_available': bool(source_config.sitemap_urls),
            'api_available': bool(source_config.api_endpoints),
            'direct_crawl_available': bool(source_config.base_urls)
        }

        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'last_success_time': None,
            'consecutive_failures': 0
        }

        # Content cache
        self.content_cache = {}
        self.cache_expiry = timedelta(minutes=30)

        logger.debug(f"Enhanced news source initialized: {source_config.domain}")

    async def discover_feeds(self) -> List[str]:
        """Automatically discover RSS feeds and other content sources."""
        discovered_feeds = list(self.config.rss_feeds)  # Start with configured feeds

        # Attempt automatic feed discovery
        for base_url in self.config.base_urls:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Common RSS feed locations
                    common_paths = [
                        '/rss', '/rss.xml', '/feed', '/feed.xml',
                        '/feeds/all.atom.xml', '/atom.xml',
                        '/index.rss', '/news.rss', '/blog/feed'
                    ]

                    for path in common_paths:
                        feed_url = urljoin(base_url, path)
                        try:
                            async with session.get(feed_url) as response:
                                if response.status == 200:
                                    content_type = response.headers.get('content-type', '')
                                    if any(t in content_type.lower() for t in ['xml', 'rss', 'atom']):
                                        if feed_url not in discovered_feeds:
                                            discovered_feeds.append(feed_url)
                                            logger.info(f"Discovered feed: {feed_url}")
                        except Exception as e:
                            logger.debug(f"Feed discovery failed for {feed_url}: {e}")
                            continue

            except Exception as e:
                logger.debug(f"Feed discovery failed for {base_url}: {e}")

        return discovered_feeds

    async def validate_feeds(self, feeds: List[str]) -> List[str]:
        """Validate and filter working RSS feeds."""
        valid_feeds = []

        async with aiohttp.ClientSession() as session:
            for feed_url in feeds:
                try:
                    async with session.get(feed_url, timeout=15) as response:
                        if response.status == 200:
                            content = await response.text()

                            # Basic RSS/Atom validation
                            if any(tag in content.lower() for tag in ['<rss', '<feed', '<atom']):
                                # Try to parse with feedparser
                                try:
                                    parsed = feedparser.parse(content)
                                    if parsed.entries and len(parsed.entries) > 0:
                                        valid_feeds.append(feed_url)
                                        logger.debug(f"Validated feed: {feed_url} ({len(parsed.entries)} entries)")
                                except Exception as e:
                                    logger.debug(f"Feed parsing failed for {feed_url}: {e}")

                except Exception as e:
                    logger.debug(f"Feed validation failed for {feed_url}: {e}")

        return valid_feeds

    def update_performance_metrics(self, success: bool, response_time_ms: float):
        """Update performance tracking metrics."""
        self.performance_metrics['total_requests'] += 1

        if success:
            self.performance_metrics['successful_requests'] += 1
            self.performance_metrics['last_success_time'] = datetime.now(timezone.utc)
            self.performance_metrics['consecutive_failures'] = 0

            # Update running average of response time
            current_avg = self.performance_metrics['avg_response_time']
            total_successful = self.performance_metrics['successful_requests']
            self.performance_metrics['avg_response_time'] = (
                (current_avg * (total_successful - 1) + response_time_ms) / total_successful
            )
        else:
            self.performance_metrics['failed_requests'] += 1
            self.performance_metrics['consecutive_failures'] += 1

    def get_health_score(self) -> float:
        """Calculate health score based on performance metrics."""
        total = self.performance_metrics['total_requests']
        if total == 0:
            return 0.5  # Neutral score for new sources

        success_rate = self.performance_metrics['successful_requests'] / total

        # Penalize consecutive failures
        failure_penalty = min(self.performance_metrics['consecutive_failures'] * 0.1, 0.5)

        # Bonus for good response times
        time_bonus = 0.1 if self.performance_metrics['avg_response_time'] < 3000 else 0.0

        return max(0.0, min(1.0, success_rate - failure_penalty + time_bonus))


class GoogleNewsClient:
    """
    Enterprise-grade Google News client with advanced filtering,
    stealth capabilities, and integration with existing systems.
    """

    def __init__(
        self,
        db_manager: PgSQLManager,
        stealth_orchestrator=None,  # Your existing Cloudflare stealth system
        crawlee_enhancer: Optional[CrawleeNewsEnhancer] = None,  # Crawlee content enhancement
        config: Dict[str, Any] = None
    ):
        """Initialize Google News client."""
        self.db_manager = db_manager
        self.stealth_orchestrator = stealth_orchestrator
        self.crawlee_enhancer = crawlee_enhancer
        self.config = config or {}

        # HTTP session management
        self._http_session = None

        # Initialize components
        self.site_filter = SiteFilteringEngine(self.config.get('filtering', {}) if self.config else {})

        # News sources registry
        self.news_sources: Dict[str, NewsSource] = {}
        self.active_sources: List[str] = []

        # Google News specific configuration
        self.google_base_url = "https://news.google.com/rss"
        self.google_search_url = "https://news.google.com/search"

        # Supported countries and languages (enhanced)
        self.countries = {
            'Kenya': 'KE', 'Ethiopia': 'ET', 'Somalia': 'SO', 'Sudan': 'SD',
            'South Sudan': 'SS', 'Uganda': 'UG', 'Tanzania': 'TZ', 'Eritrea': 'ER',
            'Djibouti': 'DJ', 'Rwanda': 'RW', 'Burundi': 'BI', 'Egypt': 'EG',
            'Libya': 'LY', 'Chad': 'TD', 'CAR': 'CF', 'DRC': 'CD',
            'United States': 'US', 'United Kingdom': 'GB', 'Germany': 'DE',
            'France': 'FR', 'Canada': 'CA', 'Australia': 'AU'
        }

        self.languages = {
            'english': 'en', 'french': 'fr', 'arabic': 'ar', 'swahili': 'sw',
            'amharic': 'am', 'somali': 'so', 'spanish': 'es', 'portuguese': 'pt'
        }

        # Performance tracking
        self.session_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'articles_discovered': 0,
            'articles_filtered': 0,
            'articles_scraped': 0,
            'start_time': datetime.now(timezone.utc)
        }

        # Content quality filters
        self.content_filters = {
            'min_title_length': 10,
            'max_title_length': 200,
            'min_description_length': 20,
            'required_fields': ['title', 'url', 'published_date'],
            'excluded_domains': set(),
            'language_filters': ['en', 'fr', 'ar', 'sw']
        }

        logger.info("Enhanced Google News client initialized")

    async def initialize(self):
        """Initialize client with source discovery and validation."""
        logger.info("Initializing Enhanced Google News client...")

        # Initialize HTTP session
        if not self._http_session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=False  # Disable SSL verification to prevent timeout issues
            )
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; EnhancedGNewsBot/1.0)'}
            )

        # Load predefined source configurations
        await self._load_source_configurations()

        # Discover and validate additional sources
        await self._discover_regional_sources()

        # Initialize stealth capabilities if available
        if self.stealth_orchestrator:
            # Integration with your existing stealth system
            logger.info("Stealth capabilities enabled")

        logger.info("Enhanced Google News client initialization completed")

        logger.info(f"Client initialized with {len(self.news_sources)} news sources")

    async def _load_source_configurations(self):
        """Load predefined source configurations for Horn of Africa."""
        # Horn of Africa sources with detailed configuration
        source_configs = [
            # Kenya
            NewsSource(
                name="The Standard Kenya",
                domain="standardmedia.co.ke",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=[
                    "https://www.standardmedia.co.ke/rss/headlines.php",
                    "https://www.standardmedia.co.ke/rss/national.php"
                ],
                geographical_focus=GeographicalFocus(
                    countries=["Kenya"],
                    regions=["East Africa"],
                    priority_multiplier=1.2
                ),
                topic_categories=["politics", "security", "business"],
                crawl_priority=CrawlPriority.HIGH,
                requests_per_minute=20
            ),

            NewsSource(
                name="Daily Nation Kenya",
                domain="nation.co.ke",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=[
                    "https://nation.co.ke/kenya/news/-/1056/1056/-/view/asFeed/-/15b8kh2/-/index.xml"
                ],
                geographical_focus=GeographicalFocus(
                    countries=["Kenya"],
                    regions=["East Africa"],
                    priority_multiplier=1.3
                ),
                topic_categories=["politics", "security", "governance"],
                crawl_priority=CrawlPriority.HIGH
            ),

            # Ethiopia
            NewsSource(
                name="Fana Broadcasting Corporate",
                domain="fanabc.com",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=["https://www.fanabc.com/english/feed/"],
                geographical_focus=GeographicalFocus(
                    countries=["Ethiopia"],
                    regions=["Horn of Africa"],
                    priority_multiplier=1.4
                ),
                topic_categories=["politics", "security", "conflict"],
                crawl_priority=CrawlPriority.HIGH
            ),

            # Somalia
            NewsSource(
                name="Hiiraan Online",
                domain="hiiraan.com",
                source_type=SourceType.INDEPENDENT_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=["http://www.hiiraan.com/rss/news.xml"],
                geographical_focus=GeographicalFocus(
                    countries=["Somalia"],
                    regions=["Horn of Africa"],
                    priority_multiplier=1.5
                ),
                topic_categories=["conflict", "security", "humanitarian"],
                crawl_priority=CrawlPriority.CRITICAL
            ),

            # Sudan
            NewsSource(
                name="Sudan Tribune",
                domain="sudantribune.com",
                source_type=SourceType.INDEPENDENT_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=["https://sudantribune.com/spip.php?page=backend"],
                geographical_focus=GeographicalFocus(
                    countries=["Sudan"],
                    regions=["Horn of Africa"],
                    priority_multiplier=1.4
                ),
                topic_categories=["conflict", "politics", "crisis"],
                crawl_priority=CrawlPriority.CRITICAL
            ),

            # International sources with regional focus
            NewsSource(
                name="Voice of America Africa",
                domain="voanews.com",
                source_type=SourceType.NEWS_AGENCY,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=[
                    "https://www.voanews.com/api/zq$tevenm",  # Africa section
                    "https://www.voanews.com/api/z_$teveoqm"  # East Africa
                ],
                geographical_focus=GeographicalFocus(
                    countries=["Kenya", "Ethiopia", "Somalia", "Sudan", "South Sudan"],
                    regions=["Africa", "East Africa"],
                    priority_multiplier=1.1
                ),
                topic_categories=["politics", "security", "conflict", "humanitarian"],
                crawl_priority=CrawlPriority.HIGH
            ),

            NewsSource(
                name="BBC Africa",
                domain="bbc.com",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=[
                    "http://feeds.bbci.co.uk/news/world/africa/rss.xml"
                ],
                geographical_focus=GeographicalFocus(
                    countries=["Kenya", "Ethiopia", "Somalia", "Sudan"],
                    regions=["Africa"],
                    priority_multiplier=1.0
                ),
                topic_categories=["politics", "security", "conflict"],
                crawl_priority=CrawlPriority.HIGH
            ),

            # Regional organizations
            NewsSource(
                name="IGAD News",
                domain="igad.int",
                source_type=SourceType.GOVERNMENT_OFFICIAL,
                content_format=ContentFormat.RSS_FEED,
                base_urls=["https://igad.int/"],
                geographical_focus=GeographicalFocus(
                    countries=["Kenya", "Ethiopia", "Somalia", "Sudan", "South Sudan", "Uganda"],
                    regions=["Horn of Africa"],
                    priority_multiplier=1.2
                ),
                topic_categories=["politics", "security", "peace", "humanitarian"],
                crawl_priority=CrawlPriority.HIGH
            )
        ]

        # Initialize enhanced sources and filter them
        for config in source_configs:
            enhanced_source = NewsSource(config)

            # Evaluate source through filtering engine
            should_include, score, metrics = await self.site_filter.evaluate_source(config)

            if should_include:
                self.news_sources[config.domain] = enhanced_source
                self.active_sources.append(config.domain)
                logger.info(f"Added source: {config.name} (score: {score:.2f})")
            else:
                logger.warning(f"Excluded source: {config.name} (score: {score:.2f})")

    async def _discover_regional_sources(self):
        """Discover additional regional news sources automatically."""
        # This would implement automatic discovery of regional sources
        # For now, we'll add a few more manually discovered sources

        additional_sources = [
            # Uganda
            NewsSource(
                name="Daily Monitor Uganda",
                domain="monitor.co.ug",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                rss_feeds=["https://www.monitor.co.ug/ugandanews.rss"],
                geographical_focus=GeographicalFocus(
                    countries=["Uganda"],
                    regions=["East Africa"],
                    priority_multiplier=1.1
                ),
                topic_categories=["politics", "security"],
                crawl_priority=CrawlPriority.NORMAL
            ),

            # Tanzania
            NewsSource(
                name="The Citizen Tanzania",
                domain="thecitizen.co.tz",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.RSS_FEED,
                base_urls=["https://www.thecitizen.co.tz/"],
                geographical_focus=GeographicalFocus(
                    countries=["Tanzania"],
                    regions=["East Africa"],
                    priority_multiplier=1.0
                ),
                topic_categories=["politics", "business"],
                crawl_priority=CrawlPriority.NORMAL
            )
        ]

        for config in additional_sources:
            enhanced_source = NewsSource(config)

            # Auto-discover feeds
            discovered_feeds = await enhanced_source.discover_feeds()
            if discovered_feeds:
                config.rss_feeds.extend(discovered_feeds)

            # Validate feeds
            if config.rss_feeds:
                valid_feeds = await enhanced_source.validate_feeds(config.rss_feeds)
                config.rss_feeds = valid_feeds

            if config.rss_feeds or config.base_urls:
                should_include, score, metrics = await self.site_filter.evaluate_source(config)

                if should_include:
                    self.news_sources[config.domain] = enhanced_source
                    self.active_sources.append(config.domain)
                    logger.info(f"Discovered and added source: {config.name}")

    async def search_news(
        self,
        query: str,
        countries: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        max_results: int = 100,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        source_filter: Optional[List[str]] = None,
        enable_crawlee: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Enhanced news search with comprehensive filtering and source validation.

        Args:
            query: Search query with support for boolean operators
            countries: List of country codes to filter by
            languages: List of language codes
            max_results: Maximum number of results to return
            time_range: Optional tuple of (start_date, end_date)
            source_filter: Optional list of domains to include/exclude
            enable_crawlee: Whether to use Crawlee for full content downloading

        Returns:
            List of enhanced news articles with metadata
        """
        search_start = time.time()
        self.session_stats['total_searches'] += 1

        logger.info(f"Starting enhanced news search: '{query}'")

        try:
            # Parse and enhance query
            enhanced_query = self._enhance_search_query(query)

            # Determine target countries and languages
            target_countries = countries if countries is not None else ['KE', 'ET', 'SO', 'SD', 'SS', 'UG', 'TZ']
            target_languages = languages if languages is not None else ['en', 'fr', 'ar']

            # Collect articles from multiple sources
            all_articles = []

            # 1. Search Google News RSS
            google_articles = await self._search_google_news_rss(
                enhanced_query, target_countries, target_languages, max_results // 2
            )
            all_articles.extend(google_articles)

            # 2. Search configured regional sources
            regional_articles = await self._search_regional_sources(
                enhanced_query, max_results // 2
            )
            all_articles.extend(regional_articles)

            # 3. Apply content filtering
            filtered_articles = self._apply_content_filtering(all_articles)

            # 4. Deduplicate and rank
            final_articles = self._deduplicate_and_rank(filtered_articles, max_results)

            # 5. Enhance with metadata
            enhanced_articles = await self._enhance_articles_metadata(final_articles)

            # 6. Optionally enhance with Crawlee full content download
            if enable_crawlee and self.crawlee_enhancer and CRAWLEE_INTEGRATION_AVAILABLE:
                logger.info(f"🚀 Enhancing {len(enhanced_articles)} articles with Crawlee content download")
                try:
                    # Convert articles to format expected by CrawleeNewsEnhancer
                    article_metadata = []
                    for article in enhanced_articles:
                        article_metadata.append({
                            'title': article.get('title', ''),
                            'url': article.get('url', ''),
                            'description': article.get('description', ''),
                            'published_date': article.get('published_date'),
                            'source': article.get('publisher', {}).get('name', ''),
                            'link': article.get('url', '')  # Alternative key for compatibility
                        })
                    
                    # Enhance articles with full content
                    enhanced_results = await self.crawlee_enhancer.enhance_articles(article_metadata)
                    
                    # Merge Crawlee results back into original articles
                    enhanced_articles = self._merge_crawlee_results(enhanced_articles, enhanced_results)
                    
                    logger.info(f"✅ Crawlee enhancement completed for {len(enhanced_results)} articles")
                    
                except Exception as e:
                    logger.warning(f"Crawlee enhancement failed, continuing with basic results: {e}")
            elif enable_crawlee and not self.crawlee_enhancer:
                logger.warning("Crawlee enhancement requested but no CrawleeNewsEnhancer provided")
            elif enable_crawlee and not CRAWLEE_INTEGRATION_AVAILABLE:
                logger.warning("Crawlee enhancement requested but crawlee_integration not available")

            # Update statistics
            self.session_stats['successful_searches'] += 1
            self.session_stats['articles_discovered'] += len(all_articles)
            self.session_stats['articles_filtered'] += len(filtered_articles)

            search_time = (time.time() - search_start) * 1000
            logger.info(f"Search completed in {search_time:.1f}ms: {len(enhanced_articles)} articles")

            return enhanced_articles

        except Exception as e:
            logger.error(f"Enhanced news search failed: {e}")
            raise

    def _merge_crawlee_results(
        self, 
        original_articles: List[Dict[str, Any]], 
        crawlee_results: List[ArticleResult]
    ) -> List[Dict[str, Any]]:
        """Merge Crawlee enhanced content with original article metadata."""
        # Create URL-based lookup for Crawlee results
        crawlee_by_url = {result.url: result for result in crawlee_results}
        
        enhanced_articles = []
        for original in original_articles:
            url = original.get('url', '')
            enhanced_article = original.copy()
            
            # Check if we have Crawlee enhancement for this URL
            if url in crawlee_by_url:
                crawlee_result = crawlee_by_url[url]
                
                # Merge enhanced content into original article
                enhanced_article.update({
                    # Enhanced content from Crawlee
                    'full_content': crawlee_result.full_content,
                    'word_count': crawlee_result.word_count,
                    'reading_time_minutes': crawlee_result.reading_time_minutes,
                    'crawlee_quality_score': crawlee_result.quality_score,
                    
                    # Enhanced metadata
                    'article_text': crawlee_result.article_text,
                    'lead_paragraph': crawlee_result.lead_paragraph,
                    'body_paragraphs': crawlee_result.body_paragraphs,
                    
                    # Enhanced media and authors
                    'images': crawlee_result.images,
                    'crawlee_authors': crawlee_result.authors,
                    'crawlee_keywords': crawlee_result.keywords,
                    'tags': crawlee_result.tags,
                    
                    # Geographic and topical relevance
                    'geographic_entities': crawlee_result.geographic_entities,
                    'conflict_indicators': crawlee_result.conflict_indicators,
                    'crawlee_relevance_score': crawlee_result.relevance_score,
                    
                    # Processing metadata
                    'content_extraction_method': crawlee_result.extraction_method,
                    'content_processing_time_ms': crawlee_result.processing_time_ms,
                    'crawlee_success': crawlee_result.crawl_success,
                    'crawlee_fallback_used': crawlee_result.fallback_used,
                    'crawlee_errors': crawlee_result.errors,
                    
                    # Update title if Crawlee found a better one
                    'title': crawlee_result.title if crawlee_result.title else original.get('title', ''),
                    
                    # Mark as enhanced
                    'crawlee_enhanced': True
                })
                
                # Merge authors intelligently
                original_authors = original.get('authors', [])
                crawlee_authors = crawlee_result.authors or []
                all_authors = list(set(original_authors + crawlee_authors))
                enhanced_article['authors'] = all_authors
                
            else:
                # No Crawlee enhancement available, mark as not enhanced
                enhanced_article['crawlee_enhanced'] = False
                enhanced_article['full_content'] = original.get('description', '')
                enhanced_article['word_count'] = len(original.get('description', '').split())
            
            enhanced_articles.append(enhanced_article)
        
        return enhanced_articles

    def _enhance_search_query(self, query: str) -> str:
        """Enhance search query with conflict-specific terms and operators."""
        # Add conflict-related context if not present
        conflict_terms = ['conflict', 'violence', 'security', 'crisis', 'peace']

        if not any(term in query.lower() for term in conflict_terms):
            # Add contextual terms with OR operator
            enhanced = f"{query} OR (conflict OR security OR crisis)"
        else:
            enhanced = query

        # Clean and optimize query
        enhanced = re.sub(r'\s+', ' ', enhanced).strip()

        return enhanced

    async def _search_google_news_rss(
        self,
        query: str,
        countries: List[str],
        languages: List[str],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Search Google News RSS with enhanced stealth and filtering."""
        articles = []

        # Search across multiple country/language combinations
        for country in countries[:3]:  # Limit to avoid rate limiting
            for language in languages[:2]:
                try:
                    # Construct Google News RSS URL
                    params = {
                        'q': quote_plus(query),
                        'hl': language,
                        'gl': country,
                        'ceid': f"{country}:{language}"
                    }

                    url = f"{self.google_base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])

                    # Use stealth capabilities if available
                    if self.stealth_orchestrator:
                        response_content = await self._fetch_with_stealth(url)
                    else:
                        response_content = await self._fetch_with_basic_client(url)

                    if response_content:
                        # Parse RSS feed
                        feed = feedparser.parse(response_content)

                        for entry in feed.entries[:max_results // len(countries)]:
                            article = self._parse_rss_entry(entry, country, language)
                            if article:
                                articles.append(article)

                    # Rate limiting
                    await asyncio.sleep(2.0)

                except Exception as e:
                    logger.debug(f"Google News search failed for {country}/{language}: {e}")
                    continue

        logger.info(f"Retrieved {len(articles)} articles from Google News RSS")
        return articles

    async def _search_regional_sources(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search configured regional news sources."""
        articles = []

        # Process active sources in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def search_source(domain: str):
            async with semaphore:
                try:
                    source = self.news_sources[domain]
                    source_articles = await self._search_single_source(source, query)
                    return source_articles
                except Exception as e:
                    logger.debug(f"Search failed for source {domain}: {e}")
                    return []

        # Execute searches
        tasks = [search_source(domain) for domain in self.active_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, list):
                articles.extend(result)

        logger.info(f"Retrieved {len(articles)} articles from regional sources")
        return articles[:max_results]

    async def _search_single_source(self, source: NewsSource, query: str, country: str = 'US', language: str = 'en') -> List[Dict[str, Any]]:
        """Search a single news source."""
        articles = []
        search_start = time.time()

        try:
            # Search RSS feeds
            for rss_url in source.config.rss_feeds:
                try:
                    if self.stealth_orchestrator:
                        content = await self._fetch_with_stealth(rss_url)
                    else:
                        content = await self._fetch_with_basic_client(rss_url)

                    if content and feedparser:
                        feed = feedparser.parse(content)

                        for entry in feed.entries:
                            if self._matches_query(entry, query):
                                article = self._parse_rss_entry(entry, rss_url, country or 'US', language or 'en')
                                if article:
                                    article['source_domain'] = source.config.domain
                                    article['source_name'] = source.config.name
                                    article['source_type'] = source.config.source_type.value
                                    articles.append(article)
                    elif not feedparser:
                        logger.warning("feedparser not available - skipping RSS parsing")

                except Exception as e:
                    logger.debug(f"RSS search failed for {rss_url}: {e}")
                    continue

            # Update source performance metrics
            search_time = (time.time() - search_start) * 1000
            source.update_performance_metrics(True, search_time)

        except Exception as e:
            search_time = (time.time() - search_start) * 1000
            source.update_performance_metrics(False, search_time)
            logger.debug(f"Source search failed for {source.config.domain}: {e}")

        return articles

    def _matches_query(self, entry, query: str) -> bool:
        """Check if RSS entry matches search query."""
        query_terms = query.lower().split()

        # Combine title and description for matching
        text_content = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()

        # Simple keyword matching (could be enhanced with more sophisticated matching)
        return any(term in text_content for term in query_terms)

    async def _fetch_with_stealth(self, url: str) -> Optional[str]:
        """Fetch URL using stealth capabilities."""
        if not self.stealth_orchestrator:
            return await self._fetch_with_basic_client(url)

        try:
            # Use your existing stealth orchestrator
            # This is a placeholder - integrate with your actual stealth system
            response = await self.stealth_orchestrator.handle_request_with_stealth(url, None)
            return response
        except Exception as e:
            logger.debug(f"Stealth fetch failed for {url}: {e}")
            return None

    async def _fetch_with_basic_client(self, url: str) -> Optional[str]:
        """Fetch URL using basic HTTP client with realistic headers."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        }

        try:
            # Use managed session if available, otherwise create temporary one
            if self._http_session and not self._http_session.closed:
                session = self._http_session
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.debug(f"HTTP {response.status} for {url}")
                        return None
            else:
                # Fallback to temporary session
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            logger.debug(f"HTTP {response.status} for {url}")
                            return None
        except Exception as e:
            logger.debug(f"Basic fetch failed for {url}: {e}")
            return None

    def _parse_rss_entry(self, entry, source_url: str, country: str, language: str) -> Optional[Dict[str, Any]]:
        """Parse RSS entry into standardized article format."""
        try:
            # Extract published date
            published_date = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'published'):
                try:
                    published_date = datetime.fromisoformat(entry.published.replace('Z', '+00:00'))
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Date parsing failed: {e}")
                    published_date = datetime.now(timezone.utc)
            else:
                published_date = datetime.now(timezone.utc)

            # Extract URL (handle Google News redirects)
            url = entry.link
            if 'news.google.com' in url and 'url=' in url:
                # Extract actual URL from Google redirect
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                if 'url' in parsed:
                    url = parsed['url'][0]

            # Create article object
            article = {
                'title': entry.title,
                'description': getattr(entry, 'summary', ''),
                'url': url,
                'published_date': published_date,
                'publisher': self._extract_publisher(entry),
                'language': language,
                'country': country,
                'source_type': 'rss',
                'content_hash': hashlib.md5(
                    (entry.title + url).encode('utf-8')
                ).hexdigest(),
                'discovery_time': datetime.now(timezone.utc),
                'tags': getattr(entry, 'tags', []),
                'categories': [tag.term for tag in getattr(entry, 'tags', [])]
            }

            return article

        except Exception as e:
            logger.debug(f"Failed to parse RSS entry: {e}")
            return None

    def _extract_publisher(self, entry) -> Dict[str, str]:
        """Extract publisher information from RSS entry."""
        publisher = {'name': 'Unknown', 'domain': 'unknown.com'}

        try:
            if hasattr(entry, 'source') and entry.source:
                publisher['name'] = entry.source.get('title', 'Unknown')
                if 'href' in entry.source:
                    domain = urlparse(entry.source['href']).netloc
                    publisher['domain'] = domain
            else:
                # Extract from URL
                domain = urlparse(entry.link).netloc.lower()
                publisher['domain'] = domain
                publisher['name'] = domain.replace('www.', '').title()
        except Exception as e:
            logger.warning(f"Publisher extraction failed: {e}")
            pass

        return publisher

    def _apply_content_filtering(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply content quality and relevance filtering."""
        filtered = []

        for article in articles:
            # Basic quality checks
            if not self._passes_quality_checks(article):
                continue

            # Language filtering
            if article.get('language') and article['language'] not in self.content_filters['language_filters']:
                continue

            # Domain filtering
            domain = urlparse(article['url']).netloc.lower()
            if domain in self.content_filters['excluded_domains']:
                continue

            # Content relevance scoring
            relevance_score = self._calculate_relevance_score(article)
            article['relevance_score'] = relevance_score

            if relevance_score >= 0.3:  # Threshold for inclusion
                filtered.append(article)

        return filtered

    def _passes_quality_checks(self, article: Dict[str, Any]) -> bool:
        """Check if article passes basic quality thresholds."""
        # Check required fields
        for field in self.content_filters['required_fields']:
            if not article.get(field):
                return False

        # Check title length
        title_length = len(article['title'])
        if not (self.content_filters['min_title_length'] <= title_length <= self.content_filters['max_title_length']):
            return False

        # Check description length
        if article.get('description'):
            desc_length = len(article['description'])
            if desc_length < self.content_filters['min_description_length']:
                return False

        # Check URL validity
        try:
            parsed_url = urlparse(article['url'])
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return False
        except Exception as e:
            logger.warning(f"RSS entry parsing failed: {e}")
            return False

        return True

    def _calculate_relevance_score(self, article: Dict[str, Any]) -> float:
        """Calculate relevance score for conflict/CEWS context."""
        score = 0.5  # Base score

        # Text for analysis
        text = f"{article['title']} {article.get('description', '')}".lower()

        # Conflict-related keywords (weighted)
        conflict_keywords = {
            'conflict': 0.3, 'violence': 0.3, 'war': 0.3, 'fighting': 0.2,
            'attack': 0.2, 'bombing': 0.2, 'shooting': 0.2, 'killing': 0.2,
            'crisis': 0.2, 'emergency': 0.2, 'humanitarian': 0.2,
            'refugee': 0.15, 'displacement': 0.15, 'evacuation': 0.15,
            'protest': 0.1, 'unrest': 0.1, 'demonstration': 0.1,
            'election': 0.1, 'political': 0.05, 'government': 0.05
        }

        for keyword, weight in conflict_keywords.items():
            if keyword in text:
                score += weight

        # Geographic relevance
        geographic_terms = [
            'kenya', 'ethiopia', 'somalia', 'sudan', 'uganda', 'tanzania',
            'eritrea', 'djibouti', 'horn of africa', 'east africa'
        ]

        if any(term in text for term in geographic_terms):
            score += 0.2

        # Temporal relevance (newer articles get higher scores)
        if article.get('published_date'):
            age_hours = (datetime.now(timezone.utc) - article['published_date']).total_seconds() / 3600
            temporal_score = max(0, 1 - (age_hours / 168))  # Decay over a week
            score += temporal_score * 0.1

        return min(score, 1.0)

    def _deduplicate_and_rank(self, articles: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Remove duplicates and rank articles by relevance."""
        # Deduplicate by content hash and URL
        seen_hashes = set()
        seen_urls = set()
        unique_articles = []

        for article in articles:
            url_normalized = article['url'].lower().strip()
            content_hash = article.get('content_hash', '')

            if url_normalized in seen_urls or content_hash in seen_hashes:
                continue

            seen_urls.add(url_normalized)
            if content_hash:
                seen_hashes.add(content_hash)

            unique_articles.append(article)

        # Rank by relevance score (with secondary criteria)
        def ranking_key(article):
            relevance = article.get('relevance_score', 0.5)
            recency = 1.0
            if article.get('published_date'):
                age_hours = (datetime.now(timezone.utc) - article['published_date']).total_seconds() / 3600
                recency = max(0.1, 1 - (age_hours / 720))  # Decay over 30 days

            return relevance * 0.7 + recency * 0.3

        ranked_articles = sorted(unique_articles, key=ranking_key, reverse=True)

        return ranked_articles[:max_results]

    async def _enhance_articles_metadata(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance articles with additional metadata and analysis."""
        enhanced = []

        for article in articles:
            try:
                # Add sentiment analysis
                text_for_analysis = f"{article['title']} {article.get('description', '')}"

                try:
                    if TextBlob:
                        blob = TextBlob(text_for_analysis)
                        article['sentiment_polarity'] = blob.sentiment.polarity
                        article['sentiment_subjectivity'] = blob.sentiment.subjectivity
                    else:
                        article['sentiment_polarity'] = 0.0
                        article['sentiment_subjectivity'] = 0.5
                except Exception as e:
                    logger.debug(f"Sentiment analysis failed: {e}")
                    article['sentiment_polarity'] = 0.0
                    article['sentiment_subjectivity'] = 0.5

                # Add language detection
                try:
                    if not article.get('language') and 'blob' in locals():
                        article['language'] = blob.detect_language()
                    elif not article.get('language'):
                        article['language'] = 'en'
                except Exception as e:
                    logger.debug(f"Language detection failed: {e}")
                    article['language'] = 'en'

                # Add source credibility if available
                domain = urlparse(article['url']).netloc.lower()
                if domain in self.news_sources:
                    source = self.news_sources[domain]
                    article['source_health_score'] = source.get_health_score()
                    if source.config.credibility_metrics:
                        article['source_credibility_score'] = source.config.credibility_metrics.calculate_composite_score()

                # Add conflict indicators
                article['conflict_indicators'] = self._extract_conflict_indicators(text_for_analysis)

                enhanced.append(article)

            except Exception as e:
                logger.debug(f"Article enhancement failed: {e}")
                enhanced.append(article)  # Include original article

        return enhanced

    def _extract_conflict_indicators(self, text: str) -> List[str]:
        """Extract conflict-related indicators from text."""
        indicators = []
        text_lower = text.lower()

        # Define indicator patterns
        indicator_patterns = {
            'violence': ['violence', 'violent', 'attack', 'assault', 'killing', 'murder'],
            'military': ['military', 'army', 'soldiers', 'troops', 'armed forces'],
            'weapons': ['weapons', 'guns', 'bombs', 'explosives', 'artillery'],
            'casualties': ['casualties', 'deaths', 'wounded', 'injured', 'fatalities'],
            'displacement': ['refugees', 'displaced', 'evacuation', 'fleeing'],
            'protest': ['protest', 'demonstration', 'riots', 'unrest'],
            'political': ['election', 'government', 'political', 'coup', 'regime'],
            'humanitarian': ['humanitarian', 'aid', 'crisis', 'emergency']
        }

        for category, patterns in indicator_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                indicators.append(category)

        return indicators

    async def scrape_full_articles(
        self,
        articles: List[Dict[str, Any]],
        use_stealth: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Scrape full article content using existing scraper infrastructure.

        Args:
            articles: List of articles to scrape
            use_stealth: Whether to use stealth crawling capabilities

        Returns:
            Articles enhanced with full content
        """
        # Content scraper not available - return articles as-is with basic metadata
        logger.warning("Content scraper not available, returning articles without full text")

        enhanced_articles = []
        for article in articles.copy():
            article['processing_stage'] = 'BASIC_INFO_ONLY'
            article['full_content'] = None
            article['content_quality_score'] = 0.0
            article['scraping_error'] = None
            enhanced_articles.append(article)

        return enhanced_articles

    async def store_articles_to_database(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Store articles to database using existing PostgreSQL manager.

        Args:
            articles: List of articles to store

        Returns:
            Dictionary with storage statistics
        """
        if not articles:
            return {'stored': 0, 'failed': 0, 'skipped': 0}

        logger.info(f"Storing {len(articles)} articles to database")

        # Get data source ID
        data_source_id = await self.db_manager.get_data_source_id("Enhanced Google News")

        # Convert articles to database format
        information_units = []

        for article in articles:
            try:
                # Create unique external ID
                url_hash = hashlib.md5(article['url'].encode()).hexdigest()[:16]
                external_id = f"egnews_{url_hash}"

                # Map article to information unit format
                unit = {
                    'data_source_id': data_source_id,
                    'external_id': external_id,
                    'title': article['title'],
                    'content_url': article['url'],
                    'content': article.get('full_content'),
                    'published_at': article['published_date'],
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'language_code': article.get('language', 'en'),
                    'summary': article.get('description', ''),
                    'relevance_score': article.get('relevance_score', 0.5),
                    'confidence_score': article.get('source_credibility_score', 0.5),
                    'sentiment_score': article.get('sentiment_polarity', 0.0),
                    'keywords': article.get('conflict_indicators', []),
                    'tags': article.get('categories', []),
                    'metadata': {
                        'publisher': article.get('publisher', {}),
                        'source_type': article.get('source_type'),
                        'discovery_method': 'enhanced_google_news',
                        'content_quality_score': article.get('content_quality_score'),
                        'sentiment_subjectivity': article.get('sentiment_subjectivity'),
                        'conflict_indicators': article.get('conflict_indicators', []),
                        'source_health_score': article.get('source_health_score'),
                        'processing_stage': article.get('processing_stage', 'DISCOVERED')
                    }
                }

                information_units.append(unit)

            except Exception as e:
                logger.error(f"Failed to convert article to database format: {e}")
                continue

        # Store using generic database operations
        try:
            # Use generic insert_batch method if available
            if hasattr(self.db_manager, 'insert_batch'):
                result = await self.db_manager.insert_batch(
                    'information_units',
                    information_units
                )
                return {
                    'stored': len(information_units),
                    'updated': 0,
                    'errors': 0
                }
            else:
                # Fallback to basic storage tracking
                logger.warning("Database manager lacks bulk insert capability")
                return {
                    'stored': 0,
                    'updated': 0,
                    'errors': len(information_units)
                }

            logger.info(f"Database storage completed: {storage_stats}")
            return storage_stats

        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            return {'stored': 0, 'failed': len(articles), 'skipped': 0}

    async def get_source_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for all news sources."""
        report = {
            'total_sources': len(self.news_sources),
            'active_sources': len(self.active_sources),
            'source_health': {},
            'performance_summary': {
                'healthy_sources': 0,
                'degraded_sources': 0,
                'failed_sources': 0,
                'avg_response_time': 0.0,
                'overall_success_rate': 0.0
            },
            'geographical_coverage': {},
            'content_type_distribution': {},
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        total_response_time = 0.0
        total_requests = 0
        total_successes = 0

        for domain, source in self.news_sources.items():
            health_score = source.get_health_score()
            metrics = source.performance_metrics

            source_info = {
                'name': source.config.name,
                'domain': domain,
                'health_score': health_score,
                'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.3 else 'failed',
                'total_requests': metrics['total_requests'],
                'success_rate': metrics['successful_requests'] / max(1, metrics['total_requests']),
                'avg_response_time_ms': metrics['avg_response_time'],
                'consecutive_failures': metrics['consecutive_failures'],
                'last_success': metrics['last_success_time'].isoformat() if metrics['last_success_time'] else None,
                'source_type': source.config.source_type.value,
                'geographical_focus': [country for country in source.config.geographical_focus.countries] if source.config.geographical_focus else [],
                'feeds_count': len(source.config.rss_feeds),
                'credibility_score': source.config.credibility_metrics.calculate_composite_score() if source.config.credibility_metrics else 0.5
            }

            report['source_health'][domain] = source_info

            # Update performance summary
            if health_score > 0.7:
                report['performance_summary']['healthy_sources'] += 1
            elif health_score > 0.3:
                report['performance_summary']['degraded_sources'] += 1
            else:
                report['performance_summary']['failed_sources'] += 1

            # Aggregate metrics
            total_response_time += metrics['avg_response_time']
            total_requests += metrics['total_requests']
            total_successes += metrics['successful_requests']

            # Geographical coverage
            if source.config.geographical_focus:
                for country in source.config.geographical_focus.countries:
                    if country not in report['geographical_coverage']:
                        report['geographical_coverage'][country] = 0
                    report['geographical_coverage'][country] += 1

            # Content type distribution
            content_type = source.config.content_format.value
            if content_type not in report['content_type_distribution']:
                report['content_type_distribution'][content_type] = 0
            report['content_type_distribution'][content_type] += 1

        # Calculate overall metrics
        if len(self.news_sources) > 0:
            report['performance_summary']['avg_response_time'] = total_response_time / len(self.news_sources)

        if total_requests > 0:
            report['performance_summary']['overall_success_rate'] = total_successes / total_requests

        return report

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        runtime_hours = (datetime.now(timezone.utc) - self.session_stats['start_time']).total_seconds() / 3600

        return {
            'session_duration_hours': runtime_hours,
            'total_searches': self.session_stats['total_searches'],
            'successful_searches': self.session_stats['successful_searches'],
            'search_success_rate': self.session_stats['successful_searches'] / max(1, self.session_stats['total_searches']),
            'articles_discovered': self.session_stats['articles_discovered'],
            'articles_filtered': self.session_stats['articles_filtered'],
            'articles_scraped': self.session_stats['articles_scraped'],
            'avg_articles_per_search': self.session_stats['articles_discovered'] / max(1, self.session_stats['successful_searches']),
            'filtering_efficiency': self.session_stats['articles_filtered'] / max(1, self.session_stats['articles_discovered']),
            'scraping_coverage': self.session_stats['articles_scraped'] / max(1, self.session_stats['articles_filtered']),
            'active_sources': len(self.active_sources),
            'total_configured_sources': len(self.news_sources)
        }

    async def close(self):
        """Clean up resources and close connections."""
        logger.info("Closing Enhanced Google News client...")

        try:
            # Generate final reports
            final_stats = self.get_session_statistics()
            source_health = await self.get_source_health_report()

            logger.info(f"Final session statistics: {json.dumps(final_stats, indent=2)}")
            logger.info(f"Source health summary: {source_health['performance_summary']}")

            # Close HTTP session properly
            if self._http_session and not self._http_session.closed:
                await self._http_session.close()
                # Wait for underlying connections to close
                await asyncio.sleep(0.1)

            # Close any open connections
            for source in self.news_sources.values():
                # Cleanup source-specific resources if needed
                pass

            logger.info("Enhanced Google News client closed successfully")

        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")
            # Force close session if it exists
            if self._http_session and not self._http_session.closed:
                try:
                    await self._http_session.close()
                except:
                    pass


# Configuration Management
class NewsSourceConfig(BaseModel):
    """Pydantic model for news source configuration validation."""

    sources: List[Dict[str, Any]]
    filtering: Dict[str, Any] = Field(default_factory=dict)
    global_settings: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        """Validate source configurations."""
        if not v:
            raise ValueError("At least one news source must be configured")

        required_fields = ['name', 'domain', 'source_type']
        for source in v:
            for field in required_fields:
                if field not in source:
                    raise ValueError(f"Missing required field '{field}' in source configuration")

        return v


def load_source_configuration(config_path: str) -> NewsSourceConfig:
    """Load and validate news source configuration from file."""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        return NewsSourceConfig(**config_data)

    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


# Factory Functions
async def create_gnews_client(
    db_manager: PgSQLManager,
    stealth_orchestrator=None,
    crawlee_enhancer: Optional[CrawleeNewsEnhancer] = None,
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None
) -> GoogleNewsClient:
    """
    Factory function to create and initialize an Enhanced Google News client.

    Args:
        db_manager: Initialized PostgreSQL database manager
        stealth_orchestrator: Optional stealth orchestrator for advanced crawling
        crawlee_enhancer: Optional Crawlee enhancer for full content downloading
        config_path: Path to configuration file
        config_dict: Direct configuration dictionary

    Returns:
        Initialized EnhancedGoogleNewsClient
    """
    # Load configuration
    if config_path:
        config = load_source_configuration(config_path)
        client_config = config.dict()
    elif config_dict:
        client_config = config_dict
    else:
        client_config = {}

    # Create client
    client = GoogleNewsClient(
        db_manager=db_manager,
        stealth_orchestrator=stealth_orchestrator,
        crawlee_enhancer=crawlee_enhancer,
        config=client_config
    )

    # Initialize
    await client.initialize()

    logger.info("Enhanced Google News client created and initialized")
    return client


async def create_basic_gnews_client(
    db_manager: PgSQLManager,
    crawlee_enhancer: Optional[CrawleeNewsEnhancer] = None
) -> GoogleNewsClient:
    """Create basic Enhanced Google News client with default configuration."""
    return await create_enhanced_gnews_client(
        db_manager=db_manager,
        stealth_orchestrator=None,
        crawlee_enhancer=crawlee_enhancer,
        config_dict={
            'filtering': {
                'min_authority_score': 0.3,
                'min_reliability_score': 0.4
            }
        }
    )


async def create_crawlee_enhanced_gnews_client(
    db_manager: PgSQLManager,
    stealth_orchestrator=None,
    crawlee_config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None
) -> GoogleNewsClient:
    """
    Factory function to create Enhanced Google News client with Crawlee integration.
    
    Args:
        db_manager: Initialized PostgreSQL database manager
        stealth_orchestrator: Optional stealth orchestrator for advanced crawling  
        crawlee_config: Configuration for Crawlee enhancer
        config_path: Path to configuration file
        config_dict: Direct configuration dictionary
        
    Returns:
        Initialized EnhancedGoogleNewsClient with Crawlee enhancer
    """
    crawlee_enhancer = None
    
    if CRAWLEE_INTEGRATION_AVAILABLE:
        try:
            from ..crawlee_integration import create_crawlee_enhancer, create_crawlee_config
            
            # Create Crawlee configuration
            if crawlee_config:
                crawlee_cfg = create_crawlee_config(**crawlee_config)
            else:
                # Default configuration optimized for news content
                crawlee_cfg = create_crawlee_config(
                    max_requests=50,
                    max_concurrent=3,
                    target_countries=["ET", "SO", "KE", "UG", "TZ"],
                    enable_full_content=True,
                    min_content_length=300,
                    enable_content_scoring=True
                )
            
            # Create and initialize Crawlee enhancer
            crawlee_enhancer = await create_crawlee_enhancer(crawlee_cfg)
            logger.info("✅ Crawlee enhancer initialized for Google News client")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Crawlee enhancer: {e}")
            crawlee_enhancer = None
    else:
        logger.warning("Crawlee integration not available. Client will work without content enhancement.")
    
    # Create Google News client with Crawlee enhancer
    return await create_enhanced_gnews_client(
        db_manager=db_manager,
        stealth_orchestrator=stealth_orchestrator,
        crawlee_enhancer=crawlee_enhancer,
        config_path=config_path,
        config_dict=config_dict
    )


# Integration Wrapper for Backward Compatibility
class GNewsCompatibilityWrapper:
    """
    Compatibility wrapper providing original GNews-like interface
    while using the enhanced implementation internally.
    """

    def __init__(
        self,
        language: str = 'en',
        country: str = 'US',
        max_results: int = 100,
        period: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exclude_websites: Optional[List[str]] = None,
        proxy: Optional[Dict[str, str]] = None,
        db_manager: Optional[PgSQLManager] = None,
        stealth_orchestrator = None
    ):
        """Initialize compatibility wrapper with original GNews-like interface."""
        self.language = language
        self.country = country
        self.max_results = max_results
        self.period = period
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_websites = exclude_websites or []
        self.proxy = proxy

        # Enhanced client (will be initialized on first use)
        self._enhanced_client = None
        self._db_manager = db_manager
        self._stealth_orchestrator = stealth_orchestrator

        # Country code mapping
        self._country_mapping = {
            'US': ['US'], 'GB': ['GB'], 'KE': ['KE'], 'ET': ['ET'],
            'SO': ['SO'], 'SD': ['SD'], 'UG': ['UG'], 'TZ': ['TZ']
        }

    async def _ensure_client_initialized(self):
        """Ensure enhanced client is initialized."""
        if self._enhanced_client is None:
            if self._db_manager is None:
                raise ValueError("Database manager is required for enhanced functionality")

            self._enhanced_client = await create_enhanced_gnews_client(
                db_manager=self._db_manager,
                stealth_orchestrator=self._stealth_orchestrator
            )

    async def get_news(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Get news articles matching keyword (enhanced implementation).

        Maintains compatibility with original GNews interface while providing
        enhanced functionality internally.
        """
        await self._ensure_client_initialized()

        # Map country to country list
        countries = self._country_mapping.get(self.country, [self.country])

        # Calculate time range if period is specified
        time_range = None
        if self.period:
            end_time = datetime.now(timezone.utc)
            if self.period.endswith('h'):
                hours = int(self.period[:-1])
                start_time = end_time - timedelta(hours=hours)
            elif self.period.endswith('d'):
                days = int(self.period[:-1])
                start_time = end_time - timedelta(days=days)
            elif self.period.endswith('m'):
                months = int(self.period[:-1])
                start_time = end_time - timedelta(days=months * 30)
            else:
                start_time = end_time - timedelta(days=7)  # Default to 1 week

            time_range = (start_time, end_time)
        elif self.start_date and self.end_date:
            time_range = (self.start_date, self.end_date)

        # Search using enhanced client
        if self._enhanced_client:
            articles = await self._enhanced_client.search_news(
                query=keyword,
                countries=countries,
                languages=[self.language],
                max_results=self.max_results,
                time_range=time_range
            )

        # Convert to original GNews format for compatibility
        compatible_articles = []
        for article in articles:
            compatible_article = {
                'title': article['title'],
                'description': article.get('description', ''),
                'published date': article['published_date'].strftime('%a, %d %b %Y %H:%M:%S GMT'),
                'url': article['url'],
                'publisher': article.get('publisher', {})
            }
            compatible_articles.append(compatible_article)

        return compatible_articles

    async def get_news_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get news by topic category."""
        # Map topic to search query
        topic_queries = {
            'WORLD': 'international world news',
            'BUSINESS': 'business economy finance',
            'TECHNOLOGY': 'technology tech innovation',
            'POLITICS': 'politics government election',
            'HEALTH': 'health medical healthcare',
            'SCIENCE': 'science research discovery'
        }

        query = topic_queries.get(topic.upper(), topic)
        return await self.get_news(query)

    async def get_news_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Get news by geographical location."""
        return await self.get_news(f"news from {location}")

    async def get_news_by_site(self, site: str) -> List[Dict[str, Any]]:
        """Get news from specific website."""
        return await self.get_news(f"site:{site}")

    async def get_full_article(self, url: str) -> Optional[Any]:
        """
        Get full article content using enhanced scraping capabilities.

        Returns article object compatible with newspaper3k format.
        """
        await self._ensure_client_initialized()

        # Create temporary article for scraping
        temp_article = {
            'url': url,
            'title': 'Unknown',
            'description': '',
            'published_date': datetime.now(timezone.utc),
            'publisher': {'name': 'Unknown'}
        }

        # Scrape using enhanced client
        scraped_articles = await self._enhanced_client.scrape_full_articles([temp_article])

        if scraped_articles and scraped_articles[0].get('full_content'):
            # Create newspaper3k-like object
            class Article:
                def __init__(self, content_data):
                    self.title = content_data.get('title', '')
                    self.text = content_data.get('full_content', '')
                    self.url = content_data.get('url', '')
                    self.authors = []
                    self.publish_date = content_data.get('published_date')
                    self.images = set()
                    self.movies = []
                    self.keywords = content_data.get('keywords', [])
                    self.summary = content_data.get('description', '')

                def download(self):
                    pass  # Already downloaded

                def parse(self):
                    pass  # Already parsed

            return Article(scraped_articles[0])

        return None


# Example Configuration Files

def create_sample_configuration() -> Dict[str, Any]:
    """Create sample configuration for Enhanced Google News client."""
    return {
        "sources": [
            {
                "name": "Kenya Standard Media",
                "domain": "standardmedia.co.ke",
                "source_type": "mainstream_media",
                "content_format": "rss_feed",
                "rss_feeds": [
                    "https://www.standardmedia.co.ke/rss/headlines.php"
                ],
                "geographical_focus": {
                    "countries": ["Kenya"],
                    "regions": ["East Africa"],
                    "priority_multiplier": 1.2
                },
                "topic_categories": ["politics", "security", "business"],
                "crawl_priority": "HIGH",
                "requests_per_minute": 20
            }
        ],
        "filtering": {
            "min_authority_score": 0.3,
            "min_reliability_score": 0.4,
            "max_bias_deviation": 0.7,
            "min_success_rate": 0.7
        },
        "global_settings": {
            "default_language": "en",
            "default_max_results": 100,
            "enable_content_scraping": True,
            "enable_stealth_mode": True,
            "cache_duration_minutes": 30
        }
    }


# Example Usage and Testing
if __name__ == "__main__":
    async def example_usage():
        """Example usage of Enhanced Google News implementation."""

        # Initialize database manager (using your existing system)
        db_manager = PgSQLManager(
            connection_string="postgresql:///lnd",
            enable_spatial_operations=True,
            enable_event_extraction=True
        )
        await db_manager.initialize()

        # Create enhanced client
        enhanced_client = await create_enhanced_gnews_client(
            db_manager=db_manager,
            config_dict=create_sample_configuration()
        )

        try:
            # Search for conflict-related news in Horn of Africa
            articles = await enhanced_client.search_news(
                query="conflict violence security crisis",
                countries=['KE', 'ET', 'SO', 'SD'],
                languages=['en', 'fr'],
                max_results=50
            )

            print(f"Found {len(articles)} articles")

            # Scrape full content for top articles
            top_articles = articles[:10]
            scraped_articles = await enhanced_client.scrape_full_articles(top_articles)

            print(f"Scraped {len([a for a in scraped_articles if a.get('full_content')])} articles")

            # Store to database
            storage_stats = await enhanced_client.store_articles_to_database(scraped_articles)
            print(f"Storage statistics: {storage_stats}")

            # Get health reports
            health_report = await enhanced_client.get_source_health_report()
            session_stats = enhanced_client.get_session_statistics()

            print("Source Health Summary:")
            print(f"- Healthy sources: {health_report['performance_summary']['healthy_sources']}")
            print(f"- Overall success rate: {health_report['performance_summary']['overall_success_rate']:.2%}")

            print("Session Statistics:")
            print(f"- Articles per search: {session_stats['avg_articles_per_search']:.1f}")
            print(f"- Filtering efficiency: {session_stats['filtering_efficiency']:.2%}")

        except Exception as e:
            print(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure proper cleanup
            try:
                await enhanced_client.close()
            except Exception as e:
                print(f"Error closing client: {e}")

            try:
                await db_manager.close()
            except Exception as e:
                print(f"Error closing database: {e}")

    # Example with compatibility wrapper
    async def compatibility_example():
        """Example using compatibility wrapper for easy migration."""

        # Note: You need to provide your actual db_manager instance
        try:
            # Initialize your database manager
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'your_database',
                'user': 'your_user',
                'password': 'your_password'
            }
            db_manager = PgSQLManager(db_config)

            # Initialize with GNews-like interface
            gnews_compatible = GNewsCompatibilityWrapper(
                language='en',
                country='KE',
                max_results=20,
                period='7d',
                db_manager=db_manager
            )

            # Use like original GNews
            articles = await gnews_compatible.get_news("Kenya election security")

            for article in articles[:3]:
                print(f"Title: {article['title']}")
                print(f"URL: {article['url']}")
                print(f"Published: {article['published date']}")
                print("-" * 50)

        except Exception as e:
            print(f"Example failed: {e}")
            print("Please configure your database connection and ensure all dependencies are installed.")

    # Run examples - commented out to prevent automatic execution
    # Uncomment the line below to run the example manually
    # asyncio.run(example_usage())

    print("""
Enhanced Google News Implementation Loaded Successfully!
======================================================

To run examples manually:
1. python test_gnews_fixes.py  (recommended for testing)
2. Uncomment the asyncio.run() line above for full example

The automatic execution has been disabled to prevent SSL timeout errors.
Use the test script for safe testing of functionality.
""")
