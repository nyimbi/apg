#!/usr/bin/env python3
"""
Comprehensive Test Suite for Google News Crawler
===============================================

Tests the Google News crawler functionality including client operations,
source filtering, article scraping, and metadata enhancement.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import pytest
import asyncio
import sys
import os
import json
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import logging
import aiohttp
import feedparser
from dataclasses import dataclass

# Add the packages_enhanced directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

try:
    from crawlers.google_news_crawler import (
        create_enhanced_gnews_client,
        create_basic_gnews_client,
        enhanced_gnews_implementation
    )
    from crawlers.google_news_crawler.api.google_news_client import (
        EnhancedGoogleNewsClient,
        NewsSource,
        EnhancedNewsSource,
        SiteFilteringEngine,
        SourceType,
        ContentFormat,
        CrawlPriority,
        GeographicalFocus,
        SourceCredibilityMetrics,
        GNewsCompatibilityWrapper
    )
    from crawlers.google_news_crawler.config import (
        NewsSourceConfig,
        load_source_configuration,
        create_sample_configuration
    )
    GOOGLE_NEWS_CRAWLER_AVAILABLE = True
except ImportError:
    GOOGLE_NEWS_CRAWLER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_news_config():
    """Return a sample news source configuration."""
    return {
        "sources": [
            {
                "name": "BBC News",
                "base_url": "https://www.bbc.com/news",
                "source_type": "MAINSTREAM_MEDIA",
                "content_format": "ARTICLE",
                "priority": "HIGH",
                "geographical_focus": "GLOBAL",
                "languages": ["en"],
                "rss_feeds": ["https://feeds.bbci.co.uk/news/world/rss.xml"],
                "reliability_score": 0.9,
                "bias_level": 0.2
            },
            {
                "name": "CNN",
                "base_url": "https://www.cnn.com",
                "source_type": "MAINSTREAM_MEDIA",
                "content_format": "ARTICLE",
                "priority": "HIGH",
                "geographical_focus": "GLOBAL",
                "languages": ["en"],
                "rss_feeds": ["http://rss.cnn.com/rss/edition_world.rss"],
                "reliability_score": 0.85,
                "bias_level": 0.3
            }
        ],
        "filtering": {
            "min_reliability_score": 0.7,
            "max_bias_level": 0.5,
            "preferred_languages": ["en"],
            "blocked_domains": []
        },
        "crawl_settings": {
            "max_articles_per_source": 10,
            "max_age_days": 7,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "request_delay": 1.0,
            "timeout": 30,
            "respect_robots_txt": True
        }
    }


@pytest.fixture
def sample_rss_content():
    """Return sample RSS feed content."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Sample News Feed</title>
    <link>https://www.samplenews.com</link>
    <description>Latest news from Sample News</description>
    <language>en-us</language>
    <item>
      <title>Major Conflict Erupts in Ethiopia</title>
      <link>https://www.samplenews.com/world/ethiopia-conflict-12345</link>
      <guid>https://www.samplenews.com/world/ethiopia-conflict-12345</guid>
      <pubDate>Mon, 24 Apr 2023 09:30:00 GMT</pubDate>
      <description>Breaking news: Major conflict has erupted in the northern region of Ethiopia with reports of casualties.</description>
      <source url="https://www.samplenews.com">Sample News</source>
      <category>World</category>
    </item>
    <item>
      <title>Peace Talks Begin in South Sudan</title>
      <link>https://www.samplenews.com/world/south-sudan-peace-67890</link>
      <guid>https://www.samplenews.com/world/south-sudan-peace-67890</guid>
      <pubDate>Mon, 24 Apr 2023 08:15:00 GMT</pubDate>
      <description>Peace negotiations have begun between rival factions in South Sudan after months of tension.</description>
      <source url="https://www.samplenews.com">Sample News</source>
      <category>World</category>
    </item>
  </channel>
</rss>"""


@pytest.fixture
def sample_article_html():
    """Return sample article HTML content."""
    return """<!DOCTYPE html>
<html>
<head>
    <title>Major Conflict Erupts in Ethiopia - Sample News</title>
    <meta name="description" content="Breaking news: Major conflict has erupted in the northern region of Ethiopia with reports of casualties.">
    <meta property="og:title" content="Major Conflict Erupts in Ethiopia">
    <meta property="og:description" content="Breaking news: Major conflict has erupted in the northern region of Ethiopia with reports of casualties.">
    <meta property="og:url" content="https://www.samplenews.com/world/ethiopia-conflict-12345">
    <meta property="article:published_time" content="2023-04-24T09:30:00Z">
    <meta property="article:author" content="John Smith">
</head>
<body>
    <header>
        <h1>Major Conflict Erupts in Ethiopia</h1>
        <div class="metadata">
            <span class="author">By John Smith</span>
            <span class="date">April 24, 2023</span>
            <span class="category">World</span>
        </div>
    </header>
    <article>
        <p>Breaking news: Major conflict has erupted in the northern region of Ethiopia with reports of casualties.</p>
        <p>The conflict began early Monday morning when armed groups clashed near the border. Local sources report at least 15 casualties and dozens wounded.</p>
        <p>International observers have expressed concern over the escalation of violence in the region, which has seen increasing tensions in recent months.</p>
        <p>The Ethiopian government has called for calm and deployed peacekeeping forces to the area.</p>
        <p>This marks the worst outbreak of violence in the region since the peace agreement signed last year.</p>
        <p>Humanitarian organizations are preparing emergency response teams to provide aid to affected communities.</p>
    </article>
    <footer>
        <p>© 2023 Sample News. All rights reserved.</p>
    </footer>
</body>
</html>"""


@pytest.fixture
async def mock_response():
    """Create a mock HTTP response."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.content_type = "application/xml"
    return mock_resp


@pytest.fixture
async def mock_news_client():
    """Create a mock news client for testing."""
    with patch('aiohttp.ClientSession'):
        client = EnhancedGoogleNewsClient(
            config_path=None,
            config=create_sample_configuration()
        )
        # Mock initialization to avoid actual HTTP requests
        client._initialized = True
        yield client
        await client.close()


@pytest.mark.skipif(not GOOGLE_NEWS_CRAWLER_AVAILABLE, reason="Google News Crawler not available")
class TestNewsSource:
    """Test NewsSource class functionality."""

    def test_news_source_initialization(self):
        """Test NewsSource initialization."""
        source = NewsSource(
            name="Test News",
            base_url="https://testnews.com",
            source_type=SourceType.MAINSTREAM_MEDIA,
            content_format=ContentFormat.ARTICLE,
            priority=CrawlPriority.MEDIUM,
            geographical_focus=GeographicalFocus.GLOBAL,
            languages=["en"],
            rss_feeds=["https://testnews.com/feed.xml"],
            reliability_score=0.8,
            bias_level=0.3
        )

        assert source.name == "Test News"
        assert source.base_url == "https://testnews.com"
        assert source.source_type == SourceType.MAINSTREAM_MEDIA
        assert source.content_format == ContentFormat.ARTICLE
        assert source.priority == CrawlPriority.MEDIUM
        assert source.geographical_focus == GeographicalFocus.GLOBAL
        assert "en" in source.languages
        assert "https://testnews.com/feed.xml" in source.rss_feeds
        assert source.reliability_score == 0.8
        assert source.bias_level == 0.3

    def test_enhanced_news_source(self):
        """Test EnhancedNewsSource functionality."""
        source = EnhancedNewsSource(
            name="Enhanced Test News",
            base_url="https://enhancedtest.com",
            source_type=SourceType.MAINSTREAM_MEDIA,
            content_format=ContentFormat.ARTICLE,
            priority=CrawlPriority.HIGH,
            geographical_focus=GeographicalFocus.REGIONAL,
            languages=["en", "fr"],
            rss_feeds=["https://enhancedtest.com/feed.xml"],
            reliability_score=0.85,
            bias_level=0.25
        )

        # Test health score calculation
        health_score = source.get_health_score()
        assert 0 <= health_score <= 1

        # Update performance metrics
        source.update_performance_metrics(
            successful_requests=45,
            failed_requests=5,
            articles_found=100,
            response_time_ms=250
        )

        # Check metrics were updated
        assert source.metrics.success_rate == 0.9  # 45 / (45 + 5)
        assert source.articles_found == 100
        assert source.avg_response_time_ms == 250


@pytest.mark.skipif(not GOOGLE_NEWS_CRAWLER_AVAILABLE, reason="Google News Crawler not available")
class TestSiteFilteringEngine:
    """Test SiteFilteringEngine functionality."""

    def test_filtering_engine_initialization(self, sample_news_config):
        """Test SiteFilteringEngine initialization."""
        config = NewsSourceConfig(**sample_news_config)
        engine = SiteFilteringEngine(config=config)

        assert engine.min_reliability_score == 0.7
        assert engine.max_bias_level == 0.5
        assert "en" in engine.preferred_languages
        assert len(engine.blocked_domains) == 0

    @pytest.mark.asyncio
    async def test_filter_sources(self, sample_news_config):
        """Test filtering of news sources."""
        config = NewsSourceConfig(**sample_news_config)
        engine = SiteFilteringEngine(config=config)

        # Create sources for testing
        sources = [
            # Good source - should pass filtering
            NewsSource(
                name="Good Source",
                base_url="https://goodsource.com",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.ARTICLE,
                priority=CrawlPriority.HIGH,
                geographical_focus=GeographicalFocus.GLOBAL,
                languages=["en"],
                rss_feeds=["https://goodsource.com/feed.xml"],
                reliability_score=0.9,
                bias_level=0.2
            ),
            # Low reliability - should be filtered
            NewsSource(
                name="Unreliable Source",
                base_url="https://unreliable.com",
                source_type=SourceType.BLOG,
                content_format=ContentFormat.ARTICLE,
                priority=CrawlPriority.LOW,
                geographical_focus=GeographicalFocus.LOCAL,
                languages=["en"],
                rss_feeds=["https://unreliable.com/feed.xml"],
                reliability_score=0.5,  # Below threshold
                bias_level=0.3
            ),
            # High bias - should be filtered
            NewsSource(
                name="Biased Source",
                base_url="https://biased.com",
                source_type=SourceType.OPINION,
                content_format=ContentFormat.ARTICLE,
                priority=CrawlPriority.MEDIUM,
                geographical_focus=GeographicalFocus.GLOBAL,
                languages=["en"],
                rss_feeds=["https://biased.com/feed.xml"],
                reliability_score=0.8,
                bias_level=0.7  # Above threshold
            ),
            # Wrong language - should be filtered
            NewsSource(
                name="Foreign Source",
                base_url="https://foreign.com",
                source_type=SourceType.MAINSTREAM_MEDIA,
                content_format=ContentFormat.ARTICLE,
                priority=CrawlPriority.MEDIUM,
                geographical_focus=GeographicalFocus.GLOBAL,
                languages=["es"],  # Not in preferred languages
                rss_feeds=["https://foreign.com/feed.xml"],
                reliability_score=0.85,
                bias_level=0.3
            )
        ]

        filtered_sources = engine.filter_sources(sources)

        # Only the good source should pass filtering
        assert len(filtered_sources) == 1
        assert filtered_sources[0].name == "Good Source"
        assert filtered_sources[0].reliability_score == 0.9


@pytest.mark.skipif(not GOOGLE_NEWS_CRAWLER_AVAILABLE, reason="Google News Crawler not available")
class TestEnhancedGoogleNewsClient:
    """Test EnhancedGoogleNewsClient functionality."""

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_search_news(self, mock_get, mock_response, mock_news_client, sample_rss_content):
        """Test searching for news articles."""
        # Mock the RSS feed response
        mock_response.text = AsyncMock(return_value=sample_rss_content)
        mock_get.return_value.__aenter__.return_value = mock_response

        # Test search function
        results = await mock_news_client.search_news(
            query="Ethiopia conflict",
            max_results=5,
            language="en"
        )

        # Should find articles from the sample RSS
        assert len(results) > 0

        # Check the first result has the expected data
        first_article = results[0]
        assert "Ethiopia" in first_article["title"]
        assert "conflict" in first_article["title"].lower()
        assert "description" in first_article
        assert "url" in first_article
        assert "publisher" in first_article
        assert "published_date" in first_article

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_with_stealth(self, mock_get, mock_response, mock_news_client, sample_article_html):
        """Test fetching articles with stealth mode."""
        # Mock the article HTML response
        mock_response.text = AsyncMock(return_value=sample_article_html)
        mock_response.content_type = "text/html"
        mock_get.return_value.__aenter__.return_value = mock_response

        # Test fetching content
        url = "https://www.samplenews.com/world/ethiopia-conflict-12345"
        content = await mock_news_client._fetch_with_stealth(url)

        # Should get HTML content
        assert content is not None
        assert "<html" in content
        assert "Major Conflict Erupts in Ethiopia" in content

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_scrape_full_articles(self, mock_get, mock_response, mock_news_client, sample_article_html):
        """Test scraping full article content."""
        # Mock the article HTML response
        mock_response.text = AsyncMock(return_value=sample_article_html)
        mock_response.content_type = "text/html"
        mock_get.return_value.__aenter__.return_value = mock_response

        # Create test articles
        articles = [
            {
                "title": "Major Conflict Erupts in Ethiopia",
                "url": "https://www.samplenews.com/world/ethiopia-conflict-12345",
                "description": "Breaking news: Major conflict has erupted in the northern region of Ethiopia with reports of casualties.",
                "publisher": "Sample News",
                "published_date": "2023-04-24T09:30:00Z"
            }
        ]

        # Test scraping
        enhanced_articles = await mock_news_client.scrape_full_articles(articles)

        # Check enhanced content
        assert len(enhanced_articles) == 1
        assert "full_content" in enhanced_articles[0]
        assert "conflict" in enhanced_articles[0]["full_content"].lower()
        assert "casualties" in enhanced_articles[0]["full_content"].lower()
        assert "metadata" in enhanced_articles[0]

    def test_calculate_relevance_score(self, mock_news_client):
        """Test calculation of article relevance score."""
        # Test article data
        article = {
            "title": "Major Conflict Erupts in Ethiopia",
            "description": "Breaking news: Major conflict has erupted in the northern region of Ethiopia with reports of casualties.",
            "published_date": "2023-04-24T09:30:00Z",
            "publisher": "Sample News",
            "url": "https://www.samplenews.com/world/ethiopia-conflict-12345"
        }

        # Calculate relevance for a relevant query
        score1 = mock_news_client._calculate_relevance_score(article, "Ethiopia conflict")

        # Calculate relevance for a less relevant query
        score2 = mock_news_client._calculate_relevance_score(article, "South Sudan peace")

        # The first score should be higher
        assert score1 > score2
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1

    def test_deduplicate_and_rank(self, mock_news_client):
        """Test deduplication and ranking of articles."""
        # Create test articles with some duplicates
        articles = [
            {
                "title": "Major Conflict in Ethiopia",
                "url": "https://www.source1.com/ethiopia-conflict",
                "description": "Conflict has erupted in Ethiopia",
                "publisher": "Source 1",
                "published_date": "2023-04-24T09:30:00Z",
                "relevance_score": 0.85
            },
            {
                "title": "Ethiopia: Conflict Erupts",  # Similar title
                "url": "https://www.source2.com/ethiopia-conflict",
                "description": "Conflict has erupted in northern Ethiopia",
                "publisher": "Source 2",
                "published_date": "2023-04-24T10:15:00Z",
                "relevance_score": 0.82
            },
            {
                "title": "Peace Talks in South Sudan",  # Different article
                "url": "https://www.source1.com/south-sudan-peace",
                "description": "Peace negotiations begin in South Sudan",
                "publisher": "Source 1",
                "published_date": "2023-04-24T08:45:00Z",
                "relevance_score": 0.75
            }
        ]

        # Deduplicate and rank
        unique_articles = mock_news_client._deduplicate_and_rank(articles)

        # Should have 2 unique articles
        assert len(unique_articles) == 2

        # First article should be the highest relevance
        assert unique_articles[0]["relevance_score"] == 0.85
        assert unique_articles[0]["title"] == "Major Conflict in Ethiopia"

        # Second article should be the peace talks (the similar one was deduplicated)
        assert unique_articles[1]["title"] == "Peace Talks in South Sudan"


@pytest.mark.skipif(not GOOGLE_NEWS_CRAWLER_AVAILABLE, reason="Google News Crawler not available")
class TestGNewsCompatibility:
    """Test GNewsCompatibilityWrapper functionality."""

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_compatibility_wrapper(self, mock_get, mock_response, sample_rss_content):
        """Test the GNews compatibility wrapper."""
        # Mock the RSS feed response
        mock_response.text = AsyncMock(return_value=sample_rss_content)
        mock_get.return_value.__aenter__.return_value = mock_response

        # Create the compatibility wrapper
        with patch('crawlers.google_news_crawler.api.google_news_client.EnhancedGoogleNewsClient'):
            wrapper = GNewsCompatibilityWrapper()

            # Mock client initialization
            wrapper.client = AsyncMock()
            wrapper.client.search_news.return_value = [
                {
                    "title": "Major Conflict Erupts in Ethiopia",
                    "url": "https://www.samplenews.com/world/ethiopia-conflict-12345",
                    "description": "Breaking news: Major conflict has erupted in the northern region of Ethiopia with reports of casualties.",
                    "publisher": "Sample News",
                    "published_date": "2023-04-24T09:30:00Z",
                    "relevance_score": 0.9
                }
            ]
            wrapper._client_initialized = True

            # Test compatibility method
            news = await wrapper.get_news("Ethiopia conflict")

            # Check the structure matches the expected GNews format
            assert len(news) > 0
            assert "title" in news[0]
            assert "url" in news[0]
            assert "description" in news[0]
            assert "published date" in news[0]
            assert "publisher" in news[0]


@pytest.mark.skipif(not GOOGLE_NEWS_CRAWLER_AVAILABLE, reason="Google News Crawler not available")
class TestConfigFunctions:
    """Test configuration-related functions."""

    def test_news_source_config(self, sample_news_config):
        """Test NewsSourceConfig initialization and validation."""
        config = NewsSourceConfig(**sample_news_config)

        assert len(config.sources) == 2
        assert config.sources[0].name == "BBC News"
        assert config.sources[1].name == "CNN"
        assert config.filtering.min_reliability_score == 0.7
        assert config.crawl_settings.max_articles_per_source == 10

    def test_create_sample_configuration(self):
        """Test creation of sample configuration."""
        config = create_sample_configuration()

        assert isinstance(config, dict)
        assert "sources" in config
        assert "filtering" in config
        assert "crawl_settings" in config
        assert len(config["sources"]) > 0


@pytest.mark.skipif(not GOOGLE_NEWS_CRAWLER_AVAILABLE, reason="Google News Crawler not available")
class TestFactoryFunctions:
    """Test client factory functions."""

    @pytest.mark.asyncio
    @patch('crawlers.google_news_crawler.api.google_news_client.EnhancedGoogleNewsClient')
    async def test_create_enhanced_client(self, mock_client_class):
        """Test enhanced client factory function."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        client = await create_enhanced_gnews_client()

        assert client is not None
        mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch('crawlers.google_news_crawler.api.google_news_client.GNewsCompatibilityWrapper')
    async def test_create_basic_client(self, mock_wrapper_class):
        """Test basic client factory function."""
        mock_wrapper = AsyncMock()
        mock_wrapper_class.return_value = mock_wrapper

        client = await create_basic_gnews_client()

        assert client is not None


@pytest.mark.skipif(not GOOGLE_NEWS_CRAWLER_AVAILABLE, reason="Google News Crawler not available")
class TestErrorHandling:
    """Test error handling in the Google News crawler."""

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_network_error_handling(self, mock_get, mock_news_client):
        """Test handling of network errors."""
        # Simulate network error
        mock_get.return_value.__aenter__.side_effect = aiohttp.ClientError("Connection error")

        # Should handle the error gracefully
        results = await mock_news_client.search_news("test query", max_results=5)

        # Should return empty results rather than crash
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_invalid_response_handling(self, mock_get, mock_response, mock_news_client):
        """Test handling of invalid responses."""
        # Simulate invalid content
        mock_response.text = AsyncMock(return_value="Not valid XML or HTML")
        mock_get.return_value.__aenter__.return_value = mock_response

        # Should handle invalid content gracefully
        results = await mock_news_client.search_news("test query", max_results=5)

        # Should return empty results rather than crash
        assert isinstance(results, list)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
