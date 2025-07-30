#!/usr/bin/env python3
"""
Test Suite for NewsAPI Crawler
=============================

This module provides comprehensive tests for the NewsAPI crawler package.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json

# Add the packages_enhanced directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

try:
    from packages_enhanced.crawlers.newsapi_crawler import (
        NewsAPIClient,
        NewsAPIAdvancedClient,
        NewsArticle,
        NewsSource,
        ArticleCollection,
        SearchParameters,
        date_to_string,
        string_to_date,
        filter_articles_by_keywords,
        calculate_relevance_score
    )
    from packages_enhanced.crawlers.newsapi_crawler.api.factory import (
        create_client,
        create_advanced_client,
        create_batch_client
    )
    from packages_enhanced.crawlers.newsapi_crawler.config import (
        NewsAPIConfig,
        get_default_config
    )
    from packages_enhanced.crawlers.newsapi_crawler.parsers import (
        ArticleParser,
        ContentExtractor,
        EventDetector
    )
    NEWSAPI_AVAILABLE = True
except ImportError as e:
    print(f"Error importing NewsAPI crawler: {e}")
    NEWSAPI_AVAILABLE = False


# Mock responses for testing
MOCK_EVERYTHING_RESPONSE = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "source": {
                "id": "bbc-news",
                "name": "BBC News"
            },
            "author": "BBC News",
            "title": "Ethiopia peace agreement signed",
            "description": "Ethiopia government and Tigrayan forces sign peace agreement in South Africa.",
            "url": "https://www.bbc.co.uk/news/world-africa-12345678",
            "urlToImage": "https://ichef.bbci.co.uk/news/1024/branded_news/example.jpg",
            "publishedAt": "2022-11-02T12:00:00Z",
            "content": "The Ethiopian government and Tigrayan forces have signed a peace agreement..."
        },
        {
            "source": {
                "id": "al-jazeera",
                "name": "Al Jazeera"
            },
            "author": "Al Jazeera",
            "title": "Somalia security situation worsens",
            "description": "Rising insecurity in central Somalia as drought continues.",
            "url": "https://www.aljazeera.com/news/2022/11/01/somalia-security-12345",
            "urlToImage": "https://www.aljazeera.com/wp-content/uploads/2022/11/example.jpg",
            "publishedAt": "2022-11-01T09:30:00Z",
            "content": "The security situation in parts of central Somalia has deteriorated..."
        }
    ]
}

MOCK_TOP_HEADLINES_RESPONSE = {
    "status": "ok",
    "totalResults": 1,
    "articles": [
        {
            "source": {
                "id": "reuters",
                "name": "Reuters"
            },
            "author": "Reuters",
            "title": "Breaking: Sudan conflict escalates",
            "description": "Conflict in Sudan has escalated with new reports of violence.",
            "url": "https://www.reuters.com/world/africa/sudan-conflict-12345",
            "urlToImage": "https://www.reuters.com/resizer/example.jpg",
            "publishedAt": "2022-11-03T10:15:00Z",
            "content": "The conflict in Sudan has escalated with new reports of violence in..."
        }
    ]
}

MOCK_SOURCES_RESPONSE = {
    "status": "ok",
    "sources": [
        {
            "id": "bbc-news",
            "name": "BBC News",
            "description": "BBC News is an operational business division of the British Broadcasting Corporation.",
            "url": "https://www.bbc.co.uk/news",
            "category": "general",
            "language": "en",
            "country": "gb"
        },
        {
            "id": "al-jazeera",
            "name": "Al Jazeera",
            "description": "News, analysis from the Middle East & worldwide, multimedia & interactives, opinions, documentaries, podcasts, long reads and broadcast schedule.",
            "url": "https://www.aljazeera.com",
            "category": "general",
            "language": "en",
            "country": "qa"
        }
    ]
}


@unittest.skipIf(not NEWSAPI_AVAILABLE, "NewsAPI crawler not available")
class TestNewsAPIClient(unittest.TestCase):
    """Test the basic NewsAPI client functionality."""

    @patch('newsapi.NewsApiClient')
    def test_client_initialization(self, mock_newsapi):
        """Test client initialization."""
        # Set up mock
        mock_instance = mock_newsapi.return_value

        # Create client
        client = NewsAPIClient(api_key="test_key")

        # Verify
        self.assertIsNotNone(client)
        mock_newsapi.assert_called_once_with(api_key="test_key")

    @patch('newsapi.NewsApiClient')
    def test_get_everything(self, mock_newsapi):
        """Test get_everything method."""
        # Set up mock
        mock_instance = mock_newsapi.return_value
        mock_instance.get_everything.return_value = MOCK_EVERYTHING_RESPONSE

        # Create client and call method
        client = NewsAPIClient(api_key="test_key")
        response = client.get_everything(q="test", language="en")

        # Verify
        self.assertEqual(response, MOCK_EVERYTHING_RESPONSE)
        mock_instance.get_everything.assert_called_once_with(q="test", language="en")

    @patch('newsapi.NewsApiClient')
    def test_get_top_headlines(self, mock_newsapi):
        """Test get_top_headlines method."""
        # Set up mock
        mock_instance = mock_newsapi.return_value
        mock_instance.get_top_headlines.return_value = MOCK_TOP_HEADLINES_RESPONSE

        # Create client and call method
        client = NewsAPIClient(api_key="test_key")
        response = client.get_top_headlines(country="us")

        # Verify
        self.assertEqual(response, MOCK_TOP_HEADLINES_RESPONSE)
        mock_instance.get_top_headlines.assert_called_once_with(country="us")

    @patch('newsapi.NewsApiClient')
    def test_get_sources(self, mock_newsapi):
        """Test get_sources method."""
        # Set up mock
        mock_instance = mock_newsapi.return_value
        mock_instance.get_sources.return_value = MOCK_SOURCES_RESPONSE

        # Create client and call method
        client = NewsAPIClient(api_key="test_key")
        response = client.get_sources()

        # Verify
        self.assertEqual(response, MOCK_SOURCES_RESPONSE)
        mock_instance.get_sources.assert_called_once()


@unittest.skipIf(not NEWSAPI_AVAILABLE, "NewsAPI crawler not available")
class TestNewsAPIAdvancedClient(unittest.TestCase):
    """Test the advanced NewsAPI client functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for cache
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_cache')
        os.makedirs(self.temp_dir, exist_ok=True)

        # Mock environment variable
        self.original_api_key = os.environ.get('NEWSAPI_KEY')
        os.environ['NEWSAPI_KEY'] = 'test_api_key'

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

        # Restore environment variable
        if self.original_api_key:
            os.environ['NEWSAPI_KEY'] = self.original_api_key
        else:
            os.environ.pop('NEWSAPI_KEY', None)

    @patch('aiohttp.ClientSession.get')
    def test_client_initialization(self, mock_get):
        """Test advanced client initialization."""
        async def _test():
            client = await create_advanced_client(
                api_key="test_key",
                cache_dir=self.temp_dir
            )
            self.assertIsNotNone(client)
            self.assertEqual(client.api_key, "test_key")
            self.assertEqual(client.cache_dir, self.temp_dir)
            await client.close()

        asyncio.run(_test())

    @patch('aiohttp.ClientSession.get')
    async def _test_get_everything_async(self, mock_get):
        """Async test for get_everything method."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = MOCK_EVERYTHING_RESPONSE
        mock_get.return_value.__aenter__.return_value = mock_response

        # Create client and call method
        client = await create_advanced_client(
            api_key="test_key",
            cache_dir=self.temp_dir
        )

        try:
            response = await client.get_everything(q="test", language="en")

            # Verify
            self.assertEqual(response["status"], "ok")
            self.assertEqual(len(response["articles"]), 2)
            mock_get.assert_called_once()
        finally:
            await client.close()

    def test_get_everything(self):
        """Test get_everything method (wrapper for async test)."""
        asyncio.run(self._test_get_everything_async())

    @patch('aiohttp.ClientSession.get')
    async def _test_search_articles_async(self, mock_get):
        """Async test for search_articles method."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = MOCK_EVERYTHING_RESPONSE
        mock_get.return_value.__aenter__.return_value = mock_response

        # Create client and call method
        client = await create_advanced_client(
            api_key="test_key",
            cache_dir=self.temp_dir
        )

        try:
            articles = await client.search_articles(
                query="test",
                language="en",
                max_results=10
            )

            # Verify
            self.assertEqual(len(articles), 2)
            self.assertEqual(articles[0]["title"], "Ethiopia peace agreement signed")
            mock_get.assert_called_once()
        finally:
            await client.close()

    def test_search_articles(self):
        """Test search_articles method (wrapper for async test)."""
        asyncio.run(self._test_search_articles_async())

    @patch('aiohttp.ClientSession.get')
    async def _test_cache_functionality_async(self, mock_get):
        """Async test for caching functionality."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = MOCK_EVERYTHING_RESPONSE
        mock_get.return_value.__aenter__.return_value = mock_response

        # Create client with cache
        client = await create_advanced_client(
            api_key="test_key",
            cache_dir=self.temp_dir,
            cache_ttl=60
        )

        try:
            # First request should hit the API
            await client.get_everything(q="test", language="en")
            self.assertEqual(mock_get.call_count, 1)

            # Second identical request should use cache
            await client.get_everything(q="test", language="en")
            # Call count should still be 1 if cache was used
            self.assertEqual(mock_get.call_count, 1)
        finally:
            await client.close()

    def test_cache_functionality(self):
        """Test caching functionality (wrapper for async test)."""
        asyncio.run(self._test_cache_functionality_async())


@unittest.skipIf(not NEWSAPI_AVAILABLE, "NewsAPI crawler not available")
class TestArticleModels(unittest.TestCase):
    """Test article data model functionality."""

    def test_news_source_creation(self):
        """Test NewsSource creation."""
        source = NewsSource(
            id="test-source",
            name="Test Source",
            description="A test source",
            url="https://test.com",
            category="general",
            language="en",
            country="us"
        )

        self.assertEqual(source.id, "test-source")
        self.assertEqual(source.name, "Test Source")

        # Test conversion to dict
        source_dict = source.to_dict()
        self.assertIsInstance(source_dict, dict)
        self.assertEqual(source_dict["id"], "test-source")

        # Test creation from API response
        api_source = {
            "id": "bbc-news",
            "name": "BBC News",
            "description": "BBC News description",
            "url": "https://www.bbc.co.uk/news",
            "category": "general",
            "language": "en",
            "country": "gb"
        }

        source = NewsSource.from_api_response(api_source)
        self.assertEqual(source.id, "bbc-news")
        self.assertEqual(source.name, "BBC News")

    def test_news_article_creation(self):
        """Test NewsArticle creation."""
        source = NewsSource(id="test-source", name="Test Source")
        article = NewsArticle(
            source=source,
            author="Test Author",
            title="Test Title",
            description="Test description",
            url="https://test.com/article",
            url_to_image="https://test.com/image.jpg",
            published_at="2022-11-03T10:15:00Z",
            content="Test content"
        )

        self.assertEqual(article.title, "Test Title")
        self.assertEqual(article.source.name, "Test Source")

        # Test datetime conversion
        self.assertIsInstance(article.published_at, datetime)

        # Test conversion to dict
        article_dict = article.to_dict()
        self.assertIsInstance(article_dict, dict)
        self.assertEqual(article_dict["title"], "Test Title")

        # Test creation from API response
        api_article = {
            "source": {"id": "bbc-news", "name": "BBC News"},
            "author": "BBC News",
            "title": "Test API Article",
            "description": "Test description",
            "url": "https://www.bbc.co.uk/news/article",
            "urlToImage": "https://www.bbc.co.uk/image.jpg",
            "publishedAt": "2022-11-03T10:15:00Z",
            "content": "Test content"
        }

        article = NewsArticle.from_api_response(api_article)
        self.assertEqual(article.title, "Test API Article")
        self.assertEqual(article.source.name, "BBC News")

    def test_article_collection(self):
        """Test ArticleCollection functionality."""
        # Create articles
        article1 = NewsArticle(
            source="Source 1",
            title="Article 1",
            url="https://test.com/1"
        )

        article2 = NewsArticle(
            source="Source 2",
            title="Article 2",
            url="https://test.com/2"
        )

        # Create collection
        collection = ArticleCollection(
            articles=[article1, article2],
            total_results=2,
            status="ok"
        )

        self.assertEqual(collection.count, 2)

        # Test filtering
        filtered = collection.filter(sources=["Source 1"])
        self.assertEqual(filtered.count, 1)
        self.assertEqual(filtered.articles[0].title, "Article 1")

        # Test sorting
        article1.relevance_score = 0.5
        article2.relevance_score = 0.8

        sorted_collection = collection.sort(key="relevance_score")
        self.assertEqual(sorted_collection.articles[0].title, "Article 2")

        # Test pagination
        paginated = collection.paginate(page=1, page_size=1)
        self.assertEqual(paginated.count, 1)


@unittest.skipIf(not NEWSAPI_AVAILABLE, "NewsAPI crawler not available")
class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_date_conversion(self):
        """Test date conversion functions."""
        # Test date_to_string
        test_date = datetime(2022, 11, 3, 10, 15, 0)
        date_str = date_to_string(test_date)
        self.assertEqual(date_str, "2022-11-03")

        # Test with custom format
        date_str = date_to_string(test_date, format="%Y/%m/%d %H:%M")
        self.assertEqual(date_str, "2022/11/03 10:15")

        # Test string_to_date
        parsed_date = string_to_date("2022-11-03T10:15:00Z")
        self.assertIsInstance(parsed_date, datetime)
        self.assertEqual(parsed_date.year, 2022)
        self.assertEqual(parsed_date.month, 11)
        self.assertEqual(parsed_date.day, 3)

    def test_article_filtering(self):
        """Test article filtering functions."""
        articles = [
            {
                "title": "Ethiopia peace agreement",
                "description": "Peace talks in Ethiopia have led to an agreement."
            },
            {
                "title": "Somalia security concerns",
                "description": "Security situation in Somalia remains tense."
            },
            {
                "title": "Kenya election results",
                "description": "Results of the Kenya election announced."
            }
        ]

        # Test filtering by keywords
        filtered = filter_articles_by_keywords(articles, ["peace", "agreement"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["title"], "Ethiopia peace agreement")

        # Test relevance calculation
        relevance = calculate_relevance_score(articles[0], "peace agreement Ethiopia")
        self.assertGreater(relevance, 0.5)

        relevance2 = calculate_relevance_score(articles[2], "peace agreement Ethiopia")
        self.assertLess(relevance2, relevance)


@unittest.skipIf(not NEWSAPI_AVAILABLE, "NewsAPI crawler not available")
class TestParsers(unittest.TestCase):
    """Test parser functionality."""

    def test_article_parser_initialization(self):
        """Test ArticleParser initialization."""
        parser = ArticleParser(use_nlp=False, use_newspaper=False)
        self.assertIsNotNone(parser)

    def test_event_detector(self):
        """Test EventDetector functionality."""
        detector = EventDetector()

        # Test with conflict text
        events = detector.detect_events(
            "Heavy fighting broke out in northern Ethiopia yesterday, " +
            "with reports of artillery fire and civilian casualties. " +
            "The conflict has displaced thousands of people."
        )

        # Even without NLP, should detect some events
        self.assertGreater(len(events), 0)

        # Test with non-conflict text
        events2 = detector.detect_events(
            "The weather today is sunny with a high of 25 degrees. " +
            "Local farmers are expecting a good harvest this year."
        )

        # Should detect fewer or no events
        self.assertLessEqual(len(events2), len(events))


@unittest.skipIf(not NEWSAPI_AVAILABLE, "NewsAPI crawler not available")
class TestConfiguration(unittest.TestCase):
    """Test configuration functionality."""

    def test_config_creation(self):
        """Test NewsAPIConfig creation."""
        config = NewsAPIConfig(
            api_key="test_key",
            client_type="advanced",
            enable_caching=True,
            cache_dir="/tmp/cache",
            default_language="en"
        )

        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.client_type, "advanced")
        self.assertTrue(config.enable_caching)

    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()

        self.assertIsNotNone(config)
        self.assertEqual(config.client_type, "advanced")
        self.assertEqual(config.default_language, "en")

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = NewsAPIConfig(
            api_key="test_key",
            client_type="advanced",
            enable_caching=True
        )

        # Convert to dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["api_key"], "test_key")

        # Create from dict
        new_config = NewsAPIConfig.from_dict(config_dict)
        self.assertEqual(new_config.api_key, "test_key")
        self.assertEqual(new_config.client_type, "advanced")


if __name__ == '__main__':
    unittest.main()
