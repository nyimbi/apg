"""
Basic Usage Examples for Google News Crawler
============================================

This file demonstrates basic usage patterns for the enhanced Google News crawler.
It covers common scenarios and provides practical examples for getting started.

Examples included:
- Basic setup and initialization
- Simple news search
- RSS feed parsing
- Article content extraction
- Database integration
- Configuration management
- Error handling

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced Google News crawler components
try:
    from lindela.packages_enhanced.crawlers.google_news_crawler import (
        create_enhanced_gnews_client,
        create_basic_gnews_client,
        create_sample_configuration,
        EnhancedGoogleNewsClient,
        GNewsCompatibilityWrapper,
        CrawlerConfig,
        get_config,
        load_config
    )
    CRAWLER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Crawler components not available: {e}")
    CRAWLER_AVAILABLE = False

# Import database manager
try:
    from lindela.packages.pgmgr import HybridIntegratedPostgreSQLManager
    DB_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database manager not available: {e}")
    DB_MANAGER_AVAILABLE = False

# Import stealth orchestrator
try:
    from lindela.packages_enhanced.crawlers.news_crawler.stealth.unified_stealth_orchestrator import UnifiedStealthOrchestrator
    STEALTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Stealth orchestrator not available: {e}")
    STEALTH_AVAILABLE = False

DEPENDENCIES_AVAILABLE = CRAWLER_AVAILABLE and DB_MANAGER_AVAILABLE


async def example_1_basic_setup():
    """
    Example 1: Basic setup and initialization
    ========================================

    This example shows how to set up the Google News crawler with minimal configuration.
    """
    print("\n" + "="*50)
    print("Example 1: Basic Setup and Initialization")
    print("="*50)

    if not CRAWLER_AVAILABLE:
        print("❌ Crawler components not available - skipping example")
        return

    try:
        # Create a mock database manager for this example
        if DB_MANAGER_AVAILABLE:
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'lindela',
                'username': 'postgres',
                'password': 'your_password'
            }
            # Initialize database manager
            db_manager = HybridIntegratedPostgreSQLManager(db_config)
        else:
            # Create mock database manager
            class MockDBManager:
                async def initialize(self):
                    pass
                async def close(self):
                    pass
                async def article_exists(self, url):
                    return False
                async def insert_article(self, article_data):
                    return f"mock_id_{hash(article_data.get('url', ''))}"

            db_manager = MockDBManager()

        # Create basic client (without stealth capabilities)
        client = await create_basic_gnews_client(db_manager)

        print("✅ Google News client created successfully!")
        print(f"📊 Client type: {type(client).__name__}")

        # Get some basic info
        if hasattr(client, 'get_session_statistics'):
            stats = client.get_session_statistics()
            print(f"📈 Session statistics: {stats}")

        # Cleanup
        await client.close()
        print("🧹 Client closed successfully")

    except Exception as e:
        print(f"❌ Error in basic setup: {e}")
        logger.error(f"Basic setup failed: {e}")


async def example_2_simple_news_search():
    """
    Example 2: Simple news search
    =============================

    This example demonstrates how to search for news articles using the crawler.
    """
    print("\n" + "="*50)
    print("Example 2: Simple News Search")
    print("="*50)

    if not DEPENDENCIES_AVAILABLE:
        print("❌ Dependencies not available - skipping example")
        return

    try:
        # Create sample configuration
        config = create_sample_configuration()

        # Mock database manager
        if DB_MANAGER_AVAILABLE:
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'lindela',
                'username': 'postgres',
                'password': 'your_password'
            }
            db_manager = HybridIntegratedPostgreSQLManager(db_config)
        else:
            class MockDBManager:
                async def initialize(self):
                    pass
                async def close(self):
                    pass
                async def article_exists(self, url):
                    return False
                async def insert_article(self, article_data):
                    return f"mock_id_{hash(article_data.get('url', ''))}"

            db_manager = MockDBManager()

        # Create client
        client = await create_enhanced_gnews_client(db_manager, config_dict=config)

        # Search for technology news
        search_query = "artificial intelligence"
        print(f"🔍 Searching for: '{search_query}'")

        try:
            # Perform search
            results = await client.search_news(
                query=search_query,
                max_results=5,
                language='en',
                country='US'
            )

            print(f"📰 Found {len(results)} articles:")

            for i, article in enumerate(results, 1):
                print(f"\n{i}. {article.get('title', 'No title')}")
                print(f"   URL: {article.get('url', 'No URL')}")
                print(f"   Publisher: {article.get('publisher', 'Unknown')}")
                print(f"   Published: {article.get('published_date', 'Unknown')}")

                # Show snippet of content
                content = article.get('content', '')
                if content:
                    snippet = content[:100] + "..." if len(content) > 100 else content
                    print(f"   Content: {snippet}")

        except Exception as search_error:
            print(f"⚠️  Search failed: {search_error}")
            # This is expected in the example since we don't have real connections

        # Cleanup
        await client.close()
        print("\n✅ Search example completed")

    except Exception as e:
        print(f"❌ Error in news search: {e}")
        logger.error(f"News search failed: {e}")


async def example_3_rss_feed_parsing():
    """
    Example 3: RSS feed parsing
    ===========================

    This example shows how to parse RSS feeds directly using the crawler's parser.
    """
    print("\n" + "="*50)
    print("Example 3: RSS Feed Parsing")
    print("="*50)

    try:
        from lindela.packages_enhanced.crawlers.google_news_crawler.parsers import RSSParser

        # Create RSS parser
        parser = RSSParser({
            'max_entries': 10,
            'extract_content': True,
            'clean_html': True
        })

        # Sample RSS feed content
        sample_rss = '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Tech News Feed</title>
        <description>Latest technology news</description>
        <link>https://example.com</link>

        <item>
            <title>AI Breakthrough in Natural Language Processing</title>
            <link>https://example.com/ai-breakthrough</link>
            <description>Researchers announce major breakthrough in NLP technology</description>
            <pubDate>Wed, 01 Jan 2024 10:00:00 GMT</pubDate>
            <author>tech@example.com (Tech Reporter)</author>
            <category>Technology</category>
        </item>

        <item>
            <title>New Quantum Computing Milestone Reached</title>
            <link>https://example.com/quantum-milestone</link>
            <description>Scientists achieve new quantum computing milestone</description>
            <pubDate>Wed, 01 Jan 2024 11:00:00 GMT</pubDate>
            <author>science@example.com (Science Reporter)</author>
            <category>Science</category>
        </item>
    </channel>
</rss>'''

        print("📝 Parsing RSS feed...")

        # Parse the RSS feed
        result = await parser.parse(sample_rss, source_url="https://example.com/feed.xml")

        if result.status.value == "success":
            articles = result.content.get('articles', [])
            print(f"✅ Successfully parsed {len(articles)} articles:")

            for i, article in enumerate(articles, 1):
                print(f"\n{i}. {article['title']}")
                print(f"   URL: {article['url']}")
                print(f"   Description: {article.get('description', 'No description')}")
                print(f"   Author: {article.get('author', 'Unknown')}")
                print(f"   Published: {article.get('published_date', 'Unknown')}")

                if article.get('tags'):
                    print(f"   Tags: {', '.join(article['tags'])}")
        else:
            print(f"❌ Parsing failed: {result.error}")

    except ImportError:
        print("❌ RSS parser not available")
    except Exception as e:
        print(f"❌ Error in RSS parsing: {e}")
        logger.error(f"RSS parsing failed: {e}")


async def example_4_html_article_extraction():
    """
    Example 4: HTML article extraction
    ==================================

    This example demonstrates extracting article content from HTML pages.
    """
    print("\n" + "="*50)
    print("Example 4: HTML Article Extraction")
    print("="*50)

    try:
        from lindela.packages_enhanced.crawlers.google_news_crawler.parsers import HTMLParser

        # Create HTML parser
        parser = HTMLParser({
            'extract_images': True,
            'clean_content': True,
            'min_content_length': 50
        })

        # Sample HTML article
        sample_html = '''<!DOCTYPE html>
<html>
<head>
    <title>Revolutionary AI System Transforms Healthcare</title>
    <meta name="description" content="New AI system shows promising results in medical diagnosis">
    <meta name="author" content="Dr. Sarah Johnson">
    <meta property="article:published_time" content="2024-01-01T10:00:00Z">
</head>
<body>
    <article>
        <h1>Revolutionary AI System Transforms Healthcare</h1>
        <div class="byline">By Dr. Sarah Johnson | January 1, 2024</div>

        <div class="article-content">
            <p>A groundbreaking artificial intelligence system has been developed that shows
            unprecedented accuracy in medical diagnosis, potentially revolutionizing healthcare delivery.</p>

            <p>The system, developed by researchers at leading medical institutions, combines
            advanced machine learning algorithms with extensive medical databases to provide
            rapid and accurate diagnostic assistance.</p>

            <p>Initial trials have shown remarkable results, with the AI system achieving
            diagnostic accuracy rates exceeding 95% across multiple medical specialties.</p>

            <img src="https://example.com/ai-healthcare.jpg" alt="AI Healthcare System">

            <p>This breakthrough represents a significant step forward in the integration of
            artificial intelligence into medical practice, promising improved patient outcomes
            and more efficient healthcare delivery.</p>
        </div>
    </article>
</body>
</html>'''

        print("🔍 Extracting content from HTML article...")

        # Parse the HTML article
        result = await parser.parse(sample_html, source_url="https://example.com/ai-healthcare")

        if result.status.value == "success":
            articles = result.content.get('articles', [])
            if articles:
                article = articles[0]
                print("✅ Successfully extracted article:")
                print(f"   Title: {article['title']}")
                print(f"   Author: {article.get('author', 'Unknown')}")
                print(f"   Published: {article.get('published_date', 'Unknown')}")
                print(f"   Word count: {article.get('word_count', 0)}")

                # Show content snippet
                content = article.get('content', '')
                if content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    print(f"   Content preview: {snippet}")

                # Show images
                images = article.get('images', [])
                if images:
                    print(f"   Images found: {len(images)}")
                    for img in images[:3]:  # Show first 3 images
                        print(f"     - {img}")
            else:
                print("❌ No articles extracted")
        else:
            print(f"❌ Extraction failed: {result.error}")

    except ImportError:
        print("❌ HTML parser not available")
    except Exception as e:
        print(f"❌ Error in HTML extraction: {e}")
        logger.error(f"HTML extraction failed: {e}")


async def example_5_configuration_management():
    """
    Example 5: Configuration management
    ==================================

    This example shows how to manage configuration for the crawler.
    """
    print("\n" + "="*50)
    print("Example 5: Configuration Management")
    print("="*50)

    try:
        from lindela.packages_enhanced.crawlers.google_news_crawler.config import (
            ConfigurationManager, CrawlerConfig, create_config_template
        )

        # Create configuration manager
        config_manager = ConfigurationManager()

        print("📋 Creating sample configuration...")

        # Create a sample configuration
        sample_config = CrawlerConfig()

        # Customize some settings
        sample_config.google_news.max_results_per_query = 50
        sample_config.filtering.min_content_length = 200
        sample_config.parsing.extract_images = True
        sample_config.performance.max_concurrent_requests = 10

        print("✅ Configuration created with settings:")
        print(f"   Max results per query: {sample_config.google_news.max_results_per_query}")
        print(f"   Min content length: {sample_config.filtering.min_content_length}")
        print(f"   Extract images: {sample_config.parsing.extract_images}")
        print(f"   Max concurrent requests: {sample_config.performance.max_concurrent_requests}")

        # Show environment-specific configurations
        print("\n🌍 Environment configurations:")

        from lindela.packages_enhanced.crawlers.google_news_crawler.config import (
            DEVELOPMENT_CONFIG, PRODUCTION_CONFIG, TESTING_CONFIG
        )

        print("   Development config highlights:")
        print(f"     - Debug mode: {DEVELOPMENT_CONFIG.get('debug', False)}")
        print(f"     - Log level: {DEVELOPMENT_CONFIG.get('logging', {}).get('level', 'INFO')}")
        print(f"     - Max concurrent: {DEVELOPMENT_CONFIG.get('performance', {}).get('max_concurrent_requests', 5)}")

        print("   Production config highlights:")
        print(f"     - Debug mode: {PRODUCTION_CONFIG.get('debug', False)}")
        print(f"     - Log level: {PRODUCTION_CONFIG.get('logging', {}).get('level', 'INFO')}")
        print(f"     - Monitoring enabled: {PRODUCTION_CONFIG.get('monitoring', {}).get('enabled', False)}")

        # Validate configuration
        print("\n🔍 Validating configuration...")
        errors = sample_config.validate()
        if errors:
            print(f"❌ Configuration has {len(errors)} errors:")
            for error in errors:
                print(f"     - {error}")
        else:
            print("✅ Configuration is valid!")

    except ImportError:
        print("❌ Configuration management not available")
    except Exception as e:
        print(f"❌ Error in configuration management: {e}")
        logger.error(f"Configuration management failed: {e}")


async def example_6_error_handling():
    """
    Example 6: Error handling and recovery
    =====================================

    This example demonstrates proper error handling patterns.
    """
    print("\n" + "="*50)
    print("Example 6: Error Handling and Recovery")
    print("="*50)

    try:
        from lindela.packages_enhanced.crawlers.google_news_crawler.parsers import (
            RSSParser, HTMLParser, JSONParser, ParseStatus
        )

        # Test error handling with different parsers
        parsers = [
            ("RSS Parser", RSSParser()),
            ("HTML Parser", HTMLParser()),
            ("JSON Parser", JSONParser())
        ]

        # Test cases with various invalid inputs
        test_cases = [
            ("Invalid XML", "<?xml version='1.0'?><rss><channel><item><title>Unclosed"),
            ("Invalid JSON", '{"title": "test", "content": }'),
            ("Empty content", ""),
            ("Malformed HTML", "<html><body><p>Unclosed paragraph<div>Bad nesting</p></div>"),
        ]

        print("🧪 Testing error handling with various invalid inputs...")

        for test_name, test_content in test_cases:
            print(f"\n📝 Test case: {test_name}")

            for parser_name, parser in parsers:
                try:
                    result = await parser.parse(test_content)

                    status_icon = {
                        ParseStatus.SUCCESS: "✅",
                        ParseStatus.PARTIAL: "⚠️",
                        ParseStatus.FAILED: "❌",
                        ParseStatus.SKIPPED: "⏭️"
                    }.get(result.status, "❓")

                    print(f"   {status_icon} {parser_name}: {result.status.value}")

                    if result.error:
                        # Show first 50 chars of error message
                        error_snippet = result.error[:50] + "..." if len(result.error) > 50 else result.error
                        print(f"      Error: {error_snippet}")

                    if result.status == ParseStatus.SUCCESS and result.content:
                        articles = result.content.get('articles', [])
                        print(f"      Articles extracted: {len(articles)}")

                except Exception as e:
                    print(f"   💥 {parser_name}: Exception - {str(e)[:50]}...")

        # Demonstrate recovery strategies
        print("\n🔄 Demonstrating recovery strategies...")

        # Try parsing with intelligent fallback
        try:
            from lindela.packages_enhanced.crawlers.google_news_crawler.parsers import IntelligentParser

            intelligent_parser = IntelligentParser()

            # Test with mixed content that might confuse single parsers
            mixed_content = '''
            This looks like plain text at first, but it contains some HTML elements like <p>paragraphs</p>
            and maybe some JSON-like structures {"key": "value"} within the text.
            '''

            result = await intelligent_parser.parse(mixed_content)
            print(f"🤖 Intelligent parser result: {result.status.value}")

            if result.error:
                print(f"   Error: {result.error}")

        except ImportError:
            print("🤖 Intelligent parser not available")

        print("\n💡 Error handling best practices:")
        print("   1. Always check the ParseStatus before using results")
        print("   2. Handle different error types appropriately")
        print("   3. Use fallback strategies for critical operations")
        print("   4. Log errors for debugging but don't crash the application")
        print("   5. Provide meaningful error messages to users")

    except Exception as e:
        print(f"❌ Error in error handling example: {e}")
        logger.error(f"Error handling example failed: {e}")


async def example_7_compatibility_wrapper():
    """
    Example 7: GNews compatibility wrapper
    =====================================

    This example shows how to use the GNews compatibility wrapper for easy migration.
    """
    print("\n" + "="*50)
    print("Example 7: GNews Compatibility Wrapper")
    print("="*50)

    if not DEPENDENCIES_AVAILABLE:
        print("❌ Dependencies not available - skipping example")
        return

    try:
        from lindela.packages_enhanced.crawlers.google_news_crawler import GNewsCompatibilityWrapper

        # Mock database manager
        class MockDBManager:
            async def initialize(self):
                pass
            async def close(self):
                pass
            async def article_exists(self, url):
                return False
            async def insert_article(self, article_data):
                return f"mock_id_{hash(article_data.get('url', ''))}"

        db_manager = MockDBManager()

        # Create compatibility wrapper (similar to original GNews API)
        gnews = GNewsCompatibilityWrapper(
            language='en',
            country='US',
            max_results=10,
            db_manager=db_manager
        )

        print("📰 Using GNews compatibility wrapper...")
        print("🔧 This provides the same API as the original GNews library")

        # Example usage (similar to original GNews)
        try:
            # Search for news (this would work like the original GNews)
            print("\n🔍 Searching for 'technology' news...")

            # Note: This is a mock example - actual implementation would connect to services
            print("   Search query: 'technology'")
            print("   Language: en")
            print("   Country: US")
            print("   Max results: 10")

            # Show what the API would look like
            print("\n📋 API methods available:")
            print("   - get_news(query)")
            print("   - get_news_by_topic(topic)")
            print("   - get_news_by_location(location)")
            print("   - get_news_by_site(site)")
            print("   - get_full_article(article)")

            print("\n💡 Migration benefits:")
            print("   ✅ Drop-in replacement for existing GNews code")
            print("   ✅ Enhanced features (database integration, stealth crawling)")
            print("   ✅ Better error handling and recovery")
            print("   ✅ Advanced filtering and content extraction")
            print("   ✅ Performance optimization and caching")

        except Exception as e:
            print(f"⚠️  Mock search demonstration: {e}")

        # Cleanup
        if hasattr(gnews, '_enhanced_client') and gnews._enhanced_client:
            await gnews._enhanced_client.close()

        print("\n✅ Compatibility wrapper example completed")

    except Exception as e:
        print(f"❌ Error in compatibility wrapper: {e}")
        logger.error(f"Compatibility wrapper failed: {e}")


async def main():
    """
    Main function to run all examples
    ================================
    """
    print("🚀 Google News Crawler - Basic Usage Examples")
    print("=" * 60)
    print("This script demonstrates various usage patterns and features")
    print("of the enhanced Google News crawler package.")
    print()

    # List of examples to run
    examples = [
        example_1_basic_setup,
        example_2_simple_news_search,
        example_3_rss_feed_parsing,
        example_4_html_article_extraction,
        example_5_configuration_management,
        example_6_error_handling,
        example_7_compatibility_wrapper
    ]

    # Run examples
    for i, example in enumerate(examples, 1):
        try:
            await example()
        except KeyboardInterrupt:
            print(f"\n⏹️  Interrupted during example {i}")
            break
        except Exception as e:
            print(f"\n💥 Example {i} failed with unexpected error: {e}")
            logger.error(f"Example {i} failed: {e}")

        # Small delay between examples
        await asyncio.sleep(0.5)

    print("\n" + "="*60)
    print("🎉 All examples completed!")
    print("\n📚 Next steps:")
    print("   1. Check the advanced_usage.py file for more complex scenarios")
    print("   2. Review the configuration options in config.py")
    print("   3. Look at the test files for more detailed examples")
    print("   4. Read the API documentation for full feature coverage")
    print("\n💬 For support and questions:")
    print("   - Check the README.md file")
    print("   - Review the inline documentation")
    print("   - Contact: nyimbi@datacraft.co.ke")


if __name__ == "__main__":
    # Run the examples
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")
