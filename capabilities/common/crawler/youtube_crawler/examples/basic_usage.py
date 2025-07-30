#!/usr/bin/env python3
"""
Basic YouTube Crawler Usage Example
===================================

This example demonstrates basic usage of the Enhanced YouTube Crawler package.
Shows how to crawl videos, channels, and perform searches with basic configuration.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import os
import logging
from datetime import datetime
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import YouTube crawler components
try:
    from youtube_crawler import (
        create_enhanced_youtube_client,
        create_basic_youtube_client,
        CrawlerConfig,
        APIConfig,
        FilteringConfig,
        ExtractionConfig,
        PerformanceConfig
    )
    from youtube_crawler.api.data_models import VideoData, ChannelData
    from youtube_crawler.api.exceptions import YouTubeCrawlerError, APIQuotaExceededError
except ImportError as e:
    logger.error(f"Failed to import YouTube crawler: {e}")
    logger.info("Make sure to install the package dependencies first")
    exit(1)


async def basic_video_crawling():
    """Demonstrate basic video crawling."""
    logger.info("=== Basic Video Crawling ===")

    # Get API key from environment
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        logger.warning("No YouTube API key found. Using scraping-only mode.")
        api_key = None

    try:
        # Create basic client
        if api_key:
            client = await create_basic_youtube_client(api_key)
        else:
            # Create client without API key (scraping only)
            config = CrawlerConfig()
            config.crawl_mode = "scraping_only"
            client = await create_enhanced_youtube_client(config)

        # Popular video IDs for testing
        test_video_ids = [
            "dQw4w9WgXcQ",  # Rick Roll (famous video)
            "jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
            "9bZkp7q19f0"   # Gangnam Style
        ]

        for video_id in test_video_ids:
            logger.info(f"Crawling video: {video_id}")

            try:
                result = await client.crawl_video(video_id)

                if result.success:
                    video = result.data
                    logger.info(f"✓ Successfully crawled: {video.title}")
                    logger.info(f"  Channel: {video.channel_title}")
                    logger.info(f"  Views: {video.view_count:,}")
                    logger.info(f"  Duration: {video.get_duration_minutes():.1f} minutes")
                    logger.info(f"  Published: {video.published_at}")
                    logger.info(f"  Source: {result.source}")
                    logger.info(f"  Execution time: {result.execution_time:.2f}s")
                else:
                    logger.error(f"✗ Failed to crawl {video_id}: {result.error_message}")

            except Exception as e:
                logger.error(f"✗ Error crawling {video_id}: {e}")

            # Small delay between requests
            await asyncio.sleep(1)

        # Get performance statistics
        stats = client.get_performance_stats()
        logger.info("\n=== Performance Statistics ===")
        logger.info(f"Videos crawled: {stats['videos_crawled']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"Total requests: {stats['total_requests']}")
        logger.info(f"Average requests per minute: {stats['requests_per_minute']:.1f}")

    except Exception as e:
        logger.error(f"Error in basic video crawling: {e}")


async def advanced_configuration_example():
    """Demonstrate advanced configuration options."""
    logger.info("\n=== Advanced Configuration Example ===")

    # Create advanced configuration
    config = CrawlerConfig()

    # API Configuration
    config.api = APIConfig(
        api_key=os.getenv('YOUTUBE_API_KEY'),
        quota_limit=10000,
        requests_per_minute=60,
        enable_quota_monitoring=True,
        fallback_to_scraping=True
    )

    # Filtering Configuration
    config.filtering = FilteringConfig(
        min_video_duration=60,      # 1 minute minimum
        max_video_duration=1800,    # 30 minutes maximum
        min_view_count=1000,        # Minimum 1K views
        allowed_languages=["en"],   # English only
        quality_threshold=0.5       # Minimum quality score
    )

    # Extraction Configuration
    config.extraction = ExtractionConfig(
        extract_comments=True,
        extract_transcripts=True,
        extract_thumbnails=True,
        max_comments=20,
        transcript_languages=["en"]
    )

    # Performance Configuration
    config.performance = PerformanceConfig(
        concurrent_requests=3,
        batch_size=10,
        optimize_for_accuracy=True
    )

    try:
        # Create client with advanced config
        client = await create_enhanced_youtube_client(config)

        # Test with a popular educational channel
        test_video_id = "kJQP7kiw5Fk"  # DeepMind AlphaGo documentary

        logger.info(f"Crawling video with advanced extraction: {test_video_id}")

        result = await client.crawl_video(test_video_id)

        if result.success:
            video = result.data
            logger.info(f"✓ Video: {video.title}")
            logger.info(f"  Quality Score: {video.quality_score:.2f}")
            logger.info(f"  Engagement Rate: {video.get_engagement_rate():.2f}%")

            # Show comment information
            if video.comments:
                logger.info(f"  Comments extracted: {len(video.comments)}")
                for i, comment in enumerate(video.comments[:3]):  # Show first 3
                    logger.info(f"    Comment {i+1}: {comment.text[:50]}...")
                    logger.info(f"      Author: {comment.author_name}, Likes: {comment.like_count}")

            # Show transcript information
            if video.transcript:
                transcript = video.transcript
                logger.info(f"  Transcript extracted: {transcript.language}")
                logger.info(f"    Word count: {transcript.word_count}")
                logger.info(f"    Duration: {transcript.get_duration_minutes():.1f} minutes")
                logger.info(f"    Text preview: {transcript.text[:100]}...")
        else:
            logger.error(f"✗ Failed: {result.error_message}")

    except APIQuotaExceededError as e:
        logger.error(f"API quota exceeded: {e}")
        logger.info("Consider using scraping mode or wait for quota reset")
    except Exception as e:
        logger.error(f"Error in advanced configuration example: {e}")


async def batch_crawling_example():
    """Demonstrate batch crawling capabilities."""
    logger.info("\n=== Batch Crawling Example ===")

    try:
        # Create client
        config = CrawlerConfig()
        config.api.api_key = os.getenv('YOUTUBE_API_KEY')
        config.performance.concurrent_requests = 3
        config.performance.batch_size = 5

        client = await create_enhanced_youtube_client(config)

        # List of video IDs to crawl
        video_ids = [
            "dQw4w9WgXcQ",  # Rick Roll
            "jNQXAC9IVRw",  # Me at the zoo
            "9bZkp7q19f0",  # Gangnam Style
            "kJQP7kiw5Fk",  # DeepMind AlphaGo
            "fJ9rUzIMcZQ"   # Charlie bit my finger
        ]

        logger.info(f"Batch crawling {len(video_ids)} videos...")

        # Perform batch crawl
        start_time = datetime.now()
        results = await client.batch_crawl_videos(video_ids)
        end_time = datetime.now()

        # Display results
        logger.info(f"✓ Batch crawl completed in {(end_time - start_time).total_seconds():.2f}s")
        logger.info(f"  Extracted: {results.extracted_count}/{len(video_ids)} videos")
        logger.info(f"  Failed: {results.failed_count} videos")
        logger.info(f"  Success rate: {(results.extracted_count / len(video_ids)) * 100:.1f}%")

        # Show successful results
        for i, video in enumerate(results.items[:3]):  # Show first 3
            logger.info(f"  Video {i+1}: {video.title}")
            logger.info(f"    Views: {video.view_count:,}")
            logger.info(f"    Channel: {video.channel_title}")

        # Show errors if any
        if results.errors:
            logger.warning("Errors encountered:")
            for error in results.errors[:3]:  # Show first 3 errors
                logger.warning(f"  - {error}")

    except Exception as e:
        logger.error(f"Error in batch crawling example: {e}")


async def search_and_crawl_example():
    """Demonstrate search and crawl functionality."""
    logger.info("\n=== Search and Crawl Example ===")

    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        logger.warning("Search functionality requires YouTube API key")
        return

    try:
        # Create client with API support
        config = CrawlerConfig()
        config.api.api_key = api_key
        config.filtering.min_view_count = 10000  # Filter for popular videos

        client = await create_enhanced_youtube_client(config)

        # Search queries to test
        search_queries = [
            "python programming tutorial",
            "machine learning explained",
            "web development 2024"
        ]

        for query in search_queries:
            logger.info(f"Searching for: '{query}'")

            try:
                # Search and crawl top 5 results
                results = await client.search_and_crawl(
                    query=query,
                    max_results=5,
                    order="relevance"
                )

                if results.extracted_count > 0:
                    logger.info(f"✓ Found {results.extracted_count} videos")

                    # Display top results
                    for i, video in enumerate(results.items[:3]):
                        logger.info(f"  {i+1}. {video.title}")
                        logger.info(f"     Channel: {video.channel_title}")
                        logger.info(f"     Views: {video.view_count:,}")
                        logger.info(f"     Duration: {video.get_duration_minutes():.1f} min")
                else:
                    logger.warning(f"No results found for '{query}'")

            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")

            # Delay between searches
            await asyncio.sleep(2)

    except Exception as e:
        logger.error(f"Error in search and crawl example: {e}")


async def channel_analysis_example():
    """Demonstrate channel analysis capabilities."""
    logger.info("\n=== Channel Analysis Example ===")

    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        logger.warning("Channel analysis requires YouTube API key")
        return

    try:
        # Create client
        client = await create_basic_youtube_client(api_key)

        # Popular channels to analyze
        channel_ids = [
            "UCuAXFkgsw1L7xaCfnd5JJOw",  # Random example channel ID
            "UC_x5XG1OV2P6uZZ5FSM9Ttw",  # Google Developers
        ]

        for channel_id in channel_ids:
            logger.info(f"Analyzing channel: {channel_id}")

            try:
                result = await client.crawl_channel(channel_id)

                if result.success:
                    channel = result.data
                    logger.info(f"✓ Channel: {channel.title}")
                    logger.info(f"  Subscribers: {channel.subscriber_count:,}")
                    logger.info(f"  Videos: {channel.video_count:,}")
                    logger.info(f"  Total views: {channel.view_count:,}")
                    logger.info(f"  Subscriber tier: {channel.get_subscriber_tier()}")
                    logger.info(f"  Activity level: {channel.get_activity_level()}")
                    logger.info(f"  Country: {channel.country}")
                    logger.info(f"  Created: {channel.created_at}")
                else:
                    logger.error(f"✗ Failed to analyze channel: {result.error_message}")

            except Exception as e:
                logger.error(f"Error analyzing channel {channel_id}: {e}")

            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in channel analysis example: {e}")


async def error_handling_example():
    """Demonstrate error handling and resilience."""
    logger.info("\n=== Error Handling Example ===")

    try:
        # Create client with minimal config
        config = CrawlerConfig()
        config.api.api_key = "invalid_api_key"  # Intentionally invalid
        config.api.fallback_to_scraping = True

        client = await create_enhanced_youtube_client(config)

        # Test with invalid video ID
        invalid_video_id = "invalid_video_id"

        logger.info(f"Testing error handling with invalid video ID: {invalid_video_id}")

        result = await client.crawl_video(invalid_video_id)

        if not result.success:
            logger.info(f"✓ Error handled gracefully: {result.error_message}")
            logger.info(f"  Error code: {result.error_code}")
            logger.info(f"  Execution time: {result.execution_time:.2f}s")
        else:
            logger.warning("Expected error but got success - check implementation")

        # Test with quota exceeded simulation
        logger.info("Testing quota exceeded handling...")

        try:
            # This would normally trigger quota exceeded in real usage
            for i in range(3):
                result = await client.crawl_video("dQw4w9WgXcQ")
                logger.info(f"Request {i+1}: {'Success' if result.success else 'Failed'}")

        except APIQuotaExceededError as e:
            logger.info(f"✓ Quota exceeded handled: {e}")
        except Exception as e:
            logger.info(f"Other error handled: {e}")

    except Exception as e:
        logger.error(f"Error in error handling example: {e}")


async def main():
    """Main function to run all examples."""
    logger.info("Starting YouTube Crawler Examples")
    logger.info("=" * 50)

    # Check for API key
    api_key = os.getenv('YOUTUBE_API_KEY')
    if api_key:
        logger.info("✓ YouTube API key found")
    else:
        logger.warning("⚠ No YouTube API key found - some features will be limited")
        logger.info("Set YOUTUBE_API_KEY environment variable for full functionality")

    try:
        # Run all examples
        await basic_video_crawling()
        await advanced_configuration_example()
        await batch_crawling_example()

        # API-dependent examples
        if api_key:
            await search_and_crawl_example()
            await channel_analysis_example()

        await error_handling_example()

        logger.info("\n" + "=" * 50)
        logger.info("All examples completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up environment
    print("Enhanced YouTube Crawler - Basic Usage Examples")
    print("=" * 50)
    print("Make sure to set your YouTube API key:")
    print("export YOUTUBE_API_KEY='your_api_key_here'")
    print()

    # Run examples
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
