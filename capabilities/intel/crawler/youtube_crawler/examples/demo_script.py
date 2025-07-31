#!/usr/bin/env python3
"""
YouTube Crawler Demonstration Script
====================================

Practical demonstration of the Enhanced YouTube Crawler package.
Shows real-world usage scenarios and features.

This script demonstrates:
- Basic video crawling
- Configuration management
- Error handling
- Performance monitoring
- Data analysis
- Export capabilities

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Add the package to Python path for demo purposes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeCrawlerDemo:
    """Demonstration class for YouTube Crawler functionality."""

    def __init__(self):
        self.results = []
        self.stats = {
            'videos_processed': 0,
            'channels_analyzed': 0,
            'errors_encountered': 0,
            'start_time': time.time()
        }

    async def demo_basic_crawling(self):
        """Demonstrate basic video crawling capabilities."""
        print("\n🎥 Demo 1: Basic Video Crawling")
        print("=" * 50)

        try:
            # Mock the YouTube crawler for demonstration
            from demo_mock_client import MockYouTubeClient

            client = MockYouTubeClient()

            # Sample video IDs for demonstration
            demo_videos = [
                "dQw4w9WgXcQ",  # Rick Roll (famous)
                "jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
                "9bZkp7q19f0",  # Gangnam Style
                "kJQP7kiw5Fk",  # Educational content
                "fJ9rUzIMcZQ"   # Popular music video
            ]

            print(f"Crawling {len(demo_videos)} popular videos...")

            for i, video_id in enumerate(demo_videos, 1):
                print(f"\n[{i}/{len(demo_videos)}] Processing: {video_id}")

                try:
                    result = await client.crawl_video(video_id)

                    if result.success:
                        video = result.data
                        self.results.append(video)
                        self.stats['videos_processed'] += 1

                        print(f"  ✓ Title: {video.title}")
                        print(f"  ✓ Channel: {video.channel_title}")
                        print(f"  ✓ Views: {video.view_count:,}")
                        print(f"  ✓ Duration: {video.get_duration_minutes():.1f} minutes")
                        print(f"  ✓ Engagement Rate: {video.get_engagement_rate():.2f}%")
                        print(f"  ✓ Source: {result.source}")
                        print(f"  ✓ Parse Time: {result.execution_time:.2f}s")

                    else:
                        print(f"  ✗ Failed: {result.error_message}")
                        self.stats['errors_encountered'] += 1

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    self.stats['errors_encountered'] += 1

                # Small delay between requests
                await asyncio.sleep(0.5)

            print(f"\n📊 Basic Crawling Summary:")
            print(f"  Videos processed: {self.stats['videos_processed']}")
            print(f"  Success rate: {(self.stats['videos_processed'] / len(demo_videos)) * 100:.1f}%")

        except Exception as e:
            print(f"✗ Demo 1 failed: {e}")

    async def demo_batch_processing(self):
        """Demonstrate batch processing capabilities."""
        print("\n⚡ Demo 2: Batch Processing")
        print("=" * 50)

        try:
            from demo_mock_client import MockYouTubeClient

            client = MockYouTubeClient()

            # Batch of video IDs
            batch_videos = [
                "dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0",
                "kJQP7kiw5Fk", "fJ9rUzIMcZQ", "oHg5SJYRHA0",
                "YQHsXMglC9A", "hTWKbfoikeg", "uelHwf8o7_U"
            ]

            print(f"Batch processing {len(batch_videos)} videos...")

            start_time = time.time()
            batch_results = await client.batch_crawl_videos(batch_videos)
            end_time = time.time()

            print(f"\n📈 Batch Processing Results:")
            print(f"  Total videos: {len(batch_videos)}")
            print(f"  Successfully crawled: {batch_results.extracted_count}")
            print(f"  Failed: {batch_results.failed_count}")
            print(f"  Success rate: {(batch_results.extracted_count / len(batch_videos)) * 100:.1f}%")
            print(f"  Total time: {end_time - start_time:.2f}s")
            print(f"  Average time per video: {(end_time - start_time) / len(batch_videos):.2f}s")

            # Show top videos by view count
            if batch_results.items:
                sorted_videos = sorted(batch_results.items, key=lambda v: v.view_count, reverse=True)
                print(f"\n🏆 Top 3 Videos by Views:")
                for i, video in enumerate(sorted_videos[:3], 1):
                    print(f"  {i}. {video.title}")
                    print(f"     Views: {video.view_count:,}")
                    print(f"     Channel: {video.channel_title}")

        except Exception as e:
            print(f"✗ Demo 2 failed: {e}")

    async def demo_advanced_configuration(self):
        """Demonstrate advanced configuration options."""
        print("\n⚙️ Demo 3: Advanced Configuration")
        print("=" * 50)

        try:
            from demo_mock_client import MockCrawlerConfig, MockYouTubeClient

            # Create advanced configuration
            config = MockCrawlerConfig()

            # API Configuration
            config.api_key = os.getenv('YOUTUBE_API_KEY', 'demo_key')
            config.quota_limit = 10000
            config.requests_per_minute = 60
            config.fallback_to_scraping = True

            # Filtering Configuration
            config.min_video_duration = 60  # 1 minute minimum
            config.max_video_duration = 1800  # 30 minutes maximum
            config.min_view_count = 1000  # Minimum 1K views
            config.quality_threshold = 0.5  # Minimum quality score

            # Extraction Configuration
            config.extract_comments = True
            config.extract_transcripts = True
            config.max_comments = 20

            # Performance Configuration
            config.concurrent_requests = 3
            config.batch_size = 10

            print("📋 Configuration Settings:")
            print(f"  API Key: {'Set' if config.api_key else 'Not set'}")
            print(f"  Quota Limit: {config.quota_limit:,}")
            print(f"  Min Duration: {config.min_video_duration}s")
            print(f"  Max Duration: {config.max_video_duration}s")
            print(f"  Min Views: {config.min_view_count:,}")
            print(f"  Extract Comments: {config.extract_comments}")
            print(f"  Extract Transcripts: {config.extract_transcripts}")
            print(f"  Concurrent Requests: {config.concurrent_requests}")

            # Test with configuration
            client = MockYouTubeClient(config)
            test_video = "dQw4w9WgXcQ"

            print(f"\n🧪 Testing with configuration on video: {test_video}")
            result = await client.crawl_video(test_video, extract_enhanced=True)

            if result.success:
                video = result.data
                print(f"  ✓ Enhanced data extracted")
                print(f"  ✓ Comments: {len(video.comments) if hasattr(video, 'comments') else 0}")
                print(f"  ✓ Has transcript: {hasattr(video, 'transcript') and video.transcript is not None}")
                print(f"  ✓ Quality score: {getattr(video, 'quality_score', 0.0):.2f}")

        except Exception as e:
            print(f"✗ Demo 3 failed: {e}")

    async def demo_error_handling(self):
        """Demonstrate error handling and resilience."""
        print("\n🛡️ Demo 4: Error Handling & Resilience")
        print("=" * 50)

        try:
            from demo_mock_client import MockYouTubeClient

            client = MockYouTubeClient()

            # Test scenarios with different error types
            test_scenarios = [
                ("valid_video", "dQw4w9WgXcQ", "Should succeed"),
                ("invalid_id", "invalid123", "Should fail - invalid ID"),
                ("not_found", "notfound11", "Should fail - not found"),
                ("private_vid", "private123", "Should fail - private video"),
                ("rate_limit", "ratelimit1", "Should trigger rate limiting")
            ]

            print("🧪 Testing error scenarios:")

            for scenario_name, video_id, description in test_scenarios:
                print(f"\n  Testing: {scenario_name}")
                print(f"  Video ID: {video_id}")
                print(f"  Expected: {description}")

                try:
                    result = await client.crawl_video(video_id)

                    if result.success:
                        print(f"  ✓ Success: {result.data.title}")
                    else:
                        print(f"  ⚠ Failed as expected: {result.error_message}")
                        print(f"  ⚠ Error code: {result.error_code}")

                except Exception as e:
                    print(f"  ⚠ Exception handled: {type(e).__name__}: {e}")

                await asyncio.sleep(0.2)

            # Test retry mechanism
            print(f"\n🔄 Testing retry mechanism:")
            print("  Simulating network issues...")

            retry_result = await client.crawl_with_retry("dQw4w9WgXcQ", max_retries=3)
            if retry_result.success:
                print(f"  ✓ Retry successful after {retry_result.retry_count} attempts")
            else:
                print(f"  ⚠ Retry failed: {retry_result.error_message}")

        except Exception as e:
            print(f"✗ Demo 4 failed: {e}")

    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        print("\n📊 Demo 5: Performance Monitoring")
        print("=" * 50)

        try:
            from demo_mock_client import MockYouTubeClient

            client = MockYouTubeClient()

            # Simulate some activity
            print("🏃 Simulating crawler activity...")

            activities = [
                ("Single video crawl", lambda: client.crawl_video("dQw4w9WgXcQ")),
                ("Channel analysis", lambda: client.crawl_channel("UC_channel123")),
                ("Batch processing", lambda: client.batch_crawl_videos(["vid1", "vid2", "vid3"])),
                ("Search operation", lambda: client.search_and_crawl("python tutorial", max_results=5))
            ]

            for activity_name, activity_func in activities:
                print(f"\n  Running: {activity_name}")
                start_time = time.time()

                try:
                    await activity_func()
                    duration = time.time() - start_time
                    print(f"  ✓ Completed in {duration:.2f}s")
                except Exception as e:
                    print(f"  ⚠ Error: {e}")

            # Get performance statistics
            stats = client.get_performance_stats()

            print(f"\n📈 Performance Statistics:")
            print(f"  Videos crawled: {stats['videos_crawled']}")
            print(f"  Channels analyzed: {stats['channels_crawled']}")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Success rate: {stats['success_rate']:.1f}%")
            print(f"  Requests per minute: {stats['requests_per_minute']:.1f}")
            print(f"  Average response time: {stats.get('avg_response_time', 0):.2f}s")
            print(f"  API quota used: {stats.get('api_quota_used', 0)}")

            # Performance recommendations
            print(f"\n💡 Performance Recommendations:")
            if stats['success_rate'] < 95:
                print("  - Consider implementing retry logic")
            if stats['requests_per_minute'] > 50:
                print("  - Consider rate limiting to avoid API quotas")
            if stats.get('avg_response_time', 0) > 5:
                print("  - Consider optimizing request handling")

            print("  - Use batch processing for multiple videos")
            print("  - Implement caching for frequently accessed content")
            print("  - Monitor API quota usage regularly")

        except Exception as e:
            print(f"✗ Demo 5 failed: {e}")

    async def demo_data_analysis(self):
        """Demonstrate data analysis capabilities."""
        print("\n📈 Demo 6: Data Analysis")
        print("=" * 50)

        if not self.results:
            print("⚠ No data available for analysis. Running basic crawling first...")
            await self.demo_basic_crawling()

        if self.results:
            print(f"📊 Analyzing {len(self.results)} videos...")

            # Basic statistics
            total_views = sum(video.view_count for video in self.results)
            total_duration = sum(video.get_duration_minutes() for video in self.results)
            avg_engagement = sum(video.get_engagement_rate() for video in self.results) / len(self.results)

            print(f"\n📋 Content Statistics:")
            print(f"  Total videos analyzed: {len(self.results)}")
            print(f"  Total views: {total_views:,}")
            print(f"  Total duration: {total_duration:.1f} minutes")
            print(f"  Average engagement rate: {avg_engagement:.2f}%")

            # Top performers
            top_by_views = sorted(self.results, key=lambda v: v.view_count, reverse=True)[:3]
            top_by_engagement = sorted(self.results, key=lambda v: v.get_engagement_rate(), reverse=True)[:3]

            print(f"\n🏆 Top Videos by Views:")
            for i, video in enumerate(top_by_views, 1):
                print(f"  {i}. {video.title[:50]}...")
                print(f"     Views: {video.view_count:,}")

            print(f"\n💬 Top Videos by Engagement:")
            for i, video in enumerate(top_by_engagement, 1):
                print(f"  {i}. {video.title[:50]}...")
                print(f"     Engagement: {video.get_engagement_rate():.2f}%")

            # Channel analysis
            channels = {}
            for video in self.results:
                if video.channel_title not in channels:
                    channels[video.channel_title] = []
                channels[video.channel_title].append(video)

            print(f"\n📺 Channel Analysis:")
            for channel, videos in channels.items():
                total_views = sum(v.view_count for v in videos)
                avg_engagement = sum(v.get_engagement_rate() for v in videos) / len(videos)
                print(f"  {channel}:")
                print(f"    Videos: {len(videos)}")
                print(f"    Total views: {total_views:,}")
                print(f"    Avg engagement: {avg_engagement:.2f}%")

        else:
            print("⚠ No data available for analysis")

    async def demo_export_capabilities(self):
        """Demonstrate data export capabilities."""
        print("\n💾 Demo 7: Data Export")
        print("=" * 50)

        if not self.results:
            print("⚠ No data to export. Generating sample data...")
            await self.demo_basic_crawling()

        if self.results:
            # Export to JSON
            export_data = {
                'export_info': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'video_count': len(self.results),
                    'crawler_version': '1.0.0'
                },
                'videos': []
            }

            for video in self.results:
                video_data = {
                    'video_id': video.video_id,
                    'title': video.title,
                    'channel_title': video.channel_title,
                    'view_count': video.view_count,
                    'duration_minutes': video.get_duration_minutes(),
                    'engagement_rate': video.get_engagement_rate(),
                    'published_at': video.published_at.isoformat() if hasattr(video, 'published_at') and video.published_at else None,
                    'crawled_at': video.crawled_at.isoformat() if hasattr(video, 'crawled_at') else datetime.utcnow().isoformat()
                }
                export_data['videos'].append(video_data)

            # Save to file
            export_filename = f"youtube_crawler_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            try:
                with open(export_filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                file_size = os.path.getsize(export_filename)
                print(f"✓ Data exported to: {export_filename}")
                print(f"✓ File size: {file_size:,} bytes")
                print(f"✓ Videos exported: {len(export_data['videos'])}")

                # Show sample of exported data
                print(f"\n📄 Sample exported data:")
                sample_video = export_data['videos'][0] if export_data['videos'] else {}
                for key, value in list(sample_video.items())[:5]:
                    print(f"  {key}: {value}")

            except Exception as e:
                print(f"✗ Export failed: {e}")

            # Export summary
            print(f"\n📋 Export Summary:")
            print(f"  Format: JSON")
            print(f"  Total videos: {len(export_data['videos'])}")
            print(f"  Fields per video: {len(export_data['videos'][0]) if export_data['videos'] else 0}")
            print(f"  File location: ./{export_filename}")

        else:
            print("⚠ No data available for export")

    def print_final_summary(self):
        """Print final demonstration summary."""
        print("\n" + "=" * 60)
        print("🎉 YouTube Crawler Demonstration Complete!")
        print("=" * 60)

        runtime = time.time() - self.stats['start_time']

        print(f"📊 Overall Statistics:")
        print(f"  Demo runtime: {runtime:.2f} seconds")
        print(f"  Videos processed: {self.stats['videos_processed']}")
        print(f"  Channels analyzed: {self.stats['channels_analyzed']}")
        print(f"  Errors encountered: {self.stats['errors_encountered']}")

        if self.stats['videos_processed'] > 0:
            success_rate = (self.stats['videos_processed'] /
                          (self.stats['videos_processed'] + self.stats['errors_encountered'])) * 100
            print(f"  Success rate: {success_rate:.1f}%")

        print(f"\n🚀 Package Features Demonstrated:")
        print(f"  ✓ Basic video crawling")
        print(f"  ✓ Batch processing")
        print(f"  ✓ Advanced configuration")
        print(f"  ✓ Error handling & resilience")
        print(f"  ✓ Performance monitoring")
        print(f"  ✓ Data analysis")
        print(f"  ✓ Export capabilities")

        print(f"\n📚 Next Steps:")
        print(f"  1. Install dependencies: pip install -r requirements.txt")
        print(f"  2. Set YouTube API key: export YOUTUBE_API_KEY='your_key'")
        print(f"  3. Run real crawling tests")
        print(f"  4. Integrate with your application")
        print(f"  5. Configure production deployment")

        print(f"\n📞 Support:")
        print(f"  Email: nyimbi@datacraft.co.ke")
        print(f"  Company: Datacraft (www.datacraft.co.ke)")
        print(f"  Documentation: See README.md")

        print("=" * 60)


async def main():
    """Main demonstration function."""
    print("🎬 YouTube Crawler Package - Live Demonstration")
    print("=" * 60)
    print("Author: Nyimbi Odero")
    print("Company: Datacraft (www.datacraft.co.ke)")
    print("Package Version: 1.0.0")
    print("=" * 60)

    # Check if we're in demo mode
    api_key = os.getenv('YOUTUBE_API_KEY')
    if api_key:
        print("✓ YouTube API key found - Real API calls possible")
    else:
        print("⚠ No YouTube API key - Using mock client for demonstration")
        print("  Set YOUTUBE_API_KEY environment variable for real testing")

    print()

    # Initialize demo
    demo = YouTubeCrawlerDemo()

    try:
        # Run all demonstrations
        await demo.demo_basic_crawling()
        await demo.demo_batch_processing()
        await demo.demo_advanced_configuration()
        await demo.demo_error_handling()
        await demo.demo_performance_monitoring()
        await demo.demo_data_analysis()
        await demo.demo_export_capabilities()

    except KeyboardInterrupt:
        print("\n\n⚠ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Demonstration error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.print_final_summary()


if __name__ == "__main__":
    # Create mock client module for demonstration
    mock_client_code = '''
"""Mock YouTube Client for Demonstration"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Any

class MockVideoData:
    def __init__(self, video_id: str, title: str = None):
        self.video_id = video_id
        self.title = title or f"Mock Video {video_id}"
        self.channel_title = "Mock Channel"
        self.view_count = hash(video_id) % 10000000  # Pseudo-random views
        self.like_count = self.view_count // 100
        self.comment_count = self.view_count // 1000
        self.duration = timedelta(minutes=hash(video_id) % 30 + 1)
        self.published_at = datetime.utcnow() - timedelta(days=hash(video_id) % 365)
        self.crawled_at = datetime.utcnow()
        self.comments = []
        self.transcript = None

    def get_duration_minutes(self):
        return self.duration.total_seconds() / 60.0

    def get_engagement_rate(self):
        if self.view_count == 0:
            return 0.0
        total_engagement = self.like_count + self.comment_count
        return (total_engagement / self.view_count) * 100

class MockCrawlResult:
    def __init__(self, success: bool, data: Any = None, error_message: str = None):
        self.success = success
        self.data = data
        self.error_message = error_message
        self.error_code = "MOCK_ERROR" if not success else None
        self.execution_time = 0.1
        self.source = "mock"
        self.retry_count = 0

class MockBatchResult:
    def __init__(self, items: List[Any]):
        self.items = items
        self.extracted_count = len(items)
        self.failed_count = 0

class MockCrawlerConfig:
    def __init__(self):
        self.api_key = None
        self.quota_limit = 10000
        self.requests_per_minute = 60
        self.fallback_to_scraping = True
        self.min_video_duration = 30
        self.max_video_duration = 7200
        self.min_view_count = 0
        self.quality_threshold = 0.0
        self.extract_comments = False
        self.extract_transcripts = False
        self.max_comments = 20
        self.concurrent_requests = 5
        self.batch_size = 50

class MockYouTubeClient:
    def __init__(self, config=None):
        self.config = config or MockCrawlerConfig()
        self.stats = {
            "videos_crawled": 0,
            "channels_crawled": 0,
            "total_requests": 0,
            "success_rate": 98.5,
            "requests_per_minute": 45.0,
            "api_quota_used": 150
        }

    async def crawl_video(self, video_id: str, extract_enhanced: bool = False):
        await asyncio.sleep(0.05)  # Simulate network delay

        # Simulate some failures
        if "invalid" in video_id or "notfound" in video_id:
            return MockCrawlResult(False, None, f"Video {video_id} not found")
        if "private" in video_id:
            return MockCrawlResult(False, None, f"Video {video_id} is private")
        if "ratelimit" in video_id:
            return MockCrawlResult(False, None, "Rate limit exceeded")

        video_data = MockVideoData(video_id)
        self.stats["videos_crawled"] += 1
        self.stats["total_requests"] += 1

        return MockCrawlResult(True, video_data)

    async def crawl_channel(self, channel_id: str):
        await asyncio.sleep(0.1)
        self.stats["channels_crawled"] += 1
        self.stats["total_requests"] += 1
        return MockCrawlResult(True, {"channel_id": channel_id, "title": f"Mock Channel {channel_id}"})

    async def batch_crawl_videos(self, video_ids: List[str]):
        await asyncio.sleep(0.2)
        results = []
        for video_id in video_ids:
            if not ("invalid" in video_id or "notfound" in video_id):
                results.append(MockVideoData(video_id))

        self.stats["videos_crawled"] += len(results)
        self.stats["total_requests"] += len(video_ids)
        return MockBatchResult(results)

    async def search_and_crawl(self, query: str, max_results: int = 5):
        await asyncio.sleep(0.3)
        results = [MockVideoData(f"search_result_{i}") for i in range(max_results)]
        self.stats["videos_crawled"] += len(results)
        self.stats["total_requests"] += 1
        return MockBatchResult(results)

    async def crawl_with_retry(self, video_id: str, max_retries: int = 3):
        result = await self.crawl_video(video_id)
        result.retry_count = 1  # Simulate one retry
        return result

    def get_performance_stats(self):
        return self.stats.copy()
'''

    # Write mock client to file for import
    with open('demo_mock_client.py', 'w') as f:
        f.write(mock_client_code)

    try:
        # Run the demonstration
        asyncio.run(main())
    finally:
        # Clean up mock file
        try:
            os.remove('demo_mock_client.py')
        except:
            pass
