# Enhanced YouTube Crawler Package

A comprehensive, enterprise-grade YouTube content crawler with advanced filtering, metadata extraction, and integration capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-Source Data Extraction**: YouTube Data API v3 + Web Scraping
- **Comprehensive Content Types**: Videos, Channels, Playlists, Comments, Transcripts
- **Advanced Filtering**: Quality-based filtering and credibility assessment
- **Stealth Crawling**: Intelligent rate limiting and request optimization
- **Database Integration**: PostgreSQL support with async operations
- **Batch Processing**: Concurrent crawling with configurable limits
- **Caching System**: Multi-level caching for performance optimization

### Data Extraction Features
- **Video Metadata**: Title, description, statistics, thumbnails, duration
- **Channel Analytics**: Subscriber count, upload frequency, verification status
- **Comment Analysis**: Sentiment analysis, threading, engagement metrics
- **Transcript Extraction**: Multiple language support, timing information
- **Thumbnail Processing**: Multi-resolution support, image analysis
- **Real-time Statistics**: View counts, likes, engagement rates

### Performance Features
- **Async Architecture**: Built on asyncio for high concurrency
- **Rate Limiting**: Intelligent request throttling to avoid API limits
- **Error Handling**: Comprehensive retry logic and fallback mechanisms
- **Monitoring**: Built-in performance metrics and health checks
- **Scalability**: Horizontal scaling support with load balancing

## 📦 Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
# Clone the repository
git clone <repository-url>
cd youtube_crawler

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Docker Installation
```bash
# Build Docker image
docker build -t youtube-crawler .

# Run with Docker Compose
docker-compose up -d
```

## 🔧 Quick Start

### Basic Usage

```python
import asyncio
from youtube_crawler import create_enhanced_youtube_client, CrawlerConfig

async def main():
    # Create configuration
    config = CrawlerConfig()
    config.api.api_key = "YOUR_YOUTUBE_API_KEY"
    
    # Create client
    client = await create_enhanced_youtube_client(config)
    
    # Crawl a single video
    result = await client.crawl_video("dQw4w9WgXcQ")
    
    if result.success:
        video_data = result.data
        print(f"Video: {video_data.title}")
        print(f"Views: {video_data.view_count:,}")
        print(f"Channel: {video_data.channel_title}")
    
    # Search and crawl videos
    search_results = await client.search_and_crawl(
        query="python programming",
        max_results=10
    )
    
    print(f"Found {search_results.extracted_count} videos")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Configuration

```python
from youtube_crawler import CrawlerConfig, APIConfig, FilteringConfig

# Create advanced configuration
config = CrawlerConfig()

# API Configuration
config.api = APIConfig(
    api_key="YOUR_API_KEY",
    quota_limit=10000,
    requests_per_minute=60,
    fallback_to_scraping=True
)

# Filtering Configuration
config.filtering = FilteringConfig(
    min_video_duration=60,  # 1 minute minimum
    max_video_duration=3600,  # 1 hour maximum
    min_view_count=1000,
    allowed_languages=["en", "es", "fr"],
    quality_threshold=0.7
)

# Extraction Configuration
config.extraction.extract_comments = True
config.extraction.extract_transcripts = True
config.extraction.extract_thumbnails = True
config.extraction.max_comments = 50

# Performance Configuration
config.performance.concurrent_requests = 10
config.performance.batch_size = 25
```

## 📊 Configuration

### Environment Variables

```bash
# API Configuration
YOUTUBE_API_KEY=your_api_key_here
YOUTUBE_CLIENT_ID=your_client_id
YOUTUBE_CLIENT_SECRET=your_client_secret

# Database Configuration
YOUTUBE_DB_HOST=localhost
YOUTUBE_DB_PORT=5432
YOUTUBE_DB_NAME=youtube_data
YOUTUBE_DB_USER=postgres
YOUTUBE_DB_PASS=your_password

# Performance Settings
YOUTUBE_CONCURRENT_REQUESTS=5
YOUTUBE_BATCH_SIZE=50
YOUTUBE_CACHE_TTL=3600

# Logging
YOUTUBE_LOG_LEVEL=INFO
YOUTUBE_LOG_FILE=/var/log/youtube_crawler.log
```

### Configuration File (config.yml)

```yaml
crawl_mode: hybrid  # api_only, scraping_only, hybrid, auto
geographical_focus: global
content_types:
  - video
  - channel

api:
  api_key: ${YOUTUBE_API_KEY}
  quota_limit: 10000
  requests_per_minute: 60
  enable_quota_monitoring: true
  fallback_to_scraping: true

scraping:
  enable_stealth: true
  request_delay: 1.0
  random_delay: true
  max_retries: 3
  timeout: 30

database:
  host: ${YOUTUBE_DB_HOST:localhost}
  port: ${YOUTUBE_DB_PORT:5432}
  database: ${YOUTUBE_DB_NAME:youtube_data}
  username: ${YOUTUBE_DB_USER:postgres}
  password: ${YOUTUBE_DB_PASS}

filtering:
  min_video_duration: 30
  max_video_duration: 7200
  min_view_count: 100
  allowed_languages: [en, es, fr, de]
  quality_threshold: 0.5

extraction:
  extract_transcripts: true
  extract_comments: true
  extract_thumbnails: true
  max_comments: 100
  transcript_languages: [en]

performance:
  concurrent_requests: 5
  batch_size: 50
  enable_compression: true
  memory_limit_mb: 512
```

## 🎯 Usage Examples

### Video Crawling

```python
# Crawl single video
result = await client.crawl_video("dQw4w9WgXcQ")

# Crawl multiple videos
video_ids = ["dQw4w9WgXcQ", "oHg5SJYRHA0", "fJ9rUzIMcZQ"]
results = await client.batch_crawl_videos(video_ids)

# Crawl with specific options
result = await client.crawl_video(
    "dQw4w9WgXcQ",
    use_api=True,
    extract_comments=True,
    extract_transcripts=True
)
```

### Channel Analysis

```python
# Crawl channel information
channel_result = await client.crawl_channel("UCuAXFkgsw1L7xaCfnd5JJOw")

if channel_result.success:
    channel = channel_result.data
    print(f"Channel: {channel.title}")
    print(f"Subscribers: {channel.subscriber_count:,}")
    print(f"Videos: {channel.video_count:,}")
    print(f"Tier: {channel.get_subscriber_tier()}")
    print(f"Activity: {channel.get_activity_level()}")
```

### Search and Discovery

```python
# Search for videos
search_results = await client.search_and_crawl(
    query="machine learning tutorial",
    max_results=50,
    order="relevance"
)

# Filter high-quality results
quality_videos = [
    video for video in search_results.items
    if video.quality_score > 0.7 and video.view_count > 10000
]

print(f"Found {len(quality_videos)} high-quality videos")
```

### Comment Analysis

```python
# Extract and analyze comments
if config.extraction.extract_comments:
    for video in search_results.items:
        if video.comments:
            print(f"\nComments for {video.title}:")
            for comment in video.comments[:5]:  # Top 5 comments
                print(f"- {comment.author_name}: {comment.text[:100]}...")
                print(f"  Likes: {comment.like_count}, Sentiment: {comment.sentiment}")
```

### Transcript Processing

```python
# Extract and process transcripts
if config.extraction.extract_transcripts:
    for video in search_results.items:
        if video.transcript:
            transcript = video.transcript
            print(f"\nTranscript for {video.title}:")
            print(f"Language: {transcript.language}")
            print(f"Duration: {transcript.get_duration_minutes():.1f} minutes")
            print(f"Word count: {transcript.word_count}")
            print(f"Speaking rate: {transcript.get_words_per_minute():.1f} WPM")
```

## 🔍 Data Models

### VideoData
```python
@dataclass
class VideoData:
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    published_at: datetime
    duration: timedelta
    view_count: int
    like_count: int
    comment_count: int
    tags: List[str]
    thumbnails: List[ThumbnailData]
    transcript: Optional[TranscriptData]
    comments: List[CommentData]
    quality_score: float
    engagement_rate: float
```

### ChannelData
```python
@dataclass
class ChannelData:
    channel_id: str
    title: str
    description: str
    subscriber_count: int
    video_count: int
    view_count: int
    upload_frequency: float
    verification_status: bool
    credibility_score: float
```

## 🎨 Parsers

The package includes specialized parsers for different content types:

### Video Parser
```python
from youtube_crawler.parsers import VideoParser, VideoMetadataParser

# Basic video parsing
parser = VideoParser()
result = await parser.parse(api_response, ContentType.VIDEO)

# Enhanced metadata parsing
metadata_parser = VideoMetadataParser()
result = await metadata_parser.parse(api_response, ContentType.VIDEO)
```

### Comment Parser
```python
from youtube_crawler.parsers import CommentParser, CommentAnalyzer

# Parse comments
comment_parser = CommentParser()
comments = await comment_parser.parse(comments_data, ContentType.COMMENT)

# Analyze sentiment
analyzer = CommentAnalyzer()
sentiment_results = await analyzer.analyze_sentiment(comments)
```

### Transcript Parser
```python
from youtube_crawler.parsers import TranscriptParser

# Parse video transcripts
transcript_parser = TranscriptParser()
transcript = await transcript_parser.parse(vtt_content, ContentType.TRANSCRIPT)
```

## 🚀 Performance Optimization

### Concurrent Processing
```python
# Configure concurrent requests
config.performance.concurrent_requests = 10
config.performance.batch_size = 100

# Use semaphores for rate limiting
semaphore = asyncio.Semaphore(5)

async def crawl_with_limit(video_id):
    async with semaphore:
        return await client.crawl_video(video_id)
```

### Caching Strategy
```python
# Configure caching
config.cache.enable_caching = True
config.cache.cache_backend = "redis"
config.cache.cache_ttl = 3600  # 1 hour

# Cache video data
await client.cache_manager.set_video_data(video_id, video_data)
cached_data = await client.cache_manager.get_video_data(video_id)
```

### Memory Management
```python
# Configure memory limits
config.performance.memory_limit_mb = 1024
config.performance.optimize_for_speed = True

# Use streaming for large datasets
async for video_batch in client.stream_crawl_videos(video_ids, batch_size=50):
    process_batch(video_batch)
```

## 📈 Monitoring and Analytics

### Performance Metrics
```python
# Get performance statistics
stats = client.get_performance_stats()
print(f"Videos crawled: {stats['videos_crawled']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Requests per minute: {stats['requests_per_minute']:.1f}")
print(f"API quota used: {stats['api_quota_used']}")
```

### Health Checks
```python
# Perform health check
health_status = await client.health_check()
print(f"Status: {health_status['status']}")
print(f"Dependencies: {health_status['dependencies']}")
```

### Error Reporting
```python
# Get error summary
from youtube_crawler.api.exceptions import error_reporter

error_summary = error_reporter.get_error_summary()
print(f"Total errors: {error_summary['total_errors']}")
print(f"Most common error: {error_summary['most_common_error']}")
```

## 🔧 Advanced Features

### Custom Filtering
```python
# Create custom filter
def custom_video_filter(video: VideoData) -> bool:
    return (
        video.view_count > 1000 and
        video.duration.total_seconds() > 300 and
        video.engagement_rate > 2.0
    )

# Apply filter
filtered_videos = [v for v in videos if custom_video_filter(v)]
```

### Data Export
```python
# Export to different formats
await client.export_data(videos, format="json", output_file="videos.json")
await client.export_data(videos, format="csv", output_file="videos.csv")
await client.export_data(videos, format="parquet", output_file="videos.parquet")
```

### Scheduled Crawling
```python
import schedule
import time

def scheduled_crawl():
    asyncio.run(client.search_and_crawl("trending topics"))

# Schedule crawling every hour
schedule.every().hour.do(scheduled_crawl)

while True:
    schedule.run_pending()
    time.sleep(1)
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=youtube_crawler

# Run specific test file
pytest tests/test_video_parser.py

# Run with verbose output
pytest -v tests/
```

### Integration Tests
```bash
# Run integration tests (requires API key)
YOUTUBE_API_KEY=your_key pytest tests/integration/

# Run with real API calls
pytest tests/integration/ --api-key=your_key
```

### Performance Tests
```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Profile memory usage
pytest tests/performance/ --profile
```

## 📝 API Reference

### Main Client
```python
class EnhancedYouTubeClient:
    async def crawl_video(video_id: str) -> CrawlResult
    async def crawl_channel(channel_id: str) -> CrawlResult
    async def batch_crawl_videos(video_ids: List[str]) -> ExtractResult
    async def search_and_crawl(query: str, max_results: int) -> ExtractResult
    def get_performance_stats() -> Dict[str, Any]
```

### Configuration Classes
```python
class CrawlerConfig:
    api: APIConfig
    scraping: ScrapingConfig
    database: DatabaseConfig
    filtering: FilteringConfig
    extraction: ExtractionConfig
    performance: PerformanceConfig
```

### Data Models
```python
class VideoData, ChannelData, CommentData, TranscriptData, ThumbnailData
class CrawlResult, ExtractResult, ValidationResult
class ContentType, VideoQuality, DataQuality
```

## 🐛 Troubleshooting

### Common Issues

**API Quota Exceeded**
```python
# Handle quota exceeded
try:
    result = await client.crawl_video(video_id)
except APIQuotaExceededError as e:
    print(f"Quota exceeded: {e.quota_used}/{e.quota_limit}")
    print(f"Reset time: {e.reset_time}")
```

**Rate Limiting**
```python
# Configure rate limiting
config.api.requests_per_minute = 30  # Reduce request rate
config.scraping.request_delay = 2.0  # Increase delay between requests
```

**Memory Issues**
```python
# Optimize memory usage
config.performance.memory_limit_mb = 512
config.performance.batch_size = 25  # Reduce batch size
config.cache.max_cache_size = 500   # Limit cache size
```

### Debugging
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get detailed error information
try:
    result = await client.crawl_video(video_id)
except Exception as e:
    print(f"Error details: {e.__dict__}")
    import traceback
    traceback.print_exc()
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd youtube_crawler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Standards
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing
- **Sphinx** for documentation

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit pull request with description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YouTube Data API v3 for official API access
- yt-dlp for video metadata extraction
- BeautifulSoup for HTML parsing
- aiohttp for async HTTP requests
- PostgreSQL for data storage

## 📞 Support

For questions, issues, or contributions:

- **Email**: nyimbi@datacraft.co.ke
- **Company**: Datacraft (www.datacraft.co.ke)
- **Issues**: GitHub Issues
- **Documentation**: Full documentation available in `/docs`

## 🔄 Changelog

### Version 1.0.0
- Initial release
- YouTube Data API v3 integration
- Web scraping capabilities
- Comprehensive data models
- Async architecture
- Multi-level caching
- Advanced filtering
- Performance optimization
- Comprehensive testing suite

---

**Built with ❤️ by Datacraft**