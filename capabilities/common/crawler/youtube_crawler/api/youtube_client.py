"""
YouTube Client Implementation
=============================

Main YouTube crawler client with API and scraping capabilities.
Provides comprehensive video, channel, and playlist data extraction.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

# Third-party imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import yt_dlp
    from bs4 import BeautifulSoup
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")

# Local imports
from .data_models import (
    VideoData, ChannelData, PlaylistData, CommentData, TranscriptData,
    CrawlResult, ExtractResult, EngagementMetrics, ThumbnailData,
    create_video_from_api_response, create_channel_from_api_response,
    ContentStatus, PrivacyStatus
)
from .exceptions import (
    YouTubeCrawlerError, APIQuotaExceededError, VideoNotFoundError,
    ChannelNotFoundError, AccessRestrictedError, RateLimitExceededError,
    NetworkError, ParsingError, TranscriptError, CommentError,
    handle_api_error, is_retriable_error, get_retry_delay
)
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type enumeration."""
    VIDEO = "video"
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    COMMENT = "comment"
    LIVE_STREAM = "live_stream"
    SHORT = "short"


class VideoQuality(Enum):
    """Video quality enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class CrawlPriority(Enum):
    """Crawl priority enumeration."""
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    URGENT = 20


class GeographicalFocus(Enum):
    """Geographical focus enumeration."""
    GLOBAL = "global"
    US = "us"
    GB = "gb"
    CA = "ca"
    AU = "au"
    IN = "in"
    DE = "de"
    FR = "fr"
    JP = "jp"
    BR = "br"
    CUSTOM = "custom"


@dataclass
class ChannelCredibilityMetrics:
    """Channel credibility assessment metrics."""
    subscriber_count: int = 0
    video_count: int = 0
    verification_status: bool = False
    content_quality_score: float = 0.0
    engagement_rate: float = 0.0
    upload_consistency: float = 0.0
    age_months: int = 0

    def calculate_credibility_score(self) -> float:
        """Calculate overall credibility score (0-100)."""
        score = 0.0

        # Subscriber count (max 30 points)
        if self.subscriber_count >= 1000000:
            score += 30
        elif self.subscriber_count >= 100000:
            score += 25
        elif self.subscriber_count >= 10000:
            score += 20
        elif self.subscriber_count >= 1000:
            score += 15
        else:
            score += min(self.subscriber_count / 100, 10)

        # Verification status (10 points)
        if self.verification_status:
            score += 10

        # Content quality (20 points)
        score += min(self.content_quality_score * 20, 20)

        # Engagement rate (20 points)
        score += min(self.engagement_rate * 2, 20)

        # Upload consistency (10 points)
        score += min(self.upload_consistency * 10, 10)

        # Channel age (10 points)
        score += min(self.age_months / 12 * 10, 10)

        return min(score, 100.0)


@dataclass
class VideoMetrics:
    """Video performance metrics."""
    view_count: int = 0
    like_count: int = 0
    dislike_count: int = 0
    comment_count: int = 0
    duration_seconds: int = 0
    age_days: int = 0

    def get_views_per_day(self) -> float:
        """Calculate views per day."""
        return self.view_count / max(self.age_days, 1)

    def get_engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.view_count == 0:
            return 0.0
        total_engagement = self.like_count + self.dislike_count + self.comment_count
        return (total_engagement / self.view_count) * 100


@dataclass
class YouTubeSourceConfig:
    """YouTube source configuration."""
    api_key: Optional[str] = None
    enable_api: bool = True
    enable_scraping: bool = True
    max_requests_per_minute: int = 60
    max_videos_per_request: int = 50
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.VIDEO])
    geographical_focus: GeographicalFocus = GeographicalFocus.GLOBAL
    quality_threshold: float = 0.0


class YouTubeVideo:
    """YouTube video representation."""

    def __init__(self, video_id: str, data: Optional[VideoData] = None):
        self.video_id = video_id
        self.data = data
        self._url = f"https://www.youtube.com/watch?v={video_id}"

    @property
    def url(self) -> str:
        return self._url

    @property
    def title(self) -> str:
        return self.data.title if self.data else ""

    @property
    def channel_id(self) -> str:
        return self.data.channel_id if self.data else ""

    def is_valid(self) -> bool:
        """Check if video data is valid."""
        return self.data is not None and self.data.content_status == ContentStatus.ACTIVE


class YouTubeChannel:
    """YouTube channel representation."""

    def __init__(self, channel_id: str, data: Optional[ChannelData] = None):
        self.channel_id = channel_id
        self.data = data
        self._url = f"https://www.youtube.com/channel/{channel_id}"

    @property
    def url(self) -> str:
        return self._url

    @property
    def title(self) -> str:
        return self.data.title if self.data else ""

    @property
    def subscriber_count(self) -> int:
        return self.data.subscriber_count if self.data else 0


class YouTubePlaylist:
    """YouTube playlist representation."""

    def __init__(self, playlist_id: str, data: Optional[PlaylistData] = None):
        self.playlist_id = playlist_id
        self.data = data
        self._url = f"https://www.youtube.com/playlist?list={playlist_id}"

    @property
    def url(self) -> str:
        return self._url

    @property
    def title(self) -> str:
        return self.data.title if self.data else ""

    @property
    def video_count(self) -> int:
        return self.data.video_count if self.data else 0


class ChannelFilteringEngine:
    """Engine for filtering channels based on credibility and quality."""

    def __init__(self, config: YouTubeSourceConfig):
        self.config = config

    def assess_channel_credibility(self, channel: ChannelData) -> ChannelCredibilityMetrics:
        """Assess channel credibility."""
        age_days = (datetime.utcnow() - channel.published_at).days

        return ChannelCredibilityMetrics(
            subscriber_count=channel.subscriber_count,
            video_count=channel.video_count,
            verification_status=False,  # Would need additional API call
            content_quality_score=0.5,  # Placeholder - would need ML model
            engagement_rate=channel.engagement_metrics.get_engagement_rate(),
            upload_consistency=channel.upload_frequency,
            age_months=age_days // 30
        )

    def filter_channels(self, channels: List[ChannelData]) -> List[ChannelData]:
        """Filter channels based on quality criteria."""
        filtered = []

        for channel in channels:
            credibility = self.assess_channel_credibility(channel)
            score = credibility.calculate_credibility_score()

            if score >= self.config.quality_threshold * 100:
                filtered.append(channel)

        return filtered


class YouTubeAPIWrapper:
    """Wrapper for YouTube Data API v3."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.service = None
        self.quota_used = 0
        self.requests_made = 0
        self.last_request_time = 0

        try:
            self.service = build('youtube', 'v3', developerKey=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API service: {e}")
            raise YouTubeCrawlerError(f"API initialization failed: {e}")

    async def get_video_details(self, video_id: str) -> VideoData:
        """Get detailed video information."""
        try:
            request = self.service.videos().list(
                part='snippet,statistics,contentDetails,status',
                id=video_id
            )
            response = request.execute()
            self.quota_used += 1

            if not response.get('items'):
                raise VideoNotFoundError(video_id)

            video_data = response['items'][0]
            return create_video_from_api_response(video_data)

        except HttpError as e:
            raise handle_api_error(json.loads(e.content.decode()))

    async def get_channel_details(self, channel_id: str) -> ChannelData:
        """Get detailed channel information."""
        try:
            request = self.service.channels().list(
                part='snippet,statistics,contentDetails',
                id=channel_id
            )
            response = request.execute()
            self.quota_used += 1

            if not response.get('items'):
                raise ChannelNotFoundError(channel_id)

            channel_data = response['items'][0]
            return create_channel_from_api_response(channel_data)

        except HttpError as e:
            raise handle_api_error(json.loads(e.content.decode()))

    async def search_videos(
        self,
        query: str,
        max_results: int = 50,
        order: str = 'relevance'
    ) -> List[str]:
        """Search for videos and return video IDs."""
        try:
            request = self.service.search().list(
                part='id',
                q=query,
                type='video',
                maxResults=min(max_results, 50),
                order=order
            )
            response = request.execute()
            self.quota_used += 100  # Search costs 100 units

            video_ids = []
            for item in response.get('items', []):
                if item['id']['kind'] == 'youtube#video':
                    video_ids.append(item['id']['videoId'])

            return video_ids

        except HttpError as e:
            raise handle_api_error(json.loads(e.content.decode()))

    async def get_video_comments(
        self,
        video_id: str,
        max_results: int = 100
    ) -> List[CommentData]:
        """Get video comments."""
        try:
            request = self.service.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(max_results, 100),
                order='relevance'
            )
            response = request.execute()
            self.quota_used += 1

            comments = []
            for item in response.get('items', []):
                comment_data = item['snippet']['topLevelComment']['snippet']
                comments.append(CommentData(
                    comment_id=item['id'],
                    author_name=comment_data['authorDisplayName'],
                    author_channel_id=comment_data.get('authorChannelId', {}).get('value'),
                    text=comment_data['textDisplay'],
                    like_count=comment_data.get('likeCount', 0),
                    reply_count=item['snippet'].get('totalReplyCount', 0),
                    published_at=datetime.fromisoformat(
                        comment_data['publishedAt'].replace('Z', '+00:00')
                    )
                ))

            return comments

        except HttpError as e:
            if e.resp.status == 403:
                raise CommentError(f"Comments disabled for video: {video_id}")
            raise handle_api_error(json.loads(e.content.decode()))


class YouTubeScrapingClient:
    """Web scraping client for YouTube content."""

    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.scraping.timeout),
            headers={'User-Agent': self.config.scraping.user_agents[0]}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def extract_video_metadata(self, video_id: str) -> VideoData:
        """Extract video metadata using yt-dlp."""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False
                )

            # Convert yt-dlp info to VideoData
            return self._convert_ytdlp_to_video_data(info)

        except Exception as e:
            raise ParsingError(f"Failed to extract video metadata: {e}", parser_type="yt-dlp")

    def _convert_ytdlp_to_video_data(self, info: Dict[str, Any]) -> VideoData:
        """Convert yt-dlp info dict to VideoData."""
        # Extract engagement metrics
        engagement = EngagementMetrics(
            view_count=info.get('view_count', 0),
            like_count=info.get('like_count', 0),
            comment_count=info.get('comment_count', 0)
        )

        # Extract thumbnails
        thumbnails = []
        for thumb in info.get('thumbnails', []):
            thumbnails.append(ThumbnailData(
                url=thumb['url'],
                width=thumb.get('width', 0),
                height=thumb.get('height', 0),
                size_name=thumb.get('id', 'unknown')
            ))

        # Parse upload date
        upload_date = info.get('upload_date')
        published_at = datetime.utcnow()
        if upload_date:
            try:
                published_at = datetime.strptime(upload_date, '%Y%m%d')
            except ValueError:
                pass

        # Create duration
        duration = timedelta(seconds=info.get('duration', 0))

        return VideoData(
            video_id=info['id'],
            title=info.get('title', ''),
            description=info.get('description', ''),
            channel_id=info.get('channel_id', ''),
            channel_title=info.get('uploader', ''),
            published_at=published_at,
            duration=duration,
            tags=info.get('tags', []),
            thumbnails=thumbnails,
            engagement_metrics=engagement
        )

    async def extract_transcript(self, video_id: str, language: str = 'en') -> TranscriptData:
        """Extract video transcript."""
        try:
            # Use yt-dlp to get transcript
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': [language],
                'skip_download': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False
                )

            # Extract transcript from subtitles
            subtitles = info.get('subtitles', {}) or info.get('automatic_captions', {})

            if language in subtitles:
                # Get the first subtitle format (usually VTT or SRT)
                subtitle_info = subtitles[language][0]

                async with self.session.get(subtitle_info['url']) as response:
                    subtitle_content = await response.text()

                # Parse VTT/SRT content to extract text
                text = self._parse_subtitle_content(subtitle_content)

                return TranscriptData(
                    text=text,
                    language=language,
                    auto_generated=language in info.get('automatic_captions', {}),
                    start_time=0.0,
                    duration=info.get('duration', 0)
                )
            else:
                raise TranscriptError(f"No transcript available for language: {language}")

        except Exception as e:
            raise TranscriptError(f"Failed to extract transcript: {e}", video_id=video_id)

    def _parse_subtitle_content(self, content: str) -> str:
        """Parse subtitle content to extract plain text."""
        # Simple parser for VTT/SRT content
        lines = content.split('\n')
        text_lines = []

        for line in lines:
            line = line.strip()
            # Skip timestamps and metadata
            if not line or '-->' in line or line.isdigit() or line.startswith('WEBVTT'):
                continue
            # Remove HTML tags
            import re
            line = re.sub(r'<[^>]+>', '', line)
            if line:
                text_lines.append(line)

        return ' '.join(text_lines)


class YouTubeClient:
    """Main YouTube crawler client."""

    def __init__(self, config: CrawlerConfig, db_manager=None):
        self.config = config
        self.db_manager = db_manager

        # Initialize API client if available
        self.api_client = None
        if config.api.api_key and config.crawl_mode != 'scraping_only':
            try:
                self.api_client = YouTubeAPIWrapper(config.api.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize API client: {e}")

        # Initialize scraping client
        self.scraping_client = YouTubeScrapingClient(config)

        # Initialize filtering engine
        source_config = YouTubeSourceConfig(
            api_key=config.api.api_key,
            quality_threshold=config.filtering.quality_threshold
        )
        self.filtering_engine = ChannelFilteringEngine(source_config)

        # Performance tracking
        self.stats = {
            'videos_crawled': 0,
            'channels_crawled': 0,
            'api_requests': 0,
            'scraping_requests': 0,
            'errors': 0,
            'start_time': time.time()
        }

    async def crawl_video(self, video_id: str, use_api: bool = True) -> CrawlResult:
        """Crawl a single video."""
        start_time = time.time()

        try:
            video_data = None
            source = "unknown"

            # Try API first if available and requested
            if use_api and self.api_client:
                try:
                    video_data = await self.api_client.get_video_details(video_id)
                    source = "api"
                    self.stats['api_requests'] += 1
                except Exception as e:
                    logger.warning(f"API failed for video {video_id}: {e}")
                    if not self.config.api.fallback_to_scraping:
                        raise

            # Fall back to scraping if API failed or not available
            if video_data is None:
                async with self.scraping_client as scraper:
                    video_data = await scraper.extract_video_metadata(video_id)
                    source = "scraping"
                    self.stats['scraping_requests'] += 1

            # Extract additional data if configured
            if self.config.extraction.extract_comments and self.api_client:
                try:
                    comments = await self.api_client.get_video_comments(
                        video_id, self.config.extraction.max_comments
                    )
                    video_data.comments = comments
                except Exception as e:
                    logger.warning(f"Failed to extract comments for {video_id}: {e}")

            if self.config.extraction.extract_transcripts:
                try:
                    async with self.scraping_client as scraper:
                        transcript = await scraper.extract_transcript(video_id)
                        video_data.transcript = transcript
                except Exception as e:
                    logger.warning(f"Failed to extract transcript for {video_id}: {e}")

            self.stats['videos_crawled'] += 1

            return CrawlResult(
                success=True,
                data=video_data,
                execution_time=time.time() - start_time,
                source=source
            )

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to crawl video {video_id}: {e}")

            return CrawlResult(
                success=False,
                data=None,
                error_message=str(e),
                error_code=getattr(e, 'error_code', 'UNKNOWN'),
                execution_time=time.time() - start_time
            )

    async def crawl_channel(self, channel_id: str, use_api: bool = True) -> CrawlResult:
        """Crawl a channel."""
        start_time = time.time()

        try:
            channel_data = None
            source = "unknown"

            # Try API first if available and requested
            if use_api and self.api_client:
                try:
                    channel_data = await self.api_client.get_channel_details(channel_id)
                    source = "api"
                    self.stats['api_requests'] += 1
                except Exception as e:
                    logger.warning(f"API failed for channel {channel_id}: {e}")
                    if not self.config.api.fallback_to_scraping:
                        raise

            # TODO: Implement channel scraping fallback
            if channel_data is None:
                raise YouTubeCrawlerError("Channel scraping not yet implemented")

            self.stats['channels_crawled'] += 1

            return CrawlResult(
                success=True,
                data=channel_data,
                execution_time=time.time() - start_time,
                source=source
            )

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to crawl channel {channel_id}: {e}")

            return CrawlResult(
                success=False,
                data=None,
                error_message=str(e),
                error_code=getattr(e, 'error_code', 'UNKNOWN'),
                execution_time=time.time() - start_time
            )

    async def batch_crawl_videos(self, video_ids: List[str]) -> ExtractResult:
        """Crawl multiple videos in batch."""
        start_time = time.time()
        results = []
        errors = []

        # Process in batches to respect rate limits
        batch_size = self.config.performance.batch_size

        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i + batch_size]

            # Process batch concurrently
            semaphore = asyncio.Semaphore(self.config.performance.concurrent_requests)

            async def crawl_with_semaphore(video_id):
                async with semaphore:
                    return await self.crawl_video(video_id)

            batch_results = await asyncio.gather(
                *[crawl_with_semaphore(vid) for vid in batch],
                return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                elif result.success:
                    results.append(result.data)
                else:
                    errors.append(result.error_message)

            # Rate limiting delay between batches
            if i + batch_size < len(video_ids):
                await asyncio.sleep(1.0)  # 1 second between batches

        return ExtractResult(
            extracted_count=len(results),
            failed_count=len(errors),
            items=results,
            errors=errors,
            execution_time=time.time() - start_time
        )

    async def search_and_crawl(
        self,
        query: str,
        max_results: int = 50,
        order: str = 'relevance'
    ) -> ExtractResult:
        """Search for videos and crawl them."""
        if not self.api_client:
            raise YouTubeCrawlerError("API client required for search functionality")

        try:
            # Search for video IDs
            video_ids = await self.api_client.search_videos(query, max_results, order)

            # Crawl the found videos
            return await self.batch_crawl_videos(video_ids)

        except Exception as e:
            logger.error(f"Search and crawl failed: {e}")
            return ExtractResult(
                extracted_count=0,
                failed_count=1,
                items=[],
                errors=[str(e)]
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        runtime = time.time() - self.stats['start_time']

        return {
            'runtime_seconds': runtime,
            'videos_crawled': self.stats['videos_crawled'],
            'channels_crawled': self.stats['channels_crawled'],
            'api_requests': self.stats['api_requests'],
            'scraping_requests': self.stats['scraping_requests'],
            'total_requests': self.stats['api_requests'] + self.stats['scraping_requests'],
            'errors': self.stats['errors'],
            'success_rate': (
                (self.stats['videos_crawled'] + self.stats['channels_crawled']) /
                max(self.stats['api_requests'] + self.stats['scraping_requests'], 1)
            ) * 100,
            'requests_per_minute': (
                (self.stats['api_requests'] + self.stats['scraping_requests']) /
                max(runtime / 60, 1)
            ),
            'api_quota_used': self.api_client.quota_used if self.api_client else 0
        }


# Factory functions

async def create_youtube_client(
    config: CrawlerConfig,
    db_manager=None
) -> YouTubeClient:
    """Create an enhanced YouTube client."""
    return YouTubeClient(config, db_manager)


async def create_basic_youtube_client(api_key: str) -> YouTubeClient:
    """Create a basic YouTube client with minimal configuration."""
    from ..config import CrawlerConfig, APIConfig

    config = CrawlerConfig()
    config.api = APIConfig(api_key=api_key)

    return YouTubeClient(config)


def create_sample_configuration() -> YouTubeSourceConfig:
    """Create a sample configuration."""
    return YouTubeSourceConfig(
        enable_api=True,
        enable_scraping=True,
        max_requests_per_minute=60,
        content_types=[ContentType.VIDEO, ContentType.CHANNEL],
        geographical_focus=GeographicalFocus.GLOBAL
    )


def load_source_configuration(filepath: str) -> YouTubeSourceConfig:
    """Load source configuration from file."""
    import json

    with open(filepath, 'r') as f:
        data = json.load(f)

    return YouTubeSourceConfig(**data)


# Alias for backward compatibility
EnhancedYouTubeClient = YouTubeClient
