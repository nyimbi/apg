"""
YouTube Crawler Data Models
============================

Comprehensive data models for YouTube content crawling and analysis.
Provides structured data classes for videos, channels, playlists, comments,
and associated metadata.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from enum import Enum
import json


class ContentStatus(Enum):
    """Content status enumeration."""
    ACTIVE = "active"
    DELETED = "deleted"
    PRIVATE = "private"
    UNLISTED = "unlisted"
    AGE_RESTRICTED = "age_restricted"
    REGION_BLOCKED = "region_blocked"
    COPYRIGHT_CLAIMED = "copyright_claimed"


class LiveStatus(Enum):
    """Live streaming status."""
    NONE = "none"
    LIVE = "live"
    UPCOMING = "upcoming"
    COMPLETED = "completed"


class PrivacyStatus(Enum):
    """Privacy status enumeration."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


@dataclass
class ThumbnailData:
    """Thumbnail information."""
    url: str
    width: int
    height: int
    size_name: str  # default, medium, high, standard, maxres
    file_size: Optional[int] = None
    format: str = "jpg"

    def get_aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class EngagementMetrics:
    """Video/Channel engagement metrics."""
    like_count: int = 0
    dislike_count: int = 0
    comment_count: int = 0
    favorite_count: int = 0
    view_count: int = 0
    share_count: int = 0

    def get_engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.view_count == 0:
            return 0.0

        total_engagement = (
            self.like_count +
            self.dislike_count +
            self.comment_count +
            self.share_count
        )
        return (total_engagement / self.view_count) * 100

    def get_like_ratio(self) -> float:
        """Calculate like to dislike ratio."""
        total_reactions = self.like_count + self.dislike_count
        if total_reactions == 0:
            return 0.0
        return (self.like_count / total_reactions) * 100


@dataclass
class PerformanceMetrics:
    """Performance analytics."""
    views_per_day: float = 0.0
    subscriber_growth_rate: float = 0.0
    average_view_duration: timedelta = field(default_factory=lambda: timedelta(0))
    audience_retention_rate: float = 0.0
    click_through_rate: float = 0.0
    estimated_revenue: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'views_per_day': self.views_per_day,
            'subscriber_growth_rate': self.subscriber_growth_rate,
            'average_view_duration_seconds': self.average_view_duration.total_seconds(),
            'audience_retention_rate': self.audience_retention_rate,
            'click_through_rate': self.click_through_rate,
            'estimated_revenue': self.estimated_revenue,
        }


@dataclass
class AudienceMetrics:
    """Audience demographics and behavior."""
    age_groups: Dict[str, float] = field(default_factory=dict)  # {"18-24": 25.5, "25-34": 35.2}
    gender_distribution: Dict[str, float] = field(default_factory=dict)  # {"male": 60.5, "female": 39.5}
    geographic_distribution: Dict[str, float] = field(default_factory=dict)  # {"US": 45.2, "UK": 12.3}
    traffic_sources: Dict[str, float] = field(default_factory=dict)  # {"youtube_search": 35.0, "suggested": 25.0}
    device_types: Dict[str, float] = field(default_factory=dict)  # {"mobile": 65.0, "desktop": 25.0, "tv": 10.0}
    watch_time_sources: Dict[str, float] = field(default_factory=dict)

    def get_primary_demographic(self) -> Dict[str, str]:
        """Get primary demographic segments."""
        return {
            'age_group': max(self.age_groups, key=self.age_groups.get) if self.age_groups else 'unknown',
            'gender': max(self.gender_distribution, key=self.gender_distribution.get) if self.gender_distribution else 'unknown',
            'geography': max(self.geographic_distribution, key=self.geographic_distribution.get) if self.geographic_distribution else 'unknown',
            'device': max(self.device_types, key=self.device_types.get) if self.device_types else 'unknown',
        }


@dataclass
class TranscriptData:
    """Video transcript information."""
    text: str
    language: str
    auto_generated: bool
    start_time: float = 0.0
    duration: float = 0.0
    confidence: Optional[float] = None

    def get_word_count(self) -> int:
        """Get word count of transcript."""
        return len(self.text.split())

    def get_duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.duration / 60.0


@dataclass
class CommentData:
    """YouTube comment information."""
    comment_id: str
    author_name: str
    author_channel_id: Optional[str]
    text: str
    like_count: int
    reply_count: int
    published_at: datetime
    updated_at: Optional[datetime]
    parent_id: Optional[str] = None  # For replies
    is_reply: bool = False
    author_channel_url: Optional[str] = None

    def is_recent(self, days: int = 7) -> bool:
        """Check if comment is recent."""
        return (datetime.utcnow() - self.published_at).days <= days

    def get_engagement_score(self) -> float:
        """Calculate comment engagement score."""
        # Simple scoring based on likes and replies
        return (self.like_count * 1.0) + (self.reply_count * 2.0)


@dataclass
class VideoData:
    """Comprehensive video data structure."""
    # Required basic information
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    published_at: datetime

    # Optional timestamps
    crawled_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    # Content details
    duration: timedelta = field(default_factory=lambda: timedelta(0))
    category_id: Optional[str] = None
    default_language: Optional[str] = None
    default_audio_language: Optional[str] = None

    # Status and privacy
    upload_status: str = "processed"
    privacy_status: PrivacyStatus = PrivacyStatus.PUBLIC
    content_status: ContentStatus = ContentStatus.ACTIVE
    live_streaming_details: Optional[Dict[str, Any]] = None

    # Media information
    thumbnails: List[ThumbnailData] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Metrics
    engagement_metrics: EngagementMetrics = field(default_factory=EngagementMetrics)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    audience_metrics: AudienceMetrics = field(default_factory=AudienceMetrics)

    # Content analysis
    transcript: Optional[TranscriptData] = None
    comments: List[CommentData] = field(default_factory=list)

    # Technical details
    definition: str = "hd"  # hd, sd
    dimension: str = "2d"   # 2d, 3d
    caption: bool = False
    licensed_content: bool = False

    # Additional metadata
    content_rating: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    topic_details: Dict[str, Any] = field(default_factory=dict)

    def get_duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.duration.total_seconds() / 60.0

    def get_duration_formatted(self) -> str:
        """Get formatted duration string."""
        total_seconds = int(self.duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def is_short_form(self) -> bool:
        """Check if video is short-form content (< 60 seconds)."""
        return self.duration.total_seconds() < 60

    def get_engagement_summary(self) -> Dict[str, Any]:
        """Get engagement summary."""
        return {
            'total_views': self.engagement_metrics.view_count,
            'engagement_rate': self.engagement_metrics.get_engagement_rate(),
            'like_ratio': self.engagement_metrics.get_like_ratio(),
            'comment_count': len(self.comments),
            'has_transcript': self.transcript is not None,
        }


@dataclass
class ChannelData:
    """Comprehensive channel data structure."""
    # Basic information
    channel_id: str
    title: str
    description: str
    published_at: datetime

    # Optional basic information
    custom_url: Optional[str] = None

    # Timestamps
    crawled_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    # Channel details
    country: Optional[str] = None
    default_language: Optional[str] = None

    # Statistics
    subscriber_count: int = 0
    video_count: int = 0
    view_count: int = 0

    # Media
    thumbnails: List[ThumbnailData] = field(default_factory=list)
    banner_external_url: Optional[str] = None

    # Content
    keywords: List[str] = field(default_factory=list)
    topic_ids: List[str] = field(default_factory=list)

    # Metrics
    engagement_metrics: EngagementMetrics = field(default_factory=EngagementMetrics)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    audience_metrics: AudienceMetrics = field(default_factory=AudienceMetrics)

    # Content analysis
    recent_videos: List[str] = field(default_factory=list)  # Video IDs
    upload_frequency: float = 0.0  # videos per week

    # Privacy and status
    privacy_status: PrivacyStatus = PrivacyStatus.PUBLIC
    is_family_safe: bool = True

    # Additional data
    branding_settings: Dict[str, Any] = field(default_factory=dict)
    content_owner_details: Dict[str, Any] = field(default_factory=dict)

    def get_subscriber_tier(self) -> str:
        """Get subscriber tier category."""
        if self.subscriber_count >= 1000000:
            return "mega"  # 1M+
        elif self.subscriber_count >= 100000:
            return "large"  # 100K+
        elif self.subscriber_count >= 10000:
            return "medium"  # 10K+
        elif self.subscriber_count >= 1000:
            return "small"  # 1K+
        else:
            return "micro"  # <1K

    def get_activity_level(self) -> str:
        """Assess channel activity level."""
        if self.upload_frequency >= 3:
            return "very_active"  # 3+ videos/week
        elif self.upload_frequency >= 1:
            return "active"  # 1+ videos/week
        elif self.upload_frequency >= 0.25:
            return "moderate"  # 1+ videos/month
        else:
            return "inactive"  # Less than monthly


@dataclass
class PlaylistData:
    """Playlist information."""
    playlist_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str

    # Timestamps
    published_at: datetime
    crawled_at: datetime = field(default_factory=datetime.utcnow)

    # Content
    video_count: int = 0
    video_ids: List[str] = field(default_factory=list)

    # Media
    thumbnails: List[ThumbnailData] = field(default_factory=list)

    # Privacy
    privacy_status: PrivacyStatus = PrivacyStatus.PUBLIC

    def get_total_duration(self, video_durations: Dict[str, timedelta]) -> timedelta:
        """Calculate total playlist duration."""
        total = timedelta(0)
        for video_id in self.video_ids:
            if video_id in video_durations:
                total += video_durations[video_id]
        return total


@dataclass
class CrawlResult:
    """Result of a crawling operation."""
    success: bool
    data: Optional[Union[VideoData, ChannelData, PlaylistData, List[CommentData]]]
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    source: str = "unknown"  # api, scraping, cache

    def is_successful(self) -> bool:
        """Check if crawl was successful."""
        return self.success and self.data is not None


@dataclass
class ExtractResult:
    """Result of data extraction operation."""
    extracted_count: int
    failed_count: int
    items: List[Union[VideoData, ChannelData, PlaylistData]]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0

    def get_success_rate(self) -> float:
        """Calculate extraction success rate."""
        total = self.extracted_count + self.failed_count
        return (self.extracted_count / total * 100) if total > 0 else 0.0


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_errors: Dict[str, List[str]] = field(default_factory=dict)

    def add_error(self, field: str, message: str) -> None:
        """Add validation error."""
        if field not in self.field_errors:
            self.field_errors[field] = []
        self.field_errors[field].append(message)
        self.errors.append(f"{field}: {message}")
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)


# Utility functions for data models

def create_video_from_api_response(api_data: Dict[str, Any]) -> VideoData:
    """Create VideoData from YouTube API response."""
    snippet = api_data.get('snippet', {})
    statistics = api_data.get('statistics', {})
    content_details = api_data.get('contentDetails', {})

    # Parse duration
    duration_str = content_details.get('duration', 'PT0S')
    duration = parse_duration(duration_str)

    # Create engagement metrics
    engagement = EngagementMetrics(
        view_count=int(statistics.get('viewCount', 0)),
        like_count=int(statistics.get('likeCount', 0)),
        dislike_count=int(statistics.get('dislikeCount', 0)),
        comment_count=int(statistics.get('commentCount', 0)),
        favorite_count=int(statistics.get('favoriteCount', 0)),
    )

    # Create thumbnails
    thumbnails = []
    for size, thumb_data in snippet.get('thumbnails', {}).items():
        thumbnails.append(ThumbnailData(
            url=thumb_data['url'],
            width=thumb_data['width'],
            height=thumb_data['height'],
            size_name=size
        ))

    return VideoData(
        video_id=api_data['id'],
        title=snippet.get('title', ''),
        description=snippet.get('description', ''),
        channel_id=snippet.get('channelId', ''),
        channel_title=snippet.get('channelTitle', ''),
        published_at=datetime.fromisoformat(snippet.get('publishedAt', '').replace('Z', '+00:00')),
        duration=duration,
        category_id=snippet.get('categoryId'),
        default_language=snippet.get('defaultLanguage'),
        tags=snippet.get('tags', []),
        thumbnails=thumbnails,
        engagement_metrics=engagement,
        definition=content_details.get('definition', 'hd'),
        dimension=content_details.get('dimension', '2d'),
        caption=content_details.get('caption') == 'true',
        licensed_content=content_details.get('licensedContent', False),
    )


def create_channel_from_api_response(api_data: Dict[str, Any]) -> ChannelData:
    """Create ChannelData from YouTube API response."""
    snippet = api_data.get('snippet', {})
    statistics = api_data.get('statistics', {})

    # Create thumbnails
    thumbnails = []
    for size, thumb_data in snippet.get('thumbnails', {}).items():
        thumbnails.append(ThumbnailData(
            url=thumb_data['url'],
            width=thumb_data['width'],
            height=thumb_data['height'],
            size_name=size
        ))

    return ChannelData(
        channel_id=api_data['id'],
        title=snippet.get('title', ''),
        description=snippet.get('description', ''),
        custom_url=snippet.get('customUrl'),
        published_at=datetime.fromisoformat(snippet.get('publishedAt', '').replace('Z', '+00:00')),
        country=snippet.get('country'),
        default_language=snippet.get('defaultLanguage'),
        subscriber_count=int(statistics.get('subscriberCount', 0)),
        video_count=int(statistics.get('videoCount', 0)),
        view_count=int(statistics.get('viewCount', 0)),
        thumbnails=thumbnails,
    )


def parse_duration(duration_str: str) -> timedelta:
    """Parse ISO 8601 duration string to timedelta."""
    # PT4M13S -> 4 minutes, 13 seconds
    import re

    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)

    if not match:
        return timedelta(0)

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)
