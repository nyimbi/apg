"""
Channel Parser Module
=====================

Specialized parsers for YouTube channel data extraction and analysis.
Handles channel metadata, statistics, content analysis, and credibility assessment.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from .base_parser import BaseParser, ParseResult, ParseStatus, ContentType
from ..api.data_models import (
    ChannelData, ThumbnailData, EngagementMetrics, PerformanceMetrics,
    AudienceMetrics, PrivacyStatus, create_channel_from_api_response
)

logger = logging.getLogger(__name__)


class ChannelParser(BaseParser):
    """Main channel parser for comprehensive channel data extraction."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata_parser = ChannelMetadataParser(**kwargs)
        self.statistics_parser = ChannelStatisticsParser(**kwargs)
        self.content_parser = ChannelContentParser(**kwargs)
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse channel data from various sources."""
        start_time = time.time()
        
        try:
            if isinstance(data, str):
                # Handle URL or HTML content
                channel_data = await self._parse_from_html(data)
            elif isinstance(data, dict):
                # Handle API response or structured data
                if 'snippet' in data and 'statistics' in data:
                    # YouTube API v3 format
                    channel_data = create_channel_from_api_response(data)
                else:
                    # Custom or scraped data format
                    channel_data = await self._parse_from_dict(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Enhance with additional parsing
            if self.config.extract_metadata:
                metadata_result = await self.metadata_parser.parse(data, content_type)
                if metadata_result.is_successful():
                    self._merge_metadata(channel_data, metadata_result.data)
            
            if self.config.extract_statistics:
                stats_result = await self.statistics_parser.parse(data, content_type)
                if stats_result.is_successful():
                    self._merge_statistics(channel_data, stats_result.data)
            
            if self.config.extract_content:
                content_result = await self.content_parser.parse(data, content_type)
                if content_result.is_successful():
                    self._merge_content(channel_data, content_result.data)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=channel_data,
                execution_time=execution_time,
                metadata={'parser': 'ChannelParser', 'source': 'api' if isinstance(data, dict) else 'scraping'}
            )
            
        except Exception as e:
            logger.error(f"Channel parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _parse_from_html(self, html_content: str) -> ChannelData:
        """Parse channel data from HTML content."""
        # This would implement web scraping logic for channel pages
        # For now, return a basic structure
        channel_id = self._extract_channel_id_from_html(html_content)
        title = self._extract_title_from_html(html_content)
        description = self._extract_description_from_html(html_content)
        
        return ChannelData(
            channel_id=channel_id or "unknown",
            title=title or "Unknown Channel",
            description=description or "",
            published_at=datetime.utcnow(),
            crawled_at=datetime.utcnow()
        )
    
    async def _parse_from_dict(self, data: Dict[str, Any]) -> ChannelData:
        """Parse channel data from dictionary format."""
        return ChannelData(
            channel_id=data.get('id', data.get('channel_id', 'unknown')),
            title=data.get('title', data.get('name', '')),
            description=data.get('description', ''),
            custom_url=data.get('custom_url'),
            published_at=self._parse_datetime(data.get('published_at', data.get('created_at'))),
            country=data.get('country'),
            default_language=data.get('default_language'),
            subscriber_count=int(data.get('subscriber_count', 0)),
            video_count=int(data.get('video_count', 0)),
            view_count=int(data.get('view_count', 0)),
            crawled_at=datetime.utcnow()
        )
    
    def _extract_channel_id_from_html(self, html: str) -> Optional[str]:
        """Extract channel ID from HTML."""
        patterns = [
            r'"channelId":"([^"]+)"',
            r'"externalId":"([^"]+)"',
            r'/channel/([^/"]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1)
        return None
    
    def _extract_title_from_html(self, html: str) -> Optional[str]:
        """Extract channel title from HTML."""
        patterns = [
            r'"title":"([^"]+)"',
            r'<title>([^<]+)</title>',
            r'"channelTitle":"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1)
        return None
    
    def _extract_description_from_html(self, html: str) -> Optional[str]:
        """Extract channel description from HTML."""
        patterns = [
            r'"description":"([^"]+)"',
            r'"shortDescription":"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                return match.group(1)
        return None
    
    def _merge_metadata(self, channel_data: ChannelData, metadata: Dict[str, Any]):
        """Merge metadata into channel data."""
        if 'keywords' in metadata:
            channel_data.keywords = metadata['keywords']
        if 'topic_ids' in metadata:
            channel_data.topic_ids = metadata['topic_ids']
        if 'thumbnails' in metadata:
            channel_data.thumbnails = metadata['thumbnails']
    
    def _merge_statistics(self, channel_data: ChannelData, statistics: Dict[str, Any]):
        """Merge statistics into channel data."""
        if 'engagement_metrics' in statistics:
            channel_data.engagement_metrics = statistics['engagement_metrics']
        if 'performance_metrics' in statistics:
            channel_data.performance_metrics = statistics['performance_metrics']
        if 'audience_metrics' in statistics:
            channel_data.audience_metrics = statistics['audience_metrics']
    
    def _merge_content(self, channel_data: ChannelData, content: Dict[str, Any]):
        """Merge content data into channel data."""
        if 'recent_videos' in content:
            channel_data.recent_videos = content['recent_videos']
        if 'upload_frequency' in content:
            channel_data.upload_frequency = content['upload_frequency']


class ChannelMetadataParser(BaseParser):
    """Parser for channel metadata extraction."""
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse channel metadata."""
        start_time = time.time()
        
        try:
            metadata = {}
            
            if isinstance(data, dict):
                # Extract from API response
                snippet = data.get('snippet', {})
                branding = data.get('brandingSettings', {})
                
                # Keywords
                if 'keywords' in snippet:
                    metadata['keywords'] = snippet['keywords'].split() if snippet['keywords'] else []
                
                # Topic details
                if 'topicDetails' in data:
                    metadata['topic_ids'] = data['topicDetails'].get('topicIds', [])
                
                # Thumbnails
                thumbnails = []
                for size, thumb_data in snippet.get('thumbnails', {}).items():
                    thumbnails.append(ThumbnailData(
                        url=thumb_data['url'],
                        width=thumb_data['width'],
                        height=thumb_data['height'],
                        size_name=size
                    ))
                metadata['thumbnails'] = thumbnails
                
                # Branding settings
                if branding:
                    metadata['branding_settings'] = branding
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Channel metadata parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )


class ChannelStatisticsParser(BaseParser):
    """Parser for channel statistics and analytics."""
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse channel statistics."""
        start_time = time.time()
        
        try:
            statistics = {}
            
            if isinstance(data, dict):
                stats = data.get('statistics', {})
                
                # Create engagement metrics
                engagement_metrics = EngagementMetrics(
                    view_count=int(stats.get('viewCount', 0)),
                    comment_count=int(stats.get('commentCount', 0))
                )
                statistics['engagement_metrics'] = engagement_metrics
                
                # Calculate performance metrics
                performance_metrics = PerformanceMetrics()
                
                # Estimate views per day (simplified calculation)
                if 'publishedAt' in data.get('snippet', {}):
                    published_date = self._parse_datetime(data['snippet']['publishedAt'])
                    if published_date:
                        days_since_creation = (datetime.utcnow() - published_date).days
                        if days_since_creation > 0:
                            performance_metrics.views_per_day = engagement_metrics.view_count / days_since_creation
                
                statistics['performance_metrics'] = performance_metrics
                
                # Placeholder for audience metrics (would require YouTube Analytics API)
                audience_metrics = AudienceMetrics()
                statistics['audience_metrics'] = audience_metrics
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=statistics,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Channel statistics parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )


class ChannelContentParser(BaseParser):
    """Parser for channel content analysis."""
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse channel content information."""
        start_time = time.time()
        
        try:
            content = {}
            
            if isinstance(data, dict):
                # Extract recent videos (if provided)
                if 'recent_videos' in data:
                    content['recent_videos'] = data['recent_videos']
                
                # Calculate upload frequency
                if 'uploads' in data:
                    uploads = data['uploads']
                    if len(uploads) >= 2:
                        # Calculate frequency based on recent uploads
                        first_upload = self._parse_datetime(uploads[0].get('published_at'))
                        last_upload = self._parse_datetime(uploads[-1].get('published_at'))
                        
                        if first_upload and last_upload:
                            time_diff = (last_upload - first_upload).days
                            if time_diff > 0:
                                content['upload_frequency'] = (len(uploads) / time_diff) * 7  # videos per week
                
                # Content quality assessment
                content['content_quality_score'] = self._assess_content_quality(data)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=content,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Channel content parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _assess_content_quality(self, data: Dict[str, Any]) -> float:
        """Assess overall content quality."""
        score = 0.0
        factors = 0
        
        # Check if verified
        if data.get('snippet', {}).get('verified'):
            score += 0.2
            factors += 1
        
        # Check subscriber count
        subscriber_count = int(data.get('statistics', {}).get('subscriberCount', 0))
        if subscriber_count > 100000:
            score += 0.3
        elif subscriber_count > 10000:
            score += 0.2
        elif subscriber_count > 1000:
            score += 0.1
        factors += 1
        
        # Check view count
        view_count = int(data.get('statistics', {}).get('viewCount', 0))
        if view_count > 1000000:
            score += 0.3
        elif view_count > 100000:
            score += 0.2
        elif view_count > 10000:
            score += 0.1
        factors += 1
        
        # Check video count
        video_count = int(data.get('statistics', {}).get('videoCount', 0))
        if video_count > 100:
            score += 0.2
        elif video_count > 10:
            score += 0.1
        factors += 1
        
        return score / factors if factors > 0 else 0.0


# Utility functions
def extract_channel_id_from_url(url: str) -> Optional[str]:
    """Extract channel ID from YouTube URL."""
    patterns = [
        r'/channel/([a-zA-Z0-9_-]+)',
        r'/user/([a-zA-Z0-9_-]+)',
        r'/c/([a-zA-Z0-9_-]+)',
        r'/@([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def assess_channel_credibility(channel_data: ChannelData) -> float:
    """Assess channel credibility score."""
    score = 0.0
    factors = []
    
    # Verification status
    if hasattr(channel_data, 'is_verified') and channel_data.is_verified:
        score += 0.3
        factors.append('verified')
    
    # Subscriber count
    if channel_data.subscriber_count > 1000000:
        score += 0.25
        factors.append('high_subscribers')
    elif channel_data.subscriber_count > 100000:
        score += 0.15
        factors.append('medium_subscribers')
    elif channel_data.subscriber_count > 10000:
        score += 0.1
        factors.append('low_subscribers')
    
    # Video count and consistency
    if channel_data.video_count > 100:
        score += 0.15
        factors.append('prolific')
    elif channel_data.video_count > 10:
        score += 0.1
        factors.append('regular')
    
    # Upload frequency
    if channel_data.upload_frequency > 1:  # More than 1 video per week
        score += 0.15
        factors.append('active')
    elif channel_data.upload_frequency > 0.25:  # More than 1 video per month
        score += 0.1
        factors.append('somewhat_active')
    
    # Channel age
    if channel_data.published_at:
        age_days = (datetime.utcnow() - channel_data.published_at).days
        if age_days > 365:  # More than 1 year old
            score += 0.15
            factors.append('established')
        elif age_days > 180:  # More than 6 months old
            score += 0.1
            factors.append('mature')
    
    return min(score, 1.0)  # Cap at 1.0


__all__ = [
    'ChannelParser',
    'ChannelMetadataParser', 
    'ChannelStatisticsParser',
    'ChannelContentParser',
    'extract_channel_id_from_url',
    'assess_channel_credibility'
]