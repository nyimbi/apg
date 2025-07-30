"""
Video Parser Implementation
===========================

Specialized parser for YouTube video content extraction.
Handles video metadata, statistics, and content analysis.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from .base_parser import BaseParser, ParseResult, ParseStatus, ContentType, MediaType
from .data_models import VideoData, DataQuality, ThumbnailData
from ..api.exceptions import ParsingError

logger = logging.getLogger(__name__)


class VideoParser(BaseParser):
    """Main video parser for YouTube video data."""

    @property
    def supported_content_types(self) -> List[ContentType]:
        return [ContentType.VIDEO]

    @property
    def supported_media_types(self) -> List[MediaType]:
        return [MediaType.JSON, MediaType.HTML, MediaType.XML]

    async def parse(self, content: Any, content_type: ContentType, **kwargs) -> ParseResult:
        """Parse video content."""
        try:
            if isinstance(content, dict):
                # API response format
                return await self._parse_api_response(content, **kwargs)
            elif isinstance(content, str):
                if content.strip().startswith('{'):
                    # JSON string
                    data = json.loads(content)
                    return await self._parse_api_response(data, **kwargs)
                elif '<html' in content.lower():
                    # HTML content
                    return await self._parse_html_content(content, **kwargs)
                else:
                    # Plain text or other format
                    return await self._parse_text_content(content, **kwargs)
            else:
                raise ParsingError(f"Unsupported content type: {type(content)}")

        except Exception as e:
            logger.error(f"Video parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                error_code="PARSING_ERROR"
            )

    async def _parse_api_response(self, data: Dict[str, Any], **kwargs) -> ParseResult:
        """Parse YouTube API response."""
        try:
            if 'items' in data and data['items']:
                # API search/list response
                video_data = data['items'][0]
            else:
                # Direct video data
                video_data = data

            video = self._extract_video_from_api(video_data)

            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=video,
                metadata={
                    'parser_type': 'api_response',
                    'source': 'youtube_api',
                    'video_id': video.video_id
                }
            )

        except Exception as e:
            logger.error(f"API response parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                error_code="API_PARSE_ERROR"
            )

    async def _parse_html_content(self, html: str, **kwargs) -> ParseResult:
        """Parse HTML content from YouTube video page."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Extract video ID from URL or page
            video_id = self._extract_video_id_from_html(soup, kwargs.get('url', ''))
            if not video_id:
                raise ParsingError("Could not extract video ID from HTML")

            # Extract video data from HTML
            video = self._extract_video_from_html(soup, video_id)

            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=video,
                metadata={
                    'parser_type': 'html_content',
                    'source': 'youtube_scraping',
                    'video_id': video.video_id
                }
            )

        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                error_code="HTML_PARSE_ERROR"
            )

    async def _parse_text_content(self, text: str, **kwargs) -> ParseResult:
        """Parse plain text content."""
        try:
            # Try to extract video ID from text (URL)
            video_id = self._extract_video_id_from_text(text)
            if not video_id:
                raise ParsingError("Could not extract video ID from text")

            # Create minimal video data
            video = VideoData(
                video_id=video_id,
                title="",
                description="",
                channel_id="",
                channel_title="",
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                source="text_extraction"
            )

            return ParseResult(
                status=ParseStatus.PARTIAL,
                data=video,
                metadata={
                    'parser_type': 'text_content',
                    'source': 'text_extraction',
                    'video_id': video.video_id
                }
            )

        except Exception as e:
            logger.error(f"Text parsing failed: {e}")
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                error_code="TEXT_PARSE_ERROR"
            )

    def _extract_video_from_api(self, data: Dict[str, Any]) -> VideoData:
        """Extract video data from API response."""
        snippet = data.get('snippet', {})
        statistics = data.get('statistics', {})
        content_details = data.get('contentDetails', {})
        status = data.get('status', {})

        # Extract basic information
        video_id = data.get('id', '')
        title = snippet.get('title', '')
        description = snippet.get('description', '')
        channel_id = snippet.get('channelId', '')
        channel_title = snippet.get('channelTitle', '')

        # Parse published date
        published_at = None
        if snippet.get('publishedAt'):
            try:
                published_at = datetime.fromisoformat(
                    snippet['publishedAt'].replace('Z', '+00:00')
                )
            except ValueError:
                pass

        # Parse duration
        duration = None
        if content_details.get('duration'):
            duration = self._parse_duration(content_details['duration'])

        # Extract statistics
        view_count = int(statistics.get('viewCount', 0))
        like_count = int(statistics.get('likeCount', 0))
        dislike_count = int(statistics.get('dislikeCount', 0))
        comment_count = int(statistics.get('commentCount', 0))

        # Extract thumbnails
        thumbnail_urls = {}
        for size, thumb_data in snippet.get('thumbnails', {}).items():
            thumbnail_urls[size] = thumb_data['url']

        # Determine video characteristics
        is_live = content_details.get('duration') == 'P0D'
        is_short = duration and duration.total_seconds() < 60
        is_age_restricted = content_details.get('contentRating', {}).get('ytRating') == 'ytAgeRestricted'

        return VideoData(
            video_id=video_id,
            title=title,
            description=description,
            channel_id=channel_id,
            channel_title=channel_title,
            published_at=published_at,
            duration=duration,
            category=snippet.get('categoryId'),
            language=snippet.get('defaultLanguage'),
            tags=snippet.get('tags', []),
            view_count=view_count,
            like_count=like_count,
            dislike_count=dislike_count,
            comment_count=comment_count,
            thumbnail_urls=thumbnail_urls,
            video_url=f"https://www.youtube.com/watch?v={video_id}",
            quality=self._assess_data_quality_api(data),
            is_live=is_live,
            is_short=is_short,
            is_age_restricted=is_age_restricted,
            availability=status.get('privacyStatus', 'public'),
            raw_data=data if self.config.include_raw_data else None,
            source="youtube_api"
        )

    def _extract_video_from_html(self, soup: BeautifulSoup, video_id: str) -> VideoData:
        """Extract video data from HTML content."""
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.text.replace(' - YouTube', '') if title_tag else ''

        # Extract description from meta tags
        description = ''
        desc_meta = soup.find('meta', {'name': 'description'})
        if desc_meta:
            description = desc_meta.get('content', '')

        # Extract channel information
        channel_id = ''
        channel_title = ''
        channel_link = soup.find('link', {'itemprop': 'url'})
        if channel_link:
            href = channel_link.get('href', '')
            if '/channel/' in href:
                channel_id = href.split('/channel/')[-1]

        # Look for channel name
        channel_name_tag = soup.find('span', {'itemprop': 'author'})
        if channel_name_tag:
            name_link = channel_name_tag.find('link', {'itemprop': 'name'})
            if name_link:
                channel_title = name_link.get('content', '')

        # Extract view count from meta tags
        view_count = 0
        view_meta = soup.find('meta', {'itemprop': 'interactionCount'})
        if view_meta:
            try:
                view_count = int(view_meta.get('content', '0'))
            except ValueError:
                pass

        # Extract duration
        duration = None
        duration_meta = soup.find('meta', {'itemprop': 'duration'})
        if duration_meta:
            duration_str = duration_meta.get('content', '')
            if duration_str:
                duration = self._parse_duration(duration_str)

        # Extract upload date
        published_at = None
        date_meta = soup.find('meta', {'itemprop': 'datePublished'})
        if date_meta:
            try:
                published_at = datetime.fromisoformat(
                    date_meta.get('content', '').replace('Z', '+00:00')
                )
            except ValueError:
                pass

        # Extract keywords/tags
        tags = []
        keywords_meta = soup.find('meta', {'name': 'keywords'})
        if keywords_meta:
            tags = [tag.strip() for tag in keywords_meta.get('content', '').split(',')]

        # Extract thumbnail URLs
        thumbnail_urls = {}
        og_image = soup.find('meta', {'property': 'og:image'})
        if og_image:
            thumbnail_urls['default'] = og_image.get('content', '')

        return VideoData(
            video_id=video_id,
            title=title,
            description=description,
            channel_id=channel_id,
            channel_title=channel_title,
            published_at=published_at,
            duration=duration,
            tags=tags,
            view_count=view_count,
            thumbnail_urls=thumbnail_urls,
            video_url=f"https://www.youtube.com/watch?v={video_id}",
            quality=self._assess_data_quality_html(soup),
            source="youtube_scraping"
        )

    def _extract_video_id_from_html(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract video ID from HTML or URL."""
        # Try to extract from URL first
        if url:
            video_id = self._extract_video_id_from_text(url)
            if video_id:
                return video_id

        # Look for video ID in meta tags
        canonical_link = soup.find('link', {'rel': 'canonical'})
        if canonical_link:
            canonical_url = canonical_link.get('href', '')
            video_id = self._extract_video_id_from_text(canonical_url)
            if video_id:
                return video_id

        # Look for video ID in page source
        page_text = soup.get_text()
        video_id_match = re.search(r'["\']videoId["\']\s*:\s*["\']([a-zA-Z0-9_-]{11})["\']', page_text)
        if video_id_match:
            return video_id_match.group(1)

        return None

    def _extract_video_id_from_text(self, text: str) -> Optional[str]:
        """Extract video ID from text/URL."""
        # YouTube video ID patterns
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'["\']videoId["\']\s*:\s*["\']([a-zA-Z0-9_-]{11})["\']',
            r'\b([a-zA-Z0-9_-]{11})\b'  # Generic 11-character pattern
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                video_id = match.group(1)
                # Validate that it looks like a YouTube video ID
                if len(video_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', video_id):
                    return video_id

        return None

    def _parse_duration(self, duration_str: str) -> Optional[timedelta]:
        """Parse ISO 8601 duration string."""
        try:
            # ISO 8601 duration format: PT4M13S
            pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
            match = re.match(pattern, duration_str)

            if not match:
                return None

            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)

            return timedelta(hours=hours, minutes=minutes, seconds=seconds)

        except Exception:
            return None

    def _assess_data_quality_api(self, data: Dict[str, Any]) -> DataQuality:
        """Assess data quality for API response."""
        score = 0.0

        # Check completeness
        snippet = data.get('snippet', {})
        statistics = data.get('statistics', {})

        if snippet.get('title'):
            score += 0.2
        if snippet.get('description'):
            score += 0.2
        if snippet.get('channelId'):
            score += 0.1
        if snippet.get('publishedAt'):
            score += 0.1
        if statistics.get('viewCount'):
            score += 0.2
        if data.get('contentDetails', {}).get('duration'):
            score += 0.1
        if snippet.get('thumbnails'):
            score += 0.1

        # Convert score to quality enum
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    def _assess_data_quality_html(self, soup: BeautifulSoup) -> DataQuality:
        """Assess data quality for HTML content."""
        score = 0.0

        # Check presence of key elements
        if soup.find('title'):
            score += 0.3
        if soup.find('meta', {'name': 'description'}):
            score += 0.2
        if soup.find('meta', {'itemprop': 'interactionCount'}):
            score += 0.2
        if soup.find('meta', {'itemprop': 'duration'}):
            score += 0.1
        if soup.find('meta', {'itemprop': 'datePublished'}):
            score += 0.1
        if soup.find('link', {'rel': 'canonical'}):
            score += 0.1

        # Convert score to quality enum
        if score >= 0.8:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR


class VideoMetadataParser(VideoParser):
    """Specialized parser for video metadata extraction."""

    async def parse(self, content: Any, content_type: ContentType, **kwargs) -> ParseResult:
        """Parse video metadata specifically."""
        result = await super().parse(content, content_type, **kwargs)

        if result.is_successful() and isinstance(result.data, VideoData):
            # Enhance metadata extraction
            video_data = result.data

            # Extract additional metadata
            if isinstance(content, dict):
                self._enhance_metadata_from_api(video_data, content)

            result.metadata['enhanced_metadata'] = True
            result.metadata['metadata_fields'] = self._count_metadata_fields(video_data)

        return result

    def _enhance_metadata_from_api(self, video_data: VideoData, api_data: Dict[str, Any]):
        """Enhance metadata from API response."""
        snippet = api_data.get('snippet', {})

        # Extract category information
        if snippet.get('categoryId'):
            video_data.category = self._get_category_name(snippet['categoryId'])

        # Extract localized info
        localized = snippet.get('localized', {})
        if localized:
            if not video_data.title:
                video_data.title = localized.get('title', '')
            if not video_data.description:
                video_data.description = localized.get('description', '')

    def _get_category_name(self, category_id: str) -> str:
        """Map category ID to category name."""
        categories = {
            '1': 'Film & Animation',
            '2': 'Autos & Vehicles',
            '10': 'Music',
            '15': 'Pets & Animals',
            '17': 'Sports',
            '19': 'Travel & Events',
            '20': 'Gaming',
            '22': 'People & Blogs',
            '23': 'Comedy',
            '24': 'Entertainment',
            '25': 'News & Politics',
            '26': 'Howto & Style',
            '27': 'Education',
            '28': 'Science & Technology'
        }
        return categories.get(category_id, f'Category {category_id}')

    def _count_metadata_fields(self, video_data: VideoData) -> int:
        """Count non-empty metadata fields."""
        count = 0
        fields = [
            video_data.video_id, video_data.title, video_data.description,
            video_data.channel_id, video_data.channel_title, video_data.published_at,
            video_data.duration, video_data.category, video_data.language
        ]

        for field in fields:
            if field:
                count += 1

        if video_data.tags:
            count += 1
        if video_data.thumbnail_urls:
            count += 1

        return count


class VideoStatisticsParser(VideoParser):
    """Specialized parser for video statistics."""

    async def parse(self, content: Any, content_type: ContentType, **kwargs) -> ParseResult:
        """Parse video statistics specifically."""
        result = await super().parse(content, content_type, **kwargs)

        if result.is_successful() and isinstance(result.data, VideoData):
            video_data = result.data

            # Calculate additional statistics
            self._calculate_engagement_metrics(video_data)

            result.metadata['statistics_enhanced'] = True
            result.metadata['engagement_rate'] = video_data.get_engagement_rate()

        return result

    def _calculate_engagement_metrics(self, video_data: VideoData):
        """Calculate additional engagement metrics."""
        # Calculate engagement rate
        if video_data.view_count > 0:
            total_engagement = (
                video_data.like_count +
                video_data.dislike_count +
                video_data.comment_count
            )
            engagement_rate = (total_engagement / video_data.view_count) * 100

            # Store in quality score for now
            video_data.quality_score = min(engagement_rate, 100.0)


class VideoContentParser(VideoParser):
    """Specialized parser for video content analysis."""

    async def parse(self, content: Any, content_type: ContentType, **kwargs) -> ParseResult:
        """Parse video content with analysis."""
        result = await super().parse(content, content_type, **kwargs)

        if result.is_successful() and isinstance(result.data, VideoData):
            video_data = result.data

            # Analyze content
            self._analyze_content(video_data)

            result.metadata['content_analyzed'] = True

        return result

    def _analyze_content(self, video_data: VideoData):
        """Analyze video content."""
        # Analyze title and description
        if video_data.title:
            video_data.tags.extend(self._extract_keywords_from_text(video_data.title))

        if video_data.description:
            video_data.tags.extend(self._extract_keywords_from_text(video_data.description))

        # Remove duplicates
        video_data.tags = list(set(video_data.tags))

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Filter common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'have', 'has', 'had', 'are', 'was', 'were', 'been', 'being'
        }

        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        # Return top 10 most common
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(10)]
