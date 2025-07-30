"""
Metadata Parser Module
======================

Specialized parsers for YouTube metadata extraction and enrichment.
Handles content metadata, technical metadata, and data validation.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import re
import time
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

from .base_parser import BaseParser, ParseResult, ParseStatus, ContentType
from ..api.data_models import ValidationResult

logger = logging.getLogger(__name__)

# Optional dependencies for enhanced metadata processing
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class MetadataParser(BaseParser):
    """Main metadata parser for comprehensive metadata extraction."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content_parser = ContentMetadataParser(**kwargs)
        self.technical_parser = TechnicalMetadataParser(**kwargs)
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse metadata from various sources."""
        start_time = time.time()
        
        try:
            metadata = {
                'content_metadata': {},
                'technical_metadata': {},
                'extraction_metadata': {},
                'validation_results': None
            }
            
            # Parse content metadata
            if self.config.extract_content_metadata:
                content_result = await self.content_parser.parse(data, content_type)
                if content_result.is_successful():
                    metadata['content_metadata'] = content_result.data or {}
            
            # Parse technical metadata
            if self.config.extract_technical_metadata:
                technical_result = await self.technical_parser.parse(data, content_type)
                if technical_result.is_successful():
                    metadata['technical_metadata'] = technical_result.data or {}
            
            # Add extraction metadata
            metadata['extraction_metadata'] = {
                'extracted_at': datetime.utcnow().isoformat(),
                'parser_version': '1.0.0',
                'content_type': content_type.value if hasattr(content_type, 'value') else str(content_type),
                'data_source': self._determine_data_source(data)
            }
            
            # Validate metadata if requested
            if self.config.validate_metadata:
                validation_result = await self._validate_metadata(metadata)
                metadata['validation_results'] = validation_result
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=metadata,
                execution_time=execution_time,
                metadata={
                    'parser': 'MetadataParser',
                    'content_items': len(metadata['content_metadata']),
                    'technical_items': len(metadata['technical_metadata'])
                }
            )
            
        except Exception as e:
            logger.error(f"Metadata parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _determine_data_source(self, data: Any) -> str:
        """Determine the source type of the data."""
        if isinstance(data, str):
            if data.startswith('http'):
                return 'url'
            elif data.startswith('{') or data.startswith('['):
                return 'json_string'
            else:
                return 'text'
        elif isinstance(data, dict):
            if 'snippet' in data or 'statistics' in data:
                return 'youtube_api'
            else:
                return 'structured_data'
        elif isinstance(data, list):
            return 'array_data'
        else:
            return 'unknown'
    
    async def _validate_metadata(self, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate extracted metadata."""
        validation = ValidationResult(is_valid=True)
        
        # Check content metadata
        content_meta = metadata.get('content_metadata', {})
        if not content_meta:
            validation.add_warning("No content metadata extracted")
        
        # Check technical metadata
        technical_meta = metadata.get('technical_metadata', {})
        if not technical_meta:
            validation.add_warning("No technical metadata extracted")
        
        # Validate specific fields
        if 'title' in content_meta:
            if not content_meta['title'] or len(content_meta['title']) < 3:
                validation.add_error('title', 'Title is missing or too short')
        
        if 'description' in content_meta:
            if len(content_meta.get('description', '')) > 10000:
                validation.add_warning("Description is unusually long")
        
        if 'published_date' in content_meta:
            pub_date = content_meta['published_date']
            if isinstance(pub_date, str):
                try:
                    parsed_date = self._parse_datetime(pub_date)
                    if parsed_date and parsed_date > datetime.utcnow():
                        validation.add_error('published_date', 'Published date is in the future')
                except:
                    validation.add_error('published_date', 'Invalid date format')
        
        return validation


class ContentMetadataParser(BaseParser):
    """Parser for content-related metadata."""
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse content metadata."""
        start_time = time.time()
        
        try:
            content_metadata = {}
            
            if isinstance(data, dict):
                content_metadata = await self._parse_from_dict(data, content_type)
            elif isinstance(data, str):
                if data.startswith('http'):
                    content_metadata = await self._parse_from_url(data)
                else:
                    try:
                        parsed_data = json.loads(data)
                        content_metadata = await self._parse_from_dict(parsed_data, content_type)
                    except json.JSONDecodeError:
                        content_metadata = await self._parse_from_text(data)
            
            # Enrich metadata
            if self.config.enrich_metadata:
                content_metadata = await self._enrich_content_metadata(content_metadata)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=content_metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Content metadata parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _parse_from_dict(self, data: Dict[str, Any], content_type: ContentType) -> Dict[str, Any]:
        """Parse content metadata from dictionary."""
        metadata = {}
        
        # Handle YouTube API format
        if 'snippet' in data:
            snippet = data['snippet']
            
            metadata.update({
                'title': snippet.get('title'),
                'description': snippet.get('description'),
                'published_date': snippet.get('publishedAt'),
                'channel_id': snippet.get('channelId'),
                'channel_title': snippet.get('channelTitle'),
                'tags': snippet.get('tags', []),
                'category_id': snippet.get('categoryId'),
                'default_language': snippet.get('defaultLanguage'),
                'default_audio_language': snippet.get('defaultAudioLanguage'),
                'live_broadcast_content': snippet.get('liveBroadcastContent')
            })
            
            # Thumbnails
            if 'thumbnails' in snippet:
                metadata['thumbnails'] = list(snippet['thumbnails'].keys())
                metadata['thumbnail_count'] = len(snippet['thumbnails'])
            
            # Localized info
            if 'localized' in snippet:
                metadata['localized_title'] = snippet['localized'].get('title')
                metadata['localized_description'] = snippet['localized'].get('description')
        
        # Handle statistics
        if 'statistics' in data:
            stats = data['statistics']
            metadata.update({
                'view_count': self._safe_int(stats.get('viewCount')),
                'like_count': self._safe_int(stats.get('likeCount')),
                'dislike_count': self._safe_int(stats.get('dislikeCount')),
                'comment_count': self._safe_int(stats.get('commentCount')),
                'favorite_count': self._safe_int(stats.get('favoriteCount')),
                'subscriber_count': self._safe_int(stats.get('subscriberCount')),
                'video_count': self._safe_int(stats.get('videoCount')),
                'hidden_subscriber_count': stats.get('hiddenSubscriberCount', False)
            })
        
        # Handle content details
        if 'contentDetails' in data:
            details = data['contentDetails']
            metadata.update({
                'duration': details.get('duration'),
                'dimension': details.get('dimension'),
                'definition': details.get('definition'),
                'caption': details.get('caption'),
                'licensed_content': details.get('licensedContent'),
                'projection': details.get('projection'),
                'upload_status': details.get('uploadStatus'),
                'privacy_status': details.get('privacyStatus'),
                'license': details.get('license'),
                'embeddable': details.get('embeddable'),
                'public_stats_viewable': details.get('publicStatsViewable')
            })
        
        # Handle status
        if 'status' in data:
            status = data['status']
            metadata.update({
                'upload_status': status.get('uploadStatus'),
                'privacy_status': status.get('privacyStatus'),
                'license': status.get('license'),
                'embeddable': status.get('embeddable'),
                'public_stats_viewable': status.get('publicStatsViewable'),
                'made_for_kids': status.get('madeForKids'),
                'self_declared_made_for_kids': status.get('selfDeclaredMadeForKids')
            })
        
        # Handle topic details
        if 'topicDetails' in data:
            topic_details = data['topicDetails']
            metadata.update({
                'topic_ids': topic_details.get('topicIds', []),
                'relevant_topic_ids': topic_details.get('relevantTopicIds', []),
                'topic_categories': topic_details.get('topicCategories', [])
            })
        
        # Handle custom format
        if 'snippet' not in data:
            # Direct field mapping for custom formats
            field_mappings = {
                'title': ['title', 'name'],
                'description': ['description', 'desc', 'summary'],
                'published_date': ['published_at', 'created_at', 'upload_date', 'date'],
                'tags': ['tags', 'keywords', 'hashtags'],
                'category': ['category', 'category_name'],
                'language': ['language', 'lang'],
                'author': ['author', 'creator', 'uploader'],
                'duration': ['duration', 'length'],
                'url': ['url', 'link', 'video_url']
            }
            
            for meta_field, possible_keys in field_mappings.items():
                for key in possible_keys:
                    if key in data and data[key] is not None:
                        metadata[meta_field] = data[key]
                        break
        
        # Clean up None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return metadata
    
    async def _parse_from_url(self, url: str) -> Dict[str, Any]:
        """Parse content metadata from URL."""
        metadata = {'source_url': url}
        
        # Extract video ID from YouTube URLs
        video_id = self._extract_video_id_from_url(url)
        if video_id:
            metadata['video_id'] = video_id
            
        # Extract playlist ID
        playlist_id = self._extract_playlist_id_from_url(url)
        if playlist_id:
            metadata['playlist_id'] = playlist_id
        
        # Extract channel ID/username
        channel_info = self._extract_channel_info_from_url(url)
        if channel_info:
            metadata.update(channel_info)
        
        # Parse URL parameters
        parsed_url = urlparse(url)
        if parsed_url.query:
            params = parse_qs(parsed_url.query)
            metadata['url_parameters'] = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
        
        return metadata
    
    async def _parse_from_text(self, text: str) -> Dict[str, Any]:
        """Parse content metadata from plain text."""
        metadata = {}
        
        # Try to extract structured information from text
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in ['title', 'description', 'tags', 'category', 'author']:
                    metadata[key] = value
        
        # Extract URLs from text
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            metadata['extracted_urls'] = urls
        
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', text)
        if hashtags:
            metadata['hashtags'] = hashtags
        
        # Extract mentions
        mentions = re.findall(r'@(\w+)', text)
        if mentions:
            metadata['mentions'] = mentions
        
        return metadata
    
    async def _enrich_content_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich content metadata with additional information."""
        enriched = metadata.copy()
        
        # Parse and normalize dates
        for date_field in ['published_date', 'created_at', 'upload_date']:
            if date_field in enriched:
                parsed_date = self._parse_datetime(enriched[date_field])
                if parsed_date:
                    enriched[f'{date_field}_parsed'] = parsed_date.isoformat()
                    enriched[f'{date_field}_timestamp'] = parsed_date.timestamp()
                    enriched[f'{date_field}_age_days'] = (datetime.utcnow() - parsed_date).days
        
        # Analyze title
        if 'title' in enriched:
            title = enriched['title']
            enriched['title_length'] = len(title)
            enriched['title_word_count'] = len(title.split())
            enriched['title_has_caps'] = any(c.isupper() for c in title)
            enriched['title_has_numbers'] = any(c.isdigit() for c in title)
            enriched['title_has_special_chars'] = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', title))
        
        # Analyze description
        if 'description' in enriched:
            desc = enriched['description']
            enriched['description_length'] = len(desc)
            enriched['description_word_count'] = len(desc.split())
            enriched['description_line_count'] = len(desc.split('\n'))
            
            # Extract URLs from description
            urls_in_desc = re.findall(r'http[s]?://\S+', desc)
            enriched['description_url_count'] = len(urls_in_desc)
            if urls_in_desc:
                enriched['description_urls'] = urls_in_desc
        
        # Process tags
        if 'tags' in enriched:
            tags = enriched['tags']
            if isinstance(tags, list):
                enriched['tag_count'] = len(tags)
                enriched['avg_tag_length'] = sum(len(tag) for tag in tags) / len(tags) if tags else 0
                enriched['longest_tag'] = max(tags, key=len) if tags else None
        
        # Duration analysis
        if 'duration' in enriched:
            duration_str = enriched['duration']
            if isinstance(duration_str, str):
                duration_seconds = self._parse_duration_to_seconds(duration_str)
                if duration_seconds:
                    enriched['duration_seconds'] = duration_seconds
                    enriched['duration_minutes'] = duration_seconds / 60
                    enriched['duration_category'] = self._categorize_duration(duration_seconds)
        
        # Engagement analysis
        metrics = ['view_count', 'like_count', 'dislike_count', 'comment_count']
        metric_values = [enriched.get(m, 0) for m in metrics if enriched.get(m, 0) > 0]
        
        if len(metric_values) >= 2:
            view_count = enriched.get('view_count', 0)
            like_count = enriched.get('like_count', 0)
            dislike_count = enriched.get('dislike_count', 0)
            comment_count = enriched.get('comment_count', 0)
            
            if view_count > 0:
                enriched['like_ratio'] = like_count / view_count
                enriched['dislike_ratio'] = dislike_count / view_count
                enriched['comment_ratio'] = comment_count / view_count
                enriched['engagement_rate'] = (like_count + dislike_count + comment_count) / view_count
                
                if like_count + dislike_count > 0:
                    enriched['like_dislike_ratio'] = like_count / (like_count + dislike_count)
        
        return enriched
    
    def _extract_video_id_from_url(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _extract_playlist_id_from_url(self, url: str) -> Optional[str]:
        """Extract YouTube playlist ID from URL."""
        pattern = r'list=([a-zA-Z0-9_-]+)'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    
    def _extract_channel_info_from_url(self, url: str) -> Dict[str, str]:
        """Extract channel information from URL."""
        info = {}
        
        # Channel ID
        channel_id_match = re.search(r'channel/([a-zA-Z0-9_-]+)', url)
        if channel_id_match:
            info['channel_id'] = channel_id_match.group(1)
        
        # Username
        user_match = re.search(r'user/([a-zA-Z0-9_-]+)', url)
        if user_match:
            info['channel_username'] = user_match.group(1)
        
        # Custom URL
        c_match = re.search(r'/c/([a-zA-Z0-9_-]+)', url)
        if c_match:
            info['channel_custom_url'] = c_match.group(1)
        
        # Handle format
        handle_match = re.search(r'/@([a-zA-Z0-9_-]+)', url)
        if handle_match:
            info['channel_handle'] = handle_match.group(1)
        
        return info
    
    def _parse_duration_to_seconds(self, duration_str: str) -> Optional[int]:
        """Parse ISO 8601 duration to seconds."""
        # PT4M13S -> 253 seconds
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if not match:
            return None
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def _categorize_duration(self, seconds: int) -> str:
        """Categorize video duration."""
        if seconds < 60:
            return 'shorts'  # YouTube Shorts
        elif seconds < 300:  # 5 minutes
            return 'short'
        elif seconds < 1200:  # 20 minutes
            return 'medium'
        elif seconds < 3600:  # 1 hour
            return 'long'
        else:
            return 'very_long'
    
    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else 0
        except (ValueError, TypeError):
            return 0


class TechnicalMetadataParser(BaseParser):
    """Parser for technical metadata."""
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse technical metadata."""
        start_time = time.time()
        
        try:
            technical_metadata = {
                'parsing_info': {
                    'parser_version': '1.0.0',
                    'content_type': content_type.value if hasattr(content_type, 'value') else str(content_type),
                    'data_type': type(data).__name__,
                    'data_size': self._get_data_size(data),
                    'parsed_at': datetime.utcnow().isoformat()
                }
            }
            
            if isinstance(data, dict):
                technical_metadata.update(await self._parse_dict_technical(data))
            elif isinstance(data, str):
                technical_metadata.update(await self._parse_string_technical(data))
            
            # Add performance metrics
            technical_metadata['performance'] = {
                'parsing_duration': time.time() - start_time,
                'memory_efficient': len(str(data)) < 1000000,  # Less than 1MB
                'complex_structure': self._assess_structure_complexity(data)
            }
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=technical_metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Technical metadata parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _parse_dict_technical(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse technical metadata from dictionary."""
        metadata = {
            'structure': {
                'total_keys': len(data),
                'nested_levels': self._count_nested_levels(data),
                'has_lists': any(isinstance(v, list) for v in data.values()),
                'has_nested_dicts': any(isinstance(v, dict) for v in data.values()),
                'key_types': list(set(type(k).__name__ for k in data.keys())),
                'value_types': list(set(type(v).__name__ for v in data.values()))
            }
        }
        
        # API-specific metadata
        if 'kind' in data:
            metadata['api_info'] = {
                'kind': data['kind'],
                'api_version': self._extract_api_version(data.get('kind', '')),
                'resource_type': self._extract_resource_type(data.get('kind', ''))
            }
        
        if 'etag' in data:
            metadata['api_info'] = metadata.get('api_info', {})
            metadata['api_info']['etag'] = data['etag']
        
        # Content details technical info
        if 'contentDetails' in data:
            details = data['contentDetails']
            metadata['content_technical'] = {
                'has_captions': details.get('caption') == 'true',
                'definition': details.get('definition'),  # hd, sd
                'dimension': details.get('dimension'),    # 2d, 3d
                'projection': details.get('projection'),  # rectangular, 360
                'licensed_content': details.get('licensedContent', False)
            }
        
        # File format info (if available)
        if 'fileDetails' in data:
            file_details = data['fileDetails']
            metadata['file_info'] = {
                'file_name': file_details.get('fileName'),
                'file_size': file_details.get('fileSize'),
                'file_type': file_details.get('fileType'),
                'container': file_details.get('container'),
                'video_streams': file_details.get('videoStreams', []),
                'audio_streams': file_details.get('audioStreams', [])
            }
        
        return metadata
    
    async def _parse_string_technical(self, data: str) -> Dict[str, Any]:
        """Parse technical metadata from string."""
        metadata = {
            'string_info': {
                'length': len(data),
                'line_count': len(data.split('\n')),
                'word_count': len(data.split()),
                'character_encoding': 'utf-8',  # Assumed
                'has_unicode': any(ord(c) > 127 for c in data),
                'is_json': self._is_json_string(data),
                'is_xml': data.strip().startswith('<') and data.strip().endswith('>'),
                'is_url': data.startswith('http'),
                'compression_potential': self._estimate_compression_ratio(data)
            }
        }
        
        # URL-specific technical info
        if data.startswith('http'):
            parsed_url = urlparse(data)
            metadata['url_info'] = {
                'scheme': parsed_url.scheme,
                'domain': parsed_url.netloc,
                'path': parsed_url.path,
                'has_query': bool(parsed_url.query),
                'has_fragment': bool(parsed_url.fragment),
                'port': parsed_url.port,
                'path_segments': len(parsed_url.path.split('/')) - 1
            }
        
        return metadata
    
    def _get_data_size(self, data: Any) -> int:
        """Get approximate size of data in bytes."""
        try:
            return len(str(data).encode('utf-8'))
        except:
            return 0
    
    def _count_nested_levels(self, obj: Any, current_level: int = 0) -> int:
        """Count maximum nesting levels in a data structure."""
        max_level = current_level
        
        if isinstance(obj, dict):
            for value in obj.values():
                level = self._count_nested_levels(value, current_level + 1)
                max_level = max(max_level, level)
        elif isinstance(obj, list):
            for item in obj:
                level = self._count_nested_levels(item, current_level + 1)
                max_level = max(max_level, level)
        
        return max_level
    
    def _assess_structure_complexity(self, data: Any) -> str:
        """Assess the complexity of the data structure."""
        size = self._get_data_size(data)
        
        if isinstance(data, dict):
            nesting = self._count_nested_levels(data)
            key_count = len(data)
            
            if nesting > 5 or key_count > 100 or size > 10000:
                return 'high'
            elif nesting > 3 or key_count > 20 or size > 5000:
                return 'medium'
            else:
                return 'low'
        elif isinstance(data, str):
            if size > 10000:
                return 'high'
            elif size > 1000:
                return 'medium'
            else:
                return 'low'
        else:
            return 'low'
    
    def _extract_api_version(self, kind: str) -> Optional[str]:
        """Extract API version from kind field."""
        # youtube#video -> v3 (implied)
        if kind.startswith('youtube#'):
            return 'v3'
        return None
    
    def _extract_resource_type(self, kind: str) -> Optional[str]:
        """Extract resource type from kind field."""
        # youtube#video -> video
        if '#' in kind:
            return kind.split('#')[1]
        return None
    
    def _is_json_string(self, text: str) -> bool:
        """Check if string is valid JSON."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    
    def _estimate_compression_ratio(self, text: str) -> float:
        """Estimate potential compression ratio for text."""
        # Simple estimation based on character repetition
        unique_chars = len(set(text))
        total_chars = len(text)
        
        if total_chars == 0:
            return 0.0
        
        # Higher repetition = better compression potential
        repetition_factor = 1 - (unique_chars / total_chars)
        return min(repetition_factor * 0.8, 0.9)  # Cap at 90% compression


# Utility functions
def extract_all_metadata(data: Any, content_type: ContentType = None) -> Dict[str, Any]:
    """Extract all available metadata from data."""
    parser = MetadataParser()
    
    # Run synchronously for utility function
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(parser.parse(data, content_type or ContentType.METADATA))
        return result.data if result.is_successful() else {}
    finally:
        loop.close()


def compare_metadata(metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two metadata dictionaries."""
    comparison = {
        'identical': metadata1 == metadata2,
        'common_keys': list(set(metadata1.keys()) & set(metadata2.keys())),
        'unique_to_first': list(set(metadata1.keys()) - set(metadata2.keys())),
        'unique_to_second': list(set(metadata2.keys()) - set(metadata1.keys())),
        'value_differences': {}
    }
    
    # Check value differences for common keys
    for key in comparison['common_keys']:
        if metadata1[key] != metadata2[key]:
            comparison['value_differences'][key] = {
                'first': metadata1[key],
                'second': metadata2[key]
            }
    
    comparison['similarity_score'] = len(comparison['common_keys']) / max(len(metadata1), len(metadata2), 1)
    
    return comparison


def validate_youtube_metadata(metadata: Dict[str, Any]) -> ValidationResult:
    """Validate YouTube-specific metadata."""
    validation = ValidationResult(is_valid=True)
    
    # Required fields for videos
    if 'video_id' in metadata:
        video_id = metadata['video_id']
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            validation.add_error('video_id', 'Invalid YouTube video ID format')
    
    # Channel ID validation
    if 'channel_id' in metadata:
        channel_id = metadata['channel_id']
        if not re.match(r'^UC[a-zA-Z0-9_-]{22}$', channel_id):
            validation.add_error('channel_id', 'Invalid YouTube channel ID format')
    
    # Duration validation
    if 'duration' in metadata:
        duration = metadata['duration']
        if isinstance(duration, str) and not re.match(r'^PT(?:\d+H)?(?:\d+M)?(?:\d+S)?$', duration):
            validation.add_error('duration', 'Invalid ISO 8601 duration format')
    
    # View count validation
    if 'view_count' in metadata:
        view_count = metadata['view_count']
        if not isinstance(view_count, int) or view_count < 0:
            validation.add_error('view_count', 'View count must be a non-negative integer')
    
    # Published date validation
    if 'published_date' in metadata:
        pub_date = metadata['published_date']
        if isinstance(pub_date, str):
            try:
                parsed_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                if parsed_date > datetime.utcnow():
                    validation.add_error('published_date', 'Published date cannot be in the future')
            except ValueError:
                validation.add_error('published_date', 'Invalid date format')
    
    return validation


__all__ = [
    'MetadataParser',
    'ContentMetadataParser',
    'TechnicalMetadataParser',
    'extract_all_metadata',
    'compare_metadata',
    'validate_youtube_metadata'
]