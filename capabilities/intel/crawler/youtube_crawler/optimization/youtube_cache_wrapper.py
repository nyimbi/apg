"""
YouTube Cache Wrapper Module
============================

Specialized cache wrapper for YouTube crawler that leverages the advanced
utils/caching system with YouTube-specific optimizations and cache keys.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ....utils.caching import CacheManager, CacheConfig, CacheStrategy

logger = logging.getLogger(__name__)


@dataclass
class YouTubeCacheConfig:
    """YouTube-specific cache configuration."""
    video_ttl: int = 3600  # 1 hour for video data
    channel_ttl: int = 7200  # 2 hours for channel data
    playlist_ttl: int = 1800  # 30 minutes for playlist data
    search_ttl: int = 900  # 15 minutes for search results
    comments_ttl: int = 600  # 10 minutes for comments
    transcripts_ttl: int = 86400  # 24 hours for transcripts
    thumbnails_ttl: int = 86400  # 24 hours for thumbnails
    metadata_ttl: int = 1800  # 30 minutes for metadata
    use_spatial_cache: bool = False  # Enable for geo-specific content
    compress_large_objects: bool = True
    encrypt_sensitive_data: bool = False


class YouTubeCacheWrapper:
    """
    YouTube-specific cache wrapper that leverages the advanced utils/caching system
    with intelligent cache key generation and content-type specific TTLs.
    """
    
    def __init__(self, cache_manager: CacheManager, config: Optional[YouTubeCacheConfig] = None):
        self.cache_manager = cache_manager
        self.config = config or YouTubeCacheConfig()
        self.key_prefix = "youtube_crawler"
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'cache_key_generations': 0
        }
    
    # Video caching methods
    async def get_video(self, video_id: str, include_stats: bool = True) -> Optional[Dict[str, Any]]:
        """Get cached video data."""
        cache_key = self._generate_video_key(video_id, include_stats)
        return await self._get_with_stats(cache_key)
    
    async def set_video(self, video_id: str, data: Dict[str, Any], include_stats: bool = True, 
                       custom_ttl: Optional[int] = None) -> bool:
        """Cache video data."""
        cache_key = self._generate_video_key(video_id, include_stats)
        ttl = custom_ttl or self.config.video_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Channel caching methods
    async def get_channel(self, channel_id: str, include_videos: bool = False) -> Optional[Dict[str, Any]]:
        """Get cached channel data."""
        cache_key = self._generate_channel_key(channel_id, include_videos)
        return await self._get_with_stats(cache_key)
    
    async def set_channel(self, channel_id: str, data: Dict[str, Any], include_videos: bool = False,
                         custom_ttl: Optional[int] = None) -> bool:
        """Cache channel data."""
        cache_key = self._generate_channel_key(channel_id, include_videos)
        ttl = custom_ttl or self.config.channel_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Playlist caching methods
    async def get_playlist(self, playlist_id: str, max_results: int = 50) -> Optional[Dict[str, Any]]:
        """Get cached playlist data."""
        cache_key = self._generate_playlist_key(playlist_id, max_results)
        return await self._get_with_stats(cache_key)
    
    async def set_playlist(self, playlist_id: str, data: Dict[str, Any], max_results: int = 50,
                          custom_ttl: Optional[int] = None) -> bool:
        """Cache playlist data."""
        cache_key = self._generate_playlist_key(playlist_id, max_results)
        ttl = custom_ttl or self.config.playlist_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Search caching methods
    async def get_search_results(self, query: str, search_type: str = 'video', 
                               max_results: int = 25, order: str = 'relevance') -> Optional[Dict[str, Any]]:
        """Get cached search results."""
        cache_key = self._generate_search_key(query, search_type, max_results, order)
        return await self._get_with_stats(cache_key)
    
    async def set_search_results(self, query: str, data: Dict[str, Any], search_type: str = 'video',
                               max_results: int = 25, order: str = 'relevance',
                               custom_ttl: Optional[int] = None) -> bool:
        """Cache search results."""
        cache_key = self._generate_search_key(query, search_type, max_results, order)
        ttl = custom_ttl or self.config.search_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Comments caching methods
    async def get_comments(self, video_id: str, max_results: int = 100, 
                          order: str = 'time') -> Optional[Dict[str, Any]]:
        """Get cached comments."""
        cache_key = self._generate_comments_key(video_id, max_results, order)
        return await self._get_with_stats(cache_key)
    
    async def set_comments(self, video_id: str, data: Dict[str, Any], max_results: int = 100,
                          order: str = 'time', custom_ttl: Optional[int] = None) -> bool:
        """Cache comments."""
        cache_key = self._generate_comments_key(video_id, max_results, order)
        ttl = custom_ttl or self.config.comments_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Transcript caching methods
    async def get_transcript(self, video_id: str, language: str = 'en') -> Optional[Dict[str, Any]]:
        """Get cached transcript."""
        cache_key = self._generate_transcript_key(video_id, language)
        return await self._get_with_stats(cache_key)
    
    async def set_transcript(self, video_id: str, data: Dict[str, Any], language: str = 'en',
                           custom_ttl: Optional[int] = None) -> bool:
        """Cache transcript."""
        cache_key = self._generate_transcript_key(video_id, language)
        ttl = custom_ttl or self.config.transcripts_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Thumbnail caching methods
    async def get_thumbnail(self, video_id: str, quality: str = 'default') -> Optional[Dict[str, Any]]:
        """Get cached thumbnail."""
        cache_key = self._generate_thumbnail_key(video_id, quality)
        return await self._get_with_stats(cache_key)
    
    async def set_thumbnail(self, video_id: str, data: Dict[str, Any], quality: str = 'default',
                           custom_ttl: Optional[int] = None) -> bool:
        """Cache thumbnail."""
        cache_key = self._generate_thumbnail_key(video_id, quality)
        ttl = custom_ttl or self.config.thumbnails_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Metadata caching methods
    async def get_metadata(self, resource_id: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata."""
        cache_key = self._generate_metadata_key(resource_id, resource_type)
        return await self._get_with_stats(cache_key)
    
    async def set_metadata(self, resource_id: str, data: Dict[str, Any], resource_type: str,
                          custom_ttl: Optional[int] = None) -> bool:
        """Cache metadata."""
        cache_key = self._generate_metadata_key(resource_id, resource_type)
        ttl = custom_ttl or self.config.metadata_ttl
        return await self._set_with_stats(cache_key, data, ttl)
    
    # Batch operations
    async def get_multiple_videos(self, video_ids: List[str], include_stats: bool = True) -> Dict[str, Any]:
        """Get multiple videos in batch."""
        results = {}
        cache_keys = [self._generate_video_key(vid, include_stats) for vid in video_ids]
        
        # Use cache manager's batch get if available
        if hasattr(self.cache_manager, 'get_batch'):
            cached_results = await self.cache_manager.get_batch(cache_keys)
            for i, video_id in enumerate(video_ids):
                if i < len(cached_results) and cached_results[i] is not None:
                    results[video_id] = cached_results[i]
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
        else:
            # Fallback to individual gets
            for video_id in video_ids:
                result = await self.get_video(video_id, include_stats)
                if result is not None:
                    results[video_id] = result
        
        return results
    
    async def set_multiple_videos(self, video_data: Dict[str, Dict[str, Any]], 
                                 include_stats: bool = True, custom_ttl: Optional[int] = None) -> Dict[str, bool]:
        """Set multiple videos in batch."""
        results = {}
        ttl = custom_ttl or self.config.video_ttl
        
        # Use cache manager's batch set if available
        if hasattr(self.cache_manager, 'set_batch'):
            cache_keys = []
            values = []
            for video_id, data in video_data.items():
                cache_keys.append(self._generate_video_key(video_id, include_stats))
                values.append(data)
            
            batch_results = await self.cache_manager.set_batch(cache_keys, values, ttl)
            for i, video_id in enumerate(video_data.keys()):
                if i < len(batch_results):
                    results[video_id] = batch_results[i]
                    if batch_results[i]:
                        self.stats['sets'] += 1
        else:
            # Fallback to individual sets
            for video_id, data in video_data.items():
                result = await self.set_video(video_id, data, include_stats, custom_ttl)
                results[video_id] = result
        
        return results
    
    # Cache invalidation methods
    async def invalidate_video(self, video_id: str) -> bool:
        """Invalidate all cached data for a video."""
        patterns = [
            self._generate_video_key(video_id, True),
            self._generate_video_key(video_id, False),
            self._generate_comments_key(video_id, '*', '*'),
            self._generate_transcript_key(video_id, '*'),
            self._generate_thumbnail_key(video_id, '*'),
            self._generate_metadata_key(video_id, 'video')
        ]
        
        results = []
        for pattern in patterns:
            if hasattr(self.cache_manager, 'delete_pattern'):
                result = await self.cache_manager.delete_pattern(pattern)
            else:
                result = await self.cache_manager.delete(pattern)
            results.append(result)
        
        return any(results)
    
    async def invalidate_channel(self, channel_id: str) -> bool:
        """Invalidate all cached data for a channel."""
        patterns = [
            self._generate_channel_key(channel_id, True),
            self._generate_channel_key(channel_id, False),
            self._generate_metadata_key(channel_id, 'channel')
        ]
        
        results = []
        for pattern in patterns:
            if hasattr(self.cache_manager, 'delete_pattern'):
                result = await self.cache_manager.delete_pattern(pattern)
            else:
                result = await self.cache_manager.delete(pattern)
            results.append(result)
        
        return any(results)
    
    async def invalidate_search(self, query: str) -> bool:
        """Invalidate all cached search results for a query."""
        # Generate pattern for all search variations of this query
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        pattern = f"{self.key_prefix}:search:{query_hash}:*"
        
        if hasattr(self.cache_manager, 'delete_pattern'):
            return await self.cache_manager.delete_pattern(pattern)
        else:
            # Fallback: try common search variations
            common_types = ['video', 'channel', 'playlist']
            common_orders = ['relevance', 'date', 'viewCount', 'rating']
            common_limits = [25, 50, 100]
            
            results = []
            for search_type in common_types:
                for order in common_orders:
                    for max_results in common_limits:
                        key = self._generate_search_key(query, search_type, max_results, order)
                        result = await self.cache_manager.delete(key)
                        results.append(result)
            
            return any(results)
    
    # Cache management methods
    async def clear_all(self) -> bool:
        """Clear all YouTube crawler cache entries."""
        if hasattr(self.cache_manager, 'delete_pattern'):
            return await self.cache_manager.delete_pattern(f"{self.key_prefix}:*")
        else:
            return await self.cache_manager.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = await self.cache_manager.get_stats() if hasattr(self.cache_manager, 'get_stats') else {}
        
        youtube_stats = self.stats.copy()
        total_requests = youtube_stats['hits'] + youtube_stats['misses']
        youtube_stats['hit_rate'] = (youtube_stats['hits'] / total_requests) if total_requests > 0 else 0.0
        youtube_stats['miss_rate'] = 1.0 - youtube_stats['hit_rate']
        
        return {
            'youtube_stats': youtube_stats,
            'cache_manager_stats': base_stats,
            'config': {
                'video_ttl': self.config.video_ttl,
                'channel_ttl': self.config.channel_ttl,
                'playlist_ttl': self.config.playlist_ttl,
                'search_ttl': self.config.search_ttl,
                'comments_ttl': self.config.comments_ttl,
                'transcripts_ttl': self.config.transcripts_ttl,
                'thumbnails_ttl': self.config.thumbnails_ttl,
                'metadata_ttl': self.config.metadata_ttl
            }
        }
    
    def reset_stats(self):
        """Reset YouTube cache statistics."""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'cache_key_generations': 0
        }
    
    # Cache key generation methods
    def _generate_video_key(self, video_id: str, include_stats: bool) -> str:
        """Generate cache key for video data."""
        self.stats['cache_key_generations'] += 1
        stats_suffix = "with_stats" if include_stats else "no_stats"
        return f"{self.key_prefix}:video:{video_id}:{stats_suffix}"
    
    def _generate_channel_key(self, channel_id: str, include_videos: bool) -> str:
        """Generate cache key for channel data."""
        self.stats['cache_key_generations'] += 1
        videos_suffix = "with_videos" if include_videos else "no_videos"
        return f"{self.key_prefix}:channel:{channel_id}:{videos_suffix}"
    
    def _generate_playlist_key(self, playlist_id: str, max_results: int) -> str:
        """Generate cache key for playlist data."""
        self.stats['cache_key_generations'] += 1
        return f"{self.key_prefix}:playlist:{playlist_id}:max{max_results}"
    
    def _generate_search_key(self, query: str, search_type: str, max_results: int, order: str) -> str:
        """Generate cache key for search results."""
        self.stats['cache_key_generations'] += 1
        # Use hash for query to handle special characters and length
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        return f"{self.key_prefix}:search:{query_hash}:{search_type}:{max_results}:{order}"
    
    def _generate_comments_key(self, video_id: str, max_results: int, order: str) -> str:
        """Generate cache key for comments."""
        self.stats['cache_key_generations'] += 1
        return f"{self.key_prefix}:comments:{video_id}:max{max_results}:{order}"
    
    def _generate_transcript_key(self, video_id: str, language: str) -> str:
        """Generate cache key for transcript."""
        self.stats['cache_key_generations'] += 1
        return f"{self.key_prefix}:transcript:{video_id}:{language}"
    
    def _generate_thumbnail_key(self, video_id: str, quality: str) -> str:
        """Generate cache key for thumbnail."""
        self.stats['cache_key_generations'] += 1
        return f"{self.key_prefix}:thumbnail:{video_id}:{quality}"
    
    def _generate_metadata_key(self, resource_id: str, resource_type: str) -> str:
        """Generate cache key for metadata."""
        self.stats['cache_key_generations'] += 1
        return f"{self.key_prefix}:metadata:{resource_type}:{resource_id}"
    
    # Helper methods
    async def _get_with_stats(self, cache_key: str) -> Optional[Any]:
        """Get from cache with statistics tracking."""
        try:
            result = await self.cache_manager.get(cache_key)
            if result is not None:
                self.stats['hits'] += 1
            else:
                self.stats['misses'] += 1
            return result
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            self.stats['misses'] += 1
            return None
    
    async def _set_with_stats(self, cache_key: str, value: Any, ttl: int) -> bool:
        """Set to cache with statistics tracking."""
        try:
            result = await self.cache_manager.set(cache_key, value, ttl)
            if result:
                self.stats['sets'] += 1
            return result
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
            return False
    
    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Perform any cleanup if needed
        pass


# Utility functions
async def create_youtube_cache(cache_config: Optional[CacheConfig] = None,
                             youtube_config: Optional[YouTubeCacheConfig] = None) -> YouTubeCacheWrapper:
    """Create a YouTube cache wrapper with optimized configuration."""
    if cache_config is None:
        cache_config = CacheConfig(
            strategy=CacheStrategy.LRU,
            max_size=10000,
            default_ttl=3600,
            enable_compression=True,
            enable_stats=True
        )
    
    cache_manager = await CacheManager.create(cache_config)
    return YouTubeCacheWrapper(cache_manager, youtube_config)


def create_cache_key_for_api_call(endpoint: str, params: Dict[str, Any]) -> str:
    """Create a standardized cache key for YouTube API calls."""
    # Sort parameters for consistent keys
    sorted_params = sorted(params.items())
    params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"youtube_api:{endpoint}:{params_hash}"


def estimate_cache_value_size(data: Any) -> int:
    """Estimate the size of data in bytes for cache management."""
    try:
        if isinstance(data, (str, bytes)):
            return len(data.encode() if isinstance(data, str) else data)
        elif isinstance(data, (dict, list)):
            return len(json.dumps(data, default=str).encode())
        else:
            return len(str(data).encode())
    except:
        return 0


__all__ = [
    'YouTubeCacheWrapper',
    'YouTubeCacheConfig',
    'create_youtube_cache',
    'create_cache_key_for_api_call',
    'estimate_cache_value_size'
]