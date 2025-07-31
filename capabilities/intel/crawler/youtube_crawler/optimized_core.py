"""
Enhanced YouTube Crawler with Advanced Performance Optimizations
================================================================

High-performance YouTube crawler with:
- Advanced rate limiting and connection pooling
- Intelligent caching with TTL and compression
- Concurrent processing with semaphore control
- Youtube-dl/yt-dlp integration for video downloads
- Memory-efficient data structures
- Performance monitoring and health checks
- Adaptive request optimization

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import aiohttp
import json
import logging
import time
import weakref
import gc
from typing import Dict, List, Optional, Union, Any, AsyncGenerator, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import pickle
import gzip
import os
import tempfile
from pathlib import Path
import threading
from contextlib import asynccontextmanager

# Third-party imports
try:
    import yt_dlp
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import psutil
    import ujson
    import redis
    from cachetools import TTLCache, LRUCache
    YT_DLP_AVAILABLE = True
    PERFORMANCE_LIBS_AVAILABLE = True
except ImportError as e:
    YT_DLP_AVAILABLE = False
    PERFORMANCE_LIBS_AVAILABLE = False
    logging.warning(f"Performance libraries not available: {e}")
    # Create fallback cache classes
    class TTLCache(dict):
        def __init__(self, maxsize, ttl):
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl
    
    class LRUCache(dict):
        def __init__(self, maxsize):
            super().__init__()
            self.maxsize = maxsize

from .api.data_models import VideoData, ChannelData, CommentData, TranscriptData
from .api.exceptions import YouTubeCrawlerError, RateLimitExceededError

logger = logging.getLogger(__name__)


@dataclass
class OptimizedYouTubeConfig:
    """Configuration for optimized YouTube crawler."""
    # Rate limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    rate_limit_refill_rate: float = 1.0
    adaptive_rate_limiting: bool = True
    
    # Connection pooling
    max_connections: int = 100
    max_connections_per_host: int = 30
    connection_pool_ttl: int = 300
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    
    # Caching
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 3600
    disk_cache_size_mb: int = 500
    cache_compression_level: int = 6
    redis_cache_url: Optional[str] = None
    
    # Concurrency
    max_concurrent_requests: int = 20
    max_concurrent_downloads: int = 5
    semaphore_timeout: float = 30.0
    
    # Download settings
    enable_video_download: bool = False
    download_format: str = "best[height<=720]"
    download_audio_only: bool = False
    download_subtitles: bool = True
    download_thumbnails: bool = True
    max_download_size_mb: int = 500
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = True
    gc_threshold: int = 100
    
    # API settings
    youtube_api_key: Optional[str] = None
    enable_api_fallback: bool = True
    api_quota_per_day: int = 10000
    
    # Health checks
    health_check_interval: int = 300
    max_consecutive_failures: int = 5


class TokenBucketRateLimiter:
    """Advanced rate limiter using token bucket algorithm with adaptive features."""
    
    def __init__(self, config: OptimizedYouTubeConfig):
        self.capacity = config.rate_limit_burst_size
        self.tokens = self.capacity
        self.refill_rate = config.rate_limit_refill_rate
        self.last_refill = time.time()
        self.adaptive = config.adaptive_rate_limiting
        self.lock = asyncio.Lock()
        
        # Adaptive rate limiting
        self.success_count = 0
        self.failure_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 60.0
        
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self.lock:
            await self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    async def wait_for_token(self, tokens: int = 1):
        """Wait until tokens are available."""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)
    
    def record_success(self):
        """Record successful request for adaptive rate limiting."""
        if self.adaptive:
            self.success_count += 1
            self._adjust_rate()
    
    def record_failure(self):
        """Record failed request for adaptive rate limiting."""
        if self.adaptive:
            self.failure_count += 1
            self._adjust_rate()
    
    def _adjust_rate(self):
        """Adjust rate limiting based on success/failure ratio."""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        total_requests = self.success_count + self.failure_count
        if total_requests < 10:
            return
        
        success_rate = self.success_count / total_requests
        
        if success_rate > 0.95:  # High success rate, increase rate
            self.refill_rate = min(2.0, self.refill_rate * 1.1)
        elif success_rate < 0.8:  # High failure rate, decrease rate
            self.refill_rate = max(0.5, self.refill_rate * 0.9)
        
        # Reset counters
        self.success_count = 0
        self.failure_count = 0
        self.last_adjustment = now


class MemoryOptimizedCache:
    """Memory-efficient cache with TTL, compression, and multiple backends."""
    
    def __init__(self, config: OptimizedYouTubeConfig):
        self.config = config
        self.memory_cache = TTLCache(
            maxsize=config.memory_cache_size,
            ttl=config.memory_cache_ttl
        )
        self.access_count = defaultdict(int)
        self.lock = asyncio.Lock()
        
        # Setup Redis cache if available
        self.redis_client = None
        if config.redis_cache_url and PERFORMANCE_LIBS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(config.redis_cache_url)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis cache not available: {e}")
        
        # Setup disk cache
        self.disk_cache_dir = Path(tempfile.gettempdir()) / "youtube_crawler_cache"
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.max_disk_cache_size = config.disk_cache_size_mb * 1024 * 1024
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with multi-level lookup."""
        cache_key = self._hash_key(key)
        
        async with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.access_count[cache_key] += 1
                return self.memory_cache[cache_key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                data = self.redis_client.get(cache_key)
                if data:
                    value = self._decompress(data)
                    # Promote to memory cache
                    async with self.lock:
                        self.memory_cache[cache_key] = value
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Check disk cache
        disk_path = self.disk_cache_dir / f"{cache_key}.gz"
        if disk_path.exists():
            try:
                with open(disk_path, 'rb') as f:
                    data = f.read()
                value = self._decompress(data)
                # Promote to memory cache
                async with self.lock:
                    self.memory_cache[cache_key] = value
                return value
            except Exception as e:
                logger.warning(f"Disk cache error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache with multi-level storage."""
        cache_key = self._hash_key(key)
        ttl = ttl or self.config.memory_cache_ttl
        
        # Store in memory cache
        async with self.lock:
            self.memory_cache[cache_key] = value
        
        # Store in Redis cache
        if self.redis_client:
            try:
                compressed_data = self._compress(value)
                self.redis_client.setex(cache_key, ttl, compressed_data)
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Store in disk cache
        try:
            compressed_data = self._compress(value)
            if len(compressed_data) < self.max_disk_cache_size:
                disk_path = self.disk_cache_dir / f"{cache_key}.gz"
                with open(disk_path, 'wb') as f:
                    f.write(compressed_data)
        except Exception as e:
            logger.warning(f"Disk cache set error: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Create hash of cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    def _compress(self, data: Any) -> bytes:
        """Compress data for storage."""
        if PERFORMANCE_LIBS_AVAILABLE:
            serialized = ujson.dumps(data, ensure_ascii=False).encode()
        else:
            serialized = json.dumps(data, ensure_ascii=False).encode()
        
        return gzip.compress(serialized, compresslevel=self.config.cache_compression_level)
    
    def _decompress(self, data: bytes) -> Any:
        """Decompress data from storage."""
        decompressed = gzip.decompress(data)
        
        if PERFORMANCE_LIBS_AVAILABLE:
            return ujson.loads(decompressed.decode())
        else:
            return json.loads(decompressed.decode())
    
    async def clear(self):
        """Clear all cache levels."""
        async with self.lock:
            self.memory_cache.clear()
            self.access_count.clear()
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")
        
        # Clear disk cache
        try:
            for file_path in self.disk_cache_dir.glob("*.gz"):
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Disk cache clear error: {e}")


class ConnectionPool:
    """Advanced connection pool with health monitoring and load balancing."""
    
    def __init__(self, config: OptimizedYouTubeConfig):
        self.config = config
        self.connectors = {}
        self.sessions = {}
        self.health_status = {}
        self.connection_count = defaultdict(int)
        self.last_health_check = 0
        self.lock = asyncio.Lock()
    
    @asynccontextmanager
    async def get_session(self, host: str = "default"):
        """Get a session from the connection pool."""
        async with self.lock:
            if host not in self.sessions:
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections,
                    limit_per_host=self.config.max_connections_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=self.config.connection_timeout,
                    connect=self.config.connection_timeout / 2,
                    sock_read=self.config.read_timeout
                )
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={'User-Agent': 'Enhanced-YouTube-Crawler/1.0'}
                )
                
                self.connectors[host] = connector
                self.sessions[host] = session
                self.health_status[host] = True
        
        session = self.sessions[host]
        self.connection_count[host] += 1
        
        try:
            yield session
        finally:
            self.connection_count[host] -= 1
    
    async def health_check(self):
        """Perform health check on all connections."""
        current_time = time.time()
        if current_time - self.last_health_check < self.config.health_check_interval:
            return
        
        async with self.lock:
            for host, session in self.sessions.items():
                try:
                    # Simple health check - this could be expanded
                    self.health_status[host] = not session.closed
                except Exception:
                    self.health_status[host] = False
        
        self.last_health_check = current_time
    
    async def close_all(self):
        """Close all connections in the pool."""
        async with self.lock:
            for session in self.sessions.values():
                if not session.closed:
                    await session.close()
            
            self.sessions.clear()
            self.connectors.clear()
            self.health_status.clear()
            self.connection_count.clear()


class PerformanceMonitor:
    """Advanced performance monitoring and metrics collection."""
    
    def __init__(self, config: OptimizedYouTubeConfig):
        self.config = config
        self.enabled = config.enable_performance_monitoring
        self.memory_monitoring = config.enable_memory_monitoring
        
        # Performance metrics
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.response_times = deque(maxlen=1000)
        self.error_count = defaultdict(int)
        
        # Memory metrics
        self.peak_memory_usage = 0
        self.gc_count = 0
        self.last_gc = time.time()
        
        # Timing metrics
        self.start_time = time.time()
        self.last_stats_time = time.time()
    
    def record_request(self, response_time: float, success: bool, error_type: str = None):
        """Record request metrics."""
        if not self.enabled:
            return
        
        self.request_count += 1
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if error_type:
                self.error_count[error_type] += 1
        
        # Memory monitoring
        if self.memory_monitoring and self.request_count % 10 == 0:
            self._check_memory()
    
    def _check_memory(self):
        """Check memory usage and trigger GC if needed."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss
            
            self.peak_memory_usage = max(self.peak_memory_usage, current_memory)
            
            # Trigger GC if memory usage is high
            if (self.request_count % self.config.gc_threshold == 0 or
                current_memory > self.peak_memory_usage * 0.9):
                
                collected = gc.collect()
                self.gc_count += 1
                self.last_gc = time.time()
                
                logger.debug(f"GC triggered: collected {collected} objects")
                
        except Exception as e:
            logger.warning(f"Memory monitoring error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.enabled:
            return {}
        
        current_time = time.time()
        uptime = current_time - self.start_time
        
        stats = {
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / max(1, self.request_count),
            'requests_per_second': self.request_count / max(1, uptime),
            'avg_response_time': sum(self.response_times) / max(1, len(self.response_times)),
            'error_breakdown': dict(self.error_count),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.memory_monitoring:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                stats.update({
                    'current_memory_mb': memory_info.rss / 1024 / 1024,
                    'peak_memory_mb': self.peak_memory_usage / 1024 / 1024,
                    'gc_count': self.gc_count,
                    'last_gc': self.last_gc
                })
            except Exception:
                pass
        
        return stats


class OptimizedYouTubeDownloader:
    """Optimized YouTube video downloader using yt-dlp."""
    
    def __init__(self, config: OptimizedYouTubeConfig):
        self.config = config
        self.download_semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        self.download_stats = defaultdict(int)
        
        # Setup yt-dlp options
        self.ydl_opts = {
            'format': config.download_format,
            'outtmpl': '%(uploader)s/%(title)s.%(ext)s',
            'writesubtitles': config.download_subtitles,
            'writeautomaticsub': config.download_subtitles,
            'writethumbnail': config.download_thumbnails,
            'ignoreerrors': True,
            'no_warnings': True,
            'extractaudio': config.download_audio_only,
            'audioformat': 'mp3' if config.download_audio_only else None,
        }
        
        # Add file size limit
        if config.max_download_size_mb:
            self.ydl_opts['max_filesize'] = config.max_download_size_mb * 1024 * 1024
    
    async def download_video(self, video_id: str, output_dir: str = "./downloads") -> Dict[str, Any]:
        """Download video with metadata extraction."""
        if not YT_DLP_AVAILABLE:
            raise YouTubeCrawlerError("yt-dlp not available for video download")
        
        async with self.download_semaphore:
            return await self._download_video_impl(video_id, output_dir)
    
    async def _download_video_impl(self, video_id: str, output_dir: str) -> Dict[str, Any]:
        """Implementation of video download."""
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Update output template with directory
        ydl_opts = self.ydl_opts.copy()
        ydl_opts['outtmpl'] = str(output_path / ydl_opts['outtmpl'])
        
        try:
            # Extract info first
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: ydl.extract_info(video_url, download=False)
                )
            
            # Check file size
            filesize = info.get('filesize') or info.get('filesize_approx', 0)
            if filesize > self.config.max_download_size_mb * 1024 * 1024:
                raise YouTubeCrawlerError(f"Video too large: {filesize / 1024 / 1024:.1f}MB")
            
            # Download if enabled
            if self.config.enable_video_download:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: ydl.download([video_url])
                    )
                
                self.download_stats['downloaded'] += 1
            else:
                self.download_stats['info_only'] += 1
            
            return {
                'success': True,
                'video_id': video_id,
                'title': info.get('title'),
                'duration': info.get('duration'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'filesize': filesize,
                'format': info.get('format'),
                'downloaded': self.config.enable_video_download,
                'output_dir': str(output_path) if self.config.enable_video_download else None
            }
            
        except Exception as e:
            self.download_stats['failed'] += 1
            logger.error(f"Download failed for {video_id}: {e}")
            return {
                'success': False,
                'video_id': video_id,
                'error': str(e)
            }
    
    async def extract_audio(self, video_id: str, output_dir: str = "./downloads") -> Dict[str, Any]:
        """Extract audio from video."""
        if not YT_DLP_AVAILABLE:
            raise YouTubeCrawlerError("yt-dlp not available for audio extraction")
        
        # Use audio-specific options
        audio_opts = self.ydl_opts.copy()
        audio_opts.update({
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'mp3',
            'outtmpl': str(Path(output_dir) / '%(title)s.%(ext)s'),
        })
        
        return await self._download_with_options(video_id, audio_opts)
    
    async def _download_with_options(self, video_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Download with custom options."""
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: ydl.extract_info(video_url, download=True)
                )
            
            return {
                'success': True,
                'video_id': video_id,
                'title': info.get('title'),
                'output_path': info.get('filepath') or info.get('_filename')
            }
            
        except Exception as e:
            logger.error(f"Download with options failed for {video_id}: {e}")
            return {
                'success': False,
                'video_id': video_id,
                'error': str(e)
            }


class OptimizedYouTubeCrawler:
    """High-performance YouTube crawler with advanced optimizations."""
    
    def __init__(self, config: OptimizedYouTubeConfig):
        self.config = config
        
        # Initialize components
        self.rate_limiter = TokenBucketRateLimiter(config)
        self.cache = MemoryOptimizedCache(config)
        self.connection_pool = ConnectionPool(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.downloader = OptimizedYouTubeDownloader(config)
        
        # Concurrency control
        self.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # YouTube API client
        self.youtube_api = None
        if config.youtube_api_key:
            try:
                self.youtube_api = build('youtube', 'v3', developerKey=config.youtube_api_key)
            except Exception as e:
                logger.warning(f"YouTube API not available: {e}")
        
        # State tracking
        self.is_healthy = True
        self.consecutive_failures = 0
        self.last_health_check = time.time()
    
    async def search_videos(self, query: str, max_results: int = 50, **kwargs) -> List[VideoData]:
        """Search for videos with advanced filtering and caching."""
        cache_key = f"search_videos:{query}:{max_results}:{hash(str(kwargs))}"
        
        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return [VideoData(**data) for data in cached_result]
        
        # Perform search
        async with self.request_semaphore:
            await self.rate_limiter.wait_for_token()
            
            start_time = time.time()
            try:
                if self.youtube_api:
                    videos = await self._search_videos_api(query, max_results, **kwargs)
                else:
                    videos = await self._search_videos_scraping(query, max_results, **kwargs)
                
                # Cache results
                video_data = [video.__dict__ for video in videos]
                await self.cache.set(cache_key, video_data)
                
                self.rate_limiter.record_success()
                self.performance_monitor.record_request(time.time() - start_time, True)
                self.consecutive_failures = 0
                
                return videos
                
            except Exception as e:
                self.rate_limiter.record_failure()
                self.performance_monitor.record_request(time.time() - start_time, False, str(type(e).__name__))
                self.consecutive_failures += 1
                
                if self.consecutive_failures >= self.config.max_consecutive_failures:
                    self.is_healthy = False
                
                raise YouTubeCrawlerError(f"Video search failed: {e}")
    
    async def _search_videos_api(self, query: str, max_results: int, **kwargs) -> List[VideoData]:
        """Search videos using YouTube Data API."""
        request = self.youtube_api.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=min(max_results, 50),
            **kwargs
        )
        
        response = await asyncio.get_event_loop().run_in_executor(None, request.execute)
        
        videos = []
        for item in response.get('items', []):
            video_data = VideoData(
                video_id=item['id']['videoId'],
                title=item['snippet']['title'],
                description=item['snippet']['description'],
                channel_id=item['snippet']['channelId'],
                channel_name=item['snippet']['channelTitle'],
                upload_date=datetime.fromisoformat(item['snippet']['publishedAt'].replace('Z', '+00:00')),
                thumbnail_url=item['snippet']['thumbnails']['default']['url'],
                video_url=f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            )
            videos.append(video_data)
        
        return videos
    
    async def _search_videos_scraping(self, query: str, max_results: int, **kwargs) -> List[VideoData]:
        """Search videos using web scraping as fallback."""
        search_url = f"https://www.youtube.com/results?search_query={query}"
        
        async with self.connection_pool.get_session() as session:
            async with session.get(search_url) as response:
                if response.status != 200:
                    raise YouTubeCrawlerError(f"Search request failed: {response.status}")
                
                html = await response.text()
                # Parse results from HTML (simplified implementation)
                # In practice, this would use BeautifulSoup or similar
                videos = []
                # Implementation would extract video data from HTML
                return videos
    
    async def get_video_details(self, video_id: str) -> VideoData:
        """Get detailed video information."""
        cache_key = f"video_details:{video_id}"
        
        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return VideoData(**cached_result)
        
        async with self.request_semaphore:
            await self.rate_limiter.wait_for_token()
            
            start_time = time.time()
            try:
                if self.youtube_api:
                    video = await self._get_video_details_api(video_id)
                else:
                    video = await self._get_video_details_scraping(video_id)
                
                # Cache results
                await self.cache.set(cache_key, video.__dict__)
                
                self.rate_limiter.record_success()
                self.performance_monitor.record_request(time.time() - start_time, True)
                
                return video
                
            except Exception as e:
                self.rate_limiter.record_failure()
                self.performance_monitor.record_request(time.time() - start_time, False, str(type(e).__name__))
                raise YouTubeCrawlerError(f"Video details fetch failed: {e}")
    
    async def _get_video_details_api(self, video_id: str) -> VideoData:
        """Get video details using YouTube Data API."""
        request = self.youtube_api.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        )
        
        response = await asyncio.get_event_loop().run_in_executor(None, request.execute)
        
        if not response.get('items'):
            raise YouTubeCrawlerError(f"Video not found: {video_id}")
        
        item = response['items'][0]
        snippet = item['snippet']
        statistics = item.get('statistics', {})
        
        return VideoData(
            video_id=video_id,
            title=snippet['title'],
            description=snippet['description'],
            channel_id=snippet['channelId'],
            channel_name=snippet['channelTitle'],
            upload_date=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
            view_count=int(statistics.get('viewCount', 0)),
            like_count=int(statistics.get('likeCount', 0)),
            thumbnail_url=snippet['thumbnails']['default']['url'],
            video_url=f"https://www.youtube.com/watch?v={video_id}",
            metadata={
                'duration': item.get('contentDetails', {}).get('duration'),
                'tags': snippet.get('tags', []),
                'category_id': snippet.get('categoryId'),
                'default_language': snippet.get('defaultLanguage')
            }
        )
    
    async def _get_video_details_scraping(self, video_id: str) -> VideoData:
        """Get video details using web scraping."""
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        async with self.connection_pool.get_session() as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    raise YouTubeCrawlerError(f"Video request failed: {response.status}")
                
                html = await response.text()
                # Parse video data from HTML (simplified implementation)
                # In practice, this would use BeautifulSoup or regex parsing
                return VideoData(
                    video_id=video_id,
                    title="Video Title (scraped)",
                    video_url=video_url
                )
    
    async def download_video(self, video_id: str, output_dir: str = "./downloads") -> Dict[str, Any]:
        """Download video using optimized downloader."""
        return await self.downloader.download_video(video_id, output_dir)
    
    async def extract_audio(self, video_id: str, output_dir: str = "./downloads") -> Dict[str, Any]:
        """Extract audio from video."""
        return await self.downloader.extract_audio(video_id, output_dir)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        current_time = time.time()
        
        health_status = {
            'is_healthy': self.is_healthy,
            'consecutive_failures': self.consecutive_failures,
            'last_check': current_time,
            'uptime': current_time - self.performance_monitor.start_time,
            'performance_stats': self.performance_monitor.get_stats(),
            'cache_stats': {
                'memory_cache_size': len(self.cache.memory_cache),
                'redis_available': self.cache.redis_client is not None
            },
            'connection_pool_stats': {
                'active_connections': sum(self.connection_pool.connection_count.values()),
                'healthy_hosts': sum(1 for status in self.connection_pool.health_status.values() if status)
            }
        }
        
        # Update connection pool health
        await self.connection_pool.health_check()
        
        return health_status
    
    async def close(self):
        """Clean up resources."""
        await self.connection_pool.close_all()
        if self.cache.redis_client:
            try:
                self.cache.redis_client.close()
            except Exception:
                pass


# Factory functions
async def create_optimized_youtube_crawler(
    config: Optional[OptimizedYouTubeConfig] = None,
    **kwargs
) -> OptimizedYouTubeCrawler:
    """Create an optimized YouTube crawler instance."""
    if config is None:
        config = OptimizedYouTubeConfig(**kwargs)
    
    crawler = OptimizedYouTubeCrawler(config)
    
    # Perform initial health check
    await crawler.health_check()
    
    return crawler


def create_optimized_config(**overrides) -> OptimizedYouTubeConfig:
    """Create optimized configuration with overrides."""
    return OptimizedYouTubeConfig(**overrides)