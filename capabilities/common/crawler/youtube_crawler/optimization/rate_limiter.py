"""
Rate Limiter Module
===================

Advanced rate limiting for YouTube API requests to respect quotas and avoid throttling.
Implements multiple rate limiting strategies including token bucket and sliding window.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    backoff_factor: float = 1.5
    max_backoff: float = 300.0
    enable_adaptive: bool = True
    track_by_endpoint: bool = True


@dataclass
class RateLimitStatus:
    """Current rate limiting status."""
    can_proceed: bool
    wait_time: float = 0.0
    remaining_quota: int = 0
    quota_reset_time: Optional[datetime] = None
    current_window_requests: int = 0
    next_available_time: Optional[datetime] = None


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    async def can_proceed(self, endpoint: str = None) -> RateLimitStatus:
        """Check if a request can proceed."""
        pass
    
    @abstractmethod
    async def record_request(self, endpoint: str = None, success: bool = True) -> None:
        """Record a completed request."""
        pass
    
    @abstractmethod
    async def wait_if_needed(self, endpoint: str = None) -> float:
        """Wait if rate limit is exceeded, return actual wait time."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset rate limiter state."""
        pass


class TokenBucketLimiter(RateLimiter):
    """
    Token bucket rate limiter implementation.
    Allows bursts up to bucket capacity, refills at steady rate.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = float(config.burst_limit)
        self.max_tokens = float(config.burst_limit)
        self.refill_rate = config.requests_per_minute / 60.0  # tokens per second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        
        # Endpoint-specific buckets
        self.endpoint_buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'tokens': float(config.burst_limit),
                'last_refill': time.time()
            }
        )
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'total_wait_time': 0.0,
            'requests_by_endpoint': defaultdict(int),
            'blocks_by_endpoint': defaultdict(int)
        }
    
    async def can_proceed(self, endpoint: str = None) -> RateLimitStatus:
        """Check if request can proceed based on token availability."""
        async with self.lock:
            await self._refill_tokens(endpoint)
            
            if self.config.track_by_endpoint and endpoint:
                bucket = self.endpoint_buckets[endpoint]
                tokens_available = bucket['tokens'] >= 1.0
                tokens = bucket['tokens']
            else:
                tokens_available = self.tokens >= 1.0
                tokens = self.tokens
            
            if tokens_available:
                return RateLimitStatus(
                    can_proceed=True,
                    remaining_quota=int(tokens),
                    current_window_requests=self.stats['total_requests']
                )
            else:
                # Calculate wait time until next token is available
                wait_time = 1.0 / self.refill_rate
                next_available = datetime.now() + timedelta(seconds=wait_time)
                
                return RateLimitStatus(
                    can_proceed=False,
                    wait_time=wait_time,
                    remaining_quota=0,
                    current_window_requests=self.stats['total_requests'],
                    next_available_time=next_available
                )
    
    async def record_request(self, endpoint: str = None, success: bool = True) -> None:
        """Record a request and consume a token."""
        async with self.lock:
            await self._refill_tokens(endpoint)
            
            if self.config.track_by_endpoint and endpoint:
                bucket = self.endpoint_buckets[endpoint]
                if bucket['tokens'] >= 1.0:
                    bucket['tokens'] -= 1.0
                    self.stats['requests_by_endpoint'][endpoint] += 1
                else:
                    self.stats['blocks_by_endpoint'][endpoint] += 1
                    self.stats['blocked_requests'] += 1
            else:
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                else:
                    self.stats['blocked_requests'] += 1
            
            self.stats['total_requests'] += 1
    
    async def wait_if_needed(self, endpoint: str = None) -> float:
        """Wait if rate limit exceeded, return actual wait time."""
        status = await self.can_proceed(endpoint)
        
        if not status.can_proceed:
            wait_time = status.wait_time
            logger.debug(f"Rate limit hit for {endpoint or 'global'}, waiting {wait_time:.2f}s")
            
            await asyncio.sleep(wait_time)
            self.stats['total_wait_time'] += wait_time
            
            return wait_time
        
        return 0.0
    
    async def _refill_tokens(self, endpoint: str = None):
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        
        if self.config.track_by_endpoint and endpoint:
            bucket = self.endpoint_buckets[endpoint]
            elapsed = current_time - bucket['last_refill']
            tokens_to_add = elapsed * self.refill_rate
            
            bucket['tokens'] = min(self.max_tokens, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = current_time
        else:
            elapsed = current_time - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        total_blocks = sum(self.stats['blocks_by_endpoint'].values()) or self.stats['blocked_requests']
        total_requests = self.stats['total_requests']
        
        return {
            'type': 'token_bucket',
            'config': {
                'requests_per_minute': self.config.requests_per_minute,
                'burst_limit': self.config.burst_limit,
                'refill_rate': self.refill_rate
            },
            'current_tokens': self.tokens,
            'max_tokens': self.max_tokens,
            'total_requests': total_requests,
            'blocked_requests': total_blocks,
            'block_rate': (total_blocks / total_requests) if total_requests > 0 else 0.0,
            'total_wait_time': self.stats['total_wait_time'],
            'avg_wait_time': (self.stats['total_wait_time'] / total_blocks) if total_blocks > 0 else 0.0,
            'endpoint_stats': dict(self.stats['requests_by_endpoint']),
            'endpoint_blocks': dict(self.stats['blocks_by_endpoint'])
        }
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        self.tokens = float(self.max_tokens)
        self.last_refill = time.time()
        self.endpoint_buckets.clear()
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'total_wait_time': 0.0,
            'requests_by_endpoint': defaultdict(int),
            'blocks_by_endpoint': defaultdict(int)
        }


class SlidingWindowLimiter(RateLimiter):
    """
    Sliding window rate limiter implementation.
    Tracks requests in time windows for more precise rate limiting.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.lock = asyncio.Lock()
        
        # Request timestamps for different time windows
        self.minute_requests: deque = deque()
        self.hour_requests: deque = deque()
        self.day_requests: deque = deque()
        
        # Endpoint-specific tracking
        self.endpoint_requests: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                'minute': deque(),
                'hour': deque(),
                'day': deque()
            }
        )
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'total_wait_time': 0.0,
            'requests_by_endpoint': defaultdict(int),
            'blocks_by_endpoint': defaultdict(int),
            'window_violations': {
                'minute': 0,
                'hour': 0,
                'day': 0
            }
        }
    
    async def can_proceed(self, endpoint: str = None) -> RateLimitStatus:
        """Check if request can proceed based on sliding windows."""
        async with self.lock:
            current_time = time.time()
            self._cleanup_old_requests(current_time, endpoint)
            
            # Check all time windows
            violations = []
            wait_times = []
            
            if self.config.track_by_endpoint and endpoint:
                endpoint_reqs = self.endpoint_requests[endpoint]
                minute_count = len(endpoint_reqs['minute'])
                hour_count = len(endpoint_reqs['hour'])
                day_count = len(endpoint_reqs['day'])
            else:
                minute_count = len(self.minute_requests)
                hour_count = len(self.hour_requests)
                day_count = len(self.day_requests)
            
            # Check minute limit
            if minute_count >= self.config.requests_per_minute:
                violations.append('minute')
                if self.config.track_by_endpoint and endpoint:
                    oldest = min(endpoint_reqs['minute']) if endpoint_reqs['minute'] else current_time
                else:
                    oldest = min(self.minute_requests) if self.minute_requests else current_time
                wait_times.append(60.0 - (current_time - oldest))
            
            # Check hour limit
            if hour_count >= self.config.requests_per_hour:
                violations.append('hour')
                if self.config.track_by_endpoint and endpoint:
                    oldest = min(endpoint_reqs['hour']) if endpoint_reqs['hour'] else current_time
                else:
                    oldest = min(self.hour_requests) if self.hour_requests else current_time
                wait_times.append(3600.0 - (current_time - oldest))
            
            # Check day limit
            if day_count >= self.config.requests_per_day:
                violations.append('day')
                if self.config.track_by_endpoint and endpoint:
                    oldest = min(endpoint_reqs['day']) if endpoint_reqs['day'] else current_time
                else:
                    oldest = min(self.day_requests) if self.day_requests else current_time
                wait_times.append(86400.0 - (current_time - oldest))
            
            if violations:
                max_wait = max(wait_times)
                next_available = datetime.now() + timedelta(seconds=max_wait)
                
                return RateLimitStatus(
                    can_proceed=False,
                    wait_time=max_wait,
                    remaining_quota=self.config.requests_per_minute - minute_count,
                    current_window_requests=minute_count,
                    next_available_time=next_available
                )
            else:
                return RateLimitStatus(
                    can_proceed=True,
                    remaining_quota=self.config.requests_per_minute - minute_count,
                    current_window_requests=minute_count
                )
    
    async def record_request(self, endpoint: str = None, success: bool = True) -> None:
        """Record a request timestamp."""
        async with self.lock:
            current_time = time.time()
            
            # Add to global windows
            self.minute_requests.append(current_time)
            self.hour_requests.append(current_time)
            self.day_requests.append(current_time)
            
            # Add to endpoint-specific windows
            if self.config.track_by_endpoint and endpoint:
                endpoint_reqs = self.endpoint_requests[endpoint]
                endpoint_reqs['minute'].append(current_time)
                endpoint_reqs['hour'].append(current_time)
                endpoint_reqs['day'].append(current_time)
                self.stats['requests_by_endpoint'][endpoint] += 1
            
            self.stats['total_requests'] += 1
            
            # Cleanup old requests
            self._cleanup_old_requests(current_time, endpoint)
    
    async def wait_if_needed(self, endpoint: str = None) -> float:
        """Wait if rate limit exceeded."""
        status = await self.can_proceed(endpoint)
        
        if not status.can_proceed:
            wait_time = status.wait_time
            logger.debug(f"Rate limit hit for {endpoint or 'global'}, waiting {wait_time:.2f}s")
            
            await asyncio.sleep(wait_time)
            self.stats['total_wait_time'] += wait_time
            
            if endpoint:
                self.stats['blocks_by_endpoint'][endpoint] += 1
            else:
                self.stats['blocked_requests'] += 1
            
            return wait_time
        
        return 0.0
    
    def _cleanup_old_requests(self, current_time: float, endpoint: str = None):
        """Remove old requests outside time windows."""
        # Cleanup global windows
        self._cleanup_deque(self.minute_requests, current_time - 60)
        self._cleanup_deque(self.hour_requests, current_time - 3600)
        self._cleanup_deque(self.day_requests, current_time - 86400)
        
        # Cleanup endpoint-specific windows
        if self.config.track_by_endpoint and endpoint:
            endpoint_reqs = self.endpoint_requests[endpoint]
            self._cleanup_deque(endpoint_reqs['minute'], current_time - 60)
            self._cleanup_deque(endpoint_reqs['hour'], current_time - 3600)
            self._cleanup_deque(endpoint_reqs['day'], current_time - 86400)
    
    def _cleanup_deque(self, request_deque: deque, cutoff_time: float):
        """Remove timestamps older than cutoff from deque."""
        while request_deque and request_deque[0] < cutoff_time:
            request_deque.popleft()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        self._cleanup_old_requests(current_time)
        
        return {
            'type': 'sliding_window',
            'config': {
                'requests_per_minute': self.config.requests_per_minute,
                'requests_per_hour': self.config.requests_per_hour,
                'requests_per_day': self.config.requests_per_day
            },
            'current_windows': {
                'minute': len(self.minute_requests),
                'hour': len(self.hour_requests),
                'day': len(self.day_requests)
            },
            'total_requests': self.stats['total_requests'],
            'blocked_requests': self.stats['blocked_requests'],
            'total_wait_time': self.stats['total_wait_time'],
            'endpoint_stats': dict(self.stats['requests_by_endpoint']),
            'endpoint_blocks': dict(self.stats['blocks_by_endpoint']),
            'window_violations': dict(self.stats['window_violations'])
        }
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        self.minute_requests.clear()
        self.hour_requests.clear()
        self.day_requests.clear()
        self.endpoint_requests.clear()
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'total_wait_time': 0.0,
            'requests_by_endpoint': defaultdict(int),
            'blocks_by_endpoint': defaultdict(int),
            'window_violations': {
                'minute': 0,
                'hour': 0,
                'day': 0
            }
        }


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on API responses.
    Implements exponential backoff for 429 errors and adjusts rates dynamically.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.base_limiter = SlidingWindowLimiter(config)
        self.lock = asyncio.Lock()
        
        # Adaptive parameters
        self.current_rate_factor = 1.0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.last_429_time = 0.0
        self.backoff_until = 0.0
        
        # Response tracking
        self.response_history: deque = deque(maxlen=100)
        self.error_rates = defaultdict(float)
        
        # Statistics
        self.stats = {
            'rate_adjustments': 0,
            'backoff_events': 0,
            'recovery_events': 0,
            'current_rate_factor': 1.0,
            'avg_response_time': 0.0
        }
    
    async def can_proceed(self, endpoint: str = None) -> RateLimitStatus:
        """Check if request can proceed with adaptive adjustments."""
        async with self.lock:
            current_time = time.time()
            
            # Check if we're in backoff period
            if current_time < self.backoff_until:
                wait_time = self.backoff_until - current_time
                next_available = datetime.fromtimestamp(self.backoff_until)
                
                return RateLimitStatus(
                    can_proceed=False,
                    wait_time=wait_time,
                    remaining_quota=0,
                    next_available_time=next_available
                )
            
            # Get base status and adjust with current rate factor
            base_status = await self.base_limiter.can_proceed(endpoint)
            
            if not base_status.can_proceed:
                # Apply rate factor to wait time
                adjusted_wait = base_status.wait_time * self.current_rate_factor
                next_available = datetime.now() + timedelta(seconds=adjusted_wait)
                
                return RateLimitStatus(
                    can_proceed=False,
                    wait_time=adjusted_wait,
                    remaining_quota=base_status.remaining_quota,
                    current_window_requests=base_status.current_window_requests,
                    next_available_time=next_available
                )
            
            return base_status
    
    async def record_request(self, endpoint: str = None, success: bool = True) -> None:
        """Record request and adapt rate based on success/failure."""
        async with self.lock:
            await self.base_limiter.record_request(endpoint, success)
            
            current_time = time.time()
            self.response_history.append({
                'time': current_time,
                'success': success,
                'endpoint': endpoint
            })
            
            if success:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
                
                # Gradually increase rate if we're having consistent success
                if self.consecutive_successes >= 10 and self.current_rate_factor > 0.5:
                    self.current_rate_factor = max(0.5, self.current_rate_factor * 0.95)
                    self.consecutive_successes = 0
                    self.stats['rate_adjustments'] += 1
                    self.stats['recovery_events'] += 1
                    logger.debug(f"Rate limit factor improved to {self.current_rate_factor:.2f}")
            
            else:
                self.consecutive_failures += 1
                self.consecutive_successes = 0
                
                # Implement exponential backoff for failures
                if self.consecutive_failures >= 3:
                    backoff_time = min(
                        self.config.backoff_factor ** self.consecutive_failures,
                        self.config.max_backoff
                    )
                    
                    self.backoff_until = current_time + backoff_time
                    self.current_rate_factor = min(3.0, self.current_rate_factor * 1.5)
                    self.stats['backoff_events'] += 1
                    self.stats['rate_adjustments'] += 1
                    
                    logger.warning(f"Rate limit backoff activated for {backoff_time:.2f}s, "
                                 f"factor increased to {self.current_rate_factor:.2f}")
            
            self.stats['current_rate_factor'] = self.current_rate_factor
            self._update_error_rates(endpoint, success)
    
    async def wait_if_needed(self, endpoint: str = None) -> float:
        """Wait with adaptive adjustments."""
        status = await self.can_proceed(endpoint)
        
        if not status.can_proceed:
            wait_time = status.wait_time
            logger.debug(f"Adaptive rate limit hit for {endpoint or 'global'}, "
                        f"waiting {wait_time:.2f}s (factor: {self.current_rate_factor:.2f})")
            
            await asyncio.sleep(wait_time)
            return wait_time
        
        return 0.0
    
    def record_429_error(self, endpoint: str = None, retry_after: Optional[int] = None):
        """Record a 429 (Too Many Requests) error for immediate backoff."""
        current_time = time.time()
        self.last_429_time = current_time
        
        # Use retry-after header if provided, otherwise use exponential backoff
        if retry_after:
            backoff_time = retry_after
        else:
            backoff_time = min(
                self.config.backoff_factor ** (self.consecutive_failures + 1),
                self.config.max_backoff
            )
        
        self.backoff_until = current_time + backoff_time
        self.current_rate_factor = min(5.0, self.current_rate_factor * 2.0)
        self.consecutive_failures += 1
        
        self.stats['backoff_events'] += 1
        self.stats['rate_adjustments'] += 1
        
        logger.warning(f"429 error received for {endpoint or 'global'}, "
                      f"backing off for {backoff_time:.2f}s")
    
    def _update_error_rates(self, endpoint: str, success: bool):
        """Update error rates for endpoints."""
        if endpoint:
            # Calculate recent error rate (last 20 requests)
            recent_requests = [r for r in self.response_history 
                              if r['endpoint'] == endpoint][-20:]
            
            if recent_requests:
                error_count = sum(1 for r in recent_requests if not r['success'])
                self.error_rates[endpoint] = error_count / len(recent_requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive rate limiter statistics."""
        base_stats = self.base_limiter.get_stats()
        
        # Calculate average response time from recent history
        recent_responses = list(self.response_history)[-50:]
        if len(recent_responses) > 1:
            time_diffs = []
            for i in range(1, len(recent_responses)):
                time_diffs.append(recent_responses[i]['time'] - recent_responses[i-1]['time'])
            self.stats['avg_response_time'] = sum(time_diffs) / len(time_diffs)
        
        adaptive_stats = {
            'type': 'adaptive',
            'base_limiter': base_stats,
            'adaptive_params': {
                'current_rate_factor': self.current_rate_factor,
                'consecutive_successes': self.consecutive_successes,
                'consecutive_failures': self.consecutive_failures,
                'is_in_backoff': time.time() < self.backoff_until,
                'backoff_remaining': max(0, self.backoff_until - time.time())
            },
            'error_rates': dict(self.error_rates),
            'adaptation_stats': self.stats.copy()
        }
        
        return adaptive_stats
    
    def reset(self) -> None:
        """Reset adaptive rate limiter state."""
        self.base_limiter.reset()
        self.current_rate_factor = 1.0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.last_429_time = 0.0
        self.backoff_until = 0.0
        self.response_history.clear()
        self.error_rates.clear()
        
        self.stats = {
            'rate_adjustments': 0,
            'backoff_events': 0,
            'recovery_events': 0,
            'current_rate_factor': 1.0,
            'avg_response_time': 0.0
        }


# Factory function
def create_rate_limiter(limiter_type: str = 'adaptive', 
                       config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Create a rate limiter of the specified type."""
    if config is None:
        config = RateLimitConfig()
    
    if limiter_type == 'token_bucket':
        return TokenBucketLimiter(config)
    elif limiter_type == 'sliding_window':
        return SlidingWindowLimiter(config)
    elif limiter_type == 'adaptive':
        return AdaptiveRateLimiter(config)
    else:
        raise ValueError(f"Unknown rate limiter type: {limiter_type}")


# Decorator for rate-limited functions
def rate_limited(limiter: RateLimiter, endpoint: str = None):
    """Decorator to apply rate limiting to functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Wait if rate limit is hit
            await limiter.wait_if_needed(endpoint)
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record successful request
                await limiter.record_request(endpoint, success=True)
                return result
                
            except Exception as e:
                # Record failed request
                await limiter.record_request(endpoint, success=False)
                
                # Handle 429 errors specifically for adaptive limiter
                if isinstance(limiter, AdaptiveRateLimiter) and hasattr(e, 'status_code'):
                    if e.status_code == 429:
                        retry_after = getattr(e, 'retry_after', None)
                        limiter.record_429_error(endpoint, retry_after)
                
                raise
        
        return wrapper
    return decorator


__all__ = [
    'RateLimiter',
    'TokenBucketLimiter', 
    'SlidingWindowLimiter',
    'AdaptiveRateLimiter',
    'RateLimitConfig',
    'RateLimitStatus',
    'create_rate_limiter',
    'rate_limited'
]