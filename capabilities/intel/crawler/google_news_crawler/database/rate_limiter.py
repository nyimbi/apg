"""
Rate Limiting Implementation for Google News Crawler
===================================================

Token bucket algorithm implementation for controlling request rates
to Google News and other sources with adaptive rate limiting.

Mathematical Model:
- Token bucket capacity: C tokens
- Refill rate: R tokens per second  
- Current tokens: T = min(C, T + (current_time - last_time) * R)
- Request allowed if: T >= cost

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
	"""Configuration for rate limiting."""
	# Token bucket parameters
	capacity: int = 100  # Maximum tokens in bucket
	refill_rate: float = 10.0  # Tokens per second
	cost_per_request: int = 1  # Tokens consumed per request
	
	# Adaptive rate limiting
	enable_adaptive: bool = True
	max_rate_multiplier: float = 2.0  # Maximum rate increase
	min_rate_multiplier: float = 0.1  # Minimum rate decrease
	
	# Burst protection
	burst_protection: bool = True
	burst_threshold: int = 50  # Requests in burst window
	burst_window_seconds: int = 60  # Burst detection window
	
	# Backoff configuration
	enable_backoff: bool = True
	initial_backoff_seconds: float = 1.0
	max_backoff_seconds: float = 300.0  # 5 minutes
	backoff_multiplier: float = 2.0

class TokenBucketRateLimiter:
	"""
	Production-ready token bucket rate limiter with adaptive behavior
	and circuit breaker integration.
	"""
	
	def __init__(self, config: RateLimitConfig):
		"""Initialize rate limiter with configuration."""
		self.config = config
		
		# Token bucket state
		self.tokens = float(config.capacity)
		self.last_refill_time = time.time()
		
		# Adaptive rate limiting state
		self.current_rate_multiplier = 1.0
		self.consecutive_failures = 0
		self.consecutive_successes = 0
		
		# Burst detection
		self.request_timestamps = []
		
		# Backoff state
		self.current_backoff = config.initial_backoff_seconds
		self.last_failure_time = None
		
		# Performance tracking
		self.stats = {
			'total_requests': 0,
			'allowed_requests': 0,
			'denied_requests': 0,
			'backoff_events': 0,
			'adaptive_adjustments': 0,
			'start_time': datetime.now()
		}
		
		# Thread safety
		self._lock = asyncio.Lock()
		
		logger.info(f"Rate limiter initialized: {config.capacity} tokens, {config.refill_rate}/sec")
	
	async def acquire(self, cost: int = None) -> bool:
		"""
		Attempt to acquire tokens for a request.
		
		Args:
			cost: Number of tokens to consume (default: config.cost_per_request)
			
		Returns:
			bool: True if request is allowed, False if rate limited
		"""
		if cost is None:
			cost = self.config.cost_per_request
		
		async with self._lock:
			current_time = time.time()
			
			# Refill tokens based on elapsed time
			await self._refill_tokens(current_time)
			
			# Check for burst protection
			if self.config.burst_protection:
				if self._is_burst_detected(current_time):
					logger.warning("Burst detected, denying request")
					self.stats['denied_requests'] += 1
					return False
			
			# Check if we have enough tokens
			if self.tokens >= cost:
				# Allow request
				self.tokens -= cost
				self.stats['total_requests'] += 1
				self.stats['allowed_requests'] += 1
				
				# Track request for burst detection
				self.request_timestamps.append(current_time)
				
				# Update adaptive rate limiting on success
				if self.config.enable_adaptive:
					await self._handle_success()
				
				return True
			else:
				# Deny request
				self.stats['total_requests'] += 1
				self.stats['denied_requests'] += 1
				
				# Update adaptive rate limiting on failure
				if self.config.enable_adaptive:
					await self._handle_failure()
				
				logger.debug(f"Rate limited: {self.tokens:.2f} tokens available, {cost} required")
				return False
	
	async def _refill_tokens(self, current_time: float) -> None:
		"""Refill tokens based on elapsed time."""
		elapsed = current_time - self.last_refill_time
		
		if elapsed > 0:
			# Calculate effective refill rate with adaptive multiplier
			effective_rate = self.config.refill_rate * self.current_rate_multiplier
			
			# Add tokens based on elapsed time
			tokens_to_add = elapsed * effective_rate
			self.tokens = min(self.config.capacity, self.tokens + tokens_to_add)
			
			self.last_refill_time = current_time
	
	def _is_burst_detected(self, current_time: float) -> bool:
		"""Check if current request pattern indicates a burst."""
		# Clean old timestamps
		cutoff_time = current_time - self.config.burst_window_seconds
		self.request_timestamps = [
			ts for ts in self.request_timestamps if ts > cutoff_time
		]
		
		# Check if we exceed burst threshold
		return len(self.request_timestamps) >= self.config.burst_threshold
	
	async def _handle_success(self) -> None:
		"""Handle successful request for adaptive rate limiting."""
		self.consecutive_successes += 1
		self.consecutive_failures = 0
		
		# Gradually increase rate if we have many successes
		if self.consecutive_successes >= 10:  # Configurable threshold
			old_multiplier = self.current_rate_multiplier
			self.current_rate_multiplier = min(
				self.config.max_rate_multiplier,
				self.current_rate_multiplier * 1.1  # 10% increase
			)
			
			if self.current_rate_multiplier != old_multiplier:
				self.stats['adaptive_adjustments'] += 1
				logger.debug(f"Adaptive rate increased: {old_multiplier:.2f} → {self.current_rate_multiplier:.2f}")
			
			self.consecutive_successes = 0  # Reset counter
	
	async def _handle_failure(self) -> None:
		"""Handle failed request for adaptive rate limiting."""
		self.consecutive_failures += 1
		self.consecutive_successes = 0
		self.last_failure_time = time.time()
		
		# Decrease rate on repeated failures
		if self.consecutive_failures >= 3:  # Configurable threshold
			old_multiplier = self.current_rate_multiplier
			self.current_rate_multiplier = max(
				self.config.min_rate_multiplier,
				self.current_rate_multiplier * 0.8  # 20% decrease
			)
			
			if self.current_rate_multiplier != old_multiplier:
				self.stats['adaptive_adjustments'] += 1
				logger.warning(f"Adaptive rate decreased: {old_multiplier:.2f} → {self.current_rate_multiplier:.2f}")
			
			self.consecutive_failures = 0  # Reset counter
	
	async def wait_if_needed(self, cost: int = None) -> bool:
		"""
		Wait until request can be satisfied, with optional backoff.
		
		Args:
			cost: Number of tokens needed
			
		Returns:
			bool: True if request can now proceed, False if should abort
		"""
		if cost is None:
			cost = self.config.cost_per_request
		
		# First attempt without waiting
		if await self.acquire(cost):
			return True
		
		# If backoff is enabled and we recently failed, apply backoff
		if self.config.enable_backoff and self.last_failure_time:
			time_since_failure = time.time() - self.last_failure_time
			if time_since_failure < self.current_backoff:
				backoff_time = self.current_backoff - time_since_failure
				logger.info(f"Applying backoff: waiting {backoff_time:.2f}s")
				await asyncio.sleep(backoff_time)
				self.stats['backoff_events'] += 1
		
		# Calculate wait time for tokens to become available
		tokens_needed = cost - self.tokens
		if tokens_needed > 0:
			effective_rate = self.config.refill_rate * self.current_rate_multiplier
			wait_time = tokens_needed / effective_rate
			
			# Cap wait time to reasonable maximum
			max_wait = 60.0  # 1 minute maximum wait
			if wait_time > max_wait:
				logger.warning(f"Required wait time ({wait_time:.2f}s) exceeds maximum, aborting")
				return False
			
			logger.debug(f"Waiting {wait_time:.2f}s for tokens to refill")
			await asyncio.sleep(wait_time)
		
		# Try again after waiting
		return await self.acquire(cost)
	
	def get_current_tokens(self) -> float:
		"""Get current number of tokens available."""
		current_time = time.time()
		elapsed = current_time - self.last_refill_time
		
		if elapsed > 0:
			effective_rate = self.config.refill_rate * self.current_rate_multiplier
			tokens_to_add = elapsed * effective_rate
			return min(self.config.capacity, self.tokens + tokens_to_add)
		
		return self.tokens
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get comprehensive rate limiter statistics."""
		current_time = datetime.now()
		uptime = current_time - self.stats['start_time']
		
		total_requests = self.stats['total_requests']
		success_rate = (self.stats['allowed_requests'] / max(1, total_requests)) * 100
		
		return {
			'uptime_seconds': uptime.total_seconds(),
			'total_requests': total_requests,
			'allowed_requests': self.stats['allowed_requests'],
			'denied_requests': self.stats['denied_requests'],
			'success_rate_percent': success_rate,
			'current_tokens': self.get_current_tokens(),
			'max_capacity': self.config.capacity,
			'current_rate_multiplier': self.current_rate_multiplier,
			'consecutive_failures': self.consecutive_failures,
			'consecutive_successes': self.consecutive_successes,
			'backoff_events': self.stats['backoff_events'],
			'adaptive_adjustments': self.stats['adaptive_adjustments'],
			'requests_per_minute': (total_requests / max(1, uptime.total_seconds())) * 60,
			'configuration': {
				'capacity': self.config.capacity,
				'refill_rate': self.config.refill_rate,
				'adaptive_enabled': self.config.enable_adaptive,
				'burst_protection': self.config.burst_protection,
				'backoff_enabled': self.config.enable_backoff
			}
		}
	
	def reset(self) -> None:
		"""Reset rate limiter state."""
		self.tokens = float(self.config.capacity)
		self.last_refill_time = time.time()
		self.current_rate_multiplier = 1.0
		self.consecutive_failures = 0
		self.consecutive_successes = 0
		self.request_timestamps.clear()
		self.current_backoff = self.config.initial_backoff_seconds
		self.last_failure_time = None
		
		# Reset stats but preserve start time
		start_time = self.stats['start_time']
		self.stats = {
			'total_requests': 0,
			'allowed_requests': 0,
			'denied_requests': 0,
			'backoff_events': 0,
			'adaptive_adjustments': 0,
			'start_time': start_time
		}
		
		logger.info("Rate limiter state reset")

class SourceSpecificRateLimiter:
	"""
	Rate limiter that manages different limits for different sources.
	"""
	
	def __init__(self):
		"""Initialize multi-source rate limiter."""
		self.limiters: Dict[str, TokenBucketRateLimiter] = {}
		self.default_config = RateLimitConfig()
		
		# Predefined configs for known sources
		self.source_configs = {
			'google_news': RateLimitConfig(
				capacity=50,
				refill_rate=5.0,  # Conservative for Google
				cost_per_request=1,
				enable_adaptive=True
			),
			'google_rss': RateLimitConfig(
				capacity=30,
				refill_rate=3.0,  # Even more conservative for RSS
				cost_per_request=1,
				enable_adaptive=True
			),
			'news_websites': RateLimitConfig(
				capacity=20,
				refill_rate=2.0,  # Respectful to individual sites
				cost_per_request=1,
				enable_adaptive=True,
				burst_protection=True
			)
		}
	
	def get_limiter(self, source: str) -> TokenBucketRateLimiter:
		"""Get or create rate limiter for specific source."""
		if source not in self.limiters:
			# Use source-specific config or default
			config = self.source_configs.get(source, self.default_config)
			self.limiters[source] = TokenBucketRateLimiter(config)
			logger.info(f"Created rate limiter for source: {source}")
		
		return self.limiters[source]
	
	async def acquire(self, source: str, cost: int = 1) -> bool:
		"""Acquire tokens for a specific source."""
		limiter = self.get_limiter(source)
		return await limiter.acquire(cost)
	
	async def wait_if_needed(self, source: str, cost: int = 1) -> bool:
		"""Wait for tokens for a specific source."""
		limiter = self.get_limiter(source)
		return await limiter.wait_if_needed(cost)
	
	def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
		"""Get stats for all source rate limiters."""
		return {
			source: limiter.get_stats() 
			for source, limiter in self.limiters.items()
		}

def create_rate_limiter(
	capacity: int = 100,
	refill_rate: float = 10.0,
	enable_adaptive: bool = True
) -> TokenBucketRateLimiter:
	"""Factory function to create a rate limiter with common settings."""
	config = RateLimitConfig(
		capacity=capacity,
		refill_rate=refill_rate,
		enable_adaptive=enable_adaptive
	)
	return TokenBucketRateLimiter(config)

def create_google_news_rate_limiter() -> TokenBucketRateLimiter:
	"""Create rate limiter optimized for Google News API."""
	config = RateLimitConfig(
		capacity=50,  # Conservative capacity
		refill_rate=5.0,  # 5 requests per second max
		cost_per_request=1,
		enable_adaptive=True,
		burst_protection=True,
		burst_threshold=20,  # Lower burst threshold
		enable_backoff=True
	)
	return TokenBucketRateLimiter(config)