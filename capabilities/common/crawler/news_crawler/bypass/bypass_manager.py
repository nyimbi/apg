"""
Comprehensive Bypass Manager
============================

Unified bypass manager that coordinates all bypass strategies including
Cloudflare, anti-403 handling, and general anti-bot measures.

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse
import random
import time

# Import bypass components
try:
    from .cloudflare_bypass import CloudflareBypass, CloudflareConfig
    CLOUDFLARE_AVAILABLE = True
except ImportError:
    CLOUDFLARE_AVAILABLE = False
    CloudflareBypass = None
    CloudflareConfig = None

try:
    from .anti_403_handler import Anti403Handler, Anti403Config
    ANTI_403_AVAILABLE = True
except ImportError:
    ANTI_403_AVAILABLE = False
    Anti403Handler = None
    Anti403Config = None

# Import utils
try:
    from ....utils.monitoring import PerformanceMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PerformanceMonitor = None

logger = logging.getLogger(__name__)


@dataclass
class BypassConfig:
    """Configuration for bypass operations."""
    # Cloudflare bypass
    enable_cloudflare_bypass: bool = True
    cloudflare_timeout: int = 60
    cloudflare_max_retries: int = 3
    
    # Anti-403 handling
    enable_403_handling: bool = True
    max_403_retries: int = 5
    rotate_user_agents_on_403: bool = True
    
    # General bypass settings
    enable_js_challenge_solving: bool = True
    js_timeout: int = 30
    enable_captcha_detection: bool = True
    
    # Rate limiting
    min_delay_between_requests: float = 1.0
    max_delay_between_requests: float = 3.0
    exponential_backoff: bool = True
    
    # Session management
    session_rotation_interval: int = 100  # requests
    max_requests_per_session: int = 200
    
    # User agent management
    randomize_user_agents: bool = True
    user_agent_pool: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
    ])


@dataclass
class BypassResult:
    """Result of bypass operation."""
    success: bool
    content: Optional[str] = None
    status_code: Optional[int] = None
    bypass_method: Optional[str] = None
    response_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BypassMetrics:
    """Bypass operation metrics."""
    total_requests: int = 0
    successful_bypasses: int = 0
    failed_bypasses: int = 0
    cloudflare_bypasses: int = 0
    anti_403_recoveries: int = 0
    avg_response_time: float = 0.0
    bypass_methods_used: Dict[str, int] = field(default_factory=dict)


class BypassManager:
    """
    Comprehensive bypass manager that coordinates all bypass strategies.
    """
    
    def __init__(self, config: Optional[BypassConfig] = None):
        """
        Initialize bypass manager.
        
        Args:
            config: Optional bypass configuration
        """
        self.config = config or BypassConfig()
        
        # Initialize bypass components
        self.cloudflare_bypass = None
        self.anti_403_handler = None
        self.performance_monitor = None
        
        if CLOUDFLARE_AVAILABLE and self.config.enable_cloudflare_bypass:
            cf_config = CloudflareConfig(
                timeout=self.config.cloudflare_timeout,
                max_retries=self.config.cloudflare_max_retries
            )
            self.cloudflare_bypass = CloudflareBypass(cf_config)
        
        if ANTI_403_AVAILABLE and self.config.enable_403_handling:
            anti_403_config = Anti403Config(
                max_retries=self.config.max_403_retries,
                rotate_user_agents=self.config.rotate_user_agents_on_403
            )
            self.anti_403_handler = Anti403Handler(anti_403_config)
        
        if MONITORING_AVAILABLE:
            self.performance_monitor = PerformanceMonitor()
        
        # Session tracking
        self.current_session_requests = 0
        self.current_user_agent = self._get_random_user_agent()
        self.last_request_time = 0.0
        
        # Metrics
        self.metrics = BypassMetrics()
        
        logger.info("BypassManager initialized")
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the pool."""
        if self.config.randomize_user_agents and self.config.user_agent_pool:
            return random.choice(self.config.user_agent_pool)
        return self.config.user_agent_pool[0] if self.config.user_agent_pool else ""
    
    async def fetch_with_bypass(self, url: str, **kwargs) -> BypassResult:
        """
        Fetch URL with comprehensive bypass strategies.
        
        Args:
            url: URL to fetch
            **kwargs: Additional parameters
            
        Returns:
            BypassResult with fetch results
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        
        # Apply rate limiting
        await self._apply_rate_limiting()
        
        # Rotate session if needed
        await self._check_session_rotation()
        
        try:
            # Try bypass strategies in order of effectiveness
            result = await self._try_bypass_strategies(url, **kwargs)
            
            # Update metrics
            self._update_metrics(result, start_time)
            
            # Record performance metrics
            if self.performance_monitor:
                response_time = (time.time() - start_time) * 1000
                self.performance_monitor.record_timing('bypass_fetch', response_time)
                if result.success:
                    self.performance_monitor.record_counter('successful_bypasses', 1)
                else:
                    self.performance_monitor.record_counter('failed_bypasses', 1)
            
            return result
            
        except Exception as e:
            logger.error(f"Bypass fetch failed for {url}: {e}")
            self.metrics.failed_bypasses += 1
            
            return BypassResult(
                success=False,
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    async def _try_bypass_strategies(self, url: str, **kwargs) -> BypassResult:
        """
        Try different bypass strategies in order.
        
        Args:
            url: URL to fetch
            **kwargs: Additional parameters
            
        Returns:
            BypassResult from successful strategy
        """
        # Strategy 1: Direct request with basic headers
        result = await self._try_direct_request(url, **kwargs)
        if result.success:
            return result
        
        # Strategy 2: Cloudflare bypass if available and needed
        if (self.cloudflare_bypass and 
            self._is_cloudflare_protected(result) and 
            result.status_code in [403, 503, 429]):
            
            result = await self._try_cloudflare_bypass(url, **kwargs)
            if result.success:
                return result
        
        # Strategy 3: Anti-403 handling
        if (self.anti_403_handler and 
            result.status_code == 403):
            
            result = await self._try_anti_403_bypass(url, **kwargs)
            if result.success:
                return result
        
        # Strategy 4: Enhanced retry with different user agent
        if self.config.randomize_user_agents:
            self.current_user_agent = self._get_random_user_agent()
            result = await self._try_direct_request(url, **kwargs)
            if result.success:
                result.bypass_method = "user_agent_rotation"
                return result
        
        # All strategies failed
        return result
    
    async def _try_direct_request(self, url: str, **kwargs) -> BypassResult:
        """
        Try direct HTTP request with basic headers.
        
        Args:
            url: URL to fetch
            **kwargs: Additional parameters
            
        Returns:
            BypassResult from direct request
        """
        try:
            import aiohttp
            
            headers = {
                'User-Agent': self.current_user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, **kwargs) as response:
                    content = await response.text()
                    
                    return BypassResult(
                        success=response.status == 200,
                        content=content if response.status == 200 else None,
                        status_code=response.status,
                        bypass_method="direct_request",
                        response_time=0.0,  # Will be calculated by caller
                        metadata={
                            'headers': dict(response.headers),
                            'url': str(response.url)
                        }
                    )
                    
        except Exception as e:
            logger.warning(f"Direct request failed for {url}: {e}")
            return BypassResult(
                success=False,
                error_message=str(e),
                bypass_method="direct_request"
            )
    
    async def _try_cloudflare_bypass(self, url: str, **kwargs) -> BypassResult:
        """
        Try Cloudflare bypass if available.
        
        Args:
            url: URL to fetch
            **kwargs: Additional parameters
            
        Returns:
            BypassResult from Cloudflare bypass
        """
        if not self.cloudflare_bypass:
            return BypassResult(success=False, error_message="Cloudflare bypass not available")
        
        try:
            cf_result = await self.cloudflare_bypass.bypass_cloudflare(url, **kwargs)
            
            if cf_result.success:
                self.metrics.cloudflare_bypasses += 1
                return BypassResult(
                    success=True,
                    content=cf_result.content,
                    status_code=cf_result.status_code,
                    bypass_method="cloudflare_bypass",
                    metadata=cf_result.metadata
                )
            else:
                return BypassResult(
                    success=False,
                    error_message=cf_result.error_message,
                    bypass_method="cloudflare_bypass"
                )
                
        except Exception as e:
            logger.warning(f"Cloudflare bypass failed for {url}: {e}")
            return BypassResult(
                success=False,
                error_message=str(e),
                bypass_method="cloudflare_bypass"
            )
    
    async def _try_anti_403_bypass(self, url: str, **kwargs) -> BypassResult:
        """
        Try Anti-403 bypass if available.
        
        Args:
            url: URL to fetch
            **kwargs: Additional parameters
            
        Returns:
            BypassResult from Anti-403 bypass
        """
        if not self.anti_403_handler:
            return BypassResult(success=False, error_message="Anti-403 handler not available")
        
        try:
            anti_403_result = await self.anti_403_handler.handle_403(url, **kwargs)
            
            if anti_403_result.success:
                self.metrics.anti_403_recoveries += 1
                return BypassResult(
                    success=True,
                    content=anti_403_result.content,
                    status_code=anti_403_result.status_code,
                    bypass_method="anti_403_bypass",
                    metadata=anti_403_result.metadata
                )
            else:
                return BypassResult(
                    success=False,
                    error_message=anti_403_result.error_message,
                    bypass_method="anti_403_bypass"
                )
                
        except Exception as e:
            logger.warning(f"Anti-403 bypass failed for {url}: {e}")
            return BypassResult(
                success=False,
                error_message=str(e),
                bypass_method="anti_403_bypass"
            )
    
    def _is_cloudflare_protected(self, result: BypassResult) -> bool:
        """
        Check if the site appears to be Cloudflare protected.
        
        Args:
            result: Previous fetch result to analyze
            
        Returns:
            True if likely Cloudflare protected
        """
        if not result.content:
            return False
        
        cloudflare_indicators = [
            'cloudflare',
            'cf-ray',
            'checking your browser',
            'ddos protection',
            'security check'
        ]
        
        content_lower = result.content.lower()
        return any(indicator in content_lower for indicator in cloudflare_indicators)
    
    async def _apply_rate_limiting(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if self.config.min_delay_between_requests > 0:
            min_delay = self.config.min_delay_between_requests
            max_delay = self.config.max_delay_between_requests
            
            # Calculate delay with exponential backoff if enabled
            if self.config.exponential_backoff and self.metrics.failed_bypasses > 0:
                delay = min(min_delay * (2 ** min(self.metrics.failed_bypasses, 5)), max_delay)
            else:
                delay = random.uniform(min_delay, max_delay)
            
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _check_session_rotation(self):
        """Check if session needs to be rotated."""
        self.current_session_requests += 1
        
        if (self.config.session_rotation_interval > 0 and 
            self.current_session_requests >= self.config.session_rotation_interval):
            
            # Rotate session
            self.current_session_requests = 0
            self.current_user_agent = self._get_random_user_agent()
            
            logger.debug("Session rotated")
    
    def _update_metrics(self, result: BypassResult, start_time: float):
        """Update bypass metrics."""
        response_time = time.time() - start_time
        result.response_time = response_time
        
        if result.success:
            self.metrics.successful_bypasses += 1
        else:
            self.metrics.failed_bypasses += 1
        
        # Update average response time
        total_responses = self.metrics.successful_bypasses + self.metrics.failed_bypasses
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (total_responses - 1) + response_time) / 
            total_responses
        )
        
        # Track bypass methods
        if result.bypass_method:
            self.metrics.bypass_methods_used[result.bypass_method] = (
                self.metrics.bypass_methods_used.get(result.bypass_method, 0) + 1
            )
    
    def get_metrics(self) -> BypassMetrics:
        """Get current bypass metrics."""
        return self.metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive bypass statistics."""
        return {
            'config': {
                'cloudflare_bypass_enabled': self.config.enable_cloudflare_bypass,
                'anti_403_handling_enabled': self.config.enable_403_handling,
                'js_challenge_solving_enabled': self.config.enable_js_challenge_solving,
                'user_agent_randomization': self.config.randomize_user_agents
            },
            'components': {
                'cloudflare_bypass_available': self.cloudflare_bypass is not None,
                'anti_403_handler_available': self.anti_403_handler is not None,
                'performance_monitor_available': self.performance_monitor is not None
            },
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'successful_bypasses': self.metrics.successful_bypasses,
                'failed_bypasses': self.metrics.failed_bypasses,
                'success_rate': (
                    self.metrics.successful_bypasses / max(self.metrics.total_requests, 1)
                ),
                'cloudflare_bypasses': self.metrics.cloudflare_bypasses,
                'anti_403_recoveries': self.metrics.anti_403_recoveries,
                'avg_response_time': self.metrics.avg_response_time,
                'bypass_methods_used': self.metrics.bypass_methods_used
            },
            'session': {
                'current_session_requests': self.current_session_requests,
                'current_user_agent': self.current_user_agent[:50] + "..." if len(self.current_user_agent) > 50 else self.current_user_agent
            }
        }
    
    async def cleanup(self):
        """Cleanup bypass manager resources."""
        if self.cloudflare_bypass:
            await self.cloudflare_bypass.cleanup()
        
        if self.anti_403_handler:
            await self.anti_403_handler.cleanup()
        
        logger.info("BypassManager cleanup completed")


# Utility functions
def create_bypass_manager(config: Optional[BypassConfig] = None) -> BypassManager:
    """Create and configure a bypass manager."""
    return BypassManager(config)


def create_stealth_bypass_config() -> BypassConfig:
    """Create a bypass configuration optimized for stealth."""
    return BypassConfig(
        enable_cloudflare_bypass=True,
        enable_403_handling=True,
        enable_js_challenge_solving=True,
        enable_captcha_detection=True,
        min_delay_between_requests=2.0,
        max_delay_between_requests=5.0,
        exponential_backoff=True,
        session_rotation_interval=50,
        randomize_user_agents=True
    )


def create_performance_bypass_config() -> BypassConfig:
    """Create a bypass configuration optimized for performance."""
    return BypassConfig(
        enable_cloudflare_bypass=True,
        enable_403_handling=True,
        enable_js_challenge_solving=False,
        enable_captcha_detection=False,
        min_delay_between_requests=0.5,
        max_delay_between_requests=1.0,
        exponential_backoff=False,
        session_rotation_interval=200,
        randomize_user_agents=False
    )