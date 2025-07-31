#!/usr/bin/env python3
"""
Anti-403 Configuration for Crawler
=================================

This module provides enhanced configuration to avoid 403 (Forbidden) errors
when crawling websites. It includes realistic user agents, headers, and
anti-detection strategies specifically tuned for news websites.

Features:
- Rotating realistic user agents
- Proper browser headers
- Regional-specific configurations
- Rate limiting strategies
- Proxy rotation support
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AntiDetectionConfig:
    """Configuration for anti-detection measures."""
    rotate_user_agents: bool = True
    use_realistic_headers: bool = True
    add_regional_headers: bool = True
    randomize_request_timing: bool = True
    min_delay_seconds: float = 1.0
    max_delay_seconds: float = 5.0
    respect_robots_txt: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


class UserAgentRotator:
    """Manages rotation of realistic user agents."""

    def __init__(self):
        self.user_agents = [
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",

            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",

            # Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",

            # Firefox on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",

            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",

            # Edge on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",

            # Mobile browsers (for sites that serve different content)
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        ]

        self.current_index = 0

    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        return random.choice(self.user_agents)

    def get_rotating_user_agent(self) -> str:
        """Get next user agent in rotation."""
        user_agent = self.user_agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return user_agent


class HeaderGenerator:
    """Generates realistic browser headers."""

    def __init__(self):
        self.languages = [
            "en-US,en;q=0.9",
            "en-GB,en;q=0.9",
            "en-US,en;q=0.9,sw;q=0.8",  # Swahili for East Africa
            "en-US,en;q=0.9,am;q=0.8",  # Amharic for Ethiopia
            "en-US,en;q=0.9,so;q=0.8",  # Somali
            "en-US,en;q=0.9,ar;q=0.8",  # Arabic for Sudan
        ]

        self.encodings = [
            "gzip, deflate, br",
            "gzip, deflate",
            "br, gzip, deflate",
        ]

        self.connections = ["keep-alive", "close"]

        # Common screen resolutions
        self.screen_resolutions = [
            "1920x1080",
            "1366x768",
            "1440x900",
            "1536x864",
            "1280x720",
            "1600x900",
        ]

    def generate_headers(self, user_agent: str, domain: str = None) -> Dict[str, str]:
        """Generate realistic headers for a request."""
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": random.choice(self.languages),
            "Accept-Encoding": random.choice(self.encodings),
            "DNT": "1",  # Do Not Track
            "Connection": random.choice(self.connections),
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        # Add Chrome-specific headers for Chrome user agents
        if "Chrome" in user_agent:
            headers.update({
                "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"' if "Windows" in user_agent else '"macOS"',
            })

        # Add regional headers for East African sites
        if domain and self._is_east_african_domain(domain):
            headers["Accept-Language"] = random.choice([
                "en-US,en;q=0.9,sw;q=0.8",  # Swahili
                "en-GB,en;q=0.9,sw;q=0.8",
                "en-US,en;q=0.9",
            ])

        return headers

    def _is_east_african_domain(self, domain: str) -> bool:
        """Check if domain is from East African region."""
        east_african_tlds = ['.ke', '.ug', '.et', '.so', '.sd', '.er', '.dj']
        east_african_domains = [
            'nation.co.ke', 'monitor.co.ug', 'addisfortune.net',
            'bbc.com/somali', 'hiiraan.com', 'shabait.com'
        ]

        domain_lower = domain.lower()
        return (any(domain_lower.endswith(tld) for tld in east_african_tlds) or
                any(domain_lower in d for d in east_african_domains))


class RateLimitManager:
    """Manages rate limiting to avoid triggering anti-bot measures."""

    def __init__(self, config: AntiDetectionConfig):
        self.config = config
        self.domain_delays = {}
        self.last_requests = {}

    async def wait_if_needed(self, domain: str):
        """Wait if needed to respect rate limits."""
        if not self.config.randomize_request_timing:
            return

        current_time = time.time()
        last_request_time = self.last_requests.get(domain, 0)

        min_delay = self.config.min_delay_seconds
        max_delay = self.config.max_delay_seconds

        # Calculate delay based on domain-specific rules
        if domain in self.domain_delays:
            min_delay = max(min_delay, self.domain_delays[domain])

        # Randomize delay
        delay = random.uniform(min_delay, max_delay)

        # Check if we need to wait
        time_since_last_request = current_time - last_request_time
        if time_since_last_request < delay:
            wait_time = delay - time_since_last_request
            await asyncio.sleep(wait_time)

        self.last_requests[domain] = time.time()

    def increase_delay(self, domain: str, factor: float = 1.5):
        """Increase delay for a domain after receiving 403 or rate limit."""
        current_delay = self.domain_delays.get(domain, self.config.min_delay_seconds)
        new_delay = min(current_delay * factor, 30.0)  # Cap at 30 seconds
        self.domain_delays[domain] = new_delay

    def reset_delay(self, domain: str):
        """Reset delay for a domain after successful requests."""
        if domain in self.domain_delays:
            del self.domain_delays[domain]


class DomainSpecificConfig:
    """Domain-specific configurations for problematic sites."""

    def __init__(self):
        self.domain_configs = {
            'monitor.co.ug': {
                'user_agent_preference': 'firefox',
                'additional_headers': {
                    'Referer': 'https://www.google.com/',
                    'X-Forwarded-For': '197.232.61.1',  # Uganda IP range
                },
                'min_delay': 3.0,
                'max_delay': 8.0,
                'respect_robots': True,
            },
            'nation.co.ke': {
                'user_agent_preference': 'chrome',
                'additional_headers': {
                    'Referer': 'https://www.google.co.ke/',
                },
                'min_delay': 2.0,
                'max_delay': 5.0,
            },
            'standardmedia.co.ke': {
                'user_agent_preference': 'safari',
                'additional_headers': {
                    'Referer': 'https://www.google.co.ke/',
                },
                'min_delay': 2.0,
                'max_delay': 6.0,
            },
            'addisfortune.net': {
                'user_agent_preference': 'chrome',
                'additional_headers': {
                    'Referer': 'https://www.google.com.et/',
                },
                'min_delay': 2.5,
                'max_delay': 7.0,
            },
            'bbc.com': {
                'user_agent_preference': 'chrome',
                'additional_headers': {
                    'Referer': 'https://www.google.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                },
                'min_delay': 1.0,
                'max_delay': 3.0,
            },
        }

    def get_config(self, domain: str) -> Dict:
        """Get configuration for a specific domain."""
        return self.domain_configs.get(domain, {})


class Anti403Manager:
    """Main manager for anti-403 strategies."""

    def __init__(self, config: AntiDetectionConfig = None):
        self.config = config or AntiDetectionConfig()
        self.user_agent_rotator = UserAgentRotator()
        self.header_generator = HeaderGenerator()
        self.rate_limiter = RateLimitManager(self.config)
        self.domain_config = DomainSpecificConfig()

        # Session management
        self.session_cookies = {}
        self.session_history = {}

    async def prepare_request(self, url: str) -> Dict[str, str]:
        """Prepare headers and timing for a request."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Wait for rate limiting
        await self.rate_limiter.wait_if_needed(domain)

        # Get domain-specific config
        domain_cfg = self.domain_config.get_config(domain)

        # Select user agent
        if self.config.rotate_user_agents:
            user_agent = self.user_agent_rotator.get_rotating_user_agent()
        else:
            user_agent = self.user_agent_rotator.user_agents[0]

        # Generate headers
        headers = self.header_generator.generate_headers(user_agent, domain)

        # Apply domain-specific headers
        if domain_cfg.get('additional_headers'):
            headers.update(domain_cfg['additional_headers'])

        return headers

    async def handle_403_response(self, url: str, attempt: int = 1) -> bool:
        """Handle 403 response and determine if retry should be attempted."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Increase delay for this domain
        self.rate_limiter.increase_delay(domain)

        # Check if we should retry
        if attempt >= self.config.max_retries:
            return False

        # Wait before retry
        await asyncio.sleep(self.config.retry_delay_seconds * attempt)

        return True

    def get_crawler_config(self, domain: str) -> Dict:
        """Get configuration for crawl4ai or newspaper3k."""
        domain_cfg = self.domain_config.get_config(domain)

        config = {
            'delay_range': (
                domain_cfg.get('min_delay', self.config.min_delay_seconds),
                domain_cfg.get('max_delay', self.config.max_delay_seconds)
            ),
            'user_agent_preference': domain_cfg.get('user_agent_preference', 'chrome'),
            'respect_robots_txt': domain_cfg.get('respect_robots', self.config.respect_robots_txt),
        }

        return config


# Import asyncio for sleep functionality
import asyncio


# Factory function for easy usage
def create_anti_403_manager(
    rotate_user_agents: bool = True,
    min_delay: float = 1.0,
    max_delay: float = 5.0,
    max_retries: int = 3
) -> Anti403Manager:
    """Create an Anti403Manager with custom configuration."""
    config = AntiDetectionConfig(
        rotate_user_agents=rotate_user_agents,
        min_delay_seconds=min_delay,
        max_delay_seconds=max_delay,
        max_retries=max_retries
    )
    return Anti403Manager(config)


class Anti403Handler:
    """Comprehensive handler for HTTP 403 (Forbidden) errors."""
    
    def __init__(self, config: Optional[AntiDetectionConfig] = None):
        self.config = config or AntiDetectionConfig()
        self.user_agent_rotator = UserAgentRotator()
        self.header_manager = HeaderManager()
        self.rate_limiter = RateLimiter(self.config)
        self.domain_config = DomainSpecificConfig()
        self.retry_counts = {}
        
    async def prepare_request(self, url: str, session_id: Optional[str] = None) -> Dict[str, str]:
        """Prepare request headers to avoid 403 errors."""
        domain = urlparse(url).netloc.lower()
        
        # Get domain-specific config
        domain_cfg = self.domain_config.get_config(domain)
        
        # Generate headers
        headers = self.header_manager.generate_headers(
            url=url,
            user_agent_preference=domain_cfg.get('user_agent_preference'),
            region=domain_cfg.get('region')
        )
        
        # Add domain-specific headers
        if 'additional_headers' in domain_cfg:
            headers.update(domain_cfg['additional_headers'])
            
        return headers
    
    async def handle_403_response(self, url: str, attempt: int = 1, 
                                response_headers: Optional[Dict[str, str]] = None) -> bool:
        """Handle 403 response and determine if retry should be attempted."""
        domain = urlparse(url).netloc.lower()
        
        # Track retry attempts
        self.retry_counts[url] = attempt
        
        # Check if we should retry
        if attempt >= self.config.max_retries:
            logger.warning(f"Max retries ({self.config.max_retries}) reached for {url}")
            return False
            
        # Analyze the 403 response
        if response_headers:
            # Check for Cloudflare
            if any(header.lower().startswith('cf-') for header in response_headers.keys()):
                logger.info(f"Cloudflare detected for {url}, may need specialized bypass")
                
            # Check for rate limiting indicators
            if 'retry-after' in response_headers:
                retry_after = response_headers.get('retry-after', '60')
                try:
                    delay = int(retry_after)
                    logger.info(f"Rate limited for {url}, waiting {delay} seconds")
                    await asyncio.sleep(min(delay, 300))  # Cap at 5 minutes
                except ValueError:
                    pass
        
        # Implement progressive backoff
        self.rate_limiter.increase_delay(domain, factor=2.0)
        
        # Wait before retry
        await self.rate_limiter.wait_for_request(url)
        
        return True
    
    async def reset_for_domain(self, domain: str):
        """Reset state for a specific domain after successful requests."""
        self.rate_limiter.reset_delay(domain)
        # Remove retry counts for this domain
        to_remove = [url for url in self.retry_counts.keys() if urlparse(url).netloc.lower() == domain]
        for url in to_remove:
            del self.retry_counts[url]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about 403 handling."""
        domain_retry_counts = {}
        for url, count in self.retry_counts.items():
            domain = urlparse(url).netloc.lower()
            domain_retry_counts[domain] = domain_retry_counts.get(domain, 0) + count
            
        return {
            'total_retries': sum(self.retry_counts.values()),
            'domains_with_issues': len(set(urlparse(url).netloc.lower() for url in self.retry_counts.keys())),
            'domain_retry_counts': domain_retry_counts,
            'current_delays': dict(self.rate_limiter.domain_delays)
        }


class Anti403Config:
    """Configuration alias for backward compatibility."""
    
    def __init__(self, **kwargs):
        # Map to AntiDetectionConfig
        self.config = AntiDetectionConfig(**kwargs)
        
    def __getattr__(self, name):
        return getattr(self.config, name)


def create_anti_403_handler(config: Optional[Anti403Config] = None) -> Anti403Handler:
    """Factory function to create Anti403Handler instance."""
    detection_config = config.config if config else None
    return Anti403Handler(detection_config)




# Example usage
if __name__ == "__main__":
    async def example_usage():
        """Example of how to use the Anti403Manager."""
        manager = create_anti_403_manager()

        # Prepare request for a problematic site
        url = "https://monitor.co.ug"
        headers = await manager.prepare_request(url)

        print("Generated headers:")
        for key, value in headers.items():
            print(f"  {key}: {value}")

        # Simulate handling a 403 response
        should_retry = await manager.handle_403_response(url, attempt=1)
        print(f"Should retry after 403: {should_retry}")

    asyncio.run(example_usage())
