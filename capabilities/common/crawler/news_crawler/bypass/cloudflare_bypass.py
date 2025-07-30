#!/usr/bin/env python3
"""
Enhanced Cloudflare Bypass Module
==================================

Provides comprehensive Cloudflare protection bypass capabilities using
multiple techniques including CloudScraper, browser automation, and
advanced header manipulation.

Author: Lindela Development Team
"""

import asyncio
import logging
import random
import time
import ssl
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp

# Enhanced HTTP libraries
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

try:
    import undetected_chromedriver as uc
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CloudflareConfig:
    """Configuration for Cloudflare bypass operations."""
    timeout: int = 60
    max_retries: int = 3
    wait_time: float = 10.0
    use_selenium: bool = True
    use_cloudscraper: bool = True
    headless: bool = True
    user_agent_rotation: bool = True
    delay_range: Tuple[float, float] = (2.0, 5.0)


@dataclass
class BypassResult:
    """Result of Cloudflare bypass operation."""
    success: bool
    content: Optional[str] = None
    status_code: Optional[int] = None
    response_headers: Dict[str, str] = field(default_factory=dict)
    bypass_method: Optional[str] = None
    response_time: float = 0.0
    error_message: Optional[str] = None
    cookies: Dict[str, str] = field(default_factory=dict)


class CloudflareBypass:
    """
    Comprehensive Cloudflare bypass implementation.
    
    Uses multiple strategies:
    1. CloudScraper (primary)
    2. Undetected Chrome (fallback)
    3. Enhanced aiohttp with stealth headers
    """
    
    def __init__(self, config: Optional[CloudflareConfig] = None):
        """Initialize Cloudflare bypass."""
        self.config = config or CloudflareConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize CloudScraper session
        self.cloudscraper_session = None
        if CLOUDSCRAPER_AVAILABLE and self.config.use_cloudscraper:
            self._initialize_cloudscraper()
        
        # Browser automation setup
        self.driver = None
        self.driver_pool = []
        
        # User agent pool for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        # Success tracking
        self.stats = {
            'total_attempts': 0,
            'successful_bypasses': 0,
            'method_success': {
                'cloudscraper': 0,
                'selenium': 0,
                'aiohttp_stealth': 0
            }
        }
    
    def _initialize_cloudscraper(self):
        """Initialize CloudScraper session with optimal settings."""
        try:
            self.cloudscraper_session = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'darwin',  # More realistic than linux
                    'desktop': True
                },
                delay=random.uniform(*self.config.delay_range),
                debug=False
            )
            
            # Set realistic headers
            self.cloudscraper_session.headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            })
            
            self.logger.info("CloudScraper session initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CloudScraper: {e}")
            self.cloudscraper_session = None
    
    def _get_random_user_agent(self) -> str:
        """Get random user agent."""
        if self.config.user_agent_rotation:
            return random.choice(self.user_agents)
        return self.user_agents[0]
    
    async def bypass_cloudflare(self, url: str, **kwargs) -> BypassResult:
        """
        Attempt to bypass Cloudflare protection using multiple methods.
        
        Args:
            url: URL to fetch
            **kwargs: Additional parameters
            
        Returns:
            BypassResult with content and metadata
        """
        start_time = time.time()
        self.stats['total_attempts'] += 1
        
        # Try methods in order of effectiveness
        methods = [
            ('cloudscraper', self._bypass_with_cloudscraper),
            ('selenium', self._bypass_with_selenium),
            ('aiohttp_stealth', self._bypass_with_aiohttp_stealth)
        ]
        
        for method_name, method_func in methods:
            if method_name == 'cloudscraper' and not self.config.use_cloudscraper:
                continue
            if method_name == 'selenium' and not self.config.use_selenium:
                continue
                
            try:
                self.logger.info(f"Attempting {method_name} bypass for {url}")
                result = await method_func(url, **kwargs)
                
                if result.success:
                    result.response_time = time.time() - start_time
                    result.bypass_method = method_name
                    self.stats['successful_bypasses'] += 1
                    self.stats['method_success'][method_name] += 1
                    self.logger.info(f"✅ {method_name} bypass successful for {url}")
                    return result
                else:
                    self.logger.warning(f"⚠️ {method_name} bypass failed: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"❌ {method_name} bypass error: {e}")
        
        # All methods failed
        self.logger.error(f"All bypass methods failed for {url}")
        return BypassResult(
            success=False,
            error_message="All bypass methods failed",
            response_time=time.time() - start_time
        )
    
    async def _bypass_with_cloudscraper(self, url: str, **kwargs) -> BypassResult:
        """Bypass using CloudScraper."""
        if not self.cloudscraper_session:
            return BypassResult(success=False, error_message="CloudScraper not available")
        
        try:
            # Add random delay
            await asyncio.sleep(random.uniform(*self.config.delay_range))
            
            # Update user agent if rotation enabled
            if self.config.user_agent_rotation:
                self.cloudscraper_session.headers['User-Agent'] = self._get_random_user_agent()
            
            # Make request in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.cloudscraper_session.get(
                    url,
                    timeout=self.config.timeout,
                    **kwargs
                )
            )
            
            if response.status_code == 200:
                return BypassResult(
                    success=True,
                    content=response.text,
                    status_code=response.status_code,
                    response_headers=dict(response.headers),
                    cookies=dict(response.cookies)
                )
            else:
                return BypassResult(
                    success=False,
                    status_code=response.status_code,
                    error_message=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return BypassResult(
                success=False,
                error_message=f"CloudScraper error: {e}"
            )
    
    async def _bypass_with_selenium(self, url: str, **kwargs) -> BypassResult:
        """Bypass using undetected Chrome."""
        if not SELENIUM_AVAILABLE:
            return BypassResult(success=False, error_message="Selenium not available")
        
        driver = None
        try:
            # Create undetected Chrome instance with correct options
            options = uc.ChromeOptions()
            if self.config.headless:
                options.add_argument('--headless=new')  # Updated headless flag
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-web-security')
            options.add_argument('--disable-features=VizDisplayCompositor')
            # Remove problematic experimental options for newer Chrome
            # options.add_experimental_option("excludeSwitches", ["enable-automation"])
            # options.add_experimental_option('useAutomationExtension', False)
            
            # Random user agent
            if self.config.user_agent_rotation:
                options.add_argument(f'--user-agent={self._get_random_user_agent()}')
            
            driver = uc.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Navigate to URL
            driver.get(url)
            
            # Wait for page load and potential Cloudflare challenge
            WebDriverWait(driver, self.config.timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Additional wait for Cloudflare
            await asyncio.sleep(self.config.wait_time)
            
            # Check if we're still on Cloudflare page
            page_source = driver.page_source
            if any(indicator in page_source.lower() for indicator in [
                'checking your browser', 'cloudflare', 'please wait', 'security check'
            ]):
                # Wait a bit more
                await asyncio.sleep(self.config.wait_time)
                page_source = driver.page_source
            
            # Get cookies for future requests
            cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}
            
            return BypassResult(
                success=True,
                content=page_source,
                status_code=200,
                cookies=cookies
            )
            
        except Exception as e:
            return BypassResult(
                success=False,
                error_message=f"Selenium error: {e}"
            )
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    async def _bypass_with_aiohttp_stealth(self, url: str, **kwargs) -> BypassResult:
        """Bypass using enhanced aiohttp with stealth headers."""
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=10,
                limit_per_host=5
            )
            
            # Enhanced stealth headers
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
                'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"'
            }
            
            # Add random delay
            await asyncio.sleep(random.uniform(*self.config.delay_range))
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            ) as session:
                async with session.get(url, **kwargs) as response:
                    content = await response.text()
                    
                    return BypassResult(
                        success=response.status == 200,
                        content=content if response.status == 200 else None,
                        status_code=response.status,
                        response_headers=dict(response.headers),
                        error_message=f"HTTP {response.status}" if response.status != 200 else None
                    )
                    
        except Exception as e:
            return BypassResult(
                success=False,
                error_message=f"aiohttp stealth error: {e}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bypass statistics."""
        success_rate = (
            self.stats['successful_bypasses'] / max(self.stats['total_attempts'], 1)
        )
        
        return {
            'total_attempts': self.stats['total_attempts'],
            'successful_bypasses': self.stats['successful_bypasses'],
            'success_rate': success_rate,
            'method_success': self.stats['method_success'],
            'cloudscraper_available': CLOUDSCRAPER_AVAILABLE,
            'selenium_available': SELENIUM_AVAILABLE
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.cloudscraper_session:
            try:
                self.cloudscraper_session.close()
            except:
                pass
        
        # Cleanup any remaining drivers
        for driver in self.driver_pool:
            try:
                driver.quit()
            except:
                pass
        self.driver_pool.clear()
        
        self.logger.info("CloudflareBypass cleanup completed")


# Convenience functions
def create_cloudflare_bypass(config: Optional[CloudflareConfig] = None) -> CloudflareBypass:
    """Create CloudflareBypass instance with configuration."""
    return CloudflareBypass(config)


def create_stealth_cloudflare_config() -> CloudflareConfig:
    """Create stealth-optimized Cloudflare configuration."""
    return CloudflareConfig(
        timeout=90,
        max_retries=3,
        wait_time=15.0,
        use_selenium=True,
        use_cloudscraper=True,
        headless=True,
        user_agent_rotation=True,
        delay_range=(3.0, 8.0)
    )


def create_fast_cloudflare_config() -> CloudflareConfig:
    """Create fast Cloudflare configuration."""
    return CloudflareConfig(
        timeout=30,
        max_retries=2,
        wait_time=5.0,
        use_selenium=False,
        use_cloudscraper=True,
        headless=True,
        user_agent_rotation=True,
        delay_range=(1.0, 3.0)
    )