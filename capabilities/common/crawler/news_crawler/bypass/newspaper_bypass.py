#!/usr/bin/env python3
"""
Simple Newspaper3k Stealth Wrapper
=================================

A lightweight wrapper for newspaper3k that provides stealth capabilities
for protected domains without heavy dependencies. This module monkey-patches
newspaper3k's HTTP requests to use our stealth headers.

Features:
- Monkey-patches newspaper3k's network module
- Uses existing stealth infrastructure
- No additional dependencies required
- Maintains full newspaper3k API compatibility
- Automatic stealth activation for protected domains

Author: Datacraft Team
License: MIT
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Global stealth components (initialized when first used)
_stealth_orchestrator = None
_anti_403_manager = None
_original_get_html = None
_protected_domains = ['nation.co.ke', 'standardmedia.co.ke', 'monitor.co.ug', 'theeastafrican.co.ke']

def _initialize_stealth_components():
    """Initialize stealth components on first use."""
    global _stealth_orchestrator, _anti_403_manager

    if _stealth_orchestrator is None:
        try:
            # Import stealth components
            try:
                from .lightweight_stealth import create_lightweight_stealth
                from .anti_403_config import create_anti_403_manager
            except ImportError:
                # Fallback for direct execution
                current_dir = Path(__file__).parent
                if str(current_dir) not in sys.path:
                    sys.path.insert(0, str(current_dir))
                from lightweight_stealth import create_lightweight_stealth
                from anti_403_config import create_anti_403_manager

            _stealth_orchestrator = create_lightweight_stealth()
            _anti_403_manager = create_anti_403_manager()

            logger.info("Newspaper stealth components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize stealth components: {e}")
            # Set dummy objects to prevent repeated attempts
            _stealth_orchestrator = False
            _anti_403_manager = False

def _is_protected_domain(url: str) -> bool:
    """Check if URL belongs to a protected domain."""
    try:
        domain = urlparse(url).netloc.lower()
        return any(protected_domain in domain for protected_domain in _protected_domains)
    except:
        return False

async def _get_stealth_headers(url: str) -> Dict[str, str]:
    """Get stealth headers for the given URL."""
    _initialize_stealth_components()

    if not _stealth_orchestrator or not _anti_403_manager:
        # Return basic headers if stealth components failed to initialize
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }

    try:
        # Apply timing delays for protected sites
        if _is_protected_domain(url):
            await _stealth_orchestrator.apply_timing_delay(url)

            # Get stealth configuration
            stealth_config = await _stealth_orchestrator.configure_stealth(url)
            headers = stealth_config.get('headers', {})

            # Merge with anti-403 headers
            anti_403_headers = await _anti_403_manager.prepare_request(url)
            headers.update(anti_403_headers)

            logger.debug(f"Generated {len(headers)} stealth headers for {urlparse(url).netloc}")
            return headers
        else:
            # Get basic anti-403 headers for non-protected sites
            return await _anti_403_manager.prepare_request(url)

    except Exception as e:
        logger.error(f"Failed to generate stealth headers for {url}: {e}")
        # Return basic fallback headers
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

def _stealth_get_html(url: str, config=None, response=None):
    """Stealth-enabled replacement for newspaper's get_html function."""
    global _original_get_html

    try:
        # Check if this domain needs stealth protection
        if _is_protected_domain(url):
            logger.info(f"Applying stealth measures for: {urlparse(url).netloc}")

            # Get stealth headers synchronously (newspaper3k doesn't support async)
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't use run_until_complete
                    # Create a task and wait for it
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(_get_stealth_headers(url))
                        )
                        headers = future.result(timeout=10)
                else:
                    headers = loop.run_until_complete(_get_stealth_headers(url))
            except:
                # Fallback to basic headers if async fails
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/"
                }

            # Make request with stealth headers
            import requests
            try:
                session = requests.Session()
                session.headers.update(headers)

                response = session.get(url, timeout=60, allow_redirects=True)

                # Track success/failure for metrics
                if response.status_code == 200:
                    if _stealth_orchestrator:
                        _stealth_orchestrator.record_success(url)
                elif response.status_code == 403:
                    if _stealth_orchestrator:
                        _stealth_orchestrator.record_failure(url, 403)

                return response.text

            except Exception as e:
                logger.error(f"Stealth request failed for {url}: {e}")
                # Record failure
                if _stealth_orchestrator:
                    _stealth_orchestrator.record_failure(url, 0)
                # Fall back to original implementation
                pass

    except Exception as e:
        logger.debug(f"Stealth wrapper error for {url}: {e}")

    # Fall back to original newspaper3k implementation
    if _original_get_html:
        return _original_get_html(url, config, response)
    else:
        # Last resort: basic requests
        import requests
        try:
            response = requests.get(url, timeout=30)
            return response.text
        except:
            return ""

def apply_newspaper_stealth_patch():
    """Apply stealth patch to newspaper3k's network module."""
    global _original_get_html

    try:
        # Import newspaper's network module
        from newspaper import network

        # Store original function
        if _original_get_html is None:
            _original_get_html = network.get_html

        # Replace with stealth version
        network.get_html = _stealth_get_html

        logger.info("Applied stealth patch to newspaper3k network module")
        return True

    except ImportError:
        logger.debug("newspaper3k not available, using basic HTTP fallback")
        return False
    except Exception as e:
        logger.error(f"Failed to apply newspaper stealth patch: {e}")
        return False

def remove_newspaper_stealth_patch():
    """Remove stealth patch and restore original newspaper3k behavior."""
    global _original_get_html

    try:
        if _original_get_html:
            from newspaper import network
            network.get_html = _original_get_html
            logger.info("Removed stealth patch from newspaper3k")
            return True
    except Exception as e:
        logger.error(f"Failed to remove newspaper stealth patch: {e}")

    return False

def get_stealth_stats() -> Dict[str, Any]:
    """Get stealth performance statistics."""
    _initialize_stealth_components()

    if _stealth_orchestrator and hasattr(_stealth_orchestrator, 'get_stealth_report'):
        try:
            return _stealth_orchestrator.get_stealth_report()
        except:
            pass

    return {
        'total_requests': 0,
        'success_rate': 0,
        'stealth_effectiveness': 0,
        'message': 'Stealth components not initialized'
    }

class NewspaperStealthManager:
    """Context manager for newspaper stealth operations."""

    def __init__(self, auto_patch: bool = True):
        self.auto_patch = auto_patch
        self.patch_applied = False

    def __enter__(self):
        if self.auto_patch:
            self.patch_applied = apply_newspaper_stealth_patch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patch_applied:
            remove_newspaper_stealth_patch()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return get_stealth_stats()

    def is_protected_domain(self, url: str) -> bool:
        """Check if domain requires stealth protection."""
        return _is_protected_domain(url)

# Configuration class for newspaper bypass
class NewspaperConfig:
    """Configuration for newspaper stealth operations."""
    
    def __init__(self, auto_patch: bool = True, protected_domains: List[str] = None):
        self.auto_patch = auto_patch
        self.protected_domains = protected_domains or _protected_domains
        
    def update_protected_domains(self, domains: List[str]):
        """Update list of protected domains."""
        global _protected_domains
        self.protected_domains = domains
        _protected_domains = domains

# Convenience functions
def enable_newspaper_stealth():
    """Enable stealth mode for newspaper3k."""
    return apply_newspaper_stealth_patch()

def disable_newspaper_stealth():
    """Disable stealth mode for newspaper3k."""
    return remove_newspaper_stealth_patch()

# Auto-apply patch when module is imported (can be disabled)
AUTO_PATCH_ON_IMPORT = True

if AUTO_PATCH_ON_IMPORT:
    try:
        apply_newspaper_stealth_patch()
    except:
        pass  # Silently fail if newspaper3k not available

# Export alias for backward compatibility
NewspaperBypass = NewspaperStealthManager

# Factory functions
def create_newspaper_bypass(auto_patch: bool = True) -> NewspaperStealthManager:
    """Create newspaper bypass manager."""
    return NewspaperStealthManager(auto_patch=auto_patch)

def create_newspaper_stealth_manager(auto_patch: bool = True) -> NewspaperStealthManager:
    """Create newspaper stealth manager."""
    return NewspaperStealthManager(auto_patch=auto_patch)

# Export all components
__all__ = [
    'NewspaperStealthManager',
    'NewspaperBypass',
    'NewspaperConfig',
    'apply_newspaper_stealth_patch',
    'remove_newspaper_stealth_patch',
    'enable_newspaper_stealth',
    'disable_newspaper_stealth',
    'get_stealth_stats',
    'create_newspaper_bypass',
    'create_newspaper_stealth_manager'
]

# Example usage
if __name__ == "__main__":
    # Test the stealth wrapper
    print("Testing Newspaper Stealth Wrapper...")

    # Test domain detection
    test_urls = [
        "https://nation.co.ke",
        "https://monitor.co.ug",
        "https://bbc.com"
    ]

    for url in test_urls:
        is_protected = _is_protected_domain(url)
        print(f"  {url}: {'Protected' if is_protected else 'Standard'}")

    # Test stealth manager
    with NewspaperStealthManager() as manager:
        print(f"  Stealth manager active")
        stats = manager.get_stats()
        print(f"  Stats: {stats}")

    print("✅ Newspaper stealth wrapper ready!")
