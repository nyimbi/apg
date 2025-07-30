"""
News Crawler Bypass Module
===========================

Bypass and evasion mechanisms for protected websites.
Handles Cloudflare, captcha, and other anti-bot measures.

Components:
- CloudflareBypass: Cloudflare-specific bypass techniques
- Anti403Handler: HTTP 403 error handling and recovery
- NewspaperBypass: Newspaper3k integration with bypass
- BypassManager: Unified bypass coordination system

Features:
- Cloudflare TLS fingerprinting bypass
- JavaScript challenge solving
- CAPTCHA detection and handling
- HTTP 403 error recovery
- Session management and rotation

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
import logging

# Version information
__version__ = "4.0.0"
__author__ = "Lindela Development Team"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import bypass components
try:
    from .cloudflare_bypass import CloudflareBypass, CloudflareConfig
    from .anti_403_handler import Anti403Handler, Anti403Config
    from .newspaper_bypass import NewspaperBypass, NewspaperConfig
    from .bypass_manager import (
        BypassManager, BypassConfig, BypassResult, BypassMetrics,
        create_bypass_manager, create_stealth_bypass_config, create_performance_bypass_config
    )
    _BYPASS_COMPONENTS_AVAILABLE = True
    logger.debug("Bypass components loaded successfully")
except ImportError as e:
    logger.warning(f"Some bypass components not available: {e}")
    _BYPASS_COMPONENTS_AVAILABLE = False
    
    # Placeholder classes
    class CloudflareBypass:
        def __init__(self, *args, **kwargs):
            raise ImportError("Cloudflare bypass component not available")
    
    class Anti403Handler:
        def __init__(self, *args, **kwargs):
            raise ImportError("Anti-403 handler component not available")
    
    class NewspaperBypass:
        def __init__(self, *args, **kwargs):
            raise ImportError("Newspaper bypass component not available")
    
    class BypassManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("Bypass manager component not available")
    
    # Placeholder configs and classes
    CloudflareConfig = None
    Anti403Config = None
    NewspaperConfig = None
    BypassConfig = None
    BypassResult = None
    BypassMetrics = None
    
    def create_bypass_manager(*args, **kwargs):
        raise ImportError("Bypass manager not available")
    
    def create_stealth_bypass_config(*args, **kwargs):
        raise ImportError("Bypass configuration not available")
    
    def create_performance_bypass_config(*args, **kwargs):
        raise ImportError("Bypass configuration not available")


# Utility functions
def get_bypass_health() -> Dict[str, Any]:
    """Get health status of bypass components."""
    return {
        'status': 'healthy' if _BYPASS_COMPONENTS_AVAILABLE else 'degraded',
        'components_available': _BYPASS_COMPONENTS_AVAILABLE,
        'version': __version__,
        'supported_bypasses': [
            'cloudflare', 'anti_403', 'newspaper', 'unified'
        ] if _BYPASS_COMPONENTS_AVAILABLE else []
    }


def get_default_bypass_config() -> Dict[str, Any]:
    """Get default bypass configuration."""
    return {
        'enable_cloudflare_bypass': True,
        'enable_403_handling': True,
        'enable_js_challenge_solving': True,
        'enable_captcha_detection': True,
        'min_delay_between_requests': 1.0,
        'max_delay_between_requests': 3.0,
        'session_rotation_interval': 100,
        'randomize_user_agents': True
    }


# Export all public components
__all__ = [
    # Core classes
    'CloudflareBypass',
    'Anti403Handler', 
    'NewspaperBypass',
    'BypassManager',
    
    # Configuration classes
    'CloudflareConfig',
    'Anti403Config',
    'NewspaperConfig',
    'BypassConfig',
    'BypassResult',
    'BypassMetrics',
    
    # Factory functions
    'create_bypass_manager',
    'create_stealth_bypass_config',
    'create_performance_bypass_config',
    
    # Utility functions
    'get_bypass_health',
    'get_default_bypass_config',
    
    # Version info
    '__version__',
    '__author__',
    '__license__'
]

# Module initialization
logger.info(f"News Crawler Bypass Module v{__version__} initialized")
logger.info(f"Bypass components available: {_BYPASS_COMPONENTS_AVAILABLE}")