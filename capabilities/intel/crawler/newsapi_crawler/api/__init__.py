"""
NewsAPI Client API Package
==========================

This package contains the NewsAPI client implementation with
advanced features like caching and rate limiting.

Components:
- newsapi_client.py: Core client implementation
- factory.py: Factory functions for creating client instances

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

from .newsapi_client import (
    NewsAPIClient,
    AdvancedNewsAPIClient,
    NewsAPIRateLimiter
)

from .factory import (
    create_client,
    create_advanced_client,
    create_batch_client,
    create_from_config,
    BatchClient
)

__all__ = [
    "NewsAPIClient",
    "AdvancedNewsAPIClient",
    "NewsAPIRateLimiter",
    "create_client",
    "create_advanced_client",
    "create_batch_client",
    "create_from_config",
    "BatchClient"
]
