"""
GDELT API Client Components
===========================

API-based crawling components for GDELT DOC 2.0 and related services.
Provides comprehensive access to GDELT's real-time API endpoints with
optimized querying and content retrieval capabilities.

Components:
- **GDELTClient**: Enhanced DOC 2.0 API client with rate limiting
- **Query Components**: Advanced query builders and parameter handling
- **Content Processing**: HTML to markdown conversion and content extraction

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

try:
    from .gdelt_client import (
        GDELTClient,
        GDELTQueryParameters,
        GDELTArticle,
        GDELTDateRange,
        GDELTMode,
        GDELTFormat
    )
    
    __all__ = [
        'GDELTClient',
        'GDELTQueryParameters', 
        'GDELTArticle',
        'GDELTDateRange',
        'GDELTMode',
        'GDELTFormat'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    __all__ = []
    import logging
    logging.getLogger(__name__).warning(f"GDELT API components not available: {e}")

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"
__license__ = "MIT"