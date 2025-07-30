"""
NewsAPI Content Parsers Package
==============================

This package contains parsers and content extractors for processing
news articles from the NewsAPI.

Components:
- content_parser.py: Advanced content extraction and analysis

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

from .content_parser import (
    ArticleParser,
    ContentExtractor,
    EventDetector
)

__all__ = [
    "ArticleParser",
    "ContentExtractor",
    "EventDetector"
]
