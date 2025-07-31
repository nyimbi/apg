"""
NewsAPI Article Models Package
=============================

This package contains data models for working with NewsAPI articles
and related entities.

Models:
- NewsArticle: Represents a news article
- NewsSource: Represents a news source
- ArticleCollection: Collection of articles with metadata
- SearchParameters: Parameters for searching articles

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

from .article import (
    NewsArticle,
    NewsSource,
    ArticleCollection,
    SearchParameters
)

__all__ = [
    "NewsArticle",
    "NewsSource",
    "ArticleCollection",
    "SearchParameters"
]
