"""
NewsAPI Utilities Package
========================

This package contains utility functions for the NewsAPI crawler.

Components:
- helpers.py: Helper functions for date handling, text processing, etc.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

from .helpers import (
    date_to_string,
    string_to_date,
    get_date_range,
    filter_articles_by_keywords,
    rank_articles_by_relevance,
    calculate_relevance_score,
    extract_locations,
    clean_html,
    truncate_text,
    merge_article_data,
    generate_article_id,
    get_article_publication_date,
    get_country_code,
    extract_domain
)

__all__ = [
    "date_to_string",
    "string_to_date",
    "get_date_range",
    "filter_articles_by_keywords",
    "rank_articles_by_relevance",
    "calculate_relevance_score",
    "extract_locations",
    "clean_html",
    "truncate_text",
    "merge_article_data",
    "generate_article_id",
    "get_article_publication_date",
    "get_country_code",
    "extract_domain"
]
