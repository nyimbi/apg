#!/usr/bin/env python3
"""
NewsAPI Helper Utilities Module
==============================

Utility functions for the NewsAPI crawler package.

This module provides:
- Date formatting functions
- Article filtering and ranking
- Text processing utilities
- Geographical utilities

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, date, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Country code mappings for Horn of Africa region
HORN_OF_AFRICA_COUNTRIES = {
    "Ethiopia": "et",
    "Somalia": "so",
    "Kenya": "ke",
    "Sudan": "sd",
    "South Sudan": "ss",
    "Djibouti": "dj",
    "Eritrea": "er",
    "Uganda": "ug"
}

# Common date formats
DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%d %B %Y",
    "%B %d, %Y"
]


def date_to_string(date_obj: Union[datetime, date, str], format: str = "%Y-%m-%d") -> str:
    """
    Convert a date object to string.

    Args:
        date_obj: Date object or string
        format: Date format string

    Returns:
        Formatted date string
    """
    if isinstance(date_obj, str):
        # Try to parse the string as a date
        for fmt in DATE_FORMATS:
            try:
                date_obj = datetime.strptime(date_obj, fmt)
                break
            except ValueError:
                continue
        else:
            # If no format works, return the original string
            return date_obj

    if isinstance(date_obj, (datetime, date)):
        return date_obj.strftime(format)

    raise TypeError(f"Unsupported date type: {type(date_obj)}")


def string_to_date(date_string: str) -> Optional[datetime]:
    """
    Convert a string to a date object.

    Args:
        date_string: Date string

    Returns:
        Datetime object or None if parsing fails
    """
    if not date_string:
        return None

    # Try each format
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue

    # If no format works, try ISO format
    try:
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except ValueError:
        logger.warning(f"Could not parse date string: {date_string}")
        return None


def get_date_range(days_back: int = 7) -> tuple[str, str]:
    """
    Get a date range for searching.

    Args:
        days_back: Number of days to look back

    Returns:
        Tuple of (from_date, to_date) as strings in YYYY-MM-DD format
    """
    today = datetime.now()
    from_date = today - timedelta(days=days_back)
    return date_to_string(from_date), date_to_string(today)


def filter_articles_by_keywords(articles: List[Dict[str, Any]],
                               keywords: List[str],
                               min_matches: int = 1) -> List[Dict[str, Any]]:
    """
    Filter articles by keywords.

    Args:
        articles: List of article dictionaries
        keywords: List of keywords to match
        min_matches: Minimum number of keyword matches required

    Returns:
        Filtered list of articles
    """
    if not keywords or min_matches <= 0:
        return articles

    filtered_articles = []
    for article in articles:
        # Get text fields
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        full_text = article.get('full_text', '')

        # Combine text for searching
        combined_text = ' '.join([
            str(title),
            str(description),
            str(content),
            str(full_text)
        ]).lower()

        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword.lower() in combined_text)

        # Add to filtered list if it meets the threshold
        if matches >= min_matches:
            # Add the match count to the article
            article_copy = article.copy()
            article_copy['keyword_matches'] = matches
            filtered_articles.append(article_copy)

    return filtered_articles


def rank_articles_by_relevance(articles: List[Dict[str, Any]],
                              keywords: List[str],
                              prioritize_recent: bool = True) -> List[Dict[str, Any]]:
    """
    Rank articles by relevance to keywords and recency.

    Args:
        articles: List of article dictionaries
        keywords: List of keywords to calculate relevance
        prioritize_recent: Whether to factor in recency in ranking

    Returns:
        Articles sorted by relevance (and optionally recency)
    """
    # Calculate relevance scores
    for article in articles:
        # Get text fields
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        full_text = article.get('full_text', '')

        # Combine text for searching
        combined_text = ' '.join([
            str(title),
            str(description),
            str(content),
            str(full_text)
        ]).lower()

        # Calculate base relevance (percentage of keywords found)
        if keywords:
            matches = sum(1 for keyword in keywords if keyword.lower() in combined_text)
            relevance = matches / len(keywords)
        else:
            relevance = 0.5  # Default mid-relevance if no keywords

        # Bonus for keywords in title (3x weight)
        title_matches = sum(1 for keyword in keywords if keyword.lower() in str(title).lower())
        if keywords:
            title_relevance = min(1.0, title_matches / len(keywords) * 3)
            relevance = 0.7 * relevance + 0.3 * title_relevance

        article['relevance_score'] = relevance

        # Handle recency if requested
        if prioritize_recent:
            published_at = article.get('publishedAt') or article.get('published_at')
            if published_at:
                try:
                    # Convert to datetime if it's a string
                    if isinstance(published_at, str):
                        published_at = string_to_date(published_at)

                    if published_at:
                        # Calculate recency score (1.0 for now, decreasing for older articles)
                        age_days = (datetime.now() - published_at).total_seconds() / 86400
                        recency_score = max(0.0, min(1.0, 1.0 - (age_days / 30)))  # 30-day window

                        # Combine relevance and recency (weighted average)
                        article['relevance_score'] = 0.7 * relevance + 0.3 * recency_score
                except Exception as e:
                    logger.warning(f"Error calculating recency score: {str(e)}")

    # Sort by relevance score
    return sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)


def calculate_relevance_score(article: Dict[str, Any], query: str) -> float:
    """
    Calculate relevance score for an article based on a query.

    Args:
        article: Article dictionary
        query: Search query

    Returns:
        Relevance score between 0.0 and 1.0
    """
    # Extract query keywords
    keywords = [word.lower() for word in query.split() if len(word) > 2]

    if not keywords:
        return 0.5  # Default mid-relevance if no valid keywords

    # Get text fields
    title = article.get('title', '')
    description = article.get('description', '')
    content = article.get('content', '')

    # Calculate matches in each field with different weights
    title_weight = 0.5
    description_weight = 0.3
    content_weight = 0.2

    title_matches = sum(1 for keyword in keywords if keyword in str(title).lower())
    title_score = min(1.0, title_matches / len(keywords))

    description_matches = sum(1 for keyword in keywords if keyword in str(description).lower())
    description_score = min(1.0, description_matches / len(keywords))

    content_matches = sum(1 for keyword in keywords if keyword in str(content).lower())
    content_score = min(1.0, content_matches / len(keywords))

    # Calculate weighted score
    relevance = (
        title_weight * title_score +
        description_weight * description_score +
        content_weight * content_score
    )

    return relevance


def extract_locations(text: str) -> List[str]:
    """
    Extract location names from text using rule-based approach.

    Args:
        text: Text to process

    Returns:
        List of location names
    """
    # This is a simple rule-based approach - would be better with NLP
    locations = []

    # Check for Horn of Africa country names
    for country in HORN_OF_AFRICA_COUNTRIES.keys():
        if country in text:
            locations.append(country)

    # List of major cities in the Horn of Africa
    cities = [
        "Addis Ababa", "Nairobi", "Mogadishu", "Khartoum", "Juba",
        "Asmara", "Djibouti City", "Kampala", "Mombasa", "Kisumu",
        "Hargeisa", "Dire Dawa", "Gondar", "Bahir Dar", "Mekelle",
        "Eldoret", "Nakuru", "Nyeri", "Garissa", "Wajir",
        "Port Sudan", "Nyala", "El Obeid", "Kassala", "Wau",
        "Bor", "Malakal", "Aweil", "Berbera", "Borama",
        "Ali Sabieh", "Tadjoura", "Obock", "Gulu", "Mbale"
    ]

    for city in cities:
        if city in text:
            locations.append(city)

    return list(set(locations))  # Remove duplicates


def clean_html(html_text: str) -> str:
    """
    Clean HTML from text.

    Args:
        html_text: Text with HTML tags

    Returns:
        Cleaned text
    """
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', ' ', html_text)

    # Fix spacing
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # Decode HTML entities
    import html
    clean_text = html.unescape(clean_text)

    return clean_text.strip()


def truncate_text(text: str, max_length: int = 200, append_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        append_ellipsis: Whether to append ellipsis to truncated text

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(' ', 1)[0]
    if append_ellipsis:
        truncated += '...'

    return truncated


def merge_article_data(base_article: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new article data into base article, intelligently handling conflicts.

    Args:
        base_article: Base article dictionary
        new_data: New data to merge in

    Returns:
        Merged article data
    """
    result = base_article.copy()

    for key, value in new_data.items():
        # Skip None values
        if value is None:
            continue

        # Skip empty strings if we already have content
        if isinstance(value, str) and not value.strip() and key in result and result[key]:
            continue

        # Skip empty lists if we already have content
        if isinstance(value, list) and not value and key in result and result[key]:
            continue

        # Handle special merging for certain fields
        if key == 'keywords' and isinstance(value, list) and key in result and isinstance(result[key], list):
            # Merge keywords and remove duplicates
            result[key] = list(set(result[key] + value))
        elif key == 'locations' and isinstance(value, list) and key in result and isinstance(result[key], list):
            # Merge locations and remove duplicates
            result[key] = list(set(result[key] + value))
        else:
            # For other fields, prefer longer content when available
            if key in result and isinstance(value, str) and isinstance(result[key], str):
                if len(value) > len(result[key]):
                    result[key] = value
            else:
                result[key] = value

    return result


def generate_article_id(article: Dict[str, Any]) -> str:
    """
    Generate a stable ID for an article based on its content.

    Args:
        article: Article dictionary

    Returns:
        Stable ID string
    """
    # Use URL as primary key if available
    url = article.get('url')
    if url:
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()

    # Fallback to title and source
    title = article.get('title', '')
    source = article.get('source', {})
    source_name = source.get('name', '') if isinstance(source, dict) else str(source)

    if title and source_name:
        key = f"{title}|{source_name}"
        return hashlib.md5(key.encode()).hexdigest()

    # Last resort: use the whole article
    article_str = json.dumps(article, sort_keys=True)
    return hashlib.md5(article_str.encode()).hexdigest()


def get_article_publication_date(article: Dict[str, Any]) -> Optional[datetime]:
    """
    Extract publication date from an article.

    Args:
        article: Article dictionary

    Returns:
        Datetime object or None if not found
    """
    # Check different possible field names
    date_fields = ['publishedAt', 'published_at', 'pubDate', 'date', 'timestamp']

    for field in date_fields:
        value = article.get(field)
        if value:
            if isinstance(value, datetime):
                return value
            elif isinstance(value, str):
                date_obj = string_to_date(value)
                if date_obj:
                    return date_obj

    return None


def get_country_code(country_name: str) -> Optional[str]:
    """
    Get the ISO country code for a country name.

    Args:
        country_name: Country name

    Returns:
        ISO country code or None if not found
    """
    # Standard country codes
    country_name = country_name.strip().title()

    # Check Horn of Africa mapping first
    if country_name in HORN_OF_AFRICA_COUNTRIES:
        return HORN_OF_AFRICA_COUNTRIES[country_name]

    # Handle common aliases
    aliases = {
        "USA": "us",
        "United States": "us",
        "America": "us",
        "UK": "gb",
        "United Kingdom": "gb",
        "Britain": "gb",
        "UAE": "ae",
        "United Arab Emirates": "ae"
    }

    return aliases.get(country_name)


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: URL string

    Returns:
        Domain name
    """
    import re
    from urllib.parse import urlparse

    if not url:
        return ""

    try:
        parsed = urlparse(url)
        domain = parsed.netloc

        # Remove www. prefix if present
        domain = re.sub(r'^www\.', '', domain)

        return domain
    except Exception:
        return ""
