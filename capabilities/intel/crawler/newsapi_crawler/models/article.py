#!/usr/bin/env python3
"""
NewsAPI Article Models Module
============================

Data models for the NewsAPI crawler package. This module defines the
core data structures used to represent news articles, sources, and search
parameters.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class NewsSource:
    """Represents a news source from the NewsAPI."""

    id: str
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    country: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsSource':
        """Create a NewsSource from a dictionary."""
        return cls(**data)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'NewsSource':
        """Create a NewsSource from an API response."""
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            description=data.get('description'),
            url=data.get('url'),
            category=data.get('category'),
            language=data.get('language'),
            country=data.get('country')
        )


@dataclass
class NewsArticle:
    """Represents a news article from the NewsAPI."""

    source: Union[str, Dict[str, Any], NewsSource]
    author: Optional[str] = None
    title: str = ""
    description: Optional[str] = None
    url: str = ""
    url_to_image: Optional[str] = None
    published_at: Optional[Union[str, datetime]] = None
    content: Optional[str] = None

    # Enhanced fields
    full_text: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    sentiment: float = 0.0
    locations: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Process fields after initialization."""
        # Convert source to NewsSource if it's a dictionary
        if isinstance(self.source, dict):
            self.source = NewsSource(
                id=self.source.get('id', ''),
                name=self.source.get('name', '')
            )
        elif isinstance(self.source, str):
            self.source = NewsSource(id='', name=self.source)

        # Convert published_at to datetime if it's a string
        if isinstance(self.published_at, str):
            try:
                self.published_at = datetime.fromisoformat(
                    self.published_at.replace('Z', '+00:00')
                )
            except ValueError:
                try:
                    # Try with format from NewsAPI
                    self.published_at = datetime.strptime(
                        self.published_at, '%Y-%m-%dT%H:%M:%SZ'
                    )
                except ValueError:
                    # Keep as string if parsing fails
                    pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)

        # Convert source to dict if it's a NewsSource
        if isinstance(self.source, NewsSource):
            data['source'] = self.source.to_dict()

        # Convert datetime to ISO format
        if isinstance(self.published_at, datetime):
            data['published_at'] = self.published_at.isoformat()

        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create a NewsArticle from a dictionary."""
        # Handle nested source
        if 'source' in data and isinstance(data['source'], dict):
            data = data.copy()  # Avoid modifying the input
            source_data = data.pop('source')
            article = cls(source=NewsSource.from_dict(source_data), **data)
        else:
            article = cls(**data)

        return article

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create a NewsArticle from an API response."""
        return cls(
            source=data.get('source', {}),
            author=data.get('author'),
            title=data.get('title', ''),
            description=data.get('description'),
            url=data.get('url', ''),
            url_to_image=data.get('urlToImage'),
            published_at=data.get('publishedAt'),
            content=data.get('content')
        )

    def calculate_relevance(self, keywords: List[str]) -> float:
        """
        Calculate relevance score based on keywords.

        Args:
            keywords: List of keywords to match

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not keywords:
            return 0.0

        text = ' '.join(filter(None, [
            self.title or '',
            self.description or '',
            self.content or '',
            self.full_text or ''
        ])).lower()

        matches = sum(1 for keyword in keywords if keyword.lower() in text)
        self.relevance_score = min(1.0, matches / len(keywords))
        return self.relevance_score


@dataclass
class SearchParameters:
    """Parameters for searching articles."""

    query: Optional[str] = None
    sources: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    from_date: Optional[Union[str, datetime]] = None
    to_date: Optional[Union[str, datetime]] = None
    language: Optional[str] = None
    sort_by: str = "publishedAt"
    page_size: int = 100
    page: int = 1

    def __post_init__(self):
        """Process fields after initialization."""
        # Convert date fields to strings if they are datetime objects
        if isinstance(self.from_date, datetime):
            self.from_date = self.from_date.strftime('%Y-%m-%d')

        if isinstance(self.to_date, datetime):
            self.to_date = self.to_date.strftime('%Y-%m-%d')

        # Convert list fields to comma-separated strings
        if isinstance(self.sources, list):
            self.sources = ','.join(self.sources)

        if isinstance(self.domains, list):
            self.domains = ','.join(self.domains)

        if isinstance(self.exclude_domains, list):
            self.exclude_domains = ','.join(self.exclude_domains)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        params = {}

        if self.query:
            params['q'] = self.query

        if self.sources:
            params['sources'] = self.sources

        if self.domains:
            params['domains'] = self.domains

        if self.exclude_domains:
            params['excludeDomains'] = self.exclude_domains

        if self.from_date:
            params['from'] = self.from_date

        if self.to_date:
            params['to'] = self.to_date

        if self.language:
            params['language'] = self.language

        params['sortBy'] = self.sort_by
        params['pageSize'] = self.page_size
        params['page'] = self.page

        return params


@dataclass
class ArticleCollection:
    """Collection of articles with metadata."""

    articles: List[NewsArticle] = field(default_factory=list)
    total_results: int = 0
    status: str = ""
    code: Optional[str] = None
    message: Optional[str] = None
    search_parameters: Optional[SearchParameters] = None

    def __post_init__(self):
        """Initialize derived properties."""
        self.timestamp = datetime.now()

    @property
    def count(self) -> int:
        """Get the number of articles in the collection."""
        return len(self.articles)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'articles': [article.to_dict() for article in self.articles],
            'total_results': self.total_results,
            'status': self.status,
            'code': self.code,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'search_parameters': asdict(self.search_parameters) if self.search_parameters else None
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_api_response(cls, response: Dict[str, Any],
                         search_parameters: Optional[SearchParameters] = None) -> 'ArticleCollection':
        """
        Create an ArticleCollection from an API response.

        Args:
            response: NewsAPI response dictionary
            search_parameters: Original search parameters

        Returns:
            ArticleCollection with parsed articles
        """
        articles = [
            NewsArticle.from_api_response(article_data)
            for article_data in response.get('articles', [])
        ]

        return cls(
            articles=articles,
            total_results=response.get('totalResults', 0),
            status=response.get('status', ''),
            code=response.get('code'),
            message=response.get('message'),
            search_parameters=search_parameters
        )

    def filter(self,
              keywords: Optional[List[str]] = None,
              min_relevance: float = 0.0,
              languages: Optional[List[str]] = None,
              sources: Optional[List[str]] = None) -> 'ArticleCollection':
        """
        Filter articles based on criteria.

        Args:
            keywords: Keywords to filter by
            min_relevance: Minimum relevance score
            languages: Languages to include
            sources: Sources to include

        Returns:
            New ArticleCollection with filtered articles
        """
        filtered_articles = self.articles.copy()

        # Filter by keywords and relevance if specified
        if keywords:
            for article in filtered_articles:
                article.calculate_relevance(keywords)

            filtered_articles = [
                article for article in filtered_articles
                if article.relevance_score >= min_relevance
            ]

        # Filter by language if specified
        if languages:
            if isinstance(self.search_parameters, SearchParameters) and self.search_parameters.language:
                # Already filtered by API
                pass
            else:
                # Need to filter in-memory
                filtered_articles = [
                    article for article in filtered_articles
                    if isinstance(article.source, NewsSource) and
                    article.source.language in languages
                ]

        # Filter by source if specified
        if sources:
            filtered_articles = [
                article for article in filtered_articles
                if (isinstance(article.source, NewsSource) and
                   (article.source.id in sources or article.source.name in sources))
            ]

        # Create new collection with filtered articles
        return ArticleCollection(
            articles=filtered_articles,
            total_results=len(filtered_articles),
            status=self.status,
            search_parameters=self.search_parameters
        )

    def sort(self, key: str = 'published_at', reverse: bool = True) -> 'ArticleCollection':
        """
        Sort articles by the specified key.

        Args:
            key: Field to sort by (published_at, relevance_score, etc.)
            reverse: Sort in descending order if True

        Returns:
            New ArticleCollection with sorted articles
        """
        def get_sort_key(article):
            if key == 'published_at':
                return article.published_at or datetime.min
            elif key == 'relevance_score':
                return article.relevance_score
            elif key == 'title':
                return article.title
            elif key == 'source':
                if isinstance(article.source, NewsSource):
                    return article.source.name
                return str(article.source)
            else:
                return getattr(article, key, None)

        sorted_articles = sorted(
            self.articles,
            key=get_sort_key,
            reverse=reverse
        )

        return ArticleCollection(
            articles=sorted_articles,
            total_results=self.total_results,
            status=self.status,
            search_parameters=self.search_parameters
        )

    def paginate(self, page: int = 1, page_size: int = 20) -> 'ArticleCollection':
        """
        Paginate the collection.

        Args:
            page: Page number (1-based)
            page_size: Number of articles per page

        Returns:
            New ArticleCollection with paginated articles
        """
        start = (page - 1) * page_size
        end = start + page_size

        paginated_articles = self.articles[start:end]

        return ArticleCollection(
            articles=paginated_articles,
            total_results=self.total_results,  # Keep original total
            status=self.status,
            search_parameters=self.search_parameters
        )
