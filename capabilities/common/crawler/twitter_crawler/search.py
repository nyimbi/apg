"""
Twitter Search Engine Module
=============================

Advanced Twitter search capabilities with comprehensive filtering and query building.
Provides intelligent search operations, result processing, and conflict-specific searches.

Author: Lindela Development Team
Version: 2.0.0
License: MIT
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Set, Callable
from enum import Enum
from datetime import datetime, timedelta
import re
import json

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from .core import TwitterCrawler, TwitterConfig, CrawlerError
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    logger.warning("Core Twitter crawler not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class SearchType(Enum):
    """Types of Twitter searches"""
    RECENT = "recent"
    POPULAR = "popular"
    MIXED = "mixed"
    LIVE = "live"


class LanguageCode(Enum):
    """Common language codes for Twitter"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"
    ARABIC = "ar"
    HINDI = "hi"
    TURKISH = "tr"
    DUTCH = "nl"
    SWEDISH = "sv"


class FilterOperator(Enum):
    """Search filter operators"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    NEAR = "NEAR"


@dataclass
class DateRange:
    """Date range for search filtering"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def to_twitter_format(self) -> Dict[str, str]:
        """Convert to Twitter API format"""
        result = {}
        if self.start_date:
            result['since'] = self.start_date.strftime('%Y-%m-%d')
        if self.end_date:
            result['until'] = self.end_date.strftime('%Y-%m-%d')
        return result


@dataclass
class GeographicFilter:
    """Geographic filtering for searches"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius: Optional[str] = None  # e.g., "10km", "5mi"
    place: Optional[str] = None
    country_code: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if geographic filter is valid"""
        return (
            (self.latitude is not None and self.longitude is not None and self.radius is not None) or
            self.place is not None or
            self.country_code is not None
        )
    
    def to_query_string(self) -> str:
        """Convert to Twitter query string format"""
        if self.latitude is not None and self.longitude is not None and self.radius:
            return f"geocode:{self.latitude},{self.longitude},{self.radius}"
        elif self.place:
            return f'place:"{self.place}"'
        elif self.country_code:
            return f"place_country:{self.country_code}"
        return ""


@dataclass
class TweetFilter:
    """Advanced tweet filtering options"""
    # Content filters
    has_links: Optional[bool] = None
    has_images: Optional[bool] = None
    has_videos: Optional[bool] = None
    has_hashtags: Optional[bool] = None
    has_mentions: Optional[bool] = None
    
    # User filters
    verified_users_only: bool = False
    min_followers: Optional[int] = None
    max_followers: Optional[int] = None
    
    # Engagement filters
    min_retweets: Optional[int] = None
    min_likes: Optional[int] = None
    min_replies: Optional[int] = None
    
    # Content type filters
    exclude_retweets: bool = False
    exclude_replies: bool = False
    include_quotes: bool = True
    
    # Language and location
    languages: List[str] = field(default_factory=list)
    exclude_languages: List[str] = field(default_factory=list)
    geographic_filter: Optional[GeographicFilter] = None
    
    # Temporal filters
    date_range: Optional[DateRange] = None
    
    # Safety filters
    safe_search: bool = True
    
    def to_query_parts(self) -> List[str]:
        """Convert filter to Twitter query parts"""
        parts = []
        
        # Content filters
        if self.has_links is True:
            parts.append("filter:links")
        elif self.has_links is False:
            parts.append("-filter:links")
        
        if self.has_images is True:
            parts.append("filter:images")
        elif self.has_images is False:
            parts.append("-filter:images")
        
        if self.has_videos is True:
            parts.append("filter:videos")
        elif self.has_videos is False:
            parts.append("-filter:videos")
        
        # User filters
        if self.verified_users_only:
            parts.append("filter:verified")
        
        if self.min_followers:
            parts.append(f"min_faves:{self.min_followers}")
        
        # Engagement filters
        if self.min_retweets:
            parts.append(f"min_retweets:{self.min_retweets}")
        
        if self.min_likes:
            parts.append(f"min_faves:{self.min_likes}")
        
        if self.min_replies:
            parts.append(f"min_replies:{self.min_replies}")
        
        # Content type filters
        if self.exclude_retweets:
            parts.append("-filter:retweets")
        
        if self.exclude_replies:
            parts.append("-filter:replies")
        
        # Language filters
        for lang in self.languages:
            parts.append(f"lang:{lang}")
        
        for lang in self.exclude_languages:
            parts.append(f"-lang:{lang}")
        
        # Geographic filter
        if self.geographic_filter and self.geographic_filter.is_valid():
            geo_query = self.geographic_filter.to_query_string()
            if geo_query:
                parts.append(geo_query)
        
        # Safety filter
        if not self.safe_search:
            parts.append("filter:safe")
        
        return parts


@dataclass
class SearchQuery:
    """Comprehensive search query specification"""
    # Core query
    query: str
    
    # Search parameters
    max_results: int = 100
    search_type: SearchType = SearchType.RECENT
    
    # Filters
    tweet_filter: Optional[TweetFilter] = None
    
    # Advanced options
    include_user_data: bool = True
    include_engagement_data: bool = True
    include_media_data: bool = False
    
    # Processing options
    deduplicate: bool = True
    sort_by: Optional[str] = None  # 'relevance', 'recent', 'popular'
    
    def build_query_string(self) -> str:
        """Build the complete Twitter search query string"""
        query_parts = [self.query]
        
        if self.tweet_filter:
            query_parts.extend(self.tweet_filter.to_query_parts())
        
        return " ".join(query_parts)
    
    def get_search_params(self) -> Dict[str, Any]:
        """Get search parameters for API call"""
        params = {
            'q': self.build_query_string(),
            'count': min(self.max_results, 100),  # Twitter API limit
            'result_type': self.search_type.value,
            'include_entities': True,
            'tweet_mode': 'extended'
        }
        
        if self.tweet_filter and self.tweet_filter.date_range:
            params.update(self.tweet_filter.date_range.to_twitter_format())
        
        if self.tweet_filter and self.tweet_filter.languages:
            params['lang'] = self.tweet_filter.languages[0]  # Twitter API takes single language
        
        return params


@dataclass
class SearchResult:
    """Search result container"""
    query: SearchQuery
    tweets: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    total_found: Optional[int] = None
    next_token: Optional[str] = None
    rate_limit_info: Optional[Dict[str, Any]] = None
    
    def to_dataframe(self) -> Optional[Any]:
        """Convert results to pandas DataFrame"""
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available for DataFrame conversion")
            return None
        
        if not self.tweets:
            return pd.DataFrame()
        
        return pd.DataFrame(self.tweets)
    
    def filter_by_engagement(self, min_engagement: int = 10) -> 'SearchResult':
        """Filter results by minimum engagement"""
        filtered_tweets = []
        
        for tweet in self.tweets:
            engagement = (
                tweet.get('retweet_count', 0) +
                tweet.get('favorite_count', 0) +
                tweet.get('reply_count', 0)
            )
            if engagement >= min_engagement:
                filtered_tweets.append(tweet)
        
        return SearchResult(
            query=self.query,
            tweets=filtered_tweets,
            metadata=self.metadata,
            execution_time=self.execution_time
        )
    
    def get_hashtags(self) -> List[str]:
        """Extract all hashtags from results"""
        hashtags = set()
        for tweet in self.tweets:
            hashtags.update(tweet.get('hashtags', []))
        return list(hashtags)
    
    def get_mentions(self) -> List[str]:
        """Extract all mentions from results"""
        mentions = set()
        for tweet in self.tweets:
            # Extract mentions from text using regex
            text = tweet.get('text', '')
            tweet_mentions = re.findall(r'@(\w+)', text)
            mentions.update(tweet_mentions)
        return list(mentions)
    
    def get_urls(self) -> List[str]:
        """Extract all URLs from results"""
        urls = set()
        for tweet in self.tweets:
            urls.update(tweet.get('urls', []))
        return list(urls)


class QueryBuilder:
    """Helper class for building complex Twitter queries"""
    
    def __init__(self):
        self.parts: List[str] = []
        self.operators: List[str] = []
    
    def add_keyword(self, keyword: str, operator: FilterOperator = FilterOperator.AND) -> 'QueryBuilder':
        """Add a keyword to the query"""
        if operator != FilterOperator.AND and self.parts:
            self.operators.append(operator.value)
        self.parts.append(f'"{keyword}"' if ' ' in keyword else keyword)
        return self
    
    def add_hashtag(self, hashtag: str, operator: FilterOperator = FilterOperator.AND) -> 'QueryBuilder':
        """Add a hashtag to the query"""
        if operator != FilterOperator.AND and self.parts:
            self.operators.append(operator.value)
        hashtag = hashtag.lstrip('#')
        self.parts.append(f"#{hashtag}")
        return self
    
    def add_mention(self, username: str, operator: FilterOperator = FilterOperator.AND) -> 'QueryBuilder':
        """Add a mention to the query"""
        if operator != FilterOperator.AND and self.parts:
            self.operators.append(operator.value)
        username = username.lstrip('@')
        self.parts.append(f"@{username}")
        return self
    
    def add_from_user(self, username: str) -> 'QueryBuilder':
        """Add from:user filter"""
        username = username.lstrip('@')
        self.parts.append(f"from:{username}")
        return self
    
    def add_to_user(self, username: str) -> 'QueryBuilder':
        """Add to:user filter"""
        username = username.lstrip('@')
        self.parts.append(f"to:{username}")
        return self
    
    def add_phrase(self, phrase: str, operator: FilterOperator = FilterOperator.AND) -> 'QueryBuilder':
        """Add an exact phrase to the query"""
        if operator != FilterOperator.AND and self.parts:
            self.operators.append(operator.value)
        self.parts.append(f'"{phrase}"')
        return self
    
    def exclude_keyword(self, keyword: str) -> 'QueryBuilder':
        """Exclude a keyword from the query"""
        self.parts.append(f'-"{keyword}"' if ' ' in keyword else f'-{keyword}')
        return self
    
    def exclude_hashtag(self, hashtag: str) -> 'QueryBuilder':
        """Exclude a hashtag from the query"""
        hashtag = hashtag.lstrip('#')
        self.parts.append(f"-#{hashtag}")
        return self
    
    def add_language(self, lang_code: str) -> 'QueryBuilder':
        """Add language filter"""
        self.parts.append(f"lang:{lang_code}")
        return self
    
    def add_location(self, location: str) -> 'QueryBuilder':
        """Add location filter"""
        self.parts.append(f'near:"{location}"')
        return self
    
    def add_geocode(self, lat: float, lon: float, radius: str) -> 'QueryBuilder':
        """Add geocode filter"""
        self.parts.append(f"geocode:{lat},{lon},{radius}")
        return self
    
    def add_date_range(self, since: str, until: str) -> 'QueryBuilder':
        """Add date range filters"""
        self.parts.append(f"since:{since}")
        self.parts.append(f"until:{until}")
        return self
    
    def build(self) -> str:
        """Build the final query string"""
        if not self.parts:
            return ""
        
        query = self.parts[0]
        
        for i, part in enumerate(self.parts[1:], 1):
            if i-1 < len(self.operators):
                query += f" {self.operators[i-1]} {part}"
            else:
                query += f" {part}"
        
        return query


class ConflictSearchTemplates:
    """Pre-built search templates for conflict monitoring"""
    
    @staticmethod
    def armed_conflict_query(location: Optional[str] = None) -> QueryBuilder:
        """Query for armed conflict events"""
        builder = QueryBuilder()
        builder.add_keyword("armed conflict", FilterOperator.OR)
        builder.add_keyword("military operation", FilterOperator.OR)
        builder.add_keyword("fighting", FilterOperator.OR)
        builder.add_keyword("battle", FilterOperator.OR)
        builder.add_keyword("combat", FilterOperator.OR)
        builder.add_keyword("clashes", FilterOperator.OR)
        
        if location:
            builder.add_location(location)
        
        return builder
    
    @staticmethod
    def refugee_crisis_query(location: Optional[str] = None) -> QueryBuilder:
        """Query for refugee and displacement events"""
        builder = QueryBuilder()
        builder.add_keyword("refugees", FilterOperator.OR)
        builder.add_keyword("displaced", FilterOperator.OR)
        builder.add_keyword("evacuation", FilterOperator.OR)
        builder.add_keyword("humanitarian crisis", FilterOperator.OR)
        builder.add_keyword("displacement", FilterOperator.OR)
        
        if location:
            builder.add_location(location)
        
        return builder
    
    @staticmethod
    def terrorism_query(location: Optional[str] = None) -> QueryBuilder:
        """Query for terrorism-related events"""
        builder = QueryBuilder()
        builder.add_keyword("terrorist attack", FilterOperator.OR)
        builder.add_keyword("bombing", FilterOperator.OR)
        builder.add_keyword("explosion", FilterOperator.OR)
        builder.add_keyword("suicide bomber", FilterOperator.OR)
        builder.add_keyword("terror", FilterOperator.OR)
        
        if location:
            builder.add_location(location)
        
        return builder
    
    @staticmethod
    def protest_unrest_query(location: Optional[str] = None) -> QueryBuilder:
        """Query for protests and civil unrest"""
        builder = QueryBuilder()
        builder.add_keyword("protest", FilterOperator.OR)
        builder.add_keyword("demonstration", FilterOperator.OR)
        builder.add_keyword("riot", FilterOperator.OR)
        builder.add_keyword("unrest", FilterOperator.OR)
        builder.add_keyword("uprising", FilterOperator.OR)
        builder.add_keyword("civil disobedience", FilterOperator.OR)
        
        if location:
            builder.add_location(location)
        
        return builder


class TwitterSearchEngine:
    """Advanced Twitter search engine"""
    
    def __init__(self, crawler: Optional[TwitterCrawler] = None, config: Optional[TwitterConfig] = None):
        if crawler:
            self.crawler = crawler
        elif config:
            self.crawler = TwitterCrawler(config)
        else:
            # Use default configuration
            self.crawler = TwitterCrawler(TwitterConfig())
        
        self.search_history: List[SearchResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def search(self, query: SearchQuery) -> SearchResult:
        """Execute a search query"""
        start_time = datetime.now()
        
        try:
            # Ensure crawler is initialized
            if self.crawler.status.value in ['idle', 'stopped']:
                await self.crawler.initialize()
            
            # Execute search
            tweets = await self._execute_search(query)
            
            # Process results
            processed_tweets = self._process_tweets(tweets, query)
            
            # Create result
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = SearchResult(
                query=query,
                tweets=processed_tweets,
                execution_time=execution_time,
                total_found=len(processed_tweets),
                rate_limit_info=self.crawler.rate_limiter.get_status()
            )
            
            # Store in history
            self.search_history.append(result)
            
            # Limit history size
            if len(self.search_history) > 100:
                self.search_history = self.search_history[-100:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise CrawlerError(f"Search failed: {e}")
    
    async def _execute_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Execute the actual search"""
        all_tweets = []
        remaining_results = query.max_results
        
        while remaining_results > 0:
            batch_size = min(remaining_results, 100)  # Twitter API limit
            
            tweets = await self.crawler.search_tweets(
                query.build_query_string(),
                count=batch_size,
                result_type=query.search_type.value
            )
            
            if not tweets:
                break
            
            all_tweets.extend(tweets)
            remaining_results -= len(tweets)
            
            # Prevent infinite loops
            if len(tweets) < batch_size:
                break
        
        return all_tweets
    
    def _process_tweets(self, tweets: List[Dict[str, Any]], query: SearchQuery) -> List[Dict[str, Any]]:
        """Process and filter tweets based on query options"""
        processed = tweets.copy()
        
        # Deduplicate if requested
        if query.deduplicate:
            seen_ids = set()
            deduplicated = []
            for tweet in processed:
                tweet_id = tweet.get('id')
                if tweet_id and tweet_id not in seen_ids:
                    seen_ids.add(tweet_id)
                    deduplicated.append(tweet)
            processed = deduplicated
        
        # Apply additional filtering
        if query.tweet_filter:
            processed = self._apply_filters(processed, query.tweet_filter)
        
        # Sort if requested
        if query.sort_by:
            processed = self._sort_tweets(processed, query.sort_by)
        
        return processed
    
    def _apply_filters(self, tweets: List[Dict[str, Any]], tweet_filter: TweetFilter) -> List[Dict[str, Any]]:
        """Apply tweet filters"""
        filtered = []
        
        for tweet in tweets:
            if self._tweet_passes_filter(tweet, tweet_filter):
                filtered.append(tweet)
        
        return filtered
    
    def _tweet_passes_filter(self, tweet: Dict[str, Any], tweet_filter: TweetFilter) -> bool:
        """Check if a tweet passes the filter criteria"""
        # Engagement filters
        if tweet_filter.min_retweets and tweet.get('retweet_count', 0) < tweet_filter.min_retweets:
            return False
        
        if tweet_filter.min_likes and tweet.get('favorite_count', 0) < tweet_filter.min_likes:
            return False
        
        if tweet_filter.min_replies and tweet.get('reply_count', 0) < tweet_filter.min_replies:
            return False
        
        # Content type filters
        if tweet_filter.exclude_retweets and tweet.get('is_retweet', False):
            return False
        
        if tweet_filter.exclude_replies and tweet.get('in_reply_to_status_id'):
            return False
        
        # User filters
        user = tweet.get('user', {})
        if tweet_filter.verified_users_only and not user.get('verified', False):
            return False
        
        if tweet_filter.min_followers and user.get('followers_count', 0) < tweet_filter.min_followers:
            return False
        
        if tweet_filter.max_followers and user.get('followers_count', 0) > tweet_filter.max_followers:
            return False
        
        return True
    
    def _sort_tweets(self, tweets: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        """Sort tweets by specified criteria"""
        if sort_by == 'recent':
            return sorted(tweets, key=lambda x: x.get('created_at', ''), reverse=True)
        elif sort_by == 'popular':
            return sorted(tweets, key=lambda x: (
                x.get('retweet_count', 0) +
                x.get('favorite_count', 0) +
                x.get('reply_count', 0)
            ), reverse=True)
        elif sort_by == 'relevance':
            # Simple relevance scoring based on engagement
            return sorted(tweets, key=lambda x: (
                x.get('retweet_count', 0) * 3 +
                x.get('favorite_count', 0) * 2 +
                x.get('reply_count', 0) * 1
            ), reverse=True)
        
        return tweets
    
    async def search_conflict_events(
        self,
        event_type: str,
        location: Optional[str] = None,
        date_range: Optional[DateRange] = None,
        max_results: int = 100
    ) -> SearchResult:
        """Search for specific types of conflict events"""
        # Build query based on event type
        if event_type.lower() == 'armed_conflict':
            builder = ConflictSearchTemplates.armed_conflict_query(location)
        elif event_type.lower() == 'refugee_crisis':
            builder = ConflictSearchTemplates.refugee_crisis_query(location)
        elif event_type.lower() == 'terrorism':
            builder = ConflictSearchTemplates.terrorism_query(location)
        elif event_type.lower() == 'protest':
            builder = ConflictSearchTemplates.protest_unrest_query(location)
        else:
            builder = QueryBuilder().add_keyword(event_type)
            if location:
                builder.add_location(location)
        
        # Create search query
        tweet_filter = TweetFilter(
            date_range=date_range,
            exclude_retweets=True,  # Focus on original content
            verified_users_only=False,
            safe_search=False  # Include all content for conflict monitoring
        )
        
        query = SearchQuery(
            query=builder.build(),
            max_results=max_results,
            search_type=SearchType.RECENT,
            tweet_filter=tweet_filter,
            deduplicate=True,
            sort_by='recent'
        )
        
        return await self.search(query)
    
    def get_search_history(self) -> List[SearchResult]:
        """Get search history"""
        return self.search_history.copy()
    
    def clear_search_history(self):
        """Clear search history"""
        self.search_history.clear()


# Utility functions
async def quick_conflict_search(
    keywords: List[str],
    location: Optional[str] = None,
    max_results: int = 50,
    crawler: Optional[TwitterCrawler] = None
) -> SearchResult:
    """Quick conflict-related search"""
    search_engine = TwitterSearchEngine(crawler)
    
    builder = QueryBuilder()
    for i, keyword in enumerate(keywords):
        operator = FilterOperator.OR if i > 0 else FilterOperator.AND
        builder.add_keyword(keyword, operator)
    
    if location:
        builder.add_location(location)
    
    query = SearchQuery(
        query=builder.build(),
        max_results=max_results,
        search_type=SearchType.RECENT
    )
    
    return await search_engine.search(query)


@dataclass
class AdvancedSearchOptions:
    """Advanced search options for comprehensive Twitter searches."""
    
    # Basic search parameters
    keywords: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    from_users: List[str] = field(default_factory=list)
    to_users: List[str] = field(default_factory=list)
    
    # Content filters
    languages: List[str] = field(default_factory=list)
    search_type: SearchType = SearchType.RECENT
    result_type: str = "recent"  # recent, popular, mixed
    
    # Time filters
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    since_id: Optional[str] = None
    max_id: Optional[str] = None
    
    # Geographic filters
    geo_filter: Optional[GeographicFilter] = None
    
    # Advanced filters
    tweet_filter: Optional[TweetFilter] = None
    include_retweets: bool = True
    include_replies: bool = True
    verified_only: bool = False
    
    # Result limits
    max_results: int = 100
    
    def to_search_query(self) -> SearchQuery:
        """Convert to SearchQuery object."""
        builder = QueryBuilder()
        
        # Add keywords
        for keyword in self.keywords:
            builder.add_keyword(keyword)
        
        # Add hashtags
        for hashtag in self.hashtags:
            builder.add_hashtag(hashtag)
        
        # Add mentions
        for mention in self.mentions:
            builder.add_mention(mention)
        
        # Add user filters
        for user in self.from_users:
            builder.add_from_user(user)
        
        for user in self.to_users:
            builder.add_to_user(user)
        
        # Add language filters
        for lang in self.languages:
            builder.add_filter(f"lang:{lang}")
        
        # Add geographic filter
        if self.geo_filter and self.geo_filter.is_valid():
            geo_query = self.geo_filter.to_query_string()
            if geo_query:
                builder.add_filter(geo_query)
        
        # Build query
        query_string = builder.build()
        
        # Create date range
        date_range = None
        if self.since or self.until:
            date_range = DateRange(start_date=self.since, end_date=self.until)
        
        return SearchQuery(
            query=query_string,
            max_results=self.max_results,
            search_type=self.search_type,
            date_range=date_range,
            tweet_filter=self.tweet_filter
        )


__all__ = [
    'TwitterSearchEngine', 'SearchQuery', 'SearchResult', 'QueryBuilder',
    'TweetFilter', 'GeographicFilter', 'DateRange', 'ConflictSearchTemplates',
    'AdvancedSearchOptions', 'SearchType', 'LanguageCode', 'FilterOperator',
    'quick_conflict_search'
]