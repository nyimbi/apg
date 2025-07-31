"""
Advanced Multi-Engine Search Crawler
====================================

Orchestrates multiple search engines with intelligent ranking, deduplication,
and integration with news_crawler for downloading and parsing results.

Features:
- Multi-engine parallel search with fallback strategies
- Intelligent result ranking and scoring
- Duplicate detection and merging
- Stealth capabilities integration
- Content downloading via news_crawler
- Caching and rate limiting
- Extensible plugin architecture

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from urllib.parse import urlparse, urljoin
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import search engines
from ..engines.base_search_engine import BaseSearchEngine, SearchResult, SearchResponse
from ..engines import SEARCH_ENGINES, get_available_engines, create_engine

# Import news crawler for content downloading
try:
    from ...news_crawler.core.news_crawler import EnhancedNewsCrawler
    from ...news_crawler.stealth.stealth_orchestrator import create_stealth_orchestrator
    NEWS_CRAWLER_AVAILABLE = True
except ImportError:
    NEWS_CRAWLER_AVAILABLE = False
    EnhancedNewsCrawler = None

# Import ranking system
try:
    from ..ranking.result_ranker import ResultRanker, RankingStrategy
    RANKER_AVAILABLE = True
except ImportError:
    RANKER_AVAILABLE = False
    ResultRanker = None

logger = logging.getLogger(__name__)


@dataclass
class SearchCrawlerConfig:
    """Configuration for search crawler."""
    # Engine configuration
    engines: List[str] = field(default_factory=lambda: ['google', 'bing', 'duckduckgo', 'yandex', 'brave', 'startpage'])
    engine_weights: Dict[str, float] = field(default_factory=lambda: {
        'google': 1.0,
        'bing': 0.8,
        'duckduckgo': 0.9,
        'yandex': 0.7,
        'baidu': 0.6,
        'yahoo': 0.7,
        'startpage': 0.8,
        'searx': 0.7,
        'brave': 0.8,
        'mojeek': 0.6,
        'swisscows': 0.6
    })
    
    # Search parameters
    max_results_per_engine: int = 20
    total_max_results: int = 50
    parallel_searches: bool = True
    timeout: float = 30.0
    
    # Ranking configuration
    ranking_strategy: str = 'hybrid'  # 'relevance', 'freshness', 'authority', 'hybrid'
    deduplication_threshold: float = 0.85
    
    # Content downloading
    download_content: bool = True
    parse_content: bool = True
    extract_metadata: bool = True
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Rate limiting
    min_delay_between_searches: float = 1.0
    max_concurrent_downloads: int = 5
    
    # Stealth options
    use_stealth: bool = True
    rotate_user_agents: bool = True
    use_proxies: bool = False
    proxy_list: List[str] = field(default_factory=list)


@dataclass
class EnhancedSearchResult(SearchResult):
    """Enhanced search result with additional metadata."""
    # Multi-engine tracking
    engines_found: List[str] = field(default_factory=list)
    engine_ranks: Dict[str, int] = field(default_factory=dict)
    
    # Content data
    content: Optional[str] = None
    parsed_content: Optional[Dict[str, Any]] = None
    download_time: Optional[float] = None
    
    # Enhanced scoring
    combined_score: float = 0.0
    quality_score: float = 0.0
    freshness_score: float = 0.0
    authority_score: float = 0.0
    
    # Deduplication
    content_hash: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None


class SearchCrawler:
    """Advanced multi-engine search crawler with content downloading."""
    
    def __init__(self, config: Optional[SearchCrawlerConfig] = None):
        """Initialize search crawler with configuration."""
        self.config = config or SearchCrawlerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize search engines
        self.engines: Dict[str, BaseSearchEngine] = {}
        self._initialize_engines()
        
        # Initialize news crawler for content downloading
        self.news_crawler = None
        if NEWS_CRAWLER_AVAILABLE and self.config.download_content:
            self._initialize_news_crawler()
        
        # Initialize ranking system
        self.ranker = None
        if RANKER_AVAILABLE:
            self.ranker = ResultRanker(strategy=self.config.ranking_strategy)
        
        # Cache for search results
        self.cache: Dict[str, Tuple[List[EnhancedSearchResult], datetime]] = {}
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_results': 0,
            'duplicates_found': 0,
            'content_downloaded': 0,
            'content_parsed': 0,
            'average_search_time': 0.0,
            'engine_performance': defaultdict(lambda: {'searches': 0, 'results': 0, 'failures': 0})
        }
        
        # Rate limiting
        self.last_search_time = 0.0
        self.download_semaphore = asyncio.Semaphore(self.config.max_concurrent_downloads)
    
    def _initialize_engines(self):
        """Initialize configured search engines."""
        for engine_name in self.config.engines:
            if engine_name in SEARCH_ENGINES:
                try:
                    engine = create_engine(engine_name)
                    self.engines[engine_name] = engine
                    self.logger.info(f"Initialized {engine_name} search engine")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {engine_name}: {e}")
            else:
                self.logger.warning(f"Unknown search engine: {engine_name}. Available: {get_available_engines()}")
    
    def _initialize_news_crawler(self):
        """Initialize news crawler for content downloading."""
        try:
            # Create stealth configuration
            stealth_config = {
                'use_stealth': self.config.use_stealth,
                'rotate_user_agents': self.config.rotate_user_agents,
                'use_proxies': self.config.use_proxies,
                'proxy_list': self.config.proxy_list
            }
            
            # Initialize news crawler
            self.news_crawler = EnhancedNewsCrawler(
                stealth_config=stealth_config,
                parse_content=self.config.parse_content,
                extract_metadata=self.config.extract_metadata
            )
            
            self.logger.info("Initialized news crawler for content downloading")
        except Exception as e:
            self.logger.error(f"Failed to initialize news crawler: {e}")
            self.news_crawler = None
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        engines: Optional[List[str]] = None,
        download_content: Optional[bool] = None,
        **kwargs
    ) -> List[EnhancedSearchResult]:
        """
        Perform multi-engine search with optional content downloading.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            engines: List of engines to use (None for all)
            download_content: Whether to download page content
            **kwargs: Additional engine-specific parameters
            
        Returns:
            List of enhanced search results
        """
        start_time = time.time()
        max_results = max_results or self.config.total_max_results
        download_content = download_content if download_content is not None else self.config.download_content
        
        # Check cache
        if self.config.enable_cache:
            cached_results = self._get_cached_results(query)
            if cached_results:
                self.logger.info(f"Returning cached results for query: {query}")
                return cached_results[:max_results]
        
        # Rate limiting
        await self._rate_limit()
        
        try:
            # Determine which engines to use
            engines_to_use = engines or list(self.engines.keys())
            engines_to_use = [e for e in engines_to_use if e in self.engines]
            
            # Perform searches
            if self.config.parallel_searches:
                all_results = await self._parallel_search(query, engines_to_use, **kwargs)
            else:
                all_results = await self._sequential_search(query, engines_to_use, **kwargs)
            
            # Merge and deduplicate results
            merged_results = self._merge_results(all_results)
            
            # Rank results
            if self.ranker:
                ranked_results = self.ranker.rank(merged_results, query)
            else:
                ranked_results = self._simple_rank(merged_results)
            
            # Limit results
            final_results = ranked_results[:max_results]
            
            # Download content if requested
            if download_content and self.news_crawler:
                await self._download_content(final_results)
            
            # Cache results
            if self.config.enable_cache:
                self._cache_results(query, final_results)
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_stats(True, search_time, len(final_results))
            
            self.logger.info(
                f"Search completed for '{query}': {len(final_results)} results "
                f"in {search_time:.2f}s"
            )
            
            return final_results
        
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            search_time = time.time() - start_time
            self._update_stats(False, search_time, 0)
            raise
    
    async def _parallel_search(
        self,
        query: str,
        engines: List[str],
        **kwargs
    ) -> Dict[str, List[SearchResult]]:
        """Perform parallel searches across multiple engines."""
        tasks = []
        
        for engine_name in engines:
            engine = self.engines[engine_name]
            task = asyncio.create_task(
                self._search_with_engine(engine_name, engine, query, **kwargs)
            )
            tasks.append((engine_name, task))
        
        # Wait for all searches with timeout
        results = {}
        done, pending = await asyncio.wait(
            [task for _, task in tasks],
            timeout=self.config.timeout
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Collect results
        for engine_name, task in tasks:
            if task in done:
                try:
                    response = await task
                    if response and response.success:
                        results[engine_name] = response.results
                        self.stats['engine_performance'][engine_name]['results'] += len(response.results)
                    else:
                        self.stats['engine_performance'][engine_name]['failures'] += 1
                except Exception as e:
                    self.logger.error(f"Error getting results from {engine_name}: {e}")
                    self.stats['engine_performance'][engine_name]['failures'] += 1
        
        return results
    
    async def _sequential_search(
        self,
        query: str,
        engines: List[str],
        **kwargs
    ) -> Dict[str, List[SearchResult]]:
        """Perform sequential searches across engines."""
        results = {}
        
        for engine_name in engines:
            engine = self.engines[engine_name]
            try:
                response = await self._search_with_engine(engine_name, engine, query, **kwargs)
                if response and response.success:
                    results[engine_name] = response.results
                    self.stats['engine_performance'][engine_name]['results'] += len(response.results)
                else:
                    self.stats['engine_performance'][engine_name]['failures'] += 1
            except Exception as e:
                self.logger.error(f"Sequential search failed for {engine_name}: {e}")
                self.stats['engine_performance'][engine_name]['failures'] += 1
        
        return results
    
    async def _search_with_engine(
        self,
        engine_name: str,
        engine: BaseSearchEngine,
        query: str,
        **kwargs
    ) -> Optional[SearchResponse]:
        """Search with a specific engine."""
        try:
            self.stats['engine_performance'][engine_name]['searches'] += 1
            
            response = await engine.search(
                query=query,
                max_results=self.config.max_results_per_engine,
                **kwargs
            )
            
            return response
        
        except Exception as e:
            self.logger.error(f"Search failed with {engine_name}: {e}")
            return None
    
    def _merge_results(self, engine_results: Dict[str, List[SearchResult]]) -> List[EnhancedSearchResult]:
        """Merge and deduplicate results from multiple engines."""
        url_to_result: Dict[str, EnhancedSearchResult] = {}
        
        for engine_name, results in engine_results.items():
            weight = self.config.engine_weights.get(engine_name, 1.0)
            
            for rank, result in enumerate(results, 1):
                url = self._normalize_url(result.url)
                
                if url in url_to_result:
                    # Update existing result
                    enhanced = url_to_result[url]
                    enhanced.engines_found.append(engine_name)
                    enhanced.engine_ranks[engine_name] = rank
                    
                    # Update relevance score (average weighted by engine importance)
                    enhanced.relevance_score = (
                        enhanced.relevance_score + (weight / rank)
                    ) / 2
                else:
                    # Create new enhanced result
                    enhanced = EnhancedSearchResult(
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                        engine=engine_name,
                        rank=rank,
                        timestamp=result.timestamp,
                        metadata=result.metadata,
                        relevance_score=weight / rank,
                        engines_found=[engine_name],
                        engine_ranks={engine_name: rank}
                    )
                    url_to_result[url] = enhanced
        
        # Detect duplicates by content similarity
        all_results = list(url_to_result.values())
        self._detect_duplicates(all_results)
        
        return all_results
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url.lower())
        
        # Remove common variations
        normalized = f"{parsed.scheme or 'https'}://{parsed.netloc}{parsed.path}"
        
        # Remove trailing slashes
        normalized = normalized.rstrip('/')
        
        # Remove common parameters
        if parsed.query:
            # Keep only important parameters
            important_params = ['id', 'article', 'post', 'p', 'page']
            params = [p for p in parsed.query.split('&') 
                     if any(p.startswith(ip) for ip in important_params)]
            if params:
                normalized += '?' + '&'.join(sorted(params))
        
        return normalized
    
    def _detect_duplicates(self, results: List[EnhancedSearchResult]):
        """Detect duplicate results based on content similarity."""
        if len(results) < 2:
            return
        
        # Extract text for similarity comparison
        texts = []
        for result in results:
            text = f"{result.title} {result.snippet}"
            texts.append(text)
            
            # Generate content hash
            result.content_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Use TF-IDF for similarity detection
        try:
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Mark duplicates
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    if similarities[i, j] >= self.config.deduplication_threshold:
                        # Mark the lower-ranked result as duplicate
                        if results[i].relevance_score > results[j].relevance_score:
                            results[j].is_duplicate = True
                            results[j].duplicate_of = results[i].url
                        else:
                            results[i].is_duplicate = True
                            results[i].duplicate_of = results[j].url
                        
                        self.stats['duplicates_found'] += 1
        
        except Exception as e:
            self.logger.error(f"Duplicate detection failed: {e}")
    
    def _simple_rank(self, results: List[EnhancedSearchResult]) -> List[EnhancedSearchResult]:
        """Simple ranking based on relevance score and engine presence."""
        # Filter out duplicates
        unique_results = [r for r in results if not r.is_duplicate]
        
        # Calculate combined score
        for result in unique_results:
            # Base score from relevance
            result.combined_score = result.relevance_score
            
            # Bonus for appearing in multiple engines
            engine_bonus = len(result.engines_found) * 0.1
            result.combined_score += engine_bonus
            
            # Penalty for lower ranks
            avg_rank = sum(result.engine_ranks.values()) / len(result.engine_ranks)
            rank_penalty = 1.0 / (1.0 + avg_rank * 0.1)
            result.combined_score *= rank_penalty
        
        # Sort by combined score
        ranked = sorted(unique_results, key=lambda x: x.combined_score, reverse=True)
        
        return ranked
    
    async def _download_content(self, results: List[EnhancedSearchResult]):
        """Download and parse content for search results."""
        if not self.news_crawler:
            return
        
        download_tasks = []
        
        for result in results:
            if not result.is_duplicate:
                task = asyncio.create_task(
                    self._download_single_result(result)
                )
                download_tasks.append(task)
        
        # Wait for all downloads
        if download_tasks:
            await asyncio.gather(*download_tasks, return_exceptions=True)
    
    async def _download_single_result(self, result: EnhancedSearchResult):
        """Download content for a single result."""
        async with self.download_semaphore:
            try:
                start_time = time.time()
                
                # Download and parse content
                article_data = await self.news_crawler.crawl_article(result.url)
                
                if article_data and article_data.get('success'):
                    result.content = article_data.get('content', '')
                    result.parsed_content = article_data
                    result.download_time = time.time() - start_time
                    
                    # Update metadata
                    if 'metadata' in article_data:
                        result.metadata.update(article_data['metadata'])
                    
                    self.stats['content_downloaded'] += 1
                    
                    if article_data.get('parsed'):
                        self.stats['content_parsed'] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to download content for {result.url}: {e}")
    
    async def _rate_limit(self):
        """Apply rate limiting between searches."""
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        
        if time_since_last < self.config.min_delay_between_searches:
            sleep_time = self.config.min_delay_between_searches - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_search_time = time.time()
    
    def _get_cached_results(self, query: str) -> Optional[List[EnhancedSearchResult]]:
        """Get cached results if available and not expired."""
        if query in self.cache:
            results, timestamp = self.cache[query]
            
            # Check if cache is still valid
            if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl):
                return results
            else:
                # Remove expired cache
                del self.cache[query]
        
        return None
    
    def _cache_results(self, query: str, results: List[EnhancedSearchResult]):
        """Cache search results."""
        self.cache[query] = (results, datetime.now())
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self.cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:100]:
                del self.cache[key]
    
    def _update_stats(self, success: bool, search_time: float, num_results: int):
        """Update crawler statistics."""
        self.stats['total_searches'] += 1
        
        if success:
            self.stats['successful_searches'] += 1
            self.stats['total_results'] += num_results
        else:
            self.stats['failed_searches'] += 1
        
        # Update average search time
        total_successful = self.stats['successful_searches']
        if total_successful > 0:
            current_avg = self.stats['average_search_time']
            self.stats['average_search_time'] = (
                (current_avg * (total_successful - 1) + search_time) / total_successful
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crawler statistics."""
        return {
            **self.stats,
            'engines': {
                name: engine.get_stats() 
                for name, engine in self.engines.items()
            },
            'cache_size': len(self.cache),
            'success_rate': (
                self.stats['successful_searches'] / max(self.stats['total_searches'], 1)
            )
        }
    
    def clear_cache(self):
        """Clear the results cache."""
        self.cache.clear()
        self.logger.info("Cleared search results cache")
    
    async def close(self):
        """Clean up resources."""
        if self.news_crawler:
            await self.news_crawler.close()
        
        self.logger.info("Search crawler closed")


# Convenience function
def create_search_crawler(config: Optional[Dict[str, Any]] = None) -> SearchCrawler:
    """Create a search crawler with configuration."""
    if config:
        crawler_config = SearchCrawlerConfig(**config)
    else:
        crawler_config = SearchCrawlerConfig()
    
    return SearchCrawler(crawler_config)