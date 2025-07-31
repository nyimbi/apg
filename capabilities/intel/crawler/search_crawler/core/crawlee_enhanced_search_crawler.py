#!/usr/bin/env python3
"""
Crawlee-Enhanced Search Crawler
===============================

Advanced multi-engine search crawler enhanced with Crawlee for robust content downloading
and parsing of search result URLs. Combines the comprehensive search capabilities of the
search_crawler with the powerful content extraction features of Crawlee.

Key Features:
- **Multi-Engine Search**: Orchestrates 11+ search engines simultaneously
- **Crawlee Integration**: Uses AdaptivePlaywrightCrawler for robust content downloading
- **Intelligent Content Extraction**: Multi-method content parsing with quality scoring
- **Advanced Deduplication**: Content-based similarity detection
- **Performance Optimization**: Concurrent downloads with rate limiting
- **Conflict-Focused Analysis**: Specialized for Horn of Africa conflict monitoring
- **Comprehensive Statistics**: Detailed performance and quality metrics

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import time
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, AsyncIterator, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import urlparse, urljoin
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import base search crawler
from .search_crawler import SearchCrawler, SearchCrawlerConfig, EnhancedSearchResult

# Import search engines
from ..engines.base_search_engine import BaseSearchEngine, SearchResult, SearchResponse
from ..engines import SEARCH_ENGINES, get_available_engines, create_engine

# Crawlee imports
try:
    from crawlee import AsyncAdaptivePlaywrightCrawler, AsyncBeautifulSoupCrawler
    from crawlee.types import RequestQueue, Request, Dataset
    from crawlee.basic_crawler import Context as CrawleeContext
    CRAWLEE_AVAILABLE = True
except ImportError:
    CRAWLEE_AVAILABLE = False
    AsyncAdaptivePlaywrightCrawler = None
    AsyncBeautifulSoupCrawler = None
    RequestQueue = None
    Request = None
    Dataset = None
    CrawleeContext = None

# Content parsing imports
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    Article = None
    NEWSPAPER_AVAILABLE = False

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    Document = None
    READABILITY_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CrawleeSearchConfig(SearchCrawlerConfig):
    """Extended configuration for Crawlee-enhanced search crawler."""
    
    # Crawlee-specific settings
    max_requests_per_crawl: int = 100
    request_handler_timeout: int = 60
    navigation_timeout: int = 30
    browser_type: str = 'chromium'  # chromium, firefox, webkit
    headless: bool = True
    
    # Content extraction settings
    preferred_extraction_method: str = "auto"  # auto, trafilatura, newspaper, readability, bs4
    enable_image_extraction: bool = True
    enable_link_extraction: bool = True
    enable_metadata_enhancement: bool = True
    
    # Quality filters
    min_content_length: int = 200
    max_content_length: int = 100000
    enable_content_scoring: bool = True
    min_quality_score: float = 0.3
    
    # Performance tuning
    crawl_delay: float = 1.0
    max_retries: int = 3
    respect_robots_txt: bool = True
    
    # Storage settings
    save_raw_html: bool = False
    enable_result_caching: bool = True
    cache_directory: Optional[Path] = None
    
    # Geographic and language targeting
    target_countries: List[str] = field(default_factory=lambda: ["ET", "SO", "ER", "DJ", "KE", "UG", "TZ", "SD", "SS"])
    target_languages: List[str] = field(default_factory=lambda: ["en", "fr", "ar", "sw"])


@dataclass
class CrawleeEnhancedResult(EnhancedSearchResult):
    """Enhanced search result with Crawlee-extracted content."""
    
    # Crawlee-specific content
    extracted_content: Optional[str] = None
    content_summary: Optional[str] = None
    extraction_method: str = "unknown"
    extraction_score: float = 0.0
    
    # Enhanced metadata
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    
    # Content analysis
    word_count: int = 0
    reading_time_minutes: float = 0.0
    language: str = "en"
    
    # Geographic and topical relevance
    geographic_entities: List[str] = field(default_factory=list)
    conflict_indicators: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    
    # Processing metadata
    crawl_timestamp: Optional[datetime] = None
    crawl_success: bool = False
    crawl_error: Optional[str] = None
    processing_time_ms: float = 0.0
    
    # Quality indicators
    content_quality_score: float = 0.0
    information_density: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        base_dict = asdict(self)
        
        # Add computed fields
        base_dict.update({
            'id': hashlib.sha256(self.url.encode()).hexdigest()[:16],
            'is_conflict_related': len(self.conflict_indicators) > 0,
            'geographic_relevance': len(self.geographic_entities) > 0,
            'has_full_content': bool(self.extracted_content),
            'content_length': len(self.extracted_content or ''),
            'metadata_completeness': self._calculate_metadata_completeness(),
            'overall_quality': (self.extraction_score + self.content_quality_score + self.relevance_score) / 3
        })
        
        return base_dict
    
    def _calculate_metadata_completeness(self) -> float:
        """Calculate metadata completeness score."""
        fields = [
            bool(self.authors),
            bool(self.keywords),
            bool(self.images),
            self.published_date is not None,
            bool(self.extracted_content),
            self.word_count > 0
        ]
        return sum(fields) / len(fields)


class CrawleeEnhancedSearchCrawler(SearchCrawler):
    """Enhanced search crawler with Crawlee integration for content downloading."""
    
    def __init__(self, config: Optional[CrawleeSearchConfig] = None):
        """Initialize Crawlee-enhanced search crawler."""
        self.crawlee_config = config or CrawleeSearchConfig()
        super().__init__(self.crawlee_config)
        
        # Crawlee components
        self.crawler = None
        self.request_queue = None
        self.dataset = None
        
        # Content extractors availability
        self.extractors_available = {
            'trafilatura': TRAFILATURA_AVAILABLE,
            'newspaper': NEWSPAPER_AVAILABLE,
            'readability': READABILITY_AVAILABLE,
            'beautifulsoup': BS4_AVAILABLE
        }
        
        # Enhanced statistics
        self.crawlee_stats = {
            'total_content_requests': 0,
            'successful_content_extractions': 0,
            'failed_content_extractions': 0,
            'content_extraction_time': 0.0,
            'average_content_length': 0.0,
            'quality_filter_passed': 0,
            'quality_filter_failed': 0,
            'extraction_method_usage': {method: 0 for method in self.extractors_available.keys()}
        }
        
        logger.info(f"CrawleeEnhancedSearchCrawler initialized with {sum(self.extractors_available.values())} content extractors")
    
    async def initialize_crawlee(self):
        """Initialize Crawlee components."""
        if not CRAWLEE_AVAILABLE:
            raise ImportError("Crawlee not available. Install with: pip install crawlee")
        
        try:
            # Initialize Crawlee crawler with optimized settings
            self.crawler = AsyncAdaptivePlaywrightCrawler(
                max_requests_per_crawl=self.crawlee_config.max_requests_per_crawl,
                request_handler_timeout_secs=self.crawlee_config.request_handler_timeout,
                navigation_timeout_secs=self.crawlee_config.navigation_timeout,
                max_request_retries=self.crawlee_config.max_retries,
                browser_type=self.crawlee_config.browser_type,
                headless=self.crawlee_config.headless
            )
            
            # Initialize request queue and dataset
            self.request_queue = await RequestQueue.open()
            self.dataset = await Dataset.open()
            
            # Set up request handler for content extraction
            @self.crawler.router.default_handler
            async def handle_content_extraction(context: CrawleeContext):
                await self._handle_content_extraction(context)
            
            logger.info("✅ Crawlee components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Crawlee components: {e}")
            raise
    
    async def search_with_content(
        self,
        query: str,
        max_results: Optional[int] = None,
        engines: Optional[List[str]] = None,
        extract_content: bool = True,
        **kwargs
    ) -> List[CrawleeEnhancedResult]:
        """
        Perform multi-engine search with full content extraction.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            engines: List of engines to use (None for all)
            extract_content: Whether to extract full content using Crawlee
            **kwargs: Additional engine-specific parameters
            
        Returns:
            List of Crawlee-enhanced search results
        """
        start_time = time.time()
        
        # Perform initial search using base crawler
        logger.info(f"🔍 Starting enhanced search for: '{query}'")
        basic_results = await super().search(
            query=query,
            max_results=max_results,
            engines=engines,
            download_content=False,  # We'll handle content with Crawlee
            **kwargs
        )
        
        # Convert to enhanced results
        enhanced_results = [
            self._convert_to_enhanced_result(result)
            for result in basic_results
        ]
        
        # Extract content with Crawlee if requested
        if extract_content and self.crawler:
            await self._extract_content_with_crawlee(enhanced_results)
        
        # Apply quality filtering and ranking
        final_results = self._apply_quality_filtering(enhanced_results)
        final_results = self._apply_enhanced_ranking(final_results, query)
        
        # Update statistics
        search_time = time.time() - start_time
        self._update_crawlee_stats(final_results, search_time)
        
        logger.info(
            f"✅ Enhanced search completed: {len(final_results)} results "
            f"with content in {search_time:.2f}s"
        )
        
        return final_results
    
    def _convert_to_enhanced_result(self, result: EnhancedSearchResult) -> CrawleeEnhancedResult:
        """Convert basic enhanced result to Crawlee-enhanced result."""
        return CrawleeEnhancedResult(
            # Copy all fields from base result
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            engine=result.engine,
            rank=result.rank,
            timestamp=result.timestamp,
            metadata=result.metadata,
            relevance_score=result.relevance_score,
            engines_found=result.engines_found,
            engine_ranks=result.engine_ranks,
            content=result.content,
            parsed_content=result.parsed_content,
            download_time=result.download_time,
            combined_score=result.combined_score,
            quality_score=result.quality_score,
            freshness_score=result.freshness_score,
            authority_score=result.authority_score,
            content_hash=result.content_hash,
            is_duplicate=result.is_duplicate,
            duplicate_of=result.duplicate_of,
            
            # Initialize Crawlee-specific fields
            crawl_timestamp=datetime.now(timezone.utc),
            crawl_success=False
        )
    
    async def _extract_content_with_crawlee(self, results: List[CrawleeEnhancedResult]):
        """Extract full content for results using Crawlee."""
        if not self.crawler:
            logger.warning("Crawlee not initialized - skipping content extraction")
            return
        
        logger.info(f"🕷️ Starting Crawlee content extraction for {len(results)} results")
        
        # Store results for access during crawling
        self._current_results = {result.url: result for result in results}
        
        try:
            # Add URLs to request queue
            for result in results:
                if not result.is_duplicate:
                    request = Request.from_url(
                        result.url,
                        user_data={
                            'original_result': result.url,
                            'search_metadata': {
                                'title': result.title,
                                'snippet': result.snippet,
                                'engines': result.engines_found
                            }
                        }
                    )
                    await self.request_queue.add_request(request)
            
            # Run the crawler
            await self.crawler.run()
            
            # Clean up temporary storage
            self._current_results = {}
            
            logger.info("✅ Crawlee content extraction completed")
            
        except Exception as e:
            logger.error(f"Crawlee content extraction failed: {e}")
            raise
    
    async def _handle_content_extraction(self, context: CrawleeContext):
        """Handle individual URL content extraction."""
        start_time = time.time()
        url = context.request.loaded_url
        original_url = context.request.user_data.get('original_result')
        
        # Find the corresponding result
        result = self._current_results.get(original_url)
        if not result:
            logger.warning(f"No result found for URL: {url}")
            return
        
        try:
            logger.debug(f"Extracting content from: {url}")
            
            # Get page HTML
            html = await context.page.content()
            
            # Extract content using multiple methods
            content_data = await self._extract_content_multi_method(url, html)
            
            # Update result with extracted content
            self._update_result_with_content(result, content_data, start_time)
            
            # Store extracted data
            await self.dataset.push_data({
                'url': url,
                'extraction_data': content_data,
                'result_metadata': result.to_dict()
            })
            
            self.crawlee_stats['successful_content_extractions'] += 1
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            result.crawl_error = str(e)
            result.crawl_success = False
            self.crawlee_stats['failed_content_extractions'] += 1
        
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
            self.crawlee_stats['total_content_requests'] += 1
    
    async def _extract_content_multi_method(self, url: str, html: str) -> Dict[str, Any]:
        """Extract content using multiple methods with fallback."""
        extraction_results = {}
        best_result = None
        best_score = 0.0
        
        # Get extraction methods to try
        methods_to_try = self._get_extraction_methods()
        
        for method in methods_to_try:
            try:
                result = await self._extract_with_method(method, url, html)
                if result and result.get('content'):
                    score = self._score_extraction_result(result)
                    extraction_results[method] = {**result, 'score': score}
                    
                    # Track method usage
                    self.crawlee_stats['extraction_method_usage'][method] += 1
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_result['method'] = method
                        
            except Exception as e:
                logger.debug(f"Extraction method {method} failed for {url}: {e}")
                extraction_results[method] = {'error': str(e), 'score': 0.0}
        
        # Return best result or create minimal result
        if best_result:
            best_result['all_attempts'] = extraction_results
            return best_result
        else:
            return {
                'content': '',
                'title': '',
                'method': 'failed',
                'score': 0.0,
                'all_attempts': extraction_results
            }
    
    def _get_extraction_methods(self) -> List[str]:
        """Get list of extraction methods to try based on configuration and availability."""
        preferred = self.crawlee_config.preferred_extraction_method
        
        if preferred != "auto" and self.extractors_available.get(preferred):
            return [preferred]
        
        # Auto mode: try methods in order of effectiveness
        methods = []
        if self.extractors_available['trafilatura']:
            methods.append('trafilatura')
        if self.extractors_available['newspaper']:
            methods.append('newspaper')
        if self.extractors_available['readability']:
            methods.append('readability')
        if self.extractors_available['beautifulsoup']:
            methods.append('beautifulsoup')
        
        return methods or ['fallback']
    
    async def _extract_with_method(self, method: str, url: str, html: str) -> Dict[str, Any]:
        """Extract content using a specific method."""
        if method == 'trafilatura' and TRAFILATURA_AVAILABLE:
            return await self._extract_with_trafilatura(url, html)
        elif method == 'newspaper' and NEWSPAPER_AVAILABLE:
            return await self._extract_with_newspaper(url, html)
        elif method == 'readability' and READABILITY_AVAILABLE:
            return await self._extract_with_readability(url, html)
        elif method == 'beautifulsoup' and BS4_AVAILABLE:
            return await self._extract_with_beautifulsoup(url, html)
        else:
            raise ValueError(f"Extraction method {method} not available")
    
    async def _extract_with_trafilatura(self, url: str, html: str) -> Dict[str, Any]:
        """Extract content using Trafilatura."""
        try:
            # Extract main content
            content = trafilatura.extract(
                html, 
                include_images=self.crawlee_config.enable_image_extraction,
                include_links=self.crawlee_config.enable_link_extraction
            )
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html)
            
            # Extract additional elements if enabled
            images = []
            links = []
            if self.crawlee_config.enable_image_extraction:
                # Basic image extraction from HTML
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(html, 'html.parser')
                    img_tags = soup.find_all('img', src=True)
                    images = [{'url': img['src'], 'alt': img.get('alt', '')} for img in img_tags[:10]]
            
            if self.crawlee_config.enable_link_extraction:
                # Basic link extraction from HTML
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(html, 'html.parser')
                    link_tags = soup.find_all('a', href=True)
                    links = [link['href'] for link in link_tags[:20]]
            
            result = {
                'content': content or '',
                'title': metadata.title if metadata else '',
                'authors': [metadata.author] if metadata and metadata.author else [],
                'published_date': metadata.date if metadata else None,
                'description': metadata.description if metadata else '',
                'keywords': [],
                'images': images,
                'links': links,
                'word_count': len(content.split()) if content else 0,
                'language': metadata.language if metadata else 'en'
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed: {e}")
            raise
    
    async def _extract_with_newspaper(self, url: str, html: str) -> Dict[str, Any]:
        """Extract content using Newspaper3k."""
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            result = {
                'content': article.text or '',
                'title': article.title or '',
                'authors': article.authors or [],
                'published_date': article.publish_date,
                'description': article.meta_description or '',
                'keywords': article.keywords or [],
                'images': [{'url': img} for img in article.images] if article.images else [],
                'links': [],
                'word_count': len(article.text.split()) if article.text else 0,
                'language': article.meta_lang or 'en'
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"Newspaper extraction failed: {e}")
            raise
    
    async def _extract_with_readability(self, url: str, html: str) -> Dict[str, Any]:
        """Extract content using Readability."""
        try:
            doc = Document(html)
            content = doc.summary()
            title = doc.title()
            
            # Parse content to get text
            if BS4_AVAILABLE:
                soup = BeautifulSoup(content, 'html.parser')
                text_content = soup.get_text(strip=True)
                
                # Extract images and links if enabled
                images = []
                links = []
                if self.crawlee_config.enable_image_extraction:
                    img_tags = soup.find_all('img', src=True)
                    images = [{'url': img['src'], 'alt': img.get('alt', '')} for img in img_tags]
                
                if self.crawlee_config.enable_link_extraction:
                    link_tags = soup.find_all('a', href=True)
                    links = [link['href'] for link in link_tags]
            else:
                text_content = content
                images = []
                links = []
            
            result = {
                'content': text_content or '',
                'title': title or '',
                'authors': [],
                'published_date': None,
                'description': '',
                'keywords': [],
                'images': images,
                'links': links,
                'word_count': len(text_content.split()) if text_content else 0,
                'language': 'en'
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"Readability extraction failed: {e}")
            raise
    
    async def _extract_with_beautifulsoup(self, url: str, html: str) -> Dict[str, Any]:
        """Extract content using BeautifulSoup (fallback method)."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ''
            
            # Extract main content using common article selectors
            content_selectors = [
                'article', '.article-content', '.content', '.post-content',
                '.entry-content', 'main', '[role="main"]', '.article-body',
                '.story-body', '.article-text'
            ]
            
            content_text = ''
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text(strip=True, separator=' ')
                    break
            
            # Fallback to body if no content found
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(strip=True, separator=' ')
            
            # Extract images and links if enabled
            images = []
            links = []
            if self.crawlee_config.enable_image_extraction:
                img_tags = soup.find_all('img', src=True)
                images = [{'url': img['src'], 'alt': img.get('alt', '')} for img in img_tags[:10]]
            
            if self.crawlee_config.enable_link_extraction:
                link_tags = soup.find_all('a', href=True)
                links = [link['href'] for link in link_tags[:20]]
            
            result = {
                'content': content_text or '',
                'title': title_text,
                'authors': [],
                'published_date': None,
                'description': '',
                'keywords': [],
                'images': images,
                'links': links,
                'word_count': len(content_text.split()) if content_text else 0,
                'language': 'en'
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
            raise
    
    def _score_extraction_result(self, result: Dict[str, Any]) -> float:
        """Score the quality of an extraction result."""
        score = 0.0
        
        # Content length scoring (0.0 - 0.5)
        content = result.get('content', '')
        word_count = len(content.split()) if content else 0
        
        if word_count >= 500:
            score += 0.5
        elif word_count >= 300:
            score += 0.4
        elif word_count >= 150:
            score += 0.3
        elif word_count >= 50:
            score += 0.2
        elif word_count >= 20:
            score += 0.1
        
        # Title quality (0.0 - 0.2)
        title = result.get('title', '')
        if title and len(title) > 10:
            score += 0.2
        elif title and len(title) > 5:
            score += 0.1
        
        # Metadata presence (0.0 - 0.3)
        if result.get('authors'):
            score += 0.1
        if result.get('published_date'):
            score += 0.1
        if result.get('keywords'):
            score += 0.05
        if result.get('description'):
            score += 0.05
        
        return min(score, 1.0)
    
    def _update_result_with_content(
        self,
        result: CrawleeEnhancedResult,
        content_data: Dict[str, Any],
        start_time: float
    ):
        """Update result with extracted content data."""
        result.extracted_content = content_data.get('content', '')
        result.extraction_method = content_data.get('method', 'unknown')
        result.extraction_score = content_data.get('score', 0.0)
        
        # Update enhanced metadata
        result.authors = content_data.get('authors', [])
        result.keywords = content_data.get('keywords', [])
        result.images = content_data.get('images', [])
        result.links = content_data.get('links', [])
        
        # Update content analysis
        result.word_count = content_data.get('word_count', 0)
        result.reading_time_minutes = result.word_count / 200  # Assume 200 WPM
        result.language = content_data.get('language', 'en')
        
        # Extract geographic and conflict indicators
        content = result.extracted_content
        result.geographic_entities = self._extract_geographic_entities(content)
        result.conflict_indicators = self._extract_conflict_indicators(content)
        result.relevance_score = self._calculate_relevance_score(
            content, result.geographic_entities, result.conflict_indicators
        )
        
        # Calculate content quality metrics
        result.content_quality_score = self._calculate_content_quality(result)
        result.information_density = self._calculate_information_density(content)
        
        # Update processing metadata
        result.crawl_timestamp = datetime.now(timezone.utc)
        result.crawl_success = bool(content)
        result.processing_time_ms = (time.time() - start_time) * 1000
    
    def _extract_geographic_entities(self, content: str) -> List[str]:
        """Extract geographic entities from content."""
        if not content:
            return []
        
        entities = []
        
        # Horn of Africa and related geographic terms
        geographic_terms = [
            # Countries
            'Ethiopia', 'Kenya', 'Somalia', 'Sudan', 'South Sudan', 'Uganda',
            'Tanzania', 'Eritrea', 'Djibouti', 'Rwanda', 'Burundi', 'DRC',
            'Democratic Republic of Congo', 'Chad', 'Central African Republic',
            
            # Major cities
            'Addis Ababa', 'Nairobi', 'Mogadishu', 'Khartoum', 'Kampala',
            'Dar es Salaam', 'Asmara', 'Djibouti City', 'Kigali', 'Bujumbura',
            'Kinshasa', 'Juba', 'N\'Djamena',
            
            # Regions
            'Horn of Africa', 'East Africa', 'Tigray', 'Oromia', 'Amhara',
            'Somaliland', 'Puntland', 'Darfur', 'Blue Nile', 'South Kordofan',
            'Kassala', 'Red Sea State', 'Northern State', 'Gezira'
        ]
        
        content_lower = content.lower()
        for term in geographic_terms:
            if term.lower() in content_lower:
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_conflict_indicators(self, content: str) -> List[str]:
        """Extract conflict-related indicators from content."""
        if not content:
            return []
        
        indicators = []
        conflict_terms = [
            # Direct violence
            'conflict', 'violence', 'war', 'battle', 'fighting', 'attack',
            'bombing', 'shooting', 'killing', 'murder', 'assassination',
            'massacre', 'genocide', 'ethnic cleansing',
            
            # Security and military
            'military', 'army', 'soldiers', 'forces', 'troops', 'militia',
            'rebels', 'insurgents', 'guerrilla', 'armed groups',
            'peacekeeping', 'peacekeepers', 'ceasefire', 'truce',
            
            # Terrorism and extremism
            'terrorism', 'terrorist', 'extremist', 'al-shabaab', 'boko haram',
            'isis', 'al-qaeda', 'jihadist', 'militant',
            
            # Political instability
            'coup', 'overthrow', 'regime change', 'government crisis',
            'political crisis', 'election violence', 'protest', 'demonstration',
            'riot', 'unrest', 'uprising', 'revolution',
            
            # Humanitarian issues
            'refugee', 'displaced', 'displacement', 'humanitarian crisis',
            'humanitarian aid', 'famine', 'drought', 'food insecurity',
            'emergency', 'crisis', 'disaster',
            
            # International involvement
            'UN', 'United Nations', 'African Union', 'IGAD', 'sanctions',
            'intervention', 'mediation', 'diplomatic'
        ]
        
        content_lower = content.lower()
        for term in conflict_terms:
            if term in content_lower:
                indicators.append(term)
        
        return list(set(indicators))  # Remove duplicates
    
    def _calculate_relevance_score(
        self,
        content: str,
        geographic_entities: List[str],
        conflict_indicators: List[str]
    ) -> float:
        """Calculate relevance score based on geographic and conflict indicators."""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Geographic relevance (0.0 - 0.4)
        if geographic_entities:
            # Bonus for Horn of Africa countries
            hoa_countries = ['Ethiopia', 'Kenya', 'Somalia', 'Sudan', 'South Sudan', 'Eritrea', 'Djibouti']
            hoa_mentions = sum(1 for entity in geographic_entities if entity in hoa_countries)
            score += min(hoa_mentions * 0.1, 0.3)
            
            # General geographic bonus
            score += min(len(geographic_entities) * 0.02, 0.1)
        
        # Conflict relevance (0.0 - 0.4)
        if conflict_indicators:
            # High-priority conflict terms
            high_priority = ['violence', 'attack', 'bombing', 'killing', 'conflict', 'war']
            high_priority_mentions = sum(1 for indicator in conflict_indicators if indicator in high_priority)
            score += min(high_priority_mentions * 0.1, 0.3)
            
            # General conflict bonus
            score += min(len(conflict_indicators) * 0.02, 0.1)
        
        # Recency bonus (0.0 - 0.2)
        recent_terms = ['today', 'yesterday', 'breaking', 'latest', 'urgent', 'developing']
        content_lower = content.lower()
        recency_mentions = sum(1 for term in recent_terms if term in content_lower)
        score += min(recency_mentions * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_content_quality(self, result: CrawleeEnhancedResult) -> float:
        """Calculate overall content quality score."""
        score = 0.0
        
        # Word count quality (0.0 - 0.3)
        if result.word_count >= 1000:
            score += 0.3
        elif result.word_count >= 500:
            score += 0.25
        elif result.word_count >= 200:
            score += 0.2
        elif result.word_count >= 100:
            score += 0.1
        
        # Metadata completeness (0.0 - 0.3)
        metadata_score = result._calculate_metadata_completeness()
        score += metadata_score * 0.3
        
        # Extraction quality (0.0 - 0.2)
        score += result.extraction_score * 0.2
        
        # Structure indicators (0.0 - 0.2)
        content = result.extracted_content or ''
        if len(content.split('\n')) > 5:  # Multiple paragraphs
            score += 0.1
        if any(word in content.lower() for word in ['quote', 'said', 'according to']):  # Has quotes
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density score."""
        if not content:
            return 0.0
        
        # Simple heuristic based on unique words vs total words
        words = content.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        density = len(unique_words) / len(words)
        
        return min(density * 2, 1.0)  # Normalize to 0-1 range
    
    def _apply_quality_filtering(self, results: List[CrawleeEnhancedResult]) -> List[CrawleeEnhancedResult]:
        """Apply quality filtering to results."""
        if not self.crawlee_config.enable_content_scoring:
            return results
        
        filtered_results = []
        for result in results:
            # Apply content length filters
            content_length = len(result.extracted_content or '')
            if content_length < self.crawlee_config.min_content_length:
                self.crawlee_stats['quality_filter_failed'] += 1
                continue
            
            if content_length > self.crawlee_config.max_content_length:
                self.crawlee_stats['quality_filter_failed'] += 1
                continue
            
            # Apply quality score filter
            if result.extraction_score < self.crawlee_config.min_quality_score:
                self.crawlee_stats['quality_filter_failed'] += 1
                continue
            
            filtered_results.append(result)
            self.crawlee_stats['quality_filter_passed'] += 1
        
        logger.info(f"Quality filtering: {len(filtered_results)}/{len(results)} results passed")
        return filtered_results
    
    def _apply_enhanced_ranking(
        self,
        results: List[CrawleeEnhancedResult],
        query: str
    ) -> List[CrawleeEnhancedResult]:
        """Apply enhanced ranking considering content quality and relevance."""
        for result in results:
            # Calculate enhanced combined score
            result.combined_score = (
                result.relevance_score * 0.3 +
                result.content_quality_score * 0.3 +
                result.extraction_score * 0.2 +
                result.authority_score * 0.1 +
                result.freshness_score * 0.1
            )
        
        # Sort by combined score
        ranked_results = sorted(results, key=lambda x: x.combined_score, reverse=True)
        
        return ranked_results
    
    def _update_crawlee_stats(self, results: List[CrawleeEnhancedResult], search_time: float):
        """Update Crawlee-specific statistics."""
        self.crawlee_stats['content_extraction_time'] += search_time
        
        # Update average content length
        content_lengths = [len(r.extracted_content or '') for r in results if r.extracted_content]
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            current_avg = self.crawlee_stats['average_content_length']
            total_extractions = self.crawlee_stats['successful_content_extractions']
            
            if total_extractions > 0:
                self.crawlee_stats['average_content_length'] = (
                    (current_avg * (total_extractions - len(content_lengths)) + sum(content_lengths)) /
                    total_extractions
                )
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including Crawlee metrics."""
        base_stats = self.get_stats()
        
        # Add Crawlee-specific statistics
        base_stats['crawlee'] = self.crawlee_stats.copy()
        base_stats['crawlee']['extractors_available'] = self.extractors_available
        base_stats['crawlee']['content_extraction_success_rate'] = (
            self.crawlee_stats['successful_content_extractions'] /
            max(self.crawlee_stats['total_content_requests'], 1)
        )
        base_stats['crawlee']['quality_filter_success_rate'] = (
            self.crawlee_stats['quality_filter_passed'] /
            max(self.crawlee_stats['quality_filter_passed'] + self.crawlee_stats['quality_filter_failed'], 1)
        )
        
        return base_stats
    
    async def close(self):
        """Clean up resources including Crawlee components."""
        # Close Crawlee components
        if self.crawler:
            await self.crawler.teardown()
        if self.request_queue:
            await self.request_queue.drop()
        if self.dataset:
            await self.dataset.drop()
        
        # Close base crawler
        await super().close()
        
        logger.info("CrawleeEnhancedSearchCrawler closed")


# Convenience functions
def create_crawlee_search_config(
    engines: Optional[List[str]] = None,
    max_results: int = 50,
    enable_content_extraction: bool = True,
    target_countries: Optional[List[str]] = None,
    **kwargs
) -> CrawleeSearchConfig:
    """Create Crawlee search configuration with sensible defaults."""
    return CrawleeSearchConfig(
        engines=engines or ['google', 'bing', 'duckduckgo', 'yandex', 'brave', 'startpage'],
        total_max_results=max_results,
        download_content=enable_content_extraction,
        target_countries=target_countries or ["ET", "SO", "ER", "DJ", "KE", "UG"],
        **kwargs
    )


async def create_crawlee_search_crawler(
    config: Optional[CrawleeSearchConfig] = None
) -> CrawleeEnhancedSearchCrawler:
    """Create and initialize Crawlee-enhanced search crawler."""
    if config is None:
        config = create_crawlee_search_config()
    
    crawler = CrawleeEnhancedSearchCrawler(config)
    if config.download_content and CRAWLEE_AVAILABLE:
        await crawler.initialize_crawlee()
    
    return crawler


# Export main components
__all__ = [
    'CrawleeSearchConfig',
    'CrawleeEnhancedResult',
    'CrawleeEnhancedSearchCrawler',
    'create_crawlee_search_config',
    'create_crawlee_search_crawler',
    'CRAWLEE_AVAILABLE'
]