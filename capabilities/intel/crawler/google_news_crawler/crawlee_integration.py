#!/usr/bin/env python3
"""
Google News Crawler + Crawlee Integration
=========================================

Integration module that uses Crawlee to download full content from Google News search results,
providing enhanced article extraction, parsing, and content analysis capabilities.

Key Features:
- **Crawlee Integration**: Uses AdaptivePlaywrightCrawler for robust content downloading
- **Content Enhancement**: Downloads full article content from search result links
- **Intelligent Parsing**: Multi-method content extraction with quality scoring
- **Fallback Strategy**: Graceful fallback to RSS metadata if content download fails
- **Performance Optimization**: Concurrent downloads with rate limiting
- **Geographic Targeting**: Horn of Africa focused with configurable regions

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
import json
import uuid

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
class CrawleeConfig:
    """Configuration for Crawlee integration."""
    
    # Crawlee settings
    max_requests_per_crawl: int = 100
    request_handler_timeout: int = 60
    navigation_timeout: int = 30
    max_concurrent: int = 5
    
    # Content extraction settings
    enable_full_content: bool = True
    enable_image_extraction: bool = True
    enable_metadata_extraction: bool = True
    preferred_extraction_method: str = "auto"  # auto, trafilatura, newspaper, readability, bs4
    
    # Quality filters
    min_content_length: int = 200
    max_content_length: int = 50000
    enable_content_scoring: bool = True
    min_quality_score: float = 0.3
    
    # Geographic and language settings
    target_countries: List[str] = field(default_factory=lambda: ["ET", "SO", "ER", "DJ", "KE", "UG", "TZ", "SD", "SS"])
    target_languages: List[str] = field(default_factory=lambda: ["en", "fr", "ar", "sw"])
    
    # Rate limiting
    crawl_delay: float = 1.0
    respect_robots_txt: bool = True
    max_retries: int = 3
    
    # Storage settings
    enable_caching: bool = True
    cache_dir: Optional[Path] = None
    save_raw_html: bool = False


@dataclass
class ArticleResult:
    """Article result with full content and metadata."""
    
    # Basic metadata (from RSS/search)
    title: str
    url: str
    published_date: Optional[datetime]
    source: str
    description: Optional[str]
    
    # Enhanced content (from Crawlee)
    full_content: Optional[str] = None
    word_count: int = 0
    reading_time_minutes: float = 0.0
    quality_score: float = 0.0
    
    # Content structure
    article_text: Optional[str] = None
    headline: Optional[str] = None
    lead_paragraph: Optional[str] = None
    body_paragraphs: List[str] = field(default_factory=list)
    
    # Media and metadata
    images: List[Dict[str, str]] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Geographic and topical relevance
    geographic_entities: List[str] = field(default_factory=list)
    conflict_indicators: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    
    # Processing metadata
    extraction_method: str = "unknown"
    processing_time_ms: float = 0.0
    crawl_success: bool = False
    fallback_used: bool = False
    
    # Error information
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'id': str(uuid.uuid4()),
            'external_id': hashlib.sha256(self.url.encode()).hexdigest()[:16],
            'title': self.title[:500] if self.title else '',
            'content': self.full_content[:10000] if self.full_content else (self.description or ''),
            'content_url': self.url,
            'published_at': self.published_date,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'source': self.source[:100] if self.source else 'google_news',
            'data_source_id': 'google_news_crawlee',
            'language': 'en',  # TODO: detect language
            'word_count': self.word_count,
            'quality_score': self.quality_score,
            'relevance_score': self.relevance_score,
            'authors': self.authors,
            'tags': self.tags + self.keywords,
            'geographic_entities': self.geographic_entities,
            'conflict_indicators': self.conflict_indicators,
            'is_conflict_related': len(self.conflict_indicators) > 0,
            'metadata': {
                'extraction_method': self.extraction_method,
                'processing_time_ms': self.processing_time_ms,
                'crawl_success': self.crawl_success,
                'fallback_used': self.fallback_used,
                'reading_time_minutes': self.reading_time_minutes,
                'images': self.images,
                'errors': self.errors
            }
        }


class CrawleeNewsEnhancer:
    """Enhanced Google News processor using Crawlee for content downloading."""
    
    def __init__(self, config: CrawleeConfig):
        self.config = config
        self.crawler = None
        self.request_queue = None
        self.dataset = None
        self.processed_articles = []
        self.processing_stats = {
            'total_requests': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'content_extracted': 0,
            'fallback_used': 0,
            'start_time': None
        }
        
        # Content extractors availability
        self.extractors_available = {
            'trafilatura': TRAFILATURA_AVAILABLE,
            'newspaper': NEWSPAPER_AVAILABLE,
            'readability': READABILITY_AVAILABLE,
            'beautifulsoup': BS4_AVAILABLE
        }
        
        logger.info(f"CrawleeNewsEnhancer initialized with {sum(self.extractors_available.values())} extractors available")
    
    async def initialize(self):
        """Initialize Crawlee components."""
        if not CRAWLEE_AVAILABLE:
            raise ImportError("Crawlee not available. Install with: pip install crawlee")
        
        try:
            # Initialize Crawlee crawler with optimized settings
            self.crawler = AsyncAdaptivePlaywrightCrawler(
                max_requests_per_crawl=self.config.max_requests_per_crawl,
                request_handler_timeout_secs=self.config.request_handler_timeout,
                navigation_timeout_secs=self.config.navigation_timeout,
                max_request_retries=self.config.max_retries,
                browser_type='chromium',
                headless=True
            )
            
            # Initialize request queue and dataset
            self.request_queue = await RequestQueue.open()
            self.dataset = await Dataset.open()
            
            # Set up request handler
            @self.crawler.router.default_handler
            async def handle_article(context: CrawleeContext):
                await self._handle_article_request(context)
            
            logger.info("✅ Crawlee components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Crawlee components: {e}")
            raise
    
    async def enhance_articles(
        self, 
        article_metadata: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[ArticleResult]:
        """
        Enhance article metadata by downloading full content using Crawlee.
        
        Args:
            article_metadata: List of basic article metadata from RSS/search
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of enhanced articles with full content
        """
        if not self.crawler:
            raise RuntimeError("CrawleeNewsEnhancer not initialized")
        
        self.processing_stats['start_time'] = time.time()
        self.processing_stats['total_requests'] = len(article_metadata)
        self.processed_articles = []
        
        logger.info(f"🚀 Starting content enhancement for {len(article_metadata)} articles")
        
        try:
            # Add all URLs to request queue
            for article in article_metadata:
                url = article.get('url') or article.get('link')
                if url:
                    request = Request.from_url(
                        url,
                        user_data={
                            'article_metadata': article,
                            'index': len(self.processed_articles)
                        }
                    )
                    await self.request_queue.add_request(request)
            
            # Run the crawler
            await self.crawler.run()
            
            # Process results and create enhanced articles
            enhanced_articles = await self._compile_enhanced_results()
            
            # Update final statistics
            processing_time = time.time() - self.processing_stats['start_time']
            success_rate = (self.processing_stats['successful_downloads'] / 
                          max(self.processing_stats['total_requests'], 1)) * 100
            
            logger.info(f"✅ Content enhancement completed!")
            logger.info(f"   Processed: {self.processing_stats['total_requests']} articles")
            logger.info(f"   Successful: {self.processing_stats['successful_downloads']}")
            logger.info(f"   Success rate: {success_rate:.1f}%")
            logger.info(f"   Processing time: {processing_time:.1f}s")
            
            return enhanced_articles
            
        except Exception as e:
            logger.error(f"Content enhancement failed: {e}")
            raise
    
    async def _handle_article_request(self, context: CrawleeContext):
        """Handle individual article content download and processing."""
        start_time = time.time()
        article_metadata = context.request.user_data['article_metadata']
        url = context.request.loaded_url
        
        try:
            logger.debug(f"Processing article: {url}")
            
            # Extract content using multiple methods
            content_result = await self._extract_article_content(
                url=url,
                html=await context.page.content(),
                metadata=article_metadata
            )
            
            # Create enhanced article result
            enhanced_article = self._create_enhanced_article(
                metadata=article_metadata,
                content_result=content_result,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Store result
            await self.dataset.push_data(enhanced_article.to_dict())
            self.processed_articles.append(enhanced_article)
            
            self.processing_stats['successful_downloads'] += 1
            if content_result.get('content'):
                self.processing_stats['content_extracted'] += 1
            
            logger.debug(f"✅ Successfully processed: {url}")
            
        except Exception as e:
            logger.warning(f"Failed to process article {url}: {e}")
            self.processing_stats['failed_downloads'] += 1
            
            # Create fallback result using only metadata
            fallback_article = self._create_fallback_article(
                metadata=article_metadata,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            await self.dataset.push_data(fallback_article.to_dict())
            self.processed_articles.append(fallback_article)
            
            self.processing_stats['fallback_used'] += 1
    
    async def _extract_article_content(
        self, 
        url: str, 
        html: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract article content using multiple methods with fallback."""
        
        extraction_results = {}
        best_result = None
        best_score = 0.0
        
        # Try different extraction methods based on configuration
        methods_to_try = self._get_extraction_methods()
        
        for method in methods_to_try:
            try:
                result = await self._extract_with_method(method, url, html)
                if result and result.get('content'):
                    score = self._score_extraction_result(result)
                    extraction_results[method] = {**result, 'score': score}
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_result['method'] = method
                        
            except Exception as e:
                logger.debug(f"Extraction method {method} failed for {url}: {e}")
                extraction_results[method] = {'error': str(e), 'score': 0.0}
        
        # Return best result or create minimal result
        if best_result:
            return best_result
        else:
            return {
                'content': '',
                'title': metadata.get('title', ''),
                'method': 'fallback',
                'score': 0.0,
                'extraction_attempts': extraction_results
            }
    
    def _get_extraction_methods(self) -> List[str]:
        """Get list of extraction methods to try based on configuration and availability."""
        preferred = self.config.preferred_extraction_method
        
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
            content = trafilatura.extract(html, include_images=True, include_links=False)
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html)
            
            result = {
                'content': content or '',
                'title': metadata.title if metadata else '',
                'authors': [metadata.author] if metadata and metadata.author else [],
                'published_date': metadata.date if metadata else None,
                'description': metadata.description if metadata else '',
                'word_count': len(content.split()) if content else 0
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
                'word_count': len(article.text.split()) if article.text else 0
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
            else:
                text_content = content
            
            result = {
                'content': text_content or '',
                'title': title or '',
                'authors': [],
                'published_date': None,
                'description': '',
                'word_count': len(text_content.split()) if text_content else 0
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
            
            # Extract main content (try common article selectors)
            content_selectors = [
                'article', '.article-content', '.content', '.post-content',
                '.entry-content', 'main', '[role="main"]', '.article-body'
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
            
            result = {
                'content': content_text or '',
                'title': title_text,
                'authors': [],
                'published_date': None,
                'description': '',
                'word_count': len(content_text.split()) if content_text else 0
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
            raise
    
    def _score_extraction_result(self, result: Dict[str, Any]) -> float:
        """Score the quality of an extraction result."""
        score = 0.0
        
        # Content length scoring
        content = result.get('content', '')
        word_count = len(content.split()) if content else 0
        
        if word_count >= 300:
            score += 0.4
        elif word_count >= 100:
            score += 0.2
        elif word_count >= 50:
            score += 0.1
        
        # Title quality
        title = result.get('title', '')
        if title and len(title) > 10:
            score += 0.2
        
        # Metadata presence
        if result.get('authors'):
            score += 0.1
        if result.get('published_date'):
            score += 0.1
        if result.get('keywords'):
            score += 0.1
        if result.get('description'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_enhanced_article(
        self,
        metadata: Dict[str, Any],
        content_result: Dict[str, Any],
        processing_time_ms: float
    ) -> ArticleResult:
        """Create enhanced article result from metadata and content."""
        
        # Extract geographic and conflict indicators
        content = content_result.get('content', '')
        geographic_entities = self._extract_geographic_entities(content)
        conflict_indicators = self._extract_conflict_indicators(content)
        
        return ArticleResult(
            # Basic metadata
            title=content_result.get('title') or metadata.get('title', ''),
            url=metadata.get('url') or metadata.get('link', ''),
            published_date=content_result.get('published_date') or self._parse_date(metadata.get('published_date')),
            source=metadata.get('source', ''),
            description=content_result.get('description') or metadata.get('description', ''),
            
            # Enhanced content
            full_content=content,
            word_count=content_result.get('word_count', 0),
            reading_time_minutes=content_result.get('word_count', 0) / 200,  # Assume 200 WPM
            quality_score=content_result.get('score', 0.0),
            
            # Content structure
            article_text=content,
            headline=content_result.get('title') or metadata.get('title', ''),
            
            # Media and metadata
            images=content_result.get('images', []),
            authors=content_result.get('authors', []),
            keywords=content_result.get('keywords', []),
            
            # Geographic and topical relevance
            geographic_entities=geographic_entities,
            conflict_indicators=conflict_indicators,
            relevance_score=self._calculate_relevance_score(content, geographic_entities, conflict_indicators),
            
            # Processing metadata
            extraction_method=content_result.get('method', 'unknown'),
            processing_time_ms=processing_time_ms,
            crawl_success=True,
            fallback_used=False
        )
    
    def _create_fallback_article(
        self,
        metadata: Dict[str, Any],
        error: str,
        processing_time_ms: float
    ) -> ArticleResult:
        """Create fallback article result when content download fails."""
        
        description = metadata.get('description', '')
        
        return ArticleResult(
            # Basic metadata only
            title=metadata.get('title', ''),
            url=metadata.get('url') or metadata.get('link', ''),
            published_date=self._parse_date(metadata.get('published_date')),
            source=metadata.get('source', ''),
            description=description,
            
            # Limited content (from RSS/metadata only)
            full_content=description,
            word_count=len(description.split()) if description else 0,
            quality_score=0.1,  # Low score for fallback
            
            # Processing metadata
            extraction_method='fallback',
            processing_time_ms=processing_time_ms,
            crawl_success=False,
            fallback_used=True,
            errors=[error]
        )
    
    def _extract_geographic_entities(self, content: str) -> List[str]:
        """Extract geographic entities from content."""
        # Simple keyword-based extraction for Horn of Africa countries/regions
        entities = []
        geographic_terms = [
            'Ethiopia', 'Kenya', 'Somalia', 'Sudan', 'South Sudan', 'Uganda',
            'Tanzania', 'Eritrea', 'Djibouti', 'Rwanda', 'Burundi',
            'Addis Ababa', 'Nairobi', 'Mogadishu', 'Khartoum', 'Kampala',
            'Horn of Africa', 'East Africa'
        ]
        
        content_lower = content.lower()
        for term in geographic_terms:
            if term.lower() in content_lower:
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_conflict_indicators(self, content: str) -> List[str]:
        """Extract conflict-related indicators from content."""
        indicators = []
        conflict_terms = [
            'conflict', 'violence', 'war', 'battle', 'fighting', 'attack',
            'bombing', 'shooting', 'terrorism', 'insurgency', 'rebellion',
            'protest', 'riot', 'unrest', 'crisis', 'emergency', 'displacement',
            'refugee', 'humanitarian', 'peacekeeping', 'ceasefire'
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
        score = 0.0
        
        # Geographic relevance (0.0 - 0.5)
        if geographic_entities:
            score += min(len(geographic_entities) * 0.1, 0.5)
        
        # Conflict relevance (0.0 - 0.5)
        if conflict_indicators:
            score += min(len(conflict_indicators) * 0.1, 0.5)
        
        return min(score, 1.0)
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Handle various date formats
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None
    
    async def _compile_enhanced_results(self) -> List[ArticleResult]:
        """Compile final enhanced results from processed articles."""
        
        # Sort by relevance score and quality
        self.processed_articles.sort(
            key=lambda x: (x.relevance_score, x.quality_score),
            reverse=True
        )
        
        # Filter by quality if enabled
        if self.config.enable_content_scoring:
            filtered_articles = [
                article for article in self.processed_articles
                if article.quality_score >= self.config.min_quality_score
            ]
        else:
            filtered_articles = self.processed_articles
        
        logger.info(f"📊 Final results: {len(filtered_articles)}/{len(self.processed_articles)} articles passed quality filter")
        
        return filtered_articles
    
    async def close(self):
        """Clean up Crawlee resources."""
        if self.crawler:
            await self.crawler.teardown()
        if self.request_queue:
            await self.request_queue.drop()
        if self.dataset:
            await self.dataset.drop()
        
        logger.info("CrawleeNewsEnhancer closed")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        if stats['start_time']:
            stats['total_time_seconds'] = time.time() - stats['start_time']
            stats['articles_per_second'] = stats['successful_downloads'] / max(stats['total_time_seconds'], 1)
        
        stats['extractors_available'] = self.extractors_available
        return stats


# Factory functions
def create_crawlee_config(
    max_requests: int = 100,
    max_concurrent: int = 5,
    target_countries: Optional[List[str]] = None,
    enable_full_content: bool = True,
    **kwargs
) -> CrawleeConfig:
    """Create CrawleeConfig with sensible defaults."""
    return CrawleeConfig(
        max_requests_per_crawl=max_requests,
        max_concurrent=max_concurrent,
        target_countries=target_countries or ["ET", "SO", "ER", "DJ", "KE", "UG"],
        enable_full_content=enable_full_content,
        **kwargs
    )


async def create_crawlee_enhancer(config: Optional[CrawleeConfig] = None) -> CrawleeNewsEnhancer:
    """Create and initialize CrawleeNewsEnhancer."""
    if config is None:
        config = create_crawlee_config()
    
    enhancer = CrawleeNewsEnhancer(config)
    await enhancer.initialize()
    return enhancer


# Export main components
__all__ = [
    'CrawleeConfig',
    'ArticleResult', 
    'CrawleeNewsEnhancer',
    'create_crawlee_config',
    'create_crawlee_enhancer',
    'CRAWLEE_AVAILABLE'
]