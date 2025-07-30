"""
Article Extractor with Utils Integration
=========================================

Article content extraction using multiple strategies and utils packages.
Integrates with packages_enhanced/utils for enhanced functionality.

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

# Import utils packages
try:
    from ....utils.nlp import TextPreprocessor, LanguageDetector
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    TextPreprocessor = None
    LanguageDetector = None

try:
    from ....utils.caching import CacheManager, CacheConfig, CacheStrategy, CacheBackend
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    CacheManager = None
    CacheConfig = None

try:
    from ....utils.monitoring import PerformanceMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PerformanceMonitor = None

# Content extraction libraries
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    Article = None

try:
    from newsplease import NewsPlease
    NEWSPLEASE_AVAILABLE = True
except ImportError:
    NEWSPLEASE_AVAILABLE = False
    NewsPlease = None

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    trafilatura = None

try:
    from goose3 import Goose
    GOOSE_AVAILABLE = True
except ImportError:
    GOOSE_AVAILABLE = False
    Goose = None

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    Document = None

logger = logging.getLogger(__name__)


@dataclass
class ArticleContent:
    """Extracted article content."""
    title: str = ""
    content: str = ""
    summary: str = ""
    authors: List[str] = field(default_factory=list)
    publish_date: Optional[datetime] = None
    language: str = ""
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    image_url: str = ""
    video_urls: List[str] = field(default_factory=list)
    meta_description: str = ""
    canonical_url: str = ""
    source_domain: str = ""
    word_count: int = 0
    reading_time: int = 0  # in minutes
    quality_score: float = 0.0
    extraction_method: str = ""
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Result of article extraction operation."""
    success: bool
    article: Optional[ArticleContent] = None
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    methods_tried: List[str] = field(default_factory=list)
    extraction_stats: Dict[str, Any] = field(default_factory=dict)


class ArticleExtractor:
    """Article extractor with multiple strategies and utils integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize article extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize utils components
        self.text_preprocessor = None
        self.language_detector = None
        self.cache_manager = None
        self.performance_monitor = None
        
        if NLP_AVAILABLE:
            if self.config.get('enable_text_preprocessing', True):
                self.text_preprocessor = TextPreprocessor()
            if self.config.get('enable_language_detection', True):
                self.language_detector = LanguageDetector()
        
        if CACHING_AVAILABLE and self.config.get('enable_caching', True):
            cache_config = CacheConfig(
                strategy=CacheStrategy.LRU,
                backend=CacheBackend.MEMORY,
                ttl=self.config.get('cache_ttl', 3600),
                max_size=self.config.get('cache_max_size', 1000)
            )
            self.cache_manager = CacheManager(cache_config)
        
        if MONITORING_AVAILABLE and self.config.get('enable_monitoring', True):
            self.performance_monitor = PerformanceMonitor()
        
        # Extractor configuration
        self.extraction_methods = self.config.get('extraction_methods', [
            'newspaper', 'newsplease', 'trafilatura', 'goose', 'readability'
        ])
        self.fallback_enabled = self.config.get('enable_fallback', True)
        self.quality_threshold = self.config.get('quality_threshold', 0.5)
        
        # Initialize external extractors
        self.goose_client = None
        if GOOSE_AVAILABLE and 'goose' in self.extraction_methods:
            try:
                self.goose_client = Goose()
            except Exception as e:
                logger.warning(f"Failed to initialize Goose: {e}")
        
        logger.info(f"ArticleExtractor initialized with methods: {self.extraction_methods}")
    
    async def extract(self, html: str, url: str = "", **kwargs) -> ExtractionResult:
        """
        Extract article content using multiple strategies.
        
        Args:
            html: HTML content to extract from
            url: Source URL for the article
            **kwargs: Additional parameters
            
        Returns:
            ExtractionResult with extracted content
        """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"article_extract_{hash(html + url)}"
        if self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug("Retrieved article extraction result from cache")
                return cached_result
        
        methods_tried = []
        errors = []
        best_article = None
        best_score = 0.0
        
        # Try each extraction method
        for method in self.extraction_methods:
            try:
                article = await self._extract_with_method(method, html, url)
                methods_tried.append(method)
                
                if article:
                    # Calculate quality score
                    quality_score = self._calculate_quality_score(article)
                    article.quality_score = quality_score
                    article.extraction_method = method
                    
                    # Keep best result
                    if quality_score > best_score:
                        best_score = quality_score
                        best_article = article
                    
                    # If quality is good enough, stop trying other methods
                    if quality_score >= self.quality_threshold:
                        break
                        
            except Exception as e:
                error_msg = f"Method {method} failed: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                methods_tried.append(f"{method}_failed")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Enhance article with utils if available
        if best_article and NLP_AVAILABLE:
            best_article = await self._enhance_with_nlp(best_article)
        
        success = best_article is not None and best_score >= self.quality_threshold
        
        result = ExtractionResult(
            success=success,
            article=best_article,
            errors=errors,
            processing_time=processing_time,
            methods_tried=methods_tried,
            extraction_stats={
                'best_score': best_score,
                'methods_available': len(self.extraction_methods),
                'methods_tried': len(methods_tried)
            }
        )
        
        # Cache result
        if self.cache_manager:
            await self.cache_manager.set(cache_key, result)
        
        # Record performance metrics
        if self.performance_monitor:
            self.performance_monitor.record_timing('article_extract', processing_time * 1000)
            self.performance_monitor.record_counter('article_extractions', 1)
            if success:
                self.performance_monitor.record_counter('successful_extractions', 1)
        
        return result
    
    async def _extract_with_method(self, method: str, html: str, url: str) -> Optional[ArticleContent]:
        """Extract article using specific method."""
        
        if method == 'newspaper' and NEWSPAPER_AVAILABLE:
            return await self._extract_with_newspaper(html, url)
        
        elif method == 'newsplease' and NEWSPLEASE_AVAILABLE:
            return await self._extract_with_newsplease(html, url)
        
        elif method == 'trafilatura' and TRAFILATURA_AVAILABLE:
            return await self._extract_with_trafilatura(html, url)
        
        elif method == 'goose' and GOOSE_AVAILABLE and self.goose_client:
            return await self._extract_with_goose(html, url)
        
        elif method == 'readability' and READABILITY_AVAILABLE:
            return await self._extract_with_readability(html, url)
        
        else:
            logger.warning(f"Extraction method {method} not available")
            return None
    
    async def _extract_with_newspaper(self, html: str, url: str) -> Optional[ArticleContent]:
        """Extract using newspaper3k."""
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            # Try to extract additional info
            try:
                article.nlp()
            except:
                pass  # NLP processing might fail
            
            return ArticleContent(
                title=article.title or "",
                content=article.text or "",
                summary=article.summary or "",
                authors=list(article.authors) or [],
                publish_date=article.publish_date,
                keywords=list(article.keywords) or [],
                image_url=article.top_image or "",
                meta_description=article.meta_description or "",
                canonical_url=article.canonical_link or url,
                source_domain=urlparse(url).netloc if url else "",
                word_count=len(article.text.split()) if article.text else 0
            )
            
        except Exception as e:
            logger.warning(f"Newspaper extraction failed: {e}")
            return None
    
    async def _extract_with_newsplease(self, html: str, url: str) -> Optional[ArticleContent]:
        """Extract using news-please."""
        try:
            article = NewsPlease.from_html(html, url=url)
            
            if not article:
                return None
            
            return ArticleContent(
                title=article.title or "",
                content=article.maintext or "",
                summary=article.description or "",
                authors=[article.authors] if article.authors else [],
                publish_date=article.date_publish,
                language=article.language or "",
                image_url=article.image_url or "",
                source_domain=article.source_domain or "",
                word_count=len(article.maintext.split()) if article.maintext else 0
            )
            
        except Exception as e:
            logger.warning(f"NewsPlease extraction failed: {e}")
            return None
    
    async def _extract_with_trafilatura(self, html: str, url: str) -> Optional[ArticleContent]:
        """Extract using trafilatura."""
        try:
            # Extract main content
            content = trafilatura.extract(html, include_comments=False, include_tables=True)
            
            if not content:
                return None
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html)
            
            return ArticleContent(
                title=metadata.title if metadata and metadata.title else "",
                content=content,
                authors=[metadata.author] if metadata and metadata.author else [],
                publish_date=metadata.date if metadata and metadata.date else None,
                canonical_url=metadata.url if metadata and metadata.url else url,
                source_domain=metadata.sitename if metadata and metadata.sitename else urlparse(url).netloc if url else "",
                word_count=len(content.split())
            )
            
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            return None
    
    async def _extract_with_goose(self, html: str, url: str) -> Optional[ArticleContent]:
        """Extract using goose3."""
        try:
            if not self.goose_client:
                return None
            
            article = self.goose_client.extract(raw_html=html, final_url=url)
            
            if not article:
                return None
            
            return ArticleContent(
                title=article.title or "",
                content=article.cleaned_text or "",
                summary=article.meta_description or "",
                authors=[article.authors] if article.authors else [],
                publish_date=article.publish_date,
                tags=list(article.tags) if article.tags else [],
                image_url=article.top_image.src if article.top_image else "",
                meta_description=article.meta_description or "",
                canonical_url=article.canonical_link or url,
                source_domain=article.domain or "",
                word_count=len(article.cleaned_text.split()) if article.cleaned_text else 0
            )
            
        except Exception as e:
            logger.warning(f"Goose extraction failed: {e}")
            return None
    
    async def _extract_with_readability(self, html: str, url: str) -> Optional[ArticleContent]:
        """Extract using readability."""
        try:
            doc = Document(html)
            content = doc.summary()
            title = doc.title()
            
            if not content:
                return None
            
            # Clean content (basic text extraction)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)
            
            return ArticleContent(
                title=title or "",
                content=text_content,
                canonical_url=url,
                source_domain=urlparse(url).netloc if url else "",
                word_count=len(text_content.split())
            )
            
        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")
            return None
    
    def _calculate_quality_score(self, article: ArticleContent) -> float:
        """Calculate quality score for extracted article."""
        score = 0.0
        
        # Content length score (0-0.4)
        if article.word_count > 100:
            score += min(0.4, article.word_count / 1000 * 0.4)
        
        # Title score (0-0.2)
        if article.title and len(article.title) > 10:
            score += 0.2
        elif article.title:
            score += 0.1
        
        # Author score (0-0.1)
        if article.authors:
            score += 0.1
        
        # Date score (0-0.1)
        if article.publish_date:
            score += 0.1
        
        # Metadata score (0-0.2)
        metadata_count = sum([
            bool(article.summary),
            bool(article.keywords),
            bool(article.image_url),
            bool(article.meta_description)
        ])
        score += metadata_count * 0.05
        
        return min(1.0, score)
    
    async def _enhance_with_nlp(self, article: ArticleContent) -> ArticleContent:
        """Enhance article with NLP processing."""
        try:
            # Language detection
            if self.language_detector and article.content:
                lang_result = await self.language_detector.detect(article.content)
                if lang_result and lang_result.language:
                    article.language = lang_result.language
            
            # Text preprocessing for quality assessment
            if self.text_preprocessor and article.content:
                processed = await self.text_preprocessor.process(
                    article.content,
                    operations=['clean', 'normalize']
                )
                # Update word count with cleaned text
                article.word_count = len(processed.split())
            
            # Calculate reading time (average 200 words per minute)
            if article.word_count > 0:
                article.reading_time = max(1, round(article.word_count / 200))
            
        except Exception as e:
            logger.warning(f"NLP enhancement failed: {e}")
        
        return article
    
    async def extract_batch(self, html_list: List[str], urls: List[str] = None) -> List[ExtractionResult]:
        """Extract multiple articles."""
        urls = urls or [""] * len(html_list)
        results = []
        
        for i, html in enumerate(html_list):
            url = urls[i] if i < len(urls) else ""
            result = await self.extract(html, url)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        stats = {
            'extraction_methods': self.extraction_methods,
            'available_methods': {
                'newspaper': NEWSPAPER_AVAILABLE,
                'newsplease': NEWSPLEASE_AVAILABLE,
                'trafilatura': TRAFILATURA_AVAILABLE,
                'goose': GOOSE_AVAILABLE,
                'readability': READABILITY_AVAILABLE
            },
            'nlp_available': NLP_AVAILABLE,
            'caching_enabled': self.cache_manager is not None,
            'monitoring_enabled': self.performance_monitor is not None,
            'quality_threshold': self.quality_threshold
        }
        
        if self.performance_monitor:
            stats['performance_stats'] = self.performance_monitor.get_summary()
        
        return stats


# Utility functions
def quick_extract_article(html: str, url: str = "", **kwargs) -> ExtractionResult:
    """Quick article extraction function."""
    extractor = ArticleExtractor(kwargs)
    import asyncio
    return asyncio.run(extractor.extract(html, url))


def calculate_reading_time(word_count: int, wpm: int = 200) -> int:
    """Calculate reading time in minutes."""
    return max(1, round(word_count / wpm))