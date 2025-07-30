"""
Unified Content Parser
======================

Unified content parsing interface that coordinates all parser components
and integrates with packages_enhanced utilities.

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

# Import parser components
from .html_parser import HTMLParser, HTMLParseResult
from .article_extractor import ArticleExtractor, ArticleContent, ExtractionResult
from .metadata_extractor import MetadataExtractor, ArticleMetadata
from .ml_content_analyzer import MLContentAnalyzer, ContentAnalysis

# Import utils packages
try:
    from ....utils.monitoring import PerformanceMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PerformanceMonitor = None

try:
    from ....utils.caching import CacheManager, CacheConfig, CacheStrategy, CacheBackend
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    CacheManager = None
    CacheConfig = None

logger = logging.getLogger(__name__)


@dataclass
class ParsedContent:
    """Unified parsed content result."""
    success: bool
    url: str = ""
    
    # HTML parsing results
    html_result: Optional[HTMLParseResult] = None
    
    # Article extraction results
    article_result: Optional[ExtractionResult] = None
    article: Optional[ArticleContent] = None
    
    # Metadata extraction results
    metadata: Optional[ArticleMetadata] = None
    
    # ML analysis results
    ml_analysis: Optional[ContentAnalysis] = None
    
    # Unified content
    title: str = ""
    content: str = ""
    summary: str = ""
    
    # Quality metrics
    overall_quality: float = 0.0
    extraction_confidence: float = 0.0
    content_score: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    parsers_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ContentParser:
    """Unified content parser with all parsing capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified content parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize parser components
        self.html_parser = HTMLParser(self.config.get('html_config', {}))
        self.article_extractor = ArticleExtractor(self.config.get('article_config', {}))
        self.metadata_extractor = MetadataExtractor(self.config.get('metadata_config', {}))
        self.ml_analyzer = MLContentAnalyzer(self.config.get('ml_config', {}))
        
        # Initialize monitoring and caching
        self.performance_monitor = None
        self.cache_manager = None
        
        if MONITORING_AVAILABLE and self.config.get('enable_monitoring', True):
            self.performance_monitor = PerformanceMonitor()
        
        if CACHING_AVAILABLE and self.config.get('enable_caching', True):
            cache_config = CacheConfig(
                strategy=CacheStrategy.LRU,
                backend=CacheBackend.MEMORY,
                ttl=self.config.get('cache_ttl', 3600),
                max_size=self.config.get('cache_max_size', 1000)
            )
            self.cache_manager = CacheManager(cache_config)
        
        # Parser configuration
        self.enable_html_parsing = self.config.get('enable_html_parsing', True)
        self.enable_article_extraction = self.config.get('enable_article_extraction', True)
        self.enable_metadata_extraction = self.config.get('enable_metadata_extraction', True)
        self.enable_ml_analysis = self.config.get('enable_ml_analysis', True)
        
        # Quality thresholds
        self.min_content_length = self.config.get('min_content_length', 100)
        self.quality_threshold = self.config.get('quality_threshold', 0.5)
        
        logger.info("ContentParser initialized with all components")
    
    async def parse(self, html: str, url: str = "", **kwargs) -> ParsedContent:
        """
        Parse content using all available parsers.
        
        Args:
            html: HTML content to parse
            url: Source URL for context
            **kwargs: Additional parameters
            
        Returns:
            ParsedContent with comprehensive parsing results
        """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"content_parse_{hash(html + url)}"
        if self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug("Retrieved content parsing result from cache")
                return cached_result
        
        result = ParsedContent(success=False, url=url)
        parsers_used = []
        errors = []
        warnings = []
        
        try:
            # HTML parsing
            if self.enable_html_parsing:
                try:
                    result.html_result = await self.html_parser.parse(html, url)
                    parsers_used.append("html_parser")
                    
                    if not result.html_result.success:
                        warnings.append("HTML parsing had issues")
                        
                except Exception as e:
                    error_msg = f"HTML parsing failed: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            
            # Article extraction
            if self.enable_article_extraction:
                try:
                    result.article_result = await self.article_extractor.extract(html, url)
                    parsers_used.append("article_extractor")
                    
                    if result.article_result.success:
                        result.article = result.article_result.article
                    else:
                        warnings.append("Article extraction failed")
                        
                except Exception as e:
                    error_msg = f"Article extraction failed: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            
            # Metadata extraction
            if self.enable_metadata_extraction:
                try:
                    result.metadata = await self.metadata_extractor.extract(html, url)
                    parsers_used.append("metadata_extractor")
                    
                except Exception as e:
                    error_msg = f"Metadata extraction failed: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            
            # Unify content from different sources
            self._unify_content(result)
            
            # ML analysis (if content is available)
            if self.enable_ml_analysis and result.content:
                try:
                    result.ml_analysis = await self.ml_analyzer.analyze(
                        result.content, 
                        result.title
                    )
                    parsers_used.append("ml_analyzer")
                    
                    if not result.ml_analysis.success:
                        warnings.append("ML analysis had issues")
                        
                except Exception as e:
                    error_msg = f"ML analysis failed: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            
            # Calculate quality metrics
            self._calculate_quality_metrics(result)
            
            # Determine overall success
            result.success = self._determine_success(result)
            
        except Exception as e:
            error_msg = f"Content parsing failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        finally:
            # Set processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.parsers_used = parsers_used
            result.errors = errors
            result.warnings = warnings
        
        # Cache successful results
        if self.cache_manager and result.success:
            await self.cache_manager.set(cache_key, result)
        
        # Record performance metrics
        if self.performance_monitor:
            self.performance_monitor.record_timing('content_parse', processing_time * 1000)
            self.performance_monitor.record_counter('content_parses', 1)
            if result.success:
                self.performance_monitor.record_counter('successful_parses', 1)
        
        return result
    
    def _unify_content(self, result: ParsedContent):
        """Unify content from different parsing sources."""
        try:
            # Prioritize content sources: article > html > metadata
            
            # Title unification
            title_sources = []
            if result.article and result.article.title:
                title_sources.append((result.article.title, 3))  # Highest priority
            if result.metadata and result.metadata.title:
                title_sources.append((result.metadata.title, 2))
            if result.html_result and result.html_result.title:
                title_sources.append((result.html_result.title, 1))
            
            if title_sources:
                # Choose title from highest priority source
                title_sources.sort(key=lambda x: x[1], reverse=True)
                result.title = title_sources[0][0]
            
            # Content unification
            content_sources = []
            if result.article and result.article.content:
                content_length = len(result.article.content)
                if content_length >= self.min_content_length:
                    content_sources.append((result.article.content, content_length, 3))
            
            if result.html_result and result.html_result.content:
                content_length = len(result.html_result.content)
                if content_length >= self.min_content_length:
                    content_sources.append((result.html_result.content, content_length, 2))
            
            if content_sources:
                # Choose content with best combination of priority and length
                content_sources.sort(key=lambda x: (x[2], x[1]), reverse=True)
                result.content = content_sources[0][0]
            
            # Summary unification
            if result.article and result.article.summary:
                result.summary = result.article.summary
            elif result.metadata and result.metadata.description:
                result.summary = result.metadata.description
            elif result.content:
                # Create basic summary from content (first 200 characters)
                result.summary = result.content[:200] + "..." if len(result.content) > 200 else result.content
            
        except Exception as e:
            logger.warning(f"Content unification failed: {e}")
    
    def _calculate_quality_metrics(self, result: ParsedContent):
        """Calculate overall quality metrics."""
        try:
            quality_factors = []
            confidence_factors = []
            
            # HTML parsing quality
            if result.html_result and result.html_result.success:
                quality_factors.append(0.6 if result.html_result.content else 0.2)
                confidence_factors.append(0.2)
            
            # Article extraction quality
            if result.article_result and result.article_result.success and result.article:
                article_quality = result.article.quality_score
                quality_factors.append(article_quality * 0.8)
                confidence_factors.append(result.article.confidence * 0.3)
            
            # Metadata quality
            if result.metadata:
                metadata_quality = result.metadata.metadata_completeness
                quality_factors.append(metadata_quality * 0.4)
                confidence_factors.append(result.metadata.extraction_confidence * 0.2)
            
            # ML analysis quality
            if result.ml_analysis and result.ml_analysis.success:
                ml_quality = result.ml_analysis.content_quality
                quality_factors.append(ml_quality * 0.6)
                confidence_factors.append(result.ml_analysis.confidence * 0.3)
            
            # Content length factor
            if result.content:
                content_length_factor = min(1.0, len(result.content) / 1000)
                quality_factors.append(content_length_factor * 0.3)
            
            # Calculate overall metrics
            if quality_factors:
                result.overall_quality = sum(quality_factors) / len(quality_factors)
            
            if confidence_factors:
                result.extraction_confidence = sum(confidence_factors) / len(confidence_factors)
            
            # Content score (for ML analysis)
            if result.ml_analysis:
                score_factors = [
                    result.ml_analysis.content_quality,
                    result.ml_analysis.relevance_score,
                    result.ml_analysis.credibility_score
                ]
                valid_factors = [f for f in score_factors if f > 0]
                if valid_factors:
                    result.content_score = sum(valid_factors) / len(valid_factors)
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
    
    def _determine_success(self, result: ParsedContent) -> bool:
        """Determine if parsing was successful overall."""
        try:
            # Must have some content
            if not result.content or len(result.content) < self.min_content_length:
                return False
            
            # Must have at least one successful parser
            successful_parsers = 0
            
            if result.html_result and result.html_result.success:
                successful_parsers += 1
            
            if result.article_result and result.article_result.success:
                successful_parsers += 1
            
            if result.metadata:
                successful_parsers += 1
            
            if result.ml_analysis and result.ml_analysis.success:
                successful_parsers += 1
            
            # Need at least 2 successful parsers for high confidence
            if successful_parsers >= 2:
                return True
            
            # Or 1 successful parser with good quality
            if successful_parsers >= 1 and result.overall_quality >= self.quality_threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Success determination failed: {e}")
            return False
    
    async def parse_batch(self, html_list: List[str], urls: List[str] = None) -> List[ParsedContent]:
        """Parse multiple content items."""
        urls = urls or [""] * len(html_list)
        results = []
        
        for i, html in enumerate(html_list):
            url = urls[i] if i < len(urls) else ""
            result = await self.parse(html, url)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive parser statistics."""
        stats = {
            'parsers_enabled': {
                'html_parsing': self.enable_html_parsing,
                'article_extraction': self.enable_article_extraction,
                'metadata_extraction': self.enable_metadata_extraction,
                'ml_analysis': self.enable_ml_analysis
            },
            'configuration': {
                'min_content_length': self.min_content_length,
                'quality_threshold': self.quality_threshold,
                'caching_enabled': self.cache_manager is not None,
                'monitoring_enabled': self.performance_monitor is not None
            }
        }
        
        # Get individual parser stats
        try:
            stats['html_parser_stats'] = self.html_parser.get_stats()
            stats['article_extractor_stats'] = self.article_extractor.get_stats()
            stats['metadata_extractor_stats'] = self.metadata_extractor.get_stats()
            stats['ml_analyzer_stats'] = self.ml_analyzer.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get parser stats: {e}")
        
        # Get performance stats
        if self.performance_monitor:
            stats['performance_stats'] = self.performance_monitor.get_summary()
        
        return stats


# Utility functions
def quick_parse_content(html: str, url: str = "", **kwargs) -> ParsedContent:
    """Quick content parsing function."""
    parser = ContentParser(kwargs)
    import asyncio
    return asyncio.run(parser.parse(html, url))


def extract_best_content(html: str, url: str = "") -> Dict[str, str]:
    """Extract best content with minimal configuration."""
    result = quick_parse_content(html, url)
    
    return {
        'title': result.title,
        'content': result.content,
        'summary': result.summary,
        'success': result.success,
        'quality': result.overall_quality
    }