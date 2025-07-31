"""
ML Content Analyzer with Scorers Integration
=============================================

ML-based content analysis using existing scorers from packages_enhanced/scorers.
Provides conflict scoring, event extraction, and content quality assessment.

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

# Import existing scorers
try:
    from ....scorers import (
        create_event_extractor,
        create_simple_ml_scorer,
        create_enhanced_ml_scorer,
        SimpleMLScorerConfig,
        EnhancedMLScorerConfig
    )
    SCORERS_AVAILABLE = True
except ImportError:
    SCORERS_AVAILABLE = False
    create_event_extractor = None
    create_simple_ml_scorer = None
    create_enhanced_ml_scorer = None
    SimpleMLScorerConfig = None
    EnhancedMLScorerConfig = None

# Import utils packages
try:
    from ....utils.nlp import (
        SentimentAnalyzer,
        NamedEntityRecognizer,
        TopicModeler,
        TextPreprocessor
    )
    NLP_UTILS_AVAILABLE = True
except ImportError:
    NLP_UTILS_AVAILABLE = False

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
class ContentAnalysis:
    """Result of ML content analysis."""
    success: bool
    confidence: float = 0.0
    
    # Conflict and event analysis
    conflict_score: float = 0.0
    event_type: Optional[str] = None
    event_nature: Optional[str] = None
    severity_level: str = "unknown"
    
    # Content quality
    content_quality: float = 0.0
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    
    # NLP analysis
    sentiment_score: Optional[float] = None
    sentiment_label: str = "neutral"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Geographic and temporal
    locations: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)
    actors: List[str] = field(default_factory=list)
    
    # Analysis metadata
    processing_time: float = 0.0
    models_used: List[str] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)


class MLContentAnalyzer:
    """ML-based content analyzer using existing scorers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML content analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize existing scorers
        self.event_extractor = None
        self.simple_ml_scorer = None
        self.enhanced_ml_scorer = None
        
        if SCORERS_AVAILABLE:
            self._initialize_scorers()
        else:
            logger.warning("Scorers package not available - using basic analysis only")
        
        # Initialize NLP utils
        self.sentiment_analyzer = None
        self.ner = None
        self.topic_modeler = None
        self.text_preprocessor = None
        
        if NLP_UTILS_AVAILABLE:
            self._initialize_nlp_components()
        
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
        
        logger.info("MLContentAnalyzer initialized")
    
    def _initialize_scorers(self):
        """Initialize ML scorers from scorers package."""
        try:
            # Initialize event extractor for basic analysis
            if create_event_extractor:
                self.event_extractor = create_event_extractor(mode="STANDARD")
                logger.info("Event extractor initialized")
            
            # Initialize simple ML scorer for lightweight analysis
            if create_simple_ml_scorer:
                self.simple_ml_scorer = create_simple_ml_scorer()
                logger.info("Simple ML scorer initialized")
            
            # Initialize enhanced ML scorer for comprehensive analysis
            if create_enhanced_ml_scorer:
                self.enhanced_ml_scorer = create_enhanced_ml_scorer()
                logger.info("Enhanced ML scorer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize scorers: {e}")
            # Set to None to prevent further errors
            self.event_extractor = None
            self.simple_ml_scorer = None
            self.enhanced_ml_scorer = None
    
    def _initialize_nlp_components(self):
        """Initialize NLP components from utils."""
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            self.ner = NamedEntityRecognizer()
            self.topic_modeler = TopicModeler()
            self.text_preprocessor = TextPreprocessor()
            logger.info("NLP components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize some NLP components: {e}")
    
    async def analyze(self, content: str, title: str = "", **kwargs) -> ContentAnalysis:
        """
        Analyze content using ML models and scorers.
        
        Args:
            content: Text content to analyze
            title: Optional title for context
            **kwargs: Additional parameters
            
        Returns:
            ContentAnalysis with analysis results
        """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"content_analysis_{hash(content + title)}"
        if self.cache_manager:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug("Retrieved content analysis from cache")
                return cached_result
        
        analysis = ContentAnalysis(success=False)
        models_used = []
        errors = []
        
        try:
            # Combine title and content for analysis
            full_text = f"{title} {content}".strip()
            
            if not full_text:
                analysis.errors.append("No content to analyze")
                return analysis
            
            # Use enhanced ML scorer if available
            if self.enhanced_ml_scorer:
                try:
                    scorer_result = await self.enhanced_ml_scorer.extract_from_text(full_text)
                    self._process_enhanced_scorer_result(analysis, scorer_result)
                    models_used.append("enhanced_ml_scorer")
                except Exception as e:
                    errors.append(f"Enhanced ML scorer failed: {e}")
            
            # Use simple ML scorer as fallback
            elif self.simple_ml_scorer:
                try:
                    simple_result = await self.simple_ml_scorer.extract_from_text(full_text)
                    self._process_simple_scorer_result(analysis, simple_result)
                    models_used.append("simple_ml_scorer")
                except Exception as e:
                    errors.append(f"Simple ML scorer failed: {e}")
            
            # Use event extractor as basic fallback
            elif self.event_extractor:
                try:
                    event_result = await self.event_extractor.extract_from_article(full_text)
                    self._process_event_extractor_result(analysis, event_result)
                    models_used.append("event_extractor")
                except Exception as e:
                    errors.append(f"Event extractor failed: {e}")
            
            # Perform NLP analysis with utils
            if NLP_UTILS_AVAILABLE:
                await self._perform_nlp_analysis(analysis, full_text)
                models_used.append("nlp_utils")
            
            # Calculate composite scores
            self._calculate_composite_scores(analysis)
            
            # Determine success
            analysis.success = len(models_used) > 0
            analysis.models_used = models_used
            analysis.errors = errors
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            analysis.errors.append(str(e))
        
        finally:
            processing_time = (datetime.now() - start_time).total_seconds()
            analysis.processing_time = processing_time
            analysis.analysis_timestamp = datetime.now()
        
        # Cache result
        if self.cache_manager and analysis.success:
            await self.cache_manager.set(cache_key, analysis)
        
        # Record performance metrics
        if self.performance_monitor:
            try:
                # Try different method names for performance monitoring
                if hasattr(self.performance_monitor, 'record_timing'):
                    self.performance_monitor.record_timing('content_analysis', processing_time * 1000)
                elif hasattr(self.performance_monitor, 'record_metric'):
                    self.performance_monitor.record_metric('content_analysis_time', processing_time * 1000)
                
                if hasattr(self.performance_monitor, 'record_counter'):
                    self.performance_monitor.record_counter('content_analyses', 1)
                    if analysis.success:
                        self.performance_monitor.record_counter('successful_analyses', 1)
                elif hasattr(self.performance_monitor, 'increment'):
                    self.performance_monitor.increment('content_analyses')
                    if analysis.success:
                        self.performance_monitor.increment('successful_analyses')
            except Exception as perf_error:
                logger.debug(f"Performance monitoring failed: {perf_error}")
        
        return analysis
    
    def _process_enhanced_scorer_result(self, analysis: ContentAnalysis, scorer_result: Any):
        """Process result from enhanced ML scorer."""
        try:
            # Extract core metrics
            if hasattr(scorer_result, 'confidence_score'):
                analysis.confidence = max(analysis.confidence, scorer_result.confidence_score)
            
            if hasattr(scorer_result, 'event_type'):
                analysis.event_type = scorer_result.event_type
            
            if hasattr(scorer_result, 'event_nature'):
                analysis.event_nature = scorer_result.event_nature
            
            if hasattr(scorer_result, 'severity_level'):
                analysis.severity_level = scorer_result.severity_level
            
            if hasattr(scorer_result, 'sentiment_score'):
                analysis.sentiment_score = scorer_result.sentiment_score
                if scorer_result.sentiment_score:
                    if scorer_result.sentiment_score > 0.1:
                        analysis.sentiment_label = "positive"
                    elif scorer_result.sentiment_score < -0.1:
                        analysis.sentiment_label = "negative"
                    else:
                        analysis.sentiment_label = "neutral"
            
            # Extract entities and locations
            if hasattr(scorer_result, 'entities') and scorer_result.entities:
                analysis.entities = scorer_result.entities
                
                # Extract specific entity types
                for entity in scorer_result.entities:
                    if isinstance(entity, dict):
                        entity_type = entity.get('type', '').lower()
                        entity_text = entity.get('text', '')
                        
                        if entity_type in ['location', 'gpe'] and entity_text:
                            analysis.locations.append(entity_text)
                        elif entity_type in ['person', 'org'] and entity_text:
                            analysis.actors.append(entity_text)
            
            # Calculate conflict score based on event nature and severity
            if analysis.event_nature:
                conflict_keywords = ['conflict', 'violence', 'war', 'attack', 'bombing', 'shooting']
                if any(keyword in analysis.event_nature.lower() for keyword in conflict_keywords):
                    analysis.conflict_score = 0.8
                elif analysis.event_nature.lower() in ['protest', 'demonstration']:
                    analysis.conflict_score = 0.6
                else:
                    analysis.conflict_score = 0.3
            
        except Exception as e:
            logger.warning(f"Failed to process enhanced scorer result: {e}")
    
    def _process_unified_scorer_result(self, analysis: ContentAnalysis, scorer_result: Any):
        """Process result from unified scorer."""
        try:
            # Update confidence with maximum
            analysis.confidence = max(analysis.confidence, scorer_result.confidence)
            
            # Use conflict score
            analysis.conflict_score = max(analysis.conflict_score, scorer_result.conflict_score)
            
            # Use event information
            if scorer_result.event_type and not analysis.event_type:
                analysis.event_type = scorer_result.event_type
            
            if scorer_result.severity_level and analysis.severity_level == "unknown":
                analysis.severity_level = scorer_result.severity_level
            
            # Extract locations and actors
            if scorer_result.locations:
                analysis.locations.extend(scorer_result.locations)
            
            if scorer_result.actors:
                analysis.actors.extend(scorer_result.actors)
            
            # Extract categories as topics
            if scorer_result.event_categories:
                analysis.topics.extend(scorer_result.event_categories)
            
        except Exception as e:
            logger.warning(f"Failed to process unified scorer result: {e}")
    
    def _process_simple_scorer_result(self, analysis: ContentAnalysis, scorer_result: Any):
        """Process result from simple ML scorer."""
        try:
            # Extract basic metrics from simple scorer
            if hasattr(scorer_result, 'confidence_score'):
                analysis.confidence = max(analysis.confidence, scorer_result.confidence_score)
            
            if hasattr(scorer_result, 'is_violent') and scorer_result.is_violent:
                analysis.conflict_score = 0.7
                analysis.event_type = "conflict"
            
            if hasattr(scorer_result, 'location_name') and scorer_result.location_name:
                analysis.locations.append(scorer_result.location_name)
            
            if hasattr(scorer_result, 'country') and scorer_result.country:
                analysis.countries.append(scorer_result.country)
            
            if hasattr(scorer_result, 'brief_description') and scorer_result.brief_description:
                analysis.event_nature = scorer_result.brief_description
            
        except Exception as e:
            logger.warning(f"Failed to process simple scorer result: {e}")
    
    def _process_event_extractor_result(self, analysis: ContentAnalysis, extractor_result: Any):
        """Process result from event extractor."""
        try:
            # Extract basic metrics from event extractor
            if hasattr(extractor_result, 'confidence') and extractor_result.confidence:
                analysis.confidence = max(analysis.confidence, extractor_result.confidence)
            
            if hasattr(extractor_result, 'event_type') and extractor_result.event_type:
                analysis.event_type = extractor_result.event_type
            
            if hasattr(extractor_result, 'locations') and extractor_result.locations:
                analysis.locations.extend(extractor_result.locations)
            
            if hasattr(extractor_result, 'actors') and extractor_result.actors:
                analysis.actors.extend(extractor_result.actors)
            
            if hasattr(extractor_result, 'summary') and extractor_result.summary:
                analysis.event_nature = extractor_result.summary
            
            # Set basic conflict score if event detected
            if analysis.event_type:
                analysis.conflict_score = 0.5
            
        except Exception as e:
            logger.warning(f"Failed to process event extractor result: {e}")
    
    async def _perform_nlp_analysis(self, analysis: ContentAnalysis, text: str):
        """Perform NLP analysis using utils components."""
        try:
            # Sentiment analysis
            if self.sentiment_analyzer:
                sentiment_result = await self.sentiment_analyzer.analyze(text)
                if sentiment_result and hasattr(sentiment_result, 'score'):
                    if analysis.sentiment_score is None:
                        analysis.sentiment_score = sentiment_result.score
                    
                    # Update sentiment label
                    if sentiment_result.score > 0.1:
                        analysis.sentiment_label = "positive"
                    elif sentiment_result.score < -0.1:
                        analysis.sentiment_label = "negative"
                    else:
                        analysis.sentiment_label = "neutral"
            
            # Named entity recognition
            if self.ner:
                entities = await self.ner.extract_entities(text)
                if entities:
                    for entity in entities:
                        entity_dict = entity.to_dict() if hasattr(entity, 'to_dict') else entity
                        analysis.entities.append(entity_dict)
                        
                        # Extract locations and actors
                        if hasattr(entity, 'entity_type'):
                            if entity.entity_type in ['LOCATION', 'GPE']:
                                analysis.locations.append(entity.text)
                            elif entity.entity_type in ['PERSON', 'ORG']:
                                analysis.actors.append(entity.text)
            
            # Topic modeling
            if self.topic_modeler:
                topics = await self.topic_modeler.extract_topics([text], num_topics=3)
                if topics and isinstance(topics, list) and len(topics) > 0:
                    # topics[0] should contain the topics for the first document
                    if isinstance(topics[0], list):
                        analysis.topics.extend(topics[0])
            
            # Text preprocessing for keywords
            if self.text_preprocessor:
                processed = await self.text_preprocessor.process(
                    text,
                    operations=['clean', 'extract_keywords']
                )
                # Extract keywords if available in processed result
                if hasattr(processed, 'keywords'):
                    analysis.keywords.extend(processed.keywords)
            
        except Exception as e:
            logger.warning(f"NLP analysis failed: {e}")
    
    def _calculate_composite_scores(self, analysis: ContentAnalysis):
        """Calculate composite quality and relevance scores."""
        try:
            # Content quality score based on available information
            quality_factors = []
            
            # Confidence factor
            if analysis.confidence > 0:
                quality_factors.append(analysis.confidence)
            
            # Entity extraction factor
            if analysis.entities:
                entity_factor = min(1.0, len(analysis.entities) / 10)
                quality_factors.append(entity_factor)
            
            # Topic factor
            if analysis.topics:
                topic_factor = min(1.0, len(analysis.topics) / 5)
                quality_factors.append(topic_factor)
            
            # Location factor
            if analysis.locations:
                location_factor = min(1.0, len(analysis.locations) / 3)
                quality_factors.append(location_factor)
            
            if quality_factors:
                analysis.content_quality = sum(quality_factors) / len(quality_factors)
            
            # Relevance score based on conflict indicators
            relevance_factors = []
            
            # Conflict score factor
            if analysis.conflict_score > 0:
                relevance_factors.append(analysis.conflict_score)
            
            # Event type factor
            if analysis.event_type:
                relevance_factors.append(0.8)
            
            # Severity factor
            severity_scores = {
                'critical': 1.0,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
            if analysis.severity_level in severity_scores:
                relevance_factors.append(severity_scores[analysis.severity_level])
            
            if relevance_factors:
                analysis.relevance_score = sum(relevance_factors) / len(relevance_factors)
            
            # Credibility score (simplified)
            credibility_factors = []
            
            # Entity consistency factor
            if len(analysis.entities) > 2:
                credibility_factors.append(0.8)
            
            # Location specificity factor
            if analysis.locations:
                credibility_factors.append(0.7)
            
            # Sentiment coherence factor
            if analysis.sentiment_score is not None:
                credibility_factors.append(0.6)
            
            if credibility_factors:
                analysis.credibility_score = sum(credibility_factors) / len(credibility_factors)
            
        except Exception as e:
            logger.warning(f"Failed to calculate composite scores: {e}")
    
    async def analyze_batch(self, content_list: List[str], titles: List[str] = None) -> List[ContentAnalysis]:
        """Analyze multiple content items."""
        titles = titles or [""] * len(content_list)
        results = []
        
        for i, content in enumerate(content_list):
            title = titles[i] if i < len(titles) else ""
            result = await self.analyze(content, title)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        stats = {
            'scorers_available': SCORERS_AVAILABLE,
            'nlp_utils_available': NLP_UTILS_AVAILABLE,
            'enhanced_scorer_available': self.enhanced_scorer is not None,
            'unified_scorer_available': self.unified_scorer is not None,
            'caching_enabled': self.cache_manager is not None,
            'monitoring_enabled': self.performance_monitor is not None
        }
        
        if self.performance_monitor:
            stats['performance_stats'] = self.performance_monitor.get_summary()
        
        return stats


# Utility functions
def quick_analyze_content(content: str, title: str = "", **kwargs) -> ContentAnalysis:
    """Quick content analysis function."""
    analyzer = MLContentAnalyzer(kwargs)
    import asyncio
    return asyncio.run(analyzer.analyze(content, title))


def calculate_content_score(analysis: ContentAnalysis) -> float:
    """Calculate overall content score."""
    factors = [
        analysis.content_quality,
        analysis.relevance_score,
        analysis.credibility_score,
        analysis.confidence
    ]
    
    valid_factors = [f for f in factors if f > 0]
    if valid_factors:
        return sum(valid_factors) / len(valid_factors)
    return 0.0