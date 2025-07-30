"""
GDELT ML Integration Wrapper
============================

A wrapper around the utils/ml and utils/nlp packages that provides GDELT-specific
ML processing functionality while leveraging the existing ML infrastructure.

This module replaces the mock ML implementation with proper integration
of the packages_enhanced/utils/ml and utils/nlp systems.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

# Import from utils packages
from ....utils.ml import MLPipeline, FeatureEngineering
from ....utils.ml.preprocessing import DataPreprocessor
from ....utils.nlp import (
    TextPreprocessor,
    SentimentAnalyzer,
    NamedEntityRecognizer,
    TopicModeler,
    LanguageDetector
)
from ....utils.nlp.classification import TextClassifier

logger = logging.getLogger(__name__)


@dataclass
class GDELTMLResult:
    """GDELT-specific ML processing result."""
    success: bool
    confidence_score: float
    event_nature: Optional[str] = None
    event_summary: Optional[str] = None
    fatalities_count: Optional[int] = None
    casualties_count: Optional[int] = None
    conflict_classification: Optional[str] = None
    sentiment_score: Optional[float] = None
    entities: List[Dict[str, Any]] = None
    topics: List[str] = None
    language: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = None


class GDELTMLWrapper:
    """
    Wrapper for GDELT-specific ML operations using utils/ml and utils/nlp.

    This replaces the mock ML implementation with proper integration
    of the existing ML infrastructure.
    """

    def __init__(self, ml_config: Optional[Dict[str, Any]] = None):
        """
        Initialize GDELT ML wrapper.

        Args:
            ml_config: Configuration for ML components
        """
        config = ml_config or {}

        # Initialize ML components from utils
        self.ml_pipeline = MLPipeline(config.get('pipeline_config', {}))
        self.feature_engineering = FeatureEngineering()
        self.data_preprocessor = DataPreprocessor()

        # Initialize NLP components
        self.text_preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ner = NamedEntityRecognizer()
        self.topic_modeler = TopicModeler()
        self.language_detector = LanguageDetector()

        # Initialize text classifier for event classification
        self.event_classifier = TextClassifier(
            model_type='transformer',
            model_name='distilbert-base-uncased',
            num_classes=len(self._get_event_classes())
        )

        # Initialize existing ML scorers from the scorers package
        self.ml_scorer = None
        self.unified_scorer = None

        # Try to import and initialize existing ML scorers
        try:
            from ....scorers import (
                create_enhanced_ml_scorer,
                create_unified_scorer,
                EnhancedMLScorerConfig,
                LLMProvider
            )

            # Initialize enhanced ML scorer
            scorer_config = EnhancedMLScorerConfig(
                provider=LLMProvider.OLLAMA,  # Use Ollama as default
                model_name="qwen3:1.7b",
                enable_reasoning=True,
                enable_confidence_scoring=True
            )
            self.ml_scorer = create_enhanced_ml_scorer(scorer_config)
            logger.info("Enhanced ML scorer initialized")

            # Initialize unified scorer for comprehensive analysis
            self.unified_scorer = create_unified_scorer()
            logger.info("Unified scorer initialized")

        except ImportError as e:
            logger.warning(f"Could not initialize existing ML scorers: {e}")
            logger.info("Using basic ML components only")

        # GDELT-specific configurations
        self.event_keywords = {
            'conflict': ['conflict', 'war', 'battle', 'fighting', 'combat', 'clash'],
            'violence': ['killed', 'dead', 'death', 'attack', 'bombing', 'shooting'],
            'protest': ['protest', 'demonstration', 'rally', 'march', 'strike'],
            'disaster': ['earthquake', 'flood', 'hurricane', 'disaster', 'emergency'],
            'political': ['election', 'vote', 'parliament', 'government', 'minister'],
            'economic': ['economy', 'market', 'trade', 'finance', 'inflation']
        }

    async def process_event(self, event_data: Dict[str, Any]) -> GDELTMLResult:
        """
        Process a GDELT event with ML pipeline using existing scorers.

        Args:
            event_data: GDELT event data

        Returns:
            GDELT ML processing result
        """
        start_time = datetime.now()

        try:
            # Prepare text content
            text_content = self._prepare_text_content(event_data)

            if not text_content:
                return GDELTMLResult(
                    success=False,
                    confidence_score=0.0,
                    metadata={'error': 'No text content available'}
                )

            # Use existing ML scorer if available
            if self.ml_scorer:
                try:
                    # Use enhanced ML scorer for comprehensive extraction
                    scorer_result = await self.ml_scorer.extract_from_text(text_content)

                    # Extract information from scorer result
                    confidence = scorer_result.confidence_score if hasattr(scorer_result, 'confidence_score') else 0.8
                    event_nature = scorer_result.event_type if hasattr(scorer_result, 'event_type') else None
                    event_summary = scorer_result.event_summary if hasattr(scorer_result, 'event_summary') else None
                    fatalities = scorer_result.fatalities_count if hasattr(scorer_result, 'fatalities_count') else None
                    casualties = scorer_result.casualties_count if hasattr(scorer_result, 'casualties_count') else None
                    sentiment_score = scorer_result.sentiment_score if hasattr(scorer_result, 'sentiment_score') else None
                    entities = scorer_result.entities if hasattr(scorer_result, 'entities') else []

                    processing_time = (datetime.now() - start_time).total_seconds() * 1000

                    return GDELTMLResult(
                        success=True,
                        confidence_score=confidence,
                        event_nature=event_nature,
                        event_summary=event_summary,
                        fatalities_count=fatalities,
                        casualties_count=casualties,
                        conflict_classification=self._determine_conflict_class(
                            event_data, event_nature, fatalities, sentiment_score
                        ),
                        sentiment_score=sentiment_score,
                        entities=entities if isinstance(entities, list) else [],
                        topics=[],  # Topics will be extracted separately if needed
                        language=None,  # Language detection will be done separately if needed
                        processing_time_ms=processing_time,
                        metadata={
                            'ml_scorer_used': 'enhanced_ml_scorer',
                            'goldstein_scale': event_data.get('goldstein_scale'),
                            'avg_tone': event_data.get('avg_tone')
                        }
                    )

                except Exception as e:
                    logger.warning(f"Enhanced ML scorer failed, falling back to basic processing: {e}")

            # Fallback to unified scorer if enhanced scorer fails
            if self.unified_scorer:
                try:
                    unified_result = await self.unified_scorer.score_single(text_content)

                    processing_time = (datetime.now() - start_time).total_seconds() * 1000

                    return GDELTMLResult(
                        success=True,
                        confidence_score=unified_result.confidence,
                        event_nature=unified_result.event_type,
                        event_summary=f"{unified_result.event_type} event" if unified_result.event_type else None,
                        fatalities_count=None,
                        casualties_count=None,
                        conflict_classification=unified_result.severity_level,
                        sentiment_score=None,
                        entities=[],
                        topics=[],
                        language=None,
                        processing_time_ms=processing_time,
                        metadata={
                            'ml_scorer_used': 'unified_scorer',
                            'conflict_score': unified_result.conflict_score,
                            'goldstein_scale': event_data.get('goldstein_scale'),
                            'avg_tone': event_data.get('avg_tone')
                        }
                    )

                except Exception as e:
                    logger.warning(f"Unified scorer failed, falling back to basic processing: {e}")

            # Fallback to basic processing using utils components
            # Detect language
            language = await self.language_detector.detect(text_content)

            # Preprocess text
            processed_text = await self.text_preprocessor.process(
                text_content,
                operations=['clean', 'normalize', 'remove_stopwords']
            )

            # Extract features
            features = await self._extract_features(event_data, processed_text)

            # Run basic ML pipeline
            ml_results = await self.ml_pipeline.predict(features)

            # Sentiment analysis
            sentiment = await self.sentiment_analyzer.analyze(text_content)

            # Named entity recognition
            entities = await self.ner.extract_entities(text_content)

            # Topic modeling
            topics = await self.topic_modeler.extract_topics([text_content], num_topics=3)

            # Event classification
            event_nature = await self._classify_event(processed_text, features)

            # Extract specific information
            fatalities, casualties = await self._extract_casualty_counts(text_content, entities)

            # Conflict classification
            conflict_class = await self._classify_conflict(
                event_data,
                event_nature,
                fatalities,
                sentiment.score if sentiment else 0.0
            )

            # Calculate confidence
            confidence = self._calculate_confidence(
                ml_results,
                sentiment,
                entities,
                event_data
            )

            # Generate summary
            summary = await self._generate_summary(
                event_data,
                event_nature,
                entities,
                sentiment
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return GDELTMLResult(
                success=True,
                confidence_score=confidence,
                event_nature=event_nature,
                event_summary=summary,
                fatalities_count=fatalities,
                casualties_count=casualties,
                conflict_classification=conflict_class,
                sentiment_score=sentiment.score if sentiment else None,
                entities=[e.to_dict() for e in entities] if entities else [],
                topics=topics[0] if topics else [],  # First topic cluster
                language=language.language if language else None,
                processing_time_ms=processing_time,
                metadata={
                    'ml_scorer_used': 'basic_pipeline',
                    'ml_model_version': self.ml_pipeline.get_version(),
                    'feature_count': len(features),
                    'goldstein_scale': event_data.get('goldstein_scale'),
                    'avg_tone': event_data.get('avg_tone')
                }
            )

        except Exception as e:
            logger.error(f"ML processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return GDELTMLResult(
                success=False,
                confidence_score=0.0,
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )

    def _prepare_text_content(self, event_data: Dict[str, Any]) -> str:
        """Prepare text content from GDELT event."""
        parts = []

        # Add title
        if event_data.get('title'):
            parts.append(event_data['title'])

        # Add content
        if event_data.get('content'):
            parts.append(event_data['content'])

        # Add event summary
        if event_data.get('event_summary'):
            parts.append(event_data['event_summary'])

        # Add description
        if event_data.get('description'):
            parts.append(event_data['description'])

        return ' '.join(parts)

    async def _extract_features(
        self,
        event_data: Dict[str, Any],
        processed_text: str
    ) -> Dict[str, Any]:
        """Extract features for ML processing."""
        # Use feature engineering from utils
        text_features = await self.feature_engineering.extract_text_features(
            processed_text,
            include_tfidf=True,
            include_embeddings=True
        )

        # Add GDELT-specific features
        gdelt_features = {
            'goldstein_scale': event_data.get('goldstein_scale', 0.0),
            'avg_tone': event_data.get('avg_tone', 0.0),
            'num_mentions': event_data.get('num_mentions', 0),
            'num_sources': event_data.get('num_sources', 0),
            'has_location': 1.0 if event_data.get('latitude') else 0.0,
            'event_code': event_data.get('event_code', ''),
            'actor1_code': event_data.get('actor1_code', ''),
            'actor2_code': event_data.get('actor2_code', '')
        }

        # Combine features
        features = {**text_features, **gdelt_features}

        # Add temporal features
        if event_data.get('event_date'):
            features.update(self._extract_temporal_features(event_data['event_date']))

        return features

    def _extract_temporal_features(self, event_date: Any) -> Dict[str, float]:
        """Extract temporal features from event date."""
        if isinstance(event_date, str):
            try:
                event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            except:
                return {}

        if not isinstance(event_date, datetime):
            return {}

        now = datetime.now(event_date.tzinfo or None)
        days_ago = (now - event_date).days

        return {
            'event_hour': float(event_date.hour),
            'event_day_of_week': float(event_date.weekday()),
            'event_month': float(event_date.month),
            'days_since_event': float(days_ago),
            'is_recent': 1.0 if days_ago <= 7 else 0.0,
            'is_weekend': 1.0 if event_date.weekday() >= 5 else 0.0
        }

    async def _classify_event(
        self,
        processed_text: str,
        features: Dict[str, Any]
    ) -> str:
        """Classify event type using text classifier."""
        # Use keyword matching as primary method
        text_lower = processed_text.lower()

        event_scores = {}
        for event_type, keywords in self.event_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            event_scores[event_type] = score

        # Get event type with highest score
        if event_scores:
            best_type = max(event_scores, key=event_scores.get)
            if event_scores[best_type] > 0:
                return best_type

        # Fallback based on Goldstein scale
        goldstein = features.get('goldstein_scale', 0.0)
        if goldstein <= -5:
            return 'conflict'
        elif goldstein <= -2:
            return 'violence'

        return 'general_event'

    async def _extract_casualty_counts(
        self,
        text: str,
        entities: List[Any]
    ) -> tuple[Optional[int], Optional[int]]:
        """Extract fatality and casualty counts from text."""
        import re

        fatalities = None
        casualties = None

        # Patterns for casualty extraction
        fatality_patterns = [
            r'(\d+)\s*(?:people\s+)?(?:killed|dead|died|fatalities)',
            r'(?:killed|dead|died)\s*(\d+)',
            r'death\s+toll\s*(?:of\s+)?(\d+)'
        ]

        casualty_patterns = [
            r'(\d+)\s*(?:people\s+)?(?:injured|wounded|hurt|casualties)',
            r'(?:injured|wounded|hurt)\s*(\d+)'
        ]

        text_lower = text.lower()

        # Extract fatalities
        for pattern in fatality_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                numbers = [int(m) for m in matches if m.isdigit()]
                if numbers:
                    fatalities = max(numbers)
                    break

        # Extract casualties
        for pattern in casualty_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                numbers = [int(m) for m in matches if m.isdigit()]
                if numbers:
                    casualties = max(numbers)
                    break

        # Also check entities for numeric entities
        for entity in entities:
            if hasattr(entity, 'entity_type') and entity.entity_type == 'NUMBER':
                # Additional logic for number entities if needed
                pass

        return fatalities, casualties

    def _determine_conflict_class(
        self,
        event_data: Dict[str, Any],
        event_nature: Optional[str],
        fatalities: Optional[int],
        sentiment_score: Optional[float]
    ) -> Optional[str]:
        """Determine conflict classification (simplified version of _classify_conflict)."""
        goldstein = event_data.get('goldstein_scale', 0.0)

        # High intensity conflict indicators
        if fatalities and fatalities >= 25:
            return 'high_intensity_conflict'
        elif goldstein <= -7:
            return 'high_intensity_conflict'

        # Medium intensity
        elif fatalities and fatalities >= 5:
            return 'medium_intensity_conflict'
        elif goldstein <= -4:
            return 'medium_intensity_conflict'

        # Low intensity
        elif event_nature in ['conflict', 'violence']:
            return 'low_intensity_conflict'
        elif goldstein <= -1:
            return 'low_intensity_conflict'

        # Non-violent conflict
        elif event_nature == 'protest':
            return 'non_violent_conflict'

        return None

    async def _classify_conflict(
        self,
        event_data: Dict[str, Any],
        event_nature: str,
        fatalities: Optional[int],
        sentiment_score: float
    ) -> Optional[str]:
        """Classify conflict intensity."""
        # Use Goldstein scale as primary indicator
        goldstein = event_data.get('goldstein_scale', 0.0)

        # High intensity conflict indicators
        if fatalities and fatalities >= 25:
            return 'high_intensity_conflict'
        elif goldstein <= -7:
            return 'high_intensity_conflict'

        # Medium intensity
        elif fatalities and fatalities >= 5:
            return 'medium_intensity_conflict'
        elif goldstein <= -4:
            return 'medium_intensity_conflict'

        # Low intensity
        elif event_nature in ['conflict', 'violence']:
            return 'low_intensity_conflict'
        elif goldstein <= -1:
            return 'low_intensity_conflict'

        # Non-violent conflict
        elif event_nature == 'protest':
            return 'non_violent_conflict'

        return None

    def _calculate_confidence(
        self,
        ml_results: Any,
        sentiment: Any,
        entities: List[Any],
        event_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for ML results."""
        confidence = 0.5  # Base confidence

        # ML model confidence
        if hasattr(ml_results, 'confidence'):
            confidence = ml_results.confidence

        # Boost for structured GDELT fields
        if event_data.get('goldstein_scale') is not None:
            confidence += 0.1

        if event_data.get('avg_tone') is not None:
            confidence += 0.1

        # Boost for good sentiment analysis
        if sentiment and hasattr(sentiment, 'confidence'):
            confidence += sentiment.confidence * 0.1

        # Boost for entity extraction
        if entities and len(entities) > 0:
            confidence += min(len(entities) * 0.02, 0.1)

        # Boost for multiple sources
        num_sources = event_data.get('num_sources', 0)
        if num_sources > 1:
            confidence += min(num_sources * 0.02, 0.1)

        return min(confidence, 0.95)  # Cap at 95%

    async def _generate_summary(
        self,
        event_data: Dict[str, Any],
        event_nature: str,
        entities: List[Any],
        sentiment: Any
    ) -> str:
        """Generate event summary."""
        location = event_data.get('location_name', 'unknown location')
        country = event_data.get('country', '')

        # Extract key actors from entities
        actors = []
        for entity in entities:
            if hasattr(entity, 'entity_type') and entity.entity_type in ['PERSON', 'ORG']:
                actors.append(entity.text)

        # Build summary based on event nature
        if event_nature == 'conflict':
            summary = f"Conflict event reported in {location}"
        elif event_nature == 'violence':
            summary = f"Violent incident in {location}"
        elif event_nature == 'protest':
            summary = f"Protest or demonstration in {location}"
        elif event_nature == 'disaster':
            summary = f"Disaster event in {location}"
        elif event_nature == 'political':
            summary = f"Political development in {location}"
        else:
            summary = f"Event reported in {location}"

        # Add country if available
        if country:
            summary += f", {country}"

        # Add actors if identified
        if actors:
            summary += f" involving {', '.join(actors[:2])}"

        # Add sentiment indicator
        if sentiment and sentiment.score < -0.5:
            summary += " (negative impact)"
        elif sentiment and sentiment.score > 0.5:
            summary += " (positive development)"

        return summary

    def _get_event_classes(self) -> List[str]:
        """Get list of event classes for classification."""
        return list(self.event_keywords.keys()) + ['general_event']

    async def batch_process_events(
        self,
        events: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> List[GDELTMLResult]:
        """
        Process multiple GDELT events in batches.

        Args:
            events: List of GDELT events
            batch_size: Batch size for processing

        Returns:
            List of ML results
        """
        results = []

        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]

            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self.process_event(event) for event in batch],
                return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    results.append(GDELTMLResult(
                        success=False,
                        confidence_score=0.0,
                        metadata={'error': str(result)}
                    ))
                else:
                    results.append(result)

        return results


# Factory function
def create_gdelt_ml_processor(config: Optional[Dict[str, Any]] = None) -> GDELTMLWrapper:
    """Create GDELT ML processor with configuration."""
    return GDELTMLWrapper(config)


# Export components
__all__ = [
    'GDELTMLWrapper',
    'GDELTMLResult',
    'create_gdelt_ml_processor'
]
