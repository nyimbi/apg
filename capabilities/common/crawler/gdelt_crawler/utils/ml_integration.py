"""
GDELT ML Integration and Event Feature Extraction
=================================================

ML integration utilities for GDELT data processing with ML Deep Scorer
integration, feature extraction, and content preparation for machine learning.

Key Features:
- **ML Deep Scorer Integration**: Seamless integration with ML scoring pipeline
- **Feature Extraction**: Event feature extraction for ML models
- **Content Preparation**: Text preprocessing for ML analysis
- **Conflict Detection**: ML-based conflict classification
- **Event Scoring**: Automated event scoring and confidence assessment

ML Operations:
- Content preprocessing and cleaning
- Feature extraction from GDELT events
- ML model input preparation
- Confidence scoring and validation
- Event classification and analysis

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

import re
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import json
import hashlib

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MLFeatures:
    """Feature set for ML processing."""
    text_features: Dict[str, Any] = field(default_factory=dict)
    numeric_features: Dict[str, float] = field(default_factory=dict)
    categorical_features: Dict[str, str] = field(default_factory=dict)
    temporal_features: Dict[str, Any] = field(default_factory=dict)
    geographic_features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLProcessingResult:
    """Result of ML processing."""
    success: bool
    confidence_score: float
    event_nature: Optional[str] = None
    event_summary: Optional[str] = None
    fatalities_count: Optional[int] = None
    casualties_count: Optional[int] = None
    conflict_classification: Optional[str] = None
    thinking_traces: Optional[Dict[str, Any]] = None
    extraction_reasoning: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    model_version: Optional[str] = None
    error_message: Optional[str] = None


class ContentPreprocessor:
    """Preprocesses content for ML analysis."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for ML processing."""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Remove URLs
        cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        # Normalize quotes
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r"[''']", "'", cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[.]{3,}', '...', cleaned)
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def extract_key_phrases(text: str) -> List[str]:
        """Extract key phrases that might indicate events."""
        if not text:
            return []
        
        # Event indicator patterns
        event_patterns = [
            r'\b(?:killed|died|dead|death|fatalities?)\b',
            r'\b(?:injured|wounded|hurt|casualties?)\b',
            r'\b(?:attack|assault|bombing|explosion)\b',
            r'\b(?:conflict|war|battle|fighting)\b',
            r'\b(?:protest|demonstration|rally)\b',
            r'\b(?:arrest|detained|captured)\b',
            r'\b(?:displaced|refugee|evacuation)\b',
            r'\b(?:crisis|emergency|disaster)\b'
        ]
        
        phrases = []
        text_lower = text.lower()
        
        for pattern in event_patterns:
            matches = re.findall(pattern, text_lower)
            phrases.extend(matches)
        
        return list(set(phrases))  # Remove duplicates
    
    @staticmethod
    def extract_numbers(text: str) -> Dict[str, List[int]]:
        """Extract numbers that might indicate casualties or counts."""
        if not text:
            return {}
        
        # Patterns for different types of numbers
        patterns = {
            'casualties': r'(\d+)\s*(?:killed|dead|died|fatalities?|casualties?)',
            'injured': r'(\d+)\s*(?:injured|wounded|hurt)',
            'displaced': r'(\d+)\s*(?:displaced|refugees?|evacuated)',
            'general_numbers': r'\b(\d+)\b'
        }
        
        extracted = {}
        text_lower = text.lower()
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                extracted[category] = [int(m) for m in matches if m.isdigit()]
        
        return extracted


class EventFeatureExtractor:
    """Extracts features from GDELT events for ML processing."""
    
    def __init__(self):
        self.preprocessor = ContentPreprocessor()
    
    def extract_features(self, event_data: Dict[str, Any]) -> MLFeatures:
        """Extract comprehensive features from a GDELT event."""
        features = MLFeatures()
        
        # Text features
        features.text_features = self._extract_text_features(event_data)
        
        # Numeric features
        features.numeric_features = self._extract_numeric_features(event_data)
        
        # Categorical features
        features.categorical_features = self._extract_categorical_features(event_data)
        
        # Temporal features
        features.temporal_features = self._extract_temporal_features(event_data)
        
        # Geographic features
        features.geographic_features = self._extract_geographic_features(event_data)
        
        # Metadata
        features.metadata = {
            'external_id': event_data.get('external_id'),
            'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
            'feature_version': '1.0'
        }
        
        return features
    
    def _extract_text_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text-based features."""
        text_features = {}
        
        # Main content
        title = event_data.get('title', '')
        content = event_data.get('content', '')
        summary = event_data.get('event_summary', '')
        
        # Clean text
        clean_title = self.preprocessor.clean_text(title)
        clean_content = self.preprocessor.clean_text(content)
        clean_summary = self.preprocessor.clean_text(summary)
        
        text_features.update({
            'title': clean_title,
            'content': clean_content,
            'summary': clean_summary,
            'combined_text': f"{clean_title} {clean_content} {clean_summary}".strip()
        })
        
        # Extract key phrases
        combined_text = text_features['combined_text']
        text_features['key_phrases'] = self.preprocessor.extract_key_phrases(combined_text)
        text_features['extracted_numbers'] = self.preprocessor.extract_numbers(combined_text)
        
        # Text statistics
        text_features['title_length'] = len(clean_title)
        text_features['content_length'] = len(clean_content)
        text_features['word_count'] = len(combined_text.split()) if combined_text else 0
        
        return text_features
    
    def _extract_numeric_features(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric features."""
        numeric_features = {}
        
        # GDELT-specific numeric fields
        numeric_fields = [
            'goldstein_scale', 'avg_tone', 'num_mentions', 'num_sources',
            'fatalities_count', 'casualties_count', 'people_displaced',
            'extraction_confidence_score', 'relevance_score', 'sentiment_score'
        ]
        
        for field in numeric_fields:
            value = event_data.get(field)
            if value is not None:
                try:
                    numeric_features[field] = float(value)
                except (ValueError, TypeError):
                    numeric_features[field] = 0.0
        
        # Calculated features
        if 'latitude' in event_data and 'longitude' in event_data:
            lat = event_data.get('latitude')
            lon = event_data.get('longitude')
            if lat is not None and lon is not None:
                numeric_features['has_coordinates'] = 1.0
                numeric_features['latitude'] = float(lat)
                numeric_features['longitude'] = float(lon)
            else:
                numeric_features['has_coordinates'] = 0.0
        
        # Severity indicators
        numeric_features['severity_score'] = self._calculate_severity_score(event_data)
        
        return numeric_features
    
    def _extract_categorical_features(self, event_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract categorical features."""
        categorical_features = {}
        
        # GDELT categorical fields
        categorical_fields = [
            'event_nature', 'conflict_classification', 'country',
            'unit_type', 'verification_status', 'event_severity'
        ]
        
        for field in categorical_fields:
            value = event_data.get(field)
            if value:
                categorical_features[field] = str(value)
        
        # Derived categories
        if event_data.get('goldstein_scale') is not None:
            goldstein = float(event_data['goldstein_scale'])
            if goldstein <= -5:
                categorical_features['goldstein_category'] = 'very_negative'
            elif goldstein <= -1:
                categorical_features['goldstein_category'] = 'negative'
            elif goldstein <= 1:
                categorical_features['goldstein_category'] = 'neutral'
            elif goldstein <= 5:
                categorical_features['goldstein_category'] = 'positive'
            else:
                categorical_features['goldstein_category'] = 'very_positive'
        
        return categorical_features
    
    def _extract_temporal_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features."""
        temporal_features = {}
        
        # Event date
        event_date = event_data.get('published_at') or event_data.get('event_date')
        if event_date:
            if isinstance(event_date, str):
                try:
                    event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                except ValueError:
                    event_date = None
            
            if event_date:
                temporal_features.update({
                    'year': event_date.year,
                    'month': event_date.month,
                    'day': event_date.day,
                    'day_of_week': event_date.weekday(),
                    'hour': event_date.hour if hasattr(event_date, 'hour') else 0
                })
                
                # Calculate days since event
                now = datetime.now(timezone.utc)
                if event_date.tzinfo is None:
                    event_date = event_date.replace(tzinfo=timezone.utc)
                
                days_since = (now - event_date).days
                temporal_features['days_since_event'] = days_since
                temporal_features['is_recent'] = days_since <= 7
        
        return temporal_features
    
    def _extract_geographic_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geographic features."""
        geographic_features = {}
        
        # Location information
        geographic_fields = [
            'country', 'location_name', 'region_state', 'city_district'
        ]
        
        for field in geographic_fields:
            value = event_data.get(field)
            if value:
                geographic_features[field] = str(value)
        
        # Coordinate-based features
        lat = event_data.get('latitude')
        lon = event_data.get('longitude')
        
        if lat is not None and lon is not None:
            geographic_features.update({
                'latitude': float(lat),
                'longitude': float(lon),
                'hemisphere': 'north' if lat >= 0 else 'south',
                'region': self._get_world_region(lat, lon)
            })
        
        return geographic_features
    
    def _calculate_severity_score(self, event_data: Dict[str, Any]) -> float:
        """Calculate a composite severity score."""
        score = 0.0
        
        # Fatalities (weighted heavily)
        fatalities = event_data.get('fatalities_count', 0) or 0
        if fatalities > 0:
            score += min(fatalities * 2, 20)  # Cap at 20 points
        
        # Casualties
        casualties = event_data.get('casualties_count', 0) or 0
        if casualties > 0:
            score += min(casualties, 10)  # Cap at 10 points
        
        # Goldstein scale (negative values indicate conflict)
        goldstein = event_data.get('goldstein_scale')
        if goldstein is not None and goldstein < 0:
            score += abs(goldstein)
        
        # Displacement
        displaced = event_data.get('people_displaced', 0) or 0
        if displaced > 0:
            score += min(displaced / 100, 5)  # Scale down and cap
        
        return min(score, 50.0)  # Overall cap
    
    def _get_world_region(self, lat: float, lon: float) -> str:
        """Get world region based on coordinates."""
        # Simplified world region classification
        if 35 <= lat <= 75 and -10 <= lon <= 70:
            return 'europe'
        elif 10 <= lat <= 40 and 25 <= lon <= 180:
            return 'asia'
        elif -35 <= lat <= 40 and -20 <= lon <= 55:
            return 'africa'
        elif 10 <= lat <= 85 and -170 <= lon <= -50:
            return 'north_america'
        elif -60 <= lat <= 15 and -85 <= lon <= -30:
            return 'south_america'
        elif -50 <= lat <= -10 and 110 <= lon <= 180:
            return 'oceania'
        else:
            return 'other'


class MLScorerIntegration:
    """Integration layer for ML Deep Scorer processing."""
    
    def __init__(self):
        self.feature_extractor = EventFeatureExtractor()
        self._ml_scorer = None
        self._scorer_available = False
        
        # Try to initialize ML scorer
        self._initialize_ml_scorer()
    
    def _initialize_ml_scorer(self):
        """Initialize ML Deep Scorer if available."""
        try:
            # This would be the actual ML scorer import
            # from ....ml_scorer import MLDeepScorer
            # self._ml_scorer = MLDeepScorer()
            # self._scorer_available = True
            
            # For now, simulate availability
            self._scorer_available = False
            logger.info("ML Deep Scorer not available - using mock processing")
        except ImportError:
            logger.info("ML Deep Scorer not available")
            self._scorer_available = False
    
    async def process_event(self, event_data: Dict[str, Any]) -> MLProcessingResult:
        """Process a GDELT event with ML scoring."""
        start_time = datetime.now()
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(event_data)
            
            # Process with ML scorer if available
            if self._scorer_available and self._ml_scorer:
                result = await self._process_with_ml_scorer(event_data, features)
            else:
                result = await self._mock_ml_processing(event_data, features)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"ML processing failed: {e}")
            return MLProcessingResult(
                success=False,
                confidence_score=0.0,
                error_message=str(e),
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def _process_with_ml_scorer(
        self,
        event_data: Dict[str, Any],
        features: MLFeatures
    ) -> MLProcessingResult:
        """Process with actual ML Deep Scorer."""
        # This would be the actual ML scorer processing
        # For now, this is a placeholder
        
        content = features.text_features.get('combined_text', '')
        
        # Mock ML processing
        return await self._mock_ml_processing(event_data, features)
    
    async def _mock_ml_processing(
        self,
        event_data: Dict[str, Any],
        features: MLFeatures
    ) -> MLProcessingResult:
        """Mock ML processing for testing/fallback."""
        
        # Extract basic information
        content = features.text_features.get('combined_text', '')
        key_phrases = features.text_features.get('key_phrases', [])
        extracted_numbers = features.text_features.get('extracted_numbers', {})
        
        # Mock event nature detection
        event_nature = self._detect_event_nature(content, key_phrases)
        
        # Mock fatality/casualty extraction
        fatalities = self._extract_fatalities(extracted_numbers)
        casualties = self._extract_casualties(extracted_numbers)
        
        # Mock conflict classification
        conflict_classification = self._classify_conflict(content, key_phrases, fatalities)
        
        # Calculate confidence based on available information
        confidence = self._calculate_confidence(event_data, features)
        
        # Create mock thinking traces
        thinking_traces = {
            'content_analysis': f"Analyzed {len(content)} characters of content",
            'key_phrases_found': key_phrases,
            'numbers_extracted': extracted_numbers,
            'processing_method': 'mock_ml_scorer'
        }
        
        return MLProcessingResult(
            success=True,
            confidence_score=confidence,
            event_nature=event_nature,
            event_summary=self._generate_summary(event_data, event_nature),
            fatalities_count=fatalities,
            casualties_count=casualties,
            conflict_classification=conflict_classification,
            thinking_traces=thinking_traces,
            extraction_reasoning={
                'method': 'rule_based_extraction',
                'confidence_factors': ['text_analysis', 'keyword_matching', 'number_extraction']
            },
            model_version='mock_v1.0'
        )
    
    def _detect_event_nature(self, content: str, key_phrases: List[str]) -> Optional[str]:
        """Detect event nature from content."""
        content_lower = content.lower()
        
        # Violence/conflict indicators
        if any(phrase in ['killed', 'dead', 'attack', 'bombing'] for phrase in key_phrases):
            return 'violent_conflict'
        
        # Protest indicators
        if any(phrase in ['protest', 'demonstration', 'rally'] for phrase in key_phrases):
            return 'protest'
        
        # General conflict
        if any(phrase in ['conflict', 'fighting', 'war'] for phrase in key_phrases):
            return 'conflict'
        
        # Disaster indicators
        if any(word in content_lower for word in ['earthquake', 'flood', 'hurricane', 'disaster']):
            return 'natural_disaster'
        
        return 'general_event'
    
    def _extract_fatalities(self, extracted_numbers: Dict[str, List[int]]) -> Optional[int]:
        """Extract fatality count from numbers."""
        casualties_numbers = extracted_numbers.get('casualties', [])
        if casualties_numbers:
            return max(casualties_numbers)  # Take the highest number
        return None
    
    def _extract_casualties(self, extracted_numbers: Dict[str, List[int]]) -> Optional[int]:
        """Extract casualty count from numbers."""
        injured_numbers = extracted_numbers.get('injured', [])
        if injured_numbers:
            return max(injured_numbers)  # Take the highest number
        return None
    
    def _classify_conflict(self, content: str, key_phrases: List[str], fatalities: Optional[int]) -> Optional[str]:
        """Classify conflict type."""
        if fatalities and fatalities > 0:
            if fatalities >= 25:
                return 'high_intensity_conflict'
            elif fatalities >= 5:
                return 'medium_intensity_conflict'
            else:
                return 'low_intensity_conflict'
        
        if any(phrase in ['attack', 'bombing', 'fighting'] for phrase in key_phrases):
            return 'violent_conflict'
        
        if any(phrase in ['protest', 'demonstration'] for phrase in key_phrases):
            return 'non_violent_conflict'
        
        return None
    
    def _calculate_confidence(self, event_data: Dict[str, Any], features: MLFeatures) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.5  # Base confidence
        
        # Boost for structured data
        if event_data.get('goldstein_scale') is not None:
            confidence += 0.1
        
        if event_data.get('avg_tone') is not None:
            confidence += 0.1
        
        # Boost for good text content
        text_length = features.text_features.get('content_length', 0)
        if text_length > 100:
            confidence += 0.1
        if text_length > 500:
            confidence += 0.1
        
        # Boost for key phrases
        key_phrases = features.text_features.get('key_phrases', [])
        if len(key_phrases) > 0:
            confidence += 0.1
        
        # Boost for numbers
        extracted_numbers = features.text_features.get('extracted_numbers', {})
        if extracted_numbers:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_summary(self, event_data: Dict[str, Any], event_nature: Optional[str]) -> str:
        """Generate event summary."""
        location = event_data.get('location_name', 'unknown location')
        country = event_data.get('country', '')
        
        if event_nature == 'violent_conflict':
            return f"Violent conflict event in {location}, {country}"
        elif event_nature == 'protest':
            return f"Protest or demonstration in {location}, {country}"
        elif event_nature == 'natural_disaster':
            return f"Natural disaster event in {location}, {country}"
        else:
            return f"Event in {location}, {country}"


# Utility functions
def prepare_content_for_ml(
    title: str,
    content: str,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Prepare content for ML processing.
    
    Args:
        title: Event title
        content: Event content
        metadata: Additional metadata
        
    Returns:
        Prepared content dictionary
    """
    preprocessor = ContentPreprocessor()
    
    return {
        'title': preprocessor.clean_text(title),
        'content': preprocessor.clean_text(content),
        'key_phrases': preprocessor.extract_key_phrases(f"{title} {content}"),
        'extracted_numbers': preprocessor.extract_numbers(f"{title} {content}"),
        'metadata': metadata or {}
    }


def extract_event_features(event_data: Dict[str, Any]) -> MLFeatures:
    """
    Extract features from GDELT event data.
    
    Args:
        event_data: GDELT event dictionary
        
    Returns:
        Extracted features
    """
    extractor = EventFeatureExtractor()
    return extractor.extract_features(event_data)


# Export all components
__all__ = [
    'MLScorerIntegration',
    'EventFeatureExtractor',
    'ContentPreprocessor',
    'MLFeatures',
    'MLProcessingResult',
    'prepare_content_for_ml',
    'extract_event_features'
]