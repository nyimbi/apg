"""
Twitter Analysis Module
=======================

Advanced text analysis, sentiment analysis, and pattern detection for Twitter data.
Provides sentiment analysis, conflict detection, network analysis, and trend analysis
specifically designed for social media intelligence and conflict monitoring.

Author: Lindela Development Team
Version: 2.0.0
License: MIT
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from .data import TweetModel, UserModel
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import textblob
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

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

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class SentimentPolarity(Enum):
    """Sentiment polarity categories"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ConflictCategory(Enum):
    """Conflict event categories"""
    ARMED_CONFLICT = "armed_conflict"
    TERRORISM = "terrorism"
    PROTESTS = "protests"
    HUMANITARIAN_CRISIS = "humanitarian_crisis"
    POLITICAL_UNREST = "political_unrest"
    NATURAL_DISASTER = "natural_disaster"
    UNKNOWN = "unknown"


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    polarity: float  # -1.0 to 1.0
    subjectivity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    category: SentimentPolarity
    compound_score: Optional[float] = None
    positive_score: Optional[float] = None
    negative_score: Optional[float] = None
    neutral_score: Optional[float] = None
    emotions: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_positive(self) -> bool:
        return self.polarity > 0.1
    
    @property
    def is_negative(self) -> bool:
        return self.polarity < -0.1
    
    @property
    def is_neutral(self) -> bool:
        return -0.1 <= self.polarity <= 0.1


@dataclass
class ConflictAnalysisResult:
    """Conflict relevance analysis result"""
    text: str
    is_conflict_related: bool
    confidence: float
    category: ConflictCategory
    keywords_found: List[str]
    urgency_score: float  # 0.0 to 1.0
    threat_level: str  # low, medium, high, critical
    entities: Dict[str, List[str]] = field(default_factory=dict)  # locations, organizations, etc.
    
    @property
    def is_urgent(self) -> bool:
        return self.urgency_score > 0.7
    
    @property
    def is_critical(self) -> bool:
        return self.threat_level == "critical"


@dataclass
class NetworkAnalysisResult:
    """Social network analysis result"""
    user_id: str
    username: str
    centrality_scores: Dict[str, float]  # betweenness, closeness, degree, etc.
    influence_score: float
    community_id: Optional[int] = None
    connections: List[str] = field(default_factory=list)
    interaction_strength: Dict[str, float] = field(default_factory=dict)


class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple methods"""
    
    def __init__(self):
        self.nltk_analyzer = None
        self.conflict_keywords = self._load_conflict_keywords()
        self.emotion_lexicon = self._load_emotion_lexicon()
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            try:
                self.nltk_analyzer = SentimentIntensityAnalyzer()
                self.stopwords = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                logger.warning("NLTK data not found. Run nltk.download() to install required data.")
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using multiple approaches"""
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # TextBlob analysis
        polarity = 0.0
        subjectivity = 0.0
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        
        # NLTK VADER analysis
        compound_score = None
        positive_score = None
        negative_score = None
        neutral_score = None
        
        if self.nltk_analyzer:
            scores = self.nltk_analyzer.polarity_scores(cleaned_text)
            compound_score = scores['compound']
            positive_score = scores['pos']
            negative_score = scores['neg']
            neutral_score = scores['neu']
            
            # Use VADER compound score if TextBlob not available
            if not TEXTBLOB_AVAILABLE:
                polarity = compound_score
        
        # Determine category
        category = self._categorize_sentiment(polarity)
        
        # Calculate confidence
        confidence = self._calculate_confidence(polarity, subjectivity)
        
        # Emotion analysis
        emotions = self._analyze_emotions(cleaned_text)
        
        return SentimentResult(
            text=text,
            polarity=polarity,
            subjectivity=subjectivity,
            confidence=confidence,
            category=category,
            compound_score=compound_score,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            emotions=emotions
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags for sentiment analysis
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _categorize_sentiment(self, polarity: float) -> SentimentPolarity:
        """Categorize sentiment based on polarity score"""
        if polarity >= 0.5:
            return SentimentPolarity.VERY_POSITIVE
        elif polarity >= 0.1:
            return SentimentPolarity.POSITIVE
        elif polarity <= -0.5:
            return SentimentPolarity.VERY_NEGATIVE
        elif polarity <= -0.1:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL
    
    def _calculate_confidence(self, polarity: float, subjectivity: float) -> float:
        """Calculate confidence score for sentiment analysis"""
        # Higher confidence for more extreme polarities and objective statements
        polarity_confidence = abs(polarity)
        subjectivity_confidence = 1.0 - abs(subjectivity - 0.5) * 2  # Peak at 0.5 subjectivity
        
        return (polarity_confidence + subjectivity_confidence) / 2
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Basic emotion analysis using keyword matching"""
        emotions = {
            'anger': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0,
            'disgust': 0.0, 'surprise': 0.0, 'trust': 0.0, 'anticipation': 0.0
        }
        
        words = text.lower().split()
        
        for word in words:
            if word in self.emotion_lexicon:
                for emotion, score in self.emotion_lexicon[word].items():
                    emotions[emotion] += score
        
        # Normalize scores
        total_score = sum(emotions.values())
        if total_score > 0:
            emotions = {k: v / total_score for k, v in emotions.items()}
        
        return emotions
    
    def _load_conflict_keywords(self) -> Dict[str, List[str]]:
        """Load conflict-related keywords"""
        return {
            'violence': [
                'kill', 'murder', 'death', 'violence', 'attack', 'assault',
                'bomb', 'explosion', 'shooting', 'stabbing', 'terror'
            ],
            'military': [
                'army', 'military', 'soldier', 'troop', 'weapon', 'gun',
                'missile', 'tank', 'aircraft', 'navy', 'air force'
            ],
            'conflict': [
                'war', 'conflict', 'battle', 'fight', 'clash', 'combat',
                'siege', 'invasion', 'occupation', 'resistance'
            ],
            'crisis': [
                'crisis', 'emergency', 'disaster', 'catastrophe', 'tragedy',
                'evacuation', 'refugee', 'displaced', 'humanitarian'
            ]
        }
    
    def _load_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load basic emotion lexicon"""
        return {
            # Joy
            'happy': {'joy': 0.8}, 'celebration': {'joy': 0.7}, 'victory': {'joy': 0.6},
            'success': {'joy': 0.5}, 'wonderful': {'joy': 0.7},
            
            # Anger
            'angry': {'anger': 0.8}, 'hate': {'anger': 0.9}, 'furious': {'anger': 0.9},
            'outrage': {'anger': 0.8}, 'rage': {'anger': 0.9},
            
            # Fear
            'afraid': {'fear': 0.8}, 'terror': {'fear': 0.9}, 'panic': {'fear': 0.8},
            'scared': {'fear': 0.7}, 'worried': {'fear': 0.6},
            
            # Sadness
            'sad': {'sadness': 0.8}, 'grief': {'sadness': 0.9}, 'cry': {'sadness': 0.7},
            'mourn': {'sadness': 0.8}, 'tragedy': {'sadness': 0.8},
            
            # Disgust
            'disgusting': {'disgust': 0.8}, 'horrible': {'disgust': 0.7},
            'awful': {'disgust': 0.7}, 'revolting': {'disgust': 0.9}
        }


class ConflictAnalyzer:
    """Analyzer for conflict-related content"""
    
    def __init__(self):
        self.conflict_patterns = self._load_conflict_patterns()
        self.urgency_indicators = self._load_urgency_indicators()
        self.threat_keywords = self._load_threat_keywords()
        self.location_patterns = self._compile_location_patterns()
    
    def analyze_conflict_relevance(self, text: str) -> ConflictAnalysisResult:
        """Analyze if text is conflict-related and extract information"""
        cleaned_text = text.lower()
        
        # Find matching keywords
        keywords_found = []
        category_scores = defaultdict(float)
        
        for category, patterns in self.conflict_patterns.items():
            for pattern in patterns:
                if isinstance(pattern, str):
                    if pattern in cleaned_text:
                        keywords_found.append(pattern)
                        category_scores[category] += 1.0
                else:  # regex pattern
                    matches = pattern.findall(cleaned_text)
                    if matches:
                        keywords_found.extend(matches)
                        category_scores[category] += len(matches) * 0.8
        
        # Determine if conflict-related
        total_score = sum(category_scores.values())
        is_conflict_related = total_score >= 1.0
        confidence = min(total_score / 3.0, 1.0)  # Normalize to 0-1
        
        # Determine category
        if category_scores:
            category = ConflictCategory(max(category_scores.items(), key=lambda x: x[1])[0])
        else:
            category = ConflictCategory.UNKNOWN
        
        # Calculate urgency score
        urgency_score = self._calculate_urgency(text, keywords_found)
        
        # Determine threat level
        threat_level = self._determine_threat_level(urgency_score, category_scores)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        return ConflictAnalysisResult(
            text=text,
            is_conflict_related=is_conflict_related,
            confidence=confidence,
            category=category,
            keywords_found=keywords_found,
            urgency_score=urgency_score,
            threat_level=threat_level,
            entities=entities
        )
    
    def _calculate_urgency(self, text: str, keywords_found: List[str]) -> float:
        """Calculate urgency score based on indicators"""
        urgency_score = 0.0
        text_lower = text.lower()
        
        # Check for urgency indicators
        for indicator, weight in self.urgency_indicators.items():
            if indicator in text_lower:
                urgency_score += weight
        
        # Time-based urgency
        time_words = ['now', 'urgent', 'immediate', 'breaking', 'just', 'happening']
        for word in time_words:
            if word in text_lower:
                urgency_score += 0.2
        
        # Caps lock indicates urgency
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            urgency_score += 0.3
        
        # Multiple exclamation marks
        exclamation_count = text.count('!')
        urgency_score += min(exclamation_count * 0.1, 0.5)
        
        return min(urgency_score, 1.0)
    
    def _determine_threat_level(self, urgency_score: float, category_scores: Dict[str, float]) -> str:
        """Determine threat level based on urgency and content"""
        max_category_score = max(category_scores.values()) if category_scores else 0
        
        combined_score = (urgency_score + max_category_score / 3.0) / 2.0
        
        if combined_score >= 0.8:
            return "critical"
        elif combined_score >= 0.6:
            return "high"
        elif combined_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (basic implementation)"""
        entities = {
            'locations': [],
            'organizations': [],
            'persons': [],
            'dates': []
        }
        
        # Basic location extraction using patterns
        location_matches = self.location_patterns.findall(text)
        entities['locations'] = list(set(location_matches))
        
        # Basic organization extraction
        org_patterns = [
            r'\b[A-Z][a-z]+ (?:Army|Force|Group|Organization|Party|Movement)\b',
            r'\b(?:UN|NATO|EU|ISIS|Al[- ]Qaeda)\b'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities['organizations'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _load_conflict_patterns(self) -> Dict[str, List[Union[str, re.Pattern]]]:
        """Load conflict detection patterns"""
        return {
            'armed_conflict': [
                'armed conflict', 'military operation', 'battle', 'combat',
                'warfare', 'fighting', 'clashes', 'skirmish', 'engagement'
            ],
            'terrorism': [
                'terrorist attack', 'terror', 'bombing', 'suicide bomber',
                'explosion', 'blast', 'extremist', 'radical'
            ],
            'protests': [
                'protest', 'demonstration', 'riot', 'unrest', 'uprising',
                'march', 'rally', 'civil disobedience', 'strike'
            ],
            'humanitarian_crisis': [
                'humanitarian crisis', 'refugee', 'displaced', 'evacuation',
                'famine', 'epidemic', 'aid', 'relief', 'emergency'
            ],
            'political_unrest': [
                'coup', 'revolution', 'government', 'election', 'political',
                'regime', 'dictatorship', 'democracy', 'parliament'
            ],
            'natural_disaster': [
                'earthquake', 'tsunami', 'hurricane', 'flood', 'drought',
                'wildfire', 'volcano', 'storm', 'cyclone', 'disaster'
            ]
        }
    
    def _load_urgency_indicators(self) -> Dict[str, float]:
        """Load urgency indicator keywords and weights"""
        return {
            'breaking': 0.8,
            'urgent': 0.7,
            'emergency': 0.8,
            'immediate': 0.6,
            'critical': 0.7,
            'alert': 0.6,
            'warning': 0.5,
            'happening now': 0.9,
            'live': 0.5,
            'ongoing': 0.4,
            'developing': 0.3
        }
    
    def _load_threat_keywords(self) -> Dict[str, float]:
        """Load threat-related keywords"""
        return {
            'kill': 0.9, 'death': 0.8, 'murder': 0.9, 'attack': 0.7,
            'bomb': 0.8, 'explosion': 0.7, 'shooting': 0.8, 'violence': 0.6,
            'threat': 0.5, 'danger': 0.5, 'crisis': 0.6, 'emergency': 0.7
        }
    
    def _compile_location_patterns(self) -> re.Pattern:
        """Compile regex patterns for location extraction"""
        # Basic patterns for location names
        pattern = r'\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+)*)\b(?=\s+(?:city|town|village|country|region|province|state))'
        return re.compile(pattern)


class NetworkAnalyzer:
    """Social network analysis for Twitter data"""
    
    def __init__(self):
        self.interaction_graph = None
        if NETWORKX_AVAILABLE:
            self.interaction_graph = nx.DiGraph()
    
    def build_network(self, tweets: List['TweetModel']) -> None:
        """Build interaction network from tweets"""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available for network analysis")
            return
        
        self.interaction_graph.clear()
        
        for tweet in tweets:
            user_id = tweet.user_id
            
            if not user_id:
                continue
            
            # Add user node
            self.interaction_graph.add_node(
                user_id,
                username=tweet.username,
                followers=tweet.user_followers_count,
                verified=tweet.user_verified
            )
            
            # Add reply relationships
            if tweet.in_reply_to_user_id:
                self.interaction_graph.add_edge(
                    user_id,
                    tweet.in_reply_to_user_id,
                    interaction_type='reply',
                    timestamp=tweet.created_at
                )
            
            # Add mention relationships
            for mention in tweet.mentions:
                mention_clean = mention.lstrip('@')
                self.interaction_graph.add_edge(
                    user_id,
                    mention_clean,
                    interaction_type='mention',
                    timestamp=tweet.created_at
                )
    
    def analyze_user_influence(self, user_id: str) -> Optional[NetworkAnalysisResult]:
        """Analyze user's influence in the network"""
        if not NETWORKX_AVAILABLE or not self.interaction_graph.has_node(user_id):
            return None
        
        try:
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(self.interaction_graph)[user_id]
            
            # Calculate other centralities if graph is not too large
            centrality_scores = {'degree': degree_centrality}
            
            if len(self.interaction_graph) < 1000:  # Limit for performance
                try:
                    betweenness = nx.betweenness_centrality(self.interaction_graph)[user_id]
                    closeness = nx.closeness_centrality(self.interaction_graph)[user_id]
                    centrality_scores['betweenness'] = betweenness
                    centrality_scores['closeness'] = closeness
                except:
                    pass  # Skip if computation fails
            
            # Calculate influence score
            influence_score = self._calculate_influence_score(user_id, centrality_scores)
            
            # Get connections
            connections = list(self.interaction_graph.neighbors(user_id))
            
            # Get interaction strengths
            interaction_strength = {}
            for neighbor in connections:
                edge_data = self.interaction_graph[user_id][neighbor]
                interaction_strength[neighbor] = 1.0  # Could be enhanced with frequency
            
            # Get user data
            node_data = self.interaction_graph.nodes[user_id]
            username = node_data.get('username', user_id)
            
            return NetworkAnalysisResult(
                user_id=user_id,
                username=username,
                centrality_scores=centrality_scores,
                influence_score=influence_score,
                connections=connections,
                interaction_strength=interaction_strength
            )
            
        except Exception as e:
            logger.error(f"Error analyzing user influence for {user_id}: {e}")
            return None
    
    def _calculate_influence_score(self, user_id: str, centrality_scores: Dict[str, float]) -> float:
        """Calculate overall influence score"""
        base_score = centrality_scores.get('degree', 0.0)
        
        # Weight by other centrality measures if available
        if 'betweenness' in centrality_scores:
            base_score += centrality_scores['betweenness'] * 0.5
        
        if 'closeness' in centrality_scores:
            base_score += centrality_scores['closeness'] * 0.3
        
        # Factor in user attributes
        node_data = self.interaction_graph.nodes[user_id]
        
        if node_data.get('verified', False):
            base_score *= 1.2
        
        followers = node_data.get('followers', 0)
        if followers > 10000:
            base_score *= 1.1
        elif followers > 100000:
            base_score *= 1.3
        
        return min(base_score, 1.0)
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities in the network"""
        if not NETWORKX_AVAILABLE or len(self.interaction_graph) == 0:
            return {}
        
        try:
            # Use simple connected components for community detection
            communities = {}
            for i, component in enumerate(nx.weakly_connected_components(self.interaction_graph)):
                for node in component:
                    communities[node] = i
            
            return communities
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return {}


class TwitterAnalyzer:
    """Main Twitter analysis coordinator"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conflict_analyzer = ConflictAnalyzer()
        self.network_analyzer = NetworkAnalyzer()
        
        self.analysis_cache = {}
        self.stats = {
            'total_analyzed': 0,
            'sentiment_analyzed': 0,
            'conflict_analyzed': 0,
            'network_analyzed': 0
        }
    
    def analyze_tweet(self, tweet: 'TweetModel', include_network: bool = False) -> Dict[str, Any]:
        """Comprehensive tweet analysis"""
        results = {}
        
        try:
            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(tweet.text)
            results['sentiment'] = sentiment_result
            self.stats['sentiment_analyzed'] += 1
            
            # Conflict analysis
            conflict_result = self.conflict_analyzer.analyze_conflict_relevance(tweet.text)
            results['conflict'] = conflict_result
            self.stats['conflict_analyzed'] += 1
            
            # Network analysis (if requested and data available)
            if include_network and tweet.user_id:
                network_result = self.network_analyzer.analyze_user_influence(tweet.user_id)
                if network_result:
                    results['network'] = network_result
                    self.stats['network_analyzed'] += 1
            
            self.stats['total_analyzed'] += 1
            
        except Exception as e:
            logger.error(f"Error analyzing tweet {tweet.id}: {e}")
        
        return results
    
    def analyze_tweet_batch(
        self,
        tweets: List['TweetModel'],
        include_network: bool = False
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of tweets"""
        if include_network:
            # Build network first
            self.network_analyzer.build_network(tweets)
        
        results = []
        for tweet in tweets:
            analysis = self.analyze_tweet(tweet, include_network)
            results.append({
                'tweet_id': tweet.id,
                'analysis': analysis
            })
        
        return results
    
    def get_analysis_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from analyses"""
        if not analyses:
            return {}
        
        # Sentiment summary
        sentiments = [a['analysis'].get('sentiment') for a in analyses if 'sentiment' in a.get('analysis', {})]
        sentiment_summary = self._summarize_sentiments(sentiments)
        
        # Conflict summary
        conflicts = [a['analysis'].get('conflict') for a in analyses if 'conflict' in a.get('analysis', {})]
        conflict_summary = self._summarize_conflicts(conflicts)
        
        return {
            'total_tweets': len(analyses),
            'sentiment_summary': sentiment_summary,
            'conflict_summary': conflict_summary,
            'analysis_stats': self.stats.copy()
        }
    
    def _summarize_sentiments(self, sentiments: List[SentimentResult]) -> Dict[str, Any]:
        """Summarize sentiment analysis results"""
        if not sentiments:
            return {}
        
        polarities = [s.polarity for s in sentiments if s]
        categories = [s.category.value for s in sentiments if s]
        
        return {
            'average_polarity': sum(polarities) / len(polarities) if polarities else 0,
            'category_distribution': dict(Counter(categories)),
            'positive_ratio': len([p for p in polarities if p > 0.1]) / len(polarities) if polarities else 0,
            'negative_ratio': len([p for p in polarities if p < -0.1]) / len(polarities) if polarities else 0,
            'neutral_ratio': len([p for p in polarities if -0.1 <= p <= 0.1]) / len(polarities) if polarities else 0
        }
    
    def _summarize_conflicts(self, conflicts: List[ConflictAnalysisResult]) -> Dict[str, Any]:
        """Summarize conflict analysis results"""
        if not conflicts:
            return {}
        
        conflict_related = [c for c in conflicts if c and c.is_conflict_related]
        categories = [c.category.value for c in conflict_related]
        threat_levels = [c.threat_level for c in conflict_related]
        urgency_scores = [c.urgency_score for c in conflict_related]
        
        return {
            'total_conflict_related': len(conflict_related),
            'conflict_ratio': len(conflict_related) / len(conflicts),
            'category_distribution': dict(Counter(categories)),
            'threat_level_distribution': dict(Counter(threat_levels)),
            'average_urgency': sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0,
            'high_urgency_count': len([u for u in urgency_scores if u > 0.7])
        }


# Utility functions
def quick_sentiment_analysis(text: str) -> SentimentResult:
    """Quick sentiment analysis function"""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_sentiment(text)


def quick_conflict_analysis(text: str) -> ConflictAnalysisResult:
    """Quick conflict analysis function"""
    analyzer = ConflictAnalyzer()
    return analyzer.analyze_conflict_relevance(text)


@dataclass
class TrendAnalysisResult:
    """Result of trend analysis."""
    trending_topics: List[str] = field(default_factory=list)
    trend_scores: Dict[str, float] = field(default_factory=dict)
    emerging_trends: List[str] = field(default_factory=list)
    declining_trends: List[str] = field(default_factory=list)
    velocity_scores: Dict[str, float] = field(default_factory=dict)
    time_window: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class TrendAnalyzer:
    """Analyze trends and trending patterns in Twitter data."""
    
    def __init__(self):
        self.trend_history = defaultdict(list)
        self.baseline_scores = {}
        
    def analyze_trends(self, tweets: List[str], time_window: str = "1h") -> TrendAnalysisResult:
        """Analyze trends in a collection of tweets."""
        # Extract hashtags and keywords
        hashtags = []
        keywords = []
        
        for tweet in tweets:
            # Extract hashtags
            tweet_hashtags = re.findall(r'#\w+', tweet.lower())
            hashtags.extend(tweet_hashtags)
            
            # Extract keywords (simple approach)
            words = re.findall(r'\b\w+\b', tweet.lower())
            # Filter out common stop words
            filtered_words = [w for w in words if len(w) > 3 and w not in ['that', 'this', 'with', 'have', 'will', 'been', 'they', 'their']]
            keywords.extend(filtered_words)
        
        # Count occurrences
        hashtag_counts = Counter(hashtags)
        keyword_counts = Counter(keywords)
        
        # Calculate trend scores (simplified scoring)
        trend_scores = {}
        for hashtag, count in hashtag_counts.most_common(20):
            trend_scores[hashtag] = count / len(tweets) if tweets else 0
        
        for keyword, count in keyword_counts.most_common(20):
            if keyword not in trend_scores:  # Avoid duplicate hashtags
                trend_scores[keyword] = count / len(tweets) if tweets else 0
        
        # Identify trending topics (top scoring items)
        trending_topics = list(dict(sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)[:10]).keys())
        
        # Determine emerging trends (simplified approach)
        emerging_trends = [topic for topic, score in trend_scores.items() if score > 0.1][:5]
        
        return TrendAnalysisResult(
            trending_topics=trending_topics,
            trend_scores=trend_scores,
            emerging_trends=emerging_trends,
            declining_trends=[],  # Would need historical data
            velocity_scores={},  # Would need time-series data
            time_window=time_window
        )
    
    def track_trend_velocity(self, topic: str, current_score: float) -> float:
        """Track the velocity of a trend (rate of change)."""
        if topic not in self.trend_history:
            self.trend_history[topic] = []
        
        self.trend_history[topic].append({
            'score': current_score,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history (last 24 hours)
        cutoff = datetime.now() - timedelta(days=1)
        self.trend_history[topic] = [
            entry for entry in self.trend_history[topic] 
            if entry['timestamp'] > cutoff
        ]
        
        # Calculate velocity (simple linear regression slope)
        if len(self.trend_history[topic]) < 2:
            return 0.0
        
        history = self.trend_history[topic]
        n = len(history)
        sum_x = sum(i for i in range(n))
        sum_y = sum(entry['score'] for entry in history)
        sum_xy = sum(i * entry['score'] for i, entry in enumerate(history))
        sum_x2 = sum(i * i for i in range(n))
        
        # Calculate slope (velocity)
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        velocity = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return velocity
    
    def identify_emerging_trends(self, trend_scores: Dict[str, float], threshold: float = 0.05) -> List[str]:
        """Identify emerging trends based on velocity and current scores."""
        emerging = []
        
        for topic, score in trend_scores.items():
            velocity = self.track_trend_velocity(topic, score)
            
            # Consider it emerging if it has positive velocity and decent score
            if velocity > threshold and score > threshold:
                emerging.append(topic)
        
        return emerging[:10]  # Return top 10 emerging trends


def quick_trend_analysis(tweets: List[str]) -> TrendAnalysisResult:
    """Quick function to analyze trends in tweets."""
    analyzer = TrendAnalyzer()
    return analyzer.analyze_trends(tweets)


__all__ = [
    'TwitterAnalyzer', 'SentimentAnalyzer', 'ConflictAnalyzer', 'NetworkAnalyzer', 'TrendAnalyzer',
    'SentimentResult', 'ConflictAnalysisResult', 'NetworkAnalysisResult', 'TrendAnalysisResult',
    'SentimentPolarity', 'ConflictCategory',
    'quick_sentiment_analysis', 'quick_conflict_analysis', 'quick_trend_analysis'
]