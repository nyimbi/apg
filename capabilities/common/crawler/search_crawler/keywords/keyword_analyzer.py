"""
Keyword Analyzer
================

Analyzes and processes keywords for search optimization and relevance scoring.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class KeywordAnalysis:
    """Results of keyword analysis."""
    keyword: str
    frequency: int
    contexts: List[str]
    sentiment_score: float
    relevance_score: float
    locations: List[str]
    entities: List[str]
    temporal_indicators: List[str]


class KeywordAnalyzer:
    """Analyzes keywords and their effectiveness in search results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Sentiment indicators
        self.positive_indicators = {
            'peace', 'peaceful', 'stability', 'agreement', 'resolution', 'solution',
            'dialogue', 'negotiation', 'cooperation', 'aid', 'help', 'support',
            'recovery', 'rebuild', 'reconstruction', 'ceasefire', 'truce'
        }
        
        self.negative_indicators = {
            'violence', 'conflict', 'war', 'attack', 'bomb', 'kill', 'death',
            'crisis', 'emergency', 'disaster', 'threat', 'danger', 'fear',
            'terror', 'chaos', 'destruction', 'collapse', 'failure'
        }
        
        # Temporal patterns
        self.temporal_patterns = [
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(this|last|next)\s+(week|month|year)\b',
            r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',  # Date patterns
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(recent|recently|latest|current|ongoing)\b'
        ]
        
        # Location patterns (for Horn of Africa)
        self.location_patterns = [
            r'\b(somalia|somali|mogadishu|hargeisa|puntland|somaliland)\b',
            r'\b(ethiopia|ethiopian|addis\s+ababa|tigray|amhara|oromia)\b',
            r'\b(eritrea|eritrean|asmara)\b',
            r'\b(djibouti|djiboutian)\b',
            r'\b(sudan|sudanese|khartoum|darfur)\b',
            r'\b(south\s+sudan|juba|unity|jonglei)\b'
        ]
    
    def analyze_text(self, text: str, keywords: List[str]) -> Dict[str, KeywordAnalysis]:
        """
        Analyze text for keyword effectiveness and context.
        
        Args:
            text: Text to analyze
            keywords: Keywords to look for
            
        Returns:
            Dictionary of keyword analyses
        """
        text_lower = text.lower()
        sentences = self._split_sentences(text)
        
        analyses = {}
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Find frequency
            frequency = text_lower.count(keyword_lower)
            if frequency == 0:
                continue
            
            # Find contexts (sentences containing the keyword)
            contexts = [sent for sent in sentences if keyword_lower in sent.lower()]
            
            # Calculate sentiment score
            sentiment = self._calculate_sentiment(contexts)
            
            # Calculate relevance score
            relevance = self._calculate_relevance(keyword, contexts)
            
            # Extract locations
            locations = self._extract_locations(contexts)
            
            # Extract entities (simplified)
            entities = self._extract_entities(contexts)
            
            # Extract temporal indicators
            temporal = self._extract_temporal_indicators(contexts)
            
            analyses[keyword] = KeywordAnalysis(
                keyword=keyword,
                frequency=frequency,
                contexts=contexts[:5],  # Limit to 5 contexts
                sentiment_score=sentiment,
                relevance_score=relevance,
                locations=locations,
                entities=entities,
                temporal_indicators=temporal
            )
        
        return analyses
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_sentiment(self, contexts: List[str]) -> float:
        """Calculate sentiment score based on contexts."""
        if not contexts:
            return 0.0
        
        total_score = 0.0
        total_indicators = 0
        
        for context in contexts:
            context_lower = context.lower()
            
            # Count positive indicators
            positive_count = sum(1 for word in self.positive_indicators 
                               if word in context_lower)
            
            # Count negative indicators
            negative_count = sum(1 for word in self.negative_indicators 
                               if word in context_lower)
            
            if positive_count > 0 or negative_count > 0:
                # Normalize to -1 to 1 scale
                context_score = (positive_count - negative_count) / (positive_count + negative_count)
                total_score += context_score
                total_indicators += 1
        
        return total_score / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_relevance(self, keyword: str, contexts: List[str]) -> float:
        """Calculate relevance score for keyword in contexts."""
        if not contexts:
            return 0.0
        
        relevance_score = 0.0
        
        # Base score from frequency
        total_length = sum(len(context.split()) for context in contexts)
        keyword_mentions = sum(context.lower().count(keyword.lower()) for context in contexts)
        
        if total_length > 0:
            frequency_score = min(keyword_mentions / total_length * 100, 1.0)
            relevance_score += frequency_score * 0.3
        
        # Bonus for appearing in titles or first sentences
        for context in contexts:
            if len(context) < 100:  # Likely a title or summary
                relevance_score += 0.2
                break
        
        # Bonus for co-occurrence with conflict terms
        conflict_terms = ['conflict', 'violence', 'attack', 'crisis', 'emergency']
        for context in contexts:
            context_lower = context.lower()
            if any(term in context_lower for term in conflict_terms):
                relevance_score += 0.3
                break
        
        # Bonus for location context
        for context in contexts:
            if self._extract_locations([context]):
                relevance_score += 0.2
                break
        
        return min(relevance_score, 1.0)
    
    def _extract_locations(self, contexts: List[str]) -> List[str]:
        """Extract location mentions from contexts."""
        locations = set()
        
        for context in contexts:
            context_lower = context.lower()
            
            for pattern in self.location_patterns:
                matches = re.findall(pattern, context_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # For patterns with groups, take the full match
                        locations.add(match[0] if match[0] else match[1])
                    else:
                        locations.add(match)
        
        return list(locations)
    
    def _extract_entities(self, contexts: List[str]) -> List[str]:
        """Extract named entities from contexts (simplified)."""
        entities = set()
        
        # Simple capitalized word extraction
        for context in contexts:
            # Find capitalized words that aren't sentence starters
            words = context.split()
            for i, word in enumerate(words):
                # Skip first word of sentence and common words
                if (i > 0 and word[0].isupper() and 
                    word.lower() not in self.stop_words and
                    len(word) > 2):
                    entities.add(word)
        
        return list(entities)[:10]  # Limit to 10 entities
    
    def _extract_temporal_indicators(self, contexts: List[str]) -> List[str]:
        """Extract temporal indicators from contexts."""
        temporal_indicators = set()
        
        for context in contexts:
            context_lower = context.lower()
            
            for pattern in self.temporal_patterns:
                matches = re.findall(pattern, context_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # Join tuple elements
                        temporal_indicators.add(' '.join(match))
                    else:
                        temporal_indicators.add(match)
        
        return list(temporal_indicators)
    
    def rank_keywords_by_effectiveness(
        self, 
        analyses: Dict[str, KeywordAnalysis],
        weight_frequency: float = 0.3,
        weight_relevance: float = 0.4,
        weight_sentiment: float = 0.2,
        weight_temporal: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Rank keywords by their effectiveness scores.
        
        Args:
            analyses: Keyword analyses
            weight_frequency: Weight for frequency component
            weight_relevance: Weight for relevance component
            weight_sentiment: Weight for sentiment component (negative sentiment = higher score for conflict monitoring)
            weight_temporal: Weight for temporal component
            
        Returns:
            List of (keyword, score) tuples sorted by effectiveness
        """
        ranked_keywords = []
        
        # Normalize frequency scores
        max_frequency = max((analysis.frequency for analysis in analyses.values()), default=1)
        
        for keyword, analysis in analyses.items():
            # Normalize frequency (0-1)
            freq_score = analysis.frequency / max_frequency
            
            # Relevance score (0-1)
            rel_score = analysis.relevance_score
            
            # Sentiment score (convert to 0-1, where negative is higher for conflict monitoring)
            sent_score = max(0, -analysis.sentiment_score)  # Negative sentiment is good for conflict
            
            # Temporal score (presence of temporal indicators)
            temp_score = 1.0 if analysis.temporal_indicators else 0.0
            
            # Calculate weighted effectiveness score
            effectiveness = (
                freq_score * weight_frequency +
                rel_score * weight_relevance +
                sent_score * weight_sentiment +
                temp_score * weight_temporal
            )
            
            ranked_keywords.append((keyword, effectiveness))
        
        # Sort by effectiveness (highest first)
        ranked_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_keywords
    
    def generate_keyword_report(self, analyses: Dict[str, KeywordAnalysis]) -> str:
        """Generate a text report of keyword analysis."""
        if not analyses:
            return "No keyword analyses available."
        
        report = []
        report.append("KEYWORD ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Keywords analyzed: {len(analyses)}")
        report.append("")
        
        # Get ranked keywords
        ranked = self.rank_keywords_by_effectiveness(analyses)
        
        report.append("TOP PERFORMING KEYWORDS:")
        report.append("-" * 30)
        
        for i, (keyword, score) in enumerate(ranked[:10], 1):
            analysis = analyses[keyword]
            report.append(f"{i}. {keyword} (Score: {score:.3f})")
            report.append(f"   Frequency: {analysis.frequency}")
            report.append(f"   Relevance: {analysis.relevance_score:.3f}")
            report.append(f"   Sentiment: {analysis.sentiment_score:.3f}")
            report.append(f"   Locations: {', '.join(analysis.locations[:3])}")
            if analysis.temporal_indicators:
                report.append(f"   Temporal: {', '.join(analysis.temporal_indicators[:2])}")
            report.append("")
        
        # Summary statistics
        total_frequency = sum(a.frequency for a in analyses.values())
        avg_relevance = sum(a.relevance_score for a in analyses.values()) / len(analyses)
        avg_sentiment = sum(a.sentiment_score for a in analyses.values()) / len(analyses)
        
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 20)
        report.append(f"Total keyword mentions: {total_frequency}")
        report.append(f"Average relevance score: {avg_relevance:.3f}")
        report.append(f"Average sentiment score: {avg_sentiment:.3f}")
        
        # Location distribution
        all_locations = []
        for analysis in analyses.values():
            all_locations.extend(analysis.locations)
        
        if all_locations:
            location_counts = Counter(all_locations)
            report.append("")
            report.append("TOP LOCATIONS MENTIONED:")
            report.append("-" * 25)
            for location, count in location_counts.most_common(5):
                report.append(f"  {location}: {count} mentions")
        
        return "\n".join(report)
    
    def suggest_new_keywords(
        self, 
        analyses: Dict[str, KeywordAnalysis],
        min_frequency: int = 3
    ) -> List[str]:
        """Suggest new keywords based on analysis of contexts."""
        suggested_keywords = set()
        
        # Collect all contexts
        all_contexts = []
        for analysis in analyses.values():
            all_contexts.extend(analysis.contexts)
        
        # Extract frequent words that aren't stop words
        all_words = []
        for context in all_contexts:
            words = re.findall(r'\b\w+\b', context.lower())
            all_words.extend([w for w in words if w not in self.stop_words and len(w) > 3])
        
        word_counts = Counter(all_words)
        
        # Suggest words that appear frequently but aren't in current keywords
        current_keywords = set(keyword.lower() for keyword in analyses.keys())
        
        for word, count in word_counts.most_common(20):
            if count >= min_frequency and word not in current_keywords:
                # Check if it has conflict-related context
                contexts_with_word = [c for c in all_contexts if word in c.lower()]
                if self._has_conflict_context(contexts_with_word):
                    suggested_keywords.add(word)
        
        return list(suggested_keywords)[:10]  # Limit to top 10 suggestions
    
    def _has_conflict_context(self, contexts: List[str]) -> bool:
        """Check if contexts contain conflict-related terms."""
        conflict_indicators = {
            'conflict', 'violence', 'attack', 'war', 'crisis', 'emergency',
            'security', 'threat', 'danger', 'instability', 'unrest'
        }
        
        for context in contexts:
            context_lower = context.lower()
            if any(indicator in context_lower for indicator in conflict_indicators):
                return True
        
        return False