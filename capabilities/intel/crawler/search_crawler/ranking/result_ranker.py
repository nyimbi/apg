"""
Result Ranker
=============

Advanced ranking algorithms for search results with multiple strategies.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Available ranking strategies."""
    RELEVANCE = "relevance"
    FRESHNESS = "freshness"
    AUTHORITY = "authority"
    HYBRID = "hybrid"
    CONFLICT_FOCUSED = "conflict_focused"
    DIVERSITY = "diversity"


@dataclass
class RankingScore:
    """Detailed ranking score breakdown."""
    total_score: float
    relevance_score: float
    freshness_score: float
    authority_score: float
    diversity_score: float
    conflict_score: float
    engine_consensus_score: float
    components: Dict[str, float]


class ResultRanker:
    """Advanced result ranking system with multiple strategies."""
    
    def __init__(self, strategy: str = 'hybrid', config: Optional[Dict[str, Any]] = None):
        """Initialize result ranker."""
        self.strategy = RankingStrategy(strategy)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Ranking weights for hybrid strategy
        self.weights = {
            'relevance': self.config.get('relevance_weight', 0.35),
            'freshness': self.config.get('freshness_weight', 0.25),
            'authority': self.config.get('authority_weight', 0.2),
            'diversity': self.config.get('diversity_weight', 0.1),
            'engine_consensus': self.config.get('consensus_weight', 0.1)
        }
        
        # Authority scores for domains
        self.domain_authority = {
            'reuters.com': 0.95,
            'bbc.com': 0.9,
            'apnews.com': 0.9,
            'cnn.com': 0.85,
            'aljazeera.com': 0.85,
            'france24.com': 0.8,
            'dw.com': 0.8,
            'africanews.com': 0.75,
            'bloomberg.com': 0.85,
            'wsj.com': 0.85,
            'ft.com': 0.8,
            'theguardian.com': 0.8,
            'nytimes.com': 0.85,
            'washingtonpost.com': 0.8,
            'economist.com': 0.85
        }
        
        # Statistics
        self.ranking_stats = {
            'total_rankings': 0,
            'strategy_usage': defaultdict(int),
            'average_scores': defaultdict(list)
        }
    
    def rank(self, results: List[Any], query: str = "") -> List[Any]:
        """
        Rank search results using the configured strategy.
        
        Args:
            results: List of search results to rank
            query: Original search query for relevance calculation
            
        Returns:
            Ranked list of results
        """
        if not results:
            return results
        
        self.ranking_stats['total_rankings'] += 1
        self.ranking_stats['strategy_usage'][self.strategy.value] += 1
        
        # Calculate scores for each result
        scored_results = []
        
        for result in results:
            score = self._calculate_score(result, query, results)
            scored_results.append((result, score))
            
            # Store score in result metadata if possible
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                result.metadata['ranking_score'] = score.total_score
                result.metadata['ranking_breakdown'] = {
                    'relevance': score.relevance_score,
                    'freshness': score.freshness_score,
                    'authority': score.authority_score,
                    'diversity': score.diversity_score,
                    'engine_consensus': score.engine_consensus_score
                }
        
        # Sort by total score (highest first)
        scored_results.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Update statistics
        scores = [score.total_score for _, score in scored_results]
        self.ranking_stats['average_scores'][self.strategy.value].extend(scores)
        
        # Return ranked results
        ranked_results = [result for result, _ in scored_results]
        
        self.logger.debug(
            f"Ranked {len(results)} results using {self.strategy.value} strategy. "
            f"Score range: {min(scores):.3f} - {max(scores):.3f}"
        )
        
        return ranked_results
    
    def _calculate_score(self, result: Any, query: str, all_results: List[Any]) -> RankingScore:
        """Calculate comprehensive ranking score for a result."""
        
        # Calculate individual component scores
        relevance_score = self._calculate_relevance_score(result, query)
        freshness_score = self._calculate_freshness_score(result)
        authority_score = self._calculate_authority_score(result)
        diversity_score = self._calculate_diversity_score(result, all_results)
        consensus_score = self._calculate_engine_consensus_score(result)
        conflict_score = self._calculate_conflict_score(result)
        
        # Apply strategy-specific weighting
        if self.strategy == RankingStrategy.RELEVANCE:
            total_score = relevance_score
        elif self.strategy == RankingStrategy.FRESHNESS:
            total_score = freshness_score
        elif self.strategy == RankingStrategy.AUTHORITY:
            total_score = authority_score
        elif self.strategy == RankingStrategy.CONFLICT_FOCUSED:
            total_score = (
                conflict_score * 0.4 +
                relevance_score * 0.3 +
                freshness_score * 0.2 +
                authority_score * 0.1
            )
        elif self.strategy == RankingStrategy.DIVERSITY:
            total_score = (
                diversity_score * 0.4 +
                relevance_score * 0.3 +
                authority_score * 0.3
            )
        else:  # HYBRID
            total_score = (
                relevance_score * self.weights['relevance'] +
                freshness_score * self.weights['freshness'] +
                authority_score * self.weights['authority'] +
                diversity_score * self.weights['diversity'] +
                consensus_score * self.weights['engine_consensus']
            )
        
        return RankingScore(
            total_score=total_score,
            relevance_score=relevance_score,
            freshness_score=freshness_score,
            authority_score=authority_score,
            diversity_score=diversity_score,
            conflict_score=conflict_score,
            engine_consensus_score=consensus_score,
            components={
                'relevance': relevance_score,
                'freshness': freshness_score,
                'authority': authority_score,
                'diversity': diversity_score,
                'consensus': consensus_score,
                'conflict': conflict_score
            }
        )
    
    def _calculate_relevance_score(self, result: Any, query: str) -> float:
        """Calculate relevance score based on title and snippet matching."""
        if not query:
            return getattr(result, 'relevance_score', 0.5)
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Get text content to analyze
        title = getattr(result, 'title', '').lower()
        snippet = getattr(result, 'snippet', '').lower()
        content = getattr(result, 'content', '')
        if content:
            content = content[:500].lower()  # First 500 chars
        
        all_text = f"{title} {snippet} {content}"
        text_terms = set(all_text.split())
        
        # Calculate term matching score
        matching_terms = query_terms.intersection(text_terms)
        term_match_score = len(matching_terms) / len(query_terms) if query_terms else 0
        
        # Boost for exact phrase matches
        phrase_boost = 0.0
        if query_lower in all_text:
            phrase_boost = 0.3
        
        # Position boost (title matches are more important)
        position_boost = 0.0
        if any(term in title for term in query_terms):
            position_boost = 0.2
        
        # Use existing relevance score if available
        base_score = getattr(result, 'relevance_score', 0.5)
        
        # Combine scores
        final_score = (
            base_score * 0.4 +
            term_match_score * 0.4 +
            phrase_boost +
            position_boost
        )
        
        return min(final_score, 1.0)
    
    def _calculate_freshness_score(self, result: Any) -> float:
        """Calculate freshness score based on publication/crawl date."""
        timestamp = getattr(result, 'timestamp', None)
        if not timestamp:
            # Check metadata for date
            metadata = getattr(result, 'metadata', {})
            if isinstance(metadata, dict):
                timestamp = metadata.get('publish_date') or metadata.get('date')
        
        if not timestamp:
            return 0.5  # Default score for unknown dates
        
        if isinstance(timestamp, str):
            # Try to parse string dates
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return 0.5
        
        # Calculate age in hours
        now = datetime.now()
        if timestamp.tzinfo and not now.tzinfo:
            now = now.replace(tzinfo=timestamp.tzinfo)
        elif not timestamp.tzinfo and now.tzinfo:
            timestamp = timestamp.replace(tzinfo=now.tzinfo)
        
        age_hours = (now - timestamp).total_seconds() / 3600
        
        # Exponential decay: newer is better
        if age_hours <= 1:
            return 1.0
        elif age_hours <= 24:
            return 0.9
        elif age_hours <= 168:  # 1 week
            return 0.7
        elif age_hours <= 720:  # 1 month
            return 0.5
        else:
            return 0.2
    
    def _calculate_authority_score(self, result: Any) -> float:
        """Calculate authority score based on domain reputation."""
        url = getattr(result, 'url', '')
        if not url:
            return 0.5
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check our authority database
            if domain in self.domain_authority:
                return self.domain_authority[domain]
            
            # Heuristic scoring for unknown domains
            if any(indicator in domain for indicator in ['.gov', '.edu', '.org']):
                return 0.8
            elif any(indicator in domain for indicator in ['news', 'times', 'post', 'journal']):
                return 0.7
            elif domain.endswith('.com'):
                return 0.6
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_diversity_score(self, result: Any, all_results: List[Any]) -> float:
        """Calculate diversity score to promote result variety."""
        url = getattr(result, 'url', '')
        if not url:
            return 0.5
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # Count how many results from same domain
            domain_count = 0
            for other_result in all_results:
                other_url = getattr(other_result, 'url', '')
                if other_url:
                    other_domain = urlparse(other_url).netloc.lower()
                    if domain == other_domain:
                        domain_count += 1
            
            # Penalize over-representation of single domain
            if domain_count == 1:
                return 1.0
            elif domain_count == 2:
                return 0.8
            elif domain_count == 3:
                return 0.6
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    def _calculate_engine_consensus_score(self, result: Any) -> float:
        """Calculate score based on how many engines found this result."""
        engines_found = getattr(result, 'engines_found', [])
        if not engines_found:
            return 0.5
        
        # More engines finding the same result = higher consensus
        num_engines = len(engines_found)
        if num_engines >= 4:
            return 1.0
        elif num_engines == 3:
            return 0.8
        elif num_engines == 2:
            return 0.6
        else:
            return 0.4
    
    def _calculate_conflict_score(self, result: Any) -> float:
        """Calculate conflict relevance score."""
        # Check if result has conflict score from conflict analysis
        if hasattr(result, 'conflict_score'):
            return getattr(result, 'conflict_score')
        
        # Fallback: analyze title and snippet for conflict keywords
        title = getattr(result, 'title', '').lower()
        snippet = getattr(result, 'snippet', '').lower()
        
        conflict_keywords = [
            'conflict', 'violence', 'attack', 'war', 'crisis', 'emergency',
            'security', 'threat', 'insurgency', 'terrorism', 'clash',
            'killed', 'death', 'casualties', 'bombing', 'explosion'
        ]
        
        text = f"{title} {snippet}"
        matches = sum(1 for keyword in conflict_keywords if keyword in text)
        
        return min(matches / 5.0, 1.0)  # Normalize to 0-1
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """Get ranking statistics."""
        stats = dict(self.ranking_stats)
        
        # Calculate averages
        for strategy, scores in self.ranking_stats['average_scores'].items():
            if scores:
                stats[f'avg_score_{strategy}'] = sum(scores) / len(scores)
        
        return stats
    
    def reset_stats(self):
        """Reset ranking statistics."""
        self.ranking_stats = {
            'total_rankings': 0,
            'strategy_usage': defaultdict(int),
            'average_scores': defaultdict(list)
        }


class ConflictRanker(ResultRanker):
    """Specialized ranker for conflict monitoring results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize conflict-focused ranker."""
        super().__init__(strategy='conflict_focused', config=config)
        
        # Conflict-specific weights
        self.weights = {
            'conflict_relevance': 0.4,
            'temporal_relevance': 0.25,
            'location_relevance': 0.2,
            'source_credibility': 0.15
        }
        
        # High-priority conflict indicators
        self.critical_indicators = [
            'breaking', 'urgent', 'alert', 'emergency', 'crisis',
            'killed', 'dead', 'casualties', 'attack', 'bombing',
            'explosion', 'violence', 'conflict', 'war'
        ]
        
        # Location boost for Horn of Africa
        self.priority_locations = [
            'somalia', 'ethiopia', 'eritrea', 'djibouti', 'sudan',
            'south sudan', 'horn of africa', 'mogadishu', 'addis ababa',
            'asmara', 'khartoum', 'juba'
        ]
    
    def _calculate_conflict_score(self, result: Any) -> float:
        """Enhanced conflict scoring for specialized monitoring."""
        base_score = super()._calculate_conflict_score(result)
        
        title = getattr(result, 'title', '').lower()
        snippet = getattr(result, 'snippet', '').lower()
        text = f"{title} {snippet}"
        
        # Boost for critical indicators
        critical_boost = 0.0
        for indicator in self.critical_indicators:
            if indicator in text:
                critical_boost += 0.1
        
        # Boost for priority locations
        location_boost = 0.0
        for location in self.priority_locations:
            if location in text:
                location_boost += 0.15
                break  # Only count once
        
        # Boost for recent temporal indicators
        temporal_boost = 0.0
        temporal_terms = ['today', 'yesterday', 'breaking', 'latest', 'current']
        if any(term in text for term in temporal_terms):
            temporal_boost = 0.1
        
        final_score = min(
            base_score + critical_boost + location_boost + temporal_boost,
            1.0
        )
        
        return final_score