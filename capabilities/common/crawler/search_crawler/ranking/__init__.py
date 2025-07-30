"""
Result Ranking Package
======================

Advanced ranking algorithms for search results.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

from .result_ranker import ResultRanker, RankingStrategy, ConflictRanker

__all__ = [
    'ResultRanker',
    'RankingStrategy', 
    'ConflictRanker'
]