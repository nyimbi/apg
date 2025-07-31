#!/usr/bin/env python3
"""
Location-Specific Search Optimizer
=================================

This module provides location-aware search optimization for NewsData.io,
specifically targeting priority locations: Aweil, Karamoja, Mandera, Assosa.

It implements a hierarchical search strategy that starts with specific locations
and expands to broader geographic areas only when insufficient news is found.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class SearchLevel(Enum):
    """Search hierarchy levels."""
    SPECIFIC_LOCATION = "specific_location"
    DISTRICT = "district"
    REGION = "region"
    COUNTRY = "country"


@dataclass
class LocationHierarchy:
    """Hierarchical location data for targeted searching."""
    specific_location: str
    district: str
    region: str
    country: str
    alternative_names: List[str]
    
    def get_search_terms(self, level: SearchLevel) -> List[str]:
        """Get search terms for a specific hierarchy level."""
        if level == SearchLevel.SPECIFIC_LOCATION:
            return [self.specific_location] + self.alternative_names
        elif level == SearchLevel.DISTRICT:
            return [self.district]
        elif level == SearchLevel.REGION:
            return [self.region]
        elif level == SearchLevel.COUNTRY:
            return [self.country]
        return []


# Priority location definitions with hierarchical data
PRIORITY_LOCATIONS = {
    "aweil": LocationHierarchy(
        specific_location="Aweil",
        district="Aweil Center County",
        region="Northern Bahr el Ghazal",
        country="South Sudan",
        alternative_names=["Aweil Town", "Aweil Center"]
    ),
    "karamoja": LocationHierarchy(
        specific_location="Karamoja",
        district="Karamoja sub-region",
        region="Northern Uganda",
        country="Uganda",
        alternative_names=["Karamoja region", "Karamojong"]
    ),
    "mandera": LocationHierarchy(
        specific_location="Mandera",
        district="Mandera County",
        region="North Eastern Province", 
        country="Kenya",
        alternative_names=["Mandera Town", "Mandera County"]
    ),
    "assosa": LocationHierarchy(
        specific_location="Assosa",
        district="Assosa Zone",
        region="Benishangul-Gumuz",
        country="Ethiopia",
        alternative_names=["Asosa", "Assosa Town"]
    )
}

# Conflict-related keywords for enhanced targeting
CONFLICT_KEYWORDS = [
    # Violence and conflict
    "conflict", "violence", "fighting", "clash", "attack", "assault",
    "battle", "war", "armed", "militant", "insurgent", "terrorist",
    
    # Deaths and casualties
    "killed", "death", "died", "casualties", "wounded", "injured",
    "fatalities", "victims", "massacre", "murder", "assassination",
    
    # Raids and criminal activity
    "raid", "robbery", "theft", "bandit", "cattle rustling", "looting",
    "kidnapping", "abduction", "hostage", "ransom",
    
    # Displacement and humanitarian
    "displaced", "refugees", "fled", "escape", "evacuation", "humanitarian crisis",
    "food insecurity", "drought", "famine", "aid",
    
    # Security and military
    "security", "military", "police", "peacekeeping", "intervention",
    "operation", "crackdown", "arrest", "detention",
    
    # Ethnic and tribal
    "ethnic", "tribal", "communal", "sectarian", "intercommunal",
    "pastoralist", "farmer", "herder"
]

# Minimum article thresholds for each search level
MIN_ARTICLES_THRESHOLDS = {
    SearchLevel.SPECIFIC_LOCATION: 2,
    SearchLevel.DISTRICT: 3,
    SearchLevel.REGION: 5,
    SearchLevel.COUNTRY: 8
}


class LocationOptimizer:
    """
    Optimizer for location-specific news searches with hierarchical expansion.
    
    This class implements a smart search strategy that:
    1. Starts with specific priority locations
    2. Expands to district/county level if insufficient results
    3. Expands to regional/provincial level if still insufficient
    4. Finally searches at country level as last resort
    """

    def __init__(self, min_articles_per_location: int = 3):
        """
        Initialize the location optimizer.

        Args:
            min_articles_per_location: Minimum articles needed before considering search complete
        """
        self.min_articles_per_location = min_articles_per_location
        self.priority_locations = PRIORITY_LOCATIONS
        self.conflict_keywords = CONFLICT_KEYWORDS

    def generate_search_queries(self, location_key: str, conflict_focus: bool = True) -> List[Dict[str, Any]]:
        """
        Generate hierarchical search queries for a specific location.

        Args:
            location_key: Key for the priority location (e.g., 'aweil', 'karamoja')
            conflict_focus: Whether to include conflict-specific keywords

        Returns:
            List of search query configurations in priority order
        """
        if location_key.lower() not in self.priority_locations:
            raise ValueError(f"Unknown location: {location_key}. Available: {list(self.priority_locations.keys())}")

        location = self.priority_locations[location_key.lower()]
        queries = []

        # Generate queries for each hierarchy level
        for level in SearchLevel:
            location_terms = location.get_search_terms(level)
            
            for location_term in location_terms:
                if conflict_focus:
                    # Create conflict-focused queries
                    for keyword in self.conflict_keywords[:10]:  # Use top 10 conflict keywords
                        query = f'"{location_term}" {keyword}'
                        queries.append({
                            "query": query,
                            "level": level,
                            "location_term": location_term,
                            "conflict_keyword": keyword,
                            "priority": self._get_priority_score(level, keyword)
                        })
                else:
                    # General news query
                    query = f'"{location_term}"'
                    queries.append({
                        "query": query,
                        "level": level,
                        "location_term": location_term,
                        "conflict_keyword": None,
                        "priority": self._get_priority_score(level, None)
                    })

        # Sort by priority (highest first)
        queries.sort(key=lambda x: x["priority"], reverse=True)
        return queries

    def _get_priority_score(self, level: SearchLevel, conflict_keyword: Optional[str]) -> int:
        """
        Calculate priority score for search query ordering.

        Args:
            level: Search hierarchy level
            conflict_keyword: Conflict keyword used (if any)

        Returns:
            Priority score (higher = more important)
        """
        base_scores = {
            SearchLevel.SPECIFIC_LOCATION: 1000,
            SearchLevel.DISTRICT: 100,
            SearchLevel.REGION: 50,
            SearchLevel.COUNTRY: 10
        }

        score = base_scores.get(level, 1)

        # Boost for high-priority conflict keywords
        if conflict_keyword:
            high_priority_keywords = ["killed", "attack", "violence", "conflict", "clash", "raid"]
            if conflict_keyword in high_priority_keywords:
                score += 500
            else:
                score += 100

        return score

    def should_expand_search(self, results: List[Dict[str, Any]], current_level: SearchLevel) -> bool:
        """
        Determine if search should expand to the next hierarchy level.

        Args:
            results: Current search results
            current_level: Current search hierarchy level

        Returns:
            True if search should expand to broader geographic area
        """
        if not results:
            return True

        # Check if we have enough articles for this level
        min_threshold = MIN_ARTICLES_THRESHOLDS.get(current_level, self.min_articles_per_location)
        
        if len(results) >= min_threshold:
            logger.info(f"Sufficient articles found at {current_level.value} level: {len(results)} articles")
            return False

        logger.info(f"Insufficient articles at {current_level.value} level: {len(results)} < {min_threshold}, expanding search")
        return True

    def filter_relevant_articles(self, articles: List[Dict[str, Any]], location_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Filter articles for location and conflict relevance.

        Args:
            articles: List of articles to filter
            location_terms: Location terms to check for

        Returns:
            Filtered list of relevant articles
        """
        relevant_articles = []

        for article in articles:
            # Check title and description for location mentions
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            content = article.get("content", "").lower()
            
            # Combine all text for searching
            full_text = f"{title} {description} {content}"

            # Check for location relevance
            location_match = any(term.lower() in full_text for term in location_terms)
            
            # Check for conflict relevance
            conflict_match = any(keyword in full_text for keyword in self.conflict_keywords)

            if location_match and conflict_match:
                # Add relevance score
                article["relevance_score"] = self._calculate_relevance_score(full_text, location_terms)
                relevant_articles.append(article)

        # Sort by relevance score
        relevant_articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant_articles

    def _calculate_relevance_score(self, text: str, location_terms: List[str]) -> float:
        """
        Calculate relevance score for an article.

        Args:
            text: Article text to analyze
            location_terms: Location terms to check for

        Returns:
            Relevance score (higher = more relevant)
        """
        score = 0.0
        text_lower = text.lower()

        # Location term frequency
        for term in location_terms:
            score += text_lower.count(term.lower()) * 10

        # Conflict keyword frequency
        for keyword in self.conflict_keywords:
            if keyword in text_lower:
                score += 5

        # Boost for high-priority conflict terms in title
        high_priority = ["killed", "attack", "violence", "raid", "conflict"]
        for term in high_priority:
            if term in text_lower:
                score += 15

        return score

    def get_optimized_search_plan(self, locations: List[str] = None, max_credits: int = 5) -> List[Dict[str, Any]]:
        """
        Generate an optimized search plan for multiple locations within credit constraints.

        Args:
            locations: List of location keys to search (default: all priority locations)
            max_credits: Maximum credits to use for the search plan

        Returns:
            List of search steps optimized for credit usage
        """
        if locations is None:
            locations = list(self.priority_locations.keys())

        search_plan = []
        credits_per_location = max(1, max_credits // len(locations))

        for location_key in locations:
            location_queries = self.generate_search_queries(location_key, conflict_focus=True)
            
            # Select top queries within credit budget
            selected_queries = location_queries[:credits_per_location]
            
            search_plan.append({
                "location": location_key,
                "location_data": self.priority_locations[location_key],
                "queries": selected_queries,
                "estimated_credits": len(selected_queries),
                "search_strategy": "hierarchical_expansion"
            })

        return search_plan

    def format_search_summary(self, results: Dict[str, Any]) -> str:
        """
        Format search results summary for logging/reporting.

        Args:
            results: Search results by location

        Returns:
            Formatted summary string
        """
        summary_lines = ["=== Location-Optimized Search Summary ==="]
        
        total_articles = 0
        for location, data in results.items():
            articles = data.get("articles", [])
            level = data.get("final_level", "unknown")
            credits_used = data.get("credits_used", 0)
            
            total_articles += len(articles)
            summary_lines.append(f"\n{location.upper()}:")
            summary_lines.append(f"  Articles found: {len(articles)}")
            summary_lines.append(f"  Search level: {level}")
            summary_lines.append(f"  Credits used: {credits_used}")
            
            if articles:
                # Show top 2 articles
                for i, article in enumerate(articles[:2], 1):
                    title = article.get("title", "N/A")[:60]
                    score = article.get("relevance_score", 0)
                    summary_lines.append(f"    {i}. {title}... (score: {score:.1f})")

        summary_lines.append(f"\nTOTAL: {total_articles} articles across {len(results)} locations")
        return "\n".join(summary_lines)


# Convenience functions for quick access
def create_location_optimizer(min_articles: int = 3) -> LocationOptimizer:
    """Create a location optimizer with default settings."""
    return LocationOptimizer(min_articles_per_location=min_articles)


def get_priority_locations() -> Dict[str, LocationHierarchy]:
    """Get the priority location definitions."""
    return PRIORITY_LOCATIONS.copy()


def get_conflict_keywords() -> List[str]:
    """Get the conflict-related keywords."""
    return CONFLICT_KEYWORDS.copy()