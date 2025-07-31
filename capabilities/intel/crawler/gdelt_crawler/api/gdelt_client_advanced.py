"""
GDELT Client with Improved Error Handling and Extended Features
==================================================================

This version provides:
1. Better async generator handling
2. Convenience methods for common use cases
3. Improved error handling and retry logic
4. Content enhancement with full text extraction
5. Conflict-specific search capabilities
6. Regional focus for Horn of Africa

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 5.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from dataclasses import dataclass
import aiohttp
from contextlib import asynccontextmanager

from .gdelt_client import (
    GDELTClient, GDELTArticle, GDELTDateRange,
    GDELTMode, GDELTFormat, GDELTQueryParameters
)

logger = logging.getLogger(__name__)

@dataclass
class ConflictSearchParams:
    """Parameters for conflict-focused GDELT searches"""
    region: str
    countries: List[str]
    conflict_terms: List[str] = None
    date_range_days: int = 7
    max_articles_per_day: int = 50
    include_content: bool = True

class GDELTClientAdvanced(GDELTClient):
    """GDELT client with convenience methods and better error handling"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default conflict terms for Horn of Africa monitoring
        self.default_conflict_terms = [
            'conflict', 'violence', 'attack', 'terrorism', 'war', 'clash',
            'protest', 'demonstration', 'riot', 'unrest', 'security',
            'displacement', 'refugee', 'crisis', 'insurgency',
            'military operation', 'peacekeeping', 'armed group'
        ]
        
        # Country code mapping for Horn of Africa
        self.horn_of_africa_codes = {
            'Ethiopia': 'ET', 'Somalia': 'SO', 'Sudan': 'SD', 'South Sudan': 'SS',
            'Kenya': 'KE', 'Uganda': 'UG', 'Djibouti': 'DJ', 'Eritrea': 'ER',
            'Tanzania': 'TZ', 'Rwanda': 'RW'
        }

    async def search_recent_conflicts(
        self,
        countries: List[str],
        days_back: int = 7,
        max_articles_per_country: int = 20
    ) -> List[GDELTArticle]:
        """
        Convenience method to search for recent conflicts in specified countries
        
        Args:
            countries: List of country names
            days_back: Number of days to look back
            max_articles_per_country: Maximum articles per country
            
        Returns:
            List of GDELT articles
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        all_articles = []
        
        for country in countries:
            try:
                country_code = self.horn_of_africa_codes.get(country, country[:2].upper())
                
                # Create conflict-focused query
                conflict_terms = ' OR '.join(self.default_conflict_terms[:5])  # Limit to avoid query length
                query = f"({conflict_terms}) (sourcecountry:{country_code} OR {country})"
                
                # Get recent articles
                articles = await self.fetch_with_timespan(
                    query=query,
                    timespan=f"{days_back}d",
                    max_records=max_articles_per_country
                )
                
                all_articles.extend(articles)
                logger.info(f"Found {len(articles)} conflict articles for {country}")
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to search conflicts for {country}: {e}")
                continue
        
        return all_articles

    async def fetch_date_range_as_list(
        self,
        query: str,
        date_range: GDELTDateRange,
        mode: GDELTMode = GDELTMode.ARTLIST,
        format: GDELTFormat = GDELTFormat.CSV,
        max_records_per_day: int = 50
    ) -> List[GDELTArticle]:
        """
        Fetch articles for a date range and return as a simple list
        (non-generator version for easier usage)
        
        Args:
            query: GDELT query string
            date_range: Date range to search
            mode: GDELT query mode
            format: Output format
            max_records_per_day: Maximum records per day
            
        Returns:
            List of all articles found across the date range
        """
        all_articles = []
        
        try:
            async for current_date, daily_articles in self.fetch_date_range(
                query=query,
                date_range=date_range,
                mode=mode,
                format=format,
                max_records_per_day=max_records_per_day
            ):
                all_articles.extend(daily_articles)
                logger.debug(f"Collected {len(daily_articles)} articles for {current_date.date()}")
                
        except Exception as e:
            logger.error(f"Error in date range fetch: {e}")
            
        return all_articles

    async def monitor_horn_of_africa_conflicts(
        self,
        days_back: int = 3,
        max_articles_total: int = 100
    ) -> Dict[str, List[GDELTArticle]]:
        """
        Monitor conflicts across Horn of Africa countries
        
        Args:
            days_back: Number of days to look back
            max_articles_total: Maximum total articles to collect
            
        Returns:
            Dictionary mapping country names to their conflict articles
        """
        results = {}
        articles_per_country = max_articles_total // len(self.horn_of_africa_codes)
        
        for country, country_code in self.horn_of_africa_codes.items():
            try:
                # Create targeted conflict query
                query = f"(conflict OR violence OR attack OR crisis) sourcecountry:{country_code}"
                
                articles = await self.fetch_with_timespan(
                    query=query,
                    timespan=f"{days_back}d",
                    max_records=articles_per_country
                )
                
                results[country] = articles
                logger.info(f"Monitored {len(articles)} conflict events in {country}")
                
                # Rate limiting between countries
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to monitor {country}: {e}")
                results[country] = []
                
        return results

    async def get_trending_conflict_topics(
        self,
        region: str = "Horn of Africa",
        days_back: int = 7
    ) -> Dict[str, int]:
        """
        Get trending conflict-related topics in a region
        
        Args:
            region: Region to focus on
            days_back: Number of days to analyze
            
        Returns:
            Dictionary of topics and their frequency
        """
        # Collect recent articles
        countries = list(self.horn_of_africa_codes.keys())
        articles = await self.search_recent_conflicts(
            countries=countries,
            days_back=days_back,
            max_articles_per_country=10
        )
        
        # Simple topic frequency analysis
        topic_counts = {}
        conflict_keywords = [
            'election', 'protest', 'military', 'police', 'government',
            'opposition', 'terrorist', 'insurgent', 'civilian', 'peacekeeping'
        ]
        
        for article in articles:
            title_lower = (article.title or '').lower()
            content_lower = (article.content or '')[:500].lower()  # First 500 chars
            text = f"{title_lower} {content_lower}"
            
            for keyword in conflict_keywords:
                if keyword in text:
                    topic_counts[keyword] = topic_counts.get(keyword, 0) + 1
        
        # Sort by frequency
        sorted_topics = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
        return sorted_topics

    async def enhanced_article_search(
        self,
        search_params: ConflictSearchParams
    ) -> List[GDELTArticle]:
        """
        Enhanced article search with conflict parameters
        
        Args:
            search_params: ConflictSearchParams object with search criteria
            
        Returns:
            List of enhanced articles
        """
        # Build query from parameters
        conflict_terms = search_params.conflict_terms or self.default_conflict_terms[:3]
        countries_query = ' OR '.join([
            f"sourcecountry:{self.horn_of_africa_codes.get(country, country[:2].upper())}"
            for country in search_params.countries
        ])
        
        query = f"({' OR '.join(conflict_terms)}) ({countries_query})"
        
        # Fetch articles
        articles = await self.fetch_with_timespan(
            query=query,
            timespan=f"{search_params.date_range_days}d",
            max_records=search_params.max_articles_per_day * search_params.date_range_days
        )
        
        logger.info(f"Enhanced search found {len(articles)} articles for {search_params.region}")
        return articles

# Convenience functions for easy usage

async def quick_conflict_search(
    countries: List[str],
    days_back: int = 7,
    max_articles: int = 50
) -> List[GDELTArticle]:
    """
    Quick conflict search without managing client lifecycle
    
    Args:
        countries: List of country names
        days_back: Days to look back
        max_articles: Maximum articles to return
        
    Returns:
        List of conflict articles
    """
    async with GDELTClientAdvanced() as client:
        return await client.search_recent_conflicts(
            countries=countries,
            days_back=days_back,
            max_articles_per_country=max_articles // len(countries)
        )

async def monitor_horn_conflicts() -> Dict[str, List[GDELTArticle]]:
    """
    Quick Horn of Africa conflict monitoring
    
    Returns:
        Dictionary of country -> conflict articles
    """
    async with GDELTClientAdvanced() as client:
        return await client.monitor_horn_of_africa_conflicts()

# Aliases for backward compatibility and naming consistency
ExtendedGDELTClient = GDELTClientAdvanced
EnhancedGDELTClient = GDELTClientAdvanced

# Export client as the main GDELT client
__all__ = [
    'GDELTClientAdvanced',
    'EnhancedGDELTClient',  # backward compatibility
    'ExtendedGDELTClient',  # backward compatibility
    'ConflictSearchParams', 
    'quick_conflict_search',
    'monitor_horn_conflicts'
]