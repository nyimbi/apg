"""
Collector-Friendly GDELT Client
===============================

A wrapper around the GDELT client that provides collector-friendly methods
with proper async handling and simplified interfaces.

Author: Nyimbi Odero  
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .gdelt_client_advanced import GDELTClientAdvanced, ConflictSearchParams
from .gdelt_client import GDELTArticle, GDELTDateRange, GDELTMode, GDELTFormat

logger = logging.getLogger(__name__)

class CollectorFriendlyGDELTClient:
    """
    GDELT client wrapper optimized for use in data collectors
    """
    
    def __init__(self, **kwargs):
        self.client_kwargs = kwargs
        self._client: Optional[GDELTClientAdvanced] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self._client = GDELTClientAdvanced(**self.client_kwargs)
        await self._client.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            
    async def collect_country_conflicts(
        self,
        country: str,
        days_back: int = 3,
        max_articles: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Collect conflict articles for a specific country in collector-friendly format
        
        Args:
            country: Country name (e.g., "Ethiopia")
            days_back: Number of days to look back
            max_articles: Maximum articles to collect
            
        Returns:
            List of article dictionaries ready for collector processing
        """
        if not self._client:
            raise RuntimeError("Client not initialized - use async context manager")
            
        try:
            # Get articles using the enhanced client
            articles = await self._client.search_recent_conflicts(
                countries=[country],
                days_back=days_back,
                max_articles_per_country=max_articles
            )
            
            # Convert to collector-friendly format
            collector_articles = []
            for article in articles:
                article_dict = {
                    'url': article.url or '',
                    'title': article.title or f'GDELT Article - {country}',
                    'content': article.content or '',
                    'published_at': article.seendate,
                    'language': 'en',
                    'author': 'GDELT Project',
                    'source': 'GDELT',
                    'domain': article.domain or '',
                    'metadata': {
                        'country': country,
                        'gdelt_id': getattr(article, 'id', None),
                        'source_country': getattr(article, 'sourcecountry', None),
                        'social_image_url': getattr(article, 'socialimage', None),
                        'language_code': getattr(article, 'language', 'en')
                    }
                }
                collector_articles.append(article_dict)
                
            logger.info(f"Collected {len(collector_articles)} articles for {country}")
            return collector_articles
            
        except Exception as e:
            logger.error(f"Failed to collect GDELT articles for {country}: {e}")
            return []
    
    async def collect_regional_conflicts(
        self,
        countries: List[str],
        days_back: int = 3,
        max_articles_per_country: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Collect conflict articles for multiple countries
        
        Args:
            countries: List of country names
            days_back: Number of days to look back  
            max_articles_per_country: Maximum articles per country
            
        Returns:
            List of article dictionaries from all countries
        """
        if not self._client:
            raise RuntimeError("Client not initialized - use async context manager")
            
        all_articles = []
        
        for country in countries:
            try:
                country_articles = await self.collect_country_conflicts(
                    country=country,
                    days_back=days_back,
                    max_articles=max_articles_per_country
                )
                all_articles.extend(country_articles)
                
                # Rate limiting between countries
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Failed to collect articles for {country}: {e}")
                continue
                
        logger.info(f"Total articles collected from {len(countries)} countries: {len(all_articles)}")
        return all_articles
    
    async def collect_date_range_articles(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_articles_per_day: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Collect articles for a specific date range using proper async iteration
        
        Args:
            query: GDELT query string
            start_date: Start date for search
            end_date: End date for search  
            max_articles_per_day: Maximum articles per day
            
        Returns:
            List of article dictionaries
        """
        if not self._client:
            raise RuntimeError("Client not initialized - use async context manager")
            
        try:
            # Create date range
            date_range = GDELTDateRange(start_date, end_date)
            
            # Use the list version for easier handling
            articles = await self._client.fetch_date_range_as_list(
                query=query,
                date_range=date_range,
                mode=GDELTMode.ARTLIST,
                format=GDELTFormat.CSV,
                max_records_per_day=max_articles_per_day
            )
            
            # Convert to collector format
            collector_articles = []
            for article in articles:
                article_dict = {
                    'url': article.url or '',
                    'title': article.title or 'GDELT Article',
                    'content': article.content or '',
                    'published_at': article.seendate,
                    'language': 'en',
                    'author': 'GDELT Project', 
                    'source': 'GDELT',
                    'domain': article.domain or '',
                    'metadata': {
                        'gdelt_query': query,
                        'gdelt_id': getattr(article, 'id', None),
                        'source_country': getattr(article, 'sourcecountry', None),
                        'social_image_url': getattr(article, 'socialimage', None),
                        'language_code': getattr(article, 'language', 'en')
                    }
                }
                collector_articles.append(article_dict)
                
            logger.info(f"Collected {len(collector_articles)} articles for date range {start_date.date()} to {end_date.date()}")
            return collector_articles
            
        except Exception as e:
            logger.error(f"Failed to collect articles for date range: {e}")
            return []
    
    async def collect_horn_of_africa_summary(
        self,
        days_back: int = 2,
        max_total_articles: int = 50
    ) -> Dict[str, Any]:
        """
        Collect a summary of Horn of Africa conflicts
        
        Args:
            days_back: Number of days to look back
            max_total_articles: Maximum total articles to collect
            
        Returns:
            Dictionary with articles and summary statistics
        """
        if not self._client:
            raise RuntimeError("Client not initialized - use async context manager")
            
        try:
            # Monitor Horn of Africa
            results = await self._client.monitor_horn_of_africa_conflicts(
                days_back=days_back,
                max_articles_total=max_total_articles
            )
            
            # Convert to collector format and create summary
            all_articles = []
            country_counts = {}
            
            for country, articles in results.items():
                country_count = 0
                for article in articles:
                    article_dict = {
                        'url': article.url or '',
                        'title': article.title or f'GDELT Article - {country}',
                        'content': article.content or '',
                        'published_at': article.seendate,
                        'language': 'en',
                        'author': 'GDELT Project',
                        'source': 'GDELT',
                        'domain': article.domain or '',
                        'metadata': {
                            'country': country,
                            'region': 'Horn of Africa',
                            'gdelt_id': getattr(article, 'id', None),
                            'source_country': getattr(article, 'sourcecountry', None)
                        }
                    }
                    all_articles.append(article_dict)
                    country_count += 1
                    
                country_counts[country] = country_count
            
            summary = {
                'articles': all_articles,
                'total_count': len(all_articles),
                'country_breakdown': country_counts,
                'date_range': f"{days_back} days back",
                'collection_timestamp': datetime.now(timezone.utc)
            }
            
            logger.info(f"Horn of Africa summary: {len(all_articles)} total articles across {len(country_counts)} countries")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to collect Horn of Africa summary: {e}")
            return {
                'articles': [],
                'total_count': 0,
                'country_breakdown': {},
                'error': str(e),
                'collection_timestamp': datetime.now(timezone.utc)
            }

# Convenience functions for direct usage

async def quick_country_collection(country: str, days_back: int = 3) -> List[Dict[str, Any]]:
    """Quick collection for a single country"""
    async with CollectorFriendlyGDELTClient() as client:
        return await client.collect_country_conflicts(country, days_back)

async def quick_regional_collection(countries: List[str], days_back: int = 3) -> List[Dict[str, Any]]:
    """Quick collection for multiple countries"""
    async with CollectorFriendlyGDELTClient() as client:
        return await client.collect_regional_conflicts(countries, days_back)

# Make this the primary client for collectors
__all__ = [
    'CollectorFriendlyGDELTClient',
    'quick_country_collection',
    'quick_regional_collection'
]