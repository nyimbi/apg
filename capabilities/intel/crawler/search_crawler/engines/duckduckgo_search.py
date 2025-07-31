"""
DuckDuckGo Search Engine
========================

DuckDuckGo search implementation with privacy-focused approach.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote
from datetime import datetime

from .base_search_engine import BaseSearchEngine, SearchResult, SearchResponse

logger = logging.getLogger(__name__)


class DuckDuckGoSearchEngine(BaseSearchEngine):
    """DuckDuckGo search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # DuckDuckGo-specific configuration
        self.base_url = "https://duckduckgo.com/html/"
        self.min_delay = self.config.get('min_delay', 1.5)  # DDG is more lenient
        self.max_delay = self.config.get('max_delay', 4.0)
        
        # DuckDuckGo result selectors
        self.result_selectors = {
            'container': '.result',
            'title': '.result__title',
            'link': '.result__url',
            'snippet': '.result__snippet'
        }
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build DuckDuckGo search URL."""
        params = {
            'q': query,
            'kl': kwargs.get('region', 'us-en'),
            'kp': '-2',  # Safe search off
            'kz': '-1',  # No instant answers
            'kc': '1',   # Auto-load images
            'kf': '-1',  # No favicons
            'ka': 'en',  # Language
            'k1': '-1',  # No ads
        }
        
        # Add time filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': 'd',
                'week': 'w',
                'month': 'm',
                'year': 'y'
            }
            if kwargs['time_filter'] in time_filters:
                params['df'] = time_filters[kwargs['time_filter']]
        
        # Build URL
        param_string = '&'.join([f"{k}={quote(str(v))}" for k, v in params.items()])
        return f"{self.base_url}?{param_string}"
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        offset: int = 0,
        **kwargs
    ) -> SearchResponse:
        """Perform DuckDuckGo search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, **kwargs)
            self.logger.debug(f"DuckDuckGo search URL: {search_url}")
            
            # Make request
            html = await self._make_request(search_url)
            if not html:
                return SearchResponse(
                    query=query,
                    engine=self.name,
                    success=False,
                    error_message="Failed to fetch search results"
                )
            
            # Parse results
            results = await self._parse_results(html, query)
            
            # Limit results
            results = results[:max_results]
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(True, response_time, len(results))
            
            return SearchResponse(
                query=query,
                engine=self.name,
                results=results,
                total_results=len(results),
                search_time=response_time,
                success=True
            )
        
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed for query '{query}': {e}")
            response_time = time.time() - start_time
            self._update_stats(False, response_time, 0)
            
            return SearchResponse(
                query=query,
                engine=self.name,
                success=False,
                error_message=str(e),
                search_time=response_time
            )
    
    async def _parse_results(self, html: str, query: str) -> List[SearchResult]:
        """Parse DuckDuckGo search results from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error("BeautifulSoup not available for parsing")
            return []
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result containers
            result_containers = soup.find_all('div', class_='result')
            
            for i, container in enumerate(result_containers):
                try:
                    # Extract title
                    title_element = container.find('a', class_='result__a')
                    if not title_element:
                        continue
                    
                    title = title_element.get_text(strip=True)
                    url = title_element.get('href', '')
                    
                    if not url:
                        continue
                    
                    # Extract snippet
                    snippet = ""
                    snippet_element = container.find('a', class_='result__snippet')
                    if snippet_element:
                        snippet = snippet_element.get_text(strip=True)
                    
                    # Extract metadata
                    metadata = self._extract_ddg_metadata(container)
                    
                    # Create search result
                    result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        engine=self.name,
                        rank=i + 1,
                        metadata=metadata
                    )
                    
                    results.append(result)
                
                except Exception as e:
                    self.logger.debug(f"Error parsing DuckDuckGo result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} DuckDuckGo results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing DuckDuckGo results: {e}")
            return []
    
    def _extract_ddg_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from DuckDuckGo result."""
        metadata = {'source': 'duckduckgo'}
        
        try:
            # Look for displayed URL
            url_element = container.find('span', class_='result__url')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Look for date information
            extras = container.find('div', class_='result__extras')
            if extras:
                extras_text = extras.get_text(strip=True)
                if extras_text:
                    metadata['extra_info'] = extras_text
            
            # DuckDuckGo often includes source info
            icon = container.find('img', class_='result__icon__img')
            if icon and icon.get('alt'):
                metadata['source_name'] = icon.get('alt')
        
        except Exception as e:
            self.logger.debug(f"Error extracting DuckDuckGo metadata: {e}")
        
        return metadata
    
    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        time_filter: str = 'week'
    ) -> SearchResponse:
        """Search DuckDuckGo News."""
        # DuckDuckGo doesn't have a dedicated news search
        # We can add news-related terms to the query
        news_query = f"{query} news OR breaking OR latest"
        
        return await self.search(
            query=news_query,
            max_results=max_results,
            time_filter=time_filter
        )
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get DuckDuckGo-specific statistics."""
        base_stats = self.get_stats()
        
        ddg_stats = {
            **base_stats,
            'engine_features': {
                'privacy_focused': True,
                'no_tracking': True,
                'time_filters': True,
                'region_filters': True,
                'instant_answers': False  # Disabled in HTML mode
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'very_lenient': True
            }
        }
        
        return ddg_stats