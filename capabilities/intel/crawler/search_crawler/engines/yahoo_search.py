"""
Yahoo Search Engine
===================

Yahoo search implementation (powered by Bing but with different results).

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


class YahooSearchEngine(BaseSearchEngine):
    """Yahoo search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Yahoo-specific configuration
        self.base_url = "https://search.yahoo.com/search"
        self.min_delay = self.config.get('min_delay', 2.0)
        self.max_delay = self.config.get('max_delay', 5.0)
        
        # Yahoo result selectors
        self.result_selectors = {
            'container': '.dd.algo',
            'title': '.compTitle h3 a',
            'link': '.compTitle h3 a',
            'snippet': '.compText'
        }
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build Yahoo search URL."""
        params = {
            'p': query,
            'b': offset + 1,  # Yahoo uses 1-based indexing
            'pz': kwargs.get('pz', 10),  # Results per page
            'ei': 'UTF-8',
            'fr': 'yfp-t',
            'fp': '1'
        }
        
        # Add time filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': 'd',
                'week': 'w',
                'month': 'm'
            }
            if kwargs['time_filter'] in time_filters:
                params['age'] = time_filters[kwargs['time_filter']]
        
        # Add region/language
        if 'region' in kwargs:
            params['vs'] = kwargs['region']
        
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
        """Perform Yahoo search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, pz=max_results, **kwargs)
            self.logger.debug(f"Yahoo search URL: {search_url}")
            
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
            self.logger.error(f"Yahoo search failed for query '{query}': {e}")
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
        """Parse Yahoo search results from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error("BeautifulSoup not available for parsing")
            return []
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result containers - Yahoo has multiple possible selectors
            result_containers = soup.find_all('div', class_=re.compile(r'dd.*algo'))
            if not result_containers:
                result_containers = soup.find_all('div', class_='Sr')
            
            for i, container in enumerate(result_containers):
                try:
                    # Extract title and link
                    title_element = container.find('h3')
                    if not title_element:
                        title_element = container.find('h4')
                    if not title_element:
                        continue
                    
                    link_element = title_element.find('a')
                    if not link_element:
                        continue
                    
                    title = link_element.get_text(strip=True)
                    url = link_element.get('href', '')
                    
                    if not url:
                        continue
                    
                    # Clean Yahoo redirect URLs
                    if 'RU=' in url:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(url.split('RU=')[1])
                        if parsed:
                            url = list(parsed.keys())[0]
                    
                    # Extract snippet
                    snippet = ""
                    snippet_selectors = [
                        '.compText',
                        '.s-desc',
                        '.ac-21th',
                        'p'
                    ]
                    
                    for selector in snippet_selectors:
                        snippet_element = container.select_one(selector)
                        if snippet_element:
                            snippet = snippet_element.get_text(strip=True)
                            break
                    
                    # Extract metadata
                    metadata = self._extract_yahoo_metadata(container)
                    
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
                    self.logger.debug(f"Error parsing Yahoo result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} Yahoo results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing Yahoo results: {e}")
            return []
    
    def _extract_yahoo_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from Yahoo result."""
        metadata = {'source': 'yahoo'}
        
        try:
            # Look for displayed URL
            url_element = container.find(class_='tc')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Look for date information
            date_element = container.find(class_='fc-2nd')
            if date_element:
                metadata['date_info'] = date_element.get_text(strip=True)
            
            # Yahoo often shows related searches
            related = container.find_all(class_='compDlink')
            if related:
                metadata['related_links'] = [link.get_text(strip=True) for link in related[:3]]
        
        except Exception as e:
            self.logger.debug(f"Error extracting Yahoo metadata: {e}")
        
        return metadata
    
    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        time_filter: str = 'week'
    ) -> SearchResponse:
        """Search Yahoo News specifically."""
        # Yahoo has a dedicated news search
        news_base_url = "https://news.search.yahoo.com/search"
        original_url = self.base_url
        self.base_url = news_base_url
        
        try:
            response = await self.search(
                query=query,
                max_results=max_results,
                time_filter=time_filter
            )
            return response
        finally:
            self.base_url = original_url
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get Yahoo-specific statistics."""
        base_stats = self.get_stats()
        
        yahoo_stats = {
            **base_stats,
            'engine_features': {
                'news_search': True,
                'powered_by_bing': True,
                'time_filters': True,
                'region_filters': True,
                'different_from_bing': True
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'moderate_restrictions': True
            }
        }
        
        return yahoo_stats