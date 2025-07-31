"""
Brave Search Engine
===================

Brave search implementation (independent search index).

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


class BraveSearchEngine(BaseSearchEngine):
    """Brave search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Brave-specific configuration
        self.base_url = "https://search.brave.com/search"
        self.min_delay = self.config.get('min_delay', 1.5)
        self.max_delay = self.config.get('max_delay', 4.0)
        
        # Brave result selectors
        self.result_selectors = {
            'container': '.snippet',
            'title': '.title',
            'link': '.title a',
            'snippet': '.snippet-description'
        }
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build Brave search URL."""
        params = {
            'q': query,
            'offset': offset,
            'spellcheck': '1',
            'source': 'web'
        }
        
        # Add time filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': 'pd',
                'week': 'pw',
                'month': 'pm',
                'year': 'py'
            }
            if kwargs['time_filter'] in time_filters:
                params['tf'] = time_filters[kwargs['time_filter']]
        
        # Add safe search
        if kwargs.get('safe_search', False):
            params['safesearch'] = 'strict'
        else:
            params['safesearch'] = 'off'
        
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
        """Perform Brave search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, **kwargs)
            self.logger.debug(f"Brave search URL: {search_url}")
            
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
            self.logger.error(f"Brave search failed for query '{query}': {e}")
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
        """Parse Brave search results from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error("BeautifulSoup not available for parsing")
            return []
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result containers
            result_containers = soup.find_all('div', class_='snippet')
            if not result_containers:
                # Fallback selectors
                result_containers = soup.find_all('div', attrs={'data-type': 'web'})
            
            for i, container in enumerate(result_containers):
                try:
                    # Skip ads and promoted results
                    if container.find(class_='ad-label') or 'ad' in container.get('class', []):
                        continue
                    
                    # Extract title and link
                    title_element = container.find(class_='title')
                    if not title_element:
                        title_element = container.find('h3')
                    if not title_element:
                        continue
                    
                    link_element = title_element.find('a')
                    if not link_element:
                        continue
                    
                    title = link_element.get_text(strip=True)
                    url = link_element.get('href', '')
                    
                    if not url:
                        continue
                    
                    # Extract snippet
                    snippet = ""
                    snippet_element = container.find(class_='snippet-description')
                    if snippet_element:
                        snippet = snippet_element.get_text(strip=True)
                    
                    # Extract metadata
                    metadata = self._extract_brave_metadata(container)
                    
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
                    self.logger.debug(f"Error parsing Brave result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} Brave results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing Brave results: {e}")
            return []
    
    def _extract_brave_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from Brave result."""
        metadata = {'source': 'brave', 'independent_index': True}
        
        try:
            # Look for displayed URL
            url_element = container.find(class_='url')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Look for date information
            date_element = container.find(class_='snippet-date')
            if date_element:
                metadata['date'] = date_element.get_text(strip=True)
            
            # Brave often shows favicons
            favicon = container.find(class_='favicon')
            if favicon:
                metadata['has_favicon'] = True
            
            # Look for featured snippets or special results
            if container.find(class_='featured-snippet'):
                metadata['result_type'] = 'featured_snippet'
        
        except Exception as e:
            self.logger.debug(f"Error extracting Brave metadata: {e}")
        
        return metadata
    
    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        time_filter: str = 'week'
    ) -> SearchResponse:
        """Search Brave News."""
        # Brave has a dedicated news search
        news_base_url = "https://search.brave.com/news"
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
        """Get Brave-specific statistics."""
        base_stats = self.get_stats()
        
        brave_stats = {
            **base_stats,
            'engine_features': {
                'independent_index': True,
                'privacy_focused': True,
                'no_tracking': True,
                'ad_blocker_company': True,
                'news_search': True,
                'time_filters': True
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'reasonable_restrictions': True
            }
        }
        
        return brave_stats