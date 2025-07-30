"""
SearX Search Engine
===================

SearX meta-search engine implementation (aggregates results from multiple engines).

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


class SearXSearchEngine(BaseSearchEngine):
    """SearX meta-search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # SearX-specific configuration
        # Using public instance - in production you'd want your own instance
        self.base_url = self.config.get('searx_instance', "https://searx.space/search")
        self.min_delay = self.config.get('min_delay', 1.0)
        self.max_delay = self.config.get('max_delay', 3.0)
        
        # SearX result selectors
        self.result_selectors = {
            'container': '.result',
            'title': '.result_header a',
            'link': '.result_header a',
            'snippet': '.result-content'
        }
        
        # Available SearX instances (fallbacks)
        self.instances = [
            "https://searx.space/search",
            "https://search.disroot.org/search",
            "https://searx.tiekoetter.com/search"
        ]
        self.current_instance_index = 0
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build SearX search URL."""
        params = {
            'q': query,
            'format': 'html',
            'language': kwargs.get('language', 'en'),
            'category_general': '1'
        }
        
        # Add time filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': 'day',
                'week': 'week',
                'month': 'month',
                'year': 'year'
            }
            if kwargs['time_filter'] in time_filters:
                params['time_range'] = time_filters[kwargs['time_filter']]
        
        # Enable specific engines if requested
        if 'engines' in kwargs:
            for engine in kwargs['engines']:
                params[f'category_{engine}'] = '1'
        
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
        """Perform SearX search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, **kwargs)
            self.logger.debug(f"SearX search URL: {search_url}")
            
            # Make request with fallback to other instances
            html = await self._make_request_with_fallback(search_url)
            if not html:
                return SearchResponse(
                    query=query,
                    engine=self.name,
                    success=False,
                    error_message="Failed to fetch search results from all SearX instances"
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
            self.logger.error(f"SearX search failed for query '{query}': {e}")
            response_time = time.time() - start_time
            self._update_stats(False, response_time, 0)
            
            return SearchResponse(
                query=query,
                engine=self.name,
                success=False,
                error_message=str(e),
                search_time=response_time
            )
    
    async def _make_request_with_fallback(self, url: str) -> Optional[str]:
        """Make request with fallback to other SearX instances."""
        # Try current instance first
        html = await self._make_request(url)
        if html:
            return html
        
        # Try other instances
        for i, instance in enumerate(self.instances):
            if i == self.current_instance_index:
                continue  # Skip current instance
            
            try:
                fallback_url = url.replace(self.base_url, instance)
                html = await self._make_request(fallback_url)
                if html:
                    # Switch to working instance
                    self.base_url = instance
                    self.current_instance_index = i
                    self.logger.info(f"Switched to SearX instance: {instance}")
                    return html
            except Exception as e:
                self.logger.debug(f"SearX instance {instance} failed: {e}")
                continue
        
        return None
    
    async def _parse_results(self, html: str, query: str) -> List[SearchResult]:
        """Parse SearX search results from HTML."""
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
                    # Extract title and link
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
                    snippet_element = container.find(class_='content')
                    if snippet_element:
                        snippet = snippet_element.get_text(strip=True)
                    
                    # Extract metadata
                    metadata = self._extract_searx_metadata(container)
                    
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
                    self.logger.debug(f"Error parsing SearX result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} SearX results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing SearX results: {e}")
            return []
    
    def _extract_searx_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from SearX result."""
        metadata = {'source': 'searx', 'meta_search': True}
        
        try:
            # Look for engine tags (which engines contributed this result)
            engine_tags = container.find_all(class_='engines')
            if engine_tags:
                engines = []
                for tag in engine_tags:
                    engines.extend([span.get_text(strip=True) for span in tag.find_all('span')])
                metadata['contributing_engines'] = engines
            
            # Look for URL display
            url_element = container.find(class_='url')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Look for publishdate
            date_element = container.find(class_='publishdate')
            if date_element:
                metadata['publish_date'] = date_element.get_text(strip=True)
        
        except Exception as e:
            self.logger.debug(f"Error extracting SearX metadata: {e}")
        
        return metadata
    
    async def search_with_specific_engines(
        self,
        query: str,
        engines: List[str],
        max_results: int = 10
    ) -> SearchResponse:
        """Search using specific engines in SearX."""
        return await self.search(
            query=query,
            max_results=max_results,
            engines=engines
        )
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get SearX-specific statistics."""
        base_stats = self.get_stats()
        
        searx_stats = {
            **base_stats,
            'engine_features': {
                'meta_search': True,
                'aggregates_multiple_engines': True,
                'open_source': True,
                'privacy_focused': True,
                'highly_configurable': True
            },
            'current_instance': self.base_url,
            'available_instances': len(self.instances),
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'very_lenient': True
            }
        }
        
        return searx_stats