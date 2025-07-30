"""
Bing Search Engine
==================

Microsoft Bing search implementation with result parsing.

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


class BingSearchEngine(BaseSearchEngine):
    """Microsoft Bing search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Bing-specific configuration
        self.base_url = "https://www.bing.com/search"
        self.min_delay = self.config.get('min_delay', 2.0)
        self.max_delay = self.config.get('max_delay', 5.0)
        
        # Bing result selectors
        self.result_selectors = {
            'container': '.b_algo',
            'title': 'h2 a',
            'link': 'h2 a',
            'snippet': '.b_caption p, .b_snippetBigText'
        }
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build Bing search URL."""
        params = {
            'q': query,
            'first': offset + 1,  # Bing uses 1-based indexing
            'count': kwargs.get('count', 10),
            'setlang': kwargs.get('language', 'en'),
            'cc': kwargs.get('country', 'us')
        }
        
        # Add freshness filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': 'Day',
                'week': 'Week',
                'month': 'Month'
            }
            if kwargs['time_filter'] in time_filters:
                params['filters'] = f"ex1:{time_filters[kwargs['time_filter']]}"
        
        # Add news filter
        if kwargs.get('news_only', False):
            params['scope'] = 'news'
        
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
        """Perform Bing search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, count=max_results, **kwargs)
            self.logger.debug(f"Bing search URL: {search_url}")
            
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
            self.logger.error(f"Bing search failed for query '{query}': {e}")
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
        """Parse Bing search results from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error("BeautifulSoup not available for parsing")
            return []
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result containers
            result_containers = soup.find_all('li', class_='b_algo')
            
            for i, container in enumerate(result_containers):
                try:
                    # Extract title and link
                    title_link = container.find('h2')
                    if not title_link:
                        continue
                    
                    link_element = title_link.find('a')
                    if not link_element:
                        continue
                    
                    title = link_element.get_text(strip=True)
                    url = link_element.get('href', '')
                    
                    if not url:
                        continue
                    
                    # Extract snippet
                    snippet = ""
                    snippet_selectors = [
                        '.b_caption p',
                        '.b_snippetBigText',
                        '.b_caption',
                        '.b_algoSlug'
                    ]
                    
                    for selector in snippet_selectors:
                        snippet_element = container.select_one(selector)
                        if snippet_element:
                            snippet = snippet_element.get_text(strip=True)
                            break
                    
                    # Extract metadata
                    metadata = self._extract_bing_metadata(container)
                    
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
                    self.logger.debug(f"Error parsing Bing result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} Bing results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing Bing results: {e}")
            return []
    
    def _extract_bing_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from Bing result."""
        metadata = {'source': 'bing'}
        
        try:
            # Look for displayed URL
            url_element = container.find('cite')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Look for date information
            date_element = container.find(class_='b_factrow')
            if date_element:
                date_text = date_element.get_text(strip=True)
                if date_text:
                    metadata['date_info'] = date_text
            
            # Look for special result types
            if container.find(class_='b_news'):
                metadata['result_type'] = 'news'
            elif container.find(class_='b_image'):
                metadata['result_type'] = 'image'
            elif container.find(class_='b_video'):
                metadata['result_type'] = 'video'
            
            # Look for site links or additional info
            sitelinks = container.find_all(class_='b_sitelink')
            if sitelinks:
                metadata['sitelinks'] = [link.get_text(strip=True) for link in sitelinks[:3]]
        
        except Exception as e:
            self.logger.debug(f"Error extracting Bing metadata: {e}")
        
        return metadata
    
    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        time_filter: str = 'week'
    ) -> SearchResponse:
        """Search Bing News specifically."""
        return await self.search(
            query=query,
            max_results=max_results,
            news_only=True,
            time_filter=time_filter
        )
    
    async def search_images(
        self,
        query: str,
        max_results: int = 10
    ) -> SearchResponse:
        """Search Bing Images (basic implementation)."""
        # For images, we modify the URL to point to images
        base_url_backup = self.base_url
        self.base_url = "https://www.bing.com/images/search"
        
        try:
            response = await self.search(query, max_results)
            return response
        finally:
            self.base_url = base_url_backup
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get Bing-specific statistics."""
        base_stats = self.get_stats()
        
        bing_stats = {
            **base_stats,
            'engine_features': {
                'news_search': True,
                'image_search': True,
                'video_search': True,
                'time_filters': True,
                'location_filters': True
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'friendlier_than_google': True
            }
        }
        
        return bing_stats