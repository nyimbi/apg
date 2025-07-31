"""
Mojeek Search Engine
====================

Mojeek search implementation (independent crawler-based search engine).

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


class MojeekSearchEngine(BaseSearchEngine):
    """Mojeek search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Mojeek-specific configuration
        self.base_url = "https://www.mojeek.com/search"
        self.min_delay = self.config.get('min_delay', 1.0)
        self.max_delay = self.config.get('max_delay', 3.0)
        
        # Mojeek result selectors
        self.result_selectors = {
            'container': '.results-standard__result',
            'title': '.title a',
            'link': '.title a',
            'snippet': '.s'
        }
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build Mojeek search URL."""
        params = {
            'q': query,
            's': offset,
            'arc': 'en',
            'fmt': 'json' if kwargs.get('format') == 'json' else 'html'
        }
        
        # Add safe search
        if kwargs.get('safe_search', False):
            params['safe'] = '1'
        
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
        """Perform Mojeek search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, **kwargs)
            self.logger.debug(f"Mojeek search URL: {search_url}")
            
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
            self.logger.error(f"Mojeek search failed for query '{query}': {e}")
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
        """Parse Mojeek search results from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error("BeautifulSoup not available for parsing")
            return []
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result containers
            result_containers = soup.find_all('div', class_='results-standard__result')
            if not result_containers:
                # Fallback selectors
                result_containers = soup.find_all('li', class_='result')
            
            for i, container in enumerate(result_containers):
                try:
                    # Extract title and link
                    title_element = container.find(class_='title')
                    if not title_element:
                        title_element = container.find('h2')
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
                    snippet_element = container.find(class_='s')
                    if snippet_element:
                        snippet = snippet_element.get_text(strip=True)
                    
                    # Extract metadata
                    metadata = self._extract_mojeek_metadata(container)
                    
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
                    self.logger.debug(f"Error parsing Mojeek result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} Mojeek results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing Mojeek results: {e}")
            return []
    
    def _extract_mojeek_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from Mojeek result."""
        metadata = {'source': 'mojeek', 'independent_crawler': True}
        
        try:
            # Look for displayed URL
            url_element = container.find(class_='url')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Mojeek sometimes shows additional info
            info_element = container.find(class_='info')
            if info_element:
                metadata['additional_info'] = info_element.get_text(strip=True)
        
        except Exception as e:
            self.logger.debug(f"Error extracting Mojeek metadata: {e}")
        
        return metadata
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get Mojeek-specific statistics."""
        base_stats = self.get_stats()
        
        mojeek_stats = {
            **base_stats,
            'engine_features': {
                'independent_crawler': True,
                'privacy_focused': True,
                'no_tracking': True,
                'uk_based': True,
                'small_but_growing': True
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'very_lenient': True
            }
        }
        
        return mojeek_stats