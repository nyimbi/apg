"""
Baidu Search Engine
===================

Baidu search implementation for Chinese content coverage.

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


class BaiduSearchEngine(BaseSearchEngine):
    """Baidu search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Baidu-specific configuration
        self.base_url = "https://www.baidu.com/s"
        self.min_delay = self.config.get('min_delay', 3.0)
        self.max_delay = self.config.get('max_delay', 8.0)
        
        # Baidu result selectors
        self.result_selectors = {
            'container': '.result',
            'title': '.t a',
            'link': '.t a',
            'snippet': '.c-abstract'
        }
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build Baidu search URL."""
        params = {
            'wd': query,
            'pn': offset,
            'rn': kwargs.get('rn', 10),  # Results per page
            'ie': 'utf-8',
            'rsv_bp': '1',
            'rsv_idx': '1'
        }
        
        # Add time filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': '1',
                'week': '7',
                'month': '30',
                'year': '365'
            }
            if kwargs['time_filter'] in time_filters:
                params['gpc'] = f"stf={time_filters[kwargs['time_filter']]}"
        
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
        """Perform Baidu search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, rn=max_results, **kwargs)
            self.logger.debug(f"Baidu search URL: {search_url}")
            
            # Make request with Chinese headers
            html = await self._make_request_with_chinese_headers(search_url)
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
            self.logger.error(f"Baidu search failed for query '{query}': {e}")
            response_time = time.time() - start_time
            self._update_stats(False, response_time, 0)
            
            return SearchResponse(
                query=query,
                engine=self.name,
                success=False,
                error_message=str(e),
                search_time=response_time
            )
    
    async def _make_request_with_chinese_headers(self, url: str) -> Optional[str]:
        """Make request with Chinese-specific headers."""
        try:
            import aiohttp
            import random
        except ImportError:
            self.logger.error("aiohttp not available")
            return None
        
        # Rate limiting
        await self._rate_limit()
        
        # Chinese-specific headers
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Charset': 'utf-8',
            'Connection': 'keep-alive'
        }
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=headers
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text(encoding='utf-8')
                    else:
                        self.logger.warning(f"Baidu returned HTTP {response.status}")
                        return None
        
        except Exception as e:
            self.logger.error(f"Baidu request failed: {e}")
            return None
    
    async def _parse_results(self, html: str, query: str) -> List[SearchResult]:
        """Parse Baidu search results from HTML."""
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
                    # Skip Baidu's own services and ads
                    if container.find(class_='c-showurl') and 'baidu.com' in container.find(class_='c-showurl').get_text():
                        continue
                    
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
                    snippet_element = container.find(class_='c-abstract')
                    if snippet_element:
                        snippet = snippet_element.get_text(strip=True)
                    
                    # Extract metadata
                    metadata = self._extract_baidu_metadata(container)
                    
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
                    self.logger.debug(f"Error parsing Baidu result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} Baidu results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing Baidu results: {e}")
            return []
    
    def _extract_baidu_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from Baidu result."""
        metadata = {'source': 'baidu'}
        
        try:
            # Look for displayed URL
            url_element = container.find(class_='c-showurl')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Look for date information
            date_element = container.find(class_='c-author')
            if date_element:
                metadata['author_info'] = date_element.get_text(strip=True)
            
            # Look for Baidu's internal links
            if container.find(class_='c-cache'):
                metadata['has_cache'] = True
        
        except Exception as e:
            self.logger.debug(f"Error extracting Baidu metadata: {e}")
        
        return metadata
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get Baidu-specific statistics."""
        base_stats = self.get_stats()
        
        baidu_stats = {
            **base_stats,
            'engine_features': {
                'chinese_content': True,
                'china_focused': True,
                'time_filters': True,
                'good_for_asia': True
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'strict_restrictions': True
            }
        }
        
        return baidu_stats