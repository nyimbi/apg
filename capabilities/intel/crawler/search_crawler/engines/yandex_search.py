"""
Yandex Search Engine
====================

Yandex search implementation for broader international coverage.

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


class YandexSearchEngine(BaseSearchEngine):
    """Yandex search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Yandex-specific configuration - use English version
        self.base_url = "https://yandex.com/search/"
        self.min_delay = self.config.get('min_delay', 3.0)  # Longer delay for Yandex
        self.max_delay = self.config.get('max_delay', 8.0)
        
        # Updated Yandex result selectors based on current HTML structure
        self.result_selectors = {
            'container': 'li.serp-item',
            'title': 'a[href]',
            'link': 'a[href]',
            'snippet': '.organic__text, .text'
        }
        
        # Add Yandex-specific headers - simpler approach
        self.yandex_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Session for maintaining cookies
        self._session = None
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build Yandex search URL."""
        # Use US region to avoid blocking
        params = {
            'text': query,
            'p': offset // 10,  # Yandex uses page numbers
            'lr': kwargs.get('region_id', 84),  # 84 = US region
        }
        
        # Only add language if explicitly specified
        if 'language' in kwargs:
            params['lang'] = kwargs['language']
        
        # Add time filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': 'day',
                'week': 'week', 
                'month': 'month'
            }
            if kwargs['time_filter'] in time_filters:
                params['within'] = time_filters[kwargs['time_filter']]
        
        # Build URL with proper encoding
        from urllib.parse import urlencode
        param_string = urlencode(params)
        return f"{self.base_url}?{param_string}"
    
    async def _make_request(self, url: str) -> Optional[str]:
        """Make HTTP request with session-based cookies for Yandex."""
        try:
            import aiohttp
            import random
        except ImportError:
            self.logger.error("aiohttp not available for search requests")
            return None
        
        # Rate limiting
        await self._rate_limit()
        
        # Use Yandex-specific headers and random user agent
        headers = self.yandex_headers.copy()
        headers['User-Agent'] = random.choice(self.user_agents)
        
        try:
            # Create session if not exists
            if not self._session:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=45),
                    headers=headers
                )
            
            # First visit main page to get session cookies
            try:
                async with self._session.get('https://yandex.com/') as main_response:
                    if main_response.status != 200:
                        self.logger.warning(f"Failed to get main page: {main_response.status}")
                    # Small delay to let cookies settle
                    await asyncio.sleep(0.5)
            except:
                pass  # Continue even if main page fails
            
            # Then make the search request
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
    
    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        offset: int = 0,
        **kwargs
    ) -> SearchResponse:
        """Perform Yandex search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, **kwargs)
            self.logger.debug(f"Yandex search URL: {search_url}")
            
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
            self.logger.error(f"Yandex search failed for query '{query}': {e}")
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
        """Parse Yandex search results from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error("BeautifulSoup not available for parsing")
            return []
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Debug: Check what we actually have
            self.logger.debug(f"HTML length: {len(html)}")
            
            # Try multiple selectors for containers
            result_containers = []
            
            # Try different container selectors
            for selector in [
                'li.serp-item',
                '.serp-item',
                '.content div[data-cid]',
                '.serp-list .serp-item',
                '.organic',
                '.search-result',
                'div[data-cid]',
                '.main .content div',
                '.content .main div'
            ]:
                containers = soup.select(selector)
                if containers:
                    self.logger.debug(f"Found {len(containers)} containers with selector: {selector}")
                    result_containers = containers
                    break
            
            # Fallback: look for any links in the main content area
            if not result_containers:
                self.logger.debug("Trying fallback approach - looking for links in main content")
                main_content = soup.find('div', class_='main') or soup.find('div', class_='content') or soup
                if main_content:
                    # Look for links that might be results
                    links = main_content.find_all('a', href=True)
                    valid_links = []
                    for link in links:
                        href = link.get('href', '')
                        if href and not href.startswith('#') and 'http' in href:
                            # Create a fake container for this link
                            fake_container = link.parent or link
                            valid_links.append(fake_container)
                    
                    if valid_links:
                        self.logger.debug(f"Found {len(valid_links)} links as fallback containers")
                        result_containers = valid_links[:10]  # Limit to 10 results
            
            if not result_containers:
                self.logger.warning(f"No result containers found for query: {query}")
                return []
            
            for i, container in enumerate(result_containers):
                try:
                    # Skip ads and other non-organic results
                    if container.find(class_='serp-adv__found'):
                        continue
                    
                    # Try multiple selectors for title and link
                    title = ""
                    url = ""
                    
                    # Try different title/link selectors based on current Yandex structure
                    for title_selector in [
                        '.OrganicTitle-Link',  # Main title link
                        '.organic__title .Link',
                        '.organic__url.link',
                        'a.Link.OrganicTitle-Link',
                        'a[href]'
                    ]:
                        link_element = container.select_one(title_selector)
                        if link_element:
                            title = link_element.get_text(strip=True)
                            url = link_element.get('href', '')
                            # Skip if title is empty or just whitespace
                            if title and title.strip():
                                break
                    
                    if not title or not url:
                        continue
                    
                    # Extract snippet
                    snippet = ""
                    for snippet_selector in [
                        '.organic__text',
                        '.OrganicText',
                        '.organic__text-wrapper',
                        '.Organic-Text',
                        '.text-container',
                        '.snippet'
                    ]:
                        snippet_element = container.select_one(snippet_selector)
                        if snippet_element:
                            snippet = snippet_element.get_text(strip=True)
                            if snippet:  # Only break if we found actual content
                                break
                    
                    # Extract metadata
                    metadata = self._extract_yandex_metadata(container)
                    
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
                    self.logger.debug(f"Error parsing Yandex result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} Yandex results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing Yandex results: {e}")
            return []
    
    def _extract_yandex_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from Yandex result."""
        metadata = {'source': 'yandex'}
        
        try:
            # Look for displayed URL
            url_element = container.find(class_='organic__url')
            if url_element:
                metadata['display_url'] = url_element.get_text(strip=True)
            
            # Look for date information
            date_element = container.find(class_='organic__date')
            if date_element:
                metadata['date'] = date_element.get_text(strip=True)
            
            # Look for favicon
            favicon = container.find(class_='favicon')
            if favicon and favicon.get('alt'):
                metadata['site_name'] = favicon.get('alt')
        
        except Exception as e:
            self.logger.debug(f"Error extracting Yandex metadata: {e}")
        
        return metadata
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get Yandex-specific statistics."""
        base_stats = self.get_stats()
        
        yandex_stats = {
            **base_stats,
            'engine_features': {
                'international_coverage': True,
                'russian_focus': True,
                'time_filters': True,
                'region_filters': True,
                'good_for_eastern_europe': True
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'moderate_restrictions': True
            }
        }
        
        return yandex_stats