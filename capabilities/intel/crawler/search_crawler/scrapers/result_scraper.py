"""
Result Scraper
==============

Scrapes content from search result URLs using the news crawler.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

from ..engines.base_search_engine import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Scraped content from a search result."""
    url: str
    title: str
    content: str
    summary: str
    publish_date: Optional[datetime] = None
    author: Optional[str] = None
    language: str = 'en'
    images: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scrape_timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""


class ResultScraper:
    """Scrapes content from search result URLs using news crawler."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize news crawler
        self.news_crawler = None
        self._init_news_crawler()
        
        # Scraping configuration
        self.max_concurrent = self.config.get('max_concurrent', 5)
        self.timeout = self.config.get('timeout', 30)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        
        # Content filtering
        self.min_content_length = self.config.get('min_content_length', 100)
        self.max_content_length = self.config.get('max_content_length', 100000)
        
        # Statistics
        self.stats = {
            'total_scraped': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'average_content_length': 0.0
        }
    
    def _init_news_crawler(self):
        """Initialize the news crawler."""
        try:
            from ...news_crawler import NewsCrawler
            
            # Configure news crawler for result scraping
            crawler_config = {
                'enable_stealth': True,
                'enable_cloudflare_bypass': True,
                'timeout': self.timeout,
                'retry_attempts': self.retry_attempts,
                'extract_content': True,
                'extract_metadata': True,
                'extract_images': True,
                'extract_links': True
            }
            
            self.news_crawler = NewsCrawler(crawler_config)
            self.logger.info("News crawler initialized for result scraping")
        
        except ImportError as e:
            self.logger.error(f"Failed to import news crawler: {e}")
            self.news_crawler = None
        except Exception as e:
            self.logger.error(f"Failed to initialize news crawler: {e}")
            self.news_crawler = None
    
    async def scrape_results(
        self,
        results: List[SearchResult],
        max_results: Optional[int] = None
    ) -> List[ScrapedContent]:
        """
        Scrape content from multiple search results.
        
        Args:
            results: List of search results to scrape
            max_results: Maximum number of results to scrape
            
        Returns:
            List of scraped content
        """
        if not self.news_crawler:
            self.logger.error("News crawler not available for scraping")
            return []
        
        # Limit results if specified
        if max_results:
            results = results[:max_results]
        
        self.logger.info(f"Starting to scrape {len(results)} search results")
        
        # Create semaphore for concurrent scraping
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create scraping tasks
        tasks = [
            self._scrape_result_with_semaphore(semaphore, result)
            for result in results
        ]
        
        # Execute tasks concurrently
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and failed scrapes
        valid_contents = []
        for content in scraped_contents:
            if isinstance(content, ScrapedContent):
                valid_contents.append(content)
            elif isinstance(content, Exception):
                self.logger.error(f"Scraping task failed: {content}")
        
        # Update statistics
        self._update_stats(valid_contents)
        
        self.logger.info(f"Successfully scraped {len(valid_contents)}/{len(results)} results")
        return valid_contents
    
    async def _scrape_result_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        result: SearchResult
    ) -> ScrapedContent:
        """Scrape a single result with semaphore control."""
        async with semaphore:
            return await self.scrape_single_result(result)
    
    async def scrape_single_result(self, result: SearchResult) -> ScrapedContent:
        """
        Scrape content from a single search result.
        
        Args:
            result: Search result to scrape
            
        Returns:
            Scraped content
        """
        start_time = datetime.now()
        
        try:
            self.logger.debug(f"Scraping URL: {result.url}")
            
            # Use news crawler to extract content
            crawl_result = await self.news_crawler.crawl_url(result.url)
            
            if not crawl_result or not crawl_result.get('success', False):
                error_msg = crawl_result.get('error', 'Unknown error') if crawl_result else 'No result returned'
                return ScrapedContent(
                    url=result.url,
                    title=result.title,
                    content="",
                    summary="",
                    success=False,
                    error_message=f"News crawler failed: {error_msg}"
                )
            
            # Extract content data
            content_data = crawl_result.get('content', {})
            
            # Build scraped content object
            scraped_content = ScrapedContent(
                url=result.url,
                title=content_data.get('title', result.title),
                content=content_data.get('text', ''),
                summary=content_data.get('summary', result.snippet),
                publish_date=self._parse_date(content_data.get('publish_date')),
                author=content_data.get('author'),
                language=content_data.get('language', 'en'),
                images=content_data.get('images', []),
                links=content_data.get('links', []),
                keywords=content_data.get('keywords', []),
                metadata={
                    'original_result': result,
                    'scrape_time': datetime.now() - start_time,
                    'crawler_metadata': content_data.get('metadata', {}),
                    'domain': urlparse(result.url).netloc
                },
                success=True
            )
            
            # Validate content quality
            if not self._validate_content(scraped_content):
                scraped_content.success = False
                scraped_content.error_message = "Content quality validation failed"
            
            return scraped_content
        
        except Exception as e:
            self.logger.error(f"Failed to scrape {result.url}: {e}")
            return ScrapedContent(
                url=result.url,
                title=result.title,
                content="",
                summary="",
                success=False,
                error_message=str(e),
                metadata={
                    'original_result': result,
                    'scrape_time': datetime.now() - start_time
                }
            )
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Try common date formats
            import dateutil.parser
            return dateutil.parser.parse(date_str)
        except:
            try:
                # Fallback to basic parsing
                from datetime import datetime
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                self.logger.debug(f"Failed to parse date: {date_str}")
                return None
    
    def _validate_content(self, content: ScrapedContent) -> bool:
        """Validate scraped content quality."""
        # Check minimum content length
        if len(content.content) < self.min_content_length:
            self.logger.debug(f"Content too short: {len(content.content)} chars")
            return False
        
        # Check maximum content length
        if len(content.content) > self.max_content_length:
            self.logger.debug(f"Content too long: {len(content.content)} chars")
            return False
        
        # Check for valid title
        if not content.title or len(content.title.strip()) < 5:
            self.logger.debug("Invalid or missing title")
            return False
        
        # Content should not be mostly navigation/boilerplate
        words = content.content.split()
        if len(words) < 20:
            self.logger.debug("Content has too few words")
            return False
        
        return True
    
    def _update_stats(self, scraped_contents: List[ScrapedContent]):
        """Update scraping statistics."""
        self.stats['total_scraped'] += len(scraped_contents)
        
        successful = [c for c in scraped_contents if c.success]
        self.stats['successful_scrapes'] += len(successful)
        self.stats['failed_scrapes'] += len(scraped_contents) - len(successful)
        
        # Update average content length
        if successful:
            total_length = sum(len(c.content) for c in successful)
            current_avg = self.stats['average_content_length']
            total_successful = self.stats['successful_scrapes']
            
            if total_successful > 0:
                self.stats['average_content_length'] = (
                    (current_avg * (total_successful - len(successful)) + total_length) /
                    total_successful
                )
    
    async def scrape_with_filter(
        self,
        results: List[SearchResult],
        content_filter: Optional[callable] = None,
        max_results: Optional[int] = None
    ) -> List[ScrapedContent]:
        """
        Scrape results with content filtering.
        
        Args:
            results: Search results to scrape
            content_filter: Function to filter results before scraping
            max_results: Maximum results to process
            
        Returns:
            Filtered and scraped content
        """
        # Apply pre-scraping filter
        if content_filter:
            filtered_results = [r for r in results if content_filter(r)]
            self.logger.info(f"Filtered {len(results)} -> {len(filtered_results)} results")
            results = filtered_results
        
        # Scrape content
        scraped_contents = await self.scrape_results(results, max_results)
        
        # Apply post-scraping content filter if needed
        if content_filter and hasattr(content_filter, '__name__') and 'content' in content_filter.__name__:
            filtered_contents = [c for c in scraped_contents if c.success and content_filter(c)]
            self.logger.info(f"Post-scrape filtered {len(scraped_contents)} -> {len(filtered_contents)} contents")
            return filtered_contents
        
        return scraped_contents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        success_rate = 0.0
        if self.stats['total_scraped'] > 0:
            success_rate = self.stats['successful_scrapes'] / self.stats['total_scraped']
        
        return {
            'total_scraped': self.stats['total_scraped'],
            'successful_scrapes': self.stats['successful_scrapes'],
            'failed_scrapes': self.stats['failed_scrapes'],
            'success_rate': success_rate,
            'average_content_length': self.stats['average_content_length'],
            'configuration': {
                'max_concurrent': self.max_concurrent,
                'timeout': self.timeout,
                'min_content_length': self.min_content_length,
                'max_content_length': self.max_content_length
            }
        }
    
    def reset_stats(self):
        """Reset scraping statistics."""
        self.stats = {
            'total_scraped': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'average_content_length': 0.0
        }