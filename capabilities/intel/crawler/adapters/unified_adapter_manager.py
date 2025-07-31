"""
Unified Adapter Manager
=======================

Central manager for all specialized crawler adapters. Provides:
- Automatic adapter selection based on query/URL type
- Unified interface for all specialized crawlers  
- Load balancing and failover strategies
- Comprehensive monitoring and statistics

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse
from datetime import datetime
from enum import Enum

from .base_adapter import BaseSpecializedCrawlerAdapter, AdapterResult, CrawlerType
from .search_adapter import SearchCrawlerAdapter

# Import other adapters when they become available
try:
	from .gdelt_adapter import GdeltCrawlerAdapter
	GDELT_AVAILABLE = True
except ImportError:
	GDELT_AVAILABLE = False
	GdeltCrawlerAdapter = None

try:
	from .google_news_adapter import GoogleNewsCrawlerAdapter
	GOOGLE_NEWS_AVAILABLE = True
except ImportError:
	GOOGLE_NEWS_AVAILABLE = False
	GoogleNewsCrawlerAdapter = None

try:
	from .twitter_adapter import TwitterCrawlerAdapter
	TWITTER_AVAILABLE = True
except ImportError:
	TWITTER_AVAILABLE = False
	TwitterCrawlerAdapter = None

try:
	from .youtube_adapter import YouTubeCrawlerAdapter
	YOUTUBE_AVAILABLE = True
except ImportError:
	YOUTUBE_AVAILABLE = False
	YouTubeCrawlerAdapter = None

logger = logging.getLogger(__name__)

class QueryType(Enum):
	"""Types of queries/requests for adapter selection"""
	URL = "url"                    # Direct URL to crawl
	SEARCH_QUERY = "search_query"  # Search engine query
	NEWS_QUERY = "news_query"      # News-specific search
	SOCIAL_QUERY = "social_query"  # Social media query
	VIDEO_QUERY = "video_query"    # Video content query
	EVENT_QUERY = "event_query"    # Global events query
	UNKNOWN = "unknown"

class UnifiedAdapterManager:
	"""
	Central manager for all specialized crawler adapters.
	
	Features:
	- Automatic adapter selection based on query type
	- Load balancing across multiple adapters
	- Failover and retry strategies
	- Comprehensive monitoring and statistics
	- Unified interface for all crawler types
	"""
	
	def __init__(self, tenant_id: str = "default", config: Optional[Dict[str, Any]] = None):
		self.tenant_id = tenant_id
		self.config = config or {}
		self.logger = logging.getLogger(__name__)
		
		# Initialize adapters
		self.adapters: Dict[CrawlerType, BaseSpecializedCrawlerAdapter] = {}
		self.adapter_configs = self.config.get('adapters', {})
		
		# Statistics
		self.stats = {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'requests_by_type': {crawler_type.value: 0 for crawler_type in CrawlerType},
			'adapter_performance': {},
			'query_type_detection': {query_type.value: 0 for query_type in QueryType},
			'fallback_usage': 0
		}
		
		# Configuration
		self.enable_fallback = self.config.get('enable_fallback', True)
		self.fallback_order = self.config.get('fallback_order', [
			CrawlerType.SEARCH,
			CrawlerType.GOOGLE_NEWS,
			CrawlerType.GENERAL
		])
		
		# URL patterns for adapter selection
		self.url_patterns = {
			CrawlerType.YOUTUBE: [
				r'(?:youtube\.com|youtu\.be)',
				r'youtube\.com/watch',
				r'youtu\.be/'
			],
			CrawlerType.TWITTER: [
				r'twitter\.com',
				r'x\.com',
				r't\.co'
			],
			CrawlerType.GOOGLE_NEWS: [
				r'news\.google\.com',
				r'news\.google\.',
				r'google\.com/news'
			]
		}
		
		# Keywords for query type detection
		self.query_keywords = {
			QueryType.NEWS_QUERY: [
				'news', 'breaking', 'report', 'article', 'journalist', 'media',
				'press', 'newspaper', 'headline', 'story', 'coverage'
			],
			QueryType.SOCIAL_QUERY: [
				'tweet', 'twitter', 'social', 'facebook', 'instagram', 'linkedin',
				'post', 'viral', 'trending', 'hashtag', 'social media'
			],
			QueryType.VIDEO_QUERY: [
				'video', 'youtube', 'watch', 'streaming', 'channel', 'playlist',
				'documentary', 'movie', 'film', 'clip', 'tutorial'
			],
			QueryType.EVENT_QUERY: [
				'event', 'happening', 'incident', 'crisis', 'disaster', 'conflict',
				'protest', 'election', 'summit', 'conference', 'emergency'
			]
		}
	
	async def initialize(self) -> None:
		"""Initialize all available adapters."""
		try:
			# Always initialize SearchCrawler as it's our primary fallback
			search_config = self.adapter_configs.get('search', {})
			self.adapters[CrawlerType.SEARCH] = SearchCrawlerAdapter(self.tenant_id, search_config)
			await self.adapters[CrawlerType.SEARCH].initialize()
			self.logger.info("SearchCrawlerAdapter initialized")
			
			# Initialize other adapters if available
			if GDELT_AVAILABLE:
				gdelt_config = self.adapter_configs.get('gdelt', {})
				self.adapters[CrawlerType.GDELT] = GdeltCrawlerAdapter(self.tenant_id, gdelt_config)
				await self.adapters[CrawlerType.GDELT].initialize()
				self.logger.info("GdeltCrawlerAdapter initialized")
			
			if GOOGLE_NEWS_AVAILABLE:
				news_config = self.adapter_configs.get('google_news', {})
				self.adapters[CrawlerType.GOOGLE_NEWS] = GoogleNewsCrawlerAdapter(self.tenant_id, news_config)
				await self.adapters[CrawlerType.GOOGLE_NEWS].initialize()
				self.logger.info("GoogleNewsCrawlerAdapter initialized")
			
			if TWITTER_AVAILABLE:
				twitter_config = self.adapter_configs.get('twitter', {})
				self.adapters[CrawlerType.TWITTER] = TwitterCrawlerAdapter(self.tenant_id, twitter_config)
				await self.adapters[CrawlerType.TWITTER].initialize()
				self.logger.info("TwitterCrawlerAdapter initialized")
			
			if YOUTUBE_AVAILABLE:
				youtube_config = self.adapter_configs.get('youtube', {})
				self.adapters[CrawlerType.YOUTUBE] = YouTubeCrawlerAdapter(self.tenant_id, youtube_config)
				await self.adapters[CrawlerType.YOUTUBE].initialize()
				self.logger.info("YouTubeCrawlerAdapter initialized")
			
			self.logger.info(f"UnifiedAdapterManager initialized with {len(self.adapters)} adapters")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize UnifiedAdapterManager: {e}")
			raise
	
	async def crawl_single(
		self, 
		query_or_url: str, 
		preferred_adapter: Optional[CrawlerType] = None,
		**kwargs
	) -> AdapterResult:
		"""
		Crawl a single item using the most appropriate adapter.
		
		Args:
			query_or_url: Query string or URL to crawl
			preferred_adapter: Specific adapter to use (optional)
			**kwargs: Additional parameters for the adapter
			
		Returns:
			AdapterResult from the selected adapter
		"""
		self.stats['total_requests'] += 1
		
		try:
			# Determine the best adapter
			if preferred_adapter and preferred_adapter in self.adapters:
				selected_adapter = preferred_adapter
			else:
				selected_adapter = self._select_adapter(query_or_url)
			
			# Update statistics
			self.stats['requests_by_type'][selected_adapter.value] += 1
			
			# Get the adapter
			adapter = self.adapters.get(selected_adapter)
			if not adapter:
				# Fallback to search adapter
				if self.enable_fallback and CrawlerType.SEARCH in self.adapters:
					adapter = self.adapters[CrawlerType.SEARCH]
					selected_adapter = CrawlerType.SEARCH
					self.stats['fallback_usage'] += 1
					self.logger.warning(f"Falling back to SearchCrawler for: {query_or_url}")
				else:
					raise ValueError(f"No adapter available for {selected_adapter}")
			
			# Perform the crawl
			result = await adapter.crawl_single(query_or_url, **kwargs)
			
			# Update statistics
			if result.success:
				self.stats['successful_requests'] += 1
			else:
				self.stats['failed_requests'] += 1
			
			# Add adapter info to result metadata
			result.processing_metadata.update({
				'selected_adapter': selected_adapter.value,
				'adapter_class': adapter.__class__.__name__,
				'fallback_used': selected_adapter != self._select_adapter(query_or_url)
			})
			
			return result
			
		except Exception as e:
			self.stats['failed_requests'] += 1
			self.logger.error(f"Crawl failed for '{query_or_url}': {e}")
			
			# Try fallback if enabled and not already using it
			if (self.enable_fallback and 
				selected_adapter != CrawlerType.SEARCH and 
				CrawlerType.SEARCH in self.adapters):
				
				try:
					self.stats['fallback_usage'] += 1
					fallback_adapter = self.adapters[CrawlerType.SEARCH]
					result = await fallback_adapter.crawl_single(query_or_url, **kwargs)
					result.processing_metadata['fallback_used'] = True
					result.processing_metadata['original_adapter_failed'] = selected_adapter.value
					return result
				except Exception as fallback_error:
					self.logger.error(f"Fallback also failed: {fallback_error}")
			
			# Create error result
			return AdapterResult(
				url=query_or_url,
				title=None,
				content=f"# Crawl Failed\n\nUnable to crawl: {str(e)}",
				success=False,
				error=str(e),
				crawler_type=CrawlerType.GENERAL,
				tenant_id=self.tenant_id
			)
	
	async def crawl_batch(
		self,
		queries_or_urls: List[str],
		max_concurrent: int = 5,
		preferred_adapter: Optional[CrawlerType] = None,
		**kwargs
	) -> List[AdapterResult]:
		"""
		Crawl multiple items concurrently using appropriate adapters.
		
		Args:
			queries_or_urls: List of queries or URLs
			max_concurrent: Maximum concurrent requests
			preferred_adapter: Specific adapter to use for all items
			**kwargs: Additional parameters for adapters
			
		Returns:
			List of AdapterResult objects
		"""
		# Group by adapter type for efficient batch processing
		if preferred_adapter:
			# Use the same adapter for all
			adapter_groups = {preferred_adapter: queries_or_urls}
		else:
			# Group by automatically selected adapters
			adapter_groups = {}
			for item in queries_or_urls:
				adapter_type = self._select_adapter(item)
				if adapter_type not in adapter_groups:
					adapter_groups[adapter_type] = []
				adapter_groups[adapter_type].append(item)
		
		# Process each group
		all_tasks = []
		for adapter_type, items in adapter_groups.items():
			adapter = self.adapters.get(adapter_type)
			if adapter:
				# Use adapter's batch method if available
				if hasattr(adapter, 'crawl_batch'):
					task = adapter.crawl_batch(items, max_concurrent, **kwargs)
				else:
					# Fallback to individual processing
					semaphore = asyncio.Semaphore(max_concurrent)
					async def crawl_with_semaphore(item):
						async with semaphore:
							return await adapter.crawl_single(item, **kwargs)
					
					task = asyncio.gather(*[crawl_with_semaphore(item) for item in items])
				
				all_tasks.append(task)
		
		# Wait for all groups to complete
		if all_tasks:
			group_results = await asyncio.gather(*all_tasks, return_exceptions=True)
			
			# Flatten results
			all_results = []
			for group_result in group_results:
				if isinstance(group_result, Exception):
					# Handle group failure
					self.logger.error(f"Batch group failed: {group_result}")
					continue
				
				if isinstance(group_result, list):
					all_results.extend(group_result)
				else:
					all_results.append(group_result)
			
			return all_results
		
		return []
	
	def _select_adapter(self, query_or_url: str) -> CrawlerType:
		"""
		Select the most appropriate adapter based on the query/URL.
		
		Args:
			query_or_url: Query string or URL to analyze
			
		Returns:
			CrawlerType of the selected adapter
		"""
		# Detect query type
		query_type = self._detect_query_type(query_or_url)
		self.stats['query_type_detection'][query_type.value] += 1
		
		# URL-based selection
		if query_type == QueryType.URL:
			parsed_url = urlparse(query_or_url.lower())
			domain = parsed_url.netloc
			
			# Check domain patterns
			for crawler_type, patterns in self.url_patterns.items():
				if crawler_type in self.adapters:
					for pattern in patterns:
						if re.search(pattern, domain):
							return crawler_type
		
		# Query-based selection
		elif query_type == QueryType.NEWS_QUERY and CrawlerType.GOOGLE_NEWS in self.adapters:
			return CrawlerType.GOOGLE_NEWS
		elif query_type == QueryType.EVENT_QUERY and CrawlerType.GDELT in self.adapters:
			return CrawlerType.GDELT
		elif query_type == QueryType.SOCIAL_QUERY and CrawlerType.TWITTER in self.adapters:
			return CrawlerType.TWITTER
		elif query_type == QueryType.VIDEO_QUERY and CrawlerType.YOUTUBE in self.adapters:
			return CrawlerType.YOUTUBE
		
		# Default to search
		return CrawlerType.SEARCH
	
	def _detect_query_type(self, query_or_url: str) -> QueryType:
		"""
		Detect the type of query or URL.
		
		Args:
			query_or_url: Query string or URL to analyze
			
		Returns:
			QueryType enum value
		"""
		# Check if it's a URL
		if query_or_url.startswith(('http://', 'https://', 'www.')):
			return QueryType.URL
		
		# Check for URL-like patterns without protocol
		if '.' in query_or_url and '/' in query_or_url:
			return QueryType.URL
		
		# Analyze query keywords
		query_lower = query_or_url.lower()
		
		for query_type, keywords in self.query_keywords.items():
			keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
			if keyword_matches >= 1:  # Match threshold
				return query_type
		
		# Default to search query
		return QueryType.SEARCH_QUERY
	
	async def cleanup(self) -> None:
		"""Clean up all adapter resources."""
		for adapter in self.adapters.values():
			try:
				await adapter.cleanup()
			except Exception as e:
				self.logger.error(f"Error cleaning up adapter {adapter.__class__.__name__}: {e}")
		
		self.adapters.clear()
		self.logger.info("UnifiedAdapterManager cleaned up")
	
	def get_available_adapters(self) -> Dict[str, Dict[str, Any]]:
		"""Get information about available adapters."""
		return {
			adapter_type.value: {
				'available': adapter_type in self.adapters,
				'class': self.adapters[adapter_type].__class__.__name__ if adapter_type in self.adapters else None,
				'supported_parameters': self.adapters[adapter_type].get_supported_parameters() if adapter_type in self.adapters else {}
			}
			for adapter_type in CrawlerType
		}
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get comprehensive statistics from all adapters."""
		# Compile adapter-specific stats
		adapter_stats = {}
		for adapter_type, adapter in self.adapters.items():
			try:
				adapter_stats[adapter_type.value] = adapter.get_stats()
			except Exception as e:
				self.logger.error(f"Error getting stats from {adapter_type.value}: {e}")
				adapter_stats[adapter_type.value] = {'error': str(e)}
		
		# Calculate overall success rate
		success_rate = 0.0
		if self.stats['total_requests'] > 0:
			success_rate = self.stats['successful_requests'] / self.stats['total_requests']
		
		return {
			'manager_stats': {
				**self.stats,
				'success_rate': success_rate,
				'active_adapters': len(self.adapters),
				'tenant_id': self.tenant_id
			},
			'adapter_stats': adapter_stats,
			'available_adapters': self.get_available_adapters()
		}
	
	def reset_stats(self):
		"""Reset all statistics."""
		self.stats = {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'requests_by_type': {crawler_type.value: 0 for crawler_type in CrawlerType},
			'adapter_performance': {},
			'query_type_detection': {query_type.value: 0 for query_type in QueryType},
			'fallback_usage': 0
		}
		
		# Reset individual adapter stats
		for adapter in self.adapters.values():
			adapter.reset_stats()
		
		self.logger.info("All statistics reset")

# Convenience function for creating a unified manager
def create_unified_manager(
	tenant_id: str = "default", 
	config: Optional[Dict[str, Any]] = None
) -> UnifiedAdapterManager:
	"""
	Create and initialize a UnifiedAdapterManager.
	
	Args:
		tenant_id: Tenant identifier
		config: Configuration dictionary
		
	Returns:
		Initialized UnifiedAdapterManager
	"""
	return UnifiedAdapterManager(tenant_id, config)