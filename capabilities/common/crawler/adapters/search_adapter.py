"""
Search Crawler Adapter
======================

Adapter for integrating the existing SearchCrawler (multi-engine search)
with the APG Simple API. Provides unified access to multiple search engines
with intelligent ranking and content downloading.

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_adapter import BaseSpecializedCrawlerAdapter, AdapterResult, CrawlerType

# Import the existing SearchCrawler
try:
	from ..search_crawler.core.search_crawler import SearchCrawler, SearchCrawlerConfig, EnhancedSearchResult
	SEARCH_CRAWLER_AVAILABLE = True
except ImportError:
	SEARCH_CRAWLER_AVAILABLE = False
	SearchCrawler = None
	SearchCrawlerConfig = None

logger = logging.getLogger(__name__)

class SearchCrawlerAdapter(BaseSpecializedCrawlerAdapter):
	"""
	Adapter for the multi-engine SearchCrawler.
	
	Features:
	- Multi-engine search with intelligent ranking
	- Content downloading and parsing
	- Deduplication and quality scoring
	- Caching and rate limiting
	"""
	
	def __init__(self, tenant_id: str = "default", config: Optional[Dict[str, Any]] = None):
		super().__init__(CrawlerType.SEARCH, tenant_id)
		self.config = config or {}
		self.search_crawler: Optional[SearchCrawler] = None
		
	async def initialize(self) -> None:
		"""Initialize the SearchCrawler with configuration."""
		if not SEARCH_CRAWLER_AVAILABLE:
			raise ImportError("SearchCrawler not available. Check search_crawler module.")
		
		try:
			# Create SearchCrawler configuration
			search_config = SearchCrawlerConfig(
				engines=self.config.get('engines', ['google', 'bing', 'duckduckgo']),
				max_results_per_engine=self.config.get('max_results_per_engine', 10),
				total_max_results=self.config.get('total_max_results', 20),
				parallel_searches=self.config.get('parallel_searches', True),
				download_content=self.config.get('download_content', True),
				parse_content=self.config.get('parse_content', True),
				ranking_strategy=self.config.get('ranking_strategy', 'hybrid'),
				use_stealth=self.config.get('use_stealth', True)
			)
			
			# Initialize SearchCrawler
			self.search_crawler = SearchCrawler(search_config)
			
			self.logger.info(f"SearchCrawler initialized with engines: {search_config.engines}")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize SearchCrawler: {e}")
			raise
	
	async def crawl_single(self, query: str, **kwargs) -> AdapterResult:
		"""
		Perform a single search query using the SearchCrawler.
		
		Args:
			query: Search query string
			**kwargs: Additional search parameters
			
		Returns:
			AdapterResult with search results in markdown format
		"""
		if not self.search_crawler:
			await self.initialize()
		
		try:
			return await self._execute_with_stats(
				self._perform_single_search(query, **kwargs)
			)
		except Exception as e:
			return self._create_error_result(
				query,
				f"Search failed: {str(e)}",
				query=query,
				parameters=kwargs
			)
	
	async def _perform_single_search(self, query: str, **kwargs) -> AdapterResult:
		"""Perform the actual search operation."""
		# Extract parameters
		max_results = kwargs.get('max_results', 10)
		engines = kwargs.get('engines', None)
		download_content = kwargs.get('download_content', True)
		
		# Perform search
		search_results = await self.search_crawler.search(
			query=query,
			max_results=max_results,
			engines=engines,
			download_content=download_content
		)
		
		if not search_results:
			return self._create_error_result(
				query,
				"No search results found",
				query=query,
				engines_tried=engines or "all"
			)
		
		# Convert to markdown format
		markdown_content = self._format_search_results_as_markdown(query, search_results)
		
		# Extract primary result for title
		primary_result = search_results[0] if search_results else None
		title = f"Search Results: {query}"
		
		return AdapterResult(
			url=f"search://{query}",
			title=title,
			content=markdown_content,
			success=True,
			crawler_type=CrawlerType.SEARCH,
			original_source="multi_engine_search",
			tenant_id=self.tenant_id,
			language=self._detect_language(markdown_content),
			word_count=self._count_words(markdown_content),
			specialized_metadata={
				'query': query,
				'total_results': len(search_results),
				'engines_used': list(set(r.engines_found for r in search_results if hasattr(r, 'engines_found'))),
				'has_content': any(r.content for r in search_results if hasattr(r, 'content')),
				'duplicates_found': sum(1 for r in search_results if hasattr(r, 'is_duplicate') and r.is_duplicate),
				'average_relevance': sum(r.relevance_score for r in search_results) / len(search_results) if search_results else 0
			},
			processing_metadata={
				'adapter': 'SearchCrawlerAdapter',
				'timestamp': datetime.now().isoformat(),
				'parameters': kwargs
			}
		)
	
	async def crawl_batch(
		self, 
		queries: List[str], 
		max_concurrent: int = 3,
		**kwargs
	) -> List[AdapterResult]:
		"""
		Perform multiple search queries concurrently.
		
		Args:
			queries: List of search queries
			max_concurrent: Maximum concurrent searches
			**kwargs: Additional search parameters
			
		Returns:
			List of AdapterResult objects
		"""
		if not self.search_crawler:
			await self.initialize()
		
		# Create semaphore for concurrency control
		semaphore = asyncio.Semaphore(max_concurrent)
		
		async def search_with_semaphore(query: str) -> AdapterResult:
			async with semaphore:
				return await self.crawl_single(query, **kwargs)
		
		# Execute all searches concurrently
		tasks = [search_with_semaphore(query) for query in queries]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Convert exceptions to error results
		final_results = []
		for i, result in enumerate(results):
			if isinstance(result, Exception):
				error_result = self._create_error_result(
					queries[i],
					f"Batch search failed: {str(result)}",
					query=queries[i],
					batch_index=i
				)
				final_results.append(error_result)
			else:
				final_results.append(result)
		
		return final_results
	
	async def cleanup(self) -> None:
		"""Clean up SearchCrawler resources."""
		if self.search_crawler:
			await self.search_crawler.close()
			self.search_crawler = None
			self.logger.info("SearchCrawler resources cleaned up")
	
	def get_supported_parameters(self) -> Dict[str, Any]:
		"""Get parameters supported by the SearchCrawler."""
		return {
			'engines': {
				'type': 'list[str]',
				'description': 'Search engines to use',
				'options': ['google', 'bing', 'duckduckgo', 'yandex', 'baidu', 'yahoo', 'startpage', 'searx', 'brave', 'mojeek', 'swisscows'],
				'default': ['google', 'bing', 'duckduckgo']
			},
			'max_results': {
				'type': 'int',
				'description': 'Maximum number of results to return',
				'default': 10,
				'range': [1, 100]
			},
			'download_content': {
				'type': 'bool',
				'description': 'Whether to download and parse page content',
				'default': True
			},
			'ranking_strategy': {
				'type': 'str',
				'description': 'Result ranking strategy',
				'options': ['relevance', 'freshness', 'authority', 'hybrid'],
				'default': 'hybrid'
			},
			'use_stealth': {
				'type': 'bool',
				'description': 'Use stealth techniques for crawling',
				'default': True
			},
			'language': {
				'type': 'str',
				'description': 'Preferred language for results',
				'optional': True
			},
			'region': {
				'type': 'str',
				'description': 'Geographic region for results',
				'optional': True
			},
			'time_range': {
				'type': 'str',
				'description': 'Time range for results (day, week, month, year)',
				'options': ['day', 'week', 'month', 'year', 'all'],
				'optional': True
			}
		}
	
	def _format_search_results_as_markdown(
		self, 
		query: str, 
		results: List[EnhancedSearchResult]
	) -> str:
		"""Format search results as clean markdown."""
		markdown_parts = [
			f"# Search Results: {query}",
			f"",
			f"Found **{len(results)}** results from multiple search engines.",
			f"",
		]
		
		for i, result in enumerate(results, 1):
			# Basic result information
			markdown_parts.extend([
				f"## {i}. {result.title or 'No Title'}",
				f"",
				f"**URL:** {result.url}",
				f"**Source:** {result.engine}",
				f"**Rank:** {result.rank}",
			])
			
			# Enhanced metadata if available
			if hasattr(result, 'engines_found') and result.engines_found:
				engines_list = ', '.join(result.engines_found)
				markdown_parts.append(f"**Found in:** {engines_list}")
			
			if hasattr(result, 'relevance_score'):
				markdown_parts.append(f"**Relevance:** {result.relevance_score:.2f}")
			
			# Snippet
			if result.snippet:
				markdown_parts.extend([
					f"",
					f"**Summary:**",
					f"{result.snippet}",
				])
			
			# Full content if available
			if hasattr(result, 'content') and result.content:
				content_preview = result.content[:500]
				if len(result.content) > 500:
					content_preview += "..."
				
				markdown_parts.extend([
					f"",
					f"**Content Preview:**",
					f"```",
					content_preview,
					f"```",
				])
			
			# Metadata
			if result.metadata:
				markdown_parts.extend([
					f"",
					f"**Additional Info:**",
				])
				for key, value in result.metadata.items():
					if value and key not in ['raw_html', 'headers']:
						markdown_parts.append(f"- **{key.replace('_', ' ').title()}:** {value}")
			
			markdown_parts.append(f"")
			markdown_parts.append("---")
			markdown_parts.append(f"")
		
		# Search statistics
		if hasattr(results[0], 'engines_found'):
			all_engines = set()
			for result in results:
				if hasattr(result, 'engines_found'):
					all_engines.update(result.engines_found)
			
			markdown_parts.extend([
				f"## Search Statistics",
				f"",
				f"**Engines Used:** {', '.join(sorted(all_engines))}",
				f"**Total Results:** {len(results)}",
				f"**Unique URLs:** {len(set(r.url for r in results))}",
			])
			
			# Duplicates info
			duplicates = sum(1 for r in results if hasattr(r, 'is_duplicate') and r.is_duplicate)
			if duplicates > 0:
				markdown_parts.append(f"**Duplicates Filtered:** {duplicates}")
		
		return '\n'.join(markdown_parts)
	
	def get_search_stats(self) -> Dict[str, Any]:
		"""Get SearchCrawler-specific statistics."""
		base_stats = self.get_stats()
		
		if self.search_crawler:
			search_stats = self.search_crawler.get_stats()
			base_stats.update({
				'search_crawler_stats': search_stats,
				'engines_available': len(self.search_crawler.engines),
				'cache_size': search_stats.get('cache_size', 0),
				'engine_performance': search_stats.get('engine_performance', {})
			})
		
		return base_stats