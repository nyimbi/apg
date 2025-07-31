"""
Search Content Pipeline Integration
==================================

Two-stage pipeline that combines multi-engine search discovery with gen_crawler content downloading:

1. **Discovery Stage**: Multi-engine search crawler finds relevant URLs
2. **Download Stage**: Gen crawler downloads full content from discovered URLs

This creates a comprehensive search-to-content pipeline that discovers relevant pages
across multiple search engines and then retrieves their full content for analysis.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import time
import hashlib

# Import search crawler components
try:
	from .core.search_crawler import SearchCrawler, SearchCrawlerConfig
	from .core.conflict_search_crawler import ConflictSearchCrawler
	from .engines.base_search_engine import SearchResult
	SEARCH_CRAWLER_AVAILABLE = True
except ImportError:
	SearchCrawler = None
	SearchCrawlerConfig = None
	ConflictSearchCrawler = None
	SearchResult = None
	SEARCH_CRAWLER_AVAILABLE = False

# Import gen_crawler components  
try:
	from ..gen_crawler import GenCrawler, GenCrawlResult, GenSiteResult, create_gen_crawler
	from ..gen_crawler.config import GenCrawlerConfig
	GEN_CRAWLER_AVAILABLE = True
except ImportError:
	GenCrawler = None
	GenCrawlResult = None
	GenSiteResult = None
	create_gen_crawler = None
	GenCrawlerConfig = None
	GEN_CRAWLER_AVAILABLE = False

# Import database components
try:
	from ..google_news_crawler.database import InformationUnitsManager, create_information_units_manager
	DATABASE_AVAILABLE = True
except ImportError:
	InformationUnitsManager = None
	create_information_units_manager = None
	DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SearchContentPipelineConfig:
	"""Configuration for the search content pipeline."""
	# Search discovery settings
	search_engines: List[str] = field(default_factory=lambda: ['google', 'bing', 'duckduckgo', 'yandex'])  # Yandex re-enabled
	max_results_per_engine: int = 20
	total_max_results: int = 100
	search_timeout: float = 30.0
	
	# Gen crawler download settings
	gen_crawler_max_pages: int = 1  # Usually just download the main page
	gen_crawler_timeout: int = 30
	gen_crawler_concurrent_downloads: int = 5
	
	# Pipeline settings
	download_full_content: bool = True
	skip_duplicate_urls: bool = True
	enable_content_analysis: bool = True
	enable_database_storage: bool = True
	database_url: str = "postgresql:///lnd"
	
	# Performance settings
	batch_size: int = 10  # Process URLs in batches
	delay_between_batches: float = 2.0  # Respectful crawling
	max_retries: int = 2
	url_quality_threshold: float = 0.5  # Minimum quality score for URL selection
	
	# Content filtering
	allowed_domains: List[str] = field(default_factory=list)  # Empty = all domains allowed
	blocked_domains: List[str] = field(default_factory=lambda: [
		'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
		'linkedin.com', 'pinterest.com', 'reddit.com'
	])
	min_content_length: int = 200
	max_content_length: int = 100000

@dataclass 
class EnrichedSearchResult:
	"""Search result enriched with full content from gen_crawler."""
	# Original search result
	search_result: SearchResult
	search_engine: str
	search_query: str
	
	# Downloaded content from gen_crawler
	full_content: Optional[str] = None
	full_title: Optional[str] = None
	content_metadata: Dict[str, Any] = field(default_factory=dict)
	download_success: bool = False
	download_error: Optional[str] = None
	download_time: float = 0.0
	
	# Content analysis
	word_count: int = 0
	content_quality_score: float = 0.0
	content_language: Optional[str] = None
	extracted_entities: List[str] = field(default_factory=list)
	
	# URL analysis
	url_hash: str = field(default_factory=str)
	domain: str = field(default_factory=str)
	
	# Processing metadata
	enriched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	pipeline_id: str = field(default_factory=lambda: f"search_pipeline_{int(time.time())}")

class SearchContentPipeline:
	"""
	Complete search content pipeline that discovers URLs via multi-engine search
	and downloads full content via gen_crawler.
	"""
	
	def __init__(self, config: Optional[SearchContentPipelineConfig] = None):
		"""Initialize the search content pipeline."""
		self.config = config or SearchContentPipelineConfig()
		
		# Initialize search crawler
		self.search_crawler: Optional[SearchCrawler] = None
		if SEARCH_CRAWLER_AVAILABLE:
			search_config = SearchCrawlerConfig(
				engines=self.config.search_engines,
				max_results_per_engine=self.config.max_results_per_engine,
				total_max_results=self.config.total_max_results,
				timeout=self.config.search_timeout,
				download_content=False,  # We'll handle content download separately
				parallel_searches=True
			)
			self.search_crawler = SearchCrawler(config=search_config)
		
		# Initialize gen_crawler
		self.gen_crawler: Optional[GenCrawler] = None
		if GEN_CRAWLER_AVAILABLE:
			try:
				# Create gen_crawler configuration as dictionary
				gen_config = {
					'performance': {
						'max_pages_per_site': self.config.gen_crawler_max_pages,
						'request_timeout': self.config.gen_crawler_timeout,
						'max_concurrent': self.config.gen_crawler_concurrent_downloads,
						'max_retries': 2,
						'crawl_delay': 1.0
					},
					'content_filters': {
						'min_content_length': self.config.min_content_length,
						'max_content_length': self.config.max_content_length,
						'exclude_extensions': ['.pdf', '.doc', '.xls', '.zip', '.exe'],
						'exclude_patterns': ['login', 'logout', 'register', 'cart', 'checkout']
					},
					'stealth': {
						'enable_stealth': True,
						'user_agent': 'SearchContentPipeline/1.0 (+https://datacraft.co.ke)',
						'respect_robots_txt': True
					}
				}
				
				self.gen_crawler = create_gen_crawler(gen_config) if create_gen_crawler else None
				
			except Exception as e:
				logger.warning(f"Failed to configure gen_crawler: {e}")
				self.gen_crawler = None
		
		# Initialize database manager
		self.db_manager: Optional[InformationUnitsManager] = None
		if DATABASE_AVAILABLE and self.config.enable_database_storage:
			try:
				self.db_manager = create_information_units_manager(self.config.database_url)
			except Exception as e:
				logger.warning(f"Failed to initialize database manager: {e}")
				self.db_manager = None
		
		# Tracking
		self.processed_urls: Set[str] = set()
		self.stats = {
			'search_queries': 0,
			'urls_discovered': 0,
			'urls_downloaded': 0,
			'download_failures': 0,
			'duplicate_urls_skipped': 0,
			'database_records_created': 0,
			'pipeline_start_time': None
		}
		
		logger.info(f"Search content pipeline initialized (search_available: {SEARCH_CRAWLER_AVAILABLE}, gen_crawler_available: {GEN_CRAWLER_AVAILABLE})")
	
	async def initialize(self) -> None:
		"""Initialize all pipeline components."""
		logger.info("🚀 Initializing search content pipeline...")
		
		# Initialize database manager
		if self.db_manager:
			await self.db_manager.initialize()
			logger.info("✅ Database manager initialized")
		
		# Search crawler and gen_crawler are initialized on-demand
		self.stats['pipeline_start_time'] = datetime.now(timezone.utc)
		
		logger.info("✅ Search content pipeline initialization complete")
	
	async def search_and_download(self, 
								  queries: List[str],
								  languages: Optional[List[str]] = None,
								  regions: Optional[List[str]] = None) -> List[EnrichedSearchResult]:
		"""
		Main pipeline method: search for URLs and download full content.
		
		Args:
			queries: List of search queries
			languages: Language preferences for search
			regions: Regional preferences for search
			
		Returns:
			List of enriched search results with full content
		"""
		logger.info(f"🔍 Starting search and download pipeline for {len(queries)} queries")
		
		all_enriched_results = []
		
		# Stage 1: Discovery via multi-engine search
		discovered_urls = await self._discover_urls(queries, languages, regions)
		logger.info(f"🔗 Discovered {len(discovered_urls)} unique URLs from search engines")
		
		if not discovered_urls:
			logger.warning("No URLs discovered, pipeline complete")
			return all_enriched_results
		
		# Stage 2: Download full content via gen_crawler
		if self.config.download_full_content and GEN_CRAWLER_AVAILABLE:
			enriched_results = await self._download_content(discovered_urls)
			all_enriched_results.extend(enriched_results)
		else:
			# Create enriched results without full content
			for search_result, engine, query in discovered_urls:
				enriched_result = EnrichedSearchResult(
					search_result=search_result,
					search_engine=engine,
					search_query=query,
					download_success=False,
					download_error="Content download disabled or gen_crawler unavailable"
				)
				all_enriched_results.append(enriched_result)
		
		# Stage 3: Store in database
		if self.config.enable_database_storage and self.db_manager:
			await self._store_results(all_enriched_results)
		
		logger.info(f"✅ Pipeline complete: {len(all_enriched_results)} enriched results")
		return all_enriched_results
	
	async def _discover_urls(self, 
							 queries: List[str],
							 languages: Optional[List[str]],
							 regions: Optional[List[str]]) -> List[Tuple[SearchResult, str, str]]:
		"""Stage 1: Discover URLs using multi-engine search."""
		all_results = []
		
		if not self.search_crawler:
			logger.error("Search crawler not available")
			return all_results
		
		for query in queries:
			try:
				self.stats['search_queries'] += 1
				logger.info(f"🔎 Searching for: '{query}'")
				
				# Perform multi-engine search
				search_results = await self.search_crawler.search(
					query=query,
					max_results=self.config.total_max_results
				)
				
				# Process search results
				# search_results is a List[EnhancedSearchResult], not a Dict
				for enhanced_result in search_results:
					# Use the first source engine or 'unknown' if no engines listed
					engine_name = enhanced_result.engines_found[0] if enhanced_result.engines_found else 'unknown'
					
					if self._should_include_url(enhanced_result.url):
						# EnhancedSearchResult inherits from SearchResult, so we can use it directly
						all_results.append((enhanced_result, engine_name, query))
						self.stats['urls_discovered'] += 1
				
				logger.info(f"   Found {len(search_results)} results for '{query}'")
				
			except Exception as e:
				logger.error(f"Failed to search for '{query}': {e}")
		
		# Remove duplicates based on URL
		unique_results = self._deduplicate_urls(all_results)
		logger.info(f"📊 After deduplication: {len(unique_results)} unique URLs")
		
		return unique_results
	
	async def _download_content(self, 
								discovered_urls: List[Tuple[SearchResult, str, str]]) -> List[EnrichedSearchResult]:
		"""Stage 2: Download full content using gen_crawler."""
		if not self.gen_crawler:
			logger.error("Gen crawler not available for content download")
			return []
		
		logger.info(f"⬇️  Starting content download for {len(discovered_urls)} URLs")
		
		enriched_results = []
		
		# Process URLs in batches for performance
		for i in range(0, len(discovered_urls), self.config.batch_size):
			batch = discovered_urls[i:i + self.config.batch_size]
			batch_num = i // self.config.batch_size + 1
			total_batches = (len(discovered_urls) + self.config.batch_size - 1) // self.config.batch_size
			
			logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} URLs)")
			
			# Download content for batch
			batch_enriched = await self._download_batch(batch)
			enriched_results.extend(batch_enriched)
			
			# Respectful delay between batches
			if i + self.config.batch_size < len(discovered_urls):
				await asyncio.sleep(self.config.delay_between_batches)
		
		success_count = sum(1 for result in enriched_results if result.download_success)
		logger.info(f"✅ Content download complete: {success_count}/{len(enriched_results)} successful")
		
		return enriched_results
	
	async def _download_batch(self, 
							  batch: List[Tuple[SearchResult, str, str]]) -> List[EnrichedSearchResult]:
		"""Download content for a batch of URLs."""
		tasks = []
		
		for search_result, engine, query in batch:
			task = self._download_single_url(search_result, engine, query)
			tasks.append(task)
		
		# Execute downloads concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		enriched_results = []
		for result in results:
			if isinstance(result, EnrichedSearchResult):
				enriched_results.append(result)
			elif isinstance(result, Exception):
				logger.error(f"Download task failed: {result}")
		
		return enriched_results
	
	async def _download_single_url(self, 
								   search_result: SearchResult,
								   engine: str,
								   query: str) -> EnrichedSearchResult:
		"""Download content for a single URL."""
		enriched_result = EnrichedSearchResult(
			search_result=search_result,
			search_engine=engine,
			search_query=query,
			url_hash=self._generate_url_hash(search_result.url),
			domain=self._extract_domain(search_result.url)
		)
		
		# Skip if URL already processed
		if self.config.skip_duplicate_urls and search_result.url in self.processed_urls:
			enriched_result.download_error = "Duplicate URL skipped"
			self.stats['duplicate_urls_skipped'] += 1
			return enriched_result
		
		try:
			start_time = time.time()
			
			# Use gen_crawler to download full content
			crawl_result = await self._crawl_single_url(search_result.url)
			
			enriched_result.download_time = time.time() - start_time
			
			if crawl_result and crawl_result.success:
				# Extract content from crawl result
				enriched_result.full_content = crawl_result.content
				enriched_result.full_title = crawl_result.title
				enriched_result.content_metadata = crawl_result.metadata
				enriched_result.word_count = crawl_result.word_count
				enriched_result.download_success = True
				
				# Content analysis
				enriched_result.content_quality_score = self._calculate_content_quality(crawl_result)
				
				self.stats['urls_downloaded'] += 1
				
				logger.debug(f"✅ Downloaded content from {search_result.url}")
				
			else:
				enriched_result.download_error = crawl_result.error if crawl_result else "Crawl failed"
				self.stats['download_failures'] += 1
				
		except Exception as e:
			enriched_result.download_error = str(e)
			enriched_result.download_time = time.time() - start_time
			self.stats['download_failures'] += 1
			logger.warning(f"Failed to download {search_result.url}: {e}")
		
		# Track processed URL
		self.processed_urls.add(search_result.url)
		
		return enriched_result
	
	async def _crawl_single_url(self, url: str) -> Optional[GenCrawlResult]:
		"""Use gen_crawler to download content from a single URL."""
		if not self.gen_crawler:
			return None
		
		try:
			# Check if gen_crawler has the right method
			if hasattr(self.gen_crawler, 'crawl_url'):
				return await self.gen_crawler.crawl_url(url)
			elif hasattr(self.gen_crawler, 'crawl_site'):
				# Use crawl_site with max_pages=1
				site_result = await self.gen_crawler.crawl_site(url)
				if site_result and site_result.pages:
					return site_result.pages[0]
			else:
				logger.error("Gen crawler doesn't have expected crawl methods")
				return None
				
		except Exception as e:
			logger.error(f"Gen crawler failed for {url}: {e}")
			return None
	
	async def _store_results(self, enriched_results: List[EnrichedSearchResult]) -> None:
		"""Stage 3: Store results in information_units database."""
		if not self.db_manager:
			logger.warning("Database manager not available, skipping storage")
			return
		
		logger.info(f"💾 Storing {len(enriched_results)} results in database")
		
		for result in enriched_results:
			try:
				# Convert to information_units format
				info_unit_data = self._convert_to_information_unit(result)
				
				# Store in database
				await self.db_manager.store_search_result(info_unit_data)
				self.stats['database_records_created'] += 1
				
			except Exception as e:
				logger.error(f"Failed to store result for {result.search_result.url}: {e}")
		
		logger.info(f"✅ Database storage complete: {self.stats['database_records_created']} records created")
	
	def _convert_to_information_unit(self, result: EnrichedSearchResult) -> Dict[str, Any]:
		"""Convert enriched search result to information_units format."""
		return {
			'title': result.full_title or result.search_result.title,
			'content': result.full_content or result.search_result.snippet,
			'content_url': result.search_result.url,
			'summary': result.search_result.snippet,
			'source_name': result.domain,
			'source_domain': result.domain,
			'published_at': None,  # Search results don't have publication dates
			'discovered_at': result.enriched_at,
			'language_code': 'en',  # Default, could be enhanced with detection
			'content_hash': result.url_hash,
			'url_hash': result.url_hash,
			'extraction_confidence': result.content_quality_score,
			'credibility_score': self._calculate_source_credibility(result),
			'search_metadata': {
				'search_engine': result.search_engine,
				'search_query': result.search_query,
				'search_rank': getattr(result.search_result, 'rank', 0),
				'download_success': result.download_success,
				'download_time': result.download_time,
				'word_count': result.word_count
			}
		}
	
	def _should_include_url(self, url: str) -> bool:
		"""Determine if a URL should be included in processing."""
		try:
			parsed_url = urlparse(url)
			domain = parsed_url.netloc.lower()
			
			# Check blocked domains
			if any(blocked in domain for blocked in self.config.blocked_domains):
				return False
			
			# Check allowed domains (if specified)
			if self.config.allowed_domains:
				if not any(allowed in domain for allowed in self.config.allowed_domains):
					return False
			
			# Check for common non-content URLs
			path = parsed_url.path.lower()
			if any(pattern in path for pattern in ['/search', '/login', '/register', '/cart']):
				return False
			
			return True
			
		except Exception:
			return False
	
	def _deduplicate_urls(self, 
						  results: List[Tuple[SearchResult, str, str]]) -> List[Tuple[SearchResult, str, str]]:
		"""Remove duplicate URLs from search results."""
		seen_urls = set()
		unique_results = []
		
		for search_result, engine, query in results:
			if search_result.url not in seen_urls:
				seen_urls.add(search_result.url)
				unique_results.append((search_result, engine, query))
		
		return unique_results
	
	def _generate_url_hash(self, url: str) -> str:
		"""Generate hash for URL."""
		return hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
	
	def _extract_domain(self, url: str) -> str:
		"""Extract domain from URL."""
		try:
			return urlparse(url).netloc
		except:
			return ""
	
	def _calculate_content_quality(self, crawl_result: GenCrawlResult) -> float:
		"""Calculate content quality score."""
		score = 0.0
		
		# Length factor
		if crawl_result.word_count > 200:
			score += 0.3
		if crawl_result.word_count > 1000:
			score += 0.2
		
		# Title presence
		if crawl_result.title and len(crawl_result.title) > 10:
			score += 0.2
		
		# Metadata richness
		if crawl_result.metadata:
			score += 0.1 * min(len(crawl_result.metadata), 3) / 3
		
		# Success factor
		if crawl_result.success:
			score += 0.2
		
		return min(score, 1.0)
	
	def _calculate_source_credibility(self, result: EnrichedSearchResult) -> float:
		"""Calculate source credibility score."""
		score = 0.5  # Base score
		
		# Domain-based credibility
		domain = result.domain.lower()
		if any(trusted in domain for trusted in ['.gov', '.edu', '.org']):
			score += 0.3
		elif any(trusted in domain for trusted in ['.com', '.net']):
			score += 0.1
		
		# Search engine ranking boost
		engine_boost = {
			'google': 0.2,
			'bing': 0.15,
			'duckduckgo': 0.1,
			'yandex': 0.1  # Re-enabled with fixed parsing
		}
		score += engine_boost.get(result.search_engine, 0.05)
		
		return min(score, 1.0)
	
	async def get_pipeline_stats(self) -> Dict[str, Any]:
		"""Get comprehensive pipeline statistics."""
		runtime = None
		if self.stats['pipeline_start_time']:
			runtime = (datetime.now(timezone.utc) - self.stats['pipeline_start_time']).total_seconds()
		
		return {
			'pipeline_stats': self.stats.copy(),
			'runtime_seconds': runtime,
			'search_crawler_available': SEARCH_CRAWLER_AVAILABLE,
			'gen_crawler_available': GEN_CRAWLER_AVAILABLE,
			'database_available': DATABASE_AVAILABLE,
			'processed_urls_count': len(self.processed_urls),
			'performance_metrics': {
				'urls_per_query': self.stats['urls_discovered'] / max(1, self.stats['search_queries']),
				'download_success_rate': (self.stats['urls_downloaded'] / max(1, self.stats['urls_discovered'])) * 100,
				'processing_rate': self.stats['urls_discovered'] / max(1, runtime or 1),
				'storage_success_rate': (self.stats['database_records_created'] / max(1, self.stats['urls_downloaded'])) * 100
			}
		}
	
	async def close(self) -> None:
		"""Clean up pipeline resources."""
		logger.info("🔌 Closing search content pipeline...")
		
		if self.gen_crawler and hasattr(self.gen_crawler, 'cleanup'):
			await self.gen_crawler.cleanup()
		
		if self.db_manager:
			await self.db_manager.close()
		
		logger.info("✅ Search content pipeline closed")

# Factory functions

def create_search_content_pipeline(
	search_engines: Optional[List[str]] = None,
	database_url: str = "postgresql:///lnd",
	enable_content_download: bool = True,
	enable_database_storage: bool = True,
	**config_kwargs
) -> SearchContentPipeline:
	"""
	Factory function to create a complete search content pipeline.
	
	Args:
		search_engines: List of search engines to use
		database_url: Database connection string
		enable_content_download: Enable full content download via gen_crawler
		enable_database_storage: Enable database storage
		**config_kwargs: Additional configuration parameters
		
	Returns:
		SearchContentPipeline: Configured pipeline
	"""
	config = SearchContentPipelineConfig(
		search_engines=search_engines or ['google', 'bing', 'duckduckgo', 'yandex'],
		database_url=database_url,
		download_full_content=enable_content_download and GEN_CRAWLER_AVAILABLE,
		enable_database_storage=enable_database_storage and DATABASE_AVAILABLE,
		**config_kwargs
	)
	
	return SearchContentPipeline(config)

def create_conflict_search_content_pipeline(
	database_url: str = "postgresql:///lnd"
) -> SearchContentPipeline:
	"""
	Create search content pipeline optimized for conflict monitoring.
	
	Args:
		database_url: Database connection string
		
	Returns:
		SearchContentPipeline: Conflict monitoring optimized pipeline
	"""
	config = SearchContentPipelineConfig(
		search_engines=['google', 'bing', 'duckduckgo', 'yandex', 'brave'],
		max_results_per_engine=25,
		total_max_results=100,
		
		# Content settings for news/conflict content
		download_full_content=True,
		enable_database_storage=True,
		database_url=database_url,
		
		# Conflict-specific filtering
		blocked_domains=[
			'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
			'linkedin.com', 'pinterest.com', 'reddit.com', 'tiktok.com'
		],
		min_content_length=300,  # Longer content for substantial articles
		
		# Performance for conflict monitoring
		batch_size=5,  # Conservative for news sites
		delay_between_batches=3.0,  # Respectful crawling
		gen_crawler_concurrent_downloads=3
	)
	
	return SearchContentPipeline(config)

def create_horn_africa_search_pipeline(
	database_url: str = "postgresql:///lnd"
) -> SearchContentPipeline:
	"""
	Create search pipeline optimized for Horn of Africa content discovery.
	
	Args:
		database_url: Database connection string
		
	Returns:
		SearchContentPipeline: Horn of Africa optimized pipeline
	"""
	# Regional news sources and domains
	horn_africa_domains = [
		'nation.co.ke', 'standardmedia.co.ke', 'theeastafrican.co.ke',  # Kenya
		'addisstandard.com', 'fanabc.com', 'ethiopianreporter.com',    # Ethiopia
		'hiiraan.com', 'somaliguardian.com', 'raxanreeb.com',          # Somalia
		'sudantribune.com', 'sudandaily.net', 'dabangasudan.org',      # Sudan
		'eyeradio.org', 'paanluelwel.com'                              # South Sudan
	]
	
	config = SearchContentPipelineConfig(
		search_engines=['google', 'bing', 'duckduckgo', 'yandex'],
		max_results_per_engine=30,
		total_max_results=120,
		
		# Regional content focus
		allowed_domains=horn_africa_domains,
		download_full_content=True,
		enable_database_storage=True,
		database_url=database_url,
		
		# Regional content settings
		min_content_length=250,
		batch_size=6,
		delay_between_batches=2.5,
		gen_crawler_concurrent_downloads=4
	)
	
	return SearchContentPipeline(config)