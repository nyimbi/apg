"""
Google News Content Pipeline Integration
=======================================

Two-stage pipeline that combines Google News discovery with gen_crawler content downloading:

1. **Discovery Stage**: Google News crawler finds news items (metadata, URLs, summaries)
2. **Download Stage**: Gen crawler downloads full article content from discovered URLs

This creates a complete news monitoring pipeline that discovers relevant articles
and then retrieves their full content for analysis and storage.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse
import time

# Import Google News crawler components
from .enhanced_client import EnhancedGoogleNewsClient, EnhancedGoogleNewsConfig
from .database import GoogleNewsRecord, InformationUnitsManager

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

logger = logging.getLogger(__name__)

@dataclass
class ContentPipelineConfig:
	"""Configuration for the content pipeline."""
	# Google News discovery settings
	google_news_config: EnhancedGoogleNewsConfig = field(default_factory=EnhancedGoogleNewsConfig)
	
	# Gen crawler download settings
	gen_crawler_max_pages: int = 1  # Usually just download the article page
	gen_crawler_timeout: int = 30
	gen_crawler_concurrent_downloads: int = 5
	
	# Pipeline settings
	max_articles_per_query: int = 50
	download_full_content: bool = True
	skip_duplicate_urls: bool = True
	enable_content_analysis: bool = True
	
	# Performance settings
	batch_size: int = 10  # Process URLs in batches
	delay_between_batches: float = 2.0  # Respectful crawling
	max_retries: int = 2

@dataclass
class EnrichedNewsItem:
	"""News item enriched with full content from gen_crawler."""
	# Original Google News metadata
	google_news_record: GoogleNewsRecord
	
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
	
	# Processing metadata
	enriched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	pipeline_id: str = field(default_factory=lambda: f"pipeline_{int(time.time())}")

class GoogleNewsContentPipeline:
	"""
	Complete news content pipeline that discovers articles via Google News
	and downloads full content via gen_crawler.
	"""
	
	def __init__(self, config: Optional[ContentPipelineConfig] = None):
		"""Initialize the content pipeline."""
		self.config = config or ContentPipelineConfig()
		
		# Initialize Google News client
		self.google_news_client = EnhancedGoogleNewsClient(self.config.google_news_config)
		
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
						'min_content_length': 100,
						'max_content_length': 1000000,
						'exclude_extensions': ['.pdf', '.doc', '.xls', '.zip'],
						'exclude_patterns': ['login', 'logout', 'register', 'cart']
					},
					'stealth': {
						'enable_stealth': True,
						'user_agent': 'GoogleNewsContentPipeline/1.0 (+https://datacraft.co.ke)',
						'respect_robots_txt': True
					}
				}
				
				self.gen_crawler = create_gen_crawler(gen_config) if create_gen_crawler else None
				
				if self.gen_crawler:
					logger.info("✅ Gen crawler configured successfully for content download")
				else:
					logger.warning("⚠️  Gen crawler creation failed")
				
			except Exception as e:
				logger.warning(f"Failed to configure gen_crawler: {e}")
				self.gen_crawler = None
		
		# Tracking
		self.processed_urls: Set[str] = set()
		self.stats = {
			'discovery_queries': 0,
			'articles_discovered': 0,
			'articles_downloaded': 0,
			'download_failures': 0,
			'duplicate_urls_skipped': 0,
			'pipeline_start_time': None
		}
		
		logger.info(f"Content pipeline initialized (gen_crawler_available: {GEN_CRAWLER_AVAILABLE})")
	
	async def initialize(self) -> None:
		"""Initialize all pipeline components."""
		logger.info("🚀 Initializing content pipeline...")
		
		# Initialize Google News client
		await self.google_news_client.initialize()
		
		# Gen crawler is initialized on-demand
		self.stats['pipeline_start_time'] = datetime.now(timezone.utc)
		
		logger.info("✅ Content pipeline initialization complete")
	
	async def discover_and_download(self, 
									queries: List[str],
									language: str = 'en',
									country: str = 'KE') -> List[EnrichedNewsItem]:
		"""
		Main pipeline method: discover articles via Google News and download full content.
		
		Args:
			queries: List of search queries for Google News
			language: Language code for search
			country: Country code for search
			
		Returns:
			List of enriched news items with full content
		"""
		logger.info(f"🔍 Starting discovery and download pipeline for {len(queries)} queries")
		
		all_enriched_items = []
		
		# Stage 1: Discovery via Google News
		discovered_articles = await self._discover_articles(queries, language, country)
		logger.info(f"📰 Discovered {len(discovered_articles)} articles from Google News")
		
		if not discovered_articles:
			logger.warning("No articles discovered, pipeline complete")
			return all_enriched_items
		
		# Stage 2: Download full content via gen_crawler
		if self.config.download_full_content and GEN_CRAWLER_AVAILABLE:
			enriched_items = await self._download_content(discovered_articles)
			all_enriched_items.extend(enriched_items)
		else:
			# Create enriched items without full content
			for article in discovered_articles:
				enriched_item = EnrichedNewsItem(
					google_news_record=article,
					download_success=False,
					download_error="Content download disabled or gen_crawler unavailable"
				)
				all_enriched_items.append(enriched_item)
		
		logger.info(f"✅ Pipeline complete: {len(all_enriched_items)} enriched articles")
		return all_enriched_items
	
	async def _discover_articles(self, 
								 queries: List[str],
								 language: str,
								 country: str) -> List[GoogleNewsRecord]:
		"""Stage 1: Discover articles using Google News."""
		all_articles = []
		
		for query in queries:
			try:
				self.stats['discovery_queries'] += 1
				logger.info(f"🔎 Searching Google News for: '{query}'")
				
				# Search Google News
				articles = await self.google_news_client.search_news(
					query=query,
					language=language,
					country=country,
					max_results=self.config.max_articles_per_query
				)
				
				# Convert to GoogleNewsRecord objects if needed
				google_news_records = []
				for article in articles:
					if isinstance(article, dict):
						# Convert dict to GoogleNewsRecord
						record = self._dict_to_google_news_record(article, query)
						google_news_records.append(record)
					else:
						google_news_records.append(article)
				
				all_articles.extend(google_news_records)
				self.stats['articles_discovered'] += len(google_news_records)
				
				logger.info(f"   Found {len(google_news_records)} articles for '{query}'")
				
			except Exception as e:
				logger.error(f"Failed to search for '{query}': {e}")
		
		# Remove duplicates based on URL
		unique_articles = self._deduplicate_articles(all_articles)
		logger.info(f"📊 After deduplication: {len(unique_articles)} unique articles")
		
		return unique_articles
	
	async def _download_content(self, articles: List[GoogleNewsRecord]) -> List[EnrichedNewsItem]:
		"""Stage 2: Download full content using gen_crawler."""
		if not self.gen_crawler:
			logger.error("Gen crawler not available for content download")
			return []
		
		logger.info(f"⬇️  Starting content download for {len(articles)} articles")
		
		enriched_items = []
		
		# Process articles in batches for performance
		for i in range(0, len(articles), self.config.batch_size):
			batch = articles[i:i + self.config.batch_size]
			batch_num = i // self.config.batch_size + 1
			total_batches = (len(articles) + self.config.batch_size - 1) // self.config.batch_size
			
			logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")
			
			# Download content for batch
			batch_enriched = await self._download_batch(batch)
			enriched_items.extend(batch_enriched)
			
			# Respectful delay between batches
			if i + self.config.batch_size < len(articles):
				await asyncio.sleep(self.config.delay_between_batches)
		
		success_count = sum(1 for item in enriched_items if item.download_success)
		logger.info(f"✅ Content download complete: {success_count}/{len(enriched_items)} successful")
		
		return enriched_items
	
	async def _download_batch(self, articles: List[GoogleNewsRecord]) -> List[EnrichedNewsItem]:
		"""Download content for a batch of articles."""
		tasks = []
		
		for article in articles:
			task = self._download_single_article(article)
			tasks.append(task)
		
		# Execute downloads concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		enriched_items = []
		for result in results:
			if isinstance(result, EnrichedNewsItem):
				enriched_items.append(result)
			elif isinstance(result, Exception):
				logger.error(f"Download task failed: {result}")
		
		return enriched_items
	
	async def _download_single_article(self, article: GoogleNewsRecord) -> EnrichedNewsItem:
		"""Download content for a single article."""
		enriched_item = EnrichedNewsItem(google_news_record=article)
		
		# Skip if URL already processed
		if self.config.skip_duplicate_urls and article.content_url in self.processed_urls:
			enriched_item.download_error = "Duplicate URL skipped"
			self.stats['duplicate_urls_skipped'] += 1
			return enriched_item
		
		try:
			start_time = time.time()
			
			# Use gen_crawler to download full content
			crawl_result = await self._crawl_single_url(article.content_url)
			
			enriched_item.download_time = time.time() - start_time
			
			if crawl_result and crawl_result.success:
				# Extract content from crawl result
				enriched_item.full_content = crawl_result.content
				enriched_item.full_title = crawl_result.title
				enriched_item.content_metadata = crawl_result.metadata
				enriched_item.word_count = crawl_result.word_count
				enriched_item.download_success = True
				
				# Simple content analysis
				enriched_item.content_quality_score = self._calculate_content_quality(crawl_result)
				
				self.stats['articles_downloaded'] += 1
				
				logger.debug(f"✅ Downloaded content from {article.content_url}")
				
			else:
				enriched_item.download_error = crawl_result.error if crawl_result else "Crawl failed"
				self.stats['download_failures'] += 1
				
		except Exception as e:
			enriched_item.download_error = str(e)
			enriched_item.download_time = time.time() - start_time
			self.stats['download_failures'] += 1
			logger.warning(f"Failed to download {article.content_url}: {e}")
		
		# Track processed URL
		self.processed_urls.add(article.content_url)
		
		return enriched_item
	
	async def _crawl_single_url(self, url: str) -> Optional[GenCrawlResult]:
		"""Use gen_crawler to download content from a single URL."""
		if not self.gen_crawler:
			return None
		
		try:
			# Create a simple crawl request for just this URL
			# Note: This assumes gen_crawler has a method to crawl a single URL
			# If not, we might need to adapt the interface
			
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
	
	def _dict_to_google_news_record(self, article_dict: Dict[str, Any], query: str) -> GoogleNewsRecord:
		"""Convert article dictionary to GoogleNewsRecord."""
		# Parse published date if available
		published_at = None
		if article_dict.get('published_at'):
			try:
				published_at = datetime.fromisoformat(article_dict['published_at'].replace('Z', '+00:00'))
			except:
				pass
		
		return GoogleNewsRecord(
			title=article_dict.get('title', ''),
			content=article_dict.get('content', article_dict.get('summary', '')),
			content_url=article_dict.get('url', ''),
			summary=article_dict.get('summary', ''),
			source_name=article_dict.get('source_name', ''),
			source_domain=self._extract_domain(article_dict.get('url', '')),
			published_at=published_at,
			discovered_at=datetime.now(timezone.utc),
			language_code=article_dict.get('language_code', 'en'),
			google_news_metadata={
				'search_query': query,
				'original_data': article_dict.get('metadata', {})
			}
		)
	
	def _extract_domain(self, url: str) -> str:
		"""Extract domain from URL."""
		try:
			return urlparse(url).netloc
		except:
			return ""
	
	def _deduplicate_articles(self, articles: List[GoogleNewsRecord]) -> List[GoogleNewsRecord]:
		"""Remove duplicate articles based on URL."""
		seen_urls = set()
		unique_articles = []
		
		for article in articles:
			if article.content_url not in seen_urls:
				seen_urls.add(article.content_url)
				unique_articles.append(article)
		
		return unique_articles
	
	def _calculate_content_quality(self, crawl_result: GenCrawlResult) -> float:
		"""Calculate a simple content quality score."""
		score = 0.0
		
		# Length factor (longer is generally better, up to a point)
		if crawl_result.word_count > 100:
			score += 0.3
		if crawl_result.word_count > 500:
			score += 0.2
		
		# Title presence
		if crawl_result.title:
			score += 0.2
		
		# Metadata richness
		if crawl_result.metadata:
			score += 0.1 * min(len(crawl_result.metadata), 3) / 3
		
		# Success factor
		if crawl_result.success:
			score += 0.2
		
		return min(score, 1.0)
	
	async def get_pipeline_stats(self) -> Dict[str, Any]:
		"""Get comprehensive pipeline statistics."""
		runtime = None
		if self.stats['pipeline_start_time']:
			runtime = (datetime.now(timezone.utc) - self.stats['pipeline_start_time']).total_seconds()
		
		google_news_stats = await self.google_news_client.get_stats() if self.google_news_client else {}
		
		return {
			'pipeline_stats': self.stats.copy(),
			'runtime_seconds': runtime,
			'google_news_stats': google_news_stats,
			'gen_crawler_available': GEN_CRAWLER_AVAILABLE,
			'processed_urls_count': len(self.processed_urls),
			'performance_metrics': {
				'articles_per_query': self.stats['articles_discovered'] / max(1, self.stats['discovery_queries']),
				'download_success_rate': (self.stats['articles_downloaded'] / max(1, self.stats['articles_discovered'])) * 100,
				'processing_rate': self.stats['articles_discovered'] / max(1, runtime or 1)
			}
		}
	
	async def close(self) -> None:
		"""Clean up pipeline resources."""
		logger.info("🔌 Closing content pipeline...")
		
		if self.google_news_client:
			await self.google_news_client.close()
		
		if self.gen_crawler and hasattr(self.gen_crawler, 'cleanup'):
			await self.gen_crawler.cleanup()
		
		logger.info("✅ Content pipeline closed")

# Factory functions

def create_content_pipeline(
	database_url: str = "postgresql:///lnd",
	enable_google_news_features: bool = True,
	enable_content_download: bool = True,
	**config_kwargs
) -> GoogleNewsContentPipeline:
	"""
	Factory function to create a complete content pipeline.
	
	Args:
		database_url: Database connection string
		enable_google_news_features: Enable enhanced Google News features
		enable_content_download: Enable full content download via gen_crawler
		**config_kwargs: Additional configuration parameters
		
	Returns:
		GoogleNewsContentPipeline: Configured pipeline
	"""
	# Configure Google News client
	google_news_config = EnhancedGoogleNewsConfig(
		database_url=database_url,
		enable_database_storage=enable_google_news_features,
		enable_rate_limiting=enable_google_news_features,
		enable_circuit_breaker=enable_google_news_features,
		**{k: v for k, v in config_kwargs.items() if k.startswith('google_news_')}
	)
	
	# Configure pipeline
	pipeline_config = ContentPipelineConfig(
		google_news_config=google_news_config,
		download_full_content=enable_content_download and GEN_CRAWLER_AVAILABLE,
		**{k: v for k, v in config_kwargs.items() if not k.startswith('google_news_')}
	)
	
	return GoogleNewsContentPipeline(pipeline_config)

def create_horn_africa_content_pipeline(
	database_url: str = "postgresql:///lnd"
) -> GoogleNewsContentPipeline:
	"""
	Create content pipeline optimized for Horn of Africa news monitoring.
	
	Args:
		database_url: Database connection string
		
	Returns:
		GoogleNewsContentPipeline: Horn of Africa optimized pipeline
	"""
	google_news_config = EnhancedGoogleNewsConfig(
		database_url=database_url,
		enable_database_storage=True,
		enable_rate_limiting=True,
		enable_circuit_breaker=True,
		
		# Horn of Africa specific settings
		target_countries=['KE', 'ET', 'SO', 'SD', 'SS', 'UG', 'TZ', 'ER', 'DJ'],
		allowed_languages=['en', 'fr', 'ar', 'sw', 'am'],
		requests_per_second=3.0,  # Conservative for regional focus
		burst_capacity=30
	)
	
	pipeline_config = ContentPipelineConfig(
		google_news_config=google_news_config,
		download_full_content=True,
		max_articles_per_query=25,  # Focused approach
		batch_size=5,  # Conservative batching
		delay_between_batches=3.0  # Respectful crawling
	)
	
	return GoogleNewsContentPipeline(pipeline_config)