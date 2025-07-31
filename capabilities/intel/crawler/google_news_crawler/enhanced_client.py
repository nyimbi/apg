"""
Enhanced Google News Client
===========================

Production-ready Google News client integrating all enhanced components:
- Information units database integration
- Token bucket rate limiting  
- Circuit breaker pattern
- Comprehensive error handling
- Content deduplication

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

try:
	import aiohttp
	import feedparser
	from bs4 import BeautifulSoup
except ImportError as e:
	raise ImportError(f"Required dependencies not available: {e}")

from .database import (
	InformationUnitsManager,
	GoogleNewsRecord,
	create_information_units_manager,
	map_google_news_to_information_units,
	TokenBucketRateLimiter,
	RateLimitConfig,
	create_rate_limiter
)

from .resilience import (
	CircuitBreaker,
	create_circuit_breaker,
	create_google_news_circuit_breaker,
	ErrorHandler,
	ErrorCategory,
	create_error_handler
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedGoogleNewsConfig:
	"""Configuration for enhanced Google News client."""
	# Database configuration
	database_url: str = "postgresql:///lnd"
	enable_database_storage: bool = True
	
	# Rate limiting configuration
	enable_rate_limiting: bool = True
	requests_per_second: float = 5.0
	burst_capacity: int = 50
	
	# Circuit breaker configuration
	enable_circuit_breaker: bool = True
	failure_threshold: int = 3
	recovery_timeout: float = 30.0
	
	# Content filtering
	min_content_length: int = 100
	max_content_length: int = 50000
	allowed_languages: List[str] = field(default_factory=lambda: ['en', 'fr', 'ar', 'sw'])
	
	# Geographic focus (Horn of Africa by default)
	target_countries: List[str] = field(default_factory=lambda: [
		'Kenya', 'Ethiopia', 'Somalia', 'Sudan', 'South Sudan', 
		'Uganda', 'Tanzania', 'Eritrea', 'Djibouti'
	])
	
	# HTTP configuration
	request_timeout: float = 30.0
	max_concurrent_requests: int = 10
	user_agent: str = "Enhanced-GoogleNews-Crawler/1.0"

class EnhancedGoogleNewsClient:
	"""
	Production-ready Google News client with comprehensive error handling,
	rate limiting, circuit breaker, and database integration.
	"""
	
	def __init__(self, config: Optional[EnhancedGoogleNewsConfig] = None):
		"""Initialize enhanced Google News client."""
		self.config = config or EnhancedGoogleNewsConfig()
		
		# Core components
		self.db_manager: Optional[InformationUnitsManager] = None
		self.rate_limiter: Optional[TokenBucketRateLimiter] = None
		self.circuit_breaker: Optional[CircuitBreaker] = None
		self.error_handler = create_error_handler()
		
		# HTTP session
		self._http_session: Optional[aiohttp.ClientSession] = None
		
		# Google News endpoints
		self.base_urls = {
			'rss_search': 'https://news.google.com/rss/search',
			'rss_headlines': 'https://news.google.com/rss/headlines',
			'rss_topics': 'https://news.google.com/rss/topics'
		}
		
		# Performance tracking
		self.stats = {
			'searches_performed': 0,
			'articles_discovered': 0,
			'articles_stored': 0,
			'articles_skipped': 0,
			'errors_handled': 0,
			'start_time': datetime.now(timezone.utc)
		}
		
		logger.info("Enhanced Google News client initialized")
	
	async def initialize(self) -> None:
		"""Initialize all client components."""
		logger.info("🚀 Initializing Enhanced Google News Client...")
		
		# Initialize database manager
		if self.config.enable_database_storage:
			self.db_manager = create_information_units_manager(self.config.database_url)
			await self.db_manager.initialize()
			logger.info("✅ Database manager initialized")
		
		# Initialize rate limiter
		if self.config.enable_rate_limiting:
			rate_config = RateLimitConfig(
				capacity=self.config.burst_capacity,
				refill_rate=self.config.requests_per_second,
				enable_adaptive=True,
				burst_protection=True
			)
			self.rate_limiter = TokenBucketRateLimiter(rate_config)
			logger.info("✅ Rate limiter initialized")
		
		# Initialize circuit breaker
		if self.config.enable_circuit_breaker:
			self.circuit_breaker = create_google_news_circuit_breaker()
			logger.info("✅ Circuit breaker initialized")
		
		# Initialize HTTP session
		timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
		connector = aiohttp.TCPConnector(
			limit=self.config.max_concurrent_requests,
			limit_per_host=5,
			ttl_dns_cache=300,
			use_dns_cache=True
		)
		
		self._http_session = aiohttp.ClientSession(
			timeout=timeout,
			connector=connector,
			headers={'User-Agent': self.config.user_agent}
		)
		
		logger.info("✅ Enhanced Google News client fully initialized")
	
	async def search_news(self, 
						 query: str,
						 language: str = 'en',
						 country: str = 'KE',
						 max_results: int = 100) -> List[Dict[str, Any]]:
		"""
		Search Google News with enhanced error handling and storage.
		
		Args:
			query: Search query
			language: Language code (en, fr, ar, sw, etc.)
			country: Country code (KE, ET, SO, etc.)
			max_results: Maximum number of results to return
			
		Returns:
			List[Dict]: List of news articles with metadata
		"""
		if not self._http_session:
			await self.initialize()
		
		operation = f"search_news:{query}"
		articles = []
		
		try:
			# Apply rate limiting
			if self.rate_limiter:
				if not await self.rate_limiter.acquire():
					logger.warning("Rate limit exceeded, waiting...")
					if not await self.rate_limiter.wait_if_needed():
						raise Exception("Rate limit wait timeout exceeded")
			
			# Execute search with circuit breaker protection
			if self.circuit_breaker:
				search_results = await self.circuit_breaker.call(
					self._perform_google_news_search,
					query, language, country, max_results
				)
			else:
				search_results = await self._perform_google_news_search(
					query, language, country, max_results
				)
			
			# Process and store results
			for article_data in search_results:
				try:
					# Create GoogleNewsRecord
					record = self._create_google_news_record(article_data, query)
					
					# Apply content filters
					if not self._should_include_article(record):
						self.stats['articles_skipped'] += 1
						continue
					
					# Store in database if enabled
					if self.db_manager:
						await self.db_manager.store_google_news_record(record)
						self.stats['articles_stored'] += 1
					
					# Add to results
					articles.append(self._record_to_dict(record))
					self.stats['articles_discovered'] += 1
					
				except Exception as e:
					await self.error_handler.handle_error(
						e, operation="process_article", 
						source_url=article_data.get('url')
					)
					self.stats['errors_handled'] += 1
			
			self.stats['searches_performed'] += 1
			logger.info(f"✅ Search completed: {len(articles)} articles from '{query}'")
			
			return articles
			
		except Exception as e:
			error_context = await self.error_handler.handle_error(
				e, operation=operation
			)
			self.stats['errors_handled'] += 1
			
			# Re-raise if it's a critical error
			if error_context.category in [ErrorCategory.AUTH_INVALID, ErrorCategory.CONFIG_INVALID]:
				raise
			
			logger.warning(f"Search failed for '{query}': {e}")
			return []
	
	async def _perform_google_news_search(self,
										 query: str,
										 language: str,
										 country: str, 
										 max_results: int) -> List[Dict[str, Any]]:
		"""Perform the actual Google News search."""
		# Construct RSS URL
		url = f"{self.base_urls['rss_search']}?q={query}&hl={language}&gl={country}&ceid={country}:{language}"
		
		# Make HTTP request
		async with self._http_session.get(url) as response:
			if response.status != 200:
				raise aiohttp.ClientResponseError(
					request_info=response.request_info,
					history=response.history,
					status=response.status,
					message=f"HTTP {response.status}"
				)
			
			content = await response.text()
		
		# Parse RSS feed
		feed = feedparser.parse(content)
		
		if not feed.entries:
			logger.warning(f"No results found for query: {query}")
			return []
		
		# Convert feed entries to standardized format
		results = []
		for entry in feed.entries[:max_results]:
			article_data = {
				'title': entry.get('title', ''),
				'url': entry.get('link', ''),
				'summary': entry.get('summary', ''),
				'published': entry.get('published', ''),
				'source': entry.get('source', {}).get('title', ''),
				'google_news_data': {
					'entry_id': entry.get('id', ''),
					'tags': [tag.get('term', '') for tag in entry.get('tags', [])],
					'language': language,
					'country': country,
					'search_query': query
				}
			}
			results.append(article_data)
		
		return results
	
	def _create_google_news_record(self, article_data: Dict[str, Any], query: str) -> GoogleNewsRecord:
		"""Create GoogleNewsRecord from article data."""
		# Parse published date
		published_at = None
		if article_data.get('published'):
			try:
				import dateutil.parser
				published_at = dateutil.parser.parse(article_data['published'])
			except:
				pass
		
		# Extract source domain from URL
		source_domain = ""
		if article_data.get('url'):
			try:
				from urllib.parse import urlparse
				parsed_url = urlparse(article_data['url'])
				source_domain = parsed_url.netloc
			except:
				pass
		
		return GoogleNewsRecord(
			title=article_data.get('title', ''),
			content=article_data.get('summary', ''),  # RSS only provides summary
			content_url=article_data.get('url', ''),
			summary=article_data.get('summary', ''),
			source_name=article_data.get('source', ''),
			source_domain=source_domain,
			published_at=published_at,
			discovered_at=datetime.now(timezone.utc),
			language_code=article_data.get('google_news_data', {}).get('language', 'en'),
			google_news_metadata={
				'search_query': query,
				'country_code': article_data.get('google_news_data', {}).get('country', ''),
				'entry_id': article_data.get('google_news_data', {}).get('entry_id', ''),
				'tags': article_data.get('google_news_data', {}).get('tags', [])
			},
			extraction_confidence=0.8,  # RSS feeds are generally reliable
			credibility_score=0.7       # Default credibility for Google News sources
		)
	
	def _should_include_article(self, record: GoogleNewsRecord) -> bool:
		"""Apply content filters to determine if article should be included."""
		# Check content length
		content_length = len(record.content) if record.content else 0
		if content_length < self.config.min_content_length:
			return False
		if content_length > self.config.max_content_length:
			return False
		
		# Check language
		if (record.language_code and 
			record.language_code not in self.config.allowed_languages):
			return False
		
		# Check required fields
		if not record.title or not record.content_url:
			return False
		
		return True
	
	def _record_to_dict(self, record: GoogleNewsRecord) -> Dict[str, Any]:
		"""Convert GoogleNewsRecord to dictionary format."""
		return {
			'title': record.title,
			'url': record.content_url,
			'content': record.content,
			'summary': record.summary,
			'source_name': record.source_name,
			'source_domain': record.source_domain,
			'published_at': record.published_at.isoformat() if record.published_at else None,
			'discovered_at': record.discovered_at.isoformat() if record.discovered_at else None,
			'language_code': record.language_code,
			'credibility_score': record.credibility_score,
			'extraction_confidence': record.extraction_confidence,
			'metadata': record.google_news_metadata
		}
	
	async def get_headlines(self, country: str = 'KE', language: str = 'en') -> List[Dict[str, Any]]:
		"""Get top headlines for a country."""
		return await self.search_news(
			query="",  # Empty query for headlines
			language=language,
			country=country,
			max_results=50
		)
	
	async def search_by_topic(self, topic: str, **kwargs) -> List[Dict[str, Any]]:
		"""Search news by topic (business, technology, sports, etc.)."""
		# Map topic to appropriate search query
		topic_queries = {
			'business': 'business economy finance',
			'technology': 'technology tech innovation',
			'sports': 'sports football basketball',
			'health': 'health medicine healthcare',
			'politics': 'politics government election',
			'conflict': 'conflict war violence security',
			'horn_africa': 'Horn of Africa Kenya Ethiopia Somalia Sudan'
		}
		
		query = topic_queries.get(topic.lower(), topic)
		return await self.search_news(query, **kwargs)
	
	async def monitor_keywords(self, keywords: List[str], **kwargs) -> Dict[str, List[Dict[str, Any]]]:
		"""Monitor multiple keywords simultaneously."""
		results = {}
		
		# Use semaphore to limit concurrent searches
		semaphore = asyncio.Semaphore(3)  # Max 3 concurrent searches
		
		async def search_keyword(keyword: str):
			async with semaphore:
				try:
					articles = await self.search_news(keyword, **kwargs)
					return keyword, articles
				except Exception as e:
					await self.error_handler.handle_error(
						e, operation=f"monitor_keyword:{keyword}"
					)
					return keyword, []
		
		# Execute searches concurrently
		tasks = [search_keyword(keyword) for keyword in keywords]
		search_results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Process results
		for result in search_results:
			if isinstance(result, tuple):
				keyword, articles = result
				results[keyword] = articles
			else:
				logger.error(f"Keyword monitoring failed: {result}")
		
		return results
	
	async def get_stats(self) -> Dict[str, Any]:
		"""Get comprehensive client statistics."""
		uptime = datetime.now(timezone.utc) - self.stats['start_time']
		
		stats = {
			'uptime_seconds': uptime.total_seconds(),
			'performance': self.stats.copy(),
			'rates': {
				'searches_per_hour': (self.stats['searches_performed'] / max(1, uptime.total_seconds())) * 3600,
				'articles_per_search': self.stats['articles_discovered'] / max(1, self.stats['searches_performed']),
				'storage_success_rate': (self.stats['articles_stored'] / max(1, self.stats['articles_discovered'])) * 100,
				'error_rate': (self.stats['errors_handled'] / max(1, self.stats['searches_performed'])) * 100
			}
		}
		
		# Add component stats if available
		if self.rate_limiter:
			stats['rate_limiter'] = self.rate_limiter.get_stats()
		
		if self.circuit_breaker:
			stats['circuit_breaker'] = self.circuit_breaker.get_metrics()
		
		if self.error_handler:
			stats['error_handler'] = self.error_handler.get_stats()
		
		if self.db_manager:
			stats['database'] = {
				'recent_articles_count': len(await self.db_manager.get_recent_articles(100)),
				'duplicate_stats': await self.db_manager.get_duplicate_count()
			}
		
		return stats
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform comprehensive health check."""
		health = {
			'status': 'healthy',
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'components': {}
		}
		
		# Check HTTP session
		try:
			if self._http_session and not self._http_session.closed:
				health['components']['http_session'] = 'healthy'
			else:
				health['components']['http_session'] = 'unhealthy'
				health['status'] = 'degraded'
		except Exception as e:
			health['components']['http_session'] = f'error: {e}'
			health['status'] = 'unhealthy'
		
		# Check database
		if self.db_manager:
			try:
				await self.db_manager.get_recent_articles(1)
				health['components']['database'] = 'healthy'
			except Exception as e:
				health['components']['database'] = f'error: {e}'
				health['status'] = 'unhealthy'
		
		# Check circuit breaker
		if self.circuit_breaker:
			if self.circuit_breaker.is_closed():
				health['components']['circuit_breaker'] = 'closed'
			elif self.circuit_breaker.is_half_open():
				health['components']['circuit_breaker'] = 'half_open'
				health['status'] = 'degraded'
			else:
				health['components']['circuit_breaker'] = 'open'
				health['status'] = 'degraded'
		
		# Check rate limiter
		if self.rate_limiter:
			current_tokens = self.rate_limiter.get_current_tokens()
			capacity = self.rate_limiter.config.capacity
			health['components']['rate_limiter'] = f'{current_tokens:.1f}/{capacity} tokens'
		
		return health
	
	async def close(self) -> None:
		"""Clean up resources."""
		logger.info("Closing Enhanced Google News client...")
		
		if self._http_session:
			await self._http_session.close()
		
		if self.db_manager:
			await self.db_manager.close()
		
		logger.info("Enhanced Google News client closed")

# Factory functions
def create_enhanced_google_news_client(
	database_url: str = "postgresql:///lnd",
	enable_rate_limiting: bool = True,
	enable_circuit_breaker: bool = True,
	**config_kwargs
) -> EnhancedGoogleNewsClient:
	"""Factory function to create enhanced Google News client."""
	config = EnhancedGoogleNewsConfig(
		database_url=database_url,
		enable_rate_limiting=enable_rate_limiting,
		enable_circuit_breaker=enable_circuit_breaker,
		**config_kwargs
	)
	return EnhancedGoogleNewsClient(config)

def create_horn_africa_news_client(database_url: str = "postgresql:///lnd") -> EnhancedGoogleNewsClient:
	"""Create client optimized for Horn of Africa news monitoring."""
	config = EnhancedGoogleNewsConfig(
		database_url=database_url,
		target_countries=['KE', 'ET', 'SO', 'SD', 'SS', 'UG', 'TZ', 'ER', 'DJ'],
		allowed_languages=['en', 'fr', 'ar', 'sw', 'am'],
		enable_rate_limiting=True,
		enable_circuit_breaker=True,
		requests_per_second=3.0,  # Conservative for regional focus
		burst_capacity=30
	)
	return EnhancedGoogleNewsClient(config)