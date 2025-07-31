"""
Information Units Database Manager for Google News
==================================================

Complete database integration that maps Google News data to the information_units
schema with proper field mapping, deduplication, and transaction handling.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse

try:
	import asyncpg
except ImportError:
	asyncpg = None

try:
	from uuid_extensions import uuid7str
except ImportError:
	def uuid7str():
		return str(uuid.uuid4())

logger = logging.getLogger(__name__)

@dataclass
class GoogleNewsRecord:
	"""Google News data structure for information_units mapping."""
	# Core content fields
	title: str
	content: str = ""
	content_url: str = ""
	summary: str = ""
	
	# Source information
	source_name: str = ""
	source_domain: str = ""
	source_url: str = ""
	
	# Temporal data
	published_at: Optional[datetime] = None
	discovered_at: Optional[datetime] = None
	
	# Content metadata
	language_code: str = "en"
	keywords: List[str] = field(default_factory=list)
	tags: List[str] = field(default_factory=list)
	
	# Quality metrics
	credibility_score: float = 0.5
	sentiment_score: Optional[float] = None
	readability_score: Optional[float] = None
	
	# Geographical data
	coordinates: Optional[Dict[str, float]] = None
	location_names: List[str] = field(default_factory=list)
	
	# Google News specific metadata
	google_news_metadata: Dict[str, Any] = field(default_factory=dict)
	
	# Processing metadata
	extraction_confidence: float = 0.8
	verification_status: str = "unverified"
	classification_level: str = "public"

class InformationUnitsManager:
	"""
	Comprehensive database manager for storing Google News data
	in the information_units schema with full field mapping.
	"""
	
	def __init__(self, connection_string: str = "postgresql:///lnd"):
		"""Initialize the manager with database connection."""
		self.connection_string = connection_string
		self._connection_pool = None
		
		# Mapping configuration
		self.source_id_map = {
			'google_news': 'google-news-crawler-v1',
			'google_rss': 'google-rss-feed-v1'
		}
		
		# Unit type mappings for Google News content
		self.unit_type_mapping = {
			'news_article': 'news_article',
			'breaking_news': 'breaking_news',
			'opinion': 'opinion_piece', 
			'editorial': 'editorial',
			'analysis': 'news_analysis',
			'blog': 'blog_post',
			'press_release': 'press_release',
			'unknown': 'news_article'  # Default fallback
		}
	
	async def initialize(self) -> None:
		"""Initialize database connection pool."""
		if not asyncpg:
			raise ImportError("asyncpg is required for database operations")
		
		try:
			self._connection_pool = await asyncpg.create_pool(
				self.connection_string,
				min_size=2,
				max_size=10,
				command_timeout=30,
				server_settings={
					'application_name': 'google_news_crawler',
					'search_path': 'public'
				}
			)
			logger.info("✅ Database connection pool initialized")
		except Exception as e:
			logger.error(f"❌ Failed to initialize database pool: {e}")
			raise
	
	async def close(self) -> None:
		"""Close database connection pool."""
		if self._connection_pool:
			await self._connection_pool.close()
			logger.info("Database connection pool closed")
	
	async def store_google_news_record(self, record: GoogleNewsRecord) -> str:
		"""
		Store a Google News record in the information_units table.
		
		Args:
			record: GoogleNewsRecord to store
			
		Returns:
			str: The ID of the inserted record
		"""
		if not self._connection_pool:
			await self.initialize()
		
		# Convert to information_units format
		info_unit = await self._map_to_information_units(record)
		
		async with self._connection_pool.acquire() as conn:
			async with conn.transaction():
				try:
					# Check for duplicates using URL hash
					if info_unit.get('url_hash'):
						existing_id = await conn.fetchval("""
							SELECT id FROM information_units 
							WHERE url_hash = $1
						""", info_unit['url_hash'])
						
						if existing_id:
							logger.debug(f"Duplicate URL found, updating record: {existing_id}")
							return await self._update_existing_record(conn, existing_id, info_unit)
					
					# Insert new record
					record_id = await conn.fetchval("""
						INSERT INTO information_units (
							id, data_source_id, extraction_id, external_id, unit_type, title, content, 
							content_url, url_hash, content_hash, summary, keywords, tags,
							source_name, source_domain, source_url, source_type,
							language_code, published_at, discovered_at, scraped_at,
							capture_method, scraper_name, scraper_version, 
							content_length, word_count, char_count,
							sentiment_polarity, sentiment_subjectivity, sentiment_compound,
							readability_score, quality_score, credibility_score,
							coordinates, location_names, entities_mentioned,
							metadata, extraction_status, extraction_confidence_score,
							classification_level, verification_status, paywall_status,
							raw_content_snapshot, created_at, updated_at
						) VALUES (
							$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
							$18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, 
							$31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44
						) RETURNING id
					""",
					info_unit['id'], info_unit['data_source_id'], info_unit['extraction_id'],
					info_unit['external_id'], info_unit['unit_type'], info_unit['title'], info_unit['content'],
					info_unit['content_url'], info_unit['url_hash'], info_unit['content_hash'],
					info_unit['summary'], info_unit['keywords'], info_unit['tags'],
					info_unit['source_name'], info_unit['source_domain'], info_unit['source_url'],
					info_unit['source_type'], info_unit['language_code'], info_unit['published_at'],
					info_unit['discovered_at'], info_unit['scraped_at'], info_unit['capture_method'],
					info_unit['scraper_name'], info_unit['scraper_version'], info_unit['content_length'],
					info_unit['word_count'], info_unit['char_count'], info_unit['sentiment_polarity'],
					info_unit['sentiment_subjectivity'], info_unit['sentiment_compound'],
					info_unit['readability_score'], info_unit['quality_score'], info_unit['credibility_score'],
					info_unit['coordinates'], info_unit['location_names'], info_unit['entities_mentioned'],
					info_unit['metadata'], info_unit['extraction_status'], info_unit['extraction_confidence_score'],
					info_unit['classification_level'], info_unit['verification_status'],
					info_unit['paywall_status'], info_unit['raw_content_snapshot'],
					info_unit['created_at'], info_unit['updated_at']
					)
					
					logger.debug(f"✅ Stored Google News record: {record_id}")
					return record_id
					
				except Exception as e:
					logger.error(f"❌ Failed to store Google News record: {e}")
					raise
	
	async def store_search_result(self, search_data: Dict[str, Any]) -> str:
		"""
		Store a search result in the information_units table.
		
		Args:
			search_data: Dictionary containing search result data
			
		Returns:
			str: The ID of the inserted record
		"""
		if not self._connection_pool:
			await self.initialize()
		
		# Convert to information_units format
		info_unit = await self._map_search_to_information_units(search_data)
		
		async with self._connection_pool.acquire() as conn:
			async with conn.transaction():
				try:
					# Check for duplicates using URL hash
					if info_unit.get('url_hash'):
						existing_id = await conn.fetchval("""
							SELECT id FROM information_units 
							WHERE url_hash = $1
						""", info_unit['url_hash'])
						
						if existing_id:
							logger.debug(f"Duplicate URL found, updating record: {existing_id}")
							return await self._update_existing_record(conn, existing_id, info_unit)
					
					# Insert new record with extraction_id
					record_id = await conn.fetchval("""
						INSERT INTO information_units (
							id, data_source_id, extraction_id, external_id, unit_type, title, content, 
							content_url, url_hash, content_hash, summary, keywords, tags,
							source_name, source_domain, source_url, source_type,
							language_code, published_at, discovered_at, scraped_at,
							capture_method, scraper_name, scraper_version, 
							content_length, word_count, char_count,
							sentiment_polarity, sentiment_subjectivity, sentiment_compound,
							readability_score, quality_score, credibility_score,
							coordinates, location_names, entities_mentioned,
							metadata, extraction_status, extraction_confidence_score,
							classification_level, verification_status, paywall_status,
							raw_content_snapshot, created_at, updated_at
						) VALUES (
							$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
							$18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, 
							$31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44
						) RETURNING id
					""",
					info_unit['id'], info_unit['data_source_id'], info_unit['extraction_id'],
					info_unit['external_id'], info_unit['unit_type'], info_unit['title'], info_unit['content'],
					info_unit['content_url'], info_unit['url_hash'], info_unit['content_hash'],
					info_unit['summary'], info_unit['keywords'], info_unit['tags'],
					info_unit['source_name'], info_unit['source_domain'], info_unit['source_url'],
					info_unit['source_type'], info_unit['language_code'], info_unit['published_at'],
					info_unit['discovered_at'], info_unit['scraped_at'], info_unit['capture_method'],
					info_unit['scraper_name'], info_unit['scraper_version'], info_unit['content_length'],
					info_unit['word_count'], info_unit['char_count'], info_unit['sentiment_polarity'],
					info_unit['sentiment_subjectivity'], info_unit['sentiment_compound'],
					info_unit['readability_score'], info_unit['quality_score'], info_unit['credibility_score'],
					info_unit['coordinates'], info_unit['location_names'], info_unit['entities_mentioned'],
					info_unit['metadata'], info_unit['extraction_status'], info_unit['extraction_confidence_score'],
					info_unit['classification_level'], info_unit['verification_status'], info_unit['paywall_status'],
					info_unit['raw_content_snapshot'], info_unit['created_at'], info_unit['updated_at']
					)
					
					logger.debug(f"✅ Stored search result: {record_id}")
					return record_id
					
				except Exception as e:
					logger.error(f"❌ Failed to store search result: {e}")
					raise
	
	async def _map_to_information_units(self, record: GoogleNewsRecord) -> Dict[str, Any]:
		"""Map GoogleNewsRecord to information_units schema format."""
		now = datetime.now(timezone.utc)
		
		# Generate IDs and hashes
		record_id = uuid7str()
		extraction_id = uuid7str()  # Generate extraction_id to satisfy NOT NULL constraint
		external_id = f"gnews_{hashlib.md5(record.content_url.encode()).hexdigest()}"
		
		# Generate hashes for deduplication
		url_hash = None
		content_hash = None
		
		if record.content_url:
			url_hash = hashlib.sha256(record.content_url.encode('utf-8')).hexdigest()
		
		if record.content:
			content_hash = hashlib.sha256(record.content.encode('utf-8')).hexdigest()
		
		# Extract domain from URL
		source_domain = record.source_domain
		if not source_domain and record.content_url:
			try:
				parsed_url = urlparse(record.content_url)
				source_domain = parsed_url.netloc
			except:
				pass
		
		# Calculate content metrics
		content_length = len(record.content) if record.content else 0
		word_count = len(record.content.split()) if record.content else 0
		char_count = len(record.content) if record.content else 0
		
		# Create comprehensive metadata
		metadata = {
			'google_news_crawler_version': '1.0.0',
			'source_credibility_score': record.credibility_score,
			'extraction_method': 'google_news_api',
			'content_quality_metrics': {
				'has_title': bool(record.title),
				'has_content': bool(record.content),
				'has_summary': bool(record.summary),
				'word_count': word_count,
				'char_count': char_count
			},
			'google_news_specific': record.google_news_metadata,
			'processing_timestamp': now.isoformat(),
			'keywords_extracted': record.keywords,
			'tags_applied': record.tags,
			'location_data': {
				'coordinates': record.coordinates,
				'location_names': record.location_names
			}
		}
		
		return {
			'id': record_id,
			'data_source_id': None,  # Can be populated with actual data source tracking
			'extraction_id': extraction_id,  # Add extraction_id to prevent NOT NULL constraint violation
			'external_id': external_id,
			'unit_type': self.unit_type_mapping.get('news_article', 'news_article'),
			'title': record.title[:500] if record.title else None,  # Truncate if needed
			'content': record.content,
			'content_url': record.content_url,
			'url_hash': url_hash,
			'content_hash': content_hash,
			'summary': record.summary,
			'keywords': record.keywords,
			'tags': record.tags,
			'source_name': record.source_name,
			'source_domain': source_domain,
			'source_url': record.source_url,
			'source_type': 'news_website',
			'language_code': record.language_code,
			'published_at': record.published_at or now,
			'discovered_at': record.discovered_at or now,
			'scraped_at': now,
			'capture_method': 'api_crawl_google_news',
			'scraper_name': 'google_news_crawler',
			'scraper_version': '1.0.0',
			'content_length': content_length,
			'word_count': word_count,
			'char_count': char_count,
			'sentiment_polarity': record.sentiment_score,
			'sentiment_subjectivity': None,  # Can be calculated if needed
			'sentiment_compound': record.sentiment_score,  # Using same value for now
			'readability_score': record.readability_score,
			'quality_score': record.credibility_score,
			'credibility_score': record.credibility_score,
			'coordinates': json.dumps(record.coordinates) if record.coordinates else None,
			'location_names': record.location_names,
			'entities_mentioned': [],  # Can be populated with NER
			'metadata': json.dumps(metadata),
			'extraction_status': 'completed',
			'extraction_confidence_score': record.extraction_confidence,
			'classification_level': record.classification_level,
			'verification_status': record.verification_status,
			'paywall_status': 'unknown',  # Google News doesn't indicate paywall status
			'raw_content_snapshot': record.content,
			'created_at': now,
			'updated_at': now
		}
	
	async def _map_search_to_information_units(self, search_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Map search result data to information_units schema format."""
		now = datetime.now(timezone.utc)
		
		# Generate IDs and hashes
		record_id = uuid7str()
		extraction_id = uuid7str()  # Generate extraction_id to satisfy NOT NULL constraint
		
		# Get URL from search data
		content_url = search_data.get('content_url') or search_data.get('url')
		external_id = f"search_{hashlib.md5(content_url.encode()).hexdigest()}" if content_url else f"search_{record_id}"
		
		# Generate hashes for deduplication
		url_hash = None
		content_hash = None
		
		if content_url:
			url_hash = hashlib.sha256(content_url.encode('utf-8')).hexdigest()
		
		content = search_data.get('content', '')
		if content:
			content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
		
		# Extract domain from URL
		source_domain = search_data.get('source_domain')
		if not source_domain and content_url:
			try:
				parsed_url = urlparse(content_url)
				source_domain = parsed_url.netloc
			except:
				pass
		
		# Calculate content metrics
		content_length = len(content) if content else 0
		word_count = len(content.split()) if content else 0
		char_count = len(content) if content else 0
		
		# Create comprehensive metadata
		metadata = {
			'search_content_pipeline_version': '1.0.0',
			'extraction_method': search_data.get('extraction_method', 'search_content_pipeline'),
			'search_engine': search_data.get('search_engine', 'unknown'),
			'content_quality_metrics': {
				'has_title': bool(search_data.get('title')),
				'has_content': bool(content),
				'has_summary': bool(search_data.get('summary')),
				'word_count': word_count,
				'char_count': char_count
			},
			'search_metadata': search_data.get('metadata', {}),
			'extraction_timestamp': now.isoformat()
		}
		
		# Parse published date
		published_at = search_data.get('published_at')
		if published_at and isinstance(published_at, str):
			try:
				published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
			except:
				published_at = now
		elif not published_at:
			published_at = now
		
		return {
			'id': record_id,
			'data_source_id': search_data.get('data_source_id', 'search_content_pipeline'),
			'extraction_id': extraction_id,  # Add extraction_id to prevent NOT NULL constraint violation
			'external_id': external_id,
			'unit_type': search_data.get('unit_type', 'article'),
			'title': search_data.get('title'),
			'content': content,
			'content_url': content_url,
			'url_hash': url_hash,
			'content_hash': content_hash,
			'summary': search_data.get('summary'),
			'keywords': search_data.get('keywords'),
			'tags': search_data.get('tags'),
			'source_name': search_data.get('source_name'),
			'source_domain': source_domain,
			'source_url': content_url,
			'source_type': search_data.get('source_type', 'web'),
			'language_code': search_data.get('language_code', 'en'),
			'published_at': published_at,
			'discovered_at': now,
			'scraped_at': now,
			'capture_method': search_data.get('capture_method', 'search_pipeline'),
			'scraper_name': search_data.get('scraper_name', 'search_content_pipeline'),
			'scraper_version': search_data.get('scraper_version', '1.0.0'),
			'content_length': content_length,
			'word_count': word_count,
			'char_count': char_count,
			'sentiment_polarity': search_data.get('sentiment_polarity'),
			'sentiment_subjectivity': search_data.get('sentiment_subjectivity'),
			'sentiment_compound': search_data.get('sentiment_compound'),
			'readability_score': search_data.get('readability_score'),
			'quality_score': search_data.get('quality_score', 0.5),
			'credibility_score': search_data.get('credibility_score', 0.5),
			'coordinates': search_data.get('coordinates'),
			'location_names': search_data.get('location_names'),
			'entities_mentioned': search_data.get('entities_mentioned', []),
			'metadata': json.dumps(metadata),
			'extraction_status': 'completed',
			'extraction_confidence_score': search_data.get('confidence_score', 0.5),
			'classification_level': search_data.get('classification_level', 'public'),
			'verification_status': search_data.get('verification_status', 'unverified'),
			'paywall_status': search_data.get('paywall_status', 'unknown'),
			'raw_content_snapshot': content,
			'created_at': now,
			'updated_at': now
		}
	
	async def _update_existing_record(self, conn, record_id: str, info_unit: Dict[str, Any]) -> str:
		"""Update an existing record with new information."""
		now = datetime.now(timezone.utc)
		
		await conn.execute("""
			UPDATE information_units SET
				title = COALESCE($2, title),
				content = COALESCE($3, content),
				summary = COALESCE($4, summary),
				metadata = $5,
				updated_at = $6,
				scraped_at = $6
			WHERE id = $1
		""", record_id, info_unit['title'], info_unit['content'], 
		info_unit['summary'], info_unit['metadata'], now)
		
		logger.debug(f"✅ Updated existing record: {record_id}")
		return record_id
	
	async def get_recent_articles(self, limit: int = 50) -> List[Dict[str, Any]]:
		"""Get recent articles from Google News crawler."""
		if not self._connection_pool:
			await self.initialize()
		
		async with self._connection_pool.acquire() as conn:
			records = await conn.fetch("""
				SELECT id, title, content_url, source_domain, published_at, 
					   scraped_at, quality_score, extraction_confidence_score
				FROM information_units 
				WHERE scraper_name = 'google_news_crawler'
				ORDER BY scraped_at DESC 
				LIMIT $1
			""", limit)
			
			return [dict(record) for record in records]
	
	async def get_duplicate_count(self) -> Dict[str, int]:
		"""Get statistics on duplicate content."""
		if not self._connection_pool:
			await self.initialize()
		
		async with self._connection_pool.acquire() as conn:
			url_duplicates = await conn.fetchval("""
				SELECT COUNT(*) FROM (
					SELECT url_hash FROM information_units 
					WHERE scraper_name = 'google_news_crawler' AND url_hash IS NOT NULL
					GROUP BY url_hash HAVING COUNT(*) > 1
				) AS dupes
			""")
			
			content_duplicates = await conn.fetchval("""
				SELECT COUNT(*) FROM (
					SELECT content_hash FROM information_units 
					WHERE scraper_name = 'google_news_crawler' AND content_hash IS NOT NULL
					GROUP BY content_hash HAVING COUNT(*) > 1
				) AS dupes
			""")
			
			return {
				'url_duplicates': url_duplicates,
				'content_duplicates': content_duplicates
			}

def map_google_news_to_information_units(
	title: str,
	url: str,
	content: str = "",
	source_name: str = "",
	published_at: Optional[datetime] = None,
	**kwargs
) -> GoogleNewsRecord:
	"""
	Convenience function to create GoogleNewsRecord from basic Google News data.
	
	Args:
		title: Article title
		url: Article URL
		content: Article content
		source_name: Name of the source
		published_at: Publication timestamp
		**kwargs: Additional fields for GoogleNewsRecord
		
	Returns:
		GoogleNewsRecord: Structured record ready for database storage
	"""
	# Extract domain from URL
	source_domain = ""
	if url:
		try:
			parsed_url = urlparse(url)
			source_domain = parsed_url.netloc
		except:
			pass
	
	return GoogleNewsRecord(
		title=title,
		content=content,
		content_url=url,
		source_name=source_name,
		source_domain=source_domain,
		published_at=published_at,
		discovered_at=datetime.now(timezone.utc),
		**kwargs
	)

def create_information_units_manager(connection_string: str = "postgresql:///lnd") -> InformationUnitsManager:
	"""Factory function to create InformationUnitsManager."""
	return InformationUnitsManager(connection_string)