"""
Base Adapter Interface for Specialized Crawlers
==============================================

Defines the common interface for integrating existing specialized crawlers
with the APG Simple API. All adapters inherit from this base class.

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class CrawlerType(Enum):
	"""Types of specialized crawlers"""
	SEARCH = "search"
	GDELT = "gdelt"
	GOOGLE_NEWS = "google_news"
	TWITTER = "twitter"
	YOUTUBE = "youtube"
	GENERAL = "general"

@dataclass
class AdapterResult:
	"""
	Standardized result from specialized crawler adapters.
	Maps specialized crawler outputs to APG Simple API format.
	"""
	# Core identification
	url: str
	title: Optional[str]
	content: str  # Markdown formatted content
	
	# Status information
	success: bool
	error: Optional[str] = None
	
	# Source information
	crawler_type: CrawlerType = CrawlerType.GENERAL
	original_source: Optional[str] = None
	
	# Temporal information
	crawl_timestamp: datetime = field(default_factory=datetime.now)
	publish_date: Optional[datetime] = None
	
	# Content metadata
	language: Optional[str] = None
	word_count: int = 0
	
	# Specialized metadata (varies by crawler type)
	specialized_metadata: Dict[str, Any] = field(default_factory=dict)
	
	# APG integration fields
	tenant_id: str = "default"
	processing_metadata: Dict[str, Any] = field(default_factory=dict)

class BaseSpecializedCrawlerAdapter(ABC):
	"""
	Base class for all specialized crawler adapters.
	
	Each adapter translates between specialized crawler interfaces
	and the standardized APG Simple API format.
	"""
	
	def __init__(self, crawler_type: CrawlerType, tenant_id: str = "default"):
		self.crawler_type = crawler_type
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(self.__class__.__name__)
		self.stats = {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'average_response_time': 0.0
		}
	
	@abstractmethod
	async def initialize(self) -> None:
		"""Initialize the specialized crawler and any required resources."""
		pass
	
	@abstractmethod
	async def crawl_single(self, query_or_url: str, **kwargs) -> AdapterResult:
		"""
		Crawl a single item using the specialized crawler.
		
		Args:
			query_or_url: Search query or URL to crawl
			**kwargs: Crawler-specific parameters
			
		Returns:
			AdapterResult with standardized format
		"""
		pass
	
	@abstractmethod
	async def crawl_batch(
		self, 
		queries_or_urls: List[str], 
		max_concurrent: int = 3,
		**kwargs
	) -> List[AdapterResult]:
		"""
		Crawl multiple items concurrently using the specialized crawler.
		
		Args:
			queries_or_urls: List of queries or URLs
			max_concurrent: Maximum concurrent requests
			**kwargs: Crawler-specific parameters
			
		Returns:
			List of AdapterResult objects
		"""
		pass
	
	@abstractmethod
	async def cleanup(self) -> None:
		"""Clean up any resources used by the specialized crawler."""
		pass
	
	@abstractmethod
	def get_supported_parameters(self) -> Dict[str, Any]:
		"""
		Get the parameters supported by this specialized crawler.
		
		Returns:
			Dictionary describing supported parameters and their types
		"""
		pass
	
	# Common helper methods
	
	async def _execute_with_stats(self, coro):
		"""Execute a coroutine while tracking statistics."""
		import time
		
		start_time = time.time()
		self.stats['total_requests'] += 1
		
		try:
			result = await coro
			self.stats['successful_requests'] += 1
			return result
		except Exception as e:
			self.stats['failed_requests'] += 1
			self.logger.error(f"Request failed: {e}")
			raise
		finally:
			response_time = time.time() - start_time
			# Update rolling average
			total_successful = self.stats['successful_requests']
			if total_successful > 0:
				current_avg = self.stats['average_response_time']
				self.stats['average_response_time'] = (
					(current_avg * (total_successful - 1) + response_time) / total_successful
				)
	
	def _create_error_result(
		self, 
		query_or_url: str, 
		error: str,
		**metadata
	) -> AdapterResult:
		"""Create a standardized error result."""
		return AdapterResult(
			url=query_or_url,
			title=None,
			content=f"# Error\n\nFailed to crawl: {error}",
			success=False,
			error=error,
			crawler_type=self.crawler_type,
			tenant_id=self.tenant_id,
			specialized_metadata=metadata
		)
	
	def _extract_text_content(self, raw_content: Any) -> str:
		"""Extract and clean text content from various formats."""
		if isinstance(raw_content, str):
			return raw_content.strip()
		elif isinstance(raw_content, dict):
			# Try common content fields
			for field in ['content', 'text', 'body', 'description', 'summary']:
				if field in raw_content and raw_content[field]:
					return str(raw_content[field]).strip()
			# Fallback to string representation
			return str(raw_content)
		elif hasattr(raw_content, 'text'):
			return raw_content.text.strip()
		else:
			return str(raw_content).strip()
	
	def _format_as_markdown(self, title: str, content: str, metadata: Dict = None) -> str:
		"""Format content as clean markdown."""
		markdown_parts = []
		
		# Title
		if title:
			markdown_parts.append(f"# {title}\n")
		
		# Content
		if content:
			# Clean up the content
			clean_content = content.strip()
			
			# Add basic formatting if needed
			if not any(clean_content.startswith(marker) for marker in ['#', '-', '*', '>']):
				clean_content = clean_content.replace('\n\n', '\n\n')
			
			markdown_parts.append(clean_content)
		
		# Metadata section (optional)
		if metadata:
			markdown_parts.append("\n---\n")
			markdown_parts.append("## Metadata\n")
			for key, value in metadata.items():
				if value is not None:
					markdown_parts.append(f"**{key.replace('_', ' ').title()}:** {value}\n")
		
		return '\n'.join(markdown_parts)
	
	def _count_words(self, text: str) -> int:
		"""Count words in text content."""
		if not text:
			return 0
		# Simple word count - split on whitespace
		return len(text.split())
	
	def _detect_language(self, text: str) -> Optional[str]:
		"""Detect language of text content (basic implementation)."""
		if not text or len(text) < 10:
			return None
		
		# Very basic language detection based on common words
		text_lower = text.lower()
		
		# English indicators
		english_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']
		english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
		
		if english_count > 3:
			return 'en'
		
		return None  # Unknown
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get adapter statistics."""
		success_rate = 0.0
		if self.stats['total_requests'] > 0:
			success_rate = self.stats['successful_requests'] / self.stats['total_requests']
		
		return {
			**self.stats,
			'success_rate': success_rate,
			'crawler_type': self.crawler_type.value,
			'tenant_id': self.tenant_id
		}
	
	def reset_stats(self):
		"""Reset adapter statistics."""
		self.stats = {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'average_response_time': 0.0
		}
		self.logger.info(f"Statistics reset for {self.crawler_type.value} adapter")