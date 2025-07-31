"""
Integration Layer for Enhanced Google News Crawler
==================================================

This module provides integration between the original GoogleNewsClient
and the new enhanced features (database, rate limiting, circuit breaker).

It allows backward compatibility while enabling opt-in to enhanced features.

Author: Nyimbi Odero  
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .api.google_news_client import GoogleNewsClient
from .enhanced_client import (
	EnhancedGoogleNewsClient,
	EnhancedGoogleNewsConfig,
	create_enhanced_google_news_client,
	create_horn_africa_news_client
)
from .database import (
	InformationUnitsManager,
	create_information_units_manager
)
from .resilience import (
	CircuitBreaker,
	ErrorHandler,
	create_google_news_circuit_breaker,
	create_error_handler
)

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
	"""Configuration for integration features."""
	# Database integration
	enable_database: bool = False
	database_url: str = "postgresql:///lnd"
	
	# Enhanced features
	enable_rate_limiting: bool = False
	enable_circuit_breaker: bool = False
	enable_enhanced_error_handling: bool = False
	
	# Performance tuning
	requests_per_second: float = 5.0
	max_concurrent_requests: int = 10
	
	# Regional focus
	target_region: str = "horn_of_africa"  # or "global"

class IntegratedGoogleNewsClient:
	"""
	Integrated client that combines original GoogleNewsClient 
	with enhanced production-ready features.
	"""
	
	def __init__(self, 
				 integration_config: Optional[IntegrationConfig] = None,
				 original_client_config: Optional[Dict[str, Any]] = None):
		"""Initialize integrated client."""
		self.integration_config = integration_config or IntegrationConfig()
		self.original_client_config = original_client_config or {}
		
		# Client instances
		self.original_client: Optional[GoogleNewsClient] = None
		self.enhanced_client: Optional[EnhancedGoogleNewsClient] = None
		
		# Feature flags
		self.use_enhanced_features = (
			self.integration_config.enable_database or
			self.integration_config.enable_rate_limiting or
			self.integration_config.enable_circuit_breaker
		)
		
		logger.info(f"Integrated client initialized (enhanced_features: {self.use_enhanced_features})")
	
	async def initialize(self) -> None:
		"""Initialize the appropriate client based on configuration."""
		if self.use_enhanced_features:
			await self._initialize_enhanced_client()
		else:
			await self._initialize_original_client()
	
	async def _initialize_enhanced_client(self) -> None:
		"""Initialize enhanced client with production features."""
		config = EnhancedGoogleNewsConfig(
			database_url=self.integration_config.database_url,
			enable_database_storage=self.integration_config.enable_database,
			enable_rate_limiting=self.integration_config.enable_rate_limiting,
			enable_circuit_breaker=self.integration_config.enable_circuit_breaker,
			requests_per_second=self.integration_config.requests_per_second,
			max_concurrent_requests=self.integration_config.max_concurrent_requests
		)
		
		if self.integration_config.target_region == "horn_of_africa":
			self.enhanced_client = create_horn_africa_news_client(
				self.integration_config.database_url
			)
		else:
			self.enhanced_client = EnhancedGoogleNewsClient(config)
		
		await self.enhanced_client.initialize()
		logger.info("✅ Enhanced client initialized")
	
	async def _initialize_original_client(self) -> None:
		"""Initialize original client for basic functionality."""
		# Import the PostgreSQL manager from the original system
		try:
			from lindela.packages_enhanced.database.postgresql_manager import PgSQLManager
			db_manager = PgSQLManager(self.integration_config.database_url)
		except ImportError:
			db_manager = None
			logger.warning("PostgreSQL manager not available, using original client without database")
		
		self.original_client = GoogleNewsClient(
			db_manager=db_manager,
			**self.original_client_config
		)
		await self.original_client.initialize()
		logger.info("✅ Original client initialized")
	
	async def search_news(self, 
						  query: str,
						  language: str = 'en',
						  country: str = 'KE',
						  max_results: int = 100) -> List[Dict[str, Any]]:
		"""Search news using the appropriate client."""
		if self.enhanced_client:
			return await self.enhanced_client.search_news(
				query, language, country, max_results
			)
		elif self.original_client:
			# Convert to original client format
			results = await self.original_client.search_news(
				query=query,
				language=language,
				country=country,
				max_results=max_results
			)
			# Convert results to enhanced format for consistency
			return self._convert_original_results(results)
		else:
			raise RuntimeError("No client initialized")
	
	async def get_headlines(self, country: str = 'KE', language: str = 'en') -> List[Dict[str, Any]]:
		"""Get headlines using the appropriate client."""
		if self.enhanced_client:
			return await self.enhanced_client.get_headlines(country, language)
		elif self.original_client:
			# Use original client's method
			results = await self.original_client.get_top_headlines(
				country=country,
				language=language
			)
			return self._convert_original_results(results)
		else:
			raise RuntimeError("No client initialized")
	
	async def monitor_keywords(self, keywords: List[str], **kwargs) -> Dict[str, List[Dict[str, Any]]]:
		"""Monitor multiple keywords."""
		if self.enhanced_client:
			return await self.enhanced_client.monitor_keywords(keywords, **kwargs)
		elif self.original_client:
			# Implement monitoring with original client
			results = {}
			for keyword in keywords:
				try:
					articles = await self.search_news(keyword, **kwargs)
					results[keyword] = articles
				except Exception as e:
					logger.error(f"Failed to search for '{keyword}': {e}")
					results[keyword] = []
			return results
		else:
			raise RuntimeError("No client initialized")
	
	async def get_stats(self) -> Dict[str, Any]:
		"""Get comprehensive statistics."""
		if self.enhanced_client:
			return await self.enhanced_client.get_stats()
		elif self.original_client:
			# Create basic stats for original client
			return {
				'client_type': 'original',
				'features': {
					'database': self.integration_config.enable_database,
					'rate_limiting': False,
					'circuit_breaker': False,
					'enhanced_error_handling': False
				},
				'performance': {
					'note': 'Basic statistics only available with enhanced client'
				}
			}
		else:
			return {'error': 'No client initialized'}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform health check."""
		if self.enhanced_client:
			return await self.enhanced_client.health_check()
		elif self.original_client:
			return {
				'status': 'healthy',
				'client_type': 'original',
				'components': {
					'original_client': 'operational'
				}
			}
		else:
			return {
				'status': 'unhealthy',
				'error': 'No client initialized'
			}
	
	def _convert_original_results(self, results: List[Any]) -> List[Dict[str, Any]]:
		"""Convert original client results to enhanced format."""
		converted = []
		
		for result in results:
			# Handle different result formats from original client
			if hasattr(result, '__dict__'):
				# If it's an object, convert to dict
				article_dict = result.__dict__.copy()
			elif isinstance(result, dict):
				article_dict = result.copy()
			else:
				# Handle other formats
				article_dict = {
					'title': str(result),
					'url': '',
					'content': '',
					'source_name': 'Unknown'
				}
			
			# Ensure consistent field names
			standardized = {
				'title': article_dict.get('title', ''),
				'url': article_dict.get('url', article_dict.get('link', '')),
				'content': article_dict.get('content', article_dict.get('summary', '')),
				'summary': article_dict.get('summary', ''),
				'source_name': article_dict.get('source_name', article_dict.get('source', '')),
				'published_at': article_dict.get('published_at', article_dict.get('published', '')),
				'discovered_at': article_dict.get('discovered_at', ''),
				'metadata': article_dict.get('metadata', {})
			}
			
			converted.append(standardized)
		
		return converted
	
	async def close(self) -> None:
		"""Close the appropriate client."""
		if self.enhanced_client:
			await self.enhanced_client.close()
		elif self.original_client:
			await self.original_client.close()
		
		logger.info("Integrated client closed")

# Factory functions for easy integration

def create_integrated_client(
	enable_database: bool = False,
	enable_rate_limiting: bool = False,
	enable_circuit_breaker: bool = False,
	database_url: str = "postgresql:///lnd",
	target_region: str = "horn_of_africa",
	**original_client_kwargs
) -> IntegratedGoogleNewsClient:
	"""
	Factory function to create integrated client with specified features.
	
	Args:
		enable_database: Enable database storage
		enable_rate_limiting: Enable rate limiting
		enable_circuit_breaker: Enable circuit breaker
		database_url: Database connection string
		target_region: Target region ("horn_of_africa" or "global")
		**original_client_kwargs: Arguments for original client
	
	Returns:
		IntegratedGoogleNewsClient: Configured integrated client
	"""
	integration_config = IntegrationConfig(
		enable_database=enable_database,
		enable_rate_limiting=enable_rate_limiting,
		enable_circuit_breaker=enable_circuit_breaker,
		database_url=database_url,
		target_region=target_region
	)
	
	return IntegratedGoogleNewsClient(
		integration_config=integration_config,
		original_client_config=original_client_kwargs
	)

def create_production_ready_client(
	database_url: str = "postgresql:///lnd",
	target_region: str = "horn_of_africa"
) -> IntegratedGoogleNewsClient:
	"""
	Create client with all production features enabled.
	
	Args:
		database_url: Database connection string
		target_region: Target region for optimization
	
	Returns:
		IntegratedGoogleNewsClient: Production-ready client
	"""
	return create_integrated_client(
		enable_database=True,
		enable_rate_limiting=True,
		enable_circuit_breaker=True,
		enable_enhanced_error_handling=True,
		database_url=database_url,
		target_region=target_region
	)

def create_basic_client(**original_client_kwargs) -> IntegratedGoogleNewsClient:
	"""
	Create basic client with original functionality only.
	
	Args:
		**original_client_kwargs: Arguments for original client
	
	Returns:
		IntegratedGoogleNewsClient: Basic client
	"""
	return create_integrated_client(
		enable_database=False,
		enable_rate_limiting=False,
		enable_circuit_breaker=False,
		**original_client_kwargs
	)

# Migration helper functions

def migrate_to_enhanced_features(
	existing_client: GoogleNewsClient,
	database_url: str = "postgresql:///lnd"
) -> IntegratedGoogleNewsClient:
	"""
	Helper function to migrate from existing client to enhanced features.
	
	Args:
		existing_client: Existing GoogleNewsClient instance
		database_url: Database connection string
	
	Returns:
		IntegratedGoogleNewsClient: Enhanced client
	"""
	logger.info("Migrating to enhanced features...")
	
	# Extract configuration from existing client if possible
	original_config = {}
	if hasattr(existing_client, 'config'):
		original_config = existing_client.config
	
	return create_production_ready_client(
		database_url=database_url
	)

# Usage examples in docstrings

"""
Usage Examples:
==============

1. Basic usage (original functionality):
   client = create_basic_client()
   await client.initialize()
   results = await client.search_news("Ethiopia conflict")

2. Production-ready with all features:
   client = create_production_ready_client("postgresql://user:pass@localhost/db")
   await client.initialize()
   results = await client.search_news("Somalia security")
   stats = await client.get_stats()

3. Custom feature selection:
   client = create_integrated_client(
       enable_database=True,
       enable_rate_limiting=True,
       enable_circuit_breaker=False
   )
   await client.initialize()
   results = await client.monitor_keywords(["conflict", "peace", "security"])

4. Migration from existing client:
   enhanced_client = migrate_to_enhanced_features(old_client)
   await enhanced_client.initialize()
"""