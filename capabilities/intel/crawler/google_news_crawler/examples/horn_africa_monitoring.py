#!/usr/bin/env python3
"""
Horn of Africa News Monitoring Example
=====================================

Comprehensive example demonstrating the enhanced Google News crawler
for conflict monitoring in the Horn of Africa region.

Features demonstrated:
- Enhanced Google News client with database integration
- Rate limiting with token bucket algorithm
- Circuit breaker for resilience
- Comprehensive error handling
- Real-time monitoring and health checks
- Statistics and performance tracking

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
	"""Main example demonstrating Horn of Africa news monitoring."""
	try:
		# Import the enhanced client
		from ..enhanced_client import (
			create_horn_africa_news_client,
			EnhancedGoogleNewsConfig
		)
		
		logger.info("🚀 Starting Horn of Africa News Monitoring Example")
		
		# Configure database connection
		database_url = os.getenv("DATABASE_URL", "postgresql:///lnd")
		
		# Create the enhanced client optimized for Horn of Africa
		client = create_horn_africa_news_client(database_url)
		
		# Initialize the client (this sets up database, rate limiter, circuit breaker)
		await client.initialize()
		
		logger.info("✅ Enhanced Google News Client initialized")
		
		# Perform health check
		health = await client.health_check()
		logger.info(f"🏥 Health check: {health['status']}")
		for component, status in health['components'].items():
			logger.info(f"   {component}: {status}")
		
		# Define conflict-related search queries for Horn of Africa
		conflict_queries = [
			"Ethiopia conflict Tigray",
			"Somalia Al-Shabaab security",
			"Sudan crisis Darfur",
			"South Sudan violence",
			"Horn of Africa drought humanitarian",
			"Kenya Somalia border security",
			"Ethiopian dam Egypt Sudan",
			"Eritrea Ethiopia border",
			"Djibouti military base"
		]
		
		logger.info(f"🔍 Monitoring {len(conflict_queries)} conflict-related queries")
		
		# Monitor keywords with concurrent search
		results = await client.monitor_keywords(
			keywords=conflict_queries,
			language='en',
			country='ET',  # Ethiopia as central country
			max_results=50
		)
		
		# Process and analyze results
		total_articles = 0
		for query, articles in results.items():
			article_count = len(articles)
			total_articles += article_count
			
			if article_count > 0:
				logger.info(f"📰 '{query}': {article_count} articles")
				
				# Show sample articles
				for i, article in enumerate(articles[:3]):  # Show first 3
					logger.info(f"   {i+1}. {article['title']}")
					logger.info(f"      Source: {article['source_name']}")
					logger.info(f"      Published: {article.get('published_at', 'Unknown')}")
			else:
				logger.warning(f"❌ '{query}': No articles found")
		
		logger.info(f"📊 Total articles discovered: {total_articles}")
		
		# Get comprehensive statistics
		stats = await client.get_stats()
		
		logger.info("📈 Performance Statistics:")
		logger.info(f"   Uptime: {stats['uptime_seconds']:.1f} seconds")
		logger.info(f"   Searches performed: {stats['performance']['searches_performed']}")
		logger.info(f"   Articles discovered: {stats['performance']['articles_discovered']}")
		logger.info(f"   Articles stored: {stats['performance']['articles_stored']}")
		logger.info(f"   Articles skipped: {stats['performance']['articles_skipped']}")
		logger.info(f"   Errors handled: {stats['performance']['errors_handled']}")
		
		# Performance rates
		logger.info("🎯 Performance Rates:")
		for rate_name, rate_value in stats['rates'].items():
			logger.info(f"   {rate_name}: {rate_value:.2f}")
		
		# Component statistics
		if 'rate_limiter' in stats:
			rate_stats = stats['rate_limiter']
			logger.info("⏱️  Rate Limiter Statistics:")
			logger.info(f"   Current tokens: {rate_stats['current_tokens']:.1f}/{rate_stats['max_capacity']}")
			logger.info(f"   Success rate: {rate_stats['success_rate_percent']:.1f}%")
			logger.info(f"   Rate multiplier: {rate_stats['current_rate_multiplier']:.2f}")
		
		if 'circuit_breaker' in stats:
			cb_stats = stats['circuit_breaker']
			logger.info("🔌 Circuit Breaker Statistics:")
			logger.info(f"   State: {cb_stats['current_state']}")
			logger.info(f"   Success rate: {cb_stats['success_rate']:.1f}%")
			logger.info(f"   Total requests: {cb_stats['total_requests']}")
		
		if 'error_handler' in stats:
			error_stats = stats['error_handler']
			logger.info("🚨 Error Handler Statistics:")
			logger.info(f"   Total errors: {error_stats['total_errors']}")
			logger.info(f"   Errors per hour: {error_stats['errors_per_hour']:.1f}")
			
			if error_stats['most_common_errors']:
				logger.info("   Most common errors:")
				for error in error_stats['most_common_errors']:
					logger.info(f"     {error['category']}: {error['count']}")
		
		# Demonstrate topic-based search
		logger.info("\n🎯 Topic-Based Search Example:")
		
		conflict_articles = await client.search_by_topic(
			"conflict",
			language='en',
			country='ET',
			max_results=25
		)
		
		if conflict_articles:
			logger.info(f"Found {len(conflict_articles)} conflict-related articles")
			
			# Analyze sources
			sources = {}
			for article in conflict_articles:
				source = article.get('source_name', 'Unknown')
				sources[source] = sources.get(source, 0) + 1
			
			logger.info("📊 Articles by source:")
			for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
				logger.info(f"   {source}: {count} articles")
		
		# Get headlines from different countries
		logger.info("\n📱 Regional Headlines:")
		
		countries_to_monitor = ['KE', 'ET', 'SO', 'SD']
		for country_code in countries_to_monitor:
			try:
				headlines = await client.get_headlines(
					country=country_code,
					language='en'
				)
				
				country_names = {'KE': 'Kenya', 'ET': 'Ethiopia', 'SO': 'Somalia', 'SD': 'Sudan'}
				country_name = country_names.get(country_code, country_code)
				
				logger.info(f"🏛️  {country_name}: {len(headlines)} headlines")
				
				# Show top 3 headlines
				for i, headline in enumerate(headlines[:3]):
					logger.info(f"   {i+1}. {headline['title']}")
				
			except Exception as e:
				logger.error(f"Failed to get headlines for {country_code}: {e}")
		
		# Final health check
		final_health = await client.health_check()
		logger.info(f"\n🏥 Final health check: {final_health['status']}")
		
		# Close the client properly
		await client.close()
		logger.info("✅ Enhanced Google News Client closed successfully")
		
	except Exception as e:
		logger.error(f"❌ Example failed: {e}", exc_info=True)
		raise

async def advanced_monitoring_example():
	"""Advanced monitoring example with custom configuration."""
	try:
		from ..enhanced_client import (
			EnhancedGoogleNewsClient,
			EnhancedGoogleNewsConfig
		)
		
		logger.info("🔧 Starting Advanced Monitoring Example")
		
		# Create custom configuration
		config = EnhancedGoogleNewsConfig(
			database_url=os.getenv("DATABASE_URL", "postgresql:///lnd"),
			
			# Rate limiting - more conservative for production
			enable_rate_limiting=True,
			requests_per_second=2.0,
			burst_capacity=20,
			
			# Circuit breaker - more sensitive
			enable_circuit_breaker=True,
			failure_threshold=2,
			recovery_timeout=60.0,
			
			# Content filtering
			min_content_length=200,
			max_content_length=25000,
			allowed_languages=['en', 'fr', 'ar'],
			
			# Geographic focus - very specific
			target_countries=[
				'Kenya', 'Ethiopia', 'Somalia', 'Sudan', 'South Sudan',
				'Eritrea', 'Djibouti', 'Uganda'
			],
			
			# HTTP settings
			request_timeout=15.0,
			max_concurrent_requests=5
		)
		
		# Create client with custom config
		client = EnhancedGoogleNewsClient(config)
		await client.initialize()
		
		logger.info("✅ Advanced client initialized with custom configuration")
		
		# Continuous monitoring loop (simplified for example)
		monitoring_queries = [
			"Ethiopia Tigray conflict latest",
			"Somalia Al-Shabaab attack",
			"Sudan Darfur violence"
		]
		
		for i in range(3):  # Run 3 monitoring cycles
			logger.info(f"🔄 Monitoring cycle {i+1}/3")
			
			cycle_start = datetime.now(timezone.utc)
			
			for query in monitoring_queries:
				try:
					articles = await client.search_news(
						query=query,
						language='en',
						country='ET',
						max_results=20
					)
					
					logger.info(f"   '{query}': {len(articles)} articles")
					
				except Exception as e:
					logger.warning(f"   '{query}': Search failed - {e}")
			
			# Get current stats
			stats = await client.get_stats()
			logger.info(f"   Total searches: {stats['performance']['searches_performed']}")
			logger.info(f"   Total articles: {stats['performance']['articles_discovered']}")
			
			# Wait before next cycle (in real usage, this might be much longer)
			await asyncio.sleep(2)
		
		# Final statistics
		final_stats = await client.get_stats()
		logger.info("📊 Final Advanced Monitoring Statistics:")
		logger.info(f"   Runtime: {final_stats['uptime_seconds']:.1f} seconds")
		logger.info(f"   Articles per search: {final_stats['rates']['articles_per_search']:.1f}")
		logger.info(f"   Storage success rate: {final_stats['rates']['storage_success_rate']:.1f}%")
		
		await client.close()
		logger.info("✅ Advanced monitoring example completed")
		
	except Exception as e:
		logger.error(f"❌ Advanced example failed: {e}", exc_info=True)
		raise

if __name__ == "__main__":
	"""Run the examples."""
	
	print("""
Horn of Africa News Monitoring Examples
======================================

This example demonstrates the enhanced Google News crawler
for conflict monitoring in the Horn of Africa.

Prerequisites:
1. PostgreSQL database with information_units schema
2. Database connection configured in DATABASE_URL environment variable
3. Required Python packages installed

Running examples...
""")
	
	# Run basic example
	asyncio.run(main())
	
	print("\n" + "="*60 + "\n")
	
	# Run advanced example
	asyncio.run(advanced_monitoring_example())
	
	print("""

Examples completed! 

Key Features Demonstrated:
✅ Enhanced Google News client with production-ready features
✅ Database integration with information_units schema
✅ Token bucket rate limiting with adaptive behavior
✅ Circuit breaker pattern for resilience
✅ Comprehensive error handling and recovery
✅ Real-time monitoring and health checks
✅ Performance statistics and metrics
✅ Horn of Africa regional focus
✅ Conflict-specific keyword monitoring

For production deployment:
1. Configure proper database connection strings
2. Adjust rate limiting based on your usage patterns
3. Set up monitoring and alerting
4. Consider data retention and archival policies
""")