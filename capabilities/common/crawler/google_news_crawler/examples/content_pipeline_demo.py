#!/usr/bin/env python3
"""
Google News Content Pipeline Demonstration
==========================================

Complete demonstration of the two-stage content pipeline:

1. **Discovery Stage**: Google News crawler discovers relevant articles
2. **Download Stage**: Gen crawler downloads full article content

This creates a comprehensive news monitoring system that not only discovers
relevant articles but also retrieves their complete content for analysis.

Features Demonstrated:
- Google News discovery with enhanced features
- Full content download via gen_crawler  
- Content enrichment and analysis
- Performance monitoring and statistics
- Error handling and resilience

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
	"""Main demonstration of the content pipeline."""
	try:
		# Import the content pipeline
		from ..content_pipeline import (
			create_horn_africa_content_pipeline,
			create_content_pipeline,
			ContentPipelineConfig
		)
		
		logger.info("🚀 Starting Google News Content Pipeline Demonstration")
		logger.info("=" * 70)
		
		# Configure database connection
		database_url = os.getenv("DATABASE_URL", "postgresql:///lnd")
		
		# Create Horn of Africa optimized content pipeline
		pipeline = create_horn_africa_content_pipeline(database_url)
		
		# Initialize the pipeline
		await pipeline.initialize()
		
		logger.info("✅ Content pipeline initialized successfully")
		
		# Perform health check
		if hasattr(pipeline.google_news_client, 'health_check'):
			health = await pipeline.google_news_client.health_check()
			logger.info(f"🏥 Google News client health: {health['status']}")
		
		# Define conflict monitoring queries for Horn of Africa
		monitoring_queries = [
			"Ethiopia Tigray conflict latest news",
			"Somalia Al-Shabaab security update",
			"Sudan Darfur violence recent",
			"South Sudan conflict 2024",
			"Horn of Africa drought crisis"
		]
		
		logger.info(f"🔍 Monitoring {len(monitoring_queries)} conflict-related topics")
		for i, query in enumerate(monitoring_queries, 1):
			logger.info(f"   {i}. {query}")
		
		# Execute the complete pipeline: discovery + download
		logger.info("\n" + "=" * 70)
		logger.info("🎯 STAGE 1: ARTICLE DISCOVERY VIA GOOGLE NEWS")
		logger.info("=" * 70)
		
		enriched_articles = await pipeline.discover_and_download(
			queries=monitoring_queries,
			language='en',
			country='ET'  # Ethiopia as central country for Horn of Africa
		)
		
		logger.info("\n" + "=" * 70)
		logger.info("📊 PIPELINE RESULTS ANALYSIS")
		logger.info("=" * 70)
		
		# Analyze results
		total_articles = len(enriched_articles)
		successful_downloads = sum(1 for article in enriched_articles if article.download_success)
		failed_downloads = total_articles - successful_downloads
		
		logger.info(f"📰 Total articles discovered and processed: {total_articles}")
		logger.info(f"✅ Successful content downloads: {successful_downloads}")
		logger.info(f"❌ Failed content downloads: {failed_downloads}")
		
		if successful_downloads > 0:
			success_rate = (successful_downloads / total_articles) * 100
			logger.info(f"📈 Download success rate: {success_rate:.1f}%")
		
		# Show sample results
		logger.info("\n📖 SAMPLE ENRICHED ARTICLES:")
		
		for i, article in enumerate(enriched_articles[:3], 1):  # Show first 3
			logger.info(f"\n🔸 Article {i}:")
			logger.info(f"   Title: {article.google_news_record.title}")
			logger.info(f"   Source: {article.google_news_record.source_name}")
			logger.info(f"   URL: {article.google_news_record.content_url}")
			logger.info(f"   Original Summary: {article.google_news_record.summary[:100]}...")
			
			if article.download_success:
				logger.info(f"   ✅ Full Content Downloaded:")
				logger.info(f"      Full Title: {article.full_title}")
				logger.info(f"      Word Count: {article.word_count}")
				logger.info(f"      Quality Score: {article.content_quality_score:.2f}")
				logger.info(f"      Download Time: {article.download_time:.2f}s")
				if article.full_content:
					content_preview = article.full_content[:200].replace('\n', ' ')
					logger.info(f"      Content Preview: {content_preview}...")
			else:
				logger.info(f"   ❌ Download Failed: {article.download_error}")
		
		# Content analysis
		if successful_downloads > 0:
			logger.info("\n📊 CONTENT ANALYSIS:")
			
			# Calculate statistics
			total_words = sum(article.word_count for article in enriched_articles if article.download_success)
			avg_words = total_words / successful_downloads
			avg_quality = sum(article.content_quality_score for article in enriched_articles if article.download_success) / successful_downloads
			avg_download_time = sum(article.download_time for article in enriched_articles if article.download_success) / successful_downloads
			
			logger.info(f"   📝 Total words downloaded: {total_words:,}")
			logger.info(f"   📏 Average article length: {avg_words:.0f} words")
			logger.info(f"   ⭐ Average content quality: {avg_quality:.2f}/1.0")
			logger.info(f"   ⏱️  Average download time: {avg_download_time:.2f}s")
			
			# Source analysis
			sources = {}
			for article in enriched_articles:
				source = article.google_news_record.source_name or "Unknown"
				sources[source] = sources.get(source, 0) + 1
			
			logger.info(f"\n📡 CONTENT SOURCES:")
			for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
				logger.info(f"   {source}: {count} articles")
		
		# Get comprehensive pipeline statistics
		stats = await pipeline.get_pipeline_stats()
		
		logger.info("\n" + "=" * 70)
		logger.info("📈 DETAILED PIPELINE STATISTICS")
		logger.info("=" * 70)
		
		pipeline_stats = stats['pipeline_stats']
		logger.info(f"🔍 Discovery queries executed: {pipeline_stats['discovery_queries']}")
		logger.info(f"📰 Articles discovered: {pipeline_stats['articles_discovered']}")
		logger.info(f"⬇️  Articles downloaded: {pipeline_stats['articles_downloaded']}")
		logger.info(f"❌ Download failures: {pipeline_stats['download_failures']}")
		logger.info(f"🔄 Duplicate URLs skipped: {pipeline_stats['duplicate_urls_skipped']}")
		logger.info(f"⏱️  Total runtime: {stats['runtime_seconds']:.1f} seconds")
		
		# Performance metrics
		perf_metrics = stats['performance_metrics']
		logger.info(f"\n🎯 PERFORMANCE METRICS:")
		logger.info(f"   Articles per query: {perf_metrics['articles_per_query']:.1f}")
		logger.info(f"   Download success rate: {perf_metrics['download_success_rate']:.1f}%")
		logger.info(f"   Processing rate: {perf_metrics['processing_rate']:.1f} articles/second")
		
		# Google News client statistics
		if 'google_news_stats' in stats and stats['google_news_stats']:
			gn_stats = stats['google_news_stats']
			logger.info(f"\n🔍 GOOGLE NEWS CLIENT STATS:")
			logger.info(f"   Searches performed: {gn_stats.get('performance', {}).get('searches_performed', 'N/A')}")
			logger.info(f"   Articles stored: {gn_stats.get('performance', {}).get('articles_stored', 'N/A')}")
			logger.info(f"   Errors handled: {gn_stats.get('performance', {}).get('errors_handled', 'N/A')}")
		
		# Export results to JSON for further analysis
		output_file = f"content_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
		
		export_data = {
			'pipeline_run': {
				'timestamp': datetime.now(timezone.utc).isoformat(),
				'queries': monitoring_queries,
				'total_articles': total_articles,
				'successful_downloads': successful_downloads
			},
			'statistics': stats,
			'articles': [
				{
					'title': article.google_news_record.title,
					'url': article.google_news_record.content_url,
					'source': article.google_news_record.source_name,
					'download_success': article.download_success,
					'word_count': article.word_count,
					'quality_score': article.content_quality_score,
					'download_time': article.download_time
				}
				for article in enriched_articles
			]
		}
		
		with open(output_file, 'w') as f:
			json.dump(export_data, f, indent=2)
		
		logger.info(f"\n💾 Results exported to: {output_file}")
		
		# Close the pipeline
		await pipeline.close()
		logger.info("\n✅ Content pipeline demonstration completed successfully!")
		
	except Exception as e:
		logger.error(f"❌ Pipeline demonstration failed: {e}", exc_info=True)
		raise

async def demonstration_with_custom_config():
	"""Demonstrate pipeline with custom configuration."""
	try:
		from ..content_pipeline import (
			GoogleNewsContentPipeline,
			ContentPipelineConfig,
			EnhancedGoogleNewsConfig
		)
		
		logger.info("\n" + "=" * 70)
		logger.info("🔧 CUSTOM CONFIGURATION DEMONSTRATION")
		logger.info("=" * 70)
		
		# Create custom configuration for high-volume monitoring
		google_news_config = EnhancedGoogleNewsConfig(
			database_url=os.getenv("DATABASE_URL", "postgresql:///lnd"),
			enable_database_storage=False,  # Disable for demo
			enable_rate_limiting=True,
			enable_circuit_breaker=True,
			requests_per_second=2.0,  # Conservative rate
			burst_capacity=20,
			max_concurrent_requests=3
		)
		
		pipeline_config = ContentPipelineConfig(
			google_news_config=google_news_config,
			max_articles_per_query=10,  # Limit for demo
			download_full_content=True,
			batch_size=3,
			delay_between_batches=1.0,
			enable_content_analysis=True
		)
		
		# Create pipeline with custom config
		pipeline = GoogleNewsContentPipeline(pipeline_config)
		await pipeline.initialize()
		
		# Test with focused queries
		focused_queries = [
			"Ethiopia latest news today",
			"Kenya security update"
		]
		
		logger.info(f"🎯 Testing with focused queries: {focused_queries}")
		
		enriched_articles = await pipeline.discover_and_download(
			queries=focused_queries,
			language='en',
			country='KE'
		)
		
		logger.info(f"📊 Custom config results: {len(enriched_articles)} articles processed")
		
		# Show configuration effectiveness
		stats = await pipeline.get_pipeline_stats()
		logger.info(f"⚡ Processing rate with custom config: {stats['performance_metrics']['processing_rate']:.1f} articles/sec")
		
		await pipeline.close()
		logger.info("✅ Custom configuration demonstration complete")
		
	except Exception as e:
		logger.error(f"❌ Custom config demonstration failed: {e}")

if __name__ == "__main__":
	"""Run the complete demonstration."""
	
	print("""
Google News Content Pipeline Demonstration
==========================================

This demonstration shows the complete two-stage content pipeline:

STAGE 1: DISCOVERY
- Google News crawler discovers relevant articles
- Extracts metadata, URLs, summaries
- Applies filters and deduplication

STAGE 2: DOWNLOAD  
- Gen crawler downloads full article content
- Extracts complete text, metadata
- Performs content analysis and quality scoring

INTEGRATION BENEFITS:
✅ Complete news monitoring (discovery + content)
✅ Respectful crawling with rate limiting
✅ Comprehensive error handling  
✅ Performance monitoring and statistics
✅ Database integration for storage
✅ Content analysis and quality scoring

Prerequisites:
1. PostgreSQL database (optional, can be disabled)
2. Gen crawler components (automatic fallback if unavailable)
3. Google News access (via RSS feeds)

Starting demonstration...
""")
	
	# Run main demonstration
	asyncio.run(main())
	
	print("\n" + "="*60 + "\n")
	
	# Run custom configuration demonstration
	asyncio.run(demonstration_with_custom_config())
	
	print("""

✅ Demonstration Complete!

KEY FEATURES DEMONSTRATED:
🔍 Two-stage pipeline (discovery + download)
📰 Google News article discovery
⬇️  Full content download via gen_crawler
📊 Content analysis and quality scoring
📈 Performance monitoring and statistics
🛡️  Error handling and resilience
⚙️  Custom configuration options
💾 Data export capabilities

PRODUCTION DEPLOYMENT:
1. Configure database connections
2. Set up monitoring and alerting
3. Adjust rate limiting for your use case
4. Implement data retention policies
5. Scale with multiple pipeline instances

The content pipeline provides a complete solution for comprehensive
news monitoring with both discovery and full content retrieval.
""")