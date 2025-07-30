"""
Search Content Pipeline Usage Examples
=====================================

Comprehensive examples demonstrating the search-to-content pipeline that:
1. Discovers URLs via multi-engine search
2. Downloads full content via gen_crawler
3. Stores enriched results in information_units database

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import search content pipeline components
from .search_content_pipeline import (
	SearchContentPipeline,
	SearchContentPipelineConfig,
	create_search_content_pipeline,
	create_conflict_search_content_pipeline,
	create_horn_africa_search_pipeline
)

# Example 1: Basic Search Content Pipeline
async def basic_search_content_example():
	"""Demonstrate basic search content pipeline functionality."""
	logger.info("🔍 Basic Search Content Pipeline Example")
	
	# Create pipeline with default settings
	pipeline = create_search_content_pipeline(
		search_engines=['google', 'bing', 'duckduckgo'],
		database_url="postgresql:///lnd",
		enable_content_download=True,
		enable_database_storage=True
	)
	
	try:
		# Initialize pipeline
		await pipeline.initialize()
		
		# Search and download content for general topics
		queries = ["artificial intelligence news", "climate change reports"]
		
		enriched_results = await pipeline.search_and_download(
			queries=queries,
			languages=['en'],
			regions=['US', 'UK']
		)
		
		# Display results
		logger.info(f"📊 Retrieved {len(enriched_results)} enriched results")
		
		for result in enriched_results[:3]:  # Show first 3 results
			logger.info(f"🔗 URL: {result.search_result.url}")
			logger.info(f"🏷️  Title: {result.search_result.title}")
			logger.info(f"🔍 Engine: {result.search_engine}")
			logger.info(f"📝 Content Length: {result.word_count} words")
			logger.info(f"✅ Download Success: {result.download_success}")
			logger.info("---")
		
		# Get pipeline statistics
		stats = await pipeline.get_pipeline_stats()
		logger.info(f"📈 Pipeline Statistics: {stats}")
		
	finally:
		await pipeline.close()

# Example 2: Conflict Monitoring Pipeline
async def conflict_monitoring_example():
	"""Demonstrate conflict monitoring search pipeline."""
	logger.info("⚔️  Conflict Monitoring Pipeline Example")
	
	# Create conflict-optimized pipeline
	pipeline = create_conflict_search_content_pipeline("postgresql:///lnd")
	
	try:
		await pipeline.initialize()
		
		# Conflict-specific search queries
		conflict_queries = [
			"Ethiopia conflict latest news",
			"Somalia security situation",
			"Sudan political crisis",
			"Horn of Africa tensions",
			"African Union peacekeeping"
		]
		
		enriched_results = await pipeline.search_and_download(
			queries=conflict_queries,
			languages=['en', 'fr'],
			regions=['ET', 'SO', 'SD', 'KE']
		)
		
		# Analyze conflict content
		conflict_content = [
			result for result in enriched_results 
			if result.download_success and result.word_count > 200
		]
		
		logger.info(f"⚔️  Found {len(conflict_content)} substantial conflict articles")
		
		# Display high-quality conflict articles
		for article in conflict_content[:5]:
			logger.info(f"📰 Source: {article.domain}")
			logger.info(f"🔗 URL: {article.search_result.url}")
			logger.info(f"📊 Quality Score: {article.content_quality_score:.2f}")
			logger.info(f"🔢 Word Count: {article.word_count}")
			logger.info("---")
		
		stats = await pipeline.get_pipeline_stats()
		success_rate = stats['performance_metrics']['download_success_rate']
		logger.info(f"✅ Download Success Rate: {success_rate:.1f}%")
		
	finally:
		await pipeline.close()

# Example 3: Horn of Africa Regional Pipeline
async def horn_africa_regional_example():
	"""Demonstrate Horn of Africa regional monitoring."""
	logger.info("🌍 Horn of Africa Regional Pipeline Example")
	
	# Create regional pipeline with domain filtering
	pipeline = create_horn_africa_search_pipeline("postgresql:///lnd")
	
	try:
		await pipeline.initialize()
		
		# Regional queries focusing on Horn of Africa
		regional_queries = [
			"Kenya politics news",
			"Ethiopia economic development",
			"Somalia reconstruction",
			"Sudan peace process",
			"East Africa trade",
			"IGAD regional cooperation"
		]
		
		enriched_results = await pipeline.search_and_download(
			queries=regional_queries,
			languages=['en', 'sw', 'am'],  # English, Swahili, Amharic
			regions=['KE', 'ET', 'SO', 'SD', 'UG', 'TZ']
		)
		
		# Group results by domain/source
		sources = {}
		for result in enriched_results:
			if result.download_success:
				domain = result.domain
				if domain not in sources:
					sources[domain] = []
				sources[domain].append(result)
		
		logger.info(f"🌍 Found content from {len(sources)} regional sources")
		
		# Display top regional sources
		for domain, articles in sorted(sources.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
			avg_quality = sum(a.content_quality_score for a in articles) / len(articles)
			total_words = sum(a.word_count for a in articles)
			logger.info(f"📰 {domain}: {len(articles)} articles, avg quality: {avg_quality:.2f}, total words: {total_words}")
		
		stats = await pipeline.get_pipeline_stats()
		logger.info(f"🌍 Regional pipeline processed {stats['pipeline_stats']['urls_discovered']} URLs")
		
	finally:
		await pipeline.close()

# Example 4: Custom Configuration Pipeline
async def custom_configuration_example():
	"""Demonstrate custom pipeline configuration."""
	logger.info("⚙️  Custom Configuration Pipeline Example")
	
	# Create custom pipeline configuration
	custom_config = SearchContentPipelineConfig(
		search_engines=['google', 'bing', 'yandex', 'brave'],
		max_results_per_engine=15,
		total_max_results=60,
		download_full_content=True,
		enable_database_storage=True,
		
		# Custom batch processing
		batch_size=8,
		delay_between_batches=1.5,
		
		# Custom content filtering
		blocked_domains=['facebook.com', 'twitter.com', 'youtube.com'],
		min_content_length=150,
		max_content_length=50000,
		
		# Custom gen_crawler settings
		gen_crawler_concurrent_downloads=3,
		gen_crawler_timeout=25,
		
		database_url="postgresql:///lnd"
	)
	
	pipeline = SearchContentPipeline(custom_config)
	
	try:
		await pipeline.initialize()
		
		# Technology-focused queries
		tech_queries = [
			"machine learning breakthroughs 2025",
			"quantum computing news",
			"sustainable technology innovations"
		]
		
		enriched_results = await pipeline.search_and_download(
			queries=tech_queries,
			languages=['en'],
			regions=['US', 'EU', 'CN']
		)
		
		# Analyze content by search engine
		engine_performance = {}
		for result in enriched_results:
			engine = result.search_engine
			if engine not in engine_performance:
				engine_performance[engine] = {'total': 0, 'successful': 0, 'avg_quality': 0}
			
			engine_performance[engine]['total'] += 1
			if result.download_success:
				engine_performance[engine]['successful'] += 1
				engine_performance[engine]['avg_quality'] += result.content_quality_score
		
		# Calculate averages
		for engine, stats in engine_performance.items():
			if stats['successful'] > 0:
				stats['avg_quality'] /= stats['successful']
				stats['success_rate'] = (stats['successful'] / stats['total']) * 100
			
			logger.info(f"🔍 {engine}: {stats['successful']}/{stats['total']} successful ({stats['success_rate']:.1f}%), avg quality: {stats['avg_quality']:.2f}")
		
		pipeline_stats = await pipeline.get_pipeline_stats()
		logger.info(f"⚙️  Custom pipeline performance: {pipeline_stats['performance_metrics']}")
		
	finally:
		await pipeline.close()

# Example 5: Content Analysis and Quality Assessment
async def content_analysis_example():
	"""Demonstrate content quality analysis."""
	logger.info("📊 Content Analysis and Quality Assessment Example")
	
	pipeline = create_search_content_pipeline(
		search_engines=['google', 'bing', 'duckduckgo'],
		enable_content_download=True,
		enable_database_storage=True
	)
	
	try:
		await pipeline.initialize()
		
		# News and analysis queries
		analysis_queries = [
			"economic analysis global markets",
			"scientific research publications",
			"policy analysis reports"
		]
		
		enriched_results = await pipeline.search_and_download(
			queries=analysis_queries
		)
		
		# Quality analysis
		high_quality = [r for r in enriched_results if r.content_quality_score > 0.7 and r.download_success]
		medium_quality = [r for r in enriched_results if 0.4 <= r.content_quality_score <= 0.7 and r.download_success]
		low_quality = [r for r in enriched_results if r.content_quality_score < 0.4 and r.download_success]
		
		logger.info(f"📊 Quality Distribution:")
		logger.info(f"   High Quality (>0.7): {len(high_quality)} articles")
		logger.info(f"   Medium Quality (0.4-0.7): {len(medium_quality)} articles")
		logger.info(f"   Low Quality (<0.4): {len(low_quality)} articles")
		
		# Analyze high-quality content
		if high_quality:
			avg_word_count = sum(r.word_count for r in high_quality) / len(high_quality)
			top_domains = {}
			for result in high_quality:
				domain = result.domain
				top_domains[domain] = top_domains.get(domain, 0) + 1
			
			logger.info(f"📈 High-quality content average word count: {avg_word_count:.0f}")
			logger.info("🏆 Top domains for high-quality content:")
			for domain, count in sorted(top_domains.items(), key=lambda x: x[1], reverse=True)[:3]:
				logger.info(f"   {domain}: {count} articles")
		
		stats = await pipeline.get_pipeline_stats()
		logger.info(f"📊 Overall processing rate: {stats['performance_metrics']['processing_rate']:.2f} URLs/second")
		
	finally:
		await pipeline.close()

# Main execution function
async def run_all_examples():
	"""Run all search content pipeline examples."""
	logger.info("🚀 Starting Search Content Pipeline Examples")
	
	examples = [
		("Basic Search Content", basic_search_content_example),
		("Conflict Monitoring", conflict_monitoring_example),
		("Horn of Africa Regional", horn_africa_regional_example),
		("Custom Configuration", custom_configuration_example),
		("Content Analysis", content_analysis_example)
	]
	
	for name, example_func in examples:
		logger.info(f"\n{'='*60}")
		logger.info(f"🎯 Running: {name}")
		logger.info(f"{'='*60}")
		
		try:
			await example_func()
			logger.info(f"✅ {name} completed successfully")
		except Exception as e:
			logger.error(f"❌ {name} failed: {e}")
		
		logger.info(f"⏱️  Waiting before next example...\n")
		await asyncio.sleep(2)  # Brief pause between examples
	
	logger.info("🎉 All examples completed!")

if __name__ == "__main__":
	# Run examples
	asyncio.run(run_all_examples())