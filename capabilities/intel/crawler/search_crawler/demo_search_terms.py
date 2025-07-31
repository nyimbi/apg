#!/usr/bin/env python3
"""
Search Terms Demo - Exactly What You Requested
==============================================

This demonstrates the search_crawler accepting a list of terms, searching across 
multiple search engines, then using gen_crawler to process the URLs and insert
into information_units database.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the search content pipeline
from .search_content_pipeline import create_search_content_pipeline

async def demo_search_terms_to_information_units():
	"""
	Demonstrate exactly what you requested:
	1. Accept list of search terms
	2. Search across multiple search engines  
	3. Gen_crawler iterates through URLs
	4. Results inserted into information_units
	"""
	
	logger.info("🎯 Demo: Search Terms → Multiple Engines → Gen Crawler → Information Units")
	logger.info("=" * 80)
	
	# Step 1: Define search terms list (expanded with new categories)
	search_terms = [
		# Conflict terms
		"Ethiopia conflict news", "Somalia security updates", "Sudan political situation",
		"Horn of Africa tensions", "Kenya border disputes",
		
		# Civil unrest and protest terms
		"Kenya protest news", "Ethiopia riots update", "Somalia demonstration",
		"Uganda civil unrest", "Horn of Africa protests", "East Africa strikes",
		
		# Diplomatic and peace terms
		"Ethiopia peace talks", "Somalia agreement news", "IGAD summit",
		"African Union meeting", "Horn of Africa treaty", "East Africa cooperation",
		"Kenya diplomatic meeting", "Sudan negotiations"
	]
	
	logger.info(f"📝 Search terms to process: {len(search_terms)}")
	for i, term in enumerate(search_terms, 1):
		logger.info(f"   {i}. '{term}'")
	
	# Step 2: Create search content pipeline with multiple engines
	pipeline = create_search_content_pipeline(
		search_engines=['google', 'bing', 'duckduckgo', 'yandex'],  # Multiple engines
		database_url="postgresql:///lnd",  # Information units database
		enable_content_download=True,      # Gen crawler enabled
		enable_database_storage=True,      # Insert to information_units
		max_results_per_engine=10,
		total_max_results=40
	)
	
	logger.info(f"🔍 Search engines configured: {pipeline.config.search_engines}")
	logger.info(f"💾 Database storage: {pipeline.config.enable_database_storage}")
	logger.info(f"⬇️  Content download (gen_crawler): {pipeline.config.download_full_content}")
	
	try:
		# Step 3: Initialize pipeline
		await pipeline.initialize()
		logger.info("✅ Pipeline initialized")
		
		# Step 4: Execute the complete workflow
		logger.info("\n🚀 Starting Complete Workflow:")
		logger.info("   Stage 1: Multi-engine search for URL discovery")
		logger.info("   Stage 2: Gen crawler downloads content from URLs")
		logger.info("   Stage 3: Insert enriched data into information_units")
		
		enriched_results = await pipeline.search_and_download(
			queries=search_terms,  # Your list of terms
			languages=['en'],
			regions=['ET', 'SO', 'SD', 'KE']  # Horn of Africa focus
		)
		
		# Step 5: Show results
		logger.info(f"\n📊 WORKFLOW COMPLETE")
		logger.info(f"   Total enriched results: {len(enriched_results)}")
		
		# Group by search engine to show multi-engine coverage
		engines_used = {}
		successful_downloads = 0
		total_words = 0
		
		for result in enriched_results:
			engine = result.search_engine
			engines_used[engine] = engines_used.get(engine, 0) + 1
			
			if result.download_success:
				successful_downloads += 1
				total_words += result.word_count
		
		logger.info(f"\n🔍 Multi-Engine Coverage:")
		for engine, count in engines_used.items():
			logger.info(f"   {engine}: {count} URLs discovered")
		
		logger.info(f"\n⬇️  Gen Crawler Results:")
		logger.info(f"   Successful downloads: {successful_downloads}/{len(enriched_results)}")
		logger.info(f"   Total content downloaded: {total_words:,} words")
		
		# Show sample results
		logger.info(f"\n📰 Sample Results (showing first 3):")
		for i, result in enumerate(enriched_results[:3], 1):
			logger.info(f"   {i}. URL: {result.search_result.url}")
			logger.info(f"      Engine: {result.search_engine}")
			logger.info(f"      Query: '{result.search_query}'")
			logger.info(f"      Downloaded: {result.download_success}")
			if result.download_success:
				logger.info(f"      Content: {result.word_count} words")
				logger.info(f"      Quality: {result.content_quality_score:.2f}")
			logger.info("      ---")
		
		# Step 6: Show database insertion stats
		stats = await pipeline.get_pipeline_stats()
		
		logger.info(f"\n💾 Information Units Database:")
		logger.info(f"   Records created: {stats['pipeline_stats']['database_records_created']}")
		logger.info(f"   Success rate: {stats['performance_metrics']['storage_success_rate']:.1f}%")
		
		logger.info(f"\n🎉 SUCCESS: Search terms processed through complete pipeline!")
		logger.info(f"   ✅ {len(search_terms)} search terms processed")
		logger.info(f"   ✅ {len(engines_used)} search engines used")
		logger.info(f"   ✅ {len(enriched_results)} URLs discovered")
		logger.info(f"   ✅ {successful_downloads} content downloads via gen_crawler")
		logger.info(f"   ✅ {stats['pipeline_stats']['database_records_created']} records in information_units")
		
	finally:
		await pipeline.close()
		logger.info("🔌 Pipeline closed")

# Quick demo function that matches your exact requirements
async def your_exact_requirements_demo():
	"""
	Demonstration matching your exact requirements:
	- search_crawler accepts list of terms
	- searches across multiple search engines  
	- list of URLs iterated by gen_crawler
	- inserted into information_units
	"""
	
	# Your list of terms (enhanced with new categories)
	terms_to_search = [
		# Traditional conflict monitoring
		"Ethiopia conflict update", "Somalia Al-Shabaab news",
		
		# Civil unrest monitoring
		"Kenya election protests", "Uganda opposition demonstration",
		"Horn of Africa civil unrest", "East Africa student protests",
		
		# Diplomatic monitoring
		"IGAD peace summit", "African Union mediation",
		"Ethiopia Eritrea agreement", "Horn of Africa cooperation treaty"
	]
	
	# Create the pipeline (this combines search_crawler + gen_crawler + database)
	pipeline = create_search_content_pipeline(
		search_engines=['google', 'bing', 'duckduckgo'],
		enable_content_download=True,  # gen_crawler processes URLs
		enable_database_storage=True   # insert into information_units
	)
	
	try:
		await pipeline.initialize()
		
		# This does everything you requested in one call:
		# 1. Accepts list of terms ✅
		# 2. Searches multiple engines ✅  
		# 3. Gen_crawler iterates URLs ✅
		# 4. Inserts to information_units ✅
		results = await pipeline.search_and_download(queries=terms_to_search)
		
		print(f"✅ Processed {len(terms_to_search)} search terms")
		print(f"✅ Found {len(results)} URLs across multiple engines")
		print(f"✅ Gen_crawler processed {sum(1 for r in results if r.download_success)} URLs")
		print(f"✅ Inserted into information_units database")
		
		return results
		
	finally:
		await pipeline.close()

if __name__ == "__main__":
	print("🎯 Running search terms to information_units demo...")
	asyncio.run(demo_search_terms_to_information_units())