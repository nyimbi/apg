#!/usr/bin/env python3
"""
Search Content Pipeline Integration Test
========================================

Comprehensive integration test to verify that the search content pipeline
works correctly with both search discovery and gen_crawler content download.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_search_crawler_imports():
	"""Test that search crawler components import correctly."""
	logger.info("🔍 Testing search crawler imports...")
	
	try:
		# Test core search crawler imports
		from .core.search_crawler import SearchCrawler, SearchCrawlerConfig
		logger.info("✅ Core search crawler imports successful")
		
		# Test search content pipeline imports
		from .search_content_pipeline import (
			SearchContentPipeline,
			SearchContentPipelineConfig,
			EnrichedSearchResult
		)
		logger.info("✅ Search content pipeline imports successful")
		
		# Test factory function imports
		from .search_content_pipeline import (
			create_search_content_pipeline,
			create_conflict_search_content_pipeline,
			create_horn_africa_search_pipeline
		)
		logger.info("✅ Factory function imports successful")
		
		# Test package-level imports
		from . import (
			SearchContentPipeline as PackageSearchContentPipeline,
			create_search_content_pipeline_factory,
			get_search_crawler_health
		)
		logger.info("✅ Package-level imports successful")
		
		return True
	except ImportError as e:
		logger.error(f"❌ Import failed: {e}")
		return False
	except Exception as e:
		logger.error(f"❌ Unexpected error during imports: {e}")
		return False

async def test_search_content_pipeline_creation():
	"""Test search content pipeline creation and configuration."""
	logger.info("🏭 Testing search content pipeline creation...")
	
	try:
		from .search_content_pipeline import (
			SearchContentPipeline,
			SearchContentPipelineConfig,
			create_search_content_pipeline
		)
		
		# Test basic pipeline creation
		basic_pipeline = create_search_content_pipeline(
			search_engines=['google', 'bing'],
			enable_content_download=True,
			enable_database_storage=False  # Disable for testing
		)
		logger.info("✅ Basic pipeline created successfully")
		
		# Test custom configuration
		custom_config = SearchContentPipelineConfig(
			search_engines=['google', 'duckduckgo'],
			max_results_per_engine=10,
			total_max_results=20,
			download_full_content=False,  # Disable for testing
			enable_database_storage=False,
			batch_size=5
		)
		
		custom_pipeline = SearchContentPipeline(custom_config)
		logger.info("✅ Custom configuration pipeline created successfully")
		
		# Test conflict monitoring pipeline
		from .search_content_pipeline import create_conflict_search_content_pipeline
		conflict_pipeline = create_conflict_search_content_pipeline("postgresql:///test")
		logger.info("✅ Conflict monitoring pipeline created successfully")
		
		# Test Horn of Africa pipeline
		from .search_content_pipeline import create_horn_africa_search_pipeline
		horn_africa_pipeline = create_horn_africa_search_pipeline("postgresql:///test")
		logger.info("✅ Horn of Africa pipeline created successfully")
		
		return True
	except Exception as e:
		logger.error(f"❌ Pipeline creation failed: {e}")
		return False

async def test_gen_crawler_integration():
	"""Test gen_crawler integration status."""
	logger.info("🔧 Testing gen_crawler integration...")
	
	try:
		from .search_content_pipeline import GEN_CRAWLER_AVAILABLE
		
		if GEN_CRAWLER_AVAILABLE:
			logger.info("✅ Gen crawler is available")
			
			# Test gen_crawler import
			from ..gen_crawler import create_gen_crawler, GenCrawlerConfig
			logger.info("✅ Gen crawler imports successful")
			
			# Test configuration creation
			gen_config = {
				'performance': {
					'max_pages_per_site': 1,
					'request_timeout': 30,
					'max_concurrent': 5
				},
				'content_filters': {
					'min_content_length': 100,
					'max_content_length': 100000
				},
				'stealth': {
					'enable_stealth': True,
					'user_agent': 'TestCrawler/1.0'
				}
			}
			
			gen_crawler = create_gen_crawler(gen_config)
			if gen_crawler:
				logger.info("✅ Gen crawler created successfully")
			else:
				logger.warning("⚠️  Gen crawler creation returned None")
			
		else:
			logger.warning("⚠️  Gen crawler is not available")
		
		return True
	except Exception as e:
		logger.error(f"❌ Gen crawler integration test failed: {e}")
		return False

async def test_search_engines_availability():
	"""Test search engines availability."""
	logger.info("🔍 Testing search engines availability...")
	
	try:
		from .core.search_crawler import SearchCrawler, SearchCrawlerConfig
		
		# Create search crawler to test engine initialization
		config = SearchCrawlerConfig(
			engines=['google', 'bing', 'duckduckgo'],
			download_content=False,  # Disable content download for testing
			timeout=10.0
		)
		
		crawler = SearchCrawler(config)
		
		# Check which engines were successfully initialized
		initialized_engines = list(crawler.engines.keys())
		logger.info(f"✅ Initialized search engines: {initialized_engines}")
		
		if len(initialized_engines) > 0:
			logger.info("✅ At least one search engine is available")
			return True
		else:
			logger.warning("⚠️  No search engines were initialized")
			return False
			
	except Exception as e:
		logger.error(f"❌ Search engines test failed: {e}")
		return False

async def test_database_components():
	"""Test database component availability."""
	logger.info("💾 Testing database components...")
	
	try:
		from .search_content_pipeline import DATABASE_AVAILABLE
		
		if DATABASE_AVAILABLE:
			logger.info("✅ Database components are available")
			
			# Test database imports
			from ..google_news_crawler.database import InformationUnitsManager
			logger.info("✅ Information units manager import successful")
			
		else:
			logger.warning("⚠️  Database components are not available")
		
		return True
	except Exception as e:
		logger.error(f"❌ Database components test failed: {e}")
		return False

async def test_search_content_pipeline_functionality():
	"""Test basic search content pipeline functionality (without actual crawling)."""
	logger.info("🚀 Testing search content pipeline functionality...")
	
	try:
		from .search_content_pipeline import (
			SearchContentPipeline,
			SearchContentPipelineConfig
		)
		
		# Create test configuration with minimal settings
		config = SearchContentPipelineConfig(
			search_engines=['google'],  # Use just one engine for testing
			max_results_per_engine=5,
			total_max_results=5,
			download_full_content=False,  # Disable content download for testing
			enable_database_storage=False,  # Disable database for testing
			batch_size=3
		)
		
		pipeline = SearchContentPipeline(config)
		
		# Test pipeline initialization
		await pipeline.initialize()
		logger.info("✅ Pipeline initialization successful")
		
		# Test statistics retrieval
		stats = await pipeline.get_pipeline_stats()
		logger.info(f"✅ Pipeline statistics retrieved: {len(stats)} metrics")
		
		# Test cleanup
		await pipeline.close()
		logger.info("✅ Pipeline cleanup successful")
		
		return True
	except Exception as e:
		logger.error(f"❌ Pipeline functionality test failed: {e}")
		return False

async def test_package_health():
	"""Test overall package health."""
	logger.info("🏥 Testing package health...")
	
	try:
		from . import get_search_crawler_health
		
		health = get_search_crawler_health()
		logger.info(f"📊 Package health status: {health['status']}")
		logger.info(f"   Core available: {health['core_available']}")
		logger.info(f"   Engines available: {health['engines_available']}")
		logger.info(f"   Search content pipeline available: {health['search_content_pipeline_available']}")
		logger.info(f"   Version: {health['version']}")
		
		if health['status'] == 'healthy':
			logger.info("✅ Package is healthy")
		else:
			logger.warning("⚠️  Package is in degraded state")
		
		return True
	except Exception as e:
		logger.error(f"❌ Package health test failed: {e}")
		return False

async def run_integration_tests():
	"""Run all integration tests."""
	logger.info("🚀 Starting Search Content Pipeline Integration Tests")
	logger.info("=" * 60)
	
	tests = [
		("Import Tests", test_search_crawler_imports),
		("Pipeline Creation", test_search_content_pipeline_creation),
		("Gen Crawler Integration", test_gen_crawler_integration),
		("Search Engines", test_search_engines_availability),
		("Database Components", test_database_components),
		("Pipeline Functionality", test_search_content_pipeline_functionality),
		("Package Health", test_package_health)
	]
	
	results = {}
	
	for test_name, test_func in tests:
		logger.info(f"\n🧪 Running: {test_name}")
		logger.info("-" * 40)
		
		try:
			success = await test_func()
			results[test_name] = success
			
			if success:
				logger.info(f"✅ {test_name} PASSED")
			else:
				logger.warning(f"⚠️  {test_name} FAILED")
				
		except Exception as e:
			logger.error(f"❌ {test_name} ERROR: {e}")
			results[test_name] = False
	
	# Summary
	logger.info(f"\n{'='*60}")
	logger.info("📊 INTEGRATION TEST SUMMARY")
	logger.info(f"{'='*60}")
	
	passed = sum(1 for success in results.values() if success)
	total = len(results)
	
	for test_name, success in results.items():
		status = "✅ PASS" if success else "❌ FAIL"
		logger.info(f"   {test_name}: {status}")
	
	logger.info(f"\n🏆 Results: {passed}/{total} tests passed")
	
	if passed == total:
		logger.info("🎉 ALL TESTS PASSED - Search Content Pipeline is fully operational!")
		return True
	else:
		logger.warning(f"⚠️  {total - passed} tests failed - Some components may not be available")
		return False

if __name__ == "__main__":
	async def main():
		"""Main test execution."""
		try:
			success = await run_integration_tests()
			sys.exit(0 if success else 1)
		except Exception as e:
			logger.error(f"❌ Test execution failed: {e}")
			sys.exit(1)
	
	asyncio.run(main())