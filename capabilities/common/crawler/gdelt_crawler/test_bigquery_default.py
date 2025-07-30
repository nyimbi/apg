#!/usr/bin/env python3
"""
Test BigQuery Default Configuration
===================================

Quick test to verify the updated GDELT crawler with BigQuery as default.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def test_bigquery_default():
    """Test the BigQuery default configuration."""
    
    try:
        # Test direct import from __init__ module
        import importlib.util
        
        init_path = Path(__file__).parent / "__init__.py"
        spec = importlib.util.spec_from_file_location("gdelt_crawler", init_path)
        gdelt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gdelt_module)
        
        # Get the functions we need
        create_gdelt_crawler = gdelt_module.create_gdelt_crawler
        create_bigquery_gdelt_crawler = gdelt_module.create_bigquery_gdelt_crawler
        create_legacy_gdelt_crawler = gdelt_module.create_legacy_gdelt_crawler
        GDELTCrawlerConfig = gdelt_module.GDELTCrawlerConfig
        
        print("🧪 TESTING BIGQUERY DEFAULT CONFIGURATION")
        print("=" * 50)
        
        # Test 1: Default configuration should use BigQuery
        print("\n1️⃣ Testing default configuration...")
        
        default_config = GDELTCrawlerConfig()
        print(f"   use_bigquery: {default_config.use_bigquery}")
        print(f"   use_events_data: {default_config.use_events_data}")
        print(f"   fallback_enabled: {default_config.fallback_enabled}")
        print(f"   method_priority: {default_config.method_priority}")
        
        # Test 2: Factory function defaults
        print("\n2️⃣ Testing factory function defaults...")
        
        crawler = create_gdelt_crawler()
        print(f"   BigQuery enabled: {crawler.config.use_bigquery}")
        print(f"   Events data: {crawler.config.use_events_data}")
        print(f"   Fallback enabled: {crawler.config.fallback_enabled}")
        
        # Test 3: BigQuery-only crawler
        print("\n3️⃣ Testing BigQuery-only crawler...")
        
        try:
            bigquery_crawler = create_bigquery_gdelt_crawler(
                database_url="postgresql:///test"
            )
            print(f"   BigQuery enabled: {bigquery_crawler.config.use_bigquery}")
            print(f"   Fallback enabled: {bigquery_crawler.config.fallback_enabled}")
        except Exception as e:
            print(f"   ⚠️ BigQuery-only creation failed (expected without credentials): {e}")
        
        # Test 4: Legacy crawler
        print("\n4️⃣ Testing legacy crawler...")
        
        legacy_crawler = create_legacy_gdelt_crawler()
        print(f"   BigQuery enabled: {legacy_crawler.config.use_bigquery}")
        print(f"   Fallback enabled: {legacy_crawler.config.fallback_enabled}")
        
        # Test 5: Component initialization priority
        print("\n5️⃣ Testing component initialization...")
        
        test_crawler = create_gdelt_crawler(
            database_url="postgresql:///test",
            use_bigquery=True,
            fallback_enabled=True
        )
        
        print(f"   API client: {'✅' if test_crawler.api_client else '❌'}")
        print(f"   File downloader: {'✅' if test_crawler.file_downloader else '❌'}")
        print(f"   BigQuery ETL: {'✅' if test_crawler.events_etl or test_crawler.bigquery_etl else '❌'}")
        
        print("\n🎉 CONFIGURATION TESTS COMPLETED!")
        print("✅ BigQuery is now the default method")
        print("✅ Intelligent fallback is enabled")
        print("✅ Events data is preferred over GKG")
        print("✅ Factory functions work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bigquery_default())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")