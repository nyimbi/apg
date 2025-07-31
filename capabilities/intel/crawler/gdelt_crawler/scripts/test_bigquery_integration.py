#!/usr/bin/env python3
"""
Test BigQuery Integration with GDELT Crawler
===========================================

This script demonstrates how to use the updated GDELT crawler with BigQuery integration
to populate the information_units table directly from GDELT BigQuery datasets.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: December 27, 2024
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

async def test_bigquery_integration():
    """Test the BigQuery integration with GDELT crawler."""
    
    try:
        # Set up authentication
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/nyimbiodero/.config/gcloud/application_default_credentials.json'
        
        from crawlers.gdelt_crawler import create_gdelt_crawler, GDELTCrawlerConfig
        
        print("🚀 TESTING BIGQUERY GDELT CRAWLER INTEGRATION")
        print("=" * 60)
        
        # Create configuration with BigQuery enabled
        config = GDELTCrawlerConfig(
            database_url="postgresql:///lnd",
            use_bigquery=True,
            bigquery_project="gdelt-bq",
            google_credentials_path="/Users/nyimbiodero/.config/gcloud/application_default_credentials.json",
            target_countries=["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"],
            ml_scoring_enabled=True,
            enable_monitoring=True
        )
        
        # Create crawler
        crawler = create_gdelt_crawler(
            database_url=config.database_url,
            use_bigquery=True,
            bigquery_project="gdelt-bq",
            google_credentials_path="/Users/nyimbiodero/.config/gcloud/application_default_credentials.json",
            target_countries=["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"]
        )
        
        print("✅ GDELT crawler with BigQuery support created")
        
        # Initialize crawler
        await crawler.start()
        print("✅ Crawler initialized")
        
        # Perform health check
        health = await crawler.health_check()
        print(f"\n📊 HEALTH CHECK:")
        print(f"Overall Status: {health['overall_status']}")
        for component, status in health['components'].items():
            print(f"  {component}: {status['status']} ({status.get('type', 'N/A')})")
        
        # Test BigQuery data processing for a small date range
        test_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)  # Process just one day for testing
        
        print(f"\n🔄 TESTING BIGQUERY ETL FOR {test_date.strftime('%Y-%m-%d')}")
        
        # Process data using BigQuery
        result = await crawler.process_bigquery_data(
            start_date=test_date,
            end_date=end_date
        )
        
        print(f"\n📈 BIGQUERY ETL RESULTS:")
        if result['success']:
            print(f"  Records Processed: {result['total_records_processed']:,}")
            print(f"  Records Inserted: {result['records_inserted']:,}")
            print(f"  Records Updated: {result['records_updated']:,}")
            print(f"  Records Skipped: {result['records_skipped']:,}")
            print(f"  Processing Time: {result['processing_time_seconds']:.1f} seconds")
            print(f"  Processing Rate: {result['records_per_second']:.1f} records/second")
            print(f"  BigQuery Queries: {result['bigquery_queries_executed']}")
            print(f"  BigQuery Bytes: {result['bigquery_bytes_processed']:,}")
        else:
            print(f"  ❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Test daily ETL (which should use BigQuery automatically)
        print(f"\n🔄 TESTING DAILY ETL (BigQuery Mode)")
        daily_result = await crawler.run_daily_etl(date=test_date)
        
        print(f"\n📈 DAILY ETL RESULTS:")
        print(f"  Success: {daily_result['success']}")
        if daily_result['success']:
            if 'bigquery_processing' in daily_result:
                bq_result = daily_result['bigquery_processing']
                print(f"  BigQuery Records: {bq_result['total_records_processed']:,}")
                print(f"  Processing Time: {bq_result['processing_time_seconds']:.1f} seconds")
            if 'processed_counts' in daily_result:
                print(f"  Processed Counts: {daily_result['processed_counts']}")
        else:
            print(f"  Errors: {daily_result.get('errors', [])}")
        
        print(f"\n🎉 BIGQUERY INTEGRATION TEST COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Stop crawler
        if 'crawler' in locals():
            await crawler.stop()
            print("✅ Crawler stopped")

if __name__ == "__main__":
    success = asyncio.run(test_bigquery_integration())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")